"""
RNN (Recurrent Neural Network) — built from scratch in PyTorch
==============================================================
A vanilla RNN is the simplest form of recurrent network.
At each timestep it reads one input and updates a hidden state —
think of the hidden state as the model's "working memory".

The catch: vanilla RNNs struggle with long sequences because
gradients shrink (or explode) as they travel back through time.
This is called the vanishing gradient problem — and it's exactly
why LSTMs were invented. But understanding RNNs first makes
everything else click.

Usage:
    # Train on your own CSV:
    python rnn.py --data your_data.csv --target column_name

    # Run demo with synthetic data:
    python rnn.py

Author: Niloofar Tavahoodi
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── 1. Dataset ───────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Turns a flat time-series into overlapping input/output pairs.

    Example with seq_len=3 and data [a, b, c, d, e]:
        sample 0 → input [a, b, c] → predict d
        sample 1 → input [b, c, d] → predict e

    This is the standard way to frame time-series as a supervised problem.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        self.features = torch.FloatTensor(features)
        self.targets  = torch.FloatTensor(targets)
        self.seq_len  = seq_len

    def __len__(self):
        # each sample needs seq_len rows + 1 target row
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]        # (seq_len, n_features)
        y = self.targets[idx + self.seq_len].unsqueeze(0)  # (1,)
        return x, y


# ── 2. RNN Cell (single timestep) ────────────────────────────────────────────

class RNNCell(nn.Module):
    """
    The core of a vanilla RNN — processes one timestep at a time.

    At each step it takes:
        x_t    — the current input (e.g. today's sensor readings)
        h_prev — the hidden state from the previous step (memory so far)

    And produces:
        h_t — a new hidden state that blends old memory with new input

    The formula:  h_t = tanh( W · [x_t, h_prev] + b )

    tanh squishes the output to [-1, 1], which keeps values stable
    across many timesteps (unlike if we just used a linear layer).
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # one weight matrix that operates on [input, hidden] concatenated
        # this is equivalent to W_x · x_t + W_h · h_prev, but done in one step
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t    : current input     (batch_size, input_size)
            h_prev : previous memory   (batch_size, hidden_size)

        Returns:
            h_t    : updated memory    (batch_size, hidden_size)
        """
        # concatenate input and previous hidden state side by side
        combined = torch.cat([x_t, h_prev], dim=1)

        # apply linear transform + tanh — this is the entire RNN update rule
        h_t = torch.tanh(self.W_h(combined))

        return h_t


# ── 3. Full Sequence Model ────────────────────────────────────────────────────

class RNNSequence(nn.Module):
    """
    Feeds an entire sequence through the RNN cell, one step at a time.
    At the end, passes the final hidden state through a linear layer
    to produce a prediction.

    Why use only the final hidden state?
    Because by the last timestep, the model has "seen" the whole
    sequence — its hidden state is a compressed summary of everything.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell    = RNNCell(input_size, hidden_size)

        # maps the final hidden state to a prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_sequence : (batch_size, seq_length, input_size)

        Returns:
            output : (batch_size, output_size)
        """
        batch_size = x_sequence.shape[0]
        seq_length = x_sequence.shape[1]

        # start with a blank slate — no memory at the beginning
        h_t = torch.zeros(batch_size, self.hidden_size)

        # step through the sequence one timestep at a time
        for t in range(seq_length):
            x_t = x_sequence[:, t, :]       # grab current timestep: (batch, input_size)
            h_t = self.rnn_cell(x_t, h_t)   # update memory

        # use the final memory state to make a prediction
        return self.fc(h_t)


# ── 4. Data Loading ───────────────────────────────────────────────────────────

def load_csv(path: str, target_col: str, seq_len: int, batch_size: int):
    """
    Loads a CSV, normalizes features, splits into train/val/test,
    and wraps everything in DataLoaders ready for training.
    """
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # normalize so all features are on the same scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split chronologically — no shuffling for time-series!
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, shuffle=False)
    X_val,  X_test, y_val,  y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, shuffle=False)

    train_loader = DataLoader(SequenceDataset(X_train, y_train, seq_len), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X_val,   y_val,   seq_len), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SequenceDataset(X_test,  y_test,  seq_len), batch_size=batch_size, shuffle=False)

    print(f"  Loaded   : {path}")
    print(f"  Features : {feature_cols}")
    print(f"  Target   : {target_col}")
    print(f"  Split    : {len(X_train)} train / {len(X_val)} val / {len(X_test)} test\n")

    return train_loader, val_loader, test_loader, len(feature_cols)


def make_synthetic(seq_len: int, batch_size: int, n_features: int = 10):
    """Generates a simple sine-wave dataset for demo purposes."""
    print("  No CSV provided — running demo with synthetic sine-wave data.")
    print("  To use your own data: python rnn.py --data file.csv --target column\n")

    t = np.linspace(0, 8 * np.pi, 600)
    # features: noisy sine waves at different frequencies
    X = np.column_stack([np.sin((i + 1) * t) + np.random.normal(0, 0.1, len(t))
                         for i in range(n_features)]).astype(np.float32)
    y = np.sin(t + 0.3).astype(np.float32)   # target: phase-shifted sine

    split1, split2 = 420, 510
    train_loader = DataLoader(SequenceDataset(X[:split1],       y[:split1],       seq_len), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X[split1:split2], y[split1:split2], seq_len), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SequenceDataset(X[split2:],       y[split2:],       seq_len), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_features


# ── 5. Train / Evaluate ───────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, train: bool) -> float:
    """Runs one full pass over the dataset (training or evaluation)."""
    model.train() if train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for x_batch, y_batch in loader:
            if train:
                optimizer.zero_grad()

            output = model(x_batch)
            loss   = criterion(output, y_batch)

            if train:
                loss.backward()   # compute gradients
                optimizer.step()  # update weights

            total_loss += loss.item()

    return total_loss / len(loader)


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a vanilla RNN on time-series data")
    parser.add_argument("--data",        type=str,   default=None,  help="Path to CSV file")
    parser.add_argument("--target",      type=str,   default=None,  help="Target column name")
    parser.add_argument("--seq_len",     type=int,   default=10,    help="Input sequence length (default: 10)")
    parser.add_argument("--hidden_size", type=int,   default=32,    help="RNN hidden size (default: 32)")
    parser.add_argument("--epochs",      type=int,   default=50,    help="Training epochs (default: 50)")
    parser.add_argument("--batch_size",  type=int,   default=32,    help="Batch size (default: 32)")
    parser.add_argument("--lr",          type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    print("=" * 50)
    print("  Vanilla RNN — from scratch in PyTorch")
    print("=" * 50)

    # load data
    if args.data:
        if not args.target:
            raise ValueError("Please also specify --target when using --data")
        train_loader, val_loader, test_loader, n_features = load_csv(
            args.data, args.target, args.seq_len, args.batch_size
        )
    else:
        train_loader, val_loader, test_loader, n_features = make_synthetic(
            args.seq_len, args.batch_size
        )

    # build model
    model     = RNNSequence(n_features, args.hidden_size, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # learning rate scheduler — halves LR every 10 epochs to fine-tune later
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"  Model     : RNNSequence(input={n_features}, hidden={args.hidden_size})")
    print(f"  Optimizer : Adam (lr={args.lr}, weight_decay=1e-4)")
    print(f"  Scheduler : StepLR (step=10, gamma=0.5)")
    print(f"  Epochs    : {args.epochs}\n")

    # training loop — save the best model based on validation loss
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, criterion, train=False)
        scheduler.step()

        # save checkpoint whenever validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_rnn.pt")

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # final evaluation on held-out test set
    model.load_state_dict(torch.load("best_rnn.pt"))
    test_loss = run_epoch(model, test_loader, None, criterion, train=False)

    print(f"\n  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Model saved   : best_rnn.pt")


if __name__ == "__main__":
    main()
