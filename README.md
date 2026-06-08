# Vanilla RNN from Scratch — PyTorch

A clean implementation of a **Recurrent Neural Network (RNN)** built manually in PyTorch — no `nn.RNN` used — to show exactly how sequential memory works at each timestep.

Understanding the vanilla RNN is the foundation for everything that comes after: LSTMs, GRUs, and Transformers all build on this core idea.

## The Core Idea

At every timestep, the RNN does one simple thing:

```
h_t = tanh( W · [x_t, h_prev] + b )
```

- `x_t` is the current input (e.g. today's reading)
- `h_prev` is the memory from the previous step
- `h_t` is the updated memory — a blend of old and new

After processing the full sequence, the final `h_t` is a compressed summary used to make a prediction.

## Why Vanilla RNN Falls Short

Vanilla RNNs struggle with **long sequences** — gradients shrink exponentially as they travel backwards through many timesteps (vanishing gradient problem). This is exactly why LSTMs were invented. See the [lstm-pytorch](https://github.com/Nilthd/lstm-pytorch) repo for the solution.

## Project Structure

```
rnn-pytorch/
├── rnn.py           # Full implementation + training script
├── sample_data.csv  # Example CSV to test immediately
├── requirements.txt # Dependencies
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train on the included sample data
python rnn.py --data sample_data.csv --target target

# Train on YOUR own CSV
python rnn.py --data your_file.csv --target your_target_column

# Run demo mode (no CSV needed)
python rnn.py
```

## All Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | None | Path to your CSV file |
| `--target` | None | Column name to predict |
| `--seq_len` | 10 | How many past timesteps to use |
| `--hidden_size` | 32 | RNN hidden size |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |

## CSV Format

```
time, feature1, feature2, ..., target
0.0,  0.05,    1.03,     ..., 0.53
0.06, 0.05,    1.05,     ..., 0.51
...
```

## Key Implementation Details

- `RNNCell` implements the update rule manually: `h_t = tanh(W·[x, h] + b)`
- `SequenceDataset` builds overlapping windows from time-series data
- Chronological train/val/test split (no data leakage)
- Learning rate scheduler halves LR every 10 epochs
- Best model checkpoint saved based on validation loss

## Related Projects

- [lstm-pytorch](https://github.com/Nilthd/lstm-pytorch) — LSTM with forget/input/output gates
- transformer-pytorch *(coming soon)* — attention-based sequence model

## Author

Niloofar Tavahoodi — M.A.Sc. Candidate, Electrical & Computer Engineering, University of Victoria
