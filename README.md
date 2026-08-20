# dl_impl

Hands-on practice implementations of deep learning, built from scratch while following Andrej Karpathy's "Neural Networks: Zero to Hero" style of learning. The path goes from a single neuron to a GPT-style Transformer trained on Tiny Shakespeare.

## Learning path

| # | Stage | Files | What it covers |
|---|-------|-------|----------------|
| 1 | Neuron basics | [perceptron.ipynb](perceptron.ipynb), [activation.ipynb](activation.ipynb), [loss_func.ipynb](loss_func.ipynb), [run.py](run.py) | Perceptron (incl. solving XOR with two layers), activation functions (sigmoid etc.), MSE loss, and a single sigmoid neuron trained with gradient descent in NumPy |
| 2 | Autograd & MLP | [backprop.ipynb](backprop.ipynb), [micrograd.ipynb](micrograd.ipynb), [mlp.ipynb](mlp.ipynb), [final_nn.ipynb](final_nn.ipynb) | A scalar `Value` class with backpropagation, then `Module` / `Linear` / activation / `Layer` classes composed into a small neural net library |
| 3 | Character-level LMs | [makemore/](makemore/) | Bigram model, MLP language model, batch norm and activation/gradient diagnostics, manual backprop, WaveNet-style model. Trained on `names.txt` |
| 4 | Sequence models | [rnn_2_transformer/rnn_scratch.ipynb](rnn_2_transformer/rnn_scratch.ipynb) | RNN from scratch, as the bridge to Transformers |
| 5 | GPT | [gpt-from-scratch.ipynb](gpt-from-scratch.ipynb) | Decoder-only Transformer trained on Tiny Shakespeare (`input.txt`) |

## Highlights

### makemore (`makemore/`)
- `makemore_1` – bigram counts, probabilities, sampling, smoothing, one-hot encoding, negative log-likelihood
- `makemore_2` – MLP with embeddings, train/val/test split (80/10/10)
- `makemore_3`, `makemore_part3_bn_karpathy` – initialization, BatchNorm, loss logging
- `makemore_backprop` – manual backpropagation through the network
- `wavenet` – hierarchical, WaveNet-style architecture

### GPT from scratch (`gpt-from-scratch.ipynb`)
- Character-level tokenizer (`encode` / `decode`), 90/10 train/val split
- Self-attention "mathematical trick": averaging with a lower-triangular matrix, then masked softmax
- `Head` (causal self-attention), `MultiHeadAttention`, `FeedForward`, Transformer blocks
- AdamW training loop and text generation

Hyperparameters: `n_embd=384`, `n_head=6`, `n_layer=6`, `block_size=256`, `batch_size=64`, `dropout=0.2`.

## Setup

```bash
pip install numpy torch matplotlib seaborn scipy ipywidgets jupyter
jupyter notebook
```

Notes:
- `gpt-from-scratch.ipynb` was written on Kaggle and reads the dataset from `/kaggle/input/...`. To run it locally, change that path to `input.txt`. A GPU is recommended for the full-size config.
- `rnn_scratch.ipynb` reads `../input.txt`, so run it from its own folder.
- `makemore/` notebooks read `names.txt` from the same folder.

## Data

- [input.txt](input.txt) – Tiny Shakespeare (~1.1 MB)
- [makemore/names.txt](makemore/names.txt) – list of names for the makemore models

## Other

- [claude-learn/](claude-learn/) – notes/instructions file for learning with Claude

## Status

Learning project, not a library. Notebooks contain experiments and scratch cells.
