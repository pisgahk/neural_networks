# neural_networks

A hands-on Python implementation following Andrej Karpathy's neural networks series from the ground up — starting with **Micrograd**, progressing through **Makemore**, and working toward a full **GPT** implementation.

---

## roadmap

| Stage     | Description                                          | Status         |
| --------- | ---------------------------------------------------- | -------------- |
| Micrograd | Scalar-valued autograd engine + backpropagation      | ✅ In progress |
| Makemore  | Character-level language models (bigram → MLP → RNN) | 🔜 Upcoming    |
| GPT       | Transformer-based language model from scratch        | 🔜 Upcoming    |

---

## what is this

This repo follows [Andrej Karpathy's](https://github.com/karpathy) "Neural Networks: Zero to Hero" lecture series. The goal is to build neural networks from scratch — no PyTorch autograd magic, no shortcuts — to deeply understand what's happening under the hood.

### micrograd

A tiny scalar-valued automatic differentiation engine. Every operation (`+`, `*`, `tanh`, etc.) builds a computation graph, and backpropagation is implemented manually through it. Visualisation of the computation graph is done with Graphviz.

---

## setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

**Clone the repo:**

```bash
git clone https://github.com/pisgahk/neural_networks.git
cd neural_networks
```

**Create the environment and install dependencies:**

```bash
uv venv
uv sync
```

**Run:**

```bash
uv run python main.py
```

---

## dependencies

- `numpy` — numerical operations
- `matplotlib` — plotting loss curves and activations
- `graphviz` — computation graph visualisation

> **Note:** Graphviz requires the system binary in addition to the Python package.
> Install it with `sudo apt install graphviz` on Debian/Ubuntu.

---

## project structure

```
neural_networks/
├── micrograd.py       # Autograd engine and neural net primitives
├── main.py            # Entry point / experiments
├── pyproject.toml     # Project dependencies (uv)
├── uv.lock            # Lockfile
└── .python-version    # Pinned Python version
```

---

## references

- [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [micrograd (original)](https://github.com/karpathy/micrograd)
- [makemore (original)](https://github.com/karpathy/makemore)
