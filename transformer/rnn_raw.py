"""
rnn_raw.py

A character-level Recurrent Neural Network implemented from scratch
using only numpy. No autograd -- the forward pass and the backward
pass (backpropagation through time, BPTT) are both written by hand,
so you can see exactly what PyTorch normally hides from you.

Task: given a short piece of text, learn to predict the next
character at every position (the same toy task makemore uses).

Core equations (this is the "recurrent" part):
    h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)   <- hidden state update
    y_t = Why @ h_t + by                          <- output (unnormalized logits)
    p_t = softmax(y_t)                            <- probability over next char

h_t depends on h_{t-1}, which depends on h_{t-2}, ... all the way back
to the start of the sequence. That chain is the whole idea of an RNN.
"""

import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Data: a tiny toy corpus, turned into integer-indexed characters
# ---------------------------------------------------------------------------
text = "hello world, this is a tiny rnn demo. rnns remember what came before."
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

data = [char_to_idx[ch] for ch in text]

# ---------------------------------------------------------------------------
# 2. Hyperparameters and weight initialization
# ---------------------------------------------------------------------------
hidden_size = 32
seq_length = 16     # how many characters we backprop through at once (BPTT window)
learning_rate = 0.05

# Weight matrices -- these are the SAME matrices reused at every timestep.
# That weight-sharing across time is what makes this "one network", not
# a different network per position.
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01   # input -> hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden -> hidden (the recurrence)
Why = np.random.randn(vocab_size, hidden_size) * 0.01   # hidden -> output
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx] = 1
    return v


# ---------------------------------------------------------------------------
# 3. Forward + backward pass over one sequence chunk
# ---------------------------------------------------------------------------
def loss_and_grads(inputs, targets, h_prev):
    """
    inputs, targets: lists of integer char indices, length seq_length
    h_prev: hidden state carried over from the previous chunk (shape (hidden_size, 1))

    Returns: loss, gradients for all weights, and the final hidden state
    (so the next chunk can carry it forward -- this is how "memory" persists
    across chunks longer than seq_length).
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(h_prev)
    loss = 0.0

    # ---- forward pass: unroll the recurrence one step at a time ----
    for t in range(len(inputs)):
        xs[t] = one_hot(inputs[t], vocab_size)
        hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t - 1] + bh)     # h_t = tanh(...)
        ys[t] = Why @ hs[t] + by                                 # raw logits
        exp_y = np.exp(ys[t] - np.max(ys[t]))                    # subtract max for numerical stability
        ps[t] = exp_y / np.sum(exp_y)                            # softmax
        loss += -np.log(ps[t][targets[t], 0] + 1e-12)            # cross-entropy

    # ---- backward pass: BPTT, walking back through time ----
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dh_next = np.zeros_like(hs[0])  # gradient flowing in from the *future* timestep

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1                       # gradient of softmax + cross-entropy
        dWhy += dy @ hs[t].T
        dby += dy

        # gradient reaching h_t comes from two places: the output at this
        # step, AND everything downstream that depended on h_t via h_{t+1}
        dh = Why.T @ dy + dh_next
        dh_raw = (1 - hs[t] ** 2) * dh             # backprop through tanh
        dbh += dh_raw
        dWxh += dh_raw @ xs[t].T
        dWhh += dh_raw @ hs[t - 1].T

        dh_next = Whh.T @ dh_raw                   # pass gradient further back in time

    # clip gradients -- RNNs are notorious for exploding gradients during BPTT
    for dparam in (dWxh, dWhh, dWhy, dbh, dby):
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


# ---------------------------------------------------------------------------
# 4. Sampling: generate new text from the trained model
# ---------------------------------------------------------------------------
def sample(h, seed_idx, n):
    x = one_hot(seed_idx, vocab_size)
    idxs = []
    for _ in range(n):
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        y = Why @ h + by
        exp_y = np.exp(y - np.max(y))
        p = exp_y / np.sum(exp_y)
        idx = np.random.choice(range(vocab_size), p=p.ravel())
        x = one_hot(idx, vocab_size)
        idxs.append(idx)
    return "".join(idx_to_char[i] for i in idxs)


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
def train(num_steps=2000):
    n, pos = 0, 0
    h_prev = np.zeros((hidden_size, 1))
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length

    while n < num_steps:
        # reset memory at the start of the data, or if the next chunk won't fit
        if pos + seq_length + 1 >= len(data) or n == 0:
            h_prev = np.zeros((hidden_size, 1))
            pos = 0

        inputs = data[pos:pos + seq_length]
        targets = data[pos + 1:pos + seq_length + 1]

        loss, dWxh, dWhh, dWhy, dbh, dby, h_prev = loss_and_grads(inputs, targets, h_prev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # vanilla gradient descent update
        global Wxh, Whh, Why, bh, by
        Wxh -= learning_rate * dWxh
        Whh -= learning_rate * dWhh
        Why -= learning_rate * dWhy
        bh -= learning_rate * dbh
        by -= learning_rate * dby

        if n % 200 == 0:
            print(f"step {n:4d}  smooth_loss {smooth_loss:.4f}")
            sample_text = sample(h_prev, inputs[0], 60)
            print(f"          sample: {sample_text!r}")

        pos += seq_length
        n += 1


if __name__ == "__main__":
    print(f"vocab_size={vocab_size}, chars={chars}")
    train()
