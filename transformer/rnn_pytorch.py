"""
rnn_pytorch.py

The exact same character-level RNN as rnn_raw.py, same equations,
same toy dataset -- but built with torch.nn and trained with
autograd + an optimizer instead of hand-written BPTT.

Compare this file to rnn_raw.py side by side:
  - Wxh, Whh, Why, bh, by  -->  nn.Linear layers (PyTorch bundles
    weight + bias into one object)
  - the manual `for t in reversed(range(...))` backward pass
    -->  a single call to loss.backward()
  - manual gradient descent (`Wxh -= lr * dWxh`)
    -->  optimizer.step()

Same math, same "for each timestep, carry h_t forward" structure --
just with the bookkeeping automated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# 1. Data -- identical toy corpus to rnn_raw.py
# ---------------------------------------------------------------------------
text = "hello world, this is a tiny rnn demo. rnns remember what came before."
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

# ---------------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------------
hidden_size = 32
seq_length = 16
learning_rate = 0.1


class CharRNN(nn.Module):
    """
    Same recurrence as rnn_raw.py, spelled out explicitly with nn.Linear
    layers rather than using the built-in nn.RNN module, so the equations
    line up one-to-one with the numpy version:

        h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
        y_t = Why @ h_t + by
    """

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.xh = nn.Linear(vocab_size, hidden_size, bias=True)   # Wxh, bh
        self.hh = nn.Linear(hidden_size, hidden_size, bias=False)  # Whh
        self.hy = nn.Linear(hidden_size, vocab_size, bias=True)   # Why, by

        # match the small-scale init used in rnn_raw.py
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.01)

    def forward(self, inputs, h_prev):
        """
        inputs: LongTensor of shape (seq_length,) -- char indices
        h_prev: FloatTensor of shape (1, hidden_size)

        Returns logits for every timestep, plus the final hidden state.
        """
        x_onehot = F.one_hot(inputs, num_classes=vocab_size).float()  # (seq_len, vocab)

        h = h_prev
        logits = []
        for t in range(inputs.size(0)):
            x_t = x_onehot[t].unsqueeze(0)                    # (1, vocab_size)
            h = torch.tanh(self.xh(x_t) + self.hh(h))          # h_t update
            y_t = self.hy(h)                                   # (1, vocab_size)
            logits.append(y_t)

        logits = torch.cat(logits, dim=0)   # (seq_len, vocab_size)
        return logits, h.detach()           # detach: don't backprop past this chunk


# ---------------------------------------------------------------------------
# 3. Sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(model, h, seed_idx, n):
    idx = seed_idx
    out = []
    for _ in range(n):
        x_onehot = F.one_hot(torch.tensor([idx]), num_classes=vocab_size).float()
        h = torch.tanh(model.xh(x_onehot) + model.hh(h))
        logits = model.hy(h)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        idx = torch.multinomial(probs, num_samples=1).item()
        out.append(idx)
    return "".join(idx_to_char[i] for i in out)


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
def train(num_steps=2000):
    model = CharRNN(vocab_size, hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n, pos = 0, 0
    h_prev = torch.zeros(1, hidden_size)
    smooth_loss = -torch.log(torch.tensor(1.0 / vocab_size)).item() * seq_length

    while n < num_steps:
        if pos + seq_length + 1 >= len(data) or n == 0:
            h_prev = torch.zeros(1, hidden_size)
            pos = 0

        inputs = data[pos:pos + seq_length]
        targets = data[pos + 1:pos + seq_length + 1]

        logits, h_prev = model(inputs, h_prev)
        loss = F.cross_entropy(logits, targets)   # replaces the manual softmax + -log(p) loop

        optimizer.zero_grad()
        loss.backward()                            # replaces the entire manual BPTT loop
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # same as np.clip in rnn_raw.py
        optimizer.step()                            # replaces the manual weight -= lr * grad

        smooth_loss = smooth_loss * 0.999 + loss.item() * seq_length * 0.001

        if n % 200 == 0:
            print(f"step {n:4d}  smooth_loss {smooth_loss:.4f}")
            sample_text = sample(model, h_prev, inputs[0].item(), 60)
            print(f"          sample: {sample_text!r}")

        pos += seq_length
        n += 1


if __name__ == "__main__":
    print(f"vocab_size={vocab_size}, chars={chars}")
    train()
