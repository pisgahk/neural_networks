"""
transformer_pytorch.py

The same decoder-only (GPT-style) Transformer as transformer_python.py --
same architecture, same equations -- but built with torch.nn and
actually trained with autograd, on the same toy corpus style used in
the RNN examples.

Compare structurally to transformer_python.py:
    MultiHeadAttention class   <-> same class, same Q/K/V/O math
    FeedForward class          <-> same class, same two-linear-layers-and-ReLU
    DecoderBlock class         <-> same residual + layer norm wrapping
    TinyTransformer class      <-> same embed -> blocks -> project-to-vocab flow

    manual softmax(QK^T/sqrt(d_k))V loop  -->  same formula, but via
                                                 tensor ops PyTorch can
                                                 autograd through
    (no training in the numpy version)    -->  loss.backward() + optimizer.step()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
text = ("hello world, this is a tiny transformer demo. "
        "transformers use attention instead of recurrence. "
        "attention lets every token look at every other token directly.")
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

# ---------------------------------------------------------------------------
# 2. Hyperparameters
# ---------------------------------------------------------------------------
d_model = 64
num_heads = 4
d_ff = 128
num_layers = 3
block_size = 32     # context window (max sequence length attended over)
batch_size = 16
learning_rate = 3e-3


# ---------------------------------------------------------------------------
# 3. Positional encoding (same sinusoidal formula as the numpy version)
# ---------------------------------------------------------------------------
def sinusoidal_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


# ---------------------------------------------------------------------------
# 4. Multi-head self-attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal_mask):
        # x: (batch, seq_len, d_model)
        B, T, _ = x.shape

        Q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, heads, T, d_k)
        K = self.Wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # softmax(QK^T / sqrt(d_k)) V  -- equation (1) from the paper,
        # computed for all heads and all batch elements at once
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)   # (B, heads, T, T)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                                          # (B, heads, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)       # concat heads
        return self.Wo(out)


# ---------------------------------------------------------------------------
# 5. Feed-forward network
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 6. Decoder block: attention + FFN, each with residual + layer norm
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask):
        x = self.ln1(x + self.attn(x, causal_mask))   # residual around attention
        x = self.ln2(x + self.ffn(x))                  # residual around feed-forward
        return x


# ---------------------------------------------------------------------------
# 7. Full model
# ---------------------------------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_encoding", sinusoidal_positional_encoding(block_size, d_model))
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, token_ids):
        # token_ids: (batch, seq_len)
        B, T = token_ids.shape
        x = self.token_embedding(token_ids) + self.pos_encoding[:T]

        causal_mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)  # 1 = allowed, 0 = masked
        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)   # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size:]
            logits = self(context)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_id], dim=1)
        return token_ids


# ---------------------------------------------------------------------------
# 8. Batching + training loop
# ---------------------------------------------------------------------------
def get_batch():
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x, y


def train(num_steps=1500):
    model = TinyTransformer(vocab_size, d_model, num_heads, d_ff, num_layers, block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(num_steps):
        xb, yb = get_batch()
        logits = model(xb)   # (batch, seq_len, vocab_size)

        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 150 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    return model


if __name__ == "__main__":
    print(f"vocab_size={vocab_size}, chars={chars}")
    model = train()

    prompt = "hello "
    ids = torch.tensor([[char_to_idx[c] for c in prompt]], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=60)
    generated = "".join(idx_to_char[i] for i in out[0].tolist())
    print(f"\nprompt:    {prompt!r}")
    print(f"generated: {generated!r}")
