"""
transformer_python.py

A decoder-only Transformer (GPT-style, causal self-attention) built from
scratch with numpy only -- no autograd, no PyTorch.

This implements the FORWARD PASS of every mechanism from "Attention Is
All You Need" (Vaswani et al., 2017):
    - token embeddings + sinusoidal positional encoding
    - scaled dot-product attention:  softmax(QK^T / sqrt(d_k)) V
    - multi-head attention (several attention "views" run in parallel)
    - causal masking (a token may only attend to itself and the past)
    - residual connections + layer normalization
    - position-wise feed-forward network
    - a stack of N identical decoder blocks
    - final linear projection to vocabulary logits

What's deliberately NOT here: backpropagation / training. Manually
deriving gradients through multi-head attention, layer norm, and
residual stacks by hand is a huge amount of fragile chain-rule
bookkeeping -- which is exactly the point: this is the concrete,
practical reason autograd (PyTorch) exists. See transformer_pytorch.py
for the trainable version.

Run this file and it does a forward pass on a toy sentence with
randomly initialized weights, so you can see the data flow and the
shape of every intermediate tensor.
"""

import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Toy data -- same style corpus as the RNN examples, char-level
# ---------------------------------------------------------------------------
text = "hello world, this is a tiny transformer demo."
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# ---------------------------------------------------------------------------
# 2. Hyperparameters
# ---------------------------------------------------------------------------
d_model = 32       # embedding / residual stream dimension
num_heads = 4       # number of parallel attention heads
d_k = d_model // num_heads   # dimension per head (the "d_k" in the paper's formula)
d_ff = 64       # hidden dimension of the feed-forward network
num_layers = 2       # number of stacked decoder blocks
block_size = 16       # max sequence length / context window


def linear_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * (fan_in ** -0.5)


# ---------------------------------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------------------------------
token_embedding = np.random.randn(vocab_size, d_model) * 0.02


def sinusoidal_positional_encoding(seq_len, d_model):
    """
    The fixed (non-learned) positional encoding from section 3.5 of the
    paper:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Attention itself has no notion of order (it's a weighted average over
    a *set* of positions) -- this is what injects "position" into the model.
    """
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


# ---------------------------------------------------------------------------
# 4. Core building block: scaled dot-product attention
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: shape (seq_len, d_k)
    Implements:  softmax(QK^T / sqrt(d_k)) V     -- equation (1) in the paper

    mask: optional (seq_len, seq_len) array with -inf in positions that
    must not be attended to (used for the causal mask).
    """
    scores = Q @ K.T / np.sqrt(d_k)          # (seq_len, seq_len) raw relevance scores
    if mask is not None:
        scores = scores + mask               # forbidden positions -> -inf before softmax
    weights = softmax(scores, axis=-1)        # each row sums to 1
    output = weights @ V                      # weighted sum of values
    return output, weights


def causal_mask(seq_len):
    """
    Upper-triangular mask of -inf so position t cannot see positions > t.
    This is what makes it a *decoder* (autoregressive, GPT-style) rather
    than an encoder with full bidirectional attention.
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    return mask


# ---------------------------------------------------------------------------
# 5. Multi-head attention
# ---------------------------------------------------------------------------
class MultiHeadAttention:
    """
    Instead of one attention computation over the full d_model dimension,
    split into `num_heads` smaller attention computations run in parallel,
    each free to learn a different notion of "relevance" (e.g. one head
    might track adjacent tokens, another long-range dependencies).
    Outputs are concatenated and mixed back together with W_O.
    """

    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = linear_init(d_model, d_model)
        self.Wk = linear_init(d_model, d_model)
        self.Wv = linear_init(d_model, d_model)
        self.Wo = linear_init(d_model, d_model)

    def __call__(self, x, mask=None):
        seq_len, _ = x.shape
        Q = x @ self.Wq   # (seq_len, d_model)
        K = x @ self.Wk
        V = x @ self.Wv

        # split into heads: (seq_len, d_model) -> (num_heads, seq_len, d_k)
        Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        head_outputs = []
        for h in range(self.num_heads):
            out, _ = scaled_dot_product_attention(Q[h], K[h], V[h], mask)
            head_outputs.append(out)

        # concatenate heads back: (num_heads, seq_len, d_k) -> (seq_len, d_model)
        concat = np.concatenate(head_outputs, axis=-1)
        return concat @ self.Wo   # final linear mix, equation (2) "MultiHead(...)"


# ---------------------------------------------------------------------------
# 6. Layer normalization
# ---------------------------------------------------------------------------
def layer_norm(x, gamma, beta, eps=1e-5):
    """Normalize across the feature dimension, then apply a learned scale/shift."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


# ---------------------------------------------------------------------------
# 7. Position-wise feed-forward network
# ---------------------------------------------------------------------------
class FeedForward:
    """
    Section 3.3 of the paper: FFN(x) = max(0, x W1 + b1) W2 + b2
    Applied identically (same weights) to every position independently --
    this is where each token's representation gets "thought about" after
    attention has finished mixing information *between* tokens.
    """

    def __init__(self, d_model, d_ff):
        self.W1 = linear_init(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = linear_init(d_ff, d_model)
        self.b2 = np.zeros(d_model)

    def __call__(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        return hidden @ self.W2 + self.b2


# ---------------------------------------------------------------------------
# 8. A single decoder block: attention + FFN, each wrapped in a residual
#    connection and layer norm
# ---------------------------------------------------------------------------
class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1_gamma, self.ln1_beta = np.ones(d_model), np.zeros(d_model)
        self.ln2_gamma, self.ln2_beta = np.ones(d_model), np.zeros(d_model)

    def __call__(self, x, mask):
        # residual connection: output = input + sublayer(input)
        # this is what lets gradients (and, here, information) skip
        # straight through a layer instead of being forced through it
        attn_out = self.attn(x, mask)
        x = layer_norm(x + attn_out, self.ln1_gamma, self.ln1_beta)

        ffn_out = self.ffn(x)
        x = layer_norm(x + ffn_out, self.ln2_gamma, self.ln2_beta)
        return x


# ---------------------------------------------------------------------------
# 9. The full model: embed -> N decoder blocks -> project to vocab
# ---------------------------------------------------------------------------
class TinyTransformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, block_size):
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_encoding = sinusoidal_positional_encoding(block_size, d_model)
        self.blocks = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.W_out = linear_init(d_model, vocab_size)   # final projection to logits

    def __call__(self, token_ids):
        seq_len = len(token_ids)
        x = self.token_embedding[token_ids] + self.pos_encoding[:seq_len]   # (seq_len, d_model)

        mask = causal_mask(seq_len)
        for block in self.blocks:
            x = block(x, mask)

        logits = x @ self.W_out   # (seq_len, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# 10. Demo: forward pass + greedy next-character prediction
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = TinyTransformer(vocab_size, d_model, num_heads, d_ff, num_layers, block_size)

    prompt = "hello wor"
    token_ids = [char_to_idx[ch] for ch in prompt]

    logits = model(token_ids)
    print(f"input:  {prompt!r}")
    print(f"logits shape: {logits.shape}  (seq_len={len(token_ids)}, vocab_size={vocab_size})")

    # look at the prediction for the NEXT character after the last input token
    last_logits = logits[-1]
    probs = softmax(last_logits)
    predicted_idx = np.argmax(probs)
    print(f"predicted next char (untrained, effectively random): {idx_to_char[predicted_idx]!r}")
    print("\nNote: weights are randomly initialized and never trained here --")
    print("this file exists to show the forward-pass data flow, not to produce")
    print("good predictions. See transformer_pytorch.py for a trained version.")
