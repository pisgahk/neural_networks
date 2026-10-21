import numpy as np


class EmbeddingLayer:
    # Basically converting the tokens into more arrays that will represent the meaning of the inputs.
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.02
        self.inputs = None

        # Gradients
        self.grad_embeddings = np.zeros(self.embeddings.shape)

    def forward(self, inputs):
        self.inputs = inputs
        return self.embeddings[inputs]

    def backward(self, prev_grad):
        grad_embeddings = np.zeros(self.embeddings.shape)
        np.add.at(grad_embeddings, self.inputs, prev_grad)

        self.grad_embeddings += grad_embeddings
        return grad_embeddings

    def get_params(self):
        # [ADDED] convention for every get_params(): (param, grad) pairs under layer_parameters
        return {
            "layer_name": "EmbeddingLayer",
            "layer_parameters": {"embeddings": (self.embeddings, self.grad_embeddings)},
            "children": None,
        }


class Softmax:
    def __init__(self):
        self.softmax_out = None

    def forward(self, inputs):
        # Softmax is shift-invariant so we substract max_logit for numerical stability
        max_logit = np.max(inputs, axis=-1, keepdims=True)

        # Softmax output
        softmax_out = np.exp(inputs - max_logit) / np.sum(
            np.exp(inputs - max_logit), axis=-1, keepdims=True
        )
        self.softmax_out = softmax_out

        return softmax_out

    def backward(self, prev_grad):
        return self.softmax_out * (
            prev_grad - np.sum(prev_grad * self.softmax_out, axis=-1, keepdims=True)
        )  # [ADDED]


class CrossEntropyWithSoftmax:
    def __init__(self):
        self.lse = None
        self.logits_flat = None
        self.labels_flat = None

    def forward(self, logits, labels):
        assert len(logits.shape) == 3
        B, T, V = logits.shape

        logits_flat = np.reshape(logits, (B * T, V))
        labels_flat = np.reshape(labels, (B * T))

        # Numerically stable cross entropy loss
        # cool explanation: https://jaykmody.com/blog/stable-softmax/
        max_float = np.max(logits_flat, axis=-1, keepdims=True)
        diff = logits_flat - max_float

        lse = max_float + np.log(np.sum(np.exp(diff), axis=-1, keepdims=True))
        log_softmax = (
            np.expand_dims(logits_flat[np.arange(B * T), labels_flat], axis=-1) - lse
        )

        self.lse = lse
        self.logits_flat = logits_flat
        self.labels_flat = labels_flat
        self.B = B
        self.T = T
        self.V = V

        return -np.sum(log_softmax) / logits_flat.shape[0]

    def backward(self):
        # [ADDED] dL/dlogits = (softmax - one_hot(labels)) / N, with softmax = exp(logits - lse)
        softmax = np.exp(self.logits_flat - self.lse)
        softmax[np.arange(self.B * self.T), self.labels_flat] -= 1
        grad = softmax / self.logits_flat.shape[0]
        return np.reshape(grad, (self.B, self.T, self.V))

    def get_name(self):
        return "CrossEntropyWithSoftmax"  # [ADDED]


class Z_LossWithSoftmax:
    pass  # [OPTIONAL] z-loss variant (not in the video); delete this `pass` and uncomment the class body to use it
    # PaLM-style z-loss: total = cross entropy + z_coeff * mean(logsumexp ** 2)
    # It keeps the logits from drifting to huge values. Not used by GPT unless you swap it in.

    # def __init__(self, z_coeff=1e-4):
    #     self.z_coeff = z_coeff
    #     self.ce = CrossEntropyWithSoftmax()

    # def forward(self, logits, labels):
    #     ce_loss = self.ce.forward(logits, labels)  # also caches lse, logits_flat, labels_flat, B, T, V
    #     z_loss = self.z_coeff * np.mean(np.square(self.ce.lse))
    #     return ce_loss + z_loss

    # def backward(self):
    #     ce = self.ce
    #     N = ce.logits_flat.shape[0]
    #     softmax = np.exp(ce.logits_flat - ce.lse)

    #     # d(z_loss)/d(logits) = z_coeff * 2 * lse * softmax / N   (because d lse / d logit = softmax)
    #     grad_z = self.z_coeff * 2 * ce.lse * softmax / N

    #     grad_ce = softmax.copy()
    #     grad_ce[np.arange(N), ce.labels_flat] -= 1
    #     grad_ce /= N

    #     return np.reshape(grad_ce + grad_z, (ce.B, ce.T, ce.V))

    # def get_name(self):
    #     return "Z_LossWithSoftmax"


class Linear:
    def __init__(self, input_units, output_units, include_bias=True):
        self.weights = np.random.randn(input_units, output_units) * 0.02
        if include_bias:
            self.bias = np.zeros((1, output_units))
            self.grad_bias = np.zeros(self.bias.shape)
        self.inputs = None
        self.include_bias = include_bias

        # Gradients
        self.grad_weights = np.zeros(self.weights.shape)

    def forward(self, inputs):
        self.inputs = inputs
        if self.include_bias:
            return inputs @ self.weights + self.bias
        else:
            return inputs @ self.weights

    def backward(self, prev_grad):
        inputs_flat = self.inputs.reshape(-1, self.inputs.shape[-1])
        prev_grad_flat = prev_grad.reshape(-1, prev_grad.shape[-1])

        dL_dA = prev_grad @ self.weights.T
        dL_dW = inputs_flat.T @ prev_grad_flat

        if self.include_bias:
            dL_dB = np.sum(prev_grad_flat, axis=0, keepdims=True)
            self.grad_bias += dL_dB

        # Save gradients
        self.grad_weights += dL_dW

        return dL_dA  # [ADDED]

    def get_params(self):  # [ADDED]
        params = {"weights": (self.weights, self.grad_weights)}
        if self.include_bias:
            params["bias"] = (self.bias, self.grad_bias)
        return {"layer_name": "Linear", "layer_parameters": params, "children": None}

    # (lines 170-178 were folded and never shown; the class may hold more methods)


class ReLU:
    def __init__(self):  # [ADDED] whole ReLU body
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, prev_grad):
        return prev_grad * (self.inputs > 0)


class FeedForwardNetwork:
    def __init__(self, input_units, hidden_units, output_units):
        self.linear1 = Linear(input_units, hidden_units)
        self.activation = ReLU()
        self.linear2 = Linear(hidden_units, output_units)

    def forward(self, inputs):
        x = self.linear1.forward(inputs)
        x = self.activation.forward(x)
        x = self.linear2.forward(x)
        return x

    def backward(self, prev_grad):
        prev_grad = self.linear2.backward(prev_grad)
        prev_grad = self.activation.backward(prev_grad)
        prev_grad = self.linear1.backward(prev_grad)
        return prev_grad

    def get_params(self):
        return {
            "layer_name": "FNN",
            "layer_parameters": None,
            "children": [self.linear1.get_params(), self.linear2.get_params()],
        }


class LayerNorm:
    def __init__(self, input_dim):
        # Parameters
        self.input_dim = input_dim
        self.gain = np.ones((1, input_dim))
        self.bias = np.zeros((1, input_dim))

        # Things to keep track of
        self.mean = None
        self.mu = None
        self.var = None
        self.std = None
        self.out = None

        # Gradients
        self.grad_gain = np.zeros(self.gain.shape)
        self.grad_bias = np.zeros(self.bias.shape)

    def forward(self, inputs):
        epsilon = 1e-5
        self.mean = (1 / self.input_dim) * np.sum(inputs, axis=-1, keepdims=True)
        self.mu = inputs - self.mean
        self.var = (1 / self.input_dim) * np.sum(
            np.square(self.mu), axis=-1, keepdims=True
        )
        self.std = np.sqrt(self.var + epsilon)
        self.out = self.gain * (self.mu / self.std) + self.bias

        return self.out

    def backward(self, prev_grad):
        # Derivation of LayerNorm is kinda tedious and hard
        # I used https://robotchinwag.com/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/
        direct_path = prev_grad * (self.gain / self.std)
        indirect_path = np.sum(
            (
                prev_grad[..., np.newaxis]
                * 1
                / (self.input_dim)
                * (self.gain[..., np.newaxis])
                * (
                    1 / (self.std[..., np.newaxis])
                    + (
                        np.expand_dims(self.mu, axis=-1)
                        * np.expand_dims(self.mu, axis=-2)
                    )
                    / (np.power(self.std[..., np.newaxis], 3))
                )
            ),  # [RECONSTRUCTED] screenshot cuts off after "np.power(se" -- ending derived from the LayerNorm gradient math, verify against the source
            axis=-2,
        )
        dL_dX = direct_path - indirect_path
        dL_dGain = np.sum(
            np.sum(prev_grad * (self.mu / self.std), axis=0, keepdims=True), axis=1
        )

        dL_dB = np.sum(np.sum(prev_grad, axis=0, keepdims=True), axis=1)

        # Save gradients
        self.grad_gain += dL_dGain
        self.grad_bias += dL_dB

        return dL_dX

    def get_params(self):
        return {
            "layer_name": "LayerNorm",  # [ADDED]
            "layer_parameters": {
                "gain": (self.gain, self.grad_gain),
                "bias": (self.bias, self.grad_bias),
            },
            "children": None,
        }


class MultiHeadAttention:
    def __init__(self, dim_in, dim_out, n_heads):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_heads = n_heads

        self.query_linear = Linear(dim_in, dim_out, include_bias=False)
        self.key_linear = Linear(dim_in, dim_out, include_bias=False)
        self.value_linear = Linear(dim_in, dim_out, include_bias=False)
        self.softmax = Softmax()
        self.final_projection = Linear(dim_out, dim_out, include_bias=False)

        # Save for backprop
        self.queries_matrix = None
        self.keys_matrix = None
        self.values_matrix = None
        self.attention_weights = None

    def forward(self, inputs, mask=None):
        B, T, D = inputs.shape

        queries = self.query_linear.forward(inputs)
        keys = self.key_linear.forward(inputs)
        values = self.value_linear.forward(inputs)

        # (B, token_length, number_heads, head_dimension)
        queries = np.reshape(
            queries, (B, T, self.n_heads, self.dim_out // self.n_heads)
        )
        keys = np.reshape(keys, (B, T, self.n_heads, self.dim_out // self.n_heads))
        values = np.reshape(values, (B, T, self.n_heads, self.dim_out // self.n_heads))

        # (B, number_heads, token_length, head_dimension)
        queries = np.transpose(queries, axes=[0, 2, 1, 3])
        keys = np.transpose(keys, axes=[0, 2, 1, 3])
        values = np.transpose(values, axes=[0, 2, 1, 3])

        attention_scores = queries @ np.transpose(keys, axes=[0, 1, 3, 2])
        attention_scores = attention_scores / np.sqrt(keys.shape[-1])

        if mask is not None:
            attention_scores += mask

        attention_weights = self.softmax.forward(attention_scores)

        final_vector = attention_weights @ values

        # (B, token_length, number_heads, head_dimension)
        final_vector = np.transpose(final_vector, axes=[0, 2, 1, 3])
        final_vector = np.reshape(final_vector, (B, T, self.dim_out))
        final_vector = self.final_projection.forward(final_vector)

        # Store stuff for backprop
        self.queries_matrix = queries
        self.keys_matrix = keys
        self.values_matrix = values
        self.attention_weights = attention_weights

        return final_vector

    def backward(self, prev_grad):
        B, T, D = prev_grad.shape
        assert D == self.dim_out

        prev_grad = self.final_projection.backward(prev_grad)

        prev_grad = np.reshape(
            prev_grad, (B, T, self.n_heads, self.dim_out // self.n_heads)
        )

        # (B, number_heads, token_length, head_dimension)
        prev_grad = np.transpose(prev_grad, axes=[0, 2, 1, 3])

        dL_dValue = np.transpose(self.attention_weights, axes=[0, 1, 3, 2]) @ prev_grad

        dL_dAttention_Weights = prev_grad @ np.transpose(
            self.values_matrix, axes=[0, 1, 3, 2]
        )
        dL_dAttention_Scores = (
            self.softmax.backward(dL_dAttention_Weights)
            * 1
            / np.sqrt(self.keys_matrix.shape[-1])
        )
        dL_dKeys = (
            np.transpose(dL_dAttention_Scores, axes=[0, 1, 3, 2]) @ self.queries_matrix
        )
        dL_dQueries = dL_dAttention_Scores @ self.keys_matrix

        # (B, token_length, number_heads, head_dimension)
        dL_dValue = np.transpose(dL_dValue, axes=[0, 2, 1, 3])
        dL_dKeys = np.transpose(dL_dKeys, axes=[0, 2, 1, 3])
        dL_dQueries = np.transpose(dL_dQueries, axes=[0, 2, 1, 3])

        # (B, token_length, dim_in)
        dL_dValue = np.reshape(dL_dValue, (B, T, self.dim_out))
        dL_dKeys = np.reshape(dL_dKeys, (B, T, self.dim_out))
        dL_dQueries = np.reshape(dL_dQueries, (B, T, self.dim_out))

        dL_dA_value_linear = self.value_linear.backward(dL_dValue)
        dL_dA_key_linear = self.key_linear.backward(dL_dKeys)
        dL_dA_query_linear = self.query_linear.backward(dL_dQueries)

        grad_in = dL_dA_value_linear + dL_dA_key_linear + dL_dA_query_linear

        # Return grad_in, grad_query_linear, grad_key_linear, grad_value_linear
        return grad_in  # [ADDED]

    def get_params(self):
        return {
            "layer_name": "MultiHeadAttention",
            "layer_parameters": None,
            "children": [
                self.query_linear.get_params(),
                self.key_linear.get_params(),
                self.value_linear.get_params(),
                self.final_projection.get_params(),
            ],
        }  # [RECONSTRUCTED] screenshot cuts off at "self.final_project" -- ending completed from the greyed-out hint "ion.get_params"


class PositionalEncodings:
    # [ADDED] fixed sinusoidal encodings (no parameters), computed for whatever sequence length arrives

    def __init__(self, dim_model):
        self.dim_model = dim_model

    def forward(self, inputs):
        B, T, D = inputs.shape
        position = np.arange(T)[:, np.newaxis]  # (T, 1)
        div_term = np.exp(np.arange(0, D, 2) * (-np.log(10000.0) / D))
        encodings = np.zeros((T, D))
        encodings[:, 0::2] = np.sin(position * div_term)  # even dims
        encodings[:, 1::2] = np.cos(position * div_term)  # odd dims
        return inputs + encodings  # broadcasts over the batch

    def backward(self, prev_grad):
        return prev_grad  # fixed encodings: the gradient passes straight through

    def get_params(self):
        return {
            "layer_name": "PositionalEncodings",
            "layer_parameters": None,
            "children": None,
        }


class TransformerBlock:
    def __init__(self, dim_model, n_heads, ln_position="pre", res_weight_init_scale=1):
        self.ln_position = ln_position

        self.multi_head_attention = MultiHeadAttention(
            dim_model, dim_model, n_heads
        )  # Masked
        self.layernorm1 = LayerNorm(dim_model)

        self.ffn = FeedForwardNetwork(dim_model, 4 * dim_model, dim_model)
        self.layernorm2 = LayerNorm(dim_model)

        # [OPTIONAL] GPT-2 style init: shrink the residual-branch output weights by res_weight_init_scale
        # self.multi_head_attention.final_projection.weights *= res_weight_init_scale
        # self.ffn.linear2.weights *= res_weight_init_scale

    def get_params(self):
        return {
            "layer_name": "TransformerBlock",  # [ADDED]
            "layer_parameters": None,
            "children": [
                self.multi_head_attention.get_params(),
                self.layernorm1.get_params(),
                self.ffn.get_params(),
                self.layernorm2.get_params(),
            ],
        }

    def forward(self, inputs, mask_attention=None, padding_mask=None):
        if self.ln_position == "post":
            out = inputs + self.multi_head_attention.forward(
                inputs, mask=mask_attention
            )
            out = self.layernorm1.forward(out)

            out = out + self.ffn.forward(out)
            out = self.layernorm2.forward(out)

        elif self.ln_position == "pre":
            out = inputs + self.multi_head_attention.forward(
                self.layernorm1.forward(inputs), mask=mask_attention
            )
            out = out + self.ffn.forward(self.layernorm2.forward(out))

        return out

    def backward(self, prev_grad):
        if (
            self.ln_position == "post"
        ):  # [ADDED] mirrors forward() in reverse; residuals add the skip gradient
            # forward was: out = LN1(x + MHA(x));  out = LN2(out + FFN(out))
            prev_grad = self.layernorm2.backward(prev_grad)
            prev_grad = prev_grad + self.ffn.backward(prev_grad)
            prev_grad = self.layernorm1.backward(prev_grad)
            prev_grad = prev_grad + self.multi_head_attention.backward(prev_grad)
        elif self.ln_position == "pre":
            # forward was: out = x + MHA(LN1(x));  out = out + FFN(LN2(out))
            prev_grad = prev_grad + self.layernorm2.backward(
                self.ffn.backward(prev_grad)
            )
            prev_grad = prev_grad + self.layernorm1.backward(
                self.multi_head_attention.backward(prev_grad)
            )
        return prev_grad


class GPT:
    def __init__(self, vocab_size, blocks, dim_model, n_heads, ln_position="pre"):
        self.vocab_size = vocab_size

        # Decoder
        self.embedding_decoder = EmbeddingLayer(vocab_size, embedding_dim=dim_model)
        self.positional_encodings_decoder = PositionalEncodings(dim_model)
        res_weight_init_scale = (2 * blocks) ** (-0.5)
        self.decoder_blocks = [
            TransformerBlock(dim_model, n_heads, ln_position, res_weight_init_scale)
            for _ in range(blocks)
        ]

        # Final Layer
        self.final_layernorm = LayerNorm(dim_model)
        self.final_linear = Linear(dim_model, self.vocab_size)
        self.loss_function = CrossEntropyWithSoftmax()

    def forward(self, input_tokens_decoder, mask_attention=None, padding_mask=None):
        # Decoder forward
        out_d = self.embedding_decoder.forward(input_tokens_decoder)
        out_d = self.positional_encodings_decoder.forward(out_d)

        for decoder in self.decoder_blocks:
            out_d = decoder.forward(
                inputs=out_d, mask_attention=mask_attention, padding_mask=padding_mask
            )

        out = self.final_layernorm.forward(out_d)
        return self.final_linear.forward(out)

    def backward(self, grad):
        grad = self.final_linear.backward(grad)
        grad = self.final_layernorm.backward(grad)

        # Backward pass for decoder
        for decoder in reversed(self.decoder_blocks):
            grad = decoder.backward(grad)

        grad = self.positional_encodings_decoder.backward(grad)
        grad = self.embedding_decoder.backward(grad)

        return grad

    def get_params(self):
        decoder_dict = {
            "layer_name": "Decoder",
            "layer_parameters": None,
            "children": [self.embedding_decoder.get_params()]
            + [dec.get_params() for dec in self.decoder_blocks],
        }

        return [
            decoder_dict,
            self.final_layernorm.get_params(),
            self.final_linear.get_params(),
        ]

    def save_model(self, str_file_path):
        state_dict = self.get_params()
        params = {}

        stack = []
        root_sibling_counts = {}
        for i, elem in enumerate(reversed(state_dict)):
            is_last = i == 0
            stack.append((elem, "", is_last, root_sibling_counts))

        while len(stack) > 0:
            elem, prefix, is_last, sibling_counts = stack.pop()

            name = elem["layer_name"]
            sibling_counts[name] = sibling_counts.get(name, 0) + 1
            full_name = name + "_" + str(sibling_counts[name])

            if (
                elem["layer_parameters"] is not None
            ):  # [ADDED] rest of save_model (top half of this line is visible)
                for param_name, (param, grad) in elem["layer_parameters"].items():
                    params[prefix + full_name + "." + param_name] = param

            # push the children reversed so they pop in order; siblings share one counter dict
            child_sibling_counts = {}
            children = elem["children"] if elem["children"] is not None else []
            for j, child in enumerate(reversed(children)):
                stack.append(
                    (child, prefix + full_name + ".", j == 0, child_sibling_counts)
                )

        np.savez(str_file_path, **params)

    # [OPTIONAL] not in the video: the counterpart of save_model, and sampling

    # def load_model(self, str_file_path):
    #     saved = np.load(str_file_path)

    #     stack = []
    #     root_sibling_counts = {}
    #     for i, elem in enumerate(reversed(self.get_params())):
    #         stack.append((elem, "", i == 0, root_sibling_counts))

    #     while len(stack) > 0:
    #         elem, prefix, is_last, sibling_counts = stack.pop()

    #         name = elem["layer_name"]
    #         sibling_counts[name] = sibling_counts.get(name, 0) + 1
    #         full_name = name + "_" + str(sibling_counts[name])

    #         if elem["layer_parameters"] is not None:
    #             for param_name, (param, grad) in elem["layer_parameters"].items():
    #                 param[...] = saved[prefix + full_name + "." + param_name]  # write into the live array

    #         child_sibling_counts = {}
    #         children = elem["children"] if elem["children"] is not None else []
    #         for j, child in enumerate(reversed(children)):
    #             stack.append((child, prefix + full_name + ".", j == 0, child_sibling_counts))

    # def generate(self, tokens, max_new_tokens, temperature=1.0):
    #     for _ in range(max_new_tokens):
    #         T = tokens.shape[1]
    #         mask = np.triu(np.full((T, T), -1e9), k=1)  # causal mask: no peeking at later tokens
    #         logits = self.forward(tokens, mask_attention=mask)[:, -1, :] / temperature  # last position only
    #         probs = Softmax().forward(logits)
    #         next_token = np.array([[np.random.choice(len(p), p=p)] for p in probs])
    #         tokens = np.concatenate([tokens, next_token], axis=1)
    #     return tokens


# [OPTIONAL] smoke test (not in the video): uncomment, then run `python gpt.py` to overfit one tiny batch
# if __name__ == "__main__":
#     def params_of(tree):
#         for d in (tree if isinstance(tree, list) else [tree]):
#             for p, g in (d["layer_parameters"] or {}).values():
#                 yield p, g
#             yield from params_of(d["children"] or [])

#     np.random.seed(0)
#     vocab_size, blocks, dim_model, n_heads, T = 11, 2, 16, 2, 8
#     model = GPT(vocab_size, blocks, dim_model, n_heads)

#     tokens = np.random.randint(0, vocab_size, size=(4, T + 1))
#     inputs, labels = tokens[:, :-1], tokens[:, 1:]  # predict the next token
#     mask = np.triu(np.full((T, T), -1e9), k=1)  # causal mask

#     lr = 0.5
#     for step in range(300 + 1):
#         for _, g in params_of(model.get_params()):
#             g[...] = 0  # gradients accumulate with +=, so reset them every step
#         loss = model.loss_function.forward(model.forward(inputs, mask_attention=mask), labels)
#         model.backward(model.loss_function.backward())
#         for p, g in params_of(model.get_params()):
#             p -= lr * g
#         if step % 50 == 0:
#             print(f"step {step:4d}  loss {loss:.4f}")

# ----------------------------------------------------------------------------
# Legend
#   (no tag)        code transcribed from the screenshots.
#   [ADDED]         live code that is NOT in the screenshots (it was folded or
#                   off-screen) but is needed for the model to run: backward
#                   passes, ReLU, PositionalEncodings, get_params, and the end of
#                   save_model. My reconstruction, placed in the line range it
#                   occupied.
#   [OPTIONAL]      extras, commented out: Z_LossWithSoftmax, the init scaling in
#                   TransformerBlock.__init__, load_model, generate and a
#                   __main__ smoke test. Uncomment to use.
#   [RECONSTRUCTED] a line cut off at the screen edge (LayerNorm.backward, line
#                   246, and MultiHeadAttention.get_params, line 366), completed
#                   by me.
#   Line numbers    aligned to the newest screenshots: PositionalEncodings 368,
#                   TransformerBlock 393 (50 lines), GPT 444. Everything above
#                   368 matches the earlier screenshots. The newest screenshots
#                   show TransformerBlock 4 lines longer than the earlier ones; I
#                   put those 4 lines at the end of __init__ (a guess), so its
#                   get_params (407), forward (416) and backward (430) may be a
#                   few lines off from the real file.
#   get_params()    my convention for the layers I had to write: {"layer_name":
#                   ..., "layer_parameters": {name: (param, grad)} or None,
#                   "children": [...] or None}. GPT.get_params() itself is real.
# ----------------------------------------------------------------------------
