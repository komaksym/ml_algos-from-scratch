from dataclasses import dataclass

import numpy as np

np.random.seed(42)


class BaseClass:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        pass


class BPE:
    def __init__(self):
        self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

    def encode(self, text: str):
        tokens = text.strip().split()
        return [
            self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens
        ]

    def decode(self, token_ids: list):
        reversed_dict = {v: k for k, v in self.encoder_dict.items()}
        return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])


@dataclass
class ModelConfig:
    hparams = {"n_ctx": 1024, "n_head": 12}
    params = {
        "wte": np.random.rand(3, 10),
        "wpe": np.random.rand(1024, 10),
        "blocks": [],
        "ln_f": {
            "g": np.ones(10),
            "b": np.zeros(10),
        },
    }
    bpe = BPE()

    def get_config(self):
        return self.hparams, self.params, self.bpe


class Generator(BaseClass):
    def __init__(self, model_config: ModelConfig):
        hparams, params, self.bpe = model_config.get_config()

        self.n_ctx = hparams["n_ctx"]
        self.gpt2 = GPT2(**params, n_heads=hparams["n_head"])

    def prepare(self, prompt, n_tokens_to_generate):
        inputs = self.bpe.encode(prompt)
        assert len(inputs) + n_tokens_to_generate <= self.n_ctx
        return inputs

    def forward(self, prompt, n_tokens_to_generate):
        # Encode
        inputs = self.prepare(prompt, n_tokens_to_generate)

        for _ in range(n_tokens_to_generate):
            logits = self.gpt2(inputs)  # seq_len, vocab_size
            next_id = np.argmax(logits[-1])  # (1)
            inputs.append(next_id)

        # Outputs
        outputs = inputs[len(inputs) - n_tokens_to_generate :]

        # Decode
        decoded = self.bpe.decode(outputs)  # (n, )
        return decoded


class GPT2(BaseClass):
    def __init__(self, wte, wpe, blocks, ln_f, n_heads):
        self.wte = wte
        self.wpe = wpe

        self.transformer_blocks = [
            TransformerBlock(**block, n_heads=n_heads) for block in blocks
        ]  # seq_len, n_emb
        self.layernorm = LayerNorm(**ln_f)

    def forward(self, x):
        x = self.wte[x] + self.wpe[range(len(x))]  # seq_len, n_emb
        for t_block in self.transformer_blocks:
            x = t_block(x)
        return Linear(self.layernorm(x), self.wte.T)  # seq_len, vocab_size


class TransformerBlock(BaseClass):
    def __init__(self, ln_1, ln_2, mlp, attn, n_heads):
        self.mhsa = MHSA(n_heads, **attn)
        self.ffnn = FFNN(**mlp)
        self.layernorm1 = LayerNorm(**ln_1)
        self.layernorm2 = LayerNorm(**ln_2)

    def forward(self, x):
        x = x + self.mhsa(self.layernorm1(x))  # seq_len, n_emb
        x = x + self.ffnn(self.layernorm2(x))  # seq_len, n_emb
        return x


class MHSA(BaseClass):
    def __init__(self, n_heads, c_attn, c_proj):
        self.n_heads = n_heads
        self.c_attn = c_attn
        self.c_proj = c_proj

    def compute_attention(self, q, k, v, mask):
        return Softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v  # (seq_len, v_dim)

    def project(self, out_heads):
        return Linear(np.hstack(out_heads), **self.proj)  # (seq_len, out_dim)

    def split_qkv_heads(self, x):
        qkv_heads = list(
            map(lambda x: np.split(x, self.n_heads, axis=-1), np.split(x, 3, axis=-1))
        )  # [[(seq_len, n_emb // n_heads)] x n_heads] x 3
        return qkv_heads

    def causal_mask(self, seq_len):
        return (1 - np.tri(seq_len, dtype=seq_len.dtype)) * -1e10  # (seq_len, seq_len)

    def forward(self, x):
        x = Linear(x, **self.c_attn)  # (seq_len, out_dim)
        # Slice qkv into Q, K, V and into n_heads
        qkv_heads = self.split_qkv_heads(
            x
        )  # [[(seq_len, n_emb // n_heads)] x n_heads] x 3
        # Create a mask
        mask = self.causal_mask(x.shape[0])  # (seq_len, seq_len)
        # Iterate over QKV lists and compute attention over all of them
        out_heads = [
            self.compute_attention(q, k, v, mask) for q, k, v in zip(*qkv_heads)
        ]  # [(seq_len, v_dim) x n_heads]
        # Project down to base shape
        x = self.project(out_heads)  # (seq_len, out_dim)
        return x


class FFNN(BaseClass):
    def __init__(self, c_fc, c_proj):
        self.c_fc = c_fc
        self.c_proj = c_proj

    def GELU(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def project(self, x):
        return Linear(x, **self.c_proj)

    def forward(self, x):
        self.project(self.GELU(Linear(x, **self.c_fc)))  # (seq_len, n_emb)


class LayerNorm(BaseClass):
    def __init__(self, g, b, eps=1e-5):
        self.g = g
        self.b = b
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.sum((x - mean) ** 2, axis=-1, keepdims=True) / x.shape[-1]
        return self.g * (x - mean) / np.sqrt(var + self.eps) + self.b


def Softmax(x, axis=-1):
    # Subtracting the max to avoid numerical overflow
    max_vals = np.max(x, axis=axis, keepdims=True)
    x = x - max_vals
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def Linear(x, W, b=0):
    b = W.shape[-1] if not b else b
    return x @ W + b  # (seq_len, out_dim)


def main():
    prompt = "hello"
    n_tokens_to_generate = 5

    model_config = ModelConfig()
    generator = Generator(model_config)
    print(generator(prompt, n_tokens_to_generate))


if __name__ == "__main__":
    main()
