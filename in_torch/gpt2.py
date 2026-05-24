import numpy as np
import torch


def softmax(x):
    max_v = torch.max(x, dim=-1, keepdim=True).values
    x -= max_v
    return torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)


class LayerNorm:
    def __init__(self, g, b, eps=1e-5):
        self.g = g
        self.b = b
        self.eps = eps

    def __call__(self, X):
        return (
            self.g
            * (X - torch.mean(X, dim=-1, keepdim=True))
            / torch.sqrt(torch.var(X, dim=-1, correction=0, keepdim=True) + self.eps)
            + self.b
        )


def self_attention(Q, K, V, mask):
    scores = Q @ torch.transpose(K, dim0=2, dim1=1) / (K.shape[-1] ** 0.5)
    scores += mask
    return softmax(scores) @ V


def gelu(x):
    return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x**3)))


class Linear:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, x):
        return x @ self.w + self.b


class Embedding:
    def __init__(self, wte, wpe):
        self.wte = wte
        self.wpe = wpe

    def __call__(self, X):
        positions = torch.arange(X.shape[0])
        pos_encoding = self.wpe[positions]
        return self.wte[X] + pos_encoding


class FFNN:
    def __init__(self, w1, w2, b1, b2):
        self.linear1 = Linear(w1, b1)
        self.linear2 = Linear(w2, b2)

    def __call__(self, x):
        return self.linear2(gelu(self.linear1(x)))


class MHSA:
    def __init__(
        self,
        n_heads,
        n_ctx,
        attn_c_attn_w,
        attn_c_attn_b,
        attn_c_proj_w,
        attn_c_proj_b,
    ):
        self.n_heads = n_heads
        self.n_ctx = n_ctx
        self.attn_c_attn_w = attn_c_attn_w
        self.attn_c_attn_b = attn_c_attn_b
        self.attn_c_proj_w = attn_c_proj_w
        self.attn_c_proj_b = attn_c_proj_b

    def construct_qkv(self, X):
        qkv = X @ self.attn_c_attn_w + self.attn_c_attn_b
        Q, K, V = torch.split(qkv, qkv.shape[-1] // 3, dim=-1)
        return Q, K, V

    def __call__(self, X):
        Q, K, V = self.construct_qkv(X)
        seq_len, d_model = K.shape  # assuming K.ndim == 2 here
        assert d_model % self.n_heads == 0
        d_k = d_model // self.n_heads

        mask = torch.triu(torch.ones((seq_len, seq_len)) * (-torch.inf), diagonal=1)

        Q = Q.reshape(seq_len, self.n_heads, d_k).permute(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, d_k).permute(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, d_k).permute(1, 0, 2)

        out = self_attention(Q, K, V, mask)
        catted_back = torch.transpose(out, dim0=1, dim1=0).reshape(seq_len, d_model)

        return catted_back @ self.attn_c_proj_w + self.attn_c_proj_b


class TransformerBlock:
    def __init__(
        self,
        n_heads,
        n_ctx,
        attn_c_attn_w,
        attn_c_attn_b,
        attn_c_proj_w,
        attn_c_proj_b,
        mlp_c_fc_w,
        mlp_c_proj_w,
        mlp_c_fc_b,
        mlp_c_proj_b,
        ln_1_g,
        ln_2_g,
        ln_1_b,
        ln_2_b,
    ):
        self.LayerNorm1 = LayerNorm(ln_1_g, ln_1_b)
        self.LayerNorm2 = LayerNorm(ln_2_g, ln_2_b)

        self.MHA = MHSA(
            n_heads, n_ctx, attn_c_attn_w, attn_c_attn_b, attn_c_proj_w, attn_c_proj_b
        )
        self.FFNN = FFNN(mlp_c_fc_w, mlp_c_proj_w, mlp_c_fc_b, mlp_c_proj_b)

    def __call__(self, X):
        x1 = X + self.MHA(self.LayerNorm1(X))
        x2 = x1 + self.FFNN(self.LayerNorm2(x1))
        return x2


class Decoder:
    def __init__(self, hparams, params):
        n_ctx = hparams["n_ctx"]
        n_heads = hparams["n_head"]

        self.wte = params["wte"]
        wpe = params["wpe"]

        block = params["blocks"][0]

        mlp = block["mlp"]
        attn = block["attn"]

        mlp_c_fc_w = mlp["c_fc"]["w"]
        mlp_c_fc_b = mlp["c_fc"]["b"]
        mlp_c_proj_w = mlp["c_proj"]["w"]
        mlp_c_proj_b = mlp["c_proj"]["b"]

        attn_c_attn_w = attn["c_attn"]["w"]
        attn_c_attn_b = attn["c_attn"]["b"]
        attn_c_proj_w = attn["c_proj"]["w"]
        attn_c_proj_b = attn["c_proj"]["b"]

        ln_1_g = block["ln_1"]["g"]
        ln_1_b = block["ln_1"]["b"]
        ln_2_g = block["ln_2"]["g"]
        ln_2_b = block["ln_2"]["b"]

        ln_f_g = params["ln_f"]["g"]
        ln_f_b = params["ln_f"]["b"]

        self.embedding = Embedding(self.wte, wpe)
        self.t_block = TransformerBlock(
            n_heads,
            n_ctx,
            attn_c_attn_w,
            attn_c_attn_b,
            attn_c_proj_w,
            attn_c_proj_b,
            mlp_c_fc_w,
            mlp_c_proj_w,
            mlp_c_fc_b,
            mlp_c_proj_b,
            ln_1_g,
            ln_2_g,
            ln_1_b,
            ln_2_b,
        )
        self.f_layer_norm = LayerNorm(ln_f_g, ln_f_b)

    def __call__(self, X):
        embedded_itorchut = self.embedding(X)
        t_block_out = self.t_block(embedded_itorchut)
        final_normalized = self.f_layer_norm(t_block_out)
        final_linear = final_normalized @ self.wte.T
        probs = softmax(final_linear)
        next_token = torch.argmax(probs, dim=-1)[-1]
        return next_token


def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    encoder, hparams, params = load_encoder_hparams_and_params()
    encoded_itorchut = encoder.encode(prompt)
    itorchut_prompt_len = len(encoded_itorchut)
    decoder = Decoder(hparams, params)
    for _ in range(n_tokens_to_generate):
        next_token = decoder(torch.tensor(encoded_itorchut))
        encoded_itorchut.append(next_token.item())
    return encoder.decode(encoded_itorchut[itorchut_prompt_len:])


def load_encoder_hparams_and_params(
    model_size: str = "124M", models_dir: str = "models"
):
    class DummyBPE:
        def __init__(self):
            self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

        def encode(self, text: str):
            tokens = text.strip().split()
            return [
                self.encoder_dict.get(token, self.encoder_dict["<UNK>"])
                for token in tokens
            ]

        def decode(self, token_ids: list):
            reversed_dict = {v: k for k, v in self.encoder_dict.items()}
            return " ".join(
                [reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids]
            )

    hparams = {"n_ctx": 1024, "n_head": 2}

    params = {
        "wte": torch.rand(3, 10),  # (1, 10)
        "wpe": torch.rand(1024, 10),
        "blocks": [
            {
                "mlp": {
                    "c_fc": {"w": torch.rand(10, 20), "b": torch.rand(20)},
                    "c_proj": {"w": torch.rand(20, 10), "b": torch.rand(10)},
                },
                "attn": {
                    "c_attn": {"w": torch.rand(10, 30), "b": torch.rand(30)},
                    "c_proj": {"w": torch.rand(10, 10), "b": torch.rand(10)},
                },
                "ln_1": {"g": torch.ones(10), "b": torch.zeros(10)},
                "ln_2": {"g": torch.ones(10), "b": torch.zeros(10)},
            }
        ],
        "ln_f": {
            "g": torch.ones(10),
            "b": torch.zeros(10),
        },
    }

    encoder = DummyBPE()
    return encoder, hparams, params


if __name__ == "__main__":
    np.random.seed(42)
    print(gen_text("hello", n_tokens_to_generate=5))
