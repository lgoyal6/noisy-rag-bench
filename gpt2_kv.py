"""GPT-2 small in pure NumPy with a quantized KV cache.

No PyTorch. Reads openai-community/gpt2 safetensors out of the HF cache.

The cache is genuinely quantized, not simulated: int8 is stored as int8 and int4
is bit-packed two nibbles to a byte, so the reported byte counts and the decode
latencies both include the real pack/unpack cost. Quantization is asymmetric
per (layer, head, token) over the 64 head-dim values, which is the standard
per-token KV scheme.
"""
import glob
import os

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

REPO = "openai-community/gpt2"
N_LAYER, N_HEAD, N_EMB = 12, 12, 768
HEAD = N_EMB // N_HEAD
# float32, not np.float64: under NEP 50 a float64 scalar silently promotes the whole
# forward pass to float64, which doubles the KV cache and halves throughput.
ATT_SCALE = np.float32(1.0 / np.sqrt(HEAD))


def snapshot():
    root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hits = glob.glob(os.path.join(root, "hub", "models--" + REPO.replace("/", "--"),
                                  "snapshots", "*", "model.safetensors"))
    if not hits:
        raise SystemExit(
            "GPT-2 not in the HF cache. Fetch the three files with:\n"
            "  python -c \"from huggingface_hub import hf_hub_download as d; "
            "[d('openai-community/gpt2', f) for f in "
            "['config.json','tokenizer.json','model.safetensors']]\"")
    return os.path.dirname(hits[0])


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(x.var(-1, keepdims=True) + eps) * w + b


# ------------------------------------------------------------ KV cache ------
class KVCache:
    """bits in {32, 16, 8, 4}. Stores (n_head, T, HEAD) per layer for K and V."""

    def __init__(self, bits, n_layer=N_LAYER):
        self.bits = bits
        self.k = [[] for _ in range(n_layer)]
        self.v = [[] for _ in range(n_layer)]

    # x: (n_head, t, HEAD) float32
    def _q(self, x):
        if self.bits == 32:
            return ("f", x)
        if self.bits == 16:
            return ("f", x.astype(np.float16))
        levels = (1 << self.bits) - 1
        lo = x.min(-1, keepdims=True)
        hi = x.max(-1, keepdims=True)
        scale = (hi - lo) / levels
        scale = np.where(scale == 0, 1e-8, scale)
        q = np.clip(np.rint((x - lo) / scale), 0, levels).astype(np.uint8)
        if self.bits == 4:
            assert q.shape[-1] % 2 == 0
            q = (q[..., 0::2] | (q[..., 1::2] << 4))     # real nibble packing
        return ("q", q, scale.astype(np.float32), lo.astype(np.float32))

    def _dq(self, rec):
        if rec[0] == "f":
            return rec[1].astype(np.float32, copy=False)
        _, q, scale, lo = rec
        if self.bits == 4:
            full = np.empty(q.shape[:-1] + (q.shape[-1] * 2,), dtype=np.uint8)
            full[..., 0::2] = q & 0x0F
            full[..., 1::2] = q >> 4
            q = full
        return q.astype(np.float32) * scale + lo

    def append(self, layer, k, v):
        self.k[layer].append(self._q(k))
        self.v[layer].append(self._q(v))

    def get(self, layer):
        return (np.concatenate([self._dq(r) for r in self.k[layer]], axis=1),
                np.concatenate([self._dq(r) for r in self.v[layer]], axis=1))

    def nbytes(self):
        tot = 0
        for side in (self.k, self.v):
            for layer in side:
                for rec in layer:
                    tot += sum(a.nbytes for a in rec[1:])
        return tot


# ------------------------------------------------------------ model --------
class GPT2:
    def __init__(self):
        snap = snapshot()
        self.tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
        with safe_open(os.path.join(snap, "model.safetensors"), "numpy") as f:
            # skip the baked causal mask buffers; we build the mask ourselves
            drop = {f"h.{i}.attn.{n}" for i in range(N_LAYER)
                    for n in ("bias", "masked_bias")}
            self.w = {k: f.get_tensor(k).astype(np.float32)
                      for k in f.keys() if k not in drop}

    def forward(self, ids, cache, past_len):
        """ids: (T,) int. Returns logits (T, vocab)."""
        w = self.w
        T = len(ids)
        x = w["wte.weight"][ids] + w["wpe.weight"][past_len:past_len + T]
        causal = np.triu(np.full((T, T), -1e9, dtype=np.float32), 1)
        for i in range(N_LAYER):
            p = f"h.{i}."
            h = ln(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
            qkv = h @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q, k, v = (t.reshape(T, N_HEAD, HEAD).transpose(1, 0, 2) for t in (q, k, v))
            cache.append(i, k, v)
            kk, vv = cache.get(i)
            att = q @ kk.transpose(0, 2, 1) * ATT_SCALE
            att[:, :, past_len:] += causal
            att -= att.max(-1, keepdims=True)
            np.exp(att, out=att)
            att /= att.sum(-1, keepdims=True)
            ctx = (att @ vv).transpose(1, 0, 2).reshape(T, N_EMB)
            x = x + ctx @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
            h = ln(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
            h = gelu(h @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"])
            x = x + h @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
        x = ln(x, w["ln_f.weight"], w["ln_f.bias"])
        return x @ w["wte.weight"].T


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))
