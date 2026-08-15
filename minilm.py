"""sentence-transformers/all-MiniLM-L6-v2 forward pass in pure NumPy.

No PyTorch, no transformers, no sentence-transformers. Reads the safetensors
checkpoint straight out of the HF cache. ~90 MB of weights, ~200 MB peak RSS,
which is the whole point: it is the dense retriever you can actually put on a
box inside somebody's perimeter.
"""
import glob
import os

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

REPO = "sentence-transformers/all-MiniLM-L6-v2"
N_LAYERS, N_HEADS, HID = 6, 12, 384
HEAD = HID // N_HEADS
# float32, not np.float64: under NEP 50 a float64 scalar silently promotes the whole
# forward pass to float64, doubling peak RSS for no accuracy gain.
ATT_SCALE = np.float32(1.0 / np.sqrt(HEAD))


def _snapshot():
    root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pat = os.path.join(root, "hub", "models--" + REPO.replace("/", "--"),
                       "snapshots", "*", "model.safetensors")
    hits = glob.glob(pat)
    if not hits:
        raise SystemExit(
            f"MiniLM not in the HF cache. Fetch it with:\n"
            f"  python -c \"from huggingface_hub import snapshot_download as d; "
            f"d('{REPO}')\"")
    return os.path.dirname(hits[0])


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _ln(x, w, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


class MiniLM:
    def __init__(self):
        snap = _snapshot()
        self.tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
        self.tok.enable_truncation(256)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        with safe_open(os.path.join(snap, "model.safetensors"), "numpy") as f:
            self.w = {k: f.get_tensor(k).astype(np.float32) for k in f.keys()
                      if k != "embeddings.position_ids"}

    def _layer(self, x, mask, i):
        w = self.w
        p = f"encoder.layer.{i}."
        B, T, _ = x.shape
        q = x @ w[p + "attention.self.query.weight"].T + w[p + "attention.self.query.bias"]
        k = x @ w[p + "attention.self.key.weight"].T + w[p + "attention.self.key.bias"]
        v = x @ w[p + "attention.self.value.weight"].T + w[p + "attention.self.value.bias"]
        # (B, H, T, D)
        q, k, v = (t.reshape(B, T, N_HEADS, HEAD).transpose(0, 2, 1, 3) for t in (q, k, v))
        att = q @ k.transpose(0, 1, 3, 2) * ATT_SCALE
        att = att + mask[:, None, None, :]
        att -= att.max(-1, keepdims=True)
        np.exp(att, out=att)
        att /= att.sum(-1, keepdims=True)
        ctx = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, HID)
        ctx = ctx @ w[p + "attention.output.dense.weight"].T + w[p + "attention.output.dense.bias"]
        x = _ln(x + ctx, w[p + "attention.output.LayerNorm.weight"],
                w[p + "attention.output.LayerNorm.bias"])
        h = _gelu(x @ w[p + "intermediate.dense.weight"].T + w[p + "intermediate.dense.bias"])
        h = h @ w[p + "output.dense.weight"].T + w[p + "output.dense.bias"]
        return _ln(x + h, w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"])

    def encode(self, texts, batch_size=8):
        """Mean-pooled, L2-normalized sentence embeddings. (N, 384) float32."""
        w = self.w
        out = []
        for s in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[s:s + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            am = np.array([e.attention_mask for e in enc], dtype=np.float32)
            T = ids.shape[1]
            x = (w["embeddings.word_embeddings.weight"][ids]
                 + w["embeddings.position_embeddings.weight"][None, :T]
                 + w["embeddings.token_type_embeddings.weight"][0][None, None])
            x = _ln(x, w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"])
            mask = (1.0 - am) * -1e9
            for i in range(N_LAYERS):
                x = self._layer(x, mask, i)
            pooled = (x * am[..., None]).sum(1) / np.maximum(am.sum(1, keepdims=True), 1e-9)
            out.append(pooled)
        e = np.concatenate(out, 0)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


if __name__ == "__main__":
    m = MiniLM()
    v = m.encode(["a covenant breach", "a breach of covenant", "the price of tea"])
    print("shape", v.shape, "norms", np.linalg.norm(v, axis=1).round(4))
    print("sim(0,1)", float(v[0] @ v[1]), " sim(0,2)", float(v[0] @ v[2]))
