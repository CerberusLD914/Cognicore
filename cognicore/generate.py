"""
generate.py — byte-level sampling with NO KV cache.

Because the architecture is purely recurrent, the per-token state is a fixed
(B, d_model) tensor plus 12 fixed-size hash banks. Generating 100k bytes
costs exactly the same memory as generating 100 bytes — an LLM's KV cache
grows to 100k * 24 * 2 * d floats by the end.

Generation is done in fixed-size windows purely for batching convenience;
the model itself never sees more than its recurrent state.
"""

from __future__ import annotations

import numpy as np

from .autograd import Tensor, rmsnorm, silu, swiglu, cross_entropy
from .hde import PAD
from .model import Config, CogniCore, local_context
from .ssm import selective_ssm

NL = 10          # left-pad byte for the generation window (see sample)


def greedy(model, prompt="", n=200, log=print, no_repeat=4):
    """Deterministic continuation — argmax at every step, with the same
    no-repeat-ngram guard as sample(). Shows exactly what the model believes.
    """
    chunk = model.cfg.seq_len
    pre = np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8).astype(np.int64)
    if len(pre) > chunk - 2:
        pre = pre[-(chunk - 2):]
    win = (np.tile(pre, (1, 1)).astype(np.int64) if len(pre)
           else np.full((1, 1), NL, dtype=np.int64))
    out = []
    for _ in range(n):
        lg = model.forward(win, refine=True).data[0, -1].copy()
        b = _banned_continuations(out, no_repeat)
        if b:
            lg[list(b)] = -1e9
        nxt = int(np.argmax(lg))
        out.append(nxt)
        win = np.concatenate([win, np.array([[nxt]], dtype=np.int64)], axis=1)
        if win.shape[1] > chunk:
            win = win[:, -chunk:]
    return bytes(x if x < 256 else NL for x in out).decode("utf-8", "replace")


def _banned_continuations(out, n=4):
    """no-repeat-ngram: which bytes would immediately re-create an n-gram we
    have already used recently.

    Small recurrent models fall into repetition loops easily — the state
    saturates and the argmax locks onto a cycle. Blocking the immediate
    re-occurrence of a recent n-gram is the standard, and cheapest, fix.
    """
    if n <= 0 or len(out) < n:
        return ()
    tail = tuple(out[-(n - 1):])
    banned = set()
    for i in range(len(out) - (n - 1), -1, -1):
        if tuple(out[i:i + n - 1]) == tail and i + n - 1 < len(out):
            banned.add(out[i + n - 1])
    return tuple(banned)


def sample(model, prompt="", n=400, temperature=0.85, top_k=40, seed=0,
           batch=1, log=print, chunk=None, no_repeat=4, rep_penalty=1.15):
    """Sample n bytes.  prompt is raw text.

    Memory is O(1) in the generated length: the recurrent state is a fixed
    (B, d_model) tensor plus a handful of fixed-size hash banks. An LLM's KV
    cache would grow to n * n_layers * 2 * d_model floats by the end.
    """
    rng = np.random.default_rng(seed)
    chunk = chunk or model.cfg.seq_len

    pre = np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8).astype(np.int64)
    if len(pre) > chunk - 2:
        pre = pre[-(chunk - 2):]

    # The window contains ONLY bytes the model can legitimately be in.
    # Left-padding with PAD or newline would feed the recurrent state a
    # sequence it never saw in training (a 128-byte run of newlines), and
    # because the SSM is causal that pollution reaches every later position.
    # So: seed the window with the real prompt and let it grow, sliding only
    # once it is full of genuine content.
    if len(pre):
        win = np.tile(pre, (batch, 1)).astype(np.int64)
    else:
        win = np.full((batch, 1), NL, dtype=np.int64)   # single real byte

    out = [[] for _ in range(batch)]
    produced = 0
    while produced < n:
        logits = model.forward(win, refine=True)
        lg = logits.data[:, -1, :]                      # (B, vocab)
        lg = lg / max(temperature, 1e-3)
        if top_k and top_k < lg.shape[-1]:
            kth = np.partition(lg, -top_k, axis=-1)[:, -top_k][:, None]
            lg = np.where(lg < kth, -1e9, lg)
        if no_repeat:
            for i in range(batch):
                b = _banned_continuations(out[i], no_repeat)
                if b:
                    lg[i, list(b)] = -1e9
        if rep_penalty and out[0]:
            # standard repetition penalty: damp logits of bytes that have
            # already been produced a lot. Without it a small recurrent model
            # saturates its state and locks onto a cycle.
            cnt = np.bincount(np.array(out[0], dtype=np.int64),
                              minlength=lg.shape[-1]).astype(np.float32)
            lg = np.where(cnt > 0, lg / np.power(rep_penalty, cnt), lg)
        p = np.exp(lg - lg.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        nxt = np.array([rng.choice(p.shape[-1], p=p[i] / p[i].sum()) for i in range(batch)])

        for i in range(batch):
            if produced + len(out[i]) < n:
                out[i].append(int(nxt[i]))
        produced += batch

        win = np.concatenate([win, nxt.reshape(batch, 1)], axis=1)
        if win.shape[1] > chunk:                    # slide, keeping real bytes
            win = win[:, -chunk:]
        if produced % 80 == 0:
            log(f"  ... {min(produced, n)}/{n} bytes")

    res = []
    for i in range(batch):
        b = bytes(x if x < 256 else 10 for x in out[i][:n])
        res.append(b.decode("utf-8", "replace"))
    return res[0] if batch == 1 else res


def perplexity(model, text, seq_len=None, batch=8):
    """bits-per-byte on held-out text — the metric that matters for raw bytes."""
    L = seq_len or model.cfg.seq_len
    raw = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8).astype(np.int64)
    n = (len(raw) - 1) // L
    if n < 1:
        return float("nan")
    tot, cnt = 0.0, 0
    for i in range(n):
        chunk = raw[i * L: i * L + L + 1].reshape(1, L + 1)
        ids, tgt = chunk[:, :-1], chunk[:, 1:]
        l = model.forward(ids, tgt)
        tot += float(l.data) * L
        cnt += L
    return tot / cnt / np.log(2)


def top_predictions(model, text, k=10):
    """What does the model think comes next at each position?"""
    raw = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8).astype(np.int64)
    L = model.cfg.seq_len
    raw = raw[:L] if len(raw) > L else np.pad(raw, (L - len(raw),), constant_values=10)
    logits = model.forward(raw.reshape(1, L), refine=True).data[0]
    out = []
    for t in range(L - 1, L):
        p = np.exp(logits[t] - logits[t].max())
        p /= p.sum()
        order = np.argsort(-p)[:k]
        out.append({chr(int(i)) if int(i) < 256 else "PAD": float(p[i]) for i in order})
    return out
