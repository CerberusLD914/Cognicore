"""
train.py — training loop for CogniCore on CPU.

Trained in NumPy only. No CUDA, no cuDNN, no data loader workers, no
framework. Memory is a few hundred MB for a 10M-parameter model, which is the
entire point of the architecture.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .autograd import Adam, backward, zero_grads
from .data import load_corpus, to_byte_tokens, causal_lm_batch
from .model import Config, CogniCore


def lr_at(step, total, base_lr, warmup=0.05, floor=0.1):
    w = max(1, int(total * warmup))
    if step < w:
        return base_lr * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return base_lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def train(target_params=10_000_000,
          corpus_bytes=8_000_000,
          steps=2000,
          batch=8,
          seq_len=256,
          lr=3e-3,
          out="cognicore-10M",
          seed=0,
          allow_network=True,
          log=print,
          eval_every=200,
          resume=False):
    rng = np.random.default_rng(seed)
    ck = Path("checkpoints")
    ck.mkdir(exist_ok=True)

    # ---------------- model ----------------
    cfg = Config(target_params=target_params, seq_len=seq_len, seed=1234 + seed)
    model = CogniCore(cfg)
    meta = model.meta()
    log("=" * 74)
    log("CogniCore — Hyperdimensional Encoded Local-Cognitive-cell network")
    log("NOT an LLM: no attention, no KV cache, no embedding table")
    log("=" * 74)
    log(f"  {cfg}")
    log(f"  parameters        : {model.n_params():,}")
    log(f"  HDE (embedding)   : 0  parameters  (seeded, vocabulary-free)")
    log(f"  memory            : {cfg.n_mem} slots x {cfg.d_mem} dims "
        f"= {cfg.n_mem * cfg.d_mem * 4 / 1024:.1f} KB/layer, fixed size")
    log(f"  trainable tensors : {len(model.params())}")
    log("")

    # ---------------- data ----------------
    log("[data] acquiring corpus ...")
    text = load_corpus(corpus_bytes, log=log, allow_network=allow_network, seed=seed)
    data = to_byte_tokens(text, seq_len)
    log(f"[data] {len(text):,} chars -> {data.shape[0]:,} sequences of {seq_len}")
    log("")

    val_n = min(256, data.shape[0] // 20)
    val = data[:val_n]
    train_data = data[val_n:]

    params = model.params()
    opt = Adam(params, lr=lr, weight_decay=0.02)
    start = 0
    hist = []
    best = float("inf")

    ck_path = ck / f"{out}.npz"
    hist_path = ck / f"{out}.history.json"
    if resume and ck_path.exists():
        model.load(ck_path)
        if hist_path.exists():
            hist = json.loads(hist_path.read_text())
            best = min(hist, key=lambda h: h["val"])
        start = hist[-1]["step"] if hist else 0
        log(f"[ckpt] resumed from {ck_path} at step {start}")

    log(f"[train] {steps} steps, batch {batch}, seq {seq_len}, lr {lr}")
    log("-" * 74)
    t0 = time.time()
    running = []

    for step in range(start, steps):
        rows = train_data[rng.integers(0, len(train_data), batch)]
        ids, tgt = causal_lm_batch(rows)

        zero_grads(params)
        loss = model.forward(ids, tgt)
        backward(loss)

        gnorm = opt.clip_(1.0)
        cur_lr = lr_at(step, steps, lr)
        opt.step(cur_lr)
        running.append(float(loss.data))

        if step % 20 == 0 or step == steps - 1:
            avg = float(np.mean(running[-20:]))
            el = time.time() - t0
            done = step - start + 1
            tps = done * batch * seq_len / max(el, 1e-9)
            log(f"  step {step:>6}/{steps}  loss {avg:7.4f}  "
                f"bpc {avg / np.log(2):6.3f}  lr {cur_lr:.2e}  "
                f"|g| {gnorm:6.2f}  {tps / 1000:6.1f} kB/s  "
                f"({el / 60:.1f} min)")
            hist.append({"step": step, "train": avg, "lr": cur_lr})

        if (step + 1) % eval_every == 0 or step == steps - 1:
            vl = evaluate(model, val)
            hist[-1]["val"] = vl
            log(f"    >> val loss {vl:.4f}  val bpc {vl / np.log(2):.3f}")
            if vl < best:
                best = vl
                model.save(ck_path)
                (ck / f"{out}.meta.json").write_text(json.dumps(meta, indent=2))
                log(f"    >> saved {ck_path} (best val {best:.4f})")

    if not ck_path.exists():
        model.save(ck_path)
        (ck / f"{out}.meta.json").write_text(json.dumps(meta, indent=2))
    hist_path.write_text(json.dumps(hist, indent=2))

    log("-" * 74)
    log(f"[done] best val loss {best:.4f}  (random byte baseline = ln257 = 5.549)")
    log(f"[done] weights -> {ck_path}  ({ck_path.stat().st_size / 1e6:.1f} MB)")
    return model, meta, hist, best


@np.errstate(all="ignore")
def evaluate(model, val, batch=8):
    tot, n = 0.0, 0
    for i in range(0, len(val) - batch + 1, batch):
        ids, tgt = causal_lm_batch(val[i:i + batch])
        l = model.forward(ids, tgt)
        tot += float(l.data) * len(tgt)
        n += len(tgt)
    return tot / max(n, 1)
