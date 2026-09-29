"""
live.py — continuous training with a live monitor.

Runs until you press Ctrl+C, printing the full picture every few seconds:
loss, bits/byte, validation loss, gradient norm, learning rate, throughput,
tokens seen, ETA, and a verdict on whether it is still improving or has
stalled/started overfitting.

A log of every evaluation is appended to checkpoints/<name>.log so you can
compare runs afterwards.

    py live.py                     # 10M params, the default
    py live.py --params 50000000
    py live.py --every 5 --eval-every 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import numpy as np

from cognicore.autograd import Adam, backward, zero_grads
from cognicore.data import load_corpus, to_byte_tokens, causal_lm_batch
from cognicore.model import Config, CogniCore


def bar(frac, width=28, ch="█", empty="·"):
    frac = max(0.0, min(1.0, frac))
    n = int(frac * width)
    return ch * n + empty * (width - n)


def hms(sec):
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def smooth(v, k=20):
    if not v:
        return 0.0
    w = v[-k:]
    return float(np.mean(w))


def trend(recent, older):
    """+1 improving (loss falling), -1 getting worse, 0 flat."""
    a, b = smooth(recent, 20), smooth(older, 20)
    if b <= 1e-9:
        return 0
    d = (a - b) / max(abs(b), 1e-6)
    if d < -0.02:
        return 1
    if d > 0.02:
        return -1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=int, default=10_000_000)
    ap.add_argument("--steps", type=int, default=0,
                    help="0 = run until you press Ctrl+C")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--corpus-mb", type=int, default=8)
    ap.add_argument("--every", type=int, default=25,
                    help="log every N steps")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    name = a.name or f"cognicore-{a.params // 1_000_000}M"
    ck = Path("checkpoints"); ck.mkdir(exist_ok=True)
    logf = open(ck / f"{name}.log", "a", encoding="utf-8")

    def both(*args):
        s = " ".join(str(x) for x in args)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    W = 78
    both("=" * W)
    both(f"  COGNICORE  -  continuous training on CPU   (Ctrl+C to stop)")
    both("=" * W)

    # ---------------- model ----------------
    cfg = Config(target_params=a.params, seq_len=a.seq, seed=1234 + a.seed)
    model = CogniCore(cfg)
    bd = model.n_params_breakdown()
    both(f"  architecture   CogniCore / HDS-LSM   NOT an LLM")
    both(f"  model          {cfg}")
    both(f"  parameters     {model.n_params():,}   "
         f"(target {a.params:,}, error "
         f"{abs(model.n_params()-a.params)/a.params*100:.2f}%)")
    both(f"                  fixed {bd['fixed']:,} + "
         f"{cfg.n_layers} x {bd['per_layer'][0]:,}")
    both(f"  HDE embedding  0 parameters  (seeded hyperdimensional basis, "
         f"vocabulary-free)")
    both(f"  memory         {cfg.n_mem*cfg.d_mem*4/1024:.1f} KB/layer, fixed  |  "
         f"RAM for w+g+adam ~{model.n_params()*16/1e6:.0f} MB")
    both(f"  attention      none   |   KV cache: none   |   tokenizer: none")
    both(f"  working on     raw bytes 0-255, one byte per token")
    both("")

    # ---------------- data ----------------
    both("  downloading / loading corpus ...")
    text = load_corpus(a.corpus_mb * 1_000_000, log=both,
                       allow_network=not a.no_network, seed=a.seed)
    data = to_byte_tokens(text, a.seq)
    n_val = max(32, min(256, data.shape[0] // 20))
    val, tr = data[:n_val], data[n_val:]
    both(f"  corpus         {len(text):,} chars")
    both(f"  sequences      {data.shape[0]:,} total  ->  "
         f"{tr.shape[0]:,} train / {n_val} validation")
    both(f"  batch          {a.batch} x {a.seq} bytes = {a.batch*a.seq:,} bytes/step")
    both("")

    # ---------------- schedule ----------------
    # With steps=0 we run forever, so decay on a long horizon instead: this
    # keeps the LR useful long-term rather than annealing to zero at 3000.
    horizon = a.steps if a.steps > 0 else 20000
    warm = max(50, int(0.03 * horizon))

    def lr_at(step):
        if step < warm:
            return a.lr * (step + 1) / warm
        p = (step - warm) / max(1, horizon - warm)
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(p, 1.0))))

    params = model.params()
    opt = Adam(params, lr=a.lr, weight_decay=0.02)
    rng = np.random.default_rng(a.seed)

    best = float("inf")
    best_step = 0
    hist, vhist = [], []
    step = 0
    t0 = time.time()
    loss_win = []
    stopped = False

    both(f"  {'step':>7} {'train':>8} {'bpb':>7} {'val':>8} {'best':>8} "
        f"{'lr':>9} {'|g|':>7} {'kB/s':>7} {'MB seen':>8}  status")
    both("  " + "-" * (W - 4))

    try:
        while a.steps == 0 or step < a.steps:
            rows = tr[rng.integers(0, len(tr), a.batch)]
            ids, tgt = causal_lm_batch(rows)

            zero_grads(params)
            loss = model.forward(ids, tgt)
            backward(loss)
            gnorm = opt.clip_(1.0)
            opt.step(lr_at(step))
            loss_win.append(float(loss.data))
            step += 1

            if step % a.every and step != 1:
                continue

            el = time.time() - t0
            kbs = step * a.batch * a.seq / max(el, 1e-9) / 1000
            seen = step * a.batch * a.seq / 1e6
            tr_loss = smooth(loss_win, a.every)
            bpb = tr_loss / np.log(2)

            do_eval = (step % a.eval_every == 0)
            vloss = None
            if do_eval:
                vloss = float(np.mean([
                    float(model.forward(*causal_lm_batch(val[i:i + a.batch])).data)
                    for i in range(0, len(val) - a.batch + 1, a.batch)]))
                vhist.append((step, vloss, tr_loss))
                if vloss < best:
                    best, best_step = vloss, step
                    model.save(ck / f"{name}.npz")
                    (ck / f"{name}.meta.json").write_text(
                        json.dumps(model.meta(), indent=2))

            tr_d = trend(loss_win, loss_win[:-a.every]) if len(loss_win) > 2 * a.every else 0
            status = {1: "improving", -1: "RISING", 0: "flat"}[tr_d]
            if do_eval:
                if vloss < best - 1e-9:
                    status += "  <- new best, saved"
                elif step - best_step > a.eval_every * 3:
                    status += "  <- no val gain for a while"
                if vloss > tr_loss + 0.25:
                    status += "  [overfitting]"

            vs = f"{vloss:8.4f}" if vloss is not None else " " * 8
            both(f"  {step:>7} {tr_loss:8.4f} {bpb:7.3f} {vs} {best:8.4f} "
                 f"{lr_at(step):9.2e} {gnorm:7.2f} {kbs:7.2f} {seen:8.1f}  {status}")

            if do_eval:
                both(f"          progress {bar(min(step/horizon,1.0))} "
                    f"{min(step,horizon)}/{horizon} steps   "
                    f"elapsed {hms(el)}   ~{hms(el/step*(horizon-step))} left")
    except KeyboardInterrupt:
        stopped = True
        both("")
        both("  Ctrl+C - finishing the current step and saving ...")

    # ---------------- final ----------------
    el = time.time() - t0
    model.save(ck / f"{name}.npz")
    (ck / f"{name}.meta.json").write_text(json.dumps(model.meta(), indent=2))

    both("  " + "-" * (W - 4))
    both(f"  STOPPED after {step:,} steps in {hms(el)}")
    if loss_win:
        both(f"  final train loss   {smooth(loss_win, a.every):.4f} "
             f"({smooth(loss_win, a.every)/np.log(2):.3f} bits/byte)")
    both(f"  best val loss      {best:.4f} at step {best_step:,}")
    both(f"  compression        {np.log2(256)/(best/np.log(2)):.1f}x "
         f"better than uniform random bytes")
    if vhist:
        both("")
        both("  validation history (step / val / train):")
        for s, v, t in vhist[-14:]:
            mark = " *" if s == best_step else "  "
            both(f"    {s:>7,}  {v:7.4f}  {t:7.4f}{mark}")
    both("")
    both(f"  weights  ->  checkpoints/{name}.npz")
    both(f"  history  ->  checkpoints/{name}.log")
    both(f"  talk to it:  chat.cmd  checkpoints/{name}.npz")
    both("")
    logf.close()


if __name__ == "__main__":
    main()
