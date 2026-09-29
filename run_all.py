"""
run_all.py — ONE COMMAND does everything.

    py run_all.py

  1. verifies every gradient kernel against a float64 reference
  2. downloads the training corpus automatically (HF -> GitHub -> synthetic)
  3. builds a 10M-parameter CogniCore (width/depth auto-solved to the budget)
  4. trains it on CPU
  5. reports bits-per-byte, samples code, and quantises to int8

No arguments required. Override anything with --flags.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np                                    # noqa: E402
from cognicore.model import Config, CogniCore        # noqa: E402
from cognicore import train as T                      # noqa: E402
from cognicore import generate as G                   # noqa: E402
from cognicore.quant import quantize, dequantize, report as qreport  # noqa: E402


def banner(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


def env_report():
    banner("0. ENVIRONMENT")
    import numpy
    print(f"  python     {platform.python_version()}  ({platform.machine()})")
    print(f"  numpy      {numpy.__version__}")
    print(f"  platform   {platform.system()} {platform.release()}")
    try:
        import psutil
        m = psutil.virtual_memory()
        print(f"  RAM        {m.total / 1e9:.1f} GB total, {m.available / 1e9:.1f} GB free")
    except ImportError:
        print("  RAM        (psutil not installed, skipping)")
    print("  GPU        not used — this trains on CPU by design")


def step_verify():
    banner("1. GRADIENT VERIFICATION (float64 reference)")
    r = subprocess.run([sys.executable, "gradcheck.py"],
                       cwd=str(Path(__file__).parent))
    if r.returncode != 0:
        print("\n!! gradient checks FAILED — refusing to train on a broken graph")
        sys.exit(1)
    print("\n  all kernels verified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=int, default=10_000_000,
                    help="parameter budget (default 10M)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--corpus-mb", type=int, default=8)
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    name = a.name or f"cognicore-{a.params // 1_000_000}M"
    t0 = time.time()
    env_report()
    if not a.skip_verify:
        step_verify()

    banner("2-4. BUILD + DATA + TRAIN")
    model, meta, hist, best = T.train(
        target_params=a.params,
        corpus_bytes=a.corpus_mb * 1_000_000,
        steps=a.steps, batch=a.batch, seq_len=a.seq, lr=a.lr,
        out=name, seed=a.seed, allow_network=not a.no_network,
        resume=a.resume)

    banner("5. EVALUATION")
    print(f"  parameters          : {model.n_params():,}")
    print(f"  best val loss       : {best:.4f}")
    print(f"  best val bits/byte  : {best / np.log(2):.4f}   "
          f"(uniform random over 256 bytes = 8.0000)")
    print(f"  compression         : "
          f"{np.log2(256) / (best / np.log(2)):.2f}x better than uniform")

    corpus = Path("data/corpus.txt")
    if corpus.exists():
        held = corpus.read_text(encoding="utf-8", errors="ignore")[:200000]
        bpb = G.perplexity(model, held)
        print(f"  bits/byte (corpus)  : {bpb:.4f}")

    print("\n  --- next-byte predictions on real code ---")
    probe = "def fibonacci(n):\n    if n < 2:\n        return n\n"
    for row in G.top_predictions(model, probe):
        print("    " + "  ".join(f"{repr(k)}:{v:.2f}" for k, v in row.items()))

    print("\n  --- sampling (greedy-ish, temp 0.7) ---")
    out = G.sample(model, prompt="def add(a, b):\n", n=220,
                   temperature=0.7, top_k=20, seed=a.seed)
    print("    " + out.replace("\n", "\n    "))

    banner("6. INT8 QUANTISATION (low-resource deployment)")
    rep = quantize(model, bits=8, out=Path("checkpoints") / f"{name}.int8.npz")
    qreport(rep)

    banner("DONE")
    print(f"  total wall clock : {(time.time() - t0) / 60:.1f} min")
    print(f"  weights          : checkpoints/{name}.npz")
    print(f"  run it again     : py run_all.py --resume")
    print()


if __name__ == "__main__":
    main()
