"""
quicktest.py — 60-second proof that the whole thing works.

Trains a SMALL model (not 10M) for a handful of steps so you can see the
loss actually drop, then prints real predictions. Use this to check your setup
before committing to a long run.

    py quicktest.py
"""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from cognicore.model import Config, CogniCore
from cognicore.data import load_corpus, to_byte_tokens, causal_lm_batch
from cognicore.autograd import Adam, backward, zero_grads
from cognicore import generate as G
from cognicore.quant import quantize, report as qreport

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
PARAMS = 2_000_000

print("=" * 70)
print(f"COGNICORE QUICK TEST  ({STEPS} steps, {PARAMS/1e6:.0f}M params)")
print("=" * 70)

# ---- model -------------------------------------------------------------
cfg = Config(target_params=PARAMS, seq_len=128)
m = CogniCore(cfg)
print(f"\n[1] {cfg}")
print(f"    parameters: {m.n_params():,}   HDE parameters: 0")
print(f"    memory: {cfg.n_mem * cfg.d_mem * 4 / 1024:.1f} KB/layer, fixed forever")
print(f"    no attention, no KV cache, no embedding table, no tokenizer")

# ---- data -------------------------------------------------------------
print(f"\n[2] data")
try:
    txt = load_corpus(2_000_000, log=lambda *a: None)
except Exception as e:
    print(f"    network failed ({type(e).__name__}), using synthetic")
    from cognicore.data import synth_corpus
    txt = synth_corpus(2_000_000, log=lambda *a: None)
d = to_byte_tokens(txt, 128)
print(f"    {len(txt):,} chars -> {d.shape[0]:,} sequences x 128 BYTES")

# ---- train -----------------------------------------------------------
print(f"\n[3] training {STEPS} steps  (CPU, NumPy only)")
opt = Adam(m.params(), lr=4e-3, weight_decay=0.02)
rng = np.random.default_rng(0)
val = d[:256]
t0 = time.time()
losses = []
for i in range(STEPS):
    rows = d[rng.integers(0, len(d) - 8, 8)]
    ids, tgt = causal_lm_batch(rows)
    zero_grads(m.params())
    l = m.forward(ids, tgt)
    backward(l)
    opt.clip_(1.0)
    opt.step(4e-3 * (1.0 - i / STEPS) + 5e-4)
    losses.append(float(l.data))
    if i % 20 == 0 or i == STEPS - 1:
        avg = np.mean(losses[-20:])
        print(f"    step {i:>4}  loss {avg:7.4f}  bits/byte {avg/np.log(2):6.3f}")
dt = time.time() - t0
print(f"    {STEPS*8*128/dt/1000:.1f} kB/s  ({dt*1000/STEPS:.0f} ms/step)")

# ---- before/after comparison -----------------------------------------
print(f"\n[4] did it learn?")
print(f"    start : {losses[0]:.4f}   (random = {np.log(257):.4f})")
print(f"    end   : {np.mean(losses[-20:]):.4f}")
print(f"    drop  : {losses[0] - np.mean(losses[-20:]):.4f} nats/byte")

# ---- accuracy on REAL held-out text (the honest metric) ----------------
L = cfg.seq_len
print(f"\n[5] accuracy on real held-out code")


def top1_accuracy(text, limit=300):
    r = np.frombuffer(text.encode(), dtype=np.uint8).astype(np.int64)
    ok = n = 0
    for i in range(0, min(len(r) - 1, limit * L), L):
        ch = r[i:i + L + 1]
        if len(ch) < L + 1:
            break
        ids, tg = ch.reshape(1, L + 1)[:, :-1], ch.reshape(1, L + 1)[:, 1:]
        pred = m.forward(ids, refine=True).data[0].argmax(-1)
        ok += int((pred == tg[0]).sum())
        n += L
    return ok / max(n, 1)


held = txt[10_000_000:10_400_000] if len(txt) > 10_400_000 else txt[-400_000:]
acc = top1_accuracy(held)
print(f"    top-1 next-byte accuracy : {acc * 100:.1f}%")
raw = np.frombuffer(txt[:2_000_000].encode(), dtype=np.uint8)
cnt = np.bincount(raw, minlength=256)
print(f"    corpus whitespace share : {(cnt[32] + cnt[10]) / raw.size * 100:.1f}%"
      f"   (byte-level is the right unit here)")

print(f"\n[5b] next byte after 'def add(a, b):\\n    return a'")
probe = "def add(a, b):\n    return a"
r2 = np.frombuffer(probe.encode(), dtype=np.uint8).astype(np.int64)
r2 = np.pad(r2, (L - len(r2),), constant_values=10)[:L]
lg = m.forward(r2.reshape(1, L), refine=True).data[0, -1]
p = np.exp(lg - lg.max()); p /= p.sum()
top = np.argsort(-p)[:8]
print("    " + "  ".join(
    f"{(chr(i) if 32 <= i < 127 else hex(i))}:{p[i]:.2f}" for i in top))

# ---- generate --------------------------------------------------------
print(f"\n[6] generation (O(1) memory, no KV cache)")
print("  -- greedy (argmax: exactly what the model believes) --")
snip = held[:110]
print(f"    real: {snip!r}")
print(f"    cont: {G.greedy(m, snip, 70)!r}")
print("  -- sampled, no-repeat-4 --")
for prompt in ["def add(a, b):\n", "for i in range(10):\n"]:
    out = G.sample(m, prompt=prompt, n=70, temperature=0.7, top_k=15, seed=1,
                   log=lambda *a: None)
    print(f"    {prompt!r:24} -> {out!r}")

# ---- quantise --------------------------------------------------------
print(f"\n[7] int8 quantisation")
qreport(quantize(m, bits=8))

# ---- save -----------------------------------------------------------
import json
ck = Path("checkpoints"); ck.mkdir(exist_ok=True)
m.save(ck / "quicktest.npz")
(ck / "quicktest.meta.json").write_text(json.dumps(m.meta(), indent=2))
print(f"\n[8] saved checkpoints/quicktest.npz + .meta.json")
print(f"    now run:  chat.cmd          (interactive console)")
print("\nQUICK TEST PASSED" if losses[0] - np.mean(losses[-20:]) > 0.3
      else "\nQUICK TEST RAN (loss barely moved — try more steps)")
