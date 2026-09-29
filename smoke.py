"""smoke.py — tiny end-to-end run (2 min) before committing to a real train."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from cognicore.model import Config, CogniCore
from cognicore.data import load_corpus, to_byte_tokens, causal_lm_batch, synth_corpus
from cognicore.autograd import Adam, backward, zero_grads
from cognicore import generate as G
from cognicore.quant import quantize, report as qreport

print("=" * 74)
print("SMOKE TEST")
print("=" * 74)

# 1. param solver at the real budget
cfg = Config(target_params=10_000_000, seq_len=256)
m = CogniCore(cfg)
print(f"\n[1] 10M budget -> {cfg}")
print(f"    total params : {m.n_params():,}")
bd = m.n_params_breakdown()
print(f"    fixed        : {bd['fixed']:,}")
print(f"    per layer    : {bd['per_layer'][0]:,} x {cfg.n_layers}")
print(f"    HDE          : {bd['hde']} parameters")
err = abs(m.n_params() - 10_000_000) / 10_000_000
print(f"    budget error : {err * 100:.2f}%")

# 2. data
print("\n[2] data pipeline (offline synthetic, network path tested separately)")
txt = synth_corpus(400_000, log=lambda *a: None)
d = to_byte_tokens(txt, 128)
print(f"    {len(txt):,} chars -> {d.shape[0]:,} x {d.shape[1]} int array")
print(f"    dtype {d.dtype}, range [{d.min()}, {d.max()}]")

# 3. forward/backward on the REAL 10M model
print("\n[3] forward + backward on the real 10M model")
B, L = 4, 128
rows = d[:B]
ids, tgt = causal_lm_batch(rows)
t = time.time()
loss = m.forward(ids, tgt)
t_fwd = time.time() - t
t = time.time()
zero_grads(m.params())
backward(loss)
t_bwd = time.time() - t
gn = np.sqrt(sum(float((p.grad ** 2).sum()) for p in m.params() if p.grad is not None))
print(f"    loss {float(loss.data):.4f}  (ln257 = {np.log(257):.4f})")
print(f"    forward {t_fwd * 1000:.0f} ms   backward {t_bwd * 1000:.0f} ms")
print(f"    grad norm {gn:.3f}   finite: {np.isfinite(gn)}")
print(f"    RAM for weights+grads+adam: "
      f"{(m.n_params() * 4 * 4) / 1e6:.0f} MB")

# 4. a few training steps actually reduce the loss
print("\n[4] 40 training steps (does the loss go down?)")
opt = Adam(m.params(), lr=5e-3, weight_decay=0.02)
for i in range(40):
    r = d[np.random.default_rng(i).integers(0, len(d) - 1, B)]
    ii, tt = causal_lm_batch(r)
    zero_grads(m.params())
    l = m.forward(ii, tt)
    backward(l)
    opt.clip_(1.0)
    opt.step(5e-3)
    if i % 10 == 0 or i == 39:
        print(f"    step {i:>3}  loss {float(l.data):.4f}")

# 5. generation
print("\n[5] generation")
out = G.sample(m, prompt="def add(a, b):\n", n=90, temperature=0.7, top_k=15, seed=1,
               log=lambda *a: None)
print("    " + out.replace("\n", "\n    "))

# 6. quantisation
print("\n[6] quantisation")
qreport(quantize(m, bits=8))

# 7. throughput
print("\n[7] throughput")
t = time.time()
N = 8
for i in range(N):
    r = d[i * B:(i + 1) * B]
    ii, tt = causal_lm_batch(r)
    zero_grads(m.params())
    l = m.forward(ii, tt)
    backward(l)
dt = time.time() - t
print(f"    {N * B * 128 / dt / 1000:.1f} kB/s train  "
      f"({dt / N * 1000:.0f} ms per {B}x128 step)")

print("\nSMOKE TEST OK")
