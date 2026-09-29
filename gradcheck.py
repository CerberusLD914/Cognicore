"""
gradcheck.py — numerical verification of every custom kernel.

If the hand-written backward passes are wrong, training silently produces a
model that memorises noise. This is the test that must pass first.
"""

import numpy as np

from cognicore.autograd import (Tensor, param, rmsnorm, silu, swiglu, softplus,
                                index_select, scatter_add, _transpose,
                                cross_entropy)
from cognicore.ssm import selective_ssm

rng = np.random.default_rng(0)


def num_grad(f, x, eps=1e-3):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        a = f()
        x[i] = old - eps
        b = f()
        x[i] = old
        g[i] = (a - b) / (2 * eps)
        it.iternext()
    return g


def check(name, make_inputs, fwd, loss_of, tol=2e-2):
    from cognicore.autograd import backward
    ins = make_inputs()
    # clear stale gradients — several tests share leaf Tensors
    for t in ins.values():
        if isinstance(t, Tensor):
            t.grad = None
    outs = fwd(ins)
    loss = loss_of(outs, ins)
    backward(loss)
    ok = True
    for k, t in ins.items():
        if not isinstance(t, Tensor) or not t.requires_grad or t.grad is None:
            continue
        ana = t.grad.copy()
        num = num_grad(lambda: loss_of(fwd(ins), ins).data, t.data)
        denom = max(1e-4, np.abs(ana).max(), np.abs(num).max())
        rel = np.abs(ana - num).max() / denom
        flag = "OK " if rel < tol else "FAIL"
        if rel >= tol:
            ok = False
        print(f"  [{flag}] {name}.d{k:<14} rel_err={rel:.2e}  "
              f"|ana|={np.abs(ana).max():.4f} |num|={np.abs(num).max():.4f}")
    return ok


def scalar_loss(t, key="s"):
    return t.sum()


# ---------------------------------------------------------------- tests
def test_ssm():
    print("\n[1] selective_ssm  (the core recurrence)")
    B, T, D = 2, 7, 5
    ins = {
        "u": Tensor(rng.standard_normal((B, T, D)).astype(np.float32), requires_grad=True),
        "delta": Tensor(rng.standard_normal((B, T, D)).astype(np.float32) * 0.5 - 2, requires_grad=True),
        "z": Tensor(rng.standard_normal((B, T, D)).astype(np.float32) * 0.3, requires_grad=True),
        "A": Tensor(-np.exp(rng.standard_normal(D)).astype(np.float32), requires_grad=True),
        "C": Tensor(rng.standard_normal(D).astype(np.float32), requires_grad=True),
        "db": Tensor(np.full(D, -2.0, np.float32), requires_grad=True),
        "ds": Tensor(np.full(D, -0.5, np.float32), requires_grad=True),
    }

    def fwd(i):
        return selective_ssm(i["u"], i["delta"], i["z"], i["A"], i["C"],
                             i["db"], i["ds"])

    # FIXED upstream weights — the numerical gradient perturbs inputs and
    # re-evaluates, so the loss must be deterministic.
    W = Tensor(rng.standard_normal((B, T, D)).astype(np.float32))

    def loss_of(out, i):
        from cognicore.autograd import _mul, _sum
        return _sum(_mul(out, W))

    return check("ssm", lambda: dict(ins), fwd, loss_of)


def test_ops():
    print("\n[2] primitive ops")
    ok = True
    x = Tensor(rng.standard_normal((3, 4)).astype(np.float32), requires_grad=True)
    w = Tensor(rng.standard_normal((4, 5)).astype(np.float32), requires_grad=True)
    ok &= check("matmul", lambda: {"a": x, "b": w},
                lambda i: i["a"] @ i["b"],
                lambda o, i: o.sum())
    x = Tensor(rng.standard_normal((3, 4)).astype(np.float32), requires_grad=True)
    w = Tensor(rng.standard_normal((4,)).astype(np.float32), requires_grad=True)
    ok &= check("rmsnorm", lambda: {"x": x, "w": w},
                lambda i: rmsnorm(i["x"], i["w"]), lambda o, i: o.sum())
    ok &= check("silu", lambda: {"x": x}, lambda i: silu(i["x"]), lambda o, i: o.sum())
    g = Tensor(rng.standard_normal((3, 4)).astype(np.float32), requires_grad=True)
    u = Tensor(rng.standard_normal((3, 4)).astype(np.float32), requires_grad=True)
    ok &= check("swiglu", lambda: {"g": g, "u": u}, lambda i: swiglu(i["g"], i["u"]),
                lambda o, i: o.sum())
    tab = Tensor(rng.standard_normal((6, 3)).astype(np.float32), requires_grad=True)
    idx = np.array([0, 3, 1, 1, 5, 2, 0])
    ok &= check("index_select", lambda: {"t": tab}, lambda i: index_select(i["t"], idx),
                lambda o, i: o.sum())
    dest = Tensor(np.zeros((6, 3), np.float32), requires_grad=True)
    src = Tensor(rng.standard_normal((7, 3)).astype(np.float32), requires_grad=True)
    ok &= check("scatter_add", lambda: {"d": dest, "s": src},
                lambda i: scatter_add(i["d"], idx, i["s"]), lambda o, i: o.sum())
    lg = Tensor(rng.standard_normal((3, 4, 9)).astype(np.float32), requires_grad=True)
    tg = rng.integers(0, 9, (3, 4))
    ok &= check("cross_entropy", lambda: {"l": lg},
                lambda i: cross_entropy(i["l"], tg), lambda o, i: o)
    x3 = Tensor(rng.standard_normal((2, 3, 4)).astype(np.float32), requires_grad=True)
    w3 = Tensor(rng.standard_normal((4, 5)).astype(np.float32), requires_grad=True)
    ok &= check("matmul3d", lambda: {"a": x3, "b": w3}, lambda i: i["a"] @ i["b"],
                lambda o, i: o.sum())
    return ok


def test_model():
    print("\n[3] full CogniCore end-to-end gradient sanity")
    from cognicore.model import Config, CogniCore
    cfg = Config(target_params=400_000, seq_len=32, d_hd=64)
    m = CogniCore(cfg)
    print(f"  tiny model: {m.n_params():,} params  cfg={cfg}")
    B, T = 2, 32
    ids = rng.integers(0, 256, (B, T)).astype(np.int64)
    ids[:, 0] = 10
    tg = rng.integers(0, 256, (B, T)).astype(np.int64)
    loss = m.forward(ids, tg)
    from cognicore.autograd import backward
    backward(loss)
    loss.data.backward if False else None
    dead = [p.name for p in m.params() if p.requires_grad and p.grad is None]
    n_nan = [p.name for p in m.params()
             if p.grad is not None and not np.isfinite(p.grad).all()]
    print(f"  loss = {loss.data:.4f}   (ln 257 = {np.log(257):.4f})")
    print(f"  params with no gradient : {len(dead)} {dead[:5]}")
    print(f"  params with NaN/Inf grad : {len(n_nan)} {n_nan[:5]}")
    gnorm = np.sqrt(sum(float((p.grad ** 2).sum()) for p in m.params()
                        if p.grad is not None))
    print(f"  global grad norm = {gnorm:.4f}")
    ok = (len(dead) == 0 and len(n_nan) == 0 and np.isfinite(loss.data))
    return ok


def _ref_forward(u, delta, z, A, C, db, ds):
    """Brute-force O(T) reference recurrence, float64, no log-space tricks.

    h_t = exp(A*dt_t) * h_{t-1} + dt_t * u_t
    y_t = (C * h_t) * silu(z_t)
    """
    beta = np.exp(np.clip(ds, -4, 4))
    x = (delta + db) * beta
    dt = (np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)) / beta
    B, T, D = u.shape
    h = np.zeros((B, D))
    ys = []
    for t in range(T):
        h = np.exp(A * dt[:, t, :]) * h + dt[:, t, :] * u[:, t, :]
        s = 1.0 / (1.0 + np.exp(-z[:, t, :]))
        ys.append(h * C * (z[:, t, :] * s))
    return np.stack(ys, axis=1), dt


def test_ssm_reference():
    """Compare the fast log-space kernel against a brute-force float64
    reference, for BOTH the forward pass and every gradient."""
    print("\n[1b] selective_ssm vs brute-force float64 reference")
    from cognicore.autograd import backward
    B, T, D = 2, 7, 5
    r = np.random.default_rng(7)
    raw = {
        "u": r.standard_normal((B, T, D)),
        "delta": r.standard_normal((B, T, D)) * 0.5 - 2.0,
        "z": r.standard_normal((B, T, D)) * 0.3,
        "A": -np.exp(r.standard_normal(D)),
        "C": r.standard_normal(D),
        "db": np.full(D, -2.0),
        "ds": np.full(D, -0.5),
    }
    W = r.standard_normal((B, T, D))

    # ---- reference (float64, numeric gradient) ----
    d64 = {k: v.astype(np.float64) for k, v in raw.items()}

    def ref_loss(d):
        y, _ = _ref_forward(d["u"], d["delta"], d["z"], d["A"], d["C"],
                            d["db"], d["ds"])
        return float((y * W).sum())

    num = {}
    for k, v in d64.items():
        g = np.zeros_like(v)
        it = np.nditer(v, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            o = v[i]
            v[i] = o + 1e-6
            a = ref_loss(d64)
            v[i] = o - 1e-6
            b = ref_loss(d64)
            v[i] = o
            g[i] = (a - b) / 2e-6
            it.iternext()
        num[k] = g

    # ---- fast kernel (float32, analytic gradient) ----
    ins = {k: Tensor(v.astype(np.float32), requires_grad=True) for k, v in raw.items()}
    y = selective_ssm(ins["u"], ins["delta"], ins["z"], ins["A"], ins["C"],
                      ins["db"], ins["ds"])
    ref_y, _ = _ref_forward(d64["u"], d64["delta"], d64["z"], d64["A"], d64["C"],
                            d64["db"], d64["ds"])
    fwd_err = np.abs(y.data - ref_y).max() / max(1.0, np.abs(ref_y).max())
    print(f"  forward rel_err = {fwd_err:.2e}")
    ok = fwd_err < 1e-4

    from cognicore.autograd import _mul, _sum
    loss = _sum(_mul(y, Tensor(W.astype(np.float32))))
    for t in ins.values():
        t.grad = None
    backward(loss)

    for k, t in ins.items():
        ana, ref = t.grad.astype(np.float64), num[k]
        rel = np.abs(ana - ref).max() / max(1e-6, np.abs(ref).max())
        flag = "OK " if rel < 3e-2 else "FAIL"
        if rel >= 3e-2:
            ok = False
        print(f"  [{flag}] d/d{k:<6} rel_err={rel:.2e}  "
              f"|ana|={np.abs(ana).max():.5f} |ref|={np.abs(ref).max():.5f}")
    return ok


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_ops()
    all_ok &= test_ssm()
    all_ok &= test_ssm_reference()
    all_ok &= test_model()
    print("\n" + ("ALL GRADCHECKS PASSED" if all_ok else "*** FAILURES ***"))
    raise SystemExit(0 if all_ok else 1)
