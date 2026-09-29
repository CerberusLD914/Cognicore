"""
autograd.py — Minimal reverse-mode autodiff engine in pure NumPy.

Why no PyTorch/TensorFlow?
  The whole point of CogniCore is to train on low-resource machines.
  A tensor library pulls in ~500MB-2GB of native binaries. This engine is
  ~400 lines, depends on nothing but NumPy, and trains 10M-parameter models
  on a laptop CPU.

Supported ops: matmul, add/sub/mul/div, sum/mean, reshape, transpose, concat,
split, gather (index_select), scatter_add, rmsnorm, silu, swiglu, cross_entropy.
"""

from __future__ import annotations

import numpy as np

F32 = np.float32


# --------------------------------------------------------------------------
# Tensor
# --------------------------------------------------------------------------
class Tensor:
    __slots__ = ("data", "grad", "parents", "_backward", "requires_grad", "name")

    # Make numpy defer to our operators instead of coercing Tensor via
    # __getitem__ iteration (which silently produces object arrays).
    __array_ufunc__ = None

    def __init__(self, data, parents=(), _backward=None, requires_grad=None, name=""):
        self.data = np.asarray(data, dtype=F32)
        self.parents = tuple(parents)
        self._backward = _backward
        if requires_grad is None:
            requires_grad = any(p.requires_grad for p in self.parents)
        self.requires_grad = requires_grad
        self.grad = None
        self.name = name

    # -- properties -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def sum(self):
        return _sum(self)

    def mean(self):
        return _mean(self)

    def __add__(self, o):
        return _add(self, o)

    def __radd__(self, o):
        return _add(o, self)

    def __sub__(self, o):
        return _sub(self, o)

    def __rsub__(self, o):
        return _sub(o, self)

    def __mul__(self, o):
        return _mul(self, o)

    def __rmul__(self, o):
        return _mul(o, self)

    def __truediv__(self, o):
        return _div(self, o)

    def __matmul__(self, o):
        return _matmul(self, o)

    def __rmatmul__(self, o):
        return _matmul(o, self)

    def __getitem__(self, key):
        return _getitem(self, key)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _reshape(self, shape)

    def sum(self, axis=None, keepdims=False):
        return _sum(self, axis, keepdims)


def tensor(data, requires_grad=False, name=""):
    return Tensor(data, requires_grad=requires_grad, name=name)


def zeros_like(t):
    return Tensor(np.zeros_like(t.data), requires_grad=False)


def _acc(t: Tensor):
    """Accumulate gradient into t.grad (handles multi-use nodes)."""
    if t is None or not t.requires_grad or t.grad is None:
        return
    if t.grad is None:
        t.grad = t.grad
    t._grad_acc = None  # placeholder (unused)


# --------------------------------------------------------------------------
# backward driver
# --------------------------------------------------------------------------
def zero_grads(params):
    for p in params:
        p.grad = None


def backward(loss: Tensor):
    """Reverse-mode autodiff. Sets .grad on every reachable node."""
    topo, seen = [], set()

    def build(t):
        key = id(t)
        if key in seen:
            return
        seen.add(key)
        for p in t.parents:
            build(p)
        topo.append(t)

    build(loss)
    loss.grad = np.ones_like(loss.data)

    for node in reversed(topo):
        g = node.grad
        if g is None:
            continue
        if node._backward is not None:
            node._backward(g)
    return topo


# --------------------------------------------------------------------------
# broadcasting helpers
# --------------------------------------------------------------------------
def _unbroadcast(g, shape):
    """Sum gradient back down to `shape` after NumPy broadcasting."""
    if g.shape == shape:
        return g
    # sum away extra leading dims
    extra = g.ndim - len(shape)
    if extra > 0:
        g = g.sum(axis=tuple(range(extra)))
    # sum away dims that were size 1
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


def _as_tensor(o):
    if isinstance(o, Tensor):
        return o
    return Tensor(o, requires_grad=False)


# --------------------------------------------------------------------------
# elementwise ops
# --------------------------------------------------------------------------
def _add(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    out = a.data + b.data

    def bw(g):
        ga, gb = _unbroadcast(g, a.data.shape), _unbroadcast(g, b.data.shape)
        if a.requires_grad:
            a.grad = ga if a.grad is None else a.grad + ga
        if b.requires_grad:
            b.grad = gb if b.grad is None else b.grad + gb

    return Tensor(out, (a, b), bw)


def _sub(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    out = a.data - b.data

    def bw(g):
        if a.requires_grad:
            ga = _unbroadcast(g, a.data.shape)
            a.grad = ga if a.grad is None else a.grad + ga
        if b.requires_grad:
            gb = _unbroadcast(-g, b.data.shape)
            b.grad = gb if b.grad is None else b.grad + gb

    return Tensor(out, (a, b), bw)


def _mul(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    out = a.data * b.data

    def bw(g):
        ga, gb = g * b.data, g * a.data
        if a.requires_grad:
            ga = _unbroadcast(ga, a.data.shape)
            a.grad = ga if a.grad is None else a.grad + ga
        if b.requires_grad:
            gb = _unbroadcast(gb, b.data.shape)
            b.grad = gb if b.grad is None else b.grad + gb

    return Tensor(out, (a, b), bw)


def _div(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    out = a.data / b.data

    def bw(g):
        if a.requires_grad:
            ga = _unbroadcast(g / b.data, a.data.shape)
            a.grad = ga if a.grad is None else a.grad + ga
        if b.requires_grad:
            gb = _unbroadcast(-g * a.data / (b.data * b.data), b.data.shape)
            b.grad = gb if b.grad is None else b.grad + gb

    return Tensor(out, (a, b), bw)


def _neg(a):
    a = _as_tensor(a)
    return _mul(a, -1.0)


# --------------------------------------------------------------------------
# reductions
# --------------------------------------------------------------------------
def _sum(a, axis=None, keepdims=False):
    a = _as_tensor(a)
    out = a.data.sum(axis=axis, keepdims=keepdims)

    def bw(g):
        if a.requires_grad:
            gg = g
            if axis is not None and not keepdims:
                gg = np.expand_dims(g, axis)
            gg = np.broadcast_to(gg, a.data.shape)
            a.grad = gg.astype(F32) if a.grad is None else a.grad + gg

    return Tensor(out, (a,), bw)


def _mean(a, axis=None, keepdims=False):
    a = _as_tensor(a)
    n = a.data.size if axis is None else np.prod(
        [a.data.shape[k] for k in (axis if isinstance(axis, tuple) else (axis,))]
    )
    out = a.data.mean(axis=axis, keepdims=keepdims)

    def bw(g):
        if a.requires_grad:
            gg = g / F32(n)
            if axis is not None and not keepdims:
                gg = np.expand_dims(g, axis)
            gg = np.broadcast_to(gg, a.data.shape).astype(F32)
            a.grad = gg if a.grad is None else a.grad + gg

    return Tensor(out, (a,), bw)


# --------------------------------------------------------------------------
# matmul
# --------------------------------------------------------------------------
def _matmul(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    out = a.data @ b.data

    def bw(g):
        if a.requires_grad:
            if a.data.ndim == 2 and b.data.ndim == 2:
                ga = g @ b.data.T
            elif a.data.ndim == 3 and b.data.ndim == 2:
                ga = g @ b.data.T
            elif a.data.ndim == 2 and b.data.ndim == 3:
                ga = np.einsum("bth,bdk->bhk", g, b.data)
            else:
                ga = np.matmul(g, np.swapaxes(b.data, -1, -2))
            ga = ga.astype(F32)
            a.grad = ga if a.grad is None else a.grad + ga
        if b.requires_grad:
            if a.data.ndim == 2 and b.data.ndim == 2:
                gb = a.data.T @ g
            elif a.data.ndim == 3 and b.data.ndim == 2:
                gb = np.einsum("btd,btk->dk", a.data, g)
            else:
                gb = np.matmul(np.swapaxes(a.data, -1, -2), g)
            gb = gb.astype(F32)
            b.grad = gb if b.grad is None else b.grad + gb

    return Tensor(out, (a, b), bw)


# --------------------------------------------------------------------------
# shape ops
# --------------------------------------------------------------------------
def _reshape(a, shape):
    a = _as_tensor(a)
    old = a.data.shape
    out = a.data.reshape(shape)

    def bw(g):
        if a.requires_grad:
            gg = g.reshape(old)
            a.grad = gg if a.grad is None else a.grad + gg

    return Tensor(out, (a,), bw)


def _transpose(a, axes):
    a = _as_tensor(a)
    inv = np.argsort(axes)
    out = np.transpose(a.data, axes)

    def bw(g):
        if a.requires_grad:
            gg = np.transpose(g, inv)
            a.grad = gg if a.grad is None else a.grad + gg

    return Tensor(out, (a,), bw)


def _getitem(a: Tensor, key):
    """Basic slicing. Backward scatters the gradient back into a zero tensor."""
    a = _as_tensor(a)
    out = a.data[key]

    def bw(g):
        if not a.requires_grad:
            return
        pad = np.zeros_like(a.data)
        pad[key] = g
        a.grad = pad if a.grad is None else a.grad + pad

    return Tensor(out, (a,), bw)


def _concat(tensors, axis=0):
    tensors = [_as_tensor(t) for t in tensors]
    out = np.concatenate([t.data for t in tensors], axis=axis)

    def bw(g):
        sizes = [t.data.shape[axis] for t in tensors]
        idx = np.cumsum([0] + list(sizes))
        for i, t in enumerate(tensors):
            if t.requires_grad:
                sl = tuple(
                    slice(idx[i], idx[i + 1]) if d == axis else slice(None)
                    for d in range(g.ndim)
                )
                gg = g[sl]
                t.grad = gg if t.grad is None else t.grad + gg

    parents = tuple(tensors)
    return Tensor(out, parents, bw)


# --------------------------------------------------------------------------
# gather / scatter
# --------------------------------------------------------------------------
def index_select(table: Tensor, idx: np.ndarray, axis: int = 0):
    """table[axis][idx] — forward gather, backward scatter-add."""
    if not isinstance(table, Tensor):
        table = Tensor(table)
    a = np.moveaxis(table.data, axis, 0)
    out = a[idx]

    def bw(g):
        if not table.requires_grad:
            return
        gg = np.zeros_like(table.data)
        np.add.at(np.moveaxis(gg, axis, 0), idx, np.moveaxis(g, 0, 0))
        if table.grad is None:
            table.grad = gg
        else:
            table.grad = table.grad + gg

    return Tensor(out, (table,), bw)


def scatter_add(dest: Tensor, idx: np.ndarray, src: Tensor, axis: int = 0):
    """dest[idx] += src  —  forward add-scatter, backward routes grad back.

    Forward:  out = dest with src added at idx.
    Backward: dL/dsrc = grad[idx]   (gather)
              dL/ddest = grad        (additive op -> identity).
    """
    if not isinstance(dest, Tensor):
        dest = Tensor(dest)
    d = dest.data.copy()
    mv_dst = np.moveaxis(d, axis, 0)
    mv_src = np.moveaxis(src.data, axis, 0)
    np.add.at(mv_dst, idx, mv_src)
    out = np.moveaxis(mv_dst, 0, axis)

    def bw(g):
        if src.requires_grad:
            gg = np.moveaxis(g, axis, 0)[idx]
            src.grad = gg if src.grad is None else src.grad + gg
        if dest.requires_grad:
            dest.grad = g if dest.grad is None else dest.grad + g

    return Tensor(out, (dest, src), bw)


# --------------------------------------------------------------------------
# activations / norms
# --------------------------------------------------------------------------
def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5):
    """x / sqrt(mean(x^2)+eps) * weight   (mean over the last axis)"""
    n = x.data.shape[-1]
    x2 = x.data * x.data
    ms = x2.mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    xn = x.data * inv
    out = xn * weight.data

    def bw(g):
        gx = g * weight.data
        if weight.requires_grad:
            gw = _unbroadcast((g * xn).sum(axis=tuple(range(g.ndim - 1))), weight.data.shape)
            weight.grad = gw if weight.grad is None else weight.grad + gw
        if x.requires_grad:
            # dL/dx = inv*gx - inv*xn*<gx,xn>/n
            inner = (gx * xn).sum(axis=-1, keepdims=True) / n
            gg = (gx - xn * inner) * inv
            x.grad = gg if x.grad is None else x.grad + gg

    return Tensor(out, (x, weight), bw)


def exp(x: Tensor):
    e = np.exp(np.clip(x.data, -30.0, 30.0))
    out = e.astype(F32)

    def bw(g):
        if x.requires_grad:
            x.grad = (g * out).astype(F32) if x.grad is None else x.grad + g * out

    return Tensor(out, (x,), bw)


def silu(x: Tensor):
    d = x.data
    s = 1.0 / (1.0 + np.exp(-d))
    out = d * s

    def bw(g):
        if x.requires_grad:
            sig = 1.0 / (1.0 + np.exp(-d))
            gg = g * (sig * (1.0 + d * (1.0 - sig)))
            x.grad = gg if x.grad is None else x.grad + gg

    return Tensor(out, (x,), bw)


def swiglu(gate: Tensor, up: Tensor):
    """silu(gate) * up"""
    a = silu(gate)
    out = a.data * up.data

    def bw(g):
        if gate.requires_grad:
            sig = 1.0 / (1.0 + np.exp(-gate.data))
            d = gate.data
            gg = g * up.data * sig * (1.0 + d * (1.0 - sig))
            gate.grad = gg if gate.grad is None else gate.grad + gg
        if up.requires_grad:
            gg = g * silu(gate).data
            up.grad = gg if up.grad is None else up.grad + gg

    return Tensor(out, (gate, up), bw)


def causal_depthwise_conv(x: Tensor, w: Tensor):
    """out[b,t,c] = sum_{i<k} w[i,c] * x[b, t-i, c]   (zero padded, causal)

    x: (B,T,C)   w: (k,C)   ->   (B,T,C)

    This is the model's ONLY local short-range mixer — 4 taps, k*C parameters.
    It is what a depthwise conv does in a convnet, and it is the cheapest
    possible substitute for local self-attention.
    """
    xd, wd = x.data, w.data
    B, T, C = xd.shape
    k = wd.shape[0]
    out = np.zeros_like(xd)
    for i in range(k):
        if i >= T:
            break
        if i == 0:
            out += xd * wd[0][None, None, :]
        else:
            sh = np.zeros_like(xd)
            sh[:, i:, :] = xd[:, :-i, :]
            out += sh * wd[i][None, None, :]

    def bw(g):
        if x.requires_grad:
            gx = np.zeros_like(xd)
            for i in range(k):
                if i >= T:
                    break
                contrib = g * wd[i][None, None, :]
                if i == 0:
                    gx += contrib
                else:
                    gx[:, :-i, :] += contrib[:, i:, :]
            x.grad = gx if x.grad is None else x.grad + gx
        if w.requires_grad:
            gw = np.zeros_like(wd)
            for i in range(k):
                if i >= T:
                    break
                seg = g if i == 0 else g[:, i:, :]
                if i == 0:
                    src = xd
                else:
                    src = xd[:, :-i, :]
                gw[i] = (seg * src).sum(axis=(0, 1))
            w.grad = gw if w.grad is None else w.grad + gw

    return Tensor(out, (x, w), bw)


def l2_normalize(x: Tensor, axis: int = -1, eps: float = 1e-6):
    """y = x / ||x||.  This is the differentiable 'binding similarity' used
    for HDC un-binding — see model.CogniCore._unbind."""
    nrm = np.sqrt((x.data * x.data).sum(axis=axis, keepdims=True)) + eps
    inv = 1.0 / nrm
    out = x.data * inv

    def bw(g):
        if not x.requires_grad:
            return
        y = x.data * inv
        dot = (g * y).sum(axis=axis, keepdims=True)
        gg = (g - y * dot) * inv
        x.grad = gg.astype(F32) if x.grad is None else x.grad + gg

    return Tensor(out, (x,), bw)


def softplus(x: Tensor, beta: float = 1.0):
    d = x.data
    out = np.log1p(np.exp(-np.abs(beta * d))) / beta + np.maximum(d, 0.0)

    def bw(g):
        if x.requires_grad:
            sig = 1.0 / (1.0 + np.exp(-beta * d))
            gg = g * (sig * beta) / beta
            x.grad = gg if x.grad is None else x.grad + gg

    return Tensor(out, (x,), bw)


def cross_entropy(logits: Tensor, targets: np.ndarray, ignore: int = -1):
    """Mean cross-entropy over valid positions. Fused softmax + logsumexp."""
    x = logits.data
    shp = x.shape
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    s = e.sum(axis=-1, keepdims=True)
    logp = (x - m - np.log(s)).reshape(-1, shp[-1])
    p = (e / s).reshape(-1, shp[-1])

    valid = (targets != ignore)
    safe_t = np.where(valid, targets, 0).reshape(-1)
    n = float(valid.sum()) or 1.0
    rows = np.arange(logp.shape[0])
    loss = -logp[rows, safe_t].sum() / n

    def bw(g):
        if logits.requires_grad:
            d = p.copy()
            d[rows, safe_t] -= 1.0
            d *= (np.float32(g) / n) * valid.reshape(-1)[:, None]
            d = d.reshape(shp).astype(F32)
            logits.grad = d if logits.grad is None else logits.grad + d

    return Tensor(loss, (logits,), bw)


# --------------------------------------------------------------------------
# parameter init
# --------------------------------------------------------------------------
def param(shape, scale="normal", std=0.02, seed=None, name=""):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    if scale == "normal":
        d = rng.standard_normal(shape).astype(F32) * std
    elif scale == "uniform":
        d = (rng.random(shape).astype(F32) * 2 - 1) * std
    elif scale == "zeros":
        d = np.zeros(shape, dtype=F32)
    elif scale == "ones":
        d = np.ones(shape, dtype=F32)
    elif scale == "scaled":  # scaled like GPT-2 residual projections
        d = (rng.standard_normal(shape) / np.sqrt(shape[0])).astype(F32)
    else:
        raise ValueError(scale)
    return Tensor(d, requires_grad=True, name=name)


class Adam:
    """AdamW: decoupled weight decay, cosine LR with warmup."""

    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.wd = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        bc1 = 1 - self.b1**self.t
        bc2 = 1 - self.b2**self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / bc1
            vh = self.v[i] / bc2
            upd = mh / (np.sqrt(vh) + self.eps)
            if self.wd and p.ndim >= 2:
                upd = upd + self.wd * p.data
            p.data -= (lr * upd).astype(F32)

    def clip_(self, max_norm=1.0):
        tot = 0.0
        for p in self.params:
            if p.grad is not None:
                tot += float(np.sum(p.grad.astype(np.float64) ** 2))
        norm = np.sqrt(tot)
        if norm > max_norm and norm > 0:
            s = F32(max_norm / norm)
            for p in self.params:
                if p.grad is not None:
                    p.grad *= s
        return float(norm)
