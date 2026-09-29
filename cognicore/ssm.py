"""
ssm.py — Selective State-Space scan (diagonal) with an O(T) forward AND
O(T) reverse-mode backward. No attention, no KV cache: the recurrent state
is constant size, so memory per token is O(1) instead of O(context).

Forward (per channel c):
    dt_t  = softplus(delta_t)                     input gate / step size
    a_t   = exp(A[c] * dt_t)                      decay   (A[c] < 0, learnable)
    h_t   = a_t * h_{t-1} + dt_t * u_t
    y_t   = (C[c] * h_t) * silu(z_t)              C: learnable (1, D)

Numerically-stable form (log-space cumulative) used for the forward:
    P_t   = cumsum(A*dt)_t
    y_t   = exp(P_t) * cumsum(u*exp(-P))_t * C * silu(z)

Backward uses the *state* gradient recurrence, which is a plain reverse
cumulative sum, so training stays linear in sequence length:
    dh_t  = g_t*C*silu(z_t) + dh_{t+1} * a_{t+1}
    du_t  = dh_t * dt_t
    ddt_t = dh_t * u_t + (dh_t * h_{t-1} * A * a_t)      <- the SSM "energy" term
    dA    = sum_t ddt_t * A_prime...   (accumulated through the same chain)
"""

from __future__ import annotations

import numpy as np

from .autograd import Tensor, softplus, silu, F32


def _reverse_scan(c: np.ndarray, b: np.ndarray, axis: int = 1) -> np.ndarray:
    """Solve the first-order reverse linear recurrence  x_t = b_t + c_t * x_{t+1}
    for every t, in O(log T) vectorised passes (Hillis-Steele associative scan).

    Why this instead of a Python loop: the recurrence is  x_t = B_t + C_t x_{t+1}
    with the monoid (C1,B1)∘(C2,B2) = (C1*C2, B1 + C1*B2). Composing neighbours
    at doubling strides yields the closed form in log2(T) NumPy ops — ~8 passes
    for T=256 instead of 256 interpreted iterations. Crucially, it only ever
    multiplies the (C <= 1) decays, so it cannot overflow, unlike the naive
    exp(cumsum(log a)) trick.

    Works on `axis` (default: axis 1, the time axis of a (B,T,D) tensor).
    c, b: same shape as each other  ->  x: same shape
    """
    T = b.shape[axis]
    C = np.moveaxis(c.copy(), axis, 0)
    X = np.moveaxis(b.copy(), axis, 0)
    d = 1
    while d < T:
        hi = T - d
        X[:hi] += C[:hi] * X[d:]
        C[:hi] *= C[d:]
        d <<= 1
    return np.moveaxis(X, 0, axis)


def selective_ssm(u: Tensor, delta: Tensor, z: Tensor, A: Tensor, C: Tensor,
                  delta_bias: Tensor, delta_scale: Tensor):
    """
    u        : (B, T, D)  input sequence
    delta    : (B, T, D)  raw step-size logits
    z        : (B, T, D)  output gate pre-activation
    A        : (D,)       log-decay init (negative, learnable)
    C        : (D,)       output selector (learnable)
    delta_bias, delta_scale : (D,) learnable DT calibration (Mamba-2 style)
    """
    d_raw = delta.data + delta_bias.data
    # softplus with a learned per-channel scale:  dt = softplus(beta*d_raw)/beta
    beta = np.exp(np.clip(delta_scale.data, -4, 4)).astype(F32)
    x = d_raw * beta
    dt = (np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)) / beta  # softplus/beta
    dt = np.clip(dt, 1e-5, 1.0).astype(F32)   # keeps exp(A*dt) decays in (0,1)
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))   # d softplus / d d_raw

    a_log = A.data[None, None, :] * dt            # (B,T,D) negative
    a = np.exp(a_log)                             # in (0,1)
    P = np.cumsum(a_log, axis=1)
    P = np.clip(P, -30.0, 5.0)                    # guard exp()
    inv = np.exp(-P)

    # log-space cumulative form of   h_t = a_t * h_{t-1} + dt_t * u_t
    #   y_t = C[c] * h_t * silu(z_t)
    # i.e.  y_t = exp(P_t) * cumsum(dt*u*exp(-P))_t * C * silu(z)
    cum = np.cumsum((dt * u.data) * inv, axis=1)
    h = np.exp(P) * cum                           # true state h_t
    hg = silu(z).data
    y = h * C.data[None, None, :] * hg

    def bw(g):
        B_, T_, D_ = h.shape
        Cc = C.data[None, None, :]
        # gh[:,t,:] = dL/dy_t routed through the C selector and the silu gate
        gh = g * Cc * hg

        h_prev = np.zeros_like(h)
        h_prev[:, 1:, :] = h[:, :-1, :]

        # Reverse recurrence over time for every channel independently.
        #   dh_t = gh_t + a_{t+1} * dh_{t+1}
        # (a_j influences every later state, so dh must accumulate)
        cvec = np.zeros_like(gh)
        cvec[:, :-1, :] = a[:, 1:, :]     # c_t = a_{t+1}
        dh = _reverse_scan(cvec, gh)

        #   dL/da_t = dh_t * h_{t-1}                       (pointwise: dh_t is
        # already the TOTAL derivative w.r.t. the state at t)
        #   dL/d(log a_t) = (dh_t * h_{t-1}) * a_t
        d_dlog_a = dh * h_prev * a

        if u.requires_grad:
            u.grad = (dh * dt).astype(F32)
        if delta.requires_grad:
            # dt = softplus(x)/beta with x = (delta+delta_bias)*beta
            #   =>  d dt/d delta = sigmoid(x)      (the 1/beta cancels beta)
            # dt enters twice: the input path (dh*u) and the decay path
            # (dL/d(log a) * A)
            d_dt = dh * u.data + d_dlog_a * A.data[None, None, :]
            delta.grad = (d_dt * sig).astype(F32)
        if delta_scale.requires_grad:
            # dt = softplus(x)/beta,  x = beta*d_raw,  beta = exp(delta_scale)
            #   d dt/d scale = sigmoid(x)*d_raw - dt
            ds_g = d_dt * (sig * d_raw - dt)
            delta_scale.grad = np.sum(ds_g, axis=(0, 1)).astype(F32)
        if A.requires_grad:
            A.grad = np.sum(d_dlog_a * dt, axis=(0, 1)).astype(F32)
        if C.requires_grad:
            C.grad = np.sum(g * h * hg, axis=(0, 1)).astype(F32)
        if z.requires_grad:
            sg = 1.0 / (1.0 + np.exp(-z.data))
            z.grad = (g * h * Cc * sg * (1.0 + z.data * (1.0 - sg))).astype(F32)
        if delta_bias.requires_grad:
            delta_bias.grad = delta.grad.sum(axis=(0, 1)).astype(F32)

    parents = (u, delta, z, A, C, delta_bias, delta_scale)
    return Tensor(y, parents, bw)
