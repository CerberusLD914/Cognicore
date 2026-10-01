"""
ssm.py — Selective State-Space scan (diagonal) en JAX.

Forward (per channel c):
    dt_t  = softplus(delta_t)
    a_t   = exp(A[c] * dt_t)
    h_t   = a_t * h_{t-1} + dt_t * u_t
    y_t   = (C[c] * h_t) * silu(z_t)

Usa jax.lax.scan para O(T) forward y backward automatico.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def selective_ssm(u, delta, z, A, C, delta_bias, delta_scale):
    """
    u        : (B, T, D)  input sequence
    delta    : (B, T, D)  raw step-size logits
    z        : (B, T, D)  output gate pre-activation
    A        : (D,)       log-decay init (negative, learnable)
    C        : (D,)       output selector (learnable)
    delta_bias, delta_scale : (D,) learnable DT calibration
    """
    d_raw = delta + delta_bias
    beta = jnp.exp(jnp.clip(delta_scale, -4, 4))
    x = d_raw * beta
    dt = (jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)) / beta
    dt = jnp.clip(dt, 1e-5, 1.0)

    a_log = A[None, None, :] * dt
    a = jnp.exp(a_log)

    B, T, D = u.shape

    # Transpose to (T, B, D) for scan over time dimension
    a_T = a.transpose(1, 0, 2)      # (T, B, D)
    dt_T = dt.transpose(1, 0, 2)    # (T, B, D)
    u_T = u.transpose(1, 0, 2)      # (T, B, D)
    z_T = z.transpose(1, 0, 2)      # (T, B, D)

    def scan_fn(h_prev, inputs):
        a_t, dt_t, u_t, z_t = inputs
        h_t = a_t * h_prev + dt_t * u_t
        sig = jax.nn.silu(z_t)
        y_t = C * h_t * sig
        return h_t, y_t

    h0 = jnp.zeros((B, D), dtype=jnp.float32)
    _, y_T = jax.lax.scan(scan_fn, h0, (a_T, dt_T, u_T, z_T), reverse=False)
    # y_T is (T, B, D), transpose back to (B, T, D)
    return y_T.transpose(1, 0, 2)
