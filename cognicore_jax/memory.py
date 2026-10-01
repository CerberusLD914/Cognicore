"""
memory.py — LCC: Local Cognitive Cells (memoria hash asociativa) en JAX.

Un scratchpad content-addressable por capa:
    M slots x d_mem hipervectores
    Un tensor aprendido por capa: la proyección de dirección (d x M).

Acceso: soft top-2 address con write-then-read.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .hde import random_hypervectors


class LCCMemory:
    def __init__(self, M: int, dm: int, d: int, seed: int):
        self.M = M
        self.dm = dm
        self.d = d
        self.slots = random_hypervectors(M, dm, seed)
        rng = np.random.default_rng(seed + 1)
        self.addr0 = (rng.standard_normal((d, M)) / np.sqrt(d)).astype(np.float32)
        self.addr0 = jnp.array(self.addr0)

    def init_params(self, key, k):
        """Inicializa los parámetros entrenables de la memoria."""
        return {
            "W_a": jax.random.normal(key, (self.d, self.M)) * 0.02,
        }

    def forward(self, params, x):
        """
        x: (B, T, d) residual stream
        returns: (B, T, dm) memory read-out
        """
        B, T, d = x.shape
        M, dm, k = self.M, self.dm, 2

        xf = x.reshape(B * T, d)
        W_a = self.addr0 + params["W_a"]
        scores = xf @ W_a  # (B*T, M)

        # top-k selection (non-differentiable indices)
        top = jnp.argpartition(-scores, k - 1, axis=1)[:, :k]  # (B*T, k)

        # gather selected scores differentiably
        gidx = (jnp.arange(B * T)[:, None] * M + top).reshape(-1)
        flat = scores.reshape(B * T * M)
        raw = flat[gidx].reshape(B * T, k)

        # softmax over k selected slots
        e = jnp.exp(raw - raw.max(axis=1, keepdims=True))
        w = e / e.sum(axis=1, keepdims=True)  # (B*T, k)
        wn = w

        # draft read from base bank
        g = self.slots[top.reshape(-1)]  # (B*T*k, dm)
        draft = (g * w.reshape(-1, 1)).reshape(B * T, k, dm).sum(axis=1)
        draft = draft.reshape(B, T, dm)

        # write gate
        ent = -(wn * jnp.log(wn + 1e-8)).sum(axis=1)
        gscale = (1.0 / k) * (ent > 0.30).astype(jnp.float32)
        Wf = jnp.zeros((B * T, M), dtype=jnp.float32)
        # scatter add: Wf[rows, cols] += vals
        rows = jnp.arange(B * T)[:, None]  # (B*T, 1)
        cols = top  # (B*T, k)
        vals = wn * gscale[:, None]  # (B*T, k)
        Wf = Wf.at[rows, cols].add(vals)
        Wf = Wf.reshape(B, T, M)

        # write phase: bank = W^T draft
        Wt = Wf.transpose(0, 2, 1)  # (B, M, T)
        bank = Wt @ draft  # (B, M, dm)
        bank = bank / (Wt.sum(axis=2, keepdims=True) + 0.25)

        # read phase: out = W @ bank
        return Wf @ bank  # (B, T, dm)
