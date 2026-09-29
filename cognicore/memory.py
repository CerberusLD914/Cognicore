"""
memory.py — LCC: Local Cognitive Cells (external hashed associative memory)

A tiny content-addressable scratchpad per layer:

    M slots x d_mem hypervectors   (M = 32, d_mem = 48  ->  1536 numbers)
    ONE learned tensor per layer: the address projection (d x M = ~18k).

The bank is initialised from a seeded hyperdimensional basis (same trick as
the encoder) and addressed by a random projection of the residual stream that
the model may fine-tune. Access is a soft top-2 address with a write-then-
read schedule:

    draft = soft_read(base_slots)          gather top-2 slots, softmax weights
    bank  = W_write^T @ draft / norm       "write" phase  (1 GEMM)
    out   = W_read  @ bank                 "read"  phase  (1 GEMM)

The whole memory costs two tiny GEMMs per layer instead of a KV cache that
grows linearly with context. Addresses are treated as discrete
(non-differentiable), the values are fully differentiable — the standard
convention for hash memories.

Why this matters vs an LLM: an LLM must carry a KV cache (O(T) floats per
token) to remember anything. CogniCore's memory is 1.5 KB per layer, fixed.
"""

from __future__ import annotations

import numpy as np

from .autograd import Tensor, param, index_select, _transpose, exp, F32
from .hde import random_hypervectors


class LCCMemory:
    def __init__(self, cfg, seed: int):
        self.M = cfg.n_mem
        self.dm = cfg.d_mem
        self.d = cfg.d_model
        self.slots = random_hypervectors(self.M, self.dm, seed)          # (M, dm)
        rng = np.random.default_rng(seed + 1)
        self.addr0 = (rng.standard_normal((self.d, self.M)) /
                      np.sqrt(self.d)).astype(F32)                        # (d, M)
        self.W_a = param((self.d, self.M), "scaled", seed=seed + 2,
                         name="lcc.addr")
        self.k = 2

    def params(self):
        return [self.W_a]

    def n_floats(self):
        return self.M * self.dm + self.d * self.M

    def forward(self, x: Tensor, enc: Tensor = None) -> Tensor:
        B, T, d = x.data.shape
        M, dm, k = self.M, self.dm, self.k

        # ---- addresses: top-k slots of (fixed random + learned) proj ----
        xf = x.data.reshape(B * T, d)
        W_a = self.addr0 + self.W_a                       # (d, M) Tensor
        scores = xf @ W_a                                 # (B*T, M) Tensor
        top = np.argpartition(-scores.data, k - 1, axis=1)[:, :k]

        # gather the selected scores DIFFERENTIABLY (index_select, not .data)
        flat = scores.reshape(B * T * M)
        gidx = (np.arange(B * T)[:, None] * M + top).reshape(-1)
        raw = index_select(flat, gidx).reshape(B * T, k)
        # differentiable softmax over the k selected slots (shift is constant,
        # which leaves the softmax gradient unchanged)
        e = exp(raw - Tensor(raw.data.max(axis=1, keepdims=True)))
        w = e / e.sum(axis=1, keepdims=True)               # Tensor (B*T, k)
        wn = w.data

        # ---- draft read from the base bank ------------------------------
        g = index_select(Tensor(self.slots), top.reshape(-1), axis=0)  # (B*T*k, dm)
        draft = (g * w.reshape(-1, 1)).reshape(B * T, k, dm).sum(axis=1)
        draft = draft.reshape(B, T, dm)

        # ---- write gate: only ambiguous tokens write (hard decision) -----
        ent = -(wn * np.log(wn + 1e-8)).sum(axis=1)        # (B*T,)
        gscale = (1.0 / k) * (ent > 0.30).astype(F32)
        Wf = np.zeros((B * T, M), dtype=F32)
        np.put_along_axis(Wf, top, wn * gscale[:, None], axis=1)
        Wf = Wf.reshape(B, T, M)

        # ---- write phase: bank = W^T draft (1 GEMM) ---------------------
        Wt = _transpose(Tensor(Wf), (0, 2, 1))             # (B, M, T)
        bank = Wt @ draft                                  # (B, M, dm)
        bank = bank / (Wt.sum(axis=2, keepdims=True) + 0.25)

        # ---- read phase: out = W @ bank (1 GEMM) -----------------------
        return Tensor(Wf) @ bank                           # (B, T, dm)

    __call__ = forward
