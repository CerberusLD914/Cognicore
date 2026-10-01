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

from .autograd import (Tensor, param, index_select, _transpose, exp, F32,
                       cumsum, _sum)
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

        # gather the selected scores DIFFERENTIABLY (index_select, no .data).
        # Aplanar a 1-D y usar indice plano mantiene el enlace con W_a; usar
        # .data aqui lo cortaria en silencio.
        flat = scores.reshape(B * T * M)                  # Tensor, mantiene W_a
        gidx = (np.arange(B * T)[:, None] * M + top).reshape(-1)
        raw = index_select(flat, gidx, axis=0).reshape(B * T, k)
        # differentiable softmax over the k selected slots (shift is constant,
        # which leaves the softmax gradient unchanged)
        e = exp(raw - Tensor(raw.data.max(axis=1, keepdims=True)))
        w = e / e.sum(axis=1, keepdims=True)               # Tensor (B*T, k)
        wn = w.data

        # ---- draft read from the base bank ------------------------------
        g = index_select(Tensor(self.slots), top.reshape(-1), axis=0)  # (B*T*k, dm)
        draft = (g * w.reshape(-1, 1)).reshape(B * T, k, dm).sum(axis=1)
        draft = draft.reshape(B, T, dm)
        draft = draft.reshape(B, T, dm)

        # ---- write gate: only ambiguous tokens write (hard decision) -----
        ent = -(wn * np.log(wn + 1e-8)).sum(axis=1)        # (B*T,)
        gscale = (1.0 / k) * (ent > 0.30).astype(F32)
        Wf = np.zeros((B * T, M), dtype=F32)
        np.put_along_axis(Wf, top, wn * gscale[:, None], axis=1)
        Wf = Wf.reshape(B, T, M)

        # ---- write phase: bank[t] = suma de escrituras en posiciones <= t ---
        # CAUSALIDAD: antes bank = W^T @ draft contraia el eje temporal
        # ENTERO, asi que cada posicion recibia el contenido de todas las
        # posiciones, incluidas las futuras (la que debe predecir). Con eso la
        # perdida bajaba a ~0 nats/byte sin aprender nada y el modelo generaba
        # basura. Aqui se descompone en producto exterior por posicion y se
        # acumula con cumsum sobre t -> sigue siendo O(T).
        Wt = _transpose(Tensor(Wf), (0, 2, 1))             # (B, M, T)
        # (B, M, T, 1) * (B, 1, T, dm) -> (B, M, T, dm). OJO: draft es un
        # Tensor; usar draft.data aqui cortaria el gradiente hacia W_a (por eso
        # gradcheck reportaba lcc.addr sin gradiente).
        contrib = Wt[:, :, :, None] * draft[:, None, :, :]
        bank = cumsum(contrib, axis=2)                      # (B, M, T, dm)
        norm = cumsum(Wt, axis=2)                           # (B, M, T)
        bank = bank / (norm[:, :, :, None] + 0.25)

        # ---- read phase: out[t] = suma_m Wf[t,m] * bank[m,t,:] -----------
        # bank -> (B, T, M, dm) y se contrae el eje M (contraccion pequena).
        bankT = _transpose(bank, (0, 2, 1, 3))             # (B, T, M, dm)
        out = _sum(Tensor(Wf)[:, :, :, None] * bankT, axis=2)   # (B, T, dm)
        return out                                         # (B, T, dm)

    __call__ = forward
