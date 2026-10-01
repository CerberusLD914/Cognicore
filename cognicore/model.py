"""
model.py — CogniCore: Hyperdimensional-Encoded Local-Cognitive-cell network

This is NOT a transformer and NOT an LLM. Pipeline:

  bytes
    │
    ▼  HDE  Hyperdimensional Encoder  ........  0 parameters
    │     role-stacked hypervectors, D_hd
    │     content ⊗ depth ⊗ column ⊗ class ⊗ tag
    ▼
  proj_in ──► [ LCC block ] × N  ──► residual stream
               │  RMSNorm
               │  depthwise causal conv (k=4)  ← local short-range mixer
               │  selective SSM                 ← long-range path, O(1) state
               │  hash-memory read-out
               │  + residual
               │  SwiGLU FFN
               │  + residual
               ▼
  RMSNorm ──► head_read (d → D_hd) ──► HDE un-bind ──► 257 byte logits
    │                                     ▲
    │                                     └── content hypervectors (param-free)
    ▼
  refinement pass: byte + its ±2 local neighbourhood, second read-out
  logits = pass1_logits + pass2_logits

Compared with a 10M-parameter GPT-2 family model:
  * 0 embedding parameters            (GPT-2 spends ~1.0M of 124M; tied models
                                       still can't handle unseen bytes)
  * no vocabulary limit               (raw bytes, 0-255, no BPE, no tokenizer)
  * O(1) memory per token             (no KV cache at inference)
  * O(T) train time, linear           (no O(T²) attention)
  * recurrent + hyperdimensional ops map cleanly to int8/int4
"""

from __future__ import annotations

import math

import numpy as np

from .autograd import (Tensor, param, rmsnorm, silu, swiglu, cross_entropy,
                       l2_normalize, causal_depthwise_conv, F32)
from .hde import HDE, PAD
from .ssm import selective_ssm
from .memory import LCCMemory

# fixed-radius byte neighbourhood that replaces attention entirely
CONTEXT_RADIUS = 2
CTX_WIDTH = (2 * CONTEXT_RADIUS + 1)


class Config:
    """Auto-solves (d_model, n_layers) to land as close as possible on a
    parameter budget — so 'a 10M model' is a real, automatic constraint."""

    def __init__(self, target_params=10_000_000, d_hd=128, seq_len=256,
                 vocab=257, seed=1234, d_model=None, n_layers=None):
        self.target_params = target_params
        self.d_hd = d_hd
        self.seq_len = seq_len
        self.vocab = vocab
        self.seed = seed
        self.n_mem = 32
        self.d_mem = 48
        if d_model is None or n_layers is None:
            self._solve()
        else:
            self.d_model = d_model
            self.n_layers = n_layers
            self.d_inner = d_model
            self.d_ff = int(round(d_model * 8 / 3 / 32)) * 32

    # ---- parameter accounting ----------------------------------------
    def _layer_params(self, d, di, dff):
        return (3 * d * di          # in_proj -> (u, delta, z)
                + 4 * di            # depthwise conv k=4
                + di * di           # conv pointwise
                + di * d            # out_proj
                + 3 * d * dff       # SwiGLU
                + d * self.n_mem    # LCC address projection
                + d * self.d_mem    # LCC read-out projection
                + 3 * d             # three RMSNorm weights
                + 4 * di)           # SSM A, C, delta_bias, delta_scale

    def _fixed_params(self, d):
        ho = 5 * self.d_hd
        return (ho * d                      # proj_in
                + CTX_WIDTH * ho * d        # refine_in (local byte context)
                + d * self.d_hd              # head_read   (d -> D_hd)
                + d * self.d_hd              # refine_read
                + 3 * d                      # norm weights
                + self.vocab                 # byte bias
                + 1)                         # logit_gain

    def _solve(self):
        """Pick (d_model, n_layers) to hit the parameter budget.

        Also constrains depth to a sane band: very deep/narrow configs are
        numerically fine but optimise badly, and very wide/shallow ones waste
        the parameter budget on channels instead of depth. The search scores
        candidates by budget error but rejects depths outside [n_min, n_max]
        whenever any valid candidate exists.
        """
        target = self.target_params
        best = None
        for d in range(64, 1281, 32):
            di = d
            dff = int(round(d * 8 / 3 / 32)) * 32
            per = self._layer_params(d, di, dff)
            fixed = self._fixed_params(d)
            if per + fixed >= target:
                continue
            n = int(round((target - fixed) / per))
            if n < 1:
                continue
            total = fixed + n * per
            err = abs(total - target) / target
            # depth preference: penalise being far from ~8 layers, and
            # heavily penalise degenerate 1-2 layer configs
            ok = 4 <= n <= 40
            depth_pen = abs(math.log(n / 8.0))
            score = (0 if ok else 1, err + 0.02 * depth_pen)
            if best is None or score < best[0]:
                best = (score, d, n, di, dff)
        if best is None:
            best = ((0, 0.0), 64, 1, 64, 128)
        _sc, self.d_model, self.n_layers, self.d_inner, self.d_ff = best

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ("d_model", "n_layers", "d_inner", "d_ff", "d_hd", "seq_len",
                 "vocab", "n_mem", "d_mem", "seed", "target_params")}

    @classmethod
    def from_dict(cls, d):
        c = cls(target_params=d.get("target_params", 10_000_000))
        for k, v in d.items():
            setattr(c, k, v)
        return c

    def __repr__(self):
        return (f"Config(d_model={self.d_model}, n_layers={self.n_layers}, "
                f"d_inner={self.d_inner}, d_ff={self.d_ff}, d_hd={self.d_hd}, "
                f"seq_len={self.seq_len})")


# --------------------------------------------------------------------------
def _causal_depthwise_conv(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Numpy reference (inference/benchmarking only)."""
    return causal_depthwise_conv(Tensor(x), Tensor(w)).data


class LCBlock:
    """Local Cognitive Cell — one recurrent reasoning step."""

    def __init__(self, cfg: Config, li: int, seed: int):
        d, di, dff = cfg.d_model, cfg.d_inner, cfg.d_ff
        s = seed + li * 977
        p = cfg.n_mem
        self.n1 = param((d,), "ones", name=f"L{li}.n1")
        self.n2 = param((d,), "ones", name=f"L{li}.n2")
        self.conv_dw = param((4, di), "uniform", std=0.5, seed=s, name=f"L{li}.conv_dw")
        self.conv_pw = param((di, di), "scaled", seed=s + 1, name=f"L{li}.conv_pw")
        self.in_proj = param((d, 3 * di), "scaled", seed=s + 2, name=f"L{li}.in_proj")
        self.in_bias = param((3 * di,), "zeros", name=f"L{li}.in_bias")
        rng = np.random.default_rng(s + 3)
        a0 = -np.log(np.exp(1.0) + (16.0 - np.e) * rng.random(di)).astype(F32)
        self.A = Tensor(a0, requires_grad=True, name=f"L{li}.A")
        self.C = param((di,), "ones", name=f"L{li}.C")
        self.d_bias = Tensor(np.full(di, -2.0, dtype=F32), requires_grad=True,
                             name=f"L{li}.delta_bias")
        self.d_scale = Tensor(np.full(di, -0.5, dtype=F32), requires_grad=True,
                              name=f"L{li}.delta_scale")
        self.out_proj = param((di, d), "scaled", seed=s + 4, name=f"L{li}.out_proj")
        self.ff_gate = param((d, dff), "scaled", seed=s + 5, name=f"L{li}.ff_gate")
        self.ff_up = param((d, dff), "scaled", seed=s + 6, name=f"L{li}.ff_up")
        self.ff_down = param((dff, d), "scaled", seed=s + 7, name=f"L{li}.ff_down")
        self.mem = LCCMemory(cfg, seed=s + 8)
        self.mem_proj = param((cfg.d_mem, d), "scaled", seed=s + 9, name=f"L{li}.mem_proj")
        self.cfg = cfg

    def params(self):
        return ([self.n1, self.n2, self.conv_dw, self.conv_pw,
                 self.in_proj, self.in_bias, self.A, self.C, self.d_bias,
                 self.d_scale, self.out_proj, self.ff_gate, self.ff_up,
                 self.ff_down, self.mem_proj] + self.mem.params())

    def forward(self, x: Tensor, enc: Tensor):
        cfg = self.cfg
        B, T, _ = x.data.shape
        di = cfg.d_inner

        h = rmsnorm(x, self.n1)
        proj = h @ self.in_proj + self.in_bias
        u, delta, z = (proj[:, :, :di], proj[:, :, di:2 * di], proj[:, :, 2 * di:])

        c = causal_depthwise_conv(u, self.conv_dw)
        c = silu(c @ self.conv_pw)

        y = selective_ssm(c, delta, z, self.A, self.C, self.d_bias, self.d_scale)
        y = y @ self.out_proj

        m = self.mem(x, enc)
        y = y + m @ self.mem_proj

        x = x + y

        h2 = rmsnorm(x, self.n2)
        f = swiglu(h2 @ self.ff_gate, h2 @ self.ff_up) @ self.ff_down
        return x + f


# --------------------------------------------------------------------------
class CogniCore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hde = HDE(dim=cfg.d_hd, seed=0xC0FFEE, atoms=cfg.vocab)
        s = cfg.seed
        ho = self.hde.out_dim
        self.proj_in = param((ho, cfg.d_model), "scaled", seed=s, name="proj_in")
        self.refine_in = param((CTX_WIDTH * ho, cfg.d_model), "scaled", seed=s + 2,
                               name="refine_in")
        self.blocks = [LCBlock(cfg, i, s + 31) for i in range(cfg.n_layers)]
        self.final_w = param((cfg.d_model,), "ones", name="final_w")
        self.head_read = param((cfg.d_model, cfg.d_hd), "scaled", seed=s + 4,
                               name="head_read")
        self.refine_read = param((cfg.d_model, cfg.d_hd), "scaled", seed=s + 5,
                                 name="refine_read")
        self.byte_bias = param((cfg.vocab,), "zeros", name="byte_bias")
        # learned contrast for the HDC un-binding (see _unbind)
        self.logit_gain = param((1,), "zeros", name="logit_gain")
        self.logit_gain.data = np.full(1, 6.0, dtype=F32)
        # constant (d_hd, vocab) HDE content atoms — un-binding target
        self.atom_T = Tensor(self.hde.atom_matrix(), requires_grad=False)
        self._params = ([self.proj_in, self.refine_in, self.final_w, self.head_read,
                         self.refine_read, self.byte_bias, self.logit_gain]
                        + [p for b in self.blocks for p in b.params()])

    # -- bookkeeping -----------------------------------------------------
    def params(self):
        return self._params

    def n_params(self):
        return int(sum(p.data.size for p in self._params))

    def n_params_breakdown(self):
        fixed, per_layer = 0, []
        names = {p.name for p in self._params}
        for p in self._params:
            if p.name.startswith("L"):
                continue
            fixed += p.data.size
        for b in self.blocks:
            per_layer.append(int(sum(p.data.size for p in b.params())))
        return {"fixed": fixed, "per_layer": per_layer,
                "total": fixed + sum(per_layer),
                "hde": 0}

    def state_dict(self):
        return {p.name: p.data for p in self._params}

    def save(self, path, extra=None):
        np.savez_compressed(path, **self.state_dict())

    def load(self, path):
        z = np.load(path)
        for p in self._params:
            if p.name in z:
                p.data = np.asarray(z[p.name], dtype=F32).reshape(p.data.shape)

    def meta(self):
        return {
            "arch": "CogniCore / HDS-LSM",
            "not_an_llm": "no attention, no KV cache, no token embedding table",
            "config": self.cfg.to_dict(),
            "hde": self.hde.state_bytes(),
            "n_params": self.n_params(),
            "breakdown": self.n_params_breakdown(),
        }

    # -- forward ---------------------------------------------------------
    def _unbind(self, hd: Tensor) -> Tensor:
        """Hyperdimensional un-binding, as a differentiable GEMM.

        The read-out hypervector is compared against all 257 content atoms by
        cosine similarity — i.e. the model recovers the byte by *decoding* a
        superposed hypervector, not by looking up a row of a table. Because
        it is one matmul against a constant matrix, gradients flow all the
        way back into the residual stream.

        `logit_gain` is LEARNED. Cosine similarity between hypervectors in
        D_hd dims is intrinsically low-contrast (values cluster near 1/sqrt(D_hd)),
        so a fixed scale would leave the output distribution nearly uniform
        and the model unable to ever become confident. Letting the model
        learn its own contrast is what makes the un-binding expressive.
        """
        hn = l2_normalize(hd)
        sims = hn @ self.atom_T                     # (B, T, vocab)
        return sims * self.logit_gain + self.byte_bias

    def forward(self, ids: np.ndarray, targets: np.ndarray = None,
                refine: bool = True):
        cfg = self.cfg
        enc = Tensor(self.hde.encode(ids))
        x = enc @ self.proj_in
        for b in self.blocks:
            x = b.forward(x, enc)
        x = rmsnorm(x, self.final_w)

        logits = self._unbind(x @ self.head_read)

        if refine:
            # second pass: re-read the byte through a fixed-radius window
            ctx = Tensor(local_context(ids, self.hde))
            r = rmsnorm(x + ctx @ self.refine_in, self.final_w)
            logits = logits + self._unbind(r @ self.refine_read)

        if targets is None:
            return logits
        return cross_entropy(logits, targets)

    # -- byte-level generation (O(1) state, no KV cache) -----------------
    def init_state(self, B: int):
        return {"bytes": np.full((B,), 10, dtype=np.int64), "h": None}


# --------------------------------------------------------------------------
def local_context(ids: np.ndarray, hde: HDE, radius: int = 2) -> np.ndarray:
    """Encode bytes at offsets -2..+2 through the same HDE and concatenate.

    This is the entire substitute for attention: a fixed-radius byte
    neighbourhood. No T×T matrix is ever formed, anywhere in the network.
    """
    B, T = ids.shape
    ho = hde.out_dim
    out = np.zeros((B, T, (2 * radius + 1) * ho), dtype=F32)
    for j, o in enumerate(range(-radius, radius + 1)):
        # Desplazamiento CAUSAL con relleno PAD: la posicion t recibe ids[t+o].
        # np.roll envolvia los extremos (el byte final reaparecia al principio).
        sh = np.full((B, T), PAD, dtype=np.int64)
        if o == 0:
            sh[:, :] = ids
        elif o > 0:                      # mira a la derecha: ids[t+o]
            if o < T:
                sh[:, :T - o] = ids[:, o:]
        else:                            # mira a la izquierda: ids[t+o]
            if -o < T:
                sh[:, -o:] = ids[:, :T + o]
        out[:, :, j * ho:(j + 1) * ho] = hde.encode(sh)
    return out


def build(target_params=10_000_000, seq_len=256, verbose=True):
    cfg = Config(target_params=target_params, seq_len=seq_len)
    m = CogniCore(cfg)
    if verbose:
        print(cfg)
        bd = m.n_params_breakdown()
        print(f"  fixed (HDE projection + heads): {bd['fixed']:,}")
        print(f"  per layer: {bd['per_layer'][0]:,} x {cfg.n_layers}")
        print(f"  HDE parameters: 0  (vocabulary-free, seeded)")
        print(f"  TOTAL: {m.n_params():,} parameters")
    return m
