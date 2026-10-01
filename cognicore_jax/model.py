"""
model.py — CogniCore en JAX/Flax.

Arquitectura:
  bytes -> HDE (0 params) -> proj_in -> [LCBlock x N] -> RMSNorm -> head_read
  -> HDE un-bind -> 257 byte logits
  + refinement pass con contexto local (±2 bytes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .hde import HDE, PAD
from .ssm import selective_ssm
from .memory import LCCMemory

CONTEXT_RADIUS = 2
CTX_WIDTH = 2 * CONTEXT_RADIUS + 1


@dataclass
class Config:
    target_params: int = 10_000_000
    d_hd: int = 128
    seq_len: int = 256
    vocab: int = 257
    seed: int = 1234
    n_mem: int = 32
    d_mem: int = 48
    d_model: int = None
    n_layers: int = None
    d_inner: int = None
    d_ff: int = None

    def __post_init__(self):
        if self.d_model is None or self.n_layers is None:
            self._solve()
        else:
            self.d_inner = self.d_model
            self.d_ff = int(round(self.d_model * 8 / 3 / 32)) * 32

    def _layer_params(self, d, di, dff):
        return (3 * d * di + 4 * di + di * di + di * d + 3 * d * dff
                + d * self.n_mem + d * self.d_mem + 3 * d + 4 * di)

    def _fixed_params(self, d):
        ho = 5 * self.d_hd
        return (ho * d + CTX_WIDTH * ho * d + d * self.d_hd + d * self.d_hd
                + 3 * d + self.vocab + 1)

    def _solve(self):
        import math
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
            ok = 4 <= n <= 40
            depth_pen = abs(math.log(n / 8.0))
            score = (0 if ok else 1, err + 0.02 * depth_pen)
            if best is None or score < best[0]:
                best = (score, d, n, di, dff)
        if best is None:
            best = ((0, 0.0), 64, 1, 64, 128)
        _, self.d_model, self.n_layers, self.d_inner, self.d_ff = best

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ("d_model", "n_layers", "d_inner", "d_ff", "d_hd", "seq_len",
                 "vocab", "n_mem", "d_mem", "seed", "target_params")}


class LCBlock(nn.Module):
    """Local Cognitive Cell — one recurrent reasoning step."""
    cfg: Config
    li: int

    @nn.compact
    def __call__(self, x, enc):
        cfg = self.cfg
        d, di, dff = cfg.d_model, cfg.d_inner, cfg.d_ff
        s = cfg.seed + self.li * 977

        n1 = self.param(f"L{self.li}.n1", nn.initializers.ones, (d,))
        n2 = self.param(f"L{self.li}.n2", nn.initializers.ones, (d,))
        conv_dw = self.param(f"L{self.li}.conv_dw",
                             nn.initializers.uniform(0.5), (4, di))
        conv_pw = self.param(f"L{self.li}.conv_pw",
                             nn.initializers.normal(1.0 / np.sqrt(di)), (di, di))
        in_proj = self.param(f"L{self.li}.in_proj",
                             nn.initializers.normal(1.0 / np.sqrt(d)), (d, 3 * di))
        in_bias = self.param(f"L{self.li}.in_bias", nn.initializers.zeros, (3 * di,))

        rng = np.random.default_rng(s + 3)
        a0 = -np.log(np.exp(1.0) + (16.0 - np.e) * rng.random(di)).astype(np.float32)
        A = self.param(f"L{self.li}.A", lambda _: jnp.array(a0))
        C = self.param(f"L{self.li}.C", nn.initializers.ones, (di,))
        d_bias = self.param(f"L{self.li}.delta_bias", lambda _: jnp.full(di, -2.0))
        d_scale = self.param(f"L{self.li}.delta_scale", lambda _: jnp.full(di, -0.5))

        out_proj = self.param(f"L{self.li}.out_proj",
                              nn.initializers.normal(1.0 / np.sqrt(di)), (di, d))
        ff_gate = self.param(f"L{self.li}.ff_gate",
                             nn.initializers.normal(1.0 / np.sqrt(d)), (d, dff))
        ff_up = self.param(f"L{self.li}.ff_up",
                           nn.initializers.normal(1.0 / np.sqrt(d)), (d, dff))
        ff_down = self.param(f"L{self.li}.ff_down",
                             nn.initializers.normal(1.0 / np.sqrt(dff)), (dff, d))

        # LCC memory
        mem = LCCMemory(cfg.n_mem, cfg.d_mem, d, seed=s + 8)
        mem_params = mem.init_params(jax.random.PRNGKey(s + 9), k=2)
        mem_proj = self.param(f"L{self.li}.mem_proj",
                              nn.initializers.normal(1.0 / np.sqrt(cfg.d_mem)),
                              (cfg.d_mem, d))

        # Forward
        h = rmsnorm(x, n1)
        proj = h @ in_proj + in_bias
        u, delta, z = jnp.split(proj, 3, axis=-1)

        c = causal_depthwise_conv(u, conv_dw)
        c = jax.nn.silu(c @ conv_pw)

        y = selective_ssm(c, delta, z, A, C, d_bias, d_scale)
        y = y @ out_proj

        m = mem.forward(mem_params, x)
        y = y + m @ mem_proj

        x = x + y

        h2 = rmsnorm(x, n2)
        f = swiglu(h2 @ ff_gate, h2 @ ff_up) @ ff_down
        return x + f


def rmsnorm(x, weight, eps=1e-5):
    """x / sqrt(mean(x^2)+eps) * weight (mean over last axis)."""
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / jnp.sqrt(ms + eps)
    return x * inv * weight


def causal_depthwise_conv(x, w):
    """out[b,t,c] = sum_{i<k} w[i,c] * x[b, t-i, c] (zero padded, causal).

    x: (B,T,C)   w: (k,C)   ->   (B,T,C)
    """
    B, T, C = x.shape
    k = w.shape[0]
    out = jnp.zeros_like(x)
    for i in range(k):
        if i >= T:
            break
        if i == 0:
            out += x * w[0][None, None, :]
        else:
            # shift right by i
            shifted = jnp.zeros_like(x)
            shifted = shifted.at[:, i:, :].set(x[:, :-i, :])
            out += shifted * w[i][None, None, :]
    return out


def swiglu(gate, up):
    """silu(gate) * up"""
    return jax.nn.silu(gate) * up


def l2_normalize(x, axis=-1, eps=1e-6):
    nrm = jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True)) + eps
    return x / nrm


class CogniCore(nn.Module):
    """CogniCore: Hyperdimensional-Encoded Local-Cognitive-cell network."""
    cfg: Config

    @nn.compact
    def __call__(self, ids, targets=None, refine=True):
        cfg = self.cfg
        hde = HDE(dim=cfg.d_hd, seed=0xC0FFEE, atoms=cfg.vocab)
        s = cfg.seed
        ho = hde.out_dim

        proj_in = self.param("proj_in",
                             nn.initializers.normal(1.0 / np.sqrt(ho)), (ho, cfg.d_model))
        refine_in = self.param("refine_in",
                               nn.initializers.normal(1.0 / np.sqrt(CTX_WIDTH * ho)),
                               (CTX_WIDTH * ho, cfg.d_model))
        final_w = self.param("final_w", nn.initializers.ones, (cfg.d_model,))
        head_read = self.param("head_read",
                               nn.initializers.normal(1.0 / np.sqrt(cfg.d_model)),
                               (cfg.d_model, cfg.d_hd))
        refine_read = self.param("refine_read",
                                 nn.initializers.normal(1.0 / np.sqrt(cfg.d_model)),
                                 (cfg.d_model, cfg.d_hd))
        byte_bias = self.param("byte_bias", nn.initializers.zeros, (cfg.vocab,))
        logit_gain = self.param("logit_gain", lambda _: jnp.full((1,), 6.0))

        atom_T = hde.atom_matrix()  # (d_hd, vocab)

        enc = hde.encode(ids)
        x = enc @ proj_in

        for i in range(cfg.n_layers):
            x = LCBlock(cfg, li=i)(x, enc)

        x = rmsnorm(x, final_w)
        logits = self._unbind(x @ head_read, atom_T, logit_gain, byte_bias)

        if refine:
            ctx = local_context(ids, hde)
            r = rmsnorm(x + ctx @ refine_in, final_w)
            logits = logits + self._unbind(r @ refine_read, atom_T, logit_gain, byte_bias)

        if targets is None:
            return logits
        return cross_entropy(logits, targets)

    def _unbind(self, hd, atom_T, logit_gain, byte_bias):
        hn = l2_normalize(hd)
        sims = hn @ atom_T
        return sims * logit_gain + byte_bias


def local_context(ids, hde, radius=2):
    """Encode bytes at offsets -2..+2 through the same HDE and concatenate."""
    B, T = ids.shape
    ho = hde.out_dim
    out = jnp.zeros((B, T, (2 * radius + 1) * ho), dtype=jnp.float32)
    for j, o in enumerate(range(-radius, radius + 1)):
        shifted = jnp.roll(ids, o, axis=1)
        if o > 0:
            shifted = shifted.at[:, :o].set(PAD)
        elif o < 0:
            shifted = shifted.at[:, T + o:].set(PAD)
        out = out.at[:, :, j * ho:(j + 1) * ho].set(hde.encode(shifted))
    return out


def cross_entropy(logits, targets, ignore=-1):
    """Mean cross-entropy over valid positions."""
    m = jnp.max(logits, axis=-1, keepdims=True)
    e = jnp.exp(logits - m)
    s = jnp.sum(e, axis=-1, keepdims=True)
    logp = (logits - m - jnp.log(s)).reshape(-1, logits.shape[-1])
    p = (e / s).reshape(-1, logits.shape[-1])

    valid = (targets != ignore)
    safe_t = jnp.where(valid, targets, 0).reshape(-1)
    n = jnp.maximum(jnp.sum(valid), 1.0)
    rows = jnp.arange(logp.shape[0])
    loss = -jnp.sum(logp[rows, safe_t]) / n
    return loss


def build(target_params=10_000_000, seq_len=256, seed=1234):
    cfg = Config(target_params=target_params, seq_len=seq_len, seed=seed)
    model = CogniCore(cfg)
    return model, cfg
