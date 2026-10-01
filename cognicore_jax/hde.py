"""
hde.py — Hyperdimensional Encoder (HDE) en JAX.

Codificador byte -> hipervector sin parámetros, usando un basis
hiperdimensional con semilla fija. Cero parámetros entrenables.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

NL = 10
SP = 32
PAD = 256


def random_hypervectors(n, dim, seed):
    """Hipervectores sparse ±1 (Kanerva basis)."""
    rng = np.random.default_rng(seed)
    m = np.zeros((n, dim), dtype=np.float32)
    density = 0.10
    k = max(1, int(dim * density))
    for i in range(n):
        pos = rng.choice(dim, size=k, replace=False)
        m[i, pos] = rng.choice([-1.0, 1.0], size=k).astype(np.float32)
    return jnp.array(m)


def byte_class_table():
    """8 clases de caracteres, tamaño 257 para que PAD (256) sea direccionable."""
    t = np.zeros(257, dtype=np.int64)
    for c in range(256):
        ch = chr(c)
        if ch.isalpha() or ch == "_":
            t[c] = 1
        elif ch.isdigit():
            t[c] = 2
        elif ch in " \t":
            t[c] = 3
        elif ch in "\"'`":
            t[c] = 4
        elif ch == "\n":
            t[c] = 6
        else:
            t[c] = 5
    t[256] = 7
    return jnp.array(t)


# net bracket/indent depth delta per byte value
BRACKET_DELTA = np.zeros(257, dtype=np.int32)
for _c in "([{":
    BRACKET_DELTA[ord(_c)] = 1
for _c in ")]}":
    BRACKET_DELTA[ord(_c)] = -1
BRACKET_DELTA_JAX = jnp.array(BRACKET_DELTA)


class HDE:
    """Codificador byte -> hipervector sin parámetros.

    Roles:
      0 CONTENT   el byte mismo
      1 DEPTH     nivel de anidación (mod 8)
      2 LINE      posición en línea (mod 16)
      3 CLASS     clase semántica del byte
    """

    N_ROLES = 4

    def __init__(self, dim: int = 512, seed: int = 0xC0FFEE, atoms: int = 257):
        self.dim = dim
        self.seed = seed
        self.atoms = atoms
        self._content = random_hypervectors(self.atoms, dim, seed)
        self._roles = [
            random_hypervectors(8, dim, seed + 11),
            random_hypervectors(16, dim, seed + 23),
            random_hypervectors(8, dim, seed + 37),
            random_hypervectors(4, dim, seed + 53),
        ]
        self._class_of = byte_class_table()

    @property
    def out_dim(self):
        return 5 * self.dim

    def content(self, ids):
        return self._content[ids]

    def depth(self, ids):
        base = BRACKET_DELTA_JAX[jnp.clip(ids, 0, 255)]
        cum = jnp.cumsum(base, axis=-1)
        return self._roles[0][cum % 8]

    def column(self, ids):
        nl = (ids == NL).astype(jnp.int32)
        pos = jnp.arange(ids.shape[-1], dtype=jnp.int32)
        pos = pos - jnp.maximum.accumulate(jnp.where(nl > 0, pos, 0), axis=-1)
        return self._roles[1][jnp.clip(pos, 0, 15)]

    def klass(self, ids):
        return self._roles[2][self._class_of[ids] % 8]

    def tag(self, ids):
        nl = (ids == NL).astype(jnp.int64)
        is_pad = (ids == PAD).astype(jnp.int64)
        tag = jnp.ones_like(nl)
        tag = tag.at[..., 0].set(0)
        tag = jnp.where(nl > 0, 0, tag)
        tag = jnp.where(is_pad > 0, 3, tag)
        return self._roles[3][tag % 4]

    def encode(self, ids):
        """ids: (B, T) int -> (B, T, 5*D_hd)."""
        B, T = ids.shape
        c = self.content(ids)
        d = self.depth(ids)
        k = self.column(ids)
        s = self.klass(ids)
        t = self.tag(ids)
        x = jnp.stack([c, d, k, s, t], axis=-1)
        x = x * jnp.float32(1.0 / jnp.sqrt(self.dim))
        return x.reshape(B, T, 5 * self.dim)

    def atom_matrix(self):
        """(dim, atoms) L2-normalized content hypervectors."""
        a = self._content / (jnp.linalg.norm(self._content, axis=1, keepdims=True) + 1e-8)
        return jnp.ascontiguousarray(a.T.astype(jnp.float32))
