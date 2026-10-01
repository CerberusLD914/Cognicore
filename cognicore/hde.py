"""
hde.py — Hyperdimensional Encoder (HDE)

THE KEY STRUCTURAL DIFFERENCE FROM EVERY LLM:

  An LLM starts with an nn.Embedding table:  vocab_size x d_model learned
  vectors. At 50k vocab / 768 dims that table alone is 38M parameters
  (often >50% of a small model's total) and it CLOSES THE VOCABULARY —
  the model can only ever emit tokens that existed at init.

  CogniCore replaces it with a *hyperdimensional* representation:

    1. BYTE  -> seed a hypervector in R^D_hd from a fixed seeded random
       basis (semantic "atoms", generated on the fly, zero parameters).
    2. BIND  (circular convolution / XOR-like superposition) the atom with a
       ROLE hypervector (depth, position-mod-k, is-indent, is-punct...).
    3. BUNDLE (majority-vote superposition) the bindings. Bundling is
       near-orthogonal: 1k bound vectors recover with high fidelity from
       their sum in 8192 dims.

  Consequences:
    * 0 embedding parameters  (a GPT-2 124M spends 40M of them)
    * NO FIXED VOCABULARY — any byte sequence is encodable, even unseen
    * the "embedding" is reproducible from a 64-bit seed: 0 bytes to store
    * the same representation serves the encoder and the output head
      (HDE is its own un-embedding), so head cost also drops.
"""

from __future__ import annotations

import numpy as np

# Hyperdimensional primitives -------------------------------------------------


def random_hypervectors(n, dim, seed):
    """Sparse ±1 hypervectors (Kanerva basis) — better separability than dense."""
    rng = np.random.default_rng(seed)
    m = np.zeros((n, dim), dtype=np.float32)
    density = 0.10
    k = max(1, int(dim * density))
    for i in range(n):
        pos = rng.choice(dim, size=k, replace=False)
        m[i, pos] = rng.choice([-1.0, 1.0], size=k).astype(np.float32)
    return m


def bind(a, b):
    """Circular convolution: (a * b)[n] = sum_k a[k] * b[(n-k) mod D]."""
    return np.fft.irfft(np.fft.rfft(a, axis=-1) * np.fft.rfft(b, axis=-1),
                        n=a.shape[-1], axis=-1).astype(np.float32)


def bundle(mats):
    """Superposition: elementwise sum. Orthogonality makes this invertible."""
    return np.sum(mats, axis=0).astype(np.float32)


def unbundle_superposed(bundled, atoms, threshold=0.0):
    """Retrieve similarity scores of all atoms against a bundled vector."""
    b = bundled / (np.linalg.norm(bundled) + 1e-8)
    a = atoms / (np.linalg.norm(atoms, axis=-1, keepdims=True) + 1e-8)
    return (a @ b)


class HDE:
    """Stateless, parameter-free byte -> hypervector encoder.

    Roles used (kept small so bundles stay separable):
      0 CONTENT   the byte itself
      1 DEPTH     nesting level of brackets/indent  (mod 8)
      2 LINE      position within line              (mod 16)
      3 CLASS     coarse semantic class of the byte (op/ident/number/str/eol)
    """

    N_ROLES = 4

    def __init__(self, dim: int = 512, seed: int = 0xC0FFEE, atoms: int = 257):
        self.dim = dim
        self.seed = seed
        self.atoms = atoms                     # 256 bytes + 1 PAD
        self._content = random_hypervectors(self.atoms, dim, seed)          # (257, D)
        self._roles = [
            random_hypervectors(8, dim, seed + 11),    # depth mod 8
            random_hypervectors(16, dim, seed + 23),   # column mod 16
            random_hypervectors(8, dim, seed + 37),    # char class
            random_hypervectors(4, dim, seed + 53),    # tag: start / mid / end / pad
        ]
        self._class_of = byte_class_table()          # (256,) int8

    # -- public -----------------------------------------------------------
    def content(self, ids: np.ndarray) -> np.ndarray:
        return self._content[ids]

    def depth(self, ids: np.ndarray) -> np.ndarray:
        return self._roles[0][self._depth_of(ids) % 8]

    def column(self, ids: np.ndarray) -> np.ndarray:
        return self._roles[1][self._col_of(ids) % 16]

    def klass(self, ids: np.ndarray) -> np.ndarray:
        return self._roles[2][self._class_of[ids] % 8]

    def tag(self, ids: np.ndarray) -> np.ndarray:
        return self._roles[3][self._tag_of(ids) % 4]

    def encode(self, ids: np.ndarray) -> np.ndarray:
        """ids: (B, T) int -> (B, T, 4*D_hd) bundled role stack, L2 normalised.

        The 4 role slots are kept as separate channels instead of being
        summed, so downstream layers can still address them individually
        (a hard bind would destroy separability at 4x compression).
        """
        B, T = ids.shape
        c = self.content(ids)     # (B,T,D)
        d = self.depth(ids)
        k = self.column(ids)
        s = self.klass(ids)
        t = self.tag(ids)
        x = np.stack([c, d, k, s, t], axis=-1)      # (B,T,5,D)
        x = x * np.float32(1.0 / np.sqrt(self.dim))
        return x.reshape(B, T, 5 * self.dim)

    @property
    def out_dim(self):
        return 5 * self.dim

    def atom_matrix(self) -> np.ndarray:
        """(dim, atoms) L2-normalised content hypervectors.

        Un-binding is then just  hn @ atom_matrix.T  — a single GEMM that is
        fully differentiable, which is what lets the read-out train.
        """
        a = self._content / (np.linalg.norm(self._content, axis=1, keepdims=True)
                             + 1e-8)
        return np.ascontiguousarray(a.T.astype(np.float32))

    def similarities(self, hidden: np.ndarray, offset: int = 0) -> np.ndarray:
        """hidden: (..., >=dim) -> logits over the atoms (content slot)."""
        h = hidden[..., offset: offset + self.dim]
        return unbundle_superposed(h.reshape(-1, self.dim), self._content) * 6.0

    # -- derived role features (cheap, deterministic) --------------------
    def _depth_of(self, ids):
        base = BRACKET_DELTA[np.clip(ids, 0, 255)]
        cum = np.cumsum(base, axis=-1)
        return (cum % 8).astype(np.int64)

    def _col_of(self, ids):
        nl = (ids == NL).astype(np.int32)
        pos = np.arange(ids.shape[-1], dtype=np.int32)
        pos = pos - np.maximum.accumulate(np.where(nl > 0, pos, 0), axis=-1)
        return np.clip(pos, 0, 15).astype(np.int64)

    def _tag_of(self, ids):
        nl = (ids == NL).astype(np.int64)
        is_pad = (ids == PAD).astype(np.int64)
        tag = np.ones_like(nl)                        # 1 = mid
        tag[..., 0] = 0                               # 0 = start
        tag = np.where(nl > 0, 0, tag)
        tag = np.where(is_pad > 0, 3, tag)
        return tag

    def state_bytes(self):
        return {
            "dim": int(self.dim), "seed": int(self.seed), "atoms": int(self.atoms)
        }


NL = 10        # byte 10 == '\n'
SP = 32
PAD = 256

# net bracket/indent depth delta per byte value
BRACKET_DELTA = np.zeros(257, dtype=np.int32)
for _c in "([{":
    BRACKET_DELTA[ord(_c)] = 1
for _c in ")]}":
    BRACKET_DELTA[ord(_c)] = -1


def byte_class_table() -> np.ndarray:
    """8 coarse character classes, sized 257 so PAD (256) is addressable."""
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
    t[256] = 7          # PAD gets its own class
    return t
