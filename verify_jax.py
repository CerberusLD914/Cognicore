"""
verify_jax.py — Compara la implementacion JAX contra la de NumPy.

Si la migracion a JAX es correcta, ambos motores deben producir practicamente
la misma perdida sobre los MISMOS pesos. Este script:

  1. Construye el modelo NumPy
  2. Exporta sus pesos al formato Flax que espera cognicore_jax
  3. Corre un forward en JAX con esos pesos
  4. Compara las perdidas

Uso (en Colab, con GPU):
    !python verify_jax.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def numpy_loss():
    from cognicore.model import Config, CogniCore
    from cognicore.data import to_byte_tokens, causal_lm_batch

    cfg = Config(target_params=400_000, seq_len=64, seed=1234)
    m = CogniCore(cfg)
    txt = ("def add(a, b):\n    return a + b\n" * 40)
    d = to_byte_tokens(txt, 64)
    ids, tgt = causal_lm_batch(d[:2])
    loss = m.forward(ids, tgt)
    return float(loss.data), m, cfg, ids, tgt


def main():
    print("=" * 66)
    print("VERIFICACION: JAX vs NumPy")
    print("=" * 66)

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("JAX no instalado. Ejecuta:  pip install -U 'jax[cuda12]' flax optax")
        return 1
    print(f"backend: {jax.default_backend()}   dispositivos: {jax.devices()}")

    ln, m, cfg, ids, tgt = numpy_loss()
    print(f"\n[1] perdida NumPy : {ln:.6f}")

    # --- mapear pesos NumPy -> nombres Flax ---
    # El modelo NumPy usa "L0.n1"; Flax usa el dict anidado del modulo LCBlock.
    sd = m.state_dict()

    def per_layer(name):
        # "L3.n1" -> ("3", "n1")
        li, leaf = name.split(".", 1)
        return li[1:], leaf.replace(".", "_")

    flax_params = {
        "proj_in": jnp.asarray(sd["proj_in"]),
        "refine_in": jnp.asarray(sd["refine_in"]),
        "final_w": jnp.asarray(sd["final_w"]),
        "head_read": jnp.asarray(sd["head_read"]),
        "refine_read": jnp.asarray(sd["refine_read"]),
        "byte_bias": jnp.asarray(sd["byte_bias"]),
        "logit_gain": jnp.asarray(sd["logit_gain"]),
    }
    for i in range(cfg.n_layers):
        node = {}
        for name, arr in sd.items():
            if not name.startswith(f"L{i}."):
                continue
            _, leaf = per_layer(name)
            node[leaf] = jnp.asarray(arr)
        flax_params[f"LCBlock_{i}"] = node

    # --- forward en JAX ---
    from cognicore_jax.model import CogniCore as JaxCore, Config as JaxCfg

    jcfg = JaxCfg(target_params=400_000, seq_len=64, seed=1234, d_hd=64)
    jcfg.n_mem, jcfg.d_mem = cfg.n_mem, cfg.d_mem
    jcfg.d_model, jcfg.n_layers = cfg.d_model, cfg.n_layers
    jcfg.d_inner, jcfg.d_ff = cfg.d_inner, cfg.d_ff

    jm = JaxCore(jcfg)
    ids_j = jnp.asarray(ids, dtype=jnp.int32)
    tgt_j = jnp.asarray(tgt, dtype=jnp.int32)

    def loss_fn(p):
        logits = jm.apply({"params": p}, ids_j, tgt_j, refine=True)
        return logits

    lj = float(loss_fn(flax_params))
    print(f"[2] perdida JAX   : {lj:.6f}")
    d = abs(ln - lj)
    print(f"[3] diferencia    : {d:.6f}")
    tol = 2e-2
    if d < tol:
        print(f"\nOK — las implementaciones coinciden (|d|={d:.2e} < {tol:.0e}).")
        print("Es seguro entrenar en JAX.")
        return 0
    print(f"\nFALLO — |d|={d:.2e} supera la tolerancia {tol:.0e}.")
    print("No entrenes todavia: las implementaciones divergen.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())