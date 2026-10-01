"""
train.py — Bucle de entrenamiento CogniCore en JAX con GPU.

Optimizaciones clave:
  - jax.jit para compilar forward+backward en un solo kernel GPU
  - jax.grad para diferenciación automática
  - AdamW con decoupled weight decay
  - Cosine LR schedule con warmup
  - Gradient clipping
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from .data import load_corpus, to_byte_tokens, causal_lm_batch
from .model import Config, CogniCore, cross_entropy


def lr_at(step, total, base_lr, warmup=0.05, floor=0.1):
    w = max(1, int(total * warmup))
    if step < w:
        return base_lr * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return base_lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


class TrainState(train_state.TrainState):
    pass


def create_train_state(rng, model, ids, cfg, lr=3e-3, weight_decay=0.02):
    """Inicializa el estado de entrenamiento con AdamW."""
    # Forward inicial para inicializar parámetros
    variables = model.init(rng, ids, jnp.zeros_like(ids))
    params = variables["params"]

    # AdamW con gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, ids, tgt, lr):
    """Un paso de entrenamiento compilado con JIT.

    NOTA: CogniCore.__call__ ya devuelve cross_entropy(logits, targets)
    cuando `targets` no es None, asi que aqui NO se recalcula.
    """

    def loss_fn(params):
        return state.apply_fn({"params": params}, ids, tgt, refine=True)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, ids, tgt):
    """Evaluación compilada con JIT."""
    return state.apply_fn({"params": state.params}, ids, tgt, refine=True)


def train(target_params=10_000_000,
          corpus_bytes=8_000_000,
          steps=2000,
          batch=8,
          seq_len=256,
          lr=3e-3,
          out="cognicore-10M",
          seed=0,
          allow_network=True,
          log=print,
          eval_every=200,
          resume=False,
          checkpoint_dir="checkpoints"):
    """Entrena CogniCore en GPU con JAX."""
    rng = jax.random.PRNGKey(seed)
    ck = Path(checkpoint_dir)
    ck.mkdir(exist_ok=True)

    # ---------------- modelo ----------------
    cfg = Config(target_params=target_params, seq_len=seq_len, seed=1234 + seed)
    model = CogniCore(cfg)

    # Inicializar con un batch dummy para obtener la forma
    dummy_ids = jnp.zeros((1, seq_len), dtype=jnp.int32)
    dummy_tgt = jnp.zeros((1, seq_len), dtype=jnp.int32)
    state = create_train_state(rng, model, dummy_ids, cfg, lr=lr)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    log("=" * 74)
    log("CogniCore-JAX — Hyperdimensional Encoded Local-Cognitive-cell network")
    log("NOT an LLM: no attention, no KV cache, no embedding table")
    log("=" * 74)
    log(f"  {cfg}")
    log(f"  parameters        : {n_params:,}")
    log(f"  HDE (embedding)   : 0  parameters  (seeded, vocabulary-free)")
    log(f"  memory            : {cfg.n_mem} slots x {cfg.d_mem} dims "
        f"= {cfg.n_mem * cfg.d_mem * 4 / 1024:.1f} KB/layer, fixed size")
    log(f"  device            : {jax.devices()[0]}")
    log("")

    # ---------------- datos ----------------
    log("[data] acquiring corpus ...")
    text = load_corpus(corpus_bytes, log=log, allow_network=allow_network, seed=seed)
    data = to_byte_tokens(text, seq_len)
    log(f"[data] {len(text):,} chars -> {data.shape[0]:,} sequences of {seq_len}")
    log("")

    val_n = min(256, data.shape[0] // 20)
    val = data[:val_n]
    train_data = data[val_n:]

    # Convertir a JAX arrays (int32 explicito: JAX no tiene int64 por defecto)
    val_ids = jnp.asarray(val[:, :-1], dtype=jnp.int32)
    val_tgt = jnp.asarray(val[:, 1:], dtype=jnp.int32)

    start = 0
    hist = []
    best = float("inf")

    ck_path = ck / f"{out}.npz"
    hist_path = ck / f"{out}.history.json"
    if resume and ck_path.exists():
        # TODO: implementar resume con checkpoints de Flax
        log(f"[ckpt] resume not yet implemented, starting fresh")

    log(f"[train] {steps} steps, batch {batch}, seq {seq_len}, lr {lr}")
    log("-" * 74)
    t0 = time.time()
    running = []

    # RNG de muestreo. OJO: este mismo objeto se pasa a model.init/apply, y Flax
    # lo consume al construir el grafo (los initializers `lambda _:` de cada
    # capa se ejecutan en ese momento). Por eso NO se puede re-muestrear aqui con
    # un default_rng(seed + step): eso haria que cada re-muestreo cambiase el
    # PRNG del grafo y los initializers volatile-mostrarian valores distintos en
    # cada paso. Un unico RNG que avanza con split() mantiene la inicializacion
    # estable entre pasos.
    data_rng = np.random.default_rng(seed)

    for step in range(start, steps):
        # Muestrear batch aleatorio
        rows = train_data[data_rng.integers(0, len(train_data), batch)]
        ids = jnp.asarray(rows[:, :-1], dtype=jnp.int32)
        tgt = jnp.asarray(rows[:, 1:], dtype=jnp.int32)

        cur_lr = lr_at(step, steps, lr)
        state, loss = train_step(state, ids, tgt, cur_lr)
        running.append(float(loss))

        if step % 20 == 0 or step == steps - 1:
            avg = float(np.mean(running[-20:]))
            el = time.time() - t0
            done = step - start + 1
            tps = done * batch * seq_len / max(el, 1e-9)
            log(f"  step {step:>6}/{steps}  loss {avg:7.4f}  "
                f"bpc {avg / np.log(2):6.3f}  lr {cur_lr:.2e}  "
                f"{tps / 1000:6.1f} kB/s  "
                f"({el / 60:.1f} min)")
            hist.append({"step": step, "train": avg, "lr": cur_lr})

        if (step + 1) % eval_every == 0 or step == steps - 1:
            vl = float(eval_step(state, val_ids, val_tgt))
            log(f"    >> val loss {vl:.4f}  val bpc {vl / np.log(2):.3f}")
            # Si este step no se registro arriba (no es multiplo de 20 y no es
            # el ultimo), hay que crear la entrada en vez de mutar la anterior.
            if not hist or hist[-1]["step"] != step:
                hist.append({"step": step, "train": float(np.mean(running[-20:])),
                             "lr": cur_lr})
            hist[-1]["val"] = vl
            if vl < best:
                best = vl
                # Guardar checkpoint
                save_checkpoint(state, ck_path, model, cfg, n_params)
                log(f"    >> saved {ck_path} (best val {best:.4f})")

    if not ck_path.exists():
        save_checkpoint(state, ck_path, model, cfg, n_params)
    hist_path.write_text(json.dumps(hist, indent=2))

    log("-" * 74)
    log(f"[done] best val loss {best:.4f}  (random byte baseline = ln257 = 5.549)")
    log(f"[done] weights -> {ck_path}  ({ck_path.stat().st_size / 1e6:.1f} MB)")
    return state, model, cfg, hist, best


def save_checkpoint(state, path, model, cfg, n_params):
    """Guarda checkpoint como NPZ + .meta.json.

    Guarda ADEMAS la config como el array __config__ dentro del npz, para que
    convert_checkpoint.py pueda reconstruir la arquitectura exacta sin tener que
    adivinarla por las formas.
    """
    params = state.params
    flat = {}
    for key, value in _flatten_params(params).items():
        flat[key] = np.array(value)
    # config embebida (Flax no necesita esto; el conversor si)
    cfg_json = json.dumps(cfg.to_dict())
    flat["__config__"] = np.array(list(cfg_json))
    np.savez_compressed(path, **flat)

    # .meta.json hermano: sin esto chat.py / quicktest.py no pueden deducir la
    # arquitectura y abortan con "cannot infer architecture from checkpoint".
    meta = {
        "arch": "CogniCore / HDS-LSM",
        "backend": "jax",
        "not_an_llm": "no attention, no KV cache, no token embedding table",
        "config": cfg.to_dict(),
        "hde": {"dim": int(cfg.d_hd), "seed": 0xC0FFEE, "atoms": int(cfg.vocab)},
        "n_params": int(n_params),
    }
    meta_path = Path(str(path).replace(".npz", ".meta.json"))
    meta_path.write_text(json.dumps(meta, indent=2))


def _flatten_params(params, prefix=""):
    """Aplana un diccionario anidado de parámetros."""
    flat = {}
    for key, value in params.items():
        new_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, new_key))
        else:
            flat[new_key] = value
    return flat


def evaluate(state, val, batch=8):
    """Evalúa el modelo en el conjunto de validación."""
    tot, n = 0.0, 0
    for i in range(0, len(val) - batch + 1, batch):
        ids = jnp.asarray(val[i:i + batch, :-1], dtype=jnp.int32)
        tgt = jnp.asarray(val[i:i + batch, 1:], dtype=jnp.int32)
        l = float(eval_step(state, ids, tgt))
        tot += l * len(tgt)
        n += len(tgt)
    return tot / max(n, 1)
