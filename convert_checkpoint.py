"""
convert_checkpoint.py — Convierte un checkpoint de CogniCore-JAX (Flax)
al formato del CogniCore original (NumPy), para poder usar chat.py / chat.cmd.

    py convert_checkpoint.py checkpoints/cognicore-10M.npz

Por que hace falta: los nombres de parametros de Flax ("LCBlock_0/L0_n1")
no coinciden con los del motor NumPy ("L0.n1"), y el trainer JAX ademas no
escribia el .meta.json. Este script reconstruye la arquitectura exacta desde
las formas de los tensores y renombra todo al esquema original.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from cognicore.model import Config, CogniCore

# Nombres fijos (no dependen de la capa) del modelo original.
FIXED = ["proj_in", "refine_in", "final_w", "head_read", "refine_read",
         "byte_bias", "logit_gain"]

# Sufijos de cada tensor dentro de una capa LCBlock.
LAYER_SUFFIXES = ["n1", "n2", "conv_dw", "conv_pw", "in_proj", "in_bias",
                  "A", "C", "delta_bias", "delta_scale", "out_proj",
                  "ff_gate", "ff_up", "ff_down", "mem_proj"]

# LCC: nombre original "lcc.addr" (colisiona entre capas en el motor NumPy,
# por diseño: todas las capas comparten la misma entrada).
LCC_SUFFIX = "lcc.addr"


def _basename(key: str) -> str:
    return key.split("/")[-1]


def _norm(s: str) -> str:
    """Normaliza un componente de nombre: LCBlock_0/L0_n1 -> l0.n1"""
    s = s.replace("/", ".").replace("_", ".")
    while ".." in s:
        s = s.replace("..", ".")
    return s.strip(".").lower()


def _match(keys, layer: str | None, suffix: str):
    """Busca la clave del npz que corresponde a (capa, sufijo)."""
    if layer is None:
        for k in keys:
            if _basename(k) == suffix or _norm(_basename(k)) == _norm(suffix):
                return k
        return None
    exact = (f"{layer}.{suffix}", f"{layer}_{suffix}")
    for k in keys:
        if _basename(k) in exact:
            return k
    want = _norm(f"{layer}.{suffix}")
    for k in keys:
        if _norm(_basename(k)) == want:
            return k
    # ultimo recurso: la ruta completa menciona la capa y termina en el sufijo
    for k in keys:
        nk = _norm(k)
        if nk.endswith(_norm(suffix).replace(".", "")) and f"l{layer[1:]}." in nk + ".":
            return k
    return None


def _make_cfg(d_hd, vocab, n_mem, d_mem, d_model, n_layers, seed=1234,
              seq_len=256):
    """Construye la Config original.

    OJO: Config.__init__ NO acepta n_mem/d_mem — los fija a 32/48 dentro, asi
    que hay que sobrescribirlos despues si el checkpoint usa otros valores.
    """
    cfg = Config(target_params=10_000_000, d_hd=int(d_hd), seq_len=int(seq_len),
                 vocab=int(vocab), seed=int(seed),
                 d_model=int(d_model), n_layers=max(1, int(n_layers)))
    cfg.n_mem = int(n_mem)
    cfg.d_mem = int(d_mem)
    return cfg


def infer_config(z):
    """Reconstruye la Config leyendo las formas de los tensores."""
    keys = list(z.files)

    def find(name):
        for k in keys:
            b = _basename(k)
            if b == name or _norm(b) == _norm(name):
                return k
        return None

    k_proj = find("proj_in")
    if k_proj is None:
        raise SystemExit("ERROR: no se encontro 'proj_in' en el checkpoint.")
    proj_in = z[k_proj]
    d_hd = proj_in.shape[0] // 5          # HDE emite 5 canales de rol
    d_model = proj_in.shape[1]

    k_bias = find("byte_bias")
    vocab = int(z[k_bias].shape[0]) if k_bias else 257

    # nº de capas: mayor indice L{n} presente
    n_layers = 0
    for k in keys:
        n = _norm(_basename(k))
        if n.startswith("l") and "." in n:
            head = n.split(".")[0][1:]
            if head.isdigit():
                n_layers = max(n_layers, int(head) + 1)

    # n_mem / d_mem desde las formas de lcc.addr y mem_proj
    k_addr = _match(keys, "L0", LCC_SUFFIX)
    n_mem = int(z[k_addr].shape[1]) if k_addr else 32
    k_mp = _match(keys, "L0", "mem_proj")
    d_mem = int(z[k_mp].shape[0]) if k_mp else 48

    return _make_cfg(d_hd, vocab, n_mem, d_mem, d_model, n_layers)


def convert(src: Path, dst: Path | None = None, seq_len: int = 256):
    z = np.load(src)

    # 1) config embebida por el trainer, si existe
    cfg_dict = None
    if "__config__" in z.files:
        raw = z["__config__"]
        s = "".join(chr(int(c)) for c in np.asarray(raw).ravel())
        try:
            cfg_dict = json.loads(s)
        except Exception:
            cfg_dict = None

    if cfg_dict:
        g = lambda k, dflt: cfg_dict.get(k, dflt)  # noqa: E731
        cfg = _make_cfg(
            d_hd=g("d_hd", 128), vocab=g("vocab", 257),
            n_mem=g("n_mem", 32), d_mem=g("d_mem", 48),
            d_model=g("d_model", 288), n_layers=g("n_layers", 8),
            seed=g("seed", 1234), seq_len=seq_len)
        print(f"[config] leida del checkpoint: {cfg}")
    else:
        cfg = infer_config(z)
        cfg.seq_len = seq_len
        print(f"[config] inferida por formas:  {cfg}")

    model = CogniCore(cfg)
    keys = list(z.files)
    out, missing = {}, []

    # 2) tensores fijos
    for name in FIXED:
        k = _match(keys, None, name)
        if k is None:
            missing.append(name)
            continue
        out[name] = np.asarray(z[k], dtype=np.float32)

    # 3) tensores por capa
    for i in range(cfg.n_layers):
        tag = f"L{i}"
        for suf in LAYER_SUFFIXES:
            k = _match(keys, tag, suf)
            if k is None:
                missing.append(f"{tag}.{suf}")
                continue
            out[f"{tag}.{suf}"] = np.asarray(z[k], dtype=np.float32)

    # 4) LCC: en el motor NumPy TODAS las capas comparten la entrada "lcc.addr".
    #    Replicamos la de la ultima capa, que es la que sobrevive al dict.
    addr_key = None
    for i in range(cfg.n_layers - 1, -1, -1):
        addr_key = _match(keys, f"L{i}", LCC_SUFFIX)
        if addr_key is not None:
            break
    if addr_key is not None:
        out[LCC_SUFFIX] = np.asarray(z[addr_key], dtype=np.float32)
    else:
        missing.append(LCC_SUFFIX)

    # 5) validacion de formas contra el modelo reconstruido
    probe = {p.name: tuple(p.data.shape) for p in model.params()}
    bad = []
    for name, arr in out.items():
        if name in probe and tuple(arr.shape) != probe[name]:
            bad.append(f"{name}: npz{tuple(arr.shape)} != modelo{probe[name]}")
    if bad:
        print("\n[AVISO] Desajustes de forma:")
        for b in bad:
            print("   " + b)
    if missing:
        print("\n[AVISO] No encontrados (quedan con la inicializacion aleatoria):")
        for m in missing:
            print("   " + m)

    dst = dst or src
    if dst == src:
        dst = src.with_name(src.stem + "_numpy.npz")
    model.save(dst)
    Path(str(dst).replace(".npz", ".meta.json")).write_text(
        json.dumps(model.meta(), indent=2))

    print(f"\n[OK] {len(out)} tensores -> {dst}")
    print(f"     {model.n_params():,} parametros   {cfg}")
    print(f"     meta -> {str(dst).replace('.npz', '.meta.json')}")
    print("\nAhora puedes usar:  chat.cmd  (o  py chat.py " + str(dst) + ")")
    return dst


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    seq_len = 256
    for a in sys.argv[1:]:
        if a.startswith("--seq-len="):
            seq_len = int(a.split("=", 1)[1])

    if not args:
        ck = Path("checkpoints")
        cands = sorted(ck.glob("*.npz"), key=lambda p: p.stat().st_mtime,
                       reverse=True) if ck.exists() else []
        if not cands:
            raise SystemExit("Uso: py convert_checkpoint.py <checkpoint.npz> "
                             "[--seq-len=N]")
        src = cands[0]
    else:
        src = Path(args[0])

    if not src.exists():
        raise SystemExit(f"No existe: {src}")
    print(f"Convirtiendo {src} ...\n")
    convert(src, seq_len=seq_len)


if __name__ == "__main__":
    main()