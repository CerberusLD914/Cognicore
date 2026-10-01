"""
chat.py — interactive console for a trained CogniCore model.

    py chat.py                      load newest checkpoint
    py chat.py checkpoints/x.npz   load a specific one
    py chat.py checkpoints/x.npz 300   generate 300 bytes

Commands:
    anything typed   -> becomes the prompt, model continues it
    :temp 0.9        sampling temperature
    :topk 40         top-k cutoff
    :n 400           bytes to generate
    :stats           model info
    :compare <text>  per-byte surprisal on your own text
    :save <name>     write the current sample to a file
    :quit            exit

Note on honesty: with only ~1.5k training steps the model produces
structure, not correct code. It has learned byte-level regularities of
Python. The point is that the whole loop is real and interactive.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Windows consoles default to a legacy codepage, which mangles the box/arrow
# characters used below. Force UTF-8 so the output is readable everywhere.
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from cognicore.model import Config, CogniCore
from cognicore import generate as G
from cognicore.quant import quantize, report as qreport


def find_checkpoints():
    """Candidate checkpoints, newest first."""
    ck = Path("checkpoints")
    if not ck.exists():
        return []
    return sorted(ck.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)


def find_checkpoint():
    cands = find_checkpoints()
    return cands[0] if cands else None


def load(path):
    """Load a checkpoint, reconstructing its exact config.

    The .npz only stores raw weight arrays under their parameter names, so
    the architecture has to be rebuilt. A sibling .meta.json is used when
    present; otherwise the shapes of the stored tensors are matched against
    candidate configs so an arbitrary checkpoint still loads.
    """
    import json
    print(f"loading {path} ...", end=" ", flush=True)
    z = np.load(path)
    names = set(z.files)

    def matches(cfg):
        try:
            probe = CogniCore(cfg)
        except Exception:
            return False
        have = set(probe.state_dict().keys())
        return have.issubset(names)

    cfg = None
    meta_p = Path(str(path).replace(".npz", ".meta.json"))
    if meta_p.exists():
        try:
            cfg = Config.from_dict(json.loads(meta_p.read_text()).get("config"))
        except Exception:
            cfg = None
    if cfg is None or not matches(cfg):
        cfg = None
        # search the parameter budget space for a config whose tensors match
        for target in (2_000_000, 4_000_000, 10_000_000, 20_000_000, 50_000_000):
            for seq in (128, 256, 512):
                c = Config(target_params=target, seq_len=seq)
                if matches(c):
                    cfg = c
                    break
            if cfg:
                break
    if cfg is None:
        raise RuntimeError("cannot infer architecture from checkpoint")

    m = CogniCore(cfg)
    m.load(path)
    print(f"{m.n_params():,} parameters")
    print(f"  {cfg}")
    return m, cfg


def load_any(paths, quiet=False):
    """Carga el PRIMER checkpoint que se pueda cargar de `paths`.

    Necesario porque un .npz exportado por la version JAX/Flax usa los nombres
    de parametros de Flax ("LCBlock_0/L0_n1") y NO es compatible con este motor
    NumPy, aunque sea el mas reciente. Antes se abortaba al primer fallo.
    Convierte primero con:  py convert_checkpoint.py <ckpt.npz>
    """
    errs = []
    for p in paths:
        try:
            return load(p)
        except Exception as e:
            if not quiet:
                print(f"FAILED - {e}")
            errs.append((p, e))
    print("\nNingun checkpoint de checkpoints/ se pudo cargar.")
    if errs:
        print("Intentados:")
        for p, e in errs:
            print(f"  {p.name}: {e}")
        print("\nSi el modelo se entreno en Colab con la version JAX, convierte "
              "primero el .npz:\n    py convert_checkpoint.py checkpoints\\<archivo>.npz")
    sys.exit(1)


def show_stats(m, cfg):
    print("\n" + "-" * 66)
    print("  architecture      CogniCore / HDS-LSM  (not an LLM)")
    print("  parameters        {:,}".format(m.n_params()))
    print("  layers            {} x d_model {}".format(cfg.n_layers, cfg.d_model))
    print("  HDE embedding     0 parameters (seeded hyperdimensional basis)")
    print("  attention         none - O(T) not O(T^2)")
    print("  KV cache          none - O(1) state per token, forever")
    print("  vocabulary        raw bytes 0-255, no tokenizer")
    print("  LCC memory        {} slots x {} dims = {:.1f} KB per layer, fixed".format(
        cfg.n_mem, cfg.d_mem, cfg.n_mem * cfg.d_mem * 4 / 1024))
    print("  context           {} BYTES (not tokens)".format(cfg.seq_len))
    print("-" * 66)


def surprisal(m, cfg, text):
    raw = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8).astype(np.int64)
    L = cfg.seq_len
    if len(raw) < 2:
        return None
    rows = []
    for i in range(0, max(1, len(raw) - 1), L):
        chunk = raw[i:i + L + 1]
        if len(chunk) < 2:
            break
        pad = L + 1 - len(chunk)
        chunk = np.pad(chunk, (0, pad), constant_values=10)
        ids, tgt = chunk.reshape(1, L + 1)[:, :-1], chunk.reshape(1, L + 1)[:, 1:]
        lg = m.forward(ids, refine=True).data[0]
        p = np.exp(lg - lg.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        for t, t_id in enumerate(tgt[0]):
            if i + t >= len(raw) - 1:
                break
            rows.append((int(raw[i + t]), float(-np.log(p[t, t_id] + 1e-12))))
    return rows


def main():
    if len(sys.argv) > 1 and sys.argv[1].endswith(".npz"):
        paths = [Path(sys.argv[1])]
    else:
        paths = find_checkpoints()
    if not paths:
        print("No checkpoint found in checkpoints/.")
        print("  train one:   py quicktest.py")
        print("          or:  py run_all.py")
        return
    m, cfg = load_any(paths)
    show_stats(m, cfg)

    temp, topk, nbytes = 0.75, 40, 300
    seed = 0
    chunk = cfg.seq_len
    print("\ntype a prompt and press enter.  :quit to exit, :stats for info\n")

    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if line.strip() in (":q", ":quit", ":exit"):
            break
        if line.strip() == ":stats":
            show_stats(m, cfg)
            continue
        if line.startswith(":"):
            parts = line.split()
            cmd = parts[0]
            try:
                if cmd == ":temp" and len(parts) > 1:
                    temp = float(parts[1]); print(f"temperature = {temp}")
                elif cmd == ":topk" and len(parts) > 1:
                    topk = int(parts[1]); print(f"top_k = {topk}")
                elif cmd == ":n" and len(parts) > 1:
                    nbytes = int(parts[1]); print(f"bytes = {nbytes}")
                elif cmd == ":seed" and len(parts) > 1:
                    seed = int(parts[1]); print(f"seed = {seed}")
                elif cmd == ":chunk" and len(parts) > 1:
                    # Ventana de contexto en BYTES. El SSM decae exponencialmente,
                    # asi que una ventana mas corta apenas cambia la salida pero
                    # reduce mucho el coste: ~180 ms/byte con 256, ~105 con 128,
                    # ~88 con 64 (medido en CPU, 10M params).
                    chunk = max(16, int(parts[1]))
                    print(f"ventana = {chunk} bytes")
                elif cmd == ":quant":
                    qreport(quantize(m, bits=8))
                elif cmd == ":save" and len(parts) > 1:
                    out = G.sample(m, line, nbytes, temp, topk, seed, log=lambda *a: None,
                            chunk=chunk)
                    Path(parts[1]).write_text(out, encoding="utf-8")
                    print(f"wrote {parts[1]}")
                elif cmd == ":compare" and len(parts) > 1:
                    rows = surprisal(m, cfg, " ".join(parts[1:]))
                    if rows:
                        tot = sum(r[1] for r in rows) / len(rows)
                        print(f"mean surprisal {tot:.3f} nats/byte "
                              f"({tot/np.log(2):.3f} bits/byte) over {len(rows)} bytes")
                        worst = sorted(rows, key=lambda r: -r[1])[:8]
                        print("  most surprising bytes: " + ", ".join(
                            f"{repr(chr(b)) if 32 <= b < 127 else hex(b)}:"
                            f"{s:.2f}" for b, s in worst))
                else:
                    print("commands: :temp :topk :n :seed :chunk :quant :save <f> "
                          ":compare <text> :stats :quit")
            except Exception as e:
                print(f"error: {e}")
            continue

        # normal prompt -> continue it
        try:
            out = G.sample(m, line, nbytes, temp, topk, seed, log=lambda *a: None,
                        chunk=chunk)
            print("\n" + "-" * 66)
            print(out)
            print("-" * 66)
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")

    print("bye")


if __name__ == "__main__":
    main()
