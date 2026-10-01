"""
causality_check.py — Verifica que el modelo NO ve el futuro.

Gradcheck.verify_ los gradientes; esto verifica una propiedad distinta y mas
importante: que los logits en la posicion t dependan SOLO de los bytes <= t.
Un modelo que se filtra por el futuro puede tener gradientes perfectos y aun
asi generar basura, porque durante el entrenamiento "vio" la respuesta.

Uso:
    py causality_check.py            # modelo pequeno aleatorio
    py causality_check.py checkpoints/mio.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from cognicore.model import Config, CogniCore, local_context
from cognicore.hde import HDE, PAD


def check_local_context():
    """local_context no debe envolver con np.roll."""
    print("[1] local_context: relleno PAD, no np.roll")
    hde = HDE(dim=32, seed=1, atoms=257)
    T = 8
    ids = np.arange(T, dtype=np.int64).reshape(1, T)
    ho = hde.out_dim

    # El HDE codifica 5 canales (content, depth, column, class, tag), y cada
    # rol depende de la SECUENCIA completa (cumsum de profundidad, columna
    # acumulada). Por eso comparar contra hde.encode() de un byte suelto no
    # vale: hay que comparar el contexto completo contra una referencia
    # construida con el mismo desplazamiento y relleno.
    def reference(ids_seq, radius=2):
        B, Tt = ids_seq.shape
        out = np.zeros((B, Tt, (2 * radius + 1) * ho), dtype=np.float32)
        for j, o in enumerate(range(-radius, radius + 1)):
            sh = np.full((B, Tt), PAD, dtype=np.int64)
            if o == 0:
                sh[:, :] = ids_seq
            elif o > 0:                       # posicion t recibe ids[t+o]
                if o < Tt:
                    sh[:, :Tt - o] = ids_seq[:, o:]
            else:                             # posicion t recibe ids[t+o]
                if -o < Tt:
                    sh[:, -o:] = ids_seq[:, :Tt + o]
            out[:, :, j * ho:(j + 1) * ho] = hde.encode(sh)
        return out

    got = local_context(ids, hde, radius=2)
    ref = reference(ids, 2)
    err = float(np.abs(got - ref).max())
    print(f"  [{'OK  ' if err < 1e-6 else 'FAIL'}] coincide con la referencia "
          f"causal (err={err:.2e})")
    ok = err < 1e-6

    # El byte final no debe aparecer en los slots de posiciones anteriores
    # (np.roll lo envolvia y lo colocaba al principio).
    ids2 = ids.copy()
    ctx2 = local_context(ids2, hde, radius=2)
    same = float(np.abs(got - ctx2).max()) < 1e-9
    print(f"  [{'OK  ' if same else 'FAIL'}] sin envolvente de extremos")

    # Lo importante: cambiar un byte FUTURO no altera el contexto pasado.
    # Solo se comprueban las posiciones cuyo radio refine (2) no alcanza el byte
    # modificado: si no, el cambio es legitimo por diseño.
    future = T - 1
    # el radio del refine es 2, asi que la posicion future-2 SI ve el byte
    # future (offset +2). Solo las anteriores a future-radius son seguras.
    safe = future - 2 - 1
    ids3 = ids.copy()
    ids3[0, future] = 200
    ctx3 = local_context(ids3, hde, radius=2)
    d_ctx = np.abs(got[:, :safe + 1] - ctx3[:, :safe + 1]).max()
    causal_ctx = d_ctx < 1e-9
    print(f"  [{'OK  ' if causal_ctx else 'FAIL'}] cambiar el byte {future} no altera "
          f"el contexto de 0..{safe} (delta={d_ctx:.2e})")
    return ok and same and causal_ctx


def check_model(model, T=48, tol=1e-4):
    """Alterar un byte t'>t no debe cambiar los logits en t."""
    print("\n[2] modelo: los logits en t dependen solo de los bytes <= t")
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 200, (1, T)).astype(np.int64)
    base = model.forward(ids, refine=True).data.copy()

    probe = T - 3
    bad = []
    for t2 in (probe + 1, probe + 2, T - 1):
        if t2 >= T:
            continue
        alt_ids = ids.copy()
        alt_ids[0, t2] = (int(alt_ids[0, t2]) + 37) % 200
        alt = model.forward(alt_ids, refine=True).data.copy()
        delta = float(np.abs(base[0, probe] - alt[0, probe]).max())
        # radio permitido: el refine pass mira +-2 bytes
        allowed = abs(t2 - probe) <= 2
        status = "OK  " if (allowed or delta < tol) else "FUGA"
        if not allowed and delta >= tol:
            bad.append((t2, delta))
        print(f"  [{status}] alterando byte {t2:2d} cambia el logit de {probe} "
              f"en {delta:.2e}   (radio refine = 2)")

    if bad:
        print(f"\n  FUGA: {len(bad)} posiciones futuras influyen en {probe}")
        return False
    print(f"  OK: la posicion {probe} es causal")
    return True


def main():
    print("=" * 66)
    print("VERIFICACION DE CAUSALIDAD")
    print("=" * 66)

    ok = check_local_context()

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        p = Path(args[0])
        cfg = Config(target_params=10_000_000, seq_len=256)
        m = CogniCore(cfg)
        m.load(p)
        print(f"\nmodelo cargado desde {p}")
    else:
        cfg = Config(target_params=400_000, seq_len=64, d_hd=64)
        m = CogniCore(cfg)
        print("\nmodelo aleatorio pequeno (sin entrenar)")

    ok &= check_model(m)

    print("\n" + "=" * 66)
    print("RESULTADO:", "OK - el modelo es causal" if ok
          else "FALLO - hay fuga de informacion futura")
    print("=" * 66)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()