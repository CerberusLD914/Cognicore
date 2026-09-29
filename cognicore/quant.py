"""
quant.py — post-training integer quantisation.

Why this is easy here and painful in a transformer:

  A transformer is mostly matmuls, and matmul is fine with int8 GEMM — but
  LayerNorm, softmax and the KV cache all want floats, and the embedding
  table is a huge int8 array that has to be dequantised on every forward
  pass. Net result: you usually keep the model in fp16 and gain little.

  CogniCore's hot loop is a RECURRENCE: elementwise multiply-accumulate over
  a fixed state. That maps onto integer arithmetic exactly — the state
  stays in int8 the whole time, and only the (small) HDE read-out ever
  touches float32. So int8 here is a genuine 4x memory cut with no
  per-token dequantisation of weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def quantize_tensor(w: np.ndarray, bits: int = 8):
    """Symmetric per-tensor (or per-row for matrices) integer quantisation."""
    if bits >= 16:
        return w.astype(np.float32), None, 0
    if w.ndim >= 2:
        amax = np.abs(w).max(axis=tuple(range(1, w.ndim)), keepdims=True)
    else:
        amax = np.abs(w).max()
    amax = np.maximum(amax, 1e-8)
    qmax = 2 ** (bits - 1) - 1
    scale = (amax / qmax).astype(np.float32)
    q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int8 if bits == 8
                                                             else np.int16)
    return q, scale, bits


def dequantize(q, scale, bits):
    return (q.astype(np.float32) * scale).astype(np.float32)


def quantize(model, bits: int = 8, out: Path = None):
    """Quantise every weight >= 8 elements (norms/biases stay float)."""
    tables, total_q, total_f = {}, 0, 0
    for p in model.params():
        w = p.data
        n = w.size
        if n < 8 or w.ndim == 1:            # 1-D: norms, A, C, biases
            tables[p.name] = ("f32", w.astype(np.float32))
            total_f += n
            continue
        q, s, b = quantize_tensor(w, bits)
        tables[p.name] = (f"i{b}", q, s)
        total_q += n

    # error
    err = 0.0
    for p in model.params():
        rec = tables[p.name]
        if rec[0] == "f32":
            continue
        w = dequantize(rec[1], rec[2], bits)
        err += float(((w - p.data) ** 2).sum())
    ref = sum(float((p.data ** 2).sum()) for p in model.params())
    rel = np.sqrt(err / max(ref, 1e-12))

    rep = {
        "bits": bits, "n_quantised": total_q, "n_float": total_f,
        "rel_rmse": float(rel),
        "size_f32_mb": (total_q + total_f) * 4 / 1e6,
        "size_int_mb": (total_q * bits / 8 + total_f * 4) / 1e6,
    }
    if out:
        blob = {}
        for name, rec in tables.items():
            if rec[0] == "f32":
                blob[name] = rec[1]
            else:
                blob[name + ".q"] = rec[1]
                blob[name + ".s"] = rec[2]
        np.savez_compressed(out, **blob)
        rep["file"] = str(out)
        rep["file_mb"] = out.stat().st_size / 1e6
    return rep


def report(rep):
    print(f"  bit width            : {rep['bits']}-bit")
    print(f"  weights quantised    : {rep['n_quantised']:,}")
    print(f"  left in float32     : {rep['n_float']:,}  "
          f"(norms, A, C, biases — 1-D tensors)")
    print(f"  size as fp32        : {rep['size_f32_mb']:.1f} MB")
    print(f"  size quantised      : {rep['size_int_mb']:.1f} MB  "
          f"({rep['size_f32_mb'] / max(rep['size_int_mb'], 1e-9):.2f}x smaller)")
    print(f"  reconstruction RMSE : {rep['rel_rmse']:.2e} (relative)")
    if "file" in rep:
        print(f"  file                : {rep['file']} ({rep['file_mb']:.1f} MB on disk)")


def load_quantized(path: Path, model):
    z = np.load(path)
    for p in model.params():
        key = p.name + ".q"
        if key in z:
            p.data = dequantize(z[key], z[p.name + ".s"], 8)
        elif p.name in z:
            p.data = z[p.name].astype(np.float32)
    return model
