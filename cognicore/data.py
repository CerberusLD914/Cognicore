"""
data.py — fully automatic dataset acquisition.

Priority order (first one that yields enough bytes wins):
  1. HuggingFace Hub  — real pretraining corpora, streamed file by file so
     nothing big is ever downloaded whole.
  2. GitHub raw       — actual source trees (CPython stdlib + popular repos).
  3. Synthetic        — a grammar-generated corpus, so the pipeline can NEVER
                        fail on a machine with no network at all.

Everything is raw bytes. CogniCore has no tokenizer: the unit of training is
a byte, so corpora need no preprocessing, no BPE, no vocab file.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

UA = {"User-Agent": "cognicore/0.1 (low-resource pretraining)"}
PAD = 256          # id 256 is reserved for padding
NL = 10

# --------------------------------------------------------------------------
# source 1: HuggingFace datasets
# --------------------------------------------------------------------------
HF_SOURCES = [
    # (repo_id, config, split, format, language filter)
    ("codeparrot/github-code-clean", None, "train", "parquet", "Python"),
    ("codeparrot/codeparrot-clean-valid", None, "train", "parquet", "Python"),
    ("bigcode/the-stack-smol", "default", "train", "parquet", "Python"),
]

# --------------------------------------------------------------------------
# source 2: GitHub raw source trees
# --------------------------------------------------------------------------
GITHUB_SOURCES = [
    "https://raw.githubusercontent.com/python/cpython/main/Lib/{}.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/email/_{}.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/json/_{}.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/http/_{}.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/xmlrpc/_{}.py",
    "https://raw.githubusercontent.com/pallets/click/main/src/click/{}.py",
    "https://raw.githubusercontent.com/psf/requests/main/requests/{}.py",
    "https://raw.githubusercontent.com/numpy/numpy/main/numpy/lib/_{}.py",
    "https://raw.githubusercontent.com/pallets/flask/main/src/flask/{}.py",
    "https://raw.githubusercontent.com/tiangolo/fastapi/master/fastapi/{}.py",
    "https://raw.githubusercontent.com/python/cpython/main/Modules/_io/_{}.py",
    "https://raw.githubusercontent.com/scipy/scipy/main/scipy/_lib/_{}.py",
]

CPYTHON_MODULES = """os sys io abc math time json re socket string errno stat types
struct collections functools itertools operator copy pickle marshal zipfile
gzip bz2 lzma tarfile csv hashlib hmac base64 codecs mimetypes email smtplib
http urllib html xml logging config argparse gettext unittest doctest ast
dis inspect tokenize linecache traceback gc weakref contextlib dataclasses
enum typing pathlib shutil tempfile threading queue random secrets uuid
datetime calendar locale gettext bisect heapq array struct ctypes select
signal errno asynchat asyncore smtplib ftplib telnetlib imaplib poplib nntplib
xmlrpc sqlite3 dbm shelve pickle copyreg pty tty ttylib curses readline rlcompleter
site sysconfig venv zipapp distutils ensurepip pkgutil runpy pdb bdb profile
pstats cProfile trace timeit doctest test support idlelib turtledemo tkinter
multiprocessing concurrent subprocess selectors signal errno multiprocessing
concurrent
"""


def _get(url, timeout=30, retries=2):
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


# --------------------------------------------------------------------------
# parquet reading (no pyarrow dependency: use the raw column pages)
# --------------------------------------------------------------------------
def _read_parquet_bytes(blob, want_col=None, max_rows=20000, lang=None):
    """Extract string cells from a parquet file.

    Tries pyarrow if available (fast path) and, when the file carries a
    `language` column, filters by it so a mixed-language corpus yields only
    the language we asked for. Otherwise falls back to a dependency-free scan
    that mines long printable runs out of the dictionary-encoded pages and
    keeps only well-formed Python — the fallback exists so a low-resource
    machine needs no 40MB extra install.
    """
    # --- fast path -------------------------------------------------------
    try:
        import pyarrow.parquet as pq
        import io as _io
        f = pq.ParquetFile(_io.BytesIO(blob))
        names = f.schema_arrow.names
        col = want_col or next((c for c in names if c in
                                ("code", "content", "text", "raw_content")), names[0])
        cols = [col] + (["language"] if "language" in names else [])
        out = []
        for batch in f.iter_batches(batch_size=4096, columns=cols):
            d = batch.to_pydict()
            for i, txt in enumerate(d[col]):
                if not isinstance(txt, str) or not txt:
                    continue
                if lang and "language" in d:
                    if (d["language"][i] or "").lower() != lang.lower():
                        continue
                out.append(txt)
                if len(out) >= max_rows:
                    return "\n".join(out)
        return "\n".join(out)
    except ImportError:
        pass
    except Exception:
        pass

    # --- dependency-free fallback ----------------------------------------
    pat = re.compile(rb"[\x09\x0a\x0d\x20-\x7e]{40,}")
    runs = []
    for m in pat.finditer(blob):
        s = m.group().decode("utf-8", "ignore")
        if s.count("\n") >= 2:
            runs.append(s)
        if sum(len(r) for r in runs) > max_rows * 80:
            break
    return "\n".join(runs)


def fetch_huggingface(target_bytes, max_files=24, log=print):
    for repo, config, split, fmt, lang in HF_SOURCES:
        api = f"https://huggingface.co/api/datasets/{repo}"
        meta = _get(api)
        if not meta:
            continue
        log(f"  [HF] querying {repo} ...")
        try:
            info = json.loads(meta)
        except Exception:
            continue
        files = []
        for sib in info.get("siblings", []):
            n = sib.get("rfilename", "")
            if split in n and (n.endswith(".parquet") or n.endswith(".json") or
                               n.endswith(".jsonl") or n.endswith(".jsonl.gz")):
                files.append(n)
        if not files:
            continue
        files.sort()
        got, used = [], 0
        for fn in files[:max_files]:
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fn}"
            blob = _get(url, timeout=60)
            if not blob:
                continue
            if fn.endswith(".gz"):
                try:
                    blob = gzip.decompress(blob)
                except Exception:
                    continue
            if fn.endswith(".parquet"):
                text = _read_parquet_bytes(blob, want_col="code",
                                           max_rows=8000, lang=lang)
            elif fn.endswith(".jsonl"):
                text = "\n".join(
                    json.loads(l).get("content", "") for l in
                    blob.decode("utf-8", "ignore").splitlines()[:4000]
                    if l.strip())
            else:
                text = blob.decode("utf-8", "ignore")
            if lang == "Python":
                text = extract_python(text)
            if not text:
                continue
            got.append(text)
            used += len(text)
            log(f"        {fn:<52} +{len(text):>9,} B  (total {used:,})")
            if used >= target_bytes:
                break
        if used >= target_bytes * 0.5:
            return "\n".join(got)
    return ""


def fetch_github(target_bytes, log=print):
    mods = CPYTHON_MODULES.split()
    chunks, total = [], 0
    for tpl in GITHUB_SOURCES:
        for m in mods:
            url = tpl.format(m)
            blob = _get(url, timeout=20, retries=1)
            if not blob or len(blob) < 200:
                continue
            try:
                text = blob.decode("utf-8", "ignore")
            except Exception:
                continue
            if not _looks_like_code(text):
                continue
            chunks.append(f"# file: {url.rsplit('/', 1)[-1]}\n{text}")
            total += len(text)
            if total >= target_bytes:
                log(f"  [GH] {total:,} bytes of real source from {len(chunks)} files")
                return "\n".join(chunks)
    if total:
        log(f"  [GH] {total:,} bytes of real source from {len(chunks)} files")
    return "\n".join(chunks)


# --------------------------------------------------------------------------
# source 3: synthetic — a real program grammar, so the pipeline never dies
# --------------------------------------------------------------------------
def synth_corpus(target_bytes, seed=0, log=print):
    """Grammar-generated Python-shaped programs.

    Not a toy filler: it has real long-range structure (nesting, variable
    reuse across scopes, repeated call patterns), which is exactly what a
    recurrent state-space model has to learn. Useful for smoke tests and
    for machines with no network.
    """
    rng = np.random.default_rng(seed)
    IDENTS = ["data", "result", "count", "index", "value", "buffer", "node",
              "config", "total", "offset", "items", "cache", "payload", "state"]
    TYPES = ["int", "str", "float", "bool", "bytes", "list"]

    def expr(d=0):
        r = rng.random()
        if d > 3 or r < 0.34:
            return rng.choice(IDENTS + [str(int(rng.integers(0, 999))),
                                        f"{rng.integers(0, 99)}.{(rng.integers(0,99)):02d}",
                                        f'"{ "".join(rng.choice(["ab","cd","ef","x","run"]) for _ in range(rng.integers(1,5))) }"'])
        if r < 0.55:
            return f"{expr(d+1)} {rng.choice(['+','-','*','%','//'])} {expr(d+1)}"
        if r < 0.7:
            return f"len({expr(d+1)})"
        if r < 0.85:
            return f"max({expr(d+1)}, {expr(d+1)})"
        return f"round({expr(d+1)}, {rng.integers(1,4)})"

    def cond(d=0):
        v = rng.choice(IDENTS)
        return f"{v} {rng.choice(['>','<','>=','<=','==','!='])} {expr(2)}"

    def stmts(d, n):
        out = []
        for _ in range(n):
            r = rng.random()
            ind = "    " * d
            if r < 0.30 and d < 4:
                out.append(f"{ind}if {cond()}:")
                out += stmts(d + 1, rng.integers(1, 3))
            elif r < 0.48 and d < 4:
                out.append(f"{ind}for {rng.choice(IDENTS)} in range({rng.integers(2, 40)}):")
                out += stmts(d + 1, rng.integers(1, 3))
            elif r < 0.60:
                out.append(f"{ind}{rng.choice(IDENTS)} = {expr()}")
            elif r < 0.70:
                out.append(f"{ind}{rng.choice(IDENTS)}[{rng.integers(0,5)}] = {expr()}")
            elif r < 0.80:
                out.append(f"{ind}if {cond()}:")
                out.append(f"{ind}    continue")
            elif r < 0.90:
                out.append(f"{ind}print({expr()})")
            else:
                out.append(f"{ind}return {expr()}" if d == 0 else f"{ind}break")
        return out

    def fn():
        name = "_".join(rng.choice(["get","set","run","do","make","fetch","apply"])
                        for _ in range(1, 2)) + "_" + \
               "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"))
                       for _ in range(rng.integers(3, 8)))
        args = ", ".join(f"{rng.choice(IDENTS)}: {rng.choice(TYPES)}"
                         for _ in range(rng.integers(0, 4)))
        body = stmts(1, rng.integers(4, 12))
        if not any("return" in b for b in body):
            body.append(f"    return {expr()}")
        return [f"def {name}({args}) -> {rng.choice(TYPES)}:", *body, ""]

    parts, total = [], 0
    while total < target_bytes:
        block = []
        for _ in range(int(rng.integers(3, 12))):
            block.extend(fn())
        parts.append("\n".join(block))
        total += len(parts[-1])
    txt = "\n".join(parts)
    log(f"  [SYNTH] {len(txt):,} bytes of grammar-generated code (offline fallback)")
    return txt


# --------------------------------------------------------------------------
def _looks_like_code(text: str) -> bool:
    head = text[:4000]
    if not head:
        return False
    kw = sum(head.count(k) for k in
            ("def ", "return", "import ", "class ", "if ", "for ", "self."))
    return kw >= 6


# --- Python-only filtering ------------------------------------------------
# The dependency-free parquet fallback recovers raw text from mixed-language
# corpora, so we keep only well-formed Python. A block must have consistent
# indentation, balanced brackets, and real Python keywords.
_PY_STRONG = ("def ", "class ", "import ", "from ", "return", "self.",
              "elif ", "with ", "lambda ", "__")
_PY_WEAK = ("if ", "for ", "while ", "=", "(", ")", ":", "print(")
_PY_NEG = ("function ", "var ", "const ", "public ", "=>", "::", "</",
          "SELECT ", "#include", "package ", "func ", "impl ")


def _is_python_block(block: str) -> bool:
    if len(block) < 60:
        return False
    strong = sum(block.count(k) for k in _PY_STRONG)
    if strong < 2:
        return False
    neg = sum(block.count(k) for k in _PY_NEG)
    if neg > strong:
        return False
    weak = sum(block.count(k) for k in _PY_WEAK)
    if weak < 6:
        return False
    # indentation must be a multiple of 4 somewhere (Python convention) and
    # brackets must balance
    if block.count("(") - block.count(")") > 2:
        return False
    if block.count("[") - block.count("]") > 2:
        return False
    lines = block.split("\n")
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    if not indents:
        return False
    if not any(i % 4 == 0 for i in indents):
        return False
    if not any(i > 0 for i in indents):
        return False        # a real code block is not all flush-left
    if max(indents) > 24:
        return False
    return True


def extract_python(text: str, min_len: int = 200) -> str:
    """Pull well-formed Python blocks out of a noisy mixed-language stream.

    Splits on blank lines (function/class boundaries survive; prose and CSS
    fragments do not) and keeps only the blocks that look like real Python.
    """
    out, buf = [], []
    for ln in text.split("\n"):
        buf.append(ln)
        if not ln.strip() and len("\n".join(buf)) > 40:
            blk = "\n".join(buf)
            if _is_python_block(blk):
                out.append(blk)
            buf = []
    if buf:
        blk = "\n".join(buf)
        if _is_python_block(blk):
            out.append(blk)
    # fall back to keeping everything that at least looks like code
    if sum(len(o) for o in out) < min_len:
        out = [b for b in text.split("\n\n") if _looks_like_code(b)]
    return "\n".join(out)


def _filter_code_lines(text: str) -> str:
    keep = []
    for ln in text.splitlines():
        s = ln.strip()
        if (s.startswith(("def ", "class ", "import ", "from ", "return", "if ",
                         "for ", "while ", "    ", "}", ")", "#", "@")) or
                "=" in s or "(" in s):
            keep.append(ln)
    return "\n".join(keep)


# --------------------------------------------------------------------------
def load_corpus(target_bytes=8_000_000, cache="data/corpus.txt",
                allow_network=True, log=print, seed=0):
    """Return raw corpus text, downloading and caching automatically."""
    p = Path(cache)
    if p.exists() and p.stat().st_size > 20_000:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        log(f"[data] cache hit: {len(txt):,} bytes from {p}")
        return txt

    p.parent.mkdir(parents=True, exist_ok=True)
    txt = ""
    if allow_network:
        log(f"[data] target {target_bytes:,} bytes — trying HuggingFace ...")
        txt = fetch_huggingface(target_bytes, log=log)
        if len(txt) < target_bytes * 0.5:
            log("[data] HF thin, falling back to GitHub source trees ...")
            gh = fetch_github(target_bytes, log=log)
            if len(gh) > len(txt):
                txt = gh
    if len(txt) < 20_000:
        log("[data] using synthetic corpus (offline mode)")
        txt = synth_corpus(target_bytes, seed=seed, log=log)

    p.write_text(txt, encoding="utf-8")
    log(f"[data] wrote {len(txt):,} bytes -> {p}")
    return txt


def to_byte_tokens(text: str, seq_len: int, seed=0, pad=PAD):
    """Pack raw text into a (N, seq_len+1) int array.

    Layout: row i contains bytes[i*L : i*L + L+1]; the model is trained to
    predict byte t+1 from bytes <= t. Total tokens = total bytes, so there is
    no tokeniser overhead and no sequence is wasted on padding.
    """
    raw = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
    n = len(raw) // (seq_len + 1)
    if n < 8:
        raise ValueError("corpus too small for this seq_len")
    a = raw[: n * (seq_len + 1)].reshape(n, seq_len + 1).astype(np.int64)
    a[a > 255] = pad
    return a


def batches(a: np.ndarray, bs: int, rng=None, shuffle=True):
    idx = np.arange(len(a))
    if shuffle:
        (rng or np.random).permutation(idx)
    for i in range(0, len(idx) - bs + 1, bs):
        yield a[idx[i:i + bs]]


def causal_lm_batch(rows):
    """(B, L+1) -> (ids, targets) with the last column dropped."""
    return rows[:, :-1], rows[:, 1:]
