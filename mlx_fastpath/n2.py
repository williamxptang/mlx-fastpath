"""N2-gdnhead: a copy of an OptiQ model whose 8-bit Gated DeltaNet layers and LM head are requantized to 4-bit.

Every token reads the weights, so fewer bytes per token is more speed. On Qwen3.5-9B-OptiQ-4bit this cuts the bytes
read per token from 6.02 to 5.09 GB. Measured (results.md §16-17): +14-16% plain and +14-17% MTP decoding, KL 0.009
(short prompts) and 0.054 (4k/16k) against the source, needle codes 3/3 on 13 of 14 documents (the source misses the
same one), the 800-token code file byte-identical. Caveat: its 16k needle pass rests on a 0.125-nat first-token
tie (§17.6). The other layer sets measured (all 8-bit layers; GDN + head + MLP, with or without GPTQ) failed
long-context recall and are not offered here.

Each 8-bit weight is dequantized (bf16, exactly as the model computes it) and quantized to 4 bits (affine, group 64)
in row chunks, streamed into new safetensors files (~0.5 GB of memory). embed_tokens stays 8-bit (a row lookup).
Every other tensor is copied byte for byte; other files are APFS clones.

  build(src)            -> <src>-N2-gdnhead (never overwrites; written to .partial, then renamed)
  check(src, dst)       -> list of problems (empty = good)
  resolve_model(path)   -> the path load() uses: the N2 copy if the switch is on (default) and the copy exists
CLI: mlx-fastpath n2 SRC
"""
import hashlib
import json
import os
import struct
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

GROUP = 64
SUFFIX = "-N2-gdnhead"
MANIFEST = "N2_manifest.json"
CHUNK = 16 << 20          # uint32 words of 8-bit weight per chunk: 4 values each, so 64M values = 128 MB of bf16
EMBED = "language_model.model.embed_tokens"
NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32, "I32": np.int32, "U8": np.uint8}
SIZE = {k: np.dtype(v).itemsize for k, v in NP.items()}


# ---------- safetensors, read and written directly (bf16 included, row ranges, no full-file loads) ----------
def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return h, 8 + n, meta


def read_rows(path, start, info, r0=None, r1=None):
    """Rows r0:r1 of a tensor as an mx.array (bf16 viewed back from uint16)."""
    import mlx.core as mx
    shape = info["shape"]
    row = int(np.prod(shape[1:], dtype=np.int64)) * SIZE[info["dtype"]] if len(shape) > 1 else SIZE[info["dtype"]]
    r0, r1 = (0, shape[0]) if r0 is None else (r0, r1)
    with open(path, "rb") as f:
        f.seek(start + info["data_offsets"][0] + r0 * row)
        buf = f.read((r1 - r0) * row)
    a = mx.array(np.frombuffer(buf, dtype=NP[info["dtype"]]).reshape([r1 - r0] + list(shape[1:])))
    return a.view(mx.bfloat16) if info["dtype"] == "BF16" else a


def to_bytes(a):
    import mlx.core as mx
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.array(a).tobytes()


def write_header(f, tensors, meta):
    """tensors: list of (key, dtype, shape) in file order. Returns {key: (begin, end)} relative to the data start."""
    off, h = 0, {}
    for k, dt, shape in tensors:
        n = int(np.prod(shape, dtype=np.int64)) * SIZE[dt]
        h[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps({"__metadata__": meta, **h} if meta else h, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    f.write(struct.pack("<Q", len(blob)) + blob)
    return {k: tuple(v["data_offsets"]) for k, v in h.items()}


# ---------- what to convert ----------
def selected(config):
    """The 8-bit GDN (linear-attention) layers and the LM head."""
    q = config.get("quantization") or {}
    return sorted(m for m, v in q.items() if isinstance(v, dict) and v.get("bits") == 8 and m != EMBED
                  and (".linear_attn." in m or m == "language_model.lm_head"))


def new_config(config, modules):
    c = json.loads(json.dumps(config))
    for key in ("quantization", "quantization_config"):
        if key in c:
            for m in modules:
                c[key][m] = {"bits": 4, "group_size": GROUP}
    return c


def shards(src):
    idx = json.load(open(src / "model.safetensors.index.json"))
    by = {}
    for k, s in idx["weight_map"].items():
        by.setdefault(s, []).append(k)
    return idx, by


def missing_shards(folder, by, modules):
    """Files the index lists that are not on disk. That is allowed only for a file holding none of the layers to
    convert: an optional sidecar left out of a download (e.g. optiq_vision.safetensors, the vision tower). The copy
    then lists it too and lacks it too, as its source does. A missing file that holds a layer to convert can't be
    converted, so it stops the build."""
    missing = sorted(s for s in by if not (folder / s).exists())
    for s in missing:
        hit = sorted({k.rsplit(".", 1)[0] for k in by[s]} & set(modules))
        if hit:
            raise SystemExit(f"{s}: listed in the index but not on disk, and it holds {len(hit)} layer(s) to convert "
                             f"(e.g. {hit[0]}); download it first")
    return missing


# ---------- convert ----------
def requant_rows(path, start, h, m, r0, r1):
    import mlx.core as mx
    w, s, b = (read_rows(path, start, h[f"{m}.{x}"], r0, r1) for x in ("weight", "scales", "biases"))
    dq = mx.dequantize(w, s, b, group_size=GROUP, bits=8)
    w4, s4, b4 = mx.quantize(dq, group_size=GROUP, bits=4)
    mx.eval(w4, s4, b4)
    assert s4.dtype == s.dtype and b4.dtype == b.dtype, (s4.dtype, s.dtype)
    return w4, s4, b4


def convert_shard(src_path, dst_path, modules):
    import mlx.core as mx
    h, start, meta = read_header(src_path)
    conv = {m for m in modules if f"{m}.weight" in h}
    plan = []
    # per layer: weight first, so its scales/biases (computed with it) are ready when their turn comes
    order = {"weight": 0, "scales": 1, "biases": 2}
    for k in sorted(h, key=lambda k: (k.rsplit(".", 1)[0], order.get(k.rsplit(".", 1)[-1], 3), k)):
        dt, shape = h[k]["dtype"], h[k]["shape"]
        if k.endswith(".weight") and k[:-7] in conv:
            assert dt == "U32" and shape[1] % 2 == 0, (k, dt, shape)
            shape = [shape[0], shape[1] // 2]          # 8-bit: 4 values per uint32 -> 4-bit: 8
        plan.append((k, dt, shape))
    with open(dst_path, "wb") as f:
        offs = write_header(f, plan, meta)
        data0 = f.tell()
        pending = {}                                    # converted scales/biases, written when their turn comes
        for k, dt, shape in plan:
            assert f.tell() - data0 == offs[k][0], k
            base = k.rsplit(".", 1)[0]
            if base in conv and k.endswith(".weight"):
                info = h[k]
                rows = max(1, CHUNK // info["shape"][1])
                parts_s, parts_b = [], []
                for r0 in range(0, info["shape"][0], rows):
                    w4, s4, b4 = requant_rows(src_path, start, h, base, r0, min(r0 + rows, info["shape"][0]))
                    f.write(to_bytes(w4))
                    parts_s.append(s4); parts_b.append(b4)
                pending[f"{base}.scales"] = to_bytes(mx.concatenate(parts_s))
                pending[f"{base}.biases"] = to_bytes(mx.concatenate(parts_b))
            elif base in conv:
                f.write(pending.pop(k))
            else:                                       # unchanged: raw byte copy
                b0, b1 = h[k]["data_offsets"]
                with open(src_path, "rb") as g:
                    g.seek(start + b0)
                    left = b1 - b0
                    while left:
                        chunk = g.read(min(left, 64 << 20))
                        f.write(chunk)
                        left -= len(chunk)
            assert f.tell() - data0 == offs[k][1], k
            mx.clear_cache()
        assert not pending, sorted(pending)


def build(src, dst=None):
    """Write the N2-gdnhead copy of the OptiQ model at src (default destination: <src>-N2-gdnhead). Returns it."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.name + SUFFIX)
    if dst.exists():
        raise SystemExit(f"{dst} exists: not overwriting")
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        raise SystemExit(f"{tmp} exists from an earlier attempt: move it to the Trash first")
    config = json.load(open(src / "config.json"))
    modules = selected(config)
    assert modules, "nothing to convert"
    idx, by = shards(src)
    for m in modules:
        where = {idx["weight_map"].get(f"{m}.{x}") for x in ("weight", "scales", "biases")}
        if len(where) != 1 or None in where:
            raise SystemExit(f"{m}: weight/scales/biases not in one shard ({where}); not supported")
    missing = missing_shards(src, by, modules)                 # before anything is written: a refusal leaves nothing
    present = [s for s in sorted(by) if s not in missing]
    for s in missing:
        print(f"{s}: listed in the index but not on disk, holds nothing to convert: left out, as in the source",
              flush=True)
    tmp.mkdir()
    t0 = time.time()
    for shard in present:
        convert_shard(src / shard, tmp / shard, modules)
        print(f"{shard} written ({time.time() - t0:.0f}s)", flush=True)
    # index with the new total size (of the files on disk)
    total = 0
    for shard in present:
        h, _, _ = read_header(tmp / shard)
        total += sum(v["data_offsets"][1] - v["data_offsets"][0] for v in h.values())
    json.dump({**idx, "metadata": {**idx.get("metadata", {}), "total_size": total}},
              open(tmp / "model.safetensors.index.json", "w"), indent=2)
    json.dump(new_config(config, modules), open(tmp / "config.json", "w"), indent=2)
    meta_p = src / "optiq_metadata.json"
    if meta_p.exists():
        meta = json.load(open(meta_p))
        for m in modules:
            if m in meta.get("per_layer", {}):
                meta["per_layer"][m] = {"bits": 4, "group_size": GROUP}
        meta["n2_requant"] = {"variant": "gdn_head", "modules": len(modules), "source": str(src)}
        json.dump(meta, open(tmp / "optiq_metadata.json", "w"), indent=2)
    skip = set(by) | {"model.safetensors.index.json", "config.json", "optiq_metadata.json", MANIFEST}
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir() or str(rel) in skip or any(part.startswith(".cache") for part in rel.parts):
            continue
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-c", str(p), str(tmp / rel)], check=True)   # APFS clone
    json.dump({"what": "OptiQ's 8-bit GDN layers and LM head requantized to 4-bit (N2-gdnhead), by mlx-fastpath",
               "variant": "gdn_head", "source": str(src), "script": str(Path(__file__).resolve()),
               "date": time.strftime("%Y-%m-%d %H:%M"), "converted": modules},
              open(tmp / MANIFEST, "w"), indent=2)
    tmp.rename(dst)
    print(json.dumps({"converted": len(modules), "dst": str(dst), "seconds": round(time.time() - t0)}), flush=True)
    return dst


# ---------- check ----------
def digest(path, start, info):
    b0, b1 = info["data_offsets"]
    hsh = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        f.seek(start + b0)
        left = b1 - b0
        while left:
            chunk = f.read(min(left, 64 << 20))
            hsh.update(chunk)
            left -= len(chunk)
    return hsh.hexdigest()


def check(src, dst, sample_rows=256):
    """Problems found in DST against SRC, as a list of strings (empty = good)."""
    import mlx.core as mx
    from safetensors import safe_open
    src, dst = Path(src), Path(dst)
    bad = []
    cs, cd = json.load(open(src / "config.json")), json.load(open(dst / "config.json"))
    qs, qd = cs.get("quantization"), cd.get("quantization")
    if not qs or not qd:
        return [f"config: no quantization in {'the source' if not qs else 'the copy'}"]
    if cd.get("quantization_config", qd) != qd:
        bad.append("config: quantization != quantization_config")
    modules = sorted(m for m in qd if isinstance(qd[m], dict) and qd[m] != qs.get(m))
    for m in modules:
        if not (qs[m].get("bits") == 8 and qd[m] == {"bits": 4, "group_size": GROUP}) or m == EMBED:
            bad.append(f"config: {m} {qs[m]} -> {qd[m]}")
    if {k: v for k, v in cd.items() if k not in ("quantization", "quantization_config")} != \
       {k: v for k, v in cs.items() if k not in ("quantization", "quantization_config")}:
        bad.append("config: keys other than quantization changed")
    if not modules:
        bad.append("config: no layer converted")
    idx_s, by_s = shards(src)
    idx_d, by_d = shards(dst)
    if idx_s["weight_map"] != idx_d["weight_map"]:
        bad.append("index: weight_map differs")
    skip = set()
    for shard in sorted(by_s):
        in_src, in_dst = (src / shard).exists(), (dst / shard).exists()
        if in_src and in_dst:
            continue
        if in_src or in_dst:
            bad.append(f"{shard}: missing in the {'copy' if in_src else 'source'}")
        elif {k.rsplit('.', 1)[0] for k in by_s[shard]} & set(modules):
            bad.append(f"{shard}: missing in both, and it holds converted layers")
        skip.add(shard)                     # missing in both, nothing converted in it: an optional sidecar, as built
    for shard in sorted(set(by_s) - skip):
        hs, ss, _ = read_header(src / shard)
        hd, sd, _ = read_header(dst / shard)
        with safe_open(dst / shard, framework="numpy") as so:        # the official parser accepts the file
            if set(so.keys()) != set(hs):
                bad.append(f"{shard}: keys differ")
                continue
            for k in hs:
                if list(so.get_slice(k).get_shape()) != hd[k]["shape"]:
                    bad.append(f"{k}: header shape disagrees with the parser")
        for k in sorted(hs):
            base = k.rsplit(".", 1)[0]
            want = list(hs[k]["shape"])
            if base in modules and k.endswith(".weight"):
                want = [want[0], want[1] // 2]
            if hd[k]["dtype"] != hs[k]["dtype"] or hd[k]["shape"] != want:
                bad.append(f"{k}: {hd[k]['dtype']}{hd[k]['shape']} expected {hs[k]['dtype']}{want}")
            elif base not in modules and digest(src / shard, ss, hs[k]) != digest(dst / shard, sd, hd[k]):
                bad.append(f"{k}: bytes changed but the layer was not converted")
        for m in modules:
            if f"{m}.weight" not in hs:
                continue
            n = hs[f"{m}.weight"]["shape"][0]
            for r in sorted({int(x) for x in np.linspace(0, n - 1, min(n, sample_rows))}):
                w8, s8, b8 = (read_rows(src / shard, ss, hs[f"{m}.{x}"], r, r + 1) for x in ("weight", "scales", "biases"))
                w4, s4, b4 = (read_rows(dst / shard, sd, hd[f"{m}.{x}"], r, r + 1) for x in ("weight", "scales", "biases"))
                # rows quantize independently, so each stored row must be exactly MLX's 4-bit quantization of the
                # 8-bit row's values (the values the original model computes with)
                want = mx.quantize(mx.dequantize(w8, s8, b8, group_size=GROUP, bits=8), group_size=GROUP, bits=4)
                if not all(bool(mx.array_equal(a, b)) for a, b in zip((w4, s4, b4), want)):
                    bad.append(f"{m} row {r}: not mx.quantize(4-bit) of the 8-bit values")
                    break
    return bad


# ---------- the load() switch ----------
def enabled(n2=None):
    """The switch: an explicit argument wins, else MLX_FASTPATH_N2 (default on; 0 / false / off / no turn it off)."""
    if n2 is not None:
        return bool(n2)
    return os.environ.get("MLX_FASTPATH_N2", "1").strip().lower() not in ("0", "false", "off", "no")


def resolve_model(path, n2=None):
    """The model folder to load: the finished N2-gdnhead copy next to path when the switch is on, else path."""
    path = Path(path)
    if not enabled(n2) or path.name.endswith(SUFFIX):
        return path
    copy = path.with_name(path.name + SUFFIX)
    if (copy / MANIFEST).exists():
        return copy
    if (path / "config.json").exists() and selected(json.load(open(path / "config.json"))):
        warnings.warn(f"mlx_fastpath: no N2-gdnhead copy of {path.name} (+14-22% decode); build it once with "
                      f"`mlx-fastpath n2 {path}`, or turn this off with n2=False / MLX_FASTPATH_N2=0.", stacklevel=3)
    return path
