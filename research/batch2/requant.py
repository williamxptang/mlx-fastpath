"""N2: requantize OptiQ's 8-bit layers to 4-bit (affine, group 64), streaming, so memory stays ~0.5 GB.

OptiQ keeps 132 "sensitive" layers at 8 bits (5.2 bits/weight on average). Every token reads them, so 4-bit copies cut
the bytes per token (results.md §9: 6.0 GB -> 4.5 GB with variant `all`). 2609.04098 found 4-bit Gated DeltaNet
layers lossless on a Qwen hybrid. Whether the rest is: the KL gate (kl_gate.py) decides.

Variants (embed_tokens always stays 8-bit: it is a row lookup, ~0 bytes read per token):
  all       every 8-bit layer
  gdn_head  the linear-attention (Gated DeltaNet) layers and the LM head only
  gdn_head_mlp  gdn_head plus the MLPs: every 8-bit layer but full attention

Each 8-bit weight is dequantized (bf16, exactly as the model computes it) and quantized to 4 bits with mx.quantize,
in row chunks, written straight into the new safetensors files. Every other tensor is copied byte for byte; other
files are APFS clones (no extra disk). The output gets the same shard names and key assignment as the source.

Usage:
  python requant.py convert SRC DST {all,gdn_head}     (DST must not exist; written to DST.partial, then renamed)
  python requant.py check SRC DST                      (exit 1 with a list of problems)
"""
import hashlib
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

GROUP = 64
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
def selected(config, variant):
    q = config["quantization"]
    eight = [m for m, v in q.items() if isinstance(v, dict) and v.get("bits") == 8 and m != EMBED]
    if variant == "all":
        return sorted(eight)
    if variant == "gdn_head":
        return sorted(m for m in eight if ".linear_attn." in m or m == "language_model.lm_head")
    if variant == "gdn_head_mlp":   # everything but the full-attention layers (self_attn stays 8-bit)
        return sorted(m for m in eight if ".self_attn." not in m)
    raise SystemExit(f"unknown variant {variant}")


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


# ---------- convert ----------
def requant_rows(path, start, h, m, r0, r1):
    import mlx.core as mx
    w, s, b = (read_rows(path, start, h[f"{m}.{x}"], r0, r1) for x in ("weight", "scales", "biases"))
    dq = mx.dequantize(w, s, b, group_size=GROUP, bits=8)
    w4, s4, b4 = mx.quantize(dq, group_size=GROUP, bits=4)
    mx.eval(w4, s4, b4)
    assert s4.dtype == s.dtype and b4.dtype == b.dtype, (s4.dtype, s.dtype)
    return w4, s4, b4


def convert_shard(src_path, dst_path, modules, rows_fn=None):
    rows_fn = rows_fn or requant_rows
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
                    w4, s4, b4 = rows_fn(src_path, start, h, base, r0, min(r0 + rows, info["shape"][0]))
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


def convert(src, dst, variant, modules=None, rows_fn=None, what=None):
    """modules / rows_fn: another quantizer's layer list and 4-bit rows (default: RTN of the 8-bit values)."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists: not overwriting")
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        raise SystemExit(f"{tmp} exists from an earlier attempt: move it to the Trash first")
    config = json.load(open(src / "config.json"))
    modules = modules or selected(config, variant)
    assert modules, "nothing to convert"
    idx, by = shards(src)
    for m in modules:
        where = {idx["weight_map"].get(f"{m}.{x}") for x in ("weight", "scales", "biases")}
        if len(where) != 1 or None in where:
            raise SystemExit(f"{m}: weight/scales/biases not in one shard ({where}); not supported")
    tmp.mkdir()
    t0 = time.time()
    for shard in sorted(by):
        convert_shard(src / shard, tmp / shard, modules, rows_fn)
        print(f"{shard} written ({time.time() - t0:.0f}s)", flush=True)
    # index with the new total size
    total = 0
    for shard in by:
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
        meta["n2_requant"] = {"variant": variant, "modules": len(modules), "source": str(src)}
        json.dump(meta, open(tmp / "optiq_metadata.json", "w"), indent=2)
    skip = set(by) | {"model.safetensors.index.json", "config.json", "optiq_metadata.json"}
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir() or str(rel) in skip or any(part.startswith(".cache") for part in rel.parts):
            continue
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-c", str(p), str(tmp / rel)], check=True)   # APFS clone
    json.dump({"what": what or "OptiQ 8-bit layers requantized to 4-bit (N2), local research copy",
               "variant": variant, "source": str(src), "script": str(Path(__file__).resolve()),
               "date": time.strftime("%Y-%m-%d %H:%M"), "converted": modules},
              open(tmp / "N2_manifest.json", "w"), indent=2)
    tmp.rename(dst)
    print(json.dumps({"converted": len(modules), "dst": str(dst), "seconds": round(time.time() - t0)}))


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


def check(src, dst, sample_rows=256, exact=True):
    """Problems found in DST against SRC, as a list of strings (empty = good). exact=False (GPTQ output): skip the
    row-equals-RTN rule; everything else (config, shapes, parser, untouched bytes) is still checked."""
    import mlx.core as mx
    from safetensors import safe_open
    src, dst = Path(src), Path(dst)
    bad = []
    cs, cd = json.load(open(src / "config.json")), json.load(open(dst / "config.json"))
    qs, qd = cs["quantization"], cd["quantization"]
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
    for shard in sorted(by_s):
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
        for m in (modules if exact else []):
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


def rel_error(src, dst, sample_rows=64):
    """Per converted layer: ||w4 - w8|| / ||w8|| over sampled rows (information, not a pass/fail)."""
    import mlx.core as mx
    src, dst = Path(src), Path(dst)
    qs, qd = (json.load(open(p / "config.json"))["quantization"] for p in (src, dst))
    out = {}
    for shard in sorted(shards(src)[1]):
        hs, ss, _ = read_header(src / shard)
        hd, sd, _ = read_header(dst / shard)
        for m in sorted(m for m in qd if isinstance(qd[m], dict) and qd[m] != qs.get(m) and f"{m}.weight" in hs):
            n = hs[f"{m}.weight"]["shape"][0]
            num = den = 0.0
            for r in sorted({int(x) for x in np.linspace(0, n - 1, min(n, sample_rows))}):
                a8 = mx.dequantize(*(read_rows(src / shard, ss, hs[f"{m}.{x}"], r, r + 1) for x in ("weight", "scales", "biases")),
                                   group_size=GROUP, bits=8).astype(mx.float32)
                a4 = mx.dequantize(*(read_rows(dst / shard, sd, hd[f"{m}.{x}"], r, r + 1) for x in ("weight", "scales", "biases")),
                                   group_size=GROUP, bits=4).astype(mx.float32)
                num += float(mx.sum((a4 - a8) ** 2)); den += float(mx.sum(a8 ** 2))
            out[m] = round((num / max(den, 1e-30)) ** 0.5, 4)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "convert":
        import mlx_fastpath as fp
        with fp.MemoryGuard(min_free_gb=5, max_footprint_gb=4):
            import mlx.core as mx
            mx.set_memory_limit(3 << 30)
            mx.set_cache_limit(256 << 20)
            convert(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "check":
        import mlx_fastpath as fp
        with fp.MemoryGuard(min_free_gb=5, max_footprint_gb=4):
            problems = check(sys.argv[2], sys.argv[3])
        print(json.dumps({"problems": len(problems), "list": problems[:50]}, indent=1))
        sys.exit(1 if problems else 0)
    else:
        raise SystemExit(__doc__)
