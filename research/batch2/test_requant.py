"""requant.py on a tiny fake OptiQ folder (random weights, a few MB). Run:
<venv>/bin/python -m pytest -q -p no:cacheprovider test_requant.py"""
import json
import subprocess

import mlx.core as mx
import numpy as np
import pytest

import requant

IN, OUT = 256, 70
EIGHT = ["language_model.lm_head", "language_model.model.layers.0.linear_attn.out_proj",
         "language_model.model.layers.0.mlp.up_proj", "language_model.model.embed_tokens",
         "language_model.model.layers.3.self_attn.q_proj"]
FOUR = ["language_model.model.layers.0.mlp.gate_proj"]


def make_src(root):
    mx.random.seed(1)
    src = root / "src"
    (src / "optiq").mkdir(parents=True)
    q = {"group_size": 64, "bits": 4, "mode": "affine", **{m: {"bits": 8, "group_size": 64} for m in EIGHT},
         **{m: {"bits": 4, "group_size": 64} for m in FOUR}}
    json.dump({"model_type": "fake", "quantization": q, "quantization_config": q}, open(src / "config.json", "w"))
    tensors = {}
    for m in EIGHT + FOUR:
        w = (mx.random.normal((OUT, IN)) * 0.05).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=8 if m in EIGHT else 4)
        tensors.update({f"{m}.weight": wq, f"{m}.scales": s, f"{m}.biases": b})
    tensors["language_model.model.norm.weight"] = mx.ones((IN,), dtype=mx.bfloat16)
    tensors["language_model.model.layers.0.linear_attn.A_log"] = mx.random.normal((8,))
    keys = sorted(tensors)   # a layer's weight/scales/biases share a shard, as in the real model
    first = {f"{m}.{x}" for m in (EIGHT[0], EIGHT[2], FOUR[0]) for x in ("weight", "scales", "biases")}
    split = {"model-00001-of-00002.safetensors": [k for k in keys if k in first] + keys[-1:],
             "model-00002-of-00002.safetensors": [k for k in keys[:-1] if k not in first]}
    for name, ks in split.items():
        mx.save_safetensors(str(src / name), {k: tensors[k] for k in ks}, metadata={"format": "mlx"})
    json.dump({"metadata": {"total_size": 0}, "weight_map": {k: n for n, ks in split.items() for k in ks}},
              open(src / "model.safetensors.index.json", "w"))
    json.dump({"per_layer": {m: q[m] for m in EIGHT + FOUR}}, open(src / "optiq_metadata.json", "w"))
    (src / "tokenizer.json").write_text("{}")
    (src / "optiq" / "mtp.safetensors").write_bytes(b"x" * 100)
    return src


@pytest.fixture
def src(tmp_path, monkeypatch):
    monkeypatch.setattr(requant, "CHUNK", 3 * IN // 4)   # 3 rows per chunk: the 70-row layers take 24 chunks
    return make_src(tmp_path)


def test_all_converts_every_8bit_layer_but_the_embedding_and_passes_check(src, tmp_path):
    dst = tmp_path / "all"
    requant.convert(src, dst, "all")
    assert requant.check(src, dst) == []
    q = json.load(open(dst / "config.json"))["quantization"]
    assert {m for m in EIGHT if q[m]["bits"] == 4} == set(EIGHT) - {requant.EMBED}
    assert q[requant.EMBED]["bits"] == 8
    # MLX's own reader sees the 4-bit layer, and it is the 4-bit quantization of the 8-bit layer's values
    s_all = {**mx.load(str(src / "model-00001-of-00002.safetensors")), **mx.load(str(src / "model-00002-of-00002.safetensors"))}
    d_all = {**mx.load(str(dst / "model-00001-of-00002.safetensors")), **mx.load(str(dst / "model-00002-of-00002.safetensors"))}
    m = "language_model.lm_head"
    w8 = mx.dequantize(*(s_all[f"{m}.{x}"] for x in ("weight", "scales", "biases")), group_size=64, bits=8)
    want = mx.quantize(w8, group_size=64, bits=4)
    for x, a in zip(("weight", "scales", "biases"), want):
        assert mx.array_equal(d_all[f"{m}.{x}"], a), x
    assert json.load(open(dst / "optiq_metadata.json"))["per_layer"][m]["bits"] == 4
    assert (dst / "optiq" / "mtp.safetensors").read_bytes() == b"x" * 100


def test_gdn_head_converts_only_linear_attention_and_the_head(src, tmp_path):
    dst = tmp_path / "gdn"
    requant.convert(src, dst, "gdn_head")
    assert requant.check(src, dst) == []
    q = json.load(open(dst / "config.json"))["quantization"]
    assert {m for m in EIGHT if q[m]["bits"] == 4} == {"language_model.lm_head",
                                                        "language_model.model.layers.0.linear_attn.out_proj"}


def test_gdn_head_mlp_converts_all_but_full_attention(src, tmp_path):
    dst = tmp_path / "mid"
    requant.convert(src, dst, "gdn_head_mlp")
    assert requant.check(src, dst) == []
    q = json.load(open(dst / "config.json"))["quantization"]
    assert {m for m in EIGHT if q[m]["bits"] == 4} == {"language_model.lm_head",
                                                        "language_model.model.layers.0.linear_attn.out_proj",
                                                        "language_model.model.layers.0.mlp.up_proj"}


def test_check_reports_a_changed_byte_in_a_layer_that_was_not_converted(src, tmp_path):
    dst = tmp_path / "all"
    requant.convert(src, dst, "all")
    key = "language_model.model.layers.0.mlp.gate_proj.weight"
    shard = dst / json.load(open(dst / "model.safetensors.index.json"))["weight_map"][key]
    h, start, _ = requant.read_header(shard)
    raw = bytearray(shard.read_bytes())
    raw[start + h[key]["data_offsets"][0]] ^= 1
    shard.write_bytes(bytes(raw))
    assert any(key in p and "bytes changed" in p for p in requant.check(src, dst))


def test_check_reports_4bit_codes_that_are_not_the_nearest(src, tmp_path):
    dst = tmp_path / "all"
    requant.convert(src, dst, "all")
    key = "language_model.model.layers.0.mlp.up_proj.weight"
    shard = dst / json.load(open(dst / "model.safetensors.index.json"))["weight_map"][key]
    h, start, _ = requant.read_header(shard)
    raw = bytearray(shard.read_bytes())
    b0, b1 = h[key]["data_offsets"]
    words = np.frombuffer(bytes(raw[start + b0:start + b1]), dtype=np.uint32).copy()
    words ^= 0x8                                  # one 4-bit code per word moves by 8 steps
    raw[start + b0:start + b1] = words.tobytes()
    shard.write_bytes(bytes(raw))
    assert any("up_proj" in p and "not mx.quantize" in p for p in requant.check(src, dst))


def test_convert_refuses_an_existing_destination(src, tmp_path):
    dst = tmp_path / "exists"
    dst.mkdir()
    with pytest.raises(SystemExit):
        requant.convert(src, dst, "all")


def test_clones_are_independent_copies(src, tmp_path):
    dst = tmp_path / "all"
    requant.convert(src, dst, "all")
    (dst / "tokenizer.json").write_text("changed")
    assert (src / "tokenizer.json").read_text() == "{}"
    assert subprocess.run(["cmp", "-s", src / "optiq/mtp.safetensors", dst / "optiq/mtp.safetensors"]).returncode == 0
