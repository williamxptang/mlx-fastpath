"""N2-gdnhead: building it (OptiQ's 8-bit GDN layers and LM head -> 4-bit, everything else byte-identical), checking
it, and the load() switch that uses it (default on). Tiny fake OptiQ folder: random weights, a few MB."""
import json
import warnings

import mlx.core as mx
import numpy as np
import pytest

from mlx_fastpath import n2

IN, OUT = 256, 70
EIGHT = ["language_model.lm_head", "language_model.model.layers.0.linear_attn.out_proj",
         "language_model.model.layers.0.mlp.up_proj", "language_model.model.embed_tokens",
         "language_model.model.layers.3.self_attn.q_proj"]
FOUR = ["language_model.model.layers.0.mlp.gate_proj"]
CONVERTED = {"language_model.lm_head", "language_model.model.layers.0.linear_attn.out_proj"}


def make_src(root, name="Model-OptiQ-4bit"):
    mx.random.seed(1)
    src = root / name
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
    keys = sorted(tensors)          # a layer's weight/scales/biases share a shard, as in the real model
    first = {f"{m}.{x}" for m in (EIGHT[0], EIGHT[2], FOUR[0]) for x in ("weight", "scales", "biases")}
    split = {"model-00001-of-00002.safetensors": [k for k in keys if k in first] + keys[-1:],
             "model-00002-of-00002.safetensors": [k for k in keys[:-1] if k not in first]}
    for fname, ks in split.items():
        mx.save_safetensors(str(src / fname), {k: tensors[k] for k in ks}, metadata={"format": "mlx"})
    json.dump({"metadata": {"total_size": 0}, "weight_map": {k: f for f, ks in split.items() for k in ks}},
              open(src / "model.safetensors.index.json", "w"))
    (src / "tokenizer.json").write_text("{}")
    (src / "optiq" / "mtp.safetensors").write_bytes(b"x" * 100)
    return src


@pytest.fixture
def src(tmp_path, monkeypatch):
    monkeypatch.setattr(n2, "CHUNK", 3 * IN // 4)       # 3 rows per chunk: each 70-row layer takes 24 chunks
    return make_src(tmp_path)


def test_build_converts_only_the_gdn_layers_and_the_head_and_passes_check(src):
    dst = n2.build(src)
    assert dst == src.with_name(src.name + n2.SUFFIX)
    assert n2.check(src, dst) == []
    q = json.load(open(dst / "config.json"))["quantization"]
    assert {m for m in EIGHT if q[m]["bits"] == 4} == CONVERTED
    s_all = {**mx.load(str(src / "model-00001-of-00002.safetensors")), **mx.load(str(src / "model-00002-of-00002.safetensors"))}
    d_all = {**mx.load(str(dst / "model-00001-of-00002.safetensors")), **mx.load(str(dst / "model-00002-of-00002.safetensors"))}
    m = "language_model.lm_head"                         # the 4-bit rounding of the 8-bit layer's values
    w8 = mx.dequantize(*(s_all[f"{m}.{x}"] for x in ("weight", "scales", "biases")), group_size=64, bits=8)
    for x, a in zip(("weight", "scales", "biases"), mx.quantize(w8, group_size=64, bits=4)):
        assert mx.array_equal(d_all[f"{m}.{x}"], a), x
    assert (dst / "optiq" / "mtp.safetensors").read_bytes() == b"x" * 100
    assert json.load(open(dst / n2.MANIFEST))["converted"] == sorted(CONVERTED)


def test_check_reports_a_changed_byte_in_a_layer_that_was_not_converted(src):
    dst = n2.build(src)
    key = "language_model.model.layers.0.mlp.gate_proj.weight"
    shard = dst / json.load(open(dst / "model.safetensors.index.json"))["weight_map"][key]
    h, start, _ = n2.read_header(shard)
    raw = bytearray(shard.read_bytes())
    raw[start + h[key]["data_offsets"][0]] ^= 1
    shard.write_bytes(bytes(raw))
    assert any(key in p and "bytes changed" in p for p in n2.check(src, dst))


def test_check_reports_4bit_codes_that_are_not_the_rounding(src):
    dst = n2.build(src)
    key = "language_model.lm_head.weight"
    shard = dst / json.load(open(dst / "model.safetensors.index.json"))["weight_map"][key]
    h, start, _ = n2.read_header(shard)
    raw = bytearray(shard.read_bytes())
    b0, b1 = h[key]["data_offsets"]
    words = np.frombuffer(bytes(raw[start + b0:start + b1]), dtype=np.uint32).copy()
    words ^= 0x8                                         # one 4-bit code per word moves by 8 steps
    raw[start + b0:start + b1] = words.tobytes()
    shard.write_bytes(bytes(raw))
    assert any("lm_head" in p and "not mx.quantize" in p for p in n2.check(src, dst))


def _list_a_missing_file(src, keys, fname="optiq_vision.safetensors"):
    """Index entries for tensors in a file that is not on disk (an optional sidecar left out of a download)."""
    p = src / "model.safetensors.index.json"
    idx = json.load(open(p))
    idx["weight_map"].update({k: fname for k in keys})
    json.dump(idx, open(p, "w"))


def test_a_listed_file_that_is_missing_and_holds_nothing_to_convert_is_left_out(src):
    _list_a_missing_file(src, ["vision_tower.blocks.0.attn.qkv.weight", "vision_tower.blocks.0.attn.qkv.bias"])
    dst = n2.build(src)
    assert n2.check(src, dst) == []
    assert not (dst / "optiq_vision.safetensors").exists()
    assert json.load(open(dst / "model.safetensors.index.json"))["weight_map"]["vision_tower.blocks.0.attn.qkv.weight"] \
        == "optiq_vision.safetensors"                    # the copy lists the same files as its source


def test_a_missing_file_that_holds_a_layer_to_convert_stops_the_build(src):
    m = "language_model.lm_head"
    p = src / "model.safetensors.index.json"
    idx = json.load(open(p))
    for x in ("weight", "scales", "biases"):
        idx["weight_map"][f"{m}.{x}"] = "missing-shard.safetensors"
    json.dump(idx, open(p, "w"))
    with pytest.raises(SystemExit, match="missing-shard.safetensors"):
        n2.build(src)
    assert not src.with_name(src.name + n2.SUFFIX).exists()
    assert not src.with_name(src.name + n2.SUFFIX + ".partial").exists()   # a refused build leaves nothing behind


def test_check_reports_a_file_missing_in_both_that_held_converted_layers(src):
    dst = n2.build(src)
    shard = json.load(open(dst / "model.safetensors.index.json"))["weight_map"]["language_model.lm_head.weight"]
    (src / shard).unlink()
    (dst / shard).unlink()
    assert any(shard in p and "converted" in p for p in n2.check(src, dst))


def test_check_reports_a_file_the_copy_lost(src):
    dst = n2.build(src)
    shard = sorted({s for s in json.load(open(dst / "model.safetensors.index.json"))["weight_map"].values()})[0]
    (dst / shard).unlink()
    assert any(shard in p and "missing" in p for p in n2.check(src, dst))


def test_build_refuses_an_existing_destination(src):
    src.with_name(src.name + n2.SUFFIX).mkdir()
    with pytest.raises(SystemExit):
        n2.build(src)


# ---------- the load() switch: use the N2-gdnhead copy when it exists (default on) ----------
@pytest.mark.parametrize("env, arg, want_n2", [
    (None, None, True),          # default: on
    ("0", None, False),          # MLX_FASTPATH_N2=0 turns it off
    ("1", False, False),         # an explicit argument wins over the environment
    ("0", True, True),
])
def test_the_switch_picks_the_n2_copy(src, monkeypatch, env, arg, want_n2):
    dst = n2.build(src)
    if env is None:
        monkeypatch.delenv("MLX_FASTPATH_N2", raising=False)
    else:
        monkeypatch.setenv("MLX_FASTPATH_N2", env)
    assert n2.resolve_model(src, arg) == (dst if want_n2 else src)


@pytest.mark.parametrize("env", ["0", "false", "OFF", "no", " 0 "])
def test_the_usual_spellings_of_off_in_the_environment_turn_it_off(src, monkeypatch, env):
    n2.build(src)
    monkeypatch.setenv("MLX_FASTPATH_N2", env)
    assert n2.resolve_model(src) == src


def test_check_reports_a_source_without_quantization_instead_of_crashing(src):
    dst = n2.build(src)
    cfg = json.load(open(src / "config.json"))
    del cfg["quantization"], cfg["quantization_config"]
    json.dump(cfg, open(src / "config.json", "w"))
    assert any(p.startswith("config:") for p in n2.check(src, dst))


def test_switched_on_without_a_copy_loads_the_original_and_says_how_to_build_one(src, monkeypatch):
    monkeypatch.delenv("MLX_FASTPATH_N2", raising=False)
    with pytest.warns(UserWarning, match="mlx-fastpath n2"):
        assert n2.resolve_model(src) == src


def test_an_n2_path_and_a_folder_that_is_not_a_finished_copy_are_used_as_given(src, monkeypatch):
    monkeypatch.delenv("MLX_FASTPATH_N2", raising=False)
    dst = n2.build(src)
    assert n2.resolve_model(dst) == dst
    other = make_src(src.parent, "Other-OptiQ-4bit")
    other.with_name(other.name + n2.SUFFIX).mkdir()          # same name, but no manifest: not a finished copy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert n2.resolve_model(other) == other


def test_load_uses_the_switch(src, monkeypatch):
    """load() hands mlx_lm the N2 copy when the switch is on (stand-ins only for the model load and the engine)."""
    import mlx_lm
    from optiq.runtime.engine import OptiqEngine
    import mlx_fastpath as fp
    dst = n2.build(src)
    seen = []
    monkeypatch.setattr(mlx_lm, "load", lambda path: (seen.append(str(path)) or ("model", "tok")))
    monkeypatch.setattr(OptiqEngine, "from_loaded", classmethod(lambda cls, m, t, p: ("engine", str(p))))
    monkeypatch.delenv("MLX_FASTPATH_N2", raising=False)
    assert fp.load(src, memory_limit_gb=None, cache_limit_gb=None) == ("engine", str(dst))
    assert fp.load(src, memory_limit_gb=None, cache_limit_gb=None, n2=False) == ("engine", str(src))
    assert seen == [str(dst), str(src)]


def test_a_model_that_is_not_quantized_loads_as_given_without_a_warning(tmp_path, monkeypatch):
    monkeypatch.delenv("MLX_FASTPATH_N2", raising=False)
    plain = tmp_path / "Plain-bf16"
    plain.mkdir()
    json.dump({"model_type": "fake"}, open(plain / "config.json", "w"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert n2.resolve_model(plain) == plain
