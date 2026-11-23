"""gptq_stream.py: the GPTQ math on random layers, and the whole build on a tiny random Qwen3.5 (a few MB). Run:
<venv>/bin/python -m pytest -q -p no:cacheprovider test_gptq_stream.py"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

import gptq_stream as G

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "batch2"))
import requant  # noqa: E402

ROWS, COLS = 48, 256


def correlated_hessian(seed, n=2048):
    """X^T X of inputs with strongly correlated features (as real activations are)."""
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(COLS, COLS)) / np.sqrt(COLS) + np.eye(COLS)
    x = rng.normal(size=(n, 32)) @ rng.normal(size=(32, COLS)) + 0.3 * rng.normal(size=(n, COLS)) @ mix
    return mx.array((x.T @ x).astype(np.float32))


def layer(seed):
    return mx.array(np.random.default_rng(seed).normal(scale=0.05, size=(ROWS, COLS)).astype(np.float32))


def dequant(wq, s, b):
    return mx.dequantize(wq, s, b, group_size=G.GROUP, bits=G.BITS).astype(mx.float32)


def reference_gptq(W, H, S, B):
    """The paper's algorithm column by column (no lazy batches), float64, with the given group scales/biases.
    Returns the codes and each choice's distance from a rounding tie (0 = exactly halfway)."""
    W = np.array(W, dtype=np.float64)
    Hinv, dead = G.inverse_hessian(H)
    Hinv = np.array(Hinv, dtype=np.float64)
    W[:, np.array(dead)] = 0
    codes, tie = np.zeros_like(W), np.zeros_like(W)
    for k in range(W.shape[1]):
        s, b = S[:, k // G.GROUP:k // G.GROUP + 1], B[:, k // G.GROUP:k // G.GROUP + 1]
        x = (W[:, k:k + 1] - b) / s
        c = np.clip(np.round(x), 0, 2 ** G.BITS - 1)
        e = (W[:, k:k + 1] - (c * s + b)) / Hinv[k, k]
        W[:, k + 1:] -= e @ Hinv[k:k + 1, k + 1:]
        codes[:, k:k + 1], tie[:, k:k + 1] = c, np.abs(x - np.floor(x) - 0.5)
    return codes, tie


@pytest.mark.parametrize("seed", [0, 1, 3, 4])
def test_gptq_equals_the_column_by_column_algorithm(seed):
    W, H = layer(seed), correlated_hessian(seed)
    Hinv, dead = G.inverse_hessian(H)
    _, s, b, codes = G.gptq(W, Hinv, dead)
    f64 = lambda a: np.array(a.astype(mx.float32), dtype=np.float64)
    ref, tie = reference_gptq(W, H, f64(s), f64(b))
    codes = np.array(codes)
    rows = [i for i in range(ROWS) if (codes[i] != ref[i]).any()]
    assert len(rows) <= ROWS // 8, len(rows)
    # float32 vs float64: a row may part ways only where the reference itself was at a rounding tie
    for i in rows:
        k = int(np.argmax(codes[i] != ref[i]))
        assert tie[i, k] < 5e-3, (i, k, tie[i, k])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gptq_cuts_the_layer_output_error_against_rounding(seed):
    W, H = layer(seed), correlated_hessian(seed)
    Hinv, dead = G.inverse_hessian(H)
    wq, s, b, _ = G.gptq(W, Hinv, dead)
    rtn = mx.dequantize(*mx.quantize(W, group_size=G.GROUP, bits=G.BITS), group_size=G.GROUP, bits=G.BITS)
    assert G.proxy_loss(W, dequant(wq, s, b), H) < 0.7 * G.proxy_loss(W, rtn, H)


def test_packed_codes_are_mlx_layout_and_dequantize_to_the_chosen_values():
    W = layer(3)
    wq, s, b = mx.quantize(W, group_size=G.GROUP, bits=G.BITS)
    codes = mx.round((mx.dequantize(wq, s, b, group_size=G.GROUP, bits=G.BITS) - mx.repeat(b, G.GROUP, axis=1))
                     / mx.repeat(s, G.GROUP, axis=1))
    assert mx.array_equal(G.pack(codes), wq)
    Hinv, dead = G.inverse_hessian(correlated_hessian(3))
    wq, s, b, codes = G.gptq(W, Hinv, dead)
    shifts = mx.arange(0, 32, G.BITS, dtype=mx.uint32)
    unpacked = ((wq[..., None] >> shifts) & (2 ** G.BITS - 1)).reshape(codes.shape)   # 8 codes per word, low first
    assert mx.array_equal(unpacked.astype(mx.float32), codes)
    assert s.dtype == mx.bfloat16 and b.dtype == mx.bfloat16 and wq.dtype == mx.uint32


def test_row_chunks_give_the_same_result(monkeypatch):
    W, H = layer(4), correlated_hessian(4)
    whole = G.gptq_rows(W, H)
    monkeypatch.setattr(G, "ROW_BYTES", 4 * COLS * 5)          # 5 rows per chunk
    for a, b in zip(G.gptq_rows(W, H), whole):
        assert mx.array_equal(a, b)


def test_a_dead_input_column_gets_zero_weight():
    W, H = layer(5), correlated_hessian(5)
    Hn = np.array(H)
    Hn[7, :] = 0
    Hn[:, 7] = 0
    Hinv, dead = G.inverse_hessian(mx.array(Hn))
    wq, s, b, codes = G.gptq(W, Hinv, dead)
    assert bool(dead[7]) and int(mx.sum(dead)) == 1
    s0, b0 = s[:, :1].astype(mx.float32), b[:, :1].astype(mx.float32)       # column 7 is in group 0
    assert mx.array_equal(codes[:, 7:8], mx.clip(mx.round((0 - b0) / s0), 0, 2 ** G.BITS - 1))


# ---------- the whole build on a tiny random Qwen3.5 ----------
TEXT = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=300,
            linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
            full_attention_interval=4, tie_word_embeddings=False,
            rope_parameters={"type": "default", "mrope_section": [2, 2, 0], "rope_theta": 10000,
                             "partial_rotary_factor": 0.25})
EIGHT = {"language_model.lm_head", "language_model.model.embed_tokens",
         "language_model.model.layers.0.linear_attn.out_proj", "language_model.model.layers.0.mlp.down_proj",
         "language_model.model.layers.1.linear_attn.in_proj_qkv", "language_model.model.layers.3.self_attn.q_proj",
         "language_model.model.layers.3.mlp.up_proj"}


def tiny_folders(root):
    """An OptiQ-style mixed 4/8-bit folder and the 'original' bf16 checkpoint it came from (HF key names)."""
    from mlx_lm.models import qwen3_5
    from mlx_lm.utils import save_model
    mx.random.seed(0)
    m = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=TEXT))
    m.update(tree_map(lambda p: (p * 4).astype(mx.bfloat16) if p.ndim == 2 else p, m.parameters()))
    mx.eval(m.parameters())
    bf = root / "bf16"
    bf.mkdir()
    hf = {}
    for k, v in tree_flatten(m.parameters()):
        hf["lm_head.weight" if k == "language_model.lm_head.weight" else
           "model.language_model." + k[len("language_model.model."):]] = v
    mx.save_safetensors(str(bf / "model-00001-of-00001.safetensors"), hf)
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in hf}},
              open(bf / "model.safetensors.index.json", "w"))
    per = {}

    def pred(path, mod):
        if not hasattr(mod, "to_quantized") or mod.weight.shape[-1] % 64:
            return False
        per[path] = {"group_size": 64, "bits": 8 if path in EIGHT else 4}
        return per[path]
    nn.quantize(m, class_predicate=pred)
    src = root / "optiq"
    src.mkdir()
    save_model(src, m)
    q = {"group_size": 64, "bits": 4, "mode": "affine", **per}
    json.dump({"model_type": "qwen3_5", "text_config": TEXT, "quantization": q, "quantization_config": q},
              open(src / "config.json", "w"))
    return src, bf, hf


def test_the_streamed_pass_reproduces_the_model_when_nothing_is_quantized(tmp_path):
    from mlx_lm.utils import load_model
    src, bf, _ = tiny_folders(tmp_path)
    tokens = mx.array(np.random.default_rng(0).integers(0, 300, size=(6, 24)).astype(np.int32))
    model, _ = load_model(src, lazy=True)
    want = model.language_model.model(tokens).astype(mx.float32)
    model, _ = load_model(src, lazy=True)
    got = G.run_layers(model, set(), G.FloatSource(src, "bf16", bf), tokens, tmp_path / "stage", batch=4,
                       log=lambda s: None).astype(mx.float32)
    assert float(mx.abs(got - want).max()) <= 1e-2 * float(mx.abs(want).max())


def test_build_writes_a_valid_model_with_gptq_layers_from_the_bf16_weights(tmp_path):
    from mlx_lm.utils import load_model
    src, bf, hf = tiny_folders(tmp_path)
    tokens = mx.array(np.random.default_rng(1).integers(0, 300, size=(16, 32)).astype(np.int32))
    dst = tmp_path / "gptq"
    targets = G.build(src, dst, "bf16", bf, tokens=tokens, batch=4, log=lambda s: None)
    assert set(targets) == EIGHT - {requant.EMBED}
    assert requant.check(src, dst, exact=False) == []
    q = json.load(open(dst / "config.json"))["quantization"]
    assert all(q[t] == {"bits": 4, "group_size": 64} for t in targets) and q[requant.EMBED]["bits"] == 8
    # the layers are GPTQ of the bf16 weights: not plain rounding of them, and closer to them than 8-bit -> 4-bit
    model, _ = load_model(dst, lazy=True)
    mods = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    for t in targets:
        w = hf[G.hf_key(t)].astype(mx.float32)
        got = mx.dequantize(mods[t].weight, mods[t].scales, mods[t].biases, group_size=64, bits=4)
        assert not mx.array_equal(mods[t].weight, mx.quantize(hf[G.hf_key(t)], group_size=64, bits=4)[0]), t
        assert float(mx.sum((got - w) ** 2)) < 0.25 ** 2 * float(mx.sum(w ** 2)), t      # still a close copy
    out = model(tokens[:2])
    assert out.shape == (2, 32, 300) and bool(mx.all(mx.isfinite(out)))


def test_build_quantizes_only_the_chosen_variant(tmp_path):
    src, bf, _ = tiny_folders(tmp_path)
    tokens = mx.array(np.random.default_rng(3).integers(0, 300, size=(8, 32)).astype(np.int32))
    dst = tmp_path / "mid"
    targets = G.build(src, dst, "bf16", bf, tokens=tokens, batch=4, log=lambda s: None, variant="gdn_head_mlp")
    assert set(targets) == {m for m in EIGHT if m != requant.EMBED and ".self_attn." not in m}
    q = json.load(open(dst / "config.json"))["quantization"]
    assert q["language_model.model.layers.3.self_attn.q_proj"]["bits"] == 8
    assert requant.check(src, dst, exact=False) == []


def test_a_stopped_build_resumes_from_its_stage_files(tmp_path):
    src, bf, _ = tiny_folders(tmp_path)
    tokens = mx.array(np.random.default_rng(2).integers(0, 300, size=(8, 32)).astype(np.int32))
    first = G.build(src, tmp_path / "a", "bf16", bf, tokens=tokens, batch=4, log=lambda s: None)
    stage = tmp_path / "a.stage"
    logs = []
    G.build(src, tmp_path / "b", "bf16", bf, tokens=tokens, batch=4, log=logs.append)   # b.stage is new
    (tmp_path / "b.stage" / f"{G.HEAD}.safetensors").rename(tmp_path / "head_moved_out.safetensors")
    (tmp_path / "b").rename(tmp_path / "b_old")
    logs.clear()
    G.build(src, tmp_path / "b", "bf16", bf, tokens=tokens, batch=4, log=logs.append)
    rows = [json.loads(l) for l in logs]
    assert sum(r.get("quantized", 0) for r in rows if "quantized" in r) == 0          # layers came from stage
    assert any(r.get("layer") == "lm_head" for r in rows)                             # the missing one was redone
    for t in first:
        a = mx.load(str(stage / f"{t}.safetensors"))
        b = mx.load(str(tmp_path / "b.stage" / f"{t}.safetensors"))
        assert all(mx.array_equal(a[x], b[x]) for x in ("weight", "scales", "biases")), t
