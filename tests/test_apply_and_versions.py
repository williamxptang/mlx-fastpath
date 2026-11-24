"""apply() swaps everything in and remove() restores stock; untested package versions run stock.
Version rules: VERSION first line = newest changelog line, changelog strictly descending,
the running package reports VERSION's version, and no git tag is above it."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

import mlx_fastpath
apply_mod = sys.modules["mlx_fastpath.apply"]  # the package exports a function of the same name

ROOT = Path(__file__).resolve().parent.parent


def _parse(v):
    return tuple(int(x) for x in v.split("."))


def test_version_file_is_consistent():
    lines = (ROOT / "VERSION").read_text().splitlines()
    live = lines[0].strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", live)
    log = [re.match(r"# (\d+\.\d+\.\d+) = \S", l).group(1) for l in lines[1:] if l.strip()]
    assert log, "changelog must have at least one line"
    assert log[0] == live
    assert all(_parse(a) > _parse(b) for a, b in zip(log, log[1:])), "changelog not strictly descending"


def test_package_reports_version_file_version():
    assert mlx_fastpath.__version__ == (ROOT / "VERSION").read_text().splitlines()[0].strip()


def test_no_git_tag_is_above_the_version():
    r = subprocess.run(["git", "tag", "--list", "v*"], cwd=ROOT, capture_output=True, text=True)
    tags = [t[1:] for t in r.stdout.split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    live = _parse((ROOT / "VERSION").read_text().splitlines()[0].strip())
    assert all(_parse(t) <= live for t in tags)


def test_apply_then_remove_restores_stock():
    import mlx_lm.models.base as base
    import mlx_lm.models.qwen3_next as qn
    import optiq.runtime.mtp.gdn_capture as gc
    from optiq.runtime import ngram_spec
    import optiq.runtime.spec as spec
    from optiq.runtime.engine import OptiqEngine
    stock = (qn.scaled_dot_product_attention, OptiqEngine._mtp_draft, gc.forward_with_gdn_capture,
             OptiqEngine._prefill_chunked, ngram_spec.MtpDrafter, spec.assistant_generate_step)
    try:
        done = mlx_fastpath.apply(assistant_gamma="auto")
        assert set(done) == {"attention_dispatch", "draft_vocab", "ngram_mtp_drafter", "fused_capture", "prefill_step",
                             "assistant_gamma"}
        now = (qn.scaled_dot_product_attention, OptiqEngine._mtp_draft, gc.forward_with_gdn_capture,
               OptiqEngine._prefill_chunked, ngram_spec.MtpDrafter, spec.assistant_generate_step)
        assert all(a is not b for a, b in zip(now, stock)), "every speedup must be swapped in"
    finally:
        mlx_fastpath.remove()
    assert qn.scaled_dot_product_attention is base.scaled_dot_product_attention
    assert (qn.scaled_dot_product_attention, OptiqEngine._mtp_draft, gc.forward_with_gdn_capture,
            OptiqEngine._prefill_chunked, ngram_spec.MtpDrafter, spec.assistant_generate_step) == stock


def test_untested_versions_run_stock(monkeypatch):
    import mlx_lm.models.qwen3_next as qn
    before = qn.scaled_dot_product_attention
    monkeypatch.setattr(apply_mod, "installed_versions", lambda: {**apply_mod.TESTED, "mlx": "9.9.9"})
    with pytest.warns(UserWarning, match="untested versions"):
        assert mlx_fastpath.apply() == {}
    assert qn.scaled_dot_product_attention is before


@pytest.mark.parametrize("tokens,kv_bits", [(500, None), (16_000, None), (32_000, 4)])
def test_long_prompts_switch_to_4bit_kv(tokens, kv_bits):
    assert mlx_fastpath.recommended(tokens) == {"depth": 2, "kv_bits": kv_bits}


def test_by_default_optiqs_assistant_loop_is_left_alone():
    import optiq.runtime.spec as spec
    before = spec.assistant_generate_step
    try:
        done = mlx_fastpath.apply()
        assert "assistant_gamma" not in done
        assert spec.assistant_generate_step is before
    finally:
        mlx_fastpath.remove()


def test_untested_versions_leave_the_assistant_loop_alone_too(monkeypatch):
    import optiq.runtime.spec as spec
    before = spec.assistant_generate_step
    monkeypatch.setattr(apply_mod, "installed_versions", lambda: {**apply_mod.TESTED, "mlx-optiq": "9.9.9"})
    with pytest.warns(UserWarning, match="untested versions"):
        assert mlx_fastpath.apply(assistant_gamma=2) == {}
    assert spec.assistant_generate_step is before


def test_every_version_tag_holds_its_own_version():
    """A tag v<x> must point at a commit whose VERSION first line is x."""
    r = subprocess.run(["git", "tag", "--list", "v*"], cwd=ROOT, capture_output=True, text=True)
    tags = [t for t in r.stdout.split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    assert tags, "the repo has version tags"
    wrong = []
    for t in tags:
        v = subprocess.run(["git", "show", f"{t}:VERSION"], cwd=ROOT, capture_output=True, text=True)
        if v.returncode != 0 or v.stdout.splitlines()[0].strip() != t[1:]:
            wrong.append(t)
    assert not wrong, f"tags whose commit holds another version: {wrong}"
