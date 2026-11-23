"""The CLI's N2 wiring: `generate --no-n2` and `n2 SRC` (tiny fake OptiQ folder from test_n2, no real model)."""
import contextlib
import json

import mlx.core as mx
import pytest

from mlx_fastpath import cli, n2
from test_n2 import IN, make_src


class _Stop(Exception):
    pass


@pytest.mark.parametrize("flags, want_n2", [([], None), (["--no-n2"], False)])
def test_generate_hands_the_n2_switch_to_load(monkeypatch, flags, want_n2):
    import sys
    load_module = sys.modules["mlx_fastpath.load"]   # the package's `load` name is the function; patch the module's
    seen = {}

    def fake_load(path, **kw):                    # stands in for the model load (the LLM); stops the command there
        seen.update(kw, path=path)
        raise _Stop

    monkeypatch.setattr(load_module, "load", fake_load)
    with pytest.raises(_Stop):
        cli.main(["generate", "--model", "/models/M", "--prompt", "hi", *flags])
    assert seen["path"] == "/models/M" and seen["n2"] is want_n2


@pytest.fixture
def mlx_limits():
    mem = mx.set_memory_limit(1 << 40)
    cache = mx.set_cache_limit(1 << 40)
    yield
    mx.set_memory_limit(mem)
    mx.set_cache_limit(cache)


def test_n2_command_builds_a_copy_that_passes_check(tmp_path, monkeypatch, capsys, mlx_limits):
    import mlx_fastpath.guard
    monkeypatch.setattr(n2, "CHUNK", 3 * IN // 4)
    monkeypatch.setattr(mlx_fastpath.guard, "MemoryGuard", lambda **kw: contextlib.nullcontext())  # the Mac's memory
    src = make_src(tmp_path)
    assert cli.main(["n2", str(src)]) == 0
    out = capsys.readouterr().out
    report = json.loads(out[out.index('{\n "n2"'):])            # the final report, after the per-shard progress lines
    dst = src.with_name(src.name + n2.SUFFIX)
    assert report == {"n2": str(dst), "problems": []}
    assert (dst / n2.MANIFEST).exists()
