"""run.py's job runner on a fake child (no model, no GPU). Run:
<venv>/bin/python -m pytest -q -p no:cacheprovider test_run.py"""
import io
import json
import subprocess

import pytest

import run as runner


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """A job whose child prints rows without swap growth, then a final guard report that has it."""
    stdout = "\n".join([json.dumps({"prompt": "chat", "tok_s": 20.0, "peak_footprint_gb": 7.1}),
                        json.dumps({"prompt": "math", "tok_s": 21.0, "peak_footprint_gb": 7.3}),
                        json.dumps({"guard": {"peak_footprint_gb": 7.4, "swap_growth_gb": 0.12}})])
    monkeypatch.setattr(runner, "free_gb", lambda: 16.0)
    monkeypatch.setattr(runner, "swap_used_gb", lambda: 1.0)
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr=""))
    monkeypatch.setattr(runner, "LOG", io.StringIO())
    return runner.job("fake", tmp_path, ["x.py"], 8.0, out=tmp_path / "rows.jsonl")


def test_end_line_reports_the_guard_report_swap_growth_and_peak(fake):
    assert runner.run(fake) == "ok"
    end = [l for l in runner.LOG.getvalue().splitlines() if " END fake:" in l]
    assert len(end) == 1
    assert "swap growth 0.12 GB" in end[0]
    assert "peak 7.4 GB" in end[0]


def test_a_job_with_a_raw_file_keeps_its_full_output(fake, tmp_path):
    raw = tmp_path / "job.out"
    assert runner.run({**fake, "raw": raw}) == "ok"
    kept = raw.read_text()
    assert '"prompt": "math"' in kept and '"swap_growth_gb": 0.12' in kept


def test_only_prompt_rows_are_written(fake):
    runner.run(fake)
    rows = [json.loads(l) for l in open(fake["out"])]
    assert [r["prompt"] for r in rows] == ["chat", "math"]
