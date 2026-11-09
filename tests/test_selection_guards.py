"""Guards on the model-selection workflow.

* Job outputs and report entries record how they were trained, and outputs trained
  differently (for example for another number of steps) are never pooled into one score,
  nor reused from the cache by scripts/06_sweep.py or scripts/slurm/make_joblist.py.

Job outputs are faked (a persistence forecast carrying a real training record) except in
one end-to-end sweep with a tiny --steps. Everything is written under a temporary --root.
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from carbon_world_model.eval.configs import ProvenanceError, make_config
from carbon_world_model.eval.rolling import (
    LEGACY_REPORT_STEPS, VAL_CUTOFFS, VAL_HORIZON, describe_steps, entry_steps, evaluate,
    horizon_for, job_path, job_record, summarise, training_record,
)

COMMITTED_REPORT = "results/sweep_val/report.json"
DEFAULT_STEPS = make_config("orig").train.steps


def load_script(path):
    """Import a script (their names start with a digit) without running main()."""
    spec = importlib.util.spec_from_file_location(
        "script_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_script(monkeypatch, path, *args, **attrs):
    """Call a script's main() with these arguments and module attributes replaced."""
    mod = load_script(path)
    for name, value in attrs.items():
        setattr(mod, name, value)
    monkeypatch.setattr(sys, "argv", [os.path.basename(path), *map(str, args)])
    return mod.main()


def no_fits(cmds, jobs):
    assert not cmds, f"the sweep tried to fit {len(cmds)} jobs"


def sweep(monkeypatch, *args, fits=False):
    """scripts/06_sweep.py; unless `fits`, every job must be cached, so nothing trains."""
    attrs = {} if fits else {"run_pool": no_fits}
    return run_script(monkeypatch, "scripts/06_sweep.py", *args, **attrs)


def fake_job(panel, root, config, cutoff, seed, steps=None, record=True, horizon=VAL_HORIZON):
    """A job output shaped like run_job()'s, without training: a persistence forecast with
    the training record run_job() writes for make_config(config, steps)."""
    H = horizon_for(panel, cutoff, horizon)
    k = int(np.where(panel.years == cutoff)[0][0])

    def hold(a):
        return np.repeat(a[:, k:k + 1], H + 1, axis=1) * (1 + 0.01 * seed)

    out = {"co2_realized": hold(panel.co2), "D_realized": hold(panel.primary_energy),
           "shares_realized": np.repeat(panel.shares()[:, k:k + 1], H + 1, axis=1),
           "H": np.array(H), "final_loss": np.array(0.0)}
    if record:
        out.update(training_record(make_config(config, steps=steps), seed))
    path = job_path(str(root), config, cutoff, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return path


# ------------------------------------------------- 1. steps are recorded, never pooled across
def test_evaluate_records_steps_and_refuses_mixed_steps(panel, tmp_path):
    root = str(tmp_path)
    for co in VAL_CUTOFFS:
        for s in (0, 1):
            fake_job(panel, root, "orig", co, s, steps=3000)
    res = evaluate(panel, "orig", VAL_CUTOFFS, [0, 1], root)
    assert res["steps"] == 3000 and res["train"]["steps"] == 3000
    assert "seed" not in res["train"]  # seeds differ by design; they are what is pooled
    assert all(row["steps"] == 3000 for row in res["cutoffs"].values())
    assert np.isfinite(summarise(res)["world_model"]["score"])
    with pytest.raises(ProvenanceError, match="not the 4000 required"):
        evaluate(panel, "orig", VAL_CUTOFFS, [0, 1], root, steps=4000)

    # One seed at one cutoff trained for another number of steps.
    fake_job(panel, root, "orig", 2007, 1, steps=4000)
    with pytest.raises(ProvenanceError, match=r"differing fields: train\.steps"):
        evaluate(panel, "orig", VAL_CUTOFFS, [0, 1], root)
    # Each cutoff consistent, but the cutoffs differ.
    for s in (0, 1):
        fake_job(panel, root, "orig", 2007, s, steps=4000)
    with pytest.raises(ProvenanceError, match=r"differing fields: train\.steps"):
        evaluate(panel, "orig", VAL_CUTOFFS, [0, 1], root)

    # summarise() refuses to average rows that record different step counts.
    res["cutoffs"][2007]["steps"] = 4000
    with pytest.raises(ProvenanceError, match="different step counts"):
        summarise(res)


def test_evaluate_refuses_unrecorded_outputs(panel, tmp_path):
    fake_job(panel, tmp_path, "orig", 2005, 0, record=False)
    with pytest.raises(ProvenanceError, match="no record of how it was trained"):
        evaluate(panel, "orig", VAL_CUTOFFS, [0], str(tmp_path))


def test_committed_legacy_report_still_reads():
    """The round-1 report predates step recording: read as 3000 steps, summaries unchanged."""
    report = json.load(open(COMMITTED_REPORT))
    legacy = [v for v in report.values() if "steps" not in v]
    assert legacy, "expected the round-1 entries, which do not record steps"
    assert all(entry_steps(v) == LEGACY_REPORT_STEPS == 3000 for v in legacy)
    assert "legacy entries" in describe_steps(legacy)
    for v in report.values():
        assert summarise(v["detail"]) == v["summary"]


def test_sweep_refuses_stale_cached_outputs(panel, tmp_path, monkeypatch):
    for co in VAL_CUTOFFS:
        fake_job(panel, tmp_path, "orig", co, 0, steps=3000)
    with pytest.raises(ProvenanceError,
                       match=f"train.steps: recorded 3000, expected {DEFAULT_STEPS}"):
        sweep(monkeypatch, "--phase", "val", "--configs", "orig", "--seeds", "0",
              "--root", tmp_path)
    fake_job(panel, tmp_path, "orig", 2005, 0, record=False)
    with pytest.raises(ProvenanceError, match="no record of how it was trained"):
        sweep(monkeypatch, "--phase", "val", "--configs", "orig", "--seeds", "0",
              "--steps", 3000, "--root", tmp_path)
    assert not (tmp_path / "report.json").exists()


def test_make_joblist_refuses_stale_outputs(panel, tmp_path, monkeypatch, capsys):
    script = "scripts/slurm/make_joblist.py"
    for co in VAL_CUTOFFS:
        fake_job(panel, tmp_path, "orig", co, 0, steps=3000)
    with pytest.raises(ProvenanceError, match="recorded 3000, expected"):
        run_script(monkeypatch, script, "--phase", "val", "--configs", "orig", "--seeds", "0",
                   "--root", tmp_path)
    assert capsys.readouterr().out == ""
    run_script(monkeypatch, script, "--phase", "val", "--configs", "orig", "--seeds", "0-1",
               "--steps", 3000, "--root", tmp_path)
    lines = capsys.readouterr().out.splitlines()  # only the missing seed-1 jobs
    assert len(lines) == 3 and all("--seed 1 " in x and "--steps 3000" in x for x in lines)
    run_script(monkeypatch, script, "--phase", "val", "--configs", "orig", "--seeds", "0",
               "--root", tmp_path, "--all")  # rerun everything: nothing is skipped
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_sweep_end_to_end_with_tiny_steps(panel, tmp_path, monkeypatch):
    """Real fits (2 steps) through scripts/run_job.py, scored into a temporary report."""
    sweep(monkeypatch, "--phase", "val", "--configs", "orig", "--seeds", "0", "--steps", 2,
          "--jobs", 3, "--root", tmp_path, fits=True)
    for co in VAL_CUTOFFS:
        rec = job_record(np.load(job_path(str(tmp_path), "orig", co, 0)))
        assert rec["steps"] == rec["config"]["train"]["steps"] == 2
        assert rec["config"]["train"]["seed"] == 0
    assert json.load(open(tmp_path / "report.json"))["orig|realized"]["steps"] == 2

    # Another training length on the same root is refused before anything is fitted,
    # with a message rather than a traceback.
    p = subprocess.run([sys.executable, "scripts/06_sweep.py", "--phase", "val", "--configs",
                        "orig", "--seeds", "0", "--steps", "3", "--root", str(tmp_path)],
                       capture_output=True, text=True)
    assert p.returncode == 1
    assert "recorded 2, expected 3" in p.stderr and "Traceback" not in p.stderr
