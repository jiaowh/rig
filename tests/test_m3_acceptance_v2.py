"""Tests for the honest M3 acceptance v2 harness (examples/run_m3_acceptance_v2.py).

Three things are pinned:

1. **The pass rule is NON-saturating** — :func:`m3_verdict` compares GROUND-TRUTH hit
   counts, never confidences. The unit test feeds SATURATED confidences with differing
   ground-truth hits and asserts the verdict tracks the hits (the v1 tautology — the rule
   was ``d2_conf >= heavy_conf - 0.02`` while every conf saturated at ~1.0 — could not do
   this). :func:`score_result` likewise reads ``top_hit`` from the oracle, not confidence.
2. **The smoke path executes end-to-end** on the InSilicoMachine (train GP + generator,
   run the §14.6 gate, pre-probe + select targets, run all three arms, score vs ground
   truth, write JSON).
3. **The smoke config is deterministic** — two runs' timing-stripped JSON is byte-identical.

torch/zuko + the MBE sim are required, so the end-to-end tests are gated (the pure
verdict/scoring unit tests are not — they exercise the load-bearing logic with no sim).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "examples"))

# The runner imports the zuko amortized generator eagerly, so the whole module is
# torch-gated: importorskip BEFORE importing it, else a torch-less env errors on import
# instead of skipping. (The end-to-end test additionally needs the MBE sim, gated below.)
pytest.importorskip("torch", reason="M3 v2 needs the [torch] extra (zuko flows)")
pytest.importorskip("zuko", reason="M3 v2 needs the [torch] extra (zuko flows)")

import run_m3_acceptance_v2 as m3v2  # noqa: E402

from rig.interfaces import Infeasible  # noqa: E402


class _Cand:
    """Minimal RecipeCandidate stand-in (recipe + confidence are all score_result reads)."""

    def __init__(self, recipe, confidence):
        self.recipe = recipe
        self.confidence = confidence


# --- the pass rule is non-saturating (the v1 fix) ---------------------------


def _rows(heavy_hits, light_hits, d2_hits, conf):
    """Rows with EQUAL, arbitrarily SATURATED confidences on every arm, and the given
    per-target ground-truth top-1 hits. If the verdict moves only when the hits move,
    confidence provably does not enter it."""
    rows = []
    for h, lt, d in zip(heavy_hits, light_hits, d2_hits, strict=True):
        rows.append(
            {
                "cold_heavy_top_hit": h,
                "cold_light_top_hit": lt,
                "d2_light_top_hit": d,
                # carried for reporting only — the verdict must ignore these:
                "cold_heavy": {"top_conf": conf},
                "cold_light": {"top_conf": conf},
                "d2_light": {"top_conf": conf},
            }
        )
    return rows


def test_verdict_tracks_ground_truth_hits_not_saturated_confidence():
    # d2 beats light (2 vs 1) and matches heavy (2 vs 2) -> PASS, despite conf saturated.
    passing = _rows(
        heavy_hits=[True, True], light_hits=[False, True], d2_hits=[True, True], conf=1.0
    )
    v = m3v2.m3_verdict(passing)
    assert v["gate_pass"] is True
    assert (v["gt_hits_cold_heavy"], v["gt_hits_cold_light"], v["gt_hits_d2_light"]) == (2, 1, 2)

    # FLIP one d2 hit off (now d2=1, light=1) but KEEP confidences saturated at 1.0.
    # v1's confidence rule would still "pass" (1.0 >= 1.0 - 0.02); v2 must FAIL because
    # d2 no longer strictly beats light on ground truth.
    failing = _rows(
        heavy_hits=[True, True], light_hits=[False, True], d2_hits=[False, True], conf=1.0
    )
    v2 = m3v2.m3_verdict(failing)
    assert v2["gate_pass"] is False
    assert v2["d2_gt_light"] is False

    # The ONLY thing that changed between the two row-sets is a ground-truth hit; the
    # confidences were identical and saturated. So the verdict is driven by ground truth.


def test_verdict_fails_when_d2_below_heavy_even_if_above_light():
    # d2=2 > light=1 but d2=2 < heavy=3 -> must FAIL (d2 must match the gold standard too).
    rows = _rows(
        heavy_hits=[True, True, True],
        light_hits=[True, False, False],
        d2_hits=[True, True, False],
        conf=0.9998,
    )
    v = m3v2.m3_verdict(rows)
    assert (v["gt_hits_cold_heavy"], v["gt_hits_cold_light"], v["gt_hits_d2_light"]) == (3, 1, 2)
    assert v["d2_gt_light"] is True and v["d2_ge_heavy"] is False
    assert v["gate_pass"] is False


def test_score_result_reads_the_oracle_not_confidence():
    """top_hit is a GROUND-TRUTH fact (oracle outcome in box), independent of confidence."""
    out_idx = np.array([0])
    box_lo, box_hi = np.array([0.0]), np.array([1.0])

    def oracle(recipe):  # trivial 1-output "machine": y = recipe['a']
        return np.array([recipe["a"]])

    # HIGH confidence but the true outcome is OUT of the box -> miss.
    miss = m3v2.score_result([_Cand({"a": 5.0}, 0.99999)], oracle, out_idx, box_lo, box_hi)
    assert miss["top_hit"] is False and miss["feasible"] is True
    assert miss["top_conf"] == pytest.approx(0.99999)

    # LOW confidence but the true outcome IS in the box -> hit.
    hit = m3v2.score_result([_Cand({"a": 0.5}, 0.01)], oracle, out_idx, box_lo, box_hi)
    assert hit["top_hit"] is True and hit["any_hit"] is True

    # any_hit reflects a later candidate even when the top one misses.
    mixed = m3v2.score_result(
        [_Cand({"a": 9.0}, 0.9), _Cand({"a": 0.4}, 0.2)], oracle, out_idx, box_lo, box_hi
    )
    assert mixed["top_hit"] is False and mixed["any_hit"] is True


def test_score_result_infeasible_is_a_recorded_miss():
    out_idx = np.array([0])
    box_lo, box_hi = np.array([0.0]), np.array([1.0])
    inf = Infeasible(nearest_achievable={"a": 2.0}, distance_to_feasible=1.5, reason="epi-limited")
    sc = m3v2.score_result(inf, lambda r: np.array([r["a"]]), out_idx, box_lo, box_hi)
    assert sc["status"] == "INFEASIBLE"
    assert sc["feasible"] is False and sc["top_hit"] is False and sc["any_hit"] is False


# --- end-to-end smoke + determinism (sim + torch gated) ---------------------

from rig_adapters.mbe import simlink  # noqa: E402

_sim_gate = pytest.mark.skipif(
    not simlink.sim_available(), reason=f"mbe_sim not found (set {simlink.MBE_SIM_ENV})"
)


@_sim_gate
def test_smoke_runs_end_to_end_and_is_deterministic(tmp_path):
    out1 = tmp_path / "s1.json"
    out2 = tmp_path / "s2.json"
    assert m3v2.main(["--smoke", "--out", str(out1)]) == 0
    assert m3v2.main(["--smoke", "--out", str(out2)]) == 0

    d1 = json.loads(out1.read_text(encoding="utf-8"))
    d2 = json.loads(out2.read_text(encoding="utf-8"))

    # every stage ran and produced structure
    assert d1["meta"]["machine"] == "InSilicoMachine(MBE)"
    assert d1["meta"]["okeys"] == list(m3v2.SPEC_OUTPUTS)
    assert set(d1["gate"]) >= {"passed", "sbc_passed", "tarp_passed", "sbc_p_values"}
    verdict = d1["m3_verdict"]
    assert verdict["n_targets"] == len(d1["targets"]) >= 2
    for row in d1["targets"]:
        for arm in ("cold_heavy", "cold_light", "d2_light"):
            assert arm in row
            assert isinstance(row[f"{arm}_top_hit"], bool)
    # the verdict is exactly what m3_verdict computes from the rows (no confidence leak)
    recomputed = m3v2.m3_verdict(d1["targets"])
    assert recomputed["gate_pass"] == verdict["gate_pass"]
    assert recomputed["gt_hits_d2_light"] == verdict["gt_hits_d2_light"]

    # deterministic: timing is the ONLY content allowed to differ
    d1.pop("timing")
    d2.pop("timing")
    assert d1 == d2
