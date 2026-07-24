"""Unit-scale, synthetic, hermetic tests for the material-conditioned pooling
runner (examples/real_data/empa_hipims/run_m1_empa_pooled.py) and the ICM
awareness mechanism it relies on.

No data files are read: the runner module is imported by PATH (examples/ is not a
package -- same trick as tests/test_empa_ingest.py), and every model here is fit
on tiny synthetic two-task data. These tests live with core and MUST NOT depend
on the Empa CSVs, so they run in CI without the M0 dataset present.

What they pin:
  (a) the AWARENESS mechanism the runner's Block A rests on -- on a synthetic
      two-task dataset with a genuine task offset, cross-task-conditioned
      epistemic exceeds in-task epistemic at shared-X points, and the predicted
      MEAN shifts between tasks (what a per-campaign, material-blind model cannot
      do);
  (b) the section 5.8 guarantee in the runner's MATERIAL framing -- an unknown
      material's epistemic dominates every known material's, elementwise (Block
      B's zero-shot awareness guarantee);
  (c) determinism of the runner's core helpers (seeded fits + split arithmetic).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from rig.forward.multitask import MultiToolGPForwardModel

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples" / "real_data" / "empa_hipims"


def _load_runner():
    """Import run_m1_empa_pooled by path. examples/ is not a package; the runner
    puts its own dir on sys.path (for `import prepare_empa`) at exec time, so a
    plain spec-exec is enough. Register in sys.modules before exec to be safe
    against the PEP-563 dataclass-resolution trap prepare_empa can trip."""
    module_spec = importlib.util.spec_from_file_location(
        "run_m1_empa_pooled", EX / "run_m1_empa_pooled.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


runner = _load_runner()


# -- synthetic two-task fixtures ---------------------------------------------


def _two_task_offset_data(seed: int = 7, n: int = 30):
    """Two tasks in DISJOINT input regions with a genuine offset AND different
    shapes (so the fitted task correlation is low and neither can borrow the
    other's data across the gap). Task 'al' lives in x0 in [0, 0.4]; task 'ti'
    in x0 in [0.6, 1.0] with a large additive offset. At a 'ti'-region point,
    'al'-conditioning is a genuine cross-task extrapolation -> inflated
    epistemic; 'ti'-conditioning is in-distribution -> tight."""
    rng = np.random.default_rng(seed)
    Xa = rng.uniform([0.0, 0.0], [0.4, 1.0], (n, 2))
    ya = np.sin(5.0 * Xa[:, 0])
    Xb = rng.uniform([0.6, 0.0], [1.0, 1.0], (n, 2))
    yb = 2.0 * Xb[:, 1] + 3.0
    X = np.vstack([Xa, Xb])
    y = np.concatenate([ya, yb])
    tools = ["al"] * n + ["ti"] * n
    return X, y, tools


def _ti_region_points(seed: int = 99, n: int = 15):
    return np.random.default_rng(seed).uniform([0.65, 0.1], [0.95, 0.9], (n, 2))


# -- (a) awareness mechanism: cross-task epistemic + mean shift ----------------


def test_cross_task_epistemic_inflates_and_mean_shifts_at_shared_points():
    X, y, tools = _two_task_offset_data()
    model = MultiToolGPForwardModel(rank=1, n_restarts=2, seed=0).fit(X, y, tools)
    Xstar = _ti_region_points()  # shared-X points in the 'ti' data region

    epi_in = np.asarray(model.predict(Xstar, tool_id="ti").epistemic_sigma).ravel()
    epi_cross = np.asarray(model.predict(Xstar, tool_id="al").epistemic_sigma).ravel()
    mu_in = np.asarray(model.predict(Xstar, tool_id="ti").mean).ravel()
    mu_cross = np.asarray(model.predict(Xstar, tool_id="al").mean).ravel()

    # cross-task (wrong material) epistemic strictly exceeds in-task at EVERY
    # shared point -- the runner's Block A directional criterion, fired cleanly.
    assert np.all(epi_cross > epi_in), (epi_cross, epi_in)
    assert float(epi_cross.mean()) > 5.0 * float(epi_in.mean())
    # the predicted MEAN differs between materials at the same recipes -- the
    # thing a single-material per-campaign model structurally cannot express.
    assert float(np.abs(mu_in - mu_cross).mean()) > 1.0


def test_pair_awareness_helper_flags_a_cross_material_pair():
    """The runner's _pair_awareness helper, exercised on a minimal synthetic
    `data` dict (no files), must flag the cross-material pair: cross epistemic
    inflates, mean shifts, and tool=None dominates."""
    X, y, tools = _two_task_offset_data()
    al = np.array([t == "al" for t in tools])
    Xa, ya, Xb, yb = X[al], y[al], X[~al], y[~al]

    model = MultiToolGPForwardModel(rank=1, n_restarts=2, seed=0).fit(X, y[:, None], tools)

    # minimal data-dict shape that _pair_awareness reads: material + random split
    # indices (fit, cal, test) into each campaign's own X/Y.
    def entry(Xc, Yc, material, tier):
        n = len(Xc)
        idx = np.arange(n)
        return dict(
            X=Xc,
            Y=Yc[:, None],
            material=material,
            tier=tier,
            random=(idx[: n - 6], idx[n - 6 : n - 4], idx[n - 4 :]),
        )

    data = {
        "al_1": entry(Xa, ya, "al", "T1"),
        "ti_1": entry(Xb, yb, "ti", "T1"),  # same tier, different material -> blind kind
    }
    p = runner._pair_awareness(model, data, "al_1", "ti_1", "material")
    assert p["cross_material"] and p["blind_pair"]
    assert p["epi_inflates_both"] and p["directional_pass"]
    assert p["unknown_dominates_both"]
    assert p["mean_shift_abs"][0] > 1.0


# -- (b) section 5.8 in material framing --------------------------------------


def test_unknown_material_epistemic_dominates_known_materials():
    """Block B's zero-shot awareness guarantee: an unseen material's epistemic
    dominates BOTH known materials elementwise (the population fallback, section
    5.8, holds by construction -- never a silent known-material impersonation)."""
    X, y, tools = _two_task_offset_data()
    model = MultiToolGPForwardModel(rank=1, n_restarts=2, seed=0).fit(X, y, tools)
    Q = np.random.default_rng(3).uniform(0.0, 1.0, (20, 2))
    epi_unknown = np.asarray(model.predict(Q, tool_id="unseen_material").epistemic_sigma)
    for mat in ("al", "ti"):
        epi_known = np.asarray(model.predict(Q, tool_id=mat).epistemic_sigma)
        assert np.all(epi_unknown >= epi_known - 1e-12), mat
    # tool=None takes the same fallback path (the runner uses it as the
    # "material unspecified" query)
    np.testing.assert_array_equal(
        model.predict(Q).epistemic_sigma,
        model.predict(Q, tool_id="unseen_material").epistemic_sigma,
    )


# -- (c) determinism of the runner's core helpers -----------------------------


def _synthetic_data_dict():
    """A tiny two-campaign, two-material `data` dict with a 'random' split, the
    minimal shape fit_pooled reads."""
    X, y, tools = _two_task_offset_data(seed=11, n=24)
    al = np.array([t == "al" for t in tools])

    def entry(Xc, Yc, material):
        n = len(Xc)
        idx = np.arange(n)
        return dict(
            X=Xc,
            Y=Yc[:, None],
            material=material,
            output_keys=["kpi"],
            random=(idx[: n - 8], idx[n - 8 : n - 4], idx[n - 4 :]),
        )

    return {"al_1": entry(X[al], y[al], "al"), "ti_1": entry(X[~al], y[~al], "ti")}


def test_fit_pooled_is_deterministic():
    data = _synthetic_data_dict()
    slugs = ["al_1", "ti_1"]
    kw = dict(task="material", restarts=1, max_iter=40)
    m1 = runner.fit_pooled(slugs, "random", data, ["x0", "x1"], ["kpi"], **kw)
    m2 = runner.fit_pooled(slugs, "random", data, ["x0", "x1"], ["kpi"], **kw)
    Xq = np.random.default_rng(5).uniform(0.0, 1.0, (8, 2))
    for mat in ("al", "ti"):
        np.testing.assert_array_equal(
            m1.predict(Xq, tool_id=mat).mean, m2.predict(Xq, tool_id=mat).mean
        )
        np.testing.assert_array_equal(
            m1.predict(Xq, tool_id=mat).epistemic_sigma,
            m2.predict(Xq, tool_id=mat).epistemic_sigma,
        )
    np.testing.assert_array_equal(m1.tool_covariance_, m2.tool_covariance_)


def test_conformal_metrics_is_deterministic():
    data = _synthetic_data_dict()
    model = runner.fit_pooled(
        ["al_1", "ti_1"],
        "random",
        data,
        ["x0", "x1"],
        ["kpi"],
        task="material",
        restarts=1,
        max_iter=40,
    )
    d = data["ti_1"]
    _, cal, test = d["random"]
    view = model.for_tool("ti")
    a = runner._conformal_metrics(
        view, d["X"][cal], d["Y"][cal], d["X"][test], d["Y"][test], ["kpi"], ["m"]
    )
    b = runner._conformal_metrics(
        view, d["X"][cal], d["Y"][cal], d["X"][test], d["Y"][test], ["kpi"], ["m"]
    )
    assert a == b
    assert 0.0 <= a["pooled"]["picp"] <= 1.0


def test_split_indices_matches_baseline_60_20_20():
    # the exact split arithmetic run_m1_empa uses; pinned to the six campaigns'
    # ingested row counts (601/651/401/490/601/401 -> baseline split_sizes).
    assert runner.split_indices(601) == (361, 120)
    assert runner.split_indices(651) == (391, 130)
    assert runner.split_indices(401) == (241, 80)
    assert runner.split_indices(490) == (294, 98)


def test_tier_and_material_helpers():
    assert runner.tier_of("al_120w_short_pw") == "120w_short_pw"
    assert runner.tier_of("ti_120w_short_pw") == "120w_short_pw"  # same tier, diff material
    assert runner.tier_of("al_200w_high_pw") == "200w_high_pw"
    assert runner.material_of_slug("al_120w_short_pw") == "al"
    assert runner.material_of_slug("ti_200w_high_pw") == "ti"

    # blind_pair_order returns exactly the 4 cross-material same-tier ordered pairs
    prr = ["al_120w_short_pw", "al_200w_high_pw", "ti_120w_short_pw", "ti_200w_high_pw"]
    data = {s: dict(material=runner.material_of_slug(s), tier=runner.tier_of(s)) for s in prr}
    pairs = runner.blind_pair_order(prr, data)
    assert len(pairs) == 4
    assert all(
        data[a]["material"] != data[b]["material"] and data[a]["tier"] == data[b]["tier"]
        for a, b in pairs
    )
    assert ("al_120w_short_pw", "ti_120w_short_pw") in pairs
    assert ("ti_200w_high_pw", "al_200w_high_pw") in pairs
