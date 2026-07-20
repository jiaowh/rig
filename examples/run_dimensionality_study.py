"""F9 probe: does the RIG inverse actually work above 2 input dimensions?

Ground-truth check (nothing in the repo does this): ask for a recipe hitting a spec
box, then evaluate the TRUE function at the returned recipe and see if it really lands
in spec. Scored against truth, never against the model's own opinion.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
from scipy.stats import qmc

from rig.forward import GPForwardModel
from rig.interfaces import ContinuousVariable, Infeasible
from rig.inverse.pessimistic import PessimisticInverseSolver

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # cp1252 console guard


def make_truth(d, seed=0):
    """A smooth d-dim process with 2 outputs. Every dim is active (no free lunch),
    with decaying weights so it stays learnable."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d,)) / np.sqrt(d)
    V = rng.normal(size=(d,)) / np.sqrt(d)
    ph = rng.uniform(0, 2 * np.pi, size=d)

    def f(X):
        X = np.atleast_2d(X)
        y0 = 5.0 + 3.0 * np.tanh(X @ W) + 0.6 * np.sin(X @ V + ph[0])
        y1 = 8.0 + 2.0 * (X @ V) + 0.4 * np.cos(X @ W)
        return np.stack([y0, y1], axis=-1)

    return f


def probe(d, n_train, seed=0, kappa=2.0, n_restarts=48, verbose=True):
    lo, hi = -2.0, 2.0
    truth = make_truth(d, seed)
    variables = [ContinuousVariable(f"x{i}", lo, hi) for i in range(d)]
    out_keys = ["y0", "y1"]

    # Sobol training design over the box
    sob = qmc.Sobol(d=d, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = qmc.scale(sob.random(n_train), [lo] * d, [hi] * d)
    rng = np.random.default_rng(seed + 1)
    Y = truth(X) + rng.normal(0, 0.05, size=(n_train, 2))

    t0 = time.time()
    model = GPForwardModel(n_restarts=3, seed=seed).fit(X, Y)
    t_fit = time.time() - t0

    # a REACHABLE target: evaluate truth at a held-out on-support point
    x_ref = qmc.scale(qmc.Sobol(d=d, scramble=True, seed=seed + 99).random(1), [lo] * d, [hi] * d)[
        0
    ]
    y_ref = truth(x_ref)[0]
    tol = np.array([0.8, 0.8])
    spec = {
        "targets": {
            "y0": (y_ref[0] - tol[0], y_ref[0] + tol[0]),
            "y1": (y_ref[1] - tol[1], y_ref[1] + tol[1]),
        },
        "max_candidates": 3,
    }

    solver = PessimisticInverseSolver(
        model,
        variables=variables,
        output_keys=out_keys,
        X_train=X,
        kappa=kappa,
        n_restarts=n_restarts,
        seed=seed,
    )
    t0 = time.time()
    try:
        res = solver.solve(spec)
    except Exception as e:  # noqa: BLE001
        return dict(
            d=d,
            n=n_train,
            status=f"CRASH: {type(e).__name__}: {e}",
            t_fit=t_fit,
            t_solve=time.time() - t0,
            hit=None,
        )
    t_solve = time.time() - t0

    if isinstance(res, Infeasible):
        return dict(
            d=d,
            n=n_train,
            status="INFEASIBLE",
            t_fit=t_fit,
            t_solve=t_solve,
            hit=None,
            reason=res.reason[:60],
        )

    # GROUND TRUTH: does the returned recipe really land in spec?
    hits, errs = [], []
    for cand in res:
        xv = np.array([cand.recipe[f"x{i}"] for i in range(d)])
        yt = truth(xv)[0]
        inbox = bool(
            np.all(yt >= [spec["targets"][k][0] for k in out_keys])
            and np.all(yt <= [spec["targets"][k][1] for k in out_keys])
        )
        hits.append(inbox)
        errs.append(float(np.max(np.abs(yt - y_ref))))
    return dict(
        d=d,
        n=n_train,
        status="FEASIBLE",
        t_fit=t_fit,
        t_solve=t_solve,
        n_cand=len(res),
        hit=sum(hits),
        hit_of=len(hits),
        worst_err=max(errs),
        conf=float(res[0].confidence),
    )


if __name__ == "__main__":
    print(
        f"{'d':>3} {'n_train':>7} {'status':>11} {'GT hit':>7} {'worst err':>9} "
        f"{'conf':>6} {'t_fit':>6} {'t_solve':>7}"
    )
    print("-" * 68)
    for d in (2, 4, 6, 8, 10, 15, 20):
        n = max(24, 12 * d)  # scale data with dimension
        r = probe(d, n)
        hit = "—" if r["hit"] is None else f"{r['hit']}/{r['hit_of']}"
        we = r.get("worst_err")
        cf = r.get("conf")
        we_s = "-" if we is None else f"{we:.3f}"
        cf_s = "-" if cf is None else f"{cf:.3f}"
        print(
            f"{r['d']:>3} {r['n']:>7} {r['status']:>11} {hit:>7} "
            f"{we_s:>9} {cf_s:>6} {r['t_fit']:>6.2f} {r['t_solve']:>7.2f}"
        )
        if "reason" in r:
            print(f"      reason: {r['reason']}")
