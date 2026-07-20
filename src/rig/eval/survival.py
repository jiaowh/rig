"""Cost-to-target as right-censored survival data (implementation-plan §12.2).

The headline inverse-design metric is **cost-to-target**, and §12 mandates it be
analysed as survival data with two corrections the naive comparison misses:

1. **infeasibility ≠ censoring** (§12.2 (i)). Only *feasible-but-unhit-within-
   budget* targets are legitimately right-censored. A truly-infeasible target is
   never hit at any budget — an "immune/cured" subject that violates the KM
   eventual-event assumption — so it is **excluded** from the survival analysis
   entirely (it belongs to the feasibility/abstention metrics, §12.2). This
   module never silently treats an infeasible target as censored; the caller
   passes only feasible targets, or uses :func:`split_feasible`.
2. **crossing curves ⇒ not log-rank** (§12.2 (ii)). Expert/BO is fast early,
   the pessimistic method faster near tight tolerances, so the curves are
   *expected to cross* → non-proportional hazards → log-rank power collapses.
   The primary comparator is therefore the **difference-in-RMST** test
   (restricted mean survival time to the budget horizon; Uno et al. 2014), NOT
   log-rank.

Here "survival time" is cost-to-target (dollars or #real-queries); an "event" is
hitting the spec, censoring is budget-exhaustion-without-hit. So the KM curve is
the fraction *still searching* vs cost, and a LOWER curve / SMALLER RMST is
better (cheaper).

**Log-rank is deliberately NOT implemented.** §12.2 (ii)/§12.4 keep log-rank
only as a "non-proportional-hazards-caveated *secondary*", precisely because the
curves are expected to cross (log-rank power collapses under exactly that). The
difference-in-RMST test here is the mandated PRIMARY comparator; a caveated
log-rank secondary is deferred (add it as a clearly-labelled secondary if a
reviewer requests the caveated number). numpy/scipy only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class KaplanMeier:
    """Product-limit estimate of S(t) = P(cost-to-target > t).

    ``t`` are the distinct event (hit) times; ``surv`` is right-continuous S
    evaluated just after each event time; ``var`` is the Greenwood variance of
    S at those points. ``n_at_risk``/``n_events`` are the risk-set sizes and hit
    counts at each event time (censored observations leave the risk set without
    an event).
    """

    t: np.ndarray
    surv: np.ndarray
    var: np.ndarray
    n_at_risk: np.ndarray
    n_events: np.ndarray

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """Step-function S(u): right-continuous, S=1 before the first event."""
        u = np.asarray(u, dtype=float)
        # S(u) = product of (1 - d/n) over event times <= u. searchsorted on the
        # event grid gives, per u, how many event steps have been applied.
        idx = np.searchsorted(self.t, u, side="right")  # 0..len(t)
        surv_padded = np.concatenate([[1.0], self.surv])  # S before any event = 1
        return surv_padded[idx]


def kaplan_meier(
    times: Sequence[float] | np.ndarray, events: Sequence[bool] | np.ndarray
) -> KaplanMeier:
    """Kaplan-Meier product-limit estimator with Greenwood variance.

    ``times``: cost-to-target per subject (>= 0). ``events``: True = spec hit
    (event), False = right-censored (budget exhausted without a hit). Ties are
    handled at the distinct time grid; a censoring tied with an event at the
    same time is treated as still at risk for that event (standard convention).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    if times.ndim != 1 or times.shape != events.shape:
        raise ValueError("times and events must be 1-D of equal length")
    if times.size == 0:
        raise ValueError("need at least one subject")
    if np.any(times < 0):
        raise ValueError("cost-to-target times must be non-negative")

    order = np.argsort(times, kind="stable")
    times, events = times[order], events[order]
    event_times = np.unique(times[events])

    surv_list, var_list, nrisk_list, nev_list = [], [], [], []
    s = 1.0
    gvar_sum = 0.0  # running Σ d/(n(n-d)) for Greenwood
    n = times.size
    for t in event_times:
        n_at_risk = int(np.sum(times >= t))
        d = int(np.sum((times == t) & events))
        if n_at_risk <= 0 or d <= 0:
            continue
        s *= 1.0 - d / n_at_risk
        if n_at_risk > d:
            gvar_sum += d / (n_at_risk * (n_at_risk - d))
        else:
            gvar_sum += 0.0  # S hits 0; Greenwood term undefined, contributes 0
        surv_list.append(s)
        var_list.append(s * s * gvar_sum)  # Greenwood: Var(S) = S^2 Σ d/(n(n-d))
        nrisk_list.append(n_at_risk)
        nev_list.append(d)
    _ = n
    return KaplanMeier(
        t=np.asarray(event_times[: len(surv_list)], dtype=float),
        surv=np.asarray(surv_list, dtype=float),
        var=np.asarray(var_list, dtype=float),
        n_at_risk=np.asarray(nrisk_list, dtype=int),
        n_events=np.asarray(nev_list, dtype=int),
    )


@dataclass(frozen=True)
class RMSTResult:
    """Restricted mean survival time to a horizon and its standard error."""

    rmst: float
    se: float
    horizon: float


def rmst(
    times: Sequence[float] | np.ndarray,
    events: Sequence[bool] | np.ndarray,
    horizon: float,
) -> RMSTResult:
    """Restricted mean survival time to ``horizon`` = ∫_0^τ S(u) du.

    Since here "survival" = still-searching, RMST is the mean cost-to-target
    *restricted* to the budget horizon τ (subjects unhit by τ contribute τ). A
    SMALLER RMST is better (cheaper to target). Variance is the standard
    KM-based RMST estimator variance (Klein & Moeschberger; the form Uno et al.
    2014 differences):

        Var(RMST(τ)) = Σ_{t_i ≤ τ} [ ∫_{t_i}^τ S(u) du ]² · d_i / (n_i (n_i − d_i))
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    km = kaplan_meier(times, events)
    tau = float(horizon)

    # build the step grid of S over [0, tau]: knots at 0, event times < tau, tau.
    knots = np.concatenate([[0.0], km.t[km.t < tau], [tau]])
    # S just after each knot (right-continuous step): S(0)=1 then km steps.
    s_after = km.evaluate(knots)  # S at each knot (right-continuous)
    # area = Σ S(knot_i) * (knot_{i+1} - knot_i): S is constant on [knot_i, knot_{i+1})
    widths = np.diff(knots)
    area = float(np.sum(s_after[:-1] * widths))

    # variance: for each event time t_i <= tau, the "remaining area" ∫_{t_i}^τ S.
    var = 0.0
    for i, t_i in enumerate(km.t):
        if t_i > tau:
            break
        n_i, d_i = km.n_at_risk[i], km.n_events[i]
        if n_i <= d_i:
            continue
        # remaining area from t_i to tau under the (already-estimated) S curve
        sub_knots = np.concatenate([[t_i], km.t[(km.t > t_i) & (km.t < tau)], [tau]])
        s_sub = km.evaluate(sub_knots)
        remaining = float(np.sum(s_sub[:-1] * np.diff(sub_knots)))
        var += remaining * remaining * d_i / (n_i * (n_i - d_i))
    return RMSTResult(rmst=area, se=float(np.sqrt(var)), horizon=tau)


@dataclass(frozen=True)
class RMSTDifference:
    """Difference-in-RMST two-sample test (Uno et al. 2014)."""

    delta: float  # RMST_a - RMST_b (negative ⇒ method a is cheaper)
    se: float
    z: float
    p_value: float
    rmst_a: float
    rmst_b: float
    horizon: float


def rmst_difference_test(
    times_a: Sequence[float] | np.ndarray,
    events_a: Sequence[bool] | np.ndarray,
    times_b: Sequence[float] | np.ndarray,
    events_b: Sequence[bool] | np.ndarray,
    horizon: float,
) -> RMSTDifference:
    """Difference-in-RMST test (Uno et al. 2014) — the §12.2 PRIMARY comparator
    for cost-to-target (log-rank is invalid under the expected curve crossing).

    Returns ``delta = RMST_a − RMST_b`` (negative ⇒ method a reaches targets
    more cheaply), its SE (independent-sample: √(Var_a+Var_b)), a two-sided
    Wald z and p-value. Pass ONLY feasible targets (exclude infeasible per
    §12.2 (i); see :func:`split_feasible`).
    """
    ra = rmst(times_a, events_a, horizon)
    rb = rmst(times_b, events_b, horizon)
    delta = ra.rmst - rb.rmst
    se = float(np.sqrt(ra.se**2 + rb.se**2))
    if se == 0.0:
        z = 0.0 if delta == 0.0 else np.inf * np.sign(delta)
        p = 1.0 if delta == 0.0 else 0.0
    else:
        z = delta / se
        p = float(2.0 * norm.sf(abs(z)))
    return RMSTDifference(
        delta=float(delta),
        se=se,
        z=float(z),
        p_value=p,
        rmst_a=ra.rmst,
        rmst_b=rb.rmst,
        horizon=float(horizon),
    )


def split_feasible(
    times: Sequence[float] | np.ndarray,
    events: Sequence[bool] | np.ndarray,
    feasible: Sequence[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop truly-infeasible targets before survival analysis (§12.2 (i)).

    ``feasible[i]`` is False for a pre-registered known-infeasible target — those
    are "cured" subjects that violate KM and must NOT be censored into the
    curve. Returns ``(times, events)`` restricted to feasible targets. The
    infeasible ones are scored by the feasibility/abstention metrics instead.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    feasible = np.asarray(feasible, dtype=bool)
    if not (times.shape == events.shape == feasible.shape):
        raise ValueError("times, events, feasible must have equal shape")
    return times[feasible], events[feasible]
