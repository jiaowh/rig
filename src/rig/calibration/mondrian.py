"""Mondrian / group-conditional split conformal (implementation-plan §5.6, §20).

Split conformal (:class:`~rig.calibration.conformal.SplitConformalCalibrator`)
gives MARGINAL coverage: averaged over an exchangeable test draw the band holds
at ``1 - alpha``, but nothing forbids it from over-covering a data-dense centre
and under-covering a tail so the two cancel to nominal. Two independent
2026-07-23 findings hit exactly that hole:

- the false-success study's d=8 arm: the solver hands the marginal band a
  SELECTED point (it searches where the model looks best), and a marginal miss
  slipped through at a solver-chosen recipe (docs/false-success-study-2026-07-23.md);
- the Empa conditional-coverage study: on real HiPIMS data the pooled PASS hid
  HIGH-OUTCOME-TAIL under-coverage (8/24 high-magnitude tertile cells under-cover),
  and ACI/PID cannot fix it because they adapt over TIME, not over MAGNITUDE
  (examples/real_data/empa_hipims/results/m1_empa_conditional.json).

Mondrian conformal (Vovk et al. 2003; the "Mondrian taxonomy") is the group-
conditional fix: partition the space into pre-declared groups and take a SEPARATE
conformal quantile per group, so each group gets ``1 - alpha`` coverage
CONDITIONAL on membership. This module provides:

- :class:`MondrianConformalCalibrator` — per-group standardized-residual split
  conformal, with the exact ``ceil((1-alpha)(n_g+1))`` rule per group (including
  the honest ``+inf`` small-group branch) and a ``min_group_n`` pooled fallback
  for underpowered groups.
- :func:`predicted_magnitude_group_fn` — a ready-made per-output group_fn keyed
  on the PREDICTED mean's magnitude tertile (edges frozen from a calibration
  slice), the group family the Empa study needs.
- :class:`MondrianConformalForwardModel` — a ForwardModel wrapper filling the
  canonical ``conformal_set`` with the group-conditional band, satisfying the
  SAME interface as :class:`~rig.calibration.conformal.ConformalForwardModel`,
  so the §8 solver's default §13.2 containment gate consumes it with ZERO solver
  edits.

Design decision — GROUP ON THE PREDICTED MEAN, at fit AND at predict
--------------------------------------------------------------------
``group_fn(x, y_pred) -> group_id`` is evaluated on the model's PREDICTED mean,
never the true outcome ``y``. The reason is structural: at PREDICTION time the
true outcome does not exist yet, so a magnitude group MUST be defined on
something observable then — the predicted mean. To keep the train/predict
grouping consistent we group on the predicted mean at CALIBRATION time too
(``predicted-for-both``), rather than the tempting "group calibration points by
their observed ``y``, group test points by predicted mean" split, which would
put the two on different partitions and void the per-group exchangeability the
guarantee rests on.

The CONSEQUENCE, stated honestly: group assignment inherits the model's error.
A point whose TRUE magnitude is high but whose PREDICTED mean lands in the mid
group is calibrated as a mid-group point. So the guarantee is conditional on the
PREDICTED-magnitude group, not the (unknowable-at-predict-time) true-magnitude
group — and when the two disagree (measurable as an assignment-agreement rate),
the conditioning is imperfect. Mondrian tightens a marginal band toward the tail;
it does not deliver oracle per-true-magnitude coverage.

Fallback direction — POOLED for underpowered groups, never ``+inf``
-------------------------------------------------------------------
A group with fewer than ``min_group_n`` calibration points borrows the POOLED
(all-groups) conformal quantile rather than its own. Pooled is the SAFE default:
it is a finite, marginally-valid band, only mis-conditioned. The alternative — a
per-group ``+inf`` from the ``ceil`` rule overflowing on a tiny group — would make
every small group USELESS (an infinite band rejects nothing, so the §13.2 gate
abstains on the whole group). ``min_group_n`` defaults to the smallest group that
can fund a finite ``(1-alpha)`` quantile (:func:`finite_quantile_floor`), so under
the default NO group returns ``+inf`` — it falls back to pooled first. The honest
``+inf`` branch is still REACHABLE: set ``min_group_n`` below that floor and a
group too small for a finite quantile returns ``+inf`` rather than borrowing
pooled (the caller has then explicitly chosen honesty-over-usability).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from rig.calibration.conformal import (
    DEFAULT_ALPHA,
    _mean_2d,
    _sigma_total,
    conformal_quantile,
)
from rig.interfaces import ForwardModel, PredictiveDistribution

# a group_fn maps (x, predicted-mean) -> group ids. Ids are any hashable; per
# output the returned array is (m,) for a single point and (n, m) for a batch.
GroupFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def finite_quantile_floor(alpha: float) -> int:
    """Smallest group size ``n`` for which the conformal rule
    ``ceil((1-alpha)(n+1))`` does NOT overflow ``n`` — i.e. the smallest group
    that can fund a FINITE ``(1-alpha)`` split-conformal quantile.

    This is the default ``min_group_n``: a group below it cannot produce a finite
    per-group band, so it borrows the pooled quantile rather than returning +inf.
    For ``alpha=0.1`` this is 9; for ``alpha=0.2`` it is 4.
    """
    n = 1
    while np.ceil((1.0 - alpha) * (n + 1)) > n:
        n += 1
    return n


def predicted_magnitude_group_fn(
    edges: np.ndarray, labels: Sequence[str] = ("low", "mid", "high")
) -> GroupFn:
    """Per-output group_fn keyed on the PREDICTED mean's magnitude bin.

    ``edges`` is ``(m, n_bins-1)`` interior bin edges per output (for tertiles,
    ``(m, 2)`` — the 1/3 and 2/3 quantiles of the CALIBRATION-slice predicted
    means; frozen there so the partition never depends on the test outcome).
    Each output ``j`` is binned independently by ``np.digitize`` and mapped to
    ``labels[bin]``. Returns ``(m,)`` for a single predicted mean, ``(n, m)`` for
    a batch — the shape contract :class:`MondrianConformalCalibrator` expects.

    Grouping is on the PREDICTED mean (never the true ``y``) by construction, so
    this is the leakage-free predict-time-computable family the module docstring
    describes.
    """
    edges = np.atleast_2d(np.asarray(edges, dtype=float))  # (m, n_bins-1)
    labels_arr = np.asarray(list(labels), dtype=object)
    if edges.shape[1] != labels_arr.size - 1:
        raise ValueError(
            f"predicted_magnitude_group_fn: {edges.shape[1]} interior edges per output "
            f"need {edges.shape[1] + 1} labels, got {labels_arr.size}"
        )

    def group_fn(x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y = np.asarray(y_pred, dtype=float)
        single = y.ndim == 1
        Y = y[None, :] if single else y  # (n, m)
        n, m = Y.shape
        if m != edges.shape[0]:
            raise ValueError(
                f"predicted_magnitude_group_fn: predicted mean has {m} outputs but edges "
                f"were built for {edges.shape[0]}"
            )
        out = np.empty((n, m), dtype=object)
        for j in range(m):
            b = np.digitize(Y[:, j], edges[j])  # 0..n_bins-1
            out[:, j] = labels_arr[b]
        return out[0] if single else out

    return group_fn


class MondrianConformalCalibrator:
    """Group-conditional split conformal on the standardized-residual score
    ``s = |y - mu| / sigma_total`` — a Mondrian refinement of
    :class:`~rig.calibration.conformal.SplitConformalCalibrator`.

    Same score, same variance-scaled (CQR-lite) band shape, same held-out-slice
    contract; the ONLY difference is that the conformal quantile ``kappa`` is taken
    PER GROUP (per output), the group defined by ``group_fn`` on the PREDICTED mean.
    A point at predict time gets its own group's ``kappa``, so the band widens for
    the groups that need it (e.g. the high-magnitude tail) without inflating the
    groups that already cover.

    Parameters
    ----------
    group_fn
        ``group_fn(x, y_pred) -> group_ids``; ``y_pred`` is the model's PREDICTED
        mean (never the true ``y`` — see the module docstring). Per output the
        return is ``(m,)`` for a single point / ``(n, m)`` for a batch, ids any
        hashable.
    alpha
        Miscoverage target (default 0.1 -> nominal 90% per group).
    min_group_n
        A group with fewer than this many calibration points uses the POOLED
        quantile instead of its own (the safe fallback — see the module
        docstring). Default: :func:`finite_quantile_floor` for ``alpha`` (the
        smallest group that can fund a finite quantile). Set below that floor to
        re-enable the honest per-group ``+inf`` branch for tiny groups.
    """

    def __init__(
        self,
        group_fn: GroupFn,
        alpha: float = DEFAULT_ALPHA,
        min_group_n: int | None = None,
    ) -> None:
        self.group_fn = group_fn
        self.alpha = float(alpha)
        self.min_group_n = (
            finite_quantile_floor(self.alpha) if min_group_n is None else int(min_group_n)
        )
        self.model: ForwardModel | None = None
        self.scores_: np.ndarray | None = None  # (n_cal, m) standardized residuals
        self.groups_: np.ndarray | None = None  # (n_cal, m) object group ids

    # -- fit --------------------------------------------------------------------

    def fit(self, model: ForwardModel, X_cal: np.ndarray, Y_cal: np.ndarray) -> None:
        """Calibrate on a HELD-OUT block: score each point and assign its group by
        the PREDICTED mean (leakage guard is the caller's, exactly as split
        conformal). Stores per-point scores + group ids; quantiles are formed
        lazily so an ``alpha`` override recomputes cheaply."""
        self.model = model
        X_cal = np.asarray(X_cal, dtype=float)
        Y_cal = np.asarray(Y_cal, dtype=float)
        if Y_cal.ndim == 1:
            Y_cal = Y_cal[:, None]
        n = X_cal.shape[0]
        dist = model.predict(X_cal)
        mu = _mean_2d(np.asarray(dist.mean, dtype=float), n)  # (n, m)
        sig = _mean_2d(_sigma_total(dist), n)  # (n, m)
        self.scores_ = np.abs(Y_cal - mu) / sig  # (n, m)
        self.groups_ = self._as_2d_groups(self.group_fn(X_cal, mu), n, mu.shape[1])

    @staticmethod
    def _as_2d_groups(groups: Any, n: int, m: int) -> np.ndarray:
        """Normalize a group_fn return to an ``(n, m)`` object array."""
        g = np.asarray(groups, dtype=object)
        if g.ndim == 1:  # single point -> (m,); a 1-row batch stays (1, m)
            g = g[None, :] if n == 1 else g[:, None]
        if g.shape != (n, m):
            raise ValueError(
                f"group_fn returned shape {g.shape}, expected ({n}, {m}) "
                "(per-output group id for every point)"
            )
        return g

    # -- per-group quantiles ----------------------------------------------------

    def _kappa_table(self, alpha: float | np.ndarray | None) -> tuple[list[dict], np.ndarray]:
        """Return (per-output {group_id -> kappa}, pooled kappa per output).

        Per (group, output): if the group has ``>= min_group_n`` calibration
        points, its OWN ``ceil((1-alpha)(n_g+1))`` quantile (which may be the
        honest ``+inf`` when ``n_g`` is below the finite floor); otherwise the
        pooled quantile (the safe fallback). ``alpha`` may be scalar or per-output.
        """
        assert self.scores_ is not None and self.groups_ is not None, "fit() first"
        n, m = self.scores_.shape
        alpha_arr = np.broadcast_to(
            np.asarray(self.alpha if alpha is None else alpha, dtype=float), (m,)
        )
        pooled = conformal_quantile(self.scores_, alpha_arr)  # (m,)
        table: list[dict] = []
        for j in range(m):
            col_scores = self.scores_[:, j]
            col_groups = self.groups_[:, j]
            d: dict[Any, float] = {}
            for g in _unique_objects(col_groups):
                mask = col_groups == g
                n_g = int(mask.sum())
                if n_g < self.min_group_n:
                    d[g] = float(pooled[j])  # pooled fallback (never silently shrink)
                else:
                    # per-group quantile; conformal_quantile returns +inf itself
                    # when n_g cannot fund a finite (1-alpha) band (honest branch).
                    d[g] = float(
                        conformal_quantile(col_scores[mask][:, None], float(alpha_arr[j]))[0]
                    )
            table.append(d)
        return table, pooled

    def kappa_for_groups(
        self, groups: np.ndarray, alpha: float | np.ndarray | None = None
    ) -> np.ndarray:
        """Look up the per-point/per-output ``kappa`` for ``groups`` (shape
        matching the query's predicted mean). An id unseen in calibration falls
        back to the pooled quantile (the same safe direction as an underpowered
        group)."""
        table, pooled = self._kappa_table(alpha)
        G = np.atleast_2d(np.asarray(groups, dtype=object))  # (nq, m)
        nq, m = G.shape
        out = np.empty((nq, m), dtype=float)
        for j in range(m):
            d = table[j]
            for i in range(nq):
                out[i, j] = d.get(G[i, j], float(pooled[j]))
        return out

    # -- bands ------------------------------------------------------------------

    def interval(self, x: np.ndarray, alpha: float | np.ndarray | None = None) -> np.ndarray:
        """Group-conditional calibrated interval per output. ``(m, 2)`` for a
        single ``(d,)`` point, ``(nq, m, 2)`` for a batch — the SAME shape contract
        as :meth:`SplitConformalCalibrator.interval`, so the wrapper is a drop-in."""
        assert self.model is not None, "fit() first"
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        dist = self.model.predict(x)
        nq = 1 if single else x.shape[0]
        mu = _mean_2d(np.asarray(dist.mean, dtype=float), nq)  # (nq, m)
        sig = _mean_2d(_sigma_total(dist), nq)  # (nq, m)
        groups = self._as_2d_groups(self.group_fn(x, dist.mean), nq, mu.shape[1])
        kappa = self.kappa_for_groups(groups, alpha)  # (nq, m)
        half = kappa * sig  # (nq, m); +inf kappa -> +inf half (honest unbounded band)
        band = np.stack([mu - half, mu + half], axis=-1)  # (nq, m, 2)
        return band[0] if single else band


def _unique_objects(arr: np.ndarray) -> list:
    """Deterministic unique group ids for an object array (``np.unique`` refuses to
    sort mixed/unorderable objects — preserve first-seen order instead)."""
    seen: list = []
    marker: set = set()
    for v in arr.tolist():
        if v not in marker:
            marker.add(v)
            seen.append(v)
    return seen


class MondrianConformalForwardModel:
    """ForwardModel wrapper filling ``conformal_set`` with the GROUP-CONDITIONAL
    band — the Mondrian twin of :class:`~rig.calibration.conformal.ConformalForwardModel`.

    Delegates mean/sigmas/support_score/jacobian to the base model; ``conformal_set``
    is the :class:`MondrianConformalCalibrator` interval, shape ``(m, 2)`` for a
    single point and ``(n, m, 2)`` for a batch — BYTE-for-byte the same field, shape,
    and semantics the split-conformal wrapper produces. That parity is the whole
    point: :class:`~rig.inverse.pessimistic.PessimisticInverseSolver`'s default §13.2
    ``C(x) ⊆ Z*`` gate reads ``predict(x).conformal_set`` and needs no change to
    consume a group-conditional set instead of a marginal one.

    ``update(records)`` refits the BASE model only; the calibration goes stale by
    construction (its scores came from the old model) — re-fit the calibrator on
    fresh held-out data after an update, exactly as for the split-conformal wrapper.
    """

    def __init__(self, base: ForwardModel, calibrator: MondrianConformalCalibrator) -> None:
        self.base = base
        self.calibrator = calibrator

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        dist = self.base.predict(x)
        return PredictiveDistribution(
            mean=dist.mean,
            aleatoric_sigma=dist.aleatoric_sigma,
            epistemic_sigma=dist.epistemic_sigma,
            conformal_set=self.calibrator.interval(x),
        )

    def support_score(self, x: np.ndarray) -> float:
        return self.base.support_score(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.base.jacobian(x)

    def update(self, records: Iterable[Any]) -> None:
        self.base.update(records)
