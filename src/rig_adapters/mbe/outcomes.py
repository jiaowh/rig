"""Sim-metrics -> OutcomeRecord translation layer (implementation-plan E2: "output is a
nested dict, not an OutcomeRecord").

The Phase-0 evaluation path is ``UniformityProblem.evaluate()`` (the fast
Arrhenius path), whose metric dict plays the role the kMC engine's
``observables.snapshot()`` plays on the high-fidelity path — this module is
the single place where sim metric names map to declared adapter outputs.

Metric -> output mapping (declared engineering units; SI canonicalization
happens inside :class:`rig.schema.Quantity` at OutcomeRecord construction):

====================================  =====================  ============
sim metric key                        adapter output          declared unit
====================================  =====================  ============
``combined_nonuniformity_pct``        ``nonuniformity_pct``  percent
``T_center``                          ``T_center``           K
``slip_max_ratio``                    ``slip_max_ratio``     dimensionless
``bow_cooldown_um``                   ``bow_cooldown_um``    micrometer
``thickness_grown_m``                 ``thickness_grown``    m
====================================  =====================  ============

NB ``percent`` canonicalizes to a dimensionless FRACTION (5 % -> 0.05): the
serialized OutcomeRecord magnitude is the SI-canonical fraction, while
:func:`metrics_to_output_values` (the pathology-injection working
representation) stays in the declared engineering units above.
"""

from __future__ import annotations

from collections.abc import Mapping

from rig.schema import OutcomeRecord, Quantity

# sim metric key -> (output name, declared unit)
METRIC_TO_OUTPUT: dict[str, tuple[str, str]] = {
    "combined_nonuniformity_pct": ("nonuniformity_pct", "percent"),
    "T_center": ("T_center", "K"),
    "slip_max_ratio": ("slip_max_ratio", "dimensionless"),
    "bow_cooldown_um": ("bow_cooldown_um", "micrometer"),
    "thickness_grown_m": ("thickness_grown", "m"),
}

OUTPUT_UNITS: dict[str, str] = {name: unit for name, unit in METRIC_TO_OUTPUT.values()}

# Deterministic output ordering (declared-schema order) for noise draws etc.
OUTPUT_ORDER: tuple[str, ...] = tuple(name for name, _unit in METRIC_TO_OUTPUT.values())


def metrics_to_output_values(metrics: Mapping[str, float]) -> dict[str, float]:
    """Reduce a ``UniformityProblem.evaluate()`` metric dict to the declared
    adapter outputs, in declared engineering units (percent, K, -, um, m)."""
    missing = [k for k in METRIC_TO_OUTPUT if k not in metrics]
    if missing:
        raise ValueError(f"sim metrics missing expected keys: {missing}")
    return {METRIC_TO_OUTPUT[k][0]: float(metrics[k]) for k in METRIC_TO_OUTPUT}


def output_values_to_outcomes(values: Mapping[str, float]) -> list[OutcomeRecord]:
    """Declared-unit output dict -> list of scalar_vector OutcomeRecords
    (SI-canonicalized by the Quantity validator)."""
    out = []
    for name in OUTPUT_ORDER:
        if name not in values:
            raise ValueError(f"output values missing declared output {name!r}")
        out.append(
            OutcomeRecord(
                name=name,
                modality="scalar_vector",
                value=Quantity(magnitude=float(values[name]), unit=OUTPUT_UNITS[name]),
            )
        )
    return out


def metrics_to_outcomes(metrics: Mapping[str, float]) -> list[OutcomeRecord]:
    """Full translation: sim metric dict -> OutcomeRecord list."""
    return output_values_to_outcomes(metrics_to_output_values(metrics))
