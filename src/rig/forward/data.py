"""RunRecord -> numeric matrices for the forward surrogate (implementation-plan §3.5, §5).

The GP backbone consumes plain SI-magnitude arrays. This helper maps a list
of WP-A :class:`rig.schema.RunRecord` rows to ``(X, Y)`` matrices given an
explicit key ordering. All values are already SI-canonical (schema validators
run at ingest), so magnitudes are directly comparable across records.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from rig.schema import Fraction, Quantity, RunRecord


def _recipe_scalar(record: RunRecord, key: str) -> float:
    value = record.recipe.values.get(key)
    if value is None:
        raise ValueError(f"run {record.run_id}: recipe has no value for input key {key!r}")
    if isinstance(value, Quantity):
        return float(value.magnitude)
    if isinstance(value, Fraction):
        return float(value.value)
    raise ValueError(
        f"run {record.run_id}: input key {key!r} is {type(value).__name__}; the GP "
        "backbone takes only numeric inputs (categoricals are conditioning, implementation-plan §8.3)"
    )


def _outcome_scalar(record: RunRecord, key: str) -> float:
    for outcome in record.outcomes:
        if outcome.name == key:
            if not isinstance(outcome.value, Quantity):
                raise ValueError(
                    f"run {record.run_id}: outcome {key!r} has modality "
                    f"{outcome.modality!r}; the GP v0 handles scalar_vector only"
                )
            return float(outcome.value.magnitude)
    raise ValueError(f"run {record.run_id}: no outcome named {key!r}")


def records_to_arrays(
    records: Iterable[Any],
    input_keys: Sequence[str],
    output_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map RunRecords to ``(X, Y)`` float matrices of SI magnitudes.

    ``input_keys`` follow the WP-A flattening convention: compositional
    components appear as ``"<variable>.<component>"`` keys. Raises
    ``ValueError`` on missing keys, categorical inputs, or non-scalar
    outcomes (curve/field payloads are ArrayRefs and have no inline value).
    """
    record_list = list(records)
    if not record_list:
        raise ValueError("records_to_arrays needs at least one record")
    x_rows = [[_recipe_scalar(r, k) for k in input_keys] for r in record_list]
    y_rows = [[_outcome_scalar(r, k) for k in output_keys] for r in record_list]
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


def records_to_arrays_with_tools(
    records: Iterable[Any],
    input_keys: Sequence[str],
    output_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Like :func:`records_to_arrays`, plus the per-row ``RunRecord.tool_id``.

    Opt-in sibling for the tool-aware surrogate (WP-I, implementation-plan §10.4): returns
    ``(X, Y, tools)`` where ``tools[i]`` is the tool id of row i — exactly
    what :meth:`rig.forward.multitask.MultiToolGPForwardModel.fit` consumes.
    The tool-blind function's signature is unchanged.
    """
    record_list = list(records)
    X, Y = records_to_arrays(record_list, input_keys, output_keys)
    return X, Y, [str(r.tool_id) for r in record_list]
