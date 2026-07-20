"""Generic config-driven tabular process adapter (WP-H, implementation-plan §15.6 E5 seed).

Any recipe->outcome process (etch, CVD, litho, ...) can plug into RIG by
writing a declarative process spec (TOML primary, JSON accepted) and pointing
:class:`~rig_adapters.tabular.adapter.TabularAdapter` at it — no Python
required for the schema, and CSV/JSONL ingestion produces validated,
SI-canonical :class:`~rig.schema.RunRecord` rows.

See ``docs/new-process-onboarding.md`` for the runbook.
"""
