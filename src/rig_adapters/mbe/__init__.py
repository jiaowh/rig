"""MBE process adapter + in-silico pathology machine (Phase 0, implementation-plan §15.2, E2/E3).

Public surface:

- :func:`rig_adapters.mbe.adapter.make_adapter` — entry-point factory for the
  ``mbe`` :class:`~rig.interfaces.ProcessAdapter` (registered in the
  ``rig.adapters`` entry-point group).
- :class:`rig_adapters.mbe.machine.InSilicoMachine` — the E3 stand-in
  "machine": the fast Arrhenius sim path wrapped with injectable, seeded
  pathologies (seasoning, first-wafer offset, heteroscedastic metrology
  noise, per-``tool_id`` hidden-parameter perturbation).
- :func:`rig_adapters.mbe.generate.generate_dataset` — Sobol-seeded
  RunRecord data generation (JSONL CLI: ``python -m rig_adapters.mbe.generate``).

All access to the external ``mbe_sim`` package goes through
:mod:`rig_adapters.mbe.simlink` (``MBE_SIM_PATH`` env var).
"""
