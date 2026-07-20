"""RIG - Recipe Inverse Generator, process-agnostic core (implementation-plan §3, §13).

Hard boundary: this package must NEVER import from ``rig_adapters``.
Adapters self-register via the ``rig.adapters`` entry-point group and are
discovered at runtime by :mod:`rig.registry` (enforced by import-linter).
"""

__version__ = "0.1.0"
