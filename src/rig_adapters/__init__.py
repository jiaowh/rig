"""Per-process adapters for RIG (implementation-plan §3.1).

Adapters land in WP-B+. Each adapter self-registers via the
``rig.adapters`` entry-point group; the process-agnostic core
(``rig``) never imports this package.
"""
