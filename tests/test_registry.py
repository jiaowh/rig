"""Registry discovery tests (implementation-plan §3): entry points only, no static adapter imports."""

import re
from pathlib import Path

import pytest

from rig import registry
from tests.test_interfaces import _StubAdapter

SRC_RIG = Path(__file__).resolve().parents[1] / "src" / "rig"


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear_test_registry()
    yield
    registry.clear_test_registry()


def test_register_and_get_via_testing_hook():
    registry.register_adapter_for_testing("stub_proc", lambda: _StubAdapter(process_id="stub_proc"))
    assert "stub_proc" in registry.list_adapters()
    adapter = registry.get_adapter("stub_proc")
    assert adapter.process_id == "stub_proc"


def test_get_adapter_runs_d7_validation():
    shared = lambda x: x  # noqa: E731
    registry.register_adapter_for_testing(
        "bad_proc", lambda: _StubAdapter(physics=shared, verifier=shared, process_id="bad_proc")
    )
    from rig.interfaces import AdapterValidationError

    with pytest.raises(AdapterValidationError, match="D7"):
        registry.get_adapter("bad_proc")


def test_unknown_process_id_raises_keyerror():
    with pytest.raises(KeyError, match="no adapter registered"):
        registry.get_adapter("does_not_exist")


def test_discovery_via_fake_entry_point(monkeypatch):
    """Simulate a packaged adapter self-registering via the rig.adapters group."""

    class FakeEntryPoint:
        name = "fake_proc"

        @staticmethod
        def load():
            return lambda: _StubAdapter(process_id="fake_proc")

    def fake_entry_points(*, group):
        return [FakeEntryPoint()] if group == registry.ENTRY_POINT_GROUP else []

    monkeypatch.setattr(registry, "entry_points", fake_entry_points)
    assert registry.list_adapters() == ["fake_proc"]
    adapter = registry.get_adapter("fake_proc")
    assert adapter.process_id == "fake_proc"


def test_core_never_imports_adapters_static():
    """implementation-plan §3 hard boundary: zero rig_adapters references anywhere in src/rig.

    Import-linter enforces this in CI; this is the in-suite backstop.
    """
    pattern = re.compile(r"^\s*(import\s+rig_adapters|from\s+rig_adapters)", re.MULTILINE)
    offenders = [
        str(p) for p in SRC_RIG.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
