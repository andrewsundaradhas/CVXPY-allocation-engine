from __future__ import annotations

from pathlib import Path

DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "src" / "allocator" / "dashboard.py"


def test_dashboard_module_compiles():
    """Cheap syntax check independent of Streamlit's run context."""
    import py_compile
    py_compile.compile(str(DASHBOARD_PATH), doraise=True)


def test_dashboard_renders_empty_state_without_exception():
    """Run the script via AppTest with no button click; expect the empty-state branch."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=30)
    assert not at.exception, f"dashboard raised: {at.exception}"
