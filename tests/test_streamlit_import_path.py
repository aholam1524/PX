"""Streamlit bootstrap: repo root must be on sys.path when only app/ is added (Community Cloud)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = ROOT / "app" / "streamlit_app.py"
APP_DIR = STREAMLIT_APP.parent


_SUBPROCESS_RUNNER = textwrap.dedent(
    """
    import importlib.util
    import os
    import sys
    from pathlib import Path

    app_dir = Path(os.environ["STREAMLIT_APP_DIR"]).resolve()
    repo_root = Path(os.environ["REPO_ROOT"]).resolve()

    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != repo_root]
    for name in list(sys.modules):
        if name == "housing_analyzer" or name.startswith("housing_analyzer."):
            del sys.modules[name]

    sys.path.insert(0, str(app_dir))
    if str(repo_root) in sys.path:
        raise SystemExit("REPO_ROOT still on sys.path before negative check")

    try:
        import housing_analyzer  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        raise SystemExit(
            "EXPECTED_FAIL: housing_analyzer must not import when only app/ is on sys.path"
        )

    os.environ["HOUSING_USE_FIXTURES"] = "1"

    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        raise SystemExit("SKIP: streamlit.testing.v1.AppTest not available")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(app_dir / "streamlit_app.py"))
    at.run(timeout=30)
    if at.exception:
        raise SystemExit(f"APP_EXCEPTION: {at.exception!r}")
    """
)


def test_streamlit_app_imports_with_only_app_on_sys_path(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["REPO_ROOT"] = str(ROOT)
    env["STREAMLIT_APP_DIR"] = str(APP_DIR)
    env["HOUSING_USE_FIXTURES"] = "1"
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_RUNNER],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0 and "SKIP:" in (result.stdout + result.stderr):
        pytest.skip("streamlit.testing.v1.AppTest not available in subprocess")

    assert result.returncode == 0, (
        f"subprocess failed (code {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "EXPECTED_FAIL" not in result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stdout + result.stderr
