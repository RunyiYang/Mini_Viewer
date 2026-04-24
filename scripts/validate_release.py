"""Static release-readiness checks for Mini Viewer.

The checks intentionally avoid importing heavy runtime dependencies such as
``torch``, ``viser``, or ``gsplat`` unless Python syntax compilation needs them.
They validate repository structure, metadata parseability, and line formatting.
"""

from __future__ import annotations

import argparse
import compileall
import pathlib
from collections.abc import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]

ESSENTIAL_FILES = [
    "README.md",
    "CHANGELOG.md",
    "RELEASE_CHECKLIST.md",
    "env.yml",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
    ".github/workflows/ci.yml",
    "run_viewer.py",
    "data_loader.py",
    "actions/base.py",
    "actions/language_feature.py",
    "actions/camera_path.py",
    "core/splat.py",
    "core/renderer.py",
    "core/viewer.py",
    "models/clip_query.py",
    "scripts/download_siglip2.py",
    "scripts/download_dino.py",
    "scripts/render_camera_path.py",
    "scripts/smoke_test_loaders.py",
]

PYTHON_PATHS = [
    "run_viewer.py",
    "data_loader.py",
    "actions",
    "core",
    "models",
    "scripts",
    "utils",
    "tests",
]

OBSOLETE_FILES = [
    "README_patch.md",
    "requirements-common.txt",
    "requirements-cpu.txt",
    "requirements-cuda124.txt",
    "requirements-language.txt",
    "environment-mini-viewer-cpu.yml",
    "environment-mini-viewer-cuda124.yml",
    "environment-mini-viewer-full.yml",
    "environment.yml",
]


def _failures_from_missing(paths: Iterable[str]) -> list[str]:
    return [f"missing required file: {path}" for path in paths if not (ROOT / path).exists()]


def _check_no_obsolete_files() -> list[str]:
    return [f"remove obsolete split file: {path}" for path in OBSOLETE_FILES if (ROOT / path).exists()]


def _check_text_file(path: str, *, min_lines: int = 2) -> list[str]:
    file_path = ROOT / path
    if not file_path.exists():
        return []
    text = file_path.read_text(encoding="utf8")
    failures: list[str] = []
    if "\r\n" in text:
        failures.append(f"{path}: use LF line endings")
    if len(text.splitlines()) < min_lines:
        failures.append(f"{path}: suspiciously few lines; check that newlines were preserved")
    return failures


def _check_metadata() -> list[str]:
    failures: list[str] = []
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        try:
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback.
                import tomli as tomllib  # type: ignore[no-redef]

            data = tomllib.loads(pyproject.read_text(encoding="utf8"))
            project = data.get("project", {})
            if not project.get("name"):
                failures.append("pyproject.toml: [project].name is missing")
            if not project.get("version"):
                failures.append("pyproject.toml: [project].version is missing")
            if not data.get("build-system"):
                failures.append("pyproject.toml: [build-system] is missing")
        except Exception as exc:
            failures.append(f"pyproject.toml: could not parse TOML: {exc}")

    env_file = ROOT / "env.yml"
    if env_file.exists():
        text = env_file.read_text(encoding="utf8")
        for token in ["name: mini-viewer", "dependencies:", "- pip:", "torch==2.4.1"]:
            if token not in text:
                failures.append(f"env.yml: missing expected token {token!r}")

    req_file = ROOT / "requirements.txt"
    if req_file.exists():
        text = req_file.read_text(encoding="utf8")
        for token in ["--index-url https://download.pytorch.org/whl/cu124", "torch==2.4.1"]:
            if token not in text:
                failures.append(f"requirements.txt: missing expected token {token!r}")
    return failures


def _check_python_compile() -> list[str]:
    failures: list[str] = []
    for rel in PYTHON_PATHS:
        path = ROOT / rel
        if not path.exists():
            continue
        ok = compileall.compile_file(str(path), quiet=1) if path.is_file() else compileall.compile_dir(str(path), quiet=1)
        if not ok:
            failures.append(f"python syntax check failed: {rel}")
    return failures


def run_checks(*, compile_only: bool = False) -> list[str]:
    failures: list[str] = []
    failures.extend(_check_python_compile())
    if compile_only:
        return failures

    failures.extend(_failures_from_missing(ESSENTIAL_FILES))
    failures.extend(_check_no_obsolete_files())
    for path, min_lines in [
        ("README.md", 80),
        ("env.yml", 20),
        ("requirements.txt", 20),
        ("pyproject.toml", 20),
        ("run_viewer.py", 80),
        ("core/renderer.py", 80),
        ("models/clip_query.py", 80),
    ]:
        failures.extend(_check_text_file(path, min_lines=min_lines))
    failures.extend(_check_metadata())
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Mini Viewer release-ready static files.")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    failures = run_checks(compile_only=args.compile_only)
    if failures:
        print("[release-check] FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("[release-check] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
