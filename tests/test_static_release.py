"""Static tests that do not require CUDA or heavy viewer dependencies."""

from __future__ import annotations

from scripts.validate_release import run_checks


def test_release_static_files_are_valid() -> None:
    assert run_checks() == []
