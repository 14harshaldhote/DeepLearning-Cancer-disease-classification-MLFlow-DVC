import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def samples(root) -> list[Path]:
    return sorted((root / "static" / "samples").glob("*.png"))
