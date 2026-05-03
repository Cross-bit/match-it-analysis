"""Společné cesty: vstupy v ``data/``, výstupy pipeline v ``out/``."""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "out"


def ensure_out_dir() -> None:
    """Zajistí existenci ``out/`` (výstupy skriptů)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_data_dir() -> None:
    """Zajistí existenci ``data/`` (sem patří záloha links + ``movies.csv``)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
