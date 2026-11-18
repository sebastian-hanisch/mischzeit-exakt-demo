"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (der Ring mit 4 Plätzen, λ = 1: 7 Zustände)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import msz_chain as K  # noqa: E402
import msz_mixing as M  # noqa: E402


@pytest.fixture(scope="session")
def ring4():
    """Ring mit 4 Plätzen, λ = 1, Heat-bath: 7 Zustände, π gleichverteilt (Handrechnung in tests/test_chain.py und tests/test_mixing.py)."""
    return K.ring_chain(4, 1.0, "Heat-bath")


@pytest.fixture(scope="session")
def ring4_spektrum():
    return M.ring_spectrum(4, 1.0, "Heat-bath")
