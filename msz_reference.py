"""Lädt die vorgerechneten Tabellen (`msz_reference.json`, erzeugt von tools/msz_make_tables.py; Zahlen sind „vorgerechnet“ gekennzeichnet).

Die Datei enthält: die Tabelle aller Ring-Zellen (n = 3 bis 16, acht λ, beide Ketten: t_mix, t_rel, Schranken), die Start-Tabelle (n = 10), die 14 Zellen von
d gegen d̄, die Simulationstabelle (R = 100 bis 100 000), die Fensterbreiten des Rings bis n = 24 und die Mischer auf Karten (Transpositionen, Top-zu-zufällig)."""

import json
from functools import lru_cache
from pathlib import Path

DATEI = Path(__file__).resolve().parent / "msz_reference.json"


@lru_cache(maxsize=1)
def daten() -> dict:
    return json.loads(DATEI.read_text(encoding="utf-8"))


def ring_zelle(n: int, lam: float, kette: str) -> dict | None:
    return daten()["ring_tabelle"].get(f"{n}|{lam:g}|{kette}")


def ring_zellen() -> dict:
    return daten()["ring_tabelle"]


def start_zeile(lam: float) -> dict | None:
    return daten()["start_tabelle"].get(f"{lam:g}")


def dbar_zellen() -> list:
    return daten()["dbar_tabelle"]


def sim_zeile(R: int) -> dict | None:
    return daten()["sim_tabelle"].get(str(R))


def sim_zeile_zweiter_seed(R: int) -> dict | None:
    return daten()["sim_tabelle_zweiter_seed"].get(str(R))


def fenster(n: int) -> dict | None:
    return daten()["ring_fenster"].get(str(n))


def mischer(art: str) -> dict:
    return daten()["mischer"][art]


def karten_52() -> list:
    return daten()["karten"]["52"]
