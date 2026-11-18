"""Simulation: R unabhängige Heat-bath-Ketten im Hard-Core-Ring, geschätzte Mischzeit aus dem empirischen Abstand, Rauschgrenze.

Zufall nur aus dem ganzzahligen SplitMix64 (msz_rng.py). Hier läuft derselbe Strom vektorisiert in numpy (uint64 rechnet modulo 2^64, auf jeder Plattform
gleich): `VektorSplitMix64.block(k)` liefert genau die nächsten k Werte, die k Aufrufe von `SplitMix64.next_u64()` liefern würden (tests/test_simulation.py).
Je Schritt und Kette werden zwei Werte gezogen: der Platz (Rest der Division durch n) und die Münze (53 Bit, belegen mit Wkt. p = λ/(1+λ)); die Plätze
aller R Ketten kommen zuerst, dann die Münzen.

Die geschätzte Mischzeit ist das erste t, bei dem der empirische Abstand TV(empirische Verteilung der R Ketten, π) höchstens 1/4 beträgt. Der empirische
Abstand hat auch bei perfekt gemischten Ketten eine Rauschgrenze (mittlerer Abstand von R exakten Ziehungen aus π zu π)."""

import math
from dataclasses import dataclass

import numpy as np

import msz_chain as K
import msz_constants as C
import msz_mixing as M

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_TWO53 = float(1 << 53)


class VektorSplitMix64:
    """SplitMix64-Strom in Blöcken; Zustand und Werte sind gleich denen von msz_rng.SplitMix64."""

    def __init__(self, seed: int):
        self.state = seed & MASK

    def block(self, k: int) -> np.ndarray:
        zaehler = np.arange(1, k + 1, dtype=np.uint64)
        z = np.uint64(self.state) + zaehler * np.uint64(GAMMA)          # state + i·γ modulo 2^64
        self.state = (self.state + k * GAMMA) & MASK
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))

    def zufall(self, k: int) -> np.ndarray:
        """k Zahlen in [0, 1) mit 53 Bit (wie SplitMix64.random())."""
        return (self.block(k) >> np.uint64(11)).astype(np.float64) / _TWO53

    def unterhalb(self, k: int, n: int) -> np.ndarray:
        """k Ganzzahlen in 0 … n−1 (Rest der Division, wie SplitMix64.below)."""
        return (self.block(k) % np.uint64(n)).astype(np.int64)


@dataclass(frozen=True)
class Referenz:
    """Exakte Größen des Rings (Heat-bath) für den Vergleich mit der Simulation."""
    n: int
    lam: float
    N: int
    t_exakt: int            # t_mix(1/4) aus dem Maximalstart (gleich dem Maximum über alle Starts)
    t_exakt_zahl: int       # dieselbe Größe für die Kennzahl Teilchenzahl
    kurve: np.ndarray       # d_x0(t) aus dem Maximalstart, t = 0 … t_max
    kurve_zahl: np.ndarray  # TV der Teilchenzahl-Verteilung, t = 0 … t_max


_REF: dict = {}


def referenz(n: int = C.SIM_N, lam: float = C.SIM_LAM, t_max: int = C.SIM_T_MAX) -> Referenz:
    schluessel = (n, lam, t_max)
    if schluessel in _REF:
        return _REF[schluessel]
    ch = K.ring_chain(n, lam, "Heat-bath")
    x0 = K.maximalstart(ch)
    PT = ch.P.T.tocsr()
    pi_zahl = np.bincount(ch.teilchen, weights=ch.pi, minlength=n // 2 + 1)
    p = np.zeros(ch.N)
    p[x0] = 1.0
    kurve, kurve_zahl = [], []
    for _ in range(t_max + 1):
        kurve.append(0.5 * np.abs(p - ch.pi).sum())
        pz = np.bincount(ch.teilchen, weights=p, minlength=n // 2 + 1)
        kurve_zahl.append(0.5 * np.abs(pz - pi_zahl).sum())
        p = PT @ p
    kurve, kurve_zahl = np.array(kurve), np.array(kurve_zahl)
    erste = lambda a: int(np.nonzero(a <= 0.25)[0][0])
    ref = Referenz(n, lam, ch.N, erste(kurve), erste(kurve_zahl), kurve, kurve_zahl)
    _REF[schluessel] = ref
    return ref


@dataclass(frozen=True)
class Lauf:
    tv: np.ndarray          # empirischer Abstand der ganzen Verteilung, t = 0 … t_max
    tv_zahl: np.ndarray     # empirischer Abstand der Teilchenzahl-Verteilung
    t_schaetz: int | None   # erstes t mit tv ≤ 1/4 (None: nie erreicht)
    t_schaetz_zahl: int | None


def lauf(R: int, seed: int, n: int = C.SIM_N, lam: float = C.SIM_LAM, t_max: int = C.SIM_T_MAX) -> Lauf:
    """R Ketten ab der Maximalbelegung, t_max Heat-bath-Schritte; der empirische Abstand wird nach jedem Schritt gemessen."""
    if R < 1:
        raise ValueError("Mindestens eine Kette.")
    ch = K.ring_chain(n, lam, "Heat-bath")
    suche = ch.states
    p = lam / (1.0 + lam)
    pi_zahl = np.bincount(ch.teilchen, weights=ch.pi, minlength=n // 2 + 1)
    rng = VektorSplitMix64(seed)
    x = np.full(R, K.maximalstart_maske(n), dtype=np.int64)
    tv, tv_zahl = [], []
    for t in range(t_max + 1):
        if t > 0:
            u = rng.block(2 * R)
            v = (u[:R] % np.uint64(n)).astype(np.int64)
            muenze = (u[R:] >> np.uint64(11)).astype(np.float64) / _TWO53 < p
            bit = np.left_shift(np.int64(1), v)
            links = np.right_shift(x, (v - 1) % n) & 1
            rechts = np.right_shift(x, (v + 1) % n) & 1
            frei = (links | rechts) == 0
            x = np.where(frei & muenze, x | bit, x & ~bit)
        idx = np.searchsorted(suche, x)
        emp = np.bincount(idx, minlength=ch.N) / R
        tv.append(0.5 * np.abs(emp - ch.pi).sum())
        emp_z = np.bincount(ch.teilchen[idx], minlength=n // 2 + 1) / R
        tv_zahl.append(0.5 * np.abs(emp_z - pi_zahl).sum())
    tv, tv_zahl = np.array(tv), np.array(tv_zahl)
    erste = lambda a: int(np.nonzero(a <= 0.25)[0][0]) if (a <= 0.25).any() else None
    return Lauf(tv, tv_zahl, erste(tv), erste(tv_zahl))


def rauschgrenze(R: int, seed: int, n: int = C.SIM_N, lam: float = C.SIM_LAM, wiederholungen: int = C.SIM_RAUSCH_WDH) -> tuple[float, float]:
    """Mittlerer Abstand von R exakten Ziehungen aus π zu π (ganze Verteilung, Teilchenzahl); Mittel über `wiederholungen` Ziehungen."""
    ch = K.ring_chain(n, lam, "Heat-bath")
    cdf = np.cumsum(ch.pi)
    cdf[-1] = 1.0
    pi_zahl = np.bincount(ch.teilchen, weights=ch.pi, minlength=n // 2 + 1)
    rng = VektorSplitMix64(seed)
    a, b = [], []
    for _ in range(wiederholungen):
        idx = np.searchsorted(cdf, rng.zufall(R), side="right")
        emp = np.bincount(idx, minlength=ch.N) / R
        a.append(0.5 * np.abs(emp - ch.pi).sum())
        emp_z = np.bincount(ch.teilchen[idx], minlength=n // 2 + 1) / R
        b.append(0.5 * np.abs(emp_z - pi_zahl).sum())
    return float(np.mean(a)), float(np.mean(b))


def rauschgrenze_exakt(pi: np.ndarray, R: int) -> float:
    """Erwarteter Abstand zwischen π und der empirischen Verteilung von R exakten Ziehungen aus π, exakt aus den Binomialverteilungen.

    Die Zahl c_i der Ziehungen im Zustand i ist binomialverteilt (R, π_i); nach der Linearität des Erwartungswerts ist E TV = ½ Σ_i E|c_i/R − π_i|.
    Gerechnet wird über die Wahrscheinlichkeitsfunktion (scipy), die Summe über k nur im Bereich Mittel ± 12 Standardabweichungen (Rest < 1e-30)."""
    from scipy.stats import binom
    gesamt = 0.0
    for p in np.asarray(pi, dtype=float):
        if p <= 0.0:
            continue
        sd = math.sqrt(R * p * (1.0 - p))
        k = np.arange(max(0, int(R * p - 12 * sd) - 2), min(R, int(R * p + 12 * sd) + 3) + 1)
        gesamt += float((binom.pmf(k, R, p) * np.abs(k / R - p)).sum())
    return 0.5 * gesamt


def rauschgrenzen(R: int, n: int = C.SIM_N, lam: float = C.SIM_LAM) -> tuple[float, float]:
    """Exakte Rauschgrenze der ganzen Verteilung und der Teilchenzahl-Verteilung beim Ring (Heat-bath-Gleichgewicht π)."""
    ch = K.ring_chain(n, lam, "Heat-bath")
    pi_zahl = np.bincount(ch.teilchen, weights=ch.pi, minlength=n // 2 + 1)
    return rauschgrenze_exakt(ch.pi, R), rauschgrenze_exakt(pi_zahl, R)


def rauschgrenze_naeherung(R: int, n: int = C.SIM_N, lam: float = C.SIM_LAM) -> float:
    """Näherung ½ Σ_i √(2 π_i (1 − π_i)/(π R)) (mittlerer Betrag einer Normalverteilung; eigene Rechnung, für große R genau)."""
    pi = K.ring_chain(n, lam, "Heat-bath").pi
    return float(0.5 * np.sqrt(2.0 * pi * (1.0 - pi) / (math.pi * R)).sum())


def wiederholungen(R: int, reps: int, seed: int, n: int = C.SIM_N, lam: float = C.SIM_LAM, t_max: int = C.SIM_T_MAX) -> dict:
    """`reps` unabhängige Läufe (Zufalls-Seed seed + 7·R + Nummer); gibt die geschätzten Mischzeiten und die Verläufe zurück."""
    läufe = [lauf(R, seed + 7 * R + i, n, lam, t_max) for i in range(reps)]
    return {"R": R, "seed": seed, "laeufe": läufe,
            "t": [l.t_schaetz for l in läufe], "t_zahl": [l.t_schaetz_zahl for l in läufe]}


def zusammenfassung(liste: list) -> dict:
    """Mittel, Minimum, Maximum der erreichten Werte und Zahl der nie erreichten Läufe."""
    erreicht = [a for a in liste if a is not None]
    return {"mittel": float(np.mean(erreicht)) if erreicht else None, "min": min(erreicht) if erreicht else None,
            "max": max(erreicht) if erreicht else None, "nie": len(liste) - len(erreicht), "von": len(liste)}
