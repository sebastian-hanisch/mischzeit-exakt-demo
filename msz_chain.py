"""Hard-Core-Ring: Zustände, Heat-bath- und Metropolis-Kette als Matrix (ausgedachtes Modell, Stellplätze im Ring).

n Plätze im Ring, Zustand x = Menge belegter Plätze, keine zwei Nachbarn belegt, Gewicht π(x) ∝ λ^|x|. Ein Schritt ist eine Platzaktualisierung
(Heat-bath) bzw. ein Vorschlag (Metropolis). Der Code ist neu geschrieben (Bitmasken, vektorisiert); die Tests prüfen ihn gegen eine zweite, aus der
Definition aufgezählte Bruch-Matrix (tests/oracle_helfer.py)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp


def lucas(n: int) -> int:
    """Lucas-Zahl L_n = Zahl der unabhängigen Mengen des n-Rings (L_3 = 4, L_4 = 7, L_10 = 123)."""
    a, b = 2, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def ring_states(n: int) -> np.ndarray:
    """Alle unabhängigen Mengen des n-Rings als Bitmasken (Platz v belegt = Bit v), aufsteigend sortiert."""
    if n < 3:
        raise ValueError("Der Ring braucht mindestens drei Plätze.")
    # Pfad mit m Plätzen: S_m = S_{m-1} ∪ { s + 2^(m-1) : s ∈ S_{m-2} }
    pfad = {0: np.array([0], dtype=np.int64), 1: np.array([0, 1], dtype=np.int64)}
    for m in range(2, n + 1):
        pfad[m] = np.concatenate([pfad[m - 1], pfad[m - 2] + (1 << (m - 1))])
    s = pfad[n]
    ersten_und_letzten = ((s & 1) == 1) & (((s >> (n - 1)) & 1) == 1)   # Plätze 0 und n-1 sind im Ring Nachbarn
    return np.sort(s[~ersten_und_letzten])


def popcount(s: np.ndarray, n: int) -> np.ndarray:
    c = np.zeros(len(s), dtype=np.int64)
    for v in range(n):
        c += (s >> v) & 1
    return c


def takt(n: int, lam: float, kette: str) -> float:
    """Schritte je Abfertigungsdauer: Heat-bath n(1+λ), Metropolis n·max(1, λ) (Verlustnetz, Beweis in der Nachbar-Demo zu Gibbs und Glauber)."""
    return n * (1.0 + lam) if kette == "Heat-bath" else n * max(1.0, lam)


@dataclass(frozen=True)
class RingChain:
    n: int
    lam: float
    kette: str
    states: np.ndarray
    P: sp.csr_matrix
    pi: np.ndarray
    teilchen: np.ndarray

    @property
    def N(self) -> int:
        return len(self.states)

    def index(self, maske: int) -> int:
        i = int(np.searchsorted(self.states, maske))
        if i >= self.N or self.states[i] != maske:
            raise ValueError("Keine unabhängige Menge: zwei Nachbarplätze sind belegt.")
        return i

    def dense(self) -> np.ndarray:
        return self.P.toarray()


def build_ring_chain(n: int, lam: float, kette: str = "Heat-bath") -> RingChain:
    if lam <= 0:
        raise ValueError("λ muss positiv sein.")
    if kette not in ("Heat-bath", "Metropolis"):
        raise ValueError("Kette: Heat-bath oder Metropolis.")
    s = ring_states(n)
    N = len(s)
    teilchen = popcount(s, n)
    w = float(lam) ** teilchen.astype(float)
    pi = w / w.sum()
    p = lam / (1.0 + lam)
    ar = np.arange(N)
    zeilen, spalten, werte = [], [], []
    for v in range(n):
        bit = 1 << v
        links = (s >> ((v - 1) % n)) & 1
        rechts = (s >> ((v + 1) % n)) & 1
        frei = (links | rechts) == 0          # beide Nachbarn leer
        belegt = ((s >> v) & 1) == 1
        an = s | bit
        aus = s & ~bit
        idx_aus = np.searchsorted(s, aus)
        idx_an = np.full(N, -1, dtype=np.int64)
        idx_an[frei] = np.searchsorted(s, an[frei])
        if kette == "Heat-bath":
            # frei: belegen mit p, sonst leer; nicht frei: leer
            zeilen += [ar, ar[frei]]
            spalten += [idx_aus, idx_an[frei]]
            werte += [np.where(frei, 1.0 - p, 1.0) / n, np.full(int(frei.sum()), p / n)]
        else:
            a_auf = min(1.0, lam)             # Annahme beim Belegen: min(1, π(y)/π(x)) = min(1, λ)
            a_ab = min(1.0, 1.0 / lam)        # Annahme beim Leeren: min(1, 1/λ)
            ziel = np.where(belegt, idx_aus, np.where(frei, idx_an, ar))
            bewegung = np.where(belegt, a_ab, np.where(frei, a_auf, 0.0))
            zeilen += [ar, ar]
            spalten += [ar, ziel]
            werte += [(1.0 - bewegung) / n, bewegung / n]
    P = sp.coo_matrix((np.concatenate(werte), (np.concatenate(zeilen), np.concatenate(spalten))), shape=(N, N)).tocsr()
    return RingChain(n, float(lam), kette, s, P, pi, teilchen)


@lru_cache(maxsize=64)
def ring_chain(n: int, lam: float, kette: str = "Heat-bath") -> RingChain:
    """Zwischengespeicherte Kette (nur lesen, nie verändern)."""
    return build_ring_chain(n, lam, kette)


def maximalstart_maske(n: int) -> int:
    """Maximalbelegung: jeder zweite Platz (bei geradem n eine der zwei, bei ungeradem n eine von n gleichwertigen)."""
    return sum(1 << v for v in range(0, n - (n % 2), 2))


def maximalstart(chain: RingChain) -> int:
    return chain.index(maximalstart_maske(chain.n))


def label(maske: int, n: int) -> str:
    plaetze = [str(v) for v in range(n) if (maske >> v) & 1]
    return "∅ (leer)" if not plaetze else "{" + ", ".join(plaetze) + "}"


def start_optionen(chain: RingChain) -> list[str]:
    """Beschriftungen aller Zustände (für die Auswahl eines eigenen Starts)."""
    return [label(int(m), chain.n) for m in chain.states]


def torus_zustaende(l1: int, l2: int) -> int:
    """Zahl der unabhängigen Mengen im l1×l2-Torus (periodisch in beiden Richtungen): Spur der l2-ten Potenz der Zeilen-Transfermatrix.

    Zeilenzustand = unabhängige Menge des l1-Rings (l1 ≥ 3), Übergang erlaubt, wenn die zwei Zeilen keinen Platz gemeinsam belegen."""
    zeilen = ring_states(l1)
    T = ((zeilen[:, None] & zeilen[None, :]) == 0).astype(object)
    ergebnis = np.identity(len(zeilen), dtype=object)
    for _ in range(l2):
        ergebnis = ergebnis.dot(T)
    return int(sum(ergebnis[i, i] for i in range(len(zeilen))))
