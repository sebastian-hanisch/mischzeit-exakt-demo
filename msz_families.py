"""Räume mit bekannter Antwort: Kreis, Hyperwürfel, Riffle-Mischen von Karten (Standardbeispiele aus Levin/Peres/Wilmer; keine Kulisse).

Kreis: Irrfahrt auf dem n-Kreis (träge: 1/2 bleiben, je 1/4 nach links und rechts; nicht träge: je 1/2), exakt über die Fourier-Eigenwerte.
Hyperwürfel {0,1}^n: träge Irrfahrt (1/2 bleiben, sonst eine gleichverteilte Koordinate umkehren), gerechnet als Zählkette über das Hamming-Gewicht
(n + 1 Zustände); die Verteilung der vollen Kette ist bei Start in (1, …, 1) innerhalb jedes Gewichts gleichverteilt, der Abstand ist also gleich.
Riffle-Mischen (Gilbert/Shannon/Reeds): exakte Formel der Bemerkung 8.14 (Levin/Peres/Wilmer, 2. Aufl.) mit Eulerschen Zahlen, in Brüchen.
Mischer auf Permutationen (zufällige Transpositionen, Top-zu-zufällig): Vektoriteration der Verteilung ab der Identität (Gruppenirrfahrt)."""

import itertools
import math
from fractions import Fraction
from functools import lru_cache
from math import comb, factorial

import numpy as np

import msz_constants as C

LN2 = math.log(2.0)


# --- Kreis ------------------------------------------------------------------------------------------------------------------------------
def kreis_eigenwerte(n: int, traege: bool = True) -> np.ndarray:
    k = np.arange(n)
    c = np.cos(2.0 * np.pi * k / n)
    return (1.0 + c) / 2.0 if traege else c


def kreis_d(n: int, t: int, traege: bool = True) -> float:
    """d(t) der Irrfahrt auf dem n-Kreis (Start Platz 0; alle Starts gleichwertig). Nicht träge und gerades n: periodisch, d(t) ≥ 1/2."""
    ev = kreis_eigenwerte(n, traege)
    zeile = np.real(np.fft.ifft(np.sign(ev) ** t * np.abs(ev) ** t))
    return float(0.5 * np.abs(zeile - 1.0 / n).sum())


def kreis_ist_periodisch(n: int, traege: bool) -> bool:
    return (not traege) and n % 2 == 0


def kreis_t(n: int, eps: float = 0.25, traege: bool = True, grenze: int = C.T_RECHENGRENZE):
    """t_mix(ε) des Kreises; None, wenn die Kette periodisch ist oder bis `grenze` nicht erreicht wird."""
    if kreis_ist_periodisch(n, traege):
        return None
    return _erstes(lambda t: kreis_d(n, t, traege), eps, grenze)


def _erstes(f, eps, grenze):
    if f(0) <= eps:
        return 0
    lo, hi = 0, 1
    while f(hi) > eps:
        lo = hi
        hi *= 2
        if hi > grenze:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def kreis_luecke(n: int, traege: bool = True) -> float:
    """Absolute Spektrallücke γ* = 1 − max(|λ₂|, |λ_min|) (Beispiel 12.17 bei träger Irrfahrt: (1 − cos(2π/n))/2)."""
    ev = kreis_eigenwerte(n, traege)
    return float(1.0 - np.abs(ev[1:]).max())


def kreis_kurve(n: int, ts, traege: bool = True) -> np.ndarray:
    return np.array([kreis_d(n, int(t), traege) for t in ts])


# --- Hyperwürfel -----------------------------------------------------------------------------------------------------------------------
def wuerfel_pi(n: int) -> np.ndarray:
    w = np.arange(n + 1)
    lg = np.array([math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in w]) - n * LN2
    return np.exp(lg)


@lru_cache(maxsize=32)
def wuerfel_kurve(n: int, t_ende: int) -> np.ndarray:
    """d(t) für t = 0 … t_ende (Zählkette ab Gewicht n, alle Starts gleichwertig)."""
    pi = wuerfel_pi(n)
    w = np.arange(n + 1)
    auf = (n - w) / (2.0 * n)
    ab = w / (2.0 * n)
    p = np.zeros(n + 1)
    p[n] = 1.0
    out = np.empty(t_ende + 1)
    out[0] = 0.5 * np.abs(p - pi).sum()
    for t in range(1, t_ende + 1):
        q = 0.5 * p
        q[1:] += p[:-1] * auf[:-1]
        q[:-1] += p[1:] * ab[1:]
        p = q
        out[t] = 0.5 * np.abs(p - pi).sum()
    return out


def wuerfel_zeit_schranke(n: int) -> int:
    """Obere Grenze für t_mix(0,1): ½ n ln n + 3n reicht (Cutoff bei ½ n ln n mit Fenster O(n))."""
    return int(0.5 * n * math.log(max(n, 2)) + 3 * n + 20)


def wuerfel_ts(n: int, schwellen=(0.9, 0.5, 0.25, 0.1)) -> dict:
    kurve = wuerfel_kurve(n, wuerfel_zeit_schranke(n))
    out = {}
    for e in schwellen:
        hit = np.nonzero(kurve <= e)[0]
        out[e] = int(hit[0]) if len(hit) else None
    return out


def wuerfel_luecke(n: int) -> float:
    """Spektrallücke der trägen Irrfahrt auf {0,1}^n: 1/n (Beispiel 12.16)."""
    return 1.0 / n


# --- Schranken aus der Spektrallücke ---------------------------------------------------------------------------------------------------
def kreis_ts(n: int, schwellen=(0.9, 0.5, 0.25, 0.1), traege: bool = True) -> dict:
    return {e: kreis_t(n, e, traege) for e in schwellen}


# --- Riffle-Mischen --------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=16)
def euler_zeile(n: int) -> tuple:
    """Eulersche Zahlen A(n, 0 … n−1): Zahl der Permutationen von n Karten mit genau r aufsteigenden Folgen minus 1 (Anordnungen mit r−1 Abstiegen)."""
    zeile = [1]
    for m in range(2, n + 1):
        neu = [0] * m
        for k in range(m):
            a = zeile[k] * (k + 1) if k < len(zeile) else 0
            b = zeile[k - 1] * (m - k) if k >= 1 else 0
            neu[k] = a + b
        zeile = neu
    return tuple(zeile)


@lru_cache(maxsize=4096)
def riffle_tv_bruch(n: int, k: int) -> Fraction:
    """TV(Q^k, gleichverteilt) nach k Riffle-Mischungen von n Karten (Levin/Peres/Wilmer Bemerkung 8.14), exakt in Brüchen.

    Q^k(π) = C(2^k + n − r, n) / 2^{kn} mit r = Zahl der aufsteigenden Folgen von π; TV = ½ Σ_r A(n, r−1) |Q^k − 1/n!|."""
    if n < 1:
        raise ValueError("Mindestens eine Karte.")
    a = 2 ** k
    zeile = euler_zeile(n)
    gleich = Fraction(1, factorial(n))
    return sum(zeile[r - 1] * abs(Fraction(comb(a + n - r, n), a ** n) - gleich) for r in range(1, n + 1)) / 2


def riffle_tv(n: int, k: int) -> float:
    return float(riffle_tv_bruch(n, k))


def riffle_kurve(n: int, k_max: int = C.KARTEN_K_MAX) -> list:
    return [riffle_tv(n, k) for k in range(0, k_max + 1)] if n > 1 else [0.0] * (k_max + 1)


def riffle_t(n: int, eps: float = 0.25, k_grenze: int = 60):
    """Kleinste Zahl von Mischungen mit TV ≤ ε (Anzahl Mischungen, nicht Schritte); None, wenn bis `k_grenze` nicht erreicht."""
    if n < 2:
        return 0
    grenze = Fraction(eps)
    for k in range(1, k_grenze + 1):
        if riffle_tv_bruch(n, k) <= grenze:
            return k
    return None


# --- Zufällige Transpositionen und Top-zu-zufällig (vorgerechnet; live nur für kleine n) ------------------------------------------------
def _permutationen(n: int):
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
    basis = n ** np.arange(n, dtype=np.int64)
    keys = (perms * basis).sum(axis=1)
    order = np.argsort(keys)
    sk = keys[order]

    def rang(arr):
        return order[np.searchsorted(sk, (arr * basis).sum(axis=1))]

    return perms, rang


def mischer_ts(n: int, art: str, schwellen=(0.9, 0.5, 0.25, 0.1), t_max: int = 200) -> dict:
    """t_mix(ε) für „Transpositionen“ (zwei gleichverteilte Positionen, gleiche: nichts) oder „Top-zu-zufällig“ auf n Karten, Start Identität."""
    perms, rang = _permutationen(n)
    N = len(perms)
    bewegungen = []
    if art == "Transpositionen":
        bewegungen.append((1.0 / n, rang(perms)))
        for i in range(n):
            for j in range(i + 1, n):
                b = perms.copy()
                b[:, [i, j]] = perms[:, [j, i]]
                bewegungen.append((2.0 / n ** 2, rang(b)))
    elif art == "Top-zu-zufällig":
        for j in range(n):
            b = np.concatenate([perms[:, 1:j + 1], perms[:, 0:1], perms[:, j + 1:]], axis=1)
            bewegungen.append((1.0 / n, rang(b)))
    else:
        raise ValueError("Mischer: Transpositionen oder Top-zu-zufällig.")
    mu = np.zeros(N)
    mu[0] = 1.0                       # lexikographisch erste Anordnung = Identität
    u = 1.0 / N
    d = [0.5 * np.abs(mu - u).sum()]
    for _ in range(t_max):
        neu = np.zeros(N)
        for w, ix in bewegungen:
            neu += w * np.bincount(ix, weights=mu, minlength=N)
        mu = neu
        d.append(0.5 * np.abs(mu - u).sum())
        if d[-1] < min(schwellen) * 0.9:
            break
    d = np.array(d)
    return {e: (int(np.nonzero(d <= e)[0][0]) if (d <= e).any() else None) for e in schwellen}
