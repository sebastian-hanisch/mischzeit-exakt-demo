"""Abstand zur stationären Verteilung, Mischzeiten und Schranken für eine reversible Kette (Eigenzerlegung).

Definitionen (Levin/Peres/Wilmer, Propp/Wilson):
  TV(μ, ν) = ½ Σ |μ − ν|;  d(t) = max_x TV(P^t(x,·), π);  d̄(t) = max_{x,y} TV(P^t(x,·), P^t(y,·)).
  t_mix(ε) = kleinstes t mit d(t) ≤ ε (t_mix = t_mix(1/4));  t̄_mix(ε) = kleinstes t mit d̄(t) ≤ ε;  T_mix(1/e) = t̄_mix(1/e).
  Relaxationszeit t_rel = 1/γ* mit der absoluten Spektrallücke γ* = 1 − max(|λ₂|, |λ_min|) der Matrix.
  Satz 12.4 (LPW): (t_rel − 1) ln(1/(2ε)) ≤ t_mix(ε) ≤ t_rel ln(1/(ε π_min)).

Rechenweg: A = D^{1/2} P D^{-1/2} ist bei reversibler Kette symmetrisch, A = U diag(μ) Uᵀ; dann ist P^t(x,y) = Σ_k μ_k^t U[x,k] U[y,k] √(π_y/π_x).
d(t) und d̄(t) fallen in t (nicht wachsend), darum genügt eine Halbierungssuche. Dichte Rechnung: für Ketten bis einige hundert Zustände."""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import msz_chain as K
import msz_constants as C


@dataclass(frozen=True)
class Spectrum:
    pi: np.ndarray
    mu: np.ndarray        # Eigenwerte, absteigend
    U: np.ndarray         # Eigenvektoren der symmetrisierten Matrix (Spalten)

    @property
    def N(self) -> int:
        return len(self.pi)


def spectrum_of(P_dense: np.ndarray, pi: np.ndarray) -> Spectrum:
    s = np.sqrt(pi)
    A = (s[:, None] * P_dense) / s[None, :]
    if np.abs(A - A.T).max() > 1e-12:
        raise ValueError("Die Kette ist nicht reversibel bezüglich π.")
    A = 0.5 * (A + A.T)
    mu, U = np.linalg.eigh(A)
    order = np.argsort(mu)[::-1]
    return Spectrum(pi, mu[order], U[:, order])


@lru_cache(maxsize=32)
def ring_spectrum(n: int, lam: float, kette: str) -> Spectrum:
    ch = K.ring_chain(n, lam, kette)
    return spectrum_of(ch.dense(), ch.pi)


def _potenz(mu: np.ndarray, t) -> np.ndarray:
    """μ^t für ganzzahliges t ≥ 0, auch für negative Eigenwerte; t ein Skalar oder ein Feld der Form (m, 1) (Ergebnis (m, N))."""
    return np.sign(mu) ** t * np.abs(mu) ** t


def rows_at(sp: Spectrum, t: int, rows=None) -> np.ndarray:
    """Zeilen von P^t (alle Zeilen oder die Indizes `rows`)."""
    U = sp.U if rows is None else sp.U[rows]
    At = (U * _potenz(sp.mu, int(t))[None, :]) @ sp.U.T
    s = np.sqrt(sp.pi)
    sx = s if rows is None else s[rows]
    return At * s[None, :] / sx[:, None]


def tv_rows(Pt: np.ndarray, pi: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(Pt - pi[None, :]).sum(axis=1)


def d_t(sp: Spectrum, t: int) -> float:
    return float(tv_rows(rows_at(sp, t), sp.pi).max())


def d_start_t(sp: Spectrum, x: int, t: int) -> float:
    return float(tv_rows(rows_at(sp, t, [x]), sp.pi)[0])


def dbar_rows(Pt: np.ndarray, pi: np.ndarray) -> float:
    """d̄ aus den Zeilen von P^t: max über alle Paare von TV(Zeile x, Zeile y).

    Abbruch durch die Dreiecksungleichung TV(x, y) ≤ TV(x, π) + TV(y, π): Starts werden nach ihrem Abstand zu π absteigend durchgegangen, und sobald
    TV(x, π) + (größter Abstand) nicht mehr über dem bisherigen Maximum liegt, kann kein weiteres Paar es übertreffen (exakt, keine Näherung)."""
    if len(pi) < 2:
        return 0.0
    tv = tv_rows(Pt, pi)
    order = np.argsort(-tv)
    best = 0.0
    for x in order:
        if tv[x] + tv[order[0]] <= best:
            break
        best = max(best, float(0.5 * np.abs(Pt - Pt[x]).sum(axis=1).max()))
    return best


def dbar_t(sp: Spectrum, t: int) -> float:
    return dbar_rows(rows_at(sp, t), sp.pi)


def _erstes_t(f, eps: float, grenze: int = C.T_RECHENGRENZE):
    """Kleinstes t ≥ 0 mit f(t) ≤ eps für eine nicht wachsende Funktion f; None, wenn f(grenze) > eps."""
    if f(0) <= eps:
        return 0
    lo, hi = 0, 1
    while f(hi) > eps:
        lo = hi
        hi *= 2
        if hi > grenze:
            if f(grenze) > eps:
                return None
            hi = grenze
            break
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def t_mix(sp: Spectrum, eps: float = 0.25):
    """t_mix(ε) mit d (Maximum über alle Starts)."""
    return _erstes_t(lambda t: d_t(sp, t), eps)


def t_mix_dbar(sp: Spectrum, eps: float = 0.25):
    """Kleinstes t mit d̄(t) ≤ ε (ε = 1/4: t̄_mix; ε = 1/e: T_mix(1/e)).

    Aus d ≤ d̄ ≤ 2d folgt t_mix(ε) ≤ t̄_mix(ε) ≤ t_mix(ε/2); die Halbierungssuche läuft nur in diesem Einschluss."""
    unten = t_mix(sp, eps)
    oben = t_mix(sp, eps / 2.0)
    if unten is None or oben is None:
        return _erstes_t(lambda t: dbar_t(sp, t), eps)
    lo, hi = max(unten - 1, 0), oben
    if dbar_t(sp, lo) <= eps:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if dbar_t(sp, mid) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def t_start(sp: Spectrum, x: int, eps: float = 0.25):
    """Kleinstes t, ab dem der Start x innerhalb von ε an π liegt."""
    return _erstes_t(lambda t: d_start_t(sp, x, t), eps)


def t_alle_starts(sp: Spectrum, eps: float = 0.25, grenze: int = C.T_RECHENGRENZE) -> np.ndarray:
    """t_x für jeden Start x (gleichzeitige Halbierungssuche; −1, wenn bis `grenze` nicht erreicht)."""
    N = sp.N
    s = np.sqrt(sp.pi)

    def tv_bei(ts: np.ndarray) -> np.ndarray:
        W = (sp.U * _potenz(sp.mu, ts[:, None])) @ sp.U.T      # Zeile x bei Zeitpunkt ts[x]
        return tv_rows(W * s[None, :] / s[:, None], sp.pi)

    lo = np.zeros(N, dtype=np.int64)
    hi = np.ones(N, dtype=np.int64)
    sofort = tv_bei(lo) <= eps                                   # t = 0 genügt (nur bei π-nahem Start)
    while True:                                                  # Obergrenze je Start durch Verdoppeln
        schlecht = ~sofort & (tv_bei(hi) > eps) & (hi < grenze)
        if not schlecht.any():
            break
        lo[schlecht] = hi[schlecht]
        hi[schlecht] = np.minimum(2 * hi[schlecht], grenze)
    erreicht = sofort | (tv_bei(hi) <= eps)
    ergebnis = np.full(N, -1, dtype=np.int64)
    ergebnis[sofort] = 0
    suche = erreicht & ~sofort
    while True:
        aktiv = suche & (hi - lo > 1)
        if not aktiv.any():
            break
        mid = (lo + hi) // 2
        gut = tv_bei(np.where(aktiv, mid, hi)) <= eps
        hi = np.where(aktiv & gut, mid, hi)
        lo = np.where(aktiv & ~gut, mid, lo)
    ergebnis[suche] = hi[suche]
    return ergebnis


def kurven(sp: Spectrum, ts, start: int | None = None, mit_dbar: bool = True):
    """d(t), d̄(t) und (falls `start`) d_start(t) an den Zeitpunkten `ts`."""
    d, db, ds = [], [], []
    for t in ts:
        Pt = rows_at(sp, int(t))
        tv = tv_rows(Pt, sp.pi)
        d.append(float(tv.max()))
        if mit_dbar:
            db.append(dbar_rows(Pt, sp.pi))
        if start is not None:
            ds.append(float(tv[start]))
    return np.array(d), (np.array(db) if mit_dbar else None), (np.array(ds) if start is not None else None)


def zeitraster(t_ende: int, punkte: int = 90) -> np.ndarray:
    """Ganzzahliges Zeitraster 0 … t_ende: dicht am Anfang, dann gleichmäßig auf logarithmischer Skala."""
    t_ende = max(int(t_ende), 1)
    klein = np.arange(0, min(12, t_ende) + 1)
    gross = np.unique(np.round(np.geomspace(1, t_ende, punkte)).astype(int))
    return np.unique(np.concatenate([klein, gross, [t_ende]]))


# --- Spektrallücke und Schranken -------------------------------------------------------------------------------------------------------
def absolute_luecke(mu: np.ndarray) -> float:
    """γ* = 1 − max(|λ₂|, |λ_min|); mu absteigend sortiert, mu[0] = 1."""
    if len(mu) < 2:
        return 1.0
    return float(1.0 - max(abs(mu[1]), abs(mu[-1])))


def t_rel(mu: np.ndarray) -> float:
    g = absolute_luecke(mu)
    return math.inf if g <= 1e-15 else 1.0 / g


def schranken(trel: float, pi_min: float, eps: float = 0.25) -> tuple[float, float]:
    """Satz 12.4 (Levin/Peres/Wilmer): (t_rel − 1) ln(1/(2ε)) ≤ t_mix(ε) ≤ t_rel ln(1/(ε π_min))."""
    return (trel - 1.0) * math.log(1.0 / (2.0 * eps)), trel * math.log(1.0 / (eps * pi_min))


def g_gerades_n(lam: float) -> float:
    """Eigenwert-Formel des Heat-bath bei geradem n: n(1 − λ₂) = g(λ) mit q = 1/(1+λ) (Invarianz bewiesen, „zweitgrößter Eigenwert“ gemessen, n = 4 bis 22)."""
    q = 1.0 / (1.0 + lam)
    return 4 * q * q / ((1 + 2 * q) + math.sqrt(1 + 4 * q - 4 * q * q))


# --- Zeiteinheiten ---------------------------------------------------------------------------------------------------------------------
def in_einheit(t: float | None, einheit: str, n: int, lam: float, kette: str):
    """Rechnet Schritte in „Schritte“, „Durchläufe (n Schritte)“ oder „Abfertigungsdauern“ um (None bleibt None)."""
    if t is None:
        return None
    if einheit == "Schritte":
        return float(t)
    if einheit == "Durchläufe (n Schritte)":
        return t / n
    return t / K.takt(n, lam, kette)
