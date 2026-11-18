"""Unabhängiger Rechenweg für die Orakel-Tests: der Hard-Core-Ring aus der Definition, in Brüchen und mit ganzen Zahlen.

Nichts hier benutzt msz_chain oder msz_mixing. Die Zustände werden als Teilmengen aufgezählt (itertools), jeder Übergang einzeln aus der Beschreibung der Kette
gebildet (Heat-bath: Platz ziehen, belegen mit λ/(1+λ) wenn beide Nachbarn frei; Metropolis: Belegung umkehren, annehmen mit min(1, π(neu)/π(alt))).
Die Potenzen der Übergangsmatrix laufen über ganze Zahlen (Matrix mal gemeinsamer Nenner), der Abstand wird erst am Ende als Bruch gebildet."""

from fractions import Fraction as Fr
from itertools import combinations, permutations, product
from math import comb, factorial


def mengen(n):
    """Alle Teilmengen der Plätze 0 … n−1 ohne zwei benachbarte (im Ring: v und v+1 mod n), als sortierte Liste von Bitmasken."""
    out = []
    for k in range(n + 1):
        for teil in combinations(range(n), k):
            if all(((v + 1) % n) not in teil for v in teil):
                out.append(sum(1 << v for v in teil))
    return sorted(out)


def _nachbarn_frei(x, v, n):
    return not ((x >> ((v - 1) % n)) & 1) and not ((x >> ((v + 1) % n)) & 1)


def uebergang(n, lam: Fr, kette):
    """Übergangsmatrix als Liste von Zeilen aus Brüchen, Zustände in der Reihenfolge von `mengen(n)`."""
    S = mengen(n)
    idx = {x: i for i, x in enumerate(S)}
    N = len(S)
    P = [[Fr(0)] * N for _ in range(N)]
    p = lam / (1 + lam)
    w = lambda x: lam ** bin(x).count("1")
    for x in S:
        for v in range(n):
            bit = 1 << v
            if kette == "Heat-bath":
                if _nachbarn_frei(x, v, n):
                    P[idx[x]][idx[x | bit]] += p / n
                    P[idx[x]][idx[x & ~bit]] += (1 - p) / n
                else:
                    P[idx[x]][idx[x & ~bit]] += Fr(1, n)
            else:
                y = x ^ bit
                if y in idx:
                    a = min(Fr(1), w(y) / w(x))
                    P[idx[x]][idx[y]] += a / n
                    P[idx[x]][idx[x]] += (1 - a) / n
                else:
                    P[idx[x]][idx[x]] += Fr(1, n)
    return S, P


def gleichgewicht(S, lam: Fr):
    w = [lam ** bin(x).count("1") for x in S]
    Z = sum(w)
    return [a / Z for a in w]


def _ganzzahlig(P):
    """(A, D) mit P = A/D, A ganzzahlig, D gemeinsamer Nenner."""
    from math import lcm
    D = 1
    for z in P:
        for a in z:
            D = lcm(D, a.denominator)
    return [[int(a * D) for a in z] for z in P], D


def _mal(A, B):
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(z, s)) for s in Bt] for z in A]


def abstaende(S, P, lam: Fr, t_max):
    """Listen d(t) und d̄(t) für t = 0 … t_max als exakte Brüche (Potenzen über ganze Zahlen)."""
    pi = gleichgewicht(S, lam)
    N = len(S)
    A, D = _ganzzahlig(P)
    Z = sum(lam ** bin(x).count("1") for x in S)
    w = [lam ** bin(x).count("1") for x in S]
    At = [[int(i == j) for j in range(N)] for i in range(N)]
    d, db = [], []
    for t in range(t_max + 1):
        Dt = D ** t
        # TV_x = ½ Σ |At[x][y]/Dt − w_y/Z|
        d.append(max(Fr(sum(abs(At[x][y] * Z - w[y] * Dt) for y in range(N)), 2 * Z * Dt) for x in range(N)))
        db.append(Fr(max(sum(abs(At[x][k] - At[y][k]) for k in range(N)) for x in range(N) for y in range(x + 1, N)), 2 * Dt) if N > 1 else Fr(0))
        At = _mal(At, A)
    return d, db


def aufsteigende_folgen(p):
    pos = {c: i for i, c in enumerate(p)}
    return 1 + sum(1 for v in range(len(p) - 1) if pos[v + 1] < pos[v])


def inverses_riffle_tv(n, k):
    """TV zur Gleichverteilung nach k inversen Riffle-Durchläufen (jede Karte ein Zufallsbit, Karten mit Bit 0 nach oben), alle Bitfolgen aufgezählt, in Brüchen."""
    verteilung = {tuple(range(n)): Fr(1)}
    for _ in range(k):
        neu = {}
        for stapel, pr in verteilung.items():
            for bits in product((0, 1), repeat=n):
                s = tuple(c for c, b in zip(stapel, bits) if b == 0) + tuple(c for c, b in zip(stapel, bits) if b == 1)
                neu[s] = neu.get(s, Fr(0)) + pr * Fr(1, 2 ** n)
        verteilung = neu
    return sum(abs(verteilung.get(p, Fr(0)) - Fr(1, factorial(n))) for p in permutations(range(n))) / 2


def vorwaerts_riffle_tv(n, k):
    """Zweiter Weg: die Formel Q^k(σ) = C(2^k + n − r, n)/2^{kn} über alle n! Anordnungen (r = Zahl der aufsteigenden Folgen), ohne Eulersche Zahlen."""
    a = 2 ** k
    gleich = Fr(1, factorial(n))
    return sum(abs(Fr(comb(a + n - aufsteigende_folgen(p), n), a ** n) - gleich) for p in permutations(range(n))) / 2


def wuerfel_voll_d(n, t_max):
    """d(t) der trägen Irrfahrt auf {0,1}^n aus der vollen 2^n-Kette (Vektoriteration ab (1,…,1)), als float-Liste."""
    import numpy as np
    N = 1 << n
    p = np.zeros(N)
    p[N - 1] = 1.0
    out = []
    for _ in range(t_max + 1):
        out.append(0.5 * np.abs(p - 1.0 / N).sum())
        q = 0.5 * p
        for v in range(n):
            q += 0.5 / n * p[np.arange(N) ^ (1 << v)]
        p = q
    return out
