"""Auswertung der Experimente und Verdikte (reine Funktionen: Verdikt = (Art, Text); Art ist der Name einer Streamlit-Funktion)."""

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import msz_chain as K
import msz_constants as C
import msz_families as F
import msz_format as FMT
import msz_mixing as M
import msz_reference as REF
import msz_simulation as S

E = 1.0 / math.e


# =====================================================================================================================================
# Experiment 1: die exakte Kurve einer kleinen Kette (drei Zahlen, eine Kette)
# =====================================================================================================================================
@dataclass(frozen=True)
class E1Settings:
    n: int = C.DEFAULT_N
    lam: float = C.DEFAULT_LAM
    kette: str = "Heat-bath"
    start: str = C.START_SCHLECHTESTER
    eps_name: str = C.SCHWELLE_STANDARD
    einheit: str = "Schritte"


@dataclass(frozen=True)
class E1Result:
    s: E1Settings
    N: int
    t_d: int                    # t_mix(1/4) mit d
    t_d_e: int                  # erstes t mit d ≤ 1/e
    t_dbar: int | None          # t̄_mix(1/4) mit d̄ (None: n über der Grenze für d̄)
    T_e: int | None             # T_mix(1/e) mit d̄
    d_vor: float                # d(t_d − 1)
    d_bei: float                # d(t_d)
    t_eps_d: int | None         # t_mix(ε) für die gewählte Schwelle
    t_eps_dbar: int | None
    start_index: int
    start_label: str
    t_start_14: int             # Mischzeit vom gewählten Start (1/4)
    t_start_eps: int | None
    ts: tuple
    d_kurve: tuple
    dbar_kurve: tuple | None
    start_kurve: tuple
    takt: float
    t_d_halb: int | None = None  # t_mix(1/8) mit d: obere Grenze für t̄_mix(1/4) (aus d̄ ≤ 2d)


def eps_wert(name: str) -> float:
    return C.SCHWELLEN[name]


def start_index(chain: K.RingChain, start: str) -> int:
    if start == C.START_SCHLECHTESTER:
        return K.maximalstart(chain)
    if start == C.START_LEER:
        return chain.index(0)
    labels = K.start_optionen(chain)
    return labels.index(start)


def start_liste(n: int) -> list[str]:
    """Auswahl der Starts: schlechtester, leer, dann alle übrigen Zustände mit ihrer Beschriftung (der leere Zustand steht schon als „leerer Ring“ da)."""
    ch = K.ring_chain(n, 1.0, "Heat-bath")
    return [C.START_SCHLECHTESTER, C.START_LEER] + K.start_optionen(ch)[1:]


def exp1_basis(s: E1Settings) -> E1Result:
    """Alles, was nicht an der gewählten Schwelle und Einheit hängt (Kurven, die drei Zahlen, der gewählte Start); die Felder der gewählten Schwelle gelten hier für 1/4."""
    ch = K.ring_chain(s.n, s.lam, s.kette)
    sp = M.ring_spectrum(s.n, s.lam, s.kette)
    mit_dbar = s.n <= C.E1_DBAR_N_MAX_LIVE
    t_d = M.t_mix(sp, 0.25)
    t_d_e = M.t_mix(sp, E)
    t_db = M.t_mix_dbar(sp, 0.25) if mit_dbar else None
    T_e = M.t_mix_dbar(sp, E) if mit_dbar else None
    x = start_index(ch, s.start)
    t_x14 = M.t_start(sp, x, 0.25)
    ref = t_db if t_db is not None else 2 * t_d
    ts = M.zeitraster(int(1.6 * ref) + 2, 80)
    d, db, dx = M.kurven(sp, ts, start=x, mit_dbar=mit_dbar)
    return E1Result(
        s, int(sp.N), t_d, t_d_e, t_db, T_e, M.d_t(sp, max(t_d - 1, 0)), M.d_t(sp, t_d), t_d, t_db, x, K.label(int(ch.states[x]), s.n),
        t_x14, t_x14, tuple(int(t) for t in ts), tuple(d), tuple(db) if db is not None else None, tuple(dx), K.takt(s.n, s.lam, s.kette),
        M.t_mix(sp, 0.125) if mit_dbar else None)


def exp1_schwelle(s: E1Settings) -> tuple:
    """(t_mix(ε) mit d, t̄_mix(ε) mit d̄ oder None, Mischzeit des gewählten Starts) für die gewählte Schwelle ε."""
    ch = K.ring_chain(s.n, s.lam, s.kette)
    sp = M.ring_spectrum(s.n, s.lam, s.kette)
    eps = eps_wert(s.eps_name)
    t_eps_db = M.t_mix_dbar(sp, eps) if s.n <= C.E1_DBAR_N_MAX_LIVE else None
    return M.t_mix(sp, eps), t_eps_db, M.t_start(sp, start_index(ch, s.start), eps)


def exp1_mit_schwelle(basis: E1Result, s: E1Settings, schwelle: tuple) -> E1Result:
    """Setzt die Felder der gewählten Schwelle und Einheit in das Basisergebnis ein."""
    return replace(basis, s=s, t_eps_d=schwelle[0], t_eps_dbar=schwelle[1], t_start_eps=schwelle[2])


def exp1(s: E1Settings) -> E1Result:
    return exp1_mit_schwelle(exp1_basis(s), s, exp1_schwelle(s))


def verdict_drei_zahlen(n: int, lam: float, kette: str, N: int, t_d: int, d_vor: float, d_bei: float, t_dbar, T_e, t_d_e, takt: float):
    """Drei Zahlen, eine Kette: d mit 1/4, d̄ mit 1/4, d̄ mit 1/e (die Zahlen sind die der laufenden Rechnung)."""
    text = (f"Ring n = {n}, λ = {FMT.zahl(lam)} ({N} Zustände), {kette}: d({t_d - 1}) = {FMT.dez(d_vor, 4)} liegt über 1/4, d({t_d}) = {FMT.dez(d_bei, 4)} darunter: "
            f"**{t_d} Schritte** (Abstand d, Schwelle 1/4). ")
    if t_dbar is None:
        text += (f"Der Abstand d̄ (alle Paare von Starts) wird bis n = {C.E1_DBAR_N_MAX_LIVE} live gerechnet; bei diesem n gilt er nur in den vorgerechneten Zellen der Tabelle. "
                 f"Mit d und der Schwelle 1/e ist es schon bei {t_d_e} Schritten so weit. ")
    else:
        text += (f"Mit d̄ ist die Schwelle 1/4 erst bei **{t_dbar}** erreicht, und die Mischzeit mit d̄ und der Schwelle 1/e (Propp/Wilson) ist **{T_e}**; "
                 f"mit d und der Schwelle 1/e wären es {t_d_e}. Drei Zahlen, eine Kette. ")
    text += f"In Abfertigungsdauern ({FMT.zahl(takt)} Schritte je Abfertigungsdauer): {FMT.dez(t_d / takt, 3)}."
    return "info", text


# =====================================================================================================================================
# Experiment 2: der Start entscheidet
# =====================================================================================================================================
@dataclass(frozen=True)
class E2Result:
    n: int
    lam: float
    N: int
    t_mix: int
    t_leer: int
    t_min: int
    t_mittel: float
    t_mittel_pi: float
    anteil_halb: float
    schlechteste_maximal: bool          # alle schlechtesten Starts haben ⌊n/2⌋ Teilchen
    alle_maximal_gleich: bool           # alle Maximalbelegungen haben dasselbe t_x
    zahl_schlechteste: int
    tx: tuple
    teilchen: tuple


def exp2(n: int, lam: float) -> E2Result:
    ch = K.ring_chain(n, lam, "Heat-bath")
    sp = M.ring_spectrum(n, lam, "Heat-bath")
    tx = M.t_alle_starts(sp)
    tm = int(tx.max())
    voll = n // 2
    return E2Result(
        n, lam, int(sp.N), tm, int(tx[ch.index(0)]), int(tx.min()), float(tx.mean()), float((tx * sp.pi).sum()), float((tx <= tm / 2).mean()),
        bool((ch.teilchen[tx == tm] == voll).all()), bool((tx[ch.teilchen == voll] == tm).all()), int((tx == tm).sum()),
        tuple(int(a) for a in tx), tuple(int(a) for a in ch.teilchen))


def verdict_start(r: E2Result):
    faktor = r.t_mix / r.t_leer if r.t_leer else math.inf
    wer = ("eine Maximalbelegung (jeder zweite Platz)" if r.schlechteste_maximal else "keine Maximalbelegung")
    gleich = "alle Maximalbelegungen sind gleich schlecht" if r.alle_maximal_gleich else "nicht alle Maximalbelegungen sind gleich schlecht"
    text = (f"n = {r.n}, λ = {FMT.zahl(r.lam)}, {r.N} Zustände, Heat-bath: der schlechteste Start ist {wer} und braucht **{r.t_mix} Schritte** ({gleich}), "
            f"der leere Ring {r.t_leer}, im Mittel über alle {r.N} Starts {FMT.dez(r.t_mittel, 0)} und mit π gewichtet {FMT.dez(r.t_mittel_pi, 0)}. "
            f"Wer aus dem leeren Ring startet, unterschätzt t_mix um das {FMT.dez(faktor, 1)}-fache. Ein Start, der selbst aus π gezogen wurde, hat gar keine Mischzeit (d = 0 in jedem Schritt).")
    return "info", text


# =====================================================================================================================================
# Experiment 3: Einschluss durch die Spektrallücke (die Falle, erste Gestalt)
# =====================================================================================================================================
@dataclass(frozen=True)
class Einschluss:
    raum: str
    n: int
    t_mix: int | None
    t_rel: float
    unten: float
    oben: float
    pi_min: float
    quelle: str                 # „live“ oder „vorgerechnet“
    extra: tuple = ()           # weitere (Name, Wert)-Paare (z. B. Eigenwert-Formel)

    @property
    def oben_durch_exakt(self) -> float | None:
        return self.oben / self.t_mix if self.t_mix else None

    @property
    def exakt_durch_unten(self) -> float | None:
        return self.t_mix / self.unten if (self.t_mix and self.unten > 0) else None


def einschluss_ring(n: int, lam: float, kette: str) -> Einschluss:
    sp = M.ring_spectrum(n, lam, kette)
    trel = M.t_rel(sp.mu)
    unten, oben = M.schranken(trel, float(sp.pi.min()))
    extra = ()
    if kette == "Heat-bath" and n % 2 == 0:
        g = M.g_gerades_n(lam)
        extra = (("1 − g/n (Formel)", 1.0 - g / n), ("zweitgrößter Eigenwert (gerechnet)", float(sp.mu[1])))
    return Einschluss(f"Ring n = {n}, λ = {FMT.zahl(lam)}, {kette}", n, M.t_mix(sp, 0.25), trel, unten, oben, float(sp.pi.min()), "live", extra)


def einschluss_kreis(n: int, traege: bool = True) -> Einschluss:
    t = F.kreis_t(n, 0.25, traege)
    g = F.kreis_luecke(n, traege)
    trel = math.inf if g <= 1e-15 else 1.0 / g
    unten, oben = M.schranken(trel, 1.0 / n) if math.isfinite(trel) else (math.inf, math.inf)
    return Einschluss(f"Kreis n = {n} ({'träge' if traege else 'nicht träge'})", n, t, trel, unten, oben, 1.0 / n, "live")


def einschluss_wuerfel(n: int) -> Einschluss:
    t = F.wuerfel_ts(n, (0.25,))[0.25]
    trel = float(n)
    pi_min = 2.0 ** (-n)
    unten = (trel - 1.0) * math.log(2.0)
    oben = trel * (math.log(4.0) + n * math.log(2.0))
    return Einschluss(f"Hyperwürfel n = {n} (träge)", n, t, trel, unten, oben, pi_min, "live")


def ring_einschluss_statistik() -> dict:
    """Die 30 Zellen (n = 6, 8, 10, 12, 14; λ = 0,5 bis 20; Heat-bath) aus der vorgerechneten Tabelle: obere Schranke durch exakt, exakt durch untere."""
    oben, unten = [], []
    for n in (6, 8, 10, 12, 14):
        for lam in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
            z = REF.ring_zelle(n, lam, "Heat-bath")
            oben.append(z["oben"] / z["t_mix"])
            unten.append(z["t_mix"] / z["unten"])
    return {"zellen": len(oben), "oben_min": min(oben), "oben_max": max(oben), "oben_median": float(np.median(oben)),
            "unten_min": min(unten), "unten_max": max(unten), "unten_median": float(np.median(unten))}


def lauf_verhaeltnis_trel() -> dict:
    """t_mix/t_rel für gerade n = 6 bis 16 und die acht λ der Tabelle (Heat-bath): kleinster und größter Wert."""
    werte = []
    for n in range(6, 17, 2):
        for lam in C.LAM_TABELLE:
            z = REF.ring_zelle(n, lam, "Heat-bath")
            werte.append(z["t_mix"] / z["t_rel"])
    return {"zellen": len(werte), "min": min(werte), "max": max(werte)}


def verdict_einschluss_ring(e: Einschluss, stat: dict, kreis: Einschluss, wuerfel: Einschluss, verh: dict):
    text = (f"{e.raum}: t_rel = {FMT.dez(e.t_rel, 1)}; Einschluss {FMT.dez(e.unten, 1)} ≤ **{e.t_mix}** ≤ {FMT.dez(e.oben, 1)}: die obere Schranke liegt beim "
            f"{FMT.dez(e.oben_durch_exakt, 1)}-fachen des exakten Werts, der exakte Wert beim {FMT.dez(e.exakt_durch_unten, 2)}-fachen der unteren. "
            f"In den {stat['zellen']} vorgerechneten Zellen (Heat-bath, n = 6 bis 14, λ = 0,5 bis 20) liegt die obere Schranke beim {FMT.dez(stat['oben_min'], 1)}- bis "
            f"{FMT.dez(stat['oben_max'], 1)}-fachen (Median {FMT.dez(stat['oben_median'], 1)}), beim Kreis n = {kreis.n} beim {FMT.dez(kreis.oben_durch_exakt, 1)}-fachen, "
            f"beim Hyperwürfel n = {wuerfel.n} beim {FMT.dez(wuerfel.oben_durch_exakt, 1)}-fachen. "
            f"Im Ring liegt t_mix beim {FMT.dez(verh['min'], 2)}- bis {FMT.dez(verh['max'], 2)}-fachen von t_rel (gerade n = 6 bis 16, acht λ von 0,5 bis 100, Heat-bath). "
            "Ein Einschluss, keine Vorhersage.")
    return "warning", text


def verdict_einschluss_raum(e: Einschluss):
    if e.t_mix is None:
        return "warning", (f"{e.raum}: die Kette ist periodisch (Eigenwert −1), der Abstand d fällt nie unter 1/2: t_mix gibt es nicht, als Zahl ist es unendlich. "
                           "Die Schranke aus der Spektrallücke sagt dasselbe: die absolute Lücke ist 0.")
    return "info", (f"{e.raum}: t_rel = {FMT.dez(e.t_rel, 1)}; Einschluss {FMT.dez(e.unten, 1)} ≤ **{e.t_mix}** ≤ {FMT.dez(e.oben, 1)}: die obere Schranke liegt beim "
                    f"{FMT.dez(e.oben_durch_exakt, 1)}-fachen des exakten Werts, der exakte Wert beim {FMT.dez(e.exakt_durch_unten, 2)}-fachen der unteren.")


# =====================================================================================================================================
# Experiment 4: Wie wächst die Mischzeit mit der Größe?
# =====================================================================================================================================
@dataclass(frozen=True)
class WachstumZeile:
    n: int
    ts: dict                    # {0.9: t, 0.5: t, 0.25: t, 0.1: t}
    N: int | None = None

    @property
    def t_mix(self) -> int:
        return self.ts[0.25]

    @property
    def fenster(self) -> float:
        return (self.ts[0.1] - self.ts[0.9]) / self.ts[0.5]


def kreis_zeilen(ns) -> list[WachstumZeile]:
    return [WachstumZeile(int(n), F.kreis_ts(int(n))) for n in ns]


def wuerfel_zeilen(ns) -> list[WachstumZeile]:
    return [WachstumZeile(int(n), F.wuerfel_ts(int(n))) for n in ns]


def ring_zeilen(ns) -> list[WachstumZeile]:
    out = []
    for n in ns:
        f = REF.fenster(int(n))
        out.append(WachstumZeile(int(n), {float(k): v for k, v in f["t"].items()}, f["N"]))
    return out


def kreis_kurven(zeilen, schritte: int = 60):
    """d(t) gegen t/t_mix für mehrere n (Kreis)."""
    out = {}
    for z in zeilen:
        ts = np.unique(np.round(np.linspace(0, 3.0 * z.t_mix, schritte)).astype(int))
        out[z.n] = (ts / z.t_mix, F.kreis_kurve(z.n, ts))
    return out


def wuerfel_kurven(zeilen, schritte: int = 80):
    out = {}
    for z in zeilen:
        kurve = F.wuerfel_kurve(z.n, F.wuerfel_zeit_schranke(z.n))
        ts = np.unique(np.round(np.linspace(0, len(kurve) - 1, schritte)).astype(int))
        out[z.n] = (ts / z.t_mix, kurve[ts])
    return out


def ring_kurven(zeilen):
    out = {}
    for z in zeilen:
        f = REF.fenster(z.n)["kurve"]
        out[z.n] = (np.array(f["t"]) / z.t_mix, np.array(f["d"]))
    return out


def _spanne(lo: float, hi: float, stellen: int) -> str:
    """„0,0944 bis 0,0949“, bei gleichen gerundeten Werten nur „0,0949“."""
    a, b = FMT.dez(lo, stellen), FMT.dez(hi, stellen)
    return a if a == b else f"{a} bis {b}"


def verdict_kreis(zeilen):
    z = [a for a in zeilen]
    r = [a.t_mix / a.n ** 2 for a in z]
    w = [a.fenster for a in z]
    ns = ", ".join(str(a.n) for a in z)
    gross = [a for a in z if a.n >= 100]
    teil = ""
    if gross:
        rg = [a.t_mix / a.n ** 2 for a in gross]
        teil = f" Für n ≥ 100 ({', '.join(str(a.n) for a in gross)}) liegt t_mix/n² bei {_spanne(min(rg), max(rg), 4)}."
    text = (f"Kreis (träge Irrfahrt), n = {ns}: t_mix/n² liegt bei {_spanne(min(r), max(r), 4)}.{teil} "
            f"Das Fenster (t(0,1) − t(0,9))/t(0,5) liegt bei {_spanne(min(w), max(w), 2)}"
            + (f" und fällt mit n nicht (kein Cutoff; Messung bis n = {max(a.n for a in z)}). " if len(z) > 1 and w[-1] >= w[0] - 0.05 else ". ") +
            "Schranke n²/32 ≤ t_mix ≤ n² (Levin/Peres/Wilmer, Abschnitt 5.3.2) in " +
            f"{sum(1 for a in z if a.n ** 2 / 32 <= a.t_mix <= a.n ** 2)} von {len(z)} Fällen.")
    return "info", text


def verdict_wuerfel(zeilen):
    z = list(zeilen)
    rest = [(a.t_mix - 0.5 * a.n * math.log(a.n)) / a.n for a in z]
    w = [a.fenster for a in z]
    kleinste, groesste = z[0], z[-1]
    text = (f"Hyperwürfel (träge Irrfahrt), n = {', '.join(str(a.n) for a in z)}: (t_mix − ½ n ln n)/n liegt bei {FMT.dez(min(rest), 3)} bis {FMT.dez(max(rest), 3)}. "
            f"Das Fenster (t(0,1) − t(0,9))/t(0,5) geht von {FMT.dez(w[0], 2)} (n = {kleinste.n}) auf {FMT.dez(w[-1], 2)} (n = {groesste.n}): ")
    text += ("es wird mit n schmaler, der Übergang wird relativ zu t_mix schärfer (Cutoff; Satz 18.3 gilt im Grenzwert n → ∞). " if len(z) > 1 and w[-1] < w[0] else
             "mit dieser Auswahl ist kein Fallen zu sehen. ")
    text += "Mit n ≤ 12 sieht man ihn nicht."
    return "info", text


def verdict_ring_fenster(zeilen):
    z = list(zeilen)
    w = [a.fenster for a in z]
    text = (f"Hard-Core-Ring (λ = 1, Heat-bath, Start Maximalbelegung), n = {', '.join(str(a.n) for a in z)}: das Fenster (t(0,1) − t(0,9))/t(0,5) liegt bei "
            f"{', '.join(FMT.dez(x, 2) for x in w)}. ")
    if len(z) > 1 and w[-1] < w[0]:
        text += "Es fällt mit n wie beim Hyperwürfel; "
    text += "bis n = 24 und aus einem Start ist das eine Messung, kein Beleg für einen Cutoff."
    return "info", text


def karten_tabelle(n: int, k_max: int = C.KARTEN_K_MAX) -> list[tuple[int, float]]:
    return [(k, F.riffle_tv(n, k)) for k in range(1, k_max + 1)]


def verdict_karten(n: int):
    t14 = F.riffle_t(n, 0.25)
    t01 = F.riffle_t(n, 0.01)
    if n < 2:
        return "info", "Eine Karte ist immer gemischt: der Abstand ist 0."
    text = (f"{n} Karten, Riffle-Mischen (Gilbert/Shannon/Reeds): der Abstand liegt nach {max(t14 - 1, 1)} Mischungen bei {FMT.dez(F.riffle_tv(n, max(t14 - 1, 1)), 4)}, "
            f"nach {t14} bei {FMT.dez(F.riffle_tv(n, t14), 4)}: t_mix(1/4) = **{t14} Mischungen**. Unter 0,01 liegt er erst nach {t01} Mischungen "
            f"({max(t01 - 1, 1)} geben {FMT.dez(F.riffle_tv(n, max(t01 - 1, 1)), 4)}, {t01} geben {FMT.dez(F.riffle_tv(n, t01), 4)}). "
            "Die Einheit ist die Mischung (ein Riffle-Durchlauf), nicht der Schritt einer Kette.")
    return "info", text


def mischer_zeile(n: int) -> dict:
    """Ein Vergleich der Einheiten bei n Karten: Riffle (Mischungen), Transpositionen und Top-zu-zufällig (Schritte); aus der vorgerechneten Tabelle."""
    tr = REF.mischer("Transpositionen")[str(n)]
    tt = REF.mischer("Top-zu-zufällig")[str(n)]
    return {"n": n, "riffle": F.riffle_t(n), "transpositionen": tr["0.25"], "top_zu_zufaellig": tt["0.25"]}


# =====================================================================================================================================
# Experiment 5: Schätzen aus Läufen (die Falle, zweite und dritte Gestalt)
# =====================================================================================================================================
@dataclass(frozen=True)
class E5Settings:
    R: int = 300
    kennzahl: str = "ganze Verteilung"
    reps: int = C.SIM_REPS_DEFAULT
    seed: int = C.SIM_SEED_DEFAULT


@dataclass(frozen=True)
class E5Result:
    s: E5Settings
    t_exakt: int
    t_exakt_zahl: int
    rausch: float
    rausch_zahl: float
    rausch_naeherung: float
    schaetzungen: tuple          # Mischzeiten je Lauf (None: nie erreicht), für die gewählte Kennzahl
    zusammen: dict
    kurve_emp: tuple             # empirischer Abstand des ersten Laufs (gewählte Kennzahl)
    kurve_exakt: tuple


@lru_cache(maxsize=16)
def _wiederholungen(R: int, reps: int, seed: int) -> dict:
    return S.wiederholungen(R, reps, seed)


def exp5(s: E5Settings) -> E5Result:
    ref = S.referenz()
    w = _wiederholungen(s.R, s.reps, s.seed)
    zahl = s.kennzahl == "Teilchenzahl"
    ts = w["t_zahl"] if zahl else w["t"]
    rg, rg_z = S.rauschgrenzen(s.R)
    erster = w["laeufe"][0]
    return E5Result(
        s, ref.t_exakt, ref.t_exakt_zahl, rg, rg_z, S.rauschgrenze_naeherung(s.R), tuple(ts), S.zusammenfassung(ts),
        tuple(erster.tv_zahl if zahl else erster.tv), tuple(ref.kurve_zahl if zahl else ref.kurve))


def verdict_simulation(r: E5Result):
    zahl = r.s.kennzahl == "Teilchenzahl"
    exakt = r.t_exakt_zahl if zahl else r.t_exakt
    rg = r.rausch_zahl if zahl else r.rausch
    z = r.zusammen
    kopf = (f"n = {C.SIM_N}, λ = {FMT.zahl(C.SIM_LAM)}, Heat-bath, Start Maximalbelegung, R = {r.s.R} Ketten, {z['von']} Läufe (Zufalls-Seed {r.s.seed}), "
            f"Kennzahl {'Teilchenzahl' if zahl else 'ganze Verteilung'}: exakt {exakt} Schritte; Rauschgrenze {FMT.dez(rg, 3)}. ")
    if z["nie"] == z["von"]:
        mitte = (f"Keiner der {z['von']} Läufe erreicht die Schwelle 1/4 bis t = {C.SIM_T_MAX}" +
                 (": die Rauschgrenze liegt über 1/4, selbst eine perfekt gemischte Stichprobe von R Ketten unterschreitet sie im Mittel nicht. " if rg > 0.25 else ". "))
    else:
        mitte = (f"Die geschätzte Mischzeit liegt im Mittel bei {FMT.dez(z['mittel'], 1)} (kleinster Wert {z['min']}, größter {z['max']}), {z['nie']} von {z['von']} Läufen erreichen die Schwelle nie. ")
    if zahl:
        schluss = (f"Die Kennzahl misst etwas anderes: für die Frage „ist die ganze Verteilung erreicht“ liegt sie weit daneben "
                   f"(exakt {r.t_exakt_zahl} gegen {r.t_exakt} Schritte).")
    else:
        schluss = "Auch eine perfekt gemischte Stichprobe hat im Mittel nie den Abstand 0; die Rauschgrenze fällt etwa wie 1/√R."
    return ("warning" if (z["nie"] or (z["mittel"] is not None and abs(z["mittel"] - exakt) > 0.1 * exakt)) else "success"), kopf + mitte + schluss


def stichprobe_gleichverteilt(R: int, Nz: int) -> float:
    """Erwarteter Abstand zwischen der Gleichverteilung auf Nz Zuständen und der empirischen Verteilung von R exakten Ziehungen (exakt, binomial)."""
    return S.rauschgrenze_exakt(np.full(Nz, 1.0 / Nz), R)


# =====================================================================================================================================
# Experiment 6 (Tabelle): wo die exakte Rechnung endet
# =====================================================================================================================================
def einordnung_zeilen() -> list[dict]:
    z16 = REF.ring_zelle(16, 1.0, "Heat-bath")
    f24 = REF.fenster(24)
    return [
        {"raum": "Hard-Core-Ring, n = 16", "zustaende": K.lucas(16), "mischzeit": f"{z16['t_mix']} Schritte (λ = 1, Heat-bath, alle Starts, dichte Rechnung)"},
        {"raum": "Hard-Core-Ring, n = 24", "zustaende": K.lucas(24), "mischzeit": f"{f24['t']['0.25']} Schritte (λ = 1, Heat-bath, nur aus der Maximalbelegung: Untergrenze)"},
        {"raum": "Hard-Core-Torus 8 × 8", "zustaende": K.torus_zustaende(8, 8), "mischzeit": "nicht exakt berechenbar (die Zustandszahl ist zu groß für eine Matrix)"},
        {"raum": "52 Karten", "zustaende": math.factorial(52), "mischzeit": f"{F.riffle_t(52)} Riffle-Mischungen, exakt nur über die Struktur (Eulersche Zahlen), nicht über eine Matrix"},
    ]
