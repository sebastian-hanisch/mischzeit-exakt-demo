"""Wie schnell mischt eine Kette? Mischzeit exakt gerechnet - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünf Experimente: die exakte Kurve einer kleinen Kette (drei Zahlen, eine Kette), der Start, der Einschluss durch die Spektrallücke, das Wachstum mit der Größe
(Kreis, Hyperwürfel, Hard-Core-Ring, Karten) und das Schätzen aus Läufen. Aufbau nach DEMO-PLAYBOOK §2 (Schnellstart in der Seitenleiste, Regler beim Experiment,
Kernfragen vor dem Expander).

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import msz_chain as K
import msz_constants as C
import msz_evaluation as ev
import msz_families as F
import msz_format as FMT
import msz_mixing as M
import msz_reference as REF
import msz_texts as T
from msz_layout import dunkles_design, kopf_css
from msz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from msz_visualization import (
    build_einschluss,
    build_karten,
    build_kurven,
    build_schaetzungen,
    build_simulation,
    build_start_hist,
    build_tmix_ueber_n,
    build_wachstum_kurven,
)

st.set_page_config(page_title="Wie schnell mischt eine Kette? Mischzeit exakt gerechnet – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _exp1_basis(s):
    return ev.exp1_basis(s)


@st.cache_data(show_spinner=False)
def _exp1_schwelle(s):
    return ev.exp1_schwelle(s)


@st.cache_data(show_spinner=False)
def _exp2(n, lam):
    return ev.exp2(n, lam)


@st.cache_data(show_spinner=False)
def _e2_zeilen(n):
    return [ev.exp2(n, lam) for lam in C.E2_LAMS]


@st.cache_data(show_spinner=False)
def _ring_einschluss(n, lam, kette):
    return ev.einschluss_ring(n, lam, kette)


@st.cache_data(show_spinner=False)
def _kreis_einschluss(n, traege):
    return ev.einschluss_kreis(n, traege)


@st.cache_data(show_spinner=False)
def _wuerfel_einschluss(n):
    return ev.einschluss_wuerfel(n)


@st.cache_data(show_spinner=False)
def _kreis_zeilen(ns):
    return ev.kreis_zeilen(ns)


@st.cache_data(show_spinner=False)
def _wuerfel_zeilen(ns):
    return ev.wuerfel_zeilen(ns)


@st.cache_data(show_spinner=False)
def _kreis_kurven(ns):
    return ev.kreis_kurven(_kreis_zeilen(ns))


@st.cache_data(show_spinner=False)
def _wuerfel_kurven(ns):
    return ev.wuerfel_kurven(_wuerfel_zeilen(ns))


@st.cache_data(show_spinner=False)
def _exp5(s):
    return ev.exp5(s)


def _zeit(t, einheit, n, lam, kette):
    """Mischzeit in der gewählten Einheit als Text (None: nicht berechnet)."""
    if t is None:
        return "–"
    wert = M.in_einheit(t, einheit, n, lam, kette)
    return FMT.ganz(round(wert)) if einheit == "Schritte" else FMT.dez(wert, 2 if einheit.startswith("Durchl") else 3)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in msz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⏱️ Wie schnell mischt eine Kette? Mischzeit exakt gerechnet")
st.markdown(T.EINLEITUNG)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/mischzeit-exakt-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich vor, Sie mischen Karten und fragen, wie oft man mischen muss, bis die Ausgangsordnung keine Rolle mehr spielt. Diese Demo rechnet das für eine Zufallswanderung exakt aus, an einem ausgedachten Modell: Andockstellen im Ring, von denen nie zwei benachbarte belegt sind. Sie zeigt, dass es **die** Mischzeit nicht gibt: Je nach Abstandsmaß, Schwelle und Startzustand kommt für dieselbe Kette eine andere Zahl heraus. Danach folgen drei Fallen: eine Schranke aus der Spektrallücke, die in der Demo weit danebenliegt, eine Schätzung aus zu wenigen simulierten Ketten und eine Kennzahl, die früher zur Ruhe kommt als die ganze Verteilung. Alles gilt in diesem Modell, nicht in einem echten Betrieb.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/mischzeit-exakt-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Mischzeit", expanded=True):
    st.markdown(T.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.caption("Die Regler der Experimente stehen beim jeweiligen Experiment. Hier steht nur der Zufalls-Seed für die Simulation in Experiment 5.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)
seed = int(seed)

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------
st.markdown("## ❓ Kernfragen")
st.markdown(T.KERNFRAGEN)

# =====================================================================================================================================
# Experiment 1
# =====================================================================================================================================
st.subheader("🔬 Experiment 1: Die exakte Kurve einer kleinen Kette")
st.caption(T.E1_CAPTION)
c1, c2, c3 = st.columns(3)
with c1:
    n1 = st.slider("Plätze n", *bounds("e1_n"), key="e1_n", help="Hard-Core-Ring mit n Plätzen; bis 12 Plätze (322 Zustände) wird live gerechnet.")
    kette1 = st.radio("Kette", C.KETTEN, key="e1_kette", horizontal=True, help="Heat-bath: Platz neu aus der bedingten Verteilung. Metropolis: Umkehren mit Annahmewahrscheinlichkeit.")
with c2:
    lam1 = st.slider("λ (Gewicht je belegtem Platz)", *bounds("e1_lam"), step=C.LAM_STEP, key="e1_lam", help="Im Verlustnetz-Bild das Angebot je Andockstelle in Erlang.")
    eps1 = st.select_slider("Schwelle ε", options=list(C.SCHWELLEN), key="e1_eps", help="Wie nah an π die Kette liegen soll: 1/4 ist die Konvention der Mischzeit t_mix.")
with c3:
    optionen = ev.start_liste(int(n1))
    if st.session_state.get("e1_start") not in optionen:
        st.session_state["e1_start"] = C.START_SCHLECHTESTER
    start1 = st.selectbox("Start", optionen, key="e1_start", help="Schlechtester Start: die Maximalbelegung (jeder zweite Platz). Oder ein beliebiger Zustand.")
    einheit1 = st.radio("Zeiteinheit", C.EINHEITEN, key="e1_unit", help="Durchlauf = n Schritte; Abfertigungsdauer = n(1+λ) Schritte beim Heat-bath, n·max(1, λ) bei Metropolis.")

s1 = ev.E1Settings(int(n1), float(lam1), kette1, start1, eps1, einheit1)
# die Rechnung hängt nur an (n, λ, Kette, Start); Schwelle und Einheit ändern das Basisergebnis nicht (eigener Zwischenspeicher)
r1 = ev.exp1_mit_schwelle(_exp1_basis(ev.E1Settings(s1.n, s1.lam, s1.kette, s1.start)), s1, _exp1_schwelle(ev.E1Settings(s1.n, s1.lam, s1.kette, s1.start, s1.eps_name)))
st.plotly_chart(build_kurven(r1, einheit1), width="stretch", key="e1_chart")
zeilen = [
    ("d, Schwelle 1/4 (t_mix)", r1.t_d), ("d̄, Schwelle 1/4", r1.t_dbar), ("d̄, Schwelle 1/e (T_mix)", r1.T_e), ("d, Schwelle 1/e", r1.t_d_e),
    (f"d, gewählte Schwelle {eps1}", r1.t_eps_d), (f"d̄, gewählte Schwelle {eps1}", r1.t_eps_dbar), (f"gewählter Start ({r1.start_label}), Schwelle {eps1}", r1.t_start_eps),
]
tab = pd.DataFrame({"Größe": [a for a, _ in zeilen], f"in {einheit1}": [_zeit(b, einheit1, s1.n, s1.lam, s1.kette) for _, b in zeilen],
                    "Schritte": [FMT.ganz(b) if b is not None else "–" for _, b in zeilen]})
st.dataframe(tab, hide_index=True, width="stretch")
if r1.s.n <= 4:
    ch = K.ring_chain(s1.n, s1.lam, s1.kette)
    namen = K.start_optionen(ch)
    st.markdown(f"**Die {ch.N} Zustände mit ihrer Übergangsmatrix** (Zeile = von, Spalte = nach; Zeilensummen 1):")
    st.dataframe(pd.DataFrame(ch.dense(), index=namen, columns=namen).round(4), width="stretch")
art, text = ev.verdict_drei_zahlen(s1.n, s1.lam, s1.kette, r1.N, r1.t_d, r1.d_vor, r1.d_bei, r1.t_dbar, r1.T_e, r1.t_d_e, r1.takt)
getattr(st, art)(text)
if r1.t_dbar is not None:
    st.caption(f"Eigenschaft in jedem Fall: d(t) ≤ d̄(t) ≤ 2·d(t); darum gilt t_mix(ε) ≤ t̄_mix(ε) ≤ t_mix(ε/2) (Beweis im 📐-Expander). Hier (ε = 1/4): {r1.t_d} ≤ {r1.t_dbar} ≤ {r1.t_d_halb}.")

st.markdown("---")

# =====================================================================================================================================
# Experiment 2
# =====================================================================================================================================
st.subheader("🔬 Experiment 2: Der Start entscheidet")
st.caption(T.E2_CAPTION)
c1, c2 = st.columns(2)
with c1:
    n2 = st.slider("Plätze n", *bounds("e2_n"), key="e2_n", help="Alle Starts werden live gerechnet (bis 123 Zustände).")
with c2:
    lam2 = st.slider("λ", *bounds("e2_lam"), step=C.LAM_STEP, key="e2_lam", help="Gewicht je belegtem Platz.")
r2 = _exp2(int(n2), float(lam2))
st.plotly_chart(build_start_hist(r2), width="stretch", key="e2_chart")
rows2 = _e2_zeilen(int(n2))
tab2 = pd.DataFrame({
    "λ": [FMT.zahl(r.lam) for r in rows2], "schlechtester Start (t_mix)": [r.t_mix for r in rows2], "leerer Ring": [r.t_leer for r in rows2],
    "Mittel über alle Starts": [FMT.dez(r.t_mittel, 1) for r in rows2], "Mittel mit π gewichtet": [FMT.dez(r.t_mittel_pi, 1) for r in rows2],
    "Anteil der Starts mit t_x ≤ t_mix/2": [FMT.dez(r.anteil_halb, 2) for r in rows2],
})
st.markdown(f"**Vier Werte von λ bei n = {int(n2)}** (Schritte bis zum Abstand 1/4):")
st.dataframe(tab2, hide_index=True, width="stretch")
art, text = ev.verdict_start(r2)
getattr(st, art)(text)
st.caption("Das Maximum über alle Starts ist die Definition von t_mix. In der Praxis kennt man den schlechtesten Start nicht; wer aus einem Start startet, der selbst aus π gezogen wurde, hat gar keine Mischzeit.")

st.markdown("---")

# =====================================================================================================================================
# Experiment 3
# =====================================================================================================================================
st.subheader("🔬 Experiment 3: Spektrallücke und Einschluss (die Falle, erste Gestalt)")
st.caption(T.E3_CAPTION)
stat = ev.ring_einschluss_statistik()
verh = ev.lauf_verhaeltnis_trel()
kreis_ref = _kreis_einschluss(100, True)
wuerfel_ref = _wuerfel_einschluss(400)
tab_r, tab_k, tab_w = st.tabs(["Hard-Core-Ring", "Kreis", "Hyperwürfel"])
with tab_r:
    c1, c2, c3 = st.columns(3)
    with c1:
        n3 = st.slider("Plätze n", *bounds("e3_n"), key="e3_n", help="Bis 12 Plätze wird live gerechnet.")
    with c2:
        lam3 = st.slider("λ", *bounds("e3_lam"), step=C.LAM_STEP, key="e3_lam", help="Gewicht je belegtem Platz.")
    with c3:
        kette3 = st.radio("Kette", C.KETTEN, key="e3_kette", horizontal=True)
    e3 = _ring_einschluss(int(n3), float(lam3), kette3)
    st.plotly_chart(build_einschluss(e3), width="stretch", key="e3_chart_ring")
    if e3.extra:
        st.caption(f"Gerades n, Heat-bath: der Eigenwert 1 − g(λ)/n nach der Formel ist {FMT.dez(e3.extra[0][1], 6)}, der gerechnete zweitgrößte Eigenwert {FMT.dez(e3.extra[1][1], 6)} (gemessen, Invarianz bewiesen; 📐-Expander).")
    art, text = ev.verdict_einschluss_ring(e3, stat, kreis_ref, wuerfel_ref, verh)
    getattr(st, art)(text)
    with st.expander("Tabelle: obere Schranke durch exakt | exakt durch untere Schranke (Heat-bath, vorgerechnet)"):
        lams = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
        zeilen3 = []
        for nn in (6, 8, 10, 12, 14):
            zeile = {"n": nn}
            for lam in lams:
                z = REF.ring_zelle(nn, lam, "Heat-bath")
                zeile[f"λ = {FMT.zahl(lam)}"] = f"{FMT.dez(z['oben'] / z['t_mix'], 1)} | {FMT.dez(z['t_mix'] / z['unten'], 2)}"
            zeilen3.append(zeile)
        st.dataframe(pd.DataFrame(zeilen3), hide_index=True, width="stretch")
        st.caption("Vorgerechnet (tools/msz_make_tables.py), exaktes t_mix über alle Starts. Ein Wert „4,2 | 1,55“ heißt: obere Schranke 4,2-mal t_mix, t_mix 1,55-mal die untere Schranke.")
with tab_k:
    c1, c2 = st.columns(2)
    with c1:
        nk = st.slider("Punkte n des Kreises", *bounds("e3_kreis"), key="e3_kreis", help="Träge Irrfahrt auf dem n-Kreis, exakt über die Fourier-Eigenwerte.")
    with c2:
        tr = st.radio("Irrfahrt", ("träge", "nicht träge"), key="e3_traege", horizontal=True, help="Nicht träge: je 1/2 nach links und rechts. Bei geradem n periodisch.")
    ek = _kreis_einschluss(int(nk), tr == "träge")
    if ek.t_mix is not None:
        st.plotly_chart(build_einschluss(ek), width="stretch", key="e3_chart_kreis")
    art, text = ev.verdict_einschluss_raum(ek)
    getattr(st, art)(text)
with tab_w:
    nw = st.slider("Koordinaten n des Hyperwürfels", *bounds("e3_wuerfel"), key="e3_wuerfel", help="Träge Irrfahrt auf {0,1}^n, gerechnet über die Zählkette des Hamming-Gewichts.")
    ew = _wuerfel_einschluss(int(nw))
    st.plotly_chart(build_einschluss(ew), width="stretch", key="e3_chart_wuerfel")
    art, text = ev.verdict_einschluss_raum(ew)
    getattr(st, art)(text)

st.markdown("---")

# =====================================================================================================================================
# Experiment 4
# =====================================================================================================================================
st.subheader("🔬 Experiment 4: Wie wächst die Mischzeit mit der Größe?")
st.caption(T.E4_CAPTION)
tk, tw, tr_, tc = st.tabs(["Kreis", "Hyperwürfel", "Hard-Core-Ring (λ = 1)", "Kartenmischen"])
with tk:
    ns = st.multiselect("Punkte n des Kreises (träge Irrfahrt)", list(C.KREIS_N_AUSWAHL), key="e4_kreis")
    if ns:
        ns = tuple(sorted(ns))
        zk = _kreis_zeilen(ns)
        st.plotly_chart(build_wachstum_kurven(_kreis_kurven(ns)), width="stretch", key="e4_kreis_kurven")
        st.dataframe(pd.DataFrame({"n": [z.n for z in zk], "t_mix": [FMT.ganz(z.t_mix) for z in zk], "t_mix / n²": [FMT.dez(z.t_mix / z.n ** 2, 4) for z in zk],
                                   "t(0,9)": [FMT.ganz(z.ts[0.9]) for z in zk], "t(0,5)": [FMT.ganz(z.ts[0.5]) for z in zk], "t(0,1)": [FMT.ganz(z.ts[0.1]) for z in zk],
                                   "Fenster (t(0,1) − t(0,9)) / t(0,5)": [FMT.dez(z.fenster, 2) for z in zk]}), hide_index=True, width="stretch")
        art, text = ev.verdict_kreis(zk)
        getattr(st, art)(text)
    else:
        st.info("Mindestens ein n wählen.")
with tw:
    nw4 = st.multiselect("Koordinaten n des Hyperwürfels (träge Irrfahrt)", list(C.WUERFEL_N_AUSWAHL), key="e4_wuerfel")
    if nw4:
        nw4 = tuple(sorted(nw4))
        zw = _wuerfel_zeilen(nw4)
        st.plotly_chart(build_wachstum_kurven(_wuerfel_kurven(nw4)), width="stretch", key="e4_wuerfel_kurven")
        st.dataframe(pd.DataFrame({"n": [z.n for z in zw], "t_mix": [FMT.ganz(z.t_mix) for z in zw], "(t_mix − ½ n ln n) / n": [FMT.dez((z.t_mix - 0.5 * z.n * math.log(z.n)) / z.n, 3) for z in zw],
                                   "t(0,9)": [FMT.ganz(z.ts[0.9]) for z in zw], "t(0,5)": [FMT.ganz(z.ts[0.5]) for z in zw], "t(0,1)": [FMT.ganz(z.ts[0.1]) for z in zw],
                                   "Fenster (t(0,1) − t(0,9)) / t(0,5)": [FMT.dez(z.fenster, 2) for z in zw]}), hide_index=True, width="stretch")
        art, text = ev.verdict_wuerfel(zw)
        getattr(st, art)(text)
    else:
        st.info("Mindestens ein n wählen.")
with tr_:
    nr = st.multiselect("Plätze n des Hard-Core-Rings (λ = 1, Heat-bath, Start Maximalbelegung)", list(C.RING_FENSTER_N), key="e4_ring",
                        help="n = 8 und 12 stimmen mit der Live-Rechnung überein; n = 16, 20, 24 sind vorgerechnet (Vektoriteration).")
    if nr:
        nr = tuple(sorted(nr))
        zr = ev.ring_zeilen(nr)
        st.plotly_chart(build_wachstum_kurven(ev.ring_kurven(zr)), width="stretch", key="e4_ring_kurven")
        st.dataframe(pd.DataFrame({"n": [z.n for z in zr], "Zustände": [FMT.ganz(z.N) for z in zr], "t_mix": [z.t_mix for z in zr],
                                   "t(0,9)": [z.ts[0.9] for z in zr], "t(0,5)": [z.ts[0.5] for z in zr], "t(0,1)": [z.ts[0.1] for z in zr],
                                   "Fenster (t(0,1) − t(0,9)) / t(0,5)": [FMT.dez(z.fenster, 2) for z in zr]}), hide_index=True, width="stretch")
        art, text = ev.verdict_ring_fenster(zr)
        getattr(st, art)(text)
    else:
        st.info("Mindestens ein n wählen.")
with tc:
    nc = st.slider("Zahl der Karten", *bounds("e4_karten"), key="e4_karten", help="Riffle-Mischen nach Gilbert/Shannon/Reeds; der Abstand folgt aus einer exakten Formel (Eulersche Zahlen), nicht aus einer Matrix.")
    tabelle_k = ev.karten_tabelle(int(nc))
    st.plotly_chart(build_karten(int(nc), tabelle_k), width="stretch", key="e4_karten_chart")
    st.dataframe(pd.DataFrame([{**{"Mischungen": "Abstand"}, **{str(k): FMT.dez(d, 4) for k, d in tabelle_k}}]), hide_index=True, width="stretch")
    art, text = ev.verdict_karten(int(nc))
    getattr(st, art)(text)
    st.markdown("**t_mix(1/4) für verschiedene Kartenzahlen** (Riffle-Mischen, exakt):")
    st.dataframe(pd.DataFrame([{"Karten": "t_mix(1/4) in Mischungen", **{str(n): str(F.riffle_t(n)) for n in C.KARTEN_N_TABELLE}}]), hide_index=True, width="stretch")
    st.markdown("**Andere Mischverfahren, andere Einheit** (vorgerechnet, Start Identität, t_mix(1/4)):")
    mz = [ev.mischer_zeile(n) for n in C.MISCHER_N]
    st.dataframe(pd.DataFrame({"Karten": [z["n"] for z in mz], "Riffle (Mischungen)": [z["riffle"] for z in mz], "zufällige Transpositionen (Schritte)": [z["transpositionen"] for z in mz],
                               "Top-zu-zufällig (Schritte)": [z["top_zu_zufaellig"] for z in mz]}), hide_index=True, width="stretch")
    st.caption("Ein Riffle bewegt alle Karten, ein Schritt der beiden anderen Verfahren zwei Karten bzw. eine: die Zahlen verschiedener Spalten sind nicht in einer Einheit.")

st.markdown("---")

# =====================================================================================================================================
# Experiment 5
# =====================================================================================================================================
st.subheader("🔬 Experiment 5: Schätzen aus Läufen (die Falle, zweite und dritte Gestalt)")
st.caption(T.E5_CAPTION)
c1, c2, c3 = st.columns(3)
with c1:
    R5 = st.select_slider("Zahl der Ketten R", options=list(C.SIM_R_LIVE), key="e5_r", help="Live bis 2000 Ketten; größere R stehen in der vorgerechneten Tabelle unten.")
with c2:
    kz = st.radio("Kennzahl", C.KENNZAHLEN, key="e5_kennzahl", horizontal=True, help="Ganze Verteilung der Zustände oder nur die Teilchenzahl (6 Werte).")
with c3:
    reps5 = st.slider("Wiederholungen", *bounds("e5_reps"), key="e5_reps", help="So viele unabhängige Läufe mit je R Ketten (Zufalls-Seed der Seitenleiste).")
with st.spinner("Simulation läuft …"):
    r5 = _exp5(ev.E5Settings(int(R5), kz, int(reps5), seed))
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_simulation(r5), width="stretch", key="e5_chart")
with c2:
    st.plotly_chart(build_schaetzungen(r5), width="stretch", key="e5_streifen")
art, text = ev.verdict_simulation(r5)
getattr(st, art)(text)
st.markdown("**Vorgerechnet: geschätzte Mischzeit gegen R** (Zufalls-Seed 5000, Läufe je R wie angegeben; Rauschgrenze exakt aus den Binomialverteilungen):")
zeilen5 = []
for R in C.SIM_R_TABELLE:
    z = REF.sim_zeile(R)
    ganz, zahl = z["ganz"], z["zahl"]
    zeilen5.append({
        "R": FMT.ganz(R), "Läufe": z["wiederholungen"], "Rauschgrenze (ganze Verteilung)": FMT.dez(z["rauschgrenze"], 4),
        "ganze Verteilung: Mittel [min, max]": ("keine" if ganz["mittel"] is None else f"{FMT.dez(ganz['mittel'], 1)} [{ganz['min']}, {ganz['max']}]"),
        "nie erreicht": f"{ganz['nie']} von {ganz['von']}",
        "Teilchenzahl: Mittel [min, max]": ("keine" if zahl["mittel"] is None else f"{FMT.dez(zahl['mittel'], 1)} [{zahl['min']}, {zahl['max']}]"),
    })
st.dataframe(pd.DataFrame(zeilen5), hide_index=True, width="stretch")
st.caption(f"Exakt (Matrix): 73 Schritte für die ganze Verteilung, 11 für die Teilchenzahl. "
           f"Auch eine perfekt gemischte Stichprobe hat im Mittel nie den Abstand 0: bei R = 14 Ketten auf den 7 Zuständen des Rings mit 4 Plätzen und λ = 1 liegt der erwartete Abstand bei "
           f"{FMT.dez(ev.stichprobe_gleichverteilt(14, 7), 4)}, über der Schwelle 1/4.")

st.markdown("---")

# =====================================================================================================================================
# Einordnung
# =====================================================================================================================================
with st.expander("Wo die exakte Rechnung endet (vorgerechnet)"):
    zeilen6 = ev.einordnung_zeilen()
    st.dataframe(pd.DataFrame({"Raum": [z["raum"] for z in zeilen6], "Zustände": [FMT.ganz(z["zustaende"]) if z["zustaende"] < 10 ** 12 else FMT.potenz(z["zustaende"]) for z in zeilen6],
                               "Mischzeit": [z["mischzeit"] for z in zeilen6]}), hide_index=True, width="stretch")
    st.info("Exakte Mischzeiten gibt es nur für wenige Räume (Diaconis 2009, Abschnitt 2.3); wo sie fehlen, hilft die Kopplung (Nachbar-Demo zu Kopplung von Ketten). "
            "Das ist **keine** Aussage, dass man dort einen Sampler braucht: maßgeblich für das exakte Ziehen ist die Breite des Torus, nicht seine Zustandszahl.")

st.markdown("---")

# =====================================================================================================================================
# Grenzen und Formeln
# =====================================================================================================================================
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.GRENZEN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/mischzeit-exakt-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist eine Markov-Kette mit Übergangsmatrix $P$ auf endlich vielen Zuständen und ihr Gleichgewicht $\pi$. Gesucht ist, wie viele Schritte $t$ nötig sind, bis die Kette von jedem Start aus innerhalb von $\varepsilon$ an $\pi$ liegt, und wie weit Schranken und Schätzungen davon abweichen. Gerechnet wird mit Hard-Core-Ring (Heat-bath und Metropolis), träger Irrfahrt auf dem Kreis, träger Irrfahrt auf dem Hyperwürfel und Riffle-Mischen von Karten.

| Symbol | Bedeutung |
|---|---|
| $n$ | Zahl der Plätze im Ring, Punkte des Kreises, Koordinaten des Hyperwürfels, Karten |
| $x$ | Zustand: Menge der belegten Plätze (Ring); $\lvert x\rvert$ ihre Zahl |
| $\lambda$ | Gewicht je belegtem Platz; $p=\lambda/(1+\lambda)$, $q=1-p$ |
| $\pi$, $\pi_{\min}$ | Gleichgewicht (stationäre Verteilung) und ihr kleinster Eintrag |
| $P$, $P^t(x,\cdot)$ | Übergangsmatrix; Verteilung nach $t$ Schritten vom Start $x$ aus |
| $\mathrm{TV}$ | Abstand zweier Verteilungen: $\tfrac12\sum_z\lvert\mu(z)-\nu(z)\rvert$ |
| $d(t)$, $\bar d(t)$ | Abstand zu $\pi$ vom schlechtesten Start bzw. vom schlechtesten Paar von Starts |
| $t_{\mathrm{mix}}(\varepsilon)$, $\bar t_{\mathrm{mix}}(\varepsilon)$ | erstes $t$ mit $d(t)\le\varepsilon$ bzw. $\bar d(t)\le\varepsilon$; $t_{\mathrm{mix}}=t_{\mathrm{mix}}(1/4)$ |
| $T_{\mathrm{mix}}(1/e)$ | $\bar t_{\mathrm{mix}}(1/e)$, die Mischzeit mit $\bar d$ und der Schwelle $1/e$ (Propp/Wilson) |
| $\gamma^*$, $t_{\mathrm{rel}}$ | absolute Spektrallücke der Matrix, $1-\max(\lvert\lambda_2\rvert,\lvert\lambda_{\min}\rvert)$, und $t_{\mathrm{rel}}=1/\gamma^*$ |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/mischzeit-exakt-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(T.FORMELN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html)."
)
sync_query_params({k: st.session_state[k] for k in ("seed_input", "e1_n", "e1_lam", "e1_kette", "e1_eps", "e1_unit", "e2_n", "e2_lam", "e3_n", "e3_lam", "e3_kette", "e3_kreis", "e3_traege",
                                                    "e3_wuerfel", "e4_kreis", "e4_wuerfel", "e4_ring", "e4_karten", "e5_r", "e5_kennzahl", "e5_reps")})
