"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

# --- Ring (Experimente 1 bis 3, 5) -----------------------------------------------------------------------------------------------------
RING_N_MIN, RING_N_MAX_LIVE = 3, 12        # bis 12 Plätze (322 Zustände) wird live gerechnet; darüber nur die vorgerechnete Tabelle
RING_N_MAX_TABLE = 16                      # die vorgerechnete Tabelle (msz_reference.json) geht bis 16 Plätze
LAM_MIN, LAM_MAX, LAM_STEP = 0.5, 20.0, 0.5
LAM_TABELLE = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)   # Spalten der vorgerechneten Tabelle
DEFAULT_N, DEFAULT_LAM = 10, 2.0
START_SCHLECHTESTER = "schlechtester Start (Maximalbelegung)"
START_LEER = "leerer Ring"
KETTEN = ("Heat-bath", "Metropolis")
EINHEITEN = ("Schritte", "Durchläufe (n Schritte)", "Abfertigungsdauern")
SCHWELLEN = {"1/2": 0.5, "1/4": 0.25, "1/e": 1.0 / math.e, "0,1": 0.1, "0,01": 0.01}
SCHWELLE_STANDARD = "1/4"
E1_DBAR_N_MAX_LIVE = 10                    # d̄ (alle Paare von Starts) wird bis hierher live gerechnet (darüber zu langsam)
T_RECHENGRENZE = 2_000_000                 # größtes t, bis zu dem t_mix gesucht wird; darüber „nicht erreicht“

# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------
E2_N_MIN, E2_N_MAX = 4, 10
E2_LAMS = (1.0, 2.0, 5.0, 10.0)            # die vier Zeilen der Tabelle

# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------
KREIS_N_MIN, KREIS_N_MAX, KREIS_DEFAULT = 3, 800, 25
WUERFEL_N_MIN, WUERFEL_N_MAX, WUERFEL_DEFAULT = 2, 1024, 400

# --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------
KREIS_N_AUSWAHL = (25, 50, 100, 200, 400, 800)
KREIS_N_STANDARD = (25, 100, 400)
WUERFEL_N_AUSWAHL = (12, 16, 32, 64, 128, 256, 512, 1024)
WUERFEL_N_STANDARD = (16, 128, 1024)
RING_FENSTER_N = (8, 12, 16, 20, 24)       # Hard-Core-Ring, λ = 1: 8 und 12 live, die übrigen vorgerechnet
RING_FENSTER_N_STANDARD = (8, 12, 16, 24)
KARTEN_N_MIN, KARTEN_N_MAX, KARTEN_DEFAULT = 2, 128, 32
KARTEN_N_TABELLE = (4, 8, 16, 32, 52, 64, 128)
KARTEN_K_MAX = 14                          # so viele Mischungen werden angezeigt
MISCHER_N = tuple(range(3, 10))            # Vergleich der Einheiten bei Karten (vorgerechnet)
MISCHER_N_LIVE_MAX = 6

# --- Experiment 5 -----------------------------------------------------------------------------------------------------------------------
SIM_N, SIM_LAM = 10, 2.0                   # der Ring dieses Experiments (feste Instanz)
SIM_R_LIVE = (30, 100, 300, 1000, 2000)
SIM_R_DEFAULT = 1000
SIM_R_TABELLE = (100, 300, 1000, 3000, 10000, 100000)
SIM_REPS_MIN, SIM_REPS_MAX, SIM_REPS_DEFAULT = 3, 12, 10
SIM_T_MAX = 160
SIM_SEED_DEFAULT, SIM_SEED_MAX = 5000, 999999
SIM_RAUSCH_WDH = 30                        # Wiederholungen für die Rauschgrenze
KENNZAHLEN = ("ganze Verteilung", "Teilchenzahl")
TABELLE_SEED = 5000                        # Zufalls-Seed der vorgerechneten Simulationstabelle
TABELLE_SEED_2 = 6000                      # zweiter Zufalls-Seed (Breite der Bänder)
TABELLE_REPS = {100: 20, 300: 20, 1000: 20, 3000: 10, 10000: 10, 100000: 3}

# --- Farben -----------------------------------------------------------------------------------------------------------------------------
COLOR_D = "#2F6B65"
COLOR_DBAR = "#c46f2a"
COLOR_START = "#4c78a8"
COLOR_BOUND = "rgba(150,150,150,0.9)"
COLOR_BAND = "rgba(76,120,168,0.18)"

PRESETS = {
    "Von Hand: 4 Plätze (12 / 16 / 21)": {
        "e1_n": 4, "e1_lam": 1.0, "e1_kette": "Heat-bath", "e1_start": START_SCHLECHTESTER, "e1_eps": "1/4", "e1_unit": "Schritte"},
    "10 Plätze, λ = 2: 73 Schritte": {
        "e1_n": 10, "e1_lam": 2.0, "e1_kette": "Heat-bath", "e1_start": START_SCHLECHTESTER, "e1_eps": "1/4", "e1_unit": "Schritte"},
    "Der leere Start täuscht (λ = 10)": {"e2_n": 10, "e2_lam": 10.0},
    "Die Schranke liegt 26-fach daneben (n = 12, λ = 20)": {"e3_n": 12, "e3_lam": 20.0, "e3_kette": "Heat-bath"},
    "Zu wenige Ketten (R = 300)": {"e5_r": 300, "e5_kennzahl": "ganze Verteilung"},
    "Kreis n = 100": {"e4_kreis": (25, 100, 400, 800), "e3_kreis": 100},
    "Hyperwürfel n = 1024: Cutoff": {"e4_wuerfel": (16, 128, 1024), "e3_wuerfel": 1024},
    "52 Karten: 7 Mischungen": {"e4_karten": 52},
}
STANDARD_PRESET = "10 Plätze, λ = 2: 73 Schritte"
PRESET_HELP = {
    "Von Hand: 4 Plätze (12 / 16 / 21)": "Ring mit 4 Plätzen, λ = 1, 7 Zustände: Experiment 1 mit den Zahlen aus der Handrechnung (12 mit d, 21 mit d̄, 16 mit d̄ und Schwelle 1/e).",
    "10 Plätze, λ = 2: 73 Schritte": "Ring mit 10 Plätzen und 123 Zuständen, Heat-bath: Experiment 1 mit 73 Schritten (d) gegen 118 (d̄) und 91 (d̄, 1/e).",
    "Der leere Start täuscht (λ = 10)": "Experiment 2: bei 10 Plätzen und λ = 10 braucht die Maximalbelegung 569 Schritte, der leere Ring 48.",
    "Die Schranke liegt 26-fach daneben (n = 12, λ = 20)": "Experiment 3: Ring mit 12 Plätzen, λ = 20, Heat-bath; die obere Schranke aus der Spektrallücke liegt dort beim 26,3-Fachen des exakten Werts.",
    "Zu wenige Ketten (R = 300)": "Experiment 5: bei 300 Ketten liegt die Rauschgrenze nahe der Schwelle, die geschätzte Mischzeit liegt weit über 73.",
    "Kreis n = 100": "Experiment 3 und 4: träge Irrfahrt auf dem Kreis, t_mix/n² etwa 0,095, kein Cutoff.",
    "Hyperwürfel n = 1024: Cutoff": "Experiment 3 und 4: träge Irrfahrt auf dem Hyperwürfel, das Übergangsfenster wird mit n schmaler (Cutoff).",
    "52 Karten: 7 Mischungen": "Experiment 4: 52 Karten, Riffle-Mischen: nach 7 Mischungen liegt der Abstand noch bei 0,3341.",
}
