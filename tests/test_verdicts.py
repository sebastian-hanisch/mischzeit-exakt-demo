"""Verdikte an von Hand gesetzten Werten (reine Funktionen, DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Dazu je Preset die erwartete Verdikt-Zeile mit den Zahlen der laufenden Rechnung (ein Preset meldet nur, was es zeigt)."""

import msz_constants as C
import msz_evaluation as ev
import msz_presets as P
import msz_format as FMT


# ---------------------------------------------------------------------------------------------------- Experiment 1
def test_drei_zahlen_mit_d_bar():
    art, text = ev.verdict_drei_zahlen(4, 1.0, "Heat-bath", 7, 12, 0.2631, 0.2435, 21, 16, 7, 8.0)
    assert art == "info"
    assert "Ring n = 4, λ = 1 (7 Zustände), Heat-bath" in text
    assert "d(11) = 0,2631 liegt über 1/4, d(12) = 0,2435 darunter: **12 Schritte**" in text
    assert "erst bei **21** erreicht" in text and "Schwelle 1/e (Propp/Wilson) ist **16**" in text and "wären es 7" in text
    assert "(8 Schritte je Abfertigungsdauer): 1,500." in text


def test_drei_zahlen_ohne_d_bar_nennt_die_grenze():
    art, text = ev.verdict_drei_zahlen(12, 2.0, "Metropolis", 322, 62, 0.2511, 0.2490, None, None, 41, 24.0)
    assert art == "info" and "bis n = 10 live" in text and "**62 Schritte**" in text and "schon bei 41 Schritten" in text
    assert "erst bei" not in text and "Metropolis" in text


# ---------------------------------------------------------------------------------------------------- Experiment 2
def _e2(**kw):
    basis = dict(n=10, lam=10.0, N=123, t_mix=569, t_leer=48, t_min=48, t_mittel=303.3, t_mittel_pi=442.7, anteil_halb=0.66, schlechteste_maximal=True,
                 alle_maximal_gleich=True, zahl_schlechteste=2, tx=(), teilchen=())
    basis.update(kw)
    return ev.E2Result(**basis)


def test_verdikt_start_nennt_die_zahlen_und_den_faktor():
    art, text = ev.verdict_start(_e2())
    assert art == "info"
    assert "n = 10, λ = 10, 123 Zustände" in text and "**569 Schritte**" in text and "der leere Ring 48" in text
    assert "im Mittel über alle 123 Starts 303 und mit π gewichtet 443" in text
    assert "um das 11,9-fache" in text                                  # 569/48 = 11,85
    assert "eine Maximalbelegung" in text and "alle Maximalbelegungen sind gleich schlecht" in text


def test_verdikt_start_zweige_keine_maximalbelegung_und_nicht_gleich():
    _, text = ev.verdict_start(_e2(schlechteste_maximal=False, alle_maximal_gleich=False))
    assert "keine Maximalbelegung" in text and "nicht alle Maximalbelegungen sind gleich schlecht" in text


def test_verdikt_start_leerer_start_ist_schon_gemischt():
    _, text = ev.verdict_start(_e2(t_leer=0))
    assert "um das ∞-fache" in text                                     # t_leer = 0: keine Division durch null


# ---------------------------------------------------------------------------------------------------- Experiment 3
def _einschluss(**kw):
    basis = dict(raum="Ring n = 10, λ = 2, Heat-bath", n=10, t_mix=73, t_rel=68.4233, unten=46.7343, oben=569.1957, pi_min=0.001, quelle="live")
    basis.update(kw)
    return ev.Einschluss(**basis)


def test_einschluss_verhaeltnisse():
    e = _einschluss()
    assert abs(e.oben_durch_exakt - 569.1957 / 73) < 1e-12 and abs(e.exakt_durch_unten - 73 / 46.7343) < 1e-12
    assert _einschluss(t_mix=None).oben_durch_exakt is None and _einschluss(t_mix=None).exakt_durch_unten is None


def test_verdikt_einschluss_ring_zahlen_und_art():
    stat = {"zellen": 30, "oben_min": 4.2, "oben_max": 29.1, "oben_median": 9.7, "unten_min": 1.01, "unten_max": 2.4, "unten_median": 1.4}
    verh = {"zellen": 48, "min": 0.69, "max": 1.65}
    art, text = ev.verdict_einschluss_ring(_einschluss(), stat, _einschluss(raum="Kreis n = 100 (träge)", n=100, t_mix=949, oben=6072.6),
                                           _einschluss(raum="Hyperwürfel n = 400 (träge)", n=400, t_mix=1378, oben=111458.1), verh)
    assert art == "warning"
    assert "t_rel = 68,4; Einschluss 46,7 ≤ **73** ≤ 569,2" in text and "beim 7,8-fachen des exakten Werts" in text and "beim 1,56-fachen der unteren" in text
    assert "30 vorgerechneten Zellen" in text and "4,2- bis 29,1-fachen (Median 9,7)" in text
    assert "beim Kreis n = 100 beim 6,4-fachen" in text and "beim Hyperwürfel n = 400 beim 80,9-fachen" in text
    assert "0,69- bis 1,65-fachen von t_rel" in text and text.endswith("Ein Einschluss, keine Vorhersage.")


def test_verdikt_einschluss_raum_periodisch_und_normal():
    art, text = ev.verdict_einschluss_raum(_einschluss(raum="Kreis n = 4 (nicht träge)", n=4, t_mix=None, t_rel=float("inf"), unten=float("inf"), oben=float("inf")))
    assert art == "warning" and "periodisch" in text and "unendlich" in text
    art, text = ev.verdict_einschluss_raum(_einschluss(raum="Kreis n = 100 (träge)", n=100, t_mix=949, t_rel=1013.5, unten=701.8, oben=6072.6))
    assert art == "info" and "Einschluss 701,8 ≤ **949** ≤ 6072,6" in text and "beim 6,4-fachen" in text and "1,35-fachen der unteren" in text


# ---------------------------------------------------------------------------------------------------- Experiment 4
def _z(n, t9, t5, t4, t1):
    return ev.WachstumZeile(n, {0.9: t9, 0.5: t5, 0.25: t4, 0.1: t1})


def test_wachstumszeile_fenster_und_t_mix():
    z = _z(100, 100, 600, 949, 1500)
    assert z.t_mix == 949 and abs(z.fenster - 1400 / 600) < 1e-12


def test_verdikt_kreis():
    zeilen = [_z(25, 10, 100, 59, 150), _z(100, 10, 100, 949, 150)]
    art, text = ev.verdict_kreis(zeilen)
    assert art == "info" and "n = 25, 100" in text and "Schranke n²/32 ≤ t_mix ≤ n² (Levin/Peres/Wilmer, Abschnitt 5.3.2) in 2 von 2 Fällen" in text
    assert "t_mix/n² liegt bei 0,0944 bis 0,0949" in text and "Für n ≥ 100 (100) liegt t_mix/n² bei 0,0949" in text
    assert "kein Cutoff" in text                                           # Fenster beider Zeilen gleich 1,40
    _, einzeln = ev.verdict_kreis([_z(100, 10, 100, 949, 150)])
    assert "kein Cutoff" not in einzeln                                    # mit einem n lässt sich nichts über das Fallen sagen


def test_verdikt_wuerfel_fenster_faellt_oder_nicht():
    schmal = [_z(16, 7, 18, 29, 43), _z(1024, 2337, 3243, 4009, 4961)]
    art, text = ev.verdict_wuerfel(schmal)
    assert art == "info" and "von 2,00 (n = 16) auf 0,81 (n = 1024)" in text and "wird mit n schmaler" in text and "Satz 18.3" in text
    _, text2 = ev.verdict_wuerfel([_z(16, 7, 18, 29, 43), _z(32, 7, 18, 29, 43)])
    assert "kein Fallen zu sehen" in text2


def test_verdikt_ring_fenster_nennt_messung_nicht_beleg():
    art, text = ev.verdict_ring_fenster([_z(8, 1, 14, 31, 55), _z(24, 22, 77, 133, 208)])
    assert art == "info" and "3,86, 2,42" in text and "kein Beleg für einen Cutoff" in text and "Es fällt mit n wie beim Hyperwürfel" in text
    _, text2 = ev.verdict_ring_fenster([_z(8, 1, 14, 31, 55)])
    assert "Es fällt" not in text2 and "kein Beleg für einen Cutoff" in text2        # mit einem n lässt sich nichts über das Fallen sagen


def test_verdikt_karten_52_und_einzelne_karte():
    art, text = ev.verdict_karten(52)
    assert art == "info" and "nach 7 Mischungen bei 0,3341, nach 8 bei 0,1672" in text and "t_mix(1/4) = **8 Mischungen**" in text
    assert "erst nach 13 Mischungen (12 geben 0,0108, 13 geben 0,0054)" in text
    assert "Eine Karte ist immer gemischt" in ev.verdict_karten(1)[1]


# ---------------------------------------------------------------------------------------------------- Experiment 5
def _e5(R, kennzahl, zusammen, schaetz, rausch=0.243, rausch_zahl=0.04):
    return ev.E5Result(ev.E5Settings(R, kennzahl, len(schaetz), 5000), 73, 11, rausch, rausch_zahl, 0.24, tuple(schaetz), zusammen, (), ())


def test_verdikt_simulation_nie_erreicht_mit_rauschgrenze_ueber_der_schwelle():
    zus = {"mittel": None, "min": None, "max": None, "nie": 20, "von": 20}
    art, text = ev.verdict_simulation(_e5(100, "ganze Verteilung", zus, [None] * 20, rausch=0.428))
    assert art == "warning" and "R = 100 Ketten, 20 Läufe (Zufalls-Seed 5000)" in text and "exakt 73 Schritte; Rauschgrenze 0,428" in text
    assert "Keiner der 20 Läufe erreicht die Schwelle 1/4 bis t = 160: die Rauschgrenze liegt über 1/4" in text


def test_verdikt_simulation_zu_hoch_und_gut():
    zus = {"mittel": 130.4, "min": 105, "max": 153, "nie": 0, "von": 10}
    art, text = ev.verdict_simulation(_e5(300, "ganze Verteilung", zus, [105] * 10))
    assert art == "warning" and "im Mittel bei 130,4 (kleinster Wert 105, größter 153), 0 von 10 Läufen erreichen die Schwelle nie" in text
    zus = {"mittel": 73.3, "min": 73, "max": 74, "nie": 0, "von": 3}
    art, _ = ev.verdict_simulation(_e5(100000, "ganze Verteilung", zus, [73, 74, 73], rausch=0.013))
    assert art == "success"                                             # innerhalb von 10 % des exakten Werts


def test_verdikt_simulation_kennzahl_teilchenzahl():
    zus = {"mittel": 11.2, "min": 10, "max": 12, "nie": 0, "von": 10}
    art, text = ev.verdict_simulation(_e5(300, "Teilchenzahl", zus, [11] * 10, rausch_zahl=0.0399))
    assert "Kennzahl Teilchenzahl: exakt 11 Schritte; Rauschgrenze 0,040" in text and "exakt 11 gegen 73 Schritte" in text
    assert art == "success"


def test_stichprobe_gleichverteilt():
    assert abs(ev.stichprobe_gleichverteilt(14, 7) - 0.25034) < 5e-5 and abs(ev.stichprobe_gleichverteilt(7, 7) - 0.3399) < 5e-5


# ---------------------------------------------------------------------------------------------------- je Preset die erwartete Zeile
def test_preset_von_hand_meldet_12_21_16():
    p = C.PRESETS["Von Hand: 4 Plätze (12 / 16 / 21)"]
    r = ev.exp1(ev.E1Settings(p["e1_n"], p["e1_lam"], p["e1_kette"], p["e1_start"], p["e1_eps"], p["e1_unit"]))
    _, text = ev.verdict_drei_zahlen(4, 1.0, "Heat-bath", r.N, r.t_d, r.d_vor, r.d_bei, r.t_dbar, r.T_e, r.t_d_e, r.takt)
    assert "d(11) = 0,2631" in text and "d(12) = 0,2435" in text and "**12 Schritte**" in text and "**21**" in text and "ist **16**" in text and "wären es 7" in text


def test_preset_10_plaetze_meldet_73_118_91():
    p = C.PRESETS["10 Plätze, λ = 2: 73 Schritte"]
    r = ev.exp1(ev.E1Settings(p["e1_n"], p["e1_lam"], p["e1_kette"], p["e1_start"], p["e1_eps"], p["e1_unit"]))
    assert (r.t_d, r.t_dbar, r.T_e, r.t_d_e) == (73, 118, 91, 48)
    _, text = ev.verdict_drei_zahlen(10, 2.0, "Heat-bath", r.N, r.t_d, r.d_vor, r.d_bei, r.t_dbar, r.T_e, r.t_d_e, r.takt)
    assert "**73 Schritte**" in text and "**118**" in text and "**91**" in text and "2,433" in text


def test_preset_leerer_start_meldet_569_und_48():
    p = C.PRESETS["Der leere Start täuscht (λ = 10)"]
    r = ev.exp2(p["e2_n"], p["e2_lam"])
    _, text = ev.verdict_start(r)
    assert "**569 Schritte**" in text and "der leere Ring 48" in text and "um das 11,9-fache" in text and "443" in text


def test_preset_schranke_meldet_26_fach():
    p = C.PRESETS["Die Schranke liegt 26-fach daneben (n = 12, λ = 20)"]
    e = ev.einschluss_ring(p["e3_n"], p["e3_lam"], p["e3_kette"])
    assert e.t_mix == 2276 and abs(e.oben_durch_exakt - 26.3) < 0.05


def test_preset_zu_wenige_ketten_meldet_warnung():
    p = C.PRESETS["Zu wenige Ketten (R = 300)"]
    r = ev.exp5(ev.E5Settings(p["e5_r"], p["e5_kennzahl"], C.SIM_REPS_DEFAULT, C.SIM_SEED_DEFAULT))
    art, text = ev.verdict_simulation(r)
    assert art == "warning" and "Rauschgrenze 0,243" in text and "R = 300 Ketten" in text


def test_preset_kreis_und_wuerfel_und_karten():
    zk = ev.kreis_zeilen(C.PRESETS["Kreis n = 100"]["e4_kreis"])
    _, text = ev.verdict_kreis(zk)
    assert "0,0944 bis 0,0949" in text and "kein Cutoff" in text
    zw = ev.wuerfel_zeilen(C.PRESETS["Hyperwürfel n = 1024: Cutoff"]["e4_wuerfel"])
    _, text = ev.verdict_wuerfel(zw)
    assert "von 2,00 (n = 16) auf 0,81 (n = 1024)" in text
    _, text = ev.verdict_karten(C.PRESETS["52 Karten: 7 Mischungen"]["e4_karten"])
    assert "nach 7 Mischungen bei 0,3341" in text


def test_jedes_preset_besteht_aus_bekannten_reglern_und_erlaubten_werten():
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.kind in ("int", "float"):
                assert spec.lo <= wert <= spec.hi, (name, key)
            elif spec.kind == "wahl":
                assert wert in spec.erlaubt, (name, key)
            elif spec.kind == "liste":
                assert set(wert) <= set(spec.erlaubt), (name, key)


def test_format_hilfen():
    assert FMT.dez(60718.3, 1) == "60 718,3" and FMT.dez(0.5, 3) == "0,500" and FMT.dez(float("inf")) == "∞" and FMT.dez(None) == "–"
    assert FMT.zahl(2.0) == "2" and FMT.zahl(0.5) == "0,5" and FMT.ganz(103682) == "103 682" and FMT.ganz(4009) == "4009"
    assert FMT.potenz(213256442503) == "2,1·10¹¹"
