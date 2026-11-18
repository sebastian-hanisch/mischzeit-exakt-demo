"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import msz_chain as K
import msz_constants as C

PALETTE = ["#2F6B65", "#c46f2a", "#4c78a8", "#a05195", "#7a8f2e", "#b5483e", "#5b6770", "#d1a000"]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _hlinie(fig, y, text, farbe=None):
    fig.add_hline(y=y, line=dict(color=farbe or C.COLOR_BOUND, width=1, dash="dot"), annotation_text=text, annotation_position="top right")


def build_kurven(res, einheit: str):
    """d(t), d̄(t) und die Kurve des gewählten Starts; Marken bei t_mix (d), t̄_mix (d̄, 1/4) und T_mix (d̄, 1/e)."""
    s = res.s
    faktor = {"Schritte": 1.0, "Durchläufe (n Schritte)": float(s.n), "Abfertigungsdauern": res.takt}[einheit]
    x = np.array(res.ts, dtype=float) / faktor
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=res.d_kurve, mode="lines", name="d(t): schlechtester Start", line=dict(color=C.COLOR_D, width=3),
                             hovertemplate="t = %{x:.3g}: d = %{y:.4f}<extra></extra>"))
    if res.dbar_kurve is not None:
        fig.add_trace(go.Scatter(x=x, y=res.dbar_kurve, mode="lines", name="d̄(t): schlechtestes Paar von Starts", line=dict(color=C.COLOR_DBAR, width=3),
                                 hovertemplate="t = %{x:.3g}: d̄ = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=res.start_kurve, mode="lines", name=f"gewählter Start {res.start_label}", line=dict(color=C.COLOR_START, width=2, dash="dash"),
                             hovertemplate="t = %{x:.3g}: Abstand = %{y:.4f}<extra></extra>"))
    marken = [(res.t_d, C.COLOR_D, f"t_mix = {res.t_d}", 0.25)]
    if res.t_dbar is not None:
        marken += [(res.t_dbar, C.COLOR_DBAR, f"t̄_mix = {res.t_dbar}", 0.25), (res.T_e, C.COLOR_DBAR, f"T_mix(1/e) = {res.T_e}", 1.0 / math.e)]
    fig.add_trace(go.Scatter(x=[m[0] / faktor for m in marken], y=[m[3] for m in marken], mode="markers+text", text=[m[2] for m in marken], textposition="top right",
                             marker=dict(size=11, color=[m[1] for m in marken], symbol="diamond"), name="Marken", showlegend=False, cliponaxis=False,
                             hovertemplate="%{text}<extra></extra>"))
    _hlinie(fig, 0.25, "1/4")
    _hlinie(fig, 1.0 / math.e, "1/e")
    fig.update_xaxes(title_text=f"Zeit t in {einheit}")
    fig.update_yaxes(title_text="Abstand (log. Achse)", type="log", range=[-2.6, 0.1])
    return _base(fig)


def build_start_hist(r):
    fig = go.Figure(go.Histogram(x=list(r.tx), nbinsx=40, marker_color=C.COLOR_START, hovertemplate="t_x %{x}: %{y} Starts<extra></extra>", name="Starts"))
    for wert, text, farbe in [(r.t_mix, f"schlechtester: {r.t_mix}", C.COLOR_D), (r.t_leer, f"leer: {r.t_leer}", C.COLOR_DBAR),
                              (r.t_mittel, f"Mittel: {r.t_mittel:.0f}", "#7a8f2e"), (r.t_mittel_pi, f"mit π gewichtet: {r.t_mittel_pi:.0f}", "#a05195")]:
        fig.add_vline(x=wert, line=dict(color=farbe, width=2), annotation_text=text, annotation_position="top")
    fig.update_xaxes(title_text="Schritte bis zum Abstand 1/4 (t_x je Start x)")
    fig.update_yaxes(title_text="Zahl der Starts")
    return _base(fig)


def build_einschluss(e):
    """Zahlenstrahl (logarithmisch): untere Schranke, exaktes t_mix, t_rel, obere Schranke."""
    punkte = [("untere Schranke", e.unten, "#5b6770"), ("t_rel", e.t_rel, "#a05195"), ("exaktes t_mix", e.t_mix, C.COLOR_D), ("obere Schranke", e.oben, "#b5483e")]
    punkte = [p for p in punkte if p[1] is not None and p[1] > 0 and math.isfinite(p[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p[1] for p in punkte], y=[0] * len(punkte), mode="markers+text", text=[f"{p[0]}<br>{p[1]:,.1f}".replace(",", " ") for p in punkte],
                             textposition=["top center", "bottom center", "top center", "bottom center"][: len(punkte)], marker=dict(size=14, color=[p[2] for p in punkte]),
                             hovertemplate="%{text}<extra></extra>", cliponaxis=False))
    if len(punkte) >= 2:
        fig.add_shape(type="line", x0=min(p[1] for p in punkte), x1=max(p[1] for p in punkte), y0=0, y1=0, line=dict(color=C.COLOR_BOUND, width=2))
    fig.update_xaxes(title_text="Schritte (logarithmische Achse)", type="log")
    fig.update_yaxes(visible=False, range=[-1, 1])
    return _base(fig, 230)


def build_wachstum_kurven(kurven: dict, titel_x="t / t_mix"):
    fig = go.Figure()
    for i, (n, (x, d)) in enumerate(sorted(kurven.items())):
        fig.add_trace(go.Scatter(x=x, y=d, mode="lines", name=f"n = {n}", line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                                 hovertemplate=f"n = {n}: t/t_mix = %{{x:.2f}}, d = %{{y:.4f}}<extra></extra>"))
    _hlinie(fig, 0.25, "1/4")
    fig.update_xaxes(title_text=titel_x)
    fig.update_yaxes(title_text="Abstand d", range=[0, 1.02])
    return _base(fig)


def build_tmix_ueber_n(zeilen, formel=None, formel_name="", titel_y="t_mix (Schritte)"):
    xs = [z.n for z in zeilen]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[z.t_mix for z in zeilen], mode="lines+markers", name="exakt", line=dict(color=C.COLOR_D, width=3),
                             hovertemplate="n = %{x}: t_mix = %{y}<extra></extra>"))
    if formel is not None:
        fig.add_trace(go.Scatter(x=xs, y=[formel(n) for n in xs], mode="lines", name=formel_name, line=dict(color=C.COLOR_DBAR, width=2, dash="dash"),
                                 hovertemplate="n = %{x}: %{y:.0f}<extra></extra>"))
    fig.update_xaxes(title_text="n", type="log")
    fig.update_yaxes(title_text=titel_y, type="log")
    return _base(fig, 320)


def build_karten(n, tabelle):
    ks = [k for k, _ in tabelle]
    ds = [max(d, 1e-6) for _, d in tabelle]
    fig = go.Figure(go.Scatter(x=ks, y=ds, mode="lines+markers", name=f"{n} Karten", line=dict(color=C.COLOR_D, width=3),
                               hovertemplate="%{x} Mischungen: %{y:.4f}<extra></extra>"))
    _hlinie(fig, 0.25, "1/4")
    _hlinie(fig, 0.01, "0,01")
    fig.update_xaxes(title_text="Zahl der Riffle-Mischungen")
    fig.update_yaxes(title_text="Abstand zur Gleichverteilung (log. Achse)", type="log", range=[-3, 0.1])
    return _base(fig)


def build_simulation(res, einheit_t_max=C.SIM_T_MAX):
    t = list(range(len(res.kurve_emp)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=res.kurve_exakt, mode="lines", name="exakt (Matrix)", line=dict(color=C.COLOR_D, width=3), hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=res.kurve_emp, mode="lines", name=f"empirisch, {res.s.R} Ketten (erster Lauf)", line=dict(color=C.COLOR_DBAR, width=2),
                             hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    rg = res.rausch_zahl if res.s.kennzahl == "Teilchenzahl" else res.rausch
    _hlinie(fig, 0.25, "Schwelle 1/4")
    fig.add_hline(y=rg, line=dict(color="#a05195", width=2, dash="dash"), annotation_text=f"Rauschgrenze {rg:.3f}", annotation_position="bottom right")
    fig.update_xaxes(title_text="Schritte t")
    fig.update_yaxes(title_text="Abstand zu π (log. Achse)", type="log", range=[-2.5, 0.1])
    return _base(fig)


def build_schaetzungen(res):
    zahl = res.s.kennzahl == "Teilchenzahl"
    exakt = res.t_exakt_zahl if zahl else res.t_exakt
    erreicht = [(i + 1, t) for i, t in enumerate(res.schaetzungen) if t is not None]
    nie = [i + 1 for i, t in enumerate(res.schaetzungen) if t is None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i, _ in erreicht], y=[t for _, t in erreicht], mode="markers", name="geschätzte Mischzeit", marker=dict(size=10, color=C.COLOR_DBAR),
                             hovertemplate="Lauf %{x}: %{y} Schritte<extra></extra>"))
    if nie:
        fig.add_trace(go.Scatter(x=nie, y=[C.SIM_T_MAX + 8] * len(nie), mode="markers", name=f"nie erreicht (bis t = {C.SIM_T_MAX})", marker=dict(size=10, symbol="x", color="#b5483e"),
                                 hovertemplate="Lauf %{x}: nie erreicht<extra></extra>"))
    fig.add_hline(y=exakt, line=dict(color=C.COLOR_D, width=2), annotation_text=f"exakt: {exakt}", annotation_position="top left")
    fig.update_xaxes(title_text="Lauf", dtick=1)
    fig.update_yaxes(title_text="Schritte", range=[0, C.SIM_T_MAX + 20])
    return _base(fig, 300)
