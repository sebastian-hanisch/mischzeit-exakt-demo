# Wie schnell mischt eine Kette? Mischzeit exakt gerechnet – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-mischzeit-exakt-demo.streamlit.app/)** -->

Eine von mehreren Demos zum **Abtasten von Verteilungen (MCMC)** (Bereich Simulation) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie erklärt die Zahl, die die Nachbar-Demos zu
Gibbs und Glauber und zu Metropolis auf einem kleinen Raum nur **benutzen**, um Ketten zu vergleichen: die **Mischzeit** t_mix(1/4). Gefragt wird, was die Zahl misst, wie sie entsteht und wie sicher man sie hat.
Nachbarn dieser Themen: [`exakt-oder-abtasten-demo`](https://github.com/sebastian-hanisch/exakt-oder-abtasten-demo), [`kette-gleichgewicht-demo`](https://github.com/sebastian-hanisch/kette-gleichgewicht-demo),
[`metropolis-kleiner-raum-demo`](https://github.com/sebastian-hanisch/metropolis-kleiner-raum-demo), [`gibbs-glauber-demo`](https://github.com/sebastian-hanisch/gibbs-glauber-demo) (dort der Takt von Heat-bath und Metropolis),
[`kopplung-ketten-demo`](https://github.com/sebastian-hanisch/kopplung-ketten-demo) (die Mischzeit **schätzen**, wo man sie nicht rechnen kann), [`kopplung-vergangenheit-demo`](https://github.com/sebastian-hanisch/kopplung-vergangenheit-demo),
[`diagnose-taeuscht-demo`](https://github.com/sebastian-hanisch/diagnose-taeuscht-demo), [`gipfel-tempering-demo`](https://github.com/sebastian-hanisch/gipfel-tempering-demo), [`dimension-hmc-demo`](https://github.com/sebastian-hanisch/dimension-hmc-demo),
[`sampler-verfahrenswahl-demo`](https://github.com/sebastian-hanisch/sampler-verfahrenswahl-demo).

**Die Räume sind ausgedachte Räume mit exakt bekannter Antwort, keine reale Datenquelle:** ein Hard-Core-Ring (Stellplätze im Ring, keine zwei benachbarten belegt, Gewicht λ hoch Zahl der belegten Plätze; Kulisse: Andockstellen im Ring mit
Abweisung, wenn Stelle oder Nachbarstelle belegt ist, λ = Angebot je Stelle in Erlang), dazu drei Standardbeispiele ohne Kulisse (träge Irrfahrt auf dem Kreis, träge Irrfahrt auf dem Hyperwürfel, Riffle-Mischen von Karten). In den gelesenen Quellen gibt es für diese Frage
keine reale Datenquelle; auch eine Messung in der gelesenen Literatur, dass MCMC im Betrieb bessere Entscheidungen liefert, gibt es nicht. Gezählt werden **Schritte** einer Kette (eine Platzaktualisierung bzw. ein Vorschlag), **Mischungen** beim Kartenmischen und **Ketten** in der Simulation;
die Mischzeit hängt an Abstand, Schwelle und Startzustand, und jede Aussage gilt nur für die Kette, die hier gerechnet wird.

## Warum dieses Problem

Wer eine Kette laufen lässt, fragt: **Wie viele Schritte, bis sie von jedem Start aus innerhalb von 1/4 an ihrem Gleichgewicht liegt?** Antwort in diesem Modell, Zähleinheit Schritte: im Hard-Core-Ring mit 4 Plätzen und λ = 1 (7 Zustände, Heat-bath) sind es
**12 Schritte** mit dem Abstand d (größter Abstand eines Starts zum Gleichgewicht), **21** mit d̄ (größter Abstand zweier Starts) und derselben Schwelle 1/4 und **16** mit d̄ und der Schwelle 1/e. Das sind drei Zahlen derselben Kette; wer „die Mischzeit“ aus zwei
Quellen vergleicht, vergleicht möglicherweise verschiedene Größen (Abstand, Schwelle und Start können abweichen; Levin/Peres/Wilmer meinen mit d und der Schwelle 1/4 dieselbe Größe wie hier). Mit 10 Plätzen und λ = 10 braucht der schlechteste Start (eine Maximalbelegung) 569 Schritte, der leere Ring 48.
Danach kommen die drei Zahlen, die wie die Mischzeit aussehen und es nicht sind: die **Schranke aus der Spektrallücke** (im Ring mit n = 6 bis 14 Plätzen, λ = 0,5 bis 20, Heat-bath liegt ihre obere Schranke beim 4,2- bis 29,1-fachen des exakten Werts, in 30 Zellen; beim Kreis mit
n = 100 beim 6,40-fachen, beim Hyperwürfel mit n = 400 beim 80,9-fachen), die **Schätzung aus zu wenigen Ketten** (Ring n = 10, λ = 2, exakt 73 Schritte: bei 100 Ketten erreicht keiner von 20 Läufen die Schwelle, in beiden Zufalls-Seeds; bei 300 Ketten liegt die Schätzung im Mittel bei 129,0 bzw. 131,45
Schritten, bei 3000 bei 74,6 bzw. 77,0, bei 100 000 bei 73,0) und die **Kennzahl statt der Verteilung** (die Teilchenzahl liefert bei jedem R im Mittel 11,0 bis 12,6 Schritte statt 73 (exakt 11)). Danach wächst die Zahl je nach Raum verschieden: beim Kreis ist t_mix/n² etwa 0,0949 ab n = 100 ohne Cutoff,
beim Hyperwürfel ist (t_mix − ½ n ln n)/n etwa 0,449 ab n = 400 bei fallendem Fenster (Fenster = (t(0,1) − t(0,9))/t(0,5) mit t(x) = erstes t mit d(t) ≤ x, also die Zeit, in der d von 0,9 auf 0,1 fällt, geteilt durch die Zeit bis 0,5; im Einklang mit dem Cutoff-Satz), und bei 52 Karten geben 7 Riffle-Mischungen noch den Abstand 0,3341, 8 geben 0,1672 und 13 sind für höchstens 0,01 nötig.
Alles gilt in diesem Modell: Heat-bath bzw. Metropolis mit Einzelplatz, Hard-Core-Ring bis 12 Plätze live (alle Starts), bis 14 Plätze vorgerechnet (alle Starts) und bis 16 bzw. 24 Plätze aus der Maximalbelegung; Zähleinheit Schritte, Mischungen und Ketten.
**Lehrbuchwissen** sind der Abstand, die Mischzeit, Satz 12.4, die Eigenwerte von Kreis und Hyperwürfel und die Formel für das Riffle-Mischen (Levin/Peres/Wilmer, siehe Quellen); die Zahlen im Hard-Core-Ring, die Formel für gerades n und die Simulationen sind eigene Rechnungen in diesem Modell.
Nichts davon ist inhaltlich neu; neu ist nur die Vorführbarkeit mit Zähleinheit und exakten Orakeln.

## Modell

* **Hard-Core-Ring:** n Plätze im Ring, Zustand x = Menge belegter Plätze, keine zwei Nachbarn belegt; die Zahl der Zustände ist die Lucas-Zahl (n = 4: 7, n = 10: 123, n = 12: 322, n = 16: 2207, n = 24: 103 682). Gleichgewicht π(x) ∝ λ^|x|.
  **Heat-bath** (Gibbs/Glauber): Platz gleichverteilt ziehen, mit λ/(1+λ) belegen, wenn beide Nachbarn frei sind, sonst leer lassen. **Metropolis (Einzelplatz):** Belegung des gezogenen Platzes umkehren, annehmen mit min(1, π(neu)/π(alt)).
  Beide Ketten sind reversibel (Detailed Balance), irreduzibel und aperiodisch (für n = 3 bis 8 und λ = 0,5 bis 20 geprüft: eine Potenz der Matrix hat nur positive Einträge). Beide Matrizen sind aus der Definition neu aufgezählt und gegen die Matrix der App geprüft (Orakel a).
* **Zeiteinheiten:** Schritt; Durchlauf (n Schritte); Abfertigungsdauer (im Verlustnetz der Andockstellen: Heat-bath n(1+λ) Schritte, Metropolis n·max(1, λ) Schritte; der Beweis dieses Takts steht in der Nachbar-Demo `gibbs-glauber-demo`, hier ist es eine Umrechnung); Mischungen (Karten); Ketten (Simulation).
* **Abstand und Mischzeit:** d(t) = max_x TV(P^t(x, ·), π) mit TV = ½ Σ |μ − ν|; d̄(t) = max_{x,y} TV(P^t(x, ·), P^t(y, ·)). **t_mix(ε)** = kleinstes t mit d(t) ≤ ε, **t_mix = t_mix(1/4)** (Levin/Peres/Wilmer, (4.30) bis (4.31)); **t̄_mix(ε)** dasselbe mit d̄;
  **T_mix(1/e)** = t̄_mix(1/e) (Propp/Wilson). Es gilt **d ≤ d̄ ≤ 2d** (eigener Beweis im 📐-Expander: π ist das Mittel der Zeilen von P^t, TV ist konvex, dazu die Dreiecksungleichung), also t_mix(ε) ≤ t̄_mix(ε) ≤ t_mix(ε/2).
  Umrechnung der Schwelle: t_mix(ε) ≤ ⌈log₂(1/ε)⌉ · t_mix(1/4) ((4.34)).
* **Relaxationszeit und Einschluss:** t_rel = 1/γ* mit der **absoluten Spektrallücke der Matrix** γ* = 1 − max(|λ₂|, |λ_min|); Satz 12.4: (t_rel − 1) ln(1/(2ε)) ≤ t_mix(ε) ≤ t_rel ln(1/(ε π_min)). Das ist ein **Einschluss, keine Vorhersage**.
* **Gerades n, Heat-bath (eigene Rechnung, keine Quelle):** der von f_A = Σ(−1)^v x_v und f_B = Σ(−1)^v x_v x_{v+2} aufgespannte Raum ist invariant (Beweis im 📐-Expander, in Brüchen für n = 4, 6, 8 geprüft); die 2×2-Matrix M aus n·E[Δf] = M·f hat die Eigenwerte −g(λ) und −g′(λ) (g′ = ((1+2q) + √(1+4q−4q²))/2, q = 1/(1+λ)); die Ein-Schritt-Abbildung M/n hat −g(λ)/n und −g′(λ)/n, die Kette also die Eigenwerte 1 − g(λ)/n und 1 − g′(λ)/n. Dass 1 − g(λ)/n der **zweitgrößte** Eigenwert ist, ist **gemessen**
  (gerade n = 4 bis 16 für acht λ, n = 18 und 22 für λ = 1 und 5; nicht bewiesen); daraus folgt t_rel = n/g(λ).
* **Kreis:** träge Irrfahrt (1/2 bleiben, je 1/4 links und rechts), exakt über die Fourier-Eigenwerte. **Hyperwürfel {0,1}^n:** träge Irrfahrt, gerechnet über die Zählkette des Hamming-Gewichts. **Riffle-Mischen** (Gilbert/Shannon/Reeds): die exakte Formel der Bemerkung 8.14 mit Eulerschen Zahlen, in Brüchen. <!-- lint-ok: dezimalpunkt -->
* **Simulation:** R unabhängige Heat-bath-Ketten ab der Maximalbelegung, der Abstand der R Endzustände zu π wird nach jedem Schritt gemessen, die **geschätzte Mischzeit** ist das erste t mit Abstand ≤ 1/4. Die **Rauschgrenze** ist der erwartete Abstand von R exakten Ziehungen aus π zu π (exakt aus den Binomialverteilungen).

## Methodik

* **Exakt gerechnet, nicht simuliert:** d(t) und d̄(t) kommen aus der Eigenzerlegung der symmetrisierten Übergangsmatrix (P^t(x,y) = √(π_y/π_x) Σ μ_k^t U_xk U_yk); die Mischzeit ist das erste t mit Abstand ≤ ε, gefunden durch Halbierung (d und d̄ fallen in t). Dichte Rechnung für den Ring bis 12 Plätze
  (322 Zustände); d̄ (alle Paare von Starts) live bis 10 Plätze, sonst nur d. Der Kreis läuft über die Fourier-Eigenwerte, der Hyperwürfel über die Zählkette (Vektoriteration), die Karten über die Eulerschen Zahlen in Brüchen.
* **Vorgerechnet (gekennzeichnet „vorgerechnet“ in App und README):** die Tabelle aller Ring-Zellen (n = 3 bis 16, λ = 0,5 bis 100, beide Ketten), die Start-Tabelle, die 14 Zellen von d gegen d̄, die Simulationstabelle für R = 100 bis 100 000, die Fensterbreiten des Rings bis n = 24 und die Mischer auf Karten
  stehen in `msz_reference.json`, erzeugt von `tools/msz_make_tables.py` (`--check` rechnet den schnellen Teil neu und vergleicht, `--check --lang` alles). Für n = 13 und 14 gilt t_mix über alle Starts mit einer vollen Kontrolle je Zelle; für n = 15 und 16 ist es der Wert aus der Maximalbelegung
  (volle Kontrolle nur für λ = 1 und 5), für n = 16 bis 24 in der Fenstertabelle ebenfalls nur aus der Maximalbelegung (eine Untergrenze).
* **Zufall nur aus dem ganzzahligen SplitMix64** (`msz_rng.py`, plattformfest); die Simulation läuft vektorisiert in numpy (derselbe Strom, Test gegen den Skalar-Generator). Je Schritt und Kette werden zwei Werte gezogen (Platz, Münze). **Zufalls-Seed:** Lauf i bei R Ketten benutzt den Zufalls-Seed + 7·R + i.
* **Rechenzeit:** Messung dieser README, nicht im Test; siehe Abschnitt Rechenzeiten (jede Live-Eingabe unter 2 Sekunden auf der Entwicklungsmaschine).

## Befunde (gemessen, keine Behauptungen)

Zähleinheit: Schritte (Ring, Kreis, Hyperwürfel), Mischungen (Karten), Ketten (Simulation). Die Spalte „Test“ nennt den Test in `tests/test_claims.py`, der genau diese Zahl prüft (Ganzzahlen und Brüche exakt, Gleitkommazahlen auf die angegebenen Stellen, Zufallsproben als Band um den Wert der
vorgerechneten Tabelle mit festem Zufalls-Seed). „Ring“ heißt Hard-Core-Ring, wenn nichts anderes steht; „Heat-bath, alle Starts“ ist das Maximum über alle Zustände.

| Frage | Befund | Test |
|---|---|---|
| Drei Zahlen, eine Kette (Ring n = 4, λ = 1, Heat-bath, 7 Zustände) | d(11) = 0,2631 > 1/4 ≥ d(12) = 0,2435: **12**; d̄ erreicht 1/4 bei **21** (d̄(20) > 1/4 ≥ d̄(21)); T_mix(1/e) mit d̄ = **16** (d̄(15) = 0,3858, d̄(16) = 0,3575); mit d und 1/e schon 7 | `test_c01` |
| Umrechnung der Schwelle, derselbe Ring | t_mix(2^−k) = 4, 12, 21, 30, 39 für k = 1 bis 5, jeweils ≤ k · 12 | `test_c02` |
| Drei Zahlen, Ring n = 10, λ = 2, Heat-bath | **73** (d, 1/4), **118** (d̄, 1/4), **91** (d̄, 1/e), 48 (d, 1/e) | `test_c03` |
| d gegen d̄, 14 Zellen (n = 6, 8, 10, 12; λ = 1, 2, 5, 10, bei n = 12 nur 1 und 2) | d ≤ d̄ ≤ 2d auf **jedem** ganzzahligen t von 1 bis 3·t_mix in allen 14 Zellen; Verhältnis d̄/d zwischen 1,000 und 2,000; t̄_mix/t_mix zwischen 1,50 und 1,98 (Mittel 1,72), T_mix(1/e)/t_mix zwischen 1,20 und 1,43 (Mittel 1,30) | `test_c04` |
| Schwelle und Einheit, Ring n = 10, Start Maximalbelegung | λ = 2, Heat-bath: t_mix(ε) = 29 / 73 / 48 / 134 / 180 / 289 / 446 für ε = 1/2, 1/4, 1/e, 0,1, 0,05, 0,01, 0,001 (3,96-fach bei 0,01, 6,11-fach bei 0,001 gegen t_mix(1/4)); Metropolis 20 / 49 / 32 / 89 / 120 / 192 / 296; die Schranke ⌈log₂(1/ε)⌉·t_mix(1/4) gilt in allen 6 Zellen (λ = 1, 2, 5, beide Ketten) | `test_c05` |
| Zeiteinheit, Ring n = 10 | λ = 2: 73 Schritte (Heat-bath) = 7,30 Durchläufe = **2,433** Abfertigungsdauern; 49 Schritte (Metropolis) = **2,450**; λ = 1: 2,100 gegen 2,100; λ = 5: 3,467 gegen 3,460 | `test_c06` |
| Der Start, Ring n = 10, Heat-bath, alle 123 Starts | schlechtester Start / leerer Ring / Mittel über alle Starts / Mittel mit π gewichtet: λ = 1: 42 / 9 / 30,1 / 30,1; λ = 2: 73 / 10 / 48,9 / 54,0; λ = 5: 208 / 13 / 124,3 / 157,0; λ = 10: **569** / **48** / 303,3 / 442,7; der leere Start unterschätzt t_mix um das 4,7- / 7,3- / 16- / 11,9-fache; Anteil der Starts mit t_x ≤ t_mix/2: 0,09 / 0,09 / 0,21 / 0,66 | `test_c07` |
| Schlechtester Start | in allen 192 Zellen n = 3 bis 14 (acht λ, beide Ketten, alle Starts) ist t_mix gleich dem Wert aus der Maximalbelegung: die Maximalbelegung ist ein schlechtester Start (**Messung, kein Beweis**); für n = 10 und alle vier λ der Tabelle oben sind außerdem **alle** schlechtesten Starts Maximalbelegungen, und alle Maximalbelegungen sind gleich schlecht; für n = 15 und 16 nur aus der Maximalbelegung (volle Kontrolle für λ = 1 und 5) | `test_c08`, `test_c07` |
| Einschluss, Ring n = 10, λ = 2, Heat-bath | t_rel = 68,4; 46,7 ≤ **73** ≤ 569,2: obere Schranke 7,8-fach, exakt 1,56-fach die untere | `test_c09` |
| Einschluss, 30 Zellen (Heat-bath, n = 6, 8, 10, 12, 14; λ = 0,5, 1, 2, 5, 10, 20) | obere Schranke durch exakt: **4,2 bis 29,1** (Median 9,7); exakt durch untere Schranke: 1,01 bis 2,40 (Median 1,40); n = 12, λ = 20: 26,3-fach; Satz 12.4 stimmt in allen 224 Zellen der vorgerechneten Tabelle (n = 3 bis 16, acht λ, beide Ketten; für n = 15 und 16 mit dem Wert aus der Maximalbelegung) | `test_c10` |
| t_mix gegen t_rel, Ring, Heat-bath, gerade n = 6 bis 16, acht λ von 0,5 bis 100 (48 Zellen) | t_mix/t_rel zwischen **0,69 und 1,65**; bei λ = 100 zwischen 0,693 und 0,736 (nahe ln 2 = 0,693, Beobachtung); t_rel trifft t_mix in diesen 48 Zellen auf einen Faktor zwischen 0,69 und 1,65, die obere Schranke liegt bis zum 29,1-fachen daneben | `test_c11` |
| Einschluss, Kreis (träge) | n = 5 / 25 / 100 / 400 / 800: t_mix = 3 / 59 / 949 / 15 180 / 60 718; obere Schranke durch exakt 2,89 / 4,97 / **6,40** / 7,88 / 8,62; exakt durch untere 2,28 / 1,36 / 1,35 / 1,35 / 1,35 | `test_c12` |
| Einschluss, Hyperwürfel (träge) | n = 12 / 50 / 100 / 400 / 1024: t_mix = 20 / 120 / 275 / 1378 / 4009; obere Schranke durch exakt 5,8 / 15,0 / 25,7 / **80,9** / 181,7; exakt durch untere 2,62 / 3,53 / 4,01 / 4,98 / 5,65 | `test_c13` |
| Wachstum, Kreis | t_mix/n² = 0,1200 (n = 5), 0,0944 (n = 25), **0,0949** (n = 100 bis 800); n²/32 ≤ t_mix ≤ n² in allen fünf Fällen; Fenster (t(0,1) − t(0,9))/t(0,5) = 6,23 / 6,25 / 6,26 bei n = 50 / 200 / 800: **kein Cutoff**; nicht träge und gerades n: d(t) = 1/2 für alle t ≥ 1 (n = 4), t_mix unendlich | `test_c14` |
| Wachstum, Hyperwürfel | (t_mix − ½ n ln n)/n = 0,424 / 0,444 / 0,447 / 0,449 / 0,449 (n = 12 / 50 / 100 / 400 / 1024); Fenster 2,00 / 1,18 / **0,81** bei n = 16 / 128 / 1024: das Fenster fällt bis n = 1024 (Messung), im Einklang mit dem Cutoff-Satz 18.3 (gilt im Grenzwert); mit n ≤ 12 sieht man das Fallen nicht | `test_c15` |
| Wachstum, Hard-Core-Ring (λ = 1, Heat-bath, aus der Maximalbelegung) | t_mix = 31 / 54 / 79 / 105 / 133 bei n = 8 / 12 / 16 / 20 / 24; Fenster 3,86 / 3,15 / 2,83 / 2,58 / 2,42 (fallend wie beim Hyperwürfel): **Messung, kein Cutoff-Beleg** | `test_c16` |
| Karten, Riffle-Mischen, 52 Karten | Abstand nach 5 bis 13 Mischungen: 0,9237 / 0,6135 / **0,3341** / 0,1672 / 0,0854 / 0,0429 / 0,0215 / 0,0108 / 0,0054; t_mix(1/4) = 8; höchstens 0,01 erst bei **13** (Levin/Peres/Wilmer nennen „11 oder 12“, das gibt 0,0215 bzw. 0,0108); t_mix(1/4) für 4 / 8 / 16 / 32 / 52 / 64 / 128 Karten: 3 / 4 / 5 / 7 / 8 / 8 / 10; 3 Karten: 1/3, 7/48, 13/192, 25/768 | `test_c17` |
| Mischer auf Karten, n = 8, t_mix(1/4) (vorgerechnet, Start Identität) | Riffle 4 Mischungen, zufällige Transpositionen (zwei gleichverteilte Positionen, gleiche Positionen: nichts) 9 Schritte, Top-zu-zufällig (oberste Karte an eine gleichverteilte Stelle) 13 Schritte (andere Einheit: ein Riffle bewegt alle Karten); n = 3 bis 9 Karten: Transpositionen 2 / 3 / 4 / 6 / 7 / 9 / 10, Top-zu-zufällig 2 / 4 / 6 / 8 / 10 / 13 / 15 | `test_c18` |
| Simulation, Ring n = 10, λ = 2, exakt 73 (ganze Verteilung), Zufalls-Seeds 5000 und 6000, Läufe je R wie angegeben | R = 100 (20 Läufe): **keiner** erreicht die Schwelle bis t = 160, in beiden Zufalls-Seeds, weil die Rauschgrenze (0,4284) über 1/4 liegt; R = 300 (20): Mittel 129,0 [105, 153] und 131,45 [110, 155]; R = 1000 (20): 84,25 [79, 89] und 83,15 [75, 91]; R = 3000 (10): 74,6 [72, 78] und 77,0 [74, 83]; R = 10 000 (10): 73,9 [72, 75] und 73,7 [72, 76]; R = 100 000 (3): 73,0 [73, 73] in beiden | `test_c19` |
| Rauschgrenze (exakt, ganze Verteilung) | R = 100 / 300 / 1000 / 3000 / 10 000 / 100 000: **0,4284** / 0,2435 / 0,1321 / 0,0766 / 0,0419 / 0,0132 (Näherung ½ Σ √(2π_i(1 − π_i)/(πR)): 0,4189 / 0,2419 / 0,1325 / 0,0765 / 0,0419 / 0,0132); bei R = 100 liegt sie über der Schwelle 1/4 | `test_c20` |
| Eine perfekt gemischte Stichprobe hat im Mittel einen Abstand über 0 | zwei Zustände: R = 2 Ziehungen: erwarteter Abstand 1/4, R = 4: 3/16; die 7 Zustände des Rings mit 4 Plätzen, λ = 1: R = 7: 279 936/823 543 = 0,3399, R = 14: 0,2503 **über 1/4**; bei Gleichverteilung auf N Zuständen liegt der Abstand nie unter 1 − R/N | `test_c21` |
| Kennzahl statt Verteilung (Teilchenzahl, 6 Werte) | exakt **11** Schritte gegen 73 für die ganze Verteilung (Faktor 6,6); simuliert: Mittel 11,0 bis 12,6 über alle R und beide Zufalls-Seeds (12,05 bei R = 300, Zufalls-Seed 5000) | `test_c22` |
| Formel gerades n, Heat-bath | g(1) = 1 − 1/√2 = 0,2929; n(1 − λ₂) = g(λ) für gerade n = 4 bis 12 und λ = 0,5, 1, 2, 5, 10 (Eigenwerte gerechnet, relativ 1e-9), für gerade n = 4 bis 16 und alle acht λ der vorgerechneten Tabelle (relativ 1e-6) und für n = 18 und 22 (λ = 1, 5; Lanczos-Verfahren); t_rel = n/g(λ); n = 4, λ = 1: t_rel = 13,657, Einschluss 8,77 ≤ 12 ≤ 45,5 (3,8-fach, 1,37-fach) | `test_c23` |
| Eigenwerte | Heat-bath: kein negativer Eigenwert in allen 56 Zellen (n = 3 bis 10, sieben λ von 0,5 bis 100); Metropolis, λ = 1: kleinster Eigenwert zwischen −0,3536 (n = 4) und −0,3333 (n = 3) für n = 3 bis 12 | `test_c24` |
| Wo die exakte Rechnung endet | Zustände: n = 16: 2207, n = 24: 103 682, Hard-Core-Torus 8 × 8: 213 256 442 503 (Zeilen-Transfermatrix, Spur), 52 Karten: 52! ≈ 8,07·10⁶⁷; die Mischzeit des Torus wird nicht gerechnet | `test_c25` |

Messungen dieser README, die **nicht** im Test stehen: die Rechenzeiten (Abschnitt Methodik, Rechenzeiten) und der Zahlenwert der Rauschgrenzen-Näherung für R > 3000 (Test nur bis R = 3000).

**Negativbefunde (bleiben stehen):** (1) Die obere Schranke aus der Spektrallücke liegt als Zahl für die Mischzeit weit daneben (4,2- bis 29,1-fach im Ring in 30 Zellen, 80,9-fach beim Hyperwürfel mit n = 400); t_rel selbst trifft t_mix im Ring auf einen Faktor zwischen 0,69 und 1,65 (48 Zellen); beim Hyperwürfel mit n = 400 liegt t_mix beim
4,98-fachen der unteren Schranke. (2) Die Stichproben-Abstände sind eine Falle, kein gleichwertiger Weg zur Mischzeit: bei 100 Ketten erreicht keiner der 20 Läufe die Schwelle (in beiden Zufalls-Seeds), obwohl t_mix = 73 gilt. (3) Der leere Start ist im Ring n = 10 für λ = 1, 2, 5, 10 der **beste** Start, nicht der schlechteste (t_x = 9, 10, 13, 48).

## Befunde und Korrekturen gegenüber der ersten Rechnung

* **Die Teilchenzahl-Schätzung bei R = 100 liegt höher als in der ersten Rechnung:** 12,6 [9, 16] (Zufalls-Seed 5000) und 11,4 [7, 15] (Zufalls-Seed 6000) gegen 11,3 [9, 17]; die erste Rechnung benutzte den Zufallsgenerator von numpy, diese Demo den SplitMix64 mit einer anderen Reihenfolge der Ziehungen. Die Bänder sind der Befund, nicht die zweite Stelle;
  R = 300 gab 129,0 und 131,45 (Zufalls-Seeds 5000, 6000) gegen 131,7 in der ersten Rechnung, R = 3000 gab 74,6 und 77,0 gegen 75,0.
* **Die Rauschgrenze ist jetzt exakt** (Binomialsumme) statt aus 30 simulierten Ziehungen gemittelt: 0,4284 gegen 0,4291 bei R = 100; die Näherung ist bei R = 100 (0,4189) um 2,2 % kleiner als der exakte Wert.
* **Satz 12.4 gilt in 224 von 224 Zellen** (n = 3 bis 16, acht λ, beide Ketten), die erste Rechnung hatte 176 (n = 6 bis 16).
* **Der Kreis wächst nicht streng in n für alle n:** t_mix(n = 3) = t_mix(n = 4) = 1 (träge Irrfahrt); streng wachsend gilt ab n = 4 bis 60 (im Test gemessen).
* **„d̄ live“ nur bis 10 Plätze:** bei 11 und 12 Plätzen rechnet die App nur d live (die d̄-Kurven brauchen bei n = 12, λ = 2 3,8 s gegen 0,5 s für d allein); die 14 Zellen von d gegen d̄ stehen vorgerechnet in der Tabelle.
* **Hyperwürfel bis n = 1024** (statt 2048): der Wert für n = 2048 steht nicht in der App, weil die Zählkette dort mehrere Sekunden braucht. **Karten bis 128** (statt 104): die Rechnung in Brüchen reicht dafür.
* **Die Ising-Zahlen der ersten Rechnung (Glauber 36, Metropolis 24 bei n = 8, β = 0,5) kommen hier nicht vor**; der Hard-Core-Ring ist das Vehikel dieser Demo.
* **Plattform:** Zahlen unter Windows, Python 3.14; Ganzzahlen, Brüche und die SplitMix64-Läufe sind plattformfest, Gleitkommazahlen sind in den Tests mit Toleranz geprüft (Linux-Prüfung siehe Verifikation).

## Quellen

Nur so zitiert, wie sie gelesen sind (Stufe W = Wortlaut gelesen):

* Levin, Peres, Wilmer, *Markov Chains and Mixing Times*, 2. Aufl., AMS 2017, Volltext gelesen (W): Gleichungen (4.30) bis (4.34), Abschnitt 5.3.2, Abschnitt 8.3 mit Bemerkung 8.14 (S. 112 bis 114), Satz 12.4 (S. 163), Beispiele 12.16 und 12.17, Satz 18.3 (S. 265). <!-- lint-ok: dezimalpunkt -->
* Bayer, Diaconis, *Trailing the Dovetail Shuffle to its Lair*, Ann. Appl. Probab. 2(2), 294–313 (1992), Tabelle 1 (OCR, Zeichenverlust in Satz 2; die Tabelle ist gelesen).
* Propp, Wilson (1996), Random Structures & Algorithms 9, 223–252: für d̄ und die Schwelle 1/e, **nach dem Quellenbericht** (Vorabdruck; die Lesestufe ist in den Berichten nicht einheitlich, darum keine eigene Aussage darüber hinaus).
* Diaconis, *The Markov chain Monte Carlo revolution*, Bull. AMS 46(2), 179–205 (2009), Abschnitt 2.3, sinngemäß: nur für sehr wenige praktische Probleme lässt sich beantworten, wie viele Schritte nötig sind.
* **Nicht gelesen** (nie als gelesen ausgegeben): Norris, Handbook (Buch), Robert/Casella, Sinclair/Jerrum 1989, Besag 1974, Gelman/Rubin 1992 (nur zwei Seiten). Die Beziehung d ≤ d̄ ≤ 2d steht im Lehrbuch; die Nummer des Lemmas ist nicht im Wortlaut gelesen, darum nennt diese Demo nur ihren eigenen Beweis.
* Eigene Rechnungen ohne Quelle: die Zahlen im Hard-Core-Ring, die Formel t_rel = n/g(λ) für gerades n (Invarianz bewiesen, zweitgrößter Eigenwert gemessen), die Rauschgrenze der Stichprobe und die Simulationen. Die Aussage „Schranke aus der Spektrallücke ≠ Mischzeit“ ist Lehrbuch (Satz 12.4 liefert nur einen Einschluss).

## Abgrenzung zu Nachbar-Demos

* **Zahlen „Mischzeit“ in anderen Demos sind andere Größen** (drei Definitionen in drei Demos, dazu Einschwingzeit und Potenzmethode): [`meinungsbildung-demo`](https://github.com/sebastian-hanisch/meinungsbildung-demo) (Mittelwertbildung der Meinungen: Konvergenz zu einem Konsens, Schwelle 10⁻⁶, kein Abstand zu einer stationären Verteilung; die Irrfahrt als Eigenschaft des Netzes bleibt dort),
  [`konfigurationsmodell-demo`](https://github.com/sebastian-hanisch/konfigurationsmodell-demo) (Kantentausch, „Mischzeit in Vorschlägen je Kante“ an einer Kennzahl für ein Startnetz; die Kennzahl-Falle dieser Demo, 11 gegen 73 Schritte, bewertet diese Demo **nicht**),
  [`markov-queue-demo`](https://github.com/sebastian-hanisch/markov-queue-demo) (Potenzmethode „Schritte bis L1 < 10⁻⁸“ und Einschwingzeit bis 95 % einer Kennzahl), [`zulaessige-reihenfolgen-demo`](https://github.com/sebastian-hanisch/zulaessige-reihenfolgen-demo) (Abstand nach festen Schritten). Keine dieser Zahlen ist mit denen hier vergleichbar;
  keine der vier genannten Demos nutzt die Schwelle 1/4 für den Abstand zu einer stationären Verteilung (Stand der gelesenen READMEs, 2026-11-18); hier werden Abstand und Schwelle mit beiden Zähleinheiten (d und d̄) ausgeführt.
* **Burn-in, ESS, R̂, Metropolis im stetigen Raum:** [`posterior-sampling-demo`](https://github.com/sebastian-hanisch/posterior-sampling-demo); hier nichts davon wiederholt. **Burn-in** ist ein Verfahren der Praxis, **Einschwingzeit** eine Kennzahl in `markov-queue-demo`: beide heißen hier nicht Mischzeit.
  Heute gilt R̂ < 1,01 mit mindestens vier Ketten und ESS ≥ 400 (Vehtari u. a.); die Schwelle 1,1 ist veraltet.
* **Optimierer mit Abkühlen und Tempering** ([`simulated-annealing-demo`](https://github.com/sebastian-hanisch/simulated-annealing-demo), [`parallel-tempering-demo`](https://github.com/sebastian-hanisch/parallel-tempering-demo)): keine Mischzeit. **Wilson-Algorithmus für Spannbäume** ([`random-spanning-tree-demo`](https://github.com/sebastian-hanisch/random-spanning-tree-demo)):
  Nachbar-Demo zu Kopplung aus der Vergangenheit. **Gibbs-Maß und Austrittszeiten** ([`gleichgewichtsauswahl-demo`](https://github.com/sebastian-hanisch/gleichgewichtsauswahl-demo)): `kette-gleichgewicht-demo`; keine TV-Mischzeit hier wiederholt.
* **Zuständigkeit im Themenfeld:** der Takt-Beweis (Heat-bath gegen Metropolis, Abfertigungsdauern) steht in `gibbs-glauber-demo`, die Paarkette und die Kopplungszeit in `kopplung-ketten-demo`; hier stehen d̄ (Definition, Beweis d ≤ d̄ ≤ 2d, Tabelle), Satz 12.4, die Stichproben-Abstände und die 2×2-Matrix der langsamen Mode.
  Kopplung heißt in den Monte-Carlo-Demos „gemeinsame Zufallszahlen“; hier kommt Kopplung (zwei Ketten, Kopplungszeit) nicht vor.

## Ehrliche Grenzen

* **Ausgedachte Räume, keine Kulisse:** Hard-Core-Ring (mit dem Bild der Andockstellen, das nur unter den Voraussetzungen des Verlustnetzes gilt: gleiche Rate je Stelle, kein Warten, Nachbarsperre), Kreis, Hyperwürfel und Karten sind Standardbeispiele. Eine Betriebsaussage („so viele Schritte braucht ein reales System“) folgt nicht.
  Auf dem Ring lässt sich die Verteilung mit der Transfermatrix exakt berechnen und ziehen, einen Sampler braucht man dort nicht. Die **Mischzeit** des Torus ist dagegen nicht exakt berechenbar (zu viele Zustände für eine Matrix; die Demo rechnet sie nicht). Das heißt nicht, dass man dort einen Sampler braucht: ob exaktes Ziehen geht, hängt an der Breite des Torus, nicht an der Zustandszahl (Messung der Nachbar-Demo zur Verfahrenswahl, hier nicht nachgerechnet).
* **Nur kleine Ketten exakt:** live bis 12 Plätze (322 Zustände), vorgerechnet bis 16 Plätze; n = 24 nur aus der Maximalbelegung (Untergrenze, dort ist die Frage „ist die Maximalbelegung der schlechteste Start?“ nicht geprüft; bis n = 14 gilt es in allen Zellen, für n = 15 und 16 in den Zellen mit voller Kontrolle).
* **Cutoff nur als Messung bzw. Grenzwertsatz:** beim Hyperwürfel fällt das Fenster bis n = 1024, Satz 18.3 gilt im Grenzwert; im Ring fällt es von n = 8 bis 24 aus einem Start und für λ = 1: das ist eine Messung, kein Beleg.
* **Beweis offen:** dass 1 − g(λ)/n der zweitgrößte Eigenwert ist (gemessen für gerade n = 4 bis 16 und n = 18, 22, in den getesteten Fällen).
* **Metropolis ist nur wenig gemessen:** Tabelle der Schranken und der Mischzeiten für beide Ketten (n = 3 bis 16), die Schwellen-Tabelle nur für n = 10 und drei λ; d̄ nur für Heat-bath (14 Zellen); für Kreis, Hyperwürfel, Karten ist d̄ nicht gerechnet.
* **Die Simulation misst einen Ring, eine Kennzahl, einen Start:** Ring n = 10, λ = 2, Heat-bath, Maximalbelegung, Teilchenzahl als Kennzahl; die Läufe je R sind 20, 20, 20, 10, 10, 3; „keiner von 20“ gilt für die beiden Zufalls-Seeds 5000 und 6000, nicht als Absolutaussage. Das Verhältnis der Relaxationszeit zur integrierten Autokorrelationszeit und zur ESS ist hier nicht gemessen.
* **Mischzeit ist keine Burn-in-Länge** und keine Einschwingzeit; sie hängt am Start (hier das Maximum über alle Starts, in der Praxis unbekannt), an der Schwelle und am Abstand.
* **Die Mischzeit der Gate-Kette gegen die Einschwingzeit in `markov-queue-demo`** ist nicht gemessen und wird nicht verglichen.
* **Browser-Verhalten** (Kleben des Kopfes, Darstellung der Plotly-Diagramme) sieht der AppTest nicht; die Seite ist einmal im Browser auf Skriptfehler und KaTeX-Fehlermeldungen geprüft (keine gefunden), nicht jedes Diagramm im Einzelnen und nicht mit echtem Mausklick.

## Verifikation

Die Suite hat **515 Tests** (`pytest --collect-only -q tests`); der Test `test_readme_nennt_die_testzahl_der_suite` hält die Zahl aktuell.

- **Kern:** Matrix und Zustände gegen die Handrechnung des Rings mit 4 Plätzen (`tests/test_chain.py`), Abstände, Mischzeiten und Schranken gegen Brüche von Hand (`tests/test_mixing.py`), Kreis, Hyperwürfel, Karten (`tests/test_families.py`), Simulation und Rauschgrenze (`tests/test_simulation.py`).
- **Orakel:** `tests/test_oracle_mischzeit_exakt.py`: die Übergangsmatrix gegen die aus der Definition aufgezählte Bruch-Matrix (n ≤ 8, beide Ketten); d(t) und d̄(t) gegen die Potenz der Bruch-Matrix mit ganzen Zahlen (n = 4 bis 6, bis t = 40); t_mix für n = 6 bis 12 gegen eine frühere unabhängige Rechnung (`tests/frueher_t_mix.py`);
  Eigenwerte gegen die Formel g(λ) und gegen `scipy.sparse.linalg.eigsh`; Satz 12.4 in jeder Zelle; Kreis über Fourier gegen Matrixpotenz und Eigenwerte; Hyperwürfel gegen die volle 2^n-Kette; Riffle-Formel gegen die Aufzählung des inversen Riffle und gegen die Summe über alle Anordnungen.
- **Eigenschaften:** `tests/test_eigenschaften_mischzeit.py` (je 20 bis 50 feste Zufallsinstanzen): d fällt, d ≤ d̄ ≤ 2d, t_mix fällt in ε, Umrechnung der Schwelle, Drehung und Spiegelung des Starts, alle Maximalbelegungen gleich schlecht, Zeilensummen, Gleichgewicht, Detailed Balance, Heat-bath ohne negativen Eigenwert, Kreis streng wachsend ab n = 4, Karten, Rauschgrenze.
- **Literatur:** `tests/test_literatur_mischzeit.py` (0,3341 nach 7 Mischungen bei 52 Karten, Bayer/Diaconis Tabelle 1, n²/32 ≤ t_mix ≤ n², Eigenwerte und Lücken von Kreis und Hyperwürfel, Umrechnung der Schwelle).
- **Vorgerechnete Tabellen:** `tests/test_reference.py` und `python tools/msz_make_tables.py --check` (schneller Teil) bzw. `--check --lang` (alles).
- **Claims, Verdikte, Randfälle, Layout, Oberfläche:** `tests/test_claims.py` (jede Zahl der Tabelle oben), `tests/test_verdicts.py` (jedes Verdikt an handgesetzten Werten und je Preset die erwartete Zeile), `tests/test_edge_cases.py`, `tests/test_layout.py`, `tests/test_app.py` (jeder Regler am Rand, jede Auswahl),
  `tests/test_footer.py`, `tests/test_runtime_imports.py`.
- **Linux-Prüfung:** `python tools/demo_linux_check.py mischzeit-exakt-demo` (Docker, Python 3.12, aktuelle Bibliotheken, Stand 2026-11-18): alle 515 Tests grün (257 s). Die Zahlen mit Gleitkomma sind mit Toleranz geprüft, Ganzzahlen, Brüche und die SplitMix64-Läufe exakt.

## Rechenzeiten

Messung dieser README, **nicht im Test** (2026-11-18, Windows, Python 3.14, Runtime-venv, ein Prozess; auf derselben Maschine liefen gleichzeitig andere Rechnungen, die Zahlen sind deshalb eher obere Richtwerte). Zeit je Eingabe bei leerem Zwischenspeicher
(`st.cache_data`):

| Eingabe | Zeit |
|---|---|
| Experiment 1: n = 10, λ = 20, Metropolis (mit d̄, alle drei Mischzeiten, Kurven) | 1,3 s |
| Experiment 1: n = 12, λ = 20, Heat-bath (nur d) | 0,9 s |
| Experiment 1: Schwelle wechseln (n = 10, λ = 2); die Einheit zu wechseln rechnet nichts neu | 0,5 s |
| Experiment 2: n = 10, λ = 20 (alle 123 Starts) | 1,2 s |
| Experiment 3: Ring n = 12, λ = 20; Hyperwürfel n = 1024; Kreis n = 800 | 0,5 s; 0,4 s; 0,4 s |
| Experiment 4: alle acht Hyperwürfel-n gleichzeitig; 128 Karten | 1,7 s; 0,6 s |
| Experiment 5: R = 2000 mit 12 Wiederholungen; R = 100 mit 12 Wiederholungen (Teilchenzahl) | 0,5 s; 0,5 s |
| ganze Seite beim ersten Laden mit den Voreinstellungen (ohne Start von Python und Streamlit) | 6,2 s |

Die Grenze „d̄ live bis 10 Plätze“ kommt aus dieser Messung: bei n = 12, λ = 2 brauchen die d̄-Kurven auf 80 Zeitpunkten 3,8 s, die d-Kurven allein 0,5 s. Auf Streamlit Cloud ist mit der doppelten bis dreifachen Zeit zu rechnen (nicht gemessen).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Tabelle „Wo die exakte Rechnung endet“, Grenzen, Formeln |
| `msz_chain.py` | Hard-Core-Ring: Zustände, Heat-bath- und Metropolis-Matrix, Zustandszahl des Torus |
| `msz_mixing.py` | Eigenzerlegung, d, d̄, Mischzeiten, Schranken, Einheiten |
| `msz_families.py` | Kreis, Hyperwürfel, Riffle-Mischen, Mischer auf Karten |
| `msz_simulation.py` | Simulation mit SplitMix64 (vektorisiert), Rauschgrenze |
| `msz_rng.py` | SplitMix64 (Vorlage aus dem Website-Repo, unverändert) |
| `msz_reference.py`, `msz_reference.json` | die vorgerechneten Tabellen und ihr Lader |
| `msz_evaluation.py` | Auswertung der Experimente, Verdikte als reine Funktionen |
| `msz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `msz_texts.py` | Texte der App, Formeln |
| `msz_format.py` | Zahlen im deutschen Format |
| `msz_constants.py`, `msz_presets.py`, `msz_layout.py` | Grenzen und Presets, Permalink und Hervorhebung, fester Kopf |
| `tools/msz_make_tables.py` | rechnet die vorgerechneten Tabellen neu (`--check`, `--lang`) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, scipy, pandas, plotly) und pytest |
| `tests/` | Handrechnung, Orakel, Literatur, Eigenschaften, Claims, Verdikte, Randfälle, Layout, Oberfläche, Footer, Laufzeit-Importe; `oracle_helfer.py` und `frueher_t_mix.py` sind Hilfsdaten |

## Bewusst nicht umgesetzt

* **Ising-Kreis** als zweites Vehikel (die Zahlen der ersten Rechnung bleiben dort); **Hard-Core-Torus** als Mischzeit-Rechnung (zu viele Zustände für eine Matrix); **Kopplung** und **Simulation der Mischzeit mit Kopplung** (Nachbar-Demo `kopplung-ketten-demo`);
  **Burn-in, R̂, ESS** (Nachbar-Demos); **Metropolis-Tabelle der Schranken für Kreis und Hyperwürfel**; **Hyperwürfel mit n = 2048** (Rechenzeit); **PDF-Export** (die Ergebnisse stehen in den Tabellen der App).
* **Beweis, dass 1 − g(λ)/n der zweitgrößte Eigenwert ist:** nur gemessen; ein Beweis wird nur geführt, wenn ein Leser ihn verlangt.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html).
