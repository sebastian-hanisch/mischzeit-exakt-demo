"""Texte der App (Einleitung, Erklärung, Formeln, Grenzen). Zahlen stehen hier nur, wenn sie in tests/test_claims.py geprüft werden."""

EINLEITUNG = """
Wie viele Schritte braucht eine Kette, bis sie von **jedem Start** aus innerhalb von 1/4 an ihrem Gleichgewicht liegt? Diese Demo **rechnet die Mischzeit exakt** (aus der Übergangsmatrix, nicht aus einem Lauf)
und zeigt, dass es **die** Zahl nicht gibt: sie hängt am **Abstand** (d oder d̄), an der **Schwelle** und am **Start**. Gerechnet wird an einem ausgedachten Modell,
einem **Hard-Core-Ring** (Stellplätze im Ring, keine zwei benachbarten belegt), dazu an drei Standardbeispielen mit bekannter Antwort (Kreis, Hyperwürfel, Kartenmischen); eine reale Datenquelle gibt es dafür nicht.
Danach kommen die **drei Zahlen, die wie die Mischzeit aussehen und es nicht sind**: die Schranke aus der Spektrallücke, die Schätzung aus zu wenigen Ketten und eine Kennzahl statt der Verteilung.

Weiter unten: fünf Experimente (🔬), die Tabelle **Wo die exakte Rechnung endet**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

SO_FUNKTIONIERT = """
1. **Die Kette.** Im Hard-Core-Ring mit n Plätzen ist ein Zustand die Menge der belegten Plätze (keine zwei Nachbarn belegt), das Gleichgewicht π gewichtet jeden Zustand mit λ hoch Zahl der belegten Plätze.
   Der **Heat-bath** zieht einen Platz gleichverteilt und belegt ihn mit Wahrscheinlichkeit λ/(1+λ), wenn beide Nachbarn frei sind; **Metropolis** kehrt die Belegung des gezogenen Platzes um und nimmt sie mit min(1, π(neu)/π(alt)) an.
   Ein **Schritt** ist eine Platzaktualisierung bzw. ein Vorschlag.
2. **Der Abstand.** Nach t Schritten hat die Kette vom Start x aus die Verteilung P^t(x, ·). Der Abstand zu π ist **TV = ½ Σ |Verteilung − π|**; d(t) nimmt das Maximum über alle Starts, d̄(t) das Maximum über alle **Paare** von Starts (Propp/Wilson).
3. **Die Mischzeit.** t_mix(ε) ist das kleinste t mit d(t) ≤ ε; t_mix meint ε = 1/4. Wer d̄ oder die Schwelle 1/e nimmt, rechnet eine **andere Zahl** derselben Kette: die Kurven in Experiment 1 zeigen alle drei.
4. **Was die Mischzeit nicht ist.** Sie ist keine Einbrenn-Länge (Burn-in) für eine Simulation und keine Einschwingzeit einer Kennzahl; sie sagt, wann die **ganze Verteilung** erreicht ist.
5. **Zähleinheiten.** Gezählt werden Schritte einer Kette, Mischungen beim Kartenmischen und unabhängige Ketten in der Simulation. „Abfertigungsdauer“ ist die Zeiteinheit des Verlustnetz-Bildes: n Andockstellen im Ring,
   ein Lkw wird abgewiesen, wenn seine Stelle oder eine Nachbarstelle belegt ist; Heat-bath und Metropolis sind dasselbe Verlustnetz mit anderem Takt (n(1+λ) bzw. n·max(1, λ) Schritte je Abfertigungsdauer;
   der Beweis steht in der Nachbar-Demo zu Gibbs und Glauber). Das ist ein **Modell, keine Betriebsaussage**.
"""

KERNFRAGEN = """
**1.** Wie viele Schritte, bis die Kette von jedem Start aus innerhalb von 1/4 an π liegt, und was heißt „von jedem Start“?
**2.** Hängt die Zahl am Start, an der Schwelle, am Abstand?
**3.** Was sagt die Spektrallücke, was nicht?
**4.** Wie wächst die Zahl mit der Größe?
**5.** Kann man sie aus Läufen schätzen?
"""

E1_CAPTION = ("Heat-bath oder Metropolis auf dem Hard-Core-Ring; der Abstand wird aus der Eigenzerlegung der Übergangsmatrix berechnet, nicht aus einem Lauf. "
              "d(t) ist das Maximum über alle Starts, d̄(t) das über alle Paare von Starts; die gestrichelte Kurve gehört zum gewählten Start.")
E2_CAPTION = "Für jeden der Zustände: erstes t, ab dem dieser Start innerhalb von 1/4 an π liegt (Heat-bath). Die Mischzeit t_mix ist das größte dieser t."
E3_CAPTION = ("Satz 12.4 (Levin/Peres/Wilmer): (t_rel − 1)·ln(1/(2ε)) ≤ t_mix(ε) ≤ t_rel·ln(1/(ε·π_min)) mit t_rel = 1/γ*, γ* die absolute Spektrallücke **der Matrix**. "
              "Hier ε = 1/4. Die Schranke gilt für reversible, irreduzible und aperiodische Ketten (beim Kreis ohne Trägheit und bei geradem n ist die Kette periodisch und t_mix = ∞); die Frage ist, wie weit sie daneben liegt.")
E4_CAPTION = ("Andere Räume, andere Wachstumsgesetze. Die Kurven stehen über t/t_mix, damit der Übergang von „weit weg“ nach „gemischt“ vergleichbar wird. "
              "t(x) ist das erste t, ab dem der Abstand höchstens x ist (beim Ring vom Start Maximalbelegung aus); das Fenster (t(0,1) − t(0,9))/t(0,5) misst, wie lange der Abstand für den Weg von 0,9 auf 0,1 braucht, im Verhältnis zur Zeit bis 0,5.")
E5_CAPTION = ("Rechnung statt Simulation ist hier nur möglich, weil der Ring klein ist. Wer die Mischzeit aus R unabhängigen Ketten schätzen will, misst den empirischen Abstand der R Endzustände zu π; "
              "auch bei perfekt gemischten Ketten hat er eine Rauschgrenze (Ring n = 10, λ = 2, Heat-bath, Start Maximalbelegung, feste Instanz).")

GRENZEN = """
| Annahme | Was passiert, wenn sie verletzt ist | Wo es weitergeht |
|---|---|---|
| **Ausgedachte Räume mit exakt bekannter Antwort** (Hard-Core-Ring, Kreis, Hyperwürfel, Riffle-Mischen) | Es gibt keine reale Datenquelle dafür; die Zahlen gelten für diese Ketten (Heat-bath bzw. Metropolis mit Einzelplatz) und nicht für andere Verfahren oder Räume. | Nachbar-Demo zu Kopplung: Mischzeit **schätzen**, wo man sie nicht rechnen kann |
| **Nur kleine Ketten exakt** | Der Ring ist bis 12 Plätze live gerechnet (322 Zustände), bis 16 Plätze vorgerechnet; für n = 24 liegt nur der Wert aus der Maximalbelegung vor (eine Untergrenze). Der Torus ist nicht berechenbar. | Nachbar-Demo zu Kopplung von Ketten |
| **Cutoff nur als Messung** | Beim Hyperwürfel gilt der Cutoff im Grenzwert (Satz 18.3, Levin/Peres/Wilmer); im Ring ist das fallende Fenster bis n = 24 aus einem Start gemessen, **kein Beleg**. | – |
| **Der schlechteste Start ist eine Maximalbelegung** | Das ist gemessen (alle Starts bis n = 14, ab n = 15 aus der Maximalbelegung mit voller Kontrolle in einzelnen Zellen), nicht bewiesen. In der Praxis kennt man den schlechtesten Start nicht. | – |
| **Die Simulation misst einen Ring, eine Kennzahl, einen Start** | Die Rauschgrenze und die Zahl „nie erreicht“ gelten für Ring n = 10, λ = 2, R Ketten und die gewählten Zufalls-Seeds. | Diagnose in Nachbar-Demos (Täuschung durch R̂) |
| **Mischzeit ist keine Burn-in-Länge** | Die Zahl der Schritte bis zum Abstand 1/4 sagt nicht, wie viele Schritte man einer Simulation verwerfen soll; sie hängt am Start, an der Schwelle und am Abstand. | Posterior-Sampling (Burn-in und Diagnose im stetigen Raum) |
"""

FORMELN = r"""
**Modell (Hard-Core-Ring).** Zustände sind die unabhängigen Mengen des $n$-Rings (keine zwei Nachbarplätze belegt), ihre Zahl ist die Lucas-Zahl $L_n$ ($L_4=7$, $L_{10}=123$). Das Gleichgewicht ist
$$\pi(x)=\frac{\lambda^{\lvert x\rvert}}{Z},\qquad Z=\sum_{y}\lambda^{\lvert y\rvert}.$$
*Heat-bath:* ein Platz $v$ wird gleichverteilt gezogen (Wahrscheinlichkeit $1/n$). Sind beide Nachbarn frei, wird $v$ mit Wahrscheinlichkeit $p$ belegt und sonst geleert; ist ein Nachbar belegt, wird $v$ geleert (bleibt leer).
*Metropolis (Einzelplatz):* die Belegung des gezogenen Platzes wird umgekehrt und mit $\min(1,\pi(y)/\pi(x))$ angenommen (Umkehren eines Platzes mit belegtem Nachbarn ist unzulässig und wird abgelehnt).
Beide Ketten sind reversibel bezüglich $\pi$ ($\pi(x)P(x,y)=\pi(y)P(y,x)$), irreduzibel und aperiodisch (aperiodisch geprüft für n = 3 bis 8 und λ = 0,5 bis 20: eine Potenz der Matrix hat nur positive Einträge).

**Abstand und Mischzeit.**
$$d(t)=\max_x\mathrm{TV}\big(P^t(x,\cdot),\pi\big),\qquad \bar d(t)=\max_{x,y}\mathrm{TV}\big(P^t(x,\cdot),P^t(y,\cdot)\big).$$
$$t_{\mathrm{mix}}(\varepsilon)=\min\{t:\ d(t)\le\varepsilon\},\qquad \bar t_{\mathrm{mix}}(\varepsilon)=\min\{t:\ \bar d(t)\le\varepsilon\}.$$
*Es gilt $d(t)\le\bar d(t)\le 2d(t)$.* Beweis: $\pi$ ist das Mittel der Zeilen von $P^t$ (denn $\pi P^t=\pi$), und $\mathrm{TV}$ ist in jedem Argument konvex; daher $\mathrm{TV}(P^t(x,\cdot),\pi)\le\max_y\mathrm{TV}(P^t(x,\cdot),P^t(y,\cdot))$, also $d\le\bar d$.
Die Dreiecksungleichung über $\pi$ gibt $\mathrm{TV}(P^t(x,\cdot),P^t(y,\cdot))\le d(t)+d(t)$, also $\bar d\le 2d$. Daraus folgt $t_{\mathrm{mix}}(\varepsilon)\le\bar t_{\mathrm{mix}}(\varepsilon)\le t_{\mathrm{mix}}(\varepsilon/2)$.
*Umrechnung der Schwelle* (Levin/Peres/Wilmer (4.34)): $t_{\mathrm{mix}}(\varepsilon)\le\lceil\log_2(1/\varepsilon)\rceil\,t_{\mathrm{mix}}(1/4)$.

**Eigenzerlegung und Schranken.** Für eine reversible Kette ist $A=D^{1/2}PD^{-1/2}$ mit $D=\mathrm{diag}(\pi)$ symmetrisch, $A=U\,\mathrm{diag}(\mu)\,U^{\top}$, und
$$P^t(x,y)=\sqrt{\pi(y)/\pi(x)}\ \sum_k\mu_k^{\,t}\,U_{xk}U_{yk}.$$
Damit sind $d(t)$ und $\bar d(t)$ exakt berechenbar (die App rechnet so). *Satz 12.4* (Levin/Peres/Wilmer):
$$(t_{\mathrm{rel}}-1)\ln\frac{1}{2\varepsilon}\ \le\ t_{\mathrm{mix}}(\varepsilon)\ \le\ t_{\mathrm{rel}}\ln\frac{1}{\varepsilon\,\pi_{\min}}.$$

**Gerades $n$, Heat-bath (eigene Rechnung, keine Quelle).** Mit $f_A=\sum_v(-1)^vx_v$ und $f_B=\sum_v(-1)^vx_vx_{v+2}$ gilt für gerades $n$
$$n\,\mathbb E[\Delta f_A]=(2p-1)f_A-p\,f_B,\qquad n\,\mathbb E[\Delta f_B]=2p\,f_A-2f_B,$$
$\mathrm{span}\{f_A,f_B\}$ ist also invariant. Herleitung: ein gezogener Platz $v$ ändert $x_v$ im Mittel um $[p(1-x_{v-1})(1-x_{v+1})-x_v]/n$; die Terme $x_{v-1}x_{v+2}$ und $x_{v+1}x_{v-2}$ heben sich in $f_B$ wegen des wechselnden Vorzeichens $(-1)^v$ bei geradem $n$ weg (Indexverschiebung); Produkte benachbarter Plätze sind null, weil die Menge unabhängig ist;
für ungerades $n$ schließt sich $\sum_v(-1)^v$ nicht. Die $2\times2$-Matrix $M$ aus $n\,\mathbb E[\Delta f]=M f$ hat Spur $2p-3$ und Determinante $2q^2$, ihre Eigenwerte sind $-g$ und $-g'$ (die der Ein-Schritt-Abbildung $M/n$ also $-g/n$ und $-g'/n$) mit
$$g(\lambda)=\frac{4q^2}{(1+2q)+\sqrt{1+4q-4q^2}},\qquad q=\frac{1}{1+\lambda}.$$
Dabei ist $g'=\tfrac12\big((1+2q)+\sqrt{1+4q-4q^2}\big)$ die größere der beiden Wurzeln (Produkt $g\,g'=2q^2$). Der Eigenwert $1-g(\lambda)/n$ ist **gemessen** der zweitgrößte (für gerade $n=4$ bis $16$ bei acht $\lambda$, für $n=18$ und $22$ nur bei $\lambda=1$ und $5$; nicht bewiesen); daraus folgt $t_{\mathrm{rel}}=n/g(\lambda)$, linear in $n$.

**Kreis, Hyperwürfel, Karten.** *Kreis:* träge Irrfahrt (bleibt mit $1/2$, je $1/4$ nach links und rechts), Eigenwerte $(1+\cos(2\pi j/n))/2$, $\gamma^*=(1-\cos(2\pi/n))/2$; ohne Trägheit und bei geradem $n$ ist $-1$ ein Eigenwert, die Kette ist periodisch und $t_{\mathrm{mix}}=\infty$.
*Hyperwürfel $\{0,1\}^n$:* träge Irrfahrt, Eigenwerte $1-j/n$, $\gamma^*=1/n$, $\pi_{\min}=2^{-n}$; gerechnet über die Zählkette des Hamming-Gewichts. *Riffle-Mischen* von $n$ Karten (Gilbert/Shannon/Reeds), $k$ Mischungen, $r$ = Zahl der aufsteigenden Folgen einer Anordnung:
$$Q^k(\sigma)=\binom{2^k+n-r}{n}\Big/2^{kn},\qquad \mathrm{TV}=\frac12\sum_{r=1}^{n}A(n,r-1)\,\Big|\,Q^k_r-\frac1{n!}\Big|$$
mit den Eulerschen Zahlen $A(n,\cdot)$ (Levin/Peres/Wilmer, Abschnitt 8.3).

**Rauschgrenze der Simulation.** $R$ unabhängige Ziehungen aus $\pi$ haben in Zustand $i$ die Zahl $c_i\sim\mathrm{Bin}(R,\pi_i)$; der erwartete Abstand der empirischen Verteilung zu $\pi$ ist
$$\mathbb E\,\mathrm{TV}=\tfrac12\sum_i\mathbb E\,\Big\lvert\frac{c_i}{R}-\pi_i\Big\rvert .$$

**Literatur.** D. A. Levin, Y. Peres, E. L. Wilmer: *Markov Chains and Mixing Times*, 2. Aufl., AMS 2017 (Gleichungen (4.30) bis (4.34), Abschnitte 5.3, 8.3, 12.2, Satz 12.4, die Beispiele zu Kreis und Hyperwürfel in Kapitel 12, Satz 18.3); D. Bayer, P. Diaconis: *Trailing the Dovetail Shuffle to its Lair*, Ann. Appl. Probab. 2(2), 294–313 (1992), Tabelle 1;
J. G. Propp, D. B. Wilson (1996), Random Structures & Algorithms 9, 223–252, zum Abstand $\bar d$ und zur Schwelle $1/e$ (nach dem Quellenbericht, Vorabdruck); P. Diaconis: *The Markov chain Monte Carlo revolution*, Bull. AMS 46(2), 179–205 (2009), Abschnitt 2.3.
"""
