"""Zahlen im deutschen Format: Dezimalkomma, Gruppierung mit Leerzeichen ab fünf Stellen (DIN 5008)."""

import math


def ganz(n: int) -> str:
    """16796 → „16 796“, 1430 → „1430“."""
    s = str(abs(int(n)))
    if len(s) >= 5:
        gruppen = []
        while s:
            gruppen.append(s[-3:])
            s = s[:-3]
        s = " ".join(reversed(gruppen))
    return ("-" if n < 0 else "") + s


def dez(x: float, stellen: int = 2) -> str:
    """Dezimalzahl mit Komma; ganzzahliger Teil ab fünf Stellen gruppiert (60718,3 → „60 718,3“)."""
    if x is None:
        return "–"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    text = f"{x:.{stellen}f}"
    ganzer, _, rest = text.partition(".")
    vorz = "-" if ganzer.startswith("-") else ""
    ganzer = ganz(int(ganzer.lstrip("-"))) if ganzer.lstrip("-").isdigit() else ganzer
    return vorz + ganzer + ("," + rest if rest else "")


def zahl(x: float) -> str:
    """Einstellung wie λ: 2.0 → „2“, 0.5 → „0,5“, 12.5 → „12,5“."""
    return str(int(x)) if float(x).is_integer() else f"{x:g}".replace(".", ",")


def potenz(x: float) -> str:
    """Große oder kleine Zahl in der Schreibweise 2,1·10¹¹."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.1e}".split("e")
    hoch = str(abs(int(exp))).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant.replace('.', ',')}·10{'⁻' if int(exp) < 0 else ''}{hoch}"
