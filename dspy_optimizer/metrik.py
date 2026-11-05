"""Exact Match und F1 auf HotPotQA-Antworten.

Die Normalisierung folgt der ueblichen HotPotQA-Vorschrift: Kleinschreibung, Satzzeichen und
Artikel entfernen, Leerraum vereinheitlichen. Ohne sie zaehlte "The Eiffel Tower." gegen
"Eiffel Tower" als Fehler, obwohl die Antwort stimmt.
"""

import collections
import re
import statistics
import string
from typing import NamedTuple


def normalisiert(text):
    """Die Vergleichsform: kleingeschrieben, ohne Satzzeichen und Artikel, einheitlicher Leerraum."""
    text = text.lower()
    text = "".join(zeichen for zeichen in text if zeichen not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exakte_uebereinstimmung(vorhergesagt, richtig):
    """1.0, wenn beide Antworten nach der Normalisierung Zeichen fuer Zeichen gleich sind."""
    return float(normalisiert(vorhergesagt) == normalisiert(richtig))


def f1(vorhergesagt, richtig):
    """Wortweise Ueberlappung der beiden normalisierten Antworten, als F1.

    Woerter zaehlen mit ihrer Haeufigkeit: steht ein Wort in der Vorhersage zweimal und in
    der richtigen Antwort einmal, gilt es als *ein* Treffer. Sonst koennte eine Vorhersage
    durch Wiederholung eine Genauigkeit ueber 1 erreichen.
    """
    vorhergesagte_woerter = normalisiert(vorhergesagt).split()
    richtige_woerter = normalisiert(richtig).split()

    gemeinsam = collections.Counter(vorhergesagte_woerter) & collections.Counter(richtige_woerter)
    treffer = sum(gemeinsam.values())
    if treffer == 0:
        # Faengt zugleich die leere Antwort ab, bei der der Nenner unten 0 waere.
        return 0.0

    genauigkeit = treffer / len(vorhergesagte_woerter)
    trefferquote = treffer / len(richtige_woerter)
    return 2 * genauigkeit * trefferquote / (genauigkeit + trefferquote)


class Bewertung(NamedTuple):
    """Die Bewertung eines einzelnen Beispiels. `absatztreffer` wird nur protokolliert."""

    exakt: float
    f1: float
    absatztreffer: float | None


def absatztreffer(beispiel, vorhersage):
    """Anteil der goldenen Absaetze, die Modul 1 herausgezogen hat; `None`, wenn unbekannt.

    Das ist eine Trefferquote ueber die goldenen Absaetze, keine Genauigkeit: dass Modul 1
    zusaetzlich Ablenker mitnimmt, senkt den Wert nicht. Zu viel Kontext schlaegt sich in den
    Kosten nieder, ist aber kein verfehlter Absatz.

    Verglichen wird auf der normalisierten Form, damit ein Modell nicht an einem Punkt oder
    einem grossen Anfangsbuchstaben im Titel scheitert.
    """
    genannte_titel = getattr(vorhersage, "relevant_titles", None)
    if genannte_titel is None:
        return None
    if isinstance(genannte_titel, str):
        genannte_titel = [genannte_titel]

    gewaehlte_titel = {normalisiert(titel) for titel in genannte_titel}
    goldene_titel = {normalisiert(titel) for titel in beispiel.gold_titles}
    if not goldene_titel:
        return None
    return len(gewaehlte_titel & goldene_titel) / len(goldene_titel)


def bewertung_von(beispiel, vorhersage):
    """Exact Match, F1 und Absatztreffer eines einzelnen Beispiels, nebeneinander.

    Fehlt die Antwort ganz, etwa weil der DSPy-Adapter die Ausgabe eines Kandidatenmodells
    nicht auswerten konnte, zaehlt das als leere Antwort und damit als null Punkte. Der Lauf
    darf daran nicht scheitern, sonst entwertete ein einziger Formatfehler die ganze
    Bewertung.
    """
    antwort = getattr(vorhersage, "answer", None) or ""
    return Bewertung(
        exakt=exakte_uebereinstimmung(antwort, beispiel.answer),
        f1=f1(antwort, beispiel.answer),
        absatztreffer=absatztreffer(beispiel, vorhersage),
    )


def bewerte(beispiel, vorhersage, trace=None):
    """Die Metrik, auf die optimiert wird: der F1 der Antwort.

    Der Absatztreffer geht ausdruecklich **nicht** ein; er wird nur protokolliert.

    Der Parameter `trace` bestimmt die Rolle, in der DSPy die Metrik aufruft:

    - ohne `trace` zum *Bewerten*: dann zaehlt der stetige F1, weil der gepaarte Test der
      Zwischenbewertung Abstufungen braucht;
    - mit `trace` zum *Sammeln von Few-Shot-Beispielen*: dann entscheidet der Rueckgabewert,
      ob ein Beispiel als Demonstration in den Prompt kommt. Ohne `metric_threshold` gilt dort
      jeder Wert ueber 0 als gelungen (`dspy/teleprompt/bootstrap.py` Z. 206-209). Ein F1 von
      2/3 fuer eine *falsche* Antwort wuerde so zur Demonstration. Deshalb zaehlt beim
      Sammeln nur die exakte Antwort.
    """
    bewertung = bewertung_von(beispiel, vorhersage)
    if trace is not None:
        return bewertung.exakt == 1.0
    return bewertung.f1


def zusammenfassung(bewertungen):
    """Mittelwerte ueber mehrere Beispiele, jede Groesse fuer sich.

    Der Absatztreffer wird nur ueber die Beispiele gemittelt, bei denen er bekannt ist: ein
    unbekannter Wert ist kein verfehlter Absatz. Sind alle unbekannt, bleibt er `None`.
    """
    if not bewertungen:
        return Bewertung(exakt=0.0, f1=0.0, absatztreffer=None)

    bekannte_treffer = [
        bewertung.absatztreffer for bewertung in bewertungen if bewertung.absatztreffer is not None
    ]
    return Bewertung(
        exakt=statistics.fmean(bewertung.exakt for bewertung in bewertungen),
        f1=statistics.fmean(bewertung.f1 for bewertung in bewertungen),
        absatztreffer=statistics.fmean(bekannte_treffer) if bekannte_treffer else None,
    )
