"""Der Kostenzaehler: was ein Lauf gekostet hat, aus zwei Quellen.

Der naheliegende Weg - ein `dspy.track_usage()` um den ganzen Lauf - unterschlaegt die
Aufrufe aus der parallelen Evaluation systematisch: der Tracker wird je Arbeits-Thread
tief kopiert und die Kopie danach verworfen (`utils/parallelizer.py` Z. 118-136). Gerade
dieser Anteil macht den Grossteil der Kosten aus.

Erfasst wird deshalb aus zwei Quellen:

- **Quelle 1, der Callback** `on_evaluate_end`: summiert ueber die Einzelergebnisse einer
  Evaluation (`prediction.get_lm_usage()`). Das sind die Tokens aus den Arbeits-Threads,
  dazu die Zahl der Programmlaeufe und jeder sichtbar gewordene Fehler.
- **Quelle 2, die History der Sprachmodelle**: sie ist eine gewoehnliche Liste am
  LM-Objekt und kein Thread-Local, haelt also die Aufrufe aus *allen* Threads. Von dort
  kommen die Dollar-Betraege (`eintrag["cost"]`), der bedienende Anbieter und ob eine
  Antwort am `max_tokens`-Limit abgeschnitten wurde.

Die beiden Quellen ueberlappen sich absichtlich: der Hauptthread-Anteil (Bootstrapping und
Proposer) ist die Differenz, und wo Quelle 1 mehr Tokens meldet als Quelle 2, hat der
Ringpuffer der History Eintraege verloren - das wird gemeldet statt verschwiegen. Diese
Differenz gibt es nur in Tokens, nicht in Dollar.

Die **Fixkosten** eines MIPROv2-Laufs sind dagegen in Dollar gemessen: der Dollarstand am Ende
seiner ersten vollstaendigen Bewertung (`on_evaluate_start`/`on_evaluate_end`). Sie sind
nicht dasselbe wie der Hauptthread-Anteil: sie enthalten die erste Bewertung mit, und keine
spaeteren Aufrufe des Hauptthreads.

Ein Messlauf hinterlaesst immer genau eine Protokollzeile, auch wenn er abstuerzt: was er
bis dahin ausgegeben hat, ist danach nicht mehr zu rekonstruieren.
"""

import contextlib
import pathlib
import statistics
import sys
import threading
import time
from typing import NamedTuple

import dspy
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

from dspy_optimizer.protokoll import Protokoll, pruefe_einordnung


class Tokens(NamedTuple):
    """Prompt- und Antwort-Tokens getrennt: die Preise der beiden sind verschieden."""

    prompt: int = 0
    antwort: int = 0

    @property
    def gesamt(self):
        return self.prompt + self.antwort

    def __add__(self, andere):
        return Tokens(self.prompt + andere.prompt, self.antwort + andere.antwort)

    def __sub__(self, andere):
        return Tokens(self.prompt - andere.prompt, self.antwort - andere.antwort)


class Messwerte(NamedTuple):
    """Was ein Abschnitt gekostet hat. Ein Abschnitt ist die Optimierung *oder* eine
    Bewertung des fertigen Programms - nie beides zugleich, sonst waeren
    Optimierungskosten und Ausfuehrungskosten nicht getrennt."""

    dollar: float
    lm_aufrufe: int
    programmlaeufe: int
    fehlgeschlagene_programmlaeufe: int
    auswertungen: int
    tokens: Tokens
    tokens_aus_evaluationen: Tokens
    # Anbieter und die Zahl der Aufrufe, die er bedient hat. Ein Ausweichen auf einen
    # anderen Anbieter faellt damit nicht nur auf, es ist auch bezifferbar.
    anbieter: dict
    stoerungen: tuple[str, ...]
    # Wie lange der Abschnitt lief, und je gelungenem Programmlauf, wie lange er dauerte.
    # Die Latenz wird nur berichtet und steuert nichts.
    dauer_sekunden: float | None = None
    latenzen: tuple[float, ...] = ()
    # Gescheiterte Aufrufe, nach Ursache getrennt: ein Formatfehler ist eine Schwaeche des
    # Modells, ein Anbieterfehler sagt nichts ueber das Modell.
    formatfehler: int = 0
    anbieterfehler: int = 0
    # Je Sprachmodell des Messlaufs, in dessen Reihenfolge: wie viele Antworten am
    # `max_tokens`-Limit endeten.
    abgeschnitten: tuple[int, ...] = ()
    # Die Fixkosten dieses Abschnitts: die Dollar bis zum Ende seiner ersten vollstaendigen
    # Bewertung. `None`, wo es keine gab -
    # in jeder Bewertung des fertigen Programms und in einem Lauf, der vorher abbrach.
    fixkosten: float | None = None

    @property
    def abgeschnittene_antworten(self):
        """Die abgeschnittenen Antworten, getrennt nach Kandidaten- und Vorschlagsmodell.

        Eine abgeschnittene Antwort des Kandidatenmodells zaehlt als Formatfehler und ist von
        einer Antwort im falschen Format sonst nicht zu unterscheiden; beim Vorschlagsmodell
        bricht sie den Optimierungslauf ab. Die Vorstudie zeigt so, ob `max_tokens` irgendwo
        knapp wird.

        Beim Vorschlagsmodell `None` statt 0, wenn es zu diesem Messlauf nicht gehoert - etwa
        in einer Bewertung: nicht gezaehlt ist etwas anderes als keine gefunden.
        """
        kandidatenmodell, *vorschlagsmodell = self.abgeschnitten or (0,)
        return {
            "kandidatenmodell": kandidatenmodell,
            "vorschlagsmodell": vorschlagsmodell[0] if vorschlagsmodell else None,
        }

    @property
    def tokens_aus_dem_hauptthread(self):
        """Der Anteil, der durch keine Evaluation lief: Bootstrapping und Proposer.

        Nur Tokens, und darunter auch die von Programmlaeufen, die in einer Evaluation
        scheiterten. Die Fixkosten in Dollar stehen in `fixkosten`.
        """
        return self.tokens - self.tokens_aus_evaluationen

    @property
    def dollar_je_programmlauf(self):
        """Die Ausfuehrungskosten, wenn der Abschnitt das fertige Programm ausgefuehrt hat.

        `None` statt 0.0, wenn nichts gelaufen ist: keine Messung ist etwas anderes als
        eine Messung von null.
        """
        if not self.programmlaeufe:
            return None
        return self.dollar / self.programmlaeufe

    def als_felder(self):
        """Flach und als reine JSON-Werte - so gehen die Messwerte ins Protokoll."""
        return {
            "dollar": self.dollar,
            "dollar_je_programmlauf": self.dollar_je_programmlauf,
            "fixkosten": self.fixkosten,
            "tokens_prompt": self.tokens.prompt,
            "tokens_antwort": self.tokens.antwort,
            "tokens_aus_evaluationen": self.tokens_aus_evaluationen.gesamt,
            "tokens_aus_dem_hauptthread": self.tokens_aus_dem_hauptthread.gesamt,
            "lm_aufrufe": self.lm_aufrufe,
            "programmlaeufe": self.programmlaeufe,
            "fehlgeschlagene_programmlaeufe": self.fehlgeschlagene_programmlaeufe,
            "formatfehler": self.formatfehler,
            "anbieterfehler": self.anbieterfehler,
            "abgeschnittene_antworten": self.abgeschnittene_antworten,
            "auswertungen": self.auswertungen,
            "anbieter": dict(self.anbieter),
            "stoerungen": list(self.stoerungen),
            "dauer_sekunden": self.dauer_sekunden,
            # Mittelwert und Median, weil einzelne wiederholte Anfragen den Mittelwert
            # verzerren. `None` statt 0.0, wo nichts gestoppt wurde.
            "latenz_mittelwert_sekunden": (
                statistics.fmean(self.latenzen) if self.latenzen else None
            ),
            "latenz_median_sekunden": (
                statistics.median(self.latenzen) if self.latenzen else None
            ),
        }


def _tokens_aus_verbrauch(verbrauch):
    """Die Tokens einer Prediction, summiert ueber alle beteiligten Sprachmodelle.

    `get_lm_usage()` gibt `{Modellname: {"prompt_tokens": ..., "completion_tokens": ...}}`
    zurueck (`utils/usage_tracker.py` Z. 57-65).
    """
    tokens = Tokens()
    for einzeln in (verbrauch or {}).values():
        tokens += Tokens(einzeln.get("prompt_tokens") or 0, einzeln.get("completion_tokens") or 0)
    return tokens


def _ist_abgeschnitten(antwort):
    """Ob eine Antwort am `max_tokens`-Limit endete - so meldet es der Anbieter selbst
    (`finish_reason == "length"`, wie in `spikes/modellwechsel.py` Z. 98-99)."""
    wahlen = getattr(antwort, "choices", None) or ()
    return any(getattr(wahl, "finish_reason", None) == "length" for wahl in wahlen)


class Kostenzaehler(BaseCallback):
    """Der Zaehler eines Abschnitts. Ein Objekt, von mehreren Threads benutzt.

    Gezaehlt wird, was **ab seiner Entstehung** passiert. Die History der Sprachmodelle
    haelt auch die Aufrufe frueherer Abschnitte; ohne diesen Schnitt zaehlte der zweite
    Abschnitt eines Laufs die Aufrufe des ersten noch einmal mit, und Optimierungskosten
    steckten in den Ausfuehrungskosten.

    Die Callback-Objekte werden - anders als der Usage-Tracker - beim Uebergang in einen
    Arbeits-Thread **nicht** kopiert (`utils/parallelizer.py` Z. 128-130). Der Zaehler ist
    also ein geteiltes Objekt und schuetzt seine Summen mit einem Schloss.

    Die Reihenfolge der `sprachmodelle` hat eine Bedeutung: das erste ist das
    Kandidatenmodell, das zweite, sofern MIPROv2 laeuft, das Vorschlagsmodell. Danach
    trennen sich die abgeschnittenen Antworten (`Messwerte.abgeschnittene_antworten`).
    """

    def __init__(self, *sprachmodelle):
        self._sprachmodelle = sprachmodelle
        self._schloss = threading.Lock()
        self._dollar = 0.0
        self._lm_aufrufe = 0
        self._tokens = Tokens()
        self._tokens_aus_evaluationen = Tokens()
        self._programmlaeufe = 0
        self._fehlgeschlagene_programmlaeufe = 0
        self._formatfehler = 0
        self._anbieterfehler = 0
        # Kennungen der laufenden Predict-Aufrufe, bis sie enden (`on_module_end`).
        self._predict_aufrufe = set()
        self._auswertungen = 0
        self._anbieter = {}
        self._abgeschnitten = [0] * len(sprachmodelle)
        self._stoerungen = []
        self._beendet = False
        self._beginn = time.monotonic()
        self._ende = None
        self._latenzen = []
        # Gestoppte, aber noch nicht gezaehlte Programmlaeufe: Kennung der Prediction ->
        # (Prediction, Dauer). Die Prediction bleibt mit drin, damit ihre Kennung nicht an
        # ein neues Objekt vergeben wird, solange sie hier wartet.
        self._gestoppt = {}
        # Die erste vollstaendige Bewertung: ihre Kennung, sobald sie beginnt, und der
        # Dollarstand an ihrem Ende - die Fixkosten.
        self._erste_volle_bewertung = None
        self._fixkosten = None
        self.bewertung = None
        self.exakt = None
        self.absatztreffer = None
        self.einzelbewertungen = None
        # Je Sprachmodell die Kennungen der History-Eintraege, die schon in der Summe
        # stecken - zu Beginn alles, was vor diesem Abschnitt gelaufen ist. Behalten wird
        # nur, was noch in der History steht; damit waechst die Menge nicht ueber die
        # Puffergroesse hinaus.
        self._gezaehlt = [
            {eintrag.get("uuid") for eintrag in sprachmodell.history}
            for sprachmodell in sprachmodelle
        ]

    def on_evaluate_start(self, call_id, instance, inputs):
        """Merkt sich die erste vollstaendige Bewertung eines MIPROv2-Laufs.

        MIPROv2 kennzeichnet jede Bewertung: `eval_full` fuer die vollstaendigen, `eval_minibatch`
        fuer die Minibatches (`teleprompt/utils.py` Z. 50-58). Die erste vollstaendige ist die
        des unveraenderten Programms, am Anfang von Schritt 3; sie gehoert zu den Fixkosten.
        Eine Bewertung ohne Kennzeichen - etwa die Nachbewertung - beruehrt die Fixkosten nicht.
        """
        metadaten = (inputs or {}).get("callback_metadata") or {}
        if metadaten.get("metric_key") != "eval_full":
            return
        with self._schloss:
            if self._erste_volle_bewertung is None:
                self._erste_volle_bewertung = call_id

    def on_evaluate_end(self, call_id, outputs, exception=None):
        """Quelle 1: eine abgeschlossene Evaluation, mit allen Einzelergebnissen.

        Feuert genau einmal je Trial, weil MIPROv2 jede Bewertung ueber `Evaluate.__call__`
        fuehrt. Fehler duerfen hier nicht durchschlagen: DSPy verschluckt jede Exception aus
        einem Callback (`utils/callback.py` Z. 319-335), der Lauf liefe also ohne Zaehler weiter.
        """
        with self._schloss:
            self._auswertungen += 1
            if exception is not None:
                self._stoerungen.append(f"Evaluation abgebrochen: {exception}")
            elif not getattr(outputs, "results", None):
                self._stoerungen.append(
                    "Eine Evaluation kam ohne Einzelergebnisse zurueck. DSPy macht aus einem "
                    "Fehler waehrend der Evaluation still eine Bewertung von 0,0 "
                    "(teleprompt/utils.py Z. 59-62) - diese Zahl ist keine Modellleistung."
                )
            else:
                for _, vorhersage, _ in outputs.results:
                    self._programmlaeufe += 1
                    verbrauch = vorhersage.get_lm_usage() if vorhersage is not None else None
                    if not verbrauch:
                        # Ein ausgefallenes Beispiel ersetzt DSPy durch eine leere Prediction
                        # mit der Bewertung 0,0 (evaluate/evaluate.py Z. 178).
                        self._fehlgeschlagene_programmlaeufe += 1
                        continue
                    self._tokens_aus_evaluationen += _tokens_aus_verbrauch(verbrauch)
                    # Die Latenz zaehlt genau die Programmlaeufe, die hier als gelungen
                    # gelten - dieselbe Definition wie bei den Fehlschlaegen.
                    gestoppt = self._gestoppt.pop(id(vorhersage), None)
                    if gestoppt is not None:
                        self._latenzen.append(gestoppt[1])

        self.hole_ab()
        self._halte_fixkosten_fest(call_id, exception)

    def _halte_fixkosten_fest(self, call_id, exception):
        """Am Ende der ersten vollstaendigen Bewertung: der Dollarstand bis hierher sind die
        Fixkosten.

        Abgelesen wird nach `hole_ab()`, also mit allen Aufrufen der Bewertung. Beispiele
        sammeln und Instruktionen vorschlagen laufen vorher und nie durch eine Bewertung;
        alles davor gehoert also zum festen Teil, und Aufrufe spaeterer Bewertungen koennen
        nicht hineinrutschen. Bricht die erste vollstaendige Bewertung ab, bleiben die
        Fixkosten leer: ein Bruchteil der Bewertung waere eine zu kleine Messung.
        """
        with self._schloss:
            if call_id == self._erste_volle_bewertung and exception is None:
                self._fixkosten = self._dollar

    def on_lm_end(self, call_id, outputs, exception=None):
        """Ein LM-Aufruf ist zu Ende; endete er mit einem Fehler, ist das ein Anbieterfehler.

        Gezaehlt wird an der Grenze des Aufrufs, nicht nach der Art des Fehlers: was hier
        ankommt, hat gar keine Antwort geliefert - Rate Limit, Serverfehler,
        Zeitueberschreitung. Die Wiederholungen liegen **innerhalb** des Aufrufs:
        `dspy.LM.forward` reicht `num_retries` an `litellm_completion` weiter (`clients/lm.py`
        Z. 252-256), und dort wiederholt LiteLLM (Z. 495-510). Gezaehlt wird also nur, was nach
        ihnen uebrig bleibt, und zwar einmal je Aufruf, nicht je Versuch.

        Gezaehlt wird jedes Sprachmodell des Messlaufs, auch das Vorschlagsmodell: dessen
        Anbieterfehler macht einen Lauf ebenso wenig vergleichbar.

        Die Art des Fehlers geht als Stoerung ins Protokoll, je Art einmal. Nicht jeder Fehler
        an dieser Grenze liegt beim Anbieter - ein ueberschrittenes Kontextfenster etwa haengt
        am Modell; ohne den Namen waere das hinterher nicht mehr zu erkennen.
        """
        if exception is not None:
            with self._schloss:
                self._anbieterfehler += 1
            self.melde(f"Anbieterfehler: {type(exception).__name__}")

    def on_module_start(self, call_id, instance, inputs):
        """Merkt sich die Predict-Aufrufe: nur an ihnen wird ein Formatfehler gezaehlt."""
        if isinstance(instance, dspy.Predict):
            with self._schloss:
                self._predict_aufrufe.add(call_id)

    def on_module_end(self, call_id, outputs, exception=None):
        """Ein Predict-Aufruf, dessen Antwort der Adapter nicht lesen konnte: ein Formatfehler.

        Die Antwort ist angekommen und bezahlt, nur fehlen die Marken, die der ChatAdapter
        sucht - oder sie ist leer. Beides meldet der Adapter als `AdapterParseError`
        (`adapters/base.py` Z. 165-177). Gezaehlt wird nur am Predict: derselbe Fehler steigt
        danach durch jedes umgebende Modul auf und kaeme sonst mehrfach an.
        """
        with self._schloss:
            war_predict = call_id in self._predict_aufrufe
            self._predict_aufrufe.discard(call_id)
            if war_predict and isinstance(exception, AdapterParseError):
                self._formatfehler += 1

    def vermerke_programmlaeufe(self, anzahl):
        """Programmlaeufe, die durch keine Evaluation liefen - etwa der Lauf von Hand."""
        with self._schloss:
            self._programmlaeufe += anzahl

    def vermerke_bewertung(
        self, bewertung, *, exakt=None, absatztreffer=None, einzelbewertungen=None
    ):
        """Die Bewertung, die dieser Abschnitt erbracht hat, fuer die Protokollzeile.

        Sie kommt von aussen: der Zaehler weiss, was ein Lauf gekostet hat, aber nicht, was
        er wert war. `bewertung` ist der F1; Exact Match und Absatztreffer gehen, wo sie
        gemessen wurden, daneben mit ins Protokoll.
        `einzelbewertungen` sind die F1-Werte je Beispiel, in der Reihenfolge
        der Daten - nur mit ihnen laesst sich der gepaarte Test rechnen oder eine
        Zwischenbewertung nachrechnen.
        """
        self.bewertung = bewertung
        self.exakt = exakt
        self.absatztreffer = absatztreffer
        self.einzelbewertungen = einzelbewertungen

    def mit_stoppuhr(self, programm):
        """Das Programm, mit einer Stoppuhr um jeden Programmlauf - die Latenz.

        Gemeint ist die Wartezeit auf eine beantwortete Frage, nicht auf einen einzelnen
        LM-Aufruf. DSPy misst sie nicht von selbst. `dspy.Evaluate` ruft das
        Programm bloss auf (`evaluate/evaluate.py` Z. 171), eine Funktion genuegt ihr also.
        Das Programm selbst bleibt unveraendert - wer es danach weiterreicht, reicht keine
        Stoppuhr weiter.

        Gezaehlt wird eine gestoppte Dauer erst in `on_evaluate_end`, und nur, wenn der
        Programmlauf dort als gelungen gilt. Ein gescheiterter hat keine Frage beantwortet.
        Wuerde gleich beim Stoppen gezaehlt, liefen zwei Definitionen von "gescheitert"
        auseinander: scheitert erst die Metrik, ist das Programm schon durchgelaufen, DSPy
        fuehrt das Beispiel aber als leere Prediction (`evaluate/evaluate.py` Z. 170-178).
        Und ein Beispiel, das DSPy nach 120 Sekunden ein zweites Mal startet
        (`utils/parallelizer.py` Z. 202-210), stuende zweimal in der Latenz.
        """

        def mit_stoppuhr(*args, **kwargs):
            beginn = time.monotonic()
            vorhersage = programm(*args, **kwargs)
            dauer = time.monotonic() - beginn
            with self._schloss:
                self._gestoppt[id(vorhersage)] = (vorhersage, dauer)
            return vorhersage

        return mit_stoppuhr

    def beende(self):
        """Den Abschnitt abschliessen: ein letztes Mal abholen, danach nichts mehr zaehlen.

        Ohne diesen Schnitt liefe der Zaehler weiter, solange jemand sein Sprachmodell
        benutzt - und ein spaeter abgefragter Stand enthielte die Aufrufe des naechsten
        Abschnitts.
        """
        self.hole_ab()
        with self._schloss:
            self._beendet = True
            self._ende = time.monotonic()

    def melde(self, stoerung):
        """Eine Stoerung aufnehmen, jede Art aber nur einmal - sonst ertrinkt der Bericht."""
        with self._schloss:
            if stoerung not in self._stoerungen:
                self._stoerungen.append(stoerung)

    def hole_ab(self):
        """Quelle 2: die neuen History-Eintraege abholen - Dollar, Anbieter, Tokens und ob
        die Antwort abgeschnitten wurde.

        Abgeholt wird laufend und nicht erst am Ende: die History ist ein Ringpuffer mit
        10.000 Eintraegen (`clients/base_lm.py` Z. 786-800), und ein Optimierungslauf
        sprengt den. Wer erst am Schluss summiert, misst zu wenig.
        """
        ringpuffer_uebergelaufen = False
        with self._schloss:
            if self._beendet:
                return
            for stelle, sprachmodell in enumerate(self._sprachmodelle):
                eintraege = list(sprachmodell.history)
                schon_gezaehlt = self._gezaehlt[stelle]
                neue = [
                    eintrag for eintrag in eintraege if eintrag.get("uuid") not in schon_gezaehlt
                ]
                ringpuffer_uebergelaufen = ringpuffer_uebergelaufen or (
                    bool(schon_gezaehlt)
                    and len(neue) == len(eintraege)
                    and len(eintraege) >= dspy.settings.max_history_size
                )
                for eintrag in neue:
                    self._lm_aufrufe += 1
                    self._dollar += eintrag.get("cost") or 0.0
                    verbrauch = eintrag.get("usage") or {}
                    self._tokens += Tokens(
                        verbrauch.get("prompt_tokens") or 0,
                        verbrauch.get("completion_tokens") or 0,
                    )
                    antwort = eintrag.get("response")
                    anbieter = getattr(antwort, "provider", None)
                    name = str(anbieter) if anbieter else "unbekannt"
                    self._anbieter[name] = self._anbieter.get(name, 0) + 1
                    if _ist_abgeschnitten(antwort):
                        self._abgeschnitten[stelle] += 1
                self._gezaehlt[stelle] = {eintrag.get("uuid") for eintrag in eintraege}

        if ringpuffer_uebergelaufen:
            self.melde(
                "Der History-Ringpuffer war zwischen zwei Abholungen voll; "
                "moeglicherweise sind Aufrufe unbemerkt herausgefallen."
            )

    def messwerte(self):
        """Der Stand des Zaehlers, mit der Gegenprobe zwischen beiden Quellen."""
        self.hole_ab()
        with self._schloss:
            stoerungen = list(self._stoerungen)
            if self._fehlgeschlagene_programmlaeufe:
                stoerungen.append(
                    f"{self._fehlgeschlagene_programmlaeufe} von {self._programmlaeufe} "
                    "Programmlaeufen sind ohne Ergebnis geblieben und mit 0,0 bewertet worden."
                )
            if self._tokens_aus_evaluationen.gesamt > self._tokens.gesamt:
                stoerungen.append(
                    "Aus den Evaluationen sind mehr Tokens gemeldet "
                    f"({self._tokens_aus_evaluationen.gesamt}) als in der History stehen "
                    f"({self._tokens.gesamt}): der Ringpuffer hat Eintraege verloren, die "
                    "Dollar-Summe ist damit zu klein."
                )
            return Messwerte(
                dollar=self._dollar,
                lm_aufrufe=self._lm_aufrufe,
                programmlaeufe=self._programmlaeufe,
                fehlgeschlagene_programmlaeufe=self._fehlgeschlagene_programmlaeufe,
                formatfehler=self._formatfehler,
                anbieterfehler=self._anbieterfehler,
                auswertungen=self._auswertungen,
                tokens=self._tokens,
                tokens_aus_evaluationen=self._tokens_aus_evaluationen,
                anbieter=dict(sorted(self._anbieter.items())),
                abgeschnitten=tuple(self._abgeschnitten),
                stoerungen=tuple(stoerungen),
                dauer_sekunden=(self._ende or time.monotonic()) - self._beginn,
                latenzen=tuple(self._latenzen),
                fixkosten=self._fixkosten,
            )


def messhygiene():
    """Beide Cache-Schichten abschalten. Einmal am Anfang jedes Messskripts.

    Ein Cache-Treffer kostet nichts und meldet weder Tokens noch Kosten
    (`clients/cache.py` Z. 149-158). Der On-Disk-Cache ueberlebt Prozessneustarts: ohne
    diesen Aufruf zahlt das erste Modell einer Messreihe voll und jede Wiederholung nichts.
    """
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


def pruefe_messhygiene(protokollverzeichnis):
    """Vor jedem Messlauf: stehen die Bedingungen der Messhygiene, die er nicht selbst setzt?

    Jede von ihnen macht die Kostenzahlen still unbrauchbar statt laut kaputt - deshalb
    wird hier abgebrochen und nicht gewarnt. `provide_traceback` steht nicht auf der Liste:
    das setzt `messlauf` selbst, eine Pruefung darauf koennte nie fehlschlagen.
    """
    if dspy.cache.enable_disk_cache or dspy.cache.enable_memory_cache:
        raise ValueError(
            "Der Cache ist eingeschaltet: ein Treffer kostet nichts und wuerde die "
            "Dollar-Messung verfaelschen. Erst `messhygiene()` aufrufen."
        )
    if dspy.settings.send_stream is not None:
        raise ValueError(
            "Streaming ist eingeschaltet. Gemessen wird ohne Streaming (K6, Offener Punkt 3)."
        )
    if not pathlib.Path(protokollverzeichnis).is_dir():
        raise ValueError(
            f"Das Protokollverzeichnis {protokollverzeichnis} gibt es nicht. Es ist der "
            "einzige Absturzschutz von MIPROv2 (K6, Abschn. 3.5) und muss stehen."
        )


@contextlib.contextmanager
def messlauf(
    *sprachmodelle,
    protokollverzeichnis,
    abschnitt,
    einordnung,
    konfiguration,
    seed=None,
    plan=None,
    protokoll=None,
):
    """Der Abschnitt, dessen Kosten gezaehlt werden - mit Messhygiene und Protokollzeile.

    Uebergeben werden alle Sprachmodelle, die in diesem Abschnitt Geld kosten koennen: das
    Kandidatenmodell und, sobald MIPROv2 laeuft, das Modell des Proposers.

    Innerhalb des Abschnitts ist `track_usage` eingeschaltet, damit jedes Programm-Modul
    seinen Verbrauch an die zurueckgegebene Prediction haengt (Quelle 1), und
    `provide_traceback`, damit Fehler in der Evaluation sichtbar werden.

    Beim Verlassen entsteht **immer** genau eine Protokollzeile - auch wenn der Lauf mit
    einem Fehler abbricht. Gerade dann ist sie wertvoll: ausgegeben ist das Geld trotzdem,
    und rekonstruieren laesst es sich hinterher nicht. Sie traegt die Dauer des
    Abschnitts, auch eines abgebrochenen. Die Bewertung meldet der Aufrufer waehrend des
    Laufs mit `vermerke_bewertung`; die Latenz entsteht, wenn er das Programm vor dem
    Ausfuehren durch `mit_stoppuhr` reicht.

    `abschnitt` sagt, welcher Teil eines Laufs das ist (`optimierung`, `nachbewertung`,
    `testbewertung`, `fixkosten`, oder `None` fuer einen Handlauf oder eine Vorstudie, die
    keiner davon ist), `einordnung` wozu er lief (eine `dspy_optimizer.protokoll.Einordnung`).
    Beides wird geprueft, bevor der erste Aufruf Geld kostet.

    War der Abschnitt geplant, wird sein `plan` uebergeben (ein
    `dspy_optimizer.kostenmodell.Plan`). Dann steht in derselben Zeile, was er nach Plan
    kosten sollte und wie weit er danebenlag - das Verfahren ist budget-planend und kann
    einen laufenden Lauf nicht anhalten, die Abweichung ist deshalb selbst ein Messergebnis.
    Geplant sind nur Optimierung und Nachbewertung; ein Plan an einem anderen Abschnitt bricht
    vor dem ersten Aufruf ab.
    """
    if dspy.settings.usage_tracker is not None:
        raise ValueError(
            "Um diesen Abschnitt liegt ein `dspy.track_usage()`-Kontext. Der wird je "
            "Arbeits-Thread kopiert, und weil damit schon ein Tracker gesetzt ist, haengt "
            "kein Programm-Modul seinen Verbrauch mehr an die Prediction "
            "(primitives/module.py Z. 101-106) - Quelle 1 waere stumm (ADR-0012)."
        )
    # Fruehe Pruefung: ein unbekannter Abschnitt oder eine unvollstaendige Einordnung fiele
    # sonst erst auf, wenn der Lauf schon bezahlt ist und die Zeile geschrieben werden soll.
    pruefe_einordnung(abschnitt, einordnung)
    if plan is not None:
        # Ebenso frueh: geplant werden nur Optimierung und Nachbewertung.
        plan.geplant_fuer(abschnitt)

    zaehler = Kostenzaehler(*sprachmodelle)
    with dspy.context(
        callbacks=[*(dspy.settings.callbacks or []), zaehler],
        track_usage=True,
        provide_traceback=True,
    ):
        pruefe_messhygiene(protokollverzeichnis)
        try:
            yield zaehler
        finally:
            zaehler.beende()
            abbruch = sys.exc_info()[1]
            if abbruch is not None:
                zaehler.melde(f"Der Lauf ist abgebrochen: {abbruch!r}")
            (protokoll or Protokoll()).schreibe(
                abschnitt=abschnitt,
                einordnung=einordnung,
                konfiguration=konfiguration,
                seed=seed,
                bewertung=zaehler.bewertung,
                messwerte=zaehler.messwerte(),
                plan=plan,
                exakt=zaehler.exakt,
                absatztreffer=zaehler.absatztreffer,
                einzelbewertungen=zaehler.einzelbewertungen,
            )
