# Assignment: Übung 2 -- Explorative Datenanalyse
# Block 1 | 50 Minuten | Einzelarbeit

---

## Lernziel
EDA-Workflow mit Pandas und Polars **selbst durchführen** -- anwendbar auf jeden tabellarischen Datensatz.

## Bereitgestellte Daten
- Datensatz: `data/raw/listings_berlin.csv` (Berlin Airbnb Listings, ~20.000 Zeilen)
- Notebook-Template: `notebooks/02_eda.ipynb`

---

## Aufgabe A: Geführte EDA mit Pandas (15 Min)

Öffne `notebooks/02_eda.ipynb`. Die ersten Zellen (Daten laden, fehlende Werte, `describe()`) sind als Demonstration vorbereitet. **Führe sie aus und verstehe den Output.**

Beantworte im Notebook:
- Wie viele Zeilen und Spalten hat der Datensatz?
- Welche Spalten haben mehr als 5% fehlende Werte?
- Gibt es Duplikate?

---

## Aufgabe B: Eigenständige Analyse-Aufgaben (20 Min)

Ab hier schreibst du **eigenen Code**. Im Notebook findest du leere Code-Zellen mit Aufgabenbeschreibungen. Hints sind verfügbar, falls du nicht weiterkommst.

### B.1 Preisspalte bereinigen
Die Spalte `price` enthält Dollar-Zeichen (`$1,250.00`). Bereinige sie so, dass du damit rechnen kannst.

### B.2 Top-5 Bezirke nach Median-Preis
Finde die 5 teuersten Bezirke (`neighbourhood_group_cleansed`) nach Median-Preis. Stelle das Ergebnis als **horizontales Balkendiagramm** dar.

### B.3 Eigene Visualisierung
Formuliere eine Frage, die dich an den Daten interessiert, und beantworte sie mit einer selbst erstellten Visualisierung. Die Frage ist der Titel des Plots.

### B.4 Filtern und quantifizieren
Filtere alle Listings heraus, die einen Preis über 500€ **und** weniger als 10 Reviews haben. Wie viel Prozent der Daten sind das? Was könnte der Grund für diese Kombination sein?

---

## Aufgabe C: Polars-Vergleich (10 Min)

### C.1 Aufgabe B.2 mit Polars lösen
Löse die Bezirks-Analyse (B.2) nochmal mit Polars statt Pandas.

### C.2 Performance messen
Miss die Ausführungszeit beider Varianten mit `time.time()`:
- Pandas: __ ms
- Polars: __ ms
- Faktor: __x schneller/langsamer

### C.3 Fazit
Fülle die Vergleichstabelle im Notebook aus:

| Aspekt | Pandas | Polars |
|--------|--------|--------|
| Syntax-Gefühl | | |
| Gemessene Performance | | |
| Ökosystem-Vorteil | | |
| Ich würde es nutzen wenn... | | |

---

## Aufgabe D: Plenum-Vorbereitung (5 Min)

Bereite eine **1-Minuten-Präsentation** vor:
- Zeige EINE Visualisierung (bevorzugt deine eigene aus B.3)
- Erkläre die wichtigste Erkenntnis daraus
- Nenne eine Überraschung aus den Daten

---

## Abgabekriterien

| Kriterium | Erfüllt? |
|-----------|----------|
| Geführte EDA durchgearbeitet, Fragen beantwortet | [ ] |
| Preisspalte selbständig bereinigt | [ ] |
| Top-5 Bezirke mit eigenem Code + Balkendiagramm | [ ] |
| Eigene Visualisierung mit eigener Fragestellung | [ ] |
| Filter-Aufgabe gelöst und interpretiert | [ ] |
| Polars-Vergleich durchgeführt, Performance gemessen | [ ] |
| Vergleichstabelle ausgefüllt | [ ] |
