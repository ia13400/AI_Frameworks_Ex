# Assignment: Übung 1 -- Environment Setup & erstes Notebook
# Block 1 | 50 Minuten | Einzelarbeit

---

## Lernziel
Nach dieser Übung könnt ihr ein modernes Python-Projekt mit `uv`, `pyproject.toml` und Virtual Environment aufsetzen -- übertragbar auf jedes KI-Projekt.

## Voraussetzungen
- Python 3.11+ installiert
- Git installiert
- Terminal/Kommandozeile zugänglich

---

## Aufgabe A: Projektsetup mit uv (15 Min)

### A.1 uv installieren (falls nicht vorhanden)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Prüfe: `uv --version`

### A.2 Projekt anlegen
```bash
mkdir mechinterp-ss2026
cd mechinterp-ss2026
uv init
uv venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
```

### A.3 Dependencies installieren
```bash
uv add pandas polars numpy matplotlib seaborn scikit-learn jupyter torch transformers
```

### A.4 Prüfe die Installation
Starte JupyterLab (`jupyter lab`) und öffne das Notebook `notebooks/01_setup_check.ipynb` (wird bereitgestellt). Führe alle Zellen aus -- alle Imports müssen fehlerfrei durchlaufen.

**Erwartetes Ergebnis**: Alle Bibliotheken mit Versionsnummern werden angezeigt.

---

## Aufgabe B: Git-Workflow mit eigener .gitignore (10 Min)

### B.1 Repository initialisieren
```bash
git init
```

### B.2 .gitignore selbst erstellen
Erstelle eine Datei `.gitignore` **ohne die Lösung unten zu lesen**. Überlege:
- Welche Dateien und Ordner gehören **nicht** in ein Git-Repository?
- Denke an: Virtual Environments, Python-Cache, Modell-Dateien, temporäre Daten

Schreibe deine `.gitignore` und vergleiche danach mit der Musterlösung:

<details>
<summary>Musterlösung aufklappen</summary>

```
.venv/
__pycache__/
*.pyc
*.pkl
*.pt
*.onnx
mlruns/
data/processed/
.ipynb_checkpoints/
```
</details>

### B.3 Erster Commit
```bash
git add .
git commit -m "Initial project setup with uv"
```

---

## Aufgabe C: Google Colab als Fallback (5 Min)

Öffne https://colab.research.google.com, lade `01_setup_check.ipynb` hoch und führe alle Zellen aus.

Diskutiere kurz mit deinem Nachbarn: In welchen Situationen würdet ihr lokal arbeiten, in welchen in Colab?

---

## Aufgabe D: Dependency-Management live erleben (15 Min)

In dieser Aufgabe erlebt ihr den uv-Workflow **hands-on**.

### D.1 Ein Package hinzufügen
```bash
uv add requests
```
Öffne `pyproject.toml` und beobachte: Was hat sich geändert?

### D.2 Das Package nutzen
Erstelle eine Datei `test_requests.py` mit folgendem Inhalt und führe sie aus:
```python
import requests
r = requests.get("https://httpbin.org/json")
print(r.status_code)
print(r.json()["slideshow"]["title"])
```

### D.3 Lock-Datei erzeugen und verstehen
```bash
uv lock
```
Öffne `uv.lock` und beantworte:
- Wie viele Packages stehen darin (nicht nur `requests`)?
- Warum sind es mehr als du installiert hast?
- Was passiert, wenn ein Teamkollege `uv sync` ausführt?

### D.4 Package wieder entfernen
```bash
uv remove requests
rm test_requests.py
```
Prüfe erneut `pyproject.toml`: Ist `requests` verschwunden?

### D.5 Reflexion
Notiere in 2-3 Sätzen: Was ist der Unterschied zwischen `pyproject.toml` und `uv.lock`? Warum braucht man beides?

---

## Aufgabe E: Reflexion (5 Min)

Notiere dir kurz:
- Was war neu für dich?
- Was würdest du in deinem nächsten Projekt anders machen als bisher?

---

## Abgabekriterien

| Kriterium | Erfüllt? |
|-----------|----------|
| Projekt mit uv erstellt, `.venv` aktiv | [ ] |
| Alle Imports in `01_setup_check.ipynb` erfolgreich | [ ] |
| `.gitignore` selbst erstellt, erster Commit gemacht | [ ] |
| Dependency-Zyklus (add → lock → remove) durchgeführt | [ ] |
| Reflexion zu pyproject.toml vs. uv.lock notiert | [ ] |
