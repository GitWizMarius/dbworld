# DBWorld Studienarbeit - Python Code

Code written by: Marius Armbruster

Betreuer: Prof. Dr.-Ing. Olaf Herden

Python Version: 3.8

# 1. Anleitung
## 1.1. Erste Schritte
- Installieren Sie Python 3.8 (https://www.python.org/downloads/)
- Downloaden und entpacken Sie das Repository
- Navigieren Sie in das entpackte Repository und öffnen sie eine Konsole
- Installieren Sie anhand des Befehls 'pip install -r requirements.txt' die benötigten Pakete
- Manuelle installation von PKE -> sie benötigen dazu 'git' (https://git-scm.com/downloads)
  - Befehl: 'pip install git+https://github.com/boudinfl/pke.git'

## 1.2. Datenbank Verbindung
- .env Datei in das Hauptverzeichnis des Repositories kopieren oder eine neue erstellen
- Dort folgende Daten eintragen bzw. ergänzen (falls .env bereits vorhanden aber neue DB):
  - tbd
- Benötigte Datenbank: PostgreSQL (db.py -> basiert auf der Voraussetzung, dass eine PostgreSQL Datenbank existiert und die DB + Tabellen bereits angelegt sind)

## 1.3 Datensatz
Für den Datenimport wird eine .csv Datei benötigt, welche einen Export der Mails beinhaltet.
Bei dem Export der Mail handelt es sich dabei um einen Outlook-Export mit unten aufgelisteten Attributen.
Dieser Datensatz ist aus Speichertechnischen Gründen nicht in dem Repository zu finden. 

Download über: [tbd]

Eigener db-World Mail Datensatz nutzen? 
- Outlook Downloaden
- Mit Mail Konto anmelden
- Für Export empfehlt sich folgendes Addon:
  - https://www.codetwo.de/freeware/outlook-export/

## 1.4 Ausführen des Skripts
- Öffnen Sie die Konsole und navigieren Sie in das Repository
- Befehl: 'python3 main.py'
- Das Skript wird automatisch ausgeführt
- Nun sollte sich eine GUI öffnen, anhand welcher die .csv Datei ausgewählt werden kann.
  - Anmerkung: Der erste Start des Skripts kann länger dauern, da einige benötigte Datein heruntergeladen werden. Sollte in der Konsole länger keine neue Meldung kommen, so starten Sie das Skript neu und warten bis die GUI kommt (ggf. in dieser Zeit nichts anderes anklicken, dies führte in selten Fällen zum nicht öffnen des Fensters)
- Danach sollte in der Konsole mehrere Informationen ausgegeben werden sowie eine Statusleiste zur aktuellen Bearbeitungszeit/Fortschritt
- Wichtig: Die Konsolen gibt lediglich eine Statusanzeige aus. Alle Konsolen Ausgaben in Form von print Befehlen wurden auskommentiert bzw. gelöscht, da nicht notwendig bzw. teils unübersichtlich.

## 1.5 Nutzen der Benchmarks
[tbd]


## 1.6. Fehlerbehebung und Fehlerbehandlung
- Falls 'en-core-web-sm' oder 'en-core-web-trf' einen Fehler werfen folgendes ausführen
  - Befehl: 'python3 -m spacy download en_core_web_sm'
  - Befehl: 'python3 -m spacy download en_core_web_trf'
- Falls PyTorch Probleme auftreten, PyyTorch deinstallieren und folgendes ausführen
  - Befehl: pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

  

