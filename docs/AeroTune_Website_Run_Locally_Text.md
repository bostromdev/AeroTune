# Website Run Locally Text

Use this wording on the hosted demo page:

AeroTune runs best locally because real Betaflight Blackbox logs can be large. The hosted site is mainly a demo and documentation page.

## Recommended CSV Workflow

For best results, open your `.BBL` file in Betaflight Blackbox Explorer, choose the correct flight, then export that selected flight as CSV.

If your raw log contains multiple flights, the newest flight is usually the highest number. For example:

```text
1/7 = first flight
7/7 = newest flight
```

Upload the exported CSV into AeroTune.

## Raw Log Support

Raw `.BBL`, `.BFL`, and `.TXT` logs can work locally if Betaflight `blackbox_decode` is installed. AeroTune converts the raw file to CSV first, then analyzes the decoded CSV.

If raw upload does not work, export CSV from Blackbox Explorer.

## macOS / Linux

```bash
git clone https://github.com/bostromdev/AeroTune.git
cd AeroTune
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Windows PowerShell

```powershell
git clone https://github.com/bostromdev/AeroTune.git
cd AeroTune
py -3 -m pip install -r requirements.txt
py -3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
