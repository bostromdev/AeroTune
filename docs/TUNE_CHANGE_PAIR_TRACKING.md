# AeroTune Tune-Change Pair Tracking

## Purpose

Tune-change tracking answers one practical question:

```text
Did the change I just made help, hurt, or need another test?
```

AeroTune supports two workflows.

## Workflow A: one raw Blackbox file with multiple flights

Upload one raw `.BBL`, `.BFL`, or `.TXT` file. AeroTune uses `blackbox_decode` locally, finds the decoded flights, and shows them as `Flight 1/N` through `Flight N/N`.

Default pair selection:

```text
2 flights total  -> before = Flight 1/2, after = Flight 2/2
6 flights total  -> before = Flight 5/6, after = Flight 6/6
```

The pilot can change either selector before comparing.

## Workflow B: two separate logs

Upload one before log and one after log. Each can be CSV or raw Blackbox if the converter is installed.

## Tune Change Details

The structured checklist records what changed between flights:

```text
PID changes
Filtering changes
Rates / feel changes
Hardware / setup changes
Test condition changes
```

These checkboxes are not direct PID commands. They are context for interpreting the before/after comparison. For example, if the pilot changed props, battery, route, or GoPro weight, comparison confidence is reduced because the after log is not perfectly isolated.

## PID delta guardrail

AeroTune caps final Betaflight percentage advice to ±11% per pass. Pilot-feel options and tune-change options can change gates, warnings, confidence, and next steps, but they do not add together into a +50% or +100% PID jump.

## Code map

```text
app/main.py
  API routing, raw flight-pair prepare endpoint, selected-pair tracking endpoint.

app/converter.py
  Decodes raw Blackbox logs, stores each generated flight CSV, and labels Flight 1/N through Flight N/N.

app/tune_change_options.py
  Structured tune-change checkbox catalog and normalizer.

app/tune_tracking.py
  Combines free-text notes, structured options, comparison metrics, and pair selection into keep/reduce/revert/retest decisions.

static/index.html
  UI for raw flight-pair selection, structured tune-change options, and legacy two-log tracking.
```
