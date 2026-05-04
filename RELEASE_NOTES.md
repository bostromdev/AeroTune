# AeroTune Release Notes

## AeroTune v1.6.1 - CSV-First Docs + Raw Multi-Flight Planning

### Documentation Updates

- Clarified that AeroTune is CSV-first for beginner reliability.
- Clarified that raw `.BBL`, `.BFL`, and `.TXT` support works locally only when `blackbox_decode` is installed and available.
- Added the key CSV workflow reason: Blackbox Explorer lets the pilot select the correct/latest flight before exporting CSV.
- Added warning that raw `.BBL` files may contain multiple flights.
- Added planned raw multi-flight selector behavior:

```text
Flight 1/7
Flight 2/7
...
Flight 7/7 = newest/default
```

### Current Recommendation

Use CSV export from Betaflight Blackbox Explorer when accuracy matters, especially when a raw `.BBL` contains multiple flights. Select the correct flight first, then export CSV.

### Future Feature Direction

AeroTune can later add backend support to decode all flights from a raw `.BBL`, generate a tuning-advisor profile for each flight, and default the UI to the highest-numbered/latest flight.

---

## AeroTune v1.6 - PID Tuning Advisor + Creator Card

### Major Updates

- Added PID Tuning Advice card.
- Added Betaflight PID value calculator.
- Added creator / BostromDev YouTube card.
- Cleaned public UI while keeping tune tracking available.
- Improved local setup docs for Windows users.

---

## AeroTune v1.5.1 - Saved Tune-Change Reports + Updated Guides

### Major Updates

- Added saved tune-change reports as JSON and Markdown.
- Added report storage under `reports/tune_changes/`.
- Added documentation for the V1.5 tune-change tracking workflow.
- Updated the install/run guide for local-first use, raw-log conversion, saved reports, and privacy notes.
- Updated the PID tuning guide around clean baseline first, style tuning second, one change at a time, and keep/reduce/revert/retest decisions.

---

## AeroTune v1.5 - Tune-Change Tracking

AeroTune can compare before/after logs, record what changed, and help decide whether to keep, reduce, revert, or retest a tuning change.

---

## AeroTune v1.4 - Multi-Log Comparison

Added before/after comparison metrics for tracking, noise, propwash behavior, and lag.

---

## AeroTune v1.3 - Raw Blackbox Converter Path

Added optional local raw-log conversion through `blackbox_decode`.

---

## AeroTune v1.0 - Initial Public Release

Initial Blackbox CSV analysis, conservative tuning recommendations, size profiles, and dark telemetry-style UI.
