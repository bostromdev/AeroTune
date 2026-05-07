# AeroTune Release Notes

## Phase 1 Platform UI — Home Hub + Dedicated Tool Views

### Summary

This update reorganizes the frontend from one overloaded analyzer page into a scalable AeroTune workstation layout. The backend analyzer logic remains intact; the UI now separates workflows by intent.

### What Changed

- Added a Home Hub entry screen with clear cards for Tune Analysis, Tune Change Tracking, Pilot Performance, Blackbox Tools, Advanced Diagnostics, and Docs.
- Added hash-routed tool views using plain JavaScript instead of adding a heavy frontend framework.
- Moved the normal upload/recommendation workflow into a dedicated Tune Analysis page.
- Moved raw multi-flight tune-change validation into a dedicated Tune Tracking page.
- Moved converter, optimizer, parser, and file-prep tools into Blackbox Tools.
- Added a Pilot Performance planning page with the five intended style profiles.
- Added an Advanced Diagnostics planning page for future FFT/noise, GPS, and filter tooling.
- Preserved parser reports, machine output, comparison output, and other engineering details inside expandable advanced panels.
- Stopped the old UI cleanup script from hiding cards that are now intentionally organized into separate views.
- Mounted the local `docs/` folder at `/docs` so the Docs view can open included markdown guides while the app is running.

### Design Reason

AeroTune is becoming a full FPV flight-analysis platform, not a generic PID calculator. This layout keeps beginners comfortable while keeping advanced diagnostics available for serious tuning work.

---

## V1.5.4 — Tune-Change Tracking Workflow + Pilot/Data Verdicts

### Summary

This update refines AeroTune’s tune-change tracking workflow so the raw Blackbox multi-flight path feels like one clean process instead of separate disconnected steps.

The main goal of this release is to make tune tracking more honest, less confusing, and more useful when the data is mixed but the quad clearly flies well.

### What Changed

- Removed the redundant bottom “before tune log / after tune log” file chooser from the Tune-Change Tracking card.
- The raw log workflow now focuses on:
  1. Decode raw log
  2. Choose before/after flights
  3. Enter tune-change details
  4. Track selected raw flights
- Moved the “Track Selected Raw Flights” button to the bottom of the Tune Change Details section.
- After decoding and selecting raw flights, the page now scrolls toward Tune Change Details instead of forcing the user back to the analyzer area.
- Fixed the analyzer raw-flight selector so it no longer steals scroll focus after tune-tracking raw log preparation.
- Kept the backend `/track-tune-change` endpoint for compatibility, but the UI now prioritizes the selected raw multi-flight workflow.
- Added a Pilot + Data Read section to better explain what AeroTune thinks happened.
- Added after-flight outcome options:
  - After flight felt better
  - After flight felt locked-in / controlled
  - Use this as current best tune
  - Motors hot/warm but holdable
  - Low-throttle hover band feels narrow
  - After flight felt worse

### Verdict Wording Improvements

AeroTune now uses more realistic wording when comparison confidence is low or when flights may not be similar enough.

Instead of overstating results as confirmed propwash or failed tuning, it now uses softer and more accurate language such as:

- Mixed but usable baseline
- Pilot feel supports keeping the current tune
- Do not stack Locked-In/Cinematic yet; validate same route first
- Propwash-like recovery signature

This matters because a Blackbox log can show propwash-like or loaded-vibration signatures even when the quad feels good in the air, especially on larger GoPro-loaded builds.

### Why This Update Matters

Tune tracking should not blindly override pilot feel, but it also should not ignore the data.

This update makes AeroTune better at saying:

- The quad may be flying well.
- The comparison data may still be inconclusive.
- Roll/pitch/yaw may not all improve equally.
- Propwash-like signatures may also come from throttle-load vibration, GoPro/frame resonance, prop condition, or route mismatch.
- The safest next step may be to keep the tune and repeat a cleaner validation flight instead of stacking more PID changes.

### Recommended Use

For best results:

1. Use the raw Blackbox workflow.
2. Decode the raw log.
3. Select matching before/after flights.
4. Enter the tune-change details.
5. Track selected raw flights.
6. Use pilot feel, motor temperature, and the data read together.
7. Do not apply aggressive style tuning if AeroTune says the baseline is still mixed or validation confidence is low.

### Notes

This release is focused on workflow, UI behavior, and verdict clarity. It does not turn inconclusive data into fake certainty. It makes the output more honest and more useful for real-world FPV tuning.


## AeroTune v1.8.2 - Baseline Hold Priority Fix

### Fixes

- Baseline hold no longer hides advice when the pilot also selects active tuning symptoms.
- Symptoms like loose/floaty feel, twitchiness, bounceback, propwash, weak roll/pitch, delayed tracking, high-throttle wobble, or throttle-punch oscillation bypass the baseline hold.
- Low-throttle climb/fall still gets throttle/hover-point guidance first because it is usually not a PID problem.
- The PID advice card now shows when baseline hold was bypassed by selected symptoms.

---

## AeroTune v1.8.1 - Good Baseline Hold Hotfix

### Fixes

- Added **Use this as a good baseline / hold current tune** as a Pilot Flight Feel option.
- Good/cool baseline flights now suppress PID deltas instead of showing aggressive D/D Max advice from a single log.
- The PID advice card now shows a baseline-hold message when pilot feel and one-log metrics disagree.
- AeroTune still preserves the log evidence, but asks for another comparable flight before recommending a PID correction.

---

## AeroTune v1.8 - Pilot Flight Feel + Dynamic Notch Filtering Advisor

### Major Updates

- Added Pilot Flight Feel checkbox options to the single-log upload form.
- Backend now passes `pilot_feel` into the analyzer and selected raw-flight reanalysis endpoint.
- Added `app/pilot_feel.py` as the human-context layer for checkbox normalization, filtering policy, D-term safety gates, and real-world tuning-plan generation.
- Dynamic Notch is now the default filtering recommendation path.
- RPM filtering is treated as advanced-only and is suppressed when the pilot reports RPM-filter issues, arming warnings, or failsafe trouble.
- Hot/buzzy motor reports block positive D/D Max advice for that report.
- Good flight + cool motors + no severe repeated log issue activates good-flight protection so AeroTune recommends holding the tune or making only one tiny change.
- UI now shows Pilot Feel Input, Filtering / D-Term Safety, and Real Tuning Plan cards in the useful summary.
- Raw multi-flight selector keeps pilot-feel checkbox values when analyzing a different decoded flight.
- Added inline code labels documenting the engine, upload wiring, filtering advisor, and PID safety gates.

### Why It Matters

AeroTune now combines Blackbox evidence with what the pilot actually felt. That prevents blind graph-chasing and creates safer advice: Dynamic Notch first, mechanical inspection when needed, D-term caution on large props, and no aggressive PID changes after a good/cool flight.

---

## AeroTune v1.7 - Raw Blackbox Multi-Flight Selector

### Major Updates

- Added raw Blackbox multi-flight selector for `.BBL`, `.BFL`, and `.TXT` uploads when `blackbox_decode` is available locally.
- AeroTune now keeps every decoded flight CSV from a raw Blackbox file instead of silently choosing one.
- Added UI selector for `Flight 1/N` through `Flight N/N`.
- Treats `Flight 1/N` as the oldest detected flight and `Flight N/N` as the newest/latest detected flight.
- Selects the newest/latest flight by default.
- Lets users analyze another decoded flight without re-uploading the raw log.
- Updated website/helper wording so users know raw `.BBL` upload is OK after local converter setup, while CSV remains the safest fallback.

### Notes

Raw Blackbox support still depends on Betaflight `blackbox_decode`. AeroTune does not replace the decoder; it uses the decoded CSV output for analysis.

---


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

---

## AeroTune v1.9 - Raw Flight Pair Tune Tracking + Structured Change Details

### Major Updates

- Added single raw-log tune-change tracking for multi-flight `.BBL`, `.BFL`, and `.TXT` logs.
- Added automatic latest-two-flight selection: `N-1/N` as before and `N/N` as after.
- Added manual before/after selectors plus latest-two and swap controls.
- Kept the old two-file before/after workflow.
- Added structured Tune Change Details checkboxes so pilots can record PID, filtering, rate, hardware, and test-condition changes without relying only on free text.
- Added a tune-change context layer in `app/tune_change_options.py`.
- Added output for every selected pilot-feel checkbox so each feel note creates a visible interpretation/action.
- Added final PID delta guardrails: AeroTune caps Betaflight percent advice to ±11% per test pass and prevents checkbox context from stacking into oversized PID jumps.

### Safety Notes

Tune-change options and pilot-feel options are context/gates. They explain the report and modify safety logic, but they do not directly add together into PID percentages. The before/after logs still decide whether a tune change helped, hurt, or needs retesting.

