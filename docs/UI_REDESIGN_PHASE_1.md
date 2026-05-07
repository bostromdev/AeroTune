# AeroTune UI Redesign Phase 1

## Goal

Move AeroTune away from one overloaded page and into a scalable FPV engineering workstation layout.

## Implemented Structure

```text
Home Hub
├── Tune Analysis
├── Tune Change Tracking
├── Pilot Performance
├── Blackbox Tools
├── Advanced Diagnostics
└── Docs / Learn
```

## Implementation Choice

This phase uses plain HTML, CSS, and hash-based JavaScript routing inside `static/index.html`. That keeps the current FastAPI app and analyzer endpoints working without introducing React, build tooling, or unnecessary framework complexity.

## Preserved Behavior

- Existing form IDs and result IDs are preserved.
- Existing backend endpoints are preserved.
- PID advice rendering remains handled by `static/tuning_advice_patch.js`.
- Raw multi-flight selection remains handled by `static/blackbox_flight_selector.js`.
- Parser reports, converter reports, comparison machine output, and tuning machine output remain available behind expandable advanced panels.

## Why This Matters

AeroTune is evolving into a structured FPV analysis platform. Separating tools by intent makes the project easier to use now and easier to expand later for Pilot Performance, GPS analytics, FFT/noise diagnostics, and style-based interpretation.
