# AeroTune – Current State Summary

## Overview

AeroTune is a developing FPV drone tuning and analysis platform focused on Betaflight Blackbox data. It aims to provide conservative, evidence-based tuning recommendations and evolve into a full flight-performance analysis system.

The project is transitioning from a single-page analyzer into a multi-tool platform.

---

## Current Core Features

### 1. Tune Analysis (Primary Feature)
- Blackbox log upload (CSV + raw formats via converter)
- PID tuning recommendations (P, I, D, D Max, Feedforward)
- Focus on:
  - Bounceback detection
  - Propwash handling
  - Setpoint tracking
  - Throttle behavior
- Conservative delta-based recommendations (not full PID replacements)
- Motor heat awareness

---

### 2. Pilot Feel Integration
- User selects desired feel:
  - Locked-in
  - Cinematic
- Influences tuning suggestions
- Acts as early version of style-based tuning logic

---

### 3. Tune Change Tracking System
- Before vs After log comparison
- Tracks:
  - Improvements
  - Regressions
  - Pilot notes
- Generates reports for iteration
- Helps validate whether changes actually improved flight

---

### 4. Blackbox Processing
- Supports:
  - CSV logs
  - Raw .BBL / .BFL (with local tools)
- Multi-flight detection (in progress/refined)
- Parser reporting:
  - Detected columns
  - Sample rate
  - Duration
  - Data validity

---

### 5. Logging Strategy Awareness
- Supports multiple logging rates:
  - 1/16 (default)
  - 1/8 (advanced)
  - 1/4+ (high detail)
- Designed to work with both:
  - Hosted (limited)
  - Local (full capability)

---

## In Progress / Recently Added

### GPS Integration
- Hardware added to quad
- Betaflight configuration underway
- Future use:
  - Speed analysis
  - Efficiency metrics
  - Flight path context
  - Rescue validation

---

### Feedforward vs D Logic Refinement
- New principle added:

> Use feedforward for stick authority when motor heat risk is high instead of pushing D too far.

- System beginning to distinguish:
  - Damping problems (D)
  - Stick response problems (FF)

---

## Planned Major Features

### 1. Pilot Performance Analysis System
Separate from tune analysis.

Will include:
- 5 flying styles:
  - Cinematic
  - Smooth Freestyle
  - Aggressive Freestyle
  - Racing
  - Long Range / Cruising
- Metrics:
  - Throttle smoothness
  - Stick control
  - Recovery quality
  - Propwash handling
  - Efficiency (with GPS)

---

### 2. Site Redesign (Critical Next Step)

Current issue:
- Too many features on one page

Planned solution:
```text
Home Hub → Dedicated Tool Pages
```

New structure:
- Home (dashboard)
- Tune Analysis
- Pilot Performance
- Tune Change Tracking
- Blackbox Tools
- Advanced Diagnostics
- Docs

---

### 3. GPS-Based Analysis (Future)
- Ground speed correlation
- mAh per distance
- Efficiency scoring
- Wind/load inference
- High-speed instability detection

---

### 4. Advanced Diagnostics
- FFT / noise analysis
- Filter tuning guidance
- High-rate log analysis (1/4, 1/2)
- Developer-level metrics

---

## Development Environment

### Git Setup
- Dual remotes:
  - `origin` → personal repo (bostromdev)
  - `aerotunelabs` → org repo (private)
- Currently tracking:
```text
main → origin/main
```

---

### Deployment
- Hosted version (Render)
  - Limited file size
  - Slower
- Local version
  - Full capability
  - Required for large logs and raw files

---

## Strengths of Current Project

- Real-world tested (your own quad)
- Conservative tuning logic
- Focus on safety (motor heat, battery limits)
- Practical workflow:
  - log → analyze → adjust → compare
- Already ahead of typical “PID calculators”

---

## Current Limitations

- UI becoming crowded
- No clear separation of features yet
- GPS not fully integrated
- Performance analysis not implemented yet
- Some logic still heuristic (not fully data-driven)

---

## Next Immediate Priorities

1. Fix GPS UART wiring and confirm communication
2. Validate GPS data in Blackbox logs
3. Begin UI restructure (Home page + sections)
4. Implement Pilot Performance system (basic version)
5. Continue collecting structured logs

---

## Vision Direction

AeroTune is evolving toward:

> A full FPV flight analysis platform

Not just:
- PID tuning

But:
- flight behavior analysis
- efficiency tracking
- style-based tuning
- real-world performance modeling

---

## Key Philosophy

- Conservative changes > aggressive tuning
- Data-driven > guesswork
- Safety > style preferences
- Real-world testing > theory

---

## Summary

AeroTune is no longer just a tuning script.

It is becoming a structured system for:
- tuning
- validation
- pilot improvement
- performance analysis

The next major step is organizing the interface so this power remains usable.
