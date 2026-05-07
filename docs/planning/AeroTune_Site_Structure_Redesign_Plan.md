# AeroTune Site Structure Redesign Plan

## Purpose

AeroTune is growing from a single-page Blackbox analyzer into a larger tuning, flight-behavior, and performance-analysis platform.

The current site already has several powerful features. Adding more systems directly onto the same screen will eventually make the interface overwhelming, especially for newer pilots.

The goal of this redesign is:

> Keep AeroTune powerful, but make it easier to use by separating features into clean, dedicated pages or tool areas.

This document defines a deeper long-term structure for AeroTune so future features can be added without turning the site into one giant confusing control panel.

---

# Core Design Principle

AeroTune should not remove advanced features.

Instead, AeroTune should separate features by user intent.

The user should not see every possible tool immediately.

The site should guide users through clear choices:

- I want to analyze a tune.
- I want to compare before/after logs.
- I want to evaluate my flying style.
- I want to optimize or convert a Blackbox file.
- I want advanced GPS/noise/filter diagnostics.
- I want documentation and setup help.

This creates a platform-style layout instead of a single overloaded tool page.

---

# Main Problem With The Current Layout

AeroTune currently risks becoming:

> Everything everywhere at once.

That made sense during early development because fast iteration was more important than perfect layout.

But as AeroTune adds more systems, one page becomes difficult to use.

Features already involved or planned include:

- Blackbox upload
- Raw .BBL / .BFL support
- CSV conversion
- Multi-flight selector
- PID tune advisor
- Pilot feel inputs
- Tune-change tracking
- Before/after comparison
- Parser reports
- CSV optimizer
- Logging-rate recommendations
- Motor heat warnings
- Feedforward guidance
- Filtering guidance
- GPS telemetry
- GPS-based performance analysis
- Pilot performance scoring
- Five flying-style profiles
- Long-range/cruising efficiency analysis
- Advanced developer diagnostics

All of that should not live on one main screen.

The solution is not to make AeroTune smaller.

The solution is:

> Better information architecture.

---

# Recommended High-Level Structure

AeroTune should eventually use a Home Hub with dedicated tool pages.

Recommended top-level structure:

1. Home / Dashboard
2. Analyze Tune
3. Pilot Performance
4. Tune Change Tracking
5. Blackbox Tools
6. Advanced Diagnostics
7. Docs / Learn
8. Settings / Profile

The first version does not need to fully implement all of these pages immediately.

A good first step is to create the Home Hub and split the existing tools into logical sections.

---

# 1. Home Page / Dashboard

## Purpose

The home page should be a clean entry point.

It should help users choose what they want to do without overwhelming them.

It should not show every technical control immediately.

Think of it as:

> The front door to AeroTune.

## Homepage Goals

The home page should:

- Explain what AeroTune does in one short sentence.
- Show large cards for major tools.
- Keep beginner users from feeling lost.
- Let advanced users jump quickly to the right feature.
- Make the site feel more professional.
- Make future features easier to add.

## Suggested Homepage Headline

A possible headline:

> Tune smarter with Betaflight Blackbox data.

## Suggested Homepage Subheading

A possible subheading:

> Upload flight logs, compare tune changes, analyze pilot behavior, and improve your quad with conservative evidence-based recommendations.

## Recommended Homepage Cards

### Card 1: Analyze Tune

Purpose:

> Get PID, feedforward, filtering, and motor-heat-aware tuning guidance from a Blackbox log.

Button:

> Start Tune Analysis

Best for:

- First-time users
- Normal tune work
- PID delta recommendations
- Filtering and feedforward advice

---

### Card 2: Pilot Performance

Purpose:

> Analyze how your flying matches a selected style such as Cinematic, Smooth Freestyle, Aggressive Freestyle, Racing, or Long Range.

Button:

> Analyze Pilot Style

Best for:

- Style scoring
- Throttle smoothness
- Recovery control
- GPS speed behavior
- Efficiency analysis

---

### Card 3: Tune Change Tracking

Purpose:

> Compare before/after logs and track whether a tune actually improved.

Button:

> Compare Tune Changes

Best for:

- Baseline vs updated tune
- Did it feel better?
- Tracking improvement over time
- Saved change reports

---

### Card 4: Blackbox Tools

Purpose:

> Convert, inspect, optimize, and select flights from Blackbox files.

Button:

> Open Blackbox Tools

Best for:

- Raw .BBL / .BFL handling
- Multi-flight selection
- CSV optimization
- Parser diagnostics
- File prep before analysis

---

### Card 5: Advanced Diagnostics

Purpose:

> Deep analysis tools for noise, filters, GPS, high-rate logging, and developer testing.

Button:

> Open Advanced Tools

Best for:

- 1/4 or 1/2 logging-rate analysis
- FFT/noise behavior
- GPS data review
- Filter testing
- Developer-mode outputs

---

### Card 6: Docs / Learn

Purpose:

> Learn how to record useful logs, prepare files, understand recommendations, and tune safely.

Button:

> Read Docs

Best for:

- New users
- Blackbox setup
- Logging-rate guidance
- Recommended test flights
- Local install instructions

---

# Homepage UI Rules

The homepage should avoid technical overload.

Do show:

- Short feature cards
- Clear buttons
- Simple descriptions
- A short note about hosted vs local analysis

Do not show:

- Full parser report
- PID inputs
- Advanced filter details
- Raw debug data
- Large JSON outputs
- Multiple upload sections at once

The homepage should feel calm and confident.

---

# 2. Analyze Tune Page

## Purpose

This is the main AeroTune tuning page.

It should focus on answering:

> What should I change in my tune?

This page should not try to score pilot performance as its primary job.

It can use pilot feel and style as inputs, but the output should remain tune-focused.

## Primary Features

The Analyze Tune page should include:

- Log upload
- Flight selector if raw Blackbox has multiple flights
- Drone size selection
- Current PID values
- Current feedforward values if supported
- Pilot feel selection
- Optional motor temperature input
- Optional battery/mAh notes
- Main tune recommendation output
- Evidence summary
- Safety warnings
- Parser report collapsed by default

## Recommended Page Flow

### Step 1: Upload Log

User uploads:

- .BBL
- .BFL
- .CSV
- .TXT if supported

The page should clearly say:

> Raw Betaflight Blackbox files are supported when local tools are installed. CSV exports are also supported.

For hosted Render site:

> Large logs may fail on the hosted demo. Run locally for large files or high-rate logs.

### Step 2: Select Flight

If the log contains multiple flights:

- show flight list
- default to newest or largest valid flight
- allow user to choose

This should appear near the top, not buried below results.

### Step 3: Enter Setup Context

Inputs:

- Drone size: 3, 3.5, 4, 5, 6, 7 inch
- Prop size
- Motor size / KV if known
- Battery size
- Build type
- Weight if known
- Logging rate

These should be optional, but useful.

AeroTune should still run without every field.

### Step 4: Current Tune Values

Inputs:

- Roll P/I/D/D Max/FF
- Pitch P/I/D/D Max/FF
- Yaw P/I/FF
- Optional throttle expo/mid
- Optional feedforward smoothing/boost settings

This lets AeroTune show practical delta recommendations.

### Step 5: Analyze

Primary output should be easy to read:

- Recommended changes
- Confidence
- Why
- Safety notes
- What to test next

## Output Structure

A good tune output should use cards:

### Card: Main Recommendation

Example:

> Increase Roll/Pitch D slightly and use feedforward for stick authority. Avoid pushing D Max too high because motor heat risk is elevated.

### Card: Suggested Values

Example:

- Roll D: 38 -> 40
- Roll D Max: 51 -> 54
- Roll FF: 106 -> 112
- Pitch D: 42 -> 44
- Pitch D Max: 57 -> 58
- Pitch FF: 110 -> 116

### Card: Evidence

Example:

- Bounceback detected after throttle unload.
- Recovery is slightly floaty.
- No severe high-frequency noise detected.
- Motor heat risk reported by pilot.

### Card: Safety Limits

Example:

- Do not raise D further if motors are uncomfortable to touch.
- Land immediately if mAh approaches known collapse range.
- Avoid stacking filters to hide excessive D gain.

### Card: Next Test Flight

Example:

- 60-90 second controlled test
- throttle punches
- propwash descent
- flips/rolls
- stick release recovery
- motor temp check immediately after landing

## What Should Be Hidden By Default

The following should be collapsed behind an Advanced Details button:

- Full parser report
- Raw metrics
- FFT band values
- column detection
- sample-rate details
- internal scoring values
- JSON/debug-like data

This keeps the normal user experience clean.

---

# 3. Pilot Performance Page

## Purpose

This page should answer:

> How well did the pilot fly according to the selected style?

It should not primarily recommend PID values.

It should analyze flight behavior.

## Style Selection

The user selects one of five styles:

1. Cinematic
2. Smooth Freestyle
3. Aggressive Freestyle
4. Racing
5. Long Range / Cruising

The selected style changes the scoring logic.

## Key Concept

The same behavior can be good or bad depending on the selected style.

Example:

- Smooth throttle is excellent for Cinematic.
- Aggressive throttle transitions are expected in Aggressive Freestyle.
- High-speed precision matters more in Racing.
- Efficiency matters more in Long Range.

## Recommended Pilot Performance Metrics

### Throttle Smoothness Score

Measures:

- throttle jerk
- abrupt throttle cuts
- smoothness of throttle changes
- consistency through turns/descent

Useful for:

- Cinematic
- Smooth Freestyle
- Long Range

### Stick Smoothness Score

Measures:

- roll/pitch/yaw input smoothness
- unnecessary twitching
- stick noise
- overcorrection

Useful for:

- Cinematic
- Smooth Freestyle
- Racing

### Recovery Control Score

Measures:

- how cleanly quad settles after flips/rolls
- stick release stability
- bounceback
- attitude recovery

Useful for:

- Smooth Freestyle
- Aggressive Freestyle
- Racing

### Propwash Handling Score

Measures:

- instability during descents
- oscillation after throttle unload
- recovery during dirty-air situations

Useful for:

- Smooth Freestyle
- Aggressive Freestyle
- Long Range safety

### Efficiency Score

Requires current and preferably GPS.

Measures:

- mAh per distance
- throttle vs ground speed
- cruise efficiency
- battery sag behavior

Useful for:

- Long Range
- Cinematic
- general build performance

### Style Match Score

A single score that estimates:

> How well did this flight match the selected style?

This should be explained clearly, not treated as a harsh judgment.

## Recommended Output

### Summary Card

Example:

> Style Match: Smooth Freestyle - 78%

### Strengths Card

Example:

- Good flowing movement
- Controlled throttle through mid-speed sections
- Clean roll recovery

### Improvement Card

Example:

- Throttle unloads caused wobble
- Some overcorrection after reversals
- Recovery could be more locked in

### Pilot Practice Suggestions

Example:

- Practice throttle-supported descents instead of full throttle chops.
- Use smoother exit throttle after flips.
- Repeat the same test line for better tune comparison.

### Tune vs Pilot Separation

This page should clearly separate:

- pilot behavior issues
- tune issues
- battery/setup issues

Example:

> Some wobble appears during throttle unload. This may be partly pilot throttle behavior and partly underdamping. Review tune analysis before changing large PID values.

This is important so AeroTune does not blame the pilot for a bad tune or blame the tune for pilot input.

---

# 4. Tune Change Tracking Page

## Purpose

This page should answer:

> Did the tune change actually improve the quad?

This is one of AeroTune's most important long-term features.

## Inputs

- Before log
- After log
- Before PID values
- After PID values
- Pilot notes
- Selected goal
- Motor temperature notes
- Battery notes
- Prop/setup notes

## Output

### Improvement Verdict

Examples:

- Improved
- Slightly improved
- Mixed result
- Worse
- Inconclusive

### What Improved

Examples:

- Less bounceback
- Better propwash recovery
- Smoother throttle unload
- Better setpoint tracking
- Lower noise
- Better efficiency

### What Got Worse

Examples:

- Motors hotter
- More high-frequency noise
- More twitchy feel
- Worse battery sag
- More overshoot

### Keep / Revert / Refine Recommendation

Examples:

- Keep current tune as baseline.
- Revert D Max increase.
- Keep FF change but lower D slightly.
- Try one more small D increase only if motors stay cool.

## UI Direction

This page should feel like a lab notebook.

It should save or export:

- Markdown report
- JSON report
- short summary
- before/after values
- pilot notes

---

# 5. Blackbox Tools Page

## Purpose

This page should contain file-handling utilities.

It should reduce clutter on the main Analyze Tune page.

## Tools To Include

### Raw Blackbox Converter

For:

- .BBL
- .BFL

Converts to usable CSV if blackbox_decode exists locally.

### Multi-Flight Selector

Shows:

- Flight 1/n
- Flight 2/n
- duration
- file size
- sample rate if known
- newest/default selection

### CSV Optimizer

For:

- reducing file size
- removing repeated headers
- keeping needed columns
- making hosted uploads more reliable

### Parser Inspector

For:

- detected gyro columns
- detected setpoint columns
- throttle columns
- motor columns
- sample rate
- duration
- missing fields
- usable rows

## UI Rule

Normal users should not need this page unless they have file problems.

The Analyze Tune page can link to it when needed:

> Having upload problems? Open Blackbox Tools.

---

# 6. Advanced Diagnostics Page

## Purpose

This page is for experienced users, developer testing, and future high-detail analysis.

This is where complicated tools belong so they do not overwhelm normal users.

## Tools To Eventually Include

### FFT / Noise Analysis

Show:

- low/mid/high frequency energy
- possible frame resonance
- D-term noise risk
- filter-relevant peaks

### GPS Analysis

Show:

- speed profile
- distance traveled
- route summary
- speed vs throttle
- cruise efficiency
- return behavior

### High-Rate Logging Analysis

For logs recorded at:

- 1/4
- 1/2
- 1/1

This can reveal details not visible in normal 1/16 logs.

### Filter Diagnostic Tool

Analyze:

- whether filtering is too light
- whether filtering is too heavy
- delay risk
- noise persistence
- D-term heat risk

### Developer Raw Metrics

Show:

- internal scoring values
- normalized metrics
- threshold decisions
- classification results

This should not be shown to normal users by default.

## Warning Banner

Advanced Diagnostics should include a warning:

> Advanced outputs are diagnostic evidence, not automatic tuning commands. Do not make large PID or filter changes from one metric alone.

---

# 7. Docs / Learn Page

## Purpose

This page should teach users how to collect useful data and understand AeroTune.

## Recommended Docs

### Recommended Blackbox Test Flight

Explain:

- 60-90 seconds
- controlled throttle punches
- medium turns
- quick stops
- propwash descent
- flips/rolls if freestyle
- avoid crashes
- avoid random cruising-only logs

### Logging Rate Guide

Recommended:

- 1/16 = normal users and hosted uploads
- 1/8 = advanced users
- 1/4 = high-detail local analysis
- 1/2 = developer diagnostics
- 1/1 = deep testing only

### Hosted vs Local Guide

Explain:

Hosted site:

- easier
- may be slower
- file size limits
- good for normal logs

Local version:

- better for large logs
- better for raw files
- better for high-rate logs
- better for developer testing

### How To Export Logs

Explain:

- from Betaflight Blackbox Explorer
- choose the correct flight
- export CSV if needed
- raw .BBL supported locally if tools installed

### How To Interpret Recommendations

Explain:

- deltas are conservative
- do not blindly stack changes
- motor heat overrides style preference
- battery safety matters
- pilot feel matters

### Battery Safety Guide

Include practical mAh/voltage warnings.

---

# 8. Settings / Profile Page

## Purpose

This can be a later feature.

It allows users to save common setup info.

## Possible Saved Setup Fields

- Drone name
- Frame size
- Prop size
- Motor size/KV
- Battery size
- FC/ESC
- Betaflight version
- Default flying style
- Default logging rate
- Known safe mAh limit
- Known motor heat limit
- Local vs hosted preference

## Why It Matters

If users can save their setup, AeroTune can avoid asking the same questions every time.

This also helps tune-change tracking.

---

# Suggested First Implementation Phase

Do not rebuild everything at once.

Start with a light restructuring.

## Phase 1: Home Hub + Existing Tool Sections

Goal:

> Make the site feel less overwhelming without changing analyzer logic too much.

Tasks:

1. Create a homepage section at top.
2. Add large navigation cards.
3. Move current analyzer into an Analyze Tune section.
4. Hide advanced/parser/debug sections behind collapsible panels.
5. Add placeholder cards for future Pilot Performance and Advanced Diagnostics.
6. Keep existing backend mostly unchanged.

This gives better UX quickly without huge backend risk.

---

# Suggested Second Implementation Phase

## Phase 2: Separate Frontend Views

Goal:

> Turn major tools into separate views without needing a full framework rewrite.

If the site is still plain static HTML/JS, use simple view switching:

- Home
- Analyze Tune
- Tune Change Tracking
- Blackbox Tools
- Docs

This can be done with:

- buttons
- data-view attributes
- CSS display toggling
- one-page app style

No React required yet.

Example concept:

- clicking Analyze Tune hides home and shows analyzer section
- clicking Home returns to dashboard
- URLs can be added later

This keeps deployment simple.

---

# Suggested Third Implementation Phase

## Phase 3: Pilot Performance System

Goal:

> Add the five style profiles and performance scoring.

Add:

- style selector
- pilot performance analysis button
- performance output cards
- style match score
- throttle smoothness score
- recovery score
- GPS-enhanced metrics if available

Important:

Pilot Performance should not directly replace tune analysis.

It should be its own output area.

---

# Suggested Fourth Implementation Phase

## Phase 4: GPS-Enhanced Analysis

Goal:

> Use GPS fields from Blackbox logs when available.

Add detection for:

- GPS latitude/longitude
- GPS speed
- GPS altitude
- satellite count
- distance/home if logged

Then show:

- speed range
- max speed
- average cruise speed
- distance estimate
- speed vs throttle
- efficiency if current data exists

GPS should be optional.

If missing, AeroTune should say:

> GPS data not found. Performance analysis will use gyro, setpoint, throttle, and motor data only.

---

# UI Complexity Rules

These rules should guide future UI decisions.

## Rule 1: One Primary Job Per Page

Each page should have one main purpose.

Bad:

> Analyze tune, compare tune changes, score pilot, optimize CSV, show FFT, and teach docs all on one screen.

Good:

> Analyze Tune page focuses on tune changes only.

## Rule 2: Hide Advanced Details By Default

Most users need:

- what to change
- why
- how confident
- what to test next

They do not always need raw metrics.

## Rule 3: Use Progressive Disclosure

Show simple first.

Let users expand deeper details when needed.

Example:

- Summary
- Recommended changes
- Evidence
- Advanced metrics collapsed

## Rule 4: Do Not Mix Experimental Tools With Main Recommendations

Experimental features should be labeled clearly.

Example:

> Experimental GPS efficiency analysis

not:

> Final efficiency score

## Rule 5: Safety Warnings Override Style Goals

If a user selects Aggressive Freestyle, AeroTune still should not recommend unsafe D increases if motor heat risk is high.

## Rule 6: Local vs Hosted Must Be Clear

AeroTune should clearly state:

- Hosted version is for normal-size logs.
- Local version is best for large, high-rate, raw Blackbox logs.

This prevents confusion.

---

# Recommended Navigation Labels

Use simple, user-friendly labels.

Top navigation:

- Home
- Tune Analysis
- Pilot Performance
- Compare Changes
- Blackbox Tools
- Docs
- Advanced

Avoid overly technical nav labels like:

- FFT Engine
- Parser Matrix
- Setpoint Classifier

Those can exist inside Advanced, but not as top-level beginner navigation.

---

# Recommended Visual Layout

## Homepage

- Hero section
- 2-3 sentence explanation
- Card grid
- hosted/local warning note
- support/Patreon link if desired

## Tool Pages

Each page should use:

- clear heading
- short explanation
- input card
- action button
- result cards
- advanced details accordion

## Result Cards

Use separate cards for:

- Main recommendation
- Suggested values
- Evidence
- Safety warnings
- Next test flight
- Advanced details

This prevents giant paragraphs.

---

# How This Helps AeroTune

This redesign makes AeroTune feel more professional because it separates user intent.

It also makes development easier because new features can be added to the right place instead of being forced into the main analyzer screen.

Long term, AeroTune can become:

- a tune analyzer
- a flight performance analyzer
- a tune-change lab notebook
- a Blackbox file utility
- a GPS efficiency analyzer
- a learning tool

without overwhelming the user.

---

# Final Recommendation

Yes, AeroTune should move toward a homepage/dashboard model.

The best structure is:

> Home Hub -> Dedicated Tool Pages -> Advanced Details Hidden Until Needed

This gives AeroTune room to grow while keeping the user experience clean.

Do not remove advanced features.

Organize them.

That is the difference between a powerful tool and an overwhelming one.
