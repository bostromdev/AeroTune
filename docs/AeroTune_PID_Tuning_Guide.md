# AeroTune PID Tuning Guide

**Updated for AeroTune V1.5.1**

This guide explains how to use AeroTune's output safely. AeroTune gives conservative tuning direction based on Betaflight Blackbox evidence. It does not replace pilot judgment, official Betaflight documentation, or safe test-flying practice.

## 1. Core Rule

Do not chase feel before the baseline is clean.

```text
Dirty / noisy log -> baseline cleanup only
Clean log -> style tuning allowed
```

AeroTune's safest loop is:

```text
Fly -> log -> analyze -> make one small change -> fly again -> compare before/after
```

## 2. Clean Baseline First

Start in Efficient / Smooth mode. Fix objective problems first:

- Mechanical vibration
- High-frequency noise
- Propwash recovery problems
- Bounceback
- Weak attitude hold
- Poor tracking
- High-throttle oscillation

Only move to Locked-In / Responsive or Floaty / Cinematic after the log is clean enough.

## 3. Tune Goal Meanings

### Efficient / Smooth

Baseline cleanup mode. Use it first. It favors stable behavior, lower heat risk, and clean response.

### Locked-In / Responsive

Style mode. Use it after the baseline is clean. It may point toward tiny P or feedforward changes for stronger stick connection.

### Floaty / Cinematic

Style mode. Use it after the baseline is clean. It may point toward tiny P or feedforward reductions for smoother camera movement.

## 4. How to Create a Useful Test Log

Use a controlled 60-90 second flight with repeatable maneuvers:

- Smooth cruise
- Small roll/pitch/yaw inputs
- Controlled throttle punches
- Medium turns
- Quick stops and direction changes
- One or two safe propwash recovery moments

Avoid logs dominated by crashes, heavy wind, damaged props, battery sag, or random hovering.

## 5. What PID Terms Usually Affect

```text
P  -> tracking strength, sharpness, response, possible oscillation if too high
I  -> attitude hold, drift resistance, long-term correction
D  -> damping, propwash control, bounceback reduction, possible heat/noise if too high
FF -> stick response and snap without relying only on P
Yaw D -> normally stays at 0 in Betaflight-style FPV tuning
```

AeroTune recommends direction and caution, not final perfect PID numbers.

## 6. Issue Patterns and Conservative Actions

### Clean Tune

No major PID change is needed. Use before/after tracking if changing feel.

### Propwash / Bounceback

Possible direction:

- D up slightly on roll/pitch
- P down slightly if bounce/overshoot is also present
- Check motor temperature after D-term changes

### High-Frequency Noise

Possible direction:

- Reduce aggressive P/D changes
- Check filters, props, motors, frame stiffness, and build vibration
- Do not raise D into obvious noise

### Low-Frequency Wobble

Possible direction:

- Reduce P first
- Check tune balance and mechanical causes
- Avoid stacking multiple changes at once

### Weak Hold / Drift

Possible direction:

- I up slightly
- Check whether the issue is actual drift, wind, or pilot input

### Poor Tracking

Possible direction:

- P up slightly if the log is clean enough
- Feedforward up if stick response is slow
- Do not increase P/FF aggressively on noisy logs

### Slow Stick Response

Possible direction:

- Feedforward first
- Then small P change if needed
- Confirm with before/after comparison

## 7. V1.4 Before/After Comparison

Use V1.4 comparison whenever you want proof that a tune improved.

Best practice:

```text
Before log: baseline tune
After log: same drone, same style, one tuning change
```

AeroTune compares tracking, noise, propwash behavior, axis verdicts, and overall score. If the flights are not similar, confidence may be low.

## 8. V1.5 Tune-Change Tracking Workflow

Use V1.5 when you want to validate a specific change.

```text
1. Fly a clean before log.
2. Change one thing only.
3. Fly the same route or maneuver set again.
4. Upload before and after logs.
5. Enter the tune-change note, such as "D up 5% on roll/pitch".
6. Review the keep/reduce/revert/retest decision.
```

The decision means:

```text
KEEP   -> evidence suggests the change helped
REDUCE -> some improvement, but risk or side effects increased
REVERT -> evidence suggests the change made the tune worse
RETEST -> logs are not similar enough or confidence is too low
```

## 9. V1.5.1 Saved Report Use

V1.5.1 saves tune-change reports as JSON and Markdown. Use the Markdown report as a pilot-readable record and the JSON report for future tooling or data analysis.

Report files are stored locally under:

```text
reports/tune_changes/
```

Do not publish reports unless you intentionally want to share filenames, tuning notes, and flight-analysis details.

## 10. One-Change-at-a-Time Rule

Do not change P, I, D, feedforward, filters, props, and rates all in the same test if you want meaningful evidence.

Good example:

```text
Before: baseline
Change: roll/pitch D +5%
After: same route, same props, same battery type
```

Bad example:

```text
Before: old props, old filters, old rates
Change: new props + filter changes + P up + D up + different battery
After: different route and heavier wind
```

The second example makes it hard to know what caused the result.

## 11. Safety Checks

Always:

- Make small changes
- Test one change at a time
- Check motor temperature after D-term changes
- Fix mechanical vibration before tuning around it
- Avoid aggressive changes on noisy logs
- Keep yaw D at normal Betaflight-safe defaults unless you know exactly why you are changing it
- Test in a safe area with enough room

## 12. How to Read AeroTune Recommendations

AeroTune output should be treated as engineering evidence, not an absolute command.

Use this interpretation:

```text
High confidence + similar before/after flights -> stronger evidence
Low confidence + different flights -> retest before changing tune
Noise got worse -> reduce or revert risky changes
Tracking improved but noise worsened -> reduce, not blindly keep
Clean baseline achieved -> style tuning can begin carefully
```

## 13. Final Tuning Loop

```text
1. Start with Efficient / Smooth.
2. Get a clean baseline.
3. Make one small change.
4. Run V1.4 or V1.5 comparison.
5. Keep, reduce, revert, or retest.
6. Only tune style after objective issues are solved.
```
