# AeroTune PID Tuning Guide

**Updated for AeroTune V1.6.1**

AeroTune gives conservative tuning direction based on Betaflight Blackbox evidence. It does not replace pilot judgment, official Betaflight documentation, or safe test-flying practice.

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

## 2. Use the Correct Flight Log

Before tuning anything, make sure the log is the right flight.

A raw `.BBL` file can contain multiple flights. In Betaflight Blackbox Explorer, select the correct flight first. If there are seven flights, the newest one is typically `7/7`. If there are three flights, the newest one is typically `3/3`.

For the safest workflow:

```text
Open .BBL in Blackbox Explorer
Select correct/latest flight
Export selected flight as CSV
Upload CSV into AeroTune
```

If you upload the wrong internal flight, the tuning advice can be based on the wrong behavior.

## 3. Clean Baseline First

Start in Efficient / Smooth mode. Fix objective problems first:

- Mechanical vibration
- High-frequency noise
- Propwash recovery problems
- Bounceback
- Weak attitude hold
- Poor tracking
- High-throttle oscillation

Only move to Locked-In / Responsive or Floaty / Cinematic after the log is clean enough.

## 4. Tune Goal Meanings

### Efficient / Smooth

Baseline cleanup mode. Use it first. It favors stable behavior, lower heat risk, and clean response.

### Locked-In / Responsive

Style mode. Use it after the baseline is clean. It may point toward tiny P or feedforward changes for stronger stick connection.

### Floaty / Cinematic

Style mode. Use it after the baseline is clean. It may point toward tiny P or feedforward reductions for smoother camera movement.

## 5. What PID Terms Usually Affect

```text
P     -> tracking strength, sharpness, response, possible oscillation if too high
I     -> attitude hold, drift resistance, long-term correction
D     -> damping, propwash control, bounceback reduction, possible heat/noise if too high
D Max -> extra dynamic damping during movement and aggressive stops
FF    -> stick response and snap without relying only on P
Yaw D -> normally stays at 0 in Betaflight-style FPV tuning
```

AeroTune recommends direction and caution, not final perfect PID numbers.

## 6. Reading the PID Tuning Advice Card

The PID Tuning Advice card gives:

- Axis affected
- Severity
- P/I/D/D Max/FF percentage suggestions
- Overshoot evidence
- Bounceback evidence
- Noise evidence
- A short test plan

Use it as a tuning assistant, not an automatic command.

## 7. Betaflight PID Value Calculator

After analysis, enter your current Betaflight values into the calculator. AeroTune converts percentage advice into estimated whole-number Betaflight values.

Example:

```text
Current Roll D: 30
Advice: D +9%
Suggested Roll D: 33
```

Round numbers are intentional. The goal is a safe first pass, not a fake exact tune.

## 8. Overshoot / Bounceback Pattern

Common conservative direction:

- Increase D on roll/pitch
- Increase D Max slightly with D
- Lower P slightly if the craft snaps past target
- Lower FF slightly if stick release feels twitchy
- Leave I alone unless the issue is drift or weak hold
- Leave yaw alone unless yaw itself is the problem

Check motor temperature after D/D Max changes.

## 9. High-Frequency Noise Pattern

If noise is high:

- Do not aggressively raise D
- Check props, motors, frame, stack mounting, and camera mount
- Fix mechanical noise before tuning around it

## 10. Tune-Change Tracking

Use tune-change tracking to prove whether a change helped.

```text
Before log: baseline
Change: one clear tuning change
After log: same route / same style
Result: keep, reduce, revert, or retest
```

## 11. One-Change-at-a-Time Rule

Good example:

```text
Before: baseline
Change: roll/pitch D +5%
After: same route, same props, same battery type
```

Bad example:

```text
Before: old props, old filters, old rates
Change: new props + filters + P up + D up + different battery
After: different route and heavier wind
```

The second example makes it hard to know what caused the result.

## 12. Safety Checks

Always:

- Make small changes
- Test one change at a time
- Check motor temperature after D-term changes
- Fix mechanical vibration before tuning around it
- Avoid aggressive changes on noisy logs
- Keep yaw D at normal Betaflight-safe defaults unless you know exactly why you are changing it
- Test in a safe area with enough room
