# AeroTune V1.8 Pilot Feel + Filtering Logic

## What changed

AeroTune now has a pilot-feel layer. During log upload, the pilot can select what they felt during the flight: good/cool motors, loose feel, propwash, bounceback, low-throttle climb/fall, hot motors, RPM-filter trouble, Dynamic Notch status, hardware changes, and crash/contact events.

The backend combines those selections with the Blackbox signal analysis to build a real tuning plan.

## Code map

```text
static/index.html
  Upload UI. Adds Pilot Flight Feel checkboxes and renders the Pilot Feel / Filtering / Real Plan cards.

static/blackbox_flight_selector.js
  Keeps pilot-feel checkbox values when the user switches between decoded flights from one raw .BBL/.BFL/.TXT file.

app/main.py
  API wiring. Receives repeated Form fields named pilot_feel and passes them into detect_oscillation().

app/analyzer.py
  Signal analysis engine. Reads gyro/setpoint/throttle behavior, then hands the result to the pilot-feel layer.

app/pilot_feel.py
  Human-context engine. Normalizes checkbox IDs, applies Dynamic Notch default policy, suppresses RPM-filter advice when needed, blocks unsafe D increases, and builds the real-world tuning plan.

app/tuning_advisor.py
  PID delta translator. Converts signal behavior into Betaflight-style percentage deltas, then applies the pilot-feel D-term safety gate.
```

## Filtering policy

```text
Dynamic Notch = default recommendation path
RPM filtering = advanced-only
RPM filter issue / arming warning / failsafe warning = suppress RPM-filter recommendations
Hot or buzzy motors = block positive D/D Max recommendations
Good flight + cool motors = protect the tune and save as baseline when no active tuning symptoms are selected
Use this as a good baseline / hold current tune = zero PID deltas only when it does not conflict with selected symptoms
Loose/floaty, twitchy, bounceback, propwash, weak tracking, high-throttle wobble, or similar symptoms = bypass baseline hold and build a targeted plan
If pilot feel and one-log metrics disagree = keep evidence visible, but require another comparable log before tuning
```

## Why Dynamic Notch is the default

Dynamic Notch follows gyro-detected vibration peaks. It does not require motor RPM telemetry. RPM filtering can be excellent on a correctly configured build, but it depends on bidirectional DShot, ESC firmware behavior, motor pole count, and stable RPM telemetry. If any of those are wrong or unstable, the pilot can get arming/failsafe trouble or unreliable filtering behavior.

Because AeroTune is meant to be safe for many builds, Dynamic Notch is the default path and RPM filtering is advanced-only.

## Why pilot feel matters

A log can show frequency energy, tracking error, propwash bands, and throttle-linked noise. It cannot always know whether the quad actually flew well, whether motors were hot, or whether a filter caused reliability problems.

The pilot-feel layer prevents bad advice like raising D after hot motors or changing a tune that already flew great.


## Good-baseline hold behavior

When the pilot selects **Felt good / no major issue** plus **Motors stayed cool**, or selects **Use this as a good baseline / hold current tune**, AeroTune treats the upload as a baseline only if the pilot did not also select active tuning symptoms. The analyzer can still show overshoot, bounceback, or noise evidence, but the PID advice is held at 0% because a single hard Acro maneuver can look like correction error in the graph.

This is intentional. A good flight with cool motors should not receive an aggressive “fix this now” recommendation unless the same issue repeats in another comparable log.

If the pilot also selects an active symptom such as **Felt loose / floaty**, **Felt twitchy / too sharp**, **Bounceback**, **Propwash**, **Roll/pitch tracking feels delayed**, **Roll feels weak**, **Pitch feels weak**, **Wobbles during high throttle**, or **Oscillation during throttle punches**, the baseline hold is bypassed. In that case AeroTune uses the selected symptom plus the log evidence to build a targeted plan instead of forcing all PID deltas to zero.

Low-throttle climb/fall is handled separately because it is usually throttle curve, idle, hover point, or craft weight before it is a PID problem.

## Recommended test flow

```text
1. Fly a controlled 60-90 second test.
2. Upload the log.
3. Select what the flight felt like.
4. Read the Filtering / D-Term Safety card first.
5. Make only the safest first change.
6. Retest the same route.
7. Use before/after comparison to prove whether the change helped.
```

## Per-option output and PID cap

Every selected pilot-feel checkbox now creates a visible `option_outputs` entry in the API response. These outputs explain how that feeling changes the plan, such as holding a good baseline, blocking D after hot motors, suppressing RPM-filter advice, or routing low-throttle climb/fall toward throttle curve and hover-point work.

Pilot-feel outputs are not direct additive PID commands. They are context/gates. AeroTune caps final Betaflight percentage advice to ±11% per test pass so multiple selected symptoms cannot stack into an unsafe +50% or +100% recommendation.
