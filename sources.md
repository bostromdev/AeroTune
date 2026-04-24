# AeroTune Sources & Tuning Logic References

AeroTune uses conservative Betaflight-style tuning rules. It does **not** output final PID numbers. It outputs small percentage direction changes and pilot-readable reasons.

## Primary references

### Betaflight PID Tuning Guide
https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide

Used for these rules:

- P controls tracking authority and sharpness.
- Too much P can cause oscillation, overshoot, or bounce.
- I improves hold against wind, CG bias, and throttle-change attitude movement.
- D adds damping and helps bounceback/propwash.
- Too much D can amplify noise and heat motors.
- Feedforward improves stick response.
- Yaw is normally tuned with P/I/FF, with yaw D usually kept at 0.

### Betaflight Freestyle Tuning Principles
https://betaflight.com/docs/wiki/guides/current/Freestyle-Tuning-Principles

Used for:

- High-throttle oscillation / TPA-style diagnosis.
- Motor-noise awareness at high throttle.
- Avoiding unnecessary P/D gain increases when the issue is throttle-specific.

### Betaflight 4.0 Tuning Notes
https://betaflight.com/docs/wiki/tuning/4-0-Tuning-Notes

Used for:

- D-term noise awareness.
- High-throttle motor heat/noise risk.
- TPA-style reasoning for high-throttle roughness.

## Common FPV build/tune problem references

### Oscar Liang — Propwash explanation
https://oscarliang.com/propwash/

Used for:

- Propwash symptoms: dives, drops, hard turns, 180s, turbulent air recovery.
- Treating propwash as disturbed-air instability, not simple stick tracking delay.

### Oscar Liang — D gain, oscillation, and motor heat
https://oscarliang.com/excessive-d-gain-cause-oscillations-motor-overheat/

Used for:

- D-term heat caution.
- Avoiding D increases when high-frequency noise is detected.

### Oscar Liang — Vibration / jello / hot motors troubleshooting
https://oscarliang.com/mini-quad-motors-overheat/

Used for:

- Mechanical problem checks: props, motors, frame screws, stack mounting, filtering.
- Warning users not to PID-tune around broken hardware.

### Oscar Liang — PID/filter tuning with Blackbox
https://oscarliang.com/pid-filter-tuning-blackbox/

Used for:

- Separating mechanical/filter noise from PID problems.
- D-term filtering caution.

## AeroTune rule map

### Clean tune
Detected when:

- Gyro follows setpoint strongly.
- Residual error is low.
- No dominant high/mid/dirty-air problem band.
- No throttle-transition error spikes.

Output:

- Efficient: no PID change.
- Locked-In: optional tiny P/FF increase only.
- Floaty: optional tiny P/FF decrease only.

### Propwash
Detected when:

- Error spikes around throttle movement or disturbed-air style segments.
- Propwash/mid-band residual energy is elevated.

Output:

- Roll/Pitch: D up slightly, P not raised.
- Yaw: D stays 0; use yaw P/I only if flight feel confirms.

### Bounceback / overshoot
Detected when:

- Error spikes after command stops or sharp direction changes.

Output:

- Roll/Pitch: D up slightly, P down slightly if over-correcting.
- Yaw: D stays 0; reduce yaw P if it bounces.

### High-frequency noise
Detected when:

- High/ultra residual energy is elevated.
- Gyro high-frequency content is elevated.

Output:

- D down / do not raise D.
- P down slightly only if harsh/twitchy.
- Mechanical/filter checks first.

### Mid-frequency vibration
Detected when:

- Mid-band residual vibration is elevated.

Output:

- Mechanical checks first.
- P/D down slightly only if build is clean.

### High-throttle oscillation
Detected when:

- Error rises mainly during high throttle.

Output:

- P down slightly.
- Consider TPA-style behavior.
- D down only if motors are warm/noisy.

### Low-frequency oscillation
Detected when:

- Slow residual wobble is high **and** the log is not clean.

Output:

- P down first.
- D unchanged unless propwash/bounceback remains.

### Weak hold / drift
Detected when:

- Error persists during low stick input.

Output:

- I up slightly.

### Poor tracking / slow response
Detected when:

- Correlation is low or lag is high.

Output:

- P up for authority.
- FF up for stick response delay.

## Safety rules

- Make one small change at a time.
- Retest after every change.
- Check motor temperature after D changes.
- Do not raise D into high-frequency noise.
- Fix hardware/mechanical problems before tuning PID.
- Use 30–120 second logs with throttle variation, turns, punch-outs, dives, and normal flight.
