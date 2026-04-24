# AeroTune Sources & Tuning References

AeroTune uses conservative, feel-based PID recommendations grounded in Betaflight tuning principles.

## Primary Source

Betaflight PID Tuning Guide  
https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide

Key rules used in AeroTune:

- P controls how tightly the quad tracks setpoint. Higher P feels sharper, but too much can overshoot or oscillate.
- I improves attitude hold against wind, CG imbalance, throttle changes, and persistent bias.
- D adds damping for bounceback and propwash, but it also amplifies high-frequency noise and can heat motors.
- Feedforward improves stick response without forcing P to do all the response work.
- Yaw D is normally kept at 0. Yaw is mainly tuned with P, I, and FF.
- Raise D only when it helps propwash or bounceback, and always check motor temperature after D changes.
- If high-frequency noise is detected, do not raise D. Fix mechanical/filtering issues first.

## AeroTune Rule Mapping

### Clean Tune
Detected when:
- setpoint tracking is high
- tracking error is low
- residual high-frequency noise is low
- no throttle or stop-related error spikes dominate

Output:
- no PID repair needed
- optional feel changes only for Locked-In or Floaty goals

### High-Frequency Noise
Detected when:
- residual high/ultra frequency energy is elevated
- error may be small overall, but fast residual noise dominates

Fix:
- lower D or do not raise D
- lower P slightly only if harsh/twitchy
- check props, motors, frame, stack mounting, RPM filtering, dynamic filtering

### Mid-Frequency Vibration
Detected when:
- residual or gyro mid-band energy is elevated

Fix:
- inspect mechanical build first
- soften P/D slightly if the build is clean

### Propwash / Bounceback
Detected when:
- error rises around throttle transitions, stops, or disturbed-air recovery
- residual is not primarily high-frequency noise

Fix:
- roll/pitch: raise D slightly or use D Max style damping
- do not raise P first
- check motor temperature after D changes

### Low-Frequency Oscillation
Detected when:
- residual low-band motion dominates
- tracking is not clean
- high-frequency noise is not the main issue

Fix:
- lower P first
- leave D alone unless bounceback/propwash remains

### Weak Hold / Drift
Detected when:
- residual error persists during quiet-stick or low-input sections

Fix:
- raise I slightly

### Poor Tracking / Slow Response
Detected when:
- gyro/setpoint correlation is low, or lag is high

Fix:
- raise P for tracking
- raise FF first when the issue is delayed stick response

## Log Optimizer Format

AeroTune-ready CSV format:

```csv
time,gyro_x,gyro_y,gyro_z,setpoint_roll,setpoint_pitch,setpoint_yaw,throttle
```

## Safety Notes

- Make one small change at a time.
- Retest after every change.
- Check motor temperature after raising D.
- Fix mechanical noise before using PID to hide vibration.
- These are conservative percentage deltas, not final PID numbers.
