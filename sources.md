# AeroTune Sources & PID Rule Basis

AeroTune recommendations are based on Betaflight tuning principles and conservative FPV tuning practice.

## Primary Reference

- Betaflight PID Tuning Guide: https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide

## Rules Used In The Analyzer

### P — Proportional
- P controls how tightly the quad tracks the sticks / setpoint.
- Higher P feels sharper.
- Too much P can cause overshoot, bounce, or oscillation.
- Too little P can feel soft, loose, or sloppy.

### I — Integral
- I helps the quad hold attitude against wind, CG imbalance, and throttle changes.
- If the quad drifts, slips, or does not hold angle/heading through throttle changes, raise I slightly.
- Too much I can feel stiff, robotic, or slow.

### D — Derivative / Damping
- D damps motion, bounceback, and propwash.
- D is useful when roll/pitch has bounceback or propwash.
- D amplifies gyro noise and can heat motors, so changes must be small.
- If the signal is noisy or motors are hot, lower D or improve filtering/mechanical condition.

### Feedforward — FF
- FF improves stick response.
- Raise FF when the quad follows the command but feels delayed.
- FF is not the first fix for vibration, propwash, or drift.

### Yaw
- Yaw D normally stays at 0.
- Tune yaw mainly with P, I, and FF.
- If yaw gets rough at high throttle or fast forward flight, lower yaw P slightly.
- If yaw does not hold heading during throttle changes, raise yaw I slightly.

## Analyzer Outcomes

### Clean Tune
Output:
> No PID change needed.

Used when tracking is clean, error is low, and there is no throttle-linked disturbance.

### Mostly Clean
Output:
> No required fix. Optional feel tuning only.

Used when no strong problem is detected, but the user selected a feel goal.

### Propwash / Bounceback
Output:
> Raise D slightly on roll/pitch. Do not raise P first. Check motor heat.

Used when tracking error rises during active throttle or disturbed-air sections.

### Yaw Throttle Roughness
Output:
> Keep yaw D at 0. Lower yaw P slightly if roughness appears at throttle. Raise yaw I only if heading hold is weak.

### High-Frequency Noise
Output:
> Lower D or do not raise D. Check mechanical noise.

### Mid-Frequency Vibration
Output:
> Check mechanical vibration first. Lower P/D slightly only if the build is clean.

### High-Throttle Oscillation
Output:
> Lower P slightly. Consider TPA later if the problem only appears at high throttle.

### Low-Frequency Oscillation / Loose Bounce
Output:
> Lower P first. Leave D alone unless bounceback/propwash remains.

### Weak Hold / Drift
Output:
> Raise I slightly.

### Soft Tracking
Output:
> Raise P slightly, or FF if the main issue is delayed stick feel.

### Slow Stick Response
Output:
> Raise FF first. Add P only if it still feels soft.

## Safety

- Make one small change at a time.
- Check motor temperature after raising D.
- Fix mechanical vibration before increasing D.
- Do not change PID on a clean tune unless the pilot wants a different feel.
