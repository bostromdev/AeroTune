# AeroTune Sources & Tuning References

AeroTune uses conservative, Betaflight-style tuning rules. It does not output final PID numbers. It outputs small percentage deltas and pilot-facing explanations.

## Primary source

- Betaflight PID Tuning Guide  
  https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide

## Rules implemented

### P — Proportional
- Used for tracking and authority.
- More P feels sharper.
- Too much P can overshoot or oscillate.
- AeroTune raises P only for soft tracking / poor authority.
- AeroTune lowers P for slow bounce, over-correction, and high-throttle oscillation.

### I — Integral
- Used for attitude hold against wind, CG bias, and throttle changes.
- AeroTune raises I for weak hold / drift.
- AeroTune avoids changing I for noise, propwash, or simple stick-response problems.

### D — Derivative / damping
- Used for damping bounceback and propwash.
- D can amplify gyro noise and heat motors.
- AeroTune raises D only for roll/pitch propwash or bounceback.
- AeroTune lowers or avoids D for high-frequency noise, vibration, or warm/rough motors.
- AeroTune always warns to check motor temperature after increasing D.

### Feedforward
- Used for stick response.
- AeroTune raises FF for delayed/soft stick response.
- AeroTune does not use FF to solve noise, drift, or propwash.

### Yaw
- Yaw D normally stays 0.
- AeroTune keeps yaw D at 0 and tunes yaw with P/I/FF only.

## CSV Optimizer

The optimizer converts common Blackbox CSV names into AeroTune's canonical format:

```text
time, gyro_x, gyro_y, gyro_z, setpoint_roll, setpoint_pitch, setpoint_yaw, throttle
```

This makes user logs easier to share, test, and compare.
