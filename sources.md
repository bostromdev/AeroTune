# AeroTune Sources

AeroTune uses conservative, pilot-facing PID tuning rules based on Betaflight documentation.

## Primary references

- Betaflight PID Tuning Guide  
  https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide

- Betaflight PID Tuning Tab  
  https://betaflight.com/docs/wiki/app/pid-tuning-tab

- Betaflight 4.2 Tuning Notes  
  https://betaflight.com/docs/wiki/tuning/4-2-Tuning-Notes

## Rule basis

- **P** improves tracking and authority, but too much can cause oscillation, bounceback, or roughness.
- **I** improves attitude hold through wind, CG bias, throttle changes, and persistent error.
- **D** damps bounceback and propwash, but too much D can amplify noise and heat motors.
- **Feedforward** improves stick response without relying only on P.
- **Yaw** usually keeps D at 0; yaw is tuned mainly with P, I, and Feedforward.

## AeroTune outcomes

AeroTune keeps the user-facing output simple:

- Clean tune
- Propwash
- Bounceback / overshoot
- High-frequency noise
- Mid-frequency vibration
- High-throttle oscillation
- Low-frequency wobble
- Weak hold / drift
- Poor tracking
- Slow response

## Safety rule

AeroTune recommends small percentage deltas only. It does not output fake final PID numbers.
After any D increase, do a short test flight and check motor temperature.
