# AeroTune Sample Logs

These CSV files are synthetic 1 kHz FPV-style logs generated for AeroTune testing.
They are intentionally 30–120 seconds long and include gyro, setpoint, RC command, throttle, motor, PID-term, and battery data.

## Files

- `30s_clean_tune.csv`
- `60s_locked_in_tracking.csv`
- `60s_mid_band_vibration.csv`
- `75s_high_frequency_noise.csv`
- `90s_propwash_unstable.csv`
- `120s_mixed_public_test.csv`

## Notes

- These are not real Betaflight Blackbox logs.
- They are designed for analyzer development, demos, and community experimentation.
- Real tuning decisions should always be made from real flight logs.