from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

SAMPLING_RATE = 1000
DURATION_SEC = 6.0
N = int(SAMPLING_RATE * DURATION_SEC)
TIME = np.arange(N) / SAMPLING_RATE


def throttle_profile(kind="mixed"):
    base = 0.32 + 0.22 * np.sin(2 * np.pi * 0.12 * TIME + 0.3)
    noise = rng.normal(0, 0.012, N)
    throttle = base + noise

    if kind in {"mixed", "aggressive"}:
        pulse_count = 4 if kind == "mixed" else 6
        for _ in range(pulse_count):
            start = rng.integers(200, N - 350)
            up = np.linspace(0, 0.34 if kind == "mixed" else 0.48, 120)
            down = np.linspace(up[-1], 0, 180)
            pulse = np.concatenate([up, down])
            throttle[start:start + len(pulse)] += pulse

    return np.clip(throttle, 0.08, 0.95)


def motor_hz(throttle):
    return 78 + throttle * 118


def apply_physical_lag(signal, lag_ms):
    lag_samples = int(round((lag_ms / 1000.0) * SAMPLING_RATE))
    if lag_samples <= 0:
        return signal.copy()
    delayed = np.roll(signal, lag_samples)
    delayed[:lag_samples] = delayed[lag_samples]
    return delayed


def make_axis(axis_name, throttle, scenario):
    if axis_name == "roll":
        base_freq = 1.8
        base_amp = 0.60
        phase = 0.0
    elif axis_name == "pitch":
        base_freq = 1.35
        base_amp = 0.52
        phase = 0.35
    else:
        base_freq = 0.85
        base_amp = 0.30
        phase = 0.70

    setpoint = base_amp * np.sin(2 * np.pi * base_freq * TIME + phase)

    lag_ms = {
        "clean_tune": {"roll": 4, "pitch": 5, "yaw": 7},
        "mid_band_vibration": {"roll": 7, "pitch": 8, "yaw": 9},
        "high_freq_noise": {"roll": 5, "pitch": 6, "yaw": 7},
        "propwash_unstable": {"roll": 10, "pitch": 12, "yaw": 10},
    }[scenario][axis_name]

    response_gain = {
        "clean_tune": {"roll": 0.97, "pitch": 0.95, "yaw": 0.92},
        "mid_band_vibration": {"roll": 1.02, "pitch": 1.00, "yaw": 0.94},
        "high_freq_noise": {"roll": 0.96, "pitch": 0.94, "yaw": 0.90},
        "propwash_unstable": {"roll": 0.98, "pitch": 0.93, "yaw": 0.88},
    }[scenario][axis_name]

    gyro = response_gain * apply_physical_lag(setpoint, lag_ms)

    fundamental = motor_hz(throttle)
    phase_1x = 2 * np.pi * np.cumsum(fundamental) / SAMPLING_RATE

    motor_1x_amp = {
        "clean_tune": {"roll": 0.018, "pitch": 0.016, "yaw": 0.008},
        "mid_band_vibration": {"roll": 0.026, "pitch": 0.024, "yaw": 0.010},
        "high_freq_noise": {"roll": 0.030, "pitch": 0.028, "yaw": 0.014},
        "propwash_unstable": {"roll": 0.022, "pitch": 0.020, "yaw": 0.010},
    }[scenario][axis_name]

    motor_1x = motor_1x_amp * np.sin(phase_1x + phase)
    motor_2x = (motor_1x_amp * 0.42) * np.sin(2 * phase_1x + 0.4 + phase)
    motor_3x = (motor_1x_amp * 0.20) * np.sin(3 * phase_1x + 1.0 + phase)

    extra = np.zeros(N)

    if scenario == "clean_tune":
        extra += 0.010 * np.sin(2 * np.pi * (62 + 6 * throttle) * TIME + 0.2)

    elif scenario == "mid_band_vibration":
        band_center = {"roll": 86, "pitch": 94, "yaw": 72}[axis_name]
        vib_amp = {"roll": 0.090, "pitch": 0.078, "yaw": 0.038}[axis_name]
        extra += vib_amp * np.sin(2 * np.pi * (band_center + 10 * throttle) * TIME + phase)
        extra += (vib_amp * 0.35) * np.sin(2 * np.pi * (band_center * 1.55) * TIME + 0.5)

    elif scenario == "high_freq_noise":
        extra += (motor_1x_amp * 0.85) * np.sin(3.2 * phase_1x + 0.8)
        extra += (motor_1x_amp * 0.55) * np.sin(4.1 * phase_1x + 1.6)
        extra += 0.010 * np.sin(2 * np.pi * (230 + 20 * throttle) * TIME + phase)

    elif scenario == "propwash_unstable":
        trans = np.abs(np.gradient(throttle))
        trans = trans / (np.max(trans) + 1e-9)
        extra += 0.032 * trans * np.sin(2 * np.pi * (38 + 12 * throttle) * TIME + phase)
        extra += 0.024 * trans * np.sin(2 * np.pi * (74 + 16 * throttle) * TIME + 0.3)
        for _ in range(7):
            start = rng.integers(150, N - 180)
            length = rng.integers(90, 150)
            sigma = {"roll": 0.050, "pitch": 0.060, "yaw": 0.028}[axis_name]
            burst = rng.normal(0, sigma, length)
            env = np.hanning(length)
            extra[start:start + length] += burst * env

    sensor_noise = rng.normal(0, {
        "clean_tune": {"roll": 0.006, "pitch": 0.0065, "yaw": 0.0045},
        "mid_band_vibration": {"roll": 0.010, "pitch": 0.011, "yaw": 0.006},
        "high_freq_noise": {"roll": 0.016, "pitch": 0.017, "yaw": 0.010},
        "propwash_unstable": {"roll": 0.012, "pitch": 0.013, "yaw": 0.008},
    }[scenario][axis_name], N)

    gyro = gyro + motor_1x + motor_2x + motor_3x + extra + sensor_noise
    return setpoint, gyro


def build_log(scenario):
    throttle_kind = "aggressive" if scenario in {"high_freq_noise", "propwash_unstable"} else "mixed"
    throttle = throttle_profile(throttle_kind)

    sr, gr = make_axis("roll", throttle, scenario)
    sp, gp = make_axis("pitch", throttle, scenario)
    sy, gy = make_axis("yaw", throttle, scenario)

    return pd.DataFrame({
        "time": TIME,
        "gyro_x": gr,
        "gyro_y": gp,
        "gyro_z": gy,
        "setpoint_roll": sr,
        "setpoint_pitch": sp,
        "setpoint_yaw": sy,
        "throttle": throttle,
    })


def main():
    out_dir = Path(__file__).resolve().parent.parent / "sample_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        "clean_tune",
        "mid_band_vibration",
        "high_freq_noise",
        "propwash_unstable",
    ]

    for scenario in scenarios:
        df = build_log(scenario)
        out_path = out_dir / f"{scenario}.csv"
        df.to_csv(out_path, index=False)
        print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
