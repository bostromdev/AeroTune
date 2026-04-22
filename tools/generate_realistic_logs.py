from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("sample_logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


@dataclass
class ScenarioConfig:
    name: str
    duration_s: float
    sample_rate_hz: int
    drone_size: str
    description: str


def smooth_signal(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def make_time(duration_s: float, sample_rate_hz: int) -> np.ndarray:
    n = int(duration_s * sample_rate_hz)
    return np.arange(n, dtype=float) / float(sample_rate_hz)


def colored_noise(n: int, scale: float = 1.0, alpha: float = 0.2) -> np.ndarray:
    white = RNG.normal(0.0, scale, n)
    return smooth_signal(white, alpha)


def sine_burst(
    t: np.ndarray,
    freq_hz: float,
    amplitude: float,
    start_s: float,
    end_s: float,
    phase: float = 0.0,
) -> np.ndarray:
    gate = ((t >= start_s) & (t <= end_s)).astype(float)
    if not np.any(gate):
        return np.zeros_like(t)
    fade = np.sin(np.pi * np.clip((t - start_s) / max(end_s - start_s, 1e-6), 0, 1)) ** 2
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t + phase) * gate * fade


def make_setpoint_profile(t: np.ndarray, axis_scale: float = 1.0) -> np.ndarray:
    sp = np.zeros_like(t)

    segments = [
        (0.7, 1.4, 120.0),
        (1.8, 2.7, -160.0),
        (3.0, 4.0, 210.0),
        (4.3, 5.0, -110.0),
        (5.5, 6.5, 180.0),
        (7.2, 8.2, -220.0),
    ]

    for start, end, target in segments:
        mask = (t >= start) & (t < end)
        if not np.any(mask):
            continue
        local_t = t[mask] - start
        ramp = 1.0 - np.exp(-4.0 * local_t)
        sp[mask] += target * ramp

    sp += 8.0 * np.sin(2 * np.pi * 0.8 * t)
    sp += 5.0 * np.sin(2 * np.pi * 1.6 * t + 0.7)

    return sp * axis_scale


def delayed_response(
    signal: np.ndarray,
    sample_rate_hz: int,
    lag_ms: float,
    smoothing_alpha: float,
    gain: float = 1.0,
) -> np.ndarray:
    lag_samples = max(0, int(round((lag_ms / 1000.0) * sample_rate_hz)))
    delayed = np.roll(signal, lag_samples)
    if lag_samples > 0:
        delayed[:lag_samples] = 0.0
    return smooth_signal(delayed * gain, smoothing_alpha)


def add_band_vibration(base: np.ndarray, t: np.ndarray, freq: float, amplitude: float) -> np.ndarray:
    vib = amplitude * np.sin(2.0 * np.pi * freq * t)
    vib += 0.35 * amplitude * np.sin(2.0 * np.pi * freq * 1.9 * t)
    vib += 0.2 * amplitude * np.sin(2.0 * np.pi * freq * 0.5 * t)
    return base + vib


def make_axis_trace(t: np.ndarray, sample_rate_hz: int, scenario: str, axis: str) -> Dict[str, np.ndarray]:
    axis_scale = {"roll": 1.0, "pitch": 1.1, "yaw": 0.7}[axis]

    setpoint = make_setpoint_profile(t, axis_scale)

    gyro = delayed_response(setpoint, sample_rate_hz, lag_ms=15, smoothing_alpha=0.15, gain=0.95)
    gyro += colored_noise(len(t), scale=2.0 * axis_scale, alpha=0.1)

    if scenario == "low_frequency_oscillation":
        gyro = add_band_vibration(gyro, t, 20, 10 * axis_scale)

    elif scenario == "mid_frequency_vibration":
        gyro = add_band_vibration(gyro, t, 50, 8 * axis_scale)

    elif scenario == "high_frequency_noise":
        gyro = add_band_vibration(gyro, t, 110, 5 * axis_scale)

    elif scenario == "poor_tracking":
        gyro = delayed_response(setpoint, sample_rate_hz, lag_ms=50, smoothing_alpha=0.05, gain=0.7)

    elif scenario == "slow_response":
        gyro = delayed_response(setpoint, sample_rate_hz, lag_ms=35, smoothing_alpha=0.07, gain=0.8)

    elif scenario == "propwash_or_rebound":
        gyro += sine_burst(t, 35, 10 * axis_scale, 3, 3.5)
        gyro += sine_burst(t, 35, 10 * axis_scale, 6, 6.5)

    return {
        f"{axis}_setpoint": setpoint,
        f"{axis}_gyro": gyro,
    }


def generate_log(config: ScenarioConfig) -> pd.DataFrame:
    t = make_time(config.duration_s, config.sample_rate_hz)

    data = {"time": t}

    for axis in ["roll", "pitch", "yaw"]:
        data.update(make_axis_trace(t, config.sample_rate_hz, config.name, axis))

    df = pd.DataFrame(data)
    return df


def main():
    scenarios = [
        ScenarioConfig("mostly_clean", 10.0, 1000, "7", ""),
        ScenarioConfig("low_frequency_oscillation", 10.0, 1000, "7", ""),
        ScenarioConfig("mid_frequency_vibration", 10.0, 1000, "7", ""),
        ScenarioConfig("high_frequency_noise", 10.0, 1000, "7", ""),
        ScenarioConfig("poor_tracking", 10.0, 1000, "7", ""),
        ScenarioConfig("slow_response", 10.0, 1000, "7", ""),
        ScenarioConfig("propwash_or_rebound", 10.0, 1000, "7", ""),
    ]

    for config in scenarios:
        df = generate_log(config)
        path = OUTPUT_DIR / f"{config.name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()