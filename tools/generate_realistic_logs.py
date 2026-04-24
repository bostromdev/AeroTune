from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


"""
AeroTune realistic sample Blackbox CSV generator.

Purpose:
- Generate 30–120 second CSV logs suitable for AeroTune testing.
- Include enough signal variety for PID analysis and public experimentation.
- Keep generated logs synthetic, safe, and clearly labeled as sample data.

Generated columns:
- time
- gyro_x / gyro_y / gyro_z
- setpoint_roll / setpoint_pitch / setpoint_yaw
- rc_command_roll / rc_command_pitch / rc_command_yaw
- throttle
- motor_1 / motor_2 / motor_3 / motor_4
- pid_p_roll / pid_i_roll / pid_d_roll / pid_ff_roll
- pid_p_pitch / pid_i_pitch / pid_d_pitch / pid_ff_pitch
- pid_p_yaw / pid_i_yaw / pid_d_yaw / pid_ff_yaw
- battery_voltage
- scenario
- log_quality_note

Run:
    python3 tools/generate_realistic_logs.py
"""

RNG = np.random.default_rng(42)
SAMPLING_RATE_HZ = 1000
DEFAULT_DURATION_SEC = 60.0
MIN_DURATION_SEC = 30.0
MAX_DURATION_SEC = 120.0


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    duration_sec: float
    throttle_kind: str
    quality_note: str


SCENARIOS: tuple[ScenarioConfig, ...] = (
    ScenarioConfig(
        name="30s_clean_tune",
        duration_sec=30.0,
        throttle_kind="mixed",
        quality_note="Good baseline log with hover, cruising, turns, and punch-outs.",
    ),
    ScenarioConfig(
        name="60s_locked_in_tracking",
        duration_sec=60.0,
        throttle_kind="mixed",
        quality_note="Good log with strong setpoint tracking and realistic motor harmonics.",
    ),
    ScenarioConfig(
        name="60s_mid_band_vibration",
        duration_sec=60.0,
        throttle_kind="mixed",
        quality_note="Good diagnostic log showing mid-frequency vibration useful for testing conservative P/D reductions.",
    ),
    ScenarioConfig(
        name="75s_high_frequency_noise",
        duration_sec=75.0,
        throttle_kind="aggressive",
        quality_note="Good diagnostic log showing higher-frequency noise and motor harmonic content.",
    ),
    ScenarioConfig(
        name="90s_propwash_unstable",
        duration_sec=90.0,
        throttle_kind="aggressive",
        quality_note="Good diagnostic log with throttle transitions, propwash-like bursts, and bounceback behavior.",
    ),
    ScenarioConfig(
        name="120s_mixed_public_test",
        duration_sec=120.0,
        throttle_kind="full",
        quality_note="Long public test log with varied maneuvers across the recommended 30–120 second window.",
    ),
)


def _validate_duration(duration_sec: float) -> float:
    if duration_sec < MIN_DURATION_SEC or duration_sec > MAX_DURATION_SEC:
        raise ValueError(
            f"duration_sec must be between {MIN_DURATION_SEC:.0f} and {MAX_DURATION_SEC:.0f} seconds."
        )
    return float(duration_sec)


def _timebase(duration_sec: float) -> np.ndarray:
    duration = _validate_duration(duration_sec)
    sample_count = int(SAMPLING_RATE_HZ * duration)
    return np.arange(sample_count, dtype=float) / SAMPLING_RATE_HZ


def _smooth_step(length: int, start: float, stop: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length)
    eased = x * x * (3.0 - 2.0 * x)
    return start + (stop - start) * eased


def _add_pulse(signal: np.ndarray, start: int, rise: int, hold: int, fall: int, amount: float) -> None:
    end = start + rise + hold + fall
    if start < 0 or end >= len(signal):
        return

    pulse = np.concatenate([
        _smooth_step(rise, 0.0, amount),
        np.full(hold, amount),
        _smooth_step(fall, amount, 0.0),
    ])
    signal[start:end] += pulse


def throttle_profile(time: np.ndarray, kind: str = "mixed") -> np.ndarray:
    n = len(time)
    duration = float(time[-1] - time[0]) if n > 1 else 0.0

    throttle = (
        0.34
        + 0.08 * np.sin(2 * np.pi * 0.035 * time + 0.2)
        + 0.05 * np.sin(2 * np.pi * 0.090 * time + 1.1)
        + RNG.normal(0, 0.010, n)
    )

    # Stable hover segment near the start.
    hover_end = min(n, int(SAMPLING_RATE_HZ * 5.0))
    throttle[:hover_end] = 0.30 + RNG.normal(0, 0.006, hover_end)

    if kind in {"mixed", "aggressive", "full"}:
        pulse_count = {
            "mixed": max(5, int(duration // 12)),
            "aggressive": max(8, int(duration // 8)),
            "full": max(12, int(duration // 7)),
        }[kind]

        for _ in range(pulse_count):
            start = int(RNG.integers(int(6 * SAMPLING_RATE_HZ), max(int(7 * SAMPLING_RATE_HZ), n - 900)))
            rise = int(RNG.integers(130, 260))
            hold = int(RNG.integers(90, 280))
            fall = int(RNG.integers(180, 360))
            amount = float(RNG.uniform(0.22, 0.52 if kind != "mixed" else 0.38))
            _add_pulse(throttle, start, rise, hold, fall, amount)

    if kind in {"aggressive", "full"}:
        # Add throttle chops / recovery sections for propwash-like testing.
        chop_count = max(4, int(duration // 18))
        for _ in range(chop_count):
            start = int(RNG.integers(int(8 * SAMPLING_RATE_HZ), max(int(9 * SAMPLING_RATE_HZ), n - 700)))
            length = int(RNG.integers(250, 650))
            throttle[start:start + length] -= float(RNG.uniform(0.12, 0.24))

    return np.clip(throttle, 0.08, 0.96)


def motor_fundamental_hz(throttle: np.ndarray) -> np.ndarray:
    return 72.0 + throttle * 132.0


def apply_physical_lag(signal: np.ndarray, lag_ms: float) -> np.ndarray:
    lag_samples = int(round((lag_ms / 1000.0) * SAMPLING_RATE_HZ))
    if lag_samples <= 0:
        return signal.copy()
    delayed = np.roll(signal, lag_samples)
    delayed[:lag_samples] = delayed[lag_samples]
    return delayed


def _axis_base(axis_name: str) -> tuple[float, float, float]:
    if axis_name == "roll":
        return 1.65, 0.70, 0.0
    if axis_name == "pitch":
        return 1.35, 0.60, 0.45
    return 0.85, 0.36, 0.85


def _maneuver_envelope(time: np.ndarray, axis_name: str) -> np.ndarray:
    freq, amp, phase = _axis_base(axis_name)
    signal = amp * np.sin(2 * np.pi * freq * time + phase)
    signal += 0.30 * amp * np.sin(2 * np.pi * (freq * 0.45) * time + phase + 1.2)

    # Add deliberate stick-deflection blocks so setpoint data is meaningful.
    block_interval = 9.0 if axis_name != "yaw" else 14.0
    for block_start_s in np.arange(7.0, time[-1] - 2.0, block_interval):
        start = int(block_start_s * SAMPLING_RATE_HZ)
        length = int(RNG.integers(550, 1200))
        if start + length >= len(signal):
            continue
        direction = 1.0 if RNG.random() > 0.5 else -1.0
        magnitude = amp * RNG.uniform(0.55, 1.05)
        signal[start:start + length] += direction * magnitude * np.hanning(length)

    return signal


def _scenario_axis_params(scenario: str, axis_name: str) -> tuple[float, float, float]:
    lag_map: Dict[str, Dict[str, float]] = {
        "30s_clean_tune": {"roll": 4, "pitch": 5, "yaw": 7},
        "60s_locked_in_tracking": {"roll": 3, "pitch": 4, "yaw": 6},
        "60s_mid_band_vibration": {"roll": 8, "pitch": 9, "yaw": 11},
        "75s_high_frequency_noise": {"roll": 5, "pitch": 6, "yaw": 8},
        "90s_propwash_unstable": {"roll": 13, "pitch": 15, "yaw": 12},
        "120s_mixed_public_test": {"roll": 6, "pitch": 7, "yaw": 9},
    }
    gain_map: Dict[str, Dict[str, float]] = {
        "30s_clean_tune": {"roll": 0.97, "pitch": 0.95, "yaw": 0.92},
        "60s_locked_in_tracking": {"roll": 1.00, "pitch": 0.99, "yaw": 0.95},
        "60s_mid_band_vibration": {"roll": 1.02, "pitch": 1.00, "yaw": 0.94},
        "75s_high_frequency_noise": {"roll": 0.96, "pitch": 0.94, "yaw": 0.90},
        "90s_propwash_unstable": {"roll": 0.98, "pitch": 0.93, "yaw": 0.88},
        "120s_mixed_public_test": {"roll": 0.98, "pitch": 0.96, "yaw": 0.92},
    }
    noise_map: Dict[str, Dict[str, float]] = {
        "30s_clean_tune": {"roll": 0.006, "pitch": 0.0065, "yaw": 0.0045},
        "60s_locked_in_tracking": {"roll": 0.0055, "pitch": 0.006, "yaw": 0.004},
        "60s_mid_band_vibration": {"roll": 0.010, "pitch": 0.011, "yaw": 0.006},
        "75s_high_frequency_noise": {"roll": 0.016, "pitch": 0.017, "yaw": 0.010},
        "90s_propwash_unstable": {"roll": 0.012, "pitch": 0.013, "yaw": 0.008},
        "120s_mixed_public_test": {"roll": 0.009, "pitch": 0.010, "yaw": 0.006},
    }
    return lag_map[scenario][axis_name], gain_map[scenario][axis_name], noise_map[scenario][axis_name]


def make_axis(axis_name: str, time: np.ndarray, throttle: np.ndarray, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    _, _, phase = _axis_base(axis_name)
    setpoint = _maneuver_envelope(time, axis_name)
    lag_ms, response_gain, sensor_noise_sigma = _scenario_axis_params(scenario, axis_name)

    gyro = response_gain * apply_physical_lag(setpoint, lag_ms)

    fundamental = motor_fundamental_hz(throttle)
    phase_1x = 2 * np.pi * np.cumsum(fundamental) / SAMPLING_RATE_HZ

    motor_1x_amp = {
        "30s_clean_tune": {"roll": 0.018, "pitch": 0.016, "yaw": 0.008},
        "60s_locked_in_tracking": {"roll": 0.015, "pitch": 0.014, "yaw": 0.007},
        "60s_mid_band_vibration": {"roll": 0.026, "pitch": 0.024, "yaw": 0.010},
        "75s_high_frequency_noise": {"roll": 0.030, "pitch": 0.028, "yaw": 0.014},
        "90s_propwash_unstable": {"roll": 0.022, "pitch": 0.020, "yaw": 0.010},
        "120s_mixed_public_test": {"roll": 0.020, "pitch": 0.019, "yaw": 0.009},
    }[scenario][axis_name]

    motor_1x = motor_1x_amp * np.sin(phase_1x + phase)
    motor_2x = (motor_1x_amp * 0.42) * np.sin(2 * phase_1x + 0.4 + phase)
    motor_3x = (motor_1x_amp * 0.20) * np.sin(3 * phase_1x + 1.0 + phase)

    extra = np.zeros_like(time)

    if scenario in {"30s_clean_tune", "60s_locked_in_tracking"}:
        extra += 0.008 * np.sin(2 * np.pi * (58 + 7 * throttle) * time + 0.2)

    elif scenario == "60s_mid_band_vibration":
        band_center = {"roll": 86, "pitch": 94, "yaw": 72}[axis_name]
        vib_amp = {"roll": 0.090, "pitch": 0.078, "yaw": 0.038}[axis_name]
        extra += vib_amp * np.sin(2 * np.pi * (band_center + 10 * throttle) * time + phase)
        extra += (vib_amp * 0.35) * np.sin(2 * np.pi * (band_center * 1.55) * time + 0.5)

    elif scenario == "75s_high_frequency_noise":
        extra += (motor_1x_amp * 0.85) * np.sin(3.2 * phase_1x + 0.8)
        extra += (motor_1x_amp * 0.55) * np.sin(4.1 * phase_1x + 1.6)
        extra += 0.010 * np.sin(2 * np.pi * (230 + 20 * throttle) * time + phase)

    elif scenario == "90s_propwash_unstable":
        transitions = np.abs(np.gradient(throttle))
        transitions = transitions / (np.max(transitions) + 1e-9)
        extra += 0.036 * transitions * np.sin(2 * np.pi * (38 + 12 * throttle) * time + phase)
        extra += 0.028 * transitions * np.sin(2 * np.pi * (74 + 16 * throttle) * time + 0.3)
        for _ in range(max(8, int(len(time) / SAMPLING_RATE_HZ // 8))):
            start = int(RNG.integers(150, len(time) - 220))
            length = int(RNG.integers(100, 180))
            sigma = {"roll": 0.050, "pitch": 0.060, "yaw": 0.028}[axis_name]
            burst = RNG.normal(0, sigma, length)
            extra[start:start + length] += burst * np.hanning(length)

    elif scenario == "120s_mixed_public_test":
        extra += 0.018 * np.sin(2 * np.pi * (68 + 8 * throttle) * time + phase)
        extra += 0.020 * np.abs(np.gradient(throttle)) * np.sin(2 * np.pi * (42 + 8 * throttle) * time + phase)

    sensor_noise = RNG.normal(0, sensor_noise_sigma, len(time))
    gyro = gyro + motor_1x + motor_2x + motor_3x + extra + sensor_noise
    return setpoint, gyro


def _pid_terms(setpoint: np.ndarray, gyro: np.ndarray, axis: str) -> Dict[str, np.ndarray]:
    error = setpoint - gyro
    integral = np.cumsum(error) / SAMPLING_RATE_HZ
    derivative = np.gradient(error) * SAMPLING_RATE_HZ

    p_gain = {"roll": 0.62, "pitch": 0.66, "yaw": 0.48}[axis]
    i_gain = {"roll": 0.18, "pitch": 0.20, "yaw": 0.16}[axis]
    d_gain = {"roll": 0.018, "pitch": 0.020, "yaw": 0.0}[axis]
    ff_gain = {"roll": 0.12, "pitch": 0.13, "yaw": 0.08}[axis]

    return {
        f"pid_p_{axis}": p_gain * error,
        f"pid_i_{axis}": i_gain * integral,
        f"pid_d_{axis}": d_gain * derivative,
        f"pid_ff_{axis}": ff_gain * np.gradient(setpoint) * SAMPLING_RATE_HZ,
    }


def _motor_outputs(throttle: np.ndarray, roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> Dict[str, np.ndarray]:
    roll_mix = 0.060 * np.tanh(roll)
    pitch_mix = 0.060 * np.tanh(pitch)
    yaw_mix = 0.035 * np.tanh(yaw)

    motors = {
        "motor_1": throttle + roll_mix - pitch_mix + yaw_mix,
        "motor_2": throttle - roll_mix - pitch_mix - yaw_mix,
        "motor_3": throttle - roll_mix + pitch_mix + yaw_mix,
        "motor_4": throttle + roll_mix + pitch_mix - yaw_mix,
    }
    return {key: np.clip(value, 0.0, 1.0) for key, value in motors.items()}


def _battery_voltage(time: np.ndarray, throttle: np.ndarray) -> np.ndarray:
    sag = 0.62 * throttle + 0.18 * np.maximum(0.0, np.gradient(throttle) * SAMPLING_RATE_HZ)
    slow_drop = 0.55 * (time / max(time[-1], 1e-9))
    voltage = 16.75 - slow_drop - sag + RNG.normal(0, 0.025, len(time))
    return np.clip(voltage, 13.6, 16.85)


def build_log(config: ScenarioConfig) -> pd.DataFrame:
    time = _timebase(config.duration_sec)
    throttle = throttle_profile(time, config.throttle_kind)

    setpoint_roll, gyro_x = make_axis("roll", time, throttle, config.name)
    setpoint_pitch, gyro_y = make_axis("pitch", time, throttle, config.name)
    setpoint_yaw, gyro_z = make_axis("yaw", time, throttle, config.name)

    data: Dict[str, Iterable[float] | str] = {
        "time": time,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "setpoint_roll": setpoint_roll,
        "setpoint_pitch": setpoint_pitch,
        "setpoint_yaw": setpoint_yaw,
        "rc_command_roll": setpoint_roll,
        "rc_command_pitch": setpoint_pitch,
        "rc_command_yaw": setpoint_yaw,
        "throttle": throttle,
        "battery_voltage": _battery_voltage(time, throttle),
        "scenario": config.name,
        "log_quality_note": config.quality_note,
    }

    data.update(_motor_outputs(throttle, gyro_x, gyro_y, gyro_z))

    for axis, setpoint, gyro in (
        ("roll", setpoint_roll, gyro_x),
        ("pitch", setpoint_pitch, gyro_y),
        ("yaw", setpoint_yaw, gyro_z),
    ):
        data.update(_pid_terms(setpoint, gyro, axis))

    df = pd.DataFrame(data)

    # Stable column order for people analyzing the logs manually.
    ordered = [
        "time",
        "gyro_x", "gyro_y", "gyro_z",
        "setpoint_roll", "setpoint_pitch", "setpoint_yaw",
        "rc_command_roll", "rc_command_pitch", "rc_command_yaw",
        "throttle",
        "motor_1", "motor_2", "motor_3", "motor_4",
        "pid_p_roll", "pid_i_roll", "pid_d_roll", "pid_ff_roll",
        "pid_p_pitch", "pid_i_pitch", "pid_d_pitch", "pid_ff_pitch",
        "pid_p_yaw", "pid_i_yaw", "pid_d_yaw", "pid_ff_yaw",
        "battery_voltage",
        "scenario", "log_quality_note",
    ]
    return df[ordered]


def write_manifest(out_dir: Path, generated: list[Path]) -> None:
    manifest = out_dir / "README.md"
    rows = [
        "# AeroTune Sample Logs",
        "",
        "These CSV files are synthetic 1 kHz FPV-style logs generated for AeroTune testing.",
        "They are intentionally 30–120 seconds long and include gyro, setpoint, RC command, throttle, motor, PID-term, and battery data.",
        "",
        "## Files",
        "",
    ]
    for path in generated:
        rows.append(f"- `{path.name}`")
    rows.extend([
        "",
        "## Notes",
        "",
        "- These are not real Betaflight Blackbox logs.",
        "- They are designed for analyzer development, demos, and community experimentation.",
        "- Real tuning decisions should always be made from real flight logs.",
    ])
    manifest.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "sample_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for config in SCENARIOS:
        df = build_log(config)
        out_path = out_dir / f"{config.name}.csv"
        df.to_csv(out_path, index=False)
        generated.append(out_path)
        print(
            f"Generated {out_path} "
            f"({len(df):,} rows, {df['time'].iloc[-1] - df['time'].iloc[0]:.1f}s, {len(df.columns)} columns)"
        )

    write_manifest(out_dir, generated)
    print(f"Wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
