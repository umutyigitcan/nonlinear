"""Nonlinear UAV target-tracking simulation and controller implementations.

The model is intentionally compact enough for education, while preserving the
important nonlinearities: quadratic drag, position/time dependent wind,
actuator saturation and a three-dimensional moving target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


DT = 0.05
DURATION = 60.0
WARMUP_SEC = 2.0
TRACK_TOLERANCE = 0.50
ACCURACY_SCALE = 0.75
MAX_ACCELERATION = 8.0
DRAG_COEFFICIENT = 0.028


CONTROLLER_INFO = {
    "pid": {
        "short": "Nonlinear PID",
        "title": "Kazanç Ayarlı Doğrusal Olmayan PID",
        "description": "Hata büyüklüğüne göre kazançlarını değiştirir; integral sınırı ve rüzgâr telafisiyle temel karşılaştırma sağlar.",
        "color": "#38bdf8",
    },
    "backstepping": {
        "short": "Backstepping",
        "title": "Backstepping",
        "description": "Konum hatasından sanal bir hız komutu üretir ve ikinci adımda bu hızı kararlı biçimde izler.",
        "color": "#a78bfa",
    },
    "sliding_mode": {
        "short": "Sliding Mode",
        "title": "Kayma Kipli Kontrol (SMC)",
        "description": "Belirsizlik ve bozuculara dayanıklı bir kayma yüzeyi kullanır; yumuşak tanh sınır katmanı titreşimi azaltır.",
        "color": "#fb7185",
    },
    "nmpc": {
        "short": "Nonlinear MPC",
        "title": "Doğrusal Olmayan Model Öngörülü Kontrol (NMPC)",
        "description": "Kısa ufukta aday kontrol girişlerini doğrusal olmayan model üzerinde öngörür ve en düşük maliyetliyi seçer.",
        "color": "#34d399",
    },
}


@dataclass
class SimulationResult:
    controller: str
    time: np.ndarray
    target_position: np.ndarray
    uav_position: np.ndarray
    uav_velocity: np.ndarray
    control: np.ndarray
    error: np.ndarray
    distance_error: np.ndarray
    metrics: Dict[str, float]


def target_state(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target position, velocity and acceleration at time *t*."""

    # A smooth, non-planar path: ellipse + slow drift + vertical oscillation.
    x = 22.0 * np.cos(0.105 * t) + 0.18 * t
    y = 16.0 * np.sin(0.135 * t) + 3.0 * np.sin(0.045 * t)
    z = 14.0 + 3.8 * np.sin(0.090 * t) + 1.4 * np.sin(0.230 * t)

    vx = -22.0 * 0.105 * np.sin(0.105 * t) + 0.18
    vy = 16.0 * 0.135 * np.cos(0.135 * t) + 3.0 * 0.045 * np.cos(0.045 * t)
    vz = 3.8 * 0.090 * np.cos(0.090 * t) + 1.4 * 0.230 * np.cos(0.230 * t)

    ax = -22.0 * 0.105**2 * np.cos(0.105 * t)
    ay = -16.0 * 0.135**2 * np.sin(0.135 * t) - 3.0 * 0.045**2 * np.sin(0.045 * t)
    az = -3.8 * 0.090**2 * np.sin(0.090 * t) - 1.4 * 0.230**2 * np.sin(0.230 * t)
    return np.array([x, y, z]), np.array([vx, vy, vz]), np.array([ax, ay, az])


def wind_field(position: np.ndarray, t: float) -> np.ndarray:
    """Unknown nonlinear disturbance acceleration in m/s²."""

    x, y, z = position
    return np.array(
        [
            0.32 * np.sin(0.17 * t + 0.035 * y),
            0.27 * np.cos(0.14 * t + 0.030 * x),
            0.18 * np.sin(0.21 * t + 0.025 * z),
        ]
    )


def nonlinear_acceleration(position: np.ndarray, velocity: np.ndarray, control: np.ndarray, t: float) -> np.ndarray:
    drag = -DRAG_COEFFICIENT * np.linalg.norm(velocity) * velocity
    return control + drag + wind_field(position, t)


def _compensation(position: np.ndarray, velocity: np.ndarray, t: float, wind_fraction: float) -> np.ndarray:
    drag_cancel = DRAG_COEFFICIENT * np.linalg.norm(velocity) * velocity
    return drag_cancel - wind_fraction * wind_field(position, t)


class Controller:
    def __init__(self) -> None:
        self.previous_control = np.zeros(3)

    def command(self, position: np.ndarray, velocity: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def _finish(self, raw: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(raw))
        if norm > MAX_ACCELERATION:
            raw = raw * (MAX_ACCELERATION / norm)
        self.previous_control = raw.copy()
        return raw


class GainScheduledPID(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.integral = np.zeros(3)

    def command(self, position: np.ndarray, velocity: np.ndarray, t: float) -> np.ndarray:
        target, target_velocity, target_acceleration = target_state(t)
        e_position = target - position
        e_velocity = target_velocity - velocity
        error_size = float(np.linalg.norm(e_position))
        # Gain scheduling makes the feedback law nonlinear while anti-windup
        # prevents the integral state from growing during actuator saturation.
        kp = 0.82 + 0.46 * np.tanh(error_size / 4.0)
        kd = 1.25 + 0.42 * np.tanh(error_size / 3.0)
        ki = 0.045
        self.integral = np.clip(self.integral + e_position * DT, -6.0, 6.0)
        raw = (
            target_acceleration
            + kp * e_position
            + kd * e_velocity
            + ki * self.integral
            + _compensation(position, velocity, t, 0.48)
        )
        return self._finish(raw)


class Backstepping(Controller):
    def command(self, position: np.ndarray, velocity: np.ndarray, t: float) -> np.ndarray:
        target, target_velocity, target_acceleration = target_state(t)
        e1 = target - position
        virtual_velocity = target_velocity + 1.05 * e1
        e2 = virtual_velocity - velocity
        raw = (
            target_acceleration
            + 1.05 * (target_velocity - velocity)
            + 1.45 * e2
            + 0.78 * e1
            + _compensation(position, velocity, t, 0.60)
        )
        return self._finish(raw)


class SlidingMode(Controller):
    def command(self, position: np.ndarray, velocity: np.ndarray, t: float) -> np.ndarray:
        target, target_velocity, target_acceleration = target_state(t)
        e = position - target
        e_dot = velocity - target_velocity
        lam = 1.15
        surface = e_dot + lam * e
        raw = (
            target_acceleration
            - lam * e_dot
            - 2.15 * np.tanh(surface / 0.55)
            + _compensation(position, velocity, t, 0.35)
        )
        return self._finish(raw)


class NonlinearMPC(Controller):
    """Small deterministic shooting MPC suitable for a responsive demo UI."""

    OFFSETS = np.array(
        [
            [dx, dy, dz]
            for dx in (-1.5, 0.0, 1.5)
            for dy in (-1.5, 0.0, 1.5)
            for dz in (-1.0, 0.0, 1.0)
        ],
        dtype=float,
    )

    def command(self, position: np.ndarray, velocity: np.ndarray, t: float) -> np.ndarray:
        target, target_velocity, target_acceleration = target_state(t)
        nominal = (
            target_acceleration
            + 1.05 * (target - position)
            + 1.55 * (target_velocity - velocity)
            + _compensation(position, velocity, t, 0.55)
        )
        candidates = np.vstack((nominal, self.previous_control, nominal + self.OFFSETS))
        norms = np.linalg.norm(candidates, axis=1)
        over = norms > MAX_ACCELERATION
        candidates[over] *= (MAX_ACCELERATION / norms[over])[:, None]

        p = np.repeat(position[None, :], len(candidates), axis=0)
        v = np.repeat(velocity[None, :], len(candidates), axis=0)
        costs = np.zeros(len(candidates))
        horizon_dt = 0.14
        for step in range(1, 11):
            future_t = t + step * horizon_dt
            drag = -DRAG_COEFFICIENT * np.linalg.norm(v, axis=1)[:, None] * v
            # The predictor deliberately uses only 55% of the wind model.
            wind = 0.55 * wind_field(np.mean(p, axis=0), future_t)
            v = v + horizon_dt * (candidates + drag + wind)
            p = p + horizon_dt * v
            future_target, future_velocity, _ = target_state(future_t)
            position_cost = np.sum((p - future_target) ** 2, axis=1)
            velocity_cost = np.sum((v - future_velocity) ** 2, axis=1)
            costs += (1.0 + 0.10 * step) * position_cost + 0.18 * velocity_cost

        costs += 0.07 * np.sum(candidates**2, axis=1)
        costs += 0.12 * np.sum((candidates - self.previous_control) ** 2, axis=1)
        return self._finish(candidates[int(np.argmin(costs))])


CONTROLLERS = {
    "pid": GainScheduledPID,
    "backstepping": Backstepping,
    "sliding_mode": SlidingMode,
    "nmpc": NonlinearMPC,
}


def _metrics(error: np.ndarray, distance: np.ndarray, control: np.ndarray, time: np.ndarray) -> Dict[str, float]:
    evaluation_mask = time >= WARMUP_SEC
    if not np.any(evaluation_mask):
        evaluation_mask = np.ones(len(time), dtype=bool)
    eval_error = error[evaluation_mask]
    eval_distance = distance[evaluation_mask]
    eval_control = control[evaluation_mask]
    axis_rmse = np.sqrt(np.mean(eval_error**2, axis=0))
    total_rmse = float(np.sqrt(np.mean(eval_distance**2)))
    tolerance = TRACK_TOLERANCE
    tracking_rate = float(100.0 * np.mean(eval_distance <= tolerance))
    # Smooth 0-100 score: 100 for zero error, still sensitive outside tolerance.
    accuracy = float(100.0 * np.mean(np.exp(-((eval_distance / ACCURACY_SCALE) ** 2))))
    within = distance <= tolerance
    settling_time = DURATION
    # First instant after which 90% of the next 5 seconds is inside tolerance.
    window = max(1, int(5.0 / DT))
    for index in range(len(within) - window):
        if float(np.mean(within[index : index + window])) >= 0.90:
            settling_time = float(time[index])
            break
    return {
        "rmse_total": total_rmse,
        "rmse_x": float(axis_rmse[0]),
        "rmse_y": float(axis_rmse[1]),
        "rmse_z": float(axis_rmse[2]),
        "mean_error": float(np.mean(eval_distance)),
        "max_error": float(np.max(eval_distance)),
        "min_error": float(np.min(eval_distance)),
        "p95_error": float(np.percentile(eval_distance, 95)),
        "oscillation": float(np.mean(np.abs(np.diff(eval_distance)))) if len(eval_distance) > 1 else 0.0,
        "tracking_rate": tracking_rate,
        "accuracy": accuracy,
        "settling_time": settling_time,
        "control_effort": float(np.mean(np.sum(eval_control**2, axis=1))),
    }


class LiveSimulation:
    """Incremental physical simulation used directly by the real-time UI."""

    def __init__(self, controller_name: str, duration: float = DURATION, dt: float = DT):
        if controller_name not in CONTROLLERS:
            raise ValueError(f"Bilinmeyen kontrolcü: {controller_name}")
        self.controller_name = controller_name
        self.duration = duration
        self.dt = dt
        self.controller = CONTROLLERS[controller_name]()
        initial_target, _, _ = target_state(0.0)
        self.position = initial_target + np.array([-4.0, 3.0, -2.0])
        self.velocity = np.zeros(3)
        self.current_time = 0.0
        self.times = [0.0]
        self.targets = [initial_target.copy()]
        self.positions = [self.position.copy()]
        self.velocities = [self.velocity.copy()]
        self.controls = [np.zeros(3)]

    def step_until(self, elapsed: float) -> None:
        requested_time = min(float(elapsed), self.duration)
        while self.current_time + self.dt <= requested_time + 1e-9:
            t = self.current_time
            control = self.controller.command(self.position, self.velocity, t)
            acceleration = nonlinear_acceleration(self.position, self.velocity, control, t)
            # Semi-implicit Euler: velocity is updated before position.
            self.velocity = self.velocity + self.dt * acceleration
            self.position = self.position + self.dt * self.velocity
            self.current_time = round(self.current_time + self.dt, 10)
            target, _, _ = target_state(self.current_time)
            self.times.append(self.current_time)
            self.targets.append(target.copy())
            self.positions.append(self.position.copy())
            self.velocities.append(self.velocity.copy())
            self.controls.append(control.copy())

    def result(self) -> SimulationResult:
        times = np.asarray(self.times, dtype=float)
        targets = np.asarray(self.targets, dtype=float)
        positions = np.asarray(self.positions, dtype=float)
        velocities = np.asarray(self.velocities, dtype=float)
        controls = np.asarray(self.controls, dtype=float)
        error = targets - positions
        distance = np.linalg.norm(error, axis=1)
        return SimulationResult(
            controller=self.controller_name,
            time=times,
            target_position=targets,
            uav_position=positions,
            uav_velocity=velocities,
            control=controls,
            error=error,
            distance_error=distance,
            metrics=_metrics(error, distance, controls, times),
        )


def run_simulation(controller_name: str, duration: float = DURATION, dt: float = DT) -> SimulationResult:
    simulation = LiveSimulation(controller_name, duration=duration, dt=dt)
    simulation.step_until(duration)
    return simulation.result()


def run_comparison(controller_names: Iterable[str] | None = None) -> Dict[str, SimulationResult]:
    names = list(controller_names or CONTROLLERS.keys())
    return {name: run_simulation(name) for name in names}


if __name__ == "__main__":
    results = run_comparison()
    for name, result in results.items():
        print(
            f"{CONTROLLER_INFO[name]['short']:<34} "
            f"RMSE={result.metrics['rmse_total']:.2f} m  "
            f"takip={result.metrics['tracking_rate']:.1f}%  "
            f"doğruluk={result.metrics['accuracy']:.1f}%"
        )
