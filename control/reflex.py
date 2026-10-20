# Copyright 2026 <Your Name>
# SPDX-License-Identifier: Apache-2.0
"""Tier 1 reflex: sensors -> FNO flow field -> wind at the drone -> drag-force feedforward (50 Hz).

The feedforward is injected through the derivative channel of gym-pybullet-drones' DSLPIDControl:
its position law is  F_cmd = P*e_pos + I*int(e_pos) + D*(v_target - v) + m*g,
so adding  dv = -F_wind_hat / D  to the target velocity adds exactly  -F_wind_hat  to F_cmd,
without touching the integral term (no wind-up). This mirrors the Neural-Fly recipe:
a learned aerodynamic estimate fed forward into an unchanged classical controller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RHO_AIR = 1.225


def drag_force(wind: np.ndarray, vel: np.ndarray, cd_a: float) -> np.ndarray:
    """Same quadratic drag model the simulator uses (assumed known here; on hardware it would be learned)."""
    v_rel = np.asarray(wind, float) - np.asarray(vel, float)
    return 0.5 * RHO_AIR * cd_a * np.linalg.norm(v_rel) * v_rel


def feedforward_target_vel(
    force_hat: np.ndarray, d_coeff: np.ndarray, max_dv: float = 1.5
) -> np.ndarray:
    """Velocity offset that makes DSLPIDControl add -force_hat to its commanded force."""
    dv = -np.asarray(force_hat, float) / np.asarray(d_coeff, float)
    return np.clip(dv, -max_dv, max_dv)


def bilinear(
    field: np.ndarray, xs: np.ndarray, zs: np.ndarray, x: float, z: float
) -> float:
    """Sample field[z, x] on ascending axes xs, zs; clamps to the domain edge."""
    xi = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    zi = int(np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2))
    tx = float(np.clip((x - xs[xi]) / (xs[xi + 1] - xs[xi]), 0.0, 1.0))
    tz = float(np.clip((z - zs[zi]) / (zs[zi + 1] - zs[zi]), 0.0, 1.0))
    f00, f01, f10, f11 = (
        field[zi, xi],
        field[zi, xi + 1],
        field[zi + 1, xi],
        field[zi + 1, xi + 1],
    )
    return float(
        (1 - tz) * ((1 - tx) * f00 + tx * f01) + tz * ((1 - tx) * f10 + tx * f11)
    )


@dataclass
class WindEstimate:
    wind: np.ndarray  # (3,) world frame [m/s] at the drone
    frame: int
    fno_ms: float  # model forward time (0 when served from cache)
    cached: bool


class FNOWindEstimator:
    """Runs the trained FNO on the sensor history of an LBM run and samples the wind at the drone.

    Sensors update with the LBM frame rate (40 Hz); the 50 Hz reflex reuses the last field when no new
    sensor frame has arrived. Later this class talks to the Nebius Serverless Endpoint instead of a local model.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        run_path: str | Path,
        device: str = "auto",
        sensor_noise: float = 0.0,
        delay_frames: int = 0,
        seed: int = 0,
    ):
        import torch

        from model.data import Normalizer, load_run
        from model.fno import build_model
        from model.train import pick_device, sync

        self.torch, self._sync = torch, sync
        self.dev = pick_device(device)
        c = torch.load(ckpt_path, map_location=self.dev, weights_only=False)
        self.model = (
            build_model(c["arch"], c["in_channels"], 3, **c["cfg"]).to(self.dev).eval()
        )
        self.model.load_state_dict(c["state_dict"])
        self.norm = Normalizer.from_dict(c["norm"])
        self.history = int(c["history"])
        self.run = load_run(run_path, c["downsample"])
        raw = np.load(run_path)
        ds = int(c["downsample"])
        self.xs, self.zs = raw["x"][::ds].astype(float), raw["z"][::ds].astype(float)
        self.n_frames = len(self.run.y)
        self._frame, self._uv = -1, None
        # robustness knobs: extra sensor noise (fraction of the sensor std) and transport delay in frames
        self.sensor_noise, self.delay_frames = float(sensor_noise), int(delay_frames)
        self._rng = np.random.default_rng(seed)
        self.fno_ms_log: list[float] = []
        with torch.no_grad():  # warm-up
            x = torch.zeros(
                1, c["in_channels"], *self.run.y.shape[-2:], device=self.dev
            )
            for _ in range(10):
                self.model(x)
            sync(self.dev)

    def frame_at(self, t: float) -> int:
        return int(t / self.run.dt) % self.n_frames

    def field(self, frame: int) -> tuple[np.ndarray, float, bool]:
        """Predicted (u, v) on the FNO grid for this sensor frame, in m/s."""
        if frame == self._frame:
            return self._uv, 0.0, True
        from model.data import make_input

        k = max(frame - self.delay_frames, self.history - 1)
        p_hist = self.run.p_sensors[k - self.history + 1 : k + 1]
        if self.sensor_noise > 0:
            p_hist = p_hist + self._rng.normal(
                0, self.sensor_noise * self.norm.p_std, p_hist.shape
            )
        x = make_input(p_hist, self.run, self.norm)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            y = self.model(self.torch.from_numpy(x[None]).to(self.dev))
            self._sync(self.dev)
            y = y[0].cpu().numpy()
        ms = (time.perf_counter() - t0) * 1e3
        self.fno_ms_log.append(ms)
        uv = y[:2] * self.norm.y_std[:2, None, None] + self.norm.y_mean[:2, None, None]
        self._frame, self._uv = frame, uv
        return uv, ms, False

    def estimate(self, pos: np.ndarray, t: float) -> WindEstimate:
        frame = self.frame_at(t)
        uv, ms, cached = self.field(frame)
        u = bilinear(uv[0], self.xs, self.zs, float(pos[0]), float(pos[2]))
        w = bilinear(uv[1], self.xs, self.zs, float(pos[0]), float(pos[2]))
        return WindEstimate(np.array([u, 0.0, w]), frame, ms, cached)
