# Copyright 2026 <Your Name>
# SPDX-License-Identifier: Apache-2.0
"""Sensor-to-field dataset built from the LBM runs (pure numpy, no torch needed).

Input  x (C_in, h, w):  last H frames of the K surface pressure sensors, each spread over the grid by
                        nearest-sensor (Voronoi) tessellation (Fukami et al. 2021), plus obstacle mask,
                        signed distance to the obstacle and two coordinate channels.
Target y (3, h, w):     u, v, p (normalized) of the CURRENT frame.

Using a short sensor history lets the model infer the vortex-shedding phase and the gust level,
which a single pressure snapshot cannot disambiguate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGETS = ("u", "v", "p")


@dataclass
class Run:
    name: str
    y: np.ndarray  # (T, 3, h, w) float16, raw SI units
    p_sensors: np.ndarray  # (T, K) float32, raw Pa
    voronoi: np.ndarray  # (h, w) int, nearest sensor index per cell
    static: np.ndarray  # (4, h, w) float32: mask, sdf, x, z
    dt: float


def _downsample(a: np.ndarray, ds: int) -> np.ndarray:
    return a[..., ::ds, ::ds] if ds > 1 else a


def build_static(mask: np.ndarray, sensor_ij: np.ndarray, ds: int):
    """Voronoi map, mask, signed distance and coordinates on the (downsampled) grid."""
    m = _downsample(mask.astype(np.float32), ds)
    h, w = m.shape
    rows, cols = np.mgrid[0:h, 0:w]
    s = sensor_ij.astype(np.float32) / ds  # (K, 2) row, col
    d2 = (rows[None] - s[:, 0, None, None]) ** 2 + (
        cols[None] - s[:, 1, None, None]
    ) ** 2
    voronoi = d2.argmin(0)
    solid = np.argwhere(mask > 0)
    cy, cx = solid.mean(0) / ds
    radius = np.sqrt(len(solid) / np.pi) / ds
    sdf = (np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2) - radius) / max(h, w)
    xs = np.broadcast_to(np.linspace(0, 1, w, dtype=np.float32), (h, w))
    zs = np.broadcast_to(np.linspace(0, 1, h, dtype=np.float32)[:, None], (h, w))
    static = np.stack([m, sdf.astype(np.float32), xs, zs]).astype(np.float32)
    return voronoi, static


def load_run(path: str | Path, ds: int = 2) -> Run:
    d = np.load(path)
    y = np.stack([_downsample(d[k], ds) for k in TARGETS], axis=1).astype(np.float16)
    voronoi, static = build_static(d["mask"], d["sensor_ij"], ds)
    return Run(
        Path(path).stem,
        y,
        d["p_sensors"].astype(np.float32),
        voronoi,
        static,
        float(d["dt"]),
    )


@dataclass
class Normalizer:
    y_mean: np.ndarray  # (3,)
    y_std: np.ndarray  # (3,)
    p_mean: float
    p_std: float

    @classmethod
    def fit(cls, runs: list[Run]) -> "Normalizer":
        ys = np.concatenate([r.y[::5].astype(np.float32) for r in runs])
        ps = np.concatenate([r.p_sensors for r in runs])
        return cls(
            ys.mean((0, 2, 3)),
            ys.std((0, 2, 3)) + 1e-6,
            float(ps.mean()),
            float(ps.std() + 1e-6),
        )

    def to_dict(self) -> dict:
        return {
            "y_mean": self.y_mean.tolist(),
            "y_std": self.y_std.tolist(),
            "p_mean": self.p_mean,
            "p_std": self.p_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(
            np.array(d["y_mean"], np.float32),
            np.array(d["y_std"], np.float32),
            d["p_mean"],
            d["p_std"],
        )


def make_input(p_hist: np.ndarray, run: Run, norm: Normalizer) -> np.ndarray:
    """p_hist (H, K) raw sensor pressures, oldest first -> x (H + 4, h, w) float32."""
    p = (p_hist - norm.p_mean) / norm.p_std
    fields = p[:, run.voronoi]  # (H, h, w)
    return np.concatenate([fields.astype(np.float32), run.static], axis=0)


class SensorFieldDataset:
    """Index = (run, frame). Noise and sensor dropout are applied on the fly when `train=True`."""

    def __init__(
        self,
        runs: list[Run],
        norm: Normalizer,
        history: int = 8,
        train: bool = True,
        noise: float = 0.02,
        dropout: float = 0.05,
        seed: int = 0,
    ):
        self.runs, self.norm, self.h, self.train = runs, norm, history, train
        self.noise, self.dropout = noise, dropout
        self.rng = np.random.default_rng(seed)
        self.index = [
            (ri, t) for ri, r in enumerate(runs) for t in range(history - 1, len(r.y))
        ]

    def __len__(self) -> int:
        return len(self.index)

    @property
    def in_channels(self) -> int:
        return self.h + 4

    def get(self, i: int):
        ri, t = self.index[i]
        run = self.runs[ri]
        p_hist = run.p_sensors[t - self.h + 1 : t + 1].copy()
        if self.train:
            p_hist += self.rng.normal(0, self.noise * self.norm.p_std, p_hist.shape)
            dead = self.rng.random(p_hist.shape[1]) < self.dropout
            p_hist[:, dead] = self.norm.p_mean  # failed sensor reads "no signal"
        x = make_input(p_hist, run, self.norm)
        y = (
            run.y[t].astype(np.float32) - self.norm.y_mean[:, None, None]
        ) / self.norm.y_std[:, None, None]
        return x, y.astype(np.float32)

    def batches(self, batch_size: int, shuffle: bool = True):
        order = self.rng.permutation(len(self)) if shuffle else np.arange(len(self))
        for s in range(0, len(order), batch_size):
            items = [self.get(i) for i in order[s : s + batch_size]]
            yield np.stack([a for a, _ in items]), np.stack([b for _, b in items])


def split_runs(paths: list[Path], n_val: int = 2):
    paths = sorted(paths)
    if len(paths) <= n_val:
        raise ValueError(f"need more than {n_val} runs, found {len(paths)}")
    return paths[:-n_val], paths[-n_val:]
