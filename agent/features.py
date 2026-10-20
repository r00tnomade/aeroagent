# Copyright 2026 Sami Adam Moughli
# SPDX-License-Identifier: Apache-2.0
"""Compress the FNO flow field into ~30 numbers for Nemotron (it never sees raw grids)."""

from __future__ import annotations

from collections import deque

import numpy as np

from control.reflex import bilinear


class FeatureBuilder:
    def __init__(
        self,
        estimator,
        hover_point: np.ndarray,
        x_obstacle: float,
        geofence=((0.4, 2.9), (0.25, 1.75)),
        history_s: float = 3.0,
    ):
        self.est = estimator
        self.hover = np.asarray(hover_point, float)
        self.x_obstacle = float(x_obstacle)
        self.geofence = geofence
        self._inflow = deque(
            maxlen=max(2, int(history_s / estimator.run.dt))
        )  # per build() call (legacy)
        self._hist: deque[tuple[float, float]] = deque(
            maxlen=400
        )  # (t, u_in) at the reflex rate
        self.x_in = self.x_obstacle - 0.5
        self.conv = 0.85  # wake structures travel at ~0.85 of the inflow speed (frozen-flow, assumption)
        self.thr = 0.08  # gust level that counts as "gust present"
        self._sensor_cells = self._fluid_cells_near_sensors()

    # ------------------------------------------------------------------ gust tracking at the reflex rate
    def _u_in(self, u: np.ndarray) -> float:
        zs = self.est.zs
        return float(
            np.mean(
                [
                    self._sample(u, self.x_in, zz)
                    for zz in np.linspace(zs[0] + 0.2, zs[-1] - 0.2, 7)
                ]
            )
        )

    def observe(self, t: float) -> None:
        """Call every Tier 1 tick (after the estimator update): records the inflow speed history."""
        f = self.est.last_field
        if f is not None:
            self._hist.append((float(t), self._u_in(f[0])))

    def gust_timing(self, t: float, x_drone: float) -> dict:
        """Gust level upstream now, at the drone now (delayed by the convection time), arrival and remaining time."""
        if len(self._hist) < 3:
            return {
                "level_inflow": 0.0,
                "level_at_drone": 0.0,
                "eta_s": None,
                "remaining_s": None,
                "transit_s": None,
            }
        ts = np.array([h[0] for h in self._hist])
        us = np.array([h[1] for h in self._hist])
        recent = ts >= t - 3.0
        u_ref = max(float(np.min(us[recent])), 0.3)
        lv = np.maximum(0.0, us / u_ref - 1.0)
        transit = max(x_drone - self.x_in, 0.0) / max(self.conv * u_ref, 0.3)
        level_now = float(lv[-1])
        t_seen = t - transit  # inflow moment that is reaching the drone right now
        level_at = float(np.interp(t_seen, ts, lv)) if t_seen >= ts[0] else 0.0
        eta = remaining = None
        if level_at < self.thr and level_now >= self.thr:
            above = np.nonzero((ts >= t_seen) & (lv >= self.thr))[
                0
            ]  # first inflow sample above threshold
            eta = float(ts[above[0]] + transit - t) if above.size else transit
        if level_at >= self.thr:
            later = np.nonzero((ts > t_seen) & (lv < self.thr))[
                0
            ]  # when did the inflow calm down again?
            remaining = float(ts[later[0]] + transit - t) if later.size else transit
        return {
            "level_inflow": level_now,
            "level_at_drone": level_at,
            "eta_s": eta,
            "remaining_s": remaining,
            "transit_s": transit,
        }

    def _fluid_cells_near_sensors(self) -> np.ndarray:
        """For each sensor, the nearest NON-solid cell of the (downsampled) model grid."""
        ij = np.asarray(self.est.sensor_ij_ds, float)
        mask = getattr(getattr(self.est, "run", None), "static", None)
        if mask is None:
            return ij.astype(int)
        solid = mask[0] > 0.5
        rows, cols = np.nonzero(~solid)
        out = []
        for r, c in ij:
            k = np.argmin((rows - r) ** 2 + (cols - c) ** 2)
            out.append((rows[k], cols[k]))
        return np.array(out, int)

    def _sample(self, f: np.ndarray, x: float, z: float) -> float:
        return bilinear(f, self.est.xs, self.est.zs, x, z)

    def build(
        self,
        t: float,
        pos,
        vel,
        rpy_deg,
        target,
        last_intent: dict,
        state_age_ms: float = 0.0,
    ) -> dict:
        est = self.est
        f = est.last_field
        if f is None:
            est.estimate(np.asarray(pos), t)
            f = est.last_field
        u, w, p = f
        xs, zs = est.xs, est.zs
        x, z = float(pos[0]), float(pos[2])

        wind = [self._sample(u, x, z), 0.0, self._sample(w, x, z)]
        seg = np.linspace(max(xs[0], x - 1.0), x, 8)
        up = np.array(
            [[self._sample(u, xi, z), self._sample(w, xi, z)] for xi in seg]
        ).mean(0)

        # gust: upstream level now, level at the drone now (convection delay), arrival / remaining time
        if not self._hist:
            self.observe(t)
        g = self.gust_timing(t, x)
        rd = lambda v: None if v is None else round(float(v), 2)  # noqa: E731

        # vorticity peak near the drone
        dx = xs[1] - xs[0]
        vort = np.gradient(w, dx, axis=1) - np.gradient(u, dx, axis=0)
        X, Z = np.meshgrid(xs, zs)
        near = (X - x) ** 2 + (Z - z) ** 2 <= 0.4**2
        if near.any():
            idx = np.argmax(np.where(near, np.abs(vort), -1))
            iz, ix = np.unravel_index(idx, vort.shape)
            vpeak = {
                "mag": round(float(abs(vort[iz, ix])), 2),
                "pos": [round(float(xs[ix]), 2), 0.0, round(float(zs[iz]), 2)],
            }
            speed = np.hypot(u, w)[near]
            turb = float(np.std(speed) / max(np.mean(speed), 1e-3))
        else:
            vpeak, turb = None, 0.0

        # model-vs-sensor consistency: predicted pressure at the sensor cells vs. what the sensors read
        ij = self._sensor_cells
        p_pred = p[ij[:, 0], ij[:, 1]]
        p_meas = est.last_p_sensors
        unc = (
            float(np.sqrt(np.mean((p_pred - p_meas) ** 2)) / (np.std(p_meas) + 1e-6))
            if p_meas is not None
            else 1.0
        )

        (gx0, gx1), (gz0, gz1) = self.geofence
        margin = min(x - gx0, gx1 - x, z - gz0, gz1 - z)
        r2 = lambda v: [round(float(a), 2) for a in v]  # noqa: E731
        return {
            "t": round(float(t), 2),
            "state_age_ms": round(float(state_age_ms)),
            "pose": {"pos": r2(pos), "vel": r2(vel), "rpy_deg": r2(rpy_deg)},
            "hover_err_m": round(
                float(np.linalg.norm(np.asarray(pos) - self.hover)), 3
            ),
            "track_err_m": round(
                float(np.linalg.norm(np.asarray(pos) - np.asarray(target))), 3
            ),
            "flow": {
                "wind_at_drone": r2(wind),
                "wind_upstream_mean": [
                    round(float(up[0]), 2),
                    0.0,
                    round(float(up[1]), 2),
                ],
                "gust_level": round(g["level_inflow"], 2),
                "gust_at_drone": round(g["level_at_drone"], 2),
                "gust_eta_s": rd(g["eta_s"]),
                "gust_remaining_s": rd(g["remaining_s"]),
                "gust_transit_s": rd(g["transit_s"]),
                "vorticity_peak": vpeak,
                "turb_intensity": round(turb, 2),
                "fno_uncertainty": round(unc, 2),
            },
            "mission": {
                "hover_point": r2(self.hover),
                "offset_m": r2(np.asarray(target) - self.hover),
                "geofence_margin_m": round(float(margin), 2),
            },
            "last_intent": last_intent,
        }

    def waypoint_ok(self, offset, max_offset: float = 0.25) -> bool:
        """Hover mission: offsets beyond 0.25 m from the hover point are refused (validator sets them to 0)."""
        if np.max(np.abs(offset)) > max_offset:
            return False
        (gx0, gx1), (gz0, gz1) = self.geofence
        x, z = self.hover[0] + offset[0], self.hover[2] + offset[2]
        return gx0 <= x <= gx1 and gz0 <= z <= gz1 and abs(x - self.x_obstacle) > 0.4
