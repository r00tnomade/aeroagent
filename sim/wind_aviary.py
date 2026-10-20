# Copyright 2026 Sami Adam Moughli
# SPDX-License-Identifier: Apache-2.0
"""Wind injection for gym-pybullet-drones (Tier 0 simulator).

The simulator applies the GROUND-TRUTH wind to the drone as an aerodynamic drag force.
The FNO / controller must never read from these wind fields directly (see AGENTS.md, section 2).
"""

from __future__ import annotations

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import Physics

RHO_AIR = 1.225  # kg/m^3


# --------------------------------------------------------------------------- wind fields
class WindField:
    """Base class: no wind. `sample` returns the wind velocity [m/s] at world position `pos` and time `t`."""

    def sample(self, pos: np.ndarray, t: float) -> np.ndarray:
        return np.zeros(3)


class ConstantWind(WindField):
    def __init__(self, velocity=(2.0, 0.0, 0.0)):
        self.velocity = np.asarray(velocity, dtype=float)

    def sample(self, pos: np.ndarray, t: float) -> np.ndarray:
        return self.velocity.copy()


class GustWind(WindField):
    """Steady wind plus a 1-cos gust pulse that travels with the mean wind (frozen-flow assumption).

    The gust front reaches x = 0 at `t_start` and passes a point in `duration` seconds.
    Points further downstream (larger x) see the gust later, which gives the spatial structure
    the FNO features (`gust_eta_s`) are later built on.
    """

    def __init__(
        self, mean=(1.0, 0.0, 0.0), gust_amp=(3.0, 0.0, 0.8), t_start=2.0, duration=1.5
    ):
        self.mean = np.asarray(mean, dtype=float)
        self.amp = np.asarray(gust_amp, dtype=float)
        self.t_start = t_start
        self.duration = duration
        self.convect = max(abs(self.mean[0]), 0.1)  # gust front speed along x [m/s]

    def sample(self, pos: np.ndarray, t: float) -> np.ndarray:
        tau = t - self.t_start - pos[0] / self.convect
        if 0.0 <= tau <= self.duration:
            g = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau / self.duration))
        else:
            g = 0.0
        return self.mean + g * self.amp


class GridWind(WindField):
    """2D time-resolved field from an LBM run, mapped onto the world x-z plane (y is ignored).

    Expected .npz keys: u, v -> (T, H, W) [m/s]; x -> (W,) [m]; z -> (H,) [m]; dt -> scalar [s].
    `u` is the x-component, `v` the vertical (z) component. Outside the domain the inflow
    (first column) is used. Time is looped.
    """

    def __init__(self, path: str, x_offset: float = 0.0, z_offset: float = 0.0):
        d = np.load(path)
        self.u, self.v = d["u"].astype(np.float32), d["v"].astype(np.float32)
        if self.u.ndim == 2:
            self.u, self.v = self.u[None], self.v[None]
        self.x, self.z, self.dt = d["x"] + x_offset, d["z"] + z_offset, float(d["dt"])

    def _bilinear(self, f: np.ndarray, x: float, z: float) -> float:
        xi = np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 2)
        zi = np.clip(np.searchsorted(self.z, z) - 1, 0, len(self.z) - 2)
        tx = np.clip((x - self.x[xi]) / (self.x[xi + 1] - self.x[xi]), 0.0, 1.0)
        tz = np.clip((z - self.z[zi]) / (self.z[zi + 1] - self.z[zi]), 0.0, 1.0)
        f00, f01 = f[zi, xi], f[zi, xi + 1]
        f10, f11 = f[zi + 1, xi], f[zi + 1, xi + 1]
        return float(
            (1 - tz) * ((1 - tx) * f00 + tx * f01) + tz * ((1 - tx) * f10 + tx * f11)
        )

    def sample(self, pos: np.ndarray, t: float) -> np.ndarray:
        k = int(t / self.dt) % self.u.shape[0]
        x, z = float(pos[0]), float(pos[2])
        if not (self.x[0] <= x <= self.x[-1] and self.z[0] <= z <= self.z[-1]):
            zi = int(np.clip(np.searchsorted(self.z, z), 0, len(self.z) - 1))
            return np.array([self.u[k, zi, 0], 0.0, self.v[k, zi, 0]])
        return np.array(
            [self._bilinear(self.u[k], x, z), 0.0, self._bilinear(self.v[k], x, z)]
        )


# --------------------------------------------------------------------------- environment
class WindAviary(CtrlAviary):
    """CtrlAviary + quadratic aerodynamic drag from a WindField, applied at every physics sub-step.

    F = 0.5 * rho * Cd * A * |v_rel| * v_rel,  v_rel = wind - v_drone   (world frame, on link 4 = COM)
    Defaults roughly match a Crazyflie 2.x (mass 27 g): a 3 m/s gust gives ~0.05 N, ~18 % of its weight.
    """

    def __init__(
        self,
        wind: WindField | None = None,
        drag_coeff: float = 1.0,
        frontal_area: float = 0.0085,
        show_wind: bool = True,
        **kwargs,
    ):
        self.wind = wind or WindField()
        self.cd_a = drag_coeff * frontal_area
        self.show_wind = show_wind
        self.wind_at: np.ndarray | None = None
        self.wind_force: np.ndarray | None = None
        self._debug_line = -1
        super().__init__(**kwargs)
        if self.PHYSICS == Physics.DYN:
            raise ValueError(
                "WindAviary needs a PyBullet physics mode (Physics.PYB*), not Physics.DYN."
            )

    @property
    def sim_time(self) -> float:
        return self.step_counter * self.PYB_TIMESTEP

    def _physics(self, rpm, nth):
        super()._physics(rpm, nth)  # rotor thrust and torques
        self._apply_wind(nth)

    def _apply_wind(self, nth: int) -> None:
        if self.wind_at is None:
            self.wind_at = np.zeros((self.NUM_DRONES, 3))
            self.wind_force = np.zeros((self.NUM_DRONES, 3))
        body = self.DRONE_IDS[nth]
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=self.CLIENT)
        vel, _ = p.getBaseVelocity(body, physicsClientId=self.CLIENT)
        w = self.wind.sample(np.asarray(pos), self.sim_time)
        v_rel = w - np.asarray(vel)
        force = 0.5 * RHO_AIR * self.cd_a * np.linalg.norm(v_rel) * v_rel
        p.applyExternalForce(
            body,
            4,
            forceObj=force.tolist(),
            posObj=pos,
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT,
        )
        self.wind_at[nth], self.wind_force[nth] = w, force
        if self.GUI and self.show_wind and nth == 0:
            end = (np.asarray(pos) + 0.15 * w).tolist()  # 1 m/s -> 15 cm arrow
            self._debug_line = p.addUserDebugLine(
                pos,
                end,
                lineColorRGB=[0.1, 0.6, 1.0],
                lineWidth=3,
                replaceItemUniqueId=self._debug_line,
                physicsClientId=self.CLIENT,
            )

    def _computeInfo(self):
        if self.wind_at is None:
            return {"wind": None, "wind_force": None}
        return {"wind": self.wind_at.copy(), "wind_force": self.wind_force.copy()}
