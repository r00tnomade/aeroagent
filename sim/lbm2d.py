# Copyright 2026 Sami Adam Moughli
# SPDX-License-Identifier: Apache-2.0
"""Minimal 2D Lattice-Boltzmann solver (D2Q9, BGK) for gusty flow past a cylinder.

Pure numpy, no extra dependencies. Produces the ground-truth flow fields that
(a) the simulator applies to the drone (sim.wind_aviary.GridWind) and
(b) the FNO is trained to reconstruct from sparse sensors.

Lattice units inside the solver; everything written to disk is converted to SI.
Note: the lattice Reynolds number (~100-300) is far below real drone-scale air flow.
This is a deliberate simplification for a fast, visually clear vortex street.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# D2Q9 lattice: velocities, weights, opposite directions
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
LEFT = np.array(
    [3, 6, 7]
)  # directions with cx = -1 (enter the domain through the outlet)
RHO_AIR = 1.225


@dataclass
class LBMConfig:
    nx: int = 256  # cells along the flow (world x)
    ny: int = 128  # cells across (world z)
    re: float = 150.0  # Reynolds number based on cylinder diameter
    u_lat: float = 0.08  # inflow speed in lattice units (keep <= 0.1 for stability)
    radius: float = 9.0  # cylinder radius in cells
    cyl_x: float = 64.0  # cylinder centre (cells)
    cyl_y: float = 64.0
    # gust at the inlet: u_in = U0 * (1 + gust_u * g(t)), v_in = U0 * gust_v * g(t), g = 1-cos pulse
    gust_u: float = 0.5
    gust_v: float = 0.2
    gust_start: int = 6000  # lattice steps
    gust_len: int = 3000
    # physical mapping
    length_x: float = 4.0  # m, domain length -> dx = length_x / nx
    x0: float = -1.0  # m, world x of the left edge
    z0: float = 0.0  # m, world z of the bottom edge
    u_phys: float = 2.0  # m/s that corresponds to u_lat
    n_sensors: int = 32
    seed: int = 0
    extra: dict = field(default_factory=dict)


class LBM2D:
    def __init__(self, cfg: LBMConfig):
        self.cfg = cfg
        d = 2 * cfg.radius
        nu = cfg.u_lat * d / cfg.re
        self.tau = 3 * nu + 0.5
        if self.tau < 0.505:
            raise ValueError(
                f"tau={self.tau:.4f} too close to 0.5 (unstable). Lower re or raise radius."
            )
        y, x = np.mgrid[0 : cfg.ny, 0 : cfg.nx]
        self.solid = (x - cfg.cyl_x) ** 2 + (y - cfg.cyl_y) ** 2 <= cfg.radius**2
        rng = np.random.default_rng(cfg.seed)
        u0 = np.full((cfg.ny, cfg.nx), cfg.u_lat, dtype=np.float32)
        # small transverse perturbation so shedding starts early
        v0 = (
            1e-2 * cfg.u_lat * np.sin(2 * np.pi * y / cfg.ny) * rng.standard_normal()
        ).astype(np.float32)
        u0[self.solid] = 0
        v0[self.solid] = 0
        self.f = self.feq(np.ones_like(u0), u0, v0)
        self.step_count = 0
        self.sensor_ij = self._surface_sensors(cfg.n_sensors)

    # ----------------------------------------------------------------- core
    @staticmethod
    def feq(rho, u, v):
        cu = 3.0 * (CX[:, None, None] * u + CY[:, None, None] * v)
        usq = 1.5 * (u * u + v * v)
        return (W[:, None, None] * rho * (1 + cu + 0.5 * cu * cu - usq)).astype(
            np.float32
        )

    def macro(self):
        rho = self.f.sum(0)
        u = (CX[:, None, None] * self.f).sum(0) / rho
        v = (CY[:, None, None] * self.f).sum(0) / rho
        return rho, u, v

    def gust_factor(self, step: int) -> float:
        c = self.cfg
        tau = step - c.gust_start
        if 0 <= tau <= c.gust_len:
            return 0.5 * (1 - np.cos(2 * np.pi * tau / c.gust_len))
        return 0.0

    def step(self):
        c = self.cfg
        self.f[LEFT, :, -1] = self.f[LEFT, :, -2]  # outlet: zero-gradient
        rho, u, v = self.macro()
        g = self.gust_factor(self.step_count)
        u[:, 0] = c.u_lat * (1 + c.gust_u * g)  # inlet: prescribed velocity
        v[:, 0] = c.u_lat * c.gust_v * g
        rho[:, 0] = 1.0
        feq = self.feq(rho, u, v)
        self.f[:, :, 0] = feq[:, :, 0]
        fpost = self.f - (self.f - feq) / self.tau  # BGK collision
        fpost[:, self.solid] = self.f[OPP][:, self.solid]  # bounce-back on the cylinder
        for i in range(9):  # streaming (periodic in y)
            self.f[i] = np.roll(np.roll(fpost[i], CX[i], axis=1), CY[i], axis=0)
        self.step_count += 1

    # ----------------------------------------------------------------- outputs
    def _surface_sensors(self, k: int) -> np.ndarray:
        """K fluid cells just outside the cylinder surface, evenly spaced in angle. Returns (K, 2) [row, col]."""
        c = self.cfg
        ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
        r = c.radius + 1.5
        ij = np.stack(
            [np.round(c.cyl_y + r * np.sin(ang)), np.round(c.cyl_x + r * np.cos(ang))],
            1,
        ).astype(int)
        return ij

    def fields_si(self):
        """Current u, v [m/s], p [Pa, relative], vorticity [1/s] on the (ny, nx) grid (row = z, col = x)."""
        c = self.cfg
        rho, u, v = self.macro()
        s = c.u_phys / c.u_lat  # velocity scale
        dx = c.length_x / c.nx
        u_si, v_si = u * s, v * s
        u_si[self.solid] = 0
        v_si[self.solid] = 0
        p_si = RHO_AIR * (rho - 1.0) / 3.0 * s * s  # p = cs^2 (rho - rho0), cs^2 = 1/3
        p_si[self.solid] = 0
        dvdx = np.gradient(v_si, dx, axis=1)
        dudz = np.gradient(u_si, dx, axis=0)
        return (
            u_si.astype(np.float32),
            v_si.astype(np.float32),
            p_si.astype(np.float32),
            (dvdx - dudz).astype(np.float32),
        )

    def dt_si(self) -> float:
        c = self.cfg
        dx = c.length_x / c.nx
        return dx * c.u_lat / c.u_phys  # seconds per lattice step

    def axes_si(self):
        c = self.cfg
        dx = c.length_x / c.nx
        x = c.x0 + (np.arange(c.nx) + 0.5) * dx
        z = c.z0 + (np.arange(c.ny) + 0.5) * dx
        return x.astype(np.float32), z.astype(np.float32)


def run(
    cfg: LBMConfig, steps: int, save_every: int, warmup: int = 0, progress: bool = True
) -> dict:
    """Run the solver and return SI snapshots ready for np.savez."""
    sim = LBM2D(cfg)
    us, vs, ps, ws, gs = [], [], [], [], []
    for k in range(warmup + steps):
        sim.step()
        if k >= warmup and (k - warmup) % save_every == 0:
            u, v, p, w = sim.fields_si()
            if not np.isfinite(u).all():
                raise FloatingPointError(
                    f"LBM diverged at step {k} (tau={sim.tau:.4f})."
                )
            us.append(u)
            vs.append(v)
            ps.append(p)
            ws.append(w)
            gs.append(sim.gust_factor(sim.step_count))
        if progress and k % 2000 == 0:
            print(f"  step {k:>6}/{warmup + steps}")
    x, z = sim.axes_si()
    p_all = np.stack(ps)
    si, sj = sim.sensor_ij[:, 0], sim.sensor_ij[:, 1]
    f16 = (
        np.float16
    )  # halves file size; ample precision for training targets and wind lookup
    return {
        "u": np.stack(us).astype(f16),
        "v": np.stack(vs).astype(f16),
        "p": p_all.astype(f16),
        "omega": np.stack(ws).astype(f16),
        "mask": sim.solid.astype(np.uint8),
        "x": x,
        "z": z,
        "dt": np.float32(sim.dt_si() * save_every),
        "sensor_ij": sim.sensor_ij,
        "p_sensors": p_all[:, si, sj],
        "gust": np.array(gs, dtype=np.float32),
        "re": np.float32(cfg.re),
        "tau": np.float32(sim.tau),
        "seed": np.int32(cfg.seed),
    }
