# Copyright 2026 <Your Name>
# SPDX-License-Identifier: Apache-2.0
"""Gate test for Block 3: a PID-controlled Crazyflie hovers at 1 m while wind pushes it.

Run from the repo root:
  python -m sim.hover_wind_demo --wind none            # sanity check, error should stay ~0
  python -m sim.hover_wind_demo --wind gust --gui      # watch the gust push the drone
  python -m sim.hover_wind_demo --wind const

This is the "PID only" baseline. The FNO feedforward has to beat these numbers.
`simulate()` is also used by viz/make_figures.py.
"""

import argparse
import time

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

from sim.wind_aviary import ConstantWind, GustWind, WindAviary, WindField


def make_wind(kind: str) -> WindField:
    if kind == "const":
        return ConstantWind((2.0, 0.0, 0.0))
    if kind == "gust":
        return GustWind(
            mean=(1.0, 0.0, 0.0), gust_amp=(3.0, 0.0, 0.8), t_start=2.0, duration=1.5
        )
    return WindField()


def simulate(kind: str, duration: float = 8.0, gui: bool = False) -> dict:
    """Hover at (0, 0, 1) under the given wind. Returns time series as numpy arrays."""
    target = np.array([0.0, 0.0, 1.0])
    env = WindAviary(
        wind=make_wind(kind),
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=target.reshape(1, 3),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=240,
        gui=gui,
    )
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, _ = env.reset(seed=42)
    steps = int(duration * env.CTRL_FREQ)
    t, pos, err, force, wind = [], [], [], [], []
    start = time.time()
    for k in range(steps):
        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target,
            target_rpy=np.zeros(3),
        )
        obs, _, _, _, info = env.step(rpm.reshape(1, 4))
        t.append((k + 1) * env.CTRL_TIMESTEP)
        pos.append(obs[0][0:3].copy())
        err.append(np.linalg.norm(obs[0][0:3] - target))
        if info["wind_force"] is not None:
            force.append(info["wind_force"][0].copy())
            wind.append(info["wind"][0].copy())
        else:
            force.append(np.zeros(3))
            wind.append(np.zeros(3))
        if gui:
            sync(k, start, env.CTRL_TIMESTEP)
    env.close()
    return {
        "t": np.array(t),
        "pos": np.array(pos),
        "err": np.array(err),
        "force": np.array(force),
        "wind": np.array(wind),
        "target": target,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind", choices=["none", "const", "gust"], default="gust")
    ap.add_argument("--duration", type=float, default=8.0, help="seconds")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    r = simulate(args.wind, args.duration, args.gui)
    wall = time.perf_counter() - t0
    errs = r["err"]
    print(
        f"wind={args.wind}  duration={args.duration:.1f} s  ({len(errs)} steps, {wall:.1f} s wall time)"
    )
    print(
        f"position error  RMS {np.sqrt(np.mean(errs**2)) * 100:.1f} cm | "
        f"max {errs.max() * 100:.1f} cm | final {errs[-1] * 100:.1f} cm"
    )
    print(
        f"max wind force  {np.linalg.norm(r['force'], axis=1).max() * 1000:.1f} mN  (CF2X weight ~265 mN)"
    )


if __name__ == "__main__":
    main()
