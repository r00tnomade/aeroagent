# Copyright 2026 <Your Name>
# SPDX-License-Identifier: Apache-2.0
"""Gate test for Block 3: a PID-controlled Crazyflie hovers at 1 m while wind pushes it.

Run from the repo root:
  python -m sim.hover_wind_demo --wind none            # sanity check, error should stay ~0
  python -m sim.hover_wind_demo --wind gust --gui      # watch the gust push the drone
  python -m sim.hover_wind_demo --wind const

This is the "PID only" baseline. Tomorrow the FNO feedforward has to beat these numbers.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind", choices=["none", "const", "gust"], default="gust")
    ap.add_argument("--duration", type=float, default=8.0, help="seconds")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    target = np.array([0.0, 0.0, 1.0])
    env = WindAviary(
        wind=make_wind(args.wind),
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=target.reshape(1, 3),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=240,  # Tier 0 runs at 240 Hz
        gui=args.gui,
    )
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, info = env.reset(seed=42)

    errs, forces = [], []
    start = time.time()
    t0 = time.perf_counter()
    steps = int(args.duration * env.CTRL_FREQ)
    for k in range(steps):
        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target,
            target_rpy=np.zeros(3),
        )
        obs, _, _, _, info = env.step(rpm.reshape(1, 4))
        errs.append(np.linalg.norm(obs[0][0:3] - target))
        if info["wind_force"] is not None:
            forces.append(np.linalg.norm(info["wind_force"][0]))
        if args.gui:
            sync(k, start, env.CTRL_TIMESTEP)
    wall = time.perf_counter() - t0
    env.close()

    errs = np.array(errs)
    print(
        f"wind={args.wind}  duration={args.duration:.1f} s  ({steps} steps, {wall:.1f} s wall time)"
    )
    print(
        f"position error  RMS {np.sqrt(np.mean(errs**2)) * 100:.1f} cm | "
        f"max {errs.max() * 100:.1f} cm | final {errs[-1] * 100:.1f} cm"
    )
    if forces:
        print(f"max wind force  {max(forces) * 1000:.1f} mN  (CF2X weight ~265 mN)")


if __name__ == "__main__":
    main()
