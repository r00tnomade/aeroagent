# Copyright 2026 <Your Name>
# SPDX-License-Identifier: Apache-2.0
"""Core result: does the FNO feedforward reduce tracking error in the gusty wake?

  python -m eval.ablation                              # held-out run, hover in the wake 1.2 m behind the cylinder
  python -m eval.ablation --run data/lbm/run_020.npz --x 1.6

Three controllers, same seeded scenario, same ground-truth LBM wind applied by the simulator:
  pid        DSLPIDControl only (baseline)
  fno_ff     + drag feedforward from the FNO wind estimate (AeroAgent Tier 1)
  oracle_ff  + drag feedforward from the TRUE wind (upper bound: perfect flow knowledge)

Writes results/ablation_<run>.json, docs/figures/ablation_tracking.png, docs/figures/wind_at_drone.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MODES = ("pid", "fno_ff", "oracle_ff")
LABELS = {
    "pid": "PID only",
    "fno_ff": "PID + FNO feedforward (AeroAgent)",
    "oracle_ff": "PID + true-wind feedforward (upper bound)",
}


def default_run() -> Path:
    runs = sorted(Path("data/lbm").glob("run_*.npz"))
    for name in (
        "run_020.npz",
        "run_019.npz",
    ):  # held out by the first training split (runs 0-18 trained)
        if (Path("data/lbm") / name).exists():
            return Path("data/lbm") / name
    return runs[-1]


def hover_target(run_path: Path, x_behind: float) -> np.ndarray:
    d = np.load(run_path)
    solid = np.argwhere(d["mask"] > 0)
    zc = float(d["z"][int(round(solid[:, 0].mean()))])
    xc = float(d["x"][int(round(solid[:, 1].mean()))])
    return np.array([xc + x_behind, 0.0, zc])


def run_mode(
    mode: str,
    run_path: Path,
    ckpt: Path,
    target: np.ndarray,
    duration: float,
    reflex_hz: float,
    estimator=None,
) -> dict:
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    from control.reflex import drag_force, feedforward_target_vel
    from sim.wind_aviary import GridWind, WindAviary

    wind_gt = GridWind(str(run_path))
    env = WindAviary(
        wind=wind_gt,
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=target.reshape(1, 3),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=240,
        gui=False,
        show_wind=False,
    )
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    d_coeff = np.asarray(ctrl.D_COEFF_FOR, float)
    obs, _ = env.reset(seed=42)
    every = max(1, int(round(env.CTRL_FREQ / reflex_hz)))
    dv = np.zeros(3)
    log = {k: [] for k in ("t", "err", "pos", "wind_true", "wind_hat", "tick_ms")}
    steps = int(duration * env.CTRL_FREQ)
    for k in range(steps):
        t = k * env.CTRL_TIMESTEP
        state = obs[0]
        pos, vel = state[0:3], state[10:13]
        if k % every == 0:  # Tier 1 reflex tick
            t0 = time.perf_counter()
            w_true = wind_gt.sample(pos, t)
            if mode == "pid":
                w_hat = None
                dv = np.zeros(3)
            elif mode == "oracle_ff":
                w_hat = w_true
                dv = feedforward_target_vel(drag_force(w_true, vel, env.cd_a), d_coeff)
            else:
                w_hat = estimator.estimate(pos, t).wind
                dv = feedforward_target_vel(drag_force(w_hat, vel, env.cd_a), d_coeff)
            log["tick_ms"].append((time.perf_counter() - t0) * 1e3)
            log["wind_true"].append(w_true)
            log["wind_hat"].append(w_hat if w_hat is not None else np.full(3, np.nan))
        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target,
            target_rpy=np.zeros(3),
            target_vel=dv,
        )
        obs, _, _, _, _ = env.step(rpm.reshape(1, 4))
        log["t"].append(t + env.CTRL_TIMESTEP)
        log["pos"].append(obs[0][0:3].copy())
        log["err"].append(float(np.linalg.norm(obs[0][0:3] - target)))
    env.close()
    out = {k: np.asarray(v) for k, v in log.items()}
    out["tick_t"] = np.arange(len(out["tick_ms"])) * every * env.CTRL_TIMESTEP
    return out


def metrics(r: dict, gust_mask: np.ndarray, settle_s: float) -> dict:
    keep = r["t"] >= settle_s
    e = r["err"][keep]
    g = gust_mask[keep]
    m = {
        "rms_cm": float(np.sqrt(np.mean(e**2)) * 100),
        "max_cm": float(e.max() * 100),
        "rms_gust_cm": float(np.sqrt(np.mean(e[g] ** 2)) * 100) if g.any() else None,
        "tick_ms_p50": float(np.median(r["tick_ms"])),
        "tick_ms_p95": float(np.percentile(r["tick_ms"], 95)),
    }
    wh, wt = r["wind_hat"], r["wind_true"]
    if not np.isnan(wh).all():
        diff = (wh - wt)[:, [0, 2]]
        m["wind_err_rms_ms"] = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        m["wind_true_rms_ms"] = float(
            np.sqrt(np.mean(np.sum(wt[:, [0, 2]] ** 2, axis=1)))
        )
        fluct = wt[:, [0, 2]] - wt[:, [0, 2]].mean(0)
        m["wind_fluct_rms_ms"] = float(np.sqrt(np.mean(np.sum(fluct**2, axis=1))))
    return m


def plot(
    results: dict,
    gust_t: np.ndarray,
    gust: np.ndarray,
    run_name: str,
    target,
    mets: dict,
    tag: str,
) -> None:
    import matplotlib.pyplot as plt

    from viz.make_figures import AQUA, BLUE, GRAY, INK2, ORANGE, _save  # shared style

    colors = {"pid": GRAY, "fno_ff": BLUE, "oracle_ff": AQUA}
    fig, (a0, a1) = plt.subplots(
        2,
        1,
        figsize=(9.5, 5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    a0.fill_between(gust_t, 0, gust, color=GRAY, linewidth=0)
    a0.set_ylim(0, 1.05)
    a0.set_yticks([0, 1])
    a0.set_ylabel("gust\nfactor")
    a0.set_title("Inlet gust")
    for m in MODES:
        r = results[m]
        lab = f"{LABELS[m]}: RMS {mets[m]['rms_cm']:.1f} cm, max {mets[m]['max_cm']:.1f} cm"
        a1.plot(
            r["t"],
            r["err"] * 100,
            color=colors[m],
            label=lab,
            linewidth=2.4 if m == "fno_ff" else 1.8,
        )
    a1.set_ylabel("position error [cm]")
    a1.set_xlabel("time [s]")
    a1.set_title(
        f"Hover in the cylinder wake at x = {target[0]:.2f} m, z = {target[2]:.2f} m"
    )
    a1.legend(loc="upper left", fontsize=8.5)
    a1.set_ylim(bottom=0)
    _save(
        fig,
        f"ablation_tracking_{tag}",
        f"Held-out LBM run {run_name}. Same seed, same ground-truth wind for all three controllers.",
    )

    r = results["fno_ff"]
    fig, axes = plt.subplots(
        2, 1, figsize=(9.5, 4.8), sharex=True, constrained_layout=True
    )
    for ax, i, name in ((axes[0], 0, "u (streamwise)"), (axes[1], 2, "w (vertical)")):
        ax.plot(
            r["tick_t"],
            r["wind_true"][:, i],
            color=INK2,
            linewidth=1.4,
            label="true (LBM)",
        )
        ax.plot(
            r["tick_t"],
            r["wind_hat"][:, i],
            color=BLUE,
            linewidth=2.0,
            label="FNO estimate",
        )
        ax.set_ylabel(f"{name} [m/s]")
    axes[0].set_title(
        "Wind at the drone: FNO estimate from 32 surface sensors vs. truth"
    )
    axes[0].legend(loc="upper left", ncol=2, fontsize=8.5)
    axes[1].set_xlabel("time [s]")
    m = mets["fno_ff"]
    _save(
        fig,
        f"wind_at_drone_{tag}",
        f"RMS wind error {m['wind_err_rms_ms']:.2f} m/s vs. wind fluctuation RMS "
        f"{m['wind_fluct_rms_ms']:.2f} m/s. Sampled at the 50 Hz reflex rate. Run {run_name}.",
    )
    _ = ORANGE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/fno.pt"))
    ap.add_argument(
        "--x",
        type=float,
        default=1.2,
        help="hover distance behind the cylinder centre [m]",
    )
    ap.add_argument("--duration", type=float, default=7.4)
    ap.add_argument("--reflex-hz", type=float, default=48.0)
    ap.add_argument(
        "--settle", type=float, default=0.5, help="seconds excluded from the metrics"
    )
    args = ap.parse_args()

    from control.reflex import FNOWindEstimator

    run_path = args.run or default_run()
    target = hover_target(run_path, args.x)
    print(f"run {run_path.name}  hover target {np.round(target, 2)}")
    est = FNOWindEstimator(args.ckpt, run_path)
    results, mets = {}, {}
    raw = np.load(run_path)
    for m in MODES:
        t0 = time.perf_counter()
        results[m] = run_mode(
            m, run_path, args.ckpt, target, args.duration, args.reflex_hz, est
        )
        frames = (results[m]["t"] / float(raw["dt"])).astype(int) % len(raw["gust"])
        gust_mask = raw["gust"][frames] > 0.05
        mets[m] = metrics(results[m], gust_mask, args.settle)
        print(
            f"  {m:<10} RMS {mets[m]['rms_cm']:5.1f} cm | max {mets[m]['max_cm']:5.1f} cm | "
            f"gust RMS {mets[m]['rms_gust_cm'] or 0:5.1f} cm | tick p95 {mets[m]['tick_ms_p95']:.2f} ms "
            f"({time.perf_counter() - t0:.0f} s)"
        )
    m = mets["fno_ff"]
    print(
        f"\nwind at drone: FNO error RMS {m['wind_err_rms_ms']:.2f} m/s  (true wind RMS {m['wind_true_rms_ms']:.2f}, "
        f"fluctuation RMS {m['wind_fluct_rms_ms']:.2f} m/s)"
    )
    if est.fno_ms_log:
        print(
            f"FNO forward ({est.dev}): p50 {np.median(est.fno_ms_log):.2f} ms  p95 {np.percentile(est.fno_ms_log, 95):.2f} ms"
        )
    base = mets["pid"]["rms_cm"]
    for k in ("fno_ff", "oracle_ff"):
        print(
            f"{LABELS[k]}: RMS error {100 * (1 - mets[k]['rms_cm'] / base):.0f} % lower than PID only"
        )

    Path("results").mkdir(exist_ok=True)
    tag = f"{run_path.stem}_x{args.x:.1f}".replace(".", "p")
    out = Path("results") / f"ablation_{tag}.json"
    out.write_text(
        json.dumps(
            {
                "run": run_path.name,
                "target": target.tolist(),
                "args": {k: str(v) for k, v in vars(args).items()},
                "metrics": mets,
                "fno_ms": est.fno_ms_log,
            },
            indent=2,
        )
    )
    print(f"saved {out}")
    gust_t = np.arange(len(raw["gust"])) * float(raw["dt"])
    keep = gust_t <= args.duration
    plot(results, gust_t[keep], raw["gust"][keep], run_path.stem, target, mets, tag)


if __name__ == "__main__":
    main()
