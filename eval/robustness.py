# Copyright 2026 Sami Adam Moughli
# SPDX-License-Identifier: Apache-2.0
"""How fragile is the -80 %? Perturb one thing at a time and re-run the FNO-feedforward hover.

  python -m eval.robustness                 # ~1 min on an M1 Max

Perturbations (each vs. the same PID-only baseline):
  drag model     controller assumes Cd*A that is 30 % / 50 % wrong (the simulator keeps the truth)
  sensor noise   extra Gaussian noise on all 32 pressure taps (training used 2 %)
  delay          sensor -> estimate transport delay of 1 / 2 / 4 frames (25 / 50 / 100 ms)
  unseen run     the other held-out LBM run

Writes results/robustness.json and docs/figures/robustness.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from control.reflex import FNOWindEstimator
from eval.ablation import default_run, hover_target, metrics, run_mode

CONDITIONS = [
    # (group, label, kwargs for estimator, cd_scale)
    ("nominal", "nominal", {}, 1.0),
    ("drag model", "Cd·A −50 %", {}, 0.5),
    ("drag model", "Cd·A −30 %", {}, 0.7),
    ("drag model", "Cd·A +30 %", {}, 1.3),
    ("drag model", "Cd·A +50 %", {}, 1.5),
    ("sensor noise", "noise 5 %", {"sensor_noise": 0.05}, 1.0),
    ("sensor noise", "noise 10 %", {"sensor_noise": 0.10}, 1.0),
    ("sensor noise", "noise 20 %", {"sensor_noise": 0.20}, 1.0),
    ("delay", "delay 25 ms", {"delay_frames": 1}, 1.0),
    ("delay", "delay 50 ms", {"delay_frames": 2}, 1.0),
    ("delay", "delay 100 ms", {"delay_frames": 4}, 1.0),
]


def gust_mask_for(r: dict, raw) -> np.ndarray:
    frames = (r["t"] / float(raw["dt"])).astype(int) % len(raw["gust"])
    return raw["gust"][frames] > 0.05


def evaluate(
    run_path: Path, ckpt: Path, x: float, duration: float, settle: float, conds
) -> list[dict]:
    raw = np.load(run_path)
    target = hover_target(run_path, x)
    base_r = run_mode("pid", run_path, ckpt, target, duration, 48.0)
    base = metrics(base_r, gust_mask_for(base_r, raw), settle)
    ora_r = run_mode("oracle_ff", run_path, ckpt, target, duration, 48.0)
    oracle = metrics(ora_r, gust_mask_for(ora_r, raw), settle)
    rows = []
    from sim.wind_aviary import GridWind

    gw = GridWind(str(run_path))
    ts = np.arange(0, duration, float(raw["dt"]))
    mean_wind = np.mean([gw.sample(target, t) for t in ts], axis=0)
    mr = run_mode(
        "mean_ff", run_path, ckpt, target, duration, 48.0, mean_wind=mean_wind
    )
    mm = metrics(mr, gust_mask_for(mr, raw), settle)
    mred = 100 * (1 - mm["rms_cm"] / base["rms_cm"])
    rows.append(
        {
            "run": run_path.stem,
            "group": "control",
            "label": "mean wind only (no FNO)",
            "rms_cm": mm["rms_cm"],
            "max_cm": mm["max_cm"],
            "rms_gust_cm": mm["rms_gust_cm"],
            "reduction_pct": mred,
            "pid_rms_cm": base["rms_cm"],
            "pid_rms_gust_cm": base["rms_gust_cm"],
            "oracle_reduction_pct": 100 * (1 - oracle["rms_cm"] / base["rms_cm"]),
        }
    )
    print(
        f"  {run_path.stem}  {'mean wind only':<14} RMS {mm['rms_cm']:4.2f} cm  (−{mred:.0f} % vs PID)  "
        f"gust RMS {mm['rms_gust_cm']:.2f} cm vs PID {base['rms_gust_cm']:.2f} cm"
    )
    for group, label, est_kw, cd in conds:
        est = FNOWindEstimator(ckpt, run_path, **est_kw)
        r = run_mode(
            "fno_ff", run_path, ckpt, target, duration, 48.0, estimator=est, cd_scale=cd
        )
        m = metrics(r, gust_mask_for(r, raw), settle)
        red = 100 * (1 - m["rms_cm"] / base["rms_cm"])
        rows.append(
            {
                "run": run_path.stem,
                "group": group,
                "label": label,
                "rms_cm": m["rms_cm"],
                "max_cm": m["max_cm"],
                "rms_gust_cm": m["rms_gust_cm"],
                "reduction_pct": red,
                "wind_err_rms_ms": m.get("wind_err_rms_ms"),
                "pid_rms_cm": base["rms_cm"],
                "oracle_reduction_pct": 100 * (1 - oracle["rms_cm"] / base["rms_cm"]),
            }
        )
        print(
            f"  {run_path.stem}  {label:<14} RMS {m['rms_cm']:4.2f} cm  (−{red:.0f} % vs PID {base['rms_cm']:.2f} cm)  "
            f"gust RMS {m['rms_gust_cm']:.2f} cm  wind err {m.get('wind_err_rms_ms', float('nan')):.2f} m/s"
        )
    return rows


def plot(rows: list[dict], suffix: str = "") -> None:
    import matplotlib.pyplot as plt

    from viz.make_figures import BLUE, GRAY, INK2, MUTED, _save

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.4), constrained_layout=True)
    labels = [
        f"{r['label']}" + (f"  ({r['run']})" if r["group"] == "unseen run" else "")
        for r in rows
    ]
    y = np.arange(len(rows))
    vals = [r["reduction_pct"] for r in rows]
    cols = [GRAY if r["group"] == "control" else BLUE for r in rows]
    ax.hlines(y, 0, vals, colors=cols, linewidth=2)
    ax.scatter(vals, y, s=60, c=cols, zorder=3)
    for yi, v in zip(y, vals):
        txt = f"−{v:.0f} % error" if v >= 0 else f"+{-v:.0f} % (worse)"
        ax.annotate(
            txt,
            (v, yi),
            xytext=(7 if v >= 0 else -7, 0),
            textcoords="offset points",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=8.5,
            color=INK2,
        )
    ora = rows[0]["oracle_reduction_pct"]
    ax.axvline(ora, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(
        ora,
        1.01,
        f"perfect wind knowledge (−{ora:.0f} %)",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    prev = None
    for yi, r in zip(y, rows):  # light separators between groups
        if prev is not None and r["group"] != prev:
            ax.axhline(yi - 0.5, color="#e1e0d9", linewidth=0.8)
        prev = r["group"]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(min(0, min(vals)) - (35 if min(vals) < 0 else 5), 105)
    ax.set_xlabel("RMS position error reduction vs. PID only [%]")
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "Robustness of the FNO feedforward: one perturbation at a time", pad=16
    )
    _save(
        fig,
        f"robustness{suffix}",
        "Hover 1.2 m behind the cylinder, held-out run(s). Gray: control without FNO (constant mean wind). Simulator always applies true wind and drag.",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/fno.pt"))
    ap.add_argument("--x", type=float, default=1.2)
    ap.add_argument("--duration", type=float, default=7.4)
    ap.add_argument("--settle", type=float, default=0.5)
    args = ap.parse_args()

    main_run = default_run()
    rows = evaluate(main_run, args.ckpt, args.x, args.duration, args.settle, CONDITIONS)
    other = Path("data/lbm/run_019.npz")
    if other.exists() and other != main_run:
        extra = evaluate(
            other,
            args.ckpt,
            args.x,
            args.duration,
            args.settle,
            [("unseen run", "nominal", {}, 1.0)],
        )
        rows += extra
    suffix = (
        ""
        if args.ckpt.parent.name in ("checkpoints", "")
        else f"_{args.ckpt.parent.name}"
    )
    Path("results").mkdir(exist_ok=True)
    Path(f"results/robustness{suffix}.json").write_text(json.dumps(rows, indent=2))
    print(f"saved results/robustness{suffix}.json")
    plot(rows, suffix)


if __name__ == "__main__":
    main()
