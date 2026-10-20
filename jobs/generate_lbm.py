# Copyright 2026 <Your Name>
# SPDX-License-Identifier: Apache-2.0
"""Generate gusty cylinder-wake runs with the numpy LBM solver.

  python -m jobs.generate_lbm --runs 1 --preview                 # quick check: one run + PNG
  python -m jobs.generate_lbm --runs 20 --out data/lbm            # dataset (random Re, gust, cylinder pos)

Each run is saved as data/lbm/run_XXX.npz with u, v, p, omega (T, H, W) in SI units, mask, x, z, dt,
sensor_ij, p_sensors (T, K) and the gust factor per frame. Later this script runs as a Nebius Serverless Job.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.lbm2d import LBMConfig, run


def random_config(seed: int) -> LBMConfig:
    rng = np.random.default_rng(seed)
    return LBMConfig(
        re=float(rng.uniform(100, 220)),
        cyl_y=float(rng.uniform(52, 76)),
        radius=float(rng.uniform(8, 10)),
        gust_u=float(rng.uniform(0.2, 0.7)),
        gust_v=float(rng.uniform(-0.3, 0.3)),
        gust_start=int(rng.uniform(3000, 7000)),
        gust_len=int(rng.uniform(1500, 4000)),
        seed=seed,
    )


def save_preview(data: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = int(np.argmax(data["gust"])) if data["gust"].max() > 0 else -1
    w = data["omega"][k]
    lim = np.percentile(np.abs(w), 99)
    ext = [data["x"][0], data["x"][-1], data["z"][0], data["z"][-1]]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].imshow(w, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext)
    axes[0].set_title(f"vorticity at gust peak (frame {k}), Re={float(data['re']):.0f}")
    si, sj = data["sensor_ij"][:, 0], data["sensor_ij"][:, 1]
    axes[0].scatter(data["x"][sj], data["z"][si], s=6, c="k")
    t = np.arange(len(data["gust"])) * float(data["dt"])
    axes[1].plot(t, data["p_sensors"][:, ::8])
    axes[1].plot(
        t, data["gust"] * np.abs(data["p_sensors"]).max(), "k--", label="gust (scaled)"
    )
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("sensor pressure [Pa]")
    axes[1].legend()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument(
        "--steps",
        type=int,
        default=12000,
        help="saved lattice steps per run (after warm-up)",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=4000,
        help="steps discarded while the wake develops",
    )
    ap.add_argument("--save-every", type=int, default=40)
    ap.add_argument("--out", default="data/lbm")
    ap.add_argument("--preview", action="store_true", help="also write a PNG per run")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in range(args.runs):
        seed = args.seed0 + r
        cfg = random_config(seed)
        cfg.gust_start += args.warmup  # gust must hit after the wake has developed
        t0 = time.perf_counter()
        print(f"run {r + 1}/{args.runs}  seed={seed}  Re={cfg.re:.0f}")
        data = run(
            cfg, steps=args.steps, save_every=args.save_every, warmup=args.warmup
        )
        path = out / f"run_{seed:03d}.npz"
        np.savez_compressed(path, **data)
        meta = {k: v for k, v in vars(cfg).items() if k != "extra"}
        (out / f"run_{seed:03d}.json").write_text(json.dumps(meta, indent=2))
        if args.preview:
            save_preview(data, out / f"run_{seed:03d}.png")
        print(
            f"  saved {path} ({data['u'].shape[0]} frames, {path.stat().st_size / 1e6:.1f} MB, "
            f"{time.perf_counter() - t0:.0f} s)"
        )


if __name__ == "__main__":
    main()
