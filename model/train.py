# Copyright 2026 Sami Adam Moughli
# SPDX-License-Identifier: Apache-2.0
"""Train the sensor -> flow-field FNO on the LBM runs.

  python -m model.train --epochs 2 --limit-runs 4          # smoke test (a few minutes)
  python -m model.train --epochs 30                        # full run on data/lbm/*.npz

Writes checkpoints/fno.pt (weights + config + normalization), checkpoints/metrics.json and
checkpoints/preview.png (ground truth vs. FNO vs. error on a held-out run).
Go criterion (AGENTS/plan): relative L2 of (u, v) on held-out runs < ~0.2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model.data import Normalizer, SensorFieldDataset, load_run, split_runs
from model.fno import build_model


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def rel_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 over all channels, averaged over the batch."""
    num = ((pred - true) ** 2).flatten(1).sum(1)
    den = (true**2).flatten(1).sum(1) + 1e-8
    return (num / den).sqrt().mean()


def divergence_penalty(
    pred: torch.Tensor, std_u: float, std_v: float, fluid: torch.Tensor
) -> torch.Tensor:
    """Mean squared du/dx + dv/dz (central differences, grid units) in fluid cells, scale-free."""
    u, v = pred[:, 0] * std_u, pred[:, 1] * std_v
    dudx = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / 2
    dvdz = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / 2
    div = (dudx + dvdz) * fluid[:, 1:-1, 1:-1]
    return (div**2).mean() / (std_u**2 + std_v**2)


@torch.no_grad()
def evaluate(
    model, ds: SensorFieldDataset, norm: Normalizer, dev, batch: int = 32
) -> dict:
    model.eval()
    std = torch.tensor(norm.y_std, device=dev)[None, :, None, None]
    mean = torch.tensor(norm.y_mean, device=dev)[None, :, None, None]
    num = torch.zeros(3, device=dev)
    den = torch.zeros(3, device=dev)
    num_uv = den_uv = 0.0
    for xb, yb in ds.batches(batch, shuffle=False):
        x, y = torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev)
        p = model(x) * std + mean
        t = y * std + mean
        num += ((p - t) ** 2).sum((0, 2, 3))
        den += (t**2).sum((0, 2, 3))
        num_uv += float(((p[:, :2] - t[:, :2]) ** 2).sum())
        den_uv += float((t[:, :2] ** 2).sum())
    r = (num / den).sqrt().cpu().numpy()
    return {
        "rel_l2_u": float(r[0]),
        "rel_l2_v": float(r[1]),
        "rel_l2_p": float(r[2]),
        "rel_l2_uv": float(np.sqrt(num_uv / den_uv)),
    }


@torch.no_grad()
def benchmark(model, in_ch: int, hw: tuple[int, int], dev, n: int = 200) -> dict:
    model.eval()
    x = torch.randn(1, in_ch, *hw, device=dev)
    for _ in range(20):
        model(x)
    sync(dev)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        sync(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return {
        "device": str(dev),
        "batch": 1,
        "ms_p50": float(np.median(ts)),
        "ms_p95": float(np.percentile(ts, 95)),
    }


def save_preview(
    model, ds: SensorFieldDataset, norm: Normalizer, dev, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    i = len(ds) // 2
    x, y = ds.get(i)
    with torch.no_grad():
        p = model(torch.from_numpy(x[None]).to(dev))[0].cpu().numpy()
    s, m = norm.y_std[:, None, None], norm.y_mean[:, None, None]
    y, p = y * s + m, p * s + m

    def vort(f):  # vorticity in grid units, for display
        return np.gradient(f[1], axis=1) - np.gradient(f[0], axis=0)

    wt, wp = vort(y), vort(p)
    lim = np.percentile(np.abs(wt), 99)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    for a, img, title, cm, vmin in [
        (ax[0], wt, "ground truth (LBM)", "RdBu_r", -lim),
        (ax[1], wp, "FNO reconstruction from 32 sensors", "RdBu_r", -lim),
        (ax[2], np.abs(wp - wt), "|error|", "inferno", 0),
    ]:
        a.imshow(img, origin="lower", cmap=cm, vmin=vmin, vmax=lim)
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("vorticity, held-out run")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lbm")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--arch", choices=["fno", "physicsnemo"], default="fno")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--downsample", type=int, default=2, help="2 -> 64x128 grid")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes1", type=int, default=16)
    ap.add_argument("--modes2", type=int, default=24)
    ap.add_argument("--div-weight", type=float, default=0.1)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument(
        "--val-runs",
        nargs="*",
        default=["run_019", "run_020", "run_039", "run_040"],
        help="run names held out for validation (kept fixed so eval/ stays on unseen data)",
    )
    ap.add_argument(
        "--limit-runs",
        type=int,
        default=0,
        help="use only the first N runs (smoke test)",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = pick_device(args.device)
    paths = sorted(Path(args.data).glob("run_*.npz"))
    if args.limit_runs:
        paths = paths[: args.limit_runs]
    val_names = set(args.val_runs or [])
    if val_names and any(p.stem in val_names for p in paths):
        val_p = [p for p in paths if p.stem in val_names]
        train_p = [p for p in paths if p.stem not in val_names]
    else:
        train_p, val_p = split_runs(paths, args.n_val)
    print(f"device={dev}  train runs={len(train_p)}  val runs={len(val_p)}")

    t0 = time.perf_counter()
    train_runs = [load_run(p, args.downsample) for p in train_p]
    val_runs = [load_run(p, args.downsample) for p in val_p]
    norm = Normalizer.fit(train_runs)
    train_ds = SensorFieldDataset(
        train_runs, norm, args.history, train=True, seed=args.seed
    )
    val_ds = SensorFieldDataset(val_runs, norm, args.history, train=False)
    hw = train_runs[0].y.shape[-2:]
    print(
        f"loaded in {time.perf_counter() - t0:.0f} s  samples train={len(train_ds)} val={len(val_ds)}  grid={hw}"
    )

    cfg = dict(width=args.width, modes1=args.modes1, modes2=args.modes2)
    model = build_model(args.arch, train_ds.in_channels, 3, **cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {args.arch}: {n_params / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(train_ds) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.1
    )
    su, sv = float(norm.y_std[0]), float(norm.y_std[1])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for ep in range(args.epochs):
        model.train()
        t_ep, losses = time.perf_counter(), []
        for xb, yb in train_ds.batches(args.batch):
            x, y = torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev)
            fluid = 1.0 - x[:, args.history]  # mask channel
            pred = model(x)
            loss = rel_l2(pred, y) + args.div_weight * divergence_penalty(
                pred, su, sv, fluid
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.detach().item())
        metrics = evaluate(model, val_ds, norm, dev)
        history.append(
            {"epoch": ep + 1, "train_loss": float(np.mean(losses)), **metrics}
        )
        print(
            f"epoch {ep + 1:>3}/{args.epochs}  loss {np.mean(losses):.4f}  "
            f"val relL2 uv {metrics['rel_l2_uv']:.3f}  p {metrics['rel_l2_p']:.3f}  "
            f"({time.perf_counter() - t_ep:.0f} s)"
        )
        if metrics["rel_l2_uv"] < best:
            best = metrics["rel_l2_uv"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": args.arch,
                    "cfg": cfg,
                    "in_channels": train_ds.in_channels,
                    "history": args.history,
                    "downsample": args.downsample,
                    "norm": norm.to_dict(),
                    "grid": list(hw),
                    "train_runs": [p.stem for p in train_p],
                    "val_runs": [p.stem for p in val_p],
                },
                out / "fno.pt",
            )

    ckpt = torch.load(out / "fno.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    final = evaluate(model, val_ds, norm, dev)
    bench = benchmark(model, train_ds.in_channels, hw, dev)
    save_preview(model, val_ds, norm, dev, out / "preview.png")
    verdict = (
        "GO"
        if final["rel_l2_uv"] < 0.2
        else "NO-GO (more sensors / more data / PySHRED)"
    )
    summary = {
        "best_val": final,
        "latency": bench,
        "params_M": n_params / 1e6,
        "verdict": verdict,
        "args": vars(args),
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nheld-out relL2  u {final['rel_l2_u']:.3f}  v {final['rel_l2_v']:.3f}  "
        f"p {final['rel_l2_p']:.3f}  uv {final['rel_l2_uv']:.3f}"
    )
    print(
        f"inference ({bench['device']}, batch 1)  p50 {bench['ms_p50']:.2f} ms  p95 {bench['ms_p95']:.2f} ms"
    )
    print(f"verdict: {verdict}\nsaved {out / 'fno.pt'}, metrics.json, preview.png")


if __name__ == "__main__":
    main()
