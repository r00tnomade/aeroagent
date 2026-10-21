# Copyright 2026 Sami Adam Moughli
# SPDX-License-Identifier: Apache-2.0
"""Render every documentation figure into docs/figures/.

  python -m viz.make_figures                 # everything that can be rendered right now
  python -m viz.make_figures --only lbm_gallery sensor_encoding
  python -m viz.make_figures --list

Figures that need a missing piece (trained checkpoint, torch, pybullet) are skipped with a note,
so the script is safe to run at any stage of the project. All numbers shown come from files on
disk (LBM runs, checkpoints/metrics.json, docs/data/*.json) or are measured while rendering.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = Path("docs/figures")
DATA = Path("data/lbm")
CKPT = Path("checkpoints")

# ----------------------------------------------------------------------------- style
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
BLUE, ORANGE, AQUA = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
)  # categorical slots 1-3 (validated all-pairs)
GRAY = "#c3c2b7"  # de-emphasised marks
DIVERGING = LinearSegmentedColormap.from_list(  # blue <-> red, neutral gray midpoint
    "aero_div",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b4b3", "#e34948", "#8f2525"],
)
SEQUENTIAL = LinearSegmentedColormap.from_list(  # one hue, light -> dark
    "aero_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
        "image.interpolation": "bilinear",
    }
)


def _field_ax(ax, title: str | None = None):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)


def _save(fig, name: str, note: str = "") -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    if note:
        eng = fig.get_layout_engine()
        if eng is not None and hasattr(eng, "set"):
            eng.set(rect=(0, 0.06, 1, 0.94))  # keep the footnote clear of axis labels
        fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _runs() -> list[Path]:
    return sorted(DATA.glob("run_*.npz"))


def _vort_limit(w: np.ndarray) -> float:
    return float(np.percentile(np.abs(w), 99))


def _extent(d) -> list[float]:
    return [float(d["x"][0]), float(d["x"][-1]), float(d["z"][0]), float(d["z"][-1])]


def _cylinder(ax, d, color=INK, lw=1.0):
    m = d["mask"].astype(float)
    ax.contour(d["x"], d["z"], m, levels=[0.5], colors=color, linewidths=lw)


# ----------------------------------------------------------------------------- 1. dataset gallery
def fig_lbm_gallery():
    """Small multiples: vorticity at the gust peak for up to 6 runs (shared colour scale)."""
    runs = _runs()[:6]
    if not runs:
        return "no data/lbm/*.npz"
    ds = [np.load(p) for p in runs]
    frames = [int(np.argmax(d["gust"])) for d in ds]
    lim = np.median(
        [_vort_limit(d["omega"][k].astype(np.float32)) for d, k in zip(ds, frames)]
    )
    cols = 3 if len(ds) > 2 else len(ds)
    rows = int(np.ceil(len(ds) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(max(9.0, 4.2 * cols), 2.3 * rows + 0.9),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, d, k, p in zip(axes.flat, ds, frames, runs):
        ax.set_visible(True)
        im = ax.imshow(
            d["omega"][k].astype(np.float32),
            origin="lower",
            cmap=DIVERGING,
            vmin=-lim,
            vmax=lim,
            extent=_extent(d),
        )
        _cylinder(ax, d)
        _field_ax(ax, f"{p.stem}  ·  Re {float(d['re']):.0f}")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("vorticity ω [1/s]", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(
        "Synthetic training data: gusty flow past a cylinder (D2Q9 Lattice-Boltzmann)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
        color=INK,
    )
    return _save(
        fig,
        "lbm_gallery",
        "Each panel shows the frame at the gust peak. Domain 4 m × 2 m, 256 × 128 cells.",
    )


# ----------------------------------------------------------------------------- 2. sensor encoding
def fig_sensor_encoding():
    """What the FNO sees: 32 surface sensors -> Voronoi-tessellated pressure field."""
    runs = _runs()
    if not runs:
        return "no data/lbm/*.npz"
    from model.data import build_static

    d = np.load(runs[0])
    k = int(np.argmax(d["gust"]))
    vor, _ = build_static(d["mask"], d["sensor_ij"], ds=1)
    p_s = d["p_sensors"][k]
    field = p_s[vor]
    si, sj = d["sensor_ij"][:, 0], d["sensor_ij"][:, 1]
    xs, zs = d["x"][sj], d["z"][si]
    w = d["omega"][k].astype(np.float32)
    lim_w, lim_p = _vort_limit(w), float(np.abs(p_s).max())
    ext = _extent(d)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.3), constrained_layout=True)
    ax[0].imshow(w, origin="lower", cmap=DIVERGING, vmin=-lim_w, vmax=lim_w, extent=ext)
    _cylinder(ax[0], d)
    ax[0].scatter(xs, zs, s=10, c=INK, zorder=3, linewidths=0)
    _field_ax(ax[0], "Ground truth: vorticity + 32 pressure taps")
    field = np.ma.masked_where(d["mask"] > 0, field)
    im = ax[1].imshow(
        field, origin="lower", cmap=DIVERGING, vmin=-lim_p, vmax=lim_p, extent=ext
    )
    _cylinder(ax[1], d)
    ax[1].scatter(xs, zs, s=10, c=INK, zorder=3, linewidths=0)
    _field_ax(ax[1], "FNO input: nearest-sensor (Voronoi) pressure map")
    cb = fig.colorbar(im, ax=ax[1], shrink=0.85, pad=0.01)
    cb.set_label("sensor pressure [Pa]", color=INK2)
    cb.outline.set_visible(False)
    return _save(
        fig,
        "sensor_encoding",
        "Input encoding after Fukami et al. (2021). The model also gets the last 8 frames, mask, distance field and coordinates.",
    )


# ----------------------------------------------------------------------------- 3. sensor time series
def fig_sensor_timeseries():
    runs = _runs()
    if not runs:
        return "no data/lbm/*.npz"
    d = np.load(runs[0])
    t = np.arange(len(d["gust"])) * float(d["dt"])
    p = d["p_sensors"]
    k = p.shape[1]
    picks = {
        "front (stagnation)": k // 2,
        "top shoulder": k // 4,
        "rear (wake side)": 0,
    }
    fig, (a0, a1) = plt.subplots(
        2,
        1,
        figsize=(9, 4.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    a0.fill_between(t, 0, d["gust"], color=GRAY, linewidth=0)
    a0.set_ylabel("gust\nfactor")
    a0.set_ylim(0, 1.05)
    a0.set_yticks([0, 1])
    a0.set_title("Inlet gust (1 − cos pulse)")
    for (label, idx), c in zip(picks.items(), (BLUE, ORANGE, AQUA)):
        a1.plot(t, p[:, idx], color=c, label=label)
        a1.annotate(
            label,
            (t[-1], p[-1, idx]),
            xytext=(4, 0),
            textcoords="offset points",
            color=INK2,
            fontsize=8.5,
            va="center",
        )
    a1.set_xlabel("time [s]")
    a1.set_ylabel("pressure [Pa]")
    a1.set_title(
        "Three of the 32 surface sensors: shedding oscillation + gust response"
    )
    a1.legend(loc="upper left", ncol=3, fontsize=8.5)
    a1.set_xlim(t[0], t[-1] * 1.12)
    return _save(
        fig,
        "sensor_timeseries",
        f"{runs[0].stem}, Re {float(d['re']):.0f}. Sensor angle 0° = downstream side.",
    )


# ----------------------------------------------------------------------------- 4. training curves
def fig_training_curves():
    mpath = CKPT / "metrics.json"
    if not mpath.exists():
        return "no checkpoints/metrics.json (train first)"
    m = json.loads(mpath.read_text())
    h = m["history"]
    ep = [r["epoch"] for r in h]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    a0.plot(ep, [r["train_loss"] for r in h], color=BLUE)
    a0.set_title("Training loss (rel. L2 + divergence penalty)")
    a0.set_xlabel("epoch")
    a0.set_ylabel("loss")
    for key, label, c in (
        ("rel_l2_u", "u", BLUE),
        ("rel_l2_v", "v", ORANGE),
        ("rel_l2_p", "p", AQUA),
    ):
        y = [r[key] for r in h]
        a1.plot(ep, y, color=c, label=label)
        a1.annotate(
            label,
            (ep[-1], y[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            color=INK2,
            va="center",
        )
    a1.axhline(0.2, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    a1.annotate(
        "GO threshold (u,v) 0.2",
        (ep[len(ep) // 2], 0.2),
        xytext=(0, -11),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
        ha="center",
    )
    a1.set_title("Validation error on held-out runs")
    a1.set_xlabel("epoch")
    a1.set_ylabel("relative L2 error")
    a1.set_ylim(bottom=0)
    a1.legend(loc="upper right")
    b = m["best_val"]
    return _save(
        fig,
        "training_curves",
        f"Best: u {b['rel_l2_u']:.3f} · v {b['rel_l2_v']:.3f} · p {b['rel_l2_p']:.3f} · uv {b['rel_l2_uv']:.3f}  "
        f"({m['params_M']:.1f} M parameters)",
    )


# ----------------------------------------------------------------------------- model helpers
def _load_model():
    import torch

    from model.data import Normalizer, load_run, split_runs
    from model.fno import build_model
    from model.train import pick_device

    ck = CKPT / "fno.pt"
    if not ck.exists():
        raise FileNotFoundError("no checkpoints/fno.pt")
    dev = pick_device("auto")
    c = torch.load(ck, map_location=dev, weights_only=False)
    model = build_model(c["arch"], c["in_channels"], 3, **c["cfg"]).to(dev).eval()
    model.load_state_dict(c["state_dict"])
    norm = Normalizer.from_dict(c["norm"])
    args = (
        json.loads((CKPT / "metrics.json").read_text())["args"]
        if (CKPT / "metrics.json").exists()
        else {}
    )
    if c.get("val_runs"):
        val = [p for p in _runs() if p.stem in set(c["val_runs"])]
    else:
        _, val = split_runs(_runs(), args.get("n_val", 2))
    run = load_run(val[0], c["downsample"])
    return model, norm, run, val[0], c, dev


def _predict_run(model, norm, run, history, dev, batch: int = 32):
    import torch

    from model.data import make_input

    ts = list(range(history - 1, len(run.y)))
    preds = []
    with torch.no_grad():
        for s in range(0, len(ts), batch):
            xb = np.stack(
                [
                    make_input(run.p_sensors[t - history + 1 : t + 1], run, norm)
                    for t in ts[s : s + batch]
                ]
            )
            preds.append(model(torch.from_numpy(xb).to(dev)).cpu().numpy())
    p = (
        np.concatenate(preds) * norm.y_std[None, :, None, None]
        + norm.y_mean[None, :, None, None]
    )
    y = run.y[ts].astype(np.float32)
    return np.array(ts), p, y


def _vorticity(f: np.ndarray) -> np.ndarray:
    return np.gradient(f[1], axis=-1) - np.gradient(f[0], axis=-2)


# ----------------------------------------------------------------------------- 5. reconstruction grid
def fig_reconstruction():
    try:
        model, norm, run, path, c, dev = _load_model()
    except Exception as e:  # noqa: BLE001
        return str(e)
    ts, p, y = _predict_run(model, norm, run, c["history"], dev)
    raw = np.load(path)
    gust = raw["gust"][ts]
    k = int(np.argmax(gust))
    rows = [
        ("u [m/s]", y[k, 0], p[k, 0], "seq"),
        ("v [m/s]", y[k, 1], p[k, 1], "div"),
        ("p [Pa]", y[k, 2], p[k, 2], "div"),
        ("vorticity", _vorticity(y[k]), _vorticity(p[k]), "div"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(12, 7.6), constrained_layout=True)
    for r, (name, t_, p_, kind) in enumerate(rows):
        if kind == "div":
            lim = float(np.percentile(np.abs(t_), 99))
            kw = dict(cmap=DIVERGING, vmin=-lim, vmax=lim)
        else:
            lo, hi = np.percentile(t_, [1, 99])
            kw = dict(cmap=SEQUENTIAL, vmin=lo, vmax=hi)
        axes[r, 0].imshow(t_, origin="lower", **kw)
        im = axes[r, 1].imshow(p_, origin="lower", **kw)
        err = np.abs(p_ - t_)
        ie = axes[r, 2].imshow(
            err,
            origin="lower",
            cmap=SEQUENTIAL,
            vmin=0,
            vmax=float(np.percentile(np.abs(t_ - t_.mean()), 99)),
        )
        for ci, title in enumerate(
            ("Ground truth (LBM)", "FNO from 32 sensors", "|error|")
        ):
            _field_ax(axes[r, ci], title if r == 0 else None)
        axes[r, 0].set_ylabel(name, color=INK2)
        axes[r, 0].yaxis.set_visible(True)
        axes[r, 0].set_yticks([])
        fig.colorbar(im, ax=axes[r, :2], shrink=0.9, pad=0.01).outline.set_visible(
            False
        )
        fig.colorbar(ie, ax=axes[r, 2], shrink=0.9, pad=0.01).outline.set_visible(False)
    rel = np.sqrt(((p[k, :2] - y[k, :2]) ** 2).sum() / (y[k, :2] ** 2).sum())
    fig.suptitle(
        f"Flow-field reconstruction on a held-out run ({path.stem}), gust peak",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
        color=INK,
    )
    return _save(
        fig,
        "reconstruction",
        f"Relative L2 (u,v) in this frame: {rel:.3f}. Error scale = 99th pct. of |truth − mean|.",
    )


# ----------------------------------------------------------------------------- 6. error over time
def fig_error_over_time():
    try:
        model, norm, run, path, c, dev = _load_model()
    except Exception as e:  # noqa: BLE001
        return str(e)
    ts, p, y = _predict_run(model, norm, run, c["history"], dev)
    raw = np.load(path)
    t = ts * run.dt
    rel = np.sqrt(
        ((p[:, :2] - y[:, :2]) ** 2).sum((1, 2, 3)) / (y[:, :2] ** 2).sum((1, 2, 3))
    )
    rel_p = np.sqrt(((p[:, 2] - y[:, 2]) ** 2).sum((1, 2)) / (y[:, 2] ** 2).sum((1, 2)))
    fig, (a0, a1) = plt.subplots(
        2,
        1,
        figsize=(9, 4.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    a0.fill_between(t, 0, raw["gust"][ts], color=GRAY, linewidth=0)
    a0.set_ylim(0, 1.05)
    a0.set_yticks([0, 1])
    a0.set_ylabel("gust\nfactor")
    a0.set_title("Inlet gust")
    a1.plot(t, rel, color=BLUE, label="velocity (u,v)")
    a1.plot(t, rel_p, color=ORANGE, label="pressure p")
    a1.axhline(0.2, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    a1.set_ylim(bottom=0)
    a1.set_xlabel("time [s]")
    a1.set_ylabel("relative L2 error")
    a1.set_title("Reconstruction error frame by frame (held-out run)")
    a1.legend(loc="upper left", ncol=2)
    return _save(fig, "error_over_time", f"{path.stem}. Dashed line: GO threshold 0.2.")


# ----------------------------------------------------------------------------- 7. FNO latency
def fig_fno_latency():
    try:
        import torch

        from model.train import sync

        model, norm, run, path, c, dev = _load_model()
    except Exception as e:  # noqa: BLE001
        return str(e)
    x = torch.randn(1, c["in_channels"], *c["grid"], device=dev)
    with torch.no_grad():
        for _ in range(30):
            model(x)
        sync(dev)
        ms = []
        for _ in range(500):
            t0 = time.perf_counter()
            model(x)
            sync(dev)
            ms.append((time.perf_counter() - t0) * 1e3)
    ms = np.array(ms)
    p50, p95 = np.median(ms), np.percentile(ms, 95)
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    ax.hist(ms, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=1)
    for v, lab in ((p50, "p50"), (p95, "p95")):
        ax.axvline(v, color=INK2, linewidth=1)
        ax.annotate(
            f"{lab} {v:.2f} ms",
            (v, ax.get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            color=INK2,
            fontsize=8.5,
        )
    ax.set_xlabel("forward pass [ms]")
    ax.set_ylabel("count")
    ax.set_title(
        f"FNO inference latency, batch 1, {c['grid'][0]}×{c['grid'][1]} grid ({dev})"
    )
    (OUT.parent / "data").mkdir(parents=True, exist_ok=True)
    (OUT.parent / "data" / f"fno_latency_{dev.type}.json").write_text(
        json.dumps(
            {
                "device": str(dev),
                "p50_ms": float(p50),
                "p95_ms": float(p95),
                "n": len(ms),
            },
            indent=2,
        )
    )
    return _save(
        fig,
        "fno_latency",
        f"500 timed calls after 30 warm-up calls, synchronised. Device: {dev}.",
    )


# ----------------------------------------------------------------------------- 8. Nemotron latency
def fig_nemotron_latency():
    f = Path("docs/data/nemotron_bench_2026-10-20.json")
    if not f.exists():
        return f"no {f}"
    b = json.loads(f.read_text())["models"]
    names = list(b)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    for i, n in enumerate(names):
        s = np.array(b[n]["ms"]) / 1000
        chosen = n == "Nemotron 3 Super"
        col = BLUE if chosen else GRAY
        ax.scatter(
            s,
            i + rng.uniform(-0.15, 0.15, len(s)),
            s=28,
            color=col,
            edgecolor=SURFACE,
            linewidth=1,
            zorder=3,
        )
        med, p95 = np.median(s), np.percentile(s, 95)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, linewidth=2, zorder=4)
        ax.annotate(
            f"p50 {med:.2f} s · p95 {p95:.2f} s · schema {b[n]['schema_valid']}",
            (s.max(), i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK if chosen else INK2,
        )
    ax.set_yticks(
        range(len(names)),
        [n + ("  (supervisor)" if n == "Nemotron 3 Super" else "") for n in names],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(v["ms"]) for v in b.values()) / 1000 * 1.75)
    ax.set_xlabel("wall-clock latency per call [s]")
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "Nemotron on Nebius Token Factory: one strict JSON-schema intent per call"
    )
    return _save(
        fig,
        "nemotron_latency",
        "Reasoning off except Ultra. 25 calls per model (Ultra: 3), home network, 20 Oct 2026. Black tick = median.",
    )


# ----------------------------------------------------------------------------- 9. latency tiers
def fig_latency_tiers():
    """Why the architecture is hierarchical: measured latencies vs. each loop's period, log scale."""
    nb = json.loads(Path("docs/data/nemotron_bench_2026-10-20.json").read_text())[
        "models"
    ]
    fno_file = Path("docs/data/fno_latency_mps.json")
    fno = json.loads(fno_file.read_text())["p50_ms"] if fno_file.exists() else None
    items = [
        ("Tier 0 attitude loop period (240 Hz)", 1000 / 240, "budget"),
        ("Tier 1 reflex loop period (50 Hz)", 20.0, "budget"),
        ("FNO forward pass (M1 Max, p50)", fno, "measured"),
        (
            "Nemotron 3 Super intent (p50)",
            float(np.median(nb["Nemotron 3 Super"]["ms"])),
            "measured",
        ),
        ("Tier 2 supervisor period (~1 Hz)", 1000.0, "budget"),
        (
            "Nemotron 3 Ultra plan, reasoning (p50)",
            float(np.median(nb["Nemotron 3 Ultra (reasoning)"]["ms"])),
            "measured",
        ),
    ]
    items = [it for it in items if it[1] is not None]
    fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    for i, (lab, v, kind) in enumerate(items):
        meas = kind == "measured"
        ax.scatter(
            [v],
            [i],
            s=70 if meas else 60,
            marker="o" if meas else "|",
            color=BLUE if meas else INK2,
            linewidths=2.5 if not meas else 0,
            zorder=3,
        )
        ax.annotate(
            f"{v:,.1f} ms" if v < 100 else f"{v / 1000:.2f} s",
            (v, i),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK2,
        )
    ax.set_xscale("log")
    ax.set_xlim(1, 30000)
    ax.set_yticks(range(len(items)), [it[0] for it in items])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("milliseconds (log scale)")
    ax.set_title("Each component sits in the loop it is fast enough for")
    note = "Blue dots: measured. Dark ticks: loop periods (budgets)."
    if fno is None:
        note += " FNO value appears after `--only fno_latency` has run."
    return _save(fig, "latency_tiers", note)


# ----------------------------------------------------------------------------- 10. PID baseline
def fig_pid_baseline():
    try:
        from sim.hover_wind_demo import simulate
    except Exception as e:  # noqa: BLE001
        return f"pybullet / gym-pybullet-drones unavailable ({e})"
    res = {k: simulate(k) for k in ("none", "const", "gust")}
    labels = {"none": "no wind", "const": "constant 2 m/s", "gust": "gust (1 − cos)"}
    colors = {"none": GRAY, "const": ORANGE, "gust": BLUE}
    fig, (a0, a1) = plt.subplots(
        2, 1, figsize=(9, 5), sharex=True, constrained_layout=True
    )
    for k, r in res.items():
        rms, mx = np.sqrt(np.mean(r["err"] ** 2)) * 100, r["err"].max() * 100
        a0.plot(
            r["t"],
            np.linalg.norm(r["force"], axis=1) * 1000,
            color=colors[k],
            label=labels[k],
        )
        a1.plot(
            r["t"],
            r["err"] * 100,
            color=colors[k],
            label=f"{labels[k]}: RMS {rms:.1f} cm, max {mx:.1f} cm",
        )
    a0.set_ylabel("wind force [mN]")
    a0.set_title("Aerodynamic drag on the Crazyflie (weight ≈ 265 mN)")
    a0.legend(loc="upper right", ncol=3)
    a1.set_ylabel("position error [cm]")
    a1.set_xlabel("time [s]")
    a1.set_title("Baseline: PID only, hover at 1 m")
    a1.legend(loc="upper right", fontsize=8.5)
    a1.set_xlim(0, res["gust"]["t"][-1])
    a0.set_xlim(0, res["gust"]["t"][-1])
    return _save(
        fig,
        "pid_baseline",
        "gym-pybullet-drones CF2X, DSLPIDControl at 240 Hz, seeded and deterministic.",
    )


# ----------------------------------------------------------------------------- 11. hero GIF
def fig_hero_gif():
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        model, norm, run, path, c, dev = _load_model()
    except Exception as e:  # noqa: BLE001
        return str(e)
    ts, p, y = _predict_run(model, norm, run, c["history"], dev)
    wt = np.stack([_vorticity(f) for f in y])
    wp = np.stack([_vorticity(f) for f in p])
    lim = _vort_limit(wt)
    gust = np.load(path)["gust"][ts]
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.0), constrained_layout=True)
    ims = [
        ax[0].imshow(wt[0], origin="lower", cmap=DIVERGING, vmin=-lim, vmax=lim),
        ax[1].imshow(wp[0], origin="lower", cmap=DIVERGING, vmin=-lim, vmax=lim),
    ]
    _field_ax(ax[0], "Ground truth (LBM)")
    _field_ax(ax[1], "AeroAgent FNO, 32 pressure sensors only")
    txt = fig.text(0.99, 0.01, "", ha="right", va="bottom", fontsize=8, color=INK2)
    step = max(1, len(ts) // 120)

    def update(i):
        k = i * step
        ims[0].set_data(wt[k])
        ims[1].set_data(wp[k])
        txt.set_text(f"t = {ts[k] * run.dt:4.2f} s   gust {gust[k]:.2f}")
        return ims + [txt]

    anim = FuncAnimation(fig, update, frames=len(ts) // step, blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "hero.gif"
    anim.save(out, writer=PillowWriter(fps=15), dpi=90)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


FIGURES = {
    "lbm_gallery": fig_lbm_gallery,
    "sensor_encoding": fig_sensor_encoding,
    "sensor_timeseries": fig_sensor_timeseries,
    "pid_baseline": fig_pid_baseline,
    "training_curves": fig_training_curves,
    "reconstruction": fig_reconstruction,
    "error_over_time": fig_error_over_time,
    "fno_latency": fig_fno_latency,
    "nemotron_latency": fig_nemotron_latency,
    "latency_tiers": fig_latency_tiers,
    "hero_gif": fig_hero_gif,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of figure names")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(FIGURES))
        return
    names = args.only or list(FIGURES)
    done, skipped = [], []
    for n in names:
        print(f"[{n}]")
        r = FIGURES[n]()
        (done if isinstance(r, Path) else skipped).append((n, r))
    print(f"\n{len(done)} figures written to {OUT}/")
    for n, why in skipped:
        print(f"  skipped {n}: {why}")


if __name__ == "__main__":
    main()
