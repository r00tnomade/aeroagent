# Copyright 2026 Sami Adam Moughli
# SPDX-License-Identifier: Apache-2.0
"""Render the AeroAgent demo video (1920x1080 MP4) from a real closed-loop run.

  python -m eval.closed_loop --policies pid nemotron --period 0.5     # records results/traj_*.npz
  python -m viz.make_video                                            # -> docs/media/aeroagent_demo.mp4
  python -m viz.make_video --speed 0.25 --bg truth                    # slower, ground-truth background

Left: the flow field AeroAgent reconstructs from 32 pressure sensors (FNO), the drone in it, and the
tracking error over time. Right: a 24 cm close-up of the hover point with two drones drawn to scale,
PID only (orange) vs. AeroAgent (blue), plus the live Nemotron decisions from the same run.
Everything shown is logged data; nothing is re-simulated or smoothed for the video.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

from viz.make_figures import (
    BLUE,
    DIVERGING,
    GRAY,
    INK,
    INK2,
    MUTED,
    ORANGE,
    SURFACE,  # noqa: E402
    _field_ax,
)

ARM = 0.034  # m, half span of the side-view silhouette (Crazyflie 2.x, drawn to scale)
ROTOR_W, ROTOR_H = 0.024, 0.004


class DroneSprite:
    def __init__(self, ax, color, z=5):
        (self.arm,) = ax.plot(
            [], [], color=color, linewidth=4.5, solid_capstyle="round", zorder=z
        )
        self.body = Rectangle((0, 0), 0.018, 0.008, color=color, zorder=z + 1)
        self.rotors = [
            Ellipse((0, 0), ROTOR_W, ROTOR_H, color=color, alpha=0.55, zorder=z + 1)
            for _ in range(2)
        ]
        ax.add_patch(self.body)
        for r in self.rotors:
            ax.add_patch(r)

    def update(self, x, z, pitch):
        c, s = np.cos(-pitch), np.sin(-pitch)
        ends = [(x + sgn * ARM * c, z + sgn * ARM * s) for sgn in (-1, 1)]
        self.arm.set_data([ends[0][0], ends[1][0]], [ends[0][1], ends[1][1]])
        self.body.set_xy((x - 0.009, z - 0.004))
        self.body.angle = np.degrees(-pitch)
        for r, (ex, ez) in zip(self.rotors, ends):
            r.set_center((ex - 0.006 * s, ez + 0.006 * c))
            r.angle = np.degrees(-pitch)


def vorticity(u, w, dx):
    return np.gradient(w, dx, axis=1) - np.gradient(u, dx, axis=0)


def load_traj(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "decisions"}
    out["decisions"] = json.loads(str(d["decisions"])) if "decisions" in d.files else []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run_020")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/fno.pt"))
    ap.add_argument(
        "--agent",
        default="nemotron",
        help="trajectory for the blue drone (nemotron | none | rules)",
    )
    ap.add_argument(
        "--speed", type=float, default=0.5, help="playback speed (0.5 = half speed)"
    )
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bg", choices=["fno", "truth"], default="fno")
    ap.add_argument("--out", type=Path, default=Path("docs/media/aeroagent_demo.mp4"))
    args = ap.parse_args()

    run_path = Path("data/lbm") / f"{args.run}.npz"
    agent_p = Path("results") / f"traj_{args.agent}_{args.run}.npz"
    pid_p = Path("results") / f"traj_pid_{args.run}.npz"
    for p in (run_path, agent_p, pid_p):
        if not p.exists():
            raise SystemExit(
                f"missing {p}. Run: python -m eval.closed_loop --policies pid {args.agent} --period 0.5"
            )
    A, B = load_traj(agent_p), load_traj(pid_p)
    raw = np.load(run_path)
    hover = A["hover"]
    duration = float(min(A["t"][-1], B["t"][-1]))

    # --- flow-field frames (what AeroAgent sees, or the ground truth)
    if args.bg == "fno":
        from control.reflex import FNOWindEstimator

        est = FNOWindEstimator(args.ckpt, run_path)
        xs, zs = est.xs, est.zs
        n = int(duration / est.run.dt) + 2
        fields = [est.field(k)[0].copy() for k in range(min(n, est.n_frames))]
        dt_frame = est.run.dt
    else:
        xs, zs = raw["x"].astype(float), raw["z"].astype(float)
        dt_frame = float(raw["dt"])
        n = int(duration / dt_frame) + 2
        fields = [
            np.stack([raw["u"][k].astype(float), raw["v"][k].astype(float)])
            for k in range(min(n, len(raw["u"])))
        ]
    dx = xs[1] - xs[0]
    vort = [vorticity(f[0], f[1], dx) for f in fields]
    lim = float(np.percentile(np.abs(vort[len(vort) // 2]), 99))
    ext = [xs[0] - dx / 2, xs[-1] + dx / 2, zs[0] - dx / 2, zs[-1] + dx / 2]
    gust = raw["gust"]

    # --- figure layout
    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.65, 1],
        height_ratios=[0.16, 1.0, 0.55],
        left=0.035,
        right=0.985,
        top=0.97,
        bottom=0.07,
        wspace=0.08,
        hspace=0.32,
    )
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_main = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[2, 0])
    ax_zoom = fig.add_subplot(gs[1, 1])
    ax_hud = fig.add_subplot(gs[2, 1])
    ax_hud.axis("off")

    ax_title.text(
        0, 0.75, "AeroAgent", fontsize=24, fontweight="bold", color=INK, va="center"
    )
    ax_title.text(
        0.135,
        0.75,
        "sees the air from 32 pressure sensors, flies the gust",
        fontsize=15,
        color=INK2,
        va="center",
    )
    ax_title.text(
        0,
        0.0,
        "Fourier Neural Operator · 50 Hz wind feedforward · NVIDIA Nemotron supervisor on Nebius Token Factory"
        "  ·  simulation (gym-pybullet-drones + Lattice-Boltzmann), held-out run",
        fontsize=10,
        color=MUTED,
        va="center",
    )

    bg_label = (
        "flow field reconstructed by the FNO from 32 sensors"
        if args.bg == "fno"
        else "ground-truth flow (LBM)"
    )
    im = ax_main.imshow(
        vort[0],
        origin="lower",
        extent=ext,
        cmap=DIVERGING,
        vmin=-lim,
        vmax=lim,
        interpolation="bilinear",
    )
    ax_main.contour(
        raw["x"],
        raw["z"],
        raw["mask"].astype(float),
        levels=[0.5],
        colors=INK,
        linewidths=1.2,
    )
    si, sj = raw["sensor_ij"][:, 0], raw["sensor_ij"][:, 1]
    ax_main.scatter(raw["x"][sj], raw["z"][si], s=7, c=INK, zorder=3, linewidths=0)
    _field_ax(ax_main, f"Vorticity: {bg_label}")
    half = 0.08
    ax_main.add_patch(
        Rectangle(
            (hover[0] - half, hover[2] - half),
            2 * half,
            2 * half,
            fill=False,
            edgecolor=INK,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            zorder=4,
        )
    )
    (dot_a,) = ax_main.plot(
        [], [], "o", color=BLUE, markersize=6, markeredgecolor=SURFACE, zorder=6
    )
    gust_bar = Rectangle(
        (0.32, 0.928), 0.0, 0.03, transform=ax_main.transAxes, color=ORANGE, zorder=7
    )
    ax_main.add_patch(
        Rectangle(
            (0.32, 0.928),
            0.18,
            0.03,
            transform=ax_main.transAxes,
            fill=False,
            edgecolor=INK2,
            linewidth=0.8,
            zorder=7,
        )
    )
    ax_main.add_patch(gust_bar)
    gust_txt = ax_main.text(
        0.01,
        0.97,
        "",
        transform=ax_main.transAxes,
        va="top",
        fontsize=11,
        color=INK,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor="none", alpha=0.85
        ),
    )

    # zoom panel
    zim = ax_zoom.imshow(
        vort[0],
        origin="lower",
        extent=ext,
        cmap=DIVERGING,
        vmin=-lim,
        vmax=lim,
        alpha=0.55,
    )
    ax_zoom.set_xlim(hover[0] - half, hover[0] + half)
    ax_zoom.set_ylim(hover[2] - half, hover[2] + half)
    _field_ax(ax_zoom, "Close-up of the hover point (16 cm window, drones to scale)")
    ax_zoom.axhline(hover[2], color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax_zoom.axvline(hover[0], color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    (trail_b,) = ax_zoom.plot([], [], color=ORANGE, linewidth=1.4, alpha=0.8)
    (trail_a,) = ax_zoom.plot([], [], color=BLUE, linewidth=1.8, alpha=0.9)
    spr_b, spr_a = DroneSprite(ax_zoom, ORANGE, z=5), DroneSprite(ax_zoom, BLUE, z=7)
    arr_true = ax_zoom.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2),
    )
    arr_hat = ax_zoom.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2),
    )
    box = dict(
        boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor="none", alpha=0.85
    )
    ax_zoom.text(
        0.03,
        0.97,
        "PID only",
        transform=ax_zoom.transAxes,
        fontsize=11,
        color=ORANGE,
        fontweight="bold",
        va="top",
        bbox=box,
    )
    ax_zoom.text(
        0.03,
        0.89,
        "AeroAgent",
        transform=ax_zoom.transAxes,
        fontsize=11,
        color=BLUE,
        fontweight="bold",
        va="top",
        bbox=box,
    )
    ax_zoom.text(
        0.03,
        0.03,
        "arrows: true wind (gray) vs. FNO estimate (blue), 1 m/s = 3 cm",
        transform=ax_zoom.transAxes,
        fontsize=8.5,
        color=INK2,
        bbox=box,
    )

    # error panel
    ax_err.set_xlim(0, duration)
    ax_err.set_ylim(0, max(B["err_hover"].max(), A["err_hover"].max()) * 100 * 1.15)
    ax_err.set_ylabel("distance to hover point [cm]")
    ax_err.set_xlabel("time [s]")
    ax_err.fill_between(
        np.arange(len(gust)) * float(raw["dt"]),
        0,
        gust * ax_err.get_ylim()[1],
        color=GRAY,
        alpha=0.35,
        linewidth=0,
        label="gust",
    )
    (line_b,) = ax_err.plot([], [], color=ORANGE, linewidth=2, label="PID only")
    (line_a,) = ax_err.plot([], [], color=BLUE, linewidth=2.4, label="AeroAgent")
    ax_err.legend(loc="upper right", ncol=3, fontsize=9)
    cursor = ax_err.axvline(0, color=INK2, linewidth=1)

    hud = ax_hud.text(
        0.0,
        1.0,
        "",
        va="top",
        ha="left",
        fontsize=11.5,
        color=INK,
        family="monospace",
        transform=ax_hud.transAxes,
        linespacing=1.45,
    )
    dec_txt = ax_hud.text(
        0.0,
        0.42,
        "",
        va="top",
        ha="left",
        fontsize=11,
        color=BLUE,
        transform=ax_hud.transAxes,
        linespacing=1.35,
    )

    decisions = sorted(A["decisions"], key=lambda d: d["sim_t"] + d["latency_s"])
    n_frames = int(duration / args.speed * args.fps)
    tick_ms = 6.0
    lat_json = Path("docs/data/fno_latency_mps.json")
    if lat_json.exists():
        tick_ms = json.loads(lat_json.read_text()).get("p50_ms", tick_ms)

    def idx(tr, t):
        return int(np.clip(np.searchsorted(tr["t"], t), 0, len(tr["t"]) - 1))

    def update(i):
        t = i * args.speed / args.fps
        k = min(int(t / dt_frame), len(vort) - 1)
        im.set_data(vort[k])
        zim.set_data(vort[k])
        ia, ib = idx(A, t), idx(B, t)
        pa, pb = A["pos"][ia], B["pos"][ib]
        dot_a.set_data([pa[0]], [pa[2]])
        spr_a.update(pa[0], pa[2], A["rpy"][ia][1])
        spr_b.update(pb[0], pb[2], B["rpy"][ib][1])
        w0 = max(0, ia - int(1.5 / (A["t"][1] - A["t"][0])))
        trail_a.set_data(A["pos"][w0 : ia + 1, 0], A["pos"][w0 : ia + 1, 2])
        w0b = max(0, ib - int(1.5 / (B["t"][1] - B["t"][0])))
        trail_b.set_data(B["pos"][w0b : ib + 1, 0], B["pos"][w0b : ib + 1, 2])
        scale = 0.03  # 1 m/s -> 3 cm
        wt, wh = A["wind"][ia], A["wind_hat"][ia]
        base = (pa[0], pa[2] + 0.045)
        arr_true.set_position(base)
        arr_true.xy = (base[0] + wt[0] * scale, base[1] + wt[2] * scale)
        arr_hat.set_position(base)
        arr_hat.xy = (base[0] + wh[0] * scale, base[1] + wh[2] * scale)
        line_a.set_data(A["t"][: ia + 1], A["err_hover"][: ia + 1] * 100)
        line_b.set_data(B["t"][: ib + 1], B["err_hover"][: ib + 1] * 100)
        cursor.set_xdata([t, t])
        g = float(gust[min(int(t / float(raw["dt"])), len(gust) - 1)])
        gust_txt.set_text(f"t = {t:4.2f} s    inlet gust {g:4.2f}")
        gust_bar.set_width(0.18 * g)
        hud.set_text(
            f"error now   PID only   {B['err_hover'][ib] * 100:5.2f} cm\n"
            f"            AeroAgent  {A['err_hover'][ia] * 100:5.2f} cm\n"
            f"FNO field   {tick_ms:4.1f} ms per update (M1 Max)\n"
            f"wind at drone  true {np.hypot(wt[0], wt[2]):4.2f} m/s · FNO {np.hypot(wh[0], wh[2]):4.2f} m/s"
        )
        live = [
            d
            for d in decisions
            if d["sim_t"] + d["latency_s"] <= t and d.get("accepted")
        ]
        if live:
            d = live[-1]
            why = textwrap.fill(f"“{d['intent']['rationale']}”", 46)
            dec_txt.set_text(
                f"Nemotron  {d['intent']['mode']}  ·  gain {d['intent']['pid_gain_scale']:.2f}  ·  "
                f"{d['latency_s']:.2f} s\n{why}"
            )
        else:
            dec_txt.set_text("Nemotron  waiting for first decision…")
        return []

    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = FFMpegWriter(
            fps=args.fps,
            bitrate=8000,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        )
        anim.save(args.out, writer=writer, dpi=120)
        print(
            f"wrote {args.out}  ({n_frames} frames, {n_frames / args.fps:.1f} s at {args.speed}x speed)"
        )
    except (FileNotFoundError, RuntimeError) as e:
        gif = args.out.with_suffix(".gif")
        print(
            f"ffmpeg not available ({e}); writing {gif} instead (install: conda install -c conda-forge ffmpeg)"
        )
        anim.save(gif, writer=PillowWriter(fps=min(args.fps, 15)), dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    main()
