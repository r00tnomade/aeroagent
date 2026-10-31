# Copyright 2026 Sami Adam Moughli
# SPDX-License-Identifier: Apache-2.0
"""Look inside the trained FNO.

  python -m viz.fno_inside                     # docs/figures/fno_inside.gif + fno_spectrum.png
  python -m viz.fno_inside --frames 90 --run run_019

fno_inside.gif   The network as a stack of planes in 3D: sensor input -> lifting -> 4 Fourier layers -> output
                 vorticity. Each plane shows the real activation of that layer (the most active hidden channel),
                 recorded with forward hooks while the flow evolves. The camera orbits slowly.
fno_spectrum.png What "Fourier" means here: the spectrum of a hidden layer, and the block of low frequencies
                 the spectral convolution keeps (everything outside is cut off).

Works with the plain-PyTorch FNO (--arch fno, the default checkpoint).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from viz.make_figures import (
    DIVERGING,
    INK,
    INK2,
    MUTED,
    SEQUENTIAL,
    SURFACE,
    _save,
)  # noqa: E402


# ----------------------------------------------------------------------------- activations
def record_activations(ckpt: Path, run_path: Path, frames: list[int]):
    """Returns names, list over frames of list over planes (2D arrays), and the kept Fourier modes."""
    import torch

    from model.data import Normalizer, load_run, make_input
    from model.fno import build_model
    from model.train import pick_device

    dev = pick_device("auto")
    c = torch.load(ckpt, map_location=dev, weights_only=False)
    if c["arch"] != "fno":
        raise SystemExit("fno_inside visualises the plain-PyTorch FNO (arch 'fno').")
    model = build_model(c["arch"], c["in_channels"], 3, **c["cfg"]).to(dev).eval()
    model.load_state_dict(c["state_dict"])
    norm = Normalizer.from_dict(c["norm"])
    run = load_run(run_path, c["downsample"])
    H = int(c["history"])
    h, w = run.y.shape[-2:]
    p = model.padding

    store: dict[str, np.ndarray] = {}

    def keep(name):
        def hook(_m, _inp, out):
            store[name] = out.detach()[0].float().cpu().numpy()

        return hook

    hooks = [model.lift.register_forward_hook(keep("lift"))]
    for i, (sc, pw) in enumerate(zip(model.spectral, model.pointwise)):
        hooks.append(pw.register_forward_hook(keep(f"pw{i}")))
        hooks.append(sc.register_forward_hook(keep(f"sc{i}")))

    names = (
        ["Sensor input", "Lifting (1×1 conv)"]
        + [f"Fourier layer {i + 1}" for i in range(len(model.spectral))]
        + ["Output: vorticity"]
    )
    planes_t, spec_example = [], None
    with torch.no_grad():
        for k in frames:
            k = max(k, H - 1)
            x = make_input(run.p_sensors[k - H + 1 : k + 1], run, norm)
            y = model(torch.from_numpy(x[None]).to(dev))[0].cpu().numpy()
            y = y * norm.y_std[:, None, None] + norm.y_mean[:, None, None]
            planes = [x[H - 1]]  # newest sensor frame (Voronoi map)
            lift = store["lift"]
            planes.append(_pick_channel(lift))
            for i in range(len(model.spectral)):
                a = store[f"sc{i}"] + store[f"pw{i}"]  # layer output before activation
                a = a[:, : a.shape[1] - p, : a.shape[2] - p]  # drop the padding
                planes.append(_pick_channel(a))
                if i == 0 and spec_example is None:
                    spec_example = a
            vort = np.gradient(y[1], axis=1) - np.gradient(y[0], axis=0)
            planes.append(vort)
            planes_t.append(planes)
    for hk in hooks:
        hk.remove()
    modes = (model.spectral[0].modes1, model.spectral[0].modes2)
    return names, planes_t, spec_example, modes, (h, w)


def _pick_channel(a: np.ndarray) -> np.ndarray:
    """The hidden channel with the largest spatial variance: the most 'active' feature map."""
    return a[int(np.argmax(a.reshape(a.shape[0], -1).std(1)))]


# ----------------------------------------------------------------------------- rendering
def render_stack(
    names, planes_t, out: Path, step: int = 2, fps: int = 15, elev: float = 24.0
) -> Path:
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = len(names)
    first = [pl[::step, ::step] for pl in planes_t[0]]
    hh, ww = first[0].shape
    X, Y = np.meshgrid(np.linspace(0, 2.0, ww), np.linspace(0, 1.0, hh))
    gap = 0.55
    # one colour scale per plane, fixed over time (so the motion you see is real change, not rescaling)
    lims = [
        float(np.percentile(np.abs(np.stack([pt[i] for pt in planes_t])), 99)) + 1e-6
        for i in range(n)
    ]
    cmaps = [DIVERGING] * n

    fig = plt.figure(figsize=(7.2, 7.2), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=-0.06, right=1.02, bottom=0.05, top=0.92)
    fig.suptitle(
        "Inside AeroAgent's Fourier Neural Operator",
        color=INK,
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )

    def draw(fi):
        ax.cla()
        ax.set_facecolor(SURFACE)
        ax.set_axis_off()
        planes = [pl[::step, ::step] for pl in planes_t[fi % len(planes_t)]]
        for i, pl in enumerate(planes):
            z = i * gap
            colors = cmaps[i](Normalize(-lims[i], lims[i])(pl))
            ax.plot_surface(
                X,
                Y,
                np.full_like(X, z),
                facecolors=colors,
                rstride=1,
                cstride=1,
                shade=False,
                linewidth=0,
                antialiased=False,
            )
            ax.plot(
                [0, 2, 2, 0, 0], [0, 0, 1, 1, 0], [z] * 5, color=INK2, linewidth=0.6
            )
            ax.text(
                2.12,
                0.5,
                z,
                names[i],
                color=INK if i in (0, n - 1) else INK2,
                fontsize=9,
                fontweight="bold" if i in (0, n - 1) else "normal",
                zdir=None,
            )
        ax.set_xlim(0, 2.6)
        ax.set_ylim(-0.3, 1.3)
        ax.set_zlim(-0.2, (n - 1) * gap + 0.2)
        ax.set_box_aspect((2.6, 1.6, (n - 1) * gap + 0.4), zoom=1.12)
        ax.view_init(
            elev=elev, azim=-60 + 40 * np.sin(2 * np.pi * fi / max(len(planes_t), 1))
        )
        (
            fig.text(
                0.5,
                0.03,
                "each plane: the most active feature map of that layer · bottom: 32 pressure sensors · "
                "top: reconstructed vorticity",
                ha="center",
                color=MUTED,
                fontsize=7.5,
            )
            if fi == 0
            else None
        )
        return []

    anim = FuncAnimation(fig, draw, frames=len(planes_t), blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(
        out,
        writer=PillowWriter(fps=fps),
        dpi=100,
        savefig_kwargs={"facecolor": SURFACE},
    )
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def render_spectrum(
    act: np.ndarray, modes: tuple[int, int], name: str = "fno_spectrum"
):
    """Spectrum of the first hidden layer, with the low-frequency block the spectral convolution keeps."""
    from matplotlib.patches import Rectangle

    spec = np.abs(np.fft.rfft2(act, norm="ortho")).mean(0)  # average over channels
    spec = np.fft.fftshift(spec, axes=0)
    h, wf = spec.shape
    m1, m2 = modes
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    ax[0].imshow(_pick_channel(act), origin="lower", cmap=DIVERGING)
    ax[0].set_title("A hidden feature map (Fourier layer 1)")
    im = ax[1].imshow(
        np.log10(spec + 1e-6),
        origin="lower",
        cmap=SEQUENTIAL,
        aspect="auto",
        extent=[0, wf, -h // 2, h - h // 2],
    )
    ax[1].add_patch(
        Rectangle((0, -m1), m2, 2 * m1, fill=False, edgecolor=INK, linewidth=1.6)
    )
    ax[1].annotate(
        f"kept: {2 * m1} × {m2} lowest frequencies",
        (m2, m1),
        xytext=(8, 6),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    ax[1].set_title("Its Fourier spectrum (log magnitude)")
    ax[1].set_xlabel("frequency along the flow")
    ax[1].set_ylabel("frequency across the flow")
    for a in ax:
        a.grid(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    fig.colorbar(im, ax=ax[1], shrink=0.85).outline.set_visible(False)
    return _save(
        fig,
        name,
        "The spectral convolution multiplies only the boxed modes by learned weights and discards the rest. "
        "That is why one layer can mix information across the whole field at once.",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/fno.pt"))
    ap.add_argument("--run", default="run_020")
    ap.add_argument(
        "--frames",
        type=int,
        default=60,
        help="GIF length in frames (one per LBM frame)",
    )
    ap.add_argument("--start", type=int, default=60, help="first LBM frame")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/fno_inside.gif"))
    args = ap.parse_args()

    frames = list(range(args.start, args.start + args.frames))
    names, planes_t, spec, modes, _ = record_activations(
        args.ckpt, Path("data/lbm") / f"{args.run}.npz", frames
    )
    render_spectrum(spec, modes)
    render_stack(names, planes_t, args.out)


if __name__ == "__main__":
    main()
