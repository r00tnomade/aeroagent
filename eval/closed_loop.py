# Copyright 2026 <Your Name>
# SPDX-License-Identifier: Apache-2.0
"""Full AeroAgent loop in REAL TIME: Tier 0 PID (240 Hz) + Tier 1 FNO feedforward (48 Hz)
+ Tier 2 supervisor (~1 Hz, asynchronous, through the validator).

  python -m eval.closed_loop                         # no supervisor vs. rules vs. Nemotron 3 Super
  python -m eval.closed_loop --policies nemotron     # only Nemotron (prints every decision live)
  python -m eval.closed_loop --model nvidia/Nemotron-3_5-Lightning

The simulation is paced to wall-clock time so the measured LLM latency has its real effect.
Writes results/closed_loop_<run>.json, results/decisions_<policy>_<run>.jsonl, docs/figures/closed_loop_<run>.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from agent.validator import IntentValidator, ValidatorConfig
from eval.ablation import default_run, hover_target

# 1 Hz decisions vs. ~2 s gusts: one confirmation cycle is all the time there is. Documented deviation
# from the default hysteresis of 2 (AGENTS.md section 7); everything else uses the default validator rules.
CLOSED_LOOP_VALIDATOR = ValidatorConfig(
    mode_hysteresis_cycles=1, stale_s=3.0, timeout_s=3.5, decay_s=2.0
)


def run(
    policy_name: str | None,
    run_path: Path,
    ckpt: Path,
    x_behind: float,
    duration: float,
    model: str,
    period: float,
    realtime: bool = True,
    timeout_s: float = 6.0,
) -> dict:
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    from agent.features import FeatureBuilder
    from agent.supervisor import RulePolicy, SupervisorWorker
    from control.reflex import FNOWindEstimator, drag_force, feedforward_target_vel
    from sim.wind_aviary import GridWind, WindAviary

    hover = hover_target(run_path, x_behind)
    est = FNOWindEstimator(ckpt, run_path)
    raw = np.load(run_path)
    solid = np.argwhere(raw["mask"] > 0)
    x_obs = float(raw["x"][int(round(solid[:, 1].mean()))])
    builder = FeatureBuilder(est, hover, x_obs)

    use_ff = (
        policy_name != "pid"
    )  # "pid": plain PID baseline, no FNO feedforward, no supervisor
    worker = None
    # "hybrid": Nemotron supervises, and the rule set acts as an instant MINIMUM (guardrail) at 10 Hz:
    # if the rules say GUST_RIDE, gains are at least 1.3 no matter what the LLM says or how late it is.
    use_floor = policy_name == "hybrid"
    rules = RulePolicy() if use_floor else None
    floor_active, floor_ticks = False, 0
    if policy_name in ("rules", "nemotron", "hybrid"):
        spec = {
            "name": "nemotron" if use_floor else policy_name,
            "model": model,
            "timeout_s": timeout_s,
        }
        worker = SupervisorWorker(
            spec,
            IntentValidator(CLOSED_LOOP_VALIDATOR, waypoint_ok=builder.waypoint_ok),
            period,
        )
        est.estimate(hover, 0.0)
        f0 = builder.build(
            0.0,
            hover,
            np.zeros(3),
            np.zeros(3),
            hover,
            {"mode": "TRACK", "pid_gain_scale": 1.0},
        )
        lat = worker.warmup(f0)
        print(
            f"  warm-up latency via worker process: {', '.join(f'{x:.2f} s' for x in lat) or 'failed'}"
        )

    wind_gt = GridWind(str(run_path))
    env = WindAviary(
        wind=wind_gt,
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=hover.reshape(1, 3),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=240,
        gui=False,
        show_wind=False,
    )
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    P0, D0 = np.array(ctrl.P_COEFF_FOR, float), np.array(ctrl.D_COEFF_FOR, float)
    obs, _ = env.reset(seed=42)
    every = max(1, int(round(env.CTRL_FREQ / 48.0)))
    steps = int(duration * env.CTRL_FREQ)
    target, dv, gain = hover.copy(), np.zeros(3), 1.0
    log = {
        k: []
        for k in (
            "t",
            "err_hover",
            "err_target",
            "target",
            "gain",
            "mode",
            "force",
            "tick_t",
            "pos",
            "rpy",
            "wind",
            "wind_hat",
        )
    }
    w_hat = np.zeros(3)
    last_intent = {"mode": "TRACK", "pid_gain_scale": 1.0}
    t_wall0 = time.perf_counter()
    for k in range(steps):
        t = k * env.CTRL_TIMESTEP
        state = obs[0]
        pos, vel, rpy = state[0:3], state[10:13], np.degrees(state[7:10])
        if k % every == 0:  # Tier 1 (+ Tier 2 bookkeeping)
            mode = "pid" if not use_ff else "none"
            if worker is not None:
                worker.poll()
                intent = worker.current()
                mode, g_cmd = intent.mode, float(intent.pid_gain_scale)
                if use_floor:
                    if (
                        k % (every * 5) == 0
                    ):  # rule guardrail at ~10 Hz, zero network latency
                        f_now = builder.build(t, pos, vel, rpy, target, last_intent)
                        floor_active = json.loads(rules(f_now))["mode"] == "GUST_RIDE"
                    if floor_active and g_cmd < 1.3:
                        g_cmd, mode = 1.3, mode + "+floor"
                        floor_ticks += 1
                    # same ramp the validator enforces on the LLM: <= 0.2 per 0.5 s
                    gain += float(
                        np.clip(
                            g_cmd - gain,
                            -0.4 * every * env.CTRL_TIMESTEP,
                            0.4 * every * env.CTRL_TIMESTEP,
                        )
                    )
                else:
                    gain = g_cmd
                desired = hover + np.asarray(intent.waypoint_offset_m, float)
                step = (
                    0.3 * every * env.CTRL_TIMESTEP
                )  # move the set-point at <= 0.3 m/s
                target = target + np.clip(desired - target, -step, step)
                ctrl.P_COEFF_FOR, ctrl.D_COEFF_FOR = P0 * gain, D0 * gain
                last_intent = {"mode": mode, "pid_gain_scale": round(gain, 2)}
                if worker.ready():
                    f = builder.build(t, pos, vel, rpy, target, last_intent)
                    worker.maybe_submit(f, t)
            w_hat = est.estimate(
                pos, t
            ).wind  # computed for every policy (logged for the video), used only with FF
            builder.observe(t)
            dv = (
                feedforward_target_vel(
                    drag_force(w_hat, vel, env.cd_a),
                    np.asarray(ctrl.D_COEFF_FOR, float),
                )
                if use_ff
                else np.zeros(3)
            )
            log["mode"].append(mode)
            log["gain"].append(gain)
            log["tick_t"].append(t)
        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target,
            target_rpy=np.zeros(3),
            target_vel=dv,
        )
        obs, _, _, _, info = env.step(rpm.reshape(1, 4))
        p_new = obs[0][0:3]
        log["t"].append(t + env.CTRL_TIMESTEP)
        log["err_hover"].append(float(np.linalg.norm(p_new - hover)))
        log["err_target"].append(float(np.linalg.norm(p_new - target)))
        log["target"].append(target.copy())
        log["pos"].append(p_new.copy())
        log["rpy"].append(obs[0][7:10].copy())
        log["wind"].append(
            info["wind"][0].copy() if info["wind"] is not None else np.zeros(3)
        )
        log["wind_hat"].append(w_hat.copy())
        log["force"].append(
            float(np.linalg.norm(info["wind_force"][0]))
            if info["wind_force"] is not None
            else 0.0
        )
        if realtime:  # pace to wall clock so LLM latency is real
            lag = t_wall0 + (k + 1) * env.CTRL_TIMESTEP - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
    loop_wall = (
        time.perf_counter() - t_wall0
    )  # measured BEFORE waiting for the last in-flight LLM answer
    env.close()
    if worker is not None:
        worker.join()
    out = {k: np.asarray(v) for k, v in log.items()}
    out["decisions"] = worker.log if worker else []
    out["hover"] = hover
    out["realtime_factor"] = duration / loop_wall
    out["floor_ticks"] = floor_ticks
    return out


def summarize(name: str, r: dict, settle: float = 0.5) -> dict:
    keep = r["t"] >= settle
    eh, et = r["err_hover"][keep], r["err_target"][keep]
    d = r["decisions"]
    lat = [x["latency_s"] for x in d if x["error"] is None]
    m = {
        "policy": name,
        "rms_hover_cm": float(np.sqrt(np.mean(eh**2)) * 100),
        "max_hover_cm": float(eh.max() * 100),
        "rms_target_cm": float(np.sqrt(np.mean(et**2)) * 100),
        "mean_wind_force_mN": float(np.mean(r["force"][keep]) * 1000),
        "decisions": len(d),
        "accepted": sum(x["accepted"] for x in d),
        "errors": sum(x["error"] is not None for x in d),
        "overrides": sum(len(x["overrides"]) for x in d),
        "safe_latch_events": sum(x["safe_latched"] for x in d),
        "llm_latency_p50_s": float(np.median(lat)) if lat else None,
        "llm_latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
        "modes": sorted(set(str(x) for x in r["mode"])),
        "realtime_factor": float(r["realtime_factor"]),
        "floor_ticks": int(r.get("floor_ticks", 0)),
    }
    return m


def plot(results: dict, summ: dict, raw, run_name: str) -> None:
    import matplotlib.pyplot as plt

    from viz.make_figures import AQUA, BLUE, GRAY, INK2, ORANGE, _save

    colors = {
        "pid": ORANGE,
        "none": GRAY,
        "rules": AQUA,
        "nemotron": BLUE,
        "hybrid": INK2,
    }
    labels = {
        "pid": "PID only (no FNO)",
        "none": "FNO feedforward, no supervisor",
        "rules": "+ rule supervisor",
        "nemotron": "+ Nemotron supervisor",
        "hybrid": "+ Nemotron with rule guardrail",
    }
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 3, 2]},
    )
    t_g = np.arange(len(raw["gust"])) * float(raw["dt"])
    tmax = max(r["t"][-1] for r in results.values())
    keep = t_g <= tmax
    axes[0].fill_between(t_g[keep], 0, raw["gust"][keep], color=GRAY, linewidth=0)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_yticks([0, 1])
    axes[0].set_ylabel("gust\nfactor")
    axes[0].set_title("Inlet gust")
    for n, r in results.items():
        s = summ[n]
        axes[1].plot(
            r["t"],
            r["err_hover"] * 100,
            color=colors[n],
            linewidth=2.2 if n == "nemotron" else 1.6,
            label=f"{labels[n]}: RMS {s['rms_hover_cm']:.2f} cm, max {s['max_hover_cm']:.2f} cm"
            + (
                f"  (mean of {len(s['repeats'])}: {s['rms_hover_cm_mean']:.2f} ± {s['rms_hover_cm_std']:.2f})"
                if len(s.get("repeats", [])) > 1
                else ""
            ),
        )
    axes[1].set_ylabel("distance to hover point [cm]")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Real-time closed loop, held-out run")
    axes[1].legend(loc="upper left", fontsize=8.5)
    if "nemotron" in results:
        r = results["nemotron"]
        axes[2].step(
            r["tick_t"],
            r["gain"],
            where="post",
            color=BLUE,
            label="PID gain scale (validated)",
        )
        for d in r["decisions"]:
            if d["accepted"]:
                axes[2].annotate(
                    d["intent"]["mode"],
                    (d["sim_t"] + d["latency_s"], d["intent"]["pid_gain_scale"]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    fontsize=7,
                    color=INK2,
                    ha="center",
                )
        s = summ["nemotron"]
        axes[2].set_title(
            f"Nemotron decisions: {s['decisions']} calls, latency p50 {s['llm_latency_p50_s'] or 0:.2f} s, "
            f"{s['overrides']} validator overrides"
        )
        axes[2].set_ylabel("gain scale")
        axes[2].set_ylim(0.6, 1.6)
    axes[2].set_xlabel("time [s]")
    _save(
        fig,
        f"closed_loop_{run_name}",
        "Simulation paced to wall-clock time; Nemotron via Nebius Token Factory, strict JSON schema, reasoning off.",
    )
    _ = ORANGE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/fno.pt"))
    ap.add_argument("--x", type=float, default=1.2)
    ap.add_argument("--duration", type=float, default=7.4)
    ap.add_argument("--period", type=float, default=1.0, help="supervisor period [s]")
    ap.add_argument(
        "--model",
        default="nvidia/Nemotron-3_5-Lightning",
        help="Lightning (eu-north1) is the default; Super (us-central1) measured 1.1 s at 17:00 "
        "but 11 s p50 at 22:38 on 20 Oct 2026",
    )
    ap.add_argument("--policies", nargs="*", default=["none", "rules", "nemotron"])
    ap.add_argument(
        "--timeout", type=float, default=6.0, help="Nemotron request timeout [s]"
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeats for supervised policies (LLM timing varies)",
    )
    args = ap.parse_args()

    run_path = args.run or default_run()
    raw = np.load(run_path)
    results, summ = {}, {}
    for p in args.policies:
        reps = (
            1 if p in ("pid", "none") else args.repeats
        )  # Tier 0/1 are deterministic; Tier 2 is not
        runs_p = []
        for rep in range(reps):
            print(
                f"\n[{p}] {run_path.name}, real time, {args.duration:.1f} s"
                + (f", repeat {rep + 1}/{reps}" if reps > 1 else "")
            )
            r = run(
                None if p == "none" else p,
                run_path,
                args.ckpt,
                args.x,
                args.duration,
                args.model,
                args.period,
                timeout_s=args.timeout,
            )
            s = summarize(p, r)
            runs_p.append(s)
            print(
                f"  RMS to hover point {s['rms_hover_cm']:.2f} cm (max {s['max_hover_cm']:.2f}) | "
                f"decisions {s['decisions']} (accepted {s['accepted']}, errors {s['errors']}) | "
                f"LLM p50 {s['llm_latency_p50_s'] or 0:.2f} s | real-time factor {s['realtime_factor']:.2f}"
            )
            if rep == 0:
                results[p] = r
                Path("results").mkdir(exist_ok=True)
                np.savez_compressed(
                    f"results/traj_{p}_{run_path.stem}.npz",
                    **{
                        k: r[k]
                        for k in (
                            "t",
                            "pos",
                            "rpy",
                            "wind",
                            "wind_hat",
                            "target",
                            "err_hover",
                            "gain",
                            "tick_t",
                        )
                    },
                    hover=r["hover"],
                    decisions=json.dumps(r["decisions"]),
                )
            if r["decisions"]:
                with open(
                    f"results/decisions_{p}_{run_path.stem}_rep{rep}.jsonl", "w"
                ) as fh:
                    for d in r["decisions"]:
                        fh.write(json.dumps(d) + "\n")
        agg = dict(runs_p[0])
        for key in ("rms_hover_cm", "max_hover_cm"):
            vals = np.array([x[key] for x in runs_p])
            agg[key + "_mean"], agg[key + "_std"] = float(vals.mean()), (
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            )
        agg["repeats"] = runs_p
        summ[p] = agg
        if len(runs_p) > 1:
            print(
                f"  => {p}: RMS {agg['rms_hover_cm_mean']:.2f} ± {agg['rms_hover_cm_std']:.2f} cm, "
                f"max {agg['max_hover_cm_mean']:.2f} ± {agg['max_hover_cm_std']:.2f} cm over {len(runs_p)} runs"
            )
    Path("results").mkdir(exist_ok=True)
    Path(f"results/closed_loop_{run_path.stem}.json").write_text(
        json.dumps(summ, indent=2)
    )
    print(f"\nsaved results/closed_loop_{run_path.stem}.json")
    plot(results, summ, raw, run_path.stem)


if __name__ == "__main__":
    main()
