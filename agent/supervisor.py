# Copyright 2026 <Your Name>
# SPDX-License-Identifier: Apache-2.0
"""Tier 2: asynchronous supervisor. Exactly one request in flight; the control loop never waits for it.

NemotronPolicy  - Nemotron 3 via Nebius Token Factory (OpenAI-compatible), strict JSON schema, reasoning off
RulePolicy      - deterministic baseline with the same interface (fallback and fair A/B comparison)
SupervisorWorker- background thread: takes the latest features, calls the policy, feeds the validator
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agent.prompts import INTENT_SCHEMA, SYSTEM_PROMPT
from agent.validator import IntentValidator

DEFAULT_MODEL = (
    "nvidia/Nemotron-3_5-Lightning"  # eu-north1; Super (us-central1) is the fallback
)
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) so the key never has to be typed into the shell."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class NemotronPolicy:
    name = "nemotron"

    def __init__(self, model: str = DEFAULT_MODEL, timeout_s: float = 6.0):
        from openai import OpenAI

        load_dotenv()
        key = os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("NEBIUS_API_KEY not set (environment or .env)")
        self.client = OpenAI(
            base_url=BASE_URL, api_key=key, timeout=timeout_s, max_retries=0
        )
        self.model = model

    def warmup(self, features: dict, n: int = 2) -> list[float]:
        """Open the HTTPS connection and measure latency with the real prompt BEFORE the control loop starts."""
        out = []
        for _ in range(n):
            t0 = time.monotonic()
            try:
                self(features)
                out.append(time.monotonic() - t0)
            except Exception as e:  # noqa: BLE001
                print(
                    f"  warm-up call failed after {time.monotonic() - t0:.2f} s: {type(e).__name__}: {str(e)[:120]}"
                )
        return out

    def __call__(self, features: dict) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(features, separators=(",", ":")),
                },
            ],
            temperature=0.1,
            top_p=1.0,
            max_tokens=200,
            seed=42,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "aero_intent",
                    "strict": True,
                    "schema": INTENT_SCHEMA,
                },
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return r.choices[0].message.content or ""


class RulePolicy:
    """Hand-written rules on the same features. No network, instant."""

    name = "rules"

    def __call__(self, f: dict) -> str:
        fl = f["flow"]
        intent = {
            "mode": "TRACK",
            "target_attitude": {"roll_deg": 0, "pitch_deg": 0, "yaw_rate_dps": 0},
            "thrust_norm": 0.5,
            "pid_gain_scale": 1.0,
            "waypoint_offset_m": [0, 0, 0],
            "max_tilt_deg": 25,
            "confidence": 0.9,
            "rationale_short": "nominal",
        }
        if fl["fno_uncertainty"] > 0.6:
            intent.update(
                mode="HOLD", max_tilt_deg=15, rationale_short="flow estimate unreliable"
            )
        elif fl["gust_level"] > 0.1 or (
            fl["gust_eta_s"] is not None and fl["gust_eta_s"] < 1.5
        ):
            intent.update(
                mode="GUST_RIDE",
                pid_gain_scale=1.3,
                max_tilt_deg=18,
                rationale_short="gust arriving",
            )
        return json.dumps(intent)


def _policy_process(spec: dict, req_q, resp_q) -> None:
    """Runs in its own process: owns the HTTPS client, so network I/O never competes with the control loop."""
    policy = (
        NemotronPolicy(spec["model"], spec["timeout_s"])
        if spec["name"] == "nemotron"
        else RulePolicy()
    )
    while True:
        item = req_q.get()
        if item is None:
            break
        features, sent_at, sim_t = item
        raw, err = None, None
        try:
            raw = policy(features)
        except Exception as e:  # noqa: BLE001 - every failure goes to the validator
            err = f"{type(e).__name__}: {str(e)[:120]}"
        resp_q.put((raw, err, time.monotonic(), sent_at, sim_t, features))


class SupervisorWorker:
    """Tier 2 front-end used by the control loop. Requests go to a separate process; the loop calls poll()
    every tick to hand finished answers to the validator. Exactly one request is in flight at any time.
    """

    def __init__(
        self,
        spec: dict,
        validator: IntentValidator,
        period_s: float = 1.0,
        verbose: bool = True,
    ):
        import multiprocessing as mp

        self.spec, self.validator, self.period, self.verbose = (
            spec,
            validator,
            period_s,
            verbose,
        )
        self.name = spec["name"]
        ctx = mp.get_context("spawn")
        self.req_q, self.resp_q = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(
            target=_policy_process, args=(spec, self.req_q, self.resp_q), daemon=True
        )
        self.proc.start()
        self.busy, self.last_submit = False, -1e9
        self.log: list[dict] = []

    def ready(self) -> bool:
        return not self.busy and time.monotonic() - self.last_submit >= self.period

    def maybe_submit(self, features: dict, sim_t: float) -> bool:
        if not self.ready():
            return False
        now = time.monotonic()
        self.busy, self.last_submit = True, now
        self.req_q.put((features, now, sim_t))
        return True

    def poll(self) -> None:
        import queue

        while True:
            try:
                raw, err, done, sent_at, sim_t, features = self.resp_q.get_nowait()
            except queue.Empty:
                return
            self.busy = False
            d = (
                self.validator.fail(err, done)
                if err
                else self.validator.offer(raw, sent_at, done)
            )
            entry = {
                "sim_t": round(sim_t, 3),
                "latency_s": round(done - sent_at, 3),
                "raw": raw,
                "error": err,
                "accepted": d.accepted,
                "overrides": d.overrides,
                "safe_latched": d.safe_latched,
                "intent": {
                    "mode": d.intent.mode,
                    "pid_gain_scale": round(d.intent.pid_gain_scale, 3),
                    "waypoint_offset_m": [
                        round(v, 3) for v in d.intent.waypoint_offset_m
                    ],
                    "max_tilt_deg": round(d.intent.max_tilt_deg, 1),
                    "rationale": d.intent.rationale,
                },
                "features": features,
            }
            self.log.append(entry)
            if self.verbose:
                tag = "OK " if d.accepted else "REJ"
                why = err or ("; ".join(d.overrides) if d.overrides else "")
                print(
                    f"  [t={sim_t:5.2f}s] {self.name} {done - sent_at:4.2f}s {tag} {d.intent.mode:<9} "
                    f'gain {d.intent.pid_gain_scale:.2f}  "{d.intent.rationale}"'
                    + (f"  ({why})" if why else "")
                )

    def current(self):
        return self.validator.current(time.monotonic())

    def warmup(self, features: dict, n: int = 2, timeout: float = 15.0) -> list[float]:
        """Round-trips through the worker process before the loop starts (connection + latency check)."""
        out = []
        for _ in range(n):
            t0 = time.monotonic()
            self.req_q.put((features, t0, -1.0))
            try:
                raw, err, done, *_ = self.resp_q.get(timeout=timeout)
            except Exception:  # noqa: BLE001
                print("  warm-up: no answer")
                continue
            if err:
                print(f"  warm-up failed after {done - t0:.2f} s: {err}")
            else:
                out.append(done - t0)
        return out

    def join(self, timeout: float = 8.0) -> None:
        end = time.monotonic() + timeout
        while (
            self.busy and time.monotonic() < end
        ):  # collect the last in-flight answer for the log
            self.poll()
            time.sleep(0.02)
        self.req_q.put(None)
        self.proc.join(2.0)
        if self.proc.is_alive():
            self.proc.terminate()
