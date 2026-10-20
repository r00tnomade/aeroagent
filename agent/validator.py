# Copyright 2026 Sami Adam Moughli
# SPDX-License-Identifier: Apache-2.0
"""Deterministic safety layer between Nemotron and the flight controller (AGENTS.md, section 7).

Every LLM answer goes through `IntentValidator.offer()`; every timeout / HTTP error / exception
goes through `IntentValidator.fail()`. The controller reads `IntentValidator.current(now)` once per
tick and never sees raw model output.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Callable

MODES = ("HOLD", "TRACK", "GUST_RIDE", "RETREAT", "LAND")
SAFE_MODES = ("HOLD", "LAND")  # switching to these skips hysteresis


@dataclass(frozen=True)
class Intent:
    mode: str = "HOLD"
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_rate_dps: float = 0.0
    thrust_norm: float = 0.5
    pid_gain_scale: float = 1.0
    waypoint_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_tilt_deg: float = 25.0
    confidence: float = 1.0
    rationale: str = ""

    def to_si(self) -> dict:
        """Controller-facing values in SI units (rad, rad/s). Degrees exist only at the LLM boundary."""
        return {
            "mode": self.mode,
            "roll_rad": math.radians(self.roll_deg),
            "pitch_rad": math.radians(self.pitch_deg),
            "yaw_rate_rps": math.radians(self.yaw_rate_dps),
            "thrust_bias": self.thrust_norm - 0.5,  # a bias, never a setpoint
            "pid_gain_scale": self.pid_gain_scale,
            "waypoint_offset_m": self.waypoint_offset_m,
            "max_tilt_rad": math.radians(self.max_tilt_deg),
        }


HOLD_DEFAULT = Intent()
SAFE_INTENT = Intent(
    mode="HOLD", pid_gain_scale=1.0, max_tilt_deg=15.0, rationale="SAFE latch"
)


@dataclass(frozen=True)
class ValidatorConfig:
    roll_pitch_max_deg: float = 25.0
    yaw_rate_max_dps: float = 90.0
    thrust_min: float = 0.2
    thrust_max: float = 0.85
    gain_min: float = 0.7
    gain_max: float = 1.5
    gain_step_max: float = 0.2  # per accepted cycle
    waypoint_max_m: float = 2.0  # per axis
    tilt_min_deg: float = 5.0
    tilt_max_deg: float = 25.0
    mode_hysteresis_cycles: int = 2
    low_confidence: float = 0.4
    stale_s: float = 1.5  # answers older than this (since request) are discarded
    timeout_s: float = 1.5  # no fresh intent for this long -> start decaying to HOLD
    decay_s: float = 2.0  # time constant-ish: fully HOLD after ~decay_s
    failures_to_latch: int = 3
    valid_to_unlatch: int = 3
    safe_tilt_deg: float = 15.0


@dataclass
class Decision:
    intent: Intent
    accepted: bool
    overrides: list[str] = field(default_factory=list)
    safe_latched: bool = False


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _num(d: dict, key: str) -> float:
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{key} is not a finite number")
    return float(v)


def parse_intent(raw: str | dict) -> Intent:
    """Strict structural parse. Raises ValueError on anything malformed. Does NOT clamp."""
    d = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(d, dict):
        raise ValueError("intent is not a JSON object")
    mode = d["mode"]
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    att = d["target_attitude"]
    wp = d["waypoint_offset_m"]
    if not isinstance(wp, (list, tuple)) or len(wp) != 3:
        raise ValueError("waypoint_offset_m must have 3 elements")
    wp_vals = tuple(_num({"w": w}, "w") for w in wp)
    rationale = d.get("rationale_short", "")
    return Intent(
        mode=mode,
        roll_deg=_num(att, "roll_deg"),
        pitch_deg=_num(att, "pitch_deg"),
        yaw_rate_dps=_num(att, "yaw_rate_dps"),
        thrust_norm=_num(d, "thrust_norm"),
        pid_gain_scale=_num(d, "pid_gain_scale"),
        waypoint_offset_m=wp_vals,  # type: ignore[arg-type]
        max_tilt_deg=_num(d, "max_tilt_deg"),
        confidence=_num(d, "confidence"),
        rationale=str(rationale)[:120],
    )


def _blend(a: Intent, b: Intent, w: float) -> Intent:
    """(1-w)*a + w*b for numeric fields; mode/rationale from b."""
    mix = lambda x, y: (1 - w) * x + w * y  # noqa: E731
    return replace(
        b,
        roll_deg=mix(a.roll_deg, b.roll_deg),
        pitch_deg=mix(a.pitch_deg, b.pitch_deg),
        yaw_rate_dps=mix(a.yaw_rate_dps, b.yaw_rate_dps),
        thrust_norm=mix(a.thrust_norm, b.thrust_norm),
        pid_gain_scale=mix(a.pid_gain_scale, b.pid_gain_scale),
        waypoint_offset_m=tuple(mix(x, y) for x, y in zip(a.waypoint_offset_m, b.waypoint_offset_m)),  # type: ignore[arg-type]
        max_tilt_deg=mix(a.max_tilt_deg, b.max_tilt_deg),
        confidence=mix(a.confidence, b.confidence),
    )


class IntentValidator:
    def __init__(
        self,
        cfg: ValidatorConfig | None = None,
        waypoint_ok: Callable[[tuple[float, float, float]], bool] | None = None,
    ):
        self.cfg = cfg or ValidatorConfig()
        self.waypoint_ok = waypoint_ok
        self.last = HOLD_DEFAULT
        self.last_valid_at: float | None = None
        self.failures = 0
        self.valid_streak = 0
        self.safe_latched = False
        self._pending_mode: str | None = None
        self._pending_count = 0
        self.override_log: list[tuple[float, str]] = []

    # ------------------------------------------------------------------ public API
    def offer(self, raw: str | dict, sent_at: float, now: float) -> Decision:
        if now - sent_at > self.cfg.stale_s:
            return self._reject(now, f"stale ({now - sent_at:.2f} s old)")
        try:
            cand = parse_intent(raw)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return self._reject(now, f"parse/schema: {e}")

        overrides: list[str] = []
        cand = self._clamp(cand, overrides)
        cand = self._check_waypoint(cand, overrides)
        cand = self._rate_limit(cand, overrides)
        cand = self._hysteresis(cand, overrides)
        if cand.confidence < self.cfg.low_confidence:
            cand = _blend(self.last, cand, 0.5)
            overrides.append(f"low confidence -> 50/50 blend with last intent")

        self.failures = 0
        self.valid_streak += 1
        self.last_valid_at = now
        if self.safe_latched:
            if self.valid_streak >= self.cfg.valid_to_unlatch:
                self.safe_latched = False
                overrides.append("SAFE latch released")
            else:
                overrides.append(
                    f"SAFE latch active ({self.valid_streak}/{self.cfg.valid_to_unlatch} valid)"
                )
                self._log(now, overrides)
                return Decision(
                    self._safe(), accepted=False, overrides=overrides, safe_latched=True
                )
        self.last = cand
        self._log(now, overrides)
        return Decision(cand, accepted=True, overrides=overrides)

    def fail(self, reason: str, now: float) -> Decision:
        """Call on timeout, HTTP 429/5xx, network error or any exception from the LLM call."""
        return self._reject(now, f"failure: {reason}")

    def current(self, now: float) -> Intent:
        """Intent the controller should use this tick (applies SAFE latch and timeout decay)."""
        if self.safe_latched:
            return self._safe()
        if self.last_valid_at is None:
            return HOLD_DEFAULT
        age = now - self.last_valid_at
        if age <= self.cfg.timeout_s:
            return self.last
        w = _clamp((age - self.cfg.timeout_s) / self.cfg.decay_s, 0.0, 1.0)
        decayed = _blend(self.last, HOLD_DEFAULT, w)
        return replace(decayed, mode="HOLD" if w >= 1.0 else self.last.mode)

    # ------------------------------------------------------------------ rules
    def _reject(self, now: float, reason: str) -> Decision:
        self.failures += 1
        self.valid_streak = 0
        reasons = [reason]
        if not self.safe_latched and self.failures >= self.cfg.failures_to_latch:
            self.safe_latched = True
            reasons.append(
                f"SAFE latch engaged after {self.failures} consecutive failures"
            )
        self._log(now, reasons)
        return Decision(
            self.current(now),
            accepted=False,
            overrides=reasons,
            safe_latched=self.safe_latched,
        )

    def _clamp(self, c: Intent, ov: list[str]) -> Intent:
        k = self.cfg
        fixed = replace(
            c,
            roll_deg=_clamp(c.roll_deg, -k.roll_pitch_max_deg, k.roll_pitch_max_deg),
            pitch_deg=_clamp(c.pitch_deg, -k.roll_pitch_max_deg, k.roll_pitch_max_deg),
            yaw_rate_dps=_clamp(
                c.yaw_rate_dps, -k.yaw_rate_max_dps, k.yaw_rate_max_dps
            ),
            thrust_norm=_clamp(c.thrust_norm, k.thrust_min, k.thrust_max),
            pid_gain_scale=_clamp(c.pid_gain_scale, k.gain_min, k.gain_max),
            waypoint_offset_m=tuple(_clamp(w, -k.waypoint_max_m, k.waypoint_max_m) for w in c.waypoint_offset_m),  # type: ignore[arg-type]
            max_tilt_deg=_clamp(c.max_tilt_deg, k.tilt_min_deg, k.tilt_max_deg),
            confidence=_clamp(c.confidence, 0.0, 1.0),
        )
        changed = [
            f
            for f in (
                "roll_deg",
                "pitch_deg",
                "yaw_rate_dps",
                "thrust_norm",
                "pid_gain_scale",
                "waypoint_offset_m",
                "max_tilt_deg",
                "confidence",
            )
            if getattr(fixed, f) != getattr(c, f)
        ]
        if changed:
            ov.append("clamped: " + ", ".join(changed))
        return fixed

    def _check_waypoint(self, c: Intent, ov: list[str]) -> Intent:
        if (
            self.waypoint_ok is not None
            and any(c.waypoint_offset_m)
            and not self.waypoint_ok(c.waypoint_offset_m)
        ):
            ov.append("waypoint offset violates geofence/obstacle -> 0")
            return replace(c, waypoint_offset_m=(0.0, 0.0, 0.0))
        return c

    def _rate_limit(self, c: Intent, ov: list[str]) -> Intent:
        step = self.cfg.gain_step_max
        d = c.pid_gain_scale - self.last.pid_gain_scale
        if abs(d) > step + 1e-9:
            ov.append(f"gain change {d:+.2f} limited to ±{step:.2f}")
            return replace(
                c, pid_gain_scale=self.last.pid_gain_scale + math.copysign(step, d)
            )
        return c

    def _hysteresis(self, c: Intent, ov: list[str]) -> Intent:
        if c.mode == self.last.mode:
            self._pending_mode, self._pending_count = None, 0
            return c
        if c.mode in SAFE_MODES:
            self._pending_mode, self._pending_count = None, 0
            return c
        if c.mode == self._pending_mode:
            self._pending_count += 1
        else:
            self._pending_mode, self._pending_count = c.mode, 1
        if self._pending_count >= self.cfg.mode_hysteresis_cycles:
            self._pending_mode, self._pending_count = None, 0
            return c
        ov.append(
            f"mode {c.mode} pending ({self._pending_count}/{self.cfg.mode_hysteresis_cycles})"
        )
        return replace(c, mode=self.last.mode)

    def _safe(self) -> Intent:
        return replace(SAFE_INTENT, max_tilt_deg=self.cfg.safe_tilt_deg)

    def _log(self, now: float, reasons: list[str]) -> None:
        for r in reasons:
            self.override_log.append((now, r))

    @property
    def override_count(self) -> int:
        return len(self.override_log)
