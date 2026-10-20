# Copyright 2026 Sami Adam Moughli
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from agent.validator import HOLD_DEFAULT, IntentValidator, ValidatorConfig


def intent(**kw):
    d = {
        "mode": "TRACK",
        "target_attitude": {"roll_deg": 0, "pitch_deg": 0, "yaw_rate_dps": 0},
        "thrust_norm": 0.5,
        "pid_gain_scale": 1.0,
        "waypoint_offset_m": [0, 0, 0],
        "max_tilt_deg": 25,
        "confidence": 0.9,
        "rationale_short": "test",
    }
    att = {k: kw.pop(k) for k in ("roll_deg", "pitch_deg", "yaw_rate_dps") if k in kw}
    d["target_attitude"].update(att)
    d.update(kw)
    return d


def accepted_track(v, t=0.0):
    """Get the validator into TRACK (needs 2 cycles of hysteresis)."""
    v.offer(intent(), sent_at=t, now=t + 0.1)
    return v.offer(intent(), sent_at=t + 1, now=t + 1.1)


def test_valid_intent_accepted():
    v = IntentValidator()
    d = accepted_track(v)
    assert d.accepted and d.intent.mode == "TRACK" and not d.safe_latched


def test_json_string_and_garbage():
    v = IntentValidator()
    assert v.offer(json.dumps(intent()), 0, 0.1).accepted
    d = v.offer("not json at all", 1, 1.1)
    assert not d.accepted and "parse" in d.overrides[0]


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "FLIP"},
        {"thrust_norm": "high"},
        {"waypoint_offset_m": [1, 2]},
        {"pid_gain_scale": float("nan")},
        {"confidence": True},
    ],
)
def test_schema_violations_rejected_and_last_kept(bad):
    v = IntentValidator()
    accepted_track(v)
    d = v.offer(
        intent(**bad), 1.5, 1.6
    )  # within timeout of the last valid intent (t=1.1)
    assert not d.accepted
    assert d.intent.mode == "TRACK"  # last valid intent kept


def test_missing_field_rejected():
    v = IntentValidator()
    raw = intent()
    del raw["max_tilt_deg"]
    assert not v.offer(raw, 0, 0.1).accepted


def test_clamping():
    v = IntentValidator()
    accepted_track(v)
    d = v.offer(
        intent(
            roll_deg=60,
            yaw_rate_dps=-500,
            thrust_norm=1.2,
            max_tilt_deg=90,
            waypoint_offset_m=[5, -5, 0.5],
        ),
        2,
        2.1,
    )
    i = d.intent
    assert (i.roll_deg, i.yaw_rate_dps, i.thrust_norm, i.max_tilt_deg) == (
        25,
        -90,
        0.85,
        25,
    )
    assert i.waypoint_offset_m == (2.0, -2.0, 0.5)
    assert any("clamped" in o for o in d.overrides)


def test_gain_rate_limit():
    v = IntentValidator()
    accepted_track(v)
    d = v.offer(intent(pid_gain_scale=1.5), 2, 2.1)
    assert d.intent.pid_gain_scale == pytest.approx(1.2)
    d = v.offer(intent(pid_gain_scale=1.5), 3, 3.1)
    assert d.intent.pid_gain_scale == pytest.approx(1.4)


def test_mode_hysteresis_and_safe_mode_bypass():
    v = IntentValidator()
    accepted_track(v)
    d = v.offer(intent(mode="GUST_RIDE"), 2, 2.1)
    assert d.intent.mode == "TRACK" and any("pending" in o for o in d.overrides)
    d = v.offer(intent(mode="GUST_RIDE"), 3, 3.1)
    assert d.intent.mode == "GUST_RIDE"
    d = v.offer(intent(mode="HOLD"), 4, 4.1)  # HOLD/LAND switch immediately
    assert d.intent.mode == "HOLD"


def test_low_confidence_blends():
    v = IntentValidator()
    accepted_track(v)
    d = v.offer(intent(roll_deg=10, confidence=0.2), 2, 2.1)
    assert d.intent.roll_deg == pytest.approx(5.0)


def test_stale_answer_discarded():
    v = IntentValidator()
    accepted_track(v)
    d = v.offer(intent(roll_deg=10), sent_at=2.0, now=4.0)
    assert not d.accepted and "stale" in d.overrides[0]


def test_geofence_zeroes_waypoint():
    v = IntentValidator(waypoint_ok=lambda wp: wp[0] < 1.0)
    accepted_track(v)
    d = v.offer(intent(waypoint_offset_m=[1.5, 0, 0]), 2, 2.1)
    assert d.intent.waypoint_offset_m == (0.0, 0.0, 0.0)


def test_timeout_decays_to_hold():
    v = IntentValidator(ValidatorConfig(timeout_s=1.5, decay_s=2.0))
    v.offer(intent(roll_deg=10), 0, 0.1)
    v.offer(intent(roll_deg=10), 0.9, 1.0)  # last valid at t=1.0, mode TRACK
    assert v.current(2.0).roll_deg == pytest.approx(10)  # within timeout
    mid = v.current(1.0 + 1.5 + 1.0)  # halfway through decay
    assert mid.roll_deg == pytest.approx(5.0)
    end = v.current(1.0 + 1.5 + 2.5)
    assert end.mode == "HOLD" and end.roll_deg == pytest.approx(0.0)


def test_safe_latch_after_three_failures_and_release():
    v = IntentValidator()
    accepted_track(v)
    v.fail("timeout", 3)
    v.fail("HTTP 429", 4)
    d = v.fail("HTTP 500", 5)
    assert (
        d.safe_latched
        and v.current(5.1).max_tilt_deg == 15
        and v.current(5.1).mode == "HOLD"
    )
    for k in range(2):  # 2 valid answers: still latched
        d = v.offer(intent(), 6 + k, 6.1 + k)
        assert d.safe_latched and not d.accepted
    d = v.offer(intent(), 8, 8.1)  # 3rd valid answer releases the latch
    assert not d.safe_latched and d.accepted


def test_no_intent_yet_is_hold():
    assert IntentValidator().current(0.0) == HOLD_DEFAULT


def test_to_si_units():
    v = IntentValidator()
    accepted_track(v)
    si = v.offer(intent(roll_deg=10), 2, 2.1).intent.to_si()
    assert si["roll_rad"] == pytest.approx(0.174533, rel=1e-4)
    assert si["thrust_bias"] == pytest.approx(0.0)
