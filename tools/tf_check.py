"""Token Factory Day-0 check for AeroAgent.

Usage:
  export NEBIUS_API_KEY=...
  python tools/tf_check.py models                      # list all model IDs (Nemotron highlighted)
  python tools/tf_check.py bench --model <ID> --n 25   # latency + schema + reasoning check
  python tools/tf_check.py bench --model <ID> --n 25 --thinking   # same with reasoning ON (comparison)

Requires: pip install openai jsonschema
Cost: ~600 input + ~120 output tokens per call -> 25 calls on Super ~ $0.01 (estimate).
"""

import argparse
import json
import os
import random
import statistics
import time

from jsonschema import Draft202012Validator
from openai import OpenAI

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

SYSTEM_PROMPT = """You are AeroAgent-Supervisor, the high-level flight supervisor for a simulated quadrotor.
You do NOT fly the aircraft directly. A 240 Hz attitude controller and a 50 Hz position
controller with FNO wind feedforward stabilize the vehicle. You choose a MODE and bounded
adjustments once per decision cycle (~1 s).

Units: meters, m/s, degrees, deg/s, seconds. Frame: world ENU, z up. Thrust normalized 0..1 (hover ~ 0.5).
Hard limits (validator will clamp; do not exceed): |roll|,|pitch| <= 25 deg; |yaw_rate| <= 90 deg/s;
thrust_norm 0.2-0.85; pid_gain_scale 0.7-1.5; waypoint_offset <= 2.0 m per cycle.
Safety priorities (in order): 1) avoid ground/obstacles/geofence, 2) keep thrust headroom >= 15%,
3) minimize tracking error, 4) mission progress.
Modes: HOLD, TRACK, GUST_RIDE (raise gains, reduce tilt limit, lean into forecast wind),
RETREAT (move upwind/away from vortex region), LAND.
If flow uncertainty is high (fno_uncertainty > 0.3) or data is stale, prefer HOLD with conservative limits.
Respond ONLY with JSON matching the schema. rationale_short <= 120 characters. confidence in [0,1].

Examples:
IN {"flow":{"gust_max_ms":1.1,"gust_eta_s":null,"fno_uncertainty":0.08},"track_err_m":0.05}
OUT {"mode":"TRACK","target_attitude":{"roll_deg":0,"pitch_deg":0,"yaw_rate_dps":0},"thrust_norm":0.5,"pid_gain_scale":1.0,"waypoint_offset_m":[0,0,0],"max_tilt_deg":25,"confidence":0.9,"rationale_short":"Calm flow, nominal tracking."}
IN {"flow":{"gust_max_ms":6.2,"gust_eta_s":1.0,"gust_bearing_deg":250,"fno_uncertainty":0.12},"track_err_m":0.18}
OUT {"mode":"GUST_RIDE","target_attitude":{"roll_deg":-4,"pitch_deg":3,"yaw_rate_dps":0},"thrust_norm":0.56,"pid_gain_scale":1.3,"waypoint_offset_m":[0,0,0],"max_tilt_deg":15,"confidence":0.75,"rationale_short":"Gust front in 1.0 s from 250 deg; stiffen gains, cap tilt."}
IN {"flow":{"vorticity_peak":{"mag":5.0,"pos":[4.5,0.0,1.4]},"fno_uncertainty":0.41},"track_err_m":0.30}
OUT {"mode":"HOLD","target_attitude":{"roll_deg":0,"pitch_deg":0,"yaw_rate_dps":0},"thrust_norm":0.5,"pid_gain_scale":1.0,"waypoint_offset_m":[-0.5,0,0],"max_tilt_deg":15,"confidence":0.6,"rationale_short":"Vortex on path and high flow uncertainty; hold upwind."}"""

INTENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "mode",
        "target_attitude",
        "thrust_norm",
        "pid_gain_scale",
        "waypoint_offset_m",
        "max_tilt_deg",
        "confidence",
        "rationale_short",
    ],
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["HOLD", "TRACK", "GUST_RIDE", "RETREAT", "LAND"],
        },
        "target_attitude": {
            "type": "object",
            "additionalProperties": False,
            "required": ["roll_deg", "pitch_deg", "yaw_rate_dps"],
            "properties": {
                "roll_deg": {"type": "number", "minimum": -25, "maximum": 25},
                "pitch_deg": {"type": "number", "minimum": -25, "maximum": 25},
                "yaw_rate_dps": {"type": "number", "minimum": -90, "maximum": 90},
            },
        },
        "thrust_norm": {"type": "number", "minimum": 0.2, "maximum": 0.85},
        "pid_gain_scale": {"type": "number", "minimum": 0.7, "maximum": 1.5},
        "waypoint_offset_m": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "number", "minimum": -2, "maximum": 2},
        },
        "max_tilt_deg": {"type": "number", "minimum": 5, "maximum": 25},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale_short": {"type": "string", "maxLength": 120},
    },
}
VALIDATOR = Draft202012Validator(INTENT_SCHEMA)


def random_features(rng: random.Random) -> dict:
    """Varied synthetic scenarios: calm, gust front, vortex + high uncertainty."""
    kind = rng.choice(["calm", "gust", "vortex"])
    gust = {
        "calm": rng.uniform(0.3, 1.5),
        "gust": rng.uniform(4, 8),
        "vortex": rng.uniform(2, 4),
    }[kind]
    return {
        "t": round(rng.uniform(0, 60), 2),
        "state_age_ms": rng.randint(10, 60),
        "pose": {
            "pos": [round(rng.uniform(0, 6), 2), 0.0, 1.5],
            "vel": [0.8, 0.0, -0.1],
            "rpy_deg": [
                round(rng.uniform(-5, 5), 1),
                round(rng.uniform(-5, 5), 1),
                90.0,
            ],
        },
        "track_err_m": round(rng.uniform(0.02, 0.35), 2),
        "thrust_headroom": round(rng.uniform(0.15, 0.4), 2),
        "battery": 0.72,
        "flow": {
            "wind_at_drone": [round(gust * 0.6, 2), round(rng.uniform(-1, 1), 2), 0.0],
            "gust_max_ms": round(gust, 2),
            "gust_eta_s": round(rng.uniform(0.5, 2.5), 2) if kind == "gust" else None,
            "gust_bearing_deg": rng.randint(0, 359),
            "vorticity_peak": (
                {"mag": round(rng.uniform(4, 6), 1), "pos": [4.5, 0.0, 1.4]}
                if kind == "vortex"
                else None
            ),
            "turb_intensity": round(rng.uniform(0.05, 0.3), 2),
            "fno_uncertainty": round(
                rng.uniform(0.3, 0.5) if kind == "vortex" else rng.uniform(0.05, 0.2), 2
            ),
        },
        "mission": {"next_wp": [6.0, 0.0, 1.5], "geofence_margin_m": 2.7},
        "last_intent": {"mode": "TRACK", "pid_gain_scale": 1.0},
    }


def extract_json(text: str):
    """Return (obj, had_prefix). Handles reasoning text before the JSON (Ultra puts it in `content`)."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    found = None
    for i, ch in enumerate(text):  # keep the last top-level object that decodes
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict) and "mode" in obj:
                    found = obj
            except json.JSONDecodeError:
                continue
    if found is None:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    return found, True


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def cmd_models(client: OpenAI) -> None:
    ids = sorted(m.id for m in client.models.list().data)
    print(f"{len(ids)} models available:\n")
    for i in ids:
        mark = (
            "  <-- NVIDIA"
            if "nemotron" in i.lower() or i.lower().startswith("nvidia/")
            else ""
        )
        print(f"  {i}{mark}")
    print(
        "\nNote: '-fast' variants may not be listed separately; test them with "
        "'bench --model <ID>-fast'."
    )


def constraint_kwargs(constraint: str, thinking: bool) -> dict:
    """How JSON is enforced: 'none' (prompt only), 'strict', 'loose' (json_schema), 'guided' (vLLM guided_json)."""
    extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    kw: dict = {}
    if constraint == "strict":
        kw["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "aero_intent",
                "strict": True,
                "schema": INTENT_SCHEMA,
            },
        }
    elif constraint == "loose":
        kw["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "aero_intent", "schema": INTENT_SCHEMA},
        }
    elif constraint == "guided":
        extra["guided_json"] = INTENT_SCHEMA
    kw["extra_body"] = extra
    return kw


def cmd_bench(
    client: OpenAI,
    model: str,
    n: int,
    thinking: bool,
    period: float,
    constraint: str = "none",
    max_tokens: int | None = None,
) -> None:
    rng = random.Random(42)
    lat, ok_schema, reasoning_seen, errors = [], 0, 0, []
    tokens_in, tokens_out = [], []

    for k in range(n):
        feats = random_features(rng)
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(feats, separators=(",", ":")),
                    },
                ],
                temperature=0.1,
                top_p=1.0,
                max_tokens=max_tokens or (4096 if thinking else 200),
                seed=42,
                timeout=30.0,
                **constraint_kwargs(constraint, thinking),
            )
            dt = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 - we want to count every failure type
            errors.append(f"#{k}: {type(e).__name__}: {e}"[:300])
            if type(e).__name__ == "NotFoundError" or (k >= 2 and not lat):
                print("Stopping early: first calls all failed.")
                break
            time.sleep(period)
            continue

        lat.append(dt)
        msg = resp.choices[0].message
        content = msg.content or ""
        # Reasoning can show up as a separate field or as <think> tags in the content.
        extra = getattr(msg, "model_extra", None) or {}
        if (
            extra.get("reasoning_content")
            or extra.get("reasoning")
            or "<think>" in content
        ):
            reasoning_seen += 1
        if resp.usage:
            tokens_in.append(resp.usage.prompt_tokens)
            tokens_out.append(resp.usage.completion_tokens)
        try:
            obj, had_prefix = extract_json(content)
            if had_prefix:
                reasoning_seen += 1
            errs = list(VALIDATOR.iter_errors(obj))
            if errs:
                errors.append(f"#{k}: schema: {errs[0].message}"[:200])
            else:
                ok_schema += 1
            status = f"mode={obj.get('mode')}"
        except json.JSONDecodeError:
            errors.append(f"#{k}: invalid JSON: {content[:80]!r}")
            status = "INVALID JSON"
        print(f"[{k + 1:>2}/{n}] {dt * 1000:7.0f} ms  {status}")
        time.sleep(max(0.0, period - dt))  # stay under the rate limit

    print("\n=== Result for", model, "| thinking =", thinking, "===")
    if lat:
        p50, p95 = statistics.median(lat), pct(lat, 95)
        print(
            f"Latency  p50 {p50:.2f} s | p95 {p95:.2f} s | max {max(lat):.2f} s  ({len(lat)} ok calls)"
        )
        print(f"Schema   {ok_schema}/{len(lat)} valid (incl. bounds)")
        print(
            f"Reasoning visible in {reasoning_seen}/{len(lat)} responses "
            f"({'expected' if thinking else 'should be 0 -> enable_thinking=false works'})"
        )
        if tokens_in:
            print(
                f"Tokens   in ~{statistics.mean(tokens_in):.0f} | out ~{statistics.mean(tokens_out):.0f} per call"
            )
        if not thinking:
            verdict = (
                "GO: keep plan (~1 Hz)"
                if p95 < 2.0 and ok_schema == len(lat)
                else "NO-GO: try another model or 0.5 Hz, or fix schema issues"
            )
            print("Verdict ", verdict)
    print(f"Errors   {len(errors)}")
    for e in errors[:10]:
        print("  ", e)


def cmd_diagnose(client: OpenAI, model: str) -> None:
    """Each variant changes one thing vs. the minimal request, so the failing parameter stands out."""
    feats = json.dumps(random_features(random.Random(1)), separators=(",", ":"))
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": feats},
    ]
    schema_strict = {
        "type": "json_schema",
        "json_schema": {"name": "aero_intent", "strict": True, "schema": INTENT_SCHEMA},
    }
    schema_loose = {
        "type": "json_schema",
        "json_schema": {"name": "aero_intent", "schema": INTENT_SCHEMA},
    }
    variants = [
        ("A system prompt + features only", {}),
        (
            "B + temperature/top_p/max_tokens",
            {"temperature": 0.1, "top_p": 1.0, "max_tokens": 150},
        ),
        ("C + seed", {"max_tokens": 150, "seed": 42}),
        (
            "D + enable_thinking=false",
            {
                "max_tokens": 150,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        ),
        (
            "E + json_schema strict",
            {"max_tokens": 300, "response_format": schema_strict},
        ),
        (
            "F + json_schema non-strict",
            {"max_tokens": 300, "response_format": schema_loose},
        ),
        (
            "G + guided_json (vLLM extra_body)",
            {"max_tokens": 300, "extra_body": {"guided_json": INTENT_SCHEMA}},
        ),
        (
            "H + json_object",
            {"max_tokens": 300, "response_format": {"type": "json_object"}},
        ),
        (
            "I thinking off + temp/seed",
            {
                "max_tokens": 150,
                "temperature": 0.1,
                "top_p": 1.0,
                "seed": 42,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        ),
        (
            "J thinking off + schema strict",
            {
                "max_tokens": 300,
                "response_format": schema_strict,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        ),
        (
            "K thinking off + schema non-strict",
            {
                "max_tokens": 300,
                "response_format": schema_loose,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        ),
        (
            "L thinking off + guided_json",
            {
                "max_tokens": 300,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "guided_json": INTENT_SCHEMA,
                },
            },
        ),
    ]
    for name, kw in variants:
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=model, messages=msgs, timeout=60.0, **kw
            )
            c = (r.choices[0].message.content or "").strip().replace("\n", " ")
            try:
                obj = json.loads(c)
                valid = (
                    "schema OK"
                    if not list(VALIDATOR.iter_errors(obj))
                    else "JSON, schema FAIL"
                )
            except json.JSONDecodeError:
                valid = "not pure JSON"
            print(
                f"OK   {name:<38} {(time.perf_counter() - t0) * 1000:6.0f} ms  {valid}  | {c[:70]}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name:<38} {type(e).__name__}: {str(e)[:90]}")
        time.sleep(1.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models")
    pg = sub.add_parser("ping", help="minimal call without schema/thinking flags")
    pg.add_argument("--model", required=True)
    dg = sub.add_parser(
        "diagnose", help="add request parameters one by one to find what breaks"
    )
    dg.add_argument("--model", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--model", required=True)
    b.add_argument("--n", type=int, default=25)
    b.add_argument("--thinking", action="store_true")
    b.add_argument(
        "--period", type=float, default=1.2, help="seconds between call starts"
    )
    b.add_argument(
        "--constraint",
        choices=["none", "strict", "loose", "guided"],
        default="none",
        help="JSON enforcement method (default: prompt only + validator)",
    )
    b.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="default 200, or 4096 with --thinking",
    )
    args = ap.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key=os.environ["NEBIUS_API_KEY"])
    if args.cmd == "models":
        cmd_models(client)
    elif args.cmd == "diagnose":
        cmd_diagnose(client, args.model)
    elif args.cmd == "ping":
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=args.model,
            max_tokens=20,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
        )
        print(
            f"{(time.perf_counter() - t0) * 1000:.0f} ms ->",
            repr(r.choices[0].message.content),
        )
    else:
        cmd_bench(
            client,
            args.model,
            args.n,
            args.thinking,
            args.period,
            args.constraint,
            args.max_tokens,
        )


if __name__ == "__main__":
    main()
