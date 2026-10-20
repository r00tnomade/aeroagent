# AGENTS.md — AeroAgent

Guidance for AI coding agents (Claude Code, Codex, Cursor, etc.) working in this repository.
Read this file fully before making changes. When it conflicts with a code comment, this file wins;
when it conflicts with an explicit instruction from the maintainer, the maintainer wins.

## 1. What this project is

**AeroAgent** reconstructs the 2D flow field around a drone from sparse pressure/IMU sensors with a
Fourier Neural Operator (FNO), feeds the predicted wind force forward into a fast flight controller,
and lets **NVIDIA Nemotron 3** (via **Nebius Token Factory**) supervise *how* to fly the gust.

Built solo for the **Nebius x NVIDIA Global AI Hackathon — Physical AI track**.
Hard deadline: **30 Oct 2026, 10:00 PT (18:00 CET)**. Target submission: **29 Oct**.

One-line pitch: *Sparse sensors → neural flow field in milliseconds → Nemotron-supervised drone that rides the gust.*

## 2. Core architecture (do not violate)

The system is **hierarchical**. An LLM call (~1 s TTFT on Nebius, ~10 s with reasoning) can never sit
in the fast control path. Every change must preserve this split.

| Tier | Rate | Component | Responsibility |
|---|---|---|---|
| Tier 0 – inner loop | 240 Hz | `DSLPIDControl` + mixer (gym-pybullet-drones) | Attitude/rate control → RPM. No ML, no LLM. |
| Tier 1 – reflex | 50 Hz | Sensors → FNO (Nebius Serverless Endpoint) → features → feedforward | `a_cmd = a_ref + Kp·e + Kd·ė − F̂_wind/m`. Owns the **<50 ms** claim. |
| Tier 2 – supervisor | 0.5–5 Hz, async | Nemotron via Token Factory | Returns a bounded *intent* (mode, gain scale, waypoint offset, tilt limit, rationale). |
| Watchdog | every tick | Deterministic validator | Clamp, rate-limit, timeout → HOLD/SAFE. |

Invariants:
- **The LLM never outputs motor commands or the thrust allocation.** The X-quad allocation matrix is
  deterministic linear algebra in `control/mixer.py`. Nemotron outputs an intent only.
- **Tier 1 must never await Tier 2.** Exactly one Nemotron request in flight; results are offered to
  an `IntentSlot` and consumed on the next tick.
- **Every LLM output passes the validator** (`agent/validator.py`) before it touches the controller.
- If the FNO endpoint is unreachable, Tier 1 falls back to a **local copy of the same weights**.
- If Nemotron fails 3× in a row → SAFE latch (HOLD, gain 1.0, tilt ≤ 15°, FNO feedforward stays on).
- The simulator applies the **ground-truth LBM wind field** to the drone; the FNO only ever sees sensors.
  Never feed ground truth into the controller path (it would invalidate the demo).

## 3. Repository layout

```
aeroagent/
├── jobs/            # Nebius Serverless Jobs: generate_lbm.py (XLB/lettuce), train_fno.py
├── model/           # fno.py (PhysicsNeMo FNO, fallback: neuraloperator), rasterize.py, features.py
├── serve/           # FastAPI FNO endpoint: serve.py, Dockerfile
├── sim/             # wind_aviary.py (gym-pybullet-drones subclass), planar_quad.py (2D numpy fallback)
├── control/         # reflex.py (Tier 1), mixer.py, pid params
├── agent/           # supervisor.py (async Nemotron loop), prompts/, schema.json, validator.py, rule_baseline.py
├── viz/             # rerun_app.py, matplotlib_fallback.py, colormaps
├── eval/            # ablation.py (PID / +FNO / +LLM), latency_bench.py
├── assets/          # demo.rrd, hero.gif (small files only; large data goes to object storage)
├── tests/
├── demo.py          # --replay <file.rrd> | --live --scenario <name>
├── LICENSE NOTICE CITATION.cff README.md AGENTS.md CLAUDE.md
```
Create directories as needed, but keep this structure. Don't add top-level folders without a reason.

## 4. Environment & commands

- **Python 3.10** (pybullet has no prebuilt wheels beyond 3.10). Don't bump it.
- Install: `pip install -e ".[dev]"`
- Key deps: `nvidia-physicsnemo` (fallback `neuraloperator`), `torch`, `xlb` or `lettuce`,
  `gym-pybullet-drones`, `rerun-sdk`, `openai` (Token Factory is OpenAI-compatible), `fastapi`, `httpx`, `cmocean`.
- Tests: `pytest -q` · Lint/format: `ruff check . && ruff format .`
- Offline demo (no GPU, no API key): `python demo.py --replay assets/demo.rrd`
- Live demo: `export NEBIUS_API_KEY=... FNO_URL=... FNO_TOKEN=...` then
  `python demo.py --live --scenario gust_front_seed42`
- Latency benchmark: `python eval/latency_bench.py --n 1000` (writes p50/p95/p99 CSV)
- Ablation: `python eval/ablation.py --scenario gust_front_seed42` (PID only / +FNO FF / +Nemotron)

Secrets come from environment variables only (`NEBIUS_API_KEY`, `FNO_URL`, `FNO_TOKEN`).
Never hard-code, log, or commit them. `.env` is git-ignored.

## 5. Nebius & NVIDIA integration

**Token Factory (Nemotron)**
- Base URL: `https://api.tokenfactory.nebius.com/v1/` via the `openai` SDK.
- Periodic supervisor: Nemotron 3 Super (fallback: Nemotron 3.5 Lightning, then rule-based baseline).
  Event-driven planning (pre-flight mission, gust-front deliberation): Nemotron 3 Ultra with reasoning on, never blocking.
- Model IDs live in `agent/config.py` only. Confirmed via `GET /v1/models` (20 Oct 2026) — casing is
  inconsistent, copy exactly:
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, `nvidia/nemotron-3-super-120b-a12b`,
  `nvidia/Nemotron-3-Ultra-550b-a55b`, `nvidia/Nemotron-3_5-Lightning`, `nvidia/Nemotron-3.5-Super-VL`.
  `-fast` variants do **not** exist for Nemotron on this account (404) — use the base IDs.
- **Measured 20 Oct 2026** (`tools/tf_check.py bench --constraint strict`, 25 calls, home network, ~1035 in / ~125 out tokens):

  | Model | p50 | p95 | Schema valid | Role |
  |---|---|---|---|---|
  | `nvidia/nemotron-3-super-120b-a12b` | 1.13 s | 1.63 s | 25/25 | **Primary supervisor, ~1 Hz** |
  | `nvidia/Nemotron-3_5-Lightning` | 1.19 s | 1.54 s | 25/25 | Fallback supervisor |
  | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | 1.32 s | 3.07 s | 25/25 | Not used (p95 too high) |
  | `nvidia/Nemotron-3-Ultra-550b-a55b` (reasoning on, 3 calls) | 5.72 s | 5.93 s | 3/3 | Pre-flight mission planning only |
- With reasoning on, Ultra writes its reasoning as plain text into `content` (no separate field, no `<think>`)
  and the strict schema is not enforced until the end: use `max_tokens ≥ 4096` (~1,600 output tokens seen)
  and extract the last JSON object from the text (`extract_json` in `tools/tf_check.py`).
- **Reasoning is ON by default.** Without `enable_thinking: False` the reasoning eats `max_tokens` and
  `content` comes back empty. Always send it for the supervisor.
- Do **not** use the SDK's `with_raw_response` against Token Factory: it returned HTTP 500 for every call
  while identical normal calls worked.
- Model regions (third-party catalogue, 19 Oct 2026, verify): Nemotron 3 Nano → eu-north1 (same region as the
  FNO endpoint); Nemotron 3 Super and Ultra → us-central1. The extra transatlantic RTT (~100 ms, estimate) is
  irrelevant for the async ~1 Hz supervisor, but never put a Super/Ultra call on any timing-critical path.
- **Rate limits:** dynamic, 15-min rolling windows; the documented example base is 60 requests/min
  (check the real value in the console). A 1 Hz supervisor already uses ~60 RPM → default to **≤ 1 Hz**,
  count requests, and treat HTTP 429 exactly like a timeout (validator fallback, exponential backoff).
  Never retry in a tight loop.
- Request settings for the supervisor: `temperature=0.1`, `max_tokens=150`, `seed=42`,
  `response_format={"type":"json_schema","json_schema":{"name":"aero_intent","strict":true,"schema":...}}`,
  `extra_body={"chat_template_kwargs":{"enable_thinking": False}}`, client timeout ≈ 1.5 × period.
- **Never** use `response_format={"type":"json_object"}` (accepts any JSON, even `{}`).
- `/no_think` is a Nemotron Nano 9B v2 convention — do not use it for Nemotron 3.
- Keep the system prompt in `agent/prompts/system.txt` **byte-identical** across calls (prefix caching);
  the variable feature JSON goes last, serialized with `separators=(",", ":")`.
- Nemotron sees a compact feature summary (~20–40 numbers), **never raw grids**.

**Serverless Endpoint (FNO)** — region `eu-north1`, L40S, container port 8000, bearer auth.
- `POST /infer` takes the sensor JSON, returns features (`wind_at_drone`, `gust_eta_s`, `vorticity_peak`,
  `fno_uncertainty`, `fno_ms`, …); the full field only when `return_field=true` (10 Hz for viz, not per control tick).
- `GET /health` must stay cheap. Warm up the GPU on startup.
- Use a persistent `httpx.Client` (keep-alive). For the <50 ms claim, sim + Tier 1 run **co-located in
  eu-north1**; the laptop only displays the Rerun stream.
- Endpoints bill per second: remind the maintainer to stop them after sessions.

**Serverless Jobs** — data generation (`jobs/generate_lbm.py`) and training (`jobs/train_fno.py`).
Jobs must be reproducible from a seed and a config file.

**Pass/fail rule of the hackathon:** the project must run on Nebius *and* use at least one NVIDIA open
model. The FNO is not an NVIDIA model — Nemotron must remain a working, demonstrated component.

## 6. Data & model rules

- Training data is **self-generated** 2D LBM (XLB or lettuce): channel 256×128, cylinder/NACA obstacle,
  Re ≈ 100–500, time-varying gust inlet. Stored as `.npz`/HDF5 with (u, v, p, ω, mask).
- Synthetic sensors: K = 16–32 pressure taps, 1–5 % noise (relative to dynamic pressure), random dropout.
- FNO input: 3 channels (rasterized sensor values, mask, obstacle SDF); output: (u, v, p, ω).
  Optional divergence penalty (∇·u ≈ 0) as the physics-informed regularizer.
- Do not use PINNs in the inference path (per-sample optimization is far too slow).
- **Never commit** third-party datasets with unclear or restrictive licenses (MeshGraphNets cylinder_flow,
  JHTDB, original FNO-paper data): provide download scripts instead.
- **Do not add non-commercial (CC BY-NC) assets** — Poseidon weights, FlowBench, etc.
- Large files (datasets, checkpoints > ~20 MB) go to object storage or releases, not git.

## 7. Supervisor validation rules (`agent/validator.py`)

Treat these as tested requirements; changing them needs a test update and a reason.
- Schema violation / parse error → discard, keep last valid intent.
- Re-clamp every field to schema bounds (grammar engines don't reliably enforce `minimum`/`maximum`):
  |roll|,|pitch| ≤ 25°, |yaw_rate| ≤ 90°/s, thrust_norm 0.2–0.85, pid_gain_scale 0.7–1.5,
  waypoint_offset ≤ 2 m per axis, max_tilt 5–25°.
- Rate limits: Δpid_gain_scale ≤ 0.2 per cycle; mode hysteresis ≥ 2 cycles (except switching to HOLD/LAND).
- Blending: `att_cmd = att_pid + w·att_llm`, w ≤ 0.3; `thrust_norm` is a bias, never a setpoint.
- Waypoint offsets violating geofence/obstacles → set to 0.
- `confidence < 0.4` → 50/50 blend with last intent. Responses older than 1.5 s → discard.
- Timeout → exponential decay to HOLD default over ~2 s. 3 consecutive failures → SAFE latch.
- Log every override with its reason to Rerun (override rate is a demo metric).
- `agent/rule_baseline.py` is both the fallback and the fair A/B baseline against Nemotron.

## 8. Performance budgets

| Path | Budget | How it's checked |
|---|---|---|
| FNO forward (server-side `fno_ms`) | p95 < 10 ms | `eval/latency_bench.py` |
| Tier 1 tick, co-located | p95 < 50 ms (target ~4–10 ms) | latency bench + Rerun HUD |
| Supervisor (Super-fast, reasoning off) | p95 < 2 s | supervisor stats log |
| Visualization | must never block control | separate thread/process |

Report measured numbers only. Any number in README/video that is not measured must be labeled as an estimate.
State the hardware next to every benchmark.

## 9. Coding conventions

- Type hints everywhere; dataclasses or Pydantic for messages (sensor JSON, features, intent).
- SI units in code (m, m/s, rad, s); degrees only at the LLM boundary, converted in the validator.
  World frame ENU, z up. Name variables with units when ambiguous (`yaw_rate_dps`, `gust_eta_s`).
- Determinism: every scenario takes a seed; demo recordings must be reproducible.
- Control-path code: no blocking I/O, no allocations in hot loops where avoidable, no `print` (use logging/Rerun).
- Keep functions small and testable; the controller, mixer, validator and feature extraction must have unit tests.
- Comments explain *why*, not *what*. No dead code or commented-out experiments in `main`.

## 10. Testing expectations

Before declaring a task done:
1. `pytest -q` passes; `ruff check .` is clean.
2. For control/agent changes: run the seeded scenario for 60 s with no crash and no SAFE latch
   (unless the change is about the latch).
3. For model/serving changes: rerun `eval/latency_bench.py` and compare against the budgets above.
4. `python demo.py --replay assets/demo.rrd` still works without GPU or API key.

Tests must not call real Nebius APIs. Mock the OpenAI client and the FNO endpoint; mark optional live
tests with `@pytest.mark.live` (skipped by default).

## 11. Licensing

- Repo license: **Apache-2.0** (must be visible in the GitHub "About" section — hackathon requirement).
- Every new dependency: check its license; MIT/BSD/Apache are fine. Add it to `NOTICE`.
  GPL/LGPL/NC dependencies require explicit maintainer approval.
- Model licenses to cite in README/NOTICE: Nemotron 3 Super/Nano → NVIDIA Nemotron Open Model License;
  Nemotron 3 Ultra → OpenMDW-1.1. Weights are never redistributed (API use only).
- Self-generated dataset: CC-BY-4.0. AirfRANS-derived data (if used) stays ODbL.

## 12. Hackathon deliverables (keep in sync)

- README: "Sponsor tech used" table near the top, architecture (Mermaid), quickstart ≤ 5 commands,
  ablation results table, model card, limitations, licenses.
- Video < 3 min, public on YouTube, ≥ 1 min of key modules running. No third-party trademarks or
  unlicensed music.
- Judging criteria (equal weight): Technological Implementation, Design, Potential Impact, Quality of the Idea.
- When a change affects features shown in README or the video script, update README in the same change.

## 13. Scope discipline

The deadline is fixed. Prefer finishing and hardening the core loop over new features.
Out of scope until all mandatory deliverables are done: 3D FNO (D3Q19), forecast channels (t+0.5 s, t+1 s),
Isaac Sim/Isaac Lab port, Tavily integration, real hardware. Note these under "Future work" instead.

When unsure about a design decision that affects the architecture invariants (section 2), the public
claims (latency, accuracy), or licensing — stop and ask the maintainer instead of guessing.