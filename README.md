---
title: Autonomous Dataset Auditor
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Autonomous Dataset Auditor

An agentic system that audits a tabular dataset **before** you train a model on it, and tells you whether the data is trustworthy.

You give it a CSV and a target column. It profiles the data, decides on its own which risk checks are worth running, runs them, critiques its own results, and produces a verdict (`READY` / `NEEDS ATTENTION` / `NOT READY`), a readiness score out of 100, a list of findings, and actionable recommendations.

It is a **data-integrity auditor**, not a code or security auditor. It looks for the failure modes that make a model look great in validation and fall apart in production: target leakage, train/test contamination, class imbalance, spurious shortcut features, and dead weight features.

---

## Table of contents

- [What it detects](#what-it-detects)
- [Architecture](#architecture)
- [The agent loop, step by step](#the-agent-loop-step-by-step)
- [Repository layout](#repository-layout)
- [Running it](#running-it)
- [HTTP API](#http-api)
- [Report format](#report-format)
- [Scoring and verdict](#scoring-and-verdict)
- [Persistence and learning](#persistence-and-learning)
- [Deployment](#deployment)
- [Tests and fixtures](#tests-and-fixtures)
- [Known rough edges](#known-rough-edges)

---

## What it detects

Five detectors live in [backend/tools/](backend/tools/). Each exposes the same contract — construct with a DataFrame plus target column, call `.detect()`, get back a list of finding dicts shaped `{type, severity, feature, message, evidence}` where severity is `critical` / `warning` / `info`.

| Tool key | Module | Looks for |
|---|---|---|
| `leakage_detector` | [leakage_detector.py](backend/tools/leakage_detector.py) | Near-perfect feature↔target correlation, suspicious column names, impossibly high single-feature predictive power, duplicate/derived columns |
| `contamination_detector` | [contamination_detector.py](backend/tools/contamination_detector.py) | Exact duplicate rows, hash-identical rows, near-duplicates via cosine similarity, duplicate IDs — within one file, or across a train/test pair |
| `bias_detector` | [bias_detector.py](backend/tools/bias_detector.py) | Class imbalance, per-class feature distribution shift, missingness correlated with the target, extreme skew, insufficient sample size |
| `spurious_detector` | [spurious_correlation_detector.py](backend/tools/spurious_correlation_detector.py) | Single-feature dominance, accuracy sensitivity to feature ablation, suspiciously simple decision rules, unrealistic feature importances (uses a RandomForest + cross-validation) |
| `feature_utility` | [feature_utility_detector.py](backend/tools/feature_utility_detector.py) | Constant and near-constant columns, low variance, redundant correlated pairs, low mutual information, runaway cardinality, excessive missingness |

Two detectors sample adaptively so they stay usable on big files: `BiasDetector` caps at 50 columns and 10 000 rows, and `SpuriousCorrelationDetector` uses a hybrid strategy (stratified / quantile-preserving / cluster-based) that leaves datasets under 10 000 rows completely untouched.

## Architecture

```
                       ┌──────────────────────────────────────────┐
   CSV upload ───────► │  api_server.py (Flask)                   │
                       │  job registry + background thread        │
                       └───────────────┬──────────────────────────┘
                                       │
                       ┌───────────────▼──────────────────────────┐
                       │  AutonomousDatasetAuditor (auditor.py)   │
                       │  owns the run loop                       │
                       └───┬───────────────────────────────────┬──┘
                           │                                   │
        ┌──────────────────▼──────────────┐        ┌───────────▼────────────┐
        │  agent/  — the reasoning layer  │        │  tools/  — the checks  │
        │                                 │        │                        │
        │  strategy_engine   which tools  │        │  leakage_detector      │
        │  goal_engine       when to stop │        │  contamination_det.    │
        │  contingency_pl.   if/then      │        │  bias_detector         │
        │  critic            trust level  │        │  spurious_detector     │
        │  memory1           run state    │        │  feature_utility       │
        │  meta_learning     across runs  │        │                        │
        │  planner           static order │        │                        │
        └─────────────────────────────────┘        └────────────────────────┘
```

The split that matters: **`tools/` know about data, `agent/` knows about the audit.** No detector is aware another one exists; all coordination is in the agent layer.

### The agent modules

- **[strategy_engine.py](backend/agent/strategy_engine.py)** — Profiles the dataset into a `DatasetProfile` (rows, features, class count, balance ratio, missing ratio, numeric/categorical mix, whether temporal or ID-looking columns exist) and rolls those into a `complexity_score` in `[0,1]`. Then scores every tool for relevance to *this* dataset and drops anything below a skip threshold of `0.45`. Example rules: `leakage_detector` gets +0.4 if ID-like columns exist and +0.3 for temporal columns; `bias_detector` gets +0.5 when class balance is under 0.5 and another +0.3 under 0.3. The surviving tools are ordered by score.
- **[goal_engine.py](backend/agent/goal_engine.py)** — Converts the complexity score into a goal and a time budget: complexity > 0.7 → `DEEP_INVESTIGATION` / 300 s, complexity < 0.3 → `QUICK_VALIDATION` / 60 s, otherwise `FIND_CRITICAL_FAST` / 180 s. During the run it answers three questions repeatedly: should we continue, should we deep-dive on what we just found, and should we stop early.
- **[contingency_planner.py](backend/agent/contingency_planner.py)** — Pre-registered if/then reactions keyed to `TriggerCondition`s (leakage found, severe imbalance, contamination, multiple criticals, no issues yet, time running out). A triggered plan can skip tools, boost tools, add extra checks, change the goal, or shift thresholds.
- **[critic.py](backend/agent/critic.py)** — Assigns a confidence score to each tool's output using tool-specific heuristics, and flags `needs_recheck` when results look ambiguous. Confidence bands: 0.9 high, 0.7 medium, 0.5 low. It also feeds `actionable_recommendations` into the final report.
- **[memory1.py](backend/agent/memory1.py)** — Per-run state: the ordered list of executed steps with timings and statuses, findings grouped by tool, and run metadata. This is what the report is assembled from.
- **[meta_learning_engine.py](backend/agent/meta_learning_engine.py)** — Cross-run learning. Records `StrategyPerformance` per audit (time to first critical, efficiency score, whether the goal was achieved) and returns learned recommendations on later runs: per-tool score boosts, an optimal tool sequence, and a tuned skip threshold that overrides the default `0.45`.
- **[planner.py](backend/agent/planner.py)** — The original static priority ordering (leakage → contamination → bias → spurious → feature utility) with simple skip logic. It is constructed by the auditor but the dynamic `strategy_engine` + `goal_engine` path is what actually drives execution now; `planner` is effectively legacy.

## The agent loop, step by step

From [`AutonomousDatasetAuditor.run_audit()`](backend/auditor.py#L108-L292):

1. Reset the contingency planner and start the clock.
2. `goal_engine.initialize_goal(complexity, num_tools)` — pick goal and time budget.
3. `strategy_engine.decide_audit_strategy(profile)` — pick and order tools. If it selects none, fall back to all five.
4. Loop while tools remain:
   1. Ask `goal_engine.should_continue_audit(...)`; break if it says no.
   2. Pop the next tool and run it via `_execute_tool`.
   3. Record status: `fail` if any critical finding, `warning` if any finding, else `pass`. Push into memory with its runtime.
   4. `contingency_planner.evaluate_triggers(...)` — any fired plan rewrites the remaining tool queue.
   5. `goal_engine.should_deep_dive(...)` — if yes, ask the critic; if the critic wants a recheck **and** its confidence is below 0.75, run [`_adaptive_recheck`](backend/auditor.py#L317-L364), which does a second targeted pass (variance stability for spurious findings, encoded-target scanning for leakage findings) and appends anything new.
   6. `goal_engine.adjust_strategy_mid_audit(...)` — possibly reorder what's left.
   7. `goal_engine.evaluate_stopping_early(...)` — break if the goal is met.
   8. Break if overall critic confidence drops below 0.4.
5. Record the outcome into the strategy engine, finalize memory, build the report.
6. Feed the report to the meta-learner (wrapped in try/except — a learning failure never fails the audit).
7. Attach `autonomous_strategy`, `goal_oriented`, and `contingency_summary` sections and return.

The consequence worth internalizing: **two runs on the same CSV can execute different tools.** Tool selection depends on the profile, and the profile-to-selection mapping itself drifts as the meta-learner accumulates history.

## Repository layout

```
Autonomous-Data-Build/
├── Dockerfile               two-stage: node builds frontend, python serves everything
├── render.yaml              Render web service, health check at /api/health
├── run_prod.ps1 / .cmd      one-command local production run (Windows)
│
├── backend/
│   ├── api_server.py        Flask API + static host for the built frontend
│   ├── auditor.py           AutonomousDatasetAuditor — the orchestrator
│   ├── main.py              CLI entry point
│   ├── app.py               legacy Streamlit UI (superseded by the React frontend)
│   ├── styles.css           CSS for the Streamlit UI
│   ├── requirements.txt
│   ├── agent/               reasoning layer (+ .pkl learned state)
│   ├── tools/               the five detectors
│   ├── data/                sample datasets; data/uploads/ receives API uploads
│   ├── test_data/           purpose-built fixtures (leakage, imbalance, …)
│   ├── reports/             default output directory
│   └── test_*.py            per-module scripts
│
└── frontend/                React 18 + Vite + Tailwind + shadcn/ui (separate git repo)
    ├── client/src/pages/    home → audit-run → dashboard
    ├── client/src/hooks/    use-audit.ts (start / poll / fetch report)
    ├── shared/routes.ts     the API contract, Zod-validated on both ends
    ├── server/              Express mock server (dev only, not used in production)
    └── script/build.ts      esbuild + vite build → dist/public
```

### Frontend flow

Three routes in [App.tsx](frontend/client/src/App.tsx), wired with `wouter`:

- `/` → **home** — upload a CSV, name the target column.
- `/audit/:id` → **audit-run** — live progress. [use-audit.ts](frontend/client/src/hooks/use-audit.ts) polls `/status` every 1000 ms via TanStack Query and stops the moment `isComplete` flips true. The terminal-style panel renders each stage's `liveMessages`.
- `/audit/:id/report` → **dashboard** — score dial, findings, recommendations, execution timeline, export buttons.

Every response is parsed through the Zod schemas in [shared/routes.ts](frontend/shared/routes.ts), so a backend/frontend contract drift surfaces as a parse warning rather than a silent `undefined`.

Note that [frontend/server/](frontend/server/) contains an Express implementation of the same three endpoints backed by in-memory storage. It is a Replit-era mock. In production Flask serves both the API and the built assets, and the Express server never runs.

## Running it

### Everything, one command (Windows)

```powershell
.\run_prod.cmd
```

or with options:

```powershell
.\run_prod.ps1 -Port 5000        # build frontend, install deps, serve via waitress
.\run_prod.ps1 -SkipBuild        # reuse the existing frontend/dist
.\run_prod.ps1 -SkipInstall      # skip pip install
.\run_prod.ps1 -NoServe          # set up only, don't start the server
```

Then open <http://localhost:5000>.

### Backend only (dev)

```powershell
cd backend
python -m pip install -r requirements.txt
python api_server.py             # honours $env:PORT, default 5000
```

### Frontend dev server with hot reload

```powershell
cd frontend
npm install
npm run dev:client               # Vite on :5173, proxies /api to :5000
```

Override the proxy target with `VITE_BACKEND_URL` if the backend isn't on port 5000.

### CLI, no web layer

```powershell
cd backend
python main.py --dataset data/test_dataset.csv --target target
python main.py --train train.csv --test test.csv --target label
python main.py --dataset data.csv --target label --output reports/my_audit.json --export-csv reports/findings.csv
python main.py --dataset data.csv --target label --quiet
```

`--dataset` and `--train` are mutually exclusive; `--target` is always required. Passing `--train`/`--test` is the only way to get true cross-split contamination detection — with a single file, `ContaminationDetector` can only look for duplicates within it.

### Docker

```powershell
docker build -t data-audit .
docker run -p 10000:10000 data-audit
```

The image builds the frontend with Node 20, then serves it from `python:3.10-slim` under gunicorn with 2 workers on `$PORT` (default 10000).

Completed jobs are persisted to `backend/reports/jobs/`, so a status poll landing on the worker that didn't run the audit still resolves. In-flight jobs remain worker-local, so progress polling during a run can flicker under `-w 2`; use a single worker if you want smooth live progress.

### Requirements

Python 3.10+, Node 20+ (only for building the frontend). Backend deps: flask, flask-cors, numpy, pandas, plotly, scikit-learn, scipy, streamlit. `plotly` and `streamlit` are only needed by the legacy [app.py](backend/app.py) UI — the Flask path doesn't import them.

## HTTP API

All defined in [api_server.py](backend/api_server.py). CORS is open on `/api/*`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/audit/start` | Begin an audit. Returns `{id, message}` immediately. |
| `GET` | `/api/audit/<id>/status` | Poll progress: stage list, percentage, live messages. |
| `GET` | `/api/audit/<id>/report` | The finished report. 404 until complete, 500 if the audit failed. |
| `GET` | `/api/audit/<id>/export/json` | Download the full raw report as an attachment. |
| `GET` | `/api/audit/<id>/export/cleaned-csv` | Download the source CSV after basic cleanup. |
| `GET` | `/api/health` | `{"status": "ok"}` — Render's health check. |
| `GET` | `/*` | Serves `frontend/dist/public`, falling back to `index.html` for client-side routes. |

**Starting an audit** takes either a multipart upload (`file` + `targetColumn`) or a JSON body (`{filename, targetColumn}`) naming a file already on disk. JSON filenames are resolved against `backend/`, `backend/data/`, and `backend/test_data/`. Uploads must end in `.csv`, are passed through `secure_filename`, and are stored as `data/uploads/<uuid>_<name>.csv`.

**Target column resolution** is forgiving: if the requested column isn't present, it falls back to a column literally named `target`, and failing that to the **last column** in the file. When this happens the substitution is reported in the planner stage's live messages rather than raised as an error.

**Progress** is reported over seven stages — planner, the five tools, report generator. Because `run_audit()` is synchronous, the server monkey-patches `auditor._execute_tool` to mark stages running/completed as they happen. Percentage is `(completed / total) * 100`, plus half a stage's worth of credit while one is running, clamped to 99 % until the job actually finishes. Tools the strategy engine declined to run are marked `skipped` with a reason once the run ends. Datasets over 100 000 rows add an `optimizationNotice` to the payload.

**Cleaned-CSV export** ([`_clean_dataframe_for_export`](backend/api_server.py#L223-L247)) is deliberately conservative: strip column names, drop duplicate column names, drop all-empty `Unnamed:` columns, trim string cells, drop all-NaN rows, drop duplicate rows. It does **not** act on the audit's findings — it will not remove a leaking feature for you.

### Job lifecycle

`POST /start` creates an `AuditJob`, stores it in a module-level dict under a lock, and spawns a daemon thread. When the run finishes — successfully or not — the job is written to `backend/reports/jobs/<id>.json` via a temp-file-then-rename, so it is never observed half-written. Reads go through `_get_job()`, which checks memory first and falls back to disk, so reports survive restarts and are visible to sibling gunicorn workers. Unknown IDs still 404. Uploaded CSVs persist under `data/uploads/` regardless.

## Report format

`run_audit()` returns the raw report; [`_convert_report`](backend/api_server.py#L250-L341) reshapes it into camelCase for the UI. Both are available — `/report` gives the converted form, `/export/json` gives the raw.

Raw report keys:

```
audit_metadata        start/end time, dataset shape, target column
verdict               READY | NEEDS ATTENTION | NOT READY
readiness_score       0–100 (floor of 15)
summary               tools_executed, total_findings, critical/warning/info counts
critical_blockers     findings with severity == critical
warnings              findings with severity == warning
critic_assessment     overall_confidence, reliability, actionable_recommendations
recommendations       deduplicated action items
execution_timeline    per-tool status and wall-clock time
all_findings          everything, flat
autonomous_strategy   tools selected / executed / skipped, per-tool reasoning,
                      the dataset profile, learning stats, meta-learning summary
goal_oriented         goal, budget, time used, decision log, strategy changes
contingency_summary   which plans fired and what they did
```

The UI's converted form adds `datasetShape`, `metrics` (totals plus critic confidence), a `timeline` with formatted `MM:SS.cc` start/end stamps, and stable `f1…fN` / `r1…rN` finding and recommendation IDs. Everything passes through `_json_safe`, which unwraps NumPy scalars via `.item()` and stringifies whatever is left — that's what keeps `np.float64` and `np.bool_` from blowing up `jsonify`.

## Scoring and verdict

`readiness_score` starts at 100 and takes escalating penalties ([`_calculate_readiness_score`](backend/auditor.py#L418-L446)):

| Condition | Penalty |
|---|---|
| 1 critical | −12 |
| 2 criticals | −22 |
| 3+ criticals | −22, then −7 each beyond the second |
| ≤3 warnings | −3 each |
| 4–6 warnings | −9, then −2 each beyond the third |
| 7+ warnings | −15, then −1 each beyond the sixth |
| critic confidence < 0.7 | −5 |

The score is clamped to `[15, 100]` — a catastrophic dataset still reports 15, not 0.

Verdict, evaluated in order:

- **NOT READY** — 3 or more criticals, or score below 30.
- **NEEDS ATTENTION** — any critical, or score below 60, or more than 5 warnings.
- **READY** — score ≥ 80 and at most 3 warnings.
- Anything else falls through to **NEEDS ATTENTION**.

## Persistence and learning

Two pickle files under [backend/agent/](backend/agent/) carry state between runs:

- `strategy_memory.pkl` — audit history and per-tool effectiveness for `AutonomousStrategyEngine`.
- `meta_learning.pkl` — `StrategyPerformance` records and discovered cross-dataset patterns for `MetaLearningEngine`.

Both paths are **relative** (`agent/strategy_memory.pkl`), so they resolve against the current working directory. Run from `backend/` and you get the committed history; run from the repo root and the engines silently start from an empty slate. This is the single most common source of "why did it pick different tools this time".

Delete both files to reset the system to its cold-start heuristics. They are pickles, so treat them as trusted-input only — don't load someone else's.

## Deployment

[render.yaml](render.yaml) defines a Docker web service named `autonomous-data-audit` on the starter plan with autodeploy on and `/api/health` as the health check. The Dockerfile handles the rest. `render.yaml` doesn't pin a `dockerfilePath`, relying on the root `Dockerfile`.

## Tests and fixtures

The `backend/test_*.py` files are standalone scripts, not a pytest suite — run them directly:

```powershell
cd backend
python test_full_audit.py            # end-to-end audit
python test_autonomous_system.py     # the agent layer as a whole
python test_leakage.py               # one detector at a time
python test_planner.py
python test_critic.py
python test_memory.py
python test_agentic_enhancements.py
```

[generate_test_datasets.py](backend/generate_test_datasets.py) synthesizes fixtures with known defects into [backend/test_data/](backend/test_data/) — `data_leakage_test.csv`, `class_imbalance_test.csv`, `multiple_issues_test.csv`, `clean_simple_test.csv`, `complex_large_test.csv`, plus `learning_test_v1/v2.csv` for exercising the meta-learner across consecutive runs. These are the right inputs for checking that a detector still fires after you change it.

## Benchmark results

Measured with [backend/benchmark.py](backend/benchmark.py), which audits every dataset twice — once letting the agent choose its own strategy, once forcing all five tools as a control. Ground truth is the defect deliberately injected by [generate_test_datasets.py](backend/generate_test_datasets.py), so "did the right detector fire" is known rather than inferred.

| Dataset | Rows | Injected defect | Expected detector fired | Verdict | Score | Tools run |
|---|---|---|---|---|---|---|
| Injected leakage | 1,000 | `will_default` leaks the target | 1/1 | NOT READY | 71 | 1/5 |
| Class imbalance | 2,000 | 97% / 3% class split | 1/1 | READY | 97 | 2/5 |
| Multiple issues | 1,600 | imbalance + leak + duplicates | 2/3 | NOT READY | 65 | 2/5 |
| Complex / large | 12,000 | 8% positive class + derived leak | 1/2 | READY | 94 | 2/5 |
| Clean control | 500 | none | no false positives | READY | 100 | 3/5 |

Classification metrics over 25 (dataset × detector) pairs. Ground-truth positive means the dataset carries a defect that detector is meant to catch; predicted positive means it raised a critical or warning finding.

| Metric | Agent mode | All-tools baseline |
|---|---|---|
| Precision | **1.000** | 0.538 |
| Recall | 0.714 | 1.000 |
| F1 | **0.833** | 0.700 |
| False positive rate | **0.000** | 0.333 |
| False negative rate | 0.286 | 0.000 |
| Accuracy | **0.920** | 0.760 |
| TP / FP / FN / TN | 5 / 0 / 2 / 18 | 7 / 6 / 0 / 12 |

The headline trade-off: the agent runs **60% fewer checks** and converts a noisy exhaustive scan (precision 0.54, FPR 0.33) into a precise one (**precision 1.00, FPR 0.00**) — at the cost of recall, which drops to 0.714.

Stated plainly, because the number is easy to misread: **critical retention against the baseline is 9.7%.** The agent's skipping is not free. Much of that gap is baseline noise rather than genuine misses — the baseline's own precision is 0.538 — but two expected detectors genuinely failed to fire, and that is a real limitation, not a rounding artifact.

Reproduce with:

```powershell
cd backend
python generate_test_datasets.py   # only needed once
python benchmark.py
```

The harness snapshots and restores `agent/*.pkl` so repeated runs stay comparable — without that, the meta-learner would train on its own benchmark and the numbers would drift.

## Known rough edges

Worth knowing before you build on this:

- **In-memory job cache is unbounded.** Finished jobs are now also written to `reports/jobs/<id>.json` and rehydrated on lookup, so restarts and multi-worker deployments are handled — but the in-process dict is still never evicted, and the JSON files accumulate on disk. Prune `reports/jobs/` periodically.
- **In-flight audits still don't survive a restart.** Only completed (or failed) jobs are persisted. Kill the server mid-audit and that run is gone.
- **`backend/temp_dataset.csv` is ~12 MB** and committed alongside `temp_audit_dataset.csv`. Both are leftovers.
- **Two UIs exist.** [app.py](backend/app.py) (Streamlit, 875 lines) and the React frontend implement overlapping functionality. Only the React one is served in production; Streamlit still pulls `plotly` and `streamlit` into `requirements.txt`.
- **`frontend/` is its own git repository** with its own remotes, while the repo root is not under version control. Nothing tracks the two together.
- **`agent/planner.py` is dead weight in the main path** — instantiated, but the strategy and goal engines make the real decisions.
- **CORS is `*` on all `/api/*` routes**, and uploads are unauthenticated. Fine locally; tighten both before exposing this publicly.
- **Silent target fallback.** A typo'd target column doesn't error — it audits against the last column instead. Check `targetColumn` in the report matches what you intended.
- **Severity scoring is too lenient for prevalence-based defects.** A 97%/3% class split scores READY/97, and a dataset that was 99.96% duplicate rows scored READY/91. Both were detected — they were just filed as warnings worth −3 points each. Any defect defined by *how much* of the data it affects should escalate to critical past a threshold.
- **Recall costs more than the skip rate suggests.** The learned skip threshold (0.55) is aggressive enough to skip detectors that would have caught injected defects — measured recall 0.714. Lowering the threshold trades precision for recall; the benchmark harness makes that trade measurable.
