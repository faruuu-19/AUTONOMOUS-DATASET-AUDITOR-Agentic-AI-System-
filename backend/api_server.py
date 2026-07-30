"""
Flask API server for integrating the Replit frontend with the Python auditor backend.

Endpoints:
  POST /api/audit/start
  GET  /api/audit/<id>/status
  GET  /api/audit/<id>/report
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from auditor import AutonomousDatasetAuditor, read_csv_resilient


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIST_DIR = ROOT_DIR / "frontend" / "dist" / "public"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
JOBS_DIR = BASE_DIR / "reports" / "jobs"

TOOL_ORDER: List[str] = [
    "leakage_detector",
    "contamination_detector",
    "bias_detector",
    "spurious_detector",
    "feature_utility",
]

TOOL_LABELS: Dict[str, str] = {
    "planner": "Planner",
    "leakage_detector": "Leakage Detector",
    "contamination_detector": "Train-Test Contamination Detector",
    "bias_detector": "Bias Detector",
    "spurious_detector": "Spurious Correlation Detector",
    "feature_utility": "Feature Utility Detector",
    "report_generator": "Report Generator",
}

STAGE_ORDER: List[str] = ["planner", *TOOL_ORDER, "report_generator"]


@dataclass
class StageState:
    id: str
    name: str
    status: str = "pending"
    runtime_ms: Optional[int] = None
    critic_assessment: Optional[Dict[str, Any]] = None
    live_messages: List[str] = field(default_factory=list)
    skip_reason: Optional[str] = None


@dataclass
class AuditJob:
    id: str
    filename: str
    filepath: str
    requested_target: str
    target_column: str
    rows: int = 0
    columns: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_complete: bool = False
    failed: bool = False
    error: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    raw_report: Optional[Dict[str, Any]] = None
    stages: List[StageState] = field(default_factory=list)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    return str(value)


def _format_duration(ms: int) -> str:
    minutes = ms // 60000
    seconds = (ms % 60000) // 1000
    centiseconds = (ms % 1000) // 10
    return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _tool_label(tool_name: str) -> str:
    return TOOL_LABELS.get(tool_name, tool_name.replace("_", " ").title())


def _resolve_dataset_path(filename: str) -> Optional[Path]:
    if not filename:
        return None

    candidate = Path(filename)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    fallback_candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / "test_data" / filename,
    ]
    for path in fallback_candidates:
        if path.exists():
            return path
    return None


def _create_default_stages() -> List[StageState]:
    stages: List[StageState] = []
    for idx, stage_key in enumerate(STAGE_ORDER, start=1):
        stages.append(
            StageState(
                id=f"s{idx}",
                name=_tool_label(stage_key),
            )
        )
    return stages


def _stage(job: AuditJob, key: str) -> Optional[StageState]:
    label = _tool_label(key)
    for item in job.stages:
        if item.name == label:
            return item
    return None


def _build_status_payload(job: AuditJob) -> Dict[str, Any]:
    total = max(len(job.stages), 1)
    done = sum(1 for s in job.stages if s.status in {"completed", "skipped"})
    running = any(s.status == "running" for s in job.stages)

    if job.is_complete:
        progress = 100.0
    else:
        progress = (done / total) * 100.0
        if running:
            progress += 50.0 / total
        progress = min(progress, 99.0)

    current_stage = None
    for s in job.stages:
        if s.status == "running":
            current_stage = s.name
            break

    if not current_stage:
        if job.failed:
            current_stage = "Error"
        elif job.is_complete:
            current_stage = "Complete"
        else:
            pending = next((s for s in job.stages if s.status == "pending"), None)
            current_stage = pending.name if pending else "Pending"

    stages_payload: List[Dict[str, Any]] = []
    for s in job.stages:
        item: Dict[str, Any] = {
            "id": s.id,
            "name": s.name,
            "status": s.status,
        }
        if s.skip_reason:
            item["skipReason"] = s.skip_reason
        if s.runtime_ms is not None:
            item["runtimeMs"] = s.runtime_ms
        if s.critic_assessment:
            item["criticAssessment"] = _json_safe(s.critic_assessment)
        if s.live_messages:
            item["liveMessages"] = s.live_messages[-8:]
        stages_payload.append(item)

    payload: Dict[str, Any] = {
        "id": job.id,
        "isComplete": job.is_complete,
        "progressPercentage": round(progress, 2),
        "currentStage": current_stage,
        "stages": stages_payload,
    }
    if job.rows > 100000:
        payload["optimizationNotice"] = "Large dataset detected. Using adaptive sampling."

    return payload


def _assert_job_ready(job: Optional[AuditJob]) -> Optional[tuple]:
    if not job:
        return jsonify({"message": "Report not found or not yet ready"}), 404
    if job.failed:
        return jsonify({"message": f"Audit failed: {job.error}"}), 500
    if not job.is_complete or not job.report:
        return jsonify({"message": "Report not found or not yet ready"}), 404
    return None


def _clean_dataframe_for_export(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]

    unnamed_to_drop: List[str] = []
    for col in cleaned.columns:
        col_name = str(col).strip().lower()
        if not col_name.startswith("unnamed:"):
            continue
        if cleaned[col].isna().all():
            unnamed_to_drop.append(col)

    if unnamed_to_drop:
        cleaned = cleaned.drop(columns=unnamed_to_drop)

    object_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.drop_duplicates()
    return cleaned


def _convert_report(job: AuditJob, raw_report: Dict[str, Any]) -> Dict[str, Any]:
    summary = raw_report.get("summary", {})
    critic = raw_report.get("critic_assessment", {})
    all_findings = raw_report.get("all_findings", [])
    timeline_raw = raw_report.get("execution_timeline", [])

    findings_payload: List[Dict[str, Any]] = []
    for idx, finding in enumerate(all_findings, start=1):
        severity = str(finding.get("severity", "info")).lower()
        if severity not in {"info", "warning", "critical"}:
            severity = "info"

        message = str(finding.get("message") or finding.get("type") or "Issue detected")
        payload_item: Dict[str, Any] = {
            "id": f"f{idx}",
            "severity": severity,
            "tool": _tool_label(str(finding.get("tool", "unknown"))),
            "message": message,
        }
        if finding.get("evidence") is not None:
            payload_item["evidence"] = _json_safe(finding.get("evidence"))
        findings_payload.append(payload_item)

    recommendations_payload: List[Dict[str, Any]] = []
    seen_recommendations = set()
    for idx, rec in enumerate(raw_report.get("recommendations", []), start=1):
        text = str(rec).strip()
        if not text or text in seen_recommendations:
            continue
        seen_recommendations.add(text)
        recommendations_payload.append(
            {
                "id": f"r{idx}",
                "priority": text.upper().startswith("PRIORITY"),
                "text": text,
            }
        )

    timeline_payload: List[Dict[str, Any]] = []
    elapsed_ms = 0
    for step in timeline_raw:
        duration_ms = int(max(1, round(float(step.get("execution_time", 0.0)) * 1000)))
        status_raw = str(step.get("status", "pass")).lower()
        if status_raw == "fail":
            timeline_status = "error"
        elif status_raw == "warning":
            timeline_status = "warning"
        else:
            timeline_status = "completed"

        start_ms = elapsed_ms
        end_ms = elapsed_ms + duration_ms
        elapsed_ms = end_ms

        timeline_payload.append(
            {
                "tool": _tool_label(str(step.get("tool_name", "unknown"))),
                "startTime": _format_duration(start_ms),
                "endTime": _format_duration(end_ms),
                "durationMs": duration_ms,
                "status": timeline_status,
            }
        )

    return {
        "id": job.id,
        "datasetShape": {
            "rows": int(job.rows),
            "columns": int(job.columns),
        },
        "targetColumn": job.target_column,
        "verdict": raw_report.get("verdict", "NEEDS ATTENTION"),
        "readinessScore": int(raw_report.get("readiness_score", 0)),
        "metrics": {
            "totalFindings": int(summary.get("total_findings", 0)),
            "critical": int(summary.get("critical_count", 0)),
            "warning": int(summary.get("warning_count", 0)),
            "confidence": float(critic.get("overall_confidence", 1.0)),
        },
        "findings": findings_payload,
        "recommendations": recommendations_payload,
        "timeline": timeline_payload,
        "rawData": _json_safe(
            {
                "audit_metadata": raw_report.get("audit_metadata"),
                "summary": summary,
                "critical_blockers": raw_report.get("critical_blockers", []),
                "warnings": raw_report.get("warnings", []),
                "critic_assessment": critic,
            }
        ),
    }


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

_jobs: Dict[str, AuditJob] = {}
_jobs_lock = threading.Lock()


def _persist_job(job: AuditJob) -> None:
    """Write a finished job to disk so it survives restarts and other workers."""
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": job.id,
            "filename": job.filename,
            "filepath": job.filepath,
            "requested_target": job.requested_target,
            "target_column": job.target_column,
            "rows": job.rows,
            "columns": job.columns,
            "created_at": job.created_at,
            "is_complete": job.is_complete,
            "failed": job.failed,
            "error": job.error,
            "report": job.report,
            "raw_report": job.raw_report,
            "stages": [
                {
                    "id": s.id,
                    "name": s.name,
                    "status": s.status,
                    "runtime_ms": s.runtime_ms,
                    "critic_assessment": s.critic_assessment,
                    "live_messages": s.live_messages,
                    "skip_reason": s.skip_reason,
                }
                for s in job.stages
            ],
        }
        tmp_path = JOBS_DIR / f"{job.id}.json.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=str)
        tmp_path.replace(JOBS_DIR / f"{job.id}.json")
    except Exception:
        # Persistence is best-effort; never fail an audit because of it.
        pass


def _load_job(job_id: str) -> Optional[AuditJob]:
    """Rehydrate a persisted job, or None if it was never written."""
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        return None

    try:
        with open(job_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None

    stages = [
        StageState(
            id=item.get("id", ""),
            name=item.get("name", ""),
            status=item.get("status", "pending"),
            runtime_ms=item.get("runtime_ms"),
            critic_assessment=item.get("critic_assessment"),
            live_messages=item.get("live_messages") or [],
            skip_reason=item.get("skip_reason"),
        )
        for item in payload.get("stages", [])
    ]

    return AuditJob(
        id=payload.get("id", job_id),
        filename=payload.get("filename", ""),
        filepath=payload.get("filepath", ""),
        requested_target=payload.get("requested_target", ""),
        target_column=payload.get("target_column", ""),
        rows=payload.get("rows", 0),
        columns=payload.get("columns", 0),
        created_at=payload.get("created_at", ""),
        is_complete=payload.get("is_complete", False),
        failed=payload.get("failed", False),
        error=payload.get("error"),
        report=payload.get("report"),
        raw_report=payload.get("raw_report"),
        stages=stages,
    )


def _get_job(job_id: str) -> Optional[AuditJob]:
    """Look up a job in memory, falling back to the on-disk copy."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is not None:
        return job

    job = _load_job(job_id)
    if job is not None:
        with _jobs_lock:
            _jobs.setdefault(job_id, job)
    return job


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return

    planner_stage = _stage(job, "planner")
    if planner_stage:
        planner_stage.status = "running"
        planner_stage.live_messages = [
            "Validating dataset and selecting execution plan...",
        ]

    planner_start = time.time()

    try:
        auditor = AutonomousDatasetAuditor(verbose=False)

        header = read_csv_resilient(job.filepath, nrows=0)
        columns = list(header.columns)
        if not columns:
            raise ValueError("Dataset has no columns.")

        target_column = job.requested_target.strip() if job.requested_target else ""
        if target_column not in columns:
            target_column = "target" if "target" in columns else columns[-1]

        job.target_column = target_column
        job.columns = len(columns)

        auditor.load_dataset(job.filepath, target_column)
        job.rows, job.columns = auditor.df.shape

        if planner_stage:
            planner_stage.status = "completed"
            planner_stage.runtime_ms = int((time.time() - planner_start) * 1000)
            planner_stage.live_messages.append(
                f"Dataset loaded ({job.rows} rows, {job.columns} columns)."
            )
            if job.requested_target and job.requested_target != target_column:
                planner_stage.live_messages.append(
                    f"Requested target '{job.requested_target}' not found. Using '{target_column}'."
                )

        # Instrument execution so UI can reflect per-stage progress while run_audit() is executing.
        original_execute_tool = auditor._execute_tool

        def instrumented_execute_tool(tool_name: str):
            tool_stage = _stage(job, tool_name)
            if tool_stage:
                tool_stage.status = "running"
                tool_stage.live_messages = [f"Running {_tool_label(tool_name)}..."]

            start = time.time()
            findings = original_execute_tool(tool_name)
            runtime_ms = int((time.time() - start) * 1000)

            if tool_stage:
                tool_stage.status = "completed"
                tool_stage.runtime_ms = runtime_ms
                critical_count = sum(1 for f in findings if f.get("severity") == "critical")
                if findings:
                    tool_stage.live_messages.append(
                        f"Detected {len(findings)} issue(s), including {critical_count} critical."
                    )
                else:
                    tool_stage.live_messages.append("No issues detected.")
            return findings

        auditor._execute_tool = instrumented_execute_tool  # type: ignore[method-assign]

        report_stage = _stage(job, "report_generator")
        report_start = time.time()

        raw_report = auditor.run_audit()

        if report_stage:
            report_stage.status = "completed"
            report_stage.runtime_ms = int((time.time() - report_start) * 1000)
            report_stage.live_messages = ["Report ready."]

        timeline_map = {
            item.get("tool_name"): item
            for item in raw_report.get("execution_timeline", [])
            if item.get("tool_name")
        }
        executed_tools = set(raw_report.get("autonomous_strategy", {}).get("tools_executed", []))

        for tool_key in TOOL_ORDER:
            tool_stage = _stage(job, tool_key)
            if not tool_stage:
                continue

            if tool_key in executed_tools:
                tool_stage.status = "completed"
                if tool_key in timeline_map:
                    runtime_sec = float(timeline_map[tool_key].get("execution_time", 0.0))
                    tool_stage.runtime_ms = int(runtime_sec * 1000)
            elif tool_stage.status == "pending":
                tool_stage.status = "skipped"
                tool_stage.skip_reason = "Skipped by autonomous strategy."

        job.raw_report = _json_safe(raw_report)
        job.report = _convert_report(job, raw_report)
        job.is_complete = True
        _persist_job(job)

    except Exception as exc:
        job.failed = True
        job.is_complete = True
        job.error = str(exc)

        running_stage = next((s for s in job.stages if s.status == "running"), None)
        if running_stage:
            running_stage.status = "error"
            running_stage.live_messages.append(f"Error: {job.error}")

        for s in job.stages:
            if s.status == "pending":
                s.status = "skipped"
                s.skip_reason = "Skipped due to earlier error."

        _persist_job(job)


@app.post("/api/audit/start")
def start_audit():
    requested_target = ""
    filename = ""
    dataset_path: Optional[Path] = None

    if request.files and "file" in request.files:
        uploaded_file = request.files["file"]
        if not uploaded_file or not uploaded_file.filename:
            return jsonify({"message": "CSV file is required.", "field": "file"}), 400

        requested_target = str(request.form.get("targetColumn", "")).strip()
        filename = secure_filename(uploaded_file.filename)
        if not filename.lower().endswith(".csv"):
            return (
                jsonify({"message": "Only CSV files are supported.", "field": "file"}),
                400,
            )

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        unique_name = f"{uuid4().hex}_{filename}"
        dataset_path = UPLOAD_DIR / unique_name
        uploaded_file.save(dataset_path)
    else:
        payload = request.get_json(silent=True) or {}
        filename = str(payload.get("filename", "")).strip()
        requested_target = str(payload.get("targetColumn", "")).strip()
        dataset_path = _resolve_dataset_path(filename)
        if dataset_path is None:
            return (
                jsonify(
                    {
                        "message": "Dataset file not found. Upload a CSV or provide a valid filename.",
                        "field": "filename",
                    }
                ),
                400,
            )

    job_id = str(uuid4())
    job = AuditJob(
        id=job_id,
        filename=filename,
        filepath=str(dataset_path),
        requested_target=requested_target,
        target_column=requested_target or "",
        stages=_create_default_stages(),
    )

    with _jobs_lock:
        _jobs[job_id] = job

    worker = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    worker.start()

    return jsonify({"id": job_id, "message": "Audit started successfully"}), 200


@app.get("/api/audit/<job_id>/status")
def get_audit_status(job_id: str):
    job = _get_job(job_id)

    if not job:
        return jsonify({"message": "Audit not found"}), 404

    return jsonify(_build_status_payload(job)), 200


@app.get("/api/audit/<job_id>/report")
def get_audit_report(job_id: str):
    job = _get_job(job_id)

    not_ready_response = _assert_job_ready(job)
    if not_ready_response:
        return not_ready_response

    return jsonify(job.report), 200


@app.get("/api/audit/<job_id>/export/json")
def export_audit_json(job_id: str):
    job = _get_job(job_id)

    not_ready_response = _assert_job_ready(job)
    if not_ready_response:
        return not_ready_response

    export_payload = job.raw_report or job.report or {}
    json_content = json.dumps(export_payload, indent=2, ensure_ascii=False, default=str)
    json_bytes = io.BytesIO(json_content.encode("utf-8"))

    return send_file(
        json_bytes,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"audit_report_{job.id}.json",
    )


@app.get("/api/audit/<job_id>/export/cleaned-csv")
def export_cleaned_csv(job_id: str):
    job = _get_job(job_id)

    not_ready_response = _assert_job_ready(job)
    if not_ready_response:
        return not_ready_response

    source_df = read_csv_resilient(job.filepath)
    cleaned_df = _clean_dataframe_for_export(source_df)

    csv_buffer = io.StringIO()
    cleaned_df.to_csv(csv_buffer, index=False)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode("utf-8-sig"))

    return send_file(
        csv_bytes,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"cleaned_dataset_{job.id}.csv",
    )


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    if path.startswith("api/"):
        return jsonify({"message": "Not found"}), 404

    if path and (FRONTEND_DIST_DIR / path).exists():
        return send_from_directory(str(FRONTEND_DIST_DIR), path)

    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return send_from_directory(str(FRONTEND_DIST_DIR), "index.html")

    return (
        jsonify(
            {
                "message": (
                    "Frontend build not found. Build it with: "
                    "cd ../frontend && npm install && npm run build"
                )
            }
        ),
        200,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
