#!/usr/bin/env python3
"""
benchmark.py - Measure the auditor against datasets with known defects.

Produces the numbers needed to make defensible claims about the system:

  1. Skip rate        - what fraction of diagnostic tools the agent declined to run
  2. Critical retention - whether skipping cost anything, measured against a
                        forced-all-tools baseline (the control)
  3. Detection recall - did the expected detector actually fire on each defect
  4. False positives  - critical findings raised on datasets known to be clean
  5. Runtime          - wall clock per dataset

Every dataset is audited twice: once letting the agent choose its own strategy,
and once forcing all five tools. Without that baseline, "skipped 40% of checks
with no loss of findings" is an assertion rather than a measurement.

Usage:
    python benchmark.py
    python benchmark.py --keep-learning     # don't restore the .pkl state
    python benchmark.py --output custom.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from auditor import AutonomousDatasetAuditor

BASE_DIR = Path(__file__).resolve().parent
ALL_TOOLS = [
    "leakage_detector",
    "contamination_detector",
    "bias_detector",
    "spurious_detector",
    "feature_utility",
]

# Learned state that a benchmark run would otherwise mutate.
LEARNED_STATE = [
    BASE_DIR / "agent" / "strategy_memory.pkl",
    BASE_DIR / "agent" / "meta_learning.pkl",
]


@dataclass
class Benchmark:
    """A dataset with documented defects and the detectors that should catch them."""

    name: str
    path: str
    target: str
    # Tools expected to produce at least one finding. Empty means a clean control.
    expected_tools: List[str] = field(default_factory=list)
    defect: str = "none (control)"
    source: str = "synthetic"


# Ground truth comes from generate_test_datasets.py, which injects each defect
# deliberately, so the expected detector is known rather than inferred.
BENCHMARKS: List[Benchmark] = [
    Benchmark(
        name="Injected leakage",
        path="test_data/data_leakage_test.csv",
        target="default",
        expected_tools=["leakage_detector"],
        defect="'will_default' leaks the target",
    ),
    Benchmark(
        name="Class imbalance",
        path="test_data/class_imbalance_test.csv",
        target="target",
        expected_tools=["bias_detector"],
        defect="97% / 3% class split",
    ),
    Benchmark(
        name="Multiple issues",
        path="test_data/multiple_issues_test.csv",
        target="churned",
        expected_tools=["bias_detector", "leakage_detector", "contamination_detector"],
        defect="imbalance + 'churn_probability' leak + duplicate rows",
    ),
    Benchmark(
        name="Complex / large",
        path="test_data/complex_large_test.csv",
        target="fraud",
        expected_tools=["bias_detector", "leakage_detector"],
        defect="8% positive class + derived 'future_flag' leak",
    ),
    Benchmark(
        name="Clean control",
        path="test_data/clean_simple_test.csv",
        target="category",
        expected_tools=[],
        defect="none (control)",
    ),
]

# A detector "fires" when it raises something actionable. Info-level findings are
# descriptive rather than a defect claim, so they don't count as a positive.
ACTIONABLE = {"critical", "warning"}


def fired(findings: Optional[List[Dict[str, Any]]]) -> bool:
    return bool(findings) and any(f.get("severity") in ACTIONABLE for f in findings)


def confusion_matrix(rows: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """
    Build a confusion matrix over (dataset x detector) pairs.

    Ground truth positive = the dataset has a defect this detector is meant to
    catch. Predicted positive = the detector raised a critical or warning finding.

    Caveat worth stating alongside any figure derived from this: a detector that
    fires where we expected nothing is counted as a false positive, but it may
    have found a genuine secondary issue the generator created incidentally
    (an ID column, say). This makes precision a conservative lower bound.
    """
    tp = fp = fn = tn = 0

    for row in rows:
        expected = set(row["eval"]["expected_tools"])
        findings_by_tool = row[mode]["findings_by_tool"]

        for tool in ALL_TOOLS:
            predicted = fired(findings_by_tool.get(tool))
            actual = tool in expected

            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1
            else:
                tn += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "accuracy": (tp + tn) / total if total else 0.0,
    }


def _count_critical(findings_by_tool: Dict[str, List[Dict[str, Any]]]) -> int:
    return sum(
        sum(1 for f in findings if f.get("severity") == "critical")
        for findings in findings_by_tool.values()
    )


def run_agent_mode(path: Path, target: str) -> Dict[str, Any]:
    """Audit with the agent choosing its own strategy."""
    auditor = AutonomousDatasetAuditor(verbose=False)
    auditor.load_dataset(str(path), target)

    start = time.time()
    report = auditor.run_audit()
    runtime = time.time() - start

    strategy = report.get("autonomous_strategy", {})
    executed = strategy.get("tools_executed", [])

    findings_by_tool: Dict[str, List[Dict[str, Any]]] = {}
    for finding in report.get("all_findings", []):
        findings_by_tool.setdefault(finding.get("tool", "unknown"), []).append(finding)

    return {
        "runtime": runtime,
        "verdict": report.get("verdict"),
        "score": report.get("readiness_score"),
        "tools_executed": executed,
        "tools_skipped": strategy.get("tools_skipped", []),
        "critical": report["summary"]["critical_count"],
        "total_findings": report["summary"]["total_findings"],
        "findings_by_tool": findings_by_tool,
        "rows": int(auditor.df.shape[0]),
        "columns": int(auditor.df.shape[1]),
    }


def run_baseline_mode(path: Path, target: str) -> Dict[str, Any]:
    """
    Control condition: force every tool to run, bypassing the strategy engine.

    Calls the detectors directly so the meta-learner records nothing - the
    baseline must not teach the agent anything, or it stops being a control.
    """
    auditor = AutonomousDatasetAuditor(verbose=False)
    auditor.load_dataset(str(path), target)

    findings_by_tool: Dict[str, List[Dict[str, Any]]] = {}
    start = time.time()
    for tool in ALL_TOOLS:
        try:
            findings_by_tool[tool] = auditor._execute_tool(tool)
        except Exception as exc:  # a crashing detector shouldn't void the run
            findings_by_tool[tool] = []
            print(f"      ! {tool} failed: {exc}")
    runtime = time.time() - start

    return {
        "runtime": runtime,
        "findings_by_tool": findings_by_tool,
        "critical": _count_critical(findings_by_tool),
        "total_findings": sum(len(v) for v in findings_by_tool.values()),
    }


def evaluate(bench: Benchmark, agent: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the two conditions and score detection against ground truth."""
    skip_rate = len(agent["tools_skipped"]) / len(ALL_TOOLS)

    # Retention only means something when the baseline found criticals at all.
    if baseline["critical"] > 0:
        retention = min(1.0, agent["critical"] / baseline["critical"])
    else:
        retention = None

    # A defect counts as detected when its expected detector produced a finding
    # in the baseline (is it detectable at all) and in agent mode (did the agent
    # keep it). Both are recorded so a miss can be attributed correctly.
    detected_baseline, detected_agent = [], []
    for tool in bench.expected_tools:
        if baseline["findings_by_tool"].get(tool):
            detected_baseline.append(tool)
        if agent["findings_by_tool"].get(tool):
            detected_agent.append(tool)

    return {
        "skip_rate": skip_rate,
        "retention": retention,
        "expected_tools": bench.expected_tools,
        "detected_baseline": detected_baseline,
        "detected_agent": detected_agent,
        "is_control": not bench.expected_tools,
        "false_positives": agent["critical"] if not bench.expected_tools else 0,
    }


def format_markdown(rows: List[Dict[str, Any]], totals: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark Results",
        "",
        "Each dataset audited twice: agent-selected strategy vs. a forced-all-tools",
        "baseline. Ground truth is the defect deliberately injected by",
        "`generate_test_datasets.py`.",
        "",
        "| Dataset | Rows | Known defect | Expected detector fired | Verdict | Score | Tools run | Skipped | Runtime |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in rows:
        if r["eval"]["is_control"]:
            fired = (
                "n/a (control, clean)"
                if r["agent"]["critical"] == 0
                else f"{r['agent']['critical']} false positive(s)"
            )
        else:
            hit = len(r["eval"]["detected_agent"])
            want = len(r["eval"]["expected_tools"])
            fired = f"{'PASS' if hit == want else 'PARTIAL'} {hit}/{want}"

        lines.append(
            "| {name} | {rows:,} | {defect} | {fired} | {verdict} | {score} | {run}/5 | {skipped} | {rt:.1f}s |".format(
                name=r["name"],
                rows=r["agent"]["rows"],
                defect=r["defect"],
                fired=fired,
                verdict=r["agent"]["verdict"],
                score=r["agent"]["score"],
                run=len(r["agent"]["tools_executed"]),
                skipped=len(r["agent"]["tools_skipped"]),
                rt=r["agent"]["runtime"],
            )
        )

    retention_txt = (
        f"{totals['critical_retention'] * 100:.1f}%"
        if totals["critical_retention"] is not None
        else "n/a (baseline found no criticals)"
    )

    agent_cm = totals["confusion_agent"]
    base_cm = totals["confusion_baseline"]

    lines += [
        "",
        "## Classification metrics",
        "",
        "Unit of classification is one (dataset x detector) pair, "
        f"{agent_cm['tp'] + agent_cm['fp'] + agent_cm['fn'] + agent_cm['tn']} in total. "
        "Ground truth positive means the dataset carries a defect that detector "
        "is meant to catch; predicted positive means it raised a critical or "
        "warning finding.",
        "",
        "| Metric | Agent mode | All-tools baseline |",
        "|---|---|---|",
        f"| Precision | {agent_cm['precision']:.3f} | {base_cm['precision']:.3f} |",
        f"| Recall | {agent_cm['recall']:.3f} | {base_cm['recall']:.3f} |",
        f"| F1 | {agent_cm['f1']:.3f} | {base_cm['f1']:.3f} |",
        f"| False positive rate | {agent_cm['fpr']:.3f} | {base_cm['fpr']:.3f} |",
        f"| False negative rate | {agent_cm['fnr']:.3f} | {base_cm['fnr']:.3f} |",
        f"| Accuracy | {agent_cm['accuracy']:.3f} | {base_cm['accuracy']:.3f} |",
        f"| TP / FP / FN / TN | {agent_cm['tp']} / {agent_cm['fp']} / {agent_cm['fn']} / {agent_cm['tn']} "
        f"| {base_cm['tp']} / {base_cm['fp']} / {base_cm['fn']} / {base_cm['tn']} |",
        "",
        "The baseline column isolates detector quality (every tool always runs).",
        "The agent column is end-to-end performance, so any gap between them is",
        "the cost of the agent's decision to skip tools.",
        "",
        "## Aggregate",
        "",
        "```",
        f"Detection recall            {totals['detected']}/{totals['expected']} expected detectors fired"
        f"   ({totals['recall'] * 100:.1f}%)" if totals["expected"] else "Detection recall            n/a",
        f"False positives (controls)  {totals['false_positives']} critical findings across {totals['controls']} clean dataset(s)",
        f"Tool-selection skip rate    {totals['skip_rate'] * 100:.1f}% of checks skipped",
        f"Critical retention          {retention_txt} vs forced-all-tools baseline",
        f"Median runtime              {totals['median_runtime']:.1f}s"
        f"  (range {totals['min_runtime']:.1f}s - {totals['max_runtime']:.1f}s,"
        f" up to {totals['max_rows']:,} rows)",
        "```",
        "",
        "## How these were measured",
        "",
        "- **Skip rate** = `tools_skipped / 5`, averaged across datasets.",
        "- **Critical retention** = criticals found by the agent divided by criticals",
        "  found when all five tools are forced to run. 100% means the agent's",
        "  skipping cost nothing.",
        "- **Detection recall** counts an expected detector as successful when it",
        "  produced at least one finding in agent mode.",
        "- **False positives** are critical findings on datasets generated with no",
        "  injected defects.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the dataset auditor")
    parser.add_argument("--output", default="reports/benchmark.md", help="markdown output path")
    parser.add_argument(
        "--keep-learning",
        action="store_true",
        help="keep meta-learning state changes made during the run",
    )
    args = parser.parse_args()

    # Snapshot learned state so repeated benchmark runs stay comparable.
    backups: Dict[Path, Optional[Path]] = {}
    if not args.keep_learning:
        for pkl in LEARNED_STATE:
            if pkl.exists():
                backup = pkl.with_suffix(pkl.suffix + ".bench-backup")
                shutil.copy2(pkl, backup)
                backups[pkl] = backup
            else:
                backups[pkl] = None

    rows: List[Dict[str, Any]] = []

    try:
        for bench in BENCHMARKS:
            path = BASE_DIR / bench.path
            if not path.exists():
                print(f"SKIP {bench.name}: {path} not found "
                      f"(run: python generate_test_datasets.py)")
                continue

            print(f"\n=== {bench.name} ({bench.path})")

            print("   baseline (all 5 tools)...")
            baseline = run_baseline_mode(path, bench.target)

            print("   agent mode...")
            agent = run_agent_mode(path, bench.target)

            result = evaluate(bench, agent, baseline)
            rows.append({
                "name": bench.name,
                "defect": bench.defect,
                "agent": agent,
                "baseline": baseline,
                "eval": result,
            })

            print(f"   -> {agent['verdict']} ({agent['score']}/100), "
                  f"ran {len(agent['tools_executed'])}/5, "
                  f"{agent['critical']} critical (baseline {baseline['critical']}), "
                  f"{agent['runtime']:.1f}s")
    finally:
        if not args.keep_learning:
            for pkl, backup in backups.items():
                if backup is not None:
                    shutil.move(str(backup), str(pkl))
                elif pkl.exists():
                    pkl.unlink()

    if not rows:
        print("\nNo datasets ran. Generate them first:  python generate_test_datasets.py")
        return

    runtimes = sorted(r["agent"]["runtime"] for r in rows)
    mid = len(runtimes) // 2
    median = runtimes[mid] if len(runtimes) % 2 else (runtimes[mid - 1] + runtimes[mid]) / 2

    expected = sum(len(r["eval"]["expected_tools"]) for r in rows)
    detected = sum(len(r["eval"]["detected_agent"]) for r in rows)

    agent_criticals = sum(r["agent"]["critical"] for r in rows if not r["eval"]["is_control"])
    base_criticals = sum(r["baseline"]["critical"] for r in rows if not r["eval"]["is_control"])

    totals = {
        "expected": expected,
        "detected": detected,
        "recall": detected / expected if expected else 0.0,
        "controls": sum(1 for r in rows if r["eval"]["is_control"]),
        "false_positives": sum(r["eval"]["false_positives"] for r in rows),
        "skip_rate": sum(r["eval"]["skip_rate"] for r in rows) / len(rows),
        "critical_retention": (agent_criticals / base_criticals) if base_criticals else None,
        "median_runtime": median,
        "min_runtime": runtimes[0],
        "max_runtime": runtimes[-1],
        "max_rows": max(r["agent"]["rows"] for r in rows),
        "confusion_agent": confusion_matrix(rows, "agent"),
        "confusion_baseline": confusion_matrix(rows, "baseline"),
    }

    markdown = format_markdown(rows, totals)

    out_path = BASE_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({"rows": rows, "totals": totals}, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print(markdown)
    print("=" * 70)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
