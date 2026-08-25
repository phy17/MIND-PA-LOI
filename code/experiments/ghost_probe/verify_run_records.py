#!/usr/bin/env python3
"""
Verify run records consistency for ghost-probe experiments.

This script scans an output root recursively, finds every run_summary.json,
and checks whether ground-truth collisions, planner logger summary, and files
existence are consistent. It writes a machine-readable CSV/JSON report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def verify_one(summary_path: Path) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = summary_path.parent

    gt = int(data.get("collision_count_ground_truth", 0))
    logger_count = data.get("planner_logger_collision_count")
    logger_consistent = data.get("planner_logger_consistent_with_ground_truth")
    planner_log_path = data.get("planner_log_path")

    planner_log_exists = False
    if planner_log_path:
        planner_log_exists = (Path(planner_log_path).exists() or (run_dir / Path(planner_log_path).name).exists())

    collision_report = run_dir / "imgs" / "collision_report.json"
    collision_report_exists = collision_report.exists()

    status = "PASS"
    reasons = []

    if logger_count is None:
        status = "WARN"
        reasons.append("logger_count_missing")
    elif int(logger_count) != gt:
        status = "FAIL"
        reasons.append("logger_count_mismatch")

    if logger_consistent is False:
        status = "FAIL"
        reasons.append("logger_consistency_flag_false")

    if planner_log_path and not planner_log_exists:
        status = "FAIL"
        reasons.append("planner_log_missing")

    # Ground truth collision should have report in normal run() flow
    if gt > 0 and not collision_report_exists:
        status = "WARN" if status == "PASS" else status
        reasons.append("collision_report_missing")

    return {
        "summary_path": str(summary_path),
        "run_dir": str(run_dir),
        "seq_id": data.get("seq_id"),
        "sim_name": data.get("sim_name"),
        "ghost_spawned": data.get("ghost_spawned"),
        "gt_collision_count": gt,
        "planner_logger_collision_count": logger_count,
        "planner_logger_consistent": logger_consistent,
        "planner_log_path": planner_log_path,
        "planner_log_exists": planner_log_exists,
        "collision_report_exists": collision_report_exists,
        "status": status,
        "reasons": ";".join(reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ghost-probe run record consistency")
    parser.add_argument("--output-root", default="output", help="Root directory to scan")
    parser.add_argument("--report-dir", default="output/record_audit", help="Directory for audit reports")
    args = parser.parse_args()

    root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries = sorted(root.rglob("run_summary.json"))
    rows = [verify_one(p) for p in summaries]

    json_path = report_dir / "record_audit.json"
    csv_path = report_dir / "record_audit.csv"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_warn = sum(1 for r in rows if r["status"] == "WARN")
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")

    print(f"Found {len(rows)} run summaries under {root}")
    print(f"PASS={n_pass}, WARN={n_warn}, FAIL={n_fail}")
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")


if __name__ == "__main__":
    main()
