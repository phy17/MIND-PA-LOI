#!/usr/bin/env python3
"""Validate the result files used by the T-ITS PA-LOI manuscript."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

GHOST_TSV = ROOT / "实验记录" / "clean81_ghost_task_details_threshold3mps_20260607.tsv"
NOGHOST_TSV = ROOT / "实验记录" / "clean81_noghost_efficiency_details_20260607.tsv"
SPEED_TSV = (
    ROOT
    / "实验记录"
    / "20260607_clean81_rep30_dash_speed_sensitivity_d5p5_4speeds_3baselines_30workers"
    / "speed_sensitivity_paper_table_excl23_44_70.tsv"
)
CLEAN81_JSONL = ROOT / "数据集" / "ghost_injection_clean81_clean21_plus_new60_20260607.jsonl"
REP30_JSONL = ROOT / "数据集" / "ghost_injection_clean81_rep30_speed_sensitivity_d5p5_20260607.jsonl"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_equal(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main() -> None:
    ghost = read_tsv(GHOST_TSV)
    noghost = read_tsv(NOGHOST_TSV)
    speed = read_tsv(SPEED_TSV)
    clean81 = read_jsonl(CLEAN81_JSONL)
    rep30 = read_jsonl(REP30_JSONL)

    assert_equal("clean81 scenes", len(clean81), 81)
    assert_equal("clean81 ghost tasks", len(ghost), 81 * 6 * 3)
    assert_equal("clean81 no-ghost tasks", len(noghost), 81 * 2)
    assert_equal("speed-sensitivity table rows", len(speed), 4 * 3)
    assert_equal("rep30 speed-sensitivity scenes", len(rep30), 30)
    assert_equal("speed-sensitivity tasks", sum(int(row["n"]) for row in speed), 27 * 4 * 3)

    clean81_ids = [row["scenario_id"] for row in clean81]
    assert_equal("unique clean81 scenario ids", len(set(clean81_ids)), 81)

    missing_dirs = [sid for sid in clean81_ids if not (ROOT / "data" / sid).exists()]
    if missing_dirs:
        raise AssertionError(f"missing clean81 data directories: {missing_dirs[:10]}")

    required_baselines = {"ours", "aeb_only", "mind"}
    required_distances = {"3.0", "3.5", "4.0", "4.5", "5.0", "5.5"}
    required_speeds = {"1.0", "1.5", "2.0", "3.0"}

    if {row["baseline"] for row in ghost} != required_baselines:
        raise AssertionError("unexpected ghost baselines")
    if {f"{float(row['distance']):.1f}" for row in ghost} != required_distances:
        raise AssertionError("unexpected trigger distances")
    if len({row["task_id"] for row in ghost}) != len(ghost):
        raise AssertionError("duplicate ghost task ids")
    for baseline in required_baselines:
        subset = [row for row in ghost if row["baseline"] == baseline]
        assert_equal(f"ghost tasks for {baseline}", len(subset), 81 * 6)
        for distance in required_distances:
            count = sum(f"{float(row['distance']):.1f}" == distance for row in subset)
            assert_equal(f"ghost tasks for {baseline} d={distance}", count, 81)
    if any(row["ghost_spawned"].strip().lower() != "true" for row in ghost):
        raise AssertionError("clean81 ghost table contains unspawned tasks")

    if {row["baseline"] for row in noghost} != {"ours", "mind"}:
        raise AssertionError("unexpected no-ghost baselines")
    if len({row["task_id"] for row in noghost}) != len(noghost):
        raise AssertionError("duplicate no-ghost task ids")
    for baseline in {"ours", "mind"}:
        assert_equal(
            f"no-ghost tasks for {baseline}",
            sum(row["baseline"] == baseline for row in noghost),
            81,
        )

    if {row["baseline"] for row in speed} != required_baselines:
        raise AssertionError("unexpected speed-sensitivity baselines")
    if {f"{float(row['speed']):.1f}" for row in speed} != required_speeds:
        raise AssertionError("unexpected pedestrian speeds")
    if any(int(row["n"]) != 27 for row in speed):
        raise AssertionError("unexpected per-cell speed-sensitivity task count")
    if any(int(row["spawned"]) != int(row["n"]) for row in speed):
        raise AssertionError("speed-sensitivity contains unspawned tasks")

    print("Validation passed")
    print(f"clean81 scenes: {len(clean81)}")
    print(f"ghost tasks: {len(ghost)}")
    print(f"no-ghost tasks: {len(noghost)}")
    print(f"speed sensitivity tasks: {sum(int(row['n']) for row in speed)}")


if __name__ == "__main__":
    main()
