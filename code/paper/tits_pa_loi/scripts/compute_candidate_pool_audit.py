#!/usr/bin/env python3
"""Audit new60 curation against the complete new90 candidate campaign.

The primary clean81 benchmark uses 60 scenes selected from a 90-scene
geometric candidate pool.  This script keeps every candidate with complete
strict-650 ghost-task records (88 scenes) and recomputes the three-stack
results without the clean-scene filter.  The output is Supplementary Table S4
and a machine-readable selection-flow summary.

Safety follows the manuscript definition.  A task is evaluable when the ghost
spawns and either a collision occurs or at least 1.5 s remains after spawning.
Background collisions are unsafe.  Non-spawning is exposed through the spawn
column rather than credited as safety.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

CANDIDATES = REPO / "数据集/ghost_injection_new90_le3_candidate_20260606.jsonl"
SELECTED = REPO / "数据集/ghost_injection_new90_le3_clean60_strict_20260607.jsonl"
TASKS = REPO / "实验记录/20260606_new90_le3_1800_inference/ghost_trigger_status_88_valid_tasks.csv"
OUT_TEX = HERE.parent / "tables/candidate_pool_audit.tex"
OUT_JSON = HERE.parent / "tables/candidate_pool_audit.json"
OUT_MEMBERSHIP = HERE.parent / "tables/candidate_pool_membership.tsv"

ORDER = ["ours", "aeb_only", "mind"]
LABEL = {"ours": "PA-LOI + AEB", "aeb_only": "AEB-only", "mind": "MIND"}
SEED = 20260708
N_BOOT = 10000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def truth(text: str) -> bool:
    return str(text).strip().lower() == "true"


def clustered_ci(per_scene: dict[int, tuple[int, int]]) -> tuple[float, float]:
    scenes = sorted(per_scene)
    rng = random.Random(SEED)
    values = []
    for _ in range(N_BOOT):
        k = n = 0
        for _ in scenes:
            scene = scenes[rng.randrange(len(scenes))]
            k += per_scene[scene][0]
            n += per_scene[scene][1]
        values.append(100.0 * k / n)
    values.sort()
    return values[int(0.025 * N_BOOT)], values[int(0.975 * N_BOOT)]


def main() -> None:
    candidates = read_jsonl(CANDIDATES)
    selected = read_jsonl(SELECTED)
    selected_ids = {r["scenario_id"] for r in selected}
    with TASKS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    complete_ids = {r["seq_id"] for r in rows}
    incomplete = [r for r in candidates if r["scenario_id"] not in complete_ids]
    excluded_complete = complete_ids - selected_ids

    assert len(candidates) == 90
    assert len(selected_ids) == 60
    assert len(complete_ids) == 88
    assert len(incomplete) == 2
    assert len(excluded_complete) == 28
    assert len(rows) == 88 * 6 * 3

    metrics: dict[str, dict] = {}
    for stack in ORDER:
        sub = [r for r in rows if r["baseline"] == stack]
        spawned = [r for r in sub if truth(r["ghost_spawned"])]
        evaluable = [
            r for r in spawned
            if int(r["collision_count"]) > 0 or float(r["remaining_after_spawn_s"]) >= 1.5
        ]
        low = zero = background = 0
        sum_v = sum_v2 = 0.0
        per_scene = defaultdict(lambda: [0, 0])
        for row in evaluable:
            count = int(row["collision_count"])
            kind = row["first_collision_kind"]
            impact = float(row["impact_speed_mps"]) if row["impact_speed_mps"] else 0.0
            no_collision = count == 0
            low_contact = count > 0 and kind == "ghost" and impact <= 3.0
            low += int(no_collision or low_contact)
            per_scene[int(row["scene"])][0] += int(no_collision or low_contact)
            per_scene[int(row["scene"])][1] += 1
            zero += int(no_collision)
            background += int(count > 0 and kind != "ghost")
            sum_v += impact
            sum_v2 += impact * impact
        ci_lo, ci_hi = clustered_ci({scene: tuple(v) for scene, v in per_scene.items()})
        metrics[stack] = {
            "tasks": len(sub),
            "spawned": len(spawned),
            "evaluable": len(evaluable),
            "low": low,
            "low_scene_clustered_ci95": [ci_lo, ci_hi],
            "zero": zero,
            "background_collisions": background,
            "mean_impact_speed": sum_v / len(evaluable),
            "mean_impact_speed_squared": sum_v2 / len(evaluable),
        }

    audit = {
        "selection_flow": {
            "geometric_candidates": len(candidates),
            "complete_strict650_records": len(complete_ids),
            "incomplete_record_scenes": [r.get("index") for r in incomplete],
            "selected_new60": len(selected_ids),
            "screened_out_but_log_complete": len(excluded_complete),
        },
        "accounting": "strict-650 evaluable set; scene-filter-free candidate-pool audit",
        "metrics": metrics,
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    task_by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        task_by_id[row["seq_id"]].append(row)
    membership_lines = [
        "candidate_index\tseq_id\tstatus\tcomplete_screening\tretained_new60\tghost_tasks"
        "\tnot_spawned_tasks\tlate_unresolved_tasks\tbackground_or_other_first_collision_tasks"
    ]
    for candidate in candidates:
        seq = candidate["scenario_id"]
        task_rows = task_by_id.get(seq, [])
        complete = seq in complete_ids
        retained = seq in selected_ids
        status = "retained_new60" if retained else ("screened_out" if complete else "incomplete_screening")
        not_spawned = sum(not truth(r["ghost_spawned"]) for r in task_rows)
        late = sum(r["category"] == "spawned_no_collision_late_horizon_lt1s" for r in task_rows)
        background = sum(
            int(r["collision_count"]) > 0 and r["first_collision_kind"] not in ("ghost", "none")
            for r in task_rows
        )
        membership_lines.append(
            f"{candidate.get('index')}\t{seq}\t{status}\t{str(complete).lower()}\t"
            f"{str(retained).lower()}\t{len(task_rows)}\t{not_spawned}\t{late}\t{background}"
        )
    OUT_MEMBERSHIP.write_text("\n".join(membership_lines) + "\n", encoding="utf-8")

    lines = [
        "% AUTO-GENERATED by compute_candidate_pool_audit.py.",
        "\\begin{table}[H]",
        "\\caption{Unfiltered New90 Candidate-Pool Audit. All 88 Geometric Candidates With Complete Strict-650 Records Are Retained, Including the 28 Scenes Excluded From New60. Safety Is Computed on Evaluable Tasks; Spawn Rate Exposes Non-Encounter Conservatism.}",
        "\\label{tab:candidate-pool-audit}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{7pt}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Method & Tasks & Spawn rate & Eval. tasks & Low-severity & Zero collision & Background collisions & Mean $v_{\\mathrm{imp}}^2$ \\\\",
        "\\midrule",
    ]
    for stack in ORDER:
        m = metrics[stack]
        lines.append(
            f"{LABEL[stack]} & {m['tasks']} & {100*m['spawned']/m['tasks']:.1f}\\% "
            f"& {m['evaluable']} & {m['low']}/{m['evaluable']} ({100*m['low']/m['evaluable']:.1f}\\%) "
            f"& {m['zero']}/{m['evaluable']} ({100*m['zero']/m['evaluable']:.1f}\\%) "
            f"& {m['background_collisions']} & {m['mean_impact_speed_squared']:.2f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(audit, indent=2))
    print(f"Wrote {OUT_TEX}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MEMBERSHIP}")


if __name__ == "__main__":
    main()
