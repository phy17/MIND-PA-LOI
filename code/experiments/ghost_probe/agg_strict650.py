#!/usr/bin/env python3
"""Strict-650 re-aggregation for ours/aeb_only/mind (Plan B).

The published clean81 ghost table mixes two simulation horizons: 1336 rows are
``main650`` (650-frame runs) and 122 rows are ``h800`` (800-frame reruns of
"late_or_unspawned" tasks). To make the comparison against the new
reachset/shadow baselines fair (those run at a single 650-frame horizon), this
script rebuilds the ghost table from the *original 650-frame run* of every task,
discarding the h800 reruns entirely.

Two phases:
  * ``verify``  -- reverse-engineer the TSV's derived columns from each task's
                   summary.json and confirm they reproduce all main650 rows
                   exactly. This guarantees our re-derivation of the 122 h800
                   tasks uses the same formulas as the published table.
  * ``build``   -- emit a strict-650 ghost table: the 1336 main650 rows verbatim
                   plus the 122 h800 tasks re-derived from their original 650
                   run (which may now be unspawned/late), plus a metrics report
                   with the spawn-rate "conservatism spectrum".

Nothing in the already-run experiment directories is modified; outputs are
written to new files only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PUBLISHED_TSV = os.path.join(
    REPO, "实验记录", "clean81_ghost_task_details_threshold3mps_20260607.tsv"
)
# Original 650-frame run directories (h800 reruns deliberately excluded).
ORIG_650_DIRS = [
    os.path.join(REPO, "实验记录", "20260531_qualified26_d5p5-3p0_3baselines_instant_center", "runs"),
    os.path.join(REPO, "实验记录", "20260606_new90_le3_1800_inference", "runs"),
]
LOW_SEVERITY_THRESHOLD = 3.0  # m/s, matches the published "threshold3mps" table
FPS = 50.0

# Outcome columns re-derived from the original 650 summary.json. Identity
# columns (baseline, distance, dataset, clean81_index, source_scene_index,
# task_id) and the horizon-independent raw_no_ghost_collision are kept verbatim.
OUTCOME_COLS = [
    "ghost_spawned",
    "any_collision",
    "ghost_collision",
    "background_collision",
    "zero_collision_safe",
    "impact_speed",
    "paper_safe",
    "spawn_frame",
]

# A spawned task is "evaluable" at 650 frames if it collided (definitive
# outcome) or had at least this much window left after spawn. The published
# pipeline reran everything below this (<1.5 s window) or unspawned at h800.
MIN_EVAL_WINDOW_S = 1.5
TOTAL_FRAMES = 650


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_summary(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) else data


def to_bool(value):
    return str(value).strip().lower() == "true"


def to_float(value):
    s = str(value).strip()
    if s in ("", "None", "nan", "NaN"):
        return None
    return float(s)


def derive(summary):
    """Compute the published table's derived columns from one summary.json."""
    collisions = summary.get("collision_log") or []
    ghost_speeds = [
        float(item.get("ego_vel", 0.0))
        for item in collisions
        if str(item.get("other_id")) == "GHOST_001"
    ]
    background = [item for item in collisions if str(item.get("other_id")) != "GHOST_001"]
    collision_count = int(summary.get("collision_count", len(collisions)) or 0)

    ghost_spawned = bool(summary.get("ghost_spawned"))
    any_collision = collision_count > 0
    ghost_collision = bool(ghost_speeds)
    # "Background collision" in the published table = a pure background failure:
    # the ego hit a non-ghost agent and did NOT hit the ghost. Scenes that hit
    # both are counted under ghost_collision only (ghost takes priority), so
    # ghost_collision + background_collision partitions any_collision.
    background_collision = bool(background) and not bool(ghost_speeds)
    zero_collision_safe = collision_count == 0
    # Ghost impact speed = max ego speed over ghost contacts (0 when none).
    impact_speed = max(ghost_speeds) if ghost_speeds else 0.0
    # Low-severity: no collision at all, or ghost contact at <= threshold.
    paper_safe = zero_collision_safe or (ghost_collision and impact_speed <= LOW_SEVERITY_THRESHOLD)

    spawn_time = summary.get("ghost_spawn_time_s")
    spawn_frame = ""
    if isinstance(spawn_time, (int, float)):
        spawn_frame = int(round(spawn_time * FPS))

    return {
        "ghost_spawned": ghost_spawned,
        "any_collision": any_collision,
        "ghost_collision": ghost_collision,
        "background_collision": background_collision,
        "zero_collision_safe": zero_collision_safe,
        "impact_speed": impact_speed,
        "paper_safe": paper_safe,
        "spawn_frame": spawn_frame,
    }


def index_orig_650():
    """basename(task_dir) -> summary.json path, across the original 650 dirs."""
    index = {}
    dups = []
    for runs_dir in ORIG_650_DIRS:
        if not os.path.isdir(runs_dir):
            continue
        for name in os.listdir(runs_dir):
            sp = os.path.join(runs_dir, name, "summary.json")
            if os.path.exists(sp):
                if name in index:
                    dups.append(name)
                index[name] = sp
    return index, dups


def task_basename(summary_path):
    return os.path.basename(os.path.dirname(summary_path))


def cmd_verify(_args):
    rows = read_tsv(PUBLISHED_TSV)
    main650 = [r for r in rows if r["source"] == "main650"]
    print(f"published rows: {len(rows)}  main650: {len(main650)}")

    # raw_no_ghost_collision is a no-ghost-replay scene property (not derivable
    # from the ghost run, not used in any paper table/figure); it is copied
    # verbatim in build mode and intentionally not checked here.
    bool_cols = [
        "ghost_spawned",
        "any_collision",
        "ghost_collision",
        "background_collision",
        "zero_collision_safe",
        "paper_safe",
    ]
    mism = Counter()
    examples = defaultdict(list)
    checked = 0
    for r in main650:
        sp = r["summary_path"]
        if not os.path.exists(sp):
            mism["MISSING_SUMMARY"] += 1
            continue
        d = derive(load_summary(sp))
        checked += 1
        for col in bool_cols:
            if d[col] != to_bool(r[col]):
                mism[col] += 1
                if len(examples[col]) < 3:
                    examples[col].append((r["task_id"], d[col], r[col]))
        # impact_speed numeric compare
        tsv_imp = to_float(r["impact_speed"]) or 0.0
        if abs(tsv_imp - d["impact_speed"]) > 1e-3:
            mism["impact_speed"] += 1
            if len(examples["impact_speed"]) < 5:
                examples["impact_speed"].append((r["task_id"], d["impact_speed"], r["impact_speed"]))
        # spawn_frame compare (string)
        if str(r["spawn_frame"]).strip() != str(d["spawn_frame"]).strip():
            mism["spawn_frame"] += 1
            if len(examples["spawn_frame"]) < 5:
                examples["spawn_frame"].append(
                    (r["task_id"], d["spawn_frame"], r["spawn_frame"])
                )

    print(f"checked: {checked}")
    if not mism:
        print("PERFECT: all derived columns reproduce the published main650 rows.")
    else:
        print("MISMATCHES (col: count):")
        for k, v in sorted(mism.items()):
            print(f"  {k}: {v}")
            for ex in examples[k]:
                print(f"      task={ex[0]} derived={ex[1]!r} published={ex[2]!r}")


def fmt_bool(value):
    return "True" if value else "False"


def eval_window_s(spawn_frame):
    if spawn_frame in ("", None):
        return None
    return (TOTAL_FRAMES - int(spawn_frame)) / FPS


def is_evaluable(derived):
    """Spawned and either collided or left >= MIN_EVAL_WINDOW_S after spawn."""
    if not derived["ghost_spawned"]:
        return False
    if derived["any_collision"]:
        return True
    win = eval_window_s(derived["spawn_frame"])
    return win is not None and win >= MIN_EVAL_WINDOW_S - 1e-9


def row_view(row):
    """Read a published/strict row into a normalized dict for aggregation."""
    return {
        "ghost_spawned": to_bool(row["ghost_spawned"]),
        "any_collision": to_bool(row["any_collision"]),
        "ghost_collision": to_bool(row["ghost_collision"]),
        "background_collision": to_bool(row["background_collision"]),
        "zero_collision_safe": to_bool(row["zero_collision_safe"]),
        "paper_safe": to_bool(row["paper_safe"]),
        "impact_speed": to_float(row["impact_speed"]) or 0.0,
        "spawn_frame": row["spawn_frame"].strip() if row["spawn_frame"].strip() else "",
    }


def aggregate(rows, baseline, predicate):
    sub = [row_view(r) for r in rows if r["baseline"] == baseline]
    kept = [v for v in sub if predicate(v)]
    n = len(kept)
    impacts = [v["impact_speed"] for v in kept]
    hits = [v["impact_speed"] for v in kept if v["ghost_collision"]]
    return {
        "n": n,
        "paper_safe": sum(v["paper_safe"] for v in kept),
        "zero_collision": sum(v["zero_collision_safe"] for v in kept),
        "ghost_collision": sum(v["ghost_collision"] for v in kept),
        "background_collision": sum(v["background_collision"] for v in kept),
        "paper_safe_pct": (100.0 * sum(v["paper_safe"] for v in kept) / n) if n else 0.0,
        "zero_collision_pct": (100.0 * sum(v["zero_collision_safe"] for v in kept) / n) if n else 0.0,
        "mean_impact": (sum(impacts) / n) if n else 0.0,
        "mean_impact_when_hit": (sum(hits) / len(hits)) if hits else 0.0,
        "mean_impact_v2": (sum(v * v for v in impacts) / n) if n else 0.0,
    }


def cmd_build(args):
    rows = read_tsv(PUBLISHED_TSV)
    fieldnames = list(rows[0].keys())
    index, dups = index_orig_650()
    if dups:
        raise SystemExit(f"duplicate orig-650 basenames: {dups[:5]}")

    out_rows = []
    rederived = 0
    breakdown = defaultdict(lambda: Counter())  # baseline -> {unspawned, late, evaluable}
    for r in rows:
        if r["source"] == "main650":
            nr = dict(r)
        else:
            base = task_basename(r["summary_path"])
            sp = index.get(base)
            if not sp:
                raise SystemExit(f"no original-650 run for h800 task {base}")
            d = derive(load_summary(sp))
            nr = dict(r)  # keep identity + raw_no_ghost_collision verbatim
            nr["ghost_spawned"] = fmt_bool(d["ghost_spawned"])
            nr["any_collision"] = fmt_bool(d["any_collision"])
            nr["ghost_collision"] = fmt_bool(d["ghost_collision"])
            nr["background_collision"] = fmt_bool(d["background_collision"])
            nr["zero_collision_safe"] = fmt_bool(d["zero_collision_safe"])
            nr["paper_safe"] = fmt_bool(d["paper_safe"])
            nr["impact_speed"] = repr(d["impact_speed"]) if d["ghost_collision"] else ""
            nr["spawn_frame"] = str(d["spawn_frame"]) if d["spawn_frame"] != "" else ""
            nr["source"] = "orig650"
            nr["summary_path"] = sp
            rederived += 1
            if not d["ghost_spawned"]:
                breakdown[r["baseline"]]["unspawned"] += 1
            elif is_evaluable(d):
                breakdown[r["baseline"]]["evaluable_late_ok"] += 1
            else:
                breakdown[r["baseline"]]["late_unevaluable"] += 1
        out_rows.append(nr)

    # Write strict-650 TSV (new file, originals untouched).
    out_tsv = os.path.join(REPO, "实验记录", f"clean81_ghost_strict650_{args.date}.tsv")
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    # Aggregate per baseline under several denominators.
    baselines = ["ours", "aeb_only", "mind"]
    metrics = {"thresholds": {"low_severity_mps": LOW_SEVERITY_THRESHOLD,
                              "min_eval_window_s": MIN_EVAL_WINDOW_S,
                              "total_frames": TOTAL_FRAMES},
               "rederived_tasks": rederived,
               "rederived_breakdown": {b: dict(breakdown[b]) for b in breakdown},
               "per_baseline": {}}
    for b in baselines:
        n_total = sum(1 for r in out_rows if r["baseline"] == b)
        spawned = aggregate(out_rows, b, lambda v: v["ghost_spawned"])
        evaluable = aggregate(out_rows, b, lambda v: is_evaluable(v))
        allrows = aggregate(out_rows, b, lambda v: True)
        metrics["per_baseline"][b] = {
            "n_total": n_total,
            "n_spawned": spawned["n"],
            "spawn_rate_pct": 100.0 * spawned["n"] / n_total if n_total else 0.0,
            "n_evaluable": evaluable["n"],
            "evaluable_rate_pct": 100.0 * evaluable["n"] / n_total if n_total else 0.0,
            "safety_among_evaluable": evaluable,
            "safety_among_spawned": spawned,
            "safety_over_all_rows": allrows,
        }

    out_json = os.path.join(REPO, "实验记录", f"clean81_strict650_metrics_{args.date}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Console report.
    print(f"strict-650 TSV : {out_tsv}  ({len(out_rows)} rows)")
    print(f"metrics JSON   : {out_json}")
    print(f"re-derived h800->650 rows: {rederived}")
    print("re-derived breakdown:")
    for b in breakdown:
        print(f"  {b}: {dict(breakdown[b])}")
    print()
    print("=== spawn-rate / evaluable-rate spectrum (conservatism) ===")
    print(f"{'baseline':9s} {'n':>4s} {'spawn%':>7s} {'eval%':>7s}")
    for b in baselines:
        m = metrics["per_baseline"][b]
        print(f"{b:9s} {m['n_total']:4d} {m['spawn_rate_pct']:7.1f} {m['evaluable_rate_pct']:7.1f}")
    print()
    print("=== strict-650 safety AMONG EVALUABLE (fair denominator) ===")
    hdr = f"{'baseline':9s} {'nEval':>5s} {'lowSev%':>8s} {'zeroColl%':>9s} {'gColl':>5s} {'bgColl':>6s} {'meanV':>6s} {'hitV':>6s} {'meanV2':>7s}"
    print(hdr)
    for b in baselines:
        s = metrics["per_baseline"][b]["safety_among_evaluable"]
        print(f"{b:9s} {s['n']:5d} {s['paper_safe_pct']:8.1f} {s['zero_collision_pct']:9.1f} "
              f"{s['ghost_collision']:5d} {s['background_collision']:6d} {s['mean_impact']:6.2f} "
              f"{s['mean_impact_when_hit']:6.2f} {s['mean_impact_v2']:7.2f}")
    print()
    print("=== for reference: safety OVER ALL 486 rows (naive, inflated by crawl) ===")
    for b in baselines:
        s = metrics["per_baseline"][b]["safety_over_all_rows"]
        print(f"{b:9s} n={s['n']:3d} lowSev%={s['paper_safe_pct']:.1f} meanV2={s['mean_impact_v2']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="confirm derivation reproduces published main650 rows")
    pb = sub.add_parser("build", help="emit strict-650 table + metrics")
    pb.add_argument("--date", default="20260613")
    args = parser.parse_args()
    if args.cmd == "verify":
        cmd_verify(args)
    elif args.cmd == "build":
        cmd_build(args)


if __name__ == "__main__":
    main()
