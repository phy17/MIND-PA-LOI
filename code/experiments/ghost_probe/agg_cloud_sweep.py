#!/usr/bin/env python3
"""Aggregate the cloud sweeps (E1 parameter ablation, E2 extended dash protocol).

E1: per-variant ghost safety (27 scenes x {5.5, 4.0} m, instant-center) and
    no-ghost efficiency (27 scenes), with the deployed operating point taken
    from the published clean81 TSVs restricted to the identical scene/distance
    cells, so no rerun of the baseline is needed.

E2: low-severity / zero-collision / impact metrics per (distance, speed,
    stack) over the 27-scene set, formatted like the published Table IX.

Usage (from the MIND repo root, local or on the server):
  python3 experiments/ghost_probe/agg_cloud_sweep.py --sweep sweep_results \
      [--published-ghost-tsv 实验记录/clean81_ghost_task_details_threshold3mps_20260607.tsv] \
      [--published-noghost-tsv 实验记录/clean81_noghost_efficiency_details_20260607.tsv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

LOW_SEV = 3.0
E1_DISTANCES = {"5.5", "4.0"}
VARIANT_LABELS = {
    "eta0p4": "eta=0.4", "eta0p8": "eta=0.8",
    "vfloor1p0": "v_floor=1.0", "vfloor3p0": "v_floor=3.0",
    "wmax15": "w_max=15", "wmax35": "w_max=35",
    "deployed": "deployed (0.6/2.0/25)",
}


def load_summary(task_dir: Path) -> dict | None:
    p = task_dir / "summary.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) else data


def ghost_metrics(summ: dict) -> dict:
    """Outcome fields for one ghost run summary (JSONL runner format)."""
    first = summ.get("first_collision") or None
    ghost_hit = bool(first and str(first.get("other_id", "")).startswith("GHOST"))
    bg_hit = bool(first and not str(first.get("other_id", "")).startswith("GHOST"))
    imp = float(first.get("ego_vel", 0.0)) if ghost_hit else 0.0
    spawned = bool(summ.get("ghost_spawned"))
    any_hit = bool(summ.get("collision_count", 0) > 0)
    low_sev = (not any_hit) or (ghost_hit and not bg_hit and imp <= LOW_SEV)
    return {"spawned": spawned, "ghost_hit": ghost_hit, "bg_hit": bg_hit,
            "any_hit": any_hit, "impact": imp, "low_sev": low_sev,
            "zero": not any_hit}


def scan_e1(sweep: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    e1 = sweep / "e1"
    if not e1.exists():
        return out
    for vdir in sorted(e1.iterdir()):
        if not vdir.is_dir():
            continue
        ghost, noghost = [], []
        for tdir in sorted(vdir.iterdir()):
            if not tdir.is_dir():
                continue
            summ = load_summary(tdir)
            if summ is None:
                continue
            if tdir.name.startswith("ghost_"):
                ghost.append(ghost_metrics(summ))
            elif tdir.name.startswith("noghost_"):
                noghost.append(summ)
        out[vdir.name] = {"ghost": ghost, "noghost": noghost}
    return out


def published_deployed(ghost_tsv: Path, noghost_tsv: Path, scene_ids: set[int]) -> dict:
    ghost = []
    with open(ghost_tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["baseline"] != "ours" or r["distance"] not in E1_DISTANCES:
                continue
            if int(r["clean81_index"]) not in scene_ids:
                continue
            imp = float(r["impact_speed"] or 0.0)
            ghost.append({
                "spawned": r["ghost_spawned"] == "True",
                "ghost_hit": r["ghost_collision"] == "True",
                "bg_hit": r["background_collision"] == "True",
                "any_hit": r["any_collision"] == "True",
                "impact": imp if r["ghost_collision"] == "True" else 0.0,
                "low_sev": r["paper_safe"] == "True",
                "zero": r["zero_collision_safe"] == "True",
            })
    noghost = []
    with open(noghost_tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["baseline"] != "ours" or int(r["clean81_index"]) not in scene_ids:
                continue
            noghost.append({
                "avg_speed_control_mps": float(r["avg_speed_control_mps"]),
                "distance_control_m": float(r["distance_control_m"]),
                "planner_failure": r["planner_failure"] == "True",
                "ground_truth_collision": r["ground_truth_collision"] == "True",
                "completed_full_horizon": r["completed_full_horizon"] == "True",
            })
    return {"ghost": ghost, "noghost": noghost}


def e1_row(name: str, data: dict) -> str:
    g, n = data["ghost"], data["noghost"]
    if not g:
        return f"{VARIANT_LABELS.get(name, name):>22s}  (no ghost results yet)"
    n_g = len(g)
    low = sum(x["low_sev"] for x in g)
    zero = sum(x["zero"] for x in g)
    v2 = sum(x["impact"] ** 2 for x in g) / n_g
    spawn = sum(x["spawned"] for x in g)
    speed = (sum(x["avg_speed_control_mps"] or 0.0 for x in n) / len(n)) if n else float("nan")
    fails = sum(bool(x.get("planner_failure")) for x in n)
    replays = sum(bool(x.get("ground_truth_collision")) for x in n)
    return (f"{VARIANT_LABELS.get(name, name):>22s}  ghost n={n_g:3d} spawn={spawn:3d} "
            f"lowSev={100 * low / n_g:5.1f}% zero={100 * zero / n_g:5.1f}% v2={v2:6.2f} | "
            f"no-ghost n={len(n):2d} speed={speed:5.2f} m/s fail={fails} replay={replays}")


def scan_e2(sweep: Path) -> dict[tuple, list]:
    cells = defaultdict(list)
    e2 = sweep / "e2"
    if not e2.exists():
        return cells
    pat = re.compile(r"s(\d+)_d([0-9p]+)_v([0-9p]+)_(\w+)$")
    for tdir in sorted(e2.iterdir()):
        m = pat.match(tdir.name)
        if not m or not tdir.is_dir():
            continue
        summ = load_summary(tdir)
        if summ is None:
            continue
        d = m.group(2).replace("p", ".")
        v = m.group(3).replace("p", ".")
        cells[(d, v, m.group(4))].append(ghost_metrics(summ))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=Path("sweep_results"))
    ap.add_argument("--published-ghost-tsv", type=Path,
                    default=Path("实验记录/clean81_ghost_task_details_threshold3mps_20260607.tsv"))
    ap.add_argument("--published-noghost-tsv", type=Path,
                    default=Path("实验记录/clean81_noghost_efficiency_details_20260607.tsv"))
    args = ap.parse_args()

    rec_dir = args.sweep / "records"
    scene_ids = {int(p.stem.replace("scene", "")) for p in rec_dir.glob("scene*.jsonl")}

    print("=== E1 parameter ablation (27 scenes, ghost@{5.5,4.0} m instant-center + no-ghost) ===")
    variants = scan_e1(args.sweep)
    if args.published_ghost_tsv.exists() and scene_ids:
        dep = published_deployed(args.published_ghost_tsv, args.published_noghost_tsv, scene_ids)
        print(e1_row("deployed", dep) + "   [published clean81 tables, same cells]")
    for name in ["eta0p4", "eta0p8", "vfloor1p0", "vfloor3p0", "wmax15", "wmax35"]:
        if name in variants:
            print(e1_row(name, variants[name]))

    print("\n=== E2 dash protocol at additional trigger distances (27 scenes) ===")
    cells = scan_e2(args.sweep)
    if cells:
        print(f"{'dist':>5s} {'speed':>6s} {'stack':>9s} {'n':>3s} {'lowSev%':>8s} "
              f"{'zero%':>6s} {'meanV2':>7s}")
        for (d, v, b) in sorted(cells):
            g = cells[(d, v, b)]
            n = len(g)
            print(f"{d:>5s} {v:>6s} {b:>9s} {n:3d} "
                  f"{100 * sum(x['low_sev'] for x in g) / n:8.1f} "
                  f"{100 * sum(x['zero'] for x in g) / n:6.1f} "
                  f"{sum(x['impact'] ** 2 for x in g) / n:7.2f}")
    else:
        print("(no E2 results yet)")


if __name__ == "__main__":
    main()
