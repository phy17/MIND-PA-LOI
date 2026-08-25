#!/usr/bin/env python3
"""Reproduce every paired / relative number quoted in Section V-F (parameter
sensitivity) that is not already a cell of tables/param_ablation.tex.

Quoted claims verified here (main.tex, subsec:param-sensitivity; values
corrected 2026-08-05, superseding the ad-hoc 2026-07-20 figures):
  1. eta=0.8: +0.10 m/s no-ghost speed, -5.6 pp paired low-severity;
  2. w_max=15: -11.1 pp on both paired low-severity and paired zero collision,
     the only dial with no cell moving in its favor;
  3. no variant improves on the deployed point by more than a single paired
     cell: five variants at or below its low-severity rate, one (+1 cell)
     above;
  4. v_floor=3.0: +1 paired cell (+2.0 pp) on both low-severity and zero
     collision, 2.3% faster no-ghost speed than deployed;
  5. per-scene directionality: each dial changes outcomes in 7-12 scenes and
     every dial except w_max=15 has 1-2 cells moving against its mean
     direction;
  6. deployed 27-scene no-ghost speed loss vs MIND 30.1% (perturbed range
     28.1-33.6%).

Pairing rule (stated in IV-H): comparisons use only cells where both the
variant and the deployed reference spawn the ghost; the deployed reference is
the published strict-650 table restricted to the identical 27 scenes and
{5.5, 4.0} m triggers. Low-severity / zero-collision semantics follow IV-E
exactly (a background-traffic collision counts as unsafe).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SWEEP = ROOT / "sweep_results"
LOW = 3.0

VARIANTS = ["eta0p4", "eta0p8", "vfloor1p0", "vfloor3p0", "wmax15", "wmax35"]


def scenes27() -> set[int]:
    s = {int(p.stem.replace("scene", "")) for p in (SWEEP / "records").glob("scene*.jsonl")}
    assert len(s) == 27
    return s


def cell_key(scene: int, dist: str) -> tuple[int, str]:
    return scene, {"4": "4.0", "4.0": "4.0", "5p5": "5.5", "5.5": "5.5"}[dist]


def load_summary(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    return d[0] if isinstance(d, list) else d


def ghost_metrics(s: dict) -> dict:
    fc = s.get("first_collision") or None
    ghost = bool(fc and str(fc.get("other_id", "")).startswith("GHOST"))
    bg = bool(fc and not str(fc.get("other_id", "")).startswith("GHOST"))
    any_hit = bool(s.get("collision_count", 0) > 0)
    imp = float(fc.get("ego_vel", 0.0)) if ghost else 0.0
    return {
        "spawned": bool(s.get("ghost_spawned")),
        "low": (not any_hit) or (ghost and not bg and imp <= LOW),
        "zero": not any_hit,
    }


def variant_cells(name: str) -> tuple[dict, dict]:
    ghost, ng_speed = {}, {}
    vdir = SWEEP / "e1" / name
    for td in sorted(p for p in vdir.iterdir() if p.is_dir()):
        s = load_summary(td / "summary.json")
        if td.name.startswith("ghost_"):
            _, sc, d = td.name.split("_")
            ghost[cell_key(int(sc[1:]), d[1:])] = ghost_metrics(s)
        else:
            sc = int(td.name.split("_")[1][1:])
            ng_speed[sc] = float(s["avg_speed_control_mps"])
    assert len(ghost) == 54 and len(ng_speed) == 27, name
    return ghost, ng_speed


def deployed_cells(scenes: set[int]) -> tuple[dict, dict, dict]:
    ghost = {}
    with open(ROOT / "实验记录" / "clean81_ghost_strict650_20260613.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["baseline"] != "ours" or r["distance"] not in ("5.5", "4.0"):
                continue
            sc = int(r["clean81_index"])
            if sc not in scenes:
                continue
            ghost[cell_key(sc, r["distance"])] = {
                "spawned": r["ghost_spawned"] == "True",
                "low": r["paper_safe"] == "True",
                "zero": r["zero_collision_safe"] == "True",
            }
    ours_ng, mind_ng = {}, {}
    with open(ROOT / "实验记录" / "clean81_noghost_efficiency_details_20260607.tsv",
              encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sc = int(r["clean81_index"])
            if sc not in scenes:
                continue
            if r["baseline"] == "ours":
                ours_ng[sc] = float(r["avg_speed_control_mps"])
            elif r["baseline"] == "mind":
                mind_ng[sc] = float(r["avg_speed_control_mps"])
    assert len(ghost) == 54 and len(ours_ng) == 27 and len(mind_ng) == 27
    return ghost, ours_ng, mind_ng


def main() -> None:
    scenes = scenes27()
    dep_ghost, dep_ng, mind_ng = deployed_cells(scenes)
    dep_speed = sum(dep_ng.values()) / len(dep_ng)
    mind_speed = sum(mind_ng.values()) / len(mind_ng)

    print("=== V-F paired parameter statistics (deployed vs each variant) ===")
    print(f"27-scene no-ghost speeds: deployed {dep_speed:.3f} m/s, "
          f"MIND {mind_speed:.3f} m/s -> deployed loss "
          f"{100 * (mind_speed - dep_speed) / mind_speed:.1f}%")

    losses = {}
    any_variant_above = False
    for name in VARIANTS:
        vg, vng = variant_cells(name)
        both = [k for k in vg if vg[k]["spawned"] and dep_ghost[k]["spawned"]]
        n = len(both)
        d_low = 100 * (sum(vg[k]["low"] for k in both)
                       - sum(dep_ghost[k]["low"] for k in both)) / n
        d_zero = 100 * (sum(vg[k]["zero"] for k in both)
                        - sum(dep_ghost[k]["zero"] for k in both)) / n
        if d_low > 1e-9:
            any_variant_above = True

        # directionality: cells the dial improves (+) / degrades (-) on either
        # paired metric, and the number of scenes with any changed outcome.
        plus = sum(1 for k in both if vg[k]["low"] and not dep_ghost[k]["low"])
        minus = sum(1 for k in both if dep_ghost[k]["low"] and not vg[k]["low"])
        changed_scenes = len({
            k[0] for k in both
            if vg[k]["low"] != dep_ghost[k]["low"]
            or vg[k]["zero"] != dep_ghost[k]["zero"]
        })
        against = plus if d_low < 0 else (minus if d_low > 0 else plus + minus)

        v_speed = sum(vng.values()) / len(vng)
        losses[name] = 100 * (mind_speed - v_speed) / mind_speed
        print(f"{name:>10}: paired cells {n:2d}  "
              f"dLowSev {d_low:+5.1f} pp (+{plus}/-{minus} cells)  "
              f"dZero {d_zero:+5.1f} pp  "
              f"changed scenes {changed_scenes:2d}  cells against dial {against}  "
              f"no-ghost {v_speed:.3f} m/s ({100 * (v_speed - dep_speed) / dep_speed:+.1f}% "
              f"vs deployed, loss vs MIND {losses[name]:.1f}%)")

    print(f"no variant above deployed paired low-severity: {not any_variant_above}")
    print(f"perturbed loss-vs-MIND range: "
          f"{min(losses.values()):.1f}% .. {max(losses.values()):.1f}%")


if __name__ == "__main__":
    main()
