#!/usr/bin/env python3
"""Statistical supplements for the T-ITS manuscript, computed from the frozen
result files only (no new simulation).

Outputs, printed as a report:
  1. Scene-clustered bootstrap 95% CIs for the Table `main-safety` low-severity
     and zero-collision rates (all-task accounting, mixed-horizon TSV).
  2. Strict-650 evaluable-set low-severity CI for PA-LOI (cross-reference).
  3. Matched-cell (all-five-spawn) low-severity rates with scene-clustered
     bootstrap CIs, plus exact two-sided sign tests for the pairwise
     ours-vs-reachset and ours-vs-shadow contact-speed comparisons.
  4. Sensitivity of the low-severity ranking to the reporting threshold
     (2.0 / 2.5 / 3.0 / 3.5 m/s), all-task accounting.
  5. Scene-clustered paired 95% CIs for the principal risk differences and
     reductions in the velocity-squared impact proxy.

Clustering unit is the scene (8-hex AV2 hash); outcomes within a scene are
correlated across trigger distances, so the bootstrap resamples scenes, not
tasks. Fixed seed for reproducibility.
"""
from __future__ import annotations

import math
import os
import random
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "ghost_probe"))

import agg_strict650 as A  # noqa: E402
import agg_five_system_matched as M  # noqa: E402

MAIN_TSV = os.path.join(REPO, "实验记录", "clean81_ghost_task_details_threshold3mps_20260607.tsv")
SEED = 20260708
N_BOOT = 10000
STACKS = ["ours", "aeb_only", "mind"]
LABELS = {"ours": "PA-LOI + AEB", "aeb_only": "AEB-only", "mind": "MIND"}


def scene_hash(text: str) -> str:
    m = re.search(r"[0-9a-f]{8}", text)
    return m.group(0) if m else text


def cluster_bootstrap_ci(per_scene, n_boot=N_BOOT, seed=SEED):
    """per_scene: dict scene -> (k, n). Returns (rate, lo, hi) in percent."""
    scenes = sorted(per_scene)
    k_tot = sum(per_scene[s][0] for s in scenes)
    n_tot = sum(per_scene[s][1] for s in scenes)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        k = n = 0
        for _ in scenes:
            s = scenes[rng.randrange(len(scenes))]
            k += per_scene[s][0]
            n += per_scene[s][1]
        stats.append(100.0 * k / n if n else float("nan"))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[min(int(0.975 * n_boot), n_boot - 1)]
    return 100.0 * k_tot / n_tot, lo, hi


def cluster_bootstrap_difference_ci(per_scene, scale=1.0, n_boot=N_BOOT, seed=SEED):
    """Paired cluster bootstrap for a mean difference.

    ``per_scene`` maps scene -> (sum of paired cell differences, number of
    paired cells).  Resampling scenes preserves every within-scene pairing
    across trigger distances.  ``scale=100`` reports a binary risk difference
    in percentage points; ``scale=1`` keeps a continuous metric in its units.
    """
    scenes = sorted(per_scene)
    diff_tot = sum(per_scene[s][0] for s in scenes)
    n_tot = sum(per_scene[s][1] for s in scenes)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        diff = 0.0
        n = 0
        for _ in scenes:
            s = scenes[rng.randrange(len(scenes))]
            diff += per_scene[s][0]
            n += per_scene[s][1]
        stats.append(scale * diff / n if n else float("nan"))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[min(int(0.975 * n_boot), n_boot - 1)]
    return scale * diff_tot / n_tot, lo, hi


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    k = min(wins, losses)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def main():
    rows = [A.row_view(r) | {"baseline": r["baseline"], "task_id": r["task_id"]}
            for r in A.read_tsv(MAIN_TSV)]

    print("=== 1. Table main-safety (all-task accounting): scene-clustered 95% CIs ===")
    for st in STACKS:
        sub = [r for r in rows if r["baseline"] == st]
        for metric, key in [("low-severity", "paper_safe"), ("zero-collision", "zero_collision_safe")]:
            per_scene = defaultdict(lambda: [0, 0])
            for r in sub:
                s = scene_hash(r["task_id"])
                per_scene[s][0] += int(bool(r[key]))
                per_scene[s][1] += 1
            rate, lo, hi = cluster_bootstrap_ci({s: tuple(v) for s, v in per_scene.items()})
            print(f"  {LABELS[st]:>13s} {metric:>14s}: {rate:5.1f}%  [{lo:.1f}, {hi:.1f}]")

    print("\n=== 2. Strict-650 evaluable-set low-severity CI ===")
    strict = M.load_strict()
    for st in STACKS:
        per_scene = defaultdict(lambda: [0, 0])
        for (key, sy), rec in strict.items():
            if sy != st or not rec["evaluable"]:
                continue
            per_scene[key[0]][0] += int(bool(rec["paper_safe"]))
            per_scene[key[0]][1] += 1
        rate, lo, hi = cluster_bootstrap_ci({s: tuple(v) for s, v in per_scene.items()})
        n = sum(v[1] for v in per_scene.values())
        print(f"  {LABELS[st]:>13s}: {rate:5.1f}%  [{lo:.1f}, {hi:.1f}]  (n={n} evaluable)")

    print("\n=== 3. Matched cells (all five systems spawn) ===")
    base = M.load_baselines()
    recs = dict(strict)
    recs.update(base)
    keys_by_system = defaultdict(set)
    for (key, sy) in recs:
        keys_by_system[sy].add(key)
    present = [s for s in M.ROW_ORDER if keys_by_system.get(s)]
    common = set.intersection(*[keys_by_system[s] for s in present])
    matched = [k for k in common if all(recs[(k, s)]["spawned"] for s in present)]
    print(f"  matched cells: {len(matched)} (systems: {present})")
    for st in present:
        per_scene = defaultdict(lambda: [0, 0])
        for k in matched:
            per_scene[k[0]][0] += int(bool(recs[(k, st)]["paper_safe"]))
            per_scene[k[0]][1] += 1
        rate, lo, hi = cluster_bootstrap_ci({s: tuple(v) for s, v in per_scene.items()})
        print(f"  {M.LABELS[st]:>14s} matched low-severity: {rate:5.1f}%  [{lo:.1f}, {hi:.1f}]")
    for rival in ["reachset", "shadow"]:
        wins = losses = ties = 0
        for k in matched:
            a, b = recs[(k, "ours")]["impact"], recs[(k, rival)]["impact"]
            if a < b - 1e-9:
                wins += 1
            elif a > b + 1e-9:
                losses += 1
            else:
                ties += 1
        p = sign_test_two_sided(wins, losses)
        print(f"  ours vs {rival:>8s}: lower contact speed in {wins} cells, higher in {losses}, "
              f"ties {ties}; two-sided sign test p = {p:.2e}")

    print("\n=== 4. Low-severity threshold sensitivity (all-task accounting) ===")
    for thr in [2.0, 2.5, 3.0, 3.5]:
        vals = []
        for st in STACKS:
            sub = [r for r in rows if r["baseline"] == st]
            k = sum(1 for r in sub
                    if r["zero_collision_safe"] or (r["ghost_collision"] and r["impact_speed"] <= thr))
            vals.append(f"{LABELS[st]} {100.0 * k / len(sub):.1f}%")
        print(f"  thr {thr:.1f} m/s: " + " | ".join(vals))

    print("\n=== 5. Paired scene-clustered effect-size 95% CIs ===")
    def distance_token(task_id):
        m = re.search(r"_(d(?:5p5|5|4p5|4|3p5|3))_", task_id)
        if not m:
            raise ValueError(f"cannot parse distance token from {task_id}")
        return m.group(1)

    by_key = {(scene_hash(r["task_id"]), distance_token(r["task_id"]), r["baseline"]): r
              for r in rows}
    # The distance token and scene hash identify the paired scene-distance cell.
    for rival in ["aeb_only", "mind"]:
        per_scene_low = defaultdict(lambda: [0.0, 0])
        per_scene_v2 = defaultdict(lambda: [0.0, 0])
        cell_keys = sorted({(s, d) for (s, d, st) in by_key if st == "ours"})
        for s, d in cell_keys:
            a = by_key[(s, d, "ours")]
            b = by_key[(s, d, rival)]
            per_scene_low[s][0] += float(bool(a["paper_safe"])) - float(bool(b["paper_safe"]))
            per_scene_low[s][1] += 1
            # Positive means that PA-LOI reduces the severity proxy.
            per_scene_v2[s][0] += b["impact_speed"] ** 2 - a["impact_speed"] ** 2
            per_scene_v2[s][1] += 1
        d, lo, hi = cluster_bootstrap_difference_ci(
            {s: tuple(v) for s, v in per_scene_low.items()}, scale=100.0
        )
        e, elo, ehi = cluster_bootstrap_difference_ci(
            {s: tuple(v) for s, v in per_scene_v2.items()}
        )
        print(f"  PA-LOI - {LABELS[rival]} low-severity rate difference: "
              f"{d:.1f} pp [{lo:.1f}, {hi:.1f}]")
        print(f"  {LABELS[rival]} - PA-LOI mean v_imp^2 reduction: "
              f"{e:.2f} [{elo:.2f}, {ehi:.2f}] m^2/s^2")

    for rival in ["reachset", "shadow"]:
        per_scene = defaultdict(lambda: [0.0, 0])
        for k in matched:
            per_scene[k[0]][0] += (float(bool(recs[(k, "ours")]["paper_safe"]))
                                   - float(bool(recs[(k, rival)]["paper_safe"])))
            per_scene[k[0]][1] += 1
        d, lo, hi = cluster_bootstrap_difference_ci(
            {s: tuple(v) for s, v in per_scene.items()}, scale=100.0
        )
        print(f"  PA-LOI - {M.LABELS[rival]} matched low-severity rate difference: "
              f"{d:.1f} pp [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    main()
