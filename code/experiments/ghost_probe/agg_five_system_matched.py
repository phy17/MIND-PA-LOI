#!/usr/bin/env python3
"""Five-system matched ghost-safety comparison (Plan B, Part 2).

Combines the strict-650 ours/aeb_only/mind table with the reachset/shadow
baseline runs (single 650-frame horizon) and reports the comparison the paper
needs:

  * spawn / evaluable-rate spectrum (conservatism), all five systems;
  * per-system safety AMONG EVALUABLE (each system on its own evaluable set);
  * matched safety on the (scene, distance) cells where ALL present systems
    spawned, for a strictly fair head-to-head;
  * pairwise per-scene impact comparison ours-vs-reachset and ours-vs-shadow,
    the decisive test for whether "ours is actually safer" survives matching.

The reachset/shadow fields are re-derived from each summary.json with the
exact same formulas verified in agg_strict650.py, so all five systems use one
metric definition. Works on partial baseline data (mid-run): only the
distances/cells already finished are used.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_strict650 as A  # noqa: E402  (shared derive/verify formulas)

REPO = A.REPO
STRICT_TSV = os.path.join(REPO, "实验记录", "clean81_ghost_strict650_20260613.tsv")
BASE_RUNS = os.path.join(
    REPO, "实验记录", "20260612_clean81_baselines_reachset_shadow_w24", "runs"
)
DIST_CODE = {"d5p5": 5.5, "d5": 5.0, "d4p5": 4.5, "d4": 4.0, "d3p5": 3.5, "d3": 3.0}
SYSTEMS = ["mind", "aeb_only", "reachset", "shadow", "ours"]
TSV_SYSTEMS = {"ours", "aeb_only", "mind"}


def record_from_view(v):
    return {
        "spawned": v["ghost_spawned"],
        "evaluable": A.is_evaluable(v),
        "ghost_collision": v["ghost_collision"],
        "zero_collision": v["zero_collision_safe"],
        "paper_safe": v["paper_safe"],
        "impact": v["impact_speed"],
    }


def scene_hash(text):
    """The 8-hex scene id, robust to the clean21 (scene01_<hash>_...) and
    new60 (ghost_scene01_<hash>_...) task-id conventions."""
    m = re.search(r"[0-9a-f]{8}", text)
    return m.group(0) if m else None


def load_strict():
    recs = {}
    for r in A.read_tsv(STRICT_TSV):
        h = scene_hash(r["task_id"]) or r["task_id"]
        key = (h, round(float(r["distance"]), 1))
        recs[(key, r["baseline"])] = record_from_view(A.row_view(r))
    return recs


def load_baselines():
    recs = {}
    for f in glob.glob(os.path.join(BASE_RUNS, "ghost_*", "summary.json")):
        base = os.path.basename(os.path.dirname(f))
        m = re.match(r"ghost_scene(\d+)_([0-9a-f]+)_(d[0-9p]+)_(reachset|shadow)", base)
        if not m:
            continue
        h = scene_hash(base)
        dist = DIST_CODE[m.group(3)]
        system = m.group(4)
        d = A.derive(A.load_summary(f))
        recs[((h, round(dist, 1)), system)] = record_from_view(d)
    return recs


def safety_block(records):
    n = len(records)
    impacts = [r["impact"] for r in records]
    hits = [r["impact"] for r in records if r["ghost_collision"]]
    coll = sum(r["ghost_collision"] for r in records)
    return {
        "n": n,
        "low_sev_pct": 100.0 * sum(r["paper_safe"] for r in records) / n if n else float("nan"),
        "zero_coll_pct": 100.0 * sum(r["zero_collision"] for r in records) / n if n else float("nan"),
        "coll_pct": 100.0 * coll / n if n else float("nan"),
        "mean_impact": sum(impacts) / n if n else float("nan"),
        "mean_impact_hit": sum(hits) / len(hits) if hits else float("nan"),
        "mean_v2": sum(v * v for v in impacts) / n if n else float("nan"),
    }


LABELS = {
    "ours": "PA-LOI + AEB",
    "aeb_only": "AEB-only",
    "mind": "MIND",
    "reachset": "Reachable-set",
    "shadow": "Dynamic-shadow",
}
# Conservatism order for table rows: aggressive -> conservative -> ours.
ROW_ORDER = ["mind", "aeb_only", "reachset", "shadow", "ours"]


def fmt_frac(num, den):
    p = 100.0 * num / den if den else float("nan")
    return f"{int(round(num))}/{int(den)} ({p:.1f}\\%)"


def per_distance_report(recs, dists):
    print()
    print("=== per-distance matched (cells where ours+reachset+shadow all spawned) ===")
    print(f"{'dist':>5s} {'nMatch':>6s}  ours/reach/shadow low-sev%   |  ours/reach/shadow v2")
    for d in sorted(dists, reverse=True):
        trio = ["ours", "reachset", "shadow"]
        keys = None
        for s in trio:
            ks = {k for (k, sy) in recs if sy == s and k[1] == d and recs[(k, s)]["spawned"]}
            keys = ks if keys is None else (keys & ks)
        keys = keys or set()
        if not keys:
            print(f"{d:5.1f} {0:6d}  (no common spawned cells yet)")
            continue
        cells = {s: [recs[(k, s)] for k in keys] for s in trio}
        ls = {s: 100.0 * sum(r["paper_safe"] for r in cells[s]) / len(cells[s]) for s in trio}
        v2 = {s: sum(r["impact"] ** 2 for r in cells[s]) / len(cells[s]) for s in trio}
        print(f"{d:5.1f} {len(keys):6d}  {ls['ours']:5.1f}/{ls['reachset']:5.1f}/{ls['shadow']:5.1f}"
              f"          |  {v2['ours']:5.2f}/{v2['reachset']:5.2f}/{v2['shadow']:5.2f}")


def emit_tex(recs, keys_by_system, dists, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Table 1: per-system safety among each system's own evaluable set, with
    # spawn-rate conservatism column.
    lines = [
        r"% AUTO-GENERATED by agg_five_system_matched.py (strict 650-frame horizon, all six trigger distances).",
        r"\begin{table*}[t]",
        r"\caption{Five-System Ghost-Probe Safety on Clean81 (Strict 650-Frame Horizon). "
        r"Spawn Rate Is a Conservatism Indicator; Safety Is Reported on Each System's Evaluable Set. "
        r"Low-Severity Means Zero Collision or Ghost Contact $\leq 3.0$\,m/s.}",
        r"\label{tab:five-system-safety}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & Spawn rate & \makecell{Eval.\\tasks} & Low-severity & Zero collision & "
        r"\makecell{Mean $v_{\mathrm{imp}}$\\(m/s)} & \makecell{Hit-only $v_{\mathrm{imp}}$\\(m/s)} & "
        r"\makecell{Mean $v_{\mathrm{imp}}^2$} \\",
        r"\midrule",
    ]
    for s in ROW_ORDER:
        ks = keys_by_system.get(s, set())
        nrun = len(ks)
        spawned = sum(recs[(k, s)]["spawned"] for k in ks)
        ev = [recs[(k, s)] for k in ks if recs[(k, s)]["evaluable"]]
        b = safety_block(ev)
        spawn_str = f"{100*spawned/nrun:.1f}\\%" if nrun else "--"
        lines.append(
            f"{LABELS[s]} & {spawn_str} & {b['n']} & "
            f"{fmt_frac(round(b['low_sev_pct']*b['n']/100), b['n'])} & "
            f"{fmt_frac(round(b['zero_coll_pct']*b['n']/100), b['n'])} & "
            f"{b['mean_impact']:.2f} & {b['mean_impact_hit']:.2f} & {b['mean_v2']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    with open(os.path.join(out_dir, "five_system_safety.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Table 2: matched subset where all present systems spawned.
    present = [s for s in ROW_ORDER if keys_by_system.get(s)]
    common = set.intersection(*[keys_by_system[s] for s in present]) if present else set()
    matched = [k for k in common if all(recs[(k, s)]["spawned"] for s in present)]
    m = [
        r"% AUTO-GENERATED matched-subset table.",
        r"\begin{table}[t]",
        rf"\caption{{Matched Ghost-Probe Safety on the {len(matched)} Cells Where All Systems Spawn. "
        r"Best Low-Severity and Impact-Energy Values in Bold.}",
        r"\label{tab:five-system-matched}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Low-severity & Collision rate & \makecell{Mean $v_{\mathrm{imp}}^2$} \\",
        r"\midrule",
    ]
    blocks = {s: safety_block([recs[(k, s)] for k in matched]) for s in present}
    best_ls = max(present, key=lambda s: blocks[s]["low_sev_pct"]) if present else None
    best_v2 = min(present, key=lambda s: blocks[s]["mean_v2"]) if present else None
    for s in present:
        b = blocks[s]
        ls = f"{b['low_sev_pct']:.1f}\\%"
        v2 = f"{b['mean_v2']:.2f}"
        if s == best_ls:
            ls = rf"\textbf{{{ls}}}"
        if s == best_v2:
            v2 = rf"\textbf{{{v2}}}"
        m.append(f"{LABELS[s]} & {ls} & {b['coll_pct']:.1f}\\% & {v2} \\\\")
    m += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(out_dir, "five_system_matched.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(m))
    print(f"\n[emit-tex] wrote five_system_safety.tex + five_system_matched.tex to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", default="5.5,5.0,4.5",
                    help="comma list of trigger distances to include")
    ap.add_argument("--emit-tex", default=None, help="write LaTeX tables to this directory")
    ap.add_argument("--csv", default=None, help="write per-scene paired CSV to this path")
    args = ap.parse_args()
    dists = [round(float(x), 1) for x in args.distances.split(",")]

    recs = {}
    recs.update(load_strict())
    recs.update(load_baselines())

    # Restrict to chosen distances.
    keys_by_system = defaultdict(set)
    for (key, system), _ in recs.items():
        if key[1] in dists:
            keys_by_system[system].add(key)

    print(f"distances included: {sorted(dists, reverse=True)}")
    print(f"strict TSV path : {STRICT_TSV}")
    print(f"baseline runs   : {BASE_RUNS}")
    print()
    print("=== coverage (cells present per system) ===")
    for s in SYSTEMS:
        per_d = {d: sum(1 for k in keys_by_system[s] if k[1] == d) for d in sorted(dists, reverse=True)}
        print(f"  {s:9s} total={len(keys_by_system[s]):4d}  {per_d}")

    # Spawn / evaluable spectrum.
    print()
    print("=== spawn / evaluable spectrum (own run set, chosen dists) ===")
    print(f"{'system':9s} {'nRun':>5s} {'spawn%':>7s} {'eval%':>7s}")
    for s in SYSTEMS:
        ks = keys_by_system[s]
        n = len(ks)
        sp = sum(recs[(k, s)]["spawned"] for k in ks)
        ev = sum(recs[(k, s)]["evaluable"] for k in ks)
        print(f"{s:9s} {n:5d} {100*sp/n if n else float('nan'):7.1f} {100*ev/n if n else float('nan'):7.1f}")

    # Per-system safety among each system's own evaluable cells.
    print()
    print("=== safety AMONG EVALUABLE (each system on own evaluable set) ===")
    hdr = f"{'system':9s} {'nEval':>5s} {'lowSev%':>8s} {'coll%':>6s} {'meanV':>6s} {'hitV':>6s} {'meanV2':>7s}"
    print(hdr)
    for s in SYSTEMS:
        rs = [recs[(k, s)] for k in keys_by_system[s] if recs[(k, s)]["evaluable"]]
        b = safety_block(rs)
        print(f"{s:9s} {b['n']:5d} {b['low_sev_pct']:8.1f} {b['coll_pct']:6.1f} "
              f"{b['mean_impact']:6.2f} {b['mean_impact_hit']:6.2f} {b['mean_v2']:7.2f}")

    # Matched on cells where ALL present systems spawned.
    present = [s for s in SYSTEMS if keys_by_system[s]]
    common = set.intersection(*[keys_by_system[s] for s in present]) if present else set()
    matched = [k for k in common if all(recs[(k, s)]["spawned"] for s in present)]
    print()
    print(f"=== MATCHED among cells where ALL {len(present)} systems spawned (n={len(matched)}) ===")
    print(hdr)
    for s in present:
        rs = [recs[(k, s)] for k in matched]
        b = safety_block(rs)
        print(f"{s:9s} {b['n']:5d} {b['low_sev_pct']:8.1f} {b['coll_pct']:6.1f} "
              f"{b['mean_impact']:6.2f} {b['mean_impact_hit']:6.2f} {b['mean_v2']:7.2f}")

    # Pairwise per-scene impact: ours vs each baseline on cells both spawned.
    for opp in ["reachset", "shadow"]:
        if not keys_by_system[opp]:
            continue
        both = [k for k in keys_by_system["ours"] & keys_by_system[opp]
                if recs[(k, "ours")]["spawned"] and recs[(k, opp)]["spawned"]]
        print()
        print(f"=== PAIRWISE ours vs {opp} (both spawned, n={len(both)}) ===")
        o = safety_block([recs[(k, "ours")] for k in both])
        x = safety_block([recs[(k, opp)] for k in both])
        print(f"  ours : lowSev%={o['low_sev_pct']:.1f} coll%={o['coll_pct']:.1f} meanV={o['mean_impact']:.2f} meanV2={o['mean_v2']:.2f}")
        print(f"  {opp:5s}: lowSev%={x['low_sev_pct']:.1f} coll%={x['coll_pct']:.1f} meanV={x['mean_impact']:.2f} meanV2={x['mean_v2']:.2f}")
        ours_lower = sum(1 for k in both if recs[(k, "ours")]["impact"] < recs[(k, opp)]["impact"] - 1e-6)
        opp_lower = sum(1 for k in both if recs[(k, opp)]["impact"] < recs[(k, "ours")]["impact"] - 1e-6)
        tie = len(both) - ours_lower - opp_lower
        print(f"  per-scene impact: ours-lower={ours_lower}  {opp}-lower={opp_lower}  tie/both-zero={tie}")
        both_hit = [k for k in both if recs[(k, "ours")]["ghost_collision"] and recs[(k, opp)]["ghost_collision"]]
        if both_hit:
            do = sum(recs[(k, "ours")]["impact"] for k in both_hit) / len(both_hit)
            dx = sum(recs[(k, opp)]["impact"] for k in both_hit) / len(both_hit)
            print(f"  both-collide cells (n={len(both_hit)}): mean impact ours={do:.2f} vs {opp}={dx:.2f}")

    per_distance_report(recs, dists)
    if args.emit_tex:
        emit_tex(recs, keys_by_system, dists, args.emit_tex)
    if args.csv:
        import csv as _csv
        all_keys = sorted(set().union(*keys_by_system.values()), key=lambda k: (k[1], k[0]))
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            cols = ["hash", "distance"]
            for s in ROW_ORDER:
                cols += [f"{s}_spawned", f"{s}_evaluable", f"{s}_ghost_coll",
                         f"{s}_impact", f"{s}_paper_safe"]
            w = _csv.writer(f)
            w.writerow(cols)
            for k in all_keys:
                row = [k[0], k[1]]
                for s in ROW_ORDER:
                    r = recs.get((k, s))
                    if r is None:
                        row += ["", "", "", "", ""]
                    else:
                        row += [int(r["spawned"]), int(r["evaluable"]), int(r["ghost_collision"]),
                                f"{r['impact']:.3f}" if r["ghost_collision"] else "", int(r["paper_safe"])]
                w.writerow(row)
        print(f"[csv] wrote per-scene paired table to {args.csv}")


if __name__ == "__main__":
    main()
