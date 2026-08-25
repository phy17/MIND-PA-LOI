#!/usr/bin/env python3
"""Generate the two sensitivity-sweep tables from sweep_results/.

Inputs (repo root, produced by experiments/ghost_probe/run_cloud_sweep_queue.py):
  sweep_results/e1/<variant>/{ghost_*,noghost_*}/summary.json   486 tasks
  sweep_results/e2/s*_d*_v*_<stack>/summary.json                486 tasks
  实验记录/clean81_ghost_strict650_20260613.tsv                  deployed ghost ref
  实验记录/clean81_noghost_efficiency_details_20260607.tsv       deployed no-ghost ref

Outputs:
  tables/param_ablation.tex      (E1: six one-dimensional variants vs deployed)
  tables/extended_distance.tex   (E2: dash protocol at 4.5/3.5 m triggers)

Accounting notes (audited 2026-07-20):
  - deployed ghost reference uses the strict-650 table only (no h800 cells),
    restricted to the identical 27 scenes x {5.5, 4.0} m cells;
  - unspawned variant cells count as safe in the all-cell columns, hence the
    spawn column and the spawned-only low-severity column;
  - v_imp^2 in the E1 table is over spawned cells; in the E2 table the impact
    columns are hit-only means, matching the published speed-sensitivity table;
  - no-ghost planner failures / replay collisions are read from
    planner_failure_frame / collision_count_ground_truth.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SWEEP = ROOT / "sweep_results"
TAB_DIR = Path(__file__).resolve().parents[1] / "tables"
LOW = 3.0

E1_ORDER = ["eta0p4", "eta0p8", "vfloor1p0", "vfloor3p0", "wmax15", "wmax35"]
E1_LABELS = {
    "eta0p4": r"$\eta=0.4$", "eta0p8": r"$\eta=0.8$",
    "vfloor1p0": r"$v_{\mathrm{floor}}=1.0$", "vfloor3p0": r"$v_{\mathrm{floor}}=3.0$",
    "wmax15": r"$w_{\max}=15$", "wmax35": r"$w_{\max}=35$",
}
E2_STACKS = ["ours", "aeb_only", "mind"]
E2_LABELS = {"ours": "PA-LOI + AEB", "aeb_only": "AEB-only", "mind": "MIND"}


def load(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    return d[0] if isinstance(d, list) else d


def gm(s: dict) -> dict:
    fc = s.get("first_collision") or None
    ghost = bool(fc and str(fc.get("other_id", "")).startswith("GHOST"))
    any_hit = bool(s.get("collision_count", 0) > 0)
    bg = bool(fc and not str(fc.get("other_id", "")).startswith("GHOST"))
    imp = float(fc.get("ego_vel", 0.0)) if ghost else 0.0
    return {
        "spawned": bool(s.get("ghost_spawned")), "ghost": ghost, "bg": bg,
        "any": any_hit, "imp": imp,
        "low": (not any_hit) or (ghost and not bg and imp <= LOW),
        "zero": not any_hit,
    }


def scenes27() -> set[int]:
    s = {int(p.stem.replace("scene", "")) for p in (SWEEP / "records").glob("scene*.jsonl")}
    assert len(s) == 27, f"expected 27 scenes, got {len(s)}"
    return s


def e1_variant(vdir: Path) -> tuple[list[dict], list[dict]]:
    ghost, ng = [], []
    for td in sorted(p for p in vdir.iterdir() if p.is_dir()):
        s = load(td / "summary.json")
        if td.name.startswith("ghost_"):
            ghost.append(gm(s))
        else:
            ng.append({
                "fail": s.get("planner_failure_frame") is not None,
                "replay": (s.get("collision_count_ground_truth") or 0) > 0,
                "speed": float(s["avg_speed_control_mps"]),
            })
    assert len(ghost) == 54 and len(ng) == 27, vdir
    return ghost, ng


def deployed_rows(scenes: set[int]) -> tuple[list[dict], list[dict]]:
    ghost = []
    with open(ROOT / "实验记录" / "clean81_ghost_strict650_20260613.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["baseline"] != "ours" or r["distance"] not in ("5.5", "4.0"):
                continue
            if int(r["clean81_index"]) not in scenes:
                continue
            g = r["ghost_collision"] == "True"
            ghost.append({
                "spawned": r["ghost_spawned"] == "True", "ghost": g,
                "any": r["any_collision"] == "True",
                "imp": float(r["impact_speed"] or 0) if g else 0.0,
                "low": r["paper_safe"] == "True",
                "zero": r["zero_collision_safe"] == "True",
            })
    ng = []
    with open(ROOT / "实验记录" / "clean81_noghost_efficiency_details_20260607.tsv",
              encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["baseline"] != "ours" or int(r["clean81_index"]) not in scenes:
                continue
            ng.append({
                "fail": r["planner_failure"] == "True",
                "replay": r["ground_truth_collision"] == "True",
                "speed": float(r["avg_speed_control_mps"]),
            })
    assert len(ghost) == 54 and len(ng) == 27
    return ghost, ng


def e1_line(label: str, ghost: list[dict], ng: list[dict]) -> str:
    spawn = sum(x["spawned"] for x in ghost)
    sp = [x for x in ghost if x["spawned"]]
    low_all = 100 * sum(x["low"] for x in ghost) / len(ghost)
    low_sp = 100 * sum(x["low"] for x in sp) / len(sp)
    zero_all = 100 * sum(x["zero"] for x in ghost) / len(ghost)
    v2_sp = sum(x["imp"] ** 2 for x in sp) / len(sp)
    speed = sum(x["speed"] for x in ng) / len(ng)
    fails = sum(x["fail"] for x in ng)
    replays = sum(x["replay"] for x in ng)
    return (f"{label} & {spawn}/54 & {low_all:.1f}\\% & {low_sp:.1f}\\% & "
            f"{zero_all:.1f}\\% & {v2_sp:.2f} & {speed:.2f} & {fails}/{replays} \\\\")


def build_e1() -> str:
    scenes = scenes27()
    dep_g, dep_ng = deployed_rows(scenes)
    rows = [e1_line(r"Deployed ($0.6/2.0/25$)", dep_g, dep_ng), r"\midrule"]
    for name in E1_ORDER:
        g, ng = e1_variant(SWEEP / "e1" / name)
        rows.append(e1_line(E1_LABELS[name], g, ng))
    return "\n".join([
        r"\begin{table*}[!ht]",
        r"\caption{Risk-Layer Parameter Perturbation on the 27-Scene Set (Ghost at 5.5 and 4.0\,m,"
        r" Instant-Center, 650 Frames, Plus No-Ghost Efficiency). The Deployed Row Uses the"
        r" Published Strict-650 Tables on the Identical Cells.}",
        r"\label{tab:param-ablation}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Setting & Spawn & \makecell{Low-severity\\(all 54 cells)} & \makecell{Low-severity\\(spawned)} & "
        r"\makecell{Zero collision\\(all 54 cells)} & \makecell{Mean $v_{\mathrm{imp}}^2$\\(spawned)} & "
        r"\makecell{No-ghost\\speed (m/s)} & \makecell{PF/RC} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{2pt}{\scriptsize Unspawned cells (ego never reaches the trigger within 650"
        r" frames) count as safe in all-cell columns; the spawned-only column removes this credit."
        r" PF/RC: planner failures / replay collisions over 27 no-ghost runs; the $w_{\max}=15$"
        r" speed includes one run truncated at frame 630 (3.46\,m/s excluding it).}",
        r"\end{table*}",
        "",
    ])


def build_e2() -> str:
    pat = re.compile(r"s(\d+)_d([0-9p]+)_v([0-9p]+)_(\w+)$")
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for td in sorted(p for p in (SWEEP / "e2").iterdir() if p.is_dir()):
        m = pat.match(td.name)
        if not m:
            continue
        cells[(m.group(2).replace("p", "."), m.group(3).replace("p", "."), m.group(4))].append(
            gm(load(td / "summary.json")))
    def stats(d: str, v: str, b: str) -> dict:
        g = cells[(d, v, b)]
        hits = [x for x in g if x["ghost"]]
        return {
            "n": len(g),
            "low": sum(x["low"] for x in g),
            "zero": sum(x["zero"] for x in g),
            "v2_hit": (sum(x["imp"] ** 2 for x in hits) / len(hits)) if hits else 0.0,
        }

    rows = []
    dists = sorted({d for d, _, _ in cells}, reverse=True)
    speeds = sorted({v for _, v, _ in cells}, key=float)
    for si, v in enumerate(speeds):
        per = {(d, b): stats(d, v, b) for d in dists for b in E2_STACKS}
        best_low = {d: max(E2_STACKS, key=lambda b: per[(d, b)]["low"]) for d in dists}
        best_v2 = {d: min(E2_STACKS, key=lambda b: per[(d, b)]["v2_hit"]) for d in dists}
        for k, b in enumerate(E2_STACKS):
            head = rf"\multirow{{3}}{{*}}{{{float(v):.1f}}}" if k == 0 else ""
            fields = []
            for d in dists:
                s = per[(d, b)]
                low_s = f"{s['low']}/{s['n']} ({100 * s['low'] / s['n']:.1f}\\%)"
                if b == best_low[d]:
                    low_s = rf"\textbf{{{low_s}}}"
                v2_s = f"{s['v2_hit']:.2f}"
                if b == best_v2[d]:
                    v2_s = rf"\textbf{{{v2_s}}}"
                fields += [low_s, f"{s['zero']}/{s['n']} ({100 * s['zero'] / s['n']:.1f}\\%)", v2_s]
            rows.append(f"{head} & {E2_LABELS[b]} & " + " & ".join(fields) + r" \\")
        if si != len(speeds) - 1:
            rows.append(r"\midrule")
    return "\n".join([
        r"\begin{table*}[!ht]",
        r"\caption{Moving-Pedestrian Emergence at Extended Trigger Distances (27 Scenes, 800"
        r" Frames). $v_{\mathrm{imp}}^2$ Columns Are Hit-Only Means, as in"
        r" the Main-Paper Pedestrian-Speed Table. Best Low-Severity and Impact-Energy Values per"
        r" Distance--Speed Cell in Bold.}",
        r"\label{tab:extended-distance}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{clcccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{$d_{\mathrm{trig}}=\SI{4.5}{m}$} & "
        r"\multicolumn{3}{c}{$d_{\mathrm{trig}}=\SI{3.5}{m}$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"\makecell{Speed\\(m/s)} & Method & Low-severity & Zero coll. & "
        r"\makecell{Mean $v_{\mathrm{imp}}^2$\\(m$^2$/s$^2$)} & Low-severity & Zero coll. & "
        r"\makecell{Mean $v_{\mathrm{imp}}^2$\\(m$^2$/s$^2$)} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{2pt}{\scriptsize In one scene AEB-only hits replay background traffic before"
        r" the trigger in all six cells (counted unsafe); excluding that scene for all stacks"
        r" changes no rate by more than 3.1\,pp and no ordering.}",
        r"\end{table*}",
        "",
    ])


def main() -> None:
    TAB_DIR.mkdir(exist_ok=True)
    (TAB_DIR / "param_ablation.tex").write_text(build_e1(), encoding="utf-8")
    (TAB_DIR / "extended_distance.tex").write_text(build_e2(), encoding="utf-8")
    print("wrote", TAB_DIR / "param_ablation.tex")
    print("wrote", TAB_DIR / "extended_distance.tex")


if __name__ == "__main__":
    main()
