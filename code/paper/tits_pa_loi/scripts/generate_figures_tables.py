#!/usr/bin/env python3
"""Generate T-ITS paper figures and tables from experiment result files."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PAPER_DIR / "figures"
TAB_DIR = PAPER_DIR / "tables"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PAPER_DIR / ".matplotlib"))

import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams.update(
    {
        # Match the manuscript body font (newtx Times) so figure text and
        # math render in the same family as the surrounding prose.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

GHOST_TSV = ROOT / "实验记录" / "clean81_ghost_task_details_threshold3mps_20260607.tsv"
NOGHOST_TSV = ROOT / "实验记录" / "clean81_noghost_efficiency_details_20260607.tsv"
NOGHOST_BASELINE_RUNS = (
    ROOT
    / "实验记录"
    / "20260612_clean81_baselines_reachset_shadow_w24"
    / "runs"
)
SPEED_TSV = (
    ROOT
    / "实验记录"
    / "20260607_clean81_rep30_dash_speed_sensitivity_d5p5_4speeds_3baselines_30workers"
    / "speed_sensitivity_paper_table_excl23_44_70.tsv"
)
CLEAN81_JSONL = ROOT / "数据集" / "ghost_injection_clean81_clean21_plus_new60_20260607.jsonl"
REP30_JSONL = ROOT / "数据集" / "ghost_injection_clean81_rep30_speed_sensitivity_d5p5_20260607.jsonl"

BASELINES = ["ours", "aeb_only", "mind"]
BASELINE_LABELS = {
    "ours": "PA-LOI + AEB",
    "aeb_only": "AEB-only",
    "mind": "MIND",
    "reachset": "Reachable-set",
    "shadow": "Dynamic-shadow",
}
BASELINE_COLORS = {
    "ours": "#0B6E69",
    "aeb_only": "#D8841C",
    "mind": "#6B7280",
    "reachset": "#7C3AED",
    "shadow": "#2563EB",
}
NOGHOST_ORDER = ["mind", "reachset", "shadow", "ours"]
NOGHOST_FIG_LABELS = {
    "mind": "MIND",
    "reachset": "Reach-set",
    "shadow": "Shadow",
    "ours": "PA-LOI",
}
MATCHED_LOW_SEVERITY = {
    "mind": 6.2,
    "reachset": 34.4,
    "shadow": 34.9,
    "ours": 58.5,
}
# Redundant encodings so the line figures survive grayscale printing.
BASELINE_MARKERS = {"ours": "o", "aeb_only": "s", "mind": "^"}
BASELINE_LINESTYLES = {"ours": "-", "aeb_only": (0, (4, 2)), "mind": (0, (1, 1.5))}
DISTANCES = [5.5, 5.0, 4.5, 4.0, 3.5, 3.0]
SPEEDS = [1.0, 1.5, 2.0, 3.0]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_extra_noghost_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for baseline in ["reachset", "shadow"]:
        for path in sorted(NOGHOST_BASELINE_RUNS.glob(f"noghost_*_{baseline}/summary.json")):
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            summary = data[0] if isinstance(data, list) else data
            planner_failure = summary.get("planner_failure_frame") not in (None, "")
            ground_truth_collision = int(summary.get("collision_count_ground_truth", 0) or 0) > 0
            rows.append(
                {
                    "avg_speed_control_mps": str(summary.get("avg_speed_control_mps", 0.0)),
                    "baseline": baseline,
                    "collision_count_ground_truth": str(summary.get("collision_count_ground_truth", 0)),
                    "completed_full_horizon": str(bool(summary.get("completed_full_horizon"))),
                    "distance_control_m": str(summary.get("distance_control_m", 0.0)),
                    "ground_truth_collision": str(ground_truth_collision),
                    "planner_failure": str(planner_failure),
                    "planner_failure_frame": "" if not planner_failure else str(summary.get("planner_failure_frame")),
                    "slow_pct_control_lt6": str(summary.get("slow_pct_control_lt6", 0.0)),
                    "stop_pct_control_lt0p3": str(summary.get("stop_pct_control_lt0p3", 0.0)),
                    "summary_path": str(path),
                    "task_id": path.parent.name,
                    "terminated": str(bool(summary.get("terminated"))),
                }
            )
    return rows


def as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def as_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def as_int(value: str, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def fmt_pct(num: float, den: float) -> str:
    return f"{num:.0f}/{den:.0f} ({pct(num, den):.1f}\\%)"


def validate_inputs(
    ghost_rows: list[dict[str, str]],
    noghost_rows: list[dict[str, str]],
    speed_rows: list[dict[str, str]],
    clean81_rows: list[dict],
    rep30_rows: list[dict],
) -> None:
    assert len(clean81_rows) == 81, len(clean81_rows)
    assert len(ghost_rows) == 81 * 6 * 3, len(ghost_rows)
    assert len(noghost_rows) == 81 * 2, len(noghost_rows)
    assert len(speed_rows) == 4 * 3, len(speed_rows)
    assert len(rep30_rows) == 30, len(rep30_rows)
    assert {r["baseline"] for r in ghost_rows} == set(BASELINES)
    assert {round(as_float(r["distance"]), 1) for r in ghost_rows} == set(DISTANCES)
    assert {r["baseline"] for r in noghost_rows} == {"ours", "mind"}
    assert {r["baseline"] for r in speed_rows} == set(BASELINES)
    assert {round(as_float(r["speed"]), 1) for r in speed_rows} == set(SPEEDS)
    scene_ids = [r["scenario_id"] for r in clean81_rows]
    assert len(scene_ids) == len(set(scene_ids)), "duplicate clean81 scenario IDs"


def validate_noghost_all(noghost_rows: list[dict[str, str]]) -> None:
    expected = set(NOGHOST_ORDER)
    assert {r["baseline"] for r in noghost_rows} == expected
    for baseline in expected:
        rows = [r for r in noghost_rows if r["baseline"] == baseline]
        assert len(rows) == 81, (baseline, len(rows))


def summarize_main(ghost_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for baseline in BASELINES:
        rows = [r for r in ghost_rows if r["baseline"] == baseline]
        n = len(rows)
        impacts = [as_float(r["impact_speed"]) for r in rows]
        positive_impacts = [v for v in impacts if v > 0.0]
        summary[baseline] = {
            "n": n,
            "paper_safe": sum(as_bool(r["paper_safe"]) for r in rows),
            "zero_collision": sum(as_bool(r["zero_collision_safe"]) for r in rows),
            "raw_no_ghost_collision": sum(as_bool(r["raw_no_ghost_collision"]) for r in rows),
            "ghost_spawned": sum(as_bool(r["ghost_spawned"]) for r in rows),
            "ghost_collision": sum(as_bool(r["ghost_collision"]) for r in rows),
            "background_collision": sum(as_bool(r["background_collision"]) for r in rows),
            "mean_impact": float(np.mean(impacts)),
            "mean_impact_when_hit": float(np.mean(positive_impacts)) if positive_impacts else 0.0,
            "mean_impact_v2": float(np.mean(np.square(impacts))),
        }
    return summary


def summarize_by_distance(ghost_rows: list[dict[str, str]]) -> dict[tuple[str, float], dict[str, float]]:
    summary: dict[tuple[str, float], dict[str, float]] = {}
    for baseline in BASELINES:
        for distance in DISTANCES:
            rows = [
                r
                for r in ghost_rows
                if r["baseline"] == baseline and math.isclose(as_float(r["distance"]), distance)
            ]
            impacts = [as_float(r["impact_speed"]) for r in rows]
            summary[(baseline, distance)] = {
                "n": len(rows),
                "paper_safe": sum(as_bool(r["paper_safe"]) for r in rows),
                "zero_collision": sum(as_bool(r["zero_collision_safe"]) for r in rows),
                "mean_impact": float(np.mean(impacts)),
                "mean_impact_v2": float(np.mean(np.square(impacts))),
            }
    return summary


def summarize_by_dataset(ghost_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for dataset in ["clean21", "new60"]:
        for baseline in BASELINES:
            rows = [r for r in ghost_rows if r["baseline"] == baseline and r["dataset"] == dataset]
            impacts = [as_float(r["impact_speed"]) for r in rows]
            summary[(dataset, baseline)] = {
                "n": len(rows),
                "paper_safe": sum(as_bool(r["paper_safe"]) for r in rows),
                "zero_collision": sum(as_bool(r["zero_collision_safe"]) for r in rows),
                "mean_impact": float(np.mean(impacts)),
            }
    return summary


def summarize_noghost(noghost_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for baseline in NOGHOST_ORDER:
        rows = [r for r in noghost_rows if r["baseline"] == baseline]
        summary[baseline] = {
            "n": len(rows),
            "avg_speed": float(np.mean([as_float(r["avg_speed_control_mps"]) for r in rows])),
            "distance": float(np.mean([as_float(r["distance_control_m"]) for r in rows])),
            "slow_pct": float(np.mean([as_float(r["slow_pct_control_lt6"]) for r in rows])),
            "stop_pct": float(np.mean([as_float(r["stop_pct_control_lt0p3"]) for r in rows])),
            "planner_fail": sum(as_bool(r["planner_failure"]) for r in rows),
            "gt_collision": sum(as_bool(r["ground_truth_collision"]) for r in rows),
            "full_horizon": sum(as_bool(r["completed_full_horizon"]) for r in rows),
            "terminated": sum(as_bool(r["terminated"]) for r in rows),
        }
    return summary


def summarize_speed(speed_rows: list[dict[str, str]]) -> dict[tuple[float, str], dict[str, float]]:
    summary: dict[tuple[float, str], dict[str, float]] = {}
    for r in speed_rows:
        speed = round(as_float(r["speed"]), 1)
        baseline = r["baseline"]
        summary[(speed, baseline)] = {
            "n": as_int(r["n"]),
            "spawned": as_int(r["spawned"]),
            "no_collision": as_int(r["no_collision"]),
            "ghost_collision": as_int(r["ghost_collision"]),
            "low_speed_contact": as_int(r["low_speed_contact"]),
            "paper_safe": as_int(r["paper_safe"]),
            "high_impact": as_int(r["high_impact"]),
            "bg_first": as_int(r["bg_first"]),
            "bg_after": as_int(r["bg_after"]),
            "mean_impact_v": as_float(r["mean_impact_v"]),
            "mean_impact_v2": as_float(r["mean_impact_v2"]),
        }
    return summary


def write_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_tables(
    ghost_rows: list[dict[str, str]],
    noghost_rows: list[dict[str, str]],
    speed_rows: list[dict[str, str]],
    clean81_rows: list[dict],
    rep30_rows: list[dict],
) -> None:
    main = summarize_main(ghost_rows)
    by_dist = summarize_by_distance(ghost_rows)
    by_dataset = summarize_by_dataset(ghost_rows)
    noghost = summarize_noghost(noghost_rows)
    speed = summarize_speed(speed_rows)

    split_counts = defaultdict(int)
    for r in clean81_rows:
        split_counts[r.get("_clean81_source", "unknown")] += 1

    dataset_table = r"""\begin{table}[t]
\caption{Dataset and Experiment Scale}
\label{tab:dataset-scale}
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lcccc}
\toprule
Protocol & Scenes & Baselines & Conditions & Tasks \\
\midrule
Conference seed & 2 & 3 & 6 trigger distances & 36 \\
Clean21 main set & 21 & 3 & 6 trigger distances & 378 \\
New60 extension & 60 & 3 & 6 trigger distances & 1080 \\
Clean81 ghost-probe (primary) & 81 & 3 & 6 trigger distances & 1458 \\
Occlusion-aware baselines & 81 & 2 & 6 trigger distances & 972 \\
Clean81 no-ghost & 81 & 4 & no injection & 324 \\
Representative speed set & 27 & 3 & 4 walking speeds & 324 \\
\bottomrule
\end{tabular}
\end{table}
"""
    write_file(TAB_DIR / "dataset_scale.tex", dataset_table)

    best_safe = max(BASELINES, key=lambda b: main[b]["paper_safe"])
    best_zero = max(BASELINES, key=lambda b: main[b]["zero_collision"])
    best_imp = min(BASELINES, key=lambda b: main[b]["mean_impact"])
    best_hit = min(BASELINES, key=lambda b: main[b]["mean_impact_when_hit"])
    best_v2 = min(BASELINES, key=lambda b: main[b]["mean_impact_v2"])

    def bold_if(text: str, cond: bool) -> str:
        return rf"\textbf{{{text}}}" if cond else text

    rows = []
    for baseline in BASELINES:
        s = main[baseline]
        imp = bold_if(f"{s['mean_impact']:.2f}", baseline == best_imp)
        hit = bold_if(f"{s['mean_impact_when_hit']:.2f}", baseline == best_hit)
        v2 = bold_if(f"{s['mean_impact_v2']:.2f}", baseline == best_v2)
        rows.append(
            f"{BASELINE_LABELS[baseline]} & {int(s['n'])} & "
            f"{bold_if(fmt_pct(s['paper_safe'], s['n']), baseline == best_safe)} & "
            f"{bold_if(fmt_pct(s['zero_collision'], s['n']), baseline == best_zero)} & "
            f"{int(s['ghost_collision'])} & {int(s['background_collision'])} & "
            f"{imp} & {hit} & {v2} \\\\"
        )
    main_table = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\caption{Clean81 Ghost-Probe Safety Results Over 1458 Tasks. Low-Severity Means Zero Collision or Contact Speed No Greater Than 3.0\,m/s; Zero Collision Is Reported Separately. Best Values in Bold.}",
            r"\label{tab:main-safety}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{7pt}",
            r"\begin{tabular}{lcccccccc}",
            r"\toprule",
            r"Method & Tasks & Low-severity & Zero collision & \makecell{Ghost\\collisions} & \makecell{Background\\collisions} & \makecell{Mean $v_{\mathrm{imp}}$\\(m/s)} & \makecell{Hit-only $v_{\mathrm{imp}}$\\(m/s)} & \makecell{Mean $v_{\mathrm{imp}}^2$\\(m$^2$/s$^2$)} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    write_file(TAB_DIR / "main_safety.tex", main_table)

    dist_rows = []
    for distance in DISTANCES:
        best_d_safe = max(BASELINES, key=lambda b: by_dist[(b, distance)]["paper_safe"])
        best_d_imp = min(BASELINES, key=lambda b: by_dist[(b, distance)]["mean_impact"])
        for k, baseline in enumerate(BASELINES):
            s = by_dist[(baseline, distance)]
            head = rf"\multirow{{3}}{{*}}{{{distance:.1f}}}" if k == 0 else ""
            dist_rows.append(
                f"{head} & {BASELINE_LABELS[baseline]} & "
                f"{bold_if(fmt_pct(s['paper_safe'], s['n']), baseline == best_d_safe)} & "
                f"{fmt_pct(s['zero_collision'], s['n'])} & "
                f"{bold_if(format(s['mean_impact'], '.2f'), baseline == best_d_imp)} \\\\"
            )
        if distance != DISTANCES[-1]:
            dist_rows.append(r"\midrule")
    dist_table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{Clean81 Results Grouped by Trigger Distance. Best Low-Severity and Mean-Impact Values per Distance in Bold.}",
            r"\label{tab:distance-safety}",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{clccc}",
            r"\toprule",
            r"\makecell{$d_{\mathrm{trig}}$\\(m)} & Method & Low-severity & Zero collision & \makecell{Mean $v_{\mathrm{imp}}$\\(m/s)} \\",
            r"\midrule",
            *dist_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    write_file(TAB_DIR / "distance_safety.tex", dist_table)

    noghost_rows_out = []
    for baseline in NOGHOST_ORDER:
        s = noghost[baseline]
        noghost_rows_out.append(
            f"{BASELINE_LABELS[baseline]} & {int(s['n'])} & "
            f"{s['avg_speed']:.2f} & {s['distance']:.2f} & "
            f"{s['slow_pct']:.1f}\\% & {s['stop_pct']:.2f}\\% & "
            f"{int(s['planner_fail'])} & {int(s['gt_collision'])} & {int(s['full_horizon'])} \\\\"
        )
    mind_speed = noghost["mind"]["avg_speed"]
    speed_losses = {
        baseline: pct(mind_speed - noghost[baseline]["avg_speed"], mind_speed)
        for baseline in NOGHOST_ORDER
        if baseline != "mind"
    }
    noghost_table = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\caption{No-Ghost Efficiency and Closed-Loop Stability on Clean81 (324 Runs)}",
            r"\label{tab:noghost-efficiency}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{6pt}",
            r"\begin{tabular}{lcccccccc}",
            r"\toprule",
            r"Method & Tasks & \makecell{Mean speed\\(m/s)} & \makecell{Distance\\(m)} & \makecell{Frames\\$<6$\,m/s} & \makecell{Frames\\$<0.3$\,m/s} & \makecell{Planner\\failures} & \makecell{Replay\\collisions} & \makecell{Full\\horizon} \\",
            r"\midrule",
            *noghost_rows_out,
            r"\bottomrule",
            r"\end{tabular}",
            "",
            rf"\vspace{{3pt}}{{\scriptsize Speed loss relative to MIND: PA-LOI {speed_losses['ours']:.1f}\%, Reachable-set {speed_losses['reachset']:.1f}\%, Dynamic-shadow {speed_losses['shadow']:.1f}\%.}}",
            r"\end{table*}",
            "",
        ]
    )
    write_file(TAB_DIR / "noghost_efficiency.tex", noghost_table)

    speed_rows_out = []
    for spd in SPEEDS:
        best_s_safe = max(BASELINES, key=lambda b: speed[(spd, b)]["paper_safe"])
        best_s_v2 = min(BASELINES, key=lambda b: speed[(spd, b)]["mean_impact_v2"])
        for k, baseline in enumerate(BASELINES):
            s = speed[(spd, baseline)]
            head = rf"\multirow{{3}}{{*}}{{{spd:.1f}}}" if k == 0 else ""
            speed_rows_out.append(
                f"{head} & {BASELINE_LABELS[baseline]} & "
                f"{bold_if(fmt_pct(s['paper_safe'], s['n']), baseline == best_s_safe)} & "
                f"{fmt_pct(s['no_collision'], s['n'])} & "
                f"{int(s['low_speed_contact'])} & {int(s['high_impact'])} & "
                f"{s['mean_impact_v']:.2f} & "
                f"{bold_if(format(s['mean_impact_v2'], '.2f'), baseline == best_s_v2)} \\\\"
            )
        if spd != SPEEDS[-1]:
            speed_rows_out.append(r"\midrule")
    speed_table = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\caption{Pedestrian-Speed Sensitivity on the Representative 27-Scene Set at 5.5\,m Trigger Distance. Best Low-Severity and Impact-Energy Values per Speed in Bold.}",
            r"\label{tab:speed-sensitivity}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{8pt}",
            r"\begin{tabular}{clcccccc}",
            r"\toprule",
            r"\makecell{Speed\\(m/s)} & Method & Low-severity & Zero collision & \makecell{Low-speed\\contacts} & \makecell{High-impact\\collisions} & \makecell{Mean $v_{\mathrm{imp}}$\\(hit-only, m/s)} & \makecell{Mean $v_{\mathrm{imp}}^2$\\(hit-only, m$^2$/s$^2$)} \\",
            r"\midrule",
            *speed_rows_out,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    write_file(TAB_DIR / "speed_sensitivity.tex", speed_table)

    split_rows = []
    for dataset in ["clean21", "new60"]:
        best_split_safe = max(BASELINES, key=lambda b: by_dataset[(dataset, b)]["paper_safe"])
        for k, baseline in enumerate(BASELINES):
            s = by_dataset[(dataset, baseline)]
            head = rf"\multirow{{3}}{{*}}{{{dataset}}}" if k == 0 else ""
            split_rows.append(
                f"{head} & {BASELINE_LABELS[baseline]} & "
                f"{bold_if(fmt_pct(s['paper_safe'], s['n']), baseline == best_split_safe)} & "
                f"{fmt_pct(s['zero_collision'], s['n'])} & {s['mean_impact']:.2f} \\\\"
            )
        if dataset != "new60":
            split_rows.append(r"\midrule")
    split_table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{Clean21--New60 Consistency Check. Best Low-Severity Rates per Split in Bold.}",
            r"\label{tab:split-consistency}",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{2.5pt}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Split & Method & Low-severity & Zero collision & \makecell{Mean $v_{\mathrm{imp}}$\\(m/s)} \\",
            r"\midrule",
            *split_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    write_file(TAB_DIR / "split_consistency.tex", split_table)


def draw_box(ax, xy, width, height, text, fc="#F8FAFC", ec="#0B3B8F", lw=1.2, fontsize=9):
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color="#0B3B8F", lw=1.5, connectionstyle="arc3,rad=0.0"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=4, shrinkB=4, connectionstyle=connectionstyle),
    )


def _draw_vehicle(ax, x, y, length, width, color, ec, zorder=5, heading_right=True):
    body = patches.FancyBboxPatch(
        (x - length / 2, y - width / 2),
        length,
        width,
        boxstyle="round,pad=0.0,rounding_size=0.25",
        facecolor=color,
        edgecolor=ec,
        linewidth=1.0,
        zorder=zorder,
    )
    ax.add_patch(body)
    if heading_right:
        tip = x + length / 2
        tri = patches.Polygon(
            [[tip - 0.15, y - width * 0.28], [tip - 0.15, y + width * 0.28], [tip + 0.75, y]],
            closed=True,
            facecolor=ec,
            edgecolor="none",
            zorder=zorder + 1,
        )
        ax.add_patch(tri)


def _shadow_polygon(eye, corner_a, corner_b, y_far):
    def extend(corner):
        dx = corner[0] - eye[0]
        dy = corner[1] - eye[1]
        if dy <= 1e-6:
            return [corner[0] + 40.0, corner[1]]
        t = (y_far - eye[1]) / dy
        return [eye[0] + t * dx, y_far]

    return [corner_a, extend(corner_a), extend(corner_b), corner_b]


def _draw_ghost_scene(ax, with_paloi: bool) -> None:
    x_max = 46.0
    ax.set_xlim(0, x_max)
    ax.set_ylim(-6.2, 7.4)
    ax.set_aspect("auto")
    ax.axis("off")

    # Road, parking strip, sidewalk.
    ax.add_patch(patches.Rectangle((0, -2.0), x_max, 4.0, facecolor="#E8EBF0", edgecolor="none", zorder=0))
    ax.add_patch(patches.Rectangle((0, 2.0), x_max, 1.3, facecolor="#DBD9D2", edgecolor="none", zorder=0))
    ax.add_patch(patches.Rectangle((0, 3.3), x_max, 2.4, facecolor="#F2EFE6", edgecolor="none", zorder=0))
    ax.plot([0, x_max], [2.0, 2.0], color="#FFFFFF", linewidth=1.6, zorder=1)
    ax.plot([0, x_max], [-2.0, -2.0], color="#9AA4B2", linewidth=1.2, zorder=1)
    ax.plot([0, x_max], [3.3, 3.3], color="#B8B2A4", linewidth=0.9, zorder=1)

    conflict = (29.5, -0.9)
    ego = (6.5, -0.9)
    occluder = (28.0, 2.65)

    # Ego route.
    ax.annotate(
        "",
        xy=(x_max - 2.0, ego[1]),
        xytext=(ego[0], ego[1]),
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#1F5AD6", linestyle=(0, (5, 3)), shrinkA=18, shrinkB=2),
        zorder=2,
    )
    ax.text(43.2, -3.2, "ego route", color="#1F5AD6", fontsize=6.2, ha="center")

    # Occluder vehicle with internal label.
    _draw_vehicle(ax, occluder[0], occluder[1], 6.4, 2.0, "#7C8796", "#3A4452", zorder=6, heading_right=False)
    ax.text(occluder[0], occluder[1] - 0.05, "parked\noccluder", fontsize=5.6, ha="center", va="center",
            color="#FFFFFF", zorder=8, linespacing=0.95)

    # Hidden pedestrian and dash-out path.
    ped = (29.5, 4.15)
    ax.add_patch(patches.Circle(ped, 0.42, facecolor="#D62728", edgecolor="#7F1D1D", linewidth=0.8, alpha=0.95, zorder=7))
    ax.annotate(
        "",
        xy=(conflict[0], conflict[1] + 0.25),
        xytext=ped,
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#D62728", linestyle=(0, (3, 2)), shrinkA=5, shrinkB=2),
        zorder=7,
    )
    ax.text(33.0, 6.0, "hidden pedestrian\n(bounded-rational)", fontsize=6.2, color="#7F1D1D", ha="left",
            va="center", linespacing=1.0)
    ax.annotate("", xy=(ped[0] + 0.5, ped[1] + 0.35), xytext=(32.8, 5.7),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#7F1D1D", alpha=0.6))

    # Occluded region behind the parked vehicle.
    eye = (ego[0] + 2.4, ego[1])
    shadow = _shadow_polygon(eye, [occluder[0] - 3.2, occluder[1] - 1.0], [occluder[0] + 3.2, occluder[1] - 1.0], 5.7)
    ax.add_patch(patches.Polygon(shadow, closed=True, facecolor="#F87171", alpha=0.16, edgecolor="none", zorder=2))
    ax.plot(
        [eye[0], occluder[0] - 3.2],
        [eye[1], occluder[1] - 1.0],
        color="#B91C1C",
        linewidth=0.7,
        linestyle=(0, (2, 2)),
        alpha=0.7,
        zorder=2,
    )
    ax.text(11.0, 6.55, r"occluded region $\mathcal{O}_t$", fontsize=6.4, color="#B91C1C",
            ha="center", va="center")
    ax.annotate("", xy=(21.5, 4.0), xytext=(13.5, 6.2),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#B91C1C", alpha=0.6))

    # Ego vehicle and conflict point.
    _draw_vehicle(ax, ego[0], ego[1], 4.4, 1.8, "#3B82F6", "#1E3A8A", zorder=6)
    ax.text(ego[0], -3.2, "ego (MIND)", fontsize=6.2, ha="center", color="#1E3A8A")
    ax.plot(*conflict, marker="*", markersize=11, color="#F5862C", markeredgecolor="#9A3412", markeredgewidth=0.5, zorder=8)
    ax.text(30.6, -3.2, r"conflict point $p_c$", fontsize=6.2, color="#9A3412", ha="center")

    if not with_paloi:
        ax.annotate(
            "visible-agent prediction is empty here:\nplan keeps speed (chain fracture)",
            xy=(19.5, -0.9),
            xytext=(13.5, -5.0),
            fontsize=6.6,
            color="#B91C1C",
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#B91C1C", shrinkB=3),
        )
    else:
        # Halo radii are kept small enough that the outer ring stays clear
        # of the "conflict point" label below the road.
        for radius, alpha in ((1.6, 0.15), (1.1, 0.26), (0.6, 0.40)):
            ax.add_patch(
                patches.Circle(conflict, radius, facecolor="#F59E0B", alpha=alpha, edgecolor="none", zorder=3)
            )
        grad = np.linspace(0.0, 1.0, 60)
        for g0, g1 in zip(grad[:-1], grad[1:]):
            x0 = 12.0 + g0 * (conflict[0] - 3.6 - 12.0)
            x1 = 12.0 + g1 * (conflict[0] - 3.6 - 12.0)
            ax.add_patch(
                patches.Rectangle(
                    (x0, -1.85),
                    x1 - x0,
                    1.9,
                    facecolor="#0B6E69",
                    alpha=0.05 + 0.30 * g1,
                    edgecolor="none",
                    zorder=2,
                )
            )
        # Label both constructs on one shared row beneath the elements they
        # describe (hinge band on the left, phantom halo on the right)
        # instead of using long leader arrows, which previously crossed the
        # hinge band, the ego-route line, and each other.
        ax.text(12.0, -4.35, r"hinge shaping $v \rightarrow v_{\mathrm{safe}}$",
                fontsize=6.6, color="#0B6E69", ha="center", va="center")
        ax.text(30.4, -4.35, "phantom source at $p_c$", fontsize=6.6, color="#92400E",
                ha="center", va="center")


def _speed_profile(ax, with_paloi: bool) -> None:
    conflict_x = 29.5
    xs = np.linspace(0, 46, 400)
    v = np.full_like(xs, 8.0)
    if not with_paloi:
        brake_start = 26.0
        for i, x in enumerate(xs):
            if x >= brake_start:
                v2 = 64.0 - 2.0 * 4.0 * (x - brake_start)
                v[i] = math.sqrt(max(v2, 0.0))
        impact_v = math.sqrt(64.0 - 8.0 * (conflict_x - brake_start))
        ax.plot(xs[xs <= conflict_x], v[xs <= conflict_x], color="#6B7280", linewidth=1.7)
        ax.plot(conflict_x, impact_v, marker="x", markersize=7, markeredgewidth=2.0, color="#B91C1C", zorder=5)
        ax.text(conflict_x + 1.2, impact_v + 0.5, f"impact {impact_v:.1f} m/s", fontsize=6.4, color="#B91C1C")
        ax.text(2.0, 6.4, "late reaction only", fontsize=6.4, color="#6B7280")
    else:
        ramp_start, ramp_end, v_safe = 12.0, 25.0, 2.5
        for i, x in enumerate(xs):
            if x < ramp_start:
                v[i] = 8.0
            elif x < ramp_end:
                s = (x - ramp_start) / (ramp_end - ramp_start)
                v[i] = 8.0 + (v_safe - 8.0) * (3 * s**2 - 2 * s**3)
            else:
                v[i] = v_safe
        ax.plot(xs, v, color="#0B6E69", linewidth=1.7)
        ax.plot(conflict_x, v_safe, marker="o", markersize=5, color="#0B6E69", zorder=5)
        ax.text(conflict_x + 1.2, 4.6, "passes at crawl speed", fontsize=6.4, color="#0B6E69")
        ax.axhline(2.5, color="#0B6E69", linewidth=0.6, linestyle=(0, (4, 3)), alpha=0.6)
        ax.text(1.0, 1.1, r"$v_{\mathrm{safe}}$", fontsize=6.4, color="#0B6E69")
    ax.axhline(3.0, color="#B45309", linewidth=0.6, linestyle=(0, (1, 2)), alpha=0.8)
    ax.text(1.0, 3.45, "3.0 m/s low-severity threshold", fontsize=5.6, color="#B45309", ha="left")
    ax.axvline(conflict_x, color="#9A3412", linewidth=0.6, linestyle=(0, (2, 2)), alpha=0.7)
    ax.text(conflict_x + 0.7, 0.3, r"$p_c$", fontsize=6.2, color="#9A3412", va="bottom")
    ax.set_xlim(0, 46)
    ax.set_ylim(0, 9.6)
    ax.set_ylabel("speed (m/s)", fontsize=6.5)
    ax.set_xlabel("position along route (m)", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.3)


def figure_concept() -> None:
    fig = plt.figure(figsize=(7.16, 3.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0])

    for col, with_paloi in ((0, False), (1, True)):
        ax_scene = fig.add_subplot(gs[0, col])
        _draw_ghost_scene(ax_scene, with_paloi)
        ax_scene.set_title(
            "(a) Prediction\u2013decision chain fracture" if not with_paloi
            else "(b) PA-LOI: phantom source and safe-speed bound",
            fontsize=8,
        )
        ax_speed = fig.add_subplot(gs[1, col])
        _speed_profile(ax_speed, with_paloi)

    fig.savefig(FIG_DIR / "concept_bounded_rationality.pdf", bbox_inches="tight")
    plt.close(fig)


def _group_box(ax, xy, width, height, label, ec, fc):
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.3,
        linestyle=(0, (5, 2)),
    )
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height - 0.035, label, ha="center", va="top", fontsize=7.6,
            color=ec, fontweight="bold")


def figure_system() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 3.1), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    gray, teal, orange, green, blue = "#46505E", "#0B6E69", "#B45309", "#146C43", "#0B3B8F"

    # Inputs.
    draw_box(ax, (0.015, 0.66), 0.155, 0.21, "AV2 replay\nmaps and agents\n(obs @ 10 Hz)", fontsize=7)
    draw_box(ax, (0.015, 0.20), 0.155, 0.21, "JSONL-fixed\nghost geometry\n$(p_a, p_c, d_{trig})$", fontsize=7)

    # MIND core group.
    _group_box(ax, (0.215, 0.52), 0.345, 0.42, "MIND integrated planner (unmodified)", blue, "#F6F9FF")
    draw_box(ax, (0.228, 0.565), 0.095, 0.23, "AIME\nmultimodal\nprediction", fontsize=6.6)
    draw_box(ax, (0.338, 0.565), 0.095, 0.23, "Scenario tree\n(modality\nbranching)", fontsize=6.6)
    draw_box(ax, (0.448, 0.565), 0.097, 0.23, "Trajectory-tree\niLQR\n(contingency)", fontsize=6.6)
    arrow(ax, (0.323, 0.68), (0.338, 0.68), lw=1.1)
    arrow(ax, (0.433, 0.68), (0.448, 0.68), lw=1.1)

    # PA-LOI group.
    _group_box(ax, (0.215, 0.045), 0.345, 0.40, "PA-LOI risk layer (this work)", teal, "#F0FBF9")
    draw_box(ax, (0.228, 0.09), 0.095, 0.21, "Occlusion\nscreening\npipeline", fc="#E4F6F2", ec=teal, fontsize=6.6)
    draw_box(ax, (0.338, 0.09), 0.095, 0.21, "TTA weight\n$w(t_a)$ and\n$v_{\\mathrm{safe}}$", fc="#E4F6F2", ec=teal, fontsize=6.6)
    draw_box(ax, (0.448, 0.09), 0.097, 0.21, "Velocity-hinge\npotential\n$\\Phi(p, v)$", fc="#E4F6F2", ec=teal, fontsize=6.6)
    arrow(ax, (0.323, 0.195), (0.338, 0.195), color=teal, lw=1.1)
    arrow(ax, (0.433, 0.195), (0.448, 0.195), color=teal, lw=1.1)
    # Keep the injection arrow to the right of the group-box title text so
    # it never crosses "(this work)", and put its label horizontally in the
    # empty corridor between the two dashed group boxes.
    arrow(ax, (0.532, 0.30), (0.532, 0.565), color=teal, lw=1.5)
    ax.text(0.539, 0.483, "risk cost", fontsize=6.5, color=teal, ha="left", va="center")

    # Safety column. Narrower boxes than the output column so the
    # "control" edge label fits inside the inter-column gap without
    # touching either box border.
    draw_box(ax, (0.615, 0.56), 0.135, 0.26, "AEB assessor\nTTC + RSS gap,\nlatched full brake", fc="#FFF7ED", ec=orange, fontsize=6.8)
    draw_box(ax, (0.615, 0.16), 0.135, 0.22, "Kinematic clamp\n$a \\geq -v/\\Delta t$", fc="#FFF7ED", ec=orange, fontsize=6.8)
    arrow(ax, (0.6825, 0.56), (0.6825, 0.38), color=orange, lw=1.2)

    # Output column.
    draw_box(ax, (0.840, 0.56), 0.148, 0.26, "Closed-loop sim\n50 Hz, 650 frames\ncollision check", fc="#F0FDF4", ec=green, fontsize=6.8)
    draw_box(ax, (0.840, 0.16), 0.148, 0.22, "Run records\nTSV/JSONL\n$\\rightarrow$ tables, figures", fc="#F0FDF4", ec=green, fontsize=6.8)

    # Main flow arrows.
    arrow(ax, (0.17, 0.765), (0.215, 0.73))
    arrow(ax, (0.17, 0.305), (0.228, 0.24), color=teal)
    # Leave from the bottom edge of the AV2 box (its visual border includes
    # a pad of 0.015) so no segment of this arc is drawn on the box face,
    # and bow gently rightwards through the gap between the JSONL box and
    # the two dashed group containers.
    arrow(ax, (0.132, 0.650), (0.228, 0.30), color=teal, connectionstyle="arc3,rad=-0.08")
    arrow(ax, (0.56, 0.73), (0.615, 0.71), lw=1.4)
    # Keep edge labels narrow enough to sit inside the inter-box gaps
    # without touching the box borders on either side.
    ax.text(0.5875, 0.765, "plan", fontsize=6.8, color="#0B3B8F", ha="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))
    # Command path runs AEB assessor -> kinematic clamp -> simulator, so the
    # clamp sits visibly in series on the control edge rather than dead-ending.
    arrow(ax, (0.750, 0.30), (0.840, 0.60), color=orange, lw=1.4,
          connectionstyle="arc3,rad=0.2")
    ax.text(0.773, 0.455, "control", fontsize=6.5, color=orange, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
    arrow(ax, (0.914, 0.56), (0.914, 0.38), color=green, lw=1.2)
    # Route the feedback arc above every box so it never crosses a module;
    # a flatter arc (rad 0.13) keeps the apex below the caption text, and
    # the white text bbox guarantees the arc can never strike the letters.
    arrow(ax, (0.914, 0.843), (0.0925, 0.89), color="#94A3B8", lw=1.0,
          connectionstyle="arc3,rad=0.13")
    ax.text(0.50, 1.04, "closed loop: updated ego state re-enters observation", fontsize=7.0,
            color="#475569", ha="center", va="top", clip_on=False, zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))

    # Legend.
    legend_items = [
        ("#F8FAFC", blue, "existing MIND stack"),
        ("#E4F6F2", teal, "PA-LOI additions"),
        ("#FFF7ED", orange, "safety fallback"),
        ("#F0FDF4", green, "evaluation artifacts"),
    ]
    x0 = 0.225
    for fc, ec, label in legend_items:
        ax.add_patch(patches.Rectangle((x0, -0.035), 0.018, 0.05, facecolor=fc, edgecolor=ec, linewidth=1.0,
                                       clip_on=False))
        ax.text(x0 + 0.024, -0.01, label, fontsize=6.4, va="center")
        x0 += 0.045 + len(label) * 0.0058
    fig.savefig(FIG_DIR / "system_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_risk_profiles() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.9), constrained_layout=True)

    # Opaque white legend boxes so curves and reference lines can never
    # show through or collide with the legend entries.
    legend_kw = dict(
        fontsize=5.8,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
        fancybox=False,
        handlelength=1.3,
        handletextpad=0.4,
        labelspacing=0.3,
        borderpad=0.3,
        borderaxespad=0.25,
    )

    # (a) Lateral sigmoid gate.
    ax = axes[0, 0]
    ell = np.linspace(0, 6, 300)
    for d_perp, color in ((1.0, "#94D2C8"), (2.0, "#3E9C92"), (3.0, "#0B6E69")):
        gate = 1.0 / (1.0 + np.exp(2.0 * (ell - d_perp)))
        ax.plot(ell, gate, color=color, linewidth=1.5, label=f"$d_\\perp$={d_perp:.0f} m")
    ax.set_xlabel(r"clearance $\ell$ (m)", fontsize=7)
    ax.set_ylabel(r"$S(\ell)$", fontsize=7)
    ax.legend(loc="upper right", **legend_kw)
    ax.set_title("(a) Lateral gate", fontsize=7.5)

    # (b) Hinge velocity cost for ramping activation weights.
    ax = axes[0, 1]
    v = np.linspace(0, 9, 300)
    v_safe = 2.5
    for t_a, color in ((6.0, "#F3C683"), (4.0, "#E29A3C"), (2.0, "#B45309")):
        w = 25.0 * min(max((6.5 - t_a) / 4.5, 0.0), 1.0)
        cost = w * np.maximum(v - v_safe, 0.0) ** 2
        ax.plot(v, cost, color=color, linewidth=1.5, label=f"$t_a$={t_a:.0f} s")
    # Stop the reference line at mid-height so it cannot run through the
    # legend block in the upper-left corner.
    ax.axvline(v_safe, ymax=0.50, color="#0B6E69", linewidth=0.7, linestyle=(0, (4, 3)))
    # Keep the line label near the axis, well below the legend block.
    ax.text(v_safe + 0.18, 130, r"$v_{\mathrm{safe}}$", fontsize=6.2, color="#0B6E69")
    ax.set_xlabel("ego speed $v$ (m/s)", fontsize=7)
    ax.set_ylabel(r"$\Phi$ at $S{=}1$", fontsize=7)
    ax.set_xticks([0, 3, 6, 9])
    ax.legend(loc="upper left", **legend_kw)
    ax.set_title("(b) Hinge cost", fontsize=7.5)

    # (c) Strict stopping-feasibility bound.
    ax = axes[1, 0]
    d_s = np.linspace(0, 30, 300)
    a_b, tau_r, delta = 4.0, 0.2, 0.5
    v_strict = np.maximum(0.0, -a_b * tau_r + np.sqrt((a_b * tau_r) ** 2 + 2 * a_b * np.maximum(d_s - delta, 0.0)))
    ax.plot(d_s, v_strict, color="#0B3B8F", linewidth=1.6)
    ax.set_xlabel(r"distance to conflict $d_s$ (m)", fontsize=7)
    ax.set_ylabel(r"$v_{\mathrm{safe}}(d_s)$ (m/s)", fontsize=7)
    ax.set_title("(c) Strict bound", fontsize=7.5)

    # (d) Deployed crossing-time instantiation.
    ax = axes[1, 1]
    v = np.linspace(0, 10, 300)
    for d_perp, color in ((1.0, "#94D2C8"), (2.0, "#3E9C92"), (3.0, "#0B6E69")):
        t_cross = d_perp / 2.0
        bound = np.maximum.reduce([np.minimum(4.0 * t_cross, v), 0.6 * v, np.full_like(v, 2.0)])
        ax.plot(v, bound, color=color, linewidth=1.5, label=f"$d_\\perp$={d_perp:.0f} m")
    ax.plot(v, v, color="#9CA3AF", linewidth=0.9, linestyle=(0, (4, 3)))
    # Keep the diagonal tag fully inside the axes frame: anchored lower along
    # the reference line so the rotated text can never cross the top spine.
    ax.text(6.45, 7.30, "$v_{\\mathrm{safe}}{=}v$", fontsize=6.0, color="#6B7280", rotation=38)
    ax.axhline(2.0, color="#B45309", linewidth=0.7, linestyle=(0, (1, 2)))
    # Anchor the label by its top edge well below the floor line so the
    # text can never touch the dotted line or the flat curve segments.
    ax.text(0.3, 1.55, r"$v_{\mathrm{floor}}$", fontsize=6.2, color="#B45309", va="top")
    ax.set_xlabel("current speed $v$ (m/s)", fontsize=7)
    ax.set_ylabel(r"deployed $v_{\mathrm{safe}}$ (m/s)", fontsize=7)
    ax.legend(loc="upper left", **legend_kw)
    ax.set_title("(d) Deployed bound", fontsize=7.5)

    for ax in axes.ravel():
        ax.tick_params(labelsize=6.5)
        ax.grid(True, linewidth=0.3, alpha=0.3)
    fig.savefig(FIG_DIR / "risk_field_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_scenario_overview(clean81_rows: list[dict]) -> None:
    candidates = [
        ROOT / "实验场景" / "22场景_锁定鬼探头_late5.6s" / "场景01.png",
        ROOT / "实验场景" / "22场景_锁定鬼探头_late5.6s" / "场景06.png",
        ROOT / "实验场景" / "20260606_new90_le3_preview" / "new90_le3_01.png",
        ROOT / "实验场景" / "20260606_new90_le3_preview" / "new90_le3_18.png",
        ROOT / "实验场景" / "20260606_new90_le3_preview" / "new90_le3_43.png",
        ROOT / "实验场景" / "20260606_new90_le3_preview" / "new90_le3_81.png",
        ROOT / "实验场景" / "20260606_new90_preview" / "new90_01.png",
        ROOT / "实验场景" / "20260606_new90_preview" / "new90_35.png",
    ]
    paths = [p for p in candidates if p.exists()][:6]
    if len(paths) >= 4:
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.25), constrained_layout=True)
        for ax, path, label in zip(axes.ravel(), paths, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
            img = mpimg.imread(path)
            h, w = img.shape[0], img.shape[1]
            # Crop the central viewport: removes the per-panel title, tick
            # labels, and in-panel legend while keeping the conflict geometry.
            x0, x1 = int(0.18 * w), int(0.80 * w)
            y0, y1 = int(0.16 * h), int(0.78 * h)
            ax.imshow(img[y0:y1, x0:x1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_edgecolor("#CBD5E1")
        handles = [
            mlines.Line2D([], [], color="#1f5ad6", linewidth=2.0, label="Ego route"),
            mlines.Line2D([], [], color="#8b46e0", marker="D", linestyle="None", markersize=5, label="Trigger point"),
            mlines.Line2D([], [], color="#f5862c", marker="*", linestyle="None", markersize=9, label="Lane-center conflict point"),
            mlines.Line2D([], [], color="#2ca02c", marker="o", linestyle="None", markersize=5, label="Hidden ghost start"),
            mlines.Line2D([], [], color="#d62728", linewidth=2.0, label="Ghost dash-out"),
            mlines.Line2D([], [], color="#7c3aed", linewidth=1.2, linestyle=":", label="Occluded line-of-sight"),
            patches.Patch(facecolor="#aab4c4", edgecolor="#2b3a55", label="Occluder"),
            patches.Patch(facecolor="#d4d4d8", edgecolor="#9ca3af", label="Background vehicles"),
            patches.Patch(facecolor="#fde68a", alpha=0.55, edgecolor="none", label="Pedestrian crossing"),
        ]
        # Anchor the legend fully below y=0 so it can never intrude into the
        # bottom panel row: constrained_layout does not reserve space for a
        # figure-level legend, and bbox_inches="tight" extends the canvas to
        # include it, so the legend must clear the axes region on its own.
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7.5,
                   bbox_to_anchor=(0.5, -0.105), handletextpad=0.5, columnspacing=1.0)
        # dpi controls only the embedded raster panels (crops of the 1280x1280
        # preview renders); all overlaid vector elements stay vector.
        fig.savefig(FIG_DIR / "scenario_overview.pdf", bbox_inches="tight", dpi=350)
        plt.close(fig)
        return

    frames = [r.get("estimated_trigger_frame_50hz") for r in clean81_rows if r.get("estimated_trigger_frame_50hz")]
    lats = [r.get("ghost_lateral_distance_m") for r in clean81_rows if r.get("ghost_lateral_distance_m")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    axes[0].hist([float(x) for x in frames], bins=14, color="#0B6E69", alpha=0.85)
    axes[0].set_xlabel("Estimated trigger frame")
    axes[0].set_ylabel("Scenes")
    axes[0].set_title("Trigger timing")
    axes[1].hist([float(x) for x in lats], bins=12, color="#D8841C", alpha=0.85)
    axes[1].set_xlabel("Hidden-to-lane lateral distance (m)")
    axes[1].set_title("Occlusion geometry")
    fig.savefig(FIG_DIR / "scenario_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_safety_by_distance(ghost_rows: list[dict[str, str]]) -> None:
    by_dist = summarize_by_distance(ghost_rows)
    fig, ax = plt.subplots(figsize=(3.45, 2.4), constrained_layout=True)
    for baseline in BASELINES:
        y = [pct(by_dist[(baseline, d)]["paper_safe"], by_dist[(baseline, d)]["n"]) for d in DISTANCES]
        ax.plot(DISTANCES, y, marker=BASELINE_MARKERS[baseline], markersize=4, linewidth=1.8,
                linestyle=BASELINE_LINESTYLES[baseline],
                color=BASELINE_COLORS[baseline], label=BASELINE_LABELS[baseline])
    ax.set_xlabel("Trigger distance (m)")
    ax.set_ylabel("Low-severity rate (%)")
    ax.set_xticks(DISTANCES)
    ax.set_ylim(0, 100)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="upper left")
    fig.savefig(FIG_DIR / "safety_by_distance.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_impact_energy(ghost_rows: list[dict[str, str]]) -> None:
    main = summarize_main(ghost_rows)
    labels = [BASELINE_LABELS[b] for b in BASELINES]
    x = np.arange(len(BASELINES))
    impact = [main[b]["mean_impact"] for b in BASELINES]
    energy = [main[b]["mean_impact_v2"] for b in BASELINES]

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.15), constrained_layout=True)
    for ax, values, ylabel in (
        (axes[0], impact, "Mean impact speed (m/s)"),
        (axes[1], energy, r"Mean $v_{\mathrm{imp}}^2$ (m$^2$/s$^2$)"),
    ):
        bars = ax.bar(x, values, color=[BASELINE_COLORS[b] for b in BASELINES], width=0.62)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    fig.savefig(FIG_DIR / "impact_energy.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_noghost(noghost_rows: list[dict[str, str]]) -> None:
    summary = summarize_noghost(noghost_rows)
    labels = [NOGHOST_FIG_LABELS[b] for b in NOGHOST_ORDER]
    colors = [BASELINE_COLORS[b] for b in NOGHOST_ORDER]
    x = np.arange(len(NOGHOST_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.3), constrained_layout=True)
    bars = axes[0].bar(x, [summary[b]["avg_speed"] for b in NOGHOST_ORDER], color=colors, width=0.66)
    axes[0].bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
    axes[0].set_ylabel("Mean speed (m/s)", fontsize=7.5)
    axes[0].set_ylim(0, summary["mind"]["avg_speed"] * 1.25)
    axes[0].grid(axis="y", linewidth=0.4, alpha=0.35)

    fail = [summary[b]["planner_fail"] for b in NOGHOST_ORDER]
    coll = [summary[b]["gt_collision"] for b in NOGHOST_ORDER]
    width = 0.38
    b1 = axes[1].bar(x - width / 2, fail, width=width, color="#D8841C", label="Planner failures")
    b2 = axes[1].bar(x + width / 2, coll, width=width, color="#991B1B", label="Replay collisions")
    axes[1].bar_label(b1, fontsize=6.5, padding=1)
    axes[1].bar_label(b2, fontsize=6.5, padding=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
    axes[1].set_ylabel("Count", fontsize=7.5)
    axes[1].set_ylim(0, max(coll) * 1.6)
    axes[1].legend(fontsize=6, loc="upper right")
    axes[1].grid(axis="y", linewidth=0.4, alpha=0.35)
    fig.savefig(FIG_DIR / "noghost_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_speed_sensitivity(speed_rows: list[dict[str, str]]) -> None:
    summary = summarize_speed(speed_rows)
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.15), constrained_layout=True)
    for baseline in BASELINES:
        y_safe = [pct(summary[(s, baseline)]["paper_safe"], summary[(s, baseline)]["n"]) for s in SPEEDS]
        y_v2 = [summary[(s, baseline)]["mean_impact_v2"] for s in SPEEDS]
        axes[0].plot(SPEEDS, y_safe, marker=BASELINE_MARKERS[baseline], markersize=3.5, linewidth=1.6,
                     linestyle=BASELINE_LINESTYLES[baseline],
                     color=BASELINE_COLORS[baseline], label=BASELINE_LABELS[baseline])
        axes[1].plot(SPEEDS, y_v2, marker=BASELINE_MARKERS[baseline], markersize=3.5, linewidth=1.6,
                     linestyle=BASELINE_LINESTYLES[baseline],
                     color=BASELINE_COLORS[baseline], label=BASELINE_LABELS[baseline])
    for ax in axes:
        ax.set_xlabel("Ped. speed (m/s)", fontsize=7.5)
        ax.set_xticks(SPEEDS)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.4, alpha=0.35)
    axes[0].set_ylabel("Low-severity rate (%)", fontsize=7.5)
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel(r"Mean $v_{\mathrm{imp}}^2$ (m$^2$/s$^2$)", fontsize=7.5)
    # Honest zero baseline for the ratio-scale energy proxy.
    axes[1].set_ylim(bottom=0)
    # Anchor the legend inside the empty band between the MIND (top) and
    # AEB-only (middle) curves so no curve passes behind it.
    axes[1].legend(fontsize=6.5, loc="center", bbox_to_anchor=(0.47, 0.62),
                   bbox_transform=axes[1].transAxes)
    fig.savefig(FIG_DIR / "speed_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_tradeoff(ghost_rows: list[dict[str, str]], noghost_rows: list[dict[str, str]]) -> None:
    noghost = summarize_noghost(noghost_rows)
    mind_speed = noghost["mind"]["avg_speed"]
    points = {
        baseline: (pct(mind_speed - noghost[baseline]["avg_speed"], mind_speed), MATCHED_LOW_SEVERITY[baseline])
        for baseline in ["mind", "reachset", "shadow", "ours"]
    }
    fig, ax = plt.subplots(figsize=(3.45, 2.5), constrained_layout=True)
    # The two occlusion bounds are near-coincident: draw Dynamic-shadow as an
    # open square over the filled Reachable-set dot so both stay visible.
    for baseline in ("mind", "reachset", "ours"):
        loss, safe = points[baseline]
        ax.scatter(loss, safe, s=80, color=BASELINE_COLORS[baseline], zorder=3)
    ax.scatter(*points["shadow"], s=95, marker="s", facecolors="none",
               edgecolors=BASELINE_COLORS["shadow"], linewidths=1.5, zorder=4)
    ax.annotate("MIND\n(0.0%, 6.2%)", points["mind"], textcoords="offset points",
                xytext=(8, -1), ha="left", fontsize=7)
    ax.annotate("PA-LOI + AEB\n(27.0%, 58.5%)", points["ours"], textcoords="offset points",
                xytext=(9, 0), ha="left", fontsize=7)
    ax.annotate("Dynamic-shadow\n(49.2%, 34.9%)", points["shadow"], textcoords="offset points",
                xytext=(-9, 7), ha="right", va="bottom", fontsize=7)
    ax.annotate("Reachable-set\n(49.2%, 34.4%)", points["reachset"], textcoords="offset points",
                xytext=(-9, -8), ha="right", va="top", fontsize=7)
    # Keep the guidance annotation fully inside the axes so no glyph is
    # clipped by the top spine.
    ax.annotate(
        "safer,\nmore efficient",
        xy=(4, 66),
        xytext=(20, 55),
        fontsize=6.5,
        color="#475569",
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#94A3B8"),
    )
    ax.set_xlabel("No-ghost speed loss vs. MIND (%)")
    ax.set_ylabel("Matched-cell low-severity rate (%)")
    ax.set_xlim(-4, 58)
    ax.set_ylim(0, 75)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.savefig(FIG_DIR / "safety_efficiency_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def write_compliance_note() -> None:
    note = """# T-ITS Compliance Notes

- Format target: IEEE double-column, single-spaced journal style.
- Paper type: Regular Paper. The T-ITS suggested length is 10 pages, with up to 6 additional pages allowed.
- Current draft target: 15 pages, leaving one page below the 16-page submission cap.
- Abstract target: 150--250 words, one paragraph, no citations, equations, or tables.
- Keywords: 6 total, comprising 2 methodology terms, 2 application terms,
  and 2 optional free terms under the T-ITS taxonomy.
- Conference extension disclosure: the manuscript cites the ITSC conference version and lists the new theory, experiments, and analysis added here.
- Supplementary material: Tables S1--S3 hold detailed literature positioning,
  extended-distance results, and parameter perturbations.
- Reproducibility archive: staged under `submission_artifacts/` with SHA-256
  checksums and 2,106 per-run summary records (1,134 baseline + 972 sweep).

Official source checked: https://ieee-itss.org/pub/t-its/
"""
    write_file(PAPER_DIR / "tits_compliance_notes.md", note)


def main() -> None:
    ensure_dirs()
    ghost_rows = read_tsv(GHOST_TSV)
    primary_noghost_rows = read_tsv(NOGHOST_TSV)
    noghost_rows = primary_noghost_rows + read_extra_noghost_rows()
    speed_rows = read_tsv(SPEED_TSV)
    clean81_rows = read_jsonl(CLEAN81_JSONL)
    rep30_rows = read_jsonl(REP30_JSONL)
    validate_inputs(ghost_rows, primary_noghost_rows, speed_rows, clean81_rows, rep30_rows)
    validate_noghost_all(noghost_rows)

    write_tables(ghost_rows, noghost_rows, speed_rows, clean81_rows, rep30_rows)
    figure_concept()
    figure_system()
    figure_risk_profiles()
    figure_scenario_overview(clean81_rows)
    figure_safety_by_distance(ghost_rows)
    figure_impact_energy(ghost_rows)
    figure_noghost(noghost_rows)
    figure_speed_sensitivity(speed_rows)
    figure_tradeoff(ghost_rows, noghost_rows)
    write_compliance_note()

    print("Generated figures:")
    for path in sorted(FIG_DIR.glob("*.pdf")):
        print(f"  {path.relative_to(ROOT)}")
    print("Generated tables:")
    for path in sorted(TAB_DIR.glob("*.tex")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
