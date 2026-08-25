# PA-LOI: A Proactive Risk Layer for Occluded Pedestrian Emergence

[![Reproducibility archive](https://img.shields.io/badge/artifacts-checksummed-2ea44f)](MANIFEST.tsv)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

This repository contains the code, fixed hazard geometry, per-run records, and
analysis scripts accompanying the manuscript **PA-LOI: A Proactive Risk Layer
for Occluded Pedestrian Emergence**.

PA-LOI adds an occlusion-aware velocity cost to the optimization stage of MIND
while preserving MIND's predictor and scenario-tree generator. The associated
**clean81** benchmark fixes hidden-pedestrian geometry across systems and reports
hazard spawn rate, contact severity, zero collision, replay collisions, planner
failures, and no-pedestrian efficiency separately.

## What is included

- 4,050 archived closed-loop runs across five systems and sensitivity studies.
- JSONL-fixed geometry for clean81 and the all-candidate curation audit.
- PA-LOI, AEB, Reachable-set, and Dynamic-shadow supervisory implementations.
- Every aggregation, validation, statistics, and figure-generation script used
  for the manuscript results.
- A SHA-256 manifest and an archive-to-workspace restore map.
- The manuscript and supplementary PDF, plus their LaTeX sources.

The study is a controlled AV2 replay evaluation. It does not make an on-road
deployment or calibrated pedestrian-behavior claim.

## Fast audit from archived records

The aggregate tables can be regenerated without rerunning the planner. From the
repository root:

```bash
python3 verify_archive.py
python3 restore_workspace.py --destination restored_workspace
cd restored_workspace
python3 paper/tits_pa_loi/scripts/generate_sweep_tables.py
python3 paper/tits_pa_loi/scripts/compute_statistics.py
python3 paper/tits_pa_loi/scripts/compute_param_paired_stats.py
python3 paper/tits_pa_loi/scripts/compute_candidate_pool_audit.py
```

The scripts require Python 3.10 and the numerical packages listed in
`code/requirements.txt`. New closed-loop runs additionally require the official
Argoverse 2 scenario files and the upstream MIND runtime/model weights. Raw AV2
data and the 50 MB checkpoint are not redistributed; scenario identifiers and
the exact checkpoint digest are included.

## Repository layout

- `geometry/` — fixed hazard geometry and candidate sets.
- `records/` — per-run records for primary, baseline, sweep, mechanism, and
  curation-audit analyses.
- `code/` — implementations, experiment drivers, configurations, and analyses.
- `paper/` — manuscript/supplement PDFs and LaTeX source.
- `MANIFEST.tsv` — byte size and SHA-256 for every archived file.
- `RESTORE_MAP.tsv` / `restore_workspace.py` — restore paths expected by scripts.
- `MODEL_CHECKPOINT.sha256` — exact upstream checkpoint identifier.

## Upstream project and data

PA-LOI is implemented on top of
[MIND](https://github.com/HKUST-Aerial-Robotics/MIND). Argoverse 2 data are
available from the [official Argoverse release](https://www.argoverse.org/av2.html).
Please follow the upstream projects' terms when obtaining their assets.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite the
manuscript and acknowledge MIND and Argoverse 2 when using these artifacts.

## License

The released code is provided under GPL-3.0; see [`LICENSE`](LICENSE). Dataset
files remain subject to the Argoverse terms.
