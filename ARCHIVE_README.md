# PA-LOI / clean81 reproducibility archive

This archive accompanies the T-ITS manuscript *PA-LOI: A Proactive Risk Layer
for Occluded Pedestrian Emergence*.
It contains the fixed hazard geometry, the per-run records used by every
reported table and figure, and the corresponding aggregation and validation
code.

## Layout

- `geometry/`: clean81 injection geometry, the walking-speed subset, and the
  90-candidate / selected-new60 geometry used by the curation audit.
- `records/primary/`: 1,458 main ghost-probe rows, strict-650 records,
  162 no-pedestrian rows, and the original walking-speed records.
- `records/curation_audit/`: the 88-scene strict-650 screening records and the
  machine-readable audit plus 90-scene membership table behind Supplementary
  Table S4.
- `records/baseline_runs/`: 1,134 summaries for the Reachable-set and
  Dynamic-shadow comparisons.
- `records/sweeps/`: 972 summaries, the 27-scene geometry subset, and the
  manifest for the parameter and extended-trigger-distance experiments.
- `records/mechanism_case/`: per-planning-cycle traces for the single-scene
  mechanism figure.
- `code/`: PA-LOI implementation, both supervisor implementations, experiment
  drivers, aggregation, validation, statistics, and plotting code.
- `MANIFEST.tsv`: relative path, byte size, and SHA-256 digest for every file.
- `RESTORE_MAP.tsv`: archive path to original repository-relative path.
- `restore_workspace.py`: reconstructs the expected paths in a MIND checkout.
- `MODEL_CHECKPOINT.sha256`: exact identifier of the upstream MIND checkpoint.
- `LICENSE`: release terms.

For portability and privacy, the machine-specific interpreter path in the
executed E1/E2 queue manifest is normalized to `python3`; all task identifiers,
arguments, environment overrides, output paths, and recorded outcomes are
otherwise preserved.

## Archive-only aggregate checks

First reconstruct the recorded paths in a clean MIND checkout, then run the
checks from that checkout:

```bash
python3 restore_workspace.py --destination /path/to/MIND
cd /path/to/MIND
python3 paper/tits_pa_loi/scripts/generate_sweep_tables.py
python3 paper/tits_pa_loi/scripts/compute_statistics.py
python3 paper/tits_pa_loi/scripts/compute_param_paired_stats.py
python3 paper/tits_pa_loi/scripts/compute_candidate_pool_audit.py
```

`validate_results.py`, the scene-rendering portions of
`generate_figures_tables.py`, and new closed-loop runs additionally require the
official Argoverse 2 scenario files and the upstream MIND runtime/model weights.
The archived `requirements.txt` and planner/simulator configuration capture the
software setup. The 50 MB MIND checkpoint is not duplicated in this compact
archive; its expected repository path and SHA-256 digest are recorded in
`MODEL_CHECKPOINT.sha256` and it is tracked by the public MIND-PA-LOI repository.

The execution-time MIND dependencies and model weights follow the upstream
MIND project. The archived records are sufficient to regenerate and audit the
paper's aggregate results without rerunning the planner.

## Argoverse 2 data

Raw Argoverse 2 data are not redistributed. Scenarios are referenced by their
official identifiers; obtain the dataset from the official Argoverse release.
