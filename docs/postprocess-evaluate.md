# Postprocess + per-lesion evaluation

Two CLIs that turn a finished segmentation run directory into a postprocessed
prediction set and a per-3D-lesion detection score, with an interactive 3D
Plotly HTML per case.

## When to run this

Run after `mri.cli.diagnose` (or after a fresh training run) when you want to
answer:

- "Did the model find each lesion that was actually there?"
- "Is the model staying quiet on healthy cases?"

The voxel-level Dice / precision / recall in the diagnostic report don't
answer either of those questions on their own.

## Pipeline

```
mri.cli.diagnose  →  mri.cli.postprocess  →  mri.cli.evaluate
```

`postprocess` will auto-run the dump step (the same one diagnose uses) if
the prediction cache is missing, so in practice you can skip diagnose if you
only care about the postprocessed evaluation.

## Step 1 — Postprocess

```bash
uv run python -m mri.cli.postprocess <run_dir>
```

Applies two voxel-wise rules:

1. **Target outside prostate is ignored** — `lesion_mask &= gland_mask`.
2. **No prostate ⇒ no target** — when no gland voxel passes its threshold,
   the lesion mask is fully zeroed.

Both use the *predicted* gland (not GT), so the rules mirror what would
happen at deployment.

Output: `<run_dir>/diagnostic/postprocessed/<case>/{lesion_mask.npz, gland_mask.npz, meta.json}`.

Useful flags:

- `--lesion-threshold 0.4 --gland-threshold 0.3` — override the thresholds
  (default: `metrics.segmentation_threshold` from the resolved config).
- `--force` — regenerate `postprocessed/` even if it already exists (re-run
  postprocessing under different thresholds without re-running inference).

## Step 2 — Evaluate

```bash
uv run python -m mri.cli.evaluate <run_dir>
```

For each case:

- **Positive case (≥ 1 GT lesion).** GT is split into 3D-connected
  components. Each component is scored by the *max* IoU across the slices
  it spans; a component is detected iff that max IoU > `--correctness-iou`
  (default 0.1). Detection is per-lesion, not per-case: a 2-lesion case
  with one hit and one miss contributes 1/2 to lesion-level recall.
- **Negative case (no GT lesion).** Correct iff the postprocessed
  prediction covers ≤ `--negative-area-frac` (default 0.02 = 2%) of *every*
  slice.

Outputs under `<run_dir>/diagnostic/evaluation/`:

- `metrics_by_lesion.csv` — one row per 3D GT lesion in the cohort.
- `metrics_by_case.csv` — one row per case (positive or negative).
- `summary.json` — cohort lesion-recall, negative-case accuracy, the
  parameters used, bootstrap CIs on every metric, and the official
  `picai_eval` block (see below).
- `visuals/<case>.html` — interactive 3D Plotly figure per case (toggleable
  per-component, color-coded by detection verdict).
- `visuals/index.html` — gallery linking to all per-case HTMLs.

Useful flags:

- `--correctness-iou 0.2` — tighten the detection bar.
- `--negative-area-frac 0.01` — tighten the false-alarm bar.
- `--connectivity 26` — 26-connectivity in 3D for GT CC labeling
  (default 6).
- `--visualize-only failed` — only render HTMLs for cases with any missed
  lesion or any negative-case false alarm. Use `none` to skip rendering
  entirely.
- `--downsample-vis 2` — downsample voxel grids 2x along each axis before
  isosurface extraction; cuts HTML size for large volumes.
- `--plotly-cdn` — load Plotly from CDN instead of inlining ~3 MB per HTML.

## Official picai_eval scoring + bootstrap CIs

By default, evaluate additionally scores the cohort with DIAGNijmegen's
official [`picai_eval`](https://github.com/DIAGNijmegen/picai_eval) package —
the primary metrics of the benchmark evaluation protocol. The postprocessed
lesion components are converted to the detection-map form `picai_eval`
expects (non-overlapping 3D connected components, each filled with a single
confidence = the max `prob.npz` lesion probability inside the component),
then scored for:

- **lesion AP** (average precision, IoU ≥ 0.10 matching by default),
- **case AUROC** (case score = max of the detection map, the picai_eval
  default),
- **ranking score** ((AP + AUROC) / 2, PI-CAI's leaderboard metric).

Every reported metric — these three plus the fixed-threshold lesion recall
and negative accuracy — carries a case-level percentile **bootstrap 95% CI**
(default 1000 resamples, seed 42). Degenerate resamples where a metric is
undefined (e.g. a single-class draw) are dropped; each CI records the
`n_valid` replicates its percentiles were taken over. All of it lands in
`summary.json`: CIs next to their metrics, the picai metrics under the
`picai_eval` key.

Cases without `predictions/<case>/prob.npz` are skipped for picai_eval
scoring (they carry no confidences) and listed in the block's
`cases_skipped`.

Flags:

- `--no-picai-eval` — skip the official scoring entirely.
- `--picai-min-overlap 0.10` — the lesion-matching minimum IoU.
- `--bootstrap-iters 1000` / `--bootstrap-seed 42` — resampling control;
  `--bootstrap-iters 0` disables all CIs.
- `--save-detection-maps` — persist the per-case detection maps under
  `evaluation/detection_maps/<case>.npz` (key `detection_map`), e.g. to
  re-score externally with the `picai_eval` CLI.
