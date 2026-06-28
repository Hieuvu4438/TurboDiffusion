# Experiments — Face Identity Preservation & Restoration

This package is the **research layer** built on top of the official
`turbodiffusion` library. It runs a 4-stage pipeline that takes face images,
degrades them, restores them to frontal-view videos with TurboDiffusion's
Wan2.2 I2V model, and evaluates identity preservation via face-embedding cosine
similarity (InsightFace and DeepFace backends).

The official `turbodiffusion/` package is **not** modified by anything here.

---

## Pipeline

```
 degradation  ─►  restoration  ─►  evaluation  ─►  analysis
 (blur/jpeg/     (TurboDiffusion   (cosine sim of     (aggregate
  motion/        Wan2.2 I2V)        face embeddings)   metrics)
  low-light…)
```

| Stage | Location | What it does |
|---|---|---|
| **Degradation** | `pipeline/degradation/`, `degradation/` (lib) | Apply blur / JPEG / salt-pepper / motion / low-light degradations to face images and build CSV manifests. |
| **Restoration** | `pipeline/restoration/` | `run_face_restoration.py` (engine) drives `turbodiffusion/inference/wan2.2_i2v_infer.py`; `run_all_*` runners batch over degradation types. |
| **Evaluation** | `pipeline/evaluation/` | Identity-preservation metrics on restored videos (`*_v2*`, `*_combined*`) and on degraded inputs (`run_cosine_similarity_for_degraded_input*`). |
| **Analysis** | `pipeline/analysis/` | Aggregate per-(view, noise, level) reports into summary CSVs. |
| **CSV / data prep** | `pipeline/csv_generation/`, `pipeline/data_prep/`, `pipeline/data_utils/` | Build the CSV manifests the pipeline consumes. |

## Package layout

```
experiments/
├── config.py              # ← all paths, thresholds, model defaults (single source of truth)
├── faceid/                # shared, de-duplicated utilities
│   ├── metrics.py                # cosine_similarity, classification metrics, threshold search
│   ├── preprocessing.py          # CLAHE / unsharp / median / progressive-padding fallbacks
│   ├── embeddings_insightface.py # get_embedding_robust, get_image_embedding, video best-frame  (needs insightface)
│   ├── embeddings_deepface.py    # DeepFace equivalents                                            (needs deepface)
│   ├── video.py                  # sharpness, yaw estimate, multi-criteria face scoring
│   ├── io.py                     # CSV + person-name helpers
│   └── reporting.py             # summary-report formatting
├── degradation/           # image-degradation operations library (moved from degradation_experiment/)
├── recognition/           # MTCNN + InsightFace recognition package (moved from mtcnn_insightface/)
├── pipeline/<stage>/      # the runnable scripts (see table above)
├── demo/                  # standalone demos
└── legacy/                # superseded script versions (unchanged, not maintained)
```

## Running

Scripts are importable as modules. Run from the **repo root**:

```bash
# Evaluation (InsightFace backend) — the canonical evaluator
python -m experiments.pipeline.evaluation.evaluate_identity_preservation_v2

# Evaluation (DeepFace backend)
python -m experiments.pipeline.evaluation.evaluate_identity_preservation_v2_deepface --device gpu

# Degraded-input baseline (cosine sim on the degraded images themselves)
python -m experiments.pipeline.evaluation.run_cosine_similarity_for_degraded_input

# Restoration — single image (dry-run prints the TurboDiffusion command)
python -m experiments.pipeline.restoration.run_face_restoration \
    --image_path Experiment_Data/Ref/<some>_ref.jpg --dry_run

# Restoration — CSV batch
python -m experiments.pipeline.restoration.run_face_restoration \
    --csv_path degradation_experiment/.../<some>.csv --skip_existing
```

Direct invocation (`python experiments/pipeline/.../script.py`) also works — each
script bootstraps the repo root onto `sys.path`.

## Shared utilities

```python
from experiments.faceid import cosine_similarity, compute_classification_metrics, preprocess_for_detection
from experiments.faceid.embeddings_insightface import get_embedding_robust   # requires insightface
from experiments.faceid.embeddings_deepface import get_embedding_robust      # requires deepface
from experiments.degradation import apply_gaussian_blur, apply_jpeg_compression
```

`import experiments.faceid` does **not** require `insightface` or `deepface` —
the two backends live in submodules and are imported explicitly only by the
scripts that use them.

## Configuration

Every path and constant that used to be hardcoded as
`/home/haipd/TurboDiffusion/...` now lives in **`experiments/config.py`**:

```python
from experiments.config import PROJECT_ROOT, INPUT_BASE, FRONTAL_REF, RESTORATION_ENGINE
```

Data and outputs are intentionally **left in place** at the repo root (and
gitignored — see the root `.gitignore`):

| Purpose | Location |
|---|---|
| Input face dataset (canonical) | `Experiment_Data_Split_Combined/` |
| Degraded images | `degradation_experiment/` |
| Restored videos | `output_full_new_combined/` |
| Evaluation results | `cosine_similarity_output/` |
| Extracted faces | `face_recognition_output/` |
| Model checkpoints | `checkpoints/` |

## Dependencies (research layer, on top of `turbodiffusion`)

`insightface`, `deepface`, `opencv-python`, `pandas`, `scikit-learn`,
`matplotlib`. (The `recognition/` package additionally needs `mtcnn`.)

## Legacy

`experiments/legacy/` holds superseded versions of the evaluation scripts
(`evaluate_identity_preservation.py`, `_02`, `_03`, the standalone
`_deepface`, and `run_cosine_similarity_for_degraded_input_02.py`). They are
preserved as-is for reference; use the canonical scripts under
`pipeline/evaluation/` instead.

## Notes

- The top-level `imaginaire/` directory is a vendored copy of NVIDIA's Imaginaire
  framework (structured as a project: an outer dir wrapping an inner
  `imaginaire/imaginaire/` package). It is **left in place** but is not on the
  import path. `run_all_restoration_side_fast.py` resolves `imaginaire` through
  `turbodiffusion/` on its `sys.path` (i.e. `turbodiffusion/imaginaire/`), exactly
  as it did before the refactor.
- Vietnamese inline comments from the originals are preserved.
- CLI arguments and runtime behavior of relocated scripts are unchanged.
- `run_i2v_from_gaussian_csv.py` had a pre-existing stray-backtick syntax error;
  it was fixed during relocation so the file now parses.
