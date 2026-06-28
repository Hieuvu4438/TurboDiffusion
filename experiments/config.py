"""Centralized configuration for the ``experiments`` research layer.

Every path, threshold, and constant that was previously hardcoded as
``/home/haipd/TurboDiffusion/...`` across the pipeline scripts lives here.
Import from this module instead of inlining paths:

    from experiments.config import PROJECT_ROOT, INPUT_BASE, RESTORATION_ENGINE

All data and output directories are intentionally left in their original
locations at the repository root (see ``.gitignore``); this module only gives
them stable, named handles.
"""

from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
#  Roots
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # .../TurboDiffusion
EXPERIMENTS  = PROJECT_ROOT / "experiments"
PIPELINE     = EXPERIMENTS / "pipeline"

# ────────────────────────────────────────────────────────────────────────────
#  Data & output directories (left in place; see .gitignore)
# ────────────────────────────────────────────────────────────────────────────
# Input face datasets
EXP_DATA_COMBINED = PROJECT_ROOT / "Experiment_Data_Split_Combined"   # canonical input
FRONTAL_REF       = EXP_DATA_COMBINED / "Frontal_Exp" / "Ref"
SIDE_REF          = EXP_DATA_COMBINED / "Side_Exp" / "Ref"

# Restored videos to evaluate (output of the restoration stage)
INPUT_BASE = PROJECT_ROOT / "output_full_new_combined"

# Degradation working area (degraded images + per-type CSVs)
DEGRADATION_EXP  = PROJECT_ROOT / "degradation_experiment"
BLURRED_OUTPUT   = DEGRADATION_EXP / "blurred_output_new"
BLURRED_OUTPUT_OLD = DEGRADATION_EXP / "blurred_output"

# Evaluation results & extracted faces
COSINE_OUTPUT    = PROJECT_ROOT / "cosine_similarity_output"
FACE_REC_OUTPUT  = PROJECT_ROOT / "face_recognition_output"

# CSV manifests at the repo root (hold machine-specific absolute paths)
FRONTAL_TEST_CSV = PROJECT_ROOT / "lfw_frontal_test.csv"
SIDE_TEST_CSV    = PROJECT_ROOT / "lfw_side_test.csv"
FRONTAL_ADDED_CSV = PROJECT_ROOT / "lfw_frontal_test_added.csv"
SIDE_ADDED_CSV    = PROJECT_ROOT / "lfw_side_test_added.csv"
SIDE_TO_FRONTAL_CSV = PROJECT_ROOT / "lfw_side_to_frontal.csv"

# Archives
ZIP_OUTPUT_DIR = PROJECT_ROOT / "zip_output"

# ────────────────────────────────────────────────────────────────────────────
#  Evaluation constants
# ────────────────────────────────────────────────────────────────────────────
RECOGNITION_THRESHOLD = 0.15
VIDEOS_PER_CATEGORY   = 50

# Best-frontal-frame selection (video embedding extraction)
SKIP_START_PERCENT = 0.1
SKIP_END_PERCENT   = 0.1
TOP_K_AVERAGE      = 3

# ────────────────────────────────────────────────────────────────────────────
#  Restoration (TurboDiffusion Wan2.2 I2V)
# ────────────────────────────────────────────────────────────────────────────
# The engine script invoked by the batch runners via subprocess.
RESTORATION_ENGINE = PIPELINE / "restoration" / "run_face_restoration.py"

# Inference target inside the official library.
I2V_INFER_SCRIPT = PROJECT_ROOT / "turbodiffusion" / "inference" / "wan2.2_i2v_infer.py"

LOW_MODEL  = PROJECT_ROOT / "checkpoints" / "TurboWan2.2-I2V-A14B-low-720P-quant.pth"
HIGH_MODEL = PROJECT_ROOT / "checkpoints" / "TurboWan2.2-I2V-A14B-high-720P-quant.pth"

DEFAULT_PROMPT = (
    "A high-quality portrait video strictly preserving the identity of the person "
    "and original biological gender of the subject in the source image. "
    "The person smoothly turns their head from a side-profile to a full frontal view, "
    "ending with them looking directly into the camera lens. "
    "Maintain consistent facial structure throughout the rotation."
)

DEFAULT_RESOLUTION        = "720p"
DEFAULT_NUM_STEPS         = 4
DEFAULT_NUM_FRAMES        = 81
DEFAULT_NUM_SAMPLES       = 1
DEFAULT_SEED              = 0
DEFAULT_ATTENTION_TYPE    = "sagesla"      # one of: sla, sagesla, original
DEFAULT_SLA_TOPK          = 0.1
DEFAULT_USE_ODE           = True
DEFAULT_USE_QUANT_LINEAR  = True
DEFAULT_ADAPTIVE_RESOLUTION = True

# ────────────────────────────────────────────────────────────────────────────
#  Degradation
# ────────────────────────────────────────────────────────────────────────────
BLUR_SIGMAS        = [3, 5, 8, 10, 12, 15]
DEGRADATION_TYPES  = ["downup", "jpeg", "motion", "salt-pepper", "lowlight", "screen"]
DEGRADATION_LEVELS = ["L1", "L2", "L3"]
VIEW_TYPES         = ["frontal", "side"]

# ────────────────────────────────────────────────────────────────────────────
#  Recognition backends
# ────────────────────────────────────────────────────────────────────────────
INSIGHTFACE_MODEL  = "buffalo_l"
DEEPFACE_DETECTOR  = "retinaface"
DEEPFACE_RECOGNIZER = "ArcFace"


def validate_paths() -> list:
    """Return the list of *critical* paths that do not exist on disk.

    Useful as a smoke check before a pipeline run; returns ``[]`` when all
    critical paths are present.
    """
    critical = [PROJECT_ROOT, EXP_DATA_COMBINED, FRONTAL_REF, SIDE_REF]
    return [str(p) for p in critical if not p.exists()]


if __name__ == "__main__":
    print(f"PROJECT_ROOT        : {PROJECT_ROOT}")
    print(f"INPUT_BASE          : {INPUT_BASE}")
    print(f"RESTORATION_ENGINE  : {RESTORATION_ENGINE}")
    print(f"RECOGNITION_THRESHOLD: {RECOGNITION_THRESHOLD}")
    missing = validate_paths()
    print(f"missing critical paths: {missing if missing else 'none'}")
