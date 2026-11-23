"""Task 1 (pruning and weight quantization) fixed settings."""
import os
import pathlib

SEED = 6304
MODEL_ID = "Qwen/Qwen2.5-1.5B"
ELIGIBLE = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Paths. On Kaggle the data-setup notebook output is attached as an input; override with env vars.
REPO = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = pathlib.Path(os.environ.get("PA4_DATA", REPO / "data"))
OUT_DIR = pathlib.Path(os.environ.get("PA4_OUT", REPO / "outputs" / "task1"))
RESULTS_DIR = pathlib.Path(os.environ.get("PA4_RESULTS", REPO / "results" / "task1"))
SMOKE = os.environ.get("ATML_SMOKE_TEST") == "1"

# Evaluation
WINDOW = 512
EVAL_BATCH = 8
HELLASWAG_FULL = 2000
HELLASWAG_DIAG = 512
HELLASWAG_BATCH = 16

# Calibration statistics
CALIB_SEQS = 128
CALIB_SUBSAMPLE_TOKENS = 4096      # fixed token subsample kept per linear input for AWQ search / output error
CALIB_BATCH = 8

# Pruning
SPARSITIES = (0.25, 0.50, 0.75)
P3_SPARSITIES = (0.50, 0.75)
P3_MIN, P3_MAX = 0.10, 0.85
OWL_M = 5.0                         # outlier threshold multiple of the layer mean Wanda score
P3_LAMBDAS = (0.04, 0.08, 0.12, 0.20)
P3_SENSITIVITIES = ("owl", "bi")    # candidate sensitivity measures, selected on the C4 development set

# Quantization
BITS = (8, 4, 3)
GROUP = 128
ALLOWED_BITS = (2, 3, 4, 8)
AWQ_ALPHAS = tuple(i / 10 for i in range(11))   # s = mean|x|^alpha grid
SCALE_BYTES = 2                     # FP16 group scale
ZERO_BYTES = 1                      # uint8 zero-point per group

# Resource measurement
PREFILL_LENGTHS = (512, 2048)
DECODE_TOKENS = 64
TIMING_REPS = 5
WARMUP_REPS = 2
