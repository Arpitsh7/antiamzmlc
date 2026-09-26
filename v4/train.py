# train.py (v2) — Train + validate the v2 entity-resolution model.
#
# Usage (from anywhere):
#   python model_scripts/v2/train.py
#
# Streamed preprocessing (cached in cache/v2/) -> multi-key blocking (cached)
# -> 34 features -> XGBoost (GPU if available) -> validation F0.5 with
# threshold grid search. Never touches test data. ~30-45 min first run.

from core import run_training

if __name__ == "__main__":
    run_training()
