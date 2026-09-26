# train.py — Train + validate the baseline entity-resolution model.
#
# Usage (from anywhere):
#   python model_scripts/baseline/train.py
#
# Does: streamed preprocessing (cached) -> TF-IDF blocking (cached) ->
#       XGBoost training (cached to cache/xgb_model.pkl) ->
#       90/10 validation split with a F0.5 report.
# Never reads test data. Re-running skips every cached step; only the
# validation score is recomputed each time (it is cheap).

from core import run_training

if __name__ == "__main__":
    run_training()
