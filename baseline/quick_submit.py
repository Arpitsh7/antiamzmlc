# quick_submit.py — FAST deadline submission (exact-key blocking + trained model).
#
# Usage (from anywhere):
#   python model_scripts/baseline/quick_submit.py
#
# Requires cache/xgb_model.pkl (from train.py) and the preprocessed test caches
# (already built by the full test.py run). Produces the same two submission
# files as test.py, in ~30-45 minutes instead of ~12 hours — at the cost of
# lower recall (exact-name matching instead of TF-IDF fuzzy blocking).
#
# The full test.py run remains the higher-quality submission; copy the output
# files somewhere safe before it overwrites them.

from core import run_quick_test

if __name__ == "__main__":
    run_quick_test()
