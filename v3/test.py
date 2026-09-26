# test.py (v3) — Run test inference and write submission files.
#
# Usage (from repo root or v3 directory):
#   python model_scripts/v3/test.py  (or python test.py)
#
# Requires cache/v3/xgb_model.pkl (run train.py first).
# Hybrid blocking -> 45-feature scoring -> Source-disjoint Top-1 Rank Filtering
# -> thresholding (tuned ~0.70) -> matching_results.tsv + candidate_pairs.tsv

from core import run_test

if __name__ == "__main__":
    run_test()
