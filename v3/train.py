# train.py (v3) — Train + validate the v3 entity-resolution model.
#
# Usage (from repo root or v3 directory):
#   python model_scripts/v3/train.py  (or python train.py)
#
# Streamed preprocessing + French/Indic transliteration (cached in cache/v3/)
# -> 11-key + parallel TF-IDF hybrid blocking (cached)
# -> 45 pairwise features (IDF weighting, Jaro-Winkler, LCS, house/PIN mismatch)
# -> GPU XGBoost classifier
# -> Validation F0.5 optimization + Source-disjoint Top-1 Rank Filtering thresholding.

from core import run_training

if __name__ == "__main__":
    run_training()
