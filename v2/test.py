# test.py (v2) — Run test inference and write the submission files.
#
# Usage (from anywhere):
#   python model_scripts/v2/test.py
#
# Requires cache/v2/xgb_model.pkl (run train.py first). Multi-key blocking
# (cached) -> scoring with the validation-tuned threshold -> outputs.
# End-to-end target: well under 3 hours on a fresh cache.
#
# NOTE: this overwrites output/*.tsv — copy any submission you want to keep
# (e.g. to tmp/submission1/) BEFORE running this.

from core import run_test

if __name__ == "__main__":
    run_test()
