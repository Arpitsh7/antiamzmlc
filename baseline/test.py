# test.py — Run test inference and write the submission files.
#
# Usage (from anywhere):
#   python model_scripts/baseline/test.py
#
# Requires cache/xgb_model.pkl (produced by train.py — run that first).
# Does: streamed test preprocessing (cached) -> test blocking (cached) ->
#       scoring -> output/matching_results.tsv + output/candidate_pairs.tsv
#
# Takes roughly 4-5 hours on the first run; every stage is cached, so an
# interruption resumes where it stopped.

from core import run_test

if __name__ == "__main__":
    run_test()
