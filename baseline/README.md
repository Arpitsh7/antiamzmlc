# Baseline Entity-Resolution Pipeline (v1.1, memory-optimized)

Two entry scripts, one shared core. All paths are anchored to the repo root as
absolute paths in `core.py` — caching always happens in
`E:\amazon-ml\6ab10eb3b23ba_student_resource\student_resource\cache`, and
outputs/logs/datasets resolve to the same folders no matter where you launch
the scripts from.

| Script | What it does | Touches test data? |
| :--- | :--- | :--- |
| `train.py` | Preprocess train TSVs (streamed, cached) → TF-IDF blocking (cached) → XGBoost training (cached) → **Validation F0.5** on a held-out 10% split | No |
| `test.py`  | Loads the trained model → test preprocessing (cached) → test blocking (cached) → scoring → writes `output/matching_results.tsv` + `output/candidate_pairs.tsv` | Yes |
| `quick_submit.py` | **Deadline path**: exact-name-key blocking (minutes) + the same trained model → same two output files in ~30-45 min. Lower recall than `test.py`; use when a submission is due before the full run can finish. Can run concurrently with `test.py` (reads the same caches, writes no cache). | Yes |

```bash
# from the repo root
python model_scripts/baseline/train.py     # ~1-2 h first run, then minutes
python model_scripts/baseline/test.py      # ~4-5 h first run (needs train.py first)
```

## Files
- `core.py` — all pipeline machinery: preprocessing, `stream_source`, sparse TF-IDF
  blocking (`CandidateIndex`), 26 pairwise features, XGBoost train/inference,
  submission writer, memory reporter (`mem_report`), and the two entry functions
  `run_training()` / `run_test()`.
- `train.py` / `test.py` — thin entry points.

## Caches (in `cache/`)
Every stage writes a cache and is skipped on re-runs. Delete the whole folder (or
individual files) to force a rebuild.

| File | Produced by | Contents |
| :--- | :--- | :--- |
| `train_s1_sample.parquet` | train.py | 200k sampled S1 records, normalized (90% split side) |
| `train_s23_sample.parquet` | train.py | matched + sampled negative S2/S3 pool, normalized |
| `train_gt_sample.pkl` | train.py | ground truth for the sample |
| `train_candidates.pkl` | train.py | blocking candidates (compact `CandidateIndex`) |
| `xgb_model.pkl` | train.py | trained XGBoost classifier |
| `val_s1_proc.parquet`, `val_gt.pkl`, `val_s23_pool.parquet` | train.py | validation split artifacts |
| `test_s1_proc.parquet`, `test_s23_s2_proc.parquet`, `test_s23_s3_proc.parquet` | test.py | normalized test data (per source) |
| `test_candidates.pkl` | test.py | test blocking candidates |

## Consistency rule
`train_candidates.pkl` and `xgb_model.pkl` are only valid for the train caches they
were built from. If you change sampling, split, or blocking knobs (`CHUNK_S1`,
`MAX_DF`, `FIT_SAMPLE_MAX`, `TOP_K_CANDIDATES`, …), delete `train_candidates.pkl`
**and** `xgb_model.pkl` so both are rebuilt against the new settings — otherwise a
stale model silently scores the validation split (this exact trap happened once;
stale artifacts printed `Blocking recall: 0.0001`).

## Tuning knobs
See the config block at the top of `core.py`. The ones that matter most:
`MATCH_THRESHOLD` (tune 0.30–0.60 on the validation F0.5), `TOP_K_CANDIDATES`,
`MAX_DF`, `TRAIN_SAMPLE_SIZE` inside `build_train_caches`.
