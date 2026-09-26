# v4 Architecture — Selective Feature Enhancements + Fine-Tuned Thresholding

**Goal:** Improve upon v2's proven 0.9667 validation F0.5 by adding selective features and aggressive threshold tuning. Target: **≥0.975 validation, ≥0.95 test**.

```
raw TSVs ──▶ normalize + ALL-Indic transliteration (v2 proven)
        ──▶ multi-key blocking (9 keys + TF-IDF, v2 proven)
        ──▶ 38 pairwise features (v2's 34 + 4 selective v4 additions)
        ──▶ XGBoost (GPU) + fine-tuned threshold grid (0.40-0.70)
        ──▶ matching_results.tsv + candidate_pairs.tsv
```

## Why v4 Exists — Refinement Without Over-Engineering

**v2 achieved 0.9667 validation F0.5** but there's room to improve:
1. **Threshold tuning was coarse** (0.30-0.60 in 0.05 steps = 7 points tested)
2. **Missing selective features** that target specific noise:
   - Typo-heavy names: Jaro-Winkler catches more than simple fuzz.ratio
   - Postal data robustness: House/PIN mismatches should hard-filter pairs
3. **v3 over-engineered** (45 features, 11 keys) but actually scored lower (0.9351)

**v4 philosophy:** Keep v2's proven blocking, add 4 targeted features, sweep threshold aggressively.

---

## Key Differences from v2

### 1. Preprocessing (Identical to v2)
- ALL-Indic transliteration (9 Brahmic scripts)
- Lowercase + accent stripping
- Legal suffix expansion
- Same normalization rules

### 2. Blocking (Identical to v2)
- 8-key multi-key blocking (exact name, sorted tokens, phonetic sets, house-number anchors)
- TF-IDF parallel scoring (top-30 per S1, min_sim > 0.03)
- Union candidates (45 per S1 cap)
- **Blocking recall target: ≥0.95**

### 3. Features (v2's 34 + 4 New)

**v2's proven 34 features (kept as-is):**
- Fuzz ratios (ratio, partial, token_sort, token_set, WRatio) on names & addresses
- Token overlaps (Jaccard, overlap_min, overlap_max) on names & addresses
- Levenshtein normalized similarity (names & addresses)
- Char 3-gram Jaccard (names & addresses)
- Phonetic match/Jaccard (metaphone token sets)
- House number match, PIN match
- Shape features (length diff, empty flags, first-word match)
- Number overlap (num_jaccard), country match, is_s3

**v4's 4 NEW selective features:**

| Feature | Why Added | Impact |
|---|---|---|
| `name_jw` | Jaro-Winkler captures typo prefixes better (e.g., "Gooogle" ↔ "Google") | +0.003 F0.5 expected |
| `addr_jw` | Same typo robustness for addresses | +0.002 F0.5 |
| `house_mismatch_penalty` | Hard filter: if both have house numbers but differ, strong negative signal | +0.002 F0.5 |
| `pin_mismatch_penalty` | Hard filter: if both have PINs but differ, strong negative signal | +0.002 F0.5 |

All 4 are computed in `compute_features()` alongside v2's 34.

### 4. Training & Threshold Tuning (AGGRESSIVE)

**v2 threshold grid:** `[0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]` (7 points, wide range)

**v4 threshold grid:** `[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]` (7 points, finer + lower range)

**Why the change:**
- v2's optimal threshold was 0.65 (F0.5=0.9667)
- Lower range (0.40-0.70) is more typical for matching problems
- v4's 4 new features may shift optimal point down to 0.55-0.60
- Finer granularity catches the peak

### 5. Model & Output (Identical to v2)
- XGBoost trained on 200k S1 sample
- 90/10 train/validation split
- F0.5 grid search on validation set
- Best threshold cached and reused by test.py

---

## Expected Score Improvement

| Component | Contribution |
|---|---|
| v2 baseline validation F0.5 | 0.9667 |
| Jaro-Winkler on names | +0.003 |
| Jaro-Winkler on addresses | +0.002 |
| House mismatch penalty | +0.002 |
| PIN mismatch penalty | +0.002 |
| Finer threshold tuning | +0.002 |
| **v4 expected validation** | **≥0.9758** |
| **v4 expected test** | **0.965–0.975** (accounting for test diversity) |

---

## Time Budget (Same as v2, ~90 min fresh cache)

| Stage | Est. |
|---|---|
| Train preprocessing | ~12 min |
| Train blocking + features + XGBoost | ~10 min |
| Validation (pool + score + finer grid) | ~15 min |
| Test preprocessing | ~15 min |
| Test blocking | ~5 min |
| Test scoring | ~25–35 min |
| Output writing | ~12 min |
| **Total** | **~90 min** |

---

## Files

- `core.py` — machinery with 4 new features + finer threshold grid
- `train.py` / `test.py` — entry points (same contract as v2)
- Cache namespace: **`cache/v4/`** (independent of v2, v3)

---

## How to Use

**Training:**
```bash
python v4/train.py
```
Runs preprocessing, blocking, feature engineering, XGBoost, and threshold grid search. Caches everything to `cache/v4/`.

Validation F0.5 and optimal threshold are printed at the end. Example:
```
OPTIMAL Validation F0.5 Score: 0.9758 (threshold=0.60)
```

**Testing:**
```bash
python v4/test.py
```
Loads cached model and threshold, runs blocking + scoring on full test set, writes `matching_results.tsv` and `candidate_pairs.tsv`.

---

## Consistency Rules

**Important:** Cache is tied to feature computation. If you modify `compute_features()`, delete these before re-running `train.py`:
- `cache/v4/train_candidates.pkl`
- `cache/v4/xgb_model.pkl`

Otherwise, cached candidates + model may mismatch new features.

---

## Key Insight

v4 is **not a complete rewrite**—it's a **surgical refinement** of v2:
- Keep blocking (it's at 0.97+ recall, hard to beat)
- Keep 34 proven features (they work!)
- Add 4 that target specific noise modes
- Tune threshold more aggressively

This approach minimizes risk (everything is tested in v2) while maximizing gains from the new signal.
