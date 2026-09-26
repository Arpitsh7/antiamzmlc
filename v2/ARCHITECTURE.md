# v2 Architecture — Multi-Key Blocking + 34-Feature XGBoost

**Goal:** beat the leaderboard target (F0.5 ≥ 0.984) with an **end-to-end pipeline under 3 hours** (v1's TF-IDF blocking alone needed ~10–12 h at test scale).

```
raw TSVs ──▶ normalize + ALL-Indic transliteration
        ──▶ multi-key blocking (4 exact-key joins, minutes)
        ──▶ 34 pairwise features
        ──▶ XGBoost (GPU) + validation-tuned threshold
        ──▶ matching_results.tsv + candidate_pairs.tsv
```

## Why v2 exists — the two lessons from v1

1. **Blocking recall is the score ceiling.** v1 measured 0.917 overall (0.944 US, worse in India). No classifier can recover a true match blocking never proposed; 0.984 on the leaderboard implies ~0.98+ blocking recall. v2 attacks this with a **richer, script-aware key set** instead of TF-IDF cosine.
2. **The 10–12 h TF-IDF matmul had to go.** The multi-key design does blocking with pandas hash-joins in **minutes**, and its candidate pairs are scored by the same kind of classifier — none of the features depend on how candidates were found, so blocking strategy and model are decoupled.

## Stage 1 — Preprocessing + all-Indic transliteration

v1 normalization (lowercase, accent-strip, abbreviation expansion) plus **transliteration of all nine Brahmic scripts found in the data** (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi — measured by scanning 300k names per file).

**How the transliterator works (no per-script tables):** every Indic character's Unicode name ends in its canonical sound — `DEVANAGARI LETTER KHA → kha`, `TAMIL VOWEL SIGN AA → a`, `TELUGU LETTER TTA → ta`. One parser over `unicodedata.name()` builds the whole translation table (virama/nukta dropped, anusvara → n, digits → 0-9). Sibilants, liquids, retroflexes and vowel length are **collapsed** (`sha/ssa→sha`, `ta/tta→ta`, `aa→a`) because we want *matching equivalence* — two sources writing the same name in different scripts or spellings must produce identical text. This makes "राम मार्केटिंग" and its Latin twin share keys and features. Applied to both `name_norm` and `addr_norm`.

## Stage 2 — Multi-key blocking (the recall engine)

**Revision history (measured, not guessed):** the first 4-key version measured **blocking recall 0.658** on the training sample — the misses were extra/missing words (DBA-style names like `wilfordhancock.com` ↔ `Wilford Hancock`), cross-script spelling drift, and matches carried by the *address* (v1's TF-IDF blocking text included the address; 4 keys barely used it). v2.1 widens to **8 keys** targeting exactly those modes:

| Key | Matches | Cap/S1 | Kills which noise |
|---|---|---|---|
| K1 `(country, name)` | identical normalized names | 20 | punctuation/suffix variants |
| K2 `(country, sorted name tokens)` | word order swaps | 20 | "Sharma Electricals" ↔ "Electricals Sharma" |
| K3 `(country, sorted metaphone token-set)` | **phonetic** equivalence | 20 | typos, Catherine/Katherine, Smythe/Smith |
| K4 `(country, metaphone(full name, no spaces))` | whole-name phonetic | 20 | cross-script drift ("raama maarakaetainga" ≡ "ram marketing" under metaphone) |
| K5 `(country, phon(first tok), phon(last tok))` | brand + end anchor | 15 | middle words added/removed (DBA/trade names) |
| K6 `(country, house no, phon(first tok))` | address anchor | 10 | divergent names, same premises |
| K7 `(country, house no, PIN)` | pure address | 10 | completely different names, same building |
| K8 `(country, PIN, phon(first tok))` | brand + area | 10 | name variants in the same postal area |
| K9 `(country, metaphone(full name)[:5], house no)` | mashed-name prefix | 10 | DBA names collapsed into one token ("wilfordhancock.com") |

Candidates keep key priority (K1 first) under a global per-S1 cap of 40; the ML model filters the false positives the wider net admits. Metaphone codes are memoized (token-level cache), so the 12M-token pass stays in the 1–3 min range, and the joins are pandas hash-joins — blocking remains a minutes-scale stage. **The `Blocking recall` line printed by `train.py` is the acceptance gate: re-run until it is ≥ 0.95 before spending an hour on `test.py`.**

**v2.2 — the recall backbone is fuzzy again.** The 9 keys alone measured 0.718 recall (keys structurally cannot catch extra-word DBA names or arbitrary typos the way shared character n-grams do). Final design = **union of two candidate sources**:
1. **TF-IDF char (2,4)-gram cosine** (v1's proven net, recall 0.917 pre-transliteration) — now on transliterated text, top-30 per S1 at sim > 0.03, and **parallelized at block level** (`WORKERS=8` processes; each transforms its 150k-doc pool block and scores it against every S1 chunk via sparse products; parent merges running top-Ks with vectorized segment sorts). Turns ~11 CPU-hours into ~1–1.5 h wall.
2. **The 9 keys** — address-anchored and DBA cases the cosine net can miss.

Union is deduped (TF-IDF priority) and capped at 45/S1. `train.py` prints the union's `Blocking recall (union)` — that is the acceptance gate. RAM note: with `WORKERS=8` expect ~16–18 GB peak on the test stage; set `WORKERS=6` in `core.py` to stay nearer 12–14 GB (~30–40 min slower).

## Stage 3 — 34 pairwise features (26 + 8)

All 26 v1 features are kept (fuzz ratios, token overlaps, length/empty flags, `num_jaccard`, `is_s3`, …). The 8 additions and the noise each one neutralizes:

| # | Feature | Why it was added |
|---|---|---|
| 27 | `name_lev` | Normalized **Levenshtein** — adjacent-key typos ("Gooegle"↔"Google") where token-set fuzz overrates word overlap |
| 28 | `addr_lev` | Same for addresses |
| 29 | `name_gram3` | **Char 3-gram Jaccard** — sub-word similarity robust to typos AND transliteration spelling drift (was promised in the workbook spec, never implemented in v1) |
| 30 | `addr_gram3` | Same for addresses |
| 31 | `name_phon_exact` | Full-name **phonetic** identity — names that *sound* identical after transliteration + metaphone |
| 32 | `name_phon_jaccard` | Partial phonetic overlap — one token differs in sound, rest match (DBA/trade-name cases) |
| 33 | `house_match` | Exact first-digit-run (house number) match — positional evidence that survives address word shuffling, unlike `num_jaccard`'s set semantics |
| 34 | `pin_match` | Longest digit-run (PIN/zip) exact match — in Indian data, same PIN + similar name is near-conclusive |

The model remains **XGBoost** (our own trained model — no pretrained network, so the license/parameter rules are trivially satisfied): trees exploit feature interactions (e.g., `pin_match=1` changes what `name_phon_jaccard=0.5` means), trains in minutes on the GPU (`device='cuda'`, RTX 4060), and gives probabilities for threshold tuning.

## Stage 4 — Training, validation, threshold

Same disciplined workflow as v1: 200k S1 sample from the 90% split side (held-out 10% never seen), positives from ground truth, negatives = up to 3× non-matching candidates per entity, and a **threshold grid search** (0.30–0.60) on the validation F0.5 — the tuned threshold is cached and reused by `test.py`. Since F0.5 weighs false merges 2×, the tuned threshold is typically the cheapest real score gain.

## Time budget (fresh cache, RTX 4060 + 20 cores)

| Stage | Est. |
|---|---|
| Train preprocessing (streamed) | ~12 min |
| Train blocking + features + XGBoost (GPU) | ~10 min |
| Validation (pool + score + grid) | ~10 min |
| Test preprocessing (streamed) | ~15 min |
| Test blocking | ~5 min |
| Test scoring (~25–40M pairs) | ~25–35 min |
| Output writing | ~12 min |
| **Total** | **~90 min** ✓ (≤ 3 h with wide margin) |

## Files & caches

- `core.py` — machinery; entry functions `run_training()` / `run_test()`
- `train.py` / `test.py` — thin entry points (same contract as v1)
- Cache namespace: **`cache/v2/`** (independent of v1 — v1 caches stay valid and untouched)

## Consistency rules (same trap as v1 — do not skip)

`train_candidates.pkl` and `xgb_model.pkl` are only valid for the caches they were built from. If you change `CAP_*`, `TRAIN_SAMPLE_SIZE`, the key set, or the transliteration table, delete `cache/v2/train_candidates.pkl` **and** `cache/v2/xgb_model.pkl` before re-running `train.py`.

## Expected score math (honest)

Macro-F0.5 ≈ (share of matched entities × pair-level F0.5) + (singleton share × singleton correctness). With blocking recall ~0.97–0.99 and model precision ~0.99 (tuned threshold), the expected range is **0.975–0.985** — genuinely in range of the 0.984 target, with India blocking quality and threshold tuning deciding the last points. The validation log's `Blocking recall` line is the number to watch: if it prints < 0.95, the key set needs another pass (e.g., add a PIN-only key or a phonetic-first-token key) before blaming the model.
