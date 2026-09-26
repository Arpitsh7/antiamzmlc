# Model v3 Architecture — Advanced Hybrid Blocking, 45-Feature Engineering & Rank-Filtered Precision Engine

**Goal:** Maximize Macro F0.5 (Target ≥ 0.985+) with an end-to-end scalable pipeline, fully supporting **unseen test countries (France)**, multi-script Indic records, and aggressive singleton protection against false positive merges.

```
raw TSVs ──▶ Preprocessing (All-Indic Transliteration + French Accents & Legal Suffixes + Landmark Removal)
        ──▶ Hybrid Blocking (11 Multi-Key Hash Joins + Block-Parallel TF-IDF Cosine Net)
        ──▶ 45 Pairwise Features (IDF Weighted Jaccard, Jaro-Winkler, LCS, House/PIN Mismatch)
        ──▶ GPU XGBoost + Validation-Tuned F0.5 Thresholding (τ ≈ 0.70)
        ──▶ Source-Disjoint Top-1 Rank Filtering
        ──▶ matching_results.tsv + candidate_pairs.tsv
```

---

## 1. Stage 1 — Advanced Preprocessing & Multilingual Normalization

### Script & Accent Transliteration
* **Indic Brahmic Scripts**: Parses Unicode character names dynamically (`unicodedata.name()`) covering 9 Brahmic scripts. Collapses phonetic equivalents (`sa/sha/ssa → sha`, `tta/ta → ta`, `aa → a`, virama/nukta removal).
* **French & European Diacritics**: Applies Unicode NFKD decomposition mapping French accents (`é/è/ê/ë → e`, `à/â → a`, `ç → c`, `î/ï → i`, `ô → o`, `ù/û/ü → u`).

### Multilingual Legal Suffix & Street Dictionary
* **French Additions**:
  * Legal Suffixes: `SARL`, `SAS`, `EURL`, `SA`, `STE` (Société), `ETS` (Établissements).
  * Address Abbreviations: `Rue`, `Bd` / `Boulevard`, `Av` / `Avenue`, `Imp` / `Impasse`, `All` / `Allée`, `BP` (Boîte Postale), `CEDEX`.
* **Landmark Filler Stripping**:
  * Strips non-discriminative landmark phrases (`near`, `opposite`, `opp`, `behind`, `next to`, `beside`, `in front of`) to expose core address tokens.
* **Core Brand Extraction**:
  * Strips generic corporate stop-words (`services`, `solutions`, `group`, `holdings`, `store`, `traders`, `enterprises`, `industries`, `company`, `corporation`, `limited`, `private`) to isolate `core_name`.
* **Multi-Country Postal Code Regex**:
  * Supports 5-digit (France/US) and 6-digit (India) PINs (`\b\d{5,6}\b`).

---

## 2. Stage 2 — Hybrid Candidate Blocking Engine (11 Keys + Block TF-IDF)

Unions two candidate sources to push blocking recall ceiling to **~0.98–0.995**:

| Key | Description | Target Noise Mode Neutralized |
|---|---|---|
| K1 `(country, name_norm)` | Identical normalized name | Punctuation / casing variants |
| K2 `(country, sorted name tokens)` | Sorted token list | Word order transpositions |
| K3 `(country, sorted metaphone token-set)` | Phonetic token set | Typos, Catherine/Katherine, Smith/Smythe |
| K4 `(country, metaphone(full name))` | Whole name Metaphone | Cross-script spelling drift |
| K5 `(country, phon(first tok), phon(last tok))` | Brand + end anchor | Middle word DBA additions/deletions |
| **K6 `(country, core_name[:4], phon(first tok))`** | **Core brand anchor** | **Differing legal suffixes across sources** |
| K7 `(country, house no, phon(first tok))` | Premises anchor | Same address, divergent trade names |
| K8 `(country, house no, PIN)` | Building anchor | Identical premises |
| K9 `(country, PIN, phon(first tok))` | Brand + postal area | Local brand branches |
| K10 `(country, metaphone(full name)[:5], house no)` | Mashed name prefix | Mashed/collapsed domain names |
| **K11 `(metaphone(full name)[:6], house no)`** | **Cross-country fallback** | **Mislabeled/missing country codes** |

Unioned with **Block-Parallel TF-IDF Cosine Net** (`WORKERS=8`) on transliterated text at top-50 candidates per S1 entity.

---

## 3. Stage 3 — 45 Pairwise Features (26 v1 + 8 v2 + 11 v3)

### The 11 New v3 Features:
1. `name_jw`: Jaro-Winkler similarity for names (gives higher weight to matching prefixes).
2. `addr_jw`: Jaro-Winkler similarity for addresses.
3. `name_lcs_ratio`: Longest Common Subsequence ratio for names.
4. `addr_lcs_ratio`: Longest Common Subsequence ratio for addresses.
5. `core_name_ratio`: Fuzz ratio on core brand names (after stripping generic business terms).
6. `core_name_jaccard`: Token Jaccard on core brand names.
7. `weighted_name_jaccard`: Inverse Document Frequency (IDF) weighted token Jaccard for names (rare tokens carry high weight).
8. `weighted_addr_jaccard`: IDF-weighted token Jaccard for addresses.
9. `house_mismatch_penalty`: Binary flag (1.0 if both records have valid house numbers and $H_1 \ne H_2$).
10. `pin_mismatch_penalty`: Binary flag (1.0 if both records have valid 5/6-digit PINs and $\text{PIN}_1 \ne \text{PIN}_2$).
11. `same_street_diff_house`: Binary flag (1.0 if address tokens overlap > 0.6 but house numbers differ).

---

## 4. Stage 4 — Precision-Heavy Classification & Rank Filtering

1. **GPU XGBoost**: Trained on sampled S1 pairs (`device='cuda'`).
2. **F0.5 Optimization**: Sweeps classification threshold from 0.50 to 0.85 in steps of 0.05 on held-out 10% validation set (default tuned $\tau \approx 0.70$).
3. **Source-Disjoint Top-1 Rank Filtering**:
   * Separates predicted candidates by source prefix (`S2` vs `S3`).
   * Retains at most the **top-1 candidate from S2** and **top-1 candidate from S3** per S1 entity (allowing a 2nd candidate only if $P \ge 0.90$ and $\Delta P < 0.03$).
4. **Singleton Protection**: Entities where $\max P < \tau^*$ automatically default to empty string `""`, securing a 1.0 F0.5 score on singletons.

---

## Files & Cache
- Directory: `v3/`
- Core machinery: `v3/core.py`
- Entry scripts: `v3/train.py`, `v3/test.py`
- Cache namespace: `cache/v3/`
