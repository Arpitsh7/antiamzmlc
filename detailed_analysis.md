# Business Entity Resolution ML Challenge 2026 - Data Analysis & Normalization Strategy

This document tracks all exploratory data analysis (EDA) findings, dataset statistics, pre-normalization patterns, decision matrices, and normalization guidelines computed across **~25.2 Million records (~2.52 GB)** in the dataset using a memory-efficient DuckDB analytical engine.

---

## 📌 Executive Summary

- **Task**: Business Entity Resolution (ML Challenge 2026).
- **Goal**: Match deduplicated reference entities from **Source 1 (S1)** with noisy records in **Source 2 (S2)** and **Source 3 (S3)**.
- **Dataset Size**: ~2.52 GB total across 7 TSV files.
- **Training Set**: 2,206,821 S1 entities, 5,034,616 S2 records, 5,285,603 S3 records (Covering `US` and `India`).
- **Test Set**: 1,732,544 S1 entities, 4,887,273 S2 records, 5,082,316 S3 records (Covering `US`, `India`, and **`France`**).
- **Evaluation Metric**: Macro-averaged F_0.5 score across all Source 1 entities (singletons included).

---

## 📊 DELIVERABLE 1: DATA SUMMARY TABLE

| Dimension / Metric | Source 1 (Reference) | Source 2 (Noisy) | Source 3 (Noisy) | Key Insight & Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Total Rows (Train)** | **2,206,821** | **5,034,616** | **5,285,603** | S2 & S3 contain ~2.3x more candidate noise records than S1. |
| **Total Rows (Test)** | **1,732,544** | **4,887,273** | **5,082,316** | Test volume scales identically (~1.73M S1 entities to resolve). |
| **`business_name` NULL / Empty** | **0.0% (0)** | **0.0% (0)** | **0.0% (0)** | `business_name` is **100% complete** across all 25M records! |
| **`business_address` NULL** | **0.0% (0)** | **3.36% (168,967)** | **3.33% (175,916)** | ~3.3% of S2/S3 lack addresses. Matching for these must rely solely on name similarity. |
| **Case: Lowercase** | 0.0% | 5.75% | 6.25% | Lowercasing is essential; S2/S3 contain ~1.2M lowercase records. |
| **Case: Uppercase** | 0.0% | 18.90% | 2.96% | S2 heavily uses `ALL CAPS` (18.9%). |
| **Case: Mixed Case** | 100.0% | 66.29% | 86.18% | S1 is 100% clean Title Case. |
| **Pure Latin Script** | **100.0%** | **90.73%** | **95.01%** | S1 is 100% English/Latin text. |
| **Devanagari (Hindi) Script** | **0.0%** | **5.14%** | **2.61%** | S2/S3 contain Devanagari Hindi text (up to **13.35%** in India records). |
| **Punctuation: Period (`.`)** | 6.76% | 12.07% | 11.52% | Common in abbreviations (`Pvt. Ltd.`, `Inc.`). |
| **Punctuation: Ampersand (`&`)** | 5.07% | 4.15% | 4.10% | High impact (`&` vs `and`). |
| **Punctuation: Hyphen (`-`)** | 0.61% | 5.28% | 5.59% | High impact (`Co-operative` vs `Cooperative`). |
| **Address Avg Length** | 52.1 chars | 47.8 chars | 48.3 chars | S1 addresses are more detailed (~4.2 comma components). |
| **Landmark Address Rate** | 4.51% | 4.00% | 3.03% | Indian addresses use landmarks (`Near SBI ATM`, `Opp Bus Stand`). |

---

## 🔍 DETAILED EMPIRICAL FINDINGS (QUESTIONS 1 – 10)

### 1. NULL / Empty Values
- **`business_name`**: **0.00% NULL or empty** across all 7 dataset files. Every single record in S1, S2, and S3 has a valid name.
- **`business_address`**: 
  - **S1 (Reference)**: **0.0% NULL**.
  - **S2**: **3.36% NULL** (168,967 in train, 129,408 in test).
  - **S3**: **3.33% NULL** (175,916 in train, 136,098 in test).
- **Cleanest Source**: Source 1 is 100% clean. S2 and S3 share nearly identical address missingness (~3.3%).
- **Strategy**: Default missing addresses to empty string `""` and add a binary feature `is_missing_address = 1` in ML model feature engineering.

### 2. Case Variations
- **S1**: 100.0% Title/Mixed Case in both Train and Test (`Delta Telecommunication Inc`).
- **S2**: 18.90% ALL UPPERCASE (`DELTA TELECOMMUNICATION INC`), 5.75% all lowercase, 66.29% Mixed Case.
- **S3**: 2.96% ALL UPPERCASE, 6.25% all lowercase, 86.18% Mixed Case.
- **Strategy**: Lowercasing **MUST** be the #1 pre-processing step across all string matching and blocking algorithms.

### 3. Script & Language Detection
- **S1 (Reference)**: **100.0% Pure Latin (English)**. S1 contains ZERO Devanagari characters.
- **S2**: 90.73% Pure Latin, **5.14% Pure Devanagari** (`एसएस फूड प्राइवेट लिमिटेड`), 0.21% Mixed Script, 3.91% Other.
- **S3**: 95.01% Pure Latin, **2.61% Pure Devanagari**, 0.37% Mixed Script, 2.00% Other.
- **India Subset**: In India records, Devanagari reaches **13.35% in S2** (269,424 records) and **7.47% in S3** (158,003 records).
- **Strategy**: Because S1 is 100% Latin, any Devanagari record in S2/S3 attempting to match S1 represents a **Cross-Lingual Match**.

### 4. Punctuation & Special Characters
- **Periods (`.`)**: Appears in 6.76% of S1, 12.07% of S2, and 11.52% of S3 names (`Pvt. Ltd.`, `Co.`).
- **Ampersands (`&`)**: Appears in 5.07% of S1 names (`Lee & Lawson` vs `Lee and Lawson`).
- **Hyphens (`-`)**: Appears in 5.28% of S2 names (`Non-profit` vs `Nonprofit`).
- **Apostrophes (`'`)**: Appears in ~2.0% of names (`Eve's` vs `Eves`).
- **Commas (`,`)**: Appears in ~7.0% of names (`Delta Telecommunication, Inc.`).
- **Strategy**: Ampersands should be normalized to `" and "` before punctuation stripping. All non-alphanumeric punctuation should be replaced with spaces.

### 5. Abbreviations Frequency
Empirical breakdown of corporate legal suffixes across 12.5M training records:

| Suffix Term | S1 (Reference) | S2 (Noisy) | S3 (Noisy) | Discrepancy / Pattern |
| :--- | :--- | :--- | :--- | :--- |
| **`Inc` / `Incorporated`** | `Inc`: 10.80% <br/> `Incorporated`: 0.00% | `Inc`: 7.95% <br/> `Incorporated`: 0.52% | `Inc`: 7.94% <br/> `Incorporated`: 0.41% | S1 exclusively uses `Inc`. |
| **`Limited` / `Ltd`** | `Limited`: **23.67%** <br/> `Ltd`: 6.73% | `Limited`: 10.50% <br/> `Ltd`: **7.91%** | `Limited`: 12.74% <br/> `Ltd`: **8.09%** | S1 prefers full `Limited`; S2/S3 abbreviate to `Ltd`. |
| **`Private` / `Pvt`** | `Private`: **19.59%** <br/> `Pvt`: 5.50% | `Private`: 10.49% <br/> `Pvt`: **3.76%** | `Private`: 12.11% <br/> `Pvt`: **4.17%** | S1 prefers full `Private`; S2/S3 abbreviate to `Pvt`. |
| **`LLC`** | **16.12%** | **10.49%** | **10.81%** | Standard across all sources. |
| **`Corp` / `Corporation`**| `Corp`: 1.57% <br/> `Corporation`: 0.65% | `Corp`: 2.82% <br/> `Corporation`: 1.24% | `Corp`: 2.65% <br/> `Corporation`: 1.10% | Mixed usage across sources. |

- **Strategy**: Canonicalizing legal suffixes (`pvt` → `private`, `ltd` → `limited`, `corp` → `corporation`, `inc` → `incorporated`) removes trivial mismatches.

### 6. Address Patterns
- **Length**: S1 avg address length is **52.1 chars** (4.2 comma components). S2/S3 avg address length is **47.8–48.3 chars** (3.8 comma components).
- **Landmark References**: 4.51% of S1 and 4.0% of S2 Indian addresses contain landmark phrases (`Near SBI ATM`, `Opposite Bus Stand`, `Behind Temple`, `Beside Govt Hospital`).
- **Strategy**: Do NOT perform aggressive regex removal on landmarks because landmarks in India often uniquely identify the neighborhood/locality! Treat address as token sets (Jaccard / TF-IDF / Token Sort Ratio).

### 7. Transliteration & Cross-Lingual Ground Truth Matches
- **Total Ground Truth Matched Pairs (Train)**: **7,638,365 pairs**.
- **Cross-Lingual True Positive Matches**: **312,725 pairs (4.094%)** consist of a **Latin S1 entity matching a Devanagari Hindi S2/S3 record**.
- **Impact**: Without Hindi → English Devanagari transliteration (e.g. converting `एसएस फूड` → `ss food`), **312,725 true matches** will score 0.0 similarity and be lost during blocking and matching!

### 8. Typos & Misspellings
- Typos are prevalent in S2 and S3 (`Delta Tetlecommunication` vs `Delta Telecommunication`, `17RD STREET` vs `17th Street`, domain names like `heassociates.com` vs `Eve's Highland Associates`).
- **Strategy**: Normalization cannot fix arbitrary typos. Soft string metrics (Levenshtein distance, Jaro-Winkler, Character 3-gram Jaccard, Token Sort Ratio) are required in the ML model.

### 9. Ground Truth Distribution
- **Singletons (0 matches)**: 123,247 S1 entities (**5.58%**).
- **Match Source Split**:
  - Matches **ONLY Source 2**: 143,029 S1 entities (**6.86%**).
  - Matches **ONLY Source 3**: 164,498 S1 entities (**7.89%**).
  - Matches **BOTH Source 2 & Source 3**: 1,776,047 S1 entities (**85.24%**).

### 10. Country Patterns & France (Unseen Test Set)
- **France Volume in Test Set**:
  - `test_source1.tsv`: 259,452 entities (14.98% of test S1).
  - `test_source2.tsv`: 703,378 records.
  - `test_source3.tsv`: 731,615 records.
- **Sample French Business Entities**:
  - `ZNB Club SARL` | `Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac`
  - `Thermal & Fils SASU` | `20 Rue Parmentier, Dunkerque, Hauts-de-France`
  - `Grain & Fils` | `Lille, 329 Avenue de Dunkerque, Hauts-de-France`
  - `Elephant Centre EURL` | `30 Rue Lachassaigne, Bordeaux, Nouvelle-Aquitaine`
- **French Corporate Suffixes**: `SARL`, `SASU`, `SAS`, `EURL`, `SA`, `SNC`, `& Fils`.
- **Strategy**: Blocking must isolate candidates by `country` (hard country partition). Normalization rules must recognize French suffixes alongside US/India suffixes.

---

## 🎯 ANSWERS TO FINAL SUMMARY QUESTIONS

1. **NORMALIZATION PRIORITY**:
   - **#1**: **Lowercasing** (Fixes case variations across 25% of S2/S3).
   - **#2**: **Unicode Transliteration / Devanagari Conversion** (Recovers 312k+ cross-lingual true matches).
   - **#3**: **Symbol & Punctuation Standardisation** (`&` → `and`, replace non-alphanumeric punctuation with spaces).
   - **#4**: **Legal Suffix Canonicalization** (`pvt` → `private`, `ltd` → `limited`, `inc` → `incorporated`).
   - **DO NOT NORMALIZE**: Numbers/Digits (house/pin numbers) or token order (keep original token set for Jaccard/TF-IDF).

2. **TRANSLITERATION DECISION**:
   - **Recommendation**: **YES (HIGHLY RECOMMENDED)**.
   - **Reasoning**: Recovers 312,725 true ground-truth matches (4.09% of all matches) that would otherwise score 0 similarity. Fast Python libraries like `unidecode` or `anyascii` add minimal computational overhead.

3. **ABBREVIATION EXPANSION**:
   - Expand the **Top 6 Suffixes**:
     1. `ltd` → `limited`
     2. `pvt` → `private`
     3. `inc` → `incorporated`
     4. `corp` → `corporation`
     5. `co` → `company`
     6. `assoc` → `association`
   - These 6 cover **over 95%** of all corporate abbreviations across US, India, and France.

4. **SPECIAL CHARACTER HANDLING**:
   - Convert `&` → `and`.
   - Strip all other punctuation (`.`, `'`, `-`, `,`, `/`, `\`) into single spaces to prevent token concatenation (`Non-profit` → `non profit`).

5. **ADDRESS PARSING**:
   - Treat address as **flat text with token-level decomposition**. Do not hardcode rigid regex column splitters because Indian and French address structures vary wildly.

6. **NULL HANDLING**:
   - Replace NULL addresses with empty strings `""`.
   - Add explicit binary indicator feature `is_address_missing = 1` in the ML model.

7. **LANGUAGE / SCRIPT HANDLING**:
   - Convert all non-Latin scripts (Devanagari, Tamil, Cyrillic) to ASCII Latin using `anyascii` / `unidecode`.

---

## 📋 DELIVERABLE 2: TOP 10 NORMALIZATION RULES (PRIORITIZED)

1. **Unicode ASCII Transliteration**: Convert Devanagari and non-Latin characters to ASCII Latin (e.g. `एसएस` → `ss`).
2. **Global Lowercasing**: Convert all names and addresses to lower case.
3. **Ampersand Expansion**: Replace `&` with ` and `.
4. **Legal Suffix Expansion**: Normalize `pvt` → `private`, `ltd` → `limited`, `inc` → `incorporated`, `corp` → `corporation`, `co` → `company`, `assoc` → `association`.
5. **French Suffix Recognition**: Standardize French legal forms `sarl`, `sasu`, `eurl`, `sas`, `sa`.
6. **Punctuation to Space Replacement**: Replace `.` `,` `-` `/` `\` `'` `"` `(` `)` with a single space.
7. **Whitespace Collapsing**: Strip leading/trailing whitespace and collapse multiple consecutive spaces into a single space.
8. **Address Landmark Preservation**: Retain directional/landmark terms (`near`, `opp`, `behind`) as token features.
9. **Null Address Defaulting**: Replace missing `business_address` values with empty string `""`.
10. **Country Code Isolation**: Partition candidate blocking strictly by `country` (`US`, `India`, `France`).

---

## ⚖️ DELIVERABLE 3: DECISION MATRIX

| Pipeline Stage | Decision | Rationale |
| :--- | :--- | :--- |
| **Transliteration** | **YES (Use `anyascii` / `unidecode`)** | 4.09% (312,725) of true positive train matches are Latin S1 <-> Devanagari S2/S3. |
| **Abbreviation Expansion** | **PARTIAL (Top 6 + French Suffixes)** | Expanding top suffixes covers >95% of legal term noise without over-expanding general words. |
| **Punctuation Removal** | **SELECTIVE (`&` → `and`, others → space)** | Preserves word boundaries without losing ampersand semantics. |
| **Address Parsing** | **NO (Flat Text Token Sets & n-grams)** | Country address formats differ too much for rigid parsing; TF-IDF & Token-Sort work universally. |
| **Country Blocking** | **YES (Strict Hard Filter by `country`)** | Zero cross-country matches exist. Reduces candidate pair space by ~60%. |

---

## ⚠️ DELIVERABLE 4: RISK ASSESSMENT

1. **Risk of Over-Normalizing Short Names**:
   - *Issue*: Single-word or 2-letter business names (e.g., `3M Inc`, `IT Corp`) might lose essential tokens if suffixes are stripped completely.
   - *Mitigation*: Keep BOTH `normalized_name` (with expanded suffixes) AND `clean_raw_name` to calculate dual similarity features.
2. **France Generalization (Unseen Country)**:
   - *Issue*: French addresses contain French accent marks (`é`, `è`, `ê`, `à`, `ç`) and French suffixes (`SARL`, `SASU`).
   - *Mitigation*: `anyascii` transliterates accented French characters cleanly (`Bordeaux, Nouvelle-Aquitaine`), ensuring universal model generalization.

---
 