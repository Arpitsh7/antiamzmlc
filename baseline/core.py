# Entity Resolution Pipeline for Business Matching - shared core
# =================================================
# This module holds all pipeline machinery. Entry points:
#   model_scripts/baseline/train.py -> run_training()
#   model_scripts/baseline/test.py  -> run_test()
# Architecture:
# 1. Preprocessing: Normalize names and addresses
# 2. Blocking: Country + TF-IDF on concatenated name+address → top-K candidates
# 3. Feature Engineering: String similarity features (name, address)
# 4. Classification: XGBoost classifier trained on ground truth
# 5. Output: matching_results.tsv and candidate_pairs.tsv
#
# Memory-optimized (v1.1): every stage is bounded to run end-to-end in <=12 GiB.
# - Blocking keeps everything sparse: S1 chunks x S23 blocks sparse matmul with
#   per-row top-K merging (no dense NxM similarity matrix, which needed ~309 GiB).
# - TF-IDF is fitted on a bounded sample and hyper-frequent n-grams are dropped
#   (max_df) so sparse products stay small and fast.
# - Candidates are stored as compact CSR-style arrays (CandidateIndex) instead of
#   per-pair Python tuples.
# - Source TSVs are streamed in chunks; raw frames are never held alongside
#   preprocessed ones.
# - Feature/predict stages gather rows by position instead of building
#   dict-of-dicts lookups over the full pool.
# - mem_report() logs process/system memory at each phase boundary.

import pandas as pd
import numpy as np
import re
import os
import sys
import gc
import time
import pickle
import ctypes

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
import xgboost as xgb

# Anchor all paths (logs/, cache/, dataset/, output/) to the repo root as
# ABSOLUTE paths, so train.py / test.py can be launched from any working
# directory and always read/write the same cache folder.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE_DIR)
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ============================================================
# LOGGING SETUP — writes to BOTH terminal AND a log file
# ============================================================
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")


class TeeLogger:
    """Writes to both the terminal and a log file simultaneously.

    Every print() and error message appears in the terminal as usual,
    AND gets saved to logs/run_YYYYMMDD_HHMMSS.log automatically.
    """
    def __init__(self, log_path, original_stream, encoding="utf-8"):
        self.log_file = open(log_path, "a", encoding=encoding)
        self.original = original_stream
        self.encoding = encoding

    def write(self, message):
        # Write to terminal
        try:
            self.original.write(message)
        except UnicodeEncodeError:
            # Fallback for Windows terminals that can't handle Unicode
            self.original.write(message.encode("ascii", errors="replace").decode("ascii"))
        # Write to log file
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# Redirect stdout and stderr to TeeLogger
sys.stdout = TeeLogger(LOG_FILE, sys.stdout)
sys.stderr = TeeLogger(LOG_FILE, sys.stderr)

print(f"=== Logging to: {os.path.abspath(LOG_FILE)} ===")
print(f"=== Started at: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
print()

# ============================================================
# MEMORY REPORTING — verify we stay under the RAM budget
# ============================================================
GIB = 1024 ** 3

try:
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    _WIN_MEM = True
except Exception:
    _WIN_MEM = False

if _WIN_MEM:
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    ctypes.windll.kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
    ctypes.windll.kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL


def mem_report(tag):
    """Log this process's committed memory (and system headroom). Best effort."""
    try:
        if _WIN_MEM:
            pmc = _PROCESS_MEMORY_COUNTERS_EX()
            pmc.cb = ctypes.sizeof(pmc)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            priv = peak = None
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                priv, peak = pmc.PrivateUsage, pmc.PeakPagefileUsage
            ms = _MEMORYSTATUSEX()
            ms.dwLength = ctypes.sizeof(ms)
            sys_line = ""
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                sys_line = f", sys avail {ms.ullAvailPhys / GIB:.2f}/{ms.ullTotalPhys / GIB:.2f} GiB"
            if priv is not None:
                log(f"  [mem] {tag}: process {priv / GIB:.2f} GiB (peak {peak / GIB:.2f} GiB){sys_line}")
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            log(f"  [mem] {tag}: peak RSS {peak / GIB:.2f} GiB")
    except Exception as exc:
        log(f"  [mem] {tag}: unavailable ({exc})")


# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "cache")   # all caching happens here, e.g.
                                              # E:\...\student_resource\cache
TOP_K_CANDIDATES = 20          # top-K candidates from TF-IDF blocking per S1 entity
MATCH_THRESHOLD = 0.45         # XGBoost probability threshold for final matching
USE_VALIDATION = True          # train.py: 90/10 split -> F0.5 on held-out 10% (keep True)

# --- memory-bounded blocking knobs ---
CHUNK_S1 = 1000                # S1 rows per sparse similarity product (small => bounded RAM)
S1_TRANSFORM_CHUNK = 50000     # S1 docs transformed per TF-IDF call (multiple of CHUNK_S1)
S23_BLOCK_DOCS = 150000        # S2/S3 docs transformed per block (sklearn builds Python lists internally)
FIT_SAMPLE_MAX = 100000        # docs used to fit TF-IDF vocabulary/idf (bounded fit RAM)
MAX_DF = 0.02                  # drop n-grams present in >2% of the fit sample (bounds product size)
MIN_SIM = 0.05                 # minimum cosine similarity to keep a candidate

# --- streaming IO knobs ---
READ_CHUNK = 500000            # rows per read_csv chunk when building preprocessed frames
NEG_RATE = 0.25                # per-chunk sampling rate for training negatives (downsampled later)
NEG_MULTIPLE = 3               # keep at most NEG_MULTIPLE x matched negatives
PRED_CHUNK_ROWS = 20000        # S1 rows per inference feature chunk

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# CACHING HELPERS — save/load intermediate results to disk
# ============================================================
# This lets you re-run the pipeline without repeating expensive steps.
# Delete the cache/ folder to force a full re-run.

def cache_exists(name):
    """Check if a cached file exists."""
    return os.path.exists(os.path.join(CACHE_DIR, name))

def save_cache(obj, name):
    """Save an object to cache (pickle format)."""
    path = os.path.join(CACHE_DIR, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    log(f"  Cached '{name}' ({size_mb:.1f} MB)")

def load_cache(name):
    """Load an object from cache."""
    path = os.path.join(CACHE_DIR, name)
    log(f"  Loading cached '{name}'...")
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    log(f"  Loaded '{name}' successfully")
    return obj

def save_df_cache(df, name):
    """Save a DataFrame to cache (parquet format — smaller & faster than pickle)."""
    path = os.path.join(CACHE_DIR, name)
    df.to_parquet(path, index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    log(f"  Cached '{name}' ({size_mb:.1f} MB)")

def load_df_cache(name):
    """Load a DataFrame from cache."""
    path = os.path.join(CACHE_DIR, name)
    log(f"  Loading cached '{name}'...")
    df = pd.read_parquet(path)
    log(f"  Loaded '{name}': {len(df)} rows")
    return df


# Columns needed downstream; everything else is dropped after loading/preprocessing
# to keep pandas object columns (expensive) out of memory.
NORM_COLS = ["entity_id", "name_norm", "addr_norm", "country_norm", "blocking_text"]

def prune_norm_cols(df, tag=""):
    extra = [c for c in df.columns if c not in NORM_COLS]
    if extra:
        df = df.drop(columns=extra)
        if tag:
            log(f"  Dropped {len(extra)} non-essential column(s) from {tag}")
    return df


# ============================================================
# STEP 1: PREPROCESSING
# ============================================================

# Common abbreviation mappings for business names
NAME_ABBREV = {
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bltd\b': 'limited',
    r'\bco\b': 'company',
    r'\bllc\b': 'limited liability company',
    r'\bllp\b': 'limited liability partnership',
    r'\bpvt\b': 'private',
    r'\bpvt\.?\s*ltd\b': 'private limited',
    r'\b&\b': 'and',
    r'\bdba\b': '',
    r'\bint\'?l\b': 'international',
    r'\bnatl\b': 'national',
    r'\bassoc\b': 'associates',
    r'\bsvcs?\b': 'services',
    r'\bmfg\b': 'manufacturing',
    r'\btech\b': 'technology',
    r'\bengr?g?\b': 'engineering',
    r'\bcomm\b': 'communications',
    r'\bgrp\b': 'group',
    r'\bentpr?\b': 'enterprises',
}

# Common address abbreviations
ADDR_ABBREV = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave?\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bfl\b': 'floor',
    r'\bflr\b': 'floor',
    r'\bbldg\b': 'building',
    r'\bn\b': 'north',
    r'\bs\b': 'south',
    r'\be\b': 'east',
    r'\bw\b': 'west',
    r'\bne\b': 'northeast',
    r'\bnw\b': 'northwest',
    r'\bse\b': 'southeast',
    r'\bsw\b': 'southwest',
    r'\bmt\b': 'mount',
}

# US state abbreviation mapping
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
    'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
    'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
    'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
}

# Indian state abbreviations
INDIAN_STATES = {
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam',
    'br': 'bihar', 'cg': 'chhattisgarh', 'ga': 'goa', 'gj': 'gujarat',
    'hr': 'haryana', 'hp': 'himachal pradesh', 'jk': 'jammu and kashmir',
    'jh': 'jharkhand', 'ka': 'karnataka', 'kl': 'kerala', 'mp': 'madhya pradesh',
    'mh': 'maharashtra', 'mn': 'manipur', 'ml': 'meghalaya', 'mz': 'mizoram',
    'nl': 'nagaland', 'or': 'odisha', 'pb': 'punjab', 'rj': 'rajasthan',
    'sk': 'sikkim', 'tn': 'tamil nadu', 'ts': 'telangana', 'tr': 'tripura',
    'up': 'uttar pradesh', 'uk': 'uttarakhand', 'wb': 'west bengal',
    'dl': 'delhi',
}


def normalize_text(text):
    """Basic text normalization: lowercase, strip accents where possible, remove extra whitespace."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower().strip()
    # Remove common unicode accents/diacritics but keep non-Latin scripts
    # Only strip Latin-based accents
    import unicodedata
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    # Remove combining marks only for Latin base chars
    result = []
    for i, ch in enumerate(text):
        if unicodedata.combining(ch):
            # Skip combining chars that follow Latin letters
            if i > 0 and ord(text[i-1]) < 0x0300:
                continue
        result.append(ch)
    text = ''.join(result)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name(name):
    """Normalize business name for matching."""
    text = normalize_text(name)
    if not text:
        return ""
    # Remove punctuation except alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Apply abbreviation expansions
    for pattern, replacement in NAME_ABBREV.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_address(address):
    """Normalize business address for matching."""
    text = normalize_text(address)
    if not text:
        return ""
    # Remove punctuation except alphanumeric, spaces, and common separators
    text = re.sub(r'[^\w\s,/\-#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Apply address abbreviation expansions - only for likely address abbreviations
    for pattern, replacement in ADDR_ABBREV.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_dataframe(df):
    """Add normalized columns to a raw frame, then drop the raw text columns.

    Vectorized (column-wise) instead of row-wise apply, and it never keeps the
    raw business_name/business_address/country columns alive afterwards —
    raw frames are several GiB at full scale.
    """
    if 'name_norm' in df.columns and 'blocking_text' in df.columns:
        return df  # already preprocessed
    log(f"  Preprocessing {len(df)} records...")
    df['name_norm'] = df['business_name'].apply(normalize_name)
    df['addr_norm'] = df['business_address'].apply(normalize_address)
    df['country_norm'] = df['country'].str.lower().str.strip().astype('category')
    # blocking_text = name repeated for higher weight + address (was create_blocking_key)
    df['blocking_text'] = df['name_norm'] + ' ' + df['name_norm'] + ' ' + df['addr_norm']
    raw_cols = [c for c in ('business_name', 'business_address', 'country') if c in df.columns]
    if raw_cols:
        df = df.drop(columns=raw_cols)
    return df


# ============================================================
# STREAMING SOURCE LOADER — bounded-RAM replacement for
# "read whole TSV into pandas, then filter/sample"
# ============================================================

def stream_source(path, *, wanted_ids=None, keep_all=False, neg_rate=0.0,
                  neg_multiple=None, neg_cap=None, seed=42, chunksize=READ_CHUNK,
                  max_chunks=None, tag=""):
    """Stream a source TSV in chunks, keeping only what is needed.

    - wanted_ids: collect rows whose entity_id is in this set (matched records).
    - neg_rate: additionally collect a random fraction of the remaining rows
      (training negatives); downsampled to neg_multiple*matched or neg_cap at the end.
    - keep_all: collect every row (used for the test set).

    Each collected chunk is preprocessed immediately, so raw text columns never
    accumulate. Returns a single preprocessed DataFrame.
    """
    rng = np.random.default_rng(seed)
    matched_parts, neg_parts, all_parts = [], [], []
    want_set = set(wanted_ids) if wanted_ids is not None else None
    n_read = n_matched = n_neg = 0
    t0 = time.time()

    for n_chunks, chunk in enumerate(pd.read_csv(path, sep="\t", chunksize=chunksize)):
        n_read += len(chunk)
        if keep_all:
            all_parts.append(preprocess_dataframe(chunk))
        else:
            is_match = chunk['entity_id'].isin(want_set) if want_set else None
            if is_match is not None and is_match.any():
                matched_parts.append(preprocess_dataframe(chunk[is_match].copy()))
                n_matched += int(is_match.sum())
            if neg_rate > 0 and want_set is not None:
                sel = (~is_match) & (rng.random(len(chunk)) < neg_rate)
                if sel.any():
                    neg_parts.append(preprocess_dataframe(chunk[sel].copy()))
                    n_neg += int(sel.sum())
        if max_chunks is not None and n_chunks + 1 >= max_chunks:
            log(f"  [{tag}] stopped early after {n_read} rows (max_chunks={max_chunks})")
            break
        if (n_chunks + 1) % 10 == 0:
            log(f"  [{tag}] streamed {n_read:,} rows ({time.time() - t0:.0f}s)")

    if keep_all:
        pool = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
        del all_parts
    else:
        matched = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
        negs = pd.concat(neg_parts, ignore_index=True) if neg_parts else pd.DataFrame()
        del matched_parts, neg_parts
        keep = len(negs)
        if neg_multiple is not None:
            keep = min(keep, neg_multiple * max(len(matched), 1))
        if neg_cap is not None:
            keep = min(keep, neg_cap)
        if 0 < keep < len(negs):
            negs = negs.sample(n=keep, random_state=seed)
        log(f"  [{tag}] matched: {len(matched):,}, negatives kept: {len(negs):,} (streamed {n_read:,})")
        pool = pd.concat([matched, negs], ignore_index=True) if len(negs) else matched
        del matched, negs
    gc.collect()
    log(f"  [{tag}] pool size: {len(pool):,} rows ({time.time() - t0:.0f}s total)")
    return pool


# ============================================================
# GROUND TRUTH HELPERS
# ============================================================

def build_gt_matches(gt_df):
    """dict: source1_entity_id -> set(matched_entity_ids)."""
    result = {}
    m = gt_df['matched_entity_ids']
    mask = m.notna()
    if mask.any():
        s1_arr = gt_df.loc[mask, 'source1_entity_id'].to_numpy(dtype=object)
        m_arr = m[mask].to_numpy(dtype=object)
        for s1_id, ids in zip(s1_arr, m_arr):
            result[s1_id] = set(str(ids).split(','))
    return result


def gt_all_matched_ids(gt_df):
    """Set of every S2/S3 entity id referenced by the ground truth."""
    m = gt_df['matched_entity_ids']
    m = m[m.notna()]
    if len(m) == 0:
        return set()
    ids = m.str.split(',').explode()
    return set(ids.tolist())


# ============================================================
# STEP 2: BLOCKING (sparse TF-IDF candidate generation)
# ============================================================

class CandidateIndex:
    """Memory-compact candidate set, CSR-style.

    Row i corresponds to s1_ids[i]; its candidates are
    s23_pos[indptr[i]:indptr[i+1]] — positions into the S2/S3 pool DataFrame
    (same row order the index was built with) — with similarity scores.
    """
    __slots__ = ("s1_ids", "indptr", "s23_pos", "scores", "_row_of")

    def __init__(self, s1_ids, indptr, s23_pos, scores):
        self.s1_ids = np.asarray(s1_ids, dtype=object)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.s23_pos = np.asarray(s23_pos, dtype=np.int32)
        self.scores = np.asarray(scores, dtype=np.float32)
        self._row_of = None

    def __len__(self):
        return len(self.s1_ids)

    @property
    def n_pairs(self):
        return int(len(self.s23_pos))

    def row_of(self, s1_id):
        if self._row_of is None:
            self._row_of = {eid: i for i, eid in enumerate(self.s1_ids.tolist())}
        return self._row_of.get(s1_id, -1)

    def slice_for(self, i):
        return int(self.indptr[i]), int(self.indptr[i + 1])

    def __getstate__(self):
        return (self.s1_ids, self.indptr, self.s23_pos, self.scores)

    def __setstate__(self, state):
        self.__init__(*state)


def _topk_merge(prev_scores, prev_pos, new_scores, new_pos, top_k):
    """Merge a row's running top-K with a new block's scores (both score-desc)."""
    if prev_scores is not None and len(prev_scores):
        s = np.concatenate((prev_scores, new_scores))
        p = np.concatenate((prev_pos, new_pos))
    else:
        # copy: new_scores may be a view into the block matrix we are about to free
        s = np.array(new_scores)
        p = new_pos
    if len(s) > top_k:
        keep = np.argpartition(s, len(s) - top_k)[len(s) - top_k:]
        s = s[keep]
        p = p[keep]
    order = np.argsort(s)[::-1]
    return s[order], p[order]


def _prune_csr_below(sim, thresh):
    """Drop entries <= thresh from a CSR matrix, row counts recomputed vectorized.

    Top-K per row is unchanged by this (entries <= thresh can never make the
    top-K), and it keeps the per-row argpartition cheap when a row shares a
    hyper-common n-gram with tens of thousands of pool docs.
    """
    data = sim.data
    keep = data > thresh
    if keep.all():
        return sim
    cum = np.concatenate(([0], np.cumsum(keep, dtype=np.int32)))
    counts = (cum[sim.indptr[1:]] - cum[sim.indptr[:-1]]).astype(np.int64)
    new_indptr = np.zeros(sim.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=new_indptr[1:])
    return csr_matrix((data[keep], sim.indices[keep], new_indptr), shape=sim.shape)


def generate_candidates_tfidf(s1_df, s23_df, top_k=TOP_K_CANDIDATES, chunk_rows=CHUNK_S1,
                              block_docs=S23_BLOCK_DOCS, fit_sample=FIT_SAMPLE_MAX,
                              max_df=MAX_DF, min_sim=MIN_SIM):
    """
    Generate candidate pairs using TF-IDF cosine similarity, blocked by country.

    Memory-safe version: the similarity matrix is never materialized densely.
    For every (S1 chunk, S23 block) pair we compute the sparse product
    M_block @ chunk.T (both L2-normalized, so entries are cosine similarities),
    extract each row's top-K, and keep a running global top-K per S1 entity.
    TF-IDF is fitted on a bounded sample; max_df drops hyper-frequent n-grams,
    which keeps the sparse products small AND fast.
    Returns a CandidateIndex.
    """
    log("Generating candidates using TF-IDF blocking by country...")
    log(f"  knobs: chunk_rows={chunk_rows}, block_docs={block_docs}, fit_sample={fit_sample}, max_df={max_df}")
    t0 = time.time()
    n_s1 = len(s1_df)
    s1_ids = s1_df['entity_id'].to_numpy(dtype=object)
    best_scores = [None] * n_s1
    best_pos = [None] * n_s1
    s1_ctry = s1_df['country_norm']
    s23_ctry = s23_df['country_norm']

    piece_size = max(chunk_rows, (S1_TRANSFORM_CHUNK // chunk_rows) * chunk_rows)

    for country in [c for c in s1_ctry.dropna().unique()]:
        s1_rows = np.flatnonzero((s1_ctry == country).to_numpy())
        s23_rows = np.flatnonzero((s23_ctry == country).to_numpy())
        log(f"  Processing country: {country} (S1: {len(s1_rows):,}, S23: {len(s23_rows):,})")
        if len(s1_rows) == 0 or len(s23_rows) == 0:
            continue

        # --- fit TF-IDF on a bounded sample of the S2/S3 pool ---
        if len(s23_rows) > fit_sample:
            rng = np.random.default_rng(42)
            fit_rows = s23_rows[np.sort(rng.choice(len(s23_rows), size=fit_sample, replace=False))]
        else:
            fit_rows = s23_rows
        tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=500000,
            sublinear_tf=True,
            min_df=2,
            max_df=max_df,
            dtype=np.float32,
        )
        log(f"    Fitting TF-IDF on {len(fit_rows):,} S2/S3 docs...")
        fit_texts = s23_df['blocking_text'].iloc[fit_rows].to_numpy(dtype=object, na_value="")
        tfidf.fit(fit_texts)
        del fit_texts
        log(f"    Vocabulary: {len(tfidf.vocabulary_):,} features")
        mem_report(f"after TF-IDF fit ({country})")

        # --- transform S1 once, in bounded pieces (row-sliced later per chunk) ---
        pieces = []
        for p0 in range(0, len(s1_rows), piece_size):
            p1 = min(p0 + piece_size, len(s1_rows))
            texts = s1_df['blocking_text'].iloc[s1_rows[p0:p1]].to_numpy(dtype=object, na_value="")
            pieces.append(tfidf.transform(texts))

        # --- stream S2/S3 in blocks, products against every S1 chunk ---
        n_blocks = (len(s23_rows) + block_docs - 1) // block_docs
        for bi in range(n_blocks):
            b0, b1 = bi * block_docs, min((bi + 1) * block_docs, len(s23_rows))
            block_pos = s23_rows[b0:b1]
            btexts = s23_df['blocking_text'].iloc[block_pos].to_numpy(dtype=object, na_value="")
            m_block = tfidf.transform(btexts)
            del btexts
            log(f"    Block {bi + 1}/{n_blocks}: {len(block_pos):,} S23 docs, nnz={m_block.nnz:,}")

            max_sim_nnz = 0
            for ci in range(0, len(s1_rows), chunk_rows):
                cj = min(ci + chunk_rows, len(s1_rows))
                p_idx = ci // piece_size
                off = ci - p_idx * piece_size
                chunk_t = pieces[p_idx][off:off + (cj - ci)].T.tocsr()  # (vocab x chunk)
                sim_t = m_block @ chunk_t                               # (block x chunk) sparse
                del chunk_t
                sim = sim_t.T.tocsr()                                   # (chunk x block) sparse
                del sim_t
                sim = _prune_csr_below(sim, min_sim)
                max_sim_nnz = max(max_sim_nnz, int(sim.nnz))
                indptr, data, indices = sim.indptr, sim.data, sim.indices
                for r in range(cj - ci):
                    a, b = int(indptr[r]), int(indptr[r + 1])
                    if a == b:
                        continue
                    gi = int(s1_rows[ci + r])
                    best_scores[gi], best_pos[gi] = _topk_merge(
                        best_scores[gi], best_pos[gi],
                        data[a:b], block_pos[indices[a:b]], top_k,
                    )
                del sim
            del m_block
            log(f"    Block {bi + 1}/{n_blocks} done (max chunk-product nnz: {max_sim_nnz:,})")
            mem_report(f"after block {bi + 1}/{n_blocks} ({country})")
        del pieces
        gc.collect()

    # --- assemble the compact index (applying the minimum-similarity filter) ---
    lens = np.array([0 if s is None else len(s) for s in best_scores], dtype=np.int64)
    if lens.sum() > 0:
        scores_flat = np.concatenate([s for s in best_scores if s is not None])
        pos_flat = np.concatenate([p for p in best_pos if p is not None])
        row_ids = np.repeat(np.arange(n_s1, dtype=np.int64), lens)
        keep = scores_flat > min_sim
        scores_flat = scores_flat[keep]
        pos_flat = pos_flat[keep]
        row_ids = row_ids[keep]
        counts = np.bincount(row_ids, minlength=n_s1).astype(np.int64)
        del row_ids, keep
    else:
        scores_flat = np.zeros(0, dtype=np.float32)
        pos_flat = np.zeros(0, dtype=np.int32)
        counts = np.zeros(n_s1, dtype=np.int64)
    del best_scores, best_pos, lens
    indptr = np.zeros(n_s1 + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    del counts

    index = CandidateIndex(s1_ids, indptr, pos_flat.astype(np.int32), scores_flat)
    log(f"  Total candidates generated: {index.n_pairs:,}")
    log(f"  Avg candidates per S1: {index.n_pairs / max(n_s1, 1):.1f}")
    log(f"  Blocking took {time.time() - t0:.0f}s")
    mem_report("after blocking")
    return index


# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================

def compute_features(s1_row, s23_row):
    """Compute pairwise features between an S1 record and an S2/S3 record."""
    features = {}

    name1 = s1_row.get('name_norm', '')
    name2 = s23_row.get('name_norm', '')
    addr1 = s1_row.get('addr_norm', '')
    addr2 = s23_row.get('addr_norm', '')

    # Name similarity features
    features['name_ratio'] = fuzz.ratio(name1, name2) / 100.0
    features['name_partial_ratio'] = fuzz.partial_ratio(name1, name2) / 100.0
    features['name_token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100.0
    features['name_token_set'] = fuzz.token_set_ratio(name1, name2) / 100.0
    features['name_WRatio'] = fuzz.WRatio(name1, name2) / 100.0

    # Address similarity features
    features['addr_ratio'] = fuzz.ratio(addr1, addr2) / 100.0
    features['addr_partial_ratio'] = fuzz.partial_ratio(addr1, addr2) / 100.0
    features['addr_token_sort'] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    features['addr_token_set'] = fuzz.token_set_ratio(addr1, addr2) / 100.0
    features['addr_WRatio'] = fuzz.WRatio(addr1, addr2) / 100.0

    # Name token overlap features
    tokens1 = set(name1.split()) if name1 else set()
    tokens2 = set(name2.split()) if name2 else set()
    if tokens1 and tokens2:
        intersection = tokens1 & tokens2
        features['name_jaccard'] = len(intersection) / len(tokens1 | tokens2)
        features['name_overlap_min'] = len(intersection) / min(len(tokens1), len(tokens2))
        features['name_overlap_max'] = len(intersection) / max(len(tokens1), len(tokens2))
    else:
        features['name_jaccard'] = 0.0
        features['name_overlap_min'] = 0.0
        features['name_overlap_max'] = 0.0

    # Address token overlap features
    addr_tokens1 = set(addr1.split()) if addr1 else set()
    addr_tokens2 = set(addr2.split()) if addr2 else set()
    if addr_tokens1 and addr_tokens2:
        addr_intersection = addr_tokens1 & addr_tokens2
        features['addr_jaccard'] = len(addr_intersection) / len(addr_tokens1 | addr_tokens2)
        features['addr_overlap_min'] = len(addr_intersection) / min(len(addr_tokens1), len(addr_tokens2))
    else:
        features['addr_jaccard'] = 0.0
        features['addr_overlap_min'] = 0.0

    # Length features
    features['name_len_diff'] = abs(len(name1) - len(name2)) / max(len(name1), len(name2), 1)
    features['addr_len_diff'] = abs(len(addr1) - len(addr2)) / max(len(addr1), len(addr2), 1)

    # Empty field indicators
    features['name1_empty'] = 1.0 if not name1 else 0.0
    features['name2_empty'] = 1.0 if not name2 else 0.0
    features['addr1_empty'] = 1.0 if not addr1 else 0.0
    features['addr2_empty'] = 1.0 if not addr2 else 0.0

    # Country match (should always be 1 with blocking, but useful as sanity check)
    features['country_match'] = 1.0 if s1_row.get('country_norm', '') == s23_row.get('country_norm', '') else 0.0

    # First word of name match
    first1 = name1.split()[0] if name1.split() else ""
    first2 = name2.split()[0] if name2.split() else ""
    features['first_word_match'] = 1.0 if first1 and first2 and first1 == first2 else 0.0
    features['first_word_sim'] = fuzz.ratio(first1, first2) / 100.0 if first1 and first2 else 0.0

    # Number extraction from address - useful for matching house/building numbers
    nums1 = set(re.findall(r'\b\d+\b', addr1))
    nums2 = set(re.findall(r'\b\d+\b', addr2))
    if nums1 and nums2:
        features['num_jaccard'] = len(nums1 & nums2) / len(nums1 | nums2)
    elif not nums1 and not nums2:
        features['num_jaccard'] = 1.0
    else:
        features['num_jaccard'] = 0.0

    # Source indicator (S2 vs S3)
    s23_id = s23_row.get('entity_id', '')
    features['is_s3'] = 1.0 if str(s23_id).startswith('S3') else 0.0

    return features


FEATURE_NAMES = list(compute_features(
    {'name_norm': 'test', 'addr_norm': 'test', 'country_norm': 'us', 'entity_id': 'S1-1'},
    {'name_norm': 'test', 'addr_norm': 'test', 'country_norm': 'us', 'entity_id': 'S2-1'}
).keys())


def compute_features_batch(s1_records, s23_records):
    """Compute features for a batch of pairs."""
    all_features = []
    for s1_row, s23_row in zip(s1_records, s23_records):
        feat = compute_features(s1_row, s23_row)
        all_features.append(feat)
    return pd.DataFrame(all_features)


def compute_pair_features(s1_df, s23_df, s1_idx, s23_idx):
    """Features for pairs given as positions into s1_df and s23_df.

    Gathers only the needed rows, so no dict-of-dicts lookup over the full pool
    is ever built (that alone was ~8-10 GB at test scale).
    """
    names1 = s1_df['name_norm'].iloc[s1_idx].to_numpy(dtype=object, na_value="")
    addrs1 = s1_df['addr_norm'].iloc[s1_idx].to_numpy(dtype=object, na_value="")
    ctry1 = s1_df['country_norm'].iloc[s1_idx].to_numpy(dtype=object, na_value="")
    names2 = s23_df['name_norm'].iloc[s23_idx].to_numpy(dtype=object, na_value="")
    addrs2 = s23_df['addr_norm'].iloc[s23_idx].to_numpy(dtype=object, na_value="")
    ctry2 = s23_df['country_norm'].iloc[s23_idx].to_numpy(dtype=object, na_value="")
    ids2 = s23_df['entity_id'].iloc[s23_idx].to_numpy(dtype=object, na_value="")
    rows1 = [{'name_norm': n, 'addr_norm': a, 'country_norm': c}
             for n, a, c in zip(names1, addrs1, ctry1)]
    rows2 = [{'name_norm': n, 'addr_norm': a, 'country_norm': c, 'entity_id': e}
             for n, a, c, e in zip(names2, addrs2, ctry2, ids2)]
    return compute_features_batch(rows1, rows2)


# ============================================================
# STEP 4: TRAINING
# ============================================================

def prepare_training_data(s1_df, s23_df, gt_df, cand_index, sample_neg_ratio=3):
    """
    Prepare training data from ground truth and candidates.
    Positives: ground-truth matches present in the S2/S3 pool (regardless of candidacy).
    Negatives: top-scoring candidates that are not true matches.
    """
    log("Preparing training data...")
    s23_ids = s23_df['entity_id'].to_numpy(dtype=object)
    s23_index = pd.Index(s23_ids)
    gt_matches = build_gt_matches(gt_df)
    empty = frozenset()

    pos_pairs = []   # (s1_row_index, matched_id) — ids resolved in one batch below
    neg_rows = []
    neg_pos = []
    for i in range(len(cand_index)):
        s1_id = cand_index.s1_ids[i]
        true_matches = gt_matches.get(s1_id, empty)
        for m_id in true_matches:
            pos_pairs.append((i, m_id))
        a, b = cand_index.slice_for(i)
        if b > a:
            cand_pos = cand_index.s23_pos[a:b].tolist()
            if true_matches:
                neg_avail = [p for p in cand_pos if s23_ids[p] not in true_matches]
            else:
                neg_avail = cand_pos
            n_neg = min(len(neg_avail), max(sample_neg_ratio * len(true_matches), 2))
            for p in neg_avail[:n_neg]:
                neg_rows.append(i)
                neg_pos.append(p)
        if (i + 1) % 50000 == 0:
            log(f"  Processed {i + 1:,} S1 entities, pos: {len(pos_pairs):,}, neg: {len(neg_rows):,}")

    if pos_pairs:
        rows_arr, m_ids = zip(*pos_pairs)
        m_pos = s23_index.get_indexer(np.asarray(m_ids, dtype=object))
        ok = m_pos >= 0
        pos_rows = np.asarray(rows_arr, dtype=np.int64)[ok]
        pos_cols = m_pos[ok].astype(np.int64)
        del rows_arr, m_ids, m_pos, ok
    else:
        pos_rows = np.zeros(0, dtype=np.int64)
        pos_cols = np.zeros(0, dtype=np.int64)
    del pos_pairs, s23_index
    neg_rows = np.asarray(neg_rows, dtype=np.int64)
    neg_pos = np.asarray(neg_pos, dtype=np.int64)

    log(f"  Total positives: {len(pos_rows):,}")
    log(f"  Total negatives: {len(neg_rows):,}")

    pair_s1 = np.concatenate([pos_rows, neg_rows])
    pair_s23 = np.concatenate([pos_cols, neg_pos])
    y = np.concatenate([np.ones(len(pos_rows), dtype=np.int32),
                        np.zeros(len(neg_rows), dtype=np.int32)])
    del pos_rows, pos_cols, neg_rows, neg_pos

    log("  Computing features for training pairs...")
    chunk_size = 100000
    feats = []
    if len(pair_s1) == 0:
        X = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        log(f"  Training data shape: {X.shape}, positive ratio: {y.mean():.3f}")
        return X, y
    for i0 in range(0, len(pair_s1), chunk_size):
        i1 = min(i0 + chunk_size, len(pair_s1))
        feats.append(compute_pair_features(s1_df, s23_df, pair_s1[i0:i1], pair_s23[i0:i1]))
        if (i1 // chunk_size) % 5 == 0 or i1 == len(pair_s1):
            log(f"    Features computed for {i1:,}/{len(pair_s1):,}")
    X = pd.concat(feats, ignore_index=True)[FEATURE_NAMES].to_numpy(dtype=np.float32)
    del feats
    gc.collect()

    log(f"  Training data shape: {X.shape}, positive ratio: {y.mean():.3f}")
    mem_report("after feature extraction")
    return X, y


def train_model(X, y):
    """Train XGBoost classifier."""
    log("Training XGBoost model...")

    # Calculate scale_pos_weight for imbalanced classes
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
    )

    model.fit(X, y, verbose=True)

    # Feature importance
    importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
    log("Feature importance (top 10):")
    for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        log(f"  {name}: {imp:.4f}")

    return model


# ============================================================
# STEP 5: INFERENCE
# ============================================================

def predict_matches(model, s1_df, s23_df, cand_index, threshold=MATCH_THRESHOLD):
    """Score all candidate pairs; return matched pairs as (s1_row_index, s23_pos) arrays."""
    log(f"Running inference with threshold={threshold}...")
    n = len(cand_index)
    matched_rows = []
    matched_pos = []
    chunk = PRED_CHUNK_ROWS
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        a, b = int(cand_index.indptr[i0]), int(cand_index.indptr[i1])
        if b > a:
            pos = cand_index.s23_pos[a:b]
            counts = np.diff(cand_index.indptr[i0:i1 + 1])
            rows = np.repeat(np.arange(i0, i1, dtype=np.int64), counts)
            feats = compute_pair_features(s1_df, s23_df, rows, pos)
            X = feats[FEATURE_NAMES].to_numpy(dtype=np.float32)
            del feats
            probs = model.predict_proba(X)[:, 1]
            del X
            sel = probs >= threshold
            if sel.any():
                matched_rows.append(rows[sel])
                matched_pos.append(pos[sel])
        if ((i1 // chunk) % 10 == 0) or i1 == n:
            log(f"  Processed {i1:,}/{n:,} S1 rows")
            if (i1 // chunk) % 50 == 0:
                mem_report(f"inference {i1:,}/{n:,}")

    if matched_rows:
        rows_arr = np.concatenate(matched_rows)
        pos_arr = np.concatenate(matched_pos).astype(np.int64)
    else:
        rows_arr = np.zeros(0, dtype=np.int64)
        pos_arr = np.zeros(0, dtype=np.int64)
    del matched_rows, matched_pos

    total_matches = len(rows_arr)
    matched_entities = int(np.unique(rows_arr).size) if total_matches else 0
    log(f"  Total matches: {total_matches:,}")
    log(f"  S1 entities with matches: {matched_entities:,}/{n:,}")
    mem_report("after inference")
    return rows_arr, pos_arr


# ============================================================
# STEP 6: OUTPUT
# ============================================================

def write_output(cand_index, matched_rows, matched_pos, s23_df, output_dir=OUTPUT_DIR):
    """Write matching_results.tsv and candidate_pairs.tsv (LF line endings)."""
    log("Writing output files...")
    n = len(cand_index)
    s1_ids = cand_index.s1_ids
    s23_ids_ser = s23_df['entity_id']

    # Group matched pairs by S1 row (sorted arrays, no dict-of-lists)
    if len(matched_rows):
        order = np.argsort(matched_rows, kind='stable')
        rows_sorted = matched_rows[order]
        match_ids = s23_ids_ser.iloc[matched_pos[order]].to_numpy(dtype=object, na_value="")
        starts = np.searchsorted(rows_sorted, np.arange(n), side='left')
        ends = np.searchsorted(rows_sorted, np.arange(n), side='right')
    else:
        match_ids = np.zeros(0, dtype=object)
        starts = np.zeros(n, dtype=np.int64)
        ends = np.zeros(n, dtype=np.int64)

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    with open(matching_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for i in range(n):
            seg = match_ids[starts[i]:ends[i]]
            f.write(f"{s1_ids[i]}\t{','.join(seg.tolist()) if len(seg) else ''}\n")
    log(f"  Written {matching_path}")

    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    with open(candidate_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(n):
            a, b = cand_index.slice_for(i)
            if b > a:
                ids = s23_ids_ser.iloc[cand_index.s23_pos[a:b]].to_numpy(dtype=object, na_value="")
                f.write(f"{s1_ids[i]}\t{','.join(ids.tolist())}\n")
            else:
                f.write(f"{s1_ids[i]}\t\n")
    log(f"  Written {candidate_path}")
    mem_report("after writing output")


def pairs_to_dict(cand_index, matched_rows, matched_pos, s23_ids_ser):
    """Helper (validation-scale): dict s1_id -> list of matched s23 ids."""
    out = {}
    if len(matched_rows):
        order = np.argsort(matched_rows, kind='stable')
        rows_sorted = matched_rows[order]
        ids = s23_ids_ser.iloc[matched_pos[order]].to_numpy(dtype=object, na_value="")
        for r in np.unique(rows_sorted):
            a = np.searchsorted(rows_sorted, r, side='left')
            b = np.searchsorted(rows_sorted, r, side='right')
            out[str(cand_index.s1_ids[int(r)])] = ids[a:b].tolist()
    return out


# ============================================================
# STEP 7: EVALUATION (validation)
# ============================================================

def compute_f05_score(predictions, ground_truth):
    """Compute macro-averaged F0.5 score."""
    scores = []
    for s1_id in ground_truth:
        true_set = ground_truth[s1_id]
        pred_set = set(predictions.get(s1_id, []))

        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set and pred_set:
            scores.append(0.0)
        elif true_set and not pred_set:
            scores.append(0.0)
        else:
            tp = len(true_set & pred_set)
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(true_set) if true_set else 0.0
            if precision + recall == 0:
                scores.append(0.0)
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                scores.append(f05)

    return np.mean(scores)


# ============================================================
# MAIN PIPELINE
# ============================================================

TRAIN_CACHE_FILES = ("train_s1_sample.parquet", "train_s23_sample.parquet", "train_gt_sample.pkl")


def build_train_caches():
    """First run: stream the training TSVs, build and cache the preprocessed
    200k-S1 training sample, its S2/S3 pool, and the ground-truth subset."""
    log("Loading training S1 + ground truth (streamed build)...")
    train_s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t")
    train_gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t")
    log(f"  S1: {len(train_s1):,}, GT: {len(train_gt):,}")

    train_gt_train = train_gt
    train_s1_subset = train_s1
    if USE_VALIDATION:
        log("Creating validation split (10%)...")
        from sklearn.model_selection import train_test_split
        train_gt_train, train_gt_val = train_test_split(train_gt, test_size=0.1, random_state=42)
        val_s1_ids = set(train_gt_val['source1_entity_id'].values)
        train_s1_subset = train_s1[~train_s1['entity_id'].isin(val_s1_ids)]
        val_s1_subset = train_s1[train_s1['entity_id'].isin(val_s1_ids)]
        log(f"  Train S1: {len(train_s1_subset):,}, Val S1: {len(val_s1_subset):,}")
        # Preprocess + cache validation S1 and its GT lookup now, while raw frames exist
        val_s1_proc = preprocess_dataframe(val_s1_subset.copy())
        save_df_cache(val_s1_proc, "val_s1_proc.parquet")
        save_cache(train_gt_val, "val_gt.pkl")
        del val_s1_proc, val_s1_subset
        gc.collect()

    # --- sample S1 entities for training ---
    if len(train_s1_subset) > 200000:
        log(f"  Sampling 200000 S1 entities for training...")
        train_s1_sample = train_s1_subset.sample(n=200000, random_state=42)
    else:
        train_s1_sample = train_s1_subset
    train_gt_sample = train_gt_train[train_gt_train['source1_entity_id'].isin(train_s1_sample['entity_id'])]

    train_s1_sample = preprocess_dataframe(train_s1_sample)
    save_df_cache(train_s1_sample, "train_s1_sample.parquet")
    save_cache(train_gt_sample, "train_gt_sample.pkl")
    del train_s1, train_s1_subset, train_gt, train_gt_train, train_gt_sample
    gc.collect()

    # --- stream S2/S3 pool: all ground-truth-matched rows + sampled negatives ---
    gt_ids = gt_all_matched_ids(load_cache("train_gt_sample.pkl"))
    log(f"  GT references {len(gt_ids):,} distinct S2/S3 entities")
    pool_s2 = stream_source(os.path.join(TRAIN_DIR, "train_source2.tsv"),
                            wanted_ids=gt_ids, neg_rate=NEG_RATE, neg_multiple=NEG_MULTIPLE,
                            tag="train_s2")
    pool_s3 = stream_source(os.path.join(TRAIN_DIR, "train_source3.tsv"),
                            wanted_ids=gt_ids, neg_rate=NEG_RATE, neg_multiple=NEG_MULTIPLE,
                            tag="train_s3")
    del gt_ids
    train_s23_sample = pd.concat([pool_s2, pool_s3], ignore_index=True)
    del pool_s2, pool_s3
    gc.collect()
    log(f"  Training S2/S3 pool: {len(train_s23_sample):,} rows")
    save_df_cache(train_s23_sample, "train_s23_sample.parquet")
    mem_report("after building train caches")


def build_val_pool():
    """Validation-time S2/S3 pool: matched rows for the validation sample + 100k others."""
    val_gt = load_cache("val_gt.pkl")
    val_s1_proc = prune_norm_cols(load_df_cache("val_s1_proc.parquet"), "val_s1")
    val_sample = val_s1_proc.head(10000)
    val_gt_matches = build_gt_matches(val_gt)
    wanted = set()
    for s1_id in val_sample['entity_id'].values:
        wanted |= val_gt_matches.get(s1_id, frozenset())
    pool_s2 = stream_source(os.path.join(TRAIN_DIR, "train_source2.tsv"),
                            wanted_ids=wanted, neg_rate=0.02, neg_cap=100000, tag="val_s2")
    pool_s3 = stream_source(os.path.join(TRAIN_DIR, "train_source3.tsv"),
                            wanted_ids=wanted, neg_rate=0.02, neg_cap=100000, tag="val_s3")
    val_s23 = pd.concat([pool_s2, pool_s3], ignore_index=True)
    del pool_s2, pool_s3
    gc.collect()
    save_df_cache(val_s23, "val_s23_pool.parquet")
    return val_sample, val_s23, val_gt_matches


# ============================================================
# PIPELINE ENTRY POINTS
# ============================================================

def run_training():
    """Train + validate: build/load train caches, block, train XGBoost, then
    report F0.5 on the held-out 10% validation split.
    Never touches test data — that is test.py's job."""
    log("=" * 60)
    log("Entity Resolution Pipeline - TRAINING + VALIDATION")
    log("=" * 60)
    mem_report("start")

    have_train_cache = all(cache_exists(f) for f in TRAIN_CACHE_FILES)
    if not have_train_cache:
        log("Preprocessing training data (first run — will be cached)...")
        build_train_caches()
    elif USE_VALIDATION:
        # Raw S1/GT are only needed to build the validation artifacts
        log("Train caches present; preparing validation artifacts...")
        if not (cache_exists("val_s1_proc.parquet") and cache_exists("val_gt.pkl")):
            build_train_caches()  # same code path; train caches will be reused as-is
    else:
        log("Loading preprocessed training data from cache...")

    train_s1_sample = prune_norm_cols(load_df_cache("train_s1_sample.parquet"), "train_s1_sample")
    train_s23_sample = prune_norm_cols(load_df_cache("train_s23_sample.parquet"), "train_s23_sample")
    train_gt_sample = load_cache("train_gt_sample.pkl")
    log(f"  S1 sample: {len(train_s1_sample):,}, S2/S3 pool: {len(train_s23_sample):,}, GT: {len(train_gt_sample):,}")
    mem_report("after loading train caches")

    # =========================================
    # STEP B: Blocking on training data (CACHED)
    # =========================================
    if cache_exists("train_candidates.pkl"):
        train_candidates = load_cache("train_candidates.pkl")
    else:
        log("Running blocking on training data (will be cached)...")
        train_candidates = generate_candidates_tfidf(
            train_s1_sample, train_s23_sample, top_k=TOP_K_CANDIDATES
        )
        save_cache(train_candidates, "train_candidates.pkl")

    # Check blocking recall
    gt_matches_dict = build_gt_matches(train_gt_sample)
    s23_ids_np = train_s23_sample['entity_id'].to_numpy(dtype=object)
    total_true = 0
    found_in_candidates = 0
    for s1_id, true_matches in gt_matches_dict.items():
        i = train_candidates.row_of(s1_id)
        total_true += len(true_matches)
        if i >= 0:
            a, b = train_candidates.slice_for(i)
            cand_ids = set(s23_ids_np[train_candidates.s23_pos[a:b]].tolist())
            found_in_candidates += len(true_matches & cand_ids)

    blocking_recall = found_in_candidates / total_true if total_true > 0 else 0
    log(f"  Blocking recall: {blocking_recall:.4f} ({found_in_candidates}/{total_true})")
    del gt_matches_dict, s23_ids_np
    gc.collect()

    # =========================================
    # STEP C: Train model (CACHED)
    # =========================================
    if cache_exists("xgb_model.pkl"):
        log("Loading trained model from cache...")
        model = load_cache("xgb_model.pkl")
    else:
        log("Preparing training features and training model (will be cached)...")
        X_train, y_train = prepare_training_data(
            train_s1_sample, train_s23_sample, train_gt_sample, train_candidates
        )
        model = train_model(X_train, y_train)
        save_cache(model, "xgb_model.pkl")
        del X_train, y_train
        gc.collect()

    # Free training memory before validation
    del train_s1_sample, train_s23_sample, train_gt_sample, train_candidates
    gc.collect()
    mem_report("after training stage")

    # --- Validation ---
    if USE_VALIDATION:
        log("Running validation...")
        val_sample, val_s23, val_gt_matches = build_val_pool()
        val_candidates = generate_candidates_tfidf(val_sample, val_s23, top_k=TOP_K_CANDIDATES)
        val_rows, val_pos = predict_matches(model, val_sample, val_s23, val_candidates)
        val_preds = pairs_to_dict(val_candidates, val_rows, val_pos, val_s23['entity_id'])
        val_gt_subset = {s1_id: val_gt_matches.get(s1_id, set())
                         for s1_id in val_sample['entity_id'].values}
        f05 = compute_f05_score(val_preds, val_gt_subset)
        log(f"  Validation F0.5 Score: {f05:.4f}")
        del val_sample, val_s23, val_candidates, val_rows, val_pos, val_preds, val_gt_subset, val_gt_matches
        gc.collect()

    log("=" * 60)
    log("Training + validation complete. Trained model: cache/xgb_model.pkl")
    log("Run test.py to generate the submission files when ready.")
    log("=" * 60)
    mem_report("end")


def run_test():
    """Test inference: streamed preprocessing (cached) -> blocking (cached)
    -> scoring -> output/matching_results.tsv + output/candidate_pairs.tsv.
    Requires cache/xgb_model.pkl (produced by train.py)."""
    log("=" * 60)
    log("Entity Resolution Pipeline - TEST INFERENCE")
    log("=" * 60)
    mem_report("start")

    if not cache_exists("xgb_model.pkl"):
        log("ERROR: no trained model found at cache/xgb_model.pkl.")
        log("Run train.py first — it trains and caches the model.")
        return

    model = load_cache("xgb_model.pkl")

    test_cache_files = ("test_s1_proc.parquet", "test_s23_s2_proc.parquet", "test_s23_s3_proc.parquet")
    if not all(cache_exists(f) for f in test_cache_files):
        log("Preprocessing test data (streamed, will be cached)...")
        if not cache_exists("test_s1_proc.parquet"):
            save_df_cache(stream_source(os.path.join(TEST_DIR, "test_source1.tsv"),
                                        keep_all=True, tag="test_s1"), "test_s1_proc.parquet")
        if not cache_exists("test_s23_s2_proc.parquet"):
            save_df_cache(stream_source(os.path.join(TEST_DIR, "test_source2.tsv"),
                                        keep_all=True, tag="test_s2"), "test_s23_s2_proc.parquet")
        if not cache_exists("test_s23_s3_proc.parquet"):
            save_df_cache(stream_source(os.path.join(TEST_DIR, "test_source3.tsv"),
                                        keep_all=True, tag="test_s3"), "test_s23_s3_proc.parquet")

    test_s1 = prune_norm_cols(load_df_cache("test_s1_proc.parquet"), "test_s1")
    s2_proc = prune_norm_cols(load_df_cache("test_s23_s2_proc.parquet"), "test_s2")
    s3_proc = prune_norm_cols(load_df_cache("test_s23_s3_proc.parquet"), "test_s3")
    test_s23 = pd.concat([s2_proc, s3_proc], ignore_index=True)
    del s2_proc, s3_proc
    gc.collect()
    log(f"  Test S1: {len(test_s1):,}, Test S2/S3 pool: {len(test_s23):,}")
    mem_report("after loading test data")

    # Generate test candidates
    if cache_exists("test_candidates.pkl"):
        test_candidates = load_cache("test_candidates.pkl")
    else:
        log("Running blocking on test data (will be cached)...")
        test_candidates = generate_candidates_tfidf(
            test_s1, test_s23, top_k=TOP_K_CANDIDATES
        )
        save_cache(test_candidates, "test_candidates.pkl")

    # blocking_text is no longer needed once candidates exist
    if 'blocking_text' in test_s23.columns:
        test_s23 = test_s23.drop(columns=['blocking_text'])
    gc.collect()

    # Run inference
    test_rows, test_pos = predict_matches(
        model, test_s1, test_s23, test_candidates, threshold=MATCH_THRESHOLD
    )

    # Write output
    write_output(test_candidates, test_rows, test_pos, test_s23)

    log("=" * 60)
    log("Pipeline complete! Submissions written to output/")
    log("=" * 60)
    mem_report("end")


# ============================================================
# QUICK SUBMISSION PATH — exact-key blocking + trained model
# ============================================================

def build_exact_candidates(s1_df, s23_df, max_cands=10):
    """
    Fast blocking for a deadline submission: propose candidate pairs via exact
    joins on two cheap keys — (country, normalized name) and (country, sorted
    name tokens, which survives word-order noise) — instead of TF-IDF cosine.

    Much lower recall than TF-IDF blocking, but high precision, runs in
    minutes, and the pairs are scored by the SAME trained model (none of the
    26 features depend on how candidates were found). Returns (s1_pos, s23_pos).
    """
    log("Building exact-key blocking candidates (fast path)...")
    t0 = time.time()
    s1 = s1_df.reset_index(drop=True)
    s23 = s23_df.reset_index(drop=True)

    def keys_of(df):
        """Two exact keys per row; rows with empty names get no key (they can't
        be exact-matched, and a shared empty key would collide everything)."""
        ctry = df['country_norm'].astype(str)
        name = df['name_norm'].fillna("")
        has_name = name.str.len() > 0
        tok = df['name_norm'].str.split().map(lambda ws: " ".join(sorted(ws)) if ws else "")
        exact = (ctry + '|' + name).where(has_name, other="")
        tokkey = (ctry + '|' + tok).where(has_name & (tok.str.len() > 0), other="")
        return exact, tokkey

    s1_exact, s1_tok = keys_of(s1)
    log(f"  Built S1 keys ({time.time() - t0:.0f}s); building S2/S3 keys for {len(s23):,} rows...")
    s23_exact, s23_tok = keys_of(s23)
    log(f"  S2/S3 keys built ({time.time() - t0:.0f}s)")

    parts = []
    for key_name, k1, k2 in (("exact-name", s1_exact, s23_exact),
                             ("sorted-tokens", s1_tok, s23_tok)):
        left = pd.DataFrame({"s1_pos": np.arange(len(s1), dtype=np.int64), "key": k1})
        right = pd.DataFrame({"s23_pos": np.arange(len(s23), dtype=np.int64), "key": k2})
        left = left[left["key"].str.len() > 0]
        right = right[right["key"].str.len() > 0]
        m = left.merge(right, on="key", how="inner")
        m = m.groupby("s1_pos", sort=False).head(max_cands)
        parts.append(m[["s1_pos", "s23_pos"]])
        log(f"  Key '{key_name}': {len(m):,} pairs (capped at {max_cands}/S1)")
        del left, right, m

    allp = pd.concat(parts).drop_duplicates(["s1_pos", "s23_pos"])
    del parts
    s1_pos = allp["s1_pos"].to_numpy(dtype=np.int64)
    s23_pos = allp["s23_pos"].to_numpy(dtype=np.int64)
    del allp
    log(f"  Total quick candidates: {len(s1_pos):,} for {int(np.unique(s1_pos).size):,} S1 rows "
        f"({time.time() - t0:.0f}s)")
    return s1_pos, s23_pos


def candidates_from_pairs(s1_df, s1_pos, s23_pos, score=1.0):
    """Wrap raw pair arrays into the CandidateIndex format write_output expects."""
    n_s1 = len(s1_df)
    order = np.argsort(s1_pos, kind="stable")
    rows_sorted = s1_pos[order]
    pos_sorted = s23_pos[order]
    counts = np.bincount(rows_sorted, minlength=n_s1).astype(np.int64)
    indptr = np.zeros(n_s1 + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    scores = np.full(len(pos_sorted), score, dtype=np.float32)
    s1_ids = s1_df['entity_id'].to_numpy(dtype=object)
    return CandidateIndex(s1_ids, indptr, pos_sorted.astype(np.int32), scores)


def run_quick_test(threshold=MATCH_THRESHOLD):
    """Deadline path: exact-key blocking (minutes) + the trained model ->
    output/matching_results.tsv + output/candidate_pairs.tsv.
    Lower recall than run_test's TF-IDF blocking; same output format."""
    log("=" * 60)
    log("Entity Resolution Pipeline - QUICK SUBMISSION (exact-key blocking)")
    log("=" * 60)
    mem_report("start")

    if not cache_exists("xgb_model.pkl"):
        log("ERROR: no trained model found at cache/xgb_model.pkl.")
        log("Run train.py first — it trains and caches the model.")
        return

    model = load_cache("xgb_model.pkl")

    test_cache_files = ("test_s1_proc.parquet", "test_s23_s2_proc.parquet", "test_s23_s3_proc.parquet")
    if not all(cache_exists(f) for f in test_cache_files):
        log("ERROR: preprocessed test caches missing. Run test.py once (it builds")
        log("them) or wait for the running test.py to reach blocking stage.")
        return

    test_s1 = prune_norm_cols(load_df_cache("test_s1_proc.parquet"), "test_s1")
    s2_proc = prune_norm_cols(load_df_cache("test_s23_s2_proc.parquet"), "test_s2")
    s3_proc = prune_norm_cols(load_df_cache("test_s23_s3_proc.parquet"), "test_s3")
    test_s23 = pd.concat([s2_proc, s3_proc], ignore_index=True)
    del s2_proc, s3_proc
    gc.collect()
    log(f"  Test S1: {len(test_s1):,}, Test S2/S3 pool: {len(test_s23):,}")
    mem_report("after loading test data")

    s1_pos, s23_pos = build_exact_candidates(test_s1, test_s23)
    cand = candidates_from_pairs(test_s1, s1_pos, s23_pos)
    del s1_pos, s23_pos
    gc.collect()
    mem_report("after quick blocking")

    test_rows, test_pos = predict_matches(model, test_s1, test_s23, cand, threshold=threshold)
    write_output(cand, test_rows, test_pos, test_s23)

    log("=" * 60)
    log("QUICK submission complete! Copy output/*.tsv somewhere safe before")
    log("the full test.py run overwrites them later tonight.")
    log("=" * 60)
    mem_report("end")
