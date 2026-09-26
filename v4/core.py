# Entity Resolution Pipeline v4 - shared core
# =====================================================================
# Architecture (see ARCHITECTURE.md for full details):
#   1. Preprocessing: v1 normalization + ALL-Indic transliteration
#      (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam,
#      Oriya, Gurmukhi) via Unicode-name parsing.
#   2. Blocking: multi-key inverted index (exact name, sorted tokens,
#      phonetic token set, house-number + first-token phonetic) via
#      pandas merges. Minutes instead of the 10-12 h TF-IDF matmul.
#   3. Features: 34 pairwise features (v1's 26 + Levenshtein, char 3-gram
#      Jaccard, phonetic match/Jaccard, house-number and PIN match).
#   4. Model: XGBoost (GPU if available), threshold tuned on validation.
#   5. Output: same two submission TSVs as v1.
#
# Entry points: train.py -> run_training(), test.py -> run_test()

import pandas as pd
import numpy as np
import re
import os
import sys
import gc
import time
import pickle
import ctypes
import unicodedata
import multiprocessing as mp

from scipy.sparse import csr_matrix, vstack as sp_vstack

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
import xgboost as xgb

# Worker processes spawned by the parallel blocker re-import this module;
# ER_SPAWNED suppresses their logging redirect so only the parent logs.
_IS_SPAWNED_WORKER = os.environ.get("ER_SPAWNED") == "1"

try:
    from jellyfish import metaphone as _metaphone
except Exception:  # tiny fallback so the pipeline never hard-fails
    def _metaphone(s):
        return s[:2].upper() if s else ""

# Anchor all paths to the repo root as ABSOLUTE paths so the scripts can be
# launched from any working directory and always hit the same cache folder.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_BASE = os.path.join(REPO_ROOT, "6ab10eb3b23ba_student_resource", "student_resource")
if not os.path.exists(os.path.join(DATASET_BASE, "dataset")):
    DATASET_BASE = REPO_ROOT

BASE_DIR = REPO_ROOT
os.chdir(BASE_DIR)
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ============================================================
# LOGGING SETUP - writes to BOTH terminal AND a log file
# ============================================================
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"v4_run_{time.strftime('%Y%m%d_%H%M%S')}.log")


class TeeLogger:
    def __init__(self, log_path, original_stream, encoding="utf-8"):
        self.log_file = open(log_path, "a", encoding=encoding)
        self.original = original_stream

    def write(self, message):
        try:
            self.original.write(message)
        except UnicodeEncodeError:
            self.original.write(message.encode("ascii", errors="replace").decode("ascii"))
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.original.flush()
        self.log_file.flush()


sys.stdout = TeeLogger(LOG_FILE, sys.stdout) if not _IS_SPAWNED_WORKER else sys.stdout
sys.stderr = TeeLogger(LOG_FILE, sys.stderr) if not _IS_SPAWNED_WORKER else sys.stderr

if not _IS_SPAWNED_WORKER:
    print(f"=== Logging to: {os.path.abspath(LOG_FILE)} ===")
    print(f"=== Started at: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print()

# ============================================================
# MEMORY REPORTING
# ============================================================
GIB = 1024 ** 3

try:
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    _WIN_MEM = True
except Exception:
    _WIN_MEM = False

if _WIN_MEM:
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    ctypes.windll.kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
    ctypes.windll.kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL


def mem_report(tag):
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
TRAIN_DIR = os.path.join(DATASET_BASE, "dataset", "train")
TEST_DIR = os.path.join(DATASET_BASE, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "cache", "v4")   # v2 keeps its own cache namespace

MATCH_THRESHOLD = 0.45         # fallback if no tuned threshold is cached
THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
USE_VALIDATION = True          # 90/10 split -> F0.5 + threshold tuning

TRAIN_SAMPLE_SIZE = 200000
READ_CHUNK = 500000
NEG_RATE = 0.25
NEG_MULTIPLE = 3
PRED_CHUNK_ROWS = 20000

# multi-key blocking caps (per S1 entity, per key type)
CAP_EXACT = 20
CAP_SORTED = 20
CAP_PHON = 20
CAP_PHFULL = 20
CAP_BRANDEND = 15
CAP_ADDR = 10
CAP_ADDRPIN = 10
CAP_BRANDPIN = 10
CAP_PHPREFIX = 10
CAP_TOTAL = 60

# TF-IDF fuzzy blocking (the recall backbone; unioned with the keys below).
WORKERS = 2                    # RAM-safe parallelism for test scale
TFIDF_TOP_K = 50
TFIDF_MIN_SIM = 0.15           # Test-scale precision threshold (filters n-gram noise & caps RAM)
TFIDF_CHUNK = 500              # S1 rows per sparse product slice
TFIDF_BLOCK_DOCS = 50000       # 50k docs per block task (76 total blocks instead of 191)
TFIDF_FIT_SAMPLE = 100000
TFIDF_MAX_DF = 0.01            # Drops hyper-frequent non-discriminative n-grams

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# CACHE HELPERS (v2 namespace)
# ============================================================

def cache_exists(name):
    return os.path.exists(os.path.join(CACHE_DIR, name))

def save_cache(obj, name):
    path = os.path.join(CACHE_DIR, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  Cached '{name}' ({os.path.getsize(path) / (1024 * 1024):.1f} MB)")

def load_cache(name):
    log(f"  Loading cached '{name}'...")
    with open(os.path.join(CACHE_DIR, name), 'rb') as f:
        obj = pickle.load(f)
    log(f"  Loaded '{name}' successfully")
    return obj

def save_df_cache(df, name):
    path = os.path.join(CACHE_DIR, name)
    df.to_parquet(path, index=False)
    log(f"  Cached '{name}' ({os.path.getsize(path) / (1024 * 1024):.1f} MB)")

def load_df_cache(name):
    log(f"  Loading cached '{name}'...")
    df = pd.read_parquet(os.path.join(CACHE_DIR, name))
    log(f"  Loaded '{name}': {len(df)} rows")
    return df


NORM_COLS = ["entity_id", "name_norm", "addr_norm", "country_norm"]

def prune_norm_cols(df, tag=""):
    extra = [c for c in df.columns if c not in NORM_COLS]
    if extra:
        df = df.drop(columns=extra)
        if tag:
            log(f"  Dropped {len(extra)} non-essential column(s) from {tag}")
    return df


# ============================================================
# TRANSLITERATION - script-agnostic, via Unicode names
# ============================================================
# Every Indic character's Unicode name ends with its canonical sound:
#   "DEVANAGARI LETTER KHA" -> kha, "TAMIL VOWEL SIGN AA" -> aa,
#   "TELUGU LETTER TTA" -> tta. Parsing that suffix gives one table that
#   covers all nine Brahmic scripts in the data. Sibilants/liquids/length
#   are collapsed (sa/sha/ssa -> sha etc.) because we want matching
#   equivalence, not linguistics; the model separates precision later.

def _alias(sound):
    aliases = {
        'ssa': 'sha', 'sha': 'sha', 'sa': 'sa',
        'llla': 'la', 'lla': 'la', 'la': 'la',
        'rra': 'ra', 'rha': 'ra', 'ra': 'ra',
        'nna': 'na', 'na': 'na',
        'tta': 'ta', 'ta': 'ta', 'ttha': 'tha', 'tha': 'tha',
        'dda': 'da', 'da': 'da', 'ddha': 'dha', 'dha': 'dha',
        'khha': 'kha', 'ghha': 'gha', 'zza': 'ja', 'ffa': 'pha', 'qqa': 'ka',
        'vocalic r': 'ri', 'vocalic rr': 'ri', 'vocalic l': 'l', 'vocalic ll': 'l',
        'aa': 'a', 'ii': 'i', 'uu': 'u', 'oo': 'u',
    }
    return aliases.get(sound, sound)


def _latin_for(name):
    if not name:
        return ''
    if 'ANUSVARA' in name or 'CANDRABINDU' in name:
        return 'n'
    if 'VISARGA' in name:
        return 'h'
    if 'VIRAMA' in name or 'NUKTA' in name or 'AVAGRAHA' in name:
        return ''
    if 'LENGTH MARK' in name:
        return ''
    m = re.search(r'VOWEL SIGN ([A-Z ]+)$', name)
    if m:
        return _alias(m.group(1).strip().lower())
    m = re.search(r'LETTER ([A-Z ]+)$', name)
    if m:
        return _alias(m.group(1).strip().lower())
    if name.endswith(' OM'):
        return 'om'
    return ''


def _build_translit_table():
    table = {}
    for cp in list(range(0x0900, 0x0E00)) + [0x200C, 0x200D]:  # all Brahmic blocks + ZWJ/ZWNJ
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if 'DIGIT ' in name:
            try:
                table[cp] = str(unicodedata.digit(ch))
                continue
            except (ValueError, TypeError):
                pass
        table[cp] = _latin_for(name)
    return table


TRANSLIT_TABLE = _build_translit_table()


# ============================================================
# PREPROCESSING (v1 normalization + transliteration)
# ============================================================

NAME_ABBREV = {
    r'\bcorp\b': 'corporation', r'\binc\b': 'incorporated', r'\bltd\b': 'limited',
    r'\bco\b': 'company', r'\bllc\b': 'limited liability company',
    r'\bllp\b': 'limited liability partnership', r'\bpvt\b': 'private',
    r'\bpvt\.?\s*ltd\b': 'private limited', r'\b&\b': 'and', r'\bdba\b': '',
    r'\bint\'?l\b': 'international', r'\bnatl\b': 'national', r'\bassoc\b': 'associates',
    r'\bsvcs?\b': 'services', r'\bmfg\b': 'manufacturing', r'\btech\b': 'technology',
    r'\bengr?g?\b': 'engineering', r'\bcomm\b': 'communications', r'\bgrp\b': 'group',
    r'\bentpr?\b': 'enterprises',
}

ADDR_ABBREV = {
    r'\bst\b': 'street', r'\brd\b': 'road', r'\bave?\b': 'avenue', r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive', r'\bln\b': 'lane', r'\bct\b': 'court', r'\bpl\b': 'place',
    r'\bpkwy\b': 'parkway', r'\bhwy\b': 'highway', r'\bapt\b': 'apartment',
    r'\bste\b': 'suite', r'\bfl\b': 'floor', r'\bflr\b': 'floor', r'\bbldg\b': 'building',
    r'\bn\b': 'north', r'\bs\b': 'south', r'\be\b': 'east', r'\bw\b': 'west',
    r'\bne\b': 'northeast', r'\bnw\b': 'northwest', r'\bse\b': 'southeast',
    r'\bsw\b': 'southwest', r'\bmt\b': 'mount',
}


def normalize_text(text):
    """Lowercase, strip Latin accents, transliterate ALL Indic scripts, tidy whitespace."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower().strip()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(TRANSLIT_TABLE)   # Indic -> Latin (no-op for Latin text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name(name):
    text = normalize_text(name)
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    for pattern, replacement in NAME_ABBREV.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_address(address):
    text = normalize_text(address)
    if not text:
        return ""
    text = re.sub(r'[^\w\s,/\-#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    for pattern, replacement in ADDR_ABBREV.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_dataframe(df):
    """Add normalized columns to a raw frame, drop the raw text columns."""
    if 'name_norm' in df.columns and 'blocking_text' in df.columns:
        return df
    log(f"  Preprocessing {len(df)} records...")
    df['name_norm'] = df['business_name'].apply(normalize_name)
    df['addr_norm'] = df['business_address'].apply(normalize_address)
    df['country_norm'] = df['country'].str.lower().str.strip().astype('category')
    df['blocking_text'] = df['name_norm'] + ' ' + df['addr_norm']
    raw_cols = [c for c in ('business_name', 'business_address', 'country') if c in df.columns]
    if raw_cols:
        df = df.drop(columns=raw_cols)
    return df


# ============================================================
# STREAMING SOURCE LOADER
# ============================================================

def stream_source(path, *, wanted_ids=None, keep_all=False, neg_rate=0.0,
                  neg_multiple=None, neg_cap=None, seed=42, chunksize=READ_CHUNK,
                  tag=""):
    rng = np.random.default_rng(seed)
    matched_parts, neg_parts, all_parts = [], [], []
    want_set = set(wanted_ids) if wanted_ids is not None else None
    n_read = 0
    t0 = time.time()

    for n_chunks, chunk in enumerate(pd.read_csv(path, sep="\t", chunksize=chunksize)):
        n_read += len(chunk)
        if keep_all:
            all_parts.append(preprocess_dataframe(chunk))
        else:
            is_match = chunk['entity_id'].isin(want_set) if want_set else None
            if is_match is not None and is_match.any():
                matched_parts.append(preprocess_dataframe(chunk[is_match].copy()))
            if neg_rate > 0 and want_set is not None:
                sel = (~is_match) & (rng.random(len(chunk)) < neg_rate)
                if sel.any():
                    neg_parts.append(preprocess_dataframe(chunk[sel].copy()))
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
        log(f"  [{tag}] matched: {len(matched):,}, negatives kept: {len(negs):,}")
        pool = pd.concat([matched, negs], ignore_index=True) if len(negs) else matched
        del matched, negs
    gc.collect()
    log(f"  [{tag}] pool size: {len(pool):,} rows ({time.time() - t0:.0f}s total)")
    return pool


# ============================================================
# GROUND TRUTH HELPERS
# ============================================================

def build_gt_matches(gt_df):
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
    m = gt_df['matched_entity_ids']
    m = m[m.notna()]
    if len(m) == 0:
        return set()
    return set(m.str.split(',').explode().tolist())


# ============================================================
# PHONETICS (memoized metaphone)
# ============================================================

_PHON_CACHE = {}

def phon(token):
    v = _PHON_CACHE.get(token)
    if v is None:
        v = _metaphone(token)
        if not v:
            v = token[:2]
        _PHON_CACHE[token] = v
    return v


def house_number(addr):
    m = re.search(r'\d+', addr)
    return m.group(0) if m else ""


def longest_digit_run(addr):
    runs = re.findall(r'\d{4,}', addr)
    return max(runs, key=len) if runs else ""


# ============================================================
# BLOCKING - multi-key inverted index via pandas merges
# ============================================================

class CandidateIndex:
    """CSR-style candidate set: row i <-> s1_ids[i]; candidates of row i are
    s23_pos[indptr[i]:indptr[i+1]] (positions into the pool DataFrame)."""
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


def _join_keyed(left_pos, left_key, right_pos, right_key, cap, key_name, t0, max_key_freq=1000):
    ldf = pd.DataFrame({"s1_pos": left_pos, "key": left_key})
    rdf = pd.DataFrame({"s23_pos": right_pos, "key": right_key})
    ldf = ldf[ldf["key"].str.len() > 4]
    rdf = rdf[rdf["key"].str.len() > 4]
    if len(ldf) == 0 or len(rdf) == 0:
        return pd.DataFrame(columns=["s1_pos", "s23_pos"])

    # Drop hyper-frequent keys that cause combinatorial explosions (> max_key_freq in pool)
    freq = rdf["key"].value_counts()
    heavy_keys = freq[freq > max_key_freq].index
    if len(heavy_keys) > 0:
        heavy_set = set(heavy_keys)
        ldf = ldf[~ldf["key"].isin(heavy_set)]
        rdf = rdf[~rdf["key"].isin(heavy_set)]

    if len(ldf) == 0 or len(rdf) == 0:
        return pd.DataFrame(columns=["s1_pos", "s23_pos"])

    m = ldf.merge(rdf, on="key", how="inner")
    m = m.groupby("s1_pos", sort=False).head(cap)
    log(f"  Key '{key_name}': {len(m):,} pairs (cap {cap}/S1, {time.time() - t0:.0f}s)")
    return m[["s1_pos", "s23_pos"]]


# ============================================================
# TF-IDF FUZZY BLOCKING (parallel, block-level) - recall backbone
# ============================================================
# v1's char (2,4)-gram TF-IDF cosine blocking measured recall 0.917; keys
# alone stall at ~0.72 because exact/phonetic keys cannot catch extra-word
# (DBA) names and arbitrary typos. This restores that fuzzy net, but:
#   - runs on TRANSLITERATED text (name x2 + address), so India recall rises
#   - parallelized at BLOCK level (WORKERS processes; each worker transforms
#     its pool block and scores it against every S1 chunk via sparse products,
#     returning per-row top-K) - turns ~11 CPU-hours into ~1-1.5 h wall
#   - parent keeps a running per-row top-K via vectorized segment merges

_TF_STATE = {}


def _prune_csr_below(sim, thresh):
    data = sim.data
    keep = data > thresh
    if keep.all():
        return sim
    cum = np.concatenate(([0], np.cumsum(keep, dtype=np.int32)))
    counts = (cum[sim.indptr[1:]] - cum[sim.indptr[:-1]]).astype(np.int64)
    new_indptr = np.zeros(sim.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=new_indptr[1:])
    return csr_matrix((data[keep], sim.indices[keep], new_indptr), shape=sim.shape)


def _tf_worker_init(tfidf_path, piece_prefix, n_pieces, piece_size, n_voc, n_s1_country, params):
    with open(tfidf_path, 'rb') as f:
        _TF_STATE['tfidf'] = pickle.load(f)
    pieces = []
    for p in range(n_pieces):
        pieces.append((
            np.load(f"{piece_prefix}_{p}_data.npy", mmap_mode='r'),
            np.load(f"{piece_prefix}_{p}_indices.npy", mmap_mode='r'),
            np.load(f"{piece_prefix}_{p}_indptr.npy", mmap_mode='r'),
        ))
    _TF_STATE['pieces'] = pieces
    _TF_STATE['piece_size'] = piece_size
    _TF_STATE['n_voc'] = n_voc
    _TF_STATE['n_s1'] = n_s1_country
    _TF_STATE['params'] = params  # (top_k, min_sim, chunk_rows)


def _tf_slice_piece(p_idx, off, m):
    data, indices, indptr = _TF_STATE['pieces'][p_idx]
    a, b = int(indptr[off]), int(indptr[off + m])
    base = a
    return csr_matrix((np.asarray(data[a:b]), np.asarray(indices[a:b]),
                       np.asarray(indptr[off:off + m + 1]) - base),
                      shape=(m, _TF_STATE['n_voc']))


def _tf_block_task(task_path):
    with open(task_path, 'rb') as f:
        block_pos, texts = pickle.load(f)
    tfidf = _TF_STATE['tfidf']
    top_k, min_sim, chunk_rows = _TF_STATE['params']
    n_s1 = _TF_STATE['n_s1']
    m_block = tfidf.transform(texts)
    rows_out, pos_out, sc_out = [], [], []
    for ci in range(0, n_s1, chunk_rows):
        cj = min(ci + chunk_rows, n_s1)
        p_idx, off = ci // _TF_STATE['piece_size'], ci - (ci // _TF_STATE['piece_size']) * _TF_STATE['piece_size']
        chunk_t = _tf_slice_piece(p_idx, off, cj - ci).T.tocsr()
        sim_t = m_block @ chunk_t
        del chunk_t

        # In-place zeroing of low similarities to shrink memory before transpose
        mask = sim_t.data <= min_sim
        if mask.any():
            sim_t.data[mask] = 0
            sim_t.eliminate_zeros()

        sim = sim_t.T.tocsr()
        del sim_t

        indptr, data, indices = sim.indptr, sim.data, sim.indices
        task_top_k = min(top_k, 15)
        for r in range(cj - ci):
            a, b = int(indptr[r]), int(indptr[r + 1])
            if a == b:
                continue
            sc = data[a:b]
            ps = block_pos[indices[a:b]]
            if len(sc) > task_top_k:
                keep = np.argpartition(sc, len(sc) - task_top_k)[len(sc) - task_top_k:]
                sc = sc[keep]
                ps = ps[keep]
            rows_out.append(np.full(len(sc), ci + r, dtype=np.int32))
            pos_out.append(ps.astype(np.int32))
            sc_out.append(sc)
        del sim
    del m_block
    if rows_out:
        return (np.concatenate(rows_out), np.concatenate(pos_out), np.concatenate(sc_out))
    return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))


def _merge_topk(running, rows, pos, sc, n_s1, top_k):
    """Merge a block's per-row top-K into the running global top-K (RAM-optimized)."""
    if running is None:
        indptr_r = np.zeros(n_s1 + 1, dtype=np.int64)
        pos_r = np.zeros(0, dtype=np.int32)
        sc_r = np.zeros(0, dtype=np.float32)
    else:
        indptr_r, pos_r, sc_r = running

    if len(rows) == 0:
        return (indptr_r, pos_r, sc_r)

    rows_r = np.repeat(np.arange(n_s1, dtype=np.int32), np.diff(indptr_r))
    all_rows = np.concatenate([rows_r, rows.astype(np.int32)])
    all_pos = np.concatenate([pos_r, pos.astype(np.int32)])
    all_sc = np.concatenate([sc_r, sc.astype(np.float32)])
    del rows_r

    # Fast 32-bit stable sort by row index
    order = np.argsort(all_rows, kind='stable')
    srows = all_rows[order]
    spos = all_pos[order]
    ssc = all_sc[order]
    del order, all_rows, all_pos, all_sc

    lefts = np.searchsorted(srows, np.arange(n_s1, dtype=np.int32), side='left')
    rights = np.searchsorted(srows, np.arange(n_s1, dtype=np.int32), side='right')
    counts = rights - lefts
    capped = np.minimum(counts, top_k)

    indptr = np.zeros(n_s1 + 1, dtype=np.int64)
    np.cumsum(capped, out=indptr[1:])
    N = int(indptr[-1])
    pos_out = np.empty(N, dtype=np.int32)
    sc_out = np.empty(N, dtype=np.float32)

    for i in range(n_s1):
        c = capped[i]
        if c > 0:
            a_in = lefts[i]
            b_in = rights[i]
            a_out = indptr[i]
            if b_in - a_in > top_k:
                sub_sc = ssc[a_in:b_in]
                sub_pos = spos[a_in:b_in]
                idx = np.argpartition(sub_sc, (b_in - a_in) - top_k)[(b_in - a_in) - top_k:]
                sorted_idx = idx[np.argsort(-sub_sc[idx])]
                pos_out[a_out:a_out + top_k] = sub_pos[sorted_idx]
                sc_out[a_out:a_out + top_k] = sub_sc[sorted_idx]
            else:
                pos_out[a_out:a_out + c] = spos[a_in:b_in]
                sc_out[a_out:a_out + c] = ssc[a_in:b_in]

    del srows, spos, ssc, lefts, rights, counts, capped
    gc.collect()
    return indptr, pos_out, sc_out


def _tf_texts(df, positions):
    """v1's blocking text (name weighted x2 + address), built from the cached
    normalized columns so no cache rebuild is needed."""
    names = df['name_norm'].iloc[positions].fillna("").astype(str)
    addrs = df['addr_norm'].iloc[positions].fillna("").astype(str)
    return (names + ' ' + names + ' ' + addrs).tolist()


def generate_candidates_tfidf_parallel(s1_df, s23_df, top_k=TFIDF_TOP_K,
                                       min_sim=TFIDF_MIN_SIM, workers=WORKERS):
    log(f"Generating TF-IDF candidates in parallel (workers={workers}, top_k={top_k}, "
        f"min_sim={min_sim})...")
    t0 = time.time()
    n_s1 = len(s1_df)
    s1_ids = s1_df['entity_id'].to_numpy(dtype=object)
    s1_ctry = s1_df['country_norm']
    s23_ctry = s23_df['country_norm']
    running = None
    tmp_dir = os.path.join(CACHE_DIR, "tf_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    piece_size = max(TFIDF_CHUNK, (50000 // TFIDF_CHUNK) * TFIDF_CHUNK)

    for country in [c for c in s1_ctry.dropna().unique()]:
        s1_rows = np.flatnonzero((s1_ctry == country).to_numpy())
        s23_rows = np.flatnonzero((s23_ctry == country).to_numpy())
        log(f"  Country {country}: S1 {len(s1_rows):,}, S23 {len(s23_rows):,}")
        if len(s1_rows) == 0 or len(s23_rows) == 0:
            continue

        # fit on a bounded sample (same recipe as v1, but transliterated text)
        if len(s23_rows) > TFIDF_FIT_SAMPLE:
            rng = np.random.default_rng(42)
            fit_rows = s23_rows[np.sort(rng.choice(len(s23_rows), size=TFIDF_FIT_SAMPLE, replace=False))]
        else:
            fit_rows = s23_rows
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=500000,
                                sublinear_tf=True, min_df=2, max_df=TFIDF_MAX_DF, dtype=np.float32)
        fit_texts = np.asarray(_tf_texts(s23_df, fit_rows), dtype=object)
        tfidf.fit(fit_texts)
        del fit_texts
        n_voc = len(tfidf.vocabulary_)
        log(f"    Vocabulary: {n_voc:,} features")
        tfidf_path = os.path.join(tmp_dir, f"tfidf_{country}.pkl")
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)

        # transform S1 once, save pieces for zero-copy sharing with workers
        piece_prefix = os.path.join(tmp_dir, f"piece_{country}")
        n_pieces = (len(s1_rows) + piece_size - 1) // piece_size
        for p in range(n_pieces):
            a, b = p * piece_size, min((p + 1) * piece_size, len(s1_rows))
            texts = np.asarray(_tf_texts(s1_df, s1_rows[a:b]), dtype=object)
            M = tfidf.transform(texts)
            np.save(f"{piece_prefix}_{p}_data.npy", M.data)
            np.save(f"{piece_prefix}_{p}_indices.npy", M.indices)
            np.save(f"{piece_prefix}_{p}_indptr.npy", M.indptr)
            del M
        log(f"    S1 transformed into {n_pieces} piece(s) ({time.time() - t0:.0f}s)")

        # write block tasks (pool texts) for the workers
        n_blocks = (len(s23_rows) + TFIDF_BLOCK_DOCS - 1) // TFIDF_BLOCK_DOCS
        task_paths = []
        for bi in range(n_blocks):
            b0, b1 = bi * TFIDF_BLOCK_DOCS, min((bi + 1) * TFIDF_BLOCK_DOCS, len(s23_rows))
            block_pos = s23_rows[b0:b1]
            texts = np.asarray(_tf_texts(s23_df, block_pos), dtype=object)
            path = os.path.join(tmp_dir, f"task_{country}_{bi}.pkl")
            with open(path, 'wb') as f:
                pickle.dump((block_pos, texts), f, protocol=pickle.HIGHEST_PROTOCOL)
            task_paths.append(path)

        os.environ["ER_SPAWNED"] = "1"
        done = 0
        with mp.Pool(min(workers, n_blocks), initializer=_tf_worker_init,
                     initargs=(tfidf_path, piece_prefix, n_pieces, piece_size,
                               n_voc, len(s1_rows), (top_k, min_sim, TFIDF_CHUNK))) as pool:
            for rows_local, pos_global, sc in pool.imap_unordered(_tf_block_task, task_paths):
                rows_global = s1_rows[rows_local]
                running = _merge_topk(running, rows_global, pos_global, sc, n_s1, top_k)
                done += 1
                log(f"    Block {done}/{n_blocks} merged ({time.time() - t0:.0f}s)")
        for path in task_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        for p in range(n_pieces):
            for part in ("data", "indices", "indptr"):
                try:
                    os.remove(f"{piece_prefix}_{p}_{part}.npy")
                except OSError:
                    pass
        try:
            os.remove(tfidf_path)
        except OSError:
            pass
        gc.collect()

    indptr, pos_flat, sc_flat = running if running is not None else (
        np.zeros(n_s1 + 1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
    index = CandidateIndex(s1_ids, indptr, pos_flat.astype(np.int32), sc_flat)
    log(f"  TF-IDF candidates: {index.n_pairs:,} ({time.time() - t0:.0f}s total)")
    mem_report("after TF-IDF blocking")
    return index


def union_candidate_indices(idx_a, idx_b, cap_total=CAP_TOTAL):
    """Union of two CandidateIndex objects over the same S1 frame; A's pairs
    keep priority under the per-row cap. Scores are placeholders."""
    assert len(idx_a) == len(idx_b)
    n = len(idx_a)
    rows_a = np.repeat(np.arange(n, dtype=np.int64), np.diff(idx_a.indptr))
    rows_b = np.repeat(np.arange(n, dtype=np.int64), np.diff(idx_b.indptr))
    df = pd.DataFrame({
        "r": np.concatenate([rows_a, rows_b]),
        "p": np.concatenate([idx_a.s23_pos, idx_b.s23_pos]).astype(np.int64),
    })
    del rows_a, rows_b
    df = df.drop_duplicates()
    df = df.groupby("r", sort=False).head(cap_total)
    s1_pos = df["r"].to_numpy(dtype=np.int64)
    s23_pos = df["p"].to_numpy(dtype=np.int64)
    del df
    order = np.argsort(s1_pos, kind="stable")
    rows_sorted = s1_pos[order]
    pos_sorted = s23_pos[order]
    counts = np.bincount(rows_sorted, minlength=n).astype(np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    index = CandidateIndex(idx_a.s1_ids, indptr, pos_sorted.astype(np.int32),
                           np.ones(len(pos_sorted), dtype=np.float32))
    log(f"  Union candidates: {index.n_pairs:,} (avg {index.n_pairs / max(n, 1):.1f}/S1, cap {cap_total})")
    return index


def blocking_recall(cand, gt_dict, s23_ids_np):
    total = found = 0
    for s1_id, true_matches in gt_dict.items():
        i = cand.row_of(s1_id)
        total += len(true_matches)
        if i >= 0:
            a, b = cand.slice_for(i)
            found += len(true_matches & set(s23_ids_np[cand.s23_pos[a:b]].tolist()))
    return found / max(total, 1), found, total


def generate_candidates_multikey(s1_df, s23_df, cap_total=CAP_TOTAL):
    """
    Multi-key blocking: union of capped exact-key joins (v2.1, 8 keys after the
    first 4-key attempt measured blocking recall 0.658 - the misses were
    extra/missing words (DBA-style names), cross-script spelling drift, and
    matches carried by the address).
      K1 (country, name)                          - identical normalized names
      K2 (country, sorted name tokens)            - word-order swaps
      K3 (country, sorted metaphone token set)    - typos + spelling variants
      K4 (country, metaphone(full name, no spaces)) - cross-script + minor drift
      K5 (country, phon(first tok), phon(last tok)) - brand+end anchor: middle
        words added/removed ("Reliance Industries" vs "Reliance Ind Ltd")
      K6 (country, house no, phon(first tok))       - address anchor
      K7 (country, house no, PIN)                   - same premises, DBA names
      K8 (country, PIN, phon(first tok))            - brand + area anchor
    Candidates keep key priority (K1 first) under the global per-S1 cap.
    Returns a CandidateIndex (scores are placeholders; no TF-IDF in v2).
    """
    log("Generating candidates via multi-key blocking (8 keys)...")
    t0 = time.time()
    n_s1 = len(s1_df)
    s1_ids = s1_df['entity_id'].to_numpy(dtype=object)
    s1_pos_all = np.arange(n_s1, dtype=np.int64)
    s23_pos_all = np.arange(len(s23_df), dtype=np.int64)

    def frame_keys(df, pos):
        ctry = df['country_norm'].astype(str)
        name = df['name_norm'].fillna("")
        has_name = name.str.len() > 0
        toks = name.str.split()
        sorted_key = ctry + '|' + toks.map(lambda ws: " ".join(sorted(ws)) if ws else "")
        phon_set = toks.map(lambda ws: tuple(sorted({phon(t) for t in ws if len(t) > 1})))
        phon_key = ctry + '|' + phon_set.map(lambda t: '|'.join(t) if t else "")
        exact_key = ctry + '|' + name
        full_phon = name.map(lambda s: _metaphone(re.sub(r'\s+', '', s)) if s else "")
        phfull_key = ctry + '|' + full_phon
        first_phon = toks.map(lambda ws: phon(ws[0]) if ws else "")
        last_phon = toks.map(lambda ws: phon(ws[-1]) if ws else "")
        brandend_key = ctry + '|' + first_phon + '|' + last_phon
        hnum = df['addr_norm'].fillna("").map(house_number)
        pin = df['addr_norm'].fillna("").map(longest_digit_run)
        addr_key = ctry + '|' + hnum + '|' + first_phon
        addrpin_key = ctry + '|' + hnum + '|' + pin
        brandpin_key = ctry + '|' + pin + '|' + first_phon
        phprefix_key = ctry + '|' + full_phon.str[:5] + '|' + hnum
        pos_arr = pd.Series(pos, index=df.index)
        sel_name = has_name
        sel_addr = has_name & (hnum.str.len() > 0)
        sel_pin = has_name & (pin.str.len() > 0)
        sel_addrpin = has_name & (hnum.str.len() > 0) & (pin.str.len() > 0)
        sel_phprefix = sel_addr
        return [
            (pos_arr[sel_name].to_numpy(), exact_key[sel_name]),
            (pos_arr[sel_name].to_numpy(), sorted_key[sel_name]),
            (pos_arr[sel_name].to_numpy(), phon_key[sel_name]),
            (pos_arr[sel_name].to_numpy(), phfull_key[sel_name]),
            (pos_arr[sel_name].to_numpy(), brandend_key[sel_name]),
            (pos_arr[sel_addr].to_numpy(), addr_key[sel_addr]),
            (pos_arr[sel_addrpin].to_numpy(), addrpin_key[sel_addrpin]),
            (pos_arr[sel_pin].to_numpy(), brandpin_key[sel_pin]),
            (pos_arr[sel_phprefix].to_numpy(), phprefix_key[sel_phprefix]),
        ]

    log(f"  Deriving keys for {len(s23_df):,} pool rows...")
    s23_keys = frame_keys(s23_df, s23_pos_all)
    log(f"  Deriving keys for {len(s1_df):,} S1 rows ({time.time() - t0:.0f}s)...")
    s1_keys = frame_keys(s1_df, s1_pos_all)

    parts = []
    for key_name, cap, (sk1, s23k) in zip(
            ("exact-name", "sorted-tokens", "phonetic-set", "full-phonetic",
             "brand-end", "addr-anchor", "addr-pin", "brand-pin", "phon-prefix"),
            (CAP_EXACT, CAP_SORTED, CAP_PHON, CAP_PHFULL,
             CAP_BRANDEND, CAP_ADDR, CAP_ADDRPIN, CAP_BRANDPIN, CAP_PHPREFIX),
            zip(s1_keys, s23_keys)):
        parts.append(_join_keyed(sk1[0], sk1[1], s23k[0], s23k[1], cap, key_name, t0))

    allp = pd.concat(parts).drop_duplicates(["s1_pos", "s23_pos"])
    del parts
    allp = allp.groupby("s1_pos", sort=False).head(cap_total)
    s1_pos = allp["s1_pos"].to_numpy(dtype=np.int64)
    s23_pos = allp["s23_pos"].to_numpy(dtype=np.int64)
    del allp
    gc.collect()

    order = np.argsort(s1_pos, kind="stable")
    rows_sorted = s1_pos[order]
    pos_sorted = s23_pos[order]
    counts = np.bincount(rows_sorted, minlength=n_s1).astype(np.int64)
    indptr = np.zeros(n_s1 + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    index = CandidateIndex(s1_ids, indptr, pos_sorted.astype(np.int32),
                           np.ones(len(pos_sorted), dtype=np.float32))
    log(f"  Total candidates: {index.n_pairs:,} for {int((counts > 0).sum()):,} S1 rows "
        f"(avg {index.n_pairs / max(n_s1, 1):.1f}/S1, {time.time() - t0:.0f}s total)")
    mem_report("after blocking")
    return index


# ============================================================
# FEATURE ENGINEERING - 34 pairwise features
# ============================================================

def _gram3_set(s):
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def compute_features(s1_row, s23_row):
    features = {}

    name1 = s1_row.get('name_norm', '')
    name2 = s23_row.get('name_norm', '')
    addr1 = s1_row.get('addr_norm', '')
    addr2 = s23_row.get('addr_norm', '')

    # --- name similarity (v1) ---
    features['name_ratio'] = fuzz.ratio(name1, name2) / 100.0
    features['name_partial_ratio'] = fuzz.partial_ratio(name1, name2) / 100.0
    features['name_token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100.0
    features['name_token_set'] = fuzz.token_set_ratio(name1, name2) / 100.0
    features['name_WRatio'] = fuzz.WRatio(name1, name2) / 100.0

    # --- address similarity (v1) ---
    features['addr_ratio'] = fuzz.ratio(addr1, addr2) / 100.0
    features['addr_partial_ratio'] = fuzz.partial_ratio(addr1, addr2) / 100.0
    features['addr_token_sort'] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    features['addr_token_set'] = fuzz.token_set_ratio(addr1, addr2) / 100.0
    features['addr_WRatio'] = fuzz.WRatio(addr1, addr2) / 100.0

    # --- token overlaps (v1) ---
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

    addr_tokens1 = set(addr1.split()) if addr1 else set()
    addr_tokens2 = set(addr2.split()) if addr2 else set()
    if addr_tokens1 and addr_tokens2:
        addr_intersection = addr_tokens1 & addr_tokens2
        features['addr_jaccard'] = len(addr_intersection) / len(addr_tokens1 | addr_tokens2)
        features['addr_overlap_min'] = len(addr_intersection) / min(len(addr_tokens1), len(addr_tokens2))
    else:
        features['addr_jaccard'] = 0.0
        features['addr_overlap_min'] = 0.0

    # --- shape / quality (v1) ---
    features['name_len_diff'] = abs(len(name1) - len(name2)) / max(len(name1), len(name2), 1)
    features['addr_len_diff'] = abs(len(addr1) - len(addr2)) / max(len(addr1), len(addr2), 1)
    features['name1_empty'] = 1.0 if not name1 else 0.0
    features['name2_empty'] = 1.0 if not name2 else 0.0
    features['addr1_empty'] = 1.0 if not addr1 else 0.0
    features['addr2_empty'] = 1.0 if not addr2 else 0.0
    features['country_match'] = 1.0 if s1_row.get('country_norm', '') == s23_row.get('country_norm', '') else 0.0

    first1 = name1.split()[0] if name1.split() else ""
    first2 = name2.split()[0] if name2.split() else ""
    features['first_word_match'] = 1.0 if first1 and first2 and first1 == first2 else 0.0
    features['first_word_sim'] = fuzz.ratio(first1, first2) / 100.0 if first1 and first2 else 0.0

    nums1 = set(re.findall(r'\b\d+\b', addr1))
    nums2 = set(re.findall(r'\b\d+\b', addr2))
    if nums1 and nums2:
        features['num_jaccard'] = len(nums1 & nums2) / len(nums1 | nums2)
    elif not nums1 and not nums2:
        features['num_jaccard'] = 1.0
    else:
        features['num_jaccard'] = 0.0

    features['is_s3'] = 1.0 if str(s23_row.get('entity_id', '')).startswith('S3') else 0.0

    # --- v2 additions (8) ---
    # Normalized edit distance: adjacent-key typos ("Gooegle" vs "Google").
    features['name_lev'] = Levenshtein.normalized_similarity(name1, name2) if name1 and name2 else 0.0
    features['addr_lev'] = Levenshtein.normalized_similarity(addr1, addr2) if addr1 and addr2 else 0.0

    # Character 3-gram Jaccard: sub-word robustness to typos + transliteration drift.
    g1, g2 = _gram3_set(name1), _gram3_set(name2)
    features['name_gram3'] = len(g1 & g2) / len(g1 | g2) if g1 and g2 else 0.0
    ga, gb = _gram3_set(addr1), _gram3_set(addr2)
    features['addr_gram3'] = len(ga & gb) / len(ga | gb) if ga and gb else 0.0

    # Phonetic equivalence (metaphone of tokens): Catherine/Katherine, Smythe/Smith.
    p1 = {phon(t) for t in tokens1 if len(t) > 1}
    p2 = {phon(t) for t in tokens2 if len(t) > 1}
    features['name_phon_exact'] = 1.0 if (p1 and p2 and p1 == p2) else 0.0
    features['name_phon_jaccard'] = len(p1 & p2) / len(p1 | p2) if p1 and p2 else 0.0

    # Positional address evidence: first digit run (house no.) and longest digit
    # run (PIN/zip) survive address word shuffling.
    h1, h2 = house_number(addr1), house_number(addr2)
    features['house_match'] = 1.0 if h1 and h2 and h1 == h2 else 0.0
    pin1 = max(re.findall(r'\d{4,}', addr1), key=len) if re.findall(r'\d{4,}', addr1) else ""
    pin2 = max(re.findall(r'\d{4,}', addr2), key=len) if re.findall(r'\d{4,}', addr2) else ""
    features['pin_match'] = 1.0 if pin1 and pin2 and pin1 == pin2 else 0.0

    # v4 additions: 4 selective features
    # 1. Jaro-Winkler similarity (typo robustness)
    features['name_jw'] = JaroWinkler.normalized_similarity(name1, name2) if name1 and name2 else 0.0
    features['addr_jw'] = JaroWinkler.normalized_similarity(addr1, addr2) if addr1 and addr2 else 0.0

    # 2. Hard filters for postal data
    features['house_mismatch_penalty'] = 1.0 if (h1 and h2 and h1 != h2) else 0.0
    features['pin_mismatch_penalty'] = 1.0 if (pin1 and pin2 and pin1 != pin2) else 0.0

    return features


FEATURE_NAMES = list(compute_features(
    {'name_norm': 'test', 'addr_norm': 'test', 'country_norm': 'us', 'entity_id': 'S1-1'},
    {'name_norm': 'test', 'addr_norm': 'test', 'country_norm': 'us', 'entity_id': 'S2-1'}
).keys())


def compute_features_batch(s1_records, s23_records):
    all_features = []
    for s1_row, s23_row in zip(s1_records, s23_records):
        all_features.append(compute_features(s1_row, s23_row))
    return pd.DataFrame(all_features)


def compute_pair_features(s1_df, s23_df, s1_idx, s23_idx):
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
# TRAINING
# ============================================================

def prepare_training_data(s1_df, s23_df, gt_df, cand_index, sample_neg_ratio=3):
    log("Preparing training data...")
    s23_ids = s23_df['entity_id'].to_numpy(dtype=object)
    s23_index = pd.Index(s23_ids)
    gt_matches = build_gt_matches(gt_df)
    empty = frozenset()

    pos_pairs = []
    neg_rows, neg_pos = [], []
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
    log("Training XGBoost model...")
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    def make(device):
        return xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric='logloss',
            random_state=42, n_jobs=-1, tree_method='hist', device=device,
        )

    model = make('cuda')
    try:
        model.fit(X, y, verbose=True)
        log("  Trained on GPU (device=cuda)")
    except Exception as exc:
        log(f"  GPU training unavailable ({type(exc).__name__}); falling back to CPU")
        model = make('cpu')
        model.fit(X, y, verbose=True)

    importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
    log("Feature importance (top 12):")
    for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:12]:
        log(f"  {name}: {imp:.4f}")
    return model


# ============================================================
# INFERENCE + OUTPUT
# ============================================================

def score_pairs(model, s1_df, s23_df, cand_index, chunk_rows=PRED_CHUNK_ROWS):
    """Score every candidate pair; returns (s1_row_idx, s23_pos, prob) arrays."""
    log("Scoring candidate pairs...")
    n = len(cand_index)
    rows_out, pos_out, prob_out = [], [], []
    for i0 in range(0, n, chunk_rows):
        i1 = min(i0 + chunk_rows, n)
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
            rows_out.append(rows)
            pos_out.append(pos)
            prob_out.append(probs)
        if ((i1 // chunk_rows) % 20 == 0) or i1 == n:
            log(f"  Scored {i1:,}/{n:,} S1 rows")
    if rows_out:
        return (np.concatenate(rows_out), np.concatenate(pos_out).astype(np.int64),
                np.concatenate(prob_out))
    return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))


def write_output(cand_index, matched_rows, matched_pos, s23_df, output_dir=OUTPUT_DIR):
    log("Writing output files...")
    n = len(cand_index)
    s1_ids = cand_index.s1_ids
    s23_ids_ser = s23_df['entity_id']

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


def compute_f05_score(predictions, ground_truth):
    scores = []
    for s1_id in ground_truth:
        true_set = ground_truth[s1_id]
        pred_set = set(predictions.get(s1_id, []))
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set or not pred_set:
            scores.append(0.0)
        else:
            tp = len(true_set & pred_set)
            precision = tp / len(pred_set)
            recall = tp / len(true_set)
            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append((1.25 * precision * recall) / (0.25 * precision + recall))
    return float(np.mean(scores))


# ============================================================
# PIPELINE ENTRY POINTS
# ============================================================

TRAIN_CACHE_FILES = ("train_s1_sample.parquet", "train_s23_sample.parquet", "train_gt_sample.pkl")


def build_train_caches():
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
        val_s1_proc = preprocess_dataframe(val_s1_subset.copy())
        save_df_cache(val_s1_proc, "val_s1_proc.parquet")
        save_cache(train_gt_val, "val_gt.pkl")
        del val_s1_proc, val_s1_subset
        gc.collect()

    if len(train_s1_subset) > TRAIN_SAMPLE_SIZE:
        log(f"  Sampling {TRAIN_SAMPLE_SIZE} S1 entities for training...")
        train_s1_sample = train_s1_subset.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)
    else:
        train_s1_sample = train_s1_subset
    train_gt_sample = train_gt_train[train_gt_train['source1_entity_id'].isin(train_s1_sample['entity_id'])]

    train_s1_sample = preprocess_dataframe(train_s1_sample)
    save_df_cache(train_s1_sample, "train_s1_sample.parquet")
    save_cache(train_gt_sample, "train_gt_sample.pkl")
    del train_s1, train_s1_subset, train_gt, train_gt_train, train_gt_sample
    gc.collect()

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


def run_training():
    """Build caches -> multi-key blocking -> features -> XGBoost (GPU) ->
    validation F0.5 with threshold grid search. Never touches test data."""
    log("=" * 60)
    log("Entity Resolution Pipeline v4 - TRAINING + VALIDATION")
    log("=" * 60)
    mem_report("start")

    have_train_cache = all(cache_exists(f) for f in TRAIN_CACHE_FILES)
    if not have_train_cache:
        log("Preprocessing training data (first run - will be cached)...")
        build_train_caches()
    elif USE_VALIDATION and not (cache_exists("val_s1_proc.parquet") and cache_exists("val_gt.pkl")):
        build_train_caches()

    train_s1_sample = prune_norm_cols(load_df_cache("train_s1_sample.parquet"), "train_s1_sample")
    train_s23_sample = prune_norm_cols(load_df_cache("train_s23_sample.parquet"), "train_s23_sample")
    train_gt_sample = load_cache("train_gt_sample.pkl")
    mem_report("after loading train caches")

    if cache_exists("train_candidates.pkl"):
        train_candidates = load_cache("train_candidates.pkl")
    else:
        log("Running blocking on training data (TF-IDF + keys, will be cached)...")
        tfidf_idx = generate_candidates_tfidf_parallel(train_s1_sample, train_s23_sample)
        key_idx = generate_candidates_multikey(train_s1_sample, train_s23_sample)
        train_candidates = union_candidate_indices(tfidf_idx, key_idx)
        del tfidf_idx, key_idx
        gc.collect()
        save_cache(train_candidates, "train_candidates.pkl")

    gt_matches_dict = build_gt_matches(train_gt_sample)
    s23_ids_np = train_s23_sample['entity_id'].to_numpy(dtype=object)
    recall, found, total = blocking_recall(train_candidates, gt_matches_dict, s23_ids_np)
    log(f"  Blocking recall (union): {recall:.4f} ({found}/{total})")
    del gt_matches_dict, s23_ids_np
    gc.collect()

    if cache_exists("xgb_model.pkl"):
        model = load_cache("xgb_model.pkl")
    else:
        X_train, y_train = prepare_training_data(
            train_s1_sample, train_s23_sample, train_gt_sample, train_candidates)
        model = train_model(X_train, y_train)
        save_cache(model, "xgb_model.pkl")
        del X_train, y_train
        gc.collect()

    del train_s1_sample, train_s23_sample, train_gt_sample, train_candidates
    gc.collect()
    mem_report("after training stage")

    if USE_VALIDATION:
        log("Running validation + threshold grid search...")
        val_sample, val_s23, val_gt_matches = build_val_pool()
        val_candidates = union_candidate_indices(
            generate_candidates_tfidf_parallel(val_sample, val_s23),
            generate_candidates_multikey(val_sample, val_s23))
        val_rows, val_pos, val_probs = score_pairs(model, val_sample, val_s23, val_candidates)
        val_gt_subset = {s1_id: val_gt_matches.get(s1_id, set())
                         for s1_id in val_sample['entity_id'].values}
        s23_ids_ser = val_s23['entity_id']
        best_thr, best_f05 = MATCH_THRESHOLD, -1.0
        for thr in THRESHOLD_GRID:
            sel = val_probs >= thr
            val_preds = pairs_to_dict(val_candidates, val_rows[sel], val_pos[sel], s23_ids_ser)
            f05 = compute_f05_score(val_preds, val_gt_subset)
            log(f"  threshold={thr:.2f} -> F0.5={f05:.4f}")
            if f05 > best_f05:
                best_thr, best_f05 = thr, f05
        log(f"  Validation F0.5 Score: {best_f05:.4f} (threshold={best_thr:.2f})")
        save_cache({"threshold": best_thr, "f05": best_f05}, "best_threshold.pkl")
        del val_sample, val_s23, val_candidates, val_rows, val_pos, val_probs, val_preds, val_gt_subset, val_gt_matches
        gc.collect()

    log("=" * 60)
    log("v4 Training + validation complete. Model: cache/v4/xgb_model.pkl")
    log("=" * 60)
    mem_report("end")


def run_test():
    """Multi-key blocking (cached) -> scoring with the tuned threshold ->
    output/matching_results.tsv + output/candidate_pairs.tsv."""
    log("=" * 60)
    log("Entity Resolution Pipeline v4 - TEST INFERENCE")
    log("=" * 60)
    mem_report("start")

    if not cache_exists("xgb_model.pkl"):
        log("ERROR: no trained model at cache/v4/xgb_model.pkl. Run train.py first.")
        return
    model = load_cache("xgb_model.pkl")
    if cache_exists("best_threshold.pkl"):
        tuned = load_cache("best_threshold.pkl")
        threshold = tuned["threshold"]
        log(f"  Using validation-tuned threshold: {threshold:.2f} (F0.5={tuned['f05']:.4f})")
    else:
        threshold = MATCH_THRESHOLD
        log(f"  No tuned threshold found; using default {threshold:.2f}")

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

    if cache_exists("test_candidates.pkl"):
        test_candidates = load_cache("test_candidates.pkl")
    else:
        log("Running blocking on test data (TF-IDF + keys, will be cached)...")
        tfidf_idx = generate_candidates_tfidf_parallel(test_s1, test_s23)
        key_idx = generate_candidates_multikey(test_s1, test_s23)
        test_candidates = union_candidate_indices(tfidf_idx, key_idx)
        del tfidf_idx, key_idx
        gc.collect()
        save_cache(test_candidates, "test_candidates.pkl")

    test_rows, test_pos, test_probs = score_pairs(model, test_s1, test_s23, test_candidates)
    sel = test_probs >= threshold
    log(f"  Matches at threshold={threshold:.2f}: {int(sel.sum()):,} pairs, "
        f"{int(np.unique(test_rows[sel]).size):,} S1 entities")
    write_output(test_candidates, test_rows[sel], test_pos[sel], test_s23)

    log("=" * 60)
    log("v2 Pipeline complete! Submissions written to output/")
    log("=" * 60)
    mem_report("end")
