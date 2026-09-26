import os
import sys
import time
import json
import duckdb

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

DATASET_BASE = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset")
TRAIN_DIR = os.path.join(DATASET_BASE, "train")
TEST_DIR = os.path.join(DATASET_BASE, "test")

def main():
    start_time = time.time()
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET threads TO 4;")

    print("==================================================")
    print(" DEEP EDA: PRE-NORMALIZATION DATA ANALYSIS ")
    print("==================================================")

    files = {
        "train_s1": os.path.join(TRAIN_DIR, "train_source1.tsv"),
        "train_s2": os.path.join(TRAIN_DIR, "train_source2.tsv"),
        "train_s3": os.path.join(TRAIN_DIR, "train_source3.tsv"),
        "train_gt": os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
        "test_s1": os.path.join(TEST_DIR, "test_source1.tsv"),
        "test_s2": os.path.join(TEST_DIR, "test_source2.tsv"),
        "test_s3": os.path.join(TEST_DIR, "test_source3.tsv"),
    }

    for key, path in files.items():
        safe_path = path.replace('\\', '/')
        conn.execute(f"""
            CREATE OR REPLACE VIEW {key} AS 
            SELECT * FROM read_csv_auto('{safe_path}', delim='\t', header=true, all_varchar=true, ignore_errors=true);
        """)

    # ----------------------------------------------------
    # Q1: NULL vs Empty String Analysis
    # ----------------------------------------------------
    print("\n--- Q1: NULL / EMPTY VALUES ---")
    q1_results = {}
    for key in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
        row = conn.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN business_name IS NULL THEN 1 ELSE 0 END) as name_null,
                SUM(CASE WHEN business_name IS NOT NULL AND TRIM(business_name) = '' THEN 1 ELSE 0 END) as name_empty,
                SUM(CASE WHEN business_address IS NULL THEN 1 ELSE 0 END) as addr_null,
                SUM(CASE WHEN business_address IS NOT NULL AND TRIM(business_address) = '' THEN 1 ELSE 0 END) as addr_empty
            FROM {key}
        """).fetchone()
        tot = row[0]
        q1_results[key] = {
            "total": tot,
            "name_null_pct": round(row[1]*100.0/tot, 4),
            "name_empty_pct": round(row[2]*100.0/tot, 4),
            "addr_null_pct": round(row[3]*100.0/tot, 4),
            "addr_empty_pct": round(row[4]*100.0/tot, 4),
            "name_null_cnt": row[1],
            "name_empty_cnt": row[2],
            "addr_null_cnt": row[3],
            "addr_empty_cnt": row[4]
        }
        print(f"[{key:8s}] Tot: {tot:,} | Name NULL: {row[1]} ({q1_results[key]['name_null_pct']}%), Empty: {row[2]} ({q1_results[key]['name_empty_pct']}%) | Addr NULL: {row[3]} ({q1_results[key]['addr_null_pct']}%), Empty: {row[4]} ({q1_results[key]['addr_empty_pct']}%)")

    # ----------------------------------------------------
    # Q2: Case Variations
    # ----------------------------------------------------
    print("\n--- Q2: CASE VARIATIONS ---")
    q2_results = {}
    for key in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
        row = conn.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN business_name = LOWER(business_name) AND business_name != UPPER(business_name) THEN 1 ELSE 0 END) as lower_cnt,
                SUM(CASE WHEN business_name = UPPER(business_name) AND business_name != LOWER(business_name) THEN 1 ELSE 0 END) as upper_cnt,
                SUM(CASE WHEN business_name != LOWER(business_name) AND business_name != UPPER(business_name) THEN 1 ELSE 0 END) as mixed_cnt
            FROM {key}
            WHERE business_name IS NOT NULL AND TRIM(business_name) != ''
        """).fetchone()
        tot = row[0]
        q2_results[key] = {
            "lower_pct": round(row[1]*100.0/tot, 2),
            "upper_pct": round(row[2]*100.0/tot, 2),
            "mixed_pct": round(row[3]*100.0/tot, 2)
        }
        print(f"[{key:8s}] Lowercase: {q2_results[key]['lower_pct']}% | Uppercase: {q2_results[key]['upper_pct']}% | Mixed: {q2_results[key]['mixed_pct']}%")

    # ----------------------------------------------------
    # Q3: Script & Language Detection
    # ----------------------------------------------------
    print("\n--- Q3: SCRIPT / LANGUAGE DETECTION ---")
    q3_results = {}
    for key in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
        row = conn.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN REGEXP_MATCHES(business_name, '[\u0900-\u097F]') AND NOT REGEXP_MATCHES(business_name, '[a-zA-Z]') THEN 1 ELSE 0 END) as pure_devanagari,
                SUM(CASE WHEN REGEXP_MATCHES(business_name, '[\u0900-\u097F]') AND REGEXP_MATCHES(business_name, '[a-zA-Z]') THEN 1 ELSE 0 END) as mixed_script,
                SUM(CASE WHEN NOT REGEXP_MATCHES(business_name, '[\u0900-\u097F]') AND REGEXP_MATCHES(business_name, '[a-zA-Z]') THEN 1 ELSE 0 END) as pure_latin,
                SUM(CASE WHEN NOT REGEXP_MATCHES(business_name, '[\u0900-\u097F]') AND NOT REGEXP_MATCHES(business_name, '[a-zA-Z]') THEN 1 ELSE 0 END) as other_script
            FROM {key}
            WHERE business_name IS NOT NULL
        """).fetchone()
        tot = row[0]
        q3_results[key] = {
            "pure_latin_pct": round(row[3]*100.0/tot, 2),
            "pure_devanagari_pct": round(row[1]*100.0/tot, 2),
            "mixed_script_pct": round(row[2]*100.0/tot, 2),
            "other_script_pct": round(row[4]*100.0/tot, 2),
            "pure_latin_cnt": row[3],
            "pure_devanagari_cnt": row[1],
            "mixed_cnt": row[2]
        }
        print(f"[{key:8s}] Pure Latin: {q3_results[key]['pure_latin_pct']}% | Pure Devanagari: {q3_results[key]['pure_devanagari_pct']}% | Mixed: {q3_results[key]['mixed_script_pct']}% | Other: {q3_results[key]['other_script_pct']}%")

    print("\n--- Q3.1: Script Usage in India Records ---")
    for key in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
        row = conn.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN REGEXP_MATCHES(business_name, '[\u0900-\u097F]') THEN 1 ELSE 0 END) as devanagari_cnt
            FROM {key}
            WHERE country = 'India' AND business_name IS NOT NULL
        """).fetchone()
        tot = row[0] if row[0] > 0 else 1
        pct = round(row[1]*100.0/tot, 2)
        print(f"[{key:8s} India] Total: {row[0]:,} | Has Devanagari: {row[1]:,} ({pct}%)")

    # ----------------------------------------------------
    # Q4: Punctuation & Special Characters
    # ----------------------------------------------------
    print("\n--- Q4: PUNCTUATION & SPECIAL CHARACTERS ---")
    q4_results = {}
    for key in ["train_s1", "train_s2", "train_s3"]:
        row = conn.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN business_name LIKE '%.%' THEN 1 ELSE 0 END) as has_period,
                SUM(CASE WHEN business_name LIKE '%''%' THEN 1 ELSE 0 END) as has_apostrophe,
                SUM(CASE WHEN business_name LIKE '%-%' THEN 1 ELSE 0 END) as has_hyphen,
                SUM(CASE WHEN business_name LIKE '%&%' THEN 1 ELSE 0 END) as has_ampersand,
                SUM(CASE WHEN business_name LIKE '%/%' OR business_name LIKE '%\\%' THEN 1 ELSE 0 END) as has_slash,
                SUM(CASE WHEN business_name LIKE '%,%' THEN 1 ELSE 0 END) as has_comma
            FROM {key}
            WHERE business_name IS NOT NULL
        """).fetchone()
        tot = row[0]
        q4_results[key] = {
            "period_pct": round(row[1]*100.0/tot, 2),
            "apostrophe_pct": round(row[2]*100.0/tot, 2),
            "hyphen_pct": round(row[3]*100.0/tot, 2),
            "ampersand_pct": round(row[4]*100.0/tot, 2),
            "slash_pct": round(row[5]*100.0/tot, 2),
            "comma_pct": round(row[6]*100.0/tot, 2)
        }
        print(f"[{key:8s}] Period(.): {q4_results[key]['period_pct']}% | Apo('): {q4_results[key]['apostrophe_pct']}% | Hyphen(-): {q4_results[key]['hyphen_pct']}% | Amp(&): {q4_results[key]['ampersand_pct']}% | Comma(,): {q4_results[key]['comma_pct']}%")

    # ----------------------------------------------------
    # Q5: Business Abbreviations Frequency
    # ----------------------------------------------------
    print("\n--- Q5: ABBREVIATIONS FREQUENCY ---")
    abbrs = ["Corp", "Corporation", "Inc", "Incorporated", "Ltd", "Limited", "Pvt", "Private", "Co", "Company", "Assoc", "Association", "Org", "Organization", "LLC"]
    q5_results = {}
    for key in ["train_s1", "train_s2", "train_s3"]:
        q5_results[key] = {}
        tot = file_info_map[key]
        for ab in abbrs:
            cnt = conn.execute(f"""
                SELECT COUNT(*) FROM {key}
                WHERE REGEXP_MATCHES(business_name, '(?i)\\b{ab}\\b')
            """).fetchone()[0]
            q5_results[key][ab] = {"count": cnt, "pct": round(cnt*100.0/tot, 2)}
        print(f"\n[{key:8s} Top Suffixes]:")
        for ab in ["Inc", "Incorporated", "Corp", "Corporation", "Ltd", "Limited", "Pvt", "Private", "LLC"]:
            print(f"  - {ab:12s}: {q5_results[key][ab]['count']:,} ({q5_results[key][ab]['pct']}%)")

    # ----------------------------------------------------
    # Q6: Address Patterns
    # ----------------------------------------------------
    print("\n--- Q6: ADDRESS PATTERNS ---")
    for key in ["train_s1", "train_s2", "train_s3"]:
        row = conn.execute(f"""
            SELECT 
                ROUND(AVG(LENGTH(business_address)), 1) as avg_len,
                ROUND(AVG(ARRAY_LENGTH(STRING_SPLIT(business_address, ','))), 1) as avg_components,
                SUM(CASE WHEN REGEXP_MATCHES(business_address, '(?i)\\b(Near|Opp|Opposite|Behind|Beside|Adj)\\b') THEN 1 ELSE 0 END) as landmark_cnt,
                COUNT(*) as tot
            FROM {key}
            WHERE business_address IS NOT NULL AND TRIM(business_address) != ''
        """).fetchone()
        tot = row[3]
        landmark_pct = round(row[2]*100.0/tot, 2)
        print(f"[{key:8s}] Avg Addr Length: {row[0]} chars | Avg Components (commas): {row[1]} | Landmark Addresses: {row[2]:,} ({landmark_pct}%)")

    # ----------------------------------------------------
    # Q7 & Q9: Ground Truth Pair Cross-Lingual & Match Distribution
    # ----------------------------------------------------
    print("\n--- Q7 & Q9: GROUND TRUTH CROSS-LINGUAL MATCHES & S2/S3 SPLITS ---")
    
    conn.execute("""
        CREATE OR REPLACE VIEW ground_truth_pairs AS
        SELECT 
            source1_entity_id,
            UNNEST(STRING_SPLIT(matched_entity_ids, ',')) as target_entity_id
        FROM train_gt
        WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
    """)

    conn.execute("""
        CREATE OR REPLACE VIEW gt_script_matches AS
        SELECT 
            gt.source1_entity_id,
            s1.business_name as s1_name,
            s1.country as s1_country,
            gt.target_entity_id,
            COALESCE(s2.business_name, s3.business_name) as target_name,
            REGEXP_MATCHES(s1.business_name, '[\u0900-\u097F]') as s1_devanagari,
            REGEXP_MATCHES(COALESCE(s2.business_name, s3.business_name), '[\u0900-\u097F]') as target_devanagari
        FROM ground_truth_pairs gt
        JOIN train_s1 s1 ON gt.source1_entity_id = s1.entity_id
        LEFT JOIN train_s2 s2 ON gt.target_entity_id = s2.entity_id
        LEFT JOIN train_s3 s3 ON gt.target_entity_id = s3.entity_id;
    """)

    cross_lingual = conn.execute("""
        SELECT COUNT(*) FROM gt_script_matches
        WHERE s1_devanagari != target_devanagari
    """).fetchone()[0]
    total_gt_pairs = conn.execute("SELECT COUNT(*) FROM gt_script_matches").fetchone()[0]
    print(f"Total GT Pairs: {total_gt_pairs:,}")
    print(f"Cross-Lingual Matches (Latin S1 <-> Devanagari Target): {cross_lingual:,} ({round(cross_lingual*100.0/total_gt_pairs, 4)}%)")

    # Ground truth breakdown: Match ONLY S2, ONLY S3, or BOTH S2 & S3
    conn.execute("""
        CREATE OR REPLACE VIEW gt_source_combos AS
        SELECT 
            source1_entity_id,
            MAX(CASE WHEN target_entity_id LIKE 'S2-%' THEN 1 ELSE 0 END) as has_s2,
            MAX(CASE WHEN target_entity_id LIKE 'S3-%' THEN 1 ELSE 0 END) as has_s3
        FROM ground_truth_pairs
        GROUP BY source1_entity_id;
    """)

    only_s2 = conn.execute("SELECT COUNT(*) FROM gt_source_combos WHERE has_s2 = 1 AND has_s3 = 0").fetchone()[0]
    only_s3 = conn.execute("SELECT COUNT(*) FROM gt_source_combos WHERE has_s2 = 0 AND has_s3 = 1").fetchone()[0]
    both_s2_s3 = conn.execute("SELECT COUNT(*) FROM gt_source_combos WHERE has_s2 = 1 AND has_s3 = 1").fetchone()[0]
    s1_matched_tot = conn.execute("SELECT COUNT(*) FROM gt_source_combos").fetchone()[0]

    print(f"\nS1 Entity Match Source Combination Breakdown (among matched S1 entities = {s1_matched_tot:,}):")
    print(f"  - Matches ONLY Source 2 : {only_s2:,} ({round(only_s2*100.0/s1_matched_tot, 2)}%)")
    print(f"  - Matches ONLY Source 3 : {only_s3:,} ({round(only_s3*100.0/s1_matched_tot, 2)}%)")
    print(f"  - Matches BOTH S2 & S3  : {both_s2_s3:,} ({round(both_s2_s3*100.0/s1_matched_tot, 2)}%)")

    # ----------------------------------------------------
    # Q10: France / Test Set Special Analysis
    # ----------------------------------------------------
    print("\n--- Q10: TEST SET & FRANCE ANALYSIS ---")
    france_s1 = conn.execute("SELECT COUNT(*) FROM test_s1 WHERE country = 'France'").fetchone()[0]
    france_s2 = conn.execute("SELECT COUNT(*) FROM test_s2 WHERE country = 'France'").fetchone()[0]
    france_s3 = conn.execute("SELECT COUNT(*) FROM test_s3 WHERE country = 'France'").fetchone()[0]
    print(f"France Records in Test -> S1: {france_s1:,} | S2: {france_s2:,} | S3: {france_s3:,}")

    french_samples = conn.execute("""
        SELECT business_name, business_address FROM test_s1 WHERE country = 'France' LIMIT 5
    """).fetchall()
    print("\nSample France S1 Entities:")
    for fn, fa in french_samples:
        print(f"  - Name: {fn} | Address: {fa}")

    elapsed = round(time.time() - start_time, 2)
    print(f"\nDeep Analysis finished in {elapsed} seconds.")

file_info_map = {
    "train_s1": 2206821,
    "train_s2": 5034616,
    "train_s3": 5285603,
    "test_s1": 1732544,
    "test_s2": 4887273,
    "test_s3": 5082316
}

if __name__ == "__main__":
    main()
