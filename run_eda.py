import os
import sys
import time
import json
import duckdb
import pandas as pd

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_BASE = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset")
TRAIN_DIR = os.path.join(DATASET_BASE, "train")
TEST_DIR = os.path.join(DATASET_BASE, "test")
OUTPUT_HTML = "eda_report.html"

def run_eda():
    start_time = time.time()
    conn = duckdb.connect(database=':memory:')
    
    # Configure DuckDB for multi-threading
    conn.execute("SET threads TO 4;")

    print("==================================================")
    print(" BUSINESS ENTITY RESOLUTION DATASET EDA ")
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

    # Register views in DuckDB
    for key, path in files.items():
        safe_path = path.replace('\\', '/')
        conn.execute(f"""
            CREATE OR REPLACE VIEW {key} AS 
            SELECT * FROM read_csv_auto('{safe_path}', delim='\t', header=true, all_varchar=true, ignore_errors=true);
        """)

    # 1. Dataset Shape & File Sizes
    file_info = {}
    print("\n--- File Volume & Row Counts ---")
    for key, path in files.items():
        size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
        count = conn.execute(f"SELECT COUNT(*) FROM {key}").fetchone()[0]
        file_info[key] = {"path": path, "size_mb": size_mb, "rows": count}
        print(f"File {key:10s} | {size_mb:8.2f} MB | {count:,} rows")

    # 2. Country Breakdown across datasets
    print("\n--- Country Distribution ---")
    country_stats = {}
    for key in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
        dist = conn.execute(f"""
            SELECT COALESCE(country, 'MISSING') as country, COUNT(*) as cnt,
                   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {key}), 2) as pct
            FROM {key}
            GROUP BY country ORDER BY cnt DESC
        """).fetchall()
        country_stats[key] = dist
        print(f"  [{key}] -> {dist}")

    # 3. Field Completeness (% missing values)
    print("\n--- Missing Value Analysis ---")
    missing_stats = {}
    for key in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
        m_row = conn.execute(f"""
            SELECT 
                COUNT(*) - COUNT(entity_id) as missing_id,
                COUNT(*) - COUNT(business_name) as missing_name,
                COUNT(*) - COUNT(business_address) as missing_addr,
                COUNT(*) - COUNT(country) as missing_country
            FROM {key}
        """).fetchone()
        tot = file_info[key]["rows"]
        missing_stats[key] = {
            "name_pct": round(m_row[1] * 100.0 / tot, 2),
            "addr_pct": round(m_row[2] * 100.0 / tot, 2),
            "country_pct": round(m_row[3] * 100.0 / tot, 2)
        }
        print(f"  [{key}] missing -> Name: {m_row[1]} ({missing_stats[key]['name_pct']}%), Address: {m_row[2]} ({missing_stats[key]['addr_pct']}%), Country: {m_row[3]} ({missing_stats[key]['country_pct']}%)")

    # 4. Ground Truth Match Statistics
    print("\n--- Train Ground Truth Match Distribution ---")
    gt_total = conn.execute("SELECT COUNT(*) FROM train_gt").fetchone()[0]
    
    # Process matched_entity_ids string column
    conn.execute("""
        CREATE OR REPLACE VIEW train_gt_parsed AS
        SELECT 
            source1_entity_id,
            matched_entity_ids,
            CASE 
                WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 0
                ELSE LENGTH(matched_entity_ids) - LENGTH(REPLACE(matched_entity_ids, ',', '')) + 1
            END as num_matches
        FROM train_gt;
    """)

    match_dist = conn.execute("""
        SELECT num_matches, COUNT(*) as cnt,
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM train_gt_parsed), 2) as pct
        FROM train_gt_parsed
        GROUP BY num_matches ORDER BY num_matches;
    """).fetchall()

    for num, cnt, pct in match_dist:
        label = f"{num} match(es)" if num > 0 else "0 matches (Singleton)"
        print(f"  - {label:25s}: {cnt:7,} S1 entities ({pct:5.2f}%)")

    # Source breakdown of matched entities (S2 vs S3)
    conn.execute("""
        CREATE OR REPLACE VIEW ground_truth_pairs AS
        SELECT 
            source1_entity_id,
            UNNEST(STRING_SPLIT(matched_entity_ids, ',')) as target_entity_id
        FROM train_gt
        WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
    """)

    pair_count = conn.execute("SELECT COUNT(*) FROM ground_truth_pairs").fetchone()[0]
    s2_pairs = conn.execute("SELECT COUNT(*) FROM ground_truth_pairs WHERE target_entity_id LIKE 'S2-%'").fetchone()[0]
    s3_pairs = conn.execute("SELECT COUNT(*) FROM ground_truth_pairs WHERE target_entity_id LIKE 'S3-%'").fetchone()[0]

    print(f"\nTotal True Candidate Pairs: {pair_count:,}")
    print(f"   - Matches to Source 2: {s2_pairs:,} ({round(s2_pairs*100.0/pair_count, 2)}%)")
    print(f"   - Matches to Source 3: {s3_pairs:,} ({round(s3_pairs*100.0/pair_count, 2)}%)")

    # 5. Entity Text Length Statistics
    print("\n--- String Length Statistics (Business Name & Address) ---")
    len_stats = {}
    for key in ["train_s1", "train_s2", "train_s3"]:
        row = conn.execute(f"""
            SELECT 
                ROUND(AVG(LENGTH(business_name)), 1) as avg_name_len,
                MAX(LENGTH(business_name)) as max_name_len,
                ROUND(AVG(LENGTH(business_address)), 1) as avg_addr_len,
                MAX(LENGTH(business_address)) as max_addr_len
            FROM {key}
        """).fetchone()
        len_stats[key] = {
            "avg_name_len": row[0], "max_name_len": row[1],
            "avg_addr_len": row[2], "max_addr_len": row[3]
        }
        print(f"  [{key}] Name Avg Len: {row[0]}, Max: {row[1]} | Address Avg Len: {row[2]}, Max: {row[3]}")

    # 6. Sample Ground Truth Matches Inspection
    print("\n--- Sample Ground Truth Matches (S1 vs Target S2/S3) ---")
    sample_pairs = conn.execute("""
        SELECT 
            gt.source1_entity_id,
            s1.business_name as s1_name,
            s1.business_address as s1_addr,
            gt.target_entity_id,
            COALESCE(s2.business_name, s3.business_name) as target_name,
            COALESCE(s2.business_address, s3.business_address) as target_addr,
            s1.country
        FROM ground_truth_pairs gt
        JOIN train_s1 s1 ON gt.source1_entity_id = s1.entity_id
        LEFT JOIN train_s2 s2 ON gt.target_entity_id = s2.entity_id
        LEFT JOIN train_s3 s3 ON gt.target_entity_id = s3.entity_id
        LIMIT 5;
    """).fetchall()

    sample_pair_list = []
    for s1_id, s1_name, s1_addr, tgt_id, tgt_name, tgt_addr, country in sample_pairs:
        print(f"Country: {country}")
        print(f"  S1  [{s1_id}]: {s1_name} | {s1_addr}")
        print(f"  Tgt [{tgt_id}]: {tgt_name} | {tgt_addr}\n")
        sample_pair_list.append({
            "s1_id": s1_id, "s1_name": str(s1_name), "s1_addr": str(s1_addr),
            "tgt_id": tgt_id, "tgt_name": str(tgt_name), "tgt_addr": str(tgt_addr),
            "country": str(country)
        })

    elapsed = round(time.time() - start_time, 2)
    print(f"Analysis finished in {elapsed} seconds.")

    summary_data = {
        "file_info": file_info,
        "country_stats": country_stats,
        "missing_stats": missing_stats,
        "gt_total": gt_total,
        "match_dist": [{"matches": num, "count": cnt, "pct": pct} for num, cnt, pct in match_dist],
        "pair_count": pair_count,
        "s2_pairs": s2_pairs,
        "s3_pairs": s3_pairs,
        "len_stats": len_stats,
        "sample_pairs": sample_pair_list,
        "elapsed": elapsed
    }
    
    generate_html_eda_report(summary_data)
    print(f"HTML EDA Report generated at: {os.path.abspath(OUTPUT_HTML)}")

def generate_html_eda_report(data):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Challenge 2026 - Entity Resolution EDA Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-purple: #a855f7;
            --accent-green: #22c55e;
            --accent-amber: #f59e0b;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            padding: 2rem;
            margin: 0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
        .header h1 {{
            font-size: 2.2rem;
            margin: 0 0 0.5rem 0;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header p {{
            color: var(--text-muted);
            margin: 0;
        }}
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
        }}
        .card .title {{
            font-size: 0.85rem;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        .card .val {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent-cyan);
        }}
        .section {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .section h2 {{
            font-size: 1.3rem;
            margin-top: 0;
            margin-bottom: 1rem;
            border-left: 4px solid var(--accent-purple);
            padding-left: 0.75rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: rgba(255, 255, 255, 0.05);
            color: var(--text-muted);
            font-weight: 600;
        }}
        .badge {{
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-s1 {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; }}
        .badge-s2 {{ background: rgba(168, 85, 247, 0.2); color: #a855f7; }}
        .badge-s3 {{ background: rgba(34, 197, 94, 0.2); color: #22c55e; }}
        .badge-us {{ background: rgba(245, 158, 11, 0.2); color: #f59e0b; }}
        code {{
            background: rgba(0, 0, 0, 0.3);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ML Challenge 2026: Business Entity Resolution EDA</h1>
            <p>Exploratory Data Analysis Report for Train & Test Datasets (Source 1, Source 2, Source 3)</p>
        </div>

        <div class="grid-cards">
            <div class="card">
                <div class="title">Train Source 1 Entities</div>
                <div class="val">{data['file_info']['train_s1']['rows']:,}</div>
            </div>
            <div class="card">
                <div class="title">Train Ground Truth Pairs</div>
                <div class="val">{data['pair_count']:,}</div>
            </div>
            <div class="card">
                <div class="title">Singleton Entities (0 Matches)</div>
                <div class="val">{data['match_dist'][0]['count']:,} ({data['match_dist'][0]['pct']}%)</div>
            </div>
            <div class="card">
                <div class="title">Test Source 1 Entities</div>
                <div class="val">{data['file_info']['test_s1']['rows']:,}</div>
            </div>
        </div>

        <div class="section">
            <h2>Dataset Files & Volume Overview</h2>
            <table>
                <thead>
                    <tr>
                        <th>File Name</th>
                        <th>Dataset Split</th>
                        <th>File Size (MB)</th>
                        <th>Total Rows</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f"<tr><td><code>{k}</code></td><td>{'Train' if 'train' in k else 'Test'}</td><td>{v['size_mb']} MB</td><td>{v['rows']:,}</td></tr>" for k, v in data['file_info'].items())}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Ground Truth Match Distribution (Train S1)</h2>
            <p>Distribution of matches per Source 1 reference entity in the training dataset:</p>
            <table>
                <thead>
                    <tr>
                        <th>Matches per Entity</th>
                        <th>Count of S1 Entities</th>
                        <th>Percentage (%)</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f"<tr><td><b>{item['matches']} match(es)</b> {'(Singleton)' if item['matches']==0 else ''}</td><td>{item['count']:,}</td><td>{item['pct']}%</td></tr>" for item in data['match_dist'])}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Sample Ground Truth Matches</h2>
            <table>
                <thead>
                    <tr>
                        <th>S1 Entity ID</th>
                        <th>S1 Business Name & Address</th>
                        <th>Matched Target ID</th>
                        <th>Target Business Name & Address</th>
                        <th>Country</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f"<tr><td><span class='badge badge-s1'>{p['s1_id']}</span></td><td><b>{p['s1_name']}</b><br/><small>{p['s1_addr']}</small></td><td><span class='badge badge-s2'>{p['tgt_id']}</span></td><td><b>{p['tgt_name']}</b><br/><small>{p['tgt_addr']}</small></td><td><span class='badge badge-us'>{p['country']}</span></td></tr>" for p in data['sample_pairs'])}
                </tbody>
            </table>
        </div>

    </div>
</body>
</html>
"""
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    run_eda()
