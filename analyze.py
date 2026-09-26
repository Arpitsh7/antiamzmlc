"""
TSV/CSV Big Data Analyzer (Memory-Efficient DuckDB Engine)
Designed for high-performance analysis of large datasets (e.g., 2.5 GB+) without memory overload.
"""

import os
import sys
import time
import argparse
import duckdb
import pandas as pd
from typing import Dict, Any, List

def format_bytes(size: float) -> str:
    """Format byte sizes into human readable strings."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def format_number(n: int) -> str:
    """Format integers with commas."""
    return f"{n:,}"

class DatasetAnalyzer:
    def __init__(self, file_path: str, delimiter: str = '\t'):
        self.file_path = os.path.abspath(file_path)
        self.delimiter = delimiter
        self.conn = duckdb.connect(database=':memory:')
        self.file_size = os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0

    def run_analysis(self) -> Dict[str, Any]:
        print(f"🚀 Starting analysis for dataset: {self.file_path}")
        print(f"📦 File Size: {format_bytes(self.file_size)}")
        start_time = time.time()

        # Sanitize file path for DuckDB SQL query
        safe_path = self.file_path.replace('\\', '/')
        
        # Determine exact delimiter for read_csv_auto
        delim_sql = f"delim='{self.delimiter}'" if self.delimiter else "auto_detect=true"
        
        # View creation using DuckDB read_csv_auto
        view_query = f"""
            CREATE OR REPLACE VIEW dataset AS 
            SELECT * FROM read_csv_auto('{safe_path}', {delim_sql}, ignore_errors=true);
        """
        self.conn.execute(view_query)

        # 1. General Overview
        cols_info = self.conn.execute("DESCRIBE dataset;").fetchall()
        column_names = [col[0] for col in cols_info]
        column_types = [col[1] for col in cols_info]
        
        total_rows = self.conn.execute("SELECT COUNT(*) FROM dataset;").fetchone()[0]
        total_cols = len(column_names)

        print(f"📊 Dataset Shape: {format_number(total_rows)} rows × {total_cols} columns")

        # 2. Column-by-column statistics
        column_stats = []
        total_missing_cells = 0
        total_total_cells = total_rows * total_cols if total_rows else 1

        print("🔍 Computing column statistics...")
        for col_name, col_type in zip(column_names, column_types):
            col_safe = f'"{col_name}"'
            
            # Missing & Distinct counts
            stats_query = f"""
                SELECT 
                    COUNT(*) - COUNT({col_safe}) AS missing_count,
                    COUNT(DISTINCT {col_safe}) AS distinct_count
                FROM dataset;
            """
            missing_cnt, distinct_cnt = self.conn.execute(stats_query).fetchone()
            total_missing_cells += missing_cnt
            missing_pct = (missing_cnt / total_rows * 100) if total_rows > 0 else 0
            cardinality_pct = (distinct_cnt / total_rows * 100) if total_rows > 0 else 0

            col_detail = {
                "name": col_name,
                "type": col_type,
                "missing_count": missing_cnt,
                "missing_pct": missing_pct,
                "distinct_count": distinct_cnt,
                "cardinality_pct": cardinality_pct,
                "numeric_stats": None,
                "top_categories": None
            }

            # Check if column is numeric
            is_numeric = any(t in col_type.upper() for t in ['INT', 'FLOAT', 'DOUBLE', 'DECIMAL', 'HUGEINT', 'BIGINT'])
            if is_numeric:
                try:
                    num_query = f"""
                        SELECT 
                            MIN({col_safe}) AS min_val,
                            MAX({col_safe}) AS max_val,
                            AVG({col_safe}) AS mean_val,
                            STDDEV({col_safe}) AS std_val,
                            MEDIAN({col_safe}) AS median_val,
                            QUANTILE_CONT({col_safe}, 0.25) AS p25_val,
                            QUANTILE_CONT({col_safe}, 0.75) AS p75_val
                        FROM dataset;
                    """
                    row = self.conn.execute(num_query).fetchone()
                    col_detail["numeric_stats"] = {
                        "min": round(row[0], 4) if row[0] is not None else None,
                        "max": round(row[1], 4) if row[1] is not None else None,
                        "mean": round(row[2], 4) if row[2] is not None else None,
                        "std": round(row[3], 4) if row[3] is not None else None,
                        "median": round(row[4], 4) if row[4] is not None else None,
                        "p25": round(row[5], 4) if row[5] is not None else None,
                        "p75": round(row[6], 4) if row[6] is not None else None,
                    }
                except Exception:
                    pass
            else:
                # Top Categories for non-numeric or categorical columns
                try:
                    cat_query = f"""
                        SELECT {col_safe}::VARCHAR AS val, COUNT(*) AS cnt
                        FROM dataset
                        WHERE {col_safe} IS NOT NULL
                        GROUP BY 1
                        ORDER BY cnt DESC
                        LIMIT 5;
                    """
                    top_rows = self.conn.execute(cat_query).fetchall()
                    col_detail["top_categories"] = [
                        {"value": str(r[0]), "count": r[1], "pct": round((r[1] / total_rows) * 100, 2)}
                        for r in top_rows
                    ]
                except Exception:
                    pass

            column_stats.append(col_detail)

        # 3. Sample Rows Preview (Top 10)
        sample_df = self.conn.execute("SELECT * FROM dataset LIMIT 10;").fetchdf()
        sample_records = sample_df.to_dict(orient='records')

        execution_time = round(time.time() - start_time, 2)
        print(f"✅ Analysis completed in {execution_time} seconds!")

        return {
            "file_name": os.path.basename(self.file_path),
            "file_path": self.file_path,
            "file_size": format_bytes(self.file_size),
            "total_rows": total_rows,
            "total_cols": total_cols,
            "total_missing_pct": round((total_missing_cells / total_total_cells) * 100, 2),
            "execution_time": execution_time,
            "column_stats": column_stats,
            "sample_records": sample_records,
            "column_names": column_names
        }

def generate_html_report(data: Dict[str, Any], output_path: str = "eda_report.html"):
    """Generates a standalone, rich interactive HTML report."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Analysis Report - {data['file_name']}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-purple: #c084fc;
            --accent-green: #4ade80;
            --accent-red: #f87171;
            --border-color: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
        body {{ background-color: var(--bg-primary); color: var(--text-primary); padding: 2rem; line-height: 1.5; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 1.5rem; margin-bottom: 2rem; }}
        .header h1 {{ font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .header .meta {{ color: var(--text-secondary); font-size: 0.9rem; text-align: right; }}

        .grid-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.5rem; margin-bottom: 2.5rem; }}
        .metric-card {{ background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.5rem; position: relative; overflow: hidden; }}
        .metric-card::before {{ content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--accent-blue); }}
        .metric-title {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); font-weight: 600; margin-bottom: 0.5rem; }}
        .metric-value {{ font-size: 1.8rem; font-weight: 700; color: var(--text-primary); }}

        .section-title {{ font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem; border-left: 4px solid var(--accent-purple); padding-left: 0.75rem; }}

        .card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.5rem; margin-bottom: 2.5rem; overflow-x: auto; }}

        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }}
        th {{ background-color: rgba(51, 65, 85, 0.5); color: var(--text-secondary); font-weight: 600; padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); white-space: nowrap; }}
        td {{ padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); color: var(--text-primary); }}
        tr:hover {{ background-color: rgba(255, 255, 255, 0.02); }}

        .badge {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }}
        .badge-type {{ background-color: rgba(56, 189, 248, 0.15); color: var(--accent-blue); }}
        .progress-bar {{ width: 100%; background: var(--border-color); height: 6px; border-radius: 3px; overflow: hidden; margin-top: 4px; }}
        .progress-fill {{ height: 100%; background: var(--accent-blue); border-radius: 3px; }}
        .progress-missing {{ background: var(--accent-red); }}

        .top-cat-item {{ display: flex; justify-content: space-between; font-size: 0.8rem; padding: 2px 0; }}
        .top-cat-val {{ color: var(--text-secondary); max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>📊 Dataset Analysis Report</h1>
            <p style="color: var(--text-secondary); font-size: 0.95rem; margin-top: 0.3rem;">File: <code>{data['file_name']}</code></p>
        </div>
        <div class="meta">
            <p>Engine: DuckDB High-Performance SQL</p>
            <p>Processed in <strong>{data['execution_time']}s</strong></p>
        </div>
    </div>

    <!-- Metrics Grid -->
    <div class="grid-metrics">
        <div class="metric-card">
            <div class="metric-title">Total Rows</div>
            <div class="metric-value">{format_number(data['total_rows'])}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Total Columns</div>
            <div class="metric-value">{data['total_cols']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">File Size</div>
            <div class="metric-value">{data['file_size']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Missing Data Cell %</div>
            <div class="metric-value" style="color: {'var(--accent-red)' if data['total_missing_pct'] > 10 else 'var(--accent-green)'}">
                {data['total_missing_pct']}%
            </div>
        </div>
    </div>

    <!-- Column Profile -->
    <div class="section-title">Column Profile & Statistics</div>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Column Name</th>
                    <th>Data Type</th>
                    <th>Missing Values</th>
                    <th>Unique Values</th>
                    <th>Numerical Stats (Min / Mean / Max / Std)</th>
                    <th>Top Frequent Values</th>
                </tr>
            </thead>
            <tbody>
"""

    for col in data['column_stats']:
        missing_bar = f"""
            <div>{format_number(col['missing_count'])} ({col['missing_pct']:.1f}%)</div>
            <div class="progress-bar"><div class="progress-fill progress-missing" style="width: {min(col['missing_pct'], 100)}%;"></div></div>
        """
        
        cardinality = f"{format_number(col['distinct_count'])} ({col['cardinality_pct']:.1f}%)"
        
        num_str = "-"
        if col['numeric_stats']:
            s = col['numeric_stats']
            num_str = f"""
                <strong>Min:</strong> {s['min']} | <strong>Max:</strong> {s['max']}<br>
                <strong>Mean:</strong> {s['mean']} | <strong>Std:</strong> {s['std']}<br>
                <strong>Median:</strong> {s['median']} [Q1: {s['p25']}, Q3: {s['p75']}]
            """
        
        top_cat_html = "-"
        if col['top_categories']:
            items = []
            for cat in col['top_categories']:
                val_disp = cat['value'] if cat['value'] != '' else '(blank)'
                items.append(f"""
                    <div class="top-cat-item">
                        <span class="top-cat-val" title="{val_disp}">{val_disp}</span>
                        <span>{format_number(cat['count'])} ({cat['pct']}%)</span>
                    </div>
                """)
            top_cat_html = "".join(items)

        html_content += f"""
                <tr>
                    <td><strong>{col['name']}</strong></td>
                    <td><span class="badge badge-type">{col['type']}</span></td>
                    <td>{missing_bar}</td>
                    <td>{cardinality}</td>
                    <td style="font-size: 0.85rem;">{num_str}</td>
                    <td style="min-width: 220px;">{top_cat_html}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </div>

    <!-- Data Preview -->
    <div class="section-title">Data Preview (First 10 Rows)</div>
    <div class="card">
        <table>
            <thead>
                <tr>
"""
    for c in data['column_names']:
        html_content += f"<th>{c}</th>"
    
    html_content += """
                </tr>
            </thead>
            <tbody>
"""

    for row in data['sample_records']:
        html_content += "<tr>"
        for c in data['column_names']:
            val = str(row.get(c, ''))
            if len(val) > 40:
                val = val[:37] + "..."
            html_content += f"<td>{val}</td>"
        html_content += "</tr>"

    html_content += f"""
            </tbody>
        </table>
    </div>

    <footer style="text-align: center; color: var(--text-secondary); font-size: 0.85rem; margin-top: 2rem;">
        Generated by Antigravity Python EDA Tool • Engine: DuckDB
    </footer>

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✨ Interactive HTML Report saved to: {os.path.abspath(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="Analyze 2.5 GB+ TSV/CSV datasets fast using DuckDB.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to the TSV/CSV file")
    parser.add_argument("-d", "--delimiter", type=str, default="\t", help="Delimiter (default: tab '\\t')")
    parser.add_argument("-o", "--output", type=str, default="eda_report.html", help="Output HTML report path")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ Error: File not found at path: {args.file}")
        sys.exit(1)

    analyzer = DatasetAnalyzer(file_path=args.file, delimiter=args.delimiter)
    results = analyzer.run_analysis()
    generate_html_report(results, output_path=args.output)

if __name__ == "__main__":
    main()
