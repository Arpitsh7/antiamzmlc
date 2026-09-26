# 📊 2.5 GB TSV Dataset Analysis Suite

High-performance, memory-efficient Python Exploratory Data Analysis (EDA) suite powered by **DuckDB**. 

Processing a 2.5 GB TSV dataset using standard Pandas can take 10+ GB of RAM and potentially crash your memory. This tool uses DuckDB's vectorized, out-of-core SQL engine to analyze multi-gigabyte datasets in seconds without memory pressure.

---

## 📁 Directory Structure
```text
C:\Users\Arpit\.gemini\antigravity-ide\scratch\tsv_analyzer\
├── analyze.py           # Core EDA & HTML report generator script
├── generate_sample.py   # Utility script to generate dummy TSV files for testing
├── requirements.txt     # Python dependencies
└── README.md            # Quickstart guide
```

---

## 🚀 How to Run Analysis on Your 2.5 GB TSV File

### Command Usage

Run `analyze.py` by providing the path to your 2.5 GB TSV file using the `-f` / `--file` argument:

```bash
python analyze.py -f "C:\path\to\your_dataset.tsv"
```

### Options & Flags
| Flag | Short | Default | Description |
|---|---|---|---|
| `--file` | `-f` | *(Required)* | Absolute or relative path to your `.tsv` file |
| `--delimiter` | `-d` | `\t` (Tab) | Delimiter character (e.g. `\t` for TSV, `,` for CSV) |
| `--output` | `-o` | `eda_report.html` | Path to save the interactive HTML EDA report |

---

## 📊 Features Included in the Output Report

1. **Overall Dataset Metrics**:
   - Total rows & total columns count
   - Raw file size & memory footprint
   - Overall missing value cell percentage
   - Total processing execution time

2. **Column Profile & Summary Stats**:
   - **Data Types**: Auto-detected types (`BIGINT`, `VARCHAR`, `DOUBLE`, `TIMESTAMP`)
   - **Missing Data**: Visual missing percentage bar chart & row count
   - **Cardinality**: Distinct value counts and unique value percentage
   - **Numeric Columns**: Min, Max, Mean, Standard Deviation, Median, Q1 (25%), Q3 (75%)
   - **Categorical Columns**: Top 5 most frequent categories with exact counts and percentages

3. **Data Preview**:
   - 10-row sample preview table with high-contrast formatting

4. **Standalone Output**:
   - Generates `eda_report.html` which can be opened directly in Chrome, Edge, or Firefox.
