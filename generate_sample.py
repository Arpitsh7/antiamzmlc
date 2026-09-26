import pandas as pd
import numpy as np

# Generate sample TSV dataset for testing
np.random.seed(42)
n_rows = 10000

df = pd.DataFrame({
    'user_id': np.random.randint(1000, 9999, size=n_rows),
    'category': np.random.choice(['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports'], size=n_rows),
    'price': np.round(np.random.uniform(5.0, 500.0, size=n_rows), 2),
    'quantity': np.random.randint(1, 10, size=n_rows),
    'rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], size=n_rows, p=[0.05, 0.1, 0.2, 0.35, 0.25, 0.05]),
    'created_at': pd.date_range(start='2025-01-01', periods=n_rows, freq='min')
})

df.to_csv(r'C:\Users\Arpit\.gemini\antigravity-ide\scratch\tsv_analyzer\sample_test.tsv', sep='\t', index=False)
print("Sample TSV created successfully!")
