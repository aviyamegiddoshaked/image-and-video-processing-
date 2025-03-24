import pandas as pd
import sys
import os
import re

# Sanitize filenames to avoid slashes/spaces/illegal characters
def clean_filename(name):
    return re.sub(r'[\\/:*?"<>| ]', '_', name.split('.')[0])

# Check if two arguments were passed
if len(sys.argv) != 3:
    print("Usage: python compare_radius_data.py <file1.csv> <file2.csv>")
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]

# Load the CSVs
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge the dataframes on 'time'
merged_df = pd.merge(df1, df2, on="time", suffixes=('_1', '_2'))

# Calculate differences
comparison_df = pd.DataFrame()
comparison_df['time'] = merged_df['time']
comparison_df['mean_diff'] = merged_df['mean_1'] - merged_df['mean_2']
comparison_df['variance_diff'] = merged_df['variance_1'] - merged_df['variance_2']
comparison_df['std_diff'] = merged_df['standard_deviation_1'] - merged_df['standard_deviation_2']
comparison_df['min_diff'] = merged_df['minimum_1'] - merged_df['minimum_2']
comparison_df['max_diff'] = merged_df['maximum_1'] - merged_df['maximum_2']
comparison_df['count_diff'] = merged_df['count_1'] - merged_df['count_2']

# Create output folder if it doesn't exist
output_dir = "comparisons"
os.makedirs(output_dir, exist_ok=True)

# Define safe output filename
output_filename = os.path.join(
    output_dir,
    f"comparison_{clean_filename(file1)}_vs_{clean_filename(file2)}.csv"
)

# Save and print
comparison_df.to_csv(output_filename, index=False)
print(f"Comparison saved to {output_filename}\n")
print(comparison_df.head())
