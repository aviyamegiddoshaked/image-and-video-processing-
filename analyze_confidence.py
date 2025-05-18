import pandas as pd
import os

def analyze_confidence(csv_path):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist.")
        return

    try:
        df = pd.read_csv(csv_path)

        if 'time' not in df.columns or 'confidence' not in df.columns:
            print("CSV file must contain 'time' and 'confidence' columns.")
            return

        stats_by_time = df.groupby('time')['confidence'].agg([
            ('mean', 'mean'),
            ('variance', 'var'),
            ('standard_deviation', 'std'),
            ('minimum', 'min'),
            ('maximum', 'max'),
            ('count', 'count')
        ]).reset_index()

        #output_file = os.path.splitext(csv_path)[0] + '_analysis.csv'
        output_dir = "runs/confidence_analysis_outputs/after_blurring"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_analysis.csv")

        stats_by_time.to_csv(output_file, index=False)
        print(f"Analysis completed. Results saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze YOLOv5 detection confidence by frame.')
    parser.add_argument('csv_file', help='Path to the input CSV file')
    args = parser.parse_args()

    analyze_confidence(args.csv_file)
