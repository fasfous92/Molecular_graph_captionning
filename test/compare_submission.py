import pandas as pd
import os
def compare_submissions(file1_path, file2_path, output_diff_path="submission_diff.csv"):
    """
    Compares two submission CSVs (ID, description) and identifies changes.
    """
    print(f"--- Comparing Submissions ---")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    
    # 1. Load Data
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # Ensure IDs are strings to avoid mismatch issues
    df1['ID'] = df1['ID'].astype(str)
    df2['ID'] = df2['ID'].astype(str)
    
    # 2. Basic Validation
    if len(df1) != len(df2):
        print(f"WARNING: File lengths differ! ({len(df1)} vs {len(df2)})")
        # We will merge on ID, so it handles missing rows gracefully
        
    # 3. Merge on ID
    # suffixes=('_old', '_new') helps distinguish the two files
    merged = pd.merge(df1, df2, on='ID', suffixes=('_file1', '_file2'), how='inner')
    
    # 4. Find Differences
    # We compare the description strings
    merged['is_different'] = merged['description_file1'] != merged['description_file2']
    
    # Filter only the rows that are different
    diff_df = merged[merged['is_different']].copy()
    
    num_diffs = len(diff_df)
    total_rows = len(merged)
    percent_change = (num_diffs / total_rows) * 100
    
    # 5. Output Results
    print(f"\n--- Results ---")
    print(f"Total overlapping IDs: {total_rows}")
    print(f"Identical predictions: {total_rows - num_diffs}")
    print(f"Different predictions: {num_diffs}")
    print(f"Change Rate: {percent_change:.2f}%")
    
    if num_diffs > 0:
        # Save the differences to a file for review
        # We select just the relevant columns
        output_cols = ['ID', 'description_file1', 'description_file2']
        diff_df[output_cols].to_csv(output_diff_path, index=False)
        print(f"\nDifferences saved to: {output_diff_path}")
        
        # Show a few examples
        print("\n--- Example Changes ---")
        for i in range(min(3, num_diffs)):
            row = diff_df.iloc[i]
            print(f"ID: {row['ID']}")
            print(f"File 1: {row['description_file1'][:100]}...")
            print(f"File 2: {row['description_file2'][:100]}...")
            print("-" * 40)
    else:
        print("\nThe two files are identical!")

# --- Usage ---
if __name__ == "__main__":
    # Replace these with your actual filenames
    FILE_A = "submission_dual_tower_new_embedd.csv"  # e.g., output from Stage 1 only
    FILE_B = "submission_phase7_reranked.csv"      # e.g., output from Stage 2
    
    # If you only have one file generated right now, you can test it against itself 
    # or a previous version.
    
    if os.path.exists(FILE_A) and os.path.exists(FILE_B):
        compare_submissions(FILE_A, FILE_B)
    else:
        print("Please set FILE_A and FILE_B to your actual csv paths.")
