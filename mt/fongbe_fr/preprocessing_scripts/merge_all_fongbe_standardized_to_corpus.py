import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
# Where the individual standardized files are located
INPUT_DIR = Path("/home/data/projets-aps/projet6/data_preprocessing/fongbe/standardized")

# The final destination for the combined corpus
OUTPUT_DIR = Path("/home/data/projets-aps/projet6/data_preprocessing/fongbe")
OUTPUT_FILE = OUTPUT_DIR / "fongbe_corpus.csv"

def merge_fongbe_corpus():
    """
    Reads all standardized CSVs from the input directory and 
    chains them together into one comprehensive master file.
    """
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_standardized_files = list(INPUT_DIR.glob("*_standardized.csv"))
    
    if not all_standardized_files:
        print(f"❌ No standardized files found in {INPUT_DIR}")
        return

    print(f"🚀 Found {len(all_standardized_files)} files to merge.")
    
    dataframes = []
    
    for csv_path in all_standardized_files:
        print(f"📦 Reading: {csv_path.name}")
        try:
            df = pd.read_csv(csv_path)
            dataframes.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {csv_path.name}: {e}")

    # Combine all dataframes
    if dataframes:
        print("🔗 Chaining files together...")
        combined_corpus = pd.concat(dataframes, ignore_index=True)
        
        # Save to the parent fongbe directory
        combined_corpus.to_csv(OUTPUT_FILE, index=False)
        
        print("\n✅ Merge Complete!")
        print(f"📍 Final Corpus: {OUTPUT_FILE}")
        print(f"📊 Total Rows: {len(combined_corpus)}")
        
        # Display breakdown by origin
        print("\n--- Breakdown by Origin ---")
        print(combined_corpus['database_origin'].value_counts())
    else:
        print("❌ No data was loaded. Merge aborted.")

if __name__ == "__main__":
    merge_fongbe_corpus()
