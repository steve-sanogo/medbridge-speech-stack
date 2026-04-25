import pandas as pd
import dtale
import time
from pathlib import Path

def clean_sequence_and_show():
    # Update these paths to your environment
    input_path = "/home/data/projets-aps/projet6/data/data_arab/Kaggle/train.csv"
    output_path = "/home/data/projets-aps/projet6/data_preprocessing/data_arab/kaggle_sequencing_step/Kaggle_Sequenced.csv"
    
    # 1. Load and drop 'ar'
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    df = df[df['language'] != 'ar'].reset_index(drop=True)
    
    keep_indices = []
    i = 0
    total_rows = len(df)
    
    # 2. Sequence Validation Loop (NEW RHYTHM: en -> ar_eg)
    print("Validating sequence rhythm (en -> ar_eg)...")
    while i < total_rows:
        current_lang = df.iloc[i]['language']
        
        # We now look for the English row as the START of the pair
        if current_lang == 'en':
            # Check if there is a next row and if it is Egyptian
            if i + 1 < total_rows and df.iloc[i+1]['language'] == 'ar_eg':
                # Match Found!
                keep_indices.append(i)     # Keep the en
                keep_indices.append(i + 1) # Keep the ar_eg
                i += 2 
            else:
                # Stray 'en' without an Egyptian translation following it
                i += 1
        else:
            # Stray 'ar_eg' without a leading English row
            i += 1
            
    # 3. Apply Cleaning
    df_sequenced = df.iloc[keep_indices].copy()
    
    # Rhythm check: 0 = en, 1 = ar_eg
    df_sequenced['rhythm_check'] = [i % 2 for i in range(len(df_sequenced))]
    
    # 4. Save
    df_sequenced.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n--- Sequence Audit ---")
    print(f"Initial (no ar): {total_rows}")
    print(f"Final Sequenced: {len(df_sequenced)}")
    print(f"Pairs Created:   {len(df_sequenced) // 2}")
    
    # 5. Launch D-Tale
    print(f"\n🚀 Launching D-Tale on port 40005...")
    d = dtale.show(df_sequenced, host='localhost', port=40005)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down D-Tale. Goodbye!")

if __name__ == "__main__":
    clean_sequence_and_show()