import pandas as pd
from pathlib import Path
import dtale
import time

def standardize_kaggle_perfect_pairs():
    input_path = Path("/home/data/projets-aps/projet6/data_preprocessing/data_arab/kaggle_sequencing_step/Kaggle_Sequenced.csv")
    output_path = Path("/home/data/projets-aps/projet6/data_preprocessing/data_arab/standardized/Kaggle_standardized.csv")
    
    # 1. Load data
    df = pd.read_csv(input_path)
    
    # 2. Filter out Standard Arabic (ar), keep only English and Egyptian
    # We also keep category/sub_category to ensure we group correctly
    df_filtered = df[df['language'].isin(['ar_eg', 'en'])].copy()
    
    # 3. Create a 'Rank' per category group
    # This identifies 'Fact #1', 'Fact #2', etc., within each sub-topic
    df_filtered['fact_rank'] = df_filtered.groupby(['category', 'sub_category', 'language']).cumcount()
    
    # 4. Separate into two dataframes
    egy = df_filtered[df_filtered['language'] == 'ar_eg']
    eng = df_filtered[df_filtered['language'] == 'en']
    
    # 5. Merge on the metadata + rank
    # This forces the 1st Egyptian "History" fact to pair with the 1st English "History" fact
    merged = pd.merge(
        egy, 
        eng, 
        on=['category', 'sub_category', 'fact_rank'], 
        suffixes=('_egy', '_en')
    )
    
    # 6. Combine Questions and Answers (Transposition)
    # Since each fact has a Question and an Answer, we actually have TWO pairs per fact.
    # We will create a list for each column.
    
    data_rows = []
    for _, row in merged.iterrows():
        # Pair 1: The Questions
        data_rows.append({
            'database_origin': 'Kaggle_Facts_Q',
            'text_arab_egypt': row['question_egy'],
            'text_english': row['question_en']
        })
        # Pair 2: The Answers
        data_rows.append({
            'database_origin': 'Kaggle_Facts_A',
            'text_arab_egypt': row['answer_egy'],
            'text_english': row['answer_en']
        })
    
    # 7. Create final DataFrame with IDs
    standardized = pd.DataFrame(data_rows)
    standardized.insert(1, 'id', range(len(standardized)))
    
    # Final cleanup: Remove any empty or NaN rows
    standardized = standardized.dropna(subset=['text_arab_egypt', 'text_english'])
    
    # 8. Save
    standardized.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Success!")
    print(f"Merged Facts: {len(merged)}")
    print(f"Total Standardized Rows (Q+A): {len(standardized)}")
    
    return standardized

if __name__ == "__main__":
    standardized_df = standardize_kaggle_perfect_pairs()

    d = dtale.show(standardized_df, host='localhost', port='40001')

    print(f"\n🚀 SUCCESS!")
    print("Spreadsheet is running at:", d._main_url)
    print("!!! KEEP THIS TERMINAL OPEN !!!")
    print("Press Ctrl+C to stop the server and exit.")

    # 3. The Keep-Alive Loop
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server... Goodbye!")