import pandas as pd
from pathlib import Path

def standardize_dialogue():
    input_path = Path("/home/data/projets-aps/projet6/data/data_arab/Egyptian_Dialogue/egyptian_dialogue.csv")
    output_path = Path("/home/data/projets-aps/projet6/data_preprocessing/data_arab/standardized/egyptian_dialogue_standardized.csv")
    
    df = pd.read_csv(input_path)
    
    standardized = pd.DataFrame({
        'database_origin': 'Egyptian_Dialogue_Corpus',
        'id': df['id'],
        'text_arab_egypt': df['src_text'],
        'text_english': df['tgt_text']
    })
    
    standardized.to_csv(output_path, index=False)
    print(f"✅ Dialogue standardized: {len(standardized)} rows")

if __name__ == "__main__":
    standardize_dialogue()
