import pandas as pd
from pathlib import Path

def standardize_mendeley():
    input_path = Path("/home/data/projets-aps/projet6/data/data_arab/Mendeley/mendeley_egyptian_english.csv")
    output_path = Path("/home/data/projets-aps/projet6/data_preprocessing/data_arab/standardized/mendeley_standardized.csv")
    
    # Read without headers
    df = pd.read_csv(input_path, header=None)
    
    standardized = pd.DataFrame({
        'database_origin': 'Mendeley_LittlePrince',
        'id': range(len(df)),
        'text_arab_egypt': df[0],
        'text_english': df[1]
    })
    
    standardized.to_csv(output_path, index=False)
    print(f"✅ Mendeley standardized: {len(standardized)} rows")

if __name__ == "__main__":
    standardize_mendeley()
