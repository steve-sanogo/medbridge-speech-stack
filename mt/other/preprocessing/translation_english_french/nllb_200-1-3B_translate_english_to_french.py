import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/data/projets-aps/projet6/data_preprocessing/data_arab")
INPUT_FOLDER = BASE_DIR / "standardized"
OUTPUT_FOLDER = BASE_DIR / "standardized_with_french"

MODEL_NAME = "facebook/nllb-200-1.3B"
SRC_LANG = "eng_Latn"
TGT_LANG = "fra_Latn"
BATCH_SIZE = 128  
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def translate_files():
    print(f"🚀 Initializing NLLB-200 (1.3B) on {DEVICE}...")
    
    # 1. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE)
    model.eval()

    # 2. Setup Output Directory
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # 3. Get all CSVs in the standardized folder
    input_files = list(INPUT_FOLDER.glob("*.csv"))
    print(f"📂 Found {len(input_files)} files to process.")

    for file_path in input_files:
        # Create output filename: original_name + _with_french.csv
        output_file = OUTPUT_FOLDER / f"{file_path.stem}_with_french.csv"
        print(f"\n📄 Processing: {file_path.name}")
        
        df = pd.read_csv(file_path)
        
        # Resume Logic per file
        if output_file.exists():
            df_existing = pd.read_csv(output_file)
            start_idx = len(df_existing)
            print(f"🔄 Resuming {file_path.name} from row {start_idx}...")
        else:
            start_idx = 0

        texts = df['text_english'].astype(str).tolist()
        
        # 4. Translation Loop
        for i in tqdm(range(start_idx, len(texts), BATCH_SIZE), desc=f"Translating {file_path.stem}"):
            batch = texts[i : i + BATCH_SIZE]
            
            tokenized = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            
            with torch.no_grad():
                generated = model.generate(
                    **tokenized, 
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
                    max_length=MAX_LENGTH,
                    num_beams=2
                )
            
            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            # 5. Build batch dataframe and append
            batch_df = df.iloc[i : i + len(preds)].copy()
            batch_df['text_french'] = preds
            
            batch_df.to_csv(
                output_file, 
                mode='a', 
                index=False, 
                header=not output_file.exists(),
                encoding='utf-8-sig'
            )

    print(f"\n✨ All files translated and saved in {OUTPUT_FOLDER}")

if __name__ == "__main__":
    translate_files()