import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from pathlib import Path

def translate_corpus():
    # 1. Setup Paths
    base_path = Path("/home/data/projets-aps/projet6/data/data_preprocessing/data_arab")
    input_dir = base_path / "standardized"
    output_dir = base_path / "standardized_with_french"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load NLLB Model (using the 600M variant for a good balance of speed/quality)
    print("⏳ Loading NLLB model...")
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move to GPU if available
    device = 0 if torch.cuda.is_available() else -1
    translator = pipeline("translation", model=model, tokenizer=tokenizer, 
                          src_lang="eng_Latn", tgt_lang="fra_Latn", max_length=400, device=device)

    # 3. Process Files
    for file_path in input_dir.glob("*.csv"):
        print(f"🚀 Processing: {file_path.name}")
        df = pd.read_csv(file_path)
        
        # We use a list to store results for batching
        english_texts = df['text_english'].astype(str).tolist()
        
        print(f"   Translating {len(english_texts)} lines...")
        # Batch size of 16-32 is usually safe for cluster memory
        translations = translator(english_texts, batch_size=16)
        
        # Extract the text from the dictionary results
        df['text_french'] = [t['translation_text'] for t in translations]
        
        # Save output
        output_path = output_dir / file_path.name
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    translate_corpus()
