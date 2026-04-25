import pandas as pd
from pathlib import Path

BASE = Path("/home/data/projets-aps/projet6")
XLS_FILE = BASE / "data" / "data_ewe" / "speech_ug" / "Ewe" / "selected transcribed audios" / "selected transcribed audios.xlsx"
OUT_FILE = BASE / "data_preprocessing" / "ewe" / "standardized" / "speech_ug_standardized.csv"

print(f"🚀 Processing Excel: {XLS_FILE.name}")

# We read the Excel. Given your image, the actual data starts after the 
# multi-row header. We'll search for the column 'Transcription' specifically.
df_xls = pd.read_excel(XLS_FILE)

# Based on your screenshot, the column name is exactly "Transcription"
# If it failed before, it's likely because of empty rows at the top.
if "Transcription" not in df_xls.columns:
    # Fallback: find the column that actually contains long Ewe strings
    for col in df_xls.columns:
        if df_xls[col].astype(str).str.len().mean() > 20:
            target_col = col
            break
else:
    target_col = "Transcription"

df_clean = pd.DataFrame({
    "database_origin": "speech_ug_excel",
    "id": df_xls.index.astype(str),
    "text_ewe": df_xls[target_col].astype(str),
    "text_french": ""
})

# Filter out empty/numeric-only rows that aren't actually text
df_clean = df_clean[df_clean["text_ewe"].str.len() > 5]

df_clean.to_csv(OUT_FILE, index=False, encoding="utf-8")
print(f"✅ Saved {len(df_clean)} rows to {OUT_FILE}")
