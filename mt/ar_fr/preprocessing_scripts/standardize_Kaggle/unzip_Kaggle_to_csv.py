import zipfile
from pathlib import Path

def just_extract():
    # Set the path to your Kaggle folder
    base_path = Path("/home/data/projets-aps/projet6/data/data_arab/Kaggle")
    zip_path = base_path / "archive.zip"
    
    if not zip_path.exists():
        print(f"❌ File not found: {zip_path}")
        return

    print(f"📦 Extracting {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract directly into the Kaggle folder
            zip_ref.extractall(base_path)
            
        print("✅ Extraction complete.")
        print("📂 Current files in folder:")
        for item in base_path.iterdir():
            print(f" - {item.name}")
            
    except Exception as e:
        print(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    just_extract()
