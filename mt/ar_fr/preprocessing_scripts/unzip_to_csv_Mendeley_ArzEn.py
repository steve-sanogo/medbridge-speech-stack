import zipfile
import pandas as pd
from pathlib import Path

def process_mendeley_zip():
    # Define paths based on your terminal output
    base_path = Path("/home/data/projets-aps/projet6/data/data_arab/Mendeley")
    zip_file_path = base_path / "Mendely_ArzEn.zip"
    output_csv = base_path / "mendeley_egyptian_english.csv"
    extract_temp = base_path / "temp_extracted"

    if not zip_file_path.exists():
        print(f"❌ Error: {zip_file_path} not found.")
        return

    print(f"📦 Unzipping {zip_file_path.name}...")
    
    try:
        # 1. Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_temp)
        
        # 2. Find the data file (usually .xlsx or .csv)
        data_files = list(extract_temp.glob("**/*.xlsx")) + list(extract_temp.glob("**/*.csv"))
        
        if not data_files:
            print("❌ No Excel or CSV files found inside the zip.")
            return

        # 3. Read the first data file found and save as CSV
        target_file = data_files[0]
        print(f"📄 Converting {target_file.name} to CSV...")
        
        if target_file.suffix == '.xlsx':
            # Note: requires 'pip install openpyxl'
            df = pd.read_excel(target_file, engine='openpyxl')
        else:
            df = pd.read_csv(target_file)

        # 4. Save to CSV
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✅ Success! Data saved to: {output_csv}")

        # Optional: Clean up temp folder
        import shutil
        shutil.rmtree(extract_temp)
        print("🧹 Temporary files cleaned up.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    process_mendeley_zip()
