# src/unzip_all.py

import os
import zipfile
import shutil
from glob import glob

BASE_ROOT = r"C:\AquaCreds new model\BlueCarbon\Sentinel Tiles"

MONTHS = ["Dec", "Nov", "Oct", "Sept"]

print("=========================================")
print("   Automatic Sentinel-2 Unzip Tool")
print("=========================================\n")

for month in MONTHS:
    month_path = os.path.join(BASE_ROOT, month)
    
    if not os.path.exists(month_path):
        print(f"⚠ Month folder not found: {month_path}")
        continue
    
    print(f"\n📁 Checking: {month_path}")

    # find all zips
    zips = glob(os.path.join(month_path, "*.zip"))
    print(f"   → Found {len(zips)} ZIP files")

    for zip_path in zips:
        zip_name = os.path.basename(zip_path)
        print(f"\n🔄 Extracting: {zip_name}")

        # extract to temp folder
        temp_extract_path = os.path.join(month_path, zip_name.replace(".zip", ""))
        os.makedirs(temp_extract_path, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(temp_extract_path)
        except Exception as e:
            print(f"   ⛔ ERROR extracting {zip_name}: {e}")
            continue

        # FIND SAFE folder inside extraction
        safe_candidates = glob(os.path.join(temp_extract_path, "**", "*.SAFE"), recursive=True)

        if not safe_candidates:
            print(f"   ⛔ No SAFE folder found inside: {zip_name}")
            continue

        safe_folder = safe_candidates[0]
        safe_name = os.path.basename(safe_folder)
        final_safe_path = os.path.join(month_path, safe_name)

        # MOVE SAFE folder up to month directory
        if os.path.exists(final_safe_path):
            print(f"   ⚠ SAFE folder already exists: {safe_name}, skipping...")
        else:
            print(f"   📦 Moving SAFE folder → {final_safe_path}")
            shutil.move(safe_folder, final_safe_path)

        # CLEANUP: delete temporary extraction folder
        shutil.rmtree(temp_extract_path, ignore_errors=True)

        print(f"   ✓ Done extracting: {safe_name}")

print("\n=========================================")
print("   ALL EXTRACTIONS COMPLETED SUCCESSFULLY")
print("=========================================")
