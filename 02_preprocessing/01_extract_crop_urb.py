"""
01_extract_crop_urb.py

Extracts binary cropland and urban/impervious layers from the 10-class
annual land-cover maps produced by 01_land_cover_export.js.

For each year (2000–2022), this script reads the reclassified 10-class
GeoTIFF and generates two binary masks:
  - Cropland: pixels with class value 1
  - Urban/Impervious: pixels with class value 7

Input:  GLC_FCS30D_{YYYY}_10classes.tif (10-class annual maps)
Output: crop_{YYYY}.tif, urb_{YYYY}.tif (binary masks, 0/1)
"""

import os
import glob
import rasterio
import numpy as np

# ============================================================================
# Configuration — update these paths to match your local directory structure
# ============================================================================
INPUT_DIR = r"D:\Sahel\10classes_GLC"
OUTPUT_DIR = r"D:\Sahel\crop_urb_GLC_FCS30D"

# Class codes in the 10-class system
CROPLAND_CODE = 1
URBAN_CODE = 7


def extract_binary_layers(input_dir: str, output_dir: str) -> None:
    """Extract binary cropland and urban masks from 10-class land-cover maps."""

    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(input_dir, "GLC_FCS30D_*_10classes.tif")
    tif_files = sorted(glob.glob(search_pattern))

    if not tif_files:
        print(f"No matching files found in {input_dir}")
        return

    print(f"Found {len(tif_files)} files. Processing...")

    for filepath in tif_files:
        filename = os.path.basename(filepath)

        # Parse year from filename
        try:
            year = filename.split('_')[2]
            assert year.isdigit() and len(year) == 4
        except (IndexError, AssertionError):
            print(f"  [Skip] Cannot parse year from: {filename}")
            continue

        print(f"  Year {year}...")

        with rasterio.open(filepath) as src:
            data = src.read(1)
            meta = src.meta.copy()
            meta.update(driver="GTiff", dtype="uint8", count=1,
                        compress="lzw", nodata=0)

            # Binary cropland mask
            crop_mask = np.where(data == CROPLAND_CODE, 1, 0).astype(np.uint8)
            crop_path = os.path.join(output_dir, f"crop_{year}.tif")
            with rasterio.open(crop_path, "w", **meta) as dst:
                dst.write(crop_mask, 1)

            # Binary urban mask
            urb_mask = np.where(data == URBAN_CODE, 1, 0).astype(np.uint8)
            urb_path = os.path.join(output_dir, f"urb_{year}.tif")
            with rasterio.open(urb_path, "w", **meta) as dst:
                dst.write(urb_mask, 1)

    print("Extraction complete.")


if __name__ == "__main__":
    extract_binary_layers(INPUT_DIR, OUTPUT_DIR)
