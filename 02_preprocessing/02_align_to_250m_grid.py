"""
02_align_to_250m_grid.py

Harmonizes all input raster datasets to a common 250 m reference grid.

The reference grid is derived from the NDVI 2000 raster (MODIS MOD13Q1
at 250 m), clipped to the study area shapefile. Each input raster is
reprojected and resampled to match this grid exactly (same CRS, extent,
resolution, and pixel alignment).

Resampling methods:
  - Categorical data (land cover, burned area): nearest-neighbor
  - Continuous data (climate, population, NDVI, SOC, livestock): bilinear

Input:  Raw GeoTIFFs organized by dataset in separate directories.
Output: Aligned GeoTIFFs in a single output directory with subfolders.
"""

import os
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling

# ============================================================================
# Configuration — update paths to match your local directory structure
# ============================================================================

BASE_OUTPUT_DIR = Path(r"D:\Sahel\Dataset_Processed_250m")
REFERENCE_RASTER = Path(r"D:\Sahel\GEE_NDVI_250m\SAHEL_NDVI_250m_2000.tif")
SHAPEFILE = Path(r"D:\Sahel\roi\Sahel_ROI_Download.shp")

# Each entry defines an input directory, output subfolder, and data type.
# Data type controls the resampling method: 'categorical' → nearest,
# 'continuous' → bilinear.
DATASETS = [
    {"input_dir": r"D:\Sahel\ipcc_GLC",
     "output_folder": "ipcc_GLC", "type": "categorical"},

    {"input_dir": r"D:\Sahel\crop_urb_GLC_FCS30D",
     "output_folder": "crop_urb", "type": "categorical"},

    {"input_dir": r"D:\Sahel\GEE_NDVI_250m",
     "output_folder": "ndvi", "type": "continuous"},

    {"input_dir": r"D:\Sahel\GEE_SAHEL_other_Data",
     "output_folder": "climate_terrain", "type": "continuous"},

    {"input_dir": r"D:\IA and drivers\rec by ZJH\dataset\globPOP",
     "output_folder": "population", "type": "continuous"},

    {"input_dir": r"D:\IA and drivers\rec by ZJH\dataset\mining",
     "output_folder": "mining", "type": "continuous"},

    {"input_dir": r"D:\IA and drivers\rec by ZJH\dataset\livestock",
     "output_folder": "livestock", "type": "continuous"},

    {"input_dir": r"D:\Sahel\SOC",
     "output_folder": "soc", "type": "continuous"},

    {"input_dir": r"D:\Sahel\GEE_BURN_500m",
     "output_folder": "fire", "type": "categorical"},
]

# Livestock filename fragments to keep (filter out non-target species)
LIVESTOCK_KEYS = ['CTL', 'GTS', 'HRS', 'SHP', '_CT_', '_GT_', '_HO_', '_SH_']


# ============================================================================
# Processing Functions
# ============================================================================

def preprocess_raster(src_path: Path, output_path: Path,
                      ref_profile: dict, resampling_method) -> None:
    """Warp a source raster to match the reference grid."""

    if output_path.exists():
        print(f"  [Skip] Already exists: {output_path.name}")
        return

    try:
        with rasterio.open(src_path) as src:
            src_nodata = src.nodata
            if src_nodata is None:
                src_nodata = (0 if resampling_method == Resampling.nearest
                              else np.nan)

            dst_array = np.full(
                (src.count, ref_profile['height'], ref_profile['width']),
                src_nodata, dtype=ref_profile['dtype']
            )

            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),
                destination=dst_array,
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=ref_profile['transform'],
                dst_crs=ref_profile['crs'],
                dst_nodata=src_nodata,
                resampling=resampling_method
            )

            out_meta = ref_profile.copy()
            out_meta.update(count=src.count, nodata=src_nodata,
                            dtype=ref_profile['dtype'])

            with rasterio.open(output_path, 'w', **out_meta) as dst:
                dst.write(dst_array)

        print(f"  [Done] {src_path.name}")
    except Exception as e:
        print(f"  [Error] {src_path.name}: {e}")


def build_reference_profile(ref_raster: Path, shapefile: Path) -> dict:
    """Build the target raster profile from a reference raster clipped to ROI."""

    gdf = gpd.read_file(shapefile)

    with rasterio.open(ref_raster) as ref_src:
        gdf_proj = gdf.to_crs(ref_src.crs)
        shapes = [f["geometry"] for f in gdf_proj.__geo_interface__["features"]]
        out_image, out_transform = mask(ref_src, shapes, crop=True)

        profile = ref_src.meta.copy()
        profile.update(
            driver="GTiff",
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            compress="lzw"
        )

    return profile


def main():
    print("=== Aligning all datasets to 250 m reference grid ===\n")
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build reference profile
    print(f"Reference raster: {REFERENCE_RASTER.name}")
    ref_profile = build_reference_profile(REFERENCE_RASTER, SHAPEFILE)

    # Process each dataset group
    for config in DATASETS:
        input_dir = Path(config["input_dir"])
        output_dir = BASE_OUTPUT_DIR / config["output_folder"]
        output_dir.mkdir(exist_ok=True)

        method = (Resampling.nearest if config['type'] == 'categorical'
                  else Resampling.bilinear)

        print(f"\n--- {config['output_folder']} ---")

        tif_files = sorted(
            list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
        )

        if not tif_files:
            print(f"  WARNING: No TIF files found in {input_dir}")
            continue

        for fpath in tif_files:
            # For livestock directory, keep only target species files
            if "livestock" in str(input_dir).lower():
                if not any(k in fpath.name.upper() for k in LIVESTOCK_KEYS):
                    continue

            preprocess_raster(fpath, output_dir / fpath.name, ref_profile, method)

    print("\n=== All datasets aligned ===")


if __name__ == "__main__":
    main()
