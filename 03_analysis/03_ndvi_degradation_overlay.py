"""
03_ndvi_degradation_overlay.py

Generates the NDVI trend × degradation concordance map (Fig. 6 in the
manuscript). This overlay classifies each pixel into one of six categories
based on the combination of Mann-Kendall NDVI trend direction and the
integrated SDG 15.3.1 degradation status:

  1 = Greening & not degraded
  2 = Greening & degraded
  3 = Stable & not degraded
  4 = Stable & degraded
  5 = Browning & not degraded
  6 = Browning & degraded

Input:  - Study area reference raster (NDVI 2000, 250 m)
        - Composite degradation map (from Part 2)
Output: Six-class concordance GeoTIFF (degradation_ndvi_overlay.tif)
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ============================================================================
# Configuration — update paths to match your local directory structure
# ============================================================================
REFERENCE_RASTER = r"D:\Sahel\GEE_NDVI_250m\SAHEL_NDVI_250m_2000.tif"
DEGRADATION_RASTER = r"D:\Sahel\Results\degradation_250m.tif"
OUTPUT_RASTER = r"D:\Sahel\Results\degradation_ndvi_overlay.tif"


def main():
    # Read study area reference to define the valid-pixel domain
    with rasterio.open(REFERENCE_RASTER) as src_ref:
        ref_data = src_ref.read(1)
        ref_meta = src_ref.meta.copy()
        ref_nodata = src_ref.nodata

        # Initialize base layer: 0 = valid non-degraded, -9999 = outside ROI
        if ref_nodata is not None:
            base = np.where(ref_data != ref_nodata, 0, -9999).astype(np.int16)
        else:
            base = np.where(np.isfinite(ref_data), 0, -9999).astype(np.int16)

    # Read and align the degradation raster
    with rasterio.open(DEGRADATION_RASTER) as src_deg:
        deg_data = src_deg.read(1)
        deg_matched = np.full(base.shape, -9999, dtype=np.int16)

        reproject(
            source=deg_data, destination=deg_matched,
            src_transform=src_deg.transform, src_crs=src_deg.crs,
            src_nodata=src_deg.nodata,
            dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
            dst_nodata=-9999,
            resampling=Resampling.nearest
        )

    # Overlay: set degraded pixels to 1
    base[deg_matched == 1] = 1

    # Write output
    ref_meta.update(dtype="int16", count=1, nodata=-9999)
    with rasterio.open(OUTPUT_RASTER, "w", **ref_meta) as dst:
        dst.write(base, 1)

    print(f"Overlay map saved to: {OUTPUT_RASTER}")


if __name__ == "__main__":
    main()
