"""
01_lulc_dynamics_and_degradation.py

Part 1 of the analysis pipeline. This script performs:

  (A) Annual land-cover dynamics analysis using the 10-class system:
      - Transition matrix (2000 → 2022)
      - Annual area trajectories relative to 2000 baseline
      - Theil–Sen trend slopes (km² yr⁻¹)
      - Grassland gain/loss decomposition

  (B) SDG 15.3.1 degradation sub-indicator computation at 250 m:
      - Land-cover change sub-indicator (IPCC transition matrix, epochal)
      - Soil organic carbon sub-indicator (IPCC Tier 1, FLU-based)
      - Productivity sub-indicator (Mann-Kendall trend, state, performance)

Intermediate arrays are saved as .npy files for use in Part 2
(02_drivers_geodetector_xgboost.py).

Input:  Aligned 250 m rasters from preprocessing stage.
Output: CSV tables, GeoTIFF maps, and .npy intermediates.
"""

import os
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from scipy.stats import theilslopes
import pymannkendall as mk
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Configuration — update paths to match your local directory structure
# ============================================================================
RESULTS_DIR = Path(r"D:\Sahel\Results")
PROCESSED_250M = Path(r"D:\Sahel\Dataset_Processed_250m")
LC10_DIR = Path(r"D:\Sahel\10classes_GLC")
SHAPEFILE = Path(r"D:\Sahel\roi\Sahel_ROI_Download.shp")

IPCC_DIR = PROCESSED_250M / "ipcc_GLC"
TERA_DIR = PROCESSED_250M / "climate_terrain"
NDVI_DIR = PROCESSED_250M / "ndvi"
SOC_DIR = PROCESSED_250M / "soc"

INTERMEDIATE_DIR = RESULTS_DIR / "intermediate_files"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

YEARS = np.arange(2000, 2023)

# 10-class land-cover names (excluding Tundra=5 and Ice/Snow=10 for the Sahel)
CLASS_NAMES_10 = {
    1: "Cropland", 2: "Forest", 3: "Shrubland", 4: "Grassland",
    6: "Wetland", 7: "Impervious", 8: "Bare land", 9: "Water"
}
NOISE_CLASSES = [5, 10]

# ============================================================================
# IPCC Land-Cover Degradation Transition Matrix
# ============================================================================
# Rows/columns ordered as: Cropland(1), Forest(2), Grassland(3),
# Wetlands(4), Settlements(5), Other land(6).
# +1 = degradation, -1 = improvement, 0 = stable.
LC_MATRIX = np.array([
    [ 0, -1, -1, -1, +1, +1],  # from Cropland
    [+1,  0, +1, +1, +1, +1],  # from Forest
    [-1, -1,  0, -1, +1, +1],  # from Grassland
    [+1, +1, +1,  0, +1, +1],  # from Wetlands
    [-1, -1, -1, -1,  0, -1],  # from Settlements
    [-1, -1, -1, -1, +1,  0],  # from Other land
], dtype=np.int8)

# IPCC Tier 1 stock change factors (FLU) for SOC under tropical dry climate.
# Rows = 'from' class, columns = 'to' class (same order as LC_MATRIX).
SOC_TRANSITION_FACTORS = np.array([
    [1.00, 1.10, 1.02, 1.08, 0.92, 0.98],
    [0.85, 1.00, 0.92, 0.95, 0.80, 0.90],
    [0.90, 1.06, 1.00, 1.05, 0.85, 0.95],
    [0.80, 0.95, 0.90, 1.00, 0.75, 0.90],
    [0.95, 0.95, 0.95, 0.95, 1.00, 0.98],
    [0.98, 1.05, 1.03, 1.08, 0.90, 1.00],
], dtype=np.float32)


# ============================================================================
# File Resolvers
# ============================================================================

def find_ndvi_path(year: int) -> Path:
    for pattern in [f"SAHEL_NDVI_250m_{year}.tif", f"SAHEL_NDVI_{year}.tif"]:
        p = NDVI_DIR / pattern
        if p.exists():
            return p
    raise FileNotFoundError(f"NDVI file for {year} not found in {NDVI_DIR}")


def find_tera_path(year: int) -> Path:
    p = TERA_DIR / f"SAHEL_{year}.tif"
    if not p.exists():
        raise FileNotFoundError(f"Climate stack not found: {p}")
    return p


def find_lc10_path(year: int) -> Path:
    p = LC10_DIR / f"GLC_FCS30D_{year}_10classes.tif"
    if not p.exists():
        raise FileNotFoundError(f"10-class LULC not found: {p}")
    return p


def find_ipcc_path(year: int) -> Path:
    for pattern in [f"GLC_FCS30D_FCS30D_{year}_ipcc.tif",
                    f"GLC_FCS30D_{year}_ipcc.tif"]:
        p = IPCC_DIR / pattern
        if p.exists():
            return p
    raise FileNotFoundError(f"IPCC LULC not found for {year} in {IPCC_DIR}")


def find_soc_path() -> Path:
    for name in ["SOC_0_5cm_250m_SAH.tif", "SOC_0_5cm_250m_Sahel.tif"]:
        p = SOC_DIR / name
        if p.exists():
            return p
    tifs = sorted(list(SOC_DIR.glob("*.tif")) + list(SOC_DIR.glob("*.tiff")))
    if tifs:
        return tifs[0]
    raise FileNotFoundError(f"No SOC raster found in {SOC_DIR}")


# ============================================================================
# Utility Functions
# ============================================================================

def read_aligned(src_path: Path, ref_meta: dict, roi_mask: np.ndarray,
                 resampling=Resampling.nearest, band: int = 1) -> np.ndarray:
    """Read a raster band, reproject to reference grid if needed, apply mask."""
    with rasterio.open(src_path) as src:
        if (src.crs == ref_meta['crs'] and
                src.transform == ref_meta['transform'] and
                src.width == ref_meta['width'] and
                src.height == ref_meta['height']):
            arr = src.read(band).astype(np.float32)
        else:
            arr = np.full((ref_meta['height'], ref_meta['width']),
                          np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, band), destination=arr,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
                resampling=resampling,
                src_nodata=src.nodata, dst_nodata=np.nan
            )
    arr[roi_mask] = np.nan
    return arr


def pixel_area_km2(ref_meta: dict, gdf: gpd.GeoDataFrame) -> float:
    """Approximate pixel area in km² (handles geographic CRS)."""
    crs = ref_meta.get("crs")
    t = ref_meta.get("transform")
    if crs is not None and hasattr(crs, "is_projected") and crs.is_projected:
        return abs(t.a * t.e) / 1e6
    gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")
    try:
        lat = gdf_wgs.union_all().centroid.y
    except AttributeError:
        lat = gdf_wgs.unary_union.centroid.y
    km_lat = 111.32
    km_lon = 111.32 * np.cos(np.deg2rad(lat))
    return abs(t.a) * km_lon * abs(t.e) * km_lat


def lc_degradation(lc_prev: np.ndarray, lc_curr: np.ndarray) -> np.ndarray:
    """Flag pixels as degraded (1) based on IPCC transition matrix."""
    out = np.full(lc_prev.shape, np.nan, dtype=np.float32)
    valid = ~np.isnan(lc_prev) & ~np.isnan(lc_curr)
    prev = lc_prev[valid].astype(np.int16)
    curr = lc_curr[valid].astype(np.int16)
    ok = (prev >= 1) & (prev <= 6) & (curr >= 1) & (curr <= 6)
    res = np.zeros(prev.shape[0], dtype=np.uint8)
    res[ok] = (LC_MATRIX[prev[ok] - 1, curr[ok] - 1] == 1).astype(np.uint8)
    out[valid] = res.astype(np.float32)
    return out


def soc_change(soc_ref: np.ndarray, lc_from: np.ndarray,
               lc_to: np.ndarray) -> tuple:
    """Compute SOC change using IPCC Tier 1 stock change factors."""
    soc_est = np.full(soc_ref.shape, np.nan, dtype=np.float32)
    rel_change = np.full(soc_ref.shape, np.nan, dtype=np.float32)
    valid = ~np.isnan(soc_ref) & ~np.isnan(lc_from) & ~np.isnan(lc_to)
    f = lc_from[valid].astype(np.int16)
    t = lc_to[valid].astype(np.int16)
    ok = (f >= 1) & (f <= 6) & (t >= 1) & (t <= 6)
    factors = np.ones(f.shape[0], dtype=np.float32)
    factors[ok] = SOC_TRANSITION_FACTORS[f[ok] - 1, t[ok] - 1]
    estimated = soc_ref[valid] * factors
    soc_est[valid] = estimated
    rel_change[valid] = (estimated - soc_ref[valid]) / (soc_ref[valid] + 1e-9)
    return soc_est, rel_change


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Build 250 m reference grid and ROI mask
    # ------------------------------------------------------------------
    print("Building 250 m reference grid...")
    gdf = gpd.read_file(SHAPEFILE)
    ref_path = find_ndvi_path(2000)

    with rasterio.open(ref_path) as ref_src:
        ref_meta = ref_src.meta.copy()
        h, w = ref_src.height, ref_src.width
        transform = ref_src.transform
        gdf_proj = gdf.to_crs(ref_src.crs) if gdf.crs != ref_src.crs else gdf
        masked_data, _ = mask(ref_src, gdf_proj.geometry, crop=False)
        nodata = ref_src.nodata
        roi_mask = (np.isnan(masked_data[0]) |
                    (masked_data[0] == nodata if nodata is not None else False))

    px_km2 = pixel_area_km2(ref_meta, gdf_proj)
    print(f"Pixel area: {px_km2:.6f} km²")

    out_meta_u8 = ref_meta.copy()
    out_meta_u8.update(dtype='uint8', count=1, nodata=0, compress="lzw")

    # ------------------------------------------------------------------
    # (A) 10-class LULC dynamics
    # ------------------------------------------------------------------
    print("\n=== LULC Dynamics (10-class) ===")

    # Annual area time series
    area_data = []
    for y in tqdm(YEARS, desc="Annual area"):
        lc = read_aligned(find_lc10_path(int(y)), ref_meta, roi_mask)
        lc[np.isin(lc, NOISE_CLASSES)] = np.nan
        row = {'year': int(y)}
        for code, name in CLASS_NAMES_10.items():
            row[name] = float(np.nansum(lc == code) * px_km2)
        area_data.append(row)

    area_df = pd.DataFrame(area_data)

    # Change relative to 2000
    base = area_df[area_df['year'] == 2000].drop(columns='year').iloc[0]
    change_df = area_df.copy()
    for name in CLASS_NAMES_10.values():
        change_df[name] = change_df[name] - base[name]
    change_df.to_csv(RESULTS_DIR / "lulc_change_vs_2000_km2.csv", index=False)

    # Theil–Sen slopes
    slopes = []
    for name in CLASS_NAMES_10.values():
        s = theilslopes(area_df[name], area_df['year'])[0]
        slopes.append({'class': name, 'slope_km2_per_yr': float(s)})
    pd.DataFrame(slopes).to_csv(RESULTS_DIR / "lulc_theilsen_slopes.csv",
                                index=False)

    # Transition matrix (2000 → 2022)
    lc2000 = read_aligned(find_lc10_path(2000), ref_meta, roi_mask)
    lc2022 = read_aligned(find_lc10_path(2022), ref_meta, roi_mask)
    lc2000[np.isin(lc2000, NOISE_CLASSES)] = np.nan
    lc2022[np.isin(lc2022, NOISE_CLASSES)] = np.nan

    valid = ~np.isnan(lc2000) & ~np.isnan(lc2022)
    codes = (lc2000[valid].astype(int) * 100 +
             lc2022[valid].astype(int)).astype(np.int32)
    uniq, cnts = np.unique(codes, return_counts=True)
    trans_df = pd.DataFrame({
        'from_code': uniq // 100, 'to_code': uniq % 100, 'pixels': cnts
    })
    trans_df['area_km2'] = trans_df['pixels'] * px_km2
    trans_df['from_class'] = trans_df['from_code'].map(CLASS_NAMES_10)
    trans_df['to_class'] = trans_df['to_code'].map(CLASS_NAMES_10)
    trans_df = trans_df[trans_df['from_code'].isin(CLASS_NAMES_10) &
                        trans_df['to_code'].isin(CLASS_NAMES_10)]
    trans_df.to_csv(RESULTS_DIR / "transition_matrix_2000_2022.csv",
                    index=False)

    # Save 2022 land-cover map
    with rasterio.open(RESULTS_DIR / "lulc_2022_10class.tif",
                       'w', **out_meta_u8) as dst:
        dst.write(np.nan_to_num(lc2022, nan=0).astype(np.uint8), 1)

    print("LULC dynamics outputs saved.")

    # ------------------------------------------------------------------
    # (B) SDG 15.3.1 Sub-indicators
    # ------------------------------------------------------------------
    print("\n=== SDG 15.3.1 Sub-indicators ===")

    # --- Land-cover sub-indicator (epochal transitions) ---
    print("Computing land-cover sub-indicator...")
    ipcc2000 = read_aligned(find_ipcc_path(2000), ref_meta, roi_mask)
    ipcc2015 = read_aligned(find_ipcc_path(2015), ref_meta, roi_mask)
    ipcc2022 = read_aligned(find_ipcc_path(2022), ref_meta, roi_mask)

    deg_00_15 = lc_degradation(ipcc2000, ipcc2015)
    deg_15_22 = lc_degradation(ipcc2015, ipcc2022)
    lc_sub = np.where(
        np.isnan(deg_00_15) | np.isnan(deg_15_22), np.nan,
        ((deg_00_15 == 1) | (deg_15_22 == 1)).astype(np.float32)
    ).astype(np.float32)

    # --- SOC sub-indicator (Tier 1) ---
    print("Computing SOC sub-indicator...")
    soc_ref = read_aligned(find_soc_path(), ref_meta, roi_mask,
                           Resampling.bilinear)
    _, rel = soc_change(soc_ref, ipcc2000, ipcc2022)
    soc_sub = np.where(rel < -0.10, 1, 0).astype(np.float32)
    soc_sub[np.isnan(rel)] = np.nan

    # --- Productivity sub-indicator (NDVI trend, state, performance) ---
    print("Computing productivity sub-indicator...")

    # Load full NDVI time series
    ndvi_ts = np.zeros((len(YEARS), h, w), dtype=np.float32)
    for i, y in enumerate(tqdm(YEARS, desc="Loading NDVI")):
        with rasterio.open(find_ndvi_path(int(y))) as src:
            ndvi_ts[i] = src.read(1).astype(np.float32)
    ndvi_ts[:, roi_mask] = np.nan

    # State metric: z-score of recent mean vs baseline
    base_mean = np.nanmean(ndvi_ts[0:16], axis=0)   # 2000–2015
    base_sd = np.nanstd(ndvi_ts[0:16], axis=0)
    recent_mean = np.nanmean(ndvi_ts[-3:], axis=0)   # 2020–2022
    z_state = (recent_mean - base_mean) / (base_sd / np.sqrt(3) + 1e-10)
    state_deg = (z_state < -1.96).astype(np.float32)
    state_deg[np.isnan(base_mean)] = np.nan

    # Trend metric: Mann-Kendall test + Theil-Sen slope
    print("Computing pixel-level NDVI trends...")
    n_px = h * w
    ndvi_flat = ndvi_ts.reshape(len(YEARS), n_px)
    has_data = ~np.all(np.isnan(ndvi_flat), axis=0)
    ndvi_valid = ndvi_flat[:, has_data]

    trend_flat = np.full(n_px, np.nan, dtype=np.float32)
    trend_vals = np.zeros(ndvi_valid.shape[1], dtype=np.float32)

    batch = 10000
    for start in tqdm(range(0, ndvi_valid.shape[1], batch),
                      desc="Trend analysis"):
        if start % (batch * 100) == 0:
            gc.collect()
        end = min(start + batch, ndvi_valid.shape[1])
        for j in range(start, end):
            try:
                ts = ndvi_valid[:, j]
                ok = ~np.isnan(ts)
                if ok.sum() < 5 or np.unique(ts[ok]).size < 2:
                    continue
                slope = theilslopes(ts[ok], YEARS[ok])[0]
                mk_res = mk.original_test(ts[ok])
                if mk_res.p < 0.05 and slope < 0:
                    trend_vals[j] = 1
            except Exception:
                continue

    trend_flat[has_data] = trend_vals
    trend_deg = trend_flat.reshape(h, w)

    # Performance metric: observed NDVI / P90 within ecological units
    print("Computing performance metric...")
    sum_pr = np.zeros((h, w), dtype=np.float64)
    sum_tm = np.zeros((h, w), dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.uint16)

    for y in tqdm(YEARS[0:16], desc="Baseline climate"):
        with rasterio.open(find_tera_path(int(y))) as src:
            pr = src.read(6).astype(np.float32)
            tmn = src.read(11).astype(np.float32)
            tmx = src.read(12).astype(np.float32)
        pr[roi_mask] = np.nan
        tmn[roi_mask] = np.nan
        tmx[roi_mask] = np.nan
        tmean = (tmn + tmx) / 2.0
        ok = ~np.isnan(pr) & ~np.isnan(tmean)
        sum_pr[ok] += pr[ok]
        sum_tm[ok] += tmean[ok]
        cnt[ok] += 1

    mean_pr = (sum_pr / np.maximum(cnt, 1)).astype(np.float32)
    mean_tm = (sum_tm / np.maximum(cnt, 1)).astype(np.float32)
    mean_pr[cnt == 0] = np.nan
    mean_tm[cnt == 0] = np.nan

    perf_mask = (~np.isnan(base_mean) & ~np.isnan(mean_pr) &
                 ~np.isnan(mean_tm) & ~np.isnan(soc_ref) &
                 ~np.isnan(ipcc2000))

    perf_deg = np.full((h, w), np.nan, dtype=np.float32)
    if perf_mask.any():
        df = pd.DataFrame({
            'ndvi': base_mean[perf_mask], 'pr': mean_pr[perf_mask],
            'temp': mean_tm[perf_mask], 'soil': soc_ref[perf_mask],
            'lc': ipcc2000[perf_mask]
        })
        for col in ['pr', 'temp', 'soil']:
            if df[col].nunique() > 1:
                df[f'{col}_bin'] = pd.qcut(df[col], 5, labels=False,
                                           duplicates='drop')
            else:
                df[f'{col}_bin'] = 0

        groups = df.groupby(['pr_bin', 'temp_bin', 'soil_bin', 'lc'],
                            observed=False)
        df['p90'] = groups['ndvi'].transform(
            lambda g: g.quantile(0.9) if len(g) > 1 else np.nan
        )
        df['perf_deg'] = ((df['ndvi'] / (df['p90'] + 1e-10)) < 0.5).astype(int)
        perf_deg[perf_mask] = df['perf_deg'].values

    # Combined productivity flag (UNCCD lookup table logic)
    state_u = np.nan_to_num(state_deg, nan=0).astype(np.uint8)
    perf_u = np.nan_to_num(perf_deg, nan=0).astype(np.uint8)
    trend_u = np.nan_to_num(trend_deg, nan=0).astype(np.uint8)
    productivity = trend_u | (state_u & perf_u)

    # ------------------------------------------------------------------
    # Save intermediates for Part 2
    # ------------------------------------------------------------------
    print("\nSaving intermediates...")
    np.save(INTERMEDIATE_DIR / 'land_cover_sub.npy', lc_sub)
    np.save(INTERMEDIATE_DIR / 'soc_sub.npy', soc_sub)
    np.save(INTERMEDIATE_DIR / 'productivity_y.npy', productivity)
    np.save(INTERMEDIATE_DIR / 'baseline_mean.npy', base_mean)
    np.save(INTERMEDIATE_DIR / 'lc2000.npy', lc2000)
    np.save(INTERMEDIATE_DIR / 'lc2022.npy', lc2022)
    np.save(INTERMEDIATE_DIR / 'ref_meta_250.npy',
            np.array([ref_meta, h, w, transform, ref_meta['crs']],
                     dtype=object))

    print(f"Intermediates saved to {INTERMEDIATE_DIR}")
    print("Part 1 complete. Run Part 2 (02_drivers_geodetector_xgboost.py).")


if __name__ == "__main__":
    main()
