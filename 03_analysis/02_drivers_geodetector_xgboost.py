"""
02_drivers_geodetector_xgboost.py

Part 2 of the analysis pipeline. This script performs:

  (A) Composite degradation map and per-class statistics (SDG 15.3.1,
      One-Out-All-Out integration at 250 m).
  (B) GeoDetector factor and interaction detection at 10 km resolution.
  (C) XGBoost regression with SHAP interpretation for non-linear
      driver attribution.

Requires intermediate .npy files produced by Part 1
(01_lulc_dynamics_and_degradation.py).

Input:  Intermediate arrays + aligned 250 m driver datasets.
Output: Degradation maps/statistics, GeoDetector q-values and interaction
        matrix, XGBoost feature importance, SHAP summary plots.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import Resampling
import geopandas as gpd
from scipy.stats import f as fdist
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Configuration — update paths to match your local directory structure
# ============================================================================
RESULTS_DIR = Path(r"D:\Sahel\Results")
PROCESSED_250M = Path(r"D:\Sahel\Dataset_Processed_250m")
SHAPEFILE = Path(r"D:\Sahel\roi\Sahel_ROI_Download.shp")
INTERMEDIATE_DIR = RESULTS_DIR / "intermediate_files"

CROP_URB_DIR = PROCESSED_250M / "crop_urb"
POP_DIR = PROCESSED_250M / "population"
MINING_DIR = PROCESSED_250M / "mining"
LIVESTOCK_DIR = PROCESSED_250M / "livestock"
FIRE_DIR = PROCESSED_250M / "fire"
TERA_DIR = PROCESSED_250M / "climate_terrain"
NDVI_DIR = PROCESSED_250M / "ndvi"

YEARS = np.arange(2000, 2023)

CLASS_NAMES_10 = {
    1: "Cropland", 2: "Forest", 3: "Shrubland", 4: "Grassland",
    6: "Wetland", 7: "Impervious", 8: "Bare land", 9: "Water"
}

# Aggregation factor: 250 m pixels → 10 km grid cells
AGG_FACTOR = 40


# ============================================================================
# File Resolvers
# ============================================================================

def find_ndvi_path(year: int) -> Path:
    for pat in [f"SAHEL_NDVI_250m_{year}.tif", f"SAHEL_NDVI_{year}.tif"]:
        p = NDVI_DIR / pat
        if p.exists():
            return p
    raise FileNotFoundError(f"NDVI not found for {year}")


def find_tera_path(year: int) -> Path:
    p = TERA_DIR / f"SAHEL_{year}.tif"
    if not p.exists():
        raise FileNotFoundError(f"Climate stack not found: {p}")
    return p


# ============================================================================
# Utility Functions
# ============================================================================

def aggregate(arr: np.ndarray, factor: int, method: str = 'mean') -> np.ndarray:
    """Aggregate a 2D array by a block factor using mean or sum."""
    nh = arr.shape[0] // factor
    nw = arr.shape[1] // factor
    cropped = arr[:nh * factor, :nw * factor]
    reshaped = cropped.reshape(nh, factor, nw, factor)
    if method == 'mean':
        return np.nanmean(reshaped, axis=(1, 3))
    elif method == 'sum':
        return np.nansum(reshaped, axis=(1, 3))
    raise ValueError("method must be 'mean' or 'sum'")


def geodetector_q(y_vec: np.ndarray, x_vec: np.ndarray,
                  n_strata: int = 5) -> tuple:
    """Compute GeoDetector q-statistic and p-value."""
    df = pd.DataFrame({'y': y_vec, 'x': x_vec}).dropna()
    if len(df) < 2 or df['y'].var() == 0 or df['x'].nunique() < 2:
        return 0.0, 1.0

    total_var = df['y'].var()
    n = len(df)

    try:
        df['strata'] = pd.qcut(df['x'], n_strata, labels=False,
                                duplicates='drop')
        k = df['strata'].nunique()
        if k < 2:
            return 0.0, 1.0
        within_var = df.groupby('strata', observed=False)['y'].apply(
            lambda g: g.var() * len(g)
        ).sum()
        q = float(np.clip(1 - within_var / (n * total_var), 0, 1))
        if np.isnan(q) or (1 - q) == 0:
            return 0.0, 1.0
        f_stat = q / (1 - q) * (n - k) / (k - 1)
        p_val = fdist.sf(f_stat, k - 1, n - k)
        return q, p_val
    except (ValueError, IndexError):
        return 0.0, 1.0


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Load intermediates from Part 1
    # ------------------------------------------------------------------
    print("Loading intermediates from Part 1...")
    lc_sub = np.load(INTERMEDIATE_DIR / 'land_cover_sub.npy')
    soc_sub = np.load(INTERMEDIATE_DIR / 'soc_sub.npy')
    prod_sub = np.load(INTERMEDIATE_DIR / 'productivity_y.npy')
    base_mean = np.load(INTERMEDIATE_DIR / 'baseline_mean.npy')
    lc2000 = np.load(INTERMEDIATE_DIR / 'lc2000.npy')
    lc2022 = np.load(INTERMEDIATE_DIR / 'lc2022.npy')
    ref_meta, h, w, transform, crs = np.load(
        INTERMEDIATE_DIR / 'ref_meta_250.npy', allow_pickle=True
    )

    # Rebuild ROI mask
    gdf = gpd.read_file(SHAPEFILE)
    with rasterio.open(find_ndvi_path(2000)) as ref:
        gdf_proj = gdf.to_crs(ref.crs) if gdf.crs != ref.crs else gdf
        masked, _ = mask(ref, gdf_proj.geometry, crop=False)
        nodata = ref.nodata
        roi_mask = (np.isnan(masked[0]) |
                    (masked[0] == nodata if nodata is not None else False))

    def load_masked(path: Path, band: int = 1) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read(band).astype(np.float32)
        arr[roi_mask] = np.nan
        return arr

    out_u8 = ref_meta.copy()
    out_u8.update(dtype='uint8', nodata=0, count=1, compress="lzw")

    # ------------------------------------------------------------------
    # (A) Composite degradation map (One-Out-All-Out)
    # ------------------------------------------------------------------
    print("\n=== Composite Degradation Map ===")
    degradation = ((lc_sub == 1) | (prod_sub == 1) | (soc_sub == 1)
                   ).astype(np.uint8)
    degradation[np.isnan(base_mean)] = 0

    with rasterio.open(RESULTS_DIR / "degradation_250m.tif",
                       'w', **out_u8) as dst:
        dst.write(degradation, 1)

    # Degradation intensity at 1 km (4×4 blocks)
    f4 = 4
    nh1 = int(h) // f4
    nw1 = int(w) // f4
    deg_1km = np.zeros((nh1, nw1), dtype=np.float32)
    for i in range(nh1):
        for j in range(nw1):
            blk = degradation[i * f4:(i + 1) * f4, j * f4:(j + 1) * f4]
            if blk.size > 0:
                deg_1km[i, j] = np.sum(blk == 1) / blk.size * 100.0

    # Per-class degradation statistics
    mask_deg = degradation == 1
    vals = lc2022[mask_deg]
    vals = vals[~np.isnan(vals)]
    total = vals.size
    rows = []
    for code, name in CLASS_NAMES_10.items():
        share = float(np.sum(vals == code) / total * 100) if total > 0 else 0
        rows.append({'class': name, 'share_of_degraded_%': share})
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "degraded_area_by_class.csv", index=False)

    # Within-class degradation rates
    rates = []
    for code, name in CLASS_NAMES_10.items():
        cmask = lc2022 == code
        n_total = np.nansum(cmask)
        n_deg = np.nansum(degradation[cmask] == 1) if n_total > 0 else 0
        rates.append({'class': name,
                      'within_class_degradation_%': float(n_deg / max(n_total, 1) * 100)})
    pd.DataFrame(rates).to_csv(
        RESULTS_DIR / "within_class_degradation_rate.csv", index=False)

    print("Degradation outputs saved.")

    # ------------------------------------------------------------------
    # (B) GeoDetector Analysis at 10 km
    # ------------------------------------------------------------------
    print("\n=== GeoDetector Analysis ===")

    # Response variable: degradation intensity aggregated to 10 km
    y_10km = aggregate(deg_1km, factor=10, method='mean')
    y = y_10km.flatten()

    # Load 2022 driver layers
    with rasterio.open(find_tera_path(2022)) as src:
        aet = src.read(2).astype(np.float32)
        def_ = src.read(3).astype(np.float32)
        pdsi = src.read(4).astype(np.float32)
        pet = src.read(5).astype(np.float32)
        pr = src.read(6).astype(np.float32)
        ro = src.read(7).astype(np.float32)
        soil = src.read(9).astype(np.float32)
        srad = src.read(10).astype(np.float32)
        tmn = src.read(11).astype(np.float32)
        tmx = src.read(12).astype(np.float32)
        vap = src.read(13).astype(np.float32)
        vpd = src.read(14).astype(np.float32)
        vs = src.read(15).astype(np.float32)
        elev = src.read(16).astype(np.float32)
        slope = src.read(17).astype(np.float32)

    for arr in [aet, def_, pdsi, pet, pr, ro, soil, srad,
                tmn, tmx, vap, vpd, vs, elev, slope]:
        arr[roi_mask] = np.nan

    urb = load_masked(CROP_URB_DIR / "urb_2022.tif")
    crop = load_masked(CROP_URB_DIR / "crop_2022.tif")
    pop = load_masked(POP_DIR / "GlobPOP_Density_30arc_2022_F32.tiff")
    fire = load_masked(FIRE_DIR / "SAHEL_BURN_500m_2022.tif")

    # Livestock: average of GLW3 (2010) and GLW4 (2015)
    cattle = (load_masked(LIVESTOCK_DIR / "GLW3.D-DA.CTL.tif") +
              load_masked(LIVESTOCK_DIR / "5_Ct_2015_Da.tif")) / 2.0
    goats = (load_masked(LIVESTOCK_DIR / "GLW3.D-DA.GTS.tif") +
             load_masked(LIVESTOCK_DIR / "5_Gt_2015_Da.tif")) / 2.0
    sheep = (load_masked(LIVESTOCK_DIR / "GLW3.D-DA.SHP.tif") +
             load_masked(LIVESTOCK_DIR / "5_Sh_2015_Da.tif")) / 2.0

    # Aggregate all drivers to 10 km
    F = AGG_FACTOR
    drivers = {
        'crop':  aggregate(crop, F, 'sum'),
        'urb':   aggregate(np.log1p(urb), F, 'mean'),
        'pop':   aggregate(pop, F, 'sum'),
        'fire':  aggregate(fire, F, 'mean'),
        'ct':    aggregate(cattle, F, 'mean'),
        'gt':    aggregate(goats, F, 'mean'),
        'sh':    aggregate(sheep, F, 'mean'),
        'aet':   aggregate(aet, F, 'mean'),
        'def':   aggregate(def_, F, 'mean'),
        'pdsi':  aggregate(pdsi, F, 'mean'),
        'pet':   aggregate(pet, F, 'mean'),
        'pr':    aggregate(pr, F, 'mean'),
        'ro':    aggregate(ro, F, 'mean'),
        'soil':  aggregate(soil, F, 'mean'),
        'srad':  aggregate(srad, F, 'mean'),
        'tmn':   aggregate(tmn, F, 'mean'),
        'tmx':   aggregate(tmx, F, 'mean'),
        'vap':   aggregate(vap, F, 'mean'),
        'vpd':   aggregate(vpd, F, 'mean'),
        'vs':    aggregate(vs, F, 'mean'),
        'ele':   aggregate(elev, F, 'mean'),
        'slp':   aggregate(slope, F, 'mean'),
    }

    # Sparse factors: set zero/negative to NaN
    for key in ['crop', 'urb', 'pop', 'fire', 'ct', 'gt', 'sh']:
        drivers[key][drivers[key] <= 0] = np.nan

    # Factor detection (individual q-values)
    q_results = []
    for name, grid in tqdm(drivers.items(), desc="Factor detection"):
        q, p = geodetector_q(y, grid.flatten())
        q_results.append({'factor': name, 'q': q, 'p': p})
    df_q = pd.DataFrame(q_results).sort_values('q', ascending=False)
    df_q.to_csv(RESULTS_DIR / "geodetector_q_values.csv", index=False)

    # Interaction detection
    q_dict = dict(zip(df_q['factor'], df_q['q']))
    factor_names = sorted(drivers.keys())
    interactions = []

    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            n1, n2 = factor_names[i], factor_names[j]
            x1 = drivers[n1].flatten()
            x2 = drivers[n2].flatten()
            n_min = min(len(y), len(x1), len(x2))
            df_int = pd.DataFrame(
                {'y': y[:n_min], 'x1': x1[:n_min], 'x2': x2[:n_min]}
            ).dropna()
            if len(df_int) < 2:
                continue
            try:
                s1 = pd.qcut(df_int['x1'], 5, labels=False, duplicates='drop')
                s2 = pd.qcut(df_int['x2'], 5, labels=False, duplicates='drop')
                combined = s1.astype(str) + '_' + s2.astype(str)
                df_tmp = pd.DataFrame({'y': df_int['y'].values,
                                       'strata': combined.values}).dropna()
                tv = df_tmp['y'].var()
                n = len(df_tmp)
                k = df_tmp['strata'].nunique()
                if k < 2 or tv == 0:
                    continue
                wv = df_tmp.groupby('strata')['y'].apply(
                    lambda g: g.var() * len(g)
                ).sum()
                qi = float(np.clip(1 - wv / (n * tv), 0, 1))
                fs = qi / (1 - qi + 1e-12) * (n - k) / (k - 1)
                pi = fdist.sf(fs, k - 1, n - k)
                for a, b in [(n1, n2), (n2, n1)]:
                    interactions.append({
                        'factor1': a, 'factor2': b,
                        'q_interaction': qi, 'p': pi,
                        'q1': q_dict[a], 'q2': q_dict[b]
                    })
            except (ValueError, IndexError):
                continue

    pd.DataFrame(interactions).to_csv(
        RESULTS_DIR / "geodetector_interactions.csv", index=False)
    print("GeoDetector outputs saved.")

    # ------------------------------------------------------------------
    # (C) XGBoost + SHAP
    # ------------------------------------------------------------------
    print("\n=== XGBoost–SHAP Analysis ===")

    valid = ~np.isnan(y)
    X = np.column_stack([drivers[f].flatten()[valid] for f in drivers])
    y_ml = y[valid]

    X[np.isinf(X)] = np.nan
    y_ml[np.isinf(y_ml)] = np.nan
    X = SimpleImputer(strategy='mean').fit_transform(X)
    row_ok = ~np.isnan(y_ml)
    X, y_ml = X[row_ok], y_ml[row_ok]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ml, test_size=0.2, random_state=42
    )

    # Train XGBoost (GPU if available, fallback to CPU)
    try:
        model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='gpu_hist', random_state=42
        )
        model.fit(X_train, y_train)
    except Exception:
        model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', random_state=42
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"XGBoost R² = {r2:.3f}")

    fnames = list(drivers.keys())
    pd.DataFrame({'factor': fnames, 'importance': model.feature_importances_}
                 ).sort_values('importance', ascending=False).to_csv(
        RESULTS_DIR / "xgboost_feature_importance.csv", index=False)

    # SHAP values
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        mean_shap = np.mean(np.abs(shap_values), axis=0)
        pd.DataFrame({'factor': fnames, 'mean_abs_shap': mean_shap}).to_csv(
            RESULTS_DIR / "shap_importance.csv", index=False)

        shap.summary_plot(shap_values, X_test, feature_names=fnames,
                          max_display=len(fnames), show=False)
        plt.savefig(RESULTS_DIR / "shap_beeswarm.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("SHAP outputs saved.")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

    print("\nPart 2 complete.")


if __name__ == "__main__":
    main()
