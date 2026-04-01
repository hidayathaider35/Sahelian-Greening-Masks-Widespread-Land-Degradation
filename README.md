# Sahelian Greening Masks Widespread Land Degradation

**Land-cover dynamics, degradation assessment, and driver attribution in the Sahelian Acacia Savanna (2000–2022)**

[![DOI](https://img.shields.io/badge/DOI-pending-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the complete data processing and analysis pipeline for the study of land-cover dynamics, land degradation patterns, and their driving factors across the Sahelian Acacia Savanna (SAS) ecoregion from 2000 to 2022. The analysis integrates:

- **Annual 30 m land-cover mapping** using the GLC_FCS30D dataset, harmonized to IPCC land categories.
- **Land degradation assessment** following the UNCCD SDG 15.3.1 framework (land-cover change, NDVI-based productivity dynamics, and modeled soil organic carbon change).
- **Driver attribution** combining GeoDetector factor/interaction analysis with XGBoost–SHAP interpretation across 22 climatic and anthropogenic variables.

## Repository Structure

```
Sahel-Land-Degradation/
│
├── 01_gee_scripts/                        # Google Earth Engine data acquisition
│   ├── 01_land_cover_export.js            # GLC_FCS30D reclassification and export
│   ├── 02_ndvi_250m_export.js             # MODIS MOD13Q1 annual NDVI composites
│   ├── 03_climate_terrain_export.js       # TerraClimate + SRTM annual stacks
│   ├── 04_burned_area_export.js           # MODIS MCD64A1 annual burned area
│   └── 05_soc_export.js                   # SoilGrids SOC baseline (0–5 cm)
│
├── 02_preprocessing/                      # Local preprocessing and alignment
│   ├── 01_extract_crop_urb.py             # Extract binary cropland/urban layers
│   └── 02_align_to_250m_grid.py           # Harmonize all datasets to 250 m grid
│
├── 03_analysis/                           # Core analysis pipeline
│   ├── 01_lulc_dynamics_and_degradation.py  # LULC change + SDG 15.3.1 sub-indicators
│   ├── 02_drivers_geodetector_xgboost.py    # GeoDetector + XGBoost–SHAP attribution
│   └── 03_ndvi_degradation_overlay.py       # NDVI trend × degradation concordance map
│
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
└── README.md                              # This file
```

## Study Area

The Sahelian Acacia Savanna ecoregion (~3.69 × 10⁶ km²), defined using the [RESOLVE Ecoregions 2017](https://ecoregions.appspot.com/) dataset. The region spans parts of Mauritania, Senegal, Mali, Burkina Faso, Niger, Nigeria, Chad, Sudan, Eritrea, and Ethiopia.

## Data Sources

| Dataset | Resolution | Period | Source |
|---------|-----------|--------|--------|
| GLC_FCS30D | 30 m | 2000–2022 | [Zhang et al. (2024)](https://doi.org/10.5194/essd-16-1353-2024) |
| MODIS NDVI (MOD13Q1) | 250 m | 2000–2022 | [LP DAAC](https://lpdaac.usgs.gov/) |
| TerraClimate | ~4 km | 2000–2022 | [Abatzoglou et al. (2018)](https://doi.org/10.1038/sdata.2017.191) |
| SRTM DEM | 30 m | Static | [Farr et al. (2007)](https://doi.org/10.1029/2005RG000183) |
| SoilGrids SOC | 250 m | Static | [Hengl et al. (2017)](https://doi.org/10.1371/journal.pone.0169748) |
| MODIS Burned Area (MCD64A1) | 500 m | 2000–2022 | [Giglio et al. (2018)](https://doi.org/10.1016/j.rse.2018.08.005) |
| GlobPOP | ~1 km | 2000–2022 | [Liu et al. (2024)](https://doi.org/10.1038/s41597-024-02913-0) |
| GLW3/GLW4 Livestock | ~10 km | 2010/2015 | [Gilbert et al. (2018)](https://doi.org/10.1038/sdata.2018.227) |
| Global Mining Areas | ~1 km | 2000–2017 | [Maus et al. (2020)](https://doi.pangaea.de/10.1594/PANGAEA.910894) |

## Workflow

The analysis follows a three-stage pipeline:

### Stage 1: Data Acquisition (Google Earth Engine)

Scripts in `01_gee_scripts/` export annual raster datasets clipped to the SAS ecoregion. These scripts are executed in the [GEE Code Editor](https://code.earthengine.google.com/). External datasets (GlobPOP, livestock densities, mining) are downloaded manually from their respective repositories.

### Stage 2: Preprocessing

Scripts in `02_preprocessing/` extract thematic layers from the land-cover product and harmonize all datasets to a common 250 m reference grid using the NDVI 2000 raster as the spatial template.

### Stage 3: Analysis

Scripts in `03_analysis/` implement the full analytical pipeline:

1. **LULC dynamics** — Annual area trajectories, Theil–Sen trend slopes, and transition matrices.
2. **SDG 15.3.1 degradation** — Three sub-indicators (land-cover change, productivity, SOC) integrated via the One-Out-All-Out principle.
3. **Driver attribution** — GeoDetector factor/interaction detection and XGBoost regression with SHAP interpretation at a 10 km aggregation scale.
4. **NDVI–degradation concordance** — Overlay of Mann-Kendall NDVI trends with the integrated degradation map.

## Requirements

- **Google Earth Engine** account (for Stage 1)
- **Python ≥ 3.9** with dependencies listed in `requirements.txt`
- **RAM**: ≥ 32 GB recommended for the 250 m pixel-level analyses
- **Storage**: ~50 GB for raw and processed raster datasets

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the GEE scripts in `01_gee_scripts/` to export raster datasets to Google Drive.
2. Download the exported files and external datasets to a local directory.
3. Update the file paths in `02_preprocessing/` and `03_analysis/` scripts to match your local directory structure.
4. Execute the preprocessing scripts in order:
   ```bash
   python 02_preprocessing/01_extract_crop_urb.py
   python 02_preprocessing/02_align_to_250m_grid.py
   ```
5. Execute the analysis scripts in order:
   ```bash
   python 03_analysis/01_lulc_dynamics_and_degradation.py
   python 03_analysis/02_drivers_geodetector_xgboost.py
   python 03_analysis/03_ndvi_degradation_overlay.py
   ```

> **Note**: All file paths are configured at the top of each script. Update them to match your local directory layout before running.

## Citation

If you use this code, please cite:

```
[Citation will be added upon publication]
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
