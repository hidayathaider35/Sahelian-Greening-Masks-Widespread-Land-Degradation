/**
 * 05_soc_export.js
 *
 * Exports the SoilGrids soil organic carbon (SOC) mean estimate for the
 * 0–5 cm depth layer at 250 m resolution, clipped to the Sahelian Acacia
 * Savanna ecoregion. This static layer serves as the SOC reference stock
 * (SOC_Ref) for the IPCC Tier 1 SOC change assessment.
 *
 * Output: One GeoTIFF (SOC_0_5cm_250m_Sahel.tif).
 *         Saved to Google Drive folder 'GEE_Exports'.
 */

// ---------------------------------------------------------------------------
// 1. Define Region of Interest
// ---------------------------------------------------------------------------
var ecoRegions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017');
var roi = ecoRegions
  .filter(ee.Filter.stringContains('ECO_NAME', 'Sahel'))
  .geometry();
Map.centerObject(roi, 5);

// ---------------------------------------------------------------------------
// 2. Load and Clip SOC Data
// ---------------------------------------------------------------------------
var soc = ee.Image("projects/soilgrids-isric/soc_mean");
var soc_0_5cm = soc.select('soc_0-5cm_mean').clip(roi);

// ---------------------------------------------------------------------------
// 3. Export to Google Drive
// ---------------------------------------------------------------------------
Export.image.toDrive({
  image: soc_0_5cm,
  description: 'SOC_0_5cm_250m_Sahel',
  folder: 'GEE_Exports',
  region: roi.geometry().bounds(),
  scale: 250,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
