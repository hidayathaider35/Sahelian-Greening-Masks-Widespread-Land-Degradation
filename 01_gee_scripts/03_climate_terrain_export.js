/**
 * 03_climate_terrain_export.js
 *
 * Exports annual multi-band climate and terrain stacks at 1 km resolution
 * for the Sahelian Acacia Savanna ecoregion.
 *
 * Climate variables are derived from TerraClimate. Terrain variables
 * (elevation, slope) and baseline soil organic carbon are included as
 * static bands in each annual stack for convenience.
 *
 * Band order per exported image:
 *   1  = NDVI (annual mean, MODIS, scaled)
 *   2  = AET  (actual evapotranspiration, mm, annual sum)
 *   3  = DEF  (climatic water deficit, mm, annual sum)
 *   4  = PDSI (Palmer Drought Severity Index, annual mean, ×0.01)
 *   5  = PET  (potential evapotranspiration, mm, annual sum)
 *   6  = PR   (precipitation, mm, annual sum)
 *   7  = RO   (runoff, mm, annual sum)
 *   8  = SWE  (snow water equivalent, mm, annual sum)
 *   9  = SOIL (soil moisture, mm, annual mean)
 *   10 = SRAD (solar radiation, W/m², annual mean, ×0.1)
 *   11 = TMN  (minimum temperature, °C, annual mean, ×0.1)
 *   12 = TMX  (maximum temperature, °C, annual mean, ×0.1)
 *   13 = VAP  (vapor pressure, kPa, annual mean, ×0.001)
 *   14 = VPD  (vapor pressure deficit, kPa, annual mean, ×0.01)
 *   15 = VS   (wind speed, m/s, annual mean, ×0.01)
 *   16 = Elevation (m, SRTM, static)
 *   17 = Slope (degrees, SRTM-derived, static)
 *   18 = SOC  (soil organic carbon 0–5 cm, SoilGrids, static)
 *
 * Output: One GeoTIFF per year (2000–2022).
 *         Saved to Google Drive folder 'GEE_SAHEL_Data'.
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
// 2. Configuration
// ---------------------------------------------------------------------------
var startYear = 2000;
var endYear = 2022;
var years = ee.List.sequence(startYear, endYear);
var targetProjection = 'EPSG:4326';
var targetScale = 1000;

// ---------------------------------------------------------------------------
// 3. Load Static Layers
// ---------------------------------------------------------------------------
var dem = ee.Image('USGS/SRTMGL1_003');
var elevation = dem.select('elevation');
var slope = ee.Terrain.slope(dem);

var soil = ee.Image("projects/soilgrids-isric/soc_mean")
  .select('soc_0-5cm_mean')
  .rename('soil_oc');

// ---------------------------------------------------------------------------
// 4. Load Dynamic Collections
// ---------------------------------------------------------------------------
var ndviCol = ee.ImageCollection('MODIS/061/MOD13A1').select('NDVI');
var climate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE');

// ---------------------------------------------------------------------------
// 5. Build Annual Stacks
// ---------------------------------------------------------------------------
var annualData = years.map(function(year_obj) {
  var year = ee.Number(year_obj);
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = ee.Date.fromYMD(year, 12, 31);

  var annualClimate = climate.filterDate(start, end);

  // Accumulation variables (annual sum)
  var aet = annualClimate.select('aet').sum().rename('aet');
  var def = annualClimate.select('def').sum().rename('def');
  var pet = annualClimate.select('pet').sum().rename('pet');
  var pr  = annualClimate.select('pr').sum().rename('precipitation');
  var ro  = annualClimate.select('ro').sum().rename('runoff');
  var swe = annualClimate.select('swe').sum().rename('swe');

  // State variables (annual mean, with scale factors applied)
  var pdsi = annualClimate.select('pdsi').mean().multiply(0.01).rename('pdsi');
  var soilMoisture = annualClimate.select('soil').mean().rename('soil_moisture');
  var srad = annualClimate.select('srad').mean().multiply(0.1).rename('srad');
  var tmn  = annualClimate.select('tmmn').mean().multiply(0.1).rename('min_temp');
  var tmx  = annualClimate.select('tmmx').mean().multiply(0.1).rename('max_temp');
  var vap  = annualClimate.select('vap').mean().multiply(0.001).rename('vap');
  var vpd  = annualClimate.select('vpd').mean().multiply(0.01).rename('vpd');
  var vs   = annualClimate.select('vs').mean().multiply(0.01).rename('wind_speed');

  // Annual mean NDVI (1 km downscaled from MODIS 500 m)
  var annualNdvi = ndviCol.filterDate(start, end)
    .mean()
    .multiply(0.0001)
    .rename('ndvi');

  return ee.Image.cat([
    annualNdvi,
    aet, def, pdsi, pet, pr, ro, swe,
    soilMoisture, srad, tmn, tmx, vap, vpd, vs,
    elevation, slope, soil
  ])
    .reproject({crs: targetProjection, scale: targetScale})
    .clip(roi)
    .set('year', year);
});

var annualDataCollection = ee.ImageCollection.fromImages(annualData);
print('Annual Climate/Terrain Stack', annualDataCollection);

// ---------------------------------------------------------------------------
// 6. Export to Google Drive
// ---------------------------------------------------------------------------
for (var i = startYear; i <= endYear; i++) {
  var image = annualDataCollection.filter(ee.Filter.eq('year', i)).first();

  Export.image.toDrive({
    image: image.toFloat(),
    description: 'SAHEL_' + i,
    folder: 'GEE_SAHEL_Data',
    fileNamePrefix: 'SAHEL_' + i,
    region: roi,
    scale: targetScale,
    crs: targetProjection,
    maxPixels: 1e13
  });
}
