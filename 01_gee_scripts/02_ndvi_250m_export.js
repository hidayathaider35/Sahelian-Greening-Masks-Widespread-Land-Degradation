/**
 * 02_ndvi_250m_export.js
 *
 * Exports annual mean NDVI composites at 250 m resolution from MODIS
 * MOD13Q1 (Collection 6.1) for the Sahelian Acacia Savanna ecoregion.
 * Quality filtering is applied using the SummaryQA band to retain only
 * pixels flagged as 'Good' quality.
 *
 * Output: One GeoTIFF per year (2000–2022), scaled to true NDVI values.
 *         Saved to Google Drive folder 'GEE_NDVI_250m'.
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

// ---------------------------------------------------------------------------
// 3. Quality Masking and Scaling
// ---------------------------------------------------------------------------
function maskAndScaleMODIS(image) {
  var qa = image.select('SummaryQA');
  var goodData = qa.bitwiseAnd(3).eq(0);
  var ndvi = image.select('NDVI').multiply(0.0001);
  return ndvi.updateMask(goodData)
    .copyProperties(image, ['system:time_start']);
}

// ---------------------------------------------------------------------------
// 4. Create Annual Composites
// ---------------------------------------------------------------------------
var modisNDVI = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterBounds(roi);

var annualNDVI = years.map(function(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = ee.Date.fromYMD(year, 12, 31);

  return modisNDVI
    .filterDate(start, end)
    .map(maskAndScaleMODIS)
    .mean()
    .rename('NDVI')
    .clip(roi)
    .set('year', year)
    .set('system:time_start', start.millis());
});

var annualNDVICol = ee.ImageCollection.fromImages(annualNDVI);
print('Annual 250 m NDVI Collection', annualNDVICol);

// ---------------------------------------------------------------------------
// 5. Export to Google Drive
// ---------------------------------------------------------------------------
for (var i = startYear; i <= endYear; i++) {
  var image = annualNDVICol.filter(ee.Filter.eq('year', i)).first();

  Export.image.toDrive({
    image: image.toFloat(),
    description: 'SAHEL_NDVI_250m_' + i,
    folder: 'GEE_NDVI_250m',
    fileNamePrefix: 'SAHEL_NDVI_250m_' + i,
    region: roi.geometry(),
    scale: 250,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
}
