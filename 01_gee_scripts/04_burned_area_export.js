/**
 * 04_burned_area_export.js
 *
 * Exports annual binary burned-area maps at 500 m resolution from MODIS
 * MCD64A1 (Collection 6.1) for the Sahelian Acacia Savanna ecoregion.
 *
 * For each year, monthly BurnDate layers are binarized (burned = 1,
 * unburned = 0) and composited using a maximum reducer so that any pixel
 * burned in any month of the year is flagged.
 *
 * Output: One GeoTIFF per year (2000–2022), byte format (0/1).
 *         Saved to Google Drive folder 'GEE_Burned_Area'.
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
var targetScale = 500;

// ---------------------------------------------------------------------------
// 3. Load MODIS Burned Area Collection
// ---------------------------------------------------------------------------
var burnedAreaCol = ee.ImageCollection('MODIS/061/MCD64A1').select('BurnDate');

// ---------------------------------------------------------------------------
// 4. Create Annual Binary Composites
// ---------------------------------------------------------------------------
var annualBurnedArea = years.map(function(year_obj) {
  var year = ee.Number(year_obj);
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = ee.Date.fromYMD(year.add(1), 1, 1);

  var burnedBinary = burnedAreaCol
    .filterDate(start, end)
    .map(function(img) { return img.gt(0); })
    .max();

  return burnedBinary
    .reproject({crs: targetProjection, scale: targetScale})
    .clip(roi)
    .set('year', year)
    .set('system:time_start', start.millis());
});

var collection = ee.ImageCollection.fromImages(annualBurnedArea);
print('Annual Burned Area Collection', collection);

// ---------------------------------------------------------------------------
// 5. Export to Google Drive
// ---------------------------------------------------------------------------
for (var i = startYear; i <= endYear; i++) {
  var image = collection.filter(ee.Filter.eq('year', i)).first();

  Export.image.toDrive({
    image: image.toByte(),
    description: 'SAHEL_BURN_500m_' + i,
    folder: 'GEE_Burned_Area',
    fileNamePrefix: 'SAHEL_BURN_500m_' + i,
    region: roi,
    scale: targetScale,
    crs: targetProjection,
    maxPixels: 1e13
  });
}
