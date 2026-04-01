/**
 * 01_land_cover_export.js
 * 
 * Exports annual 30 m land-cover maps (GLC_FCS30D) for the Sahelian Acacia
 * Savanna ecoregion, reclassified from the original 35 fine classes into a
 * 10-class system used for the main land-cover analysis.
 *
 * Output: One GeoTIFF per year (2000–2022) with class values 1–10.
 *         Saved to Google Drive folder 'GLC_Sahel_Processed'.
 *
 * Class mapping:
 *   1 = Cropland       6 = Wetland
 *   2 = Forest         7 = Impervious
 *   3 = Shrubland      8 = Bare land
 *   4 = Grassland      9 = Water
 *   5 = Tundra        10 = Ice/Snow
 *   0 = No Data
 */

// ---------------------------------------------------------------------------
// 1. Define Region of Interest
// ---------------------------------------------------------------------------
var ecoRegions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017');
var roi = ecoRegions
  .filter(ee.Filter.stringContains('ECO_NAME', 'Sahel'))
  .geometry();
Map.centerObject(roi, 5);
Map.addLayer(roi, {color: 'red'}, 'Sahel ROI');

// ---------------------------------------------------------------------------
// 2. Load GLC_FCS30D Annual Collection
// ---------------------------------------------------------------------------
var annualCollection = ee.ImageCollection(
  "projects/sat-io/open-datasets/GLC-FCS30D/annual"
);
var glcMultiBand = annualCollection.mosaic().clip(roi);

// ---------------------------------------------------------------------------
// 3. Reclassification Lookup (Original codes → 10 classes)
// ---------------------------------------------------------------------------
var fromList = [
  10, 11, 12, 20,                          // Cropland
  51, 52, 61, 62, 71, 72, 81, 82, 91, 92,  // Forest
  120, 121, 122,                            // Shrubland
  130,                                      // Grassland
  140,                                      // Tundra
  181, 182, 183, 184, 185, 186, 187,        // Wetland
  190,                                      // Impervious
  150, 152, 153, 200, 201, 202,             // Bare land
  210,                                      // Water
  220,                                      // Ice/Snow
  0, 250                                    // No Data
];

var toList = [
  1, 1, 1, 1,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  3, 3, 3,
  4,
  5,
  6, 6, 6, 6, 6, 6, 6,
  7,
  8, 8, 8, 8, 8, 8,
  9,
  10,
  0, 0
];

// ---------------------------------------------------------------------------
// 4. Export Reclassified Maps (2000–2022)
// ---------------------------------------------------------------------------
var exportScale = 30;
var years = ee.List.sequence(2000, 2022);

years.evaluate(function(yearList) {
  yearList.forEach(function(year) {
    var bandName = 'b' + (year - 1999);
    var rawImg = glcMultiBand.select(bandName);

    var reclassified = rawImg
      .remap(fromList, toList)
      .rename('GLC_10class')
      .set('year', year)
      .toByte();

    Export.image.toDrive({
      image: reclassified,
      description: 'GLC_FCS30D_' + year + '_10classes',
      folder: 'GLC_Sahel_Processed',
      region: roi,
      scale: exportScale,
      maxPixels: 1e13,
      fileFormat: 'GeoTIFF'
    });
  });
});

print('Export tasks created. Check the Tasks tab to run them.');
