# Shallow Marine Areas Mask for Northern Australia and GBR (NESP MaC 3.17, AIMS)

## Overview

This repository contains a series of scripts to calculate a comprehensive dataset of shallow marine areas across Northern Australia and the Great Barrier Reef (GBR). Derived from Sentinel-2 composite imagery, the dataset maps intertidal zones, shallow subtidal habitats (down to 0 - -5 meters LAT in turbid waters), and offshore reef features visible at depths of up to 40 meters in clear waters. The final output provides essential inputs for reef boundary mapping and shallow water habitat modeling. 

### Application
When combined with manual reef boundary mapping (coral reefs and rocky reefs) it can be used to estimate shallow sediment areas, by subtracting these hard substrate estimates from the shallow areas mapped from this dataset. 

## What does this repo contain?
The repository contains all the code needed to reproduce the creation of the semi-automated shallow marine areas mask dataset documented at: https://doi.org/10.26274/x37r-xk75

These scripts detail downloading all the source datasets and processing the final masks from the satellite imagery. Fully reproducing this dataset requires downloading 153 GB of satellite imagery along with 9 GB of other reference dataset. The processing takes approximately 12 CPU days.

The dataset spans Northern Australia, GBR and Christmas Island, Cocos (Keeling) Islands, Lord Howe Island, and Norfolk Island.

## Repository Structure

### Scripts Overview

- **`01a-download-sentinel2.py`**
   Downloads Sentinel-2 composite imagery for specific regions and image styles. Supports resumable downloads and subsetting by region or tile. 

- **`01b-create-s2-virtual-rasters.py`**
   Creates virtual rasters (image mosaic) from the Sentinel 2 tile imagery to allow easy loading and editing of the imagery in QGIS.

- **`02-download-input-data.py`**
   Downloads supporting datasets such as coastline shapefiles and other geospatial references.

- **`03-make-rough-mask-with-gbr.py`**
   Combines the manual Rough Reef Shallow Mask with the existing GBR and Torres Strait reef boundaries dataset to produce a rough mask for the whole of Australia. This is used to prime the creation of the water estimation.  

- **`04-create-water-image_with_mask.py`**
   Generates a water estimate image by masking land and from the Rough Reef Shallow Mask and interpolates over these areas from the colour of the surrounding water. This water estimate allows reef detection under varying water conditions. The water estimates can be used to create a contrast enhance version of the satellite imagery (see `08-create-water-est-enhanced-imagery.py`). 

- **`05-create-shallow-and-reef-area-masks.py`**
   Detects shallow marine features using a combination of detectors optimized for different water conditions and sensitivity levels. This script embodies the bulk of the algorithms to map the shallow regions.

- **`06a-merge-scene-masks.py`**
   Merges individual scene outputs into a single shapefile for each region. First stage of combining.

- **`06b-merge-regions.py`**
   Combines regional shapefiles into a single dataset for all of tropical Australia. Second stage of combining.

- **`06c-combine-sensitivities.py`**
   Refines outputs by combining masks from different sensitivity levels to improve accuracy. This is used to allow mapping boundaries from a higher sensitivity level, but only including features from a lower sensitivity level. This removes features that are just on the edge of detection, helping to clean up the data.
   
- **`06d-create-virtual-rasters.py`**
   This creates virtual rasters for the input and output raster mosaics so they can be easily viewed in QGIS without having to handle many individual files. This should be run prior to opening the `Preview-maps.qgz` in QGIS. 

- **`08-create-water-est-enhanced-imagery.py`**
    Enhances Sentinel-2 imagery by applying water-based local contrast adjustments.

These scripts were originally written to process northern Australia and the GBR separately as the original plan was to only prepare the data for northern Australia. We decided process both regions, but the scripts are setup to process each region separately then be merged using 06b-merge-regions.

## Setup Instructions

### Requirements
- **Python**: Version 3.9 or higher. The scripts have been tested on Python 3.9 and 3.11.

1. Install required Python packages using the following command:
```bash
conda env create -f environment.yml
```
2. Activate the environment
```bash
conda activate reef_maps
```
  
### Required Packages
The following are the top level libraries needed. These will in turn pull in many dependencies.
  - python=3.11.11
  - affine=2.3.0
  - geopandas=0.14.2
  - matplotlib-base=3.9.2
  - numpy-base=1.26.4
  - pandas=2.1.1
  - rasterio=1.3.10
  - scipy=1.14.1
  - shapely=2.0.6
  - tqdm=4.67.1
  - scikit-image=0.24.0
  - fiona=1.10.1
  - opencv-python-headless=4.10.0 (installed via pip as conda was causing DLL issues)

The environment.yml contains all the dependencies and allows for faster reproduction. 

## Debug
When I had installed opencv using conda I got the following error: 
`ImportError: DLL load failed while importing cv2: The specified procedure could not be found.`
To fix the problem I needed to remove open-cv from conda and reinstall using pip
```bash
conda remove opencv
pip install opencv-python-headless
```

You can partly test the opencv install with:
```bash
python -c "import cv2; print(cv2.__version__)"
```

### Small trial run [~ 40 min]
The following is a small run that processes just one Sentinel 2 tile. This is useful for testing the environment setup as it doesn't require downloading and processing all the data, which literally takes days.

We process one tile from the GBR. You can process everything on a single tile up to `05-create-shallow-and-reef-area-masks.py`, but the `06` scripts are all about merging multiple tile together. These will be somewhat redundant in this tutorial, except to move and package the files to the correct locations for the subsequent scripts to pick up.
 
1. **Download Source Data for 1 sentinel 2 tile [3 - 10 min depending on network speed]**:
   This will download the Sentinel 2 data for a tile (south of Cairns) on the GBR.
   ```bash
   python 01a-download-sentinel2.py --dataset 15th_percentile --region GBR --tiles 55KDA 
   python 01a-download-sentinel2.py --dataset low_tide_true_colour --region GBR --tiles 55KDA
   ```
   If you want to process a different tile then you can determine the tile ID using the [Sentinel 2 tiling grid map](https://maps.eatlas.org.au/index.html?intro=false&z=7&ll=148.00000,-18.00000&l0=ea_ref%3AWorld_ESA_Sentinel-2-tiling-grid_Poly,ea_ea-be%3AWorld_Bright-Earth-e-Atlas-basemap)
   The region needs to match the tile ID, otherwise an error will be reported. 
   
   You can also download multiple tiles by listing more than one tile in the download script:
   ```bash
   python 01a-download-sentinel2.py --dataset 15th_percentile --region GBR --tiles 55KDA 55KDV
   python 01a-download-sentinel2.py --dataset low_tide_true_colour --region GBR --tiles 55KDA 55KDV
   ```
   Note: `01b-create-s2-virtual-rasters.py` is intended to create a virtual raster mosaic when downloading all the images for a region. Since we are only processing a single scene it doesn't need to be run.
2. **Download auxiliary input data [1 min]**:
   This will download the Australian Coastline, the GBR features dataset, the Rough-reef-shallow-mask and the Cleanup-remove-mask. These are all needed for creating various masks in the processing.
   
   This will save the source data into `input-data` and `input-data-3p`, where `3p` represents datasets that are third party to this dataset. 
   ```bash
   python 02-download-input-data.py
   ```
3. **Generate Rough Mask [15 min]**:
   Combine rough masks and GBR shapefiles to create a starting mask for the whole of Australia. This saves the result to `working-data`.
   ```bash
   python 03-make-rough-mask-with-gbr.py
   ```
4. **Create Water Estimate [6 min]**:
   Using the starting mask we calculate the estimated water without reefs and islands. Here we force it to just process a single image rather than a whole region. By default if the `--priority` and `--justpriority` options are left off then the script will process all the images downloaded for the region specified. They are redundant in this tutorial case, but I included them to highlight how you can process a subset of the available imagery.
   The `--sigma` corresponds to the amount of blurring (gaussian radius in pixels) that is applied after the mask areas have been replaced with an estimated infill from a large blur (160 pixels). Small values of sigma will make the detection less susceptible to turbid plume patterns, but less sensitive to large reefs that are not masked properly. This would probably make the result map the fringing coastal areas worse.
   
   Running the two Python calls in separate command lines will halve the execution time.
   ```bash
   python 04-create-water-image_with_mask.py --style low_tide_true_colour --sigma 40 --region GBR --priority 55KDA --justpriority True
   python 04-create-water-image_with_mask.py --style 15th_percentile --sigma 40 --region GBR --priority 55KDA --justpriority True
   ```
5. **Create Shallow Feature Mask [6 min]**
   In this stage we take the difference between the satellite imagery and the water estimate to find the shallow features. We are calculating the result for two levels of detector sensitivity so that they can be combined in Step 7. Note these commands can be run in two separate terminals to speed up the processing. This is also true in later steps.
   ```bash
   python 05-create-shallow-and-reef-area-masks.py --region GBR --detectors Medium --sigma 40
   python 05-create-shallow-and-reef-area-masks.py --region GBR --detectors High --sigma 40
   ```
   This script also supports `--priority` and `--justpriority`.
   The results of this analysis will be saved in `working-data\05-auto-mask\`. The final shapefile 
6. **Merge Scenes and Regions [1 min]**
   Normally this script would merge all the shapefiles corresponding to one shapefile per tile, into one shapefile per region, per level of detector sensitivity. Since we are only processing a single tile the merge is a bit redundant, however running this script will ensure the output is in the correct location for the next script.
   ```bash
   python 06a-merge-scene-masks.py --region GBR --detectors Medium --version 1-1 --sigma 40
   python 06a-merge-scene-masks.py --region GBR --detectors High --version 1-1 --sigma 40
   ```
   We also merge all the shapefiles from multiple regions into a single shapefile. This is used to combine the NorthernAU and GBR shapefiles into one. In this case it is redundant, other than making sure the files connect with the next script.
   ```bash
   python 06b-merge-regions.py --detectors Medium --version 1-1 --sigma 40
   python 06b-merge-regions.py --detectors High --version 1-1 --sigma 40
   ```
7. **Combine Sensitivities [1 min]**
   This script allows multiple detect sensitivities to be combined slightly improving the quality of the boundaries. This script takes the polygon boundaries from a more sensitive detector level, but only includes that boundary if it was detected at a low level of sensitivity. This gives the advantage of getting the boundaries mapped deeper, and thus more complete, without the additional false positives.
   Note: This only works if the regions have been processed at multiple levels of detector sensitivity.
   ```bash
   python 06c-combine-sensitivities.py --poly-sensitivity High --keep-sensitivity Medium --version 1-1
   ```
8. **Review the result**
   The generated result will be in `out-data`. The `AU_NESP-MaC-3-17_AIMS_Shallow-mask_High-Medium_V1-1.shp` can be viewed in QGIS.

## Editing the rough-reef-shallow-mask
The `Rough-reef-shallow-mask` is intended to mask our all areas that will pollute water estimation process. It is intended to mask shallow areas. In inshore areas where there is a strong brightness gradient the positioning of this mask is critical for the final automated mapped boundary, as it will closely follow the rough-reef-shallow-mask. For isolated islands and platform reefs the accuracy of the masking is generally less important as the water gradients around these features is typically less.

This section contain notes about editing the rough-reef-shallow-mask. Editing is only necessary for corrections or improvements to be made to the mask, and is not necessary to rerun the automated analysis. These notes will also be useful for repeating this type of analysis in other areas.

The `Rough-reef-shallow-mask` is edited in QGIS using a number of satellite image layers as reference. 

### Setup for editing
To start editing the `Rough-reef-shallow-mask` the input data needs to be downloaded.

1. Download the satellite imagery
```bash
python 01-download-sentinel2.py --dataset 15th_percentile --region NorthernAU
python 01-download-sentinel2.py --dataset low_tide_true_colour --region NorthernAU
python 01-download-sentinel2.py --dataset low_tide_infrared --region NorthernAU
python 01-download-sentinel2.py --dataset 15th_percentile --region GBR
python 01-download-sentinel2.py --dataset low_tide_true_colour --region GBR
python 01-download-sentinel2.py --dataset low_tide_infrared --region GBR
```
This is a full download of ~200GB and so will take a long while to complete.
This will create the following directories each with the Geotiffs in the lower directories.
- in-data-3p
  - AU_AIMS_S2-comp
    - low_tide_true_colour
      - NorthernAU
      - GBR
    - 15th_percentile
      - NorthernAU
      - GBR
    - low_tide_infrared
      - NorthernAU
      - GBR

2. Create virtual rasters of the satellite imagery
The virtual rasters need to be setup so that each image style can be treated as a single mosaic, rather than as hundreds of individual images.
```bash
python 01b-create-S2-virtual-rasters.py --style 15th_percentile --region all
python 01b-create-S2-virtual-rasters.py --style low_tide_true_colour --region all
python 01b-create-S2-virtual-rasters.py --style low_tide_infrared --region all
```
This will create the following virtual rasters that can be used in QGIS:
```
working-data\01b-S2-virtual-rasters\15th_percentile_all.vrt
working-data\01b-S2-virtual-rasters\low_tide_true_colour_all.vrt
working-data\01b-S2-virtual-rasters\low_tide_infrared_all.vrt
```
3. Download input vector datasets
This will download the coastlines, the Torres Strait and GBR reefs, the existing public  `Rough-reef-shallow-mask` and the `'Cleanup-remove-mask`.
```bash
python 02-download-input-data.py
```


### QGIS Editing
The `Rough-reef-shallow-mask` is edited in QGIS based on reference layers derived from Sentinel 2 composite imagery.

1. Open `Edit-masks.qgz` in QGIS. This will reference data layers downloaded by `01a-download-sentinel2-py` and `02-download-input-data.py`. It will expect the imagery to be available as virtual rasters setup using `01b-create-s2-virtual-rasters.py`.

### Setting up satellite depth reference layers
The following is a record of how the reference layers were setup in `Edit-masks.qgz`. They do not need to be resetup to edit the masks. Here we use bathymetry layers to calibrate satellite derived bathymetry layers to act as a reference for the masking. The use of the bathymetry layers is limited to calibration use along the coastline where the turbidy is the highest. 
1. Download the bathymetry data
```bash
python 07a-download-qaqc-data.py
```
2. Build a virtual raster of the `in-data-3p\Kim_GA_SDB_2021` dataset. 
```bash
python 01b-create-S2-virtual-rasters.py --kimberley
```
Alternatively you can create the virtual raster in QGIS use `Raster > Miscellaneous > Build virtual raster ...`. Add `in-data-3p\Kim_GA_SDB_2021` directory. Save to file: `working-data\01b-S2-virtual-rasters\WA_GA_Kimberley-SDB_2021.vrt`.
Note that the sections covering Clerke Reef, Cunningham Island and Mermaid Reef will be excluded because they are in a different projection. This is OK because we are not using these sections.
The layer styling was adjusted to show from -6 m to -5 m. It should be noted that this particular bathymetry dataset is derived from satellite imagery and thus prone to false shallow areas in areas with very high turbidity. The boundary clipping is also prone to falsely clipping out some dark reef areas. As such we only use this dataset as a rough guide for depth. This dataset is measured against Mean Sea Level (MSL). We ideally want to map against Lowest Astronomical Tide as it better represents areas that are exposed at low tide. The tidal range varies widely across northern Australia. In areas with a low tidal range a depth of -4 m would always be subtidal, but in areas with a high tidal range the area would be frequently exposed. 
3. Add the AHO ENC Series marine charts as a depth reference
The AHO marine charts doesn't have detailed coverage of many inshore areas. We focus on areas where there depth soundings that are -1 m - 1 m LAT, and compare these to the Kim_GA_SDB_2021 and the thresholds chosen in the rough reef mask.  

We find that the Kim_GA_SDB_2021 dataset excludes areas that are highly turbid and in areas where the tidal range is high it doesn't fully represent the areas exposed at low tide. For example in the flats just north of Yule Entrance (WA) (Latitude: -16.2514, Longitude: 124.39548)the bathymetry dataset is limited to -2.5 m, however the tidal range is > 10 m and so LAT is much lower. This means that the tidal flat based on the Kim_GA_SDB_2021 dataset would be approaximately 350 m. The AHO marine charts show the flat to be 1.4 km wide. The low-tide Sentinel composite imagery shows an exposed tidal flat 1.3 km across. This matches the extend observed by the GeoScience Australia DEA Intertidal extent, with the outer extent corresponding to a 5% exposure. This indicates that the low tide satellite observations include nearly the full tidal range, close to LAT.
This layer was added as a new WMS Connection.
- **Name**: AHO ENC Series
- **URL**: https://services.hydro.gov.au/site1/services/Basemaps/AHOENCSeries/ImageServer/WMSServer
This layer was then styled with inverted colours and an Addition image blending method so that the chart could be blended with the other bathmetry layers.




### Using bathymetry data
We intentionally don't used available bathymetry data for the masking because we reserve this data for validation. We do however use portions the bathymetry to roughly calibrate some contrast enhanced versions of the satellite data to act as depth guides for the mapping. 




## Reproducing the Dataset

Run the scripts sequentially to reproduce the dataset. Details for the command line switches for each script to reproduce the dataset are provided in the doc string at the top of each script. For parallel processing, use the `--split` and `--index` arguments.

## QAQC evaluation
The QAQC for version 1-1 is incomplete. Only a prelimary assessment was developed and the scripts to assess the dataset. This identified a range of potential improves that should be made in the next version. 

## Draft boundary assessment process (Not fully implemented in version 1-1)

The digitisation accuracy of the masks were estimated using 300 control point locations. Random locations along the outer boundary of the Rough-reef-mask-with-GBR (03-make-rough-mask-with-gbr.py) shapefile, excluding locations overlapping land where selected as an initial control point. These locations were then randomly dithered by 100 m to ensure they do not align exactly with the Rough-reef-mask-with-GBR shapefile boundary. Each of these locations were then manually adjusted to lie somewhere on the best estimate of the boundary near the original randomly selected location. This was achieved by expert visual assessment using the following data sources: 
1. All-tide and low-tide Sentinel 2 composite imagery (Hammerton and Lawrey, 2024a, 2024b), with local contrast enhancement.
2. Digital Earth Australia Intertidal Extent (Bishop-Taylor et al., 2019; Bishop-Taylor et al., 2024)
3. In the GBR and Torres Strait region the AusBathyTop 30 m bathymetry (Beaman, 2017; Beaman 2023).  
4. In the Kimberley region the Kimberley Satellite Derived Bathymetry (Twiggs, 2023)
5. In WA offshore the North West Shelf DEM Compilation (Lebrec et al., 2021; Lebrec, 2021)

None of the boundary layers (Rough-reef-mask-with-GBR or the semi-automated mask) where turned on during the positioning of the control points. 

To provide an approximate depth reference the red channel and near infrared channels of the low tide Sentinel 2 imagery were contrast enhanced to produce an -3 to -6 m contour. 
The minimum and maximum of the imagery was adjusted to align with this bathymetry range, aligning the levels to the closest reference bathymetry. A simple satellite derive bathymetry based on the ratio of the blue and green channels was also used to provide a third estimate. The thresholds used were calibrated against the Kimberley SDB on the western side of the Dampier Peninsula, where there is clear water, and using the North West Shelf DEM south of Barrow Island, which is an area that was mapped using LADS (Lebrec et al., 2021).  

In many locations the exact location of the correct boundary is highly uncertain and as a result the control points will have significant noise. This noise provides a mechanism for estimating the total uncertainty in the boundaries.

This assessment assumes that with additional time, and some additional data an expert can determine the true boundary with greater accuracy than the initial manual mapping. This is a reasonable assumption for the manual rough-shallow-reef-mask as the mapping was performed at high speed, at an average rate of one reef per 30 seconds or 1.5 seconds per vertex. The accuracy assessment allowed much more time (averaging 40 seconds per vertex) to position the control point resulting in much high positioning accuracy.  

This accuracy assessment is however flawed in several ways:
1. Any biases in the interpretation of the imagery by the expert will affect the determination of the estimated true boundary and the repositioning of the vertex.
2. Many shallow features are ambiguous in the satellite imagery, making determining the true boundary error prone regardless of the amount of time spent on each assessment.
3. In shallow soft sediment areas definition of this dataset is only loosely defined. This dataset is intended to act as a visual clipping mask with the depth threshold varying with the water clarity. This makes it difficult to determine the true boundary that should have been mapped difficult, particularly in areas where the image gradients are low. To compensate for this in low gradient, visually uncertain areas the accuracy assessment vertex was positioned so that the measured error (difference with the original vertex position) corresponded to the level of uncertainty.
4. The reference imagery used for determining the true boundaries is 10 m resolution. This low resolution, combined with image uncertainty makes it difficult to reliably determine the true boundaries better than 20 m error.

# Things that are mapped
- Intertidal zone to 0 m LAT 
- Fringing reefs (rocky and coral)
- Fringing probable seagrass
- Platform reefs (rocky and coral)
- Sand banks (raised areas of sediment above surrounding seafloor)

# Rough Reef Shallow Mask versions
Shapefile, Shapefile size (KB), Features, Time Northern AU (hr), Time GBR (hr), Total (hr)
AU_AIMS_NESP-MaC-3-17_Rough-reef-shallow-mask_18hr.shp, 903, 2583, 18.3, 0, 18.3
AU_AIMS_NESP-MaC-3-17_Rough-reef-shallow-mask_39hr.shp, 1871, 5135, 36.5, 2.58, 39.08
AU_AIMS_NESP-MaC-3-17_Rough-reef-shallow-mask_57hr.shp, 2598, 7126, 52.87, 4.32, 57.19
AU_AIMS_NESP-MaC-3-17_Rough-reef-shallow-mask_87hr.shp, 3583, 7733, 77.18, 9.86, 87.04

In this version we improved the mapping of the intertidal region to more closely align with LAT using the remote sensing LAT indicators and the AHOENC series to locally calibrate the indicators.

Tracking notes for the digitisation of the mask:


9:49 9:58 9 min 2621 KB 7135 features
7:01 9:40 2hr 39 min 2690 KB 7142 features
10:09 11:00 51 min 2720 KB 7144 features
11:47 12:28 41 min 2746 KB 7152 features
4:40 5:09 29 min 2764 KB 7155 features
5"32 6:26 54 min 2805 KB 7190 features
9 37 10:05 28 min 2822 KB 7198 features

7:01  9:05 2h 4 min 2904 KB 7219 features
9:06 9:13 7 min 2907 KB 7216 features 
3:42 5:18 1 hr 36 min 2975 KB 7278 features
5:20 6:11 51 min 3011 KB 7321 features
8:35 9:57 1 hr 22 min 3053 KB GBR 7396 features
6:44 9:40 2 hr 56 min 3170 KB 7475 features (~1:30 on GBR)
11:06 12:07 1 hr 1 min 3208 KB 7499 features
12:48 12:58 10 min 3214 KB 7505 features
2:14 4:18 2 hr 4 min 3281 KB 7584 features GBR
4:20 5:36 2 hr 16 min 3316 KB 7610 features
5:38 5:54 16 min 3326 KB 7614 features

#Checking
8:21 8:34 8:40 9:41 1 hr 14 min 3395 KB 7629 features
6:16 8:22 2 hr 6 min 3505 KB 7643 features
9:06 10:27 1 hr 1 min 3546 KB 7686 features
2:30 2:53 23 min 3563 KB 7695 features
5:09 5:38 29 min 3583 KB 7751 features
8:24 8:42 18 min 3583 KB 7751 features
New 22.48 hours Northern Au, 4.93 hours GBR
Total: 75.35 hours Northern Au, 9.86 hours GBR (Version 85 hr)

# LAT fix up
8:06 9:26 1hr 20 min 3581 KB 7744 features
11:37 12:37 30 min 3582 KB 7733 features
Total: 77.2 hours Northern AU, 9.86 hours GBR (Version 87 hr)



5:24 5:41 12 KB 16 features GBR
6:29 6:37 17 KB 24 features 
7:23 7:41 
