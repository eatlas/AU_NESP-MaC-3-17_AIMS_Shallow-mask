# Shallow Marine Areas Mask for Northern Australia and GBR (NESP MaC 3.17, AIMS)

## Overview

This repository contains a series of scripts to calculate a comprehensive dataset of shallow marine areas across Northern Australia and the Great Barrier Reef (GBR). Derived from Sentinel-2 composite imagery, the dataset maps intertidal zones, shallow subtidal habitats (down to approximately 5 meters in turbid waters), and offshore reef features visible at depths of up to 40 meters in clear waters. The final output provides essential inputs for reef boundary mapping and shallow water habitat modeling.

The repository contains all the code needed to reproduce the creation of the semi-automated shallow marine areas mask dataset documented at: https://eatlas.org.au/geonetwork/srv/eng/catalog.search#/metadata/1045e83b-62cd-4bf2-afb3-684c39a1a72d

These scripts detail downloading all the source datasets and processing the final masks from the satellite imagery. Fully reproducing this dataset requires downloading 153 GB of satellite imagery along with 9 GB of other reference dataset. The processing takes approximately 12 CPU days.

The dataset spans Northern Australia, GBR and Christmas Island, Cocos (Keeling) Islands, Lord Howe Island, and Norfolk Island.

## Repository Structure

### Scripts Overview

1. **`01-download-sentinel2.py`**
   Downloads Sentinel-2 composite imagery for specific regions and image styles. Supports resumable downloads and subsetting by region or tile.

2. **`02-download-input-data.py`**
   Downloads supporting datasets such as coastline shapefiles and other geospatial references.

3. **`03-make-rough-mask-with-gbr.py`**
   Combines the manual Rough Reef Shallow Mask with the existing GBR and Torres Strait reef boundaries dataset to produce a rough mask for the whole of Australia. This is used to prime the creation of the water estimation.  

4. **`04-create-water-image_with_mask.py`**
   Generates a water estimate image by masking land and from the Rough Reef Shallow Mask and interpolates over these areas from the colour of the surrounding water. This water estimate allows reef detection under varying water conditions. The water estimates can be used to create a contrast enhance version of the satellite imagery (see `08-create-water-est-enhanced-imagery.py`). 

5. **`05-create-shallow-and-reef-area-masks.py`**
   Detects shallow marine features using a combination of detectors optimized for different water conditions and sensitivity levels. This script embodies the bulk of the algorithms to map the shallow regions.

6. **`06a-merge-scene-masks.py`**
   Merges individual scene outputs into a single shapefile for each region. First stage of combining.

7. **`06b-merge-regions.py`**
   Combines regional shapefiles into a single dataset for all of tropical Australia. Second stage of combining.

8. **`06c-combine-sensitivities.py`**
   Refines outputs by combining masks from different sensitivity levels to improve accuracy. This is used to allow mapping boundaries from a higher sensitivity level, but only including features from a lower sensitivity level. This removes features that are just on the edge of detection, helping to clean up the data.

9. **`07a-download-qaqc-data.py`**
   Downloads additional datasets for quality assurance and quality control (QAQC). This includes bathymetry datasets.

10. **`07b-generate-qaqc-boundary-points.py`**
    Generates QAQC random points along polygon boundaries for accuracy assessment. The points created from this script must be manually aligned to the true position of the boundaries.

11. **`07c-qaqc-assess-boundary-error.py`**
    Compares boundary points against ground truth to assess positional accuracy.

12. **`07d-compare-reef-masks.py`**
    Compares manual and automated reef masks to evaluate true positives, false positives, and false negatives.

13. **`08-create-water-est-enhanced-imagery.py`**
    Enhances Sentinel-2 imagery by applying water-based local contrast adjustments.

## Setup Instructions

### Requirements
- **Python**: Version 3.9 or higher.

1. Install required Python packages using the following command:
```bash
conda env create -f environment.yml
```
2. Activate the environment
```bash
conda activate reef_maps
```
  
### Required Packages
The following are the top level libraries needed. These will in turn pull in many dependances.
  - python=3.11.11
  - affine=2.3.0
  - geopandas=0.14.2
  - matplotlib-base=3.9.2
  - numpy-base=1.26.4
  - pandas=2.1.1
  - rasterio=1.3.10
  - opencv-python-headless=4.10.0 (installed via pip as conda was causing DLL issues)

The environment.yml contains all the dependancies and allows for faster reproduction. 

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
The following is a small run that processes just one Sentinel 2 tile. This is useful for testing the environment setup as it doesn't require downloading and processing all the data, which iterally takes days.

We process one tile from the GBR and one from the NorthernAU regions. You can process everything on a single tile up to `05-create-shallow-and-reef-area-masks.py`, but the `06` scripts are all about merging multiple tile together. These will be somewhat redundant in this tutorial, except to move and package the files in the correct locations for the subsequent scripts to pick up.
 
1. **Download Source Data for 1 sentinel 2 tile [3 - 10 min]**:
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
2. **Download auxilary input data [1 min]**:
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
   The `--sigma` corresponds to the amount of blurring (gaussian radius in pixels) that is applied after the mask areas have been replaced with an estimated infill from a large blur (160 pixels). Small values of sigma will make the detection less suseptable to turbid plume patterns, but less sensitive to large reefs that are not masked properly. This would probably make the result map the fringing coastal areas worse.
   
   Run the two Python calls in separate command lines will halve the execution time.
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
4. The reference imagery used for determining the true boundaries is 10 m resolution. This low resolution, combined with image uncertainty makes the it difficult to reliably determine the true boundaries better than 20 m error.
