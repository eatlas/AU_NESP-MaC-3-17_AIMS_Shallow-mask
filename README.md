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
- **Python**: Version 3.8 or higher.
- **Dependencies**: Install required Python packages using the following command:
  ```bash
  pip install -r requirements.txt
  ```
  
### Required Packages
- `numpy`
- `pandas`
- `geopandas`
- `rasterio`
- `scipy`
- `opencv-python`
- `shapely`
- `matplotlib`
- `tqdm`

### Data Preparation
1. **Download Source Data**:
   Use `01-download-sentinel2.py` and `02-download-input-data.py` to download the required Sentinel-2 imagery and auxiliary datasets.
   ```bash
   python 01-download-sentinel2.py --dataset 15th_percentile --region NorthernAU
   python 02-download-input-data.py
   ```

2. **Generate Rough Mask**:
   Combine rough masks and GBR shapefiles.
   ```bash
   python 03-make-rough-mask-with-gbr.py
   ```



## Reproducing the Dataset

Run the scripts sequentially to reproduce the dataset. Details for the command line switches for each script to reproduce the dataset are provided in the doc string at the top of each script. For parallel processing, use the `--split` and `--index` arguments.

