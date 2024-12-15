"""
Script: 01b-create-S2-virtual-rasters.py

Purpose:
This script creates a virtual raster (VRT) from Sentinel-2 GeoTIFF images stored in the directory structure generated by 01a-download-sentinel2.py. Virtual rasters are mosaics that combine multiple raster files allowing a collection of GeoTiffs to be managed and restyled in QGIS as a single raster. This script is used to create the virtual rasters used in the Preview-maps.qgz QGIS file. 

Usage:
Run the script with the following options:
- `--style`: Specify the image style (default: '15th_percentile'). Valid options include 'low_tide_true_colour' and 'low_tide_infrared'.
- `--region`: Specify the region ('NorthernAU', 'GBR', or 'all'). Note the all region will create a virtual raster that combines both the NorthernAU and GBR regions. Only use this option if you downloaded both regions.

Dependancies:
This script relies on gdalbuildvrt to be installed in the Python environment.
'pip install gdal' 

Commands used to reproduce this dataset:
python 01b-create-S2-virtual-rasters.py --style 15th_percentile --region all
python 01b-create-S2-virtual-rasters.py --style low_tide_true_colour --region all
python 01b-create-S2-virtual-rasters.py --style low_tide_infrared --region all

Alternative Method (Using QGIS):
If you cannot run this script in Python or do not have `gdalbuildvrt` installed, you can create virtual rasters directly in QGIS:
1. Open QGIS and go to "Raster" > "Miscellaneous" > "Build Virtual Raster (Catalog)...".
2. Select all the GeoTIFF files you want to include in the VRT.
3. Set the output file location and name.
4. Adjust settings as needed (e.g., resolution, CRS) and click "Run".
5. The resulting VRT file can be loaded into QGIS for visualization or further analysis.
"""

import os
import glob
import argparse
import subprocess
from pathlib import Path

def check_gdalbuildvrt():
    """Check if gdalbuildvrt is available in the system."""
    try:
        subprocess.run(["gdalbuildvrt", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return True  # gdalbuildvrt exists but another issue occurred

def check_proj_db():
    """Check if proj.db is accessible."""
    proj_env = os.environ.get("PROJ_LIB")
    if proj_env and os.path.exists(proj_env):
        proj_db_path = os.path.join(proj_env, "proj.db")
        if os.path.exists(proj_db_path):
            return True
        else:
            print(f"Warning: proj.db not found in PROJ_LIB directory: {proj_env}")
    else:
        print("Warning: PROJ_LIB environment variable is not set or does not point to a valid directory.")
    return False

def create_virtual_raster(base_path, style, region):
    if not check_gdalbuildvrt():
        print("Error: gdalbuildvrt is not installed or not available in the system PATH.")
        print("To install gdalbuildvrt, you can use one of the following methods:")
        print("- Install GDAL via your package manager (e.g., apt, brew, or yum).")
        print("- Install GDAL in your Python environment using 'pip install gdal'.")
        print("- Ensure GDAL is properly added to your system PATH.")
        return

    if not check_proj_db():
        print("Error: PROJ library is not configured correctly or proj.db is missing.")
        print("To fix this, you can:")
        print("- Install or reinstall PROJ and ensure the proj.db file is present.")
        print("- Set the PROJ_LIB environment variable to the directory containing proj.db.")
        print("- Use a conda environment with GDAL and PROJ pre-configured (e.g., 'conda install gdal').")
        return

    # Define input paths
    input_path = os.path.join(base_path, style)
    regions = []

    if region == 'all':
        regions = ['GBR', 'NorthernAU']
    else:
        regions = [region]

    tif_files = []
    for reg in regions:
        region_path = os.path.join(input_path, reg)
        if os.path.exists(region_path):
            region_tifs = glob.glob(os.path.join(region_path, "*.tif"))
            if region_tifs:
                tif_files.extend(region_tifs)
                print(f"Found {len(region_tifs)} GeoTiff files in region: {reg}")
            else:
                print(f"No GeoTiff files found in region: {reg}")
        else:
            print(f"Warning: Region folder not found: {region_path}")

    if not tif_files:
        print("Error: No GeoTiff files were found. Exiting.")
        return

    # Prepare output path
    output_dir = os.path.join("working-data", "01b-S2-virtual-rasters")
    os.makedirs(output_dir, exist_ok=True)

    output_vrt = os.path.join(output_dir, f"{style}_{region}.vrt")

    # Create temporary file for input file list
    temp_file_list = os.path.join(output_dir, f"{style}_{region}_file_list.txt")
    with open(temp_file_list, 'w') as f:
        f.write("\n".join(tif_files))

    # Create virtual raster
    print(f"Creating virtual raster for style: {style}, region: {region}...")
    try:
        subprocess.run([
            "gdalbuildvrt", 
            "-input_file_list", temp_file_list, 
            output_vrt
        ], check=True)
        print(f"Virtual raster created successfully: {output_vrt}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual raster: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_list):
            os.remove(temp_file_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a virtual raster from Sentinel-2 GeoTiffs.")
    parser.add_argument("--style", type=str, default="15th_percentile", 
                        help="Image style to use. Default is '15th_percentile'. Valid values: 'low_tide_true_colour', 'low_tide_infrared'")
    parser.add_argument("--region", type=str, default="NorthernAU", 
                        help="Region to combine GeoTiffs into a single virtual raster. Options: 'NorthernAU', 'GBR', or 'all'")

    args = parser.parse_args()

    BASE_PATH = "in-data-3p/AU_AIMS_S2-comp"

    create_virtual_raster(BASE_PATH, args.style, args.region)
