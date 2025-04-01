import os
import argparse
import geopandas as gpd
import time
import pandas as pd

"""
This script combines the merged shallow area reef masks shapefiles from NorthernAU and GBR 
regions into a single shapefile for the entire dataset covering all of tropical Australia.
This assumes that the 06a-merge-scene-masks.py has been previously run.

Example usage:

python 06b-merge-regions.py --detectors VLow --version 1-1 --sigma 40
python 06b-merge-regions.py --detectors Low --version 1-1 --sigma 40
python 06b-merge-regions.py --detectors Medium --version 1-1 --sigma 40
python 06b-merge-regions.py --detectors High --version 1-1 --sigma 40
python 06b-merge-regions.py --detectors VHigh --version 1-1 --sigma 40
"""

BASE_PATH = 'working-data/05-auto-mask'
OUTPUT_BASE_PATH = 'working-data/06-merge'

# Parse command line arguments
parser = argparse.ArgumentParser(description="Merge outputs from different regions into one shapefile.")
parser.add_argument('--detectors', type=str, default='Low', help="Detector set to process.")
parser.add_argument('--version', type=str, default='1', help="Version of the input data to process.")
parser.add_argument('--sigma', type=str, default='40', 
                    help="Size of the blurring used to create the water estimate. Should match that used in 04-create-water-image_with_mask.py.")
args = parser.parse_args()

# Construct paths
northernau_path = f"{BASE_PATH}/NorthernAU_V{args.version}_b5g{args.sigma}_{args.detectors}/merge/NorthernAU_V{args.version}_b5g{args.sigma}_{args.detectors}_merged.shp"
gbr_path = f"{BASE_PATH}/GBR_V{args.version}_b5g{args.sigma}_{args.detectors}/merge/GBR_V{args.version}_b5g{args.sigma}_{args.detectors}_merged.shp"

output_dir = f"{OUTPUT_BASE_PATH}"
output_file = os.path.join(output_dir, f"AU_NESP-MaC-3-17_AIMS_Shallow-mask_{args.detectors}_V{args.version}.shp")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the shapefiles
start_load_time = time.time()
gdfs = []

for region, path in [("NorthernAU", northernau_path), ("GBR", gbr_path)]:
    if os.path.exists(path):
        print(f"Loading {region} shapefile: {path}")
        gdf = gpd.read_file(path)
        gdfs.append(gdf)
    else:
        print(f"Warning: Shapefile for {region} not found at {path}")

print(f"Loading time: {time.time() - start_load_time:.2f} s")

# Merge and dissolve the shapefiles
if gdfs:
    print("Merging shapefiles...")
    merged_gdf = pd.concat(gdfs, ignore_index=True)
    print("Dissolving merged shapefile...")
    dissolved = merged_gdf.dissolve()

    print("Converting to single-part polygons...")
    singlepart_final = dissolved.explode(index_parts=False).reset_index(drop=True)

    # Save the final output
    print(f"Saving combined shapefile to {output_file}...")
    singlepart_final.to_file(output_file)
    print("Processing complete.")
else:
    print("No valid shapefiles found for merging.")
