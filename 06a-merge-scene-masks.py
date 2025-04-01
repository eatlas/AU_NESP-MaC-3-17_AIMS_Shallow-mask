import os
import argparse
import geopandas as gpd
import pandas as pd
import time

"""
This script merges and dissolves the shallow area reef masks shapefile from each of the 
individual Sentinel 2 tiles into a single shapefile for the whole region.

Performing the dissolving in a binary hierarchical manner is much faster than as a single dissolve, 
or as a linear dissolve (dissolving each shapefile into the result one at a time). For the
NorthernAU region the binary dissolve was 6x faster than the linear dissolve. The single dissolve
didn't complete after a couple of hours.

The merge processing time is highly variable and some merges take several hours.

The version and sigma are used to determine the filenames of the input and output files.

The following are the command lines used to reproduce the full dataset:
python 06a-merge-scene-masks.py --region NorthernAU --detectors VLow --version 1-1 --sigma 40
python 06a-merge-scene-masks.py --region NorthernAU --detectors Low --version 1-1 --sigma 40
python 06a-merge-scene-masks.py --region NorthernAU --detectors Medium --version 1-1 --sigma 40
python 06a-merge-scene-masks.py --region NorthernAU --detectors High --version 1-1 --sigma 40
python 06a-merge-scene-masks.py --region NorthernAU --detectors VHigh --version 1-1 --sigma 40

python 06a-merge-scene-masks.py --region GBR --detectors VLow --version 1-1 --sigma 40
python 06a-merge-scene-masks.py --region GBR --detectors Low --version 1-1 --sigma 40
python 06a-merge-scene-masks.py --region GBR --detectors Medium --version 1-1 --sigma 40
python 06a-merge-scene-masks.py --region GBR --detectors High --version 1-1 --sigma 40
python 06a-merge-scene-masks.py --region GBR --detectors VHigh --version 1-1 --sigma 40

"""

BASE_PATH = 'working-data/05-auto-mask'
LAND_MASK_BUFFER = 5   # Match with 04-create-water-_with_mask.py. Use to find water estimate files.
                           # Not used in the analysis, just for finding the input files.

# Parse command line arguments
parser = argparse.ArgumentParser(description="Merge and dissolve shapefiles with binary chunking.")
parser.add_argument('--region', type=str, default='NorthernAU', help="Region to process.")
parser.add_argument('--detectors', type=str, default='Habitat', help="Detector set to process.")
parser.add_argument('--version', type=str, default='1-1', help="Which version of the input data to process. Must match that from 05-create-shallow-and-reef-area-masks")

parser.add_argument('--sigma', type=str, default='40', 
        help="Size of the blurring used to create the water estimate. Used to find the water estimate files. Should match that used in 04-create-water-image_with_mask.py"
    )
args = parser.parse_args()

# Construct paths
run_name = f'{args.region}_V{args.version}_b{LAND_MASK_BUFFER}g{args.sigma}_{args.detectors}'
input_path = f"{BASE_PATH}/{run_name}/shapefiles"
output_dir = f"{BASE_PATH}/{run_name}/merge"
output_file = os.path.join(output_dir, f"{run_name}_merged.shp")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load all shapefiles and merge non-empty ones
file_paths = [
    os.path.join(input_path, file_name) 
    for file_name in os.listdir(input_path) 
    if file_name.endswith(".shp") and file_name.startswith("Reefs_and_Shallow_")
]

start_load_time = time.time()

# Filter out empty shapefiles. These happen for tiles where the area is just 
# open water

geo_dataframes = []
for file_path in file_paths:
    gdf = gpd.read_file(file_path)
    if not gdf.empty:
        print(f"Loaded {os.path.basename(file_path)}, appending")
        geo_dataframes.append(gdf)
    else:
        print(f"Loaded {os.path.basename(file_path)}, empty")


print(f"Loading time: {time.time() - start_load_time:.2f} s")

start_dissolve_time = time.time()
# Recursive binary dissolve function
def binary_dissolve(gdf_list):
    if len(gdf_list) == 1:
        return gdf_list[0]
    
    print(f"Dissolving {len(gdf_list)/2} pairs")
    start_dissolve_level_time = time.time()
    next_level = []
    for i in range(0, len(gdf_list), 2):
        if i + 1 < len(gdf_list):
            print(f"  Dissolving pair: {i} and {i+1}")
            merged_gdf = pd.concat([gdf_list[i], gdf_list[i+1]], ignore_index=True)
            dissolved = merged_gdf.dissolve()
            next_level.append(dissolved)
        else:
            # Odd element, just carry forward
            next_level.append(gdf_list[i])
    print(f"Dissolve level time: {time.time() - start_dissolve_level_time:.2f} s")
    # Recursively process next level
    return binary_dissolve(next_level)

# Perform binary dissolve
if geo_dataframes:
    print("Starting binary dissolve...")
    dissolved_final = binary_dissolve(geo_dataframes)
    print("Binary dissolve complete.")
    
    print(f"Dissolve time: {time.time() - start_dissolve_time:.2f} s")
    
    # Convert dissolved result to single-part polygons
    print("Converting to single-part polygons...")
    singlepart_final = dissolved_final.explode(index_parts=False).reset_index(drop=True)

    # Save the final output
    print(f"Saving dissolved result to {output_file}...")
    singlepart_final.to_file(output_file)
    print("Processing complete.")
else:
    print("No valid shapefiles found for processing.")
