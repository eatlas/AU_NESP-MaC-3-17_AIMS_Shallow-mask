import os
import sys
import argparse
import geopandas as gpd
import time

"""
This script combines the outputs from two sensitivity levels into one shapefile,
keeping the polygons from 'poly-sensitivity' level only if they overlap with the polygons
from 'keep-sensitivity' level.

This strategy helps to increase the boundary sensitivity (mapping deeper larger features)
without increasing the amount of false positives. 

The results of the script are saved to the dataset output directory. Assumes the script 06b
script was run successfully.

Commands used to reproduce the dataset:
python 06c-combine-sensitivities.py --poly-sensitivity VHigh --keep-sensitivity High --version 1-1
python 06c-combine-sensitivities.py --poly-sensitivity High --keep-sensitivity Medium --version 1-1
python 06c-combine-sensitivities.py --poly-sensitivity Medium --keep-sensitivity Low --version 1-1
python 06c-combine-sensitivities.py --poly-sensitivity Low --keep-sensitivity VLow --version 1-1
"""

BASE_PATH = 'working-data/06-merge'
OUTPUT_PATH = 'out-data'

# Parse command line arguments
parser = argparse.ArgumentParser(description="Combine outputs from two sensitivity levels into one shapefile.")
parser.add_argument('--poly-sensitivity', type=str, required=True, choices=['VLow','Low', 'Medium', 'High', 'VHigh'],
                    help="Sensitivity level to use for the polygons in the final dataset.")
parser.add_argument('--keep-sensitivity', type=str, required=True, choices=['VLow','Low', 'Medium', 'High', 'VHigh'],
                    help="Sensitivity level to use to determine inclusion of a polygon.")
parser.add_argument('--version', type=str, default='1', help="Version of the input data to process.")

args = parser.parse_args()

if args.poly_sensitivity == args.keep_sensitivity:
    print("poly-sensitivity and keep-sensitivity are the same. Nothing to be done. Exiting.")
    sys.exit()

input_dir = BASE_PATH

poly_sensitivity_file = f'{input_dir}/AU_NESP-MaC-3-17_AIMS_Shallow-mask_{args.poly_sensitivity}_V{args.version}.shp'
keep_sensitivity_file = f'{input_dir}/AU_NESP-MaC-3-17_AIMS_Shallow-mask_{args.keep_sensitivity}_V{args.version}.shp'

output_file = f'{OUTPUT_PATH}/AU_NESP-MaC-3-17_AIMS_Shallow-mask_{args.poly_sensitivity}-{args.keep_sensitivity}_V{args.version}.shp'

# Check if input files exist
if not os.path.exists(poly_sensitivity_file):
    print(f"Error: poly-sensitivity file not found at {poly_sensitivity_file}")
    sys.exit(1)

if not os.path.exists(keep_sensitivity_file):
    print(f"Error: keep-sensitivity file not found at {keep_sensitivity_file}")
    sys.exit(1)

# Load the shapefiles
print(f"Loading poly-sensitivity shapefile: {poly_sensitivity_file}")
start_time = time.time()
poly_gdf = gpd.read_file(poly_sensitivity_file)
print(f"Loaded poly-sensitivity shapefile in {time.time() - start_time:.2f} seconds.")

print(f"Loading keep-sensitivity shapefile: {keep_sensitivity_file}")
start_time = time.time()
keep_gdf = gpd.read_file(keep_sensitivity_file)
print(f"Loaded keep-sensitivity shapefile in {time.time() - start_time:.2f} seconds.")

# Ensure both GeoDataFrames have the same CRS
if poly_gdf.crs != keep_gdf.crs:
    print("CRS mismatch between shapefiles. Reprojecting keep-sensitivity shapefile to match poly-sensitivity shapefile CRS.")
    keep_gdf = keep_gdf.to_crs(poly_gdf.crs)

# Spatial join to find overlapping features
print("Performing spatial join to find overlapping features...")
start_time = time.time()

# Reset indices to ensure they are unique and can be used reliably
poly_gdf = poly_gdf.reset_index(drop=True)
keep_gdf = keep_gdf.reset_index(drop=True)

# Perform spatial join
joined_gdf = gpd.sjoin(
    poly_gdf, 
    keep_gdf[['geometry']], 
    how='inner', 
    predicate='intersects'
)

print(f"Spatial join completed in {time.time() - start_time:.2f} seconds.")

# Get unique features from poly_gdf that have overlaps
print("Selecting overlapping poly-sensitivity features...")
start_time = time.time()
overlapping_poly_gdf = poly_gdf.loc[joined_gdf.index.unique()].copy()
print(f"Selected overlapping features in {time.time() - start_time:.2f} seconds.")

# Save the result
if not os.path.exists(os.path.dirname(output_file)):
   os.makedirs(os.path.dirname(output_file))
print(f"Saving combined shapefile to {output_file}...")
start_time = time.time()
overlapping_poly_gdf.to_file(output_file)
print(f"Shapefile saved in {time.time() - start_time:.2f} seconds.")
print("Processing complete.")
