#!/usr/bin/env python3
"""
Script Name: 07d-compare-reef-masks.py

Description:
This script compares two polygon shapefile datasets corresponding to shallow marine areas and reefs:
1. A manual reference reef map (rough-reef-mask).
2. An automated mapping of the same location (auto-mask).

The script identifies platform reefs (reefs not touching the mainland or islands) in both datasets,
determines the true positives, false positives, and false negatives, and reports the statistics.

Usage:
python 07d-compare-reef-masks.py [--sensitivity DETECTORS] [--version VERSION]

Arguments:
--sensitivity: Sensitivity levels from 06c (default: 'High-Medium'). 
--version:   Version of the input data to process (default: '1').

Example:
python 07d-compare-reef-masks.py --sensitivity VHigh-High --version 1
python 07d-compare-reef-masks.py --sensitivity High-Medium --version 1
python 07d-compare-reef-masks.py --sensitivity Medium-Low --version 1
"""

import argparse
import os
import sys
import geopandas as gpd
import pandas as pd

SAVE_DEBUG_SHP = False  # If true save rough_platform_reefs and auto_platform_reefs

def main():
    parser = argparse.ArgumentParser(description='Compare Shallow-mask and rough-reef-mask shapefiles to identify false positives and negatives in platform reefs.')
    #parser.add_argument('--region', type=str, default='NorthernAU', help="Region to process.")
    parser.add_argument('--sensitivity', type=str, default='Medium', 
        help="Selects which of the detector set to process. Possible options 'Low', 'Medium', 'High'"
    )
    parser.add_argument('--version', type=str, default='1', help="Which version of the input data to process. Must match that from 05-create-shallow-and-reef-area-masks.")
    #parser.add_argument('--sigma', type=str, default='40', 
    #        help="Size of the blurring used to create the water estimate. Used to find the water estimate files. Should match that used in #04-create-water-image_with_mask.py."
    #    )
    args = parser.parse_args()

    # Define file paths
    #rough_reef_mask_path = 'in-data/AU_Rough-reef-shallow-mask/AU_AIMS_NESP-MaC-3-17_Rough-reef-shallow-mask_Base.shp'
    rough_reef_mask_path = 'working-data/03-rough-reef-mask_poly/AU_Rough-reef-shallow-mask-with-GBR.shp'
    land_mask_shp_path = 'in-data/AU_AIMS_Coastline_50k_2024/Split/AU_NESP-MaC-3-17_AIMS_Aus-Coastline-50k_2024_V1-1_split.shp'
    auto_mask_path = f'out-data/V{args.version}/AU_NESP-MaC-3-17_AIMS_Shallow-mask_{args.sensitivity}_V{args.version}.shp'
    #auto_mask_path = f'working-data/deprecate/06-merge/AU_NESP-MaC-3-17_AIMS_Shallow-mask_{args.sensitivity}_V{args.version}.shp'
    
    rough_platform_reefs_path = "working-data/07-qaqc/rough_platform_reefs.shp"
    
    auto_platform_reefs_with_fp_path = f"working-data/07-qaqc/Shallow-mask_V{args.version}_{args.sensitivity}_auto-platform.shp"
    fpfn_combined_path = f"working-data/07-qaqc/Shallow-mask_V{args.version}_{args.sensitivity}_fpfn.shp"
    
    #rough_reef_mask_path = 'in-data/AU_Rough-reef-shallow-mask/AU_AIMS_NESP-MaC-3-17_Rough-reef-shallow-mask_Base.shp'
    #auto_mask_path = 'in-data/AU_Rough-reef-shallow-mask/AU_AIMS_NESP-MaC-3-17_Rough-reef-shallow-mask_Test.shp'

    # Check if files exist
    for path in [rough_reef_mask_path, land_mask_shp_path, auto_mask_path]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)

    print("Loading shapefiles...")
    # Load shapefiles
    print("  rough-reef-mask...")
    rough_reef_mask = gpd.read_file(rough_reef_mask_path)
    print("  land-mask...")
    land_mask = gpd.read_file(land_mask_shp_path)
    print("  auto-mask...")
    auto_mask = gpd.read_file(auto_mask_path)

    # Ensure CRS are the same
    if rough_reef_mask.crs != land_mask.crs:
        land_mask = land_mask.to_crs(rough_reef_mask.crs)
    if auto_mask.crs != rough_reef_mask.crs:
        auto_mask = auto_mask.to_crs(rough_reef_mask.crs)


    print("Identifying rough-reef-mask features that touch the coastline...")
    # Reset indices to ensure uniqueness. This sets the index to [0, 1, .. n-1]
    rough_reef_mask = rough_reef_mask.reset_index(drop=True)
    rough_reef_mask['rrm_index'] = rough_reef_mask.index
    
    land_mask = land_mask.reset_index(drop=True)
    land_mask['lm_index'] = land_mask.index

    # Perform spatial join to find rough_reef_mask features that intersect land_mask
    # We want to exclude features that are touching the land so that we focus on
    # platform reefs. Only features that intersect are retained in rough_coastal_reefs
    print("  Spatial join rough-reef-mask and land-mask...")
    rough_coastal_reefs = gpd.sjoin(rough_reef_mask, land_mask, how='inner', predicate='intersects')
    # Find the IDs of the rough reefs that are touching the coast. Remove duplicates.
    rough_coastal_reef_indices = rough_coastal_reefs['rrm_index'].unique()

    # Identify rough_reef_mask features not touching coastline
    print("  Identify reefs not touching coastline...")
    # Select rough_reef_mask rows, where they are not in the list of ones touching the coast. 
    rough_platform_reefs = rough_reef_mask[~rough_reef_mask['rrm_index'].isin(rough_coastal_reef_indices)].copy()
    


    print("Creating exclusion zone from coastline and rough-reef-mask coastal features...")
    print("  combine rough-reef-mask and land-mask features...")
    # Combine land_mask and rough_coastal_reefs into a single GeoDataFrame
    # Merge rough_reef_mask elements that are touching the coastline to the lask mask polygons.
    exclusion_geoms = pd.concat([land_mask[['geometry']], rough_reef_mask.loc[rough_reef_mask['rrm_index'].isin(rough_coastal_reef_indices), ['geometry']]], ignore_index=True)
    exclusion_geoms = gpd.GeoDataFrame(exclusion_geoms, crs=rough_reef_mask.crs)

    print("Identifying auto-mask features that touch the exclusion zone...")
    print("  auto-mask index creation...")
    # Ensure auto_mask index is unique
    auto_mask = auto_mask.reset_index(drop=True)
    auto_mask['am_index'] = auto_mask.index

    # Perform spatial join between auto_mask and exclusion_geoms
    print("  Spatial join between auto-mask and exclusion geometry")
    # Retain auto-mask features that overlap the exclusion geometry (land and coastal rough reef mask features)
    auto_exclusion_join = gpd.sjoin(auto_mask, exclusion_geoms, how='inner', predicate='intersects')
    
    # IDs of the auto-mask features that need to be excluded
    auto_exclusion_indices = auto_exclusion_join['am_index'].unique()

    # Identify auto_mask features not touching exclusion zone
    print("  Identify auto mask features not touching exclusion zone")
    auto_platform_reefs = auto_mask[~auto_mask['am_index'].isin(auto_exclusion_indices)].copy()
    


    print("Comparing auto-mask platform reefs with rough-reef-mask platform reefs...")
    # Reset indices for platform reefs
    auto_platform_reefs = auto_platform_reefs.reset_index(drop=True)
    auto_platform_reefs['apr_index'] = auto_platform_reefs.index
    rough_platform_reefs = rough_platform_reefs.reset_index(drop=True)
    rough_platform_reefs['rpr_index'] = rough_platform_reefs.index

    print("Looking for true positives")
    # Spatial join to find true positives (auto_platform_reefs that intersect rough_platform_reefs)
    true_positives_join = gpd.sjoin(auto_platform_reefs, rough_platform_reefs, how='inner', predicate='intersects')
    true_positive_indices = true_positives_join['apr_index'].unique()

    # True positives
    # This identifies the true positives that were detected in the auto_platform_reefs. The problem
    # is that auto_platform_reefs might have fragmented features where one of the rough_platform_reefs
    # is represented by many small polygons in the auto_platform_reefs. We therefore can't use this
    # metric to compare with the number of rough_platform_reefs as the counts will not align.
    true_positives = auto_platform_reefs.loc[auto_platform_reefs['apr_index'].isin(true_positive_indices)]

    print("Looking for true positives from rough_platform_reefs perspective...")
    # Spatial join to find rough_platform_reefs that intersect auto_platform_reefs
    rough_true_positives_join = gpd.sjoin(rough_platform_reefs, auto_platform_reefs, how='inner', predicate='intersects')
    rough_true_positive_indices = rough_true_positives_join['rpr_index'].unique()

    # True positives from the rough_platform_reefs perspective
    rough_true_positives = rough_platform_reefs.loc[rough_platform_reefs['rpr_index'].isin(rough_true_positive_indices)]

    # Count of true positives from rough_platform_reefs perspective
    num_rough_true_positives = len(rough_true_positives)


    # False positives: auto_platform_reefs not in true positives
    print("Looking for false positives")
    false_positives = auto_platform_reefs[~auto_platform_reefs['apr_index'].isin(true_positive_indices)]

    # Now, for rough_platform_reefs, find those that are detected (intersecting with auto_platform_reefs)
    # Look for false negatives. Find the reefs that overlap between between the rough_platforms 
    # and the auto_platforms. 
    detected_reefs_join = gpd.sjoin(rough_platform_reefs, auto_platform_reefs, how='inner', predicate='intersects')
    # IDs of all the reefs that overlap. The missing reefs won't be in this list.
    detected_reef_indices = detected_reefs_join['rpr_index'].unique()

    # Detected reefs
    detected_reefs = rough_platform_reefs.loc[rough_platform_reefs['rpr_index'].isin(detected_reef_indices)]
    
    print("Looking for false negatives")
    # False negatives: rough_platform_reefs not in detected_reefs
    false_negatives = rough_platform_reefs[~rough_platform_reefs['rpr_index'].isin(detected_reef_indices)]
    
    # Add a column indicating false negatives (1 = False Negative, 0 = Not False Negative)
    rough_platform_reefs['is_false_negative'] = 0
    rough_platform_reefs.loc[rough_platform_reefs['rpr_index'].isin(false_negatives['rpr_index']), 'is_false_negative'] = 1


    # Combine false positives and false negatives into a single GeoDataFrame
    print("Combining false positives and false negatives into a single GeoDataFrame...")
    false_positives['type'] = 'false_positive'  # Add a type column to differentiate
    false_negatives['type'] = 'false_negative'  # Add a type column to differentiate

    # Select relevant columns and combine
    fpfn_combined = pd.concat([false_positives[['geometry', 'type']], false_negatives[['geometry', 'type']]], ignore_index=True)
    fpfn_combined = gpd.GeoDataFrame(fpfn_combined, crs=rough_platform_reefs.crs)

    os.makedirs(os.path.dirname(fpfn_combined_path), exist_ok=True)
    # Save the combined false positives and negatives shapefile
    print(f"Saving combined false positives and false negatives to {fpfn_combined_path}...")
    fpfn_combined.to_file(fpfn_combined_path, driver="ESRI Shapefile")


    # Save the rough_platform_reefs to a shapefile
    if SAVE_DEBUG_SHP:
        print(f"Saving rough platform reefs to {rough_platform_reefs_path}...")
        os.makedirs(os.path.dirname(rough_platform_reefs_path), exist_ok=True)
        rough_platform_reefs.to_file(rough_platform_reefs_path, driver="ESRI Shapefile")
    
    # Add a column indicating false positives (1 = False Positive, 0 = Not False Positive)
    auto_platform_reefs['is_false_positive'] = 0
    auto_platform_reefs.loc[auto_platform_reefs['apr_index'].isin(false_positives['apr_index']), 'is_false_positive'] = 1

    # Save the updated auto_platform_reefs with false positives
    
    if SAVE_DEBUG_SHP:
        print(f"Saving auto platform reefs with false positives to {auto_platform_reefs_with_fp_path}...")
        os.makedirs(os.path.dirname(auto_platform_reefs_with_fp_path), exist_ok=True)
        auto_platform_reefs.to_file(auto_platform_reefs_with_fp_path, driver="ESRI Shapefile")

    # Calculate statistics
    total_rough_platform = len(rough_platform_reefs)
    total_auto_platform = len(auto_platform_reefs)
    num_true_positives = len(true_positives)
    num_false_positives = len(false_positives)
    num_false_negatives = len(false_negatives)

    
    true_positive_rate = (num_true_positives / total_rough_platform) * 100 if total_rough_platform > 0 else 0
    false_positive_rate = (num_false_positives / total_auto_platform) * 100 if total_auto_platform > 0 else 0

    false_negative_rate = (num_false_negatives / total_rough_platform) * 100 if total_rough_platform > 0 else 0

    # Calculate statistics for rough_platform_reefs perspective
    true_positive_rate_rough = (num_rough_true_positives / total_rough_platform) * 100 if total_rough_platform > 0 else 0

    print(f"Analysis Complete for {args.sensitivity}.\n")
    # Print statistics for rough_platform_reefs
    print("Platform Reefs in Rough-Reef-Mask:")
    print(f"  Total: {total_rough_platform}")
    print(f"  Detected (True Positives): {num_rough_true_positives} ({true_positive_rate_rough:.2f}%)")
    print(f"  Missed (False Negatives): {num_false_negatives} ({false_negative_rate:.2f}%)\n")

    # Existing statistics for auto_platform_reefs remain unchanged
    print("Platform Reefs in Auto-Mask:")
    print(f"  Total: {total_auto_platform}")
    print(f"  Correctly Detected (True Positives): {num_true_positives} ({(num_true_positives / total_auto_platform * 100):.2f}%)")
    print(f"  False Positives: {num_false_positives} ({false_positive_rate:.2f}%)")

if __name__ == "__main__":
    main()
