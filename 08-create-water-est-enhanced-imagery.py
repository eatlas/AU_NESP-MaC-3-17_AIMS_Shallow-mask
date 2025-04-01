#!/usr/bin/env python3
"""
08-create-water-est-enhanced-imagery.py

This script processes Sentinel-2 satellite imagery by applying a water-based local contrast 
enhancement. It uses a corresponding "water estimate" image, which provides a smoothed water 
appearance, and subtracts a scaled version of it from the original satellite image to improve 
contrast over marine areas. Land areas are masked out using a land mask derived from a coastline 
shapefile. The script outputs enhanced images for further analysis.

Key Steps:
1. Input imagery:
   - Sentinel-2 imagery in EPSG:4326 (3-band RGB, 8-bit, 1-255 range, 0 = NoData).
   - Water estimate imagery (3-band RGB, 16-bit, scaled to 8-bit by dividing by 256), half resolution 
     of Sentinel-2 imagery.

2. Processing:
   - Upsample the water estimate image to match the Sentinel-2 image size (nearest-neighbor).
   - Slightly blur the Sentinel-2 image to reduce quantization effects (using a small Gaussian blur).
   - Apply a local contrast enhancement: 
     out = clip( sat - LOCAL_CONTRAST_FACTOR * water * BRIGHTNESS_SCALAR, 1, 255 )
   - Mask out land and NoData areas.

3. Parallel Processing:
   - The script supports splitting and indexing inputs for parallel runs.
   - Priority scenes can be specified to process them first or exclusively.

4. Output:
   - Saved in `working-data/08-s2_{style}_b{LAND_MASK_BUFFER}g{sigma}/{region}`
   - Filenames match the input Sentinel-2 imagery, replacing `_water2` with `_wdiff`.

Usage Example:
    python 08-create-water-est-enhanced-imagery.py --region NorthernAU --style 15th_percentile --sigma 40 --split 1 --index 0 --priority 51LXD 50KLB --justpriority True
    python 08-create-water-est-enhanced-imagery.py --region GBR --style 15th_percentile --sigma 40 --split 1 --index 0 --priority 54LYQ 55KDV --justpriority True
    
    These were used to create the published local contrast enhanced sentinel 2 imagery.
    python 08-create-water-est-enhanced-imagery.py --region NorthernAU --style 15th_percentile --sigma 40 --split 2 --index 0
    python 08-create-water-est-enhanced-imagery.py --region NorthernAU --style 15th_percentile --sigma 40 --split 2 --index 1
    python 08-create-water-est-enhanced-imagery.py --region GBR --style 15th_percentile --sigma 40 --split 2 --index 0
    python 08-create-water-est-enhanced-imagery.py --region GBR --style 15th_percentile --sigma 40 --split 2 --index 1
    
    This was used for generating a diagram comparing the original imagery, the water estimate and the difference.
    python 08-create-water-est-enhanced-imagery.py --region NorthernAU --style 15th_percentile --sigma 40 --split 1 --index 0 --priority 51LYE 51LZE 52LBK --justpriority True --contrast diff-map

Arguments:
    --region: Region being processed (default: NorthernAU) 
    --style: Image style (default: 15th_percentile) 15th_percentile, low_tide_true_colour, low_tide_infrared
    --sigma: Gaussian sigma for directory naming and water estimation (default: 40). This is for file name matching on water estimate files.
    --priority: List of scene codes to prioritize. Sentinel 2 scene codes such as 51LXD
    --justpriority: If True, only process the priority scenes (default: False).
    --split: Number of subsets for parallel processing (default: 1)
    --index: Index of the subset to process (default: 0)
    --keep_land: If set, the script will keep the original land pixels from the input satellite 
    image in the final output, rather than masking them out as NoData. Water areas still receive the 
    contrast enhancement. Land pixels remain unchanged from the original imagery. 
    BUG NOTE: This implementation is flawed because the image is clipped to the no data area of the 
    water estimate image, since it has a much greater no data border around the image. It also has a 
    no data value for most of the land area except 1.8 km from the coastline. To fix this bug we would 
    need to potentially recalculate the outer clipping border without the land mask.
    --contrast: Amount of contrast to apply to the image. This is a set of fixed configurations
    for different purposes.

Make sure the required directory structures exist, input data is available, and the environment 
has the necessary Python packages installed (rasterio, geopandas, cv2, numpy).
"""

import argparse
import os
import sys
import glob
import numpy as np
import geopandas as gpd
import cv2
import sys

import rasterio
from imageutil import prepare_sentinel_image_list, extract_scene_code, save_image_uint, TimePrint, rasterize_shapefile, read_image

# Constants
S2_IMAGE_PARENT_PATH = 'in-data-3p/AU_AIMS_S2-comp'
LAND_MASK_BUFFER = 5

CONTRAST_OPTIONS = {
    "normal": {
        "brightness_scalar": 2.5,
        "local_contrast_factor": 0.8,
        "gamma": 1.4,
        "white_points": [230.0, 255.0, 210.0],
        "output_base_path": 'working-data'
    },
    # Intended for generating images to highlight the difference between the original imagery and the 
    # water imagery. Intended to be used on select imagery to create demonstration figures.
    "diff-map": {
        "brightness_scalar": 3,
        "local_contrast_factor": 0.9,
        "gamma": 1.4,
        "white_points": [230.0, 255.0, 210.0],
        "output_base_path": 'working-data/08-diff-map'
    }
}

LAND_MASK_RASTER_BASE_PATH = 'working-data/04-land_mask_rasters'
LAND_MASK_SHP = 'data/in-3p/AU_AIMS_Coastline_50k_2024/Split/AU_NESP-MaC-3-17_AIMS_Aus-Coastline-50k_2024_V1-1_split.shp'

GAUSSIAN_SIGMA = 0.3
GAUSSIAN_KERNEL_SIZE = (3, 3)


def main():
    parser = argparse.ArgumentParser(description="Create water estimate enhanced imagery.")
    parser.add_argument('--region', type=str, default='NorthernAU', help="Region name (e.g. NorthernAU, GBR)")
    parser.add_argument('--style', type=str, default='15th_percentile', help="Imagery style (e.g. low_tide_true_colour)")
    parser.add_argument('--sigma', type=int, default=40, help="Gaussian sigma used in water estimation directory structure.")
    parser.add_argument('--priority', type=str, nargs='+', default=[], 
                        help="List of scene codes to prioritize for processing. For example --priority 50KLB 49JGM")
    parser.add_argument('--justpriority', type=bool, default=False, 
                        help="If True only process the scenes listed in the priority list.")
    parser.add_argument('--split', type=int, default=1, 
                        help="Number of subsets to split the processing into. Default is 1.")
    parser.add_argument('--index', type=int, default=0, 
                        help="Index of the subset to process (starting from 0). Default is 0.")
    parser.add_argument('--keep_land', action='store_true',
                        help="If set, retain the original land pixels in the final image rather than setting them to NoData.")
    parser.add_argument('--contrast', type=str, default='normal', choices=['normal', 'diff-map'],
                    help="Style of the contrast enhancement to create. normal, diff-map")
    

    args = parser.parse_args()

    if args.justpriority and len(args.priority) == 0:
        parser.error("justpriority is True but no priority scenes provided.")

    # Select the appropriate contrast style set if valid
    contrast_style = CONTRAST_OPTIONS[args.contrast]
    
    # Input directories
    s2_image_base_path = os.path.join(S2_IMAGE_PARENT_PATH, args.style, args.region)
    water_estimate_base_path = f'working-data/04-s2_{args.style}_b{LAND_MASK_BUFFER}g{args.sigma}/{args.region}'

    # Output directory
    output_dir = f'{contrast_style["output_base_path"]}/08-s2_{args.style}_b{LAND_MASK_BUFFER}g{args.sigma}/{args.region}'
    os.makedirs(output_dir, exist_ok=True)

    t = TimePrint()

    # Prepare list of files to process
    t.print("Preparing scene list...")
    tiff_files = prepare_sentinel_image_list(
        base_path=s2_image_base_path,
        split=args.split,
        index=args.index,
        priority_scenes=args.priority,
        justpriority=args.justpriority
    )

    total_files = len(tiff_files)
    t.print(f"Found {total_files} files to process.")
    
    if total_files == 0:
        print("No files to process, so exiting")
        sys.exit()

    # Load land mask shapefile once
    t.print("Loading land mask shapefile...")
    landmask_shapes = gpd.read_file(LAND_MASK_SHP)

    def find_water_estimate_file(sat_file):
        base_name = os.path.basename(sat_file)
        name_no_ext = os.path.splitext(base_name)[0]
        water_name = name_no_ext + "_water2.tif"
        water_path = os.path.join(water_estimate_base_path, water_name)
        return water_path

    for i, tiff_file in enumerate(tiff_files, start=1):
        filename = os.path.basename(tiff_file)

        t.print("------------------------------------------------------------")
        t.print(f"Processing {i} of {total_files}: {filename}")

        scene_code = extract_scene_code(filename)

        # Derive output filename
        name_no_ext = os.path.splitext(filename)[0]
        output_name = name_no_ext + "_wdiff.tif"
        output_path = os.path.join(output_dir, output_name)

        # If output already exists, skip
        if os.path.exists(output_path):
            t.print(f"Skipping, output {output_path} already exists.")
            continue

        # Find water estimate file
        water_file = find_water_estimate_file(tiff_file)
        if not os.path.exists(water_file):
            t.print(f"WARNING: Skipping, water estimate file not found for {filename}.")
            continue

        # Read satellite image (original)
        sat_image, sat_profile = read_image(tiff_file)
        # Store a copy of the original imagery if we are keeping land
        if args.keep_land:
            original_sat_image = sat_image.copy()

        sat_image = sat_image.astype(np.float32)

        # In-place Gaussian blur
        for band_i in range(sat_image.shape[0]):
            sat_image[band_i, :, :] = cv2.GaussianBlur(sat_image[band_i, :, :], GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

        # Read water estimate image (16-bit)
        base_water_image, water_profile = read_image(water_file)
        base_water_image = base_water_image.astype(np.float32) / 256.0

        sat_height, sat_width = sat_profile['height'], sat_profile['width']
        water_height, water_width = water_profile['height'], water_profile['width']

        # If different resolution, upsample water image
        if (water_height != sat_height) or (water_width != sat_width):
            t.print("      Upsampling the water estimate to full resolution")
            water_image = np.zeros((3, sat_height, sat_width), dtype=np.float32)
            for band_i in range(base_water_image.shape[0]):
                water_image[band_i] = cv2.resize(base_water_image[band_i], (sat_width, sat_height), interpolation=cv2.INTER_NEAREST)
            del base_water_image
        else:
            water_image = base_water_image
            del base_water_image

        # Create nodata mask from water estimate (1 where valid)
        nodata_mask = np.all(water_image != 0, axis=0).astype(np.uint8)

        # In-place difference and scaling
        sat_image -= (contrast_style["local_contrast_factor"] * water_image)
        del water_image
        sat_image *= contrast_style["brightness_scalar"]
        np.clip(sat_image, 1, 255, out=sat_image)

        # Create or load land mask
        land_mask_raster_dir = os.path.join(LAND_MASK_RASTER_BASE_PATH, args.region)
        os.makedirs(land_mask_raster_dir, exist_ok=True)

        LAND_MASK_PREFIX = 'AU_AIMS_Coastline_50k_2024'
        land_mask_path = os.path.join(land_mask_raster_dir, f"{LAND_MASK_PREFIX}_{scene_code}_land_mask.tif")

        if not os.path.exists(land_mask_path):
            t.print(f"Land mask not found {land_mask_path}, Creating ...")
            rasterize_shapefile(
                shapefile_polygons=landmask_shapes,
                raster_profile=sat_profile,
                output_raster_path=land_mask_path,
            )
            t.print(f"    Rasterized shapefile saved to {land_mask_path}")

        t.print(f"    Reading land mask: {os.path.basename(land_mask_path)}")
        land_mask, _ = read_image(land_mask_path)
        # Convert land mask from 0-water, 255-land to binary water=1, land=0
        binary_land_mask = (land_mask[0] == 0).astype(np.uint8)
        del land_mask

        # Combine masks
        # final_mask = mask where we have valid data. Currently final_mask is nodata_mask * binary_land_mask
        # If keep_land is False: we mask out land (thus final_mask = nodata_mask * binary_land_mask)
        # If keep_land is True: we want to show land as original. Only no-data (water estimate invalid) gets masked out.
        if args.keep_land:
            final_mask = nodata_mask
        else:
            final_mask = nodata_mask * binary_land_mask

        # Prepare for color adjustments
        sat_image /= 255.0

        # Apply white balancing and gamma correction on sat_image
        black_point = 1.0
        for band_idx, white_point in enumerate(contrast_style["white_points"]):
            sat_image[band_idx] = np.clip((sat_image[band_idx] - black_point/255.0) / ((white_point - black_point)/255.0), 0, 1)

        sat_image **= (1/contrast_style["gamma"])

        # Scale back to 1–255 and convert to uint8
        sat_image = (sat_image * 255).clip(1, 255).astype(np.uint8)

        # If keeping land, overwrite land pixels with original data
        # binary_land_mask = 1 for water, 0 for land. Land pixels: where binary_land_mask == 0
        if args.keep_land:
            # Ensure original and enhanced have the same dtype for easy blending
            # original_sat_image was kept in original form (uint8)
            # Land pixels (binary_land_mask == 0)
            land_pixels = (binary_land_mask == 0)
            # Overwrite land area in sat_image with original_sat_image
            sat_image[:, land_pixels] = original_sat_image[:, land_pixels]
            del original_sat_image

        # Update profile for output
        out_profile = sat_profile.copy()
        out_profile.update(count=3, dtype='uint8', compress='lzw')

        t.print(f"Saving output to {output_path}")
        save_image_uint(
            image=sat_image,
            profile=out_profile,
            output_path=output_path,
            mask=final_mask,
            clip=True,
            bit_depth=8
        )

        # Free large arrays after each scene
        del sat_image, final_mask

    t.print("Processing complete.")


if __name__ == "__main__":
    main()