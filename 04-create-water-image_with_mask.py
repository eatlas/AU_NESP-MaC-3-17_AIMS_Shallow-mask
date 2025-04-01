"""
This script creates an estimate of the water colour in a satellite image, aiming
to create an image of what the satellite imagery would look like without reefs
or shallow areas. This water estimate can be used to assist in automated mapping
of reefs as it allows masking and compensations based on the water conditions 
that the reefs are in. For example if an area is in highly turbid water then
algorithms that assume clear water will not work. A better approach is switching
to an algorithm that can detect just the super shallow features.

Estimating the water colour is inherently ambiguous. In any one pixel location
the signal detected by the satellite is made up from multiple signals, including
atmospheric reflections, surface reflectance (sunglint), reflection in the water
column from scattering and reflectance from the bottom. This algorithm assumes that
atmospheric and sunglint have already been handled in the preparation of the input
imagery. This algorithm aims to separate the water column from the benthic reflectance.



It should be stated clearly that this algorithm is imperfect and does not directly
estimate just the signal from the water column, it mearly amplifies it in many situations
that are important for automated reef mapping.

This algorithm works on the principle that the water over a small reef feature is
going to be spatially correlated with the surrounding deeper water. The deeper the
water the stronger the water column signal. In turbid environments the water does
not need to be very deep for the water column signal to be the dominant signal.
If we could just replace the reef top with an interpolation of the colour of the
surrounding water then this would be a strong proxy for the water that is over the
top of that reef.

If we had a mask of all the reef and shallow areas then we could, infill these
areas with an interpolation of the water surrounding them. By looking at a difference
between this water estimate and the satellite imagery the reef features would be
easily identified and mapped. This is clearly a circular process. If we had a map
of the reefs we could estimate the water and make a map of the reefs.

To break this circular dependancy we rely on the assumption that water conditions
change slowly in space and thus are highly spatially correlated. This means that
if we apply a blurring to imagery, then in water areas where there are no shallow
features, then the water column signal should be largely unaffected, up to the point
where the blur radius exceeds the spatial correlation distance. In practice there
will be fine scale structure in the water colour, particularly in turbid areas,
however it is still highly spatially correlated and so blurring over even large distances
of hundreds of metres will retain most of the signal. 

Heavily blurring (say 1 km) the imagery over small isolated reef features (100 m) with smear
out their brightness values over blurred radius area. In this example the resulting
image will be 99% water colour and 1% reef colour. This means the reef will largely
disappear from the imagery. Where there is a problem is if the reef is large or very bright.
If that reef is many times brighter than the surrounding water then 1% might be enough 
to brighten the surrounding blurred 1 km that the water estimate is tainted with the 
reflected signal from the reef. Alternatively if the reef is 1 km across then the blurring
will have little effect in surpressing the reefs reflected signal. 

To overcome these problems we use an inital rough masking to remove large shallow areas
from the first stage of the estimation problem. Masking out these problem areas, even 
if that mask is highly rough and imperfect, largely removes their influence from the
water estimate. 

We first mask out the large shallow areas and perform a large blurring estimate over the 
image. This infills an estimate over the masked out areas. We then perform the estimate
again, replacing the masked shallow areas with the water estimate established in the first stage.

The second stage blends the rough masked areas (which are now replaced with the initial water estimate) with the surrounding areas, and the blurring removes the signal from the small shallow areas that were not masked out.

The following corresponds to the command line calls that were used to prepare the final dataset:

Version 18
conda activate reef_maps2

python 04-create-water-image_with_mask.py --split 1 --index 0 --style low_tide_true_colour --sigma 40 --region NorthernAU
python 04-create-water-image_with_mask.py --split 2 --index 0 --style 15th_percentile --sigma 40 --region NorthernAU
python 04-create-water-image_with_mask.py --split 2 --index 1 --style 15th_percentile --sigma 40 --region NorthernAU

python 04-create-water-image_with_mask.py --split 1 --index 0 --style low_tide_true_colour --sigma 40 --region GBR
python 04-create-water-image_with_mask.py --split 1 --index 0 --style 15th_percentile --sigma 40 --region GBR


Single Tile Demo Run: Takes approximately 2.5 min per image.
This processes just the one tile, even if more were downloaded.
python 04-create-water-image_with_mask.py --style low_tide_true_colour --sigma 40 --region GBR --priority 55KDA --justpriority True
python 04-create-water-image_with_mask.py --style 15th_percentile --sigma 40 --region GBR --priority 55KDA --justpriority True

"""
import os
import sys
import glob
import numpy as np
import argparse
import rasterio
import string
from rasterio.features import rasterize
import math
import geopandas as gpd
from scipy.ndimage import binary_dilation, binary_erosion
import cv2
from skimage.morphology import disk
from scipy.ndimage import zoom
from imageutil import (
    read_image,  # Use this for reading both the original image and the masks
    prepare_sentinel_image_list,
    gaussian_filter_nan_cv2,
    save_image_uint,
    debug_image,
    TimePrint,
    extract_scene_code,
    get_image_profile,
    create_clipping_mask_cv2,
    rasterize_shapefile
)
# Number of the script, used to prepend folders of working data created by this script
SCRIPT_NUMBER = '04'

# Parent directory of the sentinel 2 image composites. Assumes the images are organised
# into style/regions as sub-folders.
S2_IMAGE_PARENT_PATH = 'data/in-3p/AU_AIMS_S2-comp' # Default if downloaded with 01-download-sentinel2.py

# Location of the coastline dataset
LAND_MASK_SHP = 'data/in-3p/AU_AIMS_Coastline_50k_2024/Split/AU_NESP-MaC-3-17_AIMS_Aus-Coastline-50k_2024_V1-1_split.shp'

# Start of the raster version filename of the land mask
LAND_MASK_PREFIX = 'AU_AIMS_Coastline_50k_2024'

# Cache of already rendered land masks.
LAND_MASK_RASTER_BASE_PATH = f'working-data/{SCRIPT_NUMBER}-land_mask_rasters'

# Location of the manual rough reef mask
REEF_MASK_SHP = 'working-data/03-rough-reef-mask_poly/AU_Rough-reef-shallow-mask_87hr_GBR.shp'

# cache of rendered masks 
REEF_MASK_RASTER_BASE_PATH = f'working-data/{SCRIPT_NUMBER}-rough-reef-mask_rasters'

# Filename prefix for rastered version of reef mask
REEF_MASK_PREFIX = 'AU_Rough-reef-mask'
REEF_MASK_BUFFER = 0   # (pixels) Buffer applied to reef mask prior to water blurring

FINAL_LAND_CLIP_BUFFER = -5 # Buffer to apply to the land areas 

SAVE_STAGE1_BLUR = False  # Whether to save the stage one blurred image for debugging.
SAVE_STAGE1_INTERMED = False  # Whether to save the intermediate image. It is only needed to
                            # debugging and uses considerabe storage.
SAVE_WEIGHTS_OUTPUT = False  # Save the weights used in the image blurring. This is only needed for debugging.

# Apply a buffer to the land mask to remove the very shallow water
# neighbouring the land area that is likely to have a strong benthic
# reflected signal, and thus not representative of the water.
# If there was perfect shallow water masking then the land mask
# would not be needed. This land mask helps in cases where the shallow
# water mask is missing a section.
# One trade off with this buffer is that if it is made too large then
# narrow channels, such as rivers, will be fully collapsed, leaving
# no pixels left for the interpolation to operate off. As a result
# the river will receive the closest oceanic water colour, up to
# the distance associated with the stage 2 gaussian sigma * ~4.
# This mean that the water estimate in the rivers will be wrong and significantly
# to the satellite imagery. As a result this difference will then be falsely flagged
# as a shallow portion (as the area is significantly different to the water
# estimate).
LAND_MASK_BUFFER = 20   # Buffer applied to land mask prior to water blurring

LAND_MASK_BUFFER = 10   # Buffer applied to land mask prior to water blurring
                        # This will buffer 100 m, collapsing rivers less than
                        # 200 m across. The rivers are not the focus of this
                        # study and so we just need to ensure we trim small
                        # rivers out of the final mask as the results will not
                        # be valid.
LAND_MASK_BUFFER = 5   # V17: Shrink the land buffer now that masking is better

# Sigma for first stage Gaussian filter. This is what is used to infill the 
# rough manual masking. The maximum infill distance is approximately 
# STAGE1_GAUSS_SIGMA (pixels) * 10 (m / pixels) * 4.
STAGE1_GAUSS_SIGMA = 160  



def create_water_image(sat_image, sat_image_profile, land_reef_mask, gauss_sigma):
    """
    Creates an estimate of the water color in the image by masking out the land and reef portions (land_reef_mask),
    then blurring the image using a gaussian filter that interpolates over the masked land and reef areas.
    The water image is created at half resolution of the original satellite image to save on storage space and
    to speed up calculation time.
    
    Args:
        sat_image (numpy.ndarray): Satellite image to process. Expected shape (3, height, width)
        sat_image_profile (rasterio profile dict): Geospatial metadata about the image.
        land_reef_mask (numpy.ndarray): Binary mask of the land area. Expected shape (1, height, width).
        gauss_sigma (float): Sigma value for the Gaussian filter. Should be > 4.
    
    Returns:
        numpy.ndarray: The blurred water image.
        dict: Updated image profile reflecting the downsampled resolution.
    """
    t = TimePrint()
    t.print("  Calculating water estimate")
    
    # Ensure the water_image_float is a float array
    water_image_float = np.copy(sat_image).astype(np.float32)
    
    # Apply border mask
    t.print("      Applying image border mask")
    for i in range(water_image_float.shape[0]):  # Iterate over each channel
        water_image_float[i, water_image_float[0] == 0] = np.nan  # Set border areas to NaN
        
    # Apply the land and reef mask (non-water areas)
    t.print("      Applying land and reef mask")
    for i in range(water_image_float.shape[0]):  # Iterate over each channel
        water_image_float[i, land_reef_mask[0] == 1] = np.nan  # Set land and reef areas to NaN
    
    # Apply small Gaussian blur (gauss_sigma_small) at full resolution
    gauss_sigma_small = 2
    t.print(f"      Applying small Gaussian filter (sigma={gauss_sigma_small})")
    water_image = np.empty_like(water_image_float)
    for i in range(3):  # Assuming RGB channels
        t.print(f"        Filtering channel {i}")
        water_image[i], _ = gaussian_filter_nan_cv2(water_image_float[i], sigma=gauss_sigma_small)
        #water_image[i] = water_image_float[i]
    
    # Downsample the image to half the resolution
    t.print("      Downsampling the image to half resolution")
    water_image = zoom(water_image, (1, 0.5, 0.5), order=1)  # Downsample by a factor of 2 for each spatial dimension
    
    # Apply large Gaussian blur at half resolution
    gauss_sigma_downsampled = gauss_sigma / 2
    t.print(f"      Applying large Gaussian filter (sigma={gauss_sigma_downsampled})")
    weight_image = np.empty_like(water_image)
    for i in range(3):  # Assuming RGB channels
        t.print(f"        Filtering channel {i}")
        water_image[i], weight_image[i] = gaussian_filter_nan_cv2(water_image[i], sigma=gauss_sigma_downsampled)
        #water_image[i] = water_image[i]
    
    # Update the image profile for the new resolution
    profile = sat_image_profile.copy()
    profile['height'] = water_image.shape[1]  # Update the height
    profile['width'] = water_image.shape[2]   # Update the width
    profile['transform'] = rasterio.Affine(
        sat_image_profile['transform'].a * 2, sat_image_profile['transform'].b, sat_image_profile['transform'].c,
        sat_image_profile['transform'].d, sat_image_profile['transform'].e * 2, sat_image_profile['transform'].f
    )  # Scale the affine transform to adjust for the downsampled resolution

    t.print("    Creating water image time:")
    t.last()
    return water_image, profile, weight_image

    
    
def apply_clipping_mask(water_image, clipping_mask):
    """
    Applies the clipping mask to the water image by setting the border areas (defined by the mask) to NaN.

    Args:
        water_image (numpy.ndarray): The water image to be processed. Expected shape (3, height, width).
        clipping_mask (numpy.ndarray): Binary mask of the border of the image to trim off after blurring.
                                       Expected shape (height/2, width/2) for downsampled images.

    Returns:
        numpy.ndarray: The water image with the clipping mask applied.
    """
    t = TimePrint()
    
    water_image_height = water_image.shape[1]
    water_image_width = water_image.shape[2]
    
    zoom_factor_height = water_image_height / clipping_mask.shape[0]
    zoom_factor_width = water_image_width / clipping_mask.shape[1]
    
    t.print(f"      Upsampling trim mask to match water image {zoom_factor_height:.2f}")
    clipping_mask = zoom(clipping_mask, (zoom_factor_height, zoom_factor_width), order=0)
    
    t.print("      Applying clipping mask to water image")
    for i in range(3):  # Assuming RGB channels
        water_image[i, clipping_mask == 0] = np.nan  # Set edge areas to NaN after blurring
    t.last()
    return water_image

    


def main():
    t = TimePrint()
    t.print("Script started")
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Process Sentinel-2 images to create water images."
    )
    parser.add_argument('--split', type=int, default=1, 
        help="Number of subsets to split the processing into. This is useful for parallel processing. Each process does a subset of all the images to be processed. Default is 1."
    )
    parser.add_argument('--index', type=int, default=0, 
        help="Index of the subset to process (starting from 0). Default is 0."
    )
    parser.add_argument('--priority', type=str, nargs='+', default=[], 
        help="List of scene codes to prioritize for processing. This is useful for debugging on a particular scene. For example --priority 50KLB 49JGM"
    )
    
    parser.add_argument('--justpriority', type=bool, default=False, 
        help="If True only process the scenes listed in the priority list. Default: False indicating that all scenes should be processed"
    )
    
    parser.add_argument('--style', type=str, default='15th_percentile', 
        help="Image style to use. Default is '15th_percentile'. Valid values: 'low_tide_true_colour', 'low_tide_infrared'"
    )
    parser.add_argument('--sigma', type=int, default=60, 
        help="Gaussian sigma for stage 2. Default is 60."
    )
    parser.add_argument('--region', type=str, default='NorthernAU', 
        help="Region to process. Corresponds to folder in the style folder. 'NorthernAU' or 'GBR'."
    )
    
    args = parser.parse_args()
    
    
    region = args.region
    style = args.style  # Use the style argument from the command line

    s2_image_base_path = f'{S2_IMAGE_PARENT_PATH}/{style}/{region}'
    
    
    
    # Cache of already rendered land masks.
    land_mask_raster_dir = f'{LAND_MASK_RASTER_BASE_PATH}/{region}'
    

    stage2_gauss_sigma = args.sigma
    stage1_output_dir = f'working-data/{SCRIPT_NUMBER}-s1_{style}_b{LAND_MASK_BUFFER}g{STAGE1_GAUSS_SIGMA}/{region}'

    stage2_intermediate_output_dir = f'working-data/{SCRIPT_NUMBER}-s2i_{style}_b{LAND_MASK_BUFFER}g{STAGE1_GAUSS_SIGMA}/{region}'
    stage2_output_dir = f'working-data/{SCRIPT_NUMBER}-s2_{style}_b{LAND_MASK_BUFFER}g{stage2_gauss_sigma}/{region}'

    # cache of rendered masks
    reef_mask_raster_dir = f'{REEF_MASK_RASTER_BASE_PATH}/{region}'

    if not os.path.exists(stage2_intermediate_output_dir) and SAVE_STAGE1_INTERMED:
        os.makedirs(stage2_intermediate_output_dir)
    if not os.path.exists(stage2_output_dir):
        os.makedirs(stage2_output_dir)

    # Prepare the list of GeoTiff files to process
    priority_scenes = args.priority  # Priority list passed via command line
    tiff_files = prepare_sentinel_image_list(
        base_path=s2_image_base_path,  # Optionally merge with low_tide_base_path if needed
        split=args.split,
        index=args.index,
        priority_scenes=args.priority,
        justpriority=args.justpriority
    )
    
    total_files = len(tiff_files)
    
    if total_files == 0:
        t.print(f"No input GeoTiffs found in {s2_image_base_path}. Exiting")
        sys.exit()
    
    
    t.print("Loading land mask")
    # Load the shapefile using GeoPandas
    landmask_shapes = gpd.read_file(LAND_MASK_SHP)
    # Ensure output directory exists
    if not os.path.exists(land_mask_raster_dir):
        os.makedirs(land_mask_raster_dir)
        
    t.print("Loading rough reef mask")
    reefmask_shapes = gpd.read_file(REEF_MASK_SHP)
    # Ensure output directory exists
    if not os.path.exists(reef_mask_raster_dir):
        os.makedirs(reef_mask_raster_dir)
    
    for i, tiff_file in enumerate(tiff_files, start=1):
        filename = os.path.basename(tiff_file)

        t.print("------------------------------------------------------------")
        t.print(f"Processing {i} of {total_files}: {filename}")
        
        # If the final output already exists don't do the processing for that scene
        water_tif_path_stage2 = os.path.join(stage2_output_dir, filename.replace('.tif', '_water2.tif'))
        if os.path.exists(water_tif_path_stage2):
            t.print(f"Skipping {i} of {total_files}: {filename} (Stage 2 water image already exists)")
            continue
            
        # Get the Sentinel 2 scene such as 47LXQ from the filename
        scene_code = extract_scene_code(filename)

        # Read the original image
        t.print(f"    Reading satellite image: {os.path.basename(tiff_file)}")
        sat_image, sat_image_profile = read_image(tiff_file)  # Using read_image for all 8-bit images

        #------- Land mask setup ---------
        # The land mask removes land pixels from the calculations which would
        # bleed out into neighbouring water areas.
        land_mask_path = os.path.join(land_mask_raster_dir, f"{LAND_MASK_PREFIX}_{scene_code}_land_mask.tif")
        
        # Check if landmask already exists
        if not os.path.exists(land_mask_path):
            t.print(f"Land mask not found {land_mask_path}, Creating ...")
            # Rasterize the shapefile to create the land mask
            land_mask = rasterize_shapefile(landmask_shapes, sat_image_profile, land_mask_path)
            t.print(f"    Rasterized shapefile saved to {land_mask_path}")
        
        # Read from the saved result for consistency. Seems that read result is
        # is different (maybe the shape) from the form returned by rasterize_shapefile
        # but I haven't debugged why.
        t.print(f"    Reading land mask: {os.path.basename(land_mask_path)}")
        land_mask, _ = read_image(land_mask_path)

        
        t.print("    Buffering land mask to reduce bleed from nearshore")
        # Convert to 2D array for the dilation step
        # Land mask will be (band, height, width) in shape. Select 1st band.
        # returns (height, width) array.
        land_mask_2d = land_mask[0]

        # Convert the input mask from 0 - water, 255 - land to a binary mask 0 - water, 1 - land
        binary_land_mask = (land_mask_2d == 255).astype(np.uint8)


        
        # Create a circular structuring element (disk-shaped) with the desired buffer size
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LAND_MASK_BUFFER * 2 + 1, LAND_MASK_BUFFER * 2 + 1))

        # Buffer the land mask using a round dilation of LAND_MASK_BUFFER pixels
        # This is to reduce the nearshore pixels from affecting the estimate of the water colour
        buffered_land_mask = cv2.dilate(binary_land_mask, struct, iterations=1).astype(np.uint8)

        
        
        #------- Reef mask setup ---------
        # Rasterise the initial rough reef mask.
        reef_mask_path = os.path.join(reef_mask_raster_dir, f"{REEF_MASK_PREFIX}_{scene_code}_reef_mask.tif")
        
        # Check if reefmask already exists
        if not os.path.exists(reef_mask_path):
            t.print(f"Reef mask not found {reef_mask_path}, Creating ...")
            # Rasterize the shapefile to create the land mask
            reef_mask = rasterize_shapefile(reefmask_shapes, sat_image_profile, reef_mask_path)
            t.print(f"    Rasterized shapefile saved to {reef_mask_path}")

        t.print(f"    Reading reef mask: {os.path.basename(reef_mask_path)}")
        reef_mask, _ = read_image(reef_mask_path)

        binary_reef_mask = (reef_mask[0] == 255).astype(np.uint8)
        
        # Create a circular structuring element (disk-shaped) with the desired buffer size
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REEF_MASK_BUFFER * 2 + 1, REEF_MASK_BUFFER * 2 + 1))

        # Buffer the reef mask. This is not nearly as important as the 
        # the land masking, as the reef mask edges should already correspond
        # to fairly deep water.
        buffered_reef_mask = cv2.dilate(binary_reef_mask, struct, iterations=1).astype(np.uint8)
        
        
        # ---- Combine reef and land mask into a single mask ----
        # Combine reef and land mask (logical OR operation)
        t.print("    Combining reef and land masks")
        land_reef_mask = np.logical_or(buffered_land_mask, buffered_reef_mask).astype(np.uint8)
        
        # Convert from (height, width) to (band, height, width)
        land_reef_mask = np.expand_dims(land_reef_mask, axis=0)  
        

        # Create a mask that will trim the outside border of the final image
        # The trim needs to be larger than the sigma because we get a small contribution
        # from large distances, up to the point where floating point rounding errors
        # occur at about 6 - 7 sigma.
        trim_pixels = STAGE1_GAUSS_SIGMA * 5        
        #trim_pixels = 1
        t.print(f"    - Calculating clipping mask: {trim_pixels} pixels\n")
        clipping_mask = create_clipping_mask_cv2(sat_image, sat_image_profile, trim_pixels)
        
        
        t.print("    - Creating base water image to infill rough reef masked areas:\n")
        
        # Create an initial water estimate for replacing the reef masked areas with
        # For computational efficiency we would ideally only need to calculate 
        # this in areas where there is a rought reef mask as we throw the rest
        # of the image away. For simplicity we just calculate everywhere.
        #base_water_image = create_water_image(sat_image, sat_image_profile, land_reef_mask, clipping_mask, STAGE1_GAUSS_SIGMA, water_tif_path)
        base_water_image, base_water_image_profile, weight_image = create_water_image(sat_image, sat_image_profile, land_reef_mask, STAGE1_GAUSS_SIGMA)
        
        # Save the intermediate image with the reef areas replaced
        # This is optional because it is for debugging purposes
        if SAVE_STAGE1_BLUR or SAVE_WEIGHTS_OUTPUT and not os.path.exists(stage1_output_dir):
                os.makedirs(stage1_output_dir)
        if SAVE_STAGE1_BLUR:
            s1_water_tif_path = os.path.join(stage1_output_dir, filename.replace('.tif', '_water1.tif'))
            save_image_uint(base_water_image, base_water_image_profile, s1_water_tif_path, bit_depth=8)
        if SAVE_WEIGHTS_OUTPUT:
            s1_weights_tif_path = os.path.join(stage1_output_dir, filename.replace('.tif', '_weights.tif'))
            save_image_uint(weight_image, base_water_image_profile, s1_weights_tif_path, bit_depth=32)
            
        # ----- Intermediate water estimate: -----
        # Replace the rough reef areas with the base water image.
        # Clip out the rough reef areas from base water estimate, then copy
        # and paste these over the top of a copy of the satellite imagery. 
        # The goal is to make a new version of the satellite imagery where all 
        # the reefs and shallow areas (that are mostly affected by benthic reflectance)
        # are replaced by an estimate of the colour of the neighbouring water.
        # This way this new image will be close to what the water would look 
        # like without reefs and shallow areas, i.e. closer to an estimate
        # of the water colour and not benthic reflectance. This image will
        # however still have rough bits because the rough reef mask, is rough.
        # It also doesn't include masking of small reef areas and so we use a second water
        # estimation stage to smooth over these limitations. This time we use
        # a smaller level of blurring because we no longer need to interpolate
        # over large shallow and masked out areas.
        t.print("    - Replacing rough reef mask areas with stage 1 water colour estimate")
        water_image_intermediate = sat_image.copy()

        # Resize the base_water_image to match the resolution of the original sat_image
        # This is needed because the water estimate is half the original resolution.
        height, width = water_image_intermediate.shape[1], water_image_intermediate.shape[2]
        base_water_image_resized = np.zeros_like(water_image_intermediate)

        t.print("      Upsampling the water estimate to full resolution")
        # Resize each channel of the base_water_image (assuming 3 channels)
        for band in range(3):
            # Upsample using Bilinear interpolation. This is probably wildly
            # unnecessary and we could have used the faster nearest method
            # because the base water image is such a smooth image (from all the 
            # blurring).
            base_water_image_resized[band] = cv2.resize(base_water_image[band], (width, height), interpolation=cv2.INTER_NEAREST)
    
        t.print("      Replacing reef masked areas with stage 1 water estimate")
        # In the satellite imagery in areas that are masked by the rough 
        # reef mask, replace then with the initial water estimate. 
        reef_mask_expanded = np.expand_dims(buffered_reef_mask, axis=0)  # (band, height, width)
        water_image_intermediate = np.where(reef_mask_expanded == 1, base_water_image_resized, water_image_intermediate)

        # Save the intermediate image with the reef areas replaced
        # This is optional because it is for debugging purposes
        if SAVE_STAGE1_INTERMED:
            intermediate_water_image_path = os.path.join(stage2_intermediate_output_dir, filename.replace('.tif', '_wateri.tif'))
            save_image_uint(water_image_intermediate, sat_image_profile, intermediate_water_image_path, bit_depth=8)
            

        t.print("    - Creating water image for Stage 2:\n")
        # In stage 2 the aim is to estimate the final water estimate. All the
        # large reefs and large shallow have been replaced with the initial
        # estimate of the water colour, and so we don't need to remask out these
        # areas. We want them to be blended into the rest of the estimate.
        # This time we will only mask out land areas, with buffering, as we 
        # still don't want these areas to influence the result. 
        
        # If the first stage interpolation was insufficient to infill a large area we want
        # these areas to be infilled with the second pass. We therefore need to find
        # the areas that are 0. Note this will also pick up the outer border.
        gaps_mask = (water_image_intermediate[0] == 0).astype(np.uint8)
        land_and_gaps_mask = np.logical_or(buffered_land_mask, gaps_mask).astype(np.uint8)
        
        land_and_gaps_mask = np.expand_dims(land_and_gaps_mask, axis=0)  # Convert from (height, width) to (band, height, width)
        
        # For debugging
        #mask_tif_path_stage2 = os.path.join(stage2_output_dir, filename.replace('.tif', '_mask.tif'))
        #save_image_uint(land_and_gaps_mask, sat_image_profile, mask_tif_path_stage2, bit_depth=8)
            
        t.print(f"{land_and_gaps_mask.shape=}")
        t.print(f"{buffered_land_mask.shape=}")
        t.print(f"{gaps_mask.shape=}")
        t.print(f"{water_image_intermediate.shape=}")
        # For Stage 2, use the land mask only (reefs are already replaced with water)
        #create_water_image(water_image_intermediate, sat_image_profile, 
        #    buffered_land_mask, clipping_mask, stage2_gauss_sigma, water_tif_path_stage2)
        
        # Create an initial water estimate and get the updated profile
        final_water_image, final_water_image_profile, weight_image = create_water_image(water_image_intermediate, sat_image_profile, land_and_gaps_mask, stage2_gauss_sigma)
        
        # Create a land mask that trims the water estimate near the land. Without
        # this the water estimate encroaches the land by kms and the coastline is
        # confusing
        #buffered_land_mask = cv2.dilate(binary_land_mask, struct, iterations=1).astype(np.uint8)
        #water_image = zoom(water_image, (1, 0.5, 0.5), order=1)

        # Apply the clipping mask after all processing is done
        final_water_image = apply_clipping_mask(final_water_image, clipping_mask)
        
        t.print(f"      Saving water image {water_tif_path_stage2}")
        # Save the water image externally with the updated profile
        save_image_uint(final_water_image, final_water_image_profile, 
            water_tif_path_stage2, bit_depth=16)
        if SAVE_WEIGHTS_OUTPUT:
            weights_tif_path_stage2 = os.path.join(stage2_output_dir, filename.replace('.tif', '_weights2.tif'))
            save_image_uint(weight_image, final_water_image_profile, weights_tif_path_stage2, bit_depth=32)
        t.last()
        
if __name__ == "__main__":
    main()


