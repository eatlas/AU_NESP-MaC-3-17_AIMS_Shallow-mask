import os
import sys
import glob
import numpy as np
import argparse
import cv2
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from rasterio import features
from affine import Affine
import rasterio
import math
import fiona
from fiona.crs import from_epsg
from fiona import open as fiona_open

from imageutil import (
    read_image,  # Use this for reading images
    save_image_uint,
    TimePrint,
    extract_scene_code,
    create_clipping_mask_cv2,
    debug_image,
    raster_polygons,
    prepare_sentinel_image_list
)

"""
This script calculates shallow marine masks for Sentinel 2 scenes by combining multiple detector outputs
applied to both "All-tide" and "Low-tide" Sentinel 2 composite imagery. It leverages a previously generated
"water estimate" image and applies several feature detection techniques to identify shallow marine features,
including coral reefs and other submerged structures. The final result is saved as a raster image and optionally
converted to a shapefile of polygon features.

**Key Processing Steps:**

1. **Input Data Setup:**
   - Uses all-tide and low-tide Sentinel 2 composite imagery along with a corresponding "water estimate" 
     calculated by a previous script (04-create-water-image_with_mask.py).
   - The script can be run in parallel by using the `--split` and `--index` arguments to divide the workload.

2. **Detector Configurations:**
   - Multiple detectors, each tuned for different conditions (inshore, midshelf, offshore), are applied 
     to the imagery. Different "sensitivity" levels (Low, Medium, High, VHigh) adjust thresholds for 
     feature detection.
   - Each detector uses a set of bands and thresholds to isolate subtle bright or dark features against 
     a water reference image.
   - Detectors can optionally apply spatial masks to limit detection to certain brightness ranges or areas 
     (e.g., distinguishing inshore turbid areas from offshore clearer waters).

3. **Water Difference Calculation:**
   - For each detector, the script computes a water difference image: 
     `( (sat_image - water_image)² ) / (water_image + offset)` 
     across selected bands, scaled to highlight features within a specified range.
   - Includes a "dark feature" option to consider targets darker than the water background.
   - Applies median and Gaussian filtering to reduce noise.

4. **Clipping and Masking:**
   - Uses a land mask to ensure that features do not appear on land. A small negative buffer is applied 
     to allow features to slightly overlap with the land boundary, ensuring a clean vector edge in subsequent 
     processing stages.
   - Applies a cleanup mask to remove known noisy areas.

5. **Combining Detector Outputs:**
   - Accumulates results from all active detectors for each scene to create a combined difference image.
   - Performs morphological operations (dilation, erosion) to cluster nearby reef pixels and fill small holes, 
     creating more coherent features.

6. **Bloom Correction (All-tide Image):**
   - Applies unsharp masking (large-radius Gaussian filters) to the all-tide imagery to reduce "bloom" 
     artifacts—bright halos around shallow reefs.

7. **Final Output and Polygon Conversion:**
   - Saves the final combined difference raster, scaling values from 0 to 1 into a suitable 8-bit output.
   - Resamples the computed mask raster to double resolution prior to conversion to a polygon to ensure
     a smoother polygon. 
   - Simplifies polygon geometries to reduce complexity and merges all polygons into a shapefile for vector 
     GIS analysis.

The processing is performed per region and per detector sensitivity level. Each of these runs can be
split into parallel processes to speed up the processing on a multi-core machine.

A full run (Northern AU and GBR for Low, Medium, High and VHigh) takes approximately 8 CPU days of
compute.

Each process takes approximately 12 GB of RAM each. No memory optimisation has been done in the development
of the script.

The following corresponds to the command line calls that were used to prepare the final dataset:

Use two processes each to speed up calculations:
python 05-create-shallow-and-reef-area-masks.py --split 2 --index 0 --region NorthernAU --detectors VLow --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 2 --index 1 --region NorthernAU --detectors VLow --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 2 --index 0 --region NorthernAU --detectors Low --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 2 --index 1 --region NorthernAU --detectors Low --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 2 --index 0 --region NorthernAU --detectors Medium --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 2 --index 1 --region NorthernAU --detectors Medium --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 2 --index 0 --region NorthernAU --detectors High --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 2 --index 1 --region NorthernAU --detectors High --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 2 --index 1 --region NorthernAU --detectors VHigh --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 2 --index 0 --region NorthernAU --detectors VHigh --sigma 40

python 05-create-shallow-and-reef-area-masks.py --split 1 --index 0 --region GBR --detectors VLow --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 1 --index 0 --region GBR --detectors Low --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 1 --index 0 --region GBR --detectors Medium --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 1 --index 0 --region GBR --detectors High --sigma 40
python 05-create-shallow-and-reef-area-masks.py --split 1 --index 0 --region GBR --detectors VHigh --sigma 40

Examples of running a single scene:
python 05-create-shallow-and-reef-area-masks.py --split 1 --index 0 --region NorthernAU --detectors VHigh --sigma 40 --priority 50KRD --justpriority True
python 05-create-shallow-and-reef-area-masks.py --split 1 --index 0 --region GBR --detectors Medium --sigma 40 --priority 55KDA --justpriority True
"""
# Number of the script, used to prepend folders of working data created by this script
SCRIPT_NUMBER = '05'


WORKING_PATH = 'working-data'

# Location of the top level directory for the Sentinel 2 composite imagery
S2_IMG_PATH = 'data/in-3p/AU_AIMS_S2-comp'

# I am storing the imagery in a different path than the 01-download-sentinel2.py
#S2_IMG_PATH = 'D:/AU_NESP-MaC-3-17_AIMS_S2-comp/data/geoTiffs'

# Land mask used to clip the features. We perform a small negative buffer on the land to
# ensure that the features mapped contain a small overlap with the land. This will allow
# final clipping to be done, right at the end of manual processing, ensuring that there 
# are not any holes on the land side of the features.
LAND_MASK_SHP = 'data/in-3p/AU_AIMS_Coastline_50k_2024/Split/AU_NESP-MaC-3-17_AIMS_Aus-Coastline-50k_2024_V1-1_split.shp'

# This shapefile is used to remove features. It should be used to mask areas of high noise
# where there is definitely no reefs. This is intended to be a rough bulk cleanup to ensure
# the output shapefiles are reasonably small in the output. 
CLEANUP_MASK_SHP = 'data/in/AU_Cleanup-remove-mask/AU_AIMS_NESP-MaC-3-17_Cleanup-remove-mask.shp'

LAND_MASK_BUFFER = 5   # Match with 04-create-water-_with_mask.py. Use to find water estimate files.
                       # Not used in the analysis, just for finding the input files.
                       # Combined with sigma to calculate the b5g40 part of the filenames.
                       
VERSION = '1-1'     # Use this to create different sets of processing for comparison

CLUSTER_SIZE = 5        # pixels. Merge features close together with dilation and erosion. 
                            # This is added to help infill small holes in patchy clusters
                            # of features. If this is set too high then neighbouring distinct
                            # features, such as reefs with deep channels between them, will
                            # get merged together. This will also merge and infill rivers
                            # narrower than the cluster size. 
SAVE_UNSHARP = False    # Save the unsharpened version of the all tide imagery. This is for debugging.
SAVE_COMBINED_RAW = True   # Save the raw combined imagery. Set to False to save on processing.
                            # all later stages will be skipped.
SAVE_SHAPEFILE = True   # Determine whether to save the output shapefile. Setting this to False
                        # speeds up the processing for tunning of the detector parameters.

# Simplication tolerance in degrees to apply after conversion to polygon. This needs to
# be big enough to smooth out the stair case steps from the pixel processing, but
# small enough not to introduce too much additional spatial error.
SIMPLIFY_TOLERANCE_DEG = 0.00007


# Scale the threshold between THRES/DETECTION_RATIO and THRES*DETECTION_RATIO
# This sets the band of the brightness values to map, prior to the noise
# reduction operations. We use a graduation of values rather than binary operations
# to retain more of the strength information through the processing. 
DETECTION_RATIO = 1.3

TRIM_PIXELS = 10    # Amount of pixels to trim off the outer border of the image to compensate
                    # for the noise reduction filters. 
                    
# Parameters that determine the mapping of the features. The mapping is achieve by
# combining the outputs from one or more detectors. Each detector can be tuned to
# focus on different water conditions. 
# The basic calculation is:
# water_difference = ((b1-bw1)^2+(b2-bw2)^2+(b2-bw2)^2)
# difference_gain = 1/(bw1+bw2+bw3+denominator_offset)
# water_difference * difference_gain * spatial_mask > detection_threshold
# where bn is the nth band of the satellite imagery and bwn is the nth band of the 
# water estimate. This is for a 3 band detector. The spatial mask is a soft mask used to
# select the water type that the detector works best in. This works by applying level clipping
# water estimate to select inshore or offshore water.
# spatial_mask = clip(bw2, spatial_mask_min_out, spatial_mask_max_in), where clip scales the
# bw2 from spatial_mask_min_out to 0 and spatial_mask_max_in to 1.
# For example to detect offshore deep corals
# get don't want to use the red channel, because there is no signal and only noise.
# We would also want a high sensitivity (low detection threshold) and a large amount
# of median filtering to reduce the noise.
# For inshore features reefs can often be darker than the surrounding water and so
# we would want to use the include_dark option.
# "name":(str) partial filename and display title for the detector
# "detection_threshold": (float) threshold applied to the difference between the satellite 
#       image and the water estimate. The output reef mask is a soft mask normalising
#       the detection image from detection_threshold/1.3 to detection_threshold*1.3 to
#       0 to 1. 
# "noise_reduction_median": (int) Median filter in pixels to apply the detector output
# "noise_reduction_gaussian" (int) Gaussian filter in pixels to apply prior to the median filter.
# "bands":([int]) bands to perform the feature detection with 0 - Red, 1 - Green and 2 - Blue
# "image": (str) which image to use for the detector: "Low tide", "All tide"
# "denominator_offset": (int) offset to adjust the amount of gain detector. A high value will
#       mean that the sentivity is more constant across the scene.
#       The Sentinel 2 sensor swaths have a different brightness due to
#       slight angle changes with the sensor. This results in the swaths
#       that are darker having a greater gain as the denominator is smaller.
#       These darker swaths have their noise amplified more. To level off the
#       noise in the swaths we add a constant to the denominator. This 
#       reduces the variation in the total denominator. 
#       With the denominator_offset of 0 there was significant difference
#       in the swath noise levels.
#       Setting the offset to 40 significantly reduced the problem. 
#       Changing the denominator_offset affects the matching detection_threshold
#       For example if the open water brightness averages 60, then with
#       a numerator of 2 the matching threshold will be 2/60 = 0.033. If we
#       raise the denominator_offset to 50 then the average denominator will
#       be larger (50+60) and so the threshold will need to be 2/(50+60) = 0.0182
#       to achieve the same sensitivity.
#       We want to make the denominator_offset approximately equal to the 
#       brightness of the water estimate in the region that we are trying to
#       detect.
# "include_dark": (boolean) If Frue then treat features darker than the water estimate the same
#       as feature lighter than the water. If False then all the negative values in the
#       water difference are capped at zero.
#       water_difference = (max(0,b1-bw1))^2+(max(0,b2-bw2))^2+(max(0,b2-bw2))^2)
# "spatial_mask": (boolean) If True create a mask from the all tide water estimate to softly
#       crop the detector. Use this to limit the spatial extent to inshore turbid areas, or
#       offshore clear waters.
# "spatial_mask_band": (int) Which band to use as the spatial mask, from all tide water estimate.
# "spatial_mask_min_out": (int) All areas brighter than this value will be partially included.
#       Anything below will be masked out of the detector.
# "spatial_mask_max_in" : (int) All areas brighter than this will be 100% included.


# To reduce the number of tweakable parameters we standardise
# on a base level of sensitivity, that get scaled for low, medium and high
# output products.

OFFSHORE_SENSITIVITY = 0.03     # base level sensitivity for offshore detector
MIDSHELF_SENSITIVITY = 0.05     # base level sensitivity for midshelf detector
# INSHORE_SENSITIVITY = 0.03      # base level sensitivity for inshore detector V1
INSHORE_SENSITIVITY = 0.03      # base level sensitivity for inshore detector V1-1

# Scaling of the base senstivity for the mask sensitivitiy
VLOW_SCALAR = 2
LOW_SCALAR = 1.4
MEDIUM_SCALAR = 1
HIGH_SCALAR = 0.7

# Our goal with this level of sensitivity is to better map out the ancient reefs off the Pilbra
# This sensitivity will be too high is some regions resulting in excessive noise.
VHIGH_SCALAR = 0.5          

# Set the scalar for the inshore detector higher as we were picking up too many extraneous
# features for the results to be useful. Additionally the boundaries of features we would 
# want to keep were bleeding out into neighbouring areas making them un-useable.
VHIGH_SCALAR_INSHORE = 0.6 

# The base detectors are reused for Low, medium and high products
# with just a different detection threshold.
BASE_OFFSHORE_DETECTOR = {
        "name":"base_offshore",
        "detection_threshold":OFFSHORE_SENSITIVITY,    
        "noise_reduction_median":25,    # pixels. 
        "noise_reduction_gaussian":2,   # Apply filter prior to median filter. This smoothes out quantisation
                                        # noise and step artefacts from a large median filter.
        "bands":[1],                    # Blue because we are focused on the boundaries of the deepest features.
        "image":"All tide",
        "denominator_offset":60,         
        "include_dark": False,          # Only pick features brighter than surrounding water
        "spatial_mask": True,
        "spatial_mask_band": 1,
        "spatial_mask_min_out": 100,    # I tried selecting for just deep clear water, however the water estimate
                                        # imagery is not very selective for this because the brightness shifts
                                        # due to the Sentinel sensor swaths starts to dominate for darker tones.
                                        # We therefore settled on a mask that separates inshore from midshelf.
        "spatial_mask_max_in" : 70
    }
    
BASE_MIDSHELF_DETECTOR = { 
        "name":"base_midshelf",              
        "detection_threshold":MIDSHELF_SENSITIVITY,     #V5: 0.05, V6: 0.03 V13: 0.04 
        #"noise_reduction_median":21,    # pixels. Detecting deeper noisy features so filter more. V1:21
        "noise_reduction_median":21,    # pixels. Detecting deeper noisy features so filter more. V1-1: 17
        "noise_reduction_gaussian":1,
        "bands":[1,2],                  # Green, Blue
        "image":"All tide",
        "denominator_offset":60,         # V5:0, V6: 60
        "include_dark": False,          # Only pick features brighter than surrounding water
        "spatial_mask": False,
    }

BASE_INSHORE_DETECTOR = { 
        "name":"base_inshore",
        "detection_threshold":INSHORE_SENSITIVITY,    # 
        #"noise_reduction_median":17,    #V1
        "noise_reduction_median":21,    # V1-1 - Make sure we have a smooth boundary. This will 
                                        # filter out small reefs (not ideal) but these will be
                                        # added back from the manual reef mapping. 
                                        # Our goal is to make a mask that is suitable for the
                                        # UQ habitat mapping, which will be a combination of
                                        # this dataset, the reef boundary mapping and a manual
                                        # seagrass mapping. This dataset is responsible for having
                                        # a clean boundary for optically shallow areas along the coastline,
                                        # not necessarily to map all the reefs.
                                
        "noise_reduction_gaussian":1,
        "bands":[0,1,2],                # Red, Green and Blue
        "image":"Low tide",
        "denominator_offset":350,
        "include_dark": True,           # Detect features darker than surrounding water
        "spatial_mask": True,
        "spatial_mask_band": 1,         # 0 - Red, 1 - Green, 2 - Blue
        "spatial_mask_min_out": 70,
        "spatial_mask_max_in" : 100
    } 

def calculate_water_difference(sat_image, water_image, soft_mask=None, bands=None, min_clip=0.04, max_clip = 0.05, denominator_offset=160, include_dark=True):
    """
    Calculate the low tide water difference as specified.

    The calculation is:
    ((("sat_image" - "water_image") ** 2).sum(axis=0) ** 0.5) /
    (("water_image").sum(axis=0) + denominator_offset)

    Then scale values from min_clip - max_clip to 0 - 1.

    Args:
        sat_image (numpy.ndarray): The low tide true color image (float32), with 
            NaNs in NoData areas.
        water_image (numpy.ndarray): The low tide water estimate image (float32), 
            with NaNs in NoData areas.
        soft_mask (numpy.ndarray, 0 - 1 float 32): A soft mask to apply to the calculation 
            to limit the spatial extent of the water difference, prior to the clipping. 
            An example is using a scaled and clipped version of the red channel of the 
            all tide imagery as a mask to limit the area that the low tide imagery is 
            applied. This is to reduce the extent of the noise.
        bands (list or array-like, optional): Specify which bands to include in the calculation. If None, use all bands.
        include_dark (boolean): Specify whether pixels darker than the surrounding water
            should be included. If False then negative values from the "sat_image" - "water_image" is clipped to 0. When estimating deep reefs in offshore areas
            the dark patches generally correspond to noise from clouds. Use True 
            when mapping inshore turbid areas where the reefs are shallow and often
            covered in macro-algae making them much darker than surrounding areas.
           

    Returns:
        numpy.ndarray: The calculated low tide water difference scaled from 0 to 1.
        If sat_image or water_image are None then this function returns None.
    """
    
    if sat_image is None or water_image is None:
        return None

    # Assertion to ensure both images have the same shape
    assert sat_image.shape == water_image.shape, \
        f"sat_image and water_image must have the same shape. Got {sat_image.shape} and {water_image.shape}."

    # Assertion to ensure the bands specified are within the valid range
    if bands is not None:
        assert all(band >= 0 and band < sat_image.shape[0] for band in bands), \
            f"Invalid band indices: {bands}. Valid range is 0 to {sat_image.shape[0] - 1}"
    

    # If no bands are specified, use all bands
    if bands is None:
        bands = range(sat_image.shape[0])  # Use all bands if no specific ones are provided

    # Compute the squared differences only for the specified bands
    if include_dark:
        diff_squared = (sat_image[bands, :, :] - water_image[bands, :, :]) ** 2
    else:
        # Cap negative values to zero, so pixels darker than water estimate do not
        # contribute to the signal.
        diff_squared = (np.maximum(sat_image[bands, :, :] - water_image[bands, :, :],0)) ** 2
        
    # Sum the squared differences over the specified bands, ignoring NaNs
    sum_diff_squared = np.nansum(diff_squared, axis=0)

    # Take the square root
    sqrt_diff = np.sqrt(sum_diff_squared)

    # Compute the denominator by summing the water image over the specified bands
    denominator = np.nansum(water_image[bands, :, :], axis=0) + denominator_offset

    # Avoid division by zero
    denominator = np.where(denominator == 0, np.nan, denominator)

    # Compute the final result
    result = sqrt_diff / denominator

    # Apply the soft mask if supplied
    if soft_mask is not None:
        assert soft_mask.dtype == np.float32, \
        f"Expecting soft_mask to be float32 but found {soft_mask.dtype}"
        assert soft_mask.ndim == 2, \
        f"Expecting soft_mask shape height, width, got {soft_mask.shape}"
        
        result = result * soft_mask
        
    # Scale values from 0.04 - 0.05 to 0 - 1
    scaled_result = (result - min_clip) * (1.0 / (max_clip - min_clip))
    scaled_result = np.clip(scaled_result, 0, 1)

    return scaled_result

def apply_clipping_mask(image, clipping_mask):
    # Assertion to ensure both image and clipping_mask have the same shape
    assert image.shape == clipping_mask.shape, \
        f"image and clipping_mask must have the same shape. Got {image.shape} and {clipping_mask.shape}."
    
    assert image.dtype == np.float32, \
        f"Expecting image to be float32 but found {image.dtype}"
    
    assert clipping_mask.dtype == np.uint8, \
        f"Expecting image to be uint8 but found {clipping_mask.dtype}"
        
    assert clipping_mask.ndim == 2, \
        f"Expecting clipping mask shape height, width, got {clipping_mask.shape}"
        
    assert image.ndim == 2, \
        f"Expecting scaled image shape height, width, got {image.shape}"
        
    image[clipping_mask == 0] = np.nan
    return  image

def save_unit_scale_image(scaled_image, profile, output_path):
    """
    Save the scaled image using save_image_uint, scaling from 0-1 to 1-255, with 0 reserved for NoData.

    Args:
        scaled_image (numpy.ndarray, 32 bit): The scaled image array (values from 0 to 1), with NaNs in NoData areas.
        profile (dict): The image profile for saving.
        output_path (str): The path to save the image.
    """
        
    assert scaled_image.dtype == np.float32, \
        f"Expecting scaled_image to be float32 but found {scaled_image.dtype}"
    
        
    assert scaled_image.ndim == 2, \
        f"Expecting scaled image shape height, width, got {scaled_image.shape}"
    
    
    # Ensure the data is clipped to the unit scale.
    # Scale from 0-1 to 1-255
    # Keep as 32 bit prior to input to save_image_uint as NaN values represent NoData.
    scaled_image = (np.clip(scaled_image, 0, 1) * 254 + 1)
    

    save_image_uint(scaled_image, profile, output_path, clip=False, bit_depth=8)
    

def load_scene(sat_image_path, water_image_path, scene_code, image_description, TRIM_PIXELS, t):
    """
    Load the satellite composite imagery and the matching water estimate image.
    This load normalises the satellite imagery and the water image to 32 bit float 
    outputs with a scale from 1 - 255.
    All images are rescaled in size to the satellite image and so the sat_profile
    can be used for any of the returned values.
    Returns:
        sat_image (numpy.ndarray 32 bit float), 
        sat_profile(dict), 
        water_image (numpy.ndarray 32 bit float)
        clipping_mask (numpy.ndarray 32 bit float)
    """
    # Find low tide image and water image files
    t.print(f"    {sat_image_path=}")
    t.print(f"    {water_image_path=}")
    t.print(f"    {scene_code=}")
    image_paths = os.path.join(sat_image_path, f'*{scene_code}*.tif')
    water_paths = os.path.join(water_image_path, f'*{scene_code}*water2.tif')
    image_files = glob.glob(image_paths)
    water_files = glob.glob(water_paths)

    sat_image = None
    water_image = None
    clipping_mask = None
    sat_profile = None

    if image_files and water_files:
        image_file = image_files[0]
        t.print(f"    Reading {image_description} image: {os.path.basename(image_file)}")
        sat_image, sat_profile = read_image(image_file)
        
        sat_image = sat_image.astype(np.float32)
        
        water_file = water_files[0]
        t.print(f"    Reading {image_description} image: {os.path.basename(water_file)}")
        water_image, water_profile = read_image(water_file)

        # Create clipping mask from the water image. The water image is always
        # more clipped than the original satellite image.
        t.print(f"    Creating clipping mask for {image_description} image")
        clipping_mask_water_raw = create_clipping_mask_cv2(water_image, water_profile, TRIM_PIXELS)
        
        # Upscale the clipping mask from the water image to full resolution
        clipping_mask = cv2.resize(
            clipping_mask_water_raw, (sat_image.shape[2], sat_image.shape[1]), 
            interpolation=cv2.INTER_NEAREST
        )
        

        t.print(f"    Upsampling {image_description} water image to match original image resolution")
        upsampled_water_image = np.empty_like(sat_image, dtype=water_image.dtype)
        for band in range(water_image.shape[0]):
            upsampled_water_image[band] = cv2.resize(
                water_image[band], (sat_image.shape[2], sat_image.shape[1]), 
                interpolation=cv2.INTER_LINEAR
            )
        water_image = upsampled_water_image
        
        # Convert water image from uint16 to float32 and divide by 256 to match the range of 0-255
        # NoData values of 0 should stay 0. 
        water_image = water_image.astype(np.float32) / 256.0

         # Apply Gaussian filter to low tide image with sigma=1 to reduce
         # noise in the imagery. The goal is to remove fine scale noise that
         # results in single pixels triggering.
        t.print(f"    Applying Gaussian filter with sigma=1 to low tide image")
        for band in range(sat_image.shape[0]):
            sat_image[band] = cv2.GaussianBlur(sat_image[band], (0, 0), sigmaX=1, sigmaY=1)
    else:
        t.print(f"    {image_description} satellite image or water image not found for scene {scene_code}.")
        t.print(f"      Looked for {image_paths}")
        t.print(f"      and {water_paths}")
    return sat_image, sat_profile, water_image, clipping_mask

# Function to calculate spatial mask
def create_spatial_mask(image, min_out, max_in, band):
    return np.clip((image[band] - min_out) / (max_in - min_out), 0, 1)



def unsharp_mask_with_threshold_bands(image, sigma=1.0, strength=0.5, threshold=10, t=TimePrint()):
    """
    Args:
        image (numpy.ndarray 32 bit float, band, width, height): Image to sharpen
        sigma (float): radius of the sharpening
        strength (float): Relative weighting of the sharpening.
        threshold (float): Control which areas of the image are sharpened based 
            on the difference in brightness between neighboring pixels. This 
            helps prevent sharpening of areas with little to no contrast (like 
            smooth surfaces), allowing you to limit sharpening to only high-contrast 
            edges.
        
    """
    # Ensure the image is in a supported data type
    assert image.dtype == np.float32, \
        f"Expecting scaled_image to be float32 but found {image.dtype}"
    
    assert image.ndim == 3, \
        f"Expecting image to be band, height, width, got {image.shape}"

    # Process each band independently
    processed_bands = []
    for band in range(image.shape[0]):
        t.print(f'      Gaussian Blur band {band}')
        
        single_band = image[band]
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(single_band, (0, 0), sigma)
        mask = single_band - blurred  # Calculate high-pass mask
        
        # Apply threshold
        mask[np.abs(mask) < threshold] = 0  # Suppress low-contrast changes
        # Add the mask to the original band
        unsharp_band = cv2.addWeighted(single_band, 1, mask, strength, 0)
        processed_bands.append(unsharp_band)
    # Stack bands back together
    return np.stack(processed_bands, axis=0)
    
def main():
    t = TimePrint()
    t.print("Script started")
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Process Sentinel-2 images to create combined images."
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
    
    parser.add_argument('--region', type=str, default='NorthernAU', 
        help="Region to process. Corresponds to folder in the style folder. 'NorthernAU' or 'GBR'."
    )
    
    parser.add_argument('--detectors', type=str, default='Medium', 
        help="Selects which of the detector set to process. Possible options 'Low', 'Medium', 'High'"
    )
    
    parser.add_argument('--sigma', type=str, default='60', 
        help="Size of the blurring used to create the water estimate. Used to find the water estimate files. Should match that used in 04-create-water-image_with_mask.py"
    )
    

    args = parser.parse_args()
    
    assert args.split > args.index >= 0, f"Split must be larger than index. Got {args.split} and {args.index}"

    region = args.region
    t.print(f"Processing region: {region}")

    LOW_TIDE_STYLE = 'low_tide_true_colour'
    ALL_TIDE_STYLE = '15th_percentile'
    
    
    low_tide_base_path = f'{S2_IMG_PATH}/{LOW_TIDE_STYLE}/{region}'
    all_tide_base_path = f'{S2_IMG_PATH}/{ALL_TIDE_STYLE}/{region}'

    # Paths to water images
    # image_code corresponds to the land mask in pixels and the gaussian filter. 
    # Must match that generated by 04-create-water-image_with_mask.py
    image_code = f'b{LAND_MASK_BUFFER}g{args.sigma}'       
    low_tide_water_dir = f'{WORKING_PATH}/04-s2_{LOW_TIDE_STYLE}_{image_code}/{region}'
    all_tide_water_dir = f'{WORKING_PATH}/04-s2_{ALL_TIDE_STYLE}_{image_code}/{region}'

    very_low_mask_detectors = [
        { 
            **BASE_OFFSHORE_DETECTOR,
            "name":"offshore",
            "detection_threshold":OFFSHORE_SENSITIVITY*VLOW_SCALAR
        },
        { 
            **BASE_MIDSHELF_DETECTOR,
            "name":"midshelf",
            "detection_threshold":MIDSHELF_SENSITIVITY*VLOW_SCALAR
        },
        { 
            **BASE_INSHORE_DETECTOR,
            "name":"inshore",
            "detection_threshold":INSHORE_SENSITIVITY*VLOW_SCALAR
        }
    ]

    low_mask_detectors = [
        { 
            **BASE_OFFSHORE_DETECTOR,
            "name":"offshore",
            "detection_threshold":OFFSHORE_SENSITIVITY*LOW_SCALAR
        },
        { 
            **BASE_MIDSHELF_DETECTOR,
            "name":"midshelf",
            "detection_threshold":MIDSHELF_SENSITIVITY*LOW_SCALAR
        },
        { 
            **BASE_INSHORE_DETECTOR,
            "name":"inshore",
            "detection_threshold":INSHORE_SENSITIVITY*LOW_SCALAR
        }
    ]
    
    medium_mask_detectors = [
        { 
            **BASE_OFFSHORE_DETECTOR,
            "name":"offshore",
            "detection_threshold":OFFSHORE_SENSITIVITY*MEDIUM_SCALAR
        },
        { 
            **BASE_MIDSHELF_DETECTOR,
            "name":"midshelf",
            "detection_threshold":MIDSHELF_SENSITIVITY*MEDIUM_SCALAR
        },
        { 
            **BASE_INSHORE_DETECTOR,
            "name":"inshore",
            "detection_threshold":INSHORE_SENSITIVITY*MEDIUM_SCALAR
        }
    ]
    
    high_mask_detectors = [
        { 
            **BASE_OFFSHORE_DETECTOR,
            "name":"offshore",
            "detection_threshold":OFFSHORE_SENSITIVITY*HIGH_SCALAR
        },
        { 
            **BASE_MIDSHELF_DETECTOR,
            "name":"midshelf",
            "detection_threshold":MIDSHELF_SENSITIVITY*HIGH_SCALAR
        },
        { 
            **BASE_INSHORE_DETECTOR,
            "name":"inshore",
            "detection_threshold":INSHORE_SENSITIVITY*HIGH_SCALAR
        }
    ]
    
    vhigh_mask_detectors = [
        { 
            **BASE_OFFSHORE_DETECTOR,
            "name":"offshore",
            "detection_threshold":OFFSHORE_SENSITIVITY*VHIGH_SCALAR,
                      
        },
        { 
            **BASE_MIDSHELF_DETECTOR,
            "name":"midshelf",
            "detection_threshold":MIDSHELF_SENSITIVITY*VHIGH_SCALAR
        },
        { 
            **BASE_INSHORE_DETECTOR,
            "name":"inshore",
            "detection_threshold":INSHORE_SENSITIVITY*VHIGH_SCALAR_INSHORE
        }
    ]
    
    


    detector_options = {
        'VLow': very_low_mask_detectors,
        'Low': low_mask_detectors,
        'Medium': medium_mask_detectors,
        'High': high_mask_detectors,
        'VHigh': vhigh_mask_detectors
    }
    
    # Check if the specified detector set exists, and raise an error if it doesn't
    if args.detectors not in detector_options:
        raise ValueError(f"Invalid detector set specified: '{args.detectors}'. "
                         f"Available options are: {', '.join(detector_options.keys())}")

    # Select the appropriate detector set if valid
    detectors = detector_options[args.detectors]

    # Select the appropriate detector set based on the argument
    #detectors = detector_options.get(args.detectors, medium_mask_detectors)

    t.print(f"Using detectors set: {args.detectors}")
    
    dir_postfix = f"_V{VERSION}_b{LAND_MASK_BUFFER}g{args.sigma}_{args.detectors}"
    out_base_dir = f'{WORKING_PATH}/{SCRIPT_NUMBER}-auto-mask/{region}{dir_postfix}'

    out_shapefile_dir = f'{out_base_dir}/shapefile'
    
    
    t.print("Collecting scene codes")

    # Prepare the list of TIFF files using the extended function
    tiff_images = prepare_sentinel_image_list(
        base_path=all_tide_base_path,  # Optionally merge with low_tide_base_path if needed
        split=args.split,
        index=args.index,
        priority_scenes=args.priority,
        justpriority=args.justpriority
    )

    # Extract scene codes from the selected TIFF files for logging purposes
    scene_codes = [extract_scene_code(os.path.basename(f)) for f in tiff_images]
    scene_codes = list(dict.fromkeys(scene_codes))  # Remove duplicates while maintaining order

    total_scenes_to_process = len(scene_codes)
    t.print(f"Processing {total_scenes_to_process} scenes: {scene_codes}")

    if total_scenes_to_process == 0:
        t.print("No scenes to process")
        sys.exit()
    
    t.print("Loading land mask")
    # Load the shapefile using GeoPandas
    landmask_shapes = gpd.read_file(LAND_MASK_SHP)
    
    t.print("Loading cleanup mask")
    # Load the shapefile using GeoPandas
    cleanup_mask_shapes = gpd.read_file(CLEANUP_MASK_SHP)
    

    for i, scene_code in enumerate(scene_codes, start=1):

        t.print("------------------------------------------------------------")
        t.print(f"Processing {i} of {total_scenes_to_process}: Scene {scene_code}")


        combined_diff_output_file = os.path.join(
            out_base_dir, 'combined', f'Combined_{scene_code}.tif')
        
        shapefile_output = os.path.join( out_base_dir, 'shapefiles')
        os.makedirs(shapefile_output, exist_ok=True)
        shapefile_diff_output_file = os.path.join(
            shapefile_output, f'Reefs_and_Shallow_{scene_code}.shp')

        # Skip processing if combined output already exists
        if os.path.exists(shapefile_diff_output_file):
            t.print(f"Skipping scene {scene_code} (combined output file already exists)")
            continue

        
        # Initialize variables
        detector_diffs = []
        combined_diff_f32s = None
        combined_clipping_mask = None
        combined_profile = None

        # Load necessary images
        all_tide_image, all_tide_profile, all_tide_water_image, all_tide_clipping_mask = load_scene(
            all_tide_base_path, all_tide_water_dir, scene_code, "all tide", TRIM_PIXELS, t
        )
        low_tide_image, low_tide_profile, low_tide_water_image, low_tide_clipping_mask = load_scene(
            low_tide_base_path, low_tide_water_dir, scene_code, "low tide", TRIM_PIXELS, t
        )

        # Bloom correction
        # In high-contrast satellite imagery of shallow coral reefs with steep sides and in clear water
        # blooming can occur. The intense brightness from shallow reef tops extends beyond their true 
        # boundaries, creating a soft, halo-like glow into surrounding deeper water. This effect is 
        # particularly noticeable in contrast enhanced images, where brightness levels are increased 
        # to highlight deep reef structures. In some cases, this glow can spread up to 250 meters from 
        # the reef edge, giving the illusion of light diffusion into darker areas. The exact cause
        # of the blooming is unknown, but seems likely to be some combination of water column scatter or
        # imaging artefact. It is not a reflection of the benthic reflectance. 
        #
        # The bloom is most obvious in shallow reefs with steep sides in clear water, 
        # for example in ribbon reefs. This blooming results in channels between reefs being arteficially 
        # brighter than they should resulting in reefs being merged in the mapping.
        # To remove the blooming we apply an a series of unsharp masks to the imagery. The unsharp
        # mask creates a dark edge around the perimeter of these bright shallow reefs. By choosing the
        # strength of the unsharp mask we can correct for most of the blooming. We find that a
        # light large unsharpen of 50 pixel radius corrects most of the far extending bloom, but still
        # leaves a small amount of bloom much closer to the edge.
        # Setting a threshold helps to ensure that the unsharp masks are not applied to the open
        # water areas, where it would amplify the noise.
        # The strength was adjusted so that the darkness neighbouring the outer edge of the ribbon
        # reefs in 54LZM matched the brightness of the wider open water. This was checked by loading
        # the processed imagery in photoshop and performing a level adjustment of 
        # red 13 - 28
        # green 35 - 43
        # blue 59 - 72
        # This super high contrast enhancement makes the bloom very obvious in the imagery and shows
        # if it has been corrected.
        # Only apply the bloom correction to the all tide imagery because that is the imagery used
        # for mapping offshore reefs where the blooming has an obvious effect.
        if all_tide_image is not None:
            t.print('    Applying unsharpen mask to correct for image bloom - Stage 1 - 50 pixels')
            all_tide_image = unsharp_mask_with_threshold_bands(all_tide_image, sigma=50, strength=0.15, threshold=3, t=t)
            t.print('    Applying unsharpen mask to correct for image bloom - Stage 1 - 8 pixels')
            all_tide_image = unsharp_mask_with_threshold_bands(all_tide_image, sigma=10, strength=0.15, threshold=3, t=t)
            
            if SAVE_UNSHARP:
                # Save each detector output for debugging
                unsharp_folder = f'{out_base_dir}/unsharp_all_tide'
                os.makedirs(unsharp_folder, exist_ok=True)
                unsharp_image_path = f'{unsharp_folder}/unsharp_all_tide_{scene_code}.tif'
                t.print(f'    Saving unsharpened imagery {unsharp_image_path}')
                save_image_uint(all_tide_image, all_tide_profile, unsharp_image_path, bit_depth=8)
        
        # Loop through each detector in the configuration
        for detector in detectors:
            t.print(f"Processing detector: {detector['name']}")
            
            # Determine the image based on detector configuration
            if detector["image"] == "All tide":
                image, water_image, profile, clipping_mask = all_tide_image, all_tide_water_image, all_tide_profile, all_tide_clipping_mask
            elif detector["image"] == "Low tide":
                image, water_image, profile, clipping_mask = low_tide_image, low_tide_water_image, low_tide_profile, low_tide_clipping_mask
            else:
                t.print(f"    Unknown image type for detector {detector['name']}, skipping.")
                continue

            # Skip if image is missing.
            if image is None:
                t.print(f"    No {detector['image']} image found for scene {scene_code}, skipping detector {detector['name']}")
                continue

            # Create spatial mask if required
            spatial_mask = None
            if detector.get("spatial_mask", False) and all_tide_image is not None:
                t.print("    Creating spatial mask for inshore limit")
                spatial_mask = create_spatial_mask(
                    all_tide_water_image, 
                    detector["spatial_mask_min_out"], 
                    detector["spatial_mask_max_in"], 
                    detector["spatial_mask_band"]
                )

            # Calculate water difference for the current detector
            min_clip = detector["detection_threshold"] / DETECTION_RATIO
            max_clip = detector["detection_threshold"] * DETECTION_RATIO

            t.print(f"    Calculating water difference for {detector['name']}")
            diff_scaled = calculate_water_difference(
                image, water_image, 
                soft_mask=spatial_mask, 
                bands=detector["bands"], 
                min_clip=min_clip, 
                max_clip=max_clip, 
                denominator_offset=detector["denominator_offset"], 
                include_dark=detector["include_dark"]
            )

            # Apply clipping mask if needed
            if diff_scaled is not None:
                
                # Apply individual median filtering
                gaussian_size = detector.get("noise_reduction_gaussian", 0)
                if gaussian_size != 0:
                    t.print(f"    Applying {gaussian_size}-pixel gaussian filter to detector {detector['name']}")
                    diff_scaled = cv2.GaussianBlur(diff_scaled.astype(np.float32), (0, 0), sigmaX=gaussian_size, sigmaY=gaussian_size)

                # Apply individual median filtering
                median_size = detector.get("noise_reduction_median", 0)
                if median_size != 0:
                    t.print(f"    Applying {median_size}-pixel median filter to detector {detector['name']}")
                    diff_scaled_for_blur = (diff_scaled * 255).astype(np.uint8)  # Convert to 8-bit for median filtering
                    diff_filtered_8u = cv2.medianBlur(diff_scaled_for_blur, median_size)
                    diff_filtered = diff_filtered_8u.astype(np.float32) / 255.0
                else:
                    diff_filtered = diff_scaled

                # Save each detector output for debugging
                detector_folder = f'{out_base_dir}/detector-{detector["name"]}'
                os.makedirs(detector_folder, exist_ok=True)
                detector_filename = f'detector-{detector["name"]}_{scene_code}.tif'
                diff_filtered_clipped = np.where(
                    clipping_mask == 0, np.nan, diff_filtered
                )
                save_unit_scale_image(diff_filtered_clipped, profile, os.path.join(detector_folder, detector_filename))
                
                if SAVE_COMBINED_RAW:
                    # Accumulate the filtered result
                    detector_diffs.append(diff_filtered)

        if not SAVE_COMBINED_RAW:
            t.print("Skipping combination processing. See SAVE_COMBINED_RAW")
            continue
        # Accumulate all detector differences and save combined output
        if detector_diffs:
            combined_diff_f32s = np.sum(detector_diffs, axis=0).clip(0, 1)
            
            # In tile 54LXQ the extent of the low_tide_clipping_mask is smaller than
            # the all_tide_clipping_mask because there must not have been enough
            # low tide images to make half of the low tide image. The all tide image is complete.
            # If we use the low tide clipping mask then we loose the reefs in half the tile.
            # We are better off using the all_time_clipping_mask by default because
            # it is more likely to be complete.
            combined_clipping_mask = all_tide_clipping_mask if all_tide_image is not None else low_tide_clipping_mask
            combined_profile = all_tide_profile if all_tide_image is not None else low_tide_profile
            
            # This failed for 54LXQ, falsely masking half the image.
            #combined_clipping_mask = low_tide_clipping_mask if low_tide_image is not None else all_tide_clipping_mask
            #combined_profile = low_tide_profile if low_tide_image is not None else all_tide_profile

            # Save final combined difference
            combined_folder = f'{out_base_dir}/combined-raw'
            os.makedirs(combined_folder, exist_ok=True)
            combined_filename = f'combined-raw_{scene_code}.tif'
            t.print(f"    Saving raw combined water difference")
            combined_diff_clipped = np.where(
                    clipping_mask == 0, np.nan, combined_diff_f32s
            )
            save_unit_scale_image(combined_diff_clipped, combined_profile, os.path.join(combined_folder, combined_filename))
        else:
            t.print(f"    No detector differences available to combine for scene {scene_code}")

    
        # ------------- Noise reduction ------------
        
        t.print(f"    Cluster features: Applying dilation, and erosion to combined difference")

        # Convert NaNs to zero for filtering
        combined_diff_f32s = np.where(np.isnan(combined_diff_f32s), 0, combined_diff_f32s)


        # Perform clustering of closely neighbouring reef features. This process will also
        # infill small holes in reefs. 
        t.print(f"    Applying {CLUSTER_SIZE}-pixel dilation")
        struct_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLUSTER_SIZE, CLUSTER_SIZE))
        combined_diff_f32s = cv2.dilate(combined_diff_f32s, struct_dilate)

        t.print(f"    Applying {CLUSTER_SIZE-1}-pixel erosion")
        struct_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLUSTER_SIZE-1, CLUSTER_SIZE-1))
        combined_diff_f32s = cv2.erode(combined_diff_f32s, struct_erode)

        # Set NoData areas to NaN
        combined_diff_f32s = np.where(
            combined_clipping_mask == 0, np.nan, combined_diff_f32s
        )

        t.print("    Rendering land mask")
        land_mask_u8 = raster_polygons(landmask_shapes, combined_profile)
        
        # ------ Land masking ------
        # With the land masking we want to achieve two things. The first is removing all
        # the false features that may be generated on the land and secondly to remove
        # small silvers that can occur in narrow rivers and around mangrove islands where the 
        # feature is not big enough to be considered an area of interest. Without this processing
        # these areas are lined with a polygon that is ~20 m in width that often tapers off as a 
        # series of many disconnected small features. 
        # This processing step aims to remove these features, which not affecting the rest of 
        # the mapping. 
        #
        # To remove the features touching the coastline that are too thin to be considered
        # as a feature we expand the coastline with a buffer (LAND_EDGE_TRIM), to crop these features. 
        # For valid features this will result in a landward side cutout that is removed by 
        # the cropping and the features will no longer touch the land. Any small thin 
        # features that only exist neighbouring the coast will be removed by this masking.
        #
        # To ensure that the features we are keeping touch and slightly overlap the land (to
        # allow later clean cutting of the simplified polygons) we need perform a buffer, but only
        # near the land, not in all directions. To do this we create a buffered version of the land
        # that extends just 3 pixels (on average) into the neighbouring valid features 
        # (land_side_line_feature). We use this to create a cropped version of the features that will
        # be just a 2 pixels line parallel to the coast. We then perform a buffer on this line 
        # just big enough to touch the original land again, plus a bit more to ensure overlap. 
        
        # This approach is imperfect because the land_side_line_feature will be expanded in all
        # all directions by the buffering and so will expend laterally along the coast causing
        # a little nub on either end of the feature where it touches the land. The length of this
        # nub will be equal in size to the LAND_EDGE_TRIM+land overlap buffer. We therefore
        # need to keep this trimming to as small as possible.
        
        # This algorithm some what worked, but created a second problem. Often continuous thin
        # coastal strips would be trimmed, but not completely, leaving many small fragments
        # which is not really an improvement. It is for this reason that the EDGE_TRIM is
        # off by default.
        EDGE_TRIM = False
        if EDGE_TRIM:
            LAND_EDGE_TRIM = 3  # pixels
            t.print("    Expanding land by {LAND_EDGE_TRIM} pixels to crop thin shoreline features")
            struct_dilate_land = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LAND_EDGE_TRIM, LAND_EDGE_TRIM))
            land_edge_trim_mask_u8 = cv2.dilate(land_mask_u8, struct_dilate_land)
            
            # Clip off the slightly extended land, trimming out the small thin features
            # neighbouring the coast. Larger land neighbouring features will have
            # 2 pixels trimmed off their land facing side.
            combined_diff_land_clip_f32s = (1-land_edge_trim_mask_u8) * combined_diff_f32s
            
            
            LAND_SIDE_LINE = 4 # pixels 
            # Expand the land by 7 pixels as a mask to capture the inner edge of the 
            # larger land neighbouring features. This should allow us to capture 
            # up to 4 pixels pixels of the feature. 
            land_side_buffer = LAND_SIDE_LINE + LAND_EDGE_TRIM
            struct_dilate_land_side = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (land_side_buffer, land_side_buffer))
            land_side_mask_u8 = cv2.dilate(land_mask_u8, struct_dilate_land_side)
            
            
            
            # Just keep the overlap between the expanded land and the neighbouring features
            # This should create a line of pixels on the landward side of the marine features.
            land_side_line_feature_f32s = land_side_mask_u8 * combined_diff_land_clip_f32s
            
            # The problem now is that, while we have trimmed off parts of the fringing shallow
            # area that is smaller than LAND_EDGE_TRIM, features that are on the edge of
            # this size will become fragmented, when previously they would have been continuous
            # If the land side line is very small (1 - 2 pixels in width) then the feature
            # Is barely above the trimming threshold. We will therefore use an erosion
            # and dilation to remove these features just at the edge of being big enough.
            struct_clip_land_side = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            land_side_line_feature_f32s = cv2.erode(land_side_line_feature_f32s, struct_clip_land_side)
            land_side_line_feature_f32s = cv2.dilate(land_side_line_feature_f32s, struct_clip_land_side)

            
            # Expand this line feature enough that it touches the land again. Actually
            # buffer a bit more to cover slight mismatches due to polygon simplification
            # stage. 
            line_feature_buffer = LAND_EDGE_TRIM + 3
            struct_dilate_line_feature = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_feature_buffer, line_feature_buffer))
            land_side_line_infill_f32s = cv2.dilate(land_side_line_feature_f32s, struct_dilate_line_feature)
            
            # Infill the gap between the feature and the land 
            combined_diff_f32s = land_side_line_infill_f32s + combined_diff_land_clip_f32s
        else:
            # Trim to a slightly shrunken version of the land (erosion by 4 pixels)
            # We do this so that the final fringing features are guaranteed to slightly
            # overlap with the land. Because we are processing as a raster there is still
            # polygon conversion and simplification to be perform. If we trim to the exact
            # land at this stange then after then after the subsequent processing there
            # are likely to be small errors that would lead to tiny gaps in parts between
            # the foreshore features and the coastline dataset. Here we plan for a final
            # clipping to be performed outside this script against the coastline dataset
            # using polygons rather than rasters.
            
            # Apply an erosion of 4 pixels to shrink the land by 40 m to ensure
            # partial overlap with the land.
            t.print("    Eroding land mask by 4 pixels for overlap")
            struct_erode_land = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            land_mask_eroded = cv2.erode(land_mask_u8.astype(np.uint8), struct_erode_land)
            
            t.print("    Applying land mask")
            # Apply the mask. Ocean is 0, Land is 1.
            # combined_diff_f32s is 32 bit float scaled 0 - 1.
            combined_diff_f32s = (1-land_mask_eroded) * combined_diff_f32s

        t.print("    Rendering cleanup mask")
        cleanup_mask = raster_polygons(cleanup_mask_shapes, combined_profile)
        
        t.print("    Applying cleanup mask")
        # Apply the mask. Ocean is 0, Land is 1.
        # combined_diff_f32s is 32 bit float scaled 0 - 1.
        combined_diff_f32s = (1-cleanup_mask) * combined_diff_f32s
        
        # Save the combined difference
        t.print(f"    Saving combined water difference to {os.path.basename(combined_diff_output_file)}")
        
        combined_diff_f32s = apply_clipping_mask(
            combined_diff_f32s, combined_clipping_mask
        )
        save_unit_scale_image(
            combined_diff_f32s, combined_profile, 
            combined_diff_output_file
        )
        
            
        if SAVE_SHAPEFILE:
            # Double the resolution of combined_diff_f32s using bilinear interpolation
            # This double resolution makes the pixels steps in the shapefile half the size
            # making the simplication smoother.
            t.print("    Doubling resolution of combined difference using bilinear interpolation")
            double_resolution = cv2.resize(combined_diff_f32s, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            # Apply a threshold of 0.5 to turn it into a binary mask. 
            binary_mask = np.where(double_resolution >= 0.5, 1, 0).astype(np.uint8)

            # Convert the binary mask into a shapefile, saving the shapefile to
            # shapefile_diff_output_file.
            # 1 - features to convert to polygons, 0 and NaN - pixels to ignore
            t.print("    Converting binary mask to shapefile")
            # Set up affine transform for the double resolution image
            transform = combined_profile["transform"] * combined_profile["transform"].scale(0.5, 0.5)

            # Extract shapes from binary mask, convert to polygons, and ignore 0 and NaN pixels
            
            polygons = []
            for geom, value in features.shapes(binary_mask, transform=transform):
                if value == 1:
                    geometry = shape(geom)
                    # Simplify the geometry
                    simplified_geometry = geometry.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
                    # Check if the simplified geometry is a Polygon or MultiPolygon
                    if simplified_geometry.geom_type in ['Polygon', 'MultiPolygon']:
                        polygons.append(simplified_geometry)

            # Create a GeoDataFrame with only polygon geometries
            if polygons:
                gdf = gpd.GeoDataFrame(geometry=polygons, crs=combined_profile['crs'])
            else:
                # Create an empty GeoDataFrame with Polygon geometry
                gdf = gpd.GeoDataFrame(columns=['geometry'], crs=combined_profile['crs'])

            # Define the schema explicitly
            schema = {
                'geometry': 'Polygon',
                'properties': {},
            }
            
            # Save the shapefile with the specified schema
            with fiona_open(
                shapefile_diff_output_file, 'w',
                driver='ESRI Shapefile',
                crs=gdf.crs.to_dict(),
                schema=schema
            ) as shapefile:
                for geom in gdf.geometry:
                    shapefile.write({
                        'geometry': mapping(geom),
                        'properties': {},
                    })

            t.print(f"    Shapefile saved to {shapefile_diff_output_file}")

    t.print("Processing completed")

if __name__ == "__main__":
    main()
