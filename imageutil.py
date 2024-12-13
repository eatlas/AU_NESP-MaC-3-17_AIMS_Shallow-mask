import re
import rasterio
from rasterio.plot import show
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import threshold_otsu
from scipy.ndimage import zoom
import os
import sys
import math
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk, dilation, erosion
#from skimage.restoration import inpaint
import cv2
from scipy import ndimage
import time
import os
import glob
import argparse
from PIL import Image
from rasterio.enums import Resampling
import time
import random


class TimePrint:
    """
    A utility class that provides timed print functionality. Each print includes the time 
    elapsed since the previous print. It is useful for tracking the execution time of code sections 
    without needing to manually manage timers. The class also provides a function to print just the 
    elapsed time and reset the timer without printing any message.

    Attributes:
        show_time (bool): Controls whether the elapsed time is printed. Defaults to True.
        last_time (float): Stores the time of the last print for calculating elapsed time.
        first_call (bool): Tracks if this is the first call to avoid printing a newline.

    Methods:
        print(message):
            Prints the provided message with the elapsed time since the previous print. 
            The first call does not include a newline before the message.
        
        last():
            Prints only the elapsed time since the last print and resets the timer. 
            This method is used when no new message needs to be printed but the elapsed time is needed.
        
    Example usage:
        t = TimePrint()
        t.print("Starting process...")  
        # Some code execution taking 0.1 sec
        t.print("Step 1 completed")  
        # More code execution taking 1.23 sec
        t.last()  # Prints the time delta from the last print and resets the timer
        
        Starting process... 0.10 s
        Step 1 completed 1.23 s
        
    """
    
    def __init__(self, show_time=True):
        self.last_time = time.time()
        self.show_time = show_time  # Control the time printing
        self.first_call = True  # Track if this is the first call

    def _handle_first_call(self, message):
        """Handle the first call behavior by printing the message without a newline and setting first_call to False."""
        if self.first_call:
            sys.stdout.write(f"{message}")
            self.first_call = False
        else:
            sys.stdout.write(f"\n{message}")
        sys.stdout.flush()

    def _print_time_delta(self):
        """Print the time delta since the last call, if applicable."""
        current_time = time.time()
        if not self.first_call:
            delta_time = current_time - self.last_time
            if self.show_time and delta_time >= 0:
                sys.stdout.write(f" {delta_time:.2f} s")
                sys.stdout.flush()
        self.last_time = current_time  # Update last_time to the current time

    def print(self, message):
        """
        Prints the provided message along with the time elapsed since the previous print.

        Args:
            message (str): The message to be printed.
        """
        self._print_time_delta()
        self._handle_first_call(message)

    def last(self):
        """
        Prints only the time elapsed since the previous print and resets the timer. 
        This method is used when there is no message to print but you want to print the elapsed time.
        """
        self._print_time_delta()  # Just print the time delta and reset the timer
        sys.stdout.write("\n")  # Add a newline after printing the time
        sys.stdout.flush()
        self.last_time = time.time()  # Reset the timer
        
def read_image(file_path):
    """
    Read a GeoTiff image using rasterio.
    Returns:
        image (numpy.ndarray, shape=(bands, height, width))
        profile (dict) rasterio profile
    """
    with rasterio.open(file_path) as src:
        return src.read(), src.profile

def debug_image(image, bins=10):
    """Print stats about the image to help with debugging"""
    print("Data type of image:", image.dtype)
    print("Shape of image:", image.shape)
    print("Max value in image:", np.nanmax(image))  # Use nanmax to ignore NaNs
    print("Min value in image:", np.nanmin(image))  # Use nanmin to ignore NaNs
    print(f"Number of NaNs in image: {np.isnan(image).sum()}")
    # Handle case where image contains only NaNs
    if np.isnan(image).all():
        print("Image contains only NaN values.")
        return
    
    # Calculate histogram of brightness values ignoring NaNs
    non_nan_image = np.nan_to_num(image, nan=0)
    hist, bin_edges = np.histogram(non_nan_image, bins=bins)
    
    # Print histogram
    print("Histogram of image values:")
    for i in range(len(hist)):
        print(f"Range {bin_edges[i]} to {bin_edges[i+1]}: {hist[i]}")


def ensure_directory_exists(file_path):
    """ Ensure that the directory containing the file exists. """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    
def save_image_uint16(image, profile, output_path):
    """ 
    Save an image with the updated profile for 16-bit data, scaling input values from 0-255. 
    This saves with lossless LZW compression. Using 16-bit allows sufficiently fine levels 
    to be recorded that quantisation errors don't affect the calculations. This is very 
    important for operations that take the difference between the water estimation and the
    original imagery. If we used 8 bit we would get banding in the resulting images.
    Handles both colour and grey scale images.
    Using 16 bit images results in images that are slightly more than double the 8 bit
    version and so 16 bit is only used for continous intermediate values where the 
    precision is needed.
    """
    print(f"{profile['count'] = }")
    # Scale the image from float 0-255 to uint16 0-65535
    scale_factor = 65535 / 255
    image_uint16 = np.clip(image, 0, 255) * scale_factor
    image_uint16 = image_uint16.astype(np.uint16)

    # Make a local copy of the profile to modify
    local_profile = profile.copy()
    local_profile.update({
        'dtype': 'uint16',
        'compress': 'lzw',  # Adding LZW compression
    })
    # If the imagery is just a 2D array then make sure it is grey scale
    if image.ndim == 2:
        local_profile['count'] = 1
    
    ensure_directory_exists(output_path)
    print(f"{local_profile['count'] = }")
    with rasterio.open(output_path, 'w', **local_profile) as dst:
        if local_profile['count'] == 1:
            print("Writing single band image")
            # Write the single channel image
            dst.write(image_uint16, 1)
        else:
            print("Saving multiband image")
            print(f'{image_uint16.shape = }')
            # Write multi-band image
            #for i in range(image_uint16.shape[0]):
            #    dst.write(image_uint16[i], i + 1)
            dst.write(image_uint16)

    print(f"    Image saved in 16-bit format at {output_path}")
    
def read_image_uint16(input_path):
    """ 
    Read a 16-bit image, either RGB or grayscale, and scale the values back to the float range 0-255. 
    """
    with rasterio.open(input_path) as src:
        profile = src.profile
        if profile['count'] == 1:
            print("Reading single band image")
            image = src.read(1)  # Single band read
        else:
            print("Reading multi band image")
            image = src.read()  # Multi-band read

        # Scale from 16-bit (0-65535) back to float 0-255
        image = image.astype(np.float32) / (65535 / 255)
    #print("Reading image")
    #print(f"{profile['count']}")
    print(f"    uint16 Image read from {input_path}")
    return image, profile

def make_grey_8bit_profile(profile):
    """
    Create an 8 bit single channel image profile from an original multi-channel profile.
    This is useful for making a profile for saving the processed image where the final
    form is a single channel, such as a mask calculated from an original multi-channel image.
    """
    # Adjust the profile to match the single-channel image
    profile_modified = profile.copy()  # Make a copy to avoid altering the original profile
    profile_modified['count'] = 1  # Set band count to 1 for grayscale image
    profile_modified['dtype'] = 'uint8'  # Ensure dtype is correct, adjust if necessary
    profile_modified['compress'] = 'lzw'
    return(profile_modified)
    
def save_image_grey_uint8(image, profile, output_path, mask=None, clip=True):
    """
    Save a single channel image in 8 bit quantisation. Values outside 1 - 255
    are clipped prior to savaling. 0 is reserved for No Data. NaN are converted
    to 0. This saves the images in COG format with internal subsampled pyramids.
    
    Args:
        image (numpy.ndarray): Image to save
        profile (rasterio profile dict): Geospatial profile of the image to save.
        output_path (str): Where to save the image
        mask (numpy.ndarray): Set masked pixels (values not 0) in the output to 0 (NoData)
        clip (boolean): If True then clip to 1-255 prior to saving.
    
    Raises:
        ValueError: If the input image is not a single band image.
    """
    # Ensure image is 2D, especially if it might have an extra singleton dimension
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)  # Remove the singleton dimension at position 0
    elif image.ndim != 2:
        print("Image details:")
        debug_image(image)
        raise ValueError("Image must be a 2D array to save as grayscale.")
    
    # Apply mask if provided
    if mask is not None:
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # Remove the singleton dimension at position 0
        elif mask.ndim != 2:
            print("Mask details:")
            debug_image(mask)
            raise ValueError("Mask must be a 2D array to save as grayscale.")
    
    if clip:
        image = np.clip(image, 1, 255)  # Clip values to 1-255 range (saving 0 for Nodata)

    # Convert NaN values and convert to 8-bit
    out_image = np.nan_to_num(image, nan=0).astype(np.uint8)
    
    # Apply mask if provided
    if mask is not None:
        if out_image.shape != mask.shape:
            print("Image:")
            debug_image(image)
            print("Mask:")
            debug_image(mask)
            msg = f"Mask ({mask.shape}) must have the same dimensions as the image ({image.shape})."
            raise ValueError(msg)
        out_image = np.where(mask, out_image, 0)  # Apply mask: Set masked pixels to 0

    # Adjust the profile to match the single-channel image
    # Prepare for creating a COG
    profile_modified = profile.copy()
    profile_modified.update({
        'count': 1,
        'dtype': 'uint8',
        'driver': 'GTiff',
        'compress': 'deflate',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'nodata': 0,
        'predictor': 2  # Use horizontal differencing for better compression
    })
    ensure_directory_exists(output_path)
    # Write the image as a COG
    with rasterio.open(output_path, 'w', **profile_modified) as dst:
        dst.write(out_image, 1)  # Write the single channel image
        
        # Generate overviews
        overviews = [2, 4, 8, 16, 32, 64]
        dst.build_overviews(overviews, Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')

    print(f"Image saved as COG at {output_path}")

def raster_polygons(shapefile_polygons, raster_profile):
    """Rasterize a shapefile based on a provided raster profile 
    This is used to convert the land and shallow reef shapefiles to rasters that align with the 
    pixels of the Sentinel 2 imagery. The image will have a background of 0 and where there
    are polygons the value will be 1.
    
    Args:
         shapefile_polygons (Geopandas polygons): Polygons to rasterise
         raster_profile (rasterio profile dict): Profile of the raster to rasterise to. This will
            correspond to the profile of the read GeoTiff Sentinel 2 tile we are processing.
    
    Returns:
        The rasterised image (uint8)
    """
    
    # Rasterize the shapes into a new raster array
    out_image = rasterize(
        [geometry for geometry in shapefile_polygons.geometry],
        out_shape=(raster_profile['height'], raster_profile['width']),
        transform=raster_profile['transform'],
        fill=0,  # Fill value for 'background'
        all_touched=True,  # Include all pixels touched by geometries
        dtype='uint8',
        default_value = 1
    )
    
    return out_image

def save_image_uint(image, profile, output_path, mask=None, clip=True, bit_depth=8):
    """
    Save an image in 8-bit, 16-bit, or 32-bit float quantisation. 
    Values outside 1 - 255 are clipped prior to saving for 8-bit and 16-bit. 
    0 is reserved for No Data. NaN are converted to 0 for 8-bit and 16-bit, 
    and preserved for 32-bit float. This saves the images in COG format with internal subsampled pyramids.

    Args:
        image (numpy.ndarray, 32 bit float): Image to save (can be greyscale or RGB).
        profile (rasterio profile dict): Geospatial profile of the image to save.
        output_path (str): Where to save the image.
        mask (numpy.ndarray): Set masked pixels (values not 0) in the output to 0 (NoData).
        clip (boolean): If True, clip to 1-255 prior to saving for 8-bit and 16-bit.
        bit_depth (number): Bit depth. Should be 8, 16, or 32. 
            - For 8-bit and 16-bit, the input is expected to be in the range of 1 - 255.
            - For 32-bit float, input is expected to be in floating point values.

    Raises:
        ValueError: If the input image does not have the expected dimensions.
    """
    #assert image.dtype == np.float32, \
    #    f"Expecting image to be float32 but found {image.dtype}"
        
    # Ensure image is either 2D (greyscale) or 3D (multi-channel, e.g., RGB)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)  # Remove the singleton dimension at position 0
    
    if image.ndim == 2:
        is_greyscale = True
    elif image.ndim == 3 and image.shape[0] == 3:
        is_greyscale = False  # It's a color image (assume RGB with 3 channels)
    else:
        print("Image details:")
        debug_image(image)
        raise ValueError("Image must be a 2D array for greyscale or a 3D array with 3 channels for RGB.")

    # Apply mask if provided
    if mask is not None:
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # Remove the singleton dimension at position 0
        elif mask.ndim != 2:
            print("Mask details:")
            debug_image(mask)
            raise ValueError("Mask must be a 2D array to save as greyscale or RGB.")
            
    if bit_depth not in [8, 16, 32]:
        raise ValueError(f"bit_depth must be 8, 16, or 32 but was {bit_depth}")

    if clip and bit_depth in [8, 16]:
        image = np.clip(image, 1, 255)  # Clip values to 1-255 range (saving 0 for Nodata)

    # Handle different bit depths
    if bit_depth == 8:
        # Convert NaN values and convert to 8-bit
        out_image = np.nan_to_num(image, nan=0).astype(np.uint8)
        dtype = 'uint8'
    elif bit_depth == 16:
        # Convert NaN values to 0 and scale the image to the full 16-bit range (0-65535)
        out_image = np.nan_to_num(image, nan=0).astype(np.float32)
        out_image = (out_image / 255.0 * 65535).astype(np.uint16)  # Scale 0-255 to 0-65535
        dtype = 'uint16'
    elif bit_depth == 32:
        # Preserve NaN values and store as 32-bit float
        out_image = image.astype(np.float32)  # Keep float values, NaNs are preserved
        dtype = 'float32'

    # Apply mask if provided
    if mask is not None:
        if out_image.shape[-2:] != mask.shape:
            print("Image:")
            debug_image(image)
            print("Mask:")
            debug_image(mask)
            msg = f"Mask ({mask.shape}) must have the same dimensions as the image ({image.shape[-2:]})."
            raise ValueError(msg)
        out_image = np.where(np.expand_dims(mask, axis=0) if out_image.ndim == 3 else mask, out_image, 0)

    # Adjust the profile to match the image (greyscale or multi-channel)
    profile_modified = profile.copy()
    profile_modified.update({
        'count': 1 if is_greyscale else 3,  # Adjust band count for greyscale or RGB
        'dtype': dtype,
        'driver': 'GTiff',
        'compress': 'deflate' if bit_depth in [8, 16] else None,  # Disable compression for 32-bit float
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'nodata': 0 if bit_depth in [8, 16] else None,  # No nodata for float
        'predictor': 2  # Use horizontal differencing for better compression (if applicable)
    })

    ensure_directory_exists(output_path)

    # Write the image as a COG
    with rasterio.open(output_path, 'w', **profile_modified) as dst:
        dst.write(out_image, 1 if is_greyscale else [1, 2, 3])  # Write single or multi-channel image

        if bit_depth in [8, 16]:
            # Generate overviews only for integer images (not for float32)
            overviews = [2, 4, 8, 16, 32, 64]
            dst.build_overviews(overviews, Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')

    # print(f"Image saved as COG at {output_path}")


def get_image_profile(file_path):
    """Get the profile of a GeoTiff image without reading the entire image."""
    with rasterio.open(file_path) as src:
        profile = src.profile
    return profile
    
def extract_scene_code(filepath):
    """
    Extract the Sentinel 2 tile scene code from the filename.
    Assumes the filenames have the structure '_DDAAA_', where
    DD are decimal digits and AAA are capital letters.
    For example: AU_AIMS_MARB-S2-comp_p15_TrueColour_46LGM_v2_2015-2024.tif
    would extract 46LGM
    """
    # Get the basename of the file (i.e., remove the directory path)
    basename = filepath.split('/')[-1]
    # Remove the file extension
    no_extension = basename.split('.')[0]
    # Use regex to find the pattern '_DDAAA_'
    match = re.search(r'_(\d{2}[A-Z]{3})', no_extension)
    if match:
        scene_code = match.group(1)
        return scene_code
    else:
        return None  # Return None if the pattern is not found

def create_clipping_mask_cv2(sat_image, sat_image_profile, trim_pixels):
    """
    This function creates a clipping mask from the black borders of the Sentinel 2
    images. This corresponds to NoData values of 0 around edges of the image corresponding
    to the slanted rectangle of the Sentinel 2 tile inside the image. 

    The original border is expanded using dilation by trim_pixels to
    allow trimming of edge effects in the water estimate calculations (blurring
    bleeding in black from the sides of the time).
    The output clipping mask is half the resolution of the input to match the 
    create_water_image
    
    Args:
        sat_image (numpy.ndarray): Original Sentinel 2 Geotiff image loaded in
        sat_image_profile (rasterio profile dict): Profile of the sat_image image
        trim_pixels (int): Number of pixels to trim off (multiples of 4)
    Returns:
        clipping mask.
    """
    t = TimePrint()

    # Process only the first band as the border is the same in all three 
    first_band = sat_image[0]

    # Define border width and output mask size ratio
    border_width = 1
    output_mask_size_ratio = 2
    
    # Expand the image by padding with a 0 border
    padded_sat_image = np.pad(first_band, pad_width=border_width, mode='constant', constant_values=0)
    
    t.print("      Determining no data mask ...")
    mask = (padded_sat_image != 0).astype(np.uint8)  # NoData value is 0
    
    # save memory
    del padded_sat_image, first_band

    t.print("      Initial erosion")
    # Perform initial erosion to ensure thin borders don't get lost
    initial_dilation_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width+output_mask_size_ratio, border_width+output_mask_size_ratio))
    mask = cv2.erode(mask, initial_dilation_structure, iterations=1)
    
    # Trim back to original size (remove padding)
    mask = mask[border_width:-border_width, border_width:-border_width]

    t.print("      Downsampling mask by 2 ...")
    # Downsample by a factor of 2
    downsampled_mask = zoom(mask, (0.5, 0.5), order=0)

    # Subsample the mask with a smaller erosion
    downsample_ratio = 3
    t.print(f"      Erosion by {downsample_ratio} pixels...")
    downsampled_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (downsample_ratio, downsample_ratio))
    downsampled_mask = cv2.erode(downsampled_mask, downsampled_structure, iterations=1)

    t.print(f"      Downsampling mask by {downsample_ratio} ...")
    downsampled_mask = zoom(downsampled_mask, (1 / downsample_ratio, 1 / downsample_ratio), order=0)
    
    erosion_pixels = math.ceil(trim_pixels / (output_mask_size_ratio * downsample_ratio))
    t.print(f"      Erosion by {erosion_pixels} pixels ...")

    # Perform final erosion based on the trim_pixels
    final_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_pixels, erosion_pixels))
    downsampled_mask = cv2.erode(downsampled_mask, final_structure, iterations=1)

    t.print("    Clipping mask total time:")
    t.last()
    return downsampled_mask
    
def gaussian_filter_nan_cv2(image, sigma=5):
    """
    Apply a Gaussian filter, but ignore NaN areas. This is non-standard.
    Having the filter ignore NaN values allows the filtering as an interpolator
    to infill masked out areas. The OpenCV GaussianBlur doesn't support NaN areas, 
    so we convert them to 0 (black), apply the gaussian filtering, then scale the result 
    by the number of valid pixels that contributed to the filtered pixel.
    
    Args:
        image (numpy.ndarray): Image to apply the blurring to.
        sigma (int): Gaussian fall off of the filter kernel, dropping to 
            approximately 60.7% of the peak at 1 sigma, 13.5% at 2 sigma, 
            and around 0.3% at 3 sigma. Units in pixels.
    
    Returns:
        numpy.ndarray: The filtered image with NaN areas preserved.
    """
    # Create a mask where NaN values are present
    nan_mask = np.isnan(image)

    # Replace NaNs with 0 for OpenCV processing
    image_temp = np.nan_to_num(image, nan=0)

    # Create a valid mask (where values are not NaN)
    valid_mask = (~nan_mask).astype(np.float32)

    # Apply Gaussian blur to the valid mask to get the weighting factor
    weights = cv2.GaussianBlur(valid_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Apply Gaussian blur to the image (ignoring NaNs which are now 0s)
    filtered = cv2.GaussianBlur(image_temp.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Normalize the result by dividing by the weights
    with np.errstate(divide='ignore', invalid='ignore'):
        result = filtered / weights
        # Since the gaussian filter expands to many sigmas (~6) the signal
        # stops at the floating point precision. To crop the range of the
        # interpolation to a nicer edge we need to back away from the 
        # precision limit. The precision limit seems to occur at approx 1e-7
        # so trim the filtering to 1e-6.
        result[weights < 1e-6] = np.nan  # Restore NaNs where appropriate

    return result, weights
    
    
import os
import glob

def prepare_sentinel_image_list(base_path, split, index, priority_scenes, justpriority=False):
    """
    Prepare a list of all the file paths of the GeoTiffs in the base_path folder,
    taking a subset corresponding to the split and index, and making sure those
    scenes specified in the priority_scenes are in the list first, in the specified order.
    
    Args:
        base_path (str): Folder containing the GeoTiffs
        split (int): Number of subsets to divide the processing across.
        index (int): Index of the subset to process (starting from 0).
        priority_scenes (list of str): List of scene codes to process first, in order.
        justpriority (bool): If True, only the priority scenes will be processed.
    
    Returns:
        List of file paths of GeoTiffs in order of processing
    """
    # List all TIFF files in the directory
    all_tiff_files = sorted(glob.glob(os.path.join(base_path, '*.tif')))
    
    # Extract scene codes
    all_scene_codes = [extract_scene_code(os.path.basename(f)) for f in all_tiff_files]
    all_scene_codes = list(dict.fromkeys(all_scene_codes))  # Remove duplicates while maintaining order
    
    # Handle justpriority option
    if justpriority:
        # Make sure the priority_scenes match those available.
        priority_scene_codes = [code for code in priority_scenes if code in all_scene_codes]
        priority_files = [f for f in all_tiff_files if extract_scene_code(os.path.basename(f)) in priority_scene_codes]
        return [f for i, f in enumerate(priority_files) if i % split == index]
    
    # Separate files into priority and non-priority
    priority_files = [f for f in all_tiff_files if extract_scene_code(os.path.basename(f)) in priority_scenes]
    non_priority_files = [f for f in all_tiff_files if extract_scene_code(os.path.basename(f)) not in priority_scenes]

    # Distribute files to processes, taking every Nth file according to split and index
    priority_assigned = [f for i, f in enumerate(priority_files) if i % split == index]
    non_priority_assigned = [f for i, f in enumerate(non_priority_files) if i % split == index]
    
    # Combine the assigned files with priority files first
    tiff_files = priority_assigned + non_priority_assigned
    
    return tiff_files

def rasterize_shapefile(shapefile_polygons, raster_profile, output_raster_path):
    """Rasterize a shapefile based on a provided raster profile and save it with optional buffering.
    This is used to convert the land and shallow reef shapefiles to rasters that align with the 
    pixels of the Sentinel 2 imagery.
    
    Args:
         shapefile_polygons (Geopandas polygons): Polygons to rasterise
         raster_profile (rasterio profile dict): Profile of the raster to rasterise to. This will
            correspond to the profile of the read GeoTiff Sentinel 2 tile we are processing.
         output_raster_path (str): Where to save the resulting raster. If the file already
            exists then no save will be attempted.
         buffer_pixels (int): Number of pixels to apply buffering to the mask.
    
    Returns:
        The rasterised image
    """

    # Use the provided raster profile
    out_meta = raster_profile.copy()
    out_meta.update({
        'count': 1,  # Assuming you want a single mask layer
        'dtype': 'uint8',
        'compress': 'lzw'  # Adding LZW compression
    })
    
    # Rasterize the shapes into a new raster array
    out_image = rasterize(
        [geometry for geometry in shapefile_polygons.geometry],
        out_shape=(raster_profile['height'], raster_profile['width']),
        transform=raster_profile['transform'],
        fill=0,  # Fill value for 'background'
        all_touched=True,  # Include all pixels touched by geometries
        dtype='uint8',
        default_value = 255
    )

    # Save the rasterized shapefile to a new raster file. 
    # If we have two scripts running in parallel then we have situations
    # where both scripts are writing to the same file, or one is
    # reading the partially written file. To make this more robust
    # we write the file to a temporary file (with a randomised number
    # in the name) in the same output folder, then once that is done, 
    # we move the file to its final file name. This means that the 
    # final file should not exist until it is ready to be read. If
    # two threads attempt to write to same filename then the one
    # that performs the move last will be the final file. In our case
    # the rasterisations should be identical, so this shouldn't cause
    # a problem.
    
    # Generate a short random suffix
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    temp_raster_path = f"{output_raster_path}.{random_suffix}.tmp"

    # Save the rasterized shapefile to the temporary file
    with rasterio.open(temp_raster_path, 'w', **out_meta) as out_raster:
        out_raster.write(out_image, 1)

    # Move the temporary file to the final destination with race condition handling
    # If a separate process also created the destination file then
    # performing the rename will fail as the file will already exist.
    
    try:
        os.rename(temp_raster_path, output_raster_path)
    except FileExistsError:
        # Another process has already processed this image.
        # Remove the temporary file since it's no longer needed.
        os.remove(temp_raster_path)

    
    return out_image