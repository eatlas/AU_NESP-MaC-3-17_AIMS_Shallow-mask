import os
import geopandas as gpd
import pandas as pd
"""
This script combines two shapefiles to create a unified rough mask of shallow areas 
and reefs for Northern Australia and the Great Barrier Reef (GBR). The process includes 
filtering, buffering, simplifying, and merging operations to harmonize data from 
different scales and projections. The output is a simplified shapefile containing 
only geometry (no attributes), suitable for further spatial analysis or visualization.

Input 1 (GBR-features): Shapefile of reef boundaries in the GBR and Torres Strait at 1:200k scale.
Input 2 (Rough-reef-mask): Shapefile of shallow areas and reefs across Northern Australia at 1:500k scale.

Output: A single shapefile (EPSG:4326) with combined and simplified geometries saved to:
'working-data/AU_Rough-reef-shallow-mask-with-GBR.shp'
"""

def main():
    # Define file paths
    input1 = 'in-data-3p/GBR_AIMS_Complete-GBR-feat_V1b/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp'
    input2 = 'in-data/AU_Rough-reef-shallow-mask/AU_AIMS_NESP-MaC-3-17_Rough-reef-shallow-mask_Base.shp'
    output_path = 'working-data/03-rough-reef-mask_poly/AU_Rough-reef-shallow-mask-with-GBR.shp'
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print("Loading GBR features shapefile...")
    gbr = gpd.read_file(input1)
    
    # Remove the main land feature from the mask
    print("Filtering out mainland features...")
    gbr = gbr[gbr['FEAT_NAME'] != 'Mainland']
    
    # Buffer by a small amount to merge features close together
    # A 0.001 degree buffer corresponds to approximately 110 m
    # buffer.
    print("Applying small positive buffer...")
    gbr['geometry'] = gbr['geometry'].buffer(0.001)
    
    # Dissolve to combine any overlapping features together
    print("Dissolving shapes...")
    gbr = gbr.dissolve()
    
    # Negative buffer to strink back to almost normal size.
    # Make it slightly less so there is a small buffer around each
    # feature to cover small errors in the masking. 
    print("Applying negative buffer ...")
    gbr['geometry'] = gbr['geometry'].buffer(-0.0009)
    
    # Simplify with Douglas-Peucker 0.0002 degrees. This makes the 
    # reef boundaries a similar vertex density as the northern Australian
    # shallow reef mask. This simplification makes the dataset smaller
    # and allows the rendering of the mask faster.
    print("Simplifying geometry...")
    gbr['geometry'] = gbr['geometry'].simplify(0.0002)
    
    # By default the dissolve makes everything a single multi-part polygon
    # This makes performance poor as no polygons can be excluded by their
    # bounding box. Converting to single part splits the data back into
    # many separate polygons, improving performance.
    print("Converting multi-part geometries to single-part...")
    gbr = gbr.explode(index_parts=False)
    
    # Load the manual rough masking for Northern Australia
    print("Loading Rough reef mask shapefile...")
    rough_reef = gpd.read_file(input2)
    
    # Ensure both datasets are in the same CRS (EPSG:4326). GBR is GDA94.
    print("Reprojecting datasets to EPSG:4326...")
    gbr = gbr.to_crs(epsg=4326)
    rough_reef = rough_reef.to_crs(epsg=4326)
    
    # Merge all features into one dataset. The rough masking will overlap
    # with parts of the GBR mapping in Torres Strait and GBR.
    print("Merging datasets...")
    combined = gpd.GeoDataFrame(pd.concat([rough_reef, gbr], ignore_index=True), crs=rough_reef.crs)
    
    # Get rid of overlaps
    print("Dissolving all features to eliminate overlaps...")
    combined = combined.dissolve()
    
    # Multi-part to single-part to improve performance
    print("Converting multi-part geometries to single-part for improved performance...")
    combined = combined.explode(index_parts=False)
    
    # We are making a mask and so the attributes are meaningless and increase the file size.
    print("Removing attributes, keeping only geometry...")
    combined = combined[['geometry']]
    
    # Save as Rough-reef-mask-with-GBR
    print(f"Saving output shapefile to {output_path}...")
    combined.to_file(output_path)
    print("Processing complete.")

if __name__ == "__main__":
    main()
