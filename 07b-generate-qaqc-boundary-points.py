"""
Script: 07b-generate-qaqc-boundary-points.py

Description:
This script generates random QAQC points along the outer edges of polygons in a given shapefile,
excluding points that overlap with a specified coastline polygon representing land areas. The
position of points does not align with vertices of the input file, but the boundary edge. This
was to ensure a more accurate assessment as we found that vertices tend to be more accurate
that the lines between them for roughly digitised features. 

The density of the points is proportional to the perimeter of the mapped features. This means
that areas with many features will have more points assigned to them.


The script outputs a shapefile that contains an initial set of Ground Truth Points (gtp) that
should be adjusted to the most accurate estimate of the true edge of the boundary being mapped.

Arguments:
    --input       Path to the input polygon shapefile. 
    --output      Path to save the output points to. 
    --coastline   Path to the coastline polygon shapefile
    --num_points  Number of random points to generate (default: 100)

Example:
    python 07b-generate-qaqc-boundary-points.py --input working-data/03-rough-reef-mask_poly/AU_Rough-reef-shallow-mask-with-GBR.shp --output in-data/AU_Shallow-mask-boundary_qaqc/AU_Shallow-mask_gtp_B1.shp --num_points 250
"""
import geopandas as gpd
import random
from shapely.geometry import Point
import os
import sys
import argparse

# Constant to control print rate
PRINT_RATE = 20  # Print progress every 5 valid points
MAX_DITHERING_DEG = 100/40075000*360 # distance(m)/circumference of earth at equator*360 degrees

def generate_points_along_edges(input_shapefile, coastline_shapefile, num_points):
    # Load the input shapefile and coastline
    print("Reading in shapefile to assess")
    gdf = gpd.read_file(input_shapefile)
    print("Reading in coastline dataset")
    coastline_gdf = gpd.read_file(coastline_shapefile)
    
    # Collect all line segments from the polygons
    print("Collecting all line segments from polygons")
    all_segments = []
    for index, row in gdf.iterrows():
        geom = row.geometry
        if geom.type == 'Polygon':
            all_segments.extend(list(zip(geom.exterior.coords[:-1], geom.exterior.coords[1:])))
        elif geom.type == 'MultiPolygon':
            for part in geom.geoms:
                all_segments.extend(list(zip(part.exterior.coords[:-1], part.exterior.coords[1:])))
        else:
            print(f"Unsupported geometry type: {geom.type}")
    
    # Calculate the lengths of all segments
    segment_lengths = [Point(seg[0]).distance(Point(seg[1])) for seg in all_segments]
    total_length = sum(segment_lengths)
    
    # Generate random points along the line segments
    print(f"Finding {num_points} random points")
    valid_points = []
    ids = []
    attempts = 0
    while len(valid_points) < num_points and attempts < num_points * 10:
        # Choose a segment randomly, weighted by its length
        segment = random.choices(all_segments, weights=segment_lengths, k=1)[0]
        # Generate a random point along the segment
        t = random.random()
        x = segment[0][0] + t * (segment[1][0] - segment[0][0])
        y = segment[0][1] + t * (segment[1][1] - segment[0][1])
        point = Point(x, y)
        
        # Check if the point overlaps with the coastline
        if not coastline_gdf.contains(point).any():
            valid_points.append(point)
            ids.append(f"{len(valid_points)}")
            
            # Print progress
            if len(valid_points) % PRINT_RATE == 0:
                print(f"{len(valid_points)} valid points generated.")
        attempts += 1
    
    if len(valid_points) < num_points:
        print(f"Could only find {len(valid_points)} valid points after {attempts} attempts.")
    else:
        print(f"Successfully generated {len(valid_points)} valid points.")
    
    # Create GeoDataFrame for reference points
    reference_points_gdf = gpd.GeoDataFrame(
        {"ID": ids, "geometry": valid_points}, crs=gdf.crs
    )
    
    return reference_points_gdf

def add_dithering(gdf, max_distance_deg):
    """
    Apply random dithering to each point in the GeoDataFrame for EPSG:4326 CRS.
    
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame with point geometries.
    max_distance_deg (float): Maximum random displacement in degrees ( 0.0009 = ~100 m at equator).
    
    Returns:
    GeoDataFrame: A new GeoDataFrame with dithered points.
    """
    print("Applying random dithering to points in EPSG:4326")
    dithered_geometries = []
    
    for point in gdf.geometry:
        # Random displacement in longitude and latitude
        dx = random.uniform(-max_distance_deg, max_distance_deg)
        dy = random.uniform(-max_distance_deg, max_distance_deg)
        dithered_point = Point(point.x + dx, point.y + dy)
        dithered_geometries.append(dithered_point)
    
    # Return a new GeoDataFrame
    return gpd.GeoDataFrame({"ID": gdf["ID"], "geometry": dithered_geometries}, crs=gdf.crs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random points for QAQC.")
    parser.add_argument(
        "--input", 
        default="in-data/AU_Rough-reef-shallow-mask/AU_AIMS_NESP-MaC-3-17_Rough-reef-shallow-mask_Base.shp",
        help="Path to the input shapefile."
    )
    parser.add_argument(
        "--output", 
        default="in-data/AU_Shallow-mask-boundary_qaqc/AU_Shallow-mask_gtp_B1.shp",
        help="Path and filename to the output points shapefile to generate. These should be adjusted after generation to be Ground Truth Points."
    )
    parser.add_argument(
        "--coastline",
        default="in-data-3p/AU_AIMS_Coastline_50k_2024/Split/AU_NESP-MaC-3-17_AIMS_Aus-Coastline-50k_2024_V1-1_split.shp",
        help="Path to the coastline shapefile."
    )
    parser.add_argument("--num_points", type=int, default=100, help="Number of random points.")
    args = parser.parse_args()

    if os.path.isfile(args.output):
        print(f"Output points file already exists {args.output}. Cancelling to prevent overwriting. Delete to regenerate points")
        sys.exit()
        
    input_shapefile = args.input
    coastline_shapefile = args.coastline
    num_points = args.num_points

    # Generate points
    reference_points_gdf = generate_points_along_edges(input_shapefile, coastline_shapefile, num_points)
    
    # Add dithering to create match points
    gtp_points_gdf = add_dithering(reference_points_gdf, max_distance_deg=MAX_DITHERING_DEG)
    
    # Define output paths
    base_filename = os.path.splitext(os.path.basename(input_shapefile))[0]
    
    
    
    # Save a copy for match points
    gtp_file = args.output
    os.makedirs(os.path.dirname(gtp_file), exist_ok=True)
    gtp_points_gdf.to_file(gtp_file)
    
    print(f"Initial Ground Truth Points (gtp) file: {gtp_file}")
    print(f"The ground truth points must be manually adjusted to align to the best estimate of the true boundary")