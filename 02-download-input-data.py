from data_downloader import DataDownloader
from pyproj import CRS
import os

# Create an instance of the DataDownloader class
downloader = DataDownloader(download_path="data/in-3p")
VERSION = '1-1'

print("Downloading source data files. This will take a while ...")

# --------------------------------------------------------
# Function to replace the .prj file with WKT2 EPSG:4283
def replace_prj_with_epsg_4283(shapefile_path):
    prj_file = os.path.splitext(shapefile_path)[0] + ".prj"
    print(f"Replacing .prj file at {prj_file} with EPSG:4283 WKT2...")
    # Generate WKT2 string for EPSG:4283 using pyproj
    wkt_content = CRS.from_epsg(4283).to_wkt("WKT2_2019")
    with open(prj_file, "w") as f:
        f.write(wkt_content)
    print(".prj file successfully replaced!")
    
# --------------------------------------------------------
# Australian Coastline 50K 2024 (NESP MaC 3.17, AIMS)
# https://eatlas.org.au/geonetwork/srv/eng/catalog.search#/metadata/c5438e91-20bf-4253-a006-9e9600981c5f
# Hammerton, M., & Lawrey, E. (2024). Australian Coastline 50K 2024 (NESP MaC 3.17, AIMS) (2nd Ed.) [Data set]. eAtlas. https://doi.org/10.26274/qfy8-hj59
direct_download_url = 'https://nextcloud.eatlas.org.au/s/DcGmpS3F5KZjgAG/download?path=%2FV1-1%2F&files=Split'
downloader.download_and_unzip(direct_download_url, 'AU_AIMS_Coastline_50k_2024', subfolder_name='Split', flatten_directory=True)

# Use this version for overview maps
direct_download_url = 'https://nextcloud.eatlas.org.au/s/DcGmpS3F5KZjgAG/download?path=%2FV1-1%2F&files=Simp'
downloader.download_and_unzip(direct_download_url, 'AU_AIMS_Coastline_50k_2024', subfolder_name='Simp', flatten_directory=True)

# --------------------------------------------------------
#Lawrey, E. P., Stewart M. (2016) Complete Great Barrier Reef (GBR) Reef and Island Feature boundaries including Torres Strait (NESP TWQ 3.13, AIMS, TSRA, GBRMPA) [Dataset]. Australian Institute of Marine Science (AIMS), Torres Strait Regional Authority (TSRA), Great Barrier Reef Marine Park Authority [producer]. eAtlas Repository [distributor]. https://eatlas.org.au/data/uuid/d2396b2c-68d4-4f4b-aab0-52f7bc4a81f5
direct_download_url = 'https://nextcloud.eatlas.org.au/s/xQ8neGxxCbgWGSd/download/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.zip'
downloader.download_and_unzip(direct_download_url, 'GBR_AIMS_Complete-GBR-feat_V1b')

# Replace the .prj file for the GBR shapefile because it is using an older WKT format
# that fails to load correctly in stages 3.
gbr_shapefile = os.path.join(downloader.download_path, 'GBR_AIMS_Complete-GBR-feat_V1b', 'TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp')
replace_prj_with_epsg_4283(gbr_shapefile)


# --------------------------------------------------------
# McNeil, M.A., Webster, J.M., Beaman, R.J. et al. New constraints on the spatial distribution and morphology of the Halimeda bioherms of the Great Barrier Reef, Australia. Coral Reefs 35, 1343–1355 (2016). https://doi.org/10.1007/s00338-016-1492-2
# https://static-content.springer.com/esm/art%3A10.1007%2Fs00338-016-1492-2/MediaObjects/338_2016_1492_MOESM1_ESM.zip
# Also available from https://geoserver.imas.utas.edu.au/geoserver/seamap/wfs?version=1.0.0&request=GetFeature&typeName=SeamapAus_QLD_Halimeda_bioherms_2016&outputFormat=SHAPE-ZIP
# https://metadata.imas.utas.edu.au/geonetwork/srv/eng/catalog.search#/metadata/b8475bea-e24e-4374-8090-ef06514b951d
direct_download_url = 'https://static-content.springer.com/esm/art%3A10.1007%2Fs00338-016-1492-2/MediaObjects/338_2016_1492_MOESM1_ESM.zip'
downloader.download_and_unzip(direct_download_url, 'GBR_USYD_Halimeda-bioherms_2016', flatten_directory=True)

downloader.download_path = "in-data"
# --------------------------------------------------------
# The rough reef mask corresponds to the water estimate
# masking created for the creation of this dataset

direct_download_url = f'https://nextcloud.eatlas.org.au/s/iMrFB9WP9EpLPC2/download?path=%2FV{VERSION}%2Fin-data%2FAU_Rough-reef-shallow-mask'
downloader.download_and_unzip(direct_download_url, 'AU_Rough-reef-shallow-mask', flatten_directory=True)

direct_download_url = f'https://nextcloud.eatlas.org.au/s/iMrFB9WP9EpLPC2/download?path=%2FV{VERSION}%2Fin-data%2FAU_Cleanup-remove-mask'
downloader.download_and_unzip(direct_download_url, 'AU_Cleanup-remove-mask', flatten_directory=True)


