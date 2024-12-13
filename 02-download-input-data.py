from data_downloader import DataDownloader
import os

# Create an instance of the DataDownloader class
downloader = DataDownloader(download_path="in-data-3p")


print("Downloading source data files. This will take a while ...")

# --------------------------------------------------------
# Australian Coastline 50K 2024 (NESP MaC 3.17, AIMS)
# https://eatlas.org.au/geonetwork/srv/eng/catalog.search#/metadata/c5438e91-20bf-4253-a006-9e9600981c5f
direct_download_url = 'https://nextcloud.eatlas.org.au/s/DcGmpS3F5KZjgAG/download?path=%2FV1-1%2F&files=Split'
downloader.download_and_unzip(direct_download_url, 'AU_AIMS_Coastline_50k_2024', subfolder_name='Split', flatten_directory=True)

# Use this version for overview maps
direct_download_url = 'https://nextcloud.eatlas.org.au/s/DcGmpS3F5KZjgAG/download?path=%2FV1-1%2F&files=Simp'
downloader.download_and_unzip(direct_download_url, 'AU_AIMS_Coastline_50k_2024', subfolder_name='Simp', flatten_directory=True)

# --------------------------------------------------------
#Lawrey, E. P., Stewart M. (2016) Complete Great Barrier Reef (GBR) Reef and Island Feature boundaries including Torres Strait (NESP TWQ 3.13, AIMS, TSRA, GBRMPA) [Dataset]. Australian Institute of Marine Science (AIMS), Torres Strait Regional Authority (TSRA), Great Barrier Reef Marine Park Authority [producer]. eAtlas Repository [distributor]. https://eatlas.org.au/data/uuid/d2396b2c-68d4-4f4b-aab0-52f7bc4a81f5
direct_download_url = 'https://nextcloud.eatlas.org.au/s/xQ8neGxxCbgWGSd/download/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.zip'
downloader.download_and_unzip(direct_download_url, 'GBR_AIMS_Complete-GBR-feat_V1b')


# --------------------------------------------------------
# The rough reef mask corresponds to the water estimate
# masking created for the creation of this dataset
direct_download_url = 'https://nextcloud.eatlas.org.au/s/iMrFB9WP9EpLPC2/download?files=AU_Rough-reef-shallow-mask'
downloader.download_and_unzip(direct_download_url, 'AU_Rough-reef-shallow-mask', flatten_directory=True)


#direct_download_url = 'https://nextcloud.eatlas.org.au/s/iMrFB9WP9EpLPC2/download?path=%2F%2FAU_Cleanup-remove-mask'
direct_download_url = 'https://nextcloud.eatlas.org.au/s/iMrFB9WP9EpLPC2/download?files=AU_Cleanup-remove-mask'
downloader.download_and_unzip(direct_download_url, 'AU_Cleanup-remove-mask', flatten_directory=True)

print("The Kimberley Region and WA Reefs Satellite-Derived Bathymetry Acquisition (20210024S) dataset is used for QAQC")
print("This dataset must be downloaded manually because GA do not provide programmatic download")
print("https://dx.doi.org/10.26186/148669")
print("Save to in-data/WA_GA_Kimberley-SDB_2021/")
