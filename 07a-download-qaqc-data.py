from data_downloader import DataDownloader
import os
"""
This script downloads additional data files used for performing QAQC on the shallow masks.
python 07a-download-qaqc-data.py
"""
DOWNLOAD_PATH = "in-data-3p"
# Create an instance of the DataDownloader class
downloader = DataDownloader(download_path=DOWNLOAD_PATH)


print("Downloading additional source data files used for QAQC. This will take a while ...")

# Twiggs, E. 2023. Kimberley Region and WA Reefs Satellite-Derived Bathymetry Acquisition (20210024S). Geoscience Australia, Canberra. https://dx.doi.org/10.26186/148669
direct_download_url = 'https://files.ausseabed.gov.au/survey/Kimberley%20Region%20and%20WA%20Reefs%20Satellite-Derived%20Bathymetry%202021%2010m.zip'
downloader.download_and_unzip(direct_download_url, 'Kim_GA_SDB_2021')


# Beaman, R. 2023. AusBathyTopo (Torres Strait) 30m 2023 - A High-resolution Depth Model (20230006C). Geoscience Australia, Canberra. https://dx.doi.org/10.26186/144348
direct_download_url = 'https://files.ausseabed.gov.au/survey/Australian%20Bathymetry%20Topography%20(Torres%20Strait)%202023%2030m.zip'
downloader.download_and_unzip(direct_download_url, 'TS_GA_AusBathyTopo-30m_2023')

# Beaman, R.J. 2017. AusBathyTopo (Great Barrier Reef) 30m 2017 - A High-resolution Depth Model (20170025C). Geoscience Australia, Canberra. http://dx.doi.org/10.4225/25/5a207b36022d2
direct_download_url = 'https://files.ausseabed.gov.au/survey/Great%20Barrier%20Reef%20Bathymetry%202020%2030m.zip'
downloader.download_and_unzip(direct_download_url, 'GBR_GA_Bathymetry-30m_2017')


# Lebrec, U., Paumard, V., O'Leary, M. J., and Lang, S. C. 2021. Towards a regional high-resolution bathymetry of the North West Shelf of Australia based on Sentinel-2 satellite images, 3D seismic surveys and historical datasets. Earth Syst. Sci. Data Discuss. https://doi.org/10.5194/essd-13-5191-2021
# Lebrec, U. 2021. A High-resolution depth model for the North West Shelf and Outer Browse Basin (20210025C). Geoscience Australia, Canberra. https://doi.org/10.26186/144600

# Download the North West Shelf DEM Compilation (tif) [4 GB]
direct_download_url = 'https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/1d6f5de4-84ff-4177-a90d-77997ba8d688/North_West_Shelf_DEM_v2_Bathymetry_2020_30m_MSL_cog.tif'
downloader.download(direct_download_url, f'{DOWNLOAD_PATH}/WA_GA_NWS-DEM_2020/North_West_Shelf_DEM_v2_Bathymetry_2020_30m_MSL_cog.tif')


