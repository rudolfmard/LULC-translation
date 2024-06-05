#!/usr/bin/bash
#
# Helper for the download and set up of the data of ECOCLIMAP-SG-ML
#

# To be executed to the directory where the data will be stored (with at least 25GB per member)

set -vx
mkdir outputs

# 1. Weights
mkdir saved_models
cd saved_models
wget https://zenodo.org/records/11242911/files/mmt-weights-2.0.ckpt
wget https://zenodo.org/records/11242911/files/mmt-weights-2.0.config.yaml
echo "Weights downloaded in $PWD"
cd ..

# 2. ECOSG-ML files
mkdir tiff_data
mkdir tiff_data/ECOCLIMAP-SG-ML
cd tiff_data/ECOCLIMAP-SG-ML
for MB in {0..5};
do
        wget "https://zenodo.org/records/11242911/files/ecosgml-v2.0-mb00$MB.zip"
done
echo "ECOSG-ML TIF files downloaded in $PWD. Now unzipping"
for ZIP in `ls *.zip`
do
        unzip $ZIP -d ${ZIP::-4}
done
echo "ECOSG-ML TIF files unzipped"
rm *.zip
cd ../..

# 3. Training/testing data
wget https://zenodo.org/records/11242911/files/hdf5-v2.0.zip
echo "HDF5 files downloaded in $PWD. Now unzipping"
unzip -j hdf5-v2.0.zip -d hdf5_data
echo "HDF5 files unzipped"
rm hdf5-v2.0.zip


# 4. ESA WorldCover
cd tiff_data
wget https://zenodo.org/record/7254221/files/ESA_WorldCover_10m_2021_v200_60deg_macrotile_S30W060.zip
wget https://zenodo.org/record/7254221/files/ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30W060.zip
wget https://zenodo.org/record/7254221/files/ESA_WorldCover_10m_2021_v200_60deg_macrotile_S30E000.zip
wget https://zenodo.org/record/7254221/files/ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30E000.zip
echo "ESA WorldCover TIF files downloaded in $PWD. Now unzipping"
unzip '*.zip' -d ESA-WorldCover-2021
echo "ESA WorldCover TIF files unzipped"
rm *.zip
cd ..

# 5. ECOCLIMAP-SG
mkdir tiff_data/ECOCLIMAP-SG
echo "To download ECOCLIMAP-SG, execute from the package root directory: python scripts/download_ecoclimapsg.py --landingdir $PWD/tiff_data/ECOCLIMAP-SG"

# 6. ECOCLIMAP-SG
mkdir tiff_data/ECOCLIMAP-SG-plus
cd tiff_data/ECOCLIMAP-SG-plus
wget https://zenodo.org/records/10944693/files/best-guess_map.zip
unzip -j best-guess_map.zip -d bguess-ecosgp-v2.0
wget https://zenodo.org/records/10944693/files/quality_score_map.zip
unzip -j quality_score_map.zip -d qscore-ecosgp-v2.0

echo "The data directory is now ready. You may now create a symbolic link in the package directory: ln -s $PWD data"
