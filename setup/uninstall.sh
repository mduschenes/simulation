#!/bin/bash
##
## cf https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo

# MKL
apt-get autoremove intel-mkl-64bit-2020.4-912
apt-get purge intel-mkl-64bit-2020.4-912
rm -rf ${MKLROOT}

sudo apt-get autoremove intel-openmp-19.1.3-304 libiomp5
sudo apt-get purge intel-openmp-19.1.3-304 libiomp5

sudo apt-get autoremove intelpython
sudo apt-get purge intelpython3


# HDF5
sudo apt-get purge libhdf5-dev



# Conda
env=tensor
conda activate base

conda remove --name ${env} --all
