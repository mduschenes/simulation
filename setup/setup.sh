#!/bin/bash

cwd=$(pwd)

src=setup
pyversion=3.8
env=mkl
channel=intel

# Setup conda
source ~/miniconda3/etc/profile.d/conda.sh 
eval $(conda shell.bash hook)

# Clean conda
source ~/miniconda3/bin/activate base
conda remove --name ${env} --all
sudo rm -rf ${cwd}/lib/tenpy
conda clean --all

# Create conda env
conda create --name ${env}
source ~/miniconda3/bin/activate ${env}
conda info --envs

# Install packages
cd ${src}
conda install --channel ${channel} --file requirements.py.txt
conda install --channel ${channel} --file requirements.mkl.txt
conda install --channel ${channel} --file requirements.npy.txt
conda install --file requirements.pkg.txt
pip install -r requirements.pip.txt
# conda install --channel ${channel} --no-update-deps --file requirements.tenpy.txt
cd ${cwd}



# Install tenpy
src=lib
pkg=tenpy
url=https://github.com/tenpy/tenpy.git 

source ~/miniconda3/bin/activate ${env}

sudo rm -rf ${src}/${pkg}
git clone ${url} ${src}/${pkg}
cd ${src}/${pkg}
export PYTHONPATH="$PYTHONPATH:${cwd}/${src}"
git checkout v0.9.0
bash ./compile.sh
sudo ~/miniconda3/envs/${env}/bin/python${pyversion} setup.py install

source ~/miniconda3/bin/activate ${env}
bash ./compile.sh


cd ${cwd}


# Test
src=test
tests=.
options=(-s)
cd ${src}
conda activate ${env}
pytest ${options[@]} ${tests}

cd ${cwd}
