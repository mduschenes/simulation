##!/bin/bash
##
## cf https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo
cwd=$(pwd)

# MKL
cd /tmp
sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB


## all products:
#sudo wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
## just MKL
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'

sudo apt-get update
apt-get install intel-mkl-64bit-2020.4-912
apt-get install intel-openmp-19.1.3-304 libiomp5



## update alternatives
sudo update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
sudo update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
sudo update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
sudo update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu  /opt/intel/mkl/lib/intel64/libmkl_rt.so 150

sudo sh -c 'echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf'
sudo sh -c 'echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf'
ldconfig

echo "MKL_THREADING_LAYER=GNU" >> /etc/environment
echo "MKLROOT=/opt/intel/mkl" >> /etc/environment
echo "OMP_NUM_THREADS=8" >> /etc/environment

sudo bash /opt/intel/mkl/bin/mklvars.sh intel64


# HDF5

sudo apt-get install -y libhdf5-dev 
gcc -I/usr/include/hdf5/serial -Wdate-time -D_FORTIFY_SOURCE=2 -g -O2 -fdebug-prefix-map=/build/hdf5-X9JKIg/hdf5-1.10.0-patch1+docs=. -fstack-protector-strong -Wformat -Werror=format-security -L/usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.a -Wl,-Bsymbolic-functions -Wl,-z,relro -lpthread -lsz -lz -ldl -lm -Wl,-rpath -Wl,/usr/lib/x86_64-linux-gnu/hdf5/serial



# Conda

# Intel Python
src=setup
pyversion=3.8
env=mkl
channel=intel

# Numpy/Scipy from source

# root=/usr/lib/python${pyversion}

# # Pip and cython
# url=https://bootstrap.pypa.io/get-pip.py
# file=get-pip.py
# sudo wget ${url} -o ${file}
# sudo python${pyversion} ${file}
# sudo python${pyversion} -m pip install cython

# sudo apt-get install python${pyversion}-dev
# conda install --channel intel cython pythran pybind11


# folder=dist-packages
# sudo mkdir ${root}/${folder}

# # Numpy
# pkg=${scipy}
# url=https://github.com/numpy/numpy.git
# sudo git clone ${url} ${root}/${folder}/${pkg}

# cd ${root}/${folder}/${pkg}

# ./setup.py build
# ./setup.py install --prefix=${root}/${folder}/${pkg}

# # Scipy
# pkg=${scipy}
# url=https://github.com/scipy/scipy.git
# sudo git clone ${url} ${root}/${folder}/${pkg}

# cd ${root}/${folder}/${pkg}

# ./setup.py build
# ./setup.py install --prefix=${root}/${folder}/${pkg}




# Python Packages
cd ${cwd}

cd ${src}

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate base
conda remove --name ${env} --all
rm lib/tenpy -rf
conda clean --all


conda create --name ${env}
conda activate ${env}
conda info --envs

conda install --channel ${channel} --file requirements.py.txt
conda install --channel ${channel} --file requirements.mkl.txt
conda install --channel ${channel} --file requirements.npy.txt
pip install -r requirements.pip.txt
conda install --file requirements.pkg.txt
# conda install --channel ${channel} --no-update-deps --file requirements.tenpy.txt

cd ${cwd}


pkg=tenpy
url=https://github.com/tenpy/tenpy.git 
git clone ${url} ${src}/${pkg}
cd ${src}/${pkg}
bash ./compile.sh
python${pyversion} setup.py install

export PYTHONPATH="$PYTHONPATH:${src}"


cd ${cwd}

src=test
tests=.
options=(-s)
cd ${src}
pytest ${options[@]} ${tests}

cd ${cwd}

