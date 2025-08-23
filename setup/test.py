#!/usr/bin/env python

import pytest

@pytest.fixture(autouse=True,scope='session')
def cleanup(*args,**kwargs):
	import os
	directories = ['__pycache__','.pytest_cache']
	for directory in directories:
		os.system('rm -rf %s'%(directory))	
	return


def test_python(*args,**kwargs):
	import os
	os.system('python -V')
	return

def test_numpy(*args,**kwargs):
	import numpy as np
	np.show_config()
	return

def test_jax(*args,**kwargs):

	import os
	import traceback


	def func(device):

		def func(device):
			array = np.array([1,2,3])
			array *= array
			print(device,':::',xla_bridge.get_backend().platform,xla_bridge.get_backend().devices(),':::',array.devices())
			return

		if isinstance(device,str) and device.count(':'):
			platform = 'gpu'
		elif isinstance(device,str):
			platform = device
		else:
			platform = str(device).split(':')[0]

		environs = {
			'JAX_DISABLE_JIT':False,
			'JAX_PLATFORMS':platform,
			'JAX_PLATFORM_NAME':platform,
			'JAX_CUDA_VISIBLE_DEVICES':os.environ.get('JAX_CUDA_VISIBLE_DEVICES'),
			'JAX_ENABLE_X64':True,
			'TF_CPP_MIN_LOG_LEVEL':5
			}
		for name in environs:
			if environs[name] is None:
				continue
			os.environ[name],environs[name] = str(environs[name]),os.environ.get(name)
		
		import jax
		from jax.lib import xla_bridge
		import jax.numpy as np
	
		configs = {
			'jax_disable_jit':False,
			'jax_platforms':platform,
			'jax_platform_name':platform,
			'jax_cuda_visible_devices':os.environ.get('JAX_CUDA_VISIBLE_DEVICES'),
			'jax_enable_x64': True,
			}
		for name in configs:
			if configs[name] is None:
				continue
			jax.config.update(name,configs[name])

		func(device)		

		for name in environs:
			if environs[name] is None:
				continue
			os.environ[name] = environs[name]	

		return
	

	# NVIDIA_DEVICE_ORDER=PCI_BUS_ID
	# CUDA_DEVICE_ORDER=PCI_BUS_ID
	# JAX_CUDA_DEVICE_ORDER=PCI_BUS_ID
	# NVIDIA_VISIBLE_DEVICES=1
	# CUDA_VISIBLE_DEVICES=1
	# JAX_CUDA_VISIBLE_DEVICES=1

	name="CUDA_VISIBLE_DEVICES"
	# devices = ['cuda','gpu','cpu']
	# devices = [*(f"cuda:{i}" for i in os.environ[name].split(','))][-1:]
	devices = ['cuda','gpu','cpu',*(f"gpu:{i}" for i in os.environ[name].split(','))]

	for device in devices:
		try:
			func(device)
		except Exception as exception:
			print('Exception:',exception,'\n',traceback.format_exc())
			continue

	return

def test_matplotlib(*args,**kwargs):
	import sys,os
	import matplotlib
	import matplotlib.pyplot as plt

	mplstyle = 'config/plot.mplstyle'
	try:
		with matplotlib.style.context(mplstyle):
			plt.plot([1,2,3],[1,2,3],label='$\\textrm{Hi}~\\ket{\\psi}~\\norm{\\vec{v}}$')
			plt.legend()
			plt.savefig('plot.pdf')
	except Exception as exception:
		print(exception)
		plt.plot([1,2,3],[1,2,3],label='label')
		plt.legend()
		plt.savefig('plot.pdf')

	os.system('rm plot.pdf')

	return

def test_pytables(*args,**kwargs):
	import sys,os
	import numpy as np
	import pandas as pd
	import string

	path = 'data.hdf5'
	key = 'data'
	n,m = 10,5

	columns = [i for i in string.ascii_lowercase[:m]]
	data = np.random.rand(n,m)
	df = pd.DataFrame(data=data,columns=columns)

	df.to_hdf(path,key=key)
	tmp = pd.read_hdf(path,key=key)

	assert tmp.equals(df)

	os.remove(path)

	return


def main(*args,**kwargs):
	test_python()
	test_numpy()
	test_jax()
	test_matplotlib()
	test_pytables()
	return

if __name__ == '__main__':
	main()
