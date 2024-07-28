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

	def func(device):
		import os
		environs = {
				'JAX_PLATFORMS':device,
				'JAX_PLATFORM_NAME':device,
				'TF_CPP_MIN_LOG_LEVEL':5
			}
			
		for name in environs:
			os.environ[name] = str(environs[name])
		import jax
		from jax.lib import xla_bridge
		import jax.numpy as np
		
		configs = {
			'jax_disable_jit':False,
			'jax_platforms':device,
			'jax_platform_name':device,
			'jax_enable_x64': True,
			}
		for name in configs:
			jax.config.update(name,configs[name])

		print(xla_bridge.get_backend().platform)
		print(jax.devices())

		array = np.array([1,2,3])
		print(array.devices())
		array *= array

		return
	
	devices = ['cuda','gpu','cpu']
	# devices = ['cpu','gpu',]
	for device in devices:
		try:
			func(device)
			return
		except Exception as exception:
			print(exception)
			continue

	raise AttributeError("JAX IMPORT ERROR")

	return

def test_matplotlib(*args,**kwargs):
	import sys,os
	import matplotlib
	import matplotlib.pyplot as plt

	mplstyle = 'plot.mplstyle'
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
	test_numpy()
	test_matplotlib()
	test_pytables()
	return

if __name__ == '__main__':
	main()
