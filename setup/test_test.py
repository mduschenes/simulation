#!/usr/bin/env python

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
	envs = {
		'JAX_PLATFORMS':'cpu',
		'JAX_PLATFORM_NAME':'cpu',
		'TF_CPP_MIN_LOG_LEVEL':5
	}
	for var in envs:
		os.environ[var] = str(envs[var])
	import jax
	import jax.numpy as jnp
	print(jax.devices())
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
	test_pytables()
	return

if __name__ == '__main__':
	main()
