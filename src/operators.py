#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp
import jax
import jax.numpy as np
import jax.scipy as sp
import jax.example_libraries.optimizers
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'
# np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})

# Logging
import logging
logger = logging.getLogger(__name__)

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,sigmoid,abs,qr,sqrt
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,allclose
from src.utils import slice_slice
from src.utils import pi,e

from src.io import load,dump,join,split

def haar(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize haar random unitary operator
	Args:
		shape (iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''

	if bounds is None:
		bounds = [-1,1]

	if random is None:
		random = 'gaussian'

	data = rand(shape,bounds=bounds,random=random,key=seed,dtype=dtype)
	data /= sqrt(2)

	Q,R = qr(data)
	R = diag(R)
	R = diag(R/abs(R))
	
	data = Q.dot(R)

	data = data.astype(dtype)

	return data


def id(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize identity unitary operator
	Args:
		shape (iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = identity(shape)

	data = data.astype(dtype)

	return data

def cnot(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize cnot unitary operator
	Args:
		shape (iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = array([
		[1,0,0,0],
		[0,1,0,0],
		[0,0,0,1],
		[0,0,1,0]])

	data = data.astype(dtype)

	return data


def hadamard(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize hadamard unitary operator
	Args:
		shape (iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = array([
		[1,1,],
		[1,-1]])/sqrt(2)

	data = data.astype(dtype)

	return data	


def toffoli(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize toffoli unitary operator
	Args:
		shape (iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = array([
		[1,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0],
		[0,0,1,0,0,0,0,0],
		[0,0,0,1,0,0,0,0],
		[0,0,0,0,1,0,0,0],
		[0,0,0,0,0,1,0,0],
		[0,0,0,0,0,0,0,1],
		[0,0,0,0,0,0,1,0]])

	data = data.astype(dtype)

	return data	


def setup(hyperparameters,cls=None):
	'''
	Setup hyperparameters
	Args:	
		hyperparameters (dict): Hyperparameters
		cls (object): Class instance
	'''

	return

def operatorize(data,shape,hyperparameters,size=None,mapping=None,cls=None,dtype=None):
	'''
	Initialize operators
	Args:
		data (dict,str,array): Label or path or array of operator
		shape (iterable[int]): Shape of operator
		hyperparameters (dict): Dictionary of hyperparameters for operator
		size (int): size to initialize operator
		mapping(str): Type of mapping, allowed strings in ['vector','matrix','tensor']
		cls (object): Class instance to update hyperparameters		
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''

	# Setup hyperparameters
	setup(hyperparameters,cls=cls)

	# Dimension of data
	d = min(shape)

	# Delimiter for string
	delimiter = '_'

	# Properties for strings
	props = {
		**{string: {'func':haar,'locality':size} for string in ['random','U','haar']},
		**{string: {'func':hadamard,'locality':1} for string in ['hadamard','H']},
		**{string: {'func':cnot,'locality':2} for string in ['cnot','CNOT','C']},
		**{string: {'func':toffoli,'locality':3} for string in ['toffoli','TOFFOLI','T']},
		**{string: {'func':{1:id,2:cnot,3:toffoli}.get(size,id),'locality':size} for string in ['control']},
		None: {'func':haar,'locality':size},
		}


	if data is None:
		string = hyperparameters.get('string')
	elif isinstance(data,str):
		string = data
	elif isinstance(data,array):
		string = None

	if string is None or isinstance(string,str):

		if string is None:
			strings = [string]
			locality = size
		elif all(string in props for string in string.split(delimiter)):
			strings = string.split(delimiter)
			locality = sum(props[string]['locality'] for string in strings)
		else:
			strings = None
			locality = size			

		assert (size%locality == 0), 'Incorrect operator with locality %d !%% size %d'%(locality,size)

		if string is not None:
			data = tensorprod([
				props[string]['func'](shape,
					bounds=hyperparameters.get('bounds'),
					random=hyperparameters.get('random'),
					seed=hyperparameters.get('seed'),
					dtype=dtype
					)
				for string in strings
				]*(size//locality)
			)
		else:
			data = array(load(data))
	
	# Assert data is unitary
	assert allclose(eye(d),data.conj().T.dot(data))
	assert allclose(eye(d),data.dot(data.conj().T))


	# Set dtype of data
	data = data.astype(dtype=dtype)

	return data