#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial
import time
from time import time as timer

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp
import jax
import jax.numpy as np
import jax.scipy as sp
import jax.example_libraries.optimizers
jax.config.update('jax_platform_name','cpu',)
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'
# np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.2e',)) for dtype in ['float','float64',np.float64,np.float32]}})

# Logging
import logging
logger = logging.getLogger(__name__)

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import vmap,array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,sigmoid,abs,qr,sqrt
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,einsum,eig,average
from src.utils import slice_slice,datatype
from src.utils import pi,e,scalars

from src.io import load,dump,join,split



dtype = 'complex'
basis = {
	'I': array([[1,0],[0,1]],dtype=dtype),
	'X': array([[0,1],[1,0]],dtype=dtype),
	'Y': array([[0,-1j],[1j,0]],dtype=dtype),
	'Z': array([[1,0],[0,-1]],dtype=dtype),
}


def haar(shape,bounds,random,seed,dtype):
	'''
	Initialize haar random state from 0 state
	Args:
		shape (iterable[int]): Shape of state
		bounds (iterable): Bounds on state value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of state
	Returns:
		data (array): Array of state
	'''

	# ndim to initialize state matrices versus vectors
	n = 4

	ndim = len(shape)
	_dtype = datatype(dtype)

	data = rand(shape,bounds=bounds,random=random,key=seed,dtype=dtype)
	data /= sqrt(2)

	data = data.reshape(*data.shape[:1],*(1,)*(n-ndim),*data.shape[1:])

	for i in range(data.shape[0]):
		for j in range(data.shape[1]):

			Q,R = qr(data[i,j])
			R = diag(R)
			R = diag(R/abs(R))
			
			data = data.at[i,j].set(Q.dot(R))

	data = data.reshape(shape).astype(dtype)

	data = data[...,0]

	# Create random matrices versus vectors
	if ndim == n:

		data = einsum('...i,...j->...ij',data,data.conj())

		axis = 1
		size = shape[axis]
		weights = rand(size,bounds=[0,1],key=seed,dtype=_dtype)

		data = average(data,axis,weights)

		axis = 0
		shape = data.shape
		if shape[axis] <= 1:
			shape = shape[axis+1:]
			data = data.reshape(shape)

	else:
		pass

	return data


def setup(hyperparameters,cls=None):
	'''
	Setup hyperparameters
	Args:	
		hyperparameters (dict): Hyperparameters
		cls (object): Class instance
	'''

	return


def noiseize(data,shape,hyperparameters,size=None,mapping=None,cls=None,dtype=None):
	'''
	Initialize data of noise based on shape
	Args:
		data (object): Data corresponding to parameters
		shape (iterable[int]): Shape of data
		hyperparameters (dict): Dictionary of parameter groupings, with dictionary values with properties:
			'category':str : category of parameter
			'shape':iterable[int] : shape of noise
			'initialization':str : initialization type
			'random':str : type of random initialization
			'seed': int: random seed
			'bounds': iterable[float]: bounds on noise
		size (int): Size of state
		mapping(str): Type of mapping, allowed strings in ['vector','matrix','tensor']
		cls (object): Class instance to update hyperparameters
		dtype (data_type): Data type of values		
	Returns:
		states (array): Array of noise
	'''

	# Setup hyperparameters
	setup(hyperparameters,cls=cls)

	# Shape of data (either (k,*shape) or (k,l,*shape)) depending on hyperparameters['shape']
	shape,dims = hyperparameters['shape'],shape
	ndim = len(dims)
	shape[-ndim:] = dims


	# Mappings of states
	maps = ['matrix']
	if mapping in maps:
		pass
	else:
		pass
	
	# Basis
	operators = {
		attr: basis[attr].astype(dtype)
		for attr in basis
		}

	if data is None:
		string = hyperparameters['string']
	elif isinstance(data,str):
		string = data
	elif isinstance(data,array):
		string = None

	data = None

	if isinstance(string,str):
		scale = hyperparameters['scale']
		if string in ['phase']:
			data = array([
				tensorprod([sqrt(1-scale)*basis['I']]*size),
				tensorprod([sqrt(scale)*basis['Z']]*size),
				])

	if mapping not in maps:
		data = None

	return data