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

	bounds = [-1,1]
	random = 'haar'

	data = rand(shape,bounds=bounds,random=random,key=seed,dtype=dtype)

	data = data[...,0]

	# Create random matrices versus vectors
	if ndim == n:

		data = einsum('...i,...j->...ij',data,data.conj())

		axis = 1
		size = shape[:axis+1]
		bounds = [0,1]
		key = seed
		dtype = datatype(dtype)

		weights = rand(size,bounds=bounds,key=key,dtype=dtype)

		weights /= weights.sum(axis)

		data = einsum('ui,ui...->u...',data,weights)

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


def stateize(data,shape,hyperparameters,size=None,mapping=None,cls=None,dtype=None):
	'''
	Initialize data of states based on shape
	Args:
		data (object): Data corresponding to parameters
		shape (iterable[int]): Shape of data
		hyperparameters (dict): Dictionary of parameter groupings, with dictionary values with properties:
			'category':str : category of parameter
			'shape':iterable[int] : shape of states
			'initialization':str : initialization type
			'random':str : type of random initialization
			'seed': int: random seed
			'bounds': iterable[float]: bounds on states
		size (int): Size of state
		mapping (str): Type of mapping, allowed strings in ['vector','matrix','tensor']		
		cls (object): Class instance to update hyperparameters
		dtype (data_type): Data type of values		
	Returns:
		states (array): Array of states
	'''

	# Setup hyperparameters
	setup(hyperparameters,cls=cls)

	# Shape of data (either (k,*shape) or (k,d,*shape)) depending on hyperparameters['shape'] (k,d)
	shape = [*hyperparameters['shape'][:-len(shape)],*shape]

	# Delimiter for string
	delimiter = '_'

	# Properties for strings
	props = {
		**{string: {'func':haar,'locality':size} for string in ['random','U','haar']},
		None: {'func':haar,'locality':size},
		}


	if data is None:
		string = hyperparameters['string']
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

		assert (size%locality == 0), 'Incorrect state with locality %d !%% size %d'%(locality,size)

		# TODO: Vectorized tensorproduct over state samples
		assert (size == locality), 'Only locality = size states'

		if string is not None:
			data = state = tensorprod([
					props[string]['func'](shape,
						bounds=hyperparameters['bounds'],
						random=hyperparameters['random'],
						seed=hyperparameters['seed'],
						dtype=dtype
						)
					for string in strings
					]*(size//locality)
					)
		else:
			data = array(load(data))
	
	# Assert data is normalized
	# assert allclose(ones(data.shape[0]),data.conj().T.dot(data))

	# Set dtype of data
	data = data.astype(dtype=dtype)


	return data		