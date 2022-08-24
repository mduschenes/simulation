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
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer
from src.utils import slice_slice
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

	data = rand(shape,bounds=bounds,random=random,key=seed,dtype=dtype)
	data /= sqrt(2)

	if data.ndim == 2:
		data = data.reshape(1,*data.shape)

	for i in range(data.shape[0]):

		Q,R = qr(data[i])
		R = diag(R)
		R = diag(R/abs(R))
		
		data = data.at[i].set(Q.dot(R))

	data = data.reshape(shape).astype(dtype)[:,:,0]

	return data


def setup(hyperparameters,cls=None):
	'''
	Setup hyperparameters
	Args:	
		hyperparameters (dict): Hyperparameters
		cls (object): Class instance
	'''

	return


def stateize(data,shape,hyperparameters,size=None,cls=None,dtype=None):
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
		cls (object): Class instance to update hyperparameters
		dtype (data_type): Data type of values		
	Returns:
		states (array): Array of states
	'''

	# Setup hyperparameters
	setup(hyperparameters,cls=cls)

	# Shape of data
	if isinstance(shape,int):
		shape = [-1,shape,shape]
	elif len(shape) < 3:
		shape = [shape[0],shape[1],shape[1]]
	if isinstance(hyperparameters['shape'],int):
		hyperparameters['shape'] = [hyperparameters['shape'],-1,-1]
	shape = [(hyperparameters['shape'][0] if hyperparameters['shape'][0] != -1 else 1) if shape[0] == -1 else shape[0],
			*[(s if t==-1 else t) for s,t in zip(shape[1:],hyperparameters['shape'][1:])]]

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
			# def func(i,shape=shape,string=string,locality=locality,hyperparameters=hyperparameters,dtype=dtype):
			# 	return state
			# data = vmap(func)(arange(n))
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
	# assert allclose(ones(n),data.conj().T.dot(data))

	# Set dtype of data
	data = data.astype(dtype=dtype)

	return data		