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
from src.utils import slice_slice,datatype,returnargs
from src.utils import pi,e,scalars

from src.io import load,dump,join,split



dtype = 'complex'
basis = {
	'I': array([[1,0],[0,1]],dtype=dtype),
	'X': array([[0,1],[1,0]],dtype=dtype),
	'Y': array([[0,-1j],[1j,0]],dtype=dtype),
	'Z': array([[1,0],[0,-1]],dtype=dtype),
	'00':array([[1,0],[0,0]],dtype=dtype),
	'01':array([[0,1],[0,0]],dtype=dtype),
	'10':array([[0,0],[1,0]],dtype=dtype),
	'11':array([[0,0],[0,1]],dtype=dtype),
}


def haar(shape,bounds,random,seed,dtype):
	'''
	Initialize haar random state from 0 state
	Args:
		shape (int,iterable[int]): Shape of state
		bounds (iterable): Bounds on state value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of state
	Returns:
		data (array): Array of state
	'''

	# Ensure shape is iterable
	if isinstance(shape,int):
		shape = (shape,)

	ndim = len(shape)
	_dtype = datatype(dtype)

	data = rand(shape,bounds=bounds,random=random,key=seed,dtype=dtype)

	data = data.reshape(*data.shape[:1],*(1,)*(4-ndim),*data.shape[1:])

	for i in range(data.shape[0]):
		for j in range(data.shape[1]):

			Q,R = qr(data[i,j])
			R = diag(R)
			R = diag(R/abs(R))
			
			data = data.at[i,j].set(Q.dot(R))

	data = data.reshape(shape).astype(dtype)

	data = data[...,0]

	# Create random matrices versus vectors
	if ndim == 4:

		data = einsum('...i,...j->...ij',data,data.conj())

		axis = 1
		size = shape[axis]
		weights = rand(size,bounds=[0,1],key=seed,dtype=_dtype)

		data = average(data,axis,weights)

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


def noiseize(data,shape,hyperparameters,size=None,samples=None,seed=None,cls=None,dtype=None):
	'''
	Initialize data of noise based on shape
	Args:
		data (object): Data corresponding to parameters
		shape (int,iterable[int]): Shape of data
		hyperparameters (dict): Dictionary of parameter groupings, with dictionary values with properties:
			'category':str : category of parameter
			'shape':iterable[int] : shape of noise
			'initialization':str : initialization type
			'random':str : type of random initialization
			'seed': int: random seed
			'bounds': iterable[float]: bounds on noise
		size (int): Size of noise
		samples (bool,array): Weight samples (create random weights, or use samples weights)
		seed (key,int): PRNG key or seed		
		cls (object): Class instance to update hyperparameters
		dtype (data_type): Data type of values		
	Returns:
		data (array): Array of noise
		samples (array): Weights of samples
	'''

	# Setup hyperparameters
	setup(hyperparameters,cls=cls)

	# Set data
	if shape is None or hyperparameters.get('shape') is None:
		data = None
		return data

	# Ensure shape is iterable
	if isinstance(shape,int):
		shape = (shape,)

	# Get seed
	seed = hyperparameters.get('seed',seed) if hyperparameters.get('seed',seed) is not None else seed

	# Get scale
	scale = hyperparameters['scale']	

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


	# Set data
	data = None

	if scale is None:
		data = None
	elif isinstance(string,str):

		if string in ['phase']:
			data = [sqrt(1-scale)*basis['I'],
					sqrt(scale)*basis['Z']]
		elif string in ['amplitude']:
			data = [basis['00'] + sqrt(1-scale)*basis['11'],
					sqrt(scale)*basis['01']]
		elif string in ['depolarize']:
			data = [sqrt(1-scale)*basis['I'],
					sqrt(scale/3)*basis['X'],
					sqrt(scale/3)*basis['Y'],
					sqrt(scale/3)*basis['Z']]

		data = array([
			tensorprod(i)
			for i in itertools.product(data,repeat=size)
			])
		
	# Set samples
	if samples is not None and isinstance(samples,bool):
		samples = rand(len(data),bounds=[0,1],key=seed,dtype=dtype)
		samples /= samples.sum()
	elif isinstance(samples,array):
		pass		
	else:
		samples = None
	
	# Set returns
	returns = ()
	returns += (data,)

	if samples is not None:
		returns += (samples,)


	return returnargs(returns)