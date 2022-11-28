#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial
import time
from time import time as timer

# Logging
import logging
logger = logging.getLogger(__name__)

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import vmap,array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,sigmoid,abs,qr,sqrt
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,einsum,eig,average,norm
from src.utils import slice_slice,datatype,returnargs
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

	bounds = [-1,1]
	random = 'haar'

	data = rand(shape,bounds=bounds,random=random,key=seed,dtype=dtype)

	ndim = data.ndim

	# Create random matrices versus vectors
	if ndim == 1:
		pass
	elif ndim == 2:
		data = data[...,0]
	elif ndim == 3:
		data = data[...,0]
	elif ndim == 4:

		data = data[...,0]

		data = einsum('...i,...j->...ij',data,data.conj())

		axis = 1
		size = shape[:axis+1]
		bounds = [0,1]
		key = seed
		dtype = datatype(dtype)

		weights = rand(size,bounds=bounds,key=key,dtype=dtype)
		weights /= weights.sum(axis,keepdims=True)

		data = einsum('ui...,ui->u...',data,weights)

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


def stateize(data,shape,hyperparameters,size=None,samples=None,seed=None,cls=None,dtype=None):
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
		size (int): Size of states
		samples (bool,array): Weight samples (create random weights, or use samples weights)
		seed (key,int): PRNG key or seed		
		cls (object): Class instance to update hyperparameters
		dtype (data_type): Data type of values		
	Returns:
		data (array): Array of states
	'''

	# Setup hyperparameters
	setup(hyperparameters,cls=cls)

	# Set data
	if shape is None or hyperparameters.get('shape') is None or hyperparameters.get('scale') is None:
		data = None
		
		# Set returns
		returns = ()
		returns += (data,)

		return returnargs(returns)

	# Ensure shape is iterable
	if isinstance(shape,int):
		shape = (shape,)

	# Shape of data (either (k,*shape) or (k,d,*shape)) depending on hyperparameters['shape'] (k,d)
	shape = (*hyperparameters['shape'][:-max(2,len(shape))],*shape)

	# Get seed
	seed = hyperparameters.get('seed',seed) if hyperparameters.get('seed',seed) is not None else seed

	# Get scale
	scale = hyperparameters.get('scale')

	# Delimiter for string
	delimiter = '_'

	# Properties for strings
	props = {
		**{string: {'func':haar,'locality':size} for string in ['random','U','haar']},
		None: {'func':haar,'locality':size},
		}

	if scale is None:
		string = None
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
		elif isinstance(data,str):
			data = array(load(data))
	
	# Assert data is normalized
	# assert allclose(ones(data.shape[0]),data.conj().T.dot(data))


	# Set dtype of data
	data = data.astype(dtype=dtype)


	# Set samples
	if samples is not None and isinstance(samples,bool):
		samples = rand(len(data),bounds=[0,1],key=seed,dtype=dtype)
		samples /= samples.sum()
	elif isinstance(samples,array):
		pass
	else:
		samples = None

	if samples is not None:
		if data is None:
			data = None
		elif data.ndim == 3:
			data = einsum('ujk,u->jk',data,samples)
		elif data.ndim == 2 and data.shape[0] == data.shape[1]:
			data = data
		elif data.ndim == 2 and data.shape[0] != data.shape[1]:
			data = einsum('uj,u->u',data,samples)
		elif data.ndim == 1:
			data = data

	# Set returns
	returns = ()
	returns += (data,)

	return returnargs(returns)
