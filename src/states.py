#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools
from copy import deepcopy
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

from src.utils import vmap,array,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,sigmoid,abs,qr,sqrt
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,einsum,eig,average,norm
from src.utils import slice_slice,datatype,returnargs,is_array
from src.utils import pi,e,scalars,delim,null

from src.system import System
from src.io import load,dump,join,split
from src.iterables import setter


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


class State(System):

	def __init__(self,data,shape,hyperparameters,size=None,samples=None,cls=None,system=None,**kwargs):
		'''
		Initialize data of states based on shape
		Args:
			data (dict,str,array,State): Data corresponding to state
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
			cls (object): Class instance to update hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		# Setup class attributes
		self.data = data
		self.shape = shape
		self.size = size
		self.samples = samples
		self.hyperparameters = hyperparameters

		setter(kwargs,system,delimiter=delim,func=True)
		super().__init__(**kwargs)

		for attr in hyperparameters:
			value = hyperparameters[attr]
			setattr(self,attr,value)
		for attr in hyperparameters.get('system',{}):
			value = hyperparameters['system'][attr]
			setattr(self,attr,value)

		# Setup hyperparameters
		hyperparameters = deepcopy(hyperparameters)
		setup(hyperparameters,cls=cls)

		# Set data
		if isinstance(data,self.__class__):
			self.data = data.data
		elif is_array(data):
			self.data = data
			return
		elif isinstance(data,dict):
			setter(hyperparameters,data,delimiter=delim,func=True)
		elif isinstance(data,str):
			hyperparameters['string'] = data

		if shape is None or hyperparameters.get('shape') is None or hyperparameters.get('scale') is None:
			self.data = None
			return

		# Ensure shape is iterable
		if isinstance(shape,int):
			shape = (shape,)

		# Shape of data (either (k,*shape) or (k,d,*shape)) depending on hyperparameters['shape'] (k,d)
		shape = (*hyperparameters.get('shape',(1,1,))[:-max(2,len(shape))],*shape)

		# Get seed
		seed = hyperparameters.get('seed',self.seed) if hyperparameters.get('seed',self.seed) is not None else self.seed

		# Get dtype
		dtype = self.dtype

		# Get scale
		scale = hyperparameters.get('scale')

		# Delimiter for string
		delimiter = '_'

		# Properties for strings
		props = {
			**{string: {'func':haar,'locality':size} for string in ['random','U','haar']},
			None: {'func':haar,'locality':size},
			}

		string = hyperparameters.get('string')

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
							seed=seed,
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
		elif is_array(samples):
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
				data = einsum('uj,u->j',data,samples)
			elif data.ndim == 1:
				data = data


		self.data = data
		self.samples = samples
		self.string = string
		self.shape = self.data.shape if self.data is not None else None
		self.ndim = self.data.ndim if self.data is not None else None
		self.seed = seed
		self.scale = scale

		return

	def __call__(self,data=null()):
		'''
		Class data
		Args:
			data (array): Data
		Returns:
			data (array): Data
		'''
		if not isinstance(data,null):
			self.data = data
			self.shape = self.data.shape if self.data is not None else None
			self.ndim = self.data.ndim if self.data is not None else None
		return self.data
