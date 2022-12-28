#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools
from copy import deepcopy
from functools import partial

# Logging
import logging
logger = logging.getLogger(__name__)

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,sigmoid,abs,qr,sqrt
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,allclose
from src.utils import slice_slice,datatype,returnargs,is_array
from src.utils import pi,e,delim,null

from src.system import System
from src.io import load,dump,join,split
from src.iterables import setter

def haar(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize haar random unitary operator
	Args:
		shape (int,iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	bounds = [-1,1]
	random = 'haar'

	data = rand(shape,bounds=bounds,random=random,key=seed,dtype=dtype)

	return data


def id(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize identity unitary operator
	Args:
		shape (int,iterable[int]): Shape of operator
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
		shape (int,iterable[int]): Shape of operator
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
		shape (int,iterable[int]): Shape of operator
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
		shape (int,iterable[int]): Shape of operator
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

class Operator(System):

	def __init__(self,data,shape,hyperparameters,size=None,samples=None,cls=None,system=None,**kwargs):
		'''
		Initialize operators
		Args:
			data (dict,str,array,Operator): Data corresponding to operator
			shape (int,iterable[int]): Shape of operator
			hyperparameters (dict): Dictionary of hyperparameters for operator
			size (int): size of operators
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

		if shape is None or hyperparameters.get('shape') is None:
			self.data = None
			return

		# Ensure shape is iterable
		if isinstance(shape,int):
			shape = (shape,)

		# Dimension of data
		d = min(shape)

		# Delimiter for string
		delimiter = '_'

		# Get seed
		seed = hyperparameters.get('seed',self.seed) if hyperparameters.get('seed',self.seed) is not None else self.seed

		# Get dtype
		dtype = self.dtype

		# Get scale
		scale = hyperparameters.get('scale')

		# Properties for strings
		props = {
			**{string: {'func':haar,'locality':size} for string in ['random','U','haar']},
			**{string: {'func':hadamard,'locality':1} for string in ['hadamard','H']},
			**{string: {'func':cnot,'locality':2} for string in ['cnot','CNOT','C']},
			**{string: {'func':toffoli,'locality':3} for string in ['toffoli','TOFFOLI','T']},
			**{string: {'func':{1:id,2:cnot,3:toffoli}.get(size,id),'locality':size} for string in ['control']},
			None: {'func':haar,'locality':size},
			}


		string = hyperparameters.get('string')

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
					seed=hyperparameters.get('seed',seed),
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


		# Set samples
		if samples is not None and isinstance(samples,bool):
			samples = rand(len(data),bounds=[0,1],key=seed,dtype=dtype)
			samples /= samples.sum()
		elif is_array(samples):
			pass
		else:
			samples = None
			
		
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

