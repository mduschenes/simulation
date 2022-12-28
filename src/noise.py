#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
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
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,einsum,eig,average
from src.utils import slice_slice,datatype,returnargs,is_array
from src.utils import pi,e,scalars,delim,null

from src.system import System
from src.io import load,dump,join,split
from src.iterables import setter


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


class Noise(System):
	def __init__(self,data,shape,hyperparameters,size=None,samples=None,cls=None,system=None,**kwargs):
		'''
		Initialize data of noise based on shape
		Args:
			data (dict,str,array,Noise): Data corresponding to noise
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

		# Get seed
		seed = hyperparameters.get('seed',self.seed) if hyperparameters.get('seed',self.seed) is not None else self.seed

		# Get dtype
		dtype = self.dtype

		# Get scale
		scale = hyperparameters.get('scale')

		# Basis
		operators = {
			attr: basis[attr].astype(dtype)
			for attr in basis
			}

		string = hyperparameters.get('string')

		assert (scale >= 0) and (scale <= 1), "Noise scale %r not in [0,1]"%(scale)

		if string is None:
			data = [basis['I']]
		elif string in ['phase']:
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
