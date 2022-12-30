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

from src.system import Object
from src.io import load,dump,join,split
from src.iterables import setter



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


class Operator(Object):
	def __init__(self,data,shape,size=None,dims=None,samples=None,system=None,**kwargs):
		'''
		Initialize data of attribute based on shape, with highest priority of arguments of: args,data,system,kwargs
		Args:
			data (dict,str,array,Noise): Data corresponding to noise
			shape (int,iterable[int]): Shape of each data
			size (int,iterable[int]): Number of data
			dims (iterable[int]): Dimensions of N, D-dimensional sites [N,D]
			samples (bool,array): Weight samples (create random weights, or use samples weights)
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		super().__init__(data,shape,size=size,dims=dims,samples=samples,system=system,**kwargs)

		return

	def __setup__(self,**kwargs):
		'''
		Setup attribute
		Args:
			kwargs (dict): Additional keyword arguments
		'''

		# Delimiter for string
		delimiter = '_'

		# Properties for strings
		props = {
			**{string: {'func':rand,'locality':self.N} for string in ['random','U','haar']},
			**{string: {'func':hadamard,'locality':1} for string in ['hadamard','H']},
			**{string: {'func':cnot,'locality':2} for string in ['cnot','CNOT','C']},
			**{string: {'func':toffoli,'locality':3} for string in ['toffoli','TOFFOLI','T']},
			**{string: {'func':{1:id,2:cnot,3:toffoli}.get(self.N,id),'locality':self.N} for string in ['control']},
			None: {'func':rand,'locality':self.N},
			}


		if self.string is None:
			strings = [self.string]
			locality = self.N
		elif all(string in props for string in self.string.split(delimiter)):
			strings = self.string.split(delimiter)
			locality = sum(props[string]['locality'] for string in strings)
		else:
			strings = None
			locality = self.N			

		assert (self.N%locality == 0), 'Incorrect operator with locality %d !%% size %d'%(locality,self.N)

		if self.string is not None:
			data = tensorprod([
				props[string]['func'](shape,
					bounds=self.bounds,
					random=self.random,
					seed=self.seed,
					dtype=self.dtype
					)
				for string in strings
				]*(self.N//locality)
			)
		else:
			data = array(load(self.data))
		
		# Assert data is unitary
		assert allclose(eye(self.n),data.conj().T.dot(data))
		assert allclose(eye(self.n),data.dot(data.conj().T))

		self.data = data

		return

