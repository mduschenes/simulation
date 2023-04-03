#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools
from copy import deepcopy
from functools import partial
import time
from time import time as timer

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import vmap,array,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,sigmoid,abs,qr,sqrt
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,einsum,eig,average,norm
from src.utils import slice_slice,datatype,returnargs,is_array,allclose
from src.utils import pi,e,scalars,delim

from src.system import Object
from src.io import load,dump,join,split


class State(Object):
	def __init__(self,data,shape,size=None,ndim=None,dims=None,system=None,**kwargs):
		'''
		Initialize data of attribute based on shape, with highest priority of arguments of: kwargs,args,data,system
		Args:
			data (dict,str,array,Noise): Data corresponding to noise
			shape (int,iterable[int]): Shape of each data
			size (int,iterable[int]): Number of data
			ndim (int): Number of dimensions of data
			dims (iterable[int]): Dimensions of N, D-dimensional sites [N,D]
			system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		super().__init__(data,shape,size=size,ndim=ndim,dims=dims,system=system,**kwargs)

		if self.ndim == 1:
			self.data /= sqrt(einsum('...i,...i->...',self.data,self.data.conj()).real)
		elif self.ndim == 2:
			self.data /= einsum('...ii->...',self.data).real/1

		return

	def __setup__(self,**kwargs):
		'''
		Setup attribute
		Args:
			kwargs (dict): Additional keyword arguments
		'''

		# Shape of data, depending on size and shape
		if self.size is not None:
			size = self.size
		else:
			size = [*[1]*2]
		self.size = size
		self.length = len(self.size) if self.size is not None else None

		if self.shape is not None:
			shape = [*self.size,*self.shape[:self.ndim]]
		else:
			shape = self.shape
		self.shape = shape

		# Delimiter for string
		delimiter = '_'

		# Properties for strings
		props = {
			**{string: {'func':rand,'locality':self.N} for string in ['random','U','haar','zero','one','plus','minus']},
			None: {'func':rand,'locality':self.N},
			}

		if self.string is None or isinstance(self.string,str):

			if self.string is None:
				strings = [self.string]
				locality = self.N
				shape = {string: shape for string in strings}
			elif all(string in props for string in self.string.split(delimiter)):
				strings = self.string.split(delimiter)
				locality = sum(props[string]['locality'] for string in strings)
				shape = {string: shape for string in strings}
			else:
				strings = None
				locality = self.N			
				shape = {string: shape for string in strings}
			assert (self.N%locality == 0), 'Incorrect state with locality %d !%% size %d'%(locality,self.N)

			if strings is not None:
				data = tensorprod([
						props[string]['func'](
							shape=shape[string],
							bounds=self.bounds,
							random=self.random,
							seed=self.seed,
							dtype=self.dtype
							)
						for string in strings
						]*(self.N//locality)
						)
			else:
				data = array(load(self.string))

		# Assert data is normalized
		if self.ndim == 1:
			normalization = einsum('...i,...i->...',data,data.conj())
		else:
			normalization = einsum('...ii->...',data)

		eps = ones(normalization.shape,dtype=self.dtype)

		assert (eps.shape == normalization.shape), "Incorrect operator shape %r != %r"%(eps.shape,normalization.shape)

		if self.dtype in ['complex128','float64']:
			assert allclose(eps,normalization), "Incorrect normalization data%r: %r"%(data.shape,normalization)

		self.data = data
		self.shape = self.data.shape if self.data is not None else None
		self.ndim = self.data.ndim if self.data is not None else None

		return