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

from src.system import Object
from src.io import load,dump,join,split
from src.iterables import setter


class State(Object):
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

		# Shape of data, depending on size, samples and shape
		shape = [*self.size,*self.shape] if self.samples is not None else [*self.shape[:1]]

		# Delimiter for string
		delimiter = '_'

		# Properties for strings
		props = {
			**{string: {'func':rand,'locality':self.N} for string in ['random','U','haar']},
			None: {'func':rand,'locality':self.N},
			}

		if self.string is None or isinstance(self.string,str):

			if self.string is None:
				strings = [self.string]
				locality = self.N
			elif all(string in props for string in self.string.split(delimiter)):
				strings = self.string.split(delimiter)
				locality = sum(props[string]['locality'] for string in strings)
			else:
				strings = None
				locality = self.N			

			assert (self.N%locality == 0), 'Incorrect state with locality %d !%% size %d'%(locality,self.N)

			# TODO: Vectorized tensorproduct over state samples
			assert (self.N == locality), 'Only locality = size states'

			if self.string is not None:
				data = state = tensorprod([
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
		
		# Assert data is normalized
		# assert allclose(ones(data.shape[0]),data.conj().T.dot(data))

		self.data = data


		return