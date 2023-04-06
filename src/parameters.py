#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools
from copy import deepcopy
from functools import partial
import time
from time import time as timer
from math import prod


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,array,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,bound,nullbound,sin,cos,minimum,maximum,bound
from src.utils import tensorprod,trace,asscalar,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,to_list
from src.utils import initialize,slice_slice,datatype,returnargs,is_array
from src.utils import pi,itg,scalars,delim

from src.iterables import indexer,inserter,setter,getter

from src.system import Data,Object
from src.io import load,dump,join,split

class Parameters(Data):
	def __init__(self,data,index,system=None,**kwargs):
		'''
		Initialize data of parameters
		Args:
			data (dict): Dictionary of data corresponding to parameters groupings, with dictionary values with properties:
				'category' (str) : category of parameter, allowed strings in ['variable','constant']
				'group' (iterable[iterable[str]]) : iterable of groups associated with parameter grouping
				'locality' (iterable[str]) : dictionary of locality of each axis of each parameter layer
				'boundaries' (interable[dict[int,object]]) : dictionary of boundary indices and values
				'constants' (interable[dict[int,object]]) : dictionary of constant indices and values
				'seed': (int, key): Random seed for initialization
				'scale': (object): Scale of data
				'bounds': (iterable[object]): Bounds of data
			index (dict[str,iterable[iterable[int]]]): Indexes of parameters for each group
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		defaults = {
			'string':None,
			'init':True,
			'category':None,
			'method':None,
			'parameters':None,
			'scale':1,
			'samples':None,
			'initialization':'random',
			'random':'random',
			'seed':None,
			'bounds':[-1,1],
		}

		# Setup kwargs
		setter(kwargs,dict(data=data,shape=shape,system=system),delimiter=delim,func=False)
		setter(kwargs,data,delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,defaults,delimiter=delim,func=False)
		super().__init__(**kwargs)

		self.__setup__()

		return

	def __call__(self,data=None):
		'''
		Class data
		Args:
			data (array): Data
		Returns:
			data (array): Data
		'''

		if data is None:
			data = indexer(self.index,self).ravel()
		else:
			inserter(self.index,data.reshape(indexer(self.index,self).shape),self)

		data = data.ravel()

		return data

	def __setup__(self,**kwargs):
		'''
		Setup attribute
		Args:
			kwargs (dict): Additional keyword arguments
		'''

		# Get datatype of data
		dtype = datatype(self.dtype)

		# Get index
		index = {i: [list(j) if not isinstance(j,int) else list(range(j)) for j in self.index[i]] for i in self.index}

		# Todo
			# Sort out how to split up parameters into dictionary for each operator, 
			#	but initially as arrays, that must be split up, but then indexed back into parameters for each operator
			#	Sort out indexing of parameters at operators/data level
			# Fix analytical indices for gradients
			# Run tests of parameters and variables
			# Run test of gradients
			# Run all tests	

		# Get data
		for parameter in list(self):

			self[parameter] = Object(**self[parameter],system=self.system)

			index = {(*group,): [[k for k in (j if not isinstance(j,int) else range(j)) if k in [i,'%s_%s'%(str(i),''.join([str(k) for k in (j if not isinstance(j,int) else range(j))]))]] for j in self.index[i]]
				for group in self[parameter].group if }

			if not any(index[group] for group in index):
				delattr(self,parameter)
				continue

			if any(any(j in group for j in [i,*['%s_%s'%(str(i),''.join([str(k) for k in j])) for j in index[i]]]) 
				for group in self[parameter].group for i in index):
				delattr(self,parameter)


			data = self[parameter].data
			groups = [tuple(group) for group in self[parameter].group]
			shape = {}
			shape = {group: [
					  *[sum(check(self.cls.data[j],group) for j in range(i)) for i in self.shape[:1]],
					  *[i for i in self.shape[1:]],
					  ] for group in groups}

			shape = [len(shape),*[max(i) for i in zip(*(shape[group] for group in shape))]]

			data = array([[data[:shape[1]]]*shape[2]]*shape[0],dtype=dtype).transpose(0,2,1) if data is not None else data
			data = initialize(data,shape,self[parameter],dtype=dtype)
			getattr(self,parameter)(data=data)

		self.dtype = dtype
		self.index = index
		self.shape = [max(i) for i in zip(*self[parameter]().shape for parameter in self if self[parameter].category in ['variable'])]
		self.size = prod(self.shape)
		self.string = ' '.join([str(getattr(self,parameter)) for parameter in self])

		return