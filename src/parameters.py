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
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,to_list
from src.utils import initialize,slice_slice,datatype,returnargs
from src.utils import pi,itg,scalars,delim

from src.iterables import indexer,inserter,setter,getter

from src.system import System
from src.io import load,dump,join,split


class Parameter(System):

	def __init__(self,data,*args,system=None,**kwargs):
		'''
		Initialize data of parameter
		Args:
			data (iterable): data of parameter
			string (str): Name of parameter
			category (str): category of parameter, allowed strings in ['variable','constant']
			method (str): method of parameter, allowed strings in ['unconstrained','constrained','bound']
			group (iterable[iterable[str]]): iterable of groups associated with parameter grouping
			parameters (iterable): parameters of parameter
			bounds (iterable[object]): Bounds of data
			locality (iterable[str]): dictionary of locality of each axis of each parameter layer
			constants (interable[dict[int,object]]): dictionary of constant indices and values
			seed (int, key): Random seed for initialization
			random (str): Random type for initialization
			initialization (dict): Keyword arguments for initialization
			args (iterable): Additional arguments for parameter
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		defaults = dict(parameters=1)

		setter(kwargs,defaults,delimiter=delim,func=False)

		super().__init__(*args,**kwargs)
		
		self.data = array(data)

		self.shape = self.data.shape
		self.size = self.data.size
		self.ndim = self.data.ndim

		return

	def __call__(self,data=None,parameters=None):
		'''
		Class data
		Args:
			data (array): Data
			parameters (array): Parameters
		Returns:
			data (array): Data
		'''

		data = self.data if data is None else data
		parameters = self.parameters if parameters is None else parameters

		data = parameters*data

		return data	


class Parameters(System):
	def __init__(self,data,system=None,**kwargs):
		'''
		Initialize data of parameters
		Args:
			data (dict): Dictionary of data corresponding to parameters groupings, with dictionary values with properties:
				'string' (str): Name of parameter
				'category' (str): category of parameter, allowed strings in ['variable','constant']
				'method' (str): method of parameter, allowed strings in ['unconstrained','constrained','bound']
				'group' (iterable[iterable[str]]): iterable of groups associated with parameter grouping
				'data' (iterable): data of parameter
				'parameters' (iterable): parameters of parameter
				'bounds' (iterable[object]): Bounds of data
				'locality' (iterable[str]): dictionary of locality of each axis of each parameter layer
				'constants' (interable[dict[int,object]]): dictionary of constant indices and values
				'seed' (int, key): Random seed for initialization
				'random' (str): Random type for initialization
				'initialization' (dict): Keyword arguments for initialization
				'args' (iterable): Additional arguments for parameter
				'kwargs' (dict): Additional keyword arguments for parameter
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		defaults = {
			'string':None,
			'category':None,
			'method':None,
			'parameters':None,
			'initialize':None,			
			'random':None,
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

		parameters = self.parameters if self.parameters is not None else 1

		data = parameters*data.ravel()

		return data

	def __setup__(self,**kwargs):
		'''
		Setup attribute
		Args:
			kwargs (dict): Additional keyword arguments
		'''

		# Get datatype of data
		self.dtype = datatype(self.dtype)

		# Get data
		for parameter in list(self):

			self[parameter] = Parameter(**{self[parameter],**dict(system=self.system)})

			index = {(*group,): [
			[k for k in (j if not isinstance(j,int) else range(j)) if k in [i,'%s_%s'%(str(i),''.join([str(k) for k in (j if not isinstance(j,int) else range(j))]))]] for j in self.index[i]
			]
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

			data = array([[data[:shape[1]]]*shape[2]]*shape[0],dtype=self.dtype).transpose(0,2,1) if data is not None else data
			data = initialize(data,shape,self[parameter],dtype=self.dtype)
			getattr(self,parameter)(data=data)

		self.index = index
		self.shape = [max(i) for i in zip(*self[parameter]().shape for parameter in self if self[parameter].category in ['variable'])]
		self.size = prod(self.shape)
		self.string = ' '.join([str(getattr(self,parameter)) for parameter in self])

		return

	def __iter__(self):
		return self.__iterdata__()

	def __setattr__(self,key,value):
		super().__setattr__(key,value)
		self.__setdata__(key,value)
		return

	def __setitem__(self,key,value):
		super().__setitem__(key,value)
		self.__setdata__(key,value)
		return

	def __delattr__(self,key):
		super().__delattr__(key)
		self.__deldata__(key)
		return

	def __iterdata__(self):
		return self.data.__iter__()

	def __setdata__(self,key,value):
		if key in self.data:
			self.data[key] = value
		return
	
	def __deldata__(self,key):
		if key in self.data:
			self.data.pop(key)
		return

	def __call__(self,data=None):
		'''
		Class data
		Args:
			data (array): Data
		Returns:
			data (array): Data
		'''
		return self.data		