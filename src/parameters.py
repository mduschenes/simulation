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
from src.utils import slice_slice,datatype,returnargs,is_array
from src.utils import pi,itg,scalars,delim

from src.iterables import indexer,inserter,setter,getter

from src.system import System,Object
from src.io import load,dump,join,split

class Parameters(System):
	def __init__(self,data,shape,system=None,**kwargs):
		'''
		Initialize data of shapes of parameters based on shape of data. Initializes attributes of
			data (dict,array,Parameters): Dictionary of parameter hyperparameter attributes ['shape','values','slice','index','parameters','features','variables','constraints']
					for parameter,group keys and for layers ['parameters',features','variables','constraints']
					Attributes are used to yield layer outputs, given input variable parameters, with layer functions acting on slices of parameters, yielding values at indices
					
					Attributes dictionaries are of the form :
					
					{attribute:{layer:{parameter:group:...}}} for attributes: 'slice','index',layers 
					'slice' (tuple[slice]): slices along each axis of the input values to that layer for that parameter,group key
					'index' (tuple[slice]): slices along each axis of the output values of that layer for that parameter,group key
					layer (callable): Callable function with signature func(parameters,values,slices,indices) for input parameters[slices] that yields values[indices] for that layer

					{attribute:{layer:...}} for attributes: 'shape','values'
					'shape' (tuple[int]): shape of values for that layer
					'values': (array): array of values for that layer with variable + boundary/constant values
					'index' (tuple[slice]): slices along each axis of the output values of that layer for that parameter,group key
					layer (callable): Callable function with signature func(parameters,values,slices,indices) for input parameters[slices] that yields values[indices] for that layer
		Args:
			data (dict): Dictionary of data corresponding to parameters groupings, with dictionary values with properties:
				'category':str : category of parameter
				'group':iterable[iterable[str]] : iterable of groups associated with parameter grouping
				'shape':dict[str,iterable[int]] : dictionary of shape of each parameter layer
				'locality':dict[str,iterable[str]] : dictionary of locality of each axis of each parameter layer
				'boundaries':dict[str,iterable[dict[str,iterable]]] : dictionary of boundary indices and values of each axis of each parameter layer {'layer':[{'slice':[indices_axis],'value':[values_axis]}]}
				'constants':dict[str,iterable[dict[str,iterable]]] : dictionary of constant indices and values of each axis of each parameter layer {'layer':[{'slice':[indices_axis],'value':[values_axis]}]}
			shape (iterable[int]): Shape of data
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
			inserter(self.index,data.reshape(indexer(self.index,self.data).shape),self.data)

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
		self.dtype = dtype

		# Get initialize
		if self.initialize is None:
			initialize = lambda parameters,shape,hyperparameters,**kwargs: parameters
		else:
			initialize = self.initialize

		# Get check
		if self.check is None:
			check = lambda data,group: True
		else:
			check = self.check


		# Get parameters
		print(self.data)

		# Remove not used parameters of data
		for parameter in list(self):
			if not any(self.check(self.cls.data[i],group)
				for i in range(self.shape[0])
				for group in self.data[parameter].get('group',[])):
				self.data.pop(parameter)
				delattr(self,parameter)

		for parameter in self:
			setattr(self,parameter,Object(**self.data[parameter],system=self.system))
			print(getattr(self,parameter),self.data[parameter])
			self[parameter] = getattr(self,parameter)

		# Get data
		for parameter in self:
			data = self.data[parameter]['data']
			groups = [tuple(group) for group in self.data[parameter]['group']]
			shape = {group: [
					  *[sum(check(self.cls.data[j],group) for j in range(i)) for i in self.shape[:1]],
					  *[i for i in self.shape[1:]],
					  ] for group in groups}

			shape = [len(shape),*[max(i) for i in zip(*(shape[group] for group in shape))]]

			data = array([[data[:shape[1]]]*shape[2]]*shape[0],dtype=dtype).transpose(0,2,1) if data is not None else data
			data = initialize(data,shape,self.data[parameter],dtype=dtype)
			getattr(self,parameter)(data=data)

		self.shape = [max(i) for i in zip(*self.data[parameter]().shape for parameter in self.data if self.data[parameter].category in ['variable'])]
		self.size = prod(self.shape)
		self.string = ' '.join([str(getattr(self,parameter)) for parameter in self])
		self.index = [parameter for parameter in self.data if self.data[parameter].category in ['variable']]

		return