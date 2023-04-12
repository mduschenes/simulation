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

from src.utils import jit,array,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,bound,nullbound,sin,cos,minimum,maximum
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,to_list
from src.utils import initialize,slice_slice,datatype,returnargs
from src.utils import pi,itg,scalars,delim,separ

from src.iterables import indexer,inserter,setter,getter

from src.system import System
from src.io import load,dump,join,split


class Parameter(System):

	def __init__(self,data,*args,model=None,system=None,**kwargs):
		'''
		Initialize data of parameter
		Args:
			data (iterable): data of parameter
			string (str): Name of parameter
			category (str): category of parameter, allowed strings in ['variable','constant']
			method (str): method of parameter, allowed strings in ['unconstrained','constrained','bounded']
			group (iterable[iterable[str]]): iterable of groups associated with parameter grouping
			locality (str): locality of parameter across groups, allowed strings in ['shared']
			bounds (iterable[object]): Bounds of parameters
			dims (iterable[str]): Model attributes for additional dimensions of parameter
			parameters (iterable): parameters of parameter
			seed (int, key): Random seed for initialization
			random (str): Random type for initialization
			initialization (dict): Keyword arguments for initialization
			boundary (dict[str,object]): Boundary indices and values of parameters, along last axis
			constraint (dict[str,object]): Constraint indices and values of parameters, along last axis
			args (iterable): Additional arguments for parameter
			model (object): Model with additional attributes for initialization
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		defaults = dict(
			string=None,category=None,method=None,
			group=None,locality=None,bounds=None,dims=None,
			parameters=None,
			seed=None,random=None,initialization=None,
			indices=None,func=None,constraints=None,
			args=(),kwargs={}
			)

		setter(kwargs,dict(data=data,model=model,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,defaults,delimiter=delim,func=False)

		super().__init__(*args,**kwargs)

		self.__setup__()

		assert self.indices, "Zero-size data, No indices for parameter"

		self.__initialize__()

		return

	def __call__(self,data=None):
		'''
		Class data
		Args:
			data (array): data
		Returns:
			data (array): data
		'''

		if data is None:
			data = self.data.ravel() if self.data is not None else None
			return data

		data = data.reshape(self.shape)
		data = self.func(data)
		data = self.parameters*data

		return data	

	def __setup__(self,data=None,model=None):
		'''
		Setup class attributes
		Args:
			data (array): parameter data
			model (object): Model with additional attributes for initialization
		'''

		self.data = data if data is not None else self.data
		self.model = model if model is not None else self.model

		# Get data
		self.data = array(self.data,dtype=datatype(self.dtype)) if self.data is not None else None

		self.shape = self.data.shape if self.data is not None else None
		self.size = self.data.size if self.data is not None else None
		self.ndim = self.data.ndim if self.data is not None else None

		self.string = self.string if self.string is not None else None
		self.group = (*((*group,) for group in self.group),) if self.group is not None else ()
		self.parameters = self.parameters if self.parameters is not None else 1
		self.dtype = self.data.dtype if self.data is not None else None

		# Set data
		method = self.method if self.method is not None else None
		locality = self.locality if self.locality is not None else None
		group = self.group if self.group is not None else []
		model = self.model if self.model is not None else System(data={})
		dims = self.dims if self.dims is not None else []
		kwargs = self.kwargs if self.kwargs is not None else {}
		dtype = self.dtype

		# Get strings
		strings = {}
		for data in self.model.data:
			string = data.string
			if string in strings:
				continue
			strings[string] = [i for i,data in enumerate(self.model.data) if data.string == string]
		
		# Get groups
		groups = {subgroup: [strings[string][j] 
					for string in strings for i in [None,*range(len(strings[string]))] 
					for j in ([i] if i is not None else range(len(strings[string])))
					if '%s%s%s'%(string,separ if i is not None else '',str(i) if i is not None else '') in subgroup]
				for subgroup in group
				}

		# Get indices
		indices = {}
		for subgroup in groups:
			for string in strings:
				for i in strings[string]:

					if i not in groups[subgroup]:
						continue

					if locality in ['shared']:
						index = group.index(subgroup)
					else:
						index = (group.index(subgroup),groups[subgroup].index(i))
					
					indices[i] = index

		# Get shape
		ndim = max(len(indices[i]) if not isinstance(indices[i],int) else 1 for i in indices)
		shapes = [
			[max(indices[i][j] if not isinstance(indices[i],int) else indices[i] for i in indices)+1 for j in range(ndim)],
			[getattr(model,attr) for attr in dims]
			]
		shape = [int(i) for shape in shapes for i in shape if i is not None]

		# Set attributes
		self.indices = indices
		self.shape = shape
		self.size = prod(shape)
		self.ndim = len(shape)

		# Set functions
		defaults = {'indices':indices}
		kwargs.update(defaults)
		def func(parameters,kwargs=kwargs):
			return {i: parameters[kwargs['indices'][i]] for i in kwargs['indices']}
		def constraints(parameters,kwargs=kwargs):
			return default
		if method in ['constrained']:
			defaults = {'lambda':0,'constraint':{},'boundary':{}}
			funcs = [cos,sin]
			kwargs.update(defaults)
			kwargs['lambda'] = array(kwargs['lambda'])
			kwargs['constraint'] = [array([int(i) for i in self.constraint]),array([self.constraint[i] for i in self.constraint])]
			kwargs['boundary'] = [array([int(i) for i in self.boundary]),array([self.boundary[i] for i in self.boundary])]
			def func(parameters,kwargs=kwargs):
				parameters = bound(parameters)
				return {i: (parameters[(0,*kwargs['indices'][i][1:])]*
						   funcs[kwargs['indices'][i][0]%len(funcs)](pi*parameters[(1,*kwargs['indices'][i][1:])]))
						for i in kwargs['indices']}
			def constraints(parameters,kwargs=kwargs):
				return kwargs['lambda']*((parameters[...,kwargs['constraint'][0]] - kwargs['constraint'][1])**2).sum()
		elif method in ['unconstrained']:
			pass	
		elif method in ['bounded']:
			def func(parameters,kwargs=kwargs):
				parameters = bound(parameters,kwargs=kwargs)
				return {i: parameters[kwargs['indices'][i]] for i in kwargs['indices']}
		else:
			pass

		self.func = func
		self.constraints = constraints

		return

	def __initialize__(self,data=None,shape=None,dtype=None):
		'''
		Initialize class data with shape
		Args:
			data (array): Data of data
			shape (iterable[int]): Shape of data
			dtype (datatype): Data type of data
		'''
		self.data = data if data is not None else self.data
		self.shape = shape if shape is not None else self.shape
		self.dtype = dtype if dtype is not None else self.dtype

		self.data = initialize(self.data,self.shape,self,dtype=self.dtype)

		self.shape = self.data.shape if self.data is not None else None
		self.size = self.data.size if self.data is not None else None
		self.ndim = self.data.ndim if self.data is not None else None
		self.dtype = self.data.dtype if self.data is not None else None

		return

	def __str__(self):
		return str(self.string)

class Parameters(System):
	def __init__(self,data=None,model=None,system=None,**kwargs):
		'''
		Initialize data of parameters

		Parameters are split into groups with names [parameter], and can be accessed as 
		attributes Parameters.parameter, or items Parameters[parameter],
		and are contained in Parameters.data = {parameter: Parameter()} as Parameter() instances

		Setup parameters such that calling the Parameters(parameters) class with input parameters 
		i) parameters array of size (G*P*D),
			for G groups of P parameters, each of dimension D, 
			for parameter groups with category in ['variable'],
			P,D may be group dependent, and depend on Parameter.locality, Parameter.model and Parameter.dims 
		ii) parameters for each group are sliced with parameter indices (slice(P),slice(D)) and reshaped into shape (P,D)
		iii) parameters for each group are modified with Parameter() function i.e) bounds, scaling, features
		iv) parameters for all groups are concatenated to [parameter_i = Parameters[index_i]]
			with indices Parameters.indices = [indices_i]
			where index_i = [index_group_i,index_parameter_i,index_axis_i] for each data of model
		v) Parameters(parameters) returns iterable of parameters for each data in model [parameter_i]
		

		Args:
			data (dict): Dictionary of data corresponding to parameters groups, with dictionary values with properties:
				data (iterable): data of parameter
				string (str): Name of parameter
				category (str): category of parameter, allowed strings in ['variable','constant']
				method (str): method of parameter, allowed strings in ['unconstrained','constrained','bounded']
				group (iterable[iterable[str]]): iterable of groups associated with parameter grouping
				locality (str): locality of parameter across groups, allowed strings in ['shared']
				bounds (iterable[object]): Bounds of parameters
				dims (iterable[str]): Model attributes for additional dimensions of parameter
				parameters (iterable): parameters of parameter
				seed (int, key): Random seed for initialization
				random (str): Random type for initialization
				initialization (dict): Keyword arguments for initialization
				boundary (dict[str,object]): Boundary indices and values of parameters, along last axis
				constraint (dict[str,object]): Constraint indices and values of parameters, along last axis
				args (iterable): Additional arguments for parameter
				system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
				model (object): Model with additional attributes for initialization
				kwargs (dict): Additional system keyword arguments
			model (object): Model with additional attributes for initialization
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		defaults = dict(indices=None)

		if data is None:
			data = {}

		setter(kwargs,dict(data=data,model=model,system=system),delimiter=delim,func=False)
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
			data (array): data
		Returns:
			data (array,dict): data
		'''

		if data is None:
			data = []
			for parameter in self.indices:
				parameters = self[parameter]()
				if parameters is not None:
					data.extend(parameters.ravel())
			data = array(data).ravel()
		else:
			data = {}
			for parameter in self:
				parameters = self[parameter](data[self.indices[parameter]])
				if parameters is not None:
					data.update(parameters)
		return data

	def __setup__(self,data=None,model=None):
		'''
		Setup attribute
		Args:
			data (dict): Dictionary of data corresponding to parameters groups, with dictionary values with properties:
			model (object): Model with additional attributes for initialization
		'''

		self.data = data if data is not None else self.data
		self.model = model if model is not None else self.model

		if (not len(self)) or (self.model is None):
			return

		# Set data
		for parameter in list(self):
			
			args = {**getattr(self,parameter,{}),**dict(model=self.model,system=self.system)}
			try:
				setattr(self,parameter,Parameter(**args))
			except AssertionError:
				delattr(self,parameter)

		# Get indices
		indices = {}
		for parameter in self:
			if self[parameter].category in ['variable']:
				indices[parameter] = sum(indices[parameter] for parameter in indices) + self[parameter].size

		self.indices = indices

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

	def __len__(self):
		return self.data.__len__()

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

	def __str__(self):
		return ' '.join([str(self.data[parameter]) for parameter in self.data])

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		
		msg = '%s'%('\n'.join([
			*['Parameters %s %s: %s'%(self,attr,getattr(self,attr)) 
				for attr in ['string','shape']
			],
			]
			))
		self.log(msg,verbose=verbose)
		return		