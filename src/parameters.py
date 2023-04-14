#!/usr/bin/env python

# Import python modules
import os,sys
from copy import deepcopy
from functools import partial
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
			axis (int,iterable[int]): Axis of input parameter data to insert into class data
			parameters (iterable): parameters of parameter
			seed (int, key): Random seed for initialization
			random (str): Random type for initialization
			initialization (dict): Keyword arguments for initialization
			constant (dict[dict[str,object]]): constant indices and values of parameters, along axis, of the form {'axis':{'index':value}}			
			args (iterable): Additional arguments for parameter
			model (object): Model with additional attributes for initialization
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		defaults = dict(
			string=None,category=None,method=None,
			group=None,locality=None,bounds=None,dims=None,axis=None,
			parameters=None,
			seed=None,random=None,initialization=None,constant=None,
			indices=None,slices=None,func=None,constraint=None,
			args=(),kwargs={}
			)

		setter(kwargs,dict(data=data,model=model,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,defaults,delimiter=delim,func=False)

		super().__init__(*args,**kwargs)

		self.__setup__()

		self.__initialize__()

		assert (self.indices is not None and self.indices.size) and (self.slices is not None and self.slices.size), "Zero-size data, No indices,slices for parameter"

		print('data',self)
		print(self.data)

		return

	def __call__(self,parameters=None):
		'''
		Class data
		Args:
			parameters (array): parameters
		Returns:
			parameters (array): parameters
		'''

		if parameters is None:
			return self.data

		parameters = parameters

		parameters = parameters.reshape(-1,*self.shape[1:])
		
		parameters = self.func(parameters)

		return parameters	

	def __str__(self):
		return str(self.string)

	def constraints(self,parameters=None):
		'''
		Class constraints
		Args:
			parameters (array): parameters
		Returns:
			constraints (array): constraints
		'''

		constraints = self.constraint(parameters)

		return constraints


	def __setup__(self,data=None,model=None):
		'''
		Setup class attributes
		Args:
			data (array): parameter data
			model (object): Model with additional attributes for initialization
		'''

		self.data = data if data is not None else self.data
		self.model = model if model is not None else self.model
		self.dtype = datatype(self.dtype)

		# Get data
		self.data = array(self.data,dtype=self.dtype) if self.data is not None else None

		self.shape = self.data.shape if self.data is not None else None
		self.size = self.data.size if self.data is not None else None
		self.ndim = self.data.ndim if self.data is not None else None

		self.string = self.string if self.string is not None else None
		self.group = (*((*group,) for group in self.group),) if self.group is not None else ()
		self.parameters = self.parameters if self.parameters is not None else 1
		self.dtype = self.data.dtype if self.data is not None else None

		# Set data
		category = self.category if self.category is not None else None
		method = self.method if self.method is not None else None
		locality = self.locality if self.locality is not None else None
		group = self.group if self.group is not None else []
		model = self.model if self.model is not None else None
		dims = self.dims if self.dims is not None else []
		kwargs = self.kwargs if self.kwargs is not None else {}
		dtype = self.dtype

		# Get strings
		strings = {}
		if self.model is not None:
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

		# Get slices
		slices = {}
		for subgroup in groups:
			for string in strings:
				for i in strings[string]:

					if i not in groups[subgroup]:
						continue

					if locality in ['shared']:
						index = group.index(subgroup)
					else:
						index = group.index(subgroup)*len(groups[subgroup]) + groups[subgroup].index(i)
					
					slices[i] = index


		# Get shape
		shape = [
			max(slices[i] for i in slices)+1,
			*[getattr(model,attr) for attr in dims]
			]
		shape = [int(i) for i in shape if i is not None]

		indices = array([i for i in slices]) if slices else None
		slices = array([slices[i] for i in slices]) if slices else None

		# Set attributes
		self.indices = indices
		self.slices = slices
		self.shape = shape
		self.size = prod(shape)
		self.ndim = len(shape)

		return

	def __initialize__(self,data=None,shape=None,dtype=None):
		'''
		Initialize class data with shape
		Args:
			data (array): Data of data
			shape (iterable[int]): Shape of data
			dtype (datatype): Data type of data
		'''

		# Set data
		self.data = data if data is not None else self.data
		self.shape = shape if shape is not None else self.shape
		self.size = prod(shape) if shape is not None else self.size
		self.ndim = len(shape) if shape is not None else self.ndim
		self.dtype = dtype if dtype is not None else self.dtype

		kwargs = {**self,**dict(data=self.data,shape=self.shape,dtype=self.dtype)}
		self.data = initialize(**kwargs)

		self.shape = self.data.shape if self.data is not None else None
		self.size = self.data.size if self.data is not None else None
		self.ndim = self.data.ndim if self.data is not None else None
		self.dtype = self.data.dtype if self.data is not None else None


		# Set functions
		
		# defaults = {
		# 	'data':self.data,
		# 	'parameters':self.parameters,
		# 	'slices':self.slices,
		# 	'constant':self.constant,'default':0,'lambda':0,'scale':1,'length':len(self.group)}
		# category = self.category
		# method = self.method
		# kwargs = deepcopy(self.kwargs)
		# kwargs.update({attr: kwargs.get(attr,defaults[attr]) for attr in defaults})
		# kwargs = deepcopy(kwargs)
		# for attr in kwargs:

		# 	if kwargs[attr] is None:
		# 		continue
		
		# 	if attr in ['lambda','scale']:
				
		# 		kwargs[attr] = array(kwargs[attr],dtype=self.dtype)
			
		# 	elif attr in ['constant']:
				
		# 		if not all(isinstance(kwargs[attr][i],dict) for i in kwargs[attr]):
		# 			axis = -1
		# 			kwargs[attr] = {axis:kwargs[attr]}
		# 		for axis in list(kwargs[attr]):
		# 			constants = kwargs[attr].pop(axis)
		# 			indices = array([int(i) for i in constants])
		# 			values = array([constants[i] for i in constants],dtype=self.dtype)
		# 			axis = int(axis) % self.ndim
		# 			indices = (*(slice(None),)*(axis-1),indices)
		# 			kwargs[attr][axis] = {'indices':indices,'values':values}

		if self.category in ['variable']:
			def func(parameters,self):
				return self.parameters*parameters[self.slices]
		else:
			def func(parameters,self):
				return self.parameters*self.data[self.slices]
		
		def constraint(parameters,self):
			return 0
		
		# if self.method in ['constrained']:
		# 	def func(parameters,self):
		# 		funcs = [cos,sin]
		# 		parameters = bound(self,parameters)
		# 		return {i: self.parameters*(
		# 				   parameters[(0,*self.slices[i][1:])]*
		# 				   funcs[self.slices[i][0]%len(funcs)](
		# 				   pi*parameters[(1,*self.slices[i][1:])]))
		# 				for i in self.slices}
			
		# 	def constraint(parameters,self):
		# 		return kwargs['lambda']*sum(
		# 			((parameters[kwargs['constant'][axis]['slices']] - kwargs['constant'][axis]['values'])**2).sum()
		# 			for axis in kwargs['constant'])
		
		# elif self.method in ['unconstrained']:
		# 	pass	
		
		# elif self.method in ['bounded']:
		# 	def func(parameters,self):
		# 		parameters = bound(self,parameters)
		# 		return {i: parameters[self.slices[i]] for i in self.slices}
		# 
		# else:
		# 	pass


		func = jit(func,self=self)
		constraint = jit(constraint,self=self)

		self.func = func
		self.constraint = constraint


		return


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
		ii) parameters for each group are sliced with parameter slices (slice(P),slice(D)) and reshaped into shape (P,D)
		iii) parameters for each group are modified with Parameter() function i.e) bounds, scaling, features
		iv) parameters for all groups are concatenated to [parameter_i = Parameters[slices_i]]
			with slices Parameters.slices = [slices]
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
				axis (int,iterable[int]): Axis of input parameter data to insert into class data
				parameters (iterable): parameters of parameter
				seed (int, key): Random seed for initialization
				random (str): Random type for initialization
				initialization (dict): Keyword arguments for initialization
				constant (dict[dict[str,object]]): constant indices and values of parameters, along axis, of the form {'axis':{'index':value}}
				args (iterable): Additional arguments for parameter
				system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
				model (object): Model with additional attributes for initialization
				kwargs (dict): Additional system keyword arguments
			model (object): Model with additional attributes for initialization
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		defaults = dict(data=None,indices=None,slices=None,shape=None,size=None,ndim=None,dtype=None)

		if data is None:
			data = {}

		__data__ = data

		setter(kwargs,dict(data=data,model=model,system=system,__data__=__data__),delimiter=delim,func=False)
		setter(kwargs,data,delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,defaults,delimiter=delim,func=False)
		super().__init__(**kwargs)

		self.__setup__()

		for parameter in self:
			print(parameter)
			print(self[parameter].slices,self[parameter].indices)
			print(self[parameter]())
			print(self[parameter](self[parameter]()))
			print()

		print(self())
		print()
		print(self(self()))
		print('----------------')

		return

	def __call__(self,parameters=None):
		'''
		Class data
		Args:
			parameters (array): parameters
		Returns:
			parameters (array,dict): parameters
		'''

		if parameters is None:
			return self.data

		parameters = array([i for parameter in self for i in self[parameter](parameters=parameters[self.slices[parameter]])],dtype=self.dtype)[self.indices]

		return parameters

	def __setup__(self,data=None,model=None):
		'''
		Setup attribute
		Args:
			data (dict): Dictionary of data corresponding to parameters groups, with dictionary values with properties:
			model (object): Model with additional attributes for initialization
		'''

		self.data = data if data is not None else self.data
		self.model = model if model is not None else self.model
		self.dtype = datatype(self.dtype)
		self.__data__ = data if data is not None else self.__data__

		# Set data
		for parameter in list(self):
			
			args = {**getattr(self,parameter,{}),**dict(model=self.model,system=self.system)}
			try:
				setattr(self,parameter,Parameter(**args))
			except AssertionError:
				delattr(self,parameter)



		# Get indices and slices
		indices = []
		slices = {}
		for parameter in self:
			if self[parameter].category in ['variable']:
				index = max(slices[parameter].stop for parameter in slices if slices[parameter] is not None) if slices else 0
				size = self[parameter].size
				slices[parameter] = slice(index,index+size)
			else:
				slices[parameter] = None

			indices.extend(self[parameter].indices)

		indices = array([indices.index(i) for i in range(len(indices))])

		self.indices = indices
		self.slices = slices

		# Get data
		data = []
		for parameter in self.slices:
			if self.slices[parameter] is None:
				continue
			parameter = self[parameter]()
			if parameter is not None:
				data.extend(parameter.ravel())
		data = array(data,dtype=self.dtype).ravel() if data else None

		# Set attributes
		self.data = data
		self.shape = self.data.shape if data is not None else None
		self.size = self.data.size if data is not None else None
		self.ndim = self.data.ndim if data is not None else None
		self.dtype = self.data.dtype if data is not None else None

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
		return self.__data__.__len__()

	def __iterdata__(self):
		return self.__data__.__iter__()

	def __setdata__(self,key,value):
		if key in self.__data__:
			self.__data__[key] = value
		return
	
	def __deldata__(self,key):
		if key in self.__data__:
			self.__data__.pop(key)
		return

	def __str__(self):
		return ' '.join([str(self.__data__[parameter]) for parameter in self.__data__])

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