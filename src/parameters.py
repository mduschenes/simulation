#!/usr/bin/env python

# Import python modules
import os,sys
from copy import deepcopy
from functools import partial,wraps
from math import prod


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,vfunc,switch,array,arange,bound
from src.utils import concatenate,addition
from src.utils import initialize,slicing,datatype
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
			group (iterable[str],iterable[iterable[str]]): iterable of groups associated with parameter grouping
			locality (str,iterable[str],dict[iterable,str]): locality of parameter across groups, allowed strings in ['shared']
			bounds (iterable[object]): Bounds of parameters
			attributes (iterable[str]): Model attributes for additional dimensions of parameter
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
			group=None,locality=None,bounds=None,attributes=None,axis=None,
			parameters=None,
			seed=None,random=None,initialization=None,constant=None,
			slices=None,indices=None,func=None,constraint=None,
			shape=None,size=None,ndim=None,dtype=None,			
			args=(),kwargs={}
			)

		setter(kwargs,dict(data=data,model=model,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,defaults,delimiter=delim,func=False)

		super().__init__(*args,**kwargs)

		self.__setup__()

		self.__initialize__()

		assert (self.indices is not None), "Zero-size data, No indices for parameter"

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

		parameters = parameters.reshape(-1,*self.shape[1:])
		
		parameters = self.func(parameters)

		return parameters	

	def constraints(self,parameters=None):
		'''
		Class constraints
		Args:
			parameters (array): parameters
		Returns:
			constraints (array): constraints
		'''
		if parameters is None:
			return 0

		parameters = parameters.reshape(-1,*self.shape[1:])

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
		self.category = self.category if self.category is not None else None
		self.method = self.method if self.method is not None else None
		self.group = (*((*group,) if not isinstance(group,str) else (group,) for group in self.group),)  if self.group is not None else ()
		self.locality = {group: self.locality if isinstance(self.locality,str) else self.locality.get(subgroup) if isinstance(self.locality,dict) else self.locality[i] for i,group in enumerate(self.group)} if self.locality is not None else {}
		self.model = self.model if self.model is not None else None
		self.attributes = [attr for attr in self.attributes if getattr(self.model,attr,None) is not None] if self.attributes is not None else ()
		self.kwargs = self.kwargs if self.kwargs is not None else {}

		self.parameters = self.parameters if self.parameters is not None else 1
		self.dtype = self.data.dtype if self.data is not None else None

		# Set data
		category = self.category
		method = self.method
		group = self.group
		locality = self.locality
		model = self.model
		attributes = self.attributes
		kwargs = self.kwargs
		dtype = self.dtype

		# Get strings (indices of data for each unique string)
		strings = {i: data.string for i,data in enumerate(model.data)} if model is not None else {}
		strings = {strings[i]: [j for j in strings if strings[j] == strings[i]] for i in strings}
		strings = {i: {
			'string':string,'label':separ.join((str(string),str(j))),
			'indices':i,'slices':None,'group':None,
			}
			for string in strings for j,i in enumerate(strings[string])}
		
		# Get slices (indices of parameter for each data in each group (locality dependent) (sort parameters into order of indices of data))
		# Get indices (indices of data for each parameter in each group (re-order/sort parameters from slices and func())		
		# Get sizes (number of parameters per group for each group)
		# Get size (number of parameters per group)
		for i in list(strings):
			for subgroup in group:
				for attr in strings[i]:
					if strings[i][attr] in subgroup:
						strings[i]['slices'] = group.index(subgroup)
						strings[i]['group'] = subgroup
						break
			if any(strings[i][attr] is None for attr in strings[i]):
				strings.pop(i)

		sizes = []
		group = list(group)
		for subgroup in list(group):
			
			if locality.get(subgroup) in ['shared']:
				size = 1
			else:
				size = len([j for j in strings if strings[j]['group'] == subgroup])
			
			if not size:
				group.remove(subgroup)
				continue
			
			sizes.append(size)

		for i in strings:
		
			strings[i]['sizes'] = sizes

			strings[i]['size'] = strings[i]['sizes'][group.index(strings[i]['group'])]

			if locality.get(strings[i]['group']) in ['shared']:
				strings[i]['slices'] = sum(strings[i]['sizes'][:group.index(strings[i]['group'])])
			else:
				strings[i]['slices'] = sum(strings[i]['sizes'][:group.index(strings[i]['group'])]) + [j for j in strings if strings[j]['group'] == strings[i]['group']].index(i)


		slices = array([strings[i]['slices'] for i in strings]) if strings else None
		indices = array([strings[i]['indices'] for i in strings]) if strings else None
		shape = [
			sum(max(strings[i]['size'] for i in strings if strings[i]['group'] == subgroup) for subgroup in group),
			*(getattr(model,attr) for attr in attributes)] if strings else None
	
		# Set attributes
		self.slices = slices
		self.indices = indices
		self.shape = shape if shape is not None else None
		self.size = prod(shape) if shape is not None else None
		self.ndim = len(shape) if shape is not None else None

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

		self.data = self.data

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
		
		if self.method in ['bounded']:
			def func(parameters,self):
				return self.parameters*bound(parameters[self.slices])

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

		# func = jit(func,self=self)
		# constraint = jit(constraint,self=self)

		func = partial(func,self=self)
		constraint = partial(constraint,self=self)

		self.func = func
		self.constraint = constraint


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
			P,D may be group dependent, and depend on Parameter.locality, Parameter.model and Parameter.attributes 
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
				group (iterable[str],iterable[iterable[str]]): iterable of groups associated with parameter grouping
				locality (str,iterable[str],dict[iterable,str]): locality of parameter across groups, allowed strings in ['shared']
				bounds (iterable[object]): Bounds of parameters
				attributes (iterable[str]): Model attributes for additional dimensions of parameter
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

		defaults = dict(
			data=None,__data__={},
			slices=None,indices=None,func=None,constraint=None,
			shape=None,size=None,ndim=None,dtype=None,
			)

		data = data if data is not None else None
		__data__ = data if data is not None else {}

		setter(kwargs,dict(data=data,model=model,system=system,__data__=__data__),delimiter=delim,func=False)
		setter(kwargs,data,delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,defaults,delimiter=delim,func=False)
		super().__init__(**kwargs)

		self.__setup__()

		# for parameter in self:
		# 	print(parameter)
		# 	print(self[parameter].shape,self[parameter].slices,self[parameter].indices)
		# 	# print(self[parameter].data)#())
		# 	print(self[parameter]())
		# 	# print(self[parameter](self[parameter].data))#()))
		# 	print(self[parameter](self[parameter]()))
		# 	print()

		# if (self.data is not None) and (self.data.size > 0):
		# 	print(self)
		# 	print(self.slices,self.indices)
		# 	# print(self.data)#())
		# 	print(self())
		# 	print()
		# 	# print(self(self.data))#())
		# 	print(self(self()))
		# 	# print(self.constraints(self.data))#())
		# 	print(self.constraints(self()))
		# 	print('----------------')
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

		parameters = self.func(parameters)[self.indices]

		return parameters

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
		Setup attribute
		Args:
			data (dict): Dictionary of data corresponding to parameters groups, with dictionary values with properties:
			model (object): Model with additional attributes for initialization
		'''

		self.data = data if data is not None else self.data
		self.model = model if model is not None else self.model
		self.dtype = datatype(self.dtype)
		self.__data__ = data if data is not None else self.__data__

		# Set parameters
		for parameter in list(self):
			
			args = {**getattr(self,parameter,{}),**dict(model=self.model,system=self.system)}
			
			try:
				setattr(self,parameter,Parameter(**args))
			except AssertionError:
				delattr(self,parameter)

		
		# Set indices and slices
		slices = []
		indices = []
		for i,parameter in enumerate(self):
			if self[parameter].category in ['variable']:
				slc = self[parameter].size
				index = self[parameter].indices
			else:
				slc = None
				index = self[parameter].indices


			slc = [max((i[-1] for i in slices),default=0),slc] if slc is not None else [0,0]
			index = [i for i in index]

			slices.append(slc)
			indices.extend(index)

		# slices = array([[*i] for i in slices])
		# indices = array([indices.index(i) for i in range(len(indices))])

		# Set func and constraint
		funcs = []
		constraints = []
		index = arange(len(self))
		for parameter in self:
			funcs.append(self[parameter])
			constraints.append(self[parameter].constraints)


		# funcs = [jit(func) for func in funcs]
		# constraints = [jit(func) for func in constraints]

		# func = vfunc(func,in_axes=[0,0,None])
		# def func(parameters,index=index,slices=slices,func=func):
		# 	return concatenate(func(index,slices,parameters))

		size = min(len(funcs),len(slices),len(indices))

		slices = [[*i] for i in slices]
		indices = array([indices.index(i) for i in range(len(indices))])
		
		def func(parameters):#,index=index,slices=slices,function=funcs):
			
			# return concatenate([switch(i,function,slicing(parameters,*s)) for i,s in zip(index,slices)])
			# return concatenate([function[0](slicing(parameters,*s[0])) for i,s,f in zip(index,slices,function)])
			return concatenate([funcs[i](slicing(parameters,*slices[i])) for i in range(size)])

		def constraint(parameters):#,index=index,slices=slices,function=constraints):
			# return addition(array([switch(i,function,slicing(parameters,*s)) for i,s in zip(index,slices)]))
			# return addition(array([f(slicing(parameters,*s)) for i,s,f in zip(index,slices,constraints)]))
			return addition(array([constraints[i](slicing(parameters,*slices[i])) for i in range(size)]))
 

		# func = jit(func,index=index,slices=slices,function=funcs)
		# constraint = jit(constraint,index=index,slices=slices,function=constraints)

		# func = jit(func)
		# constraint = jit(constraint)


		self.indices = indices
		self.slices = slices
		self.func = func
		self.constraint = constraint

		# Set data
		data = []
		for parameter in self:
			if self[parameter].category not in ['variable']:
				continue
			# parameter = self[parameter].data#()
			parameter = self[parameter]()
			if parameter is None:
				continue
			data.extend(parameter.reshape(-1))
		data = array(data,dtype=self.dtype).reshape(-1) if data else None

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