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
			method (str): method of parameter, allowed strings in ['unconstrained','constrained','bounded']
			group (iterable[iterable[str]]): iterable of groups associated with parameter grouping
			locality (str): locality of parameter across groups, allowed strings in ['shared']
			bounds (iterable[object]): Bounds of parameters
			parameters (iterable): parameters of parameter
			seed (int, key): Random seed for initialization
			random (str): Random type for initialization
			initialization (dict): Keyword arguments for initialization
			args (iterable): Additional arguments for parameter
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		defaults = dict(
			string=None,category=None,method=None,
			group=None,locality=None,bounds=None,
			parameters=None,
			seed=None,random=None,initialization=None,
			args=(),kwargs={}
			)

		setter(kwargs,defaults,delimiter=delim,func=False)

		super().__init__(*args,**kwargs)

		self.data = array(data,dtype=datatype(self.dtype)) if data is not None else None

		self.shape = self.data.shape if self.data is not None else None
		self.size = self.data.size if self.data is not None else None
		self.ndim = self.data.ndim if self.data is not None else None
		self.dtype = self.data.dtype if self.data is not None else None

		self.string = self.string if self.string is not None else None
		self.group = (*((*group,) for group in self.group),) if self.group is not None else ()
		self.string = self.parameters if self.parameters is not None else 1


		# Set functions
		method = self.method
		locality = self.locality
		group = self.group
		kwargs = self.kwargs
		dtype = self.dtype
		default = array(0,dtype=dtype)
		def func(parameters):
				return {i: parameters[i] for i in range(len(group))}
		def constraints(parameters):
			return default
		if method in ['constrained']:
			defaults = {'lambda':0,'constraints':{}}
			kwargs.update(defaults)
			kwargs['lambda'] = array(kwargs['lambda'])
			kwargs['constraints'] = [array([int(i) for i in constraints]),array([constraints[i] for i in constraints])]
			if locality in ['shared']:
				def func(parameters):
					parameters = bound(parameters)
					return {group[0]: parameters[0]*cos(pi*parameters[1]), group[1]:parameters[0]*sin(pi*parameters[1])}
				def constraints(parameters):
					return kwargs['lambda']*((parameters[...,kwargs['constraints'][0]] - kwargs['constraints'][1])**2).sum()
			else:
				def func(parameters):
					parameters = bound(parameters)
					return {group[0]: parameters[0]*cos(pi*parameters[1]), group[1]:parameters[0]*sin(pi*parameters[1])}
				def constraints(parameters):
					return kwargs['lambda']*((parameters[...,kwargs['constraints'][0]] - kwargs['constraints'][1])**2).sum()
		elif method in ['unconstrained']:
			pass	
		elif method in ['bounded']:
			def func(parameters):
				parameters = bound(parameters)
				return {i: parameters[i] for i in range(len(group))}			
		else:
			pass

		self.func = func
		self.constraints = constraints

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

		if data is not None:
			self.data = data
		if parameters is not None:
			self.parameters = parameters

		data = self.parameters*self.func(self.data)

		return data	

	def init(self,shape=None):
		'''
		Initialize class data with shape
		Args:
			shape (iterable[int]): Shape of data
		'''

		if self.locality in ['shared']:
			self.shape = [len(self.group),*shape[1:]]
		else:
			self.shape = [len(self.group),shape[0],*shape[1:]]

		self.size = prod(self.shape)
		self.ndim = len(self.shape)

		if self.data is not None:
			ndim = max(0,self.ndim - self.data.ndim - 1)
			self.data = self.data.reshape(self.data.shape,*(1,)*ndim)
			for i in range(ndim,self.ndim):
				self.data = repeat(self.data,self.shape[i],i)

		self.data = initialize(self.data,self.shape,self.initialization,dtype=self.dtype)

		self.shape = self.data.shape if self.data is not None else None
		self.size = self.data.size if self.data is not None else None
		self.ndim = self.data.ndim if self.data is not None else None
		self.dtype = self.data.dtype if self.data is not None else None

		return

	def __str__(self):
		return str(self.string)


class Parameters(System):
	def __init__(self,data=None,system=None,**kwargs):
		'''
		Initialize data of parameters

		Setup parameters such that calling the class with input parameters 
		i) parameters array of shape (L*K*-1),
			for L groups of K parameters, each of dimension -1 (or higher dimensional shape)
		ii) parameters is reshaped into shape (dimension_group, dimension_parameter, dimension_axis) = (L,K,M)
		iii) parameters are sliced with 
			slices = {group: [slices_group,slices_parameter,slices_axis]} 
			and assigned to the L groups of data
			data = {group : array of shape (size_parameter,size_axis)}
		i.v) data is modified with features i.e) bounds, scaled
		v) indices,parameters,data are returned, and then can be indexed with indices = [indices_i]
			where index_i = [index_group_i,index_parameter_i,index_axis_i] for each parameter of data

		The initialization sets class attributes of shape,slices,indices, 
		and adds indices_i to model.data[i] class attributes

		Args:
			data (dict): Dictionary of data corresponding to parameters groupings, with dictionary values with properties:
				data (iterable): data of parameter
				string (str): Name of parameter
				category (str): category of parameter, allowed strings in ['variable','constant']
				method (str): method of parameter, allowed strings in ['unconstrained','constrained','bounded']
				group (iterable[iterable[str]]): iterable of groups associated with parameter grouping
				locality (str): locality of parameter across groups, allowed strings in ['shared']
				bounds (iterable[object]): Bounds of parameters
				parameters (iterable): parameters of parameter
				seed (int, key): Random seed for initialization
				random (str): Random type for initialization
				initialization (dict): Keyword arguments for initialization
				args (iterable): Additional arguments for parameter
				system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
				kwargs (dict): Additional system keyword arguments
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		defaults = dict(model=None,slices=None,indices=None,func=None,constraints=None)

		if data is None:
			data = {}

		setter(kwargs,dict(data=data,system=system),delimiter=delim,func=False)
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
			data (array): Data
		'''


		if data is None:
			data = self.parameters
		else:
			self.parameters = data
			data = data.reshape(self.shape)
			for parameter in self:
				if self.index[parameter]:
					inserter(self.index[parameter],indexer(self.slices[parameter],self.parameters[])
			data = self.func(data)
			for 


		# if data is None:
		# 	data = indexer(self.index,self).ravel()
		# else:
		# 	inserter(self.index,data.reshape(indexer(self.index,self).shape),self)

		# parameters = self.parameters if self.parameters is not None else 1

		# data = parameters*data.ravel()



		return data



	def __setup__(self,**kwargs):
		'''
		Setup attribute
		Args:
			kwargs (dict): Additional keyword arguments
		'''

		if (not len(self)) or (self.model is None):
			return

		# Set data
		for parameter in self:
			args = {**getattr(self,parameter,{}),**dict(system=self.system)}
			setattr(self,parameter,Parameter(**args))

		# Get strings
		data = [data.string for data in self.model.data]
		strings = {}
		for string in data:
			if string in strings:
				continue
			
			strings[string] = [i for i,substring in enumerate(data) if substring == string]

		# Get slices and indices
		categories = ['variable']
		data = {parameter: self[parameter].group for parameter in self}
		slices = {}
		indices = {}
		for parameter in data:
			group = data[parameter]
			for subgroup in group:
				for string in strings:
					for i in strings[string]:

						if not any(i in subgroup
							for i in (string,*('%s_%s'%(string,str(i)) 
							for i in range(len(strings[string]))))):
							continue

						index = (parameter,group.index(group),strings[string].index(i))

						indices[i] = index


			slices[parameter] = [parameter,(slice(None),slice(None),slice(0))]
						slices[parameter] = {subgroup: [slice()]}

			groups[group] = {subgroup: list(range(len([i for string in strings for i in strings[string]
				if any(i in subgroup
					for i in (string,*('%s_%s'%(string,str(i)) 
					for i in range(len(strings[string])))))
				]))) for subgroup in group}
		print('----')
		print(strings)
		print(groups)


		# Get shape
		shape = (len(self.model),self.model.M)

		# Set parameters
		for parameter in self:
			self[parameter].init(shape[parameter])

		slices = {group: [slice(i,i+1),slice(0,len(index[group]))] for i,group in enumerate(index)}

		# Get attributes
		ndim = max(len(slices[group]) for group in slices)
		shape = [max(slices[group][i].stop for group in slices) for i in range(ndim)]
		size = prod(shape)


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