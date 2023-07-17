#!/usr/bin/env python

# Import python modules
import os,sys
from copy import deepcopy
from functools import partial,wraps
import itertools


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,vfunc,switch,array,arange,bound
from src.utils import concatenate,addition,prod
from src.utils import initialize,slicing,datatype,to_index,to_position
from src.utils import pi,itg,scalars,arrays,delim,separ,cos,sin,exp

from src.iterables import indexer,inserter,setter,getter

from src.system import System,Dict
from src.io import load,dump,join,split


class Parameter(System):

	def __init__(self,data,*args,model=None,system=None,**kwargs):
		'''
		Initialize data of parameter
		Args:
			data (iterable): data of parameter
			string (str): Name of parameter
			category (str): category of parameter, allowed strings in ['variable','constant']
			method (str): method of parameter, allowed strings in ['unconstrained','constrained','coupled','bounded','time']
			group (iterable[str],iterable[iterable[str]]): iterable of groups associated with parameter grouping
			locality (str,iterable[str],dict[iterable,str]): locality of parameter across groups, allowed strings in ['local','global']
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
			wrapper=None,
			seed=None,random=None,initialization=None,constant=None,
			indices=None,slices=None,sort=None,instance=None,func=None,constraint=None,
			shape=None,size=None,ndim=None,dtype=None,			
			args=(),kwargs={}
			)

		setter(kwargs,dict(data=data,model=model,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,defaults,delimiter=delim,func=False)

		super().__init__(*args,**kwargs)

		self.__setup__()

		self.__initialize__()

		assert (self.size is not None), "Zero-size data"

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
		
		parameters = self.wrapper(self.func(parameters))

		return parameters	

	def constraints(self,parameters=None):
		'''
		Class constraints
		Args:
			parameters (array): parameters
		Returns:
			constraints (array): constraints
		'''

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
		self.locality = {group: self.locality if self.locality is None or isinstance(self.locality,str) else self.locality.get(subgroup) if isinstance(self.locality,dict) else self.locality[i] for i,group in enumerate(self.group)}
		self.model = self.model if self.model is not None else None
		self.attributes = [attr for attr in self.attributes if getattr(self.model,attr,None) is not None] if self.attributes is not None else ()
		self.kwargs = self.kwargs if self.kwargs is not None else {}

		self.parameters = array(self.parameters if self.parameters is not None else 1,dtype=self.dtype)
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

		# Get indices (indices of data for each unique string)
		localities = ['global']
		indices = {i: model.data[data] for i,data in enumerate(model.data)} if model is not None else {}
		indices = {indices[i].string: {j:indices[j] for j in indices if indices[j].string == indices[i].string} for i in indices}
		indices = {i: {
			'string':string,'label':separ.join((str(string),str(j))),
			'sort':i,'index':j,'slices':None,'group':None,
			'instance':indices[string][i],
			}
			for string in indices for j,i in enumerate(indices[string])}

		indices = {i: Dict(indices[i]) for i in indices}
		
		# Get slices (indices of parameter for each data in each group (locality dependent) (sort parameters into order of indices of data))
		# Get sort (indices of data for each parameter in each group (re-order/sort parameters from slices and func())		
		# Get indexes (indices of data for each parameter in each group)		
		# Get sizes (number of parameters per group for each group)
		# Get size (number of parameters per group)
		for i in list(indices):
			for subgroup in group:
				for attr in indices[i]:
					if indices[i][attr] in subgroup:
						indices[i].slices = group.index(subgroup)
						indices[i].group = subgroup
						break
			if any(indices[i][attr] is None for attr in indices[i]):
				indices.pop(i)

		sizes = []
		group = list(group)
		for subgroup in list(group):
			
			if locality.get(subgroup) in localities:
				size = 1
			else:
				size = len([j for j in indices if indices[j].group == subgroup])
			
			if not size:
				group.remove(subgroup)
				continue
			
			sizes.append(size)

		for i in indices:
		
			indices[i].sizes = sizes

			indices[i].size = indices[i].sizes[group.index(indices[i].group)]

			if locality.get(indices[i].group) in localities:
				indices[i].slices = sum(indices[i].sizes[:group.index(indices[i].group)])
			else:
				indices[i].slices = sum(indices[i].sizes[:group.index(indices[i].group)]) + [j for j in indices if indices[j].group == indices[i].group].index(i)


		slices = array([indices[i].slices for i in indices]) if indices else None
		sort = array([indices[i].sort for i in indices]) if indices else None
		instance = [indices[i].instance for i in indices]
		shape = [
			sum(max(indices[i].size for i in indices if indices[i].group == subgroup) for subgroup in group),
			*(getattr(model,attr) for attr in attributes)] if indices else None

		for i in indices:
			indices[i].shape = shape


		# Set attributes
		self.indices = indices
		self.slices = slices
		self.sort = sort
		self.instance = instance
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

		if self.size is not None:

			self.data = initialize(**kwargs)

			self.shape = self.data.shape if self.data is not None else None
			self.size = self.data.size if self.data is not None else None
			self.ndim = self.data.ndim if self.data is not None else None
			self.dtype = self.data.dtype if self.data is not None else None

		self.data = self.data

		# Set functions
		categories = ['variable']
		defaults = {
			'constant':self.constant,
			'lambda':0,
			'coefficients':[1,2*pi],
			'shift':[0,-pi/2],
			'sigmoid':1,
			'default':0,
			}
		self.kwargs.update({attr: self.kwargs.get(attr,kwargs.get(attr,defaults[attr])) for attr in defaults})

		for attr in self.kwargs:

			if self.kwargs.get(attr) is None:
				continue

			if attr in ['lambda','coefficients','shift','sigmoid','default']:
				
				self.kwargs[attr] = array(self.kwargs[attr],dtype=self.dtype)
			
			elif attr in ['constant']:
				
				if not all(isinstance(self.kwargs[attr][i],dict) for i in self.kwargs[attr]):
					axis = -1
					self.kwargs[attr] = {axis:self.kwargs[attr]}
				for axis in list(self.kwargs[attr]):
					constants = self.kwargs[attr].pop(axis)
					indices = array([int(i) for i in constants])
					values = array([constants[i] for i in constants],dtype=self.dtype)
					axis = int(axis)
					ax = self.ndim - axis if axis < 0 else axis
					indices = (*(slice(None),)*(max(0,ax-2)),indices,*(slice(None),)*(max(0,self.ndim - ax - 1)))
					self.kwargs[attr][axis] = {'indices':indices,'values':values}

		defaults = {}
		if self.method in ['time']:
			try:
				defaults.update({
					'tau':getattr(self.model,'tau',None),
					'scale':self.kwargs.get('scale',None)
					}
				)
			except:
				self.method = None
		
		self.kwargs.update(defaults)


		if self.category in categories:

			if self.method in ['bounded'] and all(self.kwargs.get(attr) is not None for attr in ['sigmoid']):
		
				def func(parameters):
					return self.parameters*bound(parameters[self.slices],scale=self.kwargs['sigmoid'])
					
			elif self.method in ['bounded']:

				def func(parameters):
					return self.parameters*bound(parameters[self.slices])

			elif self.method in ['constrained'] and all(self.kwargs.get(attr) is not None for attr in ['coefficients','shift','sigmoid']):
		
				def func(parameters):
					return self.parameters*bound(parameters[self.slices],scale=self.kwargs['sigmoid'])

			elif self.method in ['constrained']:					
		
				def func(parameters):
					return self.parameters*bound(parameters)

			elif self.method in ['coupled'] and all(self.kwargs.get(attr) is not None for attr in ['coefficients','shift','sigmoid']):
		
				def func(parameters):
					return self.parameters*bound((
						(self.kwargs['coefficients'][0]*parameters[self.slices][:self.slices.size//2])*
						cos(self.kwargs['coefficients'][1]*parameters[self.slices][self.slices.size//2:][None,...] + self.kwargs['shift'][...,None,None])
						).reshape(-1,*parameters.shape[1:]),scale=self.kwargs['sigmoid'])

			elif self.method in ['coupled']:					
		
				def func(parameters):
					return self.parameters*bound(parameters)

			elif self.method in ['unconstrained']:					
				
				def func(parameters):
					return self.parameters*parameters[self.slices]

			elif self.method in ['time'] and all(self.kwargs.get(attr) is not None for attr in ['tau','scale']):
				def func(parameters):
					return (1 - exp(-self.kwargs['tau']/self.kwargs['scale']))/2

			else:

				if isinstance(self.method,dict):
					func = self.method.get('func')
				elif isinstance(self.method,str):
					func = self.method
				else:
					func = None
				
				func = load(func)

				if func is None:
					def func(parameters):
						return self.parameters*parameters[self.slices]
				else:
					func = partial(func,self=self)
	

			if self.method in ['constrained'] and all(self.kwargs.get(attr) is not None for attr in ['lambda','constant']):
			
				def constraint(parameters):
					return self.kwargs['lambda']*sum(
						((parameters[self.kwargs['constant'][i]['indices']] - 
						  self.kwargs['constant'][i]['values'])**2).sum() 
						for i in self.kwargs['constant'])

			elif self.method in ['coupled'] and all(self.kwargs.get(attr) is not None for attr in ['lambda','constant']):
			
				def constraint(parameters):
					return self.kwargs['lambda']*sum(
						((parameters[self.kwargs['constant'][i]['indices']] - 
						  self.kwargs['constant'][i]['values'])**2).sum() 
						for i in self.kwargs['constant'])

			elif self.method in ['unconstrained']:

					def constraint(parameters):
						return self.kwargs['default']
				
			else:

				if isinstance(self.method,dict):
					constraint = self.method.get('constraint')
				elif isinstance(self.method,str):
					constraint = None
				else:
					constraint = None
				
				constraint = load(constraint)

				if constraint is None:
			
					def constraint(parameters):
						return self.kwargs['default']

				else:
					constraint = partial(constraint,self=self)

		else:
		
			def func(parameters):
				return self.parameters*self.data[self.slices]

			
			def constraint(parameters):
				return self.kwargs['default']


		def wrapper(parameters,*args,**kwargs):
			return parameters

		wrapper = self.wrapper if self.wrapper is not None else wrapper


		self.func = jit(func)
		
		self.constraint = jit(constraint)

		self.wrapper = jit(wrapper)

		return

	def __str__(self):
		return str(self.string)


class Parameters(System):

	__data__ = {}

	def __init__(self,data=None,model=None,system=None,**kwargs):
		'''
		Initialize data of parameters

		Parameters are split into groups with names [parameter], and can be accessed as 
		attributes Parameters.parameter, or items Parameters[parameter],
		and are contained in Parameters.data = {parameter: Parameter()} as Parameter() instances

		Setup parameters such that calling the Parameters(parameters) class with input parameters 
		i) 		parameters array of size (G*P*D),
					for G groups of P parameters, each of dimension D, 
					for parameter groups with category in categories = ['variable'],
					P,D may be group dependent, and depend on Parameter.locality, Parameter.model and Parameter.attributes 
		ii) 	parameters for each group are sliced with parameter slices (slice(P*D)) and reshaped into shape (P,D)
		iii) 	parameters for each group are modified with Parameter() function i.e) bounds, scaling, features
		iv) 	parameters for all groups are concatenated to [parameter_i = Parameters[slices_i]][sort]
					with slices Parameters.slices = [slices], and sorted with sort Parameters.sort = [sort]
		
		Args:
			data (dict): Dictionary of data corresponding to parameters groups, with dictionary values with properties:
				data (iterable): data of parameter
				string (str): Name of parameter
				category (str): category of parameter, allowed strings in ['variable','constant']
				method (str): method of parameter, allowed strings in ['unconstrained','constrained','coupled','bounded','time']
				group (iterable[str],iterable[iterable[str]]): iterable of groups associated with parameter grouping
				locality (str,iterable[str],dict[iterable,str]): locality of parameter across groups, allowed strings in ['local','global']
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
			string=None,category=None,method=None,
			group=None,locality=None,bounds=None,attributes=None,axis=None,
			parameters=None,
			seed=None,random=None,initialization=None,constant=None,
			indices=None,slices=None,sort=None,instance=None,func=None,constraint=None,wrapper=None,
			shape=None,size=None,ndim=None,dtype=None,			
			args=(),kwargs={}
			)

		data = data if data is not None else None
		__data__ = data if data is not None else {}

		setter(kwargs,dict(data=data,model=model,system=system,__data__=__data__),delimiter=delim,func=False)
		setter(kwargs,data,delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,defaults,delimiter=delim,func=False)
		super().__init__(**kwargs)

		self.__setup__()

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

		parameters = self.wrapper(self.func(parameters)[self.sort])

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

		
		# Set dtype
		dtype = datatype(self.dtype)

		# Set slices and sort and indices
		categories = ['variable']
		indices = {}
		group = []
		slices = []
		sort = []
		instance = []
		for i,parameter in enumerate(self):
			if self[parameter].category in categories:
				idx = self[parameter].indices
				grp = self[parameter].group
				slc = self[parameter].size
				srt = self[parameter].sort
				ins = self[parameter].instance
			else:
				idx = None
				grp = self[parameter].group							
				slc = None
				srt = self[parameter].sort
				ins = self[parameter].instance				

			idx = idx if idx is not None else {}
			grp = tuple(grp)
			slc = [sum(i[-1] for i in slices),slc] if slc is not None else [0,0]
			srt = [int(i) for i in srt]
			ins = [i for i in ins]

			indices.update(idx)
			group.append(grp)
			slices.append(slc)
			sort.extend(srt)
			instance.append(ins)

		group = group
		slices = [[*i] for i in slices]
		sort = array([sort.index(i) for i in sorted(sort)])
		instance = [i for i in instance]

		for j,i in enumerate(list(indices)):
			indices[i] = (
				max((indices[k][0] for l,k in enumerate(indices) if l < j),default=-1) + ((indices[i].index == 0) or (indices[i].size != 1)),
				(sum(self[parameter].slices.size for parameter in self if self[parameter].slices is not None),*indices[i].shape[1:])
				)

		for j,i in enumerate(list(indices)):
			k,shape = indices.pop(i)
			for j in (itertools.product(*(range(i) for i in shape[1:])) if prod(shape[1:]) else ((),)):
				indices[(i,*j)] = (k*max(prod(shape[1:]),1) + to_index(j,shape[1:]),shape)
		
		for j,i in enumerate(list(indices)):
			k,shape = indices.pop(i)
			indices[to_index(i[::-1],shape[::-1])] = k


		# Set func and constraint
		funcs = []
		constraints = []
		for parameter in self:

			func = self[parameter]
			funcs.append(func)

			func = self[parameter].constraints
			constraints.append(func)

		def func(parameters,slices=slices,funcs=funcs,dtype=dtype):
			return concatenate([func(slicing(parameters,*i)) for i,func in zip(slices,funcs)])

		def constraint(parameters,slices=slices,funcs=constraints,dtype=dtype):
			return addition(array([func(slicing(parameters,*i)) for i,func in zip(slices,funcs)]))

		def wrapper(parameters,*args,**kwargs):
			return parameters

		wrapper = self.wrapper if self.wrapper is not None else wrapper


		# Get data
		data = []
		for parameter in self:
			if self[parameter].category not in categories:
				continue
			parameter = self[parameter]()
			data.extend(parameter)

		data = array(data,dtype=self.dtype).ravel()

		# Get parameters
		parameters = []
		for parameter in self:
			parameter = self[parameter].parameters
			if parameter.size > 1:
				parameters.extend(parameter)	
			else:
				parameters.append(parameter)	

		parameters = array(parameters,dtype=self.dtype).ravel() if parameters else None


		# Set attributes
		self.data = data
		self.indices = indices
		self.slices = slices
		self.sort = sort
		self.instance = instance
		self.group = group
		self.func = func
		self.constraint = constraint
		self.parameters = parameters
		self.wrapper = wrapper
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