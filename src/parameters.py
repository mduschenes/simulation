#!/usr/bin/env python

# Import python modules
import os,sys
from functools import partial,wraps
import itertools


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,vfunc,switch,array,arange,empty,bound
from src.utils import concatenate,addition,prod
from src.utils import initialize,slicing,datatype,to_index,to_position
from src.utils import pi,itg,scalars,arrays,delim,separ,cos,sin,exp

from src.iterables import indexer,inserter,setter,getter

from src.system import System,Dict
from src.io import load,dump,join,split


class Parameter(System):

	defaults = dict(
			string=None,variable=None,method=None,
			local=None,group=None,
			parameters=None,
			seed=None,random=None,bounds=None,attributes=None,axis=None,transpose=None,
			initialization=None,constants=None,
			indices=None,func=None,constraint=None,
			shape=None,size=None,ndim=None,dtype=None,			
			args=(),kwargs={}
			)

	def __init__(self,data,*args,system=None,**kwargs):
		'''
		Initialize data of parameter
		Args:
			data (iterable): data of parameter, if None, shape must be not None to initialize data
			string (str): Name of parameter
			variable (bool): Parameter is variable or constant
			method (str): method of parameter, allowed strings in ['unconstrained','constrained','bounded','time']
			group (iterable[str],iterable[iterable[str]]): iterable of groups associated with parameter grouping			
			local (bool,dict[iterable,bool]): locality of parameter
			bounds (iterable[object]): Bounds of parameters
			attributes (iterable[str]): Attributes for additional dimensions of parameter
			axis (int,iterable[int]): Axis of input parameter data to insert into class data
			parameters (iterable): parameters of parameter
			seed (int, key): Random seed for initialization
			random (str): Random type for initialization
			initialization (dict): Keyword arguments for initialization
			constants (dict[dict[str,object]]): constant indices and values of parameters, along axis, of the form {'axis':{'index':value}}			
			args (iterable): Additional arguments for parameter
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		kwargs = Dict(kwargs)

		setter(kwargs,dict(data=data,system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(*args,**kwargs)

		self.__setup__()

		self.__initialize__()

		return

	def __call__(self,parameters=None,*args,**kwargs):
		'''
		Class parameters
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			parameters (array): parameters
		'''

	
		if parameters is not None:
			return self.func(*args,parameters=parameters,**kwargs)
		else:
			return self.func(*args,**kwargs)

	def constraints(self,parameters=None,*args,**kwargs):
		'''
		Class constraints
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			constraints (array): constraints
		'''

		if parameters is not None:
			return self.constraint(*args,parameters=parameters,**kwargs)
		else:
			return self.constraint(*args,**kwargs)


	def __setup__(self):
		'''
		Setup class attributes
		'''

		# Get data
		self.dtype = datatype(self.dtype)		
		self.shape = self.shape if self.shape is not None else None
		self.transpose = self.transpose if self.transpose is not None else None
		self.data = array(self.data,dtype=self.dtype) if self.data is not None else empty(self.shape,dtype=self.dtype) if self.shape is not None else None

		self.shape = self.shape if self.shape is not None else self.data.shape if self.data is not None else None
		self.size = self.size if self.size is not None else self.data.size if self.data is not None else prod(self.shape) if self.shape is not None else None
		self.ndim = self.ndim if self.ndim is not None else self.data.ndim if self.data is not None else len(self.shape) if self.shape is not None else None
		self.dtype = self.dtype if self.dtype is not None else self.data.dtype if self.data is not None else None

		self.string = self.string if self.string is not None else None
		self.variable = self.variable if self.variable is not None else None
		self.method = self.method if self.method is not None else None
		self.group = (*((*group,) if not isinstance(group,str) else (group,) for group in self.group),)  if self.group is not None else ()		
		self.local = self.local if isinstance(self.local,dict) else {group:self.local for group in self.group}
		self.attributes = [attr for attr in self.attributes if isinstance(attr,int) or getattr(self,attr,None) is not None] if self.attributes is not None else None
		self.kwargs = self.kwargs if self.kwargs is not None else {}

		# Set functions
		kwargs = {**self,**dict(data=self.data,shape=self.shape,dtype=self.dtype)}

		defaults = {
			'constants':self.constants,
			'lambda':0,
			'coefficients':[1,2*pi],
			'shift':[0,-pi/2],
			'sigmoid':1,
			'default':0,
			'wrapper':lambda parameters: parameters
			}
		self.kwargs.update({attr: self.kwargs.get(attr,kwargs.get(attr,defaults[attr])) for attr in defaults})

		for attr in self.kwargs:

			if self.kwargs.get(attr) is None:
				continue

			if attr in ['lambda','coefficients','shift','sigmoid','default']:
				
				self.kwargs[attr] = array(self.kwargs[attr],dtype=self.dtype)
			
			elif attr in ['constants']:
				
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
			
			elif attr in ['wrapper']:
				if self.data is None:
					def wrapper(parameters):
						return parameters
				elif self.indices is not None and self.parameters is not None:
					def wrapper(parameters):
						return self.parameters*parameters[self.indices]
				elif self.indices is None and self.parameters is not None:
					def wrapper(parameters):
						return self.parameters*parameters
				elif self.indices is not None and self.parameters is None:
					def wrapper(parameters):
						return parameters[self.indices]
				elif self.indices is None and self.parameters is None:
					def wrapper(parameters):
						return parameters
				self.kwargs[attr] = jit(wrapper)


		defaults = {}
		if self.method in ['time']:
			try:
				defaults.update({
					'tau':getattr(self,'tau',None),
					'scale':self.kwargs.get('scale',None)
					}
				)
			except:
				self.method = None
		
		self.kwargs.update(defaults)


		if self.variable:

			if self.method in ['bounded'] and all(self.kwargs.get(attr) is not None for attr in ['sigmoid']):
		
				if self.parameters is not None:
					def func(parameters,*args,**kwargs):
						return bound(self.kwargs['wrapper'](parameters),scale=self.kwargs['sigmoid'])
					
			elif self.method in ['bounded']:

				def func(parameters,*args,**kwargs):
					return bound(self.kwargs['wrapper'](parameters))

			elif self.method in ['constrained'] and all(self.kwargs.get(attr) is not None for attr in ['coefficients','shift','sigmoid']):
		
				def func(parameters,*args,**kwargs):
					return bound(self.kwargs['wrapper'](parameters),scale=self.kwargs['sigmoid'])

			elif self.method in ['constrained']:					
		
				def func(parameters,*args,**kwargs):
					return bound(self.kwargs['wrapper'](parameters))

			elif self.method in ['unconstrained']:					
				
				def func(parameters,*args,**kwargs):
					return self.kwargs['wrapper'](parameters)

			elif self.method in ['time'] and all(self.kwargs.get(attr) is not None for attr in ['tau','scale']):
				
				def func(parameters,*args,**kwargs):
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
					def func(parameters,*args,**kwargs):
						return self.kwargs['wrapper'](parameters)
				else:
					func = partial(func,self=self)
	

			if self.method in ['constrained'] and all(self.kwargs.get(attr) is not None for attr in ['lambda','constants']):
			
				def constraint(parameters,*args,**kwargs):
					return self.kwargs['lambda']*sum(
						(((parameters[self.indices])[self.kwargs['constants'][i]['indices']] - 
						  self.kwargs['constants'][i]['values'])**2).sum() 
						for i in self.kwargs['constants'])

			elif self.method in ['unconstrained']:

					def constraint(parameters,*args,**kwargs):
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
			
					def constraint(parameters,*args,**kwargs):
						return self.kwargs['default']

				else:
					constraint = partial(constraint,self=self)

		else:
			
			if self.data is not None:

				def func(parameters,*args,**kwargs):
					return self.kwargs['wrapper'](self.data)
			else:
				def func(parameters,*args,**kwargs):
					return self.data

			def constraint(parameters,*args,**kwargs):
				return self.kwargs['default']


		parameters = self.data if self.data is not None else None

		func = jit(func,parameters=parameters)
		constraint = jit(constraint,parameters=parameters)

		self.func = func
		self.constraint = constraint

		return

	def __initialize__(self,data=None,parameters=None,indices=None):
		'''
		Initialize class data with shape
		Args:
			data (array): Data of class, if None, shape must be not None to initialize data
			parameters (array): Parameters of class
			indices (array): Indices of parameters of class
		'''

		# Set data
		self.data = data if data is not None else self.data
		self.parameters = parameters if parameters is not None else self.parameters
		self.indices = indices if indices is not None else self.indices

		if self.data is not None or self.shape is not None:

			kwargs = {**self,**dict(data=self.data,shape=self.shape,dtype=self.dtype)}


			print('initializing',self)
			for kwarg in kwargs:
				print(kwarg,kwargs[kwarg])
			print('-----',self.data,self.shape)
			print()
			self.data = initialize(**kwargs)

		# if self.string == 'z':
		# 	print('s',self.shape,self.data)

		self.__setup__()

		return

	def __str__(self):
		if self.string is None:
			string = self.__class__.__name__
		else:
			string = str(self.string)
		return string 


class Parameters(System):

	defaults = {}
	data = {}
	indices = {}
	parameters = None
	shape = None
	size = None
	ndim = None
	dtype = None

	def __init__(self,parameters=None,system=None,**kwargs):
		'''
		Initialize structure of parameters

		Parameters are split into groups with names [parameter], and can be accessed as 
		attributes Parameters.parameter, or items Parameters[parameter],
		and are contained in Parameters.data = {parameter: Parameter()} as Parameter() instances

		Setup parameters such that calling the Parameters(parameters) class with input parameters 
		i) 		parameters array of size (G*P*D),
					for G groups of P parameters, each of dimension D, 
					for variable parameter groups
					P,D may be group dependent, and depend on Parameter.local, Parameter.model and Parameter.attributes 
		ii) 	parameters for each group are sliced with parameter slices (slice(P*D)) and reshaped into shape (P,D)
		iii) 	parameters for each group are modified with Parameter() function i.e) bounds, scaling, features
		iv) 	parameters for all groups are concatenated to [parameter_i = Parameters[slices_i]][sort]
					with slices Parameters.slices = [slices], and sorted with sort Parameters.sort = [sort]
		
		Args:
			parameters(dict): Dictionary of Parameters instances, with class attributes
				data (iterable): data of parameter
				string (str): Name of parameter
				variable (bool): Parameter is variable or constant
				method (str): method of parameter, allowed strings in ['unconstrained','constrained','bounded','time']
				group (iterable[str],iterable[iterable[str]]): iterable of groups associated with parameter grouping
				local (bool,dict[iterable,bool]): locality of parameter
				bounds (iterable[object]): Bounds of parameters
				attributes (iterable[str]): Model attributes for additional dimensions of parameter
				axis (int,iterable[int]): Axis of input parameter data to insert into class data
				parameters (iterable): parameters of parameter
				seed (int, key): Random seed for initialization
				random (str): Random type for initialization
				initialization (dict): Keyword arguments for initialization
				constants (dict[dict[str,object]]): constant indices and values of parameters, along axis, of the form {'axis':{'index':value}}
				args (iterable): Additional arguments for parameter
				system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
				model (object): Model with additional attributes for initialization
				kwargs (dict): Additional system keyword arguments
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		setter(kwargs,dict(parameters=parameters,system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)
		super().__init__(**kwargs)

		self.__setup__()

		return

	def __setup__(self):
		'''
		Setup class
		'''

		data = [group for parameter in self.parameters for group in self.parameters[parameter].group if self.parameters[parameter].variable]

		data = {group:Dict(indices={},slices={},local={},group={},parameters={}) for group in data}


		for group in data:
			
			parameters = {parameter:i for i,parameter in enumerate(self.parameters) if (self.parameters[parameter].variable) and (self.parameters[parameter].string in group)}
			size = max((data[group].indices[parameter] for group in data for parameter in data[group].indices),default=-1)+1
			local = any(self.parameters[parameter].local.get(group) for parameter in parameters)

			shape = {parameter: [*(self.parameters[parameter].shape[:max(0,self.parameters[parameter].ndim-(len(self.parameters[parameter].attributes) if self.parameters[parameter].attributes is not None else 0))] if self.parameters[parameter].data is not None else ()),
				 *((attr if isinstance(attr,int) else getattr(self.parameters[parameter],attr) for attr in self.parameters[parameter].attributes) if self.parameters[parameter].attributes is not None else ()),
				] if self.parameters[parameter].shape is not None or self.parameters[parameter].attributes is not None else None
				for i,parameter in enumerate(parameters)}

			print('shape',shape)

			init = {parameter: initialize(**{**self.parameters[parameter],**dict(data=self.parameters[parameter].data,shape=shape[parameter],dtype=self.parameters[parameter].dtype)})
				for i,parameter in enumerate(parameters)}

			print('init',{parameter:init[parameter].shape for parameter in init})

			data[group].indices = {parameter:parameters[parameter] for i,parameter in enumerate(parameters)}
			data[group].slices = {parameter:i for i,parameter in enumerate(parameters)}
			data[group].local = {parameter:local for i,parameter in enumerate(parameters)}
			data[group].group = {parameter:group for i,parameter in enumerate(parameters)}
			data[group].parameters = [init[parameter] for parameter in parameters]
			
			if local:
				data[group].indices = {parameter: size+i for i,parameter in enumerate(parameters)}
				data[group].slices = {parameter:i for i,parameter in enumerate(parameters)}
				data[group].local = {parameter:local for i,parameter in enumerate(parameters)}
				data[group].group = {parameter:group for i,parameter in enumerate(parameters)}
				data[group].parameters = array(data[group].parameters)
			else:
				data[group].indices = {parameter: size for i,parameter in enumerate(parameters)}
				data[group].slices = {parameter:slice(None) for i,parameter in enumerate(parameters)}
				data[group].local = {parameter:local for i,parameter in enumerate(parameters)}
				data[group].group = {parameter:group for i,parameter in enumerate(parameters)}
				data[group].parameters = sum(data[group].parameters)/len(data[group].parameters)

		data = {parameter: Dict(
				indices=data[group].indices[parameter],
				slices=data[group].slices[parameter],
				local=data[group].local[parameter],
				group=data[group].group[parameter],
				parameters=data[group].parameters[data[group].slices[parameter]]
				) 
				for group in data
				for parameter in data[group].indices}
		parameters = {data[parameter].indices: data[parameter].parameters for group in {self.parameters[parameter].group for parameter in self.parameters} for parameter in data if self.parameters[parameter].group == group}

		print('inds',{parameter:data[parameter].indices for parameter in data})
		print('final',{parameter:parameters[parameter].shape for parameter in parameters})
		parameters = array([parameters[indices] for indices in parameters])
		
		shape = parameters.shape[::-1]
		size = parameters.size
		ndim = parameters.ndim
		dtype = parameters.dtype

		parameters = parameters.T.ravel()

		indices = {parameter:data[parameter].indices for parameter in data}

		self.data = data
		self.indices = indices
		self.parameters = parameters
		self.shape = shape
		self.size = size
		self.ndim = ndim
		self.dtype = dtype

		return

	def __call__(self,parameters=None,*args,**kwargs):
		'''
		Class parameters
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			parameters (array): parameters
		'''
		if parameters is None:
			return self.parameters.reshape(self.shape)
		else:
			return parameters.T.ravel()

	def __iter__(self):
		return self.__iterdata__()

	def __getitem__(self,key):
		return self.__getdata__(key)

	def __setitem__(self,key,value):
		super().__setitem__(key,value)
		self.__setdata__(key,value)
		return

	def __len__(self):
		return self.data.__len__()

	def __iterdata__(self):
		return self.data.__iter__()

	def __getdata__(self,key):
		return self.data.get(key)

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