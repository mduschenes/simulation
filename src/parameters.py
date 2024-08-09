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

from src.utils import jit,vfunc,copy,switch,array,arange,empty,bound,gradient_bound
from src.utils import concatenate,addition,prod
from src.utils import initialize,spawn,slicing,datatype,to_index,to_position
from src.utils import pi,itg,arrays,scalars,iterables,integers,floats,delim,separ,cos,sin,exp

from src.iterables import indexer,inserter,setter,getter

from src.system import System,Dict
from src.io import load,dump,join,split


class Parameter(System):

	defaults = dict(
		data=None,string=None,parameters=None,
		shape=None,size=None,ndim=None,dtype=None,			
		variable=None,local=None,method=None,group=None,
		seed=None,random=None,bounds=None,axis=None,
		indices=None,func=None,gradient=None,constraint=None,wrapper=None,
		args=None,kwargs=None
		)

	def __init__(self,data,*args,system=None,**kwargs):
		'''
		Initialize data of parameter
		Args:
			data (iterable): data of parameter, if None, shape must be not None to initialize data
			string (str): Name of parameter
			parameters (iterable): parameters of parameter
			shape (iterable[int]): shape of parameters
			size (int): size of parameters
			ndim (int): ndim of parameters
			dtype (datatype): datatype of parameters
			variable (bool): parameter is variable or constant
			local (bool,dict[iterable,bool]): locality of parameter
			method (str): method of parameter, allowed strings in ['unconstrained','constrained','bounded','time']
			group (iterable[str],iterable[iterable[str]]): iterable of groups associated with parameter grouping			
			seed (int, key): Random seed for initialization
			random (str,dict): Random type for initialization
			bounds (iterable[object]): Bounds of parameters
			axis (str,int,iterable[str,int]): Attributes for additional dimensions of parameter
			indices (int,iterable[int]): Indices of global parameters for parameter
			func (callable): Function to wrap parameters with signature func(parameters)
			constraint (callable): Function to constrain parameters with signature constraint(parameters)
			wrapper (callable): Function to wrap parameters with parameters and indices, with signature wrapper(parameters)			
			args (iterable): Additional arguments for parameter
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		setter(kwargs,dict(data=data,system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(*args,**kwargs)

		self.setup()

		self.init()

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
			return self.data

	def grad(self,parameters=None,*args,**kwargs):
		'''
		Class gradient of parameters
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			parameters (array): parameters
		'''
	
		if parameters is not None:
			return self.gradient(*args,parameters=parameters,**kwargs)
		else:
			return 0

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


	def setup(self):
		'''
		Setup class attributes
		'''
	
		# Get data
		self.dtype = datatype(self.dtype)		
		self.shape = (self.shape,) if isinstance(self.shape,integers) else self.shape if self.shape is not None else getattr(self.data,'shape',None) if self.data is not None else None
		self.data = self.data if isinstance(self.data,str) else array(self.data,dtype=self.dtype) if self.data is not None else empty(self.shape,dtype=self.dtype) if self.shape is not None else None

		self.shape = self.shape if self.shape is not None else getattr(self.data,'shape',self.shape) if self.data is not None else None
		self.size = self.size if self.size is not None else getattr(self.data,'size',self.size) if self.data is not None else prod(self.shape) if self.shape is not None else None
		self.ndim = self.ndim if self.ndim is not None else getattr(self.data,'ndim',self.ndim) if self.data is not None else len(self.shape) if self.shape is not None else None
		self.dtype = self.dtype if self.dtype is not None else getattr(self.data,'dtype',self.dtype) if self.data is not None else None

		self.string = self.string if self.string is not None else None
		self.variable = self.variable if self.variable is not None else None
		self.group = (*(group for group in self.group),) if self.group is not None and not isinstance(self.group,str) else (self.group,) if self.group is not None else (self.string,)
		self.local = self.local if self.local is not None else True
		self.method = self.method if self.method is not None else None
		self.bounds = self.bounds if self.bounds is not None else [-1,1]
		self.axis = [attr for attr in self.axis if isinstance(attr,int) or getattr(self,attr,None) is not None] if isinstance(self.axis,iterables) else [*((self.axis,) if (isinstance(self.axis,int) or getattr(self,self.axis,None) is not None) else ()),] if self.axis is not None else None

		self.args = self.args if self.args is not None else ()
		self.kwargs = self.kwargs if self.kwargs is not None else {}

		# Set functions
		kwargs = {**self,**dict(data=self.data,shape=self.shape,dtype=self.dtype)}

		defaults = {
			'constants':{},
			'lambda':0,
			'coefficients':[1,2*pi],
			'shift':[0,-pi/2],
			'sigmoid':1,
			'default':0
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
					indices = [int(i) for i in constants]
					values = [array(constants[i],dtype=self.dtype) for i in constants]
					axis = int(axis)
					self.kwargs[attr][axis] = {i:j for i,j in zip(indices,values)}

		defaults = {
			'string':str(self),
			'parameters': prod((self.parameters[i] for i in self.parameters if self.parameters[i] is not None)) if isinstance(self.parameters,dict) else self.parameters
			}
		self.kwargs.update(defaults)

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

		if self.data is None:
			def wrapper(parameters):
				return parameters
			def gradient_wrapper(parameters):
				return 1
		elif self.indices is not None and self.parameters is not None:
			def wrapper(parameters):
				return self.kwargs['parameters']*parameters[self.indices]
			def gradient_wrapper(parameters):
				return self.kwargs['parameters']
		elif self.indices is not None and self.parameters is None:
			def wrapper(parameters):
				return parameters
			def gradient_wrapper(parameters):
				return 1				
		elif self.indices is None and self.parameters is not None:
			def wrapper(parameters):
				return self.kwargs['parameters']*parameters
			def gradient_wrapper(parameters):
				return self.kwargs['parameters']				
		elif self.indices is None and self.parameters is None:
			def wrapper(parameters):
				return parameters
			def gradient_wrapper(parameters):
				return 1
		
		self.wrapper = jit(wrapper)
		self.gradient_wrapper = jit(gradient_wrapper)


		if self.variable:

			if self.method in ['bounded'] and all(self.kwargs.get(attr) is not None for attr in ['sigmoid']):
		
				def func(parameters,*args,**kwargs):
					return self.wrapper(bound(parameters,scale=self.kwargs['sigmoid']))

				def gradient(parameters,*args,**kwargs):
					return self.wrapper(gradient_bound(parameters,scale=self.kwargs['sigmoid']))

			elif self.method in ['bounded']:

				def func(parameters,*args,**kwargs):
					return self.wrapper(bound(parameters))

				def gradient(parameters,*args,**kwargs):
					return self.wrapper(gradient_bound(parameters))

			elif self.method in ['constrained'] and all(self.kwargs.get(attr) is not None for attr in ['sigmoid']):
		
				def func(parameters,*args,**kwargs):
					return self.wrapper(bound(parameters,scale=self.kwargs['sigmoid']))

				def gradient(parameters,*args,**kwargs):
					return self.wrapper(gradient_bound(parameters,scale=self.kwargs['sigmoid']))

			elif self.method in ['constrained']:					
		
				def func(parameters,*args,**kwargs):
					return self.wrapper(bound(parameters))

				def gradient(parameters,*args,**kwargs):
					return self.wrapper(gradient_bound(parameters))

			elif self.method in ['unconstrained']:					
				
				def func(parameters,*args,**kwargs):
					return self.wrapper(parameters)

				def gradient(parameters,*args,**kwargs):
					return self.gradient_wrapper(parameters)

			elif self.method in ['time'] and all(self.kwargs.get(attr) is not None for attr in ['tau','scale']):
				
				def func(parameters,*args,**kwargs):
					return (1 - exp(-self.kwargs['tau']/self.kwargs['scale']))/2

				def gradient(parameters,*args,**kwargs):
					return 0

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
						return self.wrapper(parameters)
				else:
					func = partial(func,self=self)
	
				def gradient(parameters,*args,**kwargs):
					return self.gradient_wrapper(parameters)

			if self.method in ['constrained'] and all(self.kwargs.get(attr) is not None for attr in ['lambda','constants']):

				def constraint(parameters,*args,**kwargs):
					return self.kwargs['lambda']*sum(
						((parameters[i]- self.kwargs['constants'][axis][i])**2).sum()/parameters.shape[-1]
						for axis in self.kwargs['constants']
						for i in self.kwargs['constants'][axis]
						) if parameters.ndim else self.kwargs['default']

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
					return self.wrapper(self.data)

				def gradient(parameters,*args,**kwargs):
					return self.gradient_wrapper(self.data)

			else:
				def func(parameters,*args,**kwargs):
					return self.data

				def gradient(parameters,*args,**kwargs):
					return 0

			def constraint(parameters,*args,**kwargs):
				return self.kwargs['default']


		parameters = self.data if self.data is not None else None

		func = jit(func,parameters=parameters)
		gradient = jit(gradient,parameters=parameters)
		constraint = jit(constraint,parameters=parameters)

		self.func = func
		self.gradient = gradient
		self.constraint = constraint

		return

	def init(self,data=None,parameters=None,indices=None,variable=None):
		'''
		Initialize class data
		Args:
			data (array): Data of class, if None, shape must be not None to initialize data
			parameters (array): Parameters of class
			indices (array): Indices of parameters of class
			variable (bool): Parameter is variable or constant
		'''

		# Set data
		self.data = data if data is not None else self.data
		self.indices = indices if indices is not None else self.indices
		self.variable = variable if variable is not None else self.variable

		if self.parameters is None:
			if parameters is None:
				self.parameters = parameters
			elif not isinstance(parameters,dict):
				self.parameters = Dict(dict(parameters=parameters))
			else:
				self.parameters = Dict({i:parameters[i] for i in parameters})
		elif not isinstance(self.parameters,dict):
			if parameters is None:
				self.parameters = Dict(dict(parameters=self.parameters))
			elif not isinstance(parameters,dict):
				self.parameters = Dict(dict(parameters=parameters))
			elif isinstance(parameters,dict):
				self.parameters = Dict({**dict(parameters=self.parameters),**parameters})
		elif isinstance(self.parameters,dict):
			if parameters is None:
				self.parameters = Dict({**self.parameters})
			elif not isinstance(parameters,dict):
				self.parameters = Dict({**self.parameters,**dict(parameters=parameters)})
			else:
				self.parameters = Dict({**self.parameters,**parameters})

		self.data = initialize(**self)

		self.shape = getattr(self.data,'shape',self.shape) if self.data is not None else self.shape
		self.size = getattr(self.data,'size',self.size) if self.data is not None else self.size
		self.ndim = getattr(self.data,'ndim',self.ndim) if self.data is not None else self.ndim
		self.dtype = getattr(self.data,'dtype',self.dtype) if self.data is not None else self.dtype

		self.setup()

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
	variable = None
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

		Setup parameters such that calling the Parameters(parameters) class with input parameters reshapes parameters into shape 
		
		Args:
			parameters(dict): Dict of Parameters instances, with class attributes
				data (iterable): data of parameter, if None, shape must be not None to initialize data
				string (str): Name of parameter
				parameters (iterable): parameters of parameter
				shape (iterable[int]): shape of parameters
				size (int): size of parameters
				ndim (int): ndim of parameters
				dtype (datatype): datatype of parameters
				variable (bool): parameter is variable or constant
				local (bool,dict[iterable,bool]): locality of parameter
				method (str): method of parameter, allowed strings in ['unconstrained','constrained','bounded','time']
				group (iterable[str],iterable[iterable[str]]): iterable of groups associated with parameter grouping			
				seed (int, key): Random seed for initialization
				random (str,dict): Random type for initialization
				bounds (iterable[object]): Bounds of parameters
				axis (str,int,iterable[str,int]): Attributes for additional dimensions of parameter
				indices (int,iterable[int]): Indices of global parameters for parameter
				func (callable): Function to wrap parameters with signature func(parameters)
				constraint (callable): Function to constrain parameters with signature constraint(parameters)
				args (iterable): Additional arguments for parameter
				system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
				kwargs (dict): Additional system keyword arguments
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		setter(kwargs,dict(parameters=parameters,system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)
		super().__init__(**kwargs)

		self.setup()

		return

	def setup(self):
		'''
		Setup class
		'''

		data = {self.parameters[parameter].group: {}
			for parameter in self.parameters 
			if (self.parameters[parameter].variable)
			}

		for group in data:
			
			parameters = [parameter 
				for i,parameter in enumerate(self.parameters) 
				if ((self.parameters[parameter].variable) and 
				    (self.parameters[parameter].string in group))
				]
			index = max((data[group][parameter].indices for group in data for parameter in data[group]),default=-1)+1
			local = any(self.parameters[parameter].local for parameter in parameters)

			for i,parameter in enumerate(parameters):

				data[group][parameter] = Dict(data=None,shape=None,local=None,seed=None,indices=None)

				data[group][parameter].seed = self.parameters[parameter].seed #spawn(self.parameters[parameter].seed,size=len(parameters))[i]
				data[group][parameter].local = local
				data[group][parameter].indices = index+i if local else index
				data[group][parameter].axis = list(sorted(list(set(self.parameters[parameter].axis)),key=lambda i: self.parameters[parameter].axis.index(i)))

				data[group][parameter].data = None if self.parameters[parameter].random is not None else self.parameters[parameter].data
				data[group][parameter].shape = (
						*(self.parameters[parameter].shape[:max(0,self.parameters[parameter].ndim-(len(self.parameters[parameter].axis) if self.parameters[parameter].axis is not None else 0))] if self.parameters[parameter].data is not None else ()),
						*((attr if isinstance(attr,int) else getattr(self.parameters[parameter],attr) for attr in self.parameters[parameter].axis) if self.parameters[parameter].axis is not None else ()),
					)

				kwargs = {
					**self.parameters[parameter],
					**data[group][parameter],
					}

				data[group][parameter].data = initialize(**kwargs)

		data = {parameter: data[group][parameter] for group in data for parameter in data[group]}

		indices = {parameter:data[parameter].indices for parameter in data}
		parameters = {parameter: data[parameter].data for parameter in data}

		indices = {i: [parameter for parameter in indices if indices[parameter]==i] for i in sorted(set(indices[parameter] for parameter in indices))}		
		parameters = array([parameters[indices[i][0]] for i in indices]).transpose()

		shape = parameters.shape
		size = parameters.size
		ndim = parameters.ndim
		dtype = parameters.dtype

		indices = {i: indices[i] for i in indices}
		parameters = parameters.ravel()


		for parameter in data:
			kwargs = dict(
				indices=data[parameter].indices,
				)
			self.parameters[parameter].init(**kwargs)

		data = {str(self.parameters[parameter]):self.parameters[parameter] for parameter in self.parameters}

		for attr in data:
			setattr(self,attr,data[attr])

		self.data = data
		self.indices = indices
		self.parameters = parameters
		self.shape = shape
		self.size = size
		self.ndim = ndim
		self.dtype = dtype

		return


	def init(self,data=None,parameters=None,indices=None,variable=None):
		'''
		Initialize class data
		Args:
			data (array): Data of class, if None, shape must be not None to initialize data
			parameters (array): Parameters of class
			indices (array): Indices of parameters of class
			variable (bool): Parameter is variable or constant
		'''
	
		data = self.data if data is not None else data
		parameters = self.parameters if parameters is not None else parameters
		indices = self.indices if indices is not None else indices
		variable = self.variable if variable is not None else variable

		self.data = data
		self.parameters = parameters
		self.indices = indices
		self.variable = variable

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
			return self.parameters
		else:
			return parameters.reshape(self.shape)

	def grad(self,parameters=None,*args,**kwargs):
		'''
		Class gradient of parameters
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			parameters (array): parameters
		'''
		if parameters is None:
			return self.parameters
		else:
			return parameters.reshape(self.shape)

	def constraints(self,parameters=None,*args,**kwargs):
		'''
		Class constraints
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			parameters (array): parameters
		'''
		parameters = self(parameters)
		return sum(self[parameter].constraints(parameters) for parameter in self)

	def __iter__(self):
		return self.data.__iter__()

	def __getitem__(self,key):
		return self.data.get(key)

	def __setitem__(self,key,value):
		super().__setitem__(key,value)
		if key in self.data:
			self.data[key] = value
		return

	def __len__(self):
		return self.data.__len__()

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