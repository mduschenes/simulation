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

from src.utils import jit,vfunc,copy,switch,array,arange,empty,ones,zeros,bound,gradient_bound
from src.utils import concatenate,addition,prod
from src.utils import random,initialize,seeder,slicing,reshape,datatype,to_index,to_position
from src.utils import pi,itg,arrays,scalars,iterables,integers,floats,delim,separ,cos,sin,exp

from src.iterables import indexer,inserter,setter,getter

from src.system import System,Dict
from src.io import load,dump,join,split


class Parameter(System):

	defaults = dict(
		data=None,string=None,parameters=None,
		shape=None,size=None,ndim=None,dtype=None,			
		variable=None,constant=None,local=None,method=None,group=None,
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
			variable (bool): parameter is variable
			constant (bool): parameter is constant
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
			return self.func(parameters,*args,**kwargs)
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
			return self.gradient(parameters)
		else:
			return 0

	def init(self,data=None,parameters=None,indices=None,variable=None,constant=None,**kwargs):
		'''
		Initialize class data
		Args:
			data (array): Data of class, if None, shape must be not None to initialize data
			parameters (array): Parameters of class
			indices (array): Indices of parameters of class
			variable (bool): parameter is variable
			constant (bool): parameter is constant			
			kwargs (dict): Additional class keyword arguments
		'''

		# Set data
		self.data = data if data is not None else self.data if self.data is not None else self.parameters if self.parameters is not None else None
		self.indices = indices if indices is not None else self.indices
		self.variable = variable if variable is not None else self.variable
		self.constant = constant if constant is not None else self.constant

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

		
		for kwarg in kwargs:
			if hasattr(self,kwarg) and kwargs[kwarg] is not None:
				setattr(self,kwarg,kwargs[kwarg])		

		self.data = self.generate(**self,init=True)

		self.shape = getattr(self.data,'shape',self.shape) if self.data is not None else self.shape
		self.size = getattr(self.data,'size',self.size) if self.data is not None else self.size
		self.ndim = getattr(self.data,'ndim',self.ndim) if self.data is not None else self.ndim
		self.dtype = getattr(self.data,'dtype',self.dtype) if self.data is not None else self.dtype

		self.setup()

		return

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
		self.kwargs = Dict(self.kwargs if self.kwargs is not None else {})

		self.kwargs.update(dict(
			indices = self.indices,
			parameters = prod((self.parameters[i] for i in self.parameters if self.parameters[i] is not None)) if isinstance(self.parameters,dict) else self.parameters,
			shape = self.shape if self.shape is not None else (),
			random = self.random if self.random is not None else 'uniform',
			seed = seeder(self.seed) if self.seed is not None else seeder(self.seed),
			dtype = self.dtype if self.dtype is not None else None,
			))			

		if self.variable:
			if self.indices is None and self.parameters is None:
				
				if self.constant is None or self.constant:
					def func(parameters,*args,**kwargs):
						return parameters
					
					def gradient(parameters,*args,**kwargs):
						return 1
				else:
					def func(parameters,*args,**kwargs):
						return self.generate(*args,**{**self.kwargs,**kwargs})
					
					def gradient(parameters,*args,**kwargs):
						return 1

			elif self.indices is None and self.parameters is not None:
				
				if self.constant is None or self.constant:
					def func(parameters,*args,**kwargs):
						return self.kwargs.parameters*parameters
					
					def gradient(parameters,*args,**kwargs):
						return self.kwargs.parameters
				else:
					def func(parameters,*args,**kwargs):
						return self.kwargs.parameters*self.generate(*args,**{**self.kwargs,**kwargs})
					
					def gradient(parameters,*args,**kwargs):
						return self.kwargs.parameters

			elif self.indices is not None and self.parameters is None:
				
				if self.constant is None or self.constant:
					def func(parameters,*args,**kwargs):
						return parameters[self.indices]
					
					def gradient(parameters,*args,**kwargs):
						return 1
				else:
					def func(parameters,*args,**kwargs):
						return self.generate(*args,**{**self.kwargs,**kwargs})
					
					def gradient(parameters,*args,**kwargs):
						return 1						

			elif self.indices is not None and self.parameters is not None:

				if self.constant is None or self.constant:
					def func(parameters,*args,**kwargs):
						return self.kwargs.parameters*parameters[self.indices]
					
					def gradient(parameters,*args,**kwargs):
						return self.kwargs.parameters
				else:
					def func(parameters,*args,**kwargs):
						return self.kwargs.parameters*self.generate(*args,**{**self.kwargs,**kwargs})
					
					def gradient(parameters,*args,**kwargs):
						return self.kwargs.parameters
			
			else:

				if self.constant is None or self.constant:
					def func(parameters,*args,**kwargs):
						return parameters
					
					def gradient(parameters,*args,**kwargs):
						return 1
				else:
					def func(parameters,*args,**kwargs):
						return self.generate(*args,**{**self.kwargs,**kwargs})
					
					def gradient(parameters,*args,**kwargs):
						return 1						
		else:

			if self.indices is None and self.parameters is None:
				
				if self.constant is None or self.constant:
					def func(parameters,*args,**kwargs):
						return self.data
					
					def gradient(parameters,*args,**kwargs):
						return 0

				else:
					def func(parameters,*args,**kwargs):
						return self.generate(*args,**{**self.kwargs,**kwargs})

					def gradient(parameters,*args,**kwargs):
						return 0

			elif self.indices is None and self.parameters is not None:
				
				if self.constant is None or self.constant:
					def func(parameters,*args,**kwargs):
						return self.kwargs.parameters*self.data
					
					def gradient(parameters,*args,**kwargs):
						return 0
				else:
					def func(parameters,*args,**kwargs):
						return self.kwargs.parameters*self.generate(*args,**{**self.kwargs,**kwargs})
					
					def gradient(parameters,*args,**kwargs):
						return 0						

			elif self.indices is not None and self.parameters is None:
				
				if self.constant is None or self.constant:
					def func(parameters,*args,**kwargs):
						return self.data[self.indices]
					
					def gradient(parameters,*args,**kwargs):
						return 0
				else:
					def func(parameters,*args,**kwargs):
						return self.generate(*args,**{**self.kwargs,**kwargs})[self.indices]
					
					def gradient(parameters,*args,**kwargs):
						return 0						

			elif self.indices is not None and self.parameters is not None:
				
				if self.constant is None or self.constant:
					def func(parameters,*args,**kwargs):
						return self.kwargs.parameters*self.data[self.indices]
					
					def gradient(parameters,*args,**kwargs):
						return 0
				else:
					def func(parameters,*args,**kwargs):
						return self.kwargs.parameters*self.generate(*args,**{**self.kwargs,**kwargs})[self.indices]
					
					def gradient(parameters,*args,**kwargs):
						return 0			

			else:
			
				if self.constant is None or self.constant:
					def func(parameters,*args,**kwargs):
						return self.data
					
					def gradient(parameters,*args,**kwargs):
						return 0
				else:
					def func(parameters,*args,**kwargs):
						return self.generate(*args,**{**self.kwargs,**kwargs})
					
					def gradient(parameters,*args,**kwargs):
						return 0				

		self.func = func
		self.gradient = gradient

		return

	def generate(self,init=None,**kwargs):
		'''
		Generate class data
		Args:
			init (bool): Initialize data
			kwargs (dict): Keyword arguments for class data
		Returns:
			data (object): Class data
		'''
		if init:
			data = initialize(**kwargs)
		else:
			data = random(**kwargs)

		return data

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
	constant = None
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
				variable (bool): parameter is variable
				constant (bool): parameter is constant
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
		# if not isinstance(self.parameters,dict):
			# self.parameters = {str(self.parameters):self.parameters}

		# cls = Parameter
		# for parameter in self.parameters:
		# 	if not isinstance(self.parameters[parameter],cls):
		# 		self.parameters[parameter] = cls(**self.parameters[parameter])


		if self.parameters is None:
			return

		data = {parameter: {} 
			for parameter in self.parameters 
			if self.parameters[parameter].variable
			}

		for i,parameter in enumerate(data):

			data[parameter] = Dict(data=None,shape=None,local=None,seed=None,indices=None)

			data[parameter].seed = self.parameters[parameter].seed
			data[parameter].local = True
			data[parameter].indices = i
			data[parameter].variable = self.parameters[parameter].variable
			data[parameter].axis = list(sorted(list(set(self.parameters[parameter].axis)),key=lambda i: self.parameters[parameter].axis.index(i)))

			data[parameter].data = None if self.parameters[parameter].random is not None else self.parameters[parameter].data
			data[parameter].shape = (
				*(self.parameters[parameter].shape[:max(0,self.parameters[parameter].ndim-(len(self.parameters[parameter].axis) if self.parameters[parameter].axis is not None else 0))] if self.parameters[parameter].data is not None else ()),
				*((attr if isinstance(attr,int) else getattr(self.parameters[parameter],attr) for attr in self.parameters[parameter].axis) if self.parameters[parameter].axis is not None else ()),
			)

			kwargs = {
				**self.parameters[parameter],
				**data[parameter],
				}

			data[parameter].data = self.parameters[parameter].generate(**kwargs,init=True)

		data = {parameter: data[parameter] for parameter in data}

		indices = {parameter:data[parameter].indices for parameter in data} if data is not None else None
		variable = all(data[parameter].variable for parameter in data) if data is not None and len(data) else None
		parameters = array([data[parameter].data for parameter in data]) if data is not None else None

		parameters = (reshape(parameters,shape=len(data)) if parameters.ndim == 1 else parameters.transpose()) if parameters is not None else None

		shape = parameters.shape if parameters is not None else None
		size = parameters.size if parameters is not None else None
		ndim = parameters.ndim if parameters is not None else None
		dtype = parameters.dtype if parameters is not None else None

		indices = {i: indices[i] for i in indices} if parameters is not None else None
		variable = variable if parameters is not None else None
		parameters = parameters.ravel() if parameters is not None else None

		if data is not None:
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
		self.variable = variable
		self.parameters = parameters
		self.shape = shape
		self.size = size
		self.ndim = ndim
		self.dtype = dtype

		return


	def init(self,data=None,parameters=None,indices=None,variable=None,constant=None,**kwargs):
		'''
		Initialize class data
		Args:
			data (array): Data of class, if None, shape must be not None to initialize data
			parameters (array): Parameters of class
			indices (array): Indices of parameters of class
			variable (bool): parameter is variable
			constant (bool): parameter is constant			
			kwargs (dict): Additional class keyword arguments
		'''
	
		data = self.data if data is not None else data
		parameters = self.parameters if parameters is not None else parameters
		indices = self.indices if indices is not None else indices
		variable = self.variable if variable is not None else variable
		constant = self.constant if constant is not None else constant

		self.data = data
		self.parameters = parameters
		self.indices = indices
		self.variable = variable
		self.constant = constant

		for kwarg in kwargs:
			if hasattr(self,kwarg) and kwargs[kwarg] is not None:
				setattr(self,kwarg,kwargs[kwarg])		

		return

	def generate(self,init=None,**kwargs):
		'''
		Generate class data
		Args:
			init (bool): Initialize data
			kwargs (dict): Keyword arguments for class data
		Returns:
			data (object): Class data
		'''
		if init:
			data = initialize(**kwargs)
		else:
			data = random(**kwargs)

		return data

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
			return reshape(parameters,shape=self.shape)

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
			return 1
		else:
			return 1

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