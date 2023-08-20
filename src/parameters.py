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

from src.utils import jit,vfunc,switch,array,arange,bound
from src.utils import concatenate,addition,prod
from src.utils import initialize,slicing,datatype,to_index,to_position
from src.utils import pi,itg,scalars,arrays,delim,separ,cos,sin,exp

from src.iterables import indexer,inserter,setter,getter

from src.system import System,Dict
from src.io import load,dump,join,split


class Parameter(System):

	defaults = dict(
			string=None,variable=None,method=None,
			locality=None,bounds=None,attributes=None,axis=None,
			parameters=None,
			seed=None,random=None,initialization=None,constants=None,
			indices=None,func=None,constraint=None,
			shape=None,size=None,ndim=None,dtype=None,			
			args=(),kwargs={}
			)

	def __init__(self,data,*args,system=None,**kwargs):
		'''
		Initialize data of parameter
		Args:
			data (iterable): data of parameter
			string (str): Name of parameter
			variable (bool): Parameter is variable or constant
			method (str): method of parameter, allowed strings in ['unconstrained','constrained','bounded','time']
			locality (str,iterable[str],dict[iterable,str]): locality of parameter, allowed strings in ['local','global']
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

		setter(kwargs,dict(data=data,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,self.defaults,delimiter=delim,func=False)

		super().__init__(*args,**kwargs)

		self.__setup__()

		return

	def __call__(self,parameters=None,*args,**kwargs):
		'''
		Class data
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
		self.data = array(self.data,dtype=self.dtype) if self.data is not None else None

		self.shape = self.data.shape if self.data is not None else None
		self.size = self.data.size if self.data is not None else None
		self.ndim = self.data.ndim if self.data is not None else None

		self.string = self.string if self.string is not None else None
		self.variable = self.variable if self.variable is not None else None
		self.method = self.method if self.method is not None else None
		self.locality = self.locality if self.locality is not None else None
		self.attributes = [attr for attr in self.attributes if isinstance(attr,int) or getattr(self,attr,None) is not None] if self.attributes is not None else ()
		self.kwargs = self.kwargs if self.kwargs is not None else {}

		self.dtype = self.data.dtype if self.data is not None else None


		shape = [*(self.shape[:max(0,self.ndim-len(self.attributes))] if self.data is not None else ()),
				 *(attr if isinstance(attr,int) else getattr(self,attr) for attr in self.attributes),
				]


		# Set attributes
		self.shape = shape if shape is not None else None
		self.size = prod(shape) if shape is not None else None
		self.ndim = len(shape) if shape is not None else None


		# Set functions
		kwargs = {**self,**dict(data=self.data,shape=self.shape,dtype=self.dtype)}

		defaults = {
			'constants':self.constants,
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


		if self.indices is not None:
			
			if self.variable:

				if self.method in ['bounded'] and all(self.kwargs.get(attr) is not None for attr in ['sigmoid']):
			
					def func(parameters,*args,**kwargs):
						return self.parameters*bound(parameters[self.indices],scale=self.kwargs['sigmoid'])
						
				elif self.method in ['bounded']:

					def func(parameters,*args,**kwargs):
						return self.parameters*bound(parameters[self.indices])

				elif self.method in ['constrained'] and all(self.kwargs.get(attr) is not None for attr in ['coefficients','shift','sigmoid']):
			
					def func(parameters,*args,**kwargs):
						return self.parameters*bound(parameters[self.indices],scale=self.kwargs['sigmoid'])

				elif self.method in ['constrained']:					
			
					def func(parameters,*args,**kwargs):
						return self.parameters*bound(parameters[self.indices])

				elif self.method in ['unconstrained']:					
					
					def func(parameters,*args,**kwargs):
						return self.parameters*parameters[self.indices]

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
							return self.parameters*parameters[self.indices]
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
			
				def func(parameters,*args,**kwargs):
					return self.parameters*self.data[self.indices]

				
				def constraint(parameters,*args,**kwargs):
					return self.kwargs['default']
		else:

			if self.variable:
				
				if self.method in ['bounded'] and all(self.kwargs.get(attr) is not None for attr in ['sigmoid']):
			
					def func(parameters,*args,**kwargs):
						return self.parameters*bound(parameters,scale=self.kwargs['sigmoid'])
						
				elif self.method in ['bounded']:

					def func(parameters,*args,**kwargs):
						return self.parameters*bound(parameters)

				elif self.method in ['constrained'] and all(self.kwargs.get(attr) is not None for attr in ['coefficients','shift','sigmoid']):
			
					def func(parameters,*args,**kwargs):
						return self.parameters*bound(parameters,scale=self.kwargs['sigmoid'])

				elif self.method in ['constrained']:					
			
					def func(parameters,*args,**kwargs):
						return self.parameters*bound(parameters)


				elif self.method in ['unconstrained']:					
					
					def func(parameters,*args,**kwargs):
						return self.parameters*parameters

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
							return self.parameters*parameters
					else:
						func = partial(func,self=self)
		

				if self.method in ['constrained'] and all(self.kwargs.get(attr) is not None for attr in ['lambda','constants']):
				
					def constraint(parameters,*args,**kwargs):
						return self.kwargs['lambda']*sum(
							((parameters[self.kwargs['constants'][i]['indices']] - 
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
			
				def func(parameters,*args,**kwargs):
					return self.parameters*self.data

				
				def constraint(parameters,*args,**kwargs):
					return self.kwargs['default']							


		parameters = self.data if self.data is not None else 1

		func = jit(func,parameters=parameters)
		constraint = jit(constraint,parameters=parameters)

		self.func = func
		self.constraint = constraint

		return

	def __initialize__(self,data=None,parameters=None,indices=None):
		'''
		Initialize class data with shape
		Args:
			data (array): Data of data
			parameters (array): Parameters of data
			indices (array): Indices of parameters of data
		'''

		# Set data
		self.data = data if data is not None else self.data
		self.parameters = parameters if parameters is not None else self.parameters
		self.indices = indices if indices is not None else self.indices

		if self.shape is not None:

			kwargs = {**self,**dict(data=self.data,shape=self.shape,dtype=self.dtype)}

			self.data = initialize(**kwargs)


		self.__setup__()

		return

	def __str__(self):
		return str(self.string)


class Parameters(System):

	__data__ = {}

	defaults = dict(
			string=None,variable=None,method=None,
			group=None,locality=None,bounds=None,attributes=None,axis=None,
			parameters=None,
			seed=None,random=None,initialization=None,constants=None,
			indices=None,slices=None,sort=None,instance=None,func=None,constraint=None,
			shape=None,size=None,ndim=None,dtype=None,			
			args=(),kwargs={}
			)

	def __init__(self,data=None,model=None,system=None,**kwargs):
		'''
		Initialize data of parameters

		Parameters are split into groups with names [parameter], and can be accessed as 
		attributes Parameters.parameter, or items Parameters[parameter],
		and are contained in Parameters.data = {parameter: Parameter()} as Parameter() instances

		Setup parameters such that calling the Parameters(parameters) class with input parameters 
		i) 		parameters array of size (G*P*D),
					for G groups of P parameters, each of dimension D, 
					for variable parameter groups
					P,D may be group dependent, and depend on Parameter.locality, Parameter.model and Parameter.attributes 
		ii) 	parameters for each group are sliced with parameter slices (slice(P*D)) and reshaped into shape (P,D)
		iii) 	parameters for each group are modified with Parameter() function i.e) bounds, scaling, features
		iv) 	parameters for all groups are concatenated to [parameter_i = Parameters[slices_i]][sort]
					with slices Parameters.slices = [slices], and sorted with sort Parameters.sort = [sort]
		
		Args:
			data (dict): Dictionary of data corresponding to parameters groups, with dictionary values with properties:
				data (iterable): data of parameter
				string (str): Name of parameter
				variable (bool): Parameter is variable or constant
				method (str): method of parameter, allowed strings in ['unconstrained','constrained','bounded','time']
				group (iterable[str],iterable[iterable[str]]): iterable of groups associated with parameter grouping
				locality (str,iterable[str],dict[iterable,str]): locality of parameter across groups, allowed strings in ['local','global']
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
			model (object): Model with additional attributes for initialization
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		data = data if data is not None else None
		__data__ = data if data is not None else {}

		setter(kwargs,dict(data=data,model=model,system=system,__data__=__data__),delimiter=delim,func=False)
		setter(kwargs,data,delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,defaults,delimiter=delim,func=False)
		super().__init__(**kwargs)

		self.__setup__()

		return

	def __call__(self,parameters=None,*args,**kwargs):
		'''
		Class data
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			parameters (array,dict): parameters
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
		indices = {}
		group = []
		slices = []
		sort = []
		instance = []
		for i,parameter in enumerate(self):
			if self[parameter].variable:
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
		instance = {i:parameter for parameter in self for i in self[parameter].instance}

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

		def func(parameters,*args,**kwargs):
			return parameters

		def constraint(parameters,*args,**kwargs):
			return 0

		# Get data
		data = []
		for parameter in self:
			if self[parameter].variable:
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