#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools
from copy import deepcopy
from functools import partial
import time
from time import time as timer

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


def _variables(hyperparameters,parameter,group):
	'''
	Get variables from parameters
	Args:
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for variables
		group (str): Parameter group for variables
	Returns:
		func (callable): function with signature func(parameters), parameters (array): Array of parameters to compute constraints
	'''

	kwargs = hyperparameters[parameter]['kwargs']
	method = hyperparameters[parameter]['method']
	index = hyperparameters[parameter]['group'].index(group)

	if method in ['constrained'] and parameter in ['xy']:
		scale = [hyperparameters[parameter]['scale'],2*pi]
	else:
		scale = [hyperparameters[parameter]['scale']]*len(hyperparameters[parameter]['group'])

	if parameter in ['zz'] and group in [('zz',)]:
		scale[index] /= 1 #4*kwargs['min']*kwargs['tau']

	# Parameters shape (G,K/G,M) -> (K/G,M)
	if method in ['constrained']:
		if parameter in ['xy'] and group in [('x',)]:
			def func(parameters):
				return scale[0]*parameters[0]*cos(scale[1]*parameters[1])
		
		elif parameter in ['xy'] and group in [('y',)]:
			def func(parameters):
				return scale[0]*parameters[0]*sin(scale[1]*parameters[1])
		
		elif parameter in ['z'] and group in [('z',)]:
			def func(parameters):
				return scale[index]*parameters[index]
		
		elif parameter in ['zz'] and group in [('zz',)]:
			def func(parameters):
				return scale[index]*parameters[index]

	elif method in ['unconstrained']:
		def func(parameters):
			return scale[index]*parameters[index]	

	else:
		def func(parameters):
			return scale[index]*parameters[index]		

	return func


def _features(hyperparameters,parameter,group):
	'''
	Get features from parameters
	Args:
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for features
		group (str): Parameter group for features
	Returns:
		func (callable): function with signature func(parameters), parameters (array): Array of parameters to compute constraints
	'''

	kwargs = hyperparameters[parameter]['kwargs']
	method = hyperparameters[parameter]['method']
	size = len(hyperparameters[parameter]['group'])

	if method in ['constrained','bound']:
		wrapper = bound
	elif method in ['unconstrained']:
		wrapper = nullbound
	else:
		wrapper = nullbound

	# Parameters shape (K,M) -> (G,K/G,M)
	def func(parameters):
		shape = (size,parameters.shape[0]//size,*parameters.shape[1:])
		return wrapper(parameters,kwargs).reshape(shape)
	
	return func


def _parameters(hyperparameters,parameter,group):
	'''
	Get parameters from parameters
	Args:
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for parameters
		group (str): Parameter group for parameters
	Returns:
		func (callable): function with signature func(parameters), parameters (array): Array of parameters to compute constraints
	'''
	
	def func(parameters):
		return parameters

	return func


def _constraints(hyperparameters,parameter,group):
	'''
	Get constraints from parameters
	Args:
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for constraints
		group (str): Parameter group for constraints
	Returns:
		func (callable): function with signature func(parameters), parameters (array): Array of parameters to compute constraints
	'''
	kwargs = hyperparameters[parameter]['kwargs']
	method = hyperparameters[parameter]['method']
	scale = hyperparameters[parameter]['kwargs'].get('lambda',1)
	constants = hyperparameters[parameter]['constants']['features'][-1]
	if method in ['constrained']:
		if parameter in ['xy'] and group in [('x',),('y',)]:
			def func(parameters):
				return scale[0]*((parameters[...,constants['slice']] - constants['value'])**2).sum()

		elif parameter in ['z'] and group in [('z',)]:
			def func(parameters):
				return 0.

		elif parameter in ['zz'] and group in [('zz',)]:
			def func(parameters):
				return 0.			

	elif method in ['unconstrained']:
		def func(parameters):
			return 0.
	else:
		def func(parameters):
			return 0.

	return func


def _gradient_constraints(hyperparameters,parameter,group):
	'''
	Get gradients of constraints from parameters
	Args:
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for constraints
		group (str): Parameter group for constraints
	Returns:
		func (callable): function with signature func(parameters), parameters (array): Array of parameters to compute constraints
	'''

	#TODO (finish analytic derivatives for variables functions as a matrix of (k,l) shape for k output parameters and l parameters)
	# ie) k = m*r for r = 2N, and l = m*q for q = 2,2*N input phases and amplitudes

	kwargs = hyperparameters[parameter]['kwargs']
	method = hyperparameters[parameter]['method']
	scale = hyperparameters[parameter]['kwargs'].get('lambda',1)

	def func(parameters):
		shape = parameters.shape
		grad = zeros(shape)
		grad = grad.ravel()
		return grad

	return func


def _gradients(hyperparameters,parameter,group):
	'''
	Get gradient of variables from parameters
	Args:
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for variables
		group (str): Parameter group for variables
	Returns:
		func (callable): function with signature func(parameters), parameters (array): Array of parameters to compute constraints
	'''	

	#TODO (finish analytic derivatives for variables functions as a matrix of (k,l) shape for k output parameters and l parameters)
	# ie) k = m*r for r = 2N, and l = m*q for q = 2,2*N input phases and amplitudes

	method = hyperparameters[parameter]['method']
	scale = [hyperparameters[parameter]['scale'],2*pi] if method in ['constrained'] else [hyperparameters[parameter]['scale']]*len(hyperparameters[parameter]['group'])

	def func(parameters):
		shape = parameters.shape
		grad = zeros(shape)
		return grad

	return func


def setup(hyperparameters,cls=None):
	'''
	Setup hyperparameters
	Args:	
		hyperparameters (dict): Hyperparameters
		cls (dict): Class attributes
	'''

	# Update with checked values
	updates = {
		**{attr: {
			'value': (lambda parameter,hyperparameters,attr=attr: {kwarg: [{prop: array(i.get(prop,[]),dtype={'slice':int,'value':None}[prop]) for prop in ['slice','value']}
				for i in hyperparameters[parameter][attr][kwarg]] 
				for kwarg in hyperparameters[parameter][attr]}),
			'default': (lambda parameter,hyperparameters: {}),
			'conditions': (lambda parameter,hyperparameters: True)				
			} for attr in ['boundaries','constants']
		},
		'group': {
			'value': (lambda parameter,hyperparameters: [tuple(group) for group in hyperparameters[parameter]['group']]),
			'default': (lambda parameter,hyperparameters: []),
			'conditions': (lambda parameter,hyperparameters: True)				
		},
		**{attr: {
			'value': (lambda parameter,hyperparameters,attr=attr: {
				**hyperparameters[parameter][attr],
				**({kwarg: cls[kwarg] for kwarg in cls
					if isinstance(cls[kwarg],scalars) and not isinstance(cls[kwarg],str)
					} if cls is not None else {}),
				**{kwarg: {'min':minimum,'max':maximum}[kwarg](array(hyperparameters[parameter]['parameters'])) if hyperparameters[parameter].get('parameters') 
						  else {'min':minimum,'max':maximum}[kwarg](array(hyperparameters[parameter]['bounds']['parameters']))
					for kwarg in ['min','max']
					}, 
				}),
			'default': (lambda parameter,hyperparameters,attr=attr: {}),
			'conditions': (lambda parameter,hyperparameters,attr=attr: True)						
			} for attr in ['kwargs']
		},		
	}
	for parameter in hyperparameters:
		for attr in updates:
			hyperparameters[parameter][attr] = hyperparameters[parameter].get(attr,updates[attr]['default'](parameter,hyperparameters))
			if updates[attr]['conditions'](parameter,hyperparameters):
				if callable(updates[attr]['value'](parameter,hyperparameters)):
					for group in hyperparameters[parameter]['group']:
							group = tuple(group)
							try:
								hyperparameters[parameter][attr][group] = jit(
									updates[attr]['value'](parameter,hyperparameters)(
										hyperparameters=hyperparameters,parameter=parameter,group=group)
									)
							except:
								hyperparameters[parameter][attr][group] = jit(
									updates[attr]['value'](parameter,hyperparameters)
									)
				else:
					hyperparameters[parameter][attr] = updates[attr]['value'](parameter,hyperparameters)


	return 


class Parameters(Object):
	def __init__(self,data,shape,size=None,ndim=None,dims=None,system=None,**kwargs):
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
			size (int,iterable[int]): Number of data
			ndim (int): Number of dimensions of data
			dims (iterable[int]): Dimensions of N, D-dimensional sites [N,D]
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			cls (dict): Class attributes
			check (callable): Function with signature check(data,group) to data is part of group
			initialize (callable): Function with signature initialize(parameters,shape,hyperparameters,reset=None,dtype=None) to initialize parameter values
			kwargs (dict): Additional system keyword arguments
		'''
		
		# Implicit parameterizations that interact with the data to produce the output are called variables x
		# These variables are Parametersd by the explicit parameters theta such that x = x(theta)
		# Variables have a shape x = S = (s_0,...s_{d-1})
		
		# Each category i of parameter (variable,constant,...) has parameters with shape theta^(i) = T = (T^(i)_0,...,T^(i)_{d^(i)-1})
		# and each of these parameters yield subsets of the variables with indices I^(i) = (I^(i)_0,...,I^(i)_{d^(i)-1})
		# such that the union of all the indices U_i I^(i) across the categories covers all variables.
		
		# Each category i of parameter has parameter key groupings that share values
		# Each parameter key grouping has groups g^(i) that depend on a slices of theta^(i), and has shape T^(i)_g^(i) = (T^(i)_g^(i)_0,...,T^(i)_g^(i)_{d^(i)_g^(i)-1})
		# and yield subsets of the variables with indices I^(i) = (I^(i)_g^(i)_0,...,I^(i)_g^(i)_{d^(i)_g^(i)-1})

		# In between theta and x, there may be intermediate layers of values, such as features phi(theta), such that x = x(phi(theta))
		# Each layer l has its own shape T_l

		# Parameters are described by the dictionary hyperparameters, with different parameter keys, 
		# each with an associated category i,
		# and groups of parameters g^(i) that use a subset of theta^(i)

		# Each set of values is described by keys of categories,parameters,groups,layers:

		# data (dict): Dictionary of parameter data, with nested keys of categories,parameters,groups,layers, and data of
		# 	'ndim': int : Number of dimensions of value corresponding to key
		# 	'locality': iterable[int] : Locality of value for each axis of value corresponding to key
		# 	'size': iterable[int] : Shape of multiplicative value of data shape for each axis of value corresponding to key
		# 	'indices': iterable[iterable[int]]: Indices of data for each axis of value corresponding to key
		# 	'boundaries': iterable[dict[int,float]]: Boundary indices and values for each axis of value corresponding to key
		# 	'constants': iterable[dict[int,float]]: Boundary indices and values for each axis of value corresponding to key
		# 	'shape': dict[str,tuple[int]]: Shape of value for different index types for each axis of value corresponding to key
		# 	'slice': dict[str,tuple[slice]]: Slices of value for different index types for each axis of value corresponding to key
		# 	layer (callable): Callable function with signature func(parameters) for input parameters that yields values for that layer


		# Shape of axes for keys of categories,parameters,groups,layers
		# Shapes are based on indices of data, plus sizes multipliers from hyperparameters
		# Sizes multipliers are either < 0 indicating multiplier of shape, or > 0 indicating fixed value
		# Sizes with dimensions beyond (in front of) data shape have initial sizes assumed to be fixed values based on indices of [0] of fixed size 1 (for example features)
		# (s assumed to be in general 1 for 'variables',)
		# Shape of parameters and features are dependent on size of indices variables per group, which depends on shape of data and locality of parameter
		# It is assumed that all parameters within a category have the same ndim number of dimensions, and shared number of dimensions except along the [0] axis

		# Values have shapes and slices with index types of the form of tuples (<type>,<subtype>,<subtype>)

		# with types of

		# 'put': Shape - Full shape of array being assigned to, Slice: Slice of array being assigned to
		# 'take': Shape - Shape of slice of array being taken, Slice of array being taken

		# with subtypes of

		# 'key': Shape,slice for key
		# 'category': Shape,slice for key, within all keys associated with category
		# 'layer': Shape,slice for key, within all keys associated with layer
		
		# with subsubtypes of 
		
		# 'all': Shape,slice including all values + including boundaries and constants
		# 'variable': Shape,slice including all values + excluding boundaries and constants
		# 'constant': Shape,slice excluding all values + including boundaries and constants

		# For layers not in ['variables'], shapes and slices are shared along all axes amongst all groups for a given parameter, and so have all same shape and slice for these layers
		# For layers in ['variables'],shapes and slices are individual along axes [0] amongst all groups for a given parameter and otherwise are shared
		# Note that all parameter,group within a category or layer should have the same boundary and constant indices, so all 'category' or 'layer' values have same shape,
		# otherwise parameter,group with more boundary/constant indices will have redundant points within the encompassing shape, depending how these boundary constant points are imposed

		# Depending on locality, functions of 'take' indexed values that return to 'put' indexed values 
		# should return either arrays of shape:
		# locality in ['local']  exact shape of the 'put' indexed variables
		# locality in ['global'] broadcastable (size 1) shape of the 'put' sliced variables
		# For example if a function of 'take' sliced values of size (l,k,) and the 'put' sliced values are of size (k,), 
		# then the function should roughly use the l values to return the correct shape (k,)

		# For a given indices,locality,and sizes of category,parameter,group,layer and axis, 
		
		# The shapes and slices for each individual set of values ('take,put','key','all') for these keys are:
		# s = sizes 
		# k = len(indices) (k[axis=0 (parameters,variables), axis=1 (features)] = O(N) for data with datum on each of N sites, k[axis=1] = O(M) for M time steps)
		# The shape of the values will be 
		# shape['features','parameters','variables'][('put','key','all')] = -s*k if s < 0 else s
		# shape['features','parameters','variables'][('take','key','all')] = -s*(k if local in ['local'] else 1) if s < 0 else s

		# slice['variables'][('put','key','all')] = indices if axis == 0 else slice(0,shape[('put','key','all')],1)
		# shape['variables'][('take','key','all')] = slice(0,shape['take_key_all'],1)
		# shape['features','parameters'][('put','key','all')] = slice(0,shape[('put','key','all')],1)
		# shape['features','parameters'][('take','key','all')] = slice(0,shape[('take','key','all')],1)

		# The other ('take,put',<type>) indexes involve summing all shapes corresponding to the keys that are within the type group, 
		# plus subtracting shapes corresponding with boundaries and constants

		setter(kwargs,data,delimiter=delim,func=False)
		super().__init__(data,shape,size=size,ndim=ndim,dims=dims,system=system,**kwargs)

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
			data = self.parameters()

		# data = bound(data.reshape(indexer(self.index,self.data).shape))
		# data = data.reshape(indexer(self.index,self.data).shape)
		inserter(self.index,data.reshape(indexer(self.index,self.data).shape),self.data)
		# data = indexer(self.index,self.data).ravel()

		return data


	def __setup__(self,**kwargs):
		'''
		Setup attribute
		Args:
			kwargs (dict): Additional keyword arguments
		'''

		# Get number of dimensions of data
		ndim = len(self.shape)
		self.ndim = ndim

		# Size of data
		size = None
		self.size = size
		self.length = len(self.size) if self.size is not None else None

		# Get datatype of data
		dtype = datatype(self.dtype)
		self.dtype = dtype

		# Get data
		self.dimensions = None
		self.attributes = {}

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

		# Remove not used parameters of data
		for parameter in list(self):
			if not any(self.check(self.cls.data[i],group)
				for i in range(self.shape[0])
				for group in self.data[parameter].get('group',[])):
				self.data.pop(parameter)
				delattr(self,parameter)

		for parameter in self.data:
			setattr(self,parameter,Object(**self.data[parameter],system=self.system))

		# Get data
		for parameter in self:
			layer = 'parameters'
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

			self[parameter] = getattr(self,parameter)

		self.dimensions = (sum(self.data[parameter]().size for parameter in self.data if self.data[parameter].category in ['variable']),)
		self.attributes = {}
		self.string = ' '.join([str(getattr(self,parameter)) for parameter in self])
		self.index = [parameter for parameter in self.data if self.data[parameter].category in ['variable']]
		self.parameters = indexer(self.index,self)

		return