#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial
import time
from time import time as timer

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp
import jax
import jax.numpy as np
import jax.scipy as sp
import jax.example_libraries.optimizers
jax.config.update('jax_platform_name','cpu',)
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'
# np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.2e',)) for dtype in ['float','float64',np.float64,np.float32]}})

# Logging
import logging
logger = logging.getLogger(__name__)

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,bound,nullbound,sin,cos
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,to_list
from src.utils import slice_slice
from src.utils import pi,e

from src.io import load,dump,join,split


def _variables(parameters,hyperparameters,parameter,group):
	'''
	Get variables from parameters
	Args:
		parameters (array): Array of parameters to compute variables
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for variables
		group (str): Parameter group for variables
	Returns:
		variable (array): variables
	'''

	kwargs = hyperparameters[parameter]['kwargs']
	method = hyperparameters[parameter]['method']
	scale = [hyperparameters[parameter]['scale']*2*pi,2*pi]
	scale = [1,1]
	index = hyperparameters[parameter]['group'].index(group)
	if method in ['constrained']:
		if parameter in ['xy'] and group in [('x',)]:
			variable = (
				scale[0]*parameters[0]*
				cos(scale[1]*parameters[1])
			)		

		elif parameter in ['xy'] and group in [('y',)]:
			variable = (
				scale[0]*parameters[0]*
				sin(scale[1]*parameters[1])
			)		

		elif parameter in ['z'] and group in [('z',)]:
			variable = (
				scale[index]*
				parameters[index]
			)
			
		elif parameter in ['zz'] and group in [('zz',)]:
			variable = (
				scale[index]/(8*kwargs['min']*kwargs['delta'])*
				parameters[index]
			)
	elif method in ['unconstrained']:
		if parameter in ['z'] and group in [('z',)]:
			variable = (
				scale[index]*
				parameters[index]
			)
		elif parameter in ['zz'] and group in [('zz',)]:
			variable = (
				scale[index]/(8*kwargs['min']*kwargs['delta'])*
				parameters[index]
			)
		else:
			variable = scale[index]*parameters[index]	
	else:
		if parameter in ['z'] and group in [('z',)]:
			variable = (
				scale[index]*
				parameters[index]
			)
		if parameter in ['zz'] and group in [('zz',)]:
			variable = (
				scale[index]/(8*kwargs['min']*kwargs['delta'])*
				parameters[index]
			)
		else:
			variable = scale[index]*parameters[index]

	return variable


def _features(parameters,hyperparameters,parameter,group):
	'''
	Get features from parameters
	Args:
		parameters (array): Array of parameters to compute features
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for features
		group (str): Parameter group for features
	Returns:
		feature (array): features
	'''

	kwargs = hyperparameters[parameter]['kwargs']
	method = hyperparameters[parameter]['method']
	l = len(hyperparameters[parameter]['group'])
	shape = (l,parameters.shape[0]//l,*parameters.shape[1:])

	if method in ['constrained']:
		wrapper = bound
	elif method in ['unconstrained']:
		wrapper = nullbound
	else:
		wrapper = nullbound


	feature = wrapper(parameters,kwargs).reshape(shape) 

	return feature


def _parameters(parameters,hyperparameters,parameter,group):
	'''
	Get parameters from parameters
	Args:
		parameters (array): Array of parameters to compute parameters
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for parameters
		group (str): Parameter group for parameters
	Returns:
		parameters (array): parameters
	'''

	return parameters


def _constraints(parameters,hyperparameters,parameter,group):
	'''
	Get constraints from parameters
	Args:
		parameters (array): Array of parameters to compute constraints
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for constraints
		group (str): Parameter group for constraints
	Returns:
		constraints (array): constraints
	'''

	kwargs = hyperparameters[parameter]['kwargs']
	method = hyperparameters[parameter]['method']
	scale = hyperparameters[parameter]['kwargs']['lambda']
	constants = hyperparameters[parameter]['constants']['features'][-1]

	if method in ['constrained']:
		if parameter in ['xy'] and group in [('x',),('y',)]:
			constraint = (
				(scale[0]*(parameters[...,constants['slice']] - constants['value'])**2).sum()
			)
		elif parameter in ['z'] and group in [('z',)]:
			constraint = 0
		
		elif parameter in ['zz'] and group in [('zz',)]:
			constraint = 0
	elif method in ['unconstrained']:
		constraint = 0
	else:
		constraint = 0

	return constraint


def _gradient_constraints(parameters,hyperparameters,parameter,group):
	'''
	Get gradients of constraints from parameters
	Args:
		parameters (array): Array of parameters to compute constraints
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for constraints
		group (str): Parameter group for constraints
	Returns:
		grad (array): gradient of constraints
	'''

	#TODO (finish analytic derivatives for variables functions as a matrix of (k,l) shape for k output parameters and l parameters)
	# ie) k = m*r for r = 2N, and l = m*q for q = 2,2*N input phases and amplitudes

	kwargs = hyperparameters[parameter]['kwargs']
	method = hyperparameters[parameter]['method']
	scale = hyperparameters[parameter]['kwargs']['lambda']

	shape = parameters.shape

	grad = zeros(shape)

	grad = grad.ravel()

	return grad	


def _gradients(parameters,hyperparameters,parameter,group):
	'''
	Get gradient of variables from parameters
	Args:
		parameters (array): Array of parameters to compute variables
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for variables
		group (str): Parameter group for variables
	Returns:
		grad (array): gradient of variables
	'''	

	#TODO (finish analytic derivatives for variables functions as a matrix of (k,l) shape for k output parameters and l parameters)
	# ie) k = m*r for r = 2N, and l = m*q for q = 2,2*N input phases and amplitudes

	method = hyperparameters[parameter]['method']
	scale = [hyperparameters[parameter]['scale']*2*pi,2*pi]

	shape = parameters.shape
	
	grad = zeros(shape)
	
	return grad	


def check(hyperparameters,cls=None):
	'''
	Check hyperparameters
	Args:	
		hyperparameters (dict): Hyperparameters
		cls (object): Class instance
	'''

	# Update with checked values
	updates = {
		**{attr: {
			'value': (lambda parameter,hyperparameters,attr=attr: {kwarg: [{prop: array(i.get(prop,[])) for prop in ['slice','value']}
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
				**{kwarg: getattr(cls,kwarg) for kwarg in cls.__dict__ 
					if isinstance(getattr(cls,kwarg),scalars)
					},
				**{kwarg: getattr(np,kwarg)(hyperparameters[parameter]['parameters']) if hyperparameters[parameter].get('parameters') 
						  else getattr(np,kwarg)(hyperparameters[parameter]['bounds']['parameters'])
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
							hyperparameters[parameter][attr][group] = jit(partial(updates[attr]['value'](parameter,hyperparameters),hyperparameters=hyperparameters,parameter=parameter,group=group))
				else:
					hyperparameters[parameter][attr] = updates[attr]['value'](parameter,hyperparameters)

	return 


def parameterize(data,shape,hyperparameters,check=None,initialize=None,cls=None,dtype=None):
	'''
	Initialize data of shapes of parameters based on shape of data
	Args:
		data (object): Data corresponding to parameters
		shape (iterable[int]): Shape of data
		hyperparameters (dict): Dictionary of parameter groupings, with dictionary values with properties:
			'category':str : category of parameter
			'group':iterable[iterable[str]] : iterable of groups associated with parameter grouping
			'shape':dict[str,iterable[int]] : dictionary of shape of each parameter layer
			'locality':dict[str,iterable[str]] : dictionary of locality of each axis of each parameter layer
			'boundaries':dict[str,iterable[dict[str,iterable]]] : dictionary of boundary indices and values of each axis of each parameter layer {'layer':[{'slice':[indices_axis],'value':[values_axis]}]}
			'constants':dict[str,iterable[dict[str,iterable]]] : dictionary of constant indices and values of each axis of each parameter layer {'layer':[{'slice':[indices_axis],'value':[values_axis]}]}
		check (callable): Function with signature check(group,index,axis) to check if index of data for axis corresponds to group
		initialize (callable): Function with signature initialize(parameters,shape,hyperparameters,reset=None,dtype=None) to initialize parameter values
		cls (object): Class instance to update hyperparameters		
		dtype (data_type): Data type of values		
	Returns:
		attributes (dict): Dictionary of parameter attributes ['shape','values','slice','index','parameters','features','variables','constraints']
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
	'''
	
	# Implicit parameterizations that interact with the data to produce the output are called variables x
	# These variables are parameterized by the explicit parameters theta such that x = x(theta)
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

	#time = timer()


	# Check hyperparameters
	check(hyperparameters,cls=cls)

	# Get number of dimensions of data
	ndim = len(shape)

	# Get properties of hyperparameters
	properties = ['category','group','shape','locality','boundaries','constants','use','parameters']
	assert all(all(prop in hyperparameters[parameter] 
		for prop in properties) 
		for parameter in hyperparameters), 'hyperparameters missing properties'


	# Remove not used parameters of hyperparameters
	for parameter in list(hyperparameters):
		if not hyperparameters[parameter]['use']:
			hyperparameters.pop(parameter)

	# Update properties of hyperparameters
	attrs = {
		('group',): (lambda parameter,attr,value: (
			[tuple(v) for v in value] if value is not None else [])),
		('boundaries','constants',): (lambda parameter,attr,value: (
			{k: [{prop:array(v.get(prop,[])) for prop in v} for v in value[k]] for k in value})),
		}
	for section in attrs:
		for attr in section:
			for parameter in hyperparameters:
				hyperparameters[parameter][attr] = attrs[section](parameter,attr,hyperparameters[parameter].get(attr))

	# Set layer functions
	funcs = {
		# 'parameters':_parameters,
		'features':_features,
		'variables':_variables,
		'constraints':_constraints,
		'gradients':_gradients,
		'gradient_constraints':_gradient_constraints,
		}
	for parameter in hyperparameters:
		for prop in funcs:
			if not isinstance(hyperparameters[parameter].get(prop),dict):
				hyperparameters[parameter][prop] = {}
			for group in hyperparameters[parameter]['group']:
				if not callable(hyperparameters[parameter][prop].get(group)):
					hyperparameters[parameter][prop][group] = funcs[prop]
				try:
					hyperparameters[parameter][prop][group] = jit(partial(
						hyperparameters[parameter][prop][group],
						hyperparameters=hyperparameters,
						parameter=parameter,
						group=group)
					)
				except:
					hyperparameters[parameter][prop][group] = jit(hyperparameters[parameter][prop][group])

	# Get attributes
	attributes = ['ndim','locality','size','indices','boundaries','constants','shape','slice','parameters','features','variables','values','constraints']

	# Get categories across all parameter groupings
	categories = [hyperparameters[parameter]['category'] for parameter in hyperparameters]
	categories = list(sorted(list(set(categories)),key=lambda i: categories.index(i)))
	

	# Get parameter groupings across categories
	parameters = {category: [parameter for parameter in hyperparameters if hyperparameters[parameter]['category'] == category]
		for category in categories
		}
	parameters = {category: list(sorted(list(set(parameters[category])),key=lambda i: parameters[category].index(i))) 
		for category in parameters
		}

	# Get layers across all parameter groupings
	layers = [layer 
		for parameter in hyperparameters 
		for prop in hyperparameters[parameter] 
		for layer in (
		hyperparameters[parameter][prop] 
		if ((prop in properties) and 
			(isinstance(hyperparameters[parameter][prop],dict)) and 
			(all(not callable(hyperparameters[parameter][prop][l]) for l in hyperparameters[parameter][prop]))) else 
		[])
		]
	layers = list(sorted(list(set(layers)),key=lambda i: layers.index(i)))


	# All groups for categories for keys of categories,parameters,groups,layers
	# Make sure groups are hashable as tuples
	groups = {
		category: {
			parameter: {tuple(group):{layer:{} for layer in layers} 
						for group in hyperparameters[parameter]['group']}
				for parameter in parameters[category]					
			}
			for category in categories
		}

	# Get function to check if index of data for axis corresponds to group 
	if check is None:
		check = lambda group,index,axis: True


	# Get indexes of attributes
	indexes = [(i,j,k) for i in ['put','take'] for j in ['key','category','layer'] for k in ['all','variable','constant']]


	# Get data: number of axes,locality,sizes multipliers,indicies of data,boundaries,constants 
	# for keys of categories,parameters,groups,layers

	# Get shapes,slices of axes based on number of variables per group and locality, of variables,features,parameters for each index type

	# - Slices of data for axes beyond (in front of) data size (of len(size)=ndim=2) are assumed to be [0] of fixed size 1

	data = {
		attribute:{
				category:{
					parameter:{
						group:{
							layer: {}
							for layer in groups[category][parameter][group]
							}
					for group in groups[category][parameter]
					}
				for parameter in groups[category]
				}
			for category in groups
			}
	for attribute in attributes
	}


	#Time = timer()
	#msg = 'setup'
	#print(msg,Time-time)
	#time = Time

	# Get non-indexed attributes for data
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:

					for l in [*layers,'constraints']:
						if isinstance(hyperparameters[parameter][l],dict):
							data[l][category][parameter][group][layer] = hyperparameters[parameter][l][group]
						else:
							data[l][category][parameter][group][layer] = hyperparameters[parameter][l]

					data['ndim'][category][parameter][group][layer] = len(hyperparameters[parameter]['shape'][layer])

					data['locality'][category][parameter][group][layer] = list(hyperparameters[parameter]['locality'][layer])

					data['size'][category][parameter][group][layer] = list(hyperparameters[parameter]['shape'][layer])

					data['indices'][category][parameter][group][layer] = [
						*[[i for i in range(1)] for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
						*[[i for i in range(shape[axis]) if check(group,i,axis)] for axis in range(0,ndim)],
						]

					data['boundaries'][category][parameter][group][layer] = [
						{
						'slice': array([((i if i>=0 else len(data['indices'][category][parameter][group][layer][axis])+i)
							if isinstance(i,(int,np.integer)) else
							(int(len(data['indices'][category][parameter][group][layer][axis])*i)))
							for i in hyperparameters[parameter]['boundaries'][layer][axis].get('slice',[])]),
						'value': array(hyperparameters[parameter]['boundaries'][layer][axis].get('value',[]))
						}
						for axis in range(0,data['ndim'][category][parameter][group][layer])
						]

					data['constants'][category][parameter][group][layer] = [
						{
						'slice': array([((i if i>=0 else len(data['indices'][category][parameter][group][layer][axis])+i)
							if isinstance(i,(int,np.integer)) else
							(int(len(data['indices'][category][parameter][group][layer][axis])*i)))						
							for i in hyperparameters[parameter]['constants'][layer][axis].get('slice',[])]),
						'value': array(hyperparameters[parameter]['constants'][layer][axis].get('value',[]))
						}						
						for axis in range(0,data['ndim'][category][parameter][group][layer])
						]
					
	# Get indexed attributes for data
	subindex = ('key','all',)
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:

					index = ('put',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							((-data['size'][category][parameter][group][layer][axis]*
							(len(data['indices'][category][parameter][group][layer][axis])
							# if (axis > (ndim-data['ndim'][category][parameter][group][layer]) or data['locality'][category][parameter][group][layer][axis] in ['local'] or layer in ['variables']) else 1))
							if (data['locality'][category][parameter][group][layer][axis] in ['local'] or layer in ['variables']) else 1))
							if data['size'][category][parameter][group][layer][axis] < 0 else 
							data['size'][category][parameter][group][layer][axis]
							)
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							*[slice(0,data['shape'][category][parameter][group][layer][index][axis],1)
							for axis in range(0,0)],
							*[slice(0,data['shape'][category][parameter][group][layer][index][axis],1)
							for axis in range(0,1)],
							*[slice(0,data['shape'][category][parameter][group][layer][index][axis],1)
							for axis in range(1,data['ndim'][category][parameter][group][layer])],
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							((-data['size'][category][parameter][group][layer][axis]*
							(len(data['indices'][category][parameter][group][layer][axis]) 
							if (data['locality'][category][parameter][group][layer][axis] in ['local']) else 1))
							if data['size'][category][parameter][group][layer][axis] < 0 else 
							data['size'][category][parameter][group][layer][axis]
							)
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])
						
						data['slice'][category][parameter][group][layer][index] = tuple([
							*[slice(0,data['shape'][category][parameter][group][layer][index][axis],1)
							for axis in range(0,0)],
							*[slice(0,data['shape'][category][parameter][group][layer][index][axis],1)
							for axis in range(0,1)],
							*[slice(0,data['shape'][category][parameter][group][layer][index][axis],1)
							for axis in range(1,data['ndim'][category][parameter][group][layer])],
							])


	subindex = ('key','variable',)
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:

					index = ('put',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:							
						data['shape'][category][parameter][group][layer][index] = tuple([
							data['shape'][category][parameter][group][layer][refindex][axis]							
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(slice(
								(data['slice'][category][parameter][group][layer][refindex][axis].start + 
								sum(any((j in data['boundaries'][category][parameter][group][layer][axis]['slice']) for j in i) 
									for i in [[0]])),
								(data['slice'][category][parameter][group][layer][refindex][axis].stop - 
								sum(any((j in data['boundaries'][category][parameter][group][layer][axis]['slice']) for j in i) 
									for i in [[data['shape'][category][parameter][group][layer][refindex][axis]-1]])),										
								(data['slice'][category][parameter][group][layer][refindex][axis].step)
								)
							if isinstance(data['slice'][category][parameter][group][layer][refindex][axis],slice) else
							[i for i in data['slice'][category][parameter][group][layer][refindex][axis] 
								if i not in data['boundaries'][category][parameter][group][layer][axis]['slice']])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis] - 
							len(data['boundaries'][category][parameter][group][layer][axis]['slice']))
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(slice(
								(data['slice'][category][parameter][group][layer][refindex][axis].start + 
								sum(any((j in data['boundaries'][category][parameter][group][layer][axis]['slice']) for j in i) 
									for i in [[0]])),
								(data['slice'][category][parameter][group][layer][refindex][axis].stop - 
								sum(any((j in data['boundaries'][category][parameter][group][layer][axis]['slice']) for j in i) 
									for i in [[data['shape'][category][parameter][group][layer][refindex][axis]-1]])),										
								(data['slice'][category][parameter][group][layer][refindex][axis].step)
								)		
							if isinstance(data['slice'][category][parameter][group][layer][refindex][axis],slice) else
							[i for i in data['slice'][category][parameter][group][layer][refindex][axis] 
								if i not in data['boundaries'][category][parameter][group][layer][axis]['slice']])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

	subindex = ('key','constant',)
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:

					index = ('put',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:							
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
							data['shape'][category][parameter][group][layer][refindex][axis])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(slice(0,data['shape'][category][parameter][group][layer][refindex][axis],1)
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else								
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+i)
							if isinstance(i,(int,np.integer)) else
							(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) for i in set([
								*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
								*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
								len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])))							
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(slice(0,data['shape'][category][parameter][group][layer][refindex][axis],1)							
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else							
							([((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+i)
								if isinstance(i,(int,np.integer)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i))))
								for i in range(*data['slice'][category][parameter][group][layer][refindex][axis].indices(
								data['shape'][category][parameter][group][layer][refindex][axis])) 
								if i in set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])
									])]								
							if isinstance(data['slice'][category][parameter][group][layer][refindex][axis],slice) else
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+i)
								if isinstance(i,(int,np.integer)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) 
								for i in data['slice'][category][parameter][group][layer][refindex][axis] 
								if i in set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])]))
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])							


	# Get indexed attributes for data
	subindex = ('category','all',)
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:
					
					index = ('put',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						

					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							*[(max(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[(sum(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[(max(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[slice_slice(*[data['slice'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									],
									index=[all([catgry==category,param==parameter,layr==layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									].index(True))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['shape'][category][parameter][group][layer][refindex][axis] 
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[slice_slice(*[data['slice'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									],
									index=[all([catgry==category,param==parameter,layr==layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									].index(True))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

	subindex = ('category','variable',)
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:
					
					index = ('put',)
					refindex = ('key','variable',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							*[(max(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[(sum(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[(max(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							*[data['slice'][category][parameter][group][layer][refindex][axis]
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[slice_slice(*[data['slice'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									],
									index=[all([catgry==category,param==parameter,layr==layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									].index(True))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

					index = ('take',)
					refindex = ('key','variable',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[slice_slice(*[data['slice'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									],
									index=[all([catgry==category,param==parameter,layr==layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									].index(True))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

	subindex = ('category','constant',)
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:
	
					index = ('put',)
					refindex = ('category','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:							
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
							data['shape'][category][parameter][group][layer][refindex][axis])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(data['slice'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else								
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+i)
							if isinstance(i,(int,np.integer)) else
							(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) for i in set([
								*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
								*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

					index = ('take',)
					refindex = ('category','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
								len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])))							
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(data['slice'][category][parameter][group][layer][refindex][axis]							
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else							
							([((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+i)
								if isinstance(i,(int,np.integer)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i))))
								for i in range(*data['slice'][category][parameter][group][layer][refindex][axis].indices(
								data['shape'][category][parameter][group][layer][refindex][axis])) 
								if i in set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])
									])]								
							if isinstance(data['slice'][category][parameter][group][layer][refindex][axis],slice) else
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+i)
								if isinstance(i,(int,np.integer)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) 
								for i in data['slice'][category][parameter][group][layer][refindex][axis] 
								if i in set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])]))
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])	




	# Get indexed attributes for data
	subindex = ('layer','all',)
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:
					
					index = ('put',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							*[(max(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[(sum(len(data['indices'][catgry][param][grp][layr][axis]) 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][:]])
									for l,layr in enumerate([layer])
									)
									if layer in ['variables'] else 
								sum(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[(max(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[(data['indices'][category][parameter][group][layer][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									],
									index=[all([catgry==category,param==parameter,layr==layer])
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									].index(True)))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[(data['indices'][category][parameter][group][layer][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									],
									index=[all([catgry==category,param==parameter,layr==layer])
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									].index(True)))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

	subindex = ('layer','variable',)
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:
					
					index = ('put',)
					refindex = ('key','variable',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							*[(max(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[(sum(len(data['indices'][catgry][param][grp][layr][axis]) 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][:]])
									for l,layr in enumerate([layer])
									)
									if layer in ['variables'] else 
								sum(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))									
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[(max(data['shape'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[(data['indices'][category][parameter][group][layer][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									],
									index=[all([catgry==category,param==parameter,layr==layer])
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									].index(True)))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

					index = ('take',)
					refindex = ('key','variable',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)],
							*[data['shape'][category][parameter][group][layer][refindex][axis]
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(0,data['ndim'][category][parameter][group][layer]-ndim)],
							*[(data['indices'][category][parameter][group][layer][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][catgry][param][grp][layr][refindex][axis] 
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									],
									index=[all([catgry==category,param==parameter,layr==layer])
									for c,catgry in enumerate([catgry for catgry in groups])
									for p,param in enumerate([param for param in groups[catgry]])
									for g,grp in enumerate([*[grp for grp in groups[catgry][param]][0:1]])
									for l,layr in enumerate([layer])
									].index(True)))
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim,
									data['ndim'][category][parameter][group][layer]-ndim+1)], 
							*[data['slice'][category][parameter][group][layer][refindex][axis] 
								for axis in range(data['ndim'][category][parameter][group][layer]-ndim+1,
									data['ndim'][category][parameter][group][layer])],
							])

	subindex = ('layer','constant',)
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:
				
					index = ('put',)
					refindex = ('layer','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:							
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
							data['shape'][category][parameter][group][layer][refindex][axis])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(data['slice'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else								
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+i)
							if isinstance(i,(int,np.integer)) else
							(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) for i in set([
								*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
								*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

					index = ('take',)
					refindex = ('layer','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
								len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])))							
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(data['slice'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])) == 0) and
								(any(len(set([
									*to_list(data['boundaries'][category][parameter][group][layer][ax]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][ax]['slice'])])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else							
							([((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+i)
								if isinstance(i,(int,np.integer)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i))))
								for i in range(*data['slice'][category][parameter][group][layer][refindex][axis].indices(
								data['shape'][category][parameter][group][layer][refindex][axis])) 
								if i in set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])
									])]								
							if isinstance(data['slice'][category][parameter][group][layer][refindex][axis],slice) else
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+i)
								if isinstance(i,(int,np.integer)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) 
								for i in data['slice'][category][parameter][group][layer][refindex][axis] 
								if i in set([
									*to_list(data['boundaries'][category][parameter][group][layer][axis]['slice']),
									*to_list(data['constants'][category][parameter][group][layer][axis]['slice'])])]))
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])						


	# for category in groups:
	# 	print(category)
	# 	for parameter in groups[category]:
	# 		for group in groups[category][parameter]:
	# 			print(group)
	# 			for layer in groups[category][parameter][group]:
	# 				print(layer)
	# 				for attr in ['shape','slice']:
	# 					print(attr)
	# 					for index in data[attr][category][parameter][group][layer]:
	# 						print(index,data[attr][category][parameter][group][layer][index])
	# 			print()
	# 	print()


	#Time = timer()
	#msg = 'dict'
	#print(msg,Time-time)
	#time = Time


	# Initialize values

	# Get values of parameters of different category

	# Initialize values parameters for each category,parameter,group,layer
	# reshape, bound, impose boundary conditions accordingly, and assign category parameters

	if initialize is None:
		initialize = lambda parameters,shape,hyperparameters,**kwargs: parameters

	attribute = 'values'

	data[attribute].update({**{category:{layer:None for layer in layers} for category in categories},**{layer:None for layer in layers}})

	for category in groups:
		#Time = timer()
		#msg = 'category %s'%(category)
		#print(msg,Time-time)
		#time = Time
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:
					
					# Initialize values for category and layer
					if data[attribute][category][layer] is None:							
						attr = 'shape'
						index = ('put','category','all')
						shape = data[attr][category][parameter][group][layer][index]
						data[attribute][category][layer] = zeros(shape,dtype=dtype)

					if data[attribute][layer] is None:							
						attr = 'shape'
						index = ('put','layer','all')
						shape = data[attr][category][parameter][group][layer][index]

						data[attribute][layer] = zeros(shape,dtype=dtype)

					# Create in category and set in both layer and category

					if layer in ['parameters']:

						# Function for values
						func = initialize

						# Hyperparameters for parameter
						hyperparams = hyperparameters[parameter]		

						# Get number of dimensions
						attr = 'ndim'
						ndim = data[attr][category][parameter][group][layer]

						# Existing parameters for parameter
						attr = 'parameters'
						values = data[attr][category][parameter][group][layer]

						# Existence of values
						reset =  values is None
			
						# Get shape of values to take
						attr = 'shape'
						reflayer = layer						
						index = ('take','layer','variable')
						shape = data[attr][category][parameter][group][reflayer][index]

						# Get shape of values to put
						attr = 'shape'
						reflayer = 'parameters'
						index = ('put','layer','variable')
						shapes = data[attr][category][parameter][group][reflayer][index]

						# Get slice of values to put
						attr = 'slice'
						reflayer = 'parameters'
						index = ('put','layer','variable')
						slices = data[attr][category][parameter][group][reflayer][index]

						# Set values depending on existence
						if reset:
							values = zeros(shape,dtype=dtype)
						else:
							values = array(values,dtype=dtype)

						#a = timer()
						values = padding(values,shape,random=None)
						#b = timer()
						#print('padding',b-a)

						#a = timer()
						values = func(values,shape,hyperparams,reset=reset,slices=slices,shapes=shapes,layer=layer,dtype=dtype)
						#b = timer()
						#print(layer,'init',b-a)

						# Get slices of values to put
						attr = 'slice'
						reflayer = layer						
						index = ('put','category','variable')
						slices = data[attr][category][parameter][group][reflayer][index]

						data[attribute][category][layer] = data[attribute][category][layer].at[slices].set(values)

						attr = 'slice'
						reflayer = layer						
						index = ('put','layer','variable')
						slices = data[attr][category][parameter][group][reflayer][index]

						data[attribute][layer] = data[attribute][layer].at[slices].set(values)

					elif layer in ['features']:

						# Function for values
						func = hyperparameters[parameter][layer][group]

						# Get shape of values to take
						attr = 'shape'
						reflayer = 'parameters'						
						index = ('take','layer','variable')
						shape = data[attr][category][parameter][group][reflayer][index]

						# Get slice of values to take
						attr = 'slice'
						reflayer = 'parameters'						
						index = ('take','layer','variable')
						indices = data[attr][category][parameter][group][reflayer][index]

						# Get values to take to put
						values = data[attribute][reflayer][indices]

						#a = timer()
						values = func(values)
						#b = timer()
						#print(layer,'init',b-a)

						# Get slices of values to put
						attr = 'slice'
						reflayer = layer						
						index = ('put','category','variable')
						slices = data[attr][category][parameter][group][reflayer][index]

						data[attribute][category][layer] = data[attribute][category][layer].at[slices].set(values)

						attr = 'slice'
						reflayer = layer						
						index = ('put','layer','variable')
						slices = data[attr][category][parameter][group][reflayer][index]

						data[attribute][layer] = data[attribute][layer].at[slices].set(values)						


					elif layer in ['variables']:

						# Function for values					
						func = hyperparameters[parameter][layer][group]

						# Get shape of values to take
						attr = 'shape'
						reflayer = 'features'						
						index = ('take','layer','variable')
						shape = data[attr][category][parameter][group][reflayer][index]

						# Get shape of values to take
						attr = 'slice'
						reflayer = 'features'
						index = ('put','layer','variable')
						indices = data[attr][category][parameter][group][reflayer][index]

						# Get values to take to put
						values = data[attribute][reflayer][indices]

						#a = timer()
						values = func(values)
						#b = timer()
						#print(layer,'init',b-a)

						# Get slices of values to put
						attr = 'slice'
						reflayer = layer						
						index = ('put','category','variable')
						slices = data[attr][category][parameter][group][reflayer][index]

						data[attribute][category][layer] = data[attribute][category][layer].at[slices].set(values)

						attr = 'slice'
						reflayer = layer						
						index = ('put','layer','variable')
						slices = data[attr][category][parameter][group][reflayer][index]

						data[attribute][layer] = data[attribute][layer].at[slices].set(values)						

					# Boundaries and constants of the form [{i:value} for axis in axes]
					attrs = ['boundaries','constants']

					ndim = min(len(data[attr][category][parameter][group][layer]) for attr in attrs)

					values = [{to_list(i): value
						for attr in attrs 
						for i,value in zip(
							data[attr][category][parameter][group][layer][axis]['slice'],
							data[attr][category][parameter][group][layer][axis]['value'])
						} 
						for axis in range(ndim)]

					values = [expand_dims(
						array([values[axis][i] for i in values[axis]]),
						[ax for ax in range(ndim) if ax != axis])
						for axis in range(ndim)]

					#a = timer()
					# Get slices and shape of boundaries,constants to initialize
					for axis in range(ndim):
						if values[axis].size > 0:

							attr = 'slice'
							index = ('put','category','constant')
							slices = data[attr][category][parameter][group][layer][index]

							try:
								data[attribute][category][layer] = data[attribute][category][layer].at[slices].set(values[axis])
							except:
								for k,i in enumerate(slices[axis]):

									refslices = tuple([slices[ax] if ax != axis else i for ax in range(ndim)])
									refindices = tuple([slice(None) if ax != axis else k for ax in range(ndim)])

									data[attribute][category][layer] = data[attribute][category][layer].at[refslices].set(values[axis][refindices])


							attr = 'slice'
							index = ('put','layer','constant')
							slices = data[attr][category][parameter][group][layer][index]

							try:
								data[attribute][layer] = data[attribute][layer].at[slices].set(values[axis])
							except:
								for k,i in enumerate(slices[axis]):

									refslices = tuple([slices[ax] if ax != axis else i for ax in range(ndim)])
									refindices = tuple([slice(None) if ax != axis else k for ax in range(ndim)])

									data[attribute][layer] = data[attribute][layer].at[refslices].set(values[axis][refindices])

					#b = timer()
					#print('bcs',b-a)

					attr = 'shape'
					index = ('take','layer','variable')
					shape = data[attr][category][parameter][group][layer][index]

					attr = 'slice'
					index = ('take','layer','variable')
					slices = data[attr][category][parameter][group][layer][index]

					attr = 'shape'
					index = ('put','layer','variable')
					shapes = data[attr][category][parameter][group][layer][index]

					attr = 'slice'
					index = ('put','layer','variable')
					indices = data[attr][category][parameter][group][layer][index]					
					
					# #print(category,parameter,group,layer,':',shape,slices,'->',shapes,indices)

					# #print(data[attribute][category][layer])
					# #print(data[attribute][layer])
					# #print()

	# Setup attributes from data
	attrs = ['shape','values','slice','index','parameters','features','variables','constraints']
	attributes = {attr:{} for attr in attrs}

	# Get parameters

	layer = 'parameters'
	attributes['shape'][layer] = None
	attributes['values'][layer] = None
	attributes['slice'][layer] = {}
	attributes['index'][layer] = {}
	attributes[layer][layer] = {}

	attribute = 'values'
	category = 'variable'
	layer = 'parameters'
	values = data[attribute][category][layer]

	attribute = 'values'
	category = 'variable'
	layer = 'parameters'
	parameters = data[attribute][category][layer]

	sliced = False

	attribute = 'slice'
	category = 'variable'

	for parameter in data[attribute][category]:
		#Time = timer()
		#msg = 'param %s'%(parameter)
		#print(msg,Time-time)
		#time = Time

		layer = 'parameters'
		attrs = ['slice','index',layer]
		for attr in attrs:
			attributes[attr][layer][parameter] = {}
		for group in data[attribute][category][parameter]:
		
			if not sliced:

				attr = 'ndim'
				layer = 'parameters'
				ndim = data[attr][category][parameter][group][layer]

				attr = 'slice'
				layer = 'parameters'
				index = ('put','category','variable')
				slices = data[attr][category][parameter][group][layer][index]

				slices = tuple([*[slice(None)]*(ndim-parameters.ndim),slice(None),*slices[ndim-parameters.ndim+1:]])

				values = parameters[slices]

				sliced = True


			layer = 'parameters'
			attr = 'slice'
			layer = 'parameters'
			index = ('put','category','variable')
			slices = data[attr][category][parameter][group][layer][index]

			slices = tuple([
				*slices[:ndim-values.ndim],
				slices[ndim-values.ndim],
				*[slice(0,values.shape[axis],1) for axis in range(ndim-values.ndim+1,ndim)]
				])

			layer = 'parameters'
			attr = 'slice'
			layer = 'parameters'
			index = ('put','category','variable')
			indices = data[attr][category][parameter][group][layer][index]	


			layer = 'parameters'
			funcs = []			

			func = lambda parameters,values,slices,indices,funcs=funcs: parameters

			layer = 'parameters'
			attr = 'slice'
			attributes[attr][layer][parameter][group] = slices

			layer = 'parameters'
			attr = 'index'				
			attributes[attr][layer][parameter][group] = indices

			layer = 'parameters'
			attr = layer
			attributes[attr][layer][parameter][group] = func


	layer = 'parameters'
	shape = values.shape
	values = values

	attributes['shape'][layer] = shape
	attributes['values'][layer] = values


	# Get features

	layer = 'features'
	attributes['shape'][layer] = None
	attributes['values'][layer] = None
	attributes['slice'][layer] = {}
	attributes['index'][layer] = {}
	attributes[layer][layer] = {}

	attribute = 'values'
	layer = 'features'
	values = data[attribute][layer]

	attribute = 'values'
	attr = 'shape'
	layer = 'parameters'
	parameters = attributes[attribute][layer].reshape(attributes[attr][layer])

	attribute = 'slice'
	category = 'variable'

	for parameter in data[attribute][category]:

		#Time = timer()
		#msg = 'param %s'%(parameter)
		#print(msg,Time-time)
		#time = Time

		layer = 'features'
		attrs = ['slice','index',layer]
		for attr in attrs:
			attributes[attr][layer][parameter] = {}
		for group in data[attribute][category][parameter]:

			attr = 'ndim'
			layer = 'parameters'
			ndim = data[attr][category][parameter][group][layer]

			index = ('take','category','variable')
			layer = 'parameters'
			slices = data[attribute][category][parameter][group][layer][index]

			slices = tuple([
				*slices[:ndim-parameters.ndim],
				slices[ndim-parameters.ndim],
				*[slice(0,parameters.shape[axis],1) for axis in range(ndim-parameters.ndim+1,ndim)]
				])

			index = ('put','layer','variable')
			layer = 'features'
			indices = data[attribute][category][parameter][group][layer][index]				

			layer = 'features'
			funcs = [data[attr][category][parameter][group][layer] for attr in ['features']]

			func = lambda parameters,values,slices,indices,funcs=funcs: values.at[indices].set(funcs[0](parameters[slices]))

			values = func(parameters,values,slices,indices)

			layer = 'features'
			attr = 'slice'
			attributes[attr][layer][parameter][group] = slices

			layer = 'features'
			attr = 'index'				
			attributes[attr][layer][parameter][group] = indices

			layer = 'features'
			attr = layer
			attributes[attr][layer][parameter][group] = func


	layer = 'features'
	shape = values.shape
	values = values

	attributes['shape'][layer] = shape
	attributes['values'][layer] = values
	


	# Get variables

	layer = 'variables'
	attributes['shape'][layer] = None
	attributes['values'][layer] = None
	attributes['slice'][layer] = {}
	attributes['index'][layer] = {}
	attributes[layer][layer] = {}

	attribute = 'values'
	layer = 'variables'
	values = data[attribute][layer]

	attribute = 'values'
	attr = 'shape'
	layer = 'parameters'
	parameters = attributes[attribute][layer].reshape(attributes[attr][layer])

	attribute = 'slice'
	category = 'variable'

	for parameter in data[attribute][category]:
		
		#Time = timer()
		#msg = 'param %s'%(parameter)
		#print(msg,Time-time)
		#time = Time		

		layer = 'variables'
		attrs = ['slice','index',layer]
		for attr in attrs:
			attributes[attr][layer][parameter] = {}
		for group in data[attribute][category][parameter]:

			attr = 'ndim'
			layer = 'parameters'
			ndim = data[attr][category][parameter][group][layer]

			index = ('take','category','variable')
			layer = 'parameters'
			slices = data[attribute][category][parameter][group][layer][index]

			slices = tuple([
				*slices[:ndim-parameters.ndim],
				slices[ndim-parameters.ndim],
				*[slice(0,parameters.shape[axis],1) for axis in range(ndim-parameters.ndim+1,ndim)]
				])

			index = ('put','layer','variable')
			layer = 'variables'
			indices = data[attribute][category][parameter][group][layer][index]				

			layer = 'variables'
			funcs = [data[attr][category][parameter][group][layer] for attr in ['features','variables']]

			func = lambda parameters,values,slices,indices,funcs=funcs: values.at[indices].set(funcs[1](funcs[0](parameters[slices])))

			values = func(parameters,values,slices,indices)

			layer = 'variables'
			attr = 'slice'
			attributes[attr][layer][parameter][group] = slices

			layer = 'variables'
			attr = 'index'				
			attributes[attr][layer][parameter][group] = indices

			layer = 'variables'
			attr = layer
			attributes[attr][layer][parameter][group] = func


	layer = 'variables'
	shape = values.shape
	values = values

	attributes['shape'][layer] = shape
	attributes['values'][layer] = values



	# Get constraints

	layer = 'constraints'
	attributes['shape'][layer] = None
	attributes['values'][layer] = None
	attributes['slice'][layer] = {}
	attributes['index'][layer] = {}
	attributes[layer][layer] = {}

	attribute = 'values'
	layer = 'variables'
	values = 0

	attribute = 'values'
	attr = 'shape'
	layer = 'parameters'
	parameters = attributes[attribute][layer].reshape(attributes[attr][layer])

	attribute = 'slice'
	category = 'variable'

	for parameter in data[attribute][category]:

		#Time = timer()
		#msg = 'param %s'%(parameter)
		#print(msg,Time-time)
		#time = Time

		layer = 'constraints'
		attrs = ['slice','index',layer]
		for attr in attrs:
			attributes[attr][layer][parameter] = {}
		for group in data[attribute][category][parameter]:

			attr = 'ndim'
			layer = 'parameters'
			ndim = data[attr][category][parameter][group][layer]

			index = ('take','category','variable')
			layer = 'parameters'
			slices = data[attribute][category][parameter][group][layer][index]

			slices = tuple([
				*slices[:ndim-parameters.ndim],
				slices[ndim-parameters.ndim],
				*[slice(0,parameters.shape[axis],1) for axis in range(ndim-parameters.ndim+1,ndim)]
				])

			index = ('put','layer','variable')
			layer = 'variables'
			indices = data[attribute][category][parameter][group][layer][index]				

			layer = 'variables'
			funcs = [data[attr][category][parameter][group][layer] for attr in ['features','constraints']]

			func = lambda parameters,values,slices,indices,funcs=funcs: values + (funcs[1](funcs[0](parameters[slices])))

			values = func(parameters,values,slices,indices)

			layer = 'constraints'
			attr = 'slice'
			attributes[attr][layer][parameter][group] = slices

			layer = 'constraints'
			attr = 'index'				
			attributes[attr][layer][parameter][group] = indices

			layer = 'constraints'
			attr = layer
			attributes[attr][layer][parameter][group] = func

	layer = 'constraints'
	shape = ()
	values = values

	attributes['shape'][layer] = shape
	attributes['values'][layer] = values


	# # Print
	# attribute = 'values'
	# for layer in attributes[attribute]:
	# 	#print(layer)
	# 	for attr in attributes:
	# 		if layer not in attributes[attr]:
	# 			continue
	# 		if isinstance(attributes[attr][layer],dict):
	# 			#print(attr)
	# 			for parameter in attributes[attr][layer]:
	# 				for group in attributes[attr][layer][parameter]:
	# 					#print(parameter,group,attributes[attr][layer][parameter][group])
	# 		elif attr not in [attribute]:
	# 			#print(attr)
	# 			#print(attributes[attr][layer])
	# 		elif attr in [attribute]:
	# 			#print(attr)
	# 			attr = 'shape'
	# 			#print(attributes[attribute][layer].reshape(attributes[attr][layer]))


	# #print()
	# #print('---- Testing Start ----')

	# # Test
	# attribute = 'values'
	# layer = 'parameters'
	# attr = 'shape'
	# parameters = attributes[attribute][layer].reshape(attributes[attr][layer])

	# parameters = parameters + 0*onp.random.rand(*parameters.shape)


	# attribute = 'shape'
	# layer = 'parameters'
	# parameters = parameters.reshape(attributes[attribute][layer])

	# attribute = 'slice'

	# layers = attributes[attribute]
	# layers = ['variables']
	# for layer in layers:
	# 	attr = 'values'
	# 	atr = 'shape'
	# 	values = attributes[attr][layer].reshape(attributes[atr][layer])
	# 	for parameter in attributes[attribute][layer]:
	# 		for group in attributes[attribute][layer][parameter]:

	# 			attr = layer
	# 			func = attributes[attr][layer][parameter][group]
				
	# 			attr = 'slice'
	# 			slices = attributes[attr][layer][parameter][group]
				
	# 			attr = 'index'
	# 			indices = attributes[attr][layer][parameter][group]

	# 			# #print(layer,parameter,group,slices,indices,parameters.shape,values.shape)

	# 			values = func(parameters,values,slices,indices)


	
	# 	#print(layer)
	# 	#print(values)
	# 	#print()


	# #print('---- Testing Complete ----')

	return attributes