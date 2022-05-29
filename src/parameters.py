#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial

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
PATHS = ['',".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,sigmoid
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer
from src.utils import slice_slice
from src.utils import pi,e

from src.io import load,dump,path_join,path_split

def init_parameters(data,shape,hyperparameters,check=None,initialize=None,dtype=None):
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
			'boundaries':dict[str,iterable[dict[str,float]]] : dictionary of boundary indices and values of each axis of each parameter layer
			'constants':dict[str,iterable[dict[str,float]]] : dictionary of constant indices and values of each axis of each parameter layer
		check (callable): Function with signature check(group,index,axis) to check if index of data for axis corresponds to group
		initialize (callable): Function with signature initialize(parameters,shape,hyperparameters,reset=None,dtype=None) to initialize parameter values
		dtype (data_type): Data type of values		
	Returns:
		data (dict): Dictionary of parameter data, with nested keys of categories,parameters,groups,layers, and data of
			'ndim': int : Number of dimensions of value corresponding to key
			'locality': iterable[int] : Locality of value for each axis of value corresponding to key
			'size': iterable[int] : Shape of multiplicative value of data shape for each axis of value corresponding to key
			'indices': iterable[iterable[int]]: Indices of data for each axis of value corresponding to key
			'boundaries': iterable[dict[int,float]]: Boundary indices and values for each axis of value corresponding to key
			'constants': iterable[dict[int,float]]: Boundary indices and values for each axis of value corresponding to key
			'shape': dict[str,tuple[int]]: Shape of value for different index types for each axis of value corresponding to key
			'slice': dict[str,tuple[slice]]: Slices of value for different index types for each axis of value corresponding to key

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

	# Each set of values is described by keys of categories,parameters,groups,layers
	

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
	# then the function should roughly use the l values to return the correct shape

	# For a given indices,locality,and sizes of category,parameter,group,layer and axis, 
	
	# The shapes and slices for each individual set of values ('take,put','key','all') for these keys are:
	# s = sizes 
	# k = len(indices) (k[axis=0 (parameters,variables), axis=1 (features)] = O(N) for data with datum on each of N sites, k[axis=1] = O(M) for M time steps)
	# The shape of the values will be 
	# shape['features','parameters','variables'][('put','key','all')] = -s*k if s < 0 else s
	# shape['features','parameters','variables'][('put','value','all')] = -s*(k if local in ['local'] else 1) if s < 0 else s

	# slice['variables'][('put','key','all')] = indices if axis == 0 else slice(0,shape[('put','key','all')],1)
	# shape['variables'][('take','key','all')] = slice(0,shape['take_key_all'],1)
	# shape['features','parameters'][('put','key','all')] = slice(0,shape[('put','key','all')],1)
	# shape['features','parameters'][('take','key','all')] = slice(0,shape[('take','key','all')],1)

	# The other ('take,put',<type>) indexes involve summing all shapes corresponding to the keys that are within the type group, 
	# plus subtracting shapes corresponding with boundaries and constants


	# Get number of dimensions of data
	ndim = len(shape)

	# Get properties of hyperparameters
	properties = ['category','group','shape','locality','boundaries','constants','parameters','features','variables']

	assert all(all(prop in hyperparameters[parameter] for prop in properties) for parameter in hyperparameters), "hyperparameters missing properties"

	# Get attributes
	attributes = ['ndim','locality','size','indices','boundaries','constants','shape','slice','parameters','features','variables','values']

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


	# Get non-indexed attributes for data
	for category in groups:
		for parameter in groups[category]:
			for group in groups[category][parameter]:			
				for layer in groups[category][parameter][group]:

					for l in layers:
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
						dict({
							((i if i>=0 else len(data['indices'][category][parameter][group][layer][axis])+int(i))
							if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
							(int(len(data['indices'][category][parameter][group][layer][axis])*float(i)))):
							hyperparameters[parameter]['boundaries'][layer][axis][i]
							for i in hyperparameters[parameter]['boundaries'][layer][axis]
							})
						for axis in range(0,data['ndim'][category][parameter][group][layer])
						]
					
					data['constants'][category][parameter][group][layer] = [
						dict({
							((i if i>=0 else len(data['indices'][category][parameter][group][layer][axis])+int(i))
							if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
							(int(len(data['indices'][category][parameter][group][layer][axis])*float(i)))):
							hyperparameters[parameter]['constants'][layer][axis][i]
							for i in hyperparameters[parameter]['constants'][layer][axis]
							})
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
							if (axis > (ndim-data['ndim'][category][parameter][group][layer]) or data['locality'][category][parameter][group][layer][axis] in ['local'] or layer in ['variables']) else 1))
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
								sum(any((j in data['boundaries'][category][parameter][group][layer][axis]) for j in i) 
									for i in [[0]])),
								(data['slice'][category][parameter][group][layer][refindex][axis].stop - 
								sum(any((j in data['boundaries'][category][parameter][group][layer][axis]) for j in i) 
									for i in [[data['shape'][category][parameter][group][layer][refindex][axis]-1]])),										
								(data['slice'][category][parameter][group][layer][refindex][axis].step)
								)
							if isinstance(data['slice'][category][parameter][group][layer][refindex][axis],slice) else
							[i for i in data['slice'][category][parameter][group][layer][refindex][axis] 
								if i not in data['boundaries'][category][parameter][group][layer][axis]])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis] - 
							len(data['boundaries'][category][parameter][group][layer][axis]))
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(slice(
								(data['slice'][category][parameter][group][layer][refindex][axis].start + 
								sum(any((j in data['boundaries'][category][parameter][group][layer][axis]) for j in i) 
									for i in [[0]])),
								(data['slice'][category][parameter][group][layer][refindex][axis].stop - 
								sum(any((j in data['boundaries'][category][parameter][group][layer][axis]) for j in i) 
									for i in [[data['shape'][category][parameter][group][layer][refindex][axis]-1]])),										
								(data['slice'][category][parameter][group][layer][refindex][axis].step)
								)		
							if isinstance(data['slice'][category][parameter][group][layer][refindex][axis],slice) else
							[i for i in data['slice'][category][parameter][group][layer][refindex][axis] 
								if i not in data['boundaries'][category][parameter][group][layer][axis]])
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
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
							data['shape'][category][parameter][group][layer][refindex][axis])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(slice(0,data['shape'][category][parameter][group][layer][refindex][axis],1)
							if ((len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else								
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+int(i))
							if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
							(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) for i in set([
								*data['boundaries'][category][parameter][group][layer][axis],
								*data['constants'][category][parameter][group][layer][axis]])])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
								len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])))							
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(slice(0,data['shape'][category][parameter][group][layer][refindex][axis],1)							
							if ((len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else							
							([((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i))))
								for i in range(*data['slice'][category][parameter][group][layer][refindex][axis].indices(
								data['shape'][category][parameter][group][layer][refindex][axis])) 
								if i in set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]
									])]								
							if isinstance(data['slice'][category][parameter][group][layer][refindex][axis],slice) else
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) 
								for i in data['slice'][category][parameter][group][layer][refindex][axis] 
								if i in set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])]))
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
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
							data['shape'][category][parameter][group][layer][refindex][axis])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(data['slice'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else								
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+int(i))
							if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
							(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) for i in set([
								*data['boundaries'][category][parameter][group][layer][axis],
								*data['constants'][category][parameter][group][layer][axis]])])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

					index = ('take',)
					refindex = ('category','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
								len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])))							
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(data['slice'][category][parameter][group][layer][refindex][axis]							
							if ((len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else							
							([((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i))))
								for i in range(*data['slice'][category][parameter][group][layer][refindex][axis].indices(
								data['shape'][category][parameter][group][layer][refindex][axis])) 
								if i in set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]
									])]								
							if isinstance(data['slice'][category][parameter][group][layer][refindex][axis],slice) else
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) 
								for i in data['slice'][category][parameter][group][layer][refindex][axis] 
								if i in set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])]))
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
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
							data['shape'][category][parameter][group][layer][refindex][axis])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(data['slice'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else								
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+int(i))
							if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
							(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) for i in set([
								*data['boundaries'][category][parameter][group][layer][axis],
								*data['constants'][category][parameter][group][layer][axis]])])
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

					index = ('take',)
					refindex = ('layer','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][category][parameter][group][layer][index] = tuple([
							(data['shape'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else
								len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])))							
							for axis in range(0,data['ndim'][category][parameter][group][layer])
							])

						data['slice'][category][parameter][group][layer][index] = tuple([
							(data['slice'][category][parameter][group][layer][refindex][axis]
							if ((len(set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][category][parameter][group][layer][ax],
									*data['constants'][category][parameter][group][layer][ax]])) > 0
								for ax in range(0,data['ndim'][category][parameter][group][layer])))) else							
							([((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i))))
								for i in range(*data['slice'][category][parameter][group][layer][refindex][axis].indices(
								data['shape'][category][parameter][group][layer][refindex][axis])) 
								if i in set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]
									])]								
							if isinstance(data['slice'][category][parameter][group][layer][refindex][axis],slice) else
							[((i if i>=0 else data['shape'][category][parameter][group][layer][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][category][parameter][group][layer][refindex][axis]*float(i)))) 
								for i in data['slice'][category][parameter][group][layer][refindex][axis] 
								if i in set([
									*data['boundaries'][category][parameter][group][layer][axis],
									*data['constants'][category][parameter][group][layer][axis]])]))
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

	# exit()
	# Initialize values

	# Get values of parameters of different category

	# Initialize values parameters for each category,parameter,group,layer
	# reshape, bound, impose boundary conditions accordingly, and assign category parameters

	if initialize is None:
		initialize = lambda parameters,shape,hyperparameters,**kwargs: parameters

	attribute = 'values'

	data[attribute].update({**{category:{layer:None for layer in layers} for category in categories},**{layer:None for layer in layers}})

	for category in groups:
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

						values = padding(values,shape,random=None)
						
						values = func(values,shape,hyperparams,reset=reset,slices=slices,shapes=shapes,layer=layer,dtype=dtype)

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

						values = func(values)

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

						values = func(values)


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

					values = [{i: data[attr][category][parameter][group][layer][axis][i]
						for attr in attrs 
						for i in data[attr][category][parameter][group][layer][axis]} 
						for axis in range(ndim)]

					values = [expand_dims(
						array([values[axis][i] for i in values[axis]]),
						[ax for ax in range(ndim) if ax != axis])
						for axis in range(ndim)]


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
					
					print(category,parameter,group,layer,':',shape,slices,'->',shapes,indices)

					# print(data[attribute][category][layer])
					values = data[attribute][layer]
					print(data[attribute][layer])
					print()



	# 				if layer in ['parameters','features','variables'] and category in ['variable','constant']:# and parameter in ['xy'] and group in [('y',)]:
	# 					if layer in ['parameters']:

	# 						func = hyperparameters[parameter]['features'][group]
	# 						func2 = hyperparameters[parameter]['variables'][group]

	# 						index = 'take'
	# 						attr = 'slice'
	# 						reflayer = layer						
	# 						refindex = (index,'category','variable')
	# 						slices = data[attr][category][parameter][group][reflayer][refindex]

	# 						attr = 'values'
	# 						values = data[attr][layer][category][slices]

	# 						_values = data[attr][layer][category]
	# 						_values_ = data[attr][layer][None]

	# 						print('layer---',parameter,group,data[attr][layer][category].shape,values.shape,slices)							
	# 						for ilayer in ['parameters','features','variables']:
	# 							print(ilayer)
	# 							for index in ['put','take']:
	# 								for ref in ['category','layer']:
	# 									attr = 'shape'
	# 									reflayer = ilayer						
	# 									refindex = (index,ref,'variable')
	# 									shape = data[attr][category][parameter][group][reflayer][refindex]

	# 									attr = 'slice'
	# 									reflayer = ilayer						
	# 									refindex = (index,ref,'variable')
	# 									slices = data[attr][category][parameter][group][reflayer][refindex]
	# 									print(index,ref,shape,slices)
							

	# 						print('func parameters',layer,values.shape)
	# 						print(values.round(3))
	# 						print(_values.round(3))
	# 						print(_values_.round(3))
	# 						print()
	# 						print('func features',func(values).shape)
	# 						print(func(values).round(3))
	# 						print()
	# 						print('func variables',func2(func(values)).shape)							
	# 						print(func2(func(values)).round(3))
	# 						print()
	# 				# if layer in ['variables']:
# 					print('vals!@!',layer,category,parameter,group)
# 					print(data[attribute][category][layer].round(3))
# 					print('vals!@!',layer,None,parameter,group)
# 					print(data[attribute][layer].round(3))
	# 				print()

	return data,values