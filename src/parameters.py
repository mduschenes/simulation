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
		data (dict): Dictionary of parameter data, with nested keys of layers,categories,parameters,groups, and data of
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

	# Each set of values is described by keys of layers,categories,parameters,groups
	

	# Shape of axes for keys of layers,categories,parameters,groups
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

	# Depending on locality, functions of 'take' indexed values that return to 'put' indexed values 
	# should return either arrays of shape:
	# locality in ['local']  exact shape of the 'put' indexed variables
	# locality in ['global'] broadcastable (size 1) shape of the 'put' sliced variables
	# For example if a function of 'take' sliced values of size (l,k,) and the 'put' sliced values are of size (k,), 
	# then the function should roughly use the l values to return the correct shape

	# For a given indices,locality,and sizes of layer,category,parameter,group and axis, 
	
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
	properties = ['category','group','shape','locality','boundaries','constants','parameters']

	assert all(all(prop in hyperparameters[parameter] for prop in properties) for parameter in hyperparameters), "hyperparameters missing properties"

	# Get attributes
	attributes = ['values','parameters','ndim','locality','size','indices','boundaries','constants','shape','slice']

	# Get layers across all parameter groupings
	layers = [layer 
		for parameter in hyperparameters 
		for prop in hyperparameters[parameter] 
		for layer in (hyperparameters[parameter][prop] if prop in properties and isinstance(hyperparameters[parameter][prop],dict) else [])
		]
	layers = list(sorted(list(set(layers)),key=lambda i: layers.index(i)))

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

	# All groups for categories for keys of layers,categories,parameters,groups
	# Make sure groups are hashable as tuples
	groups = {
		layer: {
			category: {
				parameter: [tuple(group) for group in hyperparameters[parameter]['group']]
					for parameter in parameters[category]					
				}
				for category in categories
			}
			for layer in layers
		}

	# Get function to check if index of data for axis corresponds to group 
	if check is None:
		check = lambda group,index,axis: True


	# Get indexes of attributes
	indexes = [(i,j,k) for i in ['put','take'] for j in ['key','category','layer'] for k in ['all','variable','constant']]


	# Get data: number of axes,locality,sizes multipliers,indicies of data,boundaries,constants 
	# for keys of layers,categories,parameters,groups

	# Get shapes,slices of axes based on number of variables per group and locality, of variables,features,parameters for each index type

	# - Slices of data for axes beyond (in front of) data size (of len(size)=ndim=2) are assumed to be [0] of fixed size 1

	data = {
		attribute:{
			layer:{
				category:{
					parameter:{
						group:{} 
					for group in groups[layer][category][parameter]
					}
				for parameter in groups[layer][category]
				}
			for category in groups[layer]
			}
		for layer in groups
		} 
	for attribute in attributes}


	# Get non-indexed attributes for data
	for layer in groups:
		for category in groups[layer]:
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:

					data['parameters'][layer][category][parameter][group] = hyperparameters[parameter]['parameters']
					
					data['ndim'][layer][category][parameter][group] = len(hyperparameters[parameter]['shape'][layer])

					data['locality'][layer][category][parameter][group] = list(hyperparameters[parameter]['locality'][layer])

					data['size'][layer][category][parameter][group] = list(hyperparameters[parameter]['shape'][layer])

					data['indices'][layer][category][parameter][group] = [
						*[[i for i in range(1)] for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
						*[[i for i in range(shape[axis]) if check(group,i,axis)] for axis in range(0,ndim)],
						]

					data['boundaries'][layer][category][parameter][group] = [
						dict({
							((i if i>=0 else len(data['indices'][layer][category][parameter][group][axis])+int(i))
							if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
							(int(len(data['indices'][layer][category][parameter][group][axis])*float(i)))):
							hyperparameters[parameter]['boundaries'][layer][axis][i]
							for i in hyperparameters[parameter]['boundaries'][layer][axis]
							})
						for axis in range(0,data['ndim'][layer][category][parameter][group])
						]
					
					data['constants'][layer][category][parameter][group] = [
						dict({
							((i if i>=0 else len(data['indices'][layer][category][parameter][group][axis])+int(i))
							if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
							(int(len(data['indices'][layer][category][parameter][group][axis])*float(i)))):
							hyperparameters[parameter]['constants'][layer][axis][i]
							for i in hyperparameters[parameter]['constants'][layer][axis]
							})
						for axis in range(0,data['ndim'][layer][category][parameter][group])
						]


	# Get indexed attributes for data
	subindex = ('key','all',)
	for layer in groups:
		for category in groups[layer]:
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:

					index = ('put',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							((-data['size'][layer][category][parameter][group][axis]*
							(len(data['indices'][layer][category][parameter][group][axis])
							if (axis > (ndim-data['ndim'][layer][category][parameter][group]) or data['locality'][layer][category][parameter][group][axis] in ['local'] or layer in ['variables']) else 1))
							if data['size'][layer][category][parameter][group][axis] < 0 else 
							data['size'][layer][category][parameter][group][axis]
							)
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[slice(0,data['shape'][layer][category][parameter][group][index][axis],1)
							for axis in range(0,0)],
							*[slice(0,data['shape'][layer][category][parameter][group][index][axis],1)
							for axis in range(0,1)],
							*[slice(0,data['shape'][layer][category][parameter][group][index][axis],1)
							for axis in range(1,data['ndim'][layer][category][parameter][group])],
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							((-data['size'][layer][category][parameter][group][axis]*
							(len(data['indices'][layer][category][parameter][group][axis]) 
							if (data['locality'][layer][category][parameter][group][axis] in ['local']) else 1))
							if data['size'][layer][category][parameter][group][axis] < 0 else 
							data['size'][layer][category][parameter][group][axis]
							)
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])
						
						data['slice'][layer][category][parameter][group][index] = tuple([
							*[slice(0,data['shape'][layer][category][parameter][group][index][axis],1)
							for axis in range(0,0)],
							*[slice(0,data['shape'][layer][category][parameter][group][index][axis],1)
							for axis in range(0,1)],
							*[slice(0,data['shape'][layer][category][parameter][group][index][axis],1)
							for axis in range(1,data['ndim'][layer][category][parameter][group])],
							])


	subindex = ('key','variable',)
	for layer in groups:
		for category in groups[layer]:
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:

					index = ('put',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:							
						data['shape'][layer][category][parameter][group][index] = tuple([
							data['shape'][layer][category][parameter][group][refindex][axis]							
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							(slice(
								(data['slice'][layer][category][parameter][group][refindex][axis].start + 
								sum(any((j in data['boundaries'][layer][category][parameter][group][axis]) for j in i) 
									for i in [[0]])),
								(data['slice'][layer][category][parameter][group][refindex][axis].stop - 
								sum(any((j in data['boundaries'][layer][category][parameter][group][axis]) for j in i) 
									for i in [[data['shape'][layer][category][parameter][group][refindex][axis]-1]])),										
								(data['slice'][layer][category][parameter][group][refindex][axis].step)
								)
							if isinstance(data['slice'][layer][category][parameter][group][refindex][axis],slice) else
							[i for i in data['slice'][layer][category][parameter][group][refindex][axis] 
								if i not in data['boundaries'][layer][category][parameter][group][axis]])
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							(data['shape'][layer][category][parameter][group][refindex][axis] - 
							len(data['boundaries'][layer][category][parameter][group][axis]))
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							(slice(
								(data['slice'][layer][category][parameter][group][refindex][axis].start + 
								sum(any((j in data['boundaries'][layer][category][parameter][group][axis]) for j in i) 
									for i in [[0]])),
								(data['slice'][layer][category][parameter][group][refindex][axis].stop - 
								sum(any((j in data['boundaries'][layer][category][parameter][group][axis]) for j in i) 
									for i in [[data['shape'][layer][category][parameter][group][refindex][axis]-1]])),										
								(data['slice'][layer][category][parameter][group][refindex][axis].step)
								)		
							if isinstance(data['slice'][layer][category][parameter][group][refindex][axis],slice) else
							[i for i in data['slice'][layer][category][parameter][group][refindex][axis] 
								if i not in data['boundaries'][layer][category][parameter][group][axis]])
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

	subindex = ('key','constant',)
	for layer in groups:
		for category in groups[layer]:
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:

					index = ('put',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:							
						data['shape'][layer][category][parameter][group][index] = tuple([
							(data['shape'][layer][category][parameter][group][refindex][axis]
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else
							data['shape'][layer][category][parameter][group][refindex][axis])
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							(slice(0,data['shape'][layer][category][parameter][group][refindex][axis],1)
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else								
							[((i if i>=0 else data['shape'][layer][category][parameter][group][refindex][axis]+int(i))
							if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
							(int(data['shape'][layer][category][parameter][group][refindex][axis]*float(i)))) for i in set([
								*data['boundaries'][layer][category][parameter][group][axis],
								*data['constants'][layer][category][parameter][group][axis]])])
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							(data['shape'][layer][category][parameter][group][refindex][axis]
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else
								len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])))							
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							(slice(0,data['shape'][layer][category][parameter][group][refindex][axis],1)							
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else							
							([((i if i>=0 else data['shape'][layer][category][parameter][group][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][layer][category][parameter][group][refindex][axis]*float(i))))
								for i in range(*data['slice'][layer][category][parameter][group][refindex][axis].indices(
								data['shape'][layer][category][parameter][group][refindex][axis])) 
								if i in set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]
									])]								
							if isinstance(data['slice'][layer][category][parameter][group][refindex][axis],slice) else
							[((i if i>=0 else data['shape'][layer][category][parameter][group][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][layer][category][parameter][group][refindex][axis]*float(i)))) 
								for i in data['slice'][layer][category][parameter][group][refindex][axis] 
								if i in set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])]))
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])							


	# Get indexed attributes for data
	subindex = ('category','all',)
	for layer in groups:
		for category in groups[layer]:
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:
					
					index = ('put',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						

					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])],
									index=[all([layr==layer,catgry==category,param==parameter])
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])].index(True))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['shape'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])],
									index=[all([layr==layer,catgry==category,param==parameter])
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])].index(True))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

	subindex = ('category','variable',)
	for layer in groups:
		for category in groups[layer]:
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:
					
					index = ('put',)
					refindex = ('key','variable',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis]
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])],
									index=[all([layr==layer,catgry==category,param==parameter])
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])].index(True))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

					index = ('take',)
					refindex = ('key','variable',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])],
									index=[all([layr==layer,catgry==category,param==parameter])
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([category])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])].index(True))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

	subindex = ('category','constant',)
	for layer in groups:
		for category in groups[layer]:
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:
	
					index = ('put',)
					refindex = ('category','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:							
						data['shape'][layer][category][parameter][group][index] = tuple([
							(data['shape'][layer][category][parameter][group][refindex][axis]
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else
							data['shape'][layer][category][parameter][group][refindex][axis])
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							(slice(0,data['shape'][layer][category][parameter][group][refindex][axis],1)
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else								
							[((i if i>=0 else data['shape'][layer][category][parameter][group][refindex][axis]+int(i))
							if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
							(int(data['shape'][layer][category][parameter][group][refindex][axis]*float(i)))) for i in set([
								*data['boundaries'][layer][category][parameter][group][axis],
								*data['constants'][layer][category][parameter][group][axis]])])
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

					index = ('take',)
					refindex = ('category','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							(data['shape'][layer][category][parameter][group][refindex][axis]
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else
								len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])))							
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							(slice(0,data['shape'][layer][category][parameter][group][refindex][axis],1)							
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else							
							([((i if i>=0 else data['shape'][layer][category][parameter][group][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][layer][category][parameter][group][refindex][axis]*float(i))))
								for i in range(*data['slice'][layer][category][parameter][group][refindex][axis].indices(
								data['shape'][layer][category][parameter][group][refindex][axis])) 
								if i in set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]
									])]								
							if isinstance(data['slice'][layer][category][parameter][group][refindex][axis],slice) else
							[((i if i>=0 else data['shape'][layer][category][parameter][group][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][layer][category][parameter][group][refindex][axis]*float(i)))) 
								for i in data['slice'][layer][category][parameter][group][refindex][axis] 
								if i in set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])]))
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])	




	# Get indexed attributes for data
	subindex = ('layer','all',)
	for layer in groups:
		for category in groups[layer]:
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:
					
					index = ('put',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(len(data['indices'][layr][catgry][param][grp][axis]) 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][:]]))
									if layer in ['variables'] else 
								sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(data['indices'][layer][category][parameter][group][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])],
									index=[all([layr==layer,catgry==category,param==parameter])
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])].index(True)))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(data['indices'][layer][category][parameter][group][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])],
									index=[all([layr==layer,catgry==category,param==parameter])
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])].index(True)))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

	subindex = ('layer','variable',)
	for layer in groups:
		for category in groups[layer]:
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:
					
					index = ('put',)
					refindex = ('key','variable',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(len(data['indices'][layr][catgry][param][grp][axis]) 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][:]]))
									if layer in ['variables'] else 
								sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))									
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(data['indices'][layer][category][parameter][group][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])],
									index=[all([layr==layer,catgry==category,param==parameter])
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])].index(True)))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

					index = ('take',)
					refindex = ('key','variable',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['shape'][layer][category][parameter][group][refindex][axis]
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(data['indices'][layer][category][parameter][group][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])],
									index=[all([layr==layer,catgry==category,param==parameter])
									for l,layr in enumerate([layer])
									for c,catgry in enumerate([catgry for catgry in groups[layr]])
									for p,param in enumerate([param for param in groups[layr][catgry]])
									for g,grp in enumerate([*[grp for grp in groups[layr][catgry][param]][0:1]])].index(True)))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)], 
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

	subindex = ('layer','constant',)
	for layer in groups:
		for category in groups[layer]:
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:
				
					index = ('put',)
					refindex = ('layer','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:							
						data['shape'][layer][category][parameter][group][index] = tuple([
							(data['shape'][layer][category][parameter][group][refindex][axis]
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else
							data['shape'][layer][category][parameter][group][refindex][axis])
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							(data['slice'][layer][category][parameter][group][refindex][axis]
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else								
							[((i if i>=0 else data['shape'][layer][category][parameter][group][refindex][axis]+int(i))
							if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
							(int(data['shape'][layer][category][parameter][group][refindex][axis]*float(i)))) for i in set([
								*data['boundaries'][layer][category][parameter][group][axis],
								*data['constants'][layer][category][parameter][group][axis]])])
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

					index = ('take',)
					refindex = ('layer','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							(data['shape'][layer][category][parameter][group][refindex][axis]
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else
								len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])))							
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							(data['slice'][layer][category][parameter][group][refindex][axis]
							if ((len(set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])) == 0) and
								(any(len(set([
									*data['boundaries'][layer][category][parameter][group][ax],
									*data['constants'][layer][category][parameter][group][ax]])) > 0
								for ax in range(0,data['ndim'][layer][category][parameter][group])))) else							
							([((i if i>=0 else data['shape'][layer][category][parameter][group][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][layer][category][parameter][group][refindex][axis]*float(i))))
								for i in range(*data['slice'][layer][category][parameter][group][refindex][axis].indices(
								data['shape'][layer][category][parameter][group][refindex][axis])) 
								if i in set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]
									])]								
							if isinstance(data['slice'][layer][category][parameter][group][refindex][axis],slice) else
							[((i if i>=0 else data['shape'][layer][category][parameter][group][refindex][axis]+int(i))
								if isinstance(i,int) or (isinstance(i,str) and int(i) == float(i)) else
								(int(data['shape'][layer][category][parameter][group][refindex][axis]*float(i)))) 
								for i in data['slice'][layer][category][parameter][group][refindex][axis] 
								if i in set([
									*data['boundaries'][layer][category][parameter][group][axis],
									*data['constants'][layer][category][parameter][group][axis]])]))
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])						


	# Initialize values

	# Get values of parameters of different category

	# Initialize values parameters for each layer,category,parameter,group
	# reshape, bound, impose boundary conditions accordingly, and assign category parameters

	if initialize is None:
		initialize = lambda parameters,shape,hyperparameters,**kwargs: parameters

	attribute = 'values'

	for layer in groups:
		data[attribute][layer] = {}
		data[attribute][layer][None] = None
		for category in groups[layer]:
			data[attribute][layer][category] = None
			for parameter in groups[layer][category]:			
				for group in groups[layer][category][parameter]:
					
					# Initialize values for category
					for catgry in [None,category]:
						if data[attribute][layer][catgry] is None:							
							attr = 'shape'
							index = {None: ('put','layer','all'), category: ('put','category','all')}[catgry]
							shape = data[attr][layer][category][parameter][group][index]

							data[attribute][layer][catgry] = zeros(shape,dtype=dtype)


					# Create in category and set in both layer and category

					if layer in ['parameters']:

						# Function for values
						func = initialize

						# Hyperparameters for parameter
						hyperparams = hyperparameters[parameter]		

						# Get number of dimensions
						attr = 'ndim'
						ndim = data[attr][layer][category][parameter][group]

						# Existing parameters for parameter
						attr = 'parameters'
						values = data[attr][layer][category][parameter][group]

						# Existence of values
						reset =  values is None
			
						# Get shape of values to take
						attr = 'shape'
						reflayer = layer						
						index = ('take','layer','variable')
						shape = data[attr][reflayer][category][parameter][group][index]



						# Get shape of values to put
						attr = 'shape'
						reflayer = 'parameters'
						index = ('put','layer','variable')
						shapes = data[attr][reflayer][category][parameter][group][index]

						# Get slice of values to put
						attr = 'slice'
						reflayer = 'parameters'
						index = ('put','layer','variable')
						slices = data[attr][reflayer][category][parameter][group][index]

						# Set values depending on existence
						if reset:
							values = zeros(shape,dtype=dtype)
						else:
							values = array(values,dtype=dtype)

						values = padding(values,shape,random=None)
						
						values = func(values,shape,hyperparams,reset=reset,slices=slices,shapes=shapes,layer=layer,dtype=dtype)

						# Get slices of values to put
						for catgry in [None,category]:
							attr = 'slice'
							reflayer = layer						
							index = {None: ('put','layer','variable'), category: ('put','category','variable')}[catgry]
							slices = data[attr][reflayer][category][parameter][group][index]

							data[attribute][layer][catgry] = data[attribute][layer][catgry].at[slices].set(values)

					elif layer in ['features']:

						# Function for values
						func = hyperparameters[parameter][layer][group]

						# Get shape of values to take
						attr = 'shape'
						reflayer = 'parameters'						
						index = ('take','layer','variable')
						shape = data[attr][reflayer][category][parameter][group][index]

						# Get slice of values to take
						attr = 'slice'
						reflayer = 'parameters'						
						index = ('take','layer','variable')
						indices = data[attr][reflayer][category][parameter][group][index]

						# Get values to take to put
						values = data[attribute][reflayer][None][indices]

						values = func(values)

						# Get slices of values to put
						for catgry in [None,category]:
							attr = 'slice'
							reflayer = layer						
							index = {None: ('put','layer','variable'), category: ('put','category','variable')}[catgry]
							slices = data[attr][reflayer][category][parameter][group][index]

							data[attribute][layer][catgry] = data[attribute][layer][catgry].at[slices].set(values)


					elif layer in ['variables']:

						# Function for values					
						func = hyperparameters[parameter][layer][group]

						# Get shape of values to take
						attr = 'shape'
						reflayer = 'features'						
						index = ('take','layer','variable')
						shape = data[attr][reflayer][category][parameter][group][index]

						# Get shape of values to take
						attr = 'slice'
						reflayer = 'features'
						index = ('put','layer','variable')
						indices = data[attr][reflayer][category][parameter][group][index]

						# Get values to take to put
						values = data[attribute][reflayer][None][indices]

						values = func(values)


						# Get slices of values to put
						for catgry in [None,category]:
							attr = 'slice'
							reflayer = layer						
							index = {None: ('put','layer','variable'), category: ('put','category','variable')}[catgry]
							slices = data[attr][reflayer][category][parameter][group][index]

							data[attribute][layer][catgry] = data[attribute][layer][catgry].at[slices].set(values)

					# Boundaries and constants of the form [{i:value} for axis in axes]
					attrs = ['boundaries','constants']

					ndim = min(len(data[attr][layer][category][parameter][group]) for attr in attrs)

					values = [{i: data[attr][layer][category][parameter][group][axis][i]
						for attr in attrs 
						for i in data[attr][layer][category][parameter][group][axis]} 
						for axis in range(ndim)]

					values = [expand_dims(
						array([values[axis][i] for i in values[axis]]),
						[ax for ax in range(ndim) if ax != axis])
						for axis in range(ndim)]


					# Get slices and shape of boundaries,constants to initialize
					for axis in range(ndim):
						if values[axis].size > 0:
							for catgry in [None,category]:

								attr = 'slice'
								index = {None: ('put','layer','constant'), category: ('put','category','constant')}[catgry]
								slices = data[attr][layer][category][parameter][group][index]

								try:
									data[attribute][layer][catgry] = data[attribute][layer][catgry].at[slices].set(values[axis])
								except:
									for k,i in enumerate(slices[axis]):

										refslices = tuple([slices[ax] if ax != axis else i for ax in range(ndim)])
										refindices = tuple([slice(None) if ax != axis else k for ax in range(ndim)])

										data[attribute][layer][catgry] = data[attribute][layer][catgry].at[refslices].set(values[axis][refindices])

					if layer in ['parameters','features','variables'] and category in ['variable','constant']:# and parameter in ['xy'] and group in [('y',)]:
						if layer in ['parameters']:

							func = hyperparameters[parameter]['features'][group]
							func2 = hyperparameters[parameter]['variables'][group]

							index = 'take'
							attr = 'slice'
							reflayer = layer						
							refindex = (index,'category','variable')
							slices = data[attr][reflayer][category][parameter][group][refindex]

							attr = 'values'
							values = data[attr][layer][category][slices]

							_values = data[attr][layer][category]
							_values_ = data[attr][layer][None]

							print('layer---',parameter,group,data[attr][layer][category].shape,values.shape,slices)							
							for ilayer in ['parameters','features','variables']:
								print(ilayer)
								for index in ['put','take']:
									for ref in ['category','layer']:
										attr = 'shape'
										reflayer = ilayer						
										refindex = (index,ref,'variable')
										shape = data[attr][reflayer][category][parameter][group][refindex]

										attr = 'slice'
										reflayer = ilayer						
										refindex = (index,ref,'variable')
										slices = data[attr][reflayer][category][parameter][group][refindex]
										print(index,ref,shape,slices)
							

							print('func parameters',layer,values.shape)
							print(values.round(3))
							print(_values.round(3))
							print(_values_.round(3))
							print()
							print('func features',func(values).shape)
							print(func(values).round(3))
							print()
							print('func variables',func2(func(values)).shape)							
							print(func2(func(values)).round(3))
							print()
					# if layer in ['variables']:
					for catgry in [None,category]:
						print('vals!@!',layer,category,catgry,parameter,group)
						print(data[attribute][layer][catgry].round(3))
					print()

	exit()
	return data