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

from src.utils import slice_slice
from src.utils import pi,e

from src.io import load,dump,path_join,path_split

def init_parameters(data,shape,hyperparameters,func=None):
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
		func (callable): Function with signature func(group,index,axis) to check if index of data for axis corresponds to group
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

	# 'put': Shape,slice to assign to array
	# 'take': Shape,slice to take from array

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
	# locality in ['global'] broadcastable (size 1) shape of the 'put' indexed variables

	# For a given indices,locality,and sizes of layer,category,parameter,group and axis, 
	
	# The shapes and slices for each individual set of values 'take,put_key_all' for these keys are:
	# s = sizes 
	# k = len(indices) (k[axis=0 (parameter,variables), axis=1 (features)] = O(N) for data with datum on each of N sites, k[axis=1] = O(M) for M time steps)
	# q = {'local':-s if s<0 else s,'global':1}[locality]
	# The shape of the values will be 
	# shape['put_key_all'] = -s*k if s<0 else s
	# shape['take_value'] = -s*(k if local in ['local'] else 1) if s<0 else s

	# slice['variables']['put_key_all'] = indices if axis == 0 else slice(0,shape['put_key_all'],1)
	# shape['variables']['take_key_all'] = slice(0,shape['take_key_all'],1)
	# shape['features','parameters']['put_key_all'] = slice(0,shape['put_key_all'],1)
	# shape['features','parameters']['take_key_all'] = slice(0,shape['take_key_all'],1)

	# The other 'take,put_<type>' indexes involve summing all shapes corresponding to the keys that are within the type group, 
	# plus subtracting shapes corresponding with boundaries and constants
	# i.e) For 'take,put_parameter', all shapes for keys associated with a given parameter grouping are the total shape for all of these keys
	# and all slices are indices within this larger shape.


	# Get number of dimensions of data
	ndim = len(shape)

	# Get properties of hyperparameters
	properties = ['category','group','shape','locality','boundaries','constants']

	assert all(all(prop in hyperparameters[parameter] for prop in properties) for parameter in hyperparameters), "hyperparameters missing properties"

	# Get attributes
	attributes = ['ndim','locality','size','indices','boundaries','constants','shape','slice']

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
	if func is None:
		func = lambda group,index,axis: True


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

					data['ndim'][layer][category][parameter][group] = len(hyperparameters[parameter]['shape'][layer])

					data['locality'][layer][category][parameter][group] = list(hyperparameters[parameter]['locality'][layer])

					data['size'][layer][category][parameter][group] = list(hyperparameters[parameter]['shape'][layer])

					data['indices'][layer][category][parameter][group] = [
						*[[i for i in range(1)] for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
						*[[i for i in range(shape[axis]) if func(group,i,axis)] for axis in range(0,ndim)],
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
							(
							(-data['size'][layer][category][parameter][group][axis]*
							(len(data['indices'][layer][category][parameter][group][axis])
							if (data['locality'][layer][category][parameter][group][axis] in ['local'] or layer in ['variables']) else 1))
							if data['size'][layer][category][parameter][group][axis]<0 else 
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
							(
							(-data['size'][layer][category][parameter][group][axis]*
							(len(data['indices'][layer][category][parameter][group][axis]) 
							if (data['locality'][layer][category][parameter][group][axis] in ['local']) else 1))
							if data['size'][layer][category][parameter][group][axis]<0 else 
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
							len(data['boundaries'][layer][category][parameter][group][axis])
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							[i for i in data['boundaries'][layer][category][parameter][group][axis]]
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

					index = ('take',)
					refindex = ('key','all',)
					refindex = (*index,*refindex)
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							len(data['boundaries'][layer][category][parameter][group][axis])
							for axis in range(0,data['ndim'][layer][category][parameter][group])
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							([i for i in range(*data['slice'][layer][category][parameter][group][refindex][axis].indices(
								data['shape'][layer][category][parameter][group][refindex][axis])) 
								if i in data['boundaries'][layer][category][parameter][group][axis]]								
							if isinstance(data['slice'][layer][category][parameter][group][refindex][axis],slice) else
							[i for i in data['slice'][layer][category][parameter][group][refindex][axis] 
								if i in data['boundaries'][layer][category][parameter][group][axis]])
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
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]],
									index=[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]].index(
									data['slice'][layer][category][parameter][group][refindex][axis]))							
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
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
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
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]],
									index=[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]].index(
									data['slice'][layer][category][parameter][group][refindex][axis]))							
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
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]],
									index=[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]].index(
									data['slice'][layer][category][parameter][group][refindex][axis]))							
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
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]],
									index=[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]].index(
									data['slice'][layer][category][parameter][group][refindex][axis]))							
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
					refindex = ('key','constant',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[data['shape'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['shape'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[data['slice'][layer][category][parameter][group][refindex][axis]
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

					index = ('take',)
					refindex = ('key','constant',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[data['shape'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [category]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['shape'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[data['slice'][layer][category][parameter][group][refindex][axis]							
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
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
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(data['indices'][layer][category][parameter][group][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]],
									index=[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]].index(
									data['slice'][layer][category][parameter][group][refindex][axis])))								
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
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
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
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]],
									index=[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]].index(
									data['slice'][layer][category][parameter][group][refindex][axis])))							
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
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(data['indices'][layer][category][parameter][group][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]],
									index=[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]].index(
									data['slice'][layer][category][parameter][group][refindex][axis])))							
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
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[(max(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(data['indices'][layer][category][parameter][group][axis]
								if layer in ['variables'] else
								slice_slice(*[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]],
									index=[data['slice'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]].index(
									data['slice'][layer][category][parameter][group][refindex][axis])))							
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
					refindex = ('key','constant',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[data['shape'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['shape'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[data['slice'][layer][category][parameter][group][refindex][axis]
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

					index = ('take',)
					refindex = ('key','constant',)
					refindex = (*index,*refindex)						
					for index in [(*index,*subindex)]:
						data['shape'][layer][category][parameter][group][index] = tuple([
							*[data['shape'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[(sum(data['shape'][layr][catgry][param][grp][refindex][axis] 
									for layr in [layer]
									for catgry in [catgry for catgry in groups[layr]]
									for param in [param for param in groups[layr][catgry]]
									for grp in [[grp for grp in groups[layr][catgry][param]][0]]))
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['shape'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])

						data['slice'][layer][category][parameter][group][index] = tuple([
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(0,data['ndim'][layer][category][parameter][group]-ndim)],
							*[data['slice'][layer][category][parameter][group][refindex][axis]							
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim,
									data['ndim'][layer][category][parameter][group]-ndim+1)],
							*[data['slice'][layer][category][parameter][group][refindex][axis] 
								for axis in range(data['ndim'][layer][category][parameter][group]-ndim+1,
									data['ndim'][layer][category][parameter][group])],
							])





	for layer in groups:
		print('layer: ',layer)
		for category in groups[layer]:
			print('Category: ',category)
			for parameter in groups[layer][category]:
				print('Parameter: ',parameter)
				for group in groups[layer][category][parameter]:
					print('Group: ',group)
					for attribute in data:
						if isinstance(data[attribute][layer][category][parameter][group],dict):
							for index in data[attribute][layer][category][parameter][group]:
								print(attribute,index,data[attribute][layer][category][parameter][group][index])
						else:
							print(attribute,data[attribute][layer][category][parameter][group])
					print()
				print()
			print()

	exit()



	return data