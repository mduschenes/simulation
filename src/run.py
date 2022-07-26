#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime
from copy import deepcopy as deepcopy
from time import time as timer
from functools import partial

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp
import jax
import jax.numpy as np
import jax.scipy as sp
import jax.example_libraries.optimizers
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'
# np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})

# Logging
import logging

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import logconfig
from src.utils import jit,gradient,gradient_finite,gradient_fwd
from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product
from src.utils import summation,exponentiation
from src.utils import inner_abs2,inner_real2,inner_imag2
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_abs2,gradient_inner_real2,gradient_inner_imag2
from src.utils import eigh,qr
from src.utils import maximum,minimum,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,isclose,is_naninf,to_key_value 
from src.utils import initialize,parse,to_str,to_number,scinotation,datatype,slice_size
from src.utils import pi,e,delim
from src.utils import itg,flt,dbl

from src.dictionary import updater,getter,setter,permuter
from src.dictionary import leaves,counts,plant,grow

from src.parameters import parameterize
from src.operators import operatorize

from src.io import load,dump,copy,join,split

from src.process import process

from src.plot import plot

from src.optimize import Optimizer,Objective

def plotter(hyperparameters):
	'''
	Plot models
	Args:
		hyperparameters (dict): hyperparameters of models
	'''	

	# Get paths and kwargs

	paths = {
		'data':('sys','path','data','data'),
		'settings':('sys','path','config','plot'),
		'hyperparameters':('sys','path','config','process'),
		}
	
	kwargs = {kwarg: [] for kwarg in paths}

	for kwarg in kwargs:
		for key in hyperparameters:
			path = hyperparameters[key]
			for i in paths[kwarg]:
				path = path[i]
			kwargs[kwarg].append(path)

	fig,ax = process(**kwargs)
	return



def check(hyperparameters):
	'''
	Check hyperparameters
	Args:
		hyperparameters (dict): Hyperparameters
	'''

	# Load default hyperparameters
	path = 'config/settings.json'
	func = lambda key,iterable,elements: iterable.get(key,elements[key])
	updater(hyperparameters,load(path),func=func)

	# Check sections for correct attributes
	section = None
	updates = {
		'permutations': {
			'value': (lambda hyperparameters: {
							**{attr: (hyperparameters['permutations'][attr] 
									if not isinstance(hyperparameters['permutations'][attr],int) else 
									range(hyperparameters['permutations'][attr]))
								for attr in hyperparameters.get('permutations',{})}
							}),
			'default': (lambda hyperparameters: {}),
			'conditions': (lambda hyperparameters: True)
		},
		'groups': {
			'value': (lambda hyperparameters: hyperparameters['groups']),
			'default': (lambda hyperparameters: None),
			'conditions': (lambda hyperparameters: True)
		},		
		'label': {
			'value': (lambda hyperparameters: hyperparameters['hyperparameters']['label']),
			'default': (lambda hyperparameters: None),
			'conditions': (lambda hyperparameters: hyperparameters['hyperparameters'].get('label') is not None)				
		},
	}			
	for attr in updates:								
		hyperparameters[attr] = hyperparameters.get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[attr] = updates[attr]['value'](hyperparameters)

	section = 'sys'
	updates = {
		'path': {
			'value': (lambda hyperparameters: 	{
				attr: {
					path: join(
						split(hyperparameters[section]['path'][attr][path],directory=True),
						split(hyperparameters[section]['path'][attr][path],file=True),
						ext=split(hyperparameters[section]['path'][attr][path],ext=True)
					)
					for path in hyperparameters[section]['path'][attr]
					}
				for attr in hyperparameters[section]['path']
				}),
			'default': (lambda hyperparameters: None),
			'conditions': (lambda hyperparameters: hyperparameters[section].get('path') is None)
		},		
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)

	section = 'model'
	updates = {
		'tau': {
			'value': (lambda hyperparameters: hyperparameters[section]['tau']/hyperparameters['hyperparameters']['scale']),
			'default': (lambda hyperparameters: 1),
			'conditions': (lambda hyperparameters: hyperparameters['hyperparameters'].get('scale') is not None)
		},		
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)

	section = 'hyperparameters'
	updates = {
		'iterations': {
			'value': (lambda hyperparameters: int(hyperparameters[section]['iterations'])),
			'default': (lambda hyperparameters: 0),
			'conditions': (lambda hyperparameters: True)
		},		
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)

	section = 'parameters'
	updates = {
		'boundaries': {
			'value': (lambda parameter,hyperparameters: {attr: [{prop: array(i.get(prop,[])) for prop in ['slice','value']}
				for i in hyperparameters[section][parameter]['boundaries'][attr]] 
				for attr in hyperparameters[section][parameter]['boundaries']}),
			'default': (lambda parameter,hyperparameters: {}),
			'conditions': (lambda parameter,hyperparameters: True)				
		},
		'constants': {
			'value': (lambda parameter,hyperparameters: {attr: [{prop: array(i.get(prop,[])) for prop in ['slice','value']}
				for i in hyperparameters[section][parameter]['constants'][attr]] 
				for attr in hyperparameters[section][parameter]['constants']}),
			'default': (lambda parameter,hyperparameters: []),
			'conditions': (lambda parameter,hyperparameters: True)				
		},		
		'group': {
			'value': (lambda parameter,hyperparameters: [tuple(group) for group in hyperparameters[section][parameter]['group']]),
			'default': (lambda parameter,hyperparameters: []),
			'conditions': (lambda parameter,hyperparameters: True)				
		},
		**{attr: {
			'value': (lambda parameter,hyperparameters,attr=attr: hyperparameters['hyperparameters'].get(attr)),
			'default': (lambda parameter,hyperparameters,attr=attr: None),
			'conditions': (lambda parameter,hyperparameters,attr=attr: hyperparameters['parameters'][parameter].get(attr) is None)						
			} for attr in ['scale','initialization','random','smoothness','interpolation','pad']
		},
		**{attr: {
			'value': (lambda parameter,hyperparameters,attr=attr: None),#hyperparameters.get('seed',{}).get(attr)),
			'default': (lambda parameter,hyperparameters,attr=attr: None),
			'conditions': (lambda parameter,hyperparameters,attr=attr: hyperparameters['parameters'][parameter].get(attr) is None)						
			} for attr in ['seed']
		},		
		'locality': {
			'value':(lambda parameter,hyperparameters: hyperparameters['hyperparameters']['locality']),
			'default':(lambda parameter,hyperparameters: None),
			'conditions': (lambda parameter,hyperparameters: hyperparameters['hyperparameters'].get('locality') is not None)
		},		
	}			
	for parameter in hyperparameters[section]:
		for attr in updates:						
			hyperparameters[section][parameter][attr] = hyperparameters[section][parameter].get(attr,updates[attr]['default'](parameter,hyperparameters))
			if updates[attr]['conditions'](parameter,hyperparameters):
				hyperparameters[section][parameter][attr] = updates[attr]['value'](parameter,hyperparameters)


	section = 'process'
	updates = {
		'path': {
			'value': (lambda hyperparameters: 	{
				path: join(
					split(hyperparameters['sys']['path']['plot'][path],directory=True),
					'.'.join(split(hyperparameters['sys']['path']['plot'][path],file=True).split('.')[:]),
					ext=split(hyperparameters['sys']['path']['plot'][path],ext=True)
				)
				for path in hyperparameters['sys']['path']['plot']
				}),
			'default': (lambda hyperparameters: {}),
			'conditions': (lambda hyperparameters: (hyperparameters[section].get('path') is not None))
		},		
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)

	section = 'plot'
	updates = {}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)


	return


def setup(hyperparameters):
	'''
	Setup hyperparameters
	Args:
		hyperparameters (dict): Hyperparameters
	'''

	# Get settings
	settings = {}	

	# Check hyperparameters have correct values
	if hyperparameters is None:
		hyperparameters = {}
	check(hyperparameters)

	# Get timestamp
	timestamp = datetime.datetime.now().strftime('%d.%M.%Y.%H.%M.%S.%f')

	# Get permutations of hyperparameters
	permutations = hyperparameters['permutations']
	groups = hyperparameters['groups']
	permutations = permuter(permutations,groups=groups)

	# Get seeds for number of splits/seedings, for all nested hyperparameters leaves that involve a seed
	seed = hyperparameters['seed']['seed']
	size = hyperparameters['seed']['size']
	reset = hyperparameters['seed']['reset']

	key = 'seed'
	exclude = [('seed','seed',),('model','system','seed')]
	seedlings = [branch for branch in leaves(hyperparameters,key,returns='key') if branch not in exclude]
	count = len(seedlings)
	
	shape = (size,count,-1)
	size *= count

	seeds = PRNGKey(seed=seed,size=size,reset=reset).reshape(shape)
	
	# Get all enumerated keys and seeds for permutations and seedings of hyperparameters
	keys = {'.'.join(['%d'%(k) for k in [iteration,instance]]): (permutation,seed) 
		for iteration,permutation in enumerate(permutations) 
		for instance,seed in enumerate(seeds)}


	# Set settings with key and seed instances

	settings['seed'] = seeds

	settings['boolean'] = {attr: (
			(hyperparameters['boolean'].get(attr,False)) 
			# (attr not in ['train'] or not hyperparameters['boolean'].get('load',False))
			)
			for attr in hyperparameters['boolean']}

	settings['hyperparameters'] = {key: None for key in keys}
	settings['object'] = {key: None for key in keys}
	settings['logger'] = {key: None for key in keys}

	# Update key/seed instances of hyperparameters with updates
	for key in keys:

		# Set seed and key
		iteration,instance = map(int,key.split('.'))
		permutation,seed = keys[key]
		
		# Set settings
		settings['hyperparameters'][key] = deepcopy(hyperparameters)
		settings['object'][key] = None
		settings['logger'][key] = logconfig(__name__,
			conf=settings['hyperparameters'][key]['sys']['path']['config']['logger'])

		# Set hyperparameters updates with key/instance dependent settings
		updates = {}		

		updates.update({
			'model':{
				'system':{
					'key':key,
					'seed':instance,
					'timestamp':timestamp
					},
				},
			'sys':{
				'path': {
					attr: {
						path: join(
							split(settings['hyperparameters'][key]['sys']['path'][attr][path],directory=True),
							key,
							split(settings['hyperparameters'][key]['sys']['path'][attr][path],file=True),
							# '.'.join([
							# 	split(settings['hyperparameters'][key]['sys']['path'][attr][path],file=True),
							# 	*([key] if attr not in [] else [])
							# 	]),
							ext=split(settings['hyperparameters'][key]['sys']['path'][attr][path],ext=True)
						)
						for path in settings['hyperparameters'][key]['sys']['path'][attr]
						}
					for attr in settings['hyperparameters'][key]['sys']['path']
					},
				},
			})

		for branch,leaf in zip(seedlings,seed):
			grow(updates,branch,leaf)


		# Update hyperparameters
		setter(updates,permutation,delimiter=delim,copy=True)

		updater(settings['hyperparameters'][key],updates,copy=True)
		check(settings['hyperparameters'][key])

		
		# Copy config files
		directory = settings['hyperparameters'][key]['sys']['directory']['config']
		paths = settings['hyperparameters'][key]['sys']['path']['config']
		func = lambda key,iterable,elements: iterable.get(key,elements[key])
		for path in paths:
			source = paths[path]
			destination = join(directory,split(paths[path],directory=2),split(paths[path],file=1),ext=split(paths[path],ext=1))

			if path in ['settings']:
				data = deepcopy(settings['hyperparameters'][key])
			else:
				data = settings['hyperparameters'][key].get(path,{})
			try:
				try:
					source = load(source)
				except:
					try:
						source = join(
							split(source,directory=-1),
							'.'.join(split(source,file=True).split('.')[:]),
							ext=split(source,ext=True))
						source = load(source)
					except:
						raise
				updater(data,source,func=func)
				dump(data,destination)
			except:
				copy(source,destination)


		# Update config paths
		directory = settings['hyperparameters'][key]['sys']['directory']['config']
		paths = settings['hyperparameters'][key]['sys']['path']['config']
		for path in paths:
			paths[path] = join(directory,split(paths[path],directory=2),split(paths[path],file=1),ext=split(paths[path],ext=1))

	return settings


def run(hyperparameters):
	'''
	Run simulations
	Args:
		hyperparameters (dict): hyperparameters
	'''		

	settings = setup(hyperparameters)

	for key in settings['hyperparameters']:		

		if not any(settings['boolean'][attr] for attr in ['load','dump','train']):
			continue		

		hyperparameters = settings['hyperparameters'][key]

		cls = load(hyperparameters['class'])

		obj = cls(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

		if settings['boolean']['load']:
			obj.load()

		settings['object'][key] = obj

		if settings['boolean']['train']:

			parameters = obj.parameters
			hyperparameters = hyperparameters['optimize']

			func = obj.__func__
			callback = obj.__callback__

			optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparameters)

			parameters = optimizer(parameters)
		
		if settings['boolean']['dump']:	
			obj.dump()
		
		if settings['boolean']['plot']:
			obj.plot()

	if settings['boolean']['plot']:
		hyperparameters = settings['hyperparameters']
		plotter(hyperparameters)		

	return