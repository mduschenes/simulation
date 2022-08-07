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
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,isclose,is_equal,is_naninf,to_key_value 
from src.utils import initialize,parse,to_str,to_number,scinotation,datatype,slice_size
from src.utils import pi,e,delim
from src.utils import itg,flt,dbl

from src.dictionary import updater,getter,setter,permuter
from src.dictionary import leaves,grow

from src.parameters import parameterize
from src.operators import operatorize

from src.io import load,dump,copy,join,split

from src.process import process

from src.plot import plot

from src.optimize import Optimizer,Objective


CALL = 0

DEVICES = {
	'pc':{
		'args': lambda args: ['./%s'%(args[0]),*args[1:]],
		},
	'lsf':{
		'args': lambda args: ['bsub','<',*args[:1]],
		},	
	None: {
		'args': lambda args: [],
		},
	}

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
	section = 'sys'
	updates = {
		**{attr: {
			'value': (lambda hyperparameters,attr=attr: '' if hyperparameters[section].get(attr) is None else hyperparameters[section][attr]),
			'default': (lambda hyperparameters: ''),
			'conditions': (lambda hyperparameters: (True))
			} for attr in ['pwd','cwd']
		},		
		'path': {
			'value': (lambda hyperparameters: 	{
				attr: {
					path: join(
						split(hyperparameters[section]['path'][attr][path],directory=True),
						'.'.join(split(hyperparameters[section]['path'][attr][path],file=True).split('.')[:]),
						ext=split(hyperparameters[section]['path'][attr][path],ext=True),
					)
					for path in hyperparameters[section]['path'][attr]
					}
				for attr in hyperparameters[section]['path']
				}),
			'default': (lambda hyperparameters: {}),
			'conditions': (lambda hyperparameters: (hyperparameters[section].get('path') is not None))
		},		
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)

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

	return


def allowed(index,value,values):
	'''
	Check if value is allowed as per index
	Args:
		index (dict): Dictionary of allowed integer indices or values of the form {attr: index/value (int,iterable[int]/dict,iterable[dict])}
		value (dict): Dictionary of possible value of the form {attr: value (dict)}
		values (dict): Dictionary of all values of the form {attr: values}
	Returns:
		boolean (bool) : Boolean if value is allowed
	'''
	boolean = True
	for attr in index:
		if index[attr] is None:
			index[attr] = [index[attr]]
		elif isinstance(index[attr],int):
			if index[attr] < 0:
				index[attr] += len(values[attr])
			index[attr] = [index[attr]]
		elif isinstance(index[attr],dict):
			index[attr] = [index[attr]]
		for subindex in index[attr]:
			if subindex is None:
				boolean &= True
			elif isinstance(subindex,int):
				boolean &= is_equal(values[attr].index(value[attr]),subindex)
			elif isinstance(subindex,dict):
				boolean &= all(is_equal(value[attr][key],subindex[key]) for key in subindex)

	return boolean

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
	elif isinstance(hyperparameters,str):
		hyperparameters = load(hyperparameters)

	check(hyperparameters)

	# Get timestamp
	timestamp = datetime.datetime.now().strftime('%d.%M.%Y.%H.%M.%S.%f')

	# Get permutations of hyperparameters
	permutations = hyperparameters['permutations']['permutations']
	groups = hyperparameters['permutations']['groups']
	permutations = permuter(permutations,groups=groups)

	# Get seeds for number of splits/seedings, for all nested hyperparameters leaves that involve a seed
	seed = hyperparameters['seed']['seed']
	size = hyperparameters['seed']['size']
	reset = hyperparameters['seed']['reset']

	seed = seed if seed is not None else None
	size = size if size is not None else 1
	reset = reset if reset is not None else None


	# Find keys of seeds in hyperparameters
	key = 'seed'
	exclude = [('seed','seed',),('model','system','seed')]
	seedlings = [branch[0] for branch in leaves(hyperparameters,key,returns='both') if branch[0] not in exclude and branch[1] is None]

	count = len(seedlings)
	
	shape = (size,count,-1)
	size *= count

	seeds = PRNGKey(seed=seed,size=size,reset=reset).reshape(shape)

	# Get all allowed enumerated keys and seeds for permutations and seedlings of hyperparameters
	values = {'permutations':permutations,'seed':seeds}
	index = {attr: hyperparameters[attr]['index'] for attr in values}
	keys = {
		'.'.join(['%d'%(v[0]) for k,v in zip(values,value) if len(values[k])>1]):[v[1] for v in value]
		for value in itertools.product(*(zip(range(len(values[attr])),values[attr]) for attr in values))
		if allowed(
			{attr: index[attr] for attr in index},
			{attr: dict(zip(values,value))[attr] for attr in index},
			{attr: values[attr] for attr in index},
			)
	}

	# Set settings with key and seed instances
	
	for key in keys:


		settings[key] = {}

		# Set seed and key values
		values = dict(zip(values,keys[key]))

		# Get paths
		pwd = deepcopy(hyperparameters['sys']['pwd'])
		cwd = deepcopy(hyperparameters['sys']['cwd'])
		paths = deepcopy(hyperparameters['sys']['path'])

		# Set settings
		settings[key]['seed'] = seed

		settings[key]['boolean'] = {attr: (
				(hyperparameters['boolean'].get(attr,False)) 
				# (attr not in ['train'] or not hyperparameters['boolean'].get('load',False))
				)
				for attr in hyperparameters['boolean']}

		settings[key]['hyperparameters'] = {}
		settings[key]['object'] = {}
		settings[key]['logger'] = {}


		# Set hyperparameters updates with key/instance dependent settings
		settings[key]['hyperparameters'] = deepcopy(hyperparameters)

		updates = {}		

		updates.update({
			'model':{
				'system':{
					'key':key,
					'seed':key,
					'timestamp':timestamp,
					},
				},
			'sys':{
				'path': {
					attr: {
						path: join(
							key,
							split(settings[key]['hyperparameters']['sys']['path'][attr][path],directory=True),
							split(settings[key]['hyperparameters']['sys']['path'][attr][path],file=True),
							ext=split(settings[key]['hyperparameters']['sys']['path'][attr][path],ext=True),
							root=join(cwd))
						for path in settings[key]['hyperparameters']['sys']['path'][attr]
						}
					for attr in settings[key]['hyperparameters']['sys']['path']
					},
				}
			})

		for branch,leaf in zip(seedlings,values['seed']):
			grow(updates,branch,leaf)


		# Update hyperparameters
		setter(updates,values['permutations'],delimiter=delim,copy=True)
		updater(settings[key]['hyperparameters'],updates,copy=True)
		check(settings[key]['hyperparameters'])

		
		# Copy files		
		sources = {}
		destinations = {}

		func = lambda key,iterable,elements: iterable.get(key,elements[key])

		# Set sources and destinations of files
		attrs = paths
		for attr in attrs:
			sources[attr] = {}
			destinations[attr] = {}
			for path in paths[attr]:
				sources[attr][path] = join(split(paths[attr][path],directory=True),split(paths[attr][path],file=1),ext=split(paths[attr][path],ext=1),root=join(pwd))
				destinations[attr][path] = join(split(paths[attr][path],directory=True),split(paths[attr][path],file=1),ext=split(paths[attr][path],ext=1),root=join(cwd,key))


		# Dump files
		attrs = ['config']
		for attr in attrs:
			for path in paths[attr]:

				if path in ['settings']:
					data = deepcopy(settings[key]['hyperparameters'])
				else:
					data = settings[key]['hyperparameters'].get(path,{})
				try:
					source = load(sources[attr][path])
					destination = destinations[attr][path]

					updater(data,source,func=func)
					dump(data,destination)
				except:
					source = sources[attr][path]
					destination = destinations[attr][path]

					try:					
						copy(source,destination)
					except Exception as e:
						raise e

		# Set object
		settings[key]['object'] = None

		# Set logger
		settings[key]['logger'] = logconfig(__name__,
			conf=settings[key]['hyperparameters']['sys']['path']['config']['logger'],
			**{'handler_file.formatter.file.args':settings[key]['hyperparameters']['sys']['path']['data']['log'],}
			)


	return settings


def run(hyperparameters):
	'''
	Run simulations
	Args:
		hyperparameters (dict): hyperparameters
	'''		

	settings = setup(hyperparameters)

	for key in settings:		

		if not any(settings[key]['boolean'].get(attr) for attr in ['load','dump','train','plot.obj']):
			continue		

		hyperparameters = settings[key]['hyperparameters']

		cls = load(hyperparameters['class'])

		obj = cls(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)


		# print(obj.__layers__(obj.parameters,'variables').round(3))
		# continue


		if settings[key]['boolean'].get('load'):
			obj.load()

		settings[key]['object'] = obj

		if settings[key]['boolean'].get('train'):

			parameters = obj.parameters
			hyperparameters = hyperparameters['optimize']

			func = obj.__func__
			callback = obj.__callback__

			optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparameters)

			parameters = optimizer(parameters)
		
		if settings[key]['boolean'].get('dump'):	
			obj.dump()
		
		if settings[key]['boolean'].get('plot.obj'):
			obj.plot()

	if any(settings[key]['boolean'].get('plot') for key in settings):
		hyperparameters = {key: settings[key]['hyperparameters'] for key in settings if settings[key]['boolean'].get('plot')}
		plotter(hyperparameters)		

	return