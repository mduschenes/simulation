#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime
from copy import deepcopy as deepcopy

import matplotlib
import matplotlib.pyplot as plt

# Logging
import logging

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import logconfig,PRNGKey,delim,partial
from src.dictionary import updater,getter,setter,permuter,leaves,grow
from src.io import load,dump,join,split
from src.call import copy
from src.call import submit
from src.process import process
from src.train import train

def plotter(hyperparameters):
	'''
	Plot models
	Args:
		hyperparameters (dict): Hyperparameters
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
		hyperparameters (dict,str): Hyperparameters
	'''

	# Check hyperparameters
	default = {}
	if hyperparameters is None:
		hyperparameters = default
	elif isinstance(hyperparameters,str):
		hyperparameters = load(hyperparameters,default=default)

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
		hyperparameters (dict,str): Hyperparameters
	'''

	# Get settings
	settings = {}	

	# Load default hyperparameters
	default = {}
	if hyperparameters is None:
		hyperparameters = default
	elif isinstance(hyperparameters,str):
		hyperparameters = load(hyperparameters,default=default)

	path = 'config/settings.json'
	default = {}
	func = lambda key,iterable,elements: iterable.get(key,elements[key])
	updater(hyperparameters,load(path,default=default),func=func)

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

	seeds = PRNGKey(seed=seed,size=size,reset=reset).reshape(shape).tolist()


	# Get all allowed enumerated keys and seeds for permutations and seedlings of hyperparameters
	values = {'permutations':permutations,'seed':seeds}
	index = {attr: hyperparameters[attr]['index'] for attr in values}
	# formatter = lambda instance,value,values: '%d'%(instance)	
	formatter = lambda instance,value,values: ('.'.join(['%d'%(v[0]) for k,v in zip(values,value) if len(values[k])>1]))
	keys = {
		formatter(instance,value,values):[v[1] for v in value]
		for instance,value in enumerate(itertools.product(*(zip(range(len(values[attr])),values[attr]) for attr in values)))
		if allowed(
			{attr: index[attr] for attr in index},
			{attr: dict(zip(values,value))[attr] for attr in index},
			{attr: values[attr] for attr in index},
			)
	}

	# Set settings with key and seed instances
	
	for key in keys:

		# Set attributes
		attrs = ['seed','hyperparameters','boolean','object','job']
		settings[key] = {}
		for attr in attrs:
			settings[key][attr] = {}

		# Get paths
		pwd = deepcopy(hyperparameters['sys']['pwd'])
		cwd = deepcopy(hyperparameters['sys']['cwd'])
		paths = deepcopy(hyperparameters['sys']['path'])

		# Set seed and key values
		values = dict(zip(values,keys[key]))

		# Set seed
		settings[key]['seed'] = seed

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
							# split(settings[key]['hyperparameters']['sys']['path'][attr][path],directory=True),
							split(settings[key]['hyperparameters']['sys']['path'][attr][path],file=True),
							ext=split(settings[key]['hyperparameters']['sys']['path'][attr][path],ext=True),
							root=join(cwd))
						for path in settings[key]['hyperparameters']['sys']['path'][attr]
						}
					for attr in settings[key]['hyperparameters']['sys']['path']
				 	},
				},
			'job':{
				'device': settings[key]['hyperparameters']['job'].pop('device'),
				'args': [
					join(split(settings[key]['hyperparameters']['job'].pop('job'),file_ext=True),abspath=True,root=join(cwd,key)),
					join(settings[key]['hyperparameters']['job'].pop('cmd'),abspath=True),
					join(cwd,key),
					*(join(arg,abspath=True,root=join(cwd,key)) for arg in settings[key]['hyperparameters']['job'].pop('args'))
					],
				'path': settings[key]['hyperparameters']['job'].pop('path'),
				'exe': settings[key]['hyperparameters']['job'].pop('exe'),
				}				
			})

		for branch,leaf in zip(seedlings,values['seed']):
			grow(updates,branch,leaf)

		# Update hyperparameters
		setter(updates,values['permutations'],delimiter=delim,copy=True)
		updater(settings[key]['hyperparameters'],updates,copy=True)
		check(settings[key]['hyperparameters'])

		# Set booleans
		settings[key]['boolean'] = {attr: (
				(hyperparameters['boolean'].get(attr,False)) 
				# (attr not in ['train'] or not hyperparameters['boolean'].get('load',False))
				)
				for attr in hyperparameters['boolean']}

		# Set object
		settings[key]['object'] = None

		# Set job
		settings[key]['job'] = settings[key]['hyperparameters']['job']

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
				destinations[attr][path] = join(split(paths[attr][path],file=1),ext=split(paths[attr][path],ext=1),root=join(cwd,key))

		# Dump files
		attrs = ['config']
		for attr in attrs:
			for path in paths[attr]:
				if path in ['settings']:
					data = settings[key]['hyperparameters']
					source = data
					destination = destinations[attr][path]
					dump(source,destination)
				elif path in ['process','plot']:
					data = load(sources[attr][path])
					source = deepcopy(settings[key]['hyperparameters'].get(path,{}))
					destination = destinations[attr][path]
					updater(source,data,func=func)
					dump(source,destination)					
				else:
					source = sources[attr][path]
					destination = destinations[attr][path]
					copy(source,destination)

	return settings



def run(hyperparameters):
	'''
	Run simulations
	Args:
		hyperparameters (dict,str): hyperparameters
	'''		

	settings = setup(hyperparameters)

	for key in settings:		

		if not any(settings[key]['boolean'].get(attr) for attr in ['load','dump','train','plot.obj']):
			continue
		
		job = settings[key]['job']
		settings[key]['object'] = submit(**job)


	if any(settings[key]['boolean'].get('plot') for key in settings):
		hyperparameters = {key: settings[key]['hyperparameters'] for key in settings if settings[key]['boolean'].get('plot')}
		plotter(hyperparameters)		

	return