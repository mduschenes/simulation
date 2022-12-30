#!/usr/bin/env python

# Import python modules
import os,sys,itertools
from copy import deepcopy

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import PRNGKey,delim,union,is_equal
from src.iterables import getter,setter,permuter,clearer,leaves
from src.io import load,dump,join,split
from src.call import launch

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

	# TODO: Allow certain permutations of values, and retain index for folders

	boolean = True

	return boolean

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
				boolean &= is_equal(value[attr][key],value[attr])
			elif isinstance(subindex,dict):
				boolean &= all(is_equal(value[attr][key],subindex[key]) for key in subindex)

	return boolean

def setup(settings):
	'''
	Setup settings
	Args:
		settings (dict,str): settings
	Returns:
		jobs (dict): Job submission dictionary
	'''

	# Default settings
	path = 'config/settings.json'

	# Load default settings
	default = {}
	func = False
	if settings is None:
		defaults = path
		settings = default
	elif isinstance(settings,str):
		defaults = settings
		settings = load(settings,default=default)

	setter(settings,load(path,default=default),func=func)

	# Load default hyperparameters
	default = {}
	hyperparameters = settings.get('hyperparameters',defaults)

	if hyperparameters is None:
		hyperparameters = default
	elif isinstance(hyperparameters,str):
		hyperparameters = load(hyperparameters,default=default)

	default = {}
	func = False
	setter(hyperparameters,load(path,default=default),func=func)

	# Get permutations of hyperparameters
	permutations = settings['permutations']['permutations']
	groups = settings['permutations']['groups']
	permutations = permuter(permutations,groups=groups)

	# Get seeds for number of splits/seedings, for all nested hyperparameters leaves that involve a seed
	seed = settings['seed']['seed']
	size = settings['seed']['size']
	reset = settings['seed']['reset']

	seed = seed if seed is not None else None
	size = size if size is not None else 1
	reset = reset if reset is not None else None


	# Find keys of seeds in hyperparameters
	key = 'seed'
	exclude = ['seed.seed','system.seed']
	seedlings = [delim.join(branch[0]) for branch in leaves(hyperparameters,key,returns='both') if not any(delim.join(branch[0][:len(e.split(delim))]) == e for e in exclude) and branch[1] is None]

	count = len(seedlings)
	
	shape = (size,count,-1)
	size *= count

	seeds = PRNGKey(seed=seed,size=size,reset=reset).reshape(shape).tolist()
	seeds = [dict(zip(seedlings,seed)) for seed in seeds]

	other = [{'system.key':None,'system.seed':seed}]

	# Get all allowed enumerated keys and seeds for permutations and seedlings of hyperparameters
	values = {'permutations':permutations,'seed':seeds,'other':other}
	index = {attr: hyperparameters.get(attr,{}).get('index') for attr in values}
	# formatter = lambda instance,value,values,default: '%d'%(instance)	
	formatter = lambda instance,value,values,default: ((delim.join([
		*([default] if default is not None else []),
		*['%d'%(v[0]) for k,v in zip(values,value) if len(values[k])>1]])) 
		if any(len(values[k])>1 for k in values) else default)

	keys = {}

	for instance,value in enumerate(value for value in itertools.product(*(zip(range(len(values[attr])),values[attr]) for attr in values))):

		if allowed(
			{attr: index[attr] for attr in index},
			{attr: {k:v[1] for k,v in zip(values,value)}[attr] for attr in index},
			{attr: values[attr] for attr in index},
			):

			key = formatter(instance,value,values,getter(hyperparameters,'system.key',delimiter=delim))
			value = [v[1] for v in value]
			keys[key] = {}
			for setting in value:
				for attr in setting:

					keys[key][attr] = setting[attr]

					if attr in ['system.key']:
						keys[key][attr] = key

	# Set settings with key and seed instances
	old = [attr for attr in settings]
	new = {key: deepcopy(settings) for key in keys}
	clearer(settings,new,old)

	# Set hyperparameters with key and seed instances
	old = [attr for attr in hyperparameters]
	new = {key: deepcopy(hyperparameters) for key in keys}
	clearer(hyperparameters,new,old)	

	for key in keys:
		setter(settings[key],keys[key],delimiter=delim,copy=True)
		setter(hyperparameters[key],keys[key],delimiter=delim,copy=True)

	# Set job
	jobs = {}

	names = union(*(settings[key]['jobs'] for key in keys),sort=True)

	for name in names:

		jobs[name] = {}

		for key in keys:
			
			job = settings[key]['jobs'].get(name)

			if job is None:
				continue

			for attr in job:

				if attr in ['jobs']:
					value = job[attr]
				elif attr in ['args']:
					value = job[attr]
				elif attr in ['paths']:
					value = {
						**{job[attr][path]: None
							for path in job[attr]},
						**{job[attr][path]: hyperparameters[key] 
							for path in ['settings']},
						**{job[attr][path]: hyperparameters[key].get(path,{}) 
							for path in ['plot','process']},
						}
				elif attr in ['patterns']:
					value = job[attr]
				elif attr in ['pwd','cwd']:
					value = job[attr]
				else:
					value = job[attr]

				if name in ['job']:
					if attr not in jobs[name]:
						jobs[name][attr] = {}
					if attr in ['jobs','args','paths','patterns','pwd','cwd']:
						jobs[name][attr][key] = value
					else:
						jobs[name][attr] = value
				elif name in ['preprocess']:
					pass
				elif name in ['postprocess']:
					jobs[name][attr] = value

	
	return jobs



def run(settings):
	'''
	Run simulations
	Args:
		settings (dict,str): settings
	Returns:
		results (iterable[str]): Return of commands for each job		
	'''		

	jobs = setup(settings)

	results = launch(jobs)

	return results