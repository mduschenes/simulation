#!/usr/bin/env python

# Import python modules
import os,sys,itertools

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import copy,seeder,delim,union,is_equal,funcpath,argparser
from src.iterables import getter,setter,permuter,search
from src.io import load,dump,join,split,environ
from src.call import launch,call,command

def allow(value,values,settings):
	'''
	Check if value is allowed as per values
	Args:
		value (dict): Dictionary of possible value of the form {attr: value (dict)}
		values (dict): Dictionary of all values of the form {attr: values}
		settings (dict,str): settings		
	Returns:
		boolean (bool) : Boolean if value is allowed
	'''

	# TODO: Allow certain permutations of values

	boolean = True

	return boolean


def permute(settings):
	'''
	Get permutations of settings
	Args:
		settings (dict,str): settings
	Returns:
		permutations (iterable[dict]): Permutations of settings
	'''
	
	permutations = settings['permutations'].get('permutations')
	
	groups = settings['permutations'].get('groups')
	filters = load(settings['permutations'].get('filters'),default=settings['permutations'].get('filters'))
	func = load(settings['permutations'].get('func'),default=settings['permutations'].get('func'))
	
	permutations = permuter(permutations,groups=groups,filters=filters,func=func)

	return permutations


def spawn(settings):
	'''
	Get seeds for number of splits/seedings, for all nested settings branches that involve a seed
	Args:
		settings (dict,str): settings
	Returns:
		seed (int): Seed
		seeds (iterable[dict]): All permutation of seed instances
		seedlings (dict[iterable]): All possible sets of seeds
	'''

	# Get seeds for number of splits/seedings, for all nested settings branches that involve a seed
	seed = settings['seed'].get('seed')
	size = settings['seed'].get('size')
	groups = settings['seed'].get('groups')

	# Find keys of seeds in settings
	items = ['seed']
	types = (list,dict,)
	exclude = ['seed','seed.seed','system.seed',
		*[delim.join(['permutations','permutations',*attr.split(delim)]) for attr in getter(settings,'permutations.permutations',delimiter=delim)],
		]
	seedlings = search(settings,items=items,returns=True,types=types)

	seedlings = {delim.join([*index,element]):obj for index,shape,item in seedlings if all(isinstance(i,str) for i in index) for element,obj in zip(items,item)}
	seedlings = [seedling for seedling in seedlings if (seedling not in exclude) and (seedlings[seedling] is None)]
	count = max(1,len(seedlings))

	if isinstance(size,int):
		size = [size]*count
		groups = [[i for i in seedlings]]
	elif isinstance(size,dict):
		size = [size.get(seedling,1) for seedling in seedlings]
		groups = groups		
	elif size is None:
		size = [1]*count
		groups = [[i for i in seedlings]]	
	elif len(size) == 1:
		size = [*size]*count
		groups = groups
	elif len(size) == count:
		size = [i for i in size]
		groups = groups
	else:
		size = [1]*count
		groups = [[i for i in seedlings]]

	shape = (*size,)
	size = sum(size)

	if size:
		seeds = seeder(seed=seed,size=size,data=True)
		seedlings = {seedling: seeds[sum(shape[:i]):sum(shape[:i+1])] for i,seedling in enumerate(seedlings)}
		seeds = permuter(seedlings,groups=groups)
	else:
		seeds = seeder(seed=seed,data=True)
		seedlings = {seedling: seeds[i] for i,seedling in enumerate(seedlings)}
		seeds = [{}]

	return seed,seeds,seedlings


def formatter(key,shape,value,settings,default):
	'''
	Format key for permutation instances
	Args:
		key (int,iterable[int]): Index of instance
		shape (int,iterable[int]): Shape of permutations
		value (dict,iterable[dict]): Permutation instance
		settings (dict,str): Settings
		default (str): Default string for key
	Returns:
		key (str): Formatted key string for permutation instance
	'''
	key = key if not isinstance(key,int) else [key]
	shape = shape if not isinstance(shape,int) else [shape]

	if any(i>1 for i in shape):
		string = delim.join([
			*([str(default)] if default is not None else []),
			*[str(key[i]) for i in range(len(key)) if shape[i]>1]
			])
	else:
		string = default

	return string


def iterate(settings,index=None,wrapper=None):
	'''
	Iterate settings
	Args:
		settings (dict,str): settings
		index (int): settings index
		wrapper (callable): settings wrapper
	Returns:
		settings (dict): settings
	Yields:
		key (str): settings key
		setting (dict): settings values
	'''

	i = -1

	# Get permutations of settings
	permutations = permute(settings)

	for instance,permutation in enumerate(permutations):

		if not allow(permutation,permutations,settings):
			continue

		# Update settings with permutation
		setting = copy(settings)
		boolean = lambda attr,permutation: attr.split(delim)[0] in ['seed']
		setter(setting,{attr: permutation[attr] for attr in permutation if boolean(attr,permutation)},delimiter=delim,copy=True)

		# Get seeds for number of splits/seedings, for all nested settings branches that involve a seed
		seed,seeds,seedlings = spawn(setting)

		# Get shape and default key of permutations
		shape = (len(permutations),len(seeds))
		default = getter(setting,'system.key',delimiter=delim)

		# Get all allowed enumerated keys and seeds for permutations and seedlings of settings
		for number,seedling in enumerate(seeds):

			i += 1

			if index is not None and i != index:
				continue

			key = (instance,number)
			value = (permutation,seedling)

			key = formatter(key,shape,value,setting,default)

			options = {
					'system.key':key,
					'system.instance':number if key is not None else None,
					'system.instances':{seedling: number for seedling in seedlings},
					'system.seed':seed,
					'system.seeding':seed
				}

			value = {
				**permutation,
				**seedling,
				**options,
			}

			setting = copy(setting)

			setter(setting,value,delimiter=delim,copy=True)

			if wrapper is not None:
				setting = wrapper(setting)

			if index is not None and i == index:
				yield key,setting

			yield key,setting


def setup(settings):
	'''
	Setup settings
	Args:
		settings (dict,str): settings
	Returns:
		jobs (dict): Job submission dictionary
	'''

	# Load settings
	path = settings if isinstance(settings,str) else None
	default = {}
	if settings is None:
		settings = default
	elif isinstance(settings,str):
		settings = load(path,default=default)
	else:
		settings = default

	# Iterate settings with permutations, seeds and options
	settings = {key:setting for key,setting in iterate(settings)}

	# Set job
	jobs = {}

	names = union(*(settings[key]['jobs'] for key in settings),sort=True)

	for name in names:
		
		jobs[name] = {}

		for key in settings:
			
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
						**{job[attr][string]: None
							for string in job[attr]},
						**{job[attr][string]: settings[key] if job.get('local') else path
							for string in ['settings'] if string in job[attr]},
						**{job[attr][string]: settings[key].get(string,{}) 
							for string in ['plot','process'] if string in job[attr]},
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
				else:
					jobs[name][attr] = value

			attrs = {'name':name}
			jobs[name].update({attr: attrs[attr] for attr in attrs if attr not in jobs[name]})

	jobs = {name: jobs[name] for name in jobs if len(jobs[name])}			

	return jobs



def run(settings,device=None,job=None,path=None,env=None,execute=False,verbose=None):
	'''
	Run simulations
	Args:
		settings (dict,str): settings for simulations
		device (str): Name of device to submit to
		job (str): Name of job to run simulations
		path (str): Path where to submit job
		env (str): Name of environment to run simulations
		execute (bool,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity		
	Returns:
		results (iterable[str]): Return of commands for each job		
	'''
	
	device = None if device is None else device
	job = 'job.slurm' if job is None else job
	settings = join(settings,abspath=True)
	path = join(path,abspath=True)	
	env = environ().get('CONDA_PREFIX',environ().get('VIRTUAL_ENV')) if env is None else env
	execute = True if execute is None else execute
	verbose = True if verbose is None else verbose

	cmd = funcpath(run)

	if device is not None:

		args = {
			"JOB_ARGS":settings,
			"JOB_ENV":env,
			"JOB_CMD":cmd,
			}
		kwargs = {}
		
		path = path
		exe = join(split(cmd,directory=True),job)
		flags = []
		cmd = []
		options = []
		env = []

		process = None
		processes = None

		args,env = command(args,kwargs,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

		results = call(args,path=path,env=env,execute=execute,verbose=verbose)

	else:

		jobs = setup(settings)

		results = launch(jobs,execute=execute,verbose=verbose)

	return results


def main(*args,**kwargs):

	run(*args,**kwargs)

	return

if __name__ == '__main__':

	arguments = {
		'--settings':{
			'help':'Settings',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--device':{
			'help':'Device',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--job':{
			'help':'Job',
			'type':str,
			'default':None,
			'nargs':'?'
		},	
		'--path':{
			'help':'Path',
			'type':str,
			'default':None,
			'nargs':'?'
		},		
		'--env':{
			'help':'Environment',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--dry-run':{
			'help':'Execute',
			'action':'store_true'
		},
		'--quiet':{
			'help':'Verbose',
			'action':'store_true'
		},										
		}		

	wrappers = {
		'execute': lambda kwarg,wrappers,kwargs: not kwargs.pop('dry-run',True),
		'verbose': lambda kwarg,wrappers,kwargs: not kwargs.pop('quiet',True),
		}

	args = argparser(arguments,wrappers)

	main(*args,**args)