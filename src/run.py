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
	if settings is None:
		defaults = path
		settings = default
	elif isinstance(settings,str):
		defaults = settings
		settings = load(settings,default=default)
	else:
		settings = default

	setter(settings,load(path,default=default),default=False)


	# Get permutations of settings
	permutations = permute(settings)

	keys = {}

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
		for index,seedling in enumerate(seeds):

			key = (instance,index)
			value = (permutation,seedling)

			key = formatter(key,shape,value,setting,default)

			options = {
				'system.key':key,
				'system.instance':index if key is not None else None,
				'system.instances':{seedling: index for seedling in seedlings},
				'system.seed':seed,
				'system.seeding':seed
				}

			keys[key] = {}

			for value in [permutation,seedling,options]:
				for attr in value:
					keys[key][attr] = value[attr]

	
	# Set settings with key and seed instances
	settings = {key: copy(settings) for key in keys}

	for key in keys:
		setter(settings[key],keys[key],delimiter=delim,copy=True)

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
						**{job[attr][path]: settings[key] 
							for path in ['settings'] if path in job[attr]},
						**{job[attr][path]: settings[key].get(path,{}) 
							for path in ['plot','process'] if path in job[attr]},
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

	return jobs



def run(settings,device=None,job=None,cmd=None,path=None,env=None,execute=False,verbose=None):
	'''
	Run simulations
	Args:
		settings (dict,str): settings for simulations
		device (str): Name of device to submit to
		job (str): Name of job to run simulations
		cmd (str): Name of command to run simulations
		path (str): Path where to submit job
		env (str): Name of environment to run simulations
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity		
	Returns:
		results (iterable[str]): Return of commands for each job		
	'''
	
	device = None if device is None else device
	job = 'run.slurm' if job is None else job
	settings = join(settings,abspath=True)
	cmd = funcpath(run) if cmd is None else cmd
	path = join(path,abspath=True)	
	env = environ().get('CONDA_PREFIX',environ().get('VIRTUAL_ENV')) if env is None else env
	execute = True if execute is None else execute
	verbose = True if verbose is None else verbose

	if device is not None:

		args = {
			"JOB_SETTINGS":settings,
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
		'--cmd':{
			'help':'Command',
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