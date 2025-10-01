#!/usr/bin/env python

# Import python modules
import os,sys,itertools

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import copy,seeder,prod,delim,union,is_equal,funcpath,argparser
from src.iterables import Dict,getter,setter,permuter,search
from src.io import load,dump,join,split,basename,dirname,environ
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
	
	attr = 'permutations'

	if settings.get(attr) is None:
		permutations = [{}]
		return permutations

	permutations = settings.get(attr,{}).get('permutations')
	
	groups = settings.get(attr,{}).get('groups')
	filters = load(settings.get(attr,{}).get('filters'),default=settings.get(attr,{}).get('filters'))
	func = load(settings.get(attr,{}).get('func'),default=settings.get(attr,{}).get('func'))
	
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

	attr = 'seed'

	if settings.get(attr) is None:
		seed = None
		seeds = []
		seedlings = {}
		return seed,seeds,seedlings

	seed = settings.get(attr,{}).get('seed')
	size = settings.get(attr,{}).get('size')
	groups = settings.get(attr,{}).get('groups')
	attributes = settings.get(attr,{}).get('attributes')

	# Find keys of seeds in settings
	items = ['seed']
	types = (list,dict,)
	exclude = ['seed','seed.seed','system.seed','callback.settings.seed','callback.settings.seed.seed',
		*[delim.join(['permutations','permutations',*attr.split(delim)]) for attr in getter(settings,'permutations.permutations',delimiter=delim,default={})],
		*(attributes if attributes is not None else [])
		]
	seedlings = search(settings,items=items,returns=True,types=types)

	seedlings = {delim.join([*index,element]):obj for index,shape,item in seedlings if all(isinstance(i,str) for i in index) for element,obj in zip(items,item)}
	seedlings = [seedling for seedling in seedlings if (seedling not in exclude) and (seedlings[seedling] is None)]

	seedlings = [seedling for seedling in seedlings]
	count = max(1,len(seedlings))
	groups = [[seedling for seedling in seedlings]]

	if size is None:
		size = [count,1]
	elif isinstance(size,(int,float)):
		size = [count,int(size)]
	elif isinstance(size,list):
		size = [count,*size] if len(size)>1 else [count,1,*size]
	elif isinstance(size,dict):
		seedlings = [seedling for seedling in seedlings if seedling in size]
		count = max(1,len(seedlings))
		groups = [[seedling for seedling in seedlings]]
		size = [count,1,max((size[seedling] if isinstance(size[seedling],int) else prod(size[seedling]) for seedling in seedlings),default=1)]
	else:
		size = [count,1]

	if size:
		seeds = seeder(seed=seed,size=size,data=True)
		seedlings = {seedling:seed for seedling,seed in zip(seedlings,seeds)}
		seeds = permuter(seedlings,groups=groups)
	else:
		seeds = seeder(seed=seed,data=True)
		seedlings = {seedling:seed for seedling,seed in zip(seedlings,seeds)}
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
		index (int,bool): settings index
		wrapper (callable): settings wrapper
	Yields:
		key (str): settings key
		setting (dict): settings values
		size (int): settings size
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

		# Get shape, size and default key of permutations
		shape = (len(permutations),len(seeds))
		size = prod(shape)
		default = getter(setting,'system.key',delimiter=delim)

		# Get all allowed enumerated keys and seeds for permutations and seedlings of settings
		for number,seedling in enumerate(seeds):

			i += 1

			if (index is not None) and (index is not False) and (index is not True) and (i != index):
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

			if (index is None) or (index is False) or ((index is True) or (i == index)):
				yield key,setting,size

			if (index is True) or ((index is not None) and (index is not False) and (i == index)):
				break

		if (index is True) or ((index is not None) and (index is not False) and (i == index)):
			break


def setup(settings,*args,index=None,device=None,job=None,path=None,env=None,execute=None,verbose=None,**kwargs):
	'''
	Setup settings
	Args:
		settings (dict,str): settings
		index (int,bool): settings index
		device (str): settings device
		job (str): settings job
		path (str): settings path
		env (int): settings environment
		execute (bool,int): settings execution
		verbose (int,str,bool): settings verbosity
		args (iterable): settings positional arguments
		kwargs (dict): settings keyword arguments
	Yields:
		settings (dict): settings
	'''

	default = {}
	wrapper = Dict
	defaults = Dict(
		boolean=dict(call=None,optimize=None,load=None,dump=None),
		cls=dict(module=None,model=None,state=None,label=None,callback=None),
		module=dict(),model=dict(),state=dict(),label=dict(),callback=dict(),
		optimize=dict(),options=dict(),seed=dict(),system=dict(),
		)

	if settings is None:
		settings = default
	elif isinstance(settings,str):
		settings = load(settings,default=default,wrapper=wrapper)

	setter(settings,kwargs,delimiter=delim,default=True)
	setter(settings,defaults,delimiter=delim,default=False)

	if index is None:
		yield settings
	else:
		for key,settings,size in iterate(settings,index=index,wrapper=wrapper):
			yield settings

def init(settings):
	'''
	initialize settings
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
	names = []
	attribute = 'jobs'
	keyword = 'local'
	local = isinstance(settings.get(attribute),dict) and all(isinstance(settings[attribute].get(job),dict) and settings[attribute][job].get(keyword) for job in settings[attribute])
	if local:
		options = dict(index=None)
		settings = {key:setting for key,setting,size in iterate(settings,**options)}
		names = union(*(settings[key][attribute] for key in settings),sort=True)
	else:
		options = dict(index=True)
		for key,setting,size in iterate(settings,**options):
			settings = {key:setting for key in range(size)}
			names = list(setting[attribute]) if setting.get(attribute) else []
	
	# Set job
	jobs = {}

	for name in names:
		
		jobs[name] = {}

		for key in settings:
			
			if not local and jobs[name]:
				break

			job = settings[key][attribute].get(name)

			if job is None:
				continue

			for attr in job:

				if attr in ['jobs']:
					value = job[attr]
				elif attr in ['args']:
					value = job[attr]
				elif attr in ['paths']:
					value = {
						**{variable: {data:None}
							for string in job[attr] for variable,data in (job[attr][string] if isinstance(job[attr][string],dict) else {job[attr][string]:job[attr][string]}).items() if data is not None},
						**{variable if variable is not None else basename(path):{data if data is not None else basename(path): settings[key] if job.get(keyword) else path}
							for string in ['settings'] if string in job[attr] for variable,data in (job[attr][string] if isinstance(job[attr][string],dict) else {job[attr][string]:job[attr][string]}).items()},
						**{variable:{data:settings[key].get(string,{}) if settings[key].get(string) else None}
							for string in ['plot','process'] if string in job[attr] for variable,data in (job[attr][string] if isinstance(job[attr][string],dict) else {job[attr][string]:job[attr][string]}).items()},
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
						if local:
							jobs[name][attr][key] = value
						else:
							jobs[name][attr] = {key:value for key in settings}
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

	jobs = {name:jobs[name] for name in jobs if jobs[name]}

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

		jobs = init(settings)

		results = launch(jobs,execute=execute,verbose=verbose)

	return results


def argparse(*args,**kwargs):
	'''
	Parse arguments
	Args:
		args (iterable): positional arguments
		kwargs (dict): keyword arguments
	Returns:
		args (iterable): arguments
	'''

	arguments = {
		'--settings':{
			'help':'Settings',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--index':{
			'help':'Index',
			'type':int,
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

	return args


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