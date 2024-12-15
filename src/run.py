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
	if settings is None:
		defaults = path
		settings = default
	elif isinstance(settings,str):
		defaults = settings
		settings = load(settings,default=default)
	else:
		settings = default

	setter(settings,load(path,default=default),default=False)

	# Load default hyperparameters
	default = copy(settings)
	hyperparameters = settings['hyperparameters'] if settings.get('hyperparameters') is not None else default

	if hyperparameters is None:
		hyperparameters = default
	elif isinstance(hyperparameters,str):
		hyperparameters = load(hyperparameters,default=default)
	else:
		hyperparameters = default

	default = {}
	setter(hyperparameters,load(path,default=default),default=False)

	# Get permutations of hyperparameters
	permutations = settings['permutations'].get('permutations')
	groups = settings['permutations'].get('groups')
	filters = load(settings['permutations'].get('filters'),default=settings['permutations'].get('filters'))
	func = load(settings['permutations'].get('func'),default=settings['permutations'].get('func'))
	permutations = permuter(permutations,groups=groups,filters=filters,func=func)

	# Get seeds for number of splits/seedings, for all nested hyperparameters branches that involve a seed
	seed = settings['seed'].get('seed')
	size = settings['seed'].get('size')
	groups = settings['seed'].get('groups')

	# Find keys of seeds in hyperparameters
	items = ['seed']
	types = (list,dict,)
	exclude = ['seed','seed.seed','system.seed',
	*[delim.join(['permutations','permutations',*attr.split(delim)]) for permutation in permutations for attr in permutation 
	if ((attr.split(delim)[-1] == 'seed' and permutation[attr] is not None) or 
		True
	    # (permutation[attr] is not None and isinstance(permutation[attr],dict) and 'seed' in permutation[attr])
	    )]
	]
	seedlings = search(hyperparameters,items=items,returns=True,types=types)

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
		seeds = [{}]

	other = [{'system.key':None,'system.instance':None,'system.instances':None,'system.seed':seed,'system.seeding':seed}]

	# Get all allowed enumerated keys and seeds for permutations and seedlings of hyperparameters
	values = {'permutations':permutations,'seed':seeds,'other':other}

	index = {attr: hyperparameters.get(attr,{}).get('index') for attr in values}

	def formatter(instance,value,values,default):
		if any(len(values[k])>1 for k in values):
			string = delim.join([
				*([default] if default is not None else []),
				*['%d'%(v[0]) for k,v in zip(values,value) if len(values[k])>1]])
		else:
			string = default

		return string

	keys = {}

	for instance,value in enumerate(value for value in itertools.product(*(
		zip(range(len(values[attr])),values[attr]) for attr in values)
		)):

		if not allowed(
			{attr: index[attr] for attr in index},
			{attr: {k:v[1] for k,v in zip(values,value)}[attr] for attr in index},
			{attr: values[attr] for attr in index},
			):
			continue

		key = formatter(instance,value,values,default=getter(hyperparameters,'system.key',delimiter=delim))
		indices = [v[0] for v in value]
		value = [v[1] for v in value]
		keys[key] = {}
		for setting in value:
			for attr in setting:

				keys[key][attr] = setting[attr]

				if attr in ['system.key']:
					keys[key][attr] = key
				elif attr in ['system.instance']:
					keys[key][attr] = int(key.split(delim)[-1]) if key is not None else None					
				elif attr in ['system.instances']:
					keys[key][attr] = {seedling: seedlings[seedling].index(keys[key][seedling]) for seedling in seedlings}


	# Set settings with key and seed instances
	settings = {key: copy(settings) for key in keys}

	# Set hyperparameters with key and seed instances
	hyperparameters = {key: copy(hyperparameters) for key in keys}

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
							for path in ['settings'] if path in job[attr]},
						**{job[attr][path]: hyperparameters[key].get(path,{}) 
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



def run(settings,device=None,job=None,cmd=None,path=None,env=None,execute=None,verbose=None):
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
	job = 'submit.slurm' if job is None else job
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

		results = launch(jobs)

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