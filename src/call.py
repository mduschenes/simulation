#!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,inspect,traceback
from copy import deepcopy
import subprocess

# Logging
import logging

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import intersection,scalars
from src.system	 import Logger
from src.io import cd,mkdir,join,split,load,dump,exists,environ
from src.dictionary import updater

name = __name__
path = os.getcwd()
file = 'logging.conf'
conf = os.path.join(path,file)
file = None #'log.log'
logger = Logger(name,conf,file=file)


def command(args,kwargs=None,exe=None,flags=None,cmd=None,options=None,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Command of the form $> exe flags cmd args options
	Args:
		args (dict[str,str],dict[str,iterable[str],iterable[iterable[str]]],iterable[str]): Arguments to pass to command line {arg:value} or {arg:[value]} or [value]
		kwargs (dict): Keyword arguments for args
		exe (str,iterable[str]): Executable for args
		flags (str,iterable[str]): Flags for args
		cmd (str,iterable[str]): Command for args
		options (str,iterable[str]): Options for args
		env (dict[str,str]): Environmental variables for args
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	Returns:
		args (iterable[str]): Command arguments
		env (dict[str,str]): Environment for command
	'''

	if isinstance(args,dict):
		args = {arg: 
				([(str(args[arg]) if args[arg] is not None else '')] if isinstance(args[arg],scalars) else 
				 [((str(subarg) if subarg is not None else '') if isinstance(subarg,scalars) else 
				  ' '.join([(str(subsubarg) if subsubarg is not None else '') for subsubarg in subarg])) for subarg in args[arg]]) 
				for arg in args}
	else:
		args = {None:[((str(arg) if arg is not None else '') if isinstance(arg,scalars) else 
					  [(str(subarg) if subarg is not None else '') for subarg in arg]) for arg in args]}


	exe = [] if exe is None else [exe] if isinstance(exe,str) else [*exe]
	flags = [] if flags is None else [flags] if isinstance(flags,str) else [*flags]
	cmd = [] if cmd is None else [cmd] if isinstance(cmd,str) else [*cmd]
	options = [] if options is None else [options] if isinstance(options,str) else [*options]
	env = {} if env is None else {} if not isinstance(env,dict) else env
	kwargs = {} if kwargs is None else {} if not isinstance(kwargs,dict) else kwargs

	processes = -1 if processes is None else processes

	if device in ['pc']:
		exe = [*['./%s'%(e) for e in exe[:1]],*exe[1:]]
		flags = [*flags]
		cmd = [*cmd]
		options = [*[' '.join([subarg for subarg in args[arg]]) for arg in args],*options]		
		env = {
			**{
				'SLURM_JOB_NAME':kwargs.get('key'),
				'SLURM_JOB_ID':kwargs.get('key'),
				'SLURM_ARRAY_JOB_ID':kwargs.get('key'),
				'SLURM_ARRAY_TASK_ID':kwargs.get('index'),
				'SLURM_ARRAY_TASK_STEP':kwargs.get('step'),
			},
			**env			
		}

	elif device in ['slurm']:
		exe,flags,cmd,options,env = (
				['sbatch'],
				[*flags,'%s=%s'%('--export',','.join(['%s=%s'%(arg,' '.join([subarg for subarg in args[arg]])) for arg in args]))],
				['<'],
				[*exe,*cmd,*options],
				{}
				)

	else:
		exe = [*exe]
		flags = [*flags]
		cmd = [*cmd]
		options = [*[subarg for arg in args for subarg in args[arg]],*options]
		env = {
			**{
				'SLURM_JOB_NAME':kwargs.get('key'),				
				'SLURM_JOB_ID':kwargs.get('key'),
				'SLURM_ARRAY_JOB_ID':kwargs.get('key'),
				'SLURM_ARRAY_TASK_ID':kwargs.get('index'),
				'SLURM_ARRAY_TASK_STEP':kwargs.get('step'),
			},
			**env,			
		}

	if process in ['serial']:
		pass
	elif process in ['parallel']:
		exe,flags,cmd,options,env = ['parallel'],['--jobs',processes,*exe,*flags,*cmd,r'{}',':::'],[],[*options],{**env}
	elif process in ['array']:
		pass
	else:
		pass


	args = [*exe,*flags,*cmd,*options]

	env = {str(var): str(env[var]) if env[var] is not None else '' for var in env}

	return args,env


def call(*args,path=None,kwargs=None,exe=None,flags=None,cmd=None,options=None,env=None,wrapper=None,pause=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Submit call to command line of the form $> exe flags cmd args options
	Args:
		args (dict[str,str],dict[str,iterable[str],iterable[iterable[str]]],iterable[str]): Arguments to pass to command line {arg:value} or {arg:[value]} or [value], nested iterables are piped		
		path (str): Path to call from
		kwargs (dict): Keyword arguments for args		
		exe (str,iterable[str]): Executable for args
		flags (str,iterable[str]): Flags for args
		cmd (str,iterable[str]): Command for args
		options (str,iterable[str]): Options for args	
		env (dict[str,str]): Environmental variables for args		
		wrapper (callable): Wrapper for stdout with signature wrapper(stdout,stderr,returncode)		
		pause (int,str): Time to sleep after call
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	Returns:
		result (object): Return of commands
	'''

	def caller(args,inputs=None,env=None,device=None,verbose=None):

		def run(args,stdin=None,stdout=None,stderr=None,env=None):
			env = {**environ(),**env} if env is not None else None
			try:
				result = subprocess.Popen(args,stdin=stdin,stdout=stdout,stderr=stderr,env=env)
			except (OSError,FileNotFoundError) as exception:
				result = subprocess.Popen((),stdin=stdin,stdout=stdout,stderr=stderr,env=env)
				logger.log(verbose,exception)
				logger.log(verbose,args)
			return result


		def wrap(stdout,stderr,returncode):
			stdout = '\n'.join(stdout)
			stderr = '\n'.join(stderr)
			returncode =  returncode
			return stdout,stderr,returncode

		def parse(obj):
			obj = obj.strip().decode('utf-8')
			return obj


		stdin = None
		stdout = subprocess.PIPE
		stderr = subprocess.PIPE
		returncode = None

		inputs = [inputs]*len(args) if inputs in [None] else inputs

		for arg,input in zip(args,inputs):

			stdin = open(input,'r') if input is not None else stdin

			result = run(arg,stdin=stdin,stdout=stdout,stderr=stderr,env=env)

			if stdin is not None:
				stdin.close()

			stdin = result.stdout


		stdout,stderr,returncode = [],[],result.returncode
		
		for line in result.stdout:
			stdout.append(parse(line))			
			logger.log(verbose,stdout[-1])

		returncode = result.wait()

		for line in result.stderr:	
			stderr.append(parse(line))
			if returncode is not None:
				logger.log(verbose,stderr[-1])

		stdout,stderr,returncode = wrap(stdout,stderr,returncode)

		sleep(pause,execute=execute)

		return stdout,stderr,returncode

	def wrapper(stdout,stderr,returncode,wrapper=wrapper,env=None,device=None,verbose=None):
		try:
			result = wrapper(stdout,stderr,returncode)
		except:
			result = stdout

		return result

	def parser(*args,env=None,device=None,verbose=None):

		pipe = any(not isinstance(arg,scalars) for arg in args)

		if pipe:
			args = [[str(subarg) for subarg in arg] if not isinstance(arg,scalars) else [arg] for arg in args]
		else:
			args = [[str(arg) for arg in args]]

		cmd = ' | '.join([' '.join(arg) for arg in args])

		inputs = []
		symbols = ['<']

		for arg in args:

			for symbol in symbols:
				redirect = symbol in arg
				if redirect:
					break

			if redirect:
				index = arg.index(symbol)
				input = ' '.join(arg[index+1:])
				subarg = arg[:index]
			else:
				index = None
				input = None
				subarg = arg[:]

			args[args.index(arg)] = subarg
			inputs.append(input)

		return inputs,args,cmd,env


	args,env = command(args,kwargs,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	inputs,args,cmd,env = parser(*args,env=env,device=device,verbose=verbose)
	result = None

	msg = '%s : %s'%(path,cmd) if path is not None else cmd
	logger.log(verbose,msg)

	if execute > 0:
		with cd(path):
			result = wrapper(*caller(args,inputs=inputs,env=env,device=device,verbose=verbose),env=env,device=device,verbose=verbose)

	return result

def cp(source,destination,default=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Copy objects from source to destination
	Args:
		source (str): Path of source object
		destination (str): Path of destination object
		default (str): Default path of source object
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	'''
	if not exists(source):
		source = default
	
	assert exists(source), 'source %s does not exist'%(source)

	mkdir(destination)

	exe = ['cp']
	flags = ['-rf']
	cmd = [source,destination]
	options = []
	env = []
	args = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return


def rm(path,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Remove path
	Args:
		path (str): Path to remove
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	'''

	exe = ['rm']
	flags = ['-rf']
	cmd = [path]
	options = []
	env = []	
	args = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return



def echo(*args,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Echo arguments to command line
	Args:
		args (str,iterable[str],iterable[iterable[str]]): Arguments to pass to command line, nested iterables are piped
		destination (str): Path of destination object
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	'''

	exe = ['echo']
	flags = []
	cmd = args
	options = []
	env = []	
	args = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return


def sed(path,patterns,default=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	GNU sed replace patterns in path
	Args:
		path (str): Path of file
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		default (str): Default pattern value to sed
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	'''

	if path is None or patterns is None:
		return

	delimiters = [',','#','/','$']
	replacements = {}#{r'-':r'\-',r' ':r'\ ',r'#':r'\#'}

	for pattern in patterns:

		cmd = None
		
		result = search(path,pattern,execute=True,verbose=verbose)

		if result == -1:
			result = search(path,default,execute=True,verbose=verbose)
			if result == -1:					
				cmd = None
			else:
				cmd = '%di %s'%(result+1,patterns[pattern])			
		else:			
			for delimiter in delimiters:
				if all(delimiter not in string for string in (pattern,patterns[pattern])):
					cmd = 's%s%s%s%s%sg'%(delimiter,pattern,delimiter,patterns[pattern],delimiter)
					break

		if cmd is None:
			continue

		for replacement in replacements:
			cmd = cmd.replace(replacement,replacements[replacement])


		exe = ['sed']
		flags = ['-i']
		cmd = [cmd]
		options = [path]
		env = []
		args = []

		stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return


def sleep(pause=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Sleep for pause
	Args:
		pause (int): Time to sleep
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	'''

	if pause is None:
		return

	exe = ['sleep']
	flags = []
	cmd = [pause]
	options = []
	env = []	
	args = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return



def search(path,pattern,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Search for pattern in file
	Args:
		path (str): Path of file
		pattern (str): Pattern to search
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	Returns:
		stdout (int): Line number of last pattern occurrence in file,or -1 if does not exist
	'''

	replacements = {r'-':r'\-',r' ':r'\ ',r'#':r'\#'}

	default = -1

	def wrapper(stdout,stderr,returncode,default=default):
		try:
			result = int(stdout)
		except:
			result = default
		return result

	if path is None or pattern is None:
		result = default
		return result

	for replacement in replacements:
		pattern = pattern.replace(replacement,replacements[replacement])

	args = []

	exe = ['awk']
	flags = []
	cmd = [' /%s/ {print FNR}'%(pattern),path]
	options = []
	arg = [*exe,*flags,*cmd,*options]
	args.append(arg)

	exe = ['tail']
	flags = ['--lines=1']
	cmd = []
	options = []
	arg = [*exe,*flags,*cmd,*options]
	args.append(arg)
	
	exe = []
	flags = []
	cmd = []
	options = []
	env = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,wrapper=wrapper,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return stdout

	
def update(path,patterns,kwargs=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''	
	Update path files with sed
	Args:
		path (str): Path of file
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		kwargs (dict): Additional keyword arguments to update
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity		
	'''

	def wrapper(kwargs,string='.*'):
		_wrapper = lambda pattern,string=string: str(pattern) if pattern is not None else string
		_defaults = {
			'pattern':'.*',
			'value':'.*',
			'prefix':'.*',
			'postfix':'',
			'default':''
		}
		kwargs.update({kwarg: kwargs.get(kwarg,_defaults[kwarg]) for kwarg in _defaults})
		for kwarg in kwargs:
			kwargs[kwarg] = _wrapper(kwargs[kwarg])
		return

	if path is None or patterns is None:
		return

	patterns = {str(pattern): str(patterns[pattern]) for pattern in patterns}
	kwargs = {} if not isinstance(kwargs,dict) else kwargs

	if device in ['pc']:
		default = '#SBATCH'
		def string(**kwargs):
			wrapper(kwargs)
			string = '%s%s --%s=%s%s'%(kwargs['prefix'],kwargs['default'],kwargs['pattern'],kwargs['value'],kwargs['postfix'])
			return string
	elif device in ['slurm']:
		default = '#SBATCH'
		def string(**kwargs):
			wrapper(kwargs)
			string = '%s%s --%s=%s%s'%(kwargs['prefix'],kwargs['default'],kwargs['pattern'],kwargs['value'],kwargs['postfix'])
			return string		
	else:
		default = ''
		def string(**kwargs):
			wrapper(kwargs)
			string = '%s%s=%s%s'%(kwargs['prefix'],kwargs['pattern'],kwargs['value'],kwargs['postfix'])
			return string		

	patterns.update({
			**{pattern: join(patterns.get(pattern,'.')) for pattern in ['chdir'] if pattern in patterns},
			**{pattern: '%s:%s'%(':'.join(patterns.get(pattern,'').split(':')[:-1]),','.join([str(i) for i in kwargs.get('dependencies',[]) if i is not None])) for pattern in ['dependency'] if pattern in patterns},
			**{pattern: '%s-%s:%s%%%s'%(
				patterns.get(pattern,'').split('-')[0].split(':')[0].split('%')[0],
				(str(int(patterns.get(pattern,'').split('-')[0].split(':')[0].split('%')[0])+kwargs.get('size',1)-1) if kwargs.get('size') is not None else patterns.get(pattern,'').split('-')[1].split(':')[0].split('%')[0]),
				(str(kwargs.get('step')) if kwargs.get('step') is not None else patterns.get(pattern,'').split(':')[-1].split('%')[0]),
				patterns.get(pattern,'').split('%')[-1]
				) 
				for pattern in ['array'] if pattern in patterns},
			**{pattern: join(split(patterns.get(pattern),directory_file=True) if patterns.get(pattern) is not None else '%x.%A.%a',
							ext=split(patterns.get(pattern),ext=True) if patterns.get(pattern) is not None else 'stdout',
							root=None)
							for pattern in ['output'] if pattern in patterns},
			**{pattern: join(split(patterns.get(pattern),directory_file=True) if patterns.get(pattern) is not None else '%x.%A.%a',
							ext=split(patterns.get(pattern),ext=True) if patterns.get(pattern) is not None else 'stderr',
							root=None)
							for pattern in ['error'] if pattern in patterns},			
		})

	if process in ['serial']:
		nulls = ['chdir','array']
		patterns.update({})
	elif process in ['parallel']:
		nulls = []
		patterns.clear()
	elif process in ['array']:
		nulls = ['chdir']
		patterns.update({})		
	else:
		nulls = []
		patterns.update({})

	patterns.update({
		string(pattern=pattern,default=default): 
		string(pattern=pattern,value=patterns.pop(pattern,None),prefix='',default=default)
		for pattern in list(patterns)
		if pattern not in nulls
		})

	patterns.update({
		string(pattern=pattern,default=default): 
		string(pattern=pattern,value=patterns.pop(pattern,None),prefix='#',default=default)
		for pattern in list(nulls)
		if search(path,string(pattern=pattern,default=default),execute=True,verbose=verbose) >= 0
		})


	for pattern in nulls:
		patterns.pop(pattern,None)


	sed(path,patterns,default=default,execute=execute,verbose=verbose)

	return


def configure(paths,pwd=None,cwd=None,patterns={},process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Configure paths for jobs with copied/updated files
	Args:
		paths (iterable[str],dict[str,object]): Relative paths of files to pwd/cwd, or dictionary {path:data} with data to update paths
		
		pwd (str): Input root path for files
		cwd (str): Output root path for files
		patterns (dict[str,dict[str,str]]): Patterns and values to update {path:{pattern: replacement}}
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	'''

	if paths is None:
		return

	if patterns is None:
		patterns = {}

	for path in paths:

		# Set data to update
		data = paths[path] if isinstance(paths,dict) else None

		# Set sources and destinations of files
		source = join(path,root=pwd)
		destination = join(path,root=cwd)

		# Update and Dump files
		if isinstance(data,dict):
			data,source,destination = load(source),deepcopy(data),destination
			updater(source,data,func=lambda key,iterable,elements: iterable.get(key,elements[key]))
			dump(source,destination)					
		else:
			cp(source,destination,default=path,execute=execute,verbose=verbose)

	return


def submit(jobs={},args={},paths={},patterns={},dependencies=[],pwd='.',cwd='.',pool=None,pause=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Submit job commands as tasks to command line
	Args:
		jobs (str,dict[str,str]): Submission script, or {key:job}
		args (dict[str,str],dict[str,dict[str,str]]): Arguments to pass to command line, either {arg:value} or {key:{arg:value}}
		paths (dict[str,object],dict[str,dict[str,object]]): Relative paths of files to pwd/cwd, with data to update paths {path:data} or {key:{path:data}}
		patterns (dict[str,dict[str,str]],dict[str,dict[str,dict[str,str]]]): Patterns to update files {path:{pattern:replacement}} or {key:{path:{pattern:replacement}}
		dependencies (iterable[str,int],dict[str,iterable[str,int]]): Dependences of previous jobs to job [dependency] or {key:[dependency]}
		pwd (str,dict[str,str]): Input root path for files, either path, or {key:path}
		cwd (str,dict[str,str]): Output root path for files, either path, or {key:path}
		pool (int): Number of subtasks in a pool per task (parallelized with processes number of parallel processes)
		pause (int,str): Time to sleep after call		
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	Returns:
		results (iterable[str]): Return of commands for each task
	'''

	keys = [None]

	if isinstance(jobs,str):
		jobs = {key:jobs for key in keys}

	keys = intersection(jobs)

	if all(isinstance(args[arg],str) for arg in args) or not all(key in args for key in keys) or not len(args):
		args = {key:args for key in keys}

	keys = intersection(keys,args)

	if not all(key in paths for key in keys) or not len(paths):
		paths = {key:paths for key in keys}

	keys = intersection(keys,paths)

	if not all(key in patterns for key in keys) or not len(patterns):
		patterns = {key:patterns for key in keys}

	keys = intersection(keys,patterns)

	if not isinstance(dependencies,dict) or not len(dependencies):
		dependencies = {key:dependencies for key in keys}

	keys = intersection(keys,dependencies)

	if isinstance(pwd,str):
		pwd = {key:pwd for key in keys}

	keys = intersection(keys,pwd)

	if isinstance(cwd,str):
		cwd = {key:cwd for key in keys}

	keys = intersection(keys,cwd,sort=True)

	pools = {
		path: [key for key in keys if cwd[key] == path]
			for path in sorted(set([cwd[key] for key in cwd]))
		}

	pool = 1 if pool is None else pool
	execution = True if execute == -1 else execute
	execute = False if execute == -1 else execute
	tasks = []
	results = []
	keys = {key:{} for key in keys}

	for key in keys:

		index = pools[cwd[key]].index(key) if process in ['serial'] else 0
		size = len(pools[cwd[key]])
		step = pool if process in ['array'] else 1 if process in ['serial'] else size
		
		path = pools[cwd[key]].index(key) if size>1 else None
		path = join(path,root=cwd[key])

		exe = jobs[key]
		flags = []
		cmd = []
		options = []
		env = []

		kwargs = {
			'key':key,
			'path':path,
			'pwd':pwd[key],
			'cwd':cwd[key],
			'job':jobs[key],
			'cmd':None,
			'env':None,
			'index': index,
			'size': size,
			'step': step,
			'pool':pool,
			'results':results,
			'paths':paths[key],
			'patterns':patterns[key],
			'dependencies':dependencies[key],
			}


		cmd,env = command(args[key],kwargs,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execution,verbose=False)

		configure(paths[key],pwd=pwd[key],cwd=path,patterns=patterns[key],process=None,processes=None,device=None,execute=execution,verbose=False)

		kwargs['cmd'] = cmd
		kwargs['env'] = env

		keys[key] = kwargs


	if process in ['serial']:
		def boolean(task,tasks):
			value = True
			return value
		def updates(task,tasks):
			attr = 'path'
			value = task['cwd']
			task[attr] = value
			return

	elif process in ['parallel']:
		def boolean(task,tasks):
			value = task['cwd'] not in [subtask['cwd'] for subtask in tasks]
			return value
		def updates(task,tasks):
			attr = 'path'
			value = task['cwd']
			task[attr] = value
			return

	elif process in ['array']:
		def boolean(task,tasks):
			value = task['cwd'] not in [subtask['cwd'] for subtask in tasks]
			return value
		def updates(task,tasks):
			attr = 'path'			
			value = task['cwd']
			task[attr] = value
			return

	else:
		def boolean(task,tasks):
			value = task['cwd'] not in [subtask['cwd'] for subtask in tasks]			
			return value
		def updates(task,tasks):
			attr = 'path'
			value = task['cwd']
			task[attr] = value
			return

	for key in keys:
		task = keys[key]
		if boolean(task,tasks):
			updates(task,tasks)
			tasks.append(task)

	for task in tasks:

		job = task['job']
		cmd = task['cmd']
		env = task['env']
		path = task['path']
		cwd = task['cwd']
		pwd = task['pwd']
		patterns = task['patterns']
		kwargs = task

		source = join(job,root=pwd)
		destination = join(job,root=path)

		cp(source,destination,default=job,execute=execution)

		update(destination,patterns,kwargs,process=process,processes=processes,device=device,execute=execution,verbose=False)

		result = call(*cmd,env=env,path=path,pause=pause,process=None,processes=None,device=None,execute=execute,verbose=verbose)

		results.append(result)

	return results


def launch(jobs={},wrapper=None):
	'''
	Submit jobs as job commands as tasks to command line through submit(**job) for each job in jobs
	Args:
		jobs (dict[str,dict[str,object]]): Submission jobs script as {name:job} with job dictionaries {attr:value} with attr keys:
			args (dict[str,str],dict[str,dict[str,str]]): Arguments to pass to command line, either {arg:value} or {key:{arg:value}}
			paths (dict[str,object],dict[str,dict[str,object]]): Relative paths of files to pwd/cwd, with data to update paths {path:data} or {key:{path:data}}
			patterns (dict[str,dict[str,str]],dict[str,dict[str,dict[str,str]]]): Patterns to update files {path:{pattern:replacement}} or {key:{path:{pattern:replacement}}
			dependencies (iterable[str,int],dict[str,iterable[str,int]]): Dependences of previous jobs to job [dependency] or {key:[dependency]}
			pwd (str,dict[str,str]): Input root path for files, either path, or {key:path}
			cwd (str,dict[str,str]): Output root path for files, either path, or {key:path}
			pool (int): Number of subtasks in a pool per task (parallelized with processes number of parallel processes)
			pause (int,str): Time to sleep after call		
			process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']
			processes (int): Number of processes per command			
			device (str): Name of device to submit to
			execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
			verbose (int,str,bool): Verbosity
		wrapper (callable): Wrapper for results for subsequent jobs with signature wrapper(name,jobs,results). Defaults to updating dependencies with results.
	Returns:
		results (iterable[str]): Return of commands for each job
	'''

	if wrapper is None:
		def wrapper(name,jobs,results):
			attr = 'dependencies'
			if attr in jobs[name]:
				jobs[name][attr].extend([result for name in results for result in results[name]])
			return

	results = {}

	for name in jobs:

		wrapper(name,jobs,results)

		result = submit(**jobs[name])

		results[name] = result

	results = [result for name in results for result in results[name]]

	return results
