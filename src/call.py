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
from src.io import cd,mkdir,join,split,load,dump
from src.dictionary import updater

name = __name__
path = os.getcwd()
file = 'logging.conf'
conf = os.path.join(path,file)
file = None #'log.log'
logger = Logger(name,conf,file=file)


def _submit(job,args,process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None,**kwargs):
	'''
	Update submit arguments
	Args:
		job (str): Submission script
		args (dict[str,str],dict[str,iterable[str],iterable[iterable[str]]]): Arguments to pass to command line {arg:value} or {arg:[value]}
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Additional keyword arguments to update arguments
	Returns:
		args (iterable[str]): Updated submit arguments
	'''

	args = {arg: [args[arg]] if isinstance(args[arg],str) else [subarg if isinstance(subarg,str) else ' '.join(subarg) for subarg in args[arg]] 
				for arg in args}

	if device in ['pc']:
		exe = ['./%s'%(job)]
		flags = []
		cmd = [subarg for arg in args for subarg in args[arg]]
		options = []		
	elif device in ['slurm']:
		exe = ['sbatch']
		flags = ['%s=%s'%('--export',','.join(['%s=%s'%(arg,' '.join([subarg for subarg in args[arg]])) for arg in args])),'<']
		cmd = [job]
		options = []		
	else:
		exe = ['.',job]
		flags = []
		cmd = [subarg for arg in args for subarg in args[arg]]
		options = []

	if parallelism in ['parallel']:
		options.extend(['&'])
	elif parallelism in ['serial']:
		options.extend([])		
	else:
		options.extend([])		

	args = [*exe,*flags,*cmd,*options]

	return args

def _update(path,patterns,process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None,**kwargs):
	'''
	Update submit patterns in-place
	Args:
		path (str): Path of file	
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		verbose (int,str,bool): Verbosity		
		kwargs (dict): Additional keyword arguments to update
	Returns:
		default (str): Default pattern
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

	null = []

	if process in ['serial']:
		nulls = ['chdir','array']
		null.extend(nulls)
		patterns.update({
			**{pattern: join(patterns.get(pattern,'.')) for pattern in ['chdir'] if pattern in patterns},
			**{pattern: '%s:%s'%(':'.join(patterns.get(pattern,'').split(':')[:-1]),','.join([str(i) for i in kwargs.get('dependencies',[]) if i is not None])) for pattern in ['dependency'] if pattern in patterns},
			**{pattern: join(split(patterns.get(pattern),directory_file=True) if patterns.get(pattern) is not None else '%x.%A',
							ext=split(patterns.get(pattern),ext=True) if patterns.get(pattern) is not None else 'stdout',
							root=None)
							for pattern in ['output'] if pattern in patterns},
			**{pattern: join(split(patterns.get(pattern),directory_file=True) if patterns.get(pattern) is not None else '%x.%A',
							ext=split(patterns.get(pattern),ext=True) if patterns.get(pattern) is not None else 'stderr',
							root=None)
							for pattern in ['error'] if pattern in patterns},			
			})
	elif process in ['parallel']:
		null.clear()
		patterns.clear()
	elif process in ['array']:
		nulls = ['chdir']
		null.extend(nulls)
		patterns.update({
			**{pattern: join(patterns.get(pattern),'.') for pattern in ['chdir'] if pattern in patterns},
			**{pattern: '%s:%s'%(':'.join(patterns.get(pattern,'').split(':')[:-1]),','.join([str(i) for i in kwargs.get('dependencies',[]) if i is not None])) for pattern in ['dependency'] if pattern in patterns},
			**{pattern: '%s-%s:%s%%%s'%(
				patterns.get(pattern,'').split('-')[0].split(':')[0].split('%')[0],
				(str(int(patterns.get(pattern,'').split('-')[0].split(':')[0].split('%')[0])+kwargs.get('size',1)-1) if kwargs.get('size') is not None else patterns.get(pattern,'').split('-')[1].split(':')[0].split('%')[0]),
				(str(kwargs.get('step')) if kwargs.get('step') is not None else patterns.get(pattern,'').split(':')[-1].split('%')[0]),
				patterns.get(pattern,'').split('%')[-1]
				) 
				for pattern in ['array'] if pattern in patterns},
			**{pattern: join(split(patterns.get(pattern),directory_file=True) if patterns.get(pattern) is not None else '%x.%A',
							ext=split(patterns.get(pattern),ext=True) if patterns.get(pattern) is not None else 'stdout',
							root='%a')
							for pattern in ['output'] if pattern in patterns},
			**{pattern: join(split(patterns.get(pattern),directory_file=True) if patterns.get(pattern) is not None else '%x.%A',
							ext=split(patterns.get(pattern),ext=True) if patterns.get(pattern) is not None else 'stderr',
							root='%a')
							for pattern in ['error'] if pattern in patterns},			
		})

	patterns.update({
		string(pattern=pattern,default=default): 
		string(pattern=pattern,value=patterns.pop(pattern,None),prefix='',default=default)
		for pattern in list(patterns)
		if pattern not in null
		})

	patterns.update({
		string(pattern=pattern,default=default): 
		string(pattern=pattern,value=patterns.pop(pattern,None),prefix='#',default=default)
		for pattern in list(null)
		if search(path,string(pattern=pattern,default=default),execute=True,verbose=verbose) >= 0
		})


	for pattern in null:
		patterns.pop(pattern,None)

	return default


class popen(object):
	'''
	Class to safely enter process
	Args:
		path (str): Path to change to
	'''
	def __init__(self,cls):
		self.cls = cls
		return
	def __enter__(self,*args,**kwargs):
		try:
			return self.cls.__enter__(*args,**kwargs)
		except:
			return self.cls
	def __exit__(self,etype, value, traceback):
		try:
			return self.cls.__exit__(etype, value, traceback)
		except:
			return

def call(*args,path=None,wrapper=None,pause=None,process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None,**kwargs):
	'''
	Submit call to command line
	Args:
		args (str,iterable[str],iterable[iterable[str]]): Arguments to pass to command line, nested iterables are piped
		path (str): Path to call from
		wrapper (callable): Wrapper for stdout with signature wrapper(stdout,stderr,returncode)		
		pause (int,str): Time to sleep after call
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance		
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Additional keyword arguments to call		
	Returns:
		result (object): Return of commands
	'''

	def caller(args,inputs=None,device=None,verbose=None):

		def run(args,stdin=None,stdout=None,stderr=None):
			result = subprocess.Popen(args,stdin=stdin,stdout=stdout,stderr=stderr)
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

			result = run(arg,stdin=stdin,stdout=stdout,stderr=stderr)

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

		sleep(pause,device=device,execute=execute)

		return stdout,stderr,returncode

	def wrapper(stdout,stderr,returncode,wrapper=wrapper,device=None,verbose=None):
		try:
			result = wrapper(stdout,stderr,returncode)
		except:
			result = stdout

		return result

	def parser(*args,device=None,verbose=None):

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

		return inputs,args,cmd


	inputs,args,cmd = parser(*args,device=device,verbose=verbose)
	result = None

	msg = '%s : %s'%(path,cmd) if path is not None else cmd
	logger.log(verbose,msg)

	if execute:
		with cd(path):
			result = wrapper(*caller(args,inputs=inputs,device=device,verbose=verbose),device=device,verbose=verbose)

	return result

def cp(source,destination,process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None,**kwargs):
	'''
	Copy objects from source to destination
	Args:
		source (str): Path of source object
		destination (str): Path of destination object
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance		
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Additional keyword arguments to copy
	'''
	assert os.path.exists(source), "source %s does not exist"%(source)

	mkdir(destination)

	exe = ['cp']
	flags = ['-rf']
	cmd = [source,destination]
	args = [*exe,*flags,*cmd]

	stdout = call(*args,process=process,processes=processes,parallelism=parallelism,device=device,execute=execute,verbose=verbose)

	return


def rm(path,process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None,**kwargs):
	'''
	Remove path
	Args:
		path (str): Path to remove
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance		
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Additional keyword arguments to copy
	'''

	exe = ['rm']
	flags = ['-rf']
	cmd = [path]
	args = [*exe,*flags,*cmd]

	stdout = call(*args,process=process,processes=processes,parallelism=parallelism,device=device,execute=execute,verbose=verbose)

	return


def sed(path,patterns,default=None,process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None):
	'''
	GNU sed replace patterns in path
	Args:
		path (str): Path of file
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance		
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		verbose (int,str,bool): Verbosity
	'''

	if path is None or patterns is None:
		return

	delimiters = [',','#','/','$']
	replacements = {}#{r"-":r"\-",r" ":r"\ ",r"#":r"\#"}

	exe = ['sed']
	flags = ['-i']

	for pattern in patterns:

		cmd = None
		
		result = search(path,pattern,execute=True,verbose=verbose)

		if result == -1:
			result = search(path,default,execute=True,verbose=verbose)
			if result == -1:					
				cmd = None
			else:
				cmd = "%di %s"%(result+1,patterns[pattern])			
		else:			
			for delimiter in delimiters:
				if all(delimiter not in string for string in (pattern,patterns[pattern])):
					cmd = "s%s%s%s%s%sg"%(delimiter,pattern,delimiter,patterns[pattern],delimiter)
					break

		if cmd is None:
			continue

		for replacement in replacements:
			cmd = cmd.replace(replacement,replacements[replacement])

		cmd = [cmd,path]

		args=[*exe,*flags,*cmd]

		result = call(*args,device=device,execute=execute,verbose=verbose)

	return


def sleep(pause=None,process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None,**kwargs):
	'''
	Sleep for pause
	Args:
		pause (int): Time to sleep
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance		
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Additional keyword arguments to sleep
	'''

	if pause is None:
		return

	exe = ['sleep']
	flags = []
	cmd = [pause]
	args = [*exe,*flags,*cmd]

	stdout = call(*args,process=process,processes=processes,parallelism=parallelism,device=device,execute=execute,verbose=verbose)

	return



def search(path,pattern,process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None):
	'''
	Search for pattern in file
	Args:
		path (str): Path of file
		pattern (str): Pattern to search
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance		
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		verbose (int,str,bool): Verbosity
	Returns:
		result (int): Line number of last pattern occurrence in file,or -1 if does not exist
	'''

	replacements = {r"-":r"\-",r" ":r"\ ",r"#":r"\#"}

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
	arg = [*exe,*flags,*cmd]
	args.append(arg)

	exe = ['tail']
	flags = ['--lines=1']
	cmd = []
	arg = [*exe,*flags,*cmd]
	args.append(arg)

	result = call(*args,wrapper=wrapper,process=process,processes=processes,parallelism=parallelism,device=device,execute=execute,verbose=verbose)

	return result

	
def update(path,patterns,process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None,**kwargs):
	'''	
	Update path files with sed
	Args:
		path (str): Path of file
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance		
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		verbose (int,str,bool): Verbosity		
		kwargs (dict): Additional keyword arguments to update
	'''
	if path is None or patterns is None:
		return

	patterns = deepcopy(patterns)

	default = _update(path,patterns,process=process,processes=processes,parallelism=parallelism,device=device,**kwargs)

	sed(path,patterns,default=default,execute=execute,verbose=verbose)

	return


def configure(paths,pwd=None,cwd=None,patterns={},process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None,**kwargs):
	'''
	Configure paths for jobs with copied/updated files
	Args:
		paths (iterable[str],dict[str,object]): Relative paths of files to pwd/cwd, or dictionary {path:data} with data to update paths
		pwd (str): Input root path for files
		cwd (str): Output root path for files
		patterns (dict[str,dict[str,str]]): Patterns and values to update {path:{pattern: replacement}}
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance		
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Additional keyword arguments to configure

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
			cp(source,destination,execute=execute,verbose=verbose)

	return


def submit(jobs={},args={},paths={},patterns={},dependencies=[],pwd='.',cwd='.',pause=None,process=None,processes=None,parallelism=None,device=None,execute=False,verbose=None):
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
		pause (int,str): Time to sleep after call		
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']
		processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance		
		parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
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

	processes = 1 if processes is None else processes

	unique = {
		path: {
			key: (
				[subkey for subkey in keys if cwd[subkey]==cwd[key]].index(key)//processes,
				[subkey for subkey in keys if cwd[subkey]==cwd[key]].index(key)%processes)
				for key in keys if cwd[key]==path
			}
	 for path in set([cwd[key] for key in cwd])
	 }

	unique = {attr: unique[attr] for attr in sorted(unique)}

	cmds = {}
	results = []
	tasks = []

	paths,pwd,cwd,patterns,process,processes,parallelism,device,execute,verbose
	kwargs = {
		key:{
			'index': [str(i) for i in unique[cwd[key]][key]],
			'size': len(unique[cwd[key]]),
			'step': processes,
			'results':results,
			'tasks':tasks,
			'dependencies':dependencies[key],
			}
		for key in keys
		}


	for key in keys:

		index = kwargs[key]['index']
		size = kwargs[key]['size']
		step = kwargs[key]['step']

		path = (index if processes > 1 else index[:1]) if size > 1 else [None]

		path = join(*path)

		path = join(path,root=cwd[key])

		configure(pwd=pwd[key],cwd=path,paths=paths[key],patterns=patterns[key],process=process,processes=None,parallelism=None,device=device,execute=True,**kwargs[key])

		cmd = _submit(job=jobs[key],args=args[key],process=process,processes=None,parallelism=None,device=device,execute=True)

		cmds[key] = cmd

		task = {'path':path,'key':key}

		boolean = lambda task,tasks: task not in tasks

		if boolean(task,tasks):		
			tasks.append(task)

	if process in ['serial']:
		pass
	elif process in ['parallel']:
		tasks.clear()
	elif process in ['array']:
		tasks.clear()
		for path in unique:
			for key in unique[path]:
				
				path = None

				path = join(path)

				task = {'path':path,'key':key}

				boolean = lambda task,tasks: cwd[task['key']] not in [cwd[subtask['key']] for subtask in tasks]

				if boolean(task,tasks):		
					tasks.append(task)
	else:
		pass



	for task in tasks:

		key = task['key']
		path = task['path']

		path = join(path,root=cwd[key])

		cmd = cmds[key]
		job = jobs[key]

		source = join(job,root=pwd[key])
		destination = join(job,root=path)

		cp(source,destination,execute=True)

		update(destination,patterns[key],process=process,processes=None,parallelism=None,device=device,execute=True,**kwargs[key])

		result = call(*cmd,path=path,pause=pause,process=process,processes=None,parallelism=None,device=device,execute=execute,verbose=verbose)

		results.append(result)

	return results


def launch(jobs={},wrapper=None):
	'''
	Submit jobs as job commands as tasks to command line through submit(job,args,paths,patterns,dependencies,...)
	Args:
		jobs (dict[str,dict[str,object]]): Submission jobs script as {name:job} with job dictionaries {attr:value} with attr keys:
			args (dict[str,str],dict[str,dict[str,str]]): Arguments to pass to command line, either {arg:value} or {key:{arg:value}}
			paths (dict[str,object],dict[str,dict[str,object]]): Relative paths of files to pwd/cwd, with data to update paths {path:data} or {key:{path:data}}
			patterns (dict[str,dict[str,str]],dict[str,dict[str,dict[str,str]]]): Patterns to update files {path:{pattern:replacement}} or {key:{path:{pattern:replacement}}
			dependencies (iterable[str,int],dict[str,iterable[str,int]]): Dependences of previous jobs to job [dependency] or {key:[dependency]}
			pwd (str,dict[str,str]): Input root path for files, either path, or {key:path}
			cwd (str,dict[str,str]): Output root path for files, either path, or {key:path}
			pause (int,str): Time to sleep after call		
			process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']
			processes (int,iterable[int]): Number of processes or iterable of number of nested processes for process instance			
			parallelism (str): Type of parallelism, allowed strings in ['serial','parallel']
			device (str): Name of device to submit to
			execute (boolean): Boolean whether to issue commands
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
