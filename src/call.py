#!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,inspect,traceback
from copy import deepcopy
import subprocess

# Logging
import logging
logger = logging.getLogger(__name__)
debug = 0

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import intersection
from src.io import cd,mkdir,join,split,load,dump
from src.dictionary import updater

def _submit(job,args,process=None,device=None,execute=True,**kwargs):
	'''
	Update submit arguments
	Args:
		job (str): Submission script
		args (dict[str,str]): Arguments to pass to command line {arg:value}
		process (str): Type of processing, either submission in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands		
		kwargs (dict): Additional keyword arguments to update arguments
	Returns:
		args (iterable[str]): Updated submit arguments
	'''

	if device in ['pc']:
		exe = ['./%s'%(job)]
		flags = []
		cmd = [args[arg] for arg in args]
	elif device in ['slurm']:
		exe = ['sbatch','<',job]
		flags = []
		cmd = ['%s=%s'%('--export',','.join(['%s=%s'%(arg,args[arg]) for arg in args]))]
	else:
		exe = ['.',job]
		flags = []
		cmd = [args[arg] for arg in args]						

	args = [*exe,*flags,*cmd]

	return args

def _update(path,patterns,process=None,device=None,execute=True,**kwargs):
	'''
	Update submit patterns in-place
	Args:
		path (str): Path of file	
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		process (str): Type of processing, either submission in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands		
		kwargs (dict): Additional keyword arguments to update
	Returns:
		default (str): Default pattern
	'''


	if device in ['pc']:
		default = '#SBATCH'
	elif device in ['slurm']:
		default = '#SBATCH'
	else:
		default = '#SBATCH'

	null = []

	if process in ['serial','parallel']:
		null.extend(['array'])
		patterns.update({
			**{pattern: join(patterns.get(pattern,'.')) for pattern in ['chdir']},
			**{pattern: join('%x.%A',ext='stdout') for pattern in ['output']},
			**{pattern: join('%x.%A',ext='stderr') for pattern in ['error']},
			})
	elif process in ['array']:
		null.extend([])
		patterns.update({
			**{pattern: join(patterns.get(pattern),'%a') for pattern in ['chdir']},
			**{pattern: '%d-%d:%s'%(0,kwargs.get('size',1),':'.join(patterns.get(pattern,'').split(':')[1:])) for pattern in ['array']},
			**{pattern: join('%x.%A',ext='stdout') for pattern in ['output']},
			**{pattern: join('%x.%A',ext='stderr') for pattern in ['error']},
		})

	for pattern in null:
		patterns.pop(pattern)

	null.clear()
	null.extend(list(patterns))

	patterns.update({
		'%s%s --%s=%s%s'%('.*',default,pattern,'','.*'): '%s%s --%s=%s%s'%('',default,pattern,patterns[pattern],'')
		for pattern in patterns
		})

	for pattern in null:
		patterns.pop(pattern)

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

def call(*args,path='.',wrapper=None,process=None,device=None,execute=True,**kwargs):
	'''
	Submit call to command line
	Args:
		args (str,iterable[str],iterable[iterable[str]]): Arguments to pass to command line, nested iterables are piped
		path (str): Path to call from
		wrapper (callable): Wrapper for stdout with signature wrapper(stdout,stderr,returncode)		
		process (str): Type of processing, either submission in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		kwargs (dict): Additional keyword arguments to call		
	Returns:
		result (object): Return of commands
	'''

	def parser(args,device):
		delimiter = ' '
		pipe = ' | '
		if isinstance(args,str):
			if device in ['pc']:
				args = [args]
			elif device in ['slurm']:
				args = [args]				
			else:
				args = [args]
		elif all(isinstance(arg,str) for arg in args):			
			if device in ['pc']:
				args = [arg for arg in args]
			elif device in ['slurm']:
				args = [delimiter.join(args)]								
			else:
				args = [delimiter.join(args)]				
		else:
			if device in ['pc']:
				args = [*[subarg for arg in args[:1] for subarg in arg],*[subarg for arg in args[1:-1] for subarg in [arg,pipe]],*[subarg for arg in args[:1] for subarg in arg]]
			elif device in ['slurm']:
				args = [pipe.join([delimiter.join(arg) if not isinstance(arg,str) else arg for arg in args])]				
			else:
				args = [pipe.join([delimiter.join(arg) if not isinstance(arg,str) else arg for arg in args])]				
		return args

	def caller(args,device):
		if device in ['pc']:
			run = lambda args: subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
		elif device in ['slurm']:
			run = lambda args: subprocess.run(args,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)		
		else:
			run = lambda args: subprocess.run(args,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)		

		with popen(run(args)) as result:
			if device in ['pc']:
				for line in result.stdout:
					print(line,end='')
				stdout,stderr,returncode = ''.join(result.stdout),''.join(result.stderr),result.returncode
			else:
				stdout,stderr,returncode = result.stdout,result.stderr,result.returncode
		return stdout,stderr,returncode

	def wrapper(stdout,stderr,returncode,wrapper=wrapper,device=None):
		try:
			stdout = stdout.strip().decode('utf-8')
			stderr = stderr.strip().decode('utf-8')
		except AttributeError:
			pass
		try:
			result = wrapper(stdout,stderr,returncode)
		except:
			result = stdout

		if returncode != 0:
			print(stderr)

		return result

	mkdir(path)

	with cd(path):
		if execute:
			result = wrapper(*caller(parser(args,device),device),device=device)
		else:
			result = ' '.join(parser(args))
			print(result)
	
	return result

def copy(source,destination,process=None,device=None,execute=True,**kwargs):
	'''
	Copy objects from source to destination
	Args:
		source (str): Path of source object
		destination (str): Path of destination object
		process (str): Type of processing, either submission in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		kwargs (dict): Additional keyword arguments to copy
	'''
	assert os.path.exists(source), "source %s does not exist"%(source)

	mkdir(destination)

	exe = ['cp']
	flags = ['-rf']
	cmd = [source,destination]
	args = [*exe,*flags,*cmd]

	stdout = call(*args,process=process,device=device,execute=execute)

	# shutil.copy2(source,destination)

	return


def sed(path,patterns,default=None,process=None,device=None,execute=True):
	'''
	GNU sed replace patterns in path
	Args:
		path (str): Path of file
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		process (str): Type of processing, either submission in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
	'''

	if path is None or patterns is None:
		return

	delimiters = [',','#','/','$']
	replacements = {r"-":r"\-",r" ":r"\ "}

	exe = ['sed']
	flags = ['-i']

	for pattern in patterns:

		cmd = None
		
		result = search(path,pattern)

		if result == -1:
			result = search(path,default)
			if result == -1:					
				cmd = None
				# cmd = "$a%s"%(patterns[pattern])
			else:
				cmd = "%di %s"%(result+1,patterns[pattern])			
		else:			
			for delimiter in delimiters:
				if all(delimiter not in string for string in (pattern,patterns[pattern])):
					cmd = '"s%s%s%s%s%sg"'%(delimiter,pattern,delimiter,patterns[pattern],delimiter)
					break

		if cmd is None:
			continue

		for replacement in replacements:
			cmd = cmd.replace(replacement,replacements[replacement])

		cmd = [cmd,path]

		args=[*exe,*flags,*cmd]


		result = call(*args,device=device,execute=execute)

	return




def search(path,pattern,process=None,device=None,execute=True):
	'''
	Search for pattern in file
	Args:
		path (str): Path of file
		pattern (str): Pattern to search
		process (str): Type of processing, either submission in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
	Returns:
		result (int): Line number of last pattern occurrence in file,or -1 if does not exist
	'''

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

	args = []

	exe = ['awk']
	flags = []
	cmd = ["'/%s/ {print FNR}'"%(pattern),path]
	args.append([*exe,*flags,*cmd])

	exe = ['tail']
	flags = ['--lines=1']
	cmd = []
	args.append([*exe,*flags,*cmd])

	result = call(*args,wrapper=wrapper,process=process,device=device,execute=execute)

	return result

	
def update(path,patterns,process=None,device=None,execute=True,**kwargs):
	'''	
	Update path files with sed
	Args:
		path (str): Path of file
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		process (str): Type of processing, either submission in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands		
		kwargs (dict): Additional keyword arguments to update
	'''
	if path is None or patterns is None:
		return

	patterns = deepcopy(patterns)

	default = _update(path,patterns,process=process,device=device,**kwargs)

	sed(path,patterns,default=default,execute=execute)

	return


def configure(paths,pwd=None,cwd=None,patterns={},process=None,device=None,execute=True,**kwargs):
	'''
	Configure paths for jobs with copied/updated files
	Args:
		paths (iterable[str],dict[str,object]): Relative paths of files to pwd/cwd, or dictionary {path:data} with data to update paths
		pwd (str): Input root path for files
		cwd (str): Output root path for files
		patterns (dict[str,dict[str,str]]): Patterns and values to update {path:{pattern: replacement}}
		process (str): Type of processing, either submission in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
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
			copy(source,destination,execute=execute)

	return


def submit(jobs,args={},paths={},patterns={},pwd='.',cwd='.',process=None,device=None,execute=True):
	'''
	Submit job commands to command line
	Args:
		jobs (str,dict[str,str]): Submission script, or {key:job}
		args (dict[str,str],dict[str,dict[str,str]]): Arguments to pass to command line, either {arg:value} or {key:{arg:value}}
		paths (dict[str,object],dict[str,dict[str,object]]): Relative paths of files to pwd/cwd, with data to update paths {path:data} or {key:{path:data}}
		patterns (dict[str,dict[str,str]],dict[str,dict[str,dict[str,str]]]): Patterns to update files {path:{pattern:replacement}} or {key:{path:{pattern:replacement}}
		pwd (str): Input root path for files
		cwd (str): Output root path for files		
		process (str): Type of processing, either submission in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
	Returns:
		stdouts (str,iterable[str]): Return of commands
	'''

	keys = [None]

	if isinstance(jobs,str):
		jobs = {key:jobs for key in keys}

	keys = intersection(jobs)

	if all(isinstance(args[arg],str) for arg in args) or not len(args):
		args = {key:args for key in keys}

	keys = intersection(jobs,args)

	if not all(key in paths for key in keys) or not len(paths):
		paths = {key:paths for key in keys}

	keys = intersection(jobs,args,paths)

	if not all(key in patterns for key in keys) or not len(patterns):
		patterns = {key:patterns for key in keys}

	keys = intersection(jobs,args,paths,patterns)

	keys = list(sorted(keys))

	size = len(keys)
	
	cmds = {}
	stdouts = []
	kwargs = {'size':size}

	if not size:
		return

	for i,key in enumerate(keys):

		i = str(i)
		key = key

		path = join(cwd,i)

		configure(pwd=pwd,cwd=path,paths=paths[key],patterns=patterns[key],process=process,device=device,**kwargs)

		cmd = _submit(job=jobs[key],args=args[key],process=process,device=device)

		cmds[key] = cmd

	if process in ['serial']:
		for i,key in enumerate(keys):

			i = str(i)
			key = key

			path = join(cwd,i)
			cmd = cmds[key]
			job = jobs[key]


			source = join(job,root=pwd)
			destination = join(job,root=path)

			copy(source,destination)
			update(destination,patterns[key],process=process,device=device,**kwargs)

			stdout = call(*cmd,path=path,device=device,execute=execute)

			stdouts.append(stdout)

	elif process in ['parallel']:
		pass
	
	elif process in ['array']:

		i = None
		key = keys[-1]

		path = join(cwd,i)
		cmd = cmds[key]
		job = jobs[key]

		source = join(job,root=pwd)
		destination = join(job,root=path)

		copy(source,destination)
		update(destination,patterns[key],process=process,device=device,**kwargs)

		stdout = call(*cmd,path=path,device=device,execute=execute)

		stdouts = stdout

	else:
		for i,key in enumerate(keys):
	
			i = str(i)
			key = key

			path = join(cwd,i)
			cmd = cmds[key]
			job = jobs[key]

			source = join(job,root=pwd)
			destination = join(job,root=path)

			copy(source,destination)
			update(destination,patterns[key],process=process,device=device,**kwargs)

			stdout = call(*cmd,path=path,device=device,execute=execute)

			stdouts.append(stdout)

	return stdouts
