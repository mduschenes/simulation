#!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,inspect,traceback
import subprocess
import shutil
import glob as globber
import importlib
import json,jsonpickle,h5py,pickle,dill
import numpy as np
import pandas as pd

from natsort import natsorted, ns,index_natsorted,order_by_index

# Logging
import logging
logger = logging.getLogger(__name__)
debug = 0

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.io import cd,mkdir,join,split,load,dump


DEVICES = {
	'pc': {
		'submit': lambda job,args,cwd='.',key='',size=1,tasks=True: ['.',join(cwd,job) if tasks else join(cwd,key,job),*[join(cwd,key,args[arg]) for arg in args]],
		'update': lambda pattern,value='.*':'.*#SBATCH --%s=%s'%(pattern,value),
		'tasks': lambda patterns,cwd='.',key='',size=1,tasks=True: {
			**{pattern: '%d-%d:%s'%(0,size-1,':'.join(patterns.get(pattern,'').split(':')[1:])) for pattern in ['array']},
			**{pattern: join(cwd,key,'%x.%A.%a',ext='stdout') for pattern in ['output']},
			**{pattern: join(cwd,key,'%x.%A.%a',ext='stderr') for pattern in ['error']},
			},
		},		
	'slurm': {
		'submit': lambda job,args,cwd='.',key='',size=1,tasks=True: ['sbatch','<',join(cwd,job) if tasks else join(cwd,key,job),'%s=%s'%('--export',','.join(['%s=%s'%(arg,join(cwd,'$SLURM_ARRAY_TASK_ID',args[arg])) for arg in args]))],
		'update': lambda pattern,value='.*':'.*#SBATCH --%s=%s'%(pattern,value),
		'tasks': lambda patterns,cwd='.',key='',size=1: {
			**{pattern: '%d-%d:%s'%(0,size-1,':'.join(patterns.get(pattern,'').split(':')[1:])) for pattern in ['array']},
			**{pattern: join(cwd,'$SLURM_ARRAY_TASK_ID','%x.%A.%a',ext='stdout') for pattern in ['output']},
			**{pattern: join(cwd,'$SLURM_ARRAY_TASK_ID','%x.%A.%a',ext='stderr') for pattern in ['error']},
			},		
		},
	None: {
		'submit': lambda job,args: ['.',job,*[args[arg] for arg in args]],
		'update': lambda pattern,value='.*':'.*#SBATCH --%s=%s'%(pattern,value),
		'tasks': lambda patterns,cwd='.',key='',size=1: {
			**{pattern: '%d-%d:%s'%(0,size-1,':'.join(patterns.get(pattern,'').split(':')[1:])) for pattern in ['array']},
			**{pattern: join(cwd,key,'%x.%A.%a',ext='stdout') for pattern in ['output']},
			**{pattern: join(cwd,key,'%x.%A.%a',ext='stderr') for pattern in ['error']},
			},
		},
	}

attrs = ['submit','update','tasks']
for device in DEVICES:
	for attr in attrs:
		assert attr in DEVICES[device], 'device: %s , attr: "%s" not in allowed %r'%(device,attr,list(DEVICES[device]))


def call(*args,path='.',wrapper=None,device=None,execute=True,**kwargs):
	'''
	Submit call to command line
	Args:
		args (str,iterable[str],iterable[iterable[str]]): Arguments to pass to command line, nested iterables are piped
		path (str): Path to call from
		wrapper (callable): Wrapper for stdout with signature wrapper(stdout,stderr,returncode)		
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
	Returns:
		result (object): Return of commands
	'''

	def parser(args):
		delimiter = ' '
		pipe = ' | '
		if isinstance(args,str):
			args = [args]
		elif all(isinstance(arg,str) for arg in args):			
			args = [delimiter.join(args)]
		else:
			args = [pipe.join([delimiter.join(arg) if not isinstance(arg,str) else arg for arg in args])]
		return args

	def caller(args):
		run = lambda args: subprocess.run(args,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)		
		result = run(args)
		return result.stdout,result.stderr,result.returncode

	def wrapper(stdout,stderr,returncode,wrapper=wrapper):
		try:
			stdout = stdout.strip().decode('utf-8')
			stderr = stderr.strip().decode('utf-8')
		except AttributeError:
			pass
		try:
			result = wrapper(stdout,stderr,returncode)
		except:
			result = stdout
		return result

	mkdir(path)

	with cd(path):
		if execute:
			result = wrapper(*caller(parser(args)))
		else:
			result = ' '.join(parser(args))
			print(result)
	
	return result

def copy(source,destination,device=None,execute=True,**kwargs):
	'''
	Copy objects from source to destination
	Args:
		source (str): Path of source object
		destination (str): Path of destination object
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
		kwargs (dict): Additional copying keyword arguments
	'''
	assert os.path.exists(source), "source %s does not exist"%(source)

	mkdir(destination)

	args = ['cp','-rf',source,destination]

	stdout = call(*args,device=device,execute=execute)

	# shutil.copy2(source,destination)

	return


def sed(path,patterns,default=None,device=None,execute=True):
	'''
	GNU sed replace patterns in path
	Args:
		path (str): Path of file
		patterns (dict[str,str]): Patt
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
	'''

	if path is None or patterns is None:
		return

	delimiters = [',','#','/','$']
	replacements = {r"-":r"\-",r" ":r"\ "}

	exe = 'sed'
	options = ['-i']

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

		if cmd is None:
			continue

		for replacement in replacements:
			cmd = cmd.replace(replacement,replacements[replacement])

		args=[exe,*options,cmd,path]

		result = call(*args,device=device,execute=execute)

	return




def search(path,pattern,device=None,execute=True):
	'''
	Search for pattern in file
	Args:
		path (str): Path of file
		pattern (str): Pattern to search
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

	exe = 'awk'
	options = []
	cmd = ["'/%s/ {print FNR}'"%(pattern),path]
	args.append([exe,*options,*cmd])

	exe = 'tail'
	options = ['--lines=1']
	cmd = []
	args.append([exe,*options,*cmd])

	result = call(*args,wrapper=wrapper,device=device,execute=execute)

	return result

	
def update(path,patterns,device=None,exe=True):
	'''	
	Update path files with sed
	Args:
		path (str): Path of file
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		device (str): Name of device to submit to
		exe (boolean): Boolean whether to issue commands		
	'''

	assert device in DEVICES, 'device: "%s" not in allowed %r'%(device,list(DEVICES))
	
	if path is None or patterns is None:
		return

	patterns = deepcopy(patterns)

	for pattern in patterns:
		value = str(patterns.pop(pattern))

		pattern,replacement = (
			DEVICES[device]['update'](pattern=pattern),
			DEVICES[device]['update'](pattern=pattern,value=value),
			)

		patterns[pattern] = replacement

	sed(path,patterns,default=default,device=device,exe=exe)

	return


def configure(paths,pwd=None,cwd=None,patterns={}):
	'''
	Configure paths for jobs with copied/updated files
	Args:
		paths (iterable[str],dict[str,object]): Relative paths of files to pwd/cwd, or dictionary {path:data} with data to update paths
		pwd (str): Input root path for files
		cwd (str): Output root path for files
		patterns (dict[str,dict[str,str]]): Patterns and values to update {path:{pattern: replacement}}
	'''

	if paths is None:
		return

	if patterns is None:
		patterns = {}

	for path in paths:

		# Set data to update
		data = paths[path] if isinstance(paths,dict) else None

		# Set sources and destinations of files
		source = join(split(path,directory=True),split(path,file=True),ext=split(path,ext=True),root=pwd)
		destination = join(split(path,file=True),ext=split(path,ext=True),root=cwd)

		mkdir(destination)

		# Update and Dump files
		if isinstance(data,dict):
			data,source,destination = load(source),deepcopy(data),destination
			updater(source,data,func=lambda key,iterable,elements: iterable.get(key,elements[key]))
			dump(source,destination)					
		else:
			copy(source,destination)

		pattern = patterns.get(path)

		update(destination,pattern)

	return


def submit(job,args,pwd='.',cwd='.',paths={},patterns={},tasks=True,device=None,execute=True):
	'''
	Submit job commands to command line
	Args:
		job (str): Submission script
		args (dict[str,str],dict[str,dict[str,str]]): Arguments to pass to command line, either {arg:value} or {key:{arg:value}}
		pwd (str): Input root path for files
		cwd (str): Output root path for files
		paths (dict[str,object],dict[str,dict[str,object]]): Relative paths of files to pwd/cwd, with data to update paths {path:data} or {key:{path:data}}
		patterns (dict[str,dict[str,str]],dict[str,dict[str,dict[str,str]]]): Patterns to update files {path:{pattern:replacement}} or {key:{path:{pattern:replacement}}
		tasks (boolean): Submit multiple keys as array of tasks, otherwise split into separate jobs
		device (str): Name of device to submit to
		execute (boolean): Boolean whether to issue commands
	Returns:
		stdouts (str,iterable[str]): Return of commands
	'''

	if all(isinstance(arg,str) for arg in args):
		key = ''
		args = {key:args}

	keys = list(args)
	size = len(keys)
	stdouts = []

	if not all(key in paths for key in keys):
		paths = {key:paths for key in keys}

	if not all(key in patterns for key in keys):
		patterns = {key:patterns for key in keys}

	
	for key in keys:

		if not tasks:
			patterns[key].pop('array')
		else:
			pass

		path=join(cwd,key)

		patterns[key].update(
			DEVICES[device]['tasks'](patterns[key],cwd=cwd,key=key,size=size,tasks=tasks)
			)


		configure(pwd=pwd,cwd=path,paths=paths[key],patterns=patterns[key])

		cmd = DEVICES[device]['submit'](job,args[key],cwd=cwd,key=key,size=size,tasks=tasks)

		if not tasks:
			stdout = call(*cmd,path=path,device=device,execute=execute)
			stdouts.append(stdout)
		else:
			source = join(job,root=(cwd,key))
			destination = cwd
			copy(source,destination)

	if tasks:
		path = cwd
		stdouts = call(*cmd,path=path,device=device,execute=execute)

	return stdouts