#!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,inspect,traceback
from functools import partial
from copy import deepcopy
import subprocess
from natsort import natsorted

# Logging
import logging

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import intersection,scalars
from src.io import cd,mkdir,join,split,load,dump,exists,environ
from src.dictionary import updater
from src.parallel import Parallelize,Pooler

from src.system	 import Logger
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
				'SLURM_ARRAY_TASK_ID':kwargs.get('id'),
				'SLURM_ARRAY_TASK_MIN':kwargs.get('min'),
				'SLURM_ARRAY_TASK_MAX':kwargs.get('max'),
				'SLURM_ARRAY_TASK_STEP':kwargs.get('step'),
				'SLURM_ARRAY_TASK_COUNT':kwargs.get('count'),
				'SLURM_ARRAY_TASK_SLICE':kwargs.get('slice'),
				'SLURM_ARRAY_TASK_SIZE':kwargs.get('size'),
			},
			**env			
		}

	elif device in ['slurm']:
		exe,flags,cmd,options,env = (
			['sbatch'],
			[*flags,'%s=%s'%('--export',','.join(['%s=%s'%(arg,' '.join([subarg for subarg in args[arg]])) for arg in args]))],
			['<'],
			[*exe,*cmd,*options],
			{
			**{
				'SLURM_ARRAY_TASK_SLICE':kwargs.get('slice'),
				'SLURM_ARRAY_TASK_SIZE':kwargs.get('size'),
				},
			**env
			}
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
				'SLURM_ARRAY_TASK_ID':kwargs.get('id'),
				'SLURM_ARRAY_TASK_MIN':kwargs.get('min'),
				'SLURM_ARRAY_TASK_MAX':kwargs.get('max'),
				'SLURM_ARRAY_TASK_STEP':kwargs.get('step'),
				'SLURM_ARRAY_TASK_COUNT':kwargs.get('count'),
				'SLURM_ARRAY_TASK_SLICE':kwargs.get('slice'),
				'SLURM_ARRAY_TASK_SIZE':kwargs.get('size'),				
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


def call(*args,path=None,kwargs=None,exe=None,flags=None,cmd=None,options=None,env=None,wrapper=None,pause=None,file=None,stdin=None,stdout=None,stderr=None,process=None,processes=None,device=None,execute=False,verbose=None):
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
		file (str): Write command to file
		stdin (file): Stdinput stream to command
		stdout (file): Stdoutput to command
		stderr (file): Stderro to command
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	Returns:
		result (object): Return of commands
	'''

	def caller(args,inputs=None,outputs=None,errors=None,env=None,device=None,verbose=None):

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

		stdin = None if inputs is None else inputs if isinstance(inputs,str) else inputs.pop(0) if len(inputs)>len(args) else None
		stdout = subprocess.PIPE if outputs is None else subprocess.PIPE
		stderr = subprocess.PIPE if errors is None else subprocess.PIPE
		returncode = None

		inputs = [inputs]*len(args) if inputs is None or isinstance(inputs,str) else inputs
		outputs = [outputs]*len(args) if outputs is None or isinstance(outputs,str) else outputs
		errors = [errors]*len(args) if errors is None or isinstance(errors,str) else errors

		for arg,input,output,error in zip(args,inputs,outputs,errors):

			stdin = open(input,'r') if isinstance(input,str) else input if input is not None else stdin
			stdout = open(output,'w') if isinstance(output,str) else output if output is not None else stdout
			stderr = open(error,'w') if isinstance(error,str) else error if error is not None else stderr

			result = run(arg,stdin=stdin,stdout=stdout,stderr=stderr,env=env)

			if stdin is not None:
				stdin.close()
			if isinstance(output,str):
				stdout.close()
			if isinstance(error,str):
				stderr.close()

			stdin = result.stdout


		stdout,stderr,returncode = [],[],result.returncode
		
		if result.stdout is not None:
			for line in result.stdout:
				stdout.append(parse(line))			
				logger.log(verbose,stdout[-1])
		returncode = result.wait()

		if result.stderr is not None:
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

	inputs = [stdin,*inputs]
	outputs = stdout
	errors = stderr

	msg = '%s : %s'%(path,cmd) if path is not None else cmd
	logger.log(verbose,msg)

	if file:
		with cd(path):
			touch(file,cmd,mod=True,env=env,execute=True,verbose=False)

	if execute > 0:
		with cd(path):
			result = wrapper(*caller(args,inputs=inputs,outputs=outputs,errors=errors,env=env,device=device,verbose=verbose),env=env,device=device,verbose=verbose)

	return result

def cp(source,destination,default=None,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Copy objects from source to destination
	Args:
		source (str): Path of source object
		destination (str): Path of destination object
		default (str): Default path of source object
		env (dict[str,str]): Environmental variables for args		
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
	env = [] if env is None else env
	args = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return


def rm(path,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Remove path
	Args:
		path (str): Path to remove
		env (dict[str,str]): Environmental variables for args		
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
	env = [] if env is None else env
	args = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return



def echo(*args,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Echo arguments to command line
	Args:
		args (str,iterable[str],iterable[iterable[str]]): Arguments to pass to command line, nested iterables are piped
		env (dict[str,str]): Environmental variables for args		
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
	env = [] if env is None else env
	args = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return


def run(file,path,*args,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Run path with arguments and environmental variables
	Args:
		file (str): File
		path (str): Path of file
		args (str,iterable[str],iterable[iterable[str]]): Arguments to pass to command line, nested iterables are piped
		env (dict[str,str]): Environmental variables for args
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	'''

	exe = ['./%s'%(file)]
	flags = []
	cmd = []
	options = []
	env = [] if env is None else env	
	args = [*args]

	stdout = call(*args,path=path,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return


def touch(path,*args,mod=None,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Create file with command and environmental variables
	Args:
		path (str): Path of file
		args (str,iterable[str],iterable[iterable[str]]): Arguments to pass to command line, nested iterables are piped
		mod (str,bool): Chmod file, or boolean to make executable
		env (dict[str,str]): Environmental variables for args
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	'''

	exe = ['echo']
	flags = []
	cmd = ''.join([*['#!/bin/bash\n'],*[arg for var in env for arg in ['export %s=%s\n'%(var,env[var])]],*args])
	options = []
	env = [] if env is None else env
	args = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,stdout=path,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	chmod(path,mod=mod,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return


def chmod(path,mod=None,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Chmod file with command and environmental variables
	Args:
		path (str): Path of file
		mod (str,bool): Chmod file, or boolean to make executable
		env (dict[str,str]): Environmental variables for args
		process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']		
		processes (int): Number of processes per command		
		device (str): Name of device to submit to
		execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
		verbose (int,str,bool): Verbosity
	'''

	if mod is None:
		return

	if isinstance(mod,bool):
		mod = '+x'

	exe = ['chmod']
	flags = [mod]
	cmd = [path]
	options = []
	env = [] if env is None else env
	args = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return


def sleep(pause=None,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Sleep for pause
	Args:
		pause (int): Time to sleep
		env (dict[str,str]): Environmental variables for args		
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
	env = [] if env is None else env	
	args = []

	stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return


def sed(path,patterns,default=None,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	GNU sed replace patterns in path
	Args:
		path (str): Path of file
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		default (str): Default pattern value to sed
		env (dict[str,str]): Environmental variables for args		
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
		env = [] if env is None else env
		args = []

		stdout = call(*args,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose)

	return

def search(path,pattern,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Search for pattern in file
	Args:
		path (str): Path of file
		pattern (str): Pattern to search
		env (dict[str,str]): Environmental variables for args		
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

	
def update(path,patterns,kwargs=None,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''	
	Update path files with sed
	Args:
		path (str): Path of file
		patterns (dict[str,str]): Patterns and values to update {pattern: replacement}
		kwargs (dict): Additional keyword arguments to update
		env (dict[str,str]): Environmental variables for args		
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

	for pattern in patterns:

		value = patterns[pattern]

		if pattern in ['chdir']:
			value = join(value)
		
		elif pattern in ['dependency']:
			value = '%s:%s'%(
				':'.join(value.split(':')[:-1]) if isinstance(value,str) and value.count(':') > 0 else '',
				','.join([str(i) for i in kwargs.get('dependencies',[]) if i is not None]) if kwargs.get('dependencies') is not None else ''
				)
		
		elif pattern in ['array']:
			# pattern = int(value.split('-')[:].split(':')[0].split('%')[0])
			count = kwargs.get('count') if kwargs.get('count') is not None else 1
			step = kwargs.get('step') if kwargs.get('step') is not None else 1
			min = kwargs.get('min') if kwargs.get('min') is not None else 0
			max = kwargs.get('max') if kwargs.get('max') is not None else min + count - 1
			simultaneous = kwargs.get('simultaneous') if kwargs.get('simultaneous') is not None else int(value.split('%')[-1]) if isinstance(value,str) and value.count('%') > 0 else 100
			value = '%d-%d:%d%%%d'%(min,max,step,simultaneous)
		
		elif pattern in ['output','error']:
			value = join(split(value,directory_file=True) if value is not None else '%x.%A.%a',
					ext=(split(value,ext=True) if value is not None else 
						{'output':'stdout','error':'stderr'}.get(pattern,'log')),
					root=None)
		else:
			value = value

		patterns[pattern] = value

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
		nulls = ['chdir','array']
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


	sed(path,patterns,default=default,env=env,execute=execute,verbose=verbose)

	return


def configure(paths,pwd=None,cwd=None,patterns={},env=None,process=None,processes=None,device=None,execute=False,verbose=None):
	'''
	Configure paths for jobs with copied/updated files
	Args:
		paths (iterable[str],dict[str,object]): Relative paths of files to pwd/cwd, or dictionary {path:data} with data to update paths
		pwd (str): Input root path for files
		cwd (str): Output root path for files
		patterns (dict[str,dict[str,str]]): Patterns and values to update {path:{pattern: replacement}}
		env (dict[str,str]): Environmental variables for args		
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


def init(key,
		keys=None,
		jobs=None,args=None,paths=None,patterns=None,dependencies=None,
		pwd=None,cwd=None,pool=None,pause=None,file=None,
		env=None,process=None,processes=None,device=None,execute=None,verbose=None):
		'''
		Process job commands as tasks to command line
		Args:
			task (dict[str,str]): Job task
			key (str): Name of job
			keys (dict[str,dict[str]]): Jobs with task names and arguments
			jobs (str,dict[str,str]): Submission script, or {key:job}
			args (dict[str,str],dict[str,dict[str,str]]): Arguments to pass to command line, either {arg:value} or {key:{arg:value}}
			paths (dict[str,object],dict[str,dict[str,object]]): Relative paths of files to pwd/cwd, with data to update paths {path:data} or {key:{path:data}}
			patterns (dict[str,dict[str,str]],dict[str,dict[str,dict[str,str]]]): Patterns to update files {path:{pattern:replacement}} or {key:{path:{pattern:replacement}}
			dependencies (iterable[str,int],dict[str,iterable[str,int]]): Dependences of previous jobs to job [dependency] or {key:[dependency]}
			pwd (str,dict[str,str]): Input root path for files, either path, or {key:path}
			cwd (str,dict[str,str]): Output root path for files, either path, or {key:path}
			pool (int): Number of subtasks in a pool per task (parallelized with processes number of parallel processes)
			pause (int,str): Time to sleep after call		
			file (str): Write command to file		
			process (str): Type of process instance, either in serial, in parallel, or as an array, allowed strings in ['serial','parallel','array']
			processes (int): Number of processes per command		
			device (str): Name of device to submit to
			execute (boolean,int): Boolean whether to issue commands, or int < 0 for dry run
			verbose (int,str,bool): Verbosity
		Returns:
			task (dict[str,str]): Task for job
		'''

		pool = 1 if pool is None else pool
		verbose = False
		execution = True if execute == -1 else execute
		execute = False if execute == -1 else execute

		indices = [subkey for subkey in keys if cwd[subkey] == cwd[key]]
		size = len(indices)

		if size > 1 and process not in ['serial']:
			index = indices.index(key)
			mod = index%pool			
			slice = pool
			
			id = index//pool
			min = 0
			max = size//pool + (size%pool>0) - 1
			step = 1
			count = (max - min + 1)//step
			
		else:
			index = None
			mod = None
			slice = None

			id = None
			min = None
			max = None
			step = None
			count = None

		if size > 1:
			path = indices.index(key)
		else:
			path = None
		path = join(path,root=cwd[key])

		exe = jobs[key]
		flags = []
		cmd = []
		options = []
		env = []


		task = {
			'key':key,
			'path':path,
			'pwd':pwd[key],
			'cwd':cwd[key],
			'job':jobs[key],
			'cmd':None,
			'env':None,
			'pool':pool,
			'size':size,
			'index': index,
			'mod':mod,
			'slice':slice,
			'id': id,
			'min':min,
			'max':max,
			'step': step,			
			'count': count,
			'paths':paths[key],
			'patterns':patterns[key],
			'dependencies':dependencies[key],
			}

		cmd,env = command(args[key],task,exe=exe,flags=flags,cmd=cmd,options=options,env=env,process=process,processes=processes,device=device,execute=execution,verbose=verbose)

		configure(paths[key],pwd=pwd[key],cwd=path,patterns=patterns[key],env=env,process=process,processes=processes,device=device,execute=execution,verbose=verbose)

		task['cmd'] = cmd
		task['env'] = env

		logger.log(verbose,'Job %s'%(key))

		return task

def callback(value,key,values):
	'''
	Callback for task processing
	Args:
		value (dict): Task arguments
		key (str): Task name
		values (dict): Tasks
	'''
	values[key] = value
	return

def submit(jobs={},args={},paths={},patterns={},dependencies=[],pwd='.',cwd='.',pool=None,pause=None,file=None,env=None,process=None,processes=None,device=None,execute=False,verbose=None):
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
		file (str): Write command to file		
		env (dict[str,str]): Environmental variables for args		
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

	keys = intersection(keys,cwd,sort=None)

	execution = True if execute == -1 else execute
	execute = False if execute == -1 else execute
	
	tasks = []
	results = []
	keys = {key:{} for key in keys}

	directories = set((cwd[key] for key in cwd))
	for directory in directories:
		if exists(join(directory,file)):
			result = run(file,path=directory,process=None,processes=None,device=None,execute=execute,verbose=verbose)
			results.append(result)
			return results

	iterable = [key for key in keys]
	kwds = dict(
		keys=keys,
		jobs=jobs,args=args,paths=paths,patterns=patterns,dependencies=dependencies,
		pwd=pwd,cwd=cwd,pool=pool,pause=pause,file=file,
		env=env,process=process,processes=processes,device=device,execute=execute,verbose=verbose
	)
	callback_kwds = {'values':keys}

	parallelize = Pooler(processes)

	parallelize(
		iterable,init,
		callback=callback,
		kwds=kwds,callback_kwds=callback_kwds
		)

	if process in ['serial']:
		def boolean(task,tasks):
			value = task['mod'] in [0,None]
			return value
		def updates(task,tasks):
			attr = 'path'
			value = task['path']
			task[attr] = value
			return

	elif process in ['parallel']:
		def boolean(task,tasks):
			value = task['mod'] in [0,None]
			return value
		def updates(task,tasks):
			attr = 'path'
			value = task['cwd']
			task[attr] = value
			return

	elif process in ['array']:
		def boolean(task,tasks):
			value = task['index'] in [0,None]			
			return value
		def updates(task,tasks):
			attr = 'path'			
			value = task['cwd']
			task[attr] = value
			return

	else:
		def boolean(task,tasks):
			value = task['mod'] in [0,None]			
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

		result = call(*cmd,env=env,path=path,pause=pause,file=file,process=None,processes=None,device=None,execute=execute,verbose=verbose)

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
			file (str): Write command to file			
			env (dict[str,str]): Environmental variables for args		
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
