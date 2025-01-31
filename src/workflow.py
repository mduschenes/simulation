#!/usr/bin/env python

# Import python modules
import os,sys,argparse,subprocess,shlex,itertools,json,io,re,signal,errno,time as timer
from copy import deepcopy as copy

from src.logger import Logger

integers = (int,)
floats = (float,)
scalars = (*integers,*floats,str,type(None),)
iterables = (list,tuple,set,)
streams = (io.RawIOBase,)
delimiter = '.'
ifs = ' '
pipe = '|'

logger = Logger()
info = 100
debug = 0


class Null(object):
	pass
null = Null()
nulls = (Null,)

class Dict(dict):
	'''
	Dictionary subclass with dictionary elements explicitly accessible as class attributes
	Args:
		args (dict): Dictionary elements
		kwargs (dict): Dictionary elements
	'''
	def __init__(self,*args,**kwargs):
		for arg in args:
			if isinstance(arg,dict):
				kwargs.update(arg)
			elif isinstance(arg,iterables) and all(isinstance(i,iterables) and len(i)==2 for i in arg):
				kwargs.update(dict(arg))

		for key in kwargs:
			if isinstance(kwargs[key],dict) and all(isinstance(attr,str) for attr in kwargs[key]):
				kwargs[key] = Dict(kwargs[key]) if not isinstance(kwargs[key],Dict) else kwargs[key]

		super().__init__(*args,**kwargs)
		self.__dict__ = self
		return

	def __hash__(self):
		return hash(tuple((attr,getattr(self,attr)) for attr in self))

	def __eq__(self,other):
		return hash(self) == hash(other)

class Popen(object):
	'''
	Null Popen class
	'''
	def __init__(self,*args,stdin=None,stdout=None,stderr=None,env=None,**kwargs):
		self.args = args
		self.env = env
		self.stdout = stdout
		self.stderr = stderr
		self.returncode = None
		return

	def wait(self,*args,**kwargs):
		returncode = self.returncode
		return returncode

class timeout(object):
	'''
	Timeout class for function
	Args: 
		time (int,float,bool): timeout time in seconds, or boolean for infinite time
		message (str): message once timeout
		default (object): Default return of function if timeout
	'''

	def __init__(self,time=None,message=None,default=None):
		self.time = time if (time is True or time is False) else float(time) if time is not None else self.timeout
		self.message = message if message is not None else None
		self.default = default
		self.signal = signal.SIGALRM
		self.alarm = 0
		return

	def timer(self,time):
		signal.setitimer(signal.ITIMER_REAL,time)
		return
	def handler(self, signum, frame):
		raise self.error(self.message) if self.message else self.error
	def start(self):
		if self.time is True or self.time is False:
			return
		signal.signal(self.signal,self.handler)
		self.timer(self.time)
		return
	def stop(self):
		if self.time is True or self.time is False:
			return		
		self.timer(self.alarm)
		return
	
	def __enter__(self):
		self.start()
		return

	def __exit__(self,type,value,traceback):
		self.stop()
		return

	def __call__(self,func):
		def wrapper(*args, **kwargs):
			self.start()
			try:
				result = func(*args, **kwargs)
			finally:
				self.stop()
			return result
		return wrapper

	@classmethod
	@property
	def timeout(cls):
		time = 1e-1
		return time

	class error(Exception):
		pass



def sleep(time):
	'''
	Sleep process
	Args:
		time (int,float): Time to sleep in seconds
	'''
	if time is None:
		return
	try:
		timer.sleep(time)
	except:
		pass
	return

def call(*args,path=None,file=None,wrapper=None,env=None,stdin=None,stdout=None,stderr=None,shell=None,time=None,execute=None,verbose=None):
	'''
	Submit call to command line of the form args
	Args:
		args (iterable[str]): Arguments to pass to command line
		path (str): Path to call from
		file (str): Write command to file
		wrapper (callable): Wrapper for output, with signature wrapper(stdout,stderr,returncode,env=None,shell=None,msg=None,time=None,execute=None,verbose=None)
		env (dict[str,str]): Environmental variables for args		
		stdin (file): Stdinput stream to command
		stdout (file): Stdoutput to command
		stderr (file): Stderr to command
		shell (bool) : Use shell subprocess
		time (int,float): Timeout duration in seconds
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): Verbosity
	Returns:
		result (object): Return of commands
	'''

	def caller(args,stdin=None,stdout=None,stderr=None,env=None,shell=None,time=None,msg=None,execute=None,verbose=None):

		def run(args,stdin=None,stdout=None,stderr=None,env=None,shell=None):
			env = {**environ(),**env} if env is not None else None
			args = [ifs.join(args)] if shell else shlex.split(args)
			try:
				result = subprocess.Popen(args,stdin=stdin,stdout=stdout,stderr=stderr,env=env,shell=shell)
			except (OSError,FileNotFoundError) as exception:
				result = Popen(args,stdin=stdin,stdout=stdout,stderr=stderr,env=env,shell=shell)
				logger(exception,verbose=verbose)
			return result

		def process(obj):
			parse = lambda obj: os.path.expandvars(os.path.expanduser(obj)).replace('~',os.environ['HOME'])
			if isinstance(obj,str):
				obj = parse(obj)
			elif isinstance(obj,iterables):
				obj = [parse(i) if isinstance(i,str) else [parse(j) for j in i] for i in obj]
			return obj

		def wrap(stdout,stderr,returncode):
			stdout = '\n'.join(stdout) if stdout is not None else stdout
			stderr = '\n'.join(stderr) if stderr is not None else stderr
			returncode =  returncode if returncode is not None else returncode
			return stdout,stderr,returncode

		def parse(obj):
			try:
				obj = obj.strip().decode('utf-8')
			except:
				obj = str(obj)
			return obj

		stdpipe = subprocess.PIPE

		stdin = None if stdin is None else stdin
		stdout = stdout if stdout is None else stdout
		stderr = stderr
		returncode = None

		inputs = [stdin,*[None]*(len(args)-1)]
		outputs = [*[stdpipe]*(len(args)-1),stdout if stdout is not None else stdpipe]
		errors = [*[stdpipe]*(len(args)-1),stderr if stderr is not None else stdpipe]

		options = dict(
			execute=True,
			verbose=False
			)

		for index,(arg,input,output,error) in enumerate(zip(args,inputs,outputs,errors)):
			if isinstance(output,str):
				mkdir(output,**options)
			if isinstance(error,str):
				mkdir(error,**options)

			stdin = open(input,'r') if isinstance(input,str) else input if input is not None else stdin
			stdout = open(output,'w') if isinstance(output,str) else output if output is not None else stderr
			stderr = open(error,'w') if isinstance(error,str) else error if error is not None else stdpipe

			arg = process(arg)

			result = run(arg,stdin=stdin,stdout=stdout,stderr=stderr,env=env,shell=shell)

			if isinstance(stdin,streams):
				stdin.close()
			if isinstance(stdout,streams):
				stdout.close()
			if isinstance(error,streams):
				stderr.close()

			stdin = result.stdout


		if not args:
			result = Popen()
			stdout,stderr,returncode = None,None,None
		else:
			stdout,stderr,returncode = [],[],result.returncode
		
		if result.stdout is not None:
			try:
				for line in result.stdout:
					stdout.append(parse(line))		
					logger(stdout[-1],verbose=verbose)
			except:
				stdout.append(parse(result.stdout))
				logger(stdout[-1],verbose=verbose)
		

		try:
			returncode = result.wait()
		except:
			returncode = -1
		
		if result.stdout is not None:
			result.stdout.flush()

		if result.stderr is not None:
			try:
				for line in result.stderr:	
					stderr.append(parse(line))
					if returncode is not None:
						logger('{msg}\n{line}'.format(msg=msg,line=stderr[-1]),verbose=True)
			except:
				stderr.append(parse(result.stderr))
				logger('{msg}\n{line}'.format(msg=msg,line=stderr[-1]),verbose=True)

		stdout,stderr,returncode = wrap(stdout,stderr,returncode)

		sleep(time)

		return stdout,stderr,returncode

	if not callable(wrapper):
		def wrapper(stdout,stderr,returncode,env=None,shell=None,time=None,msg=None,execute=None,verbose=None):
			result = stdout
			return result

	def parser(*args,stdin=None,stdout=None,stderr=None):

		if all(arg is None for arg in args):
			args,stdin,stdout,stderr = None,stdin,stdout,stderr
			return args,stdin,stdout,stderr

		args = pipe.join([str(arg) for arg in args if arg]) 

		symbols = ['>','<']

		for symbol in symbols:
			if not args.count(symbol):
				continue
			
			index = args.index(symbol)

			if symbol in ['>']:
				args,stdin,stdout,stderr = args[:index],stdin,args[index+1:],stderr
			elif symbol in ['<']:
				args,stdin,stdout,stderr = args[:index],args[index+1:],stdout,stderr
			else:
				args,stdin,stdout,stderr = args,stdin,stdout,stderr

			break

		args = args.strip() if isinstance(args,str) else args
		stdin = stdin.strip() if isinstance(stdin,str) else stdin
		stdout = stdout.strip() if isinstance(stdout,str) else stdout
		stderr = stderr.strip() if isinstance(stderr,str) else stderr

		assert not any(args.count(symbol) for symbol in symbols)

		args = [arg.strip() for arg in args.split(pipe) if arg]

		return args,stdin,stdout,stderr

	args,stdin,stdout,stderr = parser(*args,stdin=stdin,stdout=stdout,stderr=stderr)
	result = None

	formats = dict(
		path='',
		args=(' {pipe} '.format(pipe=pipe)).join([arg.replace('\n','\\n') for arg in args]),
		stdin=stdin,stdout=stdout
		)
	if isinstance(stdin,str) and isinstance(stdout,str):
		string = '{path}{args} < {stdin} > {stdout}'
	elif isinstance(stdin,str):
		string = '{path}{args} < {stdin}'
	elif isinstance(stdout,str):
		string = '{path}{args} > {stdout}'
	else:
		string = '{path}{args}'

	if verbose:
		msg = string.format(**{**formats,**dict(path='{path}{space}'.format(path='{path}'.format(path=path) if path else '',space='$> ' if path else ifs))})
		logger(msg,verbose=verbose)

	if file:
		with cd(path):
			data = [
				*['#!/bin/bash\n'],
				*['export {key}={value}\n'.format(key=key,value=env[key]) for key in (env if env is not None else {})],
				*[string.format(**formats)]
				]
			options = dict(execute=True,verbose=False)
			write(file,data,**options)

			data = 'chmod +x {file}'.format(file=file)
			options = dict(execute=True,verbose=False)
			call(data,**options)

	if execute:
		with cd(path):
			msg = string.format(**formats)
			result = wrapper(*caller(args,stdin=stdin,stdout=stdout,stderr=stderr,env=env,shell=shell,msg=msg,time=time,verbose=verbose),env=env,shell=shell,msg=msg,time=time,verbose=verbose)

	return result


class argparser(argparse.ArgumentParser):
	def __init__(self,arguments=None,wrappers=None):
		'''
		Parse command line arguments
		Args:
			arguments (str,iterable[str],dict[str,dict[str,object]]): Command line arguments {argument:{option:value}}
			wrappers (dict[str,[str,callable]]: Wrappers of arguments, either string for argument name, or callable(kwarg,wrappers,kwargs)
		'''

		# TODO: Allow for non-string types of iterable of values parsed by action (comma-separated values)

		class action(argparse.Action):
			def __call__(self, parser, namespace, values, option_string=None):
				_values = []
				separator = ','
				iterable = isinstance(values,list)
				if not iterable:
					values = [values]
				for value in values:
					for val in str(value).split(separator):
						_values.append(self.type(val))
					if iterable:
						setattr(namespace,self.dest,_values)
				if not iterable:
					setattr(namespace,self.dest,_values[-1])

				return

		defaults = {
			'action':action
		}

		nulls = {
			'action':['type','nargs','default']
		}

		default = lambda argument: ({
			'help':argument.replace('--','').capitalize(),
			'type':str,
			'nargs':'?',
			'default':None,
			})

		if arguments is None:
			arguments = '--args'
		if isinstance(arguments,str):
			arguments = [arguments]
		if not isinstance(arguments,dict):
			arguments = {
				'--{argument}'.format(argument=argument.replace('--','')):{**default(argument)}
				for argument in arguments
			}
		else:
			arguments = {
				'--{argument}'.format(argument=argument.replace('--','')):{
					**default(argument),
					**(arguments[argument] if isinstance(arguments[argument],dict) else 
					   dict(default=arguments[argument]))
					}
				for argument in arguments
			}

		if wrappers is None:
			wrappers = {}

		super().__init__()

		for i,argument in enumerate(arguments):


			name = '{name}'.format(name=argument.replace('--',''))
			options = {option: arguments[argument][option] for option in arguments[argument]}

			if options.get('action') is None:
				for null in nulls:
					if null in options:
						for option in nulls[null]:
							options.pop(option,None);

				options.update({option: options.get(option,defaults[option]) for option in defaults if option not in options})
				options.update({
					**{option:'?' if options.get(option) not in ['*','+'] or i>0 else '*' for option in ['nargs'] if option in options},
					**{option: argparse.SUPPRESS for option in ['default'] if option in options}
					})
				names = [name]
				self.add_argument(*names,**options)

			name = '--{argument}'.format(argument=argument.replace('--',''))
			options = {option: arguments[argument][option] for option in arguments[argument]}
			
			for null in nulls:
				if null in options:
					for option in nulls[null]:
						options.pop(option,None);

			options.update({option: options.get(option,defaults[option]) for option in defaults if option not in options})
			options.update({'dest':options.get('dest',argument.replace('--',''))})
			names = [name]
			self.add_argument(*names,**options)

		kwargs,args = self.parse_known_args()

		kwargs = {**dict(**vars(kwargs))}

		for kwarg in wrappers:
			name = kwarg.replace('--','')
			func = wrappers[kwarg] if callable(wrappers[kwarg]) else lambda kwarg,wrappers,kwargs: kwargs[wrappers[kwarg].replace('--','')]
			kwargs[name] = func(kwarg,wrappers,kwargs)

		self.args = args
		self.kwargs = kwargs

		return

	def __iter__(self):
		return self.args.__iter__()

	def __getitem__(self,item):
		return self.kwargs[item]

	def __len__(self):
		return len(self.args)+len(self.kwargs)

	def keys(self):
		return self.kwargs.keys()

	def values(self):
		return self.kwargs.values()

def join(*paths,ext=None,root=None,abspath=None):
	'''
	Join paths
	Args:
		paths (str): path
		ext (str): join ext
		root (str): Root path to insert at beginning of path if path does not already start with root
		abspath (bool): Return absolute path
	Returns:
		path (str): Joined path
	'''
	try:
		path = os.path.join(*(path for path in paths if path is not None))
		path = path if ext is None else delimiter.join([path,ext])
	except:
		path = None

	if path is not None and root is not None:
		if not dirname(path,abspath=True).startswith(dirname(root,abspath=True)):
			paths = [root,path]
			paths = [path for path in paths if path not in ['',None]]
			path = os.path.join(*paths)
	elif path is None and root is not None:
		path = root

	if path is not None:
		path = os.path.expandvars(os.path.expanduser(path))
	if path is not None and abspath:
		path = os.path.abspath(path)

	return path

def split(path,ext=None,root=None,abspath=None):
	'''
	Split paths
	Args:
		path (str): path
		ext (bool): split ext
		root (str): Root path to insert at beginning of path if path does not already start with root		
		abspath (bool): Return absolute path	
	Returns:
		paths (iterable[str]): Split paths
	'''

	if abspath:
		path = os.path.abspath(path)

	try:
		paths = path.split(os.sep) if path is not None else None
		paths = paths if not ext else [*paths[:-1],delimiter.join(paths[-1].split(delimiter)[:-1]),delimiter.join(paths[-1].split(delimiter)[-1:])] if path is not None else None
	except:
		paths = None

	if root is not None:
		root = split(root,abspath=abspath)
		while root[-1] != paths[0]:
			root,paths = root[:-1],[root[-1],*paths]

	return paths

def merge(*paths,string=None):
	'''
	Merge paths
	Args:
		paths (str): path
		string (str): merge string
	Returns:
		path (str): Merged path
	'''

	path = join(*(path for path in paths[:-1] if path is not None),join(split(paths[-1],ext=True)[:-1],string,split(paths[-1],ext=True)[-1]) if paths[-1] is not None else None)

	return path

def contains(string,pattern):
	'''
	Search for pattern in string
	Args:
		string (str): String to search
		pattern (str): Pattern to search
	Returns:
		boolean (bool): String contains pattern
	'''
	replacements = {'\\':'\\\\','.':'\\.'}
	for replacement in replacements:
		pattern = pattern.replace(replacement,replacements[replacement])
	
	boolean = bool(re.search(pattern,string))

	return boolean

def substitute(string,patterns):
	'''
	Substitute pattern in string
	Args:
		string (str): String to search
		patterns (dict[str,str]): Patterns to replace
	Returns:
		string (string): Substituted string
	'''
	replacements = {'\\':'\\\\','.':'\\.'}
	
	for pattern in patterns:
		strings = patterns[pattern]
		for replacement in replacements:
			pattern = pattern.replace(replacement,replacements[replacement])
		string = re.sub(pattern,strings,string)

	return string

def exists(path):
	'''
	Check if path exists
	Args:
		path (str): path
	Returns:
		exists (bool): Path exists
	'''

	try:
		exists = os.path.exists(path)
	except:
		exists = False

	return exists

def dirname(path,abspath=False,delimiter=delimiter):
	'''
	Return directory name of path
	Args:
		path (str): path
		abspath (bool): Return absolute path of directory
		delimiter (str): Delimiter to separate file name from extension		
	Returns:
		path (str): Directory name of path
	'''

	# TODO: Include all extensions / find method of determining if path is directory or file

	exts = [
		'py','ipynb',
		'tmp',
		'cpp','o','out','err','obj',
		'csv','txt',
		'npy','npz',
		'pickle','pkl',
		'json',
		'hdf5','h5','ckpt',
		'sh',
		'git',
		'pdf','mk',
		'tex','sty','aux','auxlock','bbl','bib','blg','snm','toc','nav','tikz',
		'docx','xlsx','pptx','doc','xls','ppt',
		'slurm','lsf',
		'ini','config',
		'mplstyle',
		'conf','log',
		'stdout','stderr'
		]

	if path is None:
		directory = path
		return directory

	file,ext = os.path.splitext(path)
	ext = delimiter.join(ext.split(delimiter)[1:])
	directory = os.path.dirname(path)

	if directory in ['']:
		if ext not in exts:
			directory = path
	elif ext not in exts:
		directory = path
	directory = os.path.expandvars(os.path.expanduser(directory))
	if abspath:
		directory = os.path.abspath(os.path.expandvars(os.path.expanduser(directory)))

	path = directory

	return path

def echo(*strings,execute=None,verbose=None):
	'''
	Echo strings
	Args:
		strings (str): strings
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): verbosity
	'''

	args = ['echo',*strings]

	call(*args,execute=execute,verbose=verbose)

	return

class cd(object):
	'''
	Class to safely change paths and return to previous path
	Args:
		path (str): Path to change to
	'''
	def __init__(self,path):
		path = os.path.abspath(os.path.expandvars(os.path.expanduser(path))) if path is not None else None
		options = dict(
			execute=True,
			verbose=False
			)
		mkdir(path,**options)
		self.path = path
		return

	def __enter__(self):
		self.cwd = cwd()
		try:
			os.chdir(self.path)
		except:
			pass
		return

	def __exit__(self,etype, value, traceback):
		os.chdir(self.cwd)
		return

def cwd(*args,**kwargs):
	'''
	Get current directory
	Args:
		args (iterable): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		path (str): Current directory
	'''
	path = os.getcwd()
	return path

def environ():
	'''
	Get environmental variables
	Returns:
		environ (dict): Environmental variables
	'''

	return os.environ

def mkdir(path,execute=None,verbose=None):
	'''
	Make path
	Args:
		path (str): path
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): verbosity
	'''
	
	if not execute:
		return

	path = dirname(path)

	logger('mkdir {path}'.format(path=path),verbose=verbose)

	if path not in ['',None] and not exists(path):
		os.makedirs(path)

	return

def rm(path,execute=None,verbose=None):
	'''
	Remove path
	Args:
		path (str): path
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): verbosity		
	'''

	if not execute:
		return

	logger('rm {path}'.format(path=path),verbose=verbose)

	try:
		os.remove(path)
	except Exception as exception:
		try:
			os.rmdir(path)
		except Exception as exception:
			pass

	return

def touch(path,execute=None,verbose=None):
	'''
	Make path
	Args:
		path (str): path
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): verbosity
	'''

	path = os.path.abspath(os.path.expandvars(os.path.expanduser(path))) if path is not None else None

	args = ['touch',path]

	if not exists(path):
		call(*args,execute=execute,verbose=verbose)

	return

def cp(source,destination,execute=None,verbose=None):
	'''
	Copy source to destination
	Args:
		source (str): path
		destination (str): path
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): verbosity
	'''

	source = os.path.abspath(os.path.expandvars(os.path.expanduser(source))) if source is not None else None
	destination = os.path.abspath(os.path.expandvars(os.path.expanduser(destination))) if destination is not None else None

	args = ['cp','-rf',source,destination]

	if exists(source):
		call(*args,execute=execute,verbose=verbose)

	return


def nonempty(path):
	'''
	Check if path is non-empty
	Args:
		path (str): path
	Return:
		boolean (bool): Path is non-empty
	'''

	try:
		boolean = bool(os.stat(path).st_size)
	except:
		boolean = False

	return boolean


def basename(path,**kwargs):
	'''
	Get base file name from path
	Args:
		path (str): Path
		kwargs (dict): Additional path keyword arguments
	Returns:
		file (str): Base filename
	'''	
	return os.path.basename(path)


def basedir(path,**kwargs):
	'''
	Get base path name from path
	Args:
		path (str): Path
		kwargs (dict): Additional path keyword arguments
	Returns:
		file (str): Base filename
	'''	
	return basename(dirname(path))


def load(path,default=None,wrapper=None,execute=None,verbose=None):
	'''
	Load data from path
	Args:
		path (str): path
		default (object): default data
		wrapper (callable): wrapper to load data with signature wrapper(data)
		execute (boolean,int): Boolean whether to call commands		
		verbose (int,str,bool): Verbosity
	Returns:
		data (object): data
	'''
	
	logger('Load: {path}'.format(path=path),verbose=verbose)
	
	try:
		with open(path,'r') as obj:
			data = json.load(obj)
	except:
		data = default
	if wrapper is not None:
		try:
			data = wrapper(data)
		except:
			pass
	
	return data

def dump(path,data,wrapper=None,execute=None,verbose=None):
	'''
	Dump data to path
	Args:
		path (str): path
		data (object): data
		wrapper (callable): wrapper to load data with signature wrapper(data)
		execute (boolean,int): Boolean whether to call commands		
		verbose (int,str,bool): Verbosity
	'''	
	
	logger('Dump: {path}'.format(path=path),verbose=verbose)
	
	if wrapper is not None:
		try:
			data = wrapper(data)
		except:
			pass	
	try:
		with open(path,'w') as obj:
			json.dump(data,obj)
	except:
		pass
	
	return


def read(path,default=None,wrapper=None,execute=None,verbose=None):
	'''
	Load data from path
	Args:
		path (str): path
		default (object): default data
		wrapper (callable): wrapper to load data with signature wrapper(data)
		execute (boolean,int): Boolean whether to call commands		
		verbose (int,str,bool): Verbosity
	Returns:
		data (object): data
	'''
	
	logger('Load: {path}'.format(path=path),verbose=verbose)
	
	try:
		with open(path,'r') as obj:
			data = obj.readlines()
	except:
		data = default
	if wrapper is not None:
		try:
			data = wrapper(data)
		except:
			pass
	
	return data

def write(path,data,wrapper=None,execute=None,verbose=None):
	'''
	Dump data to path
	Args:
		path (str): path
		data (object): data
		wrapper (callable): wrapper to load data with signature wrapper(data)
		execute (boolean,int): Boolean whether to call commands		
		verbose (int,str,bool): Verbosity
	Returns:
		data (object): data
	'''	
	
	logger('Dump: {path}'.format(path=path),verbose=verbose)
	
	if wrapper is not None:
		try:
			data = wrapper(data)
		except:
			pass	
	try:
		with open(path,'w') as obj:
			obj.writelines(data)
	
	except:
		pass
	return


def permutations(*iterables,repeat=None):
	'''
	Get product of permutations of iterables
	Args:
		iterables (iterable[iterables],iterable[int]): Iterables to permute, or iterable of int to get all permutations of range(int)
	Returns:
		iterables (generator[tuple]): Generator of tuples of all permutations of iterables
	'''
	
	if all(isinstance(i,integers) for i in iterables):
		iterables = (range(i) for i in iterables)
	
	if repeat is None:
		repeat = 1

	return itertools.product(*iterables,repeat=repeat)

def permuter(dictionary,groups=None,filters=None,func=None,ordered=True):
	'''
	Get all combinations of values of dictionary of lists

	Args:
		dictionary (dict): dictionary of keys with lists of values to be combined in all combinations across lists
		groups (list,None): List of lists of groups of keys that should not have their values permuted in all combinations, 
			but should be combined in sequence key wise. 
			For example groups = [[key0,key1]], where 
			dictionary[key0] = [value_00,value_01,value_02],
			dictionary[key1] = [value_10,value_11,value_12], 
			then the permuted dictionary will have key0 and key1 keys with only pairwise values of 
			[{key0:value_00,key1:value_10},{key0:value_01,key1:value_11},{key0:value_02,key1:value_12}].
		filters (callable): Function with signature filters(dictionaries) -> dictionaries to parse allowed dictionaries
		func (callable): Function with signature func(dictionaries) to modify dictionaries in place
		ordered (bool): Boolean on whether to return dictionaries with same ordering of keys as dictionary
	Returns:
		dictionaries (iterable[dict]) : Iterable of dictionaries with all combinations of lists of values in dictionary
	'''		
	def indexer(keys,values,groups):
		'''
		Get lists of values for each group of keys in groups
		'''
		groups = copy([group for group in groups if any(key in keys for key in group)]) if groups is not None else []
		inds = [[keys.index(key) for key in group if key in keys] for group in groups]		
		N = len(groups)
		groups.extend([[key] for key in keys if all([key not in group for group in groups])])
		inds.extend([[keys.index(key) for key in group if key in keys] for group in groups[N:]])
		values = [[values[j] for j in i] for i in inds]
		return groups,values

	def zipper(keys,values): 
		'''
		Get list of dictionaries with keys, based on list of lists in values, retaining ordering in case of grouped values
		'''
		return [{key:copy(iterable) for key,iterable in zip(keys,value)} for value in zip(*values)]

	def unzipper(dictionary):
		'''
		Zip keys of dictionary of list, and values of dictionary as list
		'''
		keys, values = zip(*((key,value) for key,value in dictionary.items() if value is not None))
		return keys,values

	def permute(dictionaries): 
		'''
		Get all list of dictionaries of all permutations of sub-dictionaries
		'''
		return ({key:dictionary[key] for dictionary in dicts for key in dictionary} for dicts in permutations(*dictionaries))

	def retriever(keys,values,groups):
		'''
		Get values of permuted nested dictionaries in values.
		Recurse permute until values are lists and not dictionaries.
		'''
		keys,values = list(keys),list(values)
		for i,(key,value) in enumerate(zip(keys,values)):
			if isinstance(value,dict):
				if isinstance(groups,dict):
					group = groups.get(key,group)
				else:
					group = groups
				values[i] = permuter(value,groups=group) 
		return keys,values


	if dictionary in [None,{}]:
		return [{}]

	# Get list of all keys from dictionary, and list of lists of values for each key
	keys,values = unzipper(dictionary)


	# Get values of permuted nested dictionaries in values
	keys,values = retriever(keys,values,groups)

	# Retain ordering of keys in dictionary
	keys_ordered = keys
	
	# Get groups of keys based on groups and get lists of values for each group
	keys,values = indexer(keys,values,groups)

	# Zip keys with lists of lists in values into list of dictionaries
	dictionaries = [zipper(key,value) for key,value in zip(keys,values)]

	# Get all permutations of list of dictionaries into one list of dictionaries with all keys
	dictionaries = permute(dictionaries)

	# Filter allowed dictionaries
	dictionaries = filters(dictionaries) if filters is not None else dictionaries

	# Retain original ordering of keys if ordered is True
	dictionaries = list(dictionaries)
	if ordered:
		for i in range(len(dictionaries)):
			dictionaries[i] = {key: dictionaries[i][key] for key in keys_ordered}

	# Modify allowed dictionaries
	if func is not None:
		func(dictionaries)

	return dictionaries


def seeder(seed=None,size=None,data=None,**kwargs):
	'''
	Generate prng key
	Args:
		seed (int,array,Key): Seed for random number generation or random key for future seeding
		size(bool,int): Number of splits of random key
		data (bool): Return key data
		kwargs (dict): Additional keyword arguments for seeding
	Returns:
		key (key,list[key]): Random key
	'''	

	# TODO merge random seeding for different numpy backends (jax vs autograd)

	if seed is None or isinstance(seed,integers):
		seed = [seed]
	else:
		seed = [*seed]

	if size:
		key = [*seed]*size
	else:
		key = seed

	if data:
		key = [*key]

	return key

def search(iterable,index=[],shape=[],returns=None,items=None,types=(list,),exceptions=()):
	'''
	Search of iterable, returning keys and indices of keys
	Args:
		iterable (iterable): Nested iterable
		index (iterable[int,str]): Index of key
		shape (iterable[int]): Shape of iterable
		returns (bool,str): Returns of search, 
			None returns item, True returns index,shape,item, False returns None, 
			allowed strings (.delimited) for combinations of ['index','shape','item']
		types (type,tuple[type]): Allowed types to be searched
		exceptions (type,tuple[type]): Disallowed types to be searched
	Yields:
		index (iterable[int,str]): Index of item
		shape (iterable[iterable[int]]): Shape of iterable at index
		item (iterable): Iterable key
	'''
	def returner(index,shape,item,returns=None):
		if returns is None:
			yield item
		elif returns is True:
			yield (index,shape,item)
		elif returns is False:
			return None
		elif returns in ['index']:
			yield index
		elif returns in ['shape']:
			yield shape
		elif returns in ['item']:
			yield item
		elif returns in ['index.shape']:
			yield (index,shape)
		elif returns in ['index.item']:
			yield (index,item)
		elif returns in ['shape.item']:
			yield (shape,item)
		elif returns in ['index.shape.item']:
			yield (index,shape,item)

	dictionaries = (dict,)
	items = [items] if (items is not None) and isinstance(items,scalars) else items
	if (not isinstance(iterable,types)) or (isinstance(iterable,exceptions)) or (items and isinstance(iterable,types) and all(item in iterable for item in items)):
		
		if items:
			if (not isinstance(iterable,types)) or (isinstance(iterable,exceptions)):
				return
			elif isinstance(iterable,dictionaries):
				item = [iterable[item] for item in items]
			else:
				item = items
		else:
			item = iterable

		yield from returner(index,shape,item,returns=returns)


	if (isinstance(iterable,types)) and (not isinstance(iterable,exceptions)):
		for i,item in enumerate(iterable):
			if isinstance(iterable,dictionaries):
				i,item = item,iterable[item]
			size = len(iterable)					
			yield from search(item,index=[*index,i],shape=[*shape,size],
				returns=returns,items=items,types=types,exceptions=exceptions)

def setter(iterable,keys,delimiter=delimiter,default=None):
	'''
	Set nested value in iterable with nested keys
	Args:
		iterable (dict): dictionary to be set in-place with value
		keys (dict,tuple): Dictionary of keys of delimiter separated strings, or tuple of string for nested keys, and values to set 
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string keys into list of nested keys
		default (callable,None,bool,iterable): Callable function with signature default(key_iterable,key_keys,iterable,keys) to modify value to be updated based on the given dictionaries, or True or False to default to keys or iterable values, or iterable of allowed types
	'''

	types = (dict,)

	if (not isinstance(iterable,types)) or (not isinstance(keys,types)):
		return

	# Setup default func as callable
	if default is None:
		func = lambda key_iterable,key_keys,iterable,keys: keys.get(key_keys)
	elif default is True:
		func = lambda key_iterable,key_keys,iterable,keys: keys.get(key_keys)
	elif default is False:
		func = lambda key_iterable,key_keys,iterable,keys: iterable.get(key_iterable,keys.get(key_keys))
	elif default in ['none','None']:
		func = lambda key_iterable,key_keys,iterable,keys: keys.get(key_keys) if iterable.get(key_iterable,keys.get(key_keys)) is None else iterable.get(key_iterable,keys.get(key_keys))
	elif default in ['replace']:
		func = lambda key_iterable,key_keys,iterable,keys: keys.get(key_keys)	
	elif not callable(default):
		instances = tuple(default)
		def func(key_iterable,key_keys,iterable,keys,instances=instances): 
			i = iterable.get(key_iterable,keys.get(key_keys))
			e = keys.get(key_keys,i)
			return e if isinstance(e,instances) else i
	else:
		func = default


	for key in keys:

		if (isinstance(key,str) and (delimiter is not None) and (key not in iterable)):
			index = key.split(delimiter)
		elif (key in iterable):
			index = (key,)
		elif isinstance(key,scalars):
			index = (key,)
		else:
			index = (*key,)

		if len(index)>1 and (delimiter is not None):
			index,other = index[0],delimiter.join(index[1:])
		else:
			index,other = index[0],null

		if index in iterable:
			if not isinstance(other,nulls):
				setter(iterable[index],{other:keys[key]},delimiter=delimiter,default=default)
			else:
				if isinstance(keys[key],types) and isinstance(iterable[index],types) and default not in ['replace']:
					setter(iterable[index],keys[key],delimiter=delimiter,default=default)
				else:
					iterable[index] = func(index,key,iterable,keys)

		else:
			if not isinstance(other,nulls):
				iterable[index] = {}
				setter(iterable[index],{other:keys[key]},delimiter=delimiter,default=default)
			else:
				iterable[index] = func(index,key,iterable,keys)

	return

def getter(iterable,keys,delimiter=delimiter,default=None):
	'''
	Get nested value in iterable with nested keys
	Args:
		iterable (dict): dictionary to get with keys
		keys (str,dict,tuple,list): Dictionary of keys of delimiter separated strings, or tuple of string for nested keys
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string keys into list of nested keys
		default (callable,None,bool,iterable): Callable function with signature default(key_iterable,key_keys,iterable,keys) to modify value to be updated based on the given dictionaries, or True or False to default to keys or iterable values, or iterable of allowed types
	'''

	types = (dict,)

	if (not isinstance(iterable,types)) or (not isinstance(keys,(str,tuple,list))):
		return iterable

	key = keys

	if (isinstance(key,str) and (delimiter is not None) and (key not in iterable)):
		index = key.split(delimiter)
	elif isinstance(key,scalars):
		index = (key,)
	else:
		index = (*key,)

	if len(index)>1 and (delimiter is not None):
		index,other = index[0],delimiter.join(index[1:])
	else:
		index,other = index[0],null

	if index in iterable:
		if not isinstance(other,nulls):
			return getter(iterable[index],other,delimiter=delimiter,default=default)
		else:
			return iterable[index]
	else:
		return default


class Job(object):
	'''
	Job class
	Args:
		name (str): name of job
		options (dict[str,str]): options for job
		device (str): Name of device to submit to
		identity (int): identity of job
		jobs (iterable[Job,str,int]): related jobs of job
		path (str): path to job
		data (str,dict[str,str]): path of job script
		file (str): path of job executable
		logger (str): Name of logger
		env (dict[str,str]): environmental variables for job
		time (int,float): Timeout duration in seconds
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Keyword arguments		
	'''
	def __init__(self,name=None,options=None,device=None,identity=None,jobs=None,path=None,data=None,file=None,logger=None,env=None,time=None,execute=None,verbose=None,**kwargs):
		
		self.name = None if name is None else name
		self.options = {} if options is None else options
		self.device = None if device is None else device
		self.identity = None if identity is None else identity
		self.jobs = [] if jobs is None else jobs
		self.path = None if path is None else path
		self.data = None if data is None else data
		self.file = None if file is None else file
		self.logger = None if logger is None else logger
		self.env  = {} if env is None else env
		self.time = None if time is None else time
		self.execute = False if execute is None else execute
		self.verbose = None if verbose is None else verbose

		self.kwargs = kwargs

		self.init()

		return

	def __call__(self,options=None,device=None,env=None,time=None,execute=None,verbose=None):
		'''
		Call job
		Args:
			options (dict[str,str]): options for job
			device (str): Name of device to submit to
			env (dict[str,str]): environmental variables for job
			time (int,float): Timeout duration in seconds
			execute (boolean,int): Boolean whether to call commands
			verbose (int,str,bool): Verbosity
			kwargs (dict): Keyword arguments
		Returns:
			identity (identity): Identity of job
		'''

		identity = self.submit(options=options,device=device,env=env,time=time,execute=execute,verbose=verbose)

		return identity

	def __str__(self):
		if isinstance(self.name,str):
			string = self.name
		else:
			string = self.__class__.__name__
		return string

	def __repr__(self):
		return self.__str__()

	def init(self,*args,**kwargs):
		'''
		Initialize job
		Args:
			args (iterable): Positional arguments
			kwargs (dict): Keyword arguments
		'''

		for attr in kwargs:
			if hasattr(self,attr):
				setattr(self,attr,kwargs[attr])

		self.name = basedir(self.path) if self.name is None and self.path is not None else __name__ if self.name is None else self.name
		self.data = {self.data:self.data} if not isinstance(self.data,dict) else self.data
		self.logger = Logger(self.logger) if not isinstance(self.logger,Logger) else self.logger

		if self.jobs is None:
			self.jobs = []
		elif not isinstance(self.jobs,iterables):
			self.jobs = [self.jobs]

		cls = Job
		self.jobs = [job if not isinstance(job,dict) else cls(**job) for job in self.jobs]

		self.identity = None if self.identity is None else self.identity

		self.set()

		return

	def setup(self,options=None,device=None,**kwargs):
		'''
		Setup job
		Args:
			options (dict[str,str]): options for job
			device (str): Name of device to submit to
			kwargs (dict): Keyword arguments
		'''

		device = device if device is not None else self.device
		options = options if isinstance(options,dict) else self.options
		path = self.path

		# Init attributes
		self.init(device=device)

		# Update options
		self.update(options,path=path)

		return

	def submit(self,options=None,device=None,env=None,time=None,execute=None,verbose=None,**kwargs):
		'''
		Submit job
		Args:
			options (dict[str,str]): options for job
			device (str): Name of device to submit to
			env (dict[str,str]): environmental variables for job
			execute (boolean,int): Boolean whether to call commands
			time (int,float): Timeout duration in seconds
			verbose (int,str,bool): Verbosity
			kwargs (dict): Keyword arguments
		Returns:
			identity (int): Identity of job
		'''

		paths = {data:self.data[data] for data in self.data}

		env = {**self.env,**env} if env is not None else self.env

		status = self.setup(options=options,device=device)

		keys = self.keys

		options = dict(
			path=self.path,
			wrapper=self.wrapper,
			file=self.file,
			env=env,
			time=time if time is not None else self.time if self.time and self.time <=1 else None,
			execute=execute if execute is not None else self.execute,
			verbose=verbose if verbose is not None else self.verbose
			)

		for path in paths:
				
			path = paths[path]

			if self.device in ['local']:
				args = '{cmd}{flags}{exe}{path}'.format(
					cmd='',
					flags=ifs.join('{key}={value}'.format(key=key,value=env[key]) for key in env),
					exe=' ./' if not path.startswith('/') else ifs,
					path=path
					)
				env.clear()
				def wrapper(data):
					try:
						data = int(data[-1])
					except:
						data = None
					return data

			elif self.device in ['slurm']:

				args = '{cmd}{flags}{exe}{path}'.format(
					cmd='sbatch ',
					flags='--export={env}'.format(env=','.join(('{key}={value}' if not env[key].count(ifs) else '{key}="{value}"').format(key=key,value=env[key]) for key in env)) if env else '',
					exe=' < ',
					path=path
					)
				env.clear()
				def wrapper(data):
					try:
						data = int(data[-1])
					except:
						data = None
					return data
			else:
				args = None
				env.clear()				
				def wrapper(data):
					try:
						data = int(data[-1])
					except:
						data = None
					return data
		
			args = self.parse(args,keys)
				
			data = call(args,**options)			

			data = wrapper(data) 

			identity = data if data is not None else self.identity

		self.identity = identity if identity is not None else self.identity

		identity = self.identity

		return identity

	def cleanup(self,**kwargs):
		'''
		Cleanup job
		Args:
			kwargs (dict): Keyword arguments
		'''

		options = dict(
			execute=self.execute,
			verbose=self.verbose
			)
		options.update({kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg in options})

		if self.device in ['local']:
		
			paths,keys = [],[]
		
		elif self.device in ['slurm']:
			paths = [self.options.get(attr) for attr in ['stdout','stderr'] if attr in self.options]

			attr = 'parallel'
			if self.options.get(attr):
				keys = Dict(
					name=self.name,
					identity=self.identity,
					index=self.index(self.options[attr])[-1]
					)
			else:
				keys = Dict(
					name=self.name,
					identity=self.identity,
					index=None
					)

			paths = [path.replace('%x','{name}').replace('%A','{identity}').replace('%a','{index}') for path in paths]

		else:
			
			paths,keys = [],[]

		if isinstance(keys.index,iterables):
			keys = [{**keys,**dict(index=index)} for index in keys.index]
		else:
			keys = [keys]

		paths = [
			*[join(self.parse(path,key),root=self.path) for path in paths for key in keys]
			]

		for path in paths:
			rm(path,**options)

		return

	def stats(self,**kwargs):
		'''
		Stats of job
		Args:
			kwargs (dict): Keyword arguments
		Returns:
			stats (iterable[dict]): stats of job
		'''

		keys = self.keys

		options = dict(
			path=self.path,
			time=self.time if self.time and self.time <=1 else timeout.timeout,
			wrapper=self.wrapper,
			execute=True,
			verbose=self.verbose
			)
		
		status = None

		if self.device in ['local']:
			args = None
			def wrapper(data):
				return [Dict(identity=self.identity,index=None,name=self.name,state=self.states.done)]
		elif self.device in ['slurm']:
			if self.identity is None:
				args = 'sacct --noheader --allocations --user={user} --format={format} --name={name} --start={history}'
				# args = "cat job/slurm.txt | grep {name}"
			elif self.identity is not None:
				args = 'sacct --noheader --allocations --user={user} --jobs={identity} --name={name} --format={format} --start={history}'
				# args = "cat job/slurm.txt | grep {name} | grep {identity}"
			def wrapper(data):

				if data is None:
					return data

				funcs = dict()

				func = 'jobid'
				def function(data,attrs):
					identity,index = self.index(data)
					attrs.update(dict(identity=identity,index=index))
					if identity is None:
						attrs.update({attr:None for attr in attrs})
					return
				funcs[func] = function

				func = 'jobname'
				def function(data,attrs):
					name = str(data) if not isinstance(data,int) else None
					attrs.update(dict(name=name))
					return
				funcs[func] = function

				func = 'state'
				def function(data,attrs):
					state = str(data) if not isinstance(data,int) else None
					attrs.update(dict(state=state))
					return
				funcs[func] = function

				if not any(i for i in data):
					data = []
				else:
					for i in range(len(data)-1,-1,-1):
						data[i] = data[i] if not isinstance(data[i],int) else [data[i]]*len(funcs)
						attrs = Dict()
						for index,attr in enumerate(funcs):
							funcs[attr](data[i][index],attrs)
						data[i] = attrs
						if not data[i] or all(data[i][attr] is None for attr in data[i]):
							data.pop(i)
						elif data[i].identity is None:
							data.pop(i)
						elif any(isinstance(data[i][attr],iterables) for attr in data[i]):
							data.extend((
								Dict({**{attr:data[i][attr] for attr in data[i] if not isinstance(data[i][attr],iterables)},**attrs})
								for attrs in permuter({attr:data[i][attr] for attr in data[i] if isinstance(data[i][attr],iterables)})
								))
							data.pop(i)
				
				boolean = lambda data,obj=max(i.identity for i in data) if data else None:data.identity==obj
				
				data = list(set(i for i in data if boolean(i)))
				
				return data		
		else:
			args = None
			def wrapper(data):
				return [Dict(identity=self.identity,index=None,name=self.name,state=self.states.done)]

		args = self.parse(args,keys)
		data = None
		time,times = 0,self.time if self.time else 100

		while data is None and time < times:
			data = call(args,**options)
			data = wrapper(data)
			time += 1

		stats = data if data is not None else []

		return stats

	def status(self,**kwargs):
		'''
		Status of job
		Args:
			kwargs (dict): Keyword arguments
		Returns:
			status (dict): Status of job
		'''

		identity = self.identification(**kwargs)

		stats = self.stats(**kwargs)

		status = Dict()
		for state in self.states:
			status[state] = Dict()
			for attr in (self.states[state] if isinstance(self.states[state],iterables) else [self.states[state]]):
				jobs =  list(sorted(set((job.identity,job.index) if job.index is not None else job.identity for job in stats if job.state == attr),
							key = lambda data: ((data,-1) if not isinstance(data,iterables) else (*data,))))
				if jobs:
					status[state][attr] = jobs

		return status

	@property
	def state(self):
		'''
		State of job
		Returns:
			state (str): State of job
		'''

		status = self.status()

		for state in self.states:
			if any(status[state]):
				break
			else:
				state = None
		
		return state

	def identification(self,**kwargs):
		'''
		Identity of job
		Args:
			kwargs (dict): Keyword arguments
		Returns:
			identity (int): identity of job
		'''

		if self.identity is not None:
			identity = self.identity
			return identity

		if self.device in ['local']:
			defaults = dict()
		elif self.device in ['slurm']:
			defaults = dict(format='jobid')
		else:
			defaults = dict()

		def wrapper(stats):
			identity = max(job.identity for job in stats) if stats else None
			return identity

		kwargs.update(defaults)

		stats = self.stats(**kwargs)

		identity = wrapper(stats)

		self.identity = identity

		identity = self.identity

		return identity

	def set(self,options=None):
		'''
		Set job options
		Args:
			options (dict): Class options
		'''
		
		options = self.options if options is None else options
		
		defaults = self.defaults

		for option in list(options):
			if option in list(defaults):
				if callable(defaults[option]):
					try:
						self.options[option] = defaults[option](option,options={**self.options,**options})
					except:
						if option in self.options:
							self.options.pop(option)					
				else:
					self.options[option] = options.get(option,self.options.get(option))
			else:
				self.options[option] = options.get(option,self.options.get(option))
		
		for option in list(defaults):
			if callable(defaults[option]):
				try:
					self.options[option] = defaults[option](option,options={**self.options,**options})
				except:
					if option in self.options:
						self.options.pop(option)	
			else:
				self.options[option] = self.options.get(option,defaults[option])
				
		self.options = {option: self.options[option] 
			for option in sorted(self.options,key=(lambda option:
				list(self.patterns).index(option) if option in self.patterns else 
				len(self.patterns)+list(self.options).index(option)))
			}
		
		return
	
	def get(self):
		'''
		Get job options
		Returns:
			options (dict): Class options
		'''
		self.set()
		return self.options

	def update(self,options=None,path=None):
		'''
		Update job options
		Args:
			options (dict): Class options
			path (str,dict[str,str]): Path to options
		'''	

		self.set(options)

		options = self.load(path)

		options = {option:options[option] for option in options if option not in self.keywords}

		self.set(options)

		options = self.get()

		self.dump(path,options)

		return

	def instructions(self,string=None,pattern=None):
		'''
		Filter instruction strings
		Args:
			string (str): string to filter
			pattern (str): replacement pattern for instructions
		Returns:
			instructions (bool,str): instruction pattern if string is None else whether string is an instruction if pattern is None else string instruction replaced by pattern
		'''

		if self.device in ['local']:
			instructions = 'export '
		elif self.device in ['slurm']:
			instructions = '#SBATCH --'
		else:
			instructions = ''
		
		if string is None:
			instructions = '%s'%(instructions)
		elif pattern is None:
			instructions = '^[^#]*%s'%(instructions)
			instructions = contains(string,instructions)
		else:
			instructions = '%s'%(instructions)
			instructions = substitute(string,{instructions:pattern})

		return instructions

	@property
	def keys(self):

		if self.device in ['local']:
			defaults = dict()
		elif self.device in ['slurm']:
			defaults = dict(
				name=self.name,
				identity=self.identity,
				path=self.path,
				user='$USER',
				format='jobid%100,jobname%100,state%100',
				history="now-1hour"
			)
		else:
			defaults = dict()			

		keys = Dict()
		keys.update(self.env)
		keys.update(defaults)
		keys.update(self.kwargs)

		return keys

	@property
	def defaults(self):

		if self.device in ['local']:

			defaults = {}

		elif self.device in ['slurm']:

			defaults = {}

			def func(option,options):
				data = self.name
				return data
			option = 'name'
			defaults[option] = func

			def func(option,options):
				if not self.jobs or not any((isinstance(job,self.__class__) and job.identity is not None) or (isinstance(job,int) and job is not None) for job in self.jobs):
					data = False
				else:
					data = options.get(option) if options.get(option) else 'afterany:'
					data = '{key}:{value}'.format(
							key=':'.join(data.split(':')[:-1]) 
								if isinstance(data,str) and data.count(':') > 0 
								else data if isinstance(data,str) 
								else'',
							value=','.join([str(job.identity if isinstance(job,self.__class__) else job) for job in self.jobs if (isinstance(job,self.__class__) and job.identity is not None) or (isinstance(job,int) and job is not None)])
						)
				return data
			option = 'jobs'
			defaults[option] = func

			def func(option,options):
				data = options.get(option)
				ext = {'stdout':'stdout','stderr':'stderr'}.get(option,'log')
				if data is None:
					data = f'%x.%A.{ext}'
				if any(attr in options for attrs in ['parallel'] for attr in [attrs,self.patterns.get(attrs,attrs)]):
					data = f'%x.%A.%a.{ext}'
				else:
					data = f'%x.%A.{ext}'
				return data				
			option = 'stdout'
			defaults[option] = func

			option = 'stderr'
			defaults[option] = func


		else:
			
			defaults = {}

		return defaults

	@property
	def patterns(self):
		if self.device in ['local']:
			patterns = {}
		elif self.device in ['slurm']:
			patterns = {'name':'job-name','jobs':'dependency','parallel':'array','stdout':'output','stderr':'error'}
		else:
			patterns = {}
		return patterns

	@property
	def keywords(self):
		keywords = (*self.options.keys(),*self.patterns.keys(),*self.patterns.values())
		return keywords

	@property
	def states(self):
		if self.device in ['local']:
			states = Dict(
				error = 'error',
				done = 'done',
				run = 'run',
				)
		elif self.device in ['slurm']:
			states = Dict(
				error = ['BOOT_FAIL','CANCELLED','DEADLINE','FAILED','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','TIMEOUT'],
				done = ['COMPLETED'],
				run = ['PENDING','RUNNING','SUSPENDED'],
				)
		else:
			states = Dict(
				error = 'error',
				done = 'done',
				run = 'run',
				)		
		return states

	@property
	def delimiters(self):
		if self.device in ['local']:
			delimiters = Dict(delimiter=',',separator='=',range='%',comment='#')
		elif self.device in ['slurm']:
			delimiters = Dict(delimiter=',',separator='=',range='%',comment='#')
		else:
			delimiters = Dict(delimiter=',',separator='=',range='%',comment='#')
		return delimiters
	
	def parse(self,string,variables):
		'''
		Format string '...{key}...' -> '...{key}...'.format(key=value)
		Args:
			string (str): String to format '...{key}...'
			variables (dict[str,str]): Format strings {key}:value
		Returns:
			string (str): Formatted string
		'''

		if string is None:
			return string

		template = '{%s}'
		preprocess = {'{{':'#{{','}}':'#}}'}
		postprocess = {'#{':'{{','#}':'}}'}
		replacements = {'{{':'{','}}':'}','\\\\':"\\"}
		while any(template%(attr) in string for attr in variables if variables[attr] is not None):
			for replacement in preprocess:
				string = string.replace(replacement,preprocess[replacement])

			string = string.format(**variables)

			for replacement in postprocess:
				string = string.replace(replacement,postprocess[replacement])
		else:
			for replacement in replacements:
					string = string.replace(replacement,replacements[replacement])
		return string

	def wrapper(self,stdout,stderr,returncode,env=None,shell=None,time=None,msg=None,execute=None,verbose=None):
		'''
		Parse call result
		'''
		def wrapper(string):
			delimiter = ifs
			types = (int,str)
			if string.count(delimiter) > 1:
				string = [i.strip() for i in string.split(delimiter) if i]
			else:
				for type in types:
					try:
						string = type(string)
						break
					except:
						pass
			return string

		delimiter = '\n'

		result = [wrapper(i) for i in stdout.split(delimiter) if i] if stdout and any(stdout.split(delimiter)) is not None else None

		return result

	def index(self,data):
		'''
		Parse job index
		Args:
			data (int,str): call result
		Returns:
			identity (int): job identity
			index (int,iterable[int]): job index or indices
		'''

		if self.device in ['local']:
		
			identity,index = None,None

		elif self.device in ['slurm']:

			separator,delimiter,splitter,parser,colon = '%',',','_','-',':'

			if data is None:
				value = None
			elif isinstance(data,int):
				value = data
			elif data.isdigit():
				value = int(data)
			elif data.count(splitter) and data.split(splitter)[0].isdigit():
				value = int(data.split(splitter)[0])
			else:
				value = None
			identity = value
			
			if data is None:
				value = None
			elif isinstance(data,int):
				value = None
			elif not data.isdigit():
				value = data.split(splitter)[-1].replace('[','').replace(']','').split(colon)[0].split(separator)[0]
				if (parser not in value) and (delimiter not in value):
					value = int(value) if value.isdigit() else False
				else:
					value = value.split(delimiter)
					value = [j
						for i in value
						if i
						for j in ([int(i) if i.isdigit() else False] if not i.count(parser) else (range(*(int(j)+k for k,j in enumerate(i.split(parser)))) if all(j.isdigit() for j in i.split(parser)) else [False]))
						]
					value = value if value and all(i is not None for i in value) else None
			else:
				value = None	
			index = value

			if index is False or (isinstance(index,iterables) and any(i is False for i in index)):
				identity,index = None,None

		else:

			identity,index = None,None

		return identity,index


	def log(self,msg,verbose=None):
		'''
		Log messages
		Args:
			msg (str): Message to log
			verbose (int,str,bool): Verbosity of message			
		'''
		if verbose is None:
			verbose = self.verbose
		if msg is None:
			return
		msg += '\n'
		self.logger.log(verbose,msg)
		return

	def info(self,display=None,ignore=None,verbose=None,**kwargs):
		'''
		Log class information
		Args:
			display (str,iterable[str]): Show attributes
			ignore (str,iterable[str]): Do not show attributes
			verbose (bool,int,str): Verbosity of message	
			kwargs (dict): Additional logging keyword arguments						
		'''	

		msg = []

		options = dict(
			align=kwargs.get('align','<'),
			space=kwargs.get('space',1),
			width=kwargs.get('width',2)
			)
	
		precision = kwargs.get('precision',8)

		parse = lambda obj: str(obj.round(precision)) if isinstance(obj,arrays) else str(obj)

		display = None if display is None else [display] if isinstance(display,str) else display
		ignore = None if ignore is None else [ignore] if isinstance(ignore,str) else ignore

		attrs = ['name','identity','jobs','path','data','file','device','options']

		for attr in attrs:

			value = getattr(self,attr,None)
			string = None

			if (display is not None and attr not in display) or (ignore is not None and attr in ignore) or (value is None):
				continue

			if callable(value):
				try:
					string = '%s : %r'%(attr,value())
				except:
					string = None
			elif isinstance(value,dict):
				string = '%s :%s%s'%(attr,
					'\n\t' if len(value)>1 else ifs,
					('\n\t' if len(value)>1 else ifs).join(['%s %s %r'%(kwarg,':',value[kwarg])
						for kwarg in value if value[kwarg] is not None and value[kwarg] is not False])) if any(value[kwarg] for kwarg in value) else None
			else:
				string = '%s : %r'%(attr,value) 
			
			if string is not None:
				msg.append(string)
		
		msg = [i if isinstance(i,str) else str(i) for i in msg if i is not None]

		msg = '\n'.join(msg)

		self.log(msg,verbose=verbose)

		return

	def load(self,path=None,data=None,wrapper=None):
		'''
		Load job
		Args:
			path (str,dict[str,str]): path to job
			data (iterable[str]): default data
			wrapper (callable): Callable for data with signature wrapper(data)
		Returns:
			data (iterable[str]): job data
		'''

		paths = {data:join(self.data[data],root=self.path) for data in self.data} if path is None else {data:join(self.data[data],root=path) for data in self.data} if not isinstance(path,dict) else path
		data = None
		separator,comment = self.delimiters.separator,self.delimiters.comment
		options = dict(
			execute=True,
			verbose=False
			)

		if wrapper is True:
			def wrapper(data):
				return data
		elif not callable(wrapper):
			wrapper = None

		def parse(data):
			replacements = {'\n':''}
			for replacement in replacements:
				data = data.replace(replacement,replacements[replacement])
			return data

		if self.device in ['local']:

			if not callable(wrapper):
				def wrapper(data):
					data = {
						key:value
						for index,string in enumerate(data)
						if self.instructions(string)
						for strings in [[i for i in (
							separator.join(self.instructions(string,'').split(separator)[:1]),
							separator.join(self.instructions(string,'').split(separator)[1:])) if i]]
						for key,value in (zip(strings[0::2],strings[1::2]) if len(strings)>1 else ((*strings,None),))
						}
					return data		

			try:
				data = [] if data is None else data
				for path in paths:
					data.extend((parse(i) for i in read(path)))
			except:
				data = None

		elif self.device in ['slurm']:
		
			if not callable(wrapper):
				def wrapper(data):
					data = {
						key:value
						for index,string in enumerate(data)
						if self.instructions(string)
						for strings in [[i for i in (
							separator.join(self.instructions(string,'').split(separator)[:1]),
							separator.join(self.instructions(string,'').split(separator)[1:])) if i]]
						for key,value in (zip(strings[0::2],strings[1::2]) if len(strings)>1 else ((*strings,None),))
						}
					return data		

			try:
				data = [] if data is None else data
				for path in paths:
					data.extend((parse(i) for i in read(path)))
			except:
				data = None

		else:

			if not callable(wrapper):
				def wrapper(data):
					data = []
					return data

			try:
				data = [] if data is None else data
				for path in paths:
					data.extend((parse(i) for i in read(path)))
			except:
				data = None

		if callable(wrapper):
			data = wrapper(data)
		
		return data

	def dump(self,path=None,data=None,wrapper=None):
		'''
		Dump job
		Args:
			path (str,dict[str,str]): path to job
			data (dict[int,str],iterable[str]): job data
			wrapper (callable): Callable for data with signature wrapper(data)
		'''

		paths = {data:join(self.data[data],root=self.path) for data in self.data} if path is None else {data:join(self.data[data],root=path) for data in self.data} if not isinstance(path,dict) else path
		data = None if data is None else data
		separator,comment = self.delimiters.separator,self.delimiters.comment
		options = dict(
			execute=True,
			verbose=False
			)

		lines = self.load(paths,wrapper=True)

		if data is None or lines is None:
			return

		if self.device in ['local']:
			
			if not callable(wrapper):

				def wrapper(data):

					string = '{instructions}'.format(instructions=self.instructions())
					count = max((index for index,line in enumerate(lines) if contains(line,string)),default=None)
					
					if count is None:
						count = len(lines)

					for option in data:

						key = self.patterns.get(option,option)
						value = data[option]

						string = '{instructions}{key}'.format(instructions=self.instructions(),key=key)

						index = min((index for index,line in enumerate(lines) if contains(line,string)),default=None)

						if value is None or value is True:
							string = '{instructions}{key}'.format(instructions=self.instructions(),key=key)
						elif value is False:
							string = '{comment}{instructions}{key}'.format(instructions=self.instructions(),key=key,separator=separator,comment=comment)
						else:
							string = '{instructions}{key}{separator}{value}'.format(instructions=self.instructions(),key=key,value=value,separator=separator)

						if index is None:
							count += 1
							index = count
							lines.insert(index,string)
						else:
							lines[index] = string

					data = lines

					return data			
		
		elif self.device in ['slurm']:
		
			if not callable(wrapper):

				def wrapper(data):

					string = '{instructions}'.format(instructions=self.instructions())
					count = max((index for index,line in enumerate(lines) if contains(line,string)),default=None)
					
					if count is None:
						count = len(lines)

					for option in data:

						key = self.patterns.get(option,option)
						value = data[option]

						string = '{instructions}{key}'.format(instructions=self.instructions(),key=key)

						index = min((index for index,line in enumerate(lines) if contains(line,string)),default=None)

						if value is None or value is True:
							string = '{instructions}{key}'.format(instructions=self.instructions(),key=key)
						elif value is False:
							string = '{comment}{instructions}{key}'.format(instructions=self.instructions(),key=key,separator=separator,comment=comment)
						else:
							string = '{instructions}{key}{separator}{value}'.format(instructions=self.instructions(),key=key,value=value,separator=separator)

						if index is None:
							count += 1
							index = count
							lines.insert(index,string)
						else:
							lines[index] = string

					data = lines

					return data		

		else:

			if not callable(wrapper):
		
				def wrapper(data):
					return data		


		if callable(wrapper):
			data = wrapper(data)

		data = ['{line}\n'.format(line=line) for line in data]

		try:
			for path in paths:

				path = paths[path]
				
				if not exists(dirname(path)):
					mkdir(dirname(path),**options)

				write(path,data)

		except:
			pass

		return


class Task(Job):
	'''
	Task class
	Args:
		name (str): name of task
		options (dict[str,str]): options for task
		device (str): Name of device to submit to
		identity (int): identity of task
		jobs (iterable[Job,dict]): related jobs of task
		path (str): path to task
		data (str,dict[str,str]): path of task script
		file (str): path of task executable
		logger (str): Name of logger
		env (dict[str,str]): environmental variables for task
		time (int,float): Timeout duration in seconds		
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Keyword arguments		
	'''
	def __init__(self,name=None,options=None,device=None,identity=None,jobs=None,path=None,data=None,file=None,logger=None,env=None,time=None,execute=None,verbose=None,**kwargs):
	
		super().__init__(name=name,options=options,device=device,identity=identity,jobs=jobs,path=path,data=data,file=file,logger=logger,env=env,time=time,execute=execute,verbose=verbose,**kwargs)

		return

	def init(self,*args,**kwargs):
		'''
		Initialize task
		Args:
			args (iterable): Positional arguments
			kwargs (dict): Keyword arguments
		'''

		super().init(*args,**kwargs)

		cls = Job
		self.jobs = [job if not isinstance(job,dict) else cls(**job) for job in self.jobs]
		
		self.identity = [job.identity for job in self.jobs] if self.identity is None else self.identity

		for job in self.jobs:
			keywords = dict(
				jobs=[
					*[i for i in self.jobs if any(
						(isinstance(j,cls) and i==j) or 
						(isinstance(j,str) and i.name==j) or
						(isinstance(i,int) and i.identity==j)
						for j in job.jobs)
						],
					*[i for i in job.jobs if (
						(isinstance(i,cls) and i not in self.jobs)
						)
						],
					],
				execute = self.execute
				)
			job.init(**keywords)

		self.set()

		return

	def submit(self,options=None,device=None,env=None,time=None,execute=None,verbose=None,**kwargs):
		'''
		Submit task
		Args:
			options (dict[str,str]): options for job
			device (str): Name of device to submit to
			env (dict[str,str]): environmental variables for job
			time (int,float): Timeout duration in seconds
			execute (boolean,int): Boolean whether to call commands
			verbose (int,str,bool): Verbosity
			kwargs (dict): Keyword arguments
		Returns:
			identity (iterable[int]): Identity of job
		'''

		for job in self.jobs:
			keywords = dict(options=options,device=device,env=env,time=time,execute=execute,verbose=verbose,**kwargs)
			identity = job.submit(**keywords)

		identity = self.identification(**kwargs)

		return identity

	def cleanup(self,**kwargs):
		'''
		Cleanup job
		Args:
			kwargs (dict): Keyword arguments
		'''

		for job in self.jobs:
			keywords = dict(**kwargs)
			job.cleanup(**keywords)

		return

	def stats(self,**kwargs):
		'''
		Stats of task
		Args:
			kwargs (dict): Keyword arguments
		Returns:
			stats (iterable[dict]): stats of job
		'''

		stats = []

		for job in self.jobs:
			keywords = dict(**kwargs)
			stats.extend(job.stats(**keywords))

		return stats

	def identification(self,**kwargs):
		'''
		Identity of task
		Args:
			kwargs (dict): Keyword arguments
		Returns:
			identity (iterable[int]): identity of job
		'''

		identity = []

		for job in self.jobs:
			keywords = dict(**kwargs)
			identity.append(job.identification(**keywords))

		self.identity = identity if any(i is not None for i in identity) else self.identity

		identity = self.identity

		return identity

	def set(self,options=None):
		'''
		Set task options
		Args:
			options (dict): Class options
		'''

		keywords = dict(options=options)
		super().set(**keywords)

		for job in self.jobs:
			keywords = dict(options={**job.options,**self.options})
			job.set(**keywords)

		return
		
	def get(self):
		'''
		Get task options
		Returns:
			options (dict): Class options
		'''
		self.set()
		return self.options

	def update(self,options=None,path=None):
		'''
		Update task options
		Args:
			options (dict): Class options
			path (str,dict[str,str]): Path to options
		'''		

		keywords = dict(options=options,path=path)
		super().update(**keywords)

		for job in self.jobs:
			keywords = dict(options={**job.options,**self.options},path=job.path)
			job.update(**keywords)

		return

	@property
	def defaults(self):

		defaults = {}

		return defaults


class Work(Task):
	'''
	Work class
	Args:
		settings (str,dict): settings
		pool (int): Number of jobs per task		
		name (str): name of work
		options (dict[str,str]): options for work
		device (str): Name of device to submit to
		identity (int): identity of work
		jobs (iterable[Job,dict]): related jobs of work
		path (str): path to work
		data (str,dict[str,str]): path of work script
		file (str): path of work executable
		logger (str): Name of logger
		env (dict[str,str]): environmental variables for work
		time (int,float): Timeout duration in seconds		
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Keyword arguments		
	'''
	def __init__(self,settings=None,pool=None,name=None,options=None,device=None,identity=None,jobs=None,path=None,data=None,file=None,logger=None,env=None,time=None,execute=None,verbose=None,**kwargs):
	
		self.settings = settings
		self.pool = pool

		super().__init__(name=name,options=options,device=device,identity=identity,jobs=jobs,path=path,data=data,file=file,logger=logger,env=env,time=time,execute=execute,verbose=verbose,**kwargs)

		return

	def init(self,*args,**kwargs):
		'''
		Initialize task
		Args:
			args (iterable): Positional arguments
			kwargs (dict): Keyword arguments
		'''

		super().init(*args,**kwargs)


		# Get settings
		settings = self.settings if self.settings is not None else None
		pool = self.pool if self.pool is not None else 1
		options = self.options

		default = {}
		if settings is None:
			settings = default
		elif isinstance(settings,str):
			settings = load(settings,default=default)
		elif isinstance(settings,dict):
			settings = {**settings}


		# Get permutations of settings
		size = 0
		permutations = self.permute(settings)
		for permutation in permutations:

			# Update settings with permutation
			setting = copy(settings)
			boolean = lambda attr,permutation: attr.split(delimiter)[0] in ['seed']
			setter(setting,{attr: permutation[attr] for attr in permutation if boolean(attr,permutation)},delimiter=delimiter)

			# Get seeds for number of splits/seedings, for all nested settings branches that involve a seed
			seed,seeds,seedlings = self.spawn(setting)

			# Get shape and default key of permutations
			size += len(seeds)

		print(size)

		if size > 1:
			min = 0
			max = size//pool + (size%pool>0) - 1
			step = 1
			slice = pool
			count = (max - min + 1)//step
			
		else:
			slice = None
			min = None
			max = None
			step = None
			count = None

		env = {
			'SLURM_JOB_NAME':self.name,
			'SLURM_JOB_ID':self.identity,
			'SLURM_ARRAY_JOB_ID':self.identity,
			'SLURM_ARRAY_TASK_ID':None,
			'SLURM_ARRAY_TASK_MIN':min,
			'SLURM_ARRAY_TASK_MAX':max,
			'SLURM_ARRAY_TASK_STEP':step,
			'SLURM_ARRAY_TASK_COUNT':count,
			'SLURM_ARRAY_TASK_SLICE':slice,
			'SLURM_ARRAY_TASK_SIZE':size,
		}


		options.update(dict(
			parallel='{start}-{stop}:{step}%{number}'.format(start=0,stop=count-1,step=step,number=options.get('parallel','100').split('%')[-1])
			)
		)

		print(env)
		print(options)

		exit()


		cls = Job
		self.jobs = [job if not isinstance(job,dict) else cls(**job) for job in self.jobs]

		self.identity = [job.identity for job in self.jobs] if self.identity is None else self.identity

		for job in self.jobs:
			keywords = dict(
				jobs=[
					*[i for i in self.jobs if any(
						(isinstance(j,cls) and i==j) or 
						(isinstance(j,str) and i.name==j) or
						(isinstance(i,int) and i.identity==j)
						for j in job.jobs)
						],
					*[i for i in job.jobs if (
						(isinstance(i,cls) and i not in self.jobs)
						)
						],
					],
				execute = self.execute
				)
			job.init(**keywords)

		self.set()

		return




class Tasks(Task):
	'''
	Tasks class
	Args:
		name (str): name of tasks
		options (dict[str,str]): options for tasks
		device (str): Name of device to submit to
		identity (int): identity of tasks
		jobs (iterable[Job,dict]): related jobs of tasks
		path (str): path to tasks
		data (str,dict[str,str]): path of tasks script
		file (str): path of tasks executable
		logger (str): Name of logger
		env (dict[str,str]): environmental variables for tasks
		time (int,float): Timeout duration in seconds		
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Keyword arguments		
	'''
	def __init__(self,name=None,options=None,device=None,identity=None,jobs=None,path=None,data=None,file=None,logger=None,env=None,time=None,execute=None,verbose=None,**kwargs):
	
		super().__init__(name=name,options=options,device=device,identity=identity,jobs=jobs,path=path,data=data,file=file,logger=logger,env=env,time=time,execute=execute,verbose=verbose,**kwargs)

		return

	def init(self,*args,**kwargs):
		'''
		Initialize task
		Args:
			args (iterable): Positional arguments
			kwargs (dict): Keyword arguments
		'''

		super().init(*args,**kwargs)

		return

	def submit(self,options=None,device=None,env=None,time=None,execute=None,verbose=None,**kwargs):
		'''
		Submit task
		Args:
			options (dict[str,str]): options for job
			device (str): Name of device to submit to
			env (dict[str,str]): environmental variables for job
			time (int,float): Timeout duration in seconds
			execute (boolean,int): Boolean whether to call commands
			verbose (int,str,bool): Verbosity
			kwargs (dict): Keyword arguments
		Returns:
			identity (iterable[int]): Identity of job
		'''

		for job in self.jobs:
			keywords = dict(options=options,device=device,env=env,time=time,execute=execute,verbose=verbose,**kwargs)
			identity = job.submit(**keywords)

		identity = self.identification(**kwargs)

		return identity

	def cleanup(self,**kwargs):
		'''
		Cleanup job
		Args:
			kwargs (dict): Keyword arguments
		'''

		for job in self.jobs:
			keywords = dict(**kwargs)
			job.cleanup(**keywords)

		return

	def stats(self,**kwargs):
		'''
		Stats of task
		Args:
			kwargs (dict): Keyword arguments
		Returns:
			stats (iterable[dict]): stats of job
		'''

		stats = []

		for job in self.jobs:
			keywords = dict(**kwargs)
			stats.extend(job.stats(**keywords))

		return stats

	@property
	def state(self):
		'''
		State of job
		Returns:
			state (str): State of job
		'''

		status = self.status()

		for state in self.states:
			if any(status[state]):
				break
			else:
				state = None
		
		return state

	def status(self,**kwargs):
		'''
		Status of job
		Args:
			kwargs (dict): Keyword arguments
		Returns:
			status (dict): Status of job
		'''

		identity = self.identification(**kwargs)

		stats = self.stats(**kwargs)

		status = Dict()
		for state in self.states:
			status[state] = Dict()
			for attr in (self.states[state] if isinstance(self.states[state],iterables) else [self.states[state]]):
				jobs =  list(sorted(set((job.identity,job.index) if job.index is not None else job.identity for job in stats if job.state == attr),
							key = lambda data: ((data,-1) if not isinstance(data,iterables) else (*data,))))
				if jobs:
					status[state][attr] = jobs

		return status

	def identification(self,**kwargs):
		'''
		Identity of task
		Args:
			kwargs (dict): Keyword arguments
		Returns:
			identity (iterable[int]): identity of job
		'''

		identity = []

		for job in self.jobs:
			keywords = dict(**kwargs)
			identity.append(job.identification(**keywords))

		self.identity = identity if any(i is not None for i in identity) else self.identity

		identity = self.identity

		return identity


def workflow(func,settings,*args,**kwargs):
	'''
	Workflow
	Args:
		func (callable): function with signature func(name,task,data,settings,*args,**kwargs)
		settings (dict): settings
		args (iterable): Positional arguments
		kwargs (dict): Keyword arguments
	Returns:
		data (dict): data
	'''	

	strings = ['key','value']

	tasks = {attr:
		[Dict(zip(strings,data)) for data in zip(*(getattr(settings.workflow.settings[attr],string) for string in strings))]
		for attr in settings.workflow.settings if all(isinstance(getattr(settings.workflow.settings[attr],string),iterables) for string in strings)}
	variables = {attr:
		{string:getattr(settings.workflow.settings[attr],string) for string in strings}
		for attr in settings.workflow.settings if not all(isinstance(getattr(settings.workflow.settings[attr],string),iterables) for string in strings)}
	names = settings.workflow.path.name if isinstance(settings.workflow.path.name,iterables) else [settings.workflow.path.name]

	data = {}

	for index,task in enumerate(permuter(tasks)):

		for name in names:

			name = name.format(**{attr:task[attr].key for attr in task})

			try:
				value = func(name,task,data,settings,*args,**kwargs)
			except Exception as exception:
				print(name,exception)
				continue

			data[name] = value

	return data


def setup(settings,*args,**kwargs):
	'''
	Setup
	Args:
		settings (dict): settings
		args (iterable): Positional arguments
		kwargs (dict): Keyword arguments
	Returns:
		data (dict): data		
	'''

	def func(name,task,data,settings,*args,**kwargs):
		'''
		Setup
		Args:
			name (str): name of task
			task (task): data of task
			data (dict): data of tasks
			settings (dict): settings
			args (iterable): Positional arguments
			kwargs (dict): Keyword arguments		
		Returns:
			status (object): status of task
		'''		
		status = None

		options = {**settings.workflow.options,**{kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg in settings.workflow.options}}
		
		hosts = settings.workflow.host if isinstance(settings.workflow.host,iterables) else [settings.workflow.host]
		for host in hosts:
			path = join(settings.workflow.path.pwd.format(host=host),name)
			if exists(path):
				break
			else:
				host = None

		if host is None:
			raise

		path = join(settings.workflow.path.cwd,name)
		if not exists(path):
			mkdir(path,**options)
			status = True

		path = join(settings.workflow.path.cwd,name,host)
		if not exists(path):
			touch(path,**options)
			status = True

		for path in settings.workflow.path.data:
			source = join(settings.workflow.path.pwd.format(host=host),name,path)
			destination = join(settings.workflow.path.cwd,name,path)
			if exists(source) and not exists(destination):
				cp(source,destination,**options)
				status = True

		for path in settings.workflow.path.settings:
			source = join(settings.workflow.path.pwd.format(host=host),name,path)
			destination = join(settings.workflow.path.cwd,name,path)
			if exists(source) and not exists(destination):
				cp(source,destination,**options)
				status = True

		if status is not None:
			logger(**options)

		return status

	data = workflow(func,settings,*args,**kwargs)

	return data


def job(settings,*args,**kwargs):
	'''
	Job
	Args:
		settings (dict): settings
	Returns:
		data (dict): data		
	'''

	def func(name,task,data,settings,*args,**kwargs):
		'''
		Job
		Args:
			name (str): name of task
			task (task): data of task
			data (dict): data of tasks
			settings (dict): settings
			args (iterable): Positional arguments
			kwargs (dict): Keyword arguments		
		Returns:
			status (object): status of task
		'''

		status = None

		path = join(name,root=settings.workflow.path.cwd)
		opts = settings.workflow.job.job.job.args
		options = {
			**{attr:settings.workflow.job.job.job[attr] for attr in settings.workflow.job.job.job if attr not in ['args','kwargs']},
			**settings.workflow.job.job.job.kwargs,
			**dict(name=name,path=path)
			}


		job = Job(*opts,**options)

		def boolean(jobs,errors,args,kwargs):
			boolean = {name: jobs[name].status(*args,**kwargs) for name in jobs}
			boolean = Dict(
				{status: [name for name in boolean in boolean[name] in [status]]
				for status in errors}
				)
			return boolean


		jobs = {**data,name:job}

		keywords = dict(
			jobs = jobs,
			errors = job.errors,
			args = (*settings.workflow.job.args,*args),
			kwargs = {**settings.workflow.job.kwargs,**kwargs},
			)
		options = {**settings.workflow.options,**{kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg in settings.workflow.options}}

		status = boolean(**keywords)

		while not status.run:
			if status.error:
				for name in status.error:
					logger("Error Job: {name}".format(name=name),**options)
					# Resubmit error job
			if not any(name in [*status.error,*status.done]):
				logger("Queue Job: {name}".format(name=name),**options)
				# Submit new job
			elif not any(name in [*status.done]):
				logger("Done Job: {name}".format(name=name),**options)
				# Clean up done job
			status = boolean(**keywords)
		else:
			if status.run:
				for name in status.run:
					logger("Running Job: {name}".format(name=name),**options)

		status = job
		
		return status

		# Check Running Job
		# pass all

		# Check Completed Job
		# cp submit* to cwd/name
		# Cleanup (rm i/ *.std{out,err} (no errors), tar)

		# Check New Job
		# submit main.py / run.py job with main.py or workflow.slurm < cmd

		# Decide which workflow tasks may be on login vs. compute nodes
		# i.e) cleanup - separate cleanup .slurm script necessary
		# coordination of job id's to make cleanup a dependency of postprocess etc?

		# Check consistent between setup and submit to pull/push files 
		# and not overwrite or copy wrong files

		# Run test workflow on vector with slurm.json workflow push and pull on local to scratch/test

		# Decide what is handled by 
		# lower: main.py/run.py : cp settings.json cwd/ , touch submit.std* , touch job.sh , job.slurm , id.std*
		# higher: workflow.py : review


	data = workflow(func,settings,*args,**kwargs)

	return

def cleanup(settings,*args,**kwargs):
	'''
	Cleanup
	Args:
		settings (dict): settings
	Returns:
		data (dict): data		
	'''

	options = {**settings.workflow.options,**{kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg in settings.workflow.options}}

	return

def main(settings,*args,func=None,**kwargs):

	def config(settings,func=None):
		
		options=dict(wrapper=Dict)
		try:
			settings = load(settings,**options)
		except:
			settings = None

		options = dict(setup=setup,job=job,cleanup=cleanup)
		try:
			func = settings.workflow.func if func is None or func not in options else func
			func = options.get(func)
		except:
			func = None

		return func,settings

	func,settings = config(settings,func)

	try:
		func(settings,*args,**kwargs)
	except:
		pass

	return

if __name__ == '__main__':

	arguments = {
		'--settings':{
			'help':'Settings',
			'type':str,
			'default':None,
			'nargs':'?'
		},	
		'--func':{
			'help':'Function',
			'type':str,
			'default':"setup",
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