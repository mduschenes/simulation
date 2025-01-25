#!/usr/bin/env python

# Import python modules
import os,sys,argparse,subprocess,shlex,itertools,json,io,re,signal

from src.logger import Logger

integers = (int,)
floats = (float,)
scalars = (*integers,*floats,str,type(None),)
iterables = (list,tuple,set,)
streams = (io.RawIOBase,)
delimiter = '.'
ifs = ' '

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

class TimeoutError(Exception):
	pass

class timeout:

	def __init__(self,time=None,message=None,default=None):
		self.time = time if time is not None else 1e-1
		self.message = message if message is not None else None
		self.default = default
		self.signal = signal.SIGALRM
		self.type = signal.ITIMER_REAL
		self.error = TimeoutError
		self.alarm = 0
		return

	def handler(self, signum, frame):
		raise self.error(self.message) if self.message else self.error
	def start(self):
		signal.signal(self.signal,self.handler)
		signal.setitimer(self.type,self.time)
		return
	def stop(self):
		signal.setitimer(self.type,self.alarm)
		return
	
	def __enter__(self):
		self.start()
	
	def __exit__(self,type,value,traceback):
		self.stop()

	def __call__(self,func):
		def wrapper(*args, **kwargs):
			self.start()
			try:
				result = func(*args, **kwargs)
			finally:
				result = self.default
				self.stop()
			return result
		return wrapper

def call(*args,path=None,wrapper=None,env=None,stdin=None,stdout=None,stderr=None,shell=None,time=None,execute=False,verbose=None):
	'''
	Submit call to command line of the form $> args
	Args:
		args (iterable[iterable[str]]],iterable[str]): Arguments to pass to command line {arg:value} or {arg:[value]} or [value], nested iterables are piped		
		path (str): Path to call from
		wrapper (callable): Wrapper for output, with signature wrapper(stdout,stderr,returncode,env=None,shell=None,time=None,verbose=None)
		env (dict[str,str]): Environmental variables for args		
		stdin (file): Stdinput stream to command
		stdout (file): Stdoutput to command
		stderr (file): Stderr to command
		shell (bool) : Use shell subprocess
		time (int): Timeout		
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): Verbosity
	Returns:
		result (object): Return of commands
	'''

	def caller(args,inputs=None,outputs=None,errors=None,env=None,shell=None,time=None,verbose=None):

		def run(args,stdin=None,stdout=None,stderr=None,env=None,shell=None):
			env = {**environ(),**env} if env is not None else None
			args = [ifs.join(args)] if shell else [j for i in args for j in shlex.split(i)]
			try:
				result = subprocess.Popen(args,stdin=stdin,stdout=stdout,stderr=stderr,env=env,shell=shell)
			except (OSError,FileNotFoundError) as exception:
				result = Popen(args,stdin=stdin,stdout=stdout,stderr=stderr,env=env,shell=shell)
				logger(exception,verbose=verbose)
			return result

		def process(obj):
			parse = lambda string: os.path.expandvars(os.path.expanduser(string)).replace('~',os.environ['HOME'])
			if isinstance(obj,str):
				obj = parse(obj)
			elif isinstance(obj,iterables):
				obj = [parse(i) if isinstance(i,str) else [parse(j) for j in i] for i in obj]
			return obj

		def wrap(stdout,stderr,returncode):
			stdout = '\n'.join(stdout)
			stderr = '\n'.join(stderr)
			returncode =  returncode
			return stdout,stderr,returncode

		def parse(obj):
			try:
				obj = obj.strip().decode('utf-8')
			except:
				obj = str(obj)
			return obj

		stdin = None if inputs is None else inputs if isinstance(inputs,str) else inputs.pop(0) if len(inputs)>len(args) else None
		stdout = subprocess.PIPE if outputs is None else subprocess.PIPE
		stderr = subprocess.PIPE if errors is None else subprocess.PIPE
		returncode = None

		inputs = [inputs]*len(args) if inputs is None or isinstance(inputs,str) else inputs
		outputs = [outputs]*len(args) if outputs is None or isinstance(outputs,str) else outputs
		errors = [errors]*len(args) if errors is None or isinstance(errors,str) else errors

		for arg,input,output,error in zip(args,inputs,outputs,errors):
			if isinstance(output,str):
				mkdir(output)
			if isinstance(error,str):
				mkdir(error)

			stdin = open(input,'r') if isinstance(input,str) else input if input is not None else stdin
			stdout = open(output,'w') if isinstance(output,str) else output if output is not None else stdout
			stderr = open(error,'w') if isinstance(error,str) else error if error is not None else stderr

			arg = process(arg)

			result = run(arg,stdin=stdin,stdout=stdout,stderr=stderr,env=env,shell=shell)

			if isinstance(stdin,streams):
				stdin.close()
			if isinstance(stdout,streams):
				stdout.close()
			if isinstance(error,streams):
				stderr.close()

			stdin = result.stdout

		stdout,stderr,returncode = [],[],result.returncode
		
		if result.stdout is not None:
			try:
				with timeout(time=time):
					for line in result.stdout:
						stdout.append(parse(line))			
						logger(stdout[-1],verbose=verbose)
			except TimeoutError:
				pass
			except:
				with timeout(time=time):
					stdout.append(parse(result.stdout))
					logger(stdout[-1],verbose=verbose)
		
		try:
			returncode = timeout(time=time,default=-1)(result.wait)()
		except TimeoutError:
			returncode = -1

		if result.stderr is not None:
			try:
				with timeout(time=time):
					for line in result.stderr:	
						stderr.append(parse(line))
						if returncode is not None:
							logger(stderr[-1],verbose=verbose)
			except TimeoutError:
				pass
			except:
				stderr.append(parse(result.stderr))
				logger(stderr[-1],verbose=verbose)

		stdout,stderr,returncode = wrap(stdout,stderr,returncode)

		return stdout,stderr,returncode

	if not callable(wrapper):
		def wrapper(stdout,stderr,returncode,env=None,shell=None,time=None,verbose=None):
			result = stdout
			return result

	def parser(*args,env=None,verbose=None):

		args = [(str(arg) if (isinstance(arg,scalars) or isinstance(arg,str) and not arg.count(symbol)) else 
				[str(subarg) for subarg in (arg if not isinstance(arg,scalars) else arg.split(symbol)) if subarg is not None]) 
				for arg in args if arg is not None]

		pipe = any(not isinstance(arg,scalars) for arg in args)

		if pipe:
			args = [[str(subarg) for subarg in arg] if not isinstance(arg,scalars) else [arg] for arg in args]
		else:
			args = [[str(arg) for arg in args]]


		pipe = '|'
		symbols = ['|']

		arguments = []
		for arg in args:
			argument = []
			for subarg in arg:
				if subarg is None:
					pass
				elif isinstance(subarg,str):
					for symbol in symbols:
						subarg = subarg.replace(symbol,pipe)
					if subarg in [pipe]:
						argument.append(pipe)
					elif subarg.count(pipe):
						argument.extend([j for i in subarg.split(pipe) for j in [i,pipe]])
					else:
						argument.append(subarg)
				else:
					argument.append(subarg)

			arg = [[]]
			for subarg in argument:
				if subarg not in [pipe]:
					arg[-1].append(subarg)
				else:
					arg.append([])

			arguments.extend([[i.strip() for i in subarg] for subarg in arg if len(subarg)])

		args = arguments

		cmd = ' | '.join([
			ifs.join([str(subarg) for subarg in arg])
			for arg in args])

		inputs = []
		symbols = ['<']

		for arg in args:

			for symbol in symbols:
				redirect = symbol in arg
				if redirect:
					break

			if redirect:
				index = arg.index(symbol)
				input = ifs.join(arg[index+1:])
				subarg = arg[:index]
			else:
				index = None
				input = None
				subarg = arg[:]

			args[args.index(arg)] = subarg
			inputs.append(input)

		return inputs,args,cmd,env


	inputs,args,cmd,env = parser(*args,env=env,verbose=verbose)
	result = None

	inputs = [stdin,*inputs]
	outputs = stdout
	errors = stderr

	msg = '{path} : {cmd}'.format(path=path,cmd=cmd) if path is not None else '{cmd}'.format(cmd=cmd)
	logger(msg,verbose=verbose)

	if execute:
		with cd(path):
			result = wrapper(*caller(args,inputs=inputs,outputs=outputs,errors=errors,env=env,shell=shell,time=time,verbose=verbose),env=env,shell=shell,time=time,verbose=verbose)

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
		mkdir(path)
		self.path = path
		return

	def __enter__(self):
		self.cwd = cwd()
		try:
			path = os.path.abspath(os.path.expandvars(os.path.expanduser(path))) if path is not None else None
			os.chdir(path)
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

	path = os.path.abspath(os.path.expandvars(os.path.expanduser(path))) if path is not None else None

	args = ['mkdir','-p',path]

	if path is not None and not exists(path):
		result = call(*args,execute=execute,verbose=verbose)


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


def load(path,default=None,wrapper=None,verbose=None):
	'''
	Load data from path
	Args:
		path (str): path
		default (object): default data
		wrapper (callable): wrapper to load data with signature wrapper(data)
		verbose (int,str,bool): Verbosity
	Returns:
		data (object): data
	'''
	logger('Load: {path}'.format(path=path),verbose=verbose)
	try:
		with open(path,'r') as file:
			data = json.load(file)
	except:
		data = default
	if wrapper is not None:
		try:
			data = wrapper(data)
		except:
			pass
	return data

def dump(path,data,wrapper=None,verbose=None):
	'''
	Dump data to path
	Args:
		path (str): path
		data (object): data
		wrapper (callable): wrapper to load data with signature wrapper(data)
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
		with open(path,'w') as file:
			data = json.dump(data,file)
	except:
		logger(exception,verbose=verbose)
		pass
	return


def parse(string,formats):
	'''
	Format string '...{key}...' -> '...{key}...'.format(key=value)
	Args:
		string (str): String to format '...{key}...'
		formats (dict[str,str]): Format strings {key}:value
	Returns:
		string (str): Formatted string
	'''
	template = '{%s}'
	preprocess = {'{{':'#{{','}}':'#}}'}
	postprocess = {'#{':'{{','#}':'}}'}
	replacements = {'{{':'{','}}':'}','\\\\':"\\"}
	while any(template%(attr) in string for attr in formats if isinstance(formats[attr],str)):
		for replacement in preprocess:
			string = string.replace(replacement,preprocess[replacement])

		string = string.format(**formats)

		for replacement in postprocess:
			string = string.replace(replacement,postprocess[replacement])
	else:
		for replacement in replacements:
			string = string.replace(replacement,replacements[replacement])
	return string

def permuter(iterable,repeat=None):
	'''
	Get product of permutations of iterable
	Args:
		iterable (iterable[iterable],iterable[int],dict): Iterables to permute, or iterable of int to get all permutations of range(int) or dict of iterables
	Returns:
		iterable (generator[tuple],generator[dict]): Generator of tuples of all permutations of iterable
	'''
	
	if all(isinstance(i,int) for i in iterable):
		iterable = [range(i) for i in iterable]

	repeat = 1 if repeat is None else repeat
	
	if isinstance(iterable,iterables):
		return itertools.product(*(i for i in iterable),repeat=repeat)
	else:
		return (dict(zip(iterable,value)) for value in itertools.product(*(iterable[i] for i in iterable),repeat=repeat))

def copier(obj,copy):
	'''
	Copy object based on copy

	Args:
		obj (object): object to be copied
		copy (bool): boolean or None whether to copy value
	Returns:
		Copy of value
	'''

	if copy:
		return deepcopy(obj)
	else:
		return obj

def setter(iterable,keys,delimiter=None,default=None,copy=False):
	'''
	Set nested value in iterable with nested keys
	Args:
		iterable (dict): dictionary to be set in-place with value
		keys (dict,tuple): Dictionary of keys of delimiter separated strings, or tuple of string for nested keys, and values to set 
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string keys into list of nested keys
		default(callable,None,bool,iterable): Callable function with signature default(key_iterable,key_keys,iterable,keys) to modify value to be updated based on the given dictionaries, or True or False to default to keys or iterable values, or iterable of allowed types
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
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
				setter(iterable[index],{other:keys[key]},delimiter=delimiter,default=default,copy=copy)
			else:
				if isinstance(keys[key],types) and isinstance(iterable[index],types) and default not in ['replace']:
					setter(iterable[index],keys[key],delimiter=delimiter,default=default,copy=copy)
				else:
					iterable[index] = copier(func(index,key,iterable,keys),copy=copy)

		else:
			if not isinstance(other,nulls):
				iterable[index] = {}
				setter(iterable[index],{other:keys[key]},delimiter=delimiter,default=default,copy=copy)
			else:
				iterable[index] = copier(func(index,key,iterable,keys),copy=copy)

	return

def getter(iterable,keys,delimiter=None,default=None,copy=False):
	'''
	Get nested value in iterable with nested keys
	Args:
		iterable (dict): dictionary to get with keys
		keys (str,dict,tuple,list): Dictionary of keys of delimiter separated strings, or tuple of string for nested keys
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string keys into list of nested keys
		default(callable,None,bool,iterable): Callable function with signature default(key_iterable,key_keys,iterable,keys) to modify value to be updated based on the given dictionaries, or True or False to default to keys or iterable values, or iterable of allowed types
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
	'''

	types = (dict,)

	if (not isinstance(iterable,types)) or (not isinstance(keys,(str,tuple,list))):
		return copier(iterable,copy=copy)

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
			return getter(iterable[index],other,delimiter=delimiter,default=default,copy=copy)
		else:
			return copier(iterable[index],copy=copy)
	else:
		return copier(default,copy=copy)


class Job(object):
	'''
	Job class
	Args:
		name (str): name of job
		options (dict[str,str]): options for job
		device (str): Name of device to submit to
		identity (int): identity of job
		jobs (iterable[Job]): related jobs of job
		path (str): path to job
		data (str,dict[str,str]): path of job script
		file (str): path of job executable
		env (dict[str,str]): environmental variables for job
		logger (str): Name of logger
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): Verbosity
		kwargs (dict): Keyword arguments		
	'''
	def __init__(self,name=None,options=None,device=None,identity=None,jobs=None,path=None,data=None,file=None,env=None,execute=False,verbose=None,**kwargs):
		
		self.name = None if name is None else name
		self.options = {} if options is None else options
		self.device = None if device is None else device
		self.identity = None if identity is None else identity
		self.jobs = [] if jobs is None else jobs
		self.path = None if path is None else path
		self.data = None if data is None else data
		self.file = None if file is None else file
		self.env  = {} if env is None else env
		self.logger = None if logger is None else None
		self.execute = False if execute is None else execute
		self.verbose = None if verbose is None else verbose

		self.kwargs = kwargs

		self.init()

		return

	def init(self,*args,**kwargs):
		'''
		Initialize job
		Args:
			args (iterable): Positional arguments
			kwargs (dict): Keyword arguments
		'''

		self.name = basedir(self.path) if self.name is None and self.path is not None else __name__ if self.name is None else self.name
		self.identity = None if self.identity is None else self.identity
		self.jobs = [] if self.jobs is None else [self.jobs] if not isinstance(self.jobs,iterables) else self.jobs
		self.data = {path:join(self.data[path],root=self.path) for path in self.data} if isinstance(self.data,dict) else {self.data:join(self.data,root=self.path)}
		self.logger = Logger(self.logger) if not isinstance(self.logger,Logger) else self.logger

		self.set()

		return

	def setup(self,options=None,device=None,execute=False,verbose=None,**kwargs):
		'''
		Setup job
		Args:
			options (dict[str,str]): options for job
			device (str): Name of device to submit to
			execute (boolean,int): Boolean whether to call commands
			verbose (int,str,bool): Verbosity
			kwargs (dict): Keyword arguments
		'''

		self.init()

		options = options if isinstance(options,dict) else self.options
		device = device if device is not None else self.device

		self.device = device

		# Update options and job script
		path = self.data
		self.update(options,path=path)

		return

	def status(self,**kwargs):
		'''
		Status of job
		Args:
			kwargs (dict): Keyword arguments
		Returns:
			status (str): Status of job
		'''

		return None

		kwargs = {**self.kwargs,**kwargs}

		kwargs = {
			**kwargs,
			# **{attr:join(kwargs.get(attr),root=self.path) for attr in ['error']},
			**dict(name=self.name,path=self.path),
			}

		device = self.device
		options = dict(execute=True,verbose=False)

		if device in ['local']:
			state = dict()
			status = dict()
		elif device in ['slurm']:
			# Options --batch-script
			state = dict(
				run = ['PENDING','RUNNING','SUSPENDED'],
				error = ['BOOT_FAIL','CANCELLED','DEADLINE','FAILED','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','TIMEOUT'],
				done = ['COMPLETED'],
				)
			status = dict(
				# run='squeue --format="%.100j" -u {user} | tr -d " " | grep {name}',
				# error = 'sacct --user {user} --noheader --format jobid,jobname%100,submit,start,end,state --start {start} | grep {name} | awk -F' ' -v OFS=' ' '$1 ~ /[0-9]*_[0-9]*/ {sub(/_.*/, "", $1)} 1' | sort -u -k1',
				# error='find {path} -maxdepth 1 -mindepth 1 -type f -size +0 -name "{error}" | sort -n | tail -1 | awk "{{print $NF}}" | sed "s:{pattern}:\\{n}:"',
				# done='find {path} -maxdepth 1 -mindepth 1 -type f -size +0 -name "{error}" | sort -n | tail -1 | awk "{{print $NF}}"',
			)
		else:
			state = dict()			
			status = dict()

		for attr in status:
			string = status[attr]
			if attr in ['run']:
				formats = {**kwargs}
				string = parse(string,formats)
				string = call(string,**options)
			elif attr in ['error']:
				formats = {**kwargs,**dict(format="[0-9]*.[0-9]*",n='1')}
				string = parse(status[attr],formats)
				string = call(string,**options)
			elif attr in ['run']:
				formats = {**kwargs,**dict(n='1')}
				string = parse(status[attr],formats)
				string = call(string,**options)

			status[attr] = string

		return status

	def set(self,options=None):
		'''
		Set class options
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
					except Exception as exception:
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
		Get class options
		Returns:
			options (dict): Class options
		'''
		self.set()
		return self.options

	def update(self,options=None,path=None):
		'''
		Update class options
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
				assert options.get(option) is not None or self.jobs
				data = '{key}:{value}'.format(
						key=':'.join(options.get(option).split(':')[:-1]) 
							if isinstance(options.get(option),str) and options.get(option).count(':') > 0 
							else options.get(option) if isinstance(options.get(option),str) 
							else'',
						value=','.join([str(job.identity if isinstance(job,self.__class__) else job) for job in self.jobs])
					)
				return data
			option = 'dependency'
			defaults[option] = func

			def func(option,options):
				data = options.get(option)
				if data is None:
					data = '%x.%A.stdout'
				if 'array' in options:
					data = '%x.%A.%a.stdout'
				else:
					data = '%x.%A.stdout'
				return data				
			option = 'output'
			defaults[option] = func

			def func(option,options):
				data = options.get(option)
				if data is None:
					data = '%x.%A.stderr'
				if 'array' in options:
					data = '%x.%A.%a.stderr'
				else:
					data = '%x.%A.stderr'
				return data				
			option = 'error'			
			defaults[option] = func

		else:
			
			defaults = {}

		return defaults

	@property
	def patterns(self):
		if self.device in ['local']:
			patterns = {}
		elif self.device in ['slurm']:
			patterns = {'name':'job-name'}
		else:
			patterns = {}
		return patterns

	@property
	def keywords(self):
		keywords = (*self.options.keys(),*self.patterns.keys(),*self.patterns.values())
		return keywords

	@property
	def errors(self):
		errors = Dict(run=1,error=-1,done=0)
		return errors

	@property
	def delimiters(self):
		if self.device in ['local']:
			delimiters = Dict(separator='=',comment='#')
		elif self.device in ['slurm']:
			delimiters = Dict(separator='=',comment='#')
		else:
			delimiters = Dict(separator='=',comment='#')
		return delimiters
	

	# def setup(self,path=None,data=None,file=None,options=None,device=None,execute=False,verbose=None,**kwargs):
	# 	'''
	# 	Call job
	# 	Args:
	# 		path (str): path to job
	# 		data (str): path of job script
	# 		file (str): path of job executable			
	# 		options (dict[str,str]): options for job
	# 		device (str): Name of device to submit to
	# 		execute (boolean,int): Boolean whether to call commands
	# 		verbose (int,str,bool): Verbosity
	# 		kwargs (dict): Keyword arguments
	# 	'''
	def __call__(self,args,kwargs=None,exe=None,flags=None,cmd=None,options=None,env=None,device=None,execute=False,verbose=None):
		'''
		Call job
		Args:
			args (dict[str,str],dict[str,iterable[str],iterable[iterable[str]]],iterable[str]): Arguments to pass to command line {arg:value} or {arg:[value]} or [value]
			kwargs (dict): Keyword arguments for args
			exe (str,iterable[str]): Executable for args
			flags (str,iterable[str]): Flags for args
			cmd (str,iterable[str]): Command for args
			options (str,iterable[str]): Options for args
			env (dict[str,str]): Environmental variables for args
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

		if device in ['local']:
			exe = [*['%s%s'%('./' if not e.startswith('/') else '',e) for e in exe[:1]],*exe[1:]]
			flags = [*flags]
			cmd = [*cmd]
			options = [*options]		
			env = {
				**{
					**({
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
						} if len(kwargs) else {}
						),
					**{arg: '%s%s%s'%("\"" if len(args[arg])>1 else '',' '.join([subarg for subarg in args[arg]]),"\"" if len(args[arg])>1 else '') for arg in args},
				},
				**env			
			}

		elif device in ['slurm']:
			exe,flags,cmd,options,env = (
				['sbatch'],
				[
				*flags,
				*(['-J',basedir(kwargs.get('cwd'))] if len(kwargs) else []),
				*(['%s=%s'%('--export',','.join(['%s=%s'%(arg,' '.join([subarg for subarg in args[arg]])) for arg in args]))] if len(args) else []),
				],
				['<'],
				[*exe,*cmd,*options],
				{
				**{
					**({
						'SLURM_ARRAY_TASK_SLICE':kwargs.get('slice'),
						'SLURM_ARRAY_TASK_SIZE':kwargs.get('size'),
					} if len(kwargs) else {}),
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
					**({
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
					} if len(kwargs) else {}),
				},
				**env,			
			}

		args = [*exe,*flags,*cmd,*options]

		env = {str(var): str(env[var]) if env[var] is not None else '' for var in env}

		return args,env

	def __str__(self):
		if isinstance(self.name,str):
			string = self.name
		else:
			string = self.__class__.__name__
		return string

	def __repr__(self):
		return self.__str__()


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

		attrs = ['name','identity','path','data','file','device','options','status']

		msg = []

		for attr in attrs:
			if not hasattr(self,attr):
				continue
			if attr in ['options']:
				string = '%s :\n\t%s'%(attr,'\n\t'.join(['%s : %r'%(kwarg,getattr(self,attr)[kwarg]) for kwarg in getattr(self,attr)]))
			elif callable(getattr(self,attr)):
				string = '%s : %r'%(attr,getattr(self,attr)())
			else:
				string = '%s : %r'%(attr,getattr(self,attr))
			
			msg.append(string)
		
		msg = [i if isinstance(i,str) else str(i) for i in msg]

		msg = '\n'.join(msg)

		self.log(msg,verbose=verbose)

		return

	def load(self,path=None,wrapper=None):
		'''
		Load job
		Args:
			path (str,dict[str,str]): path to job
			wrapper (callable): Callable for data with signature wrapper(data)
		Returns:
			data (iterable[str]): job data
		'''

		paths = {path:join(path,root=self.path) for path in self.data} if path is None else {path:path} if not isinstance(path,dict) else path
		data = None
		separator,comment = self.delimiters.separator,self.delimiters.comment
		mode = 'r'

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
						for strings in [i for i in (
							separator.join(self.instructions(string,'').split(separator)[:1]),
							separator.join(self.instructions(string,'').split(separator)[1:])) if i]		
						for key,value in (zip(strings[0::2],strings[1::2]) if len(strings)>1 else ((*strings,None),))
						}
					return data
			
			try:
				data = [] if data is None else data
				for path in paths:
					with open(path,mode) as obj:
						data.extend((parse(i) for i in obj.readlines()))
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
					with open(path,mode) as obj:
						data.extend((parse(i) for i in obj.readlines()))
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
					with open(path,mode) as obj:
						data.extend((parse(i) for i in obj.readlines()))
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

		paths = {path:join(path,root=self.path) for path in self.data} if path is None else {path:path} if not isinstance(path,dict) else path
		data = None if data is None else data
		separator,comment = self.delimiters.separator,self.delimiters.comment
		mode = 'w'

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

						index = max((index for index,line in enumerate(lines) if contains(line,string)),default=None)

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
				with open(paths[path],mode) as obj:
					obj.writelines(data)
		except:
			pass

		return



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