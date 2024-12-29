#!/usr/bin/env python

# Import python modules
import os,sys,argparse,subprocess,itertools,json

integers = (int,)
floats = (float,)
scalars = (*integers,*floats,str,type(None),)
iterables = (list,tuple,set,)
delimiter = '.'

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

		for key in kwargs:
			if isinstance(kwargs[key],dict) and all(isinstance(attr,str) for attr in kwargs[key]):
				kwargs[key] = Dict(kwargs[key]) if not isinstance(kwargs[key],Dict) else kwargs[key]

		super().__init__(*args,**kwargs)
		self.__dict__ = self
		return

def logger(*args,verbose=None,**kwargs):
	'''
	logger
	Args:
		args (iterable): Positional arguments for logger
		kwargs (dict): Keyword arguments for logger
		verbose (int,str,bool): Verbosity
	'''
	if verbose:
		print(*args)
	return

def call(*args,path=None,env=None,stdin=None,stdout=None,stderr=None,shell=None,execute=False,verbose=None):
	'''
	Submit call to command line of the form $> args
	Args:
		args (iterable[iterable[str]]],iterable[str]): Arguments to pass to command line {arg:value} or {arg:[value]} or [value], nested iterables are piped		
		path (str): Path to call from
		env (dict[str,str]): Environmental variables for args		
		stdin (file): Stdinput stream to command
		stdout (file): Stdoutput to command
		stderr (file): Stderr to command
		shell (bool) : Use shell subprocess
		execute (boolean,int): Boolean whether to call commands
		verbose (int,str,bool): Verbosity
	Returns:
		result (object): Return of commands
	'''

	def caller(args,inputs=None,outputs=None,errors=None,env=None,shell=None,verbose=None):

		def run(args,stdin=None,stdout=None,stderr=None,env=None,shell=None):
			env = {**environ(),**env} if env is not None else None
			args = [' '.join(args)] if shell else args
			try:
				result = subprocess.Popen(args,stdin=stdin,stdout=stdout,stderr=stderr,env=env,shell=shell)
			except (OSError,FileNotFoundError) as exception:
				result = Popen(args,stdin=stdin,stdout=stdout,stderr=stderr,env=env,shell=shell)
				logger(exception,verbose=verbose)
			return result


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

			result = run(arg,stdin=stdin,stdout=stdout,stderr=stderr,env=env,shell=shell)

			if stdin is not None:
				stdin.close()
			if isinstance(output,str):
				stdout.close()
			if isinstance(error,str):
				stderr.close()

			stdin = result.stdout


		stdout,stderr,returncode = [],[],result.returncode
		
		if result.stdout is not None:
			try:
				for line in result.stdout:
					stdout.append(parse(line))			
					logger(stdout[-1],verbose=verbose)
			except:
				stdout.append(parse(result.stdout))
				logger(stdout[-1],verbose=verbose)
		
		returncode = result.wait()

		if result.stderr is not None:
			try:
				for line in result.stderr:	
					stderr.append(parse(line))
					if returncode is not None:
						logger(stderr[-1],verbose=verbose)
			except:
				stderr.append(parse(result.stderr))
				logger(stderr[-1],verbose=verbose)

		stdout,stderr,returncode = wrap(stdout,stderr,returncode)

		return stdout,stderr,returncode

	def wrapper(stdout,stderr,returncode,env=None,shell=None,verbose=None):
		result = stdout
		return result

	def parser(*args,env=None,verbose=None):

		args = [((str(arg) if arg is not None else '') if isinstance(arg,scalars) else 
	 		    [(str(subarg) if subarg is not None else '') for subarg in arg]) for arg in args]

		pipe = any(not isinstance(arg,scalars) for arg in args)

		if pipe:
			args = [[str(subarg) for subarg in arg] if not isinstance(arg,scalars) else [arg] for arg in args]
		else:
			args = [[str(arg) for arg in args]]

		cmd = ' | '.join([
			' '.join([subarg if ' ' not in subarg else '"%s"'%(subarg) for subarg in arg])
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
				input = ' '.join(arg[index+1:])
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

	msg = '%s : %s'%(path,cmd) if path is not None else '%s'%(cmd)
	logger(msg,verbose=verbose)

	if execute:
		with cd(path):
			result = wrapper(*caller(args,inputs=inputs,outputs=outputs,errors=errors,env=env,shell=shell,verbose=verbose),env=env,shell=shell,verbose=verbose)

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
				delimiter = ','
				iterable = isinstance(values,list)
				if not iterable:
					values = [values]
				for value in values:
					for val in str(value).split(delimiter):
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
				'--%s'%(argument.replace('--','')):{**default(argument)}
				for argument in arguments
			}
		else:
			arguments = {
				'--%s'%(argument.replace('--','')):{
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


			name = '%s'%(argument.replace('--',''))
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

			name = '--%s'%(argument.replace('--',''))
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

def join(*paths,ext=None):
	'''
	Join paths
	Args:
		paths (str): path
		ext (str): join ext
	Returns:
		path (str): Joined path
	'''
	try:
		path = os.path.join(*paths)
		path = path if ext is None else delimiter.join([path,ext])
	except:
		path = None
	return path

def split(path,ext=None):
	'''
	Split paths
	Args:
		path (str): path
		ext (bool): split ext
	Returns:
		paths (iterable[str]): Split paths
	'''
	try:
		paths = path.split(os.sep)
		paths = paths if not ext else [*paths[:-1],delimiter.join(paths[-1].split(delimiter)[:-1]),delimiter.join(paths[-1].split(delimiter)[-1:])]
	except:
		paths = None
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

	path = join(*paths[:-1],join(split(paths[-1],ext=True)[:-1],string,split(paths[-1],ext=True)[-1]))

	return path

def exists(path):
	'''
	Check if path exists
	Args:
		path (str): path
	Returns:
		exists (bool): Path exists
	'''

	path = os.path.abspath(os.path.expandvars(os.path.expanduser(path))) if path is not None else None

	try:
		exists = os.path.exists(path)
	except:
		exists = False

	return exists

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

	if not exists(path):
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
	logger('Load: %r'%(path),verbose=verbose)
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
	logger('Dump: %r'%(path),verbose=verbose)
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



def setup(settings,*args,**kwargs):
	'''
	Setup
	Args:
		settings (dict): settings
	'''

	options = settings.workflow.options

	permutations = {attr:settings.workflow.settings[attr].string for attr in settings.workflow.settings}

	for index,permutation in enumerate(permuter(permutations)):

		name = settings.workflow.path.format.format(**permutation)

		for host in settings.workflow.hosts:
			path = join(settings.workflow.path.cwd.format(host=host),name)
			if exists(path):
				break
			else:
				host = None

		if host is None:
			continue

		path = join(settings.workflow.path.path,name)
		if not exists(path):
			mkdir(path,**options)

		path = join(settings.workflow.path.path,name,host)
		if not exists(path):
			touch(path,**options)

		for path in settings.workflow.path.data:
			source = join(settings.workflow.path.cwd.format(host=host),name,path)
			destination = join(settings.workflow.path.path,name,path)
			if exists(source) and not exists(destination):
				cp(source,destination,**options)

		for path in settings.workflow.path.settings:
			source = join(settings.workflow.path.cwd.format(host=host),name,path)
			destination = join(settings.workflow.path.path,name,path)
			if exists(source) and not exists(destination):
				cp(source,destination,**options)

	return


def submit(settings,*args,**kwargs):
	'''
	Submit
	Args:
		settings (dict): settings
	'''

	options = settings.workflow.options

	permutations = {attr:settings.workflow.settings[attr].string for attr in settings.workflow.settings}

	for index,permutation in enumerate(permuter(permutations)):

		name = settings.workflow.path.format.format(**permutation)

		for host in settings.workflow.hosts:
			path = join(settings.workflow.path.cwd.format(host=host),name)
			if exists(path):
				break
			else:
				host = None

		if host is None:
			continue

		logger(name,host,**options)

	return

def main(settings,*args,**kwargs):

	def config(settings):
		
		options=dict(wrapper=Dict)
		try:
			settings = load(settings,**options)
		except:
			settings = None

		options = dict(setup=setup,submit=submit)
		try:
			function = options.get(settings.workflow.function)
		except:
			function = None

		return function,settings

	function,settings = config(settings)

	try:
		function(settings,*args,**kwargs)
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
		}		

	args = argparser(arguments)

	main(*args,**args)