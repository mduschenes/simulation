#!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,inspect,traceback,datetime,re
import shutil
import glob as globber
import importlib
import json,pickle,h5py
import numpy as np
import pandas as pd

from natsort import natsorted

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,concatenate,padding
from src.utils import to_repr,to_eval
from src.utils import returnargs
from src.utils import arrays,scalars,iterables,nan,delim

from src.iterables import getter,setter

# Logging
from src.logger	import Logger
logger = Logger()
info = 100	
debug = 0

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

def contains(string,pattern):
	'''
	Search for pattern in string
	Args:
		string (str): String to search
		pattern (str): Pattern to search
	Returns:
		boolean (bool): String contains pattern
	'''
	replacements = {'\\':'\\\\','.':'\\.','*':'.*',}
	for replacement in replacements:
		pattern = pattern.replace(replacement,replacements[replacement])
		
	boolean = re.fullmatch(pattern,string) is not None

	return boolean


def exists(path):
	'''
	Check if path exists
	Make path
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


def dirname(path,abspath=False,delimiter='.'):
	'''
	Return directory name of path
	Make path
	Args:
		path (str): path
		abspath (bool): Return absolute path of directory
		delimiter (str): Delimiter to separate file name from extension		
	Returns:
		directory (bool): Directory name of path
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

	return directory

def relpath(path,relative=None,**kwargs):
	'''
	Get path relative to path
	Args:
		path (str): Path to get relative path
		relative (str): Relative path
		kwargs (dict): Additional keyword arguments
	Returns:
		path (str): Relative path
	'''
	if relative is None:
		relative = cwd()

	try:
		path = os.path.relpath(path,relative)
	except ValueError:
		pass

	return path
	

def mkdir(path):
	'''
	Make path
	Args:
		path (str): path
	'''
	directory = split(path,directory=True,abspath=True)

	if directory not in ['',None] and not exists(directory):
		os.makedirs(directory)

	return

def rm(path):
	'''
	Remove path
	Args:
		path (str): path
	'''

	try:
		os.remove(path)
	except Exception as exception:
		try:
			os.rmdir(path)
		except:
			pass

	return


def cp(source,destination):
	'''
	Copy paths
	Args:
		source (str): Source path
		destination (str): Destination path
	'''
	mkdir(destination)
	shutil.copy2(source,destination)

	return

def split(path,directory=False,file=False,ext=False,directory_file_ext=False,directory_file=False,file_ext=False,abspath=None,delimiter='.'):
	'''
	Split path into directory,file,ext
	Args:
		path (str): Path to split
		directory (bool,int,iterable[int]): Return split directory name, or number of intermediate directories after root / before directory containing folder
		file (bool): Return split file name
		ext (bool): Return split extension name
		directory_file_ext (bool): Return split and joined directory and file name and ext
		directory_file (bool): Return split and joined directory and file name
		file_ext (bool): Return split and joined file and extension name
		abspath (bool): Return absolute directory		
		delimiter (str): Delimiter to separate file name from extension
	Returns:
		paths (iterable): Split path,directory,file,ext depending on booleans
	'''	

	path = str(path) if path is not None else None

	returns = {'directory':directory,'file':file or directory_file_ext or directory_file or file_ext,'ext':ext}
	paths = {}

	if path is None or not (directory or file or ext or file_ext or directory_file):
		paths = tuple((path for k in returns if returns[k]))
		if len(paths) == 0:
			paths = (None,)
		return returnargs(paths)

	if not isinstance(directory,bool):
		if isinstance(directory,int):
			slices = slice(directory-1,None) if directory > 0 else slice(None,directory) 
		else:
			slices = slice(*(directory[0],directory[1],*directory[2:]))
		paths['directory'] = os.sep.join(dirname(path).split(os.sep)[slices])
	else:
		paths['directory'] = dirname(path)
	if abspath:
		paths['directory'] = os.path.abspath(os.path.expandvars(os.path.expanduser(paths['directory'])))
	paths['file'],paths['ext'] = os.path.splitext(path)
	if paths['ext'].startswith(delimiter):
		paths['ext'] = delimiter.join(paths['ext'].split(delimiter)[1:])
	if not directory_file:
		if dirname(path) == paths['file']:
			paths['file'] = None
		else:
			paths['file'] = os.path.basename(os.path.expandvars(paths['file']))
	if file_ext:
		if paths['file'] is not None:
			paths['file'] = delimiter.join([paths['file'],paths['ext']])
	
	paths = tuple((paths[k] for k in paths if returns[k]))
	
	return returnargs(paths)

def join(*paths,ext=None,abspath=False,delimiter='.',root=None):
	'''
	Join paths into path, with optional extension
	Args:
		paths (iterable[str]): Paths to join
		ext (str): Extension to add to path
		abspath (bool): Return absolute joined path
		delimiter (str): Delimiter to separate file name from extension
		root (str): Root path to insert at beginning of path if path does not already start with root
	Returns:
		paths (str): Joined path
	'''	
	paths = [str(path) for path in paths if path not in ['',None]]
	if len(paths)>0:
		path = os.path.join(*paths)
	else:
		path = None
	if path is not None and ext is not None and not path.endswith('%s%s'%(delimiter,ext)):
		paths = [path,ext]
		paths = [path for path in paths if path not in ['',None]]		
		path = delimiter.join(paths)

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

def scan(path,pattern=None,**kwargs):
	'''
	Recursively search for paths
	Args:
		path (str): Path
		pattern (str): Pattern of paths to search
		kwargs (dict): Keyword arguments for searching paths
	Yields:
		path (str): Path matching pattern
	'''
	for path in os.scandir(path):
		recursive = path.is_dir(**kwargs)
		path = path.path
		if recursive:
			yield from scan(path,pattern=pattern,**kwargs)
		elif contains(path,pattern):
			yield path

def wildcard(path,pattern='*'):
	'''
	Get base directory with pattern
	Args:
		path (str): Path
		pattern (str): Pattern of sub-directories
	Returns:
		path (str): Path of base directory
	'''
	while pattern in os.path.basename(path):
		path = os.path.dirname(path)
	if not path:
		path = '.'
	return path
			
def glob(path,include=None,recursive=False,default=None,**kwargs):

	'''
	Expand path
	Args:
		path (str): Path to expand
		include (str,callable): Type of paths to expand, allowed ['directory','file'] or callable with signature include(path)
		recursive (bool,str): Recursively find all included paths below path, or expander strings ['*','**']
		default (str): Default path to return
		kwargs (dict): Additional glob keyword arguments
	Yields:
		path (str): Expanded, absolute paths
	'''


	if include in ['file']:
		include = os.path.isfile
	elif include in ['directory']:
		include = os.path.isdir
	elif isinstance(include,str):
		include = lambda path,include=include: contains(path,include)

	if not isinstance(recursive,str):
		if recursive:
			recursive = '**'
		else:
			recursive = None

	path = join(path,recursive)

	path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))

	path,pattern = os.path.dirname(path),os.path.basename(path)
	path = wildcard(path)

	if ('*' not in path) and (not exists(path)):
		path = (path for path in [default])
	else:
		path = scan(path,pattern=pattern,**kwargs)
		# path = globber.iglob(path,recursive=True,**kwargs)

	if include is not None:
		path = natsorted(filter(include,path))

	yield from path


def glob(path,include=None,recursive=False,default=None,**kwargs):
	'''
	Expand path
	Args:
		path (str): Path to expand
		include (str,callable): Type of paths to expand, allowed ['directory','file'] or callable with signature include(path)
		recursive (bool,str): Recursively find all included paths below path, or expander strings ['*','**']
		default (str): Default path to return
		kwargs (dict): Additional glob keyword arguments
	Returns:
		path (generator[str]): Expanded, absolute paths
	'''

	if include in ['file']:
		include = os.path.isfile
	elif include in ['directory']:
		include = os.path.isdir
	elif isinstance(include,str):
		include = lambda path,include=include: contains(path,include)

	if not isinstance(recursive,str):
		if recursive:
			recursive = '**'
		else:
			recursive = None

	path = join(path,recursive)

	path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))

	if ('*' not in path) and (not exists(path)):
		path = (path for path in [default])
	else:
		path = globber.iglob(path,recursive=True,**kwargs)


	if include is not None:
		path = list(natsorted(filter(include,path)))

	return path

def edit(path,directory=None,file=None,ext=None,delimiter='.'):
	'''
	Edit directory,file,ext of path
	Args:
		path (str): Path to split
		directory (callable,str): Function to edit directory with signature(directory,file,ext,delimiter), or string to replace directory
		file (callable,str): Function to edit file with signature(directory,file,ext,delimiter), or string to replace file
		ext (callable,str): Function to edit ext with signature(directory,file,ext,delimiter), or string to replace ext
		delimiter (str): Delimiter to separate file name from extension
	Returns:
		path (str): Edited path
	'''	

	path = split(path,directory=True,file=True,ext=True,delimiter=delimiter)

	if directory is None:
		directory = path[0]
	elif callable(directory):
		directory = directory(*path,delimiter=delimiter)

	if file is None:
		file = path[1]
	elif callable(file):
		file = file(*path,delimiter=delimiter)

	if ext is None:
		ext = path[2]
	elif callable(ext):
		ext = ext(*path,delimiter=delimiter)

	path = join(directory,file,ext=ext,delimiter=delimiter)

	return path

def basename(path,**kwargs):
	'''
	Get base file name from path
	Args:
		path (str): Path
		kwargs (dict): Additional path keyword arguments
	Returns:
		file (str): Base filename
	'''	
	return os.path.basename(os.path.splitext(path)[0])


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

 
class funcclass(object):
	'''
	Class wrapper for functions
	Args:
		func (callable): Function to wrap
	'''	
	def __init__(self,func=lambda x:x):
		self.func = func
	def __call__(self,*args,**kwargs):
		return self.func(*args,**kwargs)

class encode_json(json.JSONEncoder):
	def default(self, obj):
		encode = super().encode
		default = super().default
		if isinstance(obj,arrays):
			return obj.tolist()
		elif isinstance(obj,np.bool_):
			return encode(bool(obj))
		else:
			return default(default(obj))

class decode_json(json.JSONDecoder):
	def default(self, obj):
		encode = super().encode
		default = super().default
		return default(obj)


def dump_json(obj,key='py/object',wr='w',ext='json',**kwargs):
	'''
	Serialize objects into json
	Args:
		obj (object): Object to serialize
		key (str): Key to serialize on
		wr (str): Dump mode
		ext (str): Extension type of object
		kwargs (dict): Additional loading keyword arguments		
	Returns:
		obj (object): Serialized object
	'''	
	return obj

def load_json(obj,key='py/object',wr='r',ext='json',**kwargs):
	'''
	De-serialize objects into json
	Args:
		obj (object): Object to de-serialize
		key (str): Key to de-serialize on
		wr (str): Read mode
		ext (str): Extension type of object
		kwargs (dict): Additional loading keyword arguments		
	Returns:
		obj (object): De-serialized object
	'''	
	if isinstance(obj,dict) and key in obj:
		obj = pickle.loads(str(obj[key]))
	return obj

def load_hdf5(obj,wr='r',ext='hdf5',**kwargs):
	'''
	Load objects from path into hdf5
	Args:
		obj (str,object): Path or file object to load object
		wr (str): Read mode
		ext (str): Extension type of object
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object): Loaded object
	'''		
	if isinstance(obj,str):
		with h5py.File(obj,wr) as file:
			data = _load_hdf5(file,wr=wr,ext=ext,**kwargs)
	else:
		file = obj
		data = _load_hdf5(file,wr=wr,ext=ext,**kwargs)
	return data


def _load_hdf5(obj,wr='r',ext='hdf5',**kwargs):
	'''
	Load objects from path into hdf5
	Args:
		obj (object): hdf5 object to load object
		wr (str): Read mode
		ext (str): Extension type of object
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object): Loaded object
	'''	

	data = {}
	
	if isinstance(obj, h5py._hl.group.Group):
		names = obj
		for name in names:
			key = name
			if isinstance(obj[name], h5py._hl.group.Group):	
				data[key] = _load_hdf5(obj[name],wr=wr,ext=ext,**kwargs)
			else:
				data[key] = obj[name][...]
				if data[key].dtype.kind in ['S','O']:
					data[key] = data[key].astype(str)
				
		names = obj.attrs
		for name in names:
			key = name
			data[key] = obj.attrs[name]

	else:
		data = obj.value
	
	return data

def dump_hdf5(obj,path,wr='r',ext='hdf5',**kwargs):
	'''
	Dump objects into hdf5
	Args:
		obj (object): Object to dump
		path (str,object): Path object to dump to
		wr (str): Write mode
		ext (str): Extension type of object
		kwargs (dict): Additional loading keyword arguments
	'''		
	if isinstance(path,str):
		with h5py.File(path,wr) as file:
			_dump_hdf5(obj,file,wr=wr,ext=ext,**kwargs)
	else:	
		file = path
		_dump_hdf5(obj,file,wr=wr,ext=ext,**kwargs)

	return

def _dump_hdf5(obj,path,wr='r',ext='hdf5',**kwargs):
	'''
	Dump objects into hdf5
	Args:
		obj (object): object to dump
		path (object): hdf5 object to dump to
		wr (str): Write mode
		ext (str): Extension type of object
		kwargs (dict): Additional loading keyword arguments
	'''		

	if isinstance(obj,dict):
		names = obj
		for name in names:
			key = name
			if isinstance(obj[name],dict):
				path.create_group(key)
				_dump_hdf5(obj[name],path[key],wr=wr,ext=ext,**kwargs)
			elif isinstance(obj[name],scalars):
				try:
					path.attrs[key] = obj[name]
				except TypeError:
					pass
			else:
				try:
					path[key] = obj[name]
				except:
					path[key] = np.array(obj[name],dtype='S')
	else:
		path = obj

	return




def pickleable(obj,path=None,callables=True,verbose=False):
	'''
	Check if object can be written to file and can be pickled
	Args:
		obj (object): Object to pickle
		path (str): Path to check if object can be written to
		callables (bool): Allow functions to be pickled
	Returns:
		ispickleable (bool): Whether object can be pickled
	'''	
	if isinstance(obj,dict):
		pickleables = {k: pickleable(obj[k],path,callables=callables) for k in obj} 
		for k in pickleables:
			if not pickleables[k] or (not callables and callable(pickleables[k])):
				logger.log(debug,'Exception : Cannot pickle (key,value) %r, %r'%(k,obj[k]))
				obj.pop(k);
				pickleables[k] = True		
		return all([pickleables[k] for k in pickleables])

	ispickleable = False
	if path is None:
		path  = '__tmp__.__tmp__.%d'%(np.random.randint(1,int(1e8)))
	with open(path,'wb') as fobj:
		try:
			pickle.dump(obj,fobj)
			ispickleable = True
		except Exception as exception:
			pass
	if exists(path):
		rm(path)
	return ispickleable


def jsonable(obj,path=None,callables=False,**kwargs):
	'''
	Check if object can be written to json
	Args:
		obj (object): Object to json
		path (str): Path to check if object can be written to
		callables (bool): Allow functions to be written to json
		kwargs (dict): Additional keyword arguments		
	Returns:
		isjsonable (bool): Whether object can be written to json
	'''	

	if isinstance(obj,dict):
		jsonables = {k: jsonable(obj[k],path,callables=callables) for k in obj} 
		for k in jsonables:
			if (not isinstance(k,(str, int, float, bool))) or (not jsonables[k]) or (not callables and callable(obj[k])):
				obj.pop(k);
				jsonables[k] = True
		isjsonable = all([jsonables[k] for k in jsonables])
		return isjsonable

	isjsonable = False
	if path is None:
		path  = '__tmp__.__tmp__.%d'%(np.random.randint(1,int(1e8)))
	with open(path,'w') as fobj:
		try:
			# json.dump(obj,fobj,**{'default':dump_json,'ensure_ascii':False,'indent':4})
			# json.dump(encode_json(data,**kwargs),obj,**{'default':dump_json,'ensure_ascii':False,'indent':4,**kwargs})
			json.dump(data,obj,**{'cls':encode_json,'ensure_ascii':False,'indent':4,**kwargs})
			isjsonable = True
		except Exception as exception:
			pass
	if exists(path):
		rm(path)
	return isjsonable




def load(path,wr='r',default=None,delimiter='.',wrapper=None,verbose=False,**kwargs):
	'''
	Load objects from path
	Args:
		path (str,iterable,dict[str,str]): Path to load object
		wr (str): Read mode
		default (object): Default return object if load fails
		delimiter (str): Delimiter to separate file name from extension		
		wrapper (str,callable,iterable[str,callable]): Process data, either string in ['df','np','array','dict','merge','pd'] or callable with signature wrapper(data)
		verbose (bool,int): Verbose logging of loading
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object,iterable[object],dict[str,object]): Loaded object
	'''
	exts = ['npy','npz','csv','txt','pickle','pkl','json','hdf5','h5','ckpt']
	wrs = [wr,'r','rb']
	wrapper = wrapper if isinstance(wrapper,iterables) else [wrapper]

	args = {'path':path,'wrapper':wrapper}
	kwargs.update({'wrapper':wrapper})	

	if path is None or not isinstance(path,(str,iterables)):
		return default

	if isinstance(path,str):
		paths = [path]
	else:
		paths = path
	
	if not isinstance(path,dict):
		paths = {path: path for path in paths}
	else:
		paths = path

	paths = {(delim*3).join([name,str(path)]): path
		for name in paths
		for path in natsorted(glob(paths[name],default=(None if split(paths[name],ext=True) in exts else paths[name])))
		}

	data = {}

	for name in paths:

		path = paths[name]

		datum = default

		if not isinstance(path,str):
			data[name] = datum
			continue
	
		path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
		ext = split(path,ext=True,delimiter=delimiter)

		for wr in wrs:
			try:
				datum = _load(path,wr=wr,ext=ext,**kwargs)
				break
			except (FileNotFoundError,AttributeError,TypeError,UnicodeDecodeError,ValueError,OSError,ModuleNotFoundError,ImportError,OverflowError) as exception:			
				logger.log(debug,'Exception : %r\n%r'%(exception,traceback.format_exc()))
				try:
					with open(path,wr) as obj:
						datum = _load(obj,wr=wr,ext=ext,**kwargs)
						break
				except (FileNotFoundError,AttributeError,TypeError,UnicodeDecodeError,ValueError,OSError,ModuleNotFoundError,ImportError,OverflowError) as exception:
					logger.log(debug,'Exception : %r\n%r'%(exception,traceback.format_exc()))
					pass

		data[name] = datum

		logger.log(info*verbose,'Load : %s'%(relpath(paths[name])))

	wrappers = []
	for wrapper in kwargs['wrapper']:

		if wrapper is None:
			def wrapper(data):
				return data
		elif callable(wrapper):
			if any(i in kwargs['wrapper'] for i in ['pd']):
				continue
			else:
				try:
					if isinstance(wrapper(),dict):
						def wrapper(data,wrapper=wrapper):
							if not data:
								return data
							data = data[list(data)[0]]
							data = wrapper(data)
							return data
					else:
						pass
				except:
					pass	
		elif wrapper in ['df']:
			def wrapper(data):
				options = {**{'ignore_index':True},**{kwarg: kwargs[kwarg] for kwarg in kwargs if kwarg in ['ignore_index']}}
				def func(path,data):
					for attr in data:
						if any(isinstance(i,arrays) for i in data[attr]):
							data[attr] = [tuple(i) for i in data[attr]]
					size = max([len(data[attr]) for attr in data],default=0)
					data['__path__'] = [path]*size
					return data
				try:
					data = pd.concat((pd.DataFrame(func(path,data[path])) for path in data if data[path]),**options) #.convert_dtypes()
				except Exception as exception:
					data = default
				return data
		elif wrapper in ['np']:
			def wrapper(data):
				options = {**{},**{kwargs[kwarg] for kwarg in kwargs in kwarg in []}}
				try:
					data = np.concatenate(tuple((np.array(data[path]) for path in data)),**options)
				except ValueError:
					data = default
				return data	
		elif wrapper in ['array']:
			def wrapper(data):
				options = {**{},**{kwargs[kwarg] for kwarg in kwargs in kwarg in []}}
				try:
					data = concatenate(tuple((array(data[path]) for path in data)),**options)
				except ValueError:
					data = default
				return data
		elif wrapper in ['dict']:
			def wrapper(data):
				return data
		elif wrapper in ['merge']:
			def wrapper(data):

				data = {key:[i for path in data for i in data[path][key] if data[path] is not None] 
					for key in set((key for path in data for key in data[path]))}

				for key in data:

					if not any(i.ndim>0 for i in data[key]):
						continue

					indices = {}
					for i,tmp in enumerate(data[key]):
						index = len(tmp)
						if index not in indices:
							indices[index] = []
						indices[index].append(i)

					tmp = {index: array([data[key][index] for index in indices[index]]) for index in indices}

					shape = tuple((max(tmp[index].shape[i] for index in tmp) for i in range(min(tmp[index].ndim for index in tmp))))

					tmp = {index: padding(tmp[index],shape=shape,random='zeros',dtype=tmp[index].dtype) for index in tmp}

					indices = {i:(index,indices[index].index(i)) for index in indices for i in indices[index]}
					indices = [indices[i] for i in range(len(indices))]

					data[key] = [tmp[i[0]][i[1]] for i in indices]

				return data

		elif wrapper in ['pd']:
			def wrapper(data):
				return data	
		else:
			def wrapper(data):
				return data

		wrappers.append(wrapper)

	for wrapper in wrappers:
		data = wrapper(data)

	if isinstance(args['path'],str) and (any(((i in [None,'pd'])) for i in args['wrapper'])):
		name = list(data)[-1]
		data = data[name]
	elif not isinstance(args['path'],dict) and (any(((i in [None,'pd'])) for i in args['wrapper'])):
		data = [data[name] for name in data]
	else:
		pass

	return data



def _load(obj,wr,ext,**kwargs):
	'''
	Load objects from path or file object
	Args:
		obj (str,object): Path or file object to load object
		wr (str): Read mode
		ext (str): Extension type of object
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object): Loaded object
	'''	
	wrappers = kwargs.pop('wrapper',None)

	exts = ['npy','npz','csv','txt','pickle','pkl','json','hdf5','h5','ckpt']
	try:
		assert ext in exts, "Cannot load extension %s"%(ext)
	except Exception as exception:
		obj,module = '.'.join(obj.split('.')[:-1]),obj.split('.')[-1]
		try:
			path = os.path.basename(obj).strip('.')
			data = getattr(importlib.import_module(path),module)
		except:
			path = obj
			spec = importlib.util.spec_from_file_location(module,path)
			data = importlib.util.module_from_spec(spec)
			sys.modules[module] = data
			spec.loader.exec_module(data)
			data = getattr(data,module)

	if ext in ['npy']:
		data = np.load(obj,**{'allow_pickle':True,**kwargs})
	elif ext in ['npz']:
		data = np.load(obj,**{'allow_pickle':True,**kwargs})
	elif ext in ['csv']:
		data = getattr(pd,'read_%s'%ext)(obj,**{**kwargs})
	elif ext in ['txt']:
		data = np.loadtxt(obj,**{'delimiter':',',**kwargs})
	elif ext in ['pickle','pkl']:
		# TODO: Load specific types as wrapped types (i.e) onp.array -> np.array for JAX)
		data = pickle.load(obj,**kwargs)
	elif ext in ['json']:
		data = json.load(obj,**{'cls':decode_json,'object_hook':load_json,**kwargs})
	elif ext in ['hdf5','h5','ckpt']:
		for wrapper in list(wrappers):
			if wrapper in ['pd']:
				try:
					ext = 'hdf'
					data = getattr(pd,'read_%s'%ext)(obj,**{'key':kwargs.get('key','data')})
					break
				except:
					data = load_hdf5(obj,wr=wr,ext=ext,**kwargs)
					wrappers.append('df')
					break
			else:
				data = load_hdf5(obj,wr=wr,ext=ext,**kwargs)

	return data



def dump(data,path,wr='w',delimiter='.',wrapper=None,verbose=False,**kwargs):
	'''
	Dump objects to path
	Args:
		data (object): Object to dump
		path (str,iterable[str],dict[str,str]): Path to dump object
		wr (str): Write mode
		delimiter (str): Delimiter to separate file name from extension		
		wrapper (str,callable,iterable[str,callable]): Process data, either string in ['df','np','array','dict','merge','pd'] or callable with signature wrapper(data)
		verbose (bool,int): Verbose logging of dumping
		kwargs (dict): Additional dumping keyword arguments
	'''
	wrs = [wr,'w','wb']
	wrapper = wrapper if isinstance(wrapper,iterables) else [wrapper]

	args = {'path':path,'wrapper':wrapper}
	kwargs.update({'wrapper':wrapper})

	if path is None:
		return

	wrappers = []
	for wrapper in kwargs['wrapper']:
		if wrapper is None:
			def wrapper(data):
				return data
		elif callable(wrapper):
			if any(i in kwargs['wrapper'] for i in ['pd']):
				continue
			else:
				pass
		elif wrapper in ['df']:
			def wrapper(data):
				return pd.DataFrame(data)
		elif wrapper in ['np']:
			def wrapper(data):
				return np.array(data)
		elif wrapper in ['array']:
			def wrapper(data):
				return array(data)
		elif wrapper in ['dict']:
			def wrapper(data):
				try:
					if isinstance(data,pd.DataFrame):
						return data.to_dict(orient='list')
					else:
						raise AttributeError
				except:
					return data
		elif wrapper in ['merge']:
			def wrapper(data):
				return data			
		elif wrapper in ['pd']:
			def wrapper(data):
				return data
		else:
			def wrapper(data):
				return data

		wrappers.append(wrapper)

	if isinstance(path,str):
		paths = [path]
	else:
		paths = path
	
	if not isinstance(path,dict):
		paths = {path: path for path in paths}
	else:
		paths = path

	for name in paths:
		
		path = paths[name]
		
		path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
		ext = split(path,ext=True,delimiter=delimiter)
		mkdir(path)

		for wrapper in wrappers:
			data = wrapper(data)

		for wr in wrs:	
			try:
				_dump(data,path,wr=wr,ext=ext,**kwargs)
				break
			except (ValueError,AttributeError,TypeError,OSError,ModuleNotFoundError,ImportError,OverflowError) as exception:
				logger.log(debug,'Exception : %r\n%r'%(exception,traceback.format_exc()))
				try:
					with open(path,wr) as obj:
						_dump(data,obj,wr=wr,ext=ext,**kwargs)
					break
				except (ValueError,AttributeError,TypeError,OSError,ModuleNotFoundError,ImportError,OverflowError) as exception:
					logger.log(debug,'Exception : %r\n%r'%(exception,traceback.format_exc()))
					pass
		
		logger.log(info*verbose,'Dump : %s'%(relpath(paths[name])))

	return



# Dump data - General file export
def _dump(data,obj,wr,ext,**kwargs):
	'''
	Dump objects to path or file object
	Args:
		data (object): Object to dump
		obj (str,object): Path or file object to dump object
		wr (str): Write mode
		kwargs (dict): Additional dumping keyword arguments
	'''	

	wrappers = kwargs.pop('wrapper',None)

	exts = ['npy','npz','csv','txt','pickle','pkl','json','tex','hdf5','h5','ckpt','pdf']
	assert ext in exts, "Cannot dump extension %s"%(ext)

	if ext in ['npy']:
		np.save(obj,data,**{'allow_pickle':True,**kwargs})
	if ext in ['npz']:
		if isinstance(data,dict):
			np.savez(obj,**data)
		elif isinstance(data,(tuple)):
			np.savez(obj,*data)
		else:
			np.savez(obj,data)
	elif ext in ['csv']:
		getattr(data,'to_%s'%ext)(obj,**{'index':False,**kwargs})
	elif ext in ['txt']:
		np.savetxt(obj,data,**{'delimiter':',','fmt':'%.20f',**kwargs})
	elif ext in ['pickle','pkl']:		
		pickleable(data,callables=kwargs.pop('callables',True))
		pickle.dump(data,obj,protocol=pickle.HIGHEST_PROTOCOL,**kwargs)
	elif ext in ['json']:
		# jsonable(data,callables=kwargs.pop('callables',False))	
		# json.dump(data,obj,**{'default':dump_json,'ensure_ascii':False,'indent':4,**kwargs})
		# json.dump(encode_json(data,**kwargs),obj,**{'default':dump_json,'ensure_ascii':False,'indent':4,**kwargs})
		json.dump(data,obj,**{'cls':encode_json,'ensure_ascii':False,'indent':4,**kwargs})
	elif ext in ['tex']:
		obj.write(data,**kwargs)
	elif ext in ['hdf5','h5','ckpt']:
		for wrapper in wrappers:
			if wrapper in ['pd']:
				ext = 'hdf'
				getattr(data,'to_%s'%ext)(obj,**{'key':kwargs.get('key','data'),'mode':'w'})
				break
			else:
				dump_hdf5(data,obj,wr=wr,ext=ext,**kwargs)
	elif ext in ['pdf']:
		data.savefig(obj,**{**kwargs})

	return

def append(data,path,wr='r',delimiter='.',wrapper=None,verbose=False,**kwargs):
	'''
	Append objects to path
	Args:
		data (object,callable): Object to append or function to append with signature data(path,obj)
		path (str,iterable[str],dict[str,str]): Path to append object
		wr (str): Write mode
		delimiter (str): Delimiter to separate file name from extension		
		wrapper (str,callable,iterable[str,callable]): Process data, either string in ['df','np','array','dict','merge','pd'] or callable with signature wrapper(data)
		verbose (bool,int): Verbose logging of appending
		kwargs (dict): Additional appending keyword arguments
	'''

	exts = ['npy','npz','csv','txt','pickle','pkl','json','hdf5','h5','ckpt']

	if isinstance(path,str):
		paths = [path]
	else:
		paths = path
	
	if not isinstance(path,dict):
		paths = {path: path for path in paths}
	else:
		paths = path

	paths = {(delim*3).join([name,str(path)]): path
		for name in paths
		for path in natsorted(glob(paths[name],default=(None if split(paths[name],ext=True) in exts else paths[name])))
		}

	if not callable(data):

		if isinstance(data,str):
			tmp = load(data)
		if tmp is not None:
			data = tmp
		def _append(path,obj,data=data):
			setter(obj,data,delimiter=delim)
			return obj
	else:
		def _append(path,obj,data=data):
			obj = data(path,obj)
			return obj

	for name in paths:

		path = paths[name]

		obj = load(path,wr=wr,delimiter=delimiter,wrapper=wrapper,verbose=verbose,**kwargs)
		
		obj = _append(path,obj)

		dump(obj,path,delimiter=delimiter,verbose=verbose,**kwargs)

	return


def convert(data,path,wr='r',delimiter='.',wrapper=None,verbose=False,**kwargs):
	'''
	Convert objects to path
	Args:
		data (str): Existing path
		path (str,iterable[str],dict[str,str]): Path to convert object
		wr (str): Write mode
		delimiter (str): Delimiter to separate file name from extension		
		wrapper (str,callable,iterable[str,callable]): Process data, either string in ['df','np','array','dict','merge','pd'] or callable with signature wrapper(data)
		verbose (bool,int): Verbose logging of appending
		kwargs (dict): Additional appending keyword arguments
	'''

	exts = ['npy','npz','csv','txt','pickle','pkl','json','hdf5','h5','ckpt']

	if isinstance(data,str):
		datas = [data]
	else:
		datas = data
	
	if not isinstance(data,dict):
		datas = {data: data for data in datas}
	else:
		datas = data

	paths = {(delim*3).join([name,str(data)]): join(split(data,directory=True),split(path,file_ext=True))
		for name in datas
		for data in natsorted(glob(datas[name],default=(None if split(datas[name],ext=True) in exts else datas[name])))
		}

	datas = {(delim*3).join([name,str(data)]): data
		for name in datas
		for data in natsorted(glob(datas[name],default=(None if split(datas[name],ext=True) in exts else datas[name])))
		}

	def _convert(path,data):
		try:
			try:
				data = load(data,wr=wr,delimiter=delimiter,wrapper=wrapper,verbose=verbose,**kwargs)
			except Exception as exception:
				data = load(data,wr=wr,delimiter=delimiter,wrapper=None,verbose=verbose,**kwargs)

			dump(data,path,delimiter=delimiter,verbose=verbose,**kwargs)
		except Exception as exception:
			pass
		return


	for name in datas:

		data = datas[name]
		path = paths[name]

		_convert(path,data)

	return


def merge(data,path,wr='r',delimiter='.',wrapper=None,verbose=False,**kwargs):
	'''
	Merge objects to path
	Args:
		data (str,iterable[str],dict[str,str]): Path to convert object
		path (str,iterable[str],dict[str,str]): Path to convert object
		wr (str): Write mode
		delimiter (str): Delimiter to separate file name from extension		
		wrapper (str,callable,iterable[str,callable]): Process data, either string in ['df','np','array','dict','merge','pd'] or callable with signature wrapper(data)
		verbose (bool,int): Verbose logging of appending
		kwargs (dict): Additional appending keyword arguments
	'''

	wrappers = ['merge']

	wrapper = [wrapper,*wrappers] if not isinstance(wrapper,iterables) else [*wrapper,*wrappers]

	wr = wr
	data = load(data,wr=wr,delimiter=delimiter,wrapper=wrapper,verbose=verbose,**kwargs)

	wr = 'w'
	dump(data,path,wr=wr,delimiter=delimiter,verbose=verbose,**kwargs)

	return

def cleanup(data,path,verbose=True):
	'''
	Process path
	Args:
		data (iterable[str]): Attributes to remove
		path (str): Path to remove attributes
		verbose (bool): Verbosity
	'''

	for path in glob(path):

		obj = load(path,verbose=verbose)

		for attr in data:
			if attr not in obj:
				continue
			obj.pop(attr)

		dump(obj,path)

	return

def update(dictionary,field,func):
	"""
	Update nested dictionary field with function in-place
	Args:
		dictionary (dict): Dictionary to update
		field (object): field key to update
		func (callable): function with signature func(key,value) that returns updated value
	"""
	if not isinstance(dictionary,dict):
		return
	for key in dictionary:
		if key == field:
			dictionary[key] = func(key,dictionary[key])
		else:
			update(dictionary[key],field,func)
	return


def check(dictionary,updates):
	"""
	Update nested dictionary field with function in-place
	Args:
		dictionary (dict): Dictionary to update
		updates (dict): Dictionary of updates {field: func} with signature func(key,value) that returns updated value
	"""
	for field in updates:
		update(dictionary,field,updates[field])
	return


def setup(args,defaults=[]):
	'''		
	Setup parameters based on CLI args
	Args:
		args (list): list of CLI arguments
		defaults (list,str): default list of CLI arguments or path to load defaults
	Returns:
		returns (list): returns based on args
	'''


	# Update params
	updates = {}

	# field = "verbose"
	# def func(key,value):
	# 	levels = {
	# 		'notset':0,'debug':10,'info':20,'warning':30,'error':40,'critical':50,
	# 		'Notset':0,'Debug':10,'Info':20,'Warning':30,'Error':40,'Critical':50,
	# 		'NOTSET':0,'DEBUG':10,'INFO':20,'WARNING':30,'ERROR':40,'CRITICAL':50,
	# 		10:10,20:20,30:30,40:40,50:50,
	# 		2:20,3:30,4:40,5:50,
	# 		True:20,False:0,None:0,
	# 		}
	# 	value = levels.get(value,value)
	# 	return value		
	# updates[field] = func

	# field = "logging"
	# def func(key,value):
	# 	import logging.config
	# 	logging.config.dictConfig({**value,**{"disable_existing_loggers":False}})
	# 	return value
	# updates[field] = func


	# Update defaults
	if isinstance(defaults,str):
		defaults = load(defaults,default={})

	n = max(len(args),len(defaults))
	returns = ()

	for i in range(n):
		default = defaults[i] if (defaults is not None and len(defaults) > i) else None
		value = args[i] if (args is not None and len(args) > i) else default
		value = load(value,default=default)
		check(value,updates)
		returns += (value,)

	return returnargs(returns)

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