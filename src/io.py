#!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,inspect,traceback,re
import shutil
import glob as globber
from braceexpand import braceexpand
from filelock import FileLock as FileLock
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

from src.utils import array,concatenate,padding,slicer
from src.utils import to_repr,to_eval
from src.utils import returnargs,isinstances
from src.utils import arrays,scalars,iterables,nan,delim

from src.iterables import getter,setter

# Logging
from src.logger	import Logger
logger = Logger()
info = 100	
debug = int(os.environ.get("PY_DEBUG",0))

delimiter = '.'

class Lock(object):
	'''
	Lock file class
	Args:
		lock (bool,str): Lock file path or extension
		path (str): File path
		timeout (int): Timeout (seconds)
		kwargs (dict): Additional keyword arguments
	'''
	
	lock = 'lock'
	path = 'lock'
	timeout = -1

	def __new__(cls,lock,path=None,timeout=None,**kwargs):
		if not lock or not path:
			self = super().__new__(cls,**kwargs)
		else:
			lock = delimiter.join([path if path is not None else cls.path,lock if isinstance(lock,str) else cls.path])
			timeout = timeout if timeout is not None else cls.timeout
			self = 	FileLock(lock,timeout=timeout,**kwargs)
		return self
	def __init__(self,*args,**kwargs):
		return
	def __enter__(self):
		return
	def __exit__(self,etype,value,traceback):
		return
	def acquire(self,*args,**kwargs):
		return
	def release(self,*args,**kwargs):
		return

class Backup(object):
	'''
	Backup file class
	Args:
		backup (bool,str): Backup file path
		path (str): File path
		boolean (callable): Delete backup file, with signature boolean(etype,value,traceback) -> bool
		kwargs (dict): Additional keyword arguments
	'''

	backup = 'bkp'
	path = None
	def boolean(self,etype,value,traceback):
		return etype is None
	
	def __init__(self,backup,path=None,boolean=None,**kwargs):
		self.backup = backup if isinstance(backup,str) else delimiter.join([path,self.backup]) if backup else None
		self.path = path if path is not None else self.path
		self.boolean = boolean if boolean is not None else self.boolean
		return
	def __enter__(self):
		if self.backup is not None:
			if exists(self.path):
				cp(self.path,self.backup)
			elif exists(self.backup):
				cp(self.backup,self.path)
		return
	def __exit__(self,etype,value,traceback):
		if self.backup is not None and self.boolean(etype,value,traceback):
			rm(self.backup)
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
			path = os.path.expandvars(os.path.expanduser(self.path))
			os.chdir(path)
		except:
			pass
		return
	def __exit__(self,etype,value,traceback):
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

def memory(path):
	'''
	Get size of path
	Args:
		path (str): path
	Returns:
		size (int): Size of path
	'''

	try:
		size = os.path.getsize(path)
	except:
		size = None

	return size

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
		'tmp','bkp',
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
		try:
			os.makedirs(directory)
		except FileExistsError:
			pass

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
		except Exception as exception:
			try:
				shutil.rmtree(path)
			except Exception as exception:
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

def split(path,directory=False,file=False,ext=False,directory_file_ext=False,directory_file=False,file_ext=False,abspath=None,delimiter=delimiter):
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

def join(*paths,ext=None,root=None,abspath=False,delimiter=delimiter):
	'''
	Join paths into path, with optional extension
	Args:
		paths (iterable[str]): Paths to join
		ext (str): Extension to add to path
		root (str): Root path to insert at beginning of path if path does not already start with root
		abspath (bool): Return absolute path
		delimiter (str): Delimiter to separate file name from extension
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
		path (str,iterable[str]): Path to expand
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

	paths = [path] if isinstance(path,str) else path

	paths = (join(name,recursive) for path in paths for name in braceexpand(path))

	paths = (os.path.abspath(os.path.expandvars(os.path.expanduser(path))) for path in paths)

	path = (i for path in paths 
		for i in (
			[default] if (('*' not in path) and (not exists(path)))
			else
			globber.iglob(path,recursive=True,**kwargs)
			)
		)

	if include is not None:
		path = list(natsorted(filter(include,path)))

	return path

	# path = join(path,recursive)

	# path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))

	# path,pattern = os.path.dirname(path),os.path.basename(path)
	# path = wildcard(path)

	# if ('*' not in path) and (not exists(path)):
	# 	path = (path for path in [default])
	# else:
	# 	path = scan(path,pattern=pattern,**kwargs)
	# 	# path = globber.iglob(path,recursive=True,**kwargs)

	# if include is not None:
	# 	path = natsorted(filter(include,path))

	# yield from path


def edit(path,directory=None,file=None,ext=None,delimiter=delimiter):
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

class funcclass(object):
	'''
	Class wrapper for functions
	Args:
		func (callable): Function to wrap
	'''	
	def __init__(self,func=lambda x:x):
		self.func = func
		return
	def __call__(self,*args,**kwargs):
		return self.func(*args,**kwargs)

class encode_json(json.JSONEncoder):
	def default(self, obj):
		encode = super().encode
		default = super().default
		if isinstance(obj,arrays):
			return obj.tolist()
		elif isinstance(obj,(np.integer,np.int64,np.int32,np.uint32,)):
			return int(obj)
		elif isinstance(obj,(np.floating,np.float64,np.float32,)):
			return float(obj)
		elif isinstance(obj,(np.bool_,)):
			return bool(obj)
		else:
			return default(default(obj))

class decode_json(json.JSONDecoder):
	def default(self, obj):
		encode = super().encode
		default = super().default
		return default(obj)


def serialize_json(obj,key='py/object',wr='r',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	De-serialize object
	Args:
		obj (object): Object
		key (str): Key to de-serialize on
		wr (str): Read mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments		
	Returns:
		obj (object): Serialized object
	'''	
	
	if isinstance(obj,dict) and key in obj:
		obj = pickle.loads(str(obj[key]))
	
	return obj

def deserialize_json(obj,key='py/object',wr='w',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	DeSerialize object
	Args:
		obj (object): Object
		key (str): Key to serialize on
		wr (str): Dump mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform		
		execute (bool): Execute
		verbose (bool,int): Verbosity
		kwargs (dict): Additional loading keyword arguments		
	Returns:
		obj (object): De-Serialized object
	'''	
	return obj


def load_json(path,wr='r',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Load object
	Args:
		path (str,object): Path or file object
		wr (str): Read mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform		
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments	
	Returns:
		data (object): Object
	'''
	
	if isinstance(path,str):
		raise ValueError
	else:
		data = json.load(path,**options)
	
	return data

def dump_json(data,path,wr='w',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Dump object
	Args:
		data (object): Object
		path (str,object): Path or file object
		wr (str): Write mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform		
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments
	'''

	if isinstance(path,str):
		raise ValueError
	else:	
		json.dump(data,path,**options)

	return

def merge_json(data,path,wr='a',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Merge object
	Args:
		data (object): Object
		path (str,object): Path or file object
		wr (str): Write mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments
	'''

	wr = 'r%s'%(wr[1:]) if wr is not None else wr
	with (open(path,wr) if isinstance(path,str) else path) as obj:
		try:
			tmp = load_json(obj,wr=wr,ext=ext,options=options['load'],transform=transform,execute=execute,verbose=verbose,**kwargs)
		except:
			tmp = {}	

	_merge_json(data,tmp,wr=wr,ext=ext,options=options['load'],transform=transform,execute=execute,verbose=verbose,**kwargs)

	wr = 'w%s'%(wr[1:]) if wr is not None else wr
	with (open(path,wr) if isinstance(path,str) else path) as obj:
		dump_json(tmp,obj,wr=wr,ext=ext,options=options['dump'],transform=transform,execute=execute,verbose=verbose,**kwargs)

	return	

def _merge_json(data,path,wr='a',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Merge object
	Args:
		data (object): Object
		path (str,object): Path or file object
		wr (str): Write mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments
	'''	
	
	wr = 'r'
	length = len(path)
	with (open(data,wr) if isinstance(data,str) else data) as obj:
		obj = load_json(obj,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
		for index,key in enumerate(obj):
			name = str(length+index) if transform else key
			path[name] = obj[key]
	
	return	

def load_hdf5(path,wr='r',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Load object
	Args:
		path (str,object): Path or file object
		wr (str): Read mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform		
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object): Object
	'''	

	with (h5py.File(path,wr) if isinstance(path,str) else path) as obj:
		data = _load_hdf5(obj,wr=wr,ext=ext,**kwargs)
	
	return data


def _load_hdf5(path,wr='r',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Load object
	Args:
		path (str,object): Path or file object
		wr (str): Read mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object): Object
	'''	

	data = {}
	
	if isinstance(path, h5py._hl.group.Group):
		names = path
		for name in names:
			key = name
			if isinstance(path[name], h5py._hl.group.Group):	
				data[key] = _load_hdf5(path[name],wr=wr,ext=ext,**kwargs)
			else:
				data[key] = path[name][...]
				if data[key].dtype.kind in ['S','O']:
					data[key] = data[key].astype(str)
				
		names = path.attrs
		for name in names:
			key = name
			data[key] = path.attrs[name]

	else:
		data = path.value
	
	return data

def dump_hdf5(data,path,wr='w',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Dump object
	Args:
		data (object): Object
		path (str,object): Path or file object
		wr (str): Write mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments
	'''	

	with (h5py.File(path,wr) if isinstance(path,str) else path) as obj:
		_dump_hdf5(data,obj,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)

	return

def _dump_hdf5(data,path,wr='w',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Dump object
	Args:
		data (object): Object
		path (str,object): Path or file object
		wr (str): Write mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments
	'''		
	if isinstance(data,dict):
		names = data
		for name in names:
			key = name
			if isinstance(data[name],dict):
				if key not in path:
					path.create_group(key)
				_dump_hdf5(data[name],path[key],wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
			elif isinstance(data[name],scalars):
				if key in path.attrs:
					del path.attrs[key]
				try:
					path.attrs[key] = data[name]
				except TypeError:
					pass
			else:
				if key in path:
					del path[key]
				try:
					path[key] = data[name]
				except:
					path[key] = np.array(data[name],dtype='S')
	else:
		path = data

	return


def merge_hdf5(data,path,wr='a',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Merge object
	Args:
		data (object): Object
		path (str,object): Path or file object
		wr (str): Write mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments
	'''	

	with (h5py.File(path,wr) if isinstance(path,str) else path) as obj:
		_merge_hdf5(data,obj,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)

	return


def _merge_hdf5(data,path,wr='a',ext=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Merge object
	Args:
		data (object): Object
		path (str,object): Path or file object
		wr (str): Write mode
		ext (str): Extension
		options (dict): Options
		transform (bool): Transform
		execute (bool): Execute
		verbose (bool,int): Verbosity		
		kwargs (dict): Additional loading keyword arguments
	'''	
	
	wr = 'r'
	length = len(path)
	with (h5py.File(data,wr) if isinstance(data,str) else data) as obj:
		for index,key in enumerate(obj):
			name = str(length+index) if transform else key
			if name in path:
				del path[name]
			obj.copy(key,path,name)
	
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
				logger.log(debug,'Exception: Cannot pickle (key,value) %r, %r'%(k,obj[k]))
				obj.pop(k);
				pickleables[k] = True		
		return all([pickleables[k] for k in pickleables])

	ispickleable = False
	if path is None:
		path  = '__tmp__.__tmp__.%d'%(np.random.randint(1,int(1e8)))
	with open(path,'wb') as file:
		try:
			pickle.dump(obj,file)
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
	return



def load(path,wr='r',default=None,delimiter=delimiter,chunk=None,wrapper=None,func=None,lock=None,backup=None,timeout=None,cleanup=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Load objects from path
	Args:
		path (str,iterable,dict[str,str]): Path
		wr (str): Read mode
		default (object): Default return object if load fails
		delimiter (str): Delimiter to separate file name from extension		
		chunk (int): Size of chunks of paths
		wrapper (str,callable,iterable[str,callable]): Process data, either string in ['df','np','array','dict','merge','pd'] or callable with signature wrapper(data)
		func (callable): Function for data
		lock (bool,str): Lock file of loading
		backup (bool,str): Backup file of loading
		timeout (int): Timeout of loading
		cleanup (bool): Cleanup of loading
		options (dict): Options of loading
		transform (bool): Transform of loading		
		execute (bool): Execute data load of loading
		verbose (bool,int): Verbose logging of loading
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object,iterable[object],dict[str,object]): Object
	'''

	if chunk:
		return (
			load(path,wr=wr,default=default,delimiter=delimiter,chunk=None,wrapper=wrapper,func=func,lock=lock,backup=backup,timeout=timeout,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs) 
			for path in slicer(sorted(glob(path),key=memory),chunk)
			)

	exts = ['npy','npz','csv','txt','sh','pickle','pkl','json','hdf5','h5','ckpt']
	wrs = [wr,'r','rb'] if not lock else ['r','rb']
	wrapper = wrapper if isinstance(wrapper,iterables) else [wrapper]

	verbose = verbose if verbose is not None else False

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

	funcs = {name:func if callable(func) else func.get(paths[name]) if isinstance(func,dict) else None for name in paths}

	data = {}

	for name in paths:

		path = paths[name]

		func = funcs[name]

		datum = default

		if not isinstance(path,str):
			data[name] = datum
			continue
	
		path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
		ext = split(path,ext=True,delimiter=delimiter)

		with Lock(lock=lock,path=path,timeout=timeout):
			
			with Backup(backup=backup,path=path):

				for wr in wrs:
					try:
						datum = _load(path,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
						break
					except (FileNotFoundError,AttributeError,TypeError,UnicodeDecodeError,ValueError,OSError,ModuleNotFoundError,ImportError,OverflowError) as exception:			
						logger.log(debug,'Exception:\n%r\n%r'%(exception,traceback.format_exc()))
						try:
							with open(path,wr) as obj:
								datum = _load(obj,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
								break
						except (FileNotFoundError,AttributeError,TypeError,UnicodeDecodeError,ValueError,OSError,ModuleNotFoundError,ImportError,OverflowError) as exception:
							logger.log(debug,'Exception:\n%r\n%r'%(exception,traceback.format_exc()))
							pass

				if callable(func):
					try:
						datum = func(datum)
					except Exception as exception:
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
					if isinstances(wrapper,dict):
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
				def iterable(data):
					return not isinstance(data,scalars) and getattr(data,'size',len(data)) > 1 and data.ndim>1 and any(isinstance(i,arrays) for i in data)
				def scalar(data):
					return isinstance(data,scalars) or getattr(data,'size',len(data)) <= 1				
				def function(path,data):
					for attr in data:
						if iterable(data[attr]):
							data[attr] = [tuple(i) for i in data[attr]]
					# length = max([len(data[attr]) if not scalar(data[attr]) else 1 for attr in data],default=0)
					# data['__path__'] = [path]*length
					return data
				try:
					data = pd.concat((pd.DataFrame(function(path,obj)) for path in data if data[path] for obj in ([data[path]] if any(not isinstance(data[path][attr],dict) for attr in data[path]) else (data[path][attr] for attr in data[path]))),**options) #.convert_dtypes()
				except Exception as exception:
					data = default
				return data
		elif wrapper in ['pd']:
			def wrapper(data):
				options = {**{'ignore_index':True},**{kwarg: kwargs[kwarg] for kwarg in kwargs if kwarg in ['ignore_index']}}
				def function(path,data):
					return data
				if len(data)>1:
					try:
						data = pd.concat((function(path,data[path]) for path in data if data[path] is not None),**options)
					except Exception as exception:
						data = default
				else:
					try:
						for path in list(data):
							data = function(path,data[path])
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
		else:
			def wrapper(data):
				return data

		wrappers.append(wrapper)

	for wrapper in wrappers:
		data = wrapper(data)

	if isinstance(args['path'],str) and (any(((i in [None]) or (isinstances(i,dict,reverse=True))) for i in args['wrapper'])):
		name = list(data)[-1]
		data = data[name]
	elif not isinstance(args['path'],dict) and (any(((i in [None])) for i in args['wrapper'])):
		data = [data[name] for name in data]
	else:
		pass

	return data



def _load(path,wr,ext,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Load object
	Args:
		path (str,object): Path or file object
		wr (str): Read mode
		ext (str): Extension
		options (dict): Options of loading
		transform (bool): Transform		
		execute (bool): Execute
		verbose (bool,int): Verbosity
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object): Object
	'''	
	
	verbose = verbose if verbose is not None else False

	wrappers = kwargs.pop('wrapper',None)

	exts = ['npy','npz','csv','txt','sh','pickle','pkl','json','hdf5','h5','ckpt']
	try:
		assert ext in exts, "Load extension %s Not Implemented"%(ext)
	except Exception as exception:
		path,module = delimiter.join(path.split(delimiter)[:-1]),path.split(delimiter)[-1]

		obj = os.path.basename(path).strip(delimiter)
		data = getattr(importlib.import_module(obj),module)

		try:
			obj = os.path.basename(path).strip(delimiter)
			data = getattr(importlib.import_module(obj),module)
		except (SyntaxError,) as exception:
			logger.log(debug,'Exception:\n%r\n%r'%(exception,traceback.format_exc()))
			exception = SyntaxError
			raise exception
		except Exception as exception:
			obj = path
			spec = importlib.util.spec_from_file_location(module,obj)
			data = importlib.util.module_from_spec(spec)
			sys.modules[module] = data
			spec.loader.exec_module(data)
			data = getattr(data,module)

	if ext in ['npy']:
		options = {'allow_pickle':True,**kwargs}
		data = np.load(path,**options)
	elif ext in ['npz']:
		options = {'allow_pickle':True,**kwargs}
		data = np.load(path,**options)
	elif ext in ['csv']:
		options = kwargs
		data = getattr(pd,'read_%s'%ext)(path,**options)
	elif ext in ['txt','sh']:
		options = {}
		data = path.readlines(**options)
	elif ext in ['pickle','pkl']:
		# TODO: Load specific types as wrapped types (i.e) onp.array -> np.array for JAX)
		options = kwargs
		data = pickle.load(path,**options)
	elif ext in ['json']:
		options = {'cls':decode_json,'object_hook':serialize_json,**kwargs}
		data = load_json(path,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
	elif ext in ['hdf5','h5','ckpt']:
		for wrapper in list(wrappers):
			if wrapper in ['pd']:
				try:
					options = {**{'key':kwargs.get('key','data')},**(options if isinstance(options,dict) else {})}
					ext = 'hdf'
					data = getattr(pd,'read_%s'%ext)(path,**options)
					break
				except Exception as exception:
					data = load_hdf5(path,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
					wrappers.append('df')
					break
			else:
				data = load_hdf5(path,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)

	return data



def dump(data,path,wr='w',delimiter=delimiter,chunk=None,wrapper=None,func=None,lock=None,backup=None,timeout=None,cleanup=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Dump objects to path
	Args:
		data (object): Object
		path (str,iterable[str],dict[str,str]): Path
		wr (str): Write mode
		delimiter (str): Delimiter to separate file name from extension		
		chunk (int): Size of chunks of paths		
		wrapper (str,callable,iterable[str,callable]): Process data, either string in ['df','np','array','dict','merge','pd'] or callable with signature wrapper(data)
		func (callable): Function for data		
		lock (bool,str): Lock file of dumping
		backup (bool,str): Backup file of dumping
		timeout (int): Timeout of dumping
		cleanup (bool): Cleanup of dumping
		options (dict): Options of dumping
		transform (bool): Transform of dumping		
		execute (bool): Execute data dump of dumping
		verbose (bool,int): Verbose logging of dumping
		kwargs (dict): Additional dumping keyword arguments
	'''
	wrs = [wr,'w','wb'] if not lock else ['a','ab']
	wrapper = wrapper if isinstance(wrapper,iterables) else [wrapper]

	verbose = verbose if verbose is not None else False

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
		elif wrapper in ['pd']:
			def wrapper(data):
				return data
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

	funcs = {name:func if callable(func) else func.get(paths[name]) if isinstance(func,dict) else None for name in paths}

	for name in paths:
		
		path = paths[name]
		
		func = funcs[name]

		path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
		ext = split(path,ext=True,delimiter=delimiter)
		mkdir(path)

		if callable(func):
			try:
				data = func(data)
			except Exception as exception:
				pass

		for wrapper in wrappers:
			data = wrapper(data)

		with Lock(lock=lock,path=path,timeout=timeout):

			with Backup(backup=backup,path=path):

				for wr in wrs:	
					try:
						_dump(data,path,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
						break
					except (ValueError,AttributeError,TypeError,OSError,ModuleNotFoundError,ImportError,OverflowError) as exception:
						logger.log(debug,'Exception:\n%r\n%r'%(exception,traceback.format_exc()))
						try:
							with open(path,wr) as obj:
								_dump(data,obj,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
							break
						except (ValueError,AttributeError,TypeError,OSError,ModuleNotFoundError,ImportError,OverflowError) as exception:
							logger.log(debug,'Exception:\n%r\n%r'%(exception,traceback.format_exc()))
							pass
		
		logger.log(info*verbose,'Dump : %s'%(relpath(paths[name])))

	return



# Dump data - General file export
def _dump(data,path,wr,ext,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Dump object
	Args:
		data (object): Object
		path (str,object): Path or file object
		wr (str): Write mode
		options (dict): Options
		transform (bool): Transform		
		execute (bool): Execute
		verbose (bool,int): Verbosity			
		kwargs (dict): Additional dumping keyword arguments
	'''	

	wrappers = kwargs.pop('wrapper',None)

	exts = ['npy','npz','csv','txt','sh','pickle','pkl','json','tex','hdf5','h5','ckpt','pdf']
	assert ext in exts, "Dump extension %s Not Implemented"%(ext)

	if ext in ['npy']:
		options = {'allow_pickle':True,**kwargs}
		np.save(path,data,**options)
	if ext in ['npz']:
		options = {}
		if isinstance(data,dict):
			np.savez(path,**data,**options)
		elif isinstance(data,(tuple)):
			np.savez(path,*data,**options)
		else:
			np.savez(path,data,**options)
	elif ext in ['csv']:
		options = {'index':False,**kwargs}
		getattr(data,'to_%s'%ext)(path,**options)
	elif ext in ['txt','sh']:
		options = {}
		path.dumplines(data,**options)
	elif ext in ['pickle','pkl']:	
		options = {**dict(protocol=pickle.HIGHEST_PROTOCOL),**kwargs}	
		pickleable(data,callables=kwargs.pop('callables',True))
		pickle.dump(data,path,**options)
	elif ext in ['json']:
		options = {'cls':encode_json,'ensure_ascii':False,'indent':4,**kwargs}
		dump_json(data,path,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
	elif ext in ['tex']:
		options = kwargs
		path.write(data,**options)
	elif ext in ['hdf5','h5','ckpt']:
		for wrapper in wrappers:
			if wrapper in ['pd']:
				options = {'key':kwargs.get('key','data'),'mode':'w'}
				ext = 'hdf'
				getattr(data,'to_%s'%ext)(path,**options)
				break
			else:
				dump_hdf5(data,path,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
	elif ext in ['pdf']:
		options = kwargs
		data.savefig(path,**options)

	return


def merge(data,path,wr='a',delimiter=delimiter,chunk=None,wrapper=None,func=None,lock=None,backup=None,timeout=None,cleanup=None,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Merge objects to path
	Args:
		data (str,iterable[str],dict[str,str]): Paths to merge object
		path (str): Path to merge object
		wr (str): Write mode
		delimiter (str): Delimiter to separate file name from extension		
		chunk (int): Size of chunks of paths		
		wrapper (str,callable,iterable[str,callable]): Process data, either string in ['df','np','array','dict','merge','pd'] or callable with signature wrapper(data)
		func (callable): Function for data
		lock (bool,str): Lock file of merging
		backup (bool,str): Backup file of merging
		timeout (int): Timeout of merging
		cleanup (bool): Cleanup of merging		
		options (dict): Options of merging		
		transform (bool): Transform of merging
		execute (bool): Execute data load of merging
		verbose (bool,int): Verbose logging of merging
		kwargs (dict): Additional appending keyword arguments
	'''

	exts = ['npy','npz','csv','txt','sh','pickle','pkl','json','hdf5','h5','ckpt']
	wrs = [wr,'a','ab'] if not lock else ['a','ab']
	wrapper = wrapper if isinstance(wrapper,iterables) else [wrapper]

	verbose = verbose if verbose is not None else False

	if data is None or not isinstance(data,(str,iterables)):
		return

	if isinstance(data,str):
		data = [data]
	else:
		data = data
	
	if not isinstance(data,dict):
		data = {name: name for name in data}
	else:
		data = {name:data[name] for name in data}

	data = {(delim*3).join([name,str(path)]): path
		for name in data
		for path in natsorted(glob(data[name],default=(None if split(data[name],ext=True) in exts else data[name])))
		}

	for name in data:

		name = data[name]

		if not isinstance(name,str):
			continue
	
		name = os.path.abspath(os.path.expandvars(os.path.expanduser(name)))
		ext = split(name,ext=True,delimiter=delimiter)

		with Lock(lock=lock,path=path,timeout=timeout):
			
			with Backup(backup=backup,path=path):

				for wr in wrs:

					wr = wr if exists(path) else 'w%s'%(wr[1:]) if wr is not None else wr

					try:
						_merge(name,path,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
						break
					except (FileNotFoundError,AttributeError,TypeError,UnicodeDecodeError,ValueError,OSError,ModuleNotFoundError,ImportError,OverflowError) as exception:			
						logger.log(debug,'Exception:\n%r\n%r'%(exception,traceback.format_exc()))
						try:
							with open(path,wr) as obj:
								_merge(name,obj,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
								break
						except (FileNotFoundError,AttributeError,TypeError,UnicodeDecodeError,ValueError,OSError,ModuleNotFoundError,ImportError,OverflowError) as exception:
							logger.log(debug,'Exception:\n%r\n%r'%(exception,traceback.format_exc()))
							pass

		if cleanup:
			rm(name)

		logger.log(info*verbose,'Merge : %s -> %s'%(relpath(name),relpath(path)))

	return


# Merge data - General file merge
def _merge(data,path,wr,ext,options=None,transform=None,execute=None,verbose=None,**kwargs):
	'''
	Merge object
	Args:
		data (object): Object
		path (str,object): Path or file object
		wr (str): Write mode
		options (dict): Options
		transform (bool): Transform		
		execute (bool): Execute
		verbose (bool,int): Verbosity	
		kwargs (dict): Additional dumping keyword arguments
	'''	

	exts = ['json','hdf5','h5']
	assert ext in exts, "Merge extension %s Not Implemented"%(ext)

	if ext in ['json']:
		options = dict(
			load={'cls':decode_json,'object_hook':serialize_json,**kwargs},
			dump={'cls':encode_json,'ensure_ascii':False,'indent':4,**kwargs}
			)
		merge_json(data,path,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)
	elif ext in ['hdf5','h5']:
		merge_hdf5(data,path,wr=wr,ext=ext,options=options,transform=transform,execute=execute,verbose=verbose,**kwargs)

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
	def __exit__(self,etype,value,traceback):
		try:
			return self.cls.__exit__(etype,value,traceback)
		except:
			return