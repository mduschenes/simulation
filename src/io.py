#!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,inspect,traceback
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
debug = 100

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,is_array,is_ndarray
from src.utils import returnargs
from src.utils import scalars,nan

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
		self.cwd = os.getcwd()

		try:
			os.chdir(self.path)
		except:
			pass
		return
	def __exit__(self,etype, value, traceback):
		os.chdir(self.cwd)
		return


def exists(path):
	'''
	Check if path exists
	Make path
	Args:
		path (str): path
	Returns:
		exists (bool): Path exists
	'''

	exists = os.path.exists(path)

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
		'npy','pickle','pkl',
		'json',
		'hdf5','h5',
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
	if abspath:
		directory = os.path.abspath(directory)

	return directory

	

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

def rmdir(path):
	'''
	Remove path
	Args:
		path (str): path
	'''

	try:
		os.remove(path)
	except:
		os.rmdir(path)

	return


def copy(source,destination):
	'''
	Copy paths
	Args:
		source (str): Source path
		destination (str): Destination path
	'''
	mkdir(destination)
	shutil.copy2(source,destination)

	return

def split(path,directory=False,file=False,ext=False,directory_file=False,file_ext=False,abspath=None,delimiter='.'):
	'''
	Split path into directory,file,ext
	Args:
		path (str): Path to split
		directory (bool,int,iterable[int]): Return split directory name, or number of intermediate directories after root / before directory containing folder
		file (bool): Return split file name
		ext (bool): Return split extension name
		directory_file (bool): Return split and joined directory and file name
		file_ext (bool): Return split and joined file and extension name
		abspath (bool): Return absolute directory		
		delimiter (str): Delimiter to separate file name from extension
	Returns:
		paths (iterable): Split path,directory,file,ext depending on booleans
	'''	

	returns = {'directory':directory,'file':file or directory_file or file_ext,'ext':ext}
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
		paths['directory'] = os.path.abspath(paths['directory'])

	paths['file'],paths['ext'] = os.path.splitext(path)
	if paths['ext'].startswith(delimiter):
		paths['ext'] = delimiter.join(paths['ext'].split(delimiter)[1:])
	if not directory_file:
		if dirname(path) == paths['file']:
			paths['file'] = None
		else:
			paths['file'] = os.path.basename(paths['file'])
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
	paths = [path for path in paths if path not in ['',None]]
	if len(paths)>0:
		path = os.path.join(*paths)
	else:
		path = ''
	if path is not None and ext is not None and not path.endswith('%s%s'%(delimiter,ext)):
		paths = [path,ext]
		paths = [path for path in paths if path not in ['',None]]		
		path = delimiter.join(paths)

	if path is not None and root is not None:
		if not dirname(path,abspath=True).startswith(dirname(root,abspath=True)):
			paths = [root,path]
			paths = [path for path in paths if path not in ['',None]]
			path = os.path.join(*paths)
	if path is not None and abspath:
		path = os.path.abspath(path)
	return path


def glob(path,include=None,recursive=False,**kwargs):
	'''
	Expand path
	Args:
		path (str): Path to expand
		include (str): Type of paths to expand, allowed ['directory','file']
		recursive (bool,str): Recursively find all included paths below path, or expander strings ['*','**']
		kwargs (dict): Additional glob keyword arguments
	Returns:
		paths (list[str]): Expanded, absolute path
	'''

	if include is None:
		include = lambda path:True
	elif include in ['file']:
		include = os.path.isfile
	elif include in ['directory']:
		include = os.path.isdir
	else:
		include = lambda path:True

	if not isinstance(recursive,str):
		if recursive:
			recursive = '**'
		else:
			recursive = None

	path = join(path,recursive)

	paths = globber.glob(os.path.abspath(os.path.expanduser(path)),recursive=True,**kwargs)

	paths = list(sorted(filter(include,paths)))

	return paths

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

# 
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
	if callable(obj) or isinstance(obj,(slice,range)):
		if callable(obj) and not inspect.isclass(obj):            
			obj = funcclass(obj)
		obj = jsonpickle.encode(obj)
	elif isinstance(obj,np.ndarray):
		obj = obj.tolist()
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
	def convert(name,conversion=None,**kwargs):
		if conversion is None:
			conversion = lambda name: str(name)
		key = conversion(name)
		return key

	null = {'None':'None'}

	data = {}
	
	if isinstance(obj, h5py._hl.group.Group):
		names = natsorted(obj)
		for name in names:
			key = convert(name,**kwargs)
			if isinstance(obj[name], h5py._hl.group.Group):	
				data[key] = _load_hdf5(obj[name],wr=wr,ext=ext,**kwargs)
			else:
				data[key] = obj[name][...]
				continue		
				# assert isinstance(obj[name],h5py._hl.dataset.Dataset)
				if any(attr in name for attr in ('.real','.imag')):
					name = name.replace('.real','').replace('.imag','')
					try:
						name_real = "%s.%s"%(name,'real')
						data_real = obj[name_real][...]

						name_imag = "%s.%s"%(name,'imag')
						data_imag = obj[name_imag][...]

						data[key] = data_real + 1j*data_imag
					except:
						data[key] = obj[name][...]
				else:
					data[key] = obj[name][...]

		names = list(set((name for name in obj.attrs)))
		for name in names:
			key = name #convert(name,**kwargs)
			data[key] = obj.attrs[name]
			if obj.attrs[name] in null:
				data[key] = null[obj.attrs[name]]
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

	def convert(name,conversion=None,**kwargs):
		if conversion is None:
			conversion = lambda name: str(name)
		key = conversion(name)
		return key

	null = {None: 'None'}

	if isinstance(obj,dict):
		names = obj
		for name in names:
			key = convert(name,**kwargs)
			if isinstance(obj[name],dict):
				path.create_group(key)
				_dump_hdf5(obj[name],path[key],wr=wr,ext=ext,**kwargs)
			elif isinstance(obj[name],scalars):
				try:
					path.attrs[key] = obj[name]
				except TypeError:
					if obj[name] in null:
						path.attrs[key] = null[obj[name]]
					else:
						continue
			else:
				path[key] = obj[name]
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
				logger.log(verbose,'Cannot pickle (key,value) %r, %r'%(k,obj[k]))
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
		rmdir(path)
	return ispickleable


def jsonable(obj,path=None,callables=False):
	'''
	Check if object can be written to json
	Args:
		obj (object): Object to json
		path (str): Path to check if object can be written to
		callables (bool): Allow functions to be written to json
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
			json.dump(obj,fobj,**{'default':dump_json,'ensure_ascii':False,'indent':4})
			isjsonable = True
		except Exception as exception:
			pass
	if exists(path):
		rmdir(path)
	return isjsonable




def load(path,wr='r',default=None,delimiter='.',verbose=False,**kwargs):
	'''
	Load objects from path
	Args:
		path (str): Path to load object
		wr (str): Read mode
		default (object): Default return object if load fails
		delimiter (str): Delimiter to separate file name from extension		
		verbose (bool,int): Verbose logging of loading
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object): Loaded object
	'''
	wrs = [wr,'r','rb']

	if not isinstance(path,str):
		return default
	
	ext = split(path,ext=True,delimiter=delimiter)

	for wr in wrs:
		try:
			data = _load(path,wr=wr,ext=ext,**kwargs)
			return data
		except (FileNotFoundError,AttributeError,TypeError,UnicodeDecodeError,ValueError) as exception:			
			logger.log(debug,'Path: %r\n%r'%(exception,traceback.format_exc()))
			try:
				with open(path,wr) as obj:
					data = _load(obj,wr=wr,ext=ext,**kwargs)
					return data
			except (FileNotFoundError,AttributeError,TypeError,UnicodeDecodeError,ValueError) as exception:
				logger.log(debug,'Object: %r\n%r'%(exception,traceback.format_exc()))
				pass

	return default			



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
	
	exts = ['npy','csv','txt','pickle','pkl','json','hdf5','h5']

	try:
		assert ext in exts, "Cannot load extension %s"%(ext)
	except Exception as exception:
		# try:
		obj,module = '.'.join(obj.split('.')[:-1]),obj.split('.')[-1]
		data = getattr(importlib.import_module(obj),module)
		# except Exception as exception:
		# 	raise exception

	if ext in ['npy']:
		data = np.load(obj,**{**kwargs})
	elif ext in ['csv']:
		data = getattr(pd,'read_%s'%ext)(obj,**{**kwargs})
	elif ext in ['txt']:
		data = np.loadtxt(obj,**{'delimiter':',',**kwargs})
	elif ext in ['pickle','pkl']:
		# TODO: Load specific types as wrapped types (i.e) onp.array -> np.array for JAX)
		data = pickle.load(obj,**kwargs)
	elif ext in ['json']:
		data = json.load(obj,**{'object_hook':load_json,**kwargs})
	elif ext in ['hdf5','h5']:
		data = load_hdf5(obj,wr=wr,ext=ext,**kwargs)

	return data



def dump(data,path,wr='w',delimiter='.',verbose=False,**kwargs):
	'''
	Dump objects to path
	Args:
		data (object): Object to dump
		path (str): Path to dump object
		wr (str): Write mode
		delimiter (str): Delimiter to separate file name from extension		
		verbose (bool,int): Verbose logging of dumping
		kwargs (dict): Additional dumping keyword arguments
	'''

	wrs = [wr,'w','wb']

	if not isinstance(path,str):
		return

	ext = split(path,ext=True,delimiter=delimiter)

	mkdir(path)

	for wr in wrs:	
		try:
			_dump(data,path,wr=wr,ext=ext,**kwargs)
			return
		except (ValueError,AttributeError,TypeError) as exception:
			logger.log(debug,'Path: %r\n%r'%(exception,traceback.format_exc()))
			try:
				with open(path,wr) as obj:
					_dump(data,obj,wr=wr,ext=ext,**kwargs)
				return
			except (ValueError,AttributeError,TypeError) as exception:
				logger.log(debug,'Object: %r\n%r'%(exception,traceback.format_exc()))
				pass
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

	exts = ['npy','csv','txt','pickle','pkl','json','tex','hdf5','h5','pdf']
	assert ext in exts, "Cannot dump extension %s"%(ext)

	if ext in ['npy']:
		np.save(obj,data,**{**kwargs})
	elif ext in ['csv']:
		getattr(data,'to_%s'%ext)(obj,**{'index':False,**kwargs})
	elif ext in ['txt']:
		np.savetxt(obj,data,**{'delimiter':',','fmt':'%.20f',**kwargs})
	elif ext in ['pickle','pkl']:		
		pickleable(data,callables=kwargs.pop('callables',True))
		pickle.dump(data,obj,protocol=pickle.HIGHEST_PROTOCOL,**kwargs)
	elif ext in ['json']:
		# jsonable(data,callables=kwargs.pop('callables',False))	
		json.dump(data,obj,**{'default':dump_json,'ensure_ascii':False,'indent':4,**kwargs})
	elif ext in ['tex']:
		obj.write(data,**kwargs)
	elif ext in ['hdf5','h5']:
		dump_hdf5(data,obj,wr=wr,ext=ext,**kwargs)
	elif ext in ['pdf']:
		data.savefig(obj,**{**kwargs})

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

	field = "logging"
	def func(key,value):
		import logging.config
		logging.config.dictConfig({**value,**{"disable_existing_loggers":False}})
		return value
	updates[field] = func


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