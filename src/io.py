#!/usr/bin/env python

# Import python modules
import os,sys,copy,warnings,itertools,inspect
import glob,json,jsonpickle,h5py,pickle,dill
import numpy as np
import pandas as pd

from natsort import natsorted, ns,index_natsorted,order_by_index

# Logging
import logging
logger = logging.getLogger(__name__)

# Import user modules
from src.utils import returnargs

def path_split(path,directory=False,file=False,ext=False,directory_file=False,file_ext=False,delimiter='.'):
	'''
	Split path into directory,file,ext
	Args:
		path (str): Path to split
		directory (bool): Return split directory name
		file (bool): Return split file name
		ext (bool): Return split extension name
		directory_file (bool): Return split and joined directory and file name
		file_ext (bool): Return split and joined file and extension name
		delimiter (str): Delimiter to separate file name from extension
	Returns:
		paths (iterable): Split path,directory,file,ext depending on booleans
	'''	
	if not (directory or file or ext):
		return path
	returns = {'directory':directory,'file':file or directory_file or file_ext,'ext':ext}
	paths = {}
	paths['directory'] = os.path.dirname(path)
	paths['file'],paths['ext'] = os.path.splitext(path)
	if paths['ext'].startswith(delimiter):
		paths['ext'] = delimiter.join(paths['ext'].split(delimiter)[1:])
	if not directory_file:
		paths['file'] = os.path.basename(paths['file'])
	if file_ext and paths['ext'].startswith(delimiter):
		paths['file'] = delimiter.join([paths['file'],paths['ext']])
	paths = [paths[k] for k in paths if returns[k]] 
	return paths if len(paths)>1 else paths[0]

def path_join(*paths,ext=None,abspath=False,delimiter='.'):
	'''
	Join paths into path, with optional extension
	Args:
		paths (iterable[str]): Paths to join
		ext (str): Extension to add to path
		abspath (bool): Return absolute joined path
		delimiter (str): Delimiter to separate file name from extension
	Returns:
		paths (str): Joined path
	'''	

	path = os.path.join(*paths)
	if ext is not None and not path.endswith('%s%s'%(delimiter,ext)):
		path = delimiter.join([path,ext])
	if abspath:
		path = os.path.abspath(path)
	return path


def path_glob(path,**kwargs):
	'''
	Expand path
	Args:
		path (str): Path to expand
		kwargs (dict): Additional glob keyword arguments
	Returns:
		path (str): Expanded, absolute path
	'''
	return glob.glob(os.path.abspath(os.path.expanduser(path)),**kwargs)

def path_file(path,**kwargs):
	'''
	Get file name from path
	Args:
		path (str): Path
		kwargs (dict): Additional path keyword arguments
	Returns:
		file (str): Filename
	'''	
	return os.path.basename(os.path.splitext(path)[0])

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

def serialize_json(obj,key='py/object'):
	'''
	Serialize objects into json
	Args:
		obj (object): Object to serialize
		key (str): Key to serialize on
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

def deserialize_json(obj,key='py/object'):
	'''
	De-serialize objects into json
	Args:
		obj (object): Object to de-serialize
		key (str): Key to de-serialize on
	Returns:
		obj (object): De-serialized object
	'''	
	if isinstance(obj,dict) and key in obj:
		obj = pickle.loads(str(obj[key]))
	return obj


def pickleable(obj,path=None,callables=True):
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
		except Exception as e:
			pass
	if os.path.exists(path):
		os.remove(path)
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
			json.dump(obj,fobj,**{'default':serialize_json,'ensure_ascii':False,'indent':4})
			isjsonable = True
		except Exception as e:
			pass
	if os.path.exists(path):
		os.remove(path)
	return isjsonable



def load(path,wr='r',default=None,verbose=False,**kwargs):
	'''
	Load objects from path
	Args:
		path (str): Path to load object
		wr (str): Read mode
		default (object): Default return object if load fails
		verbose (bool,int): Verbose logging of loading
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object): Loaded object
	'''

	loaders = {ext: lambda obj,wr,ext=ext,**kwargs: _load(obj,wr,ext,**kwargs)
				for ext in ['npy','csv','txt','pickle','pkl','json','hdf5']}

	if not isinstance(path,str):
		return default
	
	if path is None:
		return default

	ext = path.split('.')[-1]
	if ('.' in path) and (ext in loaders):
		paths = {ext: path}
	else:
		paths = {e: '%s.%s'%(path,e) for e in loaders}

	loaders = {paths[e]: loaders[e] for e in paths}

	for path in loaders:
		loader = loaders[path]
		for _wr in [wr,'r','rb']:
			try:
				data = loader(path,_wr,**kwargs)
				logger.log(verbose,'Loading path %s'%(path))
				return data
			except Exception as e:
				try:
					with open(path,_wr) as obj:
						data = loader(obj,_wr,**kwargs)
						logger.log(verbose,'Loading obj %s'%(path))
						return data
				except Exception as ee:
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
	
	if ext in ['npy']:
		data = np.load(obj,**{**kwargs})
	elif ext in ['csv']:
		data = getattr(pd,'read_%s'%ext)(obj,**{**kwargs})
	elif ext in ['txt']:
		data = np.loadtxt(obj,**{'delimiter':',',**kwargs})
	elif ext in ['pickle','pkl']:
		data = pickle.load(obj,**kwargs)
	elif ext in ['json']:
		data = json.load(obj,**{'object_hook':deserialize_json,**kwargs})
	elif ext in ['hdf5']:

		file = h5py.File(obj,wr)

		data = {}

		for group in natsorted(file):
			data[group] = {}

			names = list(set((name.replace('.real','').replace('.imag','') for name in file[group])))
			for name in names:

				try:
					name_real = "%s.%s"%(name,'real')
					data_real = file[group][name_real][...]

					name_imag = "%s.%s"%(name,'imag')
					data_imag = file[group][name_imag][...]

					data[group][name] = data_real + 1j*data_imag
				except:
					data[group][name] = file[group][name][...]

			names = list(set((name for name in file[group].attrs)))
			for name in names:
				data[group][name] = file[group].attrs[name]

		file.close()

	return data



def dump(data,path,wr='w',verbose=False,**kwargs):
	'''
	Dump objects to path
	Args:
		data (object): Object to dump
		path (str): Path to dump object
		wr (str): Write mode
		verbose (bool,int): Verbose logging of dumping
		kwargs (dict): Additional dumping keyword arguments
	'''
	dumpers = {ext: lambda data,obj,wr,ext=ext,**kwargs: _dump(data,obj,wr,ext,**kwargs)
				for ext in ['npy','csv','txt','pickle','pkl','json','tex','hdf5','pdf']}

	if not isinstance(path,str):
		return
	
	if path is None:
		return

	ext = path.split('.')[-1]
	if ('.' in path) and (ext in dumpers):
		paths = {ext: path}
	else:
		paths = {e: '%s.%s'%(path,e) for e in dumpers}

	dumpers = {paths[e]: dumpers[e] for e in paths}

	for path in dumpers:
		dirname = os.path.abspath(os.path.dirname(path))
		if not os.path.exists(dirname):
			os.makedirs(dirname)

	for path in dumpers:
		dumper = dumpers[path]

		for _wr in [wr,'w','wb']:		
			with open(path,_wr) as obj:
				try:
					dumper(data,path,_wr,**kwargs)
					logger.log(verbose,'Dumping path %s'%(path))
					return
				except Exception as e:
					try:
						dumper(data,obj,_wr,**kwargs)
						logger.log(verbose,'Dumping obj %s'%(path))
						return
					except Exception as ee:
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
		jsonable(data,callables=kwargs.pop('callables',False))	
		json.dump(data,obj,**{'default':serialize_json,'ensure_ascii':False,'indent':4,**kwargs})
	elif ext in ['tex']:
		obj.write(data,**kwargs)
	elif ext in ['hdf5']:
		pass
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