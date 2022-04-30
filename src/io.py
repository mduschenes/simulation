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
from .utils import returnargs


# Split path into directory,file,ext
def path_split(path,directory=False,file=False,ext=False,directory_file=False,file_ext=False,ext_delimeter='.'):
	if not (directory or file or ext):
		return path
	returns = {'directory':directory,'file':file or directory_file or file_ext,'ext':ext}
	paths = {}
	paths['directory'] = os.path.dirname(path)
	paths['file'],paths['ext'] = os.path.splitext(path)
	if paths['ext'].startswith(ext_delimeter):
		paths['ext'] = ext_delimeter.join(paths['ext'].split(ext_delimeter)[1:])
	if not directory_file:
		paths['file'] = os.path.basename(paths['file'])
	if file_ext and paths['ext'].startswith(ext_delimeter):
		paths['file'] = ext_delimeter.join([paths['file'],paths['ext']])
	paths = [paths[k] for k in paths if returns[k]] 
	return paths if len(paths)>1 else paths[0]

# Join path by directories, with optional extension
def path_join(*paths,ext=None,abspath=False,ext_delimeter='.'):
	path = os.path.join(*paths)
	if ext is not None and not path.endswith('%s%s'%(ext_delimeter,ext)):
		path = ext_delimeter.join([path,ext])
	if abspath:
		path = os.path.abspath(path)
	return path


# glob path
def path_glob(path,**kwargs):
	return glob.glob(os.path.abspath(os.path.expanduser(path)),**kwargs)

# filename
def path_file(path,**kwargs):
	return os.path.basename(os.path.splitext(path)[0])

# basename
def basename(path,**kwargs):
	return os.path.basename(os.path.splitext(path)[0])

# Class wrapper for functions
class funcclass(object):
	def __init__(self,func=lambda x:x):
		self.func = func
	def __call__(self,*args,**kwargs):
		return self.func(*args,**kwargs)

# Serialize object to JSON
def serialize_json(obj,key='py/object'):
	if callable(obj) or isinstance(obj,(slice,range)):
		if callable(obj) and not inspect.isclass(obj):            
			obj = funcclass(obj)
		obj = jsonpickle.encode(obj)
	elif isinstance(obj,np.ndarray):
		obj = obj.tolist()
	return obj

# Deserialize object from JSON
def deserialize_json(obj,key='py/object'):
	if isinstance(obj,dict) and key in obj:
		obj = pickle.loads(str(obj[key]))
	return obj

# Load data - General file import
def load(path,wr='r',default=None,verbose=False,**kwargs):
	loaders = {**{ext: (lambda obj,ext=ext,**kwargs:getattr(pd,'read_%s'%ext)(obj,**kwargs)) for ext in ['csv']},
			   **{ext: (lambda obj,ext=ext,**kwargs:np.loadtxt(obj,**{'delimiter':',',**kwargs})) for ext in ['txt']},
			   **{ext: (lambda obj,ext=ext,**kwargs:getattr(pd,'read_%s'%ext)(obj,**kwargs) if wr=='r' else (pickle.load(obj,**kwargs))) for ext in ['pickle']},
			   **{ext: (lambda obj,ext=ext,**kwargs: json.load(obj,**{'object_hook':deserialize_json,**kwargs})) for ext in ['json']},
			  }
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
		for wr in [wr,'r','rb']:
			try:
				data = loader(path,**kwargs)
				logger.log(verbose,'Loading path %s'%(path))
				return data
			except Exception as e:
				print(e)
				try:
					with open(path,wr) as obj:
						data = loader(obj,**kwargs)
						logger.log(verbose,'Loading obj %s'%(path))
						return data
				except Exception as e:
					print(e)
					pass

	return default			
		
# Dump data - General file save/export
def dump(data,path,wr='w',verbose=False,**kwargs):

	dumpers = {**{ext: (lambda data,obj,ext=ext,**kwargs:getattr(data,'to_%s'%ext)(obj,**{'index':False,**kwargs})) for ext in ['csv']},
			   **{ext: (lambda data,obj,ext=ext,**kwargs:np.savetxt(obj,data,**{'delimiter':',','fmt':'%.20f',**kwargs})) for ext in ['txt']},
			   **{ext: (lambda data,obj,ext=ext,**kwargs:getattr(data,'to_%s'%ext)(obj,**kwargs) if isinstance(data,pd.DataFrame) else pickle.dump(data,obj,protocol=pickle.HIGHEST_PROTOCOL,**kwargs)) for ext in ['pickle']},
			   **{ext: (lambda data,obj,ext=ext,**kwargs: json.dump(data,obj,**{'default':serialize_json,'ensure_ascii':False,'indent':4,**kwargs})) for ext in ['json']},
			   **{ext: (lambda data,obj,ext=ext,**kwargs: obj.write(data,**kwargs)) for ext in ['tex']},
			  }

	if path is None:
		return
	ext = path.split('.')[-1]
	if ('.' in path) and (ext in dumpers):
		paths = {ext: path}
	else:
		paths = {e: '%s.%s'%(path,e) for e in dumpers}
		return
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
					dumper(data,path,**kwargs)
					logger.log(verbose,'Dumping path %s'%(path))
					return
				except Exception as e:
					try:
						dumper(data,obj,**kwargs)
						logger.log(verbose,'Dumping obj %s'%(path))
						return
					except Exception as e:
						try:
							dumper(data,path,**kwargs)
						except:	
							try:					
								dumper(data,obj,**kwargs)
							except:
								pass

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