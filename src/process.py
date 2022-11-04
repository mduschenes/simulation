#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,copy
import numpy as np
import scipy as sp
import scipy.special
import pandas as pd
from natsort import natsorted,realsorted
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Logging
import logging
logger = logging.getLogger(__name__)


# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.utils import array,product,expand_dims,to_eval,to_repr,is_iterable,is_number,to_number,to_key_value
from src.utils import asarray,asscalar
from src.utils import argmax,difference,is_nan
from src.utils import e,pi,nan,scalars,nulls,scinotation
from src.dictionary import leaves,branches
from src.io import setup,load,dump,join,split,glob
from src.plot import plot

# Texify strings
def Texify(string,texify={},usetex=True):
	strings = {
		**texify,
	}
	if not isinstance(string,str) and string is not None:
		string = str(string)

	try:
		default = '\n'.join(['$%s$'%(strings.get(substring,substring).replace('$','')) for substring in string.split('\n')])

		if string in strings:
			string = '$%s$'%(strings.get(string).replace('$',''))
		else:
			string = default

		if not usetex or len(string) == 0:
			string = string.replace('$','')
	except AttributeError:
		string = None

	return string


# Transform of data
def transformation(transform=None):
	if transform is None:
		transform = lambda data:data
		invtransform = lambda data:data
	elif not isinstance(transform,str):
		transform,invtransform = transform
	elif callable(transform):
		transform,invtransform = transform, lambda data:data
	elif transform in ['linear']:
		transform = lambda data:data
		invtransform = lambda data:data		
	elif transform in ['log']:		
		transform = lambda data: np.log(data)
		invtransform = lambda data:np.exp(data)
	else:
		transform = lambda data:data		
		invtransform = lambda data:data		
	return transform,invtransform

# Normalization of data
def norm(data,axis=None,ord=2):
	if axis is not None and not isinstance(axis,int):
		axis = tuple(axis)
	return np.linalg.norm(data,axis=axis,ord=ord)

# Sample average of data
def mean(data,axis=None,transform=None,dtype=None,**kwargs):
	if axis is not None and not isinstance(axis,int):
		axis = tuple(axis)
	transform,invtransform = transformation(transform)
	return invtransform(np.nanmean(transform(data),axis=axis).astype(dtype))

# Sample deviation of data
def std(data,axis=None,transform=None,dtype=None,**kwargs):
	if axis is not None and not isinstance(axis,int):
		axis = tuple(axis)
	transform,invtransform = transformation(transform)	
	n = data.shape[axis]
	return invtransform((np.nanstd(transform(data),axis=axis,ddof=n>1)).astype(dtype))


# Square root of data
def sqrt(data,axis=None,transform=None,dtype=None,**kwargs):
	if axis is not None and not isinstance(axis,int):
		axis = tuple(axis)
	return np.sqrt(data)

# Size of data
def size(data,axis=None,transform=None,dtype=None,**kwargs):
	if axis is not None and not isinstance(axis,int):
		axis = tuple(axis)
	if axis is None:
		size = data.size
	elif isinstance(axis,int):
		size = data.shape[axis]		
	else:
		size = product([data.shape[ax] for ax in axis])
	return size


# Wrapper of data
def wrapping(wrapper=None,kwarg=None,stat=None,**kwargs):
	if wrapper is None:
		wrapper = lambda data,kwargs=kwargs: data
	elif wrapper in ['mean']:
		if stat is None:
			wrapper = lambda data,kwargs=kwargs: mean(data,**kwargs)
		elif 'err' in kwarg:
			wrapper = lambda data,kwargs=kwargs: sqrt(mean(data**2,**kwargs)/size(data,**kwargs))
		else:
			wrapper = lambda data,kwargs=kwargs: mean(data,**kwargs)
	else:
		wrapper = lambda data,kwargs=kwargs: data

	return wrapper

# Fit data
def fit(x,y,_x=None,func=None,wrapper=None,coef0=None,intercept=True):

	x[np.isnan(x) | np.isinf(x)] = 0

	if wrapper is None:
		wrapper = lambda x,y,*coef: y

	if func is None:
		if intercept:
			x = np.array([x,np.ones(x.size)]).T
		else:
			x = np.array([x]).T
		if _x is None:
			_x = x
		elif intercept:
			_x = np.array([_x,np.ones(_x.size)]).T
		else:
			_x = np.array([_x]).T
		try:
			coef = np.linalg.lstsq(x,y)[0] + 0.0
			_y = _x.dot(coef)
		except:
			_y = y
			coef = np.zeros(_x.shape[1])
	else:
		if _x is None:
			_x = x
		try:
			coef = sp.optimize.curve_fit(func,x,y,p0=coef0)[0] + 0.0
			_y = func(_x,*coef)
		except:
			coef = coef0
			_y = np.zeros(_x.shape[0])

	if coef is not None:
		_y = wrapper(_x,_y,*coef)
	else:
		coef = np.zeros(3)
	return _y,coef

def check(key,value,datum,keys,values,data):
	'''
	Include key if conditions on key and value are True
	Args:
		key (str): Reference key to check
		value (object): Reference value to check, allowed strings in ['$value$,','@key@,','#index#,','%start,stop,step%'] 
						for comma-delimited: values, other key, indices, or slices values
		datum (object): Data value to check
		keys (dict[str,str]): Keys of conditions	
		values (dict[str,object,iterable[object]]): Data of keys and iterable of values to compare
		data (dict[str,dict[str:object]]): Data of datasets of keys and values to compare
	Returns:
		boolean (bool): Accept inclusion of dataset
	'''
	boolean = False
	if value is None:
		if datum is not None:
			value = [datum]
		else:
			value = []
	elif isinstance(value,str):
		if value.startswith('$') and value.endswith('$'):
			parser = lambda value: (to_number(value) if len(value)>0 else 0)
			value = value.replace('$','').split(',')
			value = [parser(v) for v in value]
			value = [v for v in value]
		elif value.startswith('@') and value.endswith('@'):
			# parser = lambda value: (str(value) if len(value)>0 else 0)
			# value = value.replace('@','').split(',')
			# value = [parser(v) for v in value]
			value = [None]#,*[u for v in value for u in values[v]]]
		elif value.startswith('#') and value.endswith('#'):
			parser = lambda value: (int(value) if len(value)>0 else 0)
			indices = value.replace('#','').split(',')
			indices = [parser(index) for index in indices]
			value = [values[key][index] for index in indices]
		elif value.startswith('%') and value.endswith('%'):
			parser = lambda value: (int(value) if len(value)>0 else None)
			slices = value.replace('%','').split(',')
			if value.count(',') == 0:
				slices = None,None,parser(slices[0])
			elif value.count(',') == 1:
				slices = parser(slices[0]),parser(slices[1]),None
			elif value.count(',') == 2:
				slices = parser(slices[0]),parser(slices[1]),parser(slices[2])
			else:
				slices = None,None,None
			slices = slice(*slices)
			value = [v for v in values[key][slices] if v is not None]
		else:
			parser = lambda value: (to_number(value))
			value = [parser(value)]
	else:
		value = [value]

	boolean = datum in value
	
	return boolean


def include(name,keys,values,sort,data):
	'''
	Include data if conditions on key and value are True
	Args:
		name (str): Dataset name
		keys (dict[str,str]): Keys of conditions	
		values (dict[str,object]): Data of keys and iterable of values to compare
		sort (dict[str,iterable[object]]): Data of keys and iterable of all values to compare
		data (dict[str,dict[str:object]]): Data of datasets of keys and values to compare
	Returns:
		boolean (bool): Accept inclusion of dataset
	'''

	def exception(name,key,keys,values,data):
		boolean = True
		for prop in keys:
			if key in keys[prop]:
				if keys[prop][key] is None:
					value = [values[key]]
				elif isinstance(keys[prop][key],str):
					if keys[prop][key].startswith('$') and keys[prop][key].endswith('$'):
						parser = lambda value: (to_number(value) if len(value)>0 else 0)
						value = keys[prop][key].replace('$','').split(',')
						value = [parser(v) for v in value]
						value = [v for v in value]
					elif keys[prop][key].startswith('@') and keys[prop][key].endswith('@'):
						parser = lambda value: (str(value) if len(value)>0 else 0)
						value = keys[prop][key].replace('@','').split(',')
						value = [parser(v) for v in value]
						value = [data[name][v] for v in value]
					elif keys[prop][key].startswith('#') and keys[prop][key].endswith('#'):
						parser = lambda value: (int(value) if len(value)>0 else 0)
						indices = keys[prop][key].replace('#','').split(',')
						indices = [parser(index) for index in indices]
						value = [sort[key][index] for index in indices]
					elif keys[prop][key].startswith('%') and keys[prop][key].endswith('%'):
						parser = lambda value: (int(value) if len(value)>0 else None)
						slices = keys[prop][key].replace('%','').split(',')
						if keys[prop][key].count(',') == 0:
							slices = None,None,parser(slices[0])
						elif keys[prop][key].count(',') == 1:
							slices = parser(slices[0]),parser(slices[1]),None
						elif keys[prop][key].count(',') == 2:
							slices = parser(slices[0]),parser(slices[1]),parser(slices[2])
						else:
							slices = None,None,None
						slices = slice(*slices)
						value = sort[key][slices]
					else:
						parser = lambda value: (to_number(value))
						value = [parser(value)]
				else:
					value = [values[key]]

				boolean = data[name][key] in value

		return boolean


	rules = lambda name,key,keys,values,data: (((data[name][key]==values[key]) or (values[key] is None)) and
												exception(name,key,keys,values,data))

	boolean = all(rules(name,key,keys,values,data) for key in values)

	return boolean

def permute(key,value,keys,values):
	'''
	Get permutations of values based on grouping by subset key of keys, constrained by the value of each key
	Args:
		key (iterable[str]): Key(s) to group with
		value (iterale[str]): Values of grouping keys to constrain grouping and permutation of values to check, 
			allowed strings in ['$value,',#index,','%start,stop,step'] for comma-delimited values, indices, or slices values
		keys (iterable[str]): All keys of value to group with
		values (dict[str,dict[str,object]]): All dataset values of {name:{key:value}}
	Returns:
		permutations (dict): Dictionary of all permutations of keys {tuple(value of k in key):list(tuple(value of k for k in keys))}	
	'''

	def apply(group,key,value):
		
		def _apply(group,key,value):
			group = group.groupby(key)
			groups = list(group.groups)
			if value is None:
				value = slice(None)
			elif isinstance(value,str):
				if value.startswith('#') and value.endswith('#'):
					parser = lambda value: (int(value) if len(value)>0 else 0)
					value = value.replace('#','').split(',')
					value = [parser(v) for v in value]
				elif value.startswith('%') and value.endswith('%'):
					parser = lambda value: (int(value) if len(value)>0 else 0)
					value = value.replace('%','').split(',')
					value = [parser(v) for v in value]                
					if len(value) == 1:
						value = None,None,*value[:1]
					elif len(value) == 2:
						value = *value[:2],None
					else:
						value = *value[:3],
					value = slice(*value)
				elif value.startswith('$') and value.endswith('$'):
					parser = lambda value: (to_number(value) if len(value)>0 else 0)
					value = value.replace('$','').split(',')
					value = [parser(v) for v in value]
					value = [groups.index(v) for v in value if v in groups]
				else:
					value = slice(None)
			
			elif isinstance(value,(tuple,list,slice)):
				pass
			else:
				value = slice(None)
			groups = np.array(groups)


			return {v: group.get_group(v).index.values.tolist() for v in groups[value]}
		
		_group = {}
		for _key,_value in zip(key,value):

			by = [_key_ for _key_ in key if _key_!=_key]
			
			index = [by.index(_key_) if _key_ in by else -1 for _key_ in key]
			
			if len(by) == 0:            
				_group_ = {():_apply(group,key=_key,value=_value)}
			else:
				_group_ = dict(group.groupby(by).apply(_apply,key=_key,value=_value))

			_group[_key] = {
				tuple(([*(_key_ if isinstance(_key_,tuple) else [_key_]),__key__][i] for i in index)):__value__
				for _key_,_value_ in _group_.items()
				for __key__,__value__ in _value_.items()
			}

		unique = list(set.intersection(*map(set,_group.values())))
		return unique

	values = pd.DataFrame.from_dict(values,orient='index').reset_index()
	values[keys] = values[keys].applymap(lambda data:np.squeeze(data))
	values = values.sort_values(keys)

	by = [*[attr for attr in keys if attr not in key]]
	
	values = values.groupby(by)
	
	unique = dict(values.apply(apply,key=key,value=value))

	unique = {u if isinstance(u,tuple) else (u,):unique[u] for u in unique}

	combinations = list(sorted(set((v if isinstance(v,tuple) else (v,) for u in unique.values() for v in u))))

	permutations = {combination: [
		tuple((combination[key.index(attr)] if attr in key else u[by.index(attr)] for attr in keys))
		for u in unique if combination in unique[u]] for combination in combinations}
	
	return permutations







def find(dictionary,properties):
	'''
	Find formatted keys of the form ({prop:attr} or {prop:{'key':(attr,),'value:(values,)}})
	from dictionary, based on search properties 
	All properties are assumed to be present in any branches where one or more property is found in dictionary
	Args:
		dictionary (dict): Dictionary to search
		properties (iterable[str]): Iterable of properties to search for
	Returns:
		keys (list[dict]): Formatted keys based on found properties in dictionary
	'''

	# keys = (leaves(dictionary,prop,types=(dict,list),returns='value') for prop in properties)
	# keys = map(lambda key: dict(zip(
	# 	properties,(*key[:2],tuple((dict(zip(['key','value'],to_key_value(key[2],delimiter='='))),))
	# 	if key[2] is None or isinstance(key[2],str) 
	# 	else tuple(dict(zip(['key','value'],to_key_value(j,delimiter='='))) for j in key[2])))),
	# 	zip(*keys))

	keys = branches(dictionary,properties,types=(dict,list),returns='value')

	keys = map(lambda key: dict(zip(
		properties,(tuple((dict(zip(['key','value'],to_key_value(k,delimiter='='))),))
						if k is None or isinstance(k,str) 
						else tuple(dict(zip(['key','value'],to_key_value(j,delimiter='='))) for j in k) for k in key))),
		keys)


	keys = [{prop: {
				label:tuple(value[label] for value in key[prop]) for label in set(
					label for value in key[prop] for label in value)}
			 for prop in key} 
			 for key in keys]

	for key in keys:
		if keys.count(key)>1:
			keys.remove(key)

	return keys


def dumper(kwargs,**options):
	'''
	Load data
	Args:
		kwargs (dict[str,object]): Data to dump
		options (dict): Additional dump options
	'''

	for kwarg in kwargs:
		path = kwarg
		data = kwargs[kwarg]

		dump(data,path,**options)

		print('Dumped',path)

	return

def loader(kwargs,**options):
	'''
	Load data
	Args:
		kwargs (dict[str,object]): Data to load
		options (dict): Additional load options
	'''

	returns = {}

	returns['multiple'] = False

	for kwarg in kwargs:
		if isinstance(kwargs[kwarg],str):
			kwargs[kwarg] = [kwargs[kwarg]]		
		if not isinstance(kwargs[kwarg],dict):
			paths = kwargs[kwarg]
			paths = natsorted(set((subpath for path in set(paths) for subpath in glob(path))))
			default = {}
			kwargs[kwarg] = {}
			returns['multiple'] |= len(paths)>1
			for path in paths:
				kwargs[kwarg].update(load(path,default=default,**options))
				print('Loaded:',path)				
		else:
			kwargs[kwarg] = kwargs[kwarg]
			returns['multiple'] |= False
	

	if all(len(kwargs[kwarg]) < 1 for kwarg in kwargs):
		returns = None

	return returns


def process(data,settings,hyperparameters,fig=None,ax=None,cwd=None):
	'''
	Process data
	Args:
		data (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of data to process
		settings (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of plot settings
		hyperparameters (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of process settings
		fig (dict): dictionary of subplots of figures of plots {key: figure}
		ax (dict): dictionary of subplots of axes of plots {key: figure}
		cwd (str): Root path
	Returns:
		fig (dict): dictionary of subplots of figures of plots {key: figure}
		ax (dict): dictionary of subplots of axes of plots {key: figure}

	To process data, we find in plot settings dictionary the keys of 'x','y','label' properties for sorting.

	For 'label' property with attributes and values to sort on, 
	datasets are sorted into sets of unique datasets that correspond 
	to all possible combinations of the label values. i.e) if label is ('M','N'), 
	will sort into sets of datasets that correspond to each possible 'M','N' pair. 
	
	For each combination of specific label values, statistics about the set of sample datasets 
	corresponding to the label values are computed/
	
	Label attributes with a non-None value indicate fixed values that the datasets must equal for that attribute
	when sorting, and the set of datasets for that label combination are constrained to be those with the fixed values.

	- Iterate over all combinations of 'label' attributes and values, to get included dataset that share these attributes
	
	- Iterate over all permutations of sort attributes and values, constrained by the specific combination of 'label' attributes and values
	to get included dataset that share all sort attributes
	
	- Get statistics (mean,variance) across samples datasets that share all attributes

	- Merge datasets across permutations of sort attributes for a given combination of 'label' attributes and values
	
	- After statistics and merging, variables data for each combination for 'label' attributes and values
	has shape of (1 + ndim) dimensions, with axes: 
	(# of permutations of sort for each combination, ndim of datasets)	

	- For parameters, we plot each combination for 'label' attributes and values variables data on the same plot, these being the labels.
	  Subplots are plotted from iterating over over the 1...ndim-2 axis of the variables data, and parameters the 0 and ndim-1 axis for each 'label' set
	  If the 'x' property is None, also iterate over the 0 (# of permutations of sort) axis variables data, and plot the ndim-1 axis for each 'label' 
	'''

	# Setup kwargs
	kwargs = ['settings','hyperparameters']
	kwargs = {kwarg: value for kwarg,value in zip(kwargs,[settings,hyperparameters])}

	returns = loader(kwargs)

	if returns is None:
		return fig,ax

	settings,hyperparameters = (kwargs[kwarg] for kwarg in kwargs)

	# Get paths
	path,file,directory,ext,delimiter = {},{},{},{},{}
	hyperparameters['plot'] = {'plot':'plot.pdf','process':'process.hdf5',**hyperparameters.get('path',{})}
	for attr in hyperparameters['plot']:
		delimiter[attr] = hyperparameters.get('delimiter','.')
		directory[attr],file[attr],ext[attr] = split(
			hyperparameters['plot'][attr],
			directory=True if (not returns['multiple']) else -1,file=True,ext=True) 
		if (hyperparameters.get('cwd') is not None):
			directory[attr] = hyperparameters.get('cwd')
		else:
			directory[attr] = cwd

		path[attr] = join(directory[attr],file[attr],ext=ext[attr])

	# Get plot variables setting
	parameters = hyperparameters.get('parameters',[])
	null = hyperparameters.get('null',{})
	for instance in list(null):
		if instance not in settings:
			null.pop(instance)
			continue
		if null[instance] is None:
			null[instance] = list(settings[instance])
		
		for subinstance in null[instance]:
			if subinstance in settings[instance]:
				settings[instance].pop(subinstance)
		if len(settings[instance]) == 0:
			settings.pop(instance)


	# Get plot fig and axes
	axes = ['x','y']
	if fig is None:
		fig = {}
	if ax is None:
		ax = {}

	for instance in settings:
		if instance not in fig:
			fig[instance] = None
		if instance not in ax:
			ax[instance] = None

	# Get texify
	texify = lambda string: Texify(string,hyperparameters.get('texify',{}),usetex=hyperparameters.get('usetex',True))

	# Get plot properties and statistics from settings
	properties = [*['%s'%(axis) for axis in axes],'label']
	statistics = [*['%s'%(axis) for axis in axes],*['%serr'%(axis) for axis in axes]]
	statistics = {
		kwarg: {
			**{kwarg:{
				'property':kwarg.replace('',''),
				'statistic':{
					**{stat: lambda key,data,variables=None,dtype=None,axis=axis,stat=stat,**kwargs: mean(
						data,axis=0,dtype=dtype,
						# transform=stat[axis]
						transform='linear',#stat[axis]
						) for stat in itertools.product(['linear','log'],repeat=len(axes))
					},
					('fit','fit'): lambda key,data,variables=None,dtype=None,axis=axis,**kwargs: mean(
						data,axis=0,dtype=dtype,transform=None),
					}
				} 
				for axis,kwarg in enumerate(['%s'%(axis) for axis in axes])},
			**{kwarg:{
				'property':kwarg.replace('err',''),
				'statistic':{			
					**{stat: lambda key,data,variables=None,dtype=None,axis=axis,stat=stat,**kwargs: std(
						data,axis=0,dtype=dtype,
						transform=stat[axis]						
						# transform='linear',#stat[axis]
						) for stat in itertools.product(['linear','log'],repeat=len(axes))
					},				
					('fit','fit'): lambda key,data,variables=None,dtype=None,axis=axis,**kwargs: std(
						data,axis=0,dtype=dtype,transform=None),
					}
				}	 
				for axis,kwarg in enumerate(['%serr'%(axis) for axis in axes])},
			}[kwarg]
		for kwarg in statistics 			 	 			 	 
		}


	# Get occurrence of keys
	# TODO: Allow for hdf5 loading/dumping of occurrence tuples, to process intersections of keys only once (including keys that are not attributes for plot labelling)
	# occurrences = lambda key,keys: tuple((tuple((axis,tuple(((k,v) for k,v in zip(key[axis]['key'],key[axis]['value']))))) for axis in key))
	# _occurrences = lambda occurrence,keys: {axis: {'key': tuple((v[0] for v in value)), 'value': tuple((v[1] for v in value))} for axis,value in occurrence}
	occurrences = lambda key,keys: keys.index(key)#tuple((tuple((axis,tuple(((k,v) for k,v in zip(key[axis]['key'],key[axis]['value']))))) for axis in key))
	_occurrences = lambda occurrence,keys: keys[occurrence]#{axis: {'key': tuple((v[0] for v in value)), 'value': tuple((v[1] for v in value))} for axis,value in occurrence}


	# Get keys of properties of the form ({prop:attr} or {prop:{'key':(attr,),'value:(values,)}})
	keys = find(settings,properties)

	# Load data
	if hyperparameters.get('load'):
		attr = 'process'
		options = {
			'conversion': to_eval
			}
		variables = {attr: path[attr]}

		returns = loader(variables,**options)

		variables = variables[attr]

		combinations = {
			occurrence: [
				list(set((combination[i] for combination in variables[occurrence])))
				for i in range(min(len(combination) for combination in variables[occurrence]))
				]
			for occurrence in variables
			}

		permutations = {
			occurrence: {
				combination:[
					list(set((permutation[i] for permutation in variables[occurrence][combination])))
					for i in range(min(len(permutation) for permutation in variables[occurrence][combination]))
					]
				for combination in variables[occurrence]
				}
			for occurrence in variables
			}	

	else:

		# Setup kwargs
		kwargs = ['data']
		kwargs = {kwarg: value for kwarg,value in zip(kwargs,[data])}

		returns = loader(kwargs)

		if returns is None:
			return fig,ax

		data, = (kwargs[kwarg] for kwarg in kwargs)


		# Get dataset names of data
		names = list(natsorted(set(name for name in data),key=lambda name:name))

		# Get attributes of data
		attributes = list(set(attr for name in names 
			for attr in data[name]
			))


		subattributes = [attr
			for attr in attributes
			# if ((attr in hyperparameters.get('sort',attributes)))
			if ((attr in set((attr for key in keys for attr in key['label']['key']))))
			]


		# Get unique scalar attributes
		unique = {attr: tuple((*realsorted(set(asscalar(data[name][attr])
						for name in names 
						if ((attr in data[name]) and 
							((asarray(data[name][attr]).size <= 1) and isinstance(asscalar(data[name][attr]),scalars))
							)
						)),None))
				for attr in attributes
				}
		unique = {attr: unique[attr] for attr in unique if len(unique[attr])>0}	

		# Get attributes to sort on and attributes not to sort on if not existent in plot properties x,y,label
		sort = {attr: tuple((*realsorted(set(asscalar(data[name][attr])
						for name in names 
						if ((attr in data[name]) and 
							((asarray(data[name][attr]).size == 1) and isinstance(asscalar(data[name][attr]),scalars))
							)
						)),None))
				for attr in subattributes
				}
		sort = {attr: sort[attr] for attr in sort if len(sort[attr])>0}

		# Get combinations of attributes (including None) to sort on and attributes not to sort on if not existent in plot properties x,y,label
		allowed = list(realsorted(set((tuple((asscalar(data[name][attr])
					for attr in subattributes))
					for name in names
					))))
		allowed = [
					*allowed,
					*realsorted(set([(*value[:i],None,*value[i+1:]) for value in allowed for i in range(len(value))]),
							key = lambda x: tuple(((u is not None,u) for u in x)))
				]

		# Get data as arrays, with at least 1 leading dimension
		for name in names:
			for attr in data[name]:
				data[name][attr] = np.array(data[name][attr])
				data[name][attr] = data[name][attr].reshape(*[1]*(max(0,1-data[name][attr].ndim)),*data[name][attr].shape)


		# Get number of dimensions and maximum shape of data attributes
		ndim = {attr: min(data[name][attr].ndim for name in names) for attr in attributes}
		shape = {attr: tuple(map(max,zip(*(data[name][attr].shape for name in names)))) for attr in attributes}

		
		# Get labels of keys
		labels = {occurrences(key,keys):{
					prop: {attr:value for attr,value in zip(key[prop]['key'],key[prop]['value']) if attr in attributes} 
					for prop in key} 
				for key in keys}

		# Get combinations of key attributes and permutations of shared attributes for combination across data

		variables = {}
		combinations = {}	
		permutations = {}

		# Get variables data and statistics from keys
		# for occurrence,key in enumerate(keys):
		for key in keys:
			occurrence = occurrences(key,keys)

			parameter = [None,*[parameter for parameter in parameters 
				if (all(tuple((parameter['key'][axis],) if not is_iterable(parameter['key'][axis],exceptions=(str,)) else parameter['key'][axis]) == 
					key[axis]['key'] for axis in parameter['key']))]][-1]
			
			variables[occurrence] = {}

			# combinations[occurrence] = permute(key['label']['key'],key['label']['value'],list(sort),data)
			combinations[occurrence] = 	[
				[val for val in sort.get(attr,[]) 
				if check(attr,labels[occurrence]['label'][attr],val,labels[occurrence],unique,data)]
				for attr in labels[occurrence]['label']
				]

			permutations[occurrence] = {}

			print('key',key,combinations[occurrence],parameter)

			# for combination in combinations[occurrence]:
			for combination in itertools.product(*combinations[occurrence]):
				variables[occurrence][combination] = {}
				label = {prop: dict(zip(
					labels[occurrence][prop],
					combination if prop in ['label'] else [labels[occurrence][prop][attr] for attr in labels[occurrence][prop]])) 
				for prop in labels[occurrence]}

				if any((len(label[attr])==0) and all(len(val)>0 for val in key[attr]['key']) for attr in label):
					print('popping',combination,label)
					variables[occurrence].pop(combination);
					continue
				
				allincluded = [name for name in names 
					if include(name,{'label':labels[occurrence]['label']},label['label'],sort,data)]

				value = variables[occurrence].pop(combination)

				if len(allincluded) == 0:
					continue



				name = allincluded[-1]
				combination = tuple(sorted(tuple((
					(attr,asscalar(data[name][attr]))
					for attr in data[name] if (
						data[name][attr].size == 1)
					)),
					key = lambda x: key['label']['key'].index(x[0]) if x[0] in key['label']['key'] else -1))

				variables[occurrence][combination] = value

				permutations[occurrence][combination] = [
					permutation for permutation in allowed if all(
						(((attr not in label['label']) and (val is not None)) or ((attr in label['label']) and (val == label['label'][attr])))
						for attr,val in zip(subattributes,permutation)
						)
					]

				print('combination',label,combination)
				print({attr: labels[occurrence][prop][attr] 
								for prop in labels[occurrence] 
								for attr in labels[occurrence][prop]})
				print()

				for permutation in permutations[occurrence][combination]:
				# for permutation in itertools.product(*permutations[occurrence][combination]):

					variables[occurrence][combination][permutation] = {}
					values = dict(zip(subattributes,permutation))

					# included = [name for name in names 
					# 	if all(data[name][attr] == values[attr]
					# 	for attr in values)]
					included = [name for name in allincluded 
						if include(name,labels[occurrence],values,sort,data)
						]

					
					if len(included) == 0:
						variables[occurrence][combination].pop(permutation);
						continue

					print('permutation',label,values,permutation)
					print(included)					
					print()
					print()

					for kwarg in statistics:


						variables[occurrence][combination][permutation][kwarg] = {}

						prop = statistics[kwarg]['property']
						isnull = key[prop]['key'][-1] in nulls
						if isnull:
							prop = 'y'					
							dtype = int
						else:
							prop = prop
							name = included[-1]
							dtype = data[name][key[prop]['key'][-1]].dtype

						newshape = (len(included),*shape[key['y']['key'][-1]])
						newndim = range(0,ndim[key['y']['key'][-1]]-ndim[key[prop]['key'][-1]])

						variables[occurrence][combination][permutation][kwarg] = {}

						# Insert data into variables (with nan padding)
						for stat in statistics[kwarg]['statistic']:

							if 	(
								((parameter is None) and (stat not in [('linear','linear')])) or 
								((parameter is not None) and (parameter.get('stat') is None) and (stat not in [('linear','linear')])) or
								((parameter is not None) and ((parameter.get('stat') is not None) and (
										stat not in (tuple(substat) 
										for substat in parameter.get('stat',[[]]))))) or
								((parameter is not None) and (False))
								):
								continue

							variables[occurrence][combination][permutation][kwarg][stat] = np.nan*np.ones(newshape)


							for index,name in enumerate(included):
								if isnull:
									value = np.arange(data[name][key[prop]['key'][-1]].shape[-1])
								else:
									value = data[name][key[prop]['key'][-1]]

								if value is None or value.dtype.type is np.str_:
									dtype = variables[occurrence][combination][permutation][kwarg][stat].dtype
									continue

								# try:
								value = expand_dims(value,newndim)
								slices = (index,*(slice(data[name][key['y']['key'][-1]].shape[axis]) for axis in range(data[name][key['y']['key'][-1]].ndim)))
								variables[occurrence][combination][permutation][kwarg][stat][slices] = value
								# except:
									# raise
							variables[occurrence][combination][permutation][kwarg][stat] = statistics[kwarg]['statistic'][stat](
								key,variables[occurrence][combination][permutation][kwarg][stat],
								variables=variables[occurrence][combination][permutation],dtype=dtype)

				variables[occurrence][combination] = {
					kwarg:{
						stat: np.array([variables[occurrence][combination][permutation][kwarg][stat] 
										for permutation in variables[occurrence][combination]])
						for stat in statistics[kwarg]['statistic']
						if all(stat in variables[occurrence][combination][permutation][kwarg]
							for permutation in variables[occurrence][combination])
						}
					for kwarg in statistics}

			if len(variables[occurrence]) == 0:
				variables.pop(occurrence)

		# Delete data
		del data

		# Dump data
		if hyperparameters.get('dump'):
			attr = 'process'
			options = {
				'conversion': to_repr
			}

			kwargs = {path[attr]: variables}

			dumper(kwargs,**options)
	





	# Plot data

	# Default setting objects for each settings instance
	defaults = {}

	defaults['ax'] = {}
	defaults['fig'] = {}
	defaults['style'] = {
		'layout': {
			'nrows':1,'ncols':1,'index':1,
			'left':None,'right':None,'top':None,'bottom':None,
			'hspace':None,'wspace':None,'pad':None
			}
	}
	
	# Special settings to set variables depending on variables data shape, with additional settings to custom set form settings string
	special = {'ax':['plot','fill_between','errorbar',]}

	# Track updated keys
	updated = []

	# Set default plot settings and remove plots with no variables data
	for instance in list(settings):	
		if any(setting in settings[instance] for setting in defaults):
			subinstance = None
			settings[instance] = {subinstance: settings[instance]}

		for subinstance in list(settings[instance]):
			for setting in special:
				for attr in special[setting]:
					if attr in settings[instance][subinstance][setting]:

						if isinstance(settings[instance][subinstance][setting][attr],dict):
							settings[instance][subinstance][setting][attr] = [settings[instance][subinstance][setting][attr]]

						for i in range(len(settings[instance][subinstance][setting][attr])-1,-1,-1):							
							key = find(settings[instance][subinstance][setting][attr][i],properties)[0]
							# occurrence = keys.index(key)
							occurrence = occurrences(key,keys)
							
							parameter = [None,*[parameter for parameter in parameters 
									if (all(tuple((parameter['key'][axis],) if not is_iterable(parameter['key'][axis],exceptions=(str,)) else parameter['key'][axis]) == 
										key[axis]['key'] for axis in parameter['key']))]][-1]

							# if occurrence not in variables:
							# 	for _occurrence in variables:
							# 		_key = _occurrences(_occurrence,keys)
							# 		if all(key[axis]['key'] == _key[axis]['key'] for axis in key):
							# 			occurrence = _occurrence

							if occurrence not in variables:
								settings[instance][subinstance][setting][attr].pop(i)
								continue

						if len(settings[instance][subinstance][setting][attr]) == 0:
							settings[instance][subinstance][setting].pop(attr)
			if not any(attr in settings[instance][subinstance][setting] 
				for setting in special 
				for attr in special[setting]):
				settings[instance].pop(subinstance)

		if len(settings[instance]) == 0:
			settings.pop(instance)

	# Get layout of plot instances		
	dim = 2
	kwargslayout = list(defaults['style']['layout'])
	updated = []
	subupdated = []
	layout = {
		instance:{
			kwarg: {
				subinstance: settings[instance][subinstance]['style'].get('layout',{}).get(
					kwarg,defaults['style']['layout'][kwarg])
					for subinstance in settings[instance]
				}
			for kwarg in kwargslayout
			}
		for instance in settings
		}
	subshape = {}



	# Form grids of layout depending on shape of variables in each plot
	# Get update to layout based on (reshaped) variables data shape and 
	# reshaping of variables data based on 
	# parameters = {'y':{'x':{'axis':{attr:[[axis for ncols],[axis for nrows],[axis for labels][axis for plot]]}}}}
	for instance in list(settings):
		for subinstance in list(settings[instance]):
			subupdated.clear()
			for setting in special:
				for attr in special[setting]:
					if attr in settings[instance][subinstance][setting]:

						for i in range(len(settings[instance][subinstance][setting][attr])):							
							key = find(settings[instance][subinstance][setting][attr][i],properties)[0]
							# occurrence = keys.index(key)
							occurrence = occurrences(key,keys)

							parameter = [None,*[parameter for parameter in parameters 
								if (all(tuple((parameter['key'][axis],) if not is_iterable(parameter['key'][axis],exceptions=(str,)) else parameter['key'][axis]) == 
									key[axis]['key'] for axis in parameter['key']))]][-1]

							# if occurrence not in variables:
							# 	for _occurrence in variables:
							# 		_key = _occurrences(_occurrence,keys)
							# 		if all(key[axis]['key'] == _key[axis]['key'] for axis in key):
							# 			occurrence = _occurrence

							if occurrence not in variables:
								continue

							if occurrence not in updated:
								
								updated.append(occurrence)

								for combination in variables[occurrence]:
									for kwarg in variables[occurrence][combination]:
										for stat in variables[occurrence][combination][kwarg]:

											wrapper = wrapping(**parameter.get('wrapper',{}),kwarg=kwarg,stat=stat) if parameter is not None else None 
											if wrapper is not None:
												variables[occurrence][combination][kwarg][stat] = wrapper(variables[occurrence][combination][kwarg][stat])
											
											subndim = variables[occurrence][combination][kwarg][stat].ndim

											subaxis = parameter.get('axis',None) if parameter is not None else None

											if subaxis is None:
												subaxis = [[],[],[],[axis for axis in range(subndim)]]
											else:
												subaxis = [[axis] if isinstance(axis,int) else axis for axis in subaxis]

											subaxis = [*([a for a in axis] for axis in subaxis[:-1]),
														[a for a in range(subndim) 
														 if (a in subaxis[-1] or (a == (subndim-1) and -1 in subaxis[-1])) or 
														 a not in [a for axis in subaxis[:-1] for a in axis]]]

											transpose = [a for axis in subaxis for a in axis]

											reshape = [
												max(1,int(product(
													[variables[occurrence][combination][kwarg][stat].shape[a]
												for a in axis])))
												for axis in subaxis]
											variables[occurrence][combination][kwarg][stat] = (
												variables[occurrence][combination][kwarg][stat].transpose(
												transpose).reshape(reshape)
												)
							if occurrence not in subupdated:
								subupdated.append(occurrence)


			subshape[subinstance] = [max(variables[occurrence][combination][kwarg][stat].shape[axis]
							for occurrence in subupdated
							for combination in variables[occurrence]
							for kwarg in variables[occurrence][combination]
							for stat in variables[occurrence][combination][kwarg])
							for axis in range(dim)]

		# Get unique layouts
		setlayout = {kwarg: set(layout[instance][kwarg][subinstance] for subinstance in layout[instance][kwarg])
						for kwarg in layout[instance]
					}

		# Get subinstances for each unique layout
		layouts = {}
		for samplelayout in itertools.product(*(setlayout[kwarg] for kwarg in setlayout)):
			
			samplelayouts = dict(zip(setlayout,samplelayout))
			
			subinstances = set(subinstance 
				for kwarg in layout[instance] 
				for subinstance in layout[instance][kwarg])
			subinstances = [subinstance for subinstance in subinstances
							if all(layout[instance][kwarg][subinstance]==samplelayouts[kwarg] 
								for kwarg in samplelayouts)]
			
			if len(subinstances) == 0:
				continue

			layouts[samplelayout] = subinstances


		# Boolean whether to form grid of subsubplots of subplots, or subplots directly
		subsublayouts = len(layouts) > 1

		for samplelayout in layouts:
			for subinstance in layouts[samplelayout]:
				for index,position in enumerate(itertools.product(*(range(subshape[subinstance][axis]) for axis in range(dim)))):
					
					samplelayouts = dict(zip(setlayout,samplelayout))

					if not 	subsublayouts:
						samplelayouts.update({
							kwarg: 1 for kwarg in kwargslayout[:dim+1]
							})

					indx = samplelayouts['index']-1	
					nrow = (indx - indx%samplelayouts['ncols'])//samplelayouts['ncols']
					ncol = indx%samplelayouts['ncols']

					settings[instance][(subinstance,*position)] = copy.deepcopy(settings[instance][subinstance])

					settings[instance][(subinstance,*position)]['style']['layout'] = {
						kwarg:{
							**{kwarg: layout[instance][kwarg][subinstance] for kwarg in layout[instance]},
							**{kwargslayout[axis]:subshape[subinstance][axis] for axis in range(dim)},
							'index':index+1,
							'top':1 - (nrow)/samplelayouts['nrows'] if subsublayouts and samplelayouts['nrows']>1 else None,
							'bottom':1 - (nrow+1)/samplelayouts['nrows'] if subsublayouts and samplelayouts['nrows']>1 else None,
							'right':(ncol+1)/samplelayouts['ncols'] if subsublayouts and samplelayouts['ncols']>1 else None,
							'left':(ncol)/samplelayouts['ncols'] if subsublayouts and samplelayouts['ncols']>1 else None,							
							}[kwarg]
						for kwarg in layout[instance]
						}

					for setting in special:
						for attr in special[setting]:
							if attr in settings[instance][subinstance][setting]:

								settings[instance][(subinstance,*position)][setting][attr] = {}

								for i in range(len(settings[instance][subinstance][setting][attr])):
									key = find(settings[instance][subinstance][setting][attr][i],properties)[0]
									
									occurrence = occurrences(key,keys)
									
									parameter = [None,*[parameter for parameter in parameters 
										if (all(tuple((parameter['key'][axis],) if not is_iterable(parameter['key'][axis],exceptions=(str,)) else parameter['key'][axis]) == 
											key[axis]['key'] for axis in parameter['key']))]][-1]									

									subsize = max(variables[occurrence][combination][kwarg][stat].shape[dim-1+1]
												for combination in variables[occurrence]
												for kwarg in variables[occurrence][combination]
												for stat in variables[occurrence][combination][kwarg])												
									for enum,(combination,j) in enumerate(realsorted(
											itertools.product(variables[occurrence],range(subsize))  	,
											key=lambda x: tuple((dict(x[0]).get(k) for k in key['label']['key']))
											)):

										subsubsize = max(variables[occurrence][combination][kwarg][stat].shape[dim-1+1]
													for kwarg in variables[occurrence][combination]
													for stat in variables[occurrence][combination][kwarg])

										if j >= subsubsize:
											continue


										substatistics = set(stat 
											for kwarg in variables[occurrence][combination] 
											for stat in variables[occurrence][combination][kwarg])
										
										for stat in substatistics:

											subsettings = copy.deepcopy(settings[instance][subinstance][setting][attr][j])

											for kwarg in variables[occurrence][combination]:
												pos = tuple(
													(*(position[axis]%variables[occurrence][combination][kwarg][stat].shape[axis] for axis in range(dim)),
													j%variables[occurrence][combination][kwarg][stat].shape[dim-1+1]))

												value = variables[occurrence][combination][kwarg][stat][pos]

												slices = slice(*parameter.get('slice') if parameter.get('slice') is not None else (None,)) if parameter is not None else slice(None)

												value = value[slices]

												if kwarg in ['%serr'%(axis) for axis in axes] and norm(value) == 0:
													value = None

												subsettings[kwarg] = value

											settings[instance][(subinstance,*position)][setting][attr][
												(combination,j,occurrence,stat)] = subsettings

		for samplelayout in layouts:
			for subinstance in layouts[samplelayout]:
				settings[instance].pop(subinstance)



	# Set plot settings
	for instance in list(settings):
		for subinstance in list(settings[instance]):
			for setting in list(settings[instance][subinstance]):
				
				for a,attr in enumerate(special.get(setting,[])):

					if attr not in settings[instance][subinstance][setting]:
						continue 

					subcombinations,subj,suboccurrences,substats = zip(*settings[instance][subinstance][setting][attr])
					subcombinations = tuple((dict(subcombination) for subcombination in subcombinations))
					for i,subsubinstance in enumerate(settings[instance][subinstance][setting][attr]):

						combination,j,occurrence,stat = subsubinstance

						key = _occurrences(occurrence,keys)

						combination = dict(combination)

						sorting = realsorted(set([
							tuple((subcombination[k] for k in key['label']['key'] 
								if k in hyperparameters.get('sort'))) 
							for subcombination in subcombinations]))
						elements = tuple((combination[k] for k in key['label']['key'] 
								if k in hyperparameters.get('sort')))
						multiple = any(len(subcombination)>0 for subcombination in subcombinations)
						index = sorting.index(elements)
						number = len(sorting)
						proportion = index/number

						# Add data-dependent plots
						if key['y']['key'][-1] in []:
							pass

						kwargs = ['color','ecolor']
						for kwarg in kwargs:
							if settings[instance][subinstance][setting][attr][subsubinstance].get(kwarg) is None:
								continue
								settings[instance][subinstance][setting][attr][subsubinstance][kwarg]

							value = getattr(plt.cm,settings[instance][subinstance][setting][attr][subsubinstance][kwarg])(proportion)

							settings[instance][subinstance][setting][attr][subsubinstance][kwarg] = value

						kwargs = ['label']
						for kwarg in kwargs:
							if settings[instance][subinstance][setting][attr][subsubinstance].get(kwarg) is None:
								continue

							if attr != [attr for attr in special.get(setting,[]) if attr in settings[instance][subinstance][setting]][-1]:
								settings[instance][subinstance][setting][attr][subsubinstance].pop(kwarg)
								continue

							if stat not in [('fit','fit')]:
								value = [combination[k] for i,k in enumerate(combination) if k in key['label']['key'] and len(set([subcombination[k] for subcombination in subcombinations]))>1]
								value = [texify(scinotation(k,decimals=0,scilimits=[0,3])) for k in value]
								value = [*value,*[str(combination.get(v.replace('@',''),v)) if v is not None else k for k,v in zip(key['label']['key'],key['label']['value']) if k not in hyperparameters.get('sort')]]
								value = [v for v in value if v is not None and len(v)>0]
								value = ',~'.join(value)

							else:
								value = None

							settings[instance][subinstance][setting][attr][subsubinstance][kwarg] = value
						
						kwargs = ['linestyle']
						for kwarg in kwargs:
							if settings[instance][subinstance][setting][attr][subsubinstance].get(kwarg) is None:
								continue

							if stat is None:
								value = None
							elif stat not in [('fit','fit')]:
								value = list(sorted(list(set([(suboccurrence)
									for suboccurrence in suboccurrences 
									]))))
								value = (value.index(occurrence) if len(suboccurrences)>0 else 0)
								value = ['solid','dotted','dashed','dashdot',(0,(5,10)),(0,(1,1))][value%6]
							else:
								value = None

							settings[instance][subinstance][setting][attr][subsubinstance][kwarg] = value


						kwargs = ['alpha']
						for kwarg in kwargs:
							if settings[instance][subinstance][setting][attr][subsubinstance].get(kwarg) is None:
								continue
							if stat is None:
								value = None
							elif stat not in [('fit','fit')]:
								value = (index+1)/(number+1) if multiple else None
							else:
								value = None

							settings[instance][subinstance][setting][attr][subsubinstance][kwarg] = value



						subattr = 'legend'
						kwargs = ['set_title']
						if settings[instance][subinstance][setting].get(subattr) is None:
							continue
						for kwarg in kwargs:
							value = [
								[(k,) for i,k in enumerate(combination) if k in key['label']['key'] and len(set([subcombination[k] for subcombination in subcombinations]))>1],
								]
							value = [
								[[texify(l) for l in k if l is not None and len(l)>0] if not isinstance(k,str) else texify(k) 
								for k in [*v,[k if v is not None else None for k,v in zip(key['label']['key'],key['label']['value']) if k not in hyperparameters.get('sort')]
								] if len(k)>0]
								for i,v in enumerate(value) if len(v)>0]

							value = ['~,~'.join([': '.join([j for j in k if j is not None and len(j)>0]) if not isinstance(k,str) else k for k in v if k is not None and len(k)>0 and all(j is not None and len(j) for j in k)]) 
									for v in value if v is not None and len(v)>0]

							value = [v for v in value if len(v)>0]
							value = '\n'.join(['$%s$'%(v.replace('$','')) for v in value])

							value = value if (
								(settings[instance][subinstance][setting][subattr].get(kwarg) is None) or 
								any(len(k)>len(v) for k,v in zip(
									[v.split('~,~') for v in value.split('\n')],
									[v.split('~,~') for v in settings[instance][subinstance][setting][subattr].get(kwarg).split('\n')]))
									) else settings[instance][subinstance][setting][subattr].get(kwarg)

							settings[instance][subinstance][setting][subattr][kwarg] = value


					settings[instance][subinstance][setting][attr] = [
						settings[instance][subinstance][setting][attr][subsubinstance]
						for subsubinstance in settings[instance][subinstance][setting][attr]
						]



			# Set custom plot settings
			attrs = {
				'fig':{'savefig':['fname']},
				'ax':{'set_ylabel':['ylabel'],'errorbar':['%serr'%(axis) for axis in axes],'fill_between':['%serr'%(axis) for axis in axes]},
				}
			for setting in attrs:
				if setting not in settings[instance][subinstance]:
					settings[instance][subinstance][setting] = {}
				for attr in attrs[setting]:
					if settings[instance][subinstance][setting].get(attr) is None:
						continue
					for kwarg in attrs[setting][attr]:

						multiple = not isinstance(settings[instance][subinstance][setting][attr],dict)
						length = len(settings[instance][subinstance][setting][attr]) if multiple else None

						if not multiple:
							if kwarg not in settings[instance][subinstance][setting][attr]:
								settings[instance][subinstance][setting][attr][kwarg] = None
						else:
							for i in range(len(settings[instance][subinstance][setting][attr])):
								if kwarg not in settings[instance][subinstance][setting][attr][i]:
									settings[instance][subinstance][setting][attr][i][kwarg] = None	

						value = settings[instance][subinstance][setting][attr]

						if setting in ['fig'] and attr in ['savefig'] and kwarg in ['fname']:

							do = True

							if not do:
								continue

							subvalue = 'plot'
							new = join(directory[subvalue],
										delimiter[subvalue].join([
											*file[subvalue].split(delimiter[subvalue])[:],
											instance,
											]),
										ext=ext[subvalue])
							if not multiple:
								value[kwarg] = new
							else:
								for i in range(length):
									value[i][kwarg] = new

						elif setting in ['ax'] and attr in ['set_ylabel'] and kwarg in ['ylabel']:

							index = settings[instance][subinstance]['style']['layout']['index']
							nrows = settings[instance][subinstance]['style']['layout']['nrows']
							ncols = settings[instance][subinstance]['style']['layout']['ncols']

							do = not (((nrows is None) or (nrows == 1)) and ((ncols is None) or (ncols == 1)))

							if not do:
								continue

							index = index - 1	
							nrow = (index - index%ncols)//ncols
							ncol = index%ncols

							if not multiple:

								if isinstance(value[kwarg],str):
									new = value[kwarg]
								elif all(isinstance(v,str) for v in value[kwarg]):
									new = value[kwarg][index%len(value[kwarg])]
								elif value[kwarg] is not None:
									new = value[kwarg][nrow%len(value[kwarg])][ncol%len(value[kwarg][nrow%len(value[kwarg])])]
								else:
									new = None
								if new is not None:
									new = '{%s}_{%s%s}'%(
											new.replace('$',''),
											str(nrow) if ((isinstance(value[kwarg],str)) or (len(value[kwarg]) < nrows)) and (nrows > 1) else '',
											str(ncol) if ((isinstance(value[kwarg],str)) or (len(value[kwarg][nrow%len(value[kwarg])]) < ncols)) and (ncols > 1) else ''
											)
								value[kwarg] = new
							else:
								for i in range(length):
									if isinstance(value[i][kwarg],str):
										new = value[i][kwarg]
									elif all(isinstance(v,str) for v in value[i][kwarg]):
										new = value[i][kwarg][index%len(value[i][kwarg])]
									elif value[i][kwarg] is not None:
										new = value[i][kwarg][nrow%len(value[i][kwarg])][ncol%len(value[i][kwarg][nrow%len(value[i][kwarg])])]
									else:
										new = None
									if new is not None:
										new = '{%s}_{%s%s}'%(
											new.replace('$',''),
											str(nrow) if ((isinstance(value[i][kwarg],str)) or (len(value[i][kwarg]) < nrows)) and (nrows > 1) else '',
											str(ncol) if ((isinstance(value[i][kwarg],str)) or (len(value[i][kwarg][ncol%len(value[i][kwarg])]) < ncols)) and (ncols > 1) else ''
											)
									value[i][kwarg] = new

						elif setting in ['ax'] and attr in ['errorbar','fill_between'] and kwarg in ['%serr'%(axis) for axis in axes]:

							axis = kwarg.replace('err','')

							subattr = 'set_%sscale'%(axis)
							subkwarg = 'value'
							
							do = settings[instance][subinstance][setting].get(subattr,{}).get(subkwarg) in ['log']

							if not do:
								continue

							subsetting = setting
							subattr = attr
							subkwarg = '%s'%(axis)
							subvalue = settings[instance][subinstance][subsetting][subattr]

							if not multiple:
								if value[kwarg] is not None:
									# new = value[kwarg]
									new = [
										subvalue[subkwarg]*(1-(subvalue[subkwarg]/(subvalue[subkwarg]+value[kwarg]))),
										value[kwarg]
										]
									# new = [
									# 	-subvalue[subkwarg]*(1/value[kwarg]-1),
									# 	value[kwarg]
									# 	subvalue[subkwarg]*(value[kwarg]-1)
									# 	]
									value[kwarg] = new
							else:
								for i in range(length):
									if value[i][kwarg] is not None:
										# new = value[i][kwarg]
										new = [
											subvalue[i][subkwarg]*(1-(subvalue[i][subkwarg]/(subvalue[i][subkwarg]+value[i][kwarg]))),
											value[i][kwarg]
											]
										# new = [
										# 	-subvalue[i][kwarg]*(1/value[i][kwarg]-1),
										# 	subvalue[i][subkwarg]*(value[i][kwarg]-1)
										# 	]
										value[i][kwarg] = new



		if hyperparameters.get('plot'):
			print('Plotting: ',instance)
			fig[instance],ax[instance] = plot(fig=fig[instance],ax=ax[instance],settings=settings[instance])



	# Perform farmed out processes
	for process in hyperparameters.get('process',[]):
		process = load(process)
		path = cwd
		process(path)

	return fig,ax

def main(*args,**kwargs):

	process(*args,**kwargs)

	return

if __name__ == '__main__':
	arguments = {
		'--data':{
			'help':'Process data files',
			'type':str,
			'default':[],
			'nargs':'*'
		},
		'--settings':{
			'help':'Process plot settings',
			'type':str,
			'default':[],
			'nargs':'*'
		},
		'--hyperparameters':{
			'help':'Process process settings',
			'type':str,
			'default':[],
			'nargs':'*'
		},
		'--cwd':{
			'help':'Process cwd',
			'type':str,
			'default':None,
			'nargs':'?',
		},						
	}

	wrappers = {
		'cwd':lambda kwarg,wrappers,kwargs: split(kwargs['data'][-1],directory=True).replace('/**','').replace('**','') if kwargs.get(kwarg) is None else kwargs.get(kwarg)
	}

	args = argparser(arguments,wrappers)

	main(*args,**args)
