#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,copy
import numpy as np
import scipy as sp
import scipy.special
import pandas as pd
from natsort import natsorted
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

from src.utils import array,product,expand_dims,is_number,to_number,to_key_value
from src.utils import nan
from src.dictionary import leaves,branches
from src.io import setup,load,dump,join,split
from src.plot import plot

scalars = (int,np.integer,float,np.float,str)
nulls = ('',None)

# Texify strings
def Texify(string,texify={},usetex=True):
	strings = {
		**texify,
	}
	if not isinstance(string,str):
		string = str(string)

	default = '\n'.join(['$%s$'%(strings.get(substring,substring).replace('$','')) for substring in string.split('\n')])

	string = strings.get(string,default)

	if not usetex or len(string) == 0:
		string = string.replace('$','')

	return string


# Normalize data
def norm(a,axis=None,ord=2):
	out = np.linalg.norm(a,axis=axis,ord=ord)
	return out

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
			value = values[key][slices]
		else:
			parser = lambda value: (value)
			value = [parser(value)]
	else:
		value = [value]

	boolean = datum in value
	
	return boolean


def include(name,keys,values,data):
	'''
	Include data if conditions on key and value are True
	Args:
		name (str): Dataset name
		keys (dict[str,str]): Keys of conditions	
		values (dict[str,object,iterable[object]]): Data of keys and iterable of values to compare
		data (dict[str,dict[str:object]]): Data of datasets of keys and values to compare
	Returns:
		boolean (bool): Accept inclusion of dataset
	'''

	def exception(name,key,keys,values,data):
		boolean = True
		if key in keys:
			if keys[key] is None:
				value = [values[key]]
			elif keys[key].startswith('@') and keys[key].endswith('@'):
				parser = lambda value: (str(value) if len(value)>0 else 0)
				value = keys[key].replace('@','').split(',')
				value = [parser(v) for v in value]
				value = [data[name][v] for v in value]
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

	return keys

def loader(kwargs):
	'''
	Load data
	Args:
		kwargs (dict[str,object]): Data to load
	'''

	returns = {}

	returns['multiple'] = False

	for kwarg in kwargs:
		if isinstance(kwargs[kwarg],str):
			kwargs[kwarg] = [kwargs[kwarg]]		
		if not isinstance(kwargs[kwarg],dict):
			# paths = ['data/data.hdf5']
			paths = set(kwargs[kwarg])
			default = {}
			kwargs[kwarg] = {}
			returns['multiple'] |= len(paths)>1
			for path in paths:
				print('Loading:',path)
				directory,ext = split(path,directory=-1,ext=True)
				kwargs[kwarg].update(load(path,default=default))
			# print('Dumping',kwarg,directory,ext)
			# dump(kwargs[kwarg]['value'],join(directory,kwarg,ext=ext))
		else:
			kwargs[kwarg] = kwargs[kwarg]
			returns['multiple'] |= False
	
	return returns


def process(data,settings,hyperparameters,fig=None,ax=None):
	'''
	Process data
	Args:
		data (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of data to process
		settings (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of plot settings
		hyperparameters (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of process settings
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

	- For plotting, we plot each combination for 'label' attributes and values variables data on the same plot, these being the labels.
	  Subplots are plotted from iterating over over the 1...ndim-2 axis of the variables data, and plotting the 0 and ndim-1 axis for each 'label' set
	  If the 'x' property is None, also iterate over the 0 (# of permutations of sort) axis variables data, and plot the ndim-1 axis for each 'label' 
	'''


	# Setup kwargs
	kwargs = ['data','settings','hyperparameters']
	kwargs = {kwarg: value for kwarg,value in zip(kwargs,[data,settings,hyperparameters])}

	returns = loader(kwargs)

	data,settings,hyperparameters = (kwargs[kwarg] for kwarg in kwargs)


	# Get dataset names of data
	names = list(natsorted(set(name for name in data),key=lambda name:name))

	# Get attributes of data
	attributes = list(set(attr for name in names 
		for attr in data[name]
		))

	subattributes = [attr
		for attr in attributes
		if ((attr in hyperparameters.get('sort',attributes)) and
	    (attr not in hyperparameters.get('nullsort',[])))
	    ]


	# Get unique scalar attributes
	unique = {attr: tuple((*sorted(set(np.asscalar(data[name][attr])
					for name in names 
					if ((attr in data[name]) and 
						((data[name][attr].size == 1) and isinstance(np.asscalar(data[name][attr]),scalars))
						)
					)),None))
			for attr in attributes
			}
	unique = {attr: unique[attr] for attr in unique if len(unique[attr])>0}	

	# Get attributes to sort on and attributes not to sort on if not existent in plot properties x,y,label
	sort = {attr: tuple((*sorted(set(np.asscalar(data[name][attr])
					for name in names 
					if ((attr in data[name]) and 
						((data[name][attr].size == 1) and isinstance(np.asscalar(data[name][attr]),scalars))
						)
					)),None))
			for attr in subattributes
			}
	sort = {attr: sort[attr] for attr in sort if len(sort[attr])>0}


	# Get combinations of attributes (including None) to sort on and attributes not to sort on if not existent in plot properties x,y,label
	allowed = list(natsorted(set((tuple((np.asscalar(data[name][attr])
				for attr in subattributes))
				for name in names
				))))
	allowed = [
				*allowed,
				*sorted(set([(*value[:i],None,*value[i+1:]) for value in allowed for i in range(len(value))]),
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

	# Get paths
	path,file,directory,ext,delimiter = {},{},{},{},{}
	for attr in hyperparameters.get('path',{}):
		delimiter[attr] = hyperparameters.get('delimiter','.')
		directory[attr],file[attr],ext[attr] = split(
			hyperparameters.get('path',{}).get(attr),
			directory=True if (not returns['multiple']) else -1,file=True,ext=True)
		path[attr] = join(directory[attr],file[attr],ext=ext[attr])

	# Get plot fig and axes
	if fig is None:
		fig = {}
	if ax is None:
		ax = {}

	# Get plot variables setting
	plotting = hyperparameters.get('plotting',{})
	for attr in list(plotting):
		for subattr in list(plotting[attr]):
			if subattr in nulls:
				for null in nulls:
					plotting[attr][null] = plotting[attr][subattr]
		if attr in nulls:
			for null in nulls:
				plotting[null] = plotting[attr]

	nullplotting = hyperparameters.get('nullplotting',{})
	for instance in list(nullplotting):
		if instance not in settings:
			nullplotting.pop(instance)
			continue
		if nullplotting[instance] is None:
			nullplotting[instance] = list(settings[instance])
		
		for subinstance in nullplotting[instance]:
			if subinstance in settings[instance]:
				settings[instance].pop(subinstance)
		if len(settings[instance]) == 0:
			settings.pop(instance)

	# Get texify
	texify = lambda string: Texify(string,hyperparameters.get('texify',{}),usetex=hyperparameters.get('usetex',True))


	# Get plot properties and statistics from settings
	axes = ['x','y']
	properties = [*['%s'%(ax) for ax in axes],'label']
	statistics = [*['%s'%(ax) for ax in axes],*['%serr'%(ax) for ax in axes]]
	statistics = {
		kwarg: {
			**{kwarg:{
				'property':kwarg.replace('',''),
				'statistic':{
					None: lambda key,data,variables=None,dtype=None: np.nanmean(data,axis=0).astype(dtype),
					'fit': lambda key,data,variables=None,dtype=None: np.nanmean(data,axis=0).astype(dtype),
					}
				} 
				for kwarg in ['%s'%(ax) for ax in axes]},
			**{kwarg:{
				'property':kwarg.replace('err',''),
				'statistic':{			
					None: lambda key,data,variables=None,dtype=None: np.nanstd(data,axis=0).astype(dtype),
					'fit': lambda key,data,variables=None,dtype=None: np.nanstd(data,axis=0).astype(dtype),
					}
				}	 
				for kwarg in ['%serr'%(ax) for ax in axes]},
			}[kwarg]
		for kwarg in statistics 			 	 			 	 
		}

	# Get keys of properties of the form ({prop:attr} or {prop:{'key':(attr,),'value:(values,)}})
	keys = find(settings,properties)
	labels = [{prop: dict(zip(key[prop]['key'],key[prop]['value'])) for prop in key} for key in keys]

	# Load data
	if hyperparameters.get('load'):
		attr = 'process'
		kwargs = {
			'conversion':lambda name: (
				to_number(name) if '____' not in name else 
				tuple((to_number(i) for i in name.split('____')[1:])))
			}

		variables = load(path[attr],**kwargs)

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

		# Get combinations of key attributes and permutations of shared attributes for combination across data

		variables = {}
		combinations = {}	
		permutations = {}

		# Get variables data and statistics from keys
		for occurrence,key in enumerate(keys):
			if occurrence  < 0:
				continue
			variables[occurrence] = {}

			# combinations[occurrence] = permute(key['label']['key'],key['label']['value'],list(sort),data)
			combinations[occurrence] = 	[
				[val for val in sort.get(attr,[]) 
				if check(attr,labels[occurrence]['label'][attr],val,labels,unique,data)]
				for attr in labels[occurrence]['label']
				]

			permutations[occurrence] = {}

			print('key',key,combinations[occurrence])

			# for combination in combinations[occurrence]:
			for combination in itertools.product(*combinations[occurrence]):
				variables[occurrence][combination] = {}
				label = {prop: dict(zip(
					labels[occurrence][prop],
					combination if prop in ['label'] else [value for value in labels[occurrence][prop]])) 
				for prop in labels[occurrence]}
				
				# permutations[occurrence][combination] = combinations[occurrence][combination]
				# permutations[occurrence][combination] = [
				# 	[val for val in sort[attr] 
				# 	if (((attr not in label['label']) and (val is not None)) or ((attr in label['label']) and (val == label['label'][attr])))]
				# 	for attr in sort
					# ]

				permutations[occurrence][combination] = [
					permutation for permutation in allowed if all(
						(((attr not in label['label']) and (val is not None)) or ((attr in label['label']) and (val == label['label'][attr])))
						for attr,val in zip(subattributes,permutation)
						)
					]

				# included = [name for name in names 
				# 	if all(data[name][attr] == values[attr]
				# 	for attr in values)]
				allincluded = [name for name in names 
					if include(name,labels[occurrence]['label'],label['label'],data)] 
				
				if len(allincluded) == 0:
					variables[occurrence].pop(combination);
					# print('continue --',combination)
					continue

				print('combination',label,combination)
				print()
				print(allincluded)
				print()

				for permutation in permutations[occurrence][combination]:
				# for permutation in itertools.product(*permutations[occurrence][combination]):
					variables[occurrence][combination][permutation] = {}
					values = dict(zip(subattributes,permutation))

					# included = [name for name in names 
					# 	if all(data[name][attr] == values[attr]
					# 	for attr in values)]

					included = [name for name in allincluded 
						if include(name,
								{attr: labels[occurrence][prop][attr] 
								for prop in labels[occurrence] 
								for attr in labels[occurrence][prop]},
								values,data)
						]
					
					if len(included) == 0:
						variables[occurrence][combination].pop(permutation);
						# print('continue',permutation)
						continue

					print('permutation',label,values)
					print(included)					
					print()
					continue

					for kwarg in statistics:

						variables[occurrence][combination][permutation][kwarg] = {}

						prop = statistics[kwarg]['property']
						isnull = key[prop]['key'] in nulls
						if isnull:
							prop = 'y'					
							dtype = int
						else:
							prop = prop
							dtype = data[name][key[prop]['key']].dtype
					
						variables[occurrence][combination][permutation][kwarg] = {}


						# Insert data into variables (with nan padding)
						for stat in statistics[kwarg]['statistic']:
							if (((plotting.get(key['y']['key'],{}).get(key['x']['key'],{}).get('plot') is not None) and
								 (stat not in [None,*plotting.get(key['y']['key'],{}).get(key['x']['key'],{}).get('plot',[])])) or
								((plotting.get(key['y']['key'],{}).get(key['x']['key']) is None) and 
								 (stat not in [None]))):
								continue

							variables[occurrence][combination][permutation][kwarg][stat] = np.nan*np.ones((len(included),*shape[key[prop]['key']]))

							# if kwarg in ['y']:
							# 	print(kwarg,key,included)

							for index,name in enumerate(included):

								value = expand_dims(np.arange(data[name][key[prop]['key']].shape[-1]),range(0,ndim[key[prop]['key']]-1)) if isnull else data[name][key[prop]['key']]
								slices = (index,*(slice(data[name][key[prop]['key']].shape[axis]) for axis in range(data[name][key[prop]['key']].ndim)))
								variables[occurrence][combination][permutation][kwarg][stat][slices] = value

							# Get statistics
							variables[occurrence][combination][permutation][kwarg][stat] = statistics[kwarg]['statistic'][stat](
								key,variables[occurrence][combination][permutation][kwarg][stat],
								variables=variables[occurrence][combination][permutation],dtype=dtype)

				print()
				continue
				# print('merging')
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

	return fig,ax

	# Dump data
	if hyperparameters.get('dump') or 1:
		attr = 'process'
		kwargs = {
			'conversion':lambda name: (
				str(name) if not isinstance(name,tuple) else 
				'____'+'____'.join((str(i) for i in name)))
		}
		dump(variables,path[attr],**kwargs)
	

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
	special = {'ax':['plot','errorbar','fill_between']}

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
							occurrence = keys.index(key)

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
	# plotting = {'y':{'x':{'axis':{attr:[[axis for ncols],[axis for nrows],[axis for labels][axis for plot]]}}}}
	for instance in list(settings):
		print('Plotting',instance)
		for subinstance in list(settings[instance]):
			subupdated.clear()
			for setting in special:
				for attr in special[setting]:
					if attr in settings[instance][subinstance][setting]:

						for i in range(len(settings[instance][subinstance][setting][attr])):							
							key = find(settings[instance][subinstance][setting][attr][i],properties)[0]
							occurrence = keys.index(key)

							if occurrence not in variables:
								continue

							subndim = min(variables[occurrence][combination][kwarg][stat].ndim
								for combination in variables[occurrence]
								for kwarg in variables[occurrence][combination]
								for stat in variables[occurrence][combination][kwarg]
								)

							subaxis = plotting.get(key['y']['key'],{}).get(key['x']['key'],{}).get('axis')
							
							if subaxis is None:
								subaxis = [[],[],[],[axis for axis in range(subndim)]]
							else:
								subaxis = [[axis] if isinstance(axis,int) else axis for axis in subaxis]

							subaxis = [*([ax for ax in axis] for axis in subaxis[:-1]),
										[ax for ax in range(subndim) 
										 if (ax in subaxis[-1] or (ax == (subndim-1) and -1 in subaxis[-1])) or 
										 ax not in [ax for axis in subaxis[:-1] for ax in axis]]]


							if occurrence not in updated:
								
								updated.append(occurrence)

								for combination in variables[occurrence]:
									for kwarg in variables[occurrence][combination]:
										for stat in variables[occurrence][combination][kwarg]:
											transpose = [ax for axis in subaxis for ax in axis]											
											reshape = [
												max(1,int(product(
												[variables[occurrence][combination][kwarg][stat].shape[ax]
												for ax in axis])))
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
									occurrence = keys.index(key)
									size = max(variables[occurrence][combination][kwarg][stat].shape[dim-1+1]
												for combination in variables[occurrence]
												for kwarg in variables[occurrence][combination]
												for stat in variables[occurrence][combination][kwarg])												
									for enum,(combination,j) in enumerate(itertools.product(variables[occurrence],range(size))):
										subsize = max(variables[occurrence][combination][kwarg][stat].shape[dim-1+1]
													for kwarg in variables[occurrence][combination]
													for stat in variables[occurrence][combination][kwarg])
										if j >= subsize:
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

												if kwarg in ['%serr'%(ax) for ax in axes] and norm(value) == 0:
													value = None

												subsettings[kwarg] = value

											settings[instance][(subinstance,*position)][setting][attr][
												(combination,j,occurrence,stat)] = subsettings

		for samplelayout in layouts:
			for subinstance in layouts[samplelayout]:
				settings[instance].pop(subinstance)

	for instance in settings:
		if instance not in fig:
			fig[instance] = None
		if instance not in ax:
			ax[instance] = None

	# Set plot settings

	for instance in settings:
		for subinstance in settings[instance]:
			for setting in settings[instance][subinstance]:
				
				for attr in special.get(setting,{}):

					if attr not in settings[instance][subinstance][setting]:
						continue 

					subcombinations,subj,suboccurences,substats = zip(*settings[instance][subinstance][setting][attr])
					for i,subsubinstance in enumerate(settings[instance][subinstance][setting][attr]):

						combination,j,occurrence,stat = subsubinstance
						key = keys[occurrence]
						combination = dict(zip(key['label']['key'],combination))

						kwargs = ['color','ecolor']
						for kwarg in kwargs:
							if kwarg not in settings[instance][subinstance][setting][attr][subsubinstance]:
								continue
							value = getattr(plt.cm,settings[instance][subinstance][setting][attr][subsubinstance][kwarg])(
								(subcombinations.index(tuple((combination[k] for k in combination))))/len(subcombinations))

							settings[instance][subinstance][setting][attr][subsubinstance][kwarg] = value

						kwargs = ['label']
						for kwarg in kwargs:
							if kwarg not in settings[instance][subinstance][setting][attr][subsubinstance]:
								continue

							if stat in [None]:
								value = [k for i,k in enumerate(combination) if len(set(combinations[occurrence][i])) > 1]
								value = ',~'.join([texify(str(combination[k])) for k in value])
							else:
								value = None
							settings[instance][subinstance][setting][attr][subsubinstance][kwarg] = value
						
						kwargs = ['linestyle']
						for kwarg in kwargs:
							if kwarg not in settings[instance][subinstance][setting][attr][subsubinstance]:
								continue

							if stat in [None]:
								value = '-'
							elif stat in ['fit']:
								value = '-'
							else:
								value = None

							settings[instance][subinstance][setting][attr][subsubinstance][kwarg] = value


						subattr = 'legend'
						kwargs = ['set_title']
						if settings[instance][subinstance][setting].get(subattr) is None:
							continue
						for kwarg in kwargs:
							value = [
								[(k,combination[k]) for i,k in enumerate(combination) if len(set(combinations[occurrence][i])) == 1],
								[(k,) for i,k in enumerate(combination) if len(set(combinations[occurrence][i])) > 1],
								]
							value = ['~,'.join([': '.join([texify(l) for l in k]) for k in v]) for v in value]
							value = [v for v in value if len(v)>0]
							value = '\n'.join(value)

							settings[instance][subinstance][setting][subattr][kwarg] = value


					settings[instance][subinstance][setting][attr] = [
						settings[instance][subinstance][setting][attr][subsubinstance]
						for subsubinstance in settings[instance][subinstance][setting][attr]
						]



			# Set custom plot settings
			attrs = {
				'fig':{'savefig':['fname']},
				'ax':{'set_ylabel':['ylabel']},
				}
			for setting in attrs:
				if setting not in settings[instance][subinstance]:
					settings[instance][subinstance][setting] = {}
				for attr in attrs[setting]:
					if attr not in settings[instance][subinstance][setting]:
						settings[instance][subinstance][setting][attr] = {}
					elif settings[instance][subinstance][setting][attr] is None:
						continue
					for kwarg in attrs[setting][attr]:
						if kwarg not in settings[instance][subinstance][setting][attr]:
							settings[instance][subinstance][setting][attr][kwarg] = {}

						if setting in ['fig'] and attr in ['savefig'] and kwarg in ['fname']:
							value = 'plot'
							value = join(directory[value],
									delimiter[value].join([
										*file[value].split(delimiter[value])[:],
										instance,
										]),
									ext=ext[value])

							settings[instance][subinstance][setting][attr][kwarg] = value

						elif setting in ['ax'] and attr in ['set_ylabel'] and kwarg in ['ylabel']:

							index = settings[instance][subinstance]['style']['layout']['index']
							nrows = settings[instance][subinstance]['style']['layout']['nrows']
							ncols = settings[instance][subinstance]['style']['layout']['ncols']

							if ((nrows is None) or (nrows == 1)) and ((ncols is None) or (ncols == 1)):
								continue

							index = index - 1	
							nrow = (index - index%ncols)//ncols
							ncol = index%ncols
							if nrows == 1:
								nrow = ''
							elif ncols == 1:
								ncol = ''
							nrow = str(nrow)
							ncol = str(ncol)

							value = '{%s}_{%s%s}'%(
									settings[instance][subinstance][setting][attr][kwarg].replace('$',''),
									nrow,
									ncol
									)

							settings[instance][subinstance][setting][attr][kwarg] = value



	verbose = 0


	for instance in settings:

		if verbose:
			for subinstance in settings[instance]:
				print(subinstance)
				for setting in settings[instance][subinstance]:
					print(setting)
					for kwarg in settings[instance][subinstance][setting]:
						print(kwarg)
						if isinstance(settings[instance][subinstance][setting][kwarg],list):
							for value in settings[instance][subinstance][setting][kwarg]:
								print(value)
						else:
							value = settings[instance][subinstance][setting][kwarg]
							print(value)
						print()
					print()
				print()


		fig[instance],ax[instance] = plot(fig=fig[instance],ax=ax[instance],settings=settings[instance])

	return fig,ax

def main(args):

	Nargs = 3

	nargs = len(args)
	
	assert nargs < Nargs, 'Incorrect number of arguments passed'

	data,settings,hyperparameters = args[:Nargs]
	
	process(data,settings,hyperparameters)

	return

if __name__ == '__main__':
	main(sys.argv[1:])