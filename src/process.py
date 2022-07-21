#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,copy
import numpy as np
import scipy as sp
import scipy.special
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

scalars = (int,np.integer,float,np.float)
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


def include(key,value,datum,data):
	'''
	Include data if conditions on key and value are True
	Args:
		key (str): Reference key to check
		value (object): Reference value to check, allowed strings in ['$value,',#index,','%start,stop,step'] 
						for comma-delimited values, indices, or slices values
		datum (object): Data value to check
		data (dict[str,object,iterable[object]]): Data of keys and iterable of values to compare
	Returns:
		boolean (bool): Accept inclusion of dataset
	'''
	boolean = False
	if value is None:
		value = [datum]
	elif isinstance(value,str):			
		if value.startswith('$') and value.endswith('$'):
			parser = lambda value: (to_number(value) if len(value)>0 else 0)
			values = value.replace('$','').split(',')
			values = [parser(value) for value in values]
			value = [value for value in values]
		elif value.startswith('#') and value.endswith('#'):
			parser = lambda value: (int(value) if len(value)>0 else 0)
			indices = value.replace('#','').split(',')
			indices = [parser(index) for index in indices]
			value = [data[key][index] for index in indices]
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
			value = data[key][slices]
		else:
			parser = lambda value: (value)
			value = [parser(value)]
	else:
		value = [value]
	boolean = datum in value
	
	return boolean



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
		properties,(*key[:2],tuple((dict(zip(['key','value'],to_key_value(key[2],delimiter='='))),))
		if key[2] is None or isinstance(key[2],str) 
		else tuple(dict(zip(['key','value'],to_key_value(j,delimiter='='))) for j in key[2])))),
		keys)

	keys = [{prop:key[prop] if prop not in ['label'] else 
			{label:tuple(value[label] for value in key[prop]) for label in set(
				label for value in key[prop] for label in value)}
			 for prop in key} for key in keys]
	return keys

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
	kwargs = {kwarg: {'value':value,'size':None,} for kwarg,value in zip(kwargs,[data,settings,hyperparameters])}

	for kwarg in kwargs:
		if isinstance(kwargs[kwarg]['value'],str):
			kwargs[kwarg]['value'] = [kwargs[kwarg]['value']]
		if not isinstance(kwargs[kwarg]['value'],dict):
			paths = set(kwargs[kwarg]['value'])
			kwargs[kwarg]['value'] = {}
			kwargs[kwarg]['size'] = len(paths)
			for path in paths:
				default = {}
				kwargs[kwarg]['value'].update(load(path,default=default))
		else:
			kwargs[kwarg]['value'] = kwargs[kwarg]['value']
			kwargs[kwarg]['size'] = 1


	data,settings,hyperparameters = [kwargs[kwarg]['value'] for kwarg in kwargs]
	sizes = {kwarg: kwargs[kwarg]['size'] for kwarg in kwargs}
	multiple = sizes['hyperparameters'] > 1

	# Get dataset names of data
	names = list(sorted(set(name for name in data),key=lambda name:name))

	# Get attributes of data
	attributes = list(set(attr for name in names for attr in data[name]))

	# Get attributes to sort on and attributes not to sort on if not existent in plot properties x,y,label
	sort = {attr: tuple(sorted(set(data[name][attr] 
					for name in names 
					if (
					(attr in data[name]) and 
					(isinstance(data[name][attr],scalars)) and 
					(attr in hyperparameters.get('sort',attributes)) and
					(attr not in hyperparameters.get('nullsort',[]))
					))))
			for attr in attributes}
	sort = {attr: sort[attr] for attr in sort if len(sort[attr])>0}

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
			directory=True if not multiple else -1,file=True,ext=True)
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
			variables[occurrence] = {}
			combinations[occurrence] = [
				[val for val in sort.get(attr,[]) if include(attr,value,val,sort)]
				for attr,value in zip(key['label']['key'],key['label']['value'])
				if all(attr is not None for attr in key['label']['key'])
				]
			permutations[occurrence] = {}

			for combination in itertools.product(*combinations[occurrence]):
				variables[occurrence][combination] = {}
				values = dict(zip(key['label']['key'],combination))
				permutations[occurrence][combination] = [
					[val for val in sort[attr] if ((attr not in values) or include(attr,values[attr],val,sort))]
					for attr in sort
					]
				included = [name for name in names if all(include(attr,values[attr],data[name][attr],data[name]) for attr in values)]
				
				if len(included) == 0:
					variables[occurrence].pop(combination);
					continue

				for permutation in itertools.product(*permutations[occurrence][combination]):
					variables[occurrence][combination][permutation] = {}
					values = dict(zip(sort,permutation))
					included = [name for name in names if all(include(attr,values[attr],data[name][attr],data[name]) for attr in values)]
					
					if len(included) == 0:
						variables[occurrence][combination].pop(permutation);
						continue

					for kwarg in statistics:

						variables[occurrence][combination][permutation][kwarg] = {}

						prop = statistics[kwarg]['property']
						isnull = key[prop] in nulls
						if isnull:
							prop = 'y'					
							dtype = int
						else:
							prop = prop
							dtype = data[name][key[prop]].dtype
					
						variables[occurrence][combination][permutation][kwarg] = {}


						# Insert data into variables (with nan padding)
						for stat in statistics[kwarg]['statistic']:
							if (((plotting.get(key['y'],{}).get(key['x'],{}).get('plot') is not None) and
								 (stat not in [None,*plotting.get(key['y'],{}).get(key['x'],{}).get('plot',[])])) or
								((plotting.get(key['y'],{}).get(key['x']) is None) and 
								 (stat not in [None]))):
								continue

							variables[occurrence][combination][permutation][kwarg][stat] = np.nan*np.ones((len(included),*shape[key[prop]]))

							# if kwarg in ['y']:
							# 	print(kwarg,key,included)

							for index,name in enumerate(included):

								value = expand_dims(np.arange(data[name][key[prop]].shape[-1]),range(0,ndim[key[prop]]-1)) if isnull else data[name][key[prop]]
								slices = (index,*(slice(data[name][key[prop]].shape[axis]) for axis in range(data[name][key[prop]].ndim)))
								variables[occurrence][combination][permutation][kwarg][stat][slices] = value


								# if kwarg in ['y']:
								# 	print(name)
								# 	print(data[name][key[prop]])
								# 	print()

								# value = nan
								# slices = (index,*(slice(
								# 		data[name][key[prop]].shape[axis] if axis == (ndim[key[prop]]-1) else None,None) 
								# 	for axis in range(data[name][key[prop]].ndim)))
								# variables[occurrence][combination][permutation][kwarg][stat][slices] = value
								
							# Get statistics
							# if kwarg in ['y']:
							# 	print('---')
							# 	print(variables[occurrence][combination][permutation][kwarg][stat])
							# 	print('--- mean 0---')							
							# 	print(np.nanmean(variables[occurrence][combination][permutation][kwarg][stat],axis=0))

							variables[occurrence][combination][permutation][kwarg][stat] = statistics[kwarg]['statistic'][stat](
								key,variables[occurrence][combination][permutation][kwarg][stat],
								variables=variables[occurrence][combination][permutation],dtype=dtype)
							# if kwarg in ['y']:
							# 	print('---')
							# 	print(variables[occurrence][combination][permutation][kwarg][stat])							
							# 	print('---')							
							# 	print()
							# 	print()
							# 	print()

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

	# 	print('---value---')
	# 	for kwarg in statistics:
	# 		for stat in statistics[kwarg]['statistic']:
	# 			print(stat,variables[occurrence][combination][kwarg][stat].shape)
	# 			if stat == 'x' and key['y'] == 'parameters':
	# 				print(variables[occurrence][combination][kwarg][stat][0,0,0])
	# 				print(variables[occurrence][combination][kwarg][stat][-1,-1,-1])


		# 	print()
		# print()
		# print()	

	# Dump data
	if hyperparameters.get('dump'):
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
	special = {'ax':['plot','errorbar']}

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

							subaxis = plotting.get(key['y'],{}).get(key['x'],{}).get('axis')
							
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
								value = '--'
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