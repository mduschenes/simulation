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
from src.dictionary import leaves
from src.io import setup,load,dump,join,split
from src.plot import plot

scalars = (int,np.integer,float,np.float)


def texify(string,usetex=True):
	strings = {
	}
	if not isinstance(string,str):
		string = str(string)

	default = r'$%s$'%(string.replace('$',''))

	string = strings.get(string,default)

	if not usetex:
		string = string.replace('$','')

	return string


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
	keys = (leaves(dictionary,prop,types=(dict,list),returns='value') for prop in properties)
	keys = map(lambda key: dict(zip(
		properties,(*key[:2],tuple((dict(zip(['key','value'],to_key_value(key[2],delimiter='='))),))
		if key[2] is None or isinstance(key[2],str) 
		else tuple(dict(zip(['key','value'],to_key_value(j,delimiter='='))) for j in key[2])))),
		zip(*keys))
	keys = [{prop:key[prop] if prop not in ['label'] else 
			{label:tuple(value[label] for value in key[prop]) for label in set(
				label for value in key[prop] for label in value)}
			 for prop in key} for key in keys]
	return keys

def process(data,settings,hyperparameters):
	'''
	Process data
	Args:
		data (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of data to process
		settings (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of plot settings
		hyperparameters (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of process settings
	
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

	if isinstance(data,str):
		data = [data]
	if not isinstance(data,dict):
		paths = set(data)
		data = {}
		for path in paths:
			default = {}
			data.update(load(path,default=default))

	if isinstance(settings,str):
		settings = [settings]
	if not isinstance(settings,dict):
		paths = set(settings)
		settings = {}
		for path in paths:
			default = {}
			settings.update(load(path,default=default))

	if isinstance(hyperparameters,str):
		hyperparameters = [hyperparameters]
	if not isinstance(hyperparameters,dict):
		paths = set(hyperparameters)
		hyperparameters = {}
		for path in paths:
			default = {}
			hyperparameters.update(load(path,default=default))

	
	# Get dataset names of data
	names = list(sorted(set(name for name in data),key=lambda name:name))

	# Get attributes of data
	attrs = list(set(attr for name in names for attr in data[name]))

	# Get attributes to sort on and attributes not to sort on if not existent in plot properties x,y,label
	sort = {attr: tuple(sorted(set(data[name][attr] 
					for name in names 
					if (
					(attr in data[name]) and 
					(isinstance(data[name][attr],scalars)) and 
					(attr in hyperparameters.get('sort',attrs)) and
					(attr not in hyperparameters.get('nullsort',[]))
					))))
			for attr in attrs}
	sort = {attr: sort[attr] for attr in sort if len(sort[attr])>0}

	# Get data as arrays, with at least 1 leading dimension
	for name in names:
		for attr in data[name]:
			data[name][attr] = np.array(data[name][attr])
			data[name][attr] = data[name][attr].reshape(*[1]*(max(0,1-data[name][attr].ndim)),*data[name][attr].shape)
	
	# Get number of dimensions and maximum shape of data attributes
	ndim = {attr: min(data[name][attr].ndim for name in names) for attr in attrs}
	shape = {attr: tuple(map(max,zip(*(data[name][attr].shape for name in names)))) for attr in attrs}

	# Get hyperparameters
	file,directory,ext = {},{},{}
	for attr in hyperparameters.get('path',{}):
		file[attr],directory[attr],ext[attr] = split(hyperparameters.get('path',{}).get(attr),directory=True,file=True,ext=True)

	# Get plot properties and statistics from settings
	properties = ['x','y','label']
	properties = {
		prop:{
			'%s'%(prop): lambda key,data: np.mean(data,axis=0).astype(data.dtype),
			'%serr'%(prop): lambda key,data: np.std(data,axis=0).astype(data.dtype),
			# '%sargsort'%(prop): lambda key,data: np.argsort(data,axis=0),
			}
			if prop not in ['label'] else {} 
		for prop in properties}
	statistics = list(set(stat for prop in properties for stat in properties[prop]))

	# Get keys of properties of the form ({prop:attr} or {prop:{'key':(attr,),'value:(values,)}})
	keys = find(settings,properties)

	# Get variables data and statistics from keys
	variables = {}
	for occurrence,key in enumerate(keys):
		variables[occurrence] = {}
		combinations = (
			[val for val in sort.get(attr,[]) if include(attr,value,val,sort)]
			for attr,value in zip(key['label']['key'],key['label']['value'])
			if all(attr is not None for attr in key['label']['key'])
			)
		print('****************************************************************************************************')
		print(key)		
		for combination in itertools.product(*combinations):
			variables[occurrence][combination] = {}
			values = dict(zip(key['label']['key'],combination))
			permutations = (
				[val for val in sort[attr] if ((attr not in values) or include(attr,values[attr],val,sort))]
				for attr in sort
				)
			included = [name for name in names if all(include(attr,values[attr],data[name][attr],data[name]) for attr in values)]
			length = len(included)
			if length == 0:
				variables[occurrence].pop(combination);
				continue

			print('------------------------')
			print(values)
			for permutation in itertools.product(*permutations):
				variables[occurrence][combination][permutation] = {}
				values = dict(zip(sort,permutation))
				included = [name for name in names if all(include(attr,values[attr],data[name][attr],data[name]) for attr in values)]
				length = len(included)
				if length == 0:
					variables[occurrence][combination].pop(permutation);
					continue

				for prop in properties:
					reference = prop
					isNone = key[prop] is None
					if isNone:
						prop = 'y'					
						dtype = int
					else:
						dtype = None
					for stat in properties[reference]:
						variables[occurrence][combination][permutation][stat] = np.zeros((length,*shape[key[prop]]),dtype=dtype)
						for index,name in enumerate(included):
							if isNone:
								value = expand_dims(np.arange(data[name][key[prop]].shape[-1]),range(0,ndim[key[prop]]-1))
							else:
								value = data[name][key[prop]]

							slices = (index,*(slice(data[name][key[prop]].shape[axis]) for axis in range(data[name][key[prop]].ndim)))

							variables[occurrence][combination][permutation][stat][slices] = value

							slices = (index,*(slice(data[name][key[prop]].shape[axis],None) for axis in range(data[name][key[prop]].ndim)))
							indices = (*(-1 for axis in range(data[name][key[prop]].ndim)),)
							variables[occurrence][combination][permutation][stat][slices] = value[indices]
							
							print('***** stats %s *****'%(stat))
							print(variables[occurrence][combination][permutation][stat])

						variables[occurrence][combination][permutation][stat] = properties[reference][stat](
							key,variables[occurrence][combination][permutation][stat])
				print('----')	
				print(values)
				print(included)

			variables[occurrence][combination] = {
				stat: np.array([variables[occurrence][combination][permutation][stat] for permutation in variables[occurrence][combination]])
				for stat in statistics
			}


			print('---value---')
			for stat in variables[occurrence][combination]:
				print(stat,variables[occurrence][combination][stat].shape)
				if stat == 'x' and key['y'] == 'parameters':
					print(variables[occurrence][combination][stat][0,0,0])
					print(variables[occurrence][combination][stat][-1,-1,-1])


			print()
		print()
		print()	

	# Plot data
	
	# Default setting objects for each settings instance
	defaults = {setting:{} for setting in ['ax','fig','style']}
	
	# Checks to plot variables data keys (to check for updates to setting depending on variables data shape)
	checks = {'ax':['plot','errorbar']}

	# Updates to settings depending on variables data keys and shape
	updates = {'ax':['plot','errorbar'],'style':{'layout':['nrows','ncols','index']}}

	# Track updated keys
	updated = []

	# Set default plot settings
	for instance in settings:	
		if any(setting in settings[instance] for setting in defaults):
			subinstance = None
			settings[instance] = {subinstance: settings[instance]}


	# Get layout of plot instances		
	dim = 2
	layout = {
		instance:{
			'nrows': max(settings[instance][subinstance]['style'].get('layout',{}).get('nrows',len(settings[instance])) 
				for subinstance in settings[instance]),
			'ncols': max(settings[instance][subinstance]['style'].get('layout',{}).get('ncols',1) 
				for subinstance in settings[instance]),
			'index':{subinstance: settings[instance][subinstance]['style'].get('layout',{}).get('index',1)
				for subinstance in settings[instance]},
			}
		for instance in settings
		}



	print('-------')
	for instance in layout:
		print(instance)
		for prop in layout[instance]:
			if isinstance(layout[instance][prop],dict):
				for subinstance in layout[instance][prop]:
					print(prop,subinstance,layout[instance][prop][subinstance])
			else:
				print(prop,layout[instance][prop])
			print()
		print()

	# Update plot instance settings with variables data dependent layout
	for instance in settings:
		
		print('---------------------',list(layout[instance]['index']))

		# Get update to layout based on (reshaped) variables data shape and 
		# reshaping of variables data based on 
		# hyperparameters['axis'] = {attr:[[axis for ncols],[axis for nrows],[axis for plot]]}

		for subinstance in list(settings[instance]):
			for setting in checks:
				for check in checks[setting]:
					if check in settings[instance][subinstance][setting]:
						if isinstance(settings[instance][subinstance][setting][check],dict):
							settings[instance][subinstance][setting][check] = [settings[instance][subinstance][setting][check]]
						print(subinstance)
						size = [1 for axis in range(dim)]
						for i in range(len(settings[instance][subinstance][setting][check])):							
							key = find(settings[instance][subinstance][setting][check][i],properties)[0]
							occurrence = keys.index(key)

							subndim = min(variables[occurrence][combination][stat].ndim
								for combination in variables[occurrence]
								for stat in variables[occurrence][combination])

							subaxis = hyperparameters.get('axis',{}).get(key['y'])
							if subaxis is None:
								subaxis = [[],[],[axis for axis in range(subndim)]]
							else:
								subaxis = [[axis] if isinstance(axis,int) else axis for axis in subaxis]


							if key not in updated:
								
								updated.append(key)

								for combination in variables[occurrence]:
									for stat in variables[occurrence][combination]:
										variables[occurrence][combination][stat] = variables[occurrence][combination][stat].transpose(
											[ax for axis in subaxis for ax in axis]).reshape(
											[max(1,int(product([variables[occurrence][combination][stat].shape[ax]
												for ax in axis])))
												for axis in subaxis]
											)

							size = [max(size[axis],max(variables[occurrence][combination][stat].shape[axis]
											for combination in variables[occurrence]
											for stat in variables[occurrence][combination]))
										for axis in range(dim)]	
							index = max(layout[instance]['index'][subinst] 
								for subinst in layout[instance]['index'] 
								if (subinst == subinstance) or (isinstance(subinst,tuple) and subinstance in subinst))+1

							print(size,index)

						layout[instance].update({
							**{prop: 
								layout[instance][prop] + size[axis] - 1
								for axis,prop in enumerate(['nrows','ncols'])
							},
							'index':{
								**{(subinstance,index + i - 1): index + i - 1
									for i,position in enumerate(
									itertools.product(*(range(size[axis]) for axis in range(dim))))},
								**{subinst: layout[instance]['index'][subinst] + (int(product(size)) if layout[instance]['index'][subinst] >= index else 0)
									for subinst in layout[instance]['index']
									if (subinst != subinstance) or (isinstance(subinst,tuple) and subinstance not in subinst)},
								}
							})

						break		


	print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
	for instance in layout:
		print(instance)
		for prop in layout[instance]:
			if isinstance(layout[instance][prop],dict):
				for subinstance in layout[instance][prop]:
					print(prop,subinstance,layout[instance][prop][subinstance])
			else:
				print(prop,layout[instance][prop])
			print()
		print()
	exit()


	# Set plot settings
	for instance in settings:
		# Set standard plot settings		
		for subinstance in settings[instance]:
			for setting in defaults:
				settings[instance][subinstance][setting] = settings[instance][subinstance].get(
					settings[instance][subinstance][setting],defaults[settings])

				if setting in ['fig']:
					for attr in ['savefig']:
						settings[instance][subinstance][setting][attr] = {
							**settings[instance][subinstance][setting].get(attr,{}),
							'fname':join(
								directory['plot'],
								'.'.join([file['plot'],key['x'],key['y'],*key['label']['key']]),
								ext=ext['plot']),
							}

				elif setting in ['ax']:
					for attr in ['errorbar']:												
						if (isinstance(settings[instance][subinstance][setting].get(attr),dict)):
							settings[instance][subinstance][setting][attr] = [settings[instance][subinstance][setting][attr]]	
						elif ((isinstance(settings[instance][subinstance][setting].get(attr),list)) and 
							  (len(settings[instance][subinstance][setting].get(attr))==1)):
							key = find(settings[instance][subinstance][setting].get(attr)[0],properties)[0] 
							occurrence = keys.index(key)
							size = len(itertools.product(*(range(subshape[key['y']][axis]) for axis in range(subndim[key['y']]))))
							settings[instance][subinstance][setting][attr] *= size
						for i in range(len(settings[instance][subinstance][setting][attr])):
							key = find(settings[instance][subinstance][setting].get(attr)[0],properties)[0] 
							occurrence = keys.index(key)
							settings[instance][subinstance][setting][attr][i] = {
								**settings[instance][subinstance][setting][attr][i],
								'x': variables[occurrence][combination]['xfunc'][i],
								'y': variables[occurrence][combination]['yfunc'][i],
								'xerr': variables[occurrence][combination]['xfuncerr'][i],
								'yerr': variables[occurrence][combination]['yfuncerr'][i],						
								'color':getattr(plt.cm,settings[instance][subinstance][setting].get(attr,{}).get('color','viridis'))((len(variables[occurrence]) - 1 - i)/len(variables[occurrence])),
								'ecolor':getattr(plt.cm,settings[instance][subinstance][setting].get(attr,{}).get('ecolor','viridis'))((len(variables[occurrence]) - 1 - i)/len(variables[occurrence])),
								'label':dict(zip(key['label']['key'],combination))[key['label']['key'][0]],
							}

		# Set custom plot settings
		for subinstance in settings[instance]:
			for setting in defaults:
				pass

	exit()
	for instance in settings:
		plot(settings=settings[instance])

	return

def main(args):

	Nargs = 3

	nargs = len(args)
	
	assert nargs < Nargs, "Incorrect number of arguments passed"

	data,settings,hyperparameters = args[:Nargs]
	
	process(data,settings,hyperparameters)

	return

if __name__ == '__main__':
	main(sys.argv[1:])