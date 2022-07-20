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
		directory[attr],file[attr],ext[attr] = split(hyperparameters.get('path',{}).get(attr),directory=True,file=True,ext=True)

	# Get plot properties and statistics from settings
	axes = ['x','y']
	properties = [*['%s'%(ax) for ax in axes],'label']
	statistics = [*['%s'%(ax) for ax in axes],*['%serr'%(ax) for ax in axes]]
	statistics = {
		kwarg: {
			**{kwarg:{
				'property':kwarg.replace('',''),
				'statistic':{
					'value': lambda key,data,dtype=None: np.nanmean(data,axis=0).astype(dtype),
					}
				} 
			 	for kwarg in ['%s'%(ax) for ax in axes]},
			**{kwarg:{
				'property':kwarg.replace('err',''),
				'statistic':{			
					'value': lambda key,data,dtype=None: np.nanstd(data,axis=0).astype(dtype),
					}
				}	 
			 	for kwarg in ['%serr'%(ax) for ax in axes]},
			}[kwarg]
		for kwarg in statistics 			 	 			 	 
		}

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

			for permutation in itertools.product(*permutations):
				variables[occurrence][combination][permutation] = {}
				values = dict(zip(sort,permutation))
				included = [name for name in names if all(include(attr,values[attr],data[name][attr],data[name]) for attr in values)]
				length = len(included)
				if length == 0:
					variables[occurrence][combination].pop(permutation);
					continue

				for kwarg in statistics:

					variables[occurrence][combination][permutation][kwarg] = {}

					prop = statistics[kwarg]['property']
					isNone = key[prop] is None
					if isNone:
						prop = 'y'					
						dtype = int
					else:
						dtype = None
				
					variables[occurrence][combination][permutation][kwarg] = {}

					# Insert data into variables (with nan padding)
					for stat in statistics[kwarg]['statistic']:
						variables[occurrence][combination][permutation][kwarg][stat] = np.zeros((length,*shape[key[prop]]))
						for index,name in enumerate(included):

							value = expand_dims(np.arange(data[name][key[prop]].shape[-1]),range(0,ndim[key[prop]]-1)) if isNone else data[name][key[prop]]
							slices = (index,*(slice(data[name][key[prop]].shape[axis]) for axis in range(data[name][key[prop]].ndim)))
							variables[occurrence][combination][permutation][kwarg][stat][slices] = value

							value = nan
							slices = (index,*(slice(
									data[name][key[prop]].shape[axis] if axis >= (ndim[key[prop]]-1) else None,None) 
								for axis in range(data[name][key[prop]].ndim)))
							variables[occurrence][combination][permutation][kwarg][stat][slices] = value
							
						# Get statistics
						variables[occurrence][combination][permutation][kwarg][stat] = statistics[kwarg]['statistic'][stat](
							key,variables[occurrence][combination][permutation][kwarg][stat],dtype=dtype)

			variables[occurrence][combination] = {
				kwarg:{
					stat: np.array([variables[occurrence][combination][permutation][kwarg][stat] 
									for permutation in variables[occurrence][combination]])
					for stat in statistics[kwarg]['statistic']
					}
				for kwarg in statistics}


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
	specials = {'ax':{'plot':['color'],'errorbar':['color','ecolor']}}

	# Track updated keys
	updated = []

	# Set default plot settings
	for instance in settings:	
		if any(setting in settings[instance] for setting in defaults):
			subinstance = None
			settings[instance] = {subinstance: settings[instance]}


	# Get layout of plot instances		
	dim = 2
	kwargs = list(defaults['style']['layout'])
	updated = []
	subupdated = []
	layout = {
		instance:{
			kwarg: {
				subinstance: settings[instance][subinstance]['style'].get('layout',{}).get(
					kwarg,defaults['style']['layout'][kwarg])
					for subinstance in settings[instance]
				}
			for kwarg in kwargs
			}
		for instance in settings
		}
	subshape = {}



	# Form grids of layout depending on shape of variables in each plot
	# Get update to layout based on (reshaped) variables data shape and 
	# reshaping of variables data based on 
	# hyperparameters['axis'] = {attr:[[axis for ncols],[axis for nrows],[axis for labels][axis for plot]]}
	for instance in list(settings):

		for subinstance in list(settings[instance]):

			subupdated.clear()

			for setting in specials:
				for special in specials[setting]:
					if special in settings[instance][subinstance][setting]:

						if isinstance(settings[instance][subinstance][setting][special],dict):
							settings[instance][subinstance][setting][special] = [settings[instance][subinstance][setting][special]]

						for i in range(len(settings[instance][subinstance][setting][special])):							
							key = find(settings[instance][subinstance][setting][special][i],properties)[0]
							occurrence = keys.index(key)

							subndim = min(variables[occurrence][combination][kwarg][stat].ndim
								for combination in variables[occurrence]
								for kwarg in variables[occurrence][combination]
								for stat in variables[occurrence][combination][kwarg]
								)

							subaxis = hyperparameters.get('axis',{}).get(key['y'])
							if subaxis is None:
								subaxis = [[],[],[],[axis for axis in range(subndim)]]
							else:
								subaxis = [[axis] if isinstance(axis,int) else axis for axis in subaxis]


							if occurrence not in updated:
								
								updated.append(occurrence)

								for combination in variables[occurrence]:
									for kwarg in variables[occurrence][combination]:
										for stat in variables[occurrence][combination][kwarg]:
											variables[occurrence][combination][kwarg][stat] = variables[occurrence][combination][kwarg][stat].transpose(
												[ax for axis in subaxis for ax in axis]).reshape(
												[max(1,int(product([variables[occurrence][combination][kwarg][stat].shape[ax]
													for ax in axis])))
													for axis in subaxis]
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
							kwarg: 1 for kwarg in kwargs[:dim+1]
							})


					indx = samplelayouts['index']-1	
					nrow = (indx - indx%samplelayouts['ncols'])//samplelayouts['ncols']
					ncol = indx%samplelayouts['ncols']

					settings[instance][(subinstance,*position)] = copy.deepcopy(settings[instance][subinstance])

					settings[instance][(subinstance,*position)]['style']['layout'] = {
						kwarg:{
							**{kwarg: layout[instance][kwarg][subinstance] for kwarg in layout[instance]},
							**{kwargs[axis]:subshape[subinstance][axis] for axis in range(dim)},
							'index':index,
							'top':1 - (nrow)/samplelayouts['nrows'] if subsublayouts and samplelayouts['nrows']>1 else None,
							'bottom':1 - (nrow+1)/samplelayouts['nrows'] if subsublayouts and samplelayouts['nrows']>1 else None,
							'right':(ncol+1)/samplelayouts['ncols'] if subsublayouts and samplelayouts['ncols']>1 else None,
							'left':(ncol)/samplelayouts['ncols'] if subsublayouts and samplelayouts['ncols']>1 else None,							
							}[kwarg]
						for kwarg in layout[instance]
						}

					for setting in specials:
						for special in specials[setting]:
							if special in settings[instance][subinstance][setting]:

								settings[instance][(subinstance,*position)][setting][special].clear()

								for i in range(len(settings[instance][subinstance][setting][special])):
									key = find(settings[instance][subinstance][setting][special][i],properties)[0]
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

											subsettings = copy.deepcopy(settings[instance][subinstance][setting][special][j])

											for kwarg in variables[occurrence][combination]:
												pos = tuple(
													(*(position[axis]%variables[occurrence][combination][kwarg][stat].shape[axis] for axis in range(dim)),
													j%variables[occurrence][combination][kwarg][stat].shape[dim-1+1]))
												subsettings[kwarg] = variables[occurrence][combination][kwarg][stat][pos]

											subsettings.update({
												'label':'%s   %s'%(stat, ', '.join(['%s: %s'%(str(k),str(c)) for k,c in zip(key['label']['key'],combination)]))
												})

											settings[instance][(subinstance,*position)][setting][special].append(subsettings)

		for samplelayout in layouts:
			for subinstance in layouts[samplelayout]:
				settings[instance].pop(subinstance)


	for instance in settings:
		print('----',instance,'----')
		for subinstance in settings[instance]:
			print(subinstance)
			print(settings[instance][subinstance]['style']['layout'])
			for special in settings[instance][subinstance]['ax']['errorbar']:
				print(special)
			print()
		print()

	# Set plot settings
	for instance in settings:
		for subinstance in settings[instance]:
			for setting in settings[instance][subinstance]:

				for attr in specials.get(setting,{}):
					if attr not in settings[instance][subinstance][setting]:
						continue 
					if isinstance(settings[instance][subinstance][setting][attr],dict):
						settings[instance][subinstance][setting][attr] = [settings[instance][subinstance][setting][attr]]
					else:
						settings[instance][subinstance][setting][attr] = list(settings[instance][subinstance][setting][attr])					
					for i in range(len(settings[instance][subinstance][setting][attr])):
						settings[instance][subinstance][setting][attr][i] = {
							**settings[instance][subinstance][setting][attr][i],
							**{kwarg: getattr(plt.cm,settings[instance][subinstance][setting][attr][i][kwarg])(
								(len(settings[instance][subinstance][setting][attr]) - 1 -i)/len(variables[occurrence]))
								for kwarg in specials[setting][attr]
								if (
								(kwarg in settings[instance][subinstance][setting][attr][i]) and
								(kwarg in ['color','ecolor'])
								)},
						}

				# if setting in ['ax']:
				# 	attr = 'errorbar'
				# 	for i in range(len(settings[instance][subinstance][setting][attr])):
				# 		print('adjusting',i,settings[instance][subinstance][setting][attr][i]['color'])

				# Set custom plot settings
				if setting in ['fig']:
					for attr in ['savefig']:
						settings[instance][subinstance][setting][attr] = {
							**settings[instance][subinstance][setting].get(attr,{}),
							'fname':join(
								directory['plot'],
								'.'.join([file['plot'],instance]),
								ext=ext['plot']),
							}

				elif setting in ['ax']:
					pass

	for instance in settings:
		plot(settings=settings[instance])

	return

def main(args):

	Nargs = 3

	nargs = len(args)
	
	assert nargs < Nargs, 'Incorrect number of arguments passed'

	data,settings,hyperparameters = args[:Nargs]
	
	process(data,settings,hyperparameters)

	return

if __name__ == '__main__':
	main(sys.argv[1:])