#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings
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

from src.utils import array,product,is_number,to_number,to_key_value
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


def process(data,settings,hyperparameters):
	'''
	Process data
	Args:
		data (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of data to process
		settings (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of plot settings
		hyperparameters (str,dict,iterable[str,dict]): Path(s) to or dictionary(ies) of process settings
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

	
	# Get attributes of data
	attrs = list(set([attr for name in data for attr in data[name]]))

	# Get attributes to sort on and attributes not to sort on if not existent in plot properties x,y,label
	sort = {attr: list(sorted(set([data[name][attr] 
					for name in data if (
					attr in data[name] and 
					isinstance(data[name][attr],scalars) and 
					attr in hyperparameters.get('sort',attrs)
					)]))) 
			for attr in attrs}
	sort = {attr: sort[attr] for attr in sort if len(sort[attr])>0}

	nullsort = {attr: list(sorted(set([data[name][attr] 
					for name in data if (
					attr in data[name] and 
					isinstance(data[name][attr],scalars) and 
					attr in hyperparameters.get('nullsort',[])
					)]))) 
			for attr in attrs}
	nullsort = {attr: nullsort[attr] for attr in nullsort if len(nullsort[attr])>0}


	# Get data as arrays, with at least 1 leading dimension
	for name in data:
		for attr in data[name]:
			data[name][attr] = np.array(data[name][attr])
			data[name][attr] = data[name][attr].reshape(*[1]*(max(0,1-data[name][attr].ndim)),*data[name][attr].shape)
	
	# Data names correspond to the instances of the models and samples of that model
	# Data attributes have ndim dimensions
	# Shape of data is shape of attribute + number of iterations, which have a maximum size across the data names
	subndim = {attr: min(max(0,data[name][attr].ndim-1) for name in data) for attr in attrs}
	subshape = {attr: tuple((max(data[name][attr].shape[axis] for name in data) for axis in range(subndim[attr]))) for attr in attrs}
	ndim = {attr: min(data[name][attr].ndim for name in data) for attr in attrs}
	shape = {attr: tuple(map(max,zip(*(data[name][attr].shape for name in data)))) for attr in attrs}

	print(subndim)
	print(subshape)
	print()
	print(ndim)
	print(shape)
	print()



	# Get hyperparameters
	file,directory,ext = {},{},{}
	for attr in hyperparameters.get('path',{}):
		file[attr],directory[attr],ext[attr] = split(hyperparameters.get('path',{}).get(attr),directory=True,file=True,ext=True)

	# Get all keys from finding leaves of settings dictionary with keys in props

	# Get plot props from settings
	props = ['x','y','label']
	props = {prop:{'%s'%(prop):'mean','%serr'%(prop):'std'} if prop not in ['label'] else {} for prop in props}

	# Get keys of the form ({prop:attr} or {prop:{'key':(attr,),'value:(values,)}})
	keys = (leaves(settings,prop,types=(dict,list),returns='value') for prop in props)
	keys = map(lambda key: dict(zip(props,(*key[:2],tuple((dict(zip(['key','value'],to_key_value(key[2]))),)) if isinstance(key[2],str) else tuple((dict(zip(['key','value'],to_key_value(j))) for j in key[2]))))),zip(*keys))
	keys = [{prop:key[prop] if prop not in ['label'] else {label:tuple((value[label] for value in key[prop])) for label in set(label for value in key[prop] for label in value)}
			 for prop in key} for key in keys]

	# For 'label' prop with attributes and values to sort on, 
	# datasets are sorted into sets of unique datasets that correspond 
	# to all possible combinations of the label values. i.e) if label is ('M','N'), 
	# will sort into sets of datasets that correspond to each possible 'M','N' pair. 
	# For each combination of specific label values, statistics about the set of sample datasets 
	# corresponding to the label values are computed
	# Label attributes with a non-None value indicate fixed values that the datasets must equal for that attribute
	# when sorting, and the set of datasets for that label combination are constrained to be those with the fixed values.
	# Get function to accept dataset give label attributes and values 
	# as an acceptable sample for statistics of the particular label attributes
	# i.e) Include all datasets with attribute if attribute value is None, else only if dataset equals attribute value

	def include(key,value,data):
		'''
		Include data if conditions on key and value of data are True
		Args:
			key (str,tuple[str]): Key(s) of data to check
			value (object,tuple[object]): Values(s) of data to check
		Returns:
			boolean (bool): Accept inclusion of dataset
		'''
		boolean = False
		if not isinstance(key,tuple):
			key = (key,)
		if not isinstance(value,tuple):
			value = (value,)
		if all(v is None for v  in value):
			boolean = True			
		if value
	values = {key:None for key in keys}
	print(values)
	exit()
	
	# Get variables sets of datasets for all combinations of 'label' prop values, 
	# as per label attribute values constraints. 

	# Get variables data for each attribute of x,y,label properties
	# Shapes of variables of 2 + ndim + 1 dimensions of 
	# (
	#  # different models (excluding fixed x,y,label properties),
	#  # samples (for given fixed  x,y,label properties),
	#  attribute shape (ndim dimensions for attribute),
	#  # iterations (1 for fixed model sort that don't vary over optimization)
	#  )
	variables = {}
	for occurrence,key in enumerate(keys):
		variables[occurrence] = {}
		combinatations = [sort[attr] for attr in sort if attr]
		keysort = {attr:sort[attr] for attr in sort if attr not in [attr for attr in nullsort if attr not in [key['x'],key['y'],*key['label']['key']]]}
		combinations = [sort[attr] for attr,value in zip(key['label']['key'],key['label']['value']) if attr in sort and attr not in [key['x'],key['y']]]
		for combination in itertools.product(*combinations):
			variables[occurrence][combination] = None
			params = dict(zip(key['label']['key'],combination))

			names = [name for name in data if all(data[name][attr] == params[attr] for attr in params)]
			unique = {permute: [name for name in names if all([data[name][k] == j for k,j in zip(keysort,permute)])]
					  for permute in itertools.product(*[keysort[k] for k in keysort])
					  if all([params[k] == dict(zip(keysort,permute))[k] for k in params]) and 
					  	 len([name for name in names if all([data[name][k] == j for k,j in zip(keysort,permute)])]) > 0
					  }



	for occurrence,key in enumerate(keys):
		variables[occurrence] = {}
		keysort = {attr:sort[attr] for attr in sort if attr not in [attr for attr in nullsort if attr not in [key['x'],key['y'],*key['label']['key']]]}
		for index in itertools.product(*(range(subshape[key['y']][axis]) for axis in range(subndim[key['y']]))):
			variables[occurrence][index] = {}
			for combination in itertools.product(*[sort[attr] for attr in key['label']['key'] if attr in sort and attr not in [key['x'],key['y']]]):
				params = dict(zip(key['label']['key'],combination))
				names = [name for name in data if all(data[name][attr] == params[attr] for attr in params)]
				unique = {permute: [name for name in names if all([data[name][k] == j for k,j in zip(keysort,permute)])]
						  for permute in itertools.product(*[keysort[k] for k in keysort])
						  if all([params[k] == dict(zip(keysort,permute))[k] for k in params]) and 
						  	 len([name for name in names if all([data[name][k] == j for k,j in zip(keysort,permute)])]) > 0
						  }

				# if len(unique) == 0:

				# 	continue

				length = (len(unique),max(len(unique[permute]) for permute in unique))

				shapes = {}
				shapes['y'] = (*length,*shape[key['y']])
				shapes['x'] = (*length,*shape[key['x']]) if key['x'] in shape else shapes['y']
				# shapes['label'] = (*length,*map(max,zip(*(shape[attr] for attr in key['label']['key']))))

				print(key,index,params)		
				print(len(names),names)			
				print(shapes)
				print(unique)
				print()

				# Set props values for index and combination
				xy = {}
				for prop in props:

					if prop not in shapes:
						continue

					for _prop_ in props[prop]:
						xy[_prop_] = np.zeros(shapes[prop])
						for p,permute in enumerate(unique):
							for n,name in enumerate(unique[permute]):
								if key[prop] not in data[name]:
									data[name][key[prop]] = n*np.ones(shape[key['y']])
								print(key,prop,shapes[prop],data[name][key[prop]].shape)
								# xy[_prop_][p,n] = 

				continue



				xy = {
					'x':np.array([np.array([data[name][key['x']] for name in unique[permute]]).mean(0) for permute in unique]).astype(data[name][key['x']].dtype),
					'y':np.array([np.array([data[name][key['y']] for name in unique[permute]]).mean(0) for permute in unique]).astype(data[name][key['y']].dtype),
					'xerr':np.array([np.array([data[name][key['x']] for name in unique[permute]]).std(0) for permute in unique]).astype(data[name][key['x']].dtype),
					'yerr':np.array([np.array([data[name][key['y']] for name in unique[permute]]).std(0) for permute in unique]).astype(data[name][key['y']].dtype),
					'label':[[[data[name][attr] for attr in key['label']['key']] for name in unique[permute]] for permute in unique],
					}

				variables[occurrence][index][combination] = {}				

				variables[occurrence][index][combination]['argsort'] = np.argsort(xy['x'][index])
				variables[occurrence][index][combination]['x'] = xy['x'][index].reshape(-1)
				variables[occurrence][index][combination]['y'] = xy['y'][index].reshape(-1)
				variables[occurrence][index][combination]['xerr'] = xy['xerr'][index].reshape(-1)
				variables[occurrence][index][combination]['yerr'] = xy['yerr'][index].reshape(-1)

				variables[occurrence][index][combination]['xfunc'] = ({						
						('iteration','objective',('M',)): variables[occurrence][index][combination]['x'] if (tuple(key) == ('iteration','objective',('M',))) else variables[occurrence][index][combination]['x'],
						}[occurrence])

				variables[occurrence][index][combination]['yfunc'] = ({
						('iteration','objective',('M',)): variables[occurrence][index][combination]['y'] if (tuple(key) == ('iteration','objective',('M',))) else variables[occurrence][index][combination]['y'],
						}[occurrence])

				variables[occurrence][index][combination]['xfuncerr'] = ({						
						('iteration','objective',('M',)): variables[occurrence][index][combination]['xerr'] if (tuple(key) == ('iteration','objective',('M',))) else variables[occurrence][index][combination]['xerr'],
						}[occurrence])

				variables[occurrence][index][combination]['yfuncerr'] = ({
						('iteration','objective',('M',)): variables[occurrence][index][combination]['yerr'] if (tuple(key) == ('iteration','objective',('M',))) else variables[occurrence][index][combination]['yerr'],
						}[occurrence])				

				variables[occurrence][index][combination]['xfit'] = ({
						('iteration','objective',('M',)): variables[occurrence][index][combination]['xfunc'] if (tuple(key) == ('iteration','objective',('M',))) else variables[occurrence][index][combination]['xfunc'],
						}[occurrence])

				variables[occurrence][index][combination]['yfit'] = ({
						('iteration','objective',('M',)): variables[occurrence][index][combination]['yfunc'] if (tuple(key) == ('iteration','objective',('M',))) else variables[occurrence][index][combination]['yfunc'],
						}[occurrence])
	
				variables[occurrence][index][combination]['xcoef'] = ({
						('iteration','objective',('M',)): variables[occurrence][index][combination]['xfunc'] if (tuple(key) == ('iteration','objective',('M',))) else variables[occurrence][index][combination]['xfunc'],
						}[occurrence])

				variables[occurrence][index][combination]['ycoef'] = ({
						('iteration','objective',('M',)): variables[occurrence][index][combination]['yfunc'] if (tuple(key) == ('iteration','objective',('M',))) else variables[occurrence][index][combination]['yfunc'],
						}[occurrence])



	exit()
	settings.update({
		key: {
			index: {
				obj:{
					**{attr:
						settings[key].get(obj,{}).get(attr)
						for attr in settings[key].get(obj,{})
					},
					**{attr:{
						**settings[key].get(obj,{}).get(attr,{}),
						'fname':join(directory['plot'],'.'.join([file['plot'],key['x'],key['y'],*key['label']['key']]),ext=ext['plot']),
						}
					for attr in (['savefig'] if obj in ['fig'] else [])
					},
					**{attr:[{
						**settings[key].get(obj,{}).get(attr,{}),
						'x': variables[key][index][combination]['xfunc'],
						'y': variables[key][index][combination]['yfunc'],
						'xerr': variables[key][index][combination]['xfuncerr'],
						'yerr': variables[key][index][combination]['yfuncerr'],						
						'color':getattr(plt.cm,settings[key].get(obj,{}).get(attr,{}).get('color','viridis'))((len(variables[key][index]) - 1 - i)/len(variables[key][index])),
						'ecolor':getattr(plt.cm,settings[key].get(obj,{}).get(attr,{}).get('ecolor','viridis'))((len(variables[key][index]) - 1 - i)/len(variables[key][index])),
						'label':dict(zip(key['label']['key'],combination))[key['label']['key'][0]],
						} for i,combination in enumerate(variables[key][index])]
					for attr in (['errorbar'] if obj in ['ax'] else [])
					},
					**{attr: join(directory[attr],file[attr],ext=ext[attr])
					for attr in (['mplstyle'] if obj in ['style'] else [])
					},					
				}
				for obj in settings[key]
			}
			for index in range(size[key['y']])
		}
		for key in settings
	})



	for key in settings:
		# for index in settings[key]:
		# 	for obj in settings[key][index]:
		# 		print(obj)
		# 		print(settings[key][index][obj])
		# 		print()
		plot(settings=settings[key])

	return

def main(args):
	nargs = len(args)

	path = args[0] if nargs>0 else None

	process(path)

	return

if __name__ == '__main__':
	main(sys.argv[1:])