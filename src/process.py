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

from src.utils import array,product
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

	for name in data:
		for attr in attrs:
			if attr not in data[name]:
				print(name,attr)
	sort = {attr: list(sorted(set([data[name][attr] 
					for name in data if (
					attr in data[name] and 
					isinstance(data[name][attr],scalars) and 
					attr in hyperparameters.get('sort',attrs)
					)]))) 
			for attr in attrs}
	sort = {attr: sort[attr] for attr in sort if len(sort[attr])>0}

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

	xy = {'x':None,'y':None,'label':None}
	

	# Get hyperparameters
	file,directory,ext = {},{},{}
	for attr in hyperparameters.get('path',{}):
		file[attr],directory[attr],ext[attr] = split(hyperparameters.get('path',{}).get(attr),directory=True,file=True,ext=True)

	# Get all keys from find
	keys = (leaves(settings,prop,types=(dict,list),returns='value') for prop in xy)
	keys = map(lambda i: dict(zip(xy,(*i[:2],i[2:]))),zip(*keys))

	variables = {}

	# Get variables data for each attribute of x,y,label properties
	# Shapes of variables of 2 + ndim + 1 dimensions of 
	# (
	#  # different models (excluding fixed x,y,label properties),
	#  # samples (for given fixed  x,y,label properties),
	#  attribute shape (ndim dimensions for attribute),
	#  # iterations (1 for fixed model sort that don't vary over optimization)
	#  )
	for instance,key in enumerate(keys):
		variables[instance] = {}
		for index in itertools.product(*(range(subshape[key['y']][axis]) for axis in range(subndim[key['y']]))):
			variables[instance][index] = {}
			for permutation in itertools.product(*[sort[attr] for attr in key['label'] if attr in sort and attr not in [key['x'],key['y']]]):
				params = dict(zip(key['label'],permutation))
				names = [name for name in data if all(data[name][attr] == params[attr] for attr in params)]
				unique = {permute: [name for name in names if all([data[name][k] == j for k,j in zip(sort,permute)])]
						  for permute in itertools.product(*[sort[k] for k in sort])
						  if all([params[k] == dict(zip(sort,permute))[k] for k in params]) and 
						  	 len([name for name in names if all([data[name][k] == j for k,j in zip(sort,permute)])]) > 0
						  }

				if len(unique) == 0:
					continue

				length = (len(unique),max(len(unique[permute]) for permute in unique))

				shapes = {}
				shapes['y'] = (*length,*shape[key['y']])
				shapes['x'] = (*length,*shape[key['x']]) if key['x'] in shape else shapes['y']
				shapes['label'] = (*length,*map(max,zip(*(shape[attr] for attr in key['label']))))

				print(key,index,params)					
				print(shapes)
				print(unique)
				print()
				continue


				xy = {
					'x':np.zeros(shapes[key['x']]),
					'y':np.zeros(shapes[key['y']]),
					'xerr':np.zeros(shapes[key['x']]),
					'yerr':np.zeros(shapes[key['y']]),
					'label':np.zeros(shapes[key['label']]),
					}

				xy = {
					'x':np.array([np.array([data[name][key['x']] for name in unique[permute]]).mean(0) for permute in unique]).astype(data[name][key['x']].dtype),
					'y':np.array([np.array([data[name][key['y']] for name in unique[permute]]).mean(0) for permute in unique]).astype(data[name][key['y']].dtype),
					'xerr':np.array([np.array([data[name][key['x']] for name in unique[permute]]).std(0) for permute in unique]).astype(data[name][key['x']].dtype),
					'yerr':np.array([np.array([data[name][key['y']] for name in unique[permute]]).std(0) for permute in unique]).astype(data[name][key['y']].dtype),
					'label':[[[data[name][attr] for attr in key['label']] for name in unique[permute]] for permute in unique],
					}

				variables[instance][index][permutation] = {}				

				variables[instance][index][permutation]['argsort'] = np.argsort(xy['x'][index])
				variables[instance][index][permutation]['x'] = xy['x'][index].reshape(-1)
				variables[instance][index][permutation]['y'] = xy['y'][index].reshape(-1)
				variables[instance][index][permutation]['xerr'] = xy['xerr'][index].reshape(-1)
				variables[instance][index][permutation]['yerr'] = xy['yerr'][index].reshape(-1)

				variables[instance][index][permutation]['xfunc'] = ({						
						('iteration','objective',('M',)): variables[instance][index][permutation]['x'] if (tuple(key) == ('iteration','objective',('M',))) else variables[instance][index][permutation]['x'],
						}[instance])

				variables[instance][index][permutation]['yfunc'] = ({
						('iteration','objective',('M',)): variables[instance][index][permutation]['y'] if (tuple(key) == ('iteration','objective',('M',))) else variables[instance][index][permutation]['y'],
						}[instance])

				variables[instance][index][permutation]['xfuncerr'] = ({						
						('iteration','objective',('M',)): variables[instance][index][permutation]['xerr'] if (tuple(key) == ('iteration','objective',('M',))) else variables[instance][index][permutation]['xerr'],
						}[instance])

				variables[instance][index][permutation]['yfuncerr'] = ({
						('iteration','objective',('M',)): variables[instance][index][permutation]['yerr'] if (tuple(key) == ('iteration','objective',('M',))) else variables[instance][index][permutation]['yerr'],
						}[instance])				

				variables[instance][index][permutation]['xfit'] = ({
						('iteration','objective',('M',)): variables[instance][index][permutation]['xfunc'] if (tuple(key) == ('iteration','objective',('M',))) else variables[instance][index][permutation]['xfunc'],
						}[instance])

				variables[instance][index][permutation]['yfit'] = ({
						('iteration','objective',('M',)): variables[instance][index][permutation]['yfunc'] if (tuple(key) == ('iteration','objective',('M',))) else variables[instance][index][permutation]['yfunc'],
						}[instance])
	
				variables[instance][index][permutation]['xcoef'] = ({
						('iteration','objective',('M',)): variables[instance][index][permutation]['xfunc'] if (tuple(key) == ('iteration','objective',('M',))) else variables[instance][index][permutation]['xfunc'],
						}[instance])

				variables[instance][index][permutation]['ycoef'] = ({
						('iteration','objective',('M',)): variables[instance][index][permutation]['yfunc'] if (tuple(key) == ('iteration','objective',('M',))) else variables[instance][index][permutation]['yfunc'],
						}[instance])



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
						'fname':join(directory['plot'],'.'.join([file['plot'],key['x'],key['y'],*key['label']]),ext=ext['plot']),
						}
					for attr in (['savefig'] if obj in ['fig'] else [])
					},
					**{attr:[{
						**settings[key].get(obj,{}).get(attr,{}),
						'x': variables[key][index][permutation]['xfunc'],
						'y': variables[key][index][permutation]['yfunc'],
						'xerr': variables[key][index][permutation]['xfuncerr'],
						'yerr': variables[key][index][permutation]['yfuncerr'],						
						'color':getattr(plt.cm,settings[key].get(obj,{}).get(attr,{}).get('color','viridis'))((len(variables[key][index]) - 1 - i)/len(variables[key][index])),
						'ecolor':getattr(plt.cm,settings[key].get(obj,{}).get(attr,{}).get('ecolor','viridis'))((len(variables[key][index]) - 1 - i)/len(variables[key][index])),
						'label':dict(zip(key['label'],permutation))[key['label'][0]],
						} for i,permutation in enumerate(variables[key][index])]
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