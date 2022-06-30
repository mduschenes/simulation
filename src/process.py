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

from src.utils import array
from src.dictionary import leaves
from src.io import setup,load,dump,_load
from src.plot import plot


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


def process(data,settings):
	'''
	Process data
	Args:
		data (str,dict): Path to or dictionary of data to process
		settings (str,dict): Path to or dictionary of plot settings
	'''

	scalars = (int,np.integer,float,np.float)


	if isinstance(data,str):
		default = {}
		directory = os.path.dirname(data)
		data = load(data,default=default)
	else:
		directory = '.'

	if isinstance(settings,str):
		default = {}
		settings = load(settings,default=default)

	attrs = list(set([attr for name in data for attr in data[name]]))

	parameters = {attr: list(sorted(set([data[name][attr] 
					for name in data if attr in data[name] and isinstance(data[name][attr],scalars)]))) 
			for attr in attrs}
	
	values = {attr: [data[name][attr] for name in data if attr in data[name] and not isinstance(data[name][attr],scalars)] 
				for attr in attrs}
	
	parameters = {attr: parameters[attr] for attr in parameters if len(parameters[attr])>0}
	values = {attr: values[attr] for attr in values if len(values[attr])>0}


	for name in data:
		for attr in data[name]:
			data[name][attr] = np.array(data[name][attr])
			data[name][attr] = data[name][attr].reshape(*[1]*(max(0,2-data[name][attr].ndim)),*data[name][attr].shape)
	
	size = {attr: min([data[name][attr].shape[0] for name in data]) for attr in attrs}
	shape = {attr: tuple(map(max,zip(*(data[name][attr].shape for name in data)))) for attr in attrs}

	xy = {'x':None,'y':None,'xerr':None,'yerr':None,'label':None}
	
	for key in list(settings):
		new = key.split('__')
		new = (*new[:2],(*new[2:],))
		settings[new] = settings.pop(key)
	
	# Get all keys from find
	keys = {prop: [key for key in leaves(settings,prop,types=(dict,type),returns='value')] for prop in xy}
	keys = []
	settings = {key: settings[key] for key in settings if key in [('iteration','objective',('M',))]}


	variables = {}

	for key in settings:
		variables[key] = {}
		for index in range(size[key[1]]):
			variables[key][index] = {}
			for permutation in itertools.product(*[parameters[attr] for attr in key[2]]):
				params = dict(zip(key[2],permutation))
				names = [name for name in data if all(data[name][attr] == params[attr] for attr in params)]
				unique = {permute: [name for name in names if all([data[name][k] == j for k,j in zip(parameters,permute)])]
						  for permute in itertools.product(*[parameters[k] for k in parameters])
						  if all([params[k] == dict(zip(parameters,permute))[k] for k in params]) and 
						  	 len([name for name in names if all([data[name][k] == j for k,j in zip(parameters,permute)])]) > 0
						  }
				length = (len(unique),max(len(unique[permute]) for permute in unique))

				shapes = {
					key[0]: (*length,*shape[key[0]]),
					key[1]: (*length,*shape[key[1]]),
					key[2]: (*length,*map(max,zip(*(shape[attr] for attr in key[2]))))
					}

				print(key,index,params,shapes)	
				exit()			

				xy = {
					'x':np.zeros(shapes[key[0]]),
					'y':np.zeros(shapes[key[1]]),
					'xerr':np.zeros(shapes[key[0]]),
					'yerr':np.zeros(shapes[key[1]]),
					'label':np.zeros(shapes[key[2]]),
					}

				xy = {
					'x':np.array([np.array([data[name][key[0]] for name in unique[permute]]).mean(0) for permute in unique]).astype(data[name][key[0]].dtype),
					'y':np.array([np.array([data[name][key[1]] for name in unique[permute]]).mean(0) for permute in unique]).astype(data[name][key[1]].dtype),
					'xerr':np.array([np.array([data[name][key[0]] for name in unique[permute]]).std(0) for permute in unique]).astype(data[name][key[0]].dtype),
					'yerr':np.array([np.array([data[name][key[1]] for name in unique[permute]]).std(0) for permute in unique]).astype(data[name][key[1]].dtype),
					'label':[[[data[name][attr] for attr in key[2]] for name in unique[permute]] for permute in unique],
					}

				variables[key][index][permutation] = {}				

				variables[key][index][permutation]['argsort'] = np.argsort(xy['x'][index])
				variables[key][index][permutation]['x'] = xy['x'][index].reshape(-1)
				variables[key][index][permutation]['y'] = xy['y'][index].reshape(-1)
				variables[key][index][permutation]['xerr'] = xy['xerr'][index].reshape(-1)
				variables[key][index][permutation]['yerr'] = xy['yerr'][index].reshape(-1)

				variables[key][index][permutation]['xfunc'] = ({						
						('iteration','objective',('M',)): variables[key][index][permutation]['x'] if (key == ('iteration','objective',('M',))) else variables[key][index][permutation]['x'],
						}[key])

				variables[key][index][permutation]['yfunc'] = ({
						('iteration','objective',('M',)): variables[key][index][permutation]['y'] if (key == ('iteration','objective',('M',))) else variables[key][index][permutation]['y'],
						}[key])

				variables[key][index][permutation]['xfuncerr'] = ({						
						('iteration','objective',('M',)): variables[key][index][permutation]['xerr'] if (key == ('iteration','objective',('M',))) else variables[key][index][permutation]['xerr'],
						}[key])

				variables[key][index][permutation]['yfuncerr'] = ({
						('iteration','objective',('M',)): variables[key][index][permutation]['yerr'] if (key == ('iteration','objective',('M',))) else variables[key][index][permutation]['yerr'],
						}[key])				

				variables[key][index][permutation]['xfit'] = ({
						('iteration','objective',('M',)): variables[key][index][permutation]['xfunc'] if (key == ('iteration','objective',('M',))) else variables[key][index][permutation]['xfunc'],
						}[key])

				variables[key][index][permutation]['yfit'] = ({
						('iteration','objective',('M',)): variables[key][index][permutation]['yfunc'] if (key == ('iteration','objective',('M',))) else variables[key][index][permutation]['yfunc'],
						}[key])
	
				variables[key][index][permutation]['xcoef'] = ({
						('iteration','objective',('M',)): variables[key][index][permutation]['xfunc'] if (key == ('iteration','objective',('M',))) else variables[key][index][permutation]['xfunc'],
						}[key])

				variables[key][index][permutation]['ycoef'] = ({
						('iteration','objective',('M',)): variables[key][index][permutation]['yfunc'] if (key == ('iteration','objective',('M',))) else variables[key][index][permutation]['yfunc'],
						}[key])



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
						'fname':os.path.join(directory,'.'.join(['__'.join([*key[:2],*key[2]]),'pdf'])),			
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
						'label':dict(zip(key[2],permutation))[key[2][0]],
						} for i,permutation in enumerate(variables[key][index])]
					for attr in (['errorbar'] if obj in ['ax'] else [])
					},
				}
				for obj in settings[key]
			}
			for index in range(size[key[1]])
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