#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,copy
import numpy as np
import scipy as sp
import scipy.special
import pandas as pd
from natsort import natsorted,realsorted
import matplotlib.pyplot as plt

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
from src.utils import asndarray,asscalar
from src.utils import argmax,difference,is_nan,is_numeric,abs
from src.utils import e,pi,nan,scalars,nulls,scinotation,padder
from src.iterables import branches
from src.parallel import Parallelize,Pooler
from src.io import setup,load,dump,join,split,glob
from src.fit import fit,mean,std,normalize,sqrt,size
from src.plot import plot

AXES = ['x','y']
STATS = ['','err']
PROPS = ['%s%s'%(ax,stat) for ax in AXES for stat in STATS]


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


def process(data,settings,hyperparameters,fig=None,ax=None,cwd=None):
	'''
	Process data
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		settings (str,dict): Path to or dictionary of plot settings
		hyperparameters (str,dict): Path to or dictionary of process settings
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

	# Load plot settings
	settings = load(settings,default=settings)

	# Load process hyperparameters
	hyperparameters = load(hyperparameters,default=hyperparameters)

	# Load data
	default = {}
	paths = natsorted(set((subpath for path in set(data) for subpath in glob(path))))
	data = {path: load(path,default=default) for path in paths}

	# Get paths
	path,file,directory,ext,delimiter = {},{},{},{},{}
	hyperparameters['plot'] = {'plot':'plot.pdf','process':'process.hdf5',**hyperparameters.get('path',{})}
	for attr in hyperparameters['plot']:
		delimiter[attr] = hyperparameters.get('delimiter','.')
		directory[attr],file[attr],ext[attr] = split(
			hyperparameters['plot'][attr],
			directory=-1,file=True,ext=True) 
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
	axes = AXES
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
						data,axis=0,dtype=dtype,transform=fit),
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
						data,axis=0,dtype=dtype,transform=fit),
					}
				}	 
				for axis,kwarg in enumerate(['%serr'%(axis) for axis in axes])},
			}[kwarg]
		for kwarg in statistics 			 	 			 	 
		}


	# Get keys of properties of the form ({prop:attr} or {prop:{'key':(attr,),'value:(values,)}})
	keys = find(settings,properties)




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
