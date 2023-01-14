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
from src.utils import array,product,expand_dims,conditions
from src.utils import asndarray,asscalar
from src.utils import to_key_value,to_number,to_str,to_int,is_iterable,is_number,is_nan,is_numeric
from src.utils import argmax,difference,abs
from src.utils import e,pi,nan,scalars,delim,nulls,null,Null,scinotation
from src.iterables import brancher
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


def find(dictionary,keys,*other):
	'''
	Find formatted keys from dictionary, based on search keys of the form 'property':value
	All keys are assumed to be present in any branches where one or more property is found in dictionary
	Args:
		dictionary (dict): Dictionary to search
		keys (iterable[str]): Iterable of keys to search for
		other (iterable[str]): Iterable of keys to search for
	Returns:
		keys (dict[dict]): Formatted keys based on found keys of the form {name: {prop:{attr:value}}}
	'''

	def parser(string,separator,default):
		if string.count(separator):
			key,value = string.split(separator)[0],to_number(separator.join(string.split(separator)[1:]))
		else:
			key,value = string,default
		return key,value

	default = null
	separator = '='

	elements = [*keys,*other]
	keys = brancher(dictionary,elements)
	
	keys = {key[:-1]:dict(zip(elements,[value[-1] for value in key[-1]])) for key in keys}

	for key in keys:
		for attr in keys[key]:
			if isinstance(keys[key][attr],dict):
				keys[key][attr] = {prop: keys[key][attr][prop] if keys[key][attr] is not None else default for prop in keys[key][attr]}
			elif isinstance(keys[key][attr],str):
				keys[key][attr] = dict((parser(keys[key][attr],separator=separator,default=default),))
			else:
				keys[key][attr] = dict((parser(prop,separator=separator,default=default) for prop in keys[key][attr]))


		for attr in other:
			if attr in keys[key][attr]:
				if isinstance(keys[key][attr][attr],dict):
					keys[key][attr][attr] = {prop: keys[key][attr][attr][prop] if keys[key][attr][attr][prop] is not None else default for prop in keys[key][attr][attr]}
				elif isinstance(keys[key][attr][attr],str):
					keys[key][attr][attr] = dict((parser(keys[key][attr][attr],separator=separator,default=default)),)
				else:
					keys[key][attr][attr] = dict((parser(prop,separator=separator,default=default) for prop in keys[key][attr][attr]))
			else:
				keys[key][attr] = {attr: keys[key][attr]}

	return keys


def parse(key,value,data):
	'''
	Parse key and value condition for data, such that data[key] == value
	Args:
		key (str): key of condition
		value (str): value of condition, allowed string in 
			[None,
			'value' (explicit value),
			'@key@' (data value), 
			'#i,j,k,...#' (index value),
			'%start,stop,step%' (slice value),]
		data (dataframe): data of condition
	Returns:
		out (dataframe): Condition on data indices
	'''
	delimiters = ['$','@','#','%']
	default = data.assign(**{key:True})[key]
	separator = ','

	out = default
	
	if key not in data:
		pass
	elif isinstance(value,Null):
		pass
	elif isinstance(value,str):
		for delimiter in delimiters:

			if value.startswith(delimiter) and value.endswith(delimiter):
			
				values = value.replace(delimiter,'').split(separator)

				if delimiter in ['$']: # Explicit value: value
					parser = lambda value: (to_number(value) if len(value)>0 else null)
					values = [parser(value) for value in values]           
					values = [value for value in values if not isinstance(value,Null)]

					if values and not isinstance(values,Null):
						out = data[key].isin(values)

				elif delimiter in ['@']: # Data value: key
					parser = lambda value: (to_str(value) if len(value)>0 else null)
					values = [parser(value) for value in values]           
				  
					if values and not isinstance(values,Null):
						out = conditions([data[key]==data[value] for value in values],op='or')

				elif delimiter in ['#']: # Index value: i,j,k,...
					parser = lambda value: (to_int(value) if len(value)>0 else null)
					values = [parser(value) for value in values]
					values = [value for value in values if not isinstance(value,Null)]

					if values and not isinstance(values,Null):
						out = data[key].unique()
						out = data[key].isin(out[[value for value in values if value < out.size]])

				elif delimiter in ['%']: # Slice value start,stop,step
					parser = lambda value: (to_int(value) if len(value)>0 else None)
					values = [*(parser(value) for value in values),*[None]*(3-len(values))]
					values = [value for value in values if not isinstance(value,Null)]
			
					if values and not isinstance(values,Null):
						out = data[key].isin(data[key].unique()[slice(*values)])
	
				break
	
	return out


def apply(name,keys,data,df):
	'''
	Apply functions based on keys to data
	Args:
		name (object): Key of keys to apply functions
		keys (dict): Keys of functions to apply
		data (dict): Data to insert grouped functions
		df (dataframe): Dataframe to apply functions to
	'''

	key = keys[name]

	axes = [axis for axis in key if axis not in ['label']]
	label = key['label'].get('label',{})
	funcs = key['label'].get('func',{})

	if not funcs:
		funcs = {"":"mean","err":"std"}

	independent = [attr for axis in axes[:-1] for attr in key[axis] if attr in df]
	dependent = [attr for axis in axes[-1:] for attr in key[axis] if attr in df]
	labels = [attr for attr in label if attr in df and label[attr] is null]

	boolean = [parse(attr,label[attr],df) for attr in label]
	boolean = conditions(boolean,op='&')	

	by = [*labels,*independent]

	groupby = df[boolean].groupby(by=by,as_index=False)

	print(independent,dependent,labels)

	agg = {
		**{attr : [(attr,'first')] for attr in df},
		**{attr : [(delim.join(((attr,*func.split(delim)))),funcs[func]) for func in funcs] for attr in df if attr in dependent},
	}
	droplevel = dict(level=0,axis=1)
	by = [*labels]

	data[name] = groupby.agg(agg).droplevel(**droplevel).groupby(by=by,as_index=False)

	assert all(data[name].get_group(group).columns.nlevels == 1 for group in data[name].groups) # Possible future broken feature agg= (label,name)

	for group in data[name].groups:
		value = data[name].get_group(group)
		print(group,value.shape)
	print()


	return


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
	path = data
	default = {}
	wrapper = 'df'
	data = load(path,default=default,wrapper=wrapper)

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


	# Get keys of the form {name:{prop:{attr:value}}}
	keys = [*axes]
	other = ['label']
	keys = find(settings,keys,*other)


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
