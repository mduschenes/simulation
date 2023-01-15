#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings
from copy import deepcopy
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
from src.iterables import brancher,getter,setter
from src.parallel import Parallelize,Pooler
from src.io import load,dump,join,split
from src.fit import fit,mean,std,normalize,sqrt,size
from src.plot import plot

AXES = ['x','y']
OTHER = 'label'
PLOTS = ['plot','scatter','errorbar','histogram','axvline','axhline','vlines','hlines','plot_surface']


class GroupBy(object):
	def __init__(self,df):
		'''
		Null groupby wrapper for dataframe
		Args:
			df (dataframe): dataframe
		'''
		self.df = df
		self.groups = [None]
		return

	def get_group(self,group):
		return self.df


def Texify(string,texify={},usetex=True):
	'''
	Texify string
	Args:
		string (str): String to texify
		texify (dict): Dictionary of texify translations of strings
		usetex (bool): Use latex formatting
	Returns:
		string (str): Texified string
	'''

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


def setup(data,settings,hyperparameters,pwd=None,cwd=None):
	'''
	Setup data, settings, hyperparameters
	Args:
		data (dataframe): dataframe
		settings (dict): settings
		hyperparameters (dict): hyperparameters
		pwd (str): Root path of data
		cwd (str): Root path of plots
	'''

	# Set plot settings
	defaults = {
		'ax': {},
		'fig': {},
		'style': {
			'layout': {
				'nrows':1,'ncols':1,'index':1,
				'left':None,'right':None,'top':None,'bottom':None,
				'hspace':None,'wspace':None,'pad':None
				}
			}
		}

	for instance in list(settings):
		if settings.get(instance) is None:
			settings.pop(instance,None);

		if all(subinstance in defaults for subinstance in settings[instance]):
			settings[instance] = {None: settings[instance]}
		for subinstance in settings[instance]:
			setter(settings[instance][subinstance],defaults,delimiter=delim,func=False)

			prop = 'ax'
			for plots in PLOTS:
				if not settings[instance][subinstance][prop].get(plots):
					continue
				elif isinstance(settings[instance][subinstance][prop][plots],dict):
					settings[instance][subinstance][prop][plots] = [[[settings[instance][subinstance][prop][plots]]]]
				elif all(isinstance(subplots,dict) for subplots in settings[instance][subinstance][prop][plots]):
					settings[instance][subinstance][prop][plots] = [[[subplots]] for subplots in settings[instance][subinstance][prop][plots]]
				elif all(isinstance(subsubplots,dict) for subplots in settings[instance][subinstance][prop][plots] for subsubplots in subplots):
					settings[instance][subinstance][prop][plots] = [[[subsubplots] for subsubplots in subplots] for subplots in settings[instance][subinstance][prop][plots]]

	# Set process hyperparameters
	defaults = {
		'load':None,
		'dump':None,
		'plot':None,
		}
	setter(hyperparameters,defaults,delimiter=delim,func=False)

	# Get paths
	hyperparameters['file'],hyperparameters['directory'],hyperparameters['ext'] = {},{},{}
	for attr in hyperparameters['path']:
		hyperparameters['directory'][attr] = cwd
		hyperparameters['file'][attr],hyperparameters['ext'][attr] = split(
			hyperparameters['path'][attr],
			file=True,ext=True) 

	# Get plot fig and axes
	fig,ax = hyperparameters.get('fig'),hyperparameters.get('ax')
	if fig is None:
		fig = {}
	if ax is None:
		ax = {}

	for instance in settings:
		if instance not in fig:
			fig[instance] = None
		if instance not in ax:
			ax[instance] = None
	hyperparameters['fig'],hyperparameters['ax'] = fig,ax

	# Get texify
	texify = hyperparameters.get('texify',{})
	usetex = hyperparameters.get('usetex',False)
	hyperparameters['texify'] = lambda string,texify=texify,usetex=usetex: Texify(string,texify,usetex=usetex)


	return

def find(dictionary):
	'''
	Find formatted keys from dictionary, based on search keys of the form 'property':attr
	All keys are assumed to be present in any branches where one or more property is found in dictionary
	Args:
		dictionary (dict): Dictionary to search
	Returns:
		keys (dict[dict]): Formatted keys based on found keys of the form {name: {prop:attr} or {prop:{attr:value}}}
	'''

	keys = AXES
	other = [OTHER]

	def parser(string,separator,default):
		if string.count(separator):
			key,value = string.split(separator)[0],to_number(separator.join(string.split(separator)[1:]))
		else:
			key,value = string,default
		return key,value

	default = null
	separator = '='
	defaults = {
				'func':{'stat':{'':'mean','err':'std'}}, 
				'wrapper':{},
				'attrs':{},
				'slice':None,
				'axis':{'row':[],'col':[],'plot':['plot','group','func'],'axis':[-1]},
				'settings':{
					'ax.set_ylabel.ylabel':[['$\\textrm{Infidelity}$']],
					'ax.legend.update':'%s \\textrm{Noisy}'
				},
				'texify':{}		
	}

	elements = [*keys,*other]
	keys = brancher(dictionary,elements)
	
	keys = {key[:-1]:dict(zip(elements,[value[-1] for value in key[-1]])) for key in keys}

	for key in keys:
		for attr in keys[key]:
			
			if attr in other:

				if isinstance(keys[key][attr],dict):
					keys[key][attr] = {prop: keys[key][attr][prop] if keys[key][attr] is not None else default for prop in keys[key][attr]}
				elif isinstance(keys[key][attr],str):
					keys[key][attr] = dict((parser(keys[key][attr],separator=separator,default=default),))
				else:
					keys[key][attr] = dict((parser(prop,separator=separator,default=default) for prop in keys[key][attr]))

				if attr in keys[key][attr]:
					if isinstance(keys[key][attr][attr],dict):
						keys[key][attr][attr] = {prop: keys[key][attr][attr][prop] if keys[key][attr][attr][prop] is not None else default for prop in keys[key][attr][attr]}
					elif isinstance(keys[key][attr][attr],str):
						keys[key][attr][attr] = dict((parser(keys[key][attr][attr],separator=separator,default=default)),)
					else:
						keys[key][attr][attr] = dict((parser(prop,separator=separator,default=default) for prop in keys[key][attr][attr]))
				else:
					keys[key][attr] = {attr: keys[key][attr]}

				setter(keys[key][attr],defaults,delimiter=delim,func=False)
			
			else:
				if not keys[key][attr]:
					keys[key][attr] = default
				elif isinstance(keys[key][attr],dict):
					keys[key][attr] = keys[key][attr][list(keys[key][attr])[-1]]
				elif isinstance(keys[key][attr],str):
					keys[key][attr] = keys[key][attr]
				else:
					keys[key][attr] = keys[key][attr][-1]

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


def apply(keys,data,settings,hyperparameters):
	'''
	Apply functions based on keys to data
	Args:
		keys (dict): Keys of functions to apply
		data (dataframe): dataframe
		settings (dict): settings
		hyperparameters (dict): hyperparameters
	'''

	for name in keys:

		axes = [axis for axis in AXES if axis in keys[name]]
		other = OTHER
		label = keys[name][other].get(other,{})
		funcs = keys[name][other].get('func',{})

		if not funcs:
			funcs = {'stat':{'':'mean','err':'std'}}

		independent = [keys[name][axis] for axis in axes[:-1] if keys[name][axis] in data]
		dependent = [keys[name][axis] for axis in axes[-1:] if keys[name][axis] in data]
		labels = [attr for attr in label if attr in data and label[attr] is null]

		boolean = [parse(attr,label[attr],data) for attr in label]
		boolean = conditions(boolean,op='&')	

		by = [*labels,*independent]

		groupby = data[boolean].groupby(by=by,as_index=False)

		agg = {
			**{attr : [(attr,'first')] for attr in data},
			**{attr : [(delim.join(((attr,function,func))),funcs[function][func]) for function in funcs for func in funcs[function]] for attr in data if attr in dependent},
		}
		droplevel = dict(level=0,axis=1)
		by = [*labels]
		variables = [*independent,*dependent,*[subattr[0] for attr in dependent for subattr in agg[attr]]]

		groups = groupby.agg(agg).droplevel(**droplevel)

		if by:
			groups = groups.groupby(by=by,as_index=False)
		else:
			groups = GroupBy(groups)

		assert all(groups.get_group(group).columns.nlevels == 1 for group in groups.groups) # Possible future broken feature agg= (label,name)

		for i,group in enumerate(groups.groups):
			for j,function in enumerate(funcs):
				
				key = (*name[:-2],i,j)
				value = deepcopy(getter(settings,name,delimiter=delim))

				source = [attr for attr in data if attr not in variables]
				destination = other
				value[destination] = {attr: groups.get_group(group)[attr].to_list()[0] for attr in source}

				for func in funcs[function]:	
					for axis in axes:
						
						attr = keys[name][axis]

						source = delim.join(((attr,function,func))) if attr in dependent else attr
						destination = '%s%s'%(axis,func) if attr in dependent else axis
						value[destination] = groups.get_group(group)[source].to_numpy()
					
				# print(key,value[other])
				# print()

				setter(settings,{key:value},delimiter=delim,func=True)

	return


def plotter(keys,data,settings,hyperparameters):
	'''
	Plot data based on keys, dataframe, plot settings, process hyperparameters
	Args:
		keys (dict): keys
		data (dataframe): dataframe
		settings (dict): settings
		hyperparameters (dict): hyperparameters
	'''

	# Set layout
	layout = {}
	for instance in settings:
		for subinstance in settings[instance]:
			sublayout = settings[instance][subinstance]['style']['layout']
			if not layout.get(instance):
				layout[instance] = {prop:sublayout[prop] for prop in ['nrows','ncols']}
			layout[instance] = {
				'nrows':max(sublayout['nrows'],layout['nrows']),
				'ncols':max(sublayout['ncols'],layout['ncols']),
				'index':None,
				'left':None,'right':None,'top':None,'bottom':None,
				'hspace':None,'wspace':None,'pad':None
				}
			# 'index':index+1,
			# 'top':1 - (nrow)/samplelayouts['nrows'] if subsublayouts and samplelayouts['nrows']>1 else None,
			# 'bottom':1 - (nrow+1)/samplelayouts['nrows'] if subsublayouts and samplelayouts['nrows']>1 else None,
			# 'right':(ncol+1)/samplelayouts['ncols'] if subsublayouts and samplelayouts['ncols']>1 else None,
			# 'left':(ncol)/samplelayouts['ncols'] if subsublayouts and samplelayouts['ncols']>1 else None,											


	return



def process(data,settings,hyperparameters,fig=None,ax=None,pwd=None,cwd=None):
	'''
	Process data
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		settings (str,dict): Path to or dictionary of plot settings
		hyperparameters (str,dict): Path to or dictionary of process settings
		fig (dict): dictionary of subplots of figures of plots {key: figure}
		ax (dict): dictionary of subplots of axes of plots {key: figure}
		pwd (str): Root path of data
		cwd (str): Root path of plots
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
	path = join(settings,root=pwd) if isinstance(settings,str) else settings
	default = settings
	wrapper = None	
	settings = load(path,default=default,wrapper=wrapper)

	# Load process hyperparameters
	path = join(hyperparameters,root=pwd) if isinstance(hyperparameters,str) else hyperparameters
	default = hyperparameters
	wrapper = None
	hyperparameters = load(path,default=default,wrapper=wrapper)

	# Set settings and hyperparameters
	setup(data,settings,hyperparameters,pwd,cwd)

	# Get keys of the form {name:{prop:{attr:value}}}
	keys = find(settings)

	# Set metadata
	attr = 'metadata'
	metadata = join(hyperparameters['directory'][attr],hyperparameters['file'][attr],ext=hyperparameters['ext'][attr])

	# Load settings
	if hyperparameters['load']:

		settings = load(metadata)

	else:

		# Load data
		path = data
		default = {}
		wrapper = 'df'
		data = load(path,default=default,wrapper=wrapper)

		# Get functions of data
		apply(keys,data,settings,hyperparameters)

	# Dump settings
	if hyperparameters['dump']:
		
		dump(settings,metadata)


	# Plot data
	if hyperparameters['plot']:

		plotter(keys,data,settings,hyperparameters)

	return


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
			'default':None,
			'nargs':'?'
		},
		'--hyperparameters':{
			'help':'Process process settings',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--pwd':{
			'help':'Process pwd',
			'type':str,
			'default':None,
			'nargs':'?',
		},		
		'--cwd':{
			'help':'Process cwd',
			'type':str,
			'default':None,
			'nargs':'?',
		},						
	}

	wrappers = {
		'pwd':lambda kwarg,wrappers,kwargs: split(kwargs['data'][-1],directory=True).replace('/**','').replace('**','') if kwargs.get(kwarg) is None else kwargs.get(kwarg),
		'cwd':lambda kwarg,wrappers,kwargs: split(kwargs['data'][-1],directory=True).replace('/**','').replace('**','') if kwargs.get(kwarg) is None else kwargs.get(kwarg),
	}

	args = argparser(arguments,wrappers)

	main(*args,**args)
