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
from src.iterables import brancher,getter,setter,flatten
from src.parallel import Parallelize,Pooler
from src.io import load,dump,join,split
from src.fit import fit
from src.postprocess import postprocess
from src.plot import plot,AXIS,VARIANTS,FORMATS,ALL,OTHER,PLOTS

class GroupBy(object):
	def __init__(self,df,by=[]):
		'''
		Null groupby wrapper for dataframe
		Args:
			df (dataframe): dataframe
		'''
		class grouper(object):
			def __init__(self,by):
				self.names = by
				return

		self.df = df
		self.by = by
		self.groups = ["None"]
		self.grouper = grouper(by=by)
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


def Valify(value,valify={},useval=True):
	'''
	Valify value
	Args:
		value (str): String to valify
		valify (dict): Dictionary of valify translations of strings
		useval (bool): Use value formatting
	Returns:
		value (str): Valified string
	'''
	values = {
		**valify,
		None: valify.get('None',None),
		'None': valify.get('None',None),
		}

	try:
		value = valify.get(value,value)
	except:
		pass

	return value


def setup(data,settings,hyperparameters,pwd=None,cwd=None):
	'''
	Setup data, settings, hyperparameters
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		settings (str,dict): Path to or dictionary of plot settings
		hyperparameters (str,dict): Path to or dictionary of process settings
		pwd (str): Root path of data
		cwd (str): Root path of plots
	Returns:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		settings (dict): Plot settings
		hyperparameters (dict): Process settings
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


	# Load plot settings
	path = join(settings,root=pwd) if isinstance(settings,str) else None
	default = None if isinstance(settings,str) else settings
	wrapper = None	
	settings = load(path,default=default,wrapper=wrapper)

	# Load process hyperparameters
	path = join(hyperparameters,root=pwd) if isinstance(hyperparameters,str) else None
	default = None if isinstance(hyperparameters,str) else hyperparameters
	wrapper = None
	hyperparameters = load(path,default=default,wrapper=wrapper)

	if (settings is None) or (hyperparameters is None):
		return data,settings,hyperparameters

	for instance in list(settings):
		if (settings.get(instance) is None):
			settings.pop(instance,None);
			continue

		if all(subinstance in defaults for subinstance in settings[instance]):
			settings[instance] = {"None": settings[instance]}
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
		'process':None,
		'postprocess':None,
		}
	setter(hyperparameters,defaults,delimiter=delim,func=False)

	# Get paths
	hyperparameters['file'],hyperparameters['directory'],hyperparameters['ext'] = {},{},{}
	for attr in hyperparameters['path']:
		hyperparameters['directory'][attr] = cwd
		hyperparameters['file'][attr],hyperparameters['ext'][attr] = split(
			hyperparameters['path'][attr],
			file=True,ext=True)
		hyperparameters['path'][attr] = join(hyperparameters['directory'][attr],hyperparameters['file'][attr],ext=hyperparameters['ext'][attr])

	# Set instances
	attr = 'instance'
	if hyperparameters.get(attr) is None:
		hyperparameters[attr] = {}
	elif isinstance(hyperparameters.get(attr),(bool,int)):
		hyperparameters[attr] = {instance: bool(hyperparameters[attr][instance]) for instance in settings}
	elif isinstance(hyperparameters.get(attr),list):
		hyperparameters[attr] = {**{instance: False for instance in settings},**{instance: True for instance in hyperparameters[attr]}}
	hyperparameters[attr] = {**{instance: True for instance in settings},**{instance: bool(hyperparameters[attr][instance]) for instance in hyperparameters[attr]}}

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

	# Get valify
	valify = hyperparameters.get('valify',{})
	useval = hyperparameters.get('useval',True)
	hyperparameters['valify'] = lambda value,valify=valify,useval=useval: Valify(value,valify,useval=useval)

	return data,settings,hyperparameters

def find(dictionary):
	'''
	Find formatted keys from dictionary, based on search keys of the form 'property':attr
	All keys are assumed to be present in any branches where one or more property is found in dictionary
	Args:
		dictionary (dict): Dictionary to search
	Returns:
		keys (dict[dict]): Formatted keys based on found keys of the form {name: {prop:attr} or {prop:{attr:value}}}
	'''

	dim = 2
	keys = AXIS[:dim]
	other = [OTHER]

	def parser(string,separator,default):
		if string.count(separator):
			key,value = string.split(separator)[0],to_number(separator.join(string.split(separator)[1:]))
		else:
			key,value = string,default
		value = default if value is None else value
		return key,value

	default = null
	separator = '='
	defaults = {
				'func':{'stat':{'':'mean','err':'sem'}}, 
				'wrapper':{},
				'attrs':{},
				'slice':None,
				'include':None,
				'axis':{'row':[],'col':[],'plot':['plot','group','func'],'axis':[-1]},
				'settings':{
					'ax.set_ylabel.ylabel':[['$\\textrm{Infidelity}$']],
					'ax.legend.update':'%s \\textrm{Noisy}'
				},
				'texify':{},
				'valify': {},		
	}

	elements = [*keys,*other]
	keys = brancher(dictionary,elements)
	
	keys = {name[:-1]:dict(zip(elements,[value[-1] for value in name[-1]])) for name in keys}

	for name in keys:
		for attr in keys[name]:
			
			if attr in other:

				if isinstance(keys[name][attr],dict):
					keys[name][attr] = {prop: keys[name][attr][prop] if keys[name][attr][prop] is not None else default for prop in keys[name][attr]}
				elif isinstance(keys[name][attr],str):
					keys[name][attr] = dict((parser(keys[name][attr],separator=separator,default=default),))
				else:
					keys[name][attr] = dict((parser(prop,separator=separator,default=default) for prop in keys[name][attr]))

				if attr in keys[name][attr]:
					if isinstance(keys[name][attr][attr],dict):
						keys[name][attr][attr] = {prop: keys[name][attr][attr][prop] if keys[name][attr][attr][prop] is not None else default for prop in keys[name][attr][attr]}
					elif isinstance(keys[name][attr][attr],str):
						keys[name][attr][attr] = dict((parser(keys[name][attr][attr],separator=separator,default=default)),)
					else:
						keys[name][attr][attr] = dict((parser(prop,separator=separator,default=default) for prop in keys[name][attr][attr]))
				else:
					keys[name][attr] = {attr: keys[name][attr]}

				setter(keys[name][attr],defaults,delimiter=delim,func=False)
			
			else:
				if not keys[name][attr]:
					keys[name][attr] = default
				elif isinstance(keys[name][attr],dict):
					keys[name][attr] = keys[name][attr][list(keys[name][attr])[-1]]
				elif isinstance(keys[name][attr],str):
					keys[name][attr] = keys[name][attr] if keys[name][attr] not in [''] else default
				else:
					keys[name][attr] = keys[name][attr][-1]

	return keys


def parse(key,value,data):
	'''
	Parse key and value condition for data, such that data[key] == value
	Args:
		key (str): key of condition
		value (str,iterable): value of condition, allowed string in 
			[None,
			'value,' (explicit value),
			'@key,@' (data value), 
			'#i,j,k,...#' (index value),
			'%start,stop,step%' (slice value),
			'<upper<' (exclusive upper bound value),
			'>lower>' (exclusive lower bound value),
			'<=upper<=' (inclusive upper bound value),
			'>=lower>=' (inclusive lower bound value),
			'==value,==' (include values),
			'!=value,!=' (exclude values),
			]
		data (dataframe): data of condition
	Returns:
		out (dataframe): Condition on data indices
	'''
	delimiters = ['$','@','#','%','<','>','<=','>=','==','!=']
	parserator = ';'
	separator = ','

	try:
		default = data.assign(**{key:True})[key]
	except:
		default = True

	out = default

	if key not in data:
		pass
	elif value is null:
		pass
	elif isinstance(value,str):
		outs = [default]
		for value in value.split(parserator):

			for delimiter in delimiters:

				if value.startswith(delimiter) and value.endswith(delimiter):
				
					values = value.replace(delimiter,'').split(separator)

					if delimiter in ['$']: # Explicit value: value
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           
						values = [value for value in values if (value is not null)]

						if values and (values is not null):
							try:
								out = data[key].isin(values)
							except:
								out = data[key] in values

					elif delimiter in ['@']: # Data value: key
						parser = lambda value: (to_str(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           
					  
						if values and (values is not null):
							out = conditions([data[key]==data[value] for value in values],op='or')

					elif delimiter in ['#']: # Index value: i,j,k,...
						parser = lambda value: (to_int(value) if len(value)>0 else null)
						values = [parser(value) for value in values]
						values = [value for value in values if (value is not null)]

						if values and (values is not null):
							try:
								out = data[key].unique()
								out = data[key].isin(out[[value for value in values if value < out.size]])
							except:
								out = not default

					elif delimiter in ['%']: # Slice value start,stop,step
						parser = lambda value: (to_int(value) if len(value)>0 else None)
						values = [*(parser(value) for value in values),*[None]*(3-len(values))]
						values = [value for value in values if (value is not null)]
				
						if values and (values is not null):
							try:
								out = data[key].isin(data[key].unique()[slice(*values)])
							except:
								out = not default

					elif delimiter in ['<']: # Bound value: upper (exclusive)
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           
					  
						if values and (values is not null):
							out = conditions([data[key] < value for value in values],op='and')

					elif delimiter in ['<=']: # Bound value: upper (inclusive)
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           
					  
						if values and (values is not null):
							out = conditions([data[key] <= value for value in values],op='and')

					elif delimiter in ['>']: # Bound value: lower (exclusive)
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           
					  
						if values and (values is not null):
							out = conditions([data[key] > value for value in values],op='and')

					elif delimiter in ['>=']: # Bound value: lower (inclusive)
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           
					  
						if values and (values is not null):
							out = conditions([data[key] >= value for value in values],op='and')

					elif delimiter in ['==']: # Include value
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           
						
						if values and (values is not null):
							out = conditions([data[key] == value for value in values],op='and')																												

					elif delimiter in ['!=']: # Exclude value
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           

						if values and (values is not null):
							out = conditions([data[key] != value for value in values],op='and')																												

	
					outs.append(out)
					break

		out = conditions(outs,op='and')

	else:
		try:
			out = data[key].isin(value)
		except:
			try:
				out = data[key] in value
			except:
				out = not default
	
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

	if (keys is None) or (data is None):
		return

	if not hyperparameters['process']:
		return

	def mean(obj):
		out = np.array(list(obj))
		out = tuple(out.mean(0))
		return out
	def sem(obj):
		out = np.array(list(obj))
		out = tuple(out.std(0)/np.sqrt(out.shape[0]))
		return out		

	functions = {}			
	dtypes = {attr: ('array' if any(isinstance(i,tuple) for i in data[attr]) else 'object' if data[attr].dtype.kind in ['O'] else 'dtype') 
				for attr in data}

	for name in keys:

		if any((keys[name][axis] not in data) and (keys[name][axis] is not null) for axis in AXIS if axis in keys[name]):
			continue

		axes = [axis for axis in AXIS if axis in keys[name]]
		other = OTHER
		label = keys[name][other].get(other,{})
		funcs = keys[name][other].get('func',{})

		if not funcs:
			funcs = {'stat':{'':'mean','err':'sem'}}

		funcs = {function : {func: functions.get(funcs[function][func],funcs[function][func]) for func in funcs[function]} for function in funcs}

		independent = [keys[name][axis] for axis in axes[:-1] if keys[name][axis] in data]
		dependent = [keys[name][axis] for axis in axes[-1:] if keys[name][axis] in data]
		labels = [attr for attr in label if attr in data and label[attr] is null]
		boolean = [parse(attr,label[attr],data) for attr in label]
		boolean = conditions(boolean,op='and')	

		by = [*labels,*independent]
		
		groupby = data[boolean].groupby(by=by,as_index=False)

		agg = {
			**{attr : [(attr, {'array':mean,'object':'first','dtype':'mean'}[dtypes[attr]] if attr not in by else {'array':'first','object':'first','dtype':'first'}[dtypes[attr]])] for attr in data},
			**{attr : [(delim.join(((attr,function,func))),{'array':{'':mean,'err':sem}[func],'object':'first','dtype':funcs[function][func]}[dtypes[attr]]) for function in funcs for func in funcs[function]] for attr in data if attr in dependent},
		}

		droplevel = dict(level=0,axis=1)
		by = [*labels]
		variables = [*independent,*dependent,*[subattr[0] for attr in dependent for subattr in agg[attr]]]

		groups = groupby.agg(agg).droplevel(**droplevel)

		if by:
			groups = groups.groupby(by=by,as_index=False)
		else:
			groups = GroupBy(groups,by=by)

		assert all(groups.get_group(group).columns.nlevels == 1 for group in groups.groups) # Possible future broken feature agg= (label,name)

		for i,group in enumerate(groups.groups):
			for j,function in enumerate(funcs):

				grouping = groups.get_group(group)
				
				key = (*name[:-2],i,j)
				value = deepcopy(getter(settings,name,delimiter=delim))

				source = [attr for attr in data if attr not in variables]
				destination = other
				value[destination] = {
					**{attr: grouping[attr].to_list()[0] for attr in source},
					**{'%s%s'%(axis,func) if keys[name][axis] in dependent else axis: 
						{'group':[i,dict(zip(groups.grouper.names,group))],'func':[j,function],'axis':keys[name][axis] if keys[name][axis] is not null else None} 
						for axis in axes for func in funcs[function]},
					**{other: {attr: {subattr: keys[name][other][attr][subattr] 
						if keys[name][other][attr][subattr] is not null else None for subattr in keys[name][other][attr]}
						if isinstance(keys[name][other][attr],dict) else keys[name][other][attr] for attr in keys[name][other]}},
					}

				for func in funcs[function]:	
					for axis in axes:
						
						attr = keys[name][axis]

						source = delim.join(((attr,function,func))) if attr in dependent else attr
						destination = '%s%s'%(axis,func) if attr in dependent else axis

						if grouping.shape[0]:
							if source in grouping:
								if dtypes[attr] in ['array']:
									value[destination] = np.array(grouping[source].iloc[0])
								else:
									value[destination] = grouping[source].to_numpy()
							elif source is null:
								source = delim.join(((dependent[-1],function,func)))
								value[destination] = np.arange(len(grouping[source].iloc[0]))
							else:
								value[destination] = grouping.reset_index().index.to_numpy()

						else:
							value[destination] = None

				setter(settings,{key:value},delimiter=delim,func=True)

	return

def loader(data,settings,hyperparameters):
	'''
	Load data from settings and hyperparameters
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		settings (str,dict): Path to or dictionary of plot settings
		hyperparameters (str,dict): Path to or dictionary of process settings
	'''

	if (data is None) or (settings is None) or (hyperparameters is None):
		return

	# Get keys
	keys = find(settings)

	# Set metadata
	metadata = hyperparameters['path']['metadata']
	
	def func(key_iterable,key_elements,iterable,elements):

		if (
			(key_iterable == key_elements) and 
			(key_iterable in PLOTS) and (key_elements in PLOTS) and 
			isinstance(iterable.get(key_iterable),list) and	isinstance(elements.get(key_elements),list)
			):

			def parser(string,separator,default):
				if string.count(separator):
					key,value = string.split(separator)[0],to_number(separator.join(string.split(separator)[1:]))
				else:
					key,value = string,default
				value = default if value is None else value
				return key,value

			default = None
			separator = '='


			for index,data in enumerate(flatten(elements.get(key_elements))):
				for subindex,datum in enumerate(flatten(iterable.get(key_iterable)[index])):
					datum.update({attr: data[attr] for attr in data if attr not in [*ALL,OTHER]})

					attr = OTHER

					if (data.get(attr) is None) or (attr not in datum[attr]):
						continue
					elif isinstance(data[attr],dict) and attr in data[attr]:
						if attr in datum[attr][attr]:
							datum[attr][attr][attr] = data[attr][attr]
							datum[attr][attr].update({prop: data[attr][prop] for prop in data[attr] if prop not in [attr]})
						else:
							datum[attr][attr] = data[attr][attr]
							datum[attr].update({prop: data[attr][prop] for prop in data[attr] if prop not in [attr]})
					elif isinstance(data[attr],dict) and attr not in data[attr]:
						if attr in datum[attr][attr]:
							datum[attr][attr][attr] = data[attr]
						else:
							datum[attr][attr] = data[attr]
					elif isinstance(data[attr],str):
						continue
					else:
						if attr in datum[attr][attr]:
							datum[attr][attr][attr] = {prop: None for prop in data[attr]}
						else:
							datum[attr][attr] = {prop: None for prop in data[attr]}

			out = iterable.get(key_iterable)
		else:
			out = elements.get(key_elements)
		return out	

	if hyperparameters['load']:


		# Load settings
		path = metadata
		default = {}
		tmp = deepcopy(settings)

		settings.update(load(path,default=default))
		setter(settings,tmp,func=func)

	else:

		# Load data
		path = data
		default = None
		wrapper = 'df'
		data = load(path,default=default,wrapper=wrapper)

		# Get functions of data
		apply(keys,data,settings,hyperparameters)

	
	# Dump settings
	if hyperparameters['dump']:
		path = metadata
		
		dump(settings,metadata)


	return

def plotter(settings,hyperparameters):
	'''
	Plot data based plot settings, process hyperparameters
	Args:
		settings (dict): settings
		hyperparameters (dict): hyperparameters
	'''

	if (settings is None) or (hyperparameters is None):
		return

	if not hyperparameters['plot']:
		return

	# Variables
	path = hyperparameters['path']['plot']
	fig = hyperparameters['fig']
	ax = hyperparameters['ax']
	texify = hyperparameters['texify']
	valify = hyperparameters['valify']

	# Set layout
	layout = {}
	for instance in settings:
		for index,subinstance in enumerate(settings[instance]):
			sublayout = settings[instance][subinstance]['style']['layout']
			if not layout.get(instance):
				layout[instance] = sublayout
			layout[instance].update({
				**layout[instance],
				**{attr: max(sublayout[attr],layout[instance][attr]) 
					if (sublayout[attr] is not None) and (layout[instance][attr] is not None) else None
					for attr in ['nrows','ncols']},
				**{attr: None for attr in ['index']},
				})
		for index,subinstance in enumerate(settings[instance]):
			sublayout = deepcopy(layout[instance])

			indx = sublayout['index']-1	if sublayout['index'] is not None else index
			nrow = (indx - indx%sublayout['ncols'])//sublayout['ncols']
			ncol = indx%sublayout['ncols']

			sublayout.update({
				**{'index':index+1},
				**{
					'top':1 - (nrow)/sublayout['nrows'] if sublayout['top'] and sublayout['nrows']>1 else None,
					'bottom':1 - (nrow+1)/sublayout['nrows'] if sublayout['bottom'] and sublayout['nrows']>1 else None,
					'right':(ncol+1)/sublayout['ncols'] if sublayout['right'] and sublayout['ncols']>1 else None,
					'left':(ncol)/sublayout['ncols'] if sublayout['left'] and sublayout['ncols']>1 else None,											
					}
				})


			settings[instance][subinstance]['style']['layout'] = sublayout


	# Set data
	for instance in list(settings):
		for subinstance in list(settings[instance]):


			# variables
			attrs = OTHER
			try:
				values = {
					plots: {
						label: {
							'value': list(realsorted(set(data[attrs][label] if not isinstance(data[attrs][label],list) else tuple(data[attrs][label])
								for data in flatten(settings[instance][subinstance]['ax'][plots]) if label in data[attrs]))),
							'sort': list(realsorted(set(data[attrs][attrs][attrs][label]
								for data in flatten(settings[instance][subinstance]['ax'][plots]) if label in data[attrs][attrs][attrs]))),
							'label': any(((label in data[attrs][attrs][attrs]) and (label in data[attrs]) and (data[attrs][attrs][attrs][label] is None))
								for data in flatten(settings[instance][subinstance]['ax'][plots])),
							'other': any(((label not in data[attrs]) and (label in data[attrs][attrs][attrs]) and (data[attrs][attrs][attrs][label] in data[attrs]))
								for data in flatten(settings[instance][subinstance]['ax'][plots])),
							'legend': any(((label not in data[attrs]) and (label in data[attrs][attrs][attrs]) and (data[attrs][attrs][attrs][label] not in data[attrs]))
								for data in flatten(settings[instance][subinstance]['ax'][plots])),
							}
						for label in list(realsorted(set(label
						for data in flatten(settings[instance][subinstance]['ax'][plots])
						for label in [*data[attrs],*data[attrs][attrs][attrs]]
						if ((label not in [*ALL,OTHER])))))
						}
						for plots in PLOTS 
						if plots in settings[instance][subinstance]['ax']
						}
			except KeyError as e:
				settings[instance].pop(subinstance);
				continue
				
			# savefig
			attr = 'fname'
			data = settings[instance][subinstance]['fig'].get('savefig',{})
			value = join(delim.join([split(path,directory_file=True),instance]),ext=split(path,ext=True))
			data[attr] = value

			# legend
			attr = 'set_title'
			data = settings[instance][subinstance]['ax'].get('legend',{})

			
			value = '~,~'.join(realsorted(set((
				texify(label) for plots in values for label in values[plots] 
					if (((values[plots][label]['label']) and (len(values[plots][label]['value'])>1)) or 
						(values[plots][label]['other'])))))
				)

			data[attr] = value

			# data
			for plots in PLOTS:

				if settings[instance][subinstance]['ax'].get(plots) is None:
					continue

				for data in flatten(settings[instance][subinstance]['ax'][plots]):

					for attr in data:
						if (attr in ALL) and (data[attr] is not None):
							value = np.array([valify(value) for value in data[attr]])

							if data[OTHER][OTHER].get('slice') is not None:
								slices = slice(*data[OTHER][OTHER].get('slice'))
								value = value[slices]

							data[attr] = value


			# include
			for plots in PLOTS:

				if settings[instance][subinstance]['ax'].get(plots) is None:
					continue

				for data in flatten(settings[instance][subinstance]['ax'][plots]):

					attr = OTHER
					if data[attr][attr].get('include') is not None:
						for label in data[attr][attr]['include']:
							if not parse(label,data[attr][attr]['include'][label],data[attr]):
								data.clear()
								break


			# label
			for plots in PLOTS:

				if settings[instance][subinstance]['ax'].get(plots) is None:
					continue

				for data in flatten(settings[instance][subinstance]['ax'][plots]):

					if not data:
						continue

					attr = OTHER
					value = ', '.join([
						*[(texify(scinotation(data[attr][label],decimals=0,scilimits=[0,3],one=False)) 
						if values[plots][label]['label'] else texify(scinotation(data[attr][data[attr][attr][attr][label]],decimals=0,scilimits=[0,3],one=False)))
						for label in values[plots]
						if (((values[plots][label]['label']) and (len(values[plots][label]['value'])>1)) or 
							(values[plots][label]['other']))],
						*[texify(scinotation(data[attr][attr][attr][label],decimals=0,scilimits=[0,3],one=False)) 
						for label in values[plots]
						if values[plots][label]['legend']
						]
						])

					data[attr] = value	




	# Plot data
	for instance in settings:

		if (not hyperparameters.get('instance',{}).get(instance)) or (not settings[instance]):
			continue

		print("Plotting : %s"%(instance))

		fig[instance],ax[instance] = plot(fig=fig[instance],ax=ax[instance],settings=settings[instance])

	return


def postprocessor(hyperparameters,pwd=None,cwd=None):
	'''
	Postprocess data
	Args:
		hyperparameters (str,dict): Path to or dictionary of process settings
		pwd (str): Root path of data
		cwd (str): Root path of plots
	'''

	if not hyperparameters['postprocess']:
		return

	path = cwd
	postprocess(path)

	return


def process(data,settings,hyperparameters,pwd=None,cwd=None):
	'''
	Process data
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		settings (str,dict): Path to or dictionary of plot settings
		hyperparameters (str,dict): Path to or dictionary of process settings
		pwd (str): Root path of data
		cwd (str): Root path of plots

	Steps:
	- Load data and settings
	- Get data axes and labels based on branches of settings
	- Iterate over all distinct data branches of settings
	
	- Filter with booleans of all labels
	- Group by non-null labels and independent
	- Aggregate functions of dependent (mean,sem) for each group
	- Assign new labels for functions with label.function 
	- Regroup with non-null labels
	
	- Reshape each data into axes for [plot.type,plot.row,plot.col,plot.line=(plot.group,plot.function),plot.axis]
	- Adjust settings based on data
	
	- Plot data

	To process data, we find in plot settings dictionary the keys of ALL,OTHER ('x','y','label') properties for sorting.

	For OTHER property with attributes and values to sort on, 
	datasets are sorted into sets of unique datasets that correspond 
	to all possible combinations of the label values. i.e) if label is ('M','N'), 
	will sort into sets of datasets that correspond to each possible 'M','N' pair. 
	
	For each combination of specific label values, statistics about the set of sample datasets 
	corresponding to the label values are computed/
	
	Label attributes with a non-None value indicate fixed values that the datasets must equal for that attribute
	when sorting, and the set of datasets for that label combination are constrained to be those with the fixed values.

	- Iterate over all combinations of OTHER attributes and values, to get included dataset that share these attributes
	
	- Iterate over all permutations of sort attributes and values, constrained by the specific combination of OTHER attributes and values
	to get included dataset that share all sort attributes
	
	- Get statistics (mean,variance) across samples datasets that share all attributes

	- Merge datasets across permutations of sort attributes for a given combination of OTHER attributes and values
	
	- After statistics and merging, variables data for each combination for OTHER attributes and values
	has shape of (1 + ndim) dimensions, with axes: 
	(# of permutations of sort for each combination, ndim of datasets)	

	- For parameters, we plot each combination for OTHER attributes and values variables data on the same plot, these being the labels.
	  Subplots are plotted from iterating over over the 1...ndim-2 axis of the variables data, and parameters the 0 and ndim-1 axis for each OTHER set
	  If the 'x' property is None, also iterate over the 0 (# of permutations of sort) axis variables data, and plot the ndim-1 axis for each OTHER 
	'''

	# Set settings and hyperparameters
	data,settings,hyperparameters = setup(data,settings,hyperparameters,pwd,cwd)

	# Load data
	loader(data,settings,hyperparameters)

	# Plot data
	plotter(settings,hyperparameters)

	# Post process data
	postprocessor(hyperparameters,pwd,cwd)

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
