#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,traceback
from copy import deepcopy
import numpy as np
import scipy as sp
import scipy.stats
import scipy.special
import pandas as pd
from natsort import natsorted,realsorted

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.system	 import Logger
name = __name__
path = os.path.dirname(__file__) #os.getcwd()
file = 'logging.conf'
conf = os.path.join(path,file)
file = None #'log.log'
info = 100
debug = 0
logger = Logger(name,conf,file=file)

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



DIM = 2

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
	if texify is None:
		texify = {}
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
	if valify is None:
		valify = {}
	values = {
		**valify,
		**{to_number(value): valify[value] for value in valify},
		None: valify.get('None',None),
		'None': valify.get('None',None),
		}

	try:
		value = valify.get(value,value)
	except:
		pass

	return value


def setup(data,settings,hyperparameters,pwd=None,cwd=None,verbose=None):
	'''
	Setup data, settings, hyperparameters
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		settings (str,dict): Path to or dictionary of plot settings
		hyperparameters (str,dict): Path to or dictionary of process settings
		pwd (str): Root path of data
		cwd (str): Root path of plots
		verbose (bool): Verbosity		
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


	logger.log(info*verbose,'Paths: pwd: %s , cwd: %s'%(pwd,cwd))

	# Load plot settings
	path = join(settings,root=pwd) if isinstance(settings,str) else None
	default = None if isinstance(settings,str) else settings
	wrapper = None	
	settings = load(path,default=default,wrapper=wrapper,verbose=verbose)

	# Load process hyperparameters
	path = join(hyperparameters,root=pwd) if isinstance(hyperparameters,str) else None
	default = None if isinstance(hyperparameters,str) else hyperparameters
	wrapper = None
	hyperparameters = load(path,default=default,wrapper=wrapper,verbose=verbose)

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
	hyperparameters['texify'] = lambda string,texify=None,_texify=texify,usetex=usetex: Texify(
		string,
		texify={**(_texify if _texify is not None else {}),**(texify if texify is not None else {})},
		usetex=usetex)

	# Get valify
	valify = hyperparameters.get('valify',{})
	useval = hyperparameters.get('useval',True)
	hyperparameters['valify'] = lambda value,valify=None,_valify=valify,useval=useval: Valify(
		value,
		valify={**(_valify if _valify is not None else {}),**(valify if valify is not None else {})},
		useval=useval)

	return data,settings,hyperparameters

def find(dictionary,verbose=None):
	'''
	Find formatted keys from dictionary, based on search keys of the form 'property':attr
	All keys are assumed to be present in any branches where one or more property is found in dictionary
	Args:
		dictionary (dict): Dictionary to search
		verbose (bool): Verbosity
	Returns:
		keys (dict[dict]): Formatted keys based on found keys of the form {name: {prop:attr} or {prop:{attr:value}}}
	'''

	axes = AXIS[:DIM]
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
				'include':None,
				'exclude':None,
				'slice':None,
				'labels':None,
				'analysis':{
					# 'zscore':[{'attr':['objective'],'default':None,'kwargs':{'sigma':None}}],
					# 'quantile':[{'attr':['objective'],'default':None,'kwargs':{'sigma':None}}]
					# 'parse':[{'attr':{'__path__':'*','M':"<600<"},'default':None,'kwargs':{'sigma':None}}]
					},
				'axis':{'row':[],'col':[],'plot':['plot','group','func'],'axis':[-1]},
				'settings':{},
				'texify':{},
				'valify': {},		
				'scinotation':{'scilimits':[0,2],'decimals':0,'one':False},
	}

	elements = [*axes,*other]
	keys = brancher(dictionary,elements)
	
	keys = {name[:-1]:dict(zip(elements,[value[-1] for value in name[-1]])) for name in keys}


	for name in keys:
		for attr in keys[name]:
			
			if attr in other:

				if isinstance(keys[name][attr],dict):
					if attr in keys[name][attr]:
						keys[name][attr] = {prop: keys[name][attr][prop] if (prop != attr) or (keys[name][attr][prop] is not None) else default for prop in keys[name][attr]}
					else:
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


def parse(key,value,data,verbose=None):
	'''
	Parse key and value condition for data, such that data[key] == value
	Args:
		key (str): key of condition
		value (str,iterable): value of condition, allowed string in 
			[None,
			'$value,$' (explicit value),
			'@key,@' (data value), 
			'#i,j,k,...#' (index value),
			'%start,stop,step%' (slice value),
			'*pattern,*' (regex pattern),
			'<upper<' (exclusive upper bound value),
			'>lower>' (exclusive lower bound value),
			'<=upper<=' (inclusive upper bound value),
			'>=lower>=' (inclusive lower bound value),
			'==value,==' (include values),
			'!=value,!=' (exclude values),
			]
		data (dataframe): data of condition
		verbose (bool): Verbosity		
	Returns:
		out (dataframe): Condition on data indices
	'''
	delimiters = ['$','@','#','%','*','<','>','<=','>=','==','!=']
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
				
					values = value[len(delimiter):-len(delimiter)].split(separator)

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

					elif delimiter in ['*']: # Regex value pattern
						def parser(value):
							replacements = {'.':r'\.','*':'.'}							
							if not len(value):
								return value
							value = r'%s'%(to_str(value))
							for replacement in replacements:
								value = value.replace(replacement,replacements[replacement])
							value = r'%s'%(value)
							return value

						values = [parser(value) for value in values if (value is not None)]
						values = [value for value in values if (value is not None)]
				
						if values and (values is not null):
							out = conditions([data[key].str.contains(r'%s'%(value)) for value in values],op='or')

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
		if not isinstance(value,list):
			value = [value]
		try:
			out = data[key].isin(value)
		except Exception as exception:
			try:
				out = data[key] in value
			except:
				out = not default
	
	return out


def analyse(data,analyses=None,verbose=None):
	'''
	Analyse data, cleaning data, removing outliers etc
	Args:
		data (dataframe): data of attributes
		analyses (dict[str,dict]): Processes to analyse of the form 
			{analysis:[{'attr':{attr:value},[attr],'default':value,'kwargs':{kwarg:value}}]},
			allowed analysis strings in ['zscore','quantile','parse']
		verbose (bool): Verbosity			
	Returns:
		out (dataframe): Condition on data indices
	'''

	default = True

	out = default

	if analyses is not None:
		for analysis in analyses:
			if analysis in ['zscore']:
				def func(attr,attrs,data,default,**kwargs):
					reverse = kwargs.pop('reverse',None)					
					function = sp.stats.zscore
					sigma = kwargs.pop('sigma',None)
					out = data[[attr]].apply(function,**kwargs)[attr]
					out = (out < sigma) if sigma is not None else default
					out = ~out if reverse else out
					return out
			elif analysis in ['quantile']:
				def func(attr,attrs,data,default,**kwargs):
					reverse = kwargs.pop('reverse',None)					
					sigma = kwargs.pop('sigma',None)
					out = [data[attr].quantile(sigma),data[attr].quantile(1-sigma)] if sigma is not None else default
					out = ((data[attr] > out[0]) & (data[attr] < out[1])) if sigma is not None else default
					out = ~out if reverse else out					
					return out
			elif analysis in ['parse']:
				def func(attr,attrs,data,default,**kwargs):
					reverse = kwargs.pop('reverse',None)
					out = [parse(attr,attrs[attr],data,verbose=verbose) for attr in attrs]
					if reverse:
						out = conditions([~i for i in out],op='or')
					else:
						out = conditions([i for i in out],op='and')
					return out
			else:
				continue

			if isinstance(analyses[analysis],dict):
				args = [analyses[analysis]]
			else:
				args = analyses[analysis]

			for arg in args:
				attrs = arg.get('attr',{})
				default = arg.get('default',None)
				kwargs = arg.get('kwargs',{})

				value = [func(attr,attrs,data,default,**kwargs) for attr in attrs if attr in data]			

				out = conditions([out,*value],op='and')


	if out is True:
		out = data
	else:
		out = data[out]

	return out


def apply(keys,data,settings,hyperparameters,verbose=None):
	'''
	Apply functions based on keys to data
	Args:
		keys (dict): Keys of functions to apply
		data (dataframe): dataframe
		settings (dict): settings
		hyperparameters (dict): hyperparameters
		verbose (bool): Verbosity		
	'''

	if (keys is None) or (data is None) or (settings is None) or (hyperparameters is None):
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

		logger.log(info,"Processing : %r"%(name,))

		if any((keys[name][axis] not in data) and (keys[name][axis] is not null) for axis in AXIS if axis in keys[name]):
			key,value = name,None
			setter(settings,{key:value},delimiter=delim,func=True)
			continue

		axes = [axis for axis in AXIS if axis in keys[name]]
		other = OTHER
		label = keys[name][other].get(other,{})
		include = keys[name][other].get('include')
		exclude = keys[name][other].get('exclude')
		funcs = keys[name][other].get('func',{})
		analyses = keys[name][other].get('analysis',{})

		if not funcs:
			funcs = {'stat':{'':'mean','err':'sem'}}

		funcs = {function : {func: functions.get(funcs[function][func],funcs[function][func]) for func in funcs[function]} for function in funcs}

		independent = [keys[name][axis] for axis in axes[:-1] if keys[name][axis] in data]
		dependent = [keys[name][axis] for axis in axes[-1:] if keys[name][axis] in data]
		labels = [attr for attr in label if (attr in data) and (((label[attr] is null) and (exclude is None) and (include is None)) or ((exclude is not None) and (attr not in exclude))) or ((include is not None) and (attr in include))]
		boolean = [parse(attr,label[attr],data,verbose=verbose) for attr in label]
		boolean = conditions(boolean,op='and')

		by = [*labels,*independent]

		groupby = data[boolean].groupby(by=by,as_index=False)

		groupby = groupby.apply(analyse,analyses=analyses,verbose=verbose).reset_index(drop=True).groupby(by=by,as_index=False)
		
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

			logger.log(info,"Group : %r %r"%(group,groups.get_group(group).shape))

			for j,function in enumerate(funcs):

				grouping = groups.get_group(group)
				
				key = (*name[:-2],i,j)
				value = deepcopy(getter(settings,name,delimiter=delim))

				source = [attr for attr in data if attr not in variables]
				destination = other
				value[destination] = {
					**{attr: grouping[attr].to_list()[0] for attr in source},
					**{'%s%s'%(axis,func) if keys[name][axis] in dependent else axis: 
						{'group':[i,dict(zip(groups.grouper.names,group if isinstance(group,tuple) else (group,)))],'func':[j,function],'axis':keys[name][axis] if keys[name][axis] is not null else None} 
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

def loader(data,settings,hyperparameters,verbose=None):
	'''
	Load data from settings and hyperparameters
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		settings (str,dict): Path to or dictionary of plot settings
		hyperparameters (str,dict): Path to or dictionary of process settings
		verbose (bool): Verbosity		
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
				if index >= len(iterable.get(key_iterable)):
					# iterable[key_iterable].append(deepcopy(data))
					continue
				for subindex,datum in enumerate(flatten(iterable.get(key_iterable)[index])):
					if not datum:
						continue
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

		settings.update(load(path,default=default,verbose=verbose))
		setter(settings,tmp,func=func)

	else:

		# Load data
		path = data
		default = None
		wrapper = 'df'
		data = load(path,default=default,wrapper=wrapper,verbose=verbose)

		# Get functions of data
		apply(keys,data,settings,hyperparameters,verbose=verbose)

	
	# Dump settings
	if hyperparameters['dump']:
		path = metadata
		
		dump(settings,metadata,verbose=verbose)

	return

def plotter(settings,hyperparameters,verbose=None):
	'''
	Plot data based plot settings, process hyperparameters
	Args:
		settings (dict): settings
		hyperparameters (dict): hyperparameters
		verbose (bool): Verbosity		
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
	for instance in list(settings):

		if (not hyperparameters.get('instance',{}).get(instance)) or (not settings[instance]):
			settings.pop(instance,None);
			continue

		logger.log(info*verbose,"Setting : %s"%(instance))

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
			try:
				values = {
					plots: {
						label: {
							'value': list(realsorted(set(
								data[OTHER][label] if (
									(label in data[OTHER]) and not isinstance(data[OTHER][label],list)) else 
								tuple(data[OTHER][label]) if (
									(label in data[OTHER])) else 
								data[OTHER][data[OTHER][OTHER][OTHER][label].replace('@','')] if (
									(label in data[OTHER][OTHER][OTHER] and 
									(data[OTHER][OTHER][OTHER].get(label) is not None) and
									data[OTHER][OTHER][OTHER][label].replace('@','') in data[OTHER])) else data[OTHER][OTHER][OTHER][label] if (label in data[OTHER][OTHER][OTHER]) else None
								for data in flatten(settings[instance][subinstance]['ax'][plots]) if (
									((data) and ((label in data[OTHER]) or (label in data[OTHER][OTHER][OTHER]))))
								))),
							'sort': list(realsorted(set(data[OTHER][OTHER][OTHER][label]
								for data in flatten(settings[instance][subinstance]['ax'][plots]) if ((data) and (label in data[OTHER][OTHER][OTHER]))
								))),
							'label': any((
								(label in data[OTHER][OTHER][OTHER]) and 
								(label in data[OTHER]) and (data[OTHER][OTHER][OTHER][label] is None))
								for data in flatten(settings[instance][subinstance]['ax'][plots]) 
								if (data)
								),
							'other': any((
								(label not in data[OTHER]) and 
								(label in data[OTHER][OTHER][OTHER]) and 
								((data[OTHER][OTHER][OTHER].get(label) is not None) and
								(data[OTHER][OTHER][OTHER][label].replace('@','') in data[OTHER])))
								for data in flatten(settings[instance][subinstance]['ax'][plots])
								if (data)
								),
							'legend': any((
								(label not in data[OTHER]) and 
								(label in data[OTHER][OTHER][OTHER]) and
								((data[OTHER][OTHER][OTHER].get(label) is not None) and
								(data[OTHER][OTHER][OTHER][label].replace('@','') not in data[OTHER])))
								for data in flatten(settings[instance][subinstance]['ax'][plots])
								if (data)
								),
							'attr': {
								**{attr: {string:  data[OTHER][OTHER][attr][string]
									for data in flatten(settings[instance][subinstance]['ax'][plots]) 
									if ((data) and attr in data[OTHER][OTHER])
									for string in data[OTHER][OTHER][attr]}
									for attr in ['texify','valify']},
								**{attr: {
									**{kwarg:[
									min((data[OTHER][OTHER][attr][kwarg][0]
										for data in flatten(settings[instance][subinstance]['ax'][plots]) 
										if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))),
										default=0),
									max((data[OTHER][OTHER][attr][kwarg][1]
										for data in flatten(settings[instance][subinstance]['ax'][plots]) 
										if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))),
										default=0),											
									] for kwarg in ['scilimits']},
									**{kwarg: 
										max((data[OTHER][OTHER][attr][kwarg]
										for data in flatten(settings[instance][subinstance]['ax'][plots]) 
										if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))),
										default=0) 
										for kwarg in ['decimals']},
									**{kwarg: 
										any((data[OTHER][OTHER][attr][kwarg]
										for data in flatten(settings[instance][subinstance]['ax'][plots]) 
										if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))))
										for kwarg in ['one']},										
									}
									for attr in ['scinotation']},
								},
							}
						for label in list(realsorted(set(label
						for data in flatten(settings[instance][subinstance]['ax'][plots])
						if (data)
						for label in [*data[OTHER],*data[OTHER][OTHER][OTHER]]
						if ((data) and (label not in [*ALL,OTHER]) and (
							((data[OTHER][OTHER].get('exclude') is None) or (label not in data[OTHER][OTHER]['exclude'])) and 
							((data[OTHER][OTHER].get('include') is None) or (label in data[OTHER][OTHER]['include'])))  
						))))
						}
						for plots in PLOTS 
						if plots in settings[instance][subinstance]['ax']
						}
				_values = {}
				for plots in list(values):
					if plots not in _values:
						_values[plots] = {}
					for label in list(values[plots]):
						if not any(label in _values[_plots] for _plots in _values):
							_values[plots][label] = values[plots][label]
				values = _values
			except KeyError as e:
				# logger.log(debug,traceback.format_exc(),instance,subinstance)
				settings[instance].pop(subinstance);
				continue

			# savefig
			attr = 'fname'
			data = settings[instance][subinstance]['fig'].get('savefig',{})
			value = join(delim.join([split(path,directory_file=True),instance]),ext=split(path,ext=True))
			data[attr] = value


			# colorbar
			attr = 'set_colorbar'
			data = settings[instance][subinstance]['ax'].get(attr)
			if data is not None:

				subattr = 'values'
				label = data.get(subattr)

				value = []

				for plots in values:

					if label not in values[plots]:
						continue
					else:
						value.append(deepcopy(data))

					subattr = 'values'
					if isinstance(label,str):
						subvalue = list(realsorted(set([i for i in values[plots][label]['value']])))

						subvalue = subvalue if len(subvalue) >= 1 else None

						value[-1][subattr] = subvalue

					subattr = 'set_%slabel'
					subsubattr = '%slabel'
					for axis in ['',*AXIS]:
						subvalue = value[-1].get(subattr%(axis))
						if subvalue is None:
							continue
						value[-1][subattr%(axis)][subsubattr%(axis)] = texify(
							subvalue.get(subsubattr%(axis)),
							texify=values[plots][label]['attr']['texify']) 

					subattr = 'set_%sticks'
					subsubattr = 'ticks'
					for axis in ['',*AXIS]:
						subvalue = value[-1].get(subattr%(axis))
						if subvalue is None:
							continue
						else:
							if isinstance(subvalue.get(subsubattr),int):
								subsubvalue = list(realsorted(set([i for i in values[plots][label]['value']])))
								subsubvalue = subsubvalue[::len(subsubvalue)//subvalue.get(subsubattr)]
							else:
								subsubvalue = subvalue.get(subsubattr)
							subvalue[subsubattr] = subsubvalue

						value[-1][subattr%(axis)][subsubattr] = [i for i in subvalue[subsubattr]]

					subattr = 'set_%sticklabels'
					subsubattr = 'labels'
					for axis in ['',*AXIS]:
						subvalue = value[-1].get(subattr%(axis))
						if subvalue is None:
							continue
						else:
							if isinstance(subvalue.get(subsubattr),int):
								subsubvalue = list(realsorted(set([i for i in values[plots][label]['value']])))
								subsubvalue = subsubvalue[::len(subsubvalue)//subvalue.get(subsubattr)]
							elif subvalue.get(subsubattr) is not None:
								subsubvalue = subvalue.get(subsubattr)
							else:
								subsubvalue = value[-1].get('set_%sticks'%(axis),{}).get('ticks')

							subvalue[subsubattr] = subsubvalue

						value[-1][subattr%(axis)][subsubattr] = [
							texify(
							scinotation(i,
								**values[plots][label]['attr']['scinotation']),
							texify=values[plots][label]['attr']['texify']) for i in subvalue[subsubattr]]							

					settings[instance][subinstance]['ax'][attr] = value[-1]

					break

				if not value:
					settings[instance][subinstance]['ax'][attr] = None

			# legend
			attr = 'set_title'
			data = settings[instance][subinstance]['ax'].get('legend')

			if data is not None:
				value = [
					[
						*['%s'%(texify(label,texify=values[plots][label]['attr']['texify'])) 
							for plots,label in realsorted(set((
							(plots,label)
							for plots in values 					
							for label in values[plots] 
							if (((values[plots][label]['label']) and (len(values[plots][label]['value'])>1)) and 
								not (values[plots][label]['other'])))))],
						*['%s'%(texify(label,texify=values[plots][label]['attr']['texify']))
							for plots,label in realsorted(set((
							(plots,label)
							for plots in values 
							for label in values[plots]
							if (not ((values[plots][label]['label'])) and 
								(values[plots][label]['other']) and (len(values[plots][label]['value'])>1)))))],
						*['%s'%(texify(label,texify=values[plots][label]['attr']['texify']))
							for plots,label in realsorted(set((
							(plots,label)
							for plots in values 
							for label in values[plots]
							if (not ((values[plots][label]['label'])) and 
								(values[plots][label]['legend']) and (len(values[plots][label]['value'])>1)))))],
						],
					[
						*['%s%s%s'%(
							texify(label),' : ' if label else '',
							',~'.join([texify(scinotation(value,**values[plots][label]['attr']['scinotation']),texify=values[plots][label]['attr']['texify']) 
									for value in values[plots][label]['value']]))
							for plots in values 
							for label in realsorted(set((
							label 
							for label in values[plots]
							if (not ((values[plots][label]['label'])) and 
								(values[plots][label]['other']) and (len(values[plots][label]['value'])==1)))))],
						*['%s%s%s'%(
							texify(label),' : ' if label else '',
							',~'.join([texify(scinotation(value,**values[plots][label]['attr']['scinotation']),texify=values[plots][label]['attr']['texify']) 
									for value in values[plots][label]['value']]))
							for plots in values 
							for label in realsorted(set((
							label 
							for label in values[plots]
							if (not ((values[plots][label]['label'])) and 
								(values[plots][label]['legend']) and (len(values[plots][label]['value'])==1)))))],
						],
					]


				if data.get('multiline',None):
					separator = '\n'
				else:
					separator = '~,~'

				value = separator.join(['~,~'.join(i).replace('$','') for i in value if i])

				if isinstance(data.get(attr),str) and data[attr].count('%s'):
					data[attr] = data[attr]%(value)
				else:
					data[attr] = value

			# data
			for plots in PLOTS:

				if settings[instance][subinstance]['ax'].get(plots) is None:
					continue

				for data in flatten(settings[instance][subinstance]['ax'][plots]):

					if not data:
						continue

					slices = []
					subslices = [data[OTHER][OTHER].get('slice'),data[OTHER][OTHER].get('labels')]
					for subslice in subslices:
						if subslice is None:
							subslice = [slice(None)]
						elif isinstance(subslice,dict):
							subslice = {
								axis if (axis in data) else [subaxis 
										for subaxis in ALL if ((subaxis in data[OTHER]) and 
											(data[OTHER][subaxis]['axis']==axis))][0]: 
								subslice[axis] for axis in subslice if (
								(not isinstance(subslice[axis],str)) or
								((axis in data) or any(data[OTHER][subaxis]['axis']==axis 
									for subaxis in data[OTHER] if (
									(subaxis in ALL) and (subaxis in data[OTHER]))))
								)
								}

							if subslice:
								subslice = [
									conditions([parse(axis,subslice[axis],{axis: np.array(data[axis])},verbose=verbose) 
									for axis in subslice if isinstance(subslice[axis],str)],op='and'),
									*[slice(*subslice[axis]) for axis in subslice 
									 if not isinstance(subslice[axis],str)]
									]
							else:
								subslice = [slice(None)]
						else:
							subslice = [slice(*subslice)]
						
						slices.extend(subslice)


					slices = [
						conditions([subslice for subslice in slices if not isinstance(subslice,slice)],op='and'),
						*[subslice for subslice in slices if isinstance(subslice,slice)]
						]
					slices = [subslice if subslice is not None else slice(None) for subslice in slices]

					for attr in data:
						if (attr in ALL) and (data[attr] is not None):
							
							value = np.array([valify(value,valify=data[OTHER][OTHER].get('valify')) for value in data[attr]])
							
							for subslice in slices:
								value = value[subslice]

							data[attr] = value


			# include labels
			for plots in PLOTS:

				if settings[instance][subinstance]['ax'].get(plots) is None:
					continue

				for data in flatten(settings[instance][subinstance]['ax'][plots]):

					if not data:
						continue

					attr = OTHER
					if data[attr][attr].get('labels') is not None:
						for label in data[attr][attr]['labels']:
							if (label in data[attr]) and (label not in ALL) and not parse(label,data[attr][attr]['labels'][label],data[attr],verbose=verbose):
								data.clear()
								break

			# axis label
			for attr in settings[instance][subinstance]['ax']:

				for plots in PLOTS:

					if settings[instance][subinstance]['ax'].get(plots) is None:
						continue

					for axis in AXIS:

						for data in flatten(settings[instance][subinstance]['ax'][plots]):

							if not data:
								continue

							if attr not in ['set_%slabel'%(axis)] or (not settings[instance][subinstance]['ax'].get(attr)):
								continue

							if settings[instance][subinstance]['ax'].get(attr,{}).get('%slabel'%(axis)) is None:
								value = data[OTHER][axis]['axis']
							else:
								value = settings[instance][subinstance]['ax'].get(attr,{}).get('%slabel'%(axis))

							value = texify(value,texify=data[OTHER][OTHER].get('texify'))

							settings[instance][subinstance]['ax'][attr]['%slabel'%(axis)] = value

							break


			# label
			for plots in PLOTS:

				if settings[instance][subinstance]['ax'].get(plots) is None:
					continue

				for data in flatten(settings[instance][subinstance]['ax'][plots]):

					if not data:
						continue

					attr = OTHER
					value = ',~'.join([
						*[(texify(scinotation(data[OTHER][label],**data[OTHER][OTHER].get('scinotation',{})),texify=data[OTHER][OTHER].get('texify')) 
							if values[plots][label]['label'] else texify(scinotation(data[OTHER][data[OTHER][OTHER][OTHER][label].replace('@','')],**data[OTHER][OTHER].get('scinotation',{})),texify=data[OTHER][OTHER].get('texify'))
							) 
							for label in realsorted(set((
							label 
							for label in values[plots] 
							if (((values[plots][label]['label']) and (len(values[plots][label]['value'])>1)) and 
								not (values[plots][label]['other'])))))],
						*[(texify(scinotation(data[OTHER][data[OTHER][OTHER][OTHER][label].replace('@','')],**data[OTHER][OTHER].get('scinotation',{})),texify=data[OTHER][OTHER].get('texify'))
							)
							for label in realsorted(set((
							label 
							for label in values[plots]
							if (not ((values[plots][label]['label'])) and 
								(values[plots][label]['other']) and (len(values[plots][label]['value'])>1)))))],
						*[(texify(scinotation(data[OTHER][data[OTHER][OTHER][OTHER][label].replace('@','')],**data[OTHER][OTHER].get('scinotation',{})),texify=data[OTHER][OTHER].get('texify'))
							)
							for label in realsorted(set((
							label 
							for label in values[plots]
							if (not ((values[plots][label]['label'])) and 
								(values[plots][label]['legend']) and (len(values[plots][label]['value'])>1)))))],
						])

					value = value if value else None

					data[attr] = value	

	# Plot data
	for instance in settings:

		logger.log(info,"Plotting : %s"%(instance))

		fig[instance],ax[instance] = plot(fig=fig[instance],ax=ax[instance],settings=settings[instance])

	return


def postprocessor(hyperparameters,pwd=None,cwd=None,verbose=None):
	'''
	Postprocess data
	Args:
		hyperparameters (str,dict): Path to or dictionary of process settings
		pwd (str): Root path of data
		cwd (str): Root path of plots
		verbose (bool): Verbosity		
	'''
	if (hyperparameters is None):
		return

	if not hyperparameters['postprocess']:
		return

	path = cwd
	postprocess(path)

	return


def process(data,settings,hyperparameters,pwd=None,cwd=None,verbose=True):
	'''
	Process data
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		settings (str,dict): Path to or dictionary of plot settings
		hyperparameters (str,dict): Path to or dictionary of process settings
		pwd (str): Root path of data
		cwd (str): Root path of plots
		verbose (bool): Verbosity
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
	data,settings,hyperparameters = setup(data,settings,hyperparameters,pwd,cwd,verbose=verbose)

	# Load data
	loader(data,settings,hyperparameters,verbose=verbose)

	# Plot data
	plotter(settings,hyperparameters,verbose=verbose)

	# Post process data
	postprocessor(hyperparameters,pwd,cwd,verbose=verbose)

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
		'--verbose':{
			'help':'Verbosity',
			'action':'store_true'
		},						

	}

	wrappers = {
		'pwd':lambda kwarg,wrappers,kwargs: split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**','') if kwargs.get(kwarg) is None else kwargs.get(kwarg),
		'cwd':lambda kwarg,wrappers,kwargs: split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**','') if kwargs.get(kwarg) is None else kwargs.get(kwarg),
	}

	args = argparser(arguments,wrappers)

	main(*args,**args)
