#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,traceback
from copy import deepcopy
from functools import partial,wraps
import numpy as np
import scipy as sp
import scipy.stats
import scipy.special
import pandas as pd
from pandas.api.types import is_float_dtype
from natsort import natsorted,realsorted
from math import prod

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.utils import array,expand_dims,conditions
from src.utils import to_key_value,to_tuple,to_number,to_str,to_int,is_iterable,is_number,is_nan,is_numeric
from src.utils import argmax,argsort,difference,abs
from src.utils import e,pi,nan,scalars,arrays,delim,nulls,null,Null,scinotation
from src.iterables import getter,setter,search,inserter,indexer
from src.parallel import Parallelize,Pooler
from src.io import load,dump,join,split,exists
from src.fit import fit
from src.postprocess import postprocess
from src.plot import plot,AXES,VARIANTS,FORMATS,ALL,VARIABLES,OTHER,PLOTS,LAYOUTDIM

# Logging
from src.logger	import Logger
logger = Logger()
info = 100
debug = 100

LAYOUT = ['row','col']
GRID = [*LAYOUT,'axis','axes']
INDEXES = ['variable','label','func','axis'] 

LAYOUTDIM = len(LAYOUT)
GRIDDIM = len(GRID)
INDEXDIM = len(INDEXES)
AXISDIM = GRIDDIM - 1 - LAYOUTDIM

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

	if string is not None:
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
	obj = 'ax'

	if (settings is None) or (hyperparameters is None):
		return data,settings,hyperparameters

	for instance in list(settings):
		
		if (settings.get(instance) is None):
			settings.pop(instance,None);
			continue

		if all(subinstance in defaults for subinstance in settings[instance]):
			settings[instance] = {str(None): settings[instance]}

		for subinstance in settings[instance]:
			
			setter(settings[instance][subinstance],defaults,delimiter=delim,func=False)

			if not settings[instance][subinstance].get(obj):
				continue
			for prop in PLOTS:
				if not settings[instance][subinstance][obj].get(prop):
					continue
				tmp = []
				for index,shape,item in search(settings[instance][subinstance][obj][prop],returns=True):
					index = [*index,*[0]*(INDEXDIM-len(shape))]
					inserter(index,item,tmp)
				settings[instance][subinstance][obj][prop] = tmp


	# Set process hyperparameters
	defaults = {
		'path':{},
		'load':None,
		'dump':None,
		'plot':None,
		'process':None,
		'postprocess':None,
		}
	setter(hyperparameters,defaults,delimiter=delim,func=False)

	# Get paths
	path = data if isinstance(data,str) else None
	hyperparameters['file'],hyperparameters['directory'],hyperparameters['ext'] = {},{},{}
	defaults = {
		'data': 	join(cwd,join(split(path,file=True),ext='tmp'),ext='hdf5'),
		'metadata': join(cwd,join(split(path,file=True),ext=None),ext='json'),
	}
	setter(hyperparameters['path'],defaults,delimiter=delim,func=False)
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

	for instance in list(settings):
		if (not hyperparameters.get(attr,{}).get(instance)) or (not settings[instance]):
			settings.pop(instance,None);
			continue

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

	dimensions = AXES
	other = [OTHER]
	dim = len(dimensions)

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
				'func':{}, 
				'include':None,
				'exclude':None,
				'slice':None,
				'labels':None,
				'analysis':{
					# 'zscore':[{'objective':0.5}],
					# 'quantile':{'objective':None}
					# 'parse':[{'__path__':'*','M':"<600<"}]
					# 'abs':['alpha'],
					# 'func':{"MN":"functions.py.MN"},
					},
				'shape':None,
					#{'shape': {'row':[],'col':[],'axis':[],'axes':[]},'reshape':[],'transpose':[]},
				'legend': {
					'label':{},'include':None,'exclude':None,'sort':None,
				},
				'wrapper':{},
				'texify':{},
				'valify': {},		
				'scinotation':{'scilimits':[0,2],'decimals':0,'one':False},
				'args':None,
				'kwargs':None,
	}

	keys = {}

	for i in range(dim,0,-1):

		items = [*dimensions[:i],*other]
		types = (list,dict,)
		key = search(dictionary,items=items,returns=True,types=types)
		
		key = {tuple(index): dict(zip(items,item)) for index,shape,item in key}

		key = {index:key[index] for index in key if index not in keys}

		keys.update(key)

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
					keys[name][attr] = keys[name][attr][list(keys[name][attr])[0]]
				elif isinstance(keys[name][attr],str):
					keys[name][attr] = keys[name][attr] if keys[name][attr] not in [''] else default
				else:
					keys[name][attr] = keys[name][attr][0]

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
	negators = ['!','~']
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
	elif value is None:
		pass
	elif isinstance(value,str):
		outs = [default]
		for value in value.split(parserator):

			negate = False
			for negator in negators:
				if value.startswith(negator) and value.endswith(negator):
					negate = True
					value = value[len(negator):-len(negator)]
					break

			for delimiter in delimiters:
				
				if value.startswith(delimiter) and value.endswith(delimiter):
				
					values = value[len(delimiter):-len(delimiter)].split(separator)

					if delimiter in ['$']: # Explicit value: value
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [i for value in values for i in [parser(value),value]]           
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
							out = conditions([data[key]==data[value] for value in values if value in data],op='or')

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
							out = conditions([data[key] < value for value in values],op='or')

					elif delimiter in ['<=']: # Bound value: upper (inclusive)
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           
					  
						if values and (values is not null):
							out = conditions([data[key] <= value for value in values],op='or')

					elif delimiter in ['>']: # Bound value: lower (exclusive)
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           
					  
						if values and (values is not null):
							out = conditions([data[key] > value for value in values],op='or')

					elif delimiter in ['>=']: # Bound value: lower (inclusive)
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           
					  
						if values and (values is not null):
							out = conditions([data[key] >= value for value in values],op='or')

					elif delimiter in ['==']: # Include value
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [i for value in values for i in [parser(value),value]]           

						if values and (values is not null):
							out = conditions([data[key] == value for value in values],op='or')																												

					elif delimiter in ['!=']: # Exclude value
						parser = lambda value: (to_number(value) if len(value)>0 else null)
						values = [parser(value) for value in values]           

						if values and (values is not null):
							out = conditions([data[key] != value for value in values],op='or')																												

					if negate:
						out = ~out

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
		analyses (dict[str,iterable[iterable[dict]]]): Processes to analyse of the form 
			{analysis:[({attr:value},kwargs)]},
			allowed analysis strings in ['zscore','quantile','slice','parse','abs','log','log10','replace','func']
		verbose (bool): Verbosity			
	Returns:
		out (dataframe): Analysed data
	'''

	default = True

	out = default

	if analyses is not None:

		for analysis in analyses:
			if analysis in ['zscore']:
				def func(attrs,data):
					function = sp.stats.zscore
					value = {attr: attrs[attr] if not isinstance(attrs[attr],dict) else attrs[attr].pop('value',None) for attr in attrs}
					wrappers = {attr: None if not isinstance(attrs[attr],dict) else attrs[attr].pop('wrapper',None) for attr in attrs}
					kwargs = {attr: {} if not isinstance(attrs[attr],dict) else attrs[attr] for attr in attrs}
					out = {attr: (data[[attr]].apply(wrappers[attr])) if wrappers[attr] is not None else data[[attr]] for attr in attrs}
					out = {attr: ((out[attr].apply(function,**kwargs[attr]) <= value[attr]) if value[attr] > 0 else 
								  (out[attr].apply(function,**kwargs[attr]) >= -value[attr]))
							if ((len(out[attr])>1) and (value[attr] is not None)) else True for attr in attrs}
					out = conditions([out[attr] for attr in attrs],op='and')
					return out
			elif analysis in ['quantile']:
				def func(attrs,data):
					function = analysis
					value = {attr: attrs[attr] if not isinstance(attrs[attr],dict) else attrs[attr].pop('value',None) for attr in attrs}
					wrappers = {attr: None if not isinstance(attrs[attr],dict) else attrs[attr].pop('wrapper',None) for attr in attrs}					
					kwargs = {attr: {} if not isinstance(attrs[attr],dict) else attrs[attr] for attr in attrs}
					out = {attr: (data[[attr]].apply(wrappers[attr])) if wrappers[attr] is not None else data[[attr]] for attr in attrs}
					out = {attr: (((out[attr] > getattr(out[attr],function)(value[attr])) if value[attr] > 0 else 
								   (out[attr] <= getattr(out[attr],function)(-value[attr]))) &
								  ((out[attr] < getattr(out[attr],function)(1-value[attr])) if value[attr] > 0 else 
								   (out[attr] >= getattr(out[attr],function)(1+value[attr]))))
							if ((len(out[attr])>1) and (value[attr] is not None)) else True for attr in attrs}
					out = conditions([out[attr] for attr in attrs],op='and')
					return out
			elif analysis in ['slice']:
				def func(attrs,data):
					function = lambda data: np.argsort(data,axis=0).to_numpy().ravel()
					value = {attr: attrs[attr] if not isinstance(attrs[attr],dict) else attrs[attr].pop('value',None) for attr in attrs}
					wrappers = {attr: None if not isinstance(attrs[attr],dict) else attrs[attr].pop('wrapper',None) for attr in attrs}					
					kwargs = {attr: {} if not isinstance(attrs[attr],dict) else attrs[attr] for attr in attrs}
					out = {attr: (data[[attr]].apply(wrappers[attr])) if wrappers[attr] is not None else data[[attr]] for attr in attrs}
					out = {attr: (
							(out[attr]>=out[attr].iloc[function(out[attr])[value[attr] if value[attr] < len(out[attr]) else 0]]) & 
							(out[attr]<=out[attr].iloc[function(out[attr])[len(out[attr])-1-value[attr] if value[attr] < len(out[attr]) else -1]]) 
							)
							for attr in attrs}
					out = conditions([out[attr] for attr in attrs],op='and')
					return out					
			elif analysis in ['parse']:
				def func(attrs,data):
					function = parse
					value = {attr: attrs[attr] if not isinstance(attrs[attr],dict) else attrs[attr].pop('value',None) for attr in attrs}
					kwargs = {attr: {} if not isinstance(attrs[attr],dict) else attrs[attr] for attr in attrs}
					out = {attr: (data[[attr]].apply(wrappers[attr])) if wrappers[attr] is not None else data[[attr]] for attr in attrs}					
					out = [function(attr,value[attr],out,verbose=verbose) for attr in attrs]
					out = conditions(out,op='and')
					return out
			elif analysis in ['abs','log','log10']:
				def func(attrs,data):
					function = analysis
					kwargs = {attr: {} if not isinstance(attrs[attr],dict) else attrs[attr] for attr in attrs}
					out = data
					for attr in attrs:
						out[attr] = out[attr].apply(function,**kwargs[attr])
					return out
			elif analysis in ['replace']:
				def func(attrs,data):
					function = lambda a,b: a==b
					value = {attr: attrs[attr] if not isinstance(attrs[attr],dict) else attrs[attr].pop('value',None) for attr in attrs}
					kwargs = {attr: {} if not isinstance(attrs[attr],dict) else attrs[attr] for attr in attrs}
					out = data
					for attr in attrs:
						for kwarg in kwargs[attr]:
							value = kwargs[attr][kwarg]
							if kwarg in [None,'nan','none','None','NaN']:
								kwarg = nan
							elif isinstance(kwarg,str) and is_number(kwarg):
								kwarg = to_number(kwarg)
							if value in [None,'nan','none','None','NaN']:
								value = nan
							elif isinstance(value,str) and is_number(value):
								value = to_number(value)
							out[attr][function(out[attr],kwarg)] = value
					return out					
			elif analysis in ['func']:
				def func(attrs,data):
					out = data
					for attr in attrs:
						function = attrs[attr]
						default = None
						function = load(function,default=None)
						if function is not None:
							out[attr] = function(data)
					return out
			else:
				continue

			if isinstance(analyses[analysis],dict):
				args = [analyses[analysis]]
			elif isinstance(analyses[analysis],(list,tuple)) and all(isinstance(i,dict) for i in analyses[analysis]):
				args = [i for i in analyses[analysis]]
			elif isinstance(analyses[analysis],(list,tuple)) and all(isinstance(i,(list,tuple)) for i in analyses[analysis]):
				args = [{j:{} for j in i} for i in analyses[analysis]]
			elif isinstance(analyses[analysis],(list,tuple)) and all(isinstance(i,(str)) for i in analyses[analysis]):
				args = [{i:{}} for i in analyses[analysis]]
			elif isinstance(analyses[analysis],str):
				args = [{analyses[analysis]:{}}]
			else:
				args = []

			args = deepcopy(args)

			for attrs in args:
				if analysis in ['zscore','quantile','slice','parse']:
					value = func(attrs,data)
					value = value.to_numpy() if not isinstance(value,bool) else value
					out = conditions([out,value],op='and')
				elif analysis in ['abs','log','log10','replace','func']:
					data = func(attrs,data)

	if out is True:
		out = data
	else:
		out = data[out]

	return out



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

	# Set data boolean
	new = False

	# Set metadata
	metadata = hyperparameters['path']['metadata']
	
	def func(key_iterable,key_elements,iterable,elements):
		if (
			(key_iterable == key_elements) and 
			(key_iterable in PLOTS) and (key_elements in PLOTS) and 
			isinstance(iterable.get(key_iterable),list)
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

			if not isinstance(elements.get(key_elements),list):
				elements[key_elements] = [elements.get(key_elements)]

			for i,(item,shape,data) in enumerate(search(elements.get(key_elements),returns=True)):
				if (not iterable.get(key_iterable)) or i >= len(iterable.get(key_iterable)):
					continue

				i = None
				axes = {attr: data[attr] for attr in data if attr in ALL}
				if data.get(OTHER) is None:
					continue
				if isinstance(data[OTHER],str):
					labels = {data[OTHER]:None}
				elif isinstance(data[OTHER],dict):
					if OTHER in data[OTHER]:
						labels = {attr: data[OTHER][OTHER][attr] for attr in data[OTHER][OTHER]}
					else:
						labels = {attr: data[OTHER][attr] for attr in data[OTHER]}
				else:
					labels = {attr: None for attr in data[OTHER]}

				for j in range(len(iterable.get(key_iterable))):
					if all((
						all(datum[OTHER][attr]['label']==axes[attr] for attr in axes) and 
						(len(datum[OTHER][OTHER][OTHER]) == len(labels)) and
						all(datum[OTHER][OTHER][OTHER][attr]==labels[attr] for attr in labels)
						)
						for datum in search(iterable.get(key_iterable)[j]) if datum):
						i = j
						break


				if i is None:
					continue

				for subindex,datum in enumerate(search(iterable.get(key_iterable)[i])):
					if not datum:
						continue
					datum.update({attr: data[attr] for attr in data if attr not in [*ALL,OTHER]})
					datum.update({attr: datum[attr] for attr in datum if attr in [*ALL] and not isinstance(datum[attr],str)})

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
			tmp = []
			for index,shape,item in search(out,returns=True):
				i = index
				index = [*index,*[0]*(INDEXDIM-len(shape))]
				inserter(index,item,tmp)
			for i in range(len(tmp)):
				out[i] = tmp[i]


		else:
			out = elements.get(key_elements)
		return out	

	if hyperparameters['load']:


		# Load settings
		path = metadata
		default = None
		tmp = load(path,default=default,verbose=verbose)

		new = exists(path) and tmp is not None

		if new:
			new = deepcopy(settings)
			settings.update(tmp)
			tmp = new
			setter(settings,tmp,func=func)

	else:

		# Load data
		path = data
		tmp = hyperparameters['path']['data']
		try:
			assert exists(tmp)
			path = tmp
			wrapper = 'pd'
			default = None
			data = load(path,default=default,wrapper=wrapper,verbose=verbose)
		except Exception as exception:
			path = data
			wrapper = 'df'			
			default = None
			data = load(path,default=default,wrapper=wrapper,verbose=verbose)
			
		new = tmp is not None and data is not None

		if new:
			path = tmp
			wrapper = 'pd'
			dump(data,path,wrapper=wrapper,verbose=verbose)

		# Get functions of data
		apply(keys,data,settings,hyperparameters,verbose=verbose)


	# Check settings
	attr = 'instance'
	for instance in list(settings):
		if (not hyperparameters.get(attr,{}).get(instance)) or (not settings[instance]):
				settings.pop(instance,None);
				continue

	# Dump settings
	if hyperparameters['dump']:
		path = metadata
		wrapper = None
		dump(settings,path,wrapper=wrapper,verbose=verbose)

	return


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

	def default(obj,*args,**kwargs):
		return obj.first()

	def mean(obj,*args,**kwargs):
		obj = np.array(list(obj))
		obj = to_tuple(obj.mean(0))
		return obj
	def std(obj,*args,**kwargs):
		obj = np.array(list(obj))
		obj = to_tuple(obj.std(0,ddof=1))
		return obj
	def sem(obj,*args,**kwargs):
		obj = np.array(list(obj))
		obj = to_tuple(obj.std(0,ddof=1)/np.sqrt(obj.shape[0]))
		return obj	

	def mean_log(obj,*args,**kwargs):
		return 10**(obj.apply('log10').mean())
	def std_log(obj,*args,**kwargs):
		return 10**(obj.apply('log10').std(ddof=1))
	def sem_log(obj,*args,**kwargs):
		return 10**(obj.apply('log10').sem(ddof=1))

	# dtype = {attr: 'float128' for attr in data if is_float_dtype(data[attr].dtype)}
	# dtype = {attr: data[attr].dtype for attr in data if is_float_dtype(data[attr].dtype)}
	dtype = {attr: 'float' for attr in data if is_float_dtype(data[attr].dtype)}	
	data = data.astype(dtype)

	dtypes = {attr: ('array' if any(isinstance(i,tuple) for i in data[attr]) else 'object' if data[attr].dtype.kind in ['O'] else 'dtype') 
				for attr in data}


	for name in keys:

		logger.log(info,"Processing : %r"%(name,))

		if any((keys[name][axes] not in data) and (keys[name][axes] is not null) for axes in AXES if axes in keys[name]):
			key,value = name,None
			setter(settings,{key:value},delimiter=delim,func=True)
			continue

		dimensions = [axes for axes in AXES if axes in keys[name]]
		other = OTHER
		label = keys[name][other].get(other,{})
		include = keys[name][other].get('include')
		exclude = keys[name][other].get('exclude')
		funcs = keys[name][other].get('func',{})
		analyses = keys[name][other].get('analysis',{})
		wrappers = keys[name][other].get('wrapper',{})
		args = keys[name][other].get('args',None)
		kwargs = keys[name][other].get('kwargs',None)


		funcs = deepcopy(funcs)
		stat = 'stat'
		stats = {axes: {'':'mean','err':'std'} for axes in dimensions}
		functions = {'mean_log':mean_log,'std_log':std_log,'sem_log':'sem_log'}
		
		if not funcs:
			funcs = {stat:None}
		
		for function in funcs:
			
			if funcs[function] is None:
				funcs[function] = {}
			
			for axes in stats:

				if funcs[function].get(axes) is None:
					funcs[function][axes] = {}

				for func in stats[axes]:

					if func not in funcs[function][axes]:
						funcs[function][axes][func] = stats[axes][func]
				for func in funcs[function][axes]:
					
					obj = funcs[function][axes][func]
					
					if isinstance(obj,str):
						
						if callable(getattr(data,obj,None)):
							pass
						elif obj in functions:
							obj = functions[obj]
						else:
							obj = load(obj,default=default)

					if callable(obj):

						if args is None:
							arguments = ()
						else:
							arguments = args
						
						if isinstance(arguments,dict) and any(i in arguments for i in funcs):
							arguments = arguments.get(arguments,arguments)
						if isinstance(arguments,dict) and any(i in arguments for i in stats):
							arguments = arguments.get(axes,arguments)
						if isinstance(arguments,dict) and any(i in arguments for i in stats[axes]):
							arguments = arguments.get(func,arguments)

						if isinstance(arguments,dict):
							arguments = ()
						
						if kwargs is None:
							keywords = {}
						else:
							keywords = kwargs

						if isinstance(keywords,dict) and any(i in keywords for i in funcs):
							keywords = keywords.get(function,keywords)
						if isinstance(keywords,dict) and any(i in keywords for i in stats):
							keywords = keywords.get(axes,keywords)
						if isinstance(keywords,dict) and any(i in keywords for i in stats[axes]):
							keywords = keywords.get(func,keywords)

						if not isinstance(keywords,dict):
							keywords = {}

						try:
							obj = wraps(obj)(partial(obj,*arguments,**keywords))
						except:
							pass

					funcs[function][axes][func] = obj

		# exit()
		tmp = {}
		for attr in wrappers:

			if attr in ALL:
				continue

			wrapper = load(wrappers[attr],default=None)
			
			if attr in data:
				tmp[attr] = data[attr]
			else:
				tmp[attr] = None

			try:
				data[attr] = wrapper(data)
			except:
				pass

		independent = [keys[name][axes] for axes in dimensions[:-1] if keys[name][axes] in data]
		dependent = [keys[name][axes] for axes in dimensions[-1:] if keys[name][axes] in data]
		labels = [attr for attr in label if (attr in data) and (((label[attr] is null) and (exclude is None) and (include is None)) or ((label[attr] is null) and (exclude is None)) or ((exclude is not None) and (attr not in exclude))) or ((include is not None) and (attr in include))]
		boolean = [parse(attr,label[attr],data,verbose=verbose) for attr in label]
		boolean = conditions(boolean,op='and')
		boolean = slice(None) if ((boolean is True) or (boolean is False) or (boolean is None)) else boolean

		by = [*labels,*independent]

		if not by:
			key,value = name,None
			setter(settings,{key:value},delimiter=delim,func=True)
			continue

		groups = data[boolean].groupby(by=by,as_index=False)

		if analyses:
		
			groups = groups.apply(analyse,analyses=analyses,verbose=verbose).reset_index(drop=True).groupby(by=by,as_index=False)

		shapes = {group[:-len(independent)] if (independent) and isinstance(group,tuple) else group: groups.get_group(group).shape for group in groups.groups}

		agg = {
			**{attr : [(attr, {'array':mean,'object':'first','dtype':'mean'}[dtypes[attr]] 
					  if attr not in by else {'array':'first','object':'first','dtype':'first'}[dtypes[attr]])] 
					  for attr in data},
			**{attr : [(delim.join(((attr,function,func))),
						{'array':{'':mean,'err':std}[func],
						 'object':'first',
						 'dtype':funcs[function][axes][func]
						}[dtypes[attr]]) 
						for function in funcs for func in funcs[function][axes]] 
						for axes,attr in zip([*dimensions[:-1],*dimensions[-1:]],[*independent,*dependent])
						},
		}

		dtype = {attr: data[attr].dtype for attr in agg if attr in label}

		droplevel = dict(level=0,axis=1)
		by = [*labels]
		variables = [*independent,*dependent,*[subattr[0] for attr in [*independent,*dependent] for subattr in agg[attr]]]

		groups = groups.agg(agg).droplevel(**droplevel).astype(dtype)

		if by:
			groups = groups.groupby(by=by,as_index=False)
		else:
			groups = GroupBy(groups,by=by)

		assert all(groups.get_group(group).columns.nlevels == 1 for group in groups.groups) # Possible future broken feature agg= (label,name)

		for i,group in enumerate(groups.groups):

			logger.log(info,"Group : %r %r -> %r"%(group,shapes[group],groups.get_group(group).shape))
			for j,function in enumerate(funcs):

				grouping = groups.get_group(group)
				
				key = (*name[:-3],i,j,*name[-1:])
				value = deepcopy(getter(settings,name,delimiter=delim))

				source = [attr for attr in data if attr not in variables]
				destination = other
				value[destination] = {
					**{attr: grouping[attr].to_list()[0] for attr in source},
					**{'%s%s'%(axes,func) if keys[name][axes] in [*independent,*dependent] else axes: 
						{
						'group':[i,dict(zip(groups.grouper.names,group if isinstance(group,tuple) else (group,)))],
						'func':[j,function],
						'label':keys[name][axes] if keys[name][axes] is not null else None
						} 
						for axes in dimensions 
						for func in funcs[function][axes]
						},
					**{other: {attr: {subattr: keys[name][other][attr][subattr] 
						if keys[name][other][attr][subattr] is not null else None for subattr in keys[name][other][attr]}
						if isinstance(keys[name][other][attr],dict) else keys[name][other][attr] 
						for attr in keys[name][other]}
						},
					}

				for axes in dimensions:
					for func in funcs[function][axes]:	
						
						attr = keys[name][axes]

						source = delim.join(((attr,function,func))) if attr in [*independent,*dependent] else attr
						destination = '%s%s'%(axes,func) if attr in [*independent,*dependent] else axes

						if grouping.shape[0]:
							if source in grouping:
								if dtypes[attr] in ['array']:
									value[destination] = np.array([np.array(i) for i in grouping[source]])
								else:
									value[destination] = grouping[source].to_numpy()
							elif source is null:
								source = delim.join(((dependent[-1],function,func)))
								value[destination] = np.arange(len(grouping[source].iloc[0]))
							else:
								value[destination] = grouping.reset_index().index.to_numpy()

							if isinstance(value[destination],arrays):
								value[destination] = value[destination].tolist()

						else:
							value[destination] = None


				setter(settings,{key:value},delimiter=delim,func=True)

		for attr in tmp:
			if tmp[attr] is None:
				data.drop(columns=attr)
			else:
				data[attr] = tmp[attr]

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
	obj = 'ax'


	# Check data
	for instance in list(settings):
		for subinstance in list(settings[instance]):
			for prop in settings[instance][subinstance][obj]:

				if isinstance(settings[instance][subinstance][obj].get(prop),dict):
					settings[instance][subinstance][obj][prop] = [settings[instance][subinstance][obj][prop]]
				
				if prop in PLOTS:
					for data in search(settings[instance][subinstance][obj][prop]):

						if data is None:
							continue

						dimensions = [axes for axes in AXES if axes in data]
						independent = [axes for axes in ALL 
							for variable in VARIABLES 
							if axes in VARIABLES[variable] and variable in dimensions and dimensions.index(variable) < (len(dimensions)-1)]
						dim = len(dimensions)

						for attr in ALL:
							
							if attr not in data:
								continue

							if attr in independent:
								if (data.get(attr) is None) or isinstance(data.get(attr),str):
									data.clear()

		if all((not data) for prop in PLOTS if settings[instance][subinstance][obj].get(prop) for data in search(settings[instance][subinstance][obj][prop])):
			settings[instance].pop(subinstance);

	for instance in list(settings):
		if not settings[instance]:
			settings.pop(instance);

	# Set grid layout based on GRID
	grid = {}
	for instance in list(settings):
		for subinstance in list(settings[instance]):
			
			if not settings[instance][subinstance].get(obj):
				continue

			if grid.get(instance) is None:
				grid[instance] = {}
			
			if grid[instance].get(subinstance) is None:
				grid[instance][subinstance] = [*[1]*LAYOUTDIM,*[1]*AXISDIM,*[-1]*(GRIDDIM-LAYOUTDIM-AXISDIM)]

			for prop in PLOTS:
				
				if prop not in settings[instance][subinstance][obj]:
					continue
				
				for data in search(settings[instance][subinstance][obj][prop]):

					if not data or not data.get(OTHER) or not data[OTHER].get(OTHER):
						continue

					shapes = data[OTHER][OTHER].get('shape')
					dimensions = [axes for axes in AXES if axes in data]
					independent = [axes for axes in ALL 
						for variable in VARIABLES 
						if axes in VARIABLES[variable] and variable in dimensions and dimensions.index(variable) < (len(dimensions)-1)]
					dim = len(dimensions)

					for axes in ALL:
						
						if axes not in data or isinstance(data[axes],scalars):
							continue

						data[axes] = np.array(data[axes])

						if shapes and (axes not in independent):

							shape = shapes.get('shape')
							slices = shapes.get('slices')
							transpose = shapes.get('transpose')
							reshape = shapes.get('reshape')
							indices = shapes.get('indices')

							if slices:
								if not isinstance(slices,dict):
									slices = dict(zip(GRID,slices)) 

								for i,prop in enumerate(GRID):
									if not slices[prop]:
										continue
									indexes = [slice(None)]*GRIDDIM
									indexes[i] = slices[prop]
									indexes = tuple(indexes)
									data[axes] = data[axes][indexes]

							if transpose:
								transpose = [transpose] if isinstance(transpose,int) else transpose
								transpose = [data[axes].ndim + i if i < 0 else i for i in transpose]
								transpose = [*transpose,*[i for i in range(data[axes].ndim) if i not in transpose]]
								data[axes] = data[axes].transpose(transpose)

							if reshape:
								reshape = [reshape] if isinstance(reshape,(int,str)) else reshape
								reshape = [i if isinstance(i,int) else int(data[OTHER].get(i)) if i is not None else data[axes].shape[reshape.index(i)] for i in reshape]
								data[axes] = data[axes].reshape(reshape)
							
							if shape:
								shape = {prop: [shape[prop]] if isinstance(shape.get(prop),int) else shape.get(prop) if shape.get(prop) is not None else [] for prop in GRID}
								shape = {prop: [data[axes].ndim + i if i < 0 else i for i in shape[prop]] for prop in GRID}
								shape = {prop:{
									**{prop: shape[prop] for prop in GRID if shape.get(prop)},
									**{prop: [i for i in range(data[axes].ndim) if not any(i in shape[prop] for prop in GRID if shape.get(prop))] for prop in GRID if not shape.get(prop)}
									}[prop] for prop in GRID
									}

								transpose = [i for prop in GRID for i in shape[prop]]
								reshape = [max(1,prod(data[axes].shape[i] for i in shape[prop])) for prop in GRID]

								if transpose:
									data[axes] = data[axes].transpose(transpose)
								
								if reshape:
									data[axes] = data[axes].reshape(reshape)
							else:
								reshape = [*[1]*LAYOUTDIM,*[1]*AXISDIM,data[axes].size]
								if reshape:
									data[axes] = data[axes].reshape(reshape)
						
							if indices:
								if not isinstance(indices,dict):
									indices = dict(zip(GRID,indices)) 

								for i,prop in enumerate(GRID):
									if not indices.get(prop):
										continue
									indexes = [slice(None)]*GRIDDIM
									indexes[i] = indices[prop]
									indexes = tuple(indexes)
									data[axes] = data[axes][indexes]


						else:
							reshape = [*[1]*LAYOUTDIM,*[1]*AXISDIM,data[axes].size]
							if reshape:
								data[axes] = data[axes].reshape(reshape)

						grid[instance][subinstance] = [
							max(1,max(grid[instance][subinstance][i],data[axes].shape[i]))
							for i in range(len(grid[instance][subinstance]))]

					dimensions = [axes for axes in AXES if axes in data]
					independent = [axes for axes in ALL 
						for variable in VARIABLES 
						if axes in VARIABLES[variable] and variable in dimensions and dimensions.index(variable) < (len(dimensions)-1)]
					dim = len(dimensions)

					for axes in ALL:
						
						if axes not in data or isinstance(data[axes],scalars):
							continue

						if shapes and (axes in independent):
							data[axes] = data[AXES[dim-1]].copy()
							data[axes][...,:] = np.arange(data[AXES[dim-1]].shape[-1])


	for instance in list(settings):
		for subinstance in list(settings[instance]):
			
			if not settings[instance][subinstance].get(obj):
				continue

			for position in itertools.product(*(range(i) for i in grid[instance][subinstance][:LAYOUTDIM])):
				
				key = delim.join([subinstance,*[str(i) for i in position]])
				
				settings[instance][key] = deepcopy(settings[instance][subinstance])
				grid[instance][key] = deepcopy(grid[instance][subinstance])[:LAYOUTDIM]

				for axis in itertools.product(*(range(i) for i in grid[instance][subinstance][LAYOUTDIM:LAYOUTDIM+AXISDIM])):

					for prop in PLOTS:
						
						if prop not in settings[instance][subinstance][obj]:
							continue
						
						for index,shape,data in search(deepcopy(settings[instance][subinstance][obj][prop]),returns=True):
						
							if not data:
								continue

							for axes in ALL:
								
								if (axes not in data) or isinstance(data[axes],scalars):
									continue

								if (any(position[i]>=data[axes].shape[i] for i in range(min(data[axes].ndim-1,LAYOUTDIM))) or 
									any(axis[i-LAYOUTDIM]>=data[axes].shape[i] for i in range(LAYOUTDIM,min(data[axes].ndim-1,LAYOUTDIM+AXISDIM)))):
									data[axes] = None
								else:
									slices = tuple((*position,*axis))

									data[axes] = data[axes][slices]
									
								if isinstance(data.get(axes),arrays):
									data[axes] = data[axes].tolist()

							index = [*index[:-len(axis)],*axis]
							item = data if any(data[axes] is not None for axes in ALL if axes in data) else None
							iterable = settings[instance][key][obj][prop]
							inserter(index,item,iterable)
							
			settings[instance].pop(subinstance);
			grid[instance].pop(subinstance);

	# set layout
	# TODO: Check cases of settings containing multiple nrows,ncols + additional reshaped axes induced rows and columns
	layout = {}
	for instance in settings:

		for index,subinstance in enumerate(settings[instance]):		
			
			sublayout = settings[instance][subinstance]['style']['layout']
			
			if not layout.get(instance):
				layout[instance] = sublayout
			layout[instance].update({
				**layout[instance],
				**{attr: max(sublayout[attr]*grid[instance][subinstance][GRID.index(attr[1:-1])],layout[instance][attr])
					if (sublayout[attr] is not None) and (layout[instance][attr] is not None) else None
					for attr in ['nrows','ncols']},
				**{attr: None for attr in ['index']},
				})

		for index,subinstance in enumerate(settings[instance]):
			
			sublayout = deepcopy(layout[instance])

			index = sublayout['index']-1 if sublayout['index'] is not None else index
			nrow = (index - index%sublayout['ncols'])//sublayout['ncols']
			ncol = index%sublayout['ncols']

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

			grid[instance][subinstance] = [sublayout['n%ss'%(GRID[i])] for i in range(LAYOUTDIM)]

	# Set kwargs
	for instance in list(settings):
	
		logger.log(info*verbose,"Setting : %s"%(instance))

		for subinstance in list(settings[instance]):
			
			if not settings[instance][subinstance].get(obj):
				continue

			position = [int(i) for i in subinstance.split(delim)[-LAYOUTDIM:]]


			# variables

			values = {}
			for prop in PLOTS:
				
				if prop not in settings[instance][subinstance][obj]:
					continue

				values[prop] = {}

			for prop in values:

				labels = list(natsorted(set(label
					for data in search(settings[instance][subinstance][obj][prop])
					if (data)
					for label in [*data[OTHER],*data[OTHER][OTHER][OTHER]]
					if ((data) and (label not in [*ALL,OTHER]))
					)))

				for label in labels:
					value = {}
					value['value'] = list(realsorted(set(
							(data[OTHER][label] if not isinstance(data[OTHER][label],tuple) else None) if (
								(label in data[OTHER]) and not isinstance(data[OTHER][label],list)) else 
							to_tuple(data[OTHER][label]) if (
								(label in data[OTHER])) else 
							data[OTHER][data[OTHER][OTHER][OTHER][label].replace('@','')] if (
								(label in data[OTHER][OTHER][OTHER] and 
								(data[OTHER][OTHER][OTHER].get(label) is not None) and
								data[OTHER][OTHER][OTHER][label].replace('@','') in data[OTHER])) else data[OTHER][OTHER][OTHER][label] if (label in data[OTHER][OTHER][OTHER]) else None
							for data in search(settings[instance][subinstance][obj][prop]) if (
								((data) and ((label in data[OTHER]) or (label in data[OTHER][OTHER][OTHER]))))
							)))
					value['include'] = any((
							(((data[OTHER][OTHER]['legend']['include'] is not False) and (data[OTHER][OTHER]['legend']['exclude'] is not True))) and (
							(((not data[OTHER][OTHER]['legend']['include']) and (not data[OTHER][OTHER]['legend']['exclude']))) or
							(((not data[OTHER][OTHER]['legend']['include']) or (label in data[OTHER][OTHER]['legend']['include'])) and
							 ((not data[OTHER][OTHER]['legend']['exclude']) or (label not in data[OTHER][OTHER]['legend']['exclude']))
							))
							)
							for i in PLOTS
							if i in settings[instance][subinstance][obj]
							for data in search(settings[instance][subinstance][obj][i])
							if (data) 
							)
					value['sort'] = [k for (k,j) in sorted(set([(k,tuple(data[OTHER][OTHER]['legend']['sort']))
							for i in PLOTS
							if i in settings[instance][subinstance][obj]
							for data in search(settings[instance][subinstance][obj][i])
							if (data) and (data[OTHER][OTHER]['legend'].get('sort') is not None)
							for k in data[OTHER][OTHER]['legend']['sort']]),key=lambda i: i[-1].index(i[0]))]
					value['axes'] = False				
					value['label'] = any((
							(label in data[OTHER][OTHER][OTHER]) and 
							(label in data[OTHER]))# and (data[OTHER][OTHER][OTHER][label] is None))
							for data in search(settings[instance][subinstance][obj][prop]) 
							if (data)
							)
					value['other'] = any((
							(label in data[OTHER][OTHER][OTHER]) and 
							(label not in data[OTHER]) and 
							((data[OTHER][OTHER][OTHER].get(label) is not None) and
							(data[OTHER][OTHER][OTHER][label].replace('@','') in data[OTHER])))
							for data in search(settings[instance][subinstance][obj][prop])
							if (data)
							)
					value['legend'] = any((
							(label in data[OTHER][OTHER][OTHER]) and 
							(label not in data[OTHER]) and 
							((data[OTHER][OTHER][OTHER].get(label) is not None) and
							(data[OTHER][OTHER][OTHER][label].replace('@','') not in data[OTHER])))
							for data in search(settings[instance][subinstance][obj][prop])
							if (data)
							)
					value['attr'] = {
							**{attr: {string:  data[OTHER][OTHER][attr][string]
								for data in search(settings[instance][subinstance][obj][prop]) 
								if ((data) and attr in data[OTHER][OTHER])
								for string in data[OTHER][OTHER][attr]}
								for attr in ['texify','valify']},
							**{attr: {
								**{kwarg:[
								min((data[OTHER][OTHER][attr][kwarg][0]
									for data in search(settings[instance][subinstance][obj][prop]) 
									if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0),
								max((data[OTHER][OTHER][attr][kwarg][1]
									for data in search(settings[instance][subinstance][obj][prop]) 
									if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0),											
								] for kwarg in ['scilimits']},
								**{kwarg: 
									max((data[OTHER][OTHER][attr][kwarg]
									for data in search(settings[instance][subinstance][obj][prop]) 
									if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0) 
									for kwarg in ['decimals']},
								**{kwarg: 
									any((data[OTHER][OTHER][attr][kwarg]
									for data in search(settings[instance][subinstance][obj][prop]) 
									if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))))
									for kwarg in ['one']},										
								}
								for attr in ['scinotation']},
							}

					values[prop][label] = value

				
				labels = list(natsorted(set(label
					for data in search(settings[instance][subinstance][obj][prop])
					if (data)
					for label in data
					if ((data) and (label in [*ALL]))
					)))

				for label in labels:
					value = {}
					value['value'] = list(realsorted(set(i
							for data in search(settings[instance][subinstance][obj][prop]) if (data)
							for i in data.get(label,[]))))
					value['include'] = True
					value['sort'] = [k for (k,j) in sorted(set([(k,tuple(data[OTHER][OTHER]['legend']['sort']))
							for i in PLOTS
							if i in settings[instance][subinstance][obj]
							for data in search(settings[instance][subinstance][obj][i])
							if (data) and (data[OTHER][OTHER]['legend'].get('sort') is not None)
							for k in data[OTHER][OTHER]['legend']['sort']]),key=lambda i: i[-1].index(i[0]))]
					value['axes'] = True				
					value['label'] = False
					value['other'] = False
					value['legend'] = False
					value['attr'] = {
							**{attr: {string:  data[OTHER][OTHER][attr][string]
								for data in search(settings[instance][subinstance][obj][prop]) 
								if ((data) and attr in data[OTHER][OTHER])
								for string in data[OTHER][OTHER][attr]}
								for attr in ['texify','valify']},
							**{attr: {
								**{kwarg:[
								min((data[OTHER][OTHER][attr][kwarg][0]
									for data in search(settings[instance][subinstance][obj][prop]) 
									if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0),
								max((data[OTHER][OTHER][attr][kwarg][1]
									for data in search(settings[instance][subinstance][obj][prop]) 
									if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0),											
								] for kwarg in ['scilimits']},
								**{kwarg: 
									max((data[OTHER][OTHER][attr][kwarg]
									for data in search(settings[instance][subinstance][obj][prop]) 
									if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0) 
									for kwarg in ['decimals']},
								**{kwarg: 
									any((data[OTHER][OTHER][attr][kwarg]
									for data in search(settings[instance][subinstance][obj][prop]) 
									if ((data) and (attr in data[OTHER][OTHER]) and (kwarg in data[OTHER][OTHER][attr]))))
									for kwarg in ['one']},										
								}
								for attr in ['scinotation']},
							}

					values[prop][label] = value

					values[prop][label] = value

				for label in list(values[prop]):
					if any(label in values[i] for i in values if i not in [prop]):
						values[prop].pop(label);


			# setup values based attrs
			delimiters = ['@','__']
			for prop in settings[instance][subinstance][obj]:

				if not settings[instance][subinstance][obj].get(prop):
					continue
				
				for index,shape,data in search(settings[instance][subinstance][obj][prop],returns=True):
					
					if not data:
						continue

					for attr in data:

						if (prop in PLOTS) and (attr in [*ALL,OTHER]):
							continue

						value = deepcopy(data[attr])

						if value is None:
							continue
						elif isinstance(value,str) and any((value.startswith(delimiter) and value.endswith(delimiter)) for delimiter in delimiters):
							label,value = value,None
							value = {label:value}
						elif isinstance(value,dict) and any((label.startswith(delimiter) and label.endswith(delimiter)) for label in value for delimiter in delimiters):
							value = {label: value[label] for label in value}
						else:
							continue

						for label in list(value):
							for delimiter in delimiters:
								
								if not (label.startswith(delimiter) and label.endswith(delimiter)):
									continue


								label,val = label.replace(delimiter,''),value.pop(label)

								if delimiter in ['@']:

									if not any(label in values[prop] for prop in values):
										continue
									
									if isinstance(val,dict):
										defaults = {'value':None,'type':None,'func':None}
										val.update({prop: val.get(prop,defaults[prop]) for prop in defaults})

										if isinstance(val['func'],str):
											func = load(val['func'],default=None)
										else:
											func = val['func']

										if prop not in PLOTS:
											item = None
											items = [values[prop][label]['value'] for prop in values if label in values[prop]][0]
										elif callable(func):
											item = func({
												**{attr:data[attr] for attr in ALL if attr in data},
												**{attr:data[OTHER][attr] for attr in data[OTHER] for prop in values if label in values[prop] and attr in values[prop]}
												})
											items = func({
												**{attr:np.array([values[prop][label]['value'] for prop in values if label in values[prop]][0]) for attr in ALL if attr in data},
												**{attr:data[OTHER][attr] for attr in [label]},
												**{attr:np.array([values[prop][attr]['value'] for prop in values if label in values[prop]][0]) for attr in data[OTHER] for prop in values if (label != attr) and (label in values[prop]) and (attr in values[prop])}
												})											
										elif prop in PLOTS:
											if label in data:
												item = [i for i in data[label]]
												items = [values[prop][label]['value'] for prop in values if label in values[prop]][0]
											else:
												item = data[OTHER].get(label)
												items = [values[prop][label]['value'] for prop in values if label in values[prop]][0]
										else:
											continue

									elif prop in PLOTS:
										if label in data:
											item = None
											items = [i for i in data[label]]
										else:
											item = data[OTHER].get(label)
											items = [values[prop][label]['value'] for prop in values if label in values[prop]][0]

									else:
										item = None
										items = [values[prop][label]['value'] for prop in values if label in values[prop]][0]

									if isinstance(item,arrays):
										item = item.tolist()
									
									if isinstance(items,arrays):
										items = items.tolist()

									if prop in PLOTS:
										if label not in data[OTHER]:
											continue
										else:
											value[label] = {
												'__item__': item,
												'__items__': items,
												'__value__': val
												}
									else:
										value[label] = {
											'__item__': item,
											'__items__': items,
											'__value__': val
											}
								
								elif delimiter in ['__']:
									if label not in [*GRID[:LAYOUTDIM],*INDEXES]:
										continue
									if label in GRID[:LAYOUTDIM]:
										item = position[GRID.index(label)]
										items = list(range(grid[instance][subinstance][GRID.index(label)]))

										value[label] = {
											'__item__': item,
											'__items__': items,
											'__value__': val
											}
									elif label in INDEXES:

										if prop in PLOTS:
											item = index[INDEXES.index(label)]
											items = list(range(shape[INDEXES.index(label)]))

											value[label] = {
												'__item__': item,
												'__items__': items,
												'__value__': val
												}	
										else:
											item = None
											items = [list(range(shape[INDEXES.index(label)])) 
													for prop in PLOTS if settings[instance][subinstance][obj].get(prop)
													for item,shape,data in search(settings[instance][subinstance][obj][prop],returns=True) if data][0]
											value[label] = {
												'__item__': item,
												'__items__': items,
												'__value__': val
												}
							

						if not value:
							labels = None
							value = None
						else:
							labels = list(value)
							value['__item__'] = tuple(value[label]['__item__'] for label in labels) if all(value[label]['__item__'] in value[label]['__items__'] for label in labels) else None
							value['__items__'] = list(realsorted(itertools.product(*(value[label]['__items__'] for label in labels))))
							value['__value__'] = [value[label]['__value__'] for label in labels][0]
							value['__index__'] = value['__items__'].index(value['__item__']) if value['__item__'] in value['__items__'] else None
							value['__size__'] = len(value['__items__'])

							for label in labels:
								value.pop(label);

						data[attr] = value

			# set colorbar
			prop = 'set_colorbar'
			for data in search(settings[instance][subinstance][obj].get(prop)):

				if not data:
					continue

				attr = 'value'
				delimiter = '__'
				if (data.get(attr) is None):
				
					data[attr] = []
				
				elif isinstance(data[attr],dict) and all(prop.startswith(delimiter) and prop.endswith(delimiter) for prop in data[attr]):

					if all(len(i)>1 for i in data[attr]['__items__']):
						items = [data[attr]['__items__'].index(i)/max(1,data[attr]['__size__']-1) for i in data[attr]['__items__']]
					else:
						items = [i[0] for i in data[attr]['__items__']]

					value = data[attr]['__value__']
					indices = [data[attr]['__items__'].index(i)/max(1,data[attr]['__size__']-1) for i in data[attr]['__items__']]

					if isinstance(value,dict):
						defaults = {'value':None,'type':None,'func':None}
						value.update({prop: value.get(prop,defaults[prop]) for prop in defaults})

						if value['type'] in ['value']:
							value = items
						elif value['type'] in ['index']:
							value = indices
						else:
							value = indices

					elif value is not None:
						value = indices
					else:
						value = None

					data[attr] = value

				else:
					items = data[attr]

				attr = 'set_%slabel'
				kwarg = '%slabel'

				for axes in ['',*AXES]:
					if data.get(attr%(axes)) is None:
						continue
					
					value = texify(data[attr%(axes)][kwarg%(axes)])

					data[attr%(axes)][kwarg%(axes)] = value

				attr = 'set_%sticks'
				kwarg = 'ticks'
				for axes in ['',*AXES]:
					if data.get(attr%(axes)) is None:
						continue
					else:
						norm = data.get('norm')
						scale = data.get('scale')

						if norm is None:
							norm = {'vmin':min(data.get('value',[]),default=0),'vmax':max(data.get('value',[]),default=1)}
						elif not isinstance(norm,dict):
							norm = {'vmin':min(norm),'vmax':max(norm)}
						else:
							norm = {'vmin':norm.get('vmin',min(data.get('value',[]),default=0)),'vmax':norm.get('vmax',max(data.get('value',[]),default=1))}

						value = [min(min(data.get('value',[]),default=0),norm['vmin']),max(max(data.get('value',[]),default=1),norm['vmax'])]

						if isinstance(data[attr%(axes)].get(kwarg),int):
							
							size = data[attr%(axes)][kwarg]
							if data[attr%(axes)][kwarg] == 1:
								value = np.array(value)							
								value = [(value[0]+value[1])/2]
							elif scale in ['linear']:
								value = np.array(value)
								value = np.linspace(*value,size)
							elif scale in ['log']:
								value = np.log10(value)
								value = np.logspace(*value,size)
							else:
								value = np.array(value)
								value = np.linspace(*value,size)
						else:
							value = data[attr%(axes)][kwarg]

						if isinstance(value,arrays):
							value = value.tolist()

						data[attr%(axes)][kwarg] = value

				attr = 'set_%sticklabels'
				kwarg = 'ticklabels'
				for axes in ['',*AXES]:
					if data.get(attr%(axes)) is None:
						continue
					else:
						norm = data.get('norm')
						scale = data.get('scale')
						if norm is None:
							norm = {'vmin':min(data.get('value',[]),default=0),'vmax':max(data.get('value',[]),default=1)}
						elif not isinstance(norm,dict):
							norm = {'vmin':min(norm),'vmax':max(norm)}
						else:
							norm = {'vmin':norm.get('vmin',0),'vmax':norm.get('vmax',1)}

						value = items
						if isinstance(data[attr%(axes)].get(kwarg),int):

							length = len(value)+1
							size = max(1,len(set((*data.get('value',[]),*value)))//min(len(set((*data.get('value',[]),*value))),data[attr%(axes)][kwarg]))
							size = size-1 if (length-1)//((length-1)//size) > size else size
							slices = slice(0,length,(length-1)//size)

							if data[attr%(axes)][kwarg] == 1:
								value = [(value[0]+value[-1])/2]
							else:
								value = value[slices]
						else:
							value = data[attr%(axes)][kwarg]

					data[attr%(axes)][kwarg] = [texify(scinotation(i,decimals=1,scilimits=[-1,4])) for i in value]


			# set legend
			prop = 'legend'
			attr = 'set_title'
			for data in search(settings[instance][subinstance][obj].get(prop)):
				
				if not data:
					continue

				separator = ',~'

				value = [
					{
						**{(prop,label):'%s'%(texify(label,texify=values[prop][label]['attr']['texify']))
							for prop,label in natsorted(set((
							(prop,label)
							for prop in values 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (values[prop][label]['include']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['legend']) and (len(values[prop][label]['value'])>1))))))},
						**{(prop,label):'%s'%(texify(label,texify=values[prop][label]['attr']['texify']))
							for prop,label in natsorted(set((
							(prop,label)
							for prop in values 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (values[prop][label]['include']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['other']) and (len(values[prop][label]['value'])>1))))))},
						**{(prop,label):'%s'%(texify(label,texify=values[prop][label]['attr']['texify'])) 
							for prop,label in natsorted(set((
							(prop,label)
							for prop in values 					
							for label in values[prop] 
							if ((not values[prop][label]['axes']) and (((values[prop][label]['include']) and (values[prop][label]['label']) and (len(values[prop][label]['value'])>1)) and 
								not (values[prop][label]['other']))))))},
					},					
					{
						**{(prop,label):'%s%s%s'%(
							texify(label),' : ' if label else '',
							separator.join([texify(scinotation(value,**values[prop][label]['attr']['scinotation']),texify=values[prop][label]['attr']['texify']) 
									for value in values[prop][label]['value']]))
							for prop in values 
							for label in natsorted(set((
							label 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (values[prop][label]['include']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['legend']) and (len(values[prop][label]['value'])==1))))))},
						**{(prop,label):'%s%s%s'%(
							texify(label),' : ' if label else '',
							separator.join([texify(scinotation(value,**values[prop][label]['attr']['scinotation']),texify=values[prop][label]['attr']['texify']) 
									for value in values[prop][label]['value']]))
							for prop in values 
							for label in natsorted(set((
							label 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (values[prop][label]['include']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['other']) and (len(values[prop][label]['value'])==1))))))},
					},
					]
				
				def sorter(value):
					sorter = [i for prop in values for label in values[prop] for i in (values[prop][label]['sort'] if values[prop][label]['sort'] else [])]
					if sorter:
						sorter = sorted(set(value),key=lambda i,sorter=sorter: sorter.index(i[-1]) if i[-1] in sorter else len(sorter))
					else:
						sorter = natsorted(set(value))
					return sorter

				if data.get('multiline',None):
					separator = '\n'
				else:
					separator = '~,~'
				delimiter = '~,~'

				value = [[i[k] for k in sorter(i)] for i in value if i]
				value = separator.join([delimiter.join(i) for i in value]).replace('$','')

				if isinstance(data.get(attr),str) and data[attr].count('%s'):
					data[attr] = data[attr]%(value)
				elif value:
					data[attr] = value
				else:
					data[attr] = None


			# set kwargs data
			for prop in PLOTS:

				if not settings[instance][subinstance][obj].get(prop):
					continue

				for index,shape,data in search(settings[instance][subinstance][obj][prop],returns=True):

					if not data:
						continue

					slices = []
					subslices = [data[OTHER][OTHER].get('slice'),data[OTHER][OTHER].get('labels')]
					for subslice in subslices:
						if subslice is None:
							subslice = [slice(None)]
						elif isinstance(subslice,dict):
							subslice = {
								axes if (axes in data) else [subaxis 
										for subaxis in ALL if ((subaxis in data[OTHER]) and 
											(data[OTHER][subaxis]['label']==axes))][0]: 
								subslice[axes] for axes in subslice if (
								(not isinstance(subslice[axes],str)) or
								((axes in data) or any(data[OTHER][subaxis]['label']==axes 
									for subaxis in data[OTHER] if (
									(subaxis in ALL) and (subaxis in data[OTHER]))))
								)
								}

							if subslice:
								subslice = [
									conditions([parse(axes,subslice[axes],{axes: np.array(data[axes])},verbose=verbose) 
									for axes in subslice if isinstance(subslice[axes],str)],op='and'),
									*[slice(*subslice[axes]) for axes in subslice 
									 if not isinstance(subslice[axes],str)]
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
					slices = [subslice for subslice in slices if subslice is not None]

					wrappers = data[OTHER][OTHER].get('wrapper')
					if wrappers is None:
						wrappers = {}
					else:
						wrappers = {attr: load(wrappers[attr],default=None) for attr in wrappers if attr in ALL}

					normalize = data[OTHER][OTHER].get('normalize')
					normalizations = {
						'size': (lambda axes,data: (np.array(data[axes])/(len(data[axes])-1)) if (len(data[axes])>1) else np.array([0.5])),
						None: (lambda axes,data: data[axes]),
					}
					if not normalize:
						normalize = {}
					elif not isinstance(normalize,dict):
						normalize = {
							axes: normalizations['size']
							for axes in normalize
						}
					else:
						normalize = {
							axes: normalizations.get(normalize[axes],normalizations[None])
							for axes in normalize
						}


					for attr in data:
						
						if data.get(attr) is None:
							continue

						value = data[attr]

						if attr in [OTHER]:
						
							if value[OTHER].get('labels') is not None:
								for label in value[OTHER]['labels']:
									if (label in value) and (label not in ALL) and not parse(label,value[OTHER]['labels'][label],value,verbose=verbose):
										data.clear()
										break
						
						elif attr in ALL:
							
							if isinstance(data.get(attr),scalars):
								continue							

							value = np.array(value)

							if wrappers.get(attr):
								value = wrappers[attr]({
									**{data[OTHER][attr][OTHER]: data[attr] for attr in data if attr in ALL},
									**{attr: data[OTHER][attr] for attr in data[OTHER]},
									})

							if normalize.get(attr):
								value = normalize[attr](attr,data)

							for subslice in slices:
								value = value[subslice]


							value = np.array([valify(i,valify=data[OTHER][OTHER].get('valify')) for i in value])

						else:
							
							delimiter = '__'
							if isinstance(data[attr],dict) and all(prop.startswith(delimiter) and prop.endswith(delimiter) for prop in data[attr]):

								if all(len(i)>1 for i in data[attr]['__items__']):
									items = [data[attr]['__items__'].index(i)/max(1,data[attr]['__size__']-1) for i in data[attr]['__items__']]
								else:
									items = [i[0] for i in data[attr]['__items__']]
								
								value = data[attr]['__value__']
								indices = [data[attr]['__items__'].index(i)/max(1,data[attr]['__size__']-1) for i in data[attr]['__items__']]

								if isinstance(data[attr]['__value__'],str):
									value = data[attr]['__value__']
								
								elif isinstance(value,dict):
									defaults = {'value':None,'type':None}
									value.update({prop: value.get(prop,defaults[prop]) for prop in defaults})

									if value['type'] in ['value']:
										tmp = items
									elif value['type'] in ['index']:
										tmp = indices
									else:
										tmp = indices

									if isinstance(value['value'],dict):
										if value['type'] in ['value','index']:
											prop = 'value'
										elif value['type'] in value['value']:
											prop = value['type']
										if data[attr]['__index__'] is not None:
											value['value'][prop] = tmp[data[attr]['__index__']]
											value['value']['values'] = tmp
										else:
											value['value'][prop] = None
											value['value']['values'] = tmp
									else:
										if data[attr]['__index__'] is not None:
											value['value'] = None
										else:
											value['value'] = tmp[data[attr]['__index__']]

									value = value['value']

								elif not isinstance(data[attr]['__value__'],scalars):
									value = data[attr]['__value__'][data[attr]['__index__']%len(data[attr]['__value__'])] if not isinstance(data[attr]['__value__'],str) else data[attr]['__value__']
								else:
									value = data[attr]['__value__']

								if attr in ['color','ecolor']:
									pass
								elif attr in ['alpha']:
									value = (data[attr]['__index__'] + 0.5)/(data[attr]['__size__'])
								elif attr in ['zorder']:
									value = 1000*data[attr]['__index__']
								elif attr in ['marker']:
									pass
								elif attr in ['linestyle']:
									value = (data[attr]['__index__'] + 0.5)/(data[attr]['__size__'])
									value = '-' if value < 1/3 else '--' if value < 2/3 else '---'
								else:
									pass

						data[attr] = value

			# set title and axes label
			prop = 'set_%slabel'
			attr = '%slabel'
			for axes in ['',*AXES]:
				prop = 'set_%slabel' if axes in AXES else 'set_%stitle'
				for data in search(settings[instance][subinstance][obj].get(prop%(axes))):

					if not data:
						continue

					if data.get(attr%(axes)) is None:
						if axes in AXES:
							data[attr%(axes)] = [data[OTHER][axes]['label'] for prop in PLOTS if settings[instance][subinstance][obj].get(prop) for data in search(settings[instance][subinstance][obj][prop]) if OTHER in data]

							data[attr%(axes)] = data[attr%(axes)][0] if data[attr%(axes)] else None
					
					if isinstance(data[attr%(axes)],list):
						if not all(isinstance(i,list) for i in data[attr%(axes)]):
							data[attr%(axes)] = [[i] for i in data[attr%(axes)]]
						data[attr%(axes)] = data[attr%(axes)][position[0]%len(data[attr%(axes)])][position[1]%len(data[attr%(axes)][position[0]%len(data[attr%(axes)])])]

					if isinstance(data[attr%(axes)],str):
						data[attr%(axes)] = data[attr%(axes)]%(tuple(str(position[i]) if grid[instance][subinstance][i]>1 else '' for i in range(data[attr%(axes)].count('%s'))))

					data[attr%(axes)] = texify(data[attr%(axes)])


			# set label
			attr = 'label'
			for prop in PLOTS:

				if not settings[instance][subinstance][obj].get(prop):
					continue

				for data in search(settings[instance][subinstance][obj][prop]):

					if not data or not data.get(OTHER) or not data[OTHER].get(OTHER):
						continue

					value = {
						**{label: (texify(
							scinotation((data[OTHER][data[OTHER][OTHER][OTHER][label].replace('@','')]
							if data[OTHER][OTHER][OTHER][label].replace('@','') in data[OTHER] else 
								data[OTHER][OTHER][OTHER][label].replace('$','')) if label in data[OTHER][OTHER][OTHER] else data[OTHER][OTHER]['legend']['label'].get(label)),
								**data[OTHER][OTHER].get('scinotation',{}),
								texify=data[OTHER][OTHER].get('texify'))
							)
							for label in natsorted(set((
							label 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['legend']) and (len(values[prop][label]['value'])>1))))))},
						**{label: (texify(
							scinotation((data[OTHER][data[OTHER][OTHER][OTHER][label].replace('@','')]
								if data[OTHER][OTHER][OTHER][label].replace('@','') in data[OTHER] else 
								data[OTHER][OTHER][OTHER][label].replace('$','')) if label in data[OTHER][OTHER][OTHER] else data[OTHER][OTHER]['legend']['label'].get(label) ,
								**data[OTHER][OTHER].get('scinotation',{})),
								texify=data[OTHER][OTHER].get('texify'))
							)
							for label in natsorted(set((
							label 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['other']) and (len(values[prop][label]['value'])>1))))))},
						**{label: (texify(scinotation(data[OTHER][label],
							**data[OTHER][OTHER].get('scinotation',{})),texify=data[OTHER][OTHER].get('texify')) 
							if values[prop][label]['label'] else texify(
								scinotation(data[OTHER][data[OTHER][OTHER][OTHER][label].replace('@','')] 
									if data[OTHER][OTHER][OTHER][label].replace('@','') in data[OTHER] else 
									data[OTHER][OTHER][OTHER][label].replace('$',''),
									**data[OTHER][OTHER].get('scinotation',{})),
								texify=data[OTHER][OTHER].get('texify'))
							) if label in data[OTHER][OTHER][OTHER] else texify(scinotation(data[OTHER][OTHER]['legend']['label'].get(label) ,
																			**data[OTHER][OTHER].get('scinotation',{})),texify=data[OTHER][OTHER].get('texify')) 
							for label in natsorted(set((
							label 
							for label in values[prop] 
							if ((not values[prop][label]['axes']) and ((((values[prop][label]['label']) and (len(values[prop][label]['value'])>1)) and 
								not (values[prop][label]['other'])))))))},							
						
						}

					def sorter(value):
						if data[OTHER][OTHER]['legend']['sort'] is not None:
							sorter = [*[i for i in data[OTHER][OTHER]['legend']['sort']],*[i for i in natsorted(set(value)) if i not in data[OTHER][OTHER]['legend']['sort']]]
							sorter = sorted(set(value),key=lambda i,sorter=sorter: sorter.index(i))
						else:
							sorter = natsorted(set(value))
						return sorter
				
					separator = ',~'

					value = [value[label] for label in sorter(value)
							if (
							(((data[OTHER][OTHER]['legend']['include'] is not False) and (data[OTHER][OTHER]['legend']['exclude'] is not True))) and (
							(((not data[OTHER][OTHER]['legend']['include']) and (not data[OTHER][OTHER]['legend']['exclude']))) or
							(((not data[OTHER][OTHER]['legend']['include']) or (label in data[OTHER][OTHER]['legend']['include'])) and
							 ((not data[OTHER][OTHER]['legend']['exclude']) or (label not in data[OTHER][OTHER]['legend']['exclude']))
							)))
							]

					value = separator.join(value)

					value = value if value else None

					data[attr] = value	

			
			# savefig
			prop = 'savefig'
			attr = 'fname'
			for data in search(settings[instance][subinstance]['fig'].get(prop)):

				value = [split(path,directory_file=True),instance,data[attr]]
				value = [i for i in value if i is not None]
				value = join(delim.join(value),ext=split(path,ext=True))
				
				data[attr] = value
		

	# Plot data
	for instance in settings:

		logger.log(info,"Plotting : %s"%(instance))

		fig[instance],ax[instance] = plot(fig=fig[instance],ax=ax[instance],settings=settings[instance])

	return


def postprocesser(hyperparameters,pwd=None,cwd=None,verbose=None):
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
	- Aggregate functions of dependent (mean,std) for each group
	- Assign new labels for functions with label.function 
	- Regroup with non-null labels
	
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
	postprocesser(hyperparameters,pwd,cwd,verbose=verbose)

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
		'data':lambda kwarg,wrappers,kwargs: kwargs['data'][-1] if len(kwargs['data']) ==1 else kwargs['data'],
	}

	args = argparser(arguments,wrappers)

	main(*args,**args)