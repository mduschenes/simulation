#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,traceback
from functools import partial,wraps
import numpy as np
import scipy as sp
import scipy.stats
import scipy.special
import pandas as pd
from pandas.api.types import is_float_dtype
from natsort import natsorted,realsorted

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,copy
from src.utils import array,dataframe,expand_dims,conditions,prod,bootstrap
from src.utils import to_key_value,to_tuple,to_number,to_str,to_int,to_position,to_index,is_iterable,is_number,is_nan,is_numeric
from src.utils import e,pi,nan,scalars,integers,floats,iterables,arrays,delim,nulls,null,Null,scinotation
from src.iterables import search,inserter,indexer,sizer,permuter,regex,Dict
from src.io import load,dump,join,split,exists,glob
from src.fit import fit
from src.postprocess import postprocess
from src.plot import plot,AXES,VARIANTS,FORMATS,ALL,VARIABLES,OTHER,PLOTS

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
	def __init__(self,df,by=[],**kwargs):
		'''
		Null groupby wrapper for dataframe
		Args:
			df (dataframe): dataframe
			by (iterable[str,tuple]): Attributes to group by
			kwargs (dict): Additional keyword arguments for groupby
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
	nulls = {'$':'','\\textrm':'','\\text':''}

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
				for null in nulls:
					string = string.replace(null,nulls[null])
					
		except AttributeError:
			string = None

	return string

def copier(key,value,copy):
	'''
	Copy value based on associated key 

	Args:
		key (string): key associated with value to be copied
		value (object): data to be copied
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
	Returns:
		Copy of value
	'''

	# Check if copy is a dictionary and key is in copy and is True to copy value
	if ((not copy) or (isinstance(copy,dict) and (not copy.get(key)))):
		return value
	else:
		return deepcopy(value)





def setter(iterable,elements,delimiter=False,copy=False,reset=False,clear=False,default=None):
	'''
	Set nested value in iterable with nested elements keys
	Args:
		iterable (dict): dictionary to be set in-place with value
		elements (dict): Dictionary of keys of delimiter separated strings, or tuple of string for nested keys, and values to set 
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		reset (bool): boolean on whether to replace value at key with value, or update the nested dictionary
		clear (bool): boolean of whether to clear iterable when the element's value is an empty dictionary
		default(callable,None,bool,iterable): Callable function with signature default(key_iterable,key_elements,iterable,elements) to modify value to be updated based on the given dictionaries, or True or False to default to elements or iterable values, or iterable of allowed types
	'''

	if (not isinstance(iterable,(dict,list))) or (not isinstance(elements,dict)):
		return

	# Setup default as callable
	if default is None:
		func = lambda key_iterable,key_elements,iterable,elements: elements.get(key_elements)
	elif default is True:
		func = lambda key_iterable,key_elements,iterable,elements: elements.get(key_elements)
	elif default is False:
		func = lambda key_iterable,key_elements,iterable,elements: iterable.get(key_iterable,elements.get(key_elements))
	elif default in ['none','None']:
		func = lambda key_iterable,key_elements,iterable,elements: elements.get(key_elements) if iterable.get(key_iterable,elements.get(key_elements)) is None else iterable.get(key_iterable,elements.get(key_elements))
	elif not callable(default):
		types = tuple(default)
		def func(key_iterable,key_elements,iterable,elements,types=types): 
			i = iterable.get(key_iterable,elements.get(key_elements))
			e = elements.get(key_elements,i)
			return e if isinstance(e,types) else i
	else:
		func = default

	# Clear iterable if clear and elements is empty dictionary
	if clear and elements == {}:
		iterable.clear()

	# Set nested elements
	for element in elements:

		# Get iterable, and index of tuple of nested element key
		i = iterable
		index = 0

		# Convert string instance of elements to list, splitting string based on delimiter delimiter
		try:
			if (
				(isinstance(element,str) and delimiter) and 
				(element not in iterable)):
				#((element.count(delimiter)>0) and ((element not in iterable)) or (element.split(delimiter)[0] in iterable))):
				# e = element.split(delimiter)

				e = []
				_element = element.split(delimiter)
				_iterable = iterable
				while _element and isinstance(_iterable,dict):
					for l in range(len(_element),-1,-1):
						_e = delimiter.join(_element[:l])

						if _e in _iterable:
							_iterable = _iterable.get(_e)
							e.append(_e)
							_element = _element[l:]
							break
					if l == 0:
						e.extend(_element)
						break
				e = tuple(e)

			elif is_iterable(element,exceptions=scalars):
				e = tuple(element)
			else:
				e = tuple((element,))

			# Update iterable with elements 
			while index<(len(e)-1):
				if isinstance(i,list):
					if (e[index] >= len(i)):
						i.extend([[] if isinstance(e[index+1],integers) else {} for j in range(e[index]-len(i)+1)])
				elif (isinstance(i,dict) and (not isinstance(i.get(e[index]),(dict,list)))):
					i[e[index]] = [] if isinstance(e[index+1],integers) else {}
				i = i[e[index]]
				index+=1

			# try:
			value = copier(element,func(e[index],element,i,elements),copy)

			if isinstance(i,list) and (e[index] >= len(i)):
				i.extend([{} for j in range(e[index]-len(i)+1)])

			if reset:
				i[e[index]] = value
			elif e[index] not in i or not isinstance(i[e[index]],(dict,list)):
				i[e[index]] = value
			elif isinstance(elements[element],dict):
				setter(i[e[index]],elements[element],delimiter=delimiter,copy=copy,reset=reset,clear=clear,default=default)
			else:
				i[e[index]] = value
		except:
			pass

	return


def getter(iterable,elements,default=None,delimiter=False,copy=False):
	'''
	Get nested value in iterable with nested elements keys

	Args:
		iterable (dict): dictionary of values
		elements (str,iterable[str]): delimiter separated string or list to nested keys of location to get value
		default (object): default data to return if elements not in nested iterable
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
	Returns:
		value (object): Value at nested keys elements of iterable
	'''	

	# Convert string instance of elements to list, splitting string based on delimiter delimiter
	if isinstance(elements,str):
		if delimiter and (elements not in iterable):
			elements = elements.split(delimiter)
		else:
			elements = [elements]

	# Get nested element if iterable, based on elements
	if not isinstance(elements,(list,tuple)):
		# elements is object and value is to be got from iterable at first level of nesting
		try:
			return copier(elements,iterable[elements],copy)
		except:
			return default
	elif not elements:
		return copier(elements,iterable,copy)
	else:
		# elements is list of nested keys and the nested values are to be extracted from iterable
		try:
			i = iterable
			e = 0
			while e<len(elements):
				i = i[elements[e]]
				e+=1			
			return copier(elements[e-1],i,copy)
		except:
			return default

	return default


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
		value = valify.get(value,valify.get(str(value),value))
	except:
		pass

	return value


def setup(data,plots,processes,pwd=None,cwd=None,verbose=None):
	'''
	Setup data, plots, processes
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		plots (str,dict): Path to or dictionary of plot plots
		processes (str,dict): Path to or dictionary of process plots
		pwd (str): Root path of plots, processes		
		cwd (str): Root path of data
		verbose (bool): Verbosity		
	Returns:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		plots (dict): Plot plots
		processes (dict): Process plots
	'''

	# Set plot plots
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

	# Load process plots
	path = join(plots,root=pwd) if isinstance(plots,str) else None
	default = None if isinstance(plots,str) else plots
	wrapper = None	
	plots = load(path,default=default,wrapper=wrapper,verbose=verbose)

	# Load process processes
	path = join(processes,root=pwd) if isinstance(processes,str) else None
	default = None if isinstance(processes,str) else processes
	wrapper = None
	processes = load(path,default=default,wrapper=wrapper,verbose=verbose)
	

	# Parse process processes and plots
	
	if (plots is None) or (processes is None):
		return data,plots,processes


	# Process plots and processes dictionaries
	attr = 'patterns'
	patterns = processes.get(attr)
	for iterable in [plots,processes]:
		regex(iterable,patterns)
	processes[attr] = patterns

	obj = 'ax'

	for instance in list(plots):

		if (plots.get(instance) is None):
			plots.pop(instance,None);
			continue

		if all(subinstance in defaults for subinstance in plots[instance]):
			plots[instance] = {str(None): plots[instance]}

		for subinstance in plots[instance]:
			
			setter(plots[instance][subinstance],defaults,delimiter=delim,default=False)

			if not plots[instance][subinstance].get(obj):
				continue
			for prop in PLOTS:
				if not plots[instance][subinstance][obj].get(prop):
					continue
				tmp = []
				for index,shape,item in search(plots[instance][subinstance][obj][prop],returns=True):
					index = [*index,*[0]*(INDEXDIM-len(shape))]
					inserter(index,item,tmp)
				plots[instance][subinstance][obj][prop] = tmp



	defaults = {
		'path':{},
		'load':None,
		'dump':None,
		'reset':None,
		'convert':None,
		'patterns':None,
		'plot':None,
		'process':None,
		'postprocess':None,
		}
	setter(processes,defaults,delimiter=delim,default=False)


	# Get paths
	path = data if isinstance(data,str) else processes.get('path') if isinstance(processes.get('path'),str) else None
	processes['path'] = {} if not isinstance(processes.get('path'),dict) else processes.get('path')
	processes['file'],processes['directory'],processes['ext'] = {},{},{}
	processes['cwd'],processes['pwd'] = cwd,pwd
	defaults = {
		'data': 	join(cwd,'**',join(split(path,file=True),ext='tmp'),ext=split(data,ext=True) if isinstance(data,str) else 'hdf5'),
		'tmp': 	join(cwd,join(split(path,file=True),ext='tmp'),ext=split(data,ext=True) if isinstance(data,str) else 'hdf5'),
		'metadata': join(cwd,join(split(path,file=True),ext=None),ext=split(plots,ext=True) if isinstance(plots,str) else 'json'),
	}
	processes['path'].update({attr:defaults[attr] for attr in defaults if processes['path'].get(attr) is None})
	for attr in processes['path']:
		processes['directory'][attr] = split(processes['path'][attr],directory=True) if split(processes['path'][attr],directory=True) else cwd
		processes['file'][attr],processes['ext'][attr] = split(processes['path'][attr],file=True,ext=True)
		processes['path'][attr] = join(processes['directory'][attr],processes['file'][attr],ext=processes['ext'][attr])

	# Set instances
	attr = 'instance'
	if processes.get(attr) is None:
		processes[attr] = {}
	elif isinstance(processes.get(attr),(bool,*integers)):
		processes[attr] = {instance: bool(processes[attr][instance]) for instance in plots}
	elif isinstance(processes.get(attr),list):
		processes[attr] = {**{instance: False for instance in plots},**{instance: True for instance in processes[attr]}}
	elif isinstance(processes.get(attr),dict):
		processes[attr] = {**{instance: False for instance in plots},**{instance: processes[attr][instance] for instance in processes[attr]}}

	for instance in list(plots):

		if (not processes.get(attr,{}).get(instance)) or (not plots[instance]):
			plots.pop(instance,None);
			continue

	# Get plot fig and axes
	fig,ax = processes.get('fig'),processes.get('ax')
	if fig is None:
		fig = {}
	if ax is None:
		ax = {}

	for instance in plots:
		if instance not in fig:
			fig[instance] = None
		if instance not in ax:
			ax[instance] = None
	processes['fig'],processes['ax'] = fig,ax


	# Get configuration
	options = {'position':None,'value':None,'kwargs':None}
	configuration = processes.get('configuration',processes.get('instance',{}))

	if isinstance(configuration,dict):
		if all(instance in plots for instance in configuration if configuration[instance]):
			configuration = {instance: configuration[instance] if isinstance(configuration[instance],iterables) else [configuration[instance]] for instance in configuration}
		else:
			configuration = {instance: [configuration] for instance in plots}
	elif isinstance(configuration,iterables):
		if all(isinstance(config,dict) for config in configuration):
			configuration = {instance:[config for config in configuration] for instance in plots}
		elif all(isinstance(config,(dict,*iterables)) and len(config)==2 for config in configuration):
			configuration = {instance:config for instances,config in configuration for instance in instances if instance in plots}
	else:
		raise Exception("Incorrect configuration format\n%r"%(configuration))

	configuration = {instance:configuration[instance] for instance in configuration if instance in plots}
	configuration = {instance: [{**config,**{option:config.get(option,options[option]) for option in options}}
		for config in configuration[instance]]
		for instance in configuration}

	processes['configuration'] = configuration


	# Get texify
	texify = processes.get('texify',{})
	usetex = processes.get('usetex',False)
	processes['texify'] = lambda string,texify=None,_texify=texify,usetex=usetex: Texify(
		string,
		texify={**(_texify if _texify is not None else {}),**(texify if texify is not None else {})},
		usetex=usetex)

	# Get valify
	valify = processes.get('valify',{})
	useval = processes.get('useval',True)
	processes['valify'] = lambda value,valify=None,_valify=valify,useval=useval: Valify(
		value,
		valify={**(_valify if _valify is not None else {}),**(valify if valify is not None else {})},
		useval=useval)

	return data,plots,processes

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

	strings = [
		*[[*dimensions[:i],*other] for i in range(dim,0,-1)],
		# *[[i,*other] for i in dimensions]
		]


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
					'label':None,'include':None,'exclude':None,'sort':None,'parse':None,
				},
				'wrapper':{},
				'texify':{},
				'valify': {},		
				'scinotation':{'scilimits':[0,2],'decimals':0,'one':False},
				'args':None,
				'kwargs':None,
	}

	keys = {}

	for items in strings:
		types = (list,dict,)
		key = search(dictionary,items=items,returns=True,types=types)
		
		key = {tuple(index): dict(zip(items,item)) for index,shape,item in key}

		# key = {index:key[index] for index in key
		# 	if (index) and (
		# 	(isinstance(index[-1],str) and index[-1] in PLOTS) or 
		# 	(isinstance(index[-1],integers)))
		# 	}

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
					keys[name][attr] = {attr: dict((parser(prop,separator=separator,default=default) for prop in keys[name][attr]))}
					# keys[name][attr] = dict((parser(prop,separator=separator,default=default) for prop in keys[name][attr]))

				if attr in keys[name][attr]:
					if isinstance(keys[name][attr][attr],dict):
						keys[name][attr][attr] = {prop: keys[name][attr][attr][prop] if keys[name][attr][attr][prop] is not None else default for prop in keys[name][attr][attr]}
					elif isinstance(keys[name][attr][attr],str):
						keys[name][attr][attr] = dict((parser(keys[name][attr][attr],separator=separator,default=default)),)
					else:
						keys[name][attr][attr] = dict((parser(prop,separator=separator,default=default) for prop in keys[name][attr][attr]))
				else:
					keys[name][attr] = {attr: keys[name][attr]}

				setter(keys[name][attr],defaults,delimiter=delim,default=False)
			
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
		value (str,iterable,dict[str,str,iterable],callable): value of condition, allowed string in 
			[None,
			'$value,$' (explicit value),
			'@key,@' (data value), 
			'#i,j,k,...#' (index value),
			'%start,stop,step%' (slice value),
			'*pattern,*' (regex pattern),
			':func,:' (function type), 			
			'<upper<' (exclusive upper bound value),
			'>lower>' (exclusive lower bound value),
			'<=upper<=' (inclusive upper bound value),
			'>=lower>=' (inclusive lower bound value),
			'==value,==' (include values),
			'!=value,!=' (exclude values),
			]
			or callable or string to load callable with signature value(key,data) -> bool
		data (dataframe): data of condition
		verbose (bool): Verbosity		
	Returns:
		out (dataframe): Condition on data indices
	'''
	negators = ['!','~']
	delimiters = ['$','@','*',':','#','%','<','>','<=','>=','==','!=']
	parserator = ';'
	separator = ','

	try:
		default = data.assign(**{key:True})[key]
	except:
		default = True

	out = default

	if isinstance(value,dict):
		value = value.get(key,null)

	if key not in data:
		pass
	elif value is null:
		pass
	elif value is None:
		pass
	elif callable(value):
		try:
			func = value
			out = func(key,data)
		except:
			pass
	elif isinstance(value,str):
		try:
			func = load(value,default=None)
			out = func(data)
		except:
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
									out = np.sort(data[key].unique())
									out = data[key].isin(out[[value if value >=0 else len(out)+value for value in values if value < len(out)]])
								except:
									if isinstance(default,bool):
										out = not default
									else:
										out = ~ default

						elif delimiter in ['%']: # Slice value start,stop,step
							parser = lambda value: (to_int(value) if len(value)>0 else None)
							values = [*(parser(value) for value in values),*[None]*(3-len(values))]
							values = [value for value in values if (value is not null)]
					
							if values and (values is not null):
								try:
									out = np.sort(data[key].unique())
									out = data[key].isin(out[slice(*values)])
								except:
									if isinstance(default,bool):
										out = not default
									else:
										out = ~ default

						elif delimiter in [':']: # Data value: func
							
							parser = lambda value: (to_str(value) if len(value)>0 else null)
							values = [i for value in values for i in [parser(value),value]]           
							values = [value for value in values if (value is not null)]

							if values and (values is not null):
								try:
									out = conditions([data[key]==getattr(data[key],value)() for value in values if hasattr(data[key],value)],op='or')
								except:
									if isinstance(default,bool):
										out = not default
									else:
										out = ~ default								

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
		except:
			try:
				out = data[key] in value
			except:
				if isinstance(default,bool):
					out = not default
				else:
					out = ~ default
	
	return out


def analyse(data,analyses=None,verbose=None):
	'''
	Analyse data, cleaning data, removing outliers etc
	Args:
		data (dataframe): data of attributes
		analyses (dict[str,iterable[iterable[dict]]]): Processes to analyse of the form 
			{analysis:[({attr:value},kwargs)]},
			allowed analysis strings in ['zscore','quantile','slice','parse','abs','log','log10','log2','replace','func']
		verbose (bool): Verbosity			
	Returns:
		out (dataframe): Analysed data
	'''

	analyses = copy(analyses)

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
			elif analysis in ['abs','log','log10','log2']:
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
					value = {attr: attrs[attr] if not isinstance(attrs[attr],dict) else attrs[attr].pop('value',None) for attr in attrs}
					kwargs = {attr: {} if not isinstance(attrs[attr],dict) else attrs[attr] for attr in attrs}
					out = data
					for attr in attrs:
						for kwarg in kwargs[attr]:
							function = kwargs[attr][kwarg]
							default = None
							function = load(function,default=default)
							if function is not None:
								out = function(data,**kwargs[attr])
					return out
			elif isinstance(analyses[analysis],(str,dict,*iterables)):
				if isinstance(analyses[analysis],str):
					analyses[analysis] = {analysis:{analyses[analysis]:{}}}
				elif isinstance(analyses[analysis],dict):
					analyses[analysis] = {analysis:analyses[analysis]}
				else:
					analyses[analysis] = {analysis:{i:{} for i in analyses[analysis]}}
				def func(attrs,data):
					value = {attr: {i:i for i in attrs[attr]} for attr in attrs}
					kwargs = {attr: {} if not isinstance(attrs[attr],dict) else attrs[attr] for attr in attrs}
					out = data
					for attr in attrs:
						for kwarg in kwargs[attr]:
							function = value[attr][kwarg]
							default = None
							function = load(function,default=default)
							if function is not None:
								out = function(data,**kwargs[attr][kwarg])
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

			args = copy(args)

			for attrs in args:
				if analysis in ['zscore','quantile','slice','parse']:
					value = func(attrs,data)
					value = value.to_numpy() if not isinstance(value,bool) else value
					out = conditions([out,value],op='and')
				elif analysis in ['abs','log','log10','log2','replace','func']:
					data = func(attrs,data)
				elif func is not None:
					data = func(attrs,data)

	if out is True:
		out = data
	else:
		out = data[out]

	return out


def sort(data,sorting,keys=None):
	'''
	Sort data by conditions
	Args:
		data (dataframe): data
		sorting (iterable[str],dict[str,str,iterable],iterable[dict[str,str,iterable]]): attributes or iterable of attibutes of conditions {attr or (attr_ for reverse): conditions}
		keys (callable,iterable[callable]): sorting key with signature func(element) -> index
	Returns:
		index (dict[int,int]): dictionary of original and sorted index of data
	'''

	index = {}

	if sorting is None:
		return index

	if data is None or not len(data):
		return index

	if isinstance(sorting,dict):
		sorting = [sorting]
	elif all(isinstance(attr,str) for attr in sorting):
		sorting = [{attr : None for attr in sorting}]
	
	if keys is None or callable(keys):
		keys = [keys for i in sorting]

	for sorts,key in zip(sorting,keys):

		sorts = {
			attr if (attr in data) else '_'.join(attr.split('_')[:-1]) : {
				'conditions':sorts[attr],
				'sort': True if attr in data else False
				} 
			for attr in sorts 
			if (attr in data) or ((attr.endswith('_')) and ('_'.join(attr.split('_')[:-1]) in data))
			}

		by = [attr for attr in sorts]
		slices = conditions([parse(attr,sorts[attr].get('conditions'),data) for attr in sorts],op='and')	
		ascending = [sorts[attr].get('sort',True) for attr in sorts]
		ignore_index = False
		indices = data[slices].index

		tmp = data[slices].sort_values(by=by,ascending=ascending,key=key,ignore_index=ignore_index)

		for i in range(len(indices)):
			index[indices[i]] = tmp.index[i]

	return index

def loader(data,plots,processes,verbose=None):
	'''
	Load data from plots and processes
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		plots (str,dict): Path to or dictionary of plot plots
		processes (str,dict): Path to or dictionary of process plots
		verbose (bool): Verbosity		
	'''

	if (data is None) or (plots is None) or (processes is None):
		return

	# Set data boolean
	new = False

	# Set metadata
	metadata = processes['path']['metadata']
	
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
				k = None

				for j in range(len(iterable.get(key_iterable))):

					if all((
						all(datum[OTHER][attr]['label']==axes[attr] for attr in axes) and 
						(len(datum[OTHER][OTHER][OTHER]) == len(labels)) and
						all(datum[OTHER][OTHER][OTHER][attr]==labels[attr] for attr in labels)
						)
						for datum in search(iterable.get(key_iterable)[j]) if datum and all(datum[OTHER].get(attr) is not None for attr in axes)):
						if k is not None:
							k = True
							break
						k = j

				if k is None:
					continue
				elif k is True:
					k = i

				for subindex,datum in enumerate(search(iterable.get(key_iterable)[k])):
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

	if processes['load'] and not processes['reset']:

		# Load plots
		path = metadata
		default = None
		tmp = load(path,default=default,verbose=verbose)

		new = exists(path) and tmp is not None

		if new:
			new = copy(plots)
			plots.update(tmp)
			tmp = new
			setter(plots,tmp,default=func)

		patterns = processes.get('patterns')
		regex(plots,patterns)

	else:

		# Load data
		path = data
		tmp = processes['path']['data'] if not exists(processes['path']['tmp']) else processes['path']['tmp']
		
		try:
			assert any(exists(i) for i in glob(tmp)) and not processes['reset']
			path = tmp
			wrapper = 'pd'
			default = None
			data = load(path,default=default,wrapper=wrapper,verbose=verbose)
		except:
			path = data
			wrapper = 'df'			
			default = None
			data = load(path,default=default,wrapper=wrapper,verbose=verbose)

		new = tmp is not None and data is not None

		if new and processes['dump']:
			path = processes['path']['tmp']
			wrapper = processes['convert'] if isinstance(processes['convert'],str) else 'pd'
			dump(data,path,wrapper=wrapper,verbose=verbose)


		# Get keys
		keys = find(plots)

		# Get functions of data
		apply(keys,data,plots,processes,verbose=verbose)

	
	# Check plots
	attr = 'instance'
	for instance in list(plots):
		if (not processes.get(attr,{}).get(instance)) or (not plots[instance]):
				plots.pop(instance,None);
				continue

	# Dump plots
	if processes['dump']:
		path = metadata
		wrapper = None
		dump(plots,path,wrapper=wrapper,verbose=verbose)

	return


def apply(keys,data,plots,processes,verbose=None):
	'''
	Apply functions based on keys to data
	Args:
		keys (dict): Keys of functions to apply
		data (dataframe): dataframe
		plots (dict): plots
		processes (dict): processes
		verbose (bool): Verbosity		
	'''

	if (keys is None) or (data is None) or (plots is None) or (processes is None):
		return

	if not processes['process']:
		return

	def default(obj,*args,**kwargs):
		try:
			return obj.first()
		except:
			return obj.iloc[0]

	def exp(obj,*args,**kwargs):
		return np.exp(obj)
	
	def log(obj,*args,**kwargs):
		return np.log(obj)		

	def sqrt(obj,*args,**kwargs):
		return np.sqrt(obj)	

	def mean(obj,*args,**kwargs):
		dtypes = {(np.dtype('O'),):mean_obj}
		default = mean_default
		otype = obj.dtype
		for dtype in dtypes:
			if otype in dtype:
				return dtypes[dtype](obj,*args,**kwargs)
		return default(obj,*args,**kwargs)
	def std(obj,*args,**kwargs):
		dtypes = {(np.dtype('O'),):std_obj}
		default = std_default
		otype = obj.dtype
		for dtype in dtypes:
			if otype in dtype:
				return dtypes[dtype](obj,*args,**kwargs)
		return default(obj,*args,**kwargs)
	def sem(obj,*args,**kwargs):
		dtypes = {(np.dtype('O'),):sem_obj}
		default = sem_default
		otype = obj.dtype
		for dtype in dtypes:
			if otype in dtype:
				return dtypes[dtype](obj,*args,**kwargs)
		return default(obj,*args,**kwargs)

	def none(obj,*args,**kwargs):
		obj[...] = nan
		# return obj			
		return nan	

	def mean_default(obj,*args,**kwargs):
		return obj.mean()
	def std_default(obj,*args,**kwargs):
		return obj.std(ddof=kwargs.get('ddof',obj.shape[0]>1))		
	def sem_default(obj,*args,**kwargs):
		return obj.sem(ddof=kwargs.get('ddof',obj.shape[0]>1))	

	def mean_obj(obj,*args,**kwargs):
		try:
			assert all(not(i,iterables) for i in obj) or (len(set(len(i) for i in obj))==1)
			obj = np.array(list(obj))
			obj = to_tuple(obj.mean(axis=0))
		except:
			obj = np.array([list(obj)[0]])
			obj = obj[0]
		return obj
	def std_obj(obj,*args,**kwargs):
		try:
			assert any(not(i,iterables) for i in obj) or (len(set(len(i) for i in obj))==1)
			obj = np.array(list(obj))
			obj = to_tuple(obj.std(axis=0,ddof=obj.shape[0]>1))
		except:
			obj = np.array([list(obj)[0]])
			obj = to_tuple(np.zeros(obj.shape[1:],dtype=obj.dtype))
		return obj
	def sem_obj(obj,*args,**kwargs):
		try:
			assert any(not(i,iterables) for i in obj) or (len(set(len(i) for i in obj))==1)
			obj = np.array(list(obj))
			obj = to_tuple(obj.std(axis=0,ddof=obj.shape[0]>1))
		except:
			obj = np.array([list(obj)[0]])
			obj = to_tuple(np.zeros(obj.shape[1:],dtype=obj.dtype))		
		return obj	

	def mean_arithmetic(obj,*args,**kwargs):
		return obj.mean()
	def std_arithmetic(obj,*args,**kwargs):
		return obj.std(ddof=kwargs.get('ddof',obj.shape[0]>1))		
	def sem_arithmetic(obj,*args,**kwargs):
		return obj.sem(ddof=kwargs.get('ddof',obj.shape[0]>1))		

	def mean_geometric(obj,*args,**kwargs):
		return exp(log(obj).mean())
	def std_geometric(obj,*args,**kwargs):
		return sqrt(mean_geometric(obj**2,*args,**kwargs) - mean_geometric(obj,*args,**kwargs)**2)
	def sem_geometric(obj,*args,**kwargs):
		return sqrt(mean_geometric(obj**2,*args,**kwargs) - mean_geometric(obj,*args,**kwargs)**2)/sqrt(obj.size)

	def mean_log(obj,*args,**kwargs):
		return exp(log(obj).mean())
	def std_log(obj,*args,**kwargs):
		return exp(log(obj).std(ddof=kwargs.get('ddof',obj.shape[0]>1)))
	def sem_log(obj,*args,**kwargs):
		return exp(log(obj).sem(ddof=kwargs.get('ddof',obj.shape[0]>1)))		

	def mean_bootstrap(obj,*args,**kwargs):
		return bootstrap(obj,*args,**kwargs).mean()
	def std_bootstrap(obj,*args,**kwargs):
		return bootstrap(obj,*args,**kwargs).std(ddof=kwargs.get('ddof',obj.shape[0]>1))		
	def sem_bootstrap(obj,*args,**kwargs):
		return bootstrap(obj,*args,**kwargs).sem(ddof=kwargs.get('ddof',obj.shape[0]>1))		

	# dtype = {attr: 'float128' for attr in data if is_float_dtype(data[attr].dtype)}
	# dtype = {attr: data[attr].dtype for attr in data if is_float_dtype(data[attr].dtype)}
	updates = {
		'nan':Dict(
			boolean=lambda attr,data: data[attr].isna().all(),
			func = lambda attr,data: 'none'
			)
		}
	for update in updates:
		for attr in data:
			if updates[update].boolean(attr,data):
				data[attr] = updates[update].func(attr,data)

	dtype = {attr: 'float' for attr in data if is_float_dtype(data[attr].dtype)}	
	data = data.astype(dtype)

	for name in keys:

		logger.log(info,"Processing : %r"%(name,))

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

		funcs = copy(funcs)
		stat = 'stat'
		stats = {axes: {'':'mean','err':'std'} for axes in dimensions}
		funcs = {} if not isinstance(funcs,dict) else funcs
		analyses = {} if not isinstance(analyses,dict) else analyses
		wrappers = {} if not isinstance(wrappers,dict) else wrappers
		functions = {
			'mean_log':mean_log,'std_log':std_log,'sem_log':'sem_log',
			'mean_arithmetic':mean_arithmetic,'std_arithmetic':std_arithmetic,'sem_arithmetic':sem_arithmetic,
			'mean_geometric':mean_geometric,'std_geometric':std_geometric,'sem_geometric':sem_geometric,
			'mean_bootstrap':mean_bootstrap,'std_bootstrap':std_bootstrap,'sem_bootstrap':sem_bootstrap,
			}

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



		wrappers = {attr: wrappers[attr] for attr in wrappers if attr not in ALL}

		for attr in list(wrappers):

			if wrappers[attr] is None:
				wrappers.pop(attr)
				continue
			elif isinstance(wrappers[attr],str):
				wrappers[attr] = {wrappers[attr]:{}}
			elif isinstance(wrappers[attr],dict) and all(isinstance(wrappers[attr][func],dict) for func in wrappers[attr]):
				wrappers[attr] = {func:wrappers[attr][func] for func in wrappers[attr]}
			else:
				continue

			for func in list(wrappers[attr]):
				try:
					wrappers[attr][func] = partial(load(func),**wrappers[attr][func])
				except:
					wrappers[attr].pop(func)
					continue

		tmp = {}

		for attr in wrappers:

			if attr in data:
				tmp[attr] = data[attr]
			else:
				tmp[attr] = None

			for func in wrappers[attr]:
				try:
					data[attr] = wrappers[attr][func](data)
				except:
					pass



		attributes = list(set((
			*[attr for attr in data],
			*[attr for attr in analyses if any(i in attr for i in data)],
			*[attr for attr in wrappers if wrappers.get(attr) is not None and attr in data]
			)))

		if any((keys[name][axes] not in attributes) and (keys[name][axes] is not null) for axes in AXES if axes in keys[name]):
			key,value = name,None
			setter(plots,{key:value},delimiter=delim,default=True)
			continue


		independent = [keys[name][axes] for axes in dimensions[:-1] if keys[name][axes] in attributes]
		dependent = [keys[name][axes] for axes in dimensions[-1:] if keys[name][axes] in attributes]
		labels = [attr for attr in label if (attr in attributes) and (((label[attr] is null) and (exclude is None) and (include is None)) or ((isinstance(label[attr],iterables)) and (exclude is None)) or ((exclude is not None) and (attr not in exclude))) or ((include is not None) and (attr in include))]

		boolean = [parse(attr,label[attr],data,verbose=verbose) for attr in label]
		boolean = conditions(boolean,op='and')
		boolean = slice(None) if ((boolean is True) or (boolean is False) or (boolean is None)) else boolean

		by = [*labels,*independent]

		if not by:
			key,value = name,None
			setter(plots,{key:value},delimiter=delim,default=True)
			continue

		options = dict(as_index=False,dropna=False)
		groups = data[boolean].groupby(by=by,**options)

		properties = {}
		variables = independent
		func = lambda group,variables: (group[:-len(variables)] if (variables) and isinstance(group,tuple) else group)
		for group in groups.groups:
			prop = func(group,variables)
			if prop in properties:
				continue
			properties[prop] = {grouping: groups.get_group(grouping) for grouping in groups.groups if func(grouping,variables)==prop}
			properties[prop] = {grouping: Dict({attr: getattr(properties[prop][grouping],attr) for attr in ['shape','size','ndim'] if hasattr(properties[prop][grouping],attr)}) for grouping in properties[prop]}

		
		if analyses:
			options = dict(as_index=False,dropna=False)
			groups = groups.apply(analyse,analyses=analyses,verbose=verbose).reset_index(drop=True)
			groups = groups.groupby(by=by,**options)


		if not all(attr in groups.get_group(group) for group in groups.groups for attr in attributes):
			key,value = name,None
			setter(plots,{key:value},delimiter=delim,default=True)
			continue


		shapes = {prop: tuple(((min(properties[prop][grouping].shape[i] for grouping in properties[prop]),
								max(properties[prop][grouping].shape[i] for grouping in properties[prop]))
					for i in range(groups.ndim)))
					for prop in properties}
		shapes = {prop: tuple((i[0] if len(set(i))==1 else i for i in shapes[prop])) for prop in shapes}

		dtypes = {attr: (
				('array' if any(isinstance(i,tuple) for i in data[attr]) else 'object' if data[attr].dtype.kind in ['O'] else 'dtype')
				if attr in data else 'dtype') 
				for attr in attributes}

		agg = {
			**{attr : [(attr, {'array':mean,'object':'first','dtype':'mean'}[dtypes[attr]] 
					  if attr not in by else {'array':'first','object':'first','dtype':'first'}[dtypes[attr]])] 
					  for attr in attributes},
			**{attr : [(delim.join(((attr,function,func))),
						{'array':{'':mean,'err':std}[func],
						 'object':{'':'first','err':none}[func],
						 'dtype':funcs[function][axes][func]
						}[dtypes[attr]]) 
						for function in funcs for func in funcs[function][axes]
						if delim.join(((attr,function,func))) not in data] 
						for axes,attr in zip([*dimensions[:-1],*dimensions[-1:]],[*independent,*dependent])						
						},
		}

		variables = [
			*independent,
			*dependent,
			*[kwarg[0] for attr in [*independent,*dependent] for kwarg in agg[attr]]
			]

		options = dict(level=0,axis=1)

		dtype = {attr: data[attr].dtype for attr in agg if attr in label or attr not in variables}
	
		groups = groups.agg(agg).droplevel(**options).astype(dtype)

		# agg = {
		# 	**{attr : {attr: pd.NamedAgg(
		# 				column=attr,
		# 				aggfunc={
		# 					'array':mean,'object':'first','dtype':'mean'
		# 					}[dtypes[attr]] 
		# 		  		if attr not in by else {'array':'first','object':'first','dtype':'first'}[dtypes[attr]])
		# 				}						
		# 			  for attr in data},
		# 	**{attr : {delim.join(((attr,function,func))): pd.NamedAgg(
		# 				column=attr,
		# 				aggfunc= {
		# 					'array':{'':mean,'err':std}[func],
		# 				 	'object':{'':'first','err':none}[func],
		# 				 	'dtype':funcs[function][axes][func]
		# 					}[dtypes[attr]]) 
		# 				for function in funcs for func in funcs[function][axes]} 
		# 				for axes,attr in zip([*dimensions[:-1],*dimensions[-1:]],[*independent,*dependent])
		# 				},
		# }		

		# by = [*labels]
	
		# variables = [
		# 	*independent,
		# 	*dependent,
		# 	*[agg[attr][kwarg].column for attr in [*independent,*dependent] for kwarg in agg[attr]]
		# 	]
		# agg = {kwarg: agg[attr][kwarg] for attr in agg for kwarg in agg[attr]}
		
		# droplevel = dict(level=0,axis=1)

		# dtype = {attr: data[attr].dtype for attr in agg if attr in label or attr not in variables}
	
		# groups = groups.agg(**agg).astype(dtype)


		by = [*labels]

		options = dict(as_index=False,dropna=False)

		if by:
			groups = groups.groupby(by=by,**options)
		else:
			groups = GroupBy(groups,by=by,**options)

		assert all(groups.get_group(group).columns.nlevels == 1 for group in groups.groups) # Possible future broken feature agg= (label,name)

		for i,group in enumerate(groups.groups):
			
			logger.log(info,"Group : %r %r %r -> %r"%(group,tuple((value for attr in label if attr not in by for value in (label[attr] if isinstance(label[attr],iterables) else [label[attr]]))),shapes.get(group) if group in shapes else shapes.get((group,)) if not isinstance(group,tuple) and (group,) in shapes  else '',groups.get_group(group).shape))
			
			for j,function in enumerate(funcs):

				grouping = groups.get_group(group)

				key = (*name[:-3],i,j,*name[-1:])
				value = copy(getter(plots,name,delimiter=delim))

				source = [attr for attr in attributes if attr not in variables]
				destination = other

				value[destination] = {
					**{attr: grouping[attr].to_list()[0] for attr in source if len(grouping[attr])},
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
								value[destination] = np.arange(1,len(grouping[source].iloc[0])+1)
							else:
								value[destination] = grouping.reset_index().index.to_numpy()

							if isinstance(value[destination],arrays):
								value[destination] = value[destination].tolist()

						else:
							value[destination] = None
						
				setter(plots,{key:value},delimiter=delim,default=True)

		for attr in tmp:
			if tmp[attr] is None and attr in data:
				data.drop(columns=attr)
			else:
				data[attr] = tmp[attr]

	return


def plotter(plots,processes,verbose=None):
	'''
	Plot data based plot plots, process processes
	Args:
		plots (dict): plots
		processes (dict): processes
		verbose (bool): Verbosity		
	'''

	if (plots is None) or (processes is None):
		return

	if not processes['plot']:
		return

	# Variables
	path = processes['path']['plot']
	pwd = processes['pwd']
	cwd = processes['cwd']
	fig = processes['fig']
	ax = processes['ax']
	configuration = processes['configuration']
	texify = processes['texify']
	valify = processes['valify']
	
	obj = 'ax'

	# Check data
	for instance in list(plots):
		for subinstance in list(plots[instance]):
			for prop in plots[instance][subinstance][obj]:

				if isinstance(plots[instance][subinstance][obj].get(prop),dict):
					plots[instance][subinstance][obj][prop] = [plots[instance][subinstance][obj][prop]]
				
				if prop in PLOTS:
					for data in search(plots[instance][subinstance][obj][prop]):

						if data is None:
							continue

						if OTHER in data and OTHER in data[OTHER]:
							wrappers = data[OTHER][OTHER].get('wrapper')

							if wrappers is None:
								wrappers = {}
							else:
								wrappers = {attr: wrappers[attr] for attr in wrappers}
							
							wrappers = {attr: wrappers[attr] for attr in wrappers if attr not in ALL}

							for attr in list(wrappers):

								if wrappers[attr] is None:
									wrappers.pop(attr)
									continue
								elif isinstance(wrappers[attr],str):
									wrappers[attr] = {wrappers[attr]:{}}
								elif isinstance(wrappers[attr],dict) and all(isinstance(wrappers[attr][func],dict) for func in wrappers[attr]):
									wrappers[attr] = {func:wrappers[attr][func] for func in wrappers[attr]}
								else:
									continue

								for func in list(wrappers[attr]):
									try:
										wrappers[attr][func] = partial(load(func),**wrappers[attr][func])
									except:
										wrappers[attr].pop(func)
										continue

							for attr in data[OTHER]:
								if wrappers.get(attr) is not None:
									value = {
										**{attr: data[OTHER][attr] for attr in data[OTHER]},
										**{data[OTHER][attr][OTHER]: data[attr] for attr in data if attr in VARIABLES},
										**{attr: data[attr] for attr in data if attr in ALL},
										}
									for func in wrappers[attr]:
										value = wrappers[attr][func](value)
									data[OTHER][attr] = value

						dimensions = [axes for axes in AXES if axes in data]
						independent = [axes for axes in ALL 
							for variable in VARIABLES 
							if axes in VARIABLES[variable] and variable in dimensions and dimensions.index(variable) < (len(dimensions)-1)]
						dim = len(dimensions)

						for attr in ALL:
							
							if attr not in data:
								continue

							if attr in independent:
								if attr.endswith('err'):
									continue
								if (data.get(attr) is None) or isinstance(data.get(attr),str):
									data.clear()

		if all((not data) for prop in PLOTS if plots[instance][subinstance][obj].get(prop) for data in search(plots[instance][subinstance][obj][prop])):
			plots[instance].pop(subinstance);

	for instance in list(plots):
		if not plots[instance]:
			plots.pop(instance);

	# Set grid layout based on GRID {instance:{subinstance:shape}} 
	# where shape = [*layout,*axis,*axes] 
	# for layout = [rows,cols of subfigures], axis = [*data dimensions axis], axes = [number of data points]
	
	grid = {}

	for instance in list(plots):

		logger.log(info*verbose,"Setting : %s"%(instance))

		for subinstance in list(plots[instance]):
			
			if not plots[instance][subinstance].get(obj):
				continue

			if grid.get(instance) is None:
				grid[instance] = {}
			
			if grid[instance].get(subinstance) is None:
				grid[instance][subinstance] = [*[1]*LAYOUTDIM,*[1]*AXISDIM,*[-1]*(GRIDDIM-LAYOUTDIM-AXISDIM)]

			for prop in PLOTS:
				
				if prop not in plots[instance][subinstance][obj]:
					continue
				
				for data in search(plots[instance][subinstance][obj][prop]):

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
								transpose = [transpose] if isinstance(transpose,integers) else transpose
								transpose = [data[axes].ndim + i if i < 0 else i for i in transpose]
								transpose = [*transpose,*[i for i in range(data[axes].ndim) if i not in transpose]]
								data[axes] = data[axes].transpose(transpose)

							if reshape:
								reshape = [reshape] if isinstance(reshape,(int,str)) else reshape
								reshape = [i if isinstance(i,integers) else int(data[OTHER].get(i)) if i is not None else data[axes].shape[reshape.index(i)] for i in reshape]
								data[axes] = data[axes].reshape(reshape)
							
							if shape:
								shape = {prop: [shape[prop]] if isinstance(shape.get(prop),integers) else shape.get(prop) if shape.get(prop) is not None else [] for prop in GRID}
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

							if axes.endswith('err'):
								data[axes][...] = 0
							else:
								data[axes][...,:] = np.arange(1,data[AXES[dim-1]].shape[-1]+1)

	# Set layout from data
	# Each plot in grid as a separate subinstance with key (subinstance,*position)
	# for position (row,col) in layout
	# where data in search(plots[instance][subinstance][obj][prop]
	# for each instance,subinstance,position is itself a set of data to plot of arbitrary shape [*labels,]

	for instance in list(plots):
		for subinstance in list(plots[instance]):
			
			if not plots[instance][subinstance].get(obj):
				continue

			plts = copy(plots[instance][subinstance])
			grd = [i for i in grid[instance][subinstance]][:LAYOUTDIM]

			for index,position in enumerate(itertools.product(*(range(i) for i in grid[instance][subinstance][:LAYOUTDIM]))):
				
				key = subinstance
				coordinate = [i for i in position]

				key = delim.join([str(key),*[str(i) for i in coordinate]])
				
				plots[instance][key] = copy(plts)
				grid[instance][key] = [*[i for i in grd][:LAYOUTDIM],index+1]

				boolean = lambda data: any(data[axes] is not None for axes in ALL if axes in data)

				for axis in itertools.product(*(range(i) for i in grid[instance][subinstance][LAYOUTDIM:LAYOUTDIM+AXISDIM])):

					for prop in PLOTS:
						
						if prop not in plots[instance][subinstance][obj]:
							continue
						
						for index,shape,data in search(copy(plots[instance][subinstance][obj][prop]),returns=True):
						
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
							item = data if boolean(data) else None
							iterable = plots[instance][key][obj][prop]
							inserter(index,item,iterable)
							
			plots[instance].pop(subinstance);
			grid[instance].pop(subinstance);


	# Set layout from configuration
	# Each plot in grid as a separate subinstance with key (subinstance,*position)

	# TODO: Cases of plots with position-dependent, not just row,col independent labels for layout

	information = {}
	metadata = {}

	for instance in list(plots):

		information[instance] = {}
		metadata[instance] = {}
	
		for subinstance in list(plots[instance]):
			
			if not plots[instance][subinstance].get(obj):
				continue

			information[instance][subinstance] = {}
			metadata[instance][subinstance] = {}

			for prop in PLOTS:
				
				if prop not in plots[instance][subinstance][obj]:
					continue

				information[instance][subinstance][prop] = {}

			for prop in information[instance][subinstance]:

				data = list(natsorted(set(label
					for subinstance in plots[instance] if obj in plots[instance][subinstance] and prop in plots[instance][subinstance][obj]
					for data in search(plots[instance][subinstance][obj][prop])
					if (data)
					for label in [*data[OTHER],*data[OTHER][OTHER][OTHER]]
					if ((data) and (label not in [*ALL,OTHER])) and (label not in ['legend','scinotation','labels'])
					)))
				data = {label: [
						(data[OTHER][label] if not isinstance(data[OTHER][label],tuple) else to_tuple(data[OTHER][label])) if (
								(label in data[OTHER]) and not isinstance(data[OTHER][label],list)) else 
							to_tuple(data[OTHER][label]) if (
								(label in data[OTHER])) else 
							data[OTHER][data[OTHER][OTHER][OTHER][label].replace('@','')] if (
								(label in data[OTHER][OTHER][OTHER] and 
								(data[OTHER][OTHER][OTHER].get(label) is not None) and
								data[OTHER][OTHER][OTHER][label].replace('@','') in data[OTHER])) else data[OTHER][OTHER][OTHER][label] if (label in data[OTHER][OTHER][OTHER]) else None
						for subinstance in plots[instance] if obj in plots[instance][subinstance]
						for prop in PLOTS
						if prop in plots[instance][subinstance][obj]
						for data in search(plots[instance][subinstance][obj][prop]) if (
							((data) and ((label in data[OTHER]) or (label in data[OTHER][OTHER][OTHER]))))
						]
						for label in list(data)}

				information[instance][subinstance][prop] = data

			
			data = [{attr: data[OTHER][attr] for attr in data[OTHER] if attr not in [*ALL,OTHER]}
				for prop in PLOTS if prop in plots[instance][subinstance][obj] for data in search(plots[instance][subinstance][obj][prop])]
			data = {attr:[i[attr] if isinstance(i[attr],scalars) else tuple(i[attr]) for i in data] for attr in set(attr for i in data for attr in i if all(attr in i for i in data))}
			data = {attr: sorted(set(data[attr]),key=lambda i:data[attr].index(i)) for attr in data}
			data = {attr: [i if isinstance(i,scalars) else [*i] for i in data[attr]] for attr in data}

			metadata[instance][subinstance] = data

	
	for instance in list(plots):

		for subinstance in list(plots[instance]):

			if not plots[instance][subinstance].get(obj):
				continue

			if not configuration.get(instance):
				continue

			if all(config[attr] is None for config in configuration[instance] for attr in config):
				continue

			data = metadata[instance][subinstance]

			config = {position: {attr:config['value'][attr] if isinstance(config['value'],dict) else None for config in configuration[instance] if config['position'] in [position,None] for attr in (config['value'] if isinstance(config['value'],(dict,*iterables)) else []) if attr in data}
				for position in ['row','col'][:LAYOUTDIM]}

			layout = {position: list(permuter({attr: sorted([i for i in data[attr] if parse(attr,config[position][attr],{attr:i})],key=lambda i: data[attr].index(i) if not isinstance(config[position][attr],iterables) else list(config[position][attr]).index(i))
				for attr in config[position]}))
				for position in config}
			layout = {position: [{**i,**{attr:data[attr][0] for attr in data if len(data[attr])==1}} for i in layout[position]] for position in layout}
			layout = [[{**i,**j} if all(parse(attr,config[position][attr],{**i,**j}) for position in config for attr in config[position]) else None for j in layout['col']] for i in layout['row']]

			shapes = [len(layout),max(len(i) for i in layout)]

			plts = copy(plots[instance].pop(subinstance))
			grd = [i for i in grid[instance].pop(subinstance)][:LAYOUTDIM]

			inf = information[instance].pop(subinstance)			
			meta = metadata[instance].pop(subinstance)

			options = {position:[config['kwargs'] for config in configuration[instance] if config['position'] in [position,None] and config['kwargs'] is not None]
				for position in ['row','col'][:LAYOUTDIM]}
			options = {position:[{}] if not len(options[position]) else [i for kwargs in options[position] for i in (kwargs if not isinstance(kwargs,dict) else [kwargs])] for position in options}
			options = [[{**options[['row','col'][0]][i%len(options[['row','col'][0]])],**options[['row','col'][1]][j%len(options[['row','col'][1]])]} for j in range(shapes[1])] for i in range(shapes[0])]

			for index,position in enumerate(itertools.product(*(range(i) for i in shapes[:LAYOUTDIM]))):

				if indexer(position,layout) is None:
					continue

				key,coordinate = delim.join(subinstance.split(delim)[:-LAYOUTDIM]),[int(i) for i in subinstance.split(delim)[-LAYOUTDIM:]]
				key,coordinate = key,[position[i]*grd[i]+coordinate[i] for i in range(LAYOUTDIM)]
				key,coordinate = delim.join([str(key),*[str(i) for i in coordinate]]),coordinate

				plots[instance][key] = copy(plts)
				grid[instance][key] = [*[i*j for i,j in zip(grd,shapes)][:LAYOUTDIM],index+1]

				information[instance][key] = inf				
				metadata[instance][key] = {**meta,**indexer(position,layout)}

				boolean = lambda data,item=indexer(position,layout): (item is not None) and all(data[OTHER][attr]==item[attr] for attr in item if attr in data[OTHER] if not isinstance(item[attr],list))

				for prop in PLOTS:
							
					if prop not in plts[obj]:
						continue
					
					for index,shape,data in search(copy(plts[obj][prop]),returns=True):
					
						if not data:
							continue

						index = index
						item = data if boolean(data) else None
						iterable = plots[instance][key][obj][prop]
						inserter(index,item,iterable)

				opts = indexer(position,options)
				vals = indexer(position,layout)
				for attr in list(opts):

					attr,tmp = attr.split(delim),opts.pop(attr)
						
					if attr[0] in [obj]:
						attr,index = (delim.join(attr[:2]),delim.join(attr[2:])) if len(attr)>2 else (delim.join(attr[:1]),delim.join(attr[1:]))
					else:
						attr,index = delim.join(attr[:-1]),delim.join(attr[-1:])

					if any(i in PLOTS for i in attr.split(delim)) and any(i in [OTHER] for i in index.split(delim)):
						index = delim.join([OTHER,*index.split(delim)])

					if isinstance(tmp,str):
						try:
							function = load(tmp,default=None)
						except:
							function = lambda data,plots,tmp=tmp:tmp
					else:
						function = lambda data,plots,tmp=tmp:tmp

					def func(data,plots,tmp=tmp,function=function):
						try:
							return function(data,plots)
						except:
							return tmp

					value = getter(plots[instance][key],attr,delimiter=delim)
					item = func({i: metadata[instance][key][i] for i in metadata[instance][key] if not isinstance(metadata[instance][key],list)},information[instance][key])

					if value is None:
						values = [plots[instance][key]]
						items = {delim.join([attr,index]):item}
					elif isinstance(value,list):
						values = search(value)
						items = {index:item} if index else item
					else:
						values = [value]
						items = {index:item} if index else item

					for data in values:

						if not data:
							continue

						setter(data,copy(items),delimiter=delim)


	for instance in list(plots):

		for subinstance in list(plots[instance]):

			data = {attr: information[instance][subinstance][attr] for attr in information[instance][subinstance]}
			information[instance][subinstance] = data

			data = {attr: metadata[instance][subinstance][attr][0] 
				if (isinstance(metadata[instance][subinstance][attr],list) and len(metadata[instance][subinstance][attr]) == 1)
				else metadata[instance][subinstance][attr] 
				for attr in metadata[instance][subinstance]
				}
			metadata[instance][subinstance] = data


	# Set layout
	
	layout = {}
	
	for instance in plots:

		for index,subinstance in enumerate(plots[instance]):		
			
			config = plots[instance][subinstance]['style']['layout']
			
			if not layout.get(instance):
				layout[instance] = config
			layout[instance].update({
				**layout[instance],
				**{attr: max(config[attr],grid[instance][subinstance][GRID.index(attr[1:-1])],layout[instance][attr])
					if (config[attr] is not None) and (layout[instance][attr] is not None) else None
					for attr in ['nrows','ncols']},
				**{attr: None for attr in ['index']},
				})

		for index,subinstance in enumerate(plots[instance]):
			
			config = copy(layout[instance])

			index = min(config['nrows']*config['ncols']-1,config['index']-1 if config['index'] is not None else grid[instance][subinstance][-1]-1 if grid[instance].get(subinstance) else index)
			nrow = (index - index%config['ncols'])//config['ncols']
			ncol = index%config['ncols']

			config.update({
				**{'index':index+1},
				**{
					'top':1 - (nrow)/config['nrows'] if config['top'] and config['nrows']>1 else None,
					'bottom':1 - (nrow+1)/config['nrows'] if config['bottom'] and config['nrows']>1 else None,
					'right':(ncol+1)/config['ncols'] if config['right'] and config['ncols']>1 else None,
					'left':(ncol)/config['ncols'] if config['left'] and config['ncols']>1 else None,											
					}
				})

			plots[instance][subinstance]['style']['layout'] = config

			grid[instance][subinstance] = [*[config['n%ss'%(GRID[i])] for i in range(LAYOUTDIM)],index+1]


	# Set kwargs

	for instance in list(plots):
	
		for subinstance in list(plots[instance]):
			
			if not plots[instance][subinstance].get(obj):
				continue

			position = [int(i) for i in subinstance.split(delim)[-LAYOUTDIM:]]

			values = {}

			for prop in information[instance][subinstance]:
				values[prop] = {}
				for label in information[instance][subinstance][prop]:
					value = {}
					value['value'] = information[instance][subinstance][prop][label]
					value['include'] = any((
							(((data[OTHER][OTHER]['legend'].get('include') is not False) and (data[OTHER][OTHER]['legend'].get('exclude') is not True))) and (
							(((not data[OTHER][OTHER]['legend'].get('include')) and (not data[OTHER][OTHER]['legend'].get('exclude')))) or
							(((not data[OTHER][OTHER]['legend'].get('include')) or (label in data[OTHER][OTHER]['legend'].get('include'))) and
							 ((not data[OTHER][OTHER]['legend'].get('exclude')) or (label not in data[OTHER][OTHER]['legend'].get('exclude')))
							))
							)
							for i in PLOTS
							if i in plots[instance][subinstance][obj]
							for data in search(plots[instance][subinstance][obj][i])
							if (data) 
							)
					value['sort'] = [k for (k,j) in sorted(set([(k,tuple(data[OTHER][OTHER]['legend'].get('sort') if (data[OTHER][OTHER]['legend'].get('sort') is not None) else
								data[OTHER][OTHER]['legend'].get('include')
								if ((data[OTHER][OTHER]['legend'].get('sort') is None) and 
								 (isinstance(data[OTHER][OTHER]['legend'].get('include'),iterables))) else 
								[]))							
							for i in PLOTS
							if i in plots[instance][subinstance][obj]
							for data in search(plots[instance][subinstance][obj][i])
							if (data) and (
								(data[OTHER][OTHER]['legend'].get('sort') is not None) or 
								((data[OTHER][OTHER]['legend'].get('sort') is None) and 
								 (isinstance(data[OTHER][OTHER]['legend'].get('include'),iterables))))
							for k in (
								data[OTHER][OTHER]['legend'].get('sort') 
								if (data[OTHER][OTHER]['legend'].get('sort') is not None) else
								data[OTHER][OTHER]['legend'].get('include')
								if ((data[OTHER][OTHER]['legend'].get('sort') is None) and 
								 (isinstance(data[OTHER][OTHER]['legend'].get('include'),iterables))) else 
								[]
								)
								]),key=lambda i: i[-1].index(i[0]))]
					value['axes'] = False				
					value['label'] = any((
							(label in data[OTHER][OTHER][OTHER]) and 
							(label in data[OTHER]))# and (data[OTHER][OTHER][OTHER][label] is None))
							for data in search(plots[instance][subinstance][obj][prop]) 
							if (data)
							)
					value['other'] = any((
							(label in data[OTHER][OTHER][OTHER]) and 
							(label not in data[OTHER]) and 
							((data[OTHER][OTHER][OTHER].get(label) is not None) and
							(data[OTHER][OTHER][OTHER][label].replace('@','') in data[OTHER])))
							for data in search(plots[instance][subinstance][obj][prop])
							if (data)
							)
					value['title'] = any((
							(label in data[OTHER][OTHER][OTHER]))
							for data in search(plots[instance][subinstance][obj][prop])
							if (data)
							)
					value['legend'] = any((
							(label in data[OTHER][OTHER][OTHER]) #and 
							# (label not in data[OTHER]) and 
							# (
								# (data[OTHER][OTHER][OTHER].get(label) is not None) and
							# (data[OTHER][OTHER][OTHER][label].replace('@','') not in data[OTHER]))
							)
							for data in search(plots[instance][subinstance][obj][prop])
							if (data)
							)
					value['labels'] = {attr: value
							for data in search(plots[instance][subinstance][obj][prop])
							if data 
							for attr,value in (data[OTHER][OTHER]['legend'].get('label') if isinstance(data[OTHER][OTHER].get('legend',{}).get('label'),dict) else {None:data[OTHER][OTHER].get('legend',{}).get('label')}
								).items()							
							if attr not in information[instance][subinstance][prop]
							}
					value['attr'] = {
							**{attr: {string:  data[OTHER][OTHER][attr][string]
								for data in search(plots[instance][subinstance][obj][prop]) 
								if ((data) and (data[OTHER][OTHER].get(attr) is not None))
								for string in data[OTHER][OTHER][attr]}
								for attr in ['texify','valify']},
							**{attr: {
								**{kwarg:[
								min((data[OTHER][OTHER][attr][kwarg][0]
									for data in search(plots[instance][subinstance][obj][prop]) 
									if ((data) and (data[OTHER][OTHER].get(attr) is not None) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0),
								max((data[OTHER][OTHER][attr][kwarg][1]
									for data in search(plots[instance][subinstance][obj][prop]) 
									if ((data) and (data[OTHER][OTHER].get(attr) is not None) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0),											
								] for kwarg in ['scilimits']},
								**{kwarg: 
									max((data[OTHER][OTHER][attr][kwarg]
									for data in search(plots[instance][subinstance][obj][prop]) 
									if ((data) and (data[OTHER][OTHER].get(attr) is not None) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0) 
									for kwarg in ['decimals']},
								**{kwarg: 
									any((data[OTHER][OTHER][attr][kwarg]
									for data in search(plots[instance][subinstance][obj][prop]) 
									if ((data) and (data[OTHER][OTHER].get(attr) is not None) and (kwarg in data[OTHER][OTHER][attr]))))
									for kwarg in ['one']},										
								}
								for attr in ['scinotation']},
							}

					values[prop][label] = value

				labels = list(natsorted(set(label
					for data in search(plots[instance][subinstance][obj][prop])
					if (data)
					for label in data
					if ((data) and (label in [*ALL]))
					)))
				tmp = {label:list(realsorted(set(i
							for data in search(plots[instance][subinstance][obj][prop]) if (data)
							for i in data.get(label,[])))) for label in labels}
				for label in labels:
					value = {}
					value['value'] = tmp[label]
					value['include'] = True
					value['sort'] = [k for (k,j) in sorted(set([(k,tuple(data[OTHER][OTHER]['legend'].get('sort')))							
							for i in PLOTS
							if i in plots[instance][subinstance][obj]
							for data in search(plots[instance][subinstance][obj][i])
							if (data) and (data[OTHER][OTHER]['legend'].get('sort') is not None)
							for k in data[OTHER][OTHER]['legend'].get('sort')]),key=lambda i: i[-1].index(i[0]))]
					value['axes'] = True				
					value['label'] = False
					value['other'] = False
					value['legend'] = False
					value['labels'] = {}
					value['attr'] = {
							**{attr: {string:  data[OTHER][OTHER][attr][string]
								for data in search(plots[instance][subinstance][obj][prop]) 
								if ((data) and data[OTHER][OTHER].get(attr) is not None)
								for string in data[OTHER][OTHER][attr]}
								for attr in ['texify','valify']},
							**{attr: {
								**{kwarg:[
								min((data[OTHER][OTHER][attr][kwarg][0]
									for data in search(plots[instance][subinstance][obj][prop]) 
									if ((data) and (data[OTHER][OTHER].get(attr) is not None) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0),
								max((data[OTHER][OTHER][attr][kwarg][1]
									for data in search(plots[instance][subinstance][obj][prop]) 
									if ((data) and (data[OTHER][OTHER].get(attr) is not None) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0),											
								] for kwarg in ['scilimits']},
								**{kwarg: 
									max((data[OTHER][OTHER][attr][kwarg]
									for data in search(plots[instance][subinstance][obj][prop]) 
									if ((data) and (data[OTHER][OTHER].get(attr) is not None) and (kwarg in data[OTHER][OTHER][attr]))),
									default=0) 
									for kwarg in ['decimals']},
								**{kwarg: 
									any((data[OTHER][OTHER][attr][kwarg]
									for data in search(plots[instance][subinstance][obj][prop]) 
									if ((data) and (data[OTHER][OTHER].get(attr) is not None) and (kwarg in data[OTHER][OTHER][attr]))))
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
			for prop in plots[instance][subinstance][obj]:

				if not plots[instance][subinstance][obj].get(prop):
					continue
				
				for index,shape,data in search(plots[instance][subinstance][obj][prop],returns=True):
					
					if not data:
						continue

					for attr in data:

						if (prop in PLOTS) and (attr in [*ALL,OTHER]):
							continue

						value = copy(data[attr])

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
									
									if isinstance(val,dict) and any(prop in val for prop in ['value','type','func']):
										defaults = {'value':None,'type':None,'func':None}
										val.update({prop: val.get(prop,defaults[prop]) for prop in defaults})

										if isinstance(val['func'],str):
											func = load(val['func'],default=None)
										else:
											func = val['func']

										if prop not in PLOTS:
											item = None
											items = [list(realsorted(set(values[prop][label]['value']))) for prop in values if label in values[prop]][0]
										elif callable(func):
											item = func({
												**{attr:data[attr] for attr in ALL if attr in data},
												**{attr:data[OTHER][attr] for attr in data[OTHER] for prop in values if label in values[prop] and attr in values[prop]}
												})
											items = func({
												**{attr:np.array([list(realsorted(set(values[prop][label]['value']))) for prop in values if label in values[prop]][0]) for attr in ALL if attr in data},
												**{attr:data[OTHER][attr] for attr in [label]},
												**{attr:np.array([list(realsorted(set(values[prop][label]['value']))) for prop in values if label in values[prop]][0]) for attr in data[OTHER] for prop in values if (label != attr) and (label in values[prop]) and (attr in values[prop])}
												})											
										elif prop in PLOTS:
											if label in data:
												item = [i for i in data[label]]
												items = [list(realsorted(set(values[prop][label]['value']))) for prop in values if label in values[prop]][0]
											else:
												item = data[OTHER].get(label)
												items = [list(realsorted(set(values[prop][label]['value']))) for prop in values if label in values[prop]][0]
										else:
											continue

									elif isinstance(val,dict) and not any(prop in val for prop in ['value','type','func']):

										val = {i:{label:val[i]} for i in val}

										if prop not in PLOTS:
											item = None
											items = [list(realsorted(set(values[prop][label]['value']))) for prop in values if label in values[prop]][0]
										elif prop in PLOTS:
											if label in data:
												item = [i for i in data[label]]
												items = [list(realsorted(set(values[prop][label]['value']))) for prop in values if label in values[prop]][0]
											else:
												item = data[OTHER].get(label)
												items = [list(realsorted(set(values[prop][label]['value']))) for prop in values if label in values[prop]][0]
										else:
											continue

									elif prop in PLOTS:
										if label in data:
											item = None
											items = [i for i in data[label]]
										else:
											item = data[OTHER].get(label)
											items = [list(realsorted(set(values[prop][label]['value']))) for prop in values if label in values[prop]][0]

									else:
										item = None
										items = [list(realsorted(set(values[prop][label]['value']))) for prop in values if label in values[prop]][0]

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
													for prop in PLOTS if plots[instance][subinstance][obj].get(prop)
													for item,shape,data in search(plots[instance][subinstance][obj][prop],returns=True) if data][0]
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
			for data in search(plots[instance][subinstance][obj].get(prop)):

				if not data:
					continue

				attr = 'value'
				delimiter = '__'
				if (data.get(attr) is None):
				
					data[attr] = []

					items = []
				
				elif isinstance(data[attr],dict) and all(prop.startswith(delimiter) and prop.endswith(delimiter) for prop in data[attr]):

					if all(len(i)>1 for i in data[attr]['__items__']):
						items = [data[attr]['__items__'].index(i)/max(1,data[attr]['__size__']-1) for i in data[attr]['__items__']]
					else:
						items = [i[0] for i in data[attr]['__items__']]

					value = data[attr]['__value__']
					indices = [data[attr]['__items__'].index(i)/max(1,data[attr]['__size__']-1) for i in data[attr]['__items__']]

					if isinstance(value,dict) and any(prop in value for prop in ['value','type','func']):
						defaults = {'value':None,'type':None,'func':None}
						value.update({prop: value.get(prop,defaults[prop]) for prop in defaults})

						if value['type'] in ['value']:
							value = items
						elif value['type'] in ['index']:
							value = indices
						else:
							value = indices

					elif isinstance(value,dict) and not any(prop in value for prop in ['value','type','func']):
						pass

					elif value is not None:
						value = indices
					else:
						value = None

					data[attr] = value

				else:
					items = data[attr]

				attr = 'set_%slabel'
				kwargs = ['%slabel']
				for axes in ['',*AXES]:
					if data.get(attr%(axes)) is None:
						continue
					for kwarg in kwargs:
					
						if data[attr%(axes)].get(kwarg%(axes)) is None:
							continue

						value = texify(data[attr%(axes)][kwarg%(axes)])

						data[attr%(axes)][kwarg%(axes)] = value

				attr = 'set_%sticks'
				kwargs = ['ticks']
				for axes in ['',*AXES]:
					if data.get(attr%(axes)) is None:
						continue
					for kwarg in kwargs:
					
						if data[attr%(axes)].get(kwarg) is None:
							continue

						norm = data.get('norm')
						scale = data.get('scale')
						base = data.get('base')
						length = len(data.get('value',[]))
						if norm is None:
							norm = {'vmin':min(data.get('value',[]),default=0),'vmax':max(data.get('value',[]),default=1)}
						elif not isinstance(norm,dict):
							norm = {'vmin':min(norm),'vmax':max(norm)}
						else:
							norm = {'vmin':norm.get('vmin',min(data.get('value',[]),default=0)),'vmax':norm.get('vmax',max(data.get('value',[]),default=1))}

						value = [min(min(data.get('value',[]),default=0),norm['vmin']),max(max(data.get('value',[]),default=1),norm['vmax'])]
						if isinstance(data[attr%(axes)].get(kwarg),integers):
							
							size = data[attr%(axes)][kwarg]

							if data[attr%(axes)][kwarg] == 1:
								value = np.array(value)							
								value = [(value[0]+value[1])/2]
							elif scale in ['linear']:
								value = np.array(value)
								value = np.linspace(*value,size,endpoint=True)
							elif scale in ['log','symlog']:
								base = 10 if base is None else base
								value = np.log10(value)/np.log(base)
								value = np.logspace(*value,size,base=base,endpoint=True)
							else:
								value = np.array(value)
								value = np.linspace(*value,size,endpoint=True)
						else:
							value = data[attr%(axes)][kwarg]

						if isinstance(value,arrays):
							value = value.tolist()

						if value is not None:
							data[attr%(axes)][kwarg] = value
						else:
							data[attr%(axes)][kwarg] = value

				attr = 'set_%sticklabels'
				kwargs = ['labels','ticklabels']
				for axes in ['',*AXES]:
					if data.get(attr%(axes)) is None:
						continue

					for kwarg in kwargs:

						if data[attr%(axes)].get(kwarg) is None:
							continue

						scale = data.get('scale')
						base = data.get('base')
						value = items
						
						if isinstance(data[attr%(axes)].get(kwarg),integers):

							length = len(value)
							size = min(len(data.get('set_%sticks'%(axes),{}).get('ticks',[])),data[attr%(axes)][kwarg])
							if size == 1:
								value = [(value[0]+value[-1])/2]
							elif ((length+1)%size) == 0:
								value = [items[0],*items[slice(1,length-1,max(1,length-2)//max(1,(size-3)))],items[-1]]
							else:
								size = min(len(data.get('set_%sticks'%(axes),{}).get('ticks',[])),length)
								if scale in ['log','symlog']:
									base = 10 if base is None else base
									value = np.logspace(min(value),max(value),size,base=base,endpoint=True)
								elif scale in ['linear']:
									value = np.linspace(min(value),max(value),size,endpoint=True)
								else:
									value = None

								if value is None:
									pass
								elif any(isinstance(i,integers) for i in items):
									value = [int(i) for i in value]
								else:
									value = [i for i in value]
						else:
							value = data[attr%(axes)][kwarg]

						options = {attr: data.pop(attr,dict()) 
							for attr,default in {
								'texify':dict(),
								'scinotation':dict(decimals=1,scilimits=[-1,4])}.items()
							}

						if value is not None:
							data[attr%(axes)][kwarg] = [texify(scinotation(i,**options['scinotation']),**options['texify']) for i in value]
						else:
							data[attr%(axes)][kwarg] = value

			# set legend
			prop = 'legend'
			attr = 'set_title'
			for data in search(plots[instance][subinstance][obj].get(prop)):
				
				if not data:
					continue

				separator = ',~'
				texifies = data.get('texify')

				value = [
					{
						**{(prop,label):'%s'%(texify(label,texify={**(values[prop][label]['attr']['texify'] if isinstance(values[prop][label]['attr']['texify'],dict) else {}),**(texifies if isinstance(texifies,dict) else {})}))
							for prop,label in natsorted(set((
							(prop,label)
							for prop in values 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (values[prop][label]['include']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['legend']) and (len(set(values[prop][label]['value']))>=1))))))},
						**{(prop,label):'%s'%(texify(label,texify={**(values[prop][label]['attr']['texify'] if isinstance(values[prop][label]['attr']['texify'],dict) else {}),**(texifies if isinstance(texifies,dict) else {})}))
							for prop,label in natsorted(set((
							(prop,label)
							for prop in values 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (values[prop][label]['include']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['other']) and (len(set(values[prop][label]['value']))>=1))))))},
						**{(prop,label):'%s'%(texify(label,texify={**(values[prop][label]['attr']['texify'] if isinstance(values[prop][label]['attr']['texify'],dict) else {}),**(texifies if isinstance(texifies,dict) else {})})) 
							for prop,label in natsorted(set((
							(prop,label)
							for prop in values 					
							for label in values[prop] 
							if ((not values[prop][label]['axes']) and (((values[prop][label]['include']) and (values[prop][label]['label']) and (len(set(values[prop][label]['value']))>=1)) and 
								not (values[prop][label]['other']))))))},
					},					
					{
						**{(prop,label):'%s%s%s'%(
							texify(label),' : ' if label else '',
							separator.join([texify(scinotation(value,**values[prop][label]['attr']['scinotation']),texify={**(values[prop][label]['attr']['texify'] if isinstance(values[prop][label]['attr']['texify'],dict) else {}),**(texifies if isinstance(texifies,dict) else {})}) 
									for value in list(realsorted(set(values[prop][label]['value'])))]))
							for prop in values 
							for label in natsorted(set((
							label 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (values[prop][label]['include']) and (
								(values[prop][label]['legend']) and (len(set(values[prop][label]['value']))<1))))))},
						**{(prop,label):'%s%s%s'%(
							texify(label),' : ' if label else '',
							separator.join([texify(scinotation(value,**values[prop][label]['attr']['scinotation']),texify={**(values[prop][label]['attr']['texify'] if isinstance(values[prop][label]['attr']['texify'],dict) else {}),**(texifies if isinstance(texifies,dict) else {})}) 
									for value in list(realsorted(set(values[prop][label]['value'])))]))
							for prop in values 
							for label in natsorted(set((
							label 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (values[prop][label]['include']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['other']) and (len(set(values[prop][label]['value']))<1))))))},
					},
					{
						**{(prop,attr):'%s'%(texify(attr if '%s' not in attr else attr%(''),texify={**(values[prop][label]['attr']['texify'] if isinstance(values[prop][label]['attr']['texify'],dict) else {}),**(texifies if isinstance(texifies,dict) else {})}))
							for prop,attr in natsorted(set((
							(prop,attr)
							for prop in values 
							for label in values[prop]
							for attr in values[prop][label]['labels']
							if attr is not None
							)))},
					},					
					]

				def func(value,key=None):
					index = [i for prop in values for label in values[prop] for i in (values[prop][label]['sort'] if values[prop][label]['sort'] else [])]
					if index:
						index = sorted(set(value),key=lambda i,index=index: index.index(i[-1]) if i[-1] in index else len(index))
					else:
						index = natsorted(set(value))
					return index

				if data.get('multiline',None):
					separator = '\n'
				else:
					separator = '~,~'
				delimiter = ',~'


				indexes = [i for prop in values for label in values[prop] for i in (values[prop][label]['sort'] if values[prop][label]['sort'] else [])]
				index = [[k for k in func(i)] for i in value if i]
				index = sorted(range(len(index)),key=lambda i,indexes=indexes: sum(indexes.index(j[-1]) if j[-1] in indexes else len(indexes) for j in index[i]))
				value = [[i[k] for k in func(i)] for i in value if i]
				value = [value[i] for i in index]
				value = separator.join([delimiter.join(i) for i in value]).replace('$','')
				if isinstance(data.get(attr),str) and data[attr].count('%s'):
					value = data[attr]%(value)
				elif isinstance(data.get(attr),str):
					value = data[attr]
				elif value:
					value = value
				else:
					value = None

				data[attr] = value

			prop = 'legend'
			attr = 'sort'
			for data in search(plots[instance][subinstance][obj].get(prop)):

				if not data:
					continue

				if not data.get(attr):
					continue

				value = list(set(label for value in (data[attr] if not isinstance(data[attr],dict) else [data[attr]]) for label in value))
				value = dataframe({label: values[prop][label]['value'] for prop in values for label in values[prop] if (((label in value) or ('%s_'%(label) in value)) and (label not in [*ALL,OTHER]) and (len(set(values[prop][label]['value']))>=1))})

				def func(value,attrs=None):

					value = sort(value,attrs)

					return value

				value = func(value,data[attr])

				data[attr] = value

			prop = 'legend'
			attr = 'set_color'
			for data in search(plots[instance][subinstance][obj].get(prop)):

				if not data:
					continue

				if not data.get(attr):
					continue

				if isinstance(data[attr],str):
					continue

				value = list(set(label for value in ([i['value'] for i in data[attr]] if not isinstance(data[attr],dict) else [data[attr]]) for label in value))
				value = dataframe({label: values[prop][label]['value'] for prop in values for label in values[prop] if (((label in value) or ('%s_'%(label) in value)) and (label not in [*ALL,OTHER]) and (len(set(values[prop][label]['value']))>=1))})

				def func(value,attrs=None):

					index = {}
					for attr in attrs:
						indices = sort(value,attr['value'])

						for i in indices:
							index[i] = attr['color']

					return index

				value = func(value,data[attr])

				data[attr] = value				


			# set kwargs data
			for prop in PLOTS:

				if not plots[instance][subinstance][obj].get(prop):
					continue

				for index,shape,data in search(plots[instance][subinstance][obj][prop],returns=True):

					if not data:
						continue

					slices = []
					subslices = copy([data[OTHER][OTHER].get('slice'),data[OTHER][OTHER].get('labels')])

					for subslice in subslices:
						if subslice is None:
							subslice = [slice(None)]
						elif isinstance(subslice,dict):
							for axes in list(subslice):
								if subslice[axes] is None or ((axes not in data) and (not any(data[OTHER][subaxis]['label']==axes for subaxis in ALL if subaxis in data[OTHER]))):
									subslice.pop(axes)

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
					slices = [subslice for subslice in slices if subslice is not None and subslice is not True and subslice is not False]

					wrappers = data[OTHER][OTHER].get('wrapper')

					if wrappers is None:
						wrappers = {}
					else:
						wrappers = {attr: wrappers[attr] for attr in wrappers}

					wrappers = {attr: wrappers[attr] for attr in wrappers if attr in ALL}

					for attr in list(wrappers):

						if wrappers[attr] is None:
							wrappers.pop(attr)
							continue
						elif isinstance(wrappers[attr],str):
							wrappers[attr] = {wrappers[attr]:{}}
						elif isinstance(wrappers[attr],dict) and all(isinstance(wrappers[attr][func],dict) for func in wrappers[attr]):
							wrappers[attr] = {func:wrappers[attr][func] for func in wrappers[attr]}
						else:
							continue

						for func in list(wrappers[attr]):
							try:
								wrappers[attr][func] = partial(load(func),**wrappers[attr][func])
							except:
								wrappers[attr].pop(func)
								continue


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


					for attr in list(data):

						if data.get(attr) is None:
							continue

						value = data[attr]

						if wrappers.get(attr) is not None:
							value = {
								**{data[OTHER][attr][OTHER]: data[attr] for attr in data if attr in VARIABLES},
								**{attr: data[OTHER][attr] for attr in data[OTHER]},
								**{attr: data[attr] for attr in data if attr in ALL},
								}
							for func in wrappers[attr]:
								value = wrappers[attr][func](value)

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

							if normalize.get(attr):
								value = normalize[attr](attr,data)

							l = len(value)

							for subslice in slices:
								value = value[subslice]

							value = np.array([valify(i,valify=data[OTHER][OTHER].get('valify')) for i in value])

						else:
							
							delimiter = '__'
							if isinstance(data[attr],dict) and all(prop.startswith(delimiter) and prop.endswith(delimiter) for prop in data[attr]):

								if not len(data[attr]):
									data.pop(attr);
									continue

								if all(len(i)>1 for i in data[attr]['__items__']):
									items = [data[attr]['__items__'].index(i)/max(1,data[attr]['__size__']-1) for i in data[attr]['__items__']]
								else:
									items = [i[0] for i in data[attr]['__items__']]
								
								value = data[attr]['__value__']
								indices = [data[attr]['__items__'].index(i)/max(1,data[attr]['__size__']-1) for i in data[attr]['__items__']]

								if isinstance(data[attr]['__value__'],str):
									value = data[attr]['__value__']
								
								elif isinstance(value,dict) and any(prop in value for prop in ['value','type']):
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
								elif isinstance(value,dict) and not any(prop in value for prop in ['value','type']):
									value = data[attr]['__value__']
								elif not isinstance(data[attr]['__value__'],scalars):
									value = data[attr]['__value__'][data[attr]['__index__']%len(data[attr]['__value__'])] if not isinstance(data[attr]['__value__'],str) else data[attr]['__value__']
								else:
									value = data[attr]['__value__']


								if attr in ['alpha','zorder','marker','linestyle']:

									if isinstance(value,dict):
										value = [i for i in value if all(
											((not isinstance(value[i][attr],iterables) and 
												data[OTHER][attr] == value[i][attr]) or 
											(isinstance(value[i][attr],iterables) and 
												data[OTHER][attr] in value[i][attr]))
											for attr in value[i] if attr in data[OTHER])]
										value = value[0] if value else None
									elif isinstance(value,iterables):
										if data[attr]['__index__'] is not None:
											value = value[data[attr]['__index__']%len(value)] 

								if attr in ['color','ecolor']:
									pass
								elif attr in ['alpha']:
									if not isinstance(value,scalars):
										value = (data[attr]['__index__'] + 0.5)/(data[attr]['__size__'])
								elif attr in ['zorder']:
									value = 1000*data[attr]['__index__']
								elif attr in ['marker']:
									pass
								elif attr in ['linestyle']:
									pass
								else:
									pass

						data[attr] = value

			# set title and axes label
			prop = 'set_%slabel'
			attr = '%slabel'
			for axes in ['',*AXES]:
				prop = 'set_%slabel' if axes in AXES else 'set_%stitle'
				for data in search(plots[instance][subinstance][obj].get(prop%(axes))):

					if not data:
						continue

					if data.get(attr%(axes)) is None:
						if axes in AXES:
							data[attr%(axes)] = [data[OTHER][axes]['label'] for prop in PLOTS if plots[instance][subinstance][obj].get(prop) for data in search(plots[instance][subinstance][obj][prop]) if OTHER in data]

							data[attr%(axes)] = data[attr%(axes)][0] if data[attr%(axes)] else None
					
					if isinstance(data[attr%(axes)],list):
						if not all(isinstance(i,list) for i in data[attr%(axes)]):
							data[attr%(axes)] = [[i] for i in data[attr%(axes)]]
						data[attr%(axes)] = data[attr%(axes)][position[0]%len(data[attr%(axes)])][position[1]%len(data[attr%(axes)][position[0]%len(data[attr%(axes)])])]

					if isinstance(data[attr%(axes)],str):
						if axes in AXES:
							data[attr%(axes)] = data[attr%(axes)]%(tuple(str(position[i]) if grid[instance][subinstance][i]>1 else '' for i in range(data[attr%(axes)].count('%s'))))
						else:
							objs = {i: data[OTHER][i] for prop in PLOTS if plots[instance][subinstance][obj].get(prop) for data in search(plots[instance][subinstance][obj][prop]) if data and OTHER in data for i in data[OTHER]}

							if metadata[instance].get(subinstance) is not None:
								for i in metadata[instance][subinstance]:
									if isinstance(metadata[instance][subinstance][i],list):
										objs.pop(i,None);
									elif i not in objs:
										objs[i] = metadata[instance][subinstance][i]

							if data[attr%(axes)] in objs:
								data[attr%(axes)] = "%s = %s"%(data[attr%(axes)],objs[data[attr%(axes)]])

					data[attr%(axes)] = texify(data[attr%(axes)])

			# Set tick labels
			attr = 'set_%sticklabels'
			kwargs = ['labels','ticklabels']
			for axes in ['',*AXES]:
				
				for data in search(plots[instance][subinstance][obj].get(attr%(axes))):
				
					if data is None:
						continue

					for kwarg in kwargs:

						if data.get(kwarg) is None:
							continue

						value = data[kwarg]

						options = {attr: data.pop(attr,default) 
							for attr,default in {
								'texify':dict(),
								'scinotation':dict(decimals=2,scilimits=[-1,4])}.items()
							}

						if value is not None:
							data[kwarg] = [texify(scinotation(i,**options['scinotation']),**options['texify']) for i in value]
						else:
							data[kwarg] = value

			# set label
			attr = 'label'
			for prop in PLOTS:

				if not plots[instance][subinstance][obj].get(prop):
					continue

				for data in search(plots[instance][subinstance][obj][prop]):

					if not data or not data.get(OTHER) or not data[OTHER].get(OTHER):
						continue


					def func(label,labels):

						if label in data[OTHER][OTHER][OTHER]:
							if values[prop][label]['label']:
								string = data[OTHER][label]
							else:
								string = data[OTHER].get(data[OTHER][OTHER][OTHER][label].replace('@',''),
									data[OTHER][OTHER][OTHER][label].replace('@',''))
						else:
							string = None

						if isinstance(data[OTHER][OTHER]['legend'].get('label'),dict):
							strings = data[OTHER][OTHER]['legend'].get('label').get(label,label)
						elif isinstance(data[OTHER][OTHER]['legend'].get('label'),str):
							strings = string
						else:
							strings = string

						if not isinstance(strings,dict):
							strings = {string: strings}

						func = strings.get(string,strings.get(str(string),string))
						default = (lambda string,label,func=func: (
							(func%(string) if func.count('%s') else func) if isinstance(func,str) else func)
						)

						func = load(func,default=default)
						string = func(string,label)

						string = texify(
							scinotation(string,**data[OTHER][OTHER].get('scinotation',{})),
							texify=data[OTHER][OTHER].get('texify')) if string is not None else labels[label]

						return string

					def sorter(value):
						if data[OTHER][OTHER]['legend'].get('sort') is not None:
							index = [*[i for i in data[OTHER][OTHER]['legend'].get('sort')],*[i for i in natsorted(set(value)) if i not in data[OTHER][OTHER]['legend'].get('sort')]]
							index = sorted(set(value),key=lambda i,index=index: index.index(i))
						else:
							index = natsorted(set(value))

						return index						

					value = {
						**{label: label
							for label in natsorted(set((
							label 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['legend']) and (len(set(values[prop][label]['value']))>1))))))},
						**{label: label
							for label in natsorted(set((
							label 
							for label in values[prop]
							if ((not values[prop][label]['axes']) and (not ((values[prop][label]['label'])) and 
								(values[prop][label]['other']) and (len(set(values[prop][label]['value']))>=1))))))},
						**{label: label 
							for label in natsorted(set((
							label 
							for label in values[prop] 
							if ((not values[prop][label]['axes']) and ((((values[prop][label]['label']) and 
								(len(set(values[prop][label]['value']))>=1)) and 
								not (values[prop][label]['other'])))))))},							
						**({label: label
							for label in data[OTHER][OTHER]['legend'].get('label')} 
							if isinstance(data[OTHER][OTHER]['legend'].get('label'),dict) else 
							{None:data[OTHER][OTHER]['legend'].get('label')} 
							if data[OTHER][OTHER]['legend'].get('label') is not None else {}
							)
						}

					value = {label:value[label] for label in sorter(value)}
					value = {label:func(label,value) for label in value}

					separator = ',~'

					tmp = dataframe({label: [data[OTHER][label] if values[prop][label]['label'] else data[OTHER].get(data[OTHER][OTHER][OTHER][label].replace('@',''),
						data[OTHER][OTHER][OTHER][label].replace('@',''))] 
						for label in value if label is not None})

					value = [(value[label],) if label is None else value[label] for label in value
							if (
							(label is None and value[label].count('%s')==0) or ( (label is not None) and
							(((data[OTHER][OTHER]['legend'].get('include') is not False) and (data[OTHER][OTHER]['legend'].get('exclude') is not True))) and (
							(((not data[OTHER][OTHER]['legend'].get('include')) and (not data[OTHER][OTHER]['legend'].get('exclude')))) or
							((((not isinstance(data[OTHER][OTHER]['legend'].get('include'),dict)) and 
								(not isinstance(data[OTHER][OTHER]['legend'].get('parse'),dict))) or (
								all(parse(label,
									{**(data[OTHER][OTHER]['legend'].get('include') if isinstance(data[OTHER][OTHER]['legend'].get('include'),dict) else {}),
									 **(data[OTHER][OTHER]['legend'].get('parse') if isinstance(data[OTHER][OTHER]['legend'].get('parse'),dict) else {}),},
									 tmp)))
							  ) and
							 ((not data[OTHER][OTHER]['legend'].get('include')) or 
							  (label in data[OTHER][OTHER]['legend'].get('include'))) and
							 ((not data[OTHER][OTHER]['legend'].get('exclude')) or 
							  (label not in data[OTHER][OTHER]['legend'].get('exclude')))
							))))
							]

					if any(isinstance(i,tuple) for i in value) and any(isinstance(i,str) for i in value):
						value = ': '.join([
							separator.join([j for i in value if isinstance(i,tuple) for j in i]),
							separator.join([j for i in value if isinstance(i,str) for j in [i]])
							])
					elif any(isinstance(i,tuple) for i in value):
						value = (
							separator.join([j for i in value if isinstance(i,tuple) for j in i])
							)
					elif any(isinstance(i,str) for i in value):
						value = (
							separator.join([j for i in value if isinstance(i,str) for j in [i]])
							)
					else:
						value = None

					value = value if value else None

					data[attr] = value	

			# mplstyle
			attr = 'mplstyle'
			data = plots[instance][subinstance]['style']
			if data.get(attr) is not None:
				value = join(pwd,split(data.get(attr),file=True),ext=split(data.get(attr),ext=True))
				data[attr] = value
			
			# savefig
			prop = 'savefig'
			attr = 'fname'
			for data in search(plots[instance][subinstance]['fig'].get(prop)):

				value = [split(path,file=True),instance,data[attr]]
				value = [i for i in value if i is not None]
				value = [split(path,directory=True),'plot',delim.join(value)]
				value = join(*value,ext=split(path,ext=True))
				
				data[attr] = value


	# Plot data
	for instance in plots:

		logger.log(info,"Plotting : %s"%(instance))

		fig[instance],ax[instance] = plot(fig=fig[instance],ax=ax[instance],settings=plots[instance])

	return


def postprocesser(processes,pwd=None,cwd=None,verbose=None):
	'''
	Postprocess data
	Args:
		processes (str,dict): Path to or dictionary of process plots
		pwd (str): Root path of plots, processes		
		cwd (str): Root path of data
		verbose (bool): Verbosity		
	'''
	if (processes is None):
		return

	if not processes['postprocess']:
		return

	path = cwd
	postprocess(path)

	return


def process(data,plots,processes,pwd=None,cwd=None,verbose=True):
	'''
	Process data
	Args:
		data (str,dict,iterable[str,dict]): Paths to or dictionary of data to process
		plots (str,dict): Path to or dictionary of plot plots
		processes (str,dict): Path to or dictionary of process plots
		pwd (str): Root path of plots, processes
		cwd (str): Root path of data
		verbose (bool): Verbosity
	Steps:
	- Load data and plots
	- Get data axes and labels based on branches of plots
	- Iterate over all distinct data branches of plots
	
	- Filter with booleans of all labels
	- Group by non-null labels and independent
	- Aggregate functions of dependent (mean,std) for each group
	- Assign new labels for functions with label.function 
	- Regroup with non-null labels
	
	- Adjust plots based on data
	
	- Plot data

	To process data, we find in plot plots dictionary the keys of ALL,OTHER ('x','y','label') properties for sorting.

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
	
	- Get statistics (mean,std) across samples datasets that share all attributes

	- Merge datasets across permutations of sort attributes for a given combination of OTHER attributes and values
	
	- After statistics and merging, variables data for each combination for OTHER attributes and values
	has shape of (1 + ndim) dimensions, with axes: 
	(# of permutations of sort for each combination, ndim of datasets)	

	- For parameters, we plot each combination for OTHER attributes and values variables data on the same plot, these being the labels.
	  Subplots are plotted from iterating over over the 1...ndim-2 axis of the variables data, and parameters the 0 and ndim-1 axis for each OTHER set
	  If the 'x' property is None, also iterate over the 0 (# of permutations of sort) axis variables data, and plot the ndim-1 axis for each OTHER 
	'''

	# Set plots and processes
	data,plots,processes = setup(data,plots,processes,pwd,cwd,verbose=verbose)

	# Load data
	loader(data,plots,processes,verbose=verbose)

	# Plot data
	plotter(plots,processes,verbose=verbose)

	# Post process data
	postprocesser(processes,pwd,cwd,verbose=verbose)

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
		'--plots':{
			'help':'Process plot plots',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--processes':{
			'help':'Process process plots',
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
