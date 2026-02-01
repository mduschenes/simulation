#!/usr/bin/env python

'''
Miscellaneous utility functions (for processing, plotting)
'''

# Import python modules
import os,sys,itertools,warnings,traceback
import numpy as np
import scipy as sp
import scipy.stats
import scipy.special
import scipy.optimize
import scipy.integrate
import pandas as pd
from pandas.api.types import is_float_dtype
from natsort import natsorted,realsorted
from math import prod

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,dataframe,zeros,rand,random,randint,linspace,logspace,seeded,finfo,texify,scinotation,histogram,entropy,information
from src.utils import addition,multiply,divide,power,matmul,sqrt,floor,exp,log,log10,absolute,maximum,minimum,sort,mean,std,cumsum,difference
from src.utils import to_tuple,is_nan,is_naninf,asscalar
from src.utils import grouper,conditions,flatten,concatenate,inplace,partial,epsilon,bootstrapper,interval
from src.utils import orng as rng
from src.utils import arrays,scalars,dataframes,iterables,numbers,integers,floats,nonzero,delim,nan,pi

from src.quantum import measurement

from src.iterables import permuter,setter,getter,search,Dictionary

from src.fit import fit

from src.plot import ALL,AXES,OTHER,DELIMITER

from src.io import load,dump

# Processing

def func_stat(data,func="min",stat='mean',**kwargs):
	default = lambda data: (lambda *args,data=data,**kwargs: data)
	stat = {'sem':lambda data,**kwargs:data.std()/np.sqrt(data.size)}.get(stat,stat) if isinstance(stat,str) else stat
	out = getattr(data,func,default(data))(**kwargs) if isinstance(func,str) else func(data,**kwargs)
	return getattr(out,stat,default(out))(**kwargs) if isinstance(stat,str) else stat(out,**kwargs)

def func_attr_stat(data,attr="objective",func="min",stat='mean',**kwargs):
	default = lambda data: (lambda *args,data=data,**kwargs: data)
	stat = {'sem':lambda data,**kwargs:data.std()/np.sqrt(data.size)}.get(stat,stat) if isinstance(stat,str) else stat
	attr = slice(None) if attr is None else attr
	out = getattr(data,func,default(data))(**kwargs) if isinstance(func,str) else func(data,**kwargs)
	return getattr(out,stat,default(out))(**kwargs) if isinstance(stat,str) else stat(out,**kwargs)

def func_stat_bootstrap(data,func=None,x=None,y=None,xerr=None,yerr=None,**kwargs):

	if y not in data:
		return data

	if func is None:
		func = 'argmax'
	if hasattr(np,func):
		func = lambda a,func=func: a[...,getattr(np,func)(a,axis=-1)]
	elif isinstance(func,str):
		func = load(func,default=lambda a:np.mean(a,axis=-1))

	def func(data,func=func):
		options = {
			**dict(
				a = data[y].to_numpy(),
				scale = data[yerr].to_numpy(),
				),
			**kwargs
			}
		a = bootstrapper(**options)

		a = func(a)

		data = {**{attr:[data[attr].iloc[0]] for attr in data},**{y:[a.mean().item()],yerr:[(a.std(ddof=a.size>1)/np.sqrt(a.size)).item()]}}

		data = dataframe(data)

		return data

	def agg(data):
		by = x
		agg = {**{attr:[(attr,'first')] for attr in data},**{y:[(y,'mean'),(yerr,'sem')]}}
		options = dict(by=by,agg=agg)
		data = grouper(data,**options)
		data = func(data)
		return data

	data = agg(data)

	return data

def func_func_fit(data,function=None,x=None,y=None,xerr=None,yerr=None,settings=None,**kwargs):

	keys = dict(zip(['x','y','xerr','yerr'],[f'{x}',f'{y}',f'{x}.error',f'{y}.error']))

	def agg(data,function=function):

		def function(data,*args,function=function,**kwargs):
			return measurement(data,*args,function=function,**kwargs)

		def apply(data):

			apply = {}

			def application(data,attr=None):
				data = data[attr].iloc[0]
				return data
			for attr in data:
				apply[attr] = partial(application,attr=attr)

			attr = keys['x']
			def application(data):
				data = 1/data[keys['x']].iloc[0]
				return data
			apply[attr] = application

			attr = keys['y']
			def application(data):
				info,size = function(data),data.size
				data = data[keys['y']].mean()
				data = abs(data)
				# data = data/np.log(size*info.size)
				return data
			apply[attr] = application

			attr = keys['xerr']
			def application(data):
				data = None
				return data
			apply[attr] = application

			attr = keys['yerr']
			def application(data):
				info,size = function(data),data[attr].size
				data = data[keys['yerr']].mean() - data[keys['y']].mean()**2
				data = abs(data)
				data = np.sqrt(data/(size*info.size))
				# data = data/np.log(size*info.size)
				return data
			apply[attr] = application

			apply = {attr:apply[attr] for attr in apply if attr in data}

			value = {}
			for attr in apply:
				if isinstance(apply[attr],str):
					value[attr] = getattr(data,apply[attr])
				else:
					value[attr] = apply[attr](data)
				value[attr] = [value[attr]]

			data = dataframe(value)

			return data

		attr = keys['y']
		info,size = function(data),data[attr].size

		by = keys['x']
		options = dict(by=by,apply=apply)
		data = grouper(data,**options)

		indices = data[keys['y']]>0

		values = {key:data.get(keys[key])[indices].to_numpy() for key in keys if keys[key] in data}

		func,y,parameters,yerr,cov,other = fit(**values,**settings)

		x = None
		y = parameters[-1].item()
		yerr = cov[-1][-1].item()

		data = {attr:[data[attr].iloc[0]] for attr in data}

		attr = keys['y']
		data[attr] = y
		data[attr] = data[attr]/np.log(size*info.size)

		attr = keys['yerr']
		data[attr] = yerr
		data[attr] = data[attr]/np.log(size*info.size)

		data = dataframe(data)

		return data

	data = agg(data)

	return data


def func_samples(data,*args,**kwargs):
	for i in data:
		if isinstance(i,tuple) and i and any(isinstance(j,tuple) for j in i):
			data = tuple(k for i in data for j in i for k in j)
		else:
			data = tuple(j for i in data for j in i)
		break
	return data

def func_samples_err(data,*args,**kwargs):
	data = tuple((None,))
	return data

def func_samples_process(data,values,metadata,properties,*args,**kwargs):
	if isinstance(values,arrays):
		data = concatenate((values,data),axis=1)
	return data

def func_samples_process_err(data,values,metadata,properties,*args,**kwargs):
	return data

def func_hist(data,*args,attr=None,**kwargs):

	try:
		kwargs.update({
			'none':dict(scale='linear',range=[0,1]),
			'povm':dict(scale='log',range=[1e-20,1e0]),
			}.get(data['measure'].iloc[0])
			)
		data = data[attr]
	except:
		pass

	data = func_samples(data,*args,**kwargs)
	x,y = histogram(data,*args,**kwargs)
	x,y = to_tuple(x),to_tuple(y)
	return x,y

def func_hist_err(data,*args,**kwargs):
	data = tuple((None,None))
	return data

def func_hist_process(data,values,metadata,properties,*args,**kwargs):
	if isinstance(values,arrays):
		data += values
	return data

def func_hist_process_err(data,values,metadata,properties,*args,**kwargs):
	return data

def func_hist_x(data,*args,**kwargs):
	x,y = func_hist(data,*args,**kwargs)
	return x

def func_hist_y(data,*args,**kwargs):
	x,y = func_hist(data,*args,**kwargs)
	return y

def func_hist_xerr(data,*args,**kwargs):
	data = func_hist_err(data,*args,**kwargs)
	return data

def func_hist_yerr(data,*args,**kwargs):
	data = func_hist_err(data,*args,**kwargs)
	return data

def func_sample_y(data,*args,**kwargs):
	data = sum(np.array(i) for i in data)
	data = data.reshape(1,*data.shape)
	return data

def func_sample_x(data,*args,**kwargs):
	data = data.iloc[0]
	return data

def func_sample_yerr(data,*args,**kwargs):
	data = tuple((None,))
	return data

def func_sample_xerr(data,*args,**kwargs):
	data = tuple((None,))
	return data

def func_sample_process_x(data,values,metadata,properties,*args,**kwargs):
	return data

def func_sample_process_y(data,values,metadata,properties,*args,**kwargs):
	if isinstance(values,arrays):
		data += values
	return data

def func_sample_process_xerr(data,values,metadata,properties,*args,**kwargs):
	return data

def func_sample_process_yerr(data,values,metadata,properties,*args,**kwargs):
	return data

def func_sample_function(data,*args,function=None,**kwargs):

	def parse(attr,data):
		nulls = {}
		if data is None:
			data = None
		elif all(i is None for i in data):
			data = None
		if data is not None and attr in nulls:
			data = np.array(data)
			data[(is_naninf(data))|(data<epsilon(data.dtype))] = nulls[attr]
		return data

	def function(data,*args,function=function,**kwargs):
		return measurement(data,*args,function=function,**kwargs)

	funcs = {}

	attr = 'x'
	def func(attr,data):
		data = data[attr]
		return data
	funcs[attr] = func

	attr = 'y'
	def func(attr,data):
		data = data[attr]
		return data
	funcs[attr] = func

	attr = 'xerr'
	def func(attr,data):
		data = data[attr]
		return data
	funcs[attr] = func

	attr = 'yerr'
	def func(attr,data):
		data = data[attr]
		return data
	if attr:
		funcs[attr] = func

	funcs = {attr:parse(attr,funcs[attr](attr,data)) for attr in funcs if getter(data,attr,delimiter=delim) is not None}

	setter(data,funcs,delimiter=delim,default=True)

	return data

def func_sample_wrapper_x(data,*args,function=None,**kwargs):

	def function(data,*args,function=function,**kwargs):
		return measurement(data,*args,function=function,**kwargs)

	info = function(data)

	data = data['x']*info.size

	return data

def func_process_x(data,*args,**kwargs):
	data = data.iloc[0]
	return data

def func_process_y(data,*args,**kwargs):
	data = tuple(data) if isinstance(data,iterables) and len(data)>1 else data
	return data

def func_process_xerr(data,*args,**kwargs):
	data = None
	return data

def func_process_yerr(data,*args,**kwargs):
	data = tuple(data) if isinstance(data,iterables) and len(data)>1 else data
	return data

def func_information_x(data,*args,**kwargs):
	data = data.iloc[0]
	return data

def func_information_y(data,*args,**kwargs):
	data = tuple(data) if isinstance(data,iterables) and len(data)>1 else data
	return data

def func_information_xerr(data,*args,**kwargs):
	data = None
	return data

def func_information_yerr(data,*args,**kwargs):
	data = tuple(data) if isinstance(data,iterables) and len(data)>1 else data
	return data

def func_information_process_x(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if not isinstance(data,iterables) else data
	for key,i in zip(keys,data):
		values[key] = i
	data = values
	return data

def func_information_process_y(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if not isinstance(data,iterables) else data
	for key,i in zip(keys,data):
		values[key] = np.array([*values.get(key,[]),*flatten(i)])
	data = values
	return data

def func_information_process_xerr(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if not isinstance(data,iterables) else data
	for key,i in zip(keys,data):
		values[key] = i
	data = values
	return data

def func_information_process_yerr(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if not isinstance(data,iterables) else data
	for key,i in zip(keys,data):
		values[key] = np.array([*values.get(key,[]),*flatten(i)])
	data = values
	return data

def func_information_function(data,*args,function=None,**kwargs):

	keys = data['y']
	keys = list(keys) if isinstance(keys,dict) else range(len(keys)) if keys is not None else None
	keys = natsorted(keys) if keys is not None else None

	def parse(attr,data):
		nulls = {'y':nan}
		if data is None:
			data = None
		elif all(i is None for i in data):
			data = None
		if data is not None and attr in nulls:
			data = np.array(data)
			data[(is_naninf(data))|(data<epsilon(data.dtype))] = nulls[attr]
		elif isinstance(data,iterables):
			data = np.array(data)
		return data

	def function(attr,key,data,*args,function=function,**kwargs):
		data = {attr:data[attr] if not isinstance(data[attr],dict) or key not in data[attr] else data[attr][key] for attr in data}
		data = {
			**({attr:data[attr] for attr in data if attr in ALL} if any(attr in ALL for attr in data) else {}),
			**({attr:data[OTHER][attr] for attr in data[OTHER] if attr not in ALL} if (OTHER in data) else {}),
			**({data[OTHER][attr][OTHER]:data[attr] for attr in data[OTHER] if attr in ALL and attr in data and data.get(attr) is not None} if (OTHER in data) else {}),
			}
		return measurement(data,*args,function=function,**kwargs)

	funcs = {}

	attr = 'x'
	def func(attr,key,data):
		data = 1/data[attr][key]
		return data
	funcs[attr] = func

	attr = 'y'
	def func(attr,key,data):
		if None in data[attr][key]:
			return 0
		info,size = function(attr,key,data),data[attr][key].size
		data = mean(data[attr][key])
		data = abs(data)
		data = data/log(info.size)
		return data
	funcs[attr] = func

	attr = 'xerr'
	def func(attr,key,data):
		data = data[attr][key]
		return data
	funcs[attr] = func

	attr = 'yerr'
	def func(attr,key,data):
		if None in data[attr][key]:
			return 0
		info,size = function(attr,key,data),data[attr][key].size
		data = mean(data[attr][key]) - mean(data[attr[0]][key])**2
		data = abs(data)
		data = sqrt(data/(size*info.size))/log(size)
		return data
	if attr:
		funcs[attr] = func

	funcs = {attr:parse(attr,[funcs[attr](attr,key,data) for key in keys]) for attr in funcs if getter(data,attr,delimiter=delim) is not None} if keys is not None else {}

	setter(data,funcs,delimiter=delim,default=True)

	return data

def func_stats_x(data,*args,**kwargs):
	data = data.iloc[0]
	return data

def func_stats_y(data,*args,**kwargs):
	data = sum(np.array(i) for i in data)
	return data

def func_stats_xerr(data,*args,**kwargs):
	data = None
	return data

def func_stats_yerr(data,*args,**kwargs):
	data = None
	return data

def func_stats_process_x(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if not isinstance(data,iterables) else data
	for key,i in zip(keys,data):
		if key not in values:
			values[key] = i
		else:
			values[key] = i
	data = values
	return data

def func_stats_process_y(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if not isinstance(data,iterables) else data
	for key,i in zip(keys,data):
		if key not in values:
			values[key] = i
		else:
			values[key] += i
	data = values
	return data

def func_stats_process_xerr(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if not isinstance(data,iterables) else data
	for key,i in zip(keys,data):
		if key not in values:
			values[key] = i
		else:
			values[key] = i
	data = values
	return data

def func_stats_process_yerr(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if not isinstance(data,iterables) else data
	for key,i in zip(keys,data):
		if key not in values:
			values[key] = i
		else:
			values[key] = i
	data = values
	return data

def func_stats_function(data,*args,function=None,x=None,y=None,xerr=None,yerr=None,settings=None,**kwargs):

	keys = data['y']
	keys = list(keys) if isinstance(keys,dict) else range(len(keys)) if keys is not None else None
	keys = natsorted(keys) if keys is not None else None

	def parse(attr,data):
		nulls = {'y':nan}
		if data is None:
			data = None
		elif all(i is None for i in data):
			data = None
		if data is not None and attr in nulls:
			data = np.array(data)
			data[(is_naninf(data))|(data<epsilon(data.dtype))] = nulls[attr]
		elif isinstance(data,iterables):
			data = np.array(data)
		return data

	def function(attr,key,data,*args,function=function,**kwargs):
		data = {attr:data[attr] if not isinstance(data[attr],dict) or key not in data[attr] else data[attr][key] for attr in data}
		data = {
			**({attr:data[attr] for attr in data if attr in ALL} if any(attr in ALL for attr in data) else {}),
			**({attr:data[OTHER][attr] for attr in data[OTHER] if attr not in ALL} if (OTHER in data) else {}),
			**({data[OTHER][attr][OTHER]:data[attr] for attr in data[OTHER] if attr in ALL and attr in data and data.get(attr) is not None} if (OTHER in data) else {}),
			}
		return measurement(data,*args,function=function,**kwargs)

	settings = {} if settings is None else settings
	settings = {**settings,**{setting:default for setting,default in {'func':None,'options':{}}.items() if settings.get(setting) is None}}

	funcs = {}

	attr = 'x'
	def func(attr,key,data):
		data = data[attr][key]
		return data
	funcs[attr] = func

	attr = 'y'
	def func(attr,key,data):
		if None in data[attr][key]:
			return 0
		info,size = function(attr,key,data),data[attr][key].size

		X = array(data[x][key] if x in ALL else data[OTHER][x])
		Y = None
		data = data[attr][key]

		if settings['func'] is None or settings['func'] in ['distance']:
			Y = info.func(X)
			data,Y = data/sum(data)/interval(X,**settings['options']),Y
			data = (1/2)*addition(absolute(data-Y))
		elif settings['func'] in ['cumulative']:
			Y = info.functional(X)
			data,Y = data/addition(data),Y
			data = maximum(absolute((cumsum(data)-Y))+absolute(difference(Y,append=Y[-1]-Y[-2])))
			print(maximum(absolute(difference(Y,append=0))),data)
		else:
			data = None

		data = data.item() if data is not None else None

		return data
	funcs[attr] = func

	attr = 'xerr'
	def func(attr,key,data):
		data = data[attr][key]
		return data
	funcs[attr] = func

	attr = 'yerr'
	def func(attr,key,data):
		data = data[attr][key]
		return data

	if attr:
		funcs[attr] = func

	funcs = {attr:parse(attr,[funcs[attr](attr,key,data) for key in keys]) for attr in funcs if getter(data,attr,delimiter=delim) is not None} if keys is not None else {}

	setter(data,funcs,delimiter=delim,default=True)

	return data

def func_histogram(data,*args,**kwargs):
	key = ['x','y']
	value = histogram(data,*args,**kwargs)
	data = dict(zip(key,value))
	return data

def func_information(data,*args,**kwargs):

	model = kwargs.get('model')
	function = kwargs.get('function')

	def function(data,*args,function=function,**kwargs):
		return measurement(data,*args,function=function,**kwargs)

	info = function(model)

	func = info.func

	data = information(func,data)

	key = [None,'error']
	value = mean(data,axis=-1),mean(data**2,axis=-1)

	key = [i for i in key]
	value = [[*i] if i.ndim>0 else i for i in value]

	data = dict(zip(key,value))

	return data

def func_transform(data,*args,**kwargs):

	boolean = {
		'attr':(lambda attr,data: all([
			value and
			(
			((key in data[attr]) and any(i not in value for i in data[attr][key][...])) or
			((key in data[attr].attrs) and (data[attr].attrs[key] not in value))
			)
			for key,value in {'noise.parameters':[]}.items()
			])),
		'key':(lambda key,data: data[key].ndim>1)
		}

	for attr in data:
		if boolean['attr'](attr,data):
			del data[attr]
		else:
			for key in data[attr]:
				if boolean['key'](key,data[attr]):
					del data[attr][key]

	path = data.filename
	data.close()
	os.system(f'h5repack {path} {path}.tmp && mv {path}.tmp {path}')

	return

def func_y(data):
	return np.abs(np.array(data['y']))#*(data['N']*np.log(data['D']))/np.log(2)

def func_yerr(data):
	return np.abs(np.array(data['yerr']))#*(data['N']*np.log(data['D']))/np.log(2)

def func_y_scale(data):
	y = np.array(data['y'])
	# y = np.array(data['y'])
	# i,j = np.argmax(y),np.argmin(y)
	# y = (y-min(y))/(max(y)-min(y))
	return y

def func_yerr_scale(data):
	y = np.array(data['yerr'])
	# y = np.array(data['y'])
	# z = np.array(data['yerr'])
	# i,j = np.argmax(y),np.argmin(y)
	# y = ((z-z[j]) - (y-y[j])*(z[i]-z[j])/(y[i]-y[j]))/(y[i]-y[j])
	return y

def func_line(data,attr=None):
	if attr not in data:
		return data
	values = {i: data[i].mean() for i in ['D','N']}
	data[attr] = 2*(sum(1/i for i in range(int(values['D']**(values['N']/2))+1,int(values['D']**(values['N'])))) -
		   ((int(values['D']**(values['N']/2))-1)/(2*int(values['D']**(values['N']/2)))))/np.log(values['D']**(2*values['N']/2))
	return data

def func_line_err(data):
	return 0

def func_objective(data):
	return np.abs(data['objective'])

def func_mutual_measure(data):
	return np.array(data['mutual.quantum']) - np.array(data['discord.quantum'])

def func_infidelity(data):
	return 1 - np.abs((1-np.array(data['y']))/(1-np.array(data['norm.pure'])))#*(data['N']*np.log(data['D']))/np.log(2)

def func_infidelity_err(data):
	return np.abs((np.array(data['yerr']))/(1-np.array(data['norm.pure'])))#*(data['N']*np.log(data['D']))/np.log(2)

def func_max_bond(data):
	return data['D']**(data['N']//2) <= data['max_bond'] <= data['D']**(data['N'])

def func_title(data,metadata):
	attr = 'N'
	return 2.2 if (data.get(attr) == max((i for key in metadata if attr in metadata[key] for i in metadata[key].get(attr)),default=None)) else None

def func_xlabel(data,metadata):
	attr = 'N'
	return -2 if not (data.get(attr) == max((i for key in metadata if attr in metadata[key] for i in metadata[key].get(attr)),default=None)) else None

def func_ylabel(data,metadata):
	attr = 'N'
	return 2 if (data.get(attr) == max((i for key in metadata if attr in metadata[key] for i in metadata[key].get(attr)),default=None)) else None


def func_array(data,eps=1e-15):
	data = np.array(data['y'])
	data[data<=eps] = nan
	return np.abs(data)

def func_state(data,eps=1e-15):
	data = np.array(data['y'])
	data[data<=eps] = nan
	return np.abs(data)

def func_kurtosis(data,attr=None):
	if attr not in data:
		raise ValueError("Incorrect attribute %s"%(attr))
		return
	def func(data):
		data = np.array(list(data))
		indices = np.arange(len(data))
		data /= np.sum(data)
		mean = np.sum(data*((indices)*1))
		variance = np.sum(data*((indices-mean)**2))
		quartic = np.sum(data*((indices-mean)**4))
		data = quartic/variance
		return data
	data = [func(i) for i in data[attr]]
	data = (np.array(data)-min(data))/(max(data)-min(data))
	data = data[0] if len(data) == 1 else data
	return data

def func_spectrum(data,attr=None):
	if attr not in data:
		raise ValueError("Incorrect attribute %s"%(attr))
		return
	def func(data):
		data = sorted(data,reverse=True)/max(np.abs(i) for i in data)
		# data = [np.array([*sort((data[i][~is_nan(data[i])]))[::-1],*data[i][is_nan(data[i])]])/np.max(np.abs(data[i][~is_nan(data[i])])) for i in range(n)]
		return data
	data = (func(i) for i in data[attr])
	data = to_tuple(data)
	return data

def func_spectrum_rank(data,attr=None,eps=None):
	if attr not in data:
		raise ValueError("Incorrect attribute %s"%(attr))
		return
	def func(data):
		data = np.array(list(data))
		if data.size and data[~is_nan(data)].size:
			data = asscalar(
				nonzero(
					sort(
						np.abs(data[~is_nan(data)])/np.max(np.abs(data[~is_nan(data)]))
					),eps=eps)
				)
		else:
			data = asscalar(
				nonzero(
					sort(
						np.abs(data[~is_nan(data)])/np.max(np.abs(data))
					),eps=eps)
				)
		return data
	data = [func(i) for i in data[attr]]
	data = data[0] if len(data) == 1 else data
	return data

def func_spectrum_sign(data,attr=None,eps=None):
	if attr not in data:
		raise ValueError("Incorrect attribute %s"%(attr))
		return
	eps = 1e-16 if eps is None else eps
	def func(data):
		data = np.array(list(data))
		data = np.abs(np.sum(data[data<eps])/np.sum(data[data>=eps]))
		return data
	data = np.array([func(i) for i in data[attr]])
	return data


def func_MN(data):
	return data['M']/data['N']

def func_tau_unit(data):
	return data['tau']/data.get('unit',1)

def func_T_unit(data):
	return data['T']/data.get('unit',1)

def func_tau_noise_scale(data):
	return data['tau']/data.get('noise.scale',1)

def func_T_noise_scale(data):
	return data['T']/data.get('noise.scale',1)

def func_tau_J(data):
	return data['tau']*data.get('parameters.zz.parameters',1)

def func_T_J(data):
	return data['T']*data.get('parameters.zz.parameters',1)

def func_variables_relative_mean(data):
	out = np.array(data['variables.relative.mean'])
	return (out/max(1,np.max(out)) if out.size else out)

def func_fisher_rank(data):
	out = np.array(list(data['fisher.eigenvalues']))
	out = sort(np.abs(out))
	out = (out/np.max(out) if out.size else out)
	out = asscalar(nonzero(out,axis=-1,eps=1e-13))
	return out

def func_fisher_eigenvalues(data):
	out = np.array(list(data['fisher.eigenvalues']))
	out = np.abs(out)
	out = (out/np.max(out) if out.size else out)
	out = to_tuple(out)
	return out

def func_hessian_rank(data):
	out = np.array(list(data['hessian.eigenvalues']))
	out = sort(np.abs(out))
	out = (out/np.max(out) if out.size else out)
	out = asscalar(nonzero(out,axis=-1,eps=1e-16))
	return out

def func_hessian_eigenvalues(data):
	out = np.array(list(data['hessian.eigenvalues']))
	out = np.abs(out)
	out = (out/np.max(out) if out.size else out)
	out = to_tuple(out)
	return out


def func_objective_func(data):
	out = data['N']*np.array(data['M'])*data['noise.parameters']*((data['D']**data['N']-1)/(data['D']**data['N']))
	return out

def func_objective_func_err(data):
	out = 0*np.array(data['M'])
	return out

def func_entropy(data):
	out = np.array(data['entropy'])/np.log(data['D']**data['N'])
	return out

def func_entropy_func(data):
	# Incorrect
	out = data['N']*np.array(data['M'])*data['noise.parameters']
	return out

def func_entropy_func_err(data):
	out = 0*np.array(data['M'])
	return out

def func_purity(data):
	out = 1-np.array(data['purity'])
	return out

def func_purity_func(data):
	out = 2*data['N']*np.array(data['M'])*data['noise.parameters']*((data['D']**data['N']-1)/(data['D']**data['N']))
	return out

def func_purity_func_err(data):
	out = 0*np.array(data['M'])
	return out

def func_similarity(data):
	out = 1-np.array(data['similarity'])
	return out

def func_similarity_func(data):
	out = 1-np.array(data['similarity'])
	return out

def func_similarity_func_err(data):
	out = 0*np.array(data['M'])
	return out

def func_divergence(data):
	out = np.array(data['divergence'])/np.log(data['D']**data['N'])
	return out

def func_divergence_func(data):
	# Incorrect
	out = data['N']*np.array(data['M'])*data['noise.parameters']*((data['D']**data['N']-1)/(data['D']**data['N']))/np.log(data['D']**data['N'])
	return out

def func_divergence_func_err(data):
	out = 0*np.array(data['M'])
	return out

def func_plot_histogram(args,kwargs,data,*arguments,function=None,settings=None,**keywords):

	def function(data,*args,function=function,**kwargs):
		return measurement(data,*args,function=function,**kwargs)

	info = function(data)

	def process(args,kwargs,data):

		x,y,xerr,yerr = args

		if settings is not None:

			def func(parameters,x,info=info):
				info.env = parameters
				return info.func(x)

			# indices = y>epsilon()
			# x,y = array(x[indices]),array(y[indices])
			# parameters = array([float(info.env)])

			# objective = lambda parameters,x=x,y=y,func=func: addition(absolute(func(parameters,x)-y)**2)/addition(absolute(y)**2)
			# options = {**dict(func=objective,parameters=parameters),**settings}

			# func,y,parameters,yerr,cov,other = fit(x,y,**options)

			indices = y>epsilon()
			x,y = x[indices],y[indices]
			parameters = info.env

			model = scipy.optimize.leastsq
			objective = lambda parameters,x,y=y,func=func: np.sum(np.abs(func(parameters,x)-y)**2)/np.sum(np.abs(y)**2)
			options = dict()

			parameters,status = model(objective,parameters,(x,y),**options)

			x = info.data
			y = func(parameters,x)

			attr = 'errorbar'
			kwarg = 'label'
			options = {
				'texify':dict(usetex=True),
				'scinotation':dict(decimals=3,scilimits=[0,1],one=False,strip=True)
				}
			string = '$%s$'%(scinotation(info.env,**options['scinotation']))
			string = texify(string,**options['texify'])
			kwargs[attr][kwarg] = string

			attr = 'legend'
			kwarg = 'set_title'
			if kwargs.get(attr):
				for value in search(kwargs[attr]):
					if not value or not value.get(kwarg):
						continue
					options = {
						'texify':dict(usetex=True),
						'scinotation':dict(decimals=3,scilimits=[0,0],one=False,strip=True)
						}
					string = 's ~:~ P(p) \\sim p^{s-1}(1-p)^{(d-1)s-1}'
					value[kwarg] = '%s'%(string) if string else ''
					value[kwarg] = texify(value[kwarg],**options['texify'])

		else:

			x = info.data
			func = info.func

			x = x
			y = func(x)

		x = np.array(x)
		y = np.array(y)

		xerr = None
		yerr = None

		return x,y,xerr,yerr

	x,y,xerr,yerr = process(args,kwargs,data)

	return x,y,xerr,yerr

def label(string,label):
	strings = {
		'epsilon':{
			1e-7:"32~\\textrm{bit}~(\\varepsilon \\sim 10^{-7})",
			1e-16:"64~\\textrm{bit}~(\\varepsilon \\sim 10^{-16})",
			1e-19:"128~\\textrm{bit}~(\\varepsilon \\sim 10^{-19})",
		}
	}

	string = strings.get(label,{}).get(string,string)

	return string


def error(data,*args,**kwargs):
	'''
	Workflow function
	Args:
		data (str,dict): Workflow data
		args (iterable): Workflow positional arguments
		kwargs (dict): Workflow keyword arguments
	Returns:
		data (dict): Workflow data
	'''

	def func(data,*args,**kwargs):

		def generator(iteration,shape,scale,seed=None,dtype=None):

			# seeded(seed)

			value = 2*random((2,*shape),dtype=dtype)-1
			value = (value[0] + 1j*value[1])/sqrt((value**2).sum(0))
			value = (scale/2)*value

			return value

		# def matmul(a,b,dtype=None):
		# 	c = zeros(a.shape,dtype=a.dtype)
		# 	for i in range(a.shape[0]):
		# 		for j in range(a.shape[1]):
		# 			c[i,j] = a[i].dot(b.T[j])
		# 	return c

		size = kwargs.get('size',2)
		ord = kwargs.get('ord',2)
		maxdtype = 'complex%d'%(int(kwargs.get('dtype').replace('complex',''))//1)
		maxftype = 'float%d'%(int(kwargs.get('dtype').replace('complex',''))//2)

		indexes = kwargs.get('indexes',[])
		iterations = range(
			max(2,(indexes.start if isinstance(indexes,range) else min(indexes))),
			max(1,(indexes.stop  if isinstance(indexes,range) else max(indexes)))+1
		)
		samples = range(kwargs.get('sample'))
		seeded(kwargs.get('seed'))
		seeds = {sample: rand(bounds=[0,1e12],random='randint') for sample in samples}

		bits = {sample:
				{
				**{types:
					{int(floor(log10(finfo('complex%d'%(bit)).eps))): bit
					for bit in kwargs.get('bits',[])}
					for types in ['numerical']
				},
				**{types: {bit: bit for bit in kwargs.get('epsilon',[])}
					for types in ['analytical','probabilistic']
				},
			}
			for sample in samples}
		eps = {sample: {types: {bit:
			power(addition([power(power(
				1+(power(10,bit,dtype=maxftype)),
				size-i-1,dtype=maxftype)-1,
				ord,dtype=maxftype)
				for i in range(size)],dtype=maxftype),
				1/ord,dtype=maxftype)
			for bit in bits[sample][types]}
			for types in bits[sample]}
			for sample in bits}


		dtype = {sample: {
			**{types: {bit: 'complex%d'%(bits[sample][types][bit]) for bit in bits[sample][types]} for types in ['numerical']},
			**{types: {bit: maxdtype for bit in bits[sample][types]} for types in ['analytical','probabilistic']},
		} for sample in bits}
		ftype = {sample: {
			**{types: {bit: 'float%d'%(bits[sample][types][bit]//2) for bit in bits[sample][types]} for types in ['numerical']},
			**{types: {bit: maxftype for bit in bits[sample][types]} for types in ['analytical','probabilistic']},
		}
		for sample in bits}

		V = sp.Matrix([[sp.exp(sp.Mul(sp.I,2*sp.pi,sp.Rational(i*j,size)))
			for j in range(size)]
			for i in range(size)])/sp.sqrt(size)

		S = [sp.Rational(randint(shape=None,bounds=[1,i]) if i>1 else 0,i) for i in randint(shape=size,bounds=[1,size**2])]
		D = lambda k=1: sp.diag(*(sp.exp(sp.Mul(sp.I,2*sp.pi,s,k)) for s in S))

		matrix = lambda k=1: V*D(k)*V.H
		norm = lambda A,bit=maxftype,ord=ord: ((((np.abs(A,dtype=bit))**ord).sum(dtype=bit))**(1/ord)).real
		numerical = lambda A,bit: array(sp.N(A,bit),dtype=maxdtype)

		A = {sample: {
			**{types: {bit: (lambda bit,types,sample,i=None,A=numerical(matrix(),bit=-bit): A)
				for bit in bits[sample][types]} for types in ['numerical']},
			**{types: {bit: (lambda bit,types,sample,i=None,A=1: A)
				for bit in bits[sample][types]} for types in ['analytical']},
			**{types: {bit: (lambda bit,types,sample,i=None,A=numerical(matrix(),bit=-bit): A)
				for bit in bits[sample][types]} for types in ['probabilistic']},
		} for sample in bits}
		B = {sample: {types: {bit: A[sample][types][bit](bit,types,sample) for bit in bits[sample][types]} for types in bits[sample]} for sample in bits}
		C = {sample: {
			**{types: {bit:
				(lambda bit,types,sample,i=None,A=numerical(matrix(),bit=-bit): 0)
				for bit in bits[sample][types]} for types in ['numerical']},
			**{types: {bit:
				(lambda bit,types,sample,i=None,A=1: 0)
				for bit in bits[sample][types]} for types in ['analytical']},
			**{types: {bit:
				(lambda bit,types,sample,i=None,A=numerical(matrix(),bit=-bit): generator(iteration=i,shape=A.shape,scale=eps[sample][types][bit],seed=seeds[sample],dtype=maxftype))
				for bit in bits[sample][types]} for types in ['probabilistic']},
		} for sample in bits}
		functions = {sample: {
			**{types: {bit:
				(lambda bit,types,sample,i=None: matmul(A[sample][types][bit](bit,types,sample,i),B[sample][types][bit],dtype=dtype[sample][types][bit]) + C[sample][types][bit](bit,types,sample,i))
				for bit in bits[sample][types]} for types in ['numerical']},

			**{types: {bit:
				(lambda bit,types,sample,i=None: multiply(A[sample][types][bit](bit,types,sample,i),B[sample][types][bit],dtype=dtype[sample][types][bit]) + C[sample][types][bit](bit,types,sample,i))
				for bit in bits[sample][types]} for types in ['analytical']},

			**{types: {bit:
				(lambda bit,types,sample,i=None: matmul(A[sample][types][bit](bit,types,sample,i),B[sample][types][bit],dtype=dtype[sample][types][bit]) + C[sample][types][bit](bit,types,sample,i))
				for bit in bits[sample][types]} for types in ['probabilistic']},
		} for sample in bits}
		values = {sample: {
			**{types: {bit:
				(lambda bit,types,sample,i=None: norm(B[sample][types][bit] - numerical(matrix(i),bit=-bit))/norm(B[sample][types][bit]))
				for bit in bits[sample][types]} for types in ['numerical']},

			**{types: {bit:
				(lambda bit,types,sample,i=None: multiply(
					divide(
						power(normalization[sample][types][bit],i,dtype=ftype[sample][types][bit]),
						norm(B[sample][types][bit]),dtype=ftype[sample][types][bit]),
						((power(1+eps[sample][types][bit],i,dtype=ftype[sample][types][bit]) - 1)),dtype=ftype[sample][types][bit]))
				for bit in bits[sample][types]} for types in ['analytical']},

			**{types: {bit:
				(lambda bit,types,sample,i=None: norm(B[sample][types][bit] - numerical(matrix(i),bit=-bit))/norm(B[sample][types][bit]))
				for bit in bits[sample][types]} for types in ['probabilistic']},
		} for sample in bits}

		normalization = {sample: {types: {bit: norm(A[sample][types][bit](bit,types,sample),ord=ord).real
			for bit in A[sample][types]} for types in A[sample]}
			for sample in bits}

		for i in iterations:

			for sample in bits:
				for types in bits[sample]:
					for bit in bits[sample][types]:

							B[sample][types][bit] = functions[sample][types][bit](bit,types,sample,i)

			if i in indexes:

				for sample in bits:
					for types in bits[sample]:
						for bit in bits[sample][types]:

							value = values[sample][types][bit](bit,types,sample,i)

							data['index'].append(i)
							data['value'].append(value)
							data['size'].append(size)
							data['epsilon'].append(10**(bit))
							data['type'].append(types)
							data['sample'].append(sample)
							data['seed'].append(seeds[sample])

		return


	permutations = {}
	groups = []
	data.update({attr:[] for attr in ['index','value','size','epsilon','type','sample','seed']})

	permutations = permuter(permutations,groups=groups) if permutations else [{}]

	for permutation in permutations:

		kwargs.update(permutation)

		func(data,**kwargs)

	return



def permutations(dictionaries,*args,**kwargs):
	for dictionary in dictionaries:
		delim = '.'
		settings = {key: getter(dictionary,attr,delimiter=delim) for key,attr in []}
		setter(dictionary,settings,delimiter=delim,default=None)
	return


def state(*args,**kwargs):
	data = array([
		[ 0.19470377-0.j,-0.32788293+0.22200675j],
		[-0.32788293-0.22200675j,0.80529623+0.j]
		])
	return data


def layout(iterable,sort=False,group=False):

	def key(key,iterable=iterable,sort=sort,group=group):
		if sort and group:
			index = (list(iterable).index(key),iterable[key])
		elif sort:
			index = list(iterable).index(key)
		elif group:
			index = iterable[key]
		else:
			index = key

		return index

	return key

def test(*args,**kwargs):
	return args,kwargs