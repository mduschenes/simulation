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
import pandas as pd
from pandas.api.types import is_float_dtype
from natsort import natsorted,realsorted
from math import prod

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,zeros,rand,random,randint,linspace,logspace,seeded,finfo,texify,scinotation,histogram,information
from src.utils import addition,multiply,divide,power,matmul,sqrt,floor,exp,log,log10,absolute,maximum,minimum,sort
from src.utils import to_tuple,is_nan,is_naninf,asscalar
from src.utils import grouper,conditions,flatten,concatenate,inplace,epsilon
from src.utils import orng as rng
from src.utils import arrays,scalars,dataframes,integers,floats,nonzero,delim,nan

from src.iterables import permuter,setter,getter,search,Dictionary

from src.plot import ALL,AXES,DELIMITER

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

def func_stat_group(data,samples=None,seed=None,independent=None,dependent=None,func=None,sort=None,**kwargs):

	independent = [independent] if isinstance(independent,str) else independent
	dependent = [dependent] if isinstance(dependent,str) else dependent

	if independent is None or dependent is None or any(attr not in data for attr in independent) or any(attr not in data for attr in dependent):
		return data

	if seed is not None:
		options = dict(seed=seed)
		seeded(**options)

	if func is None:
		func = 'mean'
	if sort is None:
		sort = 'idxmax'

	if isinstance(func,str):
		pass
	if isinstance(sort,str):
		sort = lambda data,sort=sort: getattr(data.abs(),sort)()

	def split(data):
		options = dict(seed=seed)
		key = seeded(**options)

		options = dict(drop=True)
		data = data.reset_index(**options)

		options = dict(frac=1,random_state=rng)
		data = data.sample(**options)

		options = dict(drop=True)
		data = data.reset_index(**options)

		data[attr] = data.index % samples

		return data

	def agg(data):
		by = independent
		agg = {**{attr:'first' for attr in data},**{attr:func for attr in dependent}}
		options = dict(by=by,agg=agg)
		data = grouper(data,**options)
		return data

	def mask(data):
		booleans = {attr:lambda data,attr=attr:(data[attr].index == sort(data[attr])) for attr in dependent}
		boolean = conditions([booleans[attr](data) for attr in booleans],op='and')
		data = data[boolean]
		return data

	if samples is not None:

		attr = '__group__'

		by = independent
		apply = split
		options = dict(by=by,apply=apply)
		data = grouper(data,**options)

		by = attr
		apply = lambda data: mask(agg(data))
		options = dict(by=by,apply=apply)
		data = grouper(data,**options)

	else:

		data = mask(agg(data))

	def func(data):
		try:
			return len(data)
		except:
			return data

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
	data = sum((np.array(i) for i in data))
	data = data.reshape(1,*data.shape)
	return data

def func_sample_x(data,*args,**kwargs):
	data = sum((np.array(i) for i in data))/len(data)
	return data

def func_sample_yerr(data,*args,**kwargs):
	data = tuple((None,))
	return data

def func_sample_xerr(data,*args,**kwargs):
	data = tuple((None,))
	return data

def func_sample_process(data,values,metadata,properties,*args,**kwargs):
	if isinstance(values,arrays):
		data += values
	return data

def func_sample_process_err(data,values,metadata,properties,*args,**kwargs):
	return

def func_sample_function(data,*args,eps=None,**kwargs):
	data = data['y']
	if eps:
		data = np.array(data)
		boolean = data>0
		data[boolean] = np.log(data[boolean])
		key = boolean & (np.abs((data-data[boolean].max())/(((data-data[boolean].max())**2).mean()))>eps)
		data[boolean] = np.exp(data[boolean])
		value = 0
		data[key] = value
	return data

def func_sample_function_err(data,*args,eps=None,**kwargs):
	data = data['yerr']
	return data

def func_info_y(data,*args,**kwargs):
	data = sum((np.array(i) for i in data))
	data = data.reshape(*data.shape)
	return data

def func_info_yerr(data,*args,**kwargs):
	data = sum((np.array(i) for i in data))
	data = data.reshape(*data.shape)
	return data

def func_info_process_y(data,values,metadata,properties,*args,**kwargs):
	if isinstance(values,arrays):
		data += values
	return data

def func_info_process_yerr(data,values,metadata,properties,*args,**kwargs):
	return data

def func_info_function_y(data,*args,**kwargs):
	func = lambda x,n: n*np.exp(-n*x)
	norm = lambda x: x/np.sum(x,axis=-1)[...,None]

	size = min(len(data['x']),len(data['y']))

	x = data['x']
	z = np.array([*data['label']['sample.array.linear.x']])[None,:]

	n = ((data['label']['D'])**(x))[:,None]

	y = np.array(data['y'])
	y = norm(y)

	z = func(z,n)
	z = norm(z)

	data['x'] = 1/x
	data['y'] = -np.sum((y*(np.log(z)-np.log(y)))*((y!=0) & (z!=0)),axis=-1)
	data['xerr'] = None
	data['yerr'] = None

	return data

def func_information_x(data,*args,**kwargs):
	data = data.iloc[0]
	return data

def func_information_y(data,*args,**kwargs):
	data = tuple(data)
	return data

def func_information_xerr(data,*args,**kwargs):
	data = None
	return data

def func_information_yerr(data,*args,**kwargs):
	data = tuple(data)
	return data

def func_information_process_x(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if data is None else data
	for key,i in zip(keys,data):
		values[key] = i
	data = values
	return data

def func_information_process_y(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if data is None else data
	for key,i in zip(keys,data):
		values[key] = np.array([*values.get(key,[]),*flatten(i)])
	data = values
	return data

def func_information_process_xerr(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if data is None else data
	for key,i in zip(keys,data):
		values[key] = i
	data = values
	return data

def func_information_process_yerr(data,values,metadata,properties,*args,**kwargs):
	keys = metadata['x']
	values = {} if not isinstance(values,dict) else values
	data = [data for key in keys] if data is None else data
	for key,i in zip(keys,data):
		values[key] = np.array([*values.get(key,[]),*flatten(i)])
	data = values
	return data

def func_information_function(data,*args,function=None,**kwargs):

	keys = data['y']
	keys = list(keys) if isinstance(keys,dict) else range(len(keys)) if keys is not None else None
	keys = natsorted(keys) if keys is not None else None

	def parse(attr,key,data):
		if data is None:
			data = None
		elif all(i is None for i in data):
			data = None
		if data is not None:
			data = np.array(data)
			data[(is_naninf(data))|(data<epsilon(data.dtype))] = {'x':0,'y':nan,'xerr':0,'yerr':0}.get(attr,0)
		return data

	default = lambda data,*args,**kwargs: 1
	if function is None:
		function = default
	elif isinstance(function,str):
		function = load(function,default=default)
	def decorator(function):
		def wrapper(attr,key,data,*args,**kwargs):
			data = {attr:data[attr] if not isinstance(data[attr],dict) or key not in data[attr] else data[attr][key] for attr in data}
			return function(data,*args,**kwargs)
		return wrapper

	function = decorator(function)

	funcs = {}

	attr = 'x'
	def func(attr,key,data):
		data = 1/data[attr][key]
		return data
	funcs[attr] = func

	attr = 'y'
	def func(attr,key,data):
		number,size = function(attr,key,data),data[attr][key].size
		data = np.mean(data[attr][key])
		data = data/np.log(number) - 1
		return data
	funcs[attr] = func

	attr = 'xerr'
	def func(attr,key,data):
		data = data[attr][key]
		return data
	funcs[attr] = func

	attr = 'yerr'
	def func(attr,key,data):
		number,size = function(attr,key,data),data[attr][key].size
		data = np.mean(data[attr][key]) - np.mean(data[attr[0]][key])**2
		data = np.sqrt(data/(size*number))/np.log(size)
		return data
	funcs[attr] = func

	funcs = {attr:parse(attr,key,[funcs[attr](attr,key,data) for key in keys]) for attr in funcs if attr in data} if keys is not None else {}

	data.update(funcs)

	return data

def func_histogram(obj,*args,**kwargs):
	key = ['x','y']
	value = histogram(obj,*args,**kwargs)
	data = dict(zip(key,value))
	return data

def func_information(obj,*args,**kwargs):
	n = obj.size
	eps = epsilon(obj.dtype)
	def func(x,n=n,eps=eps):
		x = (n-1)*((1-x)**(n-2)) # (n/(1-np.exp(-n)))*np.exp(-n*obj) # n*np.exp(-n*obj)
		x /= addition(x)
		x = inplace(x,x<eps,1)
		return x
	key = ['','err']
	value = information(func,obj)
	value = addition(value)/n,addition(value**2)/n
	data = dict(zip(key,value))
	return data

def func_size_array(data,*args,**kwargs):
	return data['label']['D']**(2*data['x'])

def func_size_state(data,*args,**kwargs):
	return data['label']['D']**(1*data['x'])

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


def func_fit_histogram(args,kwargs,attributes):

	def process(args,kwargs,attributes):

		x,y,xerr,yerr = args

		indices = (y != 0) & (~is_nan(y)) if y is not None else None

		if not indices.any():
			return x,y,xerr,yerr

		x = x[indices] if x is not None else None
		y = y[indices] if y is not None else None
		xerr = xerr[indices] if xerr is not None else None
		yerr = yerr[indices] if yerr is not None else None

		# window = lambda size: np.ones(size)/size
		# options = dict(mode='same')
		# size = max(10,len(y)//100)
		# y = np.convolve(y,window(size),**options)

		attributes['d'] = attributes['D']**attributes['N']

		return x,y,xerr,yerr

	x,y,xerr,yerr = process(args,kwargs,attributes)

	func = lambda parameters,x: parameters[0]*attributes['d']*np.exp(-parameters[1]*attributes['d']*x)
	delta = lambda parameters,x: np.array([np.exp(-parameters[1]*d*x),-x*parameters[0]*d*np.exp(-parameters[1]*d*x)])

	model = scipy.optimize.leastsq
	objective = lambda parameters,x,y,func=func,delta=delta: np.abs(func(parameters,x)-y)
	error = lambda parameters,x,y,err,func=func,delta=delta: np.einsum('i...,j...,ij->...',*[delta(parameters,x)]*2,err)
	parameters = [1,1]
	options = dict(full_output=True)

	if len(x)>1 and len(y)>1:
		parameters,err,info,msg,code = model(objective,parameters,(x,y),**options)
	else:
		parameters,err,info,msg,code = parameters,None,None,None,None

	x = x
	y = func(parameters,x)
	xerr = None
	yerr = None#error(parameters,x,y,err)

	# x,y,xerr,yerr = x[::10 if len(x)>10*10 else 1],y[::10 if len(y)>10*10 else 1],xerr,yerr

	attr = 'errorbar'
	kwarg = 'label'
	if kwargs.get(attr) and kwargs[attr].get(kwarg):
		options = {
			'texify':dict(usetex=True),
			'scinotation':dict(decimals=3,scilimits=[0,0],one=False,strip=True)
			}
		string = ''
		for prop in ['set_%sscale'%(axes) for axes in AXES]:
			if not kwargs.get(prop):
				continue
			values = [data for data in search(kwargs.get(prop)) if data]
			if (len(set(data.get('value') for data in values)) == 1) or not any(data.get('value') not in [None,'linear'] or data.get('obj') not in [None] for data in values):
				continue
			values = [data.get('value') for data in values if data.get('obj')==kwargs[attr].get('obj')]
			if not values:
				continue
			string = 'Linear' if all(data=='linear' for data in values) else 'Log'
			string = '~(\\textrm{%s})'%(string)
			break

		strings = '\n'.join([
				texify('%s = %s'%(
				[r'\alpha',r'\beta'][i],
				scinotation(parameters[i],error=err[i][i] if err is not None else None,**options['scinotation'])),
				**options['texify']
				)
			for i in range(len(parameters))])
		# kwargs[attr][kwarg] = ('%s%s'%(kwargs[attr][kwarg],string) + '\n' + strings) if isinstance(kwargs[attr].get(kwarg),str) else strings
		# kwargs[attr][kwarg] = strings
		kwargs[attr][kwarg] = '%s%s%s%s'%(DELIMITER,kwargs[attr][kwarg].replace('$',''),DELIMITER,strings) if isinstance(kwargs[attr].get(kwarg),str) else strings

	attr = 'legend'
	kwarg = 'set_title'
	if kwargs.get(attr):
		for data in search(kwargs[attr]):
			if not data or not data.get(kwarg):
				continue
			string = 'P(p) = \\alpha D^{N} e^{-\\beta D^{N} p}'
			# data[kwarg] = '%s ~:~ %s'%(data[kwarg],string) if isinstance(data.get(kwarg),str) and not data.get(kwarg).replace('$','').endswith(string) else data[kwarg]
			data[kwarg] = '%s'%(string) if string else '' #if isinstance(data.get(kwarg),str) and not data.get(kwarg).replace('$','').endswith(string) else data[kwarg]
			data[kwarg] = texify(data[kwarg],**options['texify'])

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

# def func(data,*args,**kwargs):
# 	if data is None:
# 		data = {}
	
# 	default = {'index':[],'value':[]}

# 	data.update({attr: default[attr] for attr in default if attr not in data})

# 	data.update({attr: [*data[attr]] for attr in data})

# 	for i in kwargs.get('indexes'):
# 		data['index'].append(max(data['index'],default=-1)+1)
# 		data['value'].append(i)

# 	return