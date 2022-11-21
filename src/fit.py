#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,copy
warnings.filterwarnings('ignore')

# Logging
import logging
logger = logging.getLogger(__name__)

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,zeros,ones
from src.utils import exp,log,abs,sqrt,norm,nanmean,nanstd,nansqrt,lstsq,curve_fit,product

def transformation(transform=None):
	'''
	Function to transform data
	Args:
		transform (str,callable): Name of transform or function with signature transform(data)
	Returns:
		transform (callable): Function to transform data
		invtransform (callable): Inverse function to transform data
	'''
	if transform is None:
		transform = lambda data:data
		invtransform = lambda data:data
	elif not isinstance(transform,str):
		transform,invtransform = transform
	elif callable(transform):
		transform,invtransform = transform, lambda data:data
	elif transform in ['linear']:
		transform = lambda data:data
		invtransform = lambda data:data		
	elif transform in ['log']:		
		transform = lambda data: log(data)
		invtransform = lambda data: exp(data)
	else:
		transform = lambda data:data		
		invtransform = lambda data:data		
	return transform,invtransform

def normalize(data,axis=None,ord=2,transform=None,dtype=None,**kwargs):
	'''
	Norm transform of data
	Args:
		data (array): Data to transform
		axis (int): Axis to perform transform
		ord (int): Order of transform
		transform (str,callable): Name of transform or function with signature transform(data)
		dtype (data_type): Data type
		kwargs (dict): Additional keyword arguments for transform
	Returns:
		out (array): Transform of data
	'''	
	if axis is not None and not isinstance(axis,int):
		axis = tuple(axis)
	transform,invtransform = transformation(transform)	
	return invtransform(norm(transform(data),axis=axis,ord=ord).astype(dtype))

def mean(data,axis=None,transform=None,dtype=None,**kwargs):
	'''
	Mean transform of data 
	Args:
		data (array): Data to transform
		axis (int): Axis to perform transform
		transform (str,callable): Name of transform or function with signature transform(data)
		dtype (data_type): Data type
		kwargs (dict): Additional keyword arguments for transform
	Returns:
		out (array): Transform of data
	'''		
	if axis is not None and not isinstance(axis,int):
		axis = tuple(axis)
	transform,invtransform = transformation(transform)
	return invtransform(nanmean(transform(data),axis=axis).astype(dtype))

def std(data,axis=None,transform=None,dtype=None,**kwargs):
	'''
	Standard deviation transform of data
	Args:
		data (array): Data to transform
		axis (int): Axis to perform transform
		transform (str,callable): Name of transform or function with signature transform(data)
		dtype (data_type): Data type
		kwargs (dict): Additional keyword arguments for transform
	Returns:
		out (array): Transform of data
	'''		
	if axis is not None and not isinstance(axis,int):
		axis = tuple(axis)
	transform,invtransform = transformation(transform)	
	n = data.shape[axis]
	return invtransform((nanstd(transform(data),axis=axis,ddof=n>1)).astype(dtype))


def sqrt(data,axis=None,transform=None,dtype=None,**kwargs):
	'''
	Square root transform of data
	Args:
		data (array): Data to transform
		axis (int): Axis to perform transform
		transform (str,callable): Name of transform or function with signature transform(data)
		dtype (data_type): Data type
		kwargs (dict): Additional keyword arguments for transform
	Returns:
		out (array): Transform of data
	'''		
	if axis is not None and not isinstance(axis,int):
		axis = tuple(axis)
	transform,invtransform = transformation(transform)	
	return invtransform(nansqrt(transform(data)))

def size(data,axis=None,transform=None,dtype=None,**kwargs):
	'''
	Size transform of data
	Args:
		data (array): Data to transform
		axis (int): Axis to perform transform
		transform (str,callable): Name of transform or function with signature transform(data)
		dtype (data_type): Data type
		kwargs (dict): Additional keyword arguments for transform
	Returns:
		out (array): Transform of data
	'''		
	if axis is not None and not isinstance(axis,int):
		axis = tuple(axis)
	transform,invtransform = transformation(transform)	
	if axis is None:
		out = data.size
	elif isinstance(axis,int):
		out = data.shape[axis]		
	else:
		out = product([data.shape[ax] for ax in axis])
	return out


def fit(x,y,_x=None,func=None,wrapper=None,coef0=None,intercept=True):
	'''
	Fit of data
	Args:
		x (array): Input data
		y (array): Output data
		_x (array): Output points to evaluate fit
		func (callable): Function to fit to data with signature func(x,*coef)
		wrapper (callable): Function to wrap fit data with signature wrapper(x,y,*coef)
		coef0 (array): Initial estimate of fit coefficients
		intercept (bool): Include intercept in fit
	Returns:
		_y (array): Fit data at _x
		coef (array): Fit model parameters
	'''	


	x[is_nan(x) | is_inf(x)] = 0

	if wrapper is None:
		wrapper = lambda x,y,*coef: y

	if func is None:
		if intercept:
			x = array([x,ones(x.size)]).T
		else:
			x = array([x]).T
		if _x is None:
			_x = x
		elif intercept:
			_x = array([_x,ones(_x.size)]).T
		else:
			_x = array([_x]).T
		try:
			coef = lstsq(x,y)[0] + 0.0
			_y = _x.dot(coef)
		except:
			_y = y
			coef = zeros(_x.shape[1])
	else:
		if _x is None:
			_x = x
		try:
			coef = curve_fit(func,x,y,p0=coef0)[0] + 0.0
			_y = func(_x,*coef)
		except:
			coef = coef0
			_y = zeros(_x.shape[0])

	if coef is not None:
		_y = wrapper(_x,_y,*coef)
	else:
		coef = zeros(3)

	return _y,coef