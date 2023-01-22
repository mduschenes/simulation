#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,copy
warnings.filterwarnings('ignore')

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import gradient,einsum
from src.utils import array,zeros,ones
from src.utils import lstsq,curve_fit,interp
from src.utils import exp,log,abs,sqrt,norm,nanmean,nanstd,nansqrt,product,is_naninf

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


def fit(x,y,_x=None,_y=None,func=None,grad=None,preprocess=None,postprocess=None,xerr=None,yerr=None,coef0=None,intercept=True,uncertainty=False,**kwargs):
	'''
	Fit of data
	Args:
		x (array): Input data
		y (array): Output data
		_x (array): Input points to evaluate fit
		_y (array): Output points to evaluate fit
		func (callable,str): Function to fit to data with signature func(x,*coef), or string for spline fit, ['linear','cubic']
		grad (callable): Gradient of function to fit to data with signature grad(x,*coef)
		preprocess (callable): Function to preprocess data with signature x,y = preprocess(x,y,*coef)
		postprocess (callable): Function to postprocess data with signature x,y = preprocess(x,y,*coef)
		xerr (array): Input error
		yerr (array): Output error
		coef0 (array): Initial estimate of fit coefficients
		intercept (bool): Include intercept in fit
		uncertainty (bool): Calculate uncertainty
		kwargs (dict[str,object]): Additional keyword arguments for fitting
	Returns:
		_y (array): Fit data at _x
		coef (array): Fit model parameters
		_y (array): Fit data at _x
		coef (array): Fit model parameters
		r (float): Fit coefficient
	'''	





	if coef0 is None:
		ncoef = 1
		kwargs['p0'] = coef0
		coef0 = (None,)
	else:
		ncoef = len(coef0)
		kwargs['p0'] = coef0
		coef0 = coef0


	if preprocess is None:
		preprocess = lambda x,y,*coef: (x,y)
	
	if postprocess is None:
		postprocess = lambda x,y,*coef: (x,y)

	x,y = preprocess(x,y,*coef0)

	x = x.at[is_naninf(x)].set(0)
	y_ = _y

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
			coefferr = zeros((*coef.shape,*coef.shape))
			_yerr = zeros(_y.shape)
		except:
			coef = zeros(_x.shape[1])
			_y = y
			coefferr = zeros((*coef.shape,*coef.shape))
			_yerr = zeros(_y.shape)

	if func is not None:
		if _x is None:
			_x = x
		if callable(func):
			try:
				coef,coefferr = curve_fit(func,x,y,**kwargs)
			except Exception as e:
				coef,coefferr = zeros(ncoef),zeros((ncoef,ncoef))
			
			if grad is None:
				grad = gradient(func,argnums=tuple(range(1,ncoef+1)),mode='fwd')
			
			coef = array(coef)
			coefferr = array(coefferr)	

			_y = func(_x,*coef)
			_grad = grad(_x,*coef)		
			_grad = array(_grad).T
			_yerr = sqrt(einsum('ui,ij,uj->u',_grad,coefferr,_grad))

		elif isinstance(func,str):
			kind = func
			smooth = kwargs.get('smooth')
			_func = interp(x,y,kind,smooth)
			func = lambda x,*coef,_func=_func: _func(x)
			coef,coefferr = zeros(ncoef),zeros((ncoef,ncoef))

			_y = func(_x,*coef)

			if yerr is not None:
				_yerr = 0
				_funcerr = interp(x,y+yerr,kind,smooth)
				funcerr = lambda x,*coef,_func=_funcerr: _func(x)
				_yerr += abs(funcerr(_x,*coef) - _y)

				_funcerr = interp(x,y-yerr,kind,smooth)
				funcerr = lambda x,*coef,_func=_funcerr: _func(x)
				_yerr += abs(funcerr(_x,*coef) - _y)

				_yerr /= 2

	else:
		func = lambda x,*coef: y
		

	if coef is not None:
		_x,_y = postprocess(_x,_y,*coef)
	elif coef0 is not None:
		coef = zeros(coef0.shape)
	else:
		coef = None

	if uncertainty:
		y_ = func(x,*coef)
		r = 1 - (((y - y_)**2).sum()/((y - y.mean())**2).sum())

	if uncertainty:
		return _y,coef,_yerr,coefferr,r
	else:
		return _y,coef