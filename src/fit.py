#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,copy,traceback
warnings.filterwarnings('ignore')

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import gradient,diag
from src.utils import array,zeros,ones
from src.utils import lstsq,curve_fit,piecewise_fit,interp,piecewise,standardize
from src.utils import exp,log,abs,sqrt,sort,norm,nanmean,nanstd,nansqrt,product,is_naninf,allclose
from src.utils import nan,null

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


def fits(x,y,_x=None,_y=None,func=None,preprocess=None,postprocess=None,xerr=None,yerr=None,coef=None,coeferr=None,coefframe=None,intercept=False,uncertainty=False,**kwargs):
	'''
	Piecewise fit of data
	Args:
		x (array): Input data
		y (array): Output data
		_x (array): Input points to evaluate fit
		_y (array): Output points to evaluate fit
		funcs (iterable[callable,str]): Functions to fit to data with signature func(coef,x), or string for spline fit, ['linear','cubic']
		preprocess (callable): Function to preprocess data with signature x,y,coef = preprocess(x,y,coef) (with coef argument/return optional)
		postprocess (callable): Function to postprocess data with signature x,y,coef = postprocess(x,y,coef) (with coef argument/return optional)
		xerr (array): Input error
		yerr (array): Output error
		coef (array): Model coefficients
		coeferr (array): Model coefficients error
		coefframe (bool): Coefficients are in transformed frame
		intercept (bool): Include intercept in fit
		shape (iterable[int]): shape of piecewise coefficients
		uncertainty (bool): Calculate uncertainty
		kwargs (dict[str,object]): Additional keyword arguments for fitting
	Returns:
		_y (array): Fit data at _x
		_coef (array): Fit model parameters
		_yerr (array): Fit data error at _x
		_coeferr (array): Fit model parameters error
		_r (float): Fit coefficient
	'''	

	if callable(func):
		funcs = [funcs]
	else:
		funcs = func

	n = len(funcs)


	if n == 1

		bounds,coefs = [None],coef
	else:
		assert (shape is not None), "shape must be supplied for n>1 funcs"
		indices = [slice(sum(shape[:i-1]),sum(shape[:i])) for i in range(1,n+2)]

		bounds,coefs = coef[indices[0]] if coef is not None else,[coefs[index] for index in indices[1:]]

	yerrs = yerr
	
	for i in range(n):
		
		func = funcs[i]
		coef = coefs[i]
		bound = 
		if bounds[i] is
		condition = (x<=bounds[i]) if bounds[i] is not None else x==x) if i==0 else ((x>=bounds[i-1]) & (x<=bounds[i])) if i < (n-1) else (x>=bounds[i-1])
		
		_x = x[condition]
		_y = y[condition]
		_yerr = y[condition]

		kwargs['coef'] = coef
		kwargs['yerr'] = _yerr

		_coef,_coeferr = curve_fit(func,_x,_y,**kwargs)

		_coefs.append(_coef)
		_coeferrs.append(_coeferr)

	func = function
	_coef = array(_coefs)
	_coeferr = array(_coeferrs)

	return func,_coef,_coeferr



def fit(x,y,_x=None,_y=None,func=None,preprocess=None,postprocess=None,xerr=None,yerr=None,coef=None,coeferr=None,coefframe=None,intercept=False,uncertainty=False,**kwargs):
	'''
	Fit of data
	Args:
		x (array): Input data
		y (array): Output data
		_x (array): Input points to evaluate fit
		_y (array): Output points to evaluate fit
		func (callable,str): Function to fit to data with signature func(coef,x), or string for spline fit, ['linear','cubic']
		preprocess (callable): Function to preprocess data with signature x,y,coef = preprocess(x,y,coef) (with coef argument/return optional)
		postprocess (callable): Function to postprocess data with signature x,y,coef = postprocess(x,y,coef) (with coef argument/return optional)
		xerr (array): Input error
		yerr (array): Output error
		coef (array): Model coefficients
		coeferr (array): Model coefficients error
		coefframe (bool): Coefficients are in transformed frame
		intercept (bool): Include intercept in fit
		uncertainty (bool): Calculate uncertainty
		kwargs (dict[str,object]): Additional keyword arguments for fitting
	Returns:
		_y (array): Fit data at _x
		_coef (array): Fit model parameters
		_yerr (array): Fit data error at _x
		_coeferr (array): Fit model parameters error
		_r (float): Fit coefficient
	'''	

	if _x is None:	
		_x = x
	if _y is None:
		_y = y
	if xerr is None:
		xerr = None
	if yerr is None:
		yerr = None
	if coef is None:
		coef = None
	if coeferr is None:
		coeferr = None

	_xerr = xerr
	_yerr = yerr
	_coef = coef
	_coeferr = coeferr

	transform,invtransform = standardize(x,y,coef,preprocess=preprocess,postprocess=postprocess,**kwargs)
	gradtransform = gradient(invtransform,argnums=(0,1,2),mode='fwd')

	if not coefframe:
		x,y,coef = transform(x,y,coef)
		_x,_y,_coef = transform(_x,_y,_coef)
	else:
		x,y = transform(x,y)
		_x,_y = transform(_x,_y)

	invgrad = gradtransform(x,y,coef)
	if xerr is not None:
		i = 0
		if xerr.ndim == 1:
			jac = 1/diag(invgrad[i][i])
			xerr = abs(jac*xerr)
		else:
			jac = inv(invgrad[i][i])
			xerr = jac.dot(xerr).dot(jac.T)
	if yerr is not None:
		i = 1
		if yerr.ndim == 1:
			jac = 1/diag(invgrad[i][i])
			yerr = abs(jac*yerr)
		else:
			jac = inv(invgrad[i][i])
			yerr = jac.dot(yerr).dot(jac.T)

	_invgrad = gradtransform(_x,_y,_coef)
	if _xerr is not None:
		i = 0
		if _xerr.ndim == 1:
			_jac = 1/diag(_invgrad[i][i])
			_xerr = abs(_jac*_xerr)
		else:
			_jac = inv(_invgrad[i][i])
			_xerr = _jac.dot(_xerr).dot(_jac.T)
	if _yerr is not None:
		i = 1
		if _yerr.ndim == 1:
			_jac = 1/diag(_invgrad[i][i])
			_yerr = abs(_jac*_yerr)
		else:
			_jac = inv(_invgrad[i][i])
			_yerr = _jac.dot(_yerr).dot(_jac.T)

	if not coefframe:
		_invgrad = gradtransform(x,y,coef)
		if _coeferr is not None:
			i = 2
			if _coeferr.ndim == 1:
				_jac = diag(_invgrad[i][i])
				_coeferr = abs(_jac*_coeferr)
			else:
				_jac = _invgrad[i][i]
				_coeferr = jac.dot(_coeferr).dot(jac.T)

	if func is None:

		func = lambda coef,x: x.dot(coef)

		if intercept:
			x = array([x,ones(x.size)]).T
		else:
			x = array([x]).T
		
		if intercept:
			_x = array([_x,ones(_x.size)]).T
		else:
			_x = array([_x]).T

		_coef = lstsq(x,y)
		_y = func(x)
		
		if yerr.dim == 1:
			yerr = diag(yerr)
		_coeferr = inv(x.dot(x.T)).dot(x.T).dot(yerr).dot(x).dot(inv(x.dot(x.T)))

	elif callable(func) or isinstance(func,(tuple,list)):

		kwargs.update({'coef':coef,'yerr':yerr,'xerr':xerr})

		if isinstance(func,(tuple,list)):
			func,_coef,_coeferr = piecewise_fit(func,x,y,**kwargs)
		else:
			func,_coef,_coeferr = curve_fit(func,x,y,**kwargs)

		grad = gradient(func,argnums=0,mode='fwd')


		_y = func(_coef,_x)
		_grad = grad(_coef,_x)

		if _coeferr.ndim == 1:
			_yerr = abs(diag(_grad)*_coeferr)
		else:
			_yerr = diag(_grad.dot(_coeferr).dot(_grad.T))

	elif isinstance(func,str):

		kwargs.update({'kind':func})
		
		_func = interp(x,y,**kwargs)
		
		func = lambda coef,x,_func=_func: _func(x)
		
		_y = func(_coef,_x)

		if yerr is not None:
			_yerr = 0
			_funcerr = interp(x,y+yerr,**kwargs)
			funcerr = lambda coef,x,_func=_funcerr: _func(x)
			_yerr += abs(funcerr(_coef,_x) - _y)

			_funcerr = interp(x,y-yerr,**kwargs)
			funcerr = lambda coef,x,_func=_funcerr: _func(x)
			_yerr += abs(funcerr(_coef,_x) - _y)

			_yerr /= 2

	else:
		func = lambda coef,x,y=y: y
	
	if uncertainty:
		_r = 1 - (((y - func(_coef,x))**2).sum()/((y - y.mean())**2).sum())

	invgrad = gradtransform(x,y,coef)
	if xerr is not None:
		i = 0
		if xerr.ndim == 1:
			jac = diag(invgrad[i][i])
			xerr = abs(jac*xerr)
		else:
			jac = invgrad[i][i]
			xerr = jac.dot(xerr).dot(jac.T)
	if yerr is not None:
		i = 1
		if yerr.ndim == 1:
			jac = diag(invgrad[i][i])
			yerr = abs(jac*yerr)
		else:
			jac = inv(invgrad[i][i])
			yerr = jac.dot(yerr).dot(jac.T)

	_invgrad = gradtransform(_x,_y,_coef)
	if _xerr is not None:
		i = 0
		if _xerr.ndim == 1:
			_jac = diag(_invgrad[i][i])
			_xerr = abs(_jac*_xerr)
		else:
			_jac = _invgrad[i][i]
			_xerr = _jac.dot(_xerr).dot(_jac.T)
	if _yerr is not None:
		i = 1
		if _yerr.ndim == 1:
			_jac = diag(_invgrad[i][i])
			_yerr = abs(_jac*_yerr)
		else:
			_jac = _invgrad[i][i]
			_yerr = _jac.dot(_yerr).dot(_jac.T)

	_invgrad = gradtransform(x,y,coef)
	if _coeferr is not None:
		i = 2
		if _coeferr.ndim == 1:
			_jac = diag(_invgrad[i][i])
			_coeferr = abs(_jac*_coeferr)
		else:
			_jac = _invgrad[i][i]
			_coeferr = _jac.dot(_coeferr).dot(_jac.T)

	x,y,coef = invtransform(x,y,coef)

	_x,_y,_coef = invtransform(_x,_y,_coef)

	if uncertainty:
		return _y,_coef,_yerr,_coeferr,_r
	else:
		return _y,_coef