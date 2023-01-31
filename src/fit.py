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
from src.utils import lstsq,curve_fit,interp,piecewise,standardize
from src.utils import exp,log,abs,sqrt,sort,norm,nanmean,nanstd,nansqrt,product,is_naninf

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


def fit(x,y,_x=None,_y=None,func=None,preprocess=None,postprocess=None,xerr=None,yerr=None,coef0=None,intercept=False,uncertainty=False,**kwargs):
	'''
	Fit of data
	Args:
		x (array): Input data
		y (array): Output data
		_x (array): Input points to evaluate fit
		_y (array): Output points to evaluate fit
		func (callable,str): Function to fit to data with signature func(x,*coef), or string for spline fit, ['linear','cubic']
		preprocess (callable): Function to preprocess data with signature x,y = preprocess(x,y)
		postprocess (callable): Function to postprocess data with signature x,y = postprocess(x,y)
		xerr (array): Input error
		yerr (array): Output error
		coef0 (array): Initial estimate of fit coefficients
		intercept (bool): Include intercept in fit
		uncertainty (bool): Calculate uncertainty
		kwargs (dict[str,object]): Additional keyword arguments for fitting
	Returns:
		_y (array): Fit data at _x
		coef (array): Fit model parameters
		_yerr (array): Fit data error at _x
		coeferr (array): Fit model parameters error
		r (float): Fit coefficient
	'''	

	if coef0 is None:
		coef0 = (coef0,)

	if _x is None:	
		_x = x
	if _y is None:
		_y = y
	if xerr is None:
		xerr = None
	if yerr is None:
		yerr = None
	x_ = _x
	y_ = _y
	_xerr = xerr
	_yerr = yerr
	coef = coef0

	ncoef = len(coef0)

	if preprocess is None:
		preprocess = lambda x,y: (x,y)
	
	if postprocess is None:
		postprocess = lambda x,y: (x,y)	

	grad = gradient(func,argnums=tuple(range(1,ncoef+1)),mode='fwd')
	
	x,y = preprocess(x,y)
	_x,_y = preprocess(_x,_y)
	xerr,xerr = preprocess(xerr,yerr)
	_xerr,_yerr = preprocess(_xerr,_yerr)

	transform,invtransform = standardize(x,y,coef0,**kwargs)

	gradprocess = gradient(postprocess,argnums=1,mode='fwd')
	gradtransform = gradient(invtransform,argnums=1,mode='fwd')
	gradcoeftransform = gradient(invtransform,argnums=2,mode='fwd')

	x,y,coef0 = transform(x,y,coef0)
	_x,_y,coef = transform(_x,_y,coef0)
	xerr,yerr = transform(xerr,yerr)
	_xerr,_yerr = transform(_x,_y)

	if func is None:

		if intercept:
			x = array([x,ones(x.size)]).T
		else:
			x = array([x]).T
		
		if intercept:
			_x = array([_x,ones(_x.size)]).T
		else:
			_x = array([_x]).T

		try:
			coef = lstsq(x,y)[0] + 0.0
			_y = _x.dot(coef)
			coeferr = zeros((*coef.shape,*coef.shape))
			_yerr = zeros(_y.shape)
		except:
			coef = zeros(_x.shape[1])
			_y = y
			coeferr = zeros((*coef.shape,*coef.shape))
			_yerr = zeros(_y.shape)

	elif callable(func):
		
		kwargs.update({'coef0':coef0,'yerr':yerr,'xerr':xerr})

		try:
			coef,coeferr = curve_fit(func,x,y,**kwargs)
		except Exception as e:
			print(traceback.format_exc())
			coef,coeferr = coef0,zeros((ncoef,ncoef))

		coef = array(coef)
		coeferr = array(coeferr)

		_y = func(_x,*coef)
		_grad = array(grad(_x,*coef)).T

		_yerr = _grad.dot(coeferr).dot(_grad.T)

	elif isinstance(func,str):

		kwargs.update({'kind':func})
		
		_func = interp(x,y,**kwargs)
		
		func = lambda x,*coef,_func=_func: _func(x)
		
		coef,coeferr = zeros(ncoef),zeros((ncoef,ncoef))

		_y = func(_x,*coef)

		if yerr is not None:
			_yerr = 0
			_funcerr = interp(x,y+yerr,**kwargs)
			funcerr = lambda x,*coef,_func=_funcerr: _func(x)
			_yerr += abs(funcerr(_x,*coef) - _y)

			_funcerr = interp(x,y-yerr,**kwargs)
			funcerr = lambda x,*coef,_func=_funcerr: _func(x)
			_yerr += abs(funcerr(_x,*coef) - _y)

			_yerr /= 2

	elif isinstance(func,(tuple,list)):
		
		func = piecewise(func,**kwargs)


		_returns = fit(
			x,y,_x=_x,_y=_y,
			func=func,grad=grad,
			xerr=xerr,yerr=yerr,
			preprocess=None,postprocess=postprocess,
			coef0=coef0,intercept=intercept,uncertainty=uncertainty,
			**kwargs)

		return _returns
		
	else:
		func = lambda x,*coef: y
	
	if uncertainty:
		y_ = func(x,*coef)
		r = 1 - (((y - y_)**2).sum()/((y - y.mean())**2).sum())

	_gradprocess = array((gradprocess(_x,_y)[1]))
	_gradtransform = array((gradtransform(_x,_y)[1]))
	_gradcoeftransform = array((gradcoeftransform(_x,_y,coef)[2]))

	x,y,coef0 = invtransform(x,y,coef0)
	_x,_y,coef = invtransform(_x,_y,coef0)
	x_,y_ = invtransform(x_,y_)
	xerr,yerr = invtransform(xerr,yerr)
	_xerr,_yerr = invtransform(_xerr,_yerr)

	x,y = postprocess(x,y)
	_x,_y = postprocess(_x,_y)
	x_,y_ = postprocess(x_,y_)
	xerr,yerr = postprocess(xerr,yerr)
	_xerr,_yerr = postprocess(_xerr,_yerr)

	_yerr = sqrt(diag(_gradprocess.dot(_yerr).dot(_gradtransform.T)))
	coeferr = _gradcoeftransform.dot(coeferr).dot(_gradcoeftransform.T)

	print(_y)
	print(postprocess(*invtransform(_x,func(transform(*preprocess(_x,_y),coef)[0],*transform(*preprocess(_x,_y),coef)[-1]))))

	print(coef)
	print(coeferr)

	if uncertainty:
		return _y,coef,_yerr,coeferr,r
	else:
		return _y,coef