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
from src.utils import array,zeros,ones,eye
from src.utils import lstsq,curve_fit,piecewise_fit,piecewise,interp,standardize,sort,norm
from src.utils import exp,log,abs,sqrt,nanmean,nanstd,nansqrt,product,is_naninf,allclose
from src.utils import nan,null,scalars

def fit(x,y,_x=None,_y=None,func=None,preprocess=None,postprocess=None,xerr=None,yerr=None,coef=None,coeferr=None,intercept=False,bounds=None,kwargs={}):
	'''
	Fit of data
	Args:
		x (array): Input data
		y (array): Output data
		_x (array): Input points to evaluate fit
		_y (array): Output points to evaluate fit
		func (callable,str,iterable[callable,str]): Functions to fit to data in preprocessed frame with signature func(coef,x), or string for spline fit, ['linear','cubic']
		preprocess (callable,iterable[callable]): Function to preprocess data with signature x,y,coef = preprocess(x,y,coef) (with coef argument/return optional)
		postprocess (callable,iterable[callable]): Function to postprocess data with signature x,y,coef = postprocess(x,y,coef) (with coef argument/return optional)
		xerr (array): Input error
		yerr (array): Output error
		coef (array,iterable[array]): Model coefficients in preprocessed frame
		coeferr (array,iterable[array]): Model coefficients error in preprocessed frame
		intercept (bool,iterable[bool]): Include intercept in fit
		bounds (iterable[object]): piecewise domains
		kwargs (dict[str,object],iterable[dict[str,object]]): Additional keyword arguments for fitting
	Returns:
		_func (callable): Fit function in postprocessed frame with signature func(coef,x)
		_y (array): Fit data at _x
		_coef (array): Fit model parameters in postprocessed frame
		_yerr (array): Fit data error at _x
		_coeferr (array): Fit model parameters error
		_r (float): Fit coefficient
		_other (dict[str,object]): Other fit returns		
	'''	

	single = callable(func) or isinstance(func,str)

	if single:
		func = [func]
	else:
		func = func

	n = len(func)

	if preprocess is None or callable(preprocess):
		preprocess = [preprocess for i in range(n)]

	if postprocess is None or callable(postprocess):
		postprocess = [postprocess for i in range(n)]

	if coef is None or isinstance(coef,(array,*scalars)):
		coef = [coef for i in range(n)]

	if coeferr is None or isinstance(coeferr,(array,*scalars)):
		coeferr = [coeferr for i in range(n)]

	if intercept is None or isinstance(intercept,bool):
		intercept = [intercept for i in range(n)]

	if kwargs is None or isinstance(kwargs,dict):
		kwargs = [kwargs for i in range(n)]

	n = min(len(i) for i in [func,preprocess,postprocess,coef,coeferr,intercept,kwargs])

	funcs = func
	funcs = [lambda x,coef,func=func,**kwargs: func(coef,x,**kwargs) for func in funcs]
	funcs,conditions = piecewise(funcs,bounds)

	_func = [None for i in range(n)]
	_y = _y
	_coef = coef
	_yerr = zeros(_y.shape)
	_coeferr = coeferr
	_r = [None for i in range(n)]
	_other = [None for i in range(n)]

	for i in range(n):
		
		condition = conditions(x)
		_condition = conditions(_x)


		returns = fitter(
			x=x[condition[i]] if x is not None else x,
			y=y[condition[i]] if y is not None else y,
			_x=_x[_condition[i]] if _x is not None else _x,
			_y=_y[_condition[i]] if _y is not None else _y,
			func=func[i] if func is not None else func,
			preprocess=preprocess[i] if preprocess is not None else preprocess,
			postprocess=postprocess[i] if postprocess is not None else postprocess,
			xerr=xerr[condition[i]] if xerr is not None else xerr,
			yerr=yerr[condition[i]] if yerr is not None else yerr,
			coef=coef[i] if coef is not None else coef,
			coeferr=coeferr[i] if coeferr is not None else coeferr,
			intercept=intercept[i] if intercept is not None else intercept,
			**(kwargs[i] if kwargs is not None and kwargs[i] is not None else {})
			)

		_func[i] = returns[0]
		_y = _y.at[_condition[i]].set(returns[1]) 
		_coef[i] = returns[2]
		_yerr = _yerr.at[_condition[i]].set(returns[3])
		_coeferr[i] = returns[4]
		_r[i] = returns[5]
		_other[i] = returns[6]


	if single:
		_func,_y,_coef,_yerr,_coeferr,_r,_other = _func[0],_y,_coef[0],_yerr,_coeferr[0],_r[0],_other[0]

	return _func,_y,_coef,_yerr,_coeferr,_r,_other

def fitter(x,y,_x=None,_y=None,func=None,preprocess=None,postprocess=None,xerr=None,yerr=None,coef=None,coeferr=None,intercept=False,**kwargs):
	'''
	Fit of data
	Args:
		x (array): Input data
		y (array): Output data
		_x (array): Input points to evaluate fit
		_y (array): Output points to evaluate fit
		func (callable,str): Function to fit to data in preprocessed frame with signature func(coef,x), or string for spline fit, ['linear','cubic']
		preprocess (callable): Function to preprocess data with signature x,y,coef = preprocess(x,y,coef) (with coef argument/return optional)
		postprocess (callable): Function to postprocess data with signature x,y,coef = postprocess(x,y,coef) (with coef argument/return optional)
		xerr (array): Input error
		yerr (array): Output error
		coef (array): Model coefficients in preprocessed frame
		coeferr (array): Model coefficients error in preprocessed frame
		intercept (bool): Include intercept in fit
		kwargs (dict[str,object]): Additional keyword arguments for fitting
	Returns:
		_func (callable): Fit function in postprocesed framed with signature func(coef,x)
		_y (array): Fit data at _x
		_coef (array): Fit model parameters
		_yerr (array): Fit data error at _x
		_coeferr (array): Fit model parameters error
		_r (float): Fit coefficient
		_other (object): Other fit returns
	'''	

	if _x is None:	
		_x = x
	if _y is None:
		_y = None
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

	x,y = transform(x,y)
	_x,_y = transform(_x,_y)

	invgrad = gradtransform(x,y,coef)
	if xerr is not None:
		i = 0
		if xerr.ndim == 1:
			jac = diag(invgrad[i][i])
			xerr = abs((1/jac)*xerr)
		else:
			jac = invgrad[i][i]
			xerr = lstsq(jac.T,lstsq(jax,xerr).T).T

	if yerr is not None:
		i = 1
		if yerr.ndim == 1:
			jac = diag(invgrad[i][i])
			yerr = abs((1/jac)*yerr)
		else:
			jac = invgrad[i][i]
			yerr = lstsq(jac.T,lstsq(jax,yerr).T).T
			
	_invgrad = gradtransform(x,y,coef)
	if _xerr is not None:
		i = 0
		if _xerr.ndim == 1:
			_jac = diag(_invgrad[i][i])
			_xerr = abs((1/_jac)*_xerr)
		else:
			_jac = invgrad[i][i]
			_xerr = lstsq(_jac.T,lstsq(_jax,_xerr).T).T

	if _yerr is not None:
		i = 1
		if _yerr.ndim == 1:
			_jac = diag(_invgrad[i][i])
			_yerr = abs((1/_jac)*_yerr)
		else:
			_jac = invgrad[i][i]
			_yerr = lstsq(_jac.T,lstsq(_jax,_yerr).T).T

	if func is None:

		def func(coef,x,*args,**kwargs):
			y = x.dot(coef)
			return y

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
		
		_coeferr = lstsq(x.T.dot(x),lstsq(x.T.dot(x),x.T.dot(yerr).dot(x)).T).T

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
		
		def func(coef,x,*args,**kwargs):
			y = _func(x)
			return y
		
		_y = func(_coef,_x)

		if yerr is not None:
			_yerr = 0
			_funcerr = interp(x,y+yerr,**kwargs)
			funcerr = lambda coef,x,_func=_funcerr: _func(x)
			_yerr += abs(funcerr(_coef,_x) - _y)

			_funcerr = interp(x,y-yerr,**kwargs)
			funcerr = lambda coef,x,_func=_funcerr: _func(x)
			_yerr += abs(funcerr(_coef,_x) - _y)

			_yerr /= 1

	else:
		def func(coef,x,*args,y=y,**kwargs):
			return y

	_r = 1 - (((y - func(_coef,x))**2).sum()/((y - y.mean())**2).sum())

	_other = {'_coef':_coef}

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
			jac = invgrad[i][i]
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

	_invgrad = gradtransform(_x,_y,_coef)
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

	def _func(coef,x,*args,func=func,transform=transform,invtransform=invtransform,**kwargs):
		x,coef = transform(x=x,coef=coef)
		y = func(coef,x,*args,**kwargs)
		y = invtransform(y=y)
		return y


	return _func,_y,_coef,_yerr,_coeferr,_r,_other