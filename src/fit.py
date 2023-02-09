#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,copy,traceback
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,gradient,hessian,einsum,diag,partial,where
from src.utils import array,ones,zeros,rand,eye
from src.utils import norm,inv,lstsq
from src.utils import exp,log,abs,sqrt,nanmean,nanstd,nansqrt,product,is_naninf,allclose
from src.utils import nan,null,scalars,delim

from src.optimize import Optimizer,Metric,Objective,Callback,Covariance
from src.iterables import setter,getter

class cov(Covariance):pass

def fit(x,y,_x=None,_y=None,func=None,preprocess=None,postprocess=None,xerr=None,yerr=None,parameters=None,covariance=None,intercept=False,bounds=None,kwargs={}):
	'''
	Fit of data
	Args:
		x (array): Input data
		y (array): Output data
		_x (array): Input points to evaluate fit
		_y (array): Output points to evaluate fit
		func (callable,str,iterable[callable,str]): Functions to fit to data in preprocessed frame with signature func(parameters,x), or string for spline fit, ['linear','cubic']
		preprocess (callable,iterable[callable]): Function to preprocess data with signature x,y,parameters = preprocess(x,y,parameters) (with parameters argument/return optional)
		postprocess (callable,iterable[callable]): Function to postprocess data with signature x,y,parameters = postprocess(x,y,parameters) (with parameters argument/return optional)
		xerr (array): Input error
		yerr (array): Output error
		parameters (array,iterable[array]): Model parameters in preprocessed frame
		covariance (array,iterable[array]): Model parameters error in preprocessed frame
		intercept (bool,iterable[bool]): Include intercept in fit
		bounds (iterable[object]): piecewise domains
		kwargs (dict[str,object],iterable[dict[str,object]]): Additional keyword arguments for fitting
	Returns:
		_func (callable): Fit function in postprocessed frame with signature func(parameters,x)
		_y (array): Fit data at _x
		_parameters (array): Fit model parameters in postprocessed frame
		_yerr (array): Fit data error at _x
		_covariance (array): Fit model parameters error
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

	if parameters is None or isinstance(parameters,(array,*scalars)):
		parameters = [parameters for i in range(n)]

	if covariance is None or isinstance(covariance,(array,*scalars)):
		covariance = [covariance for i in range(n)]

	if intercept is None or isinstance(intercept,bool):
		intercept = [intercept for i in range(n)]

	if kwargs is None or isinstance(kwargs,dict):
		kwargs = [kwargs for i in range(n)]

	n = min(len(i) for i in [func,preprocess,postprocess,parameters,covariance,intercept,kwargs])

	funcs = func
	funcs = [lambda x,parameters,func=func,**kwargs: func(parameters,x,**kwargs) for func in funcs]
	funcs,conditions = piecewise(funcs,bounds)

	_func = [None for i in range(n)]
	_y = _y
	_parameters = parameters
	_yerr = zeros(_y.shape)
	_covariance = covariance
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
			parameters=parameters[i] if parameters is not None else parameters,
			covariance=covariance[i] if covariance is not None else covariance,
			intercept=intercept[i] if intercept is not None else intercept,
			**(kwargs[i] if kwargs is not None and kwargs[i] is not None else {})
			)

		_func[i] = returns[0]
		_y = _y.at[_condition[i]].set(returns[1]) 
		_parameters[i] = returns[2]
		_yerr = _yerr.at[_condition[i]].set(returns[3])
		_covariance[i] = returns[4]
		_other[i] = returns[5]


	if single:
		_func,_y,_parameters,_yerr,_covariance,_other = _func[0],_y,_parameters[0],_yerr,_covariance[0],_other[0]

	return _func,_y,_parameters,_yerr,_covariance,_other

def fitter(x,y,_x=None,_y=None,func=None,preprocess=None,postprocess=None,xerr=None,yerr=None,parameters=None,covariance=None,intercept=False,**kwargs):
	'''
	Fit of data
	Args:
		x (array): Input data
		y (array): Output data
		_x (array): Input points to evaluate fit
		_y (array): Output points to evaluate fit
		func (callable,str): Function to fit to data in preprocessed frame with signature func(parameters,x), or string for spline fit, ['linear','cubic']
		preprocess (callable): Function to preprocess data with signature x,y,parameters = preprocess(x,y,parameters) (with parameters argument/return optional)
		postprocess (callable): Function to postprocess data with signature x,y,parameters = postprocess(x,y,parameters) (with parameters argument/return optional)
		xerr (array): Input error
		yerr (array): Output error
		parameters (array): Model parameters in preprocessed frame
		covariance (array): Model parameters error in preprocessed frame
		intercept (bool): Include intercept in fit
		kwargs (dict[str,object]): Additional keyword arguments for fitting
	Returns:
		_func (callable): Fit function in postprocesed framed with signature func(parameters,x)
		_y (array): Fit data at _x
		_parameters (array): Fit model parameters
		_yerr (array): Fit data error at _x
		_covariance (array): Fit model parameters error
		_other (object): Other fit returns
	'''	
	defaults = {
		'metric':'lstsq',
		'shapes':kwargs.pop('shapes',(y.shape if y is not None else None,y.shape if y is not None else None,yerr.shape if yerr is not None else None)),
		}
	setter(kwargs,defaults,delimiter=delim,func=False)


	if _x is None:	
		_x = x
	if _y is None:
		_y = None
	if xerr is None:
		xerr = None
	if yerr is None:
		yerr = None
	if parameters is None:
		parameters = None
	if covariance is None:
		covariance = None

	_xerr = xerr
	_yerr = yerr
	_parameters = parameters
	_covariance = covariance

	transform,invtransform = transformation(x,y,parameters,preprocess=preprocess,postprocess=postprocess,**kwargs)
	gradtransform = gradient(invtransform,argnums=(0,1,2),mode='fwd')

	x,y = transform(x,y)
	_x,_y = transform(_x,_y)

	invgrad = gradtransform(x,y,parameters)
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
			
	_invgrad = gradtransform(x,y,parameters)
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

	if func is None or (isinstance(func,str) and func in ['lstsq','mse']):

		def func(parameters,x,*args,**kwargs):
			y = x.dot(parameters)
			return y

		if intercept:
			x = array([x,ones(x.size)]).T
		else:
			x = array([x]).T
		
		if intercept:
			_x = array([_x,ones(_x.size)]).T
		else:
			_x = array([_x]).T

		_parameters = lstsq(x,y)
		_y = func(x)
		
		if yerr.dim == 1:
			yerr = diag(yerr)
		
		_covariance = lstsq(x.T.dot(x),lstsq(x.T.dot(x),x.T.dot(yerr).dot(x)).T).T

	elif isinstance(func,str):

		kwargs.update({'kind':func})
		
		_func = interp(x,y,**kwargs)
		
		def func(parameters,x,*args,**kwargs):
			y = _func(x)
			return y
		
		_y = func(_parameters,_x)

		if yerr is not None:
			_yerr = 0
			_funcerr = interp(x,y+yerr,**kwargs)
			funcerr = lambda parameters,x,_func=_funcerr: _func(x)
			_yerr += abs(funcerr(_parameters,_x) - _y)

			_funcerr = interp(x,y-yerr,**kwargs)
			funcerr = lambda parameters,x,_func=_funcerr: _func(x)
			_yerr += abs(funcerr(_parameters,_x) - _y)

			_yerr /= 1

	elif callable(func) or not isinstance(func,(str,array)):
		# yerr = 2e-1*ones(y.size)

		kwargs.update({'parameters':parameters,'yerr':yerr,'xerr':xerr})

		if not callable(func):
			func,_parameters,_covariance = piecewise_fit(func,x,y,**kwargs)
		else:
			func,_parameters,_covariance = curve_fit(func,x,y,**kwargs)

		grad = gradient(func,argnums=0,mode='fwd')
		_y = func(_parameters,_x)
		_grad = grad(_parameters,_x)

		if _covariance is None:
			pass
		elif _covariance.ndim == 1:
			_yerr = abs(diag(_grad)*_covariance)
		elif _covariance.ndim == 2:
			_yerr = sqrt(diag(_grad.dot(_covariance).dot(_grad.T)))

	elif isinstance(func,array):
		z = func
		def func(parameters,x,*args,z=z,**kwargs):
			return z
	else:
		z = y
		def func(parameters,x,*args,z=z,**kwargs):
			return z			

	r = 1 - (((y - func(_parameters,x))**2).sum()/((y - y.mean())**2).sum())

	_other = {'r':r}

	invgrad = gradtransform(x,y,parameters)
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

	_invgrad = gradtransform(_x,_y,_parameters)
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

	if _covariance is not None:
		i = 2
		if _covariance.ndim == 1:
			_jac = diag(_invgrad[i][i])
			_covariance = abs(_jac*_covariance)
		else:
			_jac = _invgrad[i][i]
			_covariance = _jac.dot(_covariance).dot(_jac.T)

	x,y,parameters = invtransform(x,y,parameters)

	_x,_y,_parameters = invtransform(_x,_y,_parameters)

	def _func(parameters,x,*args,**kwargs):
		x,parameters = transform(x=x,parameters=parameters)
		y = func(parameters,x,*args,**kwargs)
		y = invtransform(y=y)
		return y


	model = jit(_func,x=x)
	metric = kwargs.pop('metric',None)
	shapes = kwargs.pop('shapes',None)
	label = y
	weights = yerr
	hyperparameters = kwargs
	system = {}
	kwargs = {}
	cov = Covariance(model,shapes=shapes,label=label,weights=weights,metric=metric,hyperparameters=hyperparameters,system=system,**kwargs)

	_covariance = cov(_parameters)

	return _func,_y,_parameters,_yerr,_covariance,_other


# @partial(jit,static_argnums=(0,))
def curve_fit(func,x,y,**kwargs):
	'''
	Compute fit between x and y
	Args:
		func (callable): Function to fit with signature func(parameters,x)
		x (array): Array of input data
		y (array): Array of output data
		kwargs (dict[str,object]): Additional keyword arguments for fitting		
	Returns:
		func (callable): Fit function with signature func(parameters,x)
		parameters (array): Fit parameters
		covariance (array,callable): Fit parameters error
	'''
	defaults = {
		'metric':'lstsq',
		'optimizer':'cg',
		'eps':{'value':1e-20},
		'value':{'value':1},
		'parameters':kwargs.pop('parameters',kwargs.pop('parameters0',kwargs.pop('coef',kwargs.pop('coef0',None)))),
		'covariance':kwargs.pop('covariance',kwargs.pop('yerr',kwargs.pop('sigma',None))),
		}
	setter(kwargs,defaults,delimiter=delim,func=False)

	def callback(parameters,track,optimizer,model,metric,func,grad):
		attr = 'value'
		status = (abs(optimizer.attributes[attr][-1]) > 
				(optimizer.hyperparameters['eps'][attr]*optimizer.hyperparameters['value'][attr]))
		# print(optimizer.attributes[attr][-1])
		return status

	function = func

	model = jit(func,x=x)
	parameters = kwargs.pop('parameters',None)
	covariance = kwargs.pop('covariance',None)
	metric = kwargs.pop('metric',None)
	
	defaults = {
		'iterations':1000 if covariance is not None and norm(covariance)/covariance.size < 1e-3 else 1000,
		'alpha':1e-20 if covariance is not None and norm(covariance)/covariance.size < 1e-3 else 1e-6,
		'beta':1e-20 if covariance is not None and norm(covariance)/covariance.size < 1e-3 else 1e-6,
		'uncertainty':parameters.size < 1000 if parameters is not None else True,
		'shapes':kwargs.pop('shapes',(y.shape if y is not None else None,y.shape if y is not None else None,covariance.shape if covariance is not None else None)),
		}
	setter(kwargs,defaults,delimiter=delim,func=False)

	uncertainty = kwargs.pop('uncertainty',True)
	shapes = kwargs.pop('shapes',None)
	label = y
	weights = covariance
	hyperparameters = kwargs
	system = {}
	kwargs = {}
	func = []
	callback = callback

	metric,cov = (
		Metric(metric,shapes=shapes,label=label,weights=weights,hyperparameters=hyperparameters,system=system,**kwargs),
		Covariance(model,shapes=shapes,label=label,weights=weights,metric=metric,hyperparameters=hyperparameters,system=system,**kwargs)
		)
	func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system,**kwargs)
	callback = Callback(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system,**kwargs)

	optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparameters,system=system,**kwargs)

	parameters = optimizer(parameters)

	if uncertainty:
		covariance = cov(parameters)
	else:
		covariance = None
	
	func = function

	return func,parameters,covariance

# @partial(jit,static_argnums=(0,))
def piecewise_fit(func,x,y,shape,**kwargs):
	'''
	Compute piecewise curve fit between x and y
	Args:
		func (callable): Function to fit with signature func(parameters,x)
		x (array): Array of input data
		y (array): Array of output data
		shape (iterable[int]): Piecewise parameters shape, including bounds		
		kwargs (dict[str,object]): Additional keyword arguments for fitting		
	Returns:
		func (callable): Piecewise function with signature func(parameters,x)
		parameters (array): Fit parameters
		covariance (array): Fit parameters error
	'''

	function,funcs,indices = piecewises(func,shape,include=True,**kwargs)

	n = len(funcs)

	parameterss = kwargs.pop('parameters',kwargs.pop('parameters0',None))
	yerrs = kwargs.pop('yerr',kwargs.pop('sigma',None))
	
	if parameterss is None:
		raise ValueError("parameters not in kwargs")

	bounds,parameterss = parameterss[indices[0]],[parameterss[index] for index in indices[1:]]

	_parameterss,_covariances = [*bounds],[*[None]*(n-1)]

	for i in range(n):
		
		func = funcs[i]
		parameters = parameterss[i]
		
		condition = (x<=bounds[i]) if i==0 else ((x>=bounds[i-1]) and (x<=bounds[i])) if i < (n-1) else (x>=bounds[i-1])
		
		_x = x[condition]
		_y = y[condition]
		_yerr = y[condition]

		kwargs['parameters'] = parameters
		kwargs['yerr'] = _yerr

		_func,_parameters,_covariance = curve_fit(func,_x,_y,**kwargs)

		_parameterss.append(_parameters)
		_covariances.append(_covariance)

	func = function
	_parameters = array(_parameterss)
	_covariance = array(_covariances)

	return func,_parameters,_covariance

def interp(x,y,**kwargs):
	'''
	Interpolate array at new points
	Args:
		x (array): Interpolation points
		y (array): Interpolation values
		kwargs (dict): Additional keyword arguments for interpolation
			kind (int): Order of interpolation
			smooth (int,float): Smoothness of fit
			der (int): order of derivative to estimate
	Returns:
		func (callable): Interpolation function with signature func(x,*args,**kwargs)
	'''	

	def _interpolate(x,y,**kwargs):
		kinds = {'linear':1,'quadratic':2,'cubic':3,'quartic':3,'quintic':5,None:3}
		k = kinds.get(kwargs.get('k',kwargs.get('kind')),kwargs.get('k',kwargs.get('kind')))
		s = kwargs.get('s',kwargs.get('smooth'))
		der = kwargs.get('der')
		if der:
			spline = osp.interpolate.splrep(x,y,k=k,s=s)
			def func(x):
				return osp.interpolate.splev(x,spline,der=der)
		else:
			func = osp.interpolate.UnivariateSpline(x,y,k=k,s=s)
			# func = osp.interpolate.interp1d(x,y,kind)
		return func

	return _interpolate(x,y,**kwargs)


def interpolate(x,y,_x,**kwargs):
	'''
	Interpolate array at new points
	Args:
		x (array): Interpolation points
		y (array): Interpolation values
		_x (array): New points
		kwargs (dict): Additional keyword arguments for interpolation
				kind (int): Order of interpolation
				smooth (int,float): Smoothness of fit
				der (int): order of derivative to estimate
	Returns:
		out (array): Interpolated values at new points
	'''		
	is1d = (y.ndim == 1)

	if is1d:
		y = [y]

	out = array([interp(x,u,**kwargs)(_x) for u in y])

	if is1d:
		out = out[0]

	return out



# @partial(jit,static_argnums=(0,))
def piecewises(func,shape,include=None,**kwargs):
	'''
	Compute piecewise curve from func
	Args:
		func (callable,iterable[callable]): Functions to fit with signature func(x,*args,**kwargs)
		shape (iterable[int]): Piecewise parameters shape
		include (bool): Include piecewise indices of coefficients
		kwargs (dict[str,object]): Additional keyword arguments for fitting		
	Returns:
		func (callable): Piecewise function with signature func(x,*args,**kwargs)
		funcs (iterable[callable]): Piecewise functions with signature func(x,*args,**kwargs)
		indices (array): indices of coefficients for piecewise domains
	'''

	if callable(func):
		funcs = [func]
	else:
		funcs = func

	n = len(funcs)
	indices = [slice(sum(shape[:i-1]),sum(shape[:i])) for i in range(1,n+2)]

	def func(x,parameters):

		bounds,parameterss = parameters[indices[0]],[parameters[index] for index in indices[1:]]
		n = len(funcs)

		func = [lambda x,parameters,i=i: funcs[i](x,parameters[i]) for i in range(n)]

		function,conditions = piecewise(func,bounds)

		return function(x,parameters)
	
	if include:
		return func,funcs,indices
	else:
		return func


def piecewise(func,bounds,**kwargs):
	'''
	Compute piecewise curve from func
	Args:
		func (iterable[callable]): Functions to fit with signature func(x,*args,**kwargs)
		bounds (iterable[object]): Bounds for piecewise domains
		kwargs (dict): Additional keyword arguments
	Returns:
		func (callable): Piecewise function with signature func(x,*args,**kwargs)
		conditions (callable): Conditions for piecewise domains with signature conditions(x) -> iterable[bool]
	'''

	if callable(func) or isinstance(func,str):
		func = [func]
	else:
		func = func

	n = len(func)

	if bounds is None and n>1:
		raise ValueError("TODO: Allow for bounds to be fit")
	elif isinstance(bounds,scalars):
		bounds = [True for i in range(n+1)]
	elif len(bounds) == (n-1):
		bounds = [*bounds,True,True]

	function = func

	def conditions(x,*args,**kwargs):
		n = len(bounds)-1
		if x.ndim > 1:
			axis,ord,r = 1,2,x.reshape(*x.shape[:1],-1)
			r = norm(r,axis=axis,ord=axis)
		else:
			r = x
		conditions = [(
			(bool(bounds[i-1])*ones(r.shape,dtype=bool) if (bounds[i-1] is None or isinstance(bounds[i-1],bool)) else r>=bounds[i-1]) & 
			(bool(bounds[i])*ones(r.shape,dtype=bool) if (bounds[i] is None or isinstance(bounds[i],bool)) else r<=bounds[i])
			)
			for i in range(n)]
		return conditions

	def func(x,*args,**kwargs):
		func = function
		return np.piecewise(x,conditions(x,*args,**kwargs),func,*args,**kwargs)

	return func,conditions


def extrema(x,y,_x=None,**kwargs):
	'''
	Get extreme points of array
	Args:
		x (array): Interpolation points
		y (array): Interpolation values
		_x (array): New points		
		kwargs (dict): Additional keyword arguments for interpolation
			kind (int): Order of interpolation
			smooth (int,float): Smoothness of fit
			der (int): order of derivative to estimate
	Returns:
		indices (array): Indices of extreme points
	'''	

	defaults = {'kind':1,'smooth':0,'der':2}

	if _x is None:
		_x = x

	kwargs.update({kwarg: kwargs.get(kwarg,defaults[kwarg]) for kwarg in defaults})

	indices = argsort(abs(interp(x,y,**kwargs)(_x)))

	return indices


def transformation(x,y,parameters=None,axis=None,mode='linear',process=True,standardize=True,preprocess=None,postprocess=None,**kwargs):
	'''
	Compute transformation of data
	Args:
		x (array): array to compute transformation
		y (array): array to compute transformation
		parameters (array): array to compute transformation (parameters of linear model y = parameters[0] + parameters[1]*x)
		axis (int): axis to compute over. Flattens array if None.
		mode (str): Method of transformation, allowed strings in ['linear']
		process (bool,str): Process data
		standardize (bool,str): Standardize data
		preprocess (callable): Function to preprocess data with signature x,y,parameters = preprocess(x,y,parameters) (with parameters argument/return optional)
		postprocess (callable): Function to postprocess data with signature x,y,parameters = postprocess(x,y,parameters) (with parameters argument/return optional)
		kwargs (dict): Additional keyword arguments for transformation
	Returns:
		transform (callable): transformation function
		invtransform (callable): inverse transformation function
	'''

	if mode is None or mode in ['linear']:
		func = preprocess
		if process and func is not None:
			_y = y
			x,y,parameters = func(x,y,parameters)
		if standardize:
			params = array([[x.min(),x.max()],[y.min(),y.max()]])
			params = array([[param[1],0] if param[0]==param[1] else param for param in params])
		else:
			params = None

		def wrapper(func,params,process=True,standardize=True,inv=False):
			
			if standardize:
				ax = (params[0][1]-params[0][0])
				bx = params[0][0]/(params[0][1]-params[0][0])
				ay = (params[1][1]-params[1][0])
				by = params[1][0]/(params[1][1]-params[1][0])

			@jit
			def transform(x=None,y=None,parameters=None):
				
				_x,_y,_parameters = x,y,parameters

				if not inv:

					if process and func is not None:
						_x,_y,_parameters = func(_x,_y,_parameters)

					if standardize:
						_x = (1/ax)*(_x) - bx if x is not None else None
						_y = (1/ay)*(_y) - by if y is not None else None

						_parameters = array([
							(1/ay)*(_parameters[0] + _parameters[1]*ax*bx) - by,
							(1/ay)*(_parameters[1])*(ax),
							]) if parameters is not None else None
				else:

					if standardize:
						_x = (ax)*(_x + bx) if x is not None else None
						_y = (ay)*(_y + by) if y is not None else None

						_parameters = array([
								ay*(_parameters[0] - _parameters[1]*bx + by),
								ay*(_parameters[1])*(1/ax)
								]) if parameters is not None else None

					if process and func is not None:
						_x,_y,_parameters = func(_x,_y,_parameters)
					

				if x is not None:
					if y is not None:
						if parameters is not None:
							returns = _x,_y,_parameters
						else:
							returns = _x,_y
					else:
						if parameters is not None:
							returns = _x,_parameters
						else:
							returns = _x
				else:
					if y is not None:
						if parameters is not None:
							returns = _y,_parameters
						else:
							returns = _y
					else:
						if parameters is not None:
							returns = _parameters
						else:
							returns = None
			
				return returns
			
			return transform


	transform = wrapper(preprocess,params=params,process=process,standardize=standardize,inv=False)
	invtransform = wrapper(postprocess,params=params,process=process,standardize=standardize,inv=True)

	return transform,invtransform

def uncertainty_propagation(x,y,xerr,yerr,operation):
	'''
	Calculate uncertainty of binary operations
	Args:
		x (array): x array
		y (array): y array
		xerr (array): error in x
		yerr (array): error in y
		operation (str): Binary operation between x and y, allowed strings in ['+','-','*','/','plus','minus','times','divides']
	Returns:
		out (array): Result of binary operation
		err (array): Error of binary operation
	'''

	operations = ['+','-','*','/','plus','minus','times','divides']
	assert operation in operations, "operation: %s not in operations %r"%(operation,operations)

	if operation in ['+','plus']:
		func = lambda x,y: x+y
		error = lambda x,y: sqrt((xerr*1)**2+(yerr*1)**2)
	elif operation in ['-','minus']:
		func = lambda x,y: x-y
		error = lambda x,y: sqrt((xerr*1)**2+(yerr*-1)**2)
	elif operation in ['*','times']:
		func = lambda x,y: x*y
		error = lambda x,y: sqrt((xerr*y)**2+(yerr*x)**2)
	elif operation in ['/','divides']:
		func = lambda x,y: x+y
		error = lambda x,y: sqrt((xerr/y)**2+(yerr*x/-y**2)**2)

	out,err = func(x,y),error(x,y)

	return out,err
