#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy,datetime
from functools import partial

envs = {
	'JAX_PLATFORM_NAME':'cpu',
	'TF_CPP_MIN_LOG_LEVEL':5,
}
for var in envs:
	os.environ[var] = str(envs[var])


import jax
import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)

configs = {
	'jax_disable_jit':False,
	'jax_platforms':'cpu',
	'jax_enable_x64': True
	}
for name in configs:
	jax.config.update(name,configs[name])

# Logging
import logging
logger = logging.getLogger(__name__)

# Import user modules
from src.utils import jit,value_and_gradient,gradient,hessian
from src.utils import is_naninf,product,sqrt

from src.utils import normed,inner_abs2,inner_real,inner_imag
from src.utils import gradient_normed,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
from src.utils import normed_einsum,inner_abs2_einsum,inner_real_einsum,inner_imag_einsum
from src.utils import gradient_normed_einsum,gradient_inner_abs2_einsum,gradient_inner_real_einsum,gradient_inner_imag_einsum

from src.utils import itg,dbl,flt,delim,nan

from src.line_search import line_search,armijo

from src.io import dump,load,join,split,copy,rmdir,exists



def value_and_grad(func,grad=None):
	def _value_and_grad(func,grad):
		def _func_and_grad(parameters):
			return func(parameters),grad(parameters)
		return _func_and_grad

	if grad is None:
		grad = gradient(func)
		func_and_grad = value_and_gradient(func)
	else:
		func_and_grad = _value_and_grad(func,grad)

	return func_and_grad,func,grad


class LineSearchBase(object):
	def __init__(self,func,grad,hyperparameters):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
		'''
		defaults = {}
		returns = ['alpha']

		self.func = func
		self.grad = grad
		self.hyperparameters = hyperparameters
		self.defaults = defaults
		self.returns = returns

		return

	def __call__(self,parameters,alpha,value,gradient,search):
		'''
		Perform line search
		Args:
			parameters (array): Objective parameters
			alpha (iterable): Previous alpha
			value (iterable): Previous objective values
			gradient (array): Previous objective gradients
			search (array): Previous objective search directions
		Returns:
			returns (dict): Dictionary of returned values of line search
		'''
		returns = (alpha[-1],)

		returns = self.__callback__(returns,parameters,alpha,value,gradient,search)

		return returns

	def __callback__(self,returns,parameters,alpha,value,gradient,search):
		'''
		Check return values of line search
		Args:
			returns (iterable): Iterable of returned values of line search
			parameters (array): Objective parameters
			alpha (iterable): Previous alpha
			value (iterable): Previous objective values
			gradient (array): Previous objective gradients
			search (array): Previous objective search directions
		Returns:
			returns (dict): Dictionary of returned values of line search
		'''
		
		returns = dict(zip(self.returns,returns))

		attr = 'alpha'
		if returns[attr] is None or (returns[attr] < self.hyperparameters['eps'][attr]):
			if len(alpha) > 1:
				returns[attr] = alpha[-1]*gradient[-1].dot(search[-1])/gradient[-2].dot(search[-2])
			else:
				returns[attr] = alpha[-1]
		
		return returns

class LineSearch(LineSearchBase):
	'''
	Line Search class
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		hyperparameters (dict): line search hyperparameters
	'''
	def __new__(cls,func,grad,hyperparameters={}):
	
		defaults = {'line_search':None}
		hyperparameters.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		line_searches = {'line_search':Line_Search,'armijo':Armijo,None:Null_Search}

		line_search = hyperparameters['line_search']
		
		self = line_searches.get(line_search,line_searches[None])(func,grad,hyperparameters)

		return self


class Line_Search(LineSearchBase):
	def __init__(self,func,grad,hyperparameters):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
		'''
		defaults = {'c1':0.0001,'c2':0.9,'maxiter':10,'old_old_fval':None}
		returns = ['alpha','nfunc','ngrad','value','_value','slope']
		hyperparameters.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		super().__init__(func,grad,hyperparameters)

		self.defaults = defaults
		self.returns = returns

		return

	def __call__(self,parameters,alpha,value,gradient,search):
		'''
		Perform line search
		Args:
			parameters (array): Objective parameters
			alpha (iterable): Previous alpha
			value (iterable): Previous objective values
			gradient (array): Previous objective gradients
			search (array): Previous objective search directions
		Returns:
			returns (dict): Dictionary of returned values of line search
		'''
		self.defaults['old_old_fval'] = value[-2] if len(value)>1 else None
		
		returns = line_search(self.func,self.grad,
			parameters,search[-1],gradient[-1],value[-1],
			**self.defaults)
		
		returns = self.__callback__(returns,parameters,alpha,value,gradient,search)

		return returns



class Armijo(LineSearchBase):
	def __init__(self,func,grad,hyperparameters):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
		'''
		defaults = {'c1':0.0001,'alpha0':1e-4}
		returns = ['alpha','nfunc','value']
		hyperparameters.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		super().__init__(func,grad,hyperparameters)

		self.defaults = defaults
		self.returns = returns

		return

	def __call__(self,parameters,alpha,value,gradient,search):
		'''
		Perform line search
		Args:
			parameters (array): Objective parameters
			alpha (iterable): Previous alpha
			value (iterable): Previous objective values
			gradient (array): Previous objective gradients
			search (array): Previous objective search directions
		Returns:
			returns (dict): Dictionary of returned values of line search
		'''

		returns = armijo(self.func,self.grad,
			parameters,search[-1],gradient[-1],value[-1],
			**self.defaults)

		returns = self.__callback__(returns,parameters,alpha,value,gradient,search)
		
		return returns


class Null_Search(LineSearchBase):
	def __init__(self,func,grad,hyperparameters):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
		'''

		super().__init__(func,grad,hyperparameters)

		return


class Objective(object):		
	def __init__(self,func,grad=None,hess=None,callback=None,model=None,hyperparameters={}):
		'''	
		Objective class for function
		Args:
			func (callable,iterable[callable]): Objective function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function  with signature grad(parameters), or iterable of functions to sum
			hess (callable,iterable[callable]): Hessian of function, with signature hess(parameters), or iterable of functions to sum
			callback (callable): Gradient of function  with signature callback(parameters,track,attributes,func,grad,hess,funcs,grads,hesss,model,hyperparameters)
			model (object): Model instance
			hyperparameters (dict): Objective hyperparameters
		'''

		if not callable(func):
			funcs = func
			func = partial((lambda *args,funcs=None,**kwargs: sum(func(*args,**kwargs) for func in funcs)),funcs=funcs)
		else:
			funcs = [func]
			func = func

		if grad is None:
			grads = [None for func in funcs]
			grad = None
		elif not callable(grad):
			grads = grad
			grad = partial((lambda *args,funcs=None,**kwargs: sum(func(*args,**kwargs) for func in funcs)),funcs=grads)
		else:
			grads = [None for func in funcs]
			grad = grad

		if hess is None:
			hesss = [None for func in funcs]
			hess = None
		elif not callable(hess):
			hesss = hess
			hess = partial((lambda *args,funcs=None,**kwargs: sum(func(*args,**kwargs) for func in funcs)),funcs=hesss)
		else:
			hesss = [None for func in funcs]
			hess = hess

		values_and_grads = [value_and_grad(func,grad) for func,grad in zip(funcs,grads)]
		self.values_and_grads,self.funcs,self.grads = [[val_and_grad[i] for val_and_grad in values_and_grads] 
				for i in range(min(len(val_and_grad) for val_and_grad in values_and_grads))]
		self.hesss = [hessian(func) if hess is None else hess for func,hess in zip(funcs,hesss)]

		self.value_and_grad,self.func,self.grad = value_and_grad(func,grad)
		self.hess = hessian(func) if hess is None else hess

		self.hyperparameters = hyperparameters

		if callback is None:
			def callback(parameters,track,attributes,func,grad,hyperparameters):
				status = False
				return False

		self.callback = partial(callback,
				func=self.func,grad=self.grad,
				funcs=self.funcs,grads=self.grads,hess=self.hess,
				model=model,
				hyperparameters=self.hyperparameters
				)

		return

	# @partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Function call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of objective function
		'''
		return self.func(parameters)

	# @partial(jit,static_argnums=(0,))
	def __grad__(self,parameters):
		'''
		Gradient call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of gradient function
		'''
		return self.grad(parameters)	

	# @partial(jit,static_argnums=(0,))
	def __hessian__(self,parameters):
		''' 
		Hessian call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of hessian function
		'''	
		return self.hess(parameters)	

	# @partial(jit,static_argnums=(0,))
	def __callback__(self,parameters,track,attributes,hyperparameters):
		'''
		Callback call
		Args:
			parameters (array): parameters
			track (dict): callback track			
			attributes (dict): callback attributes			
			hyperparameters (dict): callback hyperparameters			
		Returns:
			out (object): Return of objective function
		'''
		return self.callback(parameters,track,attributes,hyperparameters)		


class Metric(object):
	'''
	Metric class for distance between Operators
	Args:
		metric (str,Metric): Type of metric
		shapes (iterable[tuple[int]]): Shapes of Operators
		optimize (bool,str,iterable): Contraction type		
	'''
	def __init__(self,metric,shapes,optimize=None):

		self.metric = metric
		self.shapes = shapes
		self.optimize = optimize
		self.default = None

		self.__setup__()
		
		return

	def __setup__(self):
		'''
		Setup metric attributes metric,string
		'''
		if isinstance(self.metric,Metric):
			self.metric = self.metric.metric
		if self.metric is None:
			self.metric = self.default
		self.__string__()
		self.__size__()

		self.get_metric()

		return

	def __string__(self):
		self.string = str(self.metric)
		return

	def __size__(self):
		self.size = sum(int(product(shape)**(1/len(shape))) for shape in self.shapes)//len(self.shapes)
		return 

	@partial(jit,static_argnums=(0,))
	def __call__(self,a,b):
		return self.func(a,b)

	@partial(jit,static_argnums=(0,))
	def __grad__(self,a,b,da):
		return self.grad(a,b,da)		

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return str(self.string)

	def get_metric(self):

		if callable(self.metric):
			func = jit(self.metric)
			grad = jit(gradient(self.metric))

		elif self.metric is None:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(normed_einsum(*shapes,optimize=optimize))
			# _func = normed

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_normed_einsum(*shapes,optimize=optimize))
			# _grad = gradient_normed

			@jit
			def func(a,b):
				return _func(a,b)/2/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return _grad(a,b,da)/2/sqrt(a.shape[0]*b.shape[0])	

		elif self.metric in ['norm','normed']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(normed_einsum(*shapes,optimize=optimize))
			# _func = normed

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_normed_einsum(*shapes,optimize=optimize))
			# _grad = gradient_normed

			@jit
			def func(a,b):
				return _func(a,b)/2/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return _grad(a,b,da)/2/sqrt(a.shape[0]*b.shape[0])

		elif self.metric in ['infidelity']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(inner_real_einsum(*shapes,optimize=optimize))
			# _func = inner_real

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_inner_real_einsum(*shapes,optimize=optimize))
			# _grad = gradient_inner_real

			@jit
			def func(a,b):
				return 1-_func(a,b)/(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)/(a.shape[0]*b.shape[0])

		elif self.metric in ['infidelity.abs']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(inner_abs2_einsum(*shapes,optimize=optimize))
			# _func = inner_abs2

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_inner_abs2_einsum(*shapes,optimize=optimize))
			# _grad = gradient_inner_abs2

			@jit
			def func(a,b):
				return 1-_func(a,b)/(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)/(a.shape[0]*b.shape[0])

		elif self.metric in ['infidelity.norm']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(inner_real_einsum(*shapes,optimize=optimize))
			# _func = inner_real

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_inner_real_einsum(*shapes,optimize=optimize))
			# _grad = gradient_inner_real

			@jit
			def func(a,b):
				return 1-_func(a,b)
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)

		elif self.metric in ['infidelity.real']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(inner_real_einsum(*shapes,optimize=optimize))
			# _func = inner_real

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_inner_real_einsum(*shapes,optimize=optimize))
			# _grad = gradient_inner_real

			@jit
			def func(a,b):
				return 1-_func(a,b)/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)/sqrt(a.shape[0]*b.shape[0])

		elif self.metric in ['infidelity.imag']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(inner_imag_einsum(*shapes,optimize=optimize))
			# _func = inner_imag

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_inner_imag_einsum(*shapes,optimize=optimize))
			# _grad = gradient_inner_imag

			@jit
			def func(a,b):
				return 1-_func(a,b)/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)/sqrt(a.shape[0]*b.shape[0])	

		elif self.metric in ['infidelity.real.imag']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func_real = jit(inner_real_einsum(*shapes,optimize=optimize))
			# _func_real = inner_real

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad_real = jit(gradient_inner_real_einsum(*shapes,optimize=optimize))
			# _grad_real = gradient_inner_real

			shapes = (*self.shapes,)
			optimize = self.optimize
			_func_imag = jit(inner_imag_einsum(*shapes,optimize=optimize))
			# _func_imag = inner_imag

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad_imag = jit(gradient_inner_imag_einsum(*shapes,optimize=optimize))
			# _grad_imag = gradient_inner_imag

			@jit
			def func(a,b):
				return 1-(_func_real(a,b)+_func_imag(a,b))/2/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return -(_grad_real(a,b,da)+_grad_imag(a,b,da))/2/sqrt(a.shape[0]*b.shape[0])

		else:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(normed_einsum(*shapes,optimize=optimize))
			# _func = normed

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_normed_einsum(*shapes,optimize=optimize))
			# _grad = gradient_normed

			@jit
			def func(a,b):
				return _func(a,b)/2/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return _grad(a,b,da)/2/sqrt(a.shape[0]*b.shape[0])

		self.func = func

		self.grad = grad

		return



class OptimizerBase(object):
	'''
	Base Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,attributes,hyperparameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		updates = {
			'verbose': {
				'notset':0,'debug':10,'info':20,'warning':30,'error':40,'critical':50,
				'Notset':0,'Debug':10,'Info':20,'Warning':30,'Error':40,'Critical':50,
				'NOTSET':0,'DEBUG':10,'INFO':20,'WARNING':30,'ERROR':40,'CRITICAL':50,
				10:10,20:20,30:30,40:40,50:50,
				2:20,3:30,4:40,5:50,
				True:20,False:0,None:0,
				}
			}

		defaults = {
			'optimizer':None,
			'line_search':'line_search',
			'eps':{'value':1e-4,'grad':1e-12,'alpha':1e-12,'beta':1e3},
			'alpha':0,
			'status':1,
			'timestamp':None,
			'path':None,
			'verbose':False,
			'modulo':{'log':None,'attributes':None,'callback':None,'restart':None,'dump':None},
			'length':{'log':None,'attributes':10,'callback':None,'restart':None,'dump':None},
			'attributes':{'iteration':[],'parameters':[],'value':[],'grad':[],'search':[],'alpha':[]},	
			'track':{'iteration':[],'parameters':[],'value':[],'grad':[],'search':[],'alpha':[]},		
		}

		hyperparameters.update({attr: defaults[attr] for attr in defaults if attr not in hyperparameters})
		hyperparameters.update({attr: updates.get(attr,{}).get(hyperparameters[attr],hyperparameters[attr]) 
			if attr in updates else hyperparameters[attr] for attr in hyperparameters})

		self.hyperparameters = hyperparameters

		self.value_and_grad,self.func,self.grad = value_and_grad(func,grad)

		self.line_search = LineSearch(self.func,self.grad,self.hyperparameters)

		if callback is None:
			def callback(parameters,track,attributes,hyperparameters):
				status = True
				return status
		self.callback = callback


		self.size = 0
		self.iteration = -1
		self.parameters = None
		self.optimizer = hyperparameters['optimizer']		
		self.modulo = hyperparameters['modulo']
		self.length = hyperparameters['length']
		self.attributes = hyperparameters['attributes']
		self.track = hyperparameters['track']
		self.iterations = range(int(hyperparameters['iterations']))
		self.sizes = hyperparameters['length'].get('attributes')
		self.status = hyperparameters['status']
		self.eps = hyperparameters['eps']
		self.timestamp = hyperparameters['timestamp']
		self.path = hyperparameters['path']
		self.verbose = hyperparameters['verbose']

		self.load()

		return

	def __call__(self,parameters):
		'''
		Iterate optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
		Returns:
			parameters (object): optimizer parameters
		'''

		state = self.opt_init(parameters)

		for iteration in self.iterations:

			iteration += 1
			
			state = self.opt_update(iteration,state)

			self.dump(iteration,state)

			if not self.status:
				break

		parameters = self.get_params(state)

		self.parameters = parameters

		return parameters

	def opt_init(self,parameters):
		'''
		Initialize optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
		Returns:
			state (object): optimizer state
		'''
		state = parameters
		return state

	def opt_update(self,iteration,state):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			state (object): optimizer state
		'''

		if self.size == 0:

			value,grad,parameters = self.opt_step(iteration-1,state)

			alpha = self.hyperparameters['alpha']
			search = -grad

			self.attributes['alpha'].append(alpha)
			self.attributes['search'].append(search)

			state = self.opt_init(parameters)
			parameters = self.get_params(state)
			track = self.track		
			attributes = self.attributes
			hyperparameters = self.hyperparameters			
			self.status = self.callback(parameters,track,attributes,hyperparameters)

		value,grad,parameters = self.opt_step(iteration,state)

		alpha = self.attributes['alpha'][-1]
		search = -grad

		parameters = parameters + alpha*search

		self.attributes['alpha'].append(alpha)
		self.attributes['search'].append(search)

		state = self.opt_init(parameters)
		parameters = self.get_params(state)
		track = self.track		
		attributes = self.attributes
		hyperparameters = self.hyperparameters
		self.status = self.callback(parameters,track,attributes,hyperparameters)

		return state

	def get_params(self,state):
		'''
		Get optimizer parameters with optimizer state
		Args:
			state (object): optimizer state
		Returns:
			parameters (object): optimizer parameters
		'''
		parameters = state
		return parameters

	def opt_step(self,iteration,state):
		'''
		Iterate optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			value (object): optimizer value
			grad (object): optimizer grad
		'''

		parameters = self.get_params(state)
		value,grad = self.value_and_grad(parameters)

		if (self.sizes is not None) and (self.size > 0) and (self.size >= self.sizes):
			for attr in self.attributes:
				self.attributes[attr].pop(0)

		self.iteration = iteration
		
		self.attributes['iteration'].append(iteration)
		self.attributes['parameters'].append(parameters)
		self.attributes['value'].append(value)
		self.attributes['grad'].append(grad)

		self.size += 1
		self.iteration += 1

		return value,grad,parameters


	def dump(self,iteration,state):
		'''
		Dump data
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		'''

		if self.path is None:
			return

		def parse(iteration,labels):
			string = delim.join([*(str(i) for i in labels),*([str(i) for i in [iteration]])])
			return string

		start = (self.size==1) and (iteration<self.iterations.stop)
		done = (self.size>0) and (iteration == self.iterations.stop)
		other = (self.size == 0) or (self.modulo['dump'] is None) or (iteration%self.modulo['dump'] == 0)

		if not ((not self.status) or done or start or other):
			return

		path = self.path
		track = self.track
		attributes = self.attributes

		labels = [self.timestamp]
		size = min((len(track[attr]) for attr in track if len(track[attr])>0),default=0)
		iterations = range(size)
		default = nan

		data = {}
		for iteration in iterations:
			
			string = parse(iteration,labels)
			
			value = {}
			value.update({attr: attributes[attr][-1] if ((not self.status) or done or start) else default for attr in attributes})
			value.update({attr: track[attr][iteration] for attr in track})

			data[string] = value

		dump(data,path)

		return

	def load(self):
		'''
		Load data
		Returns:
			iteration (int): optimizer iteration
			state (object): optimizer state
		'''

		if self.path is None:
			return

		def parse(string):
			iteration = int(string.split(delim)[-1])
			labels = string.split(delim)[:-1]
			return iteration,labels

		path = self.path
		default = {}
		data = load(path,default=default)

		iteration = self.iteration
		track = {}
		for string in data:
			iteration,labels = parse(string)

			for attr in data[string]:
				if attr not in track:
					track[attr] = []
				value = data[string][attr]
				track[attr].append(value)

		attr = 'iteration'	
		if attr in track:
			self.iteration = track[attr][-1]

		attr = 'parameters'		
		if attr in track:
			self.parameters = track[attr][-1]

		for attr in self.track:
			if attr in track:
				self.track[attr].extend(track[attr])

		for attr in self.attributes:
			if attr in track:
				self.attributes[attr].extend(track[attr])			

		self.iteration = max(self.iteration-1,-1)
		self.size = min((len(self.track[attr]) for attr in self.track),default=self.size)
		self.iterations = range(
			self.iterations.start+self.iteration+1,
			self.iterations.stop+self.iteration+1,
			self.iterations.step)


		iteration = self.iteration
		state = self.opt_init(self.parameters)

		return iteration,state


class Optimizer(OptimizerBase):
	'''
	Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,attributes,hyperparameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __new__(cls,func,grad=None,callback=None,hyperparameters={}):
	
		defaults = {'optimizer':None}
		hyperparameters.update({attr: defaults[attr] for attr in defaults if attr not in hyperparameters})

		optimizers = {'adam':Adam,'cg':ConjugateGradient,'gd':GradientDescent,None:GradientDescent}

		optimizer = hyperparameters['optimizer']		
		
		self = optimizers.get(optimizer,optimizers[None])(func,grad,callback,hyperparameters)

		return self
	

class GradientDescent(OptimizerBase):
	'''
	Gradient Descent Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,attributes,hyperparameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		super().__init__(func,grad,callback,hyperparameters)

		defaults = {}
		for attr in defaults:
			self.hyperparameters[attr] = self.hyperparameters.get(attr,defaults[attr])
		
		defaults = {}
		for attr in defaults:
			self.attributes[attr] = self.attributes.get(attr,defaults[attr])		

		null = ['beta']
		for attr in null:
			self.hyperparameters.pop(attr,None)

		null = ['beta']
		for attr in null:
			self.attributes.pop(attr,None)



		return

	def opt_update(self,iteration,state):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			state (object): optimizer state
		'''

		if self.size == 0:

			value,grad,parameters = self.opt_step(iteration-1,state)

			alpha = self.hyperparameters['alpha']
			search = -grad

			self.attributes['alpha'].append(alpha)
			self.attributes['search'].append(search)

			state = self.opt_init(parameters)
			parameters = self.get_params(state)
			track = self.track		
			attributes = self.attributes
			hyperparameters = self.hyperparameters			
			self.status = self.callback(parameters,track,attributes,hyperparameters)

		value,grad,parameters = self.opt_step(iteration,state)

		alpha = self.attributes['alpha'][-1]
		search = -grad

		parameters = parameters + alpha*search

		self.attributes['alpha'].append(alpha)
		self.attributes['search'].append(search)

		state = self.opt_init(parameters)
		parameters = self.get_params(state)
		track = self.track
		attributes = self.attributes
		hyperparameters = self.hyperparameters
		self.status = self.callback(parameters,track,attributes,hyperparameters)

		print('Step',self.iteration,self.size)
		for attr in self.track:
			print('Track',attr,len(self.track[attr]),self.track[attr][-1])
		for attr in self.attributes:
			print(attr)
			print('Attr',attr,len(self.attributes[attr]),self.attributes[attr][-1])
		print()

		return state


class ConjugateGradient(OptimizerBase):
	'''
	Conjugate Gradient Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,attributes,hyperparameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		super().__init__(func,grad,callback,hyperparameters)

		defaults = {'beta':0}
		for attr in defaults:
			self.hyperparameters[attr] = self.hyperparameters.get(attr,defaults[attr])
		
		defaults = {'beta':[]}
		for attr in defaults:
			self.attributes[attr] = self.attributes.get(attr,defaults[attr])		

		null = []
		for attr in null:
			self.hyperparameters.pop(attr,None)

		null = []
		for attr in null:
			self.attributes.pop(attr,None)

		return

	def opt_update(self,iteration,state):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			state (object): optimizer state
		'''

		if self.size == 0:

			value,grad,parameters = self.opt_step(iteration-1,state)

			alpha = self.hyperparameters['alpha']
			beta = self.hyperparameters['beta']
			search = -grad

			self.attributes['alpha'].append(alpha)
			self.attributes['beta'].append(beta)
			self.attributes['search'].append(search)

			state = self.opt_init(parameters)
			parameters = self.get_params(state)
			track = self.track		
			attributes = self.attributes
			hyperparameters = self.hyperparameters
			self.status = self.callback(parameters,track,attributes,hyperparameters)

		parameters = self.get_params(state)

		returns = self.line_search(
			parameters,
			self.attributes['alpha'],
			self.attributes['value'],
			self.attributes['grad'],
			self.attributes['search'])

		alpha = returns['alpha']
		search = self.attributes['search'][-1]
		grad = self.attributes['grad'][-1]

		parameters = parameters + alpha*search

		state = self.opt_init(parameters)

		_value,_grad,parameters = self.opt_step(iteration+1,state)

		# beta = (_grad.dot(_grad))/(grad.dot(grad)) # Fletcher-Reeves
		# beta = max(0,(_grad.dot(_grad-grad))/grad.dot(grad)) # Polak-Ribiere
		# beta = [(_grad.dot(_grad))/(grad.dot(grad)),max(0,(_grad.dot(_grad-grad))/grad.dot(grad))]
		# beta = -beta[0] if beta[1] < -beta[0] else beta[1] if abs(beta[1]) <= beta[0] else beta[0] # Polak-Ribiere-Fletcher-Reeves
		beta = (_grad.dot(_grad-grad))/(search.dot(_grad-grad)) #	Hestenes-Stiefel 	
		# beta = (_grad.dot(_grad))/(search.dot(_grad-grad)) # Dai-Yuan https://doi.org/10.1137/S1052623497318992
		
		restart = ((self.modulo.get('restart') is not None) and ((iteration%self.modulo['restart']) == 0))
		beta = 0 if (restart or is_naninf(beta) or beta>self.eps['beta']) else beta
		search = -_grad + beta*search


		self.attributes['alpha'].append(alpha)
		self.attributes['beta'].append(beta)
		self.attributes['search'].append(search)
		
		state = self.opt_init(parameters)
		parameters = self.get_params(state)
		track = self.track		
		attributes = self.attributes
		hyperparameters = self.hyperparameters		
		self.status = self.callback(parameters,track,attributes,hyperparameters)

		return state


class Adam(OptimizerBase):
	'''
	Adam Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,attributes,hyperparameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		super().__init__(func,grad,callback,hyperparameters)

		defaults = {}
		for attr in defaults:
			self.hyperparameters[attr] = self.hyperparameters.get(attr,defaults[attr])
		
		defaults = {}
		for attr in defaults:
			self.attributes[attr] = self.attributes.get(attr,defaults[attr])		

		null = ['beta']
		for attr in null:
			self.hyperparameters.pop(attr,None)

		null = ['beta']
		for attr in null:
			self.attributes.pop(attr,None)

		self._optimizer = getattr(jax.example_libraries.optimizers,self.optimizer)

		self._opt_init,self._opt_update,self._get_params = self._optimizer(self.hyperparameters['alpha'])

		return

	def opt_init(self,parameters):
		'''
		Initialize optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
		Returns:
			state (object): optimizer state
		'''

		state = self._opt_init(parameters)

		return state

	def opt_update(self,iteration,state):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			state (object): optimizer state
		'''


		if self.size == 0:

			value,grad,parameters = self.opt_step(iteration-1,state)

			alpha = self.hyperparameters['alpha']
			search = -grad

			self.attributes['alpha'].append(alpha)
			self.attributes['search'].append(search)

			state = self.opt_init(parameters)
			parameters = self.get_params(state)
			track = self.track		
			attributes = self.attributes
			hyperparameters = self.hyperparameters			
			self.status = self.callback(parameters,track,attributes,hyperparameters)

		value,grad,parameters = self.opt_step(iteration,state)

		alpha = self.attributes['alpha'][-1]
		search = -grad

		state = self._opt_update(iteration,grad,state)

		self.attributes['alpha'].append(alpha)
		self.attributes['search'].append(search)

		parameters = self.get_params(state)
		track = self.track		
		attributes = self.attributes
		hyperparameters = self.hyperparameters		
		self.status = self.callback(parameters,track,attributes,hyperparameters)

		return state

	def get_params(self,state):
		'''
		Get optimizer parameters with optimizer state
		Args:
			state (object): optimizer state
		Returns:
			parameters (object): optimizer parameters
		'''

		parameters = self._get_params(state)

		return parameters