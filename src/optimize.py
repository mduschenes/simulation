#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy,datetime
from functools import partial

envs = {
	'JAX_PLATFORM_NAME':'cpu',
	'TF_CPP_MIN_LOG_LEVEL':5
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
from src.utils import jit,value_and_gradient,gradient
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
			alpha (array): Returned search value
		'''
		returns = (alpha[-1],)

		returns = self.__callback__(returns,parameters,alpha,value,gradient,search)

		attr = 'alpha'
		alpha = returns[attr]

		return alpha

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
		if returns[attr] is None or (returns[attr] < self.hyperparameters['bounds'][attr][0]) or (returns[attr] > self.hyperparameters['bounds'][attr][1]):
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
	
		defaults = {'method':{'alpha':None}}
		hyperparameters.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		line_searches = {'line_search':Line_Search,'armijo':Armijo,None:Null_Search}

		line_search = hyperparameters.get('method',{}).get('alpha',defaults['method']['alpha'])
		
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
			alpha (array): Returned search value
		'''
		self.defaults['old_old_fval'] = value[-2] if len(value)>1 else None
		
		returns = line_search(self.func,self.grad,
			parameters,search[-1],gradient[-1],value[-1],
			**self.defaults)
		
		returns = self.__callback__(returns,parameters,alpha,value,gradient,search)

		attr = 'alpha'
		alpha = returns[attr]

		return alpha


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
			alpha (array): Returned search value
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


class FuncBase(object):

	def __init__(self,model,func,grad=None,callback=None,metric=None,hyperparameters={}):
		'''	
		Class for function
		Args:
			model (object): Model instance
			func (callable,iterable[callable]): Function function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			callback (callable): Callback of function with signature callback(parameters,track,attributes,model,func,grad,hyperparameters)
			metric (str,callable): Function metric
			hyperparameters (dict): Function hyperparameters
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

		values_and_grads = [value_and_grad(func,grad) for func,grad in zip(funcs,grads)]
		self.values_and_grads,self.funcs,self.grads = [[val_and_grad[i] for val_and_grad in values_and_grads] 
				for i in range(min(len(val_and_grad) for val_and_grad in values_and_grads))]

		self.value_and_grad,self.func,self.grad = value_and_grad(func,grad)

		self.model = model
		self.hyperparameters = hyperparameters

		if callback is None:
			def callback(parameters,track,attributes,model,func,grad,hyperparameters):
				status = True
				return status

		self.callback = callback

		self.metric = Metric(metric,optimize=None)		

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
	def __callback__(self,parameters,track,attributes,hyperparameters):
		'''
		Callback call
		Args:
			parameters (array): parameters
			track (dict): callback track			
			attributes (dict): callback attributes			
			hyperparameters (dict): callback hyperparameters			
		Returns:
			status (int): status of callback
		'''
		status = self.callback(parameters,track,attributes,hyperparameters=hyperparameters,model=self.model,func=self.func,grad=self.grad)
		return status


class Objective(FuncBase):		
	def __init__(self,model,func,grad=None,callback=None,metric=None,label=None,hyperparameters={}):
		'''	
		Objective class for function
		Args:
			model (object): Model instance
			func (callable,iterable[callable]): Objective function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			callback (callable): Callback of function with signature callback(parameters,track,attributes,model,func,grad,hyperparameters)			
			metric (str,callable): Objective metric
			label (str,callable): Objective label
			hyperparameters (dict): Objective hyperparameters
		'''

		super().__init__(model,func,grad=grad,callback=callback,metric=metric,hyperparameters=hyperparameters)

		self.label = label

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
		return self.metric(self.func(parameters),self.label)



class Callback(FuncBase):

	def __init__(self,model,func,grad=None,callback=None,metric=None,hyperparameters={}):
		'''	
		Class for function
		Args:
			model (object): Model instance
			func (callable,iterable[callable]): Function function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			callback (callable): Callback of function with signature callback(parameters,track,attributes,model,func,grad,hyperparameters)
			metric (str,callable): Function metric
			hyperparameters (dict): Function hyperparameters
		'''
		
		super().__init__(model,func,grad=grad,callback=callback,metric=metric,hyperparameters=hyperparameters)

		return

	# @partial(jit,static_argnums=(0,))
	def __call__(self,parameters,track,attributes,hyperparameters):
		''' 
		Callback
		Args:
			parameters (array): parameters
			track (dict): callback tracking
			attributes (dict): Callback attributes
			hyperparameters(dict): Callback hyperparameters
		Returns:
			status (int): status of callback
		'''
		status = self.callback(parameters,track,attributes,hyperparameters=hyperparameters,model=self.model,func=self.func,grad=self.grad)
		return status


class Metric(object):
	'''
	Metric class for distance between Operators
	Args:
		metric (str,Metric): Type of metric
		shapes (iterable[tuple[int]]): Shapes of Operators
		optimize (bool,str,iterable): Contraction type		
	'''
	def __init__(self,metric,shapes=None,optimize=None):

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
		if self.shapes is None:
			self.shapes = ()

		self.__string__()
		self.__size__()

		self.get_metric()

		return

	def __string__(self):
		self.string = str(self.metric)
		return

	def __size__(self):
		if self.shapes:
			self.size = sum(int(product(shape)**(1/len(shape))) for shape in self.shapes)//len(self.shapes)
		else:
			self.size = 1
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
			metric = self.metric
			func = jit(metric)
			grad = jit(gradient(metric))

		elif self.metric is None:
			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				_func = jit(normed_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_func = jit(partial(normed_einsum,optimize=optimize))

			
			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				_grad = jit(gradient_normed_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_grad = jit(partial(gradient_normed_einsum,optimize=optimize))

			@jit
			def func(a,b):
				return _func(a,b)/2/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return _grad(a,b,da)/2/sqrt(a.shape[0]*b.shape[0])	

		elif self.metric in ['normed']:
			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				_func = jit(normed_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_func = jit(partial(normed_einsum,optimize=optimize))


			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				_grad = jit(gradient_normed_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_grad = jit(partial(gradient_normed_einsum,optimize=optimize))

			@jit
			def func(a,b):
				return _func(a,b)/2/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return _grad(a,b,da)/2/sqrt(a.shape[0]*b.shape[0])

		elif self.metric in ['infidelity']:

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				_func = jit(inner_real_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_func = jit(partial(inner_real_einsum,optimize=optimize))

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				_grad = jit(gradient_inner_real_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_grad = jit(partial(gradient_inner_real_einsum,optimize=optimize))	

			@jit
			def func(a,b):
				return 1-_func(a,b)/(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)/(a.shape[0]*b.shape[0])

		elif self.metric in ['infidelity.abs','abs2']:

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
			else:
				shapes = ()
				optimize = self.optimize
			_func = jit(inner_abs2_einsum(*shapes,optimize=optimize))

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
			else:
				shapes = ()
				optimize = self.optimize
			_grad = jit(gradient_inner_abs2_einsum(*shapes,optimize=optimize))

			# @jit
			def func(a,b):
				return 1-_func(a,b)/(a.shape[0]*b.shape[0])
			# @jit
			def grad(a,b,da):
				return -_grad(a,b,da)/(a.shape[0]*b.shape[0])

		elif self.metric in ['infidelity.norm','norm']:
			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				_func = jit(inner_real_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_func = jit(partial(inner_real_einsum,optimize=optimize))

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				_grad = jit(gradient_inner_real_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_grad = jit(partial(gradient_inner_real_einsum,optimize=optimize))	

			@jit
			def func(a,b):
				return 1-_func(a,b)
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)

		elif self.metric in ['infidelity.real','real']:

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				_func = jit(inner_real_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_func = jit(partial(inner_real_einsum,optimize=optimize))

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				_grad = jit(gradient_inner_real_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_grad = jit(partial(gradient_inner_real_einsum,optimize=optimize))	

			@jit
			def func(a,b):
				return 1-_func(a,b)/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)/sqrt(a.shape[0]*b.shape[0])

		elif self.metric in ['infidelity.imag','imag']:

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				_func = jit(inner_imag_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_func = jit(partial(inner_imag_einsum,optimize=optimize))

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				_grad = jit(gradient_inner_imag_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_grad = jit(partial(gradient_inner_imag_einsum,optimize=optimize))	

			@jit
			def func(a,b):
				return 1-_func(a,b)/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)/sqrt(a.shape[0]*b.shape[0])	

		elif self.metric in ['infidelity.real.imag','real.imag']:

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				_func = jit(inner_real_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_func = jit(partial(inner_real_einsum,optimize=optimize))

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				_grad = jit(gradient_inner_real_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_grad = jit(partial(gradient_inner_real_einsum,optimize=optimize))	

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				_func_imag = jit(inner_imag_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_func_imag = jit(partial(inner_imag_einsum,optimize=optimize))

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				_grad_imag = jit(gradient_inner_imag_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_grad_imag = jit(partial(gradient_inner_imag_einsum,optimize=optimize))	

			@jit
			def func(a,b):
				return 1-(_func_real(a,b)+_func_imag(a,b))/2/sqrt(a.shape[0]*b.shape[0])
			@jit
			def grad(a,b,da):
				return -(_grad_real(a,b,da)+_grad_imag(a,b,da))/2/sqrt(a.shape[0]*b.shape[0])

		else:

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				_func = jit(normed_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_func = jit(partial(normed_einsum,optimize=optimize))

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				_grad = jit(gradient_normed_einsum(*shapes,optimize=optimize))
			else:
				shapes = ()
				optimize = self.optimize
				_grad = jit(partial(gradient_normed_einsum,optimize=optimize))

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
			'method':{'alpha':'line_search','beta':None},
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

		self.alpha = LineSearch(self.func,self.grad,self.hyperparameters)

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
		self.method = hyperparameters['method']
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

		def update(parameters,alpha,search):
			return parameters + alpha*search

		steps = self.size == 0

		for step in range(steps+1):

			init = self.size == 0

			value,grad,parameters = self.opt_step(iteration-init,state)

			alpha = self.hyperparameters['alpha']
			search = -grad

			if not init:
				parameters = update(parameters,alpha,search)

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
				if self.attributes[attr]:
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

		def update(parameters,alpha,search):
			return parameters + alpha*search

		steps = self.size == 0

		for step in range(steps+1):

			init = self.size == 0

			value,grad,parameters = self.opt_step(iteration-init,state)

			alpha = self.hyperparameters['alpha']
			search = -grad

			if not init:
				parameters = update(parameters,alpha,search)

			self.attributes['alpha'].append(alpha)
			self.attributes['search'].append(search)

			state = self.opt_init(parameters)
			parameters = self.get_params(state)
			track = self.track		
			attributes = self.attributes
			hyperparameters = self.hyperparameters			
			self.status = self.callback(parameters,track,attributes,hyperparameters)

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

		defaults = {'beta':0,'method':None}
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


		beta = self.method.get('beta')
		if beta is None:
			def beta(_grad,grad,search):
				beta = (_grad.dot(_grad))/(grad.dot(grad)) # Fletcher-Reeves
				return beta
		
		elif beta in ['fletcher_reeves']:
			def beta(_grad,grad,search):
				beta = (_grad.dot(_grad))/(grad.dot(grad)) # Fletcher-Reeves
				return beta
		
		elif beta in ['polak_ribiere']:
			def beta(_grad,grad,search):
				return max(0,(_grad.dot(_grad-grad))/grad.dot(grad)) # Polak-Ribiere
		
		elif beta in ['polak_ribiere_fletcher_reeves']:
			def beta(_grad,grad,search):
				beta = [(_grad.dot(_grad))/(grad.dot(grad)),max(0,(_grad.dot(_grad-grad))/grad.dot(grad))] # Polak-Ribiere-Fletcher-Reeves
				beta = -beta[0] if beta[1] < -beta[0] else beta[1] if abs(beta[1]) <= beta[0] else beta[0]
				return beta			

		elif beta in ['hestenes_stiefel']:
			def beta(_grad,grad,search):
				beta = (_grad.dot(_grad-grad))/(search.dot(_grad-grad)) # Hestenes-Stiefel
				return beta

		elif beta in ['dai_yuan']:
			def beta(_grad,grad,search):
				beta = (_grad.dot(_grad))/(search.dot(_grad-grad)) # Dai-Yuan https://doi.org/10.1137/S1052623497318992
				return beta								
					
		else:
			def beta(_grad,grad,search):
				beta = (_grad.dot(_grad))/(grad.dot(grad)) # Fletcher-Reeves
				return beta			

		self.beta = beta

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

		alpha = self.alpha(
			parameters,
			self.attributes['alpha'],
			self.attributes['value'],
			self.attributes['grad'],
			self.attributes['search'])

		search = self.attributes['search'][-1]
		grad = self.attributes['grad'][-1]

		parameters = parameters + alpha*search

		state = self.opt_init(parameters)

		_value,_grad,parameters = self.opt_step(iteration,state)
		
		beta = self.beta(_grad,grad,search)
		
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

		null = []
		for attr in null:
			self.hyperparameters.pop(attr,None)

		null = []
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