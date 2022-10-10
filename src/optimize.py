#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial

import jax

jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'

# Logging
import logging
logger = logging.getLogger(__name__)

# Import user modules
from src.utils import jit,value_and_gradient,gradient,hessian
from src.utils import is_naninf
from src.line_search import line_search,armijo


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
		returns = ['alpha']

		self.func = func
		self.grad = grad
		self.hyperparameters = hyperparameters
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

		if returns['alpha'] is None:
			if len(alpha) > 1:
				returns['alpha'] = alpha[-1]*gradient[-1].dot(search[-1])/gradient[-2].dot(search[-2])
			else:
				returns['alpha'] = alpha[-1]
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
		hyperparameters.update({attr: defaults[attr] for attr in defaults if attr not in hyperparameters})

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
		hyperparameters = {attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults}

		super().__init__(func,grad,hyperparameters)

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
		self.hyperparameters['old_old_fval'] = value[-2] if len(value)>1 else None
		
		# returns = osp.optimize.line_search(self.func,self.grad,
		returns = line_search(self.func,self.grad,
			parameters,search[-1],gradient[-1],value[-1],
			**self.hyperparameters)
		
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
		defaults = {'c1':0.0001}
		returns = ['alpha','nfunc','value']
		hyperparameters = {attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults}

		super().__init__(func,grad,hyperparameters)

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
		# returns = osp.optimize.linesearch.line_search_armijo(self.func,self.grad,
		returns = armijo(self.func,self.grad,
			parameters,search[-1],gradient[-1],value[-1],
			**self.hyperparameters)

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
	def __init__(self,func,grad=None,hess=None,callback=None,hyperparameters={}):
		'''	
		Objective class for function
		Args:
			func (callable,iterable[callable]): Objective function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function  with signature grad(parameters)
			hess (callable,iterable[callable]): Hessian of function, with signature hess(parameters)
			callback (callable): Gradient of function  with signature callback(parameters,func,grad,hess,funcs,grads,hesss,hyperparameters)
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
			def callback(parameters,func,grad,hyperparameters):
				status = False
				return False

		self.callback = jit(partial(callback,func=self.func,grad=self.grad,
				funcs=self.funcs,grads=self.grads,hess=self.hess,
				hyperparameters=self.hyperparameters))

		return

	@partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Function call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of objective function
		'''
		return self.func(parameters)

	@partial(jit,static_argnums=(0,))
	def __grad__(self,parameters):
		'''
		Gradient call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of gradient function
		'''
		return self.grad(parameters)	

	@partial(jit,static_argnums=(0,))
	def __hessian__(self,parameters):
		''' 
		Hessian call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of hessian function
		'''	
		return self.hess(parameters)	

	@partial(jit,static_argnums=(0,))
	def __callback__(self,parameters):
		'''
		Callback call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of objective function
		'''
		return self.callback(parameters)		


class Base(object):
	'''
	Base Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters) and returns status of optimization
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
			'eps':{'objective':1e-4,'grad':1e-12,'alpha':1e-12,'beta':1e3},
			'alpha':0,
			'iterations':0,
			'status':1,
			'reset':0,
			'verbose':False,
			'modulo':{'log':1,'history':10,'callback':1,'restart':1e10},
			'attributes':{'value':[],'grad':[],'search':[],'alpha':[]},			
			'track':{'iteration':[],'value':[],'grad':[],'search':[],'alpha':[]},			
		}

		hyperparameters.update({attr: defaults[attr] for attr in defaults if attr not in hyperparameters})
		hyperparameters.update({attr: updates.get(attr,{}).get(hyperparameters[attr],hyperparameters[attr]) 
			if attr in updates else hyperparameters[attr] for attr in hyperparameters})


		self.optimizer = hyperparameters['optimizer']		
		self.iterations = range(int(hyperparameters['iterations']))
		self.track = hyperparameters['track']
		self.modulo = hyperparameters['modulo']
		self.hyperparameters = hyperparameters


		self.value_and_grad,self.func,self.grad = value_and_grad(func,grad)

		self.line_search = LineSearch(self.func,self.grad,self.hyperparameters)

		self.callback = callback if callback is not None else (lambda parameters: None)

		self.size = 0
		self.iteration = -1
		self.parameters = None
		self.attributes = hyperparameters['attributes']
		self.status = hyperparameters['status']
		self.eps = hyperparameters['eps']
		self.reset = hyperparameters['reset']
		self.verbose = hyperparameters['verbose']

		self.__reset__()

		return

	def __call__(self,parameters):
		'''
		Iterate optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
		Returns:
			parameters (object): optimizer parameters
		'''

		if not self.reset and self.size > 0:
			parameters = self.parameters
			self.status = self.callback(parameters)

		state = self.opt_init(parameters)
		for iteration in self.iterations:
			
			state = self.opt_update(iteration,state)

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

		value,grad,parameters = self.opt_step(iteration,state)

		alpha = self.attributes['alpha'][-1] if self.size > 1 else self.hyperparameters['alpha']
		search = -grad

		parameters = parameters + alpha*search

		self.attributes['alpha'].append(alpha)
		self.attributes['search'].append(search)

		self.track['alpha'].append(self.attributes['alpha'][-1])
		self.track['search'].append(self.attributes['search'][-1])

		state = self.opt_init(parameters)
		parameters = self.get_params(state)
		
		self.status = self.callback(parameters)

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

		if self.size >= self.modulo['history']:
			for attr in self.attributes:
				self.attributes[attr].pop(0)

		self.iteration = iteration
		self.attributes['value'].append(value)
		self.attributes['grad'].append(grad)

		self.track['iteration'].append(iteration)
		self.track['value'].append(value)
		self.track['grad'].append(grad)			

		self.size += 1

		return value,grad,parameters

	def __reset__(self,reset=None):
		'''
		Reset tracking of optimization
		Args:
			reset (bool): Boolean of resetting optimization
		'''

		if reset is None:
			reset = self.reset

		if reset:
			self.size = 0
			self.iteration = -1
			for attr in self.attributes:
				self.attributes[attr].clear()
			for attr in self.track:
				self.track[attr].clear()
			self.parameters = None
		else:
			if any(len(self.track[attr])>0 for attr in self.track):
				self.size = min(len(self.track[attr]) for attr in self.track if len(self.track[attr])>0)
			else:
				self.size = 0

			if self.size > 0:
				self.iteration = self.track['iteration'][-1]
				self.parameters = self.track['parameters'][-1]
			else:
				self.iteration = 0
				self.parameters = None

			self.iteration -= 1

		self.iterations = range(
			self.iterations.start+self.iteration+1,
			self.iterations.stop+self.iteration+1,
			self.iterations.step)

		return

class Optimizer(Base):
	'''
	Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __new__(cls,func,grad=None,callback=None,hyperparameters={}):
	
		defaults = {'optimizer':None}
		hyperparameters.update({attr: defaults[attr] for attr in defaults if attr not in hyperparameters})

		optimizers = {'adam':Adam,'cg':ConjugateGradient,'gd':GradientDescent,None:GradientDescent}

		optimizer = hyperparameters['optimizer']		
		
		self = optimizers.get(optimizer,optimizers[None])(func,grad,callback,hyperparameters)

		return self
	

class GradientDescent(Base):
	'''
	Gradient Descent Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		super().__init__(func,grad,callback,hyperparameters)

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

		value,grad,parameters = self.opt_step(iteration,state)

		alpha = self.attributes['alpha'][-1] if self.size > 1 else self.hyperparameters['alpha']
		search = -grad

		parameters = parameters + alpha*search

		self.attributes['alpha'].append(alpha)
		self.attributes['search'].append(search)

		self.track['alpha'].append(self.attributes['alpha'][-1])
		self.track['search'].append(self.attributes['search'][-1])

		state = self.opt_init(parameters)
		parameters = self.get_params(state)
		
		self.status = self.callback(parameters)

		return state


class ConjugateGradient(Base):
	'''
	Conjugate Gradient Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		super().__init__(func,grad,callback,hyperparameters)

		defaults = {
			'beta':0,
			'attributes':{'beta':[]},
			'track':{'beta':[]},
			}
		self.hyperparameters.update({attr: self.hyperparameters.get(attr,defaults[attr]) for attr in defaults})
		self.track.update({attr: self.track.get(attr,defaults['track'][attr]) for attr in defaults['track']})
		self.attributes.update({attr: self.attributes.get(attr,defaults['attributes'][attr]) for attr in defaults['attributes']})

		null = {
			'attributes':{},
			'track':{}
			}
		for attr in null:
			if attr not in ['attributes','track']:
				self.hyperparameters.pop(attr,None)
		for attr in null['track']:
			self.track.pop(attr,None)
		for attr in null['attributes']:
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
			value,grad,parameters = self.opt_step(iteration,state)

			self.attributes['alpha'].append(self.hyperparameters['alpha'])
			self.attributes['beta'].append(self.hyperparameters['beta'])
			self.attributes['search'].append(-grad)

			self.track['alpha'].append(self.attributes['alpha'][-1])			
			self.track['beta'].append(self.attributes['beta'][-1])			
			self.track['search'].append(self.attributes['search'][-1])			

			state = self.opt_init(parameters)
			parameters = self.get_params(state)
			self.status = self.callback(parameters)

		parameters = self.get_params(state)

		returns = self.line_search(
			parameters,
			self.track['alpha'],
			self.track['value'],
			self.track['grad'],
			self.track['search'])

		alpha = returns['alpha']
		search = self.track['search'][-1]
		grad = self.track['grad'][-1]

		parameters = parameters + alpha*search

		state = self.opt_init(parameters)

		_value,_grad,parameters = self.opt_step(iteration+1,state)

		# beta = (_grad.dot(_grad))/(grad.dot(grad)) # Fletcher-Reeves
		# beta = max(0,(_grad.dot(_grad-grad))/grad.dot(grad)) # Polak-Ribiere
		# beta = [(_grad.dot(_grad))/(grad.dot(grad)),max(0,(_grad.dot(_grad-grad))/grad.dot(grad))]
		# beta = -beta[0] if beta[1] < -beta[0] else beta[1] if abs(beta[1]) <= beta[0] else beta[0] # Polak-Ribiere-Fletcher-Reeves
		beta = (_grad.dot(_grad-grad))/(search.dot(_grad-grad)) #	Hestenes-Stiefel 	
		# beta = (_grad.dot(_grad))/(search.dot(_grad-grad)) # Dai-Yuan https://doi.org/10.1137/S1052623497318992
		
		restart = (iteration%self.modulo['restart']) == 0
		beta = 0 if (restart or is_naninf(beta) or beta>self.eps['beta']) else beta
		search = -_grad + beta*search


		self.attributes['alpha'].append(alpha)
		self.attributes['beta'].append(beta)
		self.attributes['search'].append(search)
		
		self.track['alpha'].append(self.attributes['alpha'][-1])			
		self.track['beta'].append(self.attributes['beta'][-1])			
		self.track['search'].append(self.attributes['search'][-1])	
		
		state = self.opt_init(parameters)

		parameters = self.get_params(state)
		
		self.status = self.callback(parameters)

		return state


class Adam(Base):
	'''
	Adam Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		super().__init__(func,grad,callback,hyperparameters)

		null = {
			'beta':{},
			'attributes':{'beta':None},
			'track':{'beta':None}
			}
		for attr in null:
			if attr not in ['attributes','track']:
				self.hyperparameters.pop(attr,None)
		for attr in null['track']:
			self.track.pop(attr,None)
		for attr in null['attributes']:
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

		value,grad,parameters = self.opt_step(iteration,state)

		state = self._opt_update(iteration,grad,state)

		alpha = self.attributes['alpha'][-1] if self.size > 1 else self.hyperparameters['alpha']
		search = -grad

		self.attributes['alpha'].append(alpha)
		self.attributes['search'].append(search)

		self.track['alpha'].append(self.attributes['alpha'][-1])			
		self.track['search'].append(self.attributes['search'][-1])			

		parameters = self.get_params(state)
		
		self.status = self.callback(parameters)

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