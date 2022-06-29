#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp

import jax
import jax.numpy as np
import jax.scipy as sp
import jax.example_libraries.optimizers as optimizers
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'
# np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})

# Logging
import logging
logger = logging.getLogger(__name__)

# Import user modules
from src.utils import jit,value_and_gradient,gradient
from src.utils import isnaninf


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
		
		returns = osp.optimize.line_search(self.func,self.grad,
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
		returns = osp.optimize.linesearch.line_search_armijo(self.func,self.grad,
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
	def __init__(self,func,hyperparameters={}):
		'''	
		Objective class for function
		Args:
			func (callable): Objective function with signature func(parameters)
			hyperparameters (dict): Objective hyperparameters
		'''

		self.func = func
		self.hyperparameters = hyperparameters

		return

	@partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Plot Parameters
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of objective function
		'''
		return self.func(parameters)


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
			'search':0,
			'iterations':0,
			'status':1,
			'reset':0,
			'verbose':False,
			'modulo':{'log':1,'size':10,'callback':1,'restart':1e10},
			'track':{'size':0,'iteration':[],'value':[],'grad':[],'search':[],'alpha':[]},			
		}
		
		hyperparameters.update({attr: defaults[attr] for attr in defaults if attr not in hyperparameters})
		hyperparameters.update({attr: updates.get(attr,{}).get(hyperparameters[attr],hyperparameters[attr]) 
			if attr in updates else hyperparameters[attr] for attr in hyperparameters})


		self.optimizer = hyperparameters['optimizer']		
		self.iterations = int(hyperparameters['iterations'])
		self.track = hyperparameters['track']
		self.modulo = hyperparameters['modulo']
		self.hyperparameters = hyperparameters

		self.alpha = hyperparameters['alpha']
		self.search = hyperparameters['search'] 
		self.status = hyperparameters['status']
		self.eps = hyperparameters['eps']
		self.verbose = hyperparameters['verbose']
		
		self.reset(hyperparameters['reset'])

		self.value_and_grad,self.func,self.grad = value_and_grad(func,grad)

		self.line_search = LineSearch(self.func,self.grad,self.hyperparameters)

		self.callback = callback if callback is not None else (lambda parameters: None)

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
		for iteration in range(self.iterations):

			state = self.opt_update(iteration,state)

			if not self.status:
				break

		parameters = self.get_params(state)

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

		alpha = self.alpha
		search = -grad

		parameters = parameters + alpha*search

		self.alpha = alpha
		self.search = search

		self.track['alpha'].append(self.alpha)
		self.track['search'].append(self.search)

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

		if self.track['size'] > self.modulo['size']:
			self.track['grad'].pop(0)
			self.track['search'].pop(0)

		self.track['iteration'].append(iteration)
		self.track['value'].append(value)
		self.track['grad'].append(grad)			

		self.track['size'] += 1

		return value,grad,parameters

	def reset(self,reset=False):
		'''
		Reset tracking of optimization
		Args:
			reset (bool): Boolean of resetting optimization
		'''

		if reset:
			self.track['size'] = 0

			for attr in self.track:
				if isinstance(self.track[attr],list):
					self.track[attr].clear()

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

		alpha = self.alpha
		search = -grad

		parameters = parameters + alpha*search

		self.alpha = alpha
		self.search = search

		self.track['alpha'].append(self.alpha)
		self.track['search'].append(self.search)

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

		self.beta = 0
		self.track.update(**{attr: [] for attr in ['beta'] if attr not in self.track})

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

		if self.track['size'] == 0:
			value,grad,parameters = self.opt_step(iteration,state)

			self.alpha = self.alpha
			self.beta = self.beta
			self.search = -grad
			self.track['alpha'].append(self.alpha)			
			self.track['beta'].append(self.beta)			
			self.track['search'].append(self.search)			

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
		beta = 0 if (restart or isnaninf(beta) or beta>self.eps['beta']) else beta
		search = -_grad + beta*search


		self.alpha = alpha
		self.beta = beta
		self.search = search
		
		self.track['alpha'].append(self.alpha)
		self.track['beta'].append(self.beta)
		self.track['search'].append(self.search)

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

		self._optimizer = getattr(jax.example_libraries.optimizers,self.optimizer)

		self._opt_init,self._opt_update,self._get_params = self._optimizer(self.alpha)

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

		alpha = self.alpha
		search = -grad

		self.alpha = alpha
		self.search = search

		self.track['alpha'].append(self.alpha)			
		self.track['search'].append(self.search)			

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