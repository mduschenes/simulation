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
np.set_printoptions(linewidth=500)
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'

# Logging
import logging,logging.config
logger = logging.getLogger(__name__)
conf = 'config/logging.conf'
try: 
	logging.config.fileConfig(conf,disable_existing_loggers=False) 
except:
	pass
logger = logging.getLogger(__name__)


# Import user modules
from utils import jit,value_and_gradient,gradient


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


def line_search(func,grad,parameters,alpha,value,gradient,search,hyperparameters):
	attrs = {'c1':0.0001,'c2':0.9,'maxiter':10,'old_old_fval':value[-2] if len(value)>1 else None}
	attrs.update({attr: hyperparameters.get(attr,attrs[attr]) for attr in attrs})
	# returns = {'alpha':hyperparameters['alpha']}
	# return returns
	returns = osp.optimize.line_search(func,grad,parameters,search[-1],gradient[-1],value[-1],**attrs)
	returns = dict(zip(['alpha','func','grad','value','_value','slope'],returns))
	if returns['alpha'] is None:
		if len(alpha) > 1:
			returns['alpha'] = alpha[-1]*gradient[-1].dot(search[-1])/gradient[-2].dot(search[-2])
		else:
			returns['alpha'] = 1e-4
	returns['alpha'] = min(1,returns['alpha'])
	# elif returns['value'] > value[-1]:
	# 	returns['alpha'] = (alpha[-1] if len(alpha)>0 else 1e-1)*gradient[-1].dot(search[-1])/gradient[-min(2,len(gradient))].dot(search[-min(2,len(search))])
	return returns



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
		callback (callable): callback function with signature callback(parameters)				
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		defaults = {
			'optimizer':None,
			'alpha':0,
			'search':0,
			'iterations':0,
			'track':{'log':1,'track':10,'size':0,'iteration':[],'value':[],'grad':[],'search':[],'alpha':[]},			
		}
		hyperparameters.update({attr: defaults[attr] for attr in defaults if attr not in hyperparameters})

		self.optimizer = hyperparameters['optimizer']		
		self.iterations = hyperparameters['iterations']
		self.track = hyperparameters['track']
		self.hyperparameters = hyperparameters

		self.alpha = hyperparameters['alpha']
		self.search = hyperparameters['search'] 

		self.value_and_grad,self.func,self.grad = value_and_grad(func,grad)

		self.callback = callback if callback is not None else (lambda parameters: None)

		self.Optimizer = None

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

		state += alpha*search

		self.alpha = alpha
		self.search = search

		self.track['alpha'].append(self.alpha)
		self.track['search'].append(self.search)

		parameters = self.get_params(state)
		
		self.callback(parameters)

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

		if self.track['size'] > self.track['track']:
			self.track['grad'].pop(0)
			self.track['search'].pop(0)

		self.track['iteration'].append(iteration)
		self.track['value'].append(value)
		self.track['grad'].append(grad)			

		self.track['size'] += 1

		return value,grad,parameters


class Optimizer(Base):
	'''
	Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters)				
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		defaults = {
			'optimizer':None,
			'alpha':0,
			'search':0,
			'iterations':0,
			'track':{'log':1,'track':10,'size':0,'iteration':[],'value':[],'grad':[],'search':[],'alpha':[]},			
		}
		hyperparameters.update({attr: defaults[attr] for attr in defaults if attr not in hyperparameters})

		optimizers = {'adam':Adam,'cg':ConjugateGradient,'gd':GradientDescent,None:GradientDescent}

		self.optimizer = hyperparameters['optimizer']		
		self.iterations = hyperparameters['iterations']
		self.track = hyperparameters['track']
		self.hyperparameters = hyperparameters

		self.alpha = hyperparameters['alpha']
		self.search = hyperparameters['search'] 

		self.callback = callback if callback is not None else (lambda parameters: None)

		self.Optimizer = optimizers.get(self.optimizer,optimizers[None])(func,grad,callback,hyperparameters)
		
		self.value_and_grad,self.func,self.grad = self.Optimizer.value_and_grad,self.Optimizer.func,self.Optimizer.grad


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
		
		state = self.Optimizer.opt_init(parameters)
		
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
		state =  self.Optimizer.opt_update(iteration,state)
		
		return state

	def get_params(self,state):
		'''
		Get optimizer parameters with optimizer state
		Args:
			state (object): optimizer state
		Returns:
			parameters (object): optimizer parameters
		'''
		
		parameters = self.Optimizer.get_params(state)
		
		return parameters
	

class GradientDescent(Base):
	'''
	Gradient Descent Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters)				
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

		state += alpha*search

		self.alpha = alpha
		self.search = search

		self.track['alpha'].append(self.alpha)
		self.track['search'].append(self.search)

		parameters = self.get_params(state)
		
		self.callback(parameters)

		return state


class ConjugateGradient(Base):
	'''
	Conjugate Gradient Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters)				
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
			self.callback(parameters)


		parameters = self.get_params(state)

		returns = line_search(self.func,self.grad,parameters,
			self.track['alpha'],
			self.track['value'],
			self.track['grad'],
			self.track['search'],
			self.hyperparameters)

		alpha = returns['alpha']
		search = self.track['search'][-1]
		grad = self.track['grad'][-1]

		self.alpha = alpha
		self.search = search

		parameters += alpha*search

		_value,_grad,parameters = self.opt_step(iteration,state)

		# beta = (_grad.dot(_grad))/(grad.dot(grad)) # Fletcher-Reeves
		# beta = max(0,(_grad.dot(_grad-grad))/grad.dot(grad)) # Polak-Ribiere
		# beta = [(_grad.dot(_grad))/(grad.dot(grad)),max(0,(_grad.dot(_grad-grad))/grad.dot(grad))]
		# beta = -beta[0] if beta[1] < -beta[0] else beta[1] if abs(beta[1]) <= beta[0] else beta[0] # Polak-Ribiere-Fletcher-Reeves
		beta = (_grad.dot(_grad-grad))/(search.dot(_grad-grad)) #	Hestenes-Stiefel 	
		# beta = (_grad.dot(_grad))/(search.dot(_grad-grad)) # Dai-Yuan https://doi.org/10.1137/S1052623497318992
		
		# restart = (iteration%self.hyperparameters['restart']) == 0
		# beta = 0 if (restart or grad.dot(grad) < self.hyperparameters['tol']) else beta
		_search = -_grad + beta*search


		self.alpha = alpha
		self.beta = beta
		self.search = _search
		
		self.track['alpha'].append(self.alpha)
		self.track['beta'].append(self.beta)
		self.track['search'].append(self.search)

		state = self.opt_init(parameters)

		parameters = self.get_params(state)
		
		self.callback(parameters)


		return state


class Adam(Base):
	'''
	Adam Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters)				
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		super().__init__(func,grad,callback,hyperparameters)

		self.Optimizer = getattr(jax.example_libraries.optimizers,self.optimizer)

		self._opt_init,self._opt_update,self._get_params = self.Optimizer(self.alpha)

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

		parameters = self.get_params(state)
		self.callback(parameters)

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