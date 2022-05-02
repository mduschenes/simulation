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
from utils import jit,gradient


def value_and_grad(func,grad=None):
	def _value_and_grad(func,grad):
		def _func_and_grad(parameters):
			return func(parameters),grad(parameters)
		return _func_and_grad

	if grad is None:
		grad = gradient(func)
		func_and_grad = jit(jax.value_and_grad(func))
	else:
		func_and_grad = _value_and_grad(func,grad)

	return func_and_grad,func,grad


def line_search(func,grad,parameters,alpha,value,gradient,search,hyperparameters):
	attrs = {'c1':0.0001,'c2':0.9,'maxiter':10,'old_old_fval':value[-2] if len(value)>1 else None}
	attrs.update({attr: hyperparameters.get(attr,attrs[attr]) for attr in attrs})
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

class OptimizerBase(object):
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
			'alpha':None,
			'iterations':0,
			'track':{'log':1,'track':10,'size':0,'iteration':[],'value':[],'grad':[],'search':[],'alpha':[],'beta':[],'lambda':[]},			
		}
		hyperparameters.update({attr: defaults[attr] for attr in defaults if attr not in hyperparameters})

		self.optimizer = hyperparameters['optimizer']
		self.alpha = hyperparameters['alpha']
		self.iterations = hyperparameters['iterations']
		self.track = hyperparameters['track']
		self.hyperparameters = hyperparameters

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

		value,grad = self.opt_step(iteration,state)

		search = -grad
		alpha = self.alpha

		state += alpha*search

		self.track['search'].append(search)
		self.track['alpha'].append(alpha)

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

		if self.track['size'] == 0:
			self.track['iteration'].append(iteration)
			self.track['value'].append(value)
			self.track['grad'].append(grad)			

		self.track['iteration'].append(iteration+1)
		self.track['value'].append(value)
		self.track['grad'].append(grad)			

		self.track['size'] += 1

		return value,grad


class Optimizer(OptimizerBase):
	'''
	Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters)				
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={}):

		super().__init__(func,grad,callback,hyperparameters)

		optimizers = {'adam':Adam,'cg':ConjugateGradient,'gd':GradientDescent,None:GradientDescent}

		self.Optimizer = optimizers.get(self.optimizer,optimizers[None])(self.func,self.grad,self.callback,self.hyperparameters)

		return

	def opt_init(self,parameters):
		'''
		Initialize optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
		Returns:
			state (object): optimizer state
		'''
		return self.Optimizer.opt_init(parameters)

	def opt_update(self,iteration,state):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			state (object): optimizer state
		'''
		return self.Optimizer.opt_update(iteration,state)

	def get_params(self,state):
		'''
		Get optimizer parameters with optimizer state
		Args:
			state (object): optimizer state
		Returns:
			parameters (object): optimizer parameters
		'''
		return self.Optimizer.get_params(state)						



class GradientDescent(OptimizerBase):
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

		value,grad = self.opt_step(iteration,state)

		search = -grad
		alpha = self.alpha

		state += alpha*search

		self.track['search'].append(_search)		
		self.track['alpha'].append(alpha)

		return state



class ConjugateGradient(OptimizerBase):
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


		# value,grad = self.opt_step(iteration,state,hyperparameters)
		if self.track['size'] == 0:
			self.track['iteration'].append(iteration)
			self.hyperparameters['lambda'] = self.hyperparameters['lambda']
			value,grad = self.value_and_grad(state)
			self.track['value'].append(value)
			self.track['grad'].append(grad)			
			self.track['search'].append(-grad)			
			self.track['alpha'].append(self.hyperparameters['alpha'])			
			self.track['beta'].append(self.hyperparameters['beta'])			
			self.track['lambda'].append(self.hyperparameters['lambda'])			
			self.track['size'] += 1

			self.callback(self.get_params(state))

		returns = line_search(self.func,self.grad,state,
			self.track['alpha'],
			self.track['value'],
			self.track['grad'],
			self.track['search'],
			self.hyperparameters)
		# returns = {'alpha':self.hyperparameters['alpha']}



		alpha = returns['alpha']
		grad = self.track['grad'][-1]
		search = self.track['search'][-1]
		self.hyperparameters['lambda'] = self.hyperparameters['lambda']

		state += alpha*search

		_value,_grad = self.value_and_grad(state)

		grads = grad.dot(grad)

		# beta = (_grad.dot(_grad))/(grads) # Fletcher-Reeves
		# beta = max(0,(_grad.dot(_grad-grad))/grads) # Polak-Ribiere
		# beta = [(_grad.dot(_grad))/(grads),max(0,(_grad.dot(_grad-grad))/grads)]
		# beta = -beta[0] if beta[1] < -beta[0] else beta[1] if abs(beta[1]) <= beta[0] else beta[0] # Polak-Ribiere-Fletcher-Reeves
		beta = (_grad.dot(_grad-grad))/(search.dot(_grad-grad)) #	Hestenes-Stiefel 	
		# beta = (_grad.dot(_grad))/(search.dot(_grad-grad)) # Dai-Yuan https://doi.org/10.1137/S1052623497318992
		
		# restart = (iteration%self.hyperparameters['restart']) == 0
		# beta = 0 if (restart or grads < self.hyperparameters['tol']) else beta
		_search = -_grad + beta*search
		
		
		if self.track['size'] == self.track['track']:
			self.track['grad'].pop(0)
			self.track['search'].pop(0)

		self.track['iteration'].append(iteration+1)
		self.track['value'].append(_value)		
		self.track['grad'].append(_grad)
		self.track['search'].append(_search)
		self.track['alpha'].append(alpha)
		self.track['beta'].append(beta)
		self.track['lambda'].append(self.hyperparameters['lambda'])			
		self.track['size'] += 1

		self.callback(self.get_params(state))


		if iteration%self.track['log'] == 0:			
			# print(self.hyperparameters['label'])
			# print(state)
			logger.log(50,'%d f(x) = %0.4f'%(iteration,self.track['objective'][-1]))
			print('alpha = ',alpha)
			print('beta = ',beta)			
			print()
		return state



class Adam(OptimizerBase):
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
		return self._opt_init(parameters)

	def opt_update(self,iteration,state):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			state (object): optimizer state
		'''

		value,grad = self.opt_step(iteration,state)

		return self._opt_update(iteration,grad,state)

	def get_params(self,state):
		'''
		Get optimizer parameters with optimizer state
		Args:
			state (object): optimizer state
		Returns:
			parameters (object): optimizer parameters
		'''
		return self._get_params(state)