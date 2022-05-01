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
	attrs.update({attr: hyperparameters['hyperparameters'].get(attr,attrs[attr]) for attr in attrs})
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


class OptimizerBase(object):
	'''
	Base Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,hyperparameters={}):

		defaults = {
			'optimizer':None,
			'learning_rate':None,	
			'track':{'log':1,'track':10,'size':0,'iteration':[],'value':[],'grad':[],'search':[],'alpha':[],'beta':[],'lambda':[]},
		}
		hyperparameters.update({attr: defaults[attr] for attr in defaults if attr not in hyperparameters})

		self.optimizer = hyperparameters['optimizer']
		self.learning_rate = hyperparameters['learning_rate']
		self.hyperparameters = hyperparameters

		self.value_and_grad,self.func,self.grad = value_and_grad(func,grad)

		self.Optimizer = None

		return

	def opt_init(self,parameters,hyperparameters={}):
		'''
		Initialize optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
			hyperparameters (dict): optimizer hyperparameters						
		Returns:
			state (object): optimizer state
		'''
		state = parameters
		return state

	def opt_update(self,iteration,state,hyperparameters={}):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
			hyperparameters (dict): optimizer hyperparameters						
		Returns:
			state (object): optimizer state
		'''

		value,grad = self.opt_step(iteration,state,hyperparameters)

		search = -grad
		alpha = self.learning_rate

		state += alpha*search

		hyperparameters['track']['search'].append(search)
		hyperparameters['track']['alpha'].append(alpha)

		return state

	def get_params(self,state,hyperparameters={}):
		'''
		Get optimizer parameters with optimizer state
		Args:
			state (object): optimizer state
			hyperparameters (dict): optimizer hyperparameters						
		Returns:
			parameters (object): optimizer parameters
		'''
		parameters = state
		return parameters


	def opt_step(self,iteration,state,hyperparameters):
		'''
		Iterate optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
			hyperparameters (dict): optimizer hyperparameters						
		Returns:
			value (object): optimizer value
			grad (object): optimizer grad
		'''

		parameters = self.get_params(state,hyperparameters)
		value,grad = self.value_and_grad(parameters)

		if hyperparameters['track']['size'] > hyperparameters['track']['track']:
			hyperparameters['track']['grad'].pop(0)
			hyperparameters['track']['search'].pop(0)

		if hyperparameters['track']['size'] == 0:
			hyperparameters['track']['iteration'].append(iteration)
			hyperparameters['track']['value'].append(value)
			hyperparameters['track']['grad'].append(grad)			

		hyperparameters['track']['iteration'].append(iteration+1)
		hyperparameters['track']['value'].append(value)
		hyperparameters['track']['grad'].append(grad)			

		hyperparameters['track']['size'] += 1

		return value,grad


class Optimizer(OptimizerBase):
	'''
	Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,hyperparameters={}):

		super().__init__(func,grad,hyperparameters)

		optimizers = {'adam':Adam,'cg':ConjugateGradient,'gd':GradientDescent,None:GradientDescent}

		self.Optimizer = optimizers.get(self.optimizer,optimizers[None])(self.func,self.grad,self.hyperparameters)

		return

	def opt_init(self,parameters,hyperparameters={}):
		'''
		Initialize optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
			hyperparameters (dict): optimizer hyperparameters			
		Returns:
			state (object): optimizer state
		'''
		return self.Optimizer.opt_init(parameters,hyperparameters)

	def opt_update(self,iteration,state,hyperparameters={}):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
			hyperparameters (dict): optimizer hyperparameters						
		Returns:
			state (object): optimizer state
		'''
		return self.Optimizer.opt_update(iteration,state,hyperparameters)

	def get_params(self,state,hyperparameters={}):
		'''
		Get optimizer parameters with optimizer state
		Args:
			state (object): optimizer state
			hyperparameters (dict): optimizer hyperparameters						
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
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,hyperparameters={}):

		super().__init__(func,grad,hyperparameters)

		return

	def opt_update(self,iteration,state,hyperparameters={}):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
			hyperparameters (dict): optimizer hyperparameters						
		Returns:
			state (object): optimizer state
		'''

		value,grad = self.opt_step(iteration,state,hyperparameters)

		search = -grad
		alpha = self.learning_rate

		state += alpha*search

		hyperparameters['track']['search'].append(_search)		
		hyperparameters['track']['alpha'].append(alpha)

		return state



class ConjugateGradient(OptimizerBase):
	'''
	Conjugate Gradient Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,hyperparameters={}):

		super().__init__(func,grad,hyperparameters)

		return

	def opt_update(self,iteration,state,hyperparameters={}):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
			hyperparameters (dict): optimizer hyperparameters			
		Returns:
			state (object): optimizer state
		'''


		# value,grad = self.opt_step(iteration,state,hyperparameters)
		if hyperparameters['track']['size'] == 0:
			hyperparameters['track']['iteration'].append(iteration)
			hyperparameters['hyperparameters']['lambda'] = hyperparameters['hyperparameters']['lambda']
			value,grad = self.value_and_grad(state)
			hyperparameters['track']['value'].append(value)
			hyperparameters['track']['grad'].append(grad)			
			hyperparameters['track']['search'].append(-grad)			
			hyperparameters['track']['alpha'].append(hyperparameters['hyperparameters']['alpha'])			
			hyperparameters['track']['beta'].append(hyperparameters['hyperparameters']['beta'])			
			hyperparameters['track']['lambda'].append(hyperparameters['hyperparameters']['lambda'])			
			hyperparameters['track']['size'] += 1

			for callback in hyperparameters['callback']:
				hyperparameters['callback'][callback](state,hyperparameters)

		returns = line_search(self.func,self.grad,state,
			hyperparameters['track']['alpha'],
			hyperparameters['track']['value'],
			hyperparameters['track']['grad'],
			hyperparameters['track']['search'],
			hyperparameters)
		# returns = {'alpha':hyperparameters['hyperparameters']['alpha']}



		alpha = returns['alpha']
		grad = hyperparameters['track']['grad'][-1]
		search = hyperparameters['track']['search'][-1]
		hyperparameters['hyperparameters']['lambda'] = hyperparameters['hyperparameters']['lambda']

		state += alpha*search

		_value,_grad = self.value_and_grad(state)

		grads = grad.dot(grad)

		# beta = (_grad.dot(_grad))/(grads) # Fletcher-Reeves
		# beta = max(0,(_grad.dot(_grad-grad))/grads) # Polak-Ribiere
		# beta = [(_grad.dot(_grad))/(grads),max(0,(_grad.dot(_grad-grad))/grads)]
		# beta = -beta[0] if beta[1] < -beta[0] else beta[1] if abs(beta[1]) <= beta[0] else beta[0] # Polak-Ribiere-Fletcher-Reeves
		beta = (_grad.dot(_grad-grad))/(search.dot(_grad-grad)) #	Hestenes-Stiefel 	
		# beta = (_grad.dot(_grad))/(search.dot(_grad-grad)) # Dai-Yuan https://doi.org/10.1137/S1052623497318992
		
		# restart = (iteration%hyperparameters['hyperparameters']['restart']) == 0
		# beta = 0 if (restart or grads < hyperparameters['hyperparameters']['tol']) else beta
		_search = -_grad + beta*search
		
		
		if hyperparameters['track']['size'] == hyperparameters['track']['track']:
			hyperparameters['track']['grad'].pop(0)
			hyperparameters['track']['search'].pop(0)

		hyperparameters['track']['iteration'].append(iteration+1)
		hyperparameters['track']['value'].append(_value)		
		hyperparameters['track']['grad'].append(_grad)
		hyperparameters['track']['search'].append(_search)
		hyperparameters['track']['alpha'].append(alpha)
		hyperparameters['track']['beta'].append(beta)
		hyperparameters['track']['lambda'].append(hyperparameters['hyperparameters']['lambda'])			
		hyperparameters['track']['size'] += 1

		for callback in hyperparameters['callback']:
			hyperparameters['callback'][callback](state,hyperparameters)

		if iteration%hyperparameters['track']['log'] == 0:			
			# print(hyperparameters['label'])
			# print(state)
			logger.log(50,'%d f(x) = %0.4f'%(iteration,hyperparameters['track']['objective'][-1]))
			print('alpha = ',alpha)
			print('beta = ',beta)			
			print(
				np.linalg.norm(state)/state.size,
				state.max(),state.min(),
				state.reshape(hyperparameters['shapes']['variable'])[0],
				state.reshape(hyperparameters['shapes']['variable'])[-1],
				)
			print()
			print()
		return state



class Adam(OptimizerBase):
	'''
	Adam Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		hyperparameters (dict): optimizer hyperparameters
	'''
	def __init__(self,func,grad=None,hyperparameters={}):

		super().__init__(func,grad,hyperparameters)

		self.Optimizer = getattr(jax.example_libraries.optimizers,self.optimizer)

		self._opt_init,self._opt_update,self._get_params = self.Optimizer(self.learning_rate)

		return

	def opt_init(self,parameters,hyperparameters={}):
		'''
		Initialize optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
			hyperparameters (dict): optimizer hyperparameters						
		Returns:
			state (object): optimizer state
		'''
		return self._opt_init(parameters)

	def opt_update(self,iteration,state,hyperparameters={}):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
			hyperparameters (dict): optimizer hyperparameters
		Returns:
			state (object): optimizer state
		'''

		value,grad = self.opt_step(iteration,state,hyperparameters)

		return self._opt_update(iteration,grad,state)

	def get_params(self,state,hyperparameters={}):
		'''
		Get optimizer parameters with optimizer state
		Args:
			state (object): optimizer state
			hyperparameters (dict): optimizer hyperparameters						
		Returns:
			parameters (object): optimizer parameters
		'''
		return self._get_params(state)