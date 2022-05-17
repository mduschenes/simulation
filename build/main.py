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
import jax.example_libraries.optimizers
np.set_printoptions(linewidth=1000)#,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import logconfig

conf = 'config/logging.conf'
logger = logconfig(__name__,conf=conf)

from src.quantum import run

from src.utils import jit,array,sin,cos,sigmoid
from src.utils import gradient_sigmoid
from src.utils import pi,e
from src.io import load,dump,path_join,path_split

def bound(a,hyperparameters):
	'''
	Bound array
	Args:
		a (array): Array to bound
		hyperparameters (dict): Hyperparameters for bounds
	Returns:
		out (array): Bounded array
	'''
	return sigmoid(a,hyperparameters['hyperparameters']['sigmoid'])

def gradient_bound(a,hyperparameters):
	'''
	Gradient of bound array
	Args:
		a (array): Array to bound
		hyperparameters (dict): Hyperparameters for bounds
	Returns:
		out (array): Gradient of bounded array
	'''
	return gradient_sigmoid(a,hyperparameters['hyperparameters']['sigmoid'])	

def variables(parameters,hyperparameters,parameter,group):
	'''
	Get variables from parameters, compute features, and variables from features
	Args:
		parameters (array): Array of parameters to compute variables
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for variables
		group (str): Parameter group for variables
	Returns:
		variable (array): variables
	'''
	shape = parameters.shape
	n = shape[-1]

	feature = features(parameters,hyperparameters,parameter,group)

	if parameter in ['xy'] and group in [('x',)]:
		variable = (
			hyperparameters['parameters'][parameter]['scale']*feature[0]*
			cos(2*pi*feature[1])
		)
		variable = (
			hyperparameters['parameters'][parameter]['scale']*
			parameters[:,0:n//2]
		)

	elif parameter in ['xy'] and group in [('y',)]:
		variable = (
			hyperparameters['parameters'][parameter]['scale']*feature[0]*
			sin(2*pi*feature[1])
		)		
		variable = (
			hyperparameters['parameters'][parameter]['scale']*
			parameters[:,n//2:]
		)		

	elif parameter in ['z'] and group in [('z',)]:
		variable = (
			hyperparameters['parameters'][parameter]['scale']*
			feature[0]
		)

	elif parameter in ['zz'] and group in [('zz',)]:
		variable = (
			hyperparameters['parameters'][parameter]['scale']*
			feature[0]
		)


	elif parameter in ['xy'] and group in [('x_0','x_1'),('x_2','x_3')]:
		variable = (
			hyperparameters['parameters'][parameter]['scale']*feature[0]*
			cos(2*pi*feature[1])
		)		
		# variable = (
		# 	hyperparameters['parameters'][parameter]['scale']*
		# 	parameters[:,0:n//2]
		# )

	elif parameter in ['xy'] and group in [('y_0','y_1'),('y_2','y_3')]:
		variable = (
			hyperparameters['parameters'][parameter]['scale']*feature[0]*
			sin(2*pi*feature[1])
		)
		# variable = (
		# 	hyperparameters['parameters'][parameter]['scale']*
		# 	parameters[:,n//2:]
		# )		

	elif parameter in ['z'] and group in [('z',)]:
		variable = (
			hyperparameters['parameters'][parameter]['scale']*
			feature[0]
		)

	elif parameter in ['zz'] and group in [('zz',)]:
		variable = (
			hyperparameters['parameters'][parameter]['scale']*
			feature[0]
		)		

	return variable


def features(parameters,hyperparameters,parameter,group):
	'''
	Get features from parameters
	Args:
		parameters (array): Array of parameters to compute features
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for features
		group (str): Parameter group for features
	Returns:
		feature (array): features
	'''
	shape = parameters.shape
	n = shape[-1]

	if parameter in ['xy'] and group in [('x',)]:
		feature = (
			sigmoid(parameters[:,0::2]),
			sigmoid(parameters[:,1::2])
		)

	elif parameter in ['xy'] and group in [('y',)]:
		feature = (
			sigmoid(parameters[:,0::2]),
			sigmoid(parameters[:,1::2])
		)		

	elif parameter in ['z'] and group in [('z',)]:
		feature = (
			parameters,
		)

	elif parameter in ['zz'] and group in [('zz',)]:
		feature = (
			parameters,
		)

	return feature

def constraints(parameters,hyperparameters,parameter,group):
	'''
	Get constraints from parameters
	Args:
		parameters (array): Array of parameters to compute constraints
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for constraints
		group (str): Parameter group for constraints
	Returns:
		constraints (array): constraints
	'''
	shape = parameters.shape
	n = shape[-1]

	if parameter in ['xy'] and group in [('x',),('y',)]:
		constraint = 0
		# constraint = (
			# (hyperparameters['hyperparameters']['lambda'][0]*bound(
			# 	(hyperparameters['parameters'][parameter]['bounds'][0] - 
			# 	parameters[:,0::2]),
			# 	hyperparameters) +
			# hyperparameters['hyperparameters']['lambda'][1]*bound(
			# 	(hyperparameters['parameters'][parameter]['bounds'][0] - 
			# 	parameters[:,1::2]),
			# 	hyperparameters)
			# ).sum() +				 
			# (sum(
			# 	hyperparameters['hyperparameters']['lambda'][2]*(
			# 	(hyperparameters['parameters'][parameter]['boundaries'][0][i]-
			# 	parameters[i,:]
			# 	)**2)
			# 	for i in hyperparameters['parameters'][parameter]['boundaries'][0])
			# ).sum()
		# )
		# x = (
		# 	parameters[:,0:n//2]**2+
		# 	parameters[:,n//2:]**2
		# 	)**(1/2)

		# constraint = (
		# 	(
		# 	hyperparameters['hyperparameters']['lambda'][1]*bound(
		# 		(-hyperparameters['parameters'][parameter]['bounds'][1] + x),
		# 		hyperparameters
		# 		)
		# 	).sum()
		# 	# +
		# 	# (sum(
		# 	# 	hyperparameters['hyperparameters']['lambda'][2]*(
		# 	# 	(i[1] - x[i[0][0]]))**2
		# 	# 	for i in hyperparameters['parameters'][parameter]['boundaries'])
		# 	# ).sum()
		# )

	elif parameter in ['z'] and group in [('z',)]:
		constraint = 0
	
	elif parameter in ['zz'] and group in [('zz',)]:
		constraint = 0

	return constraint


# @partial(jit,static_argnums=(1,2,3,))
def gradient_constraints(parameters,hyperparameters,parameter,group):
	'''
	Get gradients of constraints from parameters
	Args:
		parameters (array): Array of parameters to compute constraints
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for constraints
		group (str): Parameter group for constraints
	Returns:
		grad (array): gradient of constraints
	'''
	shape = parameters.shape
	n = shape[-1]

	if parameter in ['xy'] and group in [('x',),('y',)]:
		# grad = (
		# 	(hyperparameters['hyperparameters']['lambda'][0]*bound(
		# 		(hyperparameters['parameters'][parameter]['bounds'][0] - 
		# 		parameters[:,0::2]),
		# 		hyperparameters) +
		# 	hyperparameters['hyperparameters']['lambda'][1]*bound(
		# 		(hyperparameters['parameters'][parameter]['bounds'][0] - 
		# 		parameters[:,1::2]),
		# 		hyperparameters)
		# 	).sum() +				 
		# 	(sum(
		# 		hyperparameters['hyperparameters']['lambda'][2]*(
		# 		(i[1]-
		# 		parameters[i[0][0],:]
		# 		)**2)
		# 		for i in hyperparameters['parameters'][parameter]['boundaries'])
		# 	).sum()
		# )
		x = (
			parameters[:,0:n//2]**2+
			parameters[:,n//2:]**2
			)**(1/2)

		grad = np.zeros(parameters.shape)
		_grad = (
			(
			hyperparameters['hyperparameters']['lambda'][1]*gradient_bound(
				(-hyperparameters['parameters'][parameter]['bounds'][1] + x),
				hyperparameters
				)
			)
			# +
			# (sum(
			# 	-2*hyperparameters['hyperparameters']['lambda'][2]*(
			# 	(i[1] - x[i[0][0]]))
			# 	for i in hyperparameters['parameters'][parameter]['boundaries'])
			# ).sum()
		)
	
		grad = jax.lax.dynamic_update_slice(grad,_grad*parameters[:,0:n//2]/x,(1,0))
		grad = jax.lax.dynamic_update_slice(grad,_grad*parameters[:,n//2:]/x,(1,n//2))

		# grad = grad.at[:,0:n//2].set(_grad*parameters[:,0:n//2]/x)
		# grad = grad.at[:,n//2:].set(_grad*parameters[:,n//2:]/x)
		# grad[:,n//2:] = _grad*parameters[:,n//2:]/x
	elif parameter in ['z'] and group in [('z',)]:
		grad = np.zeros(parameters.shape)
	elif parameter in ['zz'] and group in [('zz',)]:
		grad = np.zeros(parameters.shape)

	grad = grad.ravel()

	return grad	


def gradients(parameters,hyperparameters,parameter,group):
	'''
	Get gradient of variables from parameters
	Args:
		parameters (array): Array of parameters to compute variables
		hyperparameters (dict): Hyperparameters for parameters
		parameter (str): Parameter name for variables
		group (str): Parameter group for variables
	Returns:
		grad (array): gradient of variables
	'''	

	#TODO (finish analytic derivatives for variables functions as a matrix of (k,l) shape for k output parameters and l parameters)
	# ie) k = m*r for r = 2N, and l = m*q for q = 2,2*N input phases and amplitudes
	if group in [('x',),('x_0','x_1'),('x_2','x_3'),]:
		variable = hyperparameters['parameters'][parameter]['scale']*(
			parameters[:,hyperparameters['parameters'][parameter]['slice'][group][:n//2]])

	elif group in [('y',),('y_0','y_1'),('y_2','y_3')]:
		variable = hyperparameters['parameters'][parameter]['scale']*(
			parameters[:,hyperparameters['parameters'][parameter]['slice'][group][n//2:]])		

	elif parameter in ['z'] and group in [('z',)]:
		grad = np.zeros(parameters.shape)
	elif parameter in ['zz'] and group in [('zz',)]:
		grad = np.zeros(parameters.shape)
	
	return variable	



def setup(hyperparameters):
	'''
	Setup hyperparameters
	Args:
		hyperparameters (dict): Hyperparameters
	'''

	section = 'parameters'
	updates = {
		'variables':{
			'value':(lambda parameter,hyperparameters: variables),
			'default':(lambda parameter,hyperparameters: {}),
			'conditions': (lambda parameter,hyperparameters: True)
		},
		'features':{
			'value':(lambda parameter,hyperparameters: features),
			'default':(lambda parameter,hyperparameters: {}),
			'conditions': (lambda parameter,hyperparameters: True)
		},		
		'constraints':{
			'value':(lambda parameter,hyperparameters: constraints),
			'default': (lambda parameter,hyperparameters: {}),
			'conditions': (lambda parameter,hyperparameters: True)
		},
		'gradients':{
			'value':(lambda parameter,hyperparameters: gradients),
			'default':(lambda parameter,hyperparameters: {}),
			'conditions': (lambda parameter,hyperparameters: True)
		},			
		'gradient_constraints':{
			'value':(lambda parameter,hyperparameters: gradient_constraints),
			'default': (lambda parameter,hyperparameters: {}),
			'conditions': (lambda parameter,hyperparameters: True)
		},
	}
	for parameter in hyperparameters[section]:
		for attr in updates:
			hyperparameters[section][parameter][attr] = hyperparameters[section][parameter].get(attr,updates[attr]['default'](parameter,hyperparameters))
			if updates[attr]['conditions'](parameter,hyperparameters):
				for group in hyperparameters[section][parameter]['group']:
					# hyperparameters[section][parameter][attr][group] = (lambda parameters,hyperparameters,parameter=parameter,group=group,func=updates[attr]['value'](parameter,hyperparameters): func(parameters,hyperparameters,parameter=parameter,group=group))
					group = tuple(group)
					hyperparameters[section][parameter][attr][group] = (partial(updates[attr]['value'](parameter,hyperparameters),parameter=parameter,group=group))

	return

def main(args):

	nargs = len(args)

	path = args[0] if nargs>0 else None

	hyperparameters = load(path)

	setup(hyperparameters)

	run(hyperparameters)


if __name__ == '__main__':
	main(sys.argv[1:])