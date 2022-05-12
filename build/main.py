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

from src.quantum import run

from src.utils import jit,array,sin,cos,sigmoid
from src.utils import gradient_sigmoid
from src.utils import pi,e
from src.io import load,dump

# Logging
import logging,logging.config
logger = logging.getLogger(__name__)
conf = 'config/logging.conf'
try: 
	logging.config.fileConfig(conf,disable_existing_loggers=False) 
except:
	pass
logger = logging.getLogger(__name__)


# @partial(jit,static_argnums=(1,))
def bound(a,hyperparameters):
	# return 1/(1+np.exp(-eps*a))
	return sigmoid(a,hyperparameters['hyperparameters']['bound'])

# @partial(jit,static_argnums=(1,))
def gradient_bound(a,hyperparameters):
	# return 1/(1+np.exp(-eps*a))
	return gradient_sigmoid(a,hyperparameters['hyperparameters']['bound'])	

# @partial(jit,static_argnums=(1,2,3,))
def params(parameters,hyperparameters,parameter,group):

	indices = hyperparameters['parameters'][parameter]['slice'][group]
	n = len(indices)

	if parameter in ['xy'] and group in [('x',)]:
		param = (
			hyperparameters['parameters'][parameter]['scale']*
			parameters[:,indices[0::2]]*
			cos(2*pi*parameters[:,indices[1::2]])
		)
		# param = (
		# 	hyperparameters['parameters'][parameter]['scale']*
		# 	parameters[:,indices[:n//2]]
		# )

	elif parameter in ['xy'] and group in [('y',)]:
		param = (
			hyperparameters['parameters'][parameter]['scale']*
			parameters[:,indices[0::2]]*
			sin(2*pi*parameters[:,indices[1::2]])
		)		
		# param = (
		# 	hyperparameters['parameters'][parameter]['scale']*
		# 	parameters[:,indices[n//2:]]
		# )		

	elif parameter in ['z'] and group in [('z',)]:
		param = (
			hyperparameters['parameters'][parameter]['scale']*
			parameters[:,indices]
		)

	elif parameter in ['zz'] and group in [('zz',)]:
		param = (
			hyperparameters['parameters'][parameter]['scale']*
			parameters[:,indices]
		)

	return param

# @partial(jit,static_argnums=(1,2,3,))
def constraints(parameters,hyperparameters,parameter,group):

	indices = hyperparameters['parameters'][parameter]['slice'][group]
	slices = slice(1,-1)
	n = len(indices)

	if parameter in ['xy'] and group in [('x',),('y',)]:
		constraint = (
			(hyperparameters['hyperparameters']['lambda'][0]*bound(
				(hyperparameters['parameters'][parameter]['bounds'][0] - 
				parameters[:,indices[0::2]]),
				hyperparameters) +
			hyperparameters['hyperparameters']['lambda'][1]*bound(
				(hyperparameters['parameters'][parameter]['bounds'][0] - 
				parameters[:,indices[1::2]]),
				hyperparameters)
			).sum() +				 
			(sum(
				hyperparameters['hyperparameters']['lambda'][2]*(
				(hyperparameters['parameters'][parameter]['boundaries'][i]-
				parameters[i,indices]
				)**2)
				for i in hyperparameters['parameters'][parameter]['boundaries'])
			).sum()
		)
		# x = (
		# 	parameters[:,indices[:n//2]]**2+
		# 	parameters[:,indices[n//2:]]**2
		# 	)**(1/2)

		# constraint = (
		# 	(
		# 	hyperparameters['hyperparameters']['lambda'][1]*bound(
		# 		(-hyperparameters['parameters'][parameter]['bounds'][1] + x[slices]),
		# 		hyperparameters
		# 		)
		# 	).sum()
		# 	# +
		# 	# (sum(
		# 	# 	hyperparameters['hyperparameters']['lambda'][2]*(
		# 	# 	(hyperparameters['parameters'][parameter]['boundaries'][i] - x[i]))**2
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

	indices = hyperparameters['parameters'][parameter]['slice'][group]
	slices = slice(1,-1)	
	n = len(indices)

	if parameter in ['xy'] and group in [('x',),('y',)]:
		# grad = (
		# 	(hyperparameters['hyperparameters']['lambda'][0]*bound(
		# 		(hyperparameters['parameters'][parameter]['bounds'][0] - 
		# 		parameters[:,indices[0::2]]),
		# 		hyperparameters) +
		# 	hyperparameters['hyperparameters']['lambda'][1]*bound(
		# 		(hyperparameters['parameters'][parameter]['bounds'][0] - 
		# 		parameters[:,indices[1::2]]),
		# 		hyperparameters)
		# 	).sum() +				 
		# 	(sum(
		# 		hyperparameters['hyperparameters']['lambda'][2]*(
		# 		(hyperparameters['parameters'][parameter]['boundaries'][i]-
		# 		parameters[i,indices]
		# 		)**2)
		# 		for i in hyperparameters['parameters'][parameter]['boundaries'])
		# 	).sum()
		# )
		x = (
			parameters[:,indices[:n//2]]**2+
			parameters[:,indices[n//2:]]**2
			)**(1/2)

		grad = np.zeros(parameters.shape)
		_grad = (
			(
			hyperparameters['hyperparameters']['lambda'][1]*gradient_bound(
				(-hyperparameters['parameters'][parameter]['bounds'][1] + x[slices]),
				hyperparameters
				)
			)
			# +
			# (sum(
			# 	-2*hyperparameters['hyperparameters']['lambda'][2]*(
			# 	(hyperparameters['parameters'][parameter]['boundaries'][i] - x[i]))
			# 	for i in hyperparameters['parameters'][parameter]['boundaries'])
			# ).sum()
		)
	
		grad = jax.lax.dynamic_update_slice(grad,_grad*parameters[slices][:,indices[:n//2]]/x[slices],(1,indices[0]))
		grad = jax.lax.dynamic_update_slice(grad,_grad*parameters[slices][:,indices[n//2:]]/x[slices],(1,indices[n//2]))

		# grad = grad.at[slices].at[:,indices[:n//2]].set(_grad*parameters[slices][:,indices[:n//2]]/x[slices])
		# grad = grad.at[slices].at[:,indices[n//2:]].set(_grad*parameters[slices][:,indices[n//2:]]/x[slices])
		# grad[slices][:,indices[n//2:]] = _grad*parameters[slices][:,indices[n//2:]]/x[slices]
	elif parameter in ['z'] and group in [('z',)]:
		grad = np.zeros(parameters.shape)
	elif parameter in ['zz'] and group in [('zz',)]:
		grad = np.zeros(parameters.shape)

	grad = grad.ravel()

	return grad	


# @partial(jit,static_argnums=(1,2,3,))
def grads(parameters,hyperparameters,parameter,group):

	if group in [('x',)]:
		param = hyperparameters['parameters'][parameter]['scale']*(
			parameters[:,hyperparameters['parameters'][parameter]['slice'][group][:n//2]])

	elif group in [('y',)]:
		param = hyperparameters['parameters'][parameter]['scale']*(
			parameters[:,hyperparameters['parameters'][parameter]['slice'][group][n//2:]])		

	elif parameter in ['z'] and group in [('z',)]:
		grad = np.zeros(parameters.shape)
	elif parameter in ['zz'] and group in [('zz',)]:
		grad = np.zeros(parameters.shape)
	
	return param	




def main(args):

	nargs = len(args)

	path = args[0] if nargs>0 else None

	settings = load(path)

	hyperparameters = {}

	hyperparameters.update(settings)

	updates = {
		'label': {
			'value': (lambda hyperparameters: hyperparameters['hyperparameters']['label']),
			'default': None,
			'conditions': (lambda hyperparameters: not hyperparameters.get('label'))				
		},
	}			
	for attr in updates:						
		hyperparameters[attr] = hyperparameters.get(attr,updates[attr]['default'])

	if updates[attr]['conditions'](hyperparameters):
		for attr in updates:
			hyperparameters[attr] = updates[attr]['value'](hyperparameters)


	updates = {
		'runs': {
			'value': (lambda hyperparameters: list(range(hyperparameters['hyperparameters']['runs']))),
			'default': 1,
			'conditions': (lambda hyperparameters: not isinstance(hyperparameters['hyperparameters']['runs'],(list,tuple,array)))
		},
	}			
	for attr in updates:						
		hyperparameters['hyperparameters'][attr] = hyperparameters['hyperparameters'].get(attr,updates[attr]['default'])

	if updates[attr]['conditions'](hyperparameters):
		for attr in updates:
			hyperparameters['hyperparameters'][attr] = updates[attr]['value'](hyperparameters)


	updates = {
		'group': {
			'value': (lambda parameter,hyperparameters: [tuple(group) for group in hyperparameters['parameters'][parameter]['group']]),
			'default': [],
			'conditions': (lambda parameter,hyperparameters: True)				
		},
	}			
	for parameter in hyperparameters['parameters']:
		for attr in updates:						
			hyperparameters['parameters'][parameter][attr] = hyperparameters['parameters'][parameter].get(attr,updates[attr]['default'])

		if updates[attr]['conditions'](parameter,hyperparameters):
			for attr in updates:
				hyperparameters['parameters'][parameter][attr] = updates[attr]['value'](parameter,hyperparameters)

	updates = {
		'func':{
			'func':params,
			'default':{},
			'conditions': (lambda parameter,hyperparameters: True)
		},
		'constraints':{
			'func':constraints,
			'default':{},
			'conditions': (lambda parameter,hyperparameters: True)
		},
		'gradient_constraints':{
			'func':gradient_constraints,
			'default':{},
			'conditions': (lambda parameter,hyperparameters: True)
		},
	}
	for parameter in hyperparameters['parameters']:
		for attr in updates:
			hyperparameters['parameters'][parameter][attr] = hyperparameters['parameters'][parameter].get(attr,updates[attr]['default'])
		if updates[attr]['conditions'](parameter,hyperparameters):
			for attr in updates:
				for group in hyperparameters['parameters'][parameter]['group']:
					hyperparameters['parameters'][parameter][attr][group] = (lambda parameters,hyperparameters,parameter=parameter,group=group,func=updates[attr]['func']: func(parameters,hyperparameters,parameter=parameter,group=group))

	updates = {
		'locality': {
			'value':hyperparameters['hyperparameters']['locality'],
			'default':None,
			'conditions': (lambda parameter,hyperparameters: hyperparameters['parameters'][parameter]['category'] in ['variable'])
		},
	}
	for parameter in hyperparameters['parameters']:
		for attr in updates:
			hyperparameters['parameters'][parameter][attr] = hyperparameters['parameters'][parameter].get(attr,updates[attr]['default'])
		if updates[attr]['conditions'](parameter,hyperparameters):
			for attr in updates:				
				hyperparameters['parameters'][parameter][attr] = updates[attr]['value']

	run(hyperparameters)


if __name__ == '__main__':
	main(sys.argv[1:])