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

# @partial(jit,static_argnums=(1,))
def bound(a,hyperparameters):
	# return 1/(1+np.exp(-eps*a))
	return sigmoid(a,hyperparameters['hyperparameters']['sigmoid'])

# @partial(jit,static_argnums=(1,))
def gradient_bound(a,hyperparameters):
	# return 1/(1+np.exp(-eps*a))
	return gradient_sigmoid(a,hyperparameters['hyperparameters']['sigmoid'])	

# @partial(jit,static_argnums=(1,2,3,))
def params(parameters,hyperparameters,parameter,group):

	indices = hyperparameters['parameters'][parameter]['slice'][group]
	n = len(indices)

	if parameter in ['xy'] and group in [('x',)]:
		param = (
			hyperparameters['parameters'][parameter]['scale']*
			sigmoid(parameters[:,indices[0::2]])*
			cos(2*pi*sigmoid(parameters[:,indices[1::2]]))
		)
		# param = (
		# 	hyperparameters['parameters'][parameter]['scale']*
		# 	parameters[:,indices[:n//2]]
		# )

	elif parameter in ['xy'] and group in [('y',)]:
		param = (
			hyperparameters['parameters'][parameter]['scale']*
			sigmoid(parameters[:,indices[0::2]])*
			sin(2*pi*sigmoid(parameters[:,indices[1::2]]))
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
			# (hyperparameters['hyperparameters']['lambda'][0]*bound(
			# 	(hyperparameters['parameters'][parameter]['bounds'][0] - 
			# 	parameters[:,indices[0::2]]),
			# 	hyperparameters) +
			# hyperparameters['hyperparameters']['lambda'][1]*bound(
			# 	(hyperparameters['parameters'][parameter]['bounds'][0] - 
			# 	parameters[:,indices[1::2]]),
			# 	hyperparameters)
			# ).sum() +				 
			(sum(
				hyperparameters['hyperparameters']['lambda'][2]*(
				(i[1]-
				parameters[i[0][0],indices]
				)**2)
				for i in hyperparameters['parameters'][parameter]['constants'])
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
		# 	# 	(i[1] - x[i[0][0]]))**2
		# 	# 	for i in hyperparameters['parameters'][parameter]['constants'])
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
		# 		(i[1]-
		# 		parameters[i[0][0],indices]
		# 		)**2)
		# 		for i in hyperparameters['parameters'][parameter]['constants'])
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
			# 	(i[1] - x[i[0][0]]))
			# 	for i in hyperparameters['parameters'][parameter]['constants'])
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
	#TODO (finish analytic derivatives for params functions as a matrix of (k,l) shape for k output parameters and l parameters)
	# ie) k = m*r for r = 2N, and l = m*q for q = 2,2*N input phases and amplitudes
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



def setup(hyperparameters):

	section = None
	updates = {
		'label': {
			'value': (lambda hyperparameters: hyperparameters['hyperparameters']['label']),
			'default': (lambda hyperparameters: None),
			'conditions': (lambda hyperparameters: hyperparameters['hyperparameters'].get('label') is not None)				
		},
	}			
	for attr in updates:								
		hyperparameters[attr] = hyperparameters.get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[attr] = updates[attr]['value'](hyperparameters)


	section = 'sys'
	updates = {
		'path': {
			'value': (lambda hyperparameters: 	{
				attr: path_join(hyperparameters[section]['directory'][attr],
								 '.'.join([hyperparameters[section]['file'][attr]]) if attr in ['data','plot'] else hyperparameters[section]['file'][attr],
								 ext=hyperparameters[section]['ext'][attr])
						if isinstance(hyperparameters[section]['file'][attr],str) else
						{i: path_join(hyperparameters[section]['directory'][attr][i],
								 '.'.join([hyperparameters[section]['file'][attr][i]]) if attr in ['data','plot'] else hyperparameters[section]['file'][attr][i],							 
								 ext=hyperparameters[section]['ext'][attr][i])
						for i in hyperparameters[section]['file'][attr]}
				for attr in hyperparameters[section]['file']			 
			}),
			'default': (lambda hyperparameters: None),
			'conditions': (lambda hyperparameters: hyperparameters[section].get('path') is None)
		},		
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)



	section = 'model'
	updates = {
		'tau': {
			'value': (lambda hyperparameters: hyperparameters[section]['tau']/hyperparameters['hyperparameters']['scale']),
			'default': (lambda hyperparameters: 1),
			'conditions': (lambda hyperparameters: True)
		},		
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)


	section = 'hyperparameters'
	updates = {
		'runs': {
			'value': (lambda hyperparameters: list(range(hyperparameters[section]['runs']))),
			'default': (lambda hyperparameters: 1),
			'conditions': (lambda hyperparameters: not isinstance(hyperparameters[section]['runs'],(list,tuple,array)))
		},
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)


	section = 'parameters'
	updates = {
		'group': {
			'value': (lambda parameter,hyperparameters: [tuple(group) for group in hyperparameters[section][parameter]['group']]),
			'default': (lambda parameter,hyperparameters: []),
			'conditions': (lambda parameter,hyperparameters: True)				
		},
		'scale': {
			'value': (lambda parameter,hyperparameters: hyperparameters['hyperparameters']['scale']*hyperparameters[section][parameter]['scale']),
			'default': (lambda parameter,hyperparameters: 1),
			'conditions': (lambda parameter,hyperparameters: True)						
		},
		'locality': {
			'value':(lambda parameter,hyperparameters: hyperparameters['hyperparameters']['locality']),
			'default':(lambda parameter,hyperparameters: None),
			'conditions': (lambda parameter,hyperparameters: hyperparameters['parameters'][parameter]['category'] in ['variable'])
		},		
	}			
	for parameter in hyperparameters[section]:
		for attr in updates:						
			hyperparameters[section][parameter][attr] = hyperparameters[section][parameter].get(attr,updates[attr]['default'](parameter,hyperparameters))
			if updates[attr]['conditions'](parameter,hyperparameters):
				hyperparameters[section][parameter][attr] = updates[attr]['value'](parameter,hyperparameters)


	section = 'parameters'
	updates = {
		'func':{
			'value':(lambda parameter,hyperparameters: params),
			'default':(lambda parameter,hyperparameters: {}),
			'conditions': (lambda parameter,hyperparameters: True)
		},
		'constraints':{
			'value':(lambda parameter,hyperparameters: constraints),
			'default': (lambda parameter,hyperparameters: {}),
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
					hyperparameters[section][parameter][attr][group] = partial(updates[attr]['value'](parameter,hyperparameters),parameter=parameter,group=group)

	return

def main(args):

	nargs = len(args)

	path = args[0] if nargs>0 else None

	hyperparameters = load(path)

	setup(hyperparameters)

	run(hyperparameters)


if __name__ == '__main__':
	main(sys.argv[1:])