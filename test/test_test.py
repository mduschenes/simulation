#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools
from copy import deepcopy as deepcopy
	
# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import gradient
from src.utils import namespace
from src.utils import allclose

from src.iterables import getter,setter,permuter,equalizer

from src.io import load,dump

from src.optimize import Optimizer,Objective,Metric,Callback


def test_metric(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

	cls = {attr: load(hyperparameters['class'][attr]) for attr in hyperparameters.get('class',{})}

	model,label,callback = cls.pop('model'),cls.pop('label'),cls.pop('callback')

	hyperparams = hyperparameters.get('optimize',{})
	system = hyperparameters.get('system',{})

	model = model(**{**hyperparameters.get('model',{}),**{attr: hyperparameters.get(attr) for attr in cls},**dict(system=system)})
	label = label(**{**namespace(label,model),**hyperparameters.get('label',{}),**dict(model=model,system=system)})
	callback = callback(**{**namespace(callback,model),**hyperparameters.get('callback',{}),**dict(model=model,system=system)})

	metric = Metric(label=label,hyperparameters=hyperparams,system=system)

	out = metric(label())

	assert allclose(0,out), "Incorrect metric %0.5e"%(out)
	
	return


def test_objective(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		system=hyperparameters['system'])

	parameters = model.parameters()
	label = model(parameters)
	func = []
	callback = None
	hyperparams = hyperparameters['optimize']
	system = model.system

	metric = Metric(label=label,hyperparameters=hyperparams,system=system)
	func = Objective(model,metric,func=func,callback=callback,hyperparameters=hyperparams,system=system)

	out = func(parameters)

	assert allclose(0,out), "Incorrect objective %0.5e"%(out)

	return

def test_grad(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		system=hyperparameters['system'])

	func = model

	parameters = model.parameters()

	# grad of unitary
	grad_jax = model.grad
	grad_finite = gradient(model,mode='finite',tol=tol)
	grad_analytical = model.grad_analytical

	assert allclose(grad_jax(parameters),grad_finite(parameters)), "JAX grad != Finite grad"
	assert allclose(grad_finite(parameters),grad_analytical(parameters)), "Finite grad != Analytical grad"
	assert allclose(grad_jax(parameters),grad_analytical(parameters)), "JAX grad != Analytical grad"

	return


if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	# test_metric(path,tol)
	# test_objective(path,tol)
	test_grad(path,tol)