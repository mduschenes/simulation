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
from src.utils import allclose

from src.iterables import getter,setter,permuter,equalizer,namespace

from src.io import load,dump

from src.optimize import Optimizer,Objective,Metric,Callback

from src.system import Dict


def test_metric(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise Exception("Hyperparameters %s not loaded"%(path))

	hyperparameters = Dict(hyperparameters)

	model = load(hyperparameters.cls.model)
	label = load(hyperparameters.cls.label)
	callback = load(hyperparameters.cls.callback)

	hyperparams = hyperparameters.optimize
	system = hyperparameters.system

	model = model(**{**hyperparameters.model,**dict(parameters=hyperparameters.parameters,state=hyperparameters.state,noise=hyperparameters.noise),**dict(system=system)})
	label = label(**{**namespace(label,model),**hyperparameters.label,**dict(model=model,system=system)})
	callback = callback(**{**namespace(callback,model),**hyperparameters.callback,**dict(model=model,system=system)})

	metric = Metric(label=label,hyperparameters=hyperparams,system=system)

	out = metric(label())

	assert allclose(0,out), "Incorrect metric %0.5e"%(out)
	
	return


def test_objective(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise Exception("Hyperparameters %s not loaded"%(path))

	hyperparameters = Dict(hyperparameters)

	model = load(hyperparameters.cls.model)

	model = model(**hyperparameters.model,
		parameters=hyperparameters.parameters,
		state=hyperparameters.state,
		noise=hyperparameters.noise,
		system=hyperparameters.system)

	parameters = model.parameters()
	label = model(parameters)
	func = []
	callback = None
	hyperparams = hyperparameters.optimize
	system = hyperparameters.system

	metric = Metric(label=label,hyperparameters=hyperparams,system=system)
	func = Objective(model,metric,func=func,callback=callback,hyperparameters=hyperparams,system=system)

	out = func(parameters)

	assert allclose(0,out), "Incorrect objective %0.5e"%(out)

	return

def test_grad(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise Exception("Hyperparameters %s not loaded"%(path))

	hyperparameters = Dict(hyperparameters)

	model = load(hyperparameters.cls.model)
	from src.quantum import Channel as model

	label = load(hyperparameters.cls.label)
	callback = load(hyperparameters.cls.callback)

	hyperparams = hyperparameters.optimize
	system = hyperparameters.system

	model = model(**{**hyperparameters.model,**dict(parameters=hyperparameters.parameters,state=hyperparameters.state,noise=hyperparameters.noise),**dict(system=system)})

	func = model

	parameters = model.parameters()

	print(model(parameters))

	# grad of unitary
	grad_automatic = model.grad_automatic
	grad_finite = model.grad_finite
	grad_analytical = model.grad_analytical

	assert allclose(grad_automatic(parameters),grad_finite(parameters)), "JAX grad != Finite grad"
	assert allclose(grad_finite(parameters),grad_analytical(parameters)), "Finite grad != Analytical grad"
	assert allclose(grad_automatic(parameters),grad_analytical(parameters)), "JAX grad != Analytical grad"

	print('Passed')

	return


if __name__ == '__main__':
	path = 'config/settings.json'
	path = 'config/settings.test.json'
	tol = 5e-8 
	# test_metric(path,tol)
	# test_objective(path,tol)
	test_grad(path,tol)