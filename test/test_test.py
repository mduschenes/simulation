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
from src.utils import allclose,trace,dot,prng

from src.iterables import getter,setter,permuter,equalizer,namespace

from src.io import load,dump

from src.optimize import Optimizer,Objective,Metric,Callback

from src.system import Dict


def test_metric(path,tol):

	default = None
	settings = load(path,default=default)
	if settings is None:
		raise Exception("settings %s not loaded"%(path))

	settings = Dict(settings)

	model = load(settings.cls.model)
	state = load(settings.cls.state)
	label = load(settings.cls.label)
	callback = load(settings.cls.callback)

	hyperparameters = settings.optimize
	system = settings.system
	seed = prng(**settings.seed)
	func = None
	arguments = ()
	keywords = {}

	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})

	label.__initialize__(state=state)
	model.__initialize__(state=state)

	metric = Metric(state=state,label=label,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system)

	out = metric(label())

	assert allclose(0,out), "Incorrect metric %0.5e"%(out)

	print('Passed')
	
	return


def test_objective(path,tol):

	default = None
	settings = load(path,default=default)
	if settings is None:
		raise Exception("settings %s not loaded"%(path))

	settings = Dict(settings)



	model = load(settings.cls.model)
	state = load(settings.cls.state)
	label = load(settings.cls.label)
	callback = load(settings.cls.callback)

	hyperparameters = settings.optimize
	system = settings.system
	seed = prng(**settings.seed)
	func = None
	arguments = ()
	keywords = {}

	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})
	callback = callback(**{**namespace(callback,model),**settings.callback,**dict(model=model,system=system)})

	label.__initialize__(state=state)
	model.__initialize__(state=state)

	parameters = model.parameters()
	state = model.state()
	label = model(parameters,state=state)

	metric = Metric(state=state,label=label,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system)
	func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system)

	out = func(parameters,state=state)

	assert allclose(0,out), "Incorrect objective %0.5e"%(out)

	print('Passed')

	return

def test_grad(path,tol):

	default = None
	settings = load(path,default=default)
	if settings is None:
		raise Exception("settings %s not loaded"%(path))



	settings = Dict(settings)

	model = load(settings.cls.model)
	state = load(settings.cls.state)
	label = load(settings.cls.label)
	callback = load(settings.cls.callback)

	hyperparameters = settings.optimize
	system = settings.system
	seed = prng(**settings.seed)
	func = None
	arguments = ()
	keywords = {}

	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})

	label.__initialize__(state=state)
	model.__initialize__(state=state)

	parameters = model.parameters()
	state = model.state()

	# grad of unitary
	grad_automatic = model.grad_automatic
	grad_finite = model.grad_finite
	grad_analytical = model.grad_analytical

	index = slice(None)
	print('-----')
	print(grad_automatic(parameters,state)[index])
	print()
	print('-----')
	print()
	print(grad_finite(parameters,state)[index])
	print()
	print('-----')
	print()	
	print(grad_analytical(parameters,state)[index])
	print()
	print('----- ratio -----')
	print()
	print(grad_automatic(parameters,state)[index]/grad_analytical(parameters,state)[index])
	print()
	print('-----')
	print()
	assert allclose(grad_automatic(parameters,state),grad_finite(parameters,state)), "JAX grad != Finite grad"
	assert allclose(grad_automatic(parameters,state),grad_analytical(parameters,state)), "JAX grad != Analytical grad"
	assert allclose(grad_finite(parameters,state),grad_analytical(parameters,state)), "Finite grad != Analytical grad"

	print('Passed')

	return


if __name__ == '__main__':
	path = 'config/settings.json'
	path = 'config/settings.tmp.json'	
	tol = 5e-8 
	test_metric(path,tol)
	test_objective(path,tol)
	test_grad(path,tol)