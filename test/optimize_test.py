#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy
	
# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import gradient
from src.utils import allclose


from src.iterables import getter,setter,permuter,equalizer

from src.io import load,dump,join,split
from src.call import rm

from src.plot import plot

from src.optimize import Optimizer,Objective,Metric,Callback

from src.iterables import namespace
from src.system import Dict

# Logging
from src.system import Logger

# name = __name__
# path = os.getcwd()
# file = 'logging.conf'
# conf = os.path.join(path,file)
# file = None #'log.log'
# logger = Logger(name,conf,file=file)


def test_objective(path,tol):

	settings = load(path)

	settings = Dict(settings)

	model = load(settings.cls.model)
	state = load(settings.cls.state)
	label = load(settings.cls.label)
	callback = load(settings.cls.callback)

	hyperparameters = settings.optimize
	system = settings.system
	func = None
	kwargs = dict(verbose=True)

	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})
	callback = callback(**{**namespace(callback,model),**settings.callback,**dict(model=model,system=system)})

	label.init(state=state)
	model.init(state=state)

	metric = Metric(state=state,label=label,hyperparameters=hyperparameters,system=system)
	func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system)
	callback = Callback(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system)

	parameters = model.parameters()

	# Grad of objective
	grad_automatic = func.grad_automatic
	grad_finite = func.grad_finite
	grad_analytical = func.grad_analytical

	print(grad_automatic(parameters))
	print()
	print(grad_finite(parameters))
	print()
	print(grad_analytical(parameters)/grad_automatic(parameters))

	assert allclose(grad_automatic(parameters),grad_finite(parameters)), "JAX grad != Finite grad"
	assert allclose(grad_finite(parameters),grad_analytical(parameters)), "Finite grad != Analytical grad"
	assert allclose(grad_automatic(parameters),grad_analytical(parameters)), "JAX grad != Analytical grad"

	print('Passed')

	return

@pytest.mark.filterwarnings(r"ignore:Rounding errors prevent the line search from converging")
@pytest.mark.filterwarnings(r"ignore:The line search algorithm did not converge")
def test_optimizer(path,tol):

	settings = load(path)

	settings = Dict(settings)

	model = load(settings.cls.model)
	state = load(settings.cls.state)
	label = load(settings.cls.label)
	callback = load(settings.cls.callback)

	hyperparameters = settings.optimize
	system = settings.system
	func = None
	kwargs = dict(verbose=True,cleanup=True)

	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})
	callback = callback(**{**namespace(callback,model),**settings.callback,**dict(model=model,system=system)})

	label.init(state=state)
	model.init(state=state)

	metric = Metric(state=state,label=label,hyperparameters=hyperparameters,system=system)
	func = Objective(model,metric,func=func,callback=callback,hyperparameters=hyperparameters,system=system)
	callback = Callback(model,callback,func=func,metric=metric,hyperparameters=hyperparameters,system=system)

	parameters = model.parameters()
	optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparameters,system=system,**kwargs)

	parameters = optimizer(parameters)

	if all(isinstance(optimizer.data[attr],list) and optimizer.data[attr] for attr in optimizer.data):
		value = optimizer.data['objective'][-1]
		iteration = optimizer.data['iteration'][-1]
		size = min(len(optimizer.data[attr]) for attr in optimizer.data)

		optimizer.reset(clear=True)
		optimizer(parameters)

		value = optimizer.data['objective'][-1]-value
		iteration = optimizer.data['iteration'][-1]-iteration
		size = min(len(optimizer.data[attr]) for attr in optimizer.data)-size

		eps = 1e-10

		assert (abs(value) < eps) or value < 0, "Checkpointed optimizer not re-initialized with value %s"%(value)
		assert iteration == hyperparameters['iterations'], "Checkpointed optimizer not re-initialized with iteration %s"%(iteration)
		assert size == hyperparameters['iterations'], "Checkpointed optimizer not re-initialized with size %s"%(size)

	paths = [optimizer.cwd]
	execute = True
	verbose = True
	for path in paths:
		rm(path,execute=execute,verbose=verbose)

	print('Passed')

	return




if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	# test_objective(path,tol)
	test_optimizer(path,tol)
