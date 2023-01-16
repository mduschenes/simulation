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

from src.utils import jit,gradient,hessian
from src.utils import array,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product
from src.utils import summation,exponentiation
from src.utils import inner_norm,inner_abs2,inner_real,inner_imag
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_norm,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
from src.utils import eig,qr,einsum
from src.utils import maximum,minimum,difference,abs,argmax,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,is_array,is_ndarray,isclose,is_naninf
from src.utils import parse,to_string,to_number,scinotation,datatype,slice_size
from src.utils import trotter
from src.utils import pi,e,delim
from src.utils import itg,flt,dbl

from src.iterables import getter,setter,permuter,equalizer

from src.parameters import Parameters
from src.operators import Gate
from src.states import State

from src.io import load,dump,join,split
from src.call import rm

from src.plot import plot

from src.optimize import Optimizer,Objective,Metric,Callback

from src.quantum import Unitary,Hamiltonian

# Logging
from src.system import Logger

# name = __name__
# path = os.getcwd()
# file = 'logging.conf'
# conf = os.path.join(path,file)
# file = None #'log.log'
# logger = Logger(name,conf,file=file)


def test_objective(path,tol):

	hyperparameters = load(path)

	cls = {attr: load(hyperparameters['class'][attr]) for attr in hyperparameters['class']}

	model = cls['model'](**hyperparameters['model'],
			parameters=hyperparameters['parameters'],
			state=hyperparameters['state'],
			noise=hyperparameters['noise'],
			label=hyperparameters['label'],
			system=hyperparameters['system'])

	func = []
	shapes = model.shapes
	label = model.label()
	callback = cls['callback']()
	hyperparams = hyperparameters['optimize']
	system = model.system


	metric = Metric(shapes=shapes,label=label,hyperparameters=hyperparams,system=system)
	func = Objective(model,metric,func=func,callback=callback,hyperparameters=hyperparams,system=system)
	callback = Callback(model,callback,func=func,metric=metric,hyperparameters=hyperparams,system=system)

	parameters = model.parameters()

	# Grad of objective
	grad_jax = func.grad
	grad_finite = gradient(func,mode='finite',tol=tol)
	grad_grad = func.grad
	grad_analytical = func.grad_analytical


	assert allclose(grad_jax(parameters),grad_finite(parameters)), "JAX grad != Finite grad"
	assert allclose(grad_finite(parameters),grad_analytical(parameters)), "Finite grad != Analytical grad"
	assert allclose(grad_finite(parameters),grad_grad(parameters)), "Finite grad != Analytical grad"
	assert allclose(grad_jax(parameters),grad_analytical(parameters)), "JAX grad != Analytical grad"

	return


def test_optimizer(path,tol):

	hyperparameters = load(path)

	cls = {attr: load(hyperparameters['class'][attr]) for attr in hyperparameters['class']}

	model = cls['model'](**hyperparameters['model'],
			parameters=hyperparameters['parameters'],
			state=hyperparameters['state'],
			noise=hyperparameters['noise'],
			label=hyperparameters['label'],
			system=hyperparameters['system'])

	parameters = model.parameters()
	shapes = model.shapes
	label = model.label()
	hyperparams = hyperparameters['optimize']
	system = hyperparameters['system']
	kwargs = {}
	func = []
	callback = cls['callback']()

	metric = Metric(shapes=shapes,label=label,hyperparameters=hyperparams,system=system,**kwargs)
	func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparams,system=system,**kwargs)
	callback = Callback(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparams,system=system,**kwargs)

	optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparams,system=system,**kwargs)

	optimizer(parameters)

	value = optimizer.track['objective'][-1]
	iteration = optimizer.track['iteration'][-1]
	size = min(len(optimizer.track[attr]) for attr in optimizer.track)

	optimizer.reset(clear=True)
	optimizer(parameters)

	value = optimizer.track['objective'][-1]-value
	iteration = optimizer.track['iteration'][-1]-iteration
	size = min(len(optimizer.track[attr]) for attr in optimizer.track)-size

	assert value < 0, "Checkpointed optimizer not re-initialized with value %s"%(value)
	assert iteration == hyperparams['iterations'], "Checkpointed optimizer not re-initialized with iteration %s"%(iteration)
	assert size == hyperparams['iterations'], "Checkpointed optimizer not re-initialized with size %s"%(size)

	return


def test_hessian(path,tol):
	hyperparameters = load(path)

	cls = {attr: load(hyperparameters['class'][attr]) for attr in hyperparameters['class']}

	model = cls['model'](**hyperparameters['model'],
			parameters=hyperparameters['parameters'],
			state=hyperparameters['state'],
			noise=hyperparameters['noise'],
			label=hyperparameters['label'],
			system=hyperparameters['system'])

	parameters = model.parameters()
	shapes = model.shapes
	label = model.label()
	hyperparams = hyperparameters['optimize']
	system = hyperparameters['system']
	kwargs = {}
	func = []
	callback = cls['callback']()

	metric = Metric(shapes=shapes,label=label,hyperparameters=hyperparams,system=system,**kwargs)
	func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparams,system=system,**kwargs)
	callback = Callback(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparams,system=system,**kwargs)


	func = hessian(jit(lambda parameters: metric(model(parameters))))

	out = func(parameters)

	eigs = sort(abs(eig(func(parameters),compute_v=False,hermitian=True)))[::-1]
	eigs = eigs/max(1,maximum(eigs))

	rank = sort(abs(eig(func(parameters),compute_v=False,hermitian=True)))[::-1]
	rank = argmax(abs(difference(rank)/rank[:-1]))+1						

	print(eigs)
	print(rank)

	return




if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	# test_objective(path,tol)
	test_optimizer(path,tol)
	# test_hessian(path,tol)
