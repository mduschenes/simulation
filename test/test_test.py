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

from src.utils import jit,gradient
from src.utils import array,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product
from src.utils import summation,exponentiation
from src.utils import inner_norm,inner_abs2,inner_real,inner_imag
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_norm,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
from src.utils import eig,qr,einsum
from src.utils import maximum,minimum,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,unique,allclose,is_naninf,is_hermitian,is_unitary
from src.utils import parse,to_string,to_number,scinotation,datatype,slice_size
from src.utils import trotter
from src.utils import pi,e,delim,arrays,scalars
from src.utils import itg,flt,dbl

from src.iterables import getter,setter,permuter,equalizer

from src.parameters import Parameters
from src.operators import Gate
from src.states import State

from src.io import load,dump,join,split
from src.call import call,rm

from src.plot import plot

from src.optimize import Optimizer,Objective,Metric,Callback

from src.quantum import Unitary,Hamiltonian,Observable

# Logging
from src.system import Logger

# name = __name__
# path = os.getcwd()
# file = 'logging.conf'
# conf = os.path.join(path,file)
# file = None #'log.log'
# logger = Logger(name,conf,file=file)



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
		label=hyperparameters['label'],
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

def test_metric(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'],
		system=hyperparameters['system'])

	func = []
	label = model.label()
	hyperparams = hyperparameters['optimize']
	system = model.system

	metric = Metric(label=label,hyperparameters=hyperparams,system=system)

	out = metric(label)

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
		label=hyperparameters['label'],
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


if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	test_class(path,tol)
	# test_data(path,tol)
	# test_grad(path,tol)
	# test_metric(path,tol)
	# test_objective(path,tol)
