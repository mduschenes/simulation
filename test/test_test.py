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

from src.utils import jit,gradient
from src.utils import array,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product
from src.utils import summation,exponentiation
from src.utils import inner_norm,inner_abs2,inner_real,inner_imag
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_norm,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
from src.utils import eig,qr,einsum
from src.utils import maximum,minimum,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,is_array,is_ndarray,isclose,is_naninf
from src.utils import parse,to_string,to_number,scinotation,datatype,slice_size
from src.utils import trotter
from src.utils import pi,e,delim
from src.utils import itg,flt,dbl

from src.iterables import getter,setter,permuter,equalizer

from src.parameters import Parameters
from src.operators import Operator
from src.states import State

from src.io import load,dump,join,split
from src.call import call,rm

from src.plot import plot

from src.optimize import Optimizer,Objective,Metric,Callback

from src.quantum import Unitary,Hamiltonian,Object

# Logging
from src.system import Logger

# name = __name__
# path = os.getcwd()
# file = 'logging.conf'
# conf = os.path.join(path,file)
# file = None #'log.log'
# logger = Logger(name,conf,file=file)


def test_class(path,tol):

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'])

	print(model.noise())
	print(model.state())
	print(model.label())
	model.__functions__(noise={'scale':None},state={'scale':1})
	print('----')
	print(model.noise())
	print(model.state())
	print(model.label())
	model.__functions__(noise={'scale':None})
	print('----')

	print(model.noise())
	print(model.state())
	print(model.label())
	model.__functions__()
	print('----')
	print(model.noise())
	print(model.state())
	print(model.label())
	return


def test_load_dump(path,tol):

	# Set instance
	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'])

	# Set hyperparameters
	hyperparameters['optimize']['track']['alpha'] = []
	hyperparameters['optimize']['track']['alpha'].append(12345)
	hyperparameters['optimize']['attributes']['search']
	hyperparameters['optimize']['attributes']['search'].append([1,2,2,3])
	

	# Dump instance
	model.dump()

	# Set instance
	hyperparameters = load(path)
	new = cls(**hyperparameters['model'])

	new.load()

	types = (dict,list,)
	exceptions = lambda a,b: any(any(e(a) for e in exception) and any(e(b) for e in exception) 
		for exception in [[callable],[is_array,is_ndarray],
							[lambda a: isinstance(a,dict) and ((len(a)==0) or all(callable(a[item]) for item in a))]])
	
	equalizer(hyperparameters,hyperparameters,types=types,exceptions=exceptions)

	return

def test_data(path,tol):

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	hyperparameters['model']['N'] = 2
	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'])

	I = array([[1,0],[0,1]],dtype=model.dtype)
	X = array([[0,1],[1,0]],dtype=model.dtype)
	Y = array([[0,-1j],[1j,0]],dtype=model.dtype)
	Z = array([[1,0],[0,-1]],dtype=model.dtype)
	data = [
		tensorprod(array([X,I])),
		tensorprod(array([I,X])),
		tensorprod(array([Y,I])),
		tensorprod(array([I,Y])),
		tensorprod(array([Z,I])),
		tensorprod(array([I,Z])),
		tensorprod(array([Z,Z])),
		]
	string = ['XI','IX','YI','IY','ZI','IZ','ZZ']

	data = trotter(data,model.p)
	string = trotter(string,model.p)

	for i,(d,o) in enumerate(zip(data,model.data)):
		assert allclose(d,o), "data[%d] incorrect"%(i)

	return


def test_state(path,tol):

	hyperparameters = load(path)

	data = None
	shape = (hyperparameters['model']['D']**hyperparameters['model']['N'],)*2
	hyperparams = hyperparameters['state']
	hyperparams['shape'] = [2,1,-1,-1]
	hyperparams['scale'] = 1
	size = hyperparameters['model']['N']
	samples = True
	dtype = hyperparameters['model']['system']['dtype']
	cls = None

	tol = 1e-20

	state = State(data,shape,hyperparams,size=size,samples=samples,dtype=dtype,cls=cls)

	try:
		eigs = eig(state(),hermitian=True)
		assert (abs(eigs)>=tol).all()
	except TypeError:
		raise

	return

def test_grad(path,tol):

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'])

	func = model

	parameters = model.parameters

	# grad of unitary
	grad_jax = model.grad
	grad_finite = gradient(model,mode='finite',tol=tol)
	grad_analytical = model.grad_analytical

	assert allclose(grad_jax(parameters),grad_finite(parameters)), "JAX grad != Finite grad"
	assert allclose(grad_finite(parameters),grad_analytical(parameters)), "Finite grad != Analytical grad"
	assert allclose(grad_jax(parameters),grad_analytical(parameters)), "JAX grad != Analytical grad"

	return

def test_objective(path,tol):

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'])

	func = []
	shapes = model.shapes
	label = model.label()
	callback = None
	hyperparams = hyperparameters['optimize']

	metric = Metric(shapes=shapes,label=label,hyperparameters=hyperparams)
	func = Objective(model,metric,func=func,callback=callback,hyperparameters=hyperparams)

	parameters = model.parameters

	metric = Metric(shapes=shapes,hyperparameters=hyperparams)

	print(metric(label.conj(),label))
	return

	# Grad of objective
	grad_jax = gradient(func)
	grad_finite = gradient(func,mode='finite',tol=tol)
	grad_grad = func.grad
	grad_analytical = func.grad_analytical

	print(grad_jax(parameters).round(3))
	print()
	print(grad_finite(parameters).round(3))
	print()
	print(grad_grad(parameters).round(3))
	print()
	print(grad_analytical(parameters).round(3))
	print()
	print(grad_analytical(parameters)/grad_jax(parameters))

	return
	assert allclose(grad_jax(parameters),grad_finite(parameters)), "JAX grad != Finite grad"
	assert allclose(grad_finite(parameters),grad_analytical(parameters)), "Finite grad != Analytical grad"
	assert allclose(grad_jax(parameters),grad_analytical(parameters)), "JAX grad != Analytical grad"

	return



def test_model(path,tol):

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'])

	func = jit(model.__call__)

	parameters = model.parameters

	U = func(parameters)

	model.__functions__(noise=None)

	func = jit(model.__call__)

	V = func(parameters)

	model.__functions__(noise=True)

	func = jit(model.__call__)

	W = func(parameters)

	assert allclose(U,V),"Incorrect identity noise"
	assert allclose(U,W),"Incorrect restored noise"

	return

if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	test_class(path,tol)
	# test_grad(path,tol)
	# test_objective(path,tol)
