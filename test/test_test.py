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

from src.utils import jit,gradient,gradient_finite,gradient_fwd
from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product
from src.utils import summation,exponentiation
from src.utils import inner_abs2,inner_real2,inner_imag2
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_abs2,gradient_inner_real2,gradient_inner_imag2
from src.utils import eigh,qr
from src.utils import maximum,minimum,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,isarray,isclose,isnaninf
from src.utils import parse,to_str,to_number,scinotation,datatype,slice_size
from src.utils import pi,e,delim
from src.utils import itg,flt,dbl

from src.dictionary import updater,getter,setter,permuter,equalizer

from src.parameters import parameterize
from src.operators import operatorize

from src.io import load,dump,path_join,path_split

from src.plot import plot

from src.optimize import Optimizer,Objective

from src.quantum import Unitary,Hamiltonian,Object
from src.quantum import check,trotter

from src.functions import functions

from src.main import main

# Logging
from src.utils import logconfig
conf = 'config/logging.conf'
logger = logconfig(__name__,conf=conf)

def test_functions(path,tol):

	hyperparameters = functions(path)

	return

def test_check(path,tol):

	hyperparameters = functions(path)

	check(hyperparameters)

	return

def test_unitary(path,tol):

	hyperparameters = functions(path)
	check(hyperparameters)
	obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	return


def test_load_dump(path,tol):

	# Set instance
	hyperparameters = functions(path)
	check(hyperparameters)
	obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	# Set hyperparameters
	obj.hyperparameters['optimize']['track']['alpha'].append(12345)
	obj.hyperparameters['optimize']['attributes']['search'].append([1,2,2,3])
	

	# Dump instance
	obj.dump()

	# Set instance
	hyperparameters = functions(path)
	check(hyperparameters)
	new = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	new.load()

	exceptions = lambda a,b: (callable(a) and callable(b)) or (isarray(a) and isarray(b))
	equalizer(obj.hyperparameters,new.hyperparameters,exceptions=exceptions)

	return

def test_data(path,tol):

	hyperparameters = functions(path)
	check(hyperparameters)
	obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	I = array([[1,0],[0,1]],dtype=obj.dtype)
	X = array([[0,1],[1,0]],dtype=obj.dtype)
	Y = array([[0,-1j],[1j,0]],dtype=obj.dtype)
	Z = array([[1,0],[0,-1]],dtype=obj.dtype)
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

	data = trotter(data,obj.p)
	string = trotter(string,obj.p)

	for i in range(obj.size):
		assert allclose(data[i],obj.data[i]), "data[%d] incorrect"%(i)

	return

def test_derivative(path,tol):

	hyperparameters = functions(path)

	check(hyperparameters)

	obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	func = obj.__func__

	parameters = obj.parameters

	# Derivative of unitary
	derivative_jax = gradient_fwd(obj)
	derivative_finite = gradient_finite(obj,tol=tol)
	derivative_analytical = obj.__derivative__

	assert allclose(derivative_jax(parameters),derivative_finite(parameters)), "JAX derivative != Finite derivative"
	assert allclose(derivative_finite(parameters),derivative_analytical(parameters)), "Finite derivative != Analytical derivative"
	assert allclose(derivative_jax(parameters),derivative_analytical(parameters)), "JAX derivative != Analytical derivative"

	return

def test_grad(path,tol):

	hyperparameters = functions(path)
	check(hyperparameters)
	obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	func = obj.__func__

	parameters = obj.parameters

	# Grad of objective
	grad_jax = gradient(func)
	grad_finite = gradient_finite(func,tol=tol)
	grad_analytical = obj.__grad__

	assert allclose(grad_jax(parameters),grad_finite(parameters)), "JAX grad != Finite grad"
	assert allclose(grad_finite(parameters),grad_analytical(parameters)), "Finite grad != Analytical grad"
	assert allclose(grad_jax(parameters),grad_analytical(parameters)), "JAX grad != Analytical grad"

	return


def test_main(path,tol):
	main([path])
	return