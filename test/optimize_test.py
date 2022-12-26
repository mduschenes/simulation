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
from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
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

from src.dictionary import updater,getter,setter,permuter,equalizer

from src.parameters import parameterize
from src.operators import operatorize
from src.states import stateize

from src.io import load,dump,join,split
from src.call import rm

from src.plot import plot

from src.optimize import Optimizer,Objective,Metric,Callback

from src.quantum import Unitary,Hamiltonian,Object

# Logging
from src.system import Logger

name = __name__
path = os.getcwd()
file = 'logging.conf'
conf = os.path.join(path,file)
file = None #'log.log'
logger = Logger(name,conf,file=file)


def test_optimize(path,tol):

	return

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],hyperparameters=hyperparameters)

	func = []
	shapes = model.shapes
	label = model.label
	callback = None
	hyperparams = hyperparameters['optimize']

	metric = Metric(shapes=shapes,label=label,optimize=None,hyperparameters=hyperparams)
	func = Objective(model,func,callback=callback,metric=metric,hyperparameters=hyperparams)
	callback = Callback(model,func,callback=callback,metric=metric,hyperparameters=hyperparams)

	parameters = model.parameters

	# Grad of objective
	grad_jax = func.grad
	grad_finite = gradient(func,mode='finite',tol=tol)
	grad_analytical = func.grad_analytical


	print(grad_jax(parameters).round(3))
	print()
	print(grad_analytical(parameters).round(3))
	print()
	print(-gradient_inner_abs2(model(parameters),label,model.grad_analytical(parameters)).round(3)/model.n**2)
	return
	assert allclose(grad_jax(parameters),grad_finite(parameters)), "JAX grad != Finite grad"
	assert allclose(grad_finite(parameters),grad_analytical(parameters)), "Finite grad != Analytical grad"
	assert allclose(grad_jax(parameters),grad_analytical(parameters)), "JAX grad != Analytical grad"

	return



if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	test_optimize(path,tol)