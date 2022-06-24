#!/usr/bin/env python

# Import python modules
import pytest
import os,sys,itertools,functools,copy
import time
from time import time as timer
from functools import partial

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp
import jax
import jax.numpy as np
import jax.scipy as sp
import jax.example_libraries.optimizers
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'
# np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})

# Logging
import logging
logger = logging.getLogger(__name__)

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.optimize import Optimizer,Objective

from src.utils import jit,gradient,gradient_finite,gradient_fwd
from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product
from src.utils import summation,exponentiation
from src.utils import inner_abs2,inner_real2,inner_imag2
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_abs2,gradient_inner_real2,gradient_inner_imag2
from src.utils import eigh,qr
from src.utils import maximum,minimum,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,isclose,isnaninf
from src.utils import parse,to_str,to_number,scinotation,datatype,slice_size
from src.utils import pi,e,delim
from src.utils import itg,flt,dbl

from src.dictionary import updater,getter,setter,permuter

from src.parameters import parameterize
from src.operators import operatorize

from src.io import load,dump,path_join,path_split

from src.plot import plot

from src.optimize import Optimizer,Objective

from src.quantum import Unitary,Hamiltonian,Object
from src.quantum import check
from src.functions import functions
from src.main import main



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