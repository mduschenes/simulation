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
from src.utils import inner_abs2,inner_real,inner_imag
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
from src.utils import eig,qr,einsum
from src.utils import maximum,minimum,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,is_array,is_ndarray,isclose,is_naninf
from src.utils import parse,to_str,to_number,scinotation,datatype,slice_size
from src.utils import trotter
from src.utils import pi,e,delim
from src.utils import itg,flt,dbl

from src.dictionary import updater,getter,setter,permuter,equalizer

from src.parameters import parameterize
from src.operators import operatorize
from src.states import stateize

from src.io import load,dump,join,split

from src.plot import plot

from src.optimize import Optimizer,Objective

from src.quantum import Unitary,Hamiltonian,Object

from src.main import main

# Logging
from src.system import Logger

name = __name__
path = os.getcwd()
file = 'logging.conf'
conf = os.path.join(path,file)
file = None #'log.log'
logger = Logger(name,conf,file=file)


def test_unitary(path,tol):

	hyperparameters = load(path)

	obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	return


def test_load_dump(path,tol):



	# Set instance
	hyperparameters = load(path)

	obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	# Set hyperparameters
	obj.hyperparameters['optimize']['track']['alpha'].append(12345)
	obj.hyperparameters['optimize']['attributes']['search'].append([1,2,2,3])
	

	# Dump instance
	obj.dump()

	# Set instance
	hyperparameters = load(path)
	new = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	new.load()

	types = (dict,list,)
	exceptions = lambda a,b: any(any(e(a) for e in exception) and any(e(b) for e in exception) 
		for exception in [[callable],[is_array,is_ndarray],
							[lambda a: isinstance(a,dict) and ((len(a)==0) or all(callable(a[item]) for item in a))]])
	
	equalizer(obj.hyperparameters,new.hyperparameters,types=types,exceptions=exceptions)

	return

def test_data(path,tol):

	hyperparameters = load(path)

	obj = Unitary(**hyperparameters['data'],**{**hyperparameters['model'],'N':2},hyperparameters=hyperparameters)

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

	for i,(d,o) in enumerate(zip(data,obj.data)):
		assert allclose(d,o), "data[%d] incorrect"%(i)

	return


def test_state(path,tol):

	hyperparameters = load(path)


	data = None
	shape = (hyperparameters['model']['D']**hyperparameters['model']['N'],)*2
	hyperparams = hyperparameters['state']
	hyperparams['shape'] = [2,1,-1,-1]
	size = hyperparameters['model']['N']
	samples = True
	dtype = hyperparameters['model']['system']['dtype']
	cls = None

	tol = 1e-20

	state,weights = stateize(data,shape,hyperparams,size=size,samples=samples,dtype=dtype,cls=cls)

	ndim = state.ndim

	if ndim == 1:
		norm = einsum('i,i->',state.conj(),state).real
	elif ndim == 2 and state.shape[0] == state.shape[1]:
		norm = einsum('ii->',state).real
	elif ndim == 2 and state.shape[0] != state.shape[1]:
		norm = einsum('ui,ui,u->',state.conj(),state,weights).real
	elif ndim == 3:
		norm = einsum('uii,u->',state,weights).real

	try:
		eigs = eig(state,hermitian=True)
		assert (abs(eigs)>=tol).all()
	except TypeError:
		raise

	assert allclose(norm,1.0)

	return

def test_derivative(path,tol):

	hyperparameters = load(path)

	obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	func = obj.__func__

	parameters = obj.parameters

	# Derivative of unitary
	derivative_jax = obj.__derivative__
	derivative_finite = gradient(obj,mode='finite',tol=tol)
	derivative_analytical = obj.__derivative_analytical__

	assert allclose(derivative_jax(parameters),derivative_finite(parameters)), "JAX derivative != Finite derivative"
	assert allclose(derivative_finite(parameters),derivative_analytical(parameters)), "Finite derivative != Analytical derivative"
	assert allclose(derivative_jax(parameters),derivative_analytical(parameters)), "JAX derivative != Analytical derivative"

	return

def test_grad(path,tol):

	hyperparameters = load(path)

	obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	func = obj.__func__

	parameters = obj.parameters

	# Grad of objective
	grad_jax = obj.__grad__
	grad_finite = gradient(func,mode='finite',tol=tol)
	grad_analytical = obj.__grad_analytical__

	assert allclose(grad_jax(parameters),grad_finite(parameters)), "JAX grad != Finite grad"
	assert allclose(grad_finite(parameters),grad_analytical(parameters)), "Finite grad != Analytical grad"
	assert allclose(grad_jax(parameters),grad_analytical(parameters)), "JAX grad != Analytical grad"

	return


def test_call():
	return
	# process = None
	# device = None
	# execute = True
	# verbose = 'info'

	# default = -1
	# def wrapper(stdout,stderr,returncode,default=default):
	# 	try:
	# 		result = int(stdout)
	# 	except:
	# 		result = stdout
	# 	return result


	# pattern = '#SBATCH'
	# path = 'job.slurm'

	# args = []

	# exe = ['awk']
	# flags = []
	# cmd = [' /%s/ {print FNR}'%(pattern),path]
	# arg = [*exe,*flags,*cmd]
	# args.append(arg)

	# exe = ['tail']
	# flags = ['--lines=1']
	# cmd = []
	# arg = [*exe,*flags,*cmd]
	# args.append(arg)


	# exe = ['cat']
	# flags = ['<']
	# cmd = ['test.txt']
	# arg = [*exe,*flags,*cmd]
	# args.append(arg)

	# exe = ['grep']
	# flags = []
	# cmd = ['job']
	# arg = [*exe,*flags,*cmd]
	# args.append(arg)


	# exe = ['sbatch']
	# flags = ['--export=JOB_SRC=../../src,JOB_CMD=train.py,JOB_ARGS=settings.json','<']
	# cmd = [path]

	# arg = [*exe,*flags,*cmd]
	# args.append(arg)


	# result = call(*args,wrapper=wrapper,process=process,device=device,execute=execute,verbose=verbose)

def test_main(path,tol):
	# main([path])
	return

if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	test_derivative(path,tol)
	test_grad(path,tol)
