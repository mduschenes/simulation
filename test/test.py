#!/usr/bin/env python

import os,sys,traceback
from functools import partial
import jax

ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,array,zeros,rand,tensorprod,trotter,forloop
from src.utils import allclose,cosh,sinh,real,abs,minimum,to_str
from src.utils import gradient,hessian,gradient_fwd,gradient_shift,fisher
from src.utils import inner_abs2,inner_real
from src.system import Logger
from src.utils import scalars,pi
from src.io import load,dump
from src.operators import haar
from src.optimize import Optimizer

from jax import config
configs = {'jax_disable_jit':False}
for name in configs:
	config.update(name,configs[name])


name = __name__
path = os.getcwd()
file = 'logging.conf'
conf = os.path.join(path,file)
file = None #'log.log'
logger = Logger(name,conf,file=file)

def setup(kwargs):

	if not isinstance(kwargs,dict):
		return

	N = kwargs['N']
	M = kwargs['M']
	p = kwargs['p']
	k = kwargs['k']
	tau = kwargs['tau']
	seed = kwargs['seed']
	verbose = kwargs['verbose']

	bounds = [-1,1]
	random = 'gaussian'
	dtype = 'complex'

	n = 2**N

	I = array([[1,0],[0,1]],dtype=dtype)
	X = array([[0,1],[1,0]],dtype=dtype)
	Y = array([[0,-1j],[1j,0]],dtype=dtype)
	Z = array([[1,0],[0,-1]],dtype=dtype)
	data = [
		*[tensorprod(array([X if k in [i] else I for k in range(N)])) for i in range(N)],
		*[tensorprod(array([Y if k in [i] else I for k in range(N)])) for i in range(N)],
		*[tensorprod(array([Z if k in [i] else I for k in range(N)])) for i in range(N)],
		*[tensorprod(array([Z if k in [i,j] else I for k in range(N)])) for i in range(N) for j in range(N) if i<j],
		]

	K = len(data)
	dim = K
	dims = (n,n)


	data = array(trotter(data,p))
	identity = tensorprod(array([I]*N))

	V = haar(shape=dims,seed=123,dtype=dtype)

	if k is None:
		k = 2*N
	slices = (slice(M),slice(None,k))
	_slices= (slice(M),slice(k,None))
	_X = [[-0.1,0,0.1,0.05,-0.2][:N],[0.00724,-0.0130,0.005,0.008,0.002,0.02,-0.003,0.060,0.009,-0.006][:K-k-N]]


	shape = (M,K)
	subshape = (M,k)

	parameters = rand(shape=subshape,bounds=bounds,key=seed,random=random)
	X = rand(shape=shape,bounds=bounds,key=seed,random=random)

	X = X.at[slices].set(parameters)
	X = X.at[:,k:k+len(_X[0])].set(_X[0])
	X = X.at[:,k+N:k+N+len(_X[1])].set(_X[1])
	X = X.at[:,k+N:k+N+len(_X[1])].set(X[:,k+N:k+N+len(_X[1])]/(4*tau*minimum(abs(X[:,k+N:k+N+len(_X[1])]))))

	X = X.ravel()
	parameters = parameters.ravel()

	kwargs.update({
		'N':N,'M':M,'K':K,'k':k,'n':n,'p':p,
		'shapes':[dims,(dim,*dims)],
		'slices':slices,
		'X':X,'data':data,'identity':identity,
		'V':V,
	})

	for kwarg in kwargs:
		if isinstance(kwargs[kwarg],scalars):
			msg = '%s : %s'%(kwarg,kwargs[kwarg])
		else:
			msg = '%s :\n%s\n'%(kwarg,kwargs[kwarg])
		if kwarg == 'X':
			msg = '%s :\n%s\n'%(kwarg,kwargs[kwarg].reshape(M,K))
		logger.log(verbose,msg)

	return parameters

def model(parameters,**kwargs):

	D = kwargs['M']*kwargs['K']
	d = kwargs['M']*kwargs['k']

	parameters = parameters.reshape((kwargs['M'],kwargs['k']))
	
	x = kwargs['X'].reshape((kwargs['M'],kwargs['K']))
	
	x = x.at[kwargs['slices']].set(parameters)

	x = x.ravel()
	
	coefficient = -1j*2*pi/2*kwargs['tau']/kwargs['p']

	@jit
	def _func(parameters,data,identity):
		return cosh(parameters)*identity + sinh(parameters)*data

	@jit
	def func(i,out):
		return out.dot(_func(coefficient*x[i],kwargs['data'][i%kwargs['K']],kwargs['identity']))

	out = kwargs['identity']		
	out = forloop(0,D,func,out)
	return out

def func(parameters,model,**kwargs):
	V = kwargs['V']
	n = kwargs['n']
	U = model(parameters)

	out = 1 - inner_abs2(U,V)
	# out = 1-(abs((V.conj().T.dot(U)).trace())/n)**2
	# out = 1-(real((V.conj().T.dot(U)).trace())/n)
	return out 

def callback(parameters,funcs,**kwargs):
	values = kwargs['optimize']['track']
	verbose = kwargs['verbose']

	# for func in funcs:
	# 	values[func].append(funcs[func](parameters))

	msg = ' '.join(['%d'%(value) if isinstance(value,int) else '%0.5e'%(value) for value in [values['iteration'][-1],values['value'][-1],values['alpha'][-1]]])
	logger.log(verbose,msg)

	status = True

	return status


def train(parameters,model,func,callback,**kwargs):

	model = jit(partial(model,**kwargs))
	func = jit(partial(func,model=model,**kwargs))
	grad = gradient(func)
	derivative = gradient_fwd(model)
	hess = hessian(func)
	fish = fisher(model)

	funcs = {
		'model':model,
		'value':func,'grad':grad,'derivative':derivative,'hessian':hess,
		'fisher':fish
		}
	hyperparameters = kwargs['optimize']

	callback = partial(callback,funcs=funcs,**kwargs)

	optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparameters)

	parameters = optimizer(parameters)

	# g = grad(parameters).block_until_ready()
	# h = hess(parameters).block_until_ready()
	# f = fish(parameters).block_until_ready()

	msg = 'U\n%s\nV\n%s\n'%(
				to_str(abs(model(parameters)).round(4)),
				to_str(abs(kwargs['V']).round(4)))
	print(msg)
	# logger.log(kwargs['verbose'],msg)

	return


def main(path='config.json'):
	
	kwargs = load(path)
	
	parameters = setup(kwargs)

	train(parameters,model=model,func=func,callback=callback,**kwargs)

	return


def test(path='test.json'):

	from src.quantum import Unitary
	from src.io import load,dump

	hyperparameters = load(path)

	# with jax.checking_leaks():
	# 	try:
	# 		train(parameters,model=model,func=func,callback=callback,**kwargs)
	# 	except Exception as exception:
	# 		print(traceback.format_exc())


	U = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)
	parameters = U.parameters

	func = U.__func__
	grad = U.__grad__
	hess = U.__hessian__
	fisher = U.__fisher__

	h = hess(parameters)
	f = fisher(parameters)

	print(h.round(3))
	print(f.round(3))

	return