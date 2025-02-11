#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ["",".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

envs = {
	'JAX_DISABLE_JIT':False,
	'JAX_PLATFORMS':'',
	'JAX_PLATFORM_NAME':'',
	'JAX_ENABLE_X64':True,
	'TF_CPP_MIN_LOG_LEVEL':5,
	'JAX_TRACEBACK_FILTERING':'off',
	# "XLA_FLAGS":(
	# 	"--xla_cpu_multi_thread_eigen=false "
	# 	"intra_op_parallelism_threads=1"),
}
for var in envs:
	os.environ[var] = str(envs[var])

import warnings
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
	return
warnings.showwarning = warn_with_traceback

import jax
from jax import jit
import jax.numpy as np
import numpy as onp
from math import prod
from functools import partial
import itertools
import time as timer

from jaxopt import BlockCoordinateDescent,objective,prox

from string import ascii_lowercase as characters

pi = np.pi
e = np.exp(1)

nan = np.nan
inf = np.inf
integers = (int,np.integer,getattr(onp,'int',int),onp.integer)
floats = (float,np.floating,getattr(onp,'float',float),onp.floating)
scalars = (*integers,*floats,str,type(None))
arrays = (np.ndarray,onp.ndarray)

def inplace(obj,index,item,op='set'):
	obj = getattr(obj.at[index],op)(item)
	return obj

def cond(pred,true_fun,false_fun,*operands):
	return jax.lax.cond(pred,true_fun,false_fun,*operands)
	
def forloop(start,end,func,x):
	# for i in range(start,end):
	# 	x = func(i,x)
	# return x
	return jax.lax.fori_loop(start,end,func,x)

def whileloop(cond,func,x):
#     while cond(x):
#         x = func(x)
#     return x
	return jax.lax.while_loop(cond,func,x)

def permutations(*iterables,repeat=None):
	if all(isinstance(i,int) for i in iterables):
		iterables = (range(i) for i in iterables)
	
	if repeat is None:
		repeat = 1

	return itertools.product(*iterables,repeat=repeat)

def array(*args,**kwargs):
	return np.array(*args,**kwargs)

def nparray(*args,**kwargs):
	return onp.array(*args,**kwargs)

def ones(*args,**kwargs):
	return np.ones(*args,**kwargs)

def zeros(*args,**kwargs):
	return np.zeros(*args,**kwargs)

def arange(*args,**kwargs):
	return np.arange(*args,**kwargs)

def rand(*args,**kwargs):
	return jax.random.uniform(**kwargs)

def randint(*args,**kwargs):
	return jax.random.randint(**kwargs)

def randn(*args,**kwargs):
	return jax.random.normal(**kwargs)

def haar(shape=(),seed=None,key=None,dtype=None,**kwargs):

	real,imag = seeder(seed=seed if key is None else key,size=2)

	kwargs = dict(
		shape = shape,
		dtype = dtype
	)

	out = randn(key=real,**kwargs) + 1j*randn(key=imag,**kwargs)

	Q,R = qr(out)
	R = diag(R)
	R = diag(R/absolute(R))
	
	out = dot(Q,R)

	return out

def seeder(seed,size=None):
	key = jax.random.key(seed)
	if size:
		key = jax.random.split(key,num=size)
	return key

def qr(a,**kwargs):
	return np.linalg.qr(a)

def svd(a,**kwargs):
	return np.linalg.svd(a,**kwargs)

def svds(a,**kwargs):

	u,s,v = svd(a,**kwargs)

	slices = slice(None)
	k = argmax(absolute(u), axis=0)
	shift = arange(u.shape[1])
	indices = k + shift * u.shape[0]
	signs = sign(take(ravel(u.T), indices, axis=0))
	u = u*signs[None,slices]
	v = v*signs[slices,None]

	return u,s,v

def inv(a):
	return np.linalg.inv(a)

def trace(a,**kwargs):
	return np.trace(a,**kwargs)

def diag(a,**kwargs):
	return np.diag(a,**kwargs)

def mean(a,axis=None):
	return np.mean(a,axis=axis)

def std(a,axis=None,ddof=None):
	return np.std(a,axis=axis,ddof=ddof)

def add(a,axis=None):
	return a.sum(axis=axis)

def shape(a,axis=None):
	return a.shape

def reshape(a,shape=None):
	return np.reshape(a,shape)

def ravel(a):
	return a.ravel()

def astype(a,dtype):
	return a.astype(dtype)

def transpose(a,axes=None):
	return np.transpose(a,axes=axes)

def conjugate(a):
	return a.conjugate()

def dagger(a):
	return conjugate(transpose(a))

def dot(a,b):
	return np.dot(a,b)

def inner(a,b):
	return np.inner(dagger(a),b)

def outer(a,b):
	return np.outer(a,dagger(b))

def tensorprod(a):
	out = a[0]
	for i in range(1,len(a)):
		out = np.kron(out,a[i])
	return out

def einsum(subscripts,*operands):
	return np.einsum(subscripts,*operands)

def norm(a):
	return sqrt(dot(*(ravel(a),)*2))

def sqrt(a):
	return np.sqrt(a)

def absolute(a):
	return np.abs(a)

def take(a,indices,axis):
	return np.take(a,indices,axis)

def sign(a):
	return np.sign(a)

def maximums(a,b):
	return np.maximum(a,b)

def minimums(a,b):
	return np.minimum(a,b)

def argmax(a,axis=None):
	return np.argmax(a,axis=axis)

def argmin(a,axis=None):
	return np.argmin(a,axis=axis)

def real(a):
	return a.real

def imag(a):
	return a.imag

def datatype(dtype):
	return real(array([],dtype=dtype)).dtype

def finfo(dtype=float):
	return np.finfo(dtype)

def epsilon(dtype=float,eps=None):
	eps = 1 if eps is None else eps
	try:
		dtype = finfo(dtype).eps
	except:
		dtype = float
		dtype = finfo(dtype).eps

	eps = eps*dtype

	return eps

def allclose(a,b):
	return np.allclose(a,b)

def nndsvd(a,u,v,rank=None,**kwargs):

	def true(z_plus,x_plus,y_plus,z_minus,x_minus,y_minus):
		return z_plus,x_plus,y_plus

	def false(z_plus,x_plus,y_plus,z_minus,x_minus,y_minus):
		return z_minus,x_minus,y_minus  

	def func(i,x):
		
		s,u,v = x

		z,x,y = s[i],u[slices,i],v[i,slices]

		z,x,y = real(z),real(x),real(y)

		x_plus,y_plus = maximums(x,0),maximums(y,0)
		x_minus,y_minus = maximums(-x,0),maximums(-y,0)
		x_plus_norm,y_plus_norm = norm(x_plus),norm(y_plus)
		x_minus_norm,y_minus_norm = norm(x_minus),norm(y_minus)

		z_plus,z_minus = z*x_plus_norm*y_plus_norm,z*x_minus_norm*y_minus_norm

		x_plus,y_plus = x_plus/(x_plus_norm+eps),y_plus/(y_plus_norm+eps)
		x_minus,y_minus = x_minus/(x_minus_norm+eps),y_minus/(y_minus_norm+eps)

		z,x,y = cond(z_plus>z_minus,true,false,z_plus,x_plus,y_plus,z_minus,x_minus,y_minus)

		s,u,v = inplace(s,i,1),inplace(u,(slices,i),sqrt(z)*x),inplace(v,(i,slices),sqrt(z)*y)
		
		x = s,u,v

		return x

	slices = slice(None)
	eps = epsilon(a.dtype)
	rank = min(a.shape) if rank is None else rank        
	u,s,v = svds(a,full_matrices=False,compute_uv=True)

	loop = forloop
	options = dict(start=0,end=rank)
	x = (s,u,v)
	x = loop(func=func,x=x,**options)
	s,u,v = x

	return s,u,v

def nndsvda(a,u,v,rank=None,**kwargs):
	s,u,v = nndsvd(a,u=u,v=v,rank=rank) 
	
	x = mean(a)
	u,v = inplace(u,u==0,x),inplace(v,v==0,x)

	return s,u,v

def rsvd(a,u,v,rank=None,**kwargs):
	n,m = a.shape
	k = min(min(n,m),rank)
	dtype = a.dtype
	x = sqrt(mean(a)/n)
	s,u,v = (
		ones(rank),
		astype(absolute(x*randn(**{**kwargs,**dict(shape=(n,k),dtype=dtype)})),dtype),
		astype(absolute(x*randn(**{**kwargs,**dict(shape=(k,m),dtype=dtype)})),dtype)
		)
	return s,u,v

def rsvda(a,u,v,rank=None,**kwargs):
	s,u,v = rsvd(a,u=u,v=v,rank=rank,**kwargs) 
	
	x = mean(a)
	u,v = inplace(u,u==0,x),inplace(v,v==0,x)

	return s,u,v

def nmfd(u,v,rank=None,**kwargs):
	x,y = add(u,0),add(v,1)
	s,u,v = x*y,u*1/x,transpose(transpose(v)*1/y)
	return s,u,v

def coordinate_descent(a,u,v,rank=None,options=None,**kwargs):

	# options = dict() if options is None else options
	# options = dict(
	# 	fun=options.get('function',objective.least_squares),
	# 	block_prox=options.get('constraint',prox.prox_non_negative_ridge),
	# 	maxiter=options.get('iter',1000),
	# 	tol=options.get('eps',1e-16),
	# 	)
	# opts = dict(
	# 	hyperparams_prox=options.get('constraints',1e-8),
	# 	)

	# optimizer = BlockCoordinateDescent(**options)
	
	options = dict() if options is None else options

	eps = options.get('eps',epsilon(a.dtype))
	alpha = options.get('alpha',1)

	def function(i,x):
		v,z,y = x 
		v = maximums(v-alpha*(dot(z,v)+y),0)
		x = v,z,y
		return x

	loop = forloop
	options = dict(start=0,end=options.get('iter',1000))

	def func(i,x):
		a,u,v = x

		z = dot(u.T,u)
		y = -dot(u.T,a)

		v = maximums(v-alpha*(dot(z,v)+y),0)

		z = dot(v,v.T)
		y = -dot(v,a.T)

		u = maximums(u.T-alpha*(dot(z,u.T)+y),0)
		u = u.T


		# z = dot(u.T,u)
		# y = -dot(u.T,a)
		# # w = maximums(absolute(diag(z)),eps)
		# # z = (z.T/w).T
		# # y = (y.T/w).T
		# x = v,z,y
		# x = loop(func=function,x=x,**options)
		# v,z,y = x

		# z = dot(v,v.T)
		# y = -dot(v,a.T)
		# # w = maximums(absolute(diag(z)),eps)
		# # z = (z.T/w).T
		# # y = (y.T/w).T
		# x = u.T,z,y
		# x = loop(func=function,x=x,**options)
		# u,z,y = x
		# u = u.T


		# data = 	optimizer.run(
		# 	init_params=v,
		# 	data=(u,a),
		# 	**opts
		# 	)
		# v = data.params

		# data = 	optimizer.run(
		# 	init_params=u.T,
		# 	data=(v.T,a.T),
		# 	**opts			
		# 	)
		# u = data.params.T

		x = a,u,v
		return x

	return func

def multiplicative_update(a,u,v,rank=None,options=None,**kwargs):

	def func(i,x):
		a,u,v = x
		
		u,v = (
			(dot(a,transpose(v))/dot(u,dot(v,transpose(v))))*u,
			(dot(transpose(u),a)/dot(dot(transpose(u),u),v))*v
		)

		x = a,u,v
		return x

	return func

def nmf(a,u=None,v=None,rank=None,options=None,**kwargs):
	
	def initialize(a,u=None,v=None,rank=None,options=None,**kwargs):

		options = dict() if options is None else options
		init = options.get('init')
		
		if callable(init):
			pass
		elif init is None:
			init = rsvd
		elif init in ['nndsvd']:
			init = nndsvd
		elif init in ['nndsvda']:
			init = nndsvda	
		elif init in ['rsvd']:
			init = rsvd
		elif init in ['rsvda']:
			init = rsvda					
		elif init in ['random']:		
			init = rsvd
		elif u is not None and v is not None:
			init = rsvd
		else:
			init = rsvd

		a = real(a)

		s,u,v = init(a,u=u,v=v,rank=rank,**kwargs)
		
		return a,s,u,v
	
	def run(a,u=None,v=None,rank=None,options=None,**kwargs):

		options = dict() if options is None else options
		update = options.get('update')
		iteration = options.get('iteration')
		eps = options.get('eps')

		if callable(update):
			pass
		elif update is None:
			update = coordinate_descent
		elif update in ['cd','coordinate_descent']:
			update = coordinate_descent
		elif update in ['mu','multiplicative_update']:
			update = multiplicative_update
		else:
			update = coordinate_descent

		if isinstance(iteration,int):
			loop = forloop
			options = dict(start=0,end=iteration)
		elif isinstance(eps,float):
			loop = whileloop
			options = dict(cond=lambda x,a=a,eps=eps: error(x,a) > eps)
		else:
			loop = forloop
			options = dict(start=0,end=1)			

		func = update(a,u=u,v=v,rank=rank,options=options,**kwargs)
		x = (a,u,v)

		x = loop(func=func,x=x,**options)

		a,u,v = x

		s,u,v = nmfd(u,v,rank=rank)
		
		return a,s,u,v
   
	def error(x,a):
		b,u,v = x
		return norm(a-dot(u,v))


	rank = min(a.shape) if rank is None else rank        
	options = None if options is None else options

	a,s,u,v = initialize(a,u=u,v=v,rank=rank,options=options,**kwargs)
	a,s,u,v = run(a,u=u,v=v,rank=rank,options=options,**kwargs)
	
	return a,s,u,v
	
def nvd(a,options=None,**kwargs):
	a,s,u,v = nmf(a,options=options,**kwargs)
	return dot(u*s,v)

def _nvd(a,options=None,**kwargs):

	from sklearn.decomposition import NMF as _nmf

	a = nparray(real(a))
	kwargs = dict(
		n_components=kwargs.get('rank',options.get('rank')),
		init=kwargs.get('init',options.get('init')),
		max_iter=kwargs.get('iter',options.get('iteration')) if isinstance(kwargs.get('iter',options.get('iter')),int) else None,
		tol=kwargs.get('eps',options.get('eps')) if isinstance(kwargs.get('eps',options.get('eps')),float) else epsilon(a.dtype),
		solver=kwargs.get('update',options.get('update')),
		)
	u,v,n = _nmf(**kwargs)._fit_transform(a)
	return dot(u,v)

class Basis(object):

	@classmethod
	def state(cls,D,seed=None,dtype=None,**kwargs):
		data = haar(
			shape=(D,)*2,
			seed=seed,
			dtype=dtype)[0]
		data = outer(data,data)
		return data

	@classmethod
	def unitary(cls,D,seed=None,dtype=None,**kwargs):
		data = haar(
			shape=(D,)*2,
			seed=seed,
			dtype=dtype)
		return data

	@classmethod
	def povm(cls,D,seed=None,dtype=None,**kwargs):

		data = (1/(D**2-1))*array([
				cls.zero(D,seed=seed,dtype=dtype),
				cls.plus(D,seed=seed,dtype=dtype),
				cls.plusi(D,seed=seed,dtype=dtype),
			   (cls.one(D,seed=seed,dtype=dtype)+
				cls.minus(D,seed=seed,dtype=dtype)+
				cls.minusi(D,seed=seed,dtype=dtype)),
			],dtype=dtype)		
		return data

	@classmethod
	def I(cls,D,seed=None,dtype=None,**kwargs):
		data = array([[1,0],[0,1]],dtype=dtype)
		return data

	@classmethod
	def X(cls,D,seed=None,dtype=None,**kwargs):
		data = array([[0,1],[1,0]],dtype=dtype)
		return data

	@classmethod
	def Y(cls,D,seed=None,dtype=None,**kwargs):
		data = array([[0,-1j],[1j,0]],dtype=dtype)		
		return data
		
	@classmethod
	def Z(cls,D,seed=None,dtype=None,**kwargs):
		data = array([[1,0],[0,-1]],dtype=dtype)
		return data

	@classmethod
	def H(cls,D,seed=None,dtype=None,**kwargs):
		data = (1/sqrt(2))*array([[1,1],[1,-1]],dtype=dtype)
		return data

	@classmethod
	def S(cls,D,seed=None,dtype=None,**kwargs):
		data = array([[1,0,],[0,1j]],dtype=dtype)
		return data

	@classmethod
	def CNOT(cls,D,seed=None,dtype=None,**kwargs):
		data = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=dtype)
		return data

	@classmethod
	def zero(cls,D,seed=None,dtype=None,**kwargs):
		data = array([1,*[0]*(D-1)],dtype=dtype)
		data = outer(data,data)		
		return data

	@classmethod
	def one(cls,D,seed=None,dtype=None,**kwargs):
		data = array([*[0]*(D-1),1],dtype=dtype)
		data = outer(data,data)
		return data

	@classmethod
	def plus(cls,D,seed=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,1],dtype=dtype)
		data = outer(data,data)		
		return data
		
	@classmethod
	def minus(cls,D,seed=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,-1],dtype=dtype)
		data = outer(data,data)		
		return data

	@classmethod
	def plusi(cls,D,seed=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,1j],dtype=dtype)
		data = outer(data,data)		
		return data
		
	@classmethod
	def minusi(cls,D,seed=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,-1j],dtype=dtype)
		data = outer(data,data)		
		return data

	@classmethod
	def transform(cls,data,D,N=None,where=None,architecture=None,transform=True,**kwargs):
		
		basis = cls.povm(D,**kwargs)

		inverse = inv(einsum('uij,vji->uv',basis,basis))

		if N:
			basis = tensorprod([basis]*N)
			inverse = tensorprod([inverse]*N)

		if transform:
			if callable(data):
				data = einsum('uij,wji,wv->uv',basis,data(basis),inverse)				
			else:
				data = einsum('uij,ji->u',basis,data)
		else:
			if callable(data):
				raise NotImplementedError(f"Not Implemented {data}")			
			else:
				data = einsum('uij,wji,wv->uv',basis,data,inverse)

		if architecture is None:
			data = data
		elif architecture in ['array']:
			data = array(data)
		elif architecture in ['tensor']:
			if data.ndim == 1:
				if where is True:
					data = reshape(data,(-1,1))
				else:
					data = reshape(data,(-1,1,1))
			elif data.ndim > 1:
				data = data
		else:
			data = data

		return data

	@classmethod
	def shuffle(cls,data,shape,transform=True,**kwargs):
		if transform:
			n,d = len(shape),data.ndim
			shape = [*shape]*d
			axes = [i*d+j for i in range(n) for j in range(d)]
			data = transpose(reshape(data,shape),axes)
		else:
			n,d = len(shape),data.ndim//len(shape)
			shape = [prod(shape)]*d
			axes = [i*d+j for i in range(n) for j in range(d)]
			data = reshape(transpose(data,axes),shape)
		return data

	@classmethod
	def contract(cls,state,data,where=None,**kwargs):

		if isinstance(state,dict):

			where = (1,2)

			where = state if where is None else where
			N = len(where)
			boundaries = [list(state)[0],list(state)[-1]] 
			boundary = any(i==boundaries[0] for i in where),any(i==boundaries[-1] for i in where)
			boundary = 1 if all(boundary) or len(state) < 2 else 0 if boundary[0] else -1 if boundary[-1] else None
			shape = [state[i].shape[0] for i in where]
			subscripts = '%s%s,%s->%s%s'%(
				characters[N:2*N],
				characters[:N],
				','.join((
					''.join((characters[index],characters[2*N+index+1])) if i == boundaries[0] else
					''.join((characters[index],characters[2*N+index])) if i == boundaries[-1] else
					''.join((characters[index],characters[2*N+index],characters[2*N+index+1]))
					for index,i in enumerate(where)
					)),
				characters[N:2*N],
				''.join((characters[2*N] if boundary in [-1,None] else '',characters[3*N] if boundary in [0,None] else ''))
				)

			data = einsum(subscripts,cls.shuffle(data,shape,**kwargs),*(state[i] for i in where))

			if boundary == 0:
				data = reshape(data,(*data.shape[:-1],1,data.shape[-1]))
			elif boundary == 1:
				data = reshape(data,(*data.shape,1,1))
			elif boundary == -1:
				data = reshape(data,(*data.shape[:-1],data.shape[-1],1))

			# data = cls.split(data,shape,**kwargs)

			for i in where:
				state[i] = state[i]

			data = state

		elif state.ndim == 1:
			if data.ndim == 2:
				data = einsum('ij,j->i',data,state)
			else:	
				raise NotImplementedError(f"Not Implemented {data}")						
		elif state.ndim > 1:
			if data.ndim == 2:
				data = einsum('ij,...jk,kl->...il',data,state,dagger(data))
			elif data.ndim == 3:
				data = einsum('uij,...jk,ukl->...il',data,state,dagger(data))
			else:	
				raise NotImplementedError(f"Not Implemented {data}")			
		else:
			raise NotImplementedError(f"Not Implemented {state}")			
		
		return data


def test_shuffle(*args,**kwargs):

	basis = Basis()

	D = 2
	N = 2
	d = 2
	shape = [D]*N
	state = arange(D**N)
	data = reshape(arange(D**(d*N)),(D**N,)*d)

	assert allclose(data,basis.shuffle(basis.shuffle(data,shape,transform=True),shape,transform=False))

	subscripts = 'ij,j->i'
	out = einsum(subscripts,data,state)
	
	state = basis.shuffle(state,shape)
	data = basis.shuffle(data,shape)

	subscripts = f'{characters[N:2*N]}{characters[:N]},{characters[:N]}->{characters[N:2*N]}'
	_out = basis.shuffle(einsum(subscripts,data,state),shape,transform=False)

	assert allclose(out,_out)

	return


def test_mps(*args,**kwargs):

	def initialize(D,N,state=None,data=None,**kwargs):

		basis = Basis()

		state = {i:'state' for i in range(N)} if state is None else {i:state for i in range(N)} if isinstance(state,str) else state
		state = {i:getattr(basis,state[i])(D=D,**kwargs) for i in state}
		state = {i:basis.transform(state[i],D=D,where=i in [0,N-1],**kwargs) for i in state}


		data = {i:'unitary' for i in range(N) for j in range(N) if i < j} if data is None else {i:data for i in range(N) for j in range(N) if i < j} if isinstance(data,str) else data
		data = {i:lambda state,data=getattr(basis,data[i])(D=D**len(i),**kwargs),where=range(len(i)),**kwargs: basis.contract(state,data=data,where=where,**kwargs) for i in data}
		data = {i: lambda state,data=basis.transform(data[i],D=D,N=len(i),where=i,**kwargs),where=i,**kwargs:basis.contract(state,data=data,where=where,**kwargs) for i in data}

		where = data

		for i in where:
			state = data[i](state)

		return state,data

	def function(data,func,*args,**kwargs):
		time = timer.time()

		data = func(data,**options)
	
		time = timer.time() - time

		return

	N = 4
	D = 2
	seed = 123
	dtype = 'complex'

	kwargs = dict(
		D=D,N=N,
		state={i:'state' for i in range(N)},
		data={i:'unitary' for i in permutations(*(range(N),)*2) if (i[1]-i[0])==1},
		architecture='tensor',
		seed=seed,
		dtype=dtype,		
	)

	state,data = initialize(**kwargs)

	return

def test_nmf(*args,**kwargs):

	def initialize(shape,**kwargs):
		dtype = kwargs.pop('dtype',None)

		data = astype(reshape(rand(**kwargs),shape),dtype)

		data = data/add(data)

		return data

	def function(a,func,*args,**kwargs):
		time = timer.time()

		b = func(a,**options)
	
		time = timer.time() - time

		error = abs(norm(a-b)/norm(a))

		print(kwargs)
		print(add(a),add(b))
		print('%0.4f    %0.5e'%(time,error))

		return

	n = 8
	q = 2
	d = q**n
	i = 2
	k = int((q**(i+1))*(1/2))
	shape = (q**(i+1),q**(n-i-1))
	seed = 123
	key = seeder(seed)
	dtype = 'complex'

	kwargs = dict(
		shape=shape,
		seed=seed,
		dtype=dtype,
	)	

	options = dict(
		rank=min(shape),
		shape=shape,
		options=dict(
			iteration=2000000,
			iter=1,
			eps=epsilon(dtype),
			alpha=5e0,
			init='nndsvda',
			update='cd',
		),
		key=key
	)

	funcs = [
		nvd,
		# _nvd
		]

	a = initialize(**kwargs)

	for func in funcs:
		function(a,func,**options)

	return


if __name__ == "__main__":

	args = tuple()
	kwargs = dict()

	# test_shuffle(*args,**kwargs)
	test_mps(*args,**kwargs)
	# test_nmf(*args,**kwargs)
