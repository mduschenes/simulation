#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
from copy import deepcopy as copy

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
import time as timing

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

def timer(func):
	def function(*args,**kwargs):
		
		time = timing.time()

		data = func(*args,**kwargs)
	
		time = timing.time() - time

		return data
	
	return function

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

def eye(*args,**kwargs):
	return np.eye(*args,**kwargs)

def identity(*args,**kwargs):
	return np.eye(*args,**kwargs)

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

def qr(a,mode='reduced',**kwargs):
	return np.linalg.qr(a,mode=mode)

def qrs(a,mode='reduced',**kwargs):
	q,r = np.linalg.qr(a,mode=mode)

	s = signs(diag(r))
	q = dotr(q,conjugate(s))
	r = dotl(r,s)

	return q,r

def svd(a,full_matrices=False,compute_uv=True,hermitian=False,**kwargs):
	return np.linalg.svd(a,full_matrices=full_matrices,compute_uv=compute_uv,hermitian=hermitian)

def svds(a,**kwargs):

	u,s,v = svd(a,**kwargs)

	# s = sign(u[:,argmax(s)])
	slices = slice(None)
	k = argmax(absolute(u), axis=0)
	shift = arange(u.shape[1])
	indices = k + shift * u.shape[0]
	s = sign(take(ravel(u.T), indices, axis=0))
	u,v = dotr(u,s),dotl(v,s)

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

def dotr(a,b):
	return a*b[None,:]

def dotl(a,b):
	return a*b[:,None]

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

def signs(a):
	s = a == 0
	return (a + s)/(absolute(a) + s)

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
		# # z = dotl(z,1/w)
		# # y = dotl(y,1/w)
		# x = v,z,y
		# x = loop(func=function,x=x,**options)
		# v,z,y = x

		# z = dot(v,v.T)
		# y = -dot(v,a.T)
		# # w = maximums(absolute(diag(z)),eps)
		# # z = dotl(z,1/w)
		# # y = dotl(y,1/w)
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

class Dict(dict):
	'''
	Dictionary subclass with dictionary elements explicitly accessible as class attributes
	Args:
		args (dict): Dictionary elements
		kwargs (dict): Dictionary elements
	'''
	def __init__(self,*args,**kwargs):
		for arg in args:
			if isinstance(arg,dict):
				kwargs.update(arg)
			elif isinstance(arg,iterables) and all(isinstance(i,iterables) and len(i)==2 for i in arg):
				kwargs.update(dict(arg))

		for key in kwargs:
			if isinstance(kwargs[key],dict) and all(isinstance(attr,str) for attr in kwargs[key]):
				kwargs[key] = Dict(kwargs[key]) if not isinstance(kwargs[key],Dict) else kwargs[key]

		super().__init__(*args,**kwargs)
		self.__dict__ = self
		return

	def __hash__(self):
		return hash(tuple((attr,getattr(self,attr)) for attr in self))

	def __eq__(self,other):
		return hash(self) == hash(other)

class Basis(object):

	basis = None
	D = None
	architecture = None
	seed = None
	dtype = None
	kwargs = None

	def __init__(self,**kwargs):

		for kwarg in kwargs:
			setattr(self,kwarg,kwargs[kwarg])

		return

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

		if cls.basis is None:
			func = cls.pauli
		if callable(cls.basis):
			func = cls.basis			
		elif isinstance(cls.basis,str):
			func = getattr(cls,cls.basis)
		else:
			func = cls.pauli

		data = func(D=D,seed=seed,dtype=dtype,**kwargs)

		return data

	@classmethod
	def pauli(cls,D,seed=None,dtype=None,**kwargs):
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
	def identity(cls,D,seed=None,dtype=None,**kwargs):
		data = identity(D,dtype=dtype)
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
	def transform(cls,data,D,N=None,where=None,transform=True,**kwargs):
		
		basis = cls.povm(D,**kwargs)

		inverse = inv(einsum('uij,vji->uv',basis,basis))

		if transform:
			if isinstance(data,dict):
				pass
		else:
			if isinstance(data,dict):
				N = len(data)
				where = range(N) if where is None else where
				subscripts = '%s'%(
					','.join((
						''.join((characters[index],characters[N+index],characters[N+index+1]))
						for index,i in enumerate(where)
						)),
					)
				shape = [data[i].shape[0] for i in data]	
				data = ravel(reshape(einsum(subscripts,*(data[i] for i in data)),shape))
				return data
	
		if N:
			basis = tensorprod([basis]*N)
			inverse = tensorprod([inverse]*N)

		if transform:
			if callable(data):
				data = einsum('uij,wji,wv->uv',basis,data(basis),inverse)				
			elif isinstance(data,arrays):
				data = einsum('uij,ji->u',basis,data)

		else:
			if callable(data):
				raise NotImplementedError(f"Not Implemented {data}")			
			elif isinstance(data,arrays):
				data = einsum('uij,wji,wv->uv',basis,data,inverse)
		
		return data

	@classmethod
	def shuffle(cls,data,shape,where=None,transform=True,**kwargs):
		if transform:
			n,d = len(shape),data.ndim
			where = range(n) if where is None else where
			shape = [*shape]*d
			axes = [i*d+j for i in where for j in range(d)]
			data = transpose(reshape(data,shape),axes)
		else:
			n,d = len(shape),data.ndim//len(shape)
			where = range(n) if where is None else where			
			shape = [prod(shape)]*d
			axes = [i*d+j for i in where for j in range(d)]
			data = reshape(transpose(data,axes),shape)
		return data

	@classmethod
	def boundary(cls,data,where=None,transform=True,**kwargs):
		
		where = data if where is None else where
		boundary = [list(data)[0],list(data)[-1]] 
		
		for i in where:
			if i not in boundary:
				continue
			elif i == boundary[0]:
				if transform:
					data[i] = reshape(data[i],(*data[i].shape[:-1],1,*data[i].shape[-1:]))
				else:
					data[i] = reshape(data[i],(*data[i].shape[:-2],*data[i].shape[-1:]))
			elif i == boundary[-1]:
				if transform:
					data[i] = reshape(data[i],(*data[i].shape[:-1],*data[i].shape[-1:],1))
				else:
					data[i] = reshape(data[i],(*data[i].shape[:-2],*data[i].shape[-2:-1]))
		
		return data


	@classmethod
	def update(cls,data,where=None,options=None,**kwargs):
		
		options = dict() if options is None else options

		if isinstance(data,dict):

			N = len(data)
			where = (N,N) if where is None else (where,where) if isinstance(where,int) else where

			scheme = options.get('scheme')

			for i in data:

				if data[i].ndim == 1:
					axes = [0,1,2]
					shape = [*data[i].shape,1,1]
					data[i] = transpose(reshape(data[i],shape),axes)

			for i in (*range(0,min(where),1),*range(N-1,max(where)-1,-1)):

				a = data[i]

				if scheme is None or scheme in ['svd']:
					
					axes = [0,1,2]
					shape = [data[i].shape[0]*data[i].shape[1],data[i].shape[2]]
					a = reshape(transpose(a,axes),shape)

					if i < min(where):

						u,s,v = svd(a,**options)

						u,v,s = cls.normalize(data=(u,v,s),where=i,options=options,**kwargs)

						u,v,s = u,dotl(v,s),len(s)

						axes = [0,1]
						shape = [data[i].shape[0],data[i].shape[1],s]
						a = reshape(transpose(u,axes),shape)
						
						data[i] = a

						if i < (N-1):
							data[i+1] = einsum('ij,ujk->uik',v,data[i+1])

					elif i > max(where):

						u,s,v = svd(a,**options)

						u,v,s = cls.normalize(data=(u,v,s),where=i,options=options,**kwargs)

						u,v,s = u*s,v,len(s)

						axes = [0,1]
						shape = [data[i].shape[0],data[i].shape[1],s]
						a = reshape(transpose(v,axes),shape)
						
						data[i] = a

						if i > 0:
							data[i-1] = einsum('jk,uij->uik',u,data[i-1])


				elif scheme in ['qr']:
					
					if i < min(where):

						axes = [0,1,2]
						shape = [data[i].shape[0]*data[i].shape[1],data[i].shape[2]]
						a = reshape(transpose(a,axes),shape)

						u,v = qrs(a,**options)

						s = min(shape[-2:])

						u,v,s = cls.normalize(data=(u,v,s),where=i,options=options,**kwargs)

						axes = [0,1,2]
						shape = [data[i].shape[0],data[i].shape[1],s]
						a = transpose(reshape(u,shape),axes)
						
						data[i] = a

						if i < (N-1):
							data[i+1] = einsum('ij,ujk->uik',v,data[i+1])

					elif i > max(where):

						axes = [1,0,2]
						shape = [data[i].shape[1],data[i].shape[0]*data[i].shape[2]]
						a = reshape(transpose(a,axes),shape)

						u,v = qrs(dagger(a),**options)

						s = min(shape[-2:])

						u,v = dagger(v),dagger(u)

						u,v,s = cls.normalize(data=(u,v,s),where=i,options=options,**kwargs)

						axes = [1,0,2]
						shape = [s,data[i].shape[0],data[i].shape[2]]
						a = transpose(reshape(v,shape),axes)
						
						data[i] = a

						if i > 0:
							data[i-1] = einsum('jk,uij->uik',u,data[i-1])


		elif isinstance(data,arrays):

			l = 2
			where = range(l) if where is None else where

			scheme = options.get('scheme')

			a = data

			if scheme is None or scheme in ['svd']:
			
				axes = [0,2,1,3]
				shape = [data.shape[0]*data.shape[2],data.shape[1]*data.shape[3]]
				a = reshape(transpose(a,axes),shape)

				u,s,v = svd(a,**options)

				u,v,s = cls.normalize(data=(u,v,s),where=where,options=options,**kwargs)

				axes = [[0,1,2],[0,1,2]]
				shape = [[data.shape[0],data.shape[2],s],[data.shape[1],s,data.shape[3]]]
				a = {i:transpose(reshape(a,shape[index]),axes[index]) for index,(i,a) in enumerate(zip(where,(u,v)))}

				data = a

			elif scheme in ['qr']:
			
				axes = [0,2,1,3]
				shape = [data.shape[0]*data.shape[2],data.shape[1]*data.shape[3]]
				a = reshape(transpose(a,axes),shape)

				u,v = qrs(a,**options)

				s = min(shape[-2:])

				u,v,s = cls.normalize(data=(u,v,s),where=where,options=options,**kwargs)

				axes = [[0,1,2],[0,1,2]]
				shape = [[data.shape[0],data.shape[2],s],[data.shape[1],s,data.shape[3]]]
				a = {i:transpose(reshape(a,shape[index]),axes[index]) for index,(i,a) in enumerate(zip(where,(u,v)))}

				data = a

		return data

	@classmethod
	def normalize(cls,data,where=None,options=None,**kwargs):

		options = dict() if options is None else options
		scheme = options.get('scheme')

		if scheme is None or scheme in ['svd']:
			
			u,v,s = data

			data = u,v,s

		elif scheme in ['qr']:
			
			q,r,s = data

			data = q,r,s

		return data

	@classmethod
	def contract(cls,state,data,where=None,options=None,**kwargs):

		if isinstance(state,dict):

			where = state if where is None else where
			N = len(where)

			shape = [state[i].shape[0] for i in where]
			subscripts = '%s%s,%s->%s%s'%(
				characters[N:2*N],
				characters[:N],
				','.join((
					''.join((characters[index],characters[2*N+index],characters[2*N+index+1]))
					for index,i in enumerate(where)
					)),
				characters[N:2*N],
				''.join((characters[2*N],characters[3*N]))
				)
			
			state = cls.update(state,where=where,options=options,**kwargs)

        	from math import prod
			for i in state:
				print(state[i],state[i].shape)
			print(prod(state[i].sum() for i in state))
			exit()

			data = einsum(subscripts,cls.shuffle(data,shape=shape,**kwargs),*(state[i] for i in where))

			data = cls.update(data,where=where,options=options,**kwargs)

			for i in where:
				state[i] = data[i]

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

	def initialize(state=None,data=None,D=None,N=None,**kwargs):

		basis = Basis()

		state = {i:'state' for i in range(N)} if state is None else {i:state for i in range(N)} if isinstance(state,str) else state
		state = {i:getattr(basis,state[i])(D=D,**kwargs) for i in state}
		state = {i:basis.transform(state[i],D=D,where=i,**kwargs) for i in state}

		data = {i:'unitary' for i in range(N) for j in range(N) if i < j} if data is None else {i:data for i in range(N) for j in range(N) if i < j} if isinstance(data,str) else data
		data = {i:lambda state,data=getattr(basis,data[i])(D=D**len(i),**kwargs),where=range(len(i)),**kwargs: basis.contract(state,data=data,where=where,**kwargs) for i in data}
		data = {i: lambda state,data=basis.transform(data[i],D=D,N=len(i),where=i,**kwargs),where=i,**kwargs:basis.contract(state,data=data,where=where,**kwargs) for i in data}

		return state,data

	@timer
	def func(state,data,M=None,**kwargs):

		iterations = range(1 if M is None else M)
		where = data

		for k in iterations:
			for i in where:
				state = data[i](state,**kwargs)

		return state


	def _initialize(state=None,data=None,D=None,N=None,**kwargs):

		basis = Basis()

		state = {i:'state' for i in range(N)} if state is None else {i:state for i in range(N)} if isinstance(state,str) else state
		state = {i:getattr(basis,state[i])(D=D,**kwargs) for i in state}
		state = tensorprod([basis.transform(state[i],D=D,where=i,**kwargs) for i in state])

		data = {i:'unitary' for i in range(N) for j in range(N) if i < j} if data is None else {i:data for i in range(N) for j in range(N) if i < j} if isinstance(data,str) else data
		data = {i:lambda state,data=getattr(basis,data[i])(D=D**len(i),**kwargs),where=range(len(i)),**kwargs: basis.contract(state,data=data,where=where,**kwargs) for i in data}
		data = {i: lambda state,data=basis.transform(data[i],D=D,N=len(i),where=i,**kwargs),where=i,**kwargs:basis.contract(state,data=tensorprod([*[basis.identity(**{**kwargs,**dict(D=D**2)})]*(min(i)),data,*[basis.identity(**{**kwargs,**dict(D=D**2)})]*(N-max(i)-1)]),where=where,**kwargs) for i in data}

		return state,data

	@timer
	def _func(state,data,M=None,**kwargs):

		iterations = range(1 if M is None else M)
		where = data

		for k in iterations:
			for i in where:
				state = data[i](state,**kwargs)

		return state

	N = 4
	D = 2
	M = 1
	seed = 123
	dtype = 'complex'

	state = {i:'zero' for i in range(N)}
	data = {i:'identity' for i in permutations(*(range(N),)*2) if (i[1]-i[0])==1}

	kwargs = dict(
		D=D,N=N,M=M,
		architecture='tensor',		
		# options=dict(scheme='svd',full_matrices=False,compute_uv=True),
		options=dict(scheme='qr',mode='reduced'),
		seed=seed,
		dtype=dtype,		
	)

	state,data = initialize(state=state,data=data,**kwargs)

	# for i in _state:
	# 	print(_state[i])
	# exit()

	# obj = state
	# print(prod(obj[i].sum() for i in obj))
	# obj = _state
	# print(prod(obj[i].sum() for i in obj))
	# exit()

	data = func(state,data,**kwargs)

	data = Basis().transform(data,transform=False,**kwargs)
	

	_data = Basis().update(copy(state),**kwargs)

	_data = Basis().transform(_data,transform=False,**kwargs)


	parse = lambda data,p=8: data.real.round(p)
	norm = lambda data,p=1: (data**p).sum().real
	objs = [data,_data]

	for obj in objs:
		print(parse(obj),norm(obj,1),norm(obj,2))

	assert allclose(data,_data)

	exit()


	_state = {i:'zero' for i in range(N)}
	_data = {i:'identity' for i in permutations(*(range(N),)*2) if (i[1]-i[0])==1}

	_kwargs = dict(
		D=D,N=N,M=M,
		architecture='array',
		# options=dict(scheme='svd',full_matrices=False,compute_uv=True),
		options=dict(scheme='qr',mode='reduced'),		
		seed=seed,
		dtype=dtype,		
	)	

	_state,_data = _initialize(state=_state,data=_data,**_kwargs)

	_data = _func(_state,_data,**_kwargs)

	print(data.real.round(8))
	print(_data.real.round(8))

	assert allclose(data,_data)

	print('Passed')

	return

def test_nmf(*args,**kwargs):

	def initialize(shape,**kwargs):
		dtype = kwargs.pop('dtype',None)

		data = astype(reshape(rand(**kwargs),shape),dtype)

		data = data/add(data)

		return data

	def function(a,func,*args,**kwargs):
		time = timing.time()

		b = func(a,**options)
	
		time = timing.time() - time

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
