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

from string import ascii_lowercase as characters

np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.6e')) for dtype in ['float','float64',np.float64,np.float32]}})

pi = np.pi
e = np.exp(1)

nan = np.nan
inf = np.inf
integers = (int,np.integer,getattr(onp,'int',int),onp.integer)
floats = (float,np.floating,getattr(onp,'float',float),onp.floating)
scalars = (*integers,*floats,str,type(None))
arrays = (np.ndarray,onp.ndarray)
iterables = (*arrays,list,tuple,set,range)
dicts = (dict,)	

rng = jax.random

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
	# while cond(x):
	# 	x = func(x)
	# return x
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
	return rng.uniform(**kwargs)

def randint(*args,**kwargs):
	return rng.randint(**kwargs)

def randn(*args,**kwargs):
	return rng.normal(**kwargs)

def haar(shape=(),seed=None,key=None,dtype=None,**kwargs):

	real,imag = seeder(seed=seed,key=key,size=2)

	kwargs = dict(
		shape = shape,
		dtype = datatype(dtype)
	)

	out = randn(key=real,**kwargs) + 1j*randn(key=imag,**kwargs)

	Q,R = qr(out)
	R = diag(R)
	R = diag(R/absolute(R))
	
	out = dot(Q,R)

	return out

def seeder(seed,key=None,size=None):
	key = jax.random.key(seed) if key is None else key
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
	# slices = slice(None)
	# k = argmax(absolute(u), axis=0)
	# shift = arange(u.shape[1])
	# indices = k + shift * u.shape[0]
	# s = sign(take(ravel(u.T), indices, axis=0))
	# u,v = dotr(u,s),dotl(v,s)

	return u,s,v

def eig(a,compute_v=False,hermitian=False,**kwargs):
	if compute_v:
		if hermitian:
			_eig = np.linalg.eigh
		else:
			_eig = np.linalg.eig
	else:
		if hermitian:
			_eig = np.linalg.eigvalsh
		else:
			_eig = np.linalg.eigvals
	return _eig(a)

def eigs(a,compute_v=False,hermitian=False,**kwargs):
	return eig(a,compute_v=compute_v,hermitian=hermitian,**kwargs)

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

def sqr(a):
	return a**2

def absolute(a):
	return np.abs(a)

def abs2(a):
	return sqr(absolute(a))

def take(a,indices,axis):
	return np.take(a,indices,axis)

def sign(a):
	return np.sign(a)

def signs(a):
	s = a == 0
	return (a + s)/(absolute(a) + s)

def reciprocal(a):
	s = a == 0
	return (1 - s)/(a + s)

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

def cmplx(a):
	return a + 0*1j

def datatype(dtype):
	return np.dtype(dtype).type(0).real.dtype

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

	@jit
	def func(x):
		
		u,v,s,i = x

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

		u,v,s = inplace(u,(slices,i),sqrt(z)*x),inplace(v,(i,slices),sqrt(z)*y),inplace(s,i,1)
		
		i += 1

		x = u,v,s,i

		return x

	slices = slice(None)
	eps = epsilon(a.dtype)
	rank = min(a.shape) if rank is None else rank        
	u,s,v = svds(a)

	i = 0
	loop = whileloop
	options = dict(cond=lambda x: x[-1]<rank)
	x = (u,v,s,i)
	x = loop(func=func,x=x,**options)
	u,v,s,i = x

	return u,v,s

def nndsvda(a,u,v,rank=None,**kwargs):
	u,v,s = nndsvd(a,u=u,v=v,rank=rank) 
	
	x = mean(a)
	u,v = inplace(u,u==0,x),inplace(v,v==0,x)

	return u,v,s

def rsvd(a,u,v,rank=None,**kwargs):
	n,m = a.shape
	k = min(min(n,m),rank)
	dtype = a.dtype
	x = sqrt(mean(a)/n)
	u,v,s = (
		astype(absolute(x*randn(**{**{kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg in ['shape','key','dtype']},**dict(shape=(n,k),dtype=dtype)})),dtype) if u is not None else u,
		astype(absolute(x*randn(**{**{kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg in ['shape','key','dtype']},**dict(shape=(k,m),dtype=dtype)})),dtype) if v is not None else v,
		ones(rank)		
		)
	return u,v,s

def rsvda(a,u,v,rank=None,**kwargs):
	u,v,s = rsvd(a,u=u,v=v,rank=rank,**kwargs) 
	
	x = mean(a)
	u,v = inplace(u,u==0,x),inplace(v,v==0,x)

	return u,v,s

def nmfd(u,v,rank=None,**kwargs):
	x,y = add(u,0),add(v,1)
	u,v,s = dotr(u,reciprocal(x)),dotl(v,reciprocal(y)),x*y
	return u,v,s

def coordinate_descent(a,u,v,rank=None,**kwargs):

	# from jaxopt import BlockCoordinateDescent,objective,prox

	# options = dict(
	# 	fun=kwargs.get('function',objective.least_squares),
	# 	block_prox=kwargs.get('constraint',prox.prox_non_negative_ridge),
	# 	maxiter=kwargs.get('iteration',1000),
	# 	tol=kwargs.get('eps',1e-16),
	# 	)
	# opts = dict(
	# 	hyperparams_prox=kwargs.get('constraints',1e-8),
	# 	)

	# optimizer = BlockCoordinateDescent(**options)
	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	@jit
	def func(x):
		
		a,u,v,i = x

		# v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),0)

		# u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),0).T


		z = dot(u.T,u)
		y = -dot(u.T,a)
		w = maximums(absolute(diag(z)),eps)

		v = maximums(v-dotl(alpha*(dot(z,v)+y),1/w),0)

		z = dot(v,v.T)
		y = -dot(v,a.T)
		w = maximums(absolute(diag(z)),eps)

		u = maximums(u.T-dotl(alpha*(dot(z,u.T)+y),1/w),0).T


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

		i += 1

		x = a,u,v,i

		return x

	return func

def multiplicative_update(a,u,v,rank=None,**kwargs):

	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	@jit
	def func(x):
		
		a,u,v,i = x
		
		u *= (dot(a,transpose(v))/dot(u,dot(v,transpose(v))))
		v *= (dot(transpose(u),a)/dot(dot(transpose(u),u),v))

		i += 1

		x = a,u,v,i

		return x

	return func

def multiplicative_robust_update(a,u,v,rank=None,**kwargs):

	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)


	@jit
	def func(x):
		
		a,u,v,i = x

		z = 1/maximums(eps,sqrt(add(sqr(a-dot(u,v)),0)))
		
		u *= (dot(a*z,transpose(v))/dot(u,dot(v*z,transpose(v))))
		v *= (dot(transpose(u),a*z)/dot(dot(transpose(u),u),v*z))

		i += 1

		x = a,u,v,i

		return x

	return func

def multiplicative_l1_update(a,u,v,rank=None,**kwargs):

	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)


	@jit
	def func(x):
		
		a,u,v,i = x

		z = 1/sqrt(sqr(a-dot(u,v)) + sqr(eps))
		
		u *= (a*dot(z,transpose(v))/dot(u,v)*dot(z,transpose(v)))
		v *= (dot(transpose(u),a)*z/dot(dot(transpose(u),u),v)*z)

		i += 1

		x = a,u,v,i

		return x

	return func

def multiplicative_beta_update(a,u,v,rank=None,**kwargs):

	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)
	beta = kwargs.get('beta',2)

	@jit
	def func(x):
		
		a,u,v,i = x

		z = dot(u,v)

		jax.debug.print('{i} {z}',z=(z-a),i=i)

		u *= (a*dot(z**(beta-2),v.T))/(dot(z**(beta-1),v.T))

		v *= dot(u.T,a*(z**(beta-2)))/dot(u.T,z**(beta-1))
		
		i += 1

		# def true(*args,**kwargs):
		# 	exit()
		# 	return
		# def false(*args,**kwargs):
		# 	return
		# cond((i>100)*(norm(a-z)>1e-10),true,false)

		x = a,u,v,i

		return x

	return func


def nmf(a,u=None,v=None,rank=None,**kwargs):
	
	def initialize(a,u=None,v=None,rank=None,**kwargs):

		init = kwargs.get('init')
		
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

		u,v,s = init(a,u=u,v=v,rank=rank,**kwargs)
		
		return a,u,v,s
	
	def run(a,u=None,v=None,rank=None,**kwargs):

		update = kwargs.get('update',None)
		iteration = kwargs.get('iteration',100)
		eps = kwargs.get('eps',epsilon(a.dtype))

		updates = [[update,iteration,eps]] if not isinstance(update,iterables) else update

		for update,iteration,eps in updates:

			string = update

			if callable(update):
				pass
			elif update is None:
				update = coordinate_descent
			elif update in ['cd','coordinate_descent']:
				update = coordinate_descent
			elif update in ['mu','multiplicative_update']:
				update = multiplicative_update
			elif update in ['mru','multiplicative_robust_update']:
				update = multiplicative_robust_update			
			elif update in ['m1u','multiplicative_l1_update']:
				update = multiplicative_l1_update				
			elif update in ['mbu','multiplicative_beta_update']:
				update = multiplicative_beta_update				
			else:
				update = coordinate_descent

			i = 0
			x = (a,u,v,i)
			func = update(a,u=u,v=v,rank=rank,**kwargs)

			loop = whileloop
			options = dict(cond=(lambda x,a=a,iteration=iteration,eps=eps: (status(x,a,iteration=iteration,eps=eps))))

			x = loop(func=func,x=x,**options)

			a,u,v,i = x

			# print(string,i,norm(a-dot(u,v))/norm(a))

		u,v,s = nmfd(u,v,rank=rank)

		return a,u,v,s
   
	def status(x,a,iteration=None,eps=None):
		b,u,v,i = x
		e = norm(a-dot(u,v))
		e = (e > eps)*(i < iteration)
		return e


	rank = min(a.shape) if rank is None else rank    

	constant = add(a)

	a /= constant    

	a,u,v,s = initialize(a,u=u,v=v,rank=rank,**kwargs)
	a,u,v,s = run(a,u=u,v=v,rank=rank,**kwargs)
	
	s *= constant

	return u,v,s
	

def _nmf(a,**kwargs):

	from sklearn.decomposition import NMF as model

	rank = kwargs.get('rank') if kwargs.get('rank') is not None else min(a.shape)
	eps = kwargs.get('eps') if kwargs.get('eps') is not None else epsilon(a.dtype)

	u,v = kwargs.get('W',kwargs.get('u')),kwargs.get('H',kwargs.get('v'))
	u,v = real(u) if u is not None else u,real(v) if v is not None else v

	a = real(a)

	a = nparray(inplace(a,a<eps,0))
	u,v = nparray(u) if u is not None else u,nparray(v) if v is not None else v

	kwargs = dict(
		# n_components=kwargs.get('n_components',kwargs.get('rank')) if isinstance(kwargs.get('n_components',kwargs.get('rank')),int) else rank,
		init='custom' if u is not None and v is not None else kwargs.get('init',kwargs.get('initialize')) if isinstance(kwargs.get('init',kwargs.get('initialize')),str) else 'nndsvda',
		max_iter=kwargs.get('max_iter',kwargs.get('iteration')) if isinstance(kwargs.get('max_iter',kwargs.get('iteration')),int) else 100,
		tol=kwargs.get('tol',kwargs.get('eps')) if isinstance(kwargs.get('tol',kwargs.get('eps')),float) else epsilon(a.dtype),
		solver=kwargs.get('solver',kwargs.get('update')) if isinstance(kwargs.get('solver',kwargs.get('update')),str) else kwargs.get('solver',kwargs.get('update'))[0][0] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else 'cd',
		)

	options = dict(
		W=u,H=v
		)

	constant = add(a)

	a /= constant

	u,v,i = model(**kwargs)._fit_transform(a,**options)

	print(i,norm(a-dot(u,v))/norm(a))

	u,v,s = nmfd(u,v,rank=rank)

	u,v,s = cmplx(u),cmplx(v),cmplx(s)

	s *= constant

	return u,v,s
	

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
	def state(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = haar(
			shape=(D,)*2,
			seed=seed,
			key=key,
			dtype=dtype)[0]
		data = outer(data,data)
		return data

	@classmethod
	def unitary(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = haar(
			shape=(D,)*2,
			seed=seed,
			key=key,
			dtype=dtype)
		return data

	@classmethod
	def dephase(cls,D,parameters=None,seed=None,key=None,dtype=None,**kwargs):
		kwargs = Dictionary(**kwargs)
		if parameters is None:
			parameters = 0
		data = array([
			sqrt(1-parameters)*cls.I(D=D,dtype=dtype),
			sqrt(parameters)*cls.Z(D=D,dtype=dtype)
			],dtype=dtype)
		return data

	@classmethod
	def bitflip(cls,D,parameters=None,seed=None,key=None,dtype=None,**kwargs):
		kwargs = Dictionary(**kwargs)
		if parameters is None:
			parameters = 0
		data = array([
			sqrt(1-parameters)*cls.I(D=D,dtype=dtype),
			sqrt(parameters)*cls.X(D=D,dtype=dtype)
			],dtype=dtype)
		return data

	@classmethod
	def phaseflip(cls,D,parameters=None,seed=None,key=None,dtype=None,**kwargs):
		kwargs = Dictionary(**kwargs)
		if parameters is None:
			parameters = 0
		data = array([
			sqrt(1-parameters)*cls.I(D=D,dtype=dtype),
			sqrt(parameters)*cls.Y(D=D,dtype=dtype)
			],dtype=dtype)
		return data

	@classmethod
	def depolarize(cls,D,parameters=None,seed=None,key=None,dtype=None,**kwargs):
		if parameters is None:
			parameters = 0		
		data = array([
				sqrt(1-(D**2-1)*parameters/(D**2))*cls.I(D=D,dtype=dtype),
				sqrt(parameters/(D**2))*cls.X(D=D,dtype=dtype),
				sqrt(parameters/(D**2))*cls.Y(D=D,dtype=dtype),
				sqrt(parameters/(D**2))*cls.Z(D=D,dtype=dtype)
				],dtype=dtype)
		return data

	@classmethod
	def amplitude(cls,D,parameters=None,seed=None,key=None,dtype=None,**kwargs):
		kwargs = Dictionary(**kwargs)
		if parameters is None:
			parameters = 0		
		data = array([
			cls.element(D=D,data='00',dtype=dtype) + 
				sqrt(1-parameters)*cls.element(D=D,data='11',dtype=dtype),
			sqrt(parameters)*cls.element(D=D,data='01',dtype=dtype)
			],dtype=dtype)
		return data

	@classmethod
	def povm(cls,D,seed=None,key=None,dtype=None,**kwargs):

		if cls.basis is None:
			func = cls.pauli
		if callable(cls.basis):
			func = cls.basis			
		elif isinstance(cls.basis,str):
			func = getattr(cls,cls.basis)
		else:
			func = cls.pauli

		data = func(D=D,seed=seed,key=key,dtype=dtype,**kwargs)

		return data

	@classmethod
	def pauli(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = (1/(D**2-1))*array([
				cls.zero(D,seed=seed,key=key,dtype=dtype,**kwargs),
				cls.plus(D,seed=seed,key=key,dtype=dtype,**kwargs),
				cls.plusi(D,seed=seed,key=key,dtype=dtype,**kwargs),
			   (cls.one(D,seed=seed,key=key,dtype=dtype,**kwargs)+
				cls.minus(D,seed=seed,key=key,dtype=dtype,**kwargs)+
				cls.minusi(D,seed=seed,key=key,dtype=dtype,**kwargs)),
			],dtype=dtype)		
		return data

	@classmethod
	def identity(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = identity(D,dtype=dtype)
		return data

	@classmethod
	def I(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = array([[1,0],[0,1]],dtype=dtype)
		return data

	@classmethod
	def X(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = array([[0,1],[1,0]],dtype=dtype)
		return data

	@classmethod
	def Y(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = array([[0,-1j],[1j,0]],dtype=dtype)		
		return data
		
	@classmethod
	def Z(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = array([[1,0],[0,-1]],dtype=dtype)
		return data

	@classmethod
	def H(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = (1/sqrt(2))*array([[1,1],[1,-1]],dtype=dtype)
		return data

	@classmethod
	def S(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = array([[1,0,],[0,1j]],dtype=dtype)
		return data

	@classmethod
	def CNOT(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=dtype)
		return data

	@classmethod
	def zero(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = array([1,*[0]*(D-1)],dtype=dtype)
		data = outer(data,data)		
		return data

	@classmethod
	def one(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = array([*[0]*(D-1),1],dtype=dtype)
		data = outer(data,data)
		return data

	@classmethod
	def plus(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,1],dtype=dtype)
		data = outer(data,data)		
		return data
		
	@classmethod
	def minus(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,-1],dtype=dtype)
		data = outer(data,data)		
		return data

	@classmethod
	def plusi(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,1j],dtype=dtype)
		data = outer(data,data)		
		return data
		
	@classmethod
	def minusi(cls,D,seed=None,key=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,-1j],dtype=dtype)
		data = outer(data,data)		
		return data

	@classmethod
	def element(cls,D,data=None,seed=None,key=None,dtype=None,**kwargs):
		index = tuple(map(int,data)) if data is not None else None
		data = zeros((D,)*(len(index) if index is not None else 1),dtype=dtype)
		data = inplace(data,index,1) if index is not None else data
		return data


	@classmethod
	def transform(cls,data,D,N=None,where=None,transform=True,**kwargs):
		
		basis = cls.povm(D,**kwargs)

		inverse = inv(einsum('uij,vji->uv',basis,basis))

		N = len(data) if N is None and isinstance(data,dict) else N
	
		where = range(N) if where is None and N is not None else where

		if N:
			basis = tensorprod([basis]*N)
			inverse = tensorprod([inverse]*N)

		if transform:
			if callable(data):
				data = einsum('uij,wji,wv->uv',basis,data(basis,**kwargs),inverse)		
			elif isinstance(data,dict):
				pass
			elif isinstance(data,arrays):
				if data.ndim == 1:
					raise NotImplementedError(f"Not Implemented {data}")			
				elif data.ndim > 1:
					data = einsum('uij,...ji->...u',basis,data)

				if data.ndim == 1:
					axes = [0,1,2]
					shape = [1,*data.shape,1]
					data = transpose(reshape(data,shape),axes)

		else:
			if callable(data):
				raise NotImplementedError(f"Not Implemented {data}")
			elif isinstance(data,dict):
				shape = [j for i in data for j in data[i].shape[1:-1]]
				subscripts = '%s'%(
					','.join((
						''.join((characters[N+i],characters[i],characters[N+i+1]))
						for i in range(N)
						)),
					)
				data = ravel(reshape(einsum(subscripts,*(data[i] for i in data)),shape))								
			elif isinstance(data,arrays):
				if data.ndim == 1:
					data = einsum('uij,uv,v->ij',basis,inverse,data)				
				elif data.ndim > 1:
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
	def spectrum(cls,data,where=None,options=None,**kwargs):

		N = len(data)
		where = where if isinstance(where,integers) else min(N-2,max(1,(min(where)-1) if min(where) > 0 else (max(where)+1))) if where is not None else N//2
		where = min(N-1,max(0,where-1))

		options = dict() if options is None else options

		# defaults = dict(scheme='qr')
		# data = cls.update(data,where=where,options={**kwargs,**options,**defaults},**kwargs)

		data = data[where]
		

		axes = [0,1,2]
		shape = [data.shape[0]*prod(data.shape[1:-1]),data.shape[-1]]
		data = reshape(transpose(data,axes),shape)

		defaults = dict(scheme=options.get('scheme','spectrum'))
		data = cls.scheme(options={**kwargs,**options,**defaults},**kwargs)(data,**{**kwargs,**options,**defaults})

		return data

	@classmethod
	def scheme(cls,options=None,**kwargs):
		
		options = dict() if options is None else options

		scheme = options.get('scheme')

		if scheme is None:
			def scheme(a,conj=None,**options):
				defaults = dict(compute_uv=True,hermitian=False)
				u,s,v = svds(real(a),**{**kwargs,**options,**defaults})
				u,v = dotr(u,sqrt(s)),dotl(v,sqrt(s))
				u,v = cmplx(u),cmplx(v)
				s = min(*u.shape,*v.shape)
				return u,v,s				
		elif scheme in ['svd']:
			def scheme(a,conj=None,**options):
				defaults = dict(compute_uv=True,hermitian=False)
				u,s,v = svds(real(a),**{**kwargs,**options,**defaults})
				u,v = u,dotl(v,s)
				u,v = cmplx(u),cmplx(v)
				s = min(*u.shape,*v.shape)
				return u,v,s
		elif scheme in ['qr']:
			def scheme(a,conj=None,**options):
				defaults = dict(mode='reduced')
				u,v = qrs(real(dagger(a) if conj else a),**{**kwargs,**options,**defaults})
				u,v = (dagger(v),dagger(u)) if conj else (u,v)
				u,v = cmplx(u),cmplx(v)
				s = min(*u.shape,*v.shape)
				return u,v,s
		elif scheme in ['nmf']:
			def scheme(a,conj=None,**options):
				defaults = dict()				
				u,v,s = nmf(real(a),**{**kwargs,**options,**defaults})
				u,v = dotr(u,sqrt(s)),dotl(v,sqrt(s))
				s = min(*u.shape,*v.shape)
				return u,v,s
		elif scheme in ['_nmf']:
			def scheme(a,conj=None,**options):
				defaults = dict()
				u,v,s = _nmf(real(a),**{**kwargs,**options,**defaults})
				u,v = dotr(u,sqrt(s)),dotl(v,sqrt(s))
				s = min(*u.shape,*v.shape)
				return u,v,s				
		elif scheme in ['eig']:
			def scheme(a,conj=None,**options):
				defaults = dict(compute_v=False,hermitian=False)							
				s = eig(real(a),**{**kwargs,**options,**defaults})
				return s
		elif scheme in ['spectrum']:
			def scheme(a,conj=None,**options):
				defaults = dict(compute_uv=False,hermitian=False)							
				s = svd(real(a),**{**kwargs,**options,**defaults})
				return s
		elif scheme in ['probability']:
			def scheme(a,conj=None,**options):
				defaults = dict()				
				u,v,s = nmf(real(a),**{**kwargs,**options,**defaults})
				return s
		elif scheme in ['_spectrum']:
			def scheme(a,conj=None,**options):
				defaults = dict(compute_uv=False,hermitian=False)							
				s = svd(real(a),**{**kwargs,**options,**defaults})
				return s				
		elif scheme in ['_probability']:
			def scheme(a,conj=None,**options):
				defaults = dict()								
				u,v,s = _nmf(real(a),**{**kwargs,**options,**defaults})
				return s								
		return scheme		

	@classmethod
	def update(cls,data,where=None,options=None,**kwargs):
		
		options = dict() if options is None else options

		defaults = dict()

		if isinstance(data,dict):

			N = len(data)
			where = (N,N) if where is None else (where,where) if isinstance(where,integers) else where

			indices = (*range(0,min(where),1),*range(N-1,max(where)-1,-1))

			for i in indices:

				a = data[i]

				if i < min(where):

					axes = [0,1,2]
					shape = [data[i].shape[0]*prod(data[i].shape[1:-1]),data[i].shape[-1]]
					a = reshape(transpose(a,axes),shape)

					options.update(dict())

					u,v,s = cls.scheme(options={**kwargs,**options,**defaults},**kwargs)(a,**{**kwargs,**options,**defaults})

					axes = [0,1,2]
					shape = [data[i].shape[0],*data[i].shape[1:-1],s]
					a = transpose(reshape(u,shape),axes)

					data[i] = a

					if i < (N-1):
						data[i+1] = einsum('ij,j...k->i...k',v,data[i+1])

				elif i > max(where):

					axes = [0,1,2]
					shape = [data[i].shape[0],prod(data[i].shape[1:-1])*data[i].shape[-1]]
					a = reshape(transpose(a,axes),shape)

					options.update(dict())

					u,v,s = cls.scheme(options={**kwargs,**options,**defaults},**kwargs)(a,conj=True,**{**kwargs,**options,**defaults})

					axes = [0,1,2]
					shape = [s,*data[i].shape[1:-1],data[i].shape[-1]]
					a = transpose(reshape(v,shape),axes)

					data[i] = a

					if i > 0:
						data[i-1] = einsum('jk,i...j->i...k',u,data[i-1])

		elif isinstance(data,arrays):

			a = data
		
			axes = [0,1,-1,2]
			shape = [data.shape[0]*data.shape[1],data.shape[-1]*data.shape[2]]
			a = reshape(transpose(a,axes),shape)

			axes = [[0,1,2],[0,2,1]]

			options.update(
				dict(
					u=reshape(transpose(options.get('u'),[0,1,2]),[options.get('u').shape[0]*options.get('u').shape[1],options.get('u').shape[-1]]) if options.get('u') is not None else None,
					v=reshape(transpose(options.get('v'),[0,2,1]),[options.get('v').shape[0],options.get('v').shape[1]*options.get('v').shape[-1]]) if options.get('v') is not None else None
				)
				)
			u,v,s = cls.scheme(options={**kwargs,**options,**defaults},**kwargs)(a,**{**kwargs,**options,**defaults})

			axes = [[0,1,2],[0,2,1]]
			shape = [[data.shape[0],data.shape[1],s],[s,data.shape[-1],data.shape[2]]]

			a = {i:transpose(reshape(a,shape[index]),axes[index]) for index,(i,a) in enumerate(zip(where,(u,v)))}

			data = a

		return data

	@classmethod
	def contract(cls,state,data,where=None,options=None,**kwargs):

		if isinstance(state,dict):

			where = state if where is None else where
			N = len(where)

			shape = [j for i in where for j in state[i].shape[1:-1]]
			subscripts = '%s%s,%s->%s%s%s'%(
				characters[N:2*N],
				characters[:N],
				','.join((
					''.join((characters[2*N+i],characters[i],characters[2*N+i+1]))
					for i in range(N)
					)),
				characters[2*N],
				characters[N:2*N],
				characters[3*N]
				)

			# defaults = dict(scheme='qr')
			# state = cls.update(state,where=where,options={**kwargs,**options,**defaults},**kwargs)

			data = einsum(subscripts,cls.shuffle(data,shape=shape,**kwargs),*(state[i] for i in where))

			defaults = dict(scheme=options.get('scheme','svd'),u=state[list(where)[0]],v=state[list(where)[-1]])
			data = cls.update(data,where=where,options={**defaults,**options},**kwargs)

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
				data = einsum('ij,...jk,lk->...il',data,state,conjugate(data))
			elif data.ndim == 3:
				data = einsum('uij,...jk,ulk->...il',data,state,conjugate(data))
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

		def func(state,data=None,where=None,**kwargs):
			state = getattr(basis,state)(**{**kwargs,**dict(D=D)})
			state = basis.transform(state,where=where,**{**kwargs,**dict(D=D)})
			return state

		state = {i:'state' for i in range(N)} if state is None else {i:state for i in range(N)} if isinstance(state,str) else state
		state = {i: func(state=state[i],where=i,**kwargs) for i in state}


		def func(state,data=None,where=None,**kwargs):
			def func(state=None,data=data,where=where,**kwargs):
				if isinstance(data,str):
					data = getattr(basis,data)(**{**kwargs,**dict(D=D**len(where))})
				else:
					 data = tensorprod([getattr(basis,data[i])(**{**kwargs,**dict(D=D)}) for i in where])
				data = basis.contract(state,data,where=where,**kwargs)
				return data
			data = basis.transform(func,where=where,**{**kwargs,**dict(D=D,N=len(where))})
			data = basis.contract(state,data=data,where=where,**kwargs)
			return data

		data = {i:'unitary' for i in range(N) for j in range(N) if i < j} if data is None else {i:data for i in range(N) for j in range(N) if i < j} if isinstance(data,str) else data
		data = {i:lambda state,data=data[i],where=range(len(i)),**kwargs: func(state,data=data,where=where,**kwargs) for i in data}

		return state,data

	@timer
	def func(state,data,M=None,**kwargs):

		iterations = range(1 if M is None else M)
		where = data

		for k in iterations:
			for i in where:
				state = data[i](state,**kwargs)
				key,kwargs['key'] = rng.split(kwargs['key'])
		return state


	def _initialize(state=None,data=None,D=None,N=None,**kwargs):

		basis = Basis()

		def func(state,data=None,where=None,**kwargs):
			state = getattr(basis,state)(**{**kwargs,**dict(D=D)})
			return state

		state = {i:'state' for i in range(N)} if state is None else {i:state for i in range(N)} if isinstance(state,str) else state
		state = tensorprod([func(state=state[i],where=i,**kwargs) for i in state])

		def func(state,data,where=None,**kwargs):
			@jit
			def func(state=None,data=data,where=where,**kwargs):
				if isinstance(data,str):
					data = getattr(basis,data)(**{**kwargs,**dict(D=D**len(where))})
				else:
					 data = tensorprod([getattr(basis,data[i])(**{**kwargs,**dict(D=D)}) for i in where])
				data = tensorprod([*[basis.identity(**{**kwargs,**dict(D=D)})]*min(where),data,*[basis.identity(**{**kwargs,**dict(D=D)})]*(N-max(where)-1)])
				return data
			data = func(**kwargs)
			data = basis.contract(state,data=data,where=where,**kwargs)
			return data

		data = {i:'unitary' for i in range(N) for j in range(N) if i < j} if data is None else {i:data for i in range(N) for j in range(N) if i < j} if isinstance(data,str) else data
		data = {i:lambda state,data=data[i],where=range(len(i)),**kwargs: func(state,data=data,where=where,**kwargs) for i in data}

		return state,data


	@timer
	def _func(state,data,M=None,**kwargs):

		iterations = range(1 if M is None else M)
		where = data

		for k in iterations:
			for i in where:
				state = data[i](state,**kwargs)

		return state

	parse = lambda data,p=8: data.real.round(p)	
	norm = lambda data,p=1: (data**p).sum().real

	N = 8
	D = 2
	M = 15
	L = N//2
	seed = 123456789
	dtype = 'complex'

	# state = {i:'state' for i in range(N)}
	# data = {(i,i+1):'unitary' for i in range(N-1)}

	# kwargs = dict(
	# 	D=D,N=N,M=M,
	# 	parameters=1e-1,
	# 	architecture='tensor',		
	# 	options=dict(),
	# 	seed=seed,
	# 	dtype=dtype,		
	# )

	# basis = Basis()

	# state,data = initialize(state=state,data=data,**kwargs)

	# state = func(state,data,**kwargs)


	# _state = {i:'state' for i in range(N)}
	# _data = {(i,i+1):'unitary' for i in range(N-1)}

	# _kwargs = dict(
	# 	D=D,N=N,M=M,
	# 	architecture='tensor',		
	# 	options=dict(),
	# 	seed=seed,
	# 	dtype=dtype,		
	# )	

	# _state,_data = _initialize(state=_state,data=_data,**_kwargs)

	# _state = _func(_state,_data,**_kwargs)


	# state = basis.transform(basis.transform(state,transform=False,**kwargs),transform=False,**kwargs)
	# _state = _state

	# assert allclose(state,_state)


	state = {i:'state' for i in range(N)}
	data = {**{(i,i+1):data for i in range(N-1) for data in ['unitary',['depolarize','depolarize']]}}

	kwargs = dict(
		D=D,N=N,M=M,
		architecture='tensor',	
		parameters=1e-1,	
		options=dict(
			scheme='nmf',
			init='nndsvda',
			iteration=int(5e5),
			eps=2e-13,
			alpha=7e-1,
			update=[
				['cd',int(5e5),2e-15],
				# ['mbu',3,2e-13],['cd',int(1e6),2e-13]
				],
		),
		key=seeder(seed),
		seed=seed,
		dtype=dtype,		
	)

	basis = Basis()

	state,data = initialize(state=state,data=data,**kwargs)

	_state = copy(state)

	state = func(state,data,**kwargs)


	_state = {i:'state' for i in range(N)}
	_data = {**{(i,i+1):data for i in range(N-1) for data in ['unitary',['depolarize','depolarize']]}}

	_kwargs = dict(
		D=D,N=N,M=M,
		architecture='tensor',
		parameters=1e-1,	
		options=dict(
			scheme='svd',
		),
		key=seeder(seed),		
		seed=seed,
		dtype=dtype,		
	)

	basis = Basis()

	_state,_data = initialize(state=_state,data=_data,**_kwargs)

	_state = func(_state,_data,**_kwargs)



	state = basis.transform(state,transform=False,**kwargs)
	_state = basis.transform(_state,transform=False,**kwargs)

	print(np.sort((state).real.round(8))[-1:-100:-1])

	print((absolute(state-_state)**2).sum())
	print(state.real.sum(),abs(state.real.sum()-1),abs(_state.real.sum()-1))
	exit()


	kwargs = dict(
		D=D,N=N,M=M,
		architecture='tensor',		
		options=dict(scheme='spectrum'),
		key=seeder(seed),		
		seed=seed,
		dtype=dtype,		
	)

	spectrum = basis.spectrum(state,where=L,**kwargs)


	_kwargs = dict(
		D=D,N=N,M=M,
		architecture='tensor',		
		options=dict(scheme='_probability'),
		key=seeder(seed),		
		seed=seed,
		dtype=dtype,		
	)

	_spectrum = basis.spectrum(state,where=L,**_kwargs)



	print(spectrum)
	print(_spectrum)


	exit()

	assert allclose(basis.transform(state,transform=False,**kwargs).sum(),1) and allclose(prod(_state[i].sum() for i in _state),1)


	state = basis.transform(state,transform=False,**kwargs)
	_state = basis.transform(_state,transform=False,**kwargs)


	assert allclose(state,_state)


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

		u,v,s = func(a,**options)

		b = dot(dotr(u,sqrt(s)),dotl(v,sqrt(s)))

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
			eps=epsilon(dtype),
			alpha=5e0,
			init='nndsvda',
			update='cd',
		),
		key=key
	)

	funcs = [
		nmf,
		# _nmf
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
