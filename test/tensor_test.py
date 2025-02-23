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

def load(obj,wr='r',ext='hdf5',**kwargs):
	'''
	Load objects from path into hdf5
	Args:
		obj (str,object): Path or file object to load object
		wr (str): Read mode
		ext (str): Extension type of object
		kwargs (dict): Additional loading keyword arguments
	Returns:
		data (object): Loaded object
	'''

	import h5py

	def load(obj,wr='r',ext='hdf5',**kwargs):
		'''
		Load objects from path into hdf5
		Args:
			obj (object): hdf5 object to load object
			wr (str): Read mode
			ext (str): Extension type of object
			kwargs (dict): Additional loading keyword arguments
		Returns:
			data (object): Loaded object
		'''	

		data = {}
		
		if isinstance(obj, h5py._hl.group.Group):
			names = obj
			for name in names:
				key = name
				if isinstance(obj[name], h5py._hl.group.Group):	
					data[key] = _load_hdf5(obj[name],wr=wr,ext=ext,**kwargs)
				else:
					data[key] = obj[name][...]
					if data[key].dtype.kind in ['S','O']:
						data[key] = data[key].astype(str)
					
			names = obj.attrs
			for name in names:
				key = name
				data[key] = obj.attrs[name]

		else:
			data = obj.value
		
		return data


	if isinstance(obj,str):
		with h5py.File(obj,wr) as file:
			data = load(file,wr=wr,ext=ext,**kwargs)
	else:
		file = obj
		data = load(file,wr=wr,ext=ext,**kwargs)
	return data


def dump(obj,path,wr='w',ext='hdf5',**kwargs):
	'''
	Dump objects into hdf5
	Args:
		obj (object): Object to dump
		path (str,object): Path object to dump to
		wr (str): Write mode
		ext (str): Extension type of object
		kwargs (dict): Additional loading keyword arguments
	'''		

	import h5py

	def dump(obj,path,wr='w',ext='hdf5',**kwargs):
		'''
		Dump objects into hdf5
		Args:
			obj (object): object to dump
			path (object): hdf5 object to dump to
			wr (str): Write mode
			ext (str): Extension type of object
			kwargs (dict): Additional loading keyword arguments
		'''		

		if isinstance(obj,dict):
			names = obj
			for name in names:
				key = name
				if isinstance(obj[name],dict):
					path.create_group(key)
					dump(obj[name],path[key],wr=wr,ext=ext,**kwargs)
				elif isinstance(obj[name],scalars):
					try:
						path.attrs[key] = obj[name]
					except TypeError:
						pass
				else:
					try:
						path[key] = obj[name]
					except:
						path[key] = nparray(obj[name],dtype='S')
		else:
			path = obj

		return

	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(path))

	if isinstance(path,str):
		with h5py.File(path,wr) as file:
			dump(obj,file,wr=wr,ext=ext,**kwargs)
	else:	
		file = path
		dump(obj,file,wr=wr,ext=ext,**kwargs)

	return


def plot(settings={},fig=None,ax=None):

	import matplotlib
	matplotlib.use('pdf')
	import matplotlib.pyplot as plt

	import numpy as np

	defaults = {'fig':{},'ax':{},'style':{}}
	options = {'instance':['plot','errorbar'],'obj':{'ax':ax,'fig':fig},'data':['x','y','xerr','yerr'],'plot':["label","alpha","marker","markeredgecolor","markeredgewidth","markersize","linestyle","linewidth","elinewidth","capsize","color","ecolor"]}
	for key in defaults:
		if key not in settings:
			settings[key] = {}
		settings[key].update({attr: defaults[key][attr] for attr in defaults[key] if attr not in settings})


	key = 'style'
	mplstyle = settings[key].get('mplstyle','plot.mplstyle')
	with matplotlib.style.context(mplstyle):
		if fig is None and ax is None:
			fig,ax = plt.subplots()
		elif fig is not None:
			ax = fig.gca()
		elif ax is not None:
			fig = ax.get_figure()
		options['obj'] = {'ax':ax,'fig':fig}

		keys = ['ax']
		for key in keys:
			if settings.get(key) is None:
				continue
			for attr in settings[key]:
				if not attr in options['instance']:
					continue
				
				args = tuple((settings[key][attr][option] for option in options['data'] if settings[key][attr].get(option) is not None))
				kwargs = dict({option:settings[key][attr][option] for option in settings[key][attr] if option in options['plot']})
				
				obj = options['obj'][key]
				for attr in attr.split('.'):
					obj = getattr(obj,attr)				
		
				obj(*args,**kwargs)

		keys = ['ax','fig']
		for key in keys:
			if settings.get(key) is None:
				continue
			for attr in settings[key]:
				if attr in options['instance']:
					continue
				
				args = tuple()
				kwargs = dict(settings[key][attr])
				
				obj = options['obj'][key]				
				for attr in attr.split('.'):
					obj = getattr(obj,attr)

				obj(*args,**kwargs)

	return fig,ax

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

def log(a):
	return np.log(a)

def log10(a):
	return np.log10(a)

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

def maximum(a,axis=None):
	return np.max(a,axis=axis)

def minimum(a,axis=None):
	return np.min(a,axis=axis)

def argmax(a,axis=None):
	return np.argmax(a,axis=axis)

def argmin(a,axis=None):
	return np.argmin(a,axis=axis)

def maximums(a,b):
	return np.maximum(a,b)

def minimums(a,b):
	return np.minimum(a,b)

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
		astype(absolute(x*randn(**{**{kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg in ['shape','key','dtype']},**dict(shape=(n,k),dtype=dtype)})),dtype) if u is None else u,
		astype(absolute(x*randn(**{**{kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg in ['shape','key','dtype']},**dict(shape=(k,m),dtype=dtype)})),dtype) if v is None else v,
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
	# options = dict(
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

		z = dot(u,v)
		alpha = einsum('ij,ij',a,z)/einsum('ij,ij',z,z)

		v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),eps)
		u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),eps).T


		# z = dot(u.T,u)
		# y = -dot(u.T,a)
		# w = maximums(absolute(diag(z)),eps)

		# v = maximums(v-dotl(alpha*(dot(z,v)+y),1/w),eps)

		# z = dot(v,v.T)
		# y = -dot(v,a.T)
		# w = maximums(absolute(diag(z)),eps)

		# u = maximums(u.T-dotl(alpha*(dot(z,u.T)+y),1/w),eps).T


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
		# 	**options
		# 	)
		# v = data.params

		# data = 	optimizer.run(
		# 	init_params=u.T,
		# 	data=(v.T,a.T),
		# 	**options			
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

		u *= (a*dot(z**(beta-2),v.T))/(dot(z**(beta-1),v.T))

		v *= dot(u.T,a*(z**(beta-2)))/dot(u.T,z**(beta-1))
		
		i += 1

		x = a,u,v,i

		return x

	return func


def inverse_update(a,u,v,rank=None,**kwargs):

	eps = kwargs.get('eps',epsilon(a.dtype))

	@jit
	def func(x):
		
		a,u,v,i = x

		v = maximums(dot(dot(inv(dot(transpose(u),u)),transpose(u)),a),eps)
		u = maximums(dot(a,dot(transpose(v),inv(dot(u,transpose(u))))),eps)

		i += 1

		x = a,u,v,i

		return x

	return func


def nmf(a,u=None,v=None,rank=None,**kwargs):
	
	def initialize(a,u=None,v=None,rank=None,**kwargs):

		init = kwargs.get('init')
		
		if callable(init):
			pass
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
		elif init is None:
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
			elif update in ['inv','inverse_update']:
				update = inverse_update
				iteration = 1
			else:
				update = coordinate_descent

			i = 0
			x = (a,u,v,i)
			func = update(a,u=u,v=v,rank=rank,**kwargs)

			loop = whileloop
			options = dict(cond=(lambda x,a=a,iteration=iteration,eps=eps: (status(x,a,iteration=iteration,eps=eps))))

			x = loop(func=func,x=x,**options)

			a,u,v,i = x

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

	u,v = kwargs.get('W',kwargs.get('u')),kwargs.get('H',kwargs.get('v'))
	rank = min(kwargs.get('rank') if kwargs.get('rank') is not None else min(a.shape),*a.shape)
	eps = kwargs.get('eps') if kwargs.get('eps') is not None else epsilon(a.dtype)

	a = nparray(inplace(real(a),real(a)<eps,0))
	u,v = nparray(real(u)) if u is not None else u,nparray(real(v)) if v is not None else v

	kwargs = dict(
		n_components=rank,
		init='custom' if u is not None and v is not None and kwargs.get('init',kwargs.get('initialize')) is None else kwargs.get('init',kwargs.get('initialize')) if isinstance(kwargs.get('init',kwargs.get('initialize')),str) else 'nndsvda',
		max_iter=kwargs.get('max_iter',kwargs.get('iteration')) if isinstance(kwargs.get('max_iter',kwargs.get('iteration')),int) else kwargs.get('solver',kwargs.get('update'))[0][1] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else 100,
		tol=kwargs.get('tol',kwargs.get('eps')) if isinstance(kwargs.get('tol',kwargs.get('eps')),float) else kwargs.get('solver',kwargs.get('update'))[0][2] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else epsilon(a.dtype),
		solver=kwargs.get('solver',kwargs.get('update')) if isinstance(kwargs.get('solver',kwargs.get('update')),str) else kwargs.get('solver',kwargs.get('update'))[0][0] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else 'cd',
		)

	options = dict(
		W=u,H=v
		)

	func = model(**kwargs)._fit_transform

	constant = add(a)

	a /= constant

	u,v,i = func(a,**options)

	u,v,s = nmfd(u,v,rank=rank)

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

				N = len(data) if N is None else N
				where = range(N) if where is None and N is not None else where

				if transform is None:
					shape = [basis.shape[-2+i]**N for i in range(d)]
					subscripts = '%s,%s'%(
						','.join((
							''.join((characters[N+i],characters[i],characters[N+i+1]))
							for i in range(N)
							)),
						','.join((
							''.join((characters[i],characters[2*N+i],characters[3*N+i]))
							for i in range(N)
							)),						
						)
					data = reshape(einsum(subscripts,*(data[i] for i in data)),shape)
				else:
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
	def shuffle(cls,state,shape,where=None,transform=True,**kwargs):
		if transform:
			n,d = len(shape),state.ndim
			where = range(n) if where is None else where
			shape = [*shape]*d
			axes = [i*d+j for i in where for j in range(d)]
			state = transpose(reshape(state,shape),axes)
		else:
			n,d = len(shape),state.ndim//len(shape)
			where = range(n) if where is None else where			
			shape = [prod(shape)]*d
			axes = [i*d+j for i in where for j in range(d)]
			state = reshape(transpose(state,axes),shape)
		return state

	@classmethod
	def spectrum(cls,state,where=None,options=None,**kwargs):

		N = len(state)
		where = where if isinstance(where,integers) else min(N-2,max(1,(min(where)-1) if min(where) > 0 else (max(where)+1))) if where is not None else N//2
		where = min(N-1,max(0,where-1))

		options = dict() if options is None else options

		# defaults = dict(scheme='qr')
		# state = cls.update(state,where=where,options={**kwargs,**options,**defaults},**kwargs)

		state = state[where]

		axes = [0,1,2]
		shape = [state.shape[0]*prod(state.shape[1:-1]),state.shape[-1]]
		state = reshape(transpose(state,axes),shape)

		defaults = dict(scheme=options.get('scheme','spectrum'))
		state = cls.scheme(options={**kwargs,**options,**defaults},**kwargs)(state,**{**kwargs,**options,**defaults})

		return state

	@classmethod
	def scheme(cls,options=None,**kwargs):
		
		options = dict() if options is None else options

		scheme = options.get('scheme')

		if scheme is None:
			def scheme(a,conj=None,**options):
				defaults = dict(compute_uv=True,hermitian=False)
				u,s,v = svds(real(a),**{**kwargs,**options,**defaults})
				u,v = dotr(u,sqrt(s)),dotl(v,sqrt(s))
				u,v,s = cmplx(u),cmplx(v),min(*u.shape,*v.shape)
				return u,v,s				
		elif scheme in ['svd']:
			def scheme(a,conj=None,**options):
				defaults = dict(compute_uv=True,hermitian=False)
				u,s,v = svds(real(a),**{**kwargs,**options,**defaults})
				u,v = u,dotl(v,s)
				u,v,s = cmplx(u),cmplx(v),min(*u.shape,*v.shape)
				return u,v,s
		elif scheme in ['qr']:
			def scheme(a,conj=None,**options):
				defaults = dict(mode='reduced')
				u,v = qrs(real(dagger(a) if conj else a),**{**kwargs,**options,**defaults})
				u,v = (dagger(v),dagger(u)) if conj else (u,v)
				u,v,s = cmplx(u),cmplx(v),min(*u.shape,*v.shape)
				return u,v,s
		elif scheme in ['nmf']:
			def scheme(a,conj=None,**options):
				defaults = dict()			
				u,v,s = nmf(real(a),**{**kwargs,**options,**defaults})
				u,v = dotr(u,sqrt(s)),dotl(v,sqrt(s))
				u,v,s = cmplx(u),cmplx(v),min(*u.shape,*v.shape)
				return u,v,s
		elif scheme in ['_nmf']:
			def scheme(a,conj=None,**options):
				defaults = dict()
				u,v,s = _nmf(real(a),**{**kwargs,**options,**defaults})
				u,v = dotr(u,sqrt(s)),dotl(v,sqrt(s))
				u,v,s = cmplx(u),cmplx(v),min(*u.shape,*v.shape)
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
	def update(cls,state,where=None,options=None,**kwargs):
		
		options = dict() if options is None else options

		defaults = dict()

		if isinstance(state,dict):

			N = len(state)
			where = (N,N) if where is None else (where,where) if isinstance(where,integers) else where

			indices = (*range(0,min(where),1),*range(N-1,max(where)-1,-1))

			for i in indices:

				a = state[i]

				if i < min(where):

					axes = [0,1,2]
					shape = [state[i].shape[0]*prod(state[i].shape[1:-1]),state[i].shape[-1]]
					a = reshape(transpose(a,axes),shape)

					options.update(dict())

					u,v,s = cls.scheme(options={**kwargs,**options,**defaults},**kwargs)(a,**{**kwargs,**options,**defaults})

					axes = [0,1,2]
					shape = [state[i].shape[0],*state[i].shape[1:-1],s]
					a = transpose(reshape(u,shape),axes)

					state[i] = a

					if i < (N-1):
						state[i+1] = einsum('ij,j...k->i...k',v,state[i+1])

				elif i > max(where):

					axes = [0,1,2]
					shape = [state[i].shape[0],prod(state[i].shape[1:-1])*state[i].shape[-1]]
					a = reshape(transpose(a,axes),shape)

					options.update(dict())

					u,v,s = cls.scheme(options={**kwargs,**options,**defaults},**kwargs)(a,conj=True,**{**kwargs,**options,**defaults})

					axes = [0,1,2]
					shape = [s,*state[i].shape[1:-1],state[i].shape[-1]]
					a = transpose(reshape(v,shape),axes)

					state[i] = a

					if i > 0:
						state[i-1] = einsum('jk,i...j->i...k',u,state[i-1])

		elif isinstance(state,arrays):

			if len(where) == 2:

				a = state
			
				axes = [0,1,-1,2]
				shape = [state.shape[0]*state.shape[1],state.shape[-1]*state.shape[2]]
				a = reshape(transpose(a,axes),shape)

				axes = [[0,1,2],[0,2,1]]

				options.update(
					dict(
						u=reshape(transpose(options.get('u'),[0,1,2]),[options.get('u').shape[0]*options.get('u').shape[1],options.get('u').shape[-1]]) if options.get('u') is not None else None,
						v=reshape(transpose(options.get('v'),[0,2,1]),[options.get('v').shape[0],options.get('v').shape[1]*options.get('v').shape[-1]]) if options.get('v') is not None else None
					)
					)
				u,v,s = cls.scheme(options={**kwargs,**options,**defaults},**kwargs)(a,**{**kwargs,**options,**defaults})

				print(where,(norm(a-dot(u,v))/norm(a)).real,dot(u,v).real.min(),dot(u,v).real.max())

				axes = [[0,1,2],[0,2,1]]
				shape = [[state.shape[0],state.shape[1],s],[s,state.shape[-1],state.shape[2]]]

				a = {i:transpose(reshape(a,shape[index]),axes[index]) for index,(i,a) in enumerate(zip(where,(u,v)))}

				state = a

		return state

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
			where = [where] if not isinstance(where,iterables) else where
			state = getattr(basis,state)(**{**kwargs,**dict(D=D)})
			state = basis.transform(state,where=where,**{**kwargs,**dict(D=D,N=len(where))})
			return state

		state = {i:'state' for i in range(N)} if state is None else {i:state for i in range(N)} if isinstance(state,str) else state
		state = {i: func(state=state[i],where=i,**kwargs) for i in state}


		def func(state,data=None,where=None,**kwargs):
			def func(state=None,data=data,where=where,**kwargs):
				where = [where] if not isinstance(where,iterables) else where
				if isinstance(data,str):
					data = getattr(basis,data)(**{**kwargs,**dict(D=D**len(where),N=len(where))})
				else:
					 data = tensorprod([getattr(basis,data[index])(**{**kwargs,**dict(D=D)}) for index,i in enumerate(where)])
				data = basis.contract(state,data,where=where,**kwargs)
				return data
			data = basis.transform(func,where=where,**{**kwargs,**dict(D=D,N=len(where))})
			data = basis.contract(state,data=data,where=where,**kwargs)
			return data

		data = {index:(data,where) for index,(data,where) in enumerate(('unitary' if data is None else data,(i,j)) for i in range(N) for j in range(N) if i < j)} if not isinstance(data,dicts) else data
		data = {index:lambda state,data=data,where=where,**kwargs: func(state,data=data,where=where,**kwargs) for index,(data,where) in data.items()}

		return state,data

	@timer
	def func(state,data,M=None,**kwargs):

		iterations = range(1 if M is None else M)

		for k in iterations:
			for i in data:
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
					 data = tensorprod([getattr(basis,data[index])(**{**kwargs,**dict(D=D)}) for index,i in enumerate(where)])
				data = tensorprod([*[basis.identity(**{**kwargs,**dict(D=D)})]*min(where),data,*[basis.identity(**{**kwargs,**dict(D=D)})]*(N-max(where)-1)])
				return data
			data = func(**kwargs)
			data = basis.contract(state,data=data,where=where,**kwargs)
			return data

		data = {index:(data,where) for index,(data,where) in enumerate(('unitary' if data is None else data,(i,j)) for i in range(N) for j in range(N) if i < j)} if not isinstance(data,dicts) else data
		data = {index:lambda state,data=data,where=where,**kwargs: func(state,data=data,where=where,**kwargs) for index,(data,where) in data.items()}

		return state,data


	@timer
	def _func(state,data,M=None,**kwargs):

		iterations = range(1 if M is None else M)

		for k in iterations:
			for i in data:
				state = data[i](state,**kwargs)
				key,kwargs['key'] = rng.split(kwargs['key'])

		return state

	parse = lambda data,p=8: data.real.round(p)	
	norm = lambda data,p=1: (data**p).sum().real
	boolean = lambda path: os.path.exists(path) or False

	N = 8
	D = 2
	M = 2*N
	L = N//2
	K = D**N
	parameters = 1e-4
	seed = 123
	dtype = 'complex'
	path = 'data/data.hdf5'

	# state = {i:'state' for i in range(N)}
	# data = {index:(data,where) for index,(data,where) in enumerate((data,where) for i in [*range(0,N-1,2),*range(1,N-1,2)] for where in [(i,i+1)] for data in ['unitary'])}

	# kwargs = dict(
	# 	D=D,N=N,M=M,
	# 	parameters=parameters,
	# 	options=dict(),
	# 	seed=seed,
	# 	dtype=dtype,		
	# )

	# basis = Basis()

	# state,data = initialize(state=state,data=data,**kwargs)

	# state = func(state,data,**kwargs)


	# _state = {i:'state' for i in range(N)}
	# _data = {index:(data,where) for index,(data,where) in enumerate((data,where) for i in [*range(0,N-1,2),*range(1,N-1,2)] for where in [(i,i+1)] for data in ['unitary'])}

	# _kwargs = dict(
	# 	D=D,N=N,M=M,
	# 	options=dict(),
	# 	seed=seed,
	# 	dtype=dtype,		
	# )	

	# _state,_data = _initialize(state=_state,data=_data,**_kwargs)

	# _state = _func(_state,_data,**_kwargs)


	# state = basis.transform(basis.transform(state,transform=False,**{**kwargs,**dict(D=D,N=None)}),transform=False,**{**kwargs,**dict(D=D,N=None)})
	# _state = _state

	# assert allclose(state,_state)

	if not boolean(path):


		state = {i:'state' for i in range(N)}
		data = {index:(data,where) for index,(data,where) in enumerate((data,where) for i in [*range(0,N-1,2),*range(1,N-1,2)] for where in [(i,i+1)] for data in ['unitary',['depolarize','depolarize']])}


		kwargs = dict(
			D=D,N=N,M=M,
			parameters=parameters,	
			options=dict(
				scheme='nmf',
				init='nndsvd',
				alpha=7e-1,
				update=[
					['cd',int(1e2),1e-14],
					# ['mu',int(1e2),1e-14],
					# ['cd',int(1e2),1e-14]
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
		_data = {index:(data,where) for index,(data,where) in enumerate((data,where) for i in [*range(0,N-1,2),*range(1,N-1,2)] for where in [(i,i+1)] for data in ['unitary',['depolarize','depolarize']])}

		_kwargs = dict(
			D=D,N=N,M=M,
			parameters=parameters,	
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


		spectrum = basis.spectrum(state,where=L,**{**kwargs,**dict(options={**kwargs['options'],'scheme':'probability'})})
		_spectrum = basis.spectrum(state,where=L,**{**kwargs,**dict(options={**kwargs['options'],'scheme':'spectrum'})})

		spectrum = spectrum/maximum(absolute(spectrum))
		_spectrum = _spectrum/maximum(absolute(_spectrum))

		state = basis.transform(state,transform=False,**{**kwargs,**dict(D=D,N=None)})
		_state = basis.transform(_state,transform=False,**{**_kwargs,**dict(D=D,N=None)})

		tmp = np.sort((state).real)
		_tmp = np.sort((_state).real)
		print(array([*tmp[-1:-10:-1],*tmp[:10]]))
		print(array([*_tmp[-1:-10:-1],*_tmp[:10]]))

		print((absolute(state-_state)**2).sum())
		print(state.real.sum(),abs(state.real.sum()-1),abs(_state.real.sum()-1))

		data = dict(
			D=D,N=N,M=M,L=L,K=K,parameters=parameters,seed=seed,
			**{'spectrum.nmf':spectrum,'spectrum.svd':_spectrum}
			)

		dump(data,path)


	else:
		data = load(path)


	fig,ax = None,None
	settings = [
		{
			"fig": {
				"set_size_inches": {
					"w": 24,
					"h": 16
				},
				"subplots_adjust": {},
				"tight_layout": {},
				"savefig": {
					"fname": 'data/plot.pdf',
					"bbox_inches": "tight",
					"pad_inches": 0.2
				}
			},
			"ax":{
				"errorbar":{
					"x":[*arange(len(data[y]))] if x is None else x,
					'y':[*data[y]],
					"label":{'spectrum.nmf':'$\\textrm{NMF}$','spectrum.svd':'$\\textrm{SVD}$'}.get(y),
					"alpha":0.8,
					"marker":"o",
					"markersize":8,
					"linestyle": "--",
					"linewidth": 4,
					"elinewidth": 2,
					"capsize": 3,
					"color":{'spectrum.nmf':'black','spectrum.svd':'gray'}.get(y),
					},
				"set_title": {
					"label": f"$\\textrm{{Haar + Depolarize}} \\quad N = {N} ~,~ M = 2N ~,~ L = N/2 ~,~ D = {D} ~,~ \\chi = D^{{N}} ~,~ \\gamma = 10^{{{int(log10(parameters))}}}$",
					"pad":20,
					},
				"set_ylabel": {
					"ylabel": "$\\textrm{Spectrum} ~~ {\\sigma_{i}}/{\\sigma_{\\textrm{max}}}$"
				},
				"set_xlabel": {
					"xlabel": "$\\textrm{Spectrum Index} ~~ i$"
				},
				"set_xscale": {"value": "linear"},
				"set_xlim":{"xmin":-15,"xmax":265},
				"set_xticks":{"ticks":[0,50,100,150,200,250]},
				"set_yscale": {"value": "log","base": 10},
				"set_ylim":{"ymin":1e-17,"ymax":1e1},
				"set_yticks":{"ticks":[1e-16,1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0]},
				"set_ylim":{"ymin":1e-7,"ymax":1e1},				
				"set_yticks":{"ticks":[1e-6,1e-4,1e-2,1e0]},
				"set_aspect": {
					"aspect": "auto"
				},
				"grid": {
					"visible": True,
					"which": "major",
					"axis": "both"
				},
				"legend":{
					"title":"$\\textrm{Scheme}$",
					"title_fontsize": 36,
					"fontsize": 36,
					"markerscale": 2,
					"handlelength": 3,
					"framealpha": 0.8,				
				},
			},
			"style":{
				"mplstyle": "data/plot.mplstyle"
			}
		}
		for index,(x,y) in enumerate(((None,'spectrum.nmf'),(None,'spectrum.svd')))
		]
	for index,settings in enumerate(settings):
		fig,ax = plot(settings=settings,fig=fig,ax=ax)

	exit()

	assert allclose(basis.transform(state,transform=False,**{**kwargs,**dict(D=D,N=None)}).sum(),1) and allclose(prod(_state[i].sum() for i in _state),1)


	state = basis.transform(state,transform=False,**{**kwargs,**dict(D=D,N=None)})
	_state = basis.transform(_state,transform=False,**{**_kwargs,**dict(D=D,N=None)})


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
