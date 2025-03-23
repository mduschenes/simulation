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

configs = {
	'jax_disable_jit':False,
	'jax_platforms':'',
	'jax_platform_name':'',
	'jax_enable_x64': True,
	}
for name in configs:
	jax.config.update(name,configs[name])

from jax import jit,vmap
import jax.numpy as np
import jax.scipy as sp
import numpy as onp
import scipy as osp
import opt_einsum
from math import prod
from functools import partial
import itertools
import time as timing

from string import ascii_lowercase as characters

backend = 'jax'



np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.6e')) for dtype in ['float','float64',np.float64,np.float32,'complex','complex128',np.complex128]}})

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

def debug(obj=None,**options):
	if obj is None and options:
		obj,options = ' '.join([f'{option} = {{{option}}}' for option in options]),options
	elif obj is None:
		obj,options = '',options		
	elif not isinstance(obj,str):
		obj,options = '{obj}',{**dict(obj=obj),**options}
	jax.debug.print(obj,**options)
	return

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
				elif isinstance(obj[name],iterables) and all(isinstance(data,iterables) for data in obj[name]):
					try:
						path[key] = obj[name]
					except:
						size = max(len(data) for data in obj[name])
						try:
							path[key] = type(obj[name])([type(data)([*data,*[nan]*(size-len(data))]) for data in obj[name]])
						except:
							path[key] = nparray(obj[name],dtype='S')						
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

	def children(obj,attr=None):
		'''
		Return object relative to to obj
		Args:
			obj (object): Object instance
			attr (str): attribute
		Returns:
			instance (object): Object instance
		'''

		AXES = ['x','y','z']
		CHILDREN = ['twin']

		if attr is None:
			instance = obj
		elif attr in ['%s%s'%(children,axes) for children in CHILDREN for axes in AXES]:
			instance = None
			axes = attr[-1]
			siblings = getattr(obj,'get_shared_%s_axes'%(axes))().get_siblings(obj)
			for sibling in siblings:
				if sibling.bbox.bounds == obj.bbox.bounds and sibling is not obj:
					instance = sibling
					break
			if instance is None:
				instance = getattr(obj,attr)()
		else:
			instance = obj

		return instance



	defaults = {'fig':{},'ax':{},'style':{}}
	options = {'instance':['plot','errorbar'],'obj':{'ax':ax,'fig':fig},'data':['x','y','xerr','yerr'],'plot':["label","alpha","marker","markeredgecolor","markeredgewidth","markersize","linestyle","linewidth","elinewidth","capsize","color","ecolor"],"null":['obj']}
	for key in defaults:
		if key not in settings:
			settings[key] = {}
		settings[key].update({attr: defaults[key][attr] for attr in defaults[key] if attr not in settings})

	for key in settings:
		for attr in settings[key]:
			setting = settings[key][attr]
			settings[key][attr] = [setting] if not isinstance(setting,list) else [*setting]

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
				
				for setting in settings[key][attr]:

					if setting is None:
						continue

					obj = children(options['obj'][key],setting.get('obj'))
					for attr in attr.split('.'):
						obj = getattr(obj,attr)				
							
					args = tuple((setting[option] for option in options['data'] if setting.get(option) is not None))
					kwargs = dict({option:setting[option] for option in setting if option in options['plot'] and option not in options['null']})
					
					obj(*args,**kwargs)

		keys = ['ax','fig']
		for key in keys:
			if settings.get(key) is None:
				continue
			for attr in settings[key]:
				if attr in options['instance']:
					continue
				
				for setting in settings[key][attr]:

					if setting is None:
						continue

					obj = children(options['obj'][key],setting.get('obj'))
					for attr in attr.split('.'):
						obj = getattr(obj,attr)

					args = list()
					kwargs = dict({option:setting[option] for option in setting if option not in options['null']})

					if attr is None:
						pass
					elif attr in ['legend']:
						handles,labels = zip(*[data for ax in children(options['obj'][key],setting.get('obj')).get_figure().axes for data in zip(*getattr(ax,'get_legend_handles_labels')())])
						args.extend([handles,labels])

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

def asarray(*args,**kwargs):
	return np.asarray(*args,**kwargs)

def asscalar(a,*args,**kwargs):
	try:
		return a.item()
	except (AttributeError,ValueError,TypeError):
		try:
			return onp.asscalar(a,*args,**kwargs)
		except:
			return a

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

def rand(key,shape=(),dtype=None,minval=0,maxval=1,**kwargs):
	return rng.uniform(key,shape=shape,dtype=dtype,minval=minval,maxval=maxval)

def randint(key,shape=(),dtype=None,minval=0,maxval=1,**kwargs):
	return rng.randint(key,shape=shape,dtype=dtype,minval=minval,maxval=maxval)

def randn(key,shape=(),dtype=None,**kwargs):
	return rng.normal(key,shape=shape,dtype=dtype)

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

def svd(a,compute_uv=True,full_matrices=False,hermitian=False,**kwargs):
	# data = onp.linalg.svd(a,full_matrices=full_matrices,compute_uv=compute_uv,hermitian=hermitian)
	# if isinstance(data,tuple):
	# 	data = tuple(asarray(i) for i in data)
	# else:
	# 	data = asarray(data)
	# return data
	# return np.linalg.svd(a,compute_uv=compute_uv,full_matrices=full_matrices,hermitian=hermitian)
	# return sp.linalg.svd(a,compute_uv=compute_uv,full_matrices=full_matrices,lapack_driver="gesvd")
	return jax.lax.linalg.svd(a,compute_uv=compute_uv,full_matrices=full_matrices)

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

def eig(a,compute_v=True,hermitian=False,**kwargs):
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

def eigs(a,compute_v=True,hermitian=False,**kwargs):
	return eig(a,compute_v=compute_v,hermitian=hermitian,**kwargs)

def chol(a,lower=False,**kwargs):
	return sp.linalg.cholesky(a,lower=lower)

def lu(a,**kwargs):
	return sp.linalg.lu(a)

def nystrom(a,z):
	# method = 'cholesky'
	# x = dot(a,z)
	# t = norm(x)
	# t = 0
	# x + t*z
	# x = solve(dot(z.T,x),x.T,method=method,lower=True)
	# u,s,v = svd(x)
	# s = maximums(sqr(s)-t,0)

	x = dot(a,z)
	x = dot(x,dot(inv(dot(z.T,x)),x.T))
	s,u = eig(x,hermitian=True)
	return u,s

def inv(a,**kwargs):
	return np.linalg.inv(a)

def lstsq(a,b,**kwargs):
	x, resid, rank, s = np.linalg.lstsq(a,b)
	return x

def solve(a,b,method=None,**kwargs):
	if method is None:
		return solve_solve(a,b,**kwargs)
	elif method in ['triangular']:
		return solve_triangular(a,b,**kwargs)
	elif method in ['cholesky']:
		return solve_chol(a,b,**kwargs)	
	elif method in ['lu']:
		return solve_lu(a,b,**kwargs)				
	else:
		return solve_solve(a,b,**kwargs)		

def solve_solve(a,b,**kwargs):
	return sp.linalg.solve(a,b)

def solve_triangular(a,b,lower=False,**kwargs):
	return sp.linalg.solve_triangular(a,b,lower=lower)

def solve_chol(a,b,lower=False,**kwargs):
	return sp.linalg.cho_solve(sp.linalg.cho_factor(a,lower=lower),b)

def solve_lu(a,b,**kwargs):
	return sp.linalg.lu_solve(lu(a),b)

def condition_number(a,**kwargs):
	return np.linalg.cond(a)

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

def squeeze(a,axis=None):
	return np.squeeze(a,axis=axis)

def conjugate(a):
	return a.conjugate()

def dagger(a):
	return conjugate(transpose(a))

def dot(a,b,axes=1):
	return np.tensordot(a,b,axes=axes)

def tensordot(a,b,axes=2):
	return np.tensordot(a,b,axes=axes)

def inner(a,b):
	return np.inner(a,b)

def outer(a,b):
	return np.outer(a,b)

def kron(a,b):
	return np.kron(a,b)

def tensorprod(a):
	out = a[0]
	for i in range(1,len(a)):
		out = kron(out,a[i])
	return out

def symbols(index):
	return opt_einsum.get_symbol(index)
	# return characters[index]

def einsum(subscripts,*operands,backend=backend):
	return opt_einsum.contract(subscripts,*operands,backend=backend)
	# return np.einsum(subscripts,*operands)

def norm(a):
	return add(abs2(a))

def expm(a,n=16):
	return sp.linalg.expm(a,max_squarings=n)

def logm(a):
	return sp.linalg.logm(a)

def sqrtm(a):
	return sp.linalg.sqrtm(a)

def exp(a):
	return np.exp(a)

def log(a):
	return np.log(a)

def log10(a):
	return np.log10(a)

def sqrt(a):
	return np.sqrt(a)

def sqr(a):
	return a**2

def sin(a):
	return np.sin(a)

def cos(a):
	return np.cos(a)

def tan(a):
	return np.tan(a)

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

def nndsvd(a,u=None,v=None,rank=None,**kwargs):

	# u,s,v = svds(a)
	# u,v,s = absolute(u),absolute(dotl(v,s)),ones(s.shape)

	# return u,v,s

	if 0:
		def true(u_plus,v_plus,s_plus,u_minus,v_minus,s_minus):
			return u_plus,v_plus,s_plus

		def false(u_plus,v_plus,s_plus,u_minus,v_minus,s_minus):
			return u_minus,v_minus,s_minus

		def func(u,v,s):

			u_plus,v_plus = maximums(u,0),maximums(v,0)
			u_minus,v_minus = maximums(-u,0),maximums(-v,0)
			u_plus_norm,v_plus_norm = norm(u_plus),norm(v_plus)
			u_minus_norm,v_minus_norm = norm(u_minus),norm(v_minus)

			s_plus,s_minus = s*u_plus_norm*v_plus_norm,s*u_minus_norm*v_minus_norm

			u_plus,v_plus = u_plus/(u_plus_norm+eps),v_plus/(v_plus_norm+eps)
			u_minus,v_minus = u_minus/(u_minus_norm+eps),v_minus/(v_minus_norm+eps)

			u,v,s = cond(s_plus>s_minus,true,false,u_plus,v_plus,s_plus,u_minus,v_minus,s_minus)

			u,v,s = sqrt(s)*u,sqrt(s)*v,1

			return u,v,s

		rank = min(a.shape) if rank is None else rank        
		eps = epsilon(a.dtype)

		func = vmap(func,in_axes=(1,0,0),out_axes=(1,0,0))

		u,s,v = svds(a)

		u,v,s = func(u,v,s)

	else:
		def true(z_plus,x_plus,y_plus,z_minus,x_minus,y_minus):
			return z_plus,x_plus,y_plus

		def false(z_plus,x_plus,y_plus,z_minus,x_minus,y_minus):
			return z_minus,x_minus,y_minus  

		def func(i,x):
			
			u,v,s = x

			z,x,y = s[i],u[slices,i],v[i,slices]

			z,x,y = real(z),real(x),real(y)

			x_plus,y_plus = maximums(x,0),maximums(y,0)
			x_minus,y_minus = maximums(-x,0),maximums(-y,0)
			x_plus_norm,y_plus_norm = norm(x_plus),norm(y_plus)
			x_minus_norm,y_minus_norm = norm(x_minus),norm(y_minus)
			z = maximums(z,eps)

			z_plus,z_minus = z*x_plus_norm*y_plus_norm,z*x_minus_norm*y_minus_norm

			x_plus,y_plus = x_plus/(x_plus_norm+eps),y_plus/(y_plus_norm+eps)
			x_minus,y_minus = x_minus/(x_minus_norm+eps),y_minus/(y_minus_norm+eps)

			z,x,y = cond(z_plus>z_minus,true,false,z_plus,x_plus,y_plus,z_minus,x_minus,y_minus)

			u,v,s = inplace(u,(slices,i),sqrt(z)*x),inplace(v,(i,slices),sqrt(z)*y),inplace(s,i,1)
			
			x = u,v,s

			# debug(up=x_plus,um=x_minus,u=u[slices,i],v=v[i,slices])
			# debug(sp=z_plus,sm=z_minus)
			# debug(None)

			return x

		slices = slice(None)
		eps = epsilon(a.dtype)
		rank = min(a.shape) if rank is None else rank        
		u,s,v = svds(a)

		loop = forloop
		options = dict(start=0,end=rank)
		x = (u,v,s)
		x = loop(func=func,x=x,**options)
		u,v,s = x

	return u,v,s

def nndsvda(a,u=None,v=None,rank=None,**kwargs):
	u,v,s = nndsvd(a,u=u,v=v,rank=rank) 
	
	x = maximums(mean(a),epsilon(a.dtype))
	u,v = inplace(u,u==0,x),inplace(v,v==0,x)

	return u,v,s

def randd(a,u=None,v=None,rank=None,**kwargs):
	n,m = a.shape
	k = min(min(n,m),rank)
	dtype = a.dtype
	x = sqrt(mean(a)/n)
	u,v,s = (
		absolute(x*randn(**{**{kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg in ['shape','key','dtype']},**dict(shape=(n,k),dtype=dtype)})),
		absolute(x*randn(**{**{kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg in ['shape','key','dtype']},**dict(shape=(k,m),dtype=dtype)})),
		ones(k,dtype=dtype)		
		)
	return u,v,s

def initd(a,u=None,v=None,rank=None,**kwargs):
	n,m = a.shape
	k = min(min(n,m),rank)
	dtype = a.dtype
	u,v,s = real(u),real(v),ones(k,dtype=dtype)
	return u,v,s

# def search(parameters,func,grad,search,iteration,eps,bounds,constants,conditions='wolfe',method='zoom'):

# 	def zoom(alpha,beta):
# 		return

# 	def true(alpha,beta):
# 		zoom()

# 	def line(data):
# 		alpha,beta,data,i = data
# 		alpha = (alpha + bounds[-1])/2

# 		alpha *= iota
# 		i += 1
# 		data = alpha,*data,i
# 		return data

# 	if conditions in ['wolfe']:
# 		def cond(data):
# 			alpha,x,p,b,A,i = data
# 			return (i < iteration) + (-dot(f(x+alpha*p,b,A),p) < sigma*alpha*norm(p))


# 	alpha = bounds[0]
# 	beta = alpha

# 	parameter = bounds[0] + bounds
# 	def search(data):
# 		i = 0
# 		data = alpha,*data,i
# 		data = whileloop(func=line,x=data,cond=condition)
# 		parameters,*data = data
# 		return parameters


def gradient_descent(a,u,v,rank=None,**kwargs):

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
	
	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	def func(x):
		
		u,v,a,i = x

		# v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),0)

		# u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),0).T

		z = dot(u,v)
		alpha = einsum('ij,ij',a,z)/einsum('ij,ij',z,z)

		v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),eps)
		u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),eps).T


		u,v = u,v/add(dot(u,v))
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


		# v = 	optimizer.run(
		# 	init_params=v,
		# 	data=(u,a),
		# 	**options
		# 	).params

		# u = 	optimizer.run(
		# 	init_params=u.T,
		# 	data=(v.T,a.T),
		# 	**options			
		# 	).params.T

		i += 1

		x = u,v,a,i

		return x

	return func

def riemannian_gradient(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)
	n,m = a.shape
	In,Im,n1,m1,nm1 = identity(n),identity(m),ones(n),ones(m),outer(ones(n),ones(m))

	def update(i,x):
		t,s,g,h = x
		t += dot(g,t)/i
		s += dot(h,s)/i
		x = t,s,g,h
		return x
	loop = forloop

	def func(x):
		
		u,v,a,i = x

		z = dot(u,v)
		l = z-a
		s,t = dot(z.T,z),dot(z,z.T)
		e = dot(dot(n1,dot(l,s)+dot(t,l)),m1)/(m*dot(n1,dot(t,n1)) + n*dot(m1,dot(s,m1)))
		k = l - e*nm1
		g,h = -alpha*dot(dot(k,z.T),u),-alpha*dot(v,dot(z.T,k))
		u,v = u + g,v + h
		
		# t,s = expm(g,iteration),expm(h,iteration)
		# u,v = dot(t,u),dot(v,s)

		# t,s = ones(g.shape,dtype=g.dtype),ones(h.shape,dtype=h.dtype)
		# x = t,s,g,h
		# x = loop(func=update,x=x,start=1,end=iteration)
		# t,s,g,h = x
		# u,v = dot(t,u),dot(v,s)
		
		u,v = maximums(u,eps),maximums(v,eps)

		u,v = u,v/add(dot(u,v))

		# u,v = dot(In - alpha*dot(k,z.T),u),dot(v,Im - alpha*dot(z.T,k))
		# u,v = maximums(dot(In - alpha*dot(k,z.T),u),eps),maximums(dot(v,Im - alpha*dot(z.T,k)),eps)

		i += 1

		x = u,v,a,i

		return x

	return func

def conjugate_descent(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	def function(data):
		x,r,p,b,A,i = data
		q = dot(A,p)
		alpha = dot(r,r)/dot(p,q)
		x = maximums(x + alpha*p,eps)
		r = r + alpha*q
		beta = dot(r,r)/dot(*(r-alpha*q,)*2)
		p = -r + beta*p
		i += 1
		data = x,r,p,b,A,i
		# debug(f=f(x,b,A),x=norm(x),p=norm(p),r=norm(r),alpha=alpha)
		return data

	def f(x,b,A):
		return norm(dot(A,x) - b)

	def cond(data):
		x,r,p,b,A,i = data
		return ((i<=1) + ((i<iteration) * (f(x,b,A) > eps) * (norm(p)>eps) * (norm(r)>eps)))

	def init(u,v,a):
		x = v
		A = dot(u.T,u)
		b = dot(u.T,a)
		r = dot(A,x) - b
		p = -r
		i = zeros(shape=x.shape[-1],dtype=int)
		data = x,r,p,b,A,i
		return data

	def loop(*data):
		func = function
		options = dict(cond=cond)
		data = whileloop(func=func,x=data,**options)
		x,*data = data
		return x

	loop = vmap(loop,in_axes=(1,1,1,1,None,0),out_axes=1)

	def func(x):
		
		u,v,a,i = x

		v = loop(*init(u,v,a))

		u = loop(*init(v.T,u.T,a.T)).T

		i += 1

		x = u,v,a,i

		return x

	return func


def conjugate_projection(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)
	beta = kwargs.get('beta',1)
	gamma = kwargs.get('gamma',1)
	delta = kwargs.get('delta',1)
	iota = kwargs.get('iota',1)	
	sigma = kwargs.get('sigma',1)
	I = identity(rank)

	def function(data):
		x,g,r,p,b,A,i = data

		p = -g + (dot(g,r)*p - dot(g,p)*(r))/maximums(maximums(maximums(1*delta*norm(p)*norm(r),dot(p,r)),norm(g-r)),eps)

		alpha = search((x,p,b,A))
		
		y = x + alpha*p
		s = f(y,b,A)
		
		beta = dot(s,y-x)

		x = maximums(x+beta*s,eps)

		g,r = f(x,b,A),g
		r = g-r

		i += 1
		
		data = x,g,r,p,b,A,i
		# debug(f=norm(f(x,b,A)),x=norm(x),p=norm(p),r=norm(r),alpha=alpha)
		return data

	def f(x,b,A):
		return dot(A,x) - b

	def line(data):
		alpha,*data,i = data
		alpha *= iota
		i += 1
		data = alpha,*data,i
		return data

	def condition(data):
		alpha,x,p,b,A,i = data
		return (i < iteration) + (-dot(f(x+alpha*p,b,A),p) < sigma*alpha*norm(p))

	def search(data):
		i = 0
		data = alpha,*data,i
		data = whileloop(func=line,x=data,cond=condition)
		parameters,*data = data
		return parameters

	def cond(data):
		x,*data,b,A,i = data
		return ((i<=1) + (i<iteration) * (norm(f(x,b,A)) > eps))

	def init(u,v,a):
		x = v
		A = dot(u.T,u) + gamma*I
		b = dot(u.T,a)
		g = f(x,b,A)
		r = g-g
		p = -g
		i = zeros(shape=x.shape[-1],dtype=int)
		data = x,g,r,p,b,A,i
		return data

	def loop(*data):
		func = function
		options = dict(cond=cond)
		data = whileloop(func=func,x=data,**options)
		x,*data = data
		return x

	loop = vmap(loop,in_axes=(1,1,1,1,1,None,0),out_axes=1)

	def func(x):
		
		u,v,a,i = x

		v = loop(*init(u,v,a))

		u = loop(*init(v.T,u.T,a.T)).T

		i += 1

		x = u,v,a,i

		return x

	return func


def precondition_conjugate(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)
	beta = kwargs.get('beta',1)
	gamma = kwargs.get('gamma',1)
	delta = kwargs.get('delta',1)
	iota = kwargs.get('iota',1)
	sigma = kwargs.get('sigma',1)
	l = kwargs.get('l',rank)
	I = identity(rank)

	def function(data):
		x,g,r,p,b,A,i = data

		p = -g + (dot(g,r)*p - dot(g,p)*(r))/maximums(maximums(maximums(1*delta*norm(p)*norm(r),dot(p,r)),norm(g-r)),eps)

		alpha = search((x,p,b,A))
		
		y = x + alpha*p
		s = f(y,b,A)
		
		beta = dot(s,y-x)

		x = maximums(x+beta*s,eps)

		g,r = f(x,b,A),g
		r = g-r

		i += 1
		
		data = x,g,r,p,b,A,i
		# debug(f=norm(f(x,b,A)),x=norm(x),p=norm(p),r=norm(r),alpha=alpha)
		return data

	def f(x,b,A):
		return dot(A,x) - b

	def line(data):
		alpha,*data,i = data
		alpha *= iota
		i += 1
		data = alpha,*data,i
		return data

	def condition(data):
		alpha,x,p,b,A,i = data
		return (i < iteration) + (-dot(f(x+alpha*p,b,A),p) < sigma*alpha*norm(p))

	def search(data):
		i = 0
		data = alpha,*data,i
		data = whileloop(func=line,x=data,cond=condition)
		parameters,*data = data
		return parameters

	def cond(data):
		x,*data,b,A,i = data
		return ((i<=1) + (i<iteration) * (norm(f(x,b,A)) > eps))

	def init(u,v,a):
		x = v
		A = dot(u.T,u)
		b = dot(u.T,a)

		P = A
		S,U = eig(P,compute_v=True,hermitian=True)
		P = dot(U[:,:-l]*S[:-l],U[:,:-l].T)
		# P = randn(**{**kwargs,**dict(shape=A.shape,dtype=A.dtype)})
		# Q = dot(P,A)
		# P = dot(Q,solve(dot(P.T,Q),Q.T))
		# S,U = eig(P,compute_v=True,hermitian=True)
		# P = (S[-l]+gamma)*dot((U/(S+gamma*I)),U.T) + (I - dot(U,U.T))


		debug(k=(condition_number(A),condition_number(dot(P,A))))


		A = dot(P,A)
		b = dot(P,A)

		g = f(x,b,A)
		r = g-g
		p = -g
		i = zeros(shape=x.shape[-1],dtype=int)
		data = x,g,r,p,b,A,i
		return data

	def loop(*data):
		func = function
		options = dict(cond=cond)
		data = whileloop(func=func,x=data,**options)
		x,*data = data
		return x

	loop = vmap(loop,in_axes=(1,1,1,1,1,None,0),out_axes=1)

	def func(x):
		
		u,v,a,i = x

		x,g,r,p,b,A,j = init(u,v,a)

		x = u,v,a,i
		return x

		v = loop(*init(u,v,a))

		u = loop(*init(v.T,u.T,a.T)).T

		i += 1

		x = u,v,a,i

		return x

	return func


def conjugate_gradient(a,u,v,rank=None,**kwargs):

	from jaxopt import NonlinearCG as optimizer

	iteration = kwargs.get('iteration',100)
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)
	gamma = kwargs.get('gamma',1)
	I = identity(rank)

	options = dict(
		maxiter=(kwargs.get('maxiter',kwargs.get('iteration')) if isinstance(kwargs.get('maxiter',kwargs.get('iteration')),int) else kwargs.get('solver',kwargs.get('update'))[0][1] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else 100),
		maxls=max(1,(kwargs.get('maxiter',kwargs.get('iteration')) if isinstance(kwargs.get('maxiter',kwargs.get('iteration')),int) else kwargs.get('solver',kwargs.get('update'))[0][1] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else 100)/10),
		tol=kwargs.get('tol',kwargs.get('eps')) if isinstance(kwargs.get('tol',kwargs.get('eps')),float) else kwargs.get('solver',kwargs.get('update'))[0][2] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else epsilon(a.dtype),
		method='hestenes-stiefel',
		linesearch_init='current',
		increase_factor=1+1e-3,
		max_stepsize=2e-1,
		min_stepsize=1e-6,
		)

	def fun(x,b,A):
		return norm(dot(A,x)-b)

	def init(u,v,a):
		x = v
		A = dot(u.T,u) + gamma*I
		b = dot(u.T,a)
		data = x,b,A
		return data

	def loop(*data):
		data = optimizer(fun=fun,**options).run(*data).params
		return data

	loop = vmap(loop,in_axes=(1,1,None),out_axes=1)

	def func(x):
		
		u,v,a,i = x

		# x,b,A = init(u,v,a)
		# debug(f=fun(x,b,A))

		# x = u,b,a,i
		# return x

		v = loop(*init(u,v,a))

		u = loop(*init(v.T,u.T,a.T)).T

		# # v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),0)

		# # u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),0).T

		# z = dot(u,v)
		# alpha = einsum('ij,ij',a,z)/einsum('ij,ij',z,z)

		# v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),eps)
		# u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),eps).T

		i += 1

		x = u,v,a,i

		return x

	return func


def step_descent(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1e-1)
	sigma = kwargs.get('sigma',1e-2)

	def function(x):

		u,v,x,y,a,alpha,i = x

		v,y = maximums(y-alpha*(dot(dot(x.T,x),y)-dot(x.T,a)),eps),v
		u,x = u,x
		# u,x = maximums(x.T-alpha*(dot(dot(y,y.T),x.T)-dot(y,a.T)),eps).T,u

		alpha *= alpha

		i += 1

		x = u,v,x,y,a,alpha,i

		return x

	def _function(x):
		u,v,x,y,a,alpha,i = x

		# v,y = maximums(y-alpha*(dot(dot(x.T,x),y)-dot(x.T,a)),eps),v
		v,y = v,y
		u,x = maximums(x.T-alpha*(dot(dot(y,y.T),x.T)-dot(y,a.T)),eps).T,u

		alpha *= alpha

		i += 1

		x = u,v,x,y,a,alpha,i

		return x

	def error(u,v,a):
		return norm(a-dot(u,v))
	def _error(u,v,a):
		return norm(a-dot(u,v))

	def step(u,v,a,z,alpha):
		return alpha*einsum('ij,ij',dot(dot(u.T,u),v)-dot(u.T,a),z)
	def _step(u,v,a,z,alpha):
		return alpha*einsum('ij,ij',dot(dot(v,v.T),u.T)-dot(v,a.T),z.T)

	def cond(x):
		u,v,x,y,a,alpha,i = x
		debug(error=(error(x,y,a)-error(u,v,a)),step=-step(x,y,a,v-y,sigma))
		return ((i<=1) + ((i<iteration) * ((error(x,y,a)-error(u,v,a)) < -step(x,y,a,v-y,sigma))))
	def _cond(x):
		u,v,x,y,a,alpha,i = x
		# debug(_error=(_error(u,v,a)-_error(x,y,a)),_step=_step(x,y,a,u-x,sigma))
		debug(error=(_error(x,y,a)-_error(u,v,a)),step=-_step(x,y,a,u-x,sigma))
		return ((i<=1) + ((i<iteration) * ((_error(x,y,a)-_error(u,v,a)) < -_step(x,y,a,u-x,sigma))))

	loop = whileloop

	options=dict(cond=cond)
	_options=dict(cond=_cond)

	def func(x):
		
		u,v,a,i = x


		# v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),0)

		# u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),0).T

		z = dot(u,v)
		alpha = einsum('ij,ij',a,z)/einsum('ij,ij',z,z)

		j = 0
		x = u,v,u,v,a,alpha,j
		x = loop(func=function,x=x,**options)
		u,v,x,y,a,alpha,j = x



		k = 0
		x = u,v,u,v,a,alpha,k
		x = loop(func=_function,x=x,**_options)
		u,v,x,y,a,alpha,k = x

		# v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),eps)
		# u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),eps).T

		debug(j=j,k=k)

		i += 1

		x = u,v,a,i

		return x

	return func


def quadratic_programming(a,u,v,rank=None,**kwargs):

	from qpax import solve_qp_primal as optimizer

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	options = dict(
		solver_tol=kwargs.get('tol',kwargs.get('eps')) if isinstance(kwargs.get('tol',kwargs.get('eps')),float) else kwargs.get('solver',kwargs.get('update'))[0]['eps'] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else epsilon(a.dtype),
		)

	function = vmap(partial(optimizer,**options),in_axes=(None,1,None,None,None,None),out_axes=1)
	_function = vmap(partial(optimizer,**options),in_axes=(None,1,None,None,None,None),out_axes=0)

	def func(x):
		
		u,v,a,i = x

		z = dot(u,v)
		alpha = sqrt(einsum('ij,ij',a,z)/einsum('ij,ij',z,z))

		debug(inv(dot(u.T,u)))
		debug(-dot(u.T,a))
		debug(None)
		debug(inv(dot(v,v.T)))
		debug(-dot(v,a.T))

		u *= alpha
		v *= alpha

		s = u.shape[-1]
		G,h,A,b = -identity(s),zeros(s),zeros((0,s)),zeros(0)
		Q,q,_Q,_q = dot(u.T,u),-dot(u.T,a),dot(v,v.T),-dot(v,a.T)

		# debug(Q=Q,q=q,_Q=_Q,_q=_q)

		v = function(Q,q,G,h,A,b)
		u = _function(_Q,_q,G,h,A,b)

		# # v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),0)

		# # u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),0).T

		# z = dot(u,v)
		# alpha = einsum('ij,ij',a,z)/einsum('ij,ij',z,z)

		# v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),eps)
		# u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),eps).T

		i += 1

		x = u,v,a,i

		return x

	return func

def least_squares(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)
	gamma = kwargs.get('gamma',1)
	I = identity(rank)

	def function(data):
		x,A,b,i = data
		x = solve_ch(A+gamma*I,b+gamma*x)
		i += 1
		data = x,A,b,i
		return data
	
	def cond(data):
		x,A,b,i = data
		return (i<iteration)

	def init(u,v,a):
		x = v
		A = dot(u.T,u)
		b = dot(u.T,a)
		i = 0
		data = x,A,b,i
		return data

	loop = whileloop

	def func(x):
		
		u,v,a,i = x

		data = init(u,v,a)
		data = loop(func=function,x=data,cond=cond)
		x,*data = data
		v = maximums(x,eps)

		data = init(v.T,u.T,a.T)
		data = loop(func=function,x=data,cond=cond)
		x,*data = data
		u = maximums(x.T,eps)

		# v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),0)

		# u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),0).T

		# z = dot(u,v)
		# alpha = sqrt(einsum('ij,ij',a,z)/einsum('ij,ij',z,z))

		# u *= alpha
		# v *= alpha

		# # for i in range(10):
		# v = maximums(lstsq(dot(u.T,u)+gamma*conditioner,dot(u.T,a)+gamma*v),eps)
		# # for i in range(10):
		# u = maximums(lstsq(dot(v,v.T)+gamma*conditioner,dot(v,a.T)+gamma*u.T),eps).T

		# debug(v=v)
		# debug(u=u)
		# debug(None)
		# u /= alpha
		# v /= alpha

		i += 1

		x = u,v,a,i

		return x

	return func

def projective_descent(a,u,v,rank=None,**kwargs):

	from jaxopt import ProjectedGradient as optimizer
	from jaxopt.projection import projection_non_negative as projection

	iteration = kwargs.get('iteration',100)
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	options = dict(
		projection=projection,
		maxiter=(kwargs.get('maxiter',kwargs.get('iteration')) if isinstance(kwargs.get('maxiter',kwargs.get('iteration')),int) else kwargs.get('solver',kwargs.get('update'))[0][1] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else 100),
		maxls=(kwargs.get('maxiter',kwargs.get('iteration')) if isinstance(kwargs.get('maxiter',kwargs.get('iteration')),int) else kwargs.get('solver',kwargs.get('update'))[0][1] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else 100)/10,
		tol=kwargs.get('tol',kwargs.get('eps')) if isinstance(kwargs.get('tol',kwargs.get('eps')),float) else kwargs.get('solver',kwargs.get('update'))[0][2] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else epsilon(a.dtype),
		)

	def fun(v,u,a):
		return norm(a-dot(u,v))
	function = optimizer(fun=fun,**options)

	def func(x):
		
		u,v,a,i = x

		v = function.run(v,u,a).params
		u = function.run(u.T,v.T,a.T).params.T

		# # v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),0)

		# # u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),0).T

		# z = dot(u,v)
		# alpha = einsum('ij,ij',a,z)/einsum('ij,ij',z,z)

		# v = maximums(v-alpha*(dot(dot(u.T,u),v)-dot(u.T,a)),eps)
		# u = maximums(u.T-alpha*(dot(dot(v,v.T),u.T)-dot(v,a.T)),eps).T

		i += 1

		x = u,v,a,i

		return x

	return func


def multiplicative_update(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	def func(x):
		
		u,v,a,i = x
		
		u *= (dot(a,transpose(v))/dot(u,dot(v,transpose(v))))
		v *= (dot(transpose(u),a)/dot(dot(transpose(u),u),v))

		i += 1

		x = u,v,a,i

		return x

	return func

def multiplicative_robust_update(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	def func(x):
		
		u,v,a,i = x

		z = 1/maximums(eps,sqrt(norm(a-dot(u,v))))
		
		u *= (dot(a*z,transpose(v))/dot(u,dot(v*z,transpose(v))))
		v *= (dot(transpose(u),a*z)/dot(dot(transpose(u),u),v*z))

		i += 1

		x = u,v,a,i

		return x

	return func

def multiplicative_iterative_update(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	def func_u(data):
		u,v,a,z,i = data
		u *= (dot(a*z,transpose(v))/dot(u,dot(v*z,transpose(v))))
		i += 1		
		data = u,v,a,z,i
		return data

	def func_v(data):
		u,v,a,z,i = data
		v *= (dot(transpose(u),a*z)/dot(dot(transpose(u),u),v*z))
		i += 1
		data = u,v,a,z,i
		return data

	def cond(data):
		u,v,a,z,i = data
		return (i<iteration)

	loop = whileloop

	def func(x):
		
		u,v,a,i = x

		k = 0
		z = 1/maximums(eps,sqrt(norm(a-dot(u,v))))
		data = u,v,a,z,k
		data = loop(func=func_u,x=data,cond=cond)
		u,v,a,z,k = data
		
		k = 0
		z = 1/maximums(eps,sqrt(norm(a-dot(u,v))))
		data = u,v,a,z,k
		data = loop(func=func_v,x=data,cond=cond)
		u,v,a,z,k = data

		i += 1

		x = u,v,a,i

		return x

	return func

def multiplicative_l1_update(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	def func(x):
		
		u,v,a,i = x

		z = 1/sqrt(sqr(a-dot(u,v)) + sqr(eps))
		
		u *= (a*dot(z,transpose(v))/dot(u,v)*dot(z,transpose(v)))
		v *= (dot(transpose(u),a)*z/dot(dot(transpose(u),u),v)*z)

		i += 1

		x = u,v,a,i

		return x

	return func

def multiplicative_beta_update(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)
	beta = kwargs.get('beta',2)

	def func(x):
		
		u,v,a,i = x

		z = dot(u,v)

		u *= (a*dot(z**(beta-2),v.T))/(dot(z**(beta-1),v.T))

		v *= dot(u.T,a*(z**(beta-2)))/dot(u.T,z**(beta-1))
		
		i += 1

		x = u,v,a,i

		return x

	return func


def multiplicative_hierarchical_update(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	def func_u(u,v,z,a):
		debug(u=dot(u,u),v=dot(v,v))
		return (dot(a,v) - dot(z-outer(u,v),v))/dot(v,v)

	def func_v(u,v,z,a):
		debug(u=dot(u,u),v=dot(v,v))
		return (dot(transpose(a),u) - dot(transpose(z-outer(u,v)),u))/dot(u,u)

	def func_uv(u,v,a):
		z = dot(u,v)
		alpha = sqrt(einsum('ij,ij',a,z)/einsum('ij,ij',z,z))
		return z,alpha


	func_u = vmap(func_u,in_axes=(1,0,None,None),out_axes=1)
	func_v = vmap(func_v,in_axes=(1,0,None,None),out_axes=0)

	def func(x):
		
		u,v,a,i = x

		for i in range(iteration):
			z,alpha = func_uv(u,v,a)
			u = func_u(u,v,z,a)
			z,alpha = func_uv(u,v,a)
			u *= alpha
			v *= alpha

		for i in range(iteration):
			z,alpha = func_uv(u,v,a)
			v = func_v(u,v,z,a)
			z,alpha = func_uv(u,v,a)
			u *= alpha
			v *= alpha	

		for i in range(iteration):
			z,alpha = func_uv(u,v,a)
			u = func_u(u,v,z,a)
			v = func_v(u,v,z,a)
			z,alpha = func_uv(u,v,a)
			u *= alpha
			v *= alpha	


		# u = func_u(u,v,z,a)
		# v = func_v(u,v,z,a)
		
		debug()
		debug(dot(*(u[:,i],)*2))
		debug(dot(*(v[i,:],)*2))
		debug()

		i += 1

		x = u,v,a,i

		return x

	return func

def inverse_update(a,u,v,rank=None,**kwargs):

	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps',epsilon(a.dtype))
	alpha = kwargs.get('alpha',1)

	def func(x):
		
		u,v,a,i = x

		v = maximums(dot(dot(inv(dot(transpose(u),u)),transpose(u)),a),eps)
		u = maximums(dot(a,dot(transpose(v),inv(dot(u,transpose(u))))),eps)

		i += 1

		x = u,v,a,i

		return x

	return func


def nmf(a,u=None,v=None,rank=None,**kwargs):
	
	def init(a,u=None,v=None,rank=None,**kwargs):

		init = kwargs.get('init')
		
		if callable(init):
			pass
		elif init in ['nndsvd']:
			init = nndsvd
		elif init in ['nndsvda']:
			init = nndsvda						
		elif init in ['random']:		
			init = randd
		elif u is not None and v is not None:
			init = initd
		elif init is None:
			init = randd
		else:
			init = randd

		u,v,s = init(a,u=u,v=v,rank=rank,**kwargs)

		return u,v,s
	
	def call(a,u=None,v=None,rank=None,**kwargs):

		update = kwargs.get('update',None)
		iteration = kwargs.get('iteration',100)
		eps = kwargs.get('eps',epsilon(a.dtype))

		updates = [{'update':update,'iteration':iteration,'eps':eps} if isinstance(update,str) else update] if not isinstance(update,iterables) else update

		for update in updates:

			update,iteration,eps,keywords = update.get('update'),update.get('iteration'),update.get('eps'),update.get('kwargs',{})

			if callable(update):
				pass
			elif update is None:
				update = gradient_descent
			elif update in ['gd','gradient_descent']:
				update = gradient_descent
			elif update in ['rg','riemannian_gradient']:
				update = riemannian_gradient				
			# elif update in ['cg','conjugate_descent']:
			# 	update = conjugate_descent				
			elif update in ['cg','conjugate_gradient']:
				update = conjugate_gradient								
			elif update in ['pd','projective_descent']:
				update = projective_descent	
			elif update in ['cp','conjugate_projection']:
				update = conjugate_projection	
			elif update in ['pc','precondition_conjugate']:
				update = precondition_conjugate									
			elif update in ['sd','step_descent']:
				update = step_descent	
			elif update in ['ls','least_squares']:
				update = least_squares
			elif update in ['qp','quadratic_programming']:
				update = quadratic_programming
			elif update in ['mu','multiplicative_update']:
				update = multiplicative_update
			elif update in ['mru','multiplicative_robust_update']:
				update = multiplicative_robust_update			
			elif update in ['m1u','multiplicative_l1_update']:
				update = multiplicative_l1_update				
			elif update in ['mbu','multiplicative_beta_update']:
				update = multiplicative_beta_update		
			elif update in ['mhu','multiplicative_hierarchical_update']:
				update = multiplicative_hierarchical_update							
			elif update in ['miu','multiplicative_iterative_update']:
				update = multiplicative_iterative_update							
			elif update in ['inv','inverse_update']:
				update = inverse_update
				iteration = 1
			else:
				update = gradient_descent

			i = 0
			x = (u,v,a,i)
			func = update(a,u=u,v=v,rank=rank,**{**kwargs,**keywords,**dict()})

			loop = whileloop
			func = func
			options = dict(cond=(lambda x,iteration=iteration,eps=eps: (status(x,iteration=iteration,eps=eps))))

			x = loop(func=func,x=x,**options)

			u,v,a,i = x

		return u,v,s
   
	def status(x,iteration=None,eps=None):
		u,v,a,i = x
		e = norm(a-dot(u,v))
		e = (e > eps)*(i < iteration)
		return e


	rank = min(a.shape) if rank is None else rank    

	u,v,s = init(a,u=u,v=v,rank=rank,**kwargs)

	u,v,s = call(a,u=u,v=v,rank=rank,**kwargs)

	return u,v,s
	

def _nmf(a,**kwargs):

	from sklearn.decomposition import NMF as model

	u,v = kwargs.get('W',kwargs.get('u')),kwargs.get('H',kwargs.get('v'))
	rank = min(kwargs.get('rank') if kwargs.get('rank') is not None else min(a.shape),*a.shape)
	iteration = kwargs.get('iteration',100)	
	eps = kwargs.get('eps') if kwargs.get('eps') is not None else epsilon(a.dtype)

	a = nparray(inplace(real(a),real(a)<eps,0))
	u,v = nparray(real(u)) if u is not None else u,nparray(real(v)) if v is not None else v

	kwargs = dict(
		n_components=rank,
		init='custom' if u is not None and v is not None and kwargs.get('init') is None else kwargs.get('init') if isinstance(kwargs.get('init'),str) else 'nndsvda',
		max_iter=kwargs.get('max_iter',kwargs.get('iteration')) if isinstance(kwargs.get('max_iter',kwargs.get('iteration')),int) else kwargs.get('solver',kwargs.get('update'))[0]['iteration'] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else 100,
		tol=kwargs.get('tol',kwargs.get('eps')) if isinstance(kwargs.get('tol',kwargs.get('eps')),float) else kwargs.get('solver',kwargs.get('update'))[0]['eps'] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else epsilon(a.dtype),
		solver=kwargs.get('solver',kwargs.get('update')) if isinstance(kwargs.get('solver',kwargs.get('update')),str) else kwargs.get('solver',kwargs.get('update'))[0]['update'] if isinstance(kwargs.get('solver',kwargs.get('update')),iterables) else 'cd',
		)

	options = dict(
		W=u,H=v
		)

	func = model(**kwargs)._fit_transform

	constant = add(a)

	u,v,i = func(a,**options)

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

	
	def state(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = haar(
			shape=(D,)*2,
			seed=seed,
			key=key,
			dtype=dtype)[0]
		data = outer(data,conjugate(data))
		return data

	
	def unitary(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = haar(
			shape=(D,)*2,
			seed=seed,
			key=key,
			dtype=dtype)
		return data

	
	def dephase(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2
		parameters = 0 if parameters is None else parameters
		data = tensorprod([array([
			sqrt(absolute(1-parameters))*self.I(D=D,dtype=dtype),
			sqrt(absolute(parameters))*self.Z(D=D,dtype=dtype)
			],dtype=dtype)]*N)
		return data

	
	def bitflip(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2
		parameters = 0 if parameters is None else parameters
		data = tensorprod([array([
			sqrt(absolute(1-parameters))*self.I(D=D,dtype=dtype),
			sqrt(absolute(parameters))*self.X(D=D,dtype=dtype)
			],dtype=dtype)]*N)
		return data

	
	def phaseflip(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2
		parameters = 0 if parameters is None else parameters
		data = tensorprod([array([
			sqrt(absolute(1-parameters))*self.I(D=D,dtype=dtype),
			sqrt(absolute(parameters))*self.Y(D=D,dtype=dtype)
			],dtype=dtype)]*N)
		return data

	
	def depolarize(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2
		parameters = 0 if parameters is None else parameters
		data = tensorprod([array([
				sqrt(absolute(1-(D**2-1)*parameters/(D**2)))*self.I(D=D,dtype=dtype),
				sqrt(absolute(parameters/(D**2)))*self.X(D=D,dtype=dtype),
				sqrt(absolute(parameters/(D**2)))*self.Y(D=D,dtype=dtype),
				sqrt(absolute(parameters/(D**2)))*self.Z(D=D,dtype=dtype)
				],dtype=dtype)]*N)
		return data

	
	def amplitude(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2
		parameters = 0 if parameters is None else parameters
		data = tensorprod([array([
			self.element(D=D,data='00',dtype=dtype) + 
				sqrt(absolute(1-parameters))*self.element(D=D,data='11',dtype=dtype),
			sqrt(absolute(parameters))*self.element(D=D,data='01',dtype=dtype)
			],dtype=dtype)]*N)
		return data

	
	def povm(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):

		if self.basis is None:
			func = self.tetrad
		if callable(self.basis):
			func = self.basis			
		elif isinstance(self.basis,str):
			func = getattr(self,self.basis)
		else:
			func = self.tetrad

		data = func(parameters=parameters,D=D,seed=seed,key=key,dtype=dtype,**kwargs)

		return data

	
	def pauli(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = (1/(D**2-1))*array([
				self.zero(D=D,seed=seed,key=key,dtype=dtype,**kwargs),
				self.plus(D=D,seed=seed,key=key,dtype=dtype,**kwargs),
				self.minusi(D=D,seed=seed,key=key,dtype=dtype,**kwargs),
			   (self.one(D=D,seed=seed,key=key,dtype=dtype,**kwargs)+
				self.minus(D=D,seed=seed,key=key,dtype=dtype,**kwargs)+
				self.plusi(D=D,seed=seed,key=key,dtype=dtype,**kwargs)),
			],dtype=dtype)	
		# print(data)
		# tmp = einsum('uij,vji->uv',data,data)
		# print(1/tmp)
		# exit()				
		return data

	
	def tetrad(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = (1/(D**2))*array([
			sum(i*operator(D=D,seed=seed,key=key,dtype=dtype,**kwargs)
				for i,operator in 
				zip(coefficient,(self.I,self.X,self.Y,self.Z)))
			for coefficient in [
			(1,0,0,1),
			(1,2*sqrt(2)/3,0,-1/3),
			(1,-sqrt(2)/3,sqrt(2/3),-1/3),
			(1,-sqrt(2)/3,-sqrt(2/3),-1/3)
			]
			],dtype=dtype)
		# print(data)
		# tmp = einsum('uij,vji->uv',data,data)
		# print(1/tmp)
		# exit()
		return data

	
	def identity(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = identity(D,dtype=dtype)
		return data

	
	def I(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2	
		data = tensorprod([array([[1,0],[0,1]],dtype=dtype)]*N)
		if parameters is not None:
			data = cos(parameters)*tensorprod([array([[1,0],[0,1]],dtype=dtype)]*N) + -1j*sin(parameters)*data
		return data

	
	def X(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2
		data = tensorprod([array([[0,1],[1,0]],dtype=dtype)]*N)
		if parameters is not None:
			data = cos(parameters)*tensorprod([array([[1,0],[0,1]],dtype=dtype)]*N) + -1j*sin(parameters)*data
		return data

	
	def Y(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2	
		data = tensorprod([array([[0,-1j],[1j,0]],dtype=dtype)]*N)		
		if parameters is not None:
			data = cos(parameters)*tensorprod([array([[1,0],[0,1]],dtype=dtype)]*N) + -1j*sin(parameters)*data		
		return data
		
	
	def Z(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2
		data = tensorprod([array([[1,0],[0,-1]],dtype=dtype)]*N)				
		if parameters is not None:
			data = cos(parameters)*tensorprod([array([[1,0],[0,1]],dtype=dtype)]*N) + -1j*sin(parameters)*data		
		return data

	
	def H(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2
		data = tensorprod([(1/sqrt(2))*array([[1,1],[1,-1]],dtype=dtype)]*N)						
		return data

	
	def S(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2
		data = tensorprod([array([[1,0],[0,1j]],dtype=dtype)]*N)			
		return data

	
	def T(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		N,D = int(log(D)/log(2)),2
		data = tensorprod([array([[1,0],[0,e**(1j*pi/4)]],dtype=dtype)]*N)		
		return data

	
	def CNOT(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=dtype)
		return data

	
	def zero(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = array([1,*[0]*(D-1)],dtype=dtype)
		data = outer(data,conjugate(data))
		return data

	
	def one(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = array([*[0]*(D-1),1],dtype=dtype)
		data = outer(data,conjugate(data))
		return data

	
	def plus(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,1],dtype=dtype)
		data = outer(data,conjugate(data))
		return data
		
	
	def minus(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,-1],dtype=dtype)
		data = outer(data,conjugate(data))
		return data

	
	def plusi(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,1j],dtype=dtype)
		data = outer(data,conjugate(data))
		return data
		
	
	def minusi(self,parameters=None,D=None,seed=None,key=None,dtype=None,**kwargs):
		data = (1/sqrt(D))*array([1,-1j],dtype=dtype)
		data = outer(data,conjugate(data))
		return data

	
	def element(self,D,data=None,seed=None,key=None,dtype=None,**kwargs):
		index = tuple(map(int,data)) if data is not None else None
		data = zeros((D,)*(len(index) if index is not None else 1),dtype=dtype)
		data = inplace(data,index,1) if index is not None else data
		return data


	
	def transform(self,data,D,N=None,where=None,transform=True,**kwargs):
		
		basis = self.povm(D=D,**kwargs)

		inverse = inv(einsum('uij,vji->uv',basis,basis))

		if transform:
			if callable(data):
			
				if N:
					basis = tensorprod([basis]*N)
					inverse = tensorprod([inverse]*N)

				data = einsum('uij,wji,wv->uv',basis,vmap(partial(data,**kwargs))(basis),inverse)		
			
			elif isinstance(data,dict):
				pass
			
			elif isinstance(data,arrays):

				if N:
					basis = tensorprod([basis]*N)
					inverse = tensorprod([inverse]*N)

				if data.ndim == 1:
					raise NotImplementedError(f"Not Implemented {data}")			
			
				elif data.ndim > 1:
			
					data = einsum('uij,ji...->u...',basis,data)

				if data.ndim == 1:
					axes = range(data.ndim+2)
					shape = [1,*data.shape,1]
			
					data = transpose(reshape(data,shape),axes)

		else:
			if callable(data):
				raise NotImplementedError(f"Not Implemented {data}")
			
			elif isinstance(data,dict):

				N = len(data) if N is None else N
				where = range(N) if where is None and N is not None else [*where]

				shape = {i:data[i].shape for i in where}

				data = self.organize(data,where=where,transform=True,conj=False,**kwargs)

				if transform is None:
					shape = [basis.shape[-2+i]**N for i in range(2)]
					axes = range(2*N)
					subscripts = '%s,%s,%s->%s'%(
						''.join((
							symbols(i)
							for i in range(N)
							)),
						','.join((
							''.join((symbols(i),symbols(N+i)))
							for i in range(N)
							)),	
						','.join((
							''.join((symbols(N+i),symbols(2*N+i),symbols(3*N+i)))
							for i in range(N)
							)),
						''.join((
							*(''.join((symbols(2*N+i),))
							for i in range(N)),
							*(''.join((symbols(3*N+i),))
							for i in range(N)),							
							)),						
						)

					data = reshape(transpose(einsum(subscripts,data,*(inverse,)*N,*(basis,)*N),axes),shape)

				else:
					shape = [j for i in where for j in shape[i][1:-1]]
					axes = range(N)
					
					data = ravel(transpose(reshape(data,shape),axes))

			elif isinstance(data,arrays):

				if N:
					basis = tensorprod([basis]*N)
					inverse = tensorprod([inverse]*N)

				if data.ndim == 1:

					data = einsum('uij,uv,v->ij',basis,inverse,data)				
				
				elif data.ndim > 1:

					data = einsum('uij,wji,wv->uv',basis,data,inverse)
		
		return data

	
	def contract(self,state,data,where=None,options=None,**kwargs):

		if isinstance(state,dict):

			where = [*state] if where is None else [*where]
			N = len(where)

			shape = [j for i in where for j in state[i].shape[1:-1]]
			axes = None
			subscripts = '%s,%s->%s%s%s'%(
				''.join((
					''.join(symbols(i) for i in range(N)),
					''.join(symbols(N+i) for i in range(N))
					)),
				','.join(''.join((
					symbols(2*N+i),symbols(N+i),symbols(2*N+i+1))) 
					for i in range(N)
					),
				''.join((symbols(2*N),)),
				''.join(symbols(i) for i in range(N)),
				''.join((symbols(2*N+N),))
				)

			# scheme = {'svd':'stq','nmf':'stq'}.get(options.get('scheme'))
			# state = self.update(state,where=where,options={**kwargs,**options,**dict(scheme=scheme)},**kwargs)

			data = self.shuffle(data,shape=shape,**kwargs)

			data = einsum(subscripts,data,*(state[i] for i in where))

			shape = [state[i].shape for i in where]
			axes = None

			scheme = options.get('scheme')
			data = self.update(data,shape=shape,axes=axes,where=where,options={**dict(scheme=scheme,state=state),**options},**kwargs)

			for i in where:
				state[i] = data[i]

			# scheme = {'svd':'stq','nmf':'stq'}.get(options.get('scheme'))
			# state = self.update(state,where=where,options={**kwargs,**options,**dict(scheme=scheme)},**kwargs)

			data = state

		elif state.ndim == 1:
			if data.ndim == 2:
				data = einsum('ij,j->i',data,state)
			else:	
				raise NotImplementedError(f"Not Implemented {data}")						
		elif state.ndim > 1:
			if data.ndim == 2:
				data = einsum('ij,jk...,lk->il...',data,state,conjugate(data))
			elif data.ndim == 3:
				data = einsum('uij,jk...,ulk->il...',data,state,conjugate(data))
			else:
				raise NotImplementedError(f"Not Implemented {data}")			
		else:
			raise NotImplementedError(f"Not Implemented {state}")			
		
		return data

	
	def organize(self,data,shape=None,axes=None,conj=None,where=None,transform=True,**kwargs):
		
		if transform:
			
			if isinstance(data,dict):
				where = [*data] if where is None else [*where]
				N = len(where)

				shape = [k for i,j in enumerate(where) for k in ([prod(data[j].shape[:2]),*data[j].shape[2:-1]] if i in [0] else [*data[j].shape[1:-2],prod(data[j].shape[-2:])] if i in [N-1] else [*data[j].shape[1:-1]])] if shape is None else shape
				axes = range(N) if axes is None else axes
				subscripts = '%s->%s'%(
					','.join(
						''.join((symbols(N+i),symbols(i),symbols(N+i+1)))
						for i in range(N)
						),
					''.join(
						''.join((symbols(N+i),symbols(i),) if i in [0] else (symbols(i),symbols(N+i+1)) if i in [N-1] else (symbols(i),))
						for i in range(N)
						),
					)

				data = transpose(reshape(einsum(subscripts,*(data[i] for i in where)),shape),axes)
			
			elif isinstance(data,tuple):
				raise NotImplementedError(f"Not Implemented {data}")
			
			elif isinstance(data,arrays):
				if data.ndim == 1:
					if not conj:
						shape = [1,*data.shape,1] if shape is None else shape
						axes = [0] if axes is None else axes
					else:
						shape = [1,*data.shape,1] if shape is None else shape
						axes = [0] if axes is None else axes
				elif data.ndim == 2:
					if not conj:
						shape = [*data.shape] if shape is None else shape
						axes = range(data.ndim) if axes is None else axes
					else:
						shape = [*data.shape] if shape is None else shape
						axes = range(data.ndim) if axes is None else axes					
				elif data.ndim == 3:
					if not conj:
						shape = [data.shape[0]*prod(data.shape[1:-1]),data.shape[-1]] if shape is None else shape
						axes = range(data.ndim) if axes is None else axes
					else:
						shape = [data.shape[0],prod(data.shape[1:-1])*data.shape[-1]] if shape is None else shape
						axes = range(data.ndim) if axes is None else axes
				elif data.ndim == 4:
					if not conj:
						shape = [prod(data.shape[:2]),prod(data.shape[-2:])] if shape is None else shape
						axes = range(data.ndim) if axes is None else axes
					else:
						shape = [prod(data.shape[:2]),prod(data.shape[-2:])] if shape is None else shape
						axes = range(data.ndim) if axes is None else axes
				else:
					raise NotImplementedError(f"Not Implemented {data}")
				
				data = reshape(transpose(data,axes),shape)

		else:

			if isinstance(data,dict):
				raise NotImplementedError(f"Not Implemented {data}")
		
			elif isinstance(data,tuple):
				where = range(len(data)) if where is None else [*where]
				shape = [[*data[0].shape[:2],1],[1,*data[1].shape[-2:]]] if shape is None else shape
				axes = [range(data[0].ndim+1),range(data[1].ndim+1)] if axes is None else axes
				data = dict(zip(where,(transpose(reshape(data,shape),axes) for data,shape,axes in zip(data,shape,axes))))
			elif isinstance(data,arrays):
				if data.ndim == 1:
					if not conj:
						shape = [1,*data.shape,1] if shape is None else shape
						axes = range(data.ndim+2) if axes is None else axes
					else:
						shape = [1,*data.shape,1] if shape is None else shape
						axes = range(data.ndim+2) if axes is None else axes
				elif data.ndim == 2:
					if not conj:
						shape = [data.shape[0],prod(data.shape[1:-1]),-1] if shape is None else shape
						axes = range(data.ndim+1) if axes is None else axes
					else:
						shape = [-1,prod(data.shape[1:-1]),data.shape[-1]] if shape is None else shape
						axes = range(data.ndim+1) if axes is None else axes
				elif data.ndim == 3:
					if not conj:
						shape = [*data.shape] if shape is None else shape
						axes = range(data.ndim) if axes is None else axes
					else:
						shape = [*data.shape] if shape is None else shape
						axes = range(data.ndim) if axes is None else axes
				elif data.ndim == 4:
					if not conj:
						shape = [prod(data.shape[:2]),prod(data.shape[-2:])] if shape is None else shape
						axes = range(data.ndim-2) if axes is None else axes
					else:
						shape = [prod(data.shape[:2]),prod(data.shape[-2:])] if shape is None else shape
						axes = range(data.ndim-2) if axes is None else axes
				else:
					raise NotImplementedError(f"Not Implemented {data}")

				data = transpose(reshape(data,shape),axes)
		
		return data

	
	def shuffle(self,data,shape,where=None,transform=True,**kwargs):
		if transform:
			n,d = len(shape),data.ndim
			where = [range(n)] if where is None else [where,sorted(set(range(n))-set(where))]
			shape = [*shape]*d
			axes = [j*n+i for indices in where for j in range(d) for i in indices]
			data = transpose(reshape(data,shape),axes)
		else:
			n,d = len(shape),data.ndim//len(shape)
			where = [j*n+i for indices in ([range(n)] if where is None else [where,sorted(set(range(n))-set(where))]) for j in range(d) for i in indices]		
			shape = [prod(shape)]*d
			axes = [where.index(j*n+i) for j in range(d) for i in range(n)]
			data = reshape(transpose(data,axes),shape)
		return data

	
	def spectrum(self,state,where=None,options=None,**kwargs):

		N = len(state)
		where = where if isinstance(where,integers) else min(N-2,max(1,(min(where)-1) if min(where) > 0 else (max(where)+1))) if where is not None else N//2
		where = min(N-2,max(0,where-1))
		where = [where,where+1]

		options = dict() if options is None else options

		defaults = dict(scheme='stq')
		state = self.update(state,where=where,options={**kwargs,**options,**defaults},**kwargs)

		state = self.organize(state,where=where,**kwargs)

		defaults = dict(scheme=options.get('scheme','spectrum'))
		state = self.scheme(options={**kwargs,**options,**defaults},**kwargs)(state,**{**kwargs,**options,**defaults})

		return state

	
	def scheme(self,options=None,**kwargs):
		
		options = dict() if options is None else options

		scheme = options.get('scheme')
		eps = kwargs.get('eps') if kwargs.get('eps') is not None else epsilon()

		def wrapper(func):
			def decorator(a,rank=None,conj=None,**kwargs):
				a = dagger(a) if conj else a
				rank = min(a.shape) if rank is None else rank    
				u,v,s = func(a,rank=rank,conj=conj,**kwargs) 
				u,v,s = u[:,:rank],v[:rank,:],s[:rank]
				# s = add(u,0)
				# i = absolute(s)>eps				
				# u,v,s = inplace(u,(slice(None),i),u[:,i]/s[i]),inplace(v,i,(v[i].T*s[i]).T),s
				# u,v,s = dotr(u,reciprocal(s)),dotl(v,s),s
				u,v,s = (dagger(v),dagger(u),dagger(s)) if conj else (u,v,s)
				u,v,s = cmplx(u),cmplx(v),min(*u.shape,*v.shape)
				return u,v,s
			return decorator

		if scheme is None:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict(compute_uv=True,full_matrices=False,hermitian=False)
				u,s,v = svds(real(a),**{**kwargs,**options,**defaults,**dict(rank=rank)})
				u,v,s = u[:,:rank],v[:rank,:],s[:rank]
				u,v,s = dotr(u,sign(s)*sqrt(absolute(s))),dotl(v,sign(s)*sqrt(absolute(s))),ones(s.shape,dtype=s.dtype)
				return u,v,s				
		elif scheme in ['svd']:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict(compute_uv=True,full_matrices=False,hermitian=False)
				u,s,v = svds(real(a),**{**kwargs,**options,**defaults,**dict(rank=rank)})
				u,v,s = u[:,:rank],v[:rank,:],s[:rank]
				u,v = u,dotl(v,s)
				return u,v,s
		elif scheme in ['nmf']:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict()		
				u,v,s = nmf(real(a),**{**kwargs,**options,**defaults,**dict(rank=rank)})
				u,v,s = u[:,:rank],v[:rank,:],s[:rank]
				u,v = dotr(u,sign(s)*sqrt(absolute(s))),dotl(v,sign(s)*sqrt(absolute(s)))
				return u,v,s
		elif scheme in ['_nmf']:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict()
				u,v,s = _nmf(real(a),**{**kwargs,**options,**defaults,**dict(rank=rank)})
				u,s,v = u[:,:rank],s[:rank],v[:rank,:]
				u,v = dotr(u,sign(s)*sqrt(absolute(s))),dotl(v,sign(s)*sqrt(absolute(s)))
				return u,v,s

		def wrapper(func):
			def decorator(a,rank=None,conj=None,**kwargs):
				a = dagger(a) if conj else a
				rank = min(a.shape) if rank is None else rank    
				u,v,s = func(a,rank=rank,conj=conj,**kwargs) 
				u,v,s = u[:,:rank],(v[:rank,:] if v is not None else None),(s[:rank] if s is not None else None)
				# s = add(u,0)
				# i = absolute(s)>eps
				# u,v,s = inplace(u,(slice(None),i),u[:,i]/s[i]),(inplace(v,i,(v[i].T*s[i]).T) if v is not None else None),(s if s is not None else None)
				# # u,v,s = dotr(u,reciprocal(s)),(dotl(v,s) if v is not None else None),(s if s is not None else None)
				u,v,s = ((dagger(v) if v is not None else None),dagger(u),(dagger(s) if s is not None else None)) if conj else (u,v,s)
				u,v,s = (cmplx(u) if u is not None else None),(cmplx(v) if v is not None else None),min(*(u.shape if u is not None else ()),*(v.shape if v is not None else ()))
				return u,v,s
			return decorator

		if scheme in ['qr']:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict(mode='reduced')
				u,v = qrs(real(a),**{**kwargs,**options,**defaults,**dict(rank=rank)})
				s = ones(min(a.shape),dtype=a.dtype)
				return u,v,s				
		elif scheme in ['stq']:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict()		
				u,v,s = a,None,None
				return u,v,s				
		
		def wrapper(func):
			def decorator(a,rank=None,conj=None,**kwargs):
				a = dagger(a) if conj else a
				rank = min(a.shape) if rank is None else rank    
				s = func(a,rank=rank,conj=conj,**kwargs) 
				s = s[:rank]
				return s
			return decorator

		if scheme in ['eig']:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict(compute_v=False,hermitian=False)	
				rank = min(a.shape) if rank is None else rank    
				s = eig(real(a),**{**kwargs,**options,**defaults,**dict(rank=rank)})
				return s
		elif scheme in ['spectrum']:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict(compute_uv=False,full_matrices=False,hermitian=False)							
				s = svd(real(a),**{**kwargs,**options,**defaults,**dict(rank=rank)})
				s = s[:rank]
				return s
		elif scheme in ['probability']:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict()				
				u,v,s = nmf(real(a),**{**kwargs,**options,**defaults,**dict(rank=rank)})
				s = s[:rank]
				return s
		elif scheme in ['_spectrum']:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict(compute_uv=False,full_matrices=False,hermitian=False)	
				rank = min(a.shape) if rank is None else rank    
				s = svd(real(a),**{**kwargs,**options,**defaults,**dict(rank=rank)})
				s = s[:rank]
				return s				
		elif scheme in ['_probability']:
			@wrapper
			def scheme(a,rank=None,conj=None,**options):
				defaults = dict()						
				rank = min(a.shape) if rank is None else rank    
				u,v,s = _nmf(real(a),**{**kwargs,**options,**defaults,**dict(rank=rank)})
				s = s[:rank]
				return s								
		
		return scheme		

	
	def update(self,state,shape=None,axes=None,where=None,options=None,**kwargs):
		
		options = dict() if options is None else options

		defaults = dict()

		if isinstance(state,dict):

			N = len(state)
			where = [N,N] if where is None else [where,where] if isinstance(where,integers) else [*where]

			indices = (*range(0,min(where)+1,1),*range(N-1,max(where)-1,-1))

			for i in indices:

				if i < min(where):

					shape = state[i].shape
					axes = axes

					state[i] = self.organize(state[i],where=where,transform=True,conj=False,**kwargs)

					u,v,s = self.scheme(options={**kwargs,**options,**defaults},**kwargs)(state[i],conj=False,**{**kwargs,**options,**defaults})

					state[i] = self.organize(u,where=where,shape=[*shape[:-1],s],axes=axes,transform=False,conj=False,**kwargs)

					if i < (N-1) and v is not None:
						state[i+1] = dot(v,state[i+1])

				elif i > max(where):

					shape = state[i].shape
					axes = axes


					tmp = copy(state[i])
					state[i] = self.organize(state[i],where=where,transform=True,conj=True,**kwargs)
					
					u,v,s = self.scheme(options={**kwargs,**options,**defaults},**kwargs)(state[i],conj=True,**{**kwargs,**options,**defaults})

					state[i] = self.organize(v,where=where,shape=[s,*shape[1:]],axes=axes,transform=True,conj=True,**kwargs)
					
					if i > 0 and u is not None:
						state[i-1] = dot(state[i-1],u)


		elif isinstance(state,arrays):

			if len(where) == 2:

				state = self.organize(state,where=where,shape=[prod(state.shape[:len(state.shape)//2]),prod(state.shape[len(state.shape)//2:])],axes=None if axes is None else axes,transform=True,conj=False,**kwargs)

				u,v,s = self.scheme(options={**kwargs,**options,**defaults},**kwargs)(state,conj=False,**{**kwargs,**options,**defaults})

				error = (norm(state-dot(u,v))/norm(state)).real

				state = self.organize((u,v),where=where,shape=[[1,*u.shape[:-1],s],[s,*v.shape[1:],1]] if shape is None else [[*shape[0][:-1],s],[s,*shape[1][1:]]],axes=None if axes is None else axes,transform=False,conj=False,**kwargs)


				if 0:

					tmp = {**kwargs.get('state',options.get('state')),**state}
					variables = kwargs.get('variables')
					D,N,rank = kwargs.get('D',options.get('D')),None,kwargs.get('rank',options.get('rank'))
					basis = Basis()

					options = dict(D=D,N=N)
					constant = real(1-add(basis.transform(tmp,transform=False,**{**kwargs,**options})))+0.

					options = dict(D=D,N=N)
					spectrum = basis.transform(tmp,transform=None,**{**kwargs,**options})

					hermitian = real(norm(spectrum-dagger(spectrum)))+0.

					options = dict(compute_v=False,hermitian=True)
					spectrum = eig(spectrum,**options)
					ratio = real(-add(spectrum[spectrum<=0])/add(spectrum[spectrum>0]))+0.
					
					sums = {i:(
								asscalar(minimum(tmp[i].sum((0,1) if i < min(where) else (-2,-1) if i > max(where) else None) if i not in where else add(dot(tmp[min(where)],tmp[max(where)])))),
								asscalar(maximum(tmp[i].sum((0,1) if i < min(where) else (-2,-1) if i > max(where) else None) if i not in where else add(dot(tmp[min(where)],tmp[max(where)])))))
							for i in tmp}

					print('---',where,minimum(spectrum),max(spectrum),'---',1-spectrum.sum(),constant,hermitian,ratio)
					print(sums)
					print()
				# options = dict(compute_uv=False,full_matrices=False,hermitian=True)
				# variables['u.condition'].append(condition_number(dot(u.T,u).real))
				# variables['v.condition'].append(condition_number(dot(v,v.T).real))
				# variables['u.spectrum'].append(tuple(svd(dot(u.T,u).real,**options)))
				# variables['v.spectrum'].append(tuple(svd(dot(v,v.T).real,**options)))
				# variables['uv.error'].append(sqrt(norm(state-dot(u,v))/norm(state)).real)
				# variables['uv.spectrum'].append(spectrum)
				# variables['uv.rank'].append(rank)

				# parse = lambda obj: asscalar(obj.real)
				# print(where,error,parse(variables['uv.error'][-1]),{'u':[parse(variables['u.condition'][-1]),parse(u.min()),parse(u.max()),u.shape],'v':[parse(variables['v.condition'][-1]),parse(v.min()),parse(v.max()),v.shape]})


		return state


def test_shuffle(*args,**kwargs):

	basis = Basis()
	init = lambda D,N,L,d,k: (reshape(arange(D**(d*N)),(D**N,)*d),reshape(arange(D**(k*L)),(D**L,)*k))


	# Data
	D = 2
	N = 3
	L = N
	d = 1
	k = 2
	shape = [D]
	where = range(L)

	state,data = init(D,N,L,d,k)

	assert allclose(data,basis.shuffle(basis.shuffle(data,shape*L,where=where,transform=True),shape*L,where=where,transform=False))

	# Global
	D = 2
	N = 3
	L = N
	d = 1
	k = 2
	shape = [D]
	where = range(L)

	state,data = init(D,N,L,d,k)

	subscripts = 'ij,j->i'
	out = einsum(subscripts,data,state)

	

	state,data = init(D,N,L,d,k)

	state = basis.shuffle(state,shape*N,where=where,transform=True)

	data = basis.shuffle(data,shape*L,where=range(L),transform=True)

	subscripts = '%s%s,%s->%s'%(
		''.join(symbols(i) for i in range(L,2*L)),
		''.join(symbols(i) for i in range(L)),
		''.join(symbols(i) for i in range(N)),
		''.join(symbols(i) for i in range(L,2*L)),
		)
	_out = basis.shuffle(einsum(subscripts,data,state),shape*N,where=where,transform=False)

	assert allclose(out,_out)


	# Local
	D = 2
	N = 7
	L = 7
	d = 2
	k = 2
	shape = [D]
	where = [i for i in [1,3,5,7,9,0,2,4,6,8] if i < N][:L]


	state,data = init(D,N,L,d,k)

	state,data = state,reshape(
		transpose(
			reshape(
				tensorprod([*(identity(D),)*(N-N),data,*(identity(D),)*(N-L)]),
				[*shape*N]*d
				),
			[N*j+((len(where)+list(sorted(set(range(N))-set(where))).index(i)) 
			if i not in where else where.index(i)) 
			for j in range(d) for i in range(N)]
			),
		(D**N,D**N))
	subscripts = 'ij,jk,lk->il'
	out = einsum(subscripts,data,state,conjugate(data))


	state,data = init(D,N,L,d,k)

	state = basis.shuffle(state,shape*N,where=where,transform=True)

	data = basis.shuffle(data,shape*L,where=range(L),transform=True)

	subscripts = '%s%s,%s%s%s%s,%s%s->%s%s%s%s'%(
		''.join(symbols(i) for i in range(2*N,2*N+L)),
		''.join(symbols(i) for i in range(L)),

		''.join(symbols(i) for i in range(L)),
		''.join(symbols(i) for i in range(N,N+L)),
		''.join(symbols(i) for i in range(L,N)),		
		''.join(symbols(i) for i in range(N+L,2*N)),				
		
		''.join(symbols(i) for i in range(2*N+L,2*N+2*L)),
		''.join(symbols(i) for i in range(N,N+L)),

		''.join(symbols(i) for i in range(2*N,2*N+L)),
		''.join(symbols(i) for i in range(2*N+L,2*N+2*L)),
		''.join(symbols(i) for i in range(L,N)),		
		''.join(symbols(i) for i in range(N+L,2*N)),				
		)	
	_out = basis.shuffle(einsum(subscripts,data,state,conjugate(data)),shape*N,where=where,transform=False)

	assert allclose(out,_out)


	print('passed')

	return


def test_mps(*args,**kwargs):

	def initialize(state=None,data=None,D=None,N=None,**kwargs):

		basis = Basis()

		def func(state,data=None,where=None,**kwargs):
			where = [where] if not isinstance(where,iterables) else [*where]
			state = getattr(basis,state)(**{**kwargs,**dict(D=D)})
			state = basis.transform(state,where=where,**{**kwargs,**dict(D=D,N=len(where))})
			return state

		state = {i:'state' for i in range(N)} if state is None else {i:state for i in range(N)} if isinstance(state,str) else state
		state = {i: func(state=state[i],where=i,**kwargs) for i in state}

		def func(state,data=None,where=None,**kwargs):
			
			def func(state=None,data=data,where=where,**kwargs):
				where = [where] if not isinstance(where,iterables) else [*where]
				if isinstance(data,str):
					data = getattr(basis,data)(**{**kwargs,**dict(D=D**len(where),parameters=kwargs.get('parameters').get(data) if isinstance(kwargs.get('parameters'),dict) else kwargs.get('parameters'))})
				else:
					 data = tensorprod([getattr(basis,data[index])(**{**kwargs,**dict(D=D,parameters=kwargs.get('parameters').get(data[index]) if isinstance(kwargs.get('parameters'),dict) else kwargs.get('parameters'))}) for index,i in enumerate(where)])
				data = basis.contract(state,data,where=where,**kwargs)
				return data
			where = [where] if not isinstance(where,iterables) else [*where]
			data = basis.transform(func,where=where,**{**kwargs,**dict(D=D,N=len(where))})

			# #####
			# state = basis.update(state,where=where,options={**kwargs,**kwargs.get('options',{}),**dict(scheme={'svd':'stq','nmf':'stq'}.get(kwargs.get('options',{}).get('scheme')))},**{kwarg:kwargs[kwarg] for kwarg in kwargs if kwarg not in ['options']})

			# N,d = len(where),2
			# shapes = [[D**2 for i in range(N) for j in range(d)],[D**(2*d) for i in range(N)]]
			# axis = [[j*N+i for i in range(N) for j in range(d)],[i for i in range(N)]]
			# for shape,axes in zip(shapes,axis):
			# 	data = transpose(reshape(data,shape),axes)
			# U,S,V = svd(real(data))
			# A,B = [state[i] for i in where]

			# K,C = S.size,add(dot(reshape(A,[A.shape[0]*A.shape[1],A.shape[2]]),reshape(B,[B.shape[0],B.shape[1]*B.shape[2]])))

			# U,V,S = reshape(U,[*[D**2]*2,K]),reshape(dotr(V,S),[K,*[D**2]*2]),S
			# A,B = A,B/C

			# A,B = einsum('iuj,uvk->ivjk',A,U),einsum('iuj,kuv->ikvj',B,V)

			# A,B = reshape(A,[A.shape[0]*A.shape[1],A.shape[2]*A.shape[3]]),reshape(B,[B.shape[0]*B.shape[1],B.shape[2]*B.shape[3]])
			# C = add(A,0)
			# A,B = dotr(A,1/C),dotl(B,C)

			# print(add(A,0))
			# print(add(dot(A,B)))
			# print(A.shape)
			# print(B.shape)
			# exit()
			# #####

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
			print(k,{i:state[i].shape for i in state})

		return state


	def _initialize(state=None,data=None,D=None,N=None,**kwargs):

		basis = Basis()

		def func(state,data=None,where=None,**kwargs):
			state = getattr(basis,state)(**{**kwargs,**dict(D=D)})
			return state

		state = {i:'state' for i in range(N)} if state is None else {i:state for i in range(N)} if isinstance(state,str) else state
		state = tensorprod([func(state=state[i],where=i,**kwargs) for i in state])

		def func(state,data,where=None,**kwargs):
			
			def func(state=None,data=data,where=where,**kwargs):
				if isinstance(data,str):
					data = getattr(basis,data)(**{**kwargs,**dict(D=D**len(where),parameters=kwargs.get('parameters').get(data) if isinstance(kwargs.get('parameters'),dict) else kwargs.get('parameters'))})
				else:
					 data = tensorprod([getattr(basis,data[index])(**{**kwargs,**dict(D=D,parameters=kwargs.get('parameters').get(data[index]) if isinstance(kwargs.get('parameters'),dict) else kwargs.get('parameters'))}) for index,i in enumerate(where)])
				# data = tensorprod([*[basis.identity(**{**kwargs,**dict(D=D)}).reshape(*(1,)*(data.ndim-2),*(D,)*2)]*min(where),data,*[basis.identity(**{**kwargs,**dict(D=D)}).reshape(*(1,)*(data.ndim-2),*(D,)*2)]*(N-max(where)-1)])
				return data
			shape,ndim = [D]*N,state.ndim
			state = reshape(basis.shuffle(state,shape,where=where,transform=True),(*(D**len(where),)*(ndim),-1))
			data = func(**kwargs)
			data = basis.contract(state,data=data,where=where,**kwargs)
			data = basis.shuffle(reshape(data,(*(D,)*(N*ndim),)),shape,where=where,transform=False)
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

	@timer
	def function(state,data,_state,_data,M=None,kwargs={},_kwargs={}):

		iterations = range(1 if M is None else M)

		basis = Basis()

		for k in iterations:
			for i,_i in zip(data,_data):

				print(i)
				state = data[i](state,**kwargs)
				_state = _data[_i](_state,**_kwargs)

				# _state_ = basis.transform(state,transform=None,**kwargs)
				
				# error = norm(_state_-_state)/norm(_state)
				# normalization = real(1-trace(_state_))
				# purity = real(1-norm(_state_))

				# print(k,i,error,normalization,purity)

				key,kwargs['key'] = rng.split(kwargs['key'])
				_key,_kwargs['key'] = rng.split(_kwargs['key'])
			print(k,{i:state[i].shape for i in state})

		return state,_state

	parse = lambda data,p=8: data.real.round(p)	
	normalization = lambda data,p=1: (data**p).sum().real
	boolean = lambda path: not os.path.exists(path) or 1

	N = 8
	D = 2
	M = N+N//2
	L = N//2
	K = D**(N-2)
	parameters = pi/4
	noise = 1
	rank = D**(N//1)
	eps = 1e-14
	seed = 103400709
	basis = Basis()
	dtype = 'complex128'
	path = 'scratch/nmf/data/data.hdf5'
	file = 'scratch/nmf/data/variables.hdf5'

	# povm = Basis(basis='tetrad').povm(D=D,dtype=dtype)
	# spectrum,vectors = eig(povm,compute_v=True,hermitian=True)
	# print(povm)
	# print(spectrum)
	# print(vectors)
	# print(einsum('uij,ukj',spectrum[:,None,:]*vectors,conjugate(vectors)))
	# for i in vectors:
	# 	print(dot(dagger(i[:,0]),i[:,1]).round(14)+0.)
	# exit()


	state = {i:'zero' 
		for i in range(N)}
	data = {index:(data,where) 
		for index,(data,where) in enumerate((data,where) 
		# for i in [*range(0,N-1,2),*range(1,N-1,2)] for where in [(i,i+1)] 
		for i in [*range(0,N-1)] for where in [(i,i+1)] 
		# for data in ['unitary','depolarize'])}
		for data in ['unitary','depolarize'])}
		# for data in ['unitary'])}
		# for data in ['depolarize'])}
		# for data in ['unitary' if not (i%4 == 2) else 'unitary','depolarize'])}
		# for data in ['CNOT','T','depolarize'])}

	kwargs = dict(
		D=D,N=N,M=M,
		parameters={'unitary':parameters,'identity':parameters,'X':parameters,'depolarize':noise},
		variables={attr:[] for attr in ['u.condition','v.condition','u.spectrum','v.spectrum','uv.error','uv.spectrum','uv.rank']},
		options=dict(
			scheme='svd',
			rank=rank,
			eps=eps,			
		),
		key=seeder(seed),		
		seed=seed,
		dtype=dtype,		
	)

	# kwargs = dict(
	# 	D=D,N=N,M=M,
	# 	parameters={'unitary':parameters,'identity':parameters,'X':parameters,'depolarize':noise},
	# 	variables={attr:[] for attr in ['u.condition','v.condition','u.spectrum','v.spectrum','uv.error']},
	# 	options=dict(
	# 		scheme='nmf',
	# 		init='nndsvd',
	# 		iteration=int(100),
	# 		eps=eps,
	# 		alpha=1e-2,
	# 		beta=5e-1,
	# 		gamma=1e-10,
	# 		delta=1,
	# 		iota=6e-1,
	# 		sigma=1e-4,
	# 		update=[
	# 			# {'update':'gd','iteration':int(1e6),'eps':1e-14},
	# 			# {'update':'cg','iteration':int(1e5),'eps':1e-14},
	# 			# {'update':'cg','iteration':int(100),'eps':1e-14},
	# 			# {'update':'cp','iteration':int(1e1),'eps':1e-10},
	# 			# {'update':'pc','iteration':int(1e3),'eps':1e-10},
	# 			# {'update':'rg','iteration':int(1e7),'eps':1e-10},
	# 			{'update':'gd','iteration':int(1e7),'eps':1e-10},
	# 			# {'update':'sd','iteration':int(1e6),'eps':1e-14},
	# 			# {'update':'qp','iteration':int(1),'eps':1e-8},
	# 			# {'update':'pd','iteration':int(1),'eps':1e-14},
	# 			# {'update':'gd','iteration':int(1e6),'eps':1e-14},					
	# 			# {'update':'ls','iteration':int(1e5),'eps':1e-14},					
	# 			# {'update':'mhu','iteration':int(1),'eps':1e-14},
	# 			# {'update':'gd','iteration':int(1e4),'eps':1e-14},					
	# 			# {'update':'cd','iteration':int(1e6),'eps':1e-14},
	# 			# {'update':'mu','iteration':int(1e3),'eps':1e-14},
	# 			# {'update':'mru','iteration':int(1e4),'eps':1e-14},
	# 			# {'update':'mhu','iteration':int(1e4),'eps':1e-14},
	# 			# {'update':'miu','iteration':int(1e4),'eps':1e-14},
	# 			# {'update':'cd','iteration':int(1e6),'eps':1e-14},
	# 			],
	# 	),
	# 	key=seeder(seed),
	# 	seed=seed,
	# 	dtype=dtype,		
	# )


	_state = {i:'zero' 
		for i in range(N)}
	_data = {index:(data,where) 
		for index,(data,where) in enumerate((data,where) 
		# for i in [*range(0,N-1,2),*range(1,N-1,2)] for where in [(i,i+1)] 
		for i in [*range(0,N-1)] for where in [(i,i+1)]
		# for data in ['unitary','depolarize'])}
		for data in ['unitary','depolarize'])}
		# for data in ['unitary'])}
		# for data in ['depolarize'])}
		# for data in ['unitary' if not (i%4 == 2) else 'unitary','depolarize'])}
		# for data in ['CNOT','T','depolarize'])}
	_kwargs = dict(
		D=D,N=N,M=M,
		parameters={'unitary':parameters,'identity':parameters,'X':parameters,'depolarize':noise},
		variables={attr:[] for attr in ['u.condition','v.condition','u.spectrum','v.spectrum','uv.error','uv.spectrum','uv.rank']},
		options=dict(
			scheme='svd',
			rank=rank,
			eps=eps,			
		),
		key=seeder(seed),		
		seed=seed,
		dtype=dtype,		
	)	


	state,data = initialize(state=state,data=data,**kwargs)
	_state,_data = _initialize(state=_state,data=_data,**_kwargs)

	state,_state = function(state,data,_state,_data,M=M,kwargs=kwargs,_kwargs=_kwargs)

	# state = func(state,data,**kwargs)
	# _state = _func(_state,_data,**_kwargs)


	state = basis.transform(state,transform=None,**kwargs)
	_state = _state


	error = norm(state-_state)/norm(_state)	

	print(error)
	print((state-_state).ravel())
	assert allclose(state,_state)

	print('passed')

	exit()




	# state = {i:'state' for i in range(N)}
	# data = {index:(data,where) for index,(data,where) in enumerate((data,where) for i in [*range(0,N-1)] for where in [(i,i+1)] for data in ['unitary'])}

	# kwargs = dict(
	# 	D=D,N=N,M=M,
	# 	parameters=parameters,
	# 	options=dict(),
		# key=seeder(seed),		
		# seed=seed,
		# dtype=dtype,			
	# )

	# basis = Basis()

	# state,data = initialize(state=state,data=data,**kwargs)

	# state = func(state,data,**kwargs)


	# _state = {i:'state' for i in range(N)}
	# _data = {index:(data,where) for index,(data,where) in enumerate((data,where) for i in [*range(0,N-1)] for where in [(i,i+1)] for data in ['unitary'])}

	# _kwargs = dict(
	# 	D=D,N=N,M=M,
	# 	options=dict(),
	# 	seed=seed,
	# 	dtype=dtype,		
	# )	

	# _state,_data = _initialize(state=_state,data=_data,**_kwargs)

	# _state = _func(_state,_data,**_kwargs)


	# state = basis.transform(state,transform=None,**kwargs)
	# _state = _state

	# assert allclose(state,_state)

	# if boolean(file):

	# 	state = {i:'state' 
	# 		for i in range(N)}
	# 	data = {index:(data,where) 
	# 		for index,(data,where) in enumerate((data,where) 
	# 		for i in [*range(0,N-1)] for where in [(i,i+1)] 
	# 		for data in ['X'])}


	# 	kwargs = dict(
	# 		D=D,N=N,M=M,
	# 		parameters={'unitary':parameters,'identity':parameters,'X':parameters,'depolarize':noise},
	# 		variables={attr:[] for attr in ['u.condition','v.condition','u.spectrum','v.spectrum','uv.error']},
	# 		options=dict(
	# 			scheme='nmf',
	# 			init='nndsvd',
	# 			iteration=int(1e3),
	# 			eps=1e-10,
	# 			alpha=1,
	# 			beta=5e-1,
	# 			gamma=1e-10,
	# 			delta=1,
	# 			iota=6e-1,
	# 			sigma=1e-4,
	# 			update=[
	# 				# {'update':'gd','iteration':int(1e6),'eps':1e-14},
	# 				# {'update':'cg','iteration':int(1e5),'eps':1e-14},
	# 				# {'update':'cg','iteration':int(100),'eps':1e-14},
	# 				# {'update':'cp','iteration':int(1e1),'eps':1e-10},
	# 				{'update':'pc','iteration':int(1e3),'eps':1e-10},
	# 				# {'update':'sd','iteration':int(1e6),'eps':1e-14},
	# 				# {'update':'qp','iteration':int(1),'eps':1e-8},
	# 				# {'update':'pd','iteration':int(1),'eps':1e-14},
	# 				# {'update':'gd','iteration':int(1e6),'eps':1e-14},					
	# 				# {'update':'ls','iteration':int(1e5),'eps':1e-14},					
	# 				# {'update':'mhu','iteration':int(1),'eps':1e-14},
	# 				# {'update':'gd','iteration':int(1e4),'eps':1e-14},					
	# 				# {'update':'cd','iteration':int(1e6),'eps':1e-14},
	# 				# {'update':'mu','iteration':int(1e3),'eps':1e-14},
	# 				# {'update':'mru','iteration':int(1e4),'eps':1e-14},
	# 				# {'update':'mhu','iteration':int(1e4),'eps':1e-14},
	# 				# {'update':'miu','iteration':int(1e4),'eps':1e-14},
	# 				# {'update':'cd','iteration':int(1e6),'eps':1e-14},
	# 				],
	# 		),
	# 		key=seeder(seed),
	# 		seed=seed,
	# 		dtype=dtype,		
	# 	)

	# 	basis = Basis()

	# 	state,data = initialize(state=state,data=data,**kwargs)

	# 	_state = copy(state)

	# 	state = func(state,data,**kwargs)

		
	# 	data = kwargs.get('variables',{})

	# 	dump(data,file)
	
	# else:
	
	# 	data = load(file)

	# 	for variable in data:
	# 		print(variable)
	# 		print(data[variable])
	# 		print()

	# 	# Data

	# 	fig,ax = None,None
	# 	settings = [
	# 		{
	# 			"fig": {
	# 				"set_size_inches": {
	# 					"w": 24,
	# 					"h": 16
	# 				},
	# 				"subplots_adjust": {},
	# 				"tight_layout": {},
	# 				"savefig": {
	# 					"fname": 'scratch/nmf/data/variables.pdf',
	# 					"bbox_inches": "tight",
	# 					"pad_inches": 0.2
	# 				}
	# 			},
	# 			"ax":{
	# 				"errorbar":{
	# 					"x":[*arange(len(data[y]))] if x is None else x,
	# 					'y':[*data[y]],
	# 					"label":{'u.condition':'$A=U^TU$','v.condition':'$A=VV^T$','uv.error':"$M \\approx UV$"}.get(y),
	# 					"alpha":0.8,
	# 					"marker":{'u.condition':'o','v.condition':'o','uv.error':'^'}.get(y),
	# 					"markersize":12,
	# 					"linestyle":{'u.condition':'--','v.condition':'--','uv.error':'-'}.get(y),
	# 					"linewidth": 5,
	# 					"elinewidth": 2,
	# 					"capsize": 3,
	# 					"color":{'u.condition':'black','v.condition':'gray','uv.error':'mediumslateblue'}.get(y),
	# 					"obj":{'u.condition':None,'v.condition':None,'uv.error':'twinx'}.get(y),
	# 					},
	# 				"set_title": {
	# 					'uv.error':
	# 							{
	# 							"label": f"$\\textrm{{Nearest-Neighbour 1D XX Gates with}} ~ \\theta = \\pi/10^{{%s}} ~,~ N = {N} ~\\textrm{{qubits}} ~,~ L = N/{round(N/M)} ~\\textrm{{layers}}$"%(str(round(log10(pi/parameters))) if round(log10(pi/parameters))>1 else ''),
	# 							"pad":20,
	# 							}
	# 					}.get(y),
	# 				"set_ylabel": {
	# 					'v.condition':
	# 						{
	# 						"ylabel": "$\\textrm{Condition Number} ~~ \\kappa(A)$",
	# 						"obj": None
	# 						},
	# 					'uv.error':
	# 						{
	# 						"ylabel": "$\\textrm{Error} ~~ \\norm{M - UV}_{F}/\\norm{M}_{F}$",
	# 						"obj": "twinx"
	# 						}
	# 					}.get(y),
	# 				"set_xlabel": {
	# 					"xlabel": "$\\textrm{Gate Index} ~~ i \\in [L(N-1)]$"
	# 				},
	# 				"set_xscale": {"value": "linear"},
	# 				"set_xlim":{"xmin":-1,"xmax":M*(N-1)+1},
	# 				"set_xticks":{"ticks":[*range(0,M*(N-1)+M,N-1)]},					
	# 				"set_xticklabels":{"labels":[f"${i}N$" if i>1 else f"$N$" if i==1 else f"${i}$" for i in range(0,M+1)]},					
	# 				"set_yscale":{
	# 					"u.condition":{"value": "log","base": 10,"obj":None},
	# 					"v.condition":{"value": "log","base": 10,"obj":None},
	# 					"uv.error":{"value": "log","base": 10,"obj":"twinx"},
	# 				}.get(y),						
	# 				"set_ylim":{"ymin":1e-1,"ymax":1e25},
	# 				"set_yticks":{"ticks":[1e0,1e4,1e8,1e12,1e16,1e20,1e24]},
	# 				"set_ylim":{"ymin":1e-1,"ymax":1e41},
	# 				"set_yticks":{"ticks":[1e0,1e5,1e10,1e15,1e20,1e25,1e30,1e35,1e40]},	
	# 				"set_ylim":{
	# 					"u.condition":{"ymin":1e-1,"ymax":1e21,"obj":None},
	# 					"v.condition":{"ymin":1e-1,"ymax":1e21,"obj":None},
	# 					"uv.error":{"ymin":5.5e-6,"ymax":1.75e0,"obj":"twinx"},
	# 				}.get(y),
	# 				"set_yticks":{
	# 					"u.condition":{"ticks":[1e0,1e4,1e8,1e12,1e16,1e20],"obj":None},
	# 					"v.condition":{"ticks":[1e0,1e4,1e8,1e12,1e16,1e20],"obj":None},
	# 					"uv.error":{"ticks":[1e-5,1e-4,1e-3,1e-2,1e-1,1e0],"obj":"twinx"},
	# 				}.get(y),					
	# 				"set_aspect": {
	# 					"aspect": "auto"
	# 				},
	# 				"grid": {
	# 					"visible": True,
	# 					"which": "major",
	# 					"axis": "both"
	# 				},
	# 				"legend":{
	# 					"title":"$\\textrm{Matrix}$",
	# 					"title_fontsize": 36,
	# 					"fontsize": 36,
	# 					"markerscale": 1.5,
	# 					"handlelength": 3,
	# 					"framealpha": 0.8,				
	# 					"ncol":3
	# 					}
	# 			},
	# 			"style":{
	# 				"mplstyle": "scratch/nmf/data/plot.mplstyle"
	# 			}
	# 		}
	# 		for index,(x,y) in enumerate(((None,'u.condition'),(None,'v.condition'),(None,'uv.error')))
	# 		]
	# 	for index,settings in enumerate(settings):
	# 		fig,ax = plot(settings=settings,fig=fig,ax=ax)



	if boolean(path):

		state = {i:'state' 
			for i in range(N)}
		data = {index:(data,where) 
			for index,(data,where) in enumerate((data,where) 
			for i in [*range(0,N-1)] for where in [(i,i+1)] 
			for data in ['X'])}


		kwargs = dict(
			D=D,N=N,M=M,
			parameters={'unitary':parameters,'identity':parameters,'X':parameters,'depolarize':noise},
			variables={attr:[] for attr in ['u.condition','v.condition','u.spectrum','v.spectrum','uv.error']},
			options=dict(
				scheme='nmf',
				init='nndsvd',
				iteration=int(1e3),
				eps=1e-10,
				alpha=1,
				beta=5e-1,
				gamma=1e-10,
				delta=1,
				iota=6e-1,
				sigma=1e-4,
				update=[
					# {'update':'gd','iteration':int(1e6),'eps':1e-14},
					# {'update':'cg','iteration':int(1e5),'eps':1e-14},
					# {'update':'cg','iteration':int(100),'eps':1e-14},
					# {'update':'cp','iteration':int(1e1),'eps':1e-10},
					{'update':'pc','iteration':int(1e3),'eps':1e-10},
					# {'update':'sd','iteration':int(1e6),'eps':1e-14},
					# {'update':'qp','iteration':int(1),'eps':1e-8},
					# {'update':'pd','iteration':int(1),'eps':1e-14},
					# {'update':'gd','iteration':int(1e6),'eps':1e-14},					
					# {'update':'ls','iteration':int(1e5),'eps':1e-14},					
					# {'update':'mhu','iteration':int(1),'eps':1e-14},
					# {'update':'gd','iteration':int(1e4),'eps':1e-14},					
					# {'update':'cd','iteration':int(1e6),'eps':1e-14},
					# {'update':'mu','iteration':int(1e3),'eps':1e-14},
					# {'update':'mru','iteration':int(1e4),'eps':1e-14},
					# {'update':'mhu','iteration':int(1e4),'eps':1e-14},
					# {'update':'miu','iteration':int(1e4),'eps':1e-14},
					# {'update':'cd','iteration':int(1e6),'eps':1e-14},
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

		exit()

		_state = {i:'state' for i in range(N)}
		_data = {index:(data,where) for index,(data,where) in enumerate((data,where) for i in [*range(0,N-1)] for where in [(i,i+1)] for data in ['unitary'])}

		_kwargs = dict(
			D=D,N=N,M=M,
			parameters={'unitary':parameters,'depolarize':noise},
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

		# print(basis.transform(state,transform=None,**kwargs).round(14))
		# print(basis.transform(_state,transform=None,**_kwargs).round(14))

		spectrum = basis.spectrum(state,where=L,**{**kwargs,**dict(options={**kwargs['options'],'scheme':'probability'})})
		_spectrum = basis.spectrum(state,where=L,**{**kwargs,**dict(options={**kwargs['options'],'scheme':'spectrum'})})

		spectrum = spectrum/maximum(absolute(spectrum))
		_spectrum = _spectrum/maximum(absolute(_spectrum))

		state = basis.transform(state,transform=False,**kwargs)
		_state = basis.transform(_state,transform=False,**_kwargs)

		tmp = np.sort((state).real)
		_tmp = np.sort((_state).real)
		print(array([*tmp[-1:-10:-1],*tmp[:10]]))
		print(array([*_tmp[-1:-10:-1],*_tmp[:10]]))

		print((absolute(state-_state)**2).sum())
		print(state.real.sum(),abs(state.real.sum()-1),abs(_state.real.sum()-1))

		data = dict(
			D=D,N=N,M=M,L=L,K=K,parameters=parameters,noise=noise,seed=seed,
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
					"fname": "scratch/nmf/data/plot.pdf",
					"bbox_inches": "tight",
					"pad_inches": 0.2
				}
			},
			"ax":{
				"errorbar":{
					"x":[*arange(len(data[y]))] if x is None else x,
					"y":[*data[y]],
					"label":{"spectrum.nmf":"$\\textrm{NMF}$","spectrum.svd":"$\\textrm{SVD}$"}.get(y),
					"alpha":0.8,
					"marker":"o",
					"markersize":8,
					"linestyle": "--",
					"linewidth": 4,
					"elinewidth": 2,
					"capsize": 3,
					"color":{"spectrum.nmf":"black","spectrum.svd":"gray"}.get(y),
					},
				"set_title": {
					"label": f"$\\textrm{{Haar + Depolarize}} \\quad N = {N} ~,~ M = 2N ~,~ L = N/2 ~,~ D = {D} ~,~ \\chi = D^{{N}} ~,~ \\gamma = 10^{{{int(log10(noise))}}}$",
					"label": f"$\\textrm{{Haar + Depolarize}} \\quad N = {N} ~,~ M = 2N ~,~ L = N/2 ~,~ D = {D} ~,~ \\chi = D^{{N}} ~,~ \\gamma = 10^{{{int(log10(noise))}}}$",
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
				"mplstyle": "scratch/nmf/data/plot.mplstyle"
			}
		}
		for index,(x,y) in enumerate(((None,"spectrum.nmf"),(None,"spectrum.svd")))
		]
	for index,settings in enumerate(settings):
		fig,ax = plot(settings=settings,fig=fig,ax=ax)

	exit()

	assert allclose(basis.transform(state,transform=False,**kwargs).sum(),1) and allclose(prod(_state[i].sum() for i in _state),1)


	state = basis.transform(state,transform=False,**kwargs)
	_state = basis.transform(_state,transform=False,**_kwargs)


	assert allclose(state,_state)


	print('Passed')

	return

def test_precondition(*args,**kwargs):

	N = 4
	D = 2
	d = D**N
	rank = None
	seed = 123
	key = seeder(seed)
	dtype = 'float'
	length = D**2
	shape = (length,length)
	kwargs = dict(shape=shape,key=key,dtype=dtype)
	l = length
	gamma = 1
	I = identity(length,dtype=dtype)

	M = rand(key=key,shape=shape,dtype=dtype)
	M /= add(M)
	U,V,S = nndsvd(M,rank=rank)

	x = V
	A = dot(U.T,U)
	b = dot(U.T,M)


	P = randn(**{**kwargs,**dict(shape=A.shape,dtype=A.dtype)})

	U,S = nystrom(A,P)

	# U = invreg(U,S,gamma)


	print(U)
	print(S)

	return

	
	# P = randn(**{**kwargs,**dict(shape=A.shape,dtype=A.dtype)})
	# Q = dot(P,A)
	# P = dot(Q,solve(dot(P.T,Q),Q.T))
	# S,U = eig(P,compute_v=True,hermitian=True)
	# P = (S[-l]+gamma)*dot((U/(S+gamma*I)),U.T) + (I - dot(U,U.T))
	# Q = sqrtm(P)

	# P = A
	# S,U = eig(P,compute_v=True,hermitian=True)
	# P = dot(U[:,:-l]*S[:-l],U[:,:-l].T)

	debug(k=array((condition_number(A+gamma*I),condition_number(dot(dot(Q,A+gamma*I),Q)))))

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

		b = dot(dotr(u,sign(s)*sqrt(absolute(s))),dotl(v,sign(s)*sqrt(absolute(s))))

		time = timing.time() - time

		error = abs(norm(a-b)/norm(a))

		print(kwargs)
		print(add(a),add(b))
		print('%0.4f    %0.5e'%(time,error))

		return

	n = 6
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

	args = list()
	kwargs = dict()

	# test_shuffle(*args,**kwargs)
	test_mps(*args,**kwargs)
	# test_precondition(*args,**kwargs)
	# test_nmf(*args,**kwargs)
