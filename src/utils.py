#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
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
import logging,logging.config
logger = logging.getLogger(__name__)
conf = 'config/logging.conf'
try: 
	logging.config.fileConfig(conf,disable_existing_loggers=False) 
except:
	pass
logger = logging.getLogger(__name__)



# Constants
pi = np.pi
e = np.exp(1)


def jit(func,*,static_argnums=None):
	'''
	Just-in-time compile function
	Args:
		func (callable): Function to compile
		static_argnums (dict): Arguments to statically compile
	Returns:
		func (callable): Compiled function
	'''
	# return func
	return jax.jit(func,static_argnums=static_argnums)






# @partial(jit,static_argnums=(2,))	
def vmap(func,in_axes=0,out_axes=0,axes_name=None):	
	'''
	Vectorize function over input axes of iterables
	Args:
		func (callable): Function that acts on single elements of iterables
		in_axes (int,iterable): Input axes of iterables
		out_axes (int,interable): Output axes of func return
		axes_names (object): hashable Python object used to identify the mapped
			axis so that parallel collectives can be applied.
	Returns:
		vfunc (callable): Vectorized function with signature vfunc(*iterables) = [func(*iterables[axes_in][0]),...,func(*iterables[axes_in][n-1])]
	'''
	return jax.vmap(func,in_axes,out_axes,axes_name)


# @partial(jit,static_argnums=(2,))	
def forloop(start,end,func,out):	
	'''
	Perform loop of func from start to end indices
	Args:
		start (int): Start index of loop
		end (int): End index of loop
		func (callable): Function that acts on iterables with signature func(i,out)
		out (array): Initial value of loop
	Returns:
		out (array): Return of loop
	'''
	return jax.lax.fori_loop(start,end,func,out)


# @partial(jit,static_argnums=(2,))
def vfunc(funcs,index):	
	'''
	Vectorize indexed functions over operands
	Args:
		funcs (iterable[callable]): Functions that act on that acts on single elements of iterables
		index (iterable[int]): Iterable of indices of functions to call
	Returns:
		vfunc (callable): Vectorized function with signature vfunc(*iterables) = [func(*iterables[axes_in][0]),...,func(*iterables[axes_in][n-1])]
	'''
	# def vfunc(*iterables):
	# 	return array(list(map(lambda i,*x:funcs[i](*x),*[index,*iterables])))
	# return vfunc
	# func = jit(vmap(lambda i,x,funcs=funcs: jax.lax.switch(i,funcs,x)))
	func = lambda index,x: array([funcs[i](x[i]) for i in index])
	return lambda x,func=func,index=index: func(index,x)



def value_and_gradient(func):
	'''
	Compute value and gradient of function
	Args:
		func (callable): Function to compile
	Returns:
		value_and_grad (callable): Value and Gradient of function
	'''
	@jit
	def _value_and_gradient(x):
		return jax.value_and_grad(func)(x)

	return _value_and_gradient

def gradient(func):
	'''
	Compute gradient of function
	Args:
		func (callable): Function to compile
	Returns:
		grad (callable): Gradient of function
	'''
	@jit
	def _gradient(x):
		return jax.grad(func)(x)

	return _gradient	


def gradient_finite(func,tol=1e-6):
	'''
	Calculate finite difference second order derivative of function
	Args:
		func (callable): Function to derive, with signature func(x) and output shape
		tol (float): Finite difference tolerance
	Returns:
		out (array): Array of gradient
	'''
	@jit
	def _gradient(x):
		return vmap(lambda v,x=x,tol=tol: (func(x+tol*v)-func(x-tol*v))/(2*tol))(eye(x.size))

	return _gradient


def value_and_gradient_finite(func,tol=1e-6):
	'''
	Calculate finite difference second order derivative of function
	Args:
		func (callable): Function to derive, with signature func(x) and output shape
		tol (float): Finite difference tolerance		
	Returns:
		out (array): Array of gradient
	'''
	@jit
	def _value_and_gradient(x):
		return (func(x),vmap(lambda v,x=x,tol=tol: (func(x+tol*v)-func(x-tol*v))/(2*tol))(eye(x.size)))
	
	return _value_and_gradient


def gradient_fwd(func):
	'''
	Compute forward gradient of function
	Args:
		func (callable): Function to compile
	Returns:
		grad (callable): Gradient of function
	'''
	@jit
	def _gradient(x):
		return moveaxis(jax.jacfwd(func)(x),-1,0)

	return _gradient

def gradient_rev(func):
	'''
	Compute reverse gradient of function
	Args:
		func (callable): Function to compile
	Returns:
		grad (callable): Gradient of function
	'''
	@jit
	def _gradient(x):
		return moveaxis(jax.jacrev(func)(x),-1,0)

	return _gradient

class dictionary(dict):
	'''
	Dictionary subclass with dictionary elements explicitly accessible as class attributes
	Args:
		args (dict): Dictionary elements
		kwargs (dict): Dictionary elements
	'''
	def __init__(self,*args,**kwargs):
		
		args = {k:v for a in args for k,v in ({} if a is None else a).items()}		
		kwargs = kwargs
		attrs = {**args,**kwargs}

		for attr in attrs:
			setattr(self,attr,attrs[attr])

		super().__init__(attrs)

		return

	def __getattribute__(self,item):
		return super().__getattribute__(item)

	# def __getattr__(self,item):
	# 	return super().__getattr__(item)

	# def __setattr__(self,item,value):
	# 	return super().__setitem__(item,value)

	def __getitem__(self,item):
		return super().__getitem__(item)

	def __setitem__(self,item,value):
		return super().__setitem__(item,value)

	def __iter__(self):
		return super().__iter__()

	def __len__(self):
		return super().__len__()


class Array(onp.ndarray):
	'''
	Numpy array subclass, subclass of nd.ndarray with array of data of shape
	
	Classes that inherit Array must have a __setup__() in place of __init__() to properly initialize
	subclassed ndarray, that is initalized through Array's __new__() and an array.view(cls)
	
	Classes that inherit Array's __setup__() create the following attributes:
		self.data (array)
		self.string (str)

	Classes that inherit Array's __setup__() must also append newly institated attributes to the 
	self.attrs attribute dictionary to allow for proper re-instantiation of the inherited class 
	upon numpy-like views, slices etc.

	Args:
		data (array,list,tuple): array data if array or list, array shape if tuple
		args (iterable): class attributes
		kwargs (dict): class keyword attributes
	'''
	def __new__(cls,*args,**kwargs):
		attrs = {}
		clsattrs = {}
		for attr,value in kwargs.items():
			if attr in onp.ndarray.__dict__:
				clsattrs[attr] = value
			else:
				attrs[attr] = value

		field,value = 'attrs',attrs
		setattr(cls,field,value)

		cls = cls.__new__(cls,*args,**kwargs)

		obj = onp.asarray(cls.data,**clsattrs).view(cls)

		field,value = 'attrs',attrs
		setattr(obj,field,value)

		for attr,value in attrs.items():
			setattr(obj,attr,getattr(obj,attr,value))

		return obj

	def __array_finalize__(self, obj):
		field = 'attrs'
		attrs = getattr(obj,field,getattr(self,field,None))
		if obj is None or attrs is None: 
			return
		for attr in attrs:
			setattr(self,attr,getattr(obj,attr,getattr(self,attr,None)))
		return

	
	def __setup__(self,*args,**kwargs):
		
		field = 'attrs'
		getattr(self,field).update({attr:getattr(self,attr,None) for attr in kwargs if attr not in getattr(self,field)})
		
		data = args[0] if len(args)>0 else None
		if data is None:
			data = onp.array()
		elif isinstance(data,tuple):
			data = onp.zeros(data)
		else:
			data = onp.asarray(data)
		self.data = data
		self.string = data.__str__()
		return

	def __repr__(self):
		return self.string

	def to_numpy(self):
		return onp.asarray(self)

	def to_jax(self):
		return np.asarray(self)

	def log(self,msg):
		'''
		Log messages
		Args:
			msg (str): Message to log
		'''
		logger.log(self.verbose,msg)
		return

class String(str):
	'''
	String class to represent concatenated and delimited strings, subclass of str
	Args:
		string (str,tuple,list): string, tuple of strings, or list of tuple of strings or strings, where strings in tuple are concatenated and strings in list are delimited		
	'''
	def __new__(cls,string,deliminate=' + ',concatenate='*',fmt=None):
		try:
			string = [tuple([substring]) if not isinstance(substring,tuple) else substring 
						for substring in ([string] if not isinstance(string,list) else string)]
			string = deliminate.join([concatenate.join(substring) for substring in string])
		except:
			string = ''
		string = fmt%string if fmt is not None else string
		obj = super().__new__(cls,string)
		obj.deliminate = deliminate
		obj.concatenate = concatenate			
		return obj

	def __getslice__(self,item):
		try:
			obj = self.deliminate.join(self.split(self.deliminate)[item])
		except:
			obj = self
		return obj

	def __getitem__(self,item):
		try:
			obj = self.split(self.deliminate)[item]
		except:
			obj = self
		return obj


class System(dictionary):
	'''
	System attributes (dtype,format,device,seed,verbose,...)
	Args:
		dtype (str,data-type): Data type of class
		format (str): Format of array
		device (str): Device for computation
		seed (array,int): Seed for random number generation
		verbose (bool,str): Verbosity of class	
		args (dict,System): Additional system attributes
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,*args,**kwargs):


		updates = {
			'verbose':{
				'notset':0,'debug':10,'info':20,'warning':30,'error':40,'critical':50,
				'Notset':0,'Debug':10,'Info':20,'Warning':30,'Error':40,'Critical':50,
				'NOTSET':0,'DEBUG':10,'INFO':20,'WARNING':30,'ERROR':40,'CRITICAL':50,
				10:10,20:20,30:30,40:40,50:50,
				2:20,3:30,4:40,5:50,
				True:20,False:0,None:0,
				}
			}

		defaults = {
			'dtype':'complex',
			'format':'array',
			'device':'cpu',
			'seed':None,
			'verbose':False,
		}

		args = {k:v for a in args for k,v in ({} if a is None else a).items()}
		attrs = {**args,**kwargs}
		attrs.update({attr: defaults[attr] for attr in defaults if attrs.get(attr) is None})
		attrs.update({attr: updates.get(attr,{}).get(attrs[attr],attrs[attr]) for attr in attrs})

		super().__init__(**attrs)

		return



def decorator(*args,**kwargs):
	'''
	Wrap function with args and kwargs
	'''
	def wrapper(func):
		@functools.wraps
		def wrapped(*_args,**_kwargs):
			arg = (*_args,*args)
			kwarg = {**kwargs,**_kwargs}
			return func(*arg,**kwarg)
		return wrapped
	return wrapper

class objs(onp.ndarray):
	'''
	array class of objects
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,*args,**kwargs):
		return onp.array(*args,**kwargs)


class asobjs(onp.ndarray):
	'''
	array class of objects
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,*args,**kwargs):
		return onp.asarray(*args,**kwargs)


class array(np.ndarray):
	'''
	array class
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,*args,**kwargs):
		return jax.device_put(np.array(*args,**kwargs))


class asarray(onp.ndarray):
	'''
	array class
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,*args,**kwargs):
		return jax.device_put(np.asarray(*args,**kwargs))

class ones(array):
	'''
	array class of ones
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,*args,**kwargs):
		return jax.device_put(np.ones(*args,**kwargs))


class zeros(array):
	'''
	array class of zeros
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,*args,**kwargs):
		return jax.device_put(np.zeros(*args,**kwargs))

class eye(array):
	'''
	array class of eye
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,*args,**kwargs):
		return jax.device_put(np.eye(*args,**kwargs))

class arange(array):
	'''
	array class of arange
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,*args,**kwargs):
		return jax.device_put(np.arange(*args,**kwargs))


class identity(array):
	'''
	array class of identity
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,n,*args,**kwargs):
		return jax.device_put(np.eye(n,*args,**kwargs))


class hadamard(array):
	'''
	array class of hadamard
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,n,*args,**kwargs):
		if n == 1:
			out = array(1,*args,**kwargs)
			return out
		else:
			out = hadamard(n-1,*args,**kwargs)
			return 1/sqrt(2)*array([[out,out],[out,-out]],*args,**kwargs)


class phasehadamard(array):
	'''
	array class of phaseshadamard
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,n,*args,**kwargs):
		assert n == 2, "phasehadamard only defined for n = 2"
		if n == 1:
			out = array(1,*args,**kwargs)
			return out
		elif n == 2:
			out = 1/sqrt(2)*array([[1,1],[1j,-1j]],*args,**kwargs)
			return out
		else:
			out = phasehadamard(n-1,*args,**kwargs)
			return 1/sqrt(2)*array([[out,out],[out,-out]],*args,**kwargs)


class cnot(array):
	'''
	array class of cnot
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,n,*args,**kwargs):
		assert n == 4, "cnot only defined for n = 4"
		if n == 1:
			out = array(1,*args,**kwargs)
			return out
		elif n == 4:
			out = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],*args,**kwargs)
			return out
		else:
			out = cnot(n-1,*args,**kwargs)
			return array([[out,out],[out,-out]],*args,**kwargs)


class toffoli(array):
	'''
	array class of toffoli
	Args:
		args (iterable): Array arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(self,n,*args,**kwargs):
		assert n == 8, "toffoli only defined for n = 8"
		if n == 1:
			out = array(1,*args,**kwargs)
			return out
		elif n == 8:
			out = array([
				[1,0,0,0,0,0,0,0],
				[0,1,0,0,0,0,0,0],
				[0,0,1,0,0,0,0,0],
				[0,0,0,1,0,0,0,0],
				[0,0,0,0,1,0,0,0],
				[0,0,0,0,0,1,0,0],
				[0,0,0,0,0,0,0,1],
				[0,0,0,0,0,0,1,0]]
				,*args,**kwargs)
			return out
		else:
			out = toffoli(n-1,*args,**kwargs)
			return array([[out,out],[out,-out]],*args,**kwargs)


def rand(shape=None,bounds=[0,1],key=None,random='uniform'):
	'''
	Get random array
	Args:
		shape (int,iterable): Size or Shape of random array
		key (jax.key,int): PRNG key or seed
		bounds (iterable): Bounds on array
		random (str): Type of random distribution
	Returns:
		out (array): Random array
	'''	
	if shape is None:
		shape = 1
	elif isinstance(shape,int):
		shape = shape

	if key is None:
		key = PRNGKey(key)
	elif isinstance(key,int):
		key = PRNGKey(key)

	if random in ['uniform','rand']:
		return jax.random.uniform(key,shape,minval=bounds[0],maxval=bounds[1])
	elif random in ['zeros']:
		return zeros(shape)
	elif random in ['ones']:
		return ones(shape)		
	else:
		return jax.random.uniform(key,shape,minval=bounds[0],maxval=bounds[1])


def svd(A,k=None):
	'''
	Perform SVD on array, possibly reduced rank k
	Args:
		A (array): array of shape (n,m)
		k (int): reduced rank of SVD, defaults to max(n,m) if None
	Returns
		U (array): unitary of left eigenvectors of A of shape (n,k)
		S (array): array of singular values of shape (k,)
		V (array): conjugate of unitary of right eigenvectors of A of shape (k,m)
	'''

	n,m = A.shape

	U,S,V = onp.linalg.svd(A)

	p,r,q = U.shape[0],S.size,V.shape[0]

	if k > p:
		U = onp.concatenate((U,onp.zeros((n,k-p))),axis=1)
	else:
		U = U[:,:k]
	if k > r:
		S = onp.concatenate((S,onp.zeros(k-r)),axis=0)
	else:
		S = S[:k]		
	if k > q:
		V = onp.concatenate((V,onp.zeros((k-q,m))),axis=0)
	else:
		V = V[:k,:]


	S = onp.diag(S)

	return U,S,V


@partial(jit,static_argnums=(1,2,))
def norm(a,axis=None,ord=2):
	'''
	Norm of array
	Args:
		a (array): array to be normalized
		axis (int): axis to normalize over. Flattens array if None.
		ord (int,str): order of normalization
	'''

	out = np.linalg.norm(a,axis=axis,ord=ord)

	return out


@jit
def inner(a,b):
	'''
	Calculate inner product of arrays a and b
	Args:
		a (array): Array to calculate inner product
		b (array): Array to calculate inner product
	Returns:
		out (array): Inner product
	'''	
	return trace(tensordot(a,b.conj().T,1))/sqrt(a.shape[0]*b.shape[0])


@jit
def gradient_inner(a,b,da):
	'''
	Calculate gradient of inner product of arrays a and b with respect to a
	Args:
		a (array): Array to calculate inner product
		b (array): Array to calculate inner product
		da (array): Gradient of array to calculate inner product
	Returns:
		out (array): Gradient of inner product
	'''
	def func(i):
		return trace(tensordot(da[i],b.conj().T,1))/sqrt(a.shape[0]*b.shape[0])
	return vmap(func)(arange(da.shape[0]))


@jit
def inner_abs2(a,b):
	'''
	Calculate absolute square of inner product of arrays a and b
	Args:
		a (array): Array to calculate inner product
		b (array): Array to calculate inner product
	Returns:
		out (array): Absolute square of inner product
	'''	
	return abs2(trace(tensordot(a,b.conj().T,1)))/(a.shape[0]*b.shape[0])


@jit
def gradient_inner_abs2(a,b,da):
	'''
	Calculate gradient of absolute square inner product of arrays a and b with respect to a
	Args:
		a (array): Array to calculate inner product
		b (array): Array to calculate inner product
		da (array): Gradient of array to calculate inner product
	Returns:
		out (array): Gradient of inner product
	'''
	def func(i):
		return 2*(
			trace(tensordot(da[i],b.conj().T,1))*
			trace(tensordot(a.conj(),b.T,1))).real/(a.shape[0]*b.shape[0])
	return vmap(func)(arange(da.shape[0]))


@jit
def inner_real2(a,b):
	'''
	Calculate real square of inner product of arrays a and b
	Args:
		a (array): Array to calculate inner product
		b (array): Array to calculate inner product
	Returns:
		out (array): Real square of inner product
	'''	
	return (abs2((trace(tensordot(a,b.conj().T,1))).real))/(a.shape[0]*b.shape[0])


@jit
def gradient_inner_real2(a,b,da):
	'''
	Calculate gradient of real square inner product of arrays a and b with respect to a
	Args:
		a (array): Array to calculate inner product
		b (array): Array to calculate inner product
		da (array): Gradient of array to calculate inner product
	Returns:
		out (array): Gradient of inner product
	'''
	def func(i):
		return 2*(trace(tensordot(da[i],b.conj().T,1)).real)*(trace(tensordot(a,b.conj().T,1))).real/(a.shape[0]*b.shape[0])
	return vmap(func)(arange(da.shape[0]))


@jit
def inner_imag2(a,b):
	'''
	Calculate imaginary square of inner product of arrays a and b
	Args:
		a (array): Array to calculate inner product
		b (array): Array to calculate inner product
	Returns:
		out (array): Imaginary square of inner product
	'''	
	return (abs2((trace(tensordot(a,b.conj().T,1))).imag))/(a.shape[0]*b.shape[0])


@jit
def gradient_inner_imag2(a,b,da):
	'''
	Calculate gradient of imaginary square inner product of arrays a and b with respect to a
	Args:
		a (array): Array to calculate inner product
		b (array): Array to calculate inner product
		da (array): Gradient of array to calculate inner product
	Returns:
		out (array): Gradient of inner product
	'''
	def func(i):
		return 2*(trace(tensordot(da[i],b.conj().T,1)).imag)*(trace(tensordot(a,b.conj().T,1))).imag/(a.shape[0]*b.shape[0])
	return vmap(func)(arange(da.shape[0]))



@jit
def _multiply(a,b):
	'''
	Multiply arrays elementwise
	Args:
		a (array): Array to multiply
		b (array): Array to multiply
	Returns:
		out (array): Elementwise multiplication of arrays
	'''
	return np.multiply(a,b)


@jit
def multiply(a):
	'''
	Multiply list of arrays elementwise
	Args:
		a(array): Arrays to multiply
	Returns:
		out (array): Elementwise multiplication of arrays
	'''
	out = a[0]

	sparse = issparse(out)

	for b in a[1:]:
		if not sparse:
			out = out*b
		else:
			out = out.multiply(b)
	
	return out


@jit
def _add(a,b):
	'''
	Add arrays elementwise
	Args:
		a (array): Array to add
		b (array): Array to add
	Returns:
		out (ndarray): Elementwise addition of arrays
	'''
	return np.add(a,b)

@jit
def add(a):
	'''
	Add list of arrays elementwise
	Args:
		a (iterable): Arrays to add
	Returns:
		out (ndarray) if out argument is not None
	'''
	return forloop(1,len(a),lambda i,out: _add(out,a[i]),a[0])

@jit
def _matmul(a,b):
	'''
	Calculate matrix product arrays a and b
	Args:
		a (array): Array to calculate matrix product
		b (array): Array to calculate matrix product
	Returns:
		out (array): Matrix product
	'''	
	return np.matmul(a,b)

@jit
def matmul(a):
	'''
	Get matrix product of iterable of arrays [a_i], where a_{i}.shape[-1] == a_{i+1}.shape[0]
	Args:
		a (iterable): Arrays to compute product of arrays
	Returns:
		out (array): Reduced array of matrix product of arrays
	'''
	return forloop(1,len(a),lambda i,out: _matmul(out,a[i]),a[0])

@jit
def multi_dot(a):
	'''
	Get matrix product of iterable of arrays [a_i], where a_{i}.shape[-1] == a_{i+1}.shape[0] with optimized ordering of products
	Args:
		a (iterable): Arrays to compute product of arrays
	Returns:
		out (array): Reduced array of matrix product of arrays
	'''
	return np.linalg.multi_dot(a)

@partial(jit,static_argnums=(1,))
def prod(a,axes=0):
	'''
	Get product of elements in array along axes
	Args:
		a (array): Array to compute product of elements
		axes (int): axes to perform product
	Returns:
		out (array): Reduced array of product of elements along axes
	'''
	return np.prod(a,axes)

@partial(jit,static_argnums=(2,))
def vtensordot(a,b,axes=0):
	'''
	Tensordot product of arrays a and b
	Args:
		a (array): Array to multiply
		b (array): Array to multiply
		axes (int,iterable): axes to perform tensordot 
	Returns:
		out (array): Dot product of array along axes
	'''
	return vmap(lambda a: tensordot(a,b,axes))(a)

@partial(jit,static_argnums=(2,))
def tensordot(a,b,axes=0):
	'''
	Tensordot product of arrays a and b
	Args:
		a (array): Array to multiply
		b (array): Array to multiply
		axes (int,iterable): axes to perform tensordot 
	Returns:
		out (array): Dot product of array along axes
	'''
	return np.tensordot(a,b,axes)


@jit
def _tensorprod(a,b):
	'''
	Tensor (kronecker) product of arrays a and b	
	Args:
		a (array): Array to tensorproduct
		b (array): Array to tensorproduct
	Returns:
		out (array): Tensorproduct product of arrays
	'''
	return np.kron(a,b)


@jit
def tensorprod(a):
	'''
	Tensor (kronecker) product of arrays a
	Args:
		a (iterable): Array to perform kronecker product
	Returns:
		out (array): Kronecker product of array
	'''
	out = a[0]
	for i in range(1,len(a)):
		out = _tensorprod(out,a[i])
	return out
	# return forloop(1,len(a),lambda i,out: _tensorprod(out,a[i]),a[0])	

@jit
def vtensorprod(a):
	'''
	Tensor (kronecker) product of arrays a
	Args:
		a (iterable): Array to perform kronecker product
	Returns:
		out (array): Kronecker product of array
	'''
	return vmap(tensorprod)(a)



@jit
def ntensorprod(a,n):
	'''
	Tensor (kronecker) product of arrays a
	Args:
		a (iterable): Array to perform kronecker product
		n (int): Perform kronecker product n times
	Returns:
		out (array): Kronecker product of array
	'''
	out = a
	for i in range(1,n):
		out = _tensorpod(out,a)
	return out
	# return forloop(1,n,lambda i,out: _tensorprod(out,a),a)	

@jit
def vntensorprod(a,n):
	'''
	Tensor (kronecker) product of arrays a
	Args:
		a (iterable): Array to perform kronecker product
		n (int): Perform kronecker product n times
	Returns:
		out (array): Kronecker product of array
	'''
	return vmap(lambda a: ntensorprod(a,n))(a)


@jit
def einsum(a,subscripts):
	'''
	Get summation of axes in array denoted by subscripts
	Args:
		a (iterable): Arrays to compute summation of elements
		subscripts (str): operations to perform for summation
	Returns:
		out (array): Reduced array of summed array
	'''
	optimize = np.einsum_path(subscripts)
	return np.einsum(*a,optimize=optimize)




@jit
def contraction(parameters,data,identity):
	'''
	Calculate matrix product of parameters times data
	Args:
		parameters (array): parameters of shape (k,) or (k,n,) or (k,n,n)		
		data (array): Array of data to matrix product of shape (l,n,n)
		identity (array): Array of data identity
	Returns:
		out (array): Matrix product of data of shape (n,n)
	'''	
	return matmul(parameters*data)


@jit
def summation(parameters,data,identity):
	'''
	Calculate matrix sum of parameters times data
	Args:
		parameters (array): parameters of shape (k,) or (k,n,) or (k,n,n)		
		data (array): Array of data to matrix sum of shape (l,n,n)
		identity (array): Array of data identity
	Returns:
		out (array): Matrix sum of data of shape (n,n)
	'''	
	return add(parameters*data)


@jit
def multiplication(parameters,data,identity):
	'''
	Calculate tensor product of parameters times data
	Args:
		parameters (array): parameters of shape (k,) or (k,n,) or (k,n,n)		
		data (array): Array of data to tensor multiply of shape (l,n,n)
		identity (array): Array of data identity
	Returns:
		out (array): Tensor product of data of shape (n,n)
	'''		
	return tensorprod(data)


@jit
def exponentiation(parameters,data,identity):
	'''
	Calculate matrix exponential of parameters times data
	Args:
		parameters (array): parameters of shape (k,) or (k,n,) or (k,n,n)		
		data (array): Array of data to matrix exponentiate of shape (l,n,n)
		identity (array): Array of data identity
	Returns:
		out (array): Matrix exponential of data of shape (n,n)
	'''		
	return expm(parameters,data,identity)



@jit
def swap(i,j,N,D):
	'''
	Swap array elements acting on partition i <-> j on N partition array, each of D dimensions
	Args:
		i (int): partition to swap
		j (int): partition to swap
		N (int): Number of partitions
		D (int): Dimension of each partition
	Returns:
		S (array): Swap array
	'''
	s = eye(D,dtype=int)
	I = [s.copy() for n in range(N)]
	S = 0
	for a in range(D):
		for b in range(D):
			I[i][:],I[j][:] = 0,0
			I[i][a][b],I[j][b][a] = 1,1
			S += tensorprod(I)
			I[i][:],I[j][:] = s,s
	return S



@partial(jit,static_argnums=(1,2,))	
def slices(a,start,size):
	'''
	Get slice of array
	Args:
		a (array): Array to be sliced along first dimension
		start (int): Start index to slice
		size (int): Length of slice
	Returns:
		a (array): Sliced array
	'''
	return jax.lax.dynamic_slice(a,(start,*[0]*(a.ndim-1),),(size,*a.shape[1:]))


def powerset(iterable,probability=1):
	'''
	Get power set of iterable
	Args:
		iterable (iterable): iterable to get powerset of
		probability (float): probability of returning element from power set
	Yields:
		iteration (iterable): iteration of powerset of iterable
	'''

	for number,iteration in enumerate(itertools.product(*[[[], [i]] for i in iterable])):
		iteration = (j for i in iteration for j in i)
		if (probability==1) or (onp.random.rand() >= probability):
			yield iteration
		else:
			continue


@jit
def commutator(a,b):
	'''
	Calculate commutator of ab - ba
	Args:
		a (array): Array to calculate commutator
		b (array): Array to calculate commutator
	Returns:
		out (array): commutator
	'''	
	return tensordot(a,b,1) - tensordot(b,a,1)


@jit
def commutes(a,b):
	'''
	Calculate whether a and b commutate
	Args:
		a (array): Array to calculate commutator
		b (array): Array to calculate commutator
	Returns:
		commutes (bool): Boolean of a and b commuting
	'''	
	return bool(~(commutator(a,b).any()))

@jit
def anticommutator(a,b):
	'''
	Calculate anticommutator of ab + ba
	Args:
		a (array): Array to calculate anticommutator
		b (array): Array to calculate anticommutator
	Returns:
		out (array): anticommutator
	'''	
	return tensordot(a,b,1) + tensordot(b,a,1)


@jit
def anticommutes(a,b):
	'''
	Calculate whether a and b anticommutate
	Args:
		a (array): Array to calculate anticommutator
		b (array): Array to calculate anticommutator
	Returns:
		commutes (bool): Boolean of a and b anticommuting
	'''	
	return bool(~(anticommutator(a,b).any()))


@jit
def trace(a):
	'''
	Calculate trace of array
	Args:
		a (array): Array to calculate trace
	Returns:
		out (array): Trace of array
	'''	
	return np.trace(a)

@jit
def abs(a):
	'''
	Calculate absolute value of array
	Args:
		a (array): Array to calculate absolute value
	Returns:
		out (array): Absolute value of array
	'''	
	return np.abs(a)

@jit
def abs2(a):
	'''
	Calculate absolute value squared of array
	Args:
		a (array): Array to calculate absolute value
	Returns:
		out (array): Absolute value squared of array
	'''	
	return np.abs(a)**2

@jit
def real(a):
	'''
	Calculate real value of array
	Args:
		a (array): Array to calculate real value
	Returns:
		out (array): Real value of array
	'''	
	return a.real


@jit
def imag(a):
	'''
	Calculate imaginary value of array
	Args:
		a (array): Array to calculate imaginary value
	Returns:
		out (array): Imaginary value of array
	'''	
	return a.imag


@jit
def eigh(a):
	'''
	Compute eigenvalues and eigenvectors of a hermitian array
	Args:
		a (array): Array to compute eigenvalues and eigenvectors of shape (...,n,n)
	Returns:
		e (array): Vector of eigenvalues of shape (...,n)
		v (array): Array of normalized eigenvectors of shape (...,n,n)
	'''
	return np.linalg.eigh(a)
	



@jit
def sqrtm(a):
	'''
	Calculate matrix square-root of array a
	Args:
		a (array): Array to compute square root
	Returns:
		out (array): Square root of array
	'''
	return np.sqrtm(a)


@jit
def sqrt(a):
	'''
	Calculate square-root of array a
	Args:
		a (array): Array to compute square root
	Returns:
		out (array): Square root of array
	'''
	return np.sqrt(a)


@jit
def log(a):
	'''
	Calculate natural log of array a
	Args:
		a (array): Array to compute log
	Returns:
		out (array): Natural log of array
	'''
	return np.log(a)

@jit
def exp(a):
	'''
	Calculate element-wise exponential of array a
	Args:
		a (array): Array to compute element-wise exponential
	Returns:
		out (array): Element-wise exponential of array
	'''
	return np.exp(a)

@jit
def _expm(x,A,I):
	'''
	Calculate matrix exponential of parameters times data
	Args:
		x (array): parameters of shape (1,) or (n,) or (n,n)
		A (array): Array of data to matrix exponentiate of shape (n,n)
		I (array): Array of data identity
	Returns:
		out (array): Matrix exponential of A of shape (n,n)
	'''	
	return cosh(x)*I + sinh(x)*A


@jit
def expm(x,A,I):
	'''
	Calculate matrix exponential of parameters times data
	Args:
		x (array): parameters of shape (k,) or (k,n,) or (k,n,n)		
		A (array): Array of data to matrix exponentiate of shape (l,n,n)
		I (array): Array of data identity
	Returns:
		out (array): Matrix exponential of A of shape (n,n)
	'''		
	k = x.shape[0]
	l = A.shape[0]

	def func(i,out):
		return out.dot(_expm(x[i],A[i%l],I))

	return forloop(0,k,func,I)

@jit
def gradient_expm(x,A,I):
	'''
	Calculate gradient of matrix exponential of parameters times data
	Args:
		x (array): parameters of shape (k,) or (k,n,) or (k,n,n)		
		A (array): Array of data to matrix exponentiate of shape (l,n,n)
		I (array): Array of data identity
	Returns:
		out (array): Gradient of matrix exponential of A of shape (k,n,n)
	'''			
	k = x.shape[0]
	l = A.shape[0]

	def func(i,out):
		return out.dot(_expm(x[i],A[i%l],I))

	def grad(i):
		return forloop(i+1,k,func,forloop(0,i+1,func,I).dot(A[i%l]))

	return jax.vmap(grad)(arange(k))


@jit
def expspm(x,A,I):
	'''
	Calculate matrix exponential of parameters times data
	Args:
		x (array): parameters of shape (k,) or (k,n,) or (k,n,n)		
		A (array): Array of data to matrix exponentiate of shape (l,n,n)
		I (array): Array of data identity
	Returns:
		out (array): Matrix exponential of A of shape (n,n)
	'''		
	k = x.shape[0]
	l = A.shape[0]

	def func(i,out):
		return out + x[i]*A[i%l]

	out = zeros(I.shape,dtype=I.dtype)
	return sp.linalg.expm(forloop(0,k,func,out))


@jit
def sin(a):
	'''
	Calculate sine of array a
	Args:
		a (array): Array to compute sine
	Returns:
		out (array): Sine of array
	'''
	return np.sin(a)


@jit
def cos(a):
	'''
	Calculate cosine of array a
	Args:
		a (array): Array to compute cosine
	Returns:
		out (array): Cosine of array
	'''
	return np.cos(a)

@jit
def tan(a):
	'''
	Calculate tan of array a
	Args:
		a (array): Array to compute sine
	Returns:
		out (array): Tan of array
	'''
	return np.tan(a)

@jit
def sinh(a):
	'''
	Calculate sinh of array a
	Args:
		a (array): Array to compute sinh
	Returns:
		out (array): Sinh of array
	'''
	return np.sinh(a)


@jit
def cosh(a):
	'''
	Calculate cosinh of array a
	Args:
		a (array): Array to compute cosinh
	Returns:
		out (array): Cosinh of array
	'''
	return np.cosh(a)

@jit
def tanh(a):
	'''
	Calculate tanh of array a
	Args:
		a (array): Array to compute sine
	Returns:
		out (array): Tanh of array
	'''
	return np.tanh(a)




@jit
def maximum(a,b):
	'''
	Calculate maximum of array a and b
	Args:
		a (array): Array to compute maximum
		b (array): Array to compute maximum
	Returns:
		out (array): Maximum of array a and b
	'''
	return np.maximum(a,b)

@jit
def minimum(a,b):
	'''
	Calculate minimum of array a and b
	Args:
		a (array): Array to compute minimum
		b (array): Array to compute minimum
	Returns:
		out (array): Minimum of array a and b
	'''
	return np.minimum(a,b)


@partial(jit,static_argnums=(1,))
def concatenate(a,axis):
	'''
	Concatenate iterables along axis
	Args:
		a (iterable): Iterables to concatenate
		axis (int): Axis to concatenate arrays
	Returns:
		out (iterable): Concatenation row-wise
	'''
	return np.concatenate(a,axis)


@jit
def diag(a):
	'''
	Get diagonal of array
	Args:
		a (array): Array to get diagonal
	Returns:
		out (array): Diagonal of array
	'''
	return np.diag(a)

@jit
def mod(a,b):
	'''
	Element wise modular division of a mod b
	Args:
		a (array): Array to compare
		b (array): Array to divide by
	Returns:
		out (array): Modular division of a mod b
	'''
	return np.mod(a,b)


# @partial(jit,static_argnums=(1,))
def unique(a,axis=None):
	'''
	Find unique elements of array
	Args:
		a (array): Array to search for unique elements
		axis (int): Axis to search for unique elements
	Returns:
		out (array): Unique elements of array
	'''
	return np.unique(a,axis=axis)

def uniqueobjs(a,axis=None):
	'''
	Find unique elements of array
	Args:
		a (array): Array to search for unique elements
		axis (int): Axis to search for unique elements
	Returns:
		out (array): Unique elements of array
	'''
	return onp.unique(a,axis=axis)


@partial(jit,static_argnums=(1,2,))
def repeat(a,repeats,axis):
	'''
	Repeat array repeats-times along axis
	Concatenate iterables row-wise
	Args:
		a (array): Array to repeat
		repeats (int): Number of times to repeat
		axis (int): Axis along which to repeat
	Returns:
		out (array): Repeated array
	'''
	return np.repeat(a,repeats,axis)


@partial(jit,static_argnums=(1,2,))
def broadcast_to(a,shape):
	'''
	Broadcast array to shape
	Args:
		a (array): Array to broadcast
		shape (iterable): Shape to broadcast to
	Returns:
		out (array): Broadcasted array
	'''
	return np.broadcast_to(a,shape)


def moveaxis(a,source,destination):
	'''
	Move axes of array
	Args:
		a (array): Array to be moved
		source (int,iterable[int]): Initial axes
		destination (int,interable[int]): Final axes
	Returns:
		out (array): Array with moved axes
	'''

	return np.moveaxis(a,source,destination)


def expand_dims(a,axis):
	'''
	Expand axes of array
	Args:
		a (array): Array to be expanded
		axis (int,iterable[int]): Axes to expand to
	Returns:
		out (array): Array with expanded axes
	'''

	return np.expand_dims(a,axis)


@partial(jit,static_argnums=(0,1,2,))
def randomstring(K,N,D=2):
	'''
	Create K random N-qudit Pauli string arrays of shape (K,D**N,D**N)
	Args:
		K (int): Number of Pauli strings
		N (int): Number of qudits
		D (int): Dimension of qudits
	Returns:
		string (array): Pauli string arrays of shap (K,D**N,D**N)
	'''
	
	assert D==2,"Qudits for D=%d > 2 not implemented"%(D)
	
	seed = onp.random.randint(1e6)
	key = PRNGKey(seed)

	d = int(D**2)
	if D == 2:
		basis = array([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
	else:
		basis = array([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])

	alpha = jax.random.uniform(key,(K*N,d))
	
	string = vtensordot(alpha,basis,1)
	string = string.reshape((K,N,D,D))
	string = vtensorprod(string)
	
	return string
	


@partial(jit,static_argnums=(0,1,2,))
def paulistring(string,N,K,D=2):
	'''
	Create K N-qudit Pauli string arrays of shape (K,D**N,D**N)
	Args:
		K (int): Number of Pauli strings
		N (int): Number of qudits
		D (int): Dimension of qudits
	Returns:
		string (array): Pauli string arrays of shap (K,D**N,D**N)
	'''
	
	assert D==2,"Qudits for D=%d > 2 not implemented"%(D)
	
	seed = onp.random.randint(1e6)
	key = PRNGKey(seed)

	d = int(D**2)
	if D == 2:
		basis = array([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
	else:
		basis = array([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])

	alpha = jax.random.uniform(key,(K*N,d))
	
	string = vtensordot(alpha,basis,1)
	string = string.reshape((K,N,D,D))
	string = vtensorprod(string)
	
	return string
	



@partial(jit,static_argnums=(2,3,4,))
def allclose(a,b,rtol=1e-05,atol=1e-08,equal_nan=False):
	'''
	Check if arrays a and b are all close within tolerance
	Args:
		a (array): Array to compare with
		b (array): Array to compare
		rtol (float): Relative tolerance of arrays
		atol (float): Absolute tolerance of arrays
		equal_nan (bool): Compare nan's as equal
	Returns:
		out (bool): Boolean of whether a and b are all close
	'''
	return np.allclose(a,b,rtol,atol,equal_nan)	


@partial(jit,static_argnums=(2,3,4,))
def isclose(a,b,rtol=1e-05,atol=1e-08,equal_nan=False):
	'''
	Check if arrays a and b are close within tolerance
	Args:
		a (array): Array to compare with
		b (array): Array to compare
		rtol (float): Relative tolerance of arrays
		atol (float): Absolute tolerance of arrays
		equal_nan (bool): Compare nan's as equal
	Returns:
		out (array): Boolean of whether a and b are close
	'''
	return np.isclose(a,b,rtol,atol,equal_nan)	



def isiterable(obj):
	'''
	Check if object is iterable
	Args:
		obj (object): object to be tested
	Returns:
		iterable (bool): whether object is iterable
	'''
	return hasattr(obj,'__iter__')


@jit
def isdiag(a):
	'''
	Check if array is diagonal
	Args:
		a (array): Possible diagonal array of shape (n,n)
	Returns:
		iterable (bool): whether object is iterable
	'''	
	n,m = a.shape
	assert n == m
	return ~a.ravel()[:-1].reshape(n-1,m+1)[:,1:].any()


def PRNGKey(seed=None,split=False):
	'''
	Generate PRNG key
	Args:
		seed (int,array): Seed for random number generation or random key for future seeding
	Returns:
		key (array): Random key
		subkey (array): Random array (if split is True)
	'''	
	if seed is None:
		seed = onp.random.randint(1e12)
	if isinstance(seed,(int)):
		key = jax.random.PRNGKey(seed)
	else:
		key = seed
	if split:
		key,subkey = jax.random.split(key)	
		return key,subkey
	else:
		return key



def issparse(a,*args,**kwargs):
	'''
	Check if array is sparse
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is sparse
	'''
	return issparsematrix(a) or issparsearray(a)

def issparsematrix(a,*args,**kwargs):
	'''
	Check if array is sparse matrix
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is sparse matrix
	'''
	return sp.sparse.issparse(a)

def issparsearray(a,*args,**kwargs):
	'''
	Check if array is sparse array
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is sparse array
	'''
	return isinstance(a,sparray.SparseArray)

def isndarray(a,*args,**kwargs):
	'''
	Check if array is numpy array
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is numpy array
	'''
	return isinstance(a,(onp.ndarray))

def isdataframe(a,*args,**kwargs):
	'''
	Check if array is pandas dataframe
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is pandas dataframe
	'''
	return isinstance(a,(pd.DataFrame))

def isarray(a,*args,**kwargs):
	'''
	Check if array is array
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is array
	'''
	return isndarray(a) or issparse(a)

def isscalar(a,*args,**kwargs):
	'''
	Check if array is scalar
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is scalar
	'''
	return (not isarray(a) and not islisttuple(a)) or (isarray(a) and (a.ndim<1) and (a.size<2))

def isnone(a,*args,**kwargs):
	'''
	Check if array is None
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is None
	'''
	return a is None


def isnaninf(a,*args,**kwargs):
	'''
	Check if array is nan or inf
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is nan or inf
	'''
	return isnan(a,*args,**kwargs) or isinf(a,*args,**kwargs)

def isinf(a,*args,**kwargs):
	'''
	Check if array is inf
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is inf
	'''
	return np.isinf(a)

def isnan(a,*args,**kwargs):
	'''
	Check if array is nan
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is nan
	'''
	return np.isnan(a)

def islist(a,*args,**kwargs):
	'''
	Check if array is list
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is list
	'''
	return isinstance(a,(list))

def istuple(a,*args,**kwargs):
	'''
	Check if array is tuple
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is tuple
	'''
	return isinstance(a,(tuple))

def islisttuple(a,*args,**kwargs):
	'''
	Check if array is list or tuple
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is list or tuple
	'''
	return islist(a) or istuple(a)


def flattener(iterable,notiterable=(str,)):
	'''
	Flatten iterable up to layer layers deep into list
	Args:
		iterable (iterable): object to flatten
		notiterable (tuple): object types not to flatten
	Returns:
		flat (generator): generator of flattened iterable
	'''
	for i in iterable:
		if isiterable(i) and not isinstance(i,notiterable):
			for j in flattener(i,notiterable):
				yield j
		else:
			yield i


def flatten(iterable,cls=list,notiterable=(str,),unique=False):
	'''
	Flatten iterable into cls object
	Args:
		iterable (iterable): object to flatten
		cls (class): Class to initialize with flattened iterable
		notiterable (tuple): object types not to flatten		
		unique (bool): Return only unique elements from flattened iterable
	Returns:
		flat (cls): instance of flattened iterable
	'''	
	uniquecls = set if unique else list
	return cls(list(uniquecls(flattener(iterable,notiterable))))	


def parse(string,dtype):
	'''
	Parse string as numerical type
	Args:
		string (str): String to aprse
		dtype (callable): Type to convert to
	Returns:
		parsed (object): Converted string
	'''
	parsed = string
	try:
		parsed = dtype(string)
	except:
		pass
	return parsed


def returnargs(returns):
	'''
	Return tuple of returns, returning first element if tuple and length 1 iterable
	Args:
		returns (iterable): Iterable of return values
	Returns:
		returns (iterable,object): Return values
	'''

	if isinstance(returns,tuple) and len(returns) == 1:
		return returns[0]
	else:
		return returns


def generator(stop=None):
	'''
	Generator wrapper to restart stop number of times
	'''
	def wrap(func):
		def set(*args,**kwargs):
			return func(*args,*kwargs)
		@functools.wraps(func)
		def wrapper(*args,stop=stop,**kwargs):
			generator = set(*args,**kwargs)
			while stop:
				try:
					yield next(generator)
				except StopIteration:
					stop -= 1
					generator = set(*args,**kwargs)
					yield next(generator)
		return wrapper
	return wrap



def copier(key,value,_copy):
	'''
	Copy value based on associated key 

	Args:
		key (string): key associated with value to be copied
		value (python object): data to be copied
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
	Returns:
		Copy of value
	'''

	# Check if _copy is a dictionary and key is in _copy and is True to copy value
	if ((not _copy) or (isinstance(_copy,dict) and (not _copy.get(key)))):
		return value
	else:
		return copy.deepcopy(value)

def permute(dictionary,_copy=False,_groups=None,_ordered=True):
	'''
	Get all combinations of values of dictionary of lists

	Args:
		dictionary (dict): dictionary of keys with lists of values to be combined in all combinations across lists
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		_groups (list,None): List of lists of groups of keys that should not have their values permuted in all combinations, but should be combined in sequence element wise. For example groups = [[key0,key1]], where dictionary[key0] = [value_00,value_01,value_02],dictionary[key1] = [value_10,value_11,value_12], then the permuted dictionary will have key0 and key1 keys with only pairwise values of [{key0:value_00,key1:value_10},{key0:value_01,key1:value_11},{key0:value_02,key1:value_12}].
		_ordered (bool): Boolean on whether to return dictionaries with same ordering of keys as dictionary

	Returns:
		List of dictionaries with all combinations of lists of values in dictionary
	'''		
	def indexer(keys,values,_groups):
		'''
		Get lists of values for each group of keys in _groups
		'''
		_groups = copy.deepcopy(_groups)
		if _groups is not None:
			inds = [[keys.index(k) for k in g if k in keys] for g in _groups]
		else:
			inds = []
			_groups = []
		N = len(_groups)
		_groups.extend([[k] for k in keys if all([k not in g for g in _groups])])
		inds.extend([[keys.index(k) for k in g if k in keys] for g in _groups[N:]])
		values = [[values[j] for j in i ] for i in inds]
		return _groups,values

	def zipper(keys,values,_copy): 
		'''
		Get list of dictionaries with keys, based on list of lists in values, retaining ordering in case of grouped values
		'''
		return [{k:copier(k,u,_copy) for k,u in zip(keys,v)} for v in zip(*values)]

	def unzipper(dictionary):
		'''
		Zip keys of dictionary of list, and values of dictionary as list
		'''
		keys, values = zip(*dictionary.items())	
		return keys,values

	def permuter(dictionaries): 
		'''
		Get all list of dictionaries of all permutations of sub-dictionaries
		'''
		return [{k:d[k] for d in dicts for k in d} for dicts in itertools.product(*dictionaries)]

	def nester(keys,values):
		'''
		Get values of permuted nested dictionaries in values.
		Recurse permute until values are lists and not dictionaries.
		'''
		keys,values = list(keys),list(values)
		for i,(key,value) in enumerate(zip(keys,values)):
			if isinstance(value,dict):
				if isinstance(_groups,dict):
					_group = _groups.get(key,_group)
				else:
					_group = _groups
				values[i] = permute(value,_copy=_copy,_groups=_group)
		return keys,values


	if dictionary in [None,{}]:
		return [{}]

	# Get list of all keys from dictionary, and list of lists of values for each key
	keys,values = unzipper(dictionary)


	# Get values of permuted nested dictionaries in values
	keys,values = nester(keys,values)

	# Retain ordering of keys in dictionary
	keys_ordered = keys
	
	# Get groups of keys based on _groups and get lists of values for each group
	keys,values = indexer(keys,values,_groups)

	# Zip keys with lists of lists in values into list of dictionaries
	dictionaries = [zipper(k,v,_copy) for k,v in zip(keys,values)]


	# Get all permutations of list of dictionaries into one list of dictionaries with all keys
	dictionaries = permuter(dictionaries)


	# Retain original ordering of keys if _ordered is True
	if _ordered:
		for i,d in enumerate(dictionaries):
			dictionaries[i] = {k: dictionaries[i][k] for k in keys_ordered}
	return dictionaries


@partial(jit,static_argnums=(1,2,))
def binary(a,n,function):
	'''
	Perform binary function recursively on n-fold reduction of n arguments f(1(f(2,...f(n-1,n))
	Args:
		a (iterable): iterable of length n of arguments
		n (int): Number of reductions to perform
		function (callable): Binary function with signature function(x,y)
	Returns:
		out (object): Reduced output of binary function
	'''
	if n == 1:
		return a[0]
	elif n == 2:
		return function(a[0],a[1])
	else:
		m = n%2
		n = n//2
		if m:
			return function(a[0],
				function(binary(a[m:n+m],n,function),
				binary(a[n+m:],n,function)))
		else:
			return function(binary(a[m:n+m],n,function),
				binary(a[n+m:],n,function))



@partial(jit,static_argnums=(2,))
def trotter(A,U,p):
	r'''
	Perform p-order trotterization of a matrix exponential U = e^{A} ~ f_p({U_i}) + O(|A|^p)
	where f_p is a function of the matrix exponentials {U_i = e^{A_i}} of the 
	k internally commuting components {A_i} of the matrix A = \sum_i^k A_i .
	For example, for {U_i = e^{A_i}} :
		f_0 = e^{\sum_i A_i}
		f_1 = \prod_i^k U_i
		f_2 = \prod_i^k U_i^{1/2} \prod_k^i U_i^{1/2}
	For p>0, it will be checked if A_i objects have a matrix exponential module for efficient exponentials,
	otherwise the standard expm function will be used.

	Args:
		A (iterable): Array of shape (k,n,n) of k components of a square matrix of shape (n,n) A_i	
		U (iterable): Array of shape (k,n,n) of k components of the matrix exponential of a square matrix of shape (n,n) expm(A_i/p)
		p (int): Order of trotterization p>0
	Returns:
		U (array): Trotterized matrix exponential of shape (n,n)
	'''
	if p == 1:
		U = matmul(U)
	elif p == 2:
		U = matmul(asarray([*U[::1],*U[::-1]]))
	else:
		U = matmul(U)
	return U


@partial(jit,static_argnums=(2,))
def trottergrad(A,U,p):
	r'''
	Perform gradient of p-order trotterization of a matrix exponential U = e^{A} ~ f_p({U_i}) + O(|A|^p)
	where f_p is a function of the matrix exponentials {U_i = e^{A_i}} of the 
	k internally commuting components {A_i} of the matrix A = \sum_i^k A_i .
	For example, for {U_i = e^{A_i}} :
		f_0 = e^{\sum_i A_i}
		f_1 = \prod_i^k U_i
		f_2 = \prod_i^k U_i^{1/2} \prod_k^i U_i^{1/2}
	For p>0, it will be checked if A_i objects have a matrix exponential module for efficient exponentials,
	otherwise the standard expm function will be used.

	Args:
		A (iterable): Array of shape (k,n,n) of k components of a square matrix of shape (n,n) A_i
		U (iterable): Array of shape (k,n,n) of k components of the matrix exponential of a square matrix of shape (n,n) expm(A_i/p)
		p (int): Order of trotterization p>0
	Returns:
		U (array): Gradient of Trotterized matrix exponential of shape (k,n,n)
	'''
	k = len(U)
	if p == 1:
		U = array([matmul(asarray([*U[:i],A[i]/p,*slices(U,i,k-i)])) for i in range(k)])
	elif p == 2:
		U = array([matmul(asarray([*slices(U,0,i)[::1],A[i]/p,*slices(U,i,k-i)[::1],*U[::-1]])) + 
				matmul(asarray([*U[::1],*slices(U,i,k-i)[::-1],A[i]/p,*slices(U,0,i)[::-1]]))
				for i in range(k)])
	else:
		U = array([matmul(asarray([*slices(U,0,i),A[i]/p,*slices(U,i,k-i)])) for i in range(k)])
	return U


def interpolate(x,y,x_new,kind):
	'''
	Interpolate array at new points
	Args:
		x (array): Interpolation points
		y (array): Interpolation values
		x_new (array): New points
		kind (int): Order of interpolation
	Returns:
		out (array): Interpolated values at new points
	'''		
	if y.ndim>1:
		return array([osp.interpolate.interp1d(x,y[:,i],kind)(x_new) for i in range(y.shape[1])]).T
	else:
		return array(osp.interpolate.interp1d(x,y,kind)(x_new))



@jit
def heaviside(a):
	'''
	Calculate heaviside function
	Args:
		a (array): Array to calculate heaviside
	Returns:
		out (array): Heaviside
	'''		
	return np.heaviside(a,0)


@jit
def sigmoid(a,scale=1e4):
	'''
	Calculate sigmoid function with scale
	Args:
		a (array): Array to calculate sigmoid
		scale (float): scale of sigmoid
	Returns:
		out (array): Sigmoid
	'''		
	return (tanh(a*scale/2)+1)/2
	# return sp.special.expit(scale*a)

@jit
def gradient_sigmoid(a,scale=1e4):
	'''
	Calculate gradient of sigmoid function with scale
	Args:
		a (array): Array to calculate sigmoid
		scale (float): scale of sigmoid
	Returns:
		out (array): Gradient of sigmoid
	'''
	return scale*sigmoid(a,scale)*sigmoid(-a,scale)