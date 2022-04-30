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
np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'



# Logging
import logging,logging.config
logger = logging.getLogger(__name__)
conf = 'config/logging.conf'
try: 
	logging.config.fileConfig(conf,disable_existing_loggers=False) 
except:
	pass
logger = logging.getLogger(__name__)




def jit(func,*,static_argnums=None):
	return func
	# return jax.jit(func,static_argnums=static_argnums)

def device_put(array,*args,**kwargs):
	return array
	# return jax.device_put(array,*args,**kwargs)

	# return jax.jit(func,static_argnums=static_argnums)

class array(np.ndarray):
	def __new__(self,*args,**kwargs):
		return device_put(np.array(*args,**kwargs))

class ones(array):
	def __new__(self,*args,**kwargs):
		return device_put(np.ones(*args,**kwargs))

class zeros(array):
	def __new__(self,*args,**kwargs):
		return device_put(np.zeros(*args,**kwargs))

class arange(array):
	def __new__(self,*args,**kwargs):
		return device_put(np.arange(*args,**kwargs))

def rand(key,shape,bounds,random='uniform'):
	if random in ['uniform','rand']:
		return jax.random.uniform(key,shape,minval=bounds[0],maxval=bounds[1])
	else:
		return jax.random.uniform(key,shape,minval=bounds[0],maxval=bounds[1])


def gradient(func):
	return jit(jax.grad(func))


# @jit
def matmul(a,b):
	return a.dot(b)


# @jit
def matmul_diagonal(a,b):
	return a*b

@jit
def tensorprod(a,b):
	return np.kron(a,b)


@partial(jit,static_argnums=(2,))
def tensordot(a,b,axes=0):
	return np.tensordot(a,b,axes)

@jit
def multi_tensorprod(a):
	out = a[0]
	for b in a[1:]:
		out = tensorprod(out,b)
	return out

@jit
def multi_matmul(a):
	# n = a.shape[0]
	# while n > 1:
	# 	a = matmul(a[::2,...], a[1::2,...])
	#   n -= 1
	# return a
	return jax.lax.fori_loop(1,len(a),lambda i,out: matmul(out,a[i]),a[0])



@jit
def trace(a):
	return np.trace(a)

@jit
def expm(x,A,I):
	return cos(x)*I -1j*sin(x)*A


@jit
def multi_expm(x,A,I,string):
	m = x.shape[0]
	n = A.shape[0]
	_func = lambda i: expm(x[i],A[i%n],I)
	func = lambda i,out: matmul(out,_func(i))
	
	i = 0
	out = _func(i)

	for i in range(i+1,m):
		out = func(i,out)
		# print(string[i%n],x[i])
	return out
	# return jax.lax.fori_loop(i+1,m,func,out)


def broadcast_to(a,shape):
	return np.broadcast_to(a,shape)

def repeat(a,repeats,axis):
	return np.repeat(a,repeats,axis)

def interpolate(x,y,x_new,kind):
	if y.ndim>1:
		return array([osp.interpolate.interp1d(x,y[:,i],kind)(x_new) for i in range(y.shape[1])]).T
	else:
		return array(osp.interpolate.interp1d(x,y,kind)(x_new))

@jit
def maximum(a,b):
	return np.maximum(a,b)
@jit
def minimum(a,b):
	return np.minimum(a,b)

@jit
def cos(a):
	return np.cos(a)

@jit
def sin(a):
	return np.sin(a)

@jit
def abs(a):
	return np.abs(a)


# @jit
def heaviside(a):
	return np.heaviside(a,0)


# @jit
def sigmoid(a,scale=1e6):
	return np.exp(-np.logaddexp(0, -scale*a))


@jit
def inner(a,b):
	return trace(a.dot(b))
	# return norm(a.dot(b))
	# return (np.abs(a.dot(b))**2).sum()

@partial(jit,static_argnums=(1,2))
def norm(a,axis=None,ord=2):
	return np.linalg.norm(a,axis=axis,ord=ord)
