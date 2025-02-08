#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ["",".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

import jax
from jax import jit
import jax.numpy as np
import numpy as onp
from math import prod
from functools import partial

def inplace(obj,index,item,op='set'):
	obj = getattr(obj.at[index],op)(item)
	return obj

def cond(pred,true_fun,false_fun,*operands):
	return jax.lax.cond(pred,true_fun,false_fun,*operands)
	
def forloop(start,end,func,out):
	for i in range(start,end):
		out = func(i,out)
	return out
	return jax.lax.fori_loop(start,end,func,out)

def whileloop(cond,func,out):
#     while cond(out):
#         out = func(out)
#     return out
	return jax.lax.while_loop(cond,func,out)

def array(*args,**kwargs):
	return np.array(*args,**kwargs)

def ones(*args,**kwargs):
	return np.ones(*args,**kwargs)

def zeros(*args,**kwargs):
	return np.zeros(*args,**kwargs)

def arange(*args,**kwargs):
	return np.arange(*args,**kwargs)

def rand(*args,**kwargs):
	return jax.random.randint(**kwargs)

def seeder(seed):
	return jax.random.key(seed)

def svd(a,**kwargs):
	return np.linalg.svd(a,**kwargs)

def trace(a,**kwargs):
	return np.trace(a,**kwargs)

def diag(a,**kwargs):
	return np.diag(a,**kwargs)

def add(a,axis=None):
	return a.sum(axis=axis)

def shape(a,axis=None):
	return a.shape

def reshape(a,shape):
	return np.reshape(a,shape)

def ravel(a):
	return a.ravel()

def astype(a,dtype):
	return a.astype(dtype)

def transpose(a):
	return a.transpose()

def conjugate(a):
	return a.conjugate()

def dagger(a):
	return conjugate(transpose(a))

def dot(a,b):
	return np.dot(a,b)

def norm(a):
	return sqrt(dot(*(ravel(a),)*2))

def sqrt(a):
	return np.sqrt(a)

def absolute(a):
	return np.abs(a)

def maximums(a,b):
	return np.maximum(a,b)

def minimums(a,b):
	return np.minimum(a,b)

def nndsvd(a,u,v,rank=None,eps=None):

	slices = slice(None)
	
	def true(z_positive,x_positive,y_positive,z_negative,x_negative,y_negative):
		return z_positive,x_positive,y_positive

	def false(z_positive,x_positive,y_positive,z_negative,x_negative,y_negative):
		return z_negative,x_negative,y_negative  

	def func(i,x):
		
		s,u,v = x

		
		
		z,x,y = s[i],u[slices,i],v[i,slices]

		x_positive,y_positive = absolute(maximums(x,0)),absolute(maximums(y,0))
		x_negative,y_negative = absolute(minimums(x,0)),absolute(minimums(y,0))
		x_positive_norm,y_positive_norm = norm(x_positive),norm(y_positive)
		x_negative_norm,y_negative_norm = norm(x_negative),norm(y_negative)

		z_positive,z_negative = z*x_positive_norm*y_positive_norm,z*x_negative_norm*y_negative_norm

		x_positive,y_positive = x_positive/x_positive_norm,y_positive/y_positive_norm
		x_negative,y_negative = x_negative/x_negative_norm,y_negative/y_negative_norm

		z,x,y = cond(z_positive>z_negative,true,false,z_positive,x_positive,y_positive,z_negative,x_negative,y_negative)

		s,u,v = inplace(s,i,1),inplace(u,(slices,i),sqrt(z)*x),inplace(v,(i,slices),sqrt(z)*y)
		
		x = s,u,v

		return x

	rank = min(a.shape) if rank is None else rank        
	u,s,v = svd(a,full_matrices=False)
	
	start,end,x = 0,rank,(s,u,v)
	x = forloop(start,end,func,x)
	s,u,v = x
	
	return s,u,v

def nmfd(u,v,rank=None):
	rank = min(a.shape) if rank is None else rank            
	x,y = add(u,0),add(v,1)
	s,u,v = x*y,u*1/x,transpose(transpose(v)*1/y)
	return s,u,v

def nmf(a,u=None,v=None,rank=None,eps=None):
	
	def init(a,u=None,v=None,rank=None,eps=None):
		
		a = a/add(a)
		
		if u is None or v is None:
			s,u,v = nndsvd(a,u=u,v=v,rank=rank,eps=eps)
		else:
			s,u,v = ones(rank),u,v
		
		return a,s,u,v
	
	def run(a,u=None,v=None,rank=None,eps=None):
		if isinstance(eps,int):
			func = update
			start,end,func,x = 0,eps,func,(a,u,v)
			print(start,end)
			x = forloop(start,end,func,x)
			a,u,v = x
		elif isinstance(eps,float):
			cond = lambda x,a=a,eps=eps: error(x,a) > eps
			cond,func,x = cond,update,(a,u,v)
			x = whileloop(cond,func,x)
			a,u,v = x           
			
		s,u,v = nmfd(u,v,rank=rank)
		
		return a,s,u,v
   
	def update(i,x):
		
		a,u,v = x
		
		u,v = (
			(dot(a,transpose(v))/dot(u,dot(v,transpose(v))))*u,
			(dot(transpose(u),a)/dot(dot(transpose(u),u),v))*v
		)

		x = a,u,v
		return x
	
	def error(x,a):
		b,u,v = x
		return norm(a-dot(u,v))
	
	rank = min(a.shape) if rank is None else rank        
	eps = 1e-8 if eps is None else eps

	a,s,u,v = init(a,u=u,v=v,rank=rank,eps=eps)
	a,s,u,v = run(a,u=u,v=v,rank=rank,eps=eps)
	
	return a,s,u,v
	
def nvd(a,**kwargs):
	a,s,u,v = nmf(a,**kwargs)
	return dot(u*s,v)
	
def test_nmf(*args,**kwargs):

	def init(shape,kwargs,dtype=None):
		kwargs.update(dict(
			key=seeder(kwargs.pop('seed',123))
		))
		data = astype(reshape(rand(**kwargs),shape),dtype)
		data = data/trace(data)
		return data

	n = 8
	q = 2
	d = q**n
	i = 1
	k = int((q**(i+1))*(1/2))
	kwargs = dict(
		shape = (d,),
		seed = 1234,
		minval=0,
		maxval=d,
	)
	shape = (q**(i+1),q**(n-i-1))
	dtype = 'complex64'
	a = init(shape,kwargs,dtype=dtype)

	options = dict(
		rank=None,eps=100
	)

	print(a)
	exit()

	b = nvd(a,**options)

	print(norm(a-b))
	
	return


if __name__ == "__main__":

	args = tuple()
	kwargs = dict()

	test_nmf(*args,**kwargs)