#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy,warnings

import jax
import jax.numpy as np
import numpy as onp

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.io import load,dump,join,split,edit

from src.utils import array,zeros,rand,identity,datatype,allclose,sqrt,abs2
from src.utils import norm,trace,inner_abs2
from src.utils import expm,expmv,expmm,expmc,expmvc,expmmc,_expm
from src.utils import gradient_expm

from src.optimize import Metric


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
	# log = file if hasattr(file,'write') else sys.stderr
	# traceback.print_stack(file=log)
	# log.write(warnings.formatwarning(message, category, filename, lineno, line))
	return
warnings.showwarning = warn_with_traceback

def _setup(args,kwargs):
	
	n,m,d,k = kwargs['n'],kwargs['m'],kwargs['d'],kwargs['k']
	
	metric = kwargs['metric']
	
	shape = (n,n)
	key = 123
	dtype = 'complex'
	
	x = rand((m*d,),key=key,dtype=datatype(dtype))
	A = rand((d,*shape),random='hermitian',key=key,dtype=dtype)
	I = identity(shape,dtype=dtype)
	v = rand(shape,key=key,dtype=dtype)
	v /= norm(v,axis=1,ord=2,keepdims=True)
	B = rand((k,*shape),key=key,dtype=dtype)
	a = rand(shape,key=key,dtype=dtype)
	b = rand(shape,key=key+1,dtype=dtype)
	
	shapes = ((kwargs['n'],kwargs['n']),(kwargs['n'],kwargs['n']))
	
	metric = Metric(metric,shapes)
	
	
	updates = {'x':x,'A':A,'I':I,'v':v,'B':B,'a':a,'b':b,'metric':metric}
	
	kwargs.update(updates)
	
	return


def test_expm():

	def func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		out = expm(x,A,I)
		return out

	def _func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		m,d = kwargs['m'],kwargs['d']
		
		out = I
		for i in range(m*d):
			out = _expm(x[i],A[i%d],I).dot(out)

		return out

	args = ()
	kwargs = {}

	kwargs.update({
		'n': 2**2,
		'm': 13,
		'd': 11,
		'k': 6,
		'metric': 'infidelity.abs',
		'time': True,
	})

	_setup(args,kwargs)

	out = func(*args,**kwargs)
	_out = _func(*args,**kwargs)

	assert allclose(out,_out)

	return


		
def test_expmv():

	def func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		v = v[0]
		out = expmv(x,A,I,v)
		return out

	def _func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		v = v[0]
		m,d = kwargs['m'],kwargs['d']
		
		out = v
		for i in range(m*d):
			out = _expm(x[i],A[i%d],I).dot(out)

		return out

	args = ()
	kwargs = {}

	kwargs.update({
		'n': 2**2,
		'm': 13,
		'd': 11,
		'k': 6,
		'metric': 'infidelity.abs',
		'time': True,
	})

	_setup(args,kwargs)

	out = func(*args,**kwargs)
	_out = _func(*args,**kwargs)

	assert allclose(out,_out)

	return


def test_expmm():
	def func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		out = expmm(x,A,I,v)
		return out

	def _func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		m,d = kwargs['m'],kwargs['d']
		
		out = v
		for i in range(m*d):
			U = _expm(x[i],A[i%d],I)
			out = U.dot(out).dot(U.conj().T)

		return out

	args = ()
	kwargs = {}

	kwargs.update({
		'n': 2**2,
		'm': 13,
		'd': 11,
		'k': 6,
		'metric': 'infidelity.abs',
		'time': True,
	})

	_setup(args,kwargs)

	out = func(*args,**kwargs)
	_out = _func(*args,**kwargs)

	assert allclose(out,_out)

	return


def test_expmmc(*args,**kwargs):

	def func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		out = expmmc(x,A,I,v,B)
		return out

	def _func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		m,d,k = kwargs['m'],kwargs['d'],kwargs['k']
		
		out = v
		for i in range(m):
			U = I
			for j in range(d):
				y = x[i*d + j]
				V = _expm(y,A[j%d],I)
				U = V.dot(U)
			out = sum(B[l].dot(U).dot(out).dot(U.conj().T).dot(B[l].conj().T) for l in range(k))

		return out

	args = ()
	kwargs = {}

	kwargs.update({
		'n': 2**2,
		'm': 13,
		'd': 11,
		'k': 6,
		'metric': 'infidelity.abs',
		'time': True,
	})

	_setup(args,kwargs)

	out = func(*args,**kwargs)
	_out = _func(*args,**kwargs)

	assert allclose(out,_out)

	return


def test_gradient_expm():

	def func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		out = gradient_expm(x,A,I)
		return out

	def _func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		m,d = kwargs['m'],kwargs['d']
		
		out = array([I]*(m*d))
			
		for i in range(m*d):
			for j in range(m*d):
				U = _expm(x[j],A[j%d],I)
				out = out.at[i].set(U.dot(out[i]))
				if j == i:
					out = out.at[i].set(A[j%d].dot(out[i]))

		return out

	args = ()
	kwargs = {}

	kwargs.update({
		'n': 2**2,
		'm': 3,
		'd': 2,
		'k': 6,
		'metric': 'infidelity.abs',
		'time': True,
	})

	_setup(args,kwargs)

	out = func(*args,**kwargs)
	_out = _func(*args,**kwargs)

	assert allclose(out,_out)

	return


def test_expmi():

	def func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']

		B = array([I,*[0*I]*(B.shape[0]-1)])

		out = expmmc(x,A,I,v,B)

		return out

	def _func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		
		out = expmm(x,A,I,v)

		return out

	args = ()
	kwargs = {}

	kwargs.update({
		'n': 2**2,
		'm': 13,
		'd': 11,
		'k': 6,
		'metric': 'infidelity.abs',
		'time': True,
	})

	_setup(args,kwargs)

	out = func(*args,**kwargs)
	_out = _func(*args,**kwargs)

	assert allclose(out,_out)

	return