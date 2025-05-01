#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy,warnings


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import np,onp,backend
from src.utils import array,gradient,zeros,rand,eye
from src.utils import abs2,trace,dot,allclose
from src.utils import inner,inner_norm,inner_abs2,inner_real
from src.utils import gradient_inner_norm,gradient_inner_abs2,gradient_inner_real

from src.io import load,dump,join,split,edit


def _setup(args,kwargs):
	
	n,d,k = kwargs['n'],kwargs['d'],kwargs['k']
	k = n**d

	shape = (n,n)
	dtype = kwargs['dtype']

	shapes = (shape[:d],shape[:d],(k,*shape[:d]))
	a = rand(shapes[0],dtype=dtype)
	b = rand(shapes[1],dtype=dtype)
	da = rand(shapes[2],dtype=dtype)
	# da = eye(shapes[2][0],dtype=dtype).reshape((*shapes[2][1:],*shapes[2][1:]))

	updates = {'a':a,'b':b,'da':da,'shape':shape,'shapes':shapes}
	
	
	kwargs.update(updates)
	
	return



def test_dot(path=None,tol=None):

	def func(*args,**kwargs):

		a,b,da,d,shape,shapes,optimize,metric = kwargs['a'],kwargs['b'],kwargs['da'],kwargs['d'],kwargs['shape'],kwargs['shapes'],kwargs['optimize'],kwargs['metric']

		boolean = True

		if metric in ['norm']:
			f = lambda *operands,optimize,shape=shape: inner_norm
			g = lambda *operands,optimize,shape=shape: gradient_inner_norm
			if d == 1:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: ((abs2(a-b.conj())).sum()).real)
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: (2*dot(da,(a-b.conj()).T.conj())).real)				
			elif d == 2:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: ((abs2(a-b.conj())).sum()).real)
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: (2*trace(dot(da,(a-b.conj()).T.conj()),axis=(-2,-1))).real)				
				# _g = lambda *operands,optimize,shape=shape: (lambda a,b,da: (2*(da*(a-b.conj()).conj()).sum(axis=[-2,-1])).real)				
			else:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: ((abs2(a-b.conj())).sum()).real)
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: (2*trace(dot(da,(a-b.conj()).T.conj()),axis=(-2,-1))).real)				

		elif metric in ['real']:
			f = lambda *operands,optimize,shape=shape: inner_real
			g = lambda *operands,optimize,shape=shape: gradient_inner_real
			if d == 1:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: ((dot(a,b.T))).real)
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: ((dot(da,b.T))).real)
			elif d == 2:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: (trace(dot(a,b.T))).real)
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: (trace(dot(da,b.T),axis=(-2,-1))).real)
			else:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: (trace(dot(a,b.T))).real)
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: (trace(dot(da,b.T),axis=(-2,-1))).real)
		elif metric in ['abs2']:
			f = lambda *operands,optimize,shape=shape: inner_abs2
			g = lambda *operands,optimize,shape=shape: gradient_inner_abs2            
			if d == 1:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: abs2((dot(a,b.T))))
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: ((2*((dot(a,b.T)).conj()))*dot(da,b.T)).real)
			elif d == 2:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: abs2(trace(dot(a,b.T))))
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: ((2*((trace(dot(a,b.T),axis=(-2,-1))).conj()))*trace(dot(da,b.T),axis=(-2,-1))).real)
			else:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: abs2(trace(dot(a,b.T))))
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: ((2*((trace(dot(a,b.T),axis=(-2,-1))).conj()))*trace(dot(da,b.T),axis=(-2,-1))).real)             
		else:
			f = inner_abs
			g = gradient_inner_abs2
			if d == 1:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: abs2((dot(a,b.T))))
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: ((2*((dot(a,b.T)).conj()))*dot(da,b.T)).real)
			elif d == 2:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: abs2(trace(dot(a,b.T))))
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: ((2*((trace(dot(a,b.T),axis=(-2,-1))).conj()))*trace(dot(da,b.T),axis=(-2,-1))).real)
			else:
				_f = lambda *operands,optimize,shape=shape: (lambda a,b: abs2(trace(dot(a,b.T))))
				_g = lambda *operands,optimize,shape=shape: (lambda a,b,da: ((2*((trace(dot(a,b.T),axis=(-2,-1))).conj()))*trace(dot(da,b.T),axis=(-2,-1))).real)

		func = f(*shapes,optimize=optimize,shape=shape)
		_func = _f(*shapes,optimize=optimize,shape=shape)
		
		out = func(a,b)
		_out = _func(a,b)

		print(metric)
		print(allclose(out,_out))

		boolean &= allclose(out,_out)

		grad = g(*shapes,optimize=optimize,shape=shape)
		_grad = _g(*shapes,optimize=optimize,shape=shape)
		_grad_ = gradient(func)
		
		out = grad(a,b,da)
		_out = _grad(a,b,da)
		_out_ = _grad_(a,b).real
		
		print(allclose(out,_out))
		print()

		boolean &= allclose(out,_out) #and allclose(out,_out_)# and allclose(_out,_out_)

		return boolean

	n = 10
	k = 2*n
	dims = [1,2]
	metrics = [
		'norm',
		'real',
		'abs2'
		]

	for d in dims:
		for metric in metrics:
			
			args = ()
			kwargs = {}
			
			kwargs.update({
					'n': n,
					'd': d,
					'k': k,
					'optimize':None,
					'metric':metric,
					'dtype':'complex',
				})

			_setup(args,kwargs)

			boolean = func(*args,**kwargs)

			assert boolean, "%s [dim = %d] metric function error"%(metric,d)
   

	return

if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	test_dot(path,tol)	
