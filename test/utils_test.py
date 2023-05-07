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

from src.io import load,dump,join,split,edit

from src.utils import np,onp,BACKEND
from src.utils import array,zeros,rand,identity,setitem,datatype,allclose,sqrt,abs2
from src.utils import gradient,rand,eye,diag,sin,cos
from src.utils import einsum,norm,norm2,trace,mse
from src.utils import expm,expmv,expmm,expmc,expmvc,expmmn,_expm
from src.utils import gradient_expm
from src.utils import scinotation,delim

from src.optimize import Metric

from src.iterables import getter,setter


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


def test_expmmn(*args,**kwargs):

	def func(*args,**kwargs):
		x,A,I,v,B = kwargs['x'],kwargs['A'],kwargs['I'],kwargs['v'],kwargs['B']
		out = expmmn(x,A,I,v,B)
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


def test_gradient_expm(path=None,tol=None):

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
				out = setitem(out,i,U.dot(out[i]))
				if j == i:
					out = setitem(out,i,A[j%d].dot(out[i]))

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

		out = expmmn(x,A,I,v,B)

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



def test_getter(path=None,tol=None):
	iterables = {'hi':{'world':[[{'goodbye':None}],[{'di':99}]]}}
	
	elements = [
		('hi','world',1,0,'di'),
		('hi','world',4),
		('hi','world',1),        
		('hi','world',1,0),        
	]
	tests = [
		(lambda value,element,iterable: value==99),
		(lambda value,element,iterable: value is None),
		(lambda value,element,iterable: isinstance(value,list)),
		(lambda value,element,iterable: isinstance(value,dict)),
	]
	
	for element,test in zip(elements,tests):
		iterable = iterables
		value = getter(iterable,element)
		assert test(value,element,iterable), "Incorrect getter %r %r"%(element,value)
	
	return

def test_setter(path=None,tol=None):
	iterables = {'hi':{'world':[[{'goodbye':None}],[{'di':99}]]}}
	
	elements = {
		('hi','world',1,4,'di'):-99,
		('hi','world',1,2,0):89,
	}
	
	tests = [
		(lambda value,element,iterable: value==-99),
		(lambda value,element,iterable: value==89),
	]
	
	for element,test in zip(elements,tests):
#         iterable = deepcopy(iterables)
		iterable = iterables
		value = elements[element]
		setter(iterable,{element:value},delimiter=delim,func=True)
		value = getter(iterable,element)
		# print(iterable)
		assert test(value,element,iterable), "Incorrect getter %r %r"%(element,value)
	
	return

def test_scinotation(path=None,tol=None):
	number = 1e5
	_string = '10^{5}'
	kwargs = dict(decimals=1,base=10,order=20,zero=True,one=False,scilimits=[-1,2],error=None,usetex=False)
	string = scinotation(number,**kwargs)
	assert string == _string, "%s != %s"%(string,_string)

	number = 1e5
	_string = '100000'
	kwargs = dict(decimals=1,base=10,order=20,zero=True,one=False,scilimits=[-1,5],error=None,usetex=False)
	string = scinotation(number,**kwargs)
	assert string == _string, "%s != %s"%(string,_string)

	number = 2.1e-5
	_string = r'2.1\cdot10^{-5}'
	kwargs = dict(decimals=2,base=10,order=20,zero=True,one=False,scilimits=[-1,5],error=None,usetex=False)
	string = scinotation(number,**kwargs)
	assert string == _string, "%s != %s"%(string,_string)

	number = 1.1e-5
	_string = r'0.000011'
	kwargs = dict(decimals=7,base=10,order=20,zero=True,one=False,scilimits=[-5,5],error=None,usetex=False)
	string = scinotation(number,**kwargs)
	assert string == _string, "%s != %s"%(string,_string)

	return

def test_gradient(path=None,tol=None):

	if BACKEND in ['autograd']:
		return

	def func(x,y,z):
		x,y,z = sin(z),cos(x),sin(y)
		return x,y
	n = 10
	p = 3
	d = 2

	grad = gradient(func,argnums=range(p),mode='fwd')

	x,y,z = rand(n),rand(n),rand(n)

	g = grad(x,y,z)

	_g = ((zeros(n),zeros(n),diag(cos(z))),(diag(-sin(x)),zeros(n),zeros(n)))

	assert isinstance(g,tuple) and len(g)==d
	assert all(isinstance(h,tuple) and len(h)==p for i,h in enumerate(g))
	assert all(isinstance(k,array) and k.shape == (n,n) and allclose(k,_g[i][j]) for i,h in enumerate(g) for j,k in enumerate(h))

	return

def test_mult(path=None,tol=None):

	m = 5
	n = 3
	a = rand(n)
	b = rand((m,n))

	c = b*a
	d = b.dot(diag(a))

	assert allclose(c,d)

	return


def test_norm(path=None,tol=None):

	n = 10
	a = rand(n)
	b = rand(n)
	c = rand((n,n))

	if c is None:
		subscripts = 'i,i->'  
	elif c.ndim == 1:
		subscripts = 'i,i,i->'
	else:
		subscripts = 'i,j,ij->'
	shapes = (a.shape,b.shape,c.shape if c is not None else None)

	einsummation = mse(*shapes)

	d = einsum(subscripts,a-b,a-b,c)
	if c is None:
		e = ((a-b)*(a-b)).sum()
	elif c.ndim == 1:
		e = ((a-b)*c*(a-b)).sum()
	else:
		e = (a-b).dot(c).dot((a-b))
	f = einsummation(a,b,c)
	h = norm2(a-b,c)

	assert all((allclose(d,e),allclose(d,f),allclose(d,h),allclose(e,f),allclose(e,h))), "norm^2 incorrect"

	return


def test_rand(path,tol):
	from importlib import reload
	import src.utils

	kwargs = [
		{'shape':(4,3),'random':'haar'},
		{'shape':(100,),'random':'normal'},
		{'shape':(2,5,2),'random':'rand'},
		{'shape':(2,5,2),'random':'rand'},
		{'shape':(2,5,2),'random':'rand'},
		]
	seed = 1234
	size = len(kwargs)
	a = [[] for i in range(size)]


	os.environ['NUMPY_BACKEND'] = 'JAX.AUTOGRAD'
	reload(src.utils)
	from src.utils import array,rand,prng,BACKEND
	keys = prng(seed,size=size)
	for i in range(size):
		kwargs[i]['key'] = keys[i]
		a[i].append(rand(**kwargs[i]))

	os.environ['NUMPY_BACKEND'] = 'AUTOGRAD'
	reload(src.utils)
	from src.utils import array,rand,prng,BACKEND
	keys = prng(seed,size=size)
	for i in range(size):
		kwargs[i]['key'] = keys[i]
		a[i].append(rand(**kwargs[i]))

	assert all(allclose(*a[i]) for i in range(size)), "Incorrect Random Initialization"

	return



if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	# test_getter(path,tol)
	# test_setter(path,tol)
	# test_scinotation(path,tol)
	# test_gradient(path,tol)
	# test_gradient_expm(path,tol)
	# test_norm(path,tol)
	# test_expmi()	
	test_rand(path,tol)
