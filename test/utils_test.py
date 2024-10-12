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


# os.environ['NUMPY_BACKEND'] = 'NUMPY'


from src.utils import np,onp,backend
from src.utils import jit,partial
from src.utils import array,zeros,rand,arange,identity,inplace,datatype,allclose,sqrt,abs2,dagger,conjugate,convert
from src.utils import gradient,rand,eye,diag,sin,cos,prod
from src.utils import einsum,dot,add,tensorprod,norm,norm2,trace,mse
from src.utils import swap,shuffle
from src.utils import expm,expmv,expmm,expmc,expmvc,expmmn,_expm
from src.utils import gradient_expm
from src.utils import scinotation,delim
from src.utils import arrays,scalars,iterables,integers,floats,pi,asarray,asscalar

from src.optimize import Metric

from src.iterables import getter,setter,permutations,namespace
from src.io import load,dump,join,split,edit


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

	print('Passed')

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

	print('Passed')

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

	print('Passed')

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

	print('Passed')

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
				out = inplace(out,i,U.dot(out[i]))
				if j == i:
					out = inplace(out,i,A[j%d].dot(out[i]))

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

	print('Passed')

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

	print('Passed')

	return



def test_getter(path=None,tol=None):
	iterables = {'hi':{'world':{'goodbye':None,'di':99}}}
	
	elements = [
		'hi.world.di',
		'hi.world',
	]
	tests = [
		(lambda value,element,iterable: value==99),
		(lambda value,element,iterable: isinstance(value,dict)),
	]
	
	for element,test in zip(elements,tests):
		iterable = iterables
		value = getter(iterable,element,delimiter=delim)
		assert test(value,element,iterable), "Incorrect getter %r %r"%(element,value)
	
	print('Passed')

	return

def test_setter(path=None,tol=None):
	iterables = {'hi':{'world':{'goodbye':None,'di':99}}}
	
	elements = [
		{'hi.world.di':-99},
		{'hi.world':89},
		{'hi.world':{'check':'new'}},
	]
	
	test = lambda value,element,iterable,elements: value==elements[element]
	
	for element in elements:
		iterable = copy.deepcopy(iterables)
		setter(iterable,element,delimiter=delim,default='replace')
		print(element)
		print(iterables)
		print(iterable)
		print()
		assert all(test(getter(iterable,elem,delimiter=delim),elem,iterable,element) for elem in element), "Incorrect setter %r , %r -> %r"%(iterables,element,iterable)
	
	print('Passed')

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

	print('Passed')

	return

def test_gradient(path=None,tol=None):

	if backend in ['autograd']:
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
	assert all(isinstance(k,arrays) and k.shape == (n,n) and allclose(k,_g[i][j]) for i,h in enumerate(g) for j,k in enumerate(h))

	print('Passed')

	return

def test_mult(path=None,tol=None):

	m = 5
	n = 3
	a = rand(n)
	b = rand((m,n))

	c = b*a
	d = b.dot(diag(a))

	assert allclose(c,d)

	print('Passed')

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

	print('Passed')

	return


def test_rand(path=None,tol=None):
	from importlib import reload
	import src.utils

	backend = os.environ.get('NUMPY_BACKEND',None)

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
	from src.utils import array,rand,seeder,backend
	keys = seeder(seed,size=size)
	for i in range(size):
		kwargs[i]['key'] = keys[i]
		a[i].append(rand(**kwargs[i]))

	os.environ['NUMPY_BACKEND'] = 'AUTOGRAD'
	reload(src.utils)
	from src.utils import array,rand,seeder,backend
	keys = seeder(seed,size=size)
	for i in range(size):
		kwargs[i]['key'] = keys[i]
		a[i].append(rand(**kwargs[i]))

	assert all(allclose(*a[i]) for i in range(size)), "Incorrect Random Initialization"


	os.environ['NUMPY_BACKEND'] = 'JAX'
	reload(src.utils)

	print('Passed')

	return

def test_pytree(path=None,tol=None):

	def tree_map(func,*trees,is_leaf=None,**kwargs):
		'''
		Perform function on trees
		Args:
			func (callable): Callable function with signature func(*trees,**kwargs)
			trees (iterable[pytree]): Pytrees of identical structure for function
			is_leaf (type,iterable[type],callable): Boolean whether tree nodes are leaves
			kwargs (dict): Additional keyword arguments for function
		Returns:
			tree (pytree): Tree with mapped function
		'''
		if not callable(is_leaf):
			types = (dict,tuple,list,) if is_leaf is None else (*is_leaf,) if isinstance(is_leaf,iterables) else (is_leaf,)
			is_leaf = lambda tree,types=types: isinstance(tree,types)	

		if not callable(func):
			return

		def mapper(*trees,func=None,is_leaf=None,**kwargs):
			if not trees:
				return
			tree = trees[0]
			if is_leaf(tree):
				for key in tree:
					node = tree[key] if isinstance(tree,dict) else key
					nodes = (tree[key] if isinstance(tree,dict) else tree for tree in trees)
					if is_leaf(node):
						mapper(*nodes,func=func,is_leaf=is_leaf,**kwargs)
					else:
						leaf = func(*nodes,**kwargs)
						if isinstance(tree,dict):
							tree[key] = leaf
						else:
							tree[tree.index(key)] = leaf
			return

		trees = (*copy.deepcopy(trees[:1]),*trees[1:])
		mapper(*trees,func=func,is_leaf=is_leaf,**kwargs)
		tree = trees[0]

		return tree

	def tree_ravel(tree,is_leaf=None):
		'''
		Flatten tree
		Args:
			tree (pytree): Tree to flatten
			is_leaf (type,iterable[type],callable): Boolean whether tree nodes are leaves
		Yields:
			node (object): Nodes of tree
		'''
		
		if not callable(is_leaf):
			types = (dict,tuple,list,) if is_leaf is None else (*is_leaf,) if isinstance(is_leaf,iterables) else (is_leaf,)
			is_leaf = lambda tree,types=types: isinstance(tree,types)			

		if is_leaf(tree):
			for key in tree:
				node = tree[key] if isinstance(tree,dict) else key
				yield from tree_ravel(node,is_leaf=is_leaf)
		else:
			try:
				yield from tree.ravel()
			except:
				yield tree

	def tree_flatten(tree,is_leaf=None):
		'''
		Flatten tree
		Args:
			tree (pytree): Tree to flatten
			is_leaf (type,iterable[type],callable): Boolean whether tree nodes are leaves
		Returns:
			flat (array): Flattened tree
		'''
		return array([tree for tree in tree_ravel(tree,is_leaf=is_leaf)])

	def tree_func(func):
		'''
		Perform function on trees
		Args:
			func (callable): Callable function with signature func(*trees,**kwargs)
		Returns:
			tree_func (callable): Function that returns tree_map pytree of function call with signature tree_func(*trees,**kwargs)
		'''
		def tree_func(*trees,is_leaf=None,**kwargs):
			return tree_map(partial(func,**kwargs),*trees,is_leaf=is_leaf)
		return tree_func

	@tree_func
	def tree_dot(a,b):
		'''
		Perform dot product function on trees a and b
		Args:
			a (pytree): Pytree object to perform function
			b (pytree): Pytree object to perform function
		Returns:
			tree_map (pytree): Return pytree of function call
		'''	
		return dot(a.ravel(),b.ravel())

	@tree_func
	def tree_add(a,b):
		'''
		Perform add function on trees a and b
		Args:
			a (pytree): Pytree object to perform function
			b (pytree): Pytree object to perform function
		Returns:
			tree_map (pytree): Return pytree of function call
		'''
		return add(a,b)

	@tree_func
	def tree_index(a,index=None):
		'''
		Perform index function on tree a
		Args:
			a (pytree): Pytree object to perform function
			index (object): Index for pytree
		Returns:
			tree_map (pytree): Return pytree of function call
		'''
		return a[index]


	def equals(*trees,**kwargs):
		@tree_func
		def func(*trees):
			assert all(allclose(i,j) for i in trees for j in trees)
			return
		func(*trees,**kwargs)
		return

	def test(*trees,func=None,**kwargs):

		trees = copy.deepcopy(trees)

		default = None
		test = load('src.utils.tree_%s'%(func),default=default)
		check = tree_func(load('src.utils.%s'%(func),default=default))

		assert callable(test) and callable(check), "Incorrect pytree function %r"%(func)

		test = test(*trees,**kwargs)

		check = check(*trees,**kwargs)

		print(test)
		print(check)
		trees = (test,check)
		equals(*trees)

		return


	tree = {'channel':{'x':array([1,2,3],dtype=float),'y':array([1,2,3],dtype=float)},'noise':array([1,2,3],dtype=float)}
	kwargs = dict(index=1)

	print(tree)
	print(tree_dot(tree,tree))
	print(tree_add(tree,tree))
	print(tree_index(tree,**kwargs))

	print(*tree_ravel(tree))
	print(tree_flatten(tree))
	print()

	func = 'dot'
	trees = (tree,tree)
	kwargs = {}
	test(*trees,func=func,**kwargs)

	func = 'add'
	trees = (tree,tree)
	kwargs = {}
	test(*trees,func=func,**kwargs)

	func = 'index'
	trees = (tree,)
	kwargs = {'index':1}
	test(*trees,func=func,**kwargs)
	
	print('Passed')

	return

def test_concatenate(path=None,tol=None):
	d = 2
	n = 5
	q = 2
	l = 3
	k = 2
	shape = (d,n,k)
	axes = [*axis,*(i for i in range(n) if i not in axis)]
	dtype = None

	print(axes)
	
	I = eye(d,dtype=dtype)
	U = [rand(shape=(d,)*k,dtype=dtype) for i in range(l)]

	V = shuffle(tensorprod((tensorprod(U),*(I,)*(n-q-1))),axes=axes,shape=shape)

	W = tensorprod((*(U[axis.index(i)] if i in axis else I for i in range(n)),))

	assert allclose(V,W)
	
	return


def test_reshape(path=None,tol=None):

	d = 2
	n = 3
	k = 2
	# dtype = object
	dtype = None

	string = lambda number,d,n,k: ''.join(map(str,[int((number/(d)**(n*k-1-i))%(d)) for i in range(n*k)]))
	allclose = lambda a,b: all(i==j for i,j in zip(a.ravel(),b.ravel()))

	shape = (d**n,)*k
	size = (d**n)**k
	if size < 1e5:	
		# a = array([string(i,d,n,k) for i in range(size)],dtype=dtype).reshape(shape)
		a = array([i for i in range(size)],dtype=dtype).reshape(shape)
	else:
		a = arange(size).reshape(shape)

	shape = (d,n,k)
	axes = [[1,n-1]]

	b = swap(a,axes=None,shape=shape,transform=True)
	print(a)
	print(b)
	print(a.shape,b.shape)
	print()

	b = swap(a,axes=axes,shape=shape,transform=True)
	print(a)
	print(b)
	print(a.shape,b.shape)
	print()	

	b = swap(swap(a,axes=axes,shape=shape,transform=True),axes=axes,shape=shape,transform=False)
	print(a)
	print(b)
	print(a.shape,b.shape)
	print()

	assert allclose(a,swap(swap(a,axes=axes,shape=shape,transform=True),axes=axes,shape=shape,transform=False)), "Incorrect split and merge axis %d,%d,%d"%(n,d,k)


	assert allclose(swap(a,axes=axes,shape=shape,transform=True,execute=True),swap(a,axes=axes,shape=shape,transform=True,execute=False)(a)), "Incorrect split and merge axis %d,%d,%d"%(n,d,k)


	print('Passed')

	return


def test_action(path=None,tol=None):

	d = 2
	n = 6
	k = 2
	q = 2
	axis = [i for i in [5,0,4,1,3,2] if i<n]
	l = len(axis)
	dtype = int

	I = eye(d,dtype=dtype)
	U = array([[0,1],[1,0]],dtype=dtype)

	shape = (d**n,)*k
	state = rand(shape,seed=123)

	shape = (d**l,)*q
	operator = tensorprod((U,)*l)


	data = {'dense':1,'local':1}

	for attr in data:

		if not data.get(attr):
			continue

		if attr in ['dense']:
			
			shape = (d,n,q)
			axes = [*axis]
		
			tmp = swap(swap(tensorprod((operator,*(I,)*(n-l))),axes=axes,shape=shape,transform=True,permute=True),axes=None,shape=shape,transform=False,permute=True)
			
			if k == 2:
				func = lambda state,data=tmp: einsum('ij,jk,kl->il',data,state,dagger(data))
				function = lambda state: func(state)
			elif k == 1:
				func = lambda state,data=tmp: einsum('ij,j->i',data,state)
				function = lambda state: func(state)
		
		elif attr in ['local']:

			shape = (d,n,k)
			axes = [axis]

			tmp = operator

			if k == 2:
				func = lambda state,data=tmp: einsum('ij,jk...,kl->il...',data,state,dagger(data))
				function = lambda state: swap(func(swap(state,shape=shape,axes=axes,transform=True)),shape=shape,axes=axes,transform=False)
			elif k == 1:
				func = lambda state,data=tmp: einsum('ij,j...->i...',data,state)
				function = lambda state: swap(func(swap(state,shape=shape,axes=axes,transform=True)),shape=shape,axes=axes,transform=False)


		print('---',attr,'---')
		data[attr] = function(state)
		print(data[attr])
		print('-------------------------')


	assert any(not isinstance(data[attr],arrays) for attr in data) or all(allclose(data[i],data[j]) for i in data for j in data if i != j), "Incorrect dot(operator,state)"

	print('Passed')

	return

def test_inheritance(*args,**kwargs):

	class Parent(object):
		attr = 1
		other = 2
		def __init__(self,*args,**kwargs):
			super().__init__(*args,**kwargs)
			return

	class Child(Parent):
		attr = -1
		def __init__(self,*args,**kwargs):
			super().__init__(*args,**kwargs)
			return

	class New(Child):
		pass

	class Factory(Parent):
		def __new__(cls,name,*args,**kwargs):
			if name in ['parent']:
				self = Parent(*args,**kwargs)
			elif name in ['child']:
				self = Child(*args,**kwargs)
			elif name in ['new']:
				self = New(*args,**kwargs)				
			else:
				self = None
			return self

	parent = Parent()
	child = Child()
	test = Factory(name='new')

	print(Parent,parent.__class__,namespace(Parent,parent))
	print(Child,parent.__class__,namespace(Child,parent))
	print(Factory,test.__class__,namespace(New,parent))

	return


def test_convert(*args,**kwargs):

	def check(iterable,type):
		if isinstance(iterable,iterables):
			assert isinstance(iterable,type), "Incorrect type %r != <%r>"%(iterable,type)
			for i in iterable:
				check(i,type=type)
		return

	iterable = array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
	type = list
	types = (*arrays,)
	default = asscalar

	print(iterable)
	iterable = convert(iterable,type=type,types=types,default=default)
	print(iterable)

	check(iterable,type=type)

	return


def test_stability(*args,**kwargs):

	def sqrtm(n,hermitian=False):
		import jax
		import jax.numpy as np
		import jax.scipy as sp

		ndim = 2
		shape = (n,)*ndim
		dtype = 'complex'
		
		random = 'uniform'
		seed = 123
		key = jax.random.key(seed)
		rand = getattr(jax.random,random)

		if dtype in ['complex']:
			key_real,key_imag = jax.random.split(key)
			array = rand(key_real,shape) + 1j*rand(key_imag,shape)
			array = rand(key_real,(n,)) + 1j*rand(key_imag,(n,))
			array = np.outer(array,array.conj())
		else:
			array = rand(key,shape)
		array = (array + array.T.conj())/2 if hermitian else array
		array = array/np.trace(array)

		eig = np.linalg.eigh if hermitian else np.linalg.eig
		eigenvalues,eigenvectors = eig(array)
		eigenvalues,eigenvectors = eigenvalues.astype(dtype),eigenvectors.astype(dtype)

		sqrteigm = dot(eigenvectors*sqrt(eigenvalues),eigenvectors.T.conj())

		sqrtm = sp.linalg.sqrtm(array)

		assert allclose(trace(dot(sqrteigm,sqrteigm)),trace(dot(sqrtm,sqrtm))), "Incorrect sqrtm %s"%(np.linalg.norm(sqrteigm-sqrtm)/np.sqrt(np.linalg.norm(sqrteigm)*np.linalg.norm(sqrtm)))

		return sqrteigm

	n = 2**10
	hermitian = True
	sqrtm(n,hermitian=hermitian)

	return


def test_seed(path=None,tol=None):

	from src.utils import jax,rand,seeder

	seed = 213214
	size = None
	splits = True
	data = True
	shape = (3,4)

	key = seeder(seed=seed,size=size,split=split,data=data)

	for i in range(splits):
		key = seeder(seed=seed,split=splits)

	a = rand(shape=shape,key=key)

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
	# test_rand(path,tol)
	# test_gradient_expm(path,tol)
	# test_reshape(path,tol)
	# test_action(path,tol)
	# test_inheritance(path,tol)
	# test_convert(path,tol)
	# test_stability(path,tol)
	# test_concatenate(path,tol)
	test_seed(path,tol)