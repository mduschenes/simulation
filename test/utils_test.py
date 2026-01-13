#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,warnings

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','.','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))


os.environ['NUMPY_BACKEND'] = 'JAX'


from src.utils import np,onp,backend
from src.utils import jit,partial,vmap,copy
from src.utils import array,zeros,rand,arange,identity,inplace,datatype,allclose,sqrt,abs2,dagger,conjugate,convert
from src.utils import gradient,rand,eye,diag,sin,cos,prod,maximum,minimum
from src.utils import einsum,dot,add,tensorprod,norm,norm2,trace,mse
from src.utils import shuffle,swap,transpose,reshape,contraction,seeder,slicer
from src.utils import expm,expmv,expmm,expmc,expmvc,expmmn,_expm
from src.utils import gradient_expm
from src.utils import scinotation,delim,choices,samples
from src.utils import arrays,scalars,iterables,integers,floats,pi,asarray,asscalar

from src.optimize import Metric

from src.iterables import getter,setter,popper,updater,sizer,namespace,permuter,Dictionary
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

	default = None
	
	for element,test in zip(elements,tests):
		iterable = iterables
		value = getter(iterable,element,delimiter=delim,default=default)
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

	default = 'replace'
	
	test = lambda value,element,iterable,elements,default=default: value==elements[element]

	for element in elements:
		iterable = copy(iterables)
		setter(iterable,element,delimiter=delim,default=default)
		print(element)
		print(iterables)
		print(iterable)
		print()
		assert all(test(getter(iterable,elem,delimiter=delim),elem,iterable,element) for elem in element), "Incorrect setter %r , %r -> %r"%(iterables,element,iterable)
	
	print('Passed')

	return

def test_popper(path=None,tol=None):
	iterables = {'hi':{'world':{'goodbye':False,'di':99}}}

	elements = [
		'hi.world.di',
		'hi.world',
		'hi'
	]

	default = 'DEFAULT'

	test = lambda value,element,iterable,elements,default=default: value==default

	for element in elements:
		iterable = copy(iterables)
		value = popper(iterable,element,delimiter=delim,default=default)
		print(element,value,getter(iterable,element,delimiter=delim,default=default))
		assert test(getter(iterable,element,delimiter=delim,default=default),element,iterable,elements), "Incorrect setter %r , %r -> %r"%(iterables,element,iterable)

	print('Passed')

	return

def test_updater(path=None,tol=None):
	iterables = {'hi':{'world':{'goodbye':False,'di':99}}}

	elements = [
		{'hi.world.di':'hi.world.new'},
		{'hi.world':'hi.new'},
		{'hi':'new'}
	]

	default = None

	test = lambda value,element,iterable,elements,default=default: value==element

	for element in elements:
		key = list(element)[0]
		iterable = copy(iterables)
		value = popper(iterable,key,delimiter=delim,default=default)
		iterable = copy(iterables)
		updater(iterable,element,delimiter=delim,default=default)
		for i in element:
			print(i,element[i],getter(iterable,element[i].split(delim)[:-1],delimiter=delim,default=default) if element[i].count(delim) else iterable,value,getter(iterable,i,delimiter=delim,default=default))
		assert test(getter(iterable,element[key],delimiter=delim,default=default),value,iterable,elements), "Incorrect setter %r , %r -> %r"%(iterables,element,iterable)

	print('Passed')

	return

def test_sizer(path=None,tol=None):
	iterable = [[[None]*3,[None]*5],[[[None]*6,None],[None]*3,[None]*2]]
	types = (list,)
	exceptions = ()

	size = [2,3,5,6]

	shape = sizer(iterable,types=types,exceptions=exceptions)

	assert all(i==j for i,j in zip(shape,size)), "Incorrect shape %r"%(shape)

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
	_string = r'2.1 \cdot 10^{-5}'
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

		trees = (*copy(trees[:1]),*trees[1:])
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

		trees = copy(trees)

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


def test_shuffle(path=None,tol=None):

	# d = [[2,5],[3,4]]
	# s = [9,1]
	# n = max(len(i) for i in d)
	# k = len(d)+len(s)
	# dtype = None

	# allclose = lambda a,b: all(i==j for i,j in zip(a.ravel(),b.ravel()))

	# shape = (
	# 	*(prod(i) for i in d[:len(d)//2]),
	# 	*s[:len(s)//2],		
	# 	*(prod(i) for i in d[len(d)//2:]),
	# 	*s[len(s)//2:],
	# 	)

	# shape = (
	# 	*s[:len(s)//2],
	# 	*s[len(s)//2:],
	# 	*(prod(i) for i in d[:len(d)//2]),
	# 	*(prod(i) for i in d[len(d)//2:]),
	# 	)

	# size = prod(prod(i) for i in d)*prod(s)
	
	# a = arange(size).reshape(shape)

	# shape = {
	# 	**{axis: d[axis] for axis in range(0,len(d)//2)},
	# 	**{len(d)//2+axis: s[axis] for axis in range(0,len(s)//2)},
	# 	**{len(s)//2+axis: d[axis] for axis in range(len(d)//2,len(d))},		
	# 	**{len(d)+axis: s[axis] for axis in range(len(s)//2,len(s))},
	# 	}

	# shape = {
	# 	**{axis: s[axis] for axis in range(0,len(s)//2)},
	# 	**{axis: s[axis] for axis in range(len(s)//2,len(s))},	
	# 	**{len(s)+axis: d[axis] for axis in range(0,len(d)//2)},
	# 	**{len(s)+axis: d[axis] for axis in range(len(d)//2,len(d))},		
	# 	}

	# axes = ((1,0,n-1),)

	# b = shuffle(a,axes=axes,shape=shape,transform=True)

	# b = shuffle(a,axes=axes,shape=shape,transform=True)

	# b = shuffle(shuffle(a,axes=axes,shape=shape,transform=True),axes=axes,shape=shape,transform=False)

	# assert allclose(a,shuffle(shuffle(a,axes=axes,shape=shape,transform=True),axes=axes,shape=shape,transform=False)), "Incorrect split and merge axis %r,%r"%(d,s)

	# assert allclose(shuffle(a,axes=axes,shape=shape,transform=True,execute=False)(a),shuffle(a,axes=axes,shape=shape,transform=True,execute=False)(a)), "Incorrect split and merge axis %r,%r"%(d,s)


	n = 5
	k = 2
	d = 1
	l = n

	shape = (k**n,)*d
	size = prod(shape)
	ndim = len(shape)

	where = {i:i%k for i in range(l)}

	data = arange(size).reshape(shape)

	options = dict(
		axes = [[i] for i in range(n)],
		shape = [k,n,d],
		transform=True,
		) if where is not None else None
	_options = dict(
		axes = [[i] for i in range(n-l)],
		shape = [k,n-l,d],
		transform=False,
		) if where is not None else None

	function = lambda data: data[tuple(slice(None) if i not in where else where[i] for i in range(n))]

	tmp = shuffle(function(shuffle(data,**options)),**_options)

	assert tmp.size == k**((n-l)*d), "Incorrect data shuffle function"

	print('Passed')

	return

def test_concatenate(path=None,tol=None):
	
	d = [[2,3,4,3],[2,3,4,5],[3,4,2,5]]
	n = max((len(i) if not isinstance(i,integers) else 1 for i in d),default=0)
	k = len(d)
	r = []
	m = max((len(i) if not isinstance(i,integers) else 1 for i in r),default=0)
	q = len(r)
	dtype = "complex"
	
	axis = [1,0,3]
	l = len(axis)

	_axis = [i for i in range(n) if i not in axis]
	_l = len(_axis)

	dimension = {i:[*[d[i][j] for j in axis],*[d[i][j] for j in _axis]] for i in range(k)}
	dimensions = {i:r[i]**(l+_l) for i in range(q)}

	axes = [*axis]
	
	shape = {
		**{i:dimensions[axis] for i,axis in enumerate(dimensions)},
		**{len(dimensions)+i:dimension[axis] for i,axis in enumerate(dimension)},
		}

	print(axes,shape)

	U = [rand(shape=(*r,*(d[j][i] for j in range(k)),),dtype=dtype) for i in axis]
	I = [rand(shape=(*r,*(d[j][i] for j in range(k)),),dtype=dtype) for i in _axis]

	Z = tensorprod((*U,*I))

	W = tensorprod((*(
		U[axis.index(i)] if i in axis else 
		I[_axis.index(i)] if i in _axis else None
		for i in range(n)),)
	)

	V = swap(Z,axes=axes,shape=shape,execute=True)

	assert allclose(V,W), "Incorrect swap V != W"

	print(maximum(abs2(V-W)))

	V = swap(Z,axes=axes,shape=shape,execute=False)



	print(V(Z).shape,W.shape)
	# print(W)	

	assert allclose(V(Z),W), "Incorrect swap V(Z) != W"

	
	print('Passed')
	
	return


def test_contract(path=None,tol=None):

	kwargs = dict(N=[2,3,4,5,6],D=[2],d=[2,3,4],s=[None,1,2],samples=[[3,2]])
	for kwargs in permuter(kwargs):

		N = kwargs['N']
		D = kwargs['D']
		d = kwargs['d']
		s = kwargs['s']
		samples = kwargs['samples']

		k = 2
		l = s
		d = d
		s = 2 if s is None else s

		seed = 123
		dtype = 'complex'
		key = seeder(seed)

		where = [i for i in range(N%2,N,2) if i in range(N)]
		shape = [*[D**2,D]*(d-k)][:d-k]
		L = len(where)
		length = len(samples)
		size = len(shape)
		objs = Dictionary(data=rand(shape=prod(shape)*D**(L*k),key=key,dtype=dtype),state=rand(shape=prod(samples)*D**(N*s),key=key,dtype=dtype))
		attributes = Dictionary(N=N,D=D,d=d,s=s,samples=samples)

		states = {}

		attr = 'func'
		def init(data,state):
			data = reshape(
				transpose(
				reshape(
					tensorprod([reshape(data,[*shape,*[D**L]*k]),reshape(identity(D**(N-L)),[*[1]*(d-k),*[D**(N-L)]*k])]),
					[*shape,*[D]*(N*k)]),
					[*range(size),*[size+N*j+[*where,*sorted(set(range(N))-set(where))].index(i) for j in range(k) for i in range(N)]]),
					[*shape,*[D**N]*k]
				)
			state = reshape(state,[*samples,*[D**N]*s])
			return data,state
		def process(state):
			state = reshape(state,[*samples,*[D**N]*s])
			return state
		def func(data,state,**kwargs):
			def func(data,state):
				if l is not None:
					if d == 4 and s == 2:
						state = einsum('auij,...jk,aulk->...il',data,state,conjugate(data))
					elif d == 3 and s == 2:
						state = einsum('uij,...jk,ulk->...il',data,state,conjugate(data))
					elif d == 2 and s == 2:
						state = einsum('ij,...jk,lk->...il',data,state,conjugate(data))		
					elif d == 4 and s == 1:
						state = einsum('auij,...j->...i',data,state)
					elif d == 3 and s == 1:
						state = einsum('uij,...j->...i',data,state)
					elif d == 2 and s == 1:
						state = einsum('ij,...j->...i',data,state)								
					else:
						raise NotImplementedError
				else:
					if d == 4 and s == 2:
						state = einsum('auij,...jk->...ik',data,state)
					elif d == 3 and s == 2:
						state = einsum('uij,...jk->...ik',data,state)						
					elif d == 2 and s == 2:
						state = einsum('ij,...jk->...ik',data,state)		
					elif d == 4 and s == 1:
						state = einsum('auij,...j->...i',data,state)
					elif d == 3 and s == 1:
						state = einsum('uij,...j->...i',data,state)
					elif d == 2 and s == 1:
						state = einsum('ij,...j->...i',data,state)								
					else:
						raise NotImplementedError					
				return state
			return func
		data,state = init(objs.data,objs.state)
		func = func(data,state,where=where,attributes=attributes,local=False,tensor=False)
		states[attr] = func(data,state)
		states[attr] = process(states[attr])


		attr = 'nonlocal.nontensor'
		def init(data,state):
			data = reshape(
					swap(
					tensorprod([reshape(data,[*shape,*[D**L]*k]),reshape(identity(D**(N-L)),[*[1]*(d-k),*[D**(N-L)]*k])]),
					axes=[[i] for i in where],
					shape={**{i:[shape[i],*[1]*(N-1)] for i in range(d-k)},**{d-k+i:[D]*N for i in range(k)}}
					),
					[*shape,*[D**N]*k]
				)
			state = reshape(state,[*samples,*[D**N]*s])
			return data,state
		def process(state):
			state = reshape(state,[*samples,*[D**N]*s])
			return state		
		data,state = init(objs.data,objs.state)
		func = contraction(data,state if l is not None else None,where=where,attributes=attributes,local=False,tensor=False)
		states[attr] = func(data,state)
		states[attr] = process(states[attr])

		attr = 'nonlocal.tensor'
		def init(data,state):
			data = reshape(
					swap(
					tensorprod([reshape(data,[*shape,*[D**L]*k]),reshape(identity(D**(N-L)),[*[1]*(d-k),*[D**(N-L)]*k])]),
					axes=[[i] for i in where],
					shape={**{i:[shape[i],*[1]*(N-1)] for i in range(d-k)},**{d-k+i:[D]*N for i in range(k)}}
					),
					[*shape,*[D]*(N*k)]
				)
			state = reshape(state,[*samples,*[D]*(N*s)])
			return data,state
		def process(state):
			state = reshape(state,[*samples,*[D**N]*s])
			return state
		data,state = init(objs.data,objs.state)
		func = contraction(data,state if l is not None else None,where=where,attributes=attributes,local=False,tensor=True)
		states[attr] = func(data,state)
		states[attr] = process(states[attr])


		attr = 'local.nontensor'
		def init(data,state):
			data = reshape(data,[*shape,*[D**L]*k])
			state = reshape(state,[*samples,*[D**N]*s])
			return data,state
		def process(state):
			state = reshape(state,[*samples,*[D**N]*s])
			return state
		data,state = init(objs.data,objs.state)
		func = contraction(data,state if l is not None else None,where=where,attributes=attributes,local=True,tensor=False)
		states[attr] = func(data,state)
		states[attr] = process(states[attr])


		attr = 'local.tensor'
		def init(data,state):
			data = reshape(data,[*shape,*[D]*(L*k)])
			state = reshape(state,[*samples,*[D]*(N*s)])
			return data,state
		def process(state):
			state = reshape(state,[*samples,*[D**N]*s])
			return state		
		data,state = init(objs.data,objs.state)
		func = contraction(data,state if l is not None else None,where=where,attributes=attributes,local=True,tensor=True)
		states[attr] = func(data,state)
		states[attr] = process(states[attr])


		print({**kwargs,**dict(shape=shape,where=where)},list(states))

		assert all(allclose(states[i],states[j]) for i in states for j in states)


	print('Passed')


	return

def test_action(path=None,tol=None):

	d = 3
	n = 5
	k = 2

	d = [[d]*n,]*k
	
	n = max((len(i) if not isinstance(i,integers) else 1 for i in d),default=0)
	k = len(d)
	r = []
	m = max((len(i) if not isinstance(i,integers) else 1 for i in r),default=0)
	q = len(r)
	ndim = 2
	index = k-1
	dtype = "complex"
	
	axis = [i for i in [1,0,3,4] if i < n][:n]
	l = len(axis)

	_axis = [i for i in range(n) if i not in axis]
	_l = len(_axis)

	dimension = {i:[*[d[i][j] for j in axis],*[d[i][j] for j in _axis]] for i in range(k)}
	dimensions = {i:r[i]**(l+_l) for i in range(q)}

	U = [rand(shape=(*r,*(d[j][i] for j in range(k)),),dtype=dtype) for i in axis]
	I = [eye(*(d[j][i] for j in range(k)),dtype=dtype) for i in _axis]

	state = rand(shape=(*(prod(d[index]),)*ndim,),dtype=dtype)

	data = {'dense':1,'local':1,'exact':1}

	for attr in data:

		if not data.get(attr):
			continue

		if attr in ['dense']:
			
			axes = [*axis]
			shape = {
				**{i:dimensions[axis] for i,axis in enumerate(dimensions)},
				**{len(dimensions)+i:dimension[axis] for i,axis in enumerate(dimension)},
				}
			# shape = (max(max(i) for i in d),n,ndim)

			tmp = swap(tensorprod((*U,*I)),axes=axes,shape=shape,execute=True)
			
			if state.ndim == 2:
				func = lambda state,data=tmp: einsum('ij,jk,kl->il',data,state,dagger(data))
				function = lambda state: func(state)
			elif state.ndim == 1:
				func = lambda state,data=tmp: einsum('ij,j->i',data,state)
				function = lambda state: func(state)
		
		elif attr in ['local']:

			axes = [axis]
			shape = {
				**{i:d[index] for i in range(ndim)},
				}
			# shape = (max(max(i) for i in d),n,ndim)

			tmp = tensorprod(U)

			if state.ndim == 2:
				func = lambda state,data=tmp: einsum('ij,jk...,kl->il...',data,state,dagger(data))
				function = lambda state: shuffle(func(shuffle(state,shape=shape,axes=axes,transform=True)),shape=shape,axes=axes,transform=False)
			elif state.ndim == 1:
				func = lambda state,data=tmp: einsum('ij,j...->i...',data,state)
				function = lambda state: shuffle(func(shuffle(state,shape=shape,axes=axes,transform=True)),shape=shape,axes=axes,transform=False)

		elif attr in ['exact']:

			axes = None
			shape = None

			tmp = tensorprod((*(
				U[axis.index(i)] if i in axis else 
				I[_axis.index(i)] if i in _axis else None
				for i in range(n)),)
			)

			if state.ndim == 2:
				func = lambda state,data=tmp: einsum('ij,jk...,kl->il...',data,state,dagger(data))
				function = lambda state: func(state)
			elif state.ndim == 1:
				func = lambda state,data=tmp: einsum('ij,j...->i...',data,state)
				function = lambda state: func(state)


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

def test_sortgroupby(path=None,tol=None):

	from src.utils import sortby,groupby

	class Obj(object):
		def __init__(self,**kwargs):
			defaults = dict(where=lambda obj:(*obj,) if isinstance(obj,iterables) else (obj,))
			for kwarg in kwargs:
				setattr(self,kwarg,defaults[kwarg](kwargs[kwarg]) if kwarg in defaults else kwargs[kwarg])
			return

		def __repr__(self):
			return str(self)

		def __str__(self):
			return '-'.join(map(str,self.__dict__.values()))

	sizes = range(3,8)
	keys = {
		'brickwork':{
			'func':(lambda N:
					[obj
					for index in [*range(0,N-1,2),*range(1,N-1,2)]
					for obj in [
					{"where":(index+0,index+1),"unitary":True},
					{"where":(index+0,),"unitary":True},
					{"where":(index+1,),"unitary":True},
					{"where":(index+0,),"unitary":False},
					{"where":(index+1,),"unitary":False}
					]
					]),
			'options':{
				'layout':'brickwork',
				'attribute':[
					{"where":"ij","unitary":True},
					{"where":"i","unitary":True},
					{"where":"j","unitary":True},
					{"where":"i","unitary":False},
					{"where":"j","unitary":False}
					]
				}
			},
		'nearestneighbour':{
			'func':(lambda N:
					[obj
					for index in [*range(0,N-1,1),]
					for obj in [
					{"where":(index+0,index+1),"unitary":True},
					{"where":(index+0,index+1),"unitary":False},
					]
					]),
			'options':{
				'layout':'nearestneighbour',
				'attribute':[
					{"where":"ij","unitary":True},
					{"where":"ij","unitary":False},
					]
				}
			},	
		'local':{
			'func':(lambda N:
					[obj
					for index in [*range(0,N,1),]
					for obj in [
					{"where":(index+0,)},
					]
					]),
			'options':{
				'layout':'local',
				'attribute':[
					{"where":"i"},
					]
				}
			},
		}

	for key in keys:

		func = keys[key].pop('func')
		
		for N in sizes:

			iterable = {index:Obj(**obj) for index,obj in enumerate(func(N)) if obj is not None}
			
			print(key,N,len(iterable))
			print(iterable)

			tmp = copy(iterable)

			iterable = {index:iterable[index] for index in samples(list(iterable),k=len(iterable))}

			iterable = {index: [iterable[i] for i in group] for index,group in enumerate(groupby(iterable,**keys[key]))}

			print(iterable)

			assert all(all(getattr(i,attr)==getattr(j,attr) for attr in j.__dict__) for i,j in zip([i for index in iterable for i in iterable[index]],[tmp[index] for index in tmp]))

	print('Passed')

	return


def test_slicer(path=None,tol=None):

	def equalizer(a,b):
		if not isinstance(a,int) and not isinstance(b,int):
			return all(equalizer(i,j) for i,j in zip(a,b))
		else:
			return a==b

	length = 10
	size = 3
	steps = [1,-1]

	iterable = (i for i in range(length))

	for step in steps:
		key = lambda i,step=step: step*i
		tmp = [[j for j in range(size*i,min(length,size*(i+1)))][::step] for i in range(length//size+((size%length)!=0))][::step]
		assert equalizer(slicer(iterable,size),tmp)

	print('Passed')

	return



def test_reshape(path=None,tol=None):

	# ~transform: (zx,wu,y,v,s) ->
	# 	group.reshape-> (z,x,w,u,y,v,s)
	#  	group.transpose-> (x,u,y,v,z,w,s)
	#  	group.func-> (x,u,y,v,z,w,s)
	#  	split.transpose-> (x,y,z,u,v,w,s)
	#  	split.reshape-> (xyz,uvw,s)
	# 	split.func-> (s,xyz,uvw)

	x,y,z,u,v,w,s = 2,3,5,4,2,7,9

	shape = (s,x*y*z,u*v*w)
	size = prod(shape)
	ndim = len(shape)

	a = reshape(arange(size),shape)

	# transform: (s,xyz,uvw) -> (x,z,w,u,y,v,s) -> (s,zx,) 
	# with shape = {0:s,1:(x,y,z),2:(u,v,w)} and axes=[2,1]

	# 	split.func-> (xyz,uvw,s)
	axes = (1,2,0)
	a = transpose(a,axes)

	# 	split.reshape-> (x,y,z,u,v,w,s)
	shape = (x,y,z,u,v,w,s)
	a = reshape(a,shape)

	#  	split.transpose-> (x,u,y,v,z,w,s)
	axes = (0,3,1,4,2,5,6)
	a = transpose(a,axes)

	#  	group.func-> (x,u,y,v,z,w,s)
	axes = (0,1,2,3,4,5,6)
	a = transpose(a,axes)

	#  	group.transpose-> (z,x,w,u,y,v,s)
	axes = (4,0,5,1,2,3,6)
	a = transpose(a,axes)

	#  	group.reshape-> (z,x,w,u,y,v,s)
	shape = (z*x,w*u,y,v,s)
	a = reshape(a,shape)

	b = reshape(a,shape)
	c = reshape(reshape(a),shape)

	print('Passed')

	return

def test_jax(path=None,tol=None):

	shape = [3,2,4]
	a = rand(shape=shape)

	shape = [2,3,2]
	b = rand(shape=shape)

	kwargs = dict()

	shape = [1,2,3]
	option = rand(shape=shape)

	def func(a,b,**kwargs):
		return a*b.sum()

	options = dict(in_axes=(None,0),out_axes=0)
	func = vmap(func,**options)

	print(func(a,b,**kwargs))

	return


def test_tensor(path=None,tol=None):

	from src.utils import rand,tensor

	shapes = {'x':11,'y':53,'z':29,'u':41}

	indices = ['x','y','z']
	shape = [shapes[i] for i in indices]
	dtype = 'complex128'
	seed = 123

	data = rand(shape,seed=seed,dtype=dtype)
	kwargs = dict(indices=indices)
	obj = tensor(data,**kwargs)

	assert allclose(obj(),data)


	indices = ['x','u','z']
	shape = [shapes[i] for i in indices]
	dtype = 'complex128'
	seed = 123

	data = rand(shape,seed=seed,dtype=dtype)
	kwargs = dict(indices=indices)
	other = tensor(data,**kwargs)


	assert obj.intersection(obj,other) == ['x','z']
	assert obj.union(obj,other) == ['x','y','z','u']
	assert obj.complement(obj,other) == ['y','u']


	objs = {}


	indices = sorted(set(i for i in [*obj.indices,*other.indices] 
			if not (i in obj.indices and i in other.indices)),
			key=lambda i: [*obj.indices,*other.indices].index(i))
	data = einsum(obj.data,obj.indices,other.data,other.indices,indices)
	objs['einsum'] = tensor(data=data,indices=indices)

	objs['call'] = obj((obj,other))

	objs['and'] = obj & other

	obj &= other

	objs['iand'] = obj

	objs['copy'] = obj.copy(deep=True)

	for i in objs:
		print(i,objs[i])

	assert all(allclose(objs[i](),objs[j]()) for i in objs for j in objs)

	print('Passed')

	return


def test_network(path=None,tol=None):

	from src.utils import rand,tensor,network,contexts

	N = 3
	shapes = {'x{}':11,'y{}':5,'z{}':9,'s{}':3,'u{}':14,'v{}':17,'w{}':23,'t{}':8,'q{}':6,'r{}':5}

	indices = [['x{}','u{}','y{}'],['y{}','v{}','z{}'],['z{}','w{}','s{}']]
	shape = [[shapes[j] for j in indices[i]] for i in range(N)]
	dtype = 'complex128'
	seed = 123

	data = {i:rand(shape[i],seed=seed,dtype=dtype) for i in range(N)}
	kwargs = dict(indices=indices)
	obj = network(data,**kwargs)


	assert obj.intersection(obj) == []
	assert obj.union(obj) == ['x{}', 'u{}', 'y{}', 'v{}', 'z{}', 'w{}', 's{}']
	assert obj.complement(obj) == ['x{}', 'u{}', 'v{}', 'w{}', 's{}']

	_indices = ['t{}','u{}','q{}']
	_shape = [shapes[i] for i in _indices]
	dtype = 'complex128'
	seed = 123

	_data = rand(_shape,seed=seed,dtype=dtype)
	kwargs = dict(indices=_indices)
	_obj = tensor(_data,**kwargs)

	tmp = obj & _obj

	for i in obj:
		print(i,obj[i])
	print()
	for i in tmp:
		print(i,tmp[i])
	print()
	
	obj &= _obj

	for i in obj:
		print(i,obj[i])

	tmp = obj.array()

	assert allclose(tmp,einsum('xuy,yvz,zws,tuq->xvwstq',*(data[i] for i in data),_data))

	assert tmp.shape == tuple(shapes[i] for i in ['x{}', 'v{}', 'w{}', 's{}', 't{}','q{}'])


	objs = {}

	objs['obj'] = obj
	objs['copy'] = obj.copy()

	assert all(allclose(objs[i][k](),objs[j][l]()) for i in objs for j in objs for k,l in zip(objs[i],objs[j]))


	N = 3
	shapes = {'x{}':11,'y{}':5,'z{}':9,'s{}':3,'u{}':14,'v{}':17,'w{}':23,'t{}':8,'q{}':6,'r{}':5}

	indices = [['x{}','u{}','y{}'],['y{}','v{}','z{}'],['z{}','w{}','s{}']]
	shape = [[shapes[j] for j in indices[i]] for i in range(N)]
	dtype = 'complex128'
	seed = 123

	data = {i:rand(shape[i],seed=seed,dtype=dtype) for i in range(N)}
	kwargs = dict(indices=indices)
	obj = network(data,**kwargs)

	indices=[{attr:f'_{attr}' for attr in shapes} for i in range(N)]
	attribute = {i:[*obj[i].indices] for i in obj}

	print(obj.indices)

	for i in range(N):
		with contexts(*(obj[i] for i in obj),formats=i,indices=indices):
			print(i,obj.indices)
			assert obj.indices == {key:[index.format(i) for index in obj[key].indices] for key in obj}

	print(obj.indices)
	print()


	print('Passed')


	return


def test_distribution(path=None,tol=None):

	def plot(x,y,xerr=None,yerr=None,fig=None,ax=None,options=None,**kwargs):

		import matplotlib
		import matplotlib.pyplot as plt

		def setup(options):

			options = {} if options is None else options
			for option in options:
				if option in ['color','ecolor']:
					if isinstance(options[option],str):
						value = options[option].split('_') if options[option].count('_') else (options[option],0.5)
						value = getattr(plt.cm,str(value[0]))(float(value[1])) if hasattr(plt.cm,value[0]) else value[0]
				else:
					value = options[option]
				options[option] = value

			settings = {}
			settings['path'] = options.pop('path') if options.get('path') else None
			settings['mplstyle'] = options.pop('mplstyle') if options.get('mplstyle') else 'config/plot.mplstyle'

			return options,settings

		options,settings = setup(options)

		with matplotlib.style.context(settings.get('mplstyle')) if settings.get('mplstyle') else context(settings.get('mplstyle')):

			fig,ax = plt.subplots() if fig is None or ax is None else (fig,ax)

			ax.errorbar(x,y,yerr,xerr,**options)

			ax.set_xlabel(xlabel="$x$",size=45)
			ax.set_ylabel(ylabel="$f(x)$",size=45)

			# ax.set_xscale(value="linear")
			# ax.set_yscale(value="linear")
			# ax.set_xlim(xmin=-0.1,xmax=1.1)
			# ax.set_xticks(ticks=[0,0.2,0.4,0.6,0.8,1])
			# ax.tick_params(**{"axis":"y","which":"major","length":6,"width":1,"pad":10})
			# ax.tick_params(**{"axis":"y","which":"minor","length":4,"width":0})
			# ax.tick_params(**{"axis":"x","which":"major","length":6,"width":1,"pad":10})
			# ax.tick_params(**{"axis":"x","which":"minor","length":4,"width":0})

			ax.set_xscale(value="log",base=4)
			ax.set_yscale(value="log",base=10)
			ax.set_xlim(xmin=2**(-2*23),xmax=2**(1*1))
			ax.set_ylim(ymin=5e-17,ymax=2e8)
			ax.set_xticks(ticks=[2**(-2*i) for i in [22,18,14,10,6,2,0]])
			ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2$' if i not in [0] else '$1$' for i in [22,18,14,10,6,2,0]],size=45)
			ax.set_yticks(ticks=[10**(-i) for i in [16,12,8,4,0,-4,-8]])
			ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [16,12,8,4,0,-4,-8]],size=45)
			ax.tick_params(**{"axis":"y","which":"major","length":6,"width":1,"pad":10})
			ax.tick_params(**{"axis":"y","which":"minor","length":4,"width":0})
			ax.tick_params(**{"axis":"x","which":"major","length":6,"width":1,"pad":10})
			ax.tick_params(**{"axis":"x","which":"minor","length":4,"width":0})


			ax.grid(visible=True)

			handles,labels = ax.get_legend_handles_labels()
			handles,labels = [copy(handle) for handle in handles],[copy(label) for label in labels]
			for handle,label in zip(handles,labels):
				handle[0].set_linewidth(12)

			ax.legend(
				handles,labels,
				title="$\\textrm{Regime}$",
				loc="upper left",
				ncol=1,
				title_fontsize=45,
				prop={"size":45},
				markerscale=6,
				handlelength=4
			)

			if settings.get('path'):
				fig.set_size_inches(w=24,h=24)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname=settings.get('path'))

		return fig,ax

	from src.utils import array,asscalar,meshgrid,linspace,logspace,inplace,partial
	from src.utils import exp,log,log1p
	from src.utils import sqrt,real,nan,is_naninf
	from src.utils import nonzero,unique,sort,minimum,maximum,minimums,maximums
	from src.utils import eig,product,addition,permutations,partitions,products,comb,factorial,multinomial
	from src.quantum import Basis as basis

	from mpmath import exp,log,log1p
	from mpmath import quad as integral,linspace as space,mpmathify

	D = 2
	N = 4
	S = 32

	attr = 'pauli'
	options = dict(D=2)
	data = real(eig(getattr(basis,attr)(**options)))

	options = dict(start=-15,stop=0,num=1000)
	x = logspace(**options)

	def func(x,u,v,p,c,a,b,l,s,d):
		# from mpmath import exp,log,log1p
		x = exp((l*s-1)*log(x) + (((d-l)*s-1)/2)*log1p(-2*a*x + b*x**2) - log(p) - log(c))
		return x

	def function(x,u,v,p,c,a,b,l,s,d):
		from src.utils import exp,log,log1p
		x = exp((l*s-1)*log(x) + (((d-l)*s-1)/2)*log1p(-2*a*x + b*x**2) - log(p) - log(c))
		return x

	y = 0
	options = dict(axis=-1)
	eps = 1e-25
	bounds = space(0,1,100)
	boundaries = [1,0]
	for index in partitions(N,D**2):
		w = multinomial(index)/D**(2*N)
		z = tensorprod([obj for i,j in enumerate(index) for obj in [data[i]]*j])
		u,v = maximums(eps,product(minimum(z,**options))),maximums(eps,product(maximum(z,**options)))
		z = (z-u)/(v-u)
		z = z[z>eps]
		l,s,d = z.size,S,D**N

		a = addition(1/z)/l
		b = addition(1/z**2)/l
		# e = (c*(c+2))/(c*(c+2) + 1),2*(l*s-1)/((d-l)*s-1)
		c = 1
		p = 1
		o = dict(u=u,v=v,p=p,c=c,a=a,b=b,l=l,s=s,d=d)

		o.update({i:asscalar(j) for i,j in o.items()})
		f = partial(func,**o)
		p = integral(f,bounds)
		# p = max(f(asscalar(i)) for i in x)

		# o.update({i:j for i,j in dict(p=p,c=c).items()})
		# f = partial(func,**o)
		# p = integral(f,bounds)

		o.update({i:j for i,j in dict(p=p).items()})
		f = partial(func,**o)
		p = integral(f,bounds)

		w = w*(1/(v-u))*function((x-u)/(v-u),**{i:float(j) for i,j in o.items()})
		w = inplace(w,(x<u)+(x>v)+is_naninf(x),0)

		y += w

		print(index,p)

		boundaries = [min(boundaries[0],u),max(boundaries[-1],v)]

	i = (x>=boundaries[0])*(x<=boundaries[-1])
	x,y = x[i],y[i]


	fig,ax = None,None
	options = dict(
		path='examples/distribution/plot.pdf',mplstyle=None,
		label='$\\textrm{Pauli}$',
		color='viridis_%f'%(0.5),
		marker='',
		linestyle=':',
		markersize=9,
		linewidth=4,
		alpha=0.8,
	)

	fig,ax = plot(x,y,fig=fig,ax=ax,options=options)


	exit()



	s = 1000
	n = 6
	d = 2**n
	x = logspace(-20,0,s)
	a = [2/d,2/d,2,2*d]
	b = [1,1-1/d,8/9+1/d,1-1/d**2]
	c = [1,1/d,1/d,1/d]
	d = [d,1,1,1]
	opts = [
		dict(
			label='$\\# = 2$',
			color='k',
			marker='',
			linestyle='-',
			),
		dict(
			label='$\\frac{l}{d} \\to 0$',
			color='viridis_%f'%(0.25),
			marker='',
			linestyle=':',
			),
		dict(
			label='$\\frac{l}{d} \\to \\frac{1}{2}$',
			color='viridis_%f'%(0.5),
			marker='',
			linestyle=':',
			),
		dict(
			label='$\\frac{l}{d} \\to 1$',
			color='viridis_%f'%(0.75),
			marker='',
			linestyle=':',
			),
		]
	func = lambda x,a,b,c,d: ((x**a)*(1-2*b*((x/c)) + b*((x/c)**2)))**d

	fig,ax = None,None
	options = dict(
		path='examples/distribution/plot.pdf',mplstyle=None,
		markersize=9,
		linewidth=4,
		alpha=0.8,
	)

	for index,(a,b,c,d,opts) in enumerate(zip(a,b,c,d,opts)):

		y = func(x,a,b,c,d)

		fig,ax = plot(x,y,fig=fig,ax=ax,options={**options,**opts})

	exit()

	# d = 2**6
	# s = d//2
	# l = 1
	# c = (l*s-1)/((d-l)*s-1)
	# n = 10
	# a = logspace(log10(2*c),log10(2/c),n)
	# b = logspace(20,0,n)
	# a,b = meshgrid(a,b)
	# i = (a*(a+2)) > (1/(1-b))
	# f = lambda a,b,s: 1 - (b*((a+1)/(a+2))*((2*((((a+1)/(a+2))-1)*(1+s*sqrt(1-((1/b)*((a*(a+2))/((a*(a+2))+1))))))) + ((1/b)*((a+1)/(a+2))*((a*(a+2))/((a*(a+2))+1)))))

	# print(a)
	# print(b)

	# x,y = f(a,b,1),f(a,b,-1)


	# print(i)
	# # x,y = inplace(x,i,nan),inplace(y,i,nan)

	# exit()



	# fig,axes = None,None
	# index = None
	# options = dict(
	# 	path='examples/measurement/plot.pdf',mplstyle=None,
	# 	label='$10^{-%s}$'%((('%e'%(argument['noise.parameters'])).split('e')[-1])[-1]) if argument['noise.parameters'] != 0 else '$0$',
	# 	color='viridis_%f'%((index['noise.parameters']+1)/(len(arguments['noise.parameters'])+1)),alpha=0.8,
	# 	marker='o',linestyle=':',
	# 	markersize=9,
	# 	linewidth=4,
	# 	elinewidth=4,
	# 	capsize=5
	# )


	# data = dict(x=x,y=y)
	# fig,axes = plot(**data,fig=fig,axes=axes,index=index,options=options)





	from src.utils import permute

	settings = dict(
		l = [2],
		n = [4,6,8,10],
		q = [2],
		k = [2,4,8,16,32],
		d = [lambda n,q,k:q**n],
		s = [lambda n,q,k: k+1],
		)


	for setting in permute(settings):

		n = setting['n']
		l = setting['l']
		q = setting['q']
		k = setting['k']
		d = setting['d'](n,q,k)
		s = setting['s'](n,q,k)

		# from src.utils import array,rand
		# from src.utils import eig,nonzero,rank,det,trace,dagger
		# from src.utils import product,addition,exp,log
		# from src.utils import arrays,iterables,real,imag
		# from src.utils import integral,binom,gammaln

		# a = rand(shape=(l),random='rand' if l>1 else 'rand',key=123456789,dtype='float')

		# parse = lambda obj: obj.real if isinstance(obj,arrays) else [parse(i) for i in obj] if isinstance(obj,iterables) else obj
		# constant = float(exp(gammaln(d*s)-gammaln(l*s)-gammaln((d-l)*s))/product(a)**s)
		# parameters = [float(addition(a**(-i))/l) for i in [1,2]]
		# bounds = [float(min(a)) if l==d else 0,float(max(a))]
		# options = dict()

		# func = lambda x: constant*(x**(l*s-1))*((1-2*parameters[0]*x+parameters[1]*x**2)**(((d-l)*s-1)/2))
		# function = integral(func,bounds,**options)

		from math import prod
		from mpmath import log,log1p,exp,sqrt
		from mpmath import quad as integral
		from mpmath import appellf1,gamma,loggamma,binomial
		from mpmath import re

		a = [0,1/(3*d),1/(2*d),1/d][:l+1]
		u,v = min(a),max(a)
		z = [(i-u)/(v-u) for i in a if i>u] if len(a)>1 else a
		l = len(z)

		parameters = [sum(i**(-j) for i in z)/l for j in [0,1,2]]
		parameter = (parameters[1]/parameters[2])*(1+1j*sqrt((parameters[2]/(parameters[1]**2))-1))
		bounds = [0,1]
		options = {}

		func = lambda x: exp((l*s-1)*log(x) + (((d-l)*s-1)/2)*log1p(-2*parameters[1]*x+parameters[2]*x**2))

		function = float(re(integral(func,bounds,**options)))
		# function = appellf1(l*s,-(((d-l)*s)-1)/2,-(((d-l)*s)-1)/2,l*s+1,1/parameter,1/parameter.conjugate())

		strings = dict(
				n=n,l=l,k=k,
				parameters=parameters,
				integral=function
				)
		print(strings)


	return

def test_histogram(path=None,tol=None):

	from src.utils import histogram,bin,point,interval,bounds,linspace,logspace,log,rand,cumsum,difference,addition

	def _bin(bins,range=None,scale=None,base=None,**kwargs):
		if bins is None or isinstance(bins,integers):
			bins = (100 if bins is None else bins)+1
			base = 10 if base is None else base
			if scale is None or scale in ['linear']:
				range = [0,1] if range is None else range
				func,options = linspace,dict()
			elif scale in ['log','symlog']:
				base = 10 if base is None else base
				range = [log(i)/log(base) for i in ([1e-20,1e0] if range is None else range)]
				func,options = logspace,dict(base=base)
			bins = func(*range,bins,**options)
		return bins

	n = 10
	scale = 'log'
	base = 10
	density = 1
	range = [0,1] if scale is None or scale in ['linear'] else [base**(-10),base**(0)] if scale in ['log','symlog'] else None
	kwargs = dict(density=density)

	a = rand(n,bounds=range)

	bins = bin(n,range=range,scale=scale,base=base,**kwargs)

	intervals = (bins[1:]-bins[:-1]) if scale is None or scale in ['linear'] else (bins[1:]-bins[:-1]) if scale in ['log','symlog'] else None

	bound = [range[0],range[-1]]

	x,y = histogram(a,bins=bins,range=range,scale=scale,base=base,**kwargs)

	z = interval(x,range=range,scale=scale,base=base,**kwargs) if density else 1/addition(y)

	u = cumsum(y*z)

	_x,_y = histogram(a,bins=bins,range=range,scale=scale,base=base)

	_z = 1/addition(_y)

	_u = cumsum(_y*_z)

	assert allclose(bins,_bin(n,range=range,scale=scale,base=base,**kwargs))

	assert allclose(x,point(bins,range=range,scale=scale,base=base,**kwargs))

	assert allclose(bins,bin(x,range=range,scale=scale,base=base,**kwargs))

	assert allclose(intervals,interval(x,range=range,scale=scale,base=base,**kwargs))

	assert all(allclose(i,j) for i,j in zip(bound,bounds(x,range=range,scale=scale,base=base,**kwargs)))

	assert allclose(u[-1],1)

	assert allclose(u,_u)

	print('Passed')

	return


if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8
	# test_getter(path,tol)
	# test_setter(path,tol)
	# test_popper(path,tol)
	# test_updater(path,tol)
	# test_sizer(path,tol)
	# test_scinotation(path,tol)
	# test_gradient(path,tol)
	# test_gradient_expm(path,tol)
	# test_norm(path,tol)
	# test_expmi()
	# test_rand(path,tol)
	# test_gradient_expm(path,tol)
	# test_shuffle(path,tol)
	# test_contract(path,tol)
	# test_concatenate(path,tol)
	# test_reshape(path,tol)
	# test_action(path,tol)
	# test_inheritance(path,tol)
	# test_convert(path,tol)
	# test_stability(path,tol)
	# test_seed(path,tol)
	# test_sortby(path,tol)
	# test_sortgroupby(path,tol)
	# test_slicer(path,tol)
	# test_jax(path,tol)
	# test_tensor(path,tol)
	# test_network(path,tol)
	test_distribution(path,tol)
	# test_histogram(path,tol)