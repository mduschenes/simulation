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


os.environ['NUMPY_BACKEND'] = 'JAX'


from src.utils import np,onp,backend
from src.utils import jit,partial,vmap
from src.utils import array,zeros,rand,arange,identity,inplace,datatype,allclose,sqrt,abs2,dagger,conjugate,convert
from src.utils import gradient,rand,eye,diag,sin,cos,prod,maximum,minimum
from src.utils import einsum,dot,add,tensorprod,norm,norm2,trace,mse
from src.utils import shuffle,swap,transpose,reshape,contraction
from src.utils import expm,expmv,expmm,expmc,expmvc,expmmn,_expm
from src.utils import gradient_expm
from src.utils import scinotation,delim
from src.utils import arrays,scalars,iterables,integers,floats,pi,asarray,asscalar

from src.optimize import Metric

from src.iterables import getter,setter,sizer,namespace,Dictionary
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

	# b = shuffle(a,axes=axes,shape=shape,transformation=True)

	# b = shuffle(a,axes=axes,shape=shape,transformation=True)

	# b = shuffle(shuffle(a,axes=axes,shape=shape,transformation=True),axes=axes,shape=shape,transformation=False)

	# assert allclose(a,shuffle(shuffle(a,axes=axes,shape=shape,transformation=True),axes=axes,shape=shape,transformation=False)), "Incorrect split and merge axis %r,%r"%(d,s)

	# assert allclose(shuffle(a,axes=axes,shape=shape,transformation=True,execute=False)(a),shuffle(a,axes=axes,shape=shape,transformation=True,execute=False)(a)), "Incorrect split and merge axis %r,%r"%(d,s)


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
		transformation=True,
		) if where is not None else None
	_options = dict(
		axes = [[i] for i in range(n-l)],
		shape = [k,n-l,d],
		transformation=False,
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

	N = 4
	D = 2
	d = 3
	s = 2
	k = 2
	where = [0,2,3]
	samples = [7]
	shape = [3][:d-k]
	L = len(where)
	length = len(samples)
	size = len(shape)
	objs = Dictionary(data=arange(prod(shape)*D**(L*k)),state=arange(prod(samples)*D**(N*s)))
	tensor = Dictionary(N=N,D=D,d=d,s=s)


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
		state = reshape(state,[*samples,[D**N]*s])
		return data,state
	def process(state):
		state = reshape(state,[*samples,[D**N]*s])
		return state
	def func(data,state,where=None,tensor=None):
		def func(data,state):
			if d == 3 and s == 2:
				state = einsum('uij,jk,ulk',data,state,conjugate(data))
			else:
				raise NotImplementedError
			return state
		return func
	data,state = init(objs.data,objs.state)
	func = func(data,state,where=None,tensor=None)
	states[attr] = func(data,state)
	states[attr] = process(states[attr])


	attr = 'nonlocal.nontensor'
	def init(data,state):
		data = reshape(
			transpose(
			reshape(
				tensorprod([reshape(data,[*shape,*[D**L]*k]),reshape(identity(D**(N-L)),[*[1]*(d-k),*[D**(N-L)]*k])]),
				[*shape,*[D]*(N*k)]),
				[*range(size),*[size+N*j+[*where,*sorted(set(range(N))-set(where))].index(i) for j in range(k) for i in range(N)]]),
				[*shape,*[D**N]*k]
			)
		state = reshape(state,[*samples,[D**N]*s])
		return data,state
	def process(state):
		state = reshape(state,[*samples,[D**N]*s])
		return state		
	data,state = init(objs.data,objs.state)
	func = contraction(data,state,where=None,tensor=None)
	states[attr] = func(data,state)
	states[attr] = process(states[attr])


	# attr = 'nonlocal.tensor'
	# def init(data,state):
	# 	data = reshape(
	# 		transpose(
	# 		reshape(
	# 			tensorprod([reshape(data,[*shape,*[D**L]*k]),reshape(identity(D**(N-L)),[*[1]*(d-k),*[D**(N-L)]*k])]),
	# 			[*shape,*[D]*(N*k)]),
	# 			[*range(size),*[size+N*j+[*where,*sorted(set(range(N))-set(where))].index(i) for j in range(k) for i in range(N)]]),
	# 			[*shape,*[D]*(N*k)]
	# 		)
	# 	state = reshape(state,[*samples,[D]*(N*s)])
	# 	return data,state
	# def process(state):
	# 	state = reshape(state,[*samples,[D**N]*s])
	# 	return state
	# data,state = init(objs.data,objs.state)
	# func = contraction(data,state,where=None,tensor=tensor)
	# states[attr] = func(data,state)
	# states[attr] = process(states[attr])


	# attr = 'local.nontensor'
	# def init(data,state):
	# 	data = reshape(data,[*shape,*[D**L]*k])
	# 	state = reshape(state,[*samples,[D**N]*s])
	# 	return data,state
	# def process(state):
	# 	state = reshape(state,[*samples,[D**N]*s])
	# 	return state
	# data,state = init(objs.data,objs.state)
	# func = contraction(data,state,where=where,tensor=None)
	# states[attr] = func(data,state)
	# states[attr] = process(states[attr])


	# attr = 'local.tensor'
	# def init(data,state):
	# 	data = reshape(data,[*shape,*[D]*(L*k)])
	# 	state = reshape(state,[*samples,[D]*(N*s)])
	# 	return data,state
	# def process(state):
	# 	state = reshape(state,[*samples,[D**N]*s])
	# 	return state		
	# data,state = init(objs.data,objs.state)
	# func = contraction(data,state,where=where,tensor=tensor)
	# states[attr] = func(data,state)
	# states[attr] = process(states[attr])


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
				function = lambda state: shuffle(func(shuffle(state,shape=shape,axes=axes,transformation=True)),shape=shape,axes=axes,transformation=False)
			elif state.ndim == 1:
				func = lambda state,data=tmp: einsum('ij,j...->i...',data,state)
				function = lambda state: shuffle(func(shuffle(state,shape=shape,axes=axes,transformation=True)),shape=shape,axes=axes,transformation=False)

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

	class obj(object):
		def __init__(self,where):
			self.where = (*where,) if isinstance(where,iterables) else (where,)
			return

		def __repr__(self):
			return str(self)

		def __str__(self):
			return str(self.where)

	sizes = range(3,8)
	keys = {
		'src.functions.brickwork':lambda N:[*range(0,N-1,2),*range(1,N-1,2)],
		'src.functions.nearestneighbour':lambda N:[*range(0,N-1,1),],
		}

	for N in sizes:
		for key in keys:
			iterable = {index:obj(where) for index,where in enumerate(where for i in keys[key](N) for where in [(i,i+1),(i,),(i+1,)])}

			print(key,N,len(iterable))
			print(iterable)

			iterable = {index: iterable[i] for index,i in enumerate(sortby(iterable,key=key))}

			print(iterable)

			iterable = {index: [iterable[i] for i in group] for index,group in enumerate(groupby(iterable,key=key))}

			print(iterable)

			assert [i for index in iterable for i in iterable[index]] == list(range(3*(N-1)))

			print()


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

	objs['einsum'] = tensor(data=einsum(obj.data,obj.indices,other.data,other.indices),indices=sorted(set(i for i in [*obj.indices,*other.indices] if not (i in obj.indices and i in other.indices)),key=lambda i: (obj.indices.index(i) if i in obj.indices else len(obj.indices),other.indices.index(i) if i in other.indices else len(other.indices))))

	objs['call'] = obj((obj,other))

	objs['and'] = obj & other

	obj &= other

	objs['iand'] = obj

	objs['copy'] = obj.copy(deep=True)

	for i in objs:
		print(objs[i])

	assert all(allclose(objs[i](),objs[j]()) for i in objs for j in objs)

	print('Passed')

	return


def test_network(path=None,tol=None):

	from src.utils import rand,tensor,network,context

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
		with context(*(obj[i] for i in obj),formats=i,indices=indices):
			print(i,obj.indices)
			assert obj.indices == {key:[index.format(i) for index in obj[key].indices] for key in obj}

	print(obj.indices)
	print()


	print('Passed')


	return


if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	# test_getter(path,tol)
	# test_setter(path,tol)
	# test_sizer(path,tol)
	# test_scinotation(path,tol)
	# test_gradient(path,tol)
	# test_gradient_expm(path,tol)
	# test_norm(path,tol)
	# test_expmi()	
	# test_rand(path,tol)
	# test_gradient_expm(path,tol)
	# test_shuffle(path,tol)	
	test_contract(path,tol)
	# test_concatenate(path,tol)
	# test_reshape(path,tol)
	# test_action(path,tol)
	# test_inheritance(path,tol)
	# test_convert(path,tol)
	# test_stability(path,tol)
	# test_seed(path,tol)
	# test_sortby(path,tol)
	# test_sortgroupby(path,tol)
	# test_jax(path,tol)
	# test_tensor(path,tol)
	# test_network(path,tol)