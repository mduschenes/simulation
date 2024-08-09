#!/usr/bin/env python

import os,sys
import itertools
import jax
import jax.numpy as np
import jax.scipy as sp
import numpy as onp
import scipy as osp

import matplotlib.pyplot as plt


envs = {
	'JAX_PLATFORMS':'cpu',
	'JAX_PLATFORM_NAME':'cpu',
	'TF_CPP_MIN_LOG_LEVEL':5,
	'JAX_ENABLE_X64':True,
}
for var in envs:
	os.environ[var] = str(envs[var])

configs = {
		'jax_disable_jit':False,
		'jax_platforms':'cpu',
		'jax_platform_name':'cpu',
		'jax_enable_x64': True
		}
for name in configs:
	jax.config.update(name,configs[name])


delim = '.'

def gradient(func,mode='fwd',move=True,argnums=0,holomorphic=False,**kwargs):
	'''
	Compute gradient of function
	Args:
		func (callable): Function to differentiate
		mode (str): Type of gradient, allowed strings in ['rev','fwd']
		move (bool): Move differentiated axis to beginning of dimensions
		argnums (int,iterable[int]): Arguments of func to derive with respect to
		holomorphic (bool): Whether function is holomorphic	
		kwargs : Additional keyword arguments
	Returns:
		grad (callable): Gradient of function
	'''

	if mode in ['rev']:
	
		grad = jax.jit(jax.grad(func,argnums=argnums,holomorphic=holomorphic))

	elif mode in ['fwd']:
	
		_grad = jax.jit(jax.jacfwd(func,argnums=argnums,holomorphic=holomorphic))
	
		if move:
			def grad(*args,**kwargs):
				x,args = args[0],args[1:]
				ndim = x.ndim
				return np.moveaxis(_grad(x,*args,**kwargs),range(-1,-ndim-1,-1),range(ndim-1,-1,-1))
		else:
			grad = _grad

		return grad

	return grad

def einsum(subscripts,*operands,optimize=True,wrapper=None):
	'''
	Get optimal summation of axis in array denoted by subscripts
	Args:
		subscripts (str): operations to perform for summation
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)	
	Returns:
		einsummation (callable,array): Optimal einsum operator or array of optimal einsum
	'''

	if wrapper is None:
		wrapper = lambda out,*operands:out

	return wrapper(np.einsum(subscripts,*operands,optimize=optimize),*operands)

def eig(a,compute_v=False,hermitian=False):
	'''
	Compute eigenvalues and eigenvectors
	Args:
		a (array): Array to compute eigenvalues and eigenvectors of shape (...,n,n)
		compute_v (bool): Compute V eigenvectors in addition to eigenvalues
		hermitian (bool): Whether array is Hermitian
	Returns:
		eigenvalues (array): Array of eigenvalues of shape (...,n)
		eigenvectors (array): Array of normalized eigenvectors of shape (...,n,n)
	'''
	if compute_v:
		if hermitian:
			_eig = np.linalg.eigh
		else:
			_eig = sp.linalg.eig
	else:
		if hermitian:
			_eig = np.linalg.eigvalsh
		else:
			_eig = np.linalg.eigvals
	return _eig(a)

def spectrum(func,compute_v=False,hermitian=False):
	'''
	Compute eigenvalues and eigenvectors of a function
	Args:
		func (callable): Function to compute eigenvalues and eigenvectors of shape (...,n,n)
		compute_v (bool): Compute V eigenvectors in addition to eigenvalues
		hermitian (bool): Whether array is Hermitian
	Returns:
		wrapper (callable): Returns:
			eigenvalues (array): Array of eigenvalues of shape (...,n)
			eigenvectors (array): Array of normalized eigenvectors of shape (...,n,n)
	'''

	def wrapper(*args,**kwargs):
		return eig(func(*args,**kwargs),compute_v=compute_v,hermitian=hermitian)

	return wrapper

def rank(obj,hermitian=False,eps=None):
	'''
	Compute rank of array
	Args:
		obj (array,callable): Array to compute rank
		hermitian (bool): Whether array is Hermitian		
		eps (scalar): Tolerance for rank
	Returns:
		rank (int): Rank of array
	'''
	eigenvalues = eig(obj,hermitian=hermitian)

	rank = nonzero(eigenvalues,eps=eps)
	return rank


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
		out = np.kron(out,a[i])
	return out

def conjugate(a):
	'''
	Calculate conjugate of array a
	Args:
		a (array): Array to calculate conjugate
	Returns:
		out (array): Conjugate
	'''	
	return np.conj(a)

def absolute(a):
	'''
	Calculate absolute value of array
	Args:
		a (array): Array to calculate absolute value
	Returns:
		out (array): Absolute value of array
	'''	
	return np.abs(a)

def real(a):
	'''
	Calculate real value of array
	Args:
		a (array): Array to calculate real value
	Returns:
		out (array): Real value of array
	'''	
	return np.real(a)


def imag(a):
	'''
	Calculate imaginary value of array
	Args:
		a (array): Array to calculate imaginary value
	Returns:
		out (array): Imaginary value of array
	'''	
	return np.imag(a)

def epsilon(dtype=float,eps=None):
	'''	
	Get machine precision epsilon for dtype
	Args:
		dtype (data_type): Datatype to get machine precision
		eps (float): Relative machine precision multiplier
	Returns:
		eps (array): Machine precision
	'''
	eps = 1 if eps is None else eps
	eps = eps*np.finfo(dtype).eps

	return eps

def nonzero(a,axis=None,eps=None):
	'''
	Count non-zero elements of array, with eps tolerance
	Args:
		a (array): Array to count non-zero elements
		axis (int,iterable[int]): Axis to compute non-zero elements
		eps (scalar): Epsilon tolerance, defaults to epsilon precision of array dtype
	Returns:
		n (int): Number of non-zero entries
	'''
	eps = epsilon(a.dtype,eps=eps) if eps is None or isinstance(eps,int) else eps
	n = np.count_nonzero(absolute(a)>=eps,axis=axis)
	return n

def datatype(dtype):
	'''
	Get underlying data type of dtype
	Args:
		dtype (str,datatype): Datatype
	Returns:
		dtype (datatype): Underlying datatype
	'''
	
	return real(np.array([],dtype=dtype)).dtype


def fisher(func,grad,**kwargs):
	'''
	Compute fisher information of function
	Args:
		func (callable): Function to compute
		grad (callable): Gradient to compute
		kwargs (dict): Additional keyword arguments
	Returns:
		fisher (callable): Fisher information of function
	'''

	func = spectrum(func,compute_v=True,hermitian=True)

	def fisher(*args,**kwargs):
		
		function = func(*args,**kwargs)
		gradient = grad(*args,**kwargs)

		eigenvalues,eigenvectors = function

		n = eigenvalues.size
		d = nonzero(eigenvalues)
		indices,zeros = slice(n-d,n),slice(0,n-d)

		out = 0

		i,j = indices,indices
		tmp = einsum('ni,unm,mj->uij',conjugate(eigenvectors[:,i]),gradient,eigenvectors[:,j])
		out += einsum('uij,vij,ij->uv',tmp,conjugate(tmp),1/(eigenvalues[i,None] + eigenvalues[None,j]))

		i,j = indices,zeros
		tmp = einsum('ni,unm,mj->uij',conjugate(eigenvectors[:,i]),gradient,eigenvectors[:,j])
		out += 2*real(einsum('uij,vij,ij->uv',tmp,conjugate(tmp),1/(eigenvalues[i,None] + eigenvalues[None,j])))

		out = real(out)

		return out	

	return fisher		




def states(n,d,ndim,string='0',dtype=None,**kwargs):
	'''
	Initialize state
	Args:
		n (int): Number of states
		d (int): Dimension of each state
		ndim (int): Number of dimensions of state
		string (str): Name of state, allowed strings in ['0','+']
		dtype (datatype): Datatype of state
		kwargs (dict): Additional keyword arguments		
	Returns:
		state (array): State
	'''

	size = d**n
	shape = (size,)*ndim

	if string in ['0']:
		state = np.zeros(size,dtype=dtype)
		state = state.at[0].set(1)
	elif string in ['+']:
		state = np.ones(size,dtype=dtype)/np.sqrt(size)
	else:
		raise NotImplementedError("State initialization for '%s' Not Implemented"%(string))
	
	if state.ndim == 1 and ndim == 2:
		state = einsum('i,j->ij',state,conjugate(state))

	return state


def operators(n,d,ndim,m,p,string='U',noise=None,dtype=None,**kwargs):
	'''
	Initialize operator
	Args:
		n (int): Number of states
		d (int): Dimension of each state
		ndim (int): Number of dimensions of operator
		m (int): Depth of operator
		p (float): Amount of noise
		string (str): Name of operator, allowed strings in ['XZZ']
		noise (str): Name of noise, allowed strings in ['dephase']
		dtype (datatype): Datatype of operator
		kwargs (dict): Additional keyword arguments		
	Returns:
		operator (callable): operator
		shape (iterable[int]): Shape of parameters for operator
	'''

	size = d**n
	shape = (size,)*ndim

	I = np.array([[1,0],[0,1]],dtype=dtype)
	X = np.array([[0,1],[1,0]],dtype=dtype)
	Z = np.array([[1,0],[0,-1]],dtype=dtype)

	operators = []
	if string in ['XZZ']:

		for i in range(n):
			operator = tensorprod([X if j in [i] else I for j in range(n)])
			operators.append(operator)

		for i in range(n):
			for j in range(n):
				if not (((i-j) == 1) or ((j==0) and (i==(n-1)))):
					continue
				operator = tensorprod([Z if k in [i,j] else I for k in range(n)])
				operators.append(operator)

		def indexer(parameters,i,j):
			j = int(j >= n)
			return parameters[i,j]

	else:
		raise NotImplementedError("Operators initialization for '%s' Not Implemented"%(string))
	
	noises = []
	if p is None:
		noises = None
	elif noise in ['dephase']:
		for i in range(n):
			noises.append([])

			operator = np.sqrt(p)*Z
			noises[-1].append(operator)

			operator = np.sqrt(1-p)*I
			noises[-1].append(operator)

	else:
		noises = None

	l = len(operators)
	noises = np.array([tensorprod(i) for i in itertools.product(*noises)],dtype=dtype) if noises is not None else None
	identity = tensorprod([I for i in range(n)])


	def operator(parameters,state):
		parameters = parameters.reshape(shape)
		for i in range(m):
			for j in range(l):
				U = np.cos(np.pi/2*indexer(parameters,i,j))*identity + -1j*np.sin(np.pi/2*indexer(parameters,i,j))*operators[j]
				state = einsum('ij,jk,lk->il',U,state,conjugate(U))
			if noises is not None:
				state = einsum('uij,jk,ulk->il',noises,state,conjugate(noises))
		return state


	return operator,shape

def parameterize(shape,bounds,seed,string='rand',dtype=None,**kwargs):
	'''
	Initialize parameters
	Args:	
		shape (interable[int]): Shape of parameters
		bounds (iterable): Bounds of parameters
		seed (int): Seed for random generator
		string (str): Type of initialization, allowed strings in ['rand']
		dtype (datatype): Datatype of parameters
		kwargs (dict): Additional keyword arguments		
	Returns:
		parameters (array): Parameters
	'''
	key = jax.random.PRNGKey(seed)
	dtype = datatype(dtype)

	if string in ['rand']:
		parameters = jax.random.uniform(key,shape,minval=bounds[0],maxval=bounds[1],dtype=dtype)
	else:
		raise NotImplementedError("Parameters initialization for '%s' Not Implemented"%(string))
	
	parameters = parameters.ravel() if parameters is not None else None

	return parameters

def plot(obj,path=None):
	'''
	Plot spectrum of object
	Args:
		obj (array): Array to plot
		path (str): Path to save plot
	'''

	eigenvalues = eig(obj,hermitian=True)
	slices = slice(0,20,1)

	fig,ax = plt.subplots()

	x = np.arange(eigenvalues.size)
	y = np.sort(eigenvalues)[::-1]#/np.max(eigenvalues)

	x = x[slices]
	y = y[slices]

	options = {'marker':'o','linestyle':'--'}

	ax.plot(x,y,**options)
	ax.set_xlabel('Index')
	ax.set_ylabel('Eigenvalue')
	ax.set_yscale('log')
	ax.set_xticks(list(range(slices.start,slices.stop+1,2)))

	if path is not None:
		path = delim.join([*path.split(delim)[:-1],'pdf'])
		fig.savefig(path)

	return


def main(n=2,d=2,m=2,p=None,parameters='rand',operator='XZZ',state='0',noise='dephase',path=None):
	'''
	Simulation
	Args:
		n (int): Size of system
		d (int): Dimension of system
		m (int): Depth of simulation
		p (scalar): Noise in simulation
		parameters (str): Name of parameters, allowed strings in ['rand']
		operator (str): Name of operator, allowed strings in ['XZZ']
		state (str): Name of state, allowed strings in ['0','+']
		noise (str): Name of state, allowed strings in ['dephase']
		path (str): Path to save returns
	Returns:
		returns (dict): Return objects
	'''

	kwargs = {
		'n':n,'d':2,'m':m,'p':p,'path':path,
		'parameters.string':parameters,'operator.string':operator,'state.string':state,'noise.string':noise
		}

	n = 2 if n is None else n
	d = 2 if d is None else d
	m = 2 if m is None else m
	p = None if p is None else p
	parameters = parameters if parameters is not None else 'rand'
	operator = operator if operator is not None else 'XZZ'
	state = state if state is not None else '0'
	noise = noise if noise is not None else None

	dtype = 'complex'

	string = operator
	noise = noise
	ndim = 2
	operator,shape = operators(n=n,d=d,ndim=ndim,m=m,p=p,string=string,noise=noise,dtype=dtype)

	string = state
	ndim = 2	
	state = states(n=n,d=d,ndim=ndim,string=string,dtype=dtype)

	string = parameters
	shape = shape
	bounds = [-1,1]
	seed = 123
	parameters = parameterize(shape=shape,bounds=bounds,seed=seed,string=string,dtype=dtype)


	func = operator
	grad = gradient(operator,mode='fwd')
	metric = fisher(func,grad)


	qfim = metric(parameters,state)
	hermitian = True
	eps = 5e-13
	r = rank(qfim,hermitian=hermitian,eps=eps)

	obj = metric(parameters,state)
	plot(obj,path=path)

	returns = {
		**kwargs,
		'operator':operator(parameters,state),
		'state':state,
		'parameters':parameters.reshape(shape),
		'operator.shape':operator(parameters,state).shape,
		'state.shape':state.shape,
		'parameters.shape':shape,
		'eigenvalues':eig(obj,hermitian=hermitian),
		'rank':r
	}

	if path is not None:
		onp.savez(path,**returns)

	return returns



def init(*args,**kwargs):

	cls = Dict({
		"model":'src.quantum.Noise',
		"state":'src.quantum.State',
		})

	settings = Dict({
		"model":{
			"operator":"depolarize",
			"site":None,
			"string":"noise",
			"parameters":{"data":1,"parameters":1},
			"N":1,"D":2,"ndim":3,
		},
		"state": {
			"operator":"one",
			"site":None,
			"string":"psi",
			"parameters":True,
			"N":1,"D":2,"ndim":2,
			},
	})

	model = load(cls.model)
	state = load(cls.state)

	model = model(**settings.model)
	state = state(**settings.state)

	print(state())
	print()

	model.init(state=state,parameters=dict())

	print(model.data)
	print(model(model.parameters(model.parameters()),model.state()))
	print()


	model.init(state=state,parameters=dict())

	print(model.data)
	print(model(model.parameters(model.parameters()),model.state()))
	print()

	return


if __name__ == '__main__':

	permutations = {
		'n':[2,4],
		'd':[2],
		'm':[5,10],
		'p':[0,1e-3],
		'parameters':['rand'],
		'operator': ['XZZ'],
		'state': ['+'],
		'noise': ['dephase']
	}

	data = []

	for args in itertools.product(*(permutations[kwarg] for kwarg in permutations)):
		
		kwargs = dict(zip(permutations,args))

		defaults = dict(
			path = delim.join(['%s'%('_'.join(['%s%s'%(kwarg,kwargs[kwarg]) for kwarg in kwargs])),'npz'])
		)

		kwargs.update(defaults)
		
		returns = main(**kwargs)
		
		data.append((kwargs,returns))

		path = returns['path']
		returns = np.load(path)

		for attr in returns:
			print(attr,returns[attr])
		print()
