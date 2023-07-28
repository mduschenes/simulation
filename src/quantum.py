#!/usr/bin/env python

# Import python modules
import os,sys
from copy import deepcopy
from functools import partial
import traceback

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,vmap,vfunc,switch,forloop,cond,slicing,gradient,hessian,fisher
from src.utils import array,asarray,asscalar,empty,identity,ones,zeros,rand,prng,spawn,arange,diag
from src.utils import contraction,gradient_contraction
from src.utils import tensorprod,conjugate,dagger,einsum,dot,norm,eig,trace,sort,relsort,prod
from src.utils import inplace,insert,maximum,minimum,argmax,argmin,nonzero,difference,unique,cumsum,shift,interleave,abs,mod,sqrt,log,log10,sign,sin,cos,exp
from src.utils import to_index,to_position,to_string,allclose
from src.utils import pi,e,nan,null,delim,scalars,arrays,nulls,datatype

from src.iterables import setter,getattrs,hasattrs,namespace,permutations

from src.io import load,dump,join,split

from src.system import Dictionary,System,Space,Time,Lattice

from src.parameters import Parameters

from src.optimize import Objective,Metric

N = 1
D = 2
ndim = 2
random = 'haar'
dtype = 'complex'
Basis = {
	'I': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[1,0],[0,1]],dtype=dtype)),
	'X': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[0,1],[1,0]],dtype=dtype)),
	'Y': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[0,-1j],[1j,0]],dtype=dtype)),
	'Z': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[1,0],[0,-1]],dtype=dtype)),
	'CNOT': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=dtype)),
	'HADAMARD': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[1,1,],[1,-1]],dtype=dtype)/sqrt(D)),
	'PHASE': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[1,0,],[0,1j]],dtype=dtype)),
	'TOFFOLI': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
											  [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]],dtype=dtype)),
	'RANDOM': (lambda *args,N=N,D=D,ndim=ndim,random=random,dtype=dtype,**kwargs: rand(shape=(D,)*ndim,random=random,dtype=dtype)),
	'00': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[1,0],[0,0]],dtype=dtype)),
	'01': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[0,1],[0,0]],dtype=dtype)),
	'10': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[0,0],[1,0]],dtype=dtype)),
	'11': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[0,0],[0,1]],dtype=dtype)),
	'0': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([1,0],dtype=dtype)),
	'1': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([0,1],dtype=dtype)),
	'+': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([1,1,],dtype=dtype)/sqrt(D)),
	'-': (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([1,-1,],dtype=dtype)/sqrt(D)),
}

def trotter(iterable=None,p=None,shape=None):
	'''
	Trotterized iterable for order p or coefficients for order p
	Args:
		iterable (iterable): Iterable
		p (int): Order of trotterization
		shape (iterable[int]): Shape of iterable
	Returns:
		iterables (iterable): Trotterized iterable for order p
		coefficients (scalar): Coefficients for order p
	'''

	P = 2
	if isinstance(p,int) and (p > P):
		raise NotImplementedError("p = %r > %d Not Implemented"%(p,P))

	if shape is None:
		shape = (len(iterable),) if iterable is not None else (1,)
	

	if iterable is None:
		coefficients = 1/p if isinstance(p,int) else None
		return coefficients

	elif not isinstance(iterable,dict):
		slices = []
		tmp = None
		for index in range(p):
			if index == 0:
				indices = slice(None,None,1)
			elif index == 1:
				indices = slice(None,None,-1)
			slices.append(indices)

		iterables = []        
		for indices in slices[:p]:
			iterables += iterable[indices]

	elif all(isinstance(i,int) and isinstance(iterable[i],int) for i in iterable) and all(i < prod(shape) for i in iterable):
		slices = []
		tmp = (p*shape[0],*shape[1:]) 
		for index in range(p):
			indices = {}
			for i in iterable:
				k = iterable[i]
				j = to_position(i,shape[::-1])[::-1]
				if index == 0:
					j = to_index((j[0],*j[1:])[::-1],tmp[::-1])
				elif index == 1:
					j = to_index((tmp[0]-1-j[0],*j[1:])[::-1],tmp[::-1])

				indices[j] = k

			slices.append(indices)

		iterables = {}
		for indices in slices[:p]:
			iterables.update(indices)

	elif all(isinstance(i,int) and isinstance(iterable[i],int) for i in iterable) and not all(i < prod(shape) for i in iterable):
		iterables = iterable
	else:

		raise NotImplementedError("TODO: Trotterization for %r not implemented"%(iterable))

	return iterables



def scheme(parameters,state=None,label=None,conj=False,data=None,identity=None,indices=None):
	'''
	Contract data and state
	Args:
		parameters (array): parameters of shape (size,)
		state (array): state of shape (n,) or (n,n)
		label (array): state of shape (n,) or (n,n)
		conj (bool): conjugate
		data (array): data of shape (length,)
		identity (array): Array of data identity of shape (n,n)
	Returns:
		func (callable): contracted data(parameters) and state with signature func(parameters,state)
	'''

	size = parameters.shape[0] if parameters is not None else 1
	length = len(data) if data is not None else 1

	def sorter(i,size):
		return conj*(size-1) + (1-2*conj)*(i%size)

	obj = data
	step = 1
	indices = (0,size//step)

	def function(parameters,state=state,indices=indices):
		obj = switch(sorter(indices,length),data,parameters[sorter(indices,size)],state)
		return obj

	def func(parameters,state=state,indices=indices):

		state = state if state is not None else identity
		out = state
	
		def func(i,out):
			return function(parameters,out,indices=i)

		indexes = indices

		return forloop(*indexes,func,out)

	func = jit(func)
	return func			


def gradient_scheme(parameters,state=None,conj=False,data=None,identity=None,grad=None,indices=None):
	'''
	Contract data and state
	Args:
		parameters (array): parameters of shape (size,)
		state (array): state of shape (n,) or (n,n)
		conj (bool): conjugate
		data (array): data of shape (length,)
		identity (array): Array of data identity of shape (n,n)
		grad (array): data of shape (length,)
		indices (dict,iterable[int]): indices to compute gradients with respect to {index_parameter: index_gradient}
	Returns:
		func (callable): contracted data(parameters) and state with signature func(parameters,state)
	'''

	size = parameters.shape[0]
	length = len(data)

	if indices is None:
		indices = {i:i for j,i in enumerate(range(size))}
	elif isinstance(indices,dict):
		indices = {i:indices[i] for j,i in enumerate(indices)}
	else:
		indices = {i:j for j,i in enumerate(indices)}

	indices,tmp = list(set(indices[i] for i in indices)),indices

	indexer = {}
	for i in tmp:
		if tmp[i] not in indexer:
			indexer[tmp[i]] = [i]
		else:
			indexer[tmp[i]].append(i)
	indexer = [indexer[i] for i in indexer]

	sizes = [0]
	for i in range(len(indexer)):
		sizes.append(len(indexer[i]))
	sizes = cumsum(sizes)

	indices,indexer,sizes = (
		array(indices).reshape(-1),
		array([j for i in indexer for j in i]).reshape(-1),
		array(sizes).reshape(-1)
		)

	if grad is None:
		grad = [(lambda parameters,state=None,i=i: data[i].coefficients*data[i](parameters + (pi/2)/data[i].coefficients)) for i in range(length)]
	
	step = length

	function = scheme(parameters=parameters,state=state,data=data,identity=identity)
	
	def sorter(i,size):
		return conj*(size-1) + (1-2*conj)*(i%size)

	def gradient(parameters,state=state,indices=indices):
		obj = switch(sorter(indices,step),grad,parameters[sorter(indices,size)],state)
		return obj
	

	def func(parameters,state=state,indices=indices):

		def func(i,out):

			def func(j,out):

				j = indexer[sizes[indices[i]]+j]

				obj = state

				obj = function(parameters,obj,indices=(0,(j)//step))

				obj = gradient(parameters,obj,indices=j)

				obj = function(parameters,obj,indices=((j+1)//step,size//step))

				out = inplace(out,i,obj,'add')

				return out

			indexes = [0,sizes[indices[i]+1]-sizes[indices[i]]]

			return forloop(*indexes,func,out)

		state = state if state is not None else identity
		indexes = [0,len(indices)]
		out = zeros((len(indices),*state.shape),dtype=identity.dtype)
		
		return forloop(*indexes,func,out)

	func = jit(func)

	return func			



class Object(System):
	'''
	Base class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		None: (lambda *args,N=N,D=D,dtype=dtype,**kwargs: array([[1,0],[0,1]],dtype=dtype)),
	}

	default = None
	D = None
	N = None
	n = None

	hermitian = None
	unitary = None

	def __init__(self,data=None,operator=None,site=None,string=None,system=None,**kwargs):		

		defaults = dict(			
			parameters=None,
			shape=None,size=None,ndim=None,
			samples=None,identity=None,locality=None,
			conj=False,coefficients=None,func=None,gradient=None,
			contract=None,gradient_contract=None,
			)

		setter(kwargs,defaults,delimiter=delim,func=False)
		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		if self.func is None:

			def func(parameters=None,state=None):
				return self.data
				# return self.data if not conj else dagger(self.data)

			self.func = func

		if self.gradient is None:
			
			def gradient(parameters=None,state=None):
				return 0*self.data
				# return 0*(self.data if not conj else dagger(self.data))

			self.gradient = gradient	

		if self.contract is None:

			def contract(data=None,state=None):
				return data

			self.contract = contract

		if self.gradient_contract is None:
			
			def gradient_contract(grad=None,data=None,state=None):
				return grad

			self.gradient_contract = gradient_contract				

		if isinstance(data,self.__class__) or isinstance(operator,self.__class__):
			if isinstance(data,self.__class__):
				for attr in data:
					if attr not in kwargs:
						setattr(self,attr,getattr(data,attr))
			if isinstance(operator,self.__class__):
				for attr in operator:
					if attr not in kwargs:
						setattr(self,attr,getattr(operator,attr))					

			return
		
		elif (data is None) and (operator is None):
			return
		
		elif isinstance(data,(dict)):
			return

		elif (data is not None) and all(isinstance(i,Object) for i in data):
			return

		basis = self.basis
		default = self.default
		operator = self.operator
		site = self.site if self.site is None or not isinstance(self.site,int) else [self.site]
		string = self.string
		locality = self.locality if self.locality is not None else self.N
		N = self.N	
		D = self.D	
		n = self.n

		if operator is None:
			operator = data
		
		# Set operator,site,locality
		if locality is None:
			if site is None:
				if operator is None:
					locality = basis[operator].locality if operator in basis else basis[string].locality if string in basis else locality
					site = [i for i in range(locality)] if locality is not None else None
					operator = operator
				elif isinstance(operator,str):
					if operator in [default]:
						locality = N if N is not None else basis[operator].locality if operator in basis else basis[string].locality if string in basis else locality
						site = [i for i in range(locality)]
						operator = operator							
					elif operator.count(delim):
						locality = sum(basis[i].locality for i in operator.split(delim) if i in basis and basis[i].locality is not None)
						site = [i for i in range(locality)]
						operator = [i for i in operator.split(delim)]
					else:
						locality = sum(basis[i].locality for i in operator.split(delim) if i in basis and basis[i].locality is not None)
						site = [i for i in range(locality)]
						operator = operator
				elif not isinstance(operator,str) and not isinstance(operator,arrays):
					locality = len(operator)
					site = [i for i in range(locality)]
					operator = operator
				else:
					locality = basis[operator].locality if operator in basis else basis[string].locality if string in basis else locality
					site = [i for i in range(locality)] if locality is not None else None
					operator = operator

			else:
				if operator is None:
					locality = len(site)
					site = site
					operator = operator
				elif isinstance(operator,str):
					if operator in [default]:
						locality = N if N is not None else len(site)
						site = [i for i in site]
						operator = operator
					elif operator.count(delim):
						locality = len(site)
						site = [i for i in site]
						operator = [i for i in operator.split(delim)]
					else:
						locality = len(site)
						site = [i for i in site]
						operator = [operator if i in site else default for i in range(N if N is not None else locality)]
				elif not isinstance(operator,str) and not isinstance(operator,arrays):
					locality = len(site)
					site = [i for i in site]
					operator = [operator[site.index(i)] if i in site else default for i in range(N if N is not None else locality)]
				else:
					locality = len(site)
					site = [i for i in site]
					operator = operator

		else:
			if site is None:
				if operator is None:
					locality = locality
					site = [i for i in range(locality)]
					operator = operator
				elif isinstance(operator,str):
					if operator in [default]:
						locality = locality
						site = [i for i in range(locality)]
						operator = operator							
					elif operator.count(delim):
						locality = locality
						site = [i for i in range(locality)]
						operator = [i for i in operator.split(delim)]
					else:
						locality = locality
						site = [i for i in range(locality)]
						operator = operator
				elif not isinstance(operator,str) and not isinstance(operator,arrays):
					locality = locality
					site = [i for i in range(locality)]
					operator = operator
				else:
					locality = locality
					site = [i for i in range(locality)]
					operator = operator
			else:
				if operator is None:
					locality = locality
					site = site
					operator = operator
				elif isinstance(operator,str):
					if operator in [default]:
						locality = locality
						site = [i for i in site]
						operator = operator							
					elif operator.count(delim):
						locality = locality
						site = [i for i in site]
						operator = [i for i in operator.split(delim)]
					else:
						locality = locality
						site = [i for i in site]
						operator = [operator if i in site else default for i in range(N if N is not None else locality)]
				elif not isinstance(operator,str) and not isinstance(operator,arrays):
					locality = locality
					site = [i for i in site]
					operator = [operator[site.index(i)] if i in site else default for i in range(N if N is not None else locality)]
				else:
					locality = locality
					site = [i for i in site]
					operator = operator

		if not isinstance(string,str):
			string = str(string)

		N = max(self.N,max(site)+1 if site is not None else self.N) if N is not None else max(site)+1 if site is not None else 0
		D = self.D if self.D is not None else data.size**(1/max(1,data.ndim*N)) if isinstance(data,arrays) else 1
		n = D**N if (N is not None) and (D is not None) else None


		locality = min(locality,N)
		site = site[:N]
		operator = operator[:N] if not isinstance(operator,str) and not isinstance(operator,arrays) else operator

		shape = self.shape if self.shape is not None else data.shape if isinstance(data,arrays) else None
		size = self.size if self.size is not None else data.size if isinstance(data,arrays) else None
		ndim = self.ndim if self.ndim is not None else data.ndim if isinstance(data,arrays) else None
		dtype = self.dtype if self.dtype is not None else data.dtype if isinstance(data,arrays) else None

		do = (self.parameters is not False) and (self.ndim != 0)

		self.data = data if data is not None else operator if operator is not None else None
		self.operator = operator if operator is not None else None
		self.site = site if site is not None else None
		self.string = string if string is not None else None
		self.system = system if system is not None else None

		self.locality = max(self.locality if self.locality is not None else 0,len(self.site) if self.site is not None else 0)

		self.N = N
		self.D = D
		self.n = n

		self.shape = shape
		self.size = size
		self.ndim = ndim
		self.dtype = dtype

		if (do) and (((self.data is not None) or (self.operator is not None))):
			self.__setup__(data,operator,site,string)

		if not do:
			self.data = None
			self.parameters = False
		elif isinstance(self.operator,arrays):
			self.data = self.operator
			self.operator = self.string
		elif isinstance(self.data,arrays):
			pass
		elif self.operator is None:
			self.data = None
		elif isinstance(self.operator,str):
			self.data = tensorprod([self.basis.get(self.operator)() if i in self.site else self.basis.get(self.default)() for i in range(self.N)]) if self.operator in self.basis else None
		elif self.operator is not None:
			self.data = tensorprod([self.basis.get(i)() for i in self.operator]) if all(i in self.basis for i in self.operator) else None

		self.identity = tensorprod([self.basis.get(self.default)() for i in range(self.N)]) if (self.default in self.basis) else None
		
		if (self.samples is not None) and isinstance(self.data,arrays) and (self.ndim is not None) and (self.data.ndim>self.ndim):
			if isinstance(self.samples,int) and (self.samples > 0):
				shape,bounds,scale,seed,dtype = self.data.shape[:self.data.ndim-self.ndim], [0,1], 'normalize', self.seed, datatype(self.dtype)
				self.samples = rand(size,bounds=bounds,scale=scale,seed=seed,dtype=dtype)
			elif not isinstance(self.samples,arrays):
				self.samples = None

			if (self.samples is not None):
				self.data = einsum('%s,%s...->...'%((''.join(['i','j','k','l'][:self.data.ndim-self.ndim]),)*2),self.samples,self.data)

		self.data = self.data.astype(self.dtype) if self.data is not None else None
		self.identity = self.identity.astype(self.dtype) if self.identity is not None else None


		if self.func is None:

			def func(parameters=None,state=None):
				return self.data
				# return self.data if not conj else dagger(self.data)

			self.func = func

		if self.gradient is None:
			
			def gradient(parameters=None,state=None):
				return 0*self.data
				# return 0*(self.data if not conj else dagger(self.data))

			self.gradient = gradient			

		if self.contract is None:

			def contract(data=None,state=None):
				return data

			self.contract = contract

		if self.gradient_contract is None:
			
			def gradient_contract(grad=None,data=None,state=None):
				return grad

			self.gradient_contract = gradient_contract	

		self.__initialize__()
		
		self.info()

		return

	def __setup__(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments			
		'''

		return

	def __initialize__(self,data=None,state=None,label=None,conj=False):
		'''
		Initialize operator
		Args:
			data (array): data
			state (bool,dict,array,State): State to act on with class of shape self.shape, or class hyperparameters
			label (bool,dict,array,Label): Label for class of shape self.shape, or class hyperparameters
			conj (bool): conjugate
		'''

		if data is None:
			data = self.data
		elif data is True:
			data = self.data			
		elif data is False:
			data = None

		if state is None:
			state = None
		elif state is True:
			state = None
		elif state is False:
			state = None

		if label is None:
			label = None
		elif label is True:
			label = None
		elif label is False:
			label = None

		parameters = self.parameters

		hermitian = self.hermitian
		unitary = self.unitary

		contract = self.contract
		gradient_contract = self.gradient_contract

		if data is not None and state is not None:
			contract = contraction(data,state)

		if data is not None and state is not None:
			gradient_contract = gradient_contraction(data,state)
		
		self.data = data
		self.conj = conj

		self.hermitian = hermitian
		self.unitary = unitary

		self.contract = contract
		self.gradient_contract = gradient_contract

		self.shape = data.shape if isinstance(data,arrays) else None
		self.size = data.size if isinstance(data,arrays) else None
		self.ndim = data.ndim if isinstance(data,arrays) else None
		self.dtype = data.dtype if isinstance(data,arrays) else None

		self.norm()

		return


	def __call__(self,parameters=None,state=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			data (array): data
		'''

		if parameters is None:
			parameters = self.parameters

		return self.contract(self.func(parameters,state),state)

	def grad(self,parameters=None,state=None):
		'''
		Call operator gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			data (array): data
		'''

		if parameters is None:
			parameters = self.parameters

		return self.gradient_contract(self.gradient(parameters,state),self.func(parameters,state),state)


	def __str__(self):
		if isinstance(self.operator,str):
			string = self.operator
		elif self.operator is not None and not isinstance(self.operator,arrays):
			string = '%s'%(delim.join(self.operator))
		elif self.string:
			string = self.string
		else:
			string = self.__class__.__name__
		return string

	def __repr__(self):
		return self.__str__()
	
	def __len__(self):
		return len(self.operator)

	def __key__(self):
		attrs = [self.string,self.operator,self.site]
		key = []
		for attr in attrs:
			if attr is None:
				continue
			if isinstance(attr,scalars):
				key.append(attr)
			else:
				key.extend(attr)
		key = tuple(key)
		return key
	
	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		
		msg = []

		for attr in ['operator','shape','parameters']:
			string = '%s %s: %s'%(self.__class__.__name__,attr,getattr(self,attr))
			msg.append(string)

		msg = '\n'.join(msg)
		
		self.log(msg,verbose=verbose)
		return

	def norm(self):
		'''
		Normalize class
		Args:
			data (array): Data to normalize			
		'''

		try:
			data = self()
		except:
			data = self.data

		if data is None:
			return
		elif not isinstance(data,arrays):
			return

		shape = self.shape
		ndim = self.ndim
		dtype = self.dtype
		hermitian = self.hermitian		
		unitary = self.unitary	

		norm = None
		eps = None

		if hermitian is None and unitary is None:
			norm = None
			eps = None
		
		elif ndim is None:
			norm = None
			eps = None
		
		elif ndim > 3:
			if not hermitian and unitary:
				norm = einsum('...uij,...ukj->...ik',conjugate(data),data)
				eps = array([identity(shape[-2:],dtype=dtype)]*(norm.ndim-2),dtype=dtype)
		
		elif ndim == 3:
			if not hermitian and not unitary:
				norm = einsum('uij,uik->jk',conjugate(data),data)
				eps = identity(shape[-2:],dtype=dtype)
		
		elif ndim == 2:
			if hermitian and unitary:
				norm = einsum('ij,kj->ik',conjugate(data),data)
				eps = identity(shape[-2:],dtype=dtype)
			elif not hermitian and unitary:
				norm = einsum('ij,kj->ik',conjugate(data),data)
				eps = identity(shape[-2:],dtype=dtype)
			elif hermitian and not unitary:
				norm = einsum('ii->',data)
				eps = ones(shape[:-2],dtype=dtype)
			
		
		elif ndim == 1:
			if not hermitian and unitary:
				norm = einsum('i,i->',conjugate(data),data)
				eps = ones(shape=(),dtype=dtype)
		
		if norm is None or eps is None:
			return

		if dtype not in ['complex256','float128']:
			assert (eps.shape == norm.shape), "Incorrect operator shape %r != %r"%(eps.shape,norm.shape)
			assert allclose(eps,norm), "Incorrect norm data%r: %r (hermitian: %r, unitary : %r)"%(eps.shape,norm,hermitian,unitary)

		return

	def swap(self,i,j):
		'''	
		Swap indices of object
		Args:
			i (int): Index to swap
			j (int): Index to swap
		'''

		raise NotImplementedError("TODO: Implement swap for local operators")

		if (self.data is None) or (self.n is None) or (self.N is None) or (self.D is None) or (i == j) or (abs(i) >= self.N) or (abs(j) >= self.N):
			return

		data = self.data
		N = self.N
		D = self.D
		shape = self.shape
		ndim = self.shape

		i = i if i >= 0 else N + i
		j = j if j >= 0 else N + j

		i,j = min(i,j),max(i,j)

		ndims = 1
		dims = shape[:-ndims]
		dim = ndim - ndims

		data = data.reshape((*dims,*[D]*(ndims*N)))
		data = data.transpose(*range(dim),*range(dim,dim+i),j,*range(dim+i+1,dim+j),i,*range(dim+j+1,dim+N))
		data = data.reshape(shape)

		self.data = data
		self.shape = data.shape
		self.size = data.size
		self.ndim = data.ndim

		return

class Operator(Object):
	'''
	Class for Operator
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		inherit (boolean): Inherit super class when initialized
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I','i']},
			}
	default = 'I'
	D = 2
	N = None
	n = None

	def __new__(cls,data=None,operator=None,site=None,string=None,system=None,inherit=False,**kwargs):		

		# TODO: Allow multiple different classes to be part of one operator, and swap around localities

		self = None

		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,system=system),delimiter=delim,func=False)

		classes = [Gate,Pauli,Haar,State,Noise,Object]


		for subclass in classes:
			
			if any(isinstance(obj,subclass) for obj in [data,operator]):

				self = subclass(**kwargs)

				break

			if not all(j in subclass.basis for obj in [data,operator] if (obj is not None and not isinstance(obj,arrays)) for k in (obj if not isinstance(obj,str) else [obj]) for j in ([k] if k in subclass.basis else k.split(delim))):
				continue

			if cls is Operator:
			
				self = subclass(**kwargs)
			
			else:
		
				for attr in subclass.__dict__:
					setattr(cls,attr,getattr(subclass,attr))

				self = subclass.__new__(cls,**kwargs)

				subclass.__init__(self,**kwargs)

			break

		assert (self is not None),"TODO: All operators not in same class"

		return self


class Pauli(Object):
	'''
	Pauli class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I','i']},
		**{attr: Object(data=Basis['X'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['X','x']},
		**{attr: Object(data=Basis['Y'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['Y','y']},
		**{attr: Object(data=Basis['Z'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['Z','z']},
			}
	default = 'I'
	D = 2
	N = None
	n = None
	
	hermitian = None
	unitary = None

	def __setup__(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments			
		'''

		def func(parameters=None,state=None):
			coefficients = (1-2*self.conj)*self.coefficients			
			return cos(coefficients*parameters)*self.identity + -1j*sin(coefficients*parameters)*self.data

		def gradient(parameters=None,state=None):
			coefficients = (1-2*self.conj)*self.coefficients			
			return coefficients*(-sin(coefficients*parameters)*self.identity + -1j*cos(coefficients*parameters)*self.data)

		if self.parameters is None:
			self.parameters = 1

		if self.coefficients is None:
			self.coefficients = pi

		data = self.data if data is None else data

		hermitian = True
		unitary = True

		self.func = func
		self.gradient = gradient

		self.hermitian = hermitian
		self.unitary = unitary

		return


class Gate(Object):
	'''
	Gate class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I']},
		**{attr: Object(data=Basis['CNOT'](),D=2,locality=2,hermitian=True,unitary=True,string=attr) for attr in ['CNOT','C','cnot']},
		**{attr: Object(data=Basis['HADAMARD'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['HADAMARD','H']},
		**{attr: Object(data=Basis['PHASE'](),D=2,locality=1,hermitian=False,unitary=True,string=attr) for attr in ['PHASE','S']},
		**{attr: Object(data=Basis['TOFFOLI'](),D=2,locality=3,hermitian=True,unitary=True,string=attr) for attr in ['TOFFOLI','T','toffoli']},
		}
	default = 'I'
	D = 2 
	N = None
	n = None
	
	hermitian = None
	unitary = None
	
	def __setup__(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''

		data = self.data if data is None else data
		
		hermitian = False
		unitary = True

		self.data = data

		self.hermitian = hermitian
		self.unitary = unitary
		
		return

class Haar(Object):
	'''
	Haar class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I']},
		**{attr: Object(data=Basis['RANDOM'](ndim=2),D=2,locality=1,hermitian=False,unitary=True,string=attr) for attr in ['random','U','haar']},
		}
	default = 'I'
	D = 2
	N = None
	n = None

	hermitian = None
	unitary = None

	def __setup__(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments			
		'''

		if not isinstance(self.data,arrays):
			shape = (self.n,)*self.ndim
			size = prod(shape)
			random = getattr(self,'random','haar')
			seed = getattr(self,'seed',None)
			reset = getattr(self,'reset',None)
			dtype = self.dtype

			data = zeros(shape=shape,dtype=dtype)  
			operator = operator[0] if operator else None

			if operator in ['random','U','haar']:
				data = rand(shape=shape,random=random,seed=seed,reset=reset,dtype=dtype)
			else:
				data = self.data		

		else:

			data = self.data
		
		hermitian = False
		unitary = True

		self.data = data

		self.hermitian = hermitian
		self.unitary = unitary

		return


class State(Object):
	'''
	State class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I']},
		**{attr: Object(data=Basis['RANDOM'](ndim=1),D=2,locality=1,hermitian=False,unitary=True,string=attr) for attr in ['random','psi','haar']},
		**{attr: Object(data=Basis['0'](),D=2,locality=1,hermitian=False,unitary=True,string=attr) for attr in ['zero','zeros','0']},
		**{attr: Object(data=Basis['1'](),D=2,locality=1,hermitian=False,unitary=True,string=attr) for attr in ['one','ones','1']},
		**{attr: Object(data=Basis['+'](),D=2,locality=1,hermitian=False,unitary=True,string=attr) for attr in ['plus','+']},
		**{attr: Object(data=Basis['-'](),D=2,locality=1,hermitian=False,unitary=True,string=attr) for attr in ['minus','-']},
		}
	default = 'I'
	D = 2
	N = None
	n = None

	hermitian = None
	unitary = None

	def __setup__(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments				
		'''

		if not isinstance(self.data,arrays):

			N = self.N
			D = self.D
			default = self.default

			random = getattr(self,'random','haar')
			seed = getattr(self,'seed',None)
			reset = getattr(self,'reset',None)
			dtype = self.dtype

			site = list(range(self.N)) if self.site is None else self.site if not isinstance(self.site,int) else [self.site]
			operator = None if self.operator is None else [self.operator[self.site.index(i)%len(self.operator)] if i in self.site else self.default for i in range(self.N)] if not isinstance(self.operator,str) else [self.operator]*self.N
			locality = len(operator)
			samples = self.samples if self.samples is not None else 1

			local = any((
					all((operator[i] not in ['random','psi','haar'])
					for i in range(N)),
					)
					)

			data = []

			for s in range(samples):
				
				datum = []
				
				for i in range(N):
					
					if local:
						shape = (self.D,)
						seed = spawn(seed)
					else:
						shape = (self.D**self.N,)

					ndim = len(shape)
					size = prod(shape)

					tmp = zeros(shape=shape,dtype=dtype)

					if operator[i] in ['zero','zeros','0']:
						tmp = inplace(tmp,0,1)
					elif operator[i] in ['one','ones','1']:
						tmp = inplace(tmp,-1,1)
					elif operator[i] in ['plus','+']:
						tmp = inplace(tmp,slice(None),1/sqrt(size))
					elif operator[i] in ['minus','-']:
						tmp = inplace(tmp,slice(None),(-1)**arange(size)/sqrt(size))
					elif operator[i] in ['random','psi','haar']:
						tmp = rand(shape=shape,random=random,seed=seed,reset=reset,dtype=dtype)
					elif isinstance(self.data,arrays):
						tmp = self.data.reshape(N,*shape)[i]
					else:
						tmp = None

					if tmp is None:
						datum = tmp
						break

					datum.append(tmp)

					if not local:
						break

				if datum is not None:
					if not isinstance(datum,arrays):
						datum = tensorprod(datum)

					if datum.ndim < self.ndim:
						datum = einsum('...i,...j->...ij',datum,conjugate(datum))

				data.append(datum)

			if data is not None:
				if len(data) == 1:
					data = data[-1]
				else:
					data = array(data,dtype=self.dtype)

		else:

			data = self.data

		if self.ndim == 1:
			hermitian = False
			unitary = True
		else:
			hermitian = True
			unitary = False

		self.data = data

		self.hermitian = hermitian
		self.unitary = unitary

		return


class Noise(Object):
	'''
	Noise class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I','i']},
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['eps','noise','rand']},
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['depolarize']},
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['amplitude']},
		**{attr: Object(data=Basis['00'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['00']},
		**{attr: Object(data=Basis['01'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['01']},
		**{attr: Object(data=Basis['10'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['10']},
		**{attr: Object(data=Basis['11'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['11']},
		**{attr: Object(data=Basis['X'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['X','x','flip','bitflip']},
		**{attr: Object(data=Basis['Y'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['Y','y','flipphase']},
		**{attr: Object(data=Basis['Z'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['Z','z','phase','dephase']},
		}
	default = 'I'
	D = 2
	N = None
	n = None
	
	hermitian = None
	unitary = None

	def __init__(self,data=None,operator=None,site=None,string=None,system=None,**kwargs):		

		defaults = dict(			
			shape=None,size=None,ndim=None,
			samples=None,identity=None,locality=None,index=None
			)

		setter(kwargs,defaults,delimiter=delim,func=False)
		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		return

	def __setup__(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments			
		'''

		if not isinstance(self.data,arrays):

			if self.parameters is None:
				self.parameters = 0

			do = (self.parameters is not None) and (self.parameters is not False)

			if not do:
				self.data = None
				self.operator = None
				return

			N = self.N
			default = self.default
			site = list(range(self.N)) if self.site is None else self.site if not isinstance(self.site,int) else [self.site]
			operator = None if self.operator is None else [self.operator[self.site.index(i)%len(self.operator)] if i in self.site else self.default for i in range(self.N)] if not isinstance(self.operator,str) else [self.operator]*self.N
			locality = len(operator)
			parameters = [None]*self.N if self.parameters is None else [self.parameters[[self.site.index(i)%len(self.parameters)]] if i in self.site else self.default for i in range(self.N)] if not isinstance(self.parameters,scalars) and self.parameters.size > 1 else [self.parameters]*self.N

			local = any(
					True
					for i in range(N)
					)

			data = []

			assert ((isinstance(parameters,scalars) and (parameters >= 0) and (parameters <= 1)) or (all((i>=0) and (i<=1) for i in parameters))), "Noise scale %r not in [0,1]"%(parameters)

			for i in range(N):

				if operator[i] is None:
					datum = [self.basis[self.default]()]
				elif operator[i] in ['Z','z','phase','dephase']:
					datum = [sqrt(1-parameters[i])*self.basis['I'](),
							sqrt(parameters[i])*self.basis['Z']()]
				elif operator[i] in ['X','x','flip','bitflip']:
					datum = [sqrt(1-parameters[i])*self.basis['I'](),
							sqrt(parameters[i])*self.basis['X']()]
				elif operator[i] in ['Y','y','flipphase']:
					datum = [sqrt(1-parameters[i])*self.basis['I'](),
							sqrt(parameters[i])*self.basis['Y']()]												
				elif operator[i] in ['amplitude']:
					datum = [self.basis['00']() + sqrt(1-parameters[i])*self.basis['11'](),
							sqrt(parameters[i])*self.basis['01']()]
				elif operator[i] in ['depolarize']:
					datum = [sqrt(1-parameters[i])*self.basis['I'](),
							sqrt(parameters[i]/(self.D**2-1))*self.basis['X'](),
							sqrt(parameters[i]/(self.D**2-1))*self.basis['Y'](),
							sqrt(parameters[i]/(self.D**2-1))*self.basis['Z']()]
				elif operator[i] in ['eps']:
					datum = array([identity(self.n,dtype=self.dtype),diag((1+parameters[i])**(arange(self.n)+2) - 1)])
				elif operator[i] in ['noise','rand']:
					datum = array(parameters[i],dtype=datatype(self.dtype))
					seed = prng(reset=self.seed)
				else:
					datum = [self.basis[self.default]()]

				if isinstance(datum,arrays):
					data = datum
					break

				data.append(datum)

				if not local:
					break

			if not isinstance(data,arrays):
				data = array([tensorprod(i)	for i in permutations(*data)],dtype=self.dtype)

		else:

			data = self.data

		hermitian = False
		unitary = False

		if self.ndim == 0:
			def func(parameters=None,state=None):
				return state + parameters*rand(state.shape,random='uniform',bounds=[-1,1],seed=None,dtype=self.dtype)/2

			def gradient(parameters=None,state=None):
				return 0*state

		else:
			func = self.func
			gradient = self.gradient


		self.data = data

		self.func = func
		self.gradient = gradient

		self.hermitian = hermitian
		self.unitary = unitary
		
		return


class Operators(Object):
	'''
	Class for Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional operator keyword arguments			
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (iterable[str]): string labels of operators
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Simulation length scale		
		M (int): Number of time steps
		T (int): Simulation Time
		tau (float): Simulation time scale		
		P (int): Trotter order		
		space (str,Space): Type of local space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice	
		parameters (iterable[str],dict,Parameters): Type of parameters of operators
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	def __init__(self,data=None,operator=None,site=None,string=None,
		N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
		space=None,time=None,lattice=None,parameters=None,system=None,**kwargs):

		setter(kwargs,system,delimiter=delim,func=False)
		super().__init__(**kwargs)

		self.N = N
		self.D = D
		self.d = d
		self.L = L
		self.delta = delta
		self.M = M
		self.T = T
		self.tau = tau
		self.P = P
		self.space = space
		self.time = time
		self.lattice = lattice

		self.data = Dictionary()

		self.n = None
		self.g = None
		
		self.parameters = parameters
		self.state = None
		self.label = None
		self.identity = None
		self.coefficients = None
		self.conj = False

		self.func = None
		self.gradient = None
		self.gradient_automatic = None
		self.gradient_finite = None
		self.gradient_analytical = None
		
		self.system = system

		self.__time__()
		self.__space__()
		self.__lattice__()

		self.identity = Operator(Operator.default,N=self.N,D=self.D,system=self.system,verbose=False)

		self.shape = () if self.n is None else (self.n,self.n)
		self.size = prod(self.shape)
		self.ndim = len(self.shape)

		self.__setup__(data,operator,site,string)

		self.__initialize__()

		self.info()

		return	

	def __setup__(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
				string (iterable[str]): string labels of operators
				kwargs (dict): Additional operator keyword arguments			
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional class keyword arguments			
		'''

		# Get operator,site,string from data
		objs = {'operator':operator,'site':site,'string':string}

		for obj in objs:
			objs[obj] = [] if objs[obj] is None else objs[obj]

		if data is None:
			data = None
		elif all(isinstance(datum,Object) for datum in data):
			for obj in objs:
				objs[obj] = None
		elif isinstance(data,dict) and all(isinstance(data[name],dict) and (obj in data[name]) for name in data for obj in objs):
			for obj in objs:
				objs[obj].extend([data[name][obj] for name in data])

			kwargs.update({kwarg: [data[name][kwarg] if kwarg in data[name] else null for name in data] 
				for kwarg in set(kwarg for name in data for kwarg in data[name] if kwarg not in objs)
				})

			data = None

		# Set class attributes
		self.__extend__(data=data,**objs,**kwargs)

		return


	def __append__(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Append to class
		Args:
			data (str,Operator): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments			
		'''
		index = -1
		self.__insert__(index,data,operator,site,string,**kwargs)
		return

	def __extend__(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup class
		Args:
			data (iterable[str,Operator]): data of operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional operator keyword arguments			
		'''


		size = min([len(i) for i in (data,operator,site,string) if i is not None],default=0)

		length = min([len(i) for i in (kwargs[kwarg] for kwarg in kwargs) if i is not null],default=size)
		kwargs = [{kwarg: kwargs[kwarg][i] for kwarg in kwargs} for i in range(length)]
		
		if not size:
			return

		if data is None:
			data = [None]*size
		if operator is None:
			operator = [None]*size
		if site is None:
			site = [None]*size
		if string is None:
			string = [None]*size						
		if kwargs is None:
			kwargs = [{}]*size	

		for _data,_operator,_site,_string,_kwargs in zip(data,operator,site,string,kwargs):
			self.__append__(_data,_operator,_site,_string,**_kwargs)

		return


	def __insert__(self,index,data,operator,site,string,**kwargs):
		'''
		Insert to class
		Args:
			index (int): index to insert operator
			data (str,Operator): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments						
		'''

		if index == -1:
			index = len(self)

		defaults = dict(N=self.N,D=self.D,system=self.system,verbose=False)

		kwargs = {kwarg: kwargs[kwarg] for kwarg in kwargs if not isinstance(kwargs[kwarg],nulls)}

		setter(kwargs,defaults,func=False)

		data = Operator(data=data,operator=operator,site=site,string=string,**kwargs)

		self.data = insert(self.data,index,{index: data})

		return

	def __initialize__(self,parameters=None,data=None,state=None,label=None,conj=None):
		''' 
		Setup class functions
		Args:
			parameters (bool,dict,array,Parameters): Class parameters
			data (bool,dict): data of class
			state (bool,dict,array,State): State to act on with class of shape self.shape, or class hyperparameters
			label (bool,dict,array,Label): Label for class of shape self.shape, or class hyperparameters
			conj (bool): conjugate
		'''

		parameters = self.parameters if parameters is None else parameters
		conj = self.conj if conj is None else conj

		objs = {'parameters':parameters,'state':state,'label':label}
		classes = {'parameters':Parameters,'state':State,'label':Label}
		arguments = {'parameters':False,'state':True,'label':None}

		# Get functions
		for obj in objs:
			instance,cls,argument = objs[obj],classes[obj],arguments[obj]
			instance = getattr(self,obj,None) if instance is None or instance is True else instance if instance is not False else None
			if not isinstance(instance,cls):
				kwargs = {}

				args = ({**dict(data=None),**instance} if (isinstance(instance,dict) and (argument or all(attr in instance for attr in dict(instance=None)))) else dict(data=instance))
				setter(kwargs,args,func=False)

				args = dict(**namespace(cls,self),model=self,system=self.system)
				setter(kwargs,args,func=False)

				args = dict(verbose=False)
				setter(kwargs,args,func=True)

				instance = cls(**kwargs)

			setattr(self,obj,instance)


		# Check objs
		assert self.parameters(self.parameters()) is not None, "Incorrect parameters"

		# Set data
		if data is None or data is True:
			data = {}
		if data is False:
			data = {i: data for i in self.data}
		elif isinstance(data,dict):
			data = {i: data[i] for i in data}
		for i in data:
			self.data[i].__initialize__(data=data[i])

		for i in self.data:
			self.data[i].__initialize__(state=state,label=label)


		# Set attributes
		state = self.state()
		label = self.label()
		identity = self.identity()
		parameters = self.parameters()
		conj = self.conj

		indices = self.parameters.indices
		wrapper = self.parameters.wrapper


		# Set trotterized indices
		p = self.P

		indexes = {i: self.data[data] for i,data in enumerate(self.data) if (self.data[data]() is not None)}
		booleans = {}
		for i in indexes:
			if indexes[i].unitary:
				boolean = True
			elif not indexes[i].unitary:
				boolean = False
			else:
				boolean = None
			booleans[i] = boolean


		true = [True]
		false = [False]
		
		shape = (len(self),self.M)

		length = len(self)
		shape = (length,*shape[1:])
		indices = {tuple(to_position(i,shape=shape[::-1]))[::-1]:indices[i] for i in indices}

		sort = {attr: [i for i in booleans if booleans[i] == attr] for attrs in [true,false] for attr in attrs}
		sort = interleave({attr: sort[attr] for attr in sort if sort[attr]})

		slices = {}
		step = 0		
		for index,slc in sort:

			slc = [int(i) for i in slc]
			
			if index in true:
				tmp = trotter(slc,p=p)
			elif index in false:
				tmp = slc
			else:
				tmp = slc

			for i in slc:
				for k,j in enumerate(tmp):
					if j == i:
						slices[min(tmp)+k+step] = i

			step = max(slices,default=0)-max(slc,default=0)

		slices = {i: slices[i] for i in sorted(slices,key=lambda i:i)}

		length = max(slices,default=0)+1
		shape = (length,*shape[1:])
		indices = {to_index((j,*i[1:])[::-1],shape=shape[::-1]):indices[i] for i in indices for j in slices if i[0]==slices[j]}

		coefficients = {
			**{attr:(self.tau)*trotter(p=p) for attr in true},
			**{attr:1 for attr in false}
			}
		coefficients = [coefficients[booleans[slices[i]]] for i in slices]
		coefficients = array(coefficients,dtype=datatype(self.dtype))[:,None]

		data = trotter([jit(indexes[slices[i]]) for i in slices],p=p)
		grad = trotter([jit(indexes[slices[i]].grad) for i in slices],p=p)

		slices = array([slices[i] for i in slices])
		indices = {i:indices[i] for i in indices}
		def wrapper(parameters,slices,coefficients): 
			return (coefficients*parameters[slices]).T.ravel()
		wrapper = jit(wrapper,slices=slices,coefficients=coefficients)

		hermitian = all(indexes[i].hermitian for i in indexes)
		unitary = all(indexes[i].unitary for i in indexes)

		self.coefficients = coefficients

		self.parameters.indices = indices
		self.parameters.wrapper = wrapper

		parameters = self.parameters(parameters)

		func = scheme(parameters=parameters,conj=conj,data=data,identity=identity)
		
		grad_automatic = gradient(self,mode='fwd',move=True)
		grad_finite = gradient(self,mode='finite',move=True)
		# grad_analytical = grad_automatic
		grad_analytical = gradient_scheme(parameters=parameters,conj=conj,data=data,identity=identity,grad=grad,indices=indices)

		grad = grad_automatic

		# Update class attributes
		self.func = func
		self.gradient = grad
		self.gradient_automatic = grad_automatic
		self.gradient_finite = grad_finite
		self.gradient_analytical = grad_analytical
		self.hermitian = hermitian
		self.unitary = unitary

		self.shape = shape
		self.size = prod(self.shape)
		self.ndim = len(self.shape)

		self.conj = conj

		self.norm()

		return

	def __call__(self,parameters=None,state=None):
		'''
		Class function
		Args:
			parameters (array): parameters		
			state (obj): state
		Returns
			out (array): Return of function
		'''

		if parameters is None:
			parameters = self.parameters()

		parameters = self.parameters(parameters)

		return self.func(parameters,state)

	def grad(self,parameters=None,state=None):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			out (array): Return of function
		'''		
		return self.gradient(parameters,state)

	def grad_automatic(self,parameters=None,state=None):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			out (array): Return of function
		'''		
		return self.gradient_automatic(parameters,state)	

	def grad_finite(self,parameters=None,state=None):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			out (array): Return of function
		'''		
		return self.gradient_finite(parameters,state)		

	def grad_analytical(self,parameters=None,state=None):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			out (array): Return of function
		'''		
		if parameters is None:
			parameters = self.parameters()

		parameters = self.parameters(parameters)

		grad = self.coefficients*self.gradient_analytical(parameters,state)

		return grad


	def __len__(self):
		return len(self.data)

	def __space__(self,N=None,D=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			space (str,Space): Type of local space
			system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
		'''

		N = self.N if N is None else N
		D = self.D if D is None else D
		space = self.space if space is None else space
		system = self.system if system is None else system

		self.space = Space(N,D,space,system=system)

		self.N = self.space.N
		self.D = self.space.D		
		self.n = self.space.n
		self.g = self.space.g

		return


	def __time__(self,M=None,T=None,tau=None,P=None,time=None,system=None):
		'''
		Set time attributes
		Args:
			M (int): Number of time steps
			T (int): Simulation Time
			tau (float): Simulation time scale
			P (int): Trotter order		
			time (str,Time): Type of Time evolution space						
			system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
		'''
		M = self.M if M is None else M
		T = self.T if T is None else T
		tau = self.tau if tau is None else tau
		P = self.P if P is None else P
		time = self.time if time is None else time
		system = self.system if system is None else system

		self.time = Time(M,T,tau,P,time,system=system)	

		self.M = self.time.M
		self.T = self.time.T
		self.P = self.time.P
		self.tau = self.time.tau
		
		return


	def __lattice__(self,N=None,D=None,d=None,L=None,delta=None,lattice=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			d (int): Spatial dimension
			L (int,float): Scale in system
			delta (float): Simulation length scale		
			lattice (str,Lattice): Type of lattice		
			system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
		'''		
		N = self.N if N is None else N
		D = self.D if D is None else D
		d = self.d if d is None else d
		L = self.L if L is None else L
		delta = self.delta if delta is None else delta
		lattice = self.lattice if lattice is None else lattice
		system = self.system if system is None else system

		self.lattice = Lattice(N,d,L,delta,lattice,system=system)	

		self.N = self.lattice.N
		self.d = self.lattice.d
		self.L = self.lattice.L
		self.delta = self.lattice.delta

		return

	def __str__(self):
		size = len(self)
		delimiter = ' '
		multiple_time = (self.M>1) if self.M is not None else None
		multiple_space = [size>1 and False for i in range(size)]
		return '%s%s%s%s'%(
				'{' if multiple_time else '',
				delimiter.join(['%s%s%s'%(
					'(' if multiple_space[i] else '',
					self.data[i].string,
					')' if multiple_space[i] else '',
					) for i in range(size)]),
				'}' if multiple_time else '',
				'%s'%('^%s'%(self.M) if multiple_time else '') if multiple_time else '')

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		

		msg = []

		for attr in ['string','key','seed','instance','instances','N','D','d','L','delta','M','tau','T','P','n','g','unit','data','shape','size','ndim','dtype','cwd','path','backend','architecture','conf','logger','cleanup']:
			substring = getattrs(self,attr,delimiter=delim,default=None)
			substring = substring if not isinstance(substring,dict) else [substring[i] for i in substring]
			string = '%s: %s'%(attr,substring)
			msg.append(string)

		for attr in ['func.__name__']:
			string = '%s: %s'%(delim.join(attr.split(delim)[:1]),getattrs(self,attr,delimiter=delim,default=None))
			msg.append(string)

		for attr in ['parameters.%s'%(i) for i in (self.parameters if self.parameters is not None else [])]:
			string = []
			for subattr in ['category','method','locality','shape','parameters']:
				substring = getattrs(self,delim.join([attr,subattr]),delimiter=delim,default=None)
				if isinstance(substring,(str,int,list,tuple,*arrays)):
					substring = '%s'%(substring,)
				elif isinstance(substring,dict):
					substring = ', '.join(['%s: %s'%(prop,substring[prop]) for prop in substring])	
				elif substring is not None:
					substring = '%0.4e'%(substring)
				else:
					substring = str(substring)
				substring = '%s : %s'%(subattr,substring)
				
				string.append(substring)

			string = '%s %s'%(delim.join(attr.split(delim)[:2]),', '.join(string))

			msg.append(string)

		for attr in ['parameters']:
			string = []
			for subattr in [None,'shape','parameters']:
				if subattr is None:
					subattr = attr
					substring = str(getattrs(self,attr,delimiter=delim,default=None))
				else:
					substring = getattrs(self,delim.join([attr,subattr]),delimiter=delim,default=None)
				if isinstance(substring,(str,int,list,tuple,*arrays)):
					substring = '%s'%(substring,)
				elif substring is not None:
					substring = '%0.4e'%(substring)
				else:
					substring = str(substring)
				substring = '%s : %s'%(subattr,substring)

				string.append(substring)

			string = ', '.join(string)

			msg.append(string)			

		for attr in ['noise']:
			string = []
			for subattr in [None,'shape','parameters']:
				for i in self.data:
					data = self.data[i]
					if data.string != attr:
						continue
					if subattr is None:
						subattr = attr
						substring = str(data)
					else:
						substring = getattrs(data,subattr,delimiter=delim,default=None)
					if isinstance(substring,(str,int,list,tuple,*arrays)):
						substring = '%s'%(substring,)
					elif substring is not None:
						substring = '%0.4e'%(substring)
					else:
						substring = str(substring)
					substring = '%s : %s'%(subattr,substring)

					string.append(substring)

			string = ', '.join(string)

			msg.append(string)	

		# for attr in ['model']:
		# 	string = '%s :\n%s'%(delim.join(attr.split(delim)[:1]),to_string(self()))

		# 	msg.append(string)			

		# for attr in ['parameters','noise']:
		# 	string = getattrs(self,attr,default=None)
		# 	if callable(string) and string() is not None:
		# 		string = '%s :\n%s'%(delim.join(attr.split(delim)[:1]),to_string(string()))
		# 	else:
		# 		string = '%s : %s'%(delim.join(attr.split(delim)[:1]),string())

		# 	msg.append(string)						


		msg = '\n'.join(msg)

		
		self.log(msg,verbose=verbose)

		return


	def dump(self,path=None):
		'''
		Save class data		
		Args:
			path (str,dict[str,(str,bool)]): Path to dump class data, either path or boolean to dump			
		'''

		# Set data
		data = {}

		# Set path
		paths = {}
		if path is None:
			paths.update({attr: True for attr in data})			
		elif not isinstance(path,dict):
			paths.update({attr: path for attr in data if path})
		else:
			paths.update({attr: path[attr] for attr in path if path[attr]})

		paths.update({attr: paths.get(attr) if isinstance(paths.get(attr),str) else self.cwd for attr in data if paths.get(attr)})			

		# Dump data
		for attr in paths:
			root,file = split(paths[attr],directory=True,file_ext=True)
			file = file if file is not None else self.path
			path = join(file,root=root)
			dump(data[attr],path)
		
		return

	def load(self,path=None):
		'''
		Load class data		
		Args:
			path (str,dict[str,(str,bool)]): Path to load class data, either path or boolean to load
		'''

		# TODO: Determine dump/load model (.pkl?)

		# Set data
		data = {}

		# Set path
		paths = {}
		if path is None:
			paths.update({attr: True for attr in data})			
		elif not isinstance(path,dict):
			paths.update({attr: path for attr in data if path})
		else:
			paths.update({attr: path[attr] for attr in path if path[attr]})

		paths.update({attr: paths.get(attr) if isinstance(paths.get(attr),str) else self.cwd for attr in data if paths.get(attr)})			

		# Load data
		for attr in paths:
			root,file = split(paths[attr],directory=True,file_ext=True)
			file = file if file is not None else self.path
			path = join(file,root=root)
			func = (list,)
			default = data[attr]
			data[attr] = load(path,default=default)
			setter(default,data[attr],func=func)

		return



class Hamiltonian(Operators):
	'''
	Hamiltonian class of Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional operator keyword arguments			
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (iterable[str]): string labels of operators
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Simulation length scale		
		M (int): Number of time steps
		T (int): Simulation time
		tau (float): Simulation time scale
		P (int): Trotter order		
		space (str,Space): Type of local space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		parameters (iterable[str],dict,Parameters): Type of parameters of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	def __init__(self,data=None,operator=None,site=None,string=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
				space=None,time=None,lattice=None,parameters=None,system=None,**kwargs):

		super().__init__(data=data,operator=operator,site=site,string=string,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
				space=space,time=time,lattice=lattice,parameters=parameters,system=system,**kwargs)
		
		return

	def __setup__(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
				string (iterable[str]): string labels of operators
				kwargs (dict): Additional operator keyword arguments			
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional class keyword arguments		
		'''

		# Get operator,site,string from data
		objs = {'operator':operator,'site':site,'string':string}

		for obj in objs:
			objs[obj] = [] if objs[obj] is None else objs[obj]

		# Set attributes

		# Get data and kwargs
		if data is None:
			data = None
		elif all(isinstance(datum,Object) for datum in data):
			for obj in objs:
				objs[obj] = None
		elif isinstance(data,dict) and all(isinstance(data[name],dict) and (obj in data[name]) for name in data for obj in objs):
			for obj in objs:
				objs[obj].extend([data[name][obj] for name in data])
			
			kwargs.update({kwarg: [data[name][kwarg] if kwarg in data[name] else null for name in data] 
				for kwarg in set(kwarg for name in data for kwarg in data[name] if kwarg not in objs)
				})

			data = None

		# Lattice sites
		sites = self.lattice # sites types on lattice
		indices = {'i': ['i'],'<ij>':['i','j'],'>ij<':['i','j'],'i<j':['i','j'],'ij':['i','j'],'i...j':['i','j']}   # allowed symbolic indices and maximum locality of many-body site interactions

		# Get number of operators
		size = min([len(objs[obj]) for obj in objs if objs[obj] is not None],default=0)
		
		# Get attribute of symbolic indices
		attr = 'site'

		# Get data
		for index in range(size):

			key = None
			tmp = {obj: deepcopy(objs[obj].pop(0)) for obj in objs}
			tmps = {kwarg: deepcopy(kwargs[kwarg].pop(0)) for kwarg in kwargs}

			if tmp[attr] is None:
				key = None
			elif isinstance(tmp[attr],scalars) and tmp[attr] in indices:
				key = tmp[attr]
				tmp[attr] = indices[tmp[attr]]
			elif isinstance(tmp[attr],scalars):
				tmp[attr] = [tmp[attr]]
			elif not isinstance(tmp[attr],scalars):
				for i in tmp[attr]:
					if i in indices:
						key = i
						tmp[attr][tmp[attr].index(i)] = indices[i][tmp[attr].index(i)]
			if key is not None:
				for i,s in enumerate(sites(key)):
					value = {}
					for obj in objs:
						if obj in [attr]:
							value[obj] = [dict(zip(
								indices[key],
								s if not isinstance(s,int) else [s])
							).get(i,int(i) if not isinstance(i,str) else i) 
							for i in tmp[attr]]
						else:
							value[obj] = tmp[obj]

					exists = [[i if value[obj] == item else None for i,item in enumerate(objs[obj])] 
						for obj in objs]
					
					if any(len(set(i))==1 for i in zip(*exists) if any(j is not None for j in i)):
						continue

					for obj in objs:
						objs[obj].append(value[obj])
					
					for kwarg in kwargs:
						kwargs[kwarg].append(tmps[kwarg])

			else:
				for obj in objs:
					objs[obj].append(tmp[obj])	

				for kwarg in kwargs:
					kwargs[kwarg].append(tmps[kwarg])


		# Set class attributes
		self.__extend__(data=data,**objs,**kwargs)

		return



class Unitary(Hamiltonian):
	'''
	Unitary class of Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional operator keyword arguments						
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (iterable[str]): string labels of operators
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Simulation length scale				
		M (int): Number of time steps
		T (int): Simulation Time
		tau (float): Simulation time scale		
		P (int): Trotter order		
		space (str,Space): Type of local space
		time (str,Time): Type of Time evolution space
		lattice (str,Lattice): Type of lattice
		parameters (iterable[str],dict,Parameters): Type of parameters of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	def __init__(self,data=None,operator=None,site=None,string=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
				space=None,time=None,lattice=None,parameters=None,system=None,**kwargs):
		
		super().__init__(data=data,operator=operator,site=site,string=string,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
				space=space,time=time,lattice=lattice,parameters=parameters,system=system,**kwargs)

		return



class Channel(Unitary):
	'''
	Channel class of Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional operator keyword arguments			
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (iterable[str]): string labels of operators
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Simulation length scale				
		M (int): Number of time steps
		T (int): Simulation Time
		tau (float): Simulation time scale		
		P (int): Trotter order		
		space (str,Space): Type of local space
		time (str,Time): Type of Time evolution space
		lattice (str,Lattice): Type of lattice
		parameters (iterable[str],dict,Parameters): Type of parameters of operators				
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''
	def __init__(self,data=None,operator=None,site=None,string=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
				space=None,time=None,lattice=None,parameters=None,system=None,**kwargs):
		
		super().__init__(data=data,operator=operator,site=site,string=string,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
				space=space,time=time,lattice=lattice,parameters=parameters,system=system,**kwargs)

		return


class Label(Operator):
	
	basis = {}
	default = None
	D = None
	N = None
	n = None

	hermitian = None
	unitary = None

	def __new__(cls,*args,**kwargs):

		self = super().__new__(cls,*args,**kwargs)

		return self

	def __init__(self,*args,**kwargs):
		return

class Callback(System):
	def __init__(self,*args,**kwargs):
		'''	
		Class for callback
		Args:
			args (tuple): Class arguments
			kwargs (dict): Class keyword arguments
		'''

		super().__init__(*args,**kwargs)

		self.defaults = {
			'iteration':[],
			'parameters':[],'grad':[],'search':[],
			'value':[],'objective':[],
			'alpha':[],'beta':[],

			'iteration.max':[],'iteration.min':[],
			'parameters.relative':[],'parameters.relative.mean':[],
			'variables':[],'variables.relative':[],'variables.relative.mean':[],
			'objective.ideal.noise':[],'objective.diff.noise':[],'objective.rel.noise':[],
			'objective.ideal.state':[],'objective.diff.state':[],'objective.rel.state':[],
			'objective.ideal.operator':[],'objective.diff.operator':[],'objective.rel.operator':[],
			'hessian':[],'fisher':[],
			'hessian.eigenvalues':[],'fisher.eigenvalues':[],
			'hessian.rank':[],'fisher.rank':[],

			'N':[],'D':[],'d':[],'L':[],'delta':[],'M':[],'T':[],'tau':[],'P':[],
			'space':[],'time':[],'lattice':[],'architecture':[],'timestamp':[],

			"noise.string":[],"noise.ndim":[],"noise.locality":[],
			"noise.parameters":[],"noise.scale":[],"noise.tau":[],"noise.initialization":[],

			"state.string":[],"state.ndim":[],"label.string":[],"label.ndim":[],

			'hyperparameters.c1':[],'hyperparameters.c2':[],

		}

		return

	def __call__(self,parameters,track,optimizer,model,metric,func,grad):
		''' 
		Callback
		Args:
			parameters (array): parameters
			track (dict): callback tracking
			optimizer (Optimizer): callback optimizer
			model (object): Model instance
			metric (str,callable): Callback metric
			func (callable): Objective function with signature func(parameters)
			grad (callable): Objective gradient with signature grad(parameters)
		Returns:
			status (int): status of callback
		'''

		attributes = optimizer.attributes
		iterations = optimizer.iterations
		hyperparameters = optimizer.hyperparameters

		init = (len(attributes['iteration'])==1) and ((attributes['iteration'][-1]==0) or (attributes['iteration'][-1] != (iterations.stop)))
		
		done = ((len(attributes['iteration'])>1) and (attributes['iteration'][-1] == (iterations.stop)))
		
		status = (
			((len(attributes['value']) >= 1) and 
			 (attributes['iteration'][-1] <= max(1,
				hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) or
			(
			(abs(attributes['value'][-1]) > 
				(hyperparameters['eps']['value']*hyperparameters['value']['value'])) and
			(log10(abs(attributes['value'][-1] - attributes['value'][-2])) > 
				(log10(abs(hyperparameters['eps']['value.difference'])))) and
			(norm(attributes['grad'][-1])/(attributes['grad'][-1].size) > 
				  (hyperparameters['eps']['grad']*hyperparameters['value']['grad'])) and
			(norm(attributes['grad'][-1] - attributes['grad'][-2])/(attributes['grad'][-2].size) > 
				  (hyperparameters['eps']['grad.difference']*norm(attributes['grad'][-2])/(attributes['grad'][-2].size)))
			)
			)


		other = ((len(attributes['iteration']) == 1) or 
			(hyperparameters['modulo']['track'] is None) or 
			(attributes['iteration'][-1]%hyperparameters['modulo']['track'] == 0))

		stop = (
			(hyperparameters['eps'].get('value.increase') is not None) and
			(hyperparameters['eps'].get('value.increase') > 0) and
			((len(attributes['value']) > 1) and 
			 (attributes['iteration'][-1] >= max(1,
				hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) and
			((attributes['value'][-1] > attributes['value'][-2]) and
			(log10(attributes['value'][-1] - attributes['value'][-2]) > 
			(log10(hyperparameters['eps']['value.increase']*attributes['value'][-1]))))
			)

		none = (iterations.start == 0) and (iterations.stop == 0)

		status = ((status) and (not stop) and (not none))

		updates = {
			**{attr: lambda i,attr,track,default: (track[attr][-1]) for attr in ['iteration.max','iteration.min']},
			**{attr: lambda i,attr,track,default: (empty(track[attr][-1].shape) if ((i>0) and i<(len(track[attr])-1)) else track[attr][i])
				for attr in [
					'parameters','grad','search',
					'parameters.relative',
					'variables','variables.relative',
					'hessian','fisher',
					'hessian.eigenvalues','fisher.eigenvalues']},
			**{attr: None for attr in [
				'parameters.norm','grad.norm','search.norm','variables.norm',
				]},
			**{attr: lambda i,attr,track,default: (default if i<(len(track[attr])-1) else track[attr][i])
				for attr in [
				'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
				'objective.ideal.state','objective.diff.state','objective.rel.state',
				'objective.ideal.operator','objective.diff.operator','objective.rel.operator',
				'hessian.rank','fisher.rank']
			},
			}

		attrs = relsort(track,attributes)
		size = min(len(track[attr]) for attr in track)
		does = {**{attr: False for attr in attrs},**hyperparameters.get('do',{})}


		if ((status) or done or init or other):
			
			for attr in attrs:

				if ((hyperparameters['length']['track'] is not None) and 
					(len(track[attr]) > hyperparameters['length']['track'])
					):
					_value = track[attr].pop(0)
				

				index = -1 if (not stop) else -2
				parameters = attributes['parameters'][index]
			
				if attr in [
					'parameters','grad','search',
					'parameters.relative',
					'variables','variables.relative',
					'hessian','fisher',
					'hessian.eigenvalues','fisher.eigenvalues']:
					default = empty(track[attr][-1].shape) if (len(track[attr])>0) else nan
				else:
					default = nan

				do = (not ((status) and (not done) and (not init))) or does[attr]

				value = default

				if attr in attributes:
					value = attributes[attr][index]

				if (not stop):
					track[attr].append(value)

				if attr in ['iteration.max']:
					value = int(track['iteration'][-1])

				elif attr in ['iteration.min']:
					value = int(track['iteration'][argmin(abs(array(track['objective'],dtype=model.dtype)))])

				elif attr in ['value']:
					value = abs(attributes[attr][index])
				
				elif attr in ['parameters','grad','search'] and (not do):
					value = default

				elif attr in ['parameters','grad','search'] and (do):
					value = attributes[attr][index]

				elif attr in ['parameters.norm','grad.norm','search.norm']:
					value = attr.split(delim)[0]
					value = attributes[value][index]
					value = norm(value)/(value.size)

				elif attr in ['parameters.relative','parameters.relative.mean',
					] and (not do):
					value = default

				elif attr in [
					'parameters.relative','parameters.relative.mean',
					] and (do):
					eps = 1e-20
					if attr in ['parameters.relative']:
						value = parameters
						_value = attributes['parameters'][0]
						value = abs((value-_value)/(_value+eps))
					elif attr in ['parameters.relative.mean']:
						value = parameters
						_value = attributes['parameters'][0]
						value = norm((value-_value)/(_value+eps))/(value.size)

				elif attr in ['variables.norm','variables.relative','variables.relative.mean'
					] and (not do):
					value = default

				elif attr in [
					'variables','variables.norm','variables.relative','variables.relative.mean',
					] and (do):
					
					indices = model.parameters.indices
					indices = array([min(j for j in indices if indices[j] == i) for i in set(indices[i] for i in indices)])

					eps = 1e-20
					if attr in ['variables']:
						value = model.parameters(parameters)[indices]
					elif attr in ['variables.norm']:
						value = model.parameters(parameters)[indices]
						value = norm(value)/(value.size)
					elif attr in ['variables.relative']:
						value = model.parameters(parameters)[indices]
						_value = model.parameters(attributes['parameters'][0])[indices]
						value = abs((value-_value)/(_value+eps))
					elif attr in ['variables.relative.mean']:
						value = model.parameters(parameters)[indices]
						_value = model.parameters(attributes['parameters'][0])[indices]
						value = norm((value-_value)/(_value+eps))/(value.size)

				elif attr in ['objective']:
					value = abs(metric(model(parameters)))
				
				elif attr in [
					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
					'objective.ideal.state','objective.diff.state','objective.rel.state',
					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and ((status) and (not done)):
					value = default


				elif attr in [
					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
					'objective.ideal.state','objective.diff.state','objective.rel.state',
					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and (not ((status) and (not done))):


					defaults = Dictionary(
						state=metric.state,
						data={i: model.data[i].data for i in model.data},
						label=metric.label)

					tmp = Dictionary(
						state=metric.state,
						data={i: model.data[i].data for i in model.data if (not model.data[i].unitary)},
						label=metric.label)

					if attr in ['objective.ideal.noise','objective.diff.noise','objective.rel.noise']:
						tmp.update(dict())
					elif attr in ['objective.ideal.state','objective.diff.state','objective.rel.state']:						
						tmp.update(dict(data={i:False for i in tmp.data}))
					elif attr in ['objective.ideal.operator','objective.diff.operator','objective.rel.operator']:
						tmp.update(dict(state=False,data={i:False for i in tmp.data}))
				
					data = tmp.data
					state = metric.state
					label = metric.label

					state.__initialize__(data=tmp.state)
					label.__initialize__(state=state)

					model.__initialize__(data=data,state=state,label=label)

					metric.__initialize__(model=model,state=state,label=label)

					
					if attr in ['objective.ideal.noise','objective.ideal.state','objective.ideal.operator']:
						value = abs(metric(model(parameters)))
					elif attr in ['objective.diff.noise','objective.diff.state','objective.diff.operator']:
						value = abs((track['objective'][-1] - metric(model(parameters))))
					elif attr in ['objective.rel.noise','objective.rel.state','objective.rel.operator']:
						value = abs((track['objective'][-1] - metric(model(parameters)))/(track['objective'][-1]))


					data = defaults.data
					state = state
					label = label

					state.__initialize__(data=default.state)
					label.__initialize__(state=state)

					model.__initialize__(data=data,state=state,label=label)

					metric.__initialize__(model=model,state=state,label=label)
					
				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (not do):
					value = default

				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (do):
					
					if attr in ['hessian','hessian.eigenvalues','hessian.rank']:
						function = hessian(jit(lambda parameters: metric(model(parameters))))
					elif attr in ['fisher','fisher.eigenvalues','fisher.rank']:
						function = fisher(model,model.grad,shapes=(model.shape,(*parameters.shape,*model.shape)),hermitian=metric.state.hermitian,unitary=model.unitary)

					if attr in ['hessian','fisher']:
						value = function(parameters)

					elif attr in ['hessian.eigenvalues','fisher.eigenvalues']:
						value = sort(abs(eig(function(parameters),hermitian=True)))[::-1]
						value = value/maximum(value)
					elif attr in ['hessian.rank','fisher.rank']:
						value = sort(abs(eig(function(parameters),hermitian=True)))[::-1]
						value = value/maximum(value)
						value = nonzero(value,eps=50)
						# value = (argmax(abs(difference(value)/value[:-1]))+1) if value.size > 1 else 1

				elif attr in ["state.string","state.ndim","label.string","label.ndim"]:
					value = getattrs(metric,attr,default=default,delimiter=delim)

				elif attr in ["noise.string","noise.ndim","noise.locality"]:
					for i in model.data:
						if not i.unitary:
							value = getattrs(i,delim.join(attr.split(delim)[1:]),default=default,delimiter=delim)
							break

				elif attr in ["noise.parameters","noise.method"]:
					for i in model.parameters:
						if all((not j.unitary) for j in model.parameters[i].instance):
							value = getattrs(model.parameters[i],delim.join(attr.split(delim)[1:]),default=default,delimiter=delim)
							break
				
				elif attr in ["noise.scale","noise.tau"]:
					for i in model.parameters:
						if all((not j.unitary) for j in model.parameters[i].instance):
							value = getattrs(model.parameters[i].kwargs,delim.join(attr.split(delim)[1:]),default=default,delimiter=delim)
							break

				elif attr in []:
					value = [attr.split(delim)[0],delim.join(attr.split(delim)[1:])]
					value = [getattrs(model,i,default=default,delimiter=delim) for i in value]
					value = value[0]/value[1] if value[1] else value[0]

				elif attr not in attributes and hasattrs(optimizer,attr,delimiter=delim):
					value = getattrs(optimizer,attr,default=default,delimiter=delim)

				elif attr not in attributes and hasattrs(model,attr,delimiter=delim):
					value = getattrs(model,attr,default=default,delimiter=delim)

				track[attr][-1] = value

				if (not does[attr]) and (updates.get(attr) is not None):
					for i in range(len(track[attr])):
						track[attr][i] = updates[attr](i,attr,track,default)


		logging = ((len(attributes['iteration']) == 1) or 
			(hyperparameters['modulo']['log'] is None) or 
			(attributes['iteration'][-1]%hyperparameters['modulo']['log'] == 0)
			)

		if logging:

			msg = '\n'.join([
				'%d f(x) = %0.4e'%(
					attributes['iteration'][-1],
					track['objective'][-1],
				),
				'|x| = %0.4e\t\t|grad(x)| = %0.4e'%(
					norm(attributes['parameters'][-1])/
						 (attributes['parameters'][-1].size),
					norm(attributes['grad'][-1])/
						 (attributes['grad'][-1].size),
				),
				'\t\t'.join([
					'%s = %0.4e'%(attr,attributes[attr][-1])
					for attr in ['alpha','beta']
					if attr in attributes and len(attributes[attr])>0
					]),
				# 'attributes: \t %s'%({attr: attributes[attr][-1].dtype 
				# 	if isinstance(attributes[attr][-1],arrays) else type(attributes[attr][-1]) 
				# 	for attr in attributes}),
				# 'track: \t %s'%({attr: attributes[attr][-1].dtype 
				# 	if isinstance(attributes[attr][-1],arrays) else type(attributes[attr][-1]) 
				# 	for attr in attributes}),				
				# 'x\n%s'%(to_string(parameters.round(4))),
				# 'theta\n%s'%(to_string(model.parameters(parameters).round(4))),
				# 'U\n%s\nV\n%s'%(
				# 	to_string((model(parameters)).round(4)),
				# 	to_string((metric.label()).round(4))),
				])


			model.log(msg)


		return status


