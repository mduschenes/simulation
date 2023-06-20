#!/usr/bin/env python

# Import python modules
import os,sys,itertools
from copy import deepcopy
from functools import partial
from math import prod

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,vmap,vfunc,switch,forloop,slicing,gradient,hessian,fisher
from src.utils import array,asarray,empty,identity,ones,zeros,rand,prng,arange,diag
from src.utils import tensorprod,conjugate,dagger,einsum,dot,norm,eig,trace,sort,relsort
from src.utils import setitem,maximum,minimum,argmax,argmin,difference,cumsum,shift,abs,mod,sqrt,log,log10,sign,sin,cos,exp
from src.utils import to_string,is_hermitian,is_unitary,allclose
from src.utils import pi,e,nan,delim,scalars,arrays,datatype

from src.iterables import setter,getattrs,hasattrs,namespace,iterate,indexer,inserter

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

class Object(System):
	'''
	Base class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		parameters (object): parameter of operator
		state (object): state of operators
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {}
	default = None
	D = None
	N = None
	n = None

	hermitian = None
	unitary = None

	def __init__(self,data=None,operator=None,site=None,string=None,parameters=None,state=None,system=None,**kwargs):		

		defaults = dict(			
			shape=None,size=None,ndim=None,
			samples=None,identity=None,locality=None,index=None,
			conj=False,func=None,gradient=None
			)

		setter(kwargs,defaults,delimiter=delim,func=False)
		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,parameters=parameters,state=state,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		if self.func is None:
			
			def func(parameters=None,state=None,conj=False):
				return self.data if not conj else dagger(self.data)

			self.func = func

		if self.gradient is None:
			
			def gradient(parameters=None,state=None,conj=False):
				return 0*self.data

			self.gradient = gradient			

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
		locality = self.locality if self.locality is not None else self.N
		N = self.N	
		D = self.D	
		n = self.n

		if operator is None:
			operator = data
		
		if operator is None:
			pass
		elif isinstance(operator,str):
			if operator not in [default]:
				site = list(range(min(max(locality if locality is not None else 0,sum(basis[i].locality for i in operator.split(delim) if i in basis and basis[i].locality is not None)),N))) if site is None else site[:N] if not isinstance(site,int) else [site][:N]
				operator = [i for i in operator.split(delim)][:len(site)]
			elif operator in [default]:
				site = list(range(basis[operator].locality if N is None else N)) if site is None else site[:N] if not isinstance(site,int) else [site][:N]
				operator = operator
		elif not isinstance(operator,str) and not isinstance(operator,arrays):
			site = site[:N] if not isinstance(site,int) else [site][:N]
			operator = [j for i in operator for j in (i.split(delim) if isinstance(i,str) else i)][:len(site)]

		if site is None:
			pass
		elif isinstance(site,int):
			site = [site][:N]
		else:
			site = [j for i in site[:N] for j in ([i] if isinstance(i,int) else i)][:N]

		if string is None:
			pass
		elif not isinstance(string,str):
			string = str(string)

		if parameters is None:
			pass

		N = max(self.N,max(site)+1 if site is not None else self.N) if self.N is not None else max(site)+1 if site is not None else 0
		D = self.D if self.D is not None else data.size**(1/max(1,data.ndim*N)) if isinstance(data,arrays) else 1
		n = D**N if (N is not None) and (D is not None) else None

		shape = self.shape if self.shape is not None else data.shape if isinstance(data,arrays) else None
		size = self.size if self.size is not None else data.size if isinstance(data,arrays) else None
		ndim = self.ndim if self.ndim is not None else data.ndim if isinstance(data,arrays) else None
		dtype = self.dtype if self.dtype is not None else data.dtype if isinstance(data,arrays) else None

		if isinstance(operator,str):
			operator = [operator]*N
		if operator is None or isinstance(operator,arrays):
			pass
		elif len(operator) == N:
			site = list(range(N))
			operator = list((operator[i] for i in range(N)))
		elif site is not None and len(operator) == len(site):
			site = list(range(N)) if site is None else site
			operator = list((operator[site.index(i)%len(operator)] if i in site else default for i in range(N)))
		else:
			site = list(range(N)) if site is None else site

		null = (self.parameters is False) or (self.ndim == 0)

		self.data = data if data is not None else operator if operator is not None else None
		self.operator = operator if operator is not None else None
		self.site = site if site is not None else None
		self.string = string if string is not None else None
		self.system = system

		self.N = N
		self.D = D
		self.n = n

		self.shape = shape
		self.size = size
		self.ndim = ndim
		self.dtype = dtype

		if (not null) and (((self.data is not None) or (self.operator is not None)) and (not isinstance(self.data,arrays))):
			self.__setup__(data,operator,site,string,parameters)

		if null:
			self.data = None
		elif isinstance(self.operator,arrays):
			self.data = self.operator
			self.operator = [self.string]
		elif isinstance(self.data,arrays):
			pass
		elif self.operator is None:
			self.data = None
		elif self.operator is not None:
			self.data = tensorprod([self.basis.get(i)() for i in self.operator]) if all(i in self.basis for i in self.operator) else None

		self.operator = list((i for i in self.operator)) if self.operator is not None and not isinstance(self.operator,arrays) else self.operator
		self.site = list((i for i in self.site)) if self.site is not None else self.site
		self.string = str(self.string) if self.string is not None else self.string
		self.parameters = self.parameters if self.parameters is not None else self.parameters

		self.identity = tensorprod([self.basis.get(self.default)() for i in range(self.N)]) if (self.default in self.basis) else None
		
		self.locality = max(self.locality if self.locality is not None else 0,len(self.site) if self.site is not None else 0)
		self.index = self.index if self.index is not None else None

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

		self.__initialize__()
		
		self.norm()

		self.info()

		return

	def __initialize__(self,data=None,parameters=None,state=None,conj=False):
		'''
		Initialize operator
		Args:
			data (array): data
			parameters (array): parameters
			state (array): state
			conj (bool): conjugate
		'''

		if data is None:
			data = self.data
		elif data is False:
			data = None

		if parameters is None:
			parameters = self.parameters

		if state is None:
			state = self.state

		if conj is None:
			conj = self.conj
		
		hermitian = self.hermitian
		unitary = self.unitary


		self.data = data
		self.parameters = parameters			
		self.state = state
		self.conj = conj


		try:
			state = state()
			data = self(parameters,state,conj)
		except TypeError:
			state = None
			data = self.data

		if not hermitian and not unitary:
			hermitian = hermitian
			unitary = unitary
		elif data is None:
			hermitian = False
			unitary = False
		elif state is None:
			data = data
			hermitian = is_hermitian(data)
			unitary = is_unitary(data)
		elif state.ndim == 1:
			hermitian = False
			unitary = True
		elif state.ndim == 2:
			hermitian = True
			unitary = False

		self.shape = data.shape if isinstance(data,arrays) else None
		self.size = data.size if isinstance(data,arrays) else None
		self.ndim = data.ndim if isinstance(data,arrays) else None
		self.dtype = data.dtype if isinstance(data,arrays) else None

		self.hermitian = hermitian
		self.unitary = unitary

		return


	def __call__(self,parameters=None,state=None,conj=False):
		'''
		Call operator
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate			
		Returns:
			data (array): data
		'''

		if parameters is None:
			parameters = self.parameters

		return self.func(parameters,state,conj)

	def grad(self,parameters=None,state=None,conj=False):
		'''
		Call operator gradient
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate			
		Returns:
			data (array): data
		'''

		if parameters is None:
			parameters = self.parameters

		return self.gradient(parameters,state,conj)


	def __str__(self):
		string = self.__class__.__name__ if not self.string else self.string
		if self.operator is not None and not isinstance(self.operator,arrays):
			string = '%s'%(delim.join(self.operator))
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
		except TypeError as exception:
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

		if ndim > 3:
			if hermitian and not unitary:
				norm = None
				eps = None
			elif not hermitian and unitary:
				norm = einsum('...uij,...ukj->...ik',conjugate(data),data)
				eps = array([identity(shape[-2:],dtype=dtype)]*(norm.ndim-2),dtype=dtype)
			else:
				norm = None
				eps = None
		elif ndim == 3:
			if hermitian and unitary:
				norm = None
				eps = None
			elif not hermitian and unitary:
				norm = einsum('uij,ukj->ik',conjugate(data),data)
				eps = identity(shape[-2:],dtype=dtype)
			else:
				norm = None
				eps = None
		elif ndim == 2:
			if hermitian and not unitary:
				norm = einsum('ii->',data)
				eps = ones(shape[:-2],dtype=dtype)
			elif not hermitian and unitary:
				norm = einsum('ij,kj->ik',conjugate(data),data)
				eps = identity(shape[-2:],dtype=dtype)
			else:
				norm = None
				eps = None
		else:
			if not hermitian and unitary:
				norm = einsum('i,i->',conjugate(data),data)
				eps = ones(shape=(),dtype=dtype)
			else:
				norm = None
				eps = None

		if norm is None or eps is None:
			return

		assert (eps.shape == norm.shape), "Incorrect operator shape %r != %r"%(eps.shape,norm.shape)

		if dtype not in ['complex256','float128']:
			assert allclose(eps,norm), "Incorrect norm data%r: %r (hermitian: %r, unitary : %r)"%(eps.shape,norm,hermitian,unitary)

		return

	def swap(self,i,j):
		'''	
		Swap indices of object
		Args:
			i (int): Index to swap
			j (int): Index to swap
		'''

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
		parameters (object): parameter of operator		
		state (object): state of operators		
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

	def __new__(cls,data=None,operator=None,site=None,string=None,parameters=None,state=None,system=None,inherit=False,**kwargs):		

		# TODO: Allow multiple different classes to be part of one operator, and swap around localities

		self = None

		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,parameters=parameters,state=state,system=system),delimiter=delim,func=False)

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
		parameters (object): parameter of operator				
		state (object): state of operators				
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

	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''

		def sign(conj):
			return 1-2*conj

		def func(parameters=None,state=None,conj=False):
			return cos(sign(conj)*(1.0)*parameters)*self.identity + -1j*sin(sign(conj)*(1.0)*parameters)*self.data

		def gradient(parameters=None,state=None,conj=False):
			return sign(conj)*(1.0)*(-sin(sign(conj)*(1.0)*parameters)*self.identity + -1j*cos(sign(conj)*(1.0)*parameters)*self.data)

		if self.parameters is None:
			self.parameters = 1

		hermitian = False
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
		parameters (object): parameter of operator				
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I']},
		**{attr: Object(data=Basis['CNOT'](),D=2,locality=2,hermitian=True,unitary=True,string=attr) for attr in ['CNOT','C','cnot']},
		**{attr: Object(data=Basis['HADAMARD'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['HADAMARD','H']},
		**{attr: Object(data=Basis['TOFFOLI'](),D=2,locality=3,hermitian=True,unitary=True,string=attr) for attr in ['TOFFOLI','T','toffoli']},
		}
	default = 'I'
	D = 2 
	N = None
	n = None
	
	hermitian = None
	unitary = None
	
	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''
		
		hermitian = False
		unitary = True

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
		parameters (object): parameter of operator		
		state (object): state of operators		
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

	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''

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
		parameters (object): parameter of operator		
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I']},
		**{attr: Object(data=Basis['RANDOM'](ndim=1),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['random','psi','haar']},
		**{attr: Object(data=Basis['0'](),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['zero','zeros','0']},
		**{attr: Object(data=Basis['1'](),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['one','ones','1']},
		**{attr: Object(data=Basis['+'](),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['plus','+']},
		**{attr: Object(data=Basis['-'](),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['minus','-']},
		}
	default = 'I'
	D = 2
	N = None
	n = None

	hermitian = None
	unitary = None

	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''

		N = self.N
		D = self.D
		default = self.default
		shape = (self.D,)
		ndim = len(shape)
		size = prod(shape)
		random = getattr(self,'random','haar')
		seed = getattr(self,'seed',None)
		reset = getattr(self,'reset',None)
		dtype = self.dtype

		site = list(range(self.N)) if self.site is None else self.site if not isinstance(self.site,int) else [self.site]
		operator = None if self.operator is None else [self.operator[self.site.index(i)%len(self.operator)] if i in self.site else self.default for i in range(self.N)] if not isinstance(self.operator,str) else [self.operator]*self.N
		locality = len(operator)
		samples = self.samples if self.samples is not None else 1

		data = []

		for s in range(samples):
			
			datum = []
			
			for i in range(N):
				
				tmp = zeros(shape=shape,dtype=dtype)

				if operator[i] in ['zero','zeros','0']:
					tmp = setitem(tmp,0,1)
				elif operator[i] in ['one','ones','1']:
					tmp = setitem(tmp,-1,1)
				elif operator[i] in ['plus','+']:
					tmp = setitem(tmp,slice(None),1/sqrt(size))
				elif operator[i] in ['minus','-']:
					tmp = setitem(tmp,slice(None),(-1)**arange(size)/sqrt(size))
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
		parameters (object): parameter of operator		
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I','i']},
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['eps','noise','rand']},
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['depolarize']},
		**{attr: Object(data=Basis['I'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['amplitude']},
		**{attr: Object(data=Basis['00'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['00']},
		**{attr: Object(data=Basis['01'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['01']},
		**{attr: Object(data=Basis['10'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['10']},
		**{attr: Object(data=Basis['11'](),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['11']},
		**{attr: Object(data=Basis['X'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['X','x','flip','bitflip']},
		**{attr: Object(data=Basis['Y'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['Y','y','flipphase']},
		**{attr: Object(data=Basis['Z'](),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['Z','z','phase','dephase']},
		}
	default = 'I'
	D = 2
	N = None
	n = None
	
	hermitian = None
	unitary = None

	scale = 1
	tau = None
	initialization = None

	def __init__(self,data=None,operator=None,site=None,string=None,parameters=None,state=None,system=None,**kwargs):		

		defaults = dict(			
			shape=None,size=None,ndim=None,
			samples=None,identity=None,locality=None,index=None
			)

		setter(kwargs,defaults,delimiter=delim,func=False)
		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,parameters=parameters,state=state,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		return

	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''

		def operators(operator,parameters):
			'''
			Return operator
			Args:
				operator (str): Operator name
			Returns:
				data (iterable[array]): Operators for operator
				hermitian (bool): Operator is hermitian
				unitary (bool): Operator is unitary
			'''
			hermitian = True
			unitary = False
			
			if operator is None:
				data = [self.basis[self.default]()]
			elif operator in ['Z','z','phase','dephase']:
				data = [sqrt(1-parameters)*self.basis['I'](),
						sqrt(parameters)*self.basis['Z']()]
			elif operator in ['X','x','flip','bitflip']:
				data = [sqrt(1-parameters)*self.basis['I'](),
						sqrt(parameters)*self.basis['X']()]
			elif operator in ['Y','y','flipphase']:
				data = [sqrt(1-parameters)*self.basis['I'](),
						sqrt(parameters)*self.basis['Y']()]												
			elif operator in ['amplitude']:
				data = [self.basis['00']() + sqrt(1-parameters)*self.basis['11'](),
						sqrt(parameters)*self.basis['01']()]
			elif operator in ['depolarize']:
				data = [sqrt(1-parameters)*self.basis['I'](),
						sqrt(parameters/(self.D**2-1))*self.basis['X'](),
						sqrt(parameters/(self.D**2-1))*self.basis['Y'](),
						sqrt(parameters/(self.D**2-1))*self.basis['Z']()]
			elif operator in ['eps']:
				data = array([identity(self.n,dtype=self.dtype),diag((1+parameters)**(arange(self.n)+2) - 1)])
				hermitian = False
				unitary = False
			elif operator in ['noise','rand']:
				data = array(parameters,dtype=datatype(self.dtype))#[identity(self.n),diag((1+parameters)**(arange(self.n)+2) - 1)])
				seed = prng(reset=self.seed)
				hermitian = False
				unitary = False
			else:
				data = [self.basis[self.default]()]

			return data,hermitian,unitary


		if (getattr(self,'scale',None) is not None):
			if (getattr(self,'initialization',None) in ['time']):
				if (getattr(self,'tau') is not None):
					self.parameters = (1 - exp(-self.tau/self.scale))/2

		do = (self.parameters is None)

		if do:
			self.data = None
			self.operator = None
			return

		N = self.N
		default = self.default
		site = list(range(self.N)) if self.site is None else self.site if not isinstance(self.site,int) else [self.site]
		operator = None if self.operator is None else [self.operator[self.site.index(i)%len(self.operator)] if i in self.site else self.default for i in range(self.N)] if not isinstance(self.operator,str) else [self.operator]*self.N
		locality = len(operator)
		parameters = [None]*self.N if self.parameters is None else [self.parameters[i] if i in self.site else self.default for i in range(self.N)] if not isinstance(self.parameters,scalars) else [self.parameters]*self.N

		hermitian = True
		unitary = False

		data = []

		assert ((isinstance(parameters,scalars) and (parameters >= 0) and (parameters <= 1)) or (all((i>=0) and (i<=1) for i in parameters))), "Noise scale %r not in [0,1]"%(parameters)

		for i in range(N):
			datum,hermitian,unitary = operators(operator[i],parameters[i])
			
			if isinstance(datum,arrays):
				data = datum
				break

			data.append(datum)

		if not isinstance(data,arrays):
			data = array([tensorprod(i)	for i in itertools.product(*data)],dtype=self.dtype)

		self.data = data
		self.hermitian = hermitian
		self.unitary = unitary

		return


class Operators(Object):
	'''
	Class for Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,parameters dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (iterable[str]): string labels of operators
		parameters (iterable[str],dict,Parameters): parameters of operators		
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
		state (str,dict,State): Type of state	
		noise (str,dict,Noise): Type of noise
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	def __init__(self,data=None,operator=None,site=None,string=None,parameters=None,
		N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
		space=None,time=None,lattice=None,state=None,noise=None,system=None,**kwargs):

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

		self.data = []
		self.parameters = parameters

		self.n = None
		self.g = None
		
		self.state = state
		self.noise = noise
		self.identity = None
		self.coefficients = None
		self.constants = None
		self.conj = False

		self.func = None
		self.gradient = None
		self.gradient_finite = None
		self.gradient_analytical = None
		self.trotterize = None

		self.system = system

		self.__time__()
		self.__space__()
		self.__lattice__()

		self.identity = Operator(Operator.default,N=self.N,D=self.D,system=self.system,verbose=False)
		self.coefficients = array((self.tau)*(1/self.P),dtype=datatype(self.dtype))

		self.shape = () if self.n is None else (self.n,self.n)
		self.size = prod(self.shape)
		self.ndim = len(self.shape)

		self.__setup__(data,operator,site,string,parameters)

		self.__initialize__()

		self.info()

		return	

	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,parameters dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
				string (iterable[str]): string labels of operators
				parameters (iterable[object]): parameters of operators
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
		'''

		# Get operator,site,string from data
		objs = {'operator':operator,'site':site,'string':string,'parameters':parameters}

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
			data = None

		# Set class attributes
		self.__extend__(data=data,**objs)

		return


	def __append__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Append to class
		Args:
			data (str,Operator): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''
		index = -1
		self.__insert__(index,data,operator,site,string,parameters)
		return

	def __extend__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup class
		Args:
			data (iterable[str,Operator]): data of operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
		'''

		size = min([len(i) for i in [data,operator,site,string,parameters] if i is not None],default=0)

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
		if parameters is None:
			parameters = [None]*size			

		for _data,_operator,_site,_string,_parameters in zip(data,operator,site,string,parameters):
			self.__append__(_data,_operator,_site,_string,_parameters)

		return


	def __insert__(self,index,data,operator,site,string,parameters):
		'''
		Insert to class
		Args:
			index (int): index to insert operator
			data (str,Operator): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''

		if index == -1:
			index = len(self)

		kwargs = dict(N=self.N,D=self.D,system=self.system,verbose=False)

		data = Operator(data=data,operator=operator,site=site,string=string,parameters=parameters,**kwargs)
		self.data.insert(index,data)

		return

	def __initialize__(self,parameters=None,state=None,noise=None):
		''' 
		Setup class functions
		Args:
			parameters (bool,dict,array,Parameters): Class parameters
			state (bool,dict,array,State): State to act on with class of shape self.shape, or class hyperparameters, or boolean to choose self.state or None
			noise (bool,dict,array,Noise): Noise to act on with class of shape (-1,self.shape), or class hyperparameters, or boolean to choose self.noise or None
		'''

		parameters = self.parameters if parameters is None else parameters
		state = self.state if state is None else state
		noise = self.noise if noise is None else noise

		objs = {'parameters':parameters,'state':state,'noise':noise}
		classes = {'parameters':Parameters,'state':State,'noise':Noise}
		arguments = {'parameters':False,'state':True,'noise':True}

		# Get functions
		for obj in objs:
			data,cls,argument = objs[obj],classes[obj],arguments[obj]
			data = getattr(self,obj,None) if data is None or data is True else data if data is not False else None
			if not isinstance(data,cls):
				kwargs = {}

				args = ({**dict(data=None),**data} if (isinstance(data,dict) and (argument or all(attr in data for attr in dict(data=None)))) else dict(data=data))
				setter(kwargs,args,func=False)

				args = dict(**namespace(cls,self),model=self,system=self.system)
				setter(kwargs,args,func=False)

				args = dict(verbose=False)
				setter(kwargs,args,func=True)

				instance = cls(**kwargs)

			else:

				instance = data

			setattr(self,obj,instance)
				


		# Set functions
		identity = self.identity()
		parameters = self.parameters()
		state = self.state()
		noise = self.noise()
		coefficients = self.coefficients
		constants = self.constants
		conj = self.conj

		parameters = self.parameters(parameters)

		assert parameters is not None, "Incorrect parameters() initialization"

		data = trotter([jit(i) for i in self.data],self.P)
		grad = trotter([jit(i.grad) for i in self.data],self.P)
		slices = trotter(list(range(len(self))),self.P)
		trotterize = jit(lambda parameters,slices: parameters[slices].T.ravel(),slices=array(slices))

		if state is None and noise is None:
			hermitian = False
			unitary = True
			shape = identity.shape
		elif state is None and noise is not None:
			hermitian = False
			unitary = False
			shape = identity.shape
		elif state.ndim == 1 and noise is not None:
			hermitian = True
			unitary = False
			shape = state.shape
		elif state.ndim == 2 and noise is not None:
			hermitian = True
			unitary = False
			shape = state.shape
		elif state.ndim == 1 and noise is None:
			hermitian = False
			unitary = True
			shape = state.shape
		elif state.ndim == 2 and noise is None:
			hermitian = True
			unitary = False
			shape = state.shape			
		else:
			raise NotImplementedError

		parameters = trotterize(self.coefficients*parameters)

		func = scheme(parameters=parameters,state=state,conj=conj,data=data,identity=identity,constants=constants,noise=noise)
		
		grad_jax = gradient(self,mode='fwd',move=True)
		grad_finite = gradient(self,mode='finite',move=True)
		grad_analytical = gradient_scheme(parameters=parameters,state=state,conj=conj,data=data,identity=identity,constants=constants,noise=noise,grad=grad)

		grad = grad_jax

		# Update class attributes
		self.func = func
		self.gradient = grad
		self.gradient_finite = grad_finite
		self.gradient_analytical = grad_analytical
		self.trotterize = trotterize
		self.hermitian = hermitian
		self.unitary = unitary

		self.shape = shape
		self.size = prod(self.shape)
		self.ndim = len(self.shape)


		return

	def __call__(self,parameters=None,state=None,conj=False):
		'''
		Class function
		Args:
			parameters (array): parameters		
			state (obj): state
			conj (bool): conjugate
		Returns
			out (array): Return of function
		'''

		if parameters is None:
			parameters = self.parameters()

		if state is None:
			state = self.state()

		parameters = self.trotterize(self.coefficients*self.parameters(parameters))

		return self.func(parameters,state,conj)

	def grad(self,parameters=None,state=None,conj=False):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate
		Returns:
			out (array): Return of function
		'''		
		return self.gradient(parameters)

	def grad_finite(self,parameters=None,state=None,conj=False):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate
		Returns:
			out (array): Return of function
		'''		
		return self.gradient_finite(parameters)		

	def grad_analytical(self,parameters=None,state=None,conj=False):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate
		Returns:
			out (array): Return of function
		'''		
		return self.gradient_analytical(parameters)


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
			string = '%s: %s'%(attr,getattrs(self,attr,delimiter=delim,default=None))
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

		for attr in ['parameters','state','noise']:
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


		# for attr in ['model']:
		# 	string = '%s :\n%s'%(delim.join(attr.split(delim)[:1]),to_string(self()))

		# 	msg.append(string)			

		# for attr in ['parameters','state','noise']:
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
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,parameters dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (iterable[str]): string labels of operators
		parameters (iterable[object]): parameters of operators
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
		parameters (str,dict,Parameters): Type of parameters	
		state (str,dict,State): Type of state	
		noise (str,dict,Noise): Type of noise
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	def __init__(self,data=None,operator=None,site=None,string=None,parameters=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
				space=None,time=None,lattice=None,state=None,noise=None,system=None,**kwargs):

		super().__init__(data=data,operator=operator,site=site,string=string,parameters=parameters,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
				space=space,time=time,lattice=lattice,state=state,noise=noise,system=system,**kwargs)
		
		return

	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,parameters dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
				string (iterable[str]): string labels of operators
				parameters (iterable[object]): parameters of operators
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
		'''

		# Get operator,site,string,parameters from data
		objs = {'operator':operator,'site':site,'string':string,'parameters':parameters}

		for obj in objs:
			objs[obj] = [] if objs[obj] is None else objs[obj]

		# Set attributes
		attr = 'parameters'
		objs.pop(attr);

		# Get data
		if data is None:
			data = None
		elif all(isinstance(datum,Object) for datum in data):
			for obj in objs:
				objs[obj] = None
		elif isinstance(data,dict) and all(isinstance(data[name],dict) and (obj in data[name]) and (data[name][obj] is not None) for name in data for obj in objs):
			for obj in objs:
				objs[obj].extend([data[name][obj] for name in data])
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

			if isinstance(tmp[attr],scalars) and tmp[attr] in indices:
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

			else:
				for obj in objs:
					value = tmp[obj]
					objs[obj].append(value)	


		# Set class attributes
		self.__extend__(data=data,**objs)

		return



class Unitary(Hamiltonian):
	'''
	Unitary class of Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,parameters dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
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
		parameters (str,dict,Parameters): Type of parameters	
		state (str,dict,State): Type of state	
		noise (str,dict,Noise): Type of noise
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	def __init__(self,data=None,operator=None,site=None,string=None,parameters=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
				space=None,time=None,lattice=None,state=None,noise=None,system=None,**kwargs):
		
		super().__init__(data=data,operator=operator,site=site,string=string,parameters=parameters,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
				space=space,time=time,lattice=lattice,state=state,noise=noise,system=system,**kwargs)

		self.norm()

		return

	def grad_analytical(self,parameters=None,state=None,conj=False):
		'''
		Class gradient
		Args:
			parameters (array): parameters		
			state (obj): state
			conj (bool): conjugate
		Returns
			out (array): Return of function
		'''	

		if parameters is None:
			parameters = self.parameters()

		if state is None:
			state = self.state()

		parameters = self.coefficients*self.parameters(parameters)

		shape = parameters.shape
		ndim = parameters.ndim
		
		parameters = self.trotterize(parameters)

		gradient_trotterize = jit(lambda grad,P=self.P: gradient_trotter(grad,P))
		indices = self.parameters.indices
		reshape = (*shape[1:],-1,*self.shape)
		transpose = (ndim-1,*range(0,ndim-1),*range(ndim,ndim+self.ndim))
		shapes = (-1,*self.shape)

		grad = self.coefficients*self.gradient_analytical(parameters)
		grad = grad.reshape(reshape)
		grad = grad.transpose(transpose)
		grad = gradient_trotterize(grad)
		grad = grad[indices]
		grad = grad.reshape(*shapes)

		return grad



class Channel(Unitary):
	'''
	Channel class of Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,parameters dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
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
		parameters (str,dict,Parameters): Type of parameters	
		state (str,dict,State): Type of state	
		noise (str,dict,Noise): Type of noise
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''
	def __init__(self,data=None,operator=None,site=None,string=None,parameters=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
				space=None,time=None,lattice=None,state=None,noise=None,system=None,**kwargs):
		
		super().__init__(data=data,operator=operator,site=site,string=string,parameters=parameters,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
				space=space,time=time,lattice=lattice,state=state,noise=noise,system=system,**kwargs)

		return


class Label(Operator):
	
	basis = {}
	default = None
	D = None
	N = None
	n = None

	hermitian = None
	unitary = None

	state = None

	def __new__(cls,*args,**kwargs):

		self = super().__new__(cls,*args,**kwargs)

		return self

	def __init__(self,*args,**kwargs):
		return

	def __call__(self,parameters=None,state=None,conj=False):
		'''
		Call operator
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate			
		Returns:
			data (array): data
		'''

		if parameters is None:
			parameters = self.parameters

		if state is None and self.state is not None:
			state = self.state()

		data = self.func(parameters,state,conj)

		if data is None:
			data = None
		elif state is None:
			data = data
		elif state.ndim == 1:
			data = einsum('ij,j->i',data,state)
		elif state.ndim == 2:
			data = einsum('ij,jk,lk->il',data,state,conjugate(data))

		return data
		

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

			'noise.scale':[],'hyperparameters.c1':[],'hyperparameters.c2':[],

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


					defaults = Dictionary(state=model.state,noise=model.noise,label=metric.label)

					tmp = Dictionary(state=model.state,noise=model.noise,label=metric.label)

					if attr in ['objective.ideal.noise','objective.diff.noise','objective.rel.noise']:
						tmp.update(dict())
					elif attr in ['objective.ideal.state','objective.diff.state','objective.rel.state']:						
						tmp.update(dict(noise=False))
					elif attr in ['objective.ideal.operator','objective.diff.operator','objective.rel.operator']:
						tmp.update(dict(state=False,noise=False))
						tmp.update({kwarg : False for kwarg in ['state','noise']})

				
					label = metric.label

					model.__initialize__(state=tmp.state,noise=tmp.noise)

					label.__initialize__(state=model.state)

					metric.__initialize__(model=model,label=label)

					
					if attr in ['objective.ideal.noise','objective.ideal.state','objective.ideal.operator']:
						value = abs(metric(model(parameters)))
					elif attr in ['objective.diff.noise','objective.diff.state','objective.diff.operator']:
						value = abs((track['objective'][-1] - metric(model(parameters))))
					elif attr in ['objective.rel.noise','objective.rel.state','objective.rel.operator']:
						value = abs((track['objective'][-1] - metric(model(parameters)))/(track['objective'][-1]))


					model.__initialize__(state=defaults.state,noise=defaults.noise)

					label.__initialize__(state=defaults.state)

					metric.__initialize__(model=model,label=label)

					
				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (not do):
					value = default

				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (do):
					
					if attr in ['hessian','hessian.eigenvalues','hessian.rank']:
						function = hessian(jit(lambda parameters: metric(model(parameters))))
					elif attr in ['fisher','fisher.eigenvalues','fisher.rank']:
						function = fisher(model,model.grad,shapes=(model.shape,(*parameters.shape,*model.shape)))

					if attr in ['hessian','fisher']:
						value = function(parameters)

					elif attr in ['hessian.eigenvalues','fisher.eigenvalues']:
						value = sort(abs(eig(function(parameters),hermitian=True)))[::-1]
						# if model.state.ndim == 1:
						# 	print(model.state(),1-dot(dagger(model.state()),model.state()))
						# elif model.state.ndim == 2:
						# 	print(model.state(),1-trace(model.state()),1-trace(dot(dagger(model.state()),model.state())))
						# print(model.noise())
						print(value)
						value = value/maximum(value)
						print(value)
						exit()
					elif attr in ['hessian.rank','fisher.rank']:
						value = sort(abs(eig(function(parameters),hermitian=True)))[::-1]
						value = value/maximum(value)
						value = (argmax(abs(difference(value)/value[:-1]))+1) if value.size > 1 else 1

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


def trotter(iterable,p):
	'''
	Trotterized iterable with order p
	Args:
		iterable (iterable): Iterable
		p (int): Order of trotterization
	Returns:
		iterable (iterable): Trotterized iterable
	'''
	slices = [slice(None,None,1),slice(None,None,-1)]

	if p > len(slices):
		raise NotImplementedError("p = %d Not Implemented"%(p))

	i = []        

	for indices in slices[:p]:
		i += iterable[indices]
	
	return i


def gradient_trotter(iterable,p):
	'''
	Gradient of trotterized iterable with order p
	Args:
		iterable (iterable): Iterable
		p (int): Order of trotterization
	Returns:
		iterable (iterable): Gradient of trotterized iterable
	'''	
	slices = [slice(None,None,1),slice(None,None,-1)]

	if p > len(slices):
		raise NotImplementedError("p = %d Not Implemented"%(p))

	n = len(iterable)//p

	i = 0        

	for indices in slices[:p]:
		i += iterable[:n][indices] if indices.step > 0 else iterable[-n:][indices]
	
	return i


def contraction(data,state=None,conj=False,constants=None,noise=None):
	'''
	Contract data and state
	Args:
		data (array): Array of data of shape (n,n)
		state (array): state of shape (n,) or (n,n)
		conj (bool): conjugate
		constants (array): Array of constants to act of shape (n,n)
		noise (array): Array of noise to act of shape (...,n,n)
	Returns:
		func (callable): contracted data and state with signature func(data,state,conj)
	'''

	if constants is None and noise is None:
		
		if state is None:
			
			state = data

			subscripts = 'ij,jk->ik'
			shapes = (data.shape,state.shape)
			einsummation = einsum(subscripts,*shapes)

			def func(data,state,conj):
				return einsummation(data,state)

		elif state.ndim == 1:
			
			subscripts = 'ij,j...->i...'
			shapes = (data.shape,state.shape)
			einsummation = einsum(subscripts,*shapes)
			
			def func(data,state,conj):
				return einsummation(data,state)

		elif state.ndim == 2:
			
			subscripts = 'ij,jk,lk->il'
			shapes = (data.shape,state.shape,data.shape)
			einsummation = einsum(subscripts,*shapes)
			
			def func(data,state,conj):
				return einsummation(data,state,conjugate(data))

	elif constants is not None and noise is None:

		raise NotImplementedError("TODO: Implement state == Any, constants != None, noise == None scheme")
		
	elif constants is None and noise is not None:

		if state is None:

			if noise.ndim == 3:

				state = data

				subscripts = 'uij,jk,kl->il'
				shapes = (noise.shape,data.shape,state.shape)
				einsummation = einsum(subscripts,*shapes)

				def func(data,state,conj):
					return einsummation(noise,data,state)	

			elif noise.ndim == 0:

				state = data

				subscripts = 'ij,jk->ik'
				shapes = (data.shape,state.shape)
				einsummation = einsum(subscripts,*shapes)

				def func(data,state,conj):
					return einsummation(data,state) + noise*rand(state.shape,random='uniform',bounds=[-1,1],seed=None,dtype=noise.dtype)/2

		elif state.ndim == 1:
		
			if noise.ndim == 3:

				subscripts = 'uij,jk,k...->i...'
				shapes = (noise.shape,data.shape,state.shape)
				einsummation = einsum(subscripts,*shapes)

				def func(data,state,conj):
					return einsummation(noise,data,state)	

			elif noise.ndim == 0:

				subscripts = 'ij,j...->i...'
				shapes = (data.shape,state.shape)
				einsummation = einsum(subscripts,*shapes)

				def func(data,state,conj):
					return einsummation(data,state) + noise*rand(state.shape,random='uniform',bounds=[-1,1],seed=None,dtype=noise.dtype)/2


		elif state.ndim == 2:

			if noise.ndim == 3:

				subscripts = 'uij,jk,kl,ml,unm->in'
				shapes = (noise.shape,data.shape,state.shape,data.shape,noise.shape)
				einsummation = einsum(subscripts,*shapes)

				def func(data,state,conj):
					return einsummation(noise,data,state,conjugate(data),conjugate(noise))	

			elif noise.ndim == 0:

				subscripts = 'ij,jk,lk->il'
				shapes = (data.shape,state.shape,data.shape)
				einsummation = einsum(subscripts,*shapes)

				def func(data,state,conj):
					return einsummation(data,state,conjugate(data)) + noise*rand(state.shape,random='uniform',bounds=[-1,1],seed=None,dtype=noise.dtype)/2	


	elif constants is not None and noise is not None:

		raise NotImplementedError("TODO: Implement state == Any, constants != None, noise != None scheme")
		
	func = jit(func,static_argnums=(2))	

	return func


def scheme(parameters,state=None,conj=False,data=None,identity=None,constants=None,noise=None):
	'''
	Contract data and state
	Args:
		parameters (array): parameters of shape (size,)
		state (array): state of shape (n,) or (n,n)
		conj (bool): conjugate
		data (array): data of shape (length,)
		identity (array): Array of data identity of shape (n,n)
		constants (array): Array of constants to act of shape (n,n)
		noise (array): Array of noise to act of shape (...,n,n)
	Returns:
		func (callable): contracted data(parameters) and state with signature func(parameters,state,conj)
	'''

	size = parameters.shape[0] if parameters is not None else 1
	length = len(data) if data is not None else 1

	contract = contraction(identity,state=state,conj=conj,constants=constants,noise=noise)

	def index(i,size,conj):
		return (size-1)*conj + (1-2*conj)*(i%size)

	if constants is None and noise is None:
		
		if state is None:
			
			state = identity

			def func(parameters,state=state,conj=conj,contract=contract,index=index):
				def func(i,out):
					# U = switch(i%length,data,parameters[i])					
					U = switch(index(i,length,conj=conj),data,parameters[index(i,size,conj=conj)],out,conj)
					return contract(U,out,conj)

				out = identity
				return forloop(0,parameters.size,func,out)				
	
		elif state.ndim == 1:
			
			def func(parameters,state=state,conj=conj,contract=contract,index=index):
				def func(i,out):
					U = switch(index(i,length,conj=conj),data,parameters[index(i,size,conj=conj)],out,conj)
					return contract(U,out,conj)
				out = state
				return forloop(0,parameters.size,func,out)				


		elif state.ndim == 2:
			
			def func(parameters,state=state,conj=conj,contract=contract,index=index):
				def func(i,out):
					U = switch(index(i,length,conj=conj),data,parameters[index(i,size,conj=conj)],out,conj)
					return contract(U,out,conj)
				out = state
				return forloop(0,parameters.size,func,out)				

	elif constants is not None and noise is None:

		raise NotImplementedError("TODO: Implement state == Any, constants != None, noise == None scheme")

	elif constants is None and noise is not None:

		if state is None:

			state = identity

			def func(parameters,state=state,conj=conj,contract=contract,index=index):
				def func(i,out):
					U = switch(index(i,length,conj=conj),data,parameters[index(i,size,conj=conj)],out,conj)
					# U = switch(i%length,data,parameters[i])
					return contract(U,out,conj)

				out = identity
				return forloop(0,parameters.size,func,out)	

		elif state.ndim == 1:

			def func(parameters,state=state,conj=conj,contract=contract,index=index):
				def func(i,out):
					U = switch(index(i,length,conj=conj),data,parameters[index(i,size,conj=conj)],out,conj)
					return contract(U,out,conj)

				out = state
				return forloop(0,parameters.size,func,out)	

		elif state.ndim == 2:

			subparameters = slicing(parameters,0,length)
			substate = None
			subconj = conj
			subdata = data
			subidentity = identity
			subconstants = constants
			subnoise = None

			subscheme = scheme

			subfunc = subscheme(subparameters,state=substate,conj=subconj,data=subdata,identity=subidentity,constants=subconstants,noise=subnoise)

			def func(parameters,state=state,conj=conj,contract=contract,index=index):
				def func(i,out):
					x = slicing(parameters,i*length,length)
					U = subfunc(x,identity,conj)
					return contract(U,out,conj)

				out = state
				return forloop(0,parameters.size//length,func,out)	


	elif constants is not None and noise is not None:

		raise NotImplementedError("TODO: Implement state == Any, constants != None, noise != None scheme")
		
	func = jit(func,static_argnums=(2))

	return func			


def gradient_scheme(parameters,state=None,conj=False,data=None,identity=None,constants=None,noise=None,grad=None):
	'''
	Contract data and state
	Args:
		parameters (array): parameters of shape (size,)
		state (array): state of shape (n,) or (n,n)
		conj (bool): conjugate
		data (array): data of shape (length,)
		identity (array): Array of data identity of shape (n,n)
		constants (array): Array of constants to act of shape (n,n)
		noise (array): Array of noise to act of shape (...,n,n)
		grad (array): data of shape (length,)
	Returns:
		func (callable): contracted data(parameters) and state with signature func(parameters,state,conj)
	'''

	size = parameters.shape[0]
	length = len(data)

	subscripts = 'ij,jk,kl->il'
	shapes = (identity.shape,identity.shape,identity.shape)
	einsummation = einsum(subscripts,*shapes)

	if state is not None or noise is not None:
		return None
		raise NotImplementedError("TODO: Implement gradients for non-unitary contraction")

	if grad is not None:
		grad = [(lambda parameters,state=None,conj=False,i=i: (1.0)*data[i%length](parameters[i] + (pi/2))) for i in range(size)]


	def func(parameters=None,state=None,conj=False):
		def func(i):

			subparameters = slicing(parameters,0,i)
			substate = None
			subconj = conj
			subdata = data
			subidentity = identity
			subconstants = constants
			subnoise = None

			subscheme = scheme

			subfunc = subscheme(subparameters,state=substate,conj=subconj,data=subdata,identity=subidentity,constants=subconstants,noise=subnoise)

			U = subfunc(subparameters,identity,subconj)


			subparameters = slicing(parameters,i+1,size-i-1)
			substate = None
			subconj = conj
			subdata = shift(data,-((i+1)%length))
			subidentity = identity
			subconstants = constants
			subnoise = None

			subscheme = scheme

			subfunc = subscheme(subparameters,state=substate,conj=subconj,data=subdata,identity=subidentity,constants=subconstants,noise=subnoise)

			V = subfunc(subparameters,identity,subconj)

			A = grad[i](parameters,state,conj)

			return einsummation(V,A,U)

		return array([func(i) for i in range(parameters.size)])

	func = jit(func)

	return func
	