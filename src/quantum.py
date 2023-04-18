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

from src.utils import jit,vmap,vfunc,switch,gradient,hessian,fisher
from src.utils import array,asarray,empty,identity,ones,zeros,arange,rand,setitem
from src.utils import tensorprod,dagger,conj,einsum,dot
from src.utils import summation,exponentiation,summationv,exponentiationv,summationm,exponentiationm,summationmvc,exponentiationmvc,summationmmc,exponentiationmmc
from src.utils import forloop,trotter,gradient_trotter,gradient_expm
from src.utils import eig
from src.utils import maximum,minimum,argmax,argmin,difference,abs,sqrt,log,log10,sign,sin,cos
from src.utils import sort,relsort,norm
from src.utils import initialize,parse,to_string,allclose
from src.utils import pi,e,nan,null,delim,scalars,arrays,nulls,namespace
from src.utils import itg,flt,dbl

from src.iterables import setter,getter,getattrs,hasattrs,indexer,inserter

from src.io import load,dump,join,split

from src.system import System,Space,Time,Lattice

from src.parameters import Parameters

from src.optimize import Objective,Metric

class Object(System):
	'''
	Base class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
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

	hermitian = False
	unitary = False

	def __init__(self,data=None,operator=None,site=None,string=None,parameters=None,state=None,system=None,**kwargs):		

		defaults = dict(			
			shape=None,size=None,ndim=None,
			samples=None,identity=None,locality=None,index=None
			)

		setter(kwargs,defaults,delimiter=delim,func=False)
		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,parameters=parameters,state=state,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		if isinstance(data,self.__class__) or isinstance(operator,self.__class__):
			if isinstance(data,self.__class__):
				for attr in data:
					setattr(self,attr,getattr(data,attr))
			if isinstance(operator,self.__class__):
				for attr in operator:
					setattr(self,attr,getattr(operator,attr))					

			return
		
		elif (data is None) and (operator is None):
			return
		
		elif isinstance(data,dict):
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
				site = list(range(max(locality if locality is not None else 0,sum(basis[i].locality for i in operator.split(delim) if i in basis and basis[i].locality is not None),N))) if site is None else site
				operator = [i for i in operator.split(delim)]
			elif operator in [default]:
				site = list(range(basis[operator].locality if N is None else N)) if site is None else site
				operator = operator
		elif not isinstance(operator,str) and not isinstance(operator,arrays):
			operator = [j for i in operator for j in (i.split(delim) if isinstance(i,str) else i)]

		if site is None:
			pass
		elif isinstance(site,int):
			site = [i]
		else:
			site = [j for i in site for j in ([i] if isinstance(i,int) else i)]

		if string is None:
			pass
		elif not isinstance(string,str):
			string = str(string)

		if parameters is None:
			pass

		N = max(self.N,max(site)+1 if site is not None else self.N) if self.N is not None else max(site)+1 if site is not None else 0
		D = self.D if self.D is not None else data.size**(1/(data.ndim*N)) if isinstance(data,arrays) else 1
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

		if not (self.parameters is False) and (((self.data is not None) or (self.operator is not None)) and (not isinstance(self.data,arrays))):
			self.__setup__(data,operator,site,string,parameters)

		if self.parameters is False:
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
		
		if (self.samples is True) and (self.size is not None):
			shape,bounds,scale,seed,dtype = self.size, [0,1], 'normalize', self.seed, datatype(self.dtype)
			self.samples = rand(size,bounds=bounds,scale=scale,seed=seed,dtype=dtype)
		elif not isinstance(self.samples,arrays):
			self.samples = None

		if isinstance(self.data,arrays) and (self.samples is not None) and (self.ndim is not None):
			if (self.data.ndim>self.ndim):
				self.data = einsum('%s...,%s->...'%((''.join(['i','j','k','l'][:self.data.ndim-self.ndim]),)*2),self.samples,self.data)

		self.data = self.data.astype(self.dtype) if self.data is not None else None
		self.identity = self.identity.astype(self.dtype) if self.identity is not None else None

		self.shape = self.data.shape if isinstance(self.data,arrays) else None
		self.size = self.data.size if isinstance(self.data,arrays) else None
		self.ndim = self.data.ndim if isinstance(self.data,arrays) else None
		self.dtype = self.data.dtype if isinstance(self.data,arrays) else None

		self.norm()

		self.info()

		return

	def __call__(self,parameters=None,state=None,conj=None):
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

		return self.data

	def grad(self,parameters=None,state=None,conj=None):
		'''
		Call gradient
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate			
		Returns:
			data (array): data
		'''
		return zeros(parameters.size,dtype=parameters.dtype)

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
	
	def grad(self,parameters=None,state=None,conj=None):
		'''
		Call gradient
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate
		Returns:
			data (array): data
		'''
		return 0

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		
		msg = '%s'%('\n'.join([
			*['%s %s %s: %s'%(self.__class__.__name__.capitalize(),self,attr,getattr(self,attr)) 
				for attr in ['shape']
			],
			]
			))
		self.log(msg,verbose=verbose)
		return

	def norm(self):
		'''
		Normalize class
		Args:
			data (array): Data to normalize			
		'''

		# TODO: Efficient norm calculation

		try:
			data = self()
			if data is None:
				raise
		except:
			return

		# dataH = self(conj=True)
		dataH = conj(data)


		shape = self.shape
		ndim = self.ndim
		dtype = self.dtype
		hermitian = self.hermitian		
		unitary = self.unitary	

		if ndim > 3:
			normalization = einsum('...uij,...ukj->...ik',dataH,data)
			eps = array([identity(shape[-2:],dtype=dtype)]*(normalization.ndim-2),dtype=dtype)
		elif ndim == 3:
			normalization = einsum('uij,ukj->ik',dataH,data)
			eps = identity(shape[-2:],dtype=dtype)
		elif ndim == 2:
			if hermitian and not unitary:
				normalization = einsum('ii->',data)
				eps = ones(shape[:-2],dtype=dtype)
			elif not hermitian and unitary:
				normalization = einsum('ij,kj->ik',dataH,data)
				eps = identity(shape[-2:],dtype=dtype)
			else:
				normalization = None
				eps = None
		else:
			normalization = einsum('i,i->',dataH,data)
			eps = ones(shape=(),dtype=dtype)

		if normalization is None or eps is None:
			return

		assert (eps.shape == normalization.shape), "Incorrect operator shape %r != %r"%(eps.shape,normalization.shape)

		if dtype not in ['complex256','float128']:
			assert allclose(eps,normalization), "Incorrect normalization data%r: %r"%(eps.shape,normalization)

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
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
		string (str): string label of operator
		parameters (object): parameter of operator		
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		inherit (boolean): Inherit super class when initialized
		kwargs (dict): Additional system keyword arguments	
	'''
	qualname = __qualname__

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I','i']},
			}
	default = 'I'
	D = 2
	N = None
	n = None

	def __new__(cls,data=None,operator=None,site=None,string=None,parameters=None,state=None,system=None,inherit=False,**kwargs):		

		# TODO: Allow multiple different classes to be part of one operator, and swap around localities

		self = None

		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,parameters=parameters,state=state,system=system),delimiter=delim,func=False)

		classes = [Gate,Pauli,Haar,State,Noise]

		for subclass in classes:
			
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
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
		string (str): string label of operator
		parameters (object): parameter of operator				
		state (object): state of operators				
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I','i']},
		**{attr: Object(data=array([[0,1],[1,0]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['X','x']},
		**{attr: Object(data=array([[0,-1j],[1j,0]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['Y','y']},
		**{attr: Object(data=array([[1,0],[0,-1]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['Z','z']},
			}
	default = 'I'
	D = 2
	N = None
	n = None

	hermitian = True
	unitary = True

	def __call__(self,parameters=None,state=None,conj=None):
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
		if parameters is None:
			return self.data

		return cos(pi*parameters)*self.identity + -1j*sin(pi*parameters)*self.data

	def grad(self,parameters=None,state=None,conj=None):
		'''
		Call gradient
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate			
		Returns:
			data (array): data
		'''
		return -pi*sin(pi*parameters)*self.identity + -1j*pi*cos(pi*parameters)*self.data


	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''
		return

class Gate(Object):
	'''
	Gate class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
		string (str): string label of operator
		parameters (object): parameter of operator				
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I']},
		**{attr: Object(data=array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),D=2,locality=2,hermitian=True,unitary=True,string=attr) for attr in ['CNOT','C','cnot']},
		**{attr: Object(data=array([[1,1,],[1,-1]])/sqrt(2),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['HADAMARD','H']},
		**{attr: Object(data=array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
						  [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]]),D=2,locality=3,hermitian=True,unitary=True,string=attr) for attr in ['TOFFOLI','T','toffoli']},
		}
	default = 'I'
	D = 2 
	N = None
	n = None
	
	hermitian = False
	unitary = True
	
	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''
		return

class Haar(Object):
	'''
	Haar class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
		string (str): string label of operator
		parameters (object): parameter of operator		
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I']},
		**{attr: Object(data=rand(shape=(2,2),random='haar',dtype='complex'),D=2,locality=1,hermitian=False,unitary=True,string=attr) for attr in ['random','U','haar']},
		}
	default = 'I'
	D = 2
	N = None
	n = None

	hermitian = False
	unitary = True

	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''

		shape = (self.D**self.N,)*self.ndim
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
		
		self.data = data

		return


class State(Object):
	'''
	State class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
		string (str): string label of operator
		parameters (object): parameter of operator		
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I']},
		**{attr: Object(data=rand(shape=(2,),random='haar',dtype='complex'),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['random','psi','haar']},
		**{attr: Object(data=array([1,0,]),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['zeros','0']},
		**{attr: Object(data=array([0,1,]),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['ones','1']},
		**{attr: Object(data=array([1,1,])/sqrt(2),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['plus','+']},
		**{attr: Object(data=array([1,-1,])/sqrt(2),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['minus','-']},
		}
	default = 'I'
	D = 2
	N = None
	n = None

	hermitian = True
	unitary = False

	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data of operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''

		shape = (self.D**self.N,)*self.ndim
		if isinstance(self.samples,int):
			shape = (self.samples,*shape)
		ndim = len(shape)
		size = prod(shape[-self.ndim:])
		random = getattr(self,'random','haar')
		seed = getattr(self,'seed',None)
		reset = getattr(self,'reset',None)
		dtype = self.dtype

		data = zeros(shape=shape,dtype=dtype)
		operator = operator[0] if operator else None

		if operator in ['zeros','0']:
			data = setitem(data,0,1)
		elif operator in ['ones','1']:
			data = setitem(data,-1,1)			
		elif operator in ['plus','+']:
			data = setitem(data,slice(None),1/sqrt(size))			
		elif operator in ['minus','-']:
			data = setitem(data,slice(None),(-1)**arange(size)/sqrt(size))
		elif operator in ['random','psi','haar']:
			data = rand(shape=shape,random=random,seed=seed,reset=reset,dtype=dtype)
		else:
			data = self.data

		if (data is not None) and (ndim < self.ndim):
			data = einsum('...i,...j->...ij',data,conj(data))

		self.data = data

		return


class Noise(Object):
	'''
	Noise class for Quantum Objects
	Args:
		data (iterable[str]): data of operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
		string (str): string label of operator
		parameters (object): parameter of operator		
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['I','i']},
		**{attr: Object(data=array([[1,0],[0,1]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['D','depolarize']},
		**{attr: Object(data=array([[0,1],[1,0]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['X','x','amplitude']},
		**{attr: Object(data=array([[1,0],[0,0]]),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['00']},
		**{attr: Object(data=array([[0,1],[0,0]]),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['01']},
		**{attr: Object(data=array([[0,0],[1,0]]),D=2,locality=1,hermitian=False,unitary=False,string=attr) for attr in ['10']},
		**{attr: Object(data=array([[0,0],[0,1]]),D=2,locality=1,hermitian=True,unitary=False,string=attr) for attr in ['11']},
		**{attr: Object(data=array([[0,-1j],[1j,0]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['Y','y','amplitude_phase']},
		**{attr: Object(data=array([[1,0],[0,-1]]),D=2,locality=1,hermitian=True,unitary=True,string=attr) for attr in ['Z','z','phase']},
		}
	default = 'I'
	D = 2
	N = None
	n = None
	
	hermitian = False
	unitary = False


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
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','i,j','ij','i<j','i...j']
			string (str): string label of operator
			parameters (object): parameter of operator			
		'''

		if (getattr(self,'scale',None) is not None):
			if (getattr(self,'initialization',None) in ['time']):
				if (getattr(self,'tau') is not None):
					self.parameters = (1 - exp(-self.tau/self.scale))/2

		if (self.parameters is None) or (self.parameters is True):
			self.data = None
			return

		assert (self.parameters >= 0) and (self.parameters <= 1), "Noise scale %r not in [0,1]"%(self.parameters)
	
		parameters = self.parameters
		operator = operator[0] if operator else None

		if parameters > 0:
			if operator is None:
				data = [self.basis['I']()]
			elif operator in ['phase']:
				data = [sqrt(1-parameters)*self.basis['I'](),
						sqrt(parameters)*self.basis['Z']()]
			elif operator in ['amplitude']:
				data = [self.basis['00']() + sqrt(1-parameters)*self.basis['11'](),
						sqrt(parameters)*self.basis['01']()]
			elif operator in ['depolarize']:
				data = [sqrt(1-parameters)*self.basis['I'](),
						sqrt(parameters/3)*self.basis['X'](),
						sqrt(parameters/3)*self.basis['Y'](),
						sqrt(parameters/3)*self.basis['Z']()]
			else:
				data = [self.basis['I']()]
		else:
			data = [self.basis['I']()]
	
		data = array([tensorprod(i)	for i in itertools.product(data,repeat=self.N)],dtype=self.dtype)

		self.data = data

		return


def Compute(data,parameters,identity,state,noise,coefficients,n,d,m,p):
	'''
	Calculate operators
	Args:
		data (iterable[Operator]): Array of data to matrix exponentiate of shape (d,n,n)
		parameters (Parameters): Array of parameters of shape (d,m)
		identity (Operator): Array of data identity of shape (n,n)
		state (Operator): Array of state to act on of shape (n,) or (n,n) or (p,n) or (p,n,n)
		noise (Operator): Array of noise to act of shape (n,n) or (k,n,n)
		coefficients (arrray): Array of coefficients of shape (1,) or (d,)
		n (int): Size of array
		d (int): Number of data
		m (int): Number of steps
		p (int): Number of trotterizations
	Returns:
		func (callable): Function to compute operators, with signature func(parameters,state,conj), where parameters[i][j] is a nested iterable for each operator i and time step j
		grad (callable): Gradient to compute operators, with signature grad(parameters,state,conj), where parameters[i][j] is a nested iterable for each operator i and time step j
	'''	

	if parameters is None:
		return lambda *args,**kwargs: None

	if p > 2:	
		raise NotImplementedError("Trotterization p = %d not implemented for p>2"%(p))

	trotterize = lambda iterable,p: [i for s in [slice(None,None,1),slice(None,None,-1)][:p] for i in iterable[s]]

	indices = array(trotterize(arange(d),p))

	@jit
	def trotter(iterable):
		return iterable[indices]

	data = trotterize(data,p)

	funcs = lambda i,parameters: data[i%(d*p)](parameters[i])


	I = identity
	A = array([i() for i in data])

	def func(x,A,I,state=None,conj=None):
		'''
		Calculate matrix exponential of parameters times data
		Args:
			x (array): parameters of shape (m,) or (m,n,) or (m,n,n)		
			A (array): Array of data to matrix exponentiate of shape (d,n,n)
			I (array): Array of data identity
		Returns:
			out (array): Matrix exponential of A of shape (n,n)
		'''	

		x = trotter(x).T.ravel()

		m = x.size
		d,shape = A.shape[0],A.shape[1:]

		# subscripts = 'ij,jk->ik'
		# shapes = (shape,shape)
		# einsummation = einsum #(subscripts,shapes)

		# def func(i,out):
		# 	U = cos(x[i]*pi)*I + -1j*sin(x[i]*pi)*A[i%(d)]
		# 	return einsummation(subscripts,U,out)

		# return forloop(0,m,func,I)

		if conj:
			def func(i,out):
				U = cos(x[i]*pi)*I + 1j*sin(x[i]*pi)*A[i%(d)]
				return dot(U,out)#subscripts,U,out)

			return forloop(m-1,-1,func,I)		
		
		else:
			def func(i,out):
				U = cos(x[i]*pi)*I + -1j*sin(x[i]*pi)*A[i%(d)]
				return dot(U,out)#subscripts,U,out)

			return forloop(0,m,func,I)		

		# out = I
		# for i in range(0,m):
		# 	U = cos(x[i]*pi)*I + -1j*sin(x[i]*pi)*A[i%(d)]			
		# 	out = dot(U,out)
	
		# return out

	func = jit(func,A=A,I=I)

	# data = [jit(i) for i in data]#data[i]) for i in indices]
	# funcs = jit(lambda i,parameters: switch(i,data,parameters))

	# # @jit
	# def func(parameters=None,state=None,conj=None):
	# 	'''
	# 	Compute operator
	# 	Args:
	# 		parameters (array): parameters of shape (d,m,...)
	# 		state (object): state
	# 		conj (bool): conjugate
	# 	'''
	# 	parameters = coefficients*trotter(parameters).T.ravel()

	# 	@jit
	# 	def function(i,out):
	# 		return dot(funcs(i%(d*p),parameters[i]),out)

	# 	out = forloop(0,m*d*p,function,identity)

	# 	return out




	# # @jit
	# def func(parameters):#=None,state=None,conj=None):
	# 	'''
	# 	Compute operator
	# 	Args:
	# 		parameters (array): parameters of shape (d,m,...)
	# 		state (object): state
	# 		conj (bool): conjugate
	# 	'''
	# 	parameters = coefficients*trotter(parameters).T.ravel()

	# 	@jit
	# 	def function(i,out):
	# 		return dot(switch(i%(d*p),data,parameters[i]),out)
	# 		# return dot(data[i%(d*p)](parameters[i]),out)

	# 	out = forloop(0,m*d*p,function,identity)

	# 	return out

		# @jit
		# def func(parameters):
		# 	return trotter(funcs(index,parameters))

		# def function(i,out):
		# 	for operator in trotter(funcs(index,parameters[:,i])):
		# 		out = dot(operator,out)
		# 	return out

		# out = identity
		# parameters = parameters.T.ravel()
		# size = parameters.size
		
		# out = forloop(0,size,func,out)
		
		# @jit
		# def function(i,out):
		# 	return dot(funcs(i%(d*p),parameters[i]),out)

		# out = forloop(0,m*d*p,function,identity)

		# if not conj:
		# 	@jit
		# 	def function(i,out):
		# 		for operator in trotter(funcs(index,parameters[:,i])):
		# 			out = dot(operator,out)
		# 		return out

		# 	out = identity
			
		# 	out = forloop(0,m,function,out)

		# else:
		# 	@jit
		# 	def function(i,out):
		# 		for operator in trotter(funcs(index,parameters[:,i]))[::-1]:
		# 			out = dot(operator,out)
		# 		return out

		# 	out = identity
			
		# 	out = forloop(m-1,-1,function,out)			
		
		# return out



	# def func(parameters=None,state=None,conj=None):
	# 	if conj:
	# 		sign = -1
	# 		slices = slice(None,None,-1)
	# 		indices = range(m-1,-1,-1)
	# 	else:
	# 		sign = 1
	# 		slices = slice(None,None,1)
	# 		indices = range(0,m,1)
	# 	out = identity
	# 	for j in indices:
	# 		operators = [data[i](sign*coefficients[i]*parameters[i][j],conj=True) for i in range(d)]
	# 		operators = trotter(operators,p)[slices]
	# 		for operator in operators:
	# 			out = dot(operator,out)
	# 	return out

	return func
	
	# if state is None and noise is None:
	# 	self.summation = jit(summation,data=data,identity=identity)
	# 	self.exponentiation = jit(exponentiation,data=data,identity=identity)
	# elif state is not None and noise is None:
	# 	if state.ndim == 1:
	# 		self.summation = jit(summationv,data=data,identity=identity,state=state)
	# 		self.exponentiation = jit(exponentiationv,data=data,identity=identity,state=state)
	# 	elif state.ndim == 2:
	# 		self.summation = jit(summationm,data=data,identity=identity,state=state)
	# 		self.exponentiation = jit(exponentiationm,data=data,identity=identity,state=state)
	# 	else:
	# 		self.summation = jit(summation,data=data,identity=identity)
	# 		self.exponentiation = jit(exponentiation,data=data,identity=identity)
	# elif state is None and noise is not None:
	# 	self.summation = jit(summation,data=data,identity=identity)
	# 	self.exponentiation = jit(exponentiation,data=data,identity=identity)
	# elif state is not None and noise is not None:
	# 	if state.ndim == 1:
	# 		self.summation = jit(summationmvc,data=data,identity=identity,state=state,constants=noise)
	# 		self.exponentiation = jit(exponentiationmvc,data=data,identity=identity,state=state,constants=noise)
	# 	elif state.ndim == 2:
	# 		self.summation = jit(summationmmc,data=data,identity=identity,state=state,constants=noise)
	# 		self.exponentiation = jit(exponentiationmmc,data=data,identity=identity,state=state,constants=noise)
	# 	else:
	# 		self.summation = jit(summation,data=data,identity=identity)
	# 		self.exponentiation = jit(exponentiation,data=data,identity=identity)
	# else:
	# 	self.summation = jit(summation,data=data,identity=identity)
	# 	self.exponentiation = jit(exponentiation,data=data,identity=identity)



class Operators(Object):
	'''
	Class for Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,parameters dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
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
		self.compute = None

		self.system = system

		self.__time__()
		self.__space__()
		self.__lattice__()

		self.shape = () if self.n is None else (self.n,self.n)
		self.size = prod(self.shape)
		self.ndim = len(self.shape)
		self.identity = Operator(Operator.default,N=self.N,D=self.D,system=self.system,verbose=False)
		self.coefficients = array(self.tau/self.P,dtype=self.dtype)

		self.__setup__(data,operator,site,string,parameters)

		self.info()

		return	

	def __setup__(self,data=None,operator=None,site=None,string=None,parameters=None):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,parameters dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
				string (iterable[str]): string labels of operators
				parameters (iterable[object]): parameters of operators
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
		'''

		# Get operator,site,string from data
		objs = {'operator':operator,'site':site,'string':string,'parameters':parameters}

		for obj in objs:
			objs[obj] = [] if objs[obj] is None else objs[obj]

		if data is None:
			data = {}
		elif all(isinstance(datum,Operator) for datum in data):
			data = {datum.timestamp: datum for datum in data}
		
		assert isinstance(data,dict), 'Incorrect data format %r'%(type(data))			

		if all(isinstance(data[name],dict) and (obj in data[name]) for name in data for obj in objs):
			for obj in objs:
				objs[obj].extend([data[name][obj] for name in data])

		# Set class attributes
		self.__extend__(**objs)

		# Set class functions
		self.__initialize__()

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
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
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


		# Get functions
		for obj in objs:
			data,cls = objs[obj],classes[obj]
			data = getattr(self,obj,None) if data is None or data is True else data if data is not False else None
			if not isinstance(data,cls):
				kwargs = {}

				args = {**dict(data=None),**data} if isinstance(data,dict) else dict(data=data)
				setter(kwargs,args,func=False)

				args = dict(**namespace(cls,self),model=self)
				setter(kwargs,args,func=False)

				args = dict(verbose=False)
				setter(kwargs,args,func=True)

				instance = cls(**kwargs)

			else:

				instance = data

			setattr(self,obj,instance)
				
		# Set functions
		data = self.data
		identity = self.identity()
		parameters = self.parameters()
		state = self.state()
		noise = self.noise()
		coefficients = self.coefficients
		n = self.n
		d = len(self.data)
		m = self.M
		p = self.P

		self.compute = Compute(data,parameters,identity,state,noise,coefficients,n,d,m,p)

		# Update class attributes
		self.gradient = gradient(self,mode='fwd',move=True)

		return

	def __call__(self,parameters=None,state=None,conj=None):
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

		parameters = self.coefficients*self.parameters(parameters)

		return self.compute(parameters,state=state,conj=conj)

	def __grad__(self,parameters=None,state=None,conj=None):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate			
		Returns:
			out (array): Return of function
		'''	
		return self.gradient(parameters=parameters,state=state,conj=conj)


	def __value_and_grad__(self,parameters=None,state=None,conj=None):
		'''
		Class function and gradient
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function and gradient
		'''	
		return self.value_and_gradient(parameters=parameters,state=None,conj=None)


	def func(self,parameters=None,state=None,conj=None):
		'''
		Class function
		Args:
			parameters (array): parameters		
			state (obj): state
			conj (bool): conjugate
		Returns
			out (array): Return of function
		'''
		return self.__call__(parameters=parameters,state=state,conj=conj)

	def grad(self,parameters=None,state=None,conj=None):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate
		Returns:
			out (array): Return of function
		'''		
		return self.__grad__(parameters=parameters,state=state,conj=conj)

	def value_and_grad(self,parameters=None,state=None,conj=None):
		'''
		Class function and gradient
		Args:
			parameters (array): parameters		
			state (obj): state
			conj (bool): conjugate
		Returns
			out (array): Return of function and gradient
		'''	
		return self.__value_and_gradient__(parameters=parameters,state=state,conj=conj)

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

		msg = '%s'%('\n'.join([
			*['%s: %s'%(attr,getattrs(self,attr,delimiter=delim)) 
				for attr in ['string','key','seed','N','D','d','L','delta','M','tau','T','P','n','g','unit','data','shape','cwd','path','dtype','backend','architecture','conf','logger','cleanup']
			],
			*['%s: %s'%(delim.join(attr.split(delim)[:2]),', '.join([
				('%s: %s' if (
					(getattrs(self,delim.join([attr,prop]),delimiter=delim) is None) or 
					isinstance(getattrs(self,delim.join([attr,prop]),delimiter=delim),(str,int,list,tuple))) 
				else '%s: %0.3e')%(prop,getattrs(self,delim.join([attr,prop]),delimiter=delim))
				for prop in ['category','method','shape','parameters']]))
				for attr in ['parameters.%s'%(i) for i in (self.parameters if self.parameters is not None else [])]
			],
			*['%s: %s'%(delim.join(attr.split(delim)[:1]),', '.join([
				((('%s: %s' if (
					(getattrs(self,delim.join([attr,prop]),delimiter=delim) is None) or 
					isinstance(getattrs(self,delim.join([attr,prop]),delimiter=delim),(str,int,list,tuple))) 
				else '%0.3e')%(prop,getattrs(self,delim.join([attr,prop]),delimiter=delim),)) 
				if prop is not None else str(getattrs(self,attr,delimiter=delim,default=None)))
				for prop in [None,'shape','parameters']]))
				for attr in ['parameters','state','noise'] if callable(getattrs(self,attr,default=None)) and getattrs(self,attr)() is not None
			],
			# *['%s:\n%s'%(delim.join(attr.split(delim)[:1]),
			# 	to_string(getattrs(self,attr,default=lambda:None)()))
			# 	for attr in ['parameters','state','noise'] if callable(getattrs(self,attr,default=None)) and getattrs(self,attr)() is not None
			# ],			
			]
			))
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
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
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
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
				string (iterable[str]): string labels of operators
				parameters (iterable[object]): parameters of operators
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
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
			data = {}
		elif all(isinstance(datum,Operator) for datum in data):
			data = {datum.timestamp: datum for datum in data}
		
		assert isinstance(data,dict), 'Incorrect data format %r'%(type(data))			

		if all(isinstance(data[name],dict) and (obj in data[name]) and (data[name][obj] is not None) for name in data for obj in objs):
			for obj in objs:
				objs[obj].extend([data[name][obj] for name in data])

		# Lattice sites
		sites = self.lattice # sites types on lattice
		indices = {'i': ['i'],'<ij>':['i','j'],'i<j':['i','j'],'ij':['i','j'],'i...j':['i','j']}   # allowed symbolic indices and maximum locality of many-body site interactions

		# Get number of operators
		size = min(len(objs[obj]) for obj in objs)
		
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
							).get(i,parse(i,int)) 
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
		self.__extend__(**objs)

		# Set class functions
		self.__initialize__()

		return



class Unitary(Hamiltonian):
	'''
	Unitary class of Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,parameters dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
			string (iterable[str]): string labels of operators
			parameters (iterable[object]): parameters of operators
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','ij','i<j','i...j']
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

		self.hermitian = False
		self.unitary = True

		self.norm()

		return

	def __grad_analytical__(self,parameters=None,state=None,conj=None):
		'''
		Class gradient
		Args:
			parameters (array): parameters		
			state (obj): state
			conj (bool): conjugate
		Returns
			out (array): Return of function
		'''	

		# Get class attributes
		parameters = self.parameters(parameters)
		shape = indexer(self.parameters.index,self.parameters).shape

		attributes = None

		# Get data
		data = self.data
		dtype = self.dtype

		# Get trotterized shape
		P = self.P
		shape = list(shape[::-1])

		shape[-1] *= P

		ndim = len(shape)

		attribute = 'slice'
		layer = 'variables'
		slices = attributes[attribute][layer]		

		attribute = 'index'
		layer = 'variables'
		indices = attributes[attribute][layer]		

		slices = tuple([
			*[[slices[parameter][group][axis] for parameter in slices for group in slices[parameter]][0]
				for axis in range(0,1)],
			*[[slices[parameter][group][axis] for parameter in slices for group in slices[parameter]][0]
				for axis in range(1,ndim)]
		])

		indices = tuple([
			*[[i for parameter in indices for group in indices[parameter] for i in indices[parameter][group][axis]]
				for axis in range(0,1)],
			*[[indices[parameter][group][axis] for parameter in indices for group in indices[parameter]][0]
				for axis in range(1,ndim)]
		])

		# Calculate parameters and gradient
		parameters = self.parameters(parameters)

		coefficients = self.coefficients
		data = array(trotter([datum() for datum in data],P))
		identity = self.identity()

		grad = gradient_expm(coefficients*parameters,data,identity)
		grad *= coefficients

		# Reshape gradient
		axis = 1

		grad = grad.reshape((*shape,*grad.shape[axis:]))

		grad = grad.transpose(axis,0,*[i for i in range(grad.ndim) if i not in [0,axis]])

		grad = array(gradient_trotter(grad,P))

		grad = grad[indices]

		grad = grad.reshape((-1,*grad.shape[axis+1:]))

		return grad

	def grad_analytical(self,parameters=None,state=None,conj=None):
		'''
		Class gradient
		Args:
			parameters (array): parameters		
			state (obj): state
			conj (bool): conjugate
		Returns
			out (array): Return of function
		'''	
		return self.__grad_analytical__(parameters=parameters,state=state,conj=conj)


class Label(Operator):
	
	basis = {}
	default = None
	D = None
	N = None
	n = None

	hermitian = False
	unitary = False


	def __new__(cls,*args,**kwargs):

		self = super().__new__(cls,*args,**kwargs)
		
		data = asarray(self(self.parameters))
		state = asarray(self.state)


		if data is None:
			pass
		elif state is None:
			pass
		elif state.ndim == 1:
			data = einsum('ij,j->i',data,state)
		elif state.ndim == 2:
			data = einsum('ij,jk,kl->il',data,state,dagger(data))
		
		self.data = data
		self.shape = data.shape
		self.size = data.size
		self.ndim = data.ndim

		return self

	def __init__(self,*args,**kwargs):
		return

	def __call__(self,parameters=None,state=None,conj=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
			state (obj): state
			conj (bool): conjugate			
		Returns:
			data (array): data
		'''
		return self.data

		

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
			'variables':[],'variables.relative':[],'variables.relative.mean':[],
			'objective.ideal.noise':[],'objective.diff.noise':[],'objective.rel.noise':[],
			'objective.ideal.state':[],'objective.diff.state':[],'objective.rel.state':[],
			'objective.ideal.operator':[],'objective.diff.operator':[],'objective.rel.operator':[],
			'hessian':[],'fisher':[],
			'hessian.eigenvalues':[],'fisher.eigenvalues':[],
			'hessian.rank':[],'fisher.rank':[],

			'N':[],'D':[],'d':[],'L':[],'delta':[],'M':[],'T':[],'tau':[],'P':[],
			'space':[],'time':[],'lattice':[],'architecture':[],'timestamp':[],

			'noise.scale':[],'optimize.c1':[],'optimize.c2':[],

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
		
		done = (len(attributes['iteration'])>1) and (attributes['iteration'][-1] == (iterations.stop))
		
		status = (
			((len(attributes['value']) >= 1) and 
			 (attributes['iteration'][-1] <= max(1,
				hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) or
			(
			(abs(attributes['value'][-1]) > 
				(hyperparameters['eps']['value']*hyperparameters['value']['value'])) and
			(log10(abs(attributes['value'][-1] - attributes['value'][-2])) > 
				(log10(abs(hyperparameters['eps']['value.difference'])))) and
			(norm(attributes['grad'][-1])/attributes['grad'][-1].size > 
				  (hyperparameters['eps']['grad']*hyperparameters['value']['grad'])) and
			(norm(attributes['grad'][-1] - attributes['grad'][-2])/attributes['grad'][-2].size > 
				  (hyperparameters['eps']['grad.difference']*norm(attributes['grad'][-2])/attributes['grad'][-2].size))
			)
			)


		other = ((len(attributes['iteration']) == 1) or 
			(hyperparameters['modulo']['track'] is None) or 
			(attributes['iteration'][-1]%hyperparameters['modulo']['track'] == 0))

		stop = (
			(hyperparameters['eps'].get('value.increase') is not None) and
			((len(attributes['value']) > 1) and 
			 (attributes['iteration'][-1] >= max(1,
				hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) and
			((attributes['value'][-1] > attributes['value'][-2]) and
			(log10(attributes['value'][-1] - attributes['value'][-2]) > 
			(log10(hyperparameters['eps']['value.increase']*attributes['value'][-1]))))
			)


		status = (status) and (not stop)

		updates = {
			**{attr: lambda i,attr,track,default: (track[attr][-1]) for attr in ['iteration.max','iteration.min']},
			**{attr: lambda i,attr,track,default: (empty(track[attr][-1].shape) if ((i>0) and i<(len(track[attr])-1)) else track[attr][i])
				for attr in [
					'parameters','grad','search',
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
					value = int(track['iteration'][argmin(abs(array(track['objective'])))])

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

				elif attr in ['variables.norm','variables.relative','variables.relative.mean'
					] and (not do):
					value = default

				elif attr in [
					'variables','variables.norm','variables.relative','variables.relative.mean',
					] and (do):
					size = model.parameters.size//model.M
					if attr in ['variables']:
						value = model.parameters(parameters)[:size]
					elif attr in ['variables.norm']:
						value = model.parameters(parameters)[:size]
						value = norm(value)/(value.size)
					elif attr in ['variables.relative']:
						eps = 1e-20
						value = model.parameters(parameters)[:size]
						_value = model.parameters(attributes['parameters'][0])[:size]
						value = abs((value - _value + eps)/(_value + eps))
					elif attr in ['variables.relative.mean']:
						eps = 1e-20
						value = model.parameters(parameters)[:size]
						_value = model.parameters(attributes['parameters'][0])[:size]
						value = abs((value - _value + eps)/(_value + eps)).mean()

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

					_kwargs = {kwarg: {prop: hyperparameters.get('kwargs',{}).get(kwarg,{}).get(prop) if kwarg in ['noise'] else None for prop in ['scale']} for kwarg in ['state','noise','label']}
					_kwargs = {kwarg: {prop: getattrs(model,[kwarg,prop],delimiter=delim,default=_kwargs[kwarg][prop]) for prop in _kwargs[kwarg]} for kwarg in ['state','noise','label']}
					if attr in ['objective.ideal.noise','objective.diff.noise','objective.rel.noise']:
						_kwargs = {kwarg: False if kwarg in [] else _kwargs[kwarg] for kwarg in _kwargs}
						_metric = 'real'
					elif attr in ['objective.ideal.state','objective.diff.state','objective.rel.state']:						
						_kwargs = {kwarg: False if kwarg in ['noise'] else _kwargs[kwarg] for kwarg in _kwargs}
						_metric = 'real'
					elif attr in ['objective.ideal.operator','objective.diff.operator','objective.rel.operator']:
						_kwargs = {kwarg: False if kwarg in ['noise','state'] else _kwargs[kwarg] for kwarg in _kwargs}
						_metric = 'abs2'

					_model = model
					_label = Label()
					_shapes = _label.shape
					_optimize = None
					_hyperparameters = hyperparameters
					_system = model.system
					_restore = {kwarg: deepcopy(getattr(model,kwarg)) for kwarg in _kwargs}

					_model.__initialize__(**_kwargs)
					_metric = Metric(_metric,shapes=_shapes,label=_label,optimize=_optimize,hyperparameters=_hyperparameters,system=_system,verbose=False)

					if attr in ['objective.ideal.noise','objective.ideal.state','objective.ideal.operator']:
						value = abs(_metric(_model(parameters)))
					elif attr in ['objective.diff.noise','objective.diff.state','objective.diff.operator']:
						value = abs((track['objective'][-1] - _metric(_model(parameters))))
					elif attr in ['objective.rel.noise','objective.rel.state','objective.rel.operator']:
						value = abs((track['objective'][-1] - _metric(_model(parameters)))/(track['objective'][-1]))

					model.__initialize__(**_restore)


				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (not do):
					value = default

				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (do):
					
					if attr in ['hessian','hessian.eigenvalues','hessian.rank']:
						function = hessian(jit(lambda parameters: metric(model(parameters))))
					elif attr in ['fisher','fisher.eigenvalues','fisher.rank']:
						function = fisher(model,model.grad,shapes=(model.shape,(parameters.size,*model.shape)))

					if attr in ['hessian','fisher']:
						value = function(parameters)

					elif attr in ['hessian.eigenvalues','fisher.eigenvalues']:
						value = sort(abs(eig(function(parameters),hermitian=True)))[::-1]
						value = value/maximum(value)
					elif attr in ['hessian.rank','fisher.rank']:
						value = sort(abs(eig(function(parameters),hermitian=True)))[::-1]
						value = argmax(abs(difference(value)/value[:-1]))+1	
						value = value.size if (value==value.size-1) else value

				elif attr in ['tau.noise.parameters','T.noise.parameters']:
					value = [attr.split(delim)[0],delim.join(attr.split(delim)[1:])]
					value = [getattrs(model,i,default=default,delimiter=delim) for i in value]
					value = value[0]/value[1] if value[1] else value[0]

				elif attr not in attributes and not (getter(hyperparameters,attr.replace('optimize%s'%(delim),''),default=null,delimiter=delim) is null):
					value = getter(hyperparameters,attr.replace('optimize%s'%(delim),''),default=default,delimiter=delim)

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
				# 'x\n%s'%(to_string(parameters.round(4))),
				# 'theta\n%s'%(to_string(model.parameters(parameters).round(4))),
				'U\n%s\nV\n%s'%(
					to_string((model(parameters)).round(4)),
					to_string((metric.label).round(4))),
				])


			model.log(msg)


		return status



