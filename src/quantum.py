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

from src.utils import jit,gradient,hessian,fisher
from src.utils import array,empty,identity,ones,zeros,arange,rand,setitem
from src.utils import tensorprod,dagger,conj,einsum,dot
from src.utils import summation,exponentiation,summationv,exponentiationv,summationm,exponentiationm,summationmvc,exponentiationmvc,summationmmc,exponentiationmmc
from src.utils import trotter,gradient_trotter,gradient_expm
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
		data (iterable[str]): data for operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators
		string (str): string labels of operators
		interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		parameters (object): parameters of operators
		state (object): state of operators
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {}
	default = None
	dim = None

	hermitian = False
	unitary = False

	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,parameters=None,state=None,system=None,**kwargs):		

		defaults = dict(
			N=None,D=None,n=None,
			shape=None,size=None,ndim=None,
			samples=None,identity=None,locality=None,index=None
			)

		setter(kwargs,defaults,delimiter=delim,func=False)
		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,interaction=interaction,parameters=parameters,state=state,system=system),delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		if isinstance(data,Object) or isinstance(operator,Object):
			if isinstance(data,Object):
				for attr in data:
					setattr(self,attr,getattr(data,attr))
			if isinstance(operator,Object):
				for attr in operator:
					setattr(self,attr,getattr(operator,attr))					

			return
		
		elif (data is None) and (operator is None):
			return
		
		elif isinstance(data,dict):
			return
		
		default = self.default	

		if operator is None:
			operator = data
		
		if operator is None:
			pass
		elif isinstance(operator,str) and operator not in [default]:
			operator = [i for i in operator.split(delim)]
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

		if interaction is None:
			pass
		elif not isinstance(interaction,str):
			interaction = str(interaction)
	

		N = max(self.N,max(site)+1 if site is not None else self.N) if self.N is not None else max(site)+1 if site is not None else 0
		D = self.D if self.D is not None else self.dim if self.dim is not None else data.size**(1/(data.ndim*N)) if isinstance(data,arrays) else 1
		n = D**N if (N is not None) and (D is not None) else None

		shape = self.shape if self.shape is not None else data.shape if isinstance(data,arrays) else None
		size = self.size if self.size is not None else data.size if isinstance(data,arrays) else None
		ndim = self.ndim if self.ndim is not None else data.ndim if isinstance(data,arrays) else None

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
		self.interaction = interaction if interaction is not None else None
		self.system = system

		self.N = N
		self.D = D
		self.n = n

		self.shape = shape
		self.size = size
		self.ndim = ndim
		
		if not (((self.data is None) and (self.operator is None)) or (isinstance(self.data,arrays))):
			self.__setup__(data,operator,site,string,interaction)

		if self.parameters is False:
			self.data = None
		elif isinstance(self.operator,arrays):
			self.data = self.operator
		elif isinstance(self.data,arrays):
			pass
		elif self.operator is None:
			self.data = None
		elif self.data is not None:
			self.data = tensorprod([self.basis.get(i)() for i in self.operator]) if all(i in self.basis for i in self.operator) else None

		self.data = self.data.astype(self.dtype) if self.data is not None else None

		self.operator = list((i for i in self.operator)) if self.operator is not None and not isinstance(self.operator,arrays) else self.operator
		self.site = list((i for i in self.site)) if self.site is not None else self.site
		self.string = str(self.string) if self.string is not None else self.string
		self.interaction = str(self.interaction) if self.interaction is not None else self.interaction

		self.identity = tensorprod([self.basis.get(self.default)() for i in range(self.N)]) if (self.default in self.basis) else None
		self.identity = self.identity.astype(self.dtype) if self.identity is not None else None

		self.locality = max(self.locality if self.locality is not None else 0,len(self.site) if self.site is not None else 0)
		self.index = self.index if self.index is not None else None
		
		if (self.samples is True) and (self.size is not None):
			self.samples = rand(self.size,bounds=[0,1],seed=self.seed,dtype=datatype(self.dtype))
			self.samples /= self.samples.sum()
		elif not isinstance(self.samples,arrays):
			self.samples = None

		if isinstance(self.data,arrays) and (self.samples is not None) and (self.ndim is not None):
			if (self.data.ndim>self.ndim):
				self.data = einsum('%s...,%s->...'%((''.join(['i','j','k','l'][:self.data.ndim-self.ndim]),)*2),self.samples,self.data)

		self.shape = self.data.shape if isinstance(self.data,arrays) else self.shape
		self.size = self.data.size if isinstance(self.data,arrays) else self.size
		self.ndim = self.data.ndim if isinstance(self.data,arrays) else self.ndim

		self.norm()

		self.info()

		return

	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data for operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators
			string (str): string labels of operators
			interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']	
		'''
		return

	def __call__(self,parameters=None,state=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
			state (object): state
		Returns:
			data (array): data
		'''
		return self.data

	def __str__(self):
		string = self.__class__.__name__ if not self.string else self.string
		if self.operator is not None and not isinstance(self.operator,arrays):
			string = '%s'%(delim.join(self.operator))
		return string

	def __repr__(self):
		return self.__str__()
	
	def __len__(self):
		return len(self.operator)

	def __copy__(self):
		new = self.__class__(**self)
		return new

	def __deepcopy__(self):
		return self.__class__(**deepcopy(dict(self)))

	def copy(self,deep=True):
		if deep:
			return self.__deepcopy__()
		else:
			return self.__copy__()
	
	def grad(self,parameters=None,state=None):
		'''
		Call gradient
		Args:
			parameters (array): parameters
			state (object): state
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
			*['Label %s %s: %s'%(self,attr,getattr(self,attr)) 
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

		if self.data is None:
			return

		data = self.data
		shape = data.shape
		ndim = data.ndim
		dtype = data.dtype
		hermitian = self.hermitian		
		unitary = self.unitary		

		if ndim > 3:
			normalization = einsum('...uij,...ukj->...ik',conj(data),data)
			eps = array([identity(shape[-2:],dtype=dtype)]*(normalization.ndim-2),dtype=dtype)
		elif ndim == 3:
			normalization = einsum('uij,ukj->ik',conj(data),data)
			eps = identity(shape[-2:],dtype=dtype)
		elif ndim == 2:
			if hermitian and not unitary:
				normalization = einsum('ii->',data)
				eps = ones(shape[:-2],dtype=dtype)
			elif not hermitian and unitary:
				normalization = einsum('ij,kj->ik',conj(data),data)
				eps = identity(shape[-2:],dtype=dtype)							
			else:
				normalization = None
				eps = None
		else:
			normalization = einsum('i,i->',conj(data),data)
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
		data (iterable[str]): data for operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators
		string (str): string labels of operators
		interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		parameters (object): parameters of operators		
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''
	
	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['I','i']},
			}
	default = 'I'
	dim = 2

	def __new__(cls,data=None,operator=None,site=None,string=None,interaction=None,parameters=None,state=None,system=None,**kwargs):		

		# TODO: Allow multiple different classes to be part of one operator, and swap around localities

		self = None

		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,interaction=interaction,parameters=parameters,state=state,system=system),delimiter=delim,func=False)

		classes = [Gate,Pauli,Haar,State,Noise]
		for cls in classes:
			if not all(j in cls.basis for obj in [data,operator] if obj is not None for k in (obj if not isinstance(obj,str) else [obj]) for j in ([k] if k in cls.basis else k.split(delim))):
				continue
			self = cls(**kwargs)
			break

		assert (self is not None) or (self is None and data is None and operator is None), "TODO: All operators not in same class %r"%([
			*((data if not isinstance(data,str) else [data]) if data is not None else []),
			*((operator if not isinstance(operator,str) else [operator]) if operator is not None else [])
			])

		return self


class Pauli(Object):
	'''
	Pauli class for Quantum Objects
	Args:
		data (iterable[str]): data for operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators
		string (str): string labels of operators
		interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		parameters (object): parameters of operators				
		state (object): state of operators				
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['I','i']},
		**{attr: Object(data=array([[0,1],[1,0]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['X','x']},
		**{attr: Object(data=array([[0,-1j],[1j,0]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['Y','y']},
		**{attr: Object(data=array([[1,0],[0,-1]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['Z','z']},
			}
	default = 'I'
	dim = 2 

	hermitian = True
	unitary = True

	def __call__(self,parameters=None,state=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
			state (object): state		
		Returns:
			data (array): data
		'''
		parameters = self.parameters if parameters is None else parameters
		if parameters is None:
			return self.data
		else:
			return cos(parameters*pi)*self.identity + -1j*sin(parameters*pi)*self.data

	def grad(self,parameters=None,state=None):
		'''
		Call gradient
		Args:
			parameters (array): parameters
			state (object): state
		Returns:
			data (array): data
		'''
		return -pi*sin(parameters*pi)*self.identity + -1j*pi*cos(parameters*pi)*self.data

	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data for operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators
			string (str): string labels of operators
			interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']	
		'''
		return

class Gate(Object):
	'''
	Gate class for Quantum Objects
	Args:
		data (iterable[str]): data for operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators
		string (str): string labels of operators
		interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		parameters (object): parameters of operators				
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['I']},
		**{attr: Object(data=array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),dim=2,locality=2,hermitian=True,unitary=True) for attr in ['CNOT','C','cnot']},
		**{attr: Object(data=array([[1,1,],[1,-1]])/sqrt(2),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['HADAMARD','H']},
		**{attr: Object(data=array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
						  [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]]),dim=2,locality=3,hermitian=True,unitary=True) for attr in ['TOFFOLI','T','toffoli']},
		}
	default = 'I'
	dim = 2 
	
	hermitian = False
	unitary = True
	
	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data for operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators
			string (str): string labels of operators
			interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']	
		'''
		return

class Haar(Object):
	'''
	Haar class for Quantum Objects
	Args:
		data (iterable[str]): data for operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators
		string (str): string labels of operators
		interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		parameters (object): parameters of operators		
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['I']},
		**{attr: Object(data=rand(shape=(2,2),random='haar',dtype='complex'),dim=2,locality=1,hermitian=False,unitary=True) for attr in ['random','U','haar']},
		}
	default = 'I'
	dim = 2

	hermitian = False
	unitary = True

	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data for operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators
			string (str): string labels of operators
			interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']	
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
		data (iterable[str]): data for operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators
		string (str): string labels of operators
		interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		parameters (object): parameters of operators		
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['I']},
		**{attr: Object(data=rand(shape=(2,),random='haar',dtype='complex'),dim=2,locality=1,hermitian=True,unitary=False) for attr in ['random','psi','haar']},
		**{attr: Object(data=array([1,0,]),dim=2,locality=1,hermitian=True,unitary=False) for attr in ['zeros','0']},
		**{attr: Object(data=array([0,1,]),dim=2,locality=1,hermitian=True,unitary=False) for attr in ['ones','1']},
		**{attr: Object(data=array([1,1,])/sqrt(2),dim=2,locality=1,hermitian=True,unitary=False) for attr in ['plus','+']},
		**{attr: Object(data=array([1,-1,])/sqrt(2),dim=2,locality=1,hermitian=True,unitary=False) for attr in ['minus','-']},
		}
	default = 'I'
	dim = 1

	hermitian = True
	unitary = False

	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data for operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators
			string (str): string labels of operators
			interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']	
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
			setitem(data,0,1)
		elif operator in ['ones','1']:
			setitem(data,-1,1)			
		elif operator in ['plus','+']:
			setitem(data,slice(None),1/sqrt(size))			
		elif operator in ['minus','-']:
			setitem(data,slice(None),(-1)**arange(size)/sqrt(size))
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
		data (iterable[str]): data for operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators
		string (str): string labels of operators
		interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		parameters (object): parameters of operators		
		state (object): state of operators		
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['I','i']},
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['D','depolarize']},
		**{attr: Object(data=array([[0,1],[1,0]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['X','x','amplitude']},
		**{attr: Object(data=array([[1,0],[0,0]]),dim=2,locality=1,hermitian=True,unitary=False) for attr in ['00']},
		**{attr: Object(data=array([[0,1],[0,0]]),dim=2,locality=1,hermitian=False,unitary=False) for attr in ['01']},
		**{attr: Object(data=array([[0,0],[1,0]]),dim=2,locality=1,hermitian=False,unitary=False) for attr in ['10']},
		**{attr: Object(data=array([[0,0],[0,1]]),dim=2,locality=1,hermitian=True,unitary=False) for attr in ['11']},
		**{attr: Object(data=array([[0,-1j],[1j,0]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['Y','y','amplitude_phase']},
		**{attr: Object(data=array([[1,0],[0,-1]]),dim=2,locality=1,hermitian=True,unitary=True) for attr in ['Z','z','phase']},
		}
	default = 'I'
	dim = 2 
	
	hermitian = False
	unitary = False

	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data for operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators
			string (str): string labels of operators
			interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']	
		'''

		if (getattr(self,'scale',None) is not None):
			if (getattr(self,'initialization',None) in ['time']):
				if (getattr(self,'tau') is not None):
					self.parameters = (1 - exp(-self.tau/self.scale))/2

		if (self.parameters is None) or (self.parameters is False) or (self.parameters is True):
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


def Compute(data,identity,state,noise,coefficients,n,d,m,p):
	'''
	Calculate operators
	Args:
		data (iterable[Operator]): Array of data to matrix exponentiate of shape (d,n,n)
		identity (Operator): Array of data identity of shape (n,n)
		state (Operator): Array of state to act on of shape (n,) or (n,n) or (p,n) or (p,n,n)
		noise (Operator): Array of noise to act of shape (n,n) or (k,n,n)
		coefficients (arrray): Array of coefficients of shape (1,) or (d,)
		n (int): Size of array
		d (int): Number of data
		m (int): Number of steps
		p (int): number of trotterizations
	Returns:
		func (callable): Function to compute operators, with signature func(parameters), where parameters is a Parameters instance
		grad (callable): Gradient to compute operators, with signature func(parameters), where parameters is a Parameters instance
	'''		

	if coefficients is None:
		coefficients = 1
	if isinstance(coefficients,scalars):
		coefficients = [coefficients]*d

	def func(parameters):
		out = identity
		for j in range(m):
			operators = [data[i](coefficients[i]*parameters[i][j]) for i in range(d)]
			operators = trotter(operators,p)
			for operator in operators:
				out = dot(operator,out)
		return out
	
	# def grad(parameters):
	# 	out = 0
		# grad = []
		# for j in range(m):
		# 	for k in range(d*p):
		# 	for i in range(m):
		# 		operators = [data[i](coefficients[i]*parameter[i]) for i in range(d)]
		# 		operators = trotter(operators,p)
		# 		for operator in operators:
		# 			out = dot(operator,out)
		# return out	
		# out = []
		# for i in range(m):
		# 	for j in range(m):
		# 		for k in range(d)
		# 		tmp = identity

		# 	operators = [data(indexer([*data.index,i],parameters))*(1 if) for data in self.data]
		# 	operators = trotter(operators,p)
		# 	for operator in operators:
		# 		out = dot(operator,out)
		# return out		

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
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
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

	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
		N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
		space=None,time=None,lattice=None,parameters=None,state=None,noise=None,system=None,**kwargs):

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

		self.n = None
		self.g = None
		
		self.parameters = parameters
		self.state = state
		self.noise = noise
		self.identity = None
		self.coefficients = None
		self.index = None
		self.compute = None

		self.system = system

		self.__time__()
		self.__space__()
		self.__lattice__()

		self.identity = Operator(Operator.default,N=self.N,D=self.D,system=self.system,verbose=False)
		self.shape = () if self.n is None else (self.n,self.n)
		self.size = prod(self.shape)
		self.ndim = len(self.shape)
		self.coefficients = self.tau/self.P

		self.__setup__(data,operator,site,string,interaction)

		self.info()

		return	

	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
				string (iterable[str]): string labels of operators
				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''

		# Get operator,site,string,interaction from data
		objs = {'operator':operator,'site':site,'string':string,'interaction':interaction}

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


	def __append__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Append to class
		Args:
			data (str,Operator): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''
		index = -1
		self.__insert__(index,data,operator,site,string,interaction)
		return

	def __extend__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup class
		Args:
			data (iterable[str,Operator]): data of operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''
		if all(i is None for i in [operator,site,string,interaction]):
			return

		if data is None:
			size = min([len(i) for i in [operator,site,string,interaction]])
			data = [None]*size

		for _data,_operator,_site,_string,_interaction in zip(data,operator,site,string,interaction):
			self.__append__(_data,_operator,_site,_string,_interaction)

		return


	def __insert__(self,index,data,operator,site,string,interaction):
		'''
		Insert to class
		Args:
			index (int): index to insert operator
			data (str,Operator): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''

		if index == -1:
			index = len(self)

		kwargs = dict(N=self.N,D=self.D,system=self.system,verbose=False)

		data = Operator(data=data,operator=operator,site=site,string=string,interaction=interaction,**kwargs)

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
		self.compute = Compute(data,identity,state,noise,coefficients,n,d,m,p)

		# Update class attributes
		self.gradient = gradient(self,mode='fwd',move=True)

		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Class function
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function
		'''		
		parameters = self.parameters(parameters)

		return self.compute(parameters)

	#@partial(jit,static_argnums=(0,))
	def __grad__(self,parameters):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
		Returns:
			out (array): Return of function
		'''	
		return self.gradient(parameters)


	# @partial(jit,static_argnums=(0,))
	def __value_and_grad__(self,parameters):
		'''
		Class function and gradient
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function and gradient
		'''	
		return self.value_and_gradient(parameters)


	#@partial(jit,static_argnums=(0,))
	def func(self,parameters):
		'''
		Class function
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function
		'''
		return self.__call__(parameters)

	#@partial(jit,static_argnums=(0,))
	def grad(self,parameters):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
		Returns:
			out (array): Return of function
		'''		
		return self.__grad__(parameters)

	# @partial(jit,static_argnums=(0,))
	def value_and_grad(self,parameters):
		'''
		Class function and gradient
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function and gradient
		'''	
		return self.__value_and_gradient__(parameters)

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
				for attr in ['string','key','seed','N','D','d','L','delta','M','tau','T','P','n','g','unit','shape','cwd','path','dtype','backend','architecture','conf','logger','cleanup']
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
				for attr in ['state','noise']
			],
			*['%s:\n%s'%(delim.join(attr.split(delim)[:1]),
				to_string(getattrs(self,attr,default=lambda:None)()))
				for attr in ['state','noise'] if getattrs(self,attr,default=lambda:None)() is not None
			],			
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
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
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

	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
				space=None,time=None,lattice=None,parameters=None,state=None,noise=None,system=None,**kwargs):

		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
				space=space,time=time,lattice=lattice,parameters=parameters,state=state,noise=noise,system=system,**kwargs)
		
		return

	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
				string (iterable[str]): string labels of operators
				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''
		# Get operator,site,string,interaction from data
		objs = {'operator':operator,'site':site,'string':string,'interaction':interaction}

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

		# Lattice sites
		sites = {site: self.lattice(site) for site in ['i','i<j','<ij>','i...j']}	# sites types on lattice
		indices = {'i': ['i'],'<ij>':['i','j'],'i<j':['i','j'],'i...j':['i','j']}   # allowed symbolic indices and maximum locality of many-body site interactions

		# Get number of operators
		size = min(len(objs[obj]) for obj in objs)
		
		# Get all indices from symbolic indices
		for index in range(size):
			
			size -= 1
			
			_objs = {}
			for obj in objs:
				value = deepcopy(objs[obj].pop(0))
				if obj in ['site']:
					if isinstance(value,str):
						value = indices[value]
				_objs[obj] = value

			if any(i in indices[_objs['interaction']] for i in _objs['site']):
				for i,s in enumerate(sites[_objs['interaction']]):
					for obj in objs:
						if obj in ['site']:
							value = [dict(zip(
								indices[_objs['interaction']],
								s if not isinstance(s,int) else [s])
							).get(i,parse(i,int)) 
							for i in _objs['site']]
						else:
							value = _objs[obj]

						objs[obj].append(value)			

			elif len(_objs['operator']) == len(_objs['site']):
				for obj in objs:
					value = _objs[obj]
					objs[obj].append(value)					

			else:
				for obj in objs:
					value = _objs[obj]
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
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
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

	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
				space=None,time=None,lattice=None,parameters=None,state=None,noise=None,system=None,**kwargs):
		
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
				space=space,time=time,lattice=lattice,parameters=parameters,state=state,noise=noise,system=system,**kwargs)

		return

	#@partial(jit,static_argnums=(0,))
	def __grad_analytical__(self,parameters):
		'''
		Class gradient
		Args:
			parameters (array): parameters		
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

	#@partial(jit,static_argnums=(0,))
	def grad_analytical(self,parameters):
		'''
		Class gradient
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function
		'''	
		return self.__grad_analytical__(parameters)


class Label(Operator):
	
	def __init__(self,*args,**kwargs):
		'''	
		Class for label
		Args:
			args (tuple): Class arguments
			kwargs (dict): Class keyword arguments
		'''	

		super().__init__(*args,**kwargs)

		data = self.data
		state = self.state

		if state is None:
			pass
		elif state.ndim == 1:
			data = einsum('ij,j->i',data,state)
		elif state.ndim == 2:
			data = einsum('ij,jk,kl->il',data,state,dagger(data))
		
		self.data = data
		self.shape = data.shape
		self.size = data.size
		self.ndim = data.ndim

		return

class Callback(object):
	def __init__(self,*args,**kwargs):
		'''	
		Class for callback
		Args:
			args (tuple): Class arguments
			kwargs (dict): Class keyword arguments
		'''

		self.defaults = {
			'iteration':[],
			'parameters':[],'grad':[],'search':[],
			'value':[],'objective':[],
			'alpha':[],'beta':[],

			'iteration.max':[],'iteration.min':[],
			'variables':[],'variables.relative':[],'variables.relative.mean':[],
			'features':[],'features.relative':[],'features.relative.mean':[],
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
					'variables','features',
					'variables.relative','features.relative',
					'hessian','fisher',
					'hessian.eigenvalues','fisher.eigenvalues']},
			**{attr: None for attr in [
				'parameters.norm','grad.norm','search.norm',
				'variables.norm','features.norm'
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
					'variables','features',
					'variables.relative','features.relative',
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

				elif attr in [
					'variables.norm','variables.relative','variables.relative.mean',
					'features.norm','features.relative','features.relative.mean'] and (not do):
					value = default

				elif attr in [
					'variables','variables.norm','variables.relative','variables.relative.mean',
					'features','features.norm','features.relative','features.relative.mean'] and (do):

					if attr in ['variables','features']:
						value = model.parameters(parameters)
					elif attr in ['variables.norm','features.norm']:
						value = model.parameters(parameters)
						value = norm(value)/(value.size)
					elif attr in ['variables.relative','features.relative']:
						eps = 1e-20
						value = model.parameters(parameters)
						_value = model.parameters(attributes['parameters'][0])
						value = abs((value - _value + eps)/(_value + eps))
					elif attr in ['variables.relative.mean','features.relative.mean']:
						eps = 1e-20
						value = model.parameters(parameters)
						_value = model.parameters(attributes['parameters'][0])
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
				# 'U\n%s\nV\n%s'%(
				# 	to_string((model(parameters)).round(4)),
				# 	to_string((model.label()).round(4))),
				])


			model.log(msg)


		return status



