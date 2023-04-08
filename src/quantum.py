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
from src.utils import tensorprod,product,dagger,einsum,dot
from src.utils import summation,exponentiation,summationv,exponentiationv,summationm,exponentiationm,summationmvc,exponentiationmvc,summationmmc,exponentiationmmc
from src.utils import trotter,gradient_trotter,gradient_expm
from src.utils import eig
from src.utils import maximum,minimum,argmax,argmin,difference,abs,sqrt,log,log10,sign,sin,cos
from src.utils import sort,relsort,norm
from src.utils import initialize,parse,to_string,is_array
from src.utils import pi,e,nan,null,delim,scalars,nulls
from src.utils import itg,flt,dbl

from src.iterables import setter,getter,getattrs,hasattrs,indexer,inserter

# from src.parameters import Parameters
# from src.operators import Gate
# from src.states import State
# from src.noise import Noise

from src.io import load,dump,join,split

from src.system import System,Space,Time,Lattice

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
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {}
	default = None
	dim = None

	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,system=None,**kwargs):		

		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		if isinstance(data,Object):
			for attr in data:
				setattr(self,attr,getattr(data,attr))
			return

		N = max(getattr(self,'N',(max(site if not isinstance(site,int) else [site])+1) if site is not None else 0),(max(site if not isinstance(site,int) else [site])+1) if site is not None else 0)
		D = min(getattr(self,'D',self.dim),self.dim) if self.dim is not None else getattr(self,'D',data.size**(1/(data.ndim*N))) if data is not None else None
		default = self.default			

		if not operator:
			operator = data			
		
		if operator is None:
			pass
		elif isinstance(operator,str):
			operator = [i for i in operator.split(delim)]
		elif not is_array(operator):
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
		
		

		if operator is None:
			pass
		elif len(operator) == N:
			site = list(range(N))
			operator = list((operator[i] for i in range(N)))
		elif site is not None and len(operator) == len(site):
			site = list(range(N)) if site is None else site
			operator = list((operator[site.index(i)%len(operator)] if i in site else default for i in range(N) if (i not in site) or (site.index(i)<len(operator))))
		else:
			site = list(range(N)) if site is None else site

		self.data = data if data is not None else None
		self.operator = operator if operator is not None else None
		self.site = site if site is not None else None
		self.string = string if string is not None else None
		self.interaction = interaction if interaction is not None else None
		self.system = system

		self.N = N
		self.D = D

		for attr in ['identity','index','parameters']:
			setattr(self,attr,getattr(self,attr,None))
		
		for attr in ['shape','size','ndim']:
			setattr(self,attr,getattr(self.data,attr,getattr(self,attr,None)) if self.data is not None else getattr(self,attr,None))
		
		self.__setup__(data,operator,site,string,interaction)

		self.data = self.data if is_array(self.data) else tensorprod([self.basis.get(i,self.basis.get(self.default))() for i in self.operator]) if (self.operator is not None) and (not is_array(self.operator)) else self.operator if self.operator is not None else None
		self.data = self.data.astype(self.dtype) if self.data is not None else None

		self.operator = list((i for i in self.operator)) if self.operator is not None else None
		self.site = list((i for i in self.site)) if self.site is not None else None
		self.string = str(self.string) if self.string is not None else None
		self.interaction = str(self.interaction) if self.interaction is not None else None

		self.identity = tensorprod([self.basis.get(self.default)() for i in range(self.N)]) if (self.default in self.basis) else None
		self.identity = self.identity.astype(self.dtype) if self.identity is not None else None

		for attr in ['shape','size','ndim']:
			setattr(self,attr,getattr(self.data,attr,getattr(self,attr,None)) if self.data is not None else getattr(self,attr,None))

		for attr in ['locality']:
			setattr(self,attr,getattr(self,attr,len(self.site) if self.site is not None else None))

		for attr in ['D']:
			setattr(self,attr,max(getattr(self,attr,self.dim),self.dim)) if self.dim is not None else None

		for attr in ['N']:
			setattr(self,attr,max(getattr(self,attr,0),int(log(self.size)/self.ndim/log(self.D)))) if self.data is not None else None

		
		# Assert data is normalized
		if self.ndim > 3:
			normalization = einsum('...uij,...ukj->...ik',self.data.conj(),self.data)
			eps = array([identity(self.shape[-2:],dtype=self.dtype)]*(normalization.ndim-2),dtype=self.dtype)
		elif self.ndim == 3:
			normalization = einsum('uij,ukj->ik',self.data.conj(),self.data)
			eps = identity(self.shape[-2:],dtype=self.dtype)
		elif self.ndim == 2:
			normalization = einsum('ij,kj->ik',self.data.conj(),self.data)
			eps = identity(self.shape[-2:],dtype=self.dtype)
		else:
			normalization = einsum('i,i->',self.data.conj(),self.data)
			eps = ones(shape=(),dtype=self.dtype)

		assert (eps.shape == normalization.shape), "Incorrect operator shape %r != %r"%(eps.shape,normalization.shape)

		if self.dtype in ['complex128','float64']:
			assert allclose(eps,normalization), "Incorrect normalization data%r: %r"%(data.shape,normalization)

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

	def __call__(self,parameters=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
		Returns:
			operator (array): operator
		'''
		return self.data

	def __str__(self):
		return delim.join(self.operator)

	def __repr__(self):
		return self.__str__()
	
	def __len__(self):
		return len(self.operator)
	

class Operator(Object):
	'''
	Class for Operator
	Args:
		data (iterable[str]): data for operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators
		string (str): string labels of operators
		interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''
	def __new__(cls,data=None,operator=None,site=None,string=None,interaction=None,system=None,**kwargs):		

		# TODO: Allow multiple different classes to be part of one operator

		self = None

		if (data is None) and (operator is None):
			return self

		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,interaction=interaction,system=system),delimiter=delim,func=False)

		classes = [Pauli,Gate,Haar,Noise]
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
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1) for attr in ['I','i']},
		**{attr: Object(data=array([[0,1],[1,0]]),dim=2,locality=1) for attr in ['X','x']},
		**{attr: Object(data=array([[0,-1j],[1j,0]]),dim=2,locality=1) for attr in ['Y','y']},
		**{attr: Object(data=array([[1,0],[0,-1]]),dim=2,locality=1) for attr in ['Z','z']},
			}
	default = 'I'
	dim = 2 

	def __call__(self,parameters=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
		Returns:
			operator (array): operator
		'''
		parameters = self.parameters if parameters is None else parameters

		if parameters is None:
			return self.data
		else:
			return cos(parameters*pi)*self.identity + -1j*sin(parameters*pi)*self.data

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
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1) for attr in ['I']},
		**{attr: Object(data=array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),dim=2,locality=2) for attr in ['CNOT','C']},
		**{attr: Object(data=array([[1,1,],[1,-1]])/sqrt(2),dim=2,locality=1) for attr in ['HADAMARD','H']},
		**{attr: Object(data=array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
						  [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]]),dim=2,locality=3) for attr in ['TOFFOLI','T']},
		**{attr: Object(data=array([1,0,]),dim=2,locality=1) for attr in ['zeros','0']},
		**{attr: Object(data=array([0,1,]),dim=2,locality=1) for attr in ['ones','1']},
		**{attr: Object(data=array([1,1,])/sqrt(2),dim=2,locality=1) for attr in ['plus','+']},
		**{attr: Object(data=array([1,-1,])/sqrt(2),dim=2,locality=1) for attr in ['minus','-']},
		}
	default = 'I'
	dim = 2 

	def __call__(self,parameters=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
		Returns:
			operator (array): operator
		'''
		parameters = self.parameters if parameters is None else parameters

		return self.data
	
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
		else:
			data = self.data
		
		self.data = data
		
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
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1) for attr in ['I']},
		**{attr: Object(data=rand(shape=(2,2),random='haar',dtype='complex'),dim=2,locality=1) for attr in ['random','U','haar']},
		}
	default = 'I'
	dim = 2

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
		dtype = self.dtype

		data = rand(shape=shape,random=random,seed=seed,dtype=dtype)

		self.data = data

		return

	def __call__(self,parameters=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
		Returns:
			operator (array): operator
		'''
		parameters = self.parameters if parameters is None else parameters

		return self.data


class Noise(Object):
	'''
	Noise class for Quantum Objects
	Args:
		data (iterable[str]): data for operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators
		string (str): string labels of operators
		interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	basis = {
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1) for attr in ['I','i']},
		**{attr: Object(data=array([[1,0],[0,1]]),dim=2,locality=1) for attr in ['D','depolarize']},
		**{attr: Object(data=array([[0,1],[1,0]]),dim=2,locality=1) for attr in ['X','x','amplitude']},
		**{attr: Object(data=array([[1,0],[0,0]]),dim=2,locality=1) for attr in ['00']},
		**{attr: Object(data=array([[0,1],[0,0]]),dim=2,locality=1) for attr in ['01']},
		**{attr: Object(data=array([[0,0],[1,0]]),dim=2,locality=1) for attr in ['10']},
		**{attr: Object(data=array([[0,0],[0,1]]),dim=2,locality=1) for attr in ['11']},
		**{attr: Object(data=array([[0,-1j],[1j,0]]),dim=2,locality=1) for attr in ['Y','y','amplitude_phase']},
		**{attr: Object(data=array([[1,0],[0,-1]]),dim=2,locality=1) for attr in ['Z','z','phase']},
		}
	default = 'I'
	dim = 2 

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
				if (getattr(self,'model',{}).get('tau') is not None):
					self.parameters = (1 - exp(-self.model.tau/self.scale))/2

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
	

		data = array([
			tensorprod(i)
			for i in itertools.product(data,repeat=self.N)
			],dtype=self.dtype)

		self.data = data

		return

	def __call__(self,parameters=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
		Returns:
			operator (array): operator
		'''
		parameters = self.parameters if parameters is None else parameters

		return self.data


# class Operators(Operator):
# 	'''
# 	Class for Operators
# 	Args:
# 		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
# 			operator (iterable[str]): string names of operators
# 			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 			string (iterable[str]): string labels of operators
# 			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		operator (iterable[str]): string names of operators
# 		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 		string (iterable[str]): string labels of operators
# 		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		N (int): Number of qudits
# 		D (int): Dimension of qudits
# 		d (int): Spatial dimension
# 		L (int,float): Scale in system
# 		delta (float): Simulation length scale		
# 		M (int): Number of time steps
# 		T (int): Simulation Time
# 		tau (float): Simulation time scale		
# 		P (int): Trotter order		
# 		space (str,Space): Type of local space
# 		time (str,Time): Type of Time evolution space						
# 		lattice (str,Lattice): Type of lattice	
# 		parameters (str,dict,Parameters): Type of parameters	
# 		state (str,dict,State): Type of state	
# 		noise (str,dict,Noise): Type of noise
# 		label (str,dict,Gate): Type of label	
# 		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
# 		kwargs (dict): Additional system keyword arguments	
# 	'''

# 	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
# 		N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
# 		space=None,time=None,lattice=None,parameters=None,state=None,noise=None,label=None,system=None,**kwargs):

# 		setter(kwargs,system,delimiter=delim,func=False)
# 		super().__init__(**kwargs)

# 		self.N = N
# 		self.D = D
# 		self.d = d
# 		self.L = L
# 		self.delta = delta
# 		self.M = M
# 		self.T = T
# 		self.tau = tau
# 		self.P = P
# 		self.space = space
# 		self.time = time
# 		self.lattice = lattice

# 		self.data = []
# 		self.operator = []
# 		self.site = []
# 		self.string = []
# 		self.interaction = []

# 		self.n = None
# 		self.g = None
		
# 		self.parameters = parameters
# 		self.state = state
# 		self.noise = noise
# 		self.label = label
# 		self.identity = None
# 		self.constants = None
# 		self.coefficients = None
# 		self.index = None

# 		self.summation = None
# 		self.exponentiation = None 
# 		self.hermitian = False

# 		self.system = system

# 		self.__shape__()
# 		self.__time__()
# 		self.__space__()
# 		self.__lattice__()

# 		self.__setup__(data,operator,site,string,interaction)

# 		self.info()

# 		return	

# 	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
# 		'''
# 		Setup class
# 		Args:
# 			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
# 				operator (iterable[str]): string names of operators
# 				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 				string (iterable[str]): string labels of operators
# 				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 			operator (iterable[str]): string names of operators
# 			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 			string (iterable[str]): string labels of operators
# 			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		'''

# 		# Get operator,site,string,interaction from data
# 		objs = {'operator':operator,'site':site,'string':string,'interaction':interaction}

# 		for obj in objs:
# 			objs[obj] = [] if objs[obj] is None else objs[obj]

# 		if data is None:
# 			data = {}
# 		elif all(isinstance(datum,Operator) for datum in data):
# 			data = {datum.timestamp: datum for datum in data}
		
# 		assert isinstance(data,dict), 'Incorrect data format %r'%(type(data))			

# 		if all(isinstance(data[name],dict) and (obj in data[name]) for name in data for obj in objs):
# 			for obj in objs:
# 				objs[obj].extend([data[name][obj] for name in data])

# 		# Set class attributes
# 		self.__extend__(**objs)

# 		return


# 	def __append__(self,data=None,operator=None,site=None,string=None,interaction=None):
# 		'''
# 		Append to class
# 		Args:
# 			data (str,Operator): data of operator
# 			operator (str): string name of operator
# 			site (int): site of local operator
# 			string (str): string label of operator
# 			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		'''
# 		index = -1
# 		self.__insert__(index,data,operator,site,string,interaction)
# 		return

# 	def __extend__(self,data=None,operator=None,site=None,string=None,interaction=None):
# 		'''
# 		Setup class
# 		Args:
# 			data (iterable[str,Operator]): data of operator
# 			operator (iterable[str]): string names of operators
# 			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 			string (iterable[str]): string labels of operators
# 			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		'''
# 		if all(i is None for i in [operator,site,string,interaction]):
# 			return

# 		if data is None:
# 			size = min([len(i) for i in [operator,site,string,interaction]])
# 			data = [None]*size

# 		for _data,_operator,_site,_string,_interaction,_hyperparameter in zip(data,operator,site,string,interaction):
# 			self.__append__(_data,_operator,_site,_string,_interaction,_hyperparameter)

# 		return


# 	def __insert__(self,index,data,operator,site,string,interaction):
# 		'''
# 		Insert to class
# 		Args:
# 			index (int): index to insert operator
# 			data (str,Operator): data of operator
# 			operator (str): string name of operator
# 			site (int): site of local operator
# 			string (str): string label of operator
# 			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		'''

# 		if index == -1:
# 			index = len(self.data)

# 		data = Operator(data=data,operator=operator,site=site,string=string,interaction=interaction,
# 						N=self.N,D=self.D,system=self.system)

# 		self.data.insert(index,data)
# 		self.operator.insert(index,operator)
# 		self.site.insert(index,site)
# 		self.string.insert(index,string)
# 		self.interaction.insert(index,interaction)

# 		self.__shape__()

# 		return

# 	def __initialize__(self,parameters=None,state=None,noise=None,label=None):
# 		''' 
# 		Setup class functions
# 		Args:
# 			parameters (bool,dict,array,Parameters): Class parameters
# 			state (bool,dict,array,State): State to act on with class of shape self.shape, or class hyperparameters, or boolean to choose self.state or None
# 			noise (bool,dict,array,Noise): Noise to act on with class of shape (-1,self.shape), or class hyperparameters, or boolean to choose self.noise or None
# 			label (bool,dict,array,Gate): Label of class of shape self.shape, or class hyperparameters, or boolean to choose self.label or None
# 		'''

# 		# Get parameters
# 		parameters = self.parameters if parameters is None or parameters is True else parameters if parameters is not False else None
# 		system = self.system
# 		index = {string: [[(*data.site) for data in self.data if data.string==string],self.M] for string in set(self.string)}

# 		self.parameters = Parameters(parameters,index,system=system)
# 		parameters = self.parameters()

# 		# Get coefficients
# 		coefficients = self.tau/self.P

# 		for data in self.data:
# 			data.index = [data.string,*[i.index((*data.site)) for i in parameters.index[data.string][:-1]]]

# 		exit()


# 		# Function arguments
# 		data = array(trotter([data() for data in self.data],self.P))
# 		identity = self.identity()
# 		state = self.state if state is None or state is True else state if state is not False else None
# 		noise = self.noise if noise is None or noise is True else noise if noise is not False else None
# 		label = self.label if label is None or label is True else label if label is not False else None

# 		# Class arguments
# 		kwargs = {}
# 		shape = self.shape
# 		dims = [self.N,self.D]
# 		system = self.system
# 		cls = {attr: getattr(self,attr) for attr in self if attr not in ['parameters','state','noise','label']}


# 		# Get state
# 		kwargs.clear()
# 		setter(kwargs,self.state)
# 		setter(kwargs,state)
# 		setter(kwargs,dict(data=state,shape=shape,dims=dims,cls=cls,system=system))
# 		self.state = State(**kwargs)
# 		state = self.state()

# 		# Get noise
# 		kwargs.clear()
# 		setter(kwargs,self.noise)
# 		setter(kwargs,noise)
# 		setter(kwargs,dict(data=noise,shape=shape,dims=dims,cls=cls,system=system))
# 		self.noise = Noise(**kwargs)
# 		noise = self.noise()

# 		# Get label
# 		kwargs.clear()
# 		setter(kwargs,self.label)
# 		setter(kwargs,label)
# 		setter(kwargs,dict(data=label,shape=shape,dims=dims,cls=cls,system=system))
# 		self.label = Gate(**kwargs)
# 		label = self.label()

# 		# Attribute values
# 		if state is None:
# 			label = label
# 		elif state.ndim == 1:
# 			label = einsum('ij,j->i',label,state)
# 		elif state.ndim == 2:
# 			label = einsum('ij,jk,kl->il',label,state,dagger(label))
# 		else:
# 			label = label
# 		label = self.label(label)

# 		# Operator functions
# 		if state is None and noise is None:
# 			self.summation = jit(summation,data=data,identity=identity)
# 			self.exponentiation = jit(exponentiation,data=data,identity=identity)
# 			self.hermitian = False
# 		elif state is not None and noise is None:
# 			if state.ndim == 1:
# 				self.summation = jit(summationv,data=data,identity=identity,state=state)
# 				self.exponentiation = jit(exponentiationv,data=data,identity=identity,state=state)
# 				self.hermitian = True
# 			elif state.ndim == 2:
# 				self.summation = jit(summationm,data=data,identity=identity,state=state)
# 				self.exponentiation = jit(exponentiationm,data=data,identity=identity,state=state)
# 				self.hermitian = True
# 			else:
# 				self.summation = jit(summation,data=data,identity=identity)
# 				self.exponentiation = jit(exponentiation,data=data,identity=identity)
# 				self.hermitian = False
# 		elif state is None and noise is not None:
# 			self.summation = jit(summation,data=data,identity=identity)
# 			self.exponentiation = jit(exponentiation,data=data,identity=identity)
# 			self.hermitian = False
# 		elif state is not None and noise is not None:
# 			if state.ndim == 1:
# 				self.summation = jit(summationmvc,data=data,identity=identity,state=state,constants=noise)
# 				self.exponentiation = jit(exponentiationmvc,data=data,identity=identity,state=state,constants=noise)
# 				self.hermitian = True
# 			elif state.ndim == 2:
# 				self.summation = jit(summationmmc,data=data,identity=identity,state=state,constants=noise)
# 				self.exponentiation = jit(exponentiationmmc,data=data,identity=identity,state=state,constants=noise)
# 				self.hermitian = True
# 			else:
# 				self.summation = jit(summation,data=data,identity=identity)
# 				self.exponentiation = jit(exponentiation,data=data,identity=identity)
# 				self.hermitian = False
# 		else:
# 			self.summation = jit(summation,data=data,identity=identity)
# 			self.exponentiation = jit(exponentiation,data=data,identity=identity)
# 			self.hermitian = False


# 		# Update class attributes
# 		self.gradient = gradient(self,mode='fwd',move=True)
# 		self.parameters = parameters
# 		self.coefficients = coefficients
# 		self.index = index

# 		return

# 	#@partial(jit,static_argnums=(0,))
# 	def __call__(self,parameters):
# 		'''
# 		Class function
# 		Args:
# 			parameters (array): parameters		
# 		Returns
# 			out (array): Return of function
# 		'''		
# 		parameters = self.parameters(parameters)

# 		data = sum(data(self.parameters) for data in self.data)

# 		return data

# 	#@partial(jit,static_argnums=(0,))
# 	def __grad__(self,parameters):
# 		''' 
# 		Class gradient
# 		Args:
# 			parameters (array): parameters
# 		Returns:
# 			out (array): Return of function
# 		'''	
# 		return self.gradient(parameters)


# 	# @partial(jit,static_argnums=(0,))
# 	def __value_and_grad__(self,parameters):
# 		'''
# 		Class function and gradient
# 		Args:
# 			parameters (array): parameters		
# 		Returns
# 			out (array): Return of function and gradient
# 		'''	
# 		return self.value_and_gradient(parameters)


# 	#@partial(jit,static_argnums=(0,))
# 	def func(self,parameters):
# 		'''
# 		Class function
# 		Args:
# 			parameters (array): parameters		
# 		Returns
# 			out (array): Return of function
# 		'''
# 		return self.__call__(parameters)

# 	#@partial(jit,static_argnums=(0,))
# 	def grad(self,parameters):
# 		''' 
# 		Class gradient
# 		Args:
# 			parameters (array): parameters
# 		Returns:
# 			out (array): Return of function
# 		'''		
# 		return self.__grad__(parameters)

# 	# @partial(jit,static_argnums=(0,))
# 	def value_and_grad(self,parameters):
# 		'''
# 		Class function and gradient
# 		Args:
# 			parameters (array): parameters		
# 		Returns
# 			out (array): Return of function and gradient
# 		'''	
# 		return self.__value_and_gradient__(parameters)


# 	def __shape__(self,data=None,N=None,D=None,M=None):
# 		'''
# 		Set shape attributes
# 		Args:
# 			data (iterable[Operator]): Class data
# 			N (int): Number of qudits
# 			D (int): Dimension of qudits
# 			M (int): Number of time steps
# 		'''

# 		if (data is not None):
# 			self.__append__(data)
# 		if (N is not None) and (D is not None):
# 			self.__space__(N=N,D=D)
# 		if (M is not None):
# 			self.__time__(M=M)

# 		self.shape = () if self.n is None else (self.n,self.n)
# 		self.size = int(product(self.shape))
# 		self.ndim = len(self.shape)
# 		self.dims = (self.M,len(self.data),*self.shape)
# 		self.length = int(product(self.dims))
# 		self.ndims = len(self.dims)

# 		return

# 	def __space__(self,N=None,D=None,space=None,system=None):
# 		'''
# 		Set space attributes
# 		Args:
# 			N (int): Number of qudits
# 			D (int): Dimension of qudits
# 			space (str,Space): Type of local space
# 			system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
# 		'''

# 		N = self.N if N is None else N
# 		D = self.D if D is None else D
# 		space = self.space if space is None else space
# 		system = self.system if system is None else system

# 		self.space = Space(N,D,space,system=system)

# 		self.N = self.space.N
# 		self.D = self.space.D		
# 		self.n = self.space.n
# 		self.g = self.space.g

# 		self.__shape__()

# 		self.identity = Operator(N=self.N,D=self.D,system=self.system)

# 		return


# 	def __time__(self,M=None,T=None,tau=None,P=None,time=None,system=None):
# 		'''
# 		Set time attributes
# 		Args:
# 			M (int): Number of time steps
# 			T (int): Simulation Time
# 			tau (float): Simulation time scale
# 			P (int): Trotter order		
# 			time (str,Time): Type of Time evolution space						
# 			system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
# 		'''
# 		M = self.M if M is None else M
# 		T = self.T if T is None else T
# 		tau = self.tau if tau is None else tau
# 		P = self.P if P is None else P
# 		time = self.time if time is None else time
# 		system = self.system if system is None else system

# 		self.time = Time(M,T,tau,P,time,system=system)	

# 		self.M = self.time.M
# 		self.T = self.time.T
# 		self.P = self.time.P
# 		self.tau = self.time.tau
		
# 		self.__shape__()

# 		return


# 	def __lattice__(self,N=None,D=None,d=None,L=None,delta=None,lattice=None,system=None):
# 		'''
# 		Set space attributes
# 		Args:
# 			N (int): Number of qudits
# 			D (int): Dimension of qudits
# 			d (int): Spatial dimension
# 			L (int,float): Scale in system
# 			delta (float): Simulation length scale		
# 			lattice (str,Lattice): Type of lattice		
# 			system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
# 		'''		
# 		N = self.N if N is None else N
# 		D = self.D if D is None else D
# 		d = self.d if d is None else d
# 		L = self.L if L is None else L
# 		delta = self.delta if delta is None else delta
# 		lattice = self.lattice if lattice is None else lattice
# 		system = self.system if system is None else system

# 		self.lattice = Lattice(N,d,L,delta,lattice,system=system)	

# 		self.N = self.lattice.N
# 		self.d = self.lattice.d
# 		self.L = self.lattice.L
# 		self.delta = self.lattice.delta

# 		return

# 	def __str__(self):
# 		size = len(self.data)
# 		delimiter = ' '
# 		multiple_time = (self.M>1) if self.M is not None else None
# 		multiple_space = [size>1 and False for i in range(size)]
# 		return '%s%s%s%s'%(
# 				'{' if multiple_time else '',
# 				delimiter.join(['%s%s%s'%(
# 					'(' if multiple_space[i] else '',
# 					self.string[i],
# 					')' if multiple_space[i] else '',
# 					) for i in range(size)]),
# 				'}' if multiple_time else '',
# 				'%s'%('^%s'%(self.M) if multiple_time else '') if multiple_time else '')

# 	def info(self,verbose=None):
# 		'''
# 		Log class information
# 		Args:
# 			verbose (int,str): Verbosity of message			
# 		'''		

# 		msg = '%s'%('\n'.join([
# 			*['%s: %s'%(attr,getattrs(self,attr,delimiter=delim)) 
# 				for attr in ['key','seed','N','D','d','L','delta','M','tau','T','P','n','g','unit','shape','dims','cwd','path','dtype','backend','architecture','conf','logger','cleanup']
# 			],
# 			*['%s: %s'%(delim.join(attr.split(delim)[:2]),', '.join([
# 				('%s' if (
# 					(getattrs(self,delim.join([attr,prop]),delimiter=delim) is None) or 
# 					isinstance(getattrs(self,delim.join([attr,prop]),delimiter=delim),str)) 
# 				else '%0.3e')%(getattrs(self,delim.join([attr,prop]),delimiter=delim))
# 				for prop in ['category','method','shape','parameters']]))
# 				for attr in ['parameters.%s'%(i) for i in self.parameters]
# 			],
# 			*['%s: %s'%(delim.join(attr.split(delim)[:1]),', '.join([
# 				('%s' if (
# 					(getattrs(self,delim.join([attr,prop]),delimiter=delim) is None) or 
# 					isinstance(getattrs(self,delim.join([attr,prop]),delimiter=delim),str)) 
# 				else '%0.3e')%(getattrs(self,delim.join([attr,prop]),delimiter=delim))
# 				for prop in ['string','parameters']]))
# 				for attr in ['label','state','noise']
# 			],
# 			*['%s: %s'%(attr,getattrs(self,attr,delimiter=delim).__name__) 
# 				for attr in ['exponentiation']
# 			],			
# 			]
# 			))
# 		self.log(msg,verbose=verbose)

# 		return


# 	def dump(self,path=None):
# 		'''
# 		Save class data		
# 		Args:
# 			path (str,dict[str,(str,bool)]): Path to dump class data, either path or boolean to dump			
# 		'''

# 		# Set data
# 		data = {}

# 		# Set path
# 		paths = {}
# 		if path is None:
# 			paths.update({attr: True for attr in data})			
# 		elif not isinstance(path,dict):
# 			paths.update({attr: path for attr in data if path})
# 		else:
# 			paths.update({attr: path[attr] for attr in path if path[attr]})

# 		paths.update({attr: paths.get(attr) if isinstance(paths.get(attr),str) else self.cwd for attr in data if paths.get(attr)})			

# 		# Dump data
# 		for attr in paths:
# 			root,file = split(paths[attr],directory=True,file_ext=True)
# 			file = file if file is not None else self.path
# 			path = join(file,root=root)
# 			dump(data[attr],path)
		
# 		return

# 	def load(self,path=None):
# 		'''
# 		Load class data		
# 		Args:
# 			path (str,dict[str,(str,bool)]): Path to load class data, either path or boolean to load
# 		'''

# 		# TODO: Determine dump/load model (.pkl?)

# 		# Set data
# 		data = {}

# 		# Set path
# 		paths = {}
# 		if path is None:
# 			paths.update({attr: True for attr in data})			
# 		elif not isinstance(path,dict):
# 			paths.update({attr: path for attr in data if path})
# 		else:
# 			paths.update({attr: path[attr] for attr in path if path[attr]})

# 		paths.update({attr: paths.get(attr) if isinstance(paths.get(attr),str) else self.cwd for attr in data if paths.get(attr)})			

# 		# Load data
# 		for attr in paths:
# 			root,file = split(paths[attr],directory=True,file_ext=True)
# 			file = file if file is not None else self.path
# 			path = join(file,root=root)
# 			func = (list,)
# 			default = data[attr]
# 			data[attr] = load(path,default=default)
# 			setter(default,data[attr],func=func)

# 		return



# class Hamiltonian(Operators):
# 	'''
# 	Hamiltonian class of Operators
# 	Args:
# 		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
# 			operator (iterable[str]): string names of operators
# 			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 			string (iterable[str]): string labels of operators
# 			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		operator (iterable[str]): string names of operators
# 		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 		string (iterable[str]): string labels of operators
# 		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		N (int): Number of qudits
# 		D (int): Dimension of qudits
# 		d (int): Spatial dimension
# 		L (int,float): Scale in system
# 		delta (float): Simulation length scale		
# 		M (int): Number of time steps
# 		T (int): Simulation time
# 		tau (float): Simulation time scale
# 		P (int): Trotter order		
# 		space (str,Space): Type of local space
# 		time (str,Time): Type of Time evolution space						
# 		lattice (str,Lattice): Type of lattice		
# 		parameters (str,dict,Parameters): Type of parameters	
# 		state (str,dict,State): Type of state	
# 		noise (str,dict,Noise): Type of noise
# 		label (str,dict,Gate): Type of label
# 		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
# 		kwargs (dict): Additional system keyword arguments	
# 	'''

# 	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
# 				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
# 				space=None,time=None,lattice=None,parameters=None,state=None,noise=None,label=None,system=None,**kwargs):

# 		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,
# 				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
# 				space=space,time=time,lattice=lattice,parameters=parameters,state=state,noise=noise,label=label,system=system,**kwargs)
		
# 		return


# 	#@partial(jit,static_argnums=(0,))
# 	def __call__(self,parameters):
# 		'''
# 		Return parameterized operator sum(parameters*data)
# 		Args:
# 			parameters (array): Parameters to parameterize operator			
# 		Returns
# 			operator (array): Parameterized operator
# 		'''		

# 		parameters = self.parameters(parameters)

# 		data = sum(data(self.parameters) for data in self.data)

# 		return data

# 	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
# 		'''
# 		Setup class
# 		Args:
# 			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
# 				operator (iterable[str]): string names of operators
# 				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 				string (iterable[str]): string labels of operators
# 				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 			operator (iterable[str]): string names of operators
# 			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 			string (iterable[str]): string labels of operators
# 			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		'''
# 		# Get operator,site,string,interaction from data
# 		objs = {'operator':operator,'site':site,'string':string,'interaction':interaction}

# 		for obj in objs:
# 			objs[obj] = [] if objs[obj] is None else objs[obj]

# 		if data is None:
# 			data = {}
# 		elif all(isinstance(datum,Operator) for datum in data):
# 			data = {datum.timestamp: datum for datum in data}
		
# 		assert isinstance(data,dict), 'Incorrect data format %r'%(type(data))			

# 		if all(isinstance(data[name],dict) and (obj in data[name]) for name in data for obj in objs):
# 			for obj in objs:
# 				objs[obj].extend([data[name][obj] for name in data])

# 		# Lattice sites
# 		sites = {site: self.lattice(site) for site in ['i','i<j','<ij>','i...j']}	# sites types on lattice
# 		indices = {'i': ['i'],'<ij>':['i','j'],'i<j':['i','j'],'i...j':['i','j']}   # allowed symbolic indices and maximum locality of many-body site interactions

# 		# Get number of operators
# 		size = min(len(objs[obj]) for obj in objs)
		
# 		# Get all indices from symbolic indices
# 		for index in range(size):
			
# 			size -= 1
			
# 			_objs = {}
# 			for obj in objs:
# 				value = deepcopy(objs[obj].pop(0))
# 				if obj in ['site']:
# 					if isinstance(value,str):
# 						value = indices[value]
# 				_objs[obj] = value

# 			if any(i in indices[_objs['interaction']] for i in _objs['site']):
# 				for i,s in enumerate(sites[_objs['interaction']]):
# 					for obj in objs:
# 						if obj in ['site']:
# 							value = [dict(zip(
# 								indices[_objs['interaction']],
# 								s if not isinstance(s,int) else [s])
# 							).get(i,parse(i,int)) 
# 							for i in _objs['site']]
# 						else:
# 							value = _objs[obj]

# 						objs[obj].append(value)			

# 			elif len(_objs['operator']) == len(_objs['site']):
# 				for obj in objs:
# 					value = _objs[obj]
# 					objs[obj].append(value)					

# 			else:
# 				for obj in objs:
# 					value = _objs[obj]
# 					objs[obj].append(value)	


# 		# Set class attributes
# 		self.__extend__(**objs)

# 		return



# class Unitary(Hamiltonian):
# 	'''
# 	Unitary class of Operators
# 	Args:
# 		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
# 			operator (iterable[str]): string names of operators
# 			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 			string (iterable[str]): string labels of operators
# 			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		operator (iterable[str]): string names of operators
# 		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
# 		string (iterable[str]): string labels of operators
# 		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
# 		N (int): Number of qudits
# 		D (int): Dimension of qudits
# 		d (int): Spatial dimension
# 		L (int,float): Scale in system
# 		delta (float): Simulation length scale				
# 		M (int): Number of time steps
# 		T (int): Simulation Time
# 		tau (float): Simulation time scale		
# 		P (int): Trotter order		
# 		space (str,Space): Type of local space
# 		time (str,Time): Type of Time evolution space
# 		lattice (str,Lattice): Type of lattice
# 		parameters (str,dict,Parameters): Type of parameters	
# 		state (str,dict,State): Type of state	
# 		noise (str,dict,Noise): Type of noise
# 		label (str,dict,Gate): Type of label
# 		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
# 		kwargs (dict): Additional system keyword arguments	
# 	'''

# 	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
# 				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
# 				space=None,time=None,lattice=None,parameters=None,state=None,noise=None,label=None,system=None,**kwargs):
		
# 		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,
# 				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
# 				space=space,time=time,lattice=lattice,parameters=parameters,state=state,noise=noise,label=label,system=system,**kwargs)

# 		return

# 	#@partial(jit,static_argnums=(0,))
# 	def __call__(self,parameters):
# 		'''
# 		Return parameterized operator expm(parameters*data)
# 		Args:
# 			parameters (array): Parameters to parameterize operator
# 		Returns
# 			operator (array): Parameterized operator
# 		'''		

# 		parameters = self.parameters(parameters)

# 		data = self.identity()
		
# 		for i in range(self.M):
# 			operators = trotter([data(self.parameters,i) for data in self.data],self.P)
# 			for operator in operators:
# 				data = dot(operator,data)
		
# 		return data

# 		# return self.exponentiation(self.coefficients*parameters)

# 	#@partial(jit,static_argnums=(0,))
# 	def __grad_analytical__(self,parameters):
# 		'''
# 		Class gradient
# 		Args:
# 			parameters (array): parameters		
# 		Returns
# 			out (array): Return of function
# 		'''	

# 		# Get class attributes
# 		parameters = self.parameters(parameters)
# 		shape = indexer(self.parameters.index,self.parameters).shape

# 		attributes = None

# 		# Get data
# 		data = self.data
# 		dtype = self.dtype

# 		# Get trotterized shape
# 		P = self.P
# 		shape = list(shape[::-1])

# 		shape[-1] *= P

# 		ndim = len(shape)

# 		attribute = 'slice'
# 		layer = 'variables'
# 		slices = attributes[attribute][layer]		

# 		attribute = 'index'
# 		layer = 'variables'
# 		indices = attributes[attribute][layer]		

# 		slices = tuple([
# 			*[[slices[parameter][group][axis] for parameter in slices for group in slices[parameter]][0]
# 				for axis in range(0,1)],
# 			*[[slices[parameter][group][axis] for parameter in slices for group in slices[parameter]][0]
# 				for axis in range(1,ndim)]
# 		])

# 		indices = tuple([
# 			*[[i for parameter in indices for group in indices[parameter] for i in indices[parameter][group][axis]]
# 				for axis in range(0,1)],
# 			*[[indices[parameter][group][axis] for parameter in indices for group in indices[parameter]][0]
# 				for axis in range(1,ndim)]
# 		])

# 		# Calculate parameters and gradient
# 		parameters = self.parameters(parameters)

# 		coefficients = self.coefficients
# 		data = array(trotter([datum() for datum in data],P))
# 		identity = self.identity()

# 		grad = gradient_expm(coefficients*parameters,data,identity)
# 		grad *= coefficients

# 		# Reshape gradient
# 		axis = 1

# 		grad = grad.reshape((*shape,*grad.shape[axis:]))

# 		grad = grad.transpose(axis,0,*[i for i in range(grad.ndim) if i not in [0,axis]])

# 		grad = array(gradient_trotter(grad,P))

# 		grad = grad[indices]

# 		grad = grad.reshape((-1,*grad.shape[axis+1:]))

# 		return grad

# 	#@partial(jit,static_argnums=(0,))
# 	def grad_analytical(self,parameters):
# 		'''
# 		Class gradient
# 		Args:
# 			parameters (array): parameters		
# 		Returns
# 			out (array): Return of function
# 		'''	
# 		return self.__grad_analytical__(parameters)


# class Callback(object):
# 	def __init__(self,*args,**kwargs):
# 		'''	
# 		Class for callback
# 		Args:
# 			args (tuple): Class arguments
# 			kwargs (dict): Class keyword arguments
# 		'''

# 		self.defaults = {
# 			'iteration':[],
# 			'parameters':[],'grad':[],'search':[],
# 			'value':[],'objective':[],
# 			'alpha':[],'beta':[],

# 			'iteration.max':[],'iteration.min':[],
# 			'variables':[],'variables.relative':[],'variables.relative.mean':[],
# 			'features':[],'features.relative':[],'features.relative.mean':[],
# 			'objective.ideal.noise':[],'objective.diff.noise':[],'objective.rel.noise':[],
# 			'objective.ideal.state':[],'objective.diff.state':[],'objective.rel.state':[],
# 			'objective.ideal.operator':[],'objective.diff.operator':[],'objective.rel.operator':[],
# 			'hessian':[],'fisher':[],
# 			'hessian.eigenvalues':[],'fisher.eigenvalues':[],
# 			'hessian.rank':[],'fisher.rank':[],

# 			'N':[],'D':[],'d':[],'L':[],'delta':[],'M':[],'T':[],'tau':[],'P':[],
# 			'space':[],'time':[],'lattice':[],'architecture':[],'timestamp':[],

# 			'noise.scale':[],'optimize.c1':[],'optimize.c2':[],

# 		}

# 		return

# 	def __call__(self,parameters,track,optimizer,model,metric,func,grad):
# 		''' 
# 		Callback
# 		Args:
# 			parameters (array): parameters
# 			track (dict): callback tracking
# 			optimizer (Optimizer): callback optimizer
# 			model (object): Model instance
# 			metric (str,callable): Callback metric
# 			func (callable): Objective function with signature func(parameters)
# 			grad (callable): Objective gradient with signature grad(parameters)
# 		Returns:
# 			status (int): status of callback
# 		'''
# 		attributes = optimizer.attributes
# 		iterations = optimizer.iterations
# 		hyperparameters = optimizer.hyperparameters

# 		init = (len(attributes['iteration'])==1) and ((attributes['iteration'][-1]==0) or (attributes['iteration'][-1] != (iterations.stop)))
		
# 		done = (len(attributes['iteration'])>1) and (attributes['iteration'][-1] == (iterations.stop))
		
# 		status = (
# 			((len(attributes['value']) >= 1) and 
# 			 (attributes['iteration'][-1] <= max(1,
# 				hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) or
# 			(
# 			(abs(attributes['value'][-1]) > 
# 				(hyperparameters['eps']['value']*hyperparameters['value']['value'])) and
# 			(log10(abs(attributes['value'][-1] - attributes['value'][-2])) > 
# 				(log10(abs(hyperparameters['eps']['value.difference'])))) and
# 			(norm(attributes['grad'][-1])/attributes['grad'][-1].size > 
# 				  (hyperparameters['eps']['grad']*hyperparameters['value']['grad'])) and
# 			(norm(attributes['grad'][-1] - attributes['grad'][-2])/attributes['grad'][-2].size > 
# 				  (hyperparameters['eps']['grad.difference']*norm(attributes['grad'][-2])/attributes['grad'][-2].size))
# 			)
# 			)


# 		other = ((len(attributes['iteration']) == 1) or 
# 			(hyperparameters['modulo']['track'] is None) or 
# 			(attributes['iteration'][-1]%hyperparameters['modulo']['track'] == 0))

# 		stop = (
# 			(hyperparameters['eps'].get('value.increase') is not None) and
# 			((len(attributes['value']) > 1) and 
# 			 (attributes['iteration'][-1] >= max(1,
# 				hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) and
# 			((attributes['value'][-1] > attributes['value'][-2]) and
# 			(log10(attributes['value'][-1] - attributes['value'][-2]) > 
# 			(log10(hyperparameters['eps']['value.increase']*attributes['value'][-1]))))
# 			)


# 		status = (status) and (not stop)

# 		updates = {
# 			**{attr: lambda i,attr,track,default: (track[attr][-1]) for attr in ['iteration.max','iteration.min']},
# 			**{attr: lambda i,attr,track,default: (empty(track[attr][-1].shape) if ((i>0) and i<(len(track[attr])-1)) else track[attr][i])
# 				for attr in [
# 					'parameters','grad','search',
# 					'variables','features',
# 					'variables.relative','features.relative',
# 					'hessian','fisher',
# 					'hessian.eigenvalues','fisher.eigenvalues']},
# 			**{attr: None for attr in [
# 				'parameters.norm','grad.norm','search.norm',
# 				'variables.norm','features.norm'
# 				]},
# 			**{attr: lambda i,attr,track,default: (default if i<(len(track[attr])-1) else track[attr][i])
# 				for attr in [
# 				'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
# 				'objective.ideal.state','objective.diff.state','objective.rel.state',
# 				'objective.ideal.operator','objective.diff.operator','objective.rel.operator',
# 				'hessian.rank','fisher.rank']
# 			},
# 			}

# 		attrs = relsort(track,attributes)
# 		size = min(len(track[attr]) for attr in track)
# 		does = {**{attr: False for attr in attrs},**hyperparameters.get('do',{})}

# 		if ((status) or done or init or other):
			
# 			for attr in attrs:

# 				if ((hyperparameters['length']['track'] is not None) and 
# 					(len(track[attr]) > hyperparameters['length']['track'])
# 					):
# 					_value = track[attr].pop(0)
				

# 				index = -1 if (not stop) else -2
# 				parameters = attributes['parameters'][index]
			
# 				if attr in [
# 					'parameters','grad','search',
# 					'variables','features',
# 					'variables.relative','features.relative',
# 					'hessian','fisher',
# 					'hessian.eigenvalues','fisher.eigenvalues']:
# 					default = empty(track[attr][-1].shape) if (len(track[attr])>0) else nan
# 				else:
# 					default = nan

# 				do = (not ((status) and (not done) and (not init))) or does[attr]

# 				value = default

# 				if attr in attributes:
# 					value = attributes[attr][index]

# 				if (not stop):
# 					track[attr].append(value)

# 				if attr in ['iteration.max']:
# 					value = int(track['iteration'][-1])

# 				elif attr in ['iteration.min']:
# 					value = int(track['iteration'][argmin(abs(array(track['objective'])))])

# 				elif attr in ['value']:
# 					value = abs(attributes[attr][index])
				
# 				elif attr in ['parameters','grad','search'] and (not do):
# 					value = default

# 				elif attr in ['parameters','grad','search'] and (do):
# 					value = attributes[attr][index]

# 				elif attr in ['parameters.norm','grad.norm','search.norm']:
# 					value = attr.split(delim)[0]
# 					value = attributes[value][index]
# 					value = norm(value)/(value.size)

# 				elif attr in [
# 					'variables.norm','variables.relative','variables.relative.mean',
# 					'features.norm','features.relative','features.relative.mean'] and (not do):
# 					value = default

# 				elif attr in [
# 					'variables','variables.norm','variables.relative','variables.relative.mean',
# 					'features','features.norm','features.relative','features.relative.mean'] and (do):

# 					if attr in ['variables','features']:
# 						value = model.parameters(parameters)
# 					elif attr in ['variables.norm','features.norm']:
# 						value = model.parameters(parameters)
# 						value = norm(value)/(value.size)
# 					elif attr in ['variables.relative','features.relative']:
# 						eps = 1e-20
# 						value = model.parameters(parameters)
# 						_value = model.parameters(attributes['parameters'][0])
# 						value = abs((value - _value + eps)/(_value + eps))
# 					elif attr in ['variables.relative.mean','features.relative.mean']:
# 						eps = 1e-20
# 						value = model.parameters(parameters)
# 						_value = model.parameters(attributes['parameters'][0])
# 						value = abs((value - _value + eps)/(_value + eps)).mean()

# 				elif attr in ['objective']:
# 					value = abs(metric(model(parameters)))
				
# 				elif attr in [
# 					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
# 					'objective.ideal.state','objective.diff.state','objective.rel.state',
# 					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and ((status) and (not done)):
# 					value = default


# 				elif attr in [
# 					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
# 					'objective.ideal.state','objective.diff.state','objective.rel.state',
# 					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and (not ((status) and (not done))):

# 					_kwargs = {kwarg: {prop: hyperparameters.get('kwargs',{}).get(kwarg,{}).get(prop) if kwarg in ['noise'] else None for prop in ['scale']} for kwarg in ['state','noise','label']}
# 					_kwargs = {kwarg: {prop: getattrs(model,[kwarg,prop],delimiter=delim,default=_kwargs[kwarg][prop]) for prop in _kwargs[kwarg]} for kwarg in ['state','noise','label']}
# 					if attr in ['objective.ideal.noise','objective.diff.noise','objective.rel.noise']:
# 						_kwargs = {kwarg: False if kwarg in [] else _kwargs[kwarg] for kwarg in _kwargs}
# 						_metric = 'real'
# 					elif attr in ['objective.ideal.state','objective.diff.state','objective.rel.state']:						
# 						_kwargs = {kwarg: False if kwarg in ['noise'] else _kwargs[kwarg] for kwarg in _kwargs}
# 						_metric = 'real'
# 					elif attr in ['objective.ideal.operator','objective.diff.operator','objective.rel.operator']:
# 						_kwargs = {kwarg: False if kwarg in ['noise','state'] else _kwargs[kwarg] for kwarg in _kwargs}
# 						_metric = 'abs2'

# 					_model = model
# 					_shapes = model.shapes
# 					_label = model.label()
# 					_optimize = None
# 					_hyperparameters = hyperparameters
# 					_system = model.system
# 					_restore = {kwarg: deepcopy(getattr(model,kwarg)) for kwarg in _kwargs}

# 					_model.__initialize__(**_kwargs)
# 					_metric = Metric(_metric,shapes=_shapes,label=_label,optimize=_optimize,hyperparameters=_hyperparameters,system=_system,verbose=False)

# 					if attr in ['objective.ideal.noise','objective.ideal.state','objective.ideal.operator']:
# 						value = abs(_metric(_model(parameters)))
# 					elif attr in ['objective.diff.noise','objective.diff.state','objective.diff.operator']:
# 						value = abs((track['objective'][-1] - _metric(_model(parameters))))
# 					elif attr in ['objective.rel.noise','objective.rel.state','objective.rel.operator']:
# 						value = abs((track['objective'][-1] - _metric(_model(parameters)))/(track['objective'][-1]))

# 					model.__initialize__(**_restore)


# 				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (not do):
# 					value = default

# 				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (do):
					
# 					if attr in ['hessian','hessian.eigenvalues','hessian.rank']:
# 						function = hessian(jit(lambda parameters: metric(model(parameters))))
# 					elif attr in ['fisher','fisher.eigenvalues','fisher.rank']:
# 						function = fisher(model,model.grad,shapes=(model.shape,(parameters.size,*model.shape)))

# 					if attr in ['hessian','fisher']:
# 						value = function(parameters)

# 					elif attr in ['hessian.eigenvalues','fisher.eigenvalues']:
# 						value = sort(abs(eig(function(parameters),compute_v=False,hermitian=True)))[::-1]
# 						value = value/maximum(value)
# 					elif attr in ['hessian.rank','fisher.rank']:
# 						value = sort(abs(eig(function(parameters),compute_v=False,hermitian=True)))[::-1]
# 						value = argmax(abs(difference(value)/value[:-1]))+1	
# 						value = value.size if (value==value.size-1) else value

# 				elif attr in ['tau.noise.parameters','T.noise.parameters']:
# 					value = [attr.split(delim)[0],delim.join(attr.split(delim)[1:])]
# 					value = [getattrs(model,i,default=default,delimiter=delim) for i in value]
# 					value = value[0]/value[1] if value[1] else value[0]

# 				elif attr not in attributes and not (getter(hyperparameters,attr.replace('optimize%s'%(delim),''),default=null,delimiter=delim) is null):
# 					value = getter(hyperparameters,attr.replace('optimize%s'%(delim),''),default=default,delimiter=delim)

# 				elif attr not in attributes and hasattrs(model,attr,delimiter=delim):
# 					value = getattrs(model,attr,default=default,delimiter=delim)

# 				track[attr][-1] = value

# 				if (not does[attr]) and (updates.get(attr) is not None):
# 					for i in range(len(track[attr])):
# 						track[attr][i] = updates[attr](i,attr,track,default)


# 		logging = ((len(attributes['iteration']) == 1) or 
# 			(hyperparameters['modulo']['log'] is None) or 
# 			(attributes['iteration'][-1]%hyperparameters['modulo']['log'] == 0)
# 			)

# 		if logging:

# 			msg = '\n'.join([
# 				'%d f(x) = %0.4e'%(
# 					attributes['iteration'][-1],
# 					track['objective'][-1],
# 				),
# 				'|x| = %0.4e\t\t|grad(x)| = %0.4e'%(
# 					norm(attributes['parameters'][-1])/
# 						 (attributes['parameters'][-1].size),
# 					norm(attributes['grad'][-1])/
# 						 (attributes['grad'][-1].size),
# 				),
# 				'\t\t'.join([
# 					'%s = %0.4e'%(attr,attributes[attr][-1])
# 					for attr in ['alpha','beta']
# 					if attr in attributes and len(attributes[attr])>0
# 					]),
# 				# 'x\n%s'%(to_string(parameters.round(4))),
# 				# 'theta\n%s'%(to_string(model.parameters(parameters).round(4))),
# 				# 'U\n%s\nV\n%s'%(
# 				# 	to_string((model(parameters)).round(4)),
# 				# 	to_string((model.label()).round(4))),
# 				])


# 			model.log(msg)


# 		return status



