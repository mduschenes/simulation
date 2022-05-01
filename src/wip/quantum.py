#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial
import logging,logging.config
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import pandas as pd
import sparse as sparray
import jax
import jax.numpy as jnp
import jax.scipy as jsp
jax.config.update('jax_platform_name','cpu')
jax.config.update("jax_enable_x64", True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'

import qiskit as qs	

# Logging
import logging
logger = logging.getLogger(__name__)


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['',"..","../..","../../lib"]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from lib.utils.utils import System,objs,array,asarray,ones,zeros,arange,eye,identity,hadamard,phasehadamard,cnot,toffoli
from lib.utils.utils import jit,vmap,vfunc,contract,summation,product,exponentiate
from lib.utils.utils import parse,mod,concatenate,PRNG,isdiag,isclose,allclose,unique,flatten,diag
from lib.utils.utils import expm,exp,expmeuler,expeuler,log
from lib.utils.io import load,dump,basename


class Space(object):
	'''
	Hilbert space class for Operators with size n
	Args:
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		space (str,Space): Type of Hilbert space
		system (dict,System): System attributes
	'''
	def __init__(self,N,D,d,L,space,system):

		self.system = System(system)
		self.N = N if N is not None else 1
		self.D = D if D is not None else 2
		self.d = d if d is not None else 1
		self.L = L if L is not None else 1
		self.space = space		
		self.default = 'spin'

		self.__setup__()
		
		return

	def __setup__(self):
		'''
		Setup space attributes space,string,n
		'''
		if isinstance(self.space,Space):
			self.space = self.space.space
		if self.space is None:
			self.space = self.default
		self.__string__()
		self.__size__()
		return

	def __string__(self):
		self.string = self.space
		return

	def __size__(self):
		self.n = self.get_n()
		return 

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return str(self.string)

	def get_N(self,n):
		if self.space in ['spin']:
			try:
				return int(log(n)/log(self.D))
			except:
				return 1
		else:
			try:
				return int(log(n)/log(self.D))
			except:
				return 1			
		return 		

	def get_D(self,n):
		if self.space in ['spin']:
			return int(n**(1/self.N))
		else:
			return int(n**(1/self.N))
		return

	def get_n(self):
		if self.space in ['spin']:
			return self.D**self.N
		else:
			return self.D**self.N
		return		


class Time(object):
	'''
	Time evolution class for Operators with size n
	Args:
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		time (str,Time): Type of Time evolution space
		system (dict,System): System attributes
	'''
	def __init__(self,M,T,p,time,system):

		self.system = System(system)
		self.M = M if M is not None else 1
		self.T = T if T is not None else 1
		self.p = p if p is not None else 1
		self.time = time
		self.default = 'linear'

		self.__setup__()
		
		return

	def __setup__(self):
		'''
		Setup time evolution attributes tau
		'''
		if isinstance(self.time,Time):
			self.time = self.time.time
		if self.time is None:
			self.time = self.default
		self.__string__()
		self.__size__()
		return

	def __string__(self):
		self.string = self.time
		return
	def __size__(self):
		self.tau = self.get_tau()
		return 

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return str(self.string)

	def get_T(self,tau):
		if self.time in ['linear']:
			return tau*self.M
		else:
			return tau*self.M
		return 		

	def get_M(self,tau):
		if self.time in ['linear']:
			return self.T/tau
		else:
			return self.T/tau
		return

	def get_tau(self):
		if self.time in ['linear']:
			return self.T/self.M
		else:
			return self.T/self.M
		return	

class Lattice(object):
	'''
	Define a hyper lattice class
	Args:
		N (int): Lattice length along axis
		d (int): Dimension of lattice
		lattice (str,Lattice): Type of lattice, allowed strings in ['square','square-nearest']
		system (dict,System): System attributes (dtype,format,device,seed,verbose)		
	'''	
	def __init__(self,N=4,d=2,lattice='square',system=None):
		

		# Define lattice
		if isinstance(lattice,Lattice):
			lattice = lattice.lattice
		else:
			lattice = lattice

		# Define parameters of system        
		self.lattice = lattice
		self.N = N
		self.d = d

		# Define system
		self.system = System(system)
		self.dtype = self.system.dtype
		self.verbose = self.system.verbose

		self.dtype = self.dtype if self.dtype in ['int','Int32','Int64'] else int

		# Define linear size n and coordination number z	
		if self.lattice is None:
			N = 0
			d = 0
			n = 0
			z = 0
		elif self.lattice in ['square','square-nearest']:
			n = int(N**(1/d))
			z = 2*d
			assert n**d == N, "N != n^d for N=%d, d=%d, n=%d"%(N,d,n)
		else:
			n = int(N**(1/d))
			z = 2*d
			assert n**d == N, "N != n^d for N=%d, d=%d, n=%d"%(N,d,n)

		self.n = n
		self.z = z


		# Define attributes
		self.__size__()
		self.__shape__()
		self.__string__()

		# Define array of vertices
		self.vertices = np.arange(self.N)
		
		# n^i for i = 1:d array
		self.n_i = self.n**np.arange(self.d,dtype=self.dtype)
		
		# Arrays for finding coordinate and linear position in d dimensions
		self.I = np.eye(self.d)
		self.R = np.arange(1,max(2,np.ceil(self.n/2)),dtype=self.dtype)

		return


	def set(self,attr,value):
		'''	
		Set class attribute
		'''
		setattr(self,attr,value)
		return
		
	def get(self,attr,default=None):
		'''
		Get class attribute
		'''
		return getattr(self,attr,default)

	def log(self,msg):
		'''
		Log messages
		Args:
			msg (str): Message to log
		'''
		logger.log(self.verbose,msg)
		return

	def __call__(self,site=None):
		'''
		Get list of lists of sites of lattice
		Args:
			site (str,int): Type of sites, either int for unique site-length list of vertices, or string in allowed ["i","i,j","i<j"]
		Returns:
			sites (list): List of site-length lists of lattice
		'''

		# Unique site-length lists if site is int
		if isinstance(site,(int,np.integer)):
			k = site
			conditions = None
			sites = self.iterable(k,conditions)
		elif isinstance(site,(str)):
			if site in ["i"]:
				sites = [[i] for i in self.vertices]
			elif site in ["i,j"]:
				sites = [[i,j] for i in self.vertices for j in self.vertices]
			elif site in ["i<j"]:
				k = 2
				conditions = lambda i,k: all([i[j]<i[j+1] for j in range(k-1)])	
				sites = self.iterable(k,conditions)
			elif site in ["<ij>"]:
				if self.z > self.N:
					sites = []
				else:
					sites = [list(i) for i in unique(
						np.sort(
							np.vstack([
								np.repeat(range(self.N),self.z),
								self.nearestneighbours(r=1)[0].ravel()
							]),
						axis=0),
						axis=1).T]

			elif site in ['i...j']:
				sites = [range(self.N) for i in range(self.N)]
		else:
			k = 2
			conditions = None
			sites = self.iterable(k,conditions)
		return sites


	def __string__(self):
		self.string = self.lattice if self.lattice is not None else 'null'
		return
		
	def __size__(self):
		self.size = self.N
		return 

	def __shape__(self):
		self.shape = (self.N,self.z)
		return

	def __str__(self):
		return self.string

	def __repr__(self):
		return self.string
		
	def position(self,site):
		'''
		Return position coordinates in d-dimensional n^d lattice 
		from given linear site position in 1d N^d length array
		i.e) [int(site/(self.n**(i))) % self.n for i in range(self.d)]
		Args:
			site (int,array): Linear site positions on lattice
		Returns:
			position (array): Position coordinates of linear site positions 
		'''
		isint = isinstance(site,(int,np.integer))

		if isint:
			site = np.array([site])
		position = np.mod(((site[:,None]/self.n_i)).
						astype(self.dtype),self.n)
		if isint:
			return position[0]
		else:
			return position
	
	def site(self,position):
		'''
		Return linear site position in 1d N^d length array 
		from given position coordinates in d-dimensional n^d lattice
		i.e) sum(position[i]*self.n**i for i in range(self.d))
		
		Args:
			position (array): Position coordinates of linear site positions 
		Returns:
			site (int,array): Linear site positions on lattice
		'''
		is1d = isinstance(position,(list,tuple)) or position.ndim < 2

		if is1d:
			position = np.array([position])
		
		site = position.dot(self.n_i).astype(self.dtype)

		if is1d:
			return site[0]
		else:
			return site


	def nearestneighbours(self,r=None,vertices=None):
		'''
		Return array of neighbouring spin vertices 
		for a given site and r-distance bonds
		i.e) [self.site(put(self.position(site),i,
						lambda x: mod(x + s*r,self.n))) 
						for i in range(self.d)for s in [1,-1]])
		Args:
			r (int,list): Radius of number of nearest neighbours away to compute nearest neighbours on lattice of shape (l,)
			vertices (array): Vertices to compute nearest neighbours on lattice of shape (N,)
		Returns:
			nearestneighbours (array): Array of shape (l,N,z) of nearest neighbours a manhattan distance r away
		'''
		if vertices is None:
			vertices = self.vertices
		
		sitepos = self.position(vertices)[:,None]
		
		if r is None:
			Rrange = self.R
		elif isinstance(r,list):
			Rrange = r
		else:
			Rrange = [r]
		return np.array([concatenate(
							(self.site(np.mod(sitepos+R*self.I,self.n)),
							 self.site(np.mod(sitepos-R*self.I,self.n))),1)
								for R in Rrange],dtype=self.dtype)                     


	def iterable(self,k,conditions=None):
		'''
		Return iterable of k-tuples of combinations of vertices
		Conditions limit generator to certain combinations of vertices
		Args:
			k (int): Number of vertices in lists of combinations of vertices
			conditions (callable): Conditions on allowed combinations of vertices k-lists i with signature conditons(i,k)
		Returns:
			iterable (list): list of k-lists of allowed combinations of vertices
		'''

		default = lambda i,k: any([i[j] != i[l] for j in range(k) for l in range(k) if j!=l])
		conditions = default if conditions is None else conditions
		iterable =  [list(i) for i in itertools.product(self.vertices,repeat=k) if conditions(i,k)]
		return iterable


class Object(object):
	'''
	Class for object
	Args:
		obj (iterable[str,array,dict,iterable,Operator,Unitary,Gate],str,array,dict,iterable,Operator,Unitary,Gate): Path to load array, array or, (iterable) of dictionaries of fields for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (str,iterable[str],,Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator		
		site (iterable[iterable[int]],iterable[int]): site of local gate
		string (str): string label for gate
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		coefficient (int,float,complex,str): scalar coefficient for gate									
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,obj,site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):

		self.N = N
		self.D = D
		self.d = d
		self.L = L
		self.M = M
		self.T = T
		self.p = p
		self.space = space
		self.time = time
		self.lattice = lattice
		self.system = system

		self.data = []
		self.size = getattr(self,'size',0)
		self.shape = (self.size,)
		self.locality = getattr(self,'locality',0)
		self.site = []
		self.string = []
		self.coefficient = getattr(self,'coefficient',1)
		self.interaction = []

		self.delimiter = getattr(self,'delimiter',' ')
		self.basis = getattr(self,'basis',None)
		self.diagonal = []
		self._data = []
		self.funcs = lambda parameters: None
		self.expms = lambda parameters: None
		self.transform = []
		self.transformH = []
		self.index = arange(self.size)

		self.parameters = getattr(self,'parameters',None)
		self.dim = getattr(self,'dim',0)

		self.__system__(obj)
		self.__space__(obj)
		self.__time__(obj)
		self.__lattice__(obj)
		self.__setup__(obj,site,string,coefficient,interaction)
	
		return	


	def __data__(self,data,site=None,string=None,coefficient=None,interaction=None):
		'''
		Set class data
		Args:
			data (iterable): Class data
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		'''
		size = len(data)
		for i in range(size):
			data[i] = data[i]
		return

	def __system__(self,obj,system=None):
		'''
		Set system attributes
		Args:
			obj (iterable[str,array,dict,iterable,Operator,Unitary,Gate],str,array,dict,iterable,Operator,Unitary,Gate): Path to load array, array or, (iterable) of dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],,Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator		
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''
		system = self.system if system is None else system
		
		self.system = System(system)		
		self.dtype = self.system.dtype
		self.verbose = self.system.verbose

		return


	def __space__(self,obj,N=None,D=None,d=None,L=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			obj (iterable[str,array,dict,iterable,Operator,Unitary,Gate],str,array,dict,iterable,Operator,Unitary,Gate): Path to load array, array or, (iterable) of dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],,Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator		
			N (int): Number of qudits
			D (int): Dimension of qudits
			d (int): Spatial dimension
			L (int,float): Scale in system
			space (str,Space): Type of Hilbert space
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''
		N = self.N if N is None else N
		D = self.D if D is None else D
		d = self.d if d is None else d
		L = self.L if L is None else L
		space = self.space if space is None else space
		system = self.system if system is None else system

		try:
			n = max([o.n for o in obj])
			N = space.get_N(n)
		except:
			try:
				n = max([max(o.shape) for o in obj])
				N = space.get_N(n)
			except:
				pass

		self.space = Space(N,D,d,L,space,system)
		self.N = self.space.N
		self.D = self.space.D
		self.d = self.space.d
		self.L = self.space.L
		self.n = self.space.n
		self.I = identity(self.n)#Identity(N=self.N,D=self.D,d=self.d,L=self.L,space=self.space,system=self.system)()

		return


	def __time__(self,obj,M=None,T=None,p=None,time=None,system=None):
		'''
		Set time attributes
		Args:
			obj (iterable[str,array,dict,iterable,Operator,Unitary,Gate],str,array,dict,iterable,Operator,Unitary,Gate): Path to load array, array or, (iterable) of dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],,Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator		
			M (int): Number of time steps
			T (int): Simulation Time
			p (int): Trotter order		
			time (str,Time): Type of Time evolution space						
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''
		M = self.M if M is None else M
		T = self.T if T is None else T
		p = self.p if p is None else p
		time = self.time if time is None else time
		system = self.system if system is None else system

		self.time = Time(M,T,p,time,system)		
		self.M = self.time.M
		self.T = self.time.T
		self.p = self.time.p
		self.tau = self.time.tau

		return


	def __lattice__(self,obj,N=None,D=None,d=None,L=None,lattice=None,system=None):
		'''
		Set space attributes
		Args:
			obj (iterable[str,array,dict,iterable,Operator,Unitary,Gate],str,array,dict,iterable,Operator,Unitary,Gate): Path to load array, array or, (iterable) of dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],,Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator		
			N (int): Number of qudits
			D (int): Dimension of qudits
			d (int): Spatial dimension
			L (int,float): Scale in system
			lattice (str,Lattice): Type of lattice		
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''		
		N = self.N if N is None else N
		D = self.D if D is None else D
		d = self.d if d is None else d
		L = self.L if L is None else L
		lattice = self.lattice if lattice is None else lattice
		system = self.system if system is None else system

		self.lattice = Lattice(N,d,lattice,system)	

		return

	def __parameters__(self,parameters=None):
		''' 
		Setup parameters
		Args:
			parameters (array): parameters
		'''
		if parameters is None:
			parameters = ones(self.shape)
		# else:
		# 	parameters = asarray(parameters)
		# if parameters.size == 1:
		# 	parameters = parameters*ones(self.shape[:1])
		# else:
		# 	parameters = asarray(parameters)[:self.size].reshape(self.shape[:1])
		self.parameters = parameters
		return


	def __str__(self):
		size = self.size//self.M if self.size>=self.M else self.size
		multiple_time = (self.M>1) and (self.size%self.M == 0) and (self.size >= self.M)
		multiple_space = [size>1 and isinstance(self.data[i],(Operator,Gate)) and self.data[i].size>1 for i in range(size)]
		return '%s%s%s%s'%(
				'{' if multiple_time else '',
				self.delimiter.join(['%s%s%s'%(
					'(' if multiple_space[i] else '',
					self.string[i],
					')' if multiple_space[i] else '',
					) for i in range(size)]),
				'}' if multiple_time else '',
				'^%d'%(self.M) if multiple_time else '')

	def __repr__(self):
		return self.__str__()

	def __len__(self):
		return self.size

	def log(self,msg):
		'''
		Log messages
		Args:
			msg (str): Message to log
		'''
		logger.log(self.verbose,msg)
		return	

	def dump(self,path=None,parameters=None,ext='txt'):
		'''
		Save class data		
		'''
		if path is None:
			path = '%s.%s'%(self,ext)
		data = self(parameters)
		dump(data,path)
		return


	def load(self,path=None,parameters=None,ext='txt'):
		'''
		Load class data		
		'''
		if path is None:
			path = '%s.%s'%(self,ext)
		data = array(load(path,dtype=self.dtype))
		string = basename(path)
		self.append(data,string=string)
		return


class Gate(Object):
	'''
	Class for parameterized Gate operation
	Args:
		gate (iterable[str,array,dict,iterable,Operator,Unitary,Gate],str,array,dict,iterable,Operator,Unitary,Gate): Path to load array, array or, (iterable) of dictionaries of fields for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (str,iterable[str],,Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator		
		site (iterable[iterable[int]],iterable[int]): site of local gate
		string (str): string label for gate
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		coefficient (int,float,complex,str): scalar coefficient for gate									
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,gate,site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		self.delimiter = ' * '
		super().__init__(gate,site=site,string=string,coefficient=coefficient,interaction=interaction,
						 N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return		

	def __setup__(self,gate,site,string,coefficient,interaction):
		'''
		Setup class attributes data,size,shape,locality,site,string,coefficient,interaction
		Args:
			gate (iterable[str,array,dict,iterable,Operator,Unitary,Gate],str,array,dict,iterable,Operator,Unitary,Gate): Path to load array, array or, (iterable) of dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],,Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator			
			site (iterable[iterable[int]],iterable[int]): site of local gate
			string (str): string label for gate
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for gate							
		'''

		self.append(gate,site,string,coefficient,interaction)

		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Return parameterized gate
		Args:
			parameters (array): Parameters to parameterize gate
		Returns
			gate (array): Parameterized gate
		'''
		self.__parameters__(parameters)	
		return self.coefficient*contract(self.funcs(self.tau*self.parameters),self.site,self.diagonal,self.size,self.N,self.D)

	#@partial(jit,static_argnums=(0,))
	def expm(self,parameters=None):
		'''
		Return parameterized gate
		Args:
			parameters (array): Parameters to parameterize gate
		Returns
			gate (array): Parameterized gate
		'''
		return self(parameters)

	def append(self,gate,site=None,string=None,coefficient=None,interaction=None):
		'''
		Append data to class
		Args:
			gate (iterable[str,array,dict,iterable,Operator,Unitary,Gate],str,array,dict,iterable,Operator,Unitary,Gate): Path to load array, array or, (iterable) of dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],,Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator			
			site (iterable[iterable[int]],iterable[int]): site of local gate
			string (str): string label for gate
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for gate							
		'''
		self.insert(-1,gate,site,string,coefficient,interaction)
		return

	def insert(self,index,gate,site=None,string=None,coefficient=None,interaction=None):
		'''
		Insert data to class
		Args:
			gate (iterable[str,array,dict,iterable,Operator,Unitary,Gate],str,array,dict,iterable,Operator,Unitary,Gate): Path to load array, array or, (iterable) of dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],,Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator			
			site (iterable[iterable[int]],iterable[int]): site of local gate
			string (str): string label for gate
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for gate							
		'''
	
		if index in [-1] or index>self.size:
			index = self.size//self.M

		symbols = {'':1,'+':1,'-':-1,'1':1,'-1':-1,'i':1j,'+i':1j,'-i':-1j,None:1}

		kwargs = dict(N=self.N,D=self.D,d=self.d,L=self.L,M=1,T=self.T,p=self.p,space=self.space,time=self.time,system=self.system)

		if isinstance(gate,(str,array,dict,Operator,Unitary,Gate)) or (self.M>1 and isinstance(gate,(list,tuple,))):
			gate = [gate]
			site = [site] if site is not None else None
			interaction = [interaction] if interaction is not None else None

		data = []
		for g in gate:
			if isinstance(g,(Unitary,Gate)):
				data.append(g)
			elif isinstance(g,(Operator)):
				data.append(Unitary(g,**kwargs))
			elif isinstance(g,(str,array)):
				data.append(_Gate(g,**kwargs))
			elif isinstance(g,(dict,list,tuple)):
				data.append(Unitary(g,**kwargs))
		
		self.__data__(data,site=site,string=string,coefficient=coefficient,interaction=interaction)
		self.__space__(data)
		self.__lattice__(data)

		size = len(data)
		site = [flatten(d.site,cls=list,unique=True) for d in data] if site is None else [flatten(s,cls=list,unique=True) for s in site]
		string = ['%s%s'%(string if string is not None else '',str(d)) for d in data]
		coefficient = symbols.get(coefficient,coefficient)
		interaction = [flatten(d.interaction,cls=list,unique=True) for d in data] if interaction is None else [flatten(i,cls=list,unique=False) for i in interaction]
		locality = len(flatten([self.site,site],cls=list,unique=True))
		

		self.data = [d for i in range(self.M) for d in [*self.data[i*(self.size//self.M):i*(self.size//self.M)+index],*data,*self.data[i*(self.size//self.M)+index:(i+1)*(self.size//self.M)]]]
		self.site = [d for i in range(self.M) for d in [*self.site[i*(self.size//self.M):i*(self.size//self.M)+index],*site,*self.site[i*(self.size//self.M)+index:(i+1)*(self.size//self.M)]]]
		self.string = [*self.string[:index],*string,*self.string[index:]]
		self.coefficient = coefficient		
		self.interaction = [d for i in range(self.M) for d in [*self.interaction[i*(self.size//self.M):i*(self.size//self.M)+index],*interaction,*self.interaction[i*(self.size//self.M)+index:(i+1)*(self.size//self.M)]]]
		self.size = len(self.data)
		self.dim = 0
		self.shape = (self.size,*[max([d.shape[i] if len(d.shape)>i else 0 for d in self.data]) for i in range(max([len(d.shape) for d in self.data]))])		
		self.locality = locality
		self.diagonal = [all(d.diagonal) for d in self.data]

		funcs = [d for d in self.data]
		index = arange(self.size)

		self.index = index
		self.funcs = vfunc(funcs,index)

		return



class Circuit(Gate):
	'''
	Class for parameterized Circuit of Gates operation. 
	Args:
		circuit (iterable[str,array,dict,iterable,Operator,Unitary,Gate,Circuit],str,array,dict,iterable,Operator,Unitary,Gate,Circuit) Path to load array, array or, (iterable) of dictionaries of fields for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (str,iterable[str],,Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator					
		site (iterable[iterable[int]],iterable[int]): site of local circuit
		string (str): string label for circuit
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		coefficient (int,float,complex,str): scalar coefficient for circuit					
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)	
	'''

	def __init__(self,circuit,site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(circuit,site=site,string=string,coefficient=coefficient,interaction=interaction,
						 N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return	


class _Gate(Gate):
	'''
	User defined subclass of Gate
	Gate has attributes: gate,size,site,string
	Args:
		gate (iterable[str,array],str,array): Path to load array, array or, Gate for gate
		site (iterable[iterable[int]],iterable[int]): site of local gate
		string (str): string label for gate
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		coefficient (int,float,complex,str): scalar coefficient for gate									
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,gate,site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(gate,site=site,string=string,coefficient=coefficient,interaction=interaction,
						 N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	def __setup__(self,gate,site=None,string=None,coefficient=None,interaction=None):
		'''
		Setup class attributes data,size,shape,locality,site,string,coefficient,interaction
		Args:
			gate (iterable[str,array],str,array): Path to load array, array for gate
			site (iterable[iterable[int]],iterable[int]): site of local gate
			string (str): string label for gate
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for gate		
		'''	

		self.append(gate,site,string,coefficient,interaction)

		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Return parameterized gate
		Args:
			parameters (array): Parameters to parameterize gate
		Returns
			gate (array): Parameterized gate
		'''
		self.__parameters__(parameters)		
		return contract(self.coefficient*self.parameters*self.data,self.site,self.diagonal,self.size,self.N,self.D)


	def append(self,gate,site=None,string=None,coefficient=None,interaction=None):
		'''
		Append gate to class
		Args:
			gate (iterable[str,array],str,array): Path to load array, array for gate
			site (iterable[iterable[int]],iterable[int]): site of local gate
			string (str): string label for gate
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for gate				
		'''
		self.insert(-1,gate,site,string,coefficient,interaction)
		return


	def insert(self,index,gate,site=None,string=None,coefficient=None,interaction=None):
		'''
		Insert data to class
		Args:
			gate (iterable[str,array],str,array): Path to load array, array for gate
			site (iterable[iterable[int]],iterable[int]): site of local gate
			string (str): string label for gate
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for gate		
		'''
	
		symbols = {'':1,'+':1,'-':-1,'1':1,'-1':-1,'i':1j,'+i':1j,'-i':-1j,None:1}

		kwargs = dict(N=self.N,D=self.D,d=self.d,L=self.L,M=1,T=self.T,p=self.p,space=self.space,time=self.time,system=self.system)

		if isinstance(gate,(str,array)):
			gate = [gate]
			site = [site] if site is not None else None
			interaction = [interaction] if interaction is not None else None

		data = []
		for g in gate:
			if isinstance(g,(str)):
				data.append(array(load(g,default=Identity(**kwargs)(),dtype=self.dtype)))
			elif isinstance(g,(array)):
				data.append(g)

		self.__data__(data,site=site,string=string,coefficient=coefficient,interaction=interaction)		
		self.__space__(data)
		self.__lattice__(data)

		size = len(data)
		site = [list(range(self.N)) for d in data] if site is None else [flatten(s,cls=list,unique=True) for s in site]
		string = ['%s%s'%(string if string is not None else '',basename(g) if isinstance(g,str) else 'U') for g in gate]
		coefficient = symbols.get(coefficient,coefficient)
		interaction = [['i...j'] for d in data] if interaction is None else [flatten(i,cls=list,unique=False) for i in interaction]
		locality = len(flatten([self.site,site],cls=list,unique=True))

		self.data = array([d for i in range(self.M) for d in [*self.data[i*(self.size//self.M):i*(self.size//self.M)+index],*data,*self.data[i*(self.size//self.M)+index:(i+1)*(self.size//self.M)]]])
		self.site = [d for i in range(self.M) for d in [*self.site[i*(self.size//self.M):i*(self.size//self.M)+index],*site,*self.site[i*(self.size//self.M)+index:(i+1)*(self.size//self.M)]]]
		self.string = [*self.string[:index],*string,*self.string[index:]]
		self.coefficient = coefficient		
		self.interaction = [d for i in range(self.M) for d in [*self.interaction[i*(self.size//self.M):i*(self.size//self.M)+index],*interaction,*self.interaction[i*(self.size//self.M)+index:(i+1)*(self.size//self.M)]]]
		self.size = len(self.data)
		self.dim = 1
		self.shape = (self.size,)
		self.locality = locality
		self.diagonal = [isdiag(d) for d in self.data]

		funcs = [d for d in self.data]
		index = arange(self.size)

		self.index = index
		self.funcs = vfunc(funcs,index)

		return


class Operator(Object):
	'''
	Class for Operator
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator,site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		self.delimiter = ' + '
		super().__init__(operator,site=site,string=string,coefficient=coefficient,interaction=interaction,
						 N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return	

	def __setup__(self,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Setup class attributes data,size,shape,locality,site,string,coefficient,interaction
		Args:
			operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				coefficient (int,float,complex,str): scalar coefficient for operator			
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]				
		'''

		self.append(operator,site,string,coefficient,interaction)

		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Return parameterized operator sum(parameters*operator)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			operator (array): Parameterized operator
		'''		
		self.__parameters__(parameters)
		return self.coefficient*summation(self.funcs(self.parameters),self.site,self.diagonal,self.size,self.N,self.D)


	#@partial(jit,static_argnums=(0,))
	def expm(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Assumes pair-wise commuting operators: expm(sum(data)) = prod(expm(data[i]))
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return contract(self.expms(self.coefficient*self.parameters),self.site,self.diagonal,self.size,self.N,self.D)		


	def append(self,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Append data to class
		Args:
			operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				coefficient (int,float,complex,str): scalar coefficient for operator			
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]				
		'''
		self.insert(-1,operator,site,string,coefficient,interaction)
		return

	def insert(self,index,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Insert data to class
		Args:
			operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator			
			site (iterable[iterable[int]],iterable[int]): site of local gate
			string (str): string label for gate
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for gate							
		'''

		if index in [-1] or index>self.size:
			index = self.size

		# Get Operator subclasses
		allowed = ['I','X','Y','Z']
		operators = {
			I:['I'],
			X:['I','X'],
			Y:['I','Y'],
			Z:['I','Z'],
			Hadamard:['I','H'],
			PhaseHadamard:['I','P'],
			Cnot: ['I','C'],
			Toffoli: ['I','T'],
			Pauli: ['I','X','Y','Z'],
			}
		symbols = {'':1,'+':1,'-':-1,'1':1,'-1':-1,'i':1j,'+i':1j,'-i':-1j,None:1}


		kwargs = dict(N=self.N,D=self.D,d=self.d,L=self.L,M=1,T=self.T,p=self.p,space=self.space,time=self.time,system=self.system)
		
		if isinstance(operator,(str,array,dict,Operator)):
			operator = [operator]
			site = [site] if site is not None else None
			interaction = [interaction] if interaction is not None else None

		data = []
		for o in operator:
			if isinstance(o,(Operator)):
				data.append(o)
			elif isinstance(o,(str,array)):
				data.append(_Operator(o,**kwargs))
			elif isinstance(o,(dict,Operator)):
				attr = 'operator'
				assert all(s in allowed for s in list(o[attr])), "Operator field %s not in allowed %r"%(o[attr],allowed)
				for op in operators:
					if all([s in operators[op] for s in o[attr]]):
						data.append(op(o,**kwargs))
						break

		self.__data__(data,site=site,string=string,coefficient=coefficient,interaction=interaction)
		self.__space__(data)
		self.__lattice__(data)

		size = len(data)
		site = [flatten(d.site,cls=list,unique=True) for d in data] if site is None else [flatten(s,cls=list,unique=True) for s in site]
		string = ['%s%s'%(string if string is not None else '',str(d)) for d in data]
		coefficient = symbols.get(coefficient,coefficient)
		interaction = [flatten(d.interaction,cls=list,unique=True) for d in data] if interaction is None else [flatten(i,cls=list,unique=True) for i in interaction]
		locality = len(flatten([self.site,site],cls=list,unique=True))

		self.data = [*self.data[:index],*data,*self.data[index:]]
		self.site = [*self.site[:index],*site,*self.site[index:]]
		self.string = [*self.string[:index],*string,*self.string[index:]]
		self.coefficient = coefficient		
		self.interaction = [*self.interaction[:index],*interaction,*self.interaction[index:]]
		self.size = len(self.data)
		self.dim = 0
		self.shape = (self.size,*[max([d.shape[i] if len(d.shape)>i else 0 for d in self.data]) for i in range(max([len(d.shape) for d in self.data]))])
		self.locality = locality
		self.diagonal = [all(d.diagonal) for d in self.data]

		funcs = [d for d in self.data]
		expms = [d.expm for d in self.data]
		index = arange(self.size)

		self.index = index
		self.funcs = vfunc(funcs,index)
		self.expms = vfunc(expms,index)
		self.transform = []
		self.transformH = []

		return


	def __add__(self,other):
		
		assert isinstance(other,Operator), "+ operation only defined between Operators"
		assert (self.N==other.N) and (self.D==other.D) and (self.d==other.d) and (self.L==other.L), "+ operation only defined between equal dimension Operators"

		operator = [self,other]
		N = self.N
		D = self.D
		d = self.d
		L = self.L
		space = self.space
		system = self.system		

		return Operator(operator,N,D,d,L,space,system)



class Pauli(Operator):
	'''
	Class for Pauli Operator
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator,site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		N = 1 if N is None else N
		D = 2 if D is None else D
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return


	def __data__(self,data,site=None,string=None,coefficient=None,interaction=None):
		'''
		Set class data
		Args:
			data (iterable): Class data
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		'''
		size = len(data)
		coefficient = [1]*size if coefficient is None else coefficient
		for i in range(size):
			try:
				_data = ''.join(data[i])
				data[i] = coefficient[i]*array(qs.quantum_info.Pauli(_data).to_matrix(sparse=False))
			except:
				try:
					data[i] = coefficient[i]*data[i]
				except:
					try:
						data[i] = coefficient*data[i]
					except:
						data[i] = data[i]
		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Return parameterized exponential of operator e^{x*O}
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns
			gate (array): Parameterized exponential of operator
		'''		
		self.__parameters__(parameters)
		return self.coefficient*summation(self.parameters*self.data,self.site,self.diagonal,self.size,self.N,self.D)

	#@partial(jit,static_argnums=(0,))
	def expm(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		# return contract(vexmpeuler(self.data,self.coefficient*self.parameters,self.I),self.site,self.diagonal,self.size,self.N,self.D)
		return expmeuler(self.data[0],self.coefficient*self.parameters,self.I)


	def insert(self,index,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Insert data to class
		Args:
			operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator			
			site (iterable[iterable[int]],iterable[int]): site of local gate
			string (str): string label for gate
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for gate							
		'''

		if index in [-1] or index>self.size:
			index = self.size


		# Transforms of operators for more efficient computations
		Id = 'I'		
		allowed = ['I','X','Y','Z']
		transforms = {
			'I':{'pattern':'I','string':'I','diagonal':True,'transform':Identity(D=self.D)()},
			'X':{'pattern':'X','string':'Z','diagonal':False,'transform':Hadamard(D=self.D)()},
			'Y':{'pattern':'Y','string':'Z','diagonal':False,'transform':PhaseHadamard(D=self.D)()},
			'Z':{'pattern':'Z','string':'Z','diagonal':True,'transform':Identity(D=self.D)()},
			}
		operators = {I:['I'],X:['I','X'],Y:['I','Y'],Z:['I','Z']}

		symbols = {'':1,'+':1,'-':-1,'1':1,'-1':-1,'i':1j,'+i':1j,'-i':-1j,None:1}

		kwargs = dict(N=self.N,D=self.D,d=self.d,L=self.L,M=1,T=self.T,p=self.p,space=self.space,time=self.time,system=self.system)

		if isinstance(operator,(str,dict,array)):
			operator = [operator]
			site = [site] if site is not None else [None]
			interaction = [interaction] if interaction is not None else [None]

		data = []
		_data = []
		transform = []
		sites = []
		strings = []
		coefficients = []
		interactions = []

		for o,s,i in zip(operator,site,interaction):
			if isinstance(o,(str,dict)):
				if isinstance(o,str):
					_data_ = list(o)
					_site = list(range(self.N)) if s is None else list(s)
					_string = '' if string is None else string
					_coefficient = 1 if coefficient is None else coefficient
					_interaction = 'i...j' if i is None else i

				elif isinstance(o,dict):
					_data_ = list(o.get('operator',I))
					_site = list(o.get('site',range(self.N))) if s is None else list(s)
					_string = o.get('string',string if string is not None else '')
					_coefficient = o.get('coefficient',None) if coefficient is None else coefficient
					_interaction = o.get('interaction',None) if i is None else i

				
				_data_ = [Id if j not in _site else _data_[_site.index(j)] for j in range(self.N)]
				_coefficient = symbols.get(_coefficient,_coefficient)/(
							   symbols.get(coefficient,coefficient) 
							   if symbols.get(coefficient,coefficient) != 0 else 1)
				_string = '%s%s%s'%(str(_coefficient) if _coefficient != 1 else '',_string,''.join(_data_))

				# _data for transformed operator for more efficient circuit operations
				__data__ = [_data_[j] for j in range(len(_data_))]
				_transform_ = [Identity(D=self.D)()]*self.N
				_diagonal = [True]*self.N
				for j in range(self.N):
					for t in transforms:
						if transforms[t]['pattern'] in __data__[j]:
							__data__[j] = transforms[t]['string']
							_transform_[j] = transforms[t]['transform']
							_diagonal[j] = transforms[t]['diagonal']
							break

				_transform = product(_transform_,site=_site,diagonal=_diagonal,size=self.N,N=self.N,D=self.D)



			elif isinstance(o,(array)):

				o = _Pauli(o,**kwargs)

				_data_ = o.data[0]
				__data__ = o.data[0]
				_string = o.string[0] if string is None else string
				_site = o.site[0] if s is None else list(s)
				_diagonal = [True]*self.N
				_transform = product([Identity(D=self.D)()]*self.N,site=_site,diagonal=_diagonal,size=self.N,N=self.N,D=self.D)
				_coefficient = symbols.get(o.coefficient,o.coefficient)/(
							   symbols.get(coefficient,coefficient) 
							   if symbols.get(coefficient,coefficient) != 0 else 1)
				_interaction = o.interaction if i is None else i



			data.append(_data_)
			_data.append(__data__)
			transform.append(_transform)
			sites.append(_site)
			strings.append(_string)
			coefficients.append(_coefficient)
			interactions.append(_interaction)

		self.__data__(data,site=sites,string=strings,coefficient=coefficients,interaction=interactions)
		self.__data__(_data,site=sites,string=strings,coefficient=coefficients,interaction=interactions)
		self.__space__(data)
		self.__lattice__(data)

		_data = [diag(d) for d in _data]
		size = len(data)
		site = [flatten(s,cls=list,unique=True) for s in sites]		
		string = [s for s in strings]
		coefficient = symbols.get(coefficient,coefficient)
		interaction = [flatten(i,cls=list,unique=True) for i in interactions]
		locality = len(flatten([self.site,site],cls=list,unique=True))

		self.data = array([*self.data[:index],*data,*self.data[index:]])
		self._data = array([*self._data[:index],*_data,*self._data[index:]])
		self.site = [*self.site[:index],*site,*self.site[index:]]
		self.string = [*self.string[:index],*string,*self.string[index:]]
		self.coefficient = coefficient		
		self.interaction = [*self.interaction[:index],*interaction,*self.interaction[index:]]
		self.size = len(self.data)
		self.dim = 1
		self.shape = (self.size,)
		self.locality = locality
		self.diagonal = [isdiag(d) for d in self.data]

		self.index = arange(self.size)
		self.transform = [t for t in [*self.transform[:index],*transform,*self.transform[index:]]]
		self.transformH = [t.conj().T for t in [*self.transform[:index],*transform,*self.transform[index:]]]

		return


class I(Pauli):
	'''
	Pauli I subclass of Pauli (operator string must contain only I, or will be replaced)
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='I',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	#@partial(jit,static_argnums=(0,))
	def expm(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return diag(exp(self.coefficient*self.parameters*self._data[0]))


class X(Pauli):
	'''
	Pauli X subclass of Pauli (operator string must contain only I or X)
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='X',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	#@partial(jit,static_argnums=(0,))
	def expm(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return (self.transform[0]*exp(self.coefficient*self.parameters*self._data[0]))@self.transformH[0]



class Y(Pauli):
	'''
	Pauli Y subclass of Pauli (operator string must contain only I or Y)
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='Y',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	#@partial(jit,static_argnums=(0,))
	def expm(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return (self.transform[0]*exp(self.coefficient*self.parameters*self._data[0]))@self.transformH[0]


class Z(Pauli):
	'''
	Pauli Z subclass of Pauli (operator string must contain only I or Z)
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='Z',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	#@partial(jit,static_argnums=(0,))
	def expm(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return diag(exp(self.coefficient*self.parameters*self._data[0]))



class Transform(Gate):
	'''
	Class for Transformation Gate
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='I',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	def __data__(self,data,site=None,string=None,coefficient=None,interaction=None):
		'''
		Set class data
		Args:
			data (iterable): Class data
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		'''
		size = len(data)
		for i in range(size):
			data[i] = product([identity(self.D)]*self.N,site=site,diagonal=self.diagonal,size=self.N,N=self.N,D=self.D)
		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Return parameterized exponential of operator e^{x*O}
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns
			gate (array): Parameterized exponential of operator
		'''		
		self.__parameters__(parameters)
		return contract(self.coefficient*self.parameters*self.data,self.site,self.diagonal,self.size,self.N,self.D)

	#@partial(jit,static_argnums=(0,))
	def expm(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		# return contract(vexmpeuler(self.data,self.coefficient*self.parameters,self.I),self.site,self.diagonal,self.size,self.N,self.D)
		return contract(self.coefficient*self.parameters*self.data,self.site,self.diagonal,self.size,self.N,self.D)

	def insert(self,index,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Insert data to class
		Args:
			operator (iterable[str,dict],str,dict): String for Operator or Dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator			
			site (iterable[iterable[int]],iterable[int]): site of local gate
			string (str): string label for gate
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for gate							
		'''

		if index in [-1] or index>self.size:
			index = self.size

		# Transforms of operators for more efficient computations
		N = self.N//self.locality if self.locality<=self.N else self.N
		locality = self.locality if self.locality>0 else self.N
		basis = self.basis if self.basis is not None else operator if isinstance(operator,str) else 'I'
		Id = 'I'				
		allowed = ['I',basis]
		symbols = {'':1,'+':1,'-':-1,'1':1,'-1':-1,'i':1j,'+i':1j,'-i':-1j,None:1}


		if isinstance(operator,(str,dict)):
			operator = [operator]
			site = [site] if site is not None else [None]
			interaction = [interaction] if interaction is not None else [None]

		data = []
		sites = []
		strings = []
		interactions = []
		coefficients = []

		for o,s,i in zip(operator,site,interaction):
			_data_ = [basis]*N
			_site = list(range(locality)) if s is None else list(s)
			_string = '' if string is None else string
			_coefficient = 1 if coefficient is None else coefficient		
			_interaction = 'i...j' if i is None else i
			
			_data_ = [Id if j not in _site else _data_[_site.index(j)] for j in range(N)]
			_coefficient = symbols.get(_coefficient,_coefficient)/(
						   symbols.get(coefficient,coefficient) 
						   if symbols.get(coefficient,coefficient) != 0 else 1)
			_string = '%s%s%s'%(str(_coefficient) if _coefficient != 1 else '',_string,''.join(_data_))

			data.append(_data_)
			sites.append(_site)
			strings.append(_string)
			interactions.append(_interaction)
			coefficients.append(_coefficient)

		self.__data__(data,site=sites,string=strings,coefficient=coefficients,interaction=interactions)
		self.__space__(data)
		self.__lattice__(data)

				
		size = len(data)
		site = [flatten(s,cls=list,unique=True) for s in sites]		
		string = ['%s%s'%(string if string is not None else '',s) for s in strings]
		coefficient = symbols.get(coefficient,coefficient)
		interaction = [flatten(i,cls=list,unique=True) for i in interactions]
		locality = len(flatten([self.site,site],cls=list,unique=True))

		self.data = array([*self.data[:index],*data,*self.data[index:]])
		self.site = [*self.site[:index],*site,*self.site[index:]]
		self.string = [*self.string[:index],*string,*self.string[index:]]
		self.coefficient = coefficient		
		self.interaction = [*self.interaction[:index],*interaction,*self.interaction[index:]]
		self.size = len(self.data)
		self.dim = 1
		self.shape = (self.size,)
		self.locality = locality
		self.diagonal = [isdiag(d) for d in self.data]

		self.index = arange(self.size)

		return

class Identity(Transform):
	'''
	Class for Identity Operator
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='I',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		N = 1 if N is None else N
		D = 2 if D is None else D
		self.basis = 'I'
		self.locality = 1
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	def __data__(self,data,site=None,string=None,coefficient=None,interaction=None):
		'''
		Set class data
		Args:
			data (iterable): Class data
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		'''
		size = len(data)
		for i in range(size):
			data[i] = product([identity(self.D)]*self.N,site=site,diagonal=self.diagonal,size=self.N,N=self.N,D=self.D)
		return

class Hadamard(Transform):
	'''
	Class for Hadamard Operator
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='H',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		N = 1 if N is None else N
		D = 2 if D is None else D
		self.basis = 'H'
		self.locality = 2
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	def __data__(self,data,site=None,string=None,coefficient=None,interaction=None):
		'''
		Set class data
		Args:
			data (iterable): Class data
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		'''
		size = len(data)
		for i in range(size):
			data[i] = product([hadamard(self.D)]*self.N,site=site,diagonal=self.diagonal,size=self.N,N=self.N,D=self.D)
		return


class PhaseHadamard(Transform):
	'''
	Class for PhaseHadamard Operator
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='P',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		N = 1 if N is None else N
		D = 2 if D is None else D
		self.basis = 'P'		
		self.locality = 2
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	def __data__(self,data,site=None,string=None,coefficient=None,interaction=None):
		'''
		Set class data
		Args:
			data (iterable): Class data
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		'''
		size = len(data)
		for i in range(size):
			data[i] = product([phasehadamard(self.D)]*self.N,site=site,diagonal=self.diagonal,size=self.N,N=self.N,D=self.D)
		return


class Cnot(Transform):
	'''
	Class for Cnot Operator
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='C',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		N = 2 if N is None else N
		D = 2 if D is None else D
		self.basis = 'C'		
		self.locality = 2
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	def __data__(self,data,site=None,string=None,coefficient=None,interaction=None):
		'''
		Set class data
		Args:
			data (iterable): Class data
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		'''
		size = len(data)
		for i in range(size):
			try:
				data[i] = product([cnot(self.D**2)]*(self.N//2),site=site,diagonal=self.diagonal,size=self.N,N=self.N,D=self.D)
			except:
				data[i] = data[i]
		return


class Toffoli(Transform):
	'''
	Class for Tofolli Operator
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='T',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		N = 3 if N is None else N
		D = 2 if D is None else D
		self.basis = 'T'		
		self.locality = 3
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	def __data__(self,data,site=None,string=None,coefficient=None,interaction=None):
		'''
		Set class data
		Args:
			data (iterable): Class data
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		'''
		size = len(data)
		for i in range(size):
			try:
				data[i] = product([toffoli(self.D**3)]*(self.N//3),site=site,diagonal=self.diagonal,size=self.N,N=self.N,D=self.D)
			except:
				data[i] = data[i]
		return


class _Operator(Operator):
	'''
	User subclass of Operator
	Args:
		operator (iterable[str,array],str,array): Path for operator array to be loaded, or array for Operator
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]						
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator,site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	def __setup__(self,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Setup class attributes data,size,shape,locality,site,string,coefficient,interaction
		Args:
			operator (iterable[str,array],str,array): Path to load array, array for operator
			site (iterable[iterable[int]],iterable[int]): site of local operator
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator		
		'''	

		self.append(operator,site,string,coefficient,interaction)

		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Return parameterized exponential of operator e^{x*O}
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns
			gate (array): Parameterized exponential of operator
		'''		
		self.__parameters__(parameters)
		return self.coefficient*summation(self.parameters*self.data,self.site,self.diagonal,self.size,self.N,self.D)


	#@partial(jit,static_argnums=(0,))
	def expm(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return self.coefficient*contract(vexpm(self.parameters*self.data),self.site,self.diagonal,self.size,self.N,self.D)


	def append(self,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Append operator to class
		Args:
			operator (iterable[str,array],str,array): Path to load array, array for operator
			site (iterable[iterable[int]],iterable[int]): site of local operator
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator				
		'''
		self.insert(-1,operator,site,string,coefficient,interaction)
		return


	def insert(self,index,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Insert data to class
		Args:
			operator (iterable[str,array],str,array): Path to load array, array for operator
			site (iterable[iterable[int]],iterable[int]): site of local operator
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator		
		'''

		if index in [-1] or index>self.size:
			index = self.size
	
		symbols = {'':1,'+':1,'-':-1,'1':1,'-1':-1,'i':1j,'+i':1j,'-i':-1j,None:1}

		kwargs = dict(N=self.N,D=self.D,d=self.d,L=self.L,M=1,T=self.T,p=self.p,space=self.space,time=self.time,system=self.system)
		
		if isinstance(operator,(str,array)):
			operator = [operator]
			site = [site] if site is not None else None
			interaction = [interaction] if interaction is not None else None

		data = []
		for o in operator:
			if isinstance(o,(str)):
				data.append(array(load(o,default=I(**kwargs)(),dtype=self.dtype)))
			elif isinstance(o,(array)):
				data.append(o)
		
		self.__data__(data,site=site,string=string,coefficient=coefficient,interaction=interaction)
		self.__space__(data)
		self.__lattice__(data)

		
		size = len(data)
		site = [list(range(self.N)) for d in data] if site is None else [flatten(s,cls=list,unique=True) for s in site]
		string = ['%s%s'%(string if string is not None else '',basename(o) if isinstance(o,str) else 'O') for o in operator]
		coefficient = symbols.get(coefficient,coefficient)
		interaction = [['i...j'] for d in data] if interaction is None else [flatten(i,cls=list,unique=False) for i in interaction]
		locality = len(flatten([self.site,site],cls=list,unique=True))

		self.data = array([*self.data[:index],*data,*self.data[index:]])
		self.site = [*self.site[:index],*site,*self.site[index:]]
		self.string = [*self.string[:index],*string,*self.string[index:]]
		self.coefficient = coefficient		
		self.interaction = [*self.interaction[:index],*interaction,*self.interaction[index:]]
		self.size = len(self.data)
		self.dim = 1
		self.shape = (self.size,)		
		self.locality = locality
		self.diagonal = [isdiag(d) for d in self.data]

		self.index = arange(self.size)

		return


class _Pauli(Pauli):
	'''
	User subclass of Pauli
	Args:
		operator (iterable[str,array],str,array): Path for operator array to be loaded, or array for Operator
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]						
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator,site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	def __setup__(self,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Setup class attributes data,size,shape,locality,site,string,coefficient,interaction
		Args:
			operator (iterable[str,array],str,array): Path to load array, array for operator
			site (iterable[iterable[int]],iterable[int]): site of local operator
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator		
		'''	

		self.append(operator,site,string,coefficient,interaction)

		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Return parameterized exponential of operator e^{x*O}
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns
			gate (array): Parameterized exponential of operator
		'''		
		self.__parameters__(parameters)
		return self.coefficient*summation(self.parameters*self.data,self.site,self.diagonal,self.size,self.N,self.D)


	#@partial(jit,static_argnums=(0,))
	def expm(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return self.coefficient*contract(vexpm(self.parameters*self.data),self.site,self.diagonal,self.size,self.N,self.D)


	def append(self,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Append operator to class
		Args:
			operator (iterable[str,array],str,array): Path to load array, array for operator
			site (iterable[iterable[int]],iterable[int]): site of local operator
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator				
		'''
		self.insert(-1,operator,site,string,coefficient,interaction)
		return


	def insert(self,index,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Insert data to class
		Args:
			operator (iterable[str,array],str,array): Path to load array, array for operator
			site (iterable[iterable[int]],iterable[int]): site of local operator
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator		
		'''

		if index in [-1] or index>self.size:
			index = self.size
	
		symbols = {'':1,'+':1,'-':-1,'1':1,'-1':-1,'i':1j,'+i':1j,'-i':-1j,None:1}

		kwargs = dict(N=self.N,D=self.D,d=self.d,L=self.L,M=1,T=self.T,p=self.p,space=self.space,time=self.time,system=self.system)
		
		if isinstance(operator,(str,array)):
			operator = [operator]
			site = [site] if site is not None else None
			interaction = [interaction] if interaction is not None else None

		data = []
		for o in operator:
			if isinstance(o,(str)):
				data.append(array(load(o,default=I(**kwargs)(),dtype=self.dtype)))
			elif isinstance(o,(array)):
				data.append(o)
		
		self.__data__(data,site=site,string=string,coefficient=coefficient,interaction=interaction)
		self.__space__(data)
		self.__lattice__(data)

		size = len(data)
		site = [list(range(self.N)) for d in data] if site is None else [flatten(s,cls=list,unique=True) for s in site]
		string = ['%s%s'%(string if string is not None else '',basename(o) if isinstance(o,str) else 'O') for o in operator]
		coefficient = symbols.get(coefficient,coefficient)
		interaction = [['i...j'] for d in data] if interaction is None else [flatten(i,cls=list,unique=False) for i in interaction]
		locality = len(flatten([self.site,site],cls=list,unique=True))

		self.data = array([*self.data[:index],*data,*self.data[index:]])
		self.site = [*self.site[:index],*site,*self.site[index:]]
		self.string = [*self.string[:index],*string,*self.string[index:]]
		self.coefficient = coefficient		
		self.interaction = [*self.interaction[:index],*interaction,*self.interaction[index:]]
		self.size = len(self.data)
		self.dim = 1
		self.shape = (self.size,)		
		self.locality = locality
		self.diagonal = [isdiag(d) for d in self.data]

		self.index = arange(self.size)

		return

class RI(I):
	'''
	Rotation Pauli I subclass of Pauli (operator string must contain only I, or will be replaced)
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='I',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return self.expm(self.parameters)


class RX(X):
	'''
	Rotation Pauli X subclass of Pauli (operator string must contain only I or X)
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='X',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return self.expm(self.parameters)



class RY(Y):
	'''
	Rotation Pauli Y subclass of Pauli (operator string must contain only I or Y)
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator='Y',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return self.expm(self.parameters)


class RZ(Z):
	'''
	Pauli Z subclass of Pauli (operator string must contain only I or Z)
	Args:
		operator (iterable[dict,str,array,Operator],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space		

		system (dict,System): System attributes
	'''

	def __init__(self,operator='Z',site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site,string,coefficient,interaction,N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Matrix exponential of paramaterized operator
		Args:
			parameters (array): Parameters to parameterize exponential of operator
		Returns:
			out (array): Matrix exponential of parameterized operator
		'''
		self.__parameters__(parameters)
		return self.expm(self.parameters)



class Unitary(Operator):
	'''
	Class for Unitary, based on matrix exponential of Operator, Unitary(parameters) = expm(Operator(parameters))
	Args:
		operator (iterable[dict,str,array,Operator,Unitary],dict,str,array,Operator): Dictionaries of fields for Operator, or Path to load Operator, or array for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (iterable[str]): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			coefficient (int,float,complex,str): scalar coefficient for operator			
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		site (iterable[iterable[int]],iterable[int]): site of local operators
		string (str): string label for operator
		coefficient (int,float,complex,str): scalar coefficient for operator			
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]			
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''
	
	def __init__(self,operator,site=None,string=None,coefficient=None,interaction=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):

		if isinstance(operator,(Operator)):
			operator = [o for o in operator.data]

		super().__init__(operator,site=site,string=string,coefficient=coefficient,interaction=interaction,
						 N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Return parameterized matrix exponential operator expm(parameters*operator) ~ trotterized version of order p
		Args:
			parameters (array): Parameters to parameterize matrix exponential operator
		Returns
			operator (array): Parameterized matrix exponential operator
		'''		
		self.__parameters__(parameters)
		if self.p:
			return exponentiate(self.expms(self.coefficient*self.parameters/self.p),self.p,self.site,self.diagonal,self.size,self.N,self.D)
		else:
			return expm(self.coefficient*summation(self.funcs(self.parameters),self.site,self.diagonal,self.size,self.N,self.D))


class Hamiltonian(Operator):
	'''
	Hamiltonian class of grouped Operators
	Args:
		operator (iterable[str,array,dict,iterable,Operator],str,array,dict,iterable,Operator): Path to load array, array or, (iterable) of dictionaries of fields for Operator
			For dictionaries defining Operators, 
			i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
			operator (str,iterable[str],Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
			site (iterable[iterable[int]],iterable[int]): site of local operators
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator			
		site (iterable[iterable[int]],iterable[int]): site of local operator
		string (str): string label for operator
		interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		coefficient (int,float,complex,str): scalar coefficient for operator					
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator,site=None,string=None,interaction=None,coefficient=None,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site=site,string=string,coefficient=coefficient,interaction=interaction,
				 N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return


	def __setup__(self,operator,site=None,string=None,interaction=None,coefficient=None):
		'''
		Setup class attributes data,size,shape,locality,site,string,coefficient,interaction
		Args:	
			operator (iterable[str,array,dict,iterable,Operator],str,array,dict,iterable,Operator): Path to load array, array or, (iterable) of dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator			
			site (iterable[iterable[int]],iterable[int]): site of local operator
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator									
		'''	

		self.append(operator,site,string,interaction,coefficient)

		return

	def append(self,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Append data to class
		Args:
			operator (iterable[str,array,dict,iterable,Operator],str,array,dict,iterable,Operator): Path to load array, array or, (iterable) of dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator			
			site (iterable[iterable[int]],iterable[int]): site of local operator
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator							
		'''
		self.insert(-1,operator,site,string,coefficient,interaction)
		return

	def insert(self,index,operator,site=None,string=None,coefficient=None,interaction=None):
		'''
		Insert data to class
		Args:
			operator (iterable[str,array,dict,iterable,Operator],str,array,dict,iterable,Operator): Path to load array, array or, (iterable) of dictionaries of fields for Operator
				For dictionaries defining Operators, 
				i.e) {"operator":["Z","Z"],"site":[0,1],"string":"J","interaction":["i<j"],"coefficient":None}			
				operator (str,iterable[str],Operator): Path to load array, operator string of length N, iterable of paths, iterable of local operator strings, allowed strings in ["I","X","Y","Z","U"], or Operator
				site (iterable[iterable[int]],iterable[int]): site of local operators
				string (str): string label for operator
				interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				coefficient (int,float,complex,str): scalar coefficient for operator			
			site (iterable[iterable[int]],iterable[int]): site of local operator
			string (str): string label for operator
			interaction (iterable[str],str): type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			coefficient (int,float,complex,str): scalar coefficient for operator							
		'''

		# Setup Operator attributes
		attrs = ['operator','site','string','coefficient','interaction']

		interactions = {site: self.lattice(site) for site in ['i','i<j','<ij>','i...j']}	# interactions types on lattice
		locality = 2 # maximum number of body interactions
		sites = ['i','j'] # allowed symbolic sites
	

		# Ensure operator is of the form of dictionary of attributes of iterable of interable of values {attr:[[value_i_j]]}
		# where each outer iterable is a group of operators, and each inner iterable is the operators in that group
		operator = copy.deepcopy(operator)
		if isinstance(operator,(str,array,Operator)):
			super().insert(index,operator,site,string,coefficient,interaction)
			return
		elif isinstance(operator,dict):
			pass
		elif isinstance(operator,(list,tuple,objs,array)):
			for i in range(len(operator)):
				if isinstance(operator[i],(dict)):
					operator[i] = [operator[i]]
			if all(isinstance(o,dict) for o in operator):
				operator = {attr: [[o[attr]] for o in operator] for attr in attrs}	
			elif all(isinstance(o,(list,tuple,objs,array)) for o in operator):
				operator = {attr: [[s[attr] for s in o] for o in operator] for attr in attrs}
		else:
			raise "operator argument not of the form of dict, iterable[dict,iterable]"

		# Ensure all arguments are iterable of interable
		if any(not isinstance(operator[attr],(list,tuple,objs,array)) for attr in operator):
			for attr in operator:
				operator[attr] = [[operator[attr]]]

		if any(not isinstance(o,(list,tuple,objs,array)) for attr in operator for o in operator[attr]):
			for attr in operator:
				for i in range(len(operator[attr])):
					operator[attr][i] = [operator[attr][i]]


		# Get size: number of groups, and sizes: number of operators in each group
		size = min([len(operator[attr]) for attr in operator])
		sizes = [min(len(operator[attr][i]) for attr in operator) for i in range(size)]

		# Get all groupings of sites and interactions if any symbolic sites, 
		# replace symbolic with sites on lattice and update sizes
		# ensure all symbolic site replacements are allowed interactions on lattice
		for i in range(size):
			_operator = {attr: [] for attr in operator}
			for j in range(sizes[i]):
				if any(isinstance(s,str) for s in operator['site'][i][j]):
					for k,s in enumerate(interactions[operator['interaction'][i][j]]):
						if ((s not in _operator['site']) or 
							(s in _operator['site'] and any(any(operator[attr][i][j] != _operator[attr][l]
							 for attr in operator) for l in range(sizes[i]) if _operator['site'][l] == s))):
							for attr in operator:
								if attr in ['site']:
									value = [dict(zip(sites,s if not isinstance(s,int) else [s])).get(u,parse(u,int)) 
											for u in operator[attr][i][j]]
								else:
									value = copy.deepcopy(operator[attr][i][j])
								_operator[attr].append(value)
				else:
					for attr in operator:
						value = copy.deepcopy(operator[attr][i][j])
						_operator[attr].append(value)

	
				sizes[i] = min([len(_operator[attr]) for attr in operator])


			_operator = {attr: [_operator[attr][k] for k in range(sizes[i]) 
						 		if (any(_operator['site'][k] in interactions[s]
						 			for s in interactions))]
						 for attr in _operator}

			sizes[i] = min([len(_operator[attr]) for attr in operator])

			for attr in operator:
				value = _operator[attr]
				operator[attr][i] = value


		for i in range(size-1,-1,-1):
			if sizes[i] == 0:
				for attr in operator:
					operator[attr].pop(i);

		size = min([len(operator[attr]) for attr in operator])
		sizes = [min(len(operator[attr][i]) for attr in operator) for i in range(size)]


		# Initialize Operators
		kwargs = dict(N=self.N,D=self.D,d=self.d,L=self.L,M=1,T=self.T,p=self.p,space=self.space,time=self.time,system=self.system)
		
		attr = 'operator'
		for i in range(size):
			o = [{attr: operator[attr][i][j] for attr in operator} for j in range(sizes[i])]
			o = Operator(o,**kwargs)
			super().insert(index,o,site,string,coefficient,interaction)
		
		return