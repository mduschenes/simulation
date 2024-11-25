#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime,shutil,traceback
from time import time as timer
from functools import partial
import atexit

# Logging
import logging
import logging.config,configparser

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,gradient
from src.utils import array,arange,eye,rand,einsum,dot,prod
from src.utils import unique,ceil,sort,repeat,vstack,concatenate,mod,sqrt,datatype
from src.utils import inner_norm,inner_abs2,inner_real,inner_imag
from src.utils import gradient_inner_norm,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag

from src.utils import itg,dbl,flt,delim,Null,null,scalars,arrays

from src.iterables import Dict,Dictionary,getter,setter
from src.io import join,split,rm,exists
from src.logger import Logger


class System(Dictionary):
	'''
	System attributes (dtype,format,device,seed,verbose,...)
	Args:
		dtype (str,data-type): Data type of class
		format (str): Format of array
		device (str): Device for computation
		seed (array,int): Seed for random number generation
		key (object): key for class
		timestamp (str): timestamp for class
		backend (str): backend for class
		architecture (str): architecture for class
		unit (int,float): units of values
		verbose (bool,str): Verbosity of class	
		args (dict,System): Additional system attributes
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,*args,**kwargs):

		defaults = {
			'string':__name__,
			'dtype':'complex',
			'format':'array',
			'device':'cpu',
			'backend':None,
			'architecture':None,
			'unit':None,			
			'seed':None,
			'key':None,
			'instance':None,
			'instances':None,
			'timestamp':datetime.datetime.now().strftime('%d.%M.%Y.%H.%M.%S.%f'),
			'cwd':None,
			'path':None,
			'conf':None,
			'logger':None,
			'cleanup':None,
			'verbose':None,
		}

		def updates(kwargs,defaults):
			
			attr = 'unit'
			kwargs[attr] = defaults.get(attr) if kwargs.get(attr,defaults.get(attr)) is None else kwargs.get(attr)

			attr = 'backend'
			kwargs[attr] = os.environ.get('NUMPY_BACKEND',str(None)).lower() if kwargs.get(attr,defaults.get(attr)) is None else os.environ.get(kwargs.get(attr,defaults.get(attr)),kwargs.get(attr,defaults.get(attr))).lower()
			
			attr = 'instances'
			if kwargs.get(attr) is not None:
				kwargs[attr] = Dict(kwargs[attr])
			return

		updates(kwargs,defaults)
		
		setter(kwargs,defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)

		self.__logger__()
		self.__clean__()

		return

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return self.__str__()

	def __clean__(self,cleanup=None):
		'''
		Set cleanup state of class
		Args:
			cleanup (bool): Cleanup
		'''

		cleanup = self.cleanup if cleanup is None else cleanup

		if cleanup:
			atexit.register(self.__atexit__)
		else:
			atexit.unregister(self.__atexit__)

		return
		
	def __atexit__(self):
		'''
		Cleanup upon class exit
		'''
		return


	def __logger__(self):
		'''
		Setup logger
		'''

		path = self.cwd
		root = path

		if not isinstance(self.logger,Logger):
			name = __name__
			conf = join(self.conf,root=root) if exists(join(self.conf,root)) else self.conf
			file = join(self.logger,root=root)
			cleanup = self.cleanup

			self.logger = Logger(name,conf,file=file,cleanup=cleanup)

		return

	def log(self,msg,verbose=None):
		'''
		Log messages
		Args:
			msg (str): Message to log
			verbose (int,str,bool): Verbosity of message			
		'''
		if verbose is None:
			verbose = self.verbose
		if msg is None:
			return
		msg += '\n'
		self.logger.log(verbose,msg)
		return

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		
		msg = None
		self.log(msg,verbose=verbose)
		return

class Space(System):
	'''
	Hilbert space class for Operators with size n
	Args:
		N (int): Number of qudits
		D (int): Dimension of qudits
		space (str,Space): Type of Hilbert space
		system (dict,System): System attributes
		kwargs (dict): Additional keyword arguments
	'''
	def __init__(self,N,D,space,system=None,**kwargs):

		setter(kwargs,system,delimiter=delim,default=False)
		super().__init__(**kwargs)

		self.N = N
		self.D = D
		self.n = None
		self.g = None
		self.space = space
		self.string = None		
		self.default = 'spin'
		self.dtype = datatype(self.dtype)
		self.system = system

		self.__setup__()
		
		return

	def __setup__(self):
		'''
		Setup space attributes space,string,n
		'''

		wrapper = lambda func,dtype: (lambda *args,**kwargs: array(func(*args,**kwargs),dtype=dtype).item())

		funcs =  {
			'spin':{
				'N': (lambda N,D,n,g,space: round(log(n)/log(D))),
				'D': (lambda N,D,n,g,space: round(n**(1/N))),
				'n': (lambda N,D,n,g,space: round(D**N)),
				'g': (lambda N,D,n,g,space: round(n**2 - 1)),
				},
			None:{
				'N': (lambda N,D,n,g,space: round(log(n)/log(D))),
				'D': (lambda N,D,n,g,space: round(n**(1/N))),
				'n': (lambda N,D,n,g,space: round(D**N)),
				'g': (lambda N,D,n,g,space: round(n**2 - 1)),
				},				
			}

		dtypes = {
			'M': int,'T':self.dtype,'tau':self.dtype,
		}

		if isinstance(self.space,Space):
			self.space = self.space.space
		if self.space is None:
			self.space = self.default

		self.funcs = funcs.get(self.space,funcs[self.default])
		self.funcs = {attr: wrapper(self.funcs[attr],dtypes.get(attr)) for attr in self.funcs}

		self.__string__()
		self.__size__()

		return

	def __string__(self):
		self.string = self.space
		return

	def __size__(self):

		assert sum(var is not None for var in [self.N,self.D]) > 1,'2 of 2 of N, D must be non-None'

		self.n = self.funcs['n'](self.N,self.D,self.n,self.g,self.space)
		self.g = self.funcs['g'](self.N,self.D,self.n,self.g,self.space)

		self.size = self.n

		return 

	
	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return self.__str__()
		

class Time(System):
	'''
	Time evolution class for Operators with size n
	Args:
		M (int): Number of time steps
		T (int): Simulation time
		tau (float): Simulation time scale
		P (int): Trotter order
		time (str,Time): Type of Time evolution space
		system (dict,System): System attributes
		kwargs (dict): Additional keyword arguments
	'''
	def __init__(self,M,T,tau,P,time,system=None,**kwargs):

		setter(kwargs,system,delimiter=delim,default=False)
		super().__init__(**kwargs)

		self.M = M
		self.T = T
		self.tau = tau
		self.P = P
		self.time = time
		self.string = None				
		self.default = 'linear'
		self.dtype = datatype(self.dtype)
		self.system = system

		self.__setup__()
		
		return

	def __setup__(self):
		'''
		Setup time evolution attributes tau
		'''

		wrapper = lambda func,dtype: (lambda *args,**kwargs: array(func(*args,**kwargs),dtype=dtype).item())

		funcs =  {
			'linear':{
				'M': (lambda M,T,tau,time: round(self.T/self.tau)),
				'T': (lambda M,T,tau,time: self.M*self.tau),
				'tau': (lambda M,T,tau,time: self.T/self.M),
				},
			None:{
				'M': (lambda M,T,tau,time: round(self.T/self.tau)),
				'T': (lambda M,T,tau,time: self.M*self.tau),
				'tau': (lambda M,T,tau,time: self.T/self.M),
				},				
			}

		dtypes = {
			'M': int,'T':self.dtype,'tau':self.dtype,
		}

		if isinstance(self.time,Time):
			self.time = self.time.time
		if self.time is None:
			self.time = self.default

		self.funcs = funcs.get(self.time,funcs[self.default])
		self.funcs = {attr: wrapper(self.funcs[attr],dtypes.get(attr)) for attr in self.funcs}
		
		self.__string__()
		self.__size__()
	
		return

	def __string__(self):
		self.string = self.time
		return

	def __size__(self):

		assert sum(var is not None for var in [self.M,self.T,self.tau]) > 1,'2 of 3 of M, T, tau must be non-None'

		if (self.M is not None) and (self.T is not None) and (self.tau is None):
			self.tau = self.funcs['tau'](self.T,self.M,self.tau,self.time)
		elif (self.M is not None) and (self.T is None) and (self.tau is not None):
			self.T = self.funcs['T'](self.T,self.M,self.tau,self.time)
		elif (self.M is None) and (self.T is not None) and (self.tau is not None):
			self.M = self.funcs['M'](self.T,self.M,self.tau,self.time)
		return 

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return self.__str__()


class Lattice(System):
	'''
	Define a hyper lattice class
	Args:
		N (int): Lattice length along axis
		d (int): Dimension of lattice
		L (int,float): Scale in system
		delta (float): Length scale in system	
		lattice (str,Lattice): Type of lattice, allowed strings in ['square','square-nearest']
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logger,logging,cleanup,verbose)		
		kwargs (dict): Additional keyword Arguments
	'''	
	def __init__(self,N,d,L=None,delta=None,lattice='square',system=None,**kwargs):

		# Define system
		setter(kwargs,system,delimiter=delim,default=False)
		super().__init__(**kwargs)

		wrapper = lambda func,dtype: (lambda *args,**kwargs: array(func(*args,**kwargs),dtype=dtype).item())

		funcs = {
			'square': {
				'L': (lambda N,d,L,delta,n,z,lattice: L if L is not None else float(N)),
				'delta': (lambda N,d,L,delta,n,z,lattice: delta if delta is not None else L/n),
			},
			None: {
				'L': (lambda N,d,L,delta,n,z,lattice: L if L is not None else float(N)),
				'delta': (lambda N,d,L,delta,n,z,lattice: delta if delta is not None else L/n),
			}			
		}

		dtypes = {
			'M': int,'T':self.dtype,'tau':self.dtype,
		}

		# Define lattice
		if isinstance(lattice,Lattice):
			lattice = lattice.lattice
		else:
			lattice = lattice

		# Define parameters of system        
		self.N = N
		self.d = d
		self.L = L
		self.delta = delta
		self.lattice = lattice	
		self.string = None				
		self.default = 'square'
		self.dtype = datatype(self.dtype)		
		self.system = system

		# Check system
		self.datatype = int

		# Define linear size n and coordination number z	
		if self.lattice is None:
			N = 0
			S = 0
			d = 0
			n = 0
			s = 0
			z = 0
			self.lattice = self.default
		elif self.lattice in ['square','square-nearest']:
			n = round(N**(1/d))
			s = n**(d-1)
			z = 2*d
			S = 2*(((n)**(d-1)) + (d-1)*((n-1)**(d-1)))
			assert n**d == N, 'N != n^d for N=%d, d=%d, n=%d'%(N,d,n)
		else:
			n = round(N**(1/d))
			s = n**(d-1)
			z = 2*d
			S = 2*(((n)**(d-1)) + (d-1)*((n-1)**(d-1)))
			assert n**d == N, 'N != n^d for N=%d, d=%d, n=%d'%(N,d,n)

		self.S = S
		self.n = n
		self.s = s
		self.z = z

		self.funcs = funcs.get(self.lattice,funcs[self.default])
		self.funcs = {attr: wrapper(self.funcs[attr],dtypes.get(attr)) for attr in self.funcs}
	
		# Define attributes
		self.__size__()
		self.__shape__()
		self.__string__()

		# Define array of vertices
		self.vertices = arange(self.N)
		
		# n^i for i = 0:d-1 array
		if isinstance(self.n,scalars):
			self.n_i = self.n**arange(self.d,dtype=self.datatype)
		else:
			self.n_i = array([prod(self.n[i+1:]) for i in range(self.d)])
		
		# Arrays for finding coordinate and linear position in d dimensions
		self.I = eye(self.d)
		self.R = arange(1,max(2,ceil(self.n/2)),dtype=self.datatype)

		return

	def __call__(self,site=None):
		'''
		Get list of lists of sites of lattice
		Args:
			site (str,int): Type of sites, either int for unique site-length list of vertices, or allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		Returns:
			sites (generator): Generator of site-length lists of lattice
		'''

		# Unique site-length lists if site is int
		sites = None
		if isinstance(site,(int,itg)):
			k = site
			conditions = None
			sites = self.iterable(k,conditions)
		elif isinstance(site,(str)):
			if site in ['i']:
				sites = ([i] for i in self.vertices)
			elif site in ['ij']:
				sites = ([i,j] for i in self.vertices for j in self.vertices)
			elif site in ['i<j']:
				k = 2
				conditions = lambda i,k: all([i[j]<i[j+1] for j in range(k-1)])	
				sites = self.iterable(k,conditions)
			elif site in ['<ij>']:
				if self.z > self.N:
					sites = ()
				elif self.z > 0:
					sites = (i for i in self.nearestneighbours(r=1,sites=True,edges=True,periodic=True))
				else:
					sites = ()
			elif site in ['>ij<']:
				if self.z > self.N:
					sites = ()
				elif self.z > 0:
					sites = (i for i in self.nearestneighbours(r=1,sites=True,edges=True,periodic=False))
				else:
					sites = ()					
			elif site in ['i...j']:
				sites = (range(self.N) for i in range(self.N))
		else:
			k = 2
			conditions = None
			sites = self.iterable(k,conditions)
		if sites is None:
			sites = ()
		sites = (list(map(int,i)) for i in sites)
		return sites


	def __string__(self):
		self.string = self.lattice if self.lattice is not None else 'null'
		return
		
	def __size__(self):
	
		self.size = self.N

		self.L = self.funcs['L'](self.N,self.d,self.L,self.delta,self.n,self.z,self.lattice)
		self.delta = self.funcs['delta'](self.N,self.d,self.L,self.delta,self.n,self.z,self.lattice)

		return 

	def __shape__(self):
		self.shape = (self.N,self.z)
		return

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return self.__str__()



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
		isint = isinstance(site,(int,itg))

		if isint:
			site = array([site])
		position = mod(((site[:,None]/self.n_i)).
						astype(self.datatype),self.n)
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
			position = array([position])
		
		site = dot(position,self.n_i).astype(self.datatype)

		if is1d:
			return site[0]
		else:
			return site


	def nearestneighbours(self,r=None,sites=None,vertices=None,edges=None,periodic=True):
		'''
		Return array of neighbouring spin vertices 
		for a given site and r-distance bonds
		i.e) [self.site(put(self.position(site),i,
						lambda x: mod(x + s*r,self.n))) 
						for i in range(self.d)for s in [1,-1]])
		Args:
			r (int,iterable): Radius of number of nearest neighbours away to compute nearest neighbours on lattice, an integer or of shape (l,)
			sites (bool): Include sites with nearest neighbours
			vertices (array): Vertices to compute nearest neighbours on lattice of shape (N,)
			edges (bool,int,array): Edges to compute nearest neighbours, defaults to all edges, True or 1 for forward neighbours, False or -1 for backward neighbours
			periodic (bool): Include periodic nearest neighbours at boundaries
		Returns:
			nearestneighbours (array): Array of shape (N,z) or (l,N,z) of nearest neighbours a manhattan distance r away
		'''

		if vertices is None:
			vertices = self.vertices

		# TODO: Implement open boundary conditions for nearest neighbours in d>1 dimensions
		if not periodic or self.N == self.z:
			if self.d == 1:
				vertices = vertices[0:-self.S//2]
			else:
				raise NotImplementedError("Open Boundary Conditions for d = %d Not Implemented"%(self.d))

		position = self.position(vertices)[:,None]
		
		if r is None:
			R = self.R
		elif not isinstance(r,int):
			R = r
		else:
			R = [r]

		if edges is None:
			S = [1,-1]
		elif edges is True:
			S = [1]
		elif edges is False:
			S = [-1]			
		elif isinstance(edges,int):
			S = [edges]
		else:
			S = [1,-1]

		nearestneighbours = array([concatenate(
							tuple((self.site(mod(position+s*self.I,self.n)) for s in S)),axis=1)
								for r in R],dtype=self.datatype)
		if isinstance(r,int):
			nearestneighbours = nearestneighbours[0]

		if sites:
			nearestneighbours  = vstack([repeat(vertices,nearestneighbours.shape[-1],0),nearestneighbours.ravel()]).T

		return nearestneighbours


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
		iterable =  (list(i) for i in itertools.product(self.vertices,repeat=k) if conditions(i,k))
		return iterable