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
from src.utils import array,asscalar,arange,eye,rand,einsum,dot,prod
from src.utils import unique,ceil,sort,repeat,vstack,concatenate,mod,sqrt,datatype
from src.utils import inner_norm,inner_abs2,inner_real,inner_imag
from src.utils import gradient_inner_norm,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag

from src.utils import integers,floats,delim,Null,null,scalars,arrays

from src.iterables import Dict,Dictionary,getter,setter
from src.io import join,split,exists
from src.call import rm,echo
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
		
	def __atexit__(self,cleanup=None):
		'''
		Set cleanup state of class
		Args:
			cleanup (bool,str,iterable[str]): Cleanup paths and attributes of class
		'''

		cleanup = self.cleanup if cleanup is None else cleanup

		if cleanup is None:
			paths = []
		elif isinstance(cleanup,bool):
			paths = []
		elif isinstance(cleanup,str):
			paths = [getattr(self,cleanup,cleanup)]
		else:
			paths = [getattr(self,path,path) for path in cleanup]

		for path in paths:
			echo(path,execute=False,verbose=False)

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
	Hilbert space class for Operators
	Args:
		N (int): Number of spaces
		D (int): Dimension of spaces
		space (str,dict,Space): Type of Hilbert space
		system (dict,System): System attributes
		kwargs (dict): Additional keyword arguments
	'''
	def __init__(self,N,D,space=None,system=None,**kwargs):

		setter(kwargs,system,delimiter=delim,default=False)
		super().__init__(**kwargs)

		self.N = N
		self.D = D
		self.n = None
		self.g = None
		self.space = space
		self.string = None		
		self.dtype = datatype(self.dtype)
		self.system = system

		self.init()
		
		return

	def init(self):
		'''
		Setup space attributes space,string,n
		'''

		wrapper = lambda func,dtype: (lambda *args,**kwargs: asscalar(func(*args,**kwargs),dtype=dtype))

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
			'N': int,'D':int,'n':int,'g':int
		}

		if isinstance(self.space,Space):
			space = self.space.space
		elif isinstance(self.space,dict):
			space = self.space.get('space')
		else:
			space = self.space

		self.space = space

		self.funcs = funcs[self.space]
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
	Time evolution class for Operators
	Args:
		M (int): Number of time steps
		T (int): Simulation time
		tau (float): Simulation time scale
		P (int): Trotter order
		time (str,dict,Time): Type of Time evolution space
		system (dict,System): System attributes
		kwargs (dict): Additional keyword arguments
	'''
	def __init__(self,M,T=None,tau=None,P=None,time=None,system=None,**kwargs):

		setter(kwargs,system,delimiter=delim,default=False)
		super().__init__(**kwargs)

		self.M = M
		self.T = T
		self.tau = tau
		self.P = P
		self.time = time
		self.string = None				
		self.dtype = datatype(self.dtype)
		self.system = system

		self.init()
		
		return

	def init(self):
		'''
		Setup time evolution attributes
		'''

		wrapper = lambda func,dtype: (lambda *args,**kwargs: asscalar(func(*args,**kwargs),dtype=dtype))

		funcs =  {
			'linear':{
				'M': (lambda M,T,tau,P,time: round(self.T/self.tau)),
				'T': (lambda M,T,tau,P,time: self.M*self.tau),
				'tau': (lambda M,T,tau,P,time: self.T/self.M),
				'P': (lambda M,T,tau,P,time: 1),
				},
			None:{
				'M': (lambda M,T,tau,P,time: round(self.T/self.tau)),
				'T': (lambda M,T,tau,P,time: self.M*self.tau),
				'tau': (lambda M,T,tau,P,time: self.T/self.M),
				'P': (lambda M,T,tau,P,time: 1 if P is None else P),				
				},				
			}

		dtypes = {
			'M': int,'T':self.dtype,'tau':self.dtype,'P':int,
		}

		if isinstance(self.time,Time):
			time = self.time.time
		elif isinstance(self.time,dict):
			time = self.time.get('time')
		else:
			time = self.time

		self.time = time

		self.funcs = funcs[self.time]
		self.funcs = {attr: wrapper(self.funcs[attr],dtypes.get(attr)) for attr in self.funcs}
		
		self.__string__()
		self.__size__()
	
		return

	def __string__(self):
		self.string = self.time
		return

	def __size__(self):

		assert sum(var is not None for var in [self.M,self.T,self.tau,self.P]) > -1,'0 of 4 of M, T, tau, P must be non-None'

		if (self.M is None) and (self.T is None) and (self.tau is None):
			self.M = 1
			self.T = self.M
			self.tau = 1
			self.P = self.P
		elif (self.M is not None) and (self.T is None) and (self.tau is None):
			self.T = self.M
			self.tau = 1
			self.P = self.P			
		elif (self.M is None) and (self.T is not None) and (self.tau is None):
			self.M = self.T
			self.tau = 1
			self.P = self.P			
		elif (self.M is None) and (self.T is None) and (self.tau is not None):
			self.M = 1/self.tau
			self.T = 1
			self.P = self.P
		elif (self.M is not None) and (self.T is not None) and (self.tau is None):
			self.tau = self.funcs['tau'](self.T,self.M,self.tau,self.P,self.time)
		elif (self.M is not None) and (self.T is None) and (self.tau is not None):
			self.T = self.funcs['T'](self.T,self.M,self.tau,self.P,self.time)
		elif (self.M is None) and (self.T is not None) and (self.tau is not None):
			self.M = self.funcs['M'](self.T,self.M,self.tau,self.P,self.time)
		elif (self.M is not None) and (self.T is not None) and (self.tau is not None):
			pass
		
		self.M = self.funcs['M'](self.T,self.M,self.tau,self.P,self.time)
		self.T = self.funcs['T'](self.T,self.M,self.tau,self.P,self.time)
		self.tau = self.funcs['tau'](self.T,self.M,self.tau,self.P,self.time)
		self.P = self.funcs['P'](self.T,self.M,self.tau,self.P,self.time)

		return 

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return self.__str__()


class Lattice(object):
	'''
	Define a hyper lattice class
	Args:
		N (int): Lattice length along axis
		d (int): Dimension of lattice
		lattice (str,dict,Lattice): Type of lattice, allowed strings in ['square']
		kwargs (dict): Additional keyword Arguments
	'''	
	def __init__(self,N,d=None,lattice=None,**kwargs):

		# Def variables
		N = N if N is not None else 1
		d = d if d is not None else 1
		lattice = lattice if lattice is not None else None

		# Define lattice
		if isinstance(lattice,Lattice):
			lattice = lattice.lattice
		elif isinstance(lattice,dict):
			lattice = lattice.get('lattice')
		else:
			lattice = lattice

		# Define attributes
		if lattice is None:
			L = [int(N**(1/d)) for i in range(d)]
			z = 2*d
			def edge(vertex,edges=None):
				site,position = self.site,self.position
				coordinates = self.position(vertex)		
				edges = [edges] if isinstance(edges,integers) else edges if edges is not None else [1,-1]
				vertices = (site([(coordinates[j]+edge*(j==i))%(L[j]) for j in range(d)]) for i in range(d) for edge in edges)
				return vertices
			def boundary(edge):
				i,j = edge
				boundary = any(map(lambda i,j: abs(i-j)>1,self.position(i),self.position(j)))
				return boundary	
		elif lattice in ['square']:
			L = [int(N**(1/d)) for i in range(d)]
			z = 2*d
			def edge(vertex,edges=None):
				site,position = self.site,self.position
				coordinates = self.position(vertex)		
				edges = [edges] if isinstance(edges,integers) else edges if edges is not None else [1,-1]
				vertices = (site([(coordinates[j]+edge*(j==i))%(L[j]) for j in range(d)]) for i in range(d) for edge in edges)
				return vertices
			def boundary(edge):
				i,j = edge
				boundary = any(map(lambda i,j: abs(i-j)>1,self.position(i),self.position(j)))
				return boundary				
		else:
			L = [int(N**(1/d)) for i in range(d)]
			z = 2*d
			def edge(vertex,edges=None):
				site,position = self.site,self.position
				coordinates = self.position(vertex)		
				edges = [edges] if isinstance(edges,integers) else edges if edges is not None else [1,-1]
				vertices = (site([(coordinates[j]+edge*(j==i))%(L[j]) for j in range(d)]) for i in range(d) for edge in edges)
				return vertices
			def boundary(edge):
				i,j = edge
				boundary = any(map(lambda i,j: abs(i-j)>1,self.position(i),self.position(j)))
				return boundary						

		assert prod(L) == N, "Incorrect lattice size N:%d != L^d:%s"%(N,'x'.join(str(i) for i in L) if len(set(L))>1 else '%d^%d'%(sum(L)//d,d))

		self.N = N
		self.d = d
		self.lattice = lattice

		self.L = L
		self.z = z
		self.vertices = range(self.N)
		self.edge = edge
		self.boundary = boundary

		self.__size__()
		self.__shape__()
		self.__string__()

		return

	def __call__(self,vertex=None):
		'''
		Get neighbours to vertex
		Args:
			vertex (int,iterable[int],str): Vertices of neighbours, either single vertex, iterable of vertices, or allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		Returns:
			vertices (generator[int],iterable[int]): Vertices with edges to vertex
		'''

		vertices = ()

		if vertex is None:
			vertices = ((i,j) for i in self.vertices for j in self.edges(i) if i<j and not self.boundaries((i,j)))

		elif isinstance(vertex,integers):
			vertices = ((i,j) for i in [vertex] for j in self.edges(i))
		
		elif isinstance(vertex,str):
			if vertex in ['i']:
				vertices = ((i,) for i in self.vertices)
			elif vertex in ['ij']:
				vertices = ((i,j) for i in self.vertices for j in self.vertices if i!=j)
			elif vertex in ['i<j']:
				vertices = ((i,j) for i in self.vertices for j in self.vertices if i<j)
			elif vertex in ['<ij>']:
				vertices = ((i,j) for i in self.vertices for j in self.edges(i) if i<j)
			elif vertex in ['>ij<']:
				vertices = ((i,j) for i in self.vertices for j in self.edges(i) if i<j and not self.boundaries((i,j)))
			elif vertex in ['i...j']:
				vertices = ((*self.vertices,) for i in self.vertices)
		
		return vertices


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
		return str(self.string)

	def __repr__(self):
		return self.__str__()

	def __len__(self):
		return self.vertices.stop - self.vertices.start

	def __iter__(self):
		for vertex in self.vertices:
			yield vertex

	def __getitem__(self, vertex):
		return self.edges(vertex)

	def position(self,vertex):
		'''
		Return position coordinates in d-dimensional L^d lattice 
		from given linear site position in N^d length array
		Args:
			vertex (int): Linear site positions on lattice
		Returns:
			coordinates (iterable[int]): Position coordinates of linear site positions 
		'''

		coordinates = [int((vertex/self.L[i]**i)%(self.L[i])) for i in range(self.d)]

		return coordinates
	
	def site(self,coordinates):
		'''
		Return linear site position in N^d length array 
		from given position coordinates in d-dimensional L^d lattice
		Args:
			coordinates (iterable[int]): Position coordinates of linear site positions 
		Returns:
			vertex (int): Linear site positions on lattice
		'''

		vertex = sum(coordinates[i]*(self.L[i]**i) for i in range(self.d))

		return vertex

	def edges(self,vertex,edges=None):
		'''
		Edges for vertex
		Args:
			vertex (int): Linear position of vertex on lattice
			edges (int,iterable[int]): Linear offsets of edges on lattice
		Returns:
			edges (iterable[int]): Linear position of edges to vertex on lattice
		'''
		return self.edge(vertex,edges=edges)

	def boundaries(self,edge):
		'''
		Check if edge is across boundary of lattice
		Args:
			edge (iterable[int]): Edge of pair of vertices
		Returns:
			boundary (bool): Edge is across boundary
		'''
		return self.boundary(edge)