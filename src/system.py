#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime,shutil
from copy import deepcopy as deepcopy
from time import time as timer
from functools import partial

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp
import jax
import jax.numpy as np
import jax.scipy as sp
import jax.example_libraries.optimizers
import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'
np.set_printoptions(linewidth=1000)#,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})

# Logging
import logging
import logging.config,configparser

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,gradient
from src.utils import array,dictionary,arange,eye
from src.utils import unique,ceil,sort,repeat,vstack,concatenate,mod,product
from src.utils import normed,inner_abs2,inner_real,inner_imag
from src.utils import gradient_normed,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
from src.utils import normed_einsum,inner_abs2_einsum,inner_real_einsum,inner_imag_einsum
from src.utils import gradient_normed_einsum,gradient_inner_abs2_einsum,gradient_inner_real_einsum,gradient_inner_imag_einsum

from src.utils import itg,dbl,flt

from src.io import join,split,copy,rmdir,exists

def logconfig(name,conf=None,**kwargs):
	'''
	Configure logging
	Args:
		name (str): File name for logger
		conf (str): Path for logging config
		kwargs (dict): Additional keywork arguments to overwrite config
	Returns:
		logger (logger): Configured logger
	'''

	logger = logging.getLogger(name)

	existing = exists(conf)

	if not existing:
		default = 'logging.conf'
		source = join(split(__file__,directory=True,abspath=True),default)
		destination = split(conf,directory=True,abspath=True)
		copy(source,destination)

	source = conf
	destination = join(conf,ext='tmp')
	copy(source,destination)

	conf = join(conf,ext='tmp')

	if conf is not None:
		try:
			config = configparser.ConfigParser()
			config.read(conf)
		
			keys = ['formatter','keys','handlers']
			values = ['file','stdout,file','stdout,file']
			args = ['args','keys','handlers']
			funcs = [
				lambda config,**kwargs: '(%s)'%(','.join(['"%s"'%(kwargs.get('file','')),*config[1:-1].split(',')[1:]])),
				lambda config,**kwargs: 'stdout,file' if kwargs.get('file') else 'stdout',
				lambda config,**kwargs: 'stdout,file' if kwargs.get('file') else 'stdout',
				]

			for key,value,arg,func in zip(keys,values,args,funcs):

				for section in config:
					if config[section].get(key) == value:

						if config[section].get(arg) is None:
							continue

						if key in ['formatter'] and value in ['file']:
							kwarg = 'file'
							if kwargs.get(kwarg) is not None:
								file = kwargs[kwarg]
							else:
								file = str(config[section][arg][1:-1].split(',')[0])[1:-1]
							directory = os.path.abspath(os.path.dirname(os.path.abspath(file)))
							if not os.path.exists(directory):
								os.makedirs(directory)

							kwds = {kwarg:file}

						elif key in ['keys','handlers'] and value in ['stdout,file']:
							kwarg = 'file'
							kwds = {kwarg: kwargs.get(kwarg) is not None}

						config[section][arg] = func(config[section][arg],**kwds)


			with open(conf, 'w') as configfile:
			    config.write(configfile)
			logging.config.fileConfig(conf,disable_existing_loggers=False) 	

		except Exception as exception:
			pass

		logger = logging.getLogger(name)


	rmdir(conf)

	if not existing:
		conf = split(conf,file=True)
		rmdir(conf)

	return logger


class System(dictionary):
	'''
	System attributes (dtype,format,device,seed,verbose,...)
	Args:
		dtype (str,data-type): Data type of class
		format (str): Format of array
		device (str): Device for computation
		seed (array,int): Seed for random number generation
		key (object): key for class
		timestamp (str): timestamp for class
		architecture (str): architecture for class
		verbose (bool,str): Verbosity of class	
		args (dict,System): Additional system attributes
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,*args,**kwargs):

		updates = {
			}

		defaults = {
			'dtype':'complex',
			'format':'array',
			'device':'cpu',
			'seed':None,
			'key':None,
			'timestamp':None,
			'architecture':None,
			'verbose':False,
			'logger':None,
		}



		args = {k:v for a in args for k,v in ({} if a is None else a).items()}
		attrs = {**args,**kwargs}
		attrs.update({attr: defaults[attr] for attr in defaults if attrs.get(attr) is None})

		attrs.update({attr: updates.get(attr,{}).get(attrs[attr],attrs[attr]) if attr in updates else attrs[attr] for attr in attrs})

		super().__init__(**attrs)

		return



class Logger(object):
	'''
	Logger class
	Args:
		name (str,logger): Name of logger or Pythong logging logger
		conf (str): Path to configuration
		verbose (int,str,bool): Verbosity
		kwargs (dict): Additional arguments
	'''
	def __init__(self,name,conf,verbose=True,**kwargs):
		defaults = {
			'file':kwargs.get('file')
		}
		kwargs.update({kwarg: defaults[kwarg] for kwarg in defaults if kwarg not in kwargs and defaults[kwarg] is not None})

		if isinstance(name,str):
			try:
				self.logger =  logconfig(name,conf=conf,**kwargs)
			except:
				self.logger = logging.getLogger(name)
		else:
			self.logger = name

		self.name = name
		self.conf = conf

		self.verbosity = {
			'notset':0,'debug':10,'info':20,'warning':30,'error':40,'critical':50,
			'Notset':0,'Debug':10,'Info':20,'Warning':30,'Error':40,'Critical':50,
			'NOTSET':0,'DEBUG':10,'INFO':20,'WARNING':30,'ERROR':40,'CRITICAL':50,
			10:10,20:20,30:30,40:40,50:50,
			2:20,3:30,4:40,5:50,
			True:20,False:0,None:0,
			}
		self.verbose = self.verbosity.get(verbose,verbose)

		return
	
	def log(self,verbose,msg):
		'''
		Log messages
		Args:
			verbose (int): Verbosity of message
			msg (str): Message to log
		'''

		verbose = self.verbosity.get(verbose,self.verbose)

		self.logger.log(verbose,msg)
		return



class Space(object):
	'''
	Hilbert space class for Operators with size n
	Args:
		N (int): Number of qudits
		D (int): Dimension of qudits
		space (str,Space): Type of Hilbert space
		system (dict,System): System attributes
	'''
	def __init__(self,N,D,space,system):

		self.system = System(system)
		self.N = N if N is not None else 1
		self.D = D if D is not None else 2
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
		self.g = self.get_g()
		self.size = self.n
		return 

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return str(self.string)

	def get_N(self):
		if self.space in ['spin']:
			try:
				return int(log(self.n)/log(self.D))
			except:
				return 1
		else:
			try:
				return int(log(self.n)/log(self.D))
			except:
				return 1			
		return 		

	def get_D(self):
		if self.space in ['spin']:
			return int(self.n**(1/self.N))
		else:
			return int(n**(1/self.N))
		return

	def get_n(self):
		if self.space in ['spin']:
			return self.D**self.N
		else:
			return self.D**self.N
		return

	def get_g(self):
		if self.space in ['spin']:
			return self.get_n()**2-1
		else:
			return self.get_n()**2-1
		return			

class Time(object):
	'''
	Time evolution class for Operators with size n
	Args:
		M (int): Number of time steps
		T (int): Simulation time
		tau (float): Simulation time scale
		p (int): Trotter order
		time (str,Time): Type of Time evolution space
		system (dict,System): System attributes
	'''
	def __init__(self,M,T,tau,p,time,system):

		self.system = System(system)
		self.M = M if M is not None else 1
		self.T = T if T is not None else None
		self.tau = tau if tau is not None or T is not None else None
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
		assert self.T is not None or self.tau is not None, 'Either T or tau must not be None'
		self.tau = self.get_tau()
		try:
			self.T = self.get_T(self.tau)
			self.M = self.get_M()			
		except:
			self.M = self.get_M(self.tau)
			self.T = self.get_T()
		return 

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return str(self.string)

	def get_T(self,tau=None):
		if tau is None:
			tau = self.tau
		if self.T is None:
			if self.time in ['linear']:
				return tau*self.M
			else:
				return tau*self.M
		else:
			return self.T
		return 		

	def get_M(self,tau=None):
		if tau is None:
			tau = self.tau		
		if self.time in ['linear']:
			return int(round(self.T/tau))
		else:
			return int(round(self.T/tau))
		return

	def get_tau(self):
		if self.tau is None:
			if self.time in ['linear']:
				return self.T/self.M
			else:
				return self.T/self.M
		else:
			return self.tau
		return	

class Lattice(object):
	'''
	Define a hyper lattice class
	Args:
		N (int): Lattice length along axis
		d (int): Dimension of lattice
		L (int,float): Scale in system
		delta (float): Length scale in system	
		lattice (str,Lattice): Type of lattice, allowed strings in ['square','square-nearest']
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)		
	'''	
	def __init__(self,N,d,L=None,delta=None,lattice='square',system=None):
		

		# Define lattice
		if isinstance(lattice,Lattice):
			lattice = lattice.lattice
		else:
			lattice = lattice

		# Define parameters of system        
		self.lattice = lattice
		self.N = N
		self.d = d
		self.L = L if L is not None else self.N
		self.delta = delta if delta is not None else self.N//self.L

		# Define system
		self.system = System(system)
		self.dtype = self.system.dtype

		# Check system
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
			assert n**d == N, 'N != n^d for N=%d, d=%d, n=%d'%(N,d,n)
		else:
			n = int(N**(1/d))
			z = 2*d
			assert n**d == N, 'N != n^d for N=%d, d=%d, n=%d'%(N,d,n)

		self.n = n
		self.z = z


		# Define attributes
		self.__size__()
		self.__shape__()
		self.__string__()

		# Define array of vertices
		self.vertices = arange(self.N)
		
		# n^i for i = 1:d array
		self.n_i = self.n**arange(self.d,dtype=self.dtype)
		
		# Arrays for finding coordinate and linear position in d dimensions
		self.I = eye(self.d)
		self.R = arange(1,max(2,ceil(self.n/2)),dtype=self.dtype)

		return

	def __call__(self,site=None):
		'''
		Get list of lists of sites of lattice
		Args:
			site (str,int): Type of sites, either int for unique site-length list of vertices, or string in allowed ['i','i,j','i<j']
		Returns:
			sites (list): List of site-length lists of lattice
		'''

		# Unique site-length lists if site is int
		if isinstance(site,(int,itg)):
			k = site
			conditions = None
			sites = self.iterable(k,conditions)
		elif isinstance(site,(str)):
			if site in ['i']:
				sites = [[i] for i in self.vertices]
			elif site in ['i,j']:
				sites = [[i,j] for i in self.vertices for j in self.vertices]
			elif site in ['i<j']:
				k = 2
				conditions = lambda i,k: all([i[j]<i[j+1] for j in range(k-1)])	
				sites = self.iterable(k,conditions)
			elif site in ['<ij>']:
				if self.z > self.N:
					sites = []
				elif self.z > 0:
					sites = [i for i in unique(
						sort(
							vstack([
								repeat(arange(self.N),self.z,0),
								self.nearestneighbours(r=1)[0].ravel()
							]),
						axis=0),
						axis=1).T]
				else:
					sites = []

			elif site in ['i...j']:
				sites = [range(self.N) for i in range(self.N)]
		else:
			k = 2
			conditions = None
			sites = self.iterable(k,conditions)
		sites = [list(map(int,i)) for i in sites]
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
		isint = isinstance(site,(int,itg))

		if isint:
			site = array([site])
		position = mod(((site[:,None]/self.n_i)).
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
			position = array([position])
		
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
		return array([concatenate(
							(self.site(mod(sitepos+R*self.I,self.n)),
							 self.site(mod(sitepos-R*self.I,self.n))),axis=1)
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


class Metric(object):
	'''
	Metric class for distance between Operators
	Args:
		metric (str,Metric): Type of metric
		shapes (iterable[tuple[int]]): Shapes of Operators
		optimize (bool,str,iterable): Contraction type		
		system (dict,System): System attributes
	'''
	def __init__(self,metric,shapes,optimize=None,system=None):

		self.system = System(system)
		self.metric = metric
		self.shapes = shapes
		self.optimize = optimize
		self.default = None

		self.__setup__()
		
		return

	def __setup__(self):
		'''
		Setup metric attributes metric,string
		'''
		if isinstance(self.metric,Metric):
			self.metric = self.metric.metric
		if self.metric is None:
			self.metric = self.default
		self.__string__()
		self.__size__()

		self.get_metric()

		return

	def __string__(self):
		self.string = str(self.metric)
		return

	def __size__(self):
		self.size = sum(int(product(shape)**(1/len(shape))) for shape in self.shapes)//len(self.shapes)
		return 

	@partial(jit,static_argnums=(0,))
	def __call__(self,a,b):
		return self.func(a,b)

	@partial(jit,static_argnums=(0,))
	def __grad__(self,a,b,da):
		return self.grad(a,b,da)		

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return str(self.string)

	def get_metric(self):

		if callable(self.metric):
			func = jit(self.metric)
			grad = jit(gradient(self.metric))
		elif self.metric is None:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(normed_einsum(*shapes,optimize=optimize))
			# _func = normed

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_normed_einsum(*shapes,optimize=optimize))
			# _grad = gradient_normed

			@jit
			def func(a,b):
				return _func(a,b)
			@jit
			def grad(a,b,da):
				return _grad(a,b,da)		
		elif self.metric in ['norm','normed']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(normed_einsum(*shapes,optimize=optimize))
			# _func = normed

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_normed_einsum(*shapes,optimize=optimize))
			# _grad = gradient_normed

			@jit
			def func(a,b):
				return _func(a,b)
			@jit
			def grad(a,b,da):
				return _grad(a,b,da)
		elif self.metric in ['infidelity']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(inner_abs2_einsum(*shapes,optimize=optimize))
			# _func = inner_abs2

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_inner_abs2_einsum(*shapes,optimize=optimize))
			# _grad = gradient_inner_abs2

			@jit
			def func(a,b):
				return 1-_func(a,b)
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)
		elif self.metric in ['infidelity.abs']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(inner_abs2_einsum(*shapes,optimize=optimize))
			# _func = inner_abs2

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_inner_abs2_einsum(*shapes,optimize=optimize))
			# _grad = gradient_inner_abs2

			@jit
			def func(a,b):
				return 1-_func(a,b)
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)	
		elif self.metric in ['infidelity.real']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(inner_real_einsum(*shapes,optimize=optimize))
			# _func = inner_real

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_inner_real_einsum(*shapes,optimize=optimize))
			# _grad = gradient_inner_real

			@jit
			def func(a,b):
				return 1-_func(a,b)
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)
		elif self.metric in ['infidelity.imag']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(inner_imag_einsum(*shapes,optimize=optimize))
			# _func = inner_imag

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_inner_imag_einsum(*shapes,optimize=optimize))
			# _grad = gradient_inner_imag

			@jit
			def func(a,b):
				return 1-_func(a,b)
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)				
		elif self.metric in ['infidelity.real.imag']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func_real = jit(inner_real_einsum(*shapes,optimize=optimize))
			# _func_real = inner_real

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad_real = jit(gradient_inner_real_einsum(*shapes,optimize=optimize))
			# _grad_real = gradient_inner_real

			shapes = (*self.shapes,)
			optimize = self.optimize
			_func_imag = jit(inner_imag_einsum(*shapes,optimize=optimize))
			# _func_imag = inner_imag

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad_imag = jit(gradient_inner_imag_einsum(*shapes,optimize=optimize))
			# _grad_imag = gradient_inner_imag

			@jit
			def func(a,b):
				return 1-(_func_real(a,b)+_func_imag(a,b))/2
			@jit
			def grad(a,b,da):
				return -(_grad_real(a,b)+_grad_imag(a,b))/2
		elif self.metric in ['infidelity.vector']:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(inner_vectorabs2_einsum(*shapes,optimize=optimize))
			# _func = inner_abs2

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_inner_vectorabs2_einsum(*shapes,optimize=optimize))
			# _grad = gradient_inner_abs2

			@jit
			def func(a,b):
				return 1-_func(a,b)
			@jit
			def grad(a,b,da):
				return -_grad(a,b,da)					
		else:
			shapes = (*self.shapes,)
			optimize = self.optimize
			_func = jit(normed_einsum(*shapes,optimize=optimize))
			# _func = normed

			shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
			optimize = self.optimize
			_grad = jit(gradient_normed_einsum(*shapes,optimize=optimize))
			# _grad = gradient_normed

			@jit
			def func(a,b):
				return _func(a,b)
			@jit
			def grad(a,b,da):
				return _grad(a,b,da)

		self.func = func

		self.grad = grad

		return