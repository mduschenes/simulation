#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime,shutil
from copy import deepcopy as deepcopy
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
from src.utils import array,arange,eye,rand,einsum
from src.utils import unique,ceil,sort,repeat,vstack,concatenate,mod,product,sqrt,is_array,datatype
from src.utils import inner_norm,inner_abs2,inner_real,inner_imag
from src.utils import gradient_inner_norm,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag

from src.utils import itg,dbl,flt,delim,null

from src.iterables import getter,setter
from src.io import join,split,copy,rm,exists

def config(name,conf=None,**kwargs):
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

	default = 'logging.conf'
	file = kwargs.get('file')
	existing = exists(conf)

	if not existing:
		source = join(split(__file__,directory=True,abspath=True),default)
		destination = join(split(conf,directory=True,abspath=True),default,ext='tmp')
		copy(source,destination)
		conf = destination
	else:
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
				lambda config,**kwargs: '(%s)'%(','.join(['"%s"'%(kwargs.get('file')),*(config[1:-1].split(',')[1:] if not exists(kwargs.get('file')) else ['"a"'])])),
				# lambda config,**kwargs: '(%s)'%(','.join(['"%s"'%(kwargs.get('file')),*(config[1:-1].split(',')[1:])])),# if not exists(kwargs.get('file')) else ['"a"'])])),
				lambda config,**kwargs: 'stdout,file' if kwargs.get('file') else 'stdout',
				lambda config,**kwargs: 'stdout,file' if kwargs.get('file') else 'stdout',
				]

			for key,value,arg,func in zip(keys,values,args,funcs):

				for section in config:
					if config[section].get(key) == value:

						if config[section].get(arg) is None:
							continue

						if key in ['formatter'] and value in ['file']:
							if file is not None:
								path = file
								directory = os.path.abspath(os.path.dirname(os.path.abspath(path)))
								if not os.path.exists(directory):
									os.makedirs(directory)
							else:
								break

							
							kwds = {'file':path}

						elif key in ['keys','handlers'] and value in ['stdout,file']:
							kwds = {'file': file is not None}

						config[section][arg] = func(config[section][arg],**kwds)


			with open(conf, 'w') as configfile:
				config.write(configfile)
			logging.config.fileConfig(conf,disable_existing_loggers=False,defaults={'__name__':datetime.datetime.now().strftime('%d.%M.%Y.%H.%M.%S.%f')}) 	

		except Exception as exception:
			pass

		logger = logging.getLogger(name)


	if not existing:
		rm(conf)


	return logger


class Dictionary(dict):
	'''
	Dictionary subclass with dictionary elements explicitly accessible as class attributes
	Args:
		args (dict): Dictionary elements
		kwargs (dict): Dictionary elements
	'''
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.__dict__ = self
		return


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
		verbose (bool,str): Verbosity of class	
		args (dict,System): Additional system attributes
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,*args,**kwargs):

		defaults = {
			'dtype':'float',
			'format':'array',
			'device':'cpu',
			'backend':'jax',
			'architecture':None,
			'seed':None,
			'key':None,
			'timestamp':datetime.datetime.now().strftime('%d.%M.%Y.%H.%M.%S.%f'),
			'cwd':None,
			'path':None,
			'conf':None,
			'logger':None,
			'cleanup':None,
			'verbose':None,
		}

		setter(kwargs,defaults,delimiter=delim,func=False)

		super().__init__(**kwargs)

		self.__logger__()
		self.__clean__()

		return

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
			conf = join(self.conf,root=root)
			file = join(self.logger,root=root)
			cleanup = self.cleanup

			self.logger = Logger(name,conf,file=file,cleanup=cleanup)

		return

	def log(self,msg,verbose=None):
		'''
		Log messages
		Args:
			msg (str): Message to log
			verbose (int,str): Verbosity of message			
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


class Logger(object):
	def __init__(self,name,conf,file=None,cleanup=None,verbose=True,**kwargs):
		'''
		Logger class
		Args:
			name (str,logger): Name of logger or Python logging logger
			conf (str): Path to configuration
			file (str): Path to log file
			cleanup (bool): Cleanup log files upon exit
			verbose (int,str,bool): Verbosity
			kwargs (dict): Additional keyword arguments
		'''

		if isinstance(name,str):
			try:
				self.logger = config(name,conf=conf,file=file,**kwargs)
			except Exception as exception:
				self.logger = logging.getLogger(name)
		else:
			self.logger = name

		self.name = name
		self.conf = conf
		self.file = file

		self.cleanup = cleanup
		self.__clean__()

		self.verbosity = {
			'notset':0,'debug':10,'info':20,'warning':30,'error':40,'critical':50,
			'Notset':0,'Debug':10,'Info':20,'Warning':30,'Error':40,'Critical':50,
			'NOTSET':0,'DEBUG':10,'INFO':20,'WARNING':30,'ERROR':40,'CRITICAL':50,
			10:10,20:20,30:30,40:40,50:50,
			2:20,3:30,4:40,5:50,
			-1:50,
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

	def __clean__(self,cleanup=None):
		'''
		Set cleanup state of class
		Args:
			cleanup (bool): Cleanup log files upon exit	
		'''

		cleanup = self.cleanup if cleanup is None else cleanup

		if cleanup:
			atexit.register(self.__atexit__)
		else:
			atexit.unregister(self.__atexit__)

		return


	def __atexit__(self):
		'''
		Cleanup log files upon class exit
		'''

		loggers = [logging.getLogger(),self.logger,*logging.Logger.manager.loggerDict.values()]
		loggers = [handler.baseFilename for logger in loggers for handler in getattr(logger,'handlers',[]) if isinstance(handler,logging.FileHandler)]
		loggers = list(set(loggers))

		for logger in loggers:
			rm(logger)

		return

	def __str__(self):
		return str(self.file)

	def __repr__(self):
		return self.__str__()


class Object(System):
	def __init__(self,data,shape,size=None,dims=None,system=None,**kwargs):
		'''
		Initialize data of attribute based on shape, with highest priority of arguments of: kwargs,args,data,system
		Args:
			data (dict,str,array,System): Data corresponding to class
			shape (int,iterable[int]): Shape of each data
			size (int,iterable[int]): Number of data
			dims (iterable[int]): Dimensions of N, D-dimensional sites [N,D]
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional keyword arguments
		'''
		defaults = {
			'string':None,
			'category':None,
			"scale":1,
			"samples":None,
			'initialization':'random',
			'random':'random',
			'seed':None,
			'bounds':[-1,1],
		}

		# Setup kwargs
		setter(kwargs,dict(data=data,shape=shape,size=size,dims=dims,system=system),delimiter=delim,func=False)
		setter(kwargs,data,delimiter=delim,func=False)
		setter(kwargs,system,delimiter=delim,func=False)
		setter(kwargs,defaults,delimiter=delim,func=False)
		super().__init__(**kwargs)

		# Ensure shape is iterable
		if isinstance(self.shape,int):
			self.shape = (self.shape,)

		# Ensure size is iterable
		if isinstance(self.size,int):
			self.size = (self.size,)

		# Dimension of data
		self.ndim = len(self.shape) if self.shape is not None else None
		self.length = len(self.size) if self.size is not None else None
		self.n = min(self.shape)  if self.shape is not None else None

		# Number of sites and dimension of sites
		self.N,self.D = self.dims[:2] if self.dims is not None else [1,self.n]

		# Set data
		if self.shape is None or self.scale is None:
			self.data = None
			self.size = None
		
		if is_array(self.data):
			self.data = self.data
			self.size = None
		elif self.data is None:
			self.data = self.data
			self.shape = None
			self.ndim = None
			self.size = None
			self.string = None
			self.scale = None
		else:
			if isinstance(self.data,str):
				self.string = self.data
			self.__setup__(**kwargs)

		if self.data is not None:
			self.data = self.data.astype(dtype=self.dtype)

		# Set samples
		if self.size is not None:
			if not is_array(self.samples):
				self.samples = rand(self.size,bounds=[0,1],seed=self.seed,dtype=datatype(self.dtype))
				self.samples /= self.samples.sum()
		else:
			self.samples = None

		if self.samples is not None:
			if (self.data.ndim>=self.length) and all(self.data.shape[i] == self.size[i] for i in range(self.length)):
				self.data = einsum('%s...,%s->...'%((''.join(['i','j','k','l'][:self.length]),)*2),self.data,self.samples)

		self.data = self(self.data)

		return


	def __call__(self,data=null()):
		'''
		Class data
		Args:
			data (array): Data
		Returns:
			data (array): Data
		'''
		if not isinstance(data,null):
			self.data = data
			self.shape = self.data.shape if self.data is not None else None
			self.ndim = self.data.ndim if self.data is not None else None
		return self.data

	def __setup__(self,**kwargs):
		'''
		Setup attribute
		Args:
			kwargs (dict): Additional keyword arguments
		'''

		return


	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''
		msg = '\n'.join(['%s : %s'%(attr,self[attr]) for attr in self])
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

		setter(kwargs,system,delimiter=delim,func=False)
		super().__init__(**kwargs)

		self.N = N if N is not None else 1
		self.D = D if D is not None else 2
		self.space = space		
		self.default = 'spin'
		self.system = system

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

		setter(kwargs,system,delimiter=delim,func=False)
		super().__init__(**kwargs)

		self.M = M if M is not None else 1
		self.T = T if T is not None else None
		self.tau = tau if tau is not None or T is not None else None
		self.P = P if P is not None else 1
		self.time = time
		self.default = 'linear'
		self.system = system

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
		setter(kwargs,system,delimiter=delim,func=False)
		super().__init__(**kwargs)

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
		self.delta = delta if delta is not None else self.L/self.N
		self.system = system

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