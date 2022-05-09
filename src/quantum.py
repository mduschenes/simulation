#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp
import jax
import jax.numpy as np
import jax.scipy as sp
import jax.example_libraries.optimizers
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'
# np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['',".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.optimize import Optimizer,Objective
from src.utils import jit,gradient,gradient_finite,gradient_fwd
from src.utils import array,dictionary,ones,zeros,arange,rand,identity
from src.utils import tensorprod,trace,broadcast_to
from src.utils import summation,exponentiation
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner
from src.utils import maximum,minimum,abs,real,imag,cos,sin,heaviside,sigmoid,inner,norm,interpolate,unique,allclose,parse
from src.utils import pi,e

# Logging
import logging,logging.config
logger = logging.getLogger(__name__)
conf = 'config/logging.conf'
try: 
	logging.config.fileConfig(conf,disable_existing_loggers=False) 
except:
	pass
logger = logging.getLogger(__name__)


class System(dictionary):
	'''
	System attributes (dtype,format,device,seed,verbose,...)
	Args:
		dtype (str,data-type): Data type of class
		format (str): Format of array
		device (str): Device for computation
		seed (array,int): Seed for random number generation
		verbose (bool,str): Verbosity of class	
		args (dict,System): Additional system attributes
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,*args,**kwargs):


		updates = {
			'verbose':{
				'notset':0,'debug':10,'info':20,'warning':30,'error':40,'critical':50,
				'Notset':0,'Debug':10,'Info':20,'Warning':30,'Error':40,'Critical':50,
				'NOTSET':0,'DEBUG':10,'INFO':20,'WARNING':30,'ERROR':40,'CRITICAL':50,
				10:10,20:20,30:30,40:40,50:50,
				2:20,3:30,4:40,5:50,
				True:20,False:0,None:0,
				}
			}

		defaults = {
			'dtype':'complex',
			'format':'array',
			'device':'cpu',
			'seed':None,
			'verbose':False,
		}

		args = {k:v for a in args for k,v in ({} if a is None else a).items()}
		attrs = {**args,**kwargs}
		attrs.update({attr: defaults[attr] for attr in defaults if attrs.get(attr) is None})
		attrs.update({attr: updates.get(attr,{}).get(attrs[attr],attrs[attr]) for attr in attrs})

		super().__init__(**attrs)

		return


class Space(object):
	'''
	Hilbert space class for Operators with size n
	Args:
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Length scale in system		
		space (str,Space): Type of Hilbert space
		system (dict,System): System attributes
	'''
	def __init__(self,N,D,d,L,delta,space,system):

		self.system = System(system)
		self.N = N if N is not None else 1
		self.D = D if D is not None else 2
		self.d = d if d is not None else 1
		self.L = L if L is not None else None
		self.delta = delta if delta is not None else None
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
		assert self.L is not None or self.delta is not None, "Either L or delta must not be None"		
		self.delta = self.get_delta()
		self.L = self.get_L(self.delta)
		self.n = self.get_n()
		return 

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return str(self.string)

	def get_L(self,delta):
		if self.L is None:
			if self.space in ['spin']:
				return delta*self.N
			else:
				return delta*self.N
		else:
			return self.L
		return

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

	def get_delta(self):
		if self.delta is None:
			if self.space in ['spin']:
				return self.L/self.N
			else:
				return self.L/self.N
		else:
			return self.delta
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
		self.M = M if M is not None else None
		self.T = T if T is not None else None
		self.tau = tau if tau is not None else None
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
		assert self.T is not None or self.tau is not None, "Either T or tau must not be None"
		self.tau = self.get_tau()
		self.M = self.get_M(self.tau)
		self.T = self.get_T()
		return 

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return str(self.string)

	def get_T(self):
		if self.T is None:
			if self.time in ['linear']:
				return self.tau*self.M
			else:
				return self.tau*self.M
		else:
			return self.T
		return 		

	def get_M(self,tau):
		if self.time in ['linear']:
			return int(self.T/tau)
		else:
			return int(self.T/tau)
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
		system (dict,System): System attributes (dtype,format,device,seed,verbose)		
	'''	
	def __init__(self,N,d,L=1,delta=1,lattice='square',system=None):
		

		# Define lattice
		if isinstance(lattice,Lattice):
			lattice = lattice.lattice
		else:
			lattice = lattice

		# Define parameters of system        
		self.lattice = lattice
		self.N = N
		self.d = d
		self.L = L
		self.delta = delta

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
								np.repeat(np.arange(self.N),self.z),
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
		return np.array([np.concatenate(
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
		data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			operator (iterable[str]): string names of operators
		site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		hyperparameters (dict) : class hyperparameters
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Simulation length scale		
		M (int): Number of time steps
		T (int): Simulation Time
		tau (float): Simulation time scale		
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={},
		N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,space=None,time=None,lattice=None,system=None):

		self.N = N
		self.D = D
		self.d = d
		self.L = L
		self.delta = delta
		self.M = M
		self.T = T
		self.tau = tau
		self.p = p
		self.space = space
		self.time = time
		self.lattice = lattice
		self.system = system

		self.hyperparameters = hyperparameters

		self.data = array([])
		self.operator = []
		self.site = []
		self.string = []
		self.interaction = []
		self.size = getattr(self,'size',0)
		self.shape = (self.M,self.size,)

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

		self.fig = {}
		self.ax = {}

		self.__system__()
		self.__space__()
		self.__time__()
		self.__lattice__()
		self.__setup__(data,operator,site,string,interaction,hyperparameters)
	
		return	

	def __setup__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={}):
		'''
		Setup class
		Args:
			data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
				string (iterable[str]): string labels of operators
				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				operator (iterable[str]): string names of operators		
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			hyperparameters (dict) : class hyperparameters
		'''

		if operator is None:
			operator = []
		if site is None:
			site = []
		if string is None:
			string = []
		if interaction is None:
			interaction = []									

		operator.extend([data[name]['operator'] for name in data])
		site.extend([data[name]['site'] for name in data])
		string.extend([data[name]['string'] for name in data])
		interaction.extend([data[name]['interaction'] for name in data])

		size = min([len(i) for i in [operator,site,string,interaction]])

		data = [self.identity.copy() for i in range(size)]

		self.__extend__(data,operator,site,string,interaction,hyperparameters)

		return


	def __append__(self,data,operator,site,string,interaction,hyperparameters={}):
		'''
		Append to class
		Args:
			data (array): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			hyperparameters (dict) : class hyperparameters
		'''
		index = -1
		self.__insert__(index,data,operator,site,string,interaction,hyperparameters)
		return

	def __extend__(self,data,operator,site,string,interaction,hyperparameters={}):
		'''
		Setup class
		Args:
			data (iterable[array]): data of operator
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			hyperparameters (dict) : class hyperparameters
		'''

		for _data,_operator,_site,_string,_interaction in zip(data,operator,site,string,interaction):
			self.__append__(_data,_operator,_site,_string,_interaction,hyperparameters)

		return


	def __insert__(self,index,data,operator,site,string,interaction,hyperparameters={}):
		'''
		Insert to class
		Args:
			index (int): index to insert operator
			data (array): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			hyperparameters (dict) : class hyperparameters
		'''

		if index == -1:
			index = self.size

		self.data = array([*self.data[:index],data,*self.data[index:]])
		self.operator.insert(index,operator)
		self.site.insert(index,site)
		self.string.insert(index,string)
		self.interaction.insert(index,interaction)

		self.size = len(self.data)
		self.shape = (self.M,self.size)
		self.hyperparameters.update(hyperparameters)

		return


	@partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Return parameterized operator sum(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator			
		Returns
			operator (array): Parameterized operator
		'''		
		parameters = self.__parameters__(parameters)

		return summation(parameters,self.data,self.identity)


	@partial(jit,static_argnums=(0,))
	def __parameters__(self,parameters):
		''' 
		Setup parameters
		Args:
			parameters (array): parameters
		Returns:
			parameters (array): parameters		
		'''
		self.parameters = parameters
		return parameters

	@partial(jit,static_argnums=(0,))
	def __constraints__(self,parameters):
		''' 
		Setup constraints
		Args:
			parameters (array): parameters
		Returns:
			constraint (array): constraints
		'''		
		constraint = 0

		category = 'variable'
		shape = self.hyperparameters['shapes'][category]
		parameters = parameters.reshape(shape)


		for parameter in self.hyperparameters['parameters']:
			for group in self.hyperparameters['parameters'][parameter]['group']:
				constraint += self.hyperparameters['parameters'][parameter]['constraints'][group](parameters,self.hyperparameters)
		return constraint


	@partial(jit,static_argnums=(0,))
	def __objective__(self,parameters):
		''' 
		Setup objective
		Args:
			parameters (array): parameters
		Returns:
			objective (array): objective
		'''	
		return 1-distance(self(parameters),self.hyperparameters['label'])

	@partial(jit,static_argnums=(0,))
	def __loss__(self,parameters):
		''' 
		Setup loss
		Args:
			parameters (array): parameters
		Returns:
			loss (array): loss
		'''	
		return distance(self(parameters),self.hyperparameters['label'])

	@partial(jit,static_argnums=(0,))
	def __func__(self,parameters):
		''' 
		Setup objective
		Args:
			parameters (array): parameters
		Returns:
			objective (array): objective
		'''	
		return self.__loss__(parameters) + self.__constraints__(parameters)
		# return self.__loss__(parameters)

	@partial(jit,static_argnums=(0,))
	def __grad__(self,parameters):
		''' 
		Setup gradient of objective
		Args:
			parameters (array): parameters
		Returns:
			gradient (array): gradient of objective
		'''	
		return gradient(self.__func__)(parameters)

	# @partial(jit,static_argnums=(0,))
	def __callback__(self,parameters):
		''' 
		Setup log
		Args:
			parameters (array): parameters
		'''	

		self.hyperparameters['hyperparameters']['track']['objective'].append(
			# 1-self.hyperparameters['hyperparameters']['track']['value'][-1] + self.__constraints__(parameters)
			self.__objective__(parameters)
			)

		if self.hyperparameters['hyperparameters']['track']['iteration'][-1]%self.hyperparameters['hyperparameters']['track']['track']['log'] == 0:			

			self.hyperparameters['hyperparameters']['track']['parameters'].append(copy.deepcopy(parameters))		

			self.log('%d f(x) = %0.4f'%(
				self.hyperparameters['hyperparameters']['track']['iteration'][-1],
				self.hyperparameters['hyperparameters']['track']['objective'][-1],
				)
			)
			# isclose = allclose(self.hyperparameters['hyperparameters']['track']['grad'][-1],gradient_finite(self.__func__)(parameters))
			# if not isclose:
			# 	print('**************** ERROR GRAD ****************************')
			# 	print(parameters)
			# 	print(self.hyperparameters['hyperparameters']['track']['grad'][-1])
			# 	print(self.__grad__(parameters))
			# 	isclose = allclose(self.hyperparameters['hyperparameters']['track']['grad'][-1],self.__grad__(parameters))
			# 	print(allclose(self.hyperparameters['hyperparameters']['track']['value'][-1],self.__func__(parameters)))
			# 	if not isclose:
			# 		print('ERROR with index',self.hyperparameters['hyperparameters']['track']['objective'])

			# print()
			self.log('\n'.join([
				'%s = %0.3e'%(attr,self.hyperparameters['hyperparameters']['track'][attr][-1])
				for attr in ['alpha','beta']])
			)

			# self.log('x = \n%r \ngrad(x) = \n%r'%(
			# 	self.hyperparameters['hyperparameters']['track']['parameters'][-1],
			# 	self.hyperparameters['hyperparameters']['track']['search'][-1],
			# 	)
			# )

			# self.log('U\n%r\nV\n%r'%(
			# 	self(parameters),
			# 	self.hyperparameters['label']
			# 	)
			# )
			# self.log('norm: %0.4e\nmax: %0.4e\nmin: %0.4e\nbcs:\n%r\n%r\n\n'%(
			# 	np.linalg.norm(parameters)/parameters.size,
			# 	parameters.max(),parameters.min(),
			# 	parameters.reshape(self.hyperparameters['shapes']['variable'])[0],
			# 	parameters.reshape(self.hyperparameters['shapes']['variable'])[-1],
			# 	)
			# )

		return 



	def __system__(self,system=None):
		'''
		Set system attributes
		Args:
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''
		system = self.system if system is None else system
		
		self.system = System(system)		
		self.dtype = self.system.dtype
		self.verbose = self.system.verbose

		return


	def __space__(self,N=None,D=None,d=None,L=None,delta=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			d (int): Spatial dimension
			L (int,float): Scale in system
			delta (float): Length scale in system
			space (str,Space): Type of Hilbert space
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''
		N = self.N if N is None else N
		D = self.D if D is None else D
		d = self.d if d is None else d
		L = self.L if L is None else L
		delta = self.delta if delta is None else delta
		space = self.space if space is None else space
		system = self.system if system is None else system

		self.space = Space(N,D,d,L,delta,space,system)
		self.N = self.space.N
		self.D = self.space.D
		self.d = self.space.d
		self.L = self.space.L
		self.delta = self.space.delta
		self.n = self.space.n
		self.identity = identity(self.n,dtype=self.dtype)

		return


	def __time__(self,M=None,T=None,tau=None,p=None,time=None,system=None):
		'''
		Set time attributes
		Args:
			M (int): Number of time steps
			T (int): Simulation Time
			tau (float): Simulation time scale
			p (int): Trotter order		
			time (str,Time): Type of Time evolution space						
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''
		M = self.M if M is None else M
		T = self.T if T is None else T
		tau = self.tau if tau is None else tau
		p = self.p if p is None else p
		time = self.time if time is None else time
		system = self.system if system is None else system

		self.time = Time(M,T,tau,p,time,system)		
		self.M = self.time.M
		self.T = self.time.T
		self.p = self.time.p
		self.tau = self.time.tau
		self.shape = (self.M,self.size)

		return


	def __lattice__(self,N=None,D=None,d=None,L=None,delta=None,lattice=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			d (int): Spatial dimension
			L (int,float): Scale in system
			delta (float): Length scale in system			
			lattice (str,Lattice): Type of lattice		
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''		
		N = self.N if N is None else N
		D = self.D if D is None else D
		d = self.d if d is None else d
		L = self.L if L is None else L
		delta = self.delta if delta is None else delta
		lattice = self.lattice if lattice is None else lattice
		system = self.system if system is None else system

		self.lattice = Lattice(N,d,L,delta,lattice,system)	

		return


	def __str__(self):
		size = self.size
		multiple_time = (self.M>1)
		multiple_space = [size>1 and False for i in range(size)]
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

	def dump(self,path,parameters,ext='txt'):
		'''
		Save class data		
		'''
		data = self(parameters)
		dump(data,path)
		return


	def load(self,path,ext='txt'):
		'''
		Load class data		
		'''
		if path is None:
			path = '%s.%s'%(self,ext)
		data = array(load(path,dtype=self.dtype))
		string = basename(path)
		self.append(data,string=string)
		return


	def __plot__(self,parameters,**kwargs):
		'''
		Plot Parameters
		Args:
			parameters (array): Parameters
			kwargs (dict): Plot settings
		'''


		# Get hyperparameters
		hyperparameters = self.hyperparameters
		

		# Get parameters
		category = 'variable'
		shape = hyperparameters['shapes'][category]
		parameters = parameters.reshape(shape)


		# Plot attributes

		attr = 'parameters'
		fig,ax = self.fig.get(attr),self.ax.get(attr)

		path = os.path.join(hyperparameters['sys']['directories']['dump'],hyperparameters['sys']['files']['plot'][attr])
		mplstyle = os.path.join(hyperparameters['sys']['directories']['config'],hyperparameters['sys']['files']['mplstyle'])
		size = (20,20)
		iterations = [0,*[5,10,15,20],*[i*(hyperparameters['hyperparameters']['track']['size']-1)//n for n in [4] for i in range(1,n+1)]]

		with matplotlib.style.context(mplstyle):
		
			if fig is None:
				fig,ax = plt.subplots(shape[1],1)

			for j,parameters in enumerate(hyperparameters['hyperparameters']['track']['parameters']):

				if j not in iterations:
					continue

				parameters = parameters.reshape(shape)

				# iteration = {0:0,1:hyperparameters['hyperparameters']['track']['iteration'][-1]}[j]
				iteration = j#{0:0,1:hyperparameters['hyperparameters']['track']['iteration'][-1]}[j]
				# iteration = {0:0,1:hyperparameters['hyperparameters']['track']['iteration'][-1]}[j]

				for i in range(shape[1]):
					x = np.arange(shape[0])
					y = parameters[:,i]


					# y = abs((y - hyperparameters['hyperparameters']['track']['parameters'][0].reshape(shape)[:,i])/
					# 	maximum(1,abs(hyperparameters['hyperparameters']['track']['parameters'][0].reshape(shape)[:,i])))

					# if i == 0:
					# 	scale = (1)*(2*pi/4/(20e-6))
					# 	y = scale*y
					# elif i == 1:
					# 	y = 2*pi*y

					# x = x[1:-1]
					# y = y[1:-1]

					label = [r'\alpha',r'\phi'][i%2]

					ax[i].plot(x,y,
						color=getattr(plt.cm,'winter')((iterations.index(j)+1)/len(iterations)),
						marker='',alpha=0.8,linewidth=3,
						# label=r'${%s}^{(%s)}_{%s}$'%(label,str(iteration),str(i//2) if shape[1]>2 else '')
						label=r'${%s}^{(%s)}_{%s}$'%(r'\theta',str(iteration),'')
					)

					ax[i].set_xlim(xmin=0,xmax=shape[0])
					# ax[i].set_ylim(ymin=-0.25,ymax=1.25)
					ax[i].set_ylabel(ylabel=r'${%s}_{%s}$'%(label,str(i//2) if shape[1]>2 else ''))
					ax[i].set_xlabel(xlabel=r'$\textrm{%s}$'%('Time'))
					ax[i].set_yscale(value='linear')
					# ax[i].set_yscale(value='log')
					ax[i].grid(True)	

					if i == 0:
						ax[i].legend(loc=(0.15,1.1),ncol=min(4,len(ax[i].get_legend_handles_labels()[0])))


			fig.set_size_inches(*size)
			fig.subplots_adjust()
			fig.tight_layout()
			fig.savefig(path)

		self.fig[attr] = fig
		self.ax[attr] = ax




		attr = 'objective'
		fig,ax = self.fig.get(attr),self.ax.get(attr)

		path = os.path.join(hyperparameters['sys']['directories']['dump'],hyperparameters['sys']['files']['plot'][attr])
		mplstyle = os.path.join(hyperparameters['sys']['directories']['config'],hyperparameters['sys']['files']['mplstyle'])
		size = (8,8)

		with matplotlib.style.context(mplstyle):
		
			fig,ax = plt.subplots()

			x = hyperparameters['hyperparameters']['track']['iteration']
			y = hyperparameters['hyperparameters']['track']['objective']

			ax.plot(x,y,linewidth=4,marker='o',markersize=10)


			ax.set_ylabel(ylabel=r'$\textrm{%s}$'%('Objective'))
			ax.set_xlabel(xlabel=r'$\textrm{%s}$'%('Iteration'))

			# ax.set_ylim(ymin=0,ymax=1)
			# ax.set_yscale(value='linear')

			ax.set_ylim(ymin=5e-1,ymax=1e0)
			ax.set_yscale(value='log',base=10)

			ax.yaxis.offsetText.set_fontsize(fontsize=20)

			ax.set_xticks(ticks=range(int(1*min(x)),int(1.1*max(x)),int(max(x)-min(x))//8))
			# ax.set_yticks(ticks=[1e-1,2e-1,4e-1,6e-1,8e-1,1e0])
			ax.set_yticks(ticks=[5e-1,6e-1,8e-1,1e0])
			ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
			# ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0,subs=(1.0,),numticks=100))
			ax.ticklabel_format(axis='y',style='sci',scilimits=[-1,2])	

			ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(2,10)*.1,numticks=100))
			ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


			ax.tick_params(axis='y',which='major',length=8,width=1)
			ax.tick_params(axis='y',which='minor',length=4,width=0.5)
			ax.tick_params(axis='x',which='major',length=8,width=1)
			ax.tick_params(axis='x',which='minor',length=4,width=0.5)

			ax.set_aspect(aspect='auto')
			ax.grid(visible=True,which='both',axis='both')	

			fig.set_size_inches(*size)
			fig.subplots_adjust()
			fig.tight_layout()
			fig.savefig(path)


		self.fig[attr] = fig
		self.ax[attr] = ax

		return


class Hamiltonian(Object):
	'''
	Hamiltonian class of Operators
	Args:
		data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			operator (iterable[str]): string names of operators
		operator (iterable[str]): string names of operators
		site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		hyperparameters (dict) : class hyperparameters
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Simulation length scale		
		M (int): Number of time steps
		T (int): Simulation time
		tau (float): Simulation time scale
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={},
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,hyperparameters=hyperparameters,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,p=p,space=space,time=time,lattice=lattice,system=system)
		return


	@partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Return parameterized operator sum(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator			
		Returns
			operator (array): Parameterized operator
		'''		
		parameters = self.__parameters__(parameters)

		return summation(parameters,self.data,self.identity)

	def __setup__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={}):
		'''
		Setup class
		Args:
			data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
				string (iterable[str]): string labels of operators
				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
				operator (iterable[str]): string names of operators		
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			hyperparameters (dict) : class hyperparameters
		'''

		# Get parameters
		parameters = None		

		# Get hyperparameters
		hyperparameters.update(self.hyperparameters)


		# Get operator,site,string,interaction from data
		if operator is None:
			operator = []
		if site is None:
			site = []
		if string is None:
			string = []
		if interaction is None:
			interaction = []									

		operator.extend([data[name]['operator'] for name in data])
		site.extend([data[name]['site'] for name in data])
		string.extend([data[name]['string'] for name in data])
		interaction.extend([data[name]['interaction'] for name in data])


		# Get number of operators
		size = min([len(i) for i in [operator,site,string,interaction]])

		# Lattice sites
		sites = {site: self.lattice(site) for site in ['i','i<j','<ij>','i...j']}	# sites types on lattice
		indices = ['i','j'] # allowed symbolic indices and maximum number of body site interactions

		# Basis single-site operators
		basis = {
			'I': array([[1,0],[0,1]],dtype=self.dtype),
			'X': array([[0,1],[1,0]],dtype=self.dtype),
			'Y': array([[0,-1j],[1j,0]],dtype=self.dtype),
			'Z': array([[1,0],[0,-1]],dtype=self.dtype),
		}

		# Get identity operator I, to be maintained with same shape of data for Euler identities
		# with minimal redundant copying of data
		I = 'I'

		# Get all indices from symbolic indices
		for i in range(size):
			_operator = operator.pop(0);
			_site = site.pop(0);
			_string = string.pop(0);
			_interaction = interaction.pop(0);
			if any(j in indices for j in _site):
				for s in sites[_interaction]:
					_site_ = copy.deepcopy([dict(zip(indices,s if not isinstance(s,int) else [s])).get(j,parse(j,int)) for j in _site])
					_operator_ = copy.deepcopy([_operator[_site_.index(j)] if j in _site_ else I for j in range(self.N)])
					_string_ = copy.deepcopy(_string)
					_interaction_ = copy.deepcopy(_interaction)
					
					operator.append(_operator_)
					site.append(_site_)
					string.append(_string_)
					interaction.append(_interaction_)
			else:
				_site_ = copy.deepcopy(_site)
				_operator_ = copy.deepcopy([_operator[_site_.index(j)] if j in _site_ else I for j in range(self.N)])
				_string_ = copy.deepcopy(_string)
				_interaction_ = copy.deepcopy(_interaction)

				operator.append(_operator_)
				site.append(_site_)
				string.append(_string_)
				interaction.append(_interaction_)

		# Form (size,n,n) shape operator from local strings for each data term
		data = array([tensorprod([basis[j] for j in i]) for i in operator])

		# Get size of data
		size = len(data)

		# Get Trotterized order of p copies of data for products of data
		data = trotter(data,self.p)

		# Get shape of parameters
		shape = (self.M,size)

		# Get parameters and groups based on operator strings
		for parameter in list(hyperparameters['parameters']):
			for group in list(hyperparameters['parameters'][parameter]['group']):
				if not any(g in [s,'_'.join([s,''.join(['%d'%j for j in i])])] 
						for g in group
						for i,s in zip(site,string)):
					hyperparameters['parameters'][parameter]['group'].remove(group);
			if hyperparameters['parameters'][parameter]['group'] == []:
				hyperparameters['parameters'].pop(parameter);

		# Update 
		# indices of parameters within all parameters 
		# slices of parameters within parameters of category
		# included of parameters within group of parameters of category
		# length of parameters within group of parameters of category
		categories = list(set([hyperparameters['parameters'][parameter]['category'] for parameter in hyperparameters['parameters']]))
		indices = {category:{} for category in categories}
		slices = {category:{} for category in categories}
		included = {category:{} for category in categories}

		for parameter in hyperparameters['parameters']:

			category = hyperparameters['parameters'][parameter]['category']
			locality = hyperparameters['parameters'][parameter]['locality']


			length_index = max([-1,*[slices[category][group][-1] for group in slices[category]]])+1
			for group in hyperparameters['parameters'][parameter]['group']:
				indices[category][group] = [i for i,s in enumerate(string) 
					if any(g in group for g in [s,'_'.join([s,''.join(['%d'%j for j in site[i]])])])] 
			
				length_local = len(indices[category][group]) if hyperparameters['parameters'][parameter]['locality'] in ['local'] else 1
				length_parameter = hyperparameters['parameters'][parameter]['size'] if hyperparameters['parameters'][parameter].get('size') is not None else 1
				length = length_local*length_parameter
				slices[category][group] = [length_index + i*length_parameter + j for i in range(length_local) for j in range(length_parameter)]
				included[category][group] = [length_index + i*length_parameter + j for i in range(length_local) for j in range(length_parameter)]


			hyperparameters['parameters'][parameter]['index'] = {group: [j for j in indices[category][group]] for group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['slice'] = {group: [j for j in slices[category][group]] for group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['include'] = {group: [j for j in included[category][group]] for group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['length'] =  {group: length for group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['site'] =  {group: [site[j] for j in indices[category][group]] for group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['string'] = {group: ['_'.join([string[j],''.join(['%d'%(k) for k in site[j]]),''.join(operator[j])]) for j in indices[category][group]] for group in hyperparameters['parameters'][parameter]['group']}

		# Update shape of categories
		shapes = {
			category: (shape[0],
						sum([
							len(							
							set([i 
							for group in hyperparameters['parameters'][parameter]['group']
							for i in hyperparameters['parameters'][parameter]['slice'][group]])) 
							for parameter in hyperparameters['parameters'] 
					   	   if hyperparameters['parameters'][parameter]['category'] == category]),
					   *shape[2:])
			for category in categories
		}

		print('slice, index, include, length')
		for parameter in hyperparameters['parameters']:
			for group in hyperparameters['parameters'][parameter]['group']:
				print(parameter,group,hyperparameters['parameters'][parameter]['slice'][group],hyperparameters['parameters'][parameter]['index'][group],hyperparameters['parameters'][parameter]['include'][group],hyperparameters['parameters'][parameter]['length'][group])
		print()

		# Update hyperparameters
		hyperparameters['data'] = data
		hyperparameters['operator'] = operator
		hyperparameters['site'] = site
		hyperparameters['string'] = string
		hyperparameters['interaction'] = interaction
		hyperparameters['identity'] = self.identity
		hyperparameters['size'] = size
		hyperparameters['shape'] = shape
		hyperparameters['shapes'] = shapes
		hyperparameters['N'] = self.N
		hyperparameters['M'] = self.M
		hyperparameters['D'] = self.D
		hyperparameters['d'] = self.d
		hyperparameters['n'] = self.n
		hyperparameters['p'] = self.p
		hyperparameters['coefficients'] = self.tau/self.p

		# Update class attributes
		self.__extend__(data,operator,site,string,interaction,hyperparameters)

		# Initialize parameters
		self.__init__parameters__(parameters)

		return


	def __parameters__(self,parameters):
		''' 
		Setup parameters
		Args:
			parameters (array): parameters
		Returns:
			parameters (array): parameters		
		'''		

		# Get class attributes
		self.parameters = parameters
		hyperparameters = self.hyperparameters

		# Set all parameters
		
		category = 'variable'
		shape = hyperparameters['shapes'][category]
		parameters = parameters.reshape(shape)

		value = hyperparameters['value']
		for parameter in hyperparameters['parameters']:

			if hyperparameters['parameters'][parameter]['category'] is category:
				for group in hyperparameters['parameters'][parameter]['group']:
					value = value.at[:,hyperparameters['parameters'][parameter]['index'][group]].set(
							hyperparameters['parameters'][parameter]['func'][group](parameters,hyperparameters))


		
		# Get Trotterized order of copies of parameters
		p = hyperparameters['p']
		parameters = trotter(value.T,p).T

		# Get coefficients (time step tau and trotter constants)
		coefficients = hyperparameters['coefficients']
		parameters *= coefficients

		# Get reshaped parameters
		parameters = parameters.ravel()

		
		return parameters


	def __init__parameters__(self,parameters):
		''' 
		Setup initial parameters
		Args:
			parameters (array): parameters
		Returns:
			parameters (array): parameters
		'''

		# Get class attributes
		hyperparameters = self.hyperparameters

		# Initialize parameters

		# Get shape of parameters of different category
		category = 'variable'
		shape = hyperparameters['shapes'][category]
		ndim = len(shape)

		parameters = zeros(shape)

		# Initialize parameters, and reshape, bound, impose boundary conditions accordingly, and assign category variables to parameters
		for parameter in hyperparameters['parameters']:
			
			length = max(hyperparameters['parameters'][parameter]['length'][group] for group in hyperparameters['parameters'][parameter]['group'])
			shape = (*shape[0:1],length,*shape[2:])
			ndim = len(shape)		
			bounds = hyperparameters['parameters'][parameter]['bounds']

			if hyperparameters['parameters'][parameter]['category'] == category and hyperparameters['parameters'][parameter].get('parameters') is None:

				hyperparameters['parameters'][parameter]['parameters'] = zeros(shape)
				
				for group in hyperparameters['parameters'][parameter]['group']:
					params = parameters[:,hyperparameters['parameters'][parameter]['slice'][group]]
					hyperparameters['parameters'][parameter]['parameters'] = hyperparameters['parameters'][parameter]['parameters'].at[:,
						hyperparameters['parameters'][parameter]['include'][group]].set(
						initialize(params,bounds,hyperparameters))

			if hyperparameters['parameters'][parameter]['parameters'].ndim < ndim:
				hyperparameters['parameters'][parameter]['parameters'] = broadcast_to(hyperparameters['parameters'][parameter]['parameters'],shape)

			for group in hyperparameters['parameters'][parameter]['group']:
				hyperparameters['parameters'][parameter]['parameters'] = hyperparameters['parameters'][parameter]['parameters'].at[:,
						hyperparameters['parameters'][parameter]['include'][group]].set(
						minimum(bounds[1],maximum(bounds[0],hyperparameters['parameters'][parameter]['parameters'][:,
						hyperparameters['parameters'][parameter]['include'][group]]))) 




			for i in hyperparameters['parameters'][parameter]['boundaries']:

				if hyperparameters['parameters'][parameter]['boundaries'][i] is not None and i < shape[0]:
					hyperparameters['parameters'][parameter]['parameters'] = hyperparameters['parameters'][parameter]['parameters'].at[i,:].set(
						hyperparameters['parameters'][parameter]['boundaries'][i])


			if hyperparameters['parameters'][parameter]['category'] == category:
				for group in hyperparameters['parameters'][parameter]['group']:
					parameters = parameters.at[:,hyperparameters['parameters'][parameter]['slice'][group]].set(
						hyperparameters['parameters'][parameter]['parameters'][:,hyperparameters['parameters'][parameter]['include'][group]])

		print()
		print('parameters')
		print(parameters)
		print()
		for parameter in hyperparameters['parameters']:
			print(parameter)
			print(hyperparameters['parameters'][parameter]['parameters'])


		# Get value and label
		hyperparameters['value'] = zeros(hyperparameters['shape'])
		hyperparameters['label'] = hyperparameters['label'] #.conj().T

		for parameter in hyperparameters['parameters']:
			for group in hyperparameters['parameters'][parameter]['group']:
				hyperparameters['value'] = hyperparameters['value'].at[:,hyperparameters['parameters'][parameter]['index'][group]].set(
						hyperparameters['parameters'][parameter]['func'][group](parameters,hyperparameters))

		print()
		print('value')
		print(hyperparameters['value'])

		# Get reshaped parameters
		parameters = parameters.ravel()

		# Update class attributes
		self.parameters = parameters
		self.hyperparameters = hyperparameters

		return parameters



class Unitary(Hamiltonian):
	'''
	Unitary class of Operators
	Args:
		data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			operator (iterable[str]): string names of operators
		operator (iterable[str]): string names of operators
		site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		hyperparameters (dict) : class hyperparameters
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Simulation length scale				
		M (int): Number of time steps
		T (int): Simulation Time
		tau (float): Simulation time scale		
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={},
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,hyperparameters=hyperparameters,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	@partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Return parameterized operator expm(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			operator (array): Parameterized operator
		'''		
		parameters = self.__parameters__(parameters)
		return exponentiation(-1j*parameters,self.data,self.identity)


	@partial(jit,static_argnums=(0,))
	def __derivative__(self,parameters):
		'''
		Return gradient of parameterized operator expm(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			grad (array): Gradient of parameterized operator
		'''		
		
		category = 'variable'
		shape = self.hyperparameters['shapes'][category]

		parameters = self.__parameters__(parameters)
		coefficients = self.hyperparameters['coefficients']
		p = self.p

		grad = gradient_expm(-1j*parameters,self.data,self.identity)
		grad *= -1j*coefficients
		grad = grad.reshape(shape[0],-1,*grad.shape[1:])

		if p == 1:
			grad = grad[:,:shape[1]]
		elif p == 2:
			grad = grad[:,:shape[1]][:,::1] + grad[:,-shape[1]:][:,::-1]

		grad = grad.reshape(-1,*grad.shape[2:])

		return grad

def distance(a,b):
	'''
	Calculate distance between arrays
	Args:
		a (array): Array to calculate distance
		b (array): Array to calculate distance
	Returns:
		out (array): Distance between arrays
	'''
	# return norm(a-b,axis=None,ord=2)/a.shape[0]
	return 1-abs(inner(a,b))/a.shape[0]
	# return 1-(np.real(trace((a-b).conj().T.dot(a-b))/a.size)/2 - np.imag(trace((a-b).conj().T.dot(a-b))/a.size)/2)/2
	# return 2*np.sqrt(1-np.abs(np.linalg.eigvals(a.dot(b))[0])**2)


def trotter(a,p):
	'''
	Calculate p-order trotter series of array
	Args:
		a (array): Array to compute trotter series
		p (int): Order of trotter series
	Returns:
		out (array): Trotter series of array
	'''	
	a = array([v for u in [a[::i] for i in [1,-1,1,-1][:p]] for v in u])
	return a




def initialize(parameters,bounds,hyperparameters):
	'''
	Initialize parameters
	Args:
		parameters (array): parameters array
		bounds (iterable): bounds on parameters
		hyperparameters (dict): hyperparameters for initialization
	Returns:
		out (array): initialized slice of parameters
	'''	

	# Random seed
	random = hyperparameters['hyperparameters']['random']
	seed = hyperparameters['hyperparameters']['seed']
	key = jax.random.PRNGKey(seed)
	key,subkey = jax.random.split(key)
	
	# Parameters shape and bounds
	shape = parameters.shape
	bounds = [
		bounds[0] + (bounds[1]-bounds[0])*hyperparameters['hyperparameters']['init'][0],
		bounds[1]*hyperparameters['hyperparameters']['init'][1],
	]


	# Parameters are initialized as interpolated random values between bounds
	interpolation = hyperparameters['hyperparameters']['interpolation']
	smoothness = min(shape[0],hyperparameters['hyperparameters']['smoothness'])
	shape_interp = (shape[0]//smoothness + 1 - (smoothness==1),*shape[1:])

	pts_interp = smoothness*arange(shape_interp[0])
	pts = arange(shape[0])

	parameters_interp = rand(shape_interp,key=key,bounds=bounds,random=random)
	
	parameters = interpolate(pts_interp,parameters_interp,pts,interpolation)
	
	# parameters = 0.5*ones(shape)
	# parameters = rand(shape,key=key,bounds=bounds,random=random)

	return parameters




def run(index,hyperparameters={}):

	obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	parameters = obj.parameters
	hyperparameters = hyperparameters['hyperparameters']
	
	func = obj.__func__
	callback = obj.__callback__

	optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparameters)

	parameters = optimizer(parameters)

	obj.__plot__(parameters)

	# g = gradient_fwd(obj)
	# f = gradient_finite(obj,tol=1e-7)
	# h = obj.__derivative__

	# print(allclose(g(parameters),f(parameters)))
	# print(allclose(g(parameters),h(parameters)))
	# print(allclose(f(parameters),h(parameters)))

	# grad = gradient(func)
	# finitegrad = gradient_finite(func,tol=5e-8)

	# print(allclose(grad(parameters),finitegrad(parameters)))

	return