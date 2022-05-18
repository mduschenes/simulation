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

# Logging
import logging
logger = logging.getLogger(__name__)

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['',".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.optimize import Optimizer,Objective
from src.utils import jit,gradient,gradient_finite,gradient_fwd
from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,trace,broadcast_to,expand_dims,moveaxis,repeat,take,inner,outer
from src.utils import summation,exponentiation
from src.utils import inner_abs2,inner_real2,inner_imag2
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_abs2,gradient_inner_real2,gradient_inner_imag2
from src.utils import eigh,qr
from src.utils import maximum,minimum,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,isclose,isnaninf
from src.utils import parse,to_str,to_number,datatype,_len_,_iter_
from src.utils import pi,e
from src.utils import itg,flt,dbl

from src.dictionary import _update

from src.io import load,dump,path_join,path_split


class System(dictionary):
	'''
	System attributes (dtype,format,device,seed,verbose,...)
	Args:
		dtype (str,data-type): Data type of class
		format (str): Format of array
		device (str): Device for computation
		seed (array,int): Seed for random number generation
		key (object): key for class
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
			'key':None,
			'verbose':False,
		}

		args = {k:v for a in args for k,v in ({} if a is None else a).items()}
		attrs = {**args,**kwargs}
		attrs.update({attr: defaults[attr] for attr in defaults if attrs.get(attr) is None})

		attrs.update({attr: updates.get(attr,{}).get(attrs[attr],attrs[attr]) if attr in updates else attrs[attr] for attr in attrs})

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
			return round(self.T/tau)
		else:
			return round(self.T/tau)
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
		system (dict,System): System attributes (dtype,format,device,seed,key,verbose)		
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
		self.vertices = arange(self.N)
		
		# n^i for i = 1:d array
		self.n_i = self.n**arange(self.d,dtype=self.dtype)
		
		# Arrays for finding coordinate and linear position in d dimensions
		self.I = eye(self.d)
		self.R = arange(1,max(2,ceil(self.n/2)),dtype=self.dtype)

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
		if isinstance(site,(int,itg)):
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
						sort(
							vstack([
								repeat(arange(self.N),self.z,0),
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
		system (dict,System): System attributes (dtype,format,device,seed,key,verbose)
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
		self.size = 0
		self.shape = (self.M,*self.data.shape)

		self.delimiter = ' '
		self.basis = None
		self.diagonal = []
		self._data = []
		self.funcs = lambda parameters: None
		self.expms = lambda parameters: None
		self.transform = []
		self.transformH = []
		self.index = arange(self.size)

		self.parameters = None
		self.coefficients = 1
		self.dim = 0

		self.fig = {}
		self.ax = {}

		self.__system__()
		self.__space__()
		self.__time__()
		self.__lattice__()

		self.__setup__(data,operator,site,string,interaction,hyperparameters)

		self.log('%s\n'%('\n'.join(['%s: %r'%(attr,getattr(self,attr)) for attr in ['key','N','D','d','M','tau','p','seed']])))
	
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

		self.data = array([*self.data[:index],data,*self.data[index:]],dtype=self.dtype)
		self.operator.insert(index,operator)
		self.site.insert(index,site)
		self.string.insert(index,string)
		self.interaction.insert(index,interaction)

		self.size = len(self.data)
		self.shape = (self.shape[0],*self.data.shape)
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
			variables (array): variables
		'''
		self.parameters = parameters
		variables = parameters
		return variables

	@partial(jit,static_argnums=(0,))
	def __features__(self,parameters):
		''' 
		Setup features
		Args:
			parameters (array): parameters
		Returns:
			features (array): features
		'''
		features = parameters
		return features		

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
				constraint += self.hyperparameters['parameters'][parameter]['constraints'][group](parameters)
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

	@partial(jit,static_argnums=(0,))
	def __grad__(self,parameters):
		''' 
		Setup gradient of objective
		Args:
			parameters (array): parameters
		Returns:
			grad (array): gradient of objective
		'''	

		shape = parameters.shape
		grad = zeros(shape)

		grad = grad + gradient_distance(self(parameters),self.hyperparameters['label'],self.__derivative__(parameters))


		# category = 'variable'
		# shape = self.hyperparameters['shapes'][category]
		# parameters = parameters.reshape(shape)

		# for parameter in self.hyperparameters['parameters']:
		# 	for group in self.hyperparameters['parameters'][parameter]['group']:
		# 		grad = grad + self.hyperparameters['parameters'][parameter]['gradient_constraints'][group](parameters)

		# grad = grad.ravel()

		return grad
		# return gradient(self.__func__)(parameters)

	def __callback__(self,parameters):
		''' 
		Setup callback and logging
		Args:
			parameters (array): parameters
		Returns:
			status (int): status of class
		'''	


		self.hyperparameters['hyperparameters']['track']['objective'].append(
			# 1-self.hyperparameters['hyperparameters']['track']['value'][-1] + self.__constraints__(parameters)
			self.__objective__(parameters)
			)

		if self.hyperparameters['hyperparameters']['track']['iteration'][-1]%self.hyperparameters['hyperparameters']['track']['track']['log'] == 0:			

			self.hyperparameters['hyperparameters']['track']['parameters'].append(copy.deepcopy(parameters))		

			self.log('%d f(x) = %0.10f'%(
				self.hyperparameters['hyperparameters']['track']['iteration'][-1],
				self.hyperparameters['hyperparameters']['track']['objective'][-1],
				)
			)

			self.log('|x| = %0.4e\t\t|grad(x)| = %0.4e'%(
				norm(self.hyperparameters['hyperparameters']['track']['parameters'][-1])/self.hyperparameters['hyperparameters']['track']['parameters'][-1].size,
				norm(self.hyperparameters['hyperparameters']['track']['grad'][-1])/self.hyperparameters['hyperparameters']['track']['grad'][-1].size,
				)
			)

			self.log('\t\t'.join([
				'%s = %0.4e'%(attr,self.hyperparameters['hyperparameters']['track'][attr][-1])
				for attr in ['alpha','beta']])
			)

			# self.log('x = \n%r \ngrad(x) = \n%r'%(
			# 	self.hyperparameters['hyperparameters']['track']['parameters'][-1],
			# 	self.hyperparameters['hyperparameters']['track']['search'][-1],
			# 	)
			# )

			self.log('U\n%s\nV\n%s\n'%(
				to_str(abs(self(parameters)).round(4)),
				to_str(abs(self.hyperparameters['label']).round(4))
				)
			)
			# self.log('norm: %0.4e\nmax: %0.4e\nmin: %0.4e\nbcs:\n%r\n%r\n\n'%(
			# 	norm(parameters)/parameters.size,
			# 	parameters.max(),parameters.min(),
			# 	parameters.reshape(self.hyperparameters['shapes']['variable'])[0],
			# 	parameters.reshape(self.hyperparameters['shapes']['variable'])[-1],
			# 	)
			# )


		status = (
			(abs(self.hyperparameters['hyperparameters']['track']['objective'][-1] - self.hyperparameters['hyperparameters']['value']) > 
				 self.hyperparameters['hyperparameters']['eps']*self.hyperparameters['hyperparameters']['value']) and
			(norm(self.hyperparameters['hyperparameters']['track']['grad'][-1])/self.hyperparameters['hyperparameters']['track']['grad'][-1].size > 
				  self.hyperparameters['hyperparameters']['tol'])
			)

		# self.log('status = %d\n'%(status))

		return status



	def __system__(self,system=None):
		'''
		Set system attributes
		Args:
			system (dict,System): System attributes (dtype,format,device,seed,key,verbose)		
		'''
		system = self.system if system is None else system
		
		self.system = System(system)		
		self.dtype = self.system.dtype
		self._dtype = datatype(self.dtype)
		self.format = self.system.format
		self.seed = self.system.seed
		self.key = self.system.key
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
			system (dict,System): System attributes (dtype,format,device,seed,key,verbose)		
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
			system (dict,System): System attributes (dtype,format,device,seed,key,verbose)		
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
		self.coefficients = self.tau/self.p
		self.shape = (self.shape[0],*self.data.shape)

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
			system (dict,System): System attributes (dtype,format,device,seed,key,verbose)		
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
			path = path_join(self,ext=ext)
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

		# Get parameters shape and indices of features
		category = 'variable'
		shape = hyperparameters['shape'][category]
		indices = hyperparameters['index'][category]
		size = hyperparameters['hyperparameters']['track']['size']

		# Get plot config
		attr = 'mplstyle'
		mplstyle = hyperparameters['sys']['path']['config'][attr]


		# Plot attributes
		attr = 'size'
		value = hyperparameters['hyperparameters']['track'][attr]
		if value == 0:
			return


		attr = 'parameters'
		fig,ax = self.fig.get(attr),self.ax.get(attr)
		
		path = hyperparameters['sys']['path']['plot'][attr]

		layout = [shape[1],1]
		plots = [None]*layout[0]
		layout = [shape[1],2]		
		plots = [[None]*layout[1]]*layout[0]
		figsize = (20,20)
		iterations = list(sorted(list(set([min(size-1,i) 
							for i in [
							0,
							*[5,10,15,20],
							*[i*(size-1)//n 
							for n in [4] for i in range(1,n+1)]]
							]))))
		labels = [r'\alpha',r'\phi']
		lims = [[-0.25,1.25],[-1.25,1.25]]
		lims = [[None,None],[None,None]]

		j = 0
		parameters0 = self.__features__(hyperparameters['hyperparameters']['track'][attr][j])[indices]

		with matplotlib.style.context(mplstyle):
		
			if fig is None:
				fig,ax = plt.subplots(*layout)
				if layout[0] == 1 and layout[1] == 1:
					ax = [[ax]]
				elif layout[0] == 1:
					ax = [ax]
				elif layout[1] == 1:
					ax = [[a] for a in ax]
			elif ax is None:
				ax = fig.gca()

			for j in iterations:

				parameters = self.__features__(hyperparameters['hyperparameters']['track'][attr][j])[indices]

				for i in range(layout[0]):
					for k in range(layout[1]):
						x = arange(shape[0])
						y = parameters[:,i]

						if layout[1] > 1:
							if k == 0:
								y = parameters[:,i]
							elif k == 1:
								y0 = parameters0[:,i]
								y = abs((y - y0)/(maximum(y,y0)+1e-20))	
						else:
							y0 = parameters0[:,i]
							y = abs((y - y0)/(maximum(y,y0)+1e-20))

						label = labels[i%2]

						plots[i][k] = ax[i][k].plot(x,y,
							color=getattr(plt.cm,'viridis')((iterations.index(j)*10)/(len(iterations)*10)),
							marker='',alpha=0.8,linewidth=3,
							# label=r'${%s}^{(%s)}_{%s}$'%(label,str(j),str(i//2) if shape[1]>2 else '')
							label=r'${%s}^{(%s)}_{%s}$'%(r'\theta',str(j),'')
						)

						ax[i][k].set_xlim(xmin=0,xmax=shape[0])
						# ax[i][k].set_ylim(ymin=lims[i%2][0],ymax=lims[i%2][1])
						ax[i][k].set_ylabel(ylabel=r'${%s}_{%s}$'%(label,str(i//2) if shape[1]>2 else ''))
						ax[i][k].set_xlabel(xlabel=r'$\textrm{%s}$'%('Time'))
						ax[i][k].set_yscale(value='linear')
						# ax[i][k].set_yscale(value='log')
						ax[i][k].grid(True)	

						if i == 0:
							if layout[1] > 1:
								if k == 0:
									ax[i][k].legend(
										title=r'${%s}^{(%s)}_{%s} ~, ~ \abs{({%s}^{(%s)}_{%s} - {%s}^{(%s)}_{%s})/\textrm{max}({%s}^{(%s)}_{%s},{%s}^{(%s)}_{%s})}$'%(
											r'\theta','i','',r'\theta','i','',r'\theta','0','',r'\theta','i','',r'\theta','0',''),
										loc=(0.15,1.05),ncol=min(6,len(ax[i][k].get_legend_handles_labels()[0]))
										)
							else:
								ax[i][k].legend(
									loc=(0.15,1.1),ncol=min(4,len(ax[i][k].get_legend_handles_labels()[0]))
									)

			fig.set_size_inches(*figsize)
			fig.subplots_adjust()
			# fig.tight_layout()
			dump(fig,path)

		self.fig[attr] = fig
		self.ax[attr] = ax

		attr = 'objective'
		fig,ax = self.fig.get(attr),self.ax.get(attr)

		path = hyperparameters['sys']['path']['plot'][attr]

		layout = []
		plots = None
		figsize = (8,8)

		with matplotlib.style.context(mplstyle):
		
			if fig is None:
				fig,ax = plt.subplots(*layout)
			elif ax is None:
				ax = fig.gca()

			x = hyperparameters['hyperparameters']['track']['iteration']
			y = hyperparameters['hyperparameters']['track']['objective']

			plots = ax.plot(x,y,linewidth=4,marker='o',markersize=10)

			ax.set_ylabel(ylabel=r'$\textrm{%s}$'%('Objective'))
			ax.set_xlabel(xlabel=r'$\textrm{%s}$'%('Iteration'))

			# ax.set_ylim(ymin=0,ymax=1)
			# ax.set_yscale(value='linear')

			ax.set_ylim(ymin=5e-1,ymax=1e0)
			ax.set_yscale(value='log',base=10)

			ax.yaxis.offsetText.set_fontsize(fontsize=20)

			ax.set_xticks(ticks=range(int(1*min(0,0,*x)),int(1.1*max(0,0,*x)),max(1,int(max(0,0,*x)-min(0,0,*x))//8)))
			# ax.set_yticks(ticks=[1e-1,2e-1,4e-1,6e-1,8e-1,1e0])
			ax.set_yticks(ticks=[5e-1,6e-1,8e-1,1e0])
			ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
			# ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0,subs=(1.0,),numticks=100))
			ax.ticklabel_format(axis='y',style='sci',scilimits=[-1,2])	

			ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0,subs=arange(2,10)*.1,numticks=100))
			ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


			ax.tick_params(axis='y',which='major',length=8,width=1)
			ax.tick_params(axis='y',which='minor',length=4,width=0.5)
			ax.tick_params(axis='x',which='major',length=8,width=1)
			ax.tick_params(axis='x',which='minor',length=4,width=0.5)

			ax.set_aspect(aspect='auto')
			ax.grid(visible=True,which='both',axis='both')	

			fig.set_size_inches(*figsize)
			fig.subplots_adjust()
			fig.tight_layout()
			dump(fig,path)


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
		system (dict,System): System attributes (dtype,format,device,seed,key,verbose)
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

		# Assert all data satisfies data**2 = identity for matrix exponentials
		assert all(allclose(d.dot(d),self.identity) for d in data), "data is not involutory and data**2 != identity"

		# Get size of data
		size = len(data)

		# Get shape of variables
		shape = (self.shape[0],size)
		ndim = len(shape)
		axes = list(range(ndim))

		# Get Trotterized order of p copies of data for products of data
		data = trotter(data,self.p)


		# Implicit parameterizations that interact with the data to produce the output are called variables x
		# These variables are parameterized by the explicit parameters theta such that x = x(theta)
		# Variables have a shape x = S = (s_0,...s_{d-1})
		
		# Each category i of parameter (variable,constant,...) has parameters with shape theta^(i) = T = (T^(i)_0,...,T^(i)_{d^(i)-1})
		# and each of these parameters yield subsets of the variables with indices I^(i) = (I^(i)_0,...,I^(i)_{d^(i)-1})
		# such that the union of all the indices U_i I^(i) across the categories covers all variables.
		
		# Each category i of parameter has groups g^(i) that depend on a slices of theta^(i), and has shape T^(i)_g^(i) = (T^(i)_g^(i)_0,...,T^(i)_g^(i)_{d^(i)_g^(i)-1})
		# and yield subsets of the variables with indices I^(i) = (I^(i)_g^(i)_0,...,I^(i)_g^(i)_{d^(i)_g^(i)-1})

		# Parameters are described by the dictionary hyperparameters['parameters'], with different parameter keys, each with an associated category i,
		# and groups of parameters that use a subset of theta^(i)

		# Get parameters and groups based on operator strings, and ensure groups are hashable
		for parameter in list(hyperparameters['parameters']):
			for i,group in enumerate(list(hyperparameters['parameters'][parameter]['group'])):
				if not any(g in [s,'_'.join([s,''.join(['%d'%j for j in i])])] 
						for g in group
						for i,s in zip(site,string)):
					hyperparameters['parameters'][parameter]['group'].remove(group);
				else:
					hyperparameters['parameters'][parameter]['group'][i] = tuple([g for g in group])

			if hyperparameters['parameters'][parameter]['group'] == []:
				hyperparameters['parameters'].pop(parameter);



		# All categories of parameters
		categories = list(set([hyperparameters['parameters'][parameter]['category'] for parameter in hyperparameters['parameters']]))
		
		# All groups for categories
		groups = {category: list(set([group
			for parameter in hyperparameters['parameters'] 
			for group in hyperparameters['parameters'][parameter]['group']
			if hyperparameters['parameters'][parameter]['category'] == category
			]))
			for category in categories
		}

		# indices indices of variables within all variables
		indices = {category:{} for category in categories}

		# slices of parameters within all parameters of category
		slices = {category:{} for category in categories}

		# shapes of slices of parameters within all parameters of category
		shapes = {category:{} for category in categories}

		# All boundaries for category
		boundaries = {category: {group: [list(set([i 
			for parameter in hyperparameters['parameters'] 
			for i in hyperparameters['parameters'][parameter]['boundaries'][axis]
			if ((hyperparameters['parameters'][parameter]['category'] == category) and 
				(all(g in group for g in hyperparameters['parameters'][parameter]['group'])))
			]))
			for axis in axes]
			for group in groups[category]
			}
			for category in categories
		}


		for parameter in hyperparameters['parameters']:

			category = hyperparameters['parameters'][parameter]['category']
			locality = hyperparameters['parameters'][parameter]['locality']
			boundary = hyperparameters['parameters'][parameter]['boundaries']

			# Length of existing slice for parameters for this category
			length_parameter = [max([0,*[slices[category][group][axis].stop for group in slices[category]]]) for axis in axes]

			for group in hyperparameters['parameters'][parameter]['group']:

				# Get indices of variables corresponding to data for category, parameter, and group
				indices[category][group] = tuple([
					*[slice(0+sum(any((j in boundary[axis]) for j in i) for i in [[0]]),
							shape[axis]-sum(any((j in boundary[axis]) for j in i) for i in [[shape[axis],-1]]),
							1) for axis in axes[:1]],
					*[[i for i,s in enumerate(string) 
					   if any(g in group for g in [s,'_'.join([s,''.join(['%d'%j for j in site[i]])])])] for axis in axes[1:2]],
					*[slice(0+sum(any((j in boundary[axis]) for j in i) for i in [[0]]),
							shape[axis]-sum(any((j in boundary[axis]) for j in i) for i in [[shape[axis],-1]]),
							1) for axis in axes[2:]],					   
				])
			
				# Get length of variable indices for local variables
				length_local = [_len_(indices[category][group][axis]) if ((axis==0) or (locality in ['local'])) else 1 for axis in axes]

				# Get number of parameters per variable for parameter
				length_size = [hyperparameters['parameters'][parameter]['size'][axis] for axis in axes]

				# Get slice of parameters with parameters of category and group,
				# corresponding to locality and number of parameters per variable, 
				# accounting for boundaries
				slices[category][group] = tuple([
					*[slice(0+sum((any((j in boundary[axis]) for j in i)-any((j in boundaries[category][group]) for j in i)) for i in [[0]]),
						    shape[axis]-sum((any((j in boundary[axis]) for j in i)-any((j in boundaries[category][group]) for j in i)) 
						    for i in [[shape[axis],-1]]),1) 
						for axis in axes[:1]],
					*[slice(length_parameter[axis],length_parameter[axis]+length_local[axis]*length_size[axis],1) 
						for axis in axes[1:]]
					])

				# Get total shape of parameters for group
				shapes[category][group] = [(slices[category][group][axis].stop-slices[category][group][axis].start)//slices[category][group][axis].step 
										   for axis in axes]

			hyperparameters['parameters'][parameter]['index'] = {group: indices[category][group] for group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['slice'] = {group: slices[category][group] for group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['shape'] =  {group: shapes[category][group] for group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['site'] =  {group: [site[j] for j in indices[category][group][-1]] for group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['string'] = {group: ['_'.join([string[j],''.join(['%d'%(k) for k in site[j]]),''.join(operator[j])]) for j in indices[category][group][-1]] for group in hyperparameters['parameters'][parameter]['group']}



		# slices of categories, including parameters and excluding boundaries
		slcs = {
			category: tuple([
						slice(
							*array(list(set([i
							for parameter in hyperparameters['parameters']
							for group in hyperparameters['parameters'][parameter]['group']
							for i in [
								*_iter_(hyperparameters['parameters'][parameter]['slice'][group][axis]),
								] 
						   if hyperparameters['parameters'][parameter]['category'] == category])))[array([0,-1])]+array([0,1])
						,1)
						for axis in axes
						])
			for category in categories
		}

		# slices of categories, including parameters and including boundaries
		slc = {
			category: tuple([
						slice(
							*array(list(set([i
							for parameter in hyperparameters['parameters']
							for group in hyperparameters['parameters'][parameter]['group']
							for i in [
								*_iter_(hyperparameters['parameters'][parameter]['slice'][group][axis]),
								*[shape[axis]+i if i< 0 else i for i in hyperparameters['parameters'][parameter]['boundaries'][axis]]
								] 
						   if hyperparameters['parameters'][parameter]['category'] == category])))[array([0,-1])]+array([0,1])
						,1)
						for axis in axes
						])
			for category in categories
		}

		# indices of categories, including variables and excluding boundaries
		indexs = {
			category: tuple([
						slice(
							*array(list(set([i
							for parameter in hyperparameters['parameters']
							for group in hyperparameters['parameters'][parameter]['group']
							for i in [
								*_iter_(hyperparameters['parameters'][parameter]['index'][group][axis]),
								] 
						   if hyperparameters['parameters'][parameter]['category'] == category])))[array([0,-1])]+array([0,1])
						,1)
						for axis in axes
						])
			for category in categories
		}

		# indices of categories, including variables and including boundaries
		index = {
			category: tuple([
						slice(
							*array(list(set([i
							for parameter in hyperparameters['parameters']
							for group in hyperparameters['parameters'][parameter]['group']
							for i in [
								*_iter_(hyperparameters['parameters'][parameter]['index'][group][axis]),
								*[shape[axis]+i if i< 0 else i for i in hyperparameters['parameters'][parameter]['boundaries'][axis]]
								] 
						   if hyperparameters['parameters'][parameter]['category'] == category])))[array([0,-1])]+array([0,1])
						,1)
						for axis in axes
						])
			for category in categories
		}

		# shape of categories, including parameters and excluding boundaries
		shapes = {
			category: tuple([
						len(							
							set([i
							for parameter in hyperparameters['parameters']
							for group in hyperparameters['parameters'][parameter]['group']
							for i in [
								*_iter_(hyperparameters['parameters'][parameter]['slice'][group][axis]),
								] 
						   if hyperparameters['parameters'][parameter]['category'] == category])
						)
						for axis in axes
						])
			for category in categories
		}

		# shape of categories, including parameters and including boundaries
		shape = {
			category: tuple([
						len(							
							set([i
							for parameter in hyperparameters['parameters']
							for group in hyperparameters['parameters'][parameter]['group']
							for i in [
								*_iter_(hyperparameters['parameters'][parameter]['slice'][group][axis]),
								*[shape[axis]+i if i< 0 else i for i in hyperparameters['parameters'][parameter]['boundaries'][axis]]
								] 
						   if hyperparameters['parameters'][parameter]['category'] == category])
						)
						for axis in axes
						])
			for category in categories
		}
	

		# print('\t\t\t\tslice,\t\t\t\t index,\t\t   sliceshape,  catshapes,  totshape')
		# for parameter in hyperparameters['parameters']:
		# 	for group in hyperparameters['parameters'][parameter]['group']:
		# 		print(parameter,group,hyperparameters['parameters'][parameter]['category'],
		# 			hyperparameters['parameters'][parameter]['slice'][group],
		# 			hyperparameters['parameters'][parameter]['index'][group],
		# 			hyperparameters['parameters'][parameter]['shape'][group],
		# 			shapes[hyperparameters['parameters'][parameter]['category']],
		# 			shape[hyperparameters['parameters'][parameter]['category']])
		# print()

		# Update hyperparameters
		hyperparameters['size'] = size
		hyperparameters['shapes'] = shapes
		hyperparameters['shape'] = shape
		hyperparameters['slices'] = slcs
		hyperparameters['slice'] = slc
		hyperparameters['indexes'] = indexs
		hyperparameters['index'] = index

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
			variables (array): variables
		'''		

		# Get class attributes
		self.parameters = parameters
		hyperparameters = self.hyperparameters

		# Set all variables
		
		category = 'variable'
		shape = hyperparameters['shapes'][category]
		parameters = parameters.reshape(shape)

		variables = hyperparameters['variables']

		for parameter in hyperparameters['parameters']:
			if hyperparameters['parameters'][parameter]['category'] == category:
				for group in hyperparameters['parameters'][parameter]['group']:
					indices = hyperparameters['parameters'][parameter]['index'][group]
					variables = variables.at[indices].set(
						hyperparameters['parameters'][parameter]['variables'][group](parameters))

		# print('param,vars')
		# print(parameters.round(3))
		# print(variables.round(3))
		# print()

		# Get Trotterized order of copies of variables
		p = self.p
		variables = trotter(variables.T,p).T


		# Get reshaped variables
		variables = variables.ravel()
		
		return variables


	@partial(jit,static_argnums=(0,))
	def __features__(self,parameters):
		''' 
		Setup features
		Args:
			parameters (array): parameters
		Returns:
			features (array): features
		'''

		# Get class attributes
		hyperparameters = self.hyperparameters

		# Set all features
		
		category = 'variable'
		shape = hyperparameters['shapes'][category]
		ndim = len(shape)
		axes = list(range(ndim))
		index = hyperparameters['index'][category]		
		size = [max(hyperparameters['parameters'][parameter]['size'][axis] 
				for parameter in hyperparameters['parameters'] 
				if hyperparameters['parameters'][parameter]['category'] == category)
				for axis in axes]

		parameters = parameters.reshape(shape)

		features = take(hyperparameters['variables'],index,axes)

		for axis in axes:
			features = repeat(features,repeats=size[axis],axis=axis)

		for parameter in hyperparameters['parameters']:
			if hyperparameters['parameters'][parameter]['category'] == category:
				for group in hyperparameters['parameters'][parameter]['group']:
					feature = hyperparameters['parameters'][parameter]['features'][group](parameters)
					index = hyperparameters['parameters'][parameter]['index'][group]
					length = len(feature)
					for l in range(length):
						indices = tuple([(
							index[axis] if axis != axes[-1] else 
							slice(
								index[axis].start+l,
								index[axis].stop,
								index[axis].step*length)
							) if isinstance(index[axis],slice) else array(index[axis])*length+l
							for axis in axes
							])
						features = features.at[indices].set(feature[l])

		return features


	def __init__parameters__(self,parameters):
		''' 
		Setup initial parameters
		Args:
			parameters (array): parameters
		'''

		# Get class attributes
		hyperparameters = self.hyperparameters
		dtype = self._dtype

		# Initialize parameters

		# Get shape of parameters of different category
		categories = list(set([hyperparameters['parameters'][parameter]['category'] for parameter in hyperparameters['parameters']]))
		parameters = {}

		# Get parameters for each category
		# reshape, bound, impose boundary conditions accordingly, and assign category parameters
		for category in categories:
			shape = hyperparameters['shape'][category]
			ndim = len(shape)
			axes = list(range(ndim))

			parameters[category] = zeros(shape,dtype=dtype)

			# Assign group parameters to category of parameters
			for parameter in hyperparameters['parameters']:
				if not (hyperparameters['parameters'][parameter]['category'] == category):
					continue


				# Existing parameters for parameter
				params = hyperparameters['parameters'][parameter].get('parameters',None)

				# Axes to repeat existing parameters
				repeats = hyperparameters['parameters'][parameter].get('repeats',[])

				# Boundaries of the form [{i:value} for axis in axes]
				boundary = hyperparameters['parameters'][parameter].get('boundaries',[])

				# Constants of the form [{i:value} for axis in axes]
				constant = hyperparameters['parameters'][parameter].get('constants',[])				
				
				# If parameters exist
				reset =  params is None

				# Hyperparameters for parameter
				hyperparams = hyperparameters['parameters'][parameter]

				# Assign to category parameters for each group
				for group in hyperparameters['parameters'][parameter]['group']:
					
					slices = hyperparameters['parameters'][parameter]['slice'][group]
					shapes = hyperparameters['parameters'][parameter]['shape'][group]

					if reset:
						params = zeros(shapes,dtype=dtype)
					else:
						params = array(params,dtype=dtype)

					for axis in repeats:
						params = repeat(expand_dims(params,axis),shapes[axis],axis)

					params = take(params,shapes,axes)

					parameters[category] = (
						parameters[category].at[slices].set(
							initialize(params,shapes,reset=reset,dtype=dtype,hyperparameters=hyperparams)
						)
					)

				for axis in axes:
					for i in boundary[axis]:
						slices = [slice(None) for axis in axes]
						value = None
				
						slices[axis] = i
						value = boundary[axis][i]
				
						parameters[category] = parameters[category].at[tuple(slices)].set(value)


				for axis in axes:
					for i in constant[axis]:
						slices = [slice(None) for axis in axes]
						value = None
				
						slices[axis] = i
						value = constant[axis][i]
				
						parameters[category] = parameters[category].at[tuple(slices)].set(value)


		# Get variables
		hyperparameters['variables'] = zeros(self.shape[:2])
		
		for parameter in hyperparameters['parameters']:
			category = hyperparameters['parameters'][parameter]['category']
			for group in hyperparameters['parameters'][parameter]['group']:
				slices = hyperparameters['parameters'][parameter]['slice'][group]
				indices = hyperparameters['parameters'][parameter]['index'][group]
				hyperparameters['variables'] = hyperparameters['variables'].at[indices].set(
					hyperparameters['parameters'][parameter]['variables'][group](parameters[category][slices])
					)

		# Get label
		label = hyperparameters['label']
		shape = self.shape[2:]
		if label is None:
			label = (rand(shape)+ 1j*rand(shape))/sqrt(2)
			label = sp.linalg.expm(-1j*(label + label.conj().T)/2.0/self.n)

		elif isinstance(label,str):
			
			if label == 'random':
				label = (rand(shape)+ 1j*rand(shape))/sqrt(2)
				label = sp.linalg.expm(-1j*(label + label.conj().T)/2.0/self.n)

				Q,R = qr(label);
				R = diag(diag(R)/abs(diag(R)));
				label = Q.dot(R)
				assert allclose(eye(self.n),label.conj().T.dot(label))
				assert allclose(eye(self.n),label.dot(label.conj().T))

			elif label == 'rank1':
				label = diag(rand(self.n))
				I = eye(self.n)
				r = 4*self.N
				k = rand(shape=(r,2),bounds=[0,self.n],random='randint')
				for j in range(r):
					v = outer(I[k[j,0]],I[k[j,1]].T)
					c = (rand()+ 1j*rand())/sqrt(2)
					v = (v + (v.T))
					v = (c*v + c.conj()*(v.T))
					label += v
				label = sp.linalg.expm(-1j*label)

			elif label == 'gate':
				label = {
					2: array([[1,0,0,0],
							   [0,1,0,0],
							   [0,0,0,1],
							   [0,0,1,0]]),
					# 2: tensorprod(((1/sqrt(2)))*array(
					# 		[[[1,1],
					# 		  [1,-1]]]*2)),
					# 2: array([[1,0,0,0],
					# 		   [0,1,0,0],
					# 		   [0,0,1,0],
					# 		   [0,0,0,1]]),					   		
					# 2: tensorprod(((1/sqrt(2)))*array(
					# 		[[[1,1],
					# 		  [1,-1]],
					# 		 [[1,1],
					# 		  [1,-1]],
					# 		  ])),
					3: tensorprod(((1/sqrt(2)))*array(
							[[[1,1],
							  [1,-1]]]*3)),
					3: array([[1,0,0,0,0,0,0,0],
							   [0,1,0,0,0,0,0,0],
							   [0,0,1,0,0,0,0,0],
							   [0,0,0,1,0,0,0,0],
							   [0,0,0,0,1,0,0,0],
							   [0,0,0,0,0,1,0,0],
							   [0,0,0,0,0,0,0,1],
							   [0,0,0,0,0,0,1,0]]),
					4: tensorprod(
						array(
							[tensorprod((1/sqrt(2))*array(
								[[[1,1],
								  [1,-1]]]*2)),
							[[1,0,0,0],
							[0,1,0,0],
							[0,0,0,1],
							[0,0,1,0]]
						])
						),					
					# 4: tensorprod(((1/sqrt(2)))*array(
					# 		[[[1,1],
					# 		  [1,-1]],
					# 		 [[1,1],
					# 		  [1,-1]],
					# 		  [[1,1],
					# 		  [1,-1]],
					# 		 [[1,1],
					# 		  [1,-1]],							  
					# 		 ])),	
					# 4: tensorprod(array([[[1,0,0,0],
					# 		   [0,1,0,0],
					# 		   [0,0,0,1],
					# 		   [0,0,1,0]]]*2)),	
					# 4: tensorprod(array([[[1,0,0,0],
					# 		   [0,1,0,0],
					# 		   [0,0,1,0],
					# 		   [0,0,0,1]]]*2)),						   		 
					}.get(self.N)
			else:
				try:
					label = array(load(label))
				except:
					label = (rand(shape)+ 1j*rand(shape))/sqrt(2)
					label = sp.linalg.expm(-1j*(label + label.conj().T)/2.0/self.n)					
		else:
			label = array(label)

		label = label.astype(dtype=self.dtype)

		hyperparameters['label'] = label #.conj().T

		# Get reshaped parameters
		category = 'variable'		
		slices = hyperparameters['slices'][category]
		parameters = parameters[category][slices].ravel()

		# Update class attributes
		self.parameters = parameters
		self.hyperparameters = hyperparameters


		return



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
		system (dict,System): System attributes (dtype,format,device,seed,key,verbose)
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
		return exponentiation(-1j*self.coefficients*parameters,self.data,self.identity)


	@partial(jit,static_argnums=(0,))
	def __derivative__(self,parameters):
		'''
		Return gradient of parameterized operator expm(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			derivative (array): Gradient of parameterized operator
		'''		
		
		category = 'variable'
		axis = 1
		_shape = self.hyperparameters['shape'][category]
		shape = self.hyperparameters['shapes'][category]
		_index = self.hyperparameters['index'][category]
		index = self.hyperparameters['indexes'][category]
		parameters = self.__parameters__(parameters)

		grad = gradient_expm(-1j*self.coefficients*parameters,self.data,self.identity)
		grad *= -1j*self.coefficients
		grad = grad.reshape((*_shape[:1],-1,*self.shape[2:]))
		grad = grad[index[0]]
		grad = grad.reshape((shape[0],-1,*self.shape[2:]))

		grad = grad.transpose(axis,0,*[i for i in range(grad.ndim) if i not in [0,axis]])
		grad = gradient_trotter(grad,self.p)
		grad = grad[index[axis]]
		grad = grad.transpose(axis,0,*[i for i in range(grad.ndim) if i not in [0,axis]])
		
		grad = grad.reshape((-1,*grad.shape[2:]))

		derivative = grad
		
		# derivative = zeros(shape)

		# for parameter in self.hyperparameters['parameters']:
		# 	if self.hyperparameters['parameters'][parameter]['category'] == category:
		# 		for group in self.hyperparameters['parameters'][parameter]['group']:
		# 			derivative = derivative.at[self.hyperparameters['parameters'][parameter]['slice'][group][1]].set(
		# 				derivative.at[self.hyperparameters['parameters'][parameter]['slice'][group][1]] + 
		# 				self.hyperparameters['parameters'][parameter]['grad'][group](parameters).dot(
		# 				grad[self.hyperparameters['parameters'][parameter]['index'][group][-1]])
		# 				)

		return derivative



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
	# return 1-inner_real2(a,b)+inner_imag2(a,b)
	return 1-inner_abs2(a,b)
	


def gradient_distance(a,b,da):
	'''
	Calculate distance between arrays
	Args:
		a (array): Array to calculate distance
		b (array): Array to calculate distance
		da (array): Gradient of array to calculate distance		
	Returns:
		out (array): Distance between arrays
	'''
	return -gradient_inner_abs2(a,b,da)
	

def trotter(a,p):
	'''
	Calculate p-order trotter series of array
	Args:
		a (array): Array to calculate trotter series
		p (int): Order of trotter series
	Returns:
		out (array): Trotter series of array
	'''	
	return array([v for u in [a[::i] for i in [1,-1,1,-1][:p]] for v in u])

def gradient_trotter(da,p):
	'''
	Calculate gradient of p-order trotter series of array
	Args:
		da (array): Gradient of array to calculate trotter series		
		p (int): Order of trotter series
	Returns:
		out (array): Gradient of trotter series of array
	'''	
	n = da.shape[0]//p
	return sum([da[:n][::i] if i>0 else da[-n:][::i] for i in [1,-1,1,-1][:p]])


def invtrotter(a,p):
	'''
	Calculate inverse of p-order trotter series of array
	Args:
		a (array): Array to calculate inverse trotter series
		p (int): Order of trotter series
	Returns:
		out (array): Inverse trotter series of array
	'''	
	n = a.shape[0]//p
	return a[:n]


def initialize(parameters,shape,hyperparameters,reset=None,dtype=None):
	'''
	Initialize parameters
	Args:
		parameters (array): parameters array
		shape (iterable): shape of parameters
		hyperparameters (dict): hyperparameters for initialization
		reset (bool): Overwrite existing parameters
		dtype (str,datatype): data type of parameters		
	Returns:
		out (array): initialized slice of parameters
	'''	

	# Initialization hyperparameters
	bounds = hyperparameters['bounds']
	constant = hyperparameters['constants']	
	initialization = hyperparameters['initialization']
	random = hyperparameters['random']
	pad = hyperparameters['pad']
	seed = hyperparameters['seed']
	key = seed

	# Parameters shape and bounds
	shape = shape
	ndim = len(shape)
	axes = list(range(ndim))

	bounds = [to_number(i,dtype) for i in bounds]
	if not any(isnaninf(i) for i in bounds):
		bounds = [
			bounds[0] + (bounds[1]-bounds[0])*hyperparameters['init'][0],
			bounds[1]*hyperparameters['init'][1],
		]

	# Add random padding of values if parameters not reset
	if not reset:
		_shape = [i for i in parameters.shape]
		_diff = [shape[axis] - _shape[axis] for axis in axes]
		_bounds = bounds
		_random = pad
		for axis in range(ndim-1,0,-1):
			if _diff[axis] > 0:

				j = 0

				_shape[axis] = _diff[axis] 
				_key = None
				_parameters = rand(_shape,key=_key,bounds=_bounds,random=_random)
				_shape[axis] = shape[axis]

				parameters = moveaxis(parameters,axis,j)
				_parameters = moveaxis(_parameters,axis,j)

				parameters = array([*parameters,*_parameters])
				
				parameters = moveaxis(parameters,j,axis)				
				_parameters = moveaxis(_parameters,j,axis)

		axis = 0
		_shape[axis] += _diff[axis]
		_shape = tuple(_shape)
		_parameters = broadcast_to(parameters,_shape)
		parameters = _parameters

	else:
		if initialization in ["interpolation"]:
			# Parameters are initialized as interpolated random values between bounds
			interpolation = hyperparameters['interpolation']
			smoothness = min(shape[0]//2,hyperparameters['smoothness'])
			shape_interp = (shape[0]//smoothness,*shape[1:])
			pts_interp = (shape_interp[0])*smoothness*arange(shape_interp[0])
			pts = arange(shape[0])

			parameters_interp = rand(shape_interp,key=key,bounds=bounds,random=random)

			for axis in axes:
				for i in constant[axis]:
					slices = tuple([slice(None) if ax != axis else i for ax in axes])
					value = constant[axis][i]			
					parameters_interp = parameters_interp.at[slices].set(value)

			parameters = interpolate(pts_interp,parameters_interp,pts,interpolation)

			parameters = minimum(bounds[1],maximum(bounds[0],parameters))

		elif initialization in ['uniform']:
			parameters = ((bounds[0]+bounds[1])/2)*ones(shape)
		elif initialization in ['random']:
			parameters = rand(shape,key=key,bounds=bounds,random=random)
		elif initialization in ['zero']:
			parameters = zeros(shape)
		else:
			parameters = rand(shape,key=key,bounds=bounds,random=random)		

	return parameters



def plot(hyperparameters):
	'''
	Plot runs
	Args:
		hyperparameters (dict): hyperparameters of runs
	'''	

	# Get keys of hyperparameters
	keys = list(hyperparameters)
	k = len(keys)

	if k == 0:
		return

	key = keys[0]

	# Get plot config
	attr = 'mplstyle'
	mplstyle = hyperparameters[key]['sys']['path']['config'][attr]

	# Plot attributes

	attr = 'objective'
	key = keys[0]
	fig,ax = None,None

	path = hyperparameters[key]['sys']['path']['plot'][attr]
	delimiter = '.'
	directory,file,ext = path_split(path,directory=True,file=True,ext=True,delimiter=delimiter)
	file = delimiter.join([*file.split(delimiter)[:-1],'all'])
	path = path_join(directory,file,ext=ext,delimiter=delimiter)

	layout = []
	plots = None
	shape = (len(hyperparameters[key]['hyperparameters']['runs']),max(hyperparameters[key]['hyperparameters']['track']['size'] for key in keys))
	figsize = (8,8)

	with matplotlib.style.context(mplstyle):
	
		if fig is None:
			fig,ax = plt.subplots(*layout)
		elif ax is None:
			ax = fig.gca()

		x = zeros(shape)	
		y = zeros(shape)
		yerr = zeros(shape)

		for i,key in enumerate(keys):
			size = hyperparameters[key]['hyperparameters']['track']['size']

			x = x.at[i,:].set(arange(shape[1]))
			y = y.at[i,:size].set(hyperparameters[key]['hyperparameters']['track']['objective'])
			y = y.at[i,size:].set(hyperparameters[key]['hyperparameters']['track']['objective'][-1:])
			yerr = yerr.at[i,:size].set(hyperparameters[key]['hyperparameters']['track']['objective'])
			yerr = yerr.at[i,size:].set(hyperparameters[key]['hyperparameters']['track']['objective'][-1:])

		x = x.mean(0).astype(int)
		y = y.mean(0)
		yerr = yerr.std(0)

		plots = ax.errorbar(x,y,yerr,fmt='--o',ecolor='k',elinewidth=2,capsize=3)

		ax.set_ylabel(ylabel=r'$\textrm{%s}$'%('Objective'))
		ax.set_xlabel(xlabel=r'$\textrm{%s}$'%('Iteration'))

		# ax.set_ylim(ymin=0,ymax=1.1)
		# ax.set_yscale(value='linear')

		ax.set_ylim(ymin=5e-1,ymax=1.05e0)
		ax.set_yscale(value='log',base=10)

		ax.yaxis.offsetText.set_fontsize(fontsize=20)

		ax.set_xticks(ticks=range(int(1*min(0,0,*x)),int(1.1*max(0,0,*x)),max(1,int(max(0,0,*x)-min(0,0,*x))//8)))
		# ax.set_yticks(ticks=[1e-1,2e-1,4e-1,6e-1,8e-1,1e0])
		ax.set_yticks(ticks=[5e-1,6e-1,8e-1,1e0])
		ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		# ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0,subs=(1.0,),numticks=100))
		ax.ticklabel_format(axis='y',style='sci',scilimits=[-1,2])	

		ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0,subs=arange(2,10)*.1,numticks=100))
		ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


		ax.tick_params(axis='y',which='major',length=8,width=1)
		ax.tick_params(axis='y',which='minor',length=4,width=0.5)
		ax.tick_params(axis='x',which='major',length=8,width=1)
		ax.tick_params(axis='x',which='minor',length=4,width=0.5)

		ax.set_aspect(aspect='auto')
		ax.grid(visible=True,which='both',axis='both')	

		fig.set_size_inches(*figsize)
		fig.subplots_adjust()
		fig.tight_layout()
		dump(fig,path)

		plt.close(fig);



	attr = 'label'
	key = keys[0]	
	fig,ax = None,None

	path = hyperparameters[key]['sys']['path']['plot'][attr]
	delimiter = '.'
	directory,file,ext = path_split(path,directory=True,file=True,ext=True,delimiter=delimiter)
	file = delimiter.join([*file.split(delimiter)[:-1],'all'])
	path = path_join(directory,file,ext=ext,delimiter=delimiter)

	layout = [2]
	plots = [None]*layout[0]
	label = hyperparameters[key]['label'] if isinstance(hyperparameters[key]['label'],array) else rand((1,1))
	shape = (len(hyperparameters[key]['hyperparameters']['runs']),*label.shape)
	figsize = (8,8)
	labels = {'real':r'$U~\textrm{Real}$','imag':r'$U~\textrm{Imag}$'}
	dtype = label.dtype

	with matplotlib.style.context(mplstyle):
	
		if fig is None:
			fig,ax = plt.subplots(*layout)
		elif ax is None:
			ax = fig.gca()

		x = zeros(shape,dtype=dtype)

		for i,key in enumerate(keys):
			x = x.at[i].set(label)

		x = x.mean(0)

		for i,attr in enumerate(labels):
			plots[i] = ax[i].imshow(getattr(x,attr))
			
			ax[i].set_title(labels[attr])
			
			plt.colorbar(plots[i],ax=ax[i])

		fig.set_size_inches(*figsize)
		fig.subplots_adjust()
		fig.tight_layout()
		dump(fig,path)

		plt.close(fig);

	return


def check(hyperparameters):

	section = None
	updates = {
		'label': {
			'value': (lambda hyperparameters: hyperparameters['hyperparameters']['label']),
			'default': (lambda hyperparameters: None),
			'conditions': (lambda hyperparameters: hyperparameters['hyperparameters'].get('label') is not None)				
		},
	}			
	for attr in updates:								
		hyperparameters[attr] = hyperparameters.get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[attr] = updates[attr]['value'](hyperparameters)


	section = 'sys'
	updates = {
		'path': {
			'value': (lambda hyperparameters: 	{
				attr: path_join(hyperparameters[section]['directory'][attr],
								 '.'.join([hyperparameters[section]['file'][attr]]) if attr in ['data','plot'] else hyperparameters[section]['file'][attr],
								 ext=hyperparameters[section]['ext'][attr])
						if isinstance(hyperparameters[section]['file'][attr],str) else
						{i: path_join(hyperparameters[section]['directory'][attr][i],
								 '.'.join([hyperparameters[section]['file'][attr][i]]) if attr in ['data','plot'] else hyperparameters[section]['file'][attr][i],							 
								 ext=hyperparameters[section]['ext'][attr][i])
						for i in hyperparameters[section]['file'][attr]}
				for attr in hyperparameters[section]['file']			 
			}),
			'default': (lambda hyperparameters: None),
			'conditions': (lambda hyperparameters: hyperparameters[section].get('path') is None)
		},		
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)


	section = 'model'
	updates = {
		'tau': {
			'value': (lambda hyperparameters: hyperparameters[section]['tau']/hyperparameters['hyperparameters']['scale']),
			'default': (lambda hyperparameters: 1),
			'conditions': (lambda hyperparameters: True)
		},		
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)


	section = 'hyperparameters'
	updates = {
		'runs': {
			'value': (lambda hyperparameters: list(range(hyperparameters[section]['runs']))),
			'default': (lambda hyperparameters: 1),
			'conditions': (lambda hyperparameters: not isinstance(hyperparameters[section]['runs'],(list,tuple,array)))
		},
	}			
	for attr in updates:						
		hyperparameters[section][attr] = hyperparameters[section].get(attr,updates[attr]['default'](hyperparameters))
		if updates[attr]['conditions'](hyperparameters):
			hyperparameters[section][attr] = updates[attr]['value'](hyperparameters)


	section = 'parameters'
	updates = {
		'boundaries': {
			'value': (lambda parameter,hyperparameters: [{int(j):i[j] for j in i} for i in hyperparameters[section][parameter]['boundaries']]),
			'default': (lambda parameter,hyperparameters: []),
			'conditions': (lambda parameter,hyperparameters: True)				
		},
		'constants': {
			'value': (lambda parameter,hyperparameters: [{int(j):i[j] for j in i} for i in hyperparameters[section][parameter]['constants']]),
			'default': (lambda parameter,hyperparameters: []),
			'conditions': (lambda parameter,hyperparameters: True)				
		},		
		'group': {
			'value': (lambda parameter,hyperparameters: [tuple(group) for group in hyperparameters[section][parameter]['group']]),
			'default': (lambda parameter,hyperparameters: []),
			'conditions': (lambda parameter,hyperparameters: True)				
		},
		**{attr: {
			'value': (lambda parameter,hyperparameters,attr=attr: hyperparameters['hyperparameters'].get(attr)),
			'default': (lambda parameter,hyperparameters,attr=attr: None),
			'conditions': (lambda parameter,hyperparameters,attr=attr: hyperparameters['parameters'][parameter].get(attr) is None)						
			} for attr in ['scale','initialization','random','init','smoothness','interpolation','pad']
		},
		**{attr: {
			'value': (lambda parameter,hyperparameters,attr=attr: hyperparameters['hyperparameters'].get(attr)),
			'default': (lambda parameter,hyperparameters,attr=attr: None),
			'conditions': (lambda parameter,hyperparameters,attr=attr: hyperparameters['parameters'][parameter].get(attr) is not None)						
			} for attr in ['seed']
		},		
		'locality': {
			'value':(lambda parameter,hyperparameters: hyperparameters['hyperparameters']['locality']),
			'default':(lambda parameter,hyperparameters: None),
			'conditions': (lambda parameter,hyperparameters: hyperparameters['parameters'][parameter]['category'] in ['variable'])
		},		
	}			
	for parameter in hyperparameters[section]:
		for attr in updates:						
			hyperparameters[section][parameter][attr] = hyperparameters[section][parameter].get(attr,updates[attr]['default'](parameter,hyperparameters))
			if updates[attr]['conditions'](parameter,hyperparameters):
				hyperparameters[section][parameter][attr] = updates[attr]['value'](parameter,hyperparameters)

	return

def setup(hyperparameters):

	check(hyperparameters)

	settings = {}

	settings['hyperparameters'] = {key: copy.deepcopy(hyperparameters) for key in hyperparameters['hyperparameters']['runs']}

	settings['boolean'] = {attr: hyperparameters['boolean'].get(attr,False) for attr in hyperparameters['boolean']}
	settings['boolean'].update({attr: settings['boolean'].get(attr,False) and (not settings['boolean'].get('load',False)) for attr in ['train']})

	settings['seed'] = {key:seed for key,seed in zip(
		settings['hyperparameters'],
		PRNGKey(hyperparameters['hyperparameters']['seed'],
			split=len(settings['hyperparameters']),
			reset=hyperparameters['hyperparameters']['seed'])
		)}

	for key in settings['hyperparameters']:
		settings['hyperparameters'][key]['model']['system']['key'] = key
		
		settings['hyperparameters'][key]['model']['system']['seed'] = hyperparameters['hyperparameters']['seed']
		settings['hyperparameters'][key]['hyperparameters']['seed'] = settings['seed'][key]

		settings['hyperparameters'][key]['sys']['path'] = {
			attr: path_join(settings['hyperparameters'][key]['sys']['directory'][attr],
							 '.'.join([settings['hyperparameters'][key]['sys']['file'][attr],*[str(key)]]) if attr in ['data','plot'] else settings['hyperparameters'][key]['sys']['file'][attr],
							 ext=settings['hyperparameters'][key]['sys']['ext'][attr])
					if isinstance(settings['hyperparameters'][key]['sys']['file'][attr],str) else
					{i: path_join(settings['hyperparameters'][key]['sys']['directory'][attr][i],
							 '.'.join([settings['hyperparameters'][key]['sys']['file'][attr][i],*[str(key)]]) if attr in ['data','plot'] else settings['hyperparameters'][key]['sys']['file'][attr][i],							 
							 ext=settings['hyperparameters'][key]['sys']['ext'][attr][i])
					for i in settings['hyperparameters'][key]['sys']['file'][attr]}
			for attr in settings['hyperparameters'][key]['sys']['file']			 
		}


	return settings


def run(hyperparameters):
	'''
	Run simulations
	Args:
		hyperparameters (dict): hyperparameters
	'''		

	settings = setup(hyperparameters)

	defaults = copy.deepcopy(hyperparameters)

		
	for key in settings['hyperparameters']:			
		
		if settings['boolean']['load']:
			default = settings['hyperparameters'][key]
			def func(key,iterable,elements): 
				return (elements.get(key,iterable.get(key)) if not callable(iterable.get(key)) else iterable.get(key))
			path = settings['hyperparameters'][key]['sys']['path']['data']['data']
			_update(settings['hyperparameters'][key],load(path,default=default),_func=func)

		if settings['boolean']['dump']:
			data = copy.deepcopy(settings['hyperparameters'][key])
			path = settings['hyperparameters'][key]['sys']['path']['config']['settings'] 
			dump(data,path,callables=False)

		hyperparameters = settings['hyperparameters'][key]	

		obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

		parameters = obj.__parameters__(obj.parameters)
		if settings['boolean']['train']:

			parameters = obj.parameters
			hyperparameters = hyperparameters['hyperparameters']

			func = obj.__func__
			callback = obj.__callback__

			optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparameters)

			parameters = optimizer(parameters)

		
		if settings['boolean']['plot']:
			parameters = obj.parameters
			obj.__plot__(parameters)

		if settings['boolean']['dump']:
			data = settings['hyperparameters'][key]
			path = settings['hyperparameters'][key]['sys']['path']['data']['data'] 
			dump(data,path)

	if settings['boolean']['plot']:
		plot(settings['hyperparameters'])

	if settings['boolean']['test']:

		hyperparameters = defaults

		obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

		func = obj.__func__

		parameters = obj.parameters
		hyperparameters = hyperparameters['hyperparameters']

		g = gradient_fwd(obj)
		f = gradient_finite(obj,tol=6e-8)
		a = obj.__derivative__

		print(allclose(g(parameters),f(parameters)))
		print(allclose(g(parameters),a(parameters)))
		print(allclose(f(parameters),a(parameters)))

		print((g(parameters)-a(parameters))/g(parameters))

		grad = gradient(func)
		fgrad = gradient_finite(func,tol=5e-8)
		agrad = obj.__grad__

		print(allclose(grad(parameters),fgrad(parameters)))
		print(allclose(grad(parameters),agrad(parameters)))


		# print()
		# print(parameters)
		# print(obj.__constraints__(parameters))
		# print(sigmoid(parameters[:4]))
		# print(gradient(lambda x: sigmoid(x,scale=1e4).sum())(parameters[:4]))
		# print(grad(parameters))
		# print(agrad(parameters))

		# print(gradient(obj.__constraints__)(parameters))
	return