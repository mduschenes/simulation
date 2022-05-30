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
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer
from src.utils import summation,exponentiation
from src.utils import inner_abs2,inner_real2,inner_imag2
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_abs2,gradient_inner_real2,gradient_inner_imag2
from src.utils import eigh,qr
from src.utils import maximum,minimum,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,isclose,isnaninf
from src.utils import parse,to_str,to_number,datatype,slice_size
from src.utils import pi,e
from src.utils import itg,flt,dbl

from src.dictionary import _update

from src.parameters import parameterize
from src.operators import operatorize

from src.io import load,dump,path_join,path_split

from src.optimize import Optimizer,Objective


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
			'verbose': {
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

		self.data = array([])
		self.operator = []
		self.site = []
		self.string = []
		self.interaction = []
		self.size = 0
		self.shape = (*self.data.shape[0:1],*[self.M],*self.data.shape[1:])

		self.delimiter = ' '
		self.basis = None
		self.diagonal = []
		self._data = []
		self.funcs = lambda parameters: None
		self.expms = lambda parameters: None
		self.transform = []
		self.transformH = []
		self.index = arange(self.size)

		self.hyperparameters = hyperparameters
		self.parameters = None
		self.attributes = {}
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

		parameters = None

		# Update class attributes
		self.__extend__(data,operator,site,string,interaction,hyperparameters)
		
		# Initialize parameters
		self.__initialize__(parameters)

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
		self.shape = (*self.data.shape[0:1],*self.shape[1:2],*self.data.shape[1:])

		self.hyperparameters.update(hyperparameters)

		return



	def __initialize__(self,parameters):
		''' 
		Setup initial parameters and attributes
		Args:
			parameters (array): parameters
		'''

		# Get class attributes
		hyperparameters = self.hyperparameters
		dtype = self._dtype

		# Setup parameter shapes and slices based on data

		# Get site and strings for data
		site = self.site
		string = self.string

		# Get function to check if index of data for axis corresponds to group 
		check = lambda group,index,axis: (
			(axis != 0) or 
			any(g in group for g in [string[index],'_'.join([string[index],''.join(['%d'%j for j in site[index]])])]))

		# Get parameters data
		parameters = None

		# Get shape of data
		shape = self.shape[0:2]

		# Get parameters hyperparameters
		hyperparams = hyperparameters['parameters']

		# Get attributes data of parameters of the form {attribute:{parameter:{group:{layer:[]}}}
		data = parameters
		attributes = parameterize(data,shape,hyperparams,check=check,initialize=initialize,dtype=dtype)

		# Get label
		data = hyperparameters['label']
		shape = self.shape[2:]
		hyperparams = hyperparameters['hyperparameters']
		index = self.N
		dtype = self.dtype

		operator = operatorize(data,shape,hyperparams,index=index,dtype=dtype)

		hyperparameters['label'] = operator #.conj().T

		# Get reshaped parameters
		attribute = 'values'
		layer = 'parameters'
		parameters = attributes[attribute][layer]

		parameters = parameters.ravel()

		# Update class attributes
		self.parameters = parameters
		self.hyperparameters = hyperparameters
		self.attributes = attributes

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


	@partial(jit,static_argnums=(0,2))
	def __layers__(self,parameters,layer='variables'):
		''' 
		Setup layer
		Args:
			parameters (array): parameters
			layer (str): layer
		Returns:
			values (array): values
		'''

		# Get class attributes
		self.parameters = parameters
		attributes = self.attributes

		attribute = 'shape'
		layr = 'parameters'
		parameters = parameters.reshape(attributes[attribute][layr])

		attribute = 'values'
		features = attributes[attribute][layer]

		# Get variables
		attribute = 'slice'
		for parameter in attributes[attribute][layer]:
			for group in attributes[attribute][layer][parameter]:

				attr = layer
				func = attributes[attr][layer][parameter][group]
				
				attr = 'slice'
				slices = attributes[attr][layer][parameter][group]
				
				attr = 'index'
				indices = attributes[attr][layer][parameter][group]

				features = func(parameters,features,slices,indices)

		return features

	@partial(jit,static_argnums=(0,))
	def __constraints__(self,parameters):
		''' 
		Setup constraints
		Args:
			parameters (array): parameters
		Returns:
			constraints (array): constraints
		'''		

		layer = 'constraints'
		constraints = self.__layers__(parameters,layer)

		return constraints


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
			# 	parameters.reshape(self.hyperparameters['shape']['take']['parameters'])[:,0],
			# 	parameters.reshape(self.hyperparameters['shape']['take']['parameters'])[:,-1],
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

		# Get class hyperparameters and attributes
		hyperparameters = self.hyperparameters
		attributes = self.attributes


		# Get parameters shape and indices of features
		attribute = 'shape'
		layer = 'features'
		shape = attributes[attribute][layer]

		attribute = 'index'
		layer = 'features'		
		parameter = list(attributes[attribute][layer])[0]
		group = list(attributes[attribute][layer][parameter])[0]
		indices = attributes[attribute][layer][parameter][group]

		shape = [slice_size(index) if isinstance(index,slice) else len(index) for index in indices]

		# Get number of iterations
		size = hyperparameters['hyperparameters']['track']['size']

		# Get plot config
		attr = 'mplstyle'
		mplstyle = hyperparameters['sys']['path']['config'][attr]

		# Plot attributes

		attr = 'parameters'

		layer = 'features'

		fig,ax = self.fig.get(attr),self.ax.get(attr)
		
		path = hyperparameters['sys']['path']['plot'][attr]

		layout = [shape[0]*shape[1],1]
		plots = [None]*layout[0]
		layout = [shape[0]*shape[1],2]		
		plots = [[None]*layout[1]]*layout[0]
		figsize = (20,20)
		iterations = list(sorted(list(set([max(0,min(size-1,i))
							for i in [
							0,
							*[5,10,15,20],
							*[i*(size-1)//n 
							for n in [4] for i in range(1,n+1)]]
							]))))
		labels = [r'\alpha',r'\phi']
		lims = [[-0.1,1.1],[-0.1,1.1]]
		# lims = [[None,None],[None,None]]

		iteration = 0
		if iteration >= size:
			parameters0 = parameters
		else:
			parameters0 = hyperparameters['hyperparameters']['track'][attr][iteration]
		parameters0 = self.__layers__(parameters0,layer)

		parameters0 = parameters0[indices]


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

			for iteration in iterations:

				if iteration >= size:
					parameters = parameters
				else:
					parameters = hyperparameters['hyperparameters']['track'][attr][iteration]
				parameters = self.__layers__(parameters,layer)

				parameters = parameters[indices]

				for i in range(shape[0]):
					for j in range(shape[1]):
						for k in range(layout[1]):

							index = [i*shape[1] + j,k]

							x = arange(shape[-1])
							y = parameters[i][j]

							if layout[1] > 1:
								if index[1] == 0:
									y = parameters[i][j]
								elif index[1] == 1:
									y0 = parameters0[i][j]
									y = abs((y - y0)/(y0+1e-20))	
							else:
								y0 = parameters0[i][j]
								y = abs((y - y0)/(y0+1e-20))

							label = labels[i]

							plots[index[0]][index[1]] = ax[index[0]][index[1]].plot(x,y,
								color=getattr(plt.cm,'viridis')((iterations.index(iteration)*10)/(len(iterations)*10)),
								marker='',alpha=0.45,linewidth=4,zorder=max(iterations)+1-iteration,
								# label=r'${%s}^{(%s)}_{%s}$'%(label,str(iteration),str(j) if shape[1]>1 else '')
								label=r'${%s}^{(%s)}_{%s}$'%(r'\theta',str(iteration),'')
							)

							ax[index[0]][index[1]].set_xlim(xmin=0,xmax=shape[-1])
							ax[index[0]][index[1]].set_ylim(ymin=lims[i][0],ymax=lims[i][1])
							ax[index[0]][index[1]].set_ylabel(ylabel=r'${%s}_{%s}$'%(label,str(j) if shape[1]>1 else ''))
							ax[index[0]][index[1]].set_xlabel(xlabel=r'$\textrm{%s}$'%('Time'))
							ax[index[0]][index[1]].set_yscale(value='linear')
							# ax[index[0]][index[1]].set_yscale(value='log')
							ax[index[0]][index[1]].grid(True,zorder=0)	


							if i == 0 and j == 0:
								if layout[1] > 1:
									if index[1] == 0:
										ax[index[0]][index[1]].legend(
											loc=(0.2,1.15),ncol=min(5,len(ax[index[0]][index[1]].get_legend_handles_labels()[0]))
											)
									if index[1] == 0:
										ax[index[0]][index[1]].set_title(label=r'${%s}^{(%s)}_{%s}$'%(
												r'\theta','i',''))
									elif index[1] == 1:
										ax[index[0]][index[1]].set_title(label=r'$\abs{({%s}^{(%s)}_{%s} - {%s}^{(%s)}_{%s})/{%s}^{(%s)}_{%s}}$'%(
												r'\theta','i','',r'\theta','0','',r'\theta','0',''))
								else:
									ax[index[0]][index[1]].legend(
										loc=(0.15,1.1),ncol=min(4,len(ax[index[0]][index[1]].get_legend_handles_labels()[0]))
										)

			fig.set_size_inches(*figsize)
			fig.subplots_adjust()
			fig.tight_layout()
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
			y = hyperparameters['hyperparameters']['track'][attr]

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
		dtype = self.dtype
		basis = {
			'I': array([[1,0],[0,1]],dtype=dtype),
			'X': array([[0,1],[1,0]],dtype=dtype),
			'Y': array([[0,-1j],[1j,0]],dtype=dtype),
			'Z': array([[1,0],[0,-1]],dtype=dtype),
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

		# Get Trotterized order of p copies of data for products of data
		data = trotter(data,self.p)

		# Update class attributes
		self.__extend__(data,operator,site,string,interaction,hyperparameters)

		# Initialize parameters
		self.__initialize__(parameters)

		return


	@partial(jit,static_argnums=(0,))
	def __parameters__(self,parameters):
		''' 
		Setup parameters
		Args:
			parameters (array): parameters
		Returns:
			variables (array): variables
		'''		

		layer = 'variables'
		variables = self.__layers__(parameters,layer)

		# Get Trotterized order of copies of variables
		p = self.p
		variables = trotter(variables,p)

		# Get reshaped variables (transpose for shape (K,M) to (M,K) and reshape to (MK,) with periodicity of data)
		variables = variables.T.ravel()
		
		return variables





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
		
		# Get shape and indices of variable variables for gradient
		attribute = 'shape'
		layer = 'variables'
		shape = attributes[attribute][layer]

		attribute = 'index'
		layer = 'variables'
		indices = attributes[attribute][layer]		

		print(shape,indices)
		exit()

		# Calculate parameters and gradient
		parameters = self.__parameters__(parameters)

		grad = gradient_expm(-1j*self.coefficients*parameters,self.data,self.identity)
		grad *= -1j*self.coefficients

		# Reshape gradient

		# Axis of trotterization
		axis = 0

		grad = grad.transpose(axis,0,*[i for i in range(grad.ndim) if i not in [0,axis]])


		grad = grad.reshape((*_shape[:1],-1,*self.shape[2:]))
		grad = grad[index[0]]
		grad = grad.reshape((shape[0],-1,*self.shape[2:]))

		grad = gradient_trotter(grad,self.p)
		grad = grad[index[axis]]
		grad = grad.transpose(axis,0,*[i for i in range(grad.ndim) if i not in [0,axis]])
		
		grad = grad.reshape((-1,*grad.shape[2:]))

		derivative = grad

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


def initialize(parameters,shape,hyperparameters,reset=None,layer=None,slices=None,shapes=None,dtype=None):
	'''
	Initialize parameters
	Args:
		parameters (array): parameters array
		shape (iterable): shape of parameters
		hyperparameters (dict): hyperparameters for initialization
		reset (bool): Overwrite existing parameters
		layer (str): Layer type of parameters
		slices (iterable): slices of array within containing array
		shapes (iterable): shape of containing array of parameters
		dtype (str,datatype): data type of parameters		
	Returns:
		out (array): initialized slice of parameters
	'''	

	# Initialization hyperparameters
	layer = 'parameters' if layer is None else layer
	bounds = hyperparameters['bounds'][layer]
	constant = hyperparameters['constants'][layer]	
	initialization = hyperparameters['initialization']
	random = hyperparameters['random']
	pad = hyperparameters['pad']
	seed = hyperparameters['seed']
	key = seed

	# Parameters shape and bounds
	shape = shape
	ndim = len(shape)

	if shapes is None:
		shapes = shape

	if slices is None:
		slices = tuple([slice(0,shapes[axis],1) for axis in range(ndim)])

	bounds = [to_number(i,dtype) for i in bounds]
	if not any(isnaninf(i) for i in bounds):
		bounds = [
			bounds[0] + (bounds[1]-bounds[0])*hyperparameters['init'][0],
			bounds[1]*hyperparameters['init'][1],
		]

	# Add random padding of values if parameters not reset
	if not reset:
		parameters = padding(parameters,shape,key=None,bounds=bounds,random=pad)
	else:
		if initialization in ["interpolation"]:
			# Parameters are initialized as interpolated random values between bounds
			interpolation = hyperparameters['interpolation']
			smoothness = min(shape[-1]//2,hyperparameters['smoothness'])
			shape_interp = (*shape[:-1],shape[-1]//smoothness+2)
			pts_interp = smoothness*arange(shape_interp[-1])
			pts = arange(shape[-1])

			parameters_interp = rand(shape_interp,key=key,bounds=bounds,random=random)

			for axis in range(ndim):
				for i in constant[axis]:
					j = shapes[axis] + i if i < 0 else i
					if j >= slices[axis].start and j < slices[axis].stop:
						indices = tuple([slice(None) if ax != axis else i for ax in range(ndim)])
						value = constant[axis][i]			
						parameters_interp = parameters_interp.at[indices].set(value)

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
	length = len(keys)

	if length == 0:
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
			'conditions': (lambda hyperparameters: hyperparameters['hyperparameters'].get('scale') is not None)
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
			'value': (lambda parameter,hyperparameters: {attr: [{int(j):i[j] for j in i} for i in hyperparameters[section][parameter]['boundaries'][attr]] for attr in hyperparameters[section][parameter]['boundaries']}),
			'default': (lambda parameter,hyperparameters: {}),
			'conditions': (lambda parameter,hyperparameters: True)				
		},
		'constants': {
			'value': (lambda parameter,hyperparameters: {attr: [{int(j):i[j] for j in i} for i in hyperparameters[section][parameter]['constants'][attr]] for attr in hyperparameters[section][parameter]['constants']}),
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
			'conditions': (lambda parameter,hyperparameters: hyperparameters['hyperparameters'].get('locality') is not None and hyperparameters['parameters'][parameter]['category'] in ['variable'])
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
		for parameter in settings['hyperparameters'][key]['parameters']:
			settings['hyperparameters'][key]['parameters'][parameter]['seed'] = settings['seed'][key]

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

		hyperparameters = settings['hyperparameters'][key]	
		
		if settings['boolean']['load']:
			default = hyperparameters
			def func(key,iterable,elements): 
				i = iterable.get(key)
				e = elements.get(key,i)
				return e if not callable(i) else i
			path = hyperparameters['sys']['path']['data']['data']
			data = load(path,default=default)
			_update(hyperparameters,data,_func=func)

		if settings['boolean']['dump']:
			data = copy.deepcopy(hyperparameters)
			path = hyperparameters['sys']['path']['config']['settings'] 
			dump(data,path,callables=False)


		obj = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

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
			hyperparameters = obj.hyperparameters			
			data = copy.deepcopy(hyperparameters)
			path = hyperparameters['sys']['path']['data']['data'] 
			dump(data,path)

	if settings['boolean']['plot']:
		hyperparameters = settings['hyperparameters']
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