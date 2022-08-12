#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime
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

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import logconfig
from src.utils import jit,gradient,hessian,gradient_finite,gradient_fwd
from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,tensordot,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product,rank,einsum
from src.utils import summation,exponentiation
from src.utils import inner_abs2,inner_real,inner_imag
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
from src.utils import eigh,qr
from src.utils import maximum,minimum,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,isclose,is_naninf,to_key_value 
from src.utils import initialize,parse,to_str,to_number,scinotation,datatype,slice_size
from src.utils import pi,e,nan,delim,scalars,nulls
from src.utils import itg,flt,dbl

from src.dictionary import updater,getter,setter,permuter
from src.dictionary import leaves,counts,plant,grow

from src.parameters import parameterize
from src.operators import operatorize
from src.states import stateize

from src.io import load,dump,join,split

from src.process import process

from src.plot import plot

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
		timestamp (str): timestamp for class
		architecture (str): architecture for class
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
				},
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
		kwargs (dict): Additional arguments
	'''
	def __init__(self,name,conf,file=None,verbose=True,**kwargs):
		defaults = {
			'handler_file.formatter.file.args':file
		}
		kwargs.update({kwarg: defaults[kwarg] for kwarg in defaults if kwarg not in kwargs and defaults[kwarg] is not None})

		if isinstance(name,str):
			try:
				self.logger =  logconfig(name,conf=conf,**kwargs)
			except:
				self.logger = logging.getLogger(name)
		else:
			self.logger = name

		return
	
	def log(self,verbose,msg):
		'''
		Log messages
		Args:
			verbose (int): Verbosity of message
			msg (str): Message to log
		'''
		self.logger.log(verbose,msg)
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
		self.L = L if L is not None or delta is not None else 1
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
		assert self.L is not None or self.delta is not None, 'Either L or delta must not be None'		
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
		self.M = M if M is not None else 1
		self.T = T if T is not None else None
		self.tau = tau if tau is not None and T is not None else 1
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
					sites = [list(i) for i in unique(
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
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
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
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)
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
		self.indices = []		
		self.size = 0
		self.shape = (self.size,self.M,*self.data.shape[1:])

		self.key = None

		self.timestamp = None
		self.architecture = None
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
		self.label = None
		self.states = None
		self.attributes = {}
		self.constants = None
		self.coefficients = 1

		self.fig = {}
		self.ax = {}

		self.__system__()
		self.__space__()
		self.__time__()
		self.__lattice__()
		self.__logger__()

		self.__check__()

		self.__setup__(data,operator,site,string,interaction,hyperparameters)

		self.func = self.__func__
		self.grad = jit(gradient(self.func))
		self.derivative = jit(gradient_fwd(self))
		self.hessian = jit(hessian(self.func))
		# self.einsum = jit(einsum('ia,ic,uab,vbc->uv',*[self.states.shape]*2,*[self.shape[-2:]]*2))
		self.metric = jit({'abs2':lambda a,b: 1-inner_abs2(a,b),'real':lambda a,b: 1-inner_real(a,b)}['abs2'])

		self.log('%s\n'%('\n'.join(['%s: %s'%(attr,getattr(self,attr)) 
			for attr in ['key','N','D','d','M','tau','T','p','seed','architecture']]
			)))

		return	

	def __setup__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={}):
		'''
		Setup class
		Args:
			data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
				string (iterable[str]): string labels of operators
				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			hyperparameters (str,dict): Class hyperparameters, or path to load hyperparameters
		'''

		# Update data
		if operator is None:
			operator = []
		if site is None:
			site = []
		if string is None:
			string = []
		if interaction is None:
			interaction = []

		names = [name for name in data 
			if any(name in group 
				for parameter in hyperparameters['parameters'] 
				for group in hyperparameters['parameters'][parameter]['group'] 
				if hyperparameters['parameters'][parameter]['use'])]

		operator.extend([data[name]['operator'] for name in names])
		site.extend([data[name]['site'] for name in names])
		string.extend([data[name]['string'] for name in names])
		interaction.extend([data[name]['interaction'] for name in names])

		size = min([len(i) for i in [operator,site,string,interaction]])

		data = [self.identity.copy() for i in range(size)]

		parameters = None

		# Update class attributes
		self.__extend__(data,operator,site,string,interaction,hyperparameters)
		
		# Initialize parameters
		self.__initialize__(parameters)

		return


	def __check__(self,hyperparameters={}):
		'''
		Setup class hyperparameters
		Args:
			hyperparameters (str,dict): Class hyperparameters, or path to load hyperparameters
		Returns:
			hyperparameters (dict): Class hyperparameters
		'''

		path = 'config/settings.json'
		default = {}

		# Set hyperparameters
		if hyperparameters is None:
			hyperparameters = default
		elif isinstance(hyperparameters,str):
			hyperparameters = load(hyperparameters,default=default)

		func = lambda key,iterable,elements: iterable.get(key,elements[key])
		updater(hyperparameters,load(path,default=default),func=func)

		setup(hyperparameters,cls=self)

		self.hyperparameters.update(hyperparameters)

		return


	def __append__(self,data,operator,site,string,interaction,hyperparameters={}):
		'''
		Append to class
		Args:
			data (array): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			
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
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
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
			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
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
		self.shape = (self.size,self.M,*self.data.shape[1:])
		self.index = arange(self.size)

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

		# Get attributes data of parameters of the form {attribute:{parameter:{group:{layer:[]}}}
		data = None
		shape = (self.size//self.p,self.M)
		hyperparams = hyperparameters['parameters']
		check = lambda group,index,axis,site=self.site,string=self.string: (
			(axis != 0) or 
			any(g in group for g in [string[index],'_'.join([string[index],''.join(['%d'%j for j in site[index]])])]))
		dtype = self._dtype

		attributes = parameterize(data,shape,hyperparams,check=check,initialize=initialize,dtype=dtype)


		# Get reshaped parameters
		attribute = 'values'
		layer = 'parameters'
		parameters = attributes[attribute][layer]

		parameters = parameters.ravel()


		# Get label
		data = None
		shape = self.shape[2:]
		hyperparams = hyperparameters['label']
		size = self.N
		dtype = self.dtype

		label = operatorize(data,shape,hyperparams,size=size,dtype=dtype)

		# Get states
		data = None
		shape = [-1,*self.shape[2:]]
		hyperparams = hyperparameters['state']
		size = self.N
		dtype = self.dtype

		states = stateize(data,shape,hyperparams,size=size,dtype=dtype)


		# Update class attributes
		self.parameters = parameters
		self.label = label
		self.states = states
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
			parameters (array): parameters
		'''
		self.parameters = parameters
		return parameters


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
		values = attributes[attribute][layer]

		# Get values
		attribute = 'slice'
		for parameter in attributes[attribute][layer]:
			for group in attributes[attribute][layer][parameter]:

				attr = layer
				func = attributes[attr][layer][parameter][group]
				
				attr = 'slice'
				slices = attributes[attr][layer][parameter][group]
				
				attr = 'index'
				indices = attributes[attr][layer][parameter][group]

				values = func(parameters,values,slices,indices)

		return values

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
		return 1-distance(self(parameters),self.label)

	@partial(jit,static_argnums=(0,))
	def __loss__(self,parameters):
		''' 
		Setup loss
		Args:
			parameters (array): parameters
		Returns:
			loss (array): loss
		'''	
		return distance(self(parameters),self.label)

	@partial(jit,static_argnums=(0,))
	def __func__(self,parameters):
		''' 
		Class objective
		Args:
			parameters (array): parameters
		Returns:
			objective (array): objective
		'''	
		return self.__loss__(parameters) #+ self.__constraints__(parameters)

	@partial(jit,static_argnums=(0,))
	def __grad__(self,parameters):
		''' 
		Class gradient of objective
		Args:
			parameters (array): parameters
		Returns:
			grad (array): gradient of objective
		'''	

		return self.grad(parameters)

		# shape = parameters.shape
		# grad = zeros(shape)

		# grad = grad + gradient_distance(self(parameters),self.label,self.__derivative__(parameters))

		# return grad


	@partial(jit,static_argnums=(0,))
	def __hessian__(self,parameters):
		''' 
		Class hessian of objective
		Args:
			parameters (array): parameters
		Returns:
			grad (array): hessian of objective
		'''	

		return self.hessian(parameters)


	def __callback__(self,parameters):
		''' 
		Setup callback and logging
		Args:
			parameters (array): parameters
		Returns:
			status (int): status of class
		'''	


		self.hyperparameters['optimize']['track']['objective'].append(
			# 1-self.hyperparameters['optimize']['track']['value'][-1] + self.__constraints__(parameters)
			self.__objective__(parameters)
			)

		# self.hyperparameters['optimize']['track']['hessian'].append(
		# 	rank(self.__hessian__(parameters))
		# 	)	

		# fisher = einsum()

		fisher = 0
		G = self.derivative(parameters)

		for state in self.states:
			fisher += ((G.dot(state).conj()).dot((G.dot(state)).T) - 
						outer((G.dot(state).conj()).dot(state),
						  (G.dot(state).conj()).dot(state).conj())
					)
		fisher = fisher.real

		# self.hyperparameters['optimize']['track']['fisher'].append(
		# 	rank(fisher)
		# 	)

		if self.hyperparameters['optimize']['track']['iteration'][-1]%self.hyperparameters['optimize']['modulo']['log'] == 0:			

			self.hyperparameters['optimize']['track']['parameters'].append(parameters)		

			msg = '\n'.join([
				'%d f(x) = %0.10f'%(
					self.hyperparameters['optimize']['track']['iteration'][-1],
					self.hyperparameters['optimize']['track']['objective'][-1],
				),
				'|x| = %0.4e\t\t|grad(x)| = %0.4e'%(
					norm(self.hyperparameters['optimize']['track']['parameters'][-1])/
						 self.hyperparameters['optimize']['track']['parameters'][-1].size,
					norm(self.hyperparameters['optimize']['track']['grad'][-1])/
						 self.hyperparameters['optimize']['track']['grad'][-1].size,
				),
				'\t\t'.join([
					'%s = %0.4e'%(attr,self.hyperparameters['optimize']['track'][attr][-1])
					for attr in ['alpha','beta','hessian','fisher']
					if attr in self.hyperparameters['optimize']['track'] and len(self.hyperparameters['optimize']['track'][attr])>0
					]),
				'U\n%s\nV\n%s\n'%(
				to_str(abs(self(parameters)).round(4)),
				to_str(abs(self.label).round(4))),

				])


			self.log(msg)

			# print(self.__layers__(parameters,'variables').round(3))


		status = (
			(abs(self.hyperparameters['optimize']['track']['objective'][-1] - self.hyperparameters['optimize']['value']['objective']) > 
				 self.hyperparameters['optimize']['eps']['objective']*self.hyperparameters['optimize']['value']['objective']) and
			(norm(self.hyperparameters['optimize']['track']['grad'][-1] - self.hyperparameters['optimize']['value']['grad'])/self.hyperparameters['optimize']['track']['grad'][-1].size > 
				  self.hyperparameters['optimize']['eps']['grad'])
			)

		return status



	def __system__(self,system=None):
		'''
		Set system attributes
		Args:
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)		
		'''
		system = self.system if system is None else system
		
		self.system = System(system)		
		self.dtype = self.system.dtype
		self._dtype = datatype(self.dtype)
		self.format = self.system.format
		self.seed = self.system.seed
		self.key = self.system.key
		self.timestamp = self.system.timestamp
		self.architecture = self.system.architecture
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
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)		
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
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)		
		'''
		M = self.M if M is None else M
		T = self.T if T is None else T
		tau = self.tau if tau is None else tau
		p = self.p if p is None else p
		#time = self.time if time is None else time
		system = self.system if system is None else system

		self.time = Time(M,T,tau,p,time,system)		
		self.M = self.time.M
		self.T = self.time.T
		self.p = self.time.p
		self.tau = self.time.tau
		self.coefficients = self.tau/self.p		
		self.shape = (self.size,self.M,*self.shape[2:])	

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
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)		
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


	def __logger__(self,hyperparameters=None):
		'''
		Setup logger
		Args:
			hyperparameters (dict): Hyperparameters
		'''

		hyperparameters = self.hyperparameters if hyperparameters is None else hyperparameters

		name = __name__
		conf = hyperparameters['sys']['path']['config']['logger']
		file = hyperparameters['sys']['path']['data']['log']

		self.logger = Logger(name,conf,file=file)

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

	def log(self,msg,verbose=None):
		'''
		Log messages
		Args:
			msg (str): Message to log
			verbose (int): Verbosity of message			
		'''
		if verbose is None:
			verbose = self.verbose
		self.logger.log(verbose,msg)
		return	

	def dump(self):
		'''
		Save class data		
		'''

		# Process attribute values
		def func(attr,value,obj):
			'''
			Process attribute values
			Args:
				attr (str): Attribute to process
				value (object): Values to process
				obj (object): Class instance to process
			Returns:
				returns (dict): Processed values
			'''

			hyperparameters = obj.hyperparameters
			attributes = obj.attributes

			returns = {}

			if attr in obj.hyperparameters['optimize']['track']:
				new = attr
				# New = array(value)
				New = value			
				# if New.ndim > 1:
				# 	New = New.transpose(*range(1,New.ndim),0)
				returns[new] = New
			else:
				new = attr
				New = value
				returns[new] = New


			if attr in obj.hyperparameters['optimize']['track']:

				if attr in ['parameters']:
					
					layer = 'features'
					indices = tuple([(
						slice(
						min(attributes['index'][layer][parameter][group][axis].start
							for parameter in attributes['index'][layer] 
							for group in attributes['index'][layer][parameter]),
						max(attributes['index'][layer][parameter][group][axis].stop
							for parameter in attributes['index'][layer] 
							for group in attributes['index'][layer][parameter]),
						min(attributes['index'][layer][parameter][group][axis].step
							for parameter in attributes['index'][layer] 
							for group in attributes['index'][layer][parameter]))
						if all(isinstance(attributes['index'][layer][parameter][group][axis],slice)
							for parameter in attributes['index'][layer] 
							for group in attributes['index'][layer][parameter]) else
						list(set(i 
							for parameter in attributes['index'][layer] 
							for group in attributes['index'][layer][parameter] 
							for i in attributes['index'][layer][parameter][group][axis]))
						)
						for axis in range(min(len(attributes['index'][layer][parameter][group]) 
											for parameter in attributes['index'][layer] 
											for group in attributes['index'][layer][parameter]))
						])

					new = '%s.relative'%(attr)
					New = abs((obj.__layers__(value,layer)[indices] - 
						obj.__layers__(hyperparameters['optimize']['track'][attr][0],layer)[indices] + 1e-20)/(
						obj.__layers__(hyperparameters['optimize']['track'][attr][0],layer)[indices] + 1e-20))
					returns[new] = New

					new = '%s.relative.mean'%(attr)
					New = New.mean()
					returns[new] = New				

					new = attr
					New = obj.__layers__(value,layer)[indices]

					returns[new] = New

				elif attr in ['iteration']:
					new = '%s.max'%(attr)
					New = hyperparameters['optimize']['track'][attr][-1]
					returns[new] = New

					new = 'status'
					New = value/hyperparameters['optimize']['track'][attr][-1]

					returns[new] = New

				elif attr in ['objective']:
					new = 'infidelity'
					New = 1 - value
					returns[new] = New

			return returns

		# Get data
		label = [str(self.timestamp)]
		keys = [self.key]
		iterations = {
			key: range(min(len(self.hyperparameters['optimize']['track'][attr]) 
			for attr in self.hyperparameters['optimize']['track']
			if len(self.hyperparameters['optimize']['track'][attr]) > 0
			))
			for key in keys
			}

		data = {
			delim.join([*label,str(key),str(iteration)]): {
				**{attr: self.hyperparameters['optimize']['track'][attr][iteration]
					for attr in self.hyperparameters['optimize']['track']
					if len(self.hyperparameters['optimize']['track'][attr]) > 0
				},	
				**{attr: getattr(self,attr)
					for attr in self.__dict__
					if (
						(not callable(getattr(self,attr))) and
						(getattr(self,attr) is None or isinstance(getattr(self,attr),scalars))
						)
					}
				}
			for key in keys
			for iteration in iterations[key]
		}

		# Process data
		# data have axis of ('sample,model',*'value-shape','iterations') 
		# and must be transposed with func such that optimization tracked data has iterations as -1 axis
		for key in list(data):
			for attr in list(data[key]):			
				data[key].update(func(attr,data[key][attr],self))

		# Dump data
		path = join(self.hyperparameters['sys']['path']['data']['data'],root=self.hyperparameters['sys']['cwd'])
		dump(data,path)
		
		# Dump hyperparameters
		path = join(self.hyperparameters['sys']['path']['data']['model'],root=self.hyperparameters['sys']['cwd'])
		hyperparameters = self.hyperparameters	
		dump(hyperparameters,path)

		return

	def load(self):
		'''
		Load class data		
		'''
		def func(key,iterable,elements): 
			i = iterable.get(key,elements.get(key))
			e = elements.get(key,i)
			return e if not callable(i) else i
	
		path = join(self.hyperparameters['sys']['path']['data']['model'],root=self.hyperparameters['sys']['cwd'])
		default = self.hyperparameters
		hyperparameters = load(path,default=default)
		updater(self.hyperparameters,hyperparameters,func=func)
		return


	def plot(self):
		'''
		Plot class
		'''

		# Get paths and kwargs
		root = self.hyperparameters['sys']['cwd']
		paths = {
			'data':('sys','path','data','data'),
			'settings':('sys','path','config','plot'),
			'hyperparameters':('sys','path','config','process'),
			}

		hyperparameters = self.hyperparameters
		
		kwargs = {kwarg: [] for kwarg in paths}

		for kwarg in kwargs:
			path = hyperparameters
			for i in paths[kwarg]:
				path = path[i]
			path = join(path,root=root)

			kwargs[kwarg].append(path)

		self.fig,self.ax = process(**kwargs)
		return


class Hamiltonian(Object):
	'''
	Hamiltonian class of Operators
	Args:
		data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		operator (iterable[str]): string names of operators
		site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
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
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)
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
				site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
				string (iterable[str]): string labels of operators
				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			hyperparameters (dict) : class hyperparameters
		'''

		#time = timer()

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

		names = [name for name in data 
			if any(name in group 
				for parameter in hyperparameters['parameters'] 
				for group in hyperparameters['parameters'][parameter]['group'] 
				if hyperparameters['parameters'][parameter]['use'])]


		operator.extend([data[name]['operator'] for name in names])
		site.extend([data[name]['site'] for name in names])
		string.extend([data[name]['string'] for name in names])
		interaction.extend([data[name]['interaction'] for name in names])


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
					_site_ = deepcopy([dict(zip(indices,s if not isinstance(s,int) else [s])).get(j,parse(j,int)) for j in _site])
					_operator_ = deepcopy([_operator[_site_.index(j)] if j in _site_ else I for j in range(self.N)])
					_string_ = deepcopy(_string)
					_interaction_ = deepcopy(_interaction)
					
					operator.append(_operator_)
					site.append(_site_)
					string.append(_string_)
					interaction.append(_interaction_)
			else:
				_site_ = deepcopy(_site)
				_operator_ = deepcopy([_operator[_site_.index(j)] if j in _site_ else I for j in range(self.N)])
				_string_ = deepcopy(_string)
				_interaction_ = deepcopy(_interaction)

				operator.append(_operator_)
				site.append(_site_)
				string.append(_string_)
				interaction.append(_interaction_)


		# Form (size,n,n) shape operator from local strings for each data term
		data = array([tensorprod([basis[j] for j in i]) for i in operator])

		# Assert all data satisfies data**2 = identity for matrix exponentials
		assert all(allclose(d.dot(d),self.identity) for d in data), 'data is not involutory and data**2 != identity'

		# Get Trotterized order of p copies of data for products of data
		p = self.p
		data = array(trotter(data,p))
		operator = trotter(operator,p)
		site = trotter(site,p)
		string = trotter(string,p)
		interaction = trotter(interaction,p)

		# Check for case of size
		if not size:
			data = array([self.identity]*self.p)
			operator = [['I']*self.N]*self.p
			site = [list(range(self.N))]*self.p
			string = ['I']*self.p
			interaction = ['i...j']*self.p
		
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
			parameters (array): parameters
		'''		

		layer = 'variables'
		parameters = self.__layers__(parameters,layer)

		# Get Trotterized order of copies of parameters
		p = self.p
		parameters = array(trotter(parameters,p))

		# Get reshaped parameters (transpose for shape (K,M) to (M,K) and reshape to (MK,) with periodicity of data)
		parameters = parameters.T.ravel()
		
		return parameters





class Unitary(Hamiltonian):
	'''
	Unitary class of Operators
	Args:
		data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		operator (iterable[str]): string names of operators
		site (iterable[iterable[int,str]]): site of local operators, allowed strings in [['i'],['i','j']]
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
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
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)
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

		return self.derivative(parameters)

		# # Get class hyperparameters and attributes
		# hyperparameters = self.hyperparameters
		# attributes = self.attributes

		# # Get shape and indices of variable parameters for gradient
		# attribute = 'shape'
		# layer = 'variables'
		# shape = attributes[attribute][layer]

		# # Get trotterized shape
		# p = self.p
		# shape = list(shape[::-1])

		# shape[-1] *= p

		# ndim = len(shape)

		# attribute = 'slice'
		# layer = 'variables'
		# slices = attributes[attribute][layer]		

		# attribute = 'index'
		# layer = 'variables'
		# indices = attributes[attribute][layer]		

		# slices = tuple([
		# 	*[[slices[parameter][group][axis] for parameter in slices for group in slices[parameter]][0]
		# 		for axis in range(0,1)],
		# 	*[[slices[parameter][group][axis] for parameter in slices for group in slices[parameter]][0]
		# 		for axis in range(1,ndim)]
		# ])

		# indices = tuple([
		# 	*[[i for parameter in indices for group in indices[parameter] for i in indices[parameter][group][axis]]
		# 		for axis in range(0,1)],
		# 	*[[indices[parameter][group][axis] for parameter in indices for group in indices[parameter]][0]
		# 		for axis in range(1,ndim)]
		# ])



		# # Calculate parameters and gradient
		# parameters = self.__parameters__(parameters)

		# grad = gradient_expm(-1j*self.coefficients*parameters,self.data,self.identity)
		# grad *= -1j*self.coefficients*2*pi

		# # Reshape gradient
		# axis = 1

		# grad = grad.reshape((*shape,*grad.shape[axis:]))

		# grad = grad.transpose(axis,0,*[i for i in range(grad.ndim) if i not in [0,axis]])

		# grad = array(gradient_trotter(grad,p))

		# grad = grad[indices]

		# grad = grad.reshape((-1,*grad.shape[axis+1:]))

		# derivative = grad

		# return derivative


def bloch(state,path=None):
	'''
	Plot state on Bloch Sphere
	Args:
		state (array): States of shape (d,) or (n,d) or (n,d,d) for n, d dimensional states
		path (str,boolean): Path to save plot, or boolean to save
	Returns:
		fig (matplotlib.figure): Figure of plots
		ax (matplotlib.axes): Axes of plots
	'''
	
	import numpy as np
	import subprocess
	# import matplotlib
	# import matplotlib.pyplot as plt

	def coordinates(state):
		'''
		Convert state vector to Bloch vector
		Args:
			state (array): States of shape (d,) or (n,d) or (n,d,d)  for n, d dimensional states
		Returns:
			state (array): States of shape (1,d) or (n,d) or (n,d,d) for n, d dimensional states
		'''

		transformation = np.array([
			[[0, 1], [1, 0]],
			[[0, -1j], [1j, 0]],
			[[1, 0], [0, -1]]
			])

		ndim = state.ndim - 1

		if ndim == 0:
			state = np.tensordot(state.conj(),np.tensordot(transformation,state,([-1],[-1])),([-1],[-1])).reshape(1,-1)
		elif ndim == 1:
			state = np.array([np.tensordot(s.conj(),np.tensordot(transformation,s,([-1],[-1])),([-1],[-1])) for s in state])
		elif ndim == 2:
			state = np.array([np.tensordot(s.conj(),np.tensordot(transformation,s,([-1],[-1])),([-1],[-1])) for s in state])			
		else:
			pass
		return state

	try:
		mplstyle = '../src/plot.mplstyle'
		with matplotlib.style.context(mplstyle):
			try:
				try:
					fig = plt.figure(figsize=(6, 6))
					ax = fig.add_subplot(111, projection='3d')
					fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

					ax.grid(False)
					ax.set_axis_off()
					ax.view_init(30, 45)
					ax.dist = 7

					x, y, z = np.array([[-1.5,0,0], [0,-1.5,0], [0,0,-1.5]])
					u, v, w = np.array([[3,0,0], [0,3,0], [0,0,3]])
					ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05, color='black', linewidth=1.2)

					ax.text(0, 0, 1.7, r'$\ket{0}$', color='black', fontsize=16)
					ax.text(0, 0, -1.9, r'$\ket{1}$', color='black', fontsize=16)
					ax.text(1.9, 0, 0, r'$\ket{+}$', color='black', fontsize=16)
					ax.text(-1.7, 0, 0, r'$\ket{-}$', color='black', fontsize=16)
					ax.text(0, 1.7, 0, r'$\ket{i+}$', color='black', fontsize=16)
					ax.text(0,-1.9, 0, r'$\ket{i-}$', color='black', fontsize=16)

					state = coordinates(state)

					ax.scatter(state[:,0], state[:,1], state[:, 2], color=getattr(plt.cm,'viridis')(0.5),s=12,alpha=0.6)

					if path:
						if not isinstance(path,str):
							path = 'bloch.pdf'
						fig.savefig(path,bbox_inches='tight')

				except (subprocess.CalledProcessError, RuntimeError):
					pass
			except (subprocess.CalledProcessError, RuntimeError):
				pass
	except (subprocess.CalledProcessError, RuntimeError):
		pass
	return fig,ax

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
	# return 1-inner_real(a,b)#+inner_imag2(a,b)
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
	# return -gradient_inner_real(a,b,da)
	return -gradient_inner_abs2(a,b,da)
	

def trotter(a,p):
	'''
	Calculate p-order trotter series of iterable
	Args:
		a (iterable): Iterable to calculate trotter series
		p (int): Order of trotter series
	Returns:
		out (iterable): Trotter series of iterable
	'''	
	# return [v for u in [a[::i] for i in [1,-1,1,-1][:p]] for v in u]	
	return [u for i in [1,-1,1,-1][:p] for u in a[::i]]

def gradient_trotter(da,p):
	'''
	Calculate gradient of p-order trotter series of iterable
	Args:
		da (iterable): Gradient of iterable to calculate trotter series		
		p (int): Order of trotter series
	Returns:
		out (iterable): Gradient of trotter series of iterable
	'''	
	n = da.shape[0]//p
	return sum([da[:n][::i] if i>0 else da[-n:][::i] for i in [1,-1,1,-1][:p]])


def invtrotter(a,p):
	'''
	Calculate inverse of p-order trotter series of iterable
	Args:
		a (iterable): Iterable to calculate inverse trotter series
		p (int): Order of trotter series
	Returns:
		out (iterable): Inverse trotter series of iterable
	'''	
	n = a.shape[0]//p
	return a[:n]


def check(hyperparameters,cls=None):
	'''
	Check hyperparameters
	Args:
		hyperparameters (dict): Hyperparameters
		cls (object): Class instance to update hyperparameters		
	'''

	# Update with class attributes
	if cls is not None:
		section = 'model'
		if section not in hyperparameters:
			hyperparameters[section] = {}
		hyperparameters[section].update({attr: getattr(cls,attr) for attr in cls.__dict__ 
			if attr in hyperparameters[section]})


	section = 'parameters'
	updates = {
		# 'variables':{
		# 	'value':(lambda parameter,hyperparameters: variables),
		# 	'default':(lambda parameter,hyperparameters: {}),
		# 	'conditions': (lambda parameter,hyperparameters: hyperparameters[section][parameter].get('variables',{}) == {}),
		# },
		# 'features':{
		# 	'value':(lambda parameter,hyperparameters: features),
		# 	'default':(lambda parameter,hyperparameters: {}),
		# 	'conditions': (lambda parameter,hyperparameters: hyperparameters[section][parameter].get('features',{}) == {}),
		# },		
		# 'constraints':{
		# 	'value':(lambda parameter,hyperparameters: constraints),
		# 	'default': (lambda parameter,hyperparameters: {}),
		# 	'conditions': (lambda parameter,hyperparameters: hyperparameters[section][parameter].get('constraints',{}) == {}),
		# },
		# 'gradients':{
		# 	'value':(lambda parameter,hyperparameters: gradients),
		# 	'default':(lambda parameter,hyperparameters: {}),
		# 	'conditions': (lambda parameter,hyperparameters: hyperparameters[section][parameter].get('gradients',{}) == {}),			
		# },			
		# 'gradient_constraints':{
		# 	'value':(lambda parameter,hyperparameters: gradient_constraints),
		# 	'default': (lambda parameter,hyperparameters: {}),
		# 	'conditions': (lambda parameter,hyperparameters: hyperparameters[section][parameter].get('gradient_constraints',{}) == {}),
		# },
		'boundaries': {
			'value': (lambda parameter,hyperparameters: {attr: [{prop: array(i.get(prop,[])) for prop in ['slice','value']}
				for i in hyperparameters[section][parameter]['boundaries'][attr]] 
				for attr in hyperparameters[section][parameter]['boundaries']}),
			'default': (lambda parameter,hyperparameters: {}),
			'conditions': (lambda parameter,hyperparameters: True)				
		},
		'constants': {
			'value': (lambda parameter,hyperparameters: {attr: [{prop: array(i.get(prop,[])) for prop in ['slice','value']}
				for i in hyperparameters[section][parameter]['constants'][attr]] 
				for attr in hyperparameters[section][parameter]['constants']}),
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
			} for attr in ['scale','initialization','random','smoothness','interpolation','pad']
		},
		**{attr: {
			'value': (lambda parameter,hyperparameters,attr=attr: {
				**hyperparameters['parameters'][parameter][attr],
				**{kwarg: hyperparameters[section][kwarg] 
					for section in ['model'] 
					for kwarg in hyperparameters[section]
					if (
					isinstance(hyperparameters[section][kwarg],scalars) and 
					not isinstance(hyperparameters[section][kwarg],str)
					)}
				}),
			'default': (lambda parameter,hyperparameters,attr=attr: {}),
			'conditions': (lambda parameter,hyperparameters,attr=attr: True)						
			} for attr in ['kwargs']
		},		
		**{attr: {
			'value': (lambda parameter,hyperparameters,attr=attr: None),
			'default': (lambda parameter,hyperparameters,attr=attr: None),
			'conditions': (lambda parameter,hyperparameters,attr=attr: hyperparameters['parameters'][parameter].get(attr) is None)						
			} for attr in ['seed']
		},		
		'locality': {
			'value':(lambda parameter,hyperparameters: hyperparameters['hyperparameters']['locality']),
			'default':(lambda parameter,hyperparameters: None),
			'conditions': (lambda parameter,hyperparameters: hyperparameters['hyperparameters'].get('locality') is not None)
		},				
	}
	for parameter in hyperparameters[section]:
		for attr in updates:
			hyperparameters[section][parameter][attr] = hyperparameters[section][parameter].get(attr,updates[attr]['default'](parameter,hyperparameters))
			if updates[attr]['conditions'](parameter,hyperparameters):
				if callable(updates[attr]['value'](parameter,hyperparameters)):
					for group in hyperparameters[section][parameter]['group']:
							group = tuple(group)
							hyperparameters[section][parameter][attr][group] = jit(partial(updates[attr]['value'](parameter,hyperparameters),hyperparameters=hyperparameters,parameter=parameter,group=group))
				else:
					hyperparameters[section][parameter][attr] = updates[attr]['value'](parameter,hyperparameters)

	return



def setup(hyperparameters,cls=None):
	'''
	Setup hyperparameters
	Args:
		hyperparameters (dict): Hyperparameters
		cls (object): Class instance to update hyperparameters
	'''

	# Check hyperparameters have correct values
	check(hyperparameters,cls=cls)

	return