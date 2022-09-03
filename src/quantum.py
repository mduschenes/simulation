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

from src.utils import jit,gradient,hessian,gradient_finite,gradient_shift,gradient_fwd
from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,tensordot,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product,rank,einsum
from src.utils import summation,exponentiation
from src.utils import trotter,gradient_trotter
from src.utils import gradient_expm,gradient_sigmoid
from src.utils import normed,inner_abs2,inner_real,inner_imag
from src.utils import gradient_normed,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
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

from src.system import System,Logger,Space,Time,Lattice,Metric

from src.process import process

from src.plot import plot

from src.optimize import Optimizer,Objective

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
		metric (str,Metric): Type of metric
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={},
		N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,space=None,time=None,lattice=None,metric=None,system=None):

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
		self.metric = metric
		self.system = system

		self.data = array([])
		self.operator = []
		self.site = []
		self.string = []
		self.interaction = []
		self.indices = []		
		self.shape = (*self.data.shape[:1],self.M,*self.data.shape[1:])

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
		self.index = arange(len(self.data))

		self.hyperparameters = hyperparameters
		self.parameters = None
		self.label = None
		self.states = None
		self.attributes = {}
		self.constants = None
		self.coefficients = 1
		self.size = None		

		self.fig = {}
		self.ax = {}

		self.__system__()
		self.__space__()
		self.__time__()
		self.__lattice__()
		self.__metric__()
		self.__logger__()

		self.__check__()

		self.__setup__(data,operator,site,string,interaction,hyperparameters)

		self.func = self.__func__
		self.grad = jit(gradient(self.func))
		self.derivative = jit(gradient_fwd(self))
		self.hessian = jit(hessian(self.func))
		# self.einsum = jit(einsum('ia,ic,uab,vbc->uv',*[self.states.shape]*2,*[self.shape[-2:]]*2))

		self.log('%s\n'%('\n'.join(['%s: %s'%(attr,getattr(self,attr)) 
			for attr in ['key','N','D','d','L','delta','M','tau','T','p','seed','metric','architecture','shape']]
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
			if any(data[name]['string'] in group 
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


		def setup(hyperparameters,cls=None):
			'''
			Check hyperparameters
			Args:
				hyperparameters (dict): Hyperparameters
				cls (object): Class instance to update hyperparameters		
			'''

			# Update with class attributes
			sections = ['model']
			if cls is not None:
				for section in sections:
					if section not in hyperparameters:
						hyperparameters[section] = {}
					hyperparameters[section].update({attr: getattr(cls,attr) for attr in cls.__dict__ 
						if attr in hyperparameters[section] and isinstance(getattr(cls,attr),scalars)})

			return



		self.hyperparameters.update(hyperparameters)

		# Set hyperparameters
		default = {}
		if self.hyperparameters is None:
			self.hyperparameters = default
		elif isinstance(self.hyperparameters,str):
			self.hyperparameters = load(self.hyperparameters,default=default)

		# Set defaults
		path = 'config/settings.json'
		default = {}
		func = lambda key,iterable,elements: iterable.get(key,elements[key])
		updater(self.hyperparameters,load(path,default=default),func=func)

		setup(self.hyperparameters,cls=self)

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
			index = len(self.data)

		self.data = array([*self.data[:index],data,*self.data[index:]],dtype=self.dtype)
		self.operator.insert(index,operator)
		self.site.insert(index,site)
		self.string.insert(index,string)
		self.interaction.insert(index,interaction)

		self.shape = (*self.data.shape[:1],self.M,*self.data.shape[1:])
		self.index = arange(len(self.data))

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
		shape = (len(self.data)//self.p,self.M)
		hyperparams = hyperparameters['parameters']
		check = lambda group,index,axis,site=self.site,string=self.string: (
			(axis != 0) or 
			any(g in group for g in [string[index],'_'.join([string[index],''.join(['%d'%j for j in site[index]])])]))
		dtype = self._dtype

		attributes = parameterize(data,shape,hyperparams,check=check,initialize=initialize,dtype=dtype,cls=self)


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

		label = operatorize(data,shape,hyperparams,size=size,dtype=dtype,cls=self)

		# Get states
		data = None
		shape = [-1,*self.shape[2:]]
		hyperparams = hyperparameters['state']
		size = self.N
		dtype = self.dtype

		states = stateize(data,shape,hyperparams,size=size,dtype=dtype,cls=self)


		# Update class attributes
		self.parameters = parameters
		self.label = label
		self.states = states
		self.hyperparameters = hyperparameters
		self.attributes = attributes
		self.size = parameters.shape

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
		return 1-self.metric(self(parameters),self.label)

	@partial(jit,static_argnums=(0,))
	def __loss__(self,parameters):
		''' 
		Setup loss
		Args:
			parameters (array): parameters
		Returns:
			loss (array): loss
		'''	
		return self.metric(self(parameters),self.label)

	@partial(jit,static_argnums=(0,))
	def __func__(self,parameters):
		''' 
		Class objective
		Args:
			parameters (array): parameters
		Returns:
			objective (array): objective
		'''	
		return self.__loss__(parameters) + self.__constraints__(parameters)

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

	@partial(jit,static_argnums=(0,))
	def __derivative_analytical__(self,parameters):
		'''
		Return gradient of parameterized operator expm(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			derivative (array): Gradient of parameterized operator
		'''
		return self.derivative(parameters)


	@partial(jit,static_argnums=(0,))
	def __grad_analytical__(self,parameters):
		''' 
		Class gradient of objective
		Args:
			parameters (array): parameters
		Returns:
			grad (array): gradient of objective
		'''	

		grad = self.metric.__grad__(self(parameters),self.label,self.__derivative_analytical__(parameters))

		return grad


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

		# fisher = 0
		# G = self.derivative(parameters)

		# for state in self.states:
		# 	fisher += ((G.dot(state).conj()).dot((G.dot(state)).T) - 
		# 				outer((G.dot(state).conj()).dot(state),
		# 				  (G.dot(state).conj()).dot(state).conj())
		# 			)
		# fisher = fisher.real

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
		self.shape = (*self.shape[:1],self.M,*self.shape[2:])	

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


	def __metric__(self,metric=None,system=None):
		'''
		Set metric attributes
		Args:
			metric (str,Metric): Type of metric
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)		
		'''		
		metric = self.metric if metric is None else metric
		system = self.system if system is None else system

		self.metric = Metric(metric,system)	

		return


	def __logger__(self,hyperparameters=None):
		'''
		Setup logger
		Args:
			hyperparameters (dict): Hyperparameters
		'''

		hyperparameters = self.hyperparameters if hyperparameters is None else hyperparameters

		path = hyperparameters['sys']['cwd']
		root = path

		name = __name__
		conf = join(hyperparameters['sys']['path']['config']['logger'],root=root)
		file = join(hyperparameters['sys']['path']['data']['log'],root=root)

		self.logger = Logger(name,conf,file=file)

		return

	def __str__(self):
		size = len(self.data)
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
		return len(self.data)

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

	def dump(self,path=None):
		'''
		Save class data		
		Args:
			path (str): Path to dump class data
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
					New = New.mean(-1)
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

		# Set path
		if path is None:
			path = self.hyperparameters['sys']['cwd']
		root = path

		# Dump data
		path = join(self.hyperparameters['sys']['path']['data']['data'],root=root)
		dump(data,path)
		
		# Dump hyperparameters
		path = join(self.hyperparameters['sys']['path']['data']['model'],root=root)
		hyperparameters = self.hyperparameters
		dump(hyperparameters,path)

		return

	def load(self,path=None):
		'''
		Load class data		
		Args:
			path (str): Path to load class data
		'''

		# TODO: Determine which loaded hyperparameters should have precedence over new hyperparameters

		def func(key,iterable,elements): 
			types = (list,)
			i = iterable.get(key,elements.get(key))
			e = elements.get(key,i)
			return e if isinstance(e,types) else i
		
		# Set path
		if path is None:
			path = self.hyperparameters['sys']['cwd']
		root = path

		# Load data
		path = join(self.hyperparameters['sys']['path']['data']['model'],root=root)
		default = self.hyperparameters
		hyperparameters = load(path,default=default)
		updater(self.hyperparameters,hyperparameters,func=func)
		return


	def plot(self,path=None):
		'''
		Plot class
		Args:
			path (str,dict): Path to plot class, or dictionary of hyperparameters to plot
		Returns:
			fig (dict,matplotlib.figure): Plot figures
			ax (dict,matplotlib.axes): Plot axes
		'''
		# Set path
		instances = isinstance(path,dict)
		if path is None:
			key = None
			hyperparameters = {key: self.hyperparameters}
			path = self.hyperparameters['sys']['cwd']
		elif isinstance(path,str):
			key = None
			hyperparameters = {key: self.hyperparameters}			
			path = path
		elif isinstance(path,dict):
			key = None
			hyperparameters = {key: path[key] for key in path}
			path = self.hyperparameters['sys']['cwd']

		root = path

		# Get paths and kwargs
		paths = {
			'data':('sys','path','data','data'),
			'settings':('sys','path','config','plot'),
			'hyperparameters':('sys','path','config','process'),
			}

		kwargs = {kwarg: [] for kwarg in paths}


		for kwarg in kwargs:
			for key in hyperparameters:
				path = hyperparameters[key]
				for i in paths[kwarg]:
					path = path[i]
				kwargs[kwarg].append(path)

		fig,ax = process(**kwargs)

		if not instances:
			self.fig,self.ax = fig,ax

		return fig,ax


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
		metric (str,Metric): Type of metric
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={},
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,space=None,time=None,lattice=None,metric=None,system=None):
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,hyperparameters=hyperparameters,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,p=p,space=space,time=time,lattice=lattice,metric=metric,system=system)
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
			if any(data[name]['string'] in group 
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

		parameters = self.parameters

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
		metric (str,Metric): Type of metric		
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,architecture,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={},
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,space=None,time=None,lattice=None,metric=None,system=None):
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,hyperparameters=hyperparameters,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,p=p,space=space,time=time,lattice=lattice,metric=metric,system=system)
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
	def __derivative_analytical__(self,parameters):
		'''
		Return gradient of parameterized operator expm(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			derivative (array): Gradient of parameterized operator
		'''

		# Get class hyperparameters and attributes
		hyperparameters = self.hyperparameters
		attributes = self.attributes

		# Get shape and indices of variable parameters for gradient
		attribute = 'shape'
		layer = 'variables'
		shape = attributes[attribute][layer]

		# Get trotterized shape
		p = self.p
		shape = list(shape[::-1])

		shape[-1] *= p

		ndim = len(shape)

		attribute = 'slice'
		layer = 'variables'
		slices = attributes[attribute][layer]		

		attribute = 'index'
		layer = 'variables'
		indices = attributes[attribute][layer]		

		slices = tuple([
			*[[slices[parameter][group][axis] for parameter in slices for group in slices[parameter]][0]
				for axis in range(0,1)],
			*[[slices[parameter][group][axis] for parameter in slices for group in slices[parameter]][0]
				for axis in range(1,ndim)]
		])

		indices = tuple([
			*[[i for parameter in indices for group in indices[parameter] for i in indices[parameter][group][axis]]
				for axis in range(0,1)],
			*[[indices[parameter][group][axis] for parameter in indices for group in indices[parameter]][0]
				for axis in range(1,ndim)]
		])

		# Calculate parameters and gradient
		parameters = self.__parameters__(parameters)

		derivative = gradient_expm(-1j*self.coefficients*parameters,self.data,self.identity)
		derivative *= -1j*self.coefficients*2*pi

		# Reshape gradient
		axis = 1

		derivative = derivative.reshape((*shape,*derivative.shape[axis:]))

		derivative = derivative.transpose(axis,0,*[i for i in range(derivative.ndim) if i not in [0,axis]])

		derivative = array(gradient_trotter(derivative,p))

		derivative = derivative[indices]

		derivative = derivative.reshape((-1,*derivative.shape[axis+1:]))

		return derivative


