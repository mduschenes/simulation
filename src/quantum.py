#!/usr/bin/env python

# Import python modules
import os,sys
from copy import deepcopy
from functools import partial

from typing import List,Tuple

envs = {
	'JAX_PLATFORM_NAME':'cpu',
	'TF_CPP_MIN_LOG_LEVEL':5
}
for var in envs:
	os.environ[var] = str(envs[var])

import jax
import equinox as nn
import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)
# jax.set_cpu_device_count(8)

configs = {
	'jax_disable_jit':False,
	'jax_platforms':'cpu',
	'jax_enable_x64': True
	}
for name in configs:
	jax.config.update(name,configs[name])

# Logging
import logging

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,gradient,hessian,fisher
from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,tensordot,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product,dot,einsum
from src.utils import summation,exponentiation,summationv,exponentiationv,summationm,exponentiationm,summationc,exponentiationc,summationmc,exponentiationmc
from src.utils import trotter,gradient_trotter,gradient_expm,gradient_sigmoid
from src.utils import normed,inner_abs2,inner_real,inner_imag
from src.utils import gradient_normed,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
from src.utils import eig
from src.utils import maximum,minimum,argmax,argmin,difference,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,isclose,is_array,is_naninf,to_key_value 
from src.utils import initialize,parse,to_string,to_number,datatype,slice_size,intersection
from src.utils import pi,e,nan,delim,scalars,nulls
from src.utils import itg,flt,dbl

from src.dictionary import updater,getter,setter,permuter
from src.dictionary import leaves,counts,plant,grow

from src.parameters import parameterize
from src.operators import operatorize
from src.states import stateize
from src.noise import noiseize

from src.io import load,dump,join,split

from src.system import System,Logger,Space,Time,Lattice

from src.process import process

from src.plot import plot

from src.optimize import Optimizer,Metric

dtype = 'complex'
basis = {
	'I': array([[1,0],[0,1]],dtype=dtype),
	'X': array([[0,1],[1,0]],dtype=dtype),
	'Y': array([[0,-1j],[1j,0]],dtype=dtype),
	'Z': array([[1,0],[0,-1]],dtype=dtype),
}

class module(nn.Module):
	pass

class Object(object):
	'''
	Class for object
	Args:
		data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
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
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,backend,architecture,verbose)
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

		self.data = []
		self.operator = []
		self.site = []
		self.string = []
		self.interaction = []
		self.indices = []		
		self.shape = (len(self.data),self.M)
		self.ndim = len(self.shape)

		self.key = None

		self.timestamp = None
		self.backend = None
		self.architecture = None
		self.delimiter = ' '
		self.dims = ()
		self.dim = int(product(self.dims))
		self.ndims = len(self.dims)

		self.hyperparameters = hyperparameters
		self.parameters = None
		self.metrics = None
		self.labels = None
		self.label = None
		self.state = None
		self.noise = None
		self.attributes = {}
		self.identity = None
		self.constants = None
		self.coefficients = 1
		self.size = None	

		self.summation = None
		self.exponentiation = None 

		self.fig = {}
		self.ax = {}

		self.__system__()
		self.__space__()
		self.__time__()
		self.__lattice__()
		self.__logger__()

		self.__check__()

		self.__setup__(data,operator,site,string,interaction,hyperparameters)

		self.log('%s\n'%('\n'.join([
			*['%s: %s'%(attr,getattr(self,attr)) 
				for attr in ['key','N','D','d','L','delta','M','tau','T','p','seed','metric','backend','architecture','shape']
			],
			*['%s: %s'%(attr,getattr(self,attr) is not None) 
				for attr in ['state','noise']
			],
			*['%s: %s'%(attr,getattr(self,attr).__name__) 
				for attr in ['exponentiation']
			],			
			]
			)))
		return	

	def __setup__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={}):
		'''
		Setup class
		Args:
			data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
				string (iterable[str]): string labels of operators
				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
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

		# Setup functions
		self.__functions__()

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
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
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

		self.data.insert(index,data)
		self.operator.insert(index,operator)
		self.site.insert(index,site)
		self.string.insert(index,string)
		self.interaction.insert(index,interaction)

		self.shape = (len(self.data),self.M,*self.dims)
		self.ndim = len(self.shape)		

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
		size = product(shape)
		samples = None
		seed = self.seed
		cls = self
		dtype = self.dtype

		attributes = parameterize(data,shape,hyperparams,check=check,initialize=initialize,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)

		# Get reshaped parameters
		attribute = 'values'
		layer = 'parameters'
		parameters = attributes[attribute][layer]

		parameters = parameters.ravel()

		# Get states
		data = None
		shape = self.dims
		hyperparams = hyperparameters['state']
		size = self.N
		samples = True
		seed = self.seed		
		dtype = self.dtype
		cls = self

		state = stateize(data,shape,hyperparams,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)

		# Get label
		data = None
		shape = self.dims
		hyperparams = hyperparameters['label']
		size = self.N
		samples = None
		seed = self.seed
		cls = self
		dtype = self.dtype

		label = operatorize(data,shape,hyperparams,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)


		# Get noise
		data = None
		shape = self.dims
		hyperparams = hyperparameters['noise']
		size = self.N
		samples = None
		seed = self.seed		
		cls = self
		dtype = self.dtype

		noise = noiseize(data,shape,hyperparams,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)


		# Get coefficients
		coefficients = -1j*2*pi/2*self.tau/self.p		

		# Update class attributes
		self.parameters = parameters
		self.label = label
		self.state = state
		self.noise = noise
		self.coefficients = coefficients
		self.hyperparameters = hyperparameters
		self.attributes = attributes
		self.size = parameters.shape

		return

	def __functions__(self,state=None,noise=None,label=None,metric=None):
		''' 
		Setup class functions
		Args:
			state (bool,array): State to act on with class of shape self.dims, if boolean choose self.state or None
			noise (bool,array): Noise to act on with class of shape (-1,self.dims), if boolean choose self.noise or None
			label (bool,array): Label of class of shape self.dims, if boolean choose self.label or None
			metric (bool,callable): Metric for class and label, if boolean choose self.metric or None
		'''

		# Function arguments
		data = array(self.data,dtype=self.dtype)
		identity = self.identity
		state = self.state if (state is None or state is True) else state if state is not False else None
		noise = self.noise if (noise is None or noise is True) else noise if noise is not False else None
		label = self.label if (label is None or label is True) else label if label is not False else None
		metric = self.metric if (metric is None or metric is True) else metric if metric is not False else None

		# Metric functions
		self.metrics = Metric(metric,shapes=[self.dims,self.dims],optimize=None)	
		self.func = self.__func__
		self.grad = gradient(self.func)
		self.derivative = gradient(self,mode='fwd',move=True)
		self.hessian = hessian(self.func)
		self.fisher = fisher(self,self.derivative,shapes=[self.dims,(self.dim,*self.dims)])

		# Labels
		if state is None:
			self.labels = label.conj()
		elif state.ndim == 1:
			self.labels = einsum('ij,j->i',label,state).conj()
		elif state.ndim == 2:
			self.labels = einsum('ij,jk,lk->il',label,state,label.conj())
		else:
			self.labels = label.conj()


		# Operator functions
		if state is None and noise is None:
			self.summation = jit(summation,data=data,identity=identity)
			self.exponentiation = jit(exponentiation,data=data,identity=identity)
		elif state is not None and noise is None:
			if state.ndim == 1:
				self.summation = jit(summationv,data=data,identity=identity,state=state)
				self.exponentiation = jit(exponentiationv,data=data,identity=identity,state=state)
			elif state.ndim == 2:
				self.summation = jit(summationm,data=data,identity=identity,state=state)
				self.exponentiation = jit(exponentiationm,data=data,identity=identity,state=state)
			else:
				self.summation = jit(summation,data=data,identity=identity)
				self.exponentiation = jit(exponentiation,data=data,identity=identity)
		elif state is None and noise is not None:
			self.summation = jit(summation,data=data,identity=identity)
			self.exponentiation = jit(exponentiation,data=data,identity=identity)
		elif state is not None and noise is not None:
			self.summation = jit(summationmc,data=data,identity=identity,state=state,constants=noise)
			self.exponentiation = jit(exponentiationmc,data=data,identity=identity,state=state,constants=noise)
		else:
			self.summation = jit(summation,data=data,identity=identity)
			self.exponentiation = jit(exponentiation,data=data,identity=identity)

		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Return parameterized operator sum(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator			
		Returns
			operator (array): Parameterized operator
		'''		
		parameters = self.__parameters__(parameters)

		return self.summation(parameters)


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

				func = attributes[layer][layer][parameter][group]
				
				values = func(parameters,values)
		# try:
		# 	values = values.at[:2*self.N].set(parameters)
		# except:
		# 	attribute = 'slice'
		# 	for parameter in attributes[attribute][layer]:
		# 		for group in attributes[attribute][layer][parameter]:

		# 			func = attributes[layer][layer][parameter][group]
					
		# 			values = func(parameters,values)

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
		return self.__layers__(parameters,layer)

	#@partial(jit,static_argnums=(0,))
	def __objective__(self,parameters):
		''' 
		Setup objective
		Args:
			parameters (array): parameters
		Returns:
			objective (array): objective
		'''	
		return self.metrics(self(parameters),self.labels)

	#@partial(jit,static_argnums=(0,))
	def __loss__(self,parameters):
		''' 
		Setup loss
		Args:
			parameters (array): parameters
		Returns:
			loss (array): loss
		'''	
		return self.metrics(self(parameters),self.labels)

	#@partial(jit,static_argnums=(0,))
	def __func__(self,parameters):
		''' 
		Class function
		Args:
			parameters (array): parameters
		Returns:
			function (array): function
		'''	
		return self.__loss__(parameters) + self.__constraints__(parameters)

	#@partial(jit,static_argnums=(0,))
	def __grad__(self,parameters):
		''' 
		Class gradient of objective
		Args:
			parameters (array): parameters
		Returns:
			grad (array): gradient of objective
		'''	
		return self.grad(parameters)


	#@partial(jit,static_argnums=(0,))
	def __hessian__(self,parameters):
		''' 
		Class hessian of objective
		Args:
			parameters (array): parameters
		Returns:
			grad (array): hessian of objective
		'''	

		return self.hessian(parameters)


	#@partial(jit,static_argnums=(0,))
	def __derivative__(self,parameters):
		'''
		Return gradient of parameterized operator expm(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			derivative (array): Gradient of parameterized operator
		'''

		return self.derivative(parameters)

	#@partial(jit,static_argnums=(0,))
	def __derivative_analytical__(self,parameters):
		'''
		Return gradient of parameterized operator expm(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			derivative (array): Gradient of parameterized operator
		'''
		return self.derivative(parameters)


	#@partial(jit,static_argnums=(0,))
	def __grad_analytical__(self,parameters):
		''' 
		Class gradient of objective
		Args:
			parameters (array): parameters
		Returns:
			grad (array): gradient of objective
		'''	

		grad = self.metrics.__grad__(self(parameters),self.labels,self.__derivative_analytical__(parameters))

		return grad


	#@partial(jit,static_argnums=(0,))
	def __fisher__(self,parameters):
		''' 
		Class fisher information of class
		Args:
			parameters (array): parameters
		Returns:
			fisher (array): fisher information of class
		'''	
		return self.fisher(parameters)



	def __callback__(self,parameters,attributes):
		''' 
		Setup callback and logging
		Args:
			parameters (array): parameters
			attributes (dict): callback attributes
		Returns:
			status (int): status of class
		'''	

		optimize = self.hyperparameters['optimize']
		
		start = (len(attributes['iteration'])==1) and (attributes['iteration'][-1]<optimize['iterations'])
		
		done = (len(attributes['iteration'])>0) and (attributes['iteration'][-1]==optimize['iterations'])
		
		status = (
			(abs(attributes['value'][-1]) > 
				optimize['eps']['value']*optimize['value']['value']) and
			((len(attributes['value'])==1) or
			((len(attributes['value'])>1) and 
			 (abs(attributes['value'][-1] - attributes['value'][-2]) > 
				optimize['eps']['difference']*optimize['value']['value']))) and
			(norm(attributes['grad'][-1] - optimize['value']['grad'])/attributes['grad'][-1].size > 
				  optimize['eps']['grad'])
			)
		

		default = nan

		if (((not status) or done or start) or 
			(len(attributes['iteration']) == 0) or 
			(optimize['modulo']['track'] is None) or 
			(attributes['iteration'][-1]%optimize['modulo']['track'] == 0)
			):	

			for attr in ['iteration','parameters','value','grad','search','alpha','beta','objective','hessian','fisher']:

				if attr in optimize['track']:

					if ((optimize['length']['track'] is not None) and 
						(len(optimize['track'][attr]) > optimize['length']['track'])
						):
						optimize['track'][attr].pop(0)

					if attr in ['iteration','value','grad','search','alpha','beta'] and attr in attributes:
						optimize['track'][attr].append(attributes[attr][-1])

					elif attr in ['parameters'] and ((not status) or done or start):
						optimize['track'][attr].append(parameters)
					
					elif attr in ['objective']:
						optimize['track'][attr].append(
							getattr(self,'__%s__'%(attr))(parameters)
							)
					
					elif attr in ['hessian','fisher'] and ((not status) or done):
						optimize['track'][attr].append(
							getattr(self,'__%s__'%(attr))(parameters)
							)
					else:
						optimize['track'][attr].append(default)



		if ((len(attributes['iteration']) == 0) or 
			(optimize['modulo']['log'] is None) or 
			(attributes['iteration'][-1]%optimize['modulo']['log'] == 0)
			):

			msg = '\n'.join([
				'%d f(x) = %0.4e'%(
					attributes['iteration'][-1],
					optimize['track']['objective'][-1],
				),
				'|x| = %0.4e\t\t|grad(x)| = %0.4e'%(
					norm(attributes['parameters'][-1])/
						 max(1,attributes['parameters'][-1].size),
					norm(attributes['grad'][-1])/
						 max(1,attributes['grad'][-1].size),
				),
				'\t\t'.join([
					'%s = %0.4e'%(attr,attributes[attr][-1])
					for attr in ['alpha','beta']
					if attr in attributes and len(attributes[attr])>0
					]),
				# 'x\n%s'%(to_string(parameters.round(4))),
				'U\n%s\nV\n%s\n'%(
				to_string(abs(self(parameters)).round(4)),
				to_string(abs(self.labels).round(4))),
				# 'U: %0.4e\tV: %0.4e\n'%(
				# 	trace(self(parameters)).real,
				# 	trace(self.labels).real
				# 	),				
				])


			self.log(msg)


			# print(self.__layers__(parameters,'variables').T.reshape(self.M,-1).round(3))
		if (((not status) or done or start) or
			(len(attributes['iteration']) == 0) or 
			(optimize['modulo']['dump'] is None) or 
			(attributes['iteration'][-1]%optimize['modulo']['dump'] == 0)
			):
			self.dump({'data':True,'model':False})

		return status



	def __system__(self,system=None):
		'''
		Set system attributes
		Args:
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,backend,architecture,verbose)		
		'''
		system = self.system if system is None else system
		
		self.system = System(system)		

		self.dtype = self.system.dtype
		self.format = self.system.format
		self.seed = self.system.seed
		self.key = self.system.key
		self.timestamp = self.system.timestamp
		self.backend = self.system.backend
		self.architecture = self.system.architecture
		self.verbose = self.system.verbose

		return


	def __space__(self,N=None,D=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			space (str,Space): Type of Hilbert space
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,backend,architecture,verbose)		
		'''
		N = self.N if N is None else N
		D = self.D if D is None else D
		space = self.space if space is None else space
		system = self.system if system is None else system

		self.space = Space(N,D,space,system=system)

		self.N = self.space.N
		self.D = self.space.D		
		self.n = self.space.n
		self.g = self.space.g
		self.dims = (self.n,self.n)
		self.dim = int(product(self.dims))
		self.ndims = len(self.dims)
		self.shape = (len(self.data),self.M,*self.dims)
		self.ndim = len(self.shape)
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
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,backend,architecture,verbose)		
		'''
		M = self.M if M is None else M
		T = self.T if T is None else T
		tau = self.tau if tau is None else tau
		p = self.p if p is None else p
		time = self.time if time is None else time
		system = self.system if system is None else system

		self.time = Time(M,T,tau,p,time,system=system)	

		self.M = self.time.M
		self.T = self.time.T
		self.p = self.time.p
		self.tau = self.time.tau
		self.shape = (*self.shape[:1],self.M,*self.shape[2:])	
		self.ndim = len(self.shape)

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
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,backend,architecture,verbose)		
		'''		
		N = self.N if N is None else N
		D = self.D if D is None else D
		d = self.d if d is None else d
		L = self.L if L is None else L
		delta = self.delta if delta is None else delta
		lattice = self.lattice if lattice is None else lattice
		system = self.system if system is None else system

		self.lattice = Lattice(N,d,L,delta,lattice,system=system)	

		self.N = self.lattice.N
		self.d = self.lattice.d
		self.L = self.lattice.L
		self.delta = self.lattice.delta

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
			path (str,dict[str,(str,bool)]): Path to dump class data, either path or boolean to dump			
		'''

		# TODO: Transfer model dumping/loading (checkpointing) to Optimizer/Objective class

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
			optimize = obj.hyperparameters['optimize']

			returns = {}

			if attr in optimize['track']:
				new = attr
				New = value[attr]			
				returns[new] = New
			else:
				new = attr
				New = value[attr]
				returns[new] = New


			if attr in optimize['track']:

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

					if is_array(value[attr]):

						new = '%s.relative'%(attr)
						New = abs((obj.__layers__(value[attr],layer)[indices] - 
							obj.__layers__(optimize['track'][attr][0],layer)[indices] + 1e-20)/(
							obj.__layers__(optimize['track'][attr][0],layer)[indices] + 1e-20))
						returns[new] = New

						new = '%s.relative.mean'%(attr)
						New = New.mean(-1)
						returns[new] = New				

						new = attr
						New = obj.__layers__(value[attr],layer)[indices]

						returns[new] = New


						if obj.state is None:

							data = None
							shape = obj.dims
							hyperparams = deepcopy(obj.hyperparameters['state'])
							hyperparams['scale'] = 1 if hyperparams.get('scale') is None else hyperparams.get('scale')
							size = obj.N
							samples = True
							seed = obj.seed		
							dtype = obj.dtype
							cls = obj

							state = stateize(data,shape,hyperparams,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)

						else:
							state = obj.state

						if obj.noise is None:

							data = None
							shape = obj.dims
							hyperparams = deepcopy(obj.hyperparameters['noise'])
							hyperparams['scale'] = 1 if hyperparams.get('scale') is None else hyperparams.get('scale')
							size = obj.N
							samples = None
							seed = obj.seed		
							cls = obj
							dtype = obj.dtype

							noise = noiseize(data,shape,hyperparams,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)

						else:
							noise = obj.noise

						obj.__functions__(state=state,noise=noise,label=True,metric='infidelity.norm')

						new = 'objective.ideal.noise'
						New = obj.__objective__(value[attr])
						returns[new] = New

						new = 'objective.diff.noise'
						New = abs(value['objective'] - New)
						returns[new] = New

						new = 'objective.rel.noise'
						New = abs((value['objective'] - New)/max(1,New))
						returns[new] = New	


						obj.__functions__(state=state,noise=False,label=True,metric='infidelity.norm')

						new = 'objective.ideal.state'
						New = obj.__objective__(value[attr])
						returns[new] = New

						new = 'objective.diff.state'
						New = abs(value['objective'] - New)
						returns[new] = New

						new = 'objective.rel.state'
						New = abs((value['objective'] - New)/max(1,New))
						returns[new] = New					


						obj.__functions__(state=False,noise=False,label=True,metric='infidelity.abs')

						new = 'objective.ideal.operator'
						New = obj.__objective__(value[attr])
						returns[new] = New

						new = 'objective.diff.operator'
						New = abs(value['objective'] - New)
						returns[new] = New

						new = 'objective.rel.operator'
						New = abs((value['objective'] - New)/max(1,New))
						returns[new] = New					


						obj.__functions__(state=True,noise=True,label=True,metric=True)


				elif attr in ['iteration']:
					new = '%s.max'%(attr)
					New = optimize['track'][attr][-1]
					returns[new] = New

					new = '%s.min'%(attr)
					New = optimize['track'][attr][argmin(array(optimize['track']['objective']))]
					returns[new] = New

					new = 'status'
					New = value[attr]/max(1,optimize['track'][attr][-1])

					returns[new] = New

				elif attr in ['objective']:
					pass

				elif attr in ['hessian','fisher']:
					if is_array(value[attr]):
						new = '%s.eigenvalues'%(attr)						
						New  = sort(abs(eig(value[attr],compute_v=False,hermitian=True)))[::-1]
						New = New/max(1,maximum(New))
						# _New = int((argmax(abs(difference(New)/New[:-1]))+1)*1.5)
						# New = New[:_New]
						returns[new] = New

						new = '%s.rank'%(attr)
						New = argmax(abs(difference(New)/New[:-1]))+1						
						returns[new] = New


			return returns

		# Get data
		optimize = self.hyperparameters['optimize']
		label = [self.timestamp]
		keys = [self.key]
		iterations = {
			key: range(min(len(optimize['track'][attr]) 
			for attr in optimize['track']
			if len(optimize['track'][attr]) > 0
			))
			for key in keys
			}

		data = {
			delim.join([*(str(l) for l in label),str(key),str(iteration)]): {
				**{attr: optimize['track'][attr][iteration]
					for attr in optimize['track']
					if len(optimize['track'][attr]) > 0
				},	
				**{attr: getattr(self,attr)
					for attr in self.__dict__
					if (
						(not callable(getattr(self,attr))) and
						(getattr(self,attr) is None or isinstance(getattr(self,attr),scalars)) and
						attr not in optimize['track']
						)
					},
				**{attr: getter(self.hyperparameters,attr.split(delim)) 
					for attr in self.hyperparameters.get('process',{}).get('labels',[])
					},
				}
			for key in keys
			for iteration in iterations[key]
		}

		# Process data
		# data have axis of ('sample,model',*'value-shape','iterations') 
		# and must be transposed with func such that optimization tracked data has iterations as -1 axis
		for key in list(data):
			for attr in list(data[key]):			
				data[key].update(func(attr,data[key],self))
		
		# Ensure all keys have all attribute
		attrs = list(set((attr for key in data for attr in data[key])))
		for key in data:
			for attr in attrs:
				if attr not in data[key]:
					data[key][attr] = nan

		# Set data
		data = {
			'data':data,
			'model':self.hyperparameters
			}

		# Set path
		paths = {}
		if path is None:
			paths.update({attr: True for attr in data})			
		elif not isinstance(path,dict):
			paths.update({attr: path for attr in data if path})
		else:
			paths.update({attr: path[attr] for attr in path if path[attr]})

		paths.update({attr: paths.get(attr) if isinstance(paths.get(attr),str) else self.hyperparameters['sys']['cwd'] for attr in data if paths.get(attr)})			

		# Dump data
		for attr in paths:
			root,file = split(paths[attr],directory=True,file_ext=True)
			file = file if file is not None else self.hyperparameters['sys']['path']['data'][attr]
			path = join(file,root=root)
			dump(data[attr],path)
		
		return

	def load(self,path=None):
		'''
		Load class data		
		Args:
			path (str,dict[str,(str,bool)]): Path to load class data, either path or boolean to load
		'''

		# TODO: Determine which loaded hyperparameters should have precedence over new hyperparameters

		def func(key,iterable,elements): 
			types = (list,)
			i = iterable.get(key,elements.get(key))
			e = elements.get(key,i)
			return e if isinstance(e,types) else i
		
		# Set data
		data = {
			'model':self.hyperparameters
			}

		# Set path
		paths = {}
		if path is None:
			paths.update({attr: True for attr in data})			
		elif not isinstance(path,dict):
			paths.update({attr: path for attr in data if path})
		else:
			paths.update({attr: path[attr] for attr in path if path[attr]})

		paths.update({attr: paths.get(attr) if isinstance(paths.get(attr),str) else self.hyperparameters['sys']['cwd'] for attr in data if paths.get(attr)})			

		# Load data
		for attr in paths:
			root,file = split(paths[attr],directory=True,file_ext=True)
			file = file if file is not None else self.hyperparameters['sys']['path']['data'][attr]
			path = join(file,root=root)
			default = data[attr]
			data[attr] = load(path,default=default)
			updater(default,data[attr],func=func)

		try:
			self.parameters = optimize['track']['parameters'][-1]
		except:
			pass

		return



class Hamiltonian(Object):
	'''
	Hamiltonian class of Operators
	Args:
		data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
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
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,backend,architecture,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={},
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,space=None,time=None,lattice=None,metric=None,system=None):
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,hyperparameters=hyperparameters,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,p=p,space=space,time=time,lattice=lattice,metric=metric,system=system)
		return


	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Return parameterized operator sum(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator			
		Returns
			operator (array): Parameterized operator
		'''		
		parameters = self.__parameters__(parameters)

		return self.summation(parameters)

	def __setup__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={}):
		'''
		Setup class
		Args:
			data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
				string (iterable[str]): string labels of operators
				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
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
		indices = {'i': ['i'],'<ij>':['i','j'],'i<j':['i','j'],'i...j':['i','j']} # allowed symbolic indices and maximum number of body site interactions

		# Basis single-site operators
		dtype = self.dtype
		operators = {
			attr: basis[attr].astype(dtype)
			for attr in basis
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

			if isinstance(_site,str):
				_site = indices[_site]

			if any(j in indices[_interaction] for j in _site):
				for s in sites[_interaction]:
					_site_ = deepcopy([dict(zip(indices[_interaction],s if not isinstance(s,int) else [s])).get(j,parse(j,int)) for j in _site])
					_operator_ = deepcopy([_operator[_site_.index(j)] if j in _site_ else I for j in range(self.N)])
					_string_ = deepcopy(_string)
					_interaction_ = deepcopy(_interaction)
					
					if _operator_ not in operator: 
						site.append(_site_)
						operator.append(_operator_)
						string.append(_string_)
						interaction.append(_interaction_)
			else:
				_site_ = deepcopy(_site)
				_operator_ = deepcopy([_operator[_site_.index(j)] if j in _site_ else I for j in range(self.N)])
				_string_ = deepcopy(_string)
				_interaction_ = deepcopy(_interaction)

				site.append(_site_)
				operator.append(_operator_)
				string.append(_string_)
				interaction.append(_interaction_)


		# Form (size,n,n) shape operator from local strings for each data term
		data = [tensorprod([operators[j] for j in i]) for i in operator]

		# Assert all data satisfies data**2 = identity for matrix exponentials
		assert all(allclose(d.dot(d),self.identity) for d in data), 'data is not involutory and data**2 != identity'

		# Get Trotterized order of p copies of data for products of data
		p = self.p
		data = trotter(data,p)
		operator = trotter(operator,p)
		site = trotter(site,p)
		string = trotter(string,p)
		interaction = trotter(interaction,p)

		# Check for case of size
		if not size:
			data = [self.identity]*self.p
			operator = [['I']*self.N]*self.p
			site = [list(range(self.N))]*self.p
			string = ['I']*self.p
			interaction = ['i...j']*self.p
		
		

		# Update class attributes
		self.__extend__(data,operator,site,string,interaction,hyperparameters)

		# Initialize parameters
		self.__initialize__(parameters)

		# Setup functions
		self.__functions__()

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
		shape = parameters.T.shape
		parameters = parameters.T.ravel()

		return parameters



class Unitary(Hamiltonian):
	'''
	Unitary class of Operators
	Args:
		data (dict[str,dict]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
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
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,backend,architecture,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,hyperparameters={},
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,space=None,time=None,lattice=None,metric=None,system=None):
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,hyperparameters=hyperparameters,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,p=p,space=space,time=time,lattice=lattice,metric=metric,system=system)

		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Return parameterized operator expm(parameters*data)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			operator (array): Parameterized operator
		'''		
		parameters = self.__parameters__(parameters)
		return self.exponentiation(self.coefficients*parameters)

	#@partial(jit,static_argnums=(0,))
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

		derivative = gradient_expm(self.coefficients*parameters,self.data,self.identity)
		derivative *= self.coefficients

		# Reshape gradient
		axis = 1

		derivative = derivative.reshape((*shape,*derivative.shape[axis:]))

		derivative = derivative.transpose(axis,0,*[i for i in range(derivative.ndim) if i not in [0,axis]])

		derivative = array(gradient_trotter(derivative,p))

		derivative = derivative[indices]

		derivative = derivative.reshape((-1,*derivative.shape[axis+1:]))

		return derivative





class Operator(module):
	'''
	Class for Operator
	Args:
		data (dict,str,array): dictionary of operator attributes, or string or array for operator. Allowed strings in ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI'], allowed dictionary keys in
			operator (str,iterable[str],array): string or array for operator, allowed ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI']
			site (iterable[int]): site of local operators
			string (str): string label of operator
			interaction (str): interaction type of operator
		operator (str,iterable[str],array): string or array for operator, allowed ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI']
		site (iterable[int]): site of local operators
		string (str): string label of operator
		interaction (str): interaction type of operator
		hyperparameters (dict) : class hyperparameters
		N (int): Number of qudits
		D (int): Dimension of qudits
		space (str,Space): Type of Hilbert space
		system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,backend,architecture,verbose)
	'''

	data : None
	operator : str
	site : List[int]
	string : str
	interaction : str
	hyperparameters : dict
	N : int
	D : int
	space : str
	system : dict

	n : int
	shape : Tuple[int]
	size : int
	ndim : int
	locality : int

	identity : array

	dtype: str
	format : str
	seed : int
	key : List[int]
	timestamp : str
	backend : str
	architecture : str
	verbose : int

	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,hyperparameters={},
					N=None,D=None,space=None,system=None):

		self.N = N
		self.D = D
		self.space = space
		self.system = system

		self.__system__()
		self.__space__()
		self.__setup__(data,operator,site,string,interaction,hyperparameters)
		
		return
	
	def __system__(self,system=None):
		'''
		Set system attributes
		Args:
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,backend,architecture,verbose)		
		'''
		system = self.system if system is None else system

		system = System(system)		

		self.dtype = system.dtype
		self.format = system.format
		self.seed = system.seed
		self.key = system.key
		self.timestamp = system.timestamp
		self.backend = system.backend
		self.architecture = system.architecture
		self.verbose = system.verbose

		return
	
	def __space__(self,N=None,D=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			space (str,Space): Type of Hilbert space
			system (dict,System): System attributes (dtype,format,device,seed,key,timestamp,backend,architecture,verbose)		
		'''
		N = self.N if N is None else N
		D = self.D if D is None else D
		space = self.space if space is None else space
		system = self.system if system is None else system

		space = Space(N,D,space,system=system)

		self.N = space.N
		self.D = space.D		
		self.n = space.n
		self.shape = (self.n,self.n)
		self.size = 1
		self.ndim = len(self.shape)
		self.identity = identity(self.n,dtype=self.dtype)

		return
	
	def __str__(self):
		return self.string
	def __repr__(self):
		return self.__str__()
	def __len__(self):
		return len(self.data)
	
	@nn.filter_jit
	def __call__(self,parameters,state=None):
		'''
		Return parameterized operator 
		Args:
			parameters (array): Parameters to parameterize operator			
			state (array): State to apply operator
		Returns
			operator (array): Parameterized operator
		'''		
		parameters = self.__parameters__(parameters)

		if state is None:
			operator = parameters*self.data
		else:
			operator = (parameters*self.data).dot(state)
		return operator

	@nn.filter_jit
	def __parameters__(self,parameters):
		''' 
		Setup parameters
		Args:
			parameters (array): parameters
		Returns:
			parameters (array): parameters
		'''
		return parameters

	@nn.filter_jit
	def __apply__(self,parameters,state=None):
		'''
		Return parameterized operator 
		Args:
			parameters (array): Parameters to parameterize operator			
			state (array): State to apply operator
		Returns
			operator (array): Parameterized operator
		'''
		
		parameters = self.__parameters__(parameters)

		if state is None:
			operator = parameters*self.data
		else:

			for site in self.site:
				state = self.__swap__(state,site)
				operator = (parameters*self.data).dot(state)
				operator = operator.reshape(self.dims)
				state = self.__reshape__(state,site)

		return operator

	@nn.filter_jit
	def __swap__(self,state,site):
		'''
		Swap axes of state at site
		Args:
			state (array): State to apply operator of shape (n,)
			site (iterable[int]): Axes to apply operator of size locality
		Returns
			state (array): State to apply operator of shape (*(D)*locality,n/D**locality)
		'''
		# TODO
		raise NotImplementedError

		locality = len(site)
		axes = range(locality)
		shape = (*(self.D)*locality,-1)

		state = moveaxis(state,site,axes).reshape(shape)

		return state

	@nn.filter_jit
	def __reshape__(self,state,site):
		'''
		Reshape state to shape (n,)
		Args:
			state (array): State to apply operator of shape (n,)
			site (iterable[int]): Axes to apply operator
		Returns
			state (array): State to apply operator of shape (D,D,n/D**2)
		'''

		# TODO
		raise NotImplementedError

		locality = len(site)
		axes = range(locality)
		shape = (self.n,)

		state = moveaxis(state.reshape(shape),site,axes)

		return state


	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None,hyperparameters={}):
		'''
		Setup class
		Args:
			data (dict,str,array): dictionary of operator attributes, or string or array for operator. Allowed strings in ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI'], allowed dictionary keys in
						operator (str,iterable[str],array): string or array for operator, allowed ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI']
						site (iterable[int]): site of local operators
						string (str): string label of operator
						interaction (str): interaction type of operator
					operator (str,iterable[str],array): string or array for operator, allowed ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI']
					site (iterable[int]): site of local operators
					string (str): string label of operator
					interaction (str): interaction type of operator
					hyperparameters (dict) : class hyperparameters
		'''

		if isinstance(data,str):
			operator = data
		elif is_array(data):
			operator = data
				
		if operator is None:
			operator = ''
		if site is None:
			site = list(range(len(operator)))
		if string is None:
			string = '%s_%s'%(operator,''.join(['%d'%(i) for i in site]))
		else:
			string = '%s_%s'%(string,''.join(['%d'%(i) for i in site]))
		if interaction is None:
			interaction = ''
		if hyperparameters is None:
			hyperparameters = {}

		self.data = data
		self.operator = operator
		self.site = site
		self.string = string
		self.interaction = interaction
		self.locality = len(site)
		
		self.hyperparameters = hyperparameters
		
		if isinstance(self.data,dict):
			for attr in self.data:
				setattr(self,attr,self.data[attr])

		self.data = self.operator
		if not is_array(self.data):
			self.data = [basis[operator] for operator in self.data]
			self.data = tensorprod(self.data)

		self.data = self.data.astype(self.dtype)

		return