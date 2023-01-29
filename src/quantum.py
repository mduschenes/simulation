#!/usr/bin/env python

# Import python modules
import os,sys
from copy import deepcopy
from functools import partial

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,gradient,hessian,fisher
from src.utils import array,ones,zeros,empty,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product,dot,dagger,conj,transpose,einsum
from src.utils import summation,exponentiation,summationv,exponentiationv,summationm,exponentiationm,summationmvc,exponentiationmvc,summationmmc,exponentiationmmc
from src.utils import trotter,gradient_trotter,gradient_expm,gradient_sigmoid
from src.utils import inner_norm,inner_abs2,inner_real,inner_imag
from src.utils import gradient_inner_norm,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
from src.utils import eig
from src.utils import maximum,minimum,argmax,argmin,difference,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,relsort,norm,interpolate,unique,allclose,isclose,is_array,is_naninf,to_key_value 
from src.utils import initialize,parse,to_string,to_number,datatype,slice_size,intersection
from src.utils import pi,e,nan,null,delim,scalars,nulls
from src.utils import itg,flt,dbl

from src.iterables import setter,getter,getattrs,hasattrs

from src.parameters import Parameters
from src.operators import Gate
from src.states import State
from src.noise import Noise

from src.io import load,dump,join,split

from src.system import System,Space,Time,Lattice

from src.optimize import Objective,Metric

class Operator(System):
	'''
	Class for Observable
	Args:
		data (iterable[str]): data for operator
		operator (iterable[str]): string names of operators		
		site (iterable[int]): site of local operators
		string (str): string labels of operators
		interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		N (int): Number of qudits
		D (int): Dimension of qudits
		parameters (str,dict,Parameters): Type of parameters	
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''	
	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
		N=None,D=None,parameters=None,system=None,**kwargs):


		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		self.data = data
		self.operator = operator
		self.site = site
		self.string = string
		self.interaction = interaction

		self.N = N
		self.D = D
		self.parameters = parameters
		self.system = system

		self.__setup__(data,operator,site,string,interaction)

		return

	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup operator
		Args:
			data (iterable[str]): data for operator
			operator (iterable[str]): string names of operators			
			site (iterable[int]): site of local operators
			string (str): string labels of operators
			interaction (str): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']	
		'''

		basis = {
				'I': array([[1,0],[0,1]]),
				'X': array([[0,1],[1,0]]),
				'Y': array([[0,-1j],[1j,0]]),
				'Z': array([[1,0],[0,-1]]),
			}
		default = 'I'

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		site = self.site if site is None else site
		string = self.string if string is None else string
		interaction = self.interaction if interaction is None else interaction

		if operator is None:
			operator = [default]*self.N
		elif len(operator) == self.N:
			operator = [operator[j] for j in range(self.N)]
		elif len(operator) == len(site):
			operator = [operator[site.index(j)] if j in site else default for j in range(self.N)]
		else:
			operator = [i for i in operator]

		data = operator
		data = tensorprod([basis.get(i,basis[default]) for i in data])
		data = data.astype(self.dtype)

		self.data = data
		self.operator = operator
		self.site = site
		self.string = string
		self.interaction = interaction

		return

	def __call__(self,parameters=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
		Returns:
			operator (array): operator
		'''

		return self.data

	def __str__(self):
		try:
			return delim.join(self.operator)
		except:
			return self.__class__.__name__

	def __repr__(self):
		return self.__str__()
	
	def __len__(self):
		return len(self.data)


class Observable(System):
	'''
	Class for Observable
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Simulation length scale		
		M (int): Number of time steps
		T (int): Simulation Time
		tau (float): Simulation time scale		
		P (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice	
		parameters (str,dict,Parameters): Type of parameters	
		state (str,dict,State): Type of state	
		noise (str,dict,Noise): Type of noise
		label (str,dict,Gate): Type of label	
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
		N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
		space=None,time=None,lattice=None,parameters=None,state=None,noise=None,label=None,system=None,**kwargs):

		setter(kwargs,system,delimiter=delim,func=False)
		super().__init__(**kwargs)

		self.N = N
		self.D = D
		self.d = d
		self.L = L
		self.delta = delta
		self.M = M
		self.T = T
		self.tau = tau
		self.P = P
		self.space = space
		self.time = time
		self.lattice = lattice

		self.data = []
		self.operator = []
		self.site = []
		self.string = []
		self.interaction = []
		self.indices = []

		self.shape = ()
		self.size = int(product(self.shape))
		self.ndim = len(self.shape)
		self.shapes = (self.shape,self.shape)
		self.dims = (self.M if self.M is not None else 0,len(self.data),*self.shape)
		self.length = int(product(self.dims))
		self.ndims = len(self.dims)

		self.parameters = parameters
		self.state = state
		self.noise = noise
		self.label = label
		self.identity = None
		self.constants = None
		self.coefficients = None
		self.dimensions = None	

		self.summation = None
		self.exponentiation = None 
		self.hermitian = False

		self.system = system

		self.__time__()
		self.__space__()
		self.__lattice__()

		self.__setup__(data,operator,site,string,interaction)

		self.info()

		return	

	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
				string (iterable[str]): string labels of operators
				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
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


		if not isinstance(data,dict):
			data = {datum.timestamp: datum for datum in data}

		operator.extend([data[name]['operator'] for name in data])
		site.extend([data[name]['site'] for name in data])
		string.extend([data[name]['string'] for name in data])
		interaction.extend([data[name]['interaction'] for name in data])

		# Set class attributes
		self.__extend__(operator=operator,site=site,string=string,interaction=interaction)

		# Set parameters
		self.__initialize__()

		# Set functions
		self.__functions__()
		
		return


	def __append__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Append to class
		Args:
			data (str,Operator): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''
		index = -1
		self.__insert__(index,data,operator,site,string,interaction)
		return

	def __extend__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup class
		Args:
			data (iterable[str,Operator]): data of operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''
		if all(i is None for i in [operator,site,string,interaction]):
			return

		if data is None:
			size = min([len(i) for i in [operator,site,string,interaction]])
			data = [None]*size

		for _data,_operator,_site,_string,_interaction in zip(data,operator,site,string,interaction):
			self.__append__(_data,_operator,_site,_string,_interaction)

		return


	def __insert__(self,index,data,operator,site,string,interaction):
		'''
		Insert to class
		Args:
			index (int): index to insert operator
			data (str,Operator): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''

		if index == -1:
			index = len(self.data)

		if not isinstance(data,Operator):
			data = Operator(data=data,operator=operator,site=site,string=string,interaction=interaction,
							N=self.N,D=self.D,system=self.system)

		self.data.insert(index,data)
		self.operator.insert(index,operator)
		self.site.insert(index,site)
		self.string.insert(index,string)
		self.interaction.insert(index,interaction)

		self.dims = (self.M,len(self.data),*self.shape)
		self.ndims = len(self.dims)		

		return



	def __initialize__(self,parameters=None):
		''' 
		Setup initial parameters and attributes
		Args:
			parameters (array): parameters
		'''

		# Get attributes data of parameters of the form {attribute:{parameter:{group:{layer:[]}}}
		parameters = self.parameters.hyperparameters if isinstance(self.parameters,Parameters) else self.parameters
		shape = (len(self.data),self.M)
		dims = None
		cls = self
		check = lambda group,index,axis,site=self.site,string=self.string: (
			(axis != 0) or 
			any(g in group for g in [string[index],'_'.join([string[index],''.join(['%d'%j for j in site[index]])])]))
		system = self.system

		parameters = Parameters(parameters,shape,dims=dims,system=system,cls=cls,check=check,initialize=initialize)

		# Get coefficients
		coefficients = -1j*2*pi/2*self.tau/self.P		

		# Update class attributes
		self.parameters = parameters
		self.coefficients = coefficients
		self.dimensions = parameters.dimensions

		return

	def __functions__(self,state=None,noise=None,label=None):
		''' 
		Setup class functions
		Args:
			state (bool,dict,array,State): State to act on with class of shape self.shape, or class hyperparameters, or boolean to choose self.state or None
			noise (bool,dict,array,Noise): Noise to act on with class of shape (-1,self.shape), or class hyperparameters, or boolean to choose self.noise or None
			label (bool,dict,array,Gate): Label of class of shape self.shape, or class hyperparameters, or boolean to choose self.label or None
		'''

		# Function arguments
		data = array(trotter([data() for data in self.data],self.P))
		identity = self.identity()
		state = self.state if state is None or state is True else state if state is not False else None
		noise = self.noise if noise is None or noise is True else noise if noise is not False else None
		label = self.label if label is None or label is True else label if label is not False else None

		# Class arguments
		kwargs = {}
		shape = self.shape
		dims = [self.N,self.D]
		system = self.system

		# Get state
		kwargs.clear()
		setter(kwargs,self.state)
		setter(kwargs,state)
		setter(kwargs,dict(data=state,shape=shape,dims=dims,system=system))
		self.state = State(**kwargs)
		state = self.state()

		# Get noise
		kwargs.clear()
		setter(kwargs,self.noise)
		setter(kwargs,noise)
		setter(kwargs,dict(data=noise,shape=shape,dims=dims,system=system))
		self.noise = Noise(**kwargs)
		noise = self.noise()

		# Get label
		kwargs.clear()
		setter(kwargs,self.label)
		setter(kwargs,label)
		setter(kwargs,dict(data=label,shape=shape,dims=dims,system=system))
		self.label = Gate(**kwargs)
		label = self.label()

		# Attribute values
		if state is None:
			label = label
		elif state.ndim == 1:
			label = einsum('ij,j->i',label,state)
		elif state.ndim == 2:
			label = einsum('ij,jk,kl->il',label,state,dagger(label))
		else:
			label = label
		label = self.label(label)

		shapes = (self.label.shape,self.label.shape)
		self.shapes = shapes

		# Operator functions
		if state is None and noise is None:
			self.summation = jit(summation,data=data,identity=identity)
			self.exponentiation = jit(exponentiation,data=data,identity=identity)
			self.hermitian = False
		elif state is not None and noise is None:
			if state.ndim == 1:
				self.summation = jit(summationv,data=data,identity=identity,state=state)
				self.exponentiation = jit(exponentiationv,data=data,identity=identity,state=state)
				self.hermitian = True
			elif state.ndim == 2:
				self.summation = jit(summationm,data=data,identity=identity,state=state)
				self.exponentiation = jit(exponentiationm,data=data,identity=identity,state=state)
				self.hermitian = True
			else:
				self.summation = jit(summation,data=data,identity=identity)
				self.exponentiation = jit(exponentiation,data=data,identity=identity)
				self.hermitian = False
		elif state is None and noise is not None:
			self.summation = jit(summation,data=data,identity=identity)
			self.exponentiation = jit(exponentiation,data=data,identity=identity)
			self.hermitian = False
		elif state is not None and noise is not None:
			if state.ndim == 1:
				self.summation = jit(summationmvc,data=data,identity=identity,state=state,constants=noise)
				self.exponentiation = jit(exponentiationmvc,data=data,identity=identity,state=state,constants=noise)
				self.hermitian = True
			elif state.ndim == 2:
				self.summation = jit(summationmmc,data=data,identity=identity,state=state,constants=noise)
				self.exponentiation = jit(exponentiationmmc,data=data,identity=identity,state=state,constants=noise)
				self.hermitian = True
			else:
				self.summation = jit(summation,data=data,identity=identity)
				self.exponentiation = jit(exponentiation,data=data,identity=identity)
				self.hermitian = False
		else:
			self.summation = jit(summation,data=data,identity=identity)
			self.exponentiation = jit(exponentiation,data=data,identity=identity)
			self.hermitian = False



		# Functions
		self.gradient = gradient(self,mode='fwd',move=True)

		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Class function
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function
		'''		
		parameters = self.__parameters__(parameters)
		return self.summation(parameters)

	#@partial(jit,static_argnums=(0,))
	def __grad__(self,parameters):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
		Returns:
			out (array): Return of function
		'''	
		return self.gradient(parameters)


	# @partial(jit,static_argnums=(0,))
	def __value_and_grad__(self,parameters):
		'''
		Class function and gradient
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function and gradient
		'''	
		return self.value_and_gradient(parameters)


	#@partial(jit,static_argnums=(0,))
	def func(self,parameters):
		'''
		Class function
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function
		'''
		return self.__call__(parameters)

	#@partial(jit,static_argnums=(0,))
	def grad(self,parameters):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
		Returns:
			out (array): Return of function
		'''		
		return self.__grad__(parameters)

	# @partial(jit,static_argnums=(0,))
	def value_and_grad(self,parameters):
		'''
		Class function and gradient
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function and gradient
		'''	
		return self.__value_and_gradient__(parameters)

	# @partial(jit,static_argnums=(0,))
	def constraints(self,parameters):
		''' 
		Setup constraints
		Args:
			parameters (array): parameters
		Returns:
			constraints (array): constraints
		'''		
		return self.__constraints__(parameters)


	# @partial(jit,static_argnums=(0,))
	def __parameters__(self,parameters):
		''' 
		Setup parameters
		Args:
			parameters (array): parameters
		Returns:
			parameters (array): parameters
		'''
		self.parameters(parameters)
		return parameters


	# @partial(jit,static_argnums=(0,2))
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
		self.parameters(parameters)
		attributes = self.parameters.attributes

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

		return values

	# @partial(jit,static_argnums=(0,))
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


	def __space__(self,N=None,D=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			space (str,Space): Type of Hilbert space
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
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
		self.shape = (self.n,self.n)
		self.size = int(product(self.shape))
		self.ndim = len(self.shape)
		self.dims = (self.M,len(self.data),*self.shape)
		self.length = int(product(self.dims))
		self.ndims = len(self.dims)
		
		self.identity = Operator(N=self.N,D=self.D,system=self.system)

		return


	def __time__(self,M=None,T=None,tau=None,P=None,time=None,system=None):
		'''
		Set time attributes
		Args:
			M (int): Number of time steps
			T (int): Simulation Time
			tau (float): Simulation time scale
			P (int): Trotter order		
			time (str,Time): Type of Time evolution space						
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
		'''
		M = self.M if M is None else M
		T = self.T if T is None else T
		tau = self.tau if tau is None else tau
		P = self.P if P is None else P
		time = self.time if time is None else time
		system = self.system if system is None else system

		self.time = Time(M,T,tau,P,time,system=system)	

		self.M = self.time.M
		self.T = self.time.T
		self.P = self.time.P
		self.tau = self.time.tau
		self.dims = (self.M,self.dims[1:])	
		self.ndims = len(self.dims)

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
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
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

	def __str__(self):
		try:
			size = len(self.data)
			delimiter = ' '
			multiple_time = (self.M>1)
			multiple_space = [size>1 and False for i in range(size)]
			return '%s%s%s%s'%(
					'{' if multiple_time else '',
					delimiter.join(['%s%s%s'%(
						'(' if multiple_space[i] else '',
						self.string[i],
						')' if multiple_space[i] else '',
						) for i in range(size)]),
					'}' if multiple_time else '',
					'^%d'%(self.M) if multiple_time else '')
		except AttributeError:
			return self.__class__.__name__

	def __repr__(self):
		return self.__str__()

	def __len__(self):
		return len(self.data)

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		
		msg = '%s'%('\n'.join([
			*['%s: %s'%(attr,getattrs(self,attr,delimiter=delim)) 
				for attr in ['key','seed','N','D','d','L','delta','M','tau','T','P','shape','dims','cwd','path','backend','architecture','conf','logger','cleanup']
			],
			*['%s: %s'%(attr.split(delim)[0],getattrs(self,attr,delimiter=delim)) 
				for attr in ['state.scale','noise.scale']
			],
			*['%s: %s'%(attr,getattrs(self,attr,delimiter=delim).__name__) 
				for attr in ['exponentiation']
			],			
			]
			))
		self.log(msg,verbose=verbose)

		return


	def dump(self,path=None):
		'''
		Save class data		
		Args:
			path (str,dict[str,(str,bool)]): Path to dump class data, either path or boolean to dump			
		'''

		# Set data
		data = {}

		# Set path
		paths = {}
		if path is None:
			paths.update({attr: True for attr in data})			
		elif not isinstance(path,dict):
			paths.update({attr: path for attr in data if path})
		else:
			paths.update({attr: path[attr] for attr in path if path[attr]})

		paths.update({attr: paths.get(attr) if isinstance(paths.get(attr),str) else self.cwd for attr in data if paths.get(attr)})			

		# Dump data
		for attr in paths:
			root,file = split(paths[attr],directory=True,file_ext=True)
			file = file if file is not None else self.path
			path = join(file,root=root)
			dump(data[attr],path)
		
		return

	def load(self,path=None):
		'''
		Load class data		
		Args:
			path (str,dict[str,(str,bool)]): Path to load class data, either path or boolean to load
		'''

		# TODO: Determine dump/load model (.pkl?)

		# Set data
		data = {}

		# Set path
		paths = {}
		if path is None:
			paths.update({attr: True for attr in data})			
		elif not isinstance(path,dict):
			paths.update({attr: path for attr in data if path})
		else:
			paths.update({attr: path[attr] for attr in path if path[attr]})

		paths.update({attr: paths.get(attr) if isinstance(paths.get(attr),str) else self.cwd for attr in data if paths.get(attr)})			

		# Load data
		for attr in paths:
			root,file = split(paths[attr],directory=True,file_ext=True)
			file = file if file is not None else self.path
			path = join(file,root=root)
			func = (list,)
			default = data[attr]
			data[attr] = load(path,default=default)
			setter(default,data[attr],func=func)

		return



class Hamiltonian(Observable):
	'''
	Hamiltonian class of Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Simulation length scale		
		M (int): Number of time steps
		T (int): Simulation time
		tau (float): Simulation time scale
		P (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		parameters (str,dict,Parameters): Type of parameters	
		state (str,dict,State): Type of state	
		noise (str,dict,Noise): Type of noise
		label (str,dict,Gate): Type of label
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
				space=None,time=None,lattice=None,parameters=None,state=None,noise=None,label=None,system=None,**kwargs):
		
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
				space=space,time=time,lattice=lattice,parameters=parameters,state=state,noise=noise,label=label,system=system,**kwargs)
		
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

	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
				operator (iterable[str]): string names of operators
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
				string (iterable[str]): string labels of operators
				interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''

		# Get operator,site,string,interaction from data
		if operator is None:
			operator = []
		if site is None:
			site = []
		if string is None:
			string = []
		if interaction is None:
			interaction = []									

		if data is None:
			data = {}
		elif all(isinstance(datum,Operator) for datum in data):
			data = {datum.timestamp: datum for datum in data}
		
		assert isinstance(data,dict), 'Incorrect data format %r'%(type(data))			

		operator.extend([data[name]['operator'] for name in data])
		site.extend([data[name]['site'] for name in data])
		string.extend([data[name]['string'] for name in data])
		interaction.extend([data[name]['interaction'] for name in data])

		# Lattice sites
		sites = {site: self.lattice(site) for site in ['i','i<j','<ij>','i...j']}	# sites types on lattice
		indices = {'i': ['i'],'<ij>':['i','j'],'i<j':['i','j'],'i...j':['i','j']} # allowed symbolic indices and maximum number of body site interactions

		# Get identity operator I, to be maintained with same shape of data for Euler identities
		# with minimal redundant copying of data
		I = 'I'

		# Get number of operators
		size = min(len(i) for i in [operator,site,string,interaction])
		
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
					_operator_ = deepcopy(_operator)
					_string_ = deepcopy(_string)
					_interaction_ = deepcopy(_interaction)
					
					site.append(_site_)
					operator.append(_operator_)
					string.append(_string_)
					interaction.append(_interaction_)

			elif len(_operator) == len(_site):
				_site_ = deepcopy(_site)
				_operator_ = deepcopy(_operator)
				_string_ = deepcopy(_string)
				_interaction_ = deepcopy(_interaction)

				site.append(_site_)
				operator.append(_operator_)
				string.append(_string_)
				interaction.append(_interaction_)
			else:
				_site_ = deepcopy(_site)
				_operator_ = deepcopy(_operator)
				_string_ = deepcopy(_string)
				_interaction_ = deepcopy(_interaction)

				site.append(_site_)
				operator.append(_operator_)
				string.append(_string_)
				interaction.append(_interaction_)				


		# Set class attributes
		self.__extend__(operator=operator,site=site,string=string,interaction=interaction)

		# Set parameters
		self.__initialize__()

		# Set functions
		self.__functions__()

		return


	# @partial(jit,static_argnums=(0,))
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
		P = self.P
		parameters = array(trotter(parameters,P))

		# Get reshaped parameters (transpose for shape (K,M) to (M,K) and reshape to (MK,) with periodicity of data)
		shape = parameters.T.shape
		parameters = parameters.T.ravel()

		return parameters



class Unitary(Hamiltonian):
	'''
	Unitary class of Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string,interaction dictionary for operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		operator (iterable[str]): string names of operators
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		delta (float): Simulation length scale				
		M (int): Number of time steps
		T (int): Simulation Time
		tau (float): Simulation time scale		
		P (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space
		lattice (str,Lattice): Type of lattice
		parameters (str,dict,Parameters): Type of parameters	
		state (str,dict,State): Type of state	
		noise (str,dict,Noise): Type of noise
		label (str,dict,Gate): Type of label
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,P=None,
				space=None,time=None,lattice=None,parameters=None,state=None,noise=None,label=None,system=None,**kwargs):
		
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,P=P,
				space=space,time=time,lattice=lattice,parameters=parameters,state=state,noise=noise,label=label,system=system,**kwargs)

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
	def __grad_analytical__(self,parameters):
		'''
		Class gradient
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function
		'''	

		# Get class attributes
		attributes = self.parameters.attributes

		# Get shape and indices of variable parameters for gradient
		attribute = 'shape'
		layer = 'variables'
		shape = attributes[attribute][layer]


		# Get data
		data = self.data
		dtype = self.dtype

		# Get trotterized shape
		P = self.P
		shape = list(shape[::-1])

		shape[-1] *= P

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

		coefficients = self.coefficients
		data = array(trotter([datum() for datum in data],P))
		identity = self.identity()

		grad = gradient_expm(coefficients*parameters,data,identity)
		grad *= coefficients

		# Reshape gradient
		axis = 1

		grad = grad.reshape((*shape,*grad.shape[axis:]))

		grad = grad.transpose(axis,0,*[i for i in range(grad.ndim) if i not in [0,axis]])

		grad = array(gradient_trotter(grad,P))

		grad = grad[indices]

		grad = grad.reshape((-1,*grad.shape[axis+1:]))

		return grad

	#@partial(jit,static_argnums=(0,))
	def grad_analytical(self,parameters):
		'''
		Class gradient
		Args:
			parameters (array): parameters		
		Returns
			out (array): Return of function
		'''	
		return self.__grad_analytical__(parameters)


class Callback(object):
	def __init__(self,*args,**kwargs):
		'''	
		Class for callback
		Args:
			args (tuple): Class arguments
			kwargs (dict): Class keyword arguments
		'''
		self.defaults = {
			'iteration':[],
			'parameters':[],'grad':[],'search':[],
			'value':[],'objective':[],
			'alpha':[],'beta':[],

			'iteration.max':[],'iteration.min':[],
			'variables':[],'variables.mean':[],'variables.relative':[],
			'features':[],'features.mean':[],'features.relative':[],
			'objective.ideal.noise':[],'objective.diff.noise':[],'objective.rel.noise':[],
			'objective.ideal.state':[],'objective.diff.state':[],'objective.rel.state':[],
			'objective.ideal.operator':[],'objective.diff.operator':[],'objective.rel.operator':[],
			'hessian':[],'fisher':[],
			'hessian.eigenvalues':[],'fisher.eigenvalues':[],
			'hessian.rank':[],'fisher.rank':[],

			'N':[],'D':[],'d':[],'L':[],'delta':[],'M':[],'T':[],'tau':[],'P':[],
			'space':[],'time':[],'lattice':[],'architecture':[],'timestamp':[],

			'noise.scale':[],'optimize.c1':[],'optimize.c2':[],

		}

		return

	def __call__(self,parameters,track,optimizer,model,metric,func,grad):
		''' 
		Callback
		Args:
			parameters (array): parameters
			track (dict): callback tracking
			optimizer (Optimizer): callback optimizer
			model (object): Model instance
			metric (str,callable): Callback metric
			func (callable): Objective function with signature func(parameters)
			grad (callable): Objective gradient with signature grad(parameters)
		Returns:
			status (int): status of callback
		'''
		attributes = optimizer.attributes
		iterations = optimizer.iterations
		hyperparameters = optimizer.hyperparameters

		init = (len(attributes['iteration'])==1) and ((attributes['iteration'][-1]==0) or (attributes['iteration'][-1] != (iterations.stop)))
		
		done = (len(attributes['iteration'])>1) and (attributes['iteration'][-1] == (iterations.stop))
		
		status = (
			((len(attributes['value']) >= 1) and 
			 (attributes['iteration'][-1] <= max(1,
			 	hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) or
			(
			(abs(attributes['value'][-1]) > 
				(hyperparameters['eps']['value']*hyperparameters['value']['value'])) and
			(abs(attributes['value'][-1] - attributes['value'][-2]) > 
				(hyperparameters['eps']['value.difference']*attributes['value'][-2])) and
			(norm(attributes['grad'][-1])/attributes['grad'][-1].size > 
				  (hyperparameters['eps']['grad']*hyperparameters['value']['grad'])) and
			(norm(attributes['grad'][-1] - attributes['grad'][-2])/attributes['grad'][-2].size > 
				  (hyperparameters['eps']['grad.difference']*norm(attributes['grad'][-2])/attributes['grad'][-2].size))
			)
			)

		other = ((len(attributes['iteration']) == 1) or 
			(hyperparameters['modulo']['track'] is None) or 
			(attributes['iteration'][-1]%hyperparameters['modulo']['track'] == 0))

		stop = (
			(hyperparameters['eps'].get('value.increase') is not None) and
			((len(attributes['value']) > 1) and 
			 (attributes['iteration'][-1] >= max(1,
			 	hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) and			
			((attributes['value'][-1] - attributes['value'][-2]) > 
			(hyperparameters['eps']['value.increase']*attributes['value'][-2]))
			)

		status = (status) and (not stop)

		updates = {'iteration.max':True,'iteration.min':True}

		attrs = relsort(track,attributes)
		size = min(len(track[attr]) for attr in track)
		default = nan

		if ((not status) or done or init or other):
			
			for attr in attrs:

				if ((hyperparameters['length']['track'] is not None) and 
					(len(track[attr]) > hyperparameters['length']['track'])
					):
					_value = track[attr].pop(0)
				
				index = -1 if (not stop) else -2
				parameters = attributes['parameters'][index]
				value = default

				params = parameters
				assert ((index == -1) and allclose(parameters,params)) or (index == -2)

				if attr in attributes:
					value = attributes[attr][index]
				
				track[attr].append(value)

				if attr in ['iteration.max']:
					value = track['iteration'][index]

				elif attr in ['iteration.min']:
					value = track['iteration'][argmin(array(track['objective']))]

				elif attr in ['value']:
					value = attributes[attr][index]

				elif attr in ['parameters','grad','search'] and not ((not status) or done or init):
					value = empty(track[attr][index].shape)

				elif attr in ['parameters','grad','search'] and ((not status) or done or init):
					value = attributes[attr][index]

				elif attr in ['variables','features'] and not ((not status) or done or init):
					value = empty(track[attr][index].shape)

				elif attr in ['variables.mean','variables.relative.mean','features.mean','features.relative.mean'] and not ((not status) or done or init):
					value = default

				elif attr in ['variables','variables.mean','variables.relative.mean','features','features.mean','features.relative.mean'] and ((not status) or done or init):

					layer = attr.split(delim)[0]
					prop = 'index'
					indices = model.parameters.attributes[prop][layer]
					indices = tuple([(
						slice(
						min(indices[parameter][group][axis].start
							for parameter in indices 
							for group in indices[parameter]),
						max(indices[parameter][group][axis].stop
							for parameter in indices 
							for group in indices[parameter]),
						min(indices[parameter][group][axis].step
							for parameter in indices 
							for group in indices[parameter]))
						if all(isinstance(indices[parameter][group][axis],slice)
							for parameter in indices 
							for group in indices[parameter]) else
						list(set(i 
							for parameter in indices 
							for group in indices[parameter] 
							for i in indices[parameter][group][axis]))
						)
							for axis in range(min(len(indices[parameter][group]) 
											for parameter in indices 
											for group in indices[parameter]))
						])

					if attr in ['variables','features']:
						value = model.__layers__(parameters,layer)[indices]
					
					elif attr in ['variables.relative','features.relative']:
						eps = 1e-20
						value = model.__layers__(parameters,layer)[indices]
						_value = model.__layers__(attributes['parameters'][0],layer)[indices]
						value = abs((value - _value + eps)/(_value + eps))
					
					elif attr in ['variables.relative.mean','features.relative.mean']:
						eps = 1e-20
						value = model.__layers__(parameters,layer)[indices]
						_value = model.__layers__(attributes['parameters'][0],layer)[indices]						
						value = abs((value - _value + eps)/(_value + eps)).mean()

				elif attr in ['objective']:
					value = metric(model(parameters))
				
				elif attr in [
					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
					'objective.ideal.state','objective.diff.state','objective.rel.state',
					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and not ((not status) or done or init):
					value = default


				elif attr in [
					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
					'objective.ideal.state','objective.diff.state','objective.rel.state',
					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and ((not status) or done or init):

					_kwargs = {kwarg: {prop: hyperparameters.get('kwargs',{}).get(kwarg) if kwarg in ['noise'] else None for prop in ['scale']} for kwarg in ['state','noise','label']}
					_kwargs = {kwarg: {prop: getattrs(model,[kwarg,prop],delimiter=delim,default=_kwargs[kwarg][prop]) for prop in _kwargs[kwarg]} for kwarg in ['state','noise','label']}
					if attr in ['objective.ideal.noise','objective.diff.noise','objective.rel.noise']:
						_kwargs = {kwarg: False if kwarg in [] else _kwargs[kwarg] for kwarg in _kwargs}
						_metric = 'real'
					elif attr in ['objective.ideal.state','objective.diff.state','objective.rel.state']:						
						_kwargs = {kwarg: False if kwarg in ['noise'] else _kwargs[kwarg] for kwarg in _kwargs}
						_metric = 'real'
					elif attr in ['objective.ideal.operator','objective.diff.operator','objective.rel.operator']:
						_kwargs = {kwarg: False if kwarg in ['noise','state'] else _kwargs[kwarg] for kwarg in _kwargs}
						_metric = 'abs2'

					_model = model
					_shapes = model.shapes
					_label = model.label()
					_optimize = None
					_hyperparameters = hyperparameters
					_system = model.system
					_restore = {kwarg: deepcopy(getattr(model,kwarg)) for kwarg in _kwargs}

					_model.__functions__(**_kwargs)
					_metric = Metric(_metric,shapes=_shapes,label=_label,optimize=_optimize,hyperparameters=_hyperparameters,system=_system,verbose=False)

					if attr in ['objective.ideal.noise','objective.ideal.state','objective.ideal.operator']:
						value = _metric(_model(parameters))
					elif attr in ['objective.diff.noise','objective.diff.state','objective.diff.operator']:
						value = abs((track['objective'][index] - _metric(_model(parameters))))
					elif attr in ['objective.rel.noise','objective.rel.state','objective.rel.operator']:
						value = abs((track['objective'][index] - _metric(_model(parameters)))/(track['objective'][index]))

					model.__functions__(**_restore)


				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues'] and not ((not status) or done):
					if attr in ['hessian','fisher']:
						value = empty((*parameters.shape,)*2)
					elif attr in ['hessian.eigenvalues','fisher.eigenvalues']:
						value = empty((*parameters.shape,)*1)

				elif attr in ['hessian.rank','fisher.rank'] and not ((not status) or done):
					value = default

				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and ((not status) or done):
					
					if attr in ['hessian','hessian.eigenvalues','hessian.rank']:
						function = hessian(jit(lambda parameters: metric(model(parameters))))
					elif attr in ['fisher','fisher.eigenvalues','fisher.rank']:
						function = fisher(model,model.grad,shapes=(model.shape,(*model.dimensions,*model.shape)))

					if attr in ['hessian','fisher']:
						value = function(parameters)

					elif attr in ['hessian.eigenvalues','fisher.eigenvalues']:
						value = sort(abs(eig(function(parameters),compute_v=False,hermitian=True)))[::-1]
						value = value/max(1,maximum(value))
					elif attr in ['hessian.rank','fisher.rank']:
						value = sort(abs(eig(function(parameters),compute_v=False,hermitian=True)))[::-1]
						value = argmax(abs(difference(value)/value[:-1]))+1	
						value = value.size if (value==value.size-1) else value

				elif attr not in attributes and not (getter(hyperparameters,attr.replace('optimize%s'%(delim),''),default=null,delimiter=delim) is null):
					value = getter(hyperparameters,attr.replace('optimize%s'%(delim),''),default=default,delimiter=delim)

				elif attr not in attributes and hasattrs(model,attr,delimiter=delim):
					value = getattrs(model,attr,default=default,delimiter=delim)

				
				track[attr][-1] = value


				if updates.get(attr):
					update = updates[attr]
					if not callable(update):
						update = lambda i,attr,track: track[attr][-1]
					for i in range(size+1):
						track[attr][i] = update(i,attr,track)


		log = ((len(attributes['iteration']) == 1) or 
			(hyperparameters['modulo']['log'] is None) or 
			(attributes['iteration'][-1]%hyperparameters['modulo']['log'] == 0)
			)

		if log:

			msg = '\n'.join([
				'%d f(x) = %0.4e'%(
					attributes['iteration'][-1],
					track['objective'][-1],
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
				'U\n%s\nV\n%s'%(
				# to_string(abs(model(parameters)).round(4)),
				# to_string(abs(model.label()).round(4))),
				to_string((model(parameters)).round(4)),
				to_string((model.label()).round(4))),
				])


			model.log(msg)


		return status




from typing import List,Tuple
import equinox as nn

class module(nn.Module):
	pass

class OpModule(module,System):
	'''
	Class for Observable
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
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
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

	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
					N=None,D=None,space=None,system=None,**kwargs):

		setter(kwargs,system,delimiter=delim,func=False)
		super().__init__(**kwargs)

		self.N = N
		self.D = D
		self.space = space
		self.system = system

		self.__space__()
		self.__setup__(data,operator,site,string,interaction)
		
		return

	
	def __space__(self,N=None,D=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			space (str,Space): Type of Hilbert space
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
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
		self.size = int(product(self.shape))
		self.ndim = len(self.shape)
		
		self.identity = Operator(N=self.N,D=self.D,system=self.system)

		return
	
	def __str__(self):
		try:
			return str(self.string)
		except:
			return self.__class__.__name__

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
			operator = dot((parameters*self.data),state)
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
				operator = dot(parameters*self.data,state)
				operator = operator.reshape(self.shape)
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
		# TODO Implement SWAP
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

		# TODO Implement RESHAPE
		raise NotImplementedError

		locality = len(site)
		axes = range(locality)
		shape = (self.n,)

		state = moveaxis(state.reshape(shape),site,axes)

		return state


	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
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
		'''

		dtype = 'complex'
		basis = {
			'I': array([[1,0],[0,1]]),
			'X': array([[0,1],[1,0]]),
			'Y': array([[0,-1j],[1j,0]]),
			'Z': array([[1,0],[0,-1]]),
		}

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
		
		if isinstance(self.data,dict):
			for attr in self.data:
				setattr(self,attr,self.data[attr])

		self.data = self.operator
		if not is_array(self.data):
			self.data = [basis[operator] for operator in self.data]
			self.data = tensorprod(self.data)

		self.data = self.data.astype(self.dtype)

		return