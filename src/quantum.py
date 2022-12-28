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
from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,tensordot,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer,product,dot,einsum
from src.utils import summation,exponentiation,summationv,exponentiationv,summationm,exponentiationm,summationc,exponentiationc,summationmc,exponentiationmc
from src.utils import trotter,gradient_trotter,gradient_expm,gradient_sigmoid
from src.utils import inner_norm,inner_abs2,inner_real,inner_imag
from src.utils import gradient_inner_norm,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag
from src.utils import eig
from src.utils import maximum,minimum,argmax,argmin,difference,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,relsort,norm,interpolate,unique,allclose,isclose,is_array,is_naninf,to_key_value 
from src.utils import initialize,parse,to_string,to_number,datatype,slice_size,intersection
from src.utils import pi,e,nan,null,delim,scalars,nulls
from src.utils import itg,flt,dbl

from src.iterables import setter,getattrs,hasattrs
from src.iterables import leaves,counts,plant,grow

from src.parameters import Parameters
from src.operators import Operator
from src.states import State
from src.noise import Noise

from src.io import load,dump,join,split

from src.system import System,Space,Time,Lattice

from src.optimize import Objective,Metric

dtype = 'complex'
basis = {
	'I': array([[1,0],[0,1]],dtype=dtype),
	'X': array([[0,1],[1,0]],dtype=dtype),
	'Y': array([[0,-1j],[1j,0]],dtype=dtype),
	'Z': array([[1,0],[0,-1]],dtype=dtype),
}

class Object(System):
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
		parameters (str,dict,Parameters): Type of parameters	
		state (str,dict,State): Type of state	
		noise (str,dict,Noise): Type of noise
		label (str,dict,Operator): Type of label	
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,
		N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,
		space=None,time=None,lattice=None,parameters=None,state=None,noise=None,label=None,system=None):

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

		self.data = []
		self.operator = []
		self.site = []
		self.string = []
		self.interaction = []
		self.indices = []		
		self.dims = (len(self.data),self.M)
		self.length = int(product(self.dims))
		self.ndims = len(self.dims)

		self.key = None

		self.timestamp = None
		self.backend = None
		self.architecture = None
		self.delimiter = ' '
		self.shape = ()
		self.size = int(product(self.shape))
		self.ndim = len(self.shape)
		self.shapes = (self.shape,self.shape)

		self.parameters = parameters
		self.state = state
		self.noise = noise
		self.label = label
		self.identity = None
		self.constants = None
		self.coefficients = 1
		self.dimensions = None	

		self.summation = None
		self.exponentiation = None 

		super().__init__(**system)

		self.__space__()
		self.__time__()
		self.__lattice__()

		self.__setup__(data,operator,site,string,interaction)
		self.__initialize__()
		self.__functions__()
		
		self.info()

		return	

	def __setup__(self,data={},operator=None,site=None,string=None,interaction=None):
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

		operator.extend([data[name]['operator'] for name in data])
		site.extend([data[name]['site'] for name in data])
		string.extend([data[name]['string'] for name in data])
		interaction.extend([data[name]['interaction'] for name in data])

		size = min([len(i) for i in [operator,site,string,interaction]])

		data = [self.identity.copy() for i in range(size)]

		# Update class attributes
		self.__extend__(data,operator,site,string,interaction)
		
		return


	def __append__(self,data,operator,site,string,interaction):
		'''
		Append to class
		Args:
			data (array): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''
		index = -1
		self.__insert__(index,data,operator,site,string,interaction)
		return

	def __extend__(self,data,operator,site,string,interaction):
		'''
		Setup class
		Args:
			data (iterable[array]): data of operator
			operator (iterable[str]): string names of operators
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','<ij>','i<j','i...j']
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''
		for _data,_operator,_site,_string,_interaction in zip(data,operator,site,string,interaction):
			self.__append__(_data,_operator,_site,_string,_interaction)

		return


	def __insert__(self,index,data,operator,site,string,interaction):
		'''
		Insert to class
		Args:
			index (int): index to insert operator
			data (array): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			interaction (str): interaction type of operators type of interaction, i.e) nearest neighbour, allowed values in ['i','i,j','i<j','i...j']
		'''

		if index == -1:
			index = len(self.data)

		self.data.insert(index,data)
		self.operator.insert(index,operator)
		self.site.insert(index,site)
		self.string.insert(index,string)
		self.interaction.insert(index,interaction)

		self.dims = (len(self.data),self.M,*self.shape)
		self.ndims = len(self.dims)		

		return



	def __initialize__(self,parameters=None):
		''' 
		Setup initial parameters and attributes
		Args:
			parameters (array): parameters
		'''

		# Get attributes data of parameters of the form {attribute:{parameter:{group:{layer:[]}}}
		data = self.parameters
		shape = (len(self.data)//self.p,self.M)
		hyperparameters = self.parameters.hyperparameters is isinstance(self.parameters,Parameters) else self.parameters if isinstance(self.parameters,dict) else {}
		check = lambda group,index,axis,site=self.site,string=self.string: (
			(axis != 0) or 
			any(g in group for g in [string[index],'_'.join([string[index],''.join(['%d'%j for j in site[index]])])]))
		size = product(shape)
		samples = None
		seed = self.seed
		cls = self
		dtype = self.dtype

		parameters = Parameters(data,shape,hyperparameters,check=check,initialize=initialize,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)

		# Get coefficients
		coefficients = -1j*2*pi/2*self.tau/self.p		

		# Update class attributes
		self.parameters = parameters
		self.coefficients = coefficients
		self.dimensions = parameters.shape

		return

	def __functions__(self,state=None,noise=None,label=None):
		''' 
		Setup class functions
		Args:
			state (bool,array): State to act on with class of shape self.shape, if boolean choose self.state or None
			noise (bool,array): Noise to act on with class of shape (-1,self.shape), if boolean choose self.noise or None
			label (bool,array): Label of class of shape self.shape, if boolean choose self.label or None
		'''

		# Function arguments
		data = array(self.data,dtype=self.dtype)
		identity = self.identity
		state = self.state if (state is None or state is True) else state if state is not False else None
		noise = self.noise if (noise is None or noise is True) else noise if noise is not False else None
		label = self.label if (label is None or label is True) else label if label is not False else None

		shape = self.shape
		size = self.N
		seed = self.seed		
		dtype = self.dtype
		cls = self

		# Get state
		data = state
		hyperparameters = data.hyperparameters if isinstance(data,State) else data if isinstance(data,dict) else {}
		samples = True
		self.state = State(data,shape,hyperparameters,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)
		
		# Get noise
		data = noise
		hyperparameters = data.hyperparameters if isinstance(data,Noise) else data if isinstance(data,dict) else {}
		samples = None
		self.noise = Noise(data,shape,hyperparameters,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)

		# Get label
		data = label
		hyperparameters = data.hyperparameters if isinstance(data,Operator) else data if isinstance(data,dict) else {}
		samples = None
		self.label = Operator(data,shape,hyperparameters,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)

		# Attribute values
		if self.state() is None:
			state = self.state()
			noise = self.noise()
			label = self.label()
			shapes = (self.shape,self.shape)
		elif self.state.ndim == 1:
			state = self.state()
			noise = self.noise()
			label = einsum('ij,j->i',self.label(),self.state())
			shapes = ((self.n,),(self.n,))			
		elif self.state.ndim == 2:
			state = self.state()
			noise = noise
			label = einsum('ij,jk,lk->il',self.label(),self.state(),self.label())
			shapes = (self.shape,self.shape)			
		else:
			state = self.state()
			noise = self.noise()
			label = self.label()
			shapes = (self.shape,self.shape)			

		state = self.state(state)
		noise = self.noise(noise)
		label = self.label(label.conj())
		shapes = self.shapes

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


	@partial(jit,static_argnums=(0,))
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


	def __space__(self,N=None,D=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			space (str,Space): Type of Hilbert space
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)		
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
		self.dims = (len(self.data),self.M,*self.shape)
		self.length = int(product(self.dims))
		self.ndims = len(self.dims)
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
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)		
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
		self.dims = (*self.dims[:1],self.M,*self.dims[2:])	
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
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)		
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

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		
		msg = '%s'%('\n'.join([
			*['%s: %s'%(attr,getattr(self,attr)) 
				for attr in ['key','seed','N','D','d','L','delta','M','tau','T','p','shape','dims','cwd','path','backend','architecture','conf','logging']
			],
			*['%s: %s'%(attr,getattr(self,attr) is not None) 
				for attr in ['state','noise']
			],
			*['%s: %s'%(attr,getattr(self,attr).__name__) 
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
		parameters (str,dict,Parameters): Type of parameters	
		state (str,dict,State): Type of state	
		noise (str,dict,Noise): Type of noise
		label (str,dict,Operator): Type of label
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,
				space=None,time=None,lattice=None,parameters=None,state=None,noise=None,label=None,system=None):
		
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,p=p,
				space=space,time=time,lattice=lattice,parameters=parameters,state=state,noise=noise,label=label,system=system)
		
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

	def __setup__(self,data={},operator=None,site=None,string=None,interaction=None):
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

		operator.extend([data[name]['operator'] for name in data])
		site.extend([data[name]['site'] for name in data])
		string.extend([data[name]['string'] for name in data])
		interaction.extend([data[name]['interaction'] for name in data])

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
		self.__extend__(data,operator,site,string,interaction)

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
		parameters (str,dict,Parameters): Type of parameters	
		state (str,dict,State): Type of state	
		noise (str,dict,Noise): Type of noise
		label (str,dict,Operator): Type of label
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)
	'''

	def __init__(self,data={},operator=None,site=None,string=None,interaction=None,
				N=None,D=None,d=None,L=None,delta=None,M=None,T=None,tau=None,p=None,
				space=None,time=None,lattice=None,parameters=None,state=None,noise=None,label=None,system=None):
		
		super().__init__(data=data,operator=operator,site=site,string=string,interaction=interaction,
				N=N,D=D,d=d,L=L,delta=delta,M=M,T=T,tau=tau,p=p,
				space=space,time=time,lattice=lattice,parameters=parameters,state=state,noise=noise,label=label,system=system)

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

		coefficients = self.coefficients
		data = array(self.data,dtype=self.dtype)
		identity = self.identity

		grad = gradient_expm(coefficients*parameters,data,identity)
		grad *= coefficients

		# Reshape gradient
		axis = 1

		grad = grad.reshape((*shape,*grad.shape[axis:]))

		grad = grad.transpose(axis,0,*[i for i in range(grad.ndim) if i not in [0,axis]])

		grad = array(gradient_trotter(grad,p))

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
			"iteration":[],
			"parameters":[],"grad":[],"search":[],
			"value":[],"objective":[],
			"alpha":[],"beta":[],

			"iteration.max":[],"iteration.min":[],
			"features":[],"features.mean":[],"features.relative":[],
			"objective.ideal.noise":[],"objective.diff.noise":[],"objective.rel.noise":[],
			"objective.ideal.state":[],"objective.diff.state":[],"objective.rel.state":[],
			"objective.ideal.operator":[],"objective.diff.operator":[],"objective.rel.operator":[],
			"hessian":[],"fisher":[],
			"hessian.eigenvalues":[],"fisher.eigenvalues":[],
			"hessian.rank":[],"fisher.rank":[],

			"N":[],"D":[],"d":[],"L":[],"delta":[],"M":[],"T":[],"tau":[],"p":[],
			"space":[],"time":[],"lattice":[],"architecture":[],

			"noise.scale":[],"optimize.c1":[],"optimize.c2":[],

		}		
		return

	def __call__(self,parameters,track,attributes,model,metric,func,grad,hyperparameters):
		''' 
		Callback
		Args:
			parameters (array): parameters
			track (dict): callback tracking
			attributes (dict): Callback attributes
			model (object): Model instance
			metric (str,callable): Callback metric
			func (callable): Objective function with signature func(parameters)
			grad (callable): Objective gradient with signature grad(parameters)
			hyperparameters(dict): Callback hyperparameters
		Returns:
			status (int): status of callback
		'''

		start = (len(attributes['iteration'])==1) and (attributes['iteration'][-1]<hyperparameters['iterations'])
		
		done = (len(attributes['iteration'])>0) and (attributes['iteration'][-1]==hyperparameters['iterations'])
		
		status = (
			(abs(attributes['value'][-1]) > 
				(hyperparameters['eps']['value']*hyperparameters['value']['value'])) and
			((len(attributes['value'])==1) or 
			 ((len(attributes['value'])>1) and 
			 (abs(attributes['value'][-1] - attributes['value'][-2]) > 
				(hyperparameters['eps']['difference']*attributes['value'][-2])))) and
			((len(attributes['value'])==1) or 			
			 ((len(attributes['grad'])>1) and
			(norm(attributes['grad'][-1] - attributes['grad'][-2])/attributes['grad'][-2].size > 
				  (hyperparameters['eps']['grad']*norm(attributes['grad'][-2])/attributes['grad'][-2].size))))
			)

		other = ((len(attributes['iteration']) == 0) or 
			(hyperparameters['modulo']['track'] is None) or 
			(attributes['iteration'][-1]%hyperparameters['modulo']['track'] == 0))

		default = nan

		if ((not status) or done or start or other):

			attrs = relsort(track,attributes)
			size = max(len(track[attr]) for attr in track)

			for attr in attrs:

				if ((hyperparameters['length']['track'] is not None) and 
					(len(track[attr]) > hyperparameters['length']['track'])
					):
					_value = track[attr].pop(0)
				
				value = default

				if attr in attributes:
					value = attributes[attr][-1]
				
				track[attr].append(value)

				if attr in ['iteration.max']:
					value = track['iteration'][-1]

				elif attr in ['iteration.min']:
					value = track['iteration'][argmin(array(track['objective']))]

				elif attr in ['parameters','grad','search'] and not ((not status) or done or start):
					value = default

				elif attr in ['parameters','grad','search'] and ((not status) or done or start):
					value = attributes[attr][-1]

				elif attr in ['features','features.mean','features.relative']and not ((not status) or done or start):
					value = default

				elif attr in ['features','features.mean','features.relative'] and ((not status) or done or start):

					layer = 'features'
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

					if attr in ['features']:
						value = model.__layers__(parameters,layer)[indices]
					
					elif attr in ['features.relative']:
						eps = 1e-20
						value = model.__layers__(parameters,layer)[indices]
						value = abs((value - track['features'][0] + eps)/(track['features'][0] + eps))
					
					elif attr in ['features.relative']:
						eps = 1e-20
						value = model.__layers__(parameters,layer)[indices]
						value = abs((value - track['features'][0] + eps)/(track['features'][0] + eps)).mean(-1)


				elif attr in ['objective']:
					value = metric(model(parameters))

				elif attr in [
					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
					'objective.ideal.state','objective.diff.state','objective.rel.state',
					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and not ((not status) or done or start):
					value = default

				elif attr in [
					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
					'objective.ideal.state','objective.diff.state','objective.rel.state',
					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and ((not status) or done or start):

					if model.state() is None:
						state = {'scale':1}
					else:
						state = model.state

					if model.noise() is None:
						noise = {'scale':1}
					else:
						noise = model.noise

					label = True

					if attr in ['objective.ideal.noise','objective.diff.noise','objective.rel.noise']:
						_kwargs = {'state':state,'noise':noise,'label':label}
						_metric = 'real'
					elif attr in ['objective.ideal.state','objective.diff.state','objective.rel.state']:						
						_kwargs = {'state':state,'noise':False,'label':label}
						_metric = 'real'
					elif attr in ['objective.ideal.operator','objective.diff.operator','objective.rel.operator']:
						_kwargs = {'state':False,'noise':False,'label':label}
						_metric = 'abs2'

					_model = model
					_shapes = model.shapes
					_label = _model.label()
					_optimize = None
					_hyperparameters = hyperparameters

					model.__functions__(**kwargs)
					_metric = Metric(_metric,shapes=_shapes,label=_label,optimize=_optimize,hyperparameters=_hyperparameters)

					if attr in ['objective.ideal.noise','objective.ideal.state','objective.ideal.operator']:
						value = _metric(_model(parameters))
					elif attr in ['objective.diff.noise','objective.diff.state','objective.diff.operator']:
						value = abs((track['objective'][-1] - _metric(_model(parameters))))
					elif attr in ['objective.rel.noise','objective.rel.state','objective.rel.operator']:
						value = abs((track['objective'][-1] - _metric(_model(parameters)))/(track['objective'][-1]))

					model.__functions__()

				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and not ((not status) or done):
					value = default

				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and ((not status) or done):
					
					if attr in ['hessian','hessian.eigenvalues','hessian.rank']:
						function = hessian(jit(lambda parameters: metric(model(parameters))))
					elif attr in ['fisher','fisher.eigenvalues','fisher.rank']:
						function = fisher(model,model.grad,shapes=(model.shape,(model.length,*model.shape)))

					if attr in ['hessian','fisher']:
						value = function(parameters)

					elif attr in ['hessian.eigenvalues','fisher.eigenvalues']:
						value = sort(abs(eig(function(parameters),compute_v=False,hermitian=True)))[::-1]
						value = value/max(1,maximum(value))
					elif attr in ['hessian.rank','fisher.rank']:
						value = sort(abs(eig(function(parameters),compute_v=False,hermitian=True)))[::-1]
						value = argmax(abs(difference(value)/value[:-1]))+1						

				elif hasattrs(model,attr,delimiter=delim):
					value = getattrs(model,attr,default=default,delimiter=delim)


			track[attr][-1] = value


		log = ((len(attributes['iteration']) == 0) or 
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
				to_string(abs(model(parameters)).round(4)),
				to_string(abs(model.label()).round(4))),
				])


			model.log(msg)


			# print(parameters.reshape(-1,model.M)) # Raw parameters have shape (-1,M)
			# print(model.__layers__(parameters,'variables').T.reshape(model.M,-1))



		return status




from typing import List,Tuple
import equinox as nn

class module(nn.Module):
	pass

class Operator(module,System):
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
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)
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
					N=None,D=None,space=None,system=None):

		super(System).__init__(**system)

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
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)		
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