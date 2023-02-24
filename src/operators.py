#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools
from copy import deepcopy
from functools import partial

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,ones,zeros,arange,eye,rand,identity,diag,PRNGKey,sigmoid,abs,qr,sqrt
from src.utils import einsum,tensorprod,trace,broadcast_to,padding,expand_dims,moveaxis,repeat,take,inner,outer
from src.utils import slice_slice,datatype,returnargs,is_array,allclose
from src.utils import pi,e,delim

from src.system import Object,System
from src.io import load,dump,join,split



def id(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize identity unitary operator
	Args:
		shape (int,iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = identity(shape)

	data = data.astype(dtype)

	return data

def cnot(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize cnot unitary operator
	Args:
		shape (int,iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = array([
		[1,0,0,0],
		[0,1,0,0],
		[0,0,0,1],
		[0,0,1,0]])

	data = data.astype(dtype)

	return data


def hadamard(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize hadamard unitary operator
	Args:
		shape (int,iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = array([
		[1,1,],
		[1,-1]])/sqrt(2)

	data = data.astype(dtype)

	return data	


def toffoli(shape,bounds=None,random=None,seed=None,dtype=None,):
	'''
	Initialize toffoli unitary operator
	Args:
		shape (int,iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = array([
		[1,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0],
		[0,0,1,0,0,0,0,0],
		[0,0,0,1,0,0,0,0],
		[0,0,0,0,1,0,0,0],
		[0,0,0,0,0,1,0,0],
		[0,0,0,0,0,0,0,1],
		[0,0,0,0,0,0,1,0]])

	data = data.astype(dtype)

	return data	


class Gate(Object):
	def __init__(self,data,shape,size=None,dims=None,system=None,**kwargs):
		'''
		Initialize data of attribute based on shape, with highest priority of arguments of: kwargs,args,data,system
		Args:
			data (dict,str,array,Noise): Data corresponding to noise
			shape (int,iterable[int]): Shape of each data
			size (int,iterable[int]): Number of data
			dims (iterable[int]): Dimensions of N, D-dimensional sites [N,D]
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		super().__init__(data,shape,size=size,dims=dims,system=system,**kwargs)

		return

	def __setup__(self,**kwargs):
		'''
		Setup attribute
		Args:
			kwargs (dict): Additional keyword arguments
		'''

		# Size
		size = None
		self.size = size
		self.length = len(self.size) if self.size is not None else None

		# Delimiter for string
		delimiter = '_'

		# Properties for strings
		props = {
			**{string: {'func':rand,'locality':self.N} for string in ['random','U','haar']},
			**{string: {'func':hadamard,'locality':1} for string in ['hadamard','H']},
			**{string: {'func':cnot,'locality':2} for string in ['cnot','CNOT','C']},
			**{string: {'func':toffoli,'locality':3} for string in ['toffoli','TOFFOLI','T']},
			**{string: {'func':{1:id,2:cnot,3:toffoli}.get(self.N,id),'locality':self.N} for string in ['control']},
			None: {'func':rand,'locality':self.N},
			}


		if self.string is None:
			strings = [self.string]
			locality = self.N
		elif all(string in props for string in self.string.split(delimiter)):
			strings = self.string.split(delimiter)
			locality = sum(props[string]['locality'] for string in strings)
		else:
			strings = None
			locality = self.N			


		assert (self.N%locality == 0), 'Incorrect operator with locality %d !%% size %d'%(locality,self.N)

		if self.string is not None:
			data = tensorprod([
				props[string]['func'](self.shape,
					bounds=self.bounds,
					random=self.random,
					seed=self.seed,
					dtype=self.dtype
					)
				for string in strings
				]*(self.N//locality)
			)
		else:
			data = array(load(self.data))


		# Assert data is unitary
		if data.ndim == 2:
			normalization = einsum('...ij,...kj->...ik',data.conj(),data)
		else:
			normalization = einsum('...ij,...kj->...ik',data.conj(),data)

		assert allclose(eye(self.n,dtype=self.dtype),normalization), "Incorrect normalization data%r: %r"%(data.shape,normalization)

		self.data = data

		return




# from typing import List,Tuple
# import equinox as nn

# class module(nn.Module):
# 	pass

# class Op(module,System):
# 	'''
# 	Class for Observable
# 	Args:
# 		data (dict,str,array): dictionary of operator attributes, or string or array for operator. Allowed strings in ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI'], allowed dictionary keys in
# 			operator (str,iterable[str],array): string or array for operator, allowed ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI']
# 			site (iterable[int]): site of local operators
# 			string (str): string label of operator
# 			interaction (str): interaction type of operator
# 		operator (str,iterable[str],array): string or array for operator, allowed ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI']
# 		site (iterable[int]): site of local operators
# 		string (str): string label of operator
# 		interaction (str): interaction type of operator
# 		hyperparameters (dict) : class hyperparameters
# 		N (int): Number of qudits
# 		D (int): Dimension of qudits
# 		space (str,Space): Type of local space
# 		system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
# 		kwargs (dict): Additional system keyword arguments	
# 	'''

# 	data : None
# 	operator : str
# 	site : List[int]
# 	string : str
# 	interaction : str
# 	hyperparameters : dict
# 	N : int
# 	D : int
# 	space : str
# 	system : dict

# 	n : int
# 	shape : Tuple[int]
# 	size : int
# 	ndim : int
# 	locality : int

# 	identity : array

# 	dtype: str
# 	format : str
# 	seed : int
# 	key : List[int]
# 	timestamp : str
# 	backend : str
# 	architecture : str
# 	verbose : int

# 	def __init__(self,data=None,operator=None,site=None,string=None,interaction=None,
# 					N=None,D=None,space=None,system=None,**kwargs):

# 		setter(kwargs,system,delimiter=delim,func=False)
# 		super().__init__(**kwargs)

# 		self.N = N
# 		self.D = D
# 		self.space = space
# 		self.system = system

# 		self.__space__()
# 		self.__setup__(data,operator,site,string,interaction)
		
# 		return

	
# 	def __space__(self,N=None,D=None,space=None,system=None):
# 		'''
# 		Set space attributes
# 		Args:
# 			N (int): Number of qudits
# 			D (int): Dimension of qudits
# 			space (str,Space): Type of local space
# 			system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
# 		'''
# 		N = self.N if N is None else N
# 		D = self.D if D is None else D
# 		space = self.space if space is None else space
# 		system = self.system if system is None else system

# 		space = Space(N,D,space,system=system)

# 		self.N = space.N
# 		self.D = space.D		
# 		self.n = space.n
# 		self.shape = (self.n,self.n)
# 		self.size = int(product(self.shape))
# 		self.ndim = len(self.shape)
		
# 		self.identity = Operator(N=self.N,D=self.D,system=self.system)

# 		return
	
# 	def __str__(self):
# 		try:
# 			return str(self.string)
# 		except:
# 			return self.__class__.__name__

# 	def __repr__(self):
# 		return self.__str__()
	
# 	def __len__(self):
# 		return len(self.data)
	
# 	@nn.filter_jit
# 	def __call__(self,parameters,state=None):
# 		'''
# 		Return parameterized operator 
# 		Args:
# 			parameters (array): Parameters to parameterize operator			
# 			state (array): State to apply operator
# 		Returns
# 			operator (array): Parameterized operator
# 		'''		
# 		parameters = self.__parameters__(parameters)

# 		if state is None:
# 			operator = parameters*self.data
# 		else:
# 			operator = dot((parameters*self.data),state)
# 		return operator

# 	@nn.filter_jit
# 	def __parameters__(self,parameters):
# 		''' 
# 		Setup parameters
# 		Args:
# 			parameters (array): parameters
# 		Returns:
# 			parameters (array): parameters
# 		'''
# 		return parameters

# 	@nn.filter_jit
# 	def __apply__(self,parameters,state=None):
# 		'''
# 		Return parameterized operator 
# 		Args:
# 			parameters (array): Parameters to parameterize operator			
# 			state (array): State to apply operator
# 		Returns
# 			operator (array): Parameterized operator
# 		'''
		
# 		parameters = self.__parameters__(parameters)

# 		if state is None:
# 			operator = parameters*self.data
# 		else:

# 			for site in self.site:
# 				state = self.__swap__(state,site)
# 				operator = dot(parameters*self.data,state)
# 				operator = operator.reshape(self.shape)
# 				state = self.__reshape__(state,site)

# 		return operator

# 	@nn.filter_jit
# 	def __swap__(self,state,site):
# 		'''
# 		Swap axes of state at site
# 		Args:
# 			state (array): State to apply operator of shape (n,)
# 			site (iterable[int]): Axes to apply operator of size locality
# 		Returns
# 			state (array): State to apply operator of shape (*(D)*locality,n/D**locality)
# 		'''
# 		# TODO Implement SWAP
# 		raise NotImplementedError

# 		locality = len(site)
# 		axes = range(locality)
# 		shape = (*(self.D)*locality,-1)

# 		state = moveaxis(state,site,axes).reshape(shape)

# 		return state

# 	@nn.filter_jit
# 	def __reshape__(self,state,site):
# 		'''
# 		Reshape state to shape (n,)
# 		Args:
# 			state (array): State to apply operator of shape (n,)
# 			site (iterable[int]): Axes to apply operator
# 		Returns
# 			state (array): State to apply operator of shape (D,D,n/D**2)
# 		'''

# 		# TODO Implement RESHAPE
# 		raise NotImplementedError

# 		locality = len(site)
# 		axes = range(locality)
# 		shape = (self.n,)

# 		state = moveaxis(state.reshape(shape),site,axes)

# 		return state


# 	def __setup__(self,data=None,operator=None,site=None,string=None,interaction=None):
# 		'''
# 		Setup class
# 		Args:
# 			data (dict,str,array): dictionary of operator attributes, or string or array for operator. Allowed strings in ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI'], allowed dictionary keys in
# 						operator (str,iterable[str],array): string or array for operator, allowed ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI']
# 						site (iterable[int]): site of local operators
# 						string (str): string label of operator
# 						interaction (str): interaction type of operator
# 			operator (str,iterable[str],array): string or array for operator, allowed ['X','Y','Z','I','CNOT','HADAMARD','TOFFOLI']
# 			site (iterable[int]): site of local operators
# 			string (str): string label of operator
# 			interaction (str): interaction type of operator
# 		'''

# 		dtype = 'complex'
# 		basis = {
# 			'I': array([[1,0],[0,1]]),
# 			'X': array([[0,1],[1,0]]),
# 			'Y': array([[0,-1j],[1j,0]]),
# 			'Z': array([[1,0],[0,-1]]),
# 		}

# 		if isinstance(data,str):
# 			operator = data
# 		elif is_array(data):
# 			operator = data
				
# 		if operator is None:
# 			operator = ''
# 		if site is None:
# 			site = list(range(len(operator)))
# 		if string is None:
# 			string = '%s_%s'%(operator,''.join(['%d'%(i) for i in site]))
# 		else:
# 			string = '%s_%s'%(string,''.join(['%d'%(i) for i in site]))
# 		if interaction is None:
# 			interaction = ''
# 		if hyperparameters is None:
# 			hyperparameters = {}

# 		self.data = data
# 		self.operator = operator
# 		self.site = site
# 		self.string = string
# 		self.interaction = interaction
# 		self.locality = len(site)
		
# 		if isinstance(self.data,dict):
# 			for attr in self.data:
# 				setattr(self,attr,self.data[attr])

# 		self.data = self.operator
# 		if not is_array(self.data):
# 			self.data = [basis[operator] for operator in self.data]
# 			self.data = tensorprod(self.data)

# 		self.data = self.data.astype(self.dtype)

# 		return