#!/usr/bin/env python

# Import python modules
import os,sys
import traceback

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,partial,wraps,copy,vmap,vfunc,switch,forloop,cond,slicing,gradient,hessian,fisher,entropy,purity,similarity,divergence
from src.utils import array,asarray,asscalar,empty,identity,entity,ones,zeros,rand,spawn,arange,diag,inv
from src.utils import tensor,tensornetwork,gate,mps,representation
from src.utils import repeat,expand_dims
from src.utils import contraction,gradient_contraction
from src.utils import tensorprod,swap,conjugate,dagger,einsum,dot,reshape,transpose,dots,norm,real,imag,eig,trace,sort,relsort,prod,product,log
from src.utils import inplace,insertion,maximum,minimum,argmax,argmin,nonzero,difference,unique,cumsum,shift,interleaver,splitter,abs,abs2,mod,sqrt,log,log10,sign,sin,cos,exp
from src.utils import to_index,to_position,to_string,allclose,is_hermitian,is_unitary
from src.utils import pi,e,nan,null,delim,scalars,arrays,tensors,nulls,integers,floats,iterables,datatype

from src.iterables import Dict,Dictionary,setter,getter,getattrs,hasattrs,namespace,permutations

from src.io import load,dump,join,split

from src.system import System,Space,Time,Lattice

from src.parameters import Parameters,Parameter

from src.optimize import Objective,Metric

delim = '.'
separ = '_'
objects = (*arrays,*tensors)

class Basis(Dict):
	'''
	Basis Class of operators
	Args:
		D (int): Local dimension of system
		basis (str): Type of basis
		args (iterable): Additional class positional arguments
		kwargs (dict): Additional class keyword arguments
	'''

	D = 2

	shape = None
	size = None
	ndim = None
	dtype = None

	data = None
	parameters = None

	@classmethod
	def get(cls,attr):
		return getattr(cls,attr)

	@classmethod
	def set(cls,attr,value):
		setattr(cls,attr,value)
		return

	# General
	@classmethod
	@System.decorator
	def string(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.data is None:
			data = []
		elif isinstance(kwargs.data,str):
			data = kwargs.data.split(delim)
		else:
			data = [*kwargs.data]
		N = len(data)
		data = array(tensorprod([getattr(cls,i)(D=kwargs.D,dtype=kwargs.dtype) for i in data]),dtype=kwargs.dtype)
		if kwargs.parameters is not None:
			parameters = kwargs.parameters*pi/2
			identity = tensorprod([cls.identity(D=kwargs.D,dtype=kwargs.dtype)]*N)
			data = cos(parameters)*identity + -1j*sin(parameters)*data
		return data

	@classmethod
	@System.decorator
	def identity(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = identity(kwargs.D,dtype=kwargs.dtype)
		return data

	@classmethod
	@System.decorator
	def entity(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = entity(kwargs.D,dtype=kwargs.dtype)
		return data

	# Unitary
	@classmethod
	@System.decorator
	def I(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([[1,0],[0,1]],dtype=kwargs.dtype)
		if kwargs.parameters is not None:
			parameters = kwargs.parameters*pi/2
			identity = cls.identity(D=kwargs.D,dtype=kwargs.dtype)
			data = cos(parameters)*identity + -1j*sin(parameters)*data
		return data

	@classmethod
	@System.decorator
	def X(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([[0,1],[1,0]],dtype=kwargs.dtype)
		if kwargs.parameters is not None:
			parameters = kwargs.parameters*pi/2
			identity = cls.identity(D=kwargs.D,dtype=kwargs.dtype)
			data = cos(parameters)*identity + -1j*sin(parameters)*data
		return data

	@classmethod
	@System.decorator
	def Y(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([[0,-1j],[1j,0]],dtype=kwargs.dtype)
		if kwargs.parameters is not None:
			parameters = kwargs.parameters*pi/2
			identity = cls.identity(D=kwargs.D,dtype=kwargs.dtype)
			data = cos(parameters)*identity + -1j*sin(parameters)*data
		return data

	@classmethod
	@System.decorator	
	def Z(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([[1,0],[0,-1]],dtype=kwargs.dtype)
		if kwargs.parameters is not None:
			parameters = kwargs.parameters*pi/2
			identity = cls.identity(D=kwargs.D,dtype=kwargs.dtype)
			data = cos(parameters)*identity + -1j*sin(parameters)*data
		return data

	@classmethod
	@System.decorator
	def H(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([[1,1],[1,-1]],dtype=kwargs.dtype)
		return data

	@classmethod
	@System.decorator	
	def S(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([[1,0,],[0,1j]],dtype=kwargs.dtype)
		return data

	@classmethod
	@System.decorator
	def CNOT(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=kwargs.dtype)
		return data

	@classmethod
	@System.decorator	
	def unitary(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = rand(
			shape=getattr(kwargs,'shape',kwargs.D) if getattr(kwargs,'shape',None) is not None else kwargs.D,
			random=getattr(kwargs,'random','haar'),
			scale=getattr(kwargs,'scale',None),
			key=getattr(kwargs,'key',None) if getattr(kwargs,'key',None) is not None else  getattr(kwargs,'seed',None),
			dtype=kwargs.dtype)
		return data

	# Random
	@classmethod
	@System.decorator	
	def rand(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = rand(
			shape=kwargs.shape,
			random=getattr(kwargs,'random',None),
			scale=getattr(kwargs,'scale',None),
			key=getattr(kwargs,'key',None) if getattr(kwargs,'key',None) is not None else  getattr(kwargs,'seed',None),
			dtype=kwargs.dtype)
		return data	

	# State
	@classmethod
	@System.decorator	
	def state(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)		
		data = rand(
			shape=getattr(kwargs,'shape',kwargs.D) if getattr(kwargs,'shape',None) is not None else kwargs.D,
			random=getattr(kwargs,'random','haar'),
			scale=getattr(kwargs,'scale',None),
			key=getattr(kwargs,'key',None) if getattr(kwargs,'key',None) is not None else  getattr(kwargs,'seed',None),
			dtype=kwargs.dtype)
		if kwargs.ndim is not None and data.ndim < kwargs.ndim:
			data = einsum('...i,...j->...ij',data,conjugate(data))
		return data

	@classmethod
	@System.decorator	
	def zero(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([1,*[0]*(kwargs.D-1)],dtype=kwargs.dtype)
		if kwargs.ndim is not None and data.ndim < kwargs.ndim:
			data = einsum('...i,...j->...ij',data,conjugate(data))		
		return data

	@classmethod
	@System.decorator	
	def one(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([*[0]*(kwargs.D-1),1],dtype=kwargs.dtype)
		if kwargs.ndim is not None and data.ndim < kwargs.ndim:
			data = einsum('...i,...j->...ij',data,conjugate(data))		
		return data

	@classmethod
	@System.decorator	
	def plus(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/sqrt(kwargs.D))*array([1,1],dtype=kwargs.dtype)
		if kwargs.ndim is not None and data.ndim < kwargs.ndim:
			data = einsum('...i,...j->...ij',data,conjugate(data))		
		return data

	@classmethod
	@System.decorator	
	def minus(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/sqrt(kwargs.D))*array([1,-1],dtype=kwargs.dtype)
		if kwargs.ndim is not None and data.ndim < kwargs.ndim:
			data = einsum('...i,...j->...ij',data,conjugate(data))		
		return data

	@classmethod
	@System.decorator	
	def plusi(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/sqrt(kwargs.D))*array([1,1j],dtype=kwargs.dtype)
		if kwargs.ndim is not None and data.ndim < kwargs.ndim:
			data = einsum('...i,...j->...ij',data,conjugate(data))		
		return data

	@classmethod
	@System.decorator	
	def minusi(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/sqrt(kwargs.D))*array([1,-1j],dtype=kwargs.dtype)
		if kwargs.ndim is not None and data.ndim < kwargs.ndim:
			data = einsum('...i,...j->...ij',data,conjugate(data))		
		return data		

	@classmethod
	@System.decorator	
	def element(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = zeros(kwargs.shape,dtype=kwargs.dtype)
		index = tuple(map(int,kwargs.data))
		data[index] = 1
		if kwargs.ndim is not None and data.ndim < kwargs.ndim:
			data = einsum('...i,...j->...ij',data,conjugate(data))		
		return data

	# Probability
	@classmethod
	@System.decorator	
	def sample(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([*[1]*(kwargs.D)],dtype=kwargs.dtype)
		if kwargs.ndim is not None and data.ndim < kwargs.ndim:
			data = einsum('...i,...j->...ij',data,conjugate(data))
		return data

	# Operator
	@classmethod
	@System.decorator
	def projector(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = []
		size = kwargs.D
		for i in range(size):
			for j in range(size):
				obj = inplace(zeros((size,size),dtype=kwargs.dtype),(i,j),1)
				data.append(obj)
		return data

	# Noise
	@classmethod
	@System.decorator
	def dephase(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.parameters is None:
			kwargs.parameters = 0
		data = array([
			sqrt(1-kwargs.parameters)*cls.I(D=kwargs.D,dtype=kwargs.dtype),
			sqrt(kwargs.parameters)*cls.Z(D=kwargs.D,dtype=kwargs.dtype)
			],dtype=kwargs.dtype)
		return data

	@classmethod
	@System.decorator
	def bitflip(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.parameters is None:
			kwargs.parameters = 0
		data = array([
			sqrt(1-kwargs.parameters)*cls.I(D=kwargs.D,dtype=kwargs.dtype),
			sqrt(kwargs.parameters)*cls.X(D=kwargs.D,dtype=kwargs.dtype)
			],dtype=kwargs.dtype)
		return data

	@classmethod
	@System.decorator
	def phaseflip(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.parameters is None:
			kwargs.parameters = 0
		data = array([
			sqrt(1-kwargs.parameters)*cls.I(D=kwargs.D,dtype=kwargs.dtype),
			sqrt(kwargs.parameters)*cls.Y(D=kwargs.D,dtype=kwargs.dtype)
			],dtype=kwargs.dtype)
		return data

	@classmethod
	@System.decorator
	def depolarize(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.parameters is None:
			kwargs.parameters = 0		
		data = array([
				sqrt(1-(kwargs.D**2-1)*kwargs.parameters/(kwargs.D**2))*cls.I(D=kwargs.D,dtype=kwargs.dtype),
				sqrt(kwargs.parameters/(kwargs.D**2))*cls.X(D=kwargs.D,dtype=kwargs.dtype),
				sqrt(kwargs.parameters/(kwargs.D**2))*cls.Y(D=kwargs.D,dtype=kwargs.dtype),
				sqrt(kwargs.parameters/(kwargs.D**2))*cls.Z(D=kwargs.D,dtype=kwargs.dtype)
				],dtype=kwargs.dtype)
		return data

	@classmethod
	@System.decorator
	def amplitude(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.parameters is None:
			kwargs.parameters = 0		
		data = array([
			kwargs.element(D=kwargs.D,data='00',dtype=kwargs.dtype) + 
				sqrt(1-kwargs.parameters)*cls.element(D=kwargs.D,data='11',dtype=kwargs.dtype),
			sqrt(kwargs.parameters)*cls.element(D=kwargs.D,data='01',dtype=kwargs.dtype)
			],dtype=kwargs.dtype)
		return data

	# POVM
	@classmethod
	@System.decorator
	def pauli(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/(kwargs.D**2-1))*array([
			array([[1, 0],[0, 0]]),
			(1/kwargs.D)*array([[1,1],[1,1]]),
			(1/kwargs.D)*array([[1,-1j],[1j,1]]),
			(1/kwargs.D)*(
				2*array([[0,0],[0,1]]) + 
				array([[1,-1],[-1,1]]) + 
				array([[1,1j],[-1j,1]]))
			],dtype=kwargs.dtype)
		return data


	@classmethod
	@System.decorator
	def tetrad(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/(kwargs.D**2))*array([

				1*array([[1,0],[0,1]]) + 
				0*array([[0,1],[1,0]])+
				0*array([[0,-1j],[1j,0]])+
				1*array([[1,0],[0,-1]]),

				1*array([[1,0],[0,1]]) + 
				kwargs.D*sqrt(kwargs.D)/(kwargs.D**2-1)*array([[0,1],[1,0]])+
				0*array([[0,-1j],[1j,0]])+
				-1/(kwargs.D**2-1)*array([[1,0],[0,-1]]),

				1*array([[1,0],[0,1]]) + 
				-sqrt(kwargs.D)/(kwargs.D**2-1)*array([[0,1],[1,0]])+
				sqrt(kwargs.D/(kwargs.D**2-1))*array([[0,-1j],[1j,0]])+
				-1/(kwargs.D**2-1)*array([[1,0],[0,-1]]),

				1*array([[1,0],[0,1]]) + 
				-sqrt(kwargs.D)/(kwargs.D**2-1)*array([[0,1],[1,0]])+
				-sqrt(kwargs.D/(kwargs.D**2-1))*array([[0,-1j],[1j,0]])+
				-1/(kwargs.D**2-1)*array([[1,0],[0,-1]])
				],dtype=kwargs.dtype)

		return data

	@classmethod
	@System.decorator
	def trine(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/(kwargs.D**2-1))*array([
										cls.I(D=kwargs.D,dtype=kwargs.dtype) + 
			cos(i*2*pi/(kwargs.D**2-1))*cls.Z(D=kwargs.D,dtype=kwargs.dtype) + 
			sin(i*2*pi/(kwargs.D**2-1))*cls.X(D=kwargs.D,dtype=kwargs.dtype)
			for i in range(kwargs.D**2-1)
			],dtype=kwargs.dtype)
		raise ValueError('Not Informationally Complete POVM <%s>'%(sys._getframe().f_code.co_name))
		return data

	@classmethod
	@System.decorator
	def standard(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = zeros((kwargs.D**2,kwargs.D**2),dtype=kwargs.dtype)
		for i in range(kwargs.D**2):
			data = inplace(data,(i,i),1)
		data = data.reshape((kwargs.D**2,kwargs.D,kwargs.D))
		raise ValueError('Non-Normalized POVM <%s>'%(sys._getframe().f_code.co_name))		
		return data


class context(object):
	'''
	Update object attributes within context with key
	Args:
		key (object): Key to update attributes
		objs (iterable[object]): Objects with attributes to update
		formats (str,iterable[str],dict[str,dict]): Formats of attributes to update, {attr:[{attr_obj:format_attr_obj}]}
	'''
	def __init__(self,*objs,key=None,formats=None):

		if formats is None:
			formats = ['inds','tags']
		elif isinstance(formats,str):
			formats = [formats]
		if not isinstance(formats,dict):
			formats = {attr: [{index:index for index in self.attributes(obj,attr)} for obj in objs] 
				for attr in formats}
		else:
			formats = {attr: [{index:index for index in self.attributes(obj,attr)} for obj in objs] 
				if not isinstance(formats[attr],iterables) else 
				[{**{index:index for index in self.attributes(obj,attr)},**format} for obj,format in zip(objs,formats[attr])] 
				for attr in formats}
		
		attributes = [attr for attr in formats]
		formats = [{attr: formats[attr][i] for attr in formats} for i,obj in enumerate(objs)]
		
		def func(key,i,attr,objs,attrs,formats,*args,**kwargs):
			obj = objs[i]
			attrs = {attrs[i][attr][index]:formats[i][attr][index].format(key) if key is not None else formats[i][attr][index] for index in attrs[i][attr]}
			data = self.attributes(obj,attr,attrs=attrs,*args,**kwargs)
			return data
		
		def _func(key,i,attr,objs,attrs,formats,*args,**kwargs):
			obj = objs[i]
			attrs = {formats[i][attr][index].format(key) if key is not None else formats[i][attr][index]:attrs[i][attr][index] for index in attrs[i][attr]}
			data = self.attributes(obj,attr,attrs=attrs,*args,**kwargs)
			return data

		self.key = key
		self.objs = objs
		self.formats = formats
		self.attrs = [{attr: {index:index for index in self.attributes(obj,attr)} for attr in attributes} for obj in objs]
		self.funcs = [{attr: func for attr in attributes} for obj in objs]
		self._funcs = [{attr: _func for attr in attributes} for obj in objs]
		self.args = tuple()
		self.kwargs = dict(inplace=True)

		return

	def __enter__(self):
		for i in range(len(self)):
			for attr in self.funcs[i]:
				self.funcs[i][attr](self.key,i,attr,self.objs,self.attrs,self.formats,*self.args,**self.kwargs)
		return

	def __exit__(self, type, value, traceback):
		for i in range(len(self)):
			for attr in self._funcs[i]:
				self._funcs[i][attr](self.key,i,attr,self.objs,self.attrs,self.formats,*self.args,**self.kwargs)
		return
	
	def __len__(self):
		return len(self.objs)

	@classmethod
	def attributes(cls,obj,attr,attrs=None,**kwargs):
		if attrs is None:
			attributes = dict(inds='inds',tags='tags',sites='site_ind_id')
			wrapper = dict(inds=lambda obj:obj,tags=lambda obj:obj,sites=lambda obj:obj)
			wrappers = dict(inds=lambda obj:obj,tags=lambda obj:obj,sites=lambda obj:[obj])
			return wrappers[attr](getattr(obj,attributes[attr]))
		else:
			attributes = dict(inds='reindex',tags='retag',sites='reindex_sites')
			wrapper = dict(inds=lambda obj:obj,tags=lambda obj:obj,sites=lambda obj:obj[list(obj)[-1]] if obj else obj)
			wrappers = dict(inds=lambda obj:obj,tags=lambda obj:obj,sites=lambda obj:obj)
			return wrappers[attr](getattr(obj,attributes[attr])(wrapper[attr](attrs),**kwargs))	

class Measure(System):

	D = None

	basis = None

	ind = 'u{}'
	inds = ('u{}','v{}',)
	indices = ('i{}','j{}',)
	tag = 'I{}'
	tags = ()

	defaults = dict(			
		data=None,base=None,string=None,system=None,
		shape=None,size=None,ndim=None,dtype=None,
		basis=None,inverse=None,identity=None,
		parameters=None,variable=False,
		func=None,gradient=None,
		)

	def __init__(self,data=None,base=None,string=None,system=None,**kwargs):
		'''
		Measure Class
		Generate positive base valued measure basis and their overlap and inverse
		Args:
			data (str,array,tensor,Measure): data of measure
			base (str): name of measure
			string (str): string label of measure
			system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
			args (iterable): Additional class positional arguments
			kwargs (dict): Additional class keyword arguments
		'''

		setter(kwargs,dict(data=data,base=base,string=string,system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)	

		self.init()

		return

	def init(self,data=None,parameters=None,**kwargs):
		'''
		Initialize measure
		Args:
			data (str,array,tensor,Measure): data of measure
			parameters (array,dict): parameters of class
			kwargs (dict): Additional class keyword arguments			
		'''

		data = self.data if data is None else data
		parameters = self.parameters if parameters is None else parameters

		if parameters is None or not callable(parameters):
			def parameters(parameters=parameters):
				return parameters

		self.parameters = parameters

		self.setup(data)

		return

	def setup(self,data=None,base=None,string=None,**kwargs):
		'''
		Setup measure
		Args:
			data (str,array,tensor,Measure): data of measure
			base (str): name of measure
			string (str): string label of measure
			kwargs (dict): Additional class keyword arguments			
		'''

		data = self.data if data is None else data
		base = self.base if base is None else base
		string = self.string if string is None else string

		D = self.D
		base = data if base is None and isinstance(data,str) else base if data is None else base
		dtype = self.dtype

		basis = getattr(Basis,base)(D=D,dtype=dtype)
		data = einsum('u...,v...->uv',basis,conjugate(basis))
		inverse = inv(data)

		K = len(basis)
		shape = [min(i.shape[axis] for i in basis) for axis in range(min(len(i.shape) for i in basis))]
		size = prod(shape)
		ndim = len(shape)
		dtype = data.dtype

		if self.architecture is None or self.architecture in ['array','mps'] or self.architecture not in ['tensor']:
			kwargs = dict(dtype=dtype)

			basis = array(basis,**kwargs)
			data = array(data,**kwargs)
			inverse = array(inverse,**kwargs)

		elif self.architecture in ['tensor']:
			kwargs = dict(inds=(self.ind,*self.indices,),tags=(self.tag,*self.tags,))
			basis = tensor(basis,**kwargs)

			kwargs = dict(inds=(*self.inds,),tags=(self.tag,*self.tags,))
			data = tensor(data,**kwargs)
			
			kwargs = dict(inds=(*self.inds,),tags=(self.tag,*self.tags,))
			inverse = tensor(inverse,**kwargs)

		self.base = base
		self.string = string

		self.data = data
		self.basis = basis
		self.inverse = inverse

		self.shape = shape
		self.size = size
		self.ndim =  ndim
		self.dtype = dtype

		if self.architecture is None or self.architecture in ['array','mps'] or self.architecture not in ['tensor']:
			subscripts = '...u,uv,vjk->...jk'
			shapes = ((self.K,),self.inverse.shape,self.basis.shape)
			einsummation = einsum(subscripts,*shapes)
			def func(parameters,state):
				return einsummation(state,self.inverse,self.basis)

			def gradient(parameters,state):
				return 0

		elif self.architecture in ['tensor']:
			def func(parameters,state):
				for i in range(state.L):
					with context(self.inverse,self.basis,key=i,formats=dict(inds=[{index:index for index in self.inds},{self.ind:self.inds[-1]}])):
						state &= self.inverse & self.basis
				return state

			def gradient(parameters,state):
				return 0				

		self.func = func
		self.gradient = gradient

		parameters = self.parameters()
		wrapper = partial

		self.func = wrapper(self.func,parameters=parameters)
		self.gradient = wrapper(self.gradient,parameters=parameters)

		return

	def __call__(self,parameters=None,state=None,**kwargs):
		'''
		Call class for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,Probability,MPS): state of class of Probability state of shape (self.K,)*N
			kwargs (dict): Additional class keyword arguments					
		'''
		parameters = self.parameters() if parameters is None else parameters() if callable(parameters) else parameters
		
		return self.func(parameters=parameters,state=state)

	def __len__(self):
		return self.basis.shape[0]

	def __str__(self):
		if isinstance(self.string,str):
			string = self.string
		else:
			string = self.__class__.__name__
		return string

	def __repr__(self):
		return self.__str__()

	def get(self,attr):
		return getattr(self,attr)

	def set(self,attr,value):
		setattr(self,attr,value)
		return

	@property
	def K(self):
		return len(self)
	
	def info(self,display=None,ignore=None,verbose=None):
		'''
		Log class information
		Args:
			display (str,iterable[str]): Show attributes
			ignore (str,iterable[str]): Do not show attributes
			verbose (bool,int,str): Verbosity of message			
		'''		

		msg = []
		options = dict(align='<',space=1,width=2)

		display = None if display is None else [display] if isinstance(display,str) else display
		ignore = None if ignore is None else [ignore] if isinstance(ignore,str) else ignore

		for attr in [None,'string','base','K','ind','inds','tags','basis','data','inverse']:

			obj = attr
			if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
				continue

			if attr is None:
				attr = 'cls'
				substring = str(self)
			else:
				substring = getattrs(self,attr,delimiter=delim,default=None)

			if isinstance(substring,objects):
				string = '%s:\n%s'%(attr,str(substring))
			else:
				string = '%s: %s'%(attr,str(substring))

			msg.append(string)

		msg = [i if isinstance(i,str) else str(i) for i in msg]

		msg = '\n'.join(msg)

		self.log(msg,verbose=verbose)

		return


	def transform(self,parameters=None,state=None,model=None,transformation=None,**kwargs):
		'''
		Probability for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (str,iterable[str],array,Probability,MPS): state of class of shape (N,self.D,self.D) or (self.D**N,self.D**N)
			kwargs (dict): Additional class keyword arguments
			model (callable): model of operator with signature model(parameters,state) -> data
			transformation (bool,str): Type of transformation, True for amplitude -> probability or model to fun, or False for probability -> amplitude, allowed strings in ['probability','amplitude','operator','state','function','model'], default of amplitude -> probability
		Returns:
			state (array,Probability,MPS): state of class of Probability state of shape (N,self.K) or (self.K,)*N or (self.D**N,self.D**N)
			func (callable): operator with signature func(parameters,state,where,**kwargs) -> data (array) POVM operator of shape (self.K**N,self.K**N)
		'''

		if transformation in [None,True,'probability','state'] and model is None:
		
			return self.probability(parameters=parameters,state=state,**kwargs)

		elif transformation in [False,'amplitude','function']:

			return self.amplitude(parameters=parameters,state=state,**kwargs)

		elif transformation in [None,True,'operator','model'] and model is not None:
		
			state = self.transform(parameters=parameters,state=state,transformation=transformation)
		
			return self.operator(parameters=parameters,state=state,model=model,**kwargs)

		else:
		
			return state		


	def probability(self,parameters=None,state=None,**kwargs):
		'''
		Probability for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (str,iterable[str],array): state of class of shape (N,self.D,self.D)
			kwargs (dict): Additional class keyword arguments					
		Returns:
			state (array,Probability,MPS): state of class of Probability state of shape (N,self.K)
		'''
		
		parameters = self.parameters() if parameters is None else parameters() if callable(parameters) else parameters

		if not isinstance(state,objects):
			state = [getattr(Basis,i)(**kwargs) if isinstance(i,str) else i() if callable(i) else i for i in (state if not isinstance(state,str) else [state])]

		if self.architecture is None or self.architecture in ['array','mps'] or self.architecture not in ['tensor']:
			
			state = array(state,dtype=self.dtype)
			basis = self.basis
			inverse = self.inverse

			subscripts = 'uij,...ij->...u'
			shapes = (basis.shape,state.shape)
			einsummation = einsum(subscripts,*shapes)
			
			state = einsummation(basis,state)

		elif self.architecture in ['tensor']:
			
			if not isinstance(state,tensors):
				
				for i in range(len(state)):

					data = state[i]
					inds = (*self.indices,)
					tags = (self.tag,*self.tags,)

					data = tensor(data=data,inds=inds,tags=tags)

					with context(data,self.basis,key=i):
						data &= self.basis

					state[i] = representation(data,contract=True)

			else:

				state = state

			options = {**dict(site_ind_id=self.ind,site_tag_id=self.tag),**(self.options if self.options is not None else dict())}
			state = mps(state,**options)

		return state


	def amplitude(self,parameters=None,state=None,**kwargs):
		'''
		Amplitude for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,Probability,MPS): state of class of Probability of shape (N,self.K) or (self.K,)*N
			kwargs (dict): Additional class keyword arguments					
		Returns:
			state (array,Probability,MPS): state of class of Probability state of shape (self.D**N,self.D**N)
		'''
		
		parameters = self.parameters() if parameters is None else parameters() if callable(parameters) else parameters
		
		if self.architecture is None or self.architecture in ['array','mps'] or self.architecture not in ['tensor']:
			subscripts = '...u,uv,vij->...ij'
			shapes = (state.shape,self.inverse.shape,self.basis.shape)
			einsummation = einsum(subscripts,*shapes)

			state = einsummation(state,self.inverse,self.basis)

		elif self.architecture in ['tensor']:

			state = state.copy()

			N = state.L

			for i in range(N):
				with context(self.inverse,self.basis,key=i,formats=dict(inds=[{index:index for index in self.inds},{self.ind:self.inds[-1]}],tags=None)):
					state &= self.inverse & self.basis

			state = representation(state,to=self.architecture,contract=True)

		return state

	def operator(self,parameters=None,state=None,model=None,where=None,**kwargs):
		'''
		Operator for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,Probability,MPS): state of class of Probability state of shape (N,self.D,self.D)
			model (callable): model of operator with signature model(parameters,state) -> data
			where (int,str,iterable[int,str]): indices of contraction
			kwargs (dict): Additional class keyword arguments					
		Returns:
			func (callable): operator with signature func(parameters,state,where,**kwargs) -> data (array) POVM operator of shape (self.K**N,self.K**N)
		'''

		if self.architecture is None or self.architecture in ['array','mps'] or self.architecture not in ['tensor']:

			N = len(state) if state is not None else None

			if N is not None and N > 1:
				basis = array([tensorprod(i) for i in permutations(*[self.basis]*N)],dtype=self.dtype)
				inverse = array([tensorprod(i) for i in permutations(*[self.inverse]*N)],dtype=self.dtype)
			else:
				basis = self.basis
				inverse = self.inverse

			if model is None:
			
				def func(parameters,state,where=where,**kwargs):
					return None
			
			else:
				subscripts = 'uij,wij,wv,v...->u...'
				shapes = (basis.shape,basis.shape,inverse.shape,inverse[-1:])
				einsummation = einsum(subscripts,*shapes)
				
				model = vmap(model,in_axes=(None,0),out_axes=0)
				
				options = dict(axes=[where],shape=[self.D,N,self.ndim],execute=False)
				swapper = swap(state,**options,transform=True)
				_swapper = swap(state,**options,transform=False)
					
				def func(parameters,state,where=where,model=model,basis=basis,inverse=inverse,einsummation=einsummation,**kwargs):
					return _swapper(einsummation(conjugate(basis),model(parameters,basis),inverse,swapper(state)))

		elif self.architecture in ['tensor']:

			N = state.L

			if N is not None and N > 1:
				basis = array([tensorprod(i) for i in permutations(*[representation(self.basis)]*N)],dtype=self.dtype)
				inverse = array([tensorprod(i) for i in permutations(*[representation(self.inverse)]*N)],dtype=self.dtype)
			else:
				basis = representation(self.basis)
				inverse = representation(self.inverse)


			if model is None:
			
				def func(parameters,state,where=where,**kwargs):
					return None
			else:
				subscripts = 'uij,wij,wv->uv'
				shapes = (basis.shape,basis.shape,inverse.shape,inverse[-1:])
				einsummation = einsum(subscripts,*shapes)
				
				model = vmap(model,in_axes=(None,0),out_axes=0)
				
				def func(parameters,state,where=where,model=model,basis=basis,inverse=inverse,einsummation=einsummation,**kwargs):
					return state.gate(einsummation(conjugate(basis),model(parameters,basis),inverse),where=where,**kwargs)
		
		parameters = self.parameters() if parameters is None else parameters
		wrapper = partial

		func = wrapper(func,parameters=parameters)

		return func

	def calculate(self,attr=None,parameters=None,state=None,**kwargs):
		'''
		Calculate data for POVM probability measure
		Args:
			attr (str): attribute for calculation of data
			parameters (array): parameters of class
			state (array,Probability,MPS): state of class of Probability of shape (N,self.K) or (self.K,)*N
			kwargs (dict): Additional class keyword arguments					
		Returns:
			data (array): data
		'''

		if hasattr(self,attr):
			data = getattr(self,attr)(parameters=parameters,state=state,**kwargs)
		else:
			data = state

		return data

	def infidelity(self,parameters=None,state=None,other=None,**kwargs):
		'''
		Infidelity for POVM probability measure with respect to other POVM
		Args:
			parameters (array): parameters of class
			state (array,Probability,MPS): state of class of Probability of shape (N,self.K) or (self.K,)*N
			other (array,Probability,MPS): state of class of Probability of shape (N,self.K) or (self.K,)*N
			kwargs (dict): Additional class keyword arguments					
		Returns:
			data (array): data
		'''

		attr = 'infidelity_classical'
		
		data = getattr(self,attr)(parameters=parameters,state=state,other=other,**kwargs)

		return data

	def infidelity_quantum(self,parameters=None,state=None,other=None,**kwargs):
		'''
		Infidelity (Quantum) for POVM probability measure with respect to other POVM
		Args:
			parameters (array): parameters of class
			state (array,Probability,MPS): state of class of Probability of shape (N,self.K) or (self.K,)*N
			other (array,Probability,MPS): state of class of Probability of shape (N,self.K) or (self.K,)*N
			kwargs (dict): Additional class keyword arguments					
		Returns:
			data (array): data
		'''
		
		parameters = self.parameters() if parameters is None else parameters() if callable(parameters) else parameters
		state = self.state() if state is None else state() if callable(state) else state
		other = self.state() if other is None else other() if callable(other) else other
		
		if self.architecture is None or self.architecture in ['array','mps'] or self.architecture not in ['tensor']:
			
			N = len(state) if state is not None else None

			if N is not None and N > 1:
				inverse = array([tensorprod(i) for i in permutations(*[self.inverse]*N)],dtype=self.dtype)
			else:
				inverse = self.inverse

			subscripts = '...u,uv,...v->...'
			shapes = (state.shape,inverse.shape,other.shape)
			einsummation = einsum(subscripts,*shapes)
			
			data = einsummation(state,inverse,other)

		elif self.architecture in ['tensor']:
	
			state = state.copy()

			N = state.L

			for i in range(N):
				with context(self.inverse,key=i):
					state &= self.inverse

			with context(state,other,formats=dict(sites=[{self.inds[-1]:self.inds[-1]},{self.ind:self.inds[-1]}])):

				state &= other

				data = representation(state,contract=True,func=real)

		data = 1 - data

		return data		

	def infidelity_classical(self,parameters=None,state=None,other=None,**kwargs):
		'''
		Infidelity (Classical) for POVM probability measure with respect to other POVM
		Args:
			parameters (array): parameters of class
			state (array,Probability,MPS): state of class of Probability of shape (N,self.K) or (self.K,)*N
			other (array,Probability,MPS): state of class of Probability of shape (N,self.K) or (self.K,)*N
			kwargs (dict): Additional class keyword arguments					
		Returns:
			data (array): data
		'''
		
		parameters = self.parameters() if parameters is None else parameters() if callable(parameters) else parameters
		state = self.state() if state is None else state() if callable(state) else state
		other = self.state() if other is None else other() if callable(other) else other
		
		if self.architecture is None or self.architecture in ['array','mps'] or self.architecture not in ['tensor']:
			
			subscripts = '...u,...u->...'
			shapes = (state.shape,other.shape)
			einsummation = einsum(subscripts,*shapes)
			
			state = sqrt(state)
			other = sqrt(other)

			data = einsummation(state,other)

		elif self.architecture in ['tensor']:
	
			state = state.contract()
			state.modify(apply=sqrt)

			other = other.contract()
			other.modify(apply=sqrt)

			data = representation(state & other,contract=True,func=real)

		data = 1 - data

		return data

	def infidelity_pure(self,parameters=None,state=None,other=None,**kwargs):
		'''
		Infidelity (Pure State) for POVM probability measure with respect to other POVM
		Args:
			parameters (array): parameters of class
			state (array,Probability,MPS): state of class of Probability of shape (N,self.K) or (self.K,)*N
			other (array,Probability,MPS): state of class of Probability of shape (N,self.K) or (self.K,)*N
			kwargs (dict): Additional class keyword arguments					
		Returns:
			data (array): data
		'''
		
		parameters = self.parameters() if parameters is None else parameters() if callable(parameters) else parameters
		state = self.state() if state is None else state() if callable(state) else state
		other = self.state() if other is None else other() if callable(other) else other
		
		if self.architecture is None or self.architecture in ['array','mps'] or self.architecture not in ['tensor']:
			
			N = len(state) if state is not None else None

			if N is not None and N > 1:
				inverse = array([tensorprod(i) for i in permutations(*[self.inverse]*N)],dtype=self.dtype)
			else:
				inverse = self.inverse

			subscripts = '...u,uv,...v->...'
			shapes = (state.shape,inverse.shape,other.shape)
			einsummation = einsum(subscripts,*shapes)
			
			data = einsummation(state,inverse,other)

		elif self.architecture in ['tensor']:
	
			state = state.copy()

			N = state.L

			for i in range(N):
				with context(self.inverse,key=i):
					state &= self.inverse

			with context(state,other,formats=dict(sites=[{self.inds[-1]:self.inds[-1]},{self.ind:self.inds[-1]}])):

				state &= other

				data = representation(state,contract=True,func=real)

		return data




class MPS(mps): 
	'''
	Matrix Product State class
	Args:
		data (iterable,int,str,callable,array,object): Tensor data
		N (int): Tensor system size
		D (int): Tensor physical bon dimension
		S (int): Tensor virtual bond dimension
		kwargs (dict): Tensor keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,data,N=None,D=None,S=None,**kwargs):

		updates = {
			'periodic':(
				(lambda attr,value,kwargs:'cyclic'),
				(lambda attr,value,kwargs: (value is True) and N is not None and N>2)
				),
			'boundaries':(
				(lambda attr,value,kwargs:'cyclic'),
				(lambda attr,value,kwargs: ((value in ['periodic']) or (value is True)) and N is not None and N>2)
				)			
			}

		kwargs.update(dict(data=data,L=N))
		for attr in updates:
			if attr not in kwargs:
				continue
			attrs,values = updates[attr]
			attr,value = attr,kwargs.pop(attr)
			attr,value = attrs(attr,value,kwargs),values(attr,value,kwargs)
			if value is None:
				continue
			kwargs[attr] = value

		if data is None:
			kwds = {attr: kwargs.get(attr) for attr in ['random','seed','bounds','scale','dtype']}
			def data(shape,*args,**kwargs):
				return rand(shape=shape,**kwds) 
			kwargs.update(dict(phys_dim=D,bond_dim=S))
			kwargs = {attr: kwargs.get(attr) for attr in ['L','phys_dim','bond_dim','cyclic'] if attr in kwargs}
		elif isinstance(data,(str,*iterables)):
			basis = {
				**{attr: Basis.state for attr in ['psi','state','random','haar','product']},
				**{attr: Basis.zero for attr in ['zero','zeros','0']},
				**{attr: Basis.one for attr in ['one','ones','1']},
				**{attr: Basis.plus for attr in ['plus','+']},
				**{attr: Basis.minus for attr in ['minus','-']},
				**{attr: Basis.plusi for attr in ['plusi','+i']},
				**{attr: Basis.minusi for attr in ['minusi','-i']},	
			}
			data = [data]*N if isinstance(data,str) else [i for i in data]
			data = [basis[i](D=D,**kwargs) if isinstance(i,str) else i for i in data]
			kwargs.update(dict())
			kwargs = {attr: kwargs.get(attr) for attr in ['L','cyclic'] if attr in kwargs}
		elif isinstance(data,integers):
			kwargs.update(dict(phys_dim=D,bond_dim=S))			
			kwargs = {attr: kwargs.get(attr) for attr in ['L','cyclic','dtype'] if attr in kwargs}
		else:
			kwargs.update(dict(phys_dim=D,bond_dim=S))			
			kwargs = {attr: kwargs.get(attr) for attr in ['L','cyclic','dtype'] if attr in kwargs}

		kwargs.update(dict(data=data))

		self = super().__new__(cls,**kwargs)

		return self


	def __len__(self):
		return self.L

	def __call__(self,parameters,state=None,data=None,site=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
			state (obj): state
			data (obj): data for operator
			site (int,str,iterable[int,str]): Where data contracts with state
		Returns:
			data (array): data
		'''
		state = self if state is None else state
		if data is None:
			return state
		else:
			return self.gate(data,where=site)

def trotter(iterable=None,p=None,verbose=False):
	'''
	Trotterized iterable for order p or parameters for order p
	Args:
		iterable (iterable): Iterable
		p (int): Order of trotterization
		verbose (bool,int,str): Verbosity of function		
	Returns:
		iterables (iterable,scalar): Trotterized iterable for order p or parameters for order p if iterable is None
	'''
	P = 2
	if p is None and iterable is None:
		return p
	elif p is None:
		return iterable
	elif not isinstance(p,integers) or (p > P):
		raise NotImplementedError('p = %r !< %d Not Implemented'%(p,P))

	if iterable is None:
		options = {i:1/p for i in range(P+1)}
		iterables = options[p]
	else:
		options = {i:[slice(None,None,(-1)**i)] for i in range(P)}
		iterables = []        
		for i in range(p):
			for indices in options[i]:
				iterables += iterable[indices]

	return iterables


def compile(data,state=None,conj=False,size=None,compilation=None,verbose=False):
	'''
	Compile data and state
	Args:
		data (iterable[Object],dict[int,Object]): data of shape (length,)
		state (array,Object): state of shape (n,) or (n,n)
		conj (bool): conjugate
		size (int): Number of contractions of data and state
		compilation (dict[str,int,bool]): Compilation options, allowed keys 
			trotter (int): Number of cycles of each contraction of data and state
			simplify (bool): Simplification of data to most basic form
		verbose (bool,int,str): Verbosity of function
	Returns:
		data (iterable[Object]): Compiled data of shape (length,)
	'''

	data = {j: data[i] if isinstance(data,dict) else i for j,i in enumerate(data)} if data is not None else {}
	state = state() if callable(state) else state
	conj = conj if conj is None else False	
	size = size if size is not None else 1
	compilation = Dict({**dict(trotter=None,simplify=False),**(compilation if isinstance(compilation,dict) else {})})

	# Update data
	for i in data:
		if data[i] is None:
			continue
		kwargs = dict(
			parameters=dict(parameters=dict(trotter=trotter(p=compilation.trotter)) if (data[i].unitary) else None),
			)
		data[i].init(**kwargs)

	# Filter None data
	boolean = lambda i: (data[i] is not None) and (data[i].data is not None)

	data = {i: data[i] for i in data if boolean(i)}

	# Filter constant data
	if compilation.simplify:

		boolean = lambda i: (data[i] is not None) and (data[i].data is not None) and (not data[i].variable) and (data[i].unitary)
		
		obj = {i: data[i] for i in data if boolean(i)}

		if len(obj)>1:

			for obj in splitter(obj):
				if len(obj) < 2:
					continue
				
				j = min(obj)
				
				for i in obj:
					if i == j:
						continue
					data[j] = data[j] @ data.pop(i)

		data = {j: data[i] if isinstance(data,dict) else i for j,i in enumerate(data)} if data is not None else {}

	# Filter trotterized data
	boolean = lambda i: (data[i] is not None) and (data[i].data is not None) and (data[i].unitary)

	obj = [j
		for i in interleaver(
		[trotter(i,p=compilation.trotter) for i in splitter([i for i in data if boolean(i)])],
		[i for i in splitter([i for i in data if not boolean(i)])]) 
		for j in i]

	data = [data[i] for i in obj]

	return data

def variables(data,state=None,conj=False,size=None,compilation=None,verbose=False):
	'''
	Get indexes of variable parameters of data and state, 
	with positive integers representing parameter indexes for that data element, 
	and negative integers representing no parameter indexes for that data element
	Args:
		data (iterable[Object],dict[int,Object]): data of shape (length,)
		state (array,Object): state of shape (n,) or (n,n)
		conj (bool): conjugate
		size (int): Number of contractions of data and state
		compilation (dict[str,int,bool]): Compilation options, allowed keys 
			trotter (int): Number of cycles of each contraction of data and state
			simplify (bool): Simplification of data to most basic form
		verbose (bool,int,str): Verbosity of function
	Returns:
		indexes (iterable[int]): Indices of variable parameters data and state of shape (length,)
		shape (iterable[int]): Shape of variable parameters
	'''

	data = {j: data[i] if isinstance(data,dict) else i for j,i in enumerate(data)} if data is not None else {}
	state = state() if callable(state) else state
	conj = conj if conj is None else False	
	size = size if size is not None else 1
	compilation = compilation if compilation is not None else None

	boolean = lambda i: (data[i] is not None) and (data[i].variable) and (data[i].parameters is not None) and (data[i].parameters.indices is not None)
	default = -1

	length = len(set(([data[i].parameters.indices for i in data if boolean(i)])))

	shape = (size*length,)

	indexes = [j*length + data[i].parameters.indices if boolean(i) else default for j in range(size) for i in data]

	return indexes,shape




def scheme(data,parameters=None,state=None,conj=False,size=None,compilation=None,architecture=None,verbose=False):
	'''
	Contract data and state
	Args:
		data (iterable[Object],dict[int,Object]): data of shape (length,)
		parameters (array,Parameters): parameters of data
		state (array,Object): state of shape (n,) or (n,n)
		conj (bool): conjugate
		size (int): Number of contractions of data and state
		compilation (dict[str,int,bool]): Compilation options, allowed keys 
			trotter (int): Number of cycles of each contraction of data and state
			simplify (bool): Simplification of data to most basic form
		architecture (str): Architecture data structure of data
		verbose (bool,int,str): Verbosity of function		
	Returns:
		func (callable): contract data with signature func(parameters,state,indices)
	'''

	state = state() if callable(state) else state
	parameters = parameters() if callable(parameters) else parameters
	conj = conj if conj is None else False	
	size = size if size is not None else 1
	compilation = compilation if compilation is not None else None

	length = len(data) if data is not None else 1
	indices = (0,size*length)

	dimension = min(data[i].D for i in data if data[i] is not None) if data else None
	locality = len(set(j for i in data if data[i] is not None for j in data[i].site)) if data else None
	dtype = data[0].dtype if data else None
	obj = state if state is not None else tensorprod([Basis.identity(D=dimension,dtype=dtype)]*locality) if data else None

	if parameters is not None and len(parameters):
		def function(parameters,state=state,indices=indices):	
			return switch(indices%length,data,parameters[indices//length],state)
	else:
		def function(parameters,state=state,indices=indices):	
			return switch(indices%length,data,parameters,state)

	data = compile(data,state=state,conj=conj,size=size,compilation=compilation,verbose=verbose)

	length = len(data)
	obj = state if state is not None else tensorprod([Basis.identity(D=dimension,dtype=dtype)]*locality) if data else None
	
	if architecture is None:
		wrapper = jit
	elif architecture in ['array']:		
		wrapper = jit
	elif architecture in ['tensor']:		
		wrapper = jit
	elif architecture in ['mps']:
		wrapper = partial
	else:
		wrapper = jit

	data = [wrapper(data[i]) for i in range(length)] # TODO: Time/M-dependent constant data/parameters

	def func(parameters,state=state,indices=indices):

		def func(i,out):
			return function(parameters,out,indices=i)

		state = obj if state is None else state
		return forloop(*indices,func,state)

	func = wrapper(func)

	return func			




def gradient_scheme(data,parameters=None,state=None,conj=False,size=None,compilation=None,architecture=None,verbose=False):
	'''
	Contract gradient of data and state
	Args:
		data (iterable[Object],dict[int,Object]): data of shape (length,)
		parameters (array,Parameters): parameters of data
		state (array,Object): state of shape (n,) or (n,n)
		conj (bool): conjugate
		size (int): Number of contractions of data and state
		compilation (dict[str,int,bool]): Compilation options, allowed keys 
			trotter (int): Number of cycles of each contraction of data and state
			simplify (bool): Simplification of data to most basic form
		architecture (str): Architecture data structure of data
		verbose (bool,int,str): Verbosity of function		
	Returns:
		func (callable): contract gradient with signature func(parameters,state,indices)
	'''

	state = state() if callable(state) else state
	parameters = parameters() if callable(parameters) else parameters
	conj = conj if conj is None else False
	size = size if size is not None else 1
	compilation = compilation if compilation is not None else None
	
	length = len(data)
	indices = (0,size*length)
	
	dimension = min(data[i].D for i in data if data[i] is not None) if data else None
	locality = len(set(j for i in data if data[i] is not None for j in data[i].site)) if data else None
	dtype = data[0].dtype if data else None
	obj = state if state is not None else tensorprod([Basis.identity(D=dimension,dtype=dtype)]*locality) if data else None

	function = scheme(data,parameters=parameters,state=state,conj=conj,size=size,compilation=compilation,architecture=architecture)	

	if parameters is not None and len(parameters):
		def gradient(parameters,state=state,indices=indices):	
			return switch(indices%length,grad,parameters[indices//length],state)
	else:
		def gradient(parameters,state=state,indices=indices):	
			return switch(indices%length,grad,parameters,state)

	data = compile(data,state=state,conj=conj,size=size,compilation=compilation)	

	indexes,shape = variables(data,state=state,conj=conj,size=size,compilation=compilation)	

	length = len(data)
	indices = (0,size*length)
	obj = state if state is not None else tensorprod([Basis.identity(D=dimension,dtype=dtype)]*locality) if data else None
	if architecture is None:
		wrapper = jit
	elif architecture in ['array']:
		wrapper = jit
	elif architecture in ['tensor']:
		wrapper = jit
	elif architecture in ['mps']:
		wrapper = partial
	else:
		wrapper = jit

	data,grad,indexes = (
		[wrapper(data[i]) for i in range(length)],
		[wrapper(data[i].grad) for i in range(length)],
		array(indexes)
		)

	def true(i,out,parameters,state):

		obj = function(parameters,state,indices=(0,i))

		obj = gradient(parameters,obj,indices=i)

		obj = function(parameters,obj,indices=(i+1,size*length))

		out = inplace(out,indexes[i],obj,'add')

		return out

	def false(i,out,parameters,state):
		return out

	true = wrapper(true)
	false = wrapper(false)

	def func(parameters,state=state,indices=indices):

		def func(i,out):
			return cond(indexes[i]>=0,true,false,i,out,parameters,state)

		state = obj if state is None else state
		
		out = zeros((*shape,*state.shape),dtype=state.dtype)

		return forloop(*indices,func,out)

	func = wrapper(func)

	return func			


class Object(System):
	'''
	Base class for Quantum Objects
	Args:
		data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) N-length delimiter-separated string of operators 'X_Y_Z' or N-length iterable of operator strings['X','Y','Z']		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	N = None
	D = None

	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}}

	defaults = dict(			
		data=None,operator=None,site=None,string=None,system=None,
		state=None,conj=False,parameters=None,variable=None,
		locality=None,local=None,
		shape=None,size=None,ndim=None,dtype=None,
		samples=None,identity=None,base=None,
		space=None,time=None,lattice=None,
		func=None,gradient=None,
		contract=None,gradient_contract=None
		)

	def __init__(self,data=None,operator=None,site=None,string=None,system=None,**kwargs):		

		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)	

		if isinstance(data,self.__class__) or isinstance(operator,self.__class__):
			if isinstance(data,self.__class__):
				for attr in data:
					if attr not in kwargs:
						setattr(self,attr,getattr(data,attr))
			if isinstance(operator,self.__class__):
				for attr in operator:
					if attr not in kwargs:
						setattr(self,attr,getattr(operator,attr))					

			return

		basis = self.basis
		default = self.default
		
		operator = self.operator if self.operator is not None else None
		site = self.site if self.site is None or not isinstance(self.site,integers) else [self.site]
		locality = self.locality
		string = self.string
		local = self.local if self.local is not None else None


		N = self.N	
		D = self.D	


		# Set local, locality, N
		local = local
		
		if locality is not None:
			locality = locality
		elif isinstance(operator,iterables):
			locality = len(operator)
		elif isinstance(site,iterables):
			locality = len(site)
		elif isinstance(operator,str) and operator.count(delim) > 0:
			locality = operator.count(delim) + 1
		elif N is not None and not local:
			locality = N
		else:
			locality = 1

		N = locality if local and N is None else N


		# Set site,locality, operator
		if site is None:
			if operator is None:
				site = site
				locality = locality
				operator = operator
			elif isinstance(operator,str):
				if operator in [default]:
					site = [i for i in range(locality)]
					locality = locality
					operator = operator							
				elif operator.count(delim):
					site = [i for i in range(locality)]
					locality = locality
					operator = [i for i in operator.split(delim)]
				else:
					site = [i for i in range(locality)]
					locality = locality
					operator = operator
			elif not isinstance(operator,str) and not isinstance(operator,objects) and not callable(operator):
				site = [i for i in range(locality)]
				locality = locality
				operator = [i for i in operator]
			else:
				site = [i for i in range(locality)]
				locality = locality
				operator = operator
		else:
			if operator is None:
				site = [i for i in site]
				locality = locality
				operator = operator
			elif isinstance(operator,str):
				if operator in [default]:
					site = [i for i in site]
					locality = locality
					operator = operator							
				elif operator.count(delim):
					site = [i for i in site]
					locality = locality
					operator = [i for i in operator.split(delim)]
				else:
					site = [i for i in site]
					locality = locality
					operator = [operator for i in site]
			elif not isinstance(operator,str) and not isinstance(operator,objects) and not callable(operator):
				site = [i for i in site]
				locality = locality
				operator = [i for i in operator]
			else:
				site = [i for i in site]
				locality = locality
				operator = operator

		if not isinstance(string,str):
			string = str(string)

		N = max(N if N is not None else 0,locality if locality is not None else 0) if N is not None or locality is not None else None
		D = self.D if self.D is not None else getattr(data,'size',1)**(1/max(1,getattr(data,'ndim',1)*N)) if isinstance(data,objects) else 1

		site = site[:locality] if site is not None else site
		locality = min(locality if locality is not None else 0,sum(i not in [default] for i in site) if site is not None else 0,locality if local else N) if locality is not None or site is not None else None
		operator = operator[:locality] if operator is not None and not isinstance(operator,str) and not isinstance(operator,objects) and not callable(operator) else operator
		local = local

		shape = self.shape if self.shape is not None else getattr(data,'shape',self.shape) if data is not None else None
		size = self.size if self.size is not None else getattr(data,'size',self.size) if data is not None else None
		ndim = self.ndim if self.ndim is not None else getattr(data,'ndim',self.ndim) if data is not None else None
		dtype = self.dtype if self.dtype is not None else getattr(data,'dtype',self.dtype) if data is not None else None

		self.data = data if data is not None else None
		self.operator = operator if operator is not None else None
		self.site = site if site is not None else None
		self.string = string if string is not None else None
		self.system = system if system is not None else None

		self.locality = max(locality,len(self.site) if self.site is not None else 0) if locality is not None else None
		self.local = local

		self.N = N
		self.D = D

		self.shape = shape
		self.size = size
		self.ndim = ndim
		self.dtype = dtype

		self.init(data=self.data,state=self.state,conj=self.conj,parameters=self.parameters)

		self.info()

		return

	def init(self,data=None,state=None,conj=False,parameters=None,**kwargs):
		'''
		Initialize operator
		Args:
			data (bool,dict): data of class, or boolean to retain current data attribute or initialize as None
			state (bool,dict,array,Object): state of class
			conj (bool): conjugate
			parameters (array,dict): parameters of class
			kwargs (dict): Additional class keyword arguments			
		'''

		if data is None:
			data = self.data
		elif data is True:
			data = self.data			
		elif data is False:
			data = None

		if state is None:
			state = self.state
		elif state is True:
			state = self.state
		elif state is False:
			state = None

		if conj is None:
			conj = False

		self.data = data
		self.state = state
		self.conj = conj

		self.spaces()

		if parameters is None:
			parameters = None
		elif parameters is True:
			parameters = None
		elif parameters is False:
			parameters = None

		cls = Parameter
		if not isinstance(self.parameters,cls) and not isinstance(parameters,cls):
			defaults = dict()
			parameters = parameters if parameters is not None else self.parameters
			parameters = dict(data=parameters) if not isinstance(parameters,dict) else parameters
			setter(parameters,{attr: getattr(self,attr) for attr in self if attr not in cls.defaults and attr not in dict(data=None)},delimiter=delim,default=False)
			setter(parameters,dict(string=self.string,variable=self.variable,system=self.system),delimiter=delim,default=True)
			setter(parameters,defaults,delimiter=delim,default=False)
			setter(parameters,self.parameters,delimiter=delim,default=False)
			self.parameters = cls(**parameters)

		elif isinstance(self.parameters,cls):
			parameters = parameters if isinstance(parameters,dict) else dict(data=parameters) if parameters is not None else dict()
			self.parameters.init(**parameters)

		else:
			self.parameters = parameters


		if state is None or not callable(state):
			def state(parameters=None,state=state):
				return state

		self.state = state


		if self.local:
			identity = tensorprod([Basis.identity(D=self.D,system=self.system)]*self.locality)
		else:
			identity = tensorprod([Basis.identity(D=self.D,system=self.system)]*self.N)

		self.identity = identity


		if (((self.data is not None) or (self.operator is not None))):
			
			self.setup(self.data,self.operator,self.site,self.string)
		
		if (self.parameters() is None) and (not isinstance(self.data,objects)) and (not callable(self.data)):
			data = None
		elif isinstance(self.data,objects) or callable(self.data):
			data = self.data
		elif isinstance(self.operator,objects) or callable(self.operator):
			self.operator = self.string
			data = self.operator
		elif self.operator is None:
			data = None
		elif isinstance(self.operator,str):
			if self.local:
				data = tensorprod([
					self.basis.get(self.operator if i in self.site else self.default)(D=self.D,system=self.system)
				 if i in self.site else self.basis.get(self.default)(D=self.D,system=self.system) 
				 for i in self.site]) if self.operator in self.basis else None
			else:
				data = tensorprod([
					self.basis.get(self.operator if i in self.site else self.default)(D=self.D,system=self.system)
				 if i in self.site else self.basis.get(self.default)(D=self.D,system=self.system) 
				 for i in range(self.N)]) if self.operator in self.basis else None
		elif isinstance(self.operator,iterables):
			if self.local:
				data = tensorprod([
					self.basis.get((
					self.operator[self.site.index(i)]
					if isinstance(self.operator[self.site.index(i)],str) else 
					self.operator[self.site.index(i)]() 
					if callable(self.operator[self.site.index(i)]) else
					self.operator[self.site.index(i)])
					if i in self.site else self.default)(D=self.D,system=self.system)
				 for i in self.site]) if all(i in self.basis for i in self.operator) else None
			else:
				data = tensorprod([
					self.basis.get((
					self.operator[self.site.index(i)]
					if isinstance(self.operator[self.site.index(i)],str) else 
					self.operator[self.site.index(i)]() 
					if callable(self.operator[self.site.index(i)]) else
					self.operator[self.site.index(i)])
					if i in self.site else self.default)(D=self.D,system=self.system)
				 for i in range(self.N)]) if all(i in self.basis for i in self.operator) else None
		else:
			data = self.data

		self.data = data

		self.N = min(self.N if self.N is not None else 0,self.locality if self.locality is not None else 0) if self.local and (self.N is not None or self.locality is not None) else self.N if self.N is not None else None
		self.D = self.D if self.D is not None else None

		self.shape = getattr(data,'shape',self.shape) if data is not None else self.shape if self.shape is not None else None
		self.size = getattr(data,'size',self.size) if data is not None else self.size if self.size is not None else None
		self.ndim = getattr(data,'ndim',self.ndim) if data is not None else self.ndim if self.ndim is not None else None
		self.dtype = getattr(data,'dtype',self.dtype) if data is not None else self.dtype if self.dtype is not None else None


		if self.func is None:
			def func(parameters=None,state=None):
				return self.data
		else:
			func = self.func

		if self.gradient is None:
			def gradient(parameters=None,state=None):
				return 0*self.data
		else:
			gradient = self.gradient

		data = self.data
		state = self.state() if callable(self.state) else self.state
		where = self.site if self.local else None

		if self.state is not None and self.state() is not None:
			try: 
				kwargs = dict(shape=(self.state.D,self.state.N,self.state.ndim),axes=(self.site,))
			except:
				kwargs = dict(shape=(self.D,int(round(log(state.size)/log(self.D)/state.ndim)),state.ndim),axes=(self.site,))
		else:
			kwargs = dict(**{**dict(shape=(self.D,self.N,self.ndim),axes=(self.site,)),**(self.options if self.options is not None else {})})

		if self.architecture is None or self.architecture in ['array','tensor'] or self.architecture not in ['mps']:
			kwargs = dict(**{**kwargs,**{attr: self.options[attr] for attr in self.options if attr not in kwargs}}) if self.options is not None else kwargs
		elif self.architecture in ['mps']:
			kwargs = dict(**{**self.options}) if self.options is not None else kwargs

		try:
			contract = contraction(data,state,where=where,**kwargs) if self.contract is None else self.contract
		except NotImplementedError as exception:
			def contract(data,state,where=where,**kwargs):
				return state
			raise exception

		try:
			grad_contract = gradient_contraction(data,state,where=where,**kwargs) if self.gradient_contract is None else self.gradient_contract
		except NotImplementedError as exception:
			def grad_contract(grad,data,state,where=where,**kwargs):
				return 0

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = grad_contract

		if self.architecture is None or self.architecture in ['array','tensor'] or self.architecture not in ['mps']:
			parameters = self.parameters()
			state = self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			where = self.site
			wrapper = jit
		
		elif self.architecture in ['mps']:
			parameters = self.parameters()
			state = self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			where = self.site			
			wrapper = partial
		
		self.func = wrapper(self.func,parameters=parameters,state=state)
		self.gradient = wrapper(self.gradient,parameters=parameters,state=state)
		self.contract = wrapper(self.contract,state=state,where=where)
		self.gradient_contract = wrapper(self.gradient_contract,state=state,where=where)
 
		return

	def setup(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments			
		'''

		return

	# @property
	# def shape(self):
	# 	return self.data.shape if isinstance(self.data,arrays) else (self.n,)*self.ndim if self.ndim is not None else ()

	# @property
	# def size(self):
	# 	return self.data.size if isinstance(self.data,arrays) else prod(self.shape)

	# @property
	# def ndim(self):
	# 	return self.data.ndim if isinstance(self.data,arrays) else len(self.shape)

	# @property
	# def dtype(self):
	# 	return self.data.dtype if isinstance(self.data,arrays) else self._dtype if self._dtype is not None else None

	# @dtype.setter
	# def dtype(self,dtype):
	# 	self._dtype = dtype
	# 	return


	def __call__(self,parameters=None,state=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			data (array): data
		'''
		return self.contract(self.func(parameters=parameters,state=state),state=state)

	def grad(self,parameters=None,state=None):
		'''
		Call operator gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			data (array): data
		'''
		return self.gradient_contract(self.gradient(parameters=parameters,state=state),self.func(parameters=parameters,state=state),state=state)

	def spaces(self,N=None,D=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Size of system
			D (int): Local dimension of system
			space (str,dict,Space): Type of local space
			system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
		'''

		space = self.space if space is None else space

		defaults = dict(
			N = self.N if N is None else N,
			D = self.D if D is None else D,
			space = self.space if space is None else space,
			system = self.system if system is None else system,
		)

		if space is None:
			space = dict(space=space)
		elif not isinstance(space,dict):
			space = dict(space=space)
		else:
			space = dict(**space)

		setter(space,defaults,delimiter=delim,default=True)

		self.space = Space(**space)

		self.N = self.space.N
		self.D = self.space.D		
		self.n = self.space.n

		return


	def times(self,M=None,T=None,tau=None,P=None,time=None,system=None):
		'''
		Set time attributes
		Args:
			M (int): Duration of system
			T (int): Simulation time
			tau (float): Simulation time scale
			P (int): Trotter order		
			time (str,dict,Time): Type of time evolution						
			system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
		'''

		time = self.time if time is None else time

		defaults = dict(
			M = self.M if M is None else M,
			T = self.T if T is None else T,
			tau = self.tau if tau is None else tau,
			P = self.P if P is None else P,
			time = self.time if time is None else time,
			system = self.system if system is None else system,
		)

		if time is None:
			time = dict(time=time)
		elif not isinstance(time,dict):
			time = dict(time=time)
		else:
			time = dict(**time)

		setter(time,defaults,delimiter=delim,default=True)

		self.time = Time(**time)

		self.M = self.time.M
		self.T = self.time.T
		self.P = self.time.P
		self.tau = self.time.tau
		
		return


	def lattices(self,N=None,d=None,lattice=None,system=None):
		'''
		Set lattice attributes
		Args:
			N (int): Size of system
			d (int): Spatial dimension of system
			lattice (str,dict,Lattice): Type of lattice		
			system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)		
		'''		

		lattice = self.lattice if lattice is None else lattice

		defaults = dict(
			N = self.N if N is None else N,
			d = self.d if d is None else d,
			lattice = self.lattice if lattice is None else lattice,
			system = self.system if system is None else system,
		)

		if lattice is None:
			lattice = dict(lattice=lattice)
		elif not isinstance(lattice,dict):
			lattice = dict(lattice=lattice)
		else:
			lattice = dict(**lattice)

		setter(lattice,defaults,delimiter=delim,default=True)

		self.lattice = Lattice(**lattice)

		self.N = self.lattice.N
		self.d = self.lattice.d

		return

	def __str__(self):
		if isinstance(self.string,str):
			string = self.string
		elif isinstance(self.operator,str):
			string = self.operator
		elif self.operator is not None and not isinstance(self.operator,objects) and not callable(self.operator):
			string = '%s'%(delim.join(self.operator))
		elif self.string:
			string = self.string
		else:
			string = self.__class__.__name__
		return string

	def __repr__(self):
		return self.__str__()
	
	def __len__(self):
		return len(self.operator)

	def __hash__(self):
		return (
			hash(self.string) ^ 
			hash(tuple(self.operator) if not isinstance(self.operator,str) else self.operator) ^ 
			hash(tuple(self.site))
			)

	def __key__(self):
		attrs = [self.string,self.operator,self.site]
		key = []
		for attr in attrs:
			if attr is None:
				continue
			if isinstance(attr,scalars):
				key.append(attr)
			else:
				key.extend(attr)
		key = tuple(key)
		return key
	

	def __matmul__(self,other):

		if not isinstance(other,type(self)):
			return self

		assert (self.size == other.size) and (self.ndim == other.ndim) and (self.shape == other.shape), 'Incorrect dimensions %r != %r'%(self.shape,other.shape)

		#TODO: Clean up site,operator,locality initialization and merging, allowing non-local operators and sites

		if ((isinstance(self.operator,str)) or 
			(self.operator is None) or 
			(self.site is None) or 
			(not isinstance(self.operator,str) and 
			 len(self.site) != sum(i not in [self.default] for i in self.operator))
			):
			raise NotImplementedError('TODO: Allow matmul for non-local sites %r and basis operators %r'%(self.site,self.operator))

		if ((isinstance(other.operator,str)) or 
			(other.operator is None) or 
			(other.site is None) or 
			(not isinstance(other.operator,str) and 
			 len(other.site) != sum(i not in [other.default] for i in other.operator))
			):
			raise NotImplementedError('TODO: Allow matmul for non-local sites %r and basis operators %r'%(other.site,other.operator))

		if (self.variable) or (other.variable):
			raise NotImplementedError('TODO: Allow variable object matmul')


		if self.data is None and other.data is None:
			data = self.func()
		elif self.data is None and other.data is not None:
			data = other.func()
		elif isinstance(self.data,arrays) and self.data.ndim == 1 and other.data.ndim == 1:
			data = self.func()*other.func()
		elif isinstance(self.data,arrays) and self.data.ndim == 2 and other.data.ndim == 2:
			data = self.func() @ other.func()
		elif isinstance(self.data,objects):
			data = self.func() @ other.func()
		else:
			data = self.func() @ other.func()

		operator = []
		site = []
		for i in range(self.locality):
			if self.site[i] in other.site:
				s = [i for i in self.operator if i not in [self.default]][i]
				t = [i for i in other.operator if i not in [other.default]][other.site.index(self.site[i])]
				operator.append(s+t)
				site.append(self.site[i])
			else:
				s = [i for i in self.operator if i not in [self.default]][i]
				operator.append(s)
				site.append(self.site[i])
		
		for i in range(other.locality):
			if other.site[i] not in self.site:
				s = [i for i in other.operator if i not in [other.default]][i]
				operator.append(s)
				site.append(other.site[i])

		if self.string is None and other.string is None:
			string = self.string
		elif self.string is None and other.string is not None:
			string = other.string
		elif self.string is not None and other.string is not None:
			string = '@'.join([self.string,other.string])

		kwargs = {
			**other,
			**self,
			**dict(data=data,operator=operator,site=site,string=string),
			**dict(parameters=None,func=None,gradient=None),
			}

		return self.__class__(**kwargs)


	def info(self,display=None,ignore=None,verbose=None):
		'''
		Log class information
		Args:
			display (str,iterable[str]): Show attributes
			ignore (str,iterable[str]): Do not show attributes
			verbose (bool,int,str): Verbosity of message			
		'''		

		msg = []
		options = dict(align='<',space=1,width=2)

		display = None if display is None else [display] if isinstance(display,str) else display
		ignore = None if ignore is None else [ignore] if isinstance(ignore,str) else ignore

		for attr in [None,'string','key','seed','instance','instances','N','D','d','M','tau','T','P','unit','data','shape','size','ndim','dtype','cwd','path','backend','architecture','conf','logger','cleanup']:

			obj = attr
			if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
				continue

			if attr is None:
				attr = 'cls'
				substring = str(self)
			else:
				substring = getattr(self,attr,None)

			if getattr(self,attr,None) is not None:
				if attr in ['data']:
					substring = [substring[i] for i in substring if (substring[i] is not None) and (substring[i].data is not None)] if isinstance(substring,dict) else substring if not isinstance(substring,arrays) else '\n%s'%(str(substring))
				else:
					substring = [substring[i] for i in substring] if isinstance(substring,dict) else substring
				string = '%s: %s'%(attr,substring)
			else:
				string = '%s: %s'%(attr,str(substring))

			msg.append(string)

		for attr in ['operator','locality','site','local']:
		
			obj = attr
			if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
				continue

			if getattr(self,attr,None) is not None:
				string = '%s: %s'%(attr,getattr(self,attr)() if callable(getattr(self,attr)) else getattr(self,attr))
			else:
				string = '%s: %s'%(attr,str(substring))

			msg.append(string)

		if isinstance(self,Objects):

			for attr in (self.data if self.data is not None else []):

				obj = 'parameters'
				if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
					continue

				if self.data.get(attr) is None:
					continue

				string = []
				for subattr in [None,'variable','method','indices','local','site','shape','parameters']:
				
					obj = subattr
					if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
						continue

					if subattr is None:
						subattr = 'data.mean'
						if self.parameters is None or self.parameters() is None or not self.data[attr].parameters.variable:
							substring = self.data[attr].parameters()
						else:
							substring = (self.data[attr].parameters(parameters) for parameters in (
								self.parameters(self.parameters()) if self.parameters.size>1
								else (self.parameters(self.parameters()),)))
							substring = array([i for i in substring if i is not None])
							if substring.size:
								substring = norm(substring)/sqrt(substring.size)
							else:
								substring = None
						if substring is not None and substring.size:
							substring = '%0.4e'%(substring)
						else:
							substring = str(substring)
					else:
						substring = getattrs(self.data[attr].parameters,subattr,default=None,delimiter=delim)
						if isinstance(substring,(str,int,list,tuple,bool,*arrays)):
							substring = str(substring)
						elif isinstance(substring,dict):
							substring = ', '.join(['%s: %s'%(prop,substring[prop]) for prop in substring])
						elif substring is not None:
							substring = '%0.4e'%(substring)
						else:
							substring = str(substring)

					substring = '%s : %s'%(subattr,'{:{align}{space}{width}}'.format(substring,**options))
					
					string.append(substring)

				string = 'parameters.%s\n\t%s'%(self.data[attr],'\n\t'.join([i for i in string if i is not None]))

				msg.append(string)

			for attr in ['parameters']:
		
				obj = attr
				if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
					continue
				string = []
				for subattr in [None,'shape']:
					if subattr is None:
						subattr = attr
						substring = ' '.join([str(i) for i in getattr(self,attr,[]) if getattr(self,attr)[i].variable])
						substring = None if not len(substring) else substring
					else:
						substring = getattr(self.parameters,subattr)
						substring = None if (0 in substring) else substring

					if isinstance(substring,(str,int,list,tuple,*arrays)):
						substring = '%s'%(substring,)
					elif substring is not None:
						substring = '%0.4e'%(substring)
					else:
						substring = str(substring)
					
					if len(substring):
						substring = '%s : %s'%(subattr,substring)
					else:
						substring = None

					string.append(substring)

				string = ', '.join([i for i in string if i is not None])

				msg.append(string)			


		elif isinstance(self,Object):

			string = []
			for attr in [None,'variable','method','indices','local','site','shape','parameters']:
	
				obj = 'parameters'
				if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
					continue

				if attr is None:
					attr = 'data.mean'
					if self.parameters is None or self.parameters() is None or not self.parameters.variable:
						substring = self.parameters()
					else:
						substring = (self.parameters(parameters) for parameters in (
								self.parameters(self.parameters()) if self.parameters().size>1
								else (self.parameters(self.parameters()),))) 
						substring = array([i for i in substring if i is not None])
						if substring.size:
							substring = norm(substring)/sqrt(substring.size)
						else:
							substring = None
					if substring is not None:
						substring = '%0.4e'%(substring)
					else:
						substring = str(substring)
				else:
					substring = getattrs(self.parameters,attr,default=None,delimiter=delim)
					if isinstance(substring,(str,int,list,tuple,bool,*arrays)):
						substring = str(substring)
					elif isinstance(substring,dict):
						substring = ', '.join(['%s: %s'%(prop,substring[prop]) for prop in substring])
					elif substring is not None:
						substring = '%0.4e'%(substring)
					else:
						substring = str(substring)

				substring = '%s : %s'%(attr,'{:{align}{space}{width}}'.format(substring,**options))
				
				string.append(substring)			

			string = 'parameters.%s\n\t%s'%(self,'\n\t'.join(string))
			
			msg.append(string)


		msg = [i if isinstance(i,str) else str(i) for i in msg]

		msg = '\n'.join(msg)

		self.log(msg,verbose=verbose)

		return

	def norm(self,data=None):
		'''
		Normalize class
		Args:
			data (array): data to normalize			
		Returns:
			norm (array): data to normalize
		'''

		if data is None:
			try:
				data = self(self.parameters(),self.state())
			except:
				data = self.data

		if data is None:
			return
		elif not isinstance(data,objects) and not callable(data):
			return

		norm = None
		eps = None

		if self.architecture is None or self.architecture in ['array','tensor','mps']:
		
			shape = self.shape
			ndim = self.ndim
			dtype = self.dtype
			hermitian = self.hermitian		
			unitary = self.unitary	

			if hermitian is None and unitary is None:
				norm = None
				eps = None
			
			elif ndim is None:
				norm = None
				eps = None
			
			elif ndim > 3:
				if not hermitian and unitary:
					norm = einsum('...uij,...ukj->...ik',conjugate(data),data)
					eps = array([identity(shape[-2:],dtype=dtype)]*(norm.ndim-2),dtype=dtype)
			
			elif ndim == 3:
				if not hermitian and not unitary:
					norm = einsum('uij,uik->jk',conjugate(data),data)
					eps = identity(shape[-2:],dtype=dtype)
			
			elif ndim == 2:
				if hermitian and unitary:
					norm = einsum('ii->',data)
					eps = ones(shape[:-2],dtype=dtype)
				elif not hermitian and unitary:
					norm = einsum('ij,kj->ik',conjugate(data),data)
					eps = identity(shape[-2:],dtype=dtype)
				elif hermitian and not unitary:
					norm = einsum('ii->',data)
					eps = ones(shape[:-2],dtype=dtype)
				
			
			elif ndim == 1:
				if not hermitian and unitary:
					norm = einsum('i,i->',conjugate(data),data)
					eps = ones(shape=(),dtype=dtype)
		
		if norm is None or eps is None:
			return

		if dtype not in ['complex256','float128']:
			assert (eps.shape == norm.shape), 'Incorrect operator shape %r != %r'%(eps.shape,norm.shape)
			assert allclose(eps,norm), 'Incorrect norm data%r: %r (hermitian: %r, unitary : %r)'%(eps.shape,norm,hermitian,unitary)

		return norm

class Pauli(Object):
	'''
	Pauli class for Quantum Objects
	Args:
		data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.I for attr in ['I']},
		**{attr: Basis.X for attr in ['X']},
		**{attr: Basis.Y for attr in ['Y']},
		**{attr: Basis.Z for attr in ['Z']},
			}
	
	def setup(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments			
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		site = self.site if site is None else site
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		self.parameters.init(parameters=dict(scale=pi/2))

		if not isinstance(data,objects) and not callable(data):

			N = self.N
			D = self.D
			locality = self.locality
			local = self.local
			ndim = self.ndim

			basis = self.basis
			default = self.default
			dtype = self.dtype
			system = self.system

			if local:
				operator = [operator]*locality if isinstance(operator,str) else [i for i in operator][:locality] if isinstance(operator,iterables) else None
				
				data = tensorprod([(
					basis.get(operator[site.index(i)])(D=D,system=system)
					if isinstance(operator[site.index(i)],str) else 
					operator[site.index(i)]() 
					if callable(operator[site.index(i)]) else
					operator[site.index(i)])
					if i in site else basis.get(default)(D=D,system=system) 
						for i in site]) if operator is not None else None
			
			else:
				operator = [operator]*N if isinstance(operator,str) else [i for i in operator][:N]
				
				data = tensorprod([(
					basis.get(operator[site.index(i)])(D=D,system=system)
					if isinstance(operator[site.index(i)],str) else 
					operator[site.index(i)]() 
					if callable(operator[site.index(i)]) else
					operator[site.index(i)])
					if i in site else basis.get(default)(D=D,system=system) 
					for i in range(N)]) if operator is not None else None

			data = array(data,dtype=dtype) if data is not None else None

		else:

			data = self.data


		if self.state is None or self.state() is None:
			hermitian = False
			unitary = True
		elif self.state().ndim == 1:
			hermitian = False
			unitary = True
		elif self.state().ndim == 2:
			hermitian = True
			unitary = False
		
		hermitian = False
		unitary = True


		if self.parameters() is not None:

			def func(parameters=None,state=None):
				parameters = self.parameters(parameters) if parameters is not None else self.parameters(self.parameters())
				return cos(parameters)*self.identity + -1j*sin(parameters)*self.data
			
			def gradient(parameters=None,state=None):
				grad = self.parameters.grad(parameters)
				parameters = self.parameters(parameters) if parameters is not None else self.parameters(self.parameters())
				return grad*(-sin(parameters)*self.identity + -1j*cos(parameters)*self.data)

		elif self.parameters() is None:
		
			def func(parameters=None,state=None):
				parameters = self.parameters(parameters) if parameters is not None else self.parameters(self.parameters())
				return cos(parameters)*self.identity + -1j*sin(parameters)*self.data

			def gradient(parameters=None,state=None):
				parameters = self.parameters(parameters) if parameters is not None else self.parameters(self.parameters())
				return (-sin(parameters)*self.identity + -1j*cos(parameters)*self.data)


		contract = None
		gradient_contract = None

		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.site = site if site is not None else self.site
		self.string = string if string is not None else self.string
		self.locality = len(self.site) if self.site is not None else self.locality

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = gradient_contract

		self.hermitian = hermitian
		self.unitary = unitary

		return

class Gate(Object):
	'''
	Gate class for Quantum Objects
	Args:
		data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.I for attr in ['i']},
		**{attr: Basis.X for attr in ['x']},
		**{attr: Basis.Y for attr in ['y']},
		**{attr: Basis.Z for attr in ['z']},
		**{attr: Basis.CNOT for attr in ['CNOT','C','cnot']},
		**{attr: Basis.H for attr in ['HADAMARD','H']},
		**{attr: Basis.S for attr in ['PHASE','S']}
		}

	
	def setup(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		site = self.site if site is None else site
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		if not isinstance(data,objects) and not callable(data):

			N = self.N
			D = self.D
			locality = self.locality
			local = self.local
			ndim = self.ndim

			basis = self.basis
			default = self.default
			dtype = self.dtype
			system = self.system


			if local:
				operator = [operator]*locality if isinstance(operator,str) else [i for i in operator][:locality] if isinstance(operator,iterables) else None
				
				data = tensorprod([(
					basis.get(operator[site.index(i)])(D=D,system=system)
					if isinstance(operator[site.index(i)],str) else 
					operator[site.index(i)]() 
					if callable(operator[site.index(i)]) else
					operator[site.index(i)])
					if i in site else basis.get(default)(D=D,system=system) 
						for i in site]) if operator is not None else None
			
			else:
				operator = [operator]*N if isinstance(operator,str) else [i for i in operator][:N]
			
				data = tensorprod([(
					basis.get(operator[site.index(i)])(D=D,system=system)
					if isinstance(operator[site.index(i)],str) else 
					operator[site.index(i)]() 
					if callable(operator[site.index(i)]) else
					operator[site.index(i)])
					if i in site else basis.get(default)(D=D,system=system) 
					for i in range(N)]) if operator is not None else None

			data = array(data,dtype=dtype) if data is not None else None

		else:

			data = self.data


		if self.state is None or self.state() is None:
			hermitian = False
			unitary = True
		elif self.state().ndim == 1:
			hermitian = False
			unitary = True
		elif self.state().ndim == 2:
			hermitian = True
			unitary = False
		hermitian = False
		unitary = True

		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.site = site if site is not None else self.site
		self.string = string if string is not None else self.string
		self.locality = len(self.site) if self.site is not None else self.locality

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = self.gradient_contract

		self.hermitian = hermitian
		self.unitary = unitary
		
		return

class Haar(Object):
	'''
	Haar class for Quantum Objects
	Args:
		data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.I for attr in ['I']},
		**{attr: Basis.unitary for attr in ['U','random','haar']},
		}

	def setup(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments			
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		site = self.site if site is None else site
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		if not isinstance(data,objects) and not callable(data):
		
			N = self.N
			D = self.D
			locality = self.locality
			local = self.local
			ndim = self.ndim

			basis = self.basis
			default = self.default
			dtype = self.dtype
			system = self.system

			random = getattr(self,'random',None)
			seed = getattr(self,'seed',None)
			reset = getattr(self,'reset',None)

			if local:
				if isinstance(operator,str):
					ndim = 2 if ndim is None else ndim
					shape = (D**locality,)*ndim
					random = 'haar' if random is None else random
				
					data = basis.get(operator)(shape=shape,random=random,seed=seed,reset=reset,dtype=dtype)
				
				elif isinstance(operator,iterables):
					ndim = 2 if ndim is None else ndim					
					shape = (D,)*ndim
				
					data = tensorprod([(
						basis.get(operator[site.index(i)])(shape=shape,random=random,seed=seed,reset=reset,dtype=dtype,system=system)
						if isinstance(operator[site.index(i)],str) else 
						operator[site.index(i)]() 
						if callable(operator[site.index(i)]) else
						operator[site.index(i)])
						if i in site else basis.get(default)(D=D,system=system) 
							for i in site]) if operator is not None else None
				elif operator is not None:
				
					data = operator
				
				else:
				
					data = None
			else:
				if isinstance(operator,str):
					ndim = 2 if ndim is None else ndim				
					shape = (D**N,)*ndim
					random = 'haar' if random is None else random
				
					data = basis.get(operator)(shape=shape,random=random,seed=seed,reset=reset,dtype=dtype)
				
				elif isinstance(operator,iterables):
					ndim = 2 if ndim is None else ndim					
					shape = (D,)*ndim
					
					data = tensorprod([(
						basis.get(operator[site.index(i)])(shape=shape,random=random,seed=seed,reset=reset,dtype=dtype,system=system)
						if isinstance(operator[site.index(i)],str) else 
						operator[site.index(i)]() 
						if callable(operator[site.index(i)]) else
						operator[site.index(i)])
						if i in site else basis.get(default)(D=D,system=system) 
							for i in range(N)]) if operator is not None else None
				
				elif operator is not None:
				
					data = operator

				else:

					data = None

			data = array(data,dtype=dtype) if data is not None else None

		else:

			data = self.data


		if self.state is None or self.state() is None:
			hermitian = False
			unitary = True
		elif self.state().ndim == 1:
			hermitian = False
			unitary = True
		elif self.state().ndim == 2:
			hermitian = True
			unitary = False
		hermitian = False
		unitary = True

		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.site = site if site is not None else self.site
		self.string = string if string is not None else self.string
		self.locality = len(self.site) if self.site is not None else self.locality

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = gradient_contract

		self.hermitian = hermitian
		self.unitary = unitary

		return


class Noise(Object):
	'''
	Noise class for Quantum Objects
	Args:
		data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''
	
	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.I for attr in ['eps','noise','rand']},
		**{attr: Basis.depolarize for attr in ['depolarize']},
		**{attr: Basis.amplitude for attr in ['amplitude']},
		**{attr: Basis.element for attr in ['element']},
		**{attr: Basis.bitflip for attr in ['flip','bitflip']},
		**{attr: Basis.phaseflip for attr in ['phaseflip','flipphase']},
		**{attr: Basis.dephase for attr in ['phase','dephase']},
		**{attr: Basis.I for attr in ['dephase-amplitude']},
		}
	
	def __init__(self,data=None,operator=None,site=None,string=None,system=None,**kwargs):		

		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)

		return

	def setup(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments			
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		site = self.site if site is None else site
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		if not isinstance(data,objects) and not callable(data):

			if self.parameters is None:
				self.parameters = 0

			do = (self.parameters() is not None)

			if not do:
				self.data = None
				self.operator = None				
				return

			N = self.N
			D = self.D
			locality = self.locality
			local = self.local
			ndim = self.ndim

			basis = self.basis
			default = self.default
			system = self.system
			dtype = self.dtype

			parameters = self.parameters() if callable(self.parameters) else self.parameters
			

			if isinstance(operator,str):
				locality = N
				site = [i for i in range(N)]

			if parameters is None:
				if local:
					parameters = [None for i in range(locality)]
				else:
					parameters = [None for i in range(N)]
			elif isinstance(parameters,scalars) or parameters.size <= 1:
				if local:
					parameters = [parameters for i in range(locality)]
				else:
					parameters = [parameters for i in range(N)]
			elif isinstance(parameters,iterables):
				if local:
					parameters = [parameters[i] for i in range(locality)]
				else:
					parameters = [parameters[i] for i in range(N)]
			elif isinstance(parameters,arrays):
				if local:
					parameters = [parameters[i] for i in range(locality)]
				else:
					parameters = [parameters[i] for i in range(N)]				
			else:
				if local:
					parameters = [None for i in range(locality)]
				else:
					parameters = [None for i in range(N)]


			if local:
				if isinstance(operator,str):

					if operator in ['phase','dephase','flip','bitflip','phaseflip','flipphase','amplitude','depolarize']:

						operator = [operator]*locality
					
						data = tensorprod([(
							basis.get(operator[site.index(i)])(D=D,parameters=parameters[site.index(i)],system=system)
							if isinstance(operator[site.index(i)],str) else 
							operator[site.index(i)]() 
							if callable(operator[site.index(i)]) else
							operator[site.index(i)])
							if i in site else basis.get(default)(D=D,system=system) 
												for i in site]) if operator is not None else None
					elif operator in ['eps']:

						operator = [operator]*locality

						data = tensorprod([
								array([identity(D,dtype=dtype),diag((1+parameters[site.index(i)])**(arange(D)+2) - 1)]) if i in site else basis.get(default)(D=D,system=system) 
								for i in site]) if operator is not None else None
					
					elif operator in ['noise','rand']:
						seed = getattr(self,'seed',getattr(self,'key',None))
						operator = [operator]*locality

						spawn(reset=seed)
				
						def func(parameters=None,state=None):
							return state + parameters*rand(state.shape,random='uniform',bounds=[-1,1],seed=None,dtype=state.dtype)/2
						def gradient(parameters=None,state=None):
							return 0*state
						def contract(data=None,state=None):
							return data
						def gradient_contract(grad=None,data=None,state=None):
							return grad

				elif isinstance(operator,iterables):
					operator = [i for i in operator]
					
					data = tensorprod([(
						basis.get(operator[site.index(i)])(D=D,parameters=parameters[site.index(i)],system=system)
						if isinstance(operator[site.index(i)],str) else 
						operator[site.index(i)]() 
						if callable(operator[site.index(i)]) else
						operator[site.index(i)])
						if i in site else basis.get(default)(D=D,system=system) 
							for i in site]) if operator is not None else None
				
				elif operator is not None:
				
					data = operator
				
				else:
				
					data = None
			
			else:
			
				if isinstance(operator,str):
			
					if operator in ['phase','dephase','flip','bitflip','phaseflip','flipphase','amplitude','depolarize']:
						operator = [operator]*N
					
						data = tensorprod([(
							basis.get(operator[site.index(i)])(D=D,parameters=parameters[site.index(i)],system=system)
							if isinstance(operator[site.index(i)],str) else 
							operator[site.index(i)]() 
							if callable(operator[site.index(i)]) else
							operator[site.index(i)])
							if i in site else basis.get(default)(D=D,system=system) 
												for i in range(N)]) if operator is not None else None

					elif operator in ['eps']:
						operator = [operator]*N

						data = tensorprod([
								array([identity(D,dtype=dtype),diag((1+parameters[site.index(i)])**(arange(D)+2) - 1)]) if i in site else basis.get(default)(D=D,system=system) 
								for i in range(N)]) if operator is not None else None
					
					elif operator in ['noise','rand']:
						seed = getattr(self,'seed',getattr(self,'key',None))
						operator = [operator]*N

						spawn(reset=seed)
				
						def func(parameters=None,state=None):
							return state + parameters*rand(state.shape,random='uniform',bounds=[-1,1],seed=None,dtype=state.dtype)/2
						def gradient(parameters=None,state=None):
							return 0*state
						def contract(data=None,state=None):
							return data
						def gradient_contract(grad=None,data=None,state=None):
							return grad

				elif isinstance(operator,iterables):
					operator = [i for i in operator]
					
					data = tensorprod([(
						basis.get(operator[site.index(i)])(D=D,parameters=parameters[site.index(i)],system=system) 
						if isinstance(operator[site.index(i)],str) else 
						operator[site.index(i)]() 
						if callable(operator[site.index(i)]) else
						operator[site.index(i)])
						if i in site else basis.get(default)(D=D,system=system) 
							for i in range(N)]) if operator is not None else None
				
				elif operator is not None:
				
					data = operator
				
				else:
				
					data = None

		else:

			data = self.data


		if self.state is None or self.state() is None:
			hermitian = False
			unitary = False
		elif self.state().ndim == 1:
			hermitian = False
			unitary = False
		elif self.state().ndim == 2:
			hermitian = True
			unitary = False
		hermitian = False
		unitary = False

		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.site = site if site is not None else self.site
		self.string = string if string is not None else self.string
		self.locality = len(self.site) if self.site is not None else self.locality

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = gradient_contract

		self.hermitian = hermitian
		self.unitary = unitary

		return


class State(Object):
	'''
	State class for Quantum Objects
	Args:
		data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.I for attr in ['I']},
		**{attr: Basis.state for attr in ['psi','state','random','haar','product']},
		**{attr: Basis.zero for attr in ['zero','zeros','0']},
		**{attr: Basis.one for attr in ['one','ones','1']},
		**{attr: Basis.plus for attr in ['plus','+']},
		**{attr: Basis.minus for attr in ['minus','-']},
		**{attr: Basis.plusi for attr in ['plusi','+i']},
		**{attr: Basis.minusi for attr in ['minusi','-i']},		
		}
	
	def setup(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']			
			site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments				
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		site = self.site if site is None else site
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		if not isinstance(data,objects) and not callable(data):

			N = self.N
			D = self.D
			locality = self.locality
			local = self.local
			ndim = self.ndim

			basis = self.basis
			default = self.default
			dtype = self.dtype
			system = self.system

			random = getattr(self,'random','haar')
			seed = getattr(self,'seed',None)
			scale = getattr(self,'scale',None)
			reset = getattr(self,'reset',None)
			samples = getattr(self,'samples',None)

			if local:
				operator = [operator]*locality if isinstance(operator,str) else [i for i in operator][:locality] if isinstance(operator,iterables) and len(operator) >= locality else (operator*locality)[:locality] if operator is not None else None
				site = [i for i in range(locality)]

				data = tensorprod([(
					basis.get(operator[site.index(i)])(D=D,ndim=ndim,random=random,seed=seed,reset=reset,dtype=dtype,system=system) 
					if isinstance(operator[site.index(i)],str) else 
					operator[site.index(i)]() 
					if callable(operator[site.index(i)]) else
					operator[site.index(i)])
					if i in site else basis.get(default)(D=D,system=system)/D 
						for i in site]) if operator is not None else None

			else:
				operator = [operator]*N if isinstance(operator,str) else [i for i in operator][:N] if isinstance(operator,iterables) and len(operator) >= N else (operator*N)[:N] if operator is not None else None
				site = [i for i in range(N)]

				data = tensorprod([(
					basis.get(operator[site.index(i)])(D=D,ndim=ndim,random=random,seed=seed,reset=reset,dtype=dtype,system=system)
					if isinstance(operator[site.index(i)],str) else 
					operator[site.index(i)]() 
					if callable(operator[site.index(i)]) else
					operator[site.index(i)])
					if i in site else basis.get(default)(D=D,system=system)/D 
					for i in range(N)]) if operator is not None else None

			data = array(data,dtype=dtype) if data is not None else None

			if (samples is not None) and (isinstance(samples,scalars) and samples > 1) and isinstance(data,arrays) and (ndim is not None) and (data.ndim>ndim):
				if isinstance(samples,integers) and (samples > 0):
					shape,bounds,scale,seed,dtype = data.shape[:data.ndim-ndim], [0,1], 'normalize', seed, datatype(dtype)
					samples = rand(size,bounds=bounds,scale=scale,seed=seed,dtype=dtype)
				elif not isinstance(samples,arrays):
					samples = None

				if (samples is not None):
					data = einsum('%s,%s...->...'%((''.join(['i','j','k','l'][:data.ndim-ndim]),)*2),samples,data)
			else:
				samples = None

		else:

			data = self.data

		def func(parameters=None,state=None):
			return self.data

		def gradient(parameters=None,state=None):
			return 0*self.data

		if self.ndim is None:
			hermitian = True
			unitary = False
		elif self.ndim == 1:
			hermitian = False
			unitary = True
		elif self.ndim == 2:
			hermitian = True
			unitary = False
		else:
			hermitian = True
			unitary = False

		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.site = site if site is not None else self.site
		self.string = string if string is not None else self.string
		self.locality = len(self.site) if self.site is not None else self.locality

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = gradient_contract

		self.hermitian = hermitian
		self.unitary = unitary

		return

	def __call__(self,parameters=None,state=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			data (array): data
		'''
		if state is None:
			return self.func(parameters=parameters,state=state)
		else:
			return self.contract(self.func(parameters=parameters,state=state),state=state)

	def grad(self,parameters=None,state=None):
		'''
		Call operator gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			data (array): data
		'''
		if state is None:
			return self.gradient(parameters=parameters,state=state)
		else:
			return self.gradient_contract(self.gradient(parameters=parameters,state=state),self.func(parameters=parameters,state=state),state=state)

	def norm(self,data=None):
		'''
		Normalize class
		Args:
			data (array): data to normalize			
		Returns:
			norm (array): data to normalize
		'''

		if data is None:
			try:
				data = self(self.parameters(),self.state())
			except:
				data = self.data

		if data is None:
			return
		elif not isinstance(data,objects) and not callable(data):
			return

		norm = None
		eps = None

		if self.architecture is None or self.architecture in ['array','tensor']:
		
			shape = self.shape
			ndim = self.ndim
			dtype = self.dtype
			hermitian = self.hermitian		
			unitary = self.unitary	

			if hermitian is None and unitary is None:
				norm = None
				eps = None
			
			elif ndim is None:
				norm = None
				eps = None
			
			elif ndim > 3:
				if not hermitian and unitary:
					norm = einsum('...uij,...ukj->...ik',conjugate(data),data)
					eps = array([identity(shape[-2:],dtype=dtype)]*(norm.ndim-2),dtype=dtype)
			
			elif ndim == 3:
				if not hermitian and not unitary:
					norm = einsum('uij,uik->jk',conjugate(data),data)
					eps = identity(shape[-2:],dtype=dtype)
			
			elif ndim == 2:
				if hermitian and unitary:
					norm = einsum('ii->',data)
					eps = ones(shape[:-2],dtype=dtype)
				elif not hermitian and unitary:
					norm = einsum('ij,kj->ik',conjugate(data),data)
					eps = identity(shape[-2:],dtype=dtype)
				elif hermitian and not unitary:
					norm = einsum('ii->',data)
					eps = ones(shape[:-2],dtype=dtype)
				
			
			elif ndim == 1:
				if not hermitian and unitary:
					norm = einsum('i,i->',conjugate(data),data)
					eps = ones(shape=(),dtype=dtype)
		

		if norm is None or eps is None:
			return

		if dtype not in ['complex256','float128']:
			assert (eps.shape == norm.shape), 'Incorrect operator shape %r != %r'%(eps.shape,norm.shape)
			assert allclose(eps,norm), 'Incorrect norm data%r: %r (hermitian: %r, unitary : %r)'%(eps.shape,norm,hermitian,unitary)

		return norm




class Operator(Object):
	'''
	Class for Operator
	Args:
		data (str,array,tensor,iterable[str,array,tensor],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']		
		site (iterable[int]): site of local operators, i.e) nearest neighbour, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (str): string label of operator
		system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	N = None
	D = None

	default = 'I'	
	basis = {**{attr: Basis.identity for attr in [default]}}

	def __new__(cls,data=None,operator=None,site=None,string=None,system=None,**kwargs):		

		# TODO: Allow multiple different classes to be part of one operator, and swap around localities

		self = None

		setter(kwargs,dict(data=data,operator=operator,site=site,string=string,system=system),delimiter=delim,default=False)

		classes = [Gate,Pauli,Haar,Noise,State,Channel,Operators,Unitary,Hamiltonian,Object]

		for subclass in classes:
			
			if any(isinstance(obj,subclass) for obj in [data if data is not None else operator if operator is not None else None]):

				self = subclass(**kwargs)

				break

			if not all(
				j in subclass.basis for obj in [operator if operator is not None else data if data is not None else None] 
				if isinstance(obj,(str,*iterables))
				for k in (obj if isinstance(obj,iterables) else [obj]) 
				for j in ([k] if k in subclass.basis else k.split(delim) if all(j in subclass.basis for j in k.split(delim)) else [None])):
				continue

			if cls in [Operator]:

				self = subclass(**kwargs)
			
			else:
				
				for attr in subclass.__dict__:
					setattr(cls,attr,getattr(subclass,attr))

				self = subclass.__new__(cls,**kwargs)

				subclass.__init__(self,**kwargs)

			break

		assert (self is not None),'TODO: All operators not in same class'

		return self

class Objects(Object):
	'''
	Class for Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string dictionary for operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional operator keyword arguments			
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
		site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
		string (iterable[str]): string labels of operators
		N (int): Size of system
		M (int): Duration of system
		D (int): Local dimension of system
		d (int): Spatial dimension of system
		T (int): Simulation time
		tau (float): Simulation time scale		
		P (int): Trotter order		
		space (str,dict,Space): Type of local space
		time (str,dict,Time): Type of time evolution						
		lattice (str,dict,Lattice): Type of lattice	
		parameters (iterable[str],dict,Parameters): Type of parameters of operators
		system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''
	
	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}, **{attr: Basis.I for attr in ['operators']}}

	def __init__(self,data=None,operator=None,site=None,string=None,
		N=None,M=None,D=None,d=None,T=None,tau=None,P=None,
		space=None,time=None,lattice=None,parameters=None,system=None,**kwargs):

		setter(kwargs,dict(
			data=data,operator=operator,site=site,string=string,
			N=N,M=M,D=D,d=d,T=T,tau=tau,P=P,
			space=space,time=time,lattice=lattice,parameters=parameters,system=system),
			delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)

		return	

	def init(self,data=None,state=None,conj=False,parameters=None,**kwargs):
		''' 
		Setup class functions
		Args:
			data (bool,dict): data of class, or boolean to retain current data attribute or initialize as None
			state (bool,array,Object): state of class
			conj (bool): conjugate
			parameters (dict,array,Parameters): parameters of class
			kwargs (dict): Additional class keyword arguments			
		'''

		# Set attributes
		self.spaces()
		self.times()
		self.lattices()

		self.shape = () if self.n is None else (self.n,self.n)
		self.size = prod(self.shape)
		self.ndim = len(self.shape)
		self.dtype = self.dtype if self.dtype is not None else None

		self.setup(data)

		# Set data
		if data is None:
			data = {}
		elif data is True:
			data = {}
		elif data is False:
			data = {i: None for i in self.data}
		elif isinstance(data,dict):
			data = {i: data[i] for i in data if i in self.data}

		for i in data:
			if data[i] is None or data[i] is False:
				self.data.pop(i)
				self.data[i] = None
			elif isinstance(data[i],dict):
				self.data[i].init(**data[i])
			elif isinstance(data[i],type(self.data[i])):
				self.data[i] = data[i]
			elif self.data[i] is not None:
				self.data[i].init(data=data[i])


		# Set state
		if state is None:
			state = self.state
		elif state is True:
			state = self.state
		elif state is False:
			state = None

		if state is None or not callable(state):
			def state(parameters=None,state=state):
				return state

		self.state = state


		# Set conjugate
		if conj is None:
			conj = False
		elif conj is True:
			conj = True
		elif conj is False:
			conj = False

		self.conj = conj


		# Set parameters
		if parameters is None:
			parameters = None
		elif parameters is True:
			parameters = None
		elif parameters is False:
			parameters = None

		cls = Parameters
		kwargs = dict(
			parameters={i:self.data[i].parameters 
				for i in self.data 
				if ((self.data[i] is not None) and 
					(self.data[i].data is not None))
				} if parameters is None else parameters if not isinstance(parameters,dict) else None,
			system=self.system
		)

		parameters = cls(**kwargs)

		self.parameters = parameters


		# Set identity
		identity = tensorprod([Basis.identity(D=self.D,system=self.system)]*self.N) if self.identity is None else self.identity

		self.identity = identity


		# Set data
		for i in self.data:
			
			if self.data[i] is None:
				continue

			kwargs = dict(
				state=self.state
				)
			self.data[i].init(**kwargs)

		# Set attributes
		boolean = lambda i: ((self.data[i] is not None) and (self.data[i].data is not None))
		if self.state is None or self.state() is None:
			hermitian = all(self.data[i].hermitian for i in self.data if boolean(i))
			unitary = all(self.data[i].unitary for i in self.data if boolean(i))
		elif self.state().ndim == 1:
			hermitian = False
			unitary = True
		elif self.state().ndim == 2:
			hermitian = True
			unitary = False

		self.hermitian = hermitian
		self.unitary = unitary

		def func(parameters,state):
			state = state if state is not None else self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			shape = (self.M,len([i for i in self.data if self.data[i] is not None]))
			indices = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None]
			out = state
			if parameters is not None:
				for i in indices:
					out = self.data[i%shape[1]](parameters=parameters[i//shape[1]],state=out)
			else:
				for i in indices:
					out = self.data[i%shape[1]](state=out)
			return out

		def grad(parameters,state):
			state = state if state is not None else self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			grad = zeros((parameters.size,*state.shape),dtype=state.dtype)
			shape = (self.M,len([i for i in self.data if self.data[i] is not None]))
			indices = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None]
			indexes = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None and self.data[i].variable]
			if parameters is not None:
				for i in indexes:
					out = state
					for j in (j for j in indices if j<i):
						out = self.data[j%shape[1]](parameters=parameters[j//shape[1]],state=out)
					out = self.data[i%shape[1]].grad(parameters=parameters[i//shape[1]],state=out)
					for j in (j for j in indices if j>i):
						out = self.data[j%shape[1]](parameters=parameters[j//shape[1]],state=out)
					grad = inplace(grad,indexes.index(i),out,'add')
			else:
				for i in indexes:
					out = state
					for j in (j for j in indices if j<i):
						out = self.data[j%shape[1]](state=out)
					out = self.data[i%shape[1]].grad(state=out)
					for j in (j for j in indices if j>i):
						out = self.data[j%shape[1]](state=out)
					grad = inplace(grad,indexes.index(i),out,'add')
			return grad
		# Set functions
		def func(parameters,state):
			state = state if state is not None else self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			shape = (self.M,len([i for i in self.data if self.data[i] is not None]))
			indices = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None]
			out = state
			if parameters is not None:
				for i in indices:
					out = self.data[i%shape[1]](parameters=parameters[i//shape[1]],state=out)
			else:
				for i in indices:
					out = self.data[i%shape[1]](state=out)
			return out

		def grad(parameters,state):
			state = state if state is not None else self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			grad = zeros((parameters.size,*state.shape),dtype=state.dtype)
			shape = (self.M,len([i for i in self.data if self.data[i] is not None]))
			indices = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None]
			indexes = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None and self.data[i].variable]
			if parameters is not None:
				for i in indexes:
					out = state
					for j in (j for j in indices if j<i):
						out = self.data[j%shape[1]](parameters=parameters[j//shape[1]],state=out)
					out = self.data[i%shape[1]].grad(parameters=parameters[i//shape[1]],state=out)
					for j in (j for j in indices if j>i):
						out = self.data[j%shape[1]](parameters=parameters[j//shape[1]],state=out)
					grad = inplace(grad,indexes.index(i),out,'add')
			else:
				for i in indexes:
					out = state
					for j in (j for j in indices if j<i):
						out = self.data[j%shape[1]](state=out)
					out = self.data[i%shape[1]].grad(state=out)
					for j in (j for j in indices if j>i):
						out = self.data[j%shape[1]](state=out)
					grad = inplace(grad,indexes.index(i),out,'add')
			return grad

		grad_automatic = gradient(self,mode='fwd',move=True)
		grad_finite = gradient(self,mode='finite',move=True)
		grad_analytical = grad

		grad = grad_automatic

		self.func = func
		self.gradient = grad
		self.gradient_automatic = grad_automatic
		self.gradient_finite = grad_finite
		self.gradient_analytical = grad_analytical


		# Set wrapper
		if self.architecture is None or self.architecture in ['array','tensor'] or self.architecture not in ['mps']:
			parameters = self.parameters(self.parameters())
			state = self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			wrapper = jit
		
		elif self.architecture in ['mps']:
			parameters = self.parameters(self.parameters())
			state = self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			wrapper = partial
	
		self.func = wrapper(self.func,parameters=parameters,state=state)
		self.gradient = wrapper(self.gradient,parameters=parameters,state=state)
		self.gradient_automatic = wrapper(self.gradient_automatic,parameters=parameters,state=state)
		self.gradient_finite = wrapper(self.gradient_finite,parameters=parameters,state=state)
		self.gradient_analytical = wrapper(self.gradient_analytical,parameters=parameters,state=state)

		return

	def setup(self,data=None,operator=None,site=None,string=None,**kwargs):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,site,string dictionary for operator
				operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
				site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
				string (iterable[str]): string labels of operators
				kwargs (dict): Additional operator keyword arguments			
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional class keyword arguments		
		'''

		# Get operator,site,string from data
		objs = Dictionary(operator=operator,site=site,string=string)

		for obj in objs:
			objs[obj] = [] if objs[obj] is None else objs[obj] if isinstance(objs[obj],list) else [objs[obj]]

		# Get data and kwargs
		if data is None:
			data = None
		elif all(isinstance(obj,Object) for obj in data):
			for obj in objs:
				objs[obj] = None
		elif isinstance(data,dict) and all(isinstance(data[name],dict) or data[name] is None for name in data for obj in objs) and all(any(obj in data[name] for obj in objs) for name in data if data[name] is not None):
			for obj in objs:
				objs[obj].extend([data[name].get(obj) for name in data if data[name] is not None])
			
			kwargs.update({kwarg: [data[name][kwarg] if kwarg in data[name] else null for name in data if data[name] is not None] 
				for kwarg in set(kwarg for name in data if data[name] is not None for kwarg in data[name] if kwarg not in objs)
				})

			data = None

		# Lattice sites
		sites = self.lattice # sites types on lattice
		locality = self.N
		indices = {'i': ['i'],'<ij>':['i','j'],'>ij<':['i','j'],'i<j':['i','j'],'ij':['i','j'],'i...j':['i','j']}   # allowed symbolic indices and maximum locality of many-body site interactions

		# Get number of operators
		size = min([len(objs[obj]) for obj in objs if objs[obj] is not None],default=0)
		
		# Get attribute of symbolic indices
		attr = 'site'
		attrs = {
			'site': lambda attr,value,values,indices: [dict(zip(indices,
								value if not isinstance(value,integers) else (value,))
							).get(i,int(i) if not isinstance(i,str) else i) 
							for i in values[attr]],
			'string': lambda attr,value,values,indices:	values[attr]
							# delim.join([
							# '%s%s'%(str(i),str(dict(zip(indices,
							# 	value if not isinstance(value,integers) else (value,))
							# ).get(i,int(i) if not isinstance(i,str) else i))) 
							# for i,j in zip(values[attr],values['operator'])])
			}
		
		# Get data
		for index in range(size):

			key = None
			tmp = {obj: copy(objs[obj].pop(0)) for obj in objs}
			tmps = {kwarg: copy(kwargs[kwarg].pop(0)) for kwarg in kwargs}

			if tmp[attr] is None:
				key = None
			elif isinstance(tmp[attr],scalars) and tmp[attr] in indices:
				if len(indices[tmp[attr]]) > locality:
					key = None
					tmp[attr] = None
				else:
					key = tmp[attr]
					tmp[attr] = indices[tmp[attr]]
			elif isinstance(tmp[attr],scalars):
				if tmp[attr] > locality:
					key = None
					tmp[attr] = None
				else:
					tmp[attr] = [tmp[attr]]
			elif not isinstance(tmp[attr],scalars):
				if any((i in indices and len(indices[tmp[attr]]) > locality) or (i > locality) for i in tmp[attr]):
					key = None
					tmp[attr] = None
				else:
					for i in tmp[attr]:
						if i in indices:
							key = i
							tmp[attr][tmp[attr].index(i)] = indices[i][tmp[attr].index(i)]
			if key is not None:
				for i,index in enumerate(sites(key)):
					value = {}
					for obj in objs:
						
						if obj in attrs:
							value[obj] = attrs[obj](obj,index,tmp,indices[key])
						else:
							value[obj] = tmp[obj]

					exists = [[i if value[obj] == item else None for i,item in enumerate(objs[obj])] 
						for obj in objs]
					
					if any(len(set(i))==1 for i in zip(*exists) if any(j is not None for j in i)):
						continue

					for obj in objs:
						objs[obj].append(value[obj])
					
					for kwarg in kwargs:
						kwargs[kwarg].append(copy(tmps[kwarg]))

			else:
				for obj in objs:
					objs[obj].append(tmp[obj])	

				for kwarg in kwargs:
					kwargs[kwarg].append(copy(tmps[kwarg]))

		# Set class dependent attributes 
		# i.e) set parameters data with site-dependent data 
		# i.e) set parameters shape with depth-M-dependent shape
		attributes = {}
		def decorator(func):
			def wrapper(i,attr,attrs,objs,kwargs):
				if kwargs.get(attr) is None:
					kwargs[attr] = [None for j in range(i)]
				if len(kwargs[attr]) < i:
					kwargs[attr] = [*kwargs[attr],*[None for j in range(1+i-len(kwargs[attr]))]]
				func(i,attr,attrs,objs,kwargs)
				return
			return wrapper

		attribute = 'parameters.data'
		@decorator
		def func(i,attr,attrs,objs,kwargs):

			if not isinstance(kwargs[attr][i],dict):
				data = None if isinstance(kwargs[attr][i],nulls) else kwargs[attr][i]
				kwargs[attr][i] = dict(data=data)

			size = len(kwargs.get(attr,[]))
			index = [j for j in range(size) if objs.string[j] == objs.string[i]].index(i)
			obj = getter(kwargs[attr][i],attrs,delimiter=delim)
			default = None

			if obj is None:
				obj = obj
			elif isinstance(obj,dict):
				obj = obj.get(
					tuple(objs.site[i]),obj.get(str(objs.string[i]),
					obj.get(str(index),obj.get(int(index),default))))
			elif isinstance(obj,iterables):
				obj = obj[index%len(obj)] if len(obj) else default
			else:
				obj = obj

			setter(kwargs[attr][i],{attrs:obj},delimiter=delim,default=True)
			return
		attributes[attribute] = func

		attribute = 'parameters.axis'
		@decorator
		def func(i,attr,attrs,objs,kwargs):
			
			if not isinstance(kwargs[attr][i],dict):
				data = None if isinstance(kwargs[attr][i],nulls) else kwargs[attr][i]
				kwargs[attr][i] = dict(data=data)

			size = len(kwargs.get(attr,[]))
			index = [j for j in range(size) if objs.string[j] == objs.string[i]].index(i)
			obj = getter(kwargs[attr][i],attrs,delimiter=delim)
			default = ['M']

			if obj is None:
				obj = [j for j in default]
			elif isinstance(obj,iterables):
				obj = [*obj,*(j for j in default if j not in obj)]
			else:
				obj = obj

			setter(kwargs[attr][i],{attrs:obj},delimiter=delim,default=True)
			return			
		attributes[attribute] = func

		for attribute in attributes:
			attr,attrs = attribute.split(delim)[0] if attribute.count(delim)>=0 else None,delim.join(attribute.split(delim)[1:]) if attribute.count(delim)>0 else None
			size = len(kwargs.get(attr,[]))

			for i in range(size):
				if isinstance(kwargs[attr][i],nulls):
					continue
				attributes[attribute](i,attr,attrs,objs,kwargs)


		# Set class attributes

		self.extend(data=data,**objs,kwargs=kwargs)

		return


	def __call__(self,parameters=None,state=None):
		'''
		Class function
		Args:
			parameters (array): parameters		
			state (obj): state
		Returns
			out (array): Return of function
		'''

		parameters = self.parameters(parameters) if parameters is not None else self.parameters(self.parameters())

		return self.func(parameters=parameters,state=state)


	def grad(self,parameters=None,state=None):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			out (array): Return of function
		'''		
		return self.gradient(parameters=parameters,state=state)


	def grad_automatic(self,parameters=None,state=None):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			out (array): Return of function
		'''		
		return self.gradient_automatic(parameters=parameters,state=state)


	def grad_finite(self,parameters=None,state=None):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			out (array): Return of function
		'''		
		return self.gradient_finite(parameters=parameters,state=state)


	def grad_analytical(self,parameters=None,state=None):
		''' 
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			out (array): Return of function
		'''		

		parameters = self.parameters(parameters) if parameters is not None else self.parameters(self.parameters())

		return self.gradient_analytical(parameters=parameters,state=state)


	def __len__(self):
		return len(self.data)


	def __str__(self):
		if self.data is not None:
			size = len(self) if self.data is not None else 0
			delimiter = ' '
			multiple_time = (self.M>1) if self.M is not None else None
			multiple_space = [size>1 and False for i in range(size)]
			string = '%s%s%s%s'%(
					'{' if multiple_time else '',
					delimiter.join(['%s%s%s'%(
						'(' if multiple_space[i] else '',
						self.data[i].string,
						')' if multiple_space[i] else '',
						) for i in range(size) if self.data[i] is not None]),
					'}' if multiple_time else '',
					'%s'%('^%s'%(self.M) if multiple_time else '') if multiple_time else '')
		else:
			string = self.__class__.__name__
		return string


	def set(self,data=None,index=None):
		'''
		Set data of class
		Args:
			data (dict): data for class
			index (object): index of data for class
		'''

		if self.data is None or not all(isinstance(self.data[i],Object) for i in self.data):
			self.clear()

		if index is not None and data is not None:

			index = len(self)+1+index if index < 0 else index

			self.data = insertion(self.data,index,{index:data})

		elif data is not None:
			
			self.data = type(self.data)(data)

		return

	def get(self,index=None):
		'''
		Get data of class
		Args:
			index (object): index of data of class
		Returns:
			data (object): data of class
		'''

		if index is not None:
			
			index = len(self) + index if index < 0 else index

			data = self.data.get(index)
		else:
			data = self.data

		return data

	def clear(self):
		'''
		Clear data of class
		'''

		self.data = Dictionary()

		return

	def sort(self,layout=None):
		'''
		Sort data of class
		Args:
			layout (dict[str,str,object,iterable],iterable[str,dict],callable): layout of data
				dict[str,iterable]: data attributes to sort on as keys and order of data attributes as values:
					{attr:[value]} or data {attr:None} to sort by all data attr values
				iterable[str,dict]: objects to sort on, either strings or dictionaries:
					[attr], [data.string], [{attr:[value]}], 
					'_attr' may also be used for reversed sorting of attributes 
				callable: function to yield data layout, with signature layout(data) = data					
		'''

		def string(obj,data):
	
			if all(hasattr(data[i],obj) for i in data):
				index = {i: getattr(data[i],obj) for i in data}
				index = sorted(index,key=lambda i: index[i])
			elif obj.startswith('_') and all(hasattr(data[i],'_'.join(obj.split('_')[1:])) for i in data):
				index = {i: getattr(data[i],'_'.join(obj.split('_')[1:])) for i in data}
				index = sorted(index,key=lambda i: index[i])
			elif obj in [data[i].string for i in data]:
				index = [i for i in data[i] if data[i].string == obj]
			elif obj.startswith('_') and '_'.join(obj.split('_')[1:]) in [data[i].string for i in data]:
				index = [i for i in data[i] if data[i].string == '_'.join(obj.split('_')[1:])][::-1]
			else:
				index = []

			return index

		def dictionary(obj,data):
			index = {}
			for attr in obj:

				if all(hasattr(data[i],attr) for i in data):
					
					if obj[attr] is None:
						value = {i: getattr(data[i],attr) for i in data}
						value = {i: tuple(value[i]) if isinstance(value[i],iterables) else value[i] for i in value}
						values = list(sorted(set((value[i] for i in value))))
					else:
						value = {i: getattr(data[i],attr) for i in data if getattr(data[i],attr) in obj[attr]}
						values = [i for i in obj[attr]]
					
					for i in data:
						if i not in value:
							continue
						if i not in index:
							index[i] = {}
						
						index[i][attr] =  values.index(value[i])

				elif attr.startswith('_') and all(hasattr(data[i],'_'.join(attr.split('_')[1:])) for i in data):
			
					if obj[attr] is None:
						value = {i: getattr(data[i],'_'.join(attr.split('_')[1:])) for i in data}
						value = {i: tuple(value[i]) if isinstance(value[i],iterables) else value[i] for i in value}
						values = list(sorted(set((value[i] for i in value))))
					else:
						value = {i: getattr(data[i],'_'.join(attr.split('_')[1:])) for i in data if getattr(data[i],'_'.join(attr.split('_')[1:])) in obj[attr]}
						values = [i for i in obj[attr]]
					
					for i in data:
						if i not in value:
							continue
						if i not in index:
							index[i] = {}
						
						index[i][attr] =  -values.index(value[i])


			index = {i: tuple((index[i][attr] for attr in index[i])) for i in index if all(attr in index[i] for attr in obj)}

			index = sorted(index,key=lambda i: index[i])

			return index


		self.set()

		data = self.get()

		layout = self.layout if layout is None else layout


		layout = [layout] if layout is None or callable(layout) or isinstance(layout,(str,dict)) else [obj for obj in layout]

		indices = []
		for obj in layout:
			
			index = None

			if callable(obj):
				
				data = obj(data)
			
			elif isinstance(obj,dict):
				
				index = dictionary(obj,data)

			elif isinstance(obj,str):

				index = string(obj,data)

			if index is not None:

				indices.append(index)


		if indices:
			
			data = {i: data[i] for index in indices for i in index}

		
		data = {index: data[i] for index,i in enumerate(data)}

		self.set(data)

		return

	def extend(self,data=None,operator=None,site=None,string=None,kwargs=None):
		'''
		Extend to class
		Args:
			data (iterable[str,Operator]): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			site (iterable[str,iterable[int,str]]): site of local operators, allowed strings in ['i','ij','i<j','<ij>','>ij<','i...j']
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional operator keyword arguments			
		'''

		size = min([len(i) for i in (data,operator,site,string) if i is not None],default=0)

		length = min([len(i) for i in (kwargs[kwarg] for kwarg in kwargs) if i is not null],default=size) if kwargs is not None else None
		kwargs = [{kwarg: kwargs[kwarg][i] for kwarg in kwargs} for i in range(length)] if kwargs is not None else None
		
		if not size:
			return

		if data is None:
			data = [None]*size
		if operator is None:
			operator = [None]*size
		if site is None:
			site = [None]*size
		if string is None:
			string = [None]*size						
		if kwargs is None:
			kwargs = [None]*size	

		for _data,_operator,_site,_string,_kwargs in zip(data,operator,site,string,kwargs):
			self.append(_data,_operator,_site,_string,_kwargs)

		return


	def append(self,data=None,operator=None,site=None,string=None,kwargs=None):
		'''
		Append to class
		Args:
			data (str,Operator): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments			
		'''
		index = -1
		self.insert(index,data,operator,site,string,kwargs)
		return


	def insert(self,index,data,operator,site,string,kwargs=None):
		'''
		Insert to class
		Args:
			index (int): index to insert operator
			data (str,Operator): data of operator
			operator (str): string name of operator
			site (int): site of local operator
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments						
		'''

		def sites(obj,data):
			obj = list(set(j 
					for i in data
					if data[i] is not None and data[i].site is not None
					for j in (data[i].site 
					if isinstance(data[i].site,iterables) else 
					(data[i].site,)) if j is not None))
			return obj

		def operators(obj,data):
			obj = [data[i].operator[data[i].site.index(j)] 
				if isinstance(data[i].operator,iterables) and isinstance(data[i].site,iterables) and j in data[i].site 
				else data[i].operator[j] if isinstance(data[i].operator,iterables) and len(data[i].operator)>j
				else data[i].operator
				for i in data 
				for j in sites(obj,data)
				if ((data[i] is not None and data[i].operator is not None and data[i].site is not None) and (
					(isinstance(data[i].operator,iterables) and isinstance(data[i].site,iterables) and j in data[i].site) or
					(isinstance(data[i].operator,iterables) and len(data[i].operator)>j) or
					(not isinstance(data[i].operator,iterables) and data[i].operator is not None)))]
			obj = list([separ.join(tuple(sorted(list(set(obj)),key = lambda i: obj.index(i))))
				for j in sites(obj,data)])
			return obj

		def strings(obj,data):
			obj = delim.join([data[i].string 
				for i in data 
				if data[i] is not None and data[i].string is not None])
			return obj

		def variables(obj,data):
			obj = any(data[i].variable for i in data) if data is not None else False
			return obj

		def localities(obj,data):
			obj = len(sites(obj,data))
			return obj

		def attributes(N,D,local,data):
			N = max(N if N is not None else 0,max(sites(None,data),default=0)+1,max([data[i].N for i in data if data[i].N is not None],default=0))
			D =	max(D if D is not None else 0,max([data[i].D for i in data if data[i].D is not None],default=0))
			local = all(data[i].local for i in data)
			return N,D,local

		cls = Operator
		defaults = {}
		kwargs = {kwarg: kwargs[kwarg] for kwarg in kwargs if not isinstance(kwargs[kwarg],nulls)} if kwargs is not None else defaults
		
		setter(kwargs,{attr: getattr(self,attr) for attr in self if attr not in cls.defaults and attr not in ['data','operator','site','string']},delimiter=delim,default=False)
		setter(kwargs,dict(state=self.state,local=self.local,verbose=False,system=self.system),delimiter=delim,default=True)
		setter(kwargs,defaults,default=False)


		data = cls(**{**dict(data=data,operator=operator,site=site,string=string),**kwargs})

		self.set(data,index=index)

		self.sort()

		data = self.data


		operator = None
		site = None
		string = None
		variable = None
		locality = None
		N,D,local = None,None,None

		operator = operators(operator,data)
		site = sites(site,data)
		string = strings(string,data)
		variable = variables(variable,data)
		locality = localities(locality,data)

		N,D,local = attributes(N,D,local,data)

		self.N = N
		self.D = D
		self.local = local

		self.spaces()
		self.times()
		self.lattices()

		shape = () if self.n is None else (self.n,self.n)
		size = prod(self.shape)
		ndim = len(self.shape)
		dtype = self.dtype if self.dtype is not None else None

		self.data = data
		self.operator = operator
		self.site = site
		self.string = string
		self.variable = variable
		self.locality = locality

		self.shape = shape
		self.size = size
		self.ndim = ndim
		self.dtype = dtype

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
			setter(default,data[attr],default=func)

		return

class Channel(Objects):
	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}, **{attr: Basis.I for attr in ['channel']}}

	def init(self,data=None,state=None,conj=False,parameters=None):
		''' 
		Setup class functions
		Args:
			data (bool,dict): data of class, or boolean to retain current data attribute or initialize as None
			state (bool,array,Object): state of class
			conj (bool): conjugate
			parameters (dict,array,Parameters): parameters of class
		'''

		super().init(data=data,state=state,conj=conj,parameters=parameters)

		# Set data
		for i in self.data:
			
			if self.data[i] is None:
				continue

			kwargs = dict(
				parameters=dict(parameters=dict(tau=self.tau) if self.data[i].unitary else None),
				state=self.state
				)
			self.data[i].init(**kwargs)


		# Set attributes
		boolean = lambda i: ((self.data[i] is not None) and (self.data[i].data is not None))
		if self.state is None or self.state() is None:
			hermitian = all(self.data[i].hermitian for i in self.data if boolean(i))
			unitary = all(self.data[i].unitary for i in self.data if boolean(i))
		elif self.state().ndim == 1:
			hermitian = False
			unitary = True
		elif self.state().ndim == 2:
			hermitian = True
			unitary = False

		self.hermitian = hermitian
		self.unitary = unitary


		# Set functions
		func = scheme(data=self.data,parameters=self.parameters,state=self.state,conj=self.conj,size=self.M,compilation=dict(trotter=self.P,**self),architecture=self.architecture,verbose=self.verbose)
		grad = gradient_scheme(data=self.data,parameters=self.parameters,state=self.state,conj=self.conj,size=self.M,compilation=dict(trotter=self.P,**self),architecture=self.architecture,verbose=self.verbose)

		grad_automatic = gradient(self,mode='fwd',move=True)
		grad_finite = gradient(self,mode='finite',move=True)
		grad_analytical = grad

		grad = grad_automatic

		self.func = func
		self.gradient = grad
		self.gradient_automatic = grad_automatic
		self.gradient_finite = grad_finite
		self.gradient_analytical = grad_analytical		


		# Set wrapper
		if self.architecture is None or self.architecture in ['array','tensor'] or self.architecture not in ['mps']:
			parameters = self.parameters(self.parameters())
			state = self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			wrapper = jit

		elif self.architecture in ['mps']:
			parameters = self.parameters(self.parameters())
			state = self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			wrapper = partial

		self.func = wrapper(self.func,parameters=parameters,state=state)
		self.gradient = wrapper(self.gradient,parameters=parameters,state=state)
		self.gradient_automatic = wrapper(self.gradient_automatic,parameters=parameters,state=state)
		self.gradient_finite = wrapper(self.gradient_finite,parameters=parameters,state=state)
		self.gradient_analytical = wrapper(self.gradient_analytical,parameters=parameters,state=state)

		return

class Operators(Objects):
	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}, **{attr: Basis.I for attr in ['operators']}}

	def init(self,data=None,state=None,conj=False,parameters=None):
		''' 
		Setup class functions
		Args:
			data (bool,dict): data of class, or boolean to retain current data attribute or initialize as None
			state (bool,array,Object): state of class
			conj (bool): conjugate
			parameters (dict,array,Parameters): parameters of class
		'''

		super().init(data=data,state=state,conj=conj,parameters=parameters)


		# Set data
		for i in self.data:
			
			if self.data[i] is None:
				continue

			kwargs = dict(
				state=self.state
				)
			self.data[i].init(**kwargs)


		# Set attributes
		boolean = lambda i: ((self.data[i] is not None) and (self.data[i].data is not None))
		if self.state is None or self.state() is None:
			hermitian = all(self.data[i].hermitian for i in self.data if boolean(i))
			unitary = all(self.data[i].unitary for i in self.data if boolean(i))
		elif self.state().ndim == 1:
			hermitian = False
			unitary = True
		elif self.state().ndim == 2:
			hermitian = True
			unitary = False

		self.hermitian = hermitian
		self.unitary = unitary


		# Set functions
		def func(parameters,state):
			state = state if state is not None else self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			shape = (self.M,len([i for i in self.data if self.data[i] is not None]))
			indices = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None]
			out = state
			if parameters is not None and len(parameters):
				for i in indices:
					out = self.data[i%shape[1]](parameters=parameters[i//shape[1]],state=out)
			else:
				for i in indices:
					out = self.data[i%shape[1]](parameters=parameters,state=out)
			return out

		def grad(parameters,state):
			state = state if state is not None else self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
			grad = zeros((parameters.size,*state.shape),dtype=state.dtype)
			shape = (self.M,len([i for i in self.data if self.data[i] is not None]))
			indices = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None]
			indexes = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None and self.data[i].variable]
			if parameters is not None and len(parameters):
				for i in indexes:
					out = state
					for j in (j for j in indices if j<i):
						out = self.data[j%shape[1]](parameters=parameters[j//shape[1]],state=out)
					out = self.data[i%shape[1]].grad(parameters=parameters[i//shape[1]],state=out)
					for j in (j for j in indices if j>i):
						out = self.data[j%shape[1]](parameters=parameters[j//shape[1]],state=out)
					grad = inplace(grad,indexes.index(i),out,'add')
			else:
				for i in indexes:
					out = state
					for j in (j for j in indices if j<i):
						out = self.data[j%shape[1]](parameters=parameters,state=out)
					out = self.data[i%shape[1]].grad(parameters=parameters,state=out)
					for j in (j for j in indices if j>i):
						out = self.data[j%shape[1]](parameters=parameters,state=out)
					grad = inplace(grad,indexes.index(i),out,'add')
			return grad

		grad_automatic = gradient(self,mode='fwd',move=True)
		grad_finite = gradient(self,mode='finite',move=True)
		grad_analytical = grad

		grad = grad_automatic

		self.func = func
		self.gradient = grad
		self.gradient_automatic = grad_automatic
		self.gradient_finite = grad_finite
		self.gradient_analytical = grad_analytical

		# Set wrapper
		parameters = self.parameters(self.parameters())
		state = self.state() if self.state is not None and self.state() is not None else Basis.identity(D=self.n,dtype=self.dtype)
		wrapper = partial

		self.func = wrapper(self.func,parameters=parameters,state=state)
		self.gradient = wrapper(self.gradient,parameters=parameters,state=state)
		self.gradient_automatic = wrapper(self.gradient_automatic,parameters=parameters,state=state)
		self.gradient_finite = wrapper(self.gradient_finite,parameters=parameters,state=state)
		self.gradient_analytical = wrapper(self.gradient_analytical,parameters=parameters,state=state)

		return

class Hamiltonian(Channel):
	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}, **{attr: Basis.I for attr in ['hamiltonian']}}

class Unitary(Channel):
	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}, **{attr: Basis.I for attr in ['unitary']}}


class Module(System):
	'''
	Class for Module
	Args:
		model (Object,iterable[Object],dict[str,Object): model for module, iterable of models or dictionary with lattice locality where {'<ij>':model}
		N (int): Size of system
		M (int): Duration of system
		d (int): Spatial dimension of system		
		lattice (str,dict,Lattice): Type of lattice	
		state (array,State): state for module			
		parameters (iterable[str],dict,Parameters): Type of parameters of operators
		system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		kwargs (dict): Additional system keyword arguments	
	'''

	defaults = dict(
		model=None,
		N=None,M=None,d=None,
		state=None,parameters=None,variable=False,
		data=None,measure=None,lattice=None,callback=None,
		func=None,gradient=None,
		system=None,
		)

	def __init__(self,model=None,N=None,M=None,d=None,lattice=None,
		state=None,parameters=None,system=None,**kwargs):

		setter(kwargs,dict(
			model=model,N=N,M=M,d=d,lattice=lattice,
			state=state,parameters=parameters,system=system),
			delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)

		self.init()

		self.info()

		return	

	def init(self,model=None,state=None,parameters=None,**kwargs):
		''' 
		Setup class functions
		Args:
			model (Object,iterable[Object],dict[str,Object): model for module, iterable of models or dictionary with lattice locality where {'<ij>':model}
			state (State): state for module
			parameters (dict,array,Parameters): parameters of class
			kwargs (dict): Additional class keyword arguments			
		'''


		# Set attributes
		self.lattices()

		# Set model
		model = self.model if model is None else model

		self.model = model


		# Set parameters
		parameters = self.parameters if parameters is None else parameters
		if parameters is None or not callable(parameters):
			def parameters(parameters=parameters):
				return parameters
		
		self.parameters = parameters


		# Set state
		state = self.state if state is None else state

		if state is None or not callable(state):
			parameters = self.parameters()
			def state(parameters=parameters,state=state):
				return state
		
		self.state = state

	
		# Setup
		self.setup()

		return

	def setup(self,model=None,state=None,parameters=None,**kwargs):
		'''
		Setup class
		Args:
			model (Object,iterable[Object],dict[str,Object): model for module, iterable of models or dictionary with lattice locality where {'<ij>':model}
			state (State): state for module			
			parameters (dict,array,Parameters): parameters of class			
			kwargs (dict): Additional class keyword arguments		
		'''

		# Settings
		model = self.model if model is None else model
		state = self.state if state is None else state
		parameters = self.parameters if parameters is None else parameters


		# Attributes
		self.lattices()


		# Measure
		cls = Measure
		measure = self.measure if isinstance(self.measure,dict) else {}
		measure = {**namespace(cls,self),**{attr: getattr(self,attr) for attr in self.system if hasattr(self,attr)},**measure,**dict(system=self.system)}
		measure = cls(**measure)


		# Data
		models = {None:model} if model is None or callable(model) or not isinstance(model,dict) else {index: model[index] for index in model}
		options = self.options if self.options is not None else {}
		obj = state() if callable(state) else state

		data = []

		for index in models:
			
			model = models[index]

			if model is None:
				continue

			locality = model.locality

			if index is None or isinstance(index,str):
				indices = self.lattice(index)
			else:
				indices = [index]

			parameters = model.parameters()
			state = tensorprod([obj]*locality)
			model.init(state=state)

			parameters = measure.parameters()
			state = [obj]*locality
			
			model = measure.transform(parameters=parameters,state=state,model=model)

			for where in indices:

				def obj(parameters,state,where=where,model=model,options=options,**kwargs):
					return model(parameters=parameters,state=state,where=where,**{**options,**kwargs})
				
				data.append(obj)


		# Attributes
		self.measure = measure
		self.data = data


		# Functions
		def func(parameters,state,**kwargs):
			state = [state]*self.N if isinstance(state,arrays) or not isinstance(state,iterables) else state
			state = self.measure.transform(parameters=parameters,state=state,transformation=True)
			for i in range(self.M):
				for data in self.data:
					state = data(parameters=parameters,state=state,**kwargs)
			return state

		def grad(parameters,state):
			return None

		self.func = func
		self.gradient = grad


		# Wrapper
		parameters = self.parameters()
		state = self.state()
		wrapper = partial

		self.func = wrapper(self.func,parameters=parameters,state=state)
		self.gradient = wrapper(self.gradient,parameters=parameters,state=state)

		return

	def __call__(self,parameters=None,state=None,**kwargs):
		'''
		Class function
		Args:
			parameters (array): parameters		
			state (obj): state
			kwargs (dict): class keyword arguments			
		Returns
			out (array): Return of function
		'''

		parameters = self.parameters(parameters) if parameters is not None else self.parameters(self.parameters())
		state = state if state is not None else state

		return self.func(parameters=parameters,state=state,**kwargs)

	def info(self,display=None,ignore=None,verbose=None):
		'''
		Log class information
		Args:
			display (str,iterable[str]): Show attributes
			ignore (str,iterable[str]): Do not show attributes
			verbose (bool,int,str): Verbosity of message			
		'''		

		msg = []
		options = dict(align='<',space=1,width=2)

		display = None if display is None else [display] if isinstance(display,str) else display
		ignore = None if ignore is None else [ignore] if isinstance(ignore,str) else ignore

		for attr in [None,'string','N','M','d','measure','architecture','model','data']:

			obj = attr
			if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
				continue

			if attr is None:
				attr = 'cls'
				substring = str(self)
			else:
				substring = getattrs(self,attr,delimiter=delim,default=None)

			if isinstance(substring,objects):
				string = '%s:\n%s'%(attr,str(substring))
			else:
				string = '%s: %s'%(attr,str(substring))

			msg.append(string)

		msg = [i if isinstance(i,str) else str(i) for i in msg]

		msg = '\n'.join(msg)

		self.log(msg,verbose=verbose)

		return

	def dump(self,path=None):
		'''
		Save class data		
		Args:
			path (str,dict[str,(str,bool)]): Path to dump class data, either path or boolean to dump			
		'''

		# Set path
		path = join(self.path,root=self.cwd) if path is None else path

		# Set data
		data = {}

		# Set callback
		callback = self.callback if self.callback is not None else None

		if callback is not None:
			parameters = self.parameters()
			state = self.state()
			model = self
			data = data
			kwargs = dict()

			status = callback(parameters=parameters,state=state,model=model,data=data,**kwargs)

			dump(data,path)
		
		return

	def load(self,path=None):
		'''
		Load class data		
		Args:
			path (str,dict[str,(str,bool)]): Path to load class data, either path or boolean to load
		'''

		# Set path
		path = join(self.path,root=self.cwd) if path is None else path

		# Set data
		data = load(path)

		return

	def lattices(self,N=None,d=None,lattice=None,system=None):
		'''
		Set lattice attributes
		Args:
			N (int): Size of system
			d (int): Spatial dimension of system
			lattice (str,dict,Lattice): Type of lattice		
			system (dict,System): System attributes (dtype,format,device,backend,architecture,layout,base,unit,options,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)
		'''		

		lattice = self.lattice if lattice is None else lattice

		defaults = dict(
			N = self.N if N is None else N,
			d = self.d if d is None else d,
			lattice = self.lattice if lattice is None else lattice,
			system = self.system if system is None else system,
		)

		if lattice is None:
			lattice = dict(lattice=lattice)
		elif not isinstance(lattice,dict):
			lattice = dict(lattice=lattice)
		else:
			lattice = dict(**lattice)

		setter(lattice,defaults,delimiter=delim,default=False)

		self.lattice = Lattice(**lattice)

		self.N = self.lattice.N
		self.d = self.lattice.d

		return

class Label(Operator):

	N = None
	D = None

	default = 'I'

	def __new__(cls,*args,**kwargs):

		self = super().__new__(cls,*args,**kwargs)

		# Set attributes
		if self.state is None or self.state() is None:
			hermitian = self.hermitian
			unitary = self.unitary
		elif self.state().ndim == 1:
			hermitian = False
			unitary = True
		elif self.state().ndim == 2:
			hermitian = True
			unitary = False

		self.hermitian = hermitian
		self.unitary = unitary


		return self

	def __init__(self,*args,**kwargs):
		return

	def __call__(self,parameters=None,state=None):
		'''
		Call operator
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			data (array): data
		'''

		parameters = self.parameters() if parameters is None and self.parameters is not None else parameters if parameters is not None else None
		state = self.state() if state is None and self.state is not None else state if state is not None else Basis.identity(D=self.n,dtype=self.dtype)

		if state is None:
			return self.func(parameters=parameters,state=state)
		else:
			return self.contract(self.func(parameters=parameters,state=state),state=state)

	def grad(self,parameters=None,state=None):
		'''
		Call operator gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			data (array): data
		'''
		state = self.state() if state is None and self.state is not None else state if state is not None else Basis.identity(D=self.n,dtype=self.dtype)

		if state is None:
			return self.gradient(parameters=parameters,state=state)
		else:
			return self.gradient_contract(self.gradient(parameters=parameters,state=state),self.func(parameters=parameters,state=state),state=state)



class Callback(System):
	def __init__(self,*args,**kwargs):
		'''	
		Class for callback
		Args:
			args (tuple): Class arguments
			kwargs (dict): Class keyword arguments
		'''

		super().__init__(*args,**kwargs)

		self.defaults = {
			'iteration':[],
			'parameters':[],'grad':[],'search':[],
			'value':[],'objective':[],
			'alpha':[],'beta':[],

			'iteration.max':[],'iteration.min':[],
			'parameters.relative':[],'parameters.relative.mean':[],
			'variables':[],
			'variables.relative':[],'variables.relative.mean':[],
			'parameters.norm':[],'grad.norm':[],'search.norm':[],			
			'objective.ideal.noise':[],'objective.diff.noise':[],'objective.rel.noise':[],
			'objective.ideal.state':[],'objective.diff.state':[],'objective.rel.state':[],
			'objective.ideal.operator':[],'objective.diff.operator':[],'objective.rel.operator':[],
			'hessian':[],'fisher':[],
			'hessian.eigenvalues':[],'fisher.eigenvalues':[],
			'hessian.rank':[],'fisher.rank':[],
			'entropy':[],'purity':[],'similarity':[],'divergence':[],

			'N':[],'D':[],'d':[],'M':[],'T':[],'tau':[],'P':[],
			'space':[],'time':[],'lattice':[],'architecture':[],'timestamp':[],

			'noise.string':[],'noise.ndim':[],'noise.locality':[],'noise.method':[],'noise.scale':[],'noise.tau':[],'noise.initialization':[],
			'noise.parameters':[],

			'state.string':[],'state.ndim':[],'label.string':[],'label.ndim':[],

			'hyperparameters.c1':[],'hyperparameters.c2':[],

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
		
		done = ((len(attributes['iteration'])>1) and (attributes['iteration'][-1] == (iterations.stop)))
		
		status = (
			((len(attributes['value']) >= 1) and 
			 (attributes['iteration'][-1] <= max(1,
				hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) or
			(
			(abs(attributes['value'][-1]) > 
				(hyperparameters['eps']['value']*hyperparameters['value']['value'])) and
			(log10(abs(attributes['value'][-1] - attributes['value'][-2])) > 
				(log10(abs(hyperparameters['eps']['value.difference'])))) and
			(norm(attributes['grad'][-1])/(attributes['grad'][-1].size) > 
				  (hyperparameters['eps']['grad']*hyperparameters['value']['grad'])) and
			(norm(attributes['grad'][-1] - attributes['grad'][-2])/(attributes['grad'][-2].size) > 
				  (hyperparameters['eps']['grad.difference']*norm(attributes['grad'][-2])/(attributes['grad'][-2].size)))
			)
			)


		stop = (
			(
			(hyperparameters['eps'].get('value.increase') is not None) and
			(hyperparameters['eps'].get('value.increase') > 0) and
			((len(attributes['value']) > 1) and 
			 (attributes['iteration'][-1] >= max(1,
				hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) and
			((attributes['value'][-1] > attributes['value'][-2]) and
			(log10(attributes['value'][-1] - attributes['value'][-2]) > 
			(log10(hyperparameters['eps']['value.increase']*attributes['value'][-1]))))
			) or
			((iterations.start == iterations.stop))
			)

		none = (iterations.start == 0) and (iterations.stop == 0)

		status = ((status) and (not stop) and (not none))

		other = (
			(len(attributes['iteration']) == 1) or 
			(hyperparameters['modulo']['track'] is None) or 
			((hyperparameters['modulo']['track'] == -1) and (not status)) or
			((hyperparameters['modulo']['track'] > 0) and (attributes['iteration'][-1]%hyperparameters['modulo']['track'] == 0))
			)

		tracking = ((not status) or done or init or other) 

		updates = {
			**{attr: lambda i,attr,track,default: (track[attr][-1]) for attr in ['iteration.max','iteration.min']},
			**{attr: lambda i,attr,track,default: (track[attr][i])
				for attr in [
					'parameters','grad','search']},
			**{attr: lambda i,attr,track,default: (empty(track[attr][-1].shape) if ((i>0) and i<(len(track[attr])-1)) else track[attr][i])
				for attr in [
					'parameters','grad','search',
					'parameters.relative',
					'variables','variables.relative'
					'hessian','fisher',
					'hessian.eigenvalues','fisher.eigenvalues']},
			**{attr: None for attr in [
				'parameters.norm','grad.norm','search.norm',
				]},
			**{attr: lambda i,attr,track,default: (default if ((i>0) and (i<(len(track[attr])-1))) else track[attr][i])
				for attr in [
				'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
				'objective.ideal.state','objective.diff.state','objective.rel.state',
				'objective.ideal.operator','objective.diff.operator','objective.rel.operator',
				'hessian.rank','fisher.rank'
				]
			},
			}

		does = {
			'parameters':True,
			'grad':True,
			'search':True,
			'parameters.relative':True,'parameters.relative.mean':True,		
			'variables':True,'variables.relative':True,'variables.relative.mean':True,
			'entropy':True,'purity':True,'similarity':True,'divergence':True
		}	

		attrs = relsort(track,attributes)
		size = min(len(track[attr]) for attr in track)
		does = {**{attr: False for attr in attrs},**does,**hyperparameters.get('do',{})}


		if tracking:
			
			for attr in attrs:

				if ((hyperparameters['length']['track'] is not None) and 
					(len(track[attr]) > hyperparameters['length']['track'])
					):
					_value = track[attr].pop(0)
				

				index = -1 if ((not stop) or other) else -2
				parameters = attributes['parameters'][index]
				state = metric.state()
			
				if attr in [
					'parameters','grad','search',
					'parameters.relative',
					'variables','variables.relative'
					'hessian','fisher',
					'hessian.eigenvalues','fisher.eigenvalues']:
					default = empty(track[attr][-1].shape) if ((len(track[attr])>0) and isinstance(track[attr][-1],arrays)) else nan
				else:
					default = nan

				do = (not ((status) and (not done) and (not init))) or does[attr]

				value = default

				if attr in attributes:
					value = attributes[attr][index]

				if ((not stop) or other):
					track[attr].append(value)

				if attr in ['iteration.max']:
					value = int(track['iteration'][-1])

				elif attr in ['iteration.min']:
					value = int(track['iteration'][argmin(abs(array(track['objective'],dtype=model.dtype)))])

				elif attr in ['value']:
					value = abs(attributes[attr][index])
				
				elif attr in ['parameters','grad','search'] and (not do):
					value = default

				elif attr in ['parameters','grad','search'] and (do):
					value = attributes[attr][index]

				elif attr in ['parameters.norm','grad.norm','search.norm']:
					value = attr.split(delim)[0]
					value = attributes[value][index]
					value = norm(value)/(value.size)

				elif attr in ['parameters.relative','parameters.relative.mean',
					] and (not do):
					value = default

				elif attr in [
					'parameters.relative','parameters.relative.mean',
					] and (do):
					eps = 1e-20
					if attr in ['parameters.relative']:
						value = parameters
						_value = attributes['parameters'][0]
						value = abs((value-_value)/(_value+eps))
					elif attr in ['parameters.relative.mean']:
						value = parameters
						_value = attributes['parameters'][0]
						value = norm((value-_value)/(_value+eps))/(value.size)

				elif attr in ['variables.relative','variables.relative.mean',
					] and (not do):
					value = default

				elif attr in [
					'variables.relative','variables.relative.mean',
					] and (do):
					eps = 1e-20
					if attr in ['variables.relative']:
						value = model.parameters(parameters)
						value = array([model.data[i].parameters(j) for j in value for i in model.data if model.data[i].variable])
						_value = model.parameters(attributes['parameters'][0])
						_value = array([model.data[i].parameters(j) for j in _value for i in model.data if model.data[i].variable])
						value = abs((value-_value)/(_value+eps))
					elif attr in ['variables.relative.mean']:
						value = model.parameters(parameters)
						value = array([model.data[i].parameters(j) for j in value for i in model.data if model.data[i].variable])
						_value = model.parameters(attributes['parameters'][0])
						_value = array([model.data[i].parameters(j) for j in _value for i in model.data if model.data[i].variable])
						value = norm((value-_value)/(_value+eps))/(value.size)

				elif attr in ['objective']:
					value = abs(metric(model(parameters,state)))
				
				elif attr in [
					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
					'objective.ideal.state','objective.diff.state','objective.rel.state',
					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and ((status) and (not done)):
					value = default


				elif attr in [
					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
					'objective.ideal.state','objective.diff.state','objective.rel.state',
					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and (not ((status) and (not done))):


					defaults = Dictionary(
						data={i: model.data[i] for i in model.data},
						state=metric.state,
						label=metric.label)

					tmp = Dictionary(
						data={i: model.data[i] for i in model.data if (not model.data[i].unitary)},
						state=metric.state,
						label=metric.label)

					if attr in ['objective.ideal.noise','objective.diff.noise','objective.rel.noise']:
						tmp.update(dict(data=tmp.data,state=tmp.state,label=tmp.label))
					elif attr in ['objective.ideal.state','objective.diff.state','objective.rel.state']:						
						tmp.update(dict(data={i:False for i in tmp.data},state=tmp.state,label=tmp.label))
					elif attr in ['objective.ideal.operator','objective.diff.operator','objective.rel.operator']:
						tmp.update(dict(data={i:False for i in tmp.data},state=False,label=tmp.label))
				
					data = tmp.data
					state = metric.state
					label = metric.label

					state.init(data=tmp.state)
					label.init(state=state)

					model.init(data=data,state=state)

					metric.init(model=model,state=state,label=label)

					
					if attr in ['objective.ideal.noise','objective.ideal.state','objective.ideal.operator']:
						value = abs(metric(model(parameters,state)))
					elif attr in ['objective.diff.noise','objective.diff.state','objective.diff.operator']:
						value = abs((track['objective'][-1] - metric(model(parameters,state))))
					elif attr in ['objective.rel.noise','objective.rel.state','objective.rel.operator']:
						value = abs((track['objective'][-1] - metric(model(parameters,state)))/(track['objective'][-1]))


					data = defaults.data
					state = state
					label = label

					state.init(data=default.state)
					label.init(state=state)

					model.init(data=data,state=state)

					metric.init(model=model,state=state,label=label)

					state = state()
					
				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (not do):
					value = default

				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (do):
					
					shape = model.shape if ((metric.state is None) or (metric.state.shape is None)) else metric.state.shape
					grad = model.grad_analytical
					if model.architecture is None:
						wrapper = jit
					elif model.architecture in ['array']:
						wrapper = jit
					elif model.architecture in ['tensor']:
						wrapper = jit											
					elif model.architecture in ['mps']:
						wrapper = partial
					else:
						wrapper = jit

					if attr in ['hessian','hessian.eigenvalues','hessian.rank']:
						function = hessian(wrapper(lambda parameters,state: metric(model(parameters,state))))
					elif attr in ['fisher','fisher.eigenvalues','fisher.rank']:
						function = fisher(model,grad,shapes=(shape,(*parameters.shape,*shape)),hermitian=metric.state.hermitian,unitary=model.unitary)

					if attr in ['hessian','fisher']:
						value = function(parameters,state)

					elif attr in ['hessian.eigenvalues','fisher.eigenvalues']:
						value = sort(abs(eig(function(parameters,state),hermitian=True)))[::-1]
						value = value/maximum(value)
					elif attr in ['hessian.rank','fisher.rank']:
						value = sort(abs(eig(function(parameters,state),hermitian=True)))[::-1]
						value = value/maximum(value)
						value = nonzero(value,eps=1e-12)
						# value = (argmax(abs(difference(value)/value[:-1]))+1) if value.size > 1 else 1

				elif attr in ['entropy'] and (not do):
					value = default

				elif attr in ['entropy'] and (do):
					shape = model.shape if ((metric.state is None) or (metric.state.shape is None)) else metric.state.shape

					function = entropy(model,shape=shape,hermitian=metric.state.hermitian,unitary=model.unitary)

					value = function(parameters,state)

				elif attr in ['purity'] and (not do):
					value = default

				elif attr in ['purity'] and (do):
					shape = model.shape if ((metric.state is None) or (metric.state.shape is None)) else metric.state.shape

					function = purity(model,shape=shape,hermitian=metric.state.hermitian,unitary=model.unitary)

					value = function(parameters,state)

				elif attr in ['similarity'] and (not do):
					value = default

				elif attr in ['similarity'] and (do):
					shape = model.shape if ((metric.state is None) or (metric.state.shape is None)) else metric.state.shape

					function = similarity(model,metric.label,shape=shape,hermitian=metric.state.hermitian,unitary=model.unitary)

					value = function(parameters,state)

				elif attr in ['divergence'] and (not do):
					value = default

				elif attr in ['divergence'] and (do):
					shape = model.shape if ((metric.state is None) or (metric.state.shape is None)) else metric.state.shape

					function = divergence(model,metric.label,shape=shape,hermitian=metric.state.hermitian,unitary=model.unitary)

					value = function(parameters,state)

				elif attr in [
					'state.string','state.ndim',
					'label.string','label.ndim',
					]:
					value = getattrs(metric,attr,default=default,delimiter=delim)

				elif attr in [
					'noise.string','noise.ndim','noise.locality',
					'noise.method','noise.scale','noise.tau','noise.initialization'
					]:
					for i in model.data:
						if model.data[i].string == delim.join(attr.split(delim)[:1]):
							value = getattrs(model.data[i],delim.join(attr.split(delim)[1:]),default=default,delimiter=delim)
							break

				elif attr in ['noise.parameters']:
					for i in model.data:
						if model.data[i].string == delim.join(attr.split(delim)[:1]):
							value = getattrs(model.data[i],delim.join(attr.split(delim)[1:]),default=default,delimiter=delim)
							value = value(value())
							break

				elif attr in []:
					value = [attr.split(delim)[0],delim.join(attr.split(delim)[1:])]
					value = [getattrs(model,i,default=default,delimiter=delim) for i in value]
					value = value[0]/value[1] if value[1] else value[0]

				elif attr not in attributes and hasattrs(optimizer,attr,delimiter=delim):
					value = getattrs(optimizer,attr,default=default,delimiter=delim)

				elif attr not in attributes and hasattrs(model,attr,delimiter=delim):
					value = getattrs(model,attr,default=default,delimiter=delim)

				track[attr][-1] = value

				if (not does[attr]) and (updates.get(attr) is not None):
					for i in range(len(track[attr])):
						track[attr][i] = updates[attr](i,attr,track,default)


		logging = ((len(attributes['iteration']) == 1) or 
			(hyperparameters['modulo']['log'] is None) or 
			(attributes['iteration'][-1]%hyperparameters['modulo']['log'] == 0)
			)

		if logging:

			msg = '\n'.join([
				'%d f(x) = %0.4e'%(
					attributes['iteration'][-1],
					# abs(metric(model(attributes['parameters'][-1],metric.state()))),
					attributes['value'][-1],
				),
				'|x| = %0.4e\t\t|grad(x)| = %0.4e'%(
					norm(attributes['parameters'][-1])/
						 (attributes['parameters'][-1].size),
					norm(attributes['grad'][-1])/
						 (attributes['grad'][-1].size),
				),
				'\t\t'.join([
					'%s = %0.4e'%(attr,attributes[attr][-1])
					for attr in ['alpha','beta']
					if attr in attributes and len(attributes[attr])>0
					]),
				# 'attributes: \t %s'%({attr: attributes[attr][-1].dtype 
				# 	if isinstance(attributes[attr][-1],arrays) else type(attributes[attr][-1]) 
				# 	for attr in attributes}),
				# 'track: \t %s'%({attr: attributes[attr][-1].dtype 
				# 	if isinstance(attributes[attr][-1],arrays) else type(attributes[attr][-1]) 
				# 	for attr in attributes}),				
				# 'x\n%s'%(to_string(parameters.round(4))),
				# 'theta\n%s'%(to_string(model.parameters(parameters).round(4))),
				# 'U\n%s\nV\n%s'%(
				# 	to_string((model(parameters,state)).round(4)),
				# 	to_string((metric.label()).round(4))),
				])


			model.log(msg)

		return status



class Callback(System):
	def __init__(self,*args,attributes={},**kwargs):
		'''	
		Class for callback
		Args:
			attributes (dict): Attributes for callback
			args (tuple): Class arguments
			kwargs (dict): Class keyword arguments
		'''

		if attributes is None:
			attributes = dict()
		elif not isinstance(attributes,dict):
			attributes = {attr:attr for attr in attributes}
		else:
			attributes = {attr:attributes[attr] for attr in attributes}

		setter(kwargs,dict(attributes=attributes),delimiter=delim,default=False)

		super().__init__(*args,**kwargs)

		return

	def __call__(self,parameters,state,model,data,**kwargs):
		''' 
		Callback
		Args:
			parameters (array): parameters
			state (array): state
			model (object): Model instance
			data (dict): data
			kwargs (dict): additional keyword arguments for callback
		Returns:
			status (int): status of callback
		'''

		attributes = self.attributes

		for attr in attributes:

			if attr in ['objective']:
				options = {
					**{attr: model.options[attr] for attr in model.options}
					}
				other = {
					**options,
					**{attr: getattr(self,attr) for attr in options if hasattr(self,attr)},
					**{attr: self.options.get(attr) for attr in options if attr in self.options},
					**{attr: kwargs.get(attr) for attr in kwargs if attr in options},
					}
				value = getattrd(model,attributes[attr])(
					parameters=parameters,
					state=model(parameters,state,**options),
					other=model(parameters,state,**other)
					)

			elif attr in ['noise.parameters']:

				value = [model.model.data[i].parameters() for i in model.model.data if not model.model.data[i].unitary and not model.model.data[i].hermitian]

			elif hasattrs(model,attributes[attr],delimiter=delim):

				value = getattrs(model,attributes[attr],delimiter=delim)

				if callable(value):
					value = value(parameters=parameters,state=state)

			else:

				value = None

			data[attr] = value

		return


def main(settings,*args,**kwargs):

	default = {}
	if settings is None:
		settings = default
	elif isinstance(settings,str):
		settings = load(settings,default=default)

	settings = Dict(settings)

	model = load(settings.cls.model)
	system = settings.system
	model = model(**{**settings.model,**dict(system=system)})

	parameters = model.parameters()

	obj = model(parameters=parameters)

	return


if __name__ == '__main__':

	arguments = 'settings'

	from src.utils import argparser

	args = argparser(arguments)

	main(*args,**args)