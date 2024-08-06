#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

envs = {
	'JAX_DISABLE_JIT':False,
	'JAX_PLATFORMS':'',
	'JAX_PLATFORM_NAME':'',
	'JAX_ENABLE_X64':True,
	'JAX_TRACEBACK_FILTERING':'off',
	'TF_CPP_MIN_LOG_LEVEL':5,
	# "XLA_FLAGS":(
	# 	"--xla_cpu_multi_thread_eigen=false "
	# 	"intra_op_parallelism_threads=1"),
}
for var in envs:
	os.environ[var] = str(envs[var])


from src.utils import array,gradient,rand,seeder,iterables,scalars,integers,floats
from src.io import load,dump
from src.system import Lattice

from functools import partial

import typing
from typing import Any,Dict,Tuple,Sequence,Callable

import jax

configs = {
	'jax_disable_jit':False,
	'jax_platforms':'',
	'jax_platform_name':'',
	'jax_enable_x64': True,
	}
for name in configs:
	jax.config.update(name,configs[name])

import jax.numpy as np

import equinox as eqx
Module = eqx.Module
Partition = partial(eqx.partition,filter_spec=eqx.is_array)

import quimb as qu
import quimb.tensor as qtn

tensor,network,pack,unpack = qtn.Tensor,qtn.MatrixProductState,qtn.pack,qtn.unpack

import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)


class Tensor(object):
	'''
	Tensor Class
	Args:
		shape (int,iterable[int],Dict[iterable[int],int]): Tensor shape of index:size
		name (str): Tensor name
		format (dict): Tensor indices formats
		key (int,Key,iterable[int,Key]): Tensor initialization seed
	'''

	def __init__(self,shape={},name=None,format=None,key=None):
		
		self.shape = shape
		self.name = name
		self.format = format
		self.key = key

		self.init(key)

		return

	@property
	def data(self):
		return self.params				

	def init(self,key,*args,**kwargs):
		'''
		Initialize Class
		Args:
			key (int,Key,iterable[int,Key]): Class initialization seed			
		Returns:
			params (dict): Class params
		'''

		if key is None:
			key = self.key
		shape = self.shape
		name = self.name
		format = self.format

		key = seeder(key)
		key = key()
		shape = shape if isinstance(shape,dict) else {i:size for i,size in enumerate(shape)} if isinstance(shape,iterables) else {None:shape}
		name = str(name)
		format = (
			{**format,**{name:name}} if isinstance(format,dict) else 
			{**{i:format for i in shape},**{name:name}} if format is not None else 
			{**{i:'.'.join(('{}',)*len(i)) if isinstance(i,iterables) else str(i) for i in shape},**{name:name}}
			)

		inds = [format[index].format(*(index if isinstance(index,iterables) else (index,))) for index in shape]
		tags = [format[name].format(name)]
		shape = [shape[index] for index in shape]

		data = rand(shape=shape,key=key)

		data = tensor(data=data,inds=inds,tags=tags)

		self.params = data

		return self.params

	def __call__(self,*args,**kwargs):
		return self.data

	def norm(self,*args,**kwargs):
		'''
		Norm of class data
		Returns:
			norm (scalar): Class tensor norm
		'''
		data = self.get(*args,**kwargs)
		norm = (data.H @ data)**(1/2)
		return norm


class Network(object):
	'''
	Data Class
	Args:
		N (int): System size
		D (int): Physical dimension
		S (int): Virtual dimension
		d (int): Spatial dimension
		structure (str): Tensor type
		lattice (str): Lattice type
		format (dict): Data indices formats		
		key (int,Key,iterable[int,Key]): Tensor initialization seed			
	'''

	def __init__(self,N,D,S,d,structure=None,lattice=None,format=None,key=None):

		self.N = N
		self.D = D
		self.S = S
		self.d = d
		self.structure = structure
		self.lattice = lattice
		self.format = format
		self.key = key

		self.init(key)

		return

	@property
	def data(self):
		return self.get()

	@property
	def backend(self):
		return self.get().backend
	

	def init(self,key,*args,**kwargs):
		'''
		Initialize Class
		Args:
			key (int,Key,iterable[int,Key]): Class initialization seed			
		Returns:
			params (dict): Class data
		'''
		if key is None:
			key = self.key
		N = self.N
		D = self.D
		S = self.S
		d = self.d
		lattice = self.lattice
		format = self.format

		lattice = Lattice(N=N,d=d,lattice=lattice)

		key = seeder(key)
		size = len(lattice)
		keys = key(size,wrapper=lambda keys: dict(zip(lattice,keys)))

		params = {}
		for i in lattice:
			shape = {**{(j,k):S for (j,k) in lattice() if i in (j,k)},**{i:D}}
			name = str(i)
			key = keys[i]
			format = {**{i:'.'.join(('{}',)*len(i)) if isinstance(i,iterables) else str(i) for i in shape},**{name:name}}
			
			index = i
			data = Tensor(shape=shape,name=name,format=format,key=key)

			params[index] = data

		data = [params[index]().data for index in params]
		data = network(data)

		self.set(data)
		self.types = ['params','variables']

		return self.params

	def __call__(self,*args,**kwargs):
		return self.get(*args,**kwargs)

	def __getitem__(self,index):
		return self.get()[index]

	def __len__(self):
		return len(self.params)

	def __iter__(self):
		for index in self.params:
			yield index

	def set(self,data,*args,**kwargs):
		'''
		Set class params,variables
		Args:
			data (tensor): Class data
		'''	
		self.params,self.variables = pack(data)	
		return

	def get(self,*args,params=None,variables=None,**kwargs):
		'''
		Get class data
		Args:
			params (dict[array]): Class params
			variables (tensor): Class variables
		Returns:
			data (tensor): Class data
		'''
		params = self.params if params is None else params
		variables = self.variables if variables is None else variables
		return unpack(params,variables)

	def filter(self,types,*args,**kwargs):
		filters = {'params':True,'variables':False}
		filters = {attr: filters.get(attr) for attr in types}
		return filters

	def norm(self,*args,**kwargs):
		'''
		Norm of class data
		Args:
			data (tensor): Class tensor
		Returns:
			norm (scalar): Class tensor norm
		'''
		data = self.get()
		norm = (data.H @ data)**(1/2)
		return norm

	def normalize(self,method=None,*args,**kwargs):
		'''
		Normalize class data
		Args:
			method (str,int): Normalization method or index where normalization is centred, with positive indexes from left-right and negative indexes from right-left allowed strings in ['left','right']
		'''	

		data = self.get()

		if isinstance(method,integers):
			raise NotImplementedError("Central normalization not Implemented")

		elif isinstance(method,str):
			if method in ['left','right']:
				getattr(data,f'{method}_canonicalize_')(normalize=True)

		self.set(data)

		return

	def canonicalize(self,method=None,*args,**kwargs):
		'''
		Canonicalize class data
		Args:
			method (str,int): Canonicalize method or index where normalization is centred, allowed strings in ['left','right']
		'''	




class Model(Network):
	pass

# class Tensor(object):
# 	'''
# 	Tensor Class
# 	Args:
# 		D (int): Physical dimension
# 		S (int): Virtual dimension
# 		structure (str): Tensor type
# 	'''

# 	def __init__(self,D,S,structure):
		
# 		self.D = D
# 		self.S = S
# 		self.structure = structure
		
# 		self.data = tensor()

# 		return

# 	def init(self,key,index,indices,*args,**kwargs):

# 		inds = [*('%d.%d'%(*sorted((index,i)),) for i in indices),'%d'%(index)]
# 		tags = ['%d'%(index)]
# 		shape = [*[self.S]*len(indices),self.D]

# 		data = rand(shape=shape,key=key)

# 		data = tensor(data=data,inds=inds,tags=tags)

# 		self.data = data

# 		return self.data

# 	def norm(self):
# 		self.data /= (self.data.H @ self.data)**(1/2)
# 		return

# 	def __call__(self,*args,**kwargs):
# 		return self.data

# class Data(Tensor):
# 	'''
# 	Data Class
# 	Args:
# 		N (int): System size
# 		D (int): Physical dimension
# 		S (int): Virtual dimension
# 		d (int): Spatial dimension
# 		structure (str): Tensor type
# 		lattice (str): Lattice type	
# 		args (iterable): Data positional arguments
# 		kwargs (dict): Data keyword arguments
# 	'''
# 	def __init__(self,*args,**kwargs):

# 		super().__init__(*args,**kwargs)

# 		self.args = args
# 		self.kwargs = kwargs

# 		self.lattice = Lattice(N=self.N,d=self.d,lattice=self.lattice)

# 		self.data = network()

# 		return

# 	def init(self,key,*args,**kwargs):
		
# 		keys = seeder(key)(len(self.lattice),wrapper=lambda keys: dict(zip(self.lattice,keys)))
		
# 		cls = self.__class__.__bases__[-1]

# 		for index in self.lattice:

# 			key = keys[index]

# 			indices = self.lattice[index]

# 			data = cls(*self.args,**self.kwargs)

# 			data = data.init(key,index,indices,*args,**kwargs)

# 			self.data &= data

# 		self.norm()

# 		return self.data

# 	def __getitem__(self,index):
# 		return self.data[index]

# 	def __iter__(self):
# 		for data in self.data:
# 			yield data

# class Tensor(Module):
# 	'''
# 	Tensor Class
# 	Args:
# 		shape (int,iterable[int],Dict[iterable[int],int]): Tensor shape of index:size
# 		name (str): Tensor name
# 		key (int,Key,iterable[int,Key]): Tensor initialization seed
# 	'''

# 	params:Dict

# 	def __init__(self,shape={},name=None,key=None):
		
# 		self.init(key,shape,name)

# 		return

# 	def init(self,key,shape,name,*args,**kwargs):
# 		'''
# 		Initialize Class
# 		Args:
# 			key (int,Key,iterable[int,Key]): Class initialization seed			
# 			shape (int,iterable[int],Dict[iterable[int],int]): Tensor shape of index:size
# 			name (str): Tensor name
# 		Returns:
# 			params (dict): Class parameters
# 		'''

# 		key = seeder(key)
# 		key = key()

# 		shape = shape if isinstance(shape,dict) else {i:size for i,size in enumerate(shape)} if isinstance(shape,iterables) else {None:shape}
# 		name = str(name)

# 		inds = ['.'.join(str(i) for i in index) if isinstance(index,iterables) else str(index) for index in shape]
# 		tags = [name]
# 		shape = [shape[index] for index in shape]

# 		data = rand(shape=shape,key=key)

# 		data = tensor(data=data,inds=inds,tags=tags)

# 		self.params = self.pack(data)

# 		return data


# 	def pack(self,data,*args,**kwargs):
# 		return dict(zip(['params','variables'],pack(data)))

# 	def unpack(self,params,variables,*args,**kwargs):
# 		return unpack(params,variables)

# 	def filter(self,*args,**kwargs):
# 		filters = {'params':True,'variables':False}
# 		return filters

# 	def norm(self,data):
# 		'''
# 		Norm of class data
# 		Args:
# 			data (tensor): Class tensor
# 		Returns:
# 			norm (scalar): Class tensor norm
# 		'''
# 		return (data.H @ data)**(1/2)

# 	def __call__(self,*args,**kwargs):
# 		data = self.unpack(**self.params)
# 		data /= self.norm(data)
# 		return data

# class Model(Module):
# 	'''
# 	Data Class
# 	Args:
# 		N (int): System size
# 		D (int): Physical dimension
# 		S (int): Virtual dimension
# 		d (int): Spatial dimension
# 		structure (str): Tensor type
# 		lattice (str): Lattice type
# 		key (int,Key,iterable[int,Key]): Tensor initialization seed			
# 	'''

# 	params:Dict

# 	def __init__(self,N,D,S,d,structure=None,lattice=None,key=None):

# 		self.N = N
# 		self.D = D
# 		self.S = S
# 		self.d = d
# 		self.structure = structure
# 		self.lattice = lattice
# 		self.key = key

# 		self.init(key)

# 		return

# 	def init(self,key,*args,**kwargs):
# 		'''
# 		Initialize Class
# 		Args:
# 			key (int,Key,iterable[int,Key]): Class initialization seed			
# 		Returns:
# 			params (dict): Class parameters
# 		'''
# 		if key is None:
# 			key = self.key
# 		N = self.N
# 		D = self.D
# 		S = self.S
# 		d = self.d
# 		lattice = self.lattice

# 		lattice = Lattice(N=N,d=d,lattice=lattice)

# 		key = seeder(key)
# 		size = len(lattice)
# 		keys = key(size,wrapper=lambda keys: dict(zip(lattice,keys)))

# 		params = {}
# 		for i in lattice:
# 			shape = {**{(j,k):S for (j,k) in lattice() if i in (j,k)},**{i:D}}
# 			name = str(i)
# 			key = keys[i]
# 			data = Tensor(shape=shape,name=name,key=key)

# 			index = str(i)
# 			params[index] = data

# 		self.params = params

# 		return params


# 	def __call__(self,*args,**kwargs):
# 		data = [self.params[index] for index in self.params]
# 		data = network(data)
# 		return data

# 	def __getitem__(self,index):
# 		return self.params[index]

# 	def __len__(self):
# 		return len(self.params)

# 	def __iter__(self):
# 		for index in self.params:
# 			yield index



# class Model(Module):
# 	'''
# 	Model Class
# 	Args:
# 		data (Tensor): Model data
# 	'''
	
# 	data:Any

# 	def setup(self):

# 		data = self.data
# 		params,skeleton = pack(data)
# 		formatter = lambda key: 'param_%d'%(key)

# 		self.params = {key: self.param(formatter(key),lambda *args,**kwargs: params[key]) for key in params}
# 		self.skeleton = skeleton

# 		return

# 	def __call__(self,*args,**kwargs):
# 		data = unpack(self.params, self.skeleton)
# 		data /= (data.H @ data)**(1/2)
# 		return data


class Objective(Module):
	'''
	Objective Module
	Args:
		model (str,Module): Objective model
		label (callable): Objective label
		metric (str,callable): Objective metric, allowed strings in ['fidelity'], or callable with signature metric(model,label) -> func(*args,**kwargs)
		args (iterable): Objective model positional arguments
		kwargs (dict): Objective model keyword arguments
	'''

	model:Callable
	label:Callable
	metric:str|Callable
	args:Dict
	kwargs:Dict

	def setup(self):
		
		if isinstance(self.model,str):
			model = load(self.model,default=model)
		else:
			model = self.model

		self.object = model(*self.args,**self.kwargs)
		
		if callable(self.metric):
			func = self.metric(self.model,self.label)
		elif self.metric in ['fidelity']:
			label = self.label().H
			def func(*args,**kwargs):
				return 1 - (label @ self.object(*args,**kwargs))
		else:
			label = self.label().H			
			def func(*args,**kwargs):
				return 1 - (label @ self.object(*args,**kwargs))

		self.func = func
		
		return

	def __call__(self,*args,**kwargs):
		return self.func(*args,**kwargs)