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


from src.utils import array,gradient,rand,seeder
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

import flax
import flax.linen as nn
from flax.linen import Module

import quimb as qu
import quimb.tensor as qtn

tensor,network,pack,unpack = qtn.Tensor,qtn.TensorNetwork,qtn.pack,qtn.unpack

import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)


class Base(object):
	'''
	Base Class
	Args:
		N (int): System size
		D (int): Physical dimension
		S (int): Virtual dimension
		d (int): Spatial dimension
		structure (str): Tensor type
		lattice (str): Lattice type
	'''
	def __init__(self,N,D,S,d,structure=None,lattice=None):
		
		self.N = N
		self.D = D
		self.S = S
		self.d = d
		self.structure = structure
		self.lattice = Lattice(N=N,d=d,lattice=lattice)
		
		self.data = tensor()

		return

	def init(self,key,index,indices,*args,**kwargs):

		inds = ['%d'%(index),*('%d.%d'%(*sorted((index,i)),) for i in indices)]
		tags = ['%d'%(index)]
		shape = [self.D,*[self.S]*len(indices)]

		data = rand(shape=shape,key=key)

		data = tensor(data=data,inds=inds,tags=tags)

		self.data = data

		return self.data

	def norm(self):
		self.data /= (self.data.H @ self.data)**(1/2)
		return

class Data(Base):
	'''
	Data Class
	Args:
		args (iterable): Data positional arguments
		kwargs (dict): Data keyword arguments
	'''
	def __init__(self,*args,**kwargs):

		super().__init__(*args,**kwargs)

		self.args = args
		self.kwargs = kwargs
		self.data = network()

		return

	def init(self,key,*args,**kwargs):
		
		keys = seeder(key)(len(self.lattice),wrapper=lambda keys: dict(zip(self.lattice,keys)))
		
		cls = self.__class__.__bases__[-1]

		for index in self.lattice:

			key = keys[index]

			indices = self.lattice[index]

			data = cls(*self.args,**self.kwargs)

			data = data.init(key,index,indices,*args,**kwargs)

			self.data &= data

		self.norm()

		return self.data

	def norm(self):
		self.data /= (self.data.H @ self.data)**(1/2)
		return


class Model(Module):
	'''
	Tensor Class
	Args:
		data (tensor): Tensor data
	'''
	
	data:Any

	def setup(self):

		data = self.data
		params,skeleton = pack(data)
		formatter = lambda key: 'param_%d'%(key)

		self.params = {key: self.param(formatter(key),lambda *args,**kwargs: params[key]) for key in params}
		self.skeleton = skeleton

		return

	def __call__(self,*args,**kwargs):
		data = unpack(self.params, self.skeleton)
		data /= (data.H @ data)**(1/2)
		return data


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