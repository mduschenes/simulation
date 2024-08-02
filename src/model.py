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


from src.utils import gradient,rand
from src.io import load,dump


from functools import partial

import typing
from typing import Dict,Tuple,Sequence,Callable

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

import optax

from tqdm import tqdm as progressbar

import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)
	



class Model(nn.Module):
	'''
	Model Class
	Args:
		shape (iterable[int]): Shape of model
	'''
	
	shape: Sequence[int]

	def setup(self):
		self.layers = [nn.Dense(shape) for shape in self.shape]
		return

	def __call__(self, x):
		for i, layer in enumerate(self.layers):
			x = layer(x)
			if i != len(self.layers) - 1:
				x = nn.relu(x)
		return x


class Data(object):
	'''
	Data Class
	Args:
		shape (iterable[int]): Shape of data
		random (str): Type of random distribution, allowed strings in ['random','rand','uniform','random','randint','randn','gaussian','normal','haar','hermitian','symmetric','zero','one','plus','minus','zeros','ones','linspace','logspace']
	'''

	def __init__(self,shape=None,random=None):

		self.shape = shape
		self.random = random

		self.data = None

		return

	def init(self,key,*args,**kwargs):

		self.data = rand(shape=self.shape,key=key,*args,**kwargs)

		return

	def __call__(self,*args,**kwargs):
		return self.data


class Objective(nn.Module):
	'''
	Objective Module
	Args:
		model (str,nn.Module): Objective model
		label (callable): Objective label
		metric (str,callable): Objective metric, allowed strings in ['mse'], or callable with signature metric(model,label) -> func(x)
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
		elif self.metric in ['mse']:
			def func(x):
				return ((self.object(x)-self.label(x))**2).sum()
		else:
			def func(x):
				return ((self.object(x)-self.label(x))**2).sum()
		
		self.func = func
		
		return

	def __call__(self,x):
		return self.func(x)


class Label(object):
	'''
	Label Class
	Args:
		label (str,callable): Objective label, allowed strings in ['model'], or callable with signature label(model) -> func(x)
		model (str,nn.Module): Objective model
		args (iterable): Objective model positional arguments
		kwargs (dict): Objective model keyword arguments
		init (dict): Objective model init keyword arguments
	'''

	def __init__(self,label=None,model=None,args=(),kwargs={},init={}):
		
		self.label = label
		self.args = args
		self.kwargs = kwargs
		self.init = init

		if isinstance(model,str):
			model = load(model,default=model)
		else:
			model = model
		
		self.model = model(*args,**kwargs)
		params = self.model.init(**init)
		func = self.model.apply
		self.model = partial(func,params)
		
		if callable(label):
			func = label(self.model)
		elif label in ['model']:
			def func(x):
				return self.model(x)
		else:
			def func(x):
				return self.model(x)
		self.func = func

		return

	def __call__(self,x):
		return self.func(x)


class Optimizer(object):
	'''
	Optimizer Class
	Args:
		optimizer (str): Optimizer type, allowed strings in ['adam']
		iterations (int,float,iterable,range): Optimizer iterations
		progress (bool,str): Optimizer progressbar, allowed strings in ['progressbar']
		objective (callable,nn.Module): Optimizer objective, overrides func and grad arguments
		func (callable): Optimizer objective function
		grad (callable): Optimizer objective gradient
		args (iterable): Optimizer positional arguments
		kwargs (dict): Optimizer keyword arguments
	'''
	def __init__(self,optimizer=None,iterations=None,progress=None,objective=None,func=None,grad=None,*args,**kwargs):

		if optimizer is None:
			optimizer = getattr(optax,'adam')
		elif optimizer in ['adam']:
			optimizer = getattr(optax,optimizer)
		else:
			optimizer = getattr(optax,'adam')

		if iterations is None:
			iterations = range()
		elif isinstance(iterations,(int,float)):
			iterations = range(int(iterations))
		else:
			iterations = iterations

		if progress is None:
			def progress(iterable):
				return iterable
		elif progress is True:
			progress = progressbar
		elif progress in ['progressbar']:
			progress = progressbar			
		else:
			def progress(iterable):
				return iterable

		if objective is not None:
			try:
				func = objective.apply
			except:
				func = objective
			try:
				grad = objective.grad
			except:
				grad = grad
		
		if grad is None:
			grad = gradient(func)

		self.optimizer = optimizer(*args,**kwargs)
		self.iterations = iterations
		self.progress = progress

		self.func = func
		self.grad = grad

		self.opt = None

		return

	def init(self,params,*args,**kwargs):
		self.opt = self.optimizer.init(params)
		return

	def update(self,grads):
		updates,self.opt = self.optimizer.update(grads,self.opt)
		return updates

	def apply(self,params,updates,*args,**kwargs):
		params = optax.apply_updates(params,updates)
		return params

	@partial(jax.jit, static_argnums=(0,))
	def step(self,params,*args,**kwargs):
		grads = self.grad(params,*args,**kwargs)
		updates = self.update(grads)
		params = self.apply(params,updates)
		return params

	def __call__(self,params,*args,**kwargs):
		
		self.init(params)

		for iteration in self.progress(self.iterations):
			params = self.step(params,*args,**kwargs)

		return params

