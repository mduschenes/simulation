#!/usr/bin/env python

# Import python modules
import os,sys

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import progress,forloop,vmap,timestamp,histogram,partial
from src.utils import array,rand,arange,logspace,inplace,transpose,reshape,addition,tensorprod,seeder,permutations
from src.utils import einsum,exp,sqrt,prod,real,imag,abs2,iterables,arrays,scalars,integers,floats,datatype,e,pi,delim
from src.io import load,dump,split,join
from src.run import argparse,setup
from src.iterables import permuter
from src.system import Lattice


# Logging
from src.logger	import Logger
logger = Logger()
verbose = True

class Model(object):

	def __init__(self,
			N=None,D=None,d=None,T=None,
			data=None,parameters=None,
			samples=None,
			model=None,measure=None,
			random=None,seed=None,dtype=None,
			path=None,cwd=None,key=None,
			attributes=None,options=None,
			system=None,
			**kwargs):
		'''
		Model Class
		Args:
			N (int): System size
			D (int): System dimension
			d (int): Spatial dimension
			T (float): System temperature
			samples (int): System samples
			data (dict): System data
			parameters (object): System parameters
			model (str): System model, allowed strings in ['sk','ask','nn','ann','ising']
			measure (str): System measure, allowed strings in ['povm']
			random (dict,str): System randomness, rand options, or allowed strings in ['random','rand','uniform','randint','randn','constant','gaussian','normal','haar','hermitian','symmetric','zero','one','plus','minus','zeros','ones','linspace','logspace']
			seed (int,key): System seed
			dtype (datatype): System datatype
			path (str): System path
			cwd (str): System directory
			key (str): System key
			attributes (dict): System attributes
			options (dict): System options
			system (dict): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		'''

		self.N = N
		self.D = D
		self.d = d
		self.T = T
		self.samples = samples
		self.data = data
		self.parameters = parameters
		self.model = model
		self.measure = measure
		self.random = random
		self.seed = seed
		self.dtype = dtype
		self.path = path
		self.cwd = cwd
		self.key = key
		self.attributes = attributes
		self.options = options

		if system is not None:
			for attr in system:
				setattr(self,attr,system[attr])

		for attr in kwargs:
			setattr(self,attr,kwargs[attr])

		self.setup(*args,**kwargs)

		return
	
	def setup(self,*args,**kwargs):

		for kwarg in kwargs:
			setattr(self,kwarg,kwargs[kwarg])

		self.samples = 0 if not isinstance(self.samples,(*integers,*floats)) else int(self.samples)

		self.random = dict(random=self.random) if not isinstance(self.random,dict) else self.random

		self.options = dict() if not isinstance(self.options,dict) else self.options

		self.attributes = dict(
			N=None,D=None,d=None,T=None,model=None,measure=None,data=None
			) if self.attributes is None else {attr:None for attr in self.attributes} if not isinstance(self.attributes,dict) else self.attributes

		self.index = timestamp() if self.key is None else self.key

		self.data = dict() if not isinstance(self.data,dict) else self.data

		self.initialize()

		self.seed = seeder(self.seed)
		self.parameters = self.init(self.seed)

		msg = {key:getattr(self,key,value) for key,value in self.attributes.items() if isinstance(getattr(self,key,value),scalars)}
		logger.log(verbose,msg)

		return

	def init(self,sample,*args,**kwargs):

		strings = self.shape
		size = len(self.shape)

		key = {string:key for string,key in zip(strings,seeder(sample,size=size))}
		shape = {string:self.shape[string] for string in strings}
		scale = {string:self.parameters.get(string) if isinstance(self.parameters,dict) and isinstance(self.parameters.get(string),scalars) else self.parameters if self.parameters is not None and isinstance(self.parameters,scalars) else None for string in strings}
		dtype = {string:datatype(self.dtype.get(string) if isinstance(self.dtype,dict) else self.dtype if self.dtype is not None else None) for string in strings}

		random = {
			string:{
				**(self.random.get(string) if isinstance(self.random,dict) and all(string in self.random for string in strings) else self.random if self.random is not None else {}),
				**dict(key=key[string],shape=shape[string],scale=scale[string],dtype=dtype[string]),
				}
			for string in strings
			}

		initializer = self.initializer

		parameters = {string:initializer[string](**random[string]) for string in strings}

		return parameters

	def initialize(self):

		if self.model is None:
			def func(parameters,state,*args,**kwargs):
				return -einsum('i,ij,j',state,parameters['J'],state)
		elif self.model in ['sk']:
			def func(parameters,state,*args,**kwargs):
				return -einsum('i,ij,j',state,parameters['J'],state)
		elif self.model in ['ask']:
			def func(parameters,state,*args,**kwargs):
				return einsum('i,ij,j',state,parameters['J'],state)
		elif self.model in ['nn']:
			def func(parameters,state,*args,**kwargs):
				return -einsum('i,ij,j',state,parameters['J'],state)
		elif self.model in ['ann']:
			def func(parameters,state,*args,**kwargs):
				return einsum('i,ij,j',state,parameters['J'],state)								
		elif self.model in ['ising']:
			def func(parameters,state,*args,**kwargs):
				return -einsum('i,ij,j',state,parameters['J'],state) + einsum('i,i',parameters['h'],state)								
		
		if self.measure is None:
			parameters = None
			measure = None
			def measurement(data,*args,**kwargs):
				return data
		elif self.measure in ['povm']:
			parameters = dict(alpha=sqrt(1/(self.D+1)),beta=sqrt(self.D/(self.D+1)),phi=e**(1j*2*pi/(self.D**2-1)))
			measure = tensorprod([array([
				*[[0,1]],
				*[[parameters['alpha'],parameters['beta']*parameters['phi']**i] for i in range(self.D**2-1)],
				])]*self.N)
			def measurement(data,*args,**kwargs):
				data = (1/(self.D**self.N))*abs2(einsum('si,ui->su',sqrt(data),measure))
				return data

		def probability(parameters,state,*args,**kwargs):
			return exp(-(1/self.T)*self.func(parameters,state,*args,**kwargs))
		
		def state(*args,**kwargs):
			
			state = arange(self.D**self.N,dtype=self.dtype)
			sample = seeder(self.seed,size=self.samples)
			shape = self.D**arange(self.N,dtype=self.dtype)
			size = self.D

			def func(sample,index):
				parameters = self.init(sample)
				state = (index//shape)%size
				return self.probability(parameters,state,*args,**kwargs)
			
			data = vmap(vmap(func,in_axes=(None,0)),in_axes=(0,None))(sample,state)
			
			data /= addition(data,-1)[...,None]

			data = self.measurement(data,*args,**kwargs)

			return data

		def sample(data,*args,**kwargs):
			options = {**self.options,**kwargs}
			func = partial(histogram,**options)
			x,y = vmap(func)(data)
			x = addition(x,0)/x.shape[0]
			y = addition(y,0)
			data = x,y
			return data

		initializer = {}
		if self.model is None:
			name = 'J'
			def init(*args,**kwargs):
				parameters = rand(*args,**kwargs)
				parameters = (parameters + transpose(parameters))/2
				return parameters
			initializer[name] = init
		elif self.model in ['sk']:
			name = 'J'
			def init(*args,**kwargs):
				parameters = rand(*args,**kwargs)
				parameters = (parameters + transpose(parameters))/2
				return parameters
			initializer[name] = init			
		elif self.model in ['ask']:
			name = 'J'
			def init(*args,**kwargs):
				parameters = rand(*args,**kwargs)
				parameters = (parameters + transpose(parameters))/2
				return parameters
			initializer[name] = init
		elif self.model in ['nn']:
			lattice = Lattice(N=self.N,d=self.d)
			indices = tuple(zip(*(i for i in lattice('ij') if i not in lattice('<ij>'))))
			name = 'J'
			def init(*args,**kwargs):
				parameters = rand(*args,**kwargs)
				parameters = inplace(parameters,indices,0)
				parameters = (parameters + transpose(parameters))/2
				return parameters
			initializer[name] = init
		elif self.model in ['ising']:
			name = 'J'
			def init(*args,**kwargs):
				parameters = rand(*args,**kwargs)
				parameters = (parameters + transpose(parameters))/2
				return parameters
			initializer[name] = init
			
			name = 'h'
			def init(*args,**kwargs):
				parameters = rand(*args,**kwargs)
				return parameters
			initializer[name] = init			

		self.func = func
		self.initializer = initializer
		self.measurement = measurement
		self.probability = probability
		self.state = state
		self.sample = sample

		return
		
	@property
	def shape(self):
		if self.model is None:
			shape = dict(J=(self.N,self.N)) if self.N else None
		elif self.model in ['sk']:
			shape = dict(J=(self.N,self.N)) if self.N else None
		elif self.model in ['ask']:
			shape = dict(J=(self.N,self.N)) if self.N else None
		elif self.model in ['nn']:
			shape = dict(J=(self.N,self.N)) if self.N else None
		elif self.model in ['ann']:
			shape = dict(J=(self.N,self.N)) if self.N else None					
		elif self.model in ['ising']:
			shape = dict(J=(self.N,self.N),h=(self.N,)) if self.N else None
		shape = {string:shape.get(string) for string in self.parameters} if self.parameters is not None and shape is not None else shape if shape is not None else None
		return shape

	@property
	def size(self):
		return {string:prod(self.shape[string]) for string in self.shape} if self.shape else None

	@property
	def ndim(self):
		return {string:len(self.shape[string]) for string in self.shape} if self.shape else None

	def load(self,path=None):
		path = join(self.path,root=self.cwd) if path is None else path
		data = self.data
		options = dict(default=data,lock=True,backup=False)
		data = load(path,**options)
		self.data = data
		return

	def dump(self,path=None):
		path = join(self.path,root=self.cwd) if path is None else path
		data = self.data
		options = dict(lock=True,backup=False)
		dump(data,path,**options)
		return

	def append(self,*args,**kwargs):

		def value(index,attr):
			data = kwargs.get(attr,getattr(self,attr,None))
			data = [data] if isinstance(data,arrays) and not isinstance(data,scalars) else data
			return data

		data = self.data
		index = self.index

		if index not in data:
			data[index] = {attr: value(index,attr) for attr in kwargs}

		return

	def run(self,*args,**kwargs):

		index = self.index
		if index in self.data:
			data = self.data[index]
			return data

		state = self.state(*args,**kwargs)

		data = {}
		for attr in self.attributes:
			
			key = attr
			value = self.attributes[attr]

			if attr in ['data']:

				value = state

			elif attr in ['sample.linear','sample.log']:

				options = {
					**({
					'sample.linear':dict(bins=1000,scale='linear',base=10,range=[0,1]),
					'sample.log':dict(bins=1000,scale='log',base=10,range=[1e-20,1e0]),
						}.get(attr,{})),
					**self.options
					}

				key = [f'{attr}.{i}' for i in ['x','y']]

				value = self.sample(state,**options)

			elif hasattr(self,attr):

				value = getattr(self,attr)

			else:
				continue

			if isinstance(key,str):

				key = [key]
				value = [value]

			for key,value in zip(key,value):

				data[key] = value

		return data

	def __str__(self):
		name = self.__class__.__name__
		string = ' , '.join([f'{attr}:{getattr(self,attr)}' for attr in self.attributes]) if self.attributes is not None else ''
		return f'{name} {string}'

	def __repr__(self):
		return str(self)

	def __call__(self,*args,**kwargs):

		data = self.run(*args,**kwargs)

		self.append(**data)

		return


def main(settings,*args,**kwargs):

	settings = setup(settings,*args,**kwargs)

	boolean = settings.boolean
	options = settings.options
	model = settings.model
	system = settings.system

	model = Model(**{**model,**system})

	if boolean.load:
		model.load()

	if boolean.call:
		model(**options)

	if boolean.dump:
		model.dump()

	return

if __name__ == '__main__':

	args = argparse()

	main(*args,**args)
