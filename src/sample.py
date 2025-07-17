#!/usr/bin/env python

# Import python modules
import os,sys

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import progress,forloop,vmap,timestamp
from src.utils import array,rand,arange,logspace,transpose,reshape,addition,tensorprod,seeder,permutations
from src.utils import einsum,exp,sqrt,prod,real,imag,abs2,iterables,arrays,scalars,integers,floats,datatype,e,pi,delim
from src.io import load,dump,split,join
from src.run import argparse,setup
from src.iterables import permuter


# Logging
from src.logger	import Logger
logger = Logger()
verbose = True


class Model(object):

	def __init__(self,N=None,D=None,d=None,T=None,data=None,parameters=None,samples=None,model=None,measure=None,random=None,seed=None,dtype=None,path=None,cwd=None,key=None,system=None,**kwargs):
		'''
		Model Class
		Args:
			N (int): System size
			D (int): System dimension
			d (int): Spatial dimension
			T (float): System temperature
			data (dict): System data
			parameters (object): System parameters
			samples (int): System samples
			model (str): System model, allowed strings in ['sk','ask','ising']
			measure (str): System measure, allowed strings in ['povm']
			random (str): System randomness, allowed strings in ['random','rand','uniform','randint','randn','constant','gaussian','normal','haar','hermitian','symmetric','zero','one','plus','minus','zeros','ones','linspace','logspace']
			seed (int,key): System seed
			dtype (datatype): System datatype
			path (str): System path
			cwd (str): System directory
			key (str): System key
			system (dict): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		'''

		self.N = N
		self.D = D
		self.d = d
		self.T = T
		self.data = data
		self.parameters = parameters
		self.samples = samples
		self.model = model
		self.measure = measure
		self.random = random
		self.seed = seed
		self.dtype = dtype
		self.path = path
		self.cwd = cwd
		self.key = key

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

		self.samples = int(self.samples) if isinstance(self.samples,(*integers,*floats)) else 0

		self.attributes = dict(N=None,D=None,d=None,T=None,model=None,measure=None,random=None,data=None)

		self.index = self.key if self.key is not None else timestamp()

		self.data = self.data if isinstance(self.data,dict) else {}

		if self.model is None:
			def func(parameters,state,*args,**kwargs):
				return -einsum('i,ij,j',state,parameters['J'],state)
		elif self.model in ['sk']:
			def func(parameters,state,*args,**kwargs):
				return -einsum('i,ij,j',state,parameters['J'],state)
		elif self.model in ['ask']:
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
				data = (1/self.D**self.N)*abs2(einsum('si,ui->su',data,measure))
				return data

		def probability(parameters,state,*args,**kwargs):
			return exp(-(1/self.T)*self.func(parameters,state,*args,**kwargs))
		
		def sample(*args,**kwargs):
			
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

		self.seed = seeder(self.seed)
		self.parameters = self.init(self.seed)

		self.func = func
		self.measurement = measurement
		self.probability = probability
		self.sample = sample

		msg = {key:getattr(self,key,value) for key,value in self.attributes.items() if isinstance(getattr(self,key,value),scalars)}
		logger.log(verbose,msg)

		return

	def init(self,sample,*args,**kwargs):

		shape = self.shape
		key = dict(zip(self.shape,seeder(sample,size=len(self.shape))))
		random = {string:self.random.get(string) if isinstance(self.random,dict) else self.random if self.random is not None else None for string in self.shape}
		scale = {string:self.parameters.get(string) if isinstance(self.parameters,dict) and isinstance(self.parameters.get(string),scalars) else self.parameters if self.parameters is not None and isinstance(self.parameters,scalars) else None for string in self.shape}
		dtype = {string:datatype(self.dtype.get(string) if isinstance(self.dtype,dict) else self.dtype if self.dtype is not None else None) for string in self.shape}

		parameters = {string:rand(key=key[string],shape=shape[string],random=random[string],scale=scale[string],dtype=dtype[string]) for string in shape}
		
		return parameters
		
	@property
	def shape(self):
		if self.model is None:
			shape = dict(J=(self.N,self.N)) if self.N else None
		elif self.model in ['sk']:
			shape = dict(J=(self.N,self.N)) if self.N else None
		elif self.model in ['ask']:
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

		def value(index,attr,self,*args,**kwargs):
			data = kwargs.get(attr,getattr(self,attr,None))
			data = [data] if isinstance(data,arrays) and not isinstance(data,scalars) else data
			return data

		data = self.data
		index = self.index

		if index not in data:
			data[index] = {attr: value(index,attr,self,*args,**kwargs) for attr in self.attributes}

		return

	def run(self,*args,**kwargs):

		index = self.index
		if index in self.data:
			data = self.data[index]
			return data

		data = dict(data=self.sample(*args,**kwargs))

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
