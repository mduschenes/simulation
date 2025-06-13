#!/usr/bin/env python

# Import python modules
import os,sys

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.utils import array,rand,seeder,permutations
from src.utils import einsum,exp,sqrt,prod,iterables
from src.io import load,dump,split,join
from src.iterables import permuter,Dict

# Logging
from src.logger	import Logger
logger = Logger()


class Model(object):

	def __init__(self,N=None,D=None,d=None,T=None,data=None,parameters=None,samples=None,random=None,seed=None,dtype=None):
		
		self.N = N
		self.D = D
		self.d = d
		self.T = T
		self.data = data
		self.parameters = parameters
		self.samples = samples
		self.random = random
		self.seed = seed
		self.dtype = dtype

		self.setup()
		
		return
	
	def setup(self,*args,**kwargs):

		for kwarg in kwargs:
			setattr(self,kwarg,kwargs[kwarg])

		samples = int(self.samples) if isinstance(self.samples,(int,float)) else 0
		self.samples = samples

		data = {attr:[] if not isinstance(self.data[attr],list) else self.data[attr] for attr in self.data} if isinstance(self.data,dict) else {attr:[] for attr in self.data} if self.data is not None else {}
		self.data = data

		def states(*args,**kwargs):
			yield from (array(i) for i in permutations(range(-self.D//2,self.D//2),repeat=self.N))
		def func(state,**kwargs):
			return -(1/sqrt(self.N))*einsum('i,ij,j',state,self.parameters,state)
		def weight(state,*args,**kwargs):
			return exp(-(1/self.T)*func(state,**kwargs))
		def normalization(*args,**kwargs):
			return sum(weight(state,*args,**kwargs) for state in states(*args,**kwargs))
		def probability(*args,**kwargs):
			return array([weight(state,*args,**kwargs) for state in states(*args,**kwargs)])/normalization(*args,**kwargs)
	 
		self.states = states
		self.func = func
		self.weight = weight
		self.normalization = normalization
		self.probability = probability

		return

	def init(self,*args,**kwargs):

		self.setup(*args,**kwargs)

		shape = self.shape
		key = seeder(self.seed)
		dtype = self.dtype
		parameters = rand(key=key,shape=shape,dtype=dtype)
		self.parameters = parameters

		return
		
	@property
	def shape(self):
		return self.parameters.shape if self.parameters is not None else (self.N,self.N)

	@property
	def size(self):
		return self.parameters.size if self.parameters is not None else prod(self.shape)

	@property
	def ndim(self):
		return self.parameters.ndim if self.parameters is not None else len(self.shape)

	def load(self,path):
		data = self.data
		data = load(path,default=data)
		self.data = data
		return

	def dump(self,path):
		data = self.data
		dump(data,path)
		return

	def append(self,*args,**kwargs):
		for attr in self.data:
			data = kwargs.get(attr,getattr(self,attr))
			self.data[attr].append(data)
		return

	def run(self,*args,**kwargs):
		data = self.probability(*args,**kwargs)
		return data

	def __str__(self):
		name = self.__class__.__name__
		string = ' , '.join([f'{attr}:{self.data[attr]}' for attr in self.data]) if self.data is not None else ''
		return f'{name} {string}'

	def __repr__(self):
		return str(self)

	def __call__(self,*args,**kwargs):

		for seed in range(self.samples):
			print(seed)
			self.init(seed=seed,**kwargs)
			data = self.run(*args,**kwargs)
			self.append(data=data)

		return

def main(settings,*args,**kwargs):

	settings = load(settings,wrapper=Dict)

	boolean = settings.boolean
	permutations = settings.permutations
	model = settings.model
	path = join(settings.system.data,root=settings.system.path)

	model = Model(**model)

	if boolean.load:
		model.load(path)

	for options in permuter(permutations):
		model(**options)

	if boolean.dump:
		model.dump(path)

	return

if __name__ == '__main__':

	arguments = {
		'--settings':{
			'help':'Settings',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		}		

	wrappers = {
		}

	args = argparser(arguments,wrappers)

	main(*args,**args)
