#!/usr/bin/env python

# Import python modules
import os,sys

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,progressbar
from src.utils import array,rand,seeder,permutations
from src.utils import einsum,exp,sqrt,prod,iterables,scalars
from src.io import load,dump,split,join
from src.iterables import permuter,Dict

# Logging
from src.logger	import Logger
logger = Logger()
verbose = True


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

		data = self.data if isinstance(self.data,dict) else {None:{attr:None for attr in self.data}} if self.data is not None else {}
		self.data = data

		def states(*args,**kwargs):
			yield from (array(i) for i in permutations(range(-self.D//2,self.D//2),repeat=self.N))
		def func(state,**kwargs):
			return einsum('i,ij,j',state,self.parameters,state)
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

		key = seeder(self.seed)		
		shape = self.shape
		random = self.random
		dtype = self.dtype
		parameters = rand(key=key,shape=shape,random=random,dtype=dtype)
		self.parameters = parameters

		return
		
	@property
	def shape(self):
		return (self.N,self.N) if self.N else None

	@property
	def size(self):
		return prod(self.shape) if self.shape else None

	@property
	def ndim(self):
		return len(self.shape) if self.shape else None

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
		if all(i is None for i in self.data):
			attrs = self.data.pop(None)
		elif self.data:
			attrs = self.data[list(self.data)[-1]]
		else:
			attrs = []

		index = str(len(self.data))
		self.data[index] = {}

		for attr in attrs:
			data = kwargs.get(attr,getattr(self,attr))
			data = [data] if not isinstance(data,scalars) else data
			self.data[index][attr] = data

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
		for sample in progressbar(range(self.samples)):
			kwargs = {**kwargs,**dict(seed=sample)}
			self.init(**kwargs)
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
		logger.log(verbose,options)
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
