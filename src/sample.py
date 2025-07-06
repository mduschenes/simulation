#!/usr/bin/env python

# Import python modules
import os,sys

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import progress
from src.utils import array,rand,seeder,permutations
from src.utils import einsum,exp,sqrt,prod,iterables,scalars,integers,floats,delim
from src.io import load,dump,split,join
from src.run import argparse,setup
from src.iterables import permuter


# Logging
from src.logger	import Logger
logger = Logger()
verbose = True


class Model(object):

	def __init__(self,N=None,D=None,d=None,T=None,data=None,parameters=None,samples=None,model=None,random=None,seed=None,dtype=None,path=None,cwd=None,key=None,system=None,**kwargs):
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

		self.setup()

		msg = {key:getattr(self,key,value) for key,value in self.attributes.items()}
		logger.log(verbose,msg)
		
		return
	
	def setup(self,*args,**kwargs):

		for kwarg in kwargs:
			setattr(self,kwarg,kwargs[kwarg])

		samples = int(self.samples) if isinstance(self.samples,(*integers,*floats)) else 0
		self.samples = samples

		data = self.data if isinstance(self.data,dict) else {}
		self.data = data

		self.attributes = dict(N=None,D=None,d=None,T=None,model=None,random=None,seed=None,data=None)

		if self.model is None:
			def func(state,**kwargs):
				return -einsum('i,ij,j',state,self.parameters['J'],state)
		elif self.model in ['sk']:
			def func(state,**kwargs):
				return -einsum('i,ij,j',state,self.parameters['J'],state)
		elif self.model in ['ask']:
			def func(state,**kwargs):
				return einsum('i,ij,j',state,self.parameters['J'],state)
		elif self.model in ['ising']:
			def func(state,**kwargs):
				return -einsum('i,ij,j',state,self.parameters['J'],state) + einsum('i,i',self.parameters['h'],state)								
		def states(*args,**kwargs):
			yield from (array(i) for i in permutations(range(-self.D//2,self.D//2),repeat=self.N))
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
		key = dict(zip(self.shape,seeder(self.seed,len(self.shape))))		
		random = {string:self.random.get(string) if isinstance(self.random,dict) else self.random if self.random is not None else None for string in self.shape}
		scale = {string:self.parameters.get(string) if isinstance(self.parameters,dict) else self.parameters if self.parameters is not None else None for string in self.shape}
		dtype = {string:self.dtype.get(string) if isinstance(self.dtype,dict) else self.dtype if self.dtype is not None else None for string in self.shape}

		parameters = {string:rand(key=key[string],shape=shape[string],random=random[string],scale=scale[string],dtype=dtype[string]) for string in shape}
		
		self.parameters = parameters

		return
		
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
		options = dict(default=data,lock=True,backup=True)
		data = load(path,**options)
		self.data = data
		return

	def dump(self,path=None):
		path = join(self.path,root=self.cwd) if path is None else path
		data = self.data
		options = dict(lock=True,backup=True)
		dump(data,path,**options)
		return

	def append(self,*args,**kwargs):

		def func(data):

			def key(index,attr,self,*args,**kwargs):
				boolean = value(index,attr,self,*args,**kwargs) in data[index].get(attr,[])
				return boolean

			def value(index,attr,self,*args,**kwargs):
				data = kwargs.get(attr,getattr(self,attr,None))
				data = [data] if not isinstance(data,scalars) else data
				return data

			# index = None
			# for index in data:
			# 	if all(key(index,attr,self,*args,**kwargs) for attr in self.attributes if (attr not in dict(data=None,seed=None))):
			# 		break
			# 	else:
			# 		index = None

			index = self.key

			if index is None:
				index = str(len(data))

			if index not in data:
				data[index] = {}

			for attr in self.attributes:
				if attr not in data[index]:
					data[index][attr] = []
				elif not isinstance(data[index][attr],list):
					data[index][attr] = data[index][attr].tolist()
				data[index][attr].append(value(index,attr,self,*args,**kwargs))


			# path = join(self.path,root=self.cwd)
			# options = dict(func=None,lock=None,backup=None)
			# dump(data,path)
			# print('-------',path,index,max(len(data[index][attr]) for attr in data[index]))

			return data


		data = self.data
		func(data)

		# path = join(self.path,root=self.cwd)
		# options = dict(func=func,lock=True,backup=True)	
		# data = load(path,**options)

		return

	def run(self,*args,**kwargs):
		data = self.probability(*args,**kwargs)
		return data

	def __str__(self):
		name = self.__class__.__name__
		string = ' , '.join([f'{attr}:{getattr(self,attr)}' for attr in self.attributes]) if self.attributes is not None else ''
		return f'{name} {string}'

	def __repr__(self):
		return str(self)

	def __call__(self,*args,**kwargs):
		
		def options(self,sample,args,kwargs):
			arguments = args
			keywords = {**kwargs,**dict(seed=self.seed if isinstance(self.seed,iterables) else sample)}
			return arguments,keywords
		
		for sample in progress(range(self.samples)):
			arguments,keywords = options(self,sample,args,kwargs)
			self.init(*arguments,**keywords)
			data = self.run(*arguments,**keywords)
			self.append(data=data)

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

	for options in permuter(options):
		model(**options)

	if boolean.dump:
		model.dump()

	return

if __name__ == '__main__':

	args = argparse()

	main(*args,**args)
