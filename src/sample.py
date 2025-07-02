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
from src.utils import einsum,exp,sqrt,prod,iterables,scalars,integers,floats,delim
from src.io import load,dump,split,join
from src.run import iterate
from src.iterables import Dict,namespace,setter,getter,permuter


# Logging
from src.logger	import Logger
logger = Logger()
verbose = True


class Model(object):

	def __init__(self,N=None,D=None,d=None,T=None,data=None,parameters=None,samples=None,random=None,seed=None,dtype=None,path=None,key=None,system=None,**kwargs):
		
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
		self.path = path
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

		self.attributes = dict(N=None,D=None,d=None,T=None,random=None,seed=None,data=None)

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

	def load(self,path=None):
		path = self.path if path is None else path
		data = self.data
		options = dict(default=data,lock=True,backup=True)
		data = load(path,**options)
		self.data = data
		return

	def dump(self,path=None):
		path = self.path if path is None else path
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


			# path = self.path
			# options = dict(func=None,lock=None,backup=None)
			# dump(data,path)
			# print('-------',path,index,max(len(data[index][attr]) for attr in data[index]))

			return data


		data = self.data
		func(data)

		# path = self.path
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
		
		for sample in progressbar(range(self.samples)):
			arguments,keywords = options(self,sample,args,kwargs)
			self.init(*arguments,**keywords)
			data = self.run(*arguments,**keywords)
			self.append(data=data)

		return


def setup(settings,*args,index=None,device=None,job=None,path=None,env=None,execute=None,verbose=None,**kwargs):
	'''
	Setup settings
	Args:
		settings (dict,str): settings
		index (int): settings index
		device (str): settings device
		job (str): settings job
		path (str): settings path
		env (int): settings environment
		execute (bool,int): settings execution
		verbose (int,str,bool): settings verbosity
		args (iterable): settings positional arguments
		kwargs (dict): settings keyword arguments
	Returns:
		settings (dict): settings
	'''

	default = {}
	wrapper = Dict
	defaults = Dict(
		boolean=dict(call=None,optimize=None,load=None,dump=None),
		cls=dict(module=None,model=None,state=None,label=None,callback=None),
		module=dict(),model=dict(),state=dict(),label=dict(),callback=dict(),
		optimize=dict(),seed=dict(),system=dict(),
		)

	if settings is None:
		settings = default
	elif isinstance(settings,str):
		settings = load(settings,default=default,wrapper=wrapper)

	setter(settings,kwargs,delimiter=delim,default=True)
	setter(settings,defaults,delimiter=delim,default=False)

	if index is not None:
		for key,setting in iterate(settings,index=index,wrapper=wrapper):
			settings = setting
			break

	return settings


def main(settings,*args,**kwargs):

	settings = setup(settings,*args,**kwargs)

	boolean = settings.boolean
	options = settings.options
	model = settings.model
	
	system = {**settings.system,**dict(path=join(settings.system.model,root=settings.system.path),key=settings.system.key)}

	model = Model(**{**model,**system})

	if boolean.load:
		model.load()

	for options in permuter(options):
		model(**options)

	if boolean.dump:
		model.dump()

	return

if __name__ == '__main__':

	arguments = {
		'--settings':{
			'help':'Settings',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--index':{
			'help':'Index',
			'type':int,
			'default':None,
			'nargs':'?'
		},		
		}		

	wrappers = {
		}

	args = argparser(arguments,wrappers)

	main(*args,**args)
