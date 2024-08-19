#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,vmap,seeder,nester,partial,gradient,einsum,tensorprod,conjugate,allclose
from src.utils import array,zeros,ones,empty
from src.iterables import permutations
from src.io import load,dump
from src.system import Dict

from src.tensor import Model

from src.train import train

def main(settings,*args,**kwargs):

	# Settings
	default = {}
	settings = load(settings,default=default,wrapper=lambda data: Dict(nester(data,keys='cls',func=load)))


	# System
	system = settings['system']
	seed = system['seed']
	shape = len(system['attrs'])
	wrapper = lambda keys: dict(zip(system['attrs'],keys))
	key = seeder(seed=seed)(shape,wrapper=wrapper)


	# Data
	data = settings['data']['cls'](*settings['data']['args'],**settings['data']['kwargs'])
	tensor = data.init(key['data'])

	print(tensor)

	return



def main(settings,*args,**kwargs):

	# Settings
	default = {}
	settings = load(settings,default=default,wrapper=lambda data: Dict(nester(data,keys='cls',func=load)))

	# System
	system = settings['system']
	seed = system['seed']
	shape = len(system['attrs'])
	wrapper = lambda keys: dict(zip(system['attrs'],keys))
	key = seeder(seed=seed)(shape,wrapper=wrapper)

	# Model
	model = settings['model']['cls'](*settings['model']['args'],**settings['model']['kwargs'])
	params = model.init(key['model'])

	print(params)

	return


def main(settings,*args,**kwargs):

	# Settings
	default = {}
	settings = load(settings,default=default,wrapper=lambda data: Dict(nester(data,keys='cls',func=load)))

	# System
	system = settings['system']
	seed = system['seed']
	shape = len(system['attrs'])
	wrapper = lambda keys: dict(zip(system['attrs'],keys))
	key = seeder(seed=seed)(shape,wrapper=wrapper)

	# Model
	model = settings['model']['cls'](*settings['model']['args'],**settings['model']['kwargs'])
	params = model.init(key['model'])
	data = model(params)

	# # Normalize
	# method = 'left'
	# print(model.norm())
	# model.normalize(method=method)
	# print(model.norm())

	return

def main(settings,*args,**kwargs):
	
	train(settings,*args,**kwargs)
	
	return



def main(settings,*args,**kwargs):

	def init(data,parameters=None,state=None):

		locality = max(len(data[operator].operator) if not isinstance(data[operator].operator,str) else 1 for operator in data if data[operator].operator is not None)
		basis = Basis
		lattice = Lattice(locality)

		funcs = []
		for operator in data:

			obj = None

			if isinstance(data[operator].operator,str):
				objs = []
				for i in range(locality):
					objs.append(getattr(basis,data[operator].operator)(parameters=data[operator].parameters))
				obj = array([tensorprod(i) for i in permutations(*objs)])
			else:
				obj = basis.string(data=data[operator].operator,parameters=data[operator].parameters)

			if obj is None:
				continue

			if state is None:
				einsummation = None
				def contract(parameters,state,data=obj,einsummation=einsummation):
					return data
				contract = lambda parameters,state,data=obj,einsummation=einsummation: data
			elif obj.ndim == 3 and state.ndim == 2:
				subscripts = 'uij,jk,ulk->il'
				shapes = (obj.shape,state.shape,obj.shape)
				einsummation = einsum(subscripts,*shapes)
				def contract(parameters,state,data=obj,einsummation=einsummation):
					return einsummation(data,state,conjugate(data))				
			elif obj.ndim == 2 and state.ndim == 2:
				subscripts = 'ij,jk,lk->il'
				shapes = (obj.shape,state.shape,obj.shape)
				einsummation = einsum(subscripts,*shapes)
				def contract(parameters,state,data=obj,einsummation=einsummation):
					return einsummation(data,state,conjugate(data))	
			
			def func(parameters,state,contract=contract):
				return contract(parameters,state)

			funcs.append(func)


		def func(parameters=None,state=None):
			for func in funcs:
				state = func(parameters,state)
			return state 


		return func


	# Modules
	from src.quantum import Operators,State	
	from src.quantum import Basis,Measure
	from src.system import Lattice
	
	# Settings
	default = {}
	wrapper = Dict
	settings = load(settings,default=default,wrapper=wrapper)

	# System
	N = settings.state.N
	D = settings.state.D
	d = settings.state.d
	ndim = settings.state.ndim
	data = settings.model.data
	state = settings.state
	base = settings.model.base
	lattice = settings.model.lattice
	dtype = settings.system.dtype

	# Initialize
	basis = Basis
	lattice = Lattice(N,d,lattice=lattice)
	measure = Measure(base,D=D)
	structure = '<ij>'

	# Model
	parameters = None
	state = basis.zero(N=N,D=D,ndim=ndim,dtype=dtype)
	# state = None
	model = init(data,parameters,state)

	check = model(parameters,state)

	print(check)

	# Basis
	parameters=measure.parameters()
	state = measure.probability(parameters,state)
	operator = measure.operator(parameters,state,model=model)

	state = operator@state
	test = measure(parameters,state)

	print(test)


	assert allclose(test,check), "Incorrect model() - measure() conversion"


	return
	# Test
	# Function
	parameters = None
	state = basis.zero(N=N,D=D,ndim=ndim,dtype=dtype)

	value = model(parameters,state)

	# Model
	model = Operators(**settings.model)
	state = State(**settings.state)
	model.init(state=state)

	parameters = model.parameters()
	state = model.state()

	test = model(parameters,state)

	assert allclose(value,test), "Incorrect func and model"

	return



if __name__ == '__main__':

	settings = 'settings'

	args = argparser(settings)

	main(*args,**args)
