#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,vmap,seeder,nester,partial,gradient,dot,einsum,tensorprod,conjugate,allclose
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

			if obj.ndim == 3:
				subscripts = 'uij,jk...,ulk->il...'
				shapes = (obj.shape,obj.shape[-2:],obj.shape)
				einsummation = einsum(subscripts,*shapes)
				def contract(parameters,state,data=obj,einsummation=einsummation):
					return einsummation(data,state,conjugate(data))				
			elif obj.ndim == 2:
				subscripts = 'ij,jk...,lk->il...'
				shapes = (obj.shape,obj.shape[-2:],obj.shape)
				einsummation = einsum(subscripts,*shapes)
				def contract(parameters,state,data=obj,einsummation=einsummation):
					return einsummation(data,state,conjugate(data))	
			
			def func(parameters,state,contract=contract):
				return contract(parameters,state)

			funcs.append(func)


		class model(object):
			def __init__(self,locality,funcs):
				self.locality = locality
				self.funcs = funcs
				return
			def __call__(self,parameters=None,state=None):
				
				locality = int(round(log(len(state))/log(len(self.basis)))) if state is not None else None

				locality = max(len(data[operator].operator) if not isinstance(data[operator].operator,str) else 1 for operator in data if data[operator].operator is not None)

				for func in self.funcs:
					state = func(parameters,state)
				return state 


		return model(locality,funcs)


	# Modules
	from src.quantum import Operators,State,MPS
	from src.quantum import Basis,Measure
	from src.system import Lattice
	
	# Settings
	default = {}
	wrapper = Dict
	settings = load(settings,default=default,wrapper=wrapper)

	# System
	N = settings.model.N
	D = settings.model.D
	S = settings.model.S
	d = settings.model.d
	data = settings.model.data
	base = settings.model.base
	lattice = settings.model.lattice
	boundaries = settings.model.boundaries
	scheme = settings.model.scheme
	seed = settings.system.seed
	dtype = settings.system.dtype

	# Initialize
	basis = Basis
	measure = Measure(base,D=D)

	# Model
	parameters = None
	state = None
	model = init(data,parameters,state)

	# Basis
	parameters = measure.parameters()
	state = basis.zero(N=N,D=D,ndim=2,dtype=dtype)

	state = measure.probability(parameters,state)
	operator = measure.operator(parameters,state,model=model)

	# Tensor
	D = len(measure)
	S = 1
	random = 'rand'
	seed = seed
	bounds = [0,1]
	scale = None
	dtype = dtype
	state = MPS(N=N,D=D,S=S,boundaries=boundaries,random=random,seed=seed,bounds=bounds,scale=scale,dtype=dtype)
	lattice = Lattice(N,d,lattice=lattice)
	structure = '<ij>'


	_parameters = parameters
	_state = state.to_dense().ravel()
	_state = measure(parameters=_parameters,state=_state)

	for site in lattice(structure):
		state = state.gate(operator,where=site,contract=scheme)
		_state = model(parameters,_state)

	state = state.to_dense().ravel()
	state = measure(parameters=parameters,state=state)

	assert allclose(state,_state),"Incorrect state.gate(), model(state)"

	return



if __name__ == '__main__':

	settings = 'settings'

	args = argparser(settings)

	main(*args,**args)
