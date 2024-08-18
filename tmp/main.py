#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,seeder,nester,partial,gradient,einsum,tensorprod,conjugate
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

	def init(data):

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

			state = empty(obj.shape[-2:]) if obj is not None else None

			if obj is None or state is None:
				continue

			if obj.ndim == 3:
				subscripts = 'uij,jk,ulk->il'
				shapes = (obj.shape,state.shape,obj.shape)
				contract = einsum(subscripts,*shapes)
			elif obj.ndim == 2:
				subscripts = 'ij,jk,lk->il'
				shapes = (obj.shape,state.shape,obj.shape)
				contract = einsum(subscripts,*shapes)
			
			def func(parameters,state,data=obj,contract=contract):
				return contract(data,state,conjugate(data))

			funcs.append(func)


		def func(parameters,state):
			for func in funcs:
				state = func(parameters,state)
			return state 


		return func


	# Modules
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
	func = init(data)
	basis = Basis
	lattice = Lattice(N,d,lattice=lattice)
	measure = Measure(base)
	structure = '<ij>'

	# Model
	parameters = None
	state = basis.zero(N=N,D=D,ndim=ndim,dtype=dtype)

	state = func(parameters,state)

	print(state)

	return



if __name__ == '__main__':

	settings = 'settings'

	args = argparser(settings)

	main(*args,**args)
