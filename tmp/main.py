#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,seeder,nester,partial,gradient
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

	def init(**data):
		data = [data[i] for i in data]
		operators = [Dict(operator=(*operator,) if not isinstance(operator,str) else (operator,)) if not isinstance(operator,dict) else operator for operator in data]

		N = max(len(operator.operator) for operator in data)
		indices = {1:'i',2:'<ij>'}
		basis = Basis
		lattice = Lattice(N)

		data = {}
		for operator in operators:
			attr = operator.operator
			objs = []
			for i in lattice(indices[len(operator.operator)]):
				obj.append(getattr(basis,ope))
			data[attr] = array([tensorprod(i) for i in permutations(*objs)],dtype=dtype)



	# Modules
	from src.quantum import Basis,Measure
	from src.system import Lattice
	
	# Settings
	default = {}
	settings = load(settings,default=default,wrapper=Dict)

	# System
	data = settings.model.data
	base = settings.model.base

	func = init(**data)
	measure = Measure(base)






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



if __name__ == '__main__':

	settings = 'settings'

	args = argparser(settings)

	main(*args,**args)
