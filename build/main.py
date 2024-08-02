#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,seeder,loader,partial,gradient
from src.io import load,dump
from src.system import Dict

def main(settings,*args,**kwargs):

	# Settings
	default = {}
	settings = load(settings,default=default,wrapper=lambda data: Dict(loader(data,keys='cls')))


	# System
	system = settings['system']
	seed = system['seed']
	shape = len(system['attrs'])
	wrapper = lambda keys: dict(zip(system['attrs'],keys))
	key = seeder(seed=seed)(shape,wrapper=wrapper)


	# Model
	model = settings['model']['cls']
	args = settings['model']['args']
	kwargs = settings['model']['kwargs']


	# Data
	data = settings['data']['cls'](*settings['data']['args'],**settings['data']['kwargs'])
	x = data.init(key['data'])


	# Label
	label = settings['label']['cls'](*settings['label']['args'],**settings['label']['kwargs'])
	variables = label.init(key['label'],x)


	# Objective
	objective = settings['objective']['cls'](*settings['objective']['args'],**settings['objective']['kwargs'],
		model=model,label=label,args=args,kwargs=kwargs)


	# Optimizer
	optimizer = settings['optimizer']['cls'](*settings['optimizer']['args'],**settings['optimizer']['kwargs'],
		objective=objective)


	# Train
	params = objective.init(key['model'],x)
	params = optimizer(params,x)
	

	# Callback
	print(objective.apply(params,x))

	return

if __name__ == '__main__':

	settings = 'settings'

	args = argparser(settings)

	main(*args,**args)
