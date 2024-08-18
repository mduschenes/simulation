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

if __name__ == '__main__':

	settings = 'settings'

	args = argparser(settings)

	main(*args,**args)
