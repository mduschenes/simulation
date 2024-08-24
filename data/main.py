#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,nester
from src.iterables import permutations,namespace
from src.io import load,dump
from src.system import Dict

from src.quantum import Operators

def main(settings,*args,**kwargs):

	# Settings
	default = {}
	wrapper = lambda data: Dict(data)
	settings = load(settings,default=default,wrapper=wrapper)

	# Class
	module = load(settings.cls.module)
	model = load(settings.cls.model)
	state = load(settings.cls.state)
	system = settings.system

	# Model
	model = model(**{**settings.model,**dict(system=system)})

	state = state(**{**namespace(state,model),**settings.state,**dict(system=system)})

	model.info(verbose=True)
	state.info(verbose=True)
	exit()

	# Module
	module = module(**{**namespace(module,model),**namespace(module,state),**settings.module,**dict(model=model,state=state,system=system)})

	return



if __name__ == '__main__':

	settings = 'settings'

	args = argparser(settings)

	main(*args,**args)
