#!/usr/bin/env python

# Import packages
import os,sys
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.io import load
from src.system import Dict
from src.iterables import namespace


def call(settings,*args,**kwargs):

	# Load settings file (wrap in Dict class for object-style key-value pair attributes)
	settings = load(settings,wrapper=Dict)

	# Load classes (from path of class i.e) src.quantum.py)
	Module = load(settings.cls.module)
	Model = load(settings.cls.model)
	State = load(settings.cls.state)
	Callback = load(settings.cls.callback)

	# Get system settings
	system = settings.system

	# Initialize model classes (getting attributes common to previous model namespaces)
	if Module is not None and Model is not None and State is not None:

		model = Model(**{**settings.model,**dict(system=system)})
		state = State(**{**namespace(State,model),**settings.state,**dict(system=system)})
		callback = Callback(**{**settings.callback,**dict(system=system)})

		module = Module(**{**settings.module,**namespace(Module,model),**dict(model=model,state=state,callback=callback,system=system)})

		module.init()

		model = module

	elif Model is not None and State is not None:

		model = Model(**{**settings.model,**dict(system=system)})
		state = State(**{**namespace(State,model),**settings.state,**dict(system=system)})

		model.init(state=state)

	elif model is not None:

		model = Model(**{**settings.model,**dict(system=system)})

	else:

		model = None


	# Dump model
	model.dump()


	return model


def main(*args,**kwargs):
	
	call(*args,**kwargs)

	return


if __name__ == '__main__':

	# Parse command line arguments (default: settings.json)
	arguments = {'--settings':'settings.json'}
	arguments = argparser(arguments)

	main(**arguments)