#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,allclose,delim,spawn
from src.io import load,glob
from src.iterables import Dict,namespace,setter
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
logger = Logger()

def setup(settings):
	'''
	Setup settings
	Args:
		settings (dict,str): settings
	Returns:
		settings (dict): settings
	'''

	default = {}
	defaults = Dict(
		boolean=dict(call=None,train=None,load=None,dump=None),
		cls=dict(model=None,state=None,label=None,callback=None),
		model=dict(),state=dict(),label=dict(),callback=dict(),
		optimize=dict(),seed=dict(),system=dict(),
		)

	if settings is None:
		settings = default
	elif isinstance(settings,str):
		settings = load(settings,default=default,wrapper=Dict)

	setter(settings,defaults,delimiter=delim,default=False)

	return settings


def train(settings,*args,**kwargs):
	'''
	Train model
	Args:
		settings (dict,str,iterable[str,dict]): settings
	Returns:
		model (object): Model instance
	'''

	settings = setup(settings)
	model = None


	if settings.boolean.load:
		model.load()


	if settings.boolean.call:
		
		model = load(settings.cls.model)
		state = load(settings.cls.state)
		system = settings.system

		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})

		model.init(state=state)


	if settings.boolean.train:

		label = load(settings.cls.label)
		callback = load(settings.cls.callback)

		label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})
		callback = callback(**{**namespace(callback,model),**settings.callback,**dict(model=model,system=system)})

		label.init(state=state)

		func = model.parameters.constraints
		seed = spawn(**settings.seed)
		hyperparameters = settings.optimize
		arguments = ()
		keywords = {}
		
		metric = Metric(state=state,label=label,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system)
		func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system)
		callback = Callback(model,func=func,callback=callback,arguments=arguments,keywords=keywords,metric=metric,hyperparameters=hyperparameters,system=system)

		optimizer = Optimizer(func=func,arguments=arguments,keywords=keywords,callback=callback,hyperparameters=hyperparameters,system=system)

		parameters = model.parameters()
		state = model.state()

		parameters = optimizer(parameters,state=state)


	if settings.boolean.dump:	
		model.dump()


	return model

def run(settings,*args,**kwargs):
	'''
	Run models
	Args:
		settings (dict,str,iterable[str,dict]): settings
	Returns:
		model (object): Model instance
	'''
	
	if settings is None:
		models = []
	elif isinstance(settings,str):
		models = list(glob(settings))
	elif isinstance(settings,dict):
		models = [settings]

	models = {name: model for name,model in enumerate(models)}

	for name in models:
		
		settings = models[name]

		model = train(settings,*args,**kwargs)
	
		models[name] = model

	if len(models) == 1:
		models = models[name]

	return models


def main(*args,**kwargs):

	run(*args,**kwargs)

	return


if __name__ == '__main__':

	arguments = 'settings'
	args = argparser(arguments)

	main(*args,**args)