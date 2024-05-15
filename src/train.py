#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,allclose,delim,prng
from src.io import load,glob
from src.iterables import Dict,namespace
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

	# Check settings
	default = {}
	if settings is None:
		settings = default
	elif isinstance(settings,str):
		settings = load(settings,default=default)

	settings = Dict(settings)

	return settings

def train(settings):
	'''
	Train model
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

		settings = setup(settings)

		if not settings:
			model = None
			return model

		if not any(settings.boolean[attr] for attr in settings.boolean):
			model = None
			return model

		model = load(settings.cls.model)
		state = load(settings.cls.state)
		label = load(settings.cls.label)
		callback = load(settings.cls.callback)

		if any(i is None for i in [model,state,label,callback]):
			raise ValueError("Incorrect cls initialization")
			model = None
			return model

		hyperparameters = settings.optimize
		system = settings.system

		seed = prng(**settings.seed)


		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
		label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})
		callback = callback(**{**namespace(callback,model),**settings.callback,**dict(model=model,system=system)})

		if settings.boolean.load:
			model.load()

		if settings.boolean.train:

			func = model.parameters.constraints
			arguments = ()
			keywords = {}
			
			label.__initialize__(state=state)
			model.__initialize__(state=state)

			metric = Metric(state=state,label=label,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system)
			func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system)
			callback = Callback(model,func=func,callback=callback,arguments=arguments,keywords=keywords,metric=metric,hyperparameters=hyperparameters,system=system)

			optimizer = Optimizer(func=func,arguments=arguments,keywords=keywords,callback=callback,hyperparameters=hyperparameters,system=system)

			parameters = model.parameters()
			state = model.state()

			parameters = optimizer(parameters,state=state)

		if settings.boolean.dump:	
			model.dump()
	
		models[name] = model

	if len(models) == 1:
		models = models[name]

	return models


def main(*args,**kwargs):

	train(*args,**kwargs)

	return


if __name__ == '__main__':

	arguments = 'settings'
	args = argparser(arguments)

	main(*args,**args)