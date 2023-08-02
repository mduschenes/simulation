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
from src.system import Dict
from src.iterables import namespace
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
logger = Logger()

def setup(hyperparameters):
	'''
	Setup hyperparameters
	Args:
		hyperparameters (dict,str): hyperparameters
	Returns:
		hyperparameters (dict): hyperparameters
	'''

	# Check hyperparameters
	default = {}
	if hyperparameters is None:
		hyperparameters = default
	elif isinstance(hyperparameters,str):
		hyperparameters = load(hyperparameters,default=default)

	hyperparameters = Dict(hyperparameters)

	return hyperparameters

def train(hyperparameters):
	'''
	Train model
	Args:
		hyperparameters (dict,str,iterable[str,dict]): hyperparameters
	Returns:
		model (object): Model instance
	'''
	if hyperparameters is None:
		models = []
	elif isinstance(hyperparameters,str):
		models = list(glob(hyperparameters))
	elif isinstance(hyperparameters,dict):
		models = [hyperparameters]

	models = {name: model for name,model in enumerate(models)}

	for name in models:
		
		hyperparameters = models[name]

		hyperparameters = setup(hyperparameters)

		if not hyperparameters:
			model = None
			return model

		if not any(hyperparameters.boolean[attr] for attr in hyperparameters.boolean):
			model = None
			return model

		model = load(hyperparameters.cls.model)
		from src.quantum import Channel as model
		state = load(hyperparameters.cls.state)
		label = load(hyperparameters.cls.label)
		callback = load(hyperparameters.cls.callback)

		if any(i is None for i in [model,label,callback]):
			model = None
			return model

		hyperparams = hyperparameters.optimize
		system = hyperparameters.system

		seed = prng(**hyperparameters.seed)

		model = model(**{**hyperparameters.model,**dict(parameters=hyperparameters.parameters),**dict(system=system)})
		state = state(**{**namespace(state,model),**hyperparameters.state,**dict(model=model,system=system)})
		label = label(**{**namespace(label,model),**hyperparameters.label,**dict(model=model,system=system)})
		callback = callback(**{**namespace(callback,model),**hyperparameters.callback,**dict(model=model,system=system)})

		if hyperparameters.boolean.load:
			model.load()

		if hyperparameters.boolean.train:

			parameters = model.parameters()
			func = model.parameters.constraints
			arguments = ()
			keywords = {}
			
			label.__initialize__(state=state)
			model.__initialize__(state=state)

			metric = Metric(state=state,label=label,arguments=arguments,keywords=keywords,hyperparameters=hyperparams,system=system)
			func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparams,system=system)
			callback = Callback(model,func=func,callback=callback,arguments=arguments,keywords=keywords,metric=metric,hyperparameters=hyperparams,system=system)

			optimizer = Optimizer(func=func,arguments=arguments,keywords=keywords,callback=callback,hyperparameters=hyperparams,system=system)

			parameters = optimizer(parameters)

			model.parameters.data = parameters

		if hyperparameters.boolean.dump:	
			model.dump()
	
		models[name] = model

	if len(models) == 1:
		models = models[name]

	return models


def main(*args,**kwargs):

	train(*args,**kwargs)

	return


if __name__ == '__main__':

	arguments = 'hyperparameters'
	args = argparser(arguments)

	main(*args,**args)