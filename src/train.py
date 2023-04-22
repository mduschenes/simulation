#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,allclose,delim,namespace
from src.io import load,glob
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

		if not any(hyperparameters['boolean'].get(attr) for attr in ['load','dump','train']):
			model = None
			return model

		backend = hyperparameters.get('backend')
		if backend is not None:
			backend = __import__(backend)

		cls = {attr: load(hyperparameters['class'][attr]) for attr in hyperparameters.get('class',{})}
		system = hyperparameters.get('system',{})

		model,label,callback = cls.pop('model'),cls.pop('label'),cls.pop('callback')

		model = model(**{**hyperparameters.get('model',{}),**{attr: hyperparameters.get(attr) for attr in cls},**dict(system=system)})
		label = label(**{**namespace(label,model),**hyperparameters.get('label',{}),**dict(model=model,system=system)})
		callback = callback(**{**namespace(callback,model),**hyperparameters.get('callback',{}),**dict(model=model,system=system)})

		if hyperparameters['boolean'].get('load'):
			model.load()

		if hyperparameters['boolean'].get('train'):

			hyperparams = hyperparameters['optimize']		

			shapes = label.shape
			func = [model.parameters.constraints]
			
			parameters = model.parameters()
			label = label()

			metric = Metric(shapes=shapes,label=label,hyperparameters=hyperparams,system=system)
			func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparams,system=system)
			callback = Callback(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparams,system=system)

			optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparams,system=system)

			exit()
			parameters = optimizer(parameters)

			model.parameters.data = parameters

		if hyperparameters['boolean'].get('dump'):	
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