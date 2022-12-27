#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,allclose,delim
from src.io import load
from src.dictionary import resetter
from src.optimize import Optimizer,Objective,Metric,Callback


def setup(hyperparameters)
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

	updates = {
		'optimize.cwd': 'sys.cwd',
		'optimize.path': 'sys.path.data.data',
		'optimize.logger': 'sys.path.config.logger',
		'optimize.logging': 'sys.path.data.log',
		'optimize.timestamp': 'model.timestamp',
		'optimize.key': 'model.key',
	}

	resetter(hyperparameters,updates,delimiter=delim)

	return hyperparameters

def train(hyperparameters):
	'''
	Train model
	Args:
		hyperparameters (dict,str): hyperparameters
	Returns:
		model (object): Model instance
	'''

	hyperparameters = setup(hyperparameters)

	if not hyperparameters:
		model = None
		return model

	if not any(hyperparameters['boolean'].get(attr) for attr in ['load','dump','train']):
		model = None
		return model

	cls = {attr: load(hyperparameters['class'][attr]) for attr in hyperparameters['class']}

	model = cls['model'](**hyperparameters['model'],hyperparameters=hyperparameters)

	if hyperparameters['boolean'].get('load'):
		model.load()

	if hyperparameters['boolean'].get('train'):

		parameters = model.parameters
		shapes = model.shapes
		label = model.label
		hyperparams = hyperparameters['optimize']
		func = [model.__constraints__]
		callback = cls['callback']()

		metric = Metric(shapes=shapes,label=label,hyperparameters=hyperparams)
		func = Objective(model,metric,func=func,callback=callback,hyperparameters=hyperparams)
		callback = Callback(model,callback=callback,func=func,metric=metric,hyperparameters=hyperparams)

		optimizer = Optimizer(func=func,model=model,callback=callback,hyperparameters=hyperparams)

		parameters = optimizer(parameters)

		model.parameters = parameters
	
	if hyperparameters['boolean'].get('dump'):	
		model.dump()
	
	return model


def main(*args,**kwargs):

	train(*args,**kwargs)

	return


if __name__ == '__main__':

	arguments = 'hyperparameters'
	args = argparser(arguments)

	main(*args,**args)