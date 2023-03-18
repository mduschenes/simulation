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
from src.optimize import Optimizer,Objective,Metric,Callback


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

	model = cls['model'](**hyperparameters['model'],
			parameters=hyperparameters['parameters'],
			state=hyperparameters['state'],
			noise=hyperparameters['noise'],
			label=hyperparameters['label'],
			system=hyperparameters['system'])

	if hyperparameters['boolean'].get('load'):
		model.load()

	if hyperparameters['boolean'].get('train'):

		parameters = model.parameters()
		shapes = model.shapes
		label = model.label()
		hyperparams = hyperparameters['optimize']
		system = hyperparameters['system']
		kwargs = {attr: hyperparams.get(attr) for attr in system if attr in hyperparams}
		func = [model.constraints]
		callback = cls['callback']()

		metric = Metric(shapes=shapes,label=label,hyperparameters=hyperparams,system=system,**kwargs)
		func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparams,system=system,**kwargs)
		callback = Callback(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparams,system=system,**kwargs)

		optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparams,system=system,**kwargs)

		parameters = optimizer(parameters)

		model.parameters(parameters)
	
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