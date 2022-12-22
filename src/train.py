#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime
from copy import deepcopy as deepcopy
from time import time as timer
from functools import partial

# Logging
import logging

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,allclose
from src.io import load
from src.dictionary import resetter
from src.optimize import Optimizer,Objective
from src.callback import Callback


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
		'optimize.path': 'sys.path.data.data',
	}

	resetter(hyperparameters,updates)

	return hyperparameters

def train(hyperparameters):
	'''
	Train object
	Args:
		hyperparameters (dict,str): hyperparameters
	'''

	hyperparameters = setup(hyperparameters)

	if not hyperparameters:
		obj = None
		return obj

	if not any(hyperparameters['boolean'].get(attr) for attr in ['load','dump','train']):
		obj = None
		return obj

	cls = load(hyperparameters['class'])

	obj = cls(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

	if hyperparameters['boolean'].get('load'):
		obj.load()

	if hyperparameters['boolean'].get('train'):

		parameters = obj.parameters
		hyperparams = hyperparameters['optimize']
		func = [obj.__objective__,obj.__constraints__]
		model = obj

		objective = Objective(model=model,func=funcs)

		func = objective.__call__
		callback = objective.__callback__

		optimizer = Optimizer(func=func,callback=callback,hyperparameters=hyperparams)

		parameters = optimizer(parameters)
	
	if hyperparameters['boolean'].get('dump'):	
		obj.dump()
	
	return obj


def main(*args,**kwargs):

	train(*args,**kwargs)

	return


if __name__ == '__main__':

	arguments = 'hyperparameters'
	args = argparser(arguments)

	main(*args,**args)