#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,array,allclose,delim,spawn,einsum,conjugate
from src.utils import arrays,iterables,scalars,integers,floats,pi
from src.io import load,dump,glob
from src.call import rm,echo
from src.system import Dict
from src.iterables import namespace
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
# logger = Logger()

def test_simulation(*args,path=None,tol=None,**kwargs):

	data = {}
	architectures = ['array','mps']
	
	for architecture in architectures:

		settings = Dict({
			"cls":{
				"model":'src.quantum.Operator',
				"state":'src.quantum.State'			
			},
			"model":{
				"operator":'X.Y',
				"site":[0,2],
				"string":"operator",
				"parameters":0.25,
				"N":3,"D":2,"ndim":2,
				"system":{"seed":123456789,"architecture":architecture}
			},	
			"state": {
				"data":None	,
				"operator":"product",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":123456789,"architecture":architecture}
				},
		})

		verbose = True

		model = load(settings.cls.model)
		state = load(settings.cls.state)

		model = model(**settings.model)
		state = state(**settings.state)

		model.init(state=state)



		# Model

		value = model.data
		
		if architecture in ['array']:
			value = array(value)
		elif architecture in ['mps']:
			value = array(value)

		print('--- model ---')
		model.info(verbose=verbose)
		print(model.data)
		print('------')


		# State

		value = state()
		
		if architecture in ['array']:
			value = array(value)
		elif architecture in ['mps']:
			value = value.to_dense().reshape(-1)

		print('--- state ---')
		state.info(verbose=verbose)
		print(value)
		print('------')


		# Value
		
		value = model(model.parameters(model.parameters()),model.state())

		if architecture in ['array']:
			value = array(value)
		elif architecture in ['mps']:
			value = value.to_dense().reshape(-1)

		print('--- value ---')
		print(value)
		print('------')
		

		data[architecture] = value


	assert len(data)<2 or allclose(*(data[architecture] for architecture in data)), "Error - Incorrect architecture contraction"

	return


def test_probability(*args,path=None,tol=None,**kwargs):
	settings = Dict({
		"cls":{
			"model":'src.quantum.Operator',
			"state":'src.quantum.State'			
		},
		"model":{
			"operator":'X.X.X',
			"site":[0,1,2],
			"string":"operator",
			"parameters":0.25,
			"N":3,"D":2,"ndim":2,
			"system":{"seed":123456789,"architecture":"probability"}
		},	
		"state": {
			"data":[0.3,0.7],
			"operator":"probability",
			"site":None,
			"string":"probability",
			"parameters":True,
			"N":3,"D":2,"ndim":1,
			"system":{"seed":123456789,"verbose":True,"architecture":"probability"}
			},
	})

	verbose = True

	model = load(settings.cls.model)
	state = load(settings.cls.state)

	model = model(**settings.model)
	state = state(**settings.state)

	return


if __name__ == '__main__':

	arguments = 'path'
	args = argparser(arguments)

	# test_simulation(*args,**args)
	test_probability(*args,**args)
