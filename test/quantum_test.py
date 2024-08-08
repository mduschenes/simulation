#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,array,allclose,delim,prng,einsum,conjugate
from src.utils import similarity
from src.io import load,glob
from src.system import Dict
from src.iterables import namespace
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
# logger = Logger()

from src.quantum import Channel

def test_architecture(*args,**kwargs):

	data = {}
	architectures = ['array','mps']
	
	for architecture in architectures:
		cls = Dict({
			"model":'src.quantum.Operator',
			"state":'src.quantum.State',
			})

		settings = Dict({
			"model":{
				"operator":'X.Y',
				"site":[0,2],
				"string":"operator",
				"parameters":0.25,
				"N":3,"D":2,"ndim":2,
				"system":{"seed":123,"architecture":architecture}
			},	
			"state": {
				"data":"010",
				"operator":"product",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":123,"architecture":architecture}
				},
		})

		model = load(cls.model)
		state = load(cls.state)

		model = model(**settings.model)
		state = state(**settings.state)

		model.__initialize__(state=state)



		# Model

		value = model.data
		
		if architecture in ['array']:
			value = array(value)
		elif architecture in ['mps']:
			value = array(value)

		print('--- model ---')
		model.info(verbose=True)
		print(model.data)
		print('------')


		# State

		value = state()
		
		if architecture in ['array']:
			value = array(value)
		elif architecture in ['mps']:
			value = value.to_dense().reshape(-1)

		print('--- state ---')
		state.info(verbose=True)
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


	assert allclose(*(data[architecture] for architecture in data)), "Error - Incorrect architecture contraction"

	return


def test_contract(*args,**kwargs):

	architecture = 'array'

	cls = Dict({
		"model":'src.quantum.Operator',
		"state":'src.quantum.State',
		})

	settings = Dict({
		"model":{
			"operator":'X.X',
			"site":[0,1],
			"string":"operator",
			"parameters":0.5,
			"N":2,"D":2,"ndim":2,
			"system":{"architecture":architecture}
		},	
		"state": {
			"operator":"zero",
			"site":None,
			"string":"psi",
			"parameters":True,
			"N":2,"D":2,"ndim":1,
			"system":{"architecture":architecture}
			},
	})

	model = load(cls.model)
	state = load(cls.state)

	model = model(**settings.model)
	state = state(**settings.state)

	print('--- state ---')
	state.info(verbose=True)
	print(state())
	print('------')

	model.__initialize__(state=state)

	print()

	print('--- model ---')
	model.info(verbose=True)
	print(model.data)
	print('------')

	print('--- contract ---')
	print(model(model.parameters(model.parameters()),model.state()))
	print('------')

	print()


	print('--- einsum ---',model.ndim,state.ndim)
	if model.ndim == 3 and state.ndim == 2:
		print(einsum('uij,jk,ulk',model.data,model.state(),conjugate(model.data)))
	elif model.ndim == 3 and state.ndim == 1:
		print(einsum('uij,jk',model.data,model.state()))
	elif model.ndim == 2 and state.ndim == 2:
		print(einsum('ij,jk,lk',model.data,model.state(),conjugate(model.data)))
	elif model.ndim == 2 and state.ndim == 1:
		print(einsum('ij,jk',model.data,model.state()))
	print('------')

	print()


	model.__initialize__(state=state,parameters=dict())

	print(model.data)
	print(model(model.parameters(model.parameters()),model.state()))
	print()

	return


if __name__ == '__main__':

	arguments = 'settings'
	args = argparser(arguments)

	test_architecture(*args,**args)