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

def test_architecture(*args,**kwargs):

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
				"system":{"seed":123,"architecture":architecture}
			},	
			"state": {
				"data":None	,
				"operator":"product",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":123,"architecture":architecture}
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


def test_contract(*args,**kwargs):

	return

	architecture = 'array'

	settings = Dict({
		"cls":{
				"model":'src.quantum.Operator',
				"state":'src.quantum.State'
			},		
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

	verbose = True

	model = load(settings.cls.model)
	state = load(settings.cls.state)

	model = model(**settings.model)
	state = state(**settings.state)

	print('--- state ---')
	state.info(verbose=verbose)
	print(state())
	print('------')

	model.init(state=state)

	print()

	print('--- model ---')
	model.info(verbose=verbose)
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


	model.init(state=state,parameters=dict())

	print(model.data)
	print(model(model.parameters(model.parameters()),model.state()))
	print()

	return

def test_module(*args,**kwargs):

	data = {}
	architectures = [
		# None,
		'array',
		'mps'
		]
	
	for architecture in architectures:

		settings = Dict({
			"cls":{
				"model":'src.quantum.Module',
				"state":'src.quantum.State'
			},			
			"model":{
				"data":{
					"XX":{
						"operator":['X','X'],
						"site":[0,1],
						"string":"xx",
						"parameters":pi,
						"variable":True
						},
					# "noise":{
					# 	"operator":"dephase",
					# 	"site":[0,1],
					# 	"string":"noise",
					# 	"parameters":1e-1,
					# 	"variable":False						
					# 	}						
					},
				"N":2,"D":2,
				"system":{"seed":123,"architecture":architecture}
			},	
			"state": {
				# "data":"random",
				# "operator":"product",
				# "site":None,
				# "string":"psi",
				# "parameters":True,
				"N":2,"D":2,
				# "ndim":1,
				# "system":{"seed":123,"architecture":architecture}
				},
		})

		verbose = False

		model = load(settings.cls.model)
		state = load(settings.cls.state)

		model = model(**settings.model)
		state = state(**settings.state)

		model.init(state=state)



		# Model

		value = {i: model.data[i].data for i in model.data}

		if architecture is None:
			value = {i: array(value[i]) if value[i] is not None else None for i in value}			
		elif architecture in ['array']:
			value = {i: array(value[i]) if value[i] is not None else None for i in value}
		elif architecture in ['mps']:
			value = {i: array(value[i]) if value[i] is not None else None for i in value}
		else:
			value = {i: array(value[i]) if value[i] is not None else None for i in value}			

		print('--- model ---')
		model.info(verbose=verbose)
		print(value)
		print('------')


		# State

		value = state()
		
		if architecture is None:
			value = array(value) if value is not None else None			
		elif architecture in ['array']:
			value = array(value) if value is not None else None
		elif architecture in ['mps']:
			value = value.to_dense().reshape(-1) if value is not None and not isinstance(value,arrays) else array(value) if value is not None else None
		else:
			value = array(value) if value is not None else None						

		print('--- state ---')
		state.info(verbose=verbose)
		print(value)
		print('------')


		# Value
		
		value = model(model.parameters(model.parameters()),model.state())

		if architecture is None:
			value = array(value) if value is not None else None			
		elif architecture in ['array']:
			value = array(value) if value is not None else None
		elif architecture in ['mps']:
			value = value.to_dense().reshape(-1) if value is not None and not isinstance(value,arrays) else array(value) if value is not None else None
		else:
			value = array(value) if value is not None else None			

		print('--- value ---')
		print(value)
		print('------')
		

		data[architecture] = value

	assert len(data)<2 or allclose(*(data[architecture] for architecture in data)), "Error - Incorrect architecture contraction"

	print("Passed")

	return

def test_state(*args,**kwargs):
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
			"system":{"seed":123,"architecture":"array"}
		},	
		"state": {
			"data":None	,
			"operator":"product",
			"site":None,
			"string":"psi",
			"parameters":True,
			"N":3,"D":2,"ndim":1,
			"system":{"seed":123,"architecture":"array"}
			},
	})

	verbose = True

	model = load(settings.cls.model)
	state = load(settings.cls.state)

	model = model(**settings.model)
	state = state(**settings.state)

	model.init(state=state)

	data = {
		"state":model(model.parameters(),model.state()),
		"model":model(model.parameters(),model(model.parameters(),model.state())),
		}

	state.init(data=model(model.parameters(),model.state()))
	model.init(state=state)

	assert allclose(state(state.parameters(),state.state()),data['state']), "State not reinitialized with model data"
	assert allclose(model(model.parameters(),model.state()),data['model']), "Model not reinitialized with state data"

	print('Passed')

	return


if __name__ == '__main__':

	arguments = 'path'
	args = argparser(arguments)

	# main(*args,**args)

	# test_architecture(*args,**args)
	# test_contract(*args,**args)
	test_module(*args,**args)
	# test_state(*args,**args)
