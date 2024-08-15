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
from src.iterables import namespace,permuter,setter,getter,delim
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
# logger = Logger()

def test_architecture(*args,**kwargs):

	data = {}

	kwargs = {'model.system.architecture':['array','mps'],'state.system.architecture':['array','mps']}
	groups = [['model.system.architecture','state.system.architecture']]

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):
	
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
				"system":{"seed":12345,"dtype":"complex","architecture":None}
			},	
			"state": {
				"data":None	,
				"operator":"product",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":None}
				},
		})

		setter(settings,kwargs,delimiter=delim,default=True)

		verbose = True

		model = load(settings.cls.model)
		state = load(settings.cls.state)

		model = model(**settings.model)
		state = state(**settings.state)

		model.init(state=state)



		# Model

		value = model.data
		
		if kwargs['model.system.architecture'] ['array']:
			value = array(value)
		elif kwargs['model.system.architecture'] ['mps']:
			value = array(value)

		print('--- model ---')
		model.info(verbose=verbose)
		print(model.data)
		print('------')


		# State

		value = state()
		
		if kwargs['model.system.architecture'] ['array']:
			value = array(value)
		elif kwargs['model.system.architecture'] ['mps']:
			value = value.to_dense().reshape(-1)

		print('--- state ---')
		state.info(verbose=verbose)
		print(value)
		print('------')


		# Value
		
		value = model(model.parameters(model.parameters()),model.state())

		if kwargs['model.system.architecture'] ['array']:
			value = array(value)
		elif kwargs['model.system.architecture'] ['mps']:
			value = value.to_dense().reshape(-1)

		print('--- value ---')
		print(value)
		print('------')
		

		data[i] = value


	assert len(data)<2 or allclose(*(data[i] for i in data)), "Error - Incorrect architecture contraction"

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
			"system":{"architecture":None}
		},	
		"state": {
			"operator":"zero",
			"site":None,
			"string":"psi",
			"parameters":True,
			"N":2,"D":2,"ndim":1,
			"system":{"architecture":None}
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

	kwargs = {'model.system.architecture':['array','mps'],'state.system.architecture':['array','mps']}
	groups = [['model.system.architecture','state.system.architecture']]

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):

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
				"system":{"seed":12345,"dtype":"complex","architecture":None}
			},	
			"state": {
				# "data":"random",
				# "operator":"product",
				# "site":None,
				# "string":"psi",
				# "parameters":True,
				"N":2,"D":2,
				# "ndim":1,
				# "system":{"seed":12345,"dtype":"complex","architecture":None}
				},
		})

		setter(settings,kwargs,delimiter=delim,default=True)

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
		elif kwargs['model.system.architecture'] ['array']:
			value = {i: array(value[i]) if value[i] is not None else None for i in value}
		elif kwargs['model.system.architecture'] ['mps']:
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
		elif kwargs['model.system.architecture'] ['array']:
			value = array(value) if value is not None else None
		elif kwargs['model.system.architecture'] ['mps']:
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
		elif kwargs['model.system.architecture'] ['array']:
			value = array(value) if value is not None else None
		elif kwargs['model.system.architecture'] ['mps']:
			value = value.to_dense().reshape(-1) if value is not None and not isinstance(value,arrays) else array(value) if value is not None else None
		else:
			value = array(value) if value is not None else None			

		print('--- value ---')
		print(value)
		print('------')
		

		data[i] = value

	assert len(data)<2 or allclose(*(data[i] for i in data)), "Error - Incorrect architecture contraction"

	print("Passed")

	return

def test_amplitude(*args,**kwargs):

	kwargs = {'state.ndim':[1,2]}
	groups = None

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):

		settings = Dict({
			"cls":{
				"state":'src.quantum.Amplitude'
			},
			"state": {
				"data":"011",
				"operator":"string",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":"array"}
				}
			})

		setter(settings,kwargs,delimiter=delim,default=True)

		state = load(settings.cls.state)

		state = state(**settings.state)

		print(settings['state']['operator'],type(state))
		print(state.data)
		print(state(state.parameters(),state.state()))
		print(state.norm())
		print()

		assert allclose(state(),state.data), "Incorrect data for %r"%(settings.cls.state)
		assert allclose(state(state.parameters(),state.state()),state.data), "Incorrect state() for %r"%(settings.cls.state)
		assert allclose(state.norm(),1), "Incorrect normalization for %r"%(settings.cls.state)

	return

def test_probability(*args,**kwargs):

	kwargs = {'state.ndim':[1]}
	groups = None

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):

		settings = Dict({
			"cls":{
				"state":'src.quantum.Probability'
			},
			"state": {
				"data":[1/2]*2,
				"operator":"probability",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":"array"}
				}
			})

		setter(settings,kwargs,delimiter=delim,default=True)

		state = load(settings.cls.state)

		state = state(**settings.state)

		print(settings['state']['operator'],type(state))
		print(state.data)
		print(state(state.parameters(),state.state()))
		print(state.norm())
		print()

		assert allclose(state(),state.data), "Incorrect data for %r"%(settings.cls.state)
		assert allclose(state(state.parameters(),state.state()),state.data), "Incorrect state() for %r"%(settings.cls.state)
		assert allclose(state.norm(),1), "Incorrect normalization for %r"%(settings.cls.state)

	return

def test_state(*args,**kwargs):

	kwargs = {
		'cls.state':['src.quantum.State','src.quantum.Operator'],
		'state.data':[
			"random",
			[1/2]*2
			],
		'state.operator':[
			'product',
			'probability'
			],
		'state.ndim':[1],
		}
	groups = [['state.data','state.operator']]

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):

		settings = Dict({
			"cls":{
				"state":'src.quantum.State'
			},
			"state": {
				"data":"random",
				"operator":"probability",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":"array"}
				}
			})

		from src.quantum import State

		setter(settings,kwargs,delimiter=delim,default=True)

		state = load(settings.cls.state)

		state = state(**settings.state)

		print(settings['cls']['state'],settings['state']['operator'],type(state))
		print(state.data)
		print(state(state.parameters(),state.state()))
		print(state.norm())
		print()

		assert allclose(state(),state.data), "Incorrect data for %r"%(settings.cls.state)
		assert allclose(state(state.parameters(),state.state()),state.data), "Incorrect state() for %r"%(settings.cls.state)
		assert allclose(state.norm(),1), "Incorrect normalization for %r"%(settings.cls.state)

	return


def test_manifold(*args,**kwargs):

	settings = Dict({
		"cls":{
			"model":'src.quantum.Manifold',
			"state":'src.quantum.State'
		},
		"model":{
			'data':None,
			'operator':'pauli',
			'D':2,
			'dtype':'complex'
		},
		"state": {
			"data":None	,
			"operator":"probability",
			"site":None,
			"string":"psi",
			"parameters":True,
			"N":3,"D":2,"ndim":1,
			"system":{"seed":12345,"dtype":"complex","architecture":"array","base":"pauli"}
			}
		})

	model = load(settings.cls.model)
	state = load(settings.cls.state)

	model = model(**settings.model)
	state = state(**settings.state)

	model.init(state=state)

	print(model.data)

	return

def test_basis(*args,**kwargs):
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
			"system":{"seed":12345,"dtype":"complex","architecture":"array","base":"pauli"}
		},	
		"state": {
			"data":None	,
			"operator":"product",
			"site":None,
			"string":"psi",
			"parameters":True,
			"N":3,"D":2,"ndim":1,
			"system":{"seed":12345,"dtype":"complex","architecture":"array","base":"pauli"}
			},
	})

	model = load(settings.cls.model)
	state = load(settings.cls.state)

	model = model(**settings.model)
	state = state(**settings.state)

	model.init(state=state)

	print(model.base)

	return


if __name__ == '__main__':

	arguments = 'path'
	args = argparser(arguments)

	# main(*args,**args)

	# test_architecture(*args,**args)
	# test_contract(*args,**args)
	# test_module(*args,**args)
	# test_amplitude(*args,**args)
	# test_probability(*args,**args)
	# test_state(*args,**args)
	test_manifold(*args,**args)
	# test_basis(*args,**args)