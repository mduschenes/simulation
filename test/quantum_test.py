#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,allclose,delim,prng,einsum,conjugate
from src.utils import similarity
from src.io import load,glob
from src.system import Dict
from src.iterables import namespace
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
# logger = Logger()

def main(*args,**kwargs):

	cls = Dict({
		"model":'src.quantum.Noise',
		"state":'src.quantum.State',
		})

	settings = Dict({
		"model":{
			"operator":"dephase",
			"site":None,
			"string":"noise",
			"parameters":1e-2,
			"N":1,"D":2,"ndim":3,
		},
		"state": {
			"operator":"zero",
			"site":None,
			"string":"psi",
			"parameters":True,
			"N":1,"D":2,"ndim":2,
			},
	})

	model = load(cls.model)
	state = load(cls.state)

	model = model(**settings.model)
	state = state(**settings.state)

	print('--- state ---')
	print(state())
	print()

	model.__initialize__(state=state,parameters=dict())


	print('--- model ---')
	print(model.data)
	print(model(model.parameters(model.parameters()),model.state()))
	print(einsum('uij,jk,ulk',model.data,model.state(),conjugate(model.data)))
	print()


	model.__initialize__(state=state,parameters=dict())

	print(model.data)
	print(model(model.parameters(model.parameters()),model.state()))
	print()


	func = similarity(model,label=model.state,shape=model.shape,hermitian=True,unitary=False)
	print(func(model.parameters(model.parameters()),model.state()))

	return


def test_quantum():
	main()
	return


if __name__ == '__main__':

	arguments = 'settings'
	args = argparser(arguments)

	main(*args,**args)