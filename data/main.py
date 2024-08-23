#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,nester
from src.iterables import permutations,namespace
from src.io import load,dump
from src.system import Dict

def main(settings,*args,**kwargs):

	# Settings
	default = {}
	# wrapper = lambda data: Dict(nester(data,keys='cls',func=load))
	wrapper = lambda data: Dict(data)
	settings = load(settings,default=default,wrapper=wrapper)


	print(list(settings))

	# Initialize
	model = settings.cls.model
	state = settings.cls.state
	measure = settings.cls.measure
	basis = settings.cls.basis
	lattice = settings.cls.lattice
	system = settings.system

	# Model
	N = settings.model.N

	model = model(**{**settings.model,**dict(N=None,system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(N=None,model=model,system=system)})
	measure = measure(**{**namespace(measure,model),**settings.measure,**dict(system=system)})
	lattice = lattice(**{**namespace(lattice,model),**settings.lattice,**dict(system=system)})

	model.init(state=state)

	print(model.info())
	print(state.info())

	exit()

	state = [state()]
	model = settings.cls.model(data,D=D,N=N)

	# Tensor
	kwargs = dict(D=D,S=S,ndim=ndim,seed=seed,dtype=dtype)
	parameters = measure.parameters()
	state = array([getattr(Basis,state.operator)(**kwargs)]*N,dtype=dtype)

	assert allclose(tensorprod([*state]),measure.amplitude(parameters,measure.probability(parameters,state,**kwargs))), "Incorrect probability,amplitude state()"

	state = measure.probability(parameters,state,**kwargs)
	operator = measure.operator(parameters,state,model=model,N=2)

	_parameters = measure.parameters()
	_state = state
	_state = measure.amplitude(parameters=_parameters,state=_state)
	
	_parameters_ = measure.parameters()
	_state_ = state.copy()

	# _state_ = _state
	# _model_ = Model(data=data,state=_state_,N=N,D=D,d=d,M=M,dtype=dtype)
	# _parameters_ = _model_.parameters()
	# _state_ = _model_.state()

	# Calculate
	sites = list(lattice(structure))
	number = 1
	bound = M*len(sites)
	options = dict(
		contract="swap+split",
		max_bond=D**(2*2*N),
		cutoff=1e-20
		)
	# scheme = False
	for i in range(M):
		for site in sites:
			state = state.gate(operator,where=site,**options)
			_state = model(_parameters,_state,site=site)
			_state_ = _state_.gate(operator,where=site)
			
			print(i,site)
			print(state)
			print()

			number += 1

			if number > bound:
				break
		if number > bound:
			break				
	# _state_ = _model_(_parameters_,_state_)

	state = measure.amplitude(parameters=parameters,state=state)
	_state_ = measure.amplitude(parameters=_parameters_,state=_state_)

	print(abs(state-_state).sum(),maximum(abs(state-_state)))
	print(abs(_state-_state_).sum(),maximum(abs(_state-_state_)))
	# print(abs(_state_-state).sum(),maximum(abs(_state_-state)))

	assert allclose(state,_state),"Incorrect state.gate(), model(state)"

	print('Passed')

	return



if __name__ == '__main__':

	settings = 'settings'

	args = argparser(settings)

	main(*args,**args)
