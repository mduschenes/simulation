#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ["",".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,array,rand,zeros,ones,empty,allclose,product,spawn,einsum,conjugate,dagger,dot,tensorprod,abs,maximum,minimum,prod,log,swap,trace
from src.utils import arrays,iterables,scalars,integers,floats,pi,delim
from src.utils import array,tensor,mps,datastructure
from src.iterables import permutations
from src.io import load,dump,glob
from src.call import rm,echo
from src.system import Dict
from src.iterables import namespace,permuter,setter,getter
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
# logger = Logger()

from src.quantum import MPS

def test_MPS(*args,**kwargs):

	# Settings
	N = 3
	D = 4
	S = 1
	dtype = None

	# State
	data = None
	kwargs = dict(random='constant',seed=123,dtype=dtype)

	state = MPS(data=data,N=N,D=D,S=S,**kwargs)

	print(state)
	print(state.to_dense().ravel())


	return


def test_tensor(*args,**kwargs):

	# Settings
	N = 3
	D = 2
	K = 4
	S = 1
	dtype = 'complex'

	# State
	data = lambda shape: rand(shape=shape,random='constant',seed=123,dtype=None)
	kwargs = dict()

	state = mps(data,L=N,phys_dim=K,bond_dim=S,**kwargs)


	# Operator
	indices = ['k{}']
	shapes = {'i{}':D,'j{}':D}
	tag = 'I{}'
	data = lambda shape,i=1: i*rand(shape=shape,random='ones',dtype=None)
	kwargs = dict()

	state = state

	for i in range(N):
		shape = (*(state.ind_size(index.format(i)) for index in indices),*(shapes[index] for index in shapes))
		inds = (*(index.format(i) for index in indices),*(index.format(i) for index in shapes))
		tags = (tag.format(i),)
		operator = tensor(data(shape,i+1),inds=inds,tags=tags,**kwargs)
	
		state &= operator


	print(state)

	state = state.contract() 
	inds = (*((i,) for i in state.inds),)
	state = state.to_dense(*inds)

	print(state)
	print(state.shape)

	return


def test_init(*args,**kwargs):

	def init(data,parameters=None,state=None,**kwargs):
		'''
		Initialize model
		Args:
			data (dict): data for model
			parameters (array,callable): parameters for model
			state (array,callable): state for model
			kwargs (dict): Additional keyword arguments for model
		'''

		data = Dict(**data)
		kwargs = Dict(**kwargs)

		N = max(len(data[key].operator) if not isinstance(data[key].operator,str) else 1 for key in data if data[key].operator is not None)
		D = kwargs.D

		basis = Basis
		lattice = Lattice(N)

		func = {}
		for key in data:

			obj = None
			operator = data[key].operator if isinstance(data[key].operator,str) else (*data[key].operator,)
			site = data[key].site
			parameters = data[key].parameters

			if isinstance(operator,str):
				objs = []
				for i in lattice():
					objs.append(getattr(basis,operator)(D=D,parameters=parameters))
				obj = array([tensorprod(i) for i in permutations(*objs)])
			else:
				obj = basis.string(D=D,data=operator,parameters=parameters)

			if obj is None:
				continue


			N = kwargs.N
			D = kwargs.D

			shape = obj.shape
			size = obj.size
			ndim = obj.ndim
			dtype = obj.dtype

			if obj.ndim == 3:
				subscripts = 'uij,jk...,ulk->il...'
				shapes = (shape,shape[-2:],shape)
				einsummation = einsum(subscripts,*shapes)
				def function(state,data=obj,einsummation=einsummation):
					return einsummation(data,state,conjugate(data))
			
			elif obj.ndim == 2:
				subscripts = 'ij,jk...,lk->il...'
				shapes = (shape,shape[-2:],shape)
				einsummation = einsum(subscripts,*shapes)
				def function(state,data=obj,einsummation=einsummation):
					return einsummation(data,state,conjugate(data))
			
			def function(parameters,state,site=None,func=function):
				if site is not None:
					shape = (D,N,2)
					axes = (site,)
					state = swap(func(swap(state,shape=shape,axes=axes,transform=True)),shape=shape,axes=axes,transform=False)
				else:
					state = func(state)
				return state

			func[operator] = function


		class model(object):
			def __init__(self,N,D,shape,size,ndim,dtype,func):
				self.N = N
				self.D = D
				self.shape = shape
				self.size = size
				self.ndim = ndim
				self.dtype = dtype
				self.func = func
				return
			def __call__(self,parameters=None,state=None,site=None):
				for func in self.func:
					state = self.func[func](parameters,state,site=site)
				return state 


		N = N
		D = D
		ndim = 2

		shape = (D**N,)*ndim
		size = prod(shape)
		ndim = len(shape)
		dtype = 'complex'

		return model(N=N,D=D,shape=shape,size=size,ndim=ndim,dtype=dtype,func=func)

	# Modules
	from src.quantum import Operators,State,MPS
	from src.quantum import Basis,Measure
	from src.system import Lattice
	
	# Settings
	N = 6
	D = 2
	S = 1
	M = 10
	d = 1
	ndim = 2
	structure = ">ij<"
	data = 	Dict({
		"xx":{
			"operator":["X","X"],"site":structure,"string":"XX",
			"parameters":0.5,"variable":False
		},
		"noise":{
			"operator":"dephase","site":None,"string":"dephase",
			"parameters":1e-4,"variable":False
		}		
	})
	state = Dict({
		"operator":"zero",
		"site":None,
		"string":"psi",
		"parameters":True,
		"ndim":2,
		"seed":123
	})
	base = "pauli"
	lattice = "square"
	architecture = 'tensor'
	boundaries = "open"
	scheme = "swap+split"
	seed = 123
	dtype = "complex"

	# Initialize
	Model = Operators
	basis = Basis
	lattice = Lattice(N,d,lattice=lattice)	
	measure = Measure(base,D=D,architecture=architecture,dtype=dtype)

	# Model
	model = init(data,D=D,N=N)

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




if __name__ == "__main__":

	arguments = "path"
	args = argparser(arguments)

	# test_MPS(*args,**args)
	# test_tensor(*args,**args)
	test_init(*args,**args)

