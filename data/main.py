#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,vmap,seeder,nester,partial,gradient,dot,einsum,tensorprod,conjugate,allclose,prod,log
from src.utils import array,zeros,ones,empty
from src.utils import swap
from src.iterables import permutations
from src.io import load,dump
from src.system import Dict

def main(settings,*args,**kwargs):

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

		func = []
		for key in data:

			obj = None
			operator = data[key].operator
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
					N = int(round(log(state.size)/log(D)/state.ndim))
					shape = (D,N,2)
					axes = (site,)
					state = swap(func(swap(state,shape=shape,axes=axes,transform=True)),shape=shape,axes=axes,transform=False)
				else:
					state = func(state)
				return state

			func.append(function)


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
					state = func(parameters,state,site=site)
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
	default = {}
	wrapper = Dict
	settings = load(settings,default=default,wrapper=wrapper)

	# System
	N = settings.model.N
	D = settings.model.D
	S = settings.model.S
	M = settings.model.M
	d = settings.model.d
	data = settings.model.data
	base = settings.model.base
	lattice = settings.model.lattice
	boundaries = settings.model.boundaries
	scheme = settings.model.scheme
	seed = settings.system.seed
	dtype = settings.system.dtype

	# Initialize
	Model = Operators
	basis = Basis
	measure = Measure(base,D=D)

	# Model
	parameters = None
	state = None
	model = init(data,parameters,state,D=D)

	# Tensor

	parameters = measure.parameters()
	state = [settings.state.operator]*settings.model.N
	kwargs = dict(
		D=model.D, S=settings.model.S, ndim=model.ndim,
		dtype = dtype
		)

	state = measure.probability(parameters,state,**kwargs)
	operator = measure.operator(parameters,state,model=model)

	_parameters = measure.parameters()
	_state = state.copy()
	_state = measure.amplitude(parameters=_parameters,state=_state)
	# _model = Model(**settings.model,**settings.system,state=_state)

	print(_state)
	exit()

	# Calculate
	lattice = Lattice(N,d,lattice=lattice)
	structure = '>ij<'


	for i in range(M):
		for site in lattice(structure):
			state = state.gate(operator,where=site)

	for i in range(M):
		for site in lattice(structure):
			_state = model(_parameters,_state,site=site)

	state = measure.amplitude(parameters=parameters,state=state)

	print(state.round(8))
	print(_state.round(8))

	assert allclose(state,_state),"Incorrect state.gate(), model(state)"

	return



if __name__ == '__main__':

	settings = 'settings'

	args = argparser(settings)

	main(*args,**args)
