#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ["",".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,array,rand,zeros,ones,empty,allclose,product,spawn,einsum,conjugate,dot,tensorprod
from src.utils import arrays,iterables,scalars,integers,floats,pi,delim
from src.utils import array,tensor,mps
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
	data = lambda shape: rand(shape=shape,random='ones',dtype=None)
	kwargs = dict()

	for i in range(N):
		shape = (*(state.ind_size(index.format(i)) for index in indices),*(shapes[index] for index in shapes))
		inds = (*(index.format(i) for index in indices),*(index.format(i) for index in shapes))
		operator = tensor(data(shape),inds=inds,**kwargs)
	
		state &= operator

		for index in indices:
			state.contract_ind(index.format(i))

	print(state)

	return



if __name__ == "__main__":

	arguments = "path"
	args = argparser(arguments)

	# test_MPS(*args,**args)
	test_tensor(*args,**args)
