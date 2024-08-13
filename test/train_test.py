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
from src.io import load,glob
from src.call import rm
from src.system import Dict
from src.iterables import namespace
from src.logger import Logger

from src.train import train

# logger = Logger()
@pytest.mark.filterwarnings(r"ignore:Rounding errors prevent the line search from converging")
@pytest.mark.filterwarnings(r"ignore:The line search algorithm did not converge")
def test_train(path,*args,tol=None,**kwargs):

	path = 'config/settings.json'# if path is None else path
	tol = 1e-9 # if tol is None else tol

	settings = path
	args = ()
	kwargs = {}

	model,parameters,state,optimizer = train(settings,*args,**kwargs)

	paths = [optimizer.cwd]
	execute = True
	verbose = True
	for path in paths:
		rm(path,execute=execute,verbose=verbose)

	assert optimizer.track['objective'][-1] < tol, "Incorrect Optimization of %r"%(model)

	print("Passed")

	return


if __name__ == '__main__':

	arguments = 'path'
	args = argparser(arguments)

	test_train(*args,**args)
