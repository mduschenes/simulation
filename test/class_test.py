#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy
	
# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))


from src.states import State
from src.utils import einsum,allclose
from src.io import load,dump

# Logging
# from src.system import Logger
# name = __name__
# path = os.getcwd()
# file = 'logging.conf'
# conf = os.path.join(path,file)
# file = None #'log.log'
# logger = Logger(name,conf,file=file)



def test_class(path,tol):

	hyperparameters = load(path)

	classes = {'state':'src.states.State','noise':'src.noise.Noise','label':'src.operators.Operator'}

	name = 'state'
	name = 'label'
	cls = load(classes[name])


	# Initial instance
	data = hyperparameters['state']
	shape = (hyperparameters['model']['D']**hyperparameters['model']['N'],)*2
	size = [1,4]
	dims = [hyperparameters['model']['N'],hyperparameters['model']['D']]
	samples = True
	system = {'dtype':'complex','verbose':True}

	obj = cls(data,shape,size=size,dims=dims,samples=samples,system=system)

	print('Name : %s'%(name))
	obj.info()
	print()

	data = obj()


	if obj.ndim == 1:
		if obj.samples is not None:
			normalization = einsum('...i,...i->...',data,data.conj()).real/1
		else:
			normalization = einsum('...i,...i->...',data,data.conj()).real/1
	elif obj.ndim == 2:
		if name in ['state'] and obj.samples is not None:
			normalization = einsum('...ii->...',data).real/1
		else:
			normalization = einsum('...ij,...ij->...',data,data.conj()).real/obj.n
	else:
		raise AssertionError("Incorrect obj.ndim = %d"%(obj.ndim))

	assert(allclose(1,normalization)),"Incorrectly normalized obj: %0.5e"%(normalization)



	# Identical instance
	old = obj()

	data = dict(obj)
	shape = (hyperparameters['model']['D']**hyperparameters['model']['N'],)*2
	size = [1,4]
	dims = [hyperparameters['model']['N'],hyperparameters['model']['D']]
	samples = True
	system = {'dtype':'complex'}

	obj = cls(data,shape,size=size,dims=dims,samples=samples,system=system)


	print('Name : %s'%(name))
	obj.info()
	print()

	assert(allclose(obj(),old))


	# Difference instance
	data = dict(obj)
	data['scale'] = None
	data['logger'] = 'log.txt'
	data['system']['cleanup'] = True
	shape = (hyperparameters['model']['D']**hyperparameters['model']['N'],)*2
	size = [1,]
	dims = [hyperparameters['model']['N'],hyperparameters['model']['D']]
	samples = False
	system = {'dtype':'complex'}

	obj = cls(data,shape,size=size,dims=dims,samples=samples,system=system)


	print('Name : %s'%(name))
	obj.info()
	print()

	assert(obj() is None)


	# Reinit instance
	data = dict(obj)
	data['scale'] = 1
	shape = (hyperparameters['model']['D']**hyperparameters['model']['N'],)*2
	size = [1,4]
	dims = [hyperparameters['model']['N'],hyperparameters['model']['D']]
	samples = True
	system = {'dtype':'complex'}

	obj = cls(data,shape,size=size,dims=dims,samples=samples,system=system)


	print('Name : %s'%(name))
	obj.info()
	print()

	assert(allclose(obj(),old))

	return

if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	test_class(path,tol)
