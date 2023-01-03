#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools
from copy import deepcopy as deepcopy
	
# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))


from src.states import State
from src.utils import jit,einsum,allclose,is_hermitian,is_unitary,delim
from src.utils import norm,dagger,cholesky
from src.iterables import getter,setter
from src.io import load,dump

# Logging
# from src.system import Logger
# name = __name__
# path = os.getcwd()
# file = 'logging.conf'
# conf = os.path.join(path,file)
# file = None #'log.log'
# logger = Logger(name,conf,file=file)

def test_model(path,tol):

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'],
		system=hyperparameters['system'])

	return 

def test_parameters(path,tol):

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'],
		system=hyperparameters['system'])

	parameters = model.parameters()
	variables = model.__parameters__(parameters)


	parameters = parameters.reshape(-1,model.dims[0])
	variables = variables.reshape(model.dims[0],-1).T

	shape = parameters.shape
	slices = tuple((slice(size) for size in shape))
	if all(model.parameters.hyperparameters.get(parameter,{}).get('method') in [None,'unconstrained'] for parameter in model.parameters.hyperparameters):
		# assert allclose(variables[slices],parameters), "Incorrect parameter initialization %r"%(model.parameters.hyperparameters)
		if not allclose(variables[slices],parameters):
			print(parameters)
			print(variables)
			raise ValueError("Incorrect parameter initialization %r"%(model.parameters.hyperparameters))


	return



def test_functions(path,tol):

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'],
		system=hyperparameters['system'])

	parameters = model.parameters()

	copy = {attr: deepcopy(getattr(model,attr)) for attr in ['state','noise','label']}
	attrs = ['noise.string','noise.scale','exponentiation']
	kwargs = {'initial':dict(),'test':dict(noise={'scale':0}),'alter':dict(noise={'scale':None},state={'scale':None}),'restore':dict(copy)}
	U = {}

	for name in kwargs:

		kwds = {}
		setter(kwds,copy,func=True,copy=True)
		setter(kwds,kwargs[name],func=True,copy=True)
		model.__functions__(**kwds)
		func = jit(model)
		u = func(parameters)
		U[name] = u

		if attrs:
			print('---- %s ----'%(name))
			print(u)
		for attr in attrs:
			print(attr,getter(model,attr,delimiter=delim))

		if u.ndim == 1:
			assert is_unitary(u), "Non-normalized state"
		elif u.ndim == 2:
			if getter(model,'noise.scale',delimiter=delim) is not None:
				assert is_hermitian(u), "Non-hermitian state"
			else:
				assert is_unitary(u), "Non-unitary operator"
		else:
			raise ValueError("ndim = %d != 1,2"%(u.ndim))	

		if attrs:
			print()

	assert allclose(U['initial'],U['restore']),"Incorrect restored noise"

	return



def test_class(path,tol):

	hyperparameters = load(path)

	classes = {'state':'src.states.State','noise':'src.noise.Noise','label':'src.operators.Operator'}

	keys = ['state','label','noise']

	for key in keys:

		cls = load(classes[key])

		# Variables
		shape = (hyperparameters['model']['D']**hyperparameters['model']['N'],)*2
		size = [3,2]
		dims = [hyperparameters['model']['N'],hyperparameters['model']['D']]
		system = {'dtype':'complex','verbose':True}
		kwargs = {kwarg : hyperparameters[key][kwarg] for kwarg in hyperparameters[key] if kwarg not in ['data','shape','size','dims','system']}

		# Initial instance
		data = {'scale':1,'key':key}

		obj = cls(data,shape,size=size,dims=dims,system=system,**kwargs)

		# obj.info()

		data = obj()

		if obj.ndim == 1:
			if key in ['state']: # state vector
				normalization = einsum('...i,...i->...',data,data.conj()).real/1
			elif key in ['label']: # label vector
				normalization = einsum('...i,...i->...',data,data.conj()).real/1
			elif key in ['noise']: # noise vector
				normalization = einsum('...i,...i->...',data,data.conj()).real/1
			else:
				raise ValueError("Incorrect key = %s and obj.ndim = %d"%(key,obj.ndim))
		elif obj.ndim == 2:
			if key in ['state']: # state matrix
				normalization = einsum('...ii->...',data).real/1
			elif key in ['label']: # label matrix 
				normalization = einsum('...ij,...ij->...',data,data.conj()).real/obj.n
			elif key in ['noise']: # noise matrix
				normalization = einsum('...ij,...ij->...',data,data.conj()).real/obj.n
			else:
				raise ValueError("Incorrect key = %s and obj.ndim = %d"%(key,obj.ndim))
		elif obj.ndim == 3:
			if key in ['noise']:
				normalization = einsum('...uij,...uij->...',data.conj(),data).real/obj.n
			else:
				raise ValueError("Incorrect key = %s and obj.ndim = %d"%(key,obj.ndim))

		else:
			raise AssertionError("Incorrect obj.ndim = %d"%(obj.ndim))

		assert(allclose(1,normalization)),"Incorrectly normalized obj: %0.5e"%(normalization)


		# Identical instance
		old = obj()

		data = dict(obj)

		obj = cls(data,shape,size=size,dims=dims,system=system)


		# obj.info()

		assert(allclose(obj(),old))


		# Difference instance
		data = dict(obj)
		data['scale'] = None
		data['logger'] = 'log.txt'
		data['system']['cleanup'] = True

		obj = cls(data,shape,size=size,dims=dims,system=system)


		# obj.info()

		assert(obj() is None)


		# Reinit instance
		data = dict(obj)
		data['scale'] = 1

		obj = cls(data,shape,size=size,dims=dims,system=system)


		# obj.info()

		assert(allclose(obj(),old))

	return

if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	# test_class(path,tol)
	# test_model(path,tol)
	# test_functions(path,tol)
