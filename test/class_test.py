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


from src.utils import jit,array,einsum,tensorprod,allclose,is_hermitian,is_unitary,delim
from src.utils import norm,dagger,cholesky,trotter,expm,fisher,eig,difference,maximum,argmax,abs,sort
from src.iterables import getter,setter
from src.io import load,dump,exists

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



def test_logger(path,tol):
	cls = load('src.system.Object')

	data = None
	shape = None
	system = {'logger':'log.txt','cleanup':1}

	obj = cls(data,shape,system=system)


	data = None
	shape = None
	system = {'logger':'log.log','cleanup':1}

	obj = cls(data,shape,system=system)

	# assert not exists(system['logger']), "Incorrect cleanup"

	return


def test_data(path,tol):

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'],
		system=hyperparameters['system'])

	basis = {
		'I':array([[1,0],[0,1]],dtype=model.dtype),
		'X':array([[0,1],[1,0]],dtype=model.dtype),
		'Y':array([[0,-1j],[1j,0]],dtype=model.dtype),
		'Z':array([[1,0],[0,-1]],dtype=model.dtype),
		}
	default = 'I'

	N = model.N
	P = model.P

	string = [
		*[[O if k in [i] else default for k in range(N)]
			for O in ['X','Y','Z']
			for i in range(N)
			],
		*[[O if k in [i,j] else default for k in range(N)]
			for O in ['Z']
			for i in range(N)
			for j in range(N)
			if i<j
			],
		]

	data = [tensorprod(array([basis[i] for i in s])) for s in string]
	identity = tensorprod(array([basis[default]]*N))

	assert allclose(model.identity(),identity), "Incorrect model identity"

	data = trotter(data,P)
	string = trotter(string,P)
	datas = trotter([d() for d in model.data],P)
	strings = trotter([d.operator for d in model.data],P)
	sites = trotter([d.site for d in model.data],P)

	for i,(s,S,d,D,site) in enumerate(zip(string,strings,data,datas,sites)):
		assert allclose(d,D), "data[%s,%d] incorrect"%(s,i)

	return

def test_class(path,tol):

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
	attrs = ['noise.string','noise.scale','state.scale','exponentiation']
	kwargs = {'initial':dict(),'noisy':dict(noise={'scale':0},state={'scale':1}),'noiseless':dict(noise=False,state=False),'restore':dict(copy)}
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
			print('Normalized state')
			assert is_unitary(u), "Non-normalized state"
		elif u.ndim == 2:
			if getter(model,'state',delimiter=delim)() is not None:
				print('Hermitian noisy state')
				assert is_hermitian(u), "Non-hermitian state"
			else:
				print('Unitary noiseless operator')
				assert is_unitary(u), "Non-unitary operator"
		else:
			raise ValueError("ndim = %d != 1,2"%(u.ndim))	

		if attrs:
			print()

	assert allclose(U['initial'],U['restore']),"Incorrect restored obj"

	return




def test_normalization(path,tol):

	hyperparameters = load(path)

	classes = {'state':'src.states.State','noise':'src.noise.Noise','label':'src.operators.Gate'}

	keys = ['state','label','noise']

	for key in keys:

		cls = load(classes[key])

		# Variables
		shape = (hyperparameters['model']['D']**hyperparameters['model']['N'],)*2
		size = [1,1]
		dims = [hyperparameters['model']['N'],hyperparameters['model']['D']]
		system = {'dtype':'complex','verbose':True,'cleanup':True}
		kwargs = {kwarg : hyperparameters[key][kwarg] for kwarg in hyperparameters[key] if kwarg not in ['data','shape','size','dims','system']}

		# Initial instance
		kwargs.update({'scale':1,'key':key})
		data = {}
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

		copy = deepcopy(dict(obj))
		copydata = obj()

		# Identical instance
		data = dict(copy)

		obj = cls(data,shape,size=size,dims=dims,system=system)

		print(key)
		print('orig',copydata)
		print('----')
		print('identical',obj())
		print()
		# for attr in data:
		# 	print(attr,data[attr])
		# 	print()
		# obj.info()

		assert(allclose(obj(),copydata)), "Incorrect identical initialization"


		# Difference instance
		data = dict(copy)
		data['scale'] = None
		data['logger'] = 'log.txt'

		obj = cls(data,shape,size=size,dims=dims,system=system)


		# obj.info()
		print('None',obj(),obj.logger.file,obj.logger.cleanup)
		print()

		assert(obj() is None),"Incorrect data set to None"


		# Reinit instance
		data = dict(copy)
		data['scale'] = 1

		obj = cls(data,shape,size=size,dims=dims,system=system)


		# obj.info()
		print('reinit',obj(),obj.cleanup)
		print()
		print()

		assert(allclose(obj(),copydata)), "Incorrect reinitialization"

	return


def test_call(path,tol):

	hyperparameters = load(path)

	cls = load(hyperparameters['class']['model'])

	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'],
		system=hyperparameters['system'])

	parameters = model.parameters()


	parameters = model.parameters()
	variables = model.__parameters__(parameters)
	coefficients = model.coefficients
	data = array(trotter([data() for data in model.data],model.P))
	identity = model.identity()

	params = parameters.reshape(-1,model.dims[0])
	vars = variables.reshape(model.dims[0],-1).T

	_out = expm(coefficients*variables,data,identity)

	out = model(parameters)

	assert allclose(_out,out), "Incorrect model function"

	return 

def test_fisher(path,tol):
	hyperparameters = load(path)

	cls = {attr: load(hyperparameters['class'][attr]) for attr in hyperparameters['class']}

	model = cls['model'](**hyperparameters['model'],
			parameters=hyperparameters['parameters'],
			state=hyperparameters['state'],
			noise=hyperparameters['noise'],
			label=hyperparameters['label'],
			system=hyperparameters['system'])

	parameters = model.parameters()


	func = fisher(model,shapes=(model.shape,(*model.dimensions,*model.shape)))

	out = func(parameters)

	eigs = sort(abs(eig(func(parameters),compute_v=False,hermitian=True)))[::-1]
	eigs = eigs/max(1,maximum(eigs))

	rank = sort(abs(eig(func(parameters),compute_v=False,hermitian=True)))[::-1]
	rank = argmax(abs(difference(rank)/rank[:-1]))+1						

	print(eigs)
	print(rank)

	return


if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	# test_parameters(path,tol)
	# test_call(path,tol)
	# test_data(path,tol)
	# test_logger(path,tol)
	# test_class(path,tol)
	# test_model(path,tol)
	# test_normalization(path,tol)
	test_fisher(path,tol)
