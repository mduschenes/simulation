#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,time
from copy import deepcopy as deepcopy
	
# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))


from src.utils import jit,array,rand,arange,zeros,ones,eye,einsum,tensorprod,allclose,is_hermitian,is_unitary,delim,cos,sin,sigmoid
from src.utils import norm,conj,dagger,dot,cholesky,trotter,expm,fisher,eig,difference,maximum,argmax,abs,sort
from src.utils import pi,delim,arrays,scalars,namespace
from src.iterables import getter,setter
from src.io import load,dump,exists

from src.quantum import Object,Operator,Pauli,State,Gate,Haar,Noise

def test_object(path,tol):
	bases = {'Pauli':Pauli,'State':State,'Gate':Gate,'Haar':Haar,'Noise':Noise}
	arguments = {
		'Pauli': {
			'basis':'Pauli',
			'kwargs':dict(
				data=delim.join(['X','Y','Z']),operator=None,site=[0,1,2],string='XYZ',
				kwargs=dict(N=3,D=2,ndim=2,parameters=None,verbose=True),
			),
		},
		'Gate': {
			'basis':'Gate',
			'kwargs':dict(
				data=delim.join(['I','CNOT']),operator=None,site=[0,1,2],string='CX',
				kwargs=dict(N=3,D=2,ndim=2,verbose=True),
			),
		},
		'Haar':{
			'basis':'Haar',
			'kwargs': dict(
				data=delim.join(['haar']),operator=None,site=[0,1,2],string='U',
				kwargs=dict(N=3,D=2,ndim=2,seed=1,reset=1,verbose=True),
			),
		},
		'Psi':{
			'basis':'State',
			'kwargs': dict(
				data=delim.join(['minus']),operator=None,site=[0,1],string='-',
				kwargs=dict(N=2,D=2,ndim=1,seed=1,reset=1,verbose=True),
			),
		},
		'Noise':{
			'basis':'Noise',
			'kwargs': dict(
				data=delim.join(['phase']),operator=None,site=[0,1],string='K',
				kwargs=dict(N=2,D=2,ndim=3,parameters=0.25,verbose=True),
			),
		},
	}

	for name in arguments:
	
		base = bases[arguments[name]['basis']]
		args = arguments[name]['kwargs']
		kwargs = args.pop('kwargs',{})

		operator = Operator(**args,**kwargs)

		assert operator.string == args['string'], "Operator.string = %s != %s"%(operator.string,args['string'])
		assert ((arguments[name]['basis'] in ['Haar','State','Gate','Noise']) or allclose(operator(operator.parameters),
			tensorprod([base.basis[i]() for i in args['data'].split(delim)]))), "Operator.data != %r"%(operator(operator.parameters))
		assert tuple(operator.operator) == tuple(args['data'].split(delim))

		for attr in kwargs:
			assert getattr(operator,attr)==kwargs[attr], "Operator.%s = %r != %r"%(attr,getattr(operator,attr),kwargs[attr])

		for attr in operator:
			print(attr,operator[attr])
		print()


		other = base(**args,**kwargs)

		for attr in other:
			assert attr in ['timestamp','logger'] or ((operator[attr] == other[attr]) if not isinstance(operator[attr],arrays) else allclose(operator[attr],other[attr])), "Incorrect reinitialization %s %r != %r"%(attr,operator[attr],other[attr])
		assert allclose(operator(operator.parameters),other(other.parameters))
		
		args.update(dict(data=None))
		operator = base(**args,**kwargs)
		assert operator(operator.parameters) is None

		args.update(dict(data=None,operator=None))
		operator = base(**args,**kwargs)
		assert operator(operator.parameters) is None
		
		print()

	operator = Operator(verbose=True)
	print(type(operator),operator,operator(operator.parameters),operator.operator,operator.site,operator.string,operator.parameters,operator)

	operator = Operator('I',N=3,verbose=True)
	print(type(operator),operator,operator(operator.parameters),operator.operator,operator.site,operator.string,operator.parameters,operator.shape)


	return


def test_model(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

	cls = {attr: load(hyperparameters['class'][attr]) for attr in hyperparameters.get('class',{})}


	model = cls.pop('model')
	kwargs = {
		**hyperparameters.get('model',{}),
		**{attr:hyperparameters[attr] for attr in ['parameters','state','noise','system']}
		}

	model = model(**kwargs)

	parameters = model.parameters()

	t = time.time()
	parameters = rand(shape=(len(model),model.M),random='normal',bounds=[-1,1],key=1234)
	print(parameters.shape)	
	obj = model(parameters)
	print(time.time()-t)

	t = time.time()
	parameters = rand(shape=(len(model),model.M),random='normal',bounds=[-1,1],key=1234)
	print(parameters.shape)
	obj = model(parameters)
	print(time.time()-t)


	I = model.identity()
	objH = model(parameters,conj=True)
	objD = dagger(model(parameters))

	objobjH = dot(obj,objH)
	objHobj = dot(objH,obj)
	objobjD = dot(obj,objD)
	objDobj = dot(objD,obj)

	assert allclose(objobjH,I), "Incorrect unitarity model() * model(conj=True) != I"
	assert allclose(objHobj,I), "Incorrect unitarity model(conj=True) * model() != I"
	assert allclose(objobjD,I), "Incorrect unitarity model() * dagger(model()) != I"
	assert allclose(objDobj,I), "Incorrect unitarity dagger(model()) * model() != I"

	assert allclose(objH,objD), "Incorrect model(conj=True) != conj(model())"

	print('All Passed')
	return
	m,d,p = model.M,len(model),model.P
	identity = model.identity()
	parameters = rand(shape=model.parameters.shape,random='normal',bounds=[-1,1],key=1234)

	out = model(parameters)

	slices = [slice(None,None,1),slice(None,None,-1)][:p]
	data = [i for s in slices for i in model.data[s]]

	slices = array([i for s in [slice(None,None,1),slice(None,None,-1)][:p] for i in list(range(d))[s]])
	parameters = (model.coefficients*model.parameters(parameters))[slices].T.ravel()

	tmp = model.identity()
	for i in range(m*d*p):
		f = data[i%(d*p)]
		print(i,data[i%(d*p)].string)
		tmp = dot(f(parameters[i]),tmp)

	assert allclose(out,tmp), "Incorrect model() from data()"


	tmp = model.identity()
	for i in range(m*d*p):
		f = lambda x: cos(pi*x)*identity + -1j*sin(pi*x)*data[i%(d*p)].data
		print(i,data[i%(d*p)].string)
		tmp = dot(f(parameters[i]),tmp)

	assert allclose(out,tmp), "Incorrect model() from func()"


	return 

def test_parameters(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

	cls = load(hyperparameters['class']['parameters'])

	model = hyperparameters['model']
	system = hyperparameters['system']


	parameters = cls(**{**model,**hyperparameters.get('parameters',{}),**dict(system=system)})

	return
	model = cls(**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		label=hyperparameters['label'],
		system=hyperparameters['system'])

	parameters = model.parameters()
	variables = model.parameters(parameters)

	print(model.data)
	print(model.parameters)

	# Get parameters in shape (P*K,M)
	M,N = model.M,model.N
	parameters = parameters.reshape(-1,model.dims[0])
	variables = variables.reshape(model.dims[0],-1).T

	print(parameters.round(3))
	print(variables.round(6))

	shape = parameters.shape
	slices = tuple((slice(size) for size in shape))
	parameter = 'xy'	
	if (model.parameters.hyperparameters.get(parameter,{}).get('method') in [None,'unconstrained']):
		vars = parameters
		if not allclose(variables[slices],vars):
			print(vars)
			print(variables[slices])
			raise ValueError("Incorrect parameter initialization %r"%(model.parameters.hyperparameters))
	else:
		G = len(model.parameters.hyperparameters[parameter]['group'])
		wrapper = sigmoid
		funcs = [cos,sin]
		scale = [model.parameters.hyperparameters[parameter]['scale'],2*pi]
		features = wrapper(parameters.reshape(G,shape[0]//G,*shape[1:]))

		for i,func in zip(range(G),funcs):
			slices = slice(i*N,(i+1)*N)
			vars = scale[0]*features[0]*func(scale[1]*features[1])
			if not allclose(variables[slices],vars):
				print(vars)
				print(variables[slices])
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

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

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

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

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

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

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
		kwargs.update({'scale':1,'key':key,'cls':{'tau':1}})
		data = {}
		obj = cls(data,shape,size=size,dims=dims,system=system,**kwargs)

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
		data.update(kwargs)

		obj = cls(data,shape,size=size,dims=dims,system=system)

		print(key)
		print('orig',copydata)
		print('----')
		print('identical',obj())
		print()
		print(copy['cls'],copy['scale'])
		print(obj.cls,obj.scale)
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

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

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
	
	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

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


def profile(func,*args,profile=True,**kwargs):
	import cProfile, pstats
	import snakeviz.cli
	
	if not profile:
		func(*args,**kwargs)
		return

	sort = ['cumtime']
	lines = 100
	file = 'stats.profile'

	profiler = cProfile.Profile()
	profiler.enable()

	func(*args,**kwargs)

	profiler.disable()

	stats = pstats.Stats(profiler).sort_stats(*sort)
	stats.print_stats(lines)
	stats.dump_stats(filename=file)

	# snakeviz.cli.main([file])

	return

if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 

	# func = test_object
	func = test_model
	args = ()
	kwargs = dict(path=path,tol=tol,profile=False)
	profile(func,*args,**kwargs)

	# test_object(path,tol)
	# test_model(path,tol)
	# test_parameters(path,tol)
	# test_call(path,tol)
	# test_data(path,tol)
	# test_logger(path,tol)
	# test_class(path,tol)
	# test_normalization(path,tol)
	# test_fisher(path,tol)
