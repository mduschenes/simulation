#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,time
from copy import deepcopy as deepcopy
from math import prod
	
# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))


from src.utils import jit,array,rand,arange,zeros,ones,eye,einsum,tensorprod,allclose,cos,sin,bound
from src.utils import gradient,hessian,fisher
from src.utils import norm,conjugate,dagger,dot,eig,nonzero,difference,maximum,argmax,abs,sort,sqrt,real,imag
from src.utils import pi,delim,arrays,scalars,epsilon,inplace,to_index,to_position
from src.iterables import getter,setter,permuter,namespace
from src.io import load,dump,join,exists

from src.quantum import Object,Operator,Pauli,State,Gate,Haar,Noise,Label,trotter
from src.optimize import Optimizer,Objective,Metric,Callback
from src.system import Dictionary,Dict

def test_object(path,tol):
	bases = {'Pauli':Pauli,'State':State,'Gate':Gate,'Haar':Haar,'Noise':Noise}
	arguments = {
		'Pauli': {
			'basis':'Pauli',
			'kwargs':dict(
				data=delim.join(['X','Y','Z']),operator=None,site=[0,1,2],string='XYZ',
				kwargs=dict(N=3,D=2,ndim=2,parameters=1,verbose=True),
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
		assert ((arguments[name]['basis'] in ['Haar','State','Gate','Noise']) or allclose(operator.data,
			tensorprod([base.basis[i]() for i in args['data'].split(delim)]))), "Operator.data %s != %r"%(operator.string,operator(operator.parameters))
		assert tuple(operator.operator) == tuple(args['data'].split(delim))

		for attr in kwargs:
			assert getattr(operator,attr)==kwargs[attr], "Operator.%s = %r != %r"%(attr,getattr(operator,attr),kwargs[attr])

		for attr in operator:
			print(attr,operator[attr])
		print()


		other = base(**args,**kwargs)

		for attr in other:
			assert attr in ['timestamp','logger'] or callable(operator[attr]) or ((operator[attr] == other[attr]) if not isinstance(operator[attr],arrays) else allclose(operator[attr],other[attr])), "Incorrect reinitialization %s %r != %r"%(attr,operator[attr],other[attr])
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
		raise Exception("Hyperparameters %s not loaded"%(path))

	hyperparameters = Dict(hyperparameters)

	model = load(hyperparameters.cls.model)
	system = hyperparameters.system

	model = model(**{**hyperparameters.model,**dict(parameters=hyperparameters.parameters,state=hyperparameters.state,noise=hyperparameters.noise),**dict(system=system)})

	parameters = model.parameters()

	t = time.time()
	parameters = rand(shape=parameters.shape,random='normal',bounds=[-1,1],key=1234)
	obj = model(parameters)
	print(parameters.shape,obj.shape,time.time()-t)


	for i in range(10):
		t = time.time()
		parameters = rand(shape=parameters.shape,random='normal',bounds=[-1,1],key=1234)
		obj = model(parameters)
		print(i,parameters.shape,obj.shape,time.time()-t)


	I = model.identity()
	objH = dagger(model(parameters,conj=False))
	objD = dagger(model(parameters))

	objobjH = dot(obj,objH)
	objHobj = dot(objH,obj)
	objobjD = dot(obj,objD)
	objDobj = dot(objD,obj)

	print('model(conj=True) - dagger(model())',(objH-objH).min(),(objH-objH).max())
	# print(objH-objH)
	# print()

	print('model() * model(conj=True) - I',(objobjH - I).min(),(objobjH - I).max())
	# print(objobjH - I)
	# print()

	print('model(conj=True) * model() - I',(objHobj - I).min(),(objHobj - I).max())
	# print(objHobj - I)
	# print()

	print('model() * dagger(model()) - I',(objobjD - I).min(),(objobjD - I).max())
	# print(objobjD - I)
	# print()

	print('dagger(model()) * model() - I',(objDobj - I).min(),(objDobj - I).max())
	# print(objDobj - I)
	# print()

	assert allclose(objobjH,I), "Incorrect unitarity model() * model(conj=True) != I"
	assert allclose(objHobj,I), "Incorrect unitarity model(conj=True) * model() != I"
	assert allclose(objobjD,I), "Incorrect unitarity model() * dagger(model()) != I"
	assert allclose(objDobj,I), "Incorrect unitarity dagger(model()) * model() != I"

	assert allclose(objH,objD), "Incorrect model(conj=True) != conj(model())"

	m,d,p = model.M,len(model),model.P
	identity = model.identity()
	parameters = rand(shape=model.parameters.shape,random='normal',bounds=[-1,1],key=1234)

	out = model(parameters)

	slices = [slice(None,None,1),slice(None,None,-1)][:p]
	data = [i for s in slices for i in model.data[s]]

	slices = array([i for s in [slice(None,None,1),slice(None,None,-1)][:p] for i in list(range(d))[s]])
	parameters = model.parameters(parameters)

	tmp = model.identity()
	for i in range(m*d*p):
		f = data[i%(d*p)]
		# print(i,data[i%(d*p)].string)
		tmp = dot(f(parameters[i]),tmp)

	assert allclose(out,tmp), "Incorrect model() from data()"


	tmp = model.identity()
	for i in range(m*d*p):
		f = lambda x: cos(pi/2*x)*identity + -1j*sin(pi/2*x)*data[i%(d*p)].data
		# print(i,data[i%(d*p)].string)
		tmp = dot(f(parameters[i]),tmp)

	assert allclose(out,tmp), "Incorrect model() from func()"

	print('Unitary Conditions Passed')

	return 

def test_parameters(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise Exception("Hyperparameters %s not loaded"%(path))

	hyperparameters = Dict(hyperparameters)

	model = load(hyperparameters.cls.model)

	hyperparams = hyperparameters.optimize
	system = hyperparameters.system

	model = model(**{**hyperparameters.model,**dict(parameters=hyperparameters.parameters,state=hyperparameters.state,noise=hyperparameters.noise),**dict(system=system)})

	parameters = model.parameters()
	variables = model.parameters(parameters)

	# Get parameters in shape (P*K,M)
	M,N = model.M,model.N
	parameters = parameters.reshape(-1,M)
	variables = variables.reshape(M,-1).T

	variables = variables[:variables.shape[0]//model.P]/model.coefficients

	print(list(model.parameters))
	print(parameters.round(6))
	print(variables.round(6))

	parameter = 'xy'

	assert parameter in model.parameters, "Incorrect parameters: missing %s"%(parameter)

	shape = parameters.shape
	category = model.parameters[parameter].category
	method = model.parameters[parameter].method	
	size = len(model.parameters[parameter].group)
	null = lambda parameters,**kwargs: parameters

	print('Parameters / Constraints :::',category,method,model.parameters[parameter].constraints(model.parameters()))

	if (method in [None,'unconstrained']):
		
		wrappers = [null,null]
		funcs = [null,null]
		kwargs = [{},{}]

	elif (method in ['constrained']):

		wrappers = [bound,bound]
		funcs = [
			lambda parameters,**kwargs: kwargs['scale'][0]*parameters[:parameters.shape[0]//2]*cos(kwargs['scale'][1]*parameters[parameters.shape[0]//2:]),
			lambda parameters,**kwargs: kwargs['scale'][0]*parameters[:parameters.shape[0]//2]*cos(kwargs['scale'][1]*parameters[parameters.shape[0]//2:] + kwargs['shift']),
		]
		kwargs = [{'scale':[1,2*pi],'shift':0},{'scale':[1,2*pi],'shift':-pi/2}]

	elif (method in ['bounded']):

		wrappers = [bound,bound]
		funcs = [null,null]
		kwargs = [{},{}]


	for i in range(size):
		locality = model.parameters[parameter].locality.get(model.parameters[parameter].group[i])
		if (locality in ['local',None]):
			index = arange(parameters.shape[0])
		elif (locality in ['global']):
			index = array([j for j in model.parameters[parameter].slices])
		
		if method in ['constrained']:
			slices = arange(model.parameters[parameter].slices.size//size)
		else:
			slices = arange(i*(model.parameters[parameter].slices.size//size),(i+1)*(model.parameters[parameter].slices.size//size))
		_slices = arange(0,i*(model.parameters[parameter].slices.size//size))

		features = parameters[index]

		wrapper = wrappers[i]
		func = funcs[i]
		kwds = kwargs[i]
		indices = slice(_slices.size,_slices.size+slices.size)
		vars = model.parameters[parameter].parameters*wrapper(func(features,**kwds))[slices]
		print(index,slices,_slices,indices,features.shape,parameters.shape,variables[indices].shape,vars.shape)
		print(vars)
		print(variables[indices])
		if (variables[indices].shape != vars.shape) or not allclose(variables[indices],vars):
			raise ValueError("Incorrect parameter initialization %d %r"%(i,model.parameters))
		print()
	print('Done')
	return



def test_logger(path,tol):
	cls = load('src.system.System')

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
		raise Exception("Hyperparameters %s not loaded"%(path))

	hyperparameters = Dict(hyperparameters)

	model = load(hyperparameters.cls.model)
	system = hyperparameters.system

	model = model(**{**hyperparameters.model,**dict(parameters=hyperparameters.parameters,state=hyperparameters.state,noise=hyperparameters.noise),**dict(system=system)})


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
	datas = trotter([d.data for d in model.data],P)
	sites = trotter([d.site for d in model.data],P)

	for i,(s,d,D,site) in enumerate(zip(string,data,datas,sites)):
		assert allclose(d,D), "data[%s,%d] incorrect"%(s,i)

	return

def test_initialization(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise Exception("Hyperparameters %s not loaded"%(path))

	hyperparameters = Dict(hyperparameters)

	model = load(hyperparameters.cls.model)
	label = load(hyperparameters.cls.label)

	hyperparams = hyperparameters.optimize
	system = hyperparameters.system

	model = model(**{**hyperparameters.model,**dict(parameters=hyperparameters.parameters,state=hyperparameters.state,noise=hyperparameters.noise),**dict(system=system)})
	label = label(**{**namespace(label,model),**hyperparameters.label,**dict(model=model,system=system)})

	parameters = model.parameters()
	kwargs = dict(verbose=True)

	metric = Metric(label=label,hyperparameters=hyperparams,system=system,**kwargs)


	def copier(model,metric):

		copy = Dictionary(
			model=Dictionary(func=model.__call__,data=model(),state=model.state,noise=model.noise,info=model.info,hermitian=model.hermitian,unitary=model.unitary),
			metric=Dictionary(func=metric.__call__,data=metric(model()),state=metric.label.state,noise=model.noise,info=metric.info,hermitian=metric.label.hermitian,unitary=metric.label.unitary),
			label=Dictionary(func=metric.label.__call__,data=metric.label(),state=metric.label.state,info=metric.info,hermitian=metric.label.hermitian,unitary=metric.label.unitary),
			)

		return copy

	copy = copier(model,metric)

	
	defaults = Dictionary(state=model.state,noise=model.noise,label=metric.label)


	tmp = Dictionary(state=False,noise=False,label=False)

	
	label = metric.label

	model.__initialize__(state=tmp.state,noise=tmp.noise)

	label.__initialize__(state=model.state)

	metric.__initialize__(model=model,label=label)

	tmp = copier(model,metric)

	model.__initialize__(state=defaults.state,noise=defaults.noise)

	label.__initialize__(state=defaults.state)

	metric.__initialize__(model=model,label=label)

	
	new = copier(model,metric)


	
	print('--- COPY INFO ---')
	copy.model.info(verbose=True)
	print()

	print('--- TMP INFO ---')
	tmp.model.info(verbose=True)
	print()

	print('--- NEW INFO ---')
	new.model.info(verbose=True)
	print()


	print('State model (hermitian: %s, unitary: %s)'%(copy.model.hermitian,copy.model.unitary))
	print(copy.model.data)

	print('State label (hermitian: %s, unitary: %s)'%(copy.label.hermitian,copy.label.unitary))
	print(copy.label.data)

	print('State state (hermitian: %s, unitary: %s)'%(copy.label.state.hermitian,copy.label.state.unitary))
	print(copy.label.state())

	print('Unitary model (hermitian: %s, unitary: %s)'%(tmp.model.hermitian,tmp.model.unitary))
	print(tmp.model.data)

	print('Unitary label (hermitian: %s, unitary: %s)'%(tmp.label.hermitian,tmp.label.unitary))
	print(tmp.label.data)

	print('State model (hermitian: %s, unitary: %s)'%(new.model.hermitian,new.model.unitary))
	print(new.model.data)

	print('State label (hermitian: %s, unitary: %s)'%(new.label.hermitian,new.label.unitary))
	print(new.label.data)

	print('State state (hermitian: %s, unitary: %s)'%(new.label.state.hermitian,new.label.state.unitary))
	print(new.label.state())


	UpsiU = copy.model.data
	U = tmp.model.data
	psi = copy.model.state()
	K = copy.model.noise()
	VpsiV = copy.label.data
	V = tmp.label.data

	if K is None:
		if psi is None:
			return
		elif psi.ndim == 1:
			UpsiUtmp = einsum('ij,j->i',U,psi,conjugate(U))
			VpsiVtmp = einsum('ij,j->i',V,psi,conjugate(V))
		elif psi.ndim == 2:
			UpsiUtmp = einsum('ij,jk,lk->il',U,psi,conjugate(U))
			VpsiVtmp = einsum('ij,jk,lk->il',V,psi,conjugate(V))		
	elif K is not None:
		return
		if psi is None:
			return
		elif psi.ndim == 1:
			return
		elif psi.ndim == 2:
			UpsiUtmp = einsum('uij,jk,kl,ml,unm->in',K,U,psi,conjugate(U),conjugate(K))
			VpsiVtmp = einsum('ij,jk,lk->il',V,psi,conjugate(V))		


	assert allclose(UpsiUtmp,UpsiU), "Incorrect model() re-initialization"
	assert allclose(VpsiVtmp,VpsiV), "Incorrect label() re-initialization"
	assert allclose(new.metric.data,copy.metric.data), "Incorrect metric() re-initialization"
	
	return

def test_hessian(path,tol):
	
	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise Exception("Hyperparameters %s not loaded"%(path))

	hyperparameters = Dict(hyperparameters)

	model = load(hyperparameters.cls.model)
	label = load(hyperparameters.cls.label)

	hyperparams = hyperparameters.optimize
	system = hyperparameters.system

	model = model(**{**hyperparameters.model,**dict(parameters=hyperparameters.parameters,state=hyperparameters.state,noise=hyperparameters.noise),**dict(system=system)})
	label = label(**{**namespace(label,model),**hyperparameters.label,**dict(model=model,system=system)})

	parameters = model.parameters()
	kwargs = dict(verbose=True)

	metric = Metric(label=label,hyperparameters=hyperparams,system=system,**kwargs)

	func = hessian(jit(lambda parameters: metric(model(parameters))))

	out = func(parameters)

	eigs = sort(abs(eig(out,compute_v=False,hermitian=True)))[::-1] if out is not None else None
	eigs = eigs/max(1,maximum(eigs)) if eigs is not None else None
	rank = nonzero(eigs,eps=50) if eigs is not None else None

	print(model.state.ndim,rank,eigs)

	return

def test_fisher(path,tol):

	out = []
	permutations = {'state.operator':['zero'],'state.ndim':[1,2],'state.parameters':[True],'noise.parameters':[False]}

	for kwargs in permuter(permutations):

		default = None
		hyperparameters = load(path,default=default)
		if hyperparameters is None:
			raise Exception("Hyperparameters %s not loaded"%(path))


		hyperparameters = Dict(hyperparameters)

		setter(hyperparameters,kwargs,delimiter=delim)

		model = load(hyperparameters.cls.model)
		label = load(hyperparameters.cls.label)

		hyperparams = hyperparameters.optimize
		system = hyperparameters.system

		model = model(**{**hyperparameters.model,**dict(parameters=hyperparameters.parameters,state=hyperparameters.state,noise=hyperparameters.noise),**dict(system=system)})	

		parameters = model.parameters()

		func = fisher(model,model.grad,shapes=(model.shape,(*parameters.shape,*model.shape)))

		tmp = func(parameters)

		eigs = sort(abs(eig(tmp,compute_v=False,hermitian=True)))[::-1] if tmp is not None else None
		eigs = eigs/max(1,maximum(eigs)) if eigs is not None else None
		rank = nonzero(eigs,eps=50) if eigs is not None else None

		out.append(eigs)

	for i in out:
		for j in out:
			assert allclose(i,j), "Incorrect Fisher Computation\n%r\n%r\n%r"%(norm(i-j)/min(norm(i),norm(j))/sqrt(i.size*j.size),i,j)

	print("Passed")
	
	return

# @pytest.mark.skip
def check_fisher(path,tol):

	def _fisher(model):

		def trotter(iterable,p):

			slices = [slice(None,None,1),slice(None,None,-1)]

			data = []
			for i in slices[:p]:
				data += iterable[i]

			return data

		def function(model):

			parameters = model.parameters(model.parameters())
			indices = model.parameters.indices		
			M,K,P = parameters.size, parameters.size//model.M, model.P

			data = trotter(model.data,P)

			out = model.identity()

			for i in range(M):
				out = dot(data[i%K](parameters[i]),out)

			return out

		def gradient(model):

			parameters = model.parameters(model.parameters())
			indices = model.parameters.indices
			M,K,P,G,L = parameters.size, parameters.size//model.M, model.P, model.parameters().size, model.parameters().size//model.M
			
			func = function(model)
			grad = zeros((G,*model.shape),dtype=model.dtype)
			state = model.state()
			data = trotter(model.data,P)

			U = model.identity()
			_U = func
			u,_u = model.identity(),model.identity()

			for i in range(M):

				if (i%K,i//K) not in indices:
					continue

				u,_u = _u,data[i%K](parameters[i])
				U = dot(u,U)
				_U = dot(_U,dagger(_u))

				H = model.coefficients*data[i%K].grad(parameters[i])

				tmp = dot(dot(_U,dot(H,U)),state)
				
				j = indices[(i%K,i//K)]

				grad = inplace(grad,j,tmp,'add')

			return grad


		func = function(model)
		grad = gradient(model)
		state = dot(func,model.state())
		parameters = model.parameters()
		G,dtype = parameters.size,parameters.dtype

		if not G:
			out = None
			return out

		out = zeros((G,G),dtype=dtype)

		for i in range(G):
			for j in range(G):

				tmp = 2*real(dot(dagger(grad[i]),grad[j]) - dot(dagger(state),grad[i])*dagger(dot(dagger(state),grad[j])))

				out = inplace(out,(i,j),tmp)

		return out


	out = []
	permutations = {'state.ndim':[1,1,2],'state.parameters':[True],'noise.parameters':[False]}
	permutations = permuter(permutations)
	n = len(permutations)

	for i in range(n):

		kwargs = permutations[i]

		default = None
		hyperparameters = load(path,default=default)
		if hyperparameters is None:
			raise Exception("Hyperparameters %s not loaded"%(path))

		hyperparameters = Dict(hyperparameters)

		setter(hyperparameters,kwargs,delimiter=delim)

		print('N: %d, M: %d, B: %s'%(hyperparameters.model.N,hyperparameters.model.M,hyperparameters.model.data.zz.site))

		model = load(hyperparameters.cls.model)
		label = load(hyperparameters.cls.label)

		hyperparams = hyperparameters.optimize
		system = hyperparameters.system

		model = model(**{**hyperparameters.model,**dict(parameters=hyperparameters.parameters,state=hyperparameters.state,noise=hyperparameters.noise),**dict(system=system)})	

		parameters = model.parameters()

		func = fisher(model,model.grad,shapes=(model.shape,(*parameters.shape,*model.shape)))
		_func = _fisher

		if i == (n-3):
			tmp = _func(model)
		else:
			tmp = func(parameters)

		size = model.parameters(model.parameters()).size
		length = len(model)
		indices = model.parameters.indices
		index = len(set(indices[i] for i in indices))
		shape = (index,size//length)
		dtype = int
		transform = zeros(tmp.shape,dtype=dtype)
		for i in indices:
			j = to_index(to_position(indices[i],shape[::-1]),shape[:])
			transform = inplace(transform,(j,indices[i]),1)
		print('-----')
		print(transform)
		print(tmp)

		tmp = dot(transform,dot(tmp,transform.T)) 


		eigs = eig(tmp,compute_v=False,hermitian=True) if tmp is not None else None
		rank = nonzero(eigs,eps=1e-10) if eigs is not None else None


		# if i == (n-1):
		# 	stats = {'N':model.N,'M':model.M,'Bndy':{'<ij>':'closed','>ij<':'open'}.get(hyperparameters.model.data.zz.site,'')}
		# 	data = {
		# 		'parameters_%s.npy'%('_'.join(tuple((''.join([stat,str(stats[stat])]) for stat in stats)))): model.parameters().reshape(-1,model.M),
		# 		'eig_%s.npy'%('_'.join(tuple((''.join([stat,str(stats[stat])]) for stat in stats)))): eigs,
		# 		}
		# 	directory = '~/Downloads/'

		# 	for file in data:
		# 		dump(data[file],join(directory,file))

		eigs = sort(abs(eigs))[::-1] if eigs is not None else None
		# eigs = eigs/max(1,maximum(eigs))

		print('-----')
		print(rank,eigs)
		print(tmp)
		print()

		out.append(eigs)

		# exit()

	for i in range(n):
		for j in range(n):
			assert (out[i] is None and out[j] is None) or allclose(out[i],out[j]), "Incorrect Fisher Computation (%d,%d):\n%r\n%r\n%r"%(i,j,out[i]-out[j],out[i],out[j]) #/min(norm(out[i]),norm(out[j]))/sqrt(out[i].size*out[j].size)

	print("Passed")
	
	return


def profile(funcs,*args,profile=True,**kwargs):
	import cProfile, pstats
	import snakeviz.cli
	
	if callable(funcs):
		funcs = [funcs]

	if not profile:
		for func in funcs:
			func(*args,**kwargs)
		return

	sort = ['cumtime']
	lines = 100
	file = 'stats.profile'

	for func in funcs:
		profiler = cProfile.Profile()
		profiler.enable()

		func(*args,**kwargs)

		profiler.disable()

		stats = pstats.Stats(profiler).sort_stats(*sort)
		stats.print_stats(lines)
		stats.dump_stats(filename=file)

	# snakeviz.cli.main([file])

	return


def check_machine_precision(path,tol):
	import matplotlib
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	import autograd.numpy as np
	import sympy as sp

	seed = 12931
	maxbits = 256
	maxdtype = 'complex%d'%(maxbits)
	maxftype = 'float%d'%(maxbits//2)
	np.random.seed(seed)

	def analytical(k,n,eps):
		
		error = np.zeros(k,dtype=maxftype)
		
		for i in range(k):
			# e = (np.array(n,dtype=dtype)**i)*((1 + np.sqrt(sum(((1+eps)**(j+1) - 1)**2 for j in range(n))))**(i+1) - 1)
			# e = (n**(i+1))*
			e = ((1+(n**(3/2))*eps)**(i+2) - 1)
			# e = ((1+eps)**(i+2) - 1)
			error[i] = e
			print(i,e.dtype)
		return error

	def numerical(k,n,eps):
		random = 'haar'
		info = np.finfo('complex%d'%(eps))
		bits = -np.floor(np.log10(info.eps))

		dtype = 'complex%d'%(eps)
		ftype = 'float%d'%(eps//2)
		norm = lambda A,ord=2: (((np.absolute(A,dtype=maxftype)**ord).sum(dtype=maxftype))**(1/ord)).real
		
		V = sp.Matrix([[sp.exp(sp.Mul(sp.I,2*sp.pi,sp.Rational(i*j,n))) for j in range(n)] for i in range(n)])/sp.sqrt(n)
		S = [sp.Rational(np.random.randint(1,i) if i>1 else 0,i) for i in np.random.randint(1,n**2,size=n)]
		D = lambda k=1: sp.diag(*(sp.exp(sp.Mul(sp.I,2*sp.pi,s,k)) for s in S))
		W = lambda k=1: V*D(k)*V.H
		N = lambda A,bits=4*maxbits: np.array(sp.N(A,bits),dtype=maxdtype)
		E = np.diag((1+10**(-bits))**(np.arange(n)+2)-1)

		A = N(W(),bits=bits)	
		C = norm(A)
		B = A

		error = np.zeros(k,dtype=maxftype)
		
		for i in range(k):
			B = np.matmul(A,B,dtype=dtype)# + 0.5*10**(-bits)*np.random.choice([-1,1],B.shape).astype(maxdtype) 
			# B = np.matmul(A,B,dtype=maxdtype) + np.matmul(A,np.matmul(E,B,dtype=maxdtype),dtype=maxdtype) # + 0.5*10**(-bits)*np.random.choice([-1,1],B.shape).astype(maxdtype) 
			# B = A*B
			e = norm(B - N(W(i+2)))/C
			# e = norm(np.array(sp.N(B,bits),dtype=maxdtype) - N(W(i+2)))/C
			error[i] = e
			print(i,e,dtype,error[i].dtype,bits)
		
		return error		

	def machine(k,n,eps):
		bits = int(eps*np.log10(2))
		dtype = 'float%d'%(eps//2)

		N = lambda a,bits=4*maxbits: np.asscalar(np.array(sp.N(a,bits),dtype=dtype))

		v = sp.Rational(np.random.randint(1,n),n)
		a = N(v)
		c = lambda b: N(sp.Mul(b*v))
		b = N(a)

		error = np.zeros(k,dtype=dtype)
		
		for i in range(k):
			e = np.abs(a*b - c(b))/c(b)
			b = a*b
			error[i] = e
			print(i,e)
		
		return error	

	n = 2**2
	k = int(1e2)
	K = list(range(1,k+1))
	L = 13
	samples = 1
	epsilon = np.logspace(-20,-20+L-1,L,dtype=maxftype).tolist()
	precision = [64,128,256]

	mplstyle = 'config/plot.mplstyle'
	with matplotlib.style.context(mplstyle):

		fig,ax = plt.subplots()
		plots = []
		for eps in epsilon:
			error = analytical(k,n,eps)
			plot = ax.plot(K,error,
				linewidth=4,
				label='$10^{-%d}$'%(np.round(-np.log10(eps))),
				color=plt.cm.viridis((epsilon.index(eps)+1)/len(epsilon))
				)
			plots.append(plot)

		for bits in precision:
			info = np.finfo('complex%d'%(bits))
			eps = -np.floor(np.log10(info.eps))
			print(info)
			error = [numerical(k,n,bits) for i in range(samples)]
			error,errorerr = np.mean(error,axis=0,dtype=maxftype),np.std(error,axis=0,dtype=maxftype)/np.sqrt(max(1,samples-1),dtype=maxftype)
			plot = ax.errorbar(K,error,yerr=errorerr,
				alpha=0.7,#(precision.index(bits)+1)/len(precision),
				linewidth=4,
				linestyle='--',
				color=['k','r','b'][precision.index(bits)],
				label=r'$%d~\textrm{bit}~ (\varepsilon \sim 10^{-%d})$'%(bits//2,eps)
				)
				
		# ax.axhline(1e-8,color='k',linestyle='--');
		# ax.axhline(1e-16,color='k',linestyle='--');
		fig.set_size_inches(10,10)
		ax.set_xlabel(r'$\textrm{Matmul Count}~k$')
		ax.set_ylabel(r'$\textrm{Matmul Error}~ \epsilon_{k}$')
		ax.set_yscale('log');
		ax.set_xscale('log');
		# ax.set_xlim(5e-1,5e6);
		# ax.set_xlim(5e-1,5e2);
		ax.set_xlim(5e-1,2e4);
		ax.set_ylim(1e-21,1e-3);
		# ax.set_xticks([1e0,1e2,1e4,1e6]);
		# ax.set_xticks([1e0,1e1,1e2]);
		ax.set_xticks([1e0,1e1,1e2,1e3,1e4]);
		ax.set_yticks([1e-20,1e-16,1e-12,1e-8,1e-4]);
		ax.minorticks_off();
		ax.grid(True,alpha=0.3);
		ax.set_title(r'$\frac{\norm{\widetilde{A^k} - A^k}}{\norm{A^k}} \leq \epsilon_{k} = (1 + \epsilon)^{k} - 1 ~\sim~ k \epsilon$',pad=30)
		legend = ax.legend(
			title=r'$\textrm{Machine Precision} ~ \varepsilon$' + '\n' + r'$A \in \textrm{U}(n) ~,~ \epsilon \sim O(n^{3/2})\varepsilon$',
			loc=[1.1,-0.1],ncol=1);
		legend.get_title().set_ha('center')
		fig.savefig('matmul_error.pdf',bbox_inches="tight",pad_inches=0.2)

	return


if __name__ == '__main__':
	path = 'config/settings.test.json'
	path = 'config/settings.json'
	path = 'config/settings.tmp.json'
	tol = 5e-8 

	func = test_hessian
	func = check_machine_precision
	func = test_object
	func = test_model
	func = test_parameters
	func = test_initialization
	func = check_fisher
	args = ()
	kwargs = dict(path=path,tol=tol,profile=False)
	profile(func,*args,**kwargs)