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


from src.utils import jit,forloop,switch,array,rand,arange,zeros,ones,eye,einsum,tensorprod,allclose,cos,sin,bound
from src.utils import gradient,hessian,fisher
from src.utils import norm,conjugate,dagger,dot,eig,nonzero,difference,maximum,argmax,abs,sort,sqrt,real,imag
from src.utils import pi,delim,arrays,scalars,epsilon,inplace,to_index,to_position
from src.iterables import getter,setter,permuter,namespace,getattrs,setattrs
from src.io import load,dump,join,exists

from src.quantum import Object,Operator,Pauli,State,Gate,Haar,Noise,Label,trotter,compile,variables
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
		'Haar':{
			'basis':'Haar',
			'kwargs': dict(
				data=delim.join(['haar']*3),operator=None,site=[0,1,2],string='U',
				kwargs=dict(N=3,D=2,ndim=2,seed=1,parameters=1,verbose=True),
			),
		},
		'Psi':{
			'basis':'State',
			'kwargs': dict(
				data=delim.join(['minus']*2),operator=None,site=[0,1],string='-',
				kwargs=dict(N=2,D=2,ndim=1,seed=1,parameters=1,verbose=True),
			),
		},
		'Noise':{
			'basis':'Noise',
			'kwargs': dict(
				data=delim.join(['phase']*2),operator=None,site=[0,1],string='K',
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
		assert tuple(operator.operator) == tuple(args['data'].split(delim)),"%r != %r"%(tuple(operator.operator),tuple(args['data'].split(delim)))

		for attr in kwargs:
			if attr in ['parameters']:
				assert getattr(operator,attr)()==kwargs[attr], "Operator.%s = %r != %r"%(attr,getattr(operator,attr),kwargs[attr])
			else:
				assert getattrs(operator,attr)==kwargs[attr], "Operator.%s = %r != %r"%(attr,getattr(operator,attr),kwargs[attr])


	operator = Operator(data='haar',N=1,D=2,ndim=1,parameters=1,verbose=True)
	print(type(operator),operator,operator(operator.parameters(),operator.identity),operator.operator,operator.site,operator.string,operator.parameters,operator)

	operator = Operator('I',N=3,verbose=True)
	print(operator)
	print(type(operator),operator,operator(operator.parameters(),operator.identity),operator.operator,operator.site,operator.string,operator.parameters,operator.shape)


	print("Passed")
	return


def test_model(path,tol):

	default = None
	settings = load(path,default=default)
	if settings is None:
		raise Exception("settings %s not loaded"%(path))

	settings = Dict(settings)


	model = load(settings.cls.model)
	state = load(settings.cls.state)
	label = load(settings.cls.label)
	callback = load(settings.cls.callback)


	hyperparameters = settings.optimize
	system = settings.system

	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})
	callback = callback(**{**namespace(callback,model),**settings.callback,**dict(model=model,system=system)})

	label.__initialize__(state=state)
	model.__initialize__(state=state)

	parameters = model.parameters()
	state = model.state()


	t = time.time()
	parameters = rand(parameters.size,random='normal',bounds=[-1,1],key=1234)
	obj = model(parameters,model.identity)
	print(parameters.shape,obj.shape,time.time()-t)


	for i in range(10):
		t = time.time()
		parameters = rand(shape=parameters.shape,random='normal',bounds=[-1,1],key=1234)
		obj = model(parameters,state)
		print(i,parameters.shape,obj.shape,time.time()-t)


	if (not model.unitary) and (not model.hermitian):
		return


	if state is None and model.unitary:

		m,d,p = model.M,len(model),model.P
		tau = model.tau
		data = model.data
		state = model.identity
		identity = model.identity
		parameters = rand(shape=model.parameters.shape,random='normal',bounds=[-1,1],key=1234)

		out = model(parameters,state)

		data = compile(data,period=p)

		k = len(data)

		print("Doing model() test",m,d,p)

		tmp = state
		for i in range(m*k):
			f = data[i%k]
			tmp = f(parameters[i//k],tmp)

		assert allclose(out,tmp), "Incorrect model() from data()"
		

	print('Unitary Conditions Passed')

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
	settings = load(path,default=default)
	if settings is None:
		raise Exception("settings %s not loaded"%(path))

	settings = Dict(settings)

	model = load(settings.cls.model)
	state = load(settings.cls.state)
	label = load(settings.cls.label)

	hyperparameters = settings.optimize
	system = settings.system
	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})

	label.__initialize__(state=state)	
	model.__initialize__(state=state)


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

	assert allclose(model.identity,identity), "Incorrect model identity"

	data = trotter(data,P)
	string = trotter(string,P)
	datas = trotter([model.data[i].data for i in model.data if model.data[i].unitary],P)
	sites = trotter([model.data[i].site for i in model.data if model.data[i].unitary],P)

	for i,(s,d,D,site) in enumerate(zip(string,data,datas,sites)):
		assert allclose(d,D), "data[%s,%d] incorrect"%(s,i)
	print('Passed')
	return

def test_initialization(path,tol):

	default = None
	settings = load(path,default=default)
	if settings is None:
		raise Exception("settings %s not loaded"%(path))

	settings = Dict(settings)

	model = load(settings.cls.model)
	state = load(settings.cls.state)
	label = load(settings.cls.label)

	hyperparameters = settings.optimize
	system = settings.system
	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})

	label.__initialize__(state=state)	
	model.__initialize__(state=state)

	parameters = model.parameters()
	kwargs = dict(verbose=True)

	metric = Metric(state=state,label=label,hyperparameters=hyperparameters,system=system,**kwargs)

	def copier(model,metric,state,label):

		copy = Dictionary(
			model=Dictionary(func=model.__call__,data=model(model.parameters()),state=state,noise=[model.data[i] for i in model.data if (model.data[i] is not None) and (not model.data[i].unitary)],info=model.info,hermitian=model.hermitian,unitary=model.unitary),
			metric=Dictionary(func=metric.__call__,data=metric(model(model.parameters())),state=metric.state,noise=[model.data[i] for i in model.data if (model.data[i] is not None) and (not model.data[i].unitary)],info=metric.info,hermitian=label.hermitian,unitary=label.unitary),
			label=Dictionary(func=label.__call__,data=label(state=state()),state=state,info=label.info,hermitian=label.hermitian,unitary=label.unitary),
			state=Dictionary(func=state.__call__,data=state(),state=state,info=state.info,hermitian=state.hermitian,unitary=state.unitary),
			)

		return copy

	copy = copier(model,metric,state,label)

	
	defaults = Dictionary(state=state,data={i: model.data[i].data for i in model.data if (model.data[i] is not None) and (not model.data[i].unitary)},label=metric.label)


	tmp = Dictionary(state=False,data={i: model.data[i].data if (model.data[i].unitary) else False for i in model.data},label=False)

	
	label.__initialize__(state=tmp.state)	

	model.__initialize__(state=tmp.state,data=tmp.data)

	metric.__initialize__(model=model,label=label)

	tmp = copier(model,metric,state,label)

	label.__initialize__(state=defaults.state)

	model.__initialize__(state=defaults.state,data=defaults.data)

	metric.__initialize__(model=model,label=label)

	
	new = copier(model,metric,state,label)


	
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
	psi = copy.state.data
	K = copy.model.noise[-1].data
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
		#TODO: Implement test for multiple layers of noise 
		if psi is None:
			return
		elif psi.ndim == 1:
			return
		elif psi.ndim == 2 and model.M == 1:
			UpsiUtmp = einsum('uij,jk,kl,ml,unm->in',K,U,psi,conjugate(U),conjugate(K))
			VpsiVtmp = einsum('ij,jk,lk->il',V,psi,conjugate(V))		
		else:
			return


	assert allclose(UpsiUtmp,UpsiU), "Incorrect model() re-initialization"
	assert allclose(VpsiVtmp,VpsiV), "Incorrect label() re-initialization"
	assert allclose(new.metric.data,copy.metric.data), "Incorrect metric() re-initialization"
	print("Passed")
	return

def test_hessian(path,tol):
	
	default = None
	settings = load(path,default=default)
	if settings is None:
		raise Exception("settings %s not loaded"%(path))

	settings = Dict(settings)

	model = load(settings.cls.model)
	state = load(settings.cls.state)
	label = load(settings.cls.label)

	hyperparameters = settings.optimize
	system = settings.system
	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})

	label.__initialize__(state=state)
	model.__initialize__(state=state)

	parameters = model.parameters()
	kwargs = dict(verbose=True)

	metric = Metric(state=state,label=label,hyperparameters=hyperparameters,system=system,**kwargs)

	func = hessian(jit(lambda parameters: metric(model(parameters))))

	out = func(parameters)

	eigs = sort(abs(eig(out,compute_v=False,hermitian=True)))[::-1] if out is not None else None
	eigs = eigs/max(1,maximum(eigs)) if eigs is not None else None
	rank = nonzero(eigs,eps=50) if eigs is not None else None

	print(state.ndim,rank,eigs)

	print('Passed')

	return

def test_fisher(path,tol):

	out = []
	permutations = {'state.operator':['zero'],'state.ndim':[1,2]}

	for kwargs in permuter(permutations):

		print(kwargs)

		default = None
		settings = load(path,default=default)
		if settings is None:
			raise Exception("settings %s not loaded"%(path))


		settings = Dict(settings)

		setter(settings,kwargs,delimiter=delim)

		model = load(settings.cls.model)
		state = load(settings.cls.state)
		label = load(settings.cls.label)

		hyperparameters = settings.optimize
		system = settings.system
		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
		label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})

		label.__initialize__(state=state)
		model.__initialize__(state=state)
		
		parameters = model.parameters()
		state = model.state()

		func = fisher(model,model.grad,shapes=(model.state.shape,(parameters.size,*model.state.shape)),hermitian=model.state.hermitian,unitary=model.unitary)

		tmp = func(parameters=parameters,state=state)

		eigs = sort(abs(eig(tmp,compute_v=False,hermitian=True)))[::-1] if tmp is not None else None
		eigs = eigs/max(1,maximum(eigs)) if eigs is not None else None
		rank = nonzero(eigs,eps=50) if eigs is not None else None

		out.append(eigs)

		print(eigs)
		print()

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

			data = trotter([model.data[i] for i in model.data],P)

			out = model.identity

			for i in range(M):
				out = dot(data[i%K](parameters[i]),out)

			return out

		def gradient(model):

			parameters = model.parameters(model.parameters())
			indices = model.parameters.indices
			M,K,P,G,L = parameters.size, parameters.size//model.M, model.P, model.parameters().size, model.parameters().size//model.M

			# indexer = array([i for i in indices])
			# indices = array([indices[i] for i in indices])
			indices = array([indices.get(i,-1) for i in range(M)])
			
			func = function(model)
			state = state()
			data = trotter([jit(model.data[i]) for i in model.data],P)
			grads = trotter([jit(model.data[i].grad) for i in model.data],P)

			U = model.identity
			_U = func
			u,_u = model.identity,model.identity

			grad = zeros((G,*model.shape),dtype=model.dtype)

			def func(i,out):

				grad,U,_U,u,_u = out

				# i,j = indexer[i],indices[i]
				i,j = i,indices[i]

				u,_u = _u,switch(i%K,data,parameters[i])
				U = dot(u,U)
				_U = dot(_U,dagger(_u))

				H = model.coefficients[i%K]*switch(i%K,grads,parameters[i])

				tmp = dot(dot(_U,dot(H,U)),state)*(i > -1)
				
				grad = inplace(grad,j,tmp,'add')

				out = grad,U,_U,u,_u

				return out

			out = grad,U,_U,u,_u
			indexes = [0,M] 
			# for i in range(*indexes):
			# 	out = func(i,out)
			out = forloop(*indexes,func,out)
			grad,U,_U,u,_u = out

			return grad


		func = function(model)
		grad = gradient(model)
		state = dot(func,state())
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
		settings = load(path,default=default)
		if settings is None:
			raise Exception("settings %s not loaded"%(path))

		settings = Dict(settings)

		setter(settings,kwargs,delimiter=delim)

		print('N: %d, M: %d, B: %s'%(settings.model.N,settings.model.M,settings.model.data.zz.site))

		model = load(settings.cls.model)
		state = load(settings.cls.state)
		label = load(settings.cls.label)

		hyperparameters = settings.optimize
		system = settings.system
		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
		label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})

		label.__initialize__(state=state)
		model.__initialize__(state=state)

		parameters = model.parameters()

		func = fisher(model,model.grad_analytical,shapes=(model.shape,(*parameters.shape,*model.shape)),hermitian=state.hermitian,unitary=model.unitary)

		_func = _fisher

		if i == (n-3):
			tmp = _func(model)
		else:
			tmp = func(parameters)

		size = model.parameters(model.parameters()).size
		length = model.parameters().size//model.M
		indices = model.parameters.indices
		index = len(set(indices[i] for i in indices))
		shape = (length,model.M)
		dtype = int
		# transform = zeros(tmp.shape,dtype=dtype)
		# for i in indices:
		# 	j = to_index(to_position(indices[i],shape),shape)
		# 	transform = inplace(transform,(j,indices[i]),1)
		# print('-----')
		# print(transform)
		# print(tmp)

		# tmp = dot(transform,dot(tmp,transform.T)) 


		eigs = eig(tmp,compute_v=False,hermitian=True) if tmp is not None else None
		rank = nonzero(eigs,eps=1e-10) if eigs is not None else None


		# if i == (n-1):
		# 	stats = {'N':model.N,'M':model.M,'Bndy':{'<ij>':'closed','>ij<':'open'}.get(settings.model.data.zz.site,'')}
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
		# for i in indices:
		# 	print(i,indices[i])
		print(rank,eigs)
		# print(tmp)
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
	path = 'config/settings.tmp.json'
	path = 'config/settings.json'

	tol = 5e-8 

	func = check_machine_precision
	func = check_fisher
	func = check_fisher
	func = test_object
	func = test_data
	func = test_initialization
	func = test_hessian
	func = test_fisher
	func = test_model
	args = ()
	kwargs = dict(path=path,tol=tol,profile=False)
	profile(func,*args,**kwargs)