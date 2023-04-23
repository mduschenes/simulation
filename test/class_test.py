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


from src.utils import jit,array,rand,arange,zeros,ones,eye,einsum,tensorprod,allclose,is_hermitian,is_unitary,delim,cosh,sinh,cos,sin,bound
from src.utils import norm,conjugate,dagger,dot,cholesky,trotter,expm,fisher,eig,difference,maximum,argmax,abs,sort
from src.utils import pi,delim,arrays,scalars,namespace
from src.iterables import getter,setter
from src.io import load,dump,exists

from src.quantum import Object,Operator,Pauli,State,Gate,Haar,Noise,Label
from src.optimize import Optimizer,Objective,Metric,Callback
from src.system import Dictionary

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
	parameters = rand(shape=parameters.shape,random='normal',bounds=[-1,1],key=1234)
	obj = model(parameters)
	print(parameters.shape,obj.shape,time.time()-t)


	for i in range(10):
		t = time.time()
		parameters = rand(shape=parameters.shape,random='normal',bounds=[-1,1],key=1234)
		obj = model(parameters)
		print(i,parameters.shape,obj.shape,time.time()-t)


	I = model.identity()
	objH = dagger(model(parameters,conj=True))
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
	parameters = (model.coefficients*model.parameters(parameters))[slices].T.ravel()

	tmp = model.identity()
	for i in range(m*d*p):
		f = data[i%(d*p)]
		# print(i,data[i%(d*p)].string)
		tmp = dot(f(parameters[i]),tmp)

	assert allclose(out,tmp), "Incorrect model() from data()"


	tmp = model.identity()
	for i in range(m*d*p):
		f = lambda x: cosh(pi*x)*identity + sinh(pi*x)*data[i%(d*p)].data
		# print(i,data[i%(d*p)].string)
		tmp = dot(f(parameters[i]),tmp)

	assert allclose(out,tmp), "Incorrect model() from func()"

	print('Unitary Conditions Passed')

	return 

def test_parameters(path,tol):

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
	variables = model.parameters(parameters)

	# Get parameters in shape (P*K,M)
	M,N = model.M,model.N
	parameters = parameters.reshape(-1,M)
	variables = variables.reshape(-1,M)

	print(parameters.round(3))
	print(variables.round(6))


	shape = parameters.shape
	parameter = 'xy'	

	print(model.parameters[parameter].category,model.parameters[parameter].method,model.parameters[parameter].constraints(model.parameters()))

	if (model.parameters[parameter].method in [None,'unconstrained']):
		slices = slice(parameters.shape[0])
		vars = parameters
		if not allclose(variables[slices],vars):
			print(vars)
			print(variables[slices])
			raise ValueError("Incorrect parameter initialization %r"%(model.parameters))
	
	elif (model.parameters[parameter].method in ['constrained']):
		G = len(model.parameters[parameter].group)
		wrapper = bound
		funcs = [cos,sin]
		scale = [model.parameters[parameter].parameters,2*pi]
		features = wrapper(parameters)

		for i in range(G):
			func = funcs[i]
			slices = slice(i,features.shape[0],2)
			vars = scale[0]*features[0::2]*func(scale[1]*features[1::2])
			if not allclose(variables[slices],vars):
				print(vars)
				print(variables[slices])
				raise ValueError("Incorrect parameter initialization %d %r"%(i,model.parameters))

	elif (model.parameters[parameter].method in ['bounded']):
		G = len(model.parameters[parameter].group)
		wrapper = bound
		scale = [model.parameters[parameter].parameters,2*pi]		
		features = wrapper(parameters)

		for i in range(G):
			slices = slice(i,features.shape[0],2)
			vars = scale[0]*features[slices]
			if not allclose(variables[slices],vars):
				print(vars)
				print(variables[slices])
				raise ValueError("Incorrect parameter initialization %r"%(model.parameters))





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
	datas = trotter([d.data for d in model.data],P)
	sites = trotter([d.site for d in model.data],P)

	for i,(s,d,D,site) in enumerate(zip(string,data,datas,sites)):
		assert allclose(d,D), "data[%s,%d] incorrect"%(s,i)

	return

def test_label(path,tol):
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

	cls = load(hyperparameters['class']['label'])	

	label = cls(**{**namespace(cls,model),**hyperparameters.get('label',{}),**dict(model=model,system=hyperparameters['system'])})

	print(label())

	new = cls(data=label)

	print(label())

	return

def test_initialization(path,tol):

	default = None
	hyperparameters = load(path,default=default)
	if hyperparameters is None:
		raise "Hyperparameters %s not loaded"%(path)

	cls = {attr: load(hyperparameters['class'][attr]) for attr in hyperparameters['class']}

	kwargs = dict(verbose=True)

	model = cls['model'](**hyperparameters['model'],
		parameters=hyperparameters['parameters'],
		state=hyperparameters['state'],
		noise=hyperparameters['noise'],
		system=hyperparameters['system'],
		**kwargs)

	label = cls['label'](**{**namespace(cls['label'],model),**hyperparameters.get('label',{}),**dict(model=model,system=hyperparameters['system']),**kwargs})
	hyperparams = hyperparameters['optimize']	
	system = hyperparameters['system']
	
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










	copy = {attr: deepcopy(getattr(model,attr)) for attr in ['state','noise','label']}
	attrs = ['noise.string','noise.parameters','state.parameters','exponentiation']
	kwargs = {'initial':dict(),'noisy':dict(noise={'parameters':0.5},state={'parameters':0.5}),'noiseless':dict(noise=False,state=False),'restore':dict(copy)}
	U = {}

	for name in kwargs:

		kwds = {}
		setter(kwds,copy,func=True,copy=True)
		setter(kwds,kwargs[name],func=True,copy=True)
		model.__initialize__(**kwds)
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



def test_compute(path,tol):

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
	variables = model.parameters(parameters)
	coefficients = model.coefficients
	data = array(trotter([data.data for data in model.data],model.P))
	identity = model.identity()

	params = parameters.reshape(-1,model.M)
	vars = variables.reshape(-1,model.M)

	_out = expm(coefficients*variables,data,identity)

	out = model(parameters)

	assert allclose(_out,out), "Incorrect model function"

	return 

def test_fisher(path,tol):
	raise(NotImplementedError)
	return
	
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


def test_machine_precision(path,tol):
	import matplotlib
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	import numpy as np

	from src.utils import rand,dot,dagger,identity,norm,sqrt,abs,abs2,eig,diag,arctan,exp

	norm = lambda A: sqrt(abs2(A).sum())

	def alpha(n,a):
		b = [0]
		for i in range(n):
			c = b[-1]
			c = (1+a)*c + a
			b.append(c)
		return b[1:]

	def beta(n,M,bits):
		random = 'haar'
		dtype = 'complex%d'%(bits)
		A = rand((M,M),random=random,dtype=dtype)
		I = identity(M)
		L,V = eig(A,compute_v=True)
		L = diag(exp(1j*arctan(L.imag,L.real)))
		B = [I]
		b = [0]
		for i in range(1,n+1):
			c = b[-1]
			C = B[-1]
			C = dot(A,C)
			c = norm(C - dot(dot(V,L**i),dagger(V)))
			b.append(c)
			B.append(C)
		return b[1:]		


	M = 2**4
	N = int(1e4)
	K = 11
	A = np.logspace(-20,-20+K-1,K)
	B = [64,128]

	mplstyle = 'config/plot.mplstyle'
	with matplotlib.style.context(mplstyle):

		fig,ax = plt.subplots()
		plots = []
		for a in A:
			b = alpha(N,a)
			n = list(range(1,N+1))
			index = (np.log(a)-np.log(A.min()))/(np.log(A.max())-np.log(A.min()))
			slope = (np.log(b[-1])-np.log(b[0]))/(np.log(n[-1])-np.log(n[0]))
			intercept = np.log(b[0])/np.log(a)
			plot = ax.plot(n,b,linewidth=4,label='$10^{-%d}$'%(int(-np.log10(a))),color=plt.cm.viridis(index))
			plots.append(plot)

		for bits in B:
			b = [beta(N,M,bits) for a in range(100)]
			n = list(range(1,N+1))
			b,berr = np.mean(b,axis=0),np.std(b,axis=0)/sqrt(len(b)-1)
			plot = ax.errorbar(n,b,yerr=berr,alpha=(B.index(bits)+1)/len(B),linewidth=4,linestyle='--',color='gray',label=r'$%d~\textrm{bit}$'%(bits//2))
				
		# ax.axhline(1e-8,color='k',linestyle='--');
		# ax.axhline(1e-16,color='k',linestyle='--');
		fig.set_size_inches(10,10)
		ax.set_xlabel(r'$\textrm{Matmul Count}~k$')
		ax.set_ylabel(r'$\textrm{Matmul Error}~ \epsilon_{k}$')
		ax.set_yscale('log');
		ax.set_xscale('log');
		# ax.set_xlim(5e-1,5e6);
		# ax.set_ylim(1e-21,1e-3);
		# ax.set_xticks([1e0,1e2,1e4,1e6]);
		# ax.set_yticks([1e-20,1e-16,1e-12,1e-8,1e-4]);
		ax.minorticks_off();
		ax.grid(True,alpha=0.3);
		ax.set_title(r'$\norm{\widetilde{A^k} - A^k} = \epsilon_{k} = n^k\sum_{l>0}^{k} \binom{k}{l} \epsilon^{l} ~\sim~ k\epsilon$',pad=30)
		legend = ax.legend(
			title=r'$A \in \textrm{U}(n)$' + '\n' + r'$\textrm{Machine Precision} ~ \epsilon \sim \varepsilon ~ O(n)$',
			loc=[1.1,-0.05],ncol=1);
		legend.get_title().set_ha('center')
		fig.savefig('matmul_error.pdf',bbox_inches="tight",pad_inches=0.2)

	return


if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 

	func = [
		test_object,
		test_model,
		test_parameters,
		test_logger,
		test_data,
		test_initialization,
		test_compute,
		test_fisher,
		test_machine_precision,
		]
	# func = test_model
	func = test_parameters
	func = test_initialization
	# func = test_label
	func = test_machine_precision
	func = test_model
	func = test_initialization
	args = ()
	kwargs = dict(path=path,tol=tol,profile=False)
	profile(func,*args,**kwargs)