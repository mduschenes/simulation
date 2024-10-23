#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ["",".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,array,zeros,ones,empty,rand,haar,allclose,product,representation
from src.utils import einsum,conjugate,dagger,dot,tensorprod,trace,real,imag,sqrtm,sqrt,cos,sin,abs2
from src.utils import swap,shuffle,seeder,rng
from src.utils import arrays,iterables,scalars,integers,floats,pi,delim
from src.iterables import permutations
from src.io import load,dump,glob
from src.call import rm,echo
from src.system import Dict,Dictionary
from src.iterables import namespace,permuter,setter,getter
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
# logger = Logger()


def equalizer(a,b):
	if isinstance(a,arrays) and isinstance(b,arrays):
		return all(allclose(i,j) for i,j in zip(a.ravel(),b.ravel()))
	elif isinstance(a,dict) and isinstance(b,dict):
		return all(allclose(a[i],b[j]) for i,j in zip(a,b))
	elif isinstance(a,iterables) and isinstance(b,iterables):
		return all(allclose(i,j) for i,j in zip(a,b))
	else:
		return a==b

def test_basis(*args,**kwargs):

	from src.quantum import Basis as basis
	
	D = 2
	N = 1
	L = 1
	K = 2
	ndim = 2
	shape = [D**L]*(K if ndim is None else ndim)
	data="zero.depolarize.X"
	key = 123
	delim = "."
	dtype = "complex"

	options = Dict(D=D,N=N,ndim=ndim,shape=shape,data=data,key=key,dtype=dtype)

	operators = {
		"rand":Dict(locality=L,shapes={i:[options.D]*L for i in range(K if ndim is None else ndim)},dimension=2),
		"X":Dict(locality=1,shapes={i:[options.D]*options.N for i in range(K)},dimension=2),
		"depolarize":Dict(locality=1,shapes={**{i:[options.D**2]*options.N for i in range(1)},**{i:[options.D]*options.N for i in range(1,K+1)}},dimension=3),
		"string":Dict(locality=len(data.split(delim)),shapes={0:[1,options.D**2,1],1:[2,options.D,options.D],2:[options.D]*len(data.split(delim))},dimension=3),
		"pauli":Dict(locality=1,shapes={**{i:[options.D**2]*options.N for i in range(1)},**{i:[options.D]*options.N for i in range(1,K+1)}},dimension=3),		
		}

	for operator in operators:
		data = basis.get(operator)(**options)

		print(operator)
		print(data)
		for attr in operators[operator]:
			print(attr,getattr(basis,attr)(operator,**options),operators[operator][attr])
			assert operators[operator][attr] == getattr(basis,attr)(operator,**options)
		print()


	print("Passed")

	return


def test_operator(*args,**kwargs):

	data = {}

	kwargs = {
		"operator.N":[4],"state.N":[4],
		"operator.D":[2],"state.D":[2],
		"operator.ndim":[None],"state.ndim":[2,1],
		"operator.local":[False,True],
		"operator.data":[None,None,None,None,None,None,None],
		"operator.operator":["CNOT.H",["X","Z"],"haar","haar",["U","U"],["u"],["depolarize","amplitude","dephase"]],
		"operator.site":[[3,0,1],[0,2],None,[1,2,0],[3,1],[0],[0,2,1]],
		"operator.string":["test","xz","Haar","haar","U","u","noise"],
		"operator.parameters":[None,0.5,None,None,None,None,1e-6],
		"operator.variable":[False,False,False,False,False,False,False,False],
		"operator.constant":[True,False,False,False,False,False,True,True],
		"state.local":[False],
		"state.data":[None],
		"state.operator":["zero"],
		"state.site":[None],
		"state.string":["zero"],
		"state.parameters":[None],
		"state.variable":[False],
		"state.constant":[True],

		}
	groups = [[
		"operator.data","operator.operator","operator.site",
		"operator.string","operator.parameters","operator.variable","operator.constant"
		],[
		"state.data","state.operator","state.site",
		"state.string","state.parameters","state.variable","state.constant"
		],
		]
	filters = None
	func = None

	options = dict(dtype="complex")

	for index,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters,func=func)):
	
		settings = Dict({
			"cls":{
				"operator":"src.quantum.Operator",
				"state":"src.quantum.State",
				"basis":"src.quantum.Basis"
			},
			"operator":{
				"data":None,"operator":None,"site":None,"string":None,
				"N":2,"D":2,"ndim":2,"local":True,"variable":True,"constant":False,
				"system":{"seed":12345,"dtype":"complex","architecture":None}				
			},	
			"state": {
				"data":None	,
				"operator":"product",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":None}
				},
		})

		setter(settings,kwargs,delimiter=delim,default=True)

		verbose = True


		# Data
		operator = load(settings.cls.operator)

		operator = operator(**settings.operator)

		if operator.string in ["test"]:
			_data = [
				array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],**options),
				array([[1,1],[1,-1]],**options),
				]
			axes = [3,0,1,2]
			shape = {0:[2,2,2,2],1:[2,2,2,2]}
		elif operator.string in ["xz"]:
			_data = [
				array([[0,1],[1,0]],**options),
				array([[1,0],[0,-1]],**options),
				]
			axes = [0,2,1,3]
			shape = {0:[2,2,2,2],1:[2,2,2,2]}
		elif operator.string in ["Haar"]:
			_data = [
				haar(shape=(2**4,)*2,seed=seeder(operator.seed),**options),
				]
			axes = [0,1,2,3]
			shape = {0:[2,2,2,2],1:[2,2,2,2]}
		elif operator.string in ["haar"]:
			_data = [
				haar(shape=(2**3,)*2,seed=seeder(operator.seed),**options),
				]
			axes = [1,2,0,3]
			shape = {0:[2,2,2,2],1:[2,2,2,2]}
		elif operator.string in ["U"]:
			_data = [
				haar(shape=(2**1,)*2,seed=seeder(operator.seed),**options),
				haar(shape=(2**1,)*2,seed=seeder(operator.seed),**options),
				]
			axes = [3,1,0,2]
			shape = {0:[2,2,2,2],1:[2,2,2,2]}
		elif operator.string in ["u"]:
			_data = [
				haar(shape=(2,)*2,seed=seeder(operator.seed),**options),
				]
			axes = [0,3,1,2]
			shape = {0:[2,2,2,2],1:[2,2,2,2]}
		elif operator.string in ["noise"]:
			_data = [
				array([
				sqrt(1-(2**2-1)*operator.parameters()/(2**2))*array([[1,0],[0,1]],**options),
				sqrt(operator.parameters()/(2**2))*array([[0,1],[1,0]],**options),
				sqrt(operator.parameters()/(2**2))*array([[0,-1j],[1j,0]],**options),
				sqrt(operator.parameters()/(2**2))*array([[1,0],[0,-1]],**options),
				],**options),
				array([
					array([[1,0],[0,0]],**options) + 
						sqrt(1-operator.parameters())*array([[0,0],[0,1]],**options),
					sqrt(operator.parameters())*array([[0,1],[0,0]],**options)
					],**options),
				array([
					sqrt(1-operator.parameters())*array([[1,0],[0,1]],**options),
					sqrt(operator.parameters())*array([[1,0],[0,-1]],**options)
					],**options),
				]
			axes = [0,2,1,3]
			shape = {0:[4,2,2,1],1:[2,2,2,2],2:[2,2,2,2]}

			
		# if operator.local:
		# 	_tmp = tensorprod(_data)
		# else:
		_tmp = [*_data,*[array([[1,0],[0,1]],**options)]*(operator.N-operator.locality)]
		_tmp = shuffle(tensorprod(_tmp),axes=axes,shape=shape)

		if not operator.constant:
			_tmp = (cos(operator.parameters(operator.parameters()))*tensorprod([array([[1,0],[0,1]],**options)]*(operator.N)) + 
			        -1j*sin(operator.parameters(operator.parameters()))*_tmp)

		operator.info(verbose=verbose)

		parameters = operator.parameters()
		state = tensorprod((operator.basis.get(operator.default)(D=operator.D,dtype=operator.dtype),)*operator.N)
		kwargs = dict()

		tmp = operator(parameters=parameters,state=state,**kwargs)

		if verbose:
			print("----- data ------")
			print(tmp)
			print("-----------------")
			print()
			print("----- _data ------")
			print(_tmp)
			print("-----------------")
			print()

		assert (not operator.unitary) or allclose(tmp,_tmp), "Incorrect operator %r"%(operator)



		# State

		state = load(settings.cls.state)
		state = state(**settings.state)

		operator.init(state=state)

		parameters = operator.parameters()
		state = operator.state()
		kwargs = dict()

		tmp = operator(parameters=parameters,state=state,**kwargs)

		_tmp = [*_data,*[array([[1,0],[0,1]],**options)]*(operator.N-operator.locality)]
		_tmp = shuffle(tensorprod(_tmp),axes=axes,shape=shape)

		if not operator.constant:
			_tmp = (cos(operator.parameters(operator.parameters()))*tensorprod([array([[1,0],[0,1]],**options)]*(operator.N)) + 
				        -1j*sin(operator.parameters(operator.parameters()))*_tmp)

		if operator.ndim == 3:
			if state is None:
				_tmp = _tmp
			elif state.ndim == 2:
				_tmp = einsum("uij,jl,ukl->ik",_tmp,state,conjugate(_tmp))
			elif state.ndim == 1:
				continue
			else:
				raise NotImplementedError("Incompatible dimensions data and state %r"%(operator))

		elif operator.ndim == 2:
			if state is None:
				_tmp = _tmp
			elif state.ndim == 2:
				_tmp = einsum("ij,jl,kl->ik",_tmp,state,conjugate(_tmp))
			elif state.ndim == 1:
				_tmp = einsum("ij,j->i",_tmp,state)				
			else:
				raise NotImplementedError("Incompatible dimensions data and state %r"%(operator))
		
		else:
			
			raise NotImplementedError("Incompatible dimensions data and state %r"%(operator))



		if verbose:
			print("----- state ------")
			print(tmp)
			print("-----------------")
			print()
			print("----- _state ------")
			print(_tmp)
			print("-----------------")
			print()

		assert allclose(tmp,_tmp), "Incorrect operator(parameters,state) %r"%(operator)

	print("Passed")

	return	


def test_data(path,tol):

	from src.quantum import trotter

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

	label.init(state=state)	
	model.init(state=state)


	basis = {
		"I":array([[1,0],[0,1]],dtype=model.dtype),
		"X":array([[0,1],[1,0]],dtype=model.dtype),
		"Y":array([[0,-1j],[1j,0]],dtype=model.dtype),
		"Z":array([[1,0],[0,-1]],dtype=model.dtype),
		}
	default = "I"

	N = model.N
	P = model.P
	locality = model.locality
	local = model.local

	if local:
		string = [
			*[[O for k in range(N) if k in [i]]
				for O in ["X","Y","Z"]
				for i in range(N)
				],
			*[[O for k in range(N) if k in [i,j] ]
				for O in ["Z"]
				for i in range(N)
				for j in range(N)
				if i<j
				],
			]
	else:
		string = [
			*[[O if k in [i] else default for k in range(N)]
				for O in ["X","Y","Z"]
				for i in range(N)
				],
			*[[O if k in [i,j] else default for k in range(N)]
				for O in ["Z"]
				for i in range(N)
				for j in range(N)
				if i<j
				],
			]

	if local:
		data = [tensorprod(array([basis[i] for i in s])) for s in string]
		identity = tensorprod(array([basis[default]]*N))
	else:
		data = [tensorprod(array([basis[i] for i in s])) for s in string]
		identity = tensorprod(array([basis[default]]*N))		

	assert allclose(model.identity,identity), "Incorrect model identity"

	data = trotter(data,P)
	string = trotter(string,P)
	datas = trotter([model.data[i].data for i in model.data if model.data[i].unitary],P)
	sites = trotter([model.data[i].site for i in model.data if model.data[i].unitary],P)

	for i,(s,d,D,site) in enumerate(zip(string,data,datas,sites)):
		assert allclose(d,D), "data[%s,%d] incorrect"%(s,i)
	
	print("Passed")
	
	return

def test_initialization(path,tol):

	default = None
	settings = load(path,default=default)
	if settings is None:
		raise Exception("settings %s not loaded"%(path))

	settings = Dict(settings)

	verbose = True

	model = load(settings.cls.model)
	state = load(settings.cls.state)
	label = load(settings.cls.label)

	hyperparameters = settings.optimize
	system = settings.system
	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})

	label.init(state=state)	
	model.init(state=state)

	parameters = model.parameters()
	kwargs = dict(verbose=verbose)

	metric = Metric(state=state,label=label,hyperparameters=hyperparameters,system=system,**kwargs)

	def copier(model,metric,state,label):

		copy = Dictionary(
			model=Dictionary(
				func=model.__call__,
				data=model(parameters=model.parameters(),state=model.state()),
				state=state,
				noise=[model.data[i] for i in model.data 
					if (model.data[i] is not None) and (not model.data[i].unitary)],
				info=model.info,hermitian=model.hermitian,unitary=model.unitary),
			metric=Dictionary(
				func=metric.__call__,
				data=metric(model(model.parameters())),
				state=metric.state,
				noise=[model.data[i] for i in model.data 
					if (model.data[i] is not None) and (not model.data[i].unitary)],
				info=metric.info,hermitian=label.hermitian,unitary=label.unitary),
			label=Dictionary(
				func=label.__call__,
				data=label(state=state()),
				state=state,
				info=label.info,
				hermitian=label.hermitian,
				unitary=label.unitary),
			state=Dictionary(
				func=state.__call__,
				data=state(),
				state=state,
				info=state.info,hermitian=state.hermitian,unitary=state.unitary),
			)

		return copy

	defaults = Dictionary(state=state,data={i: model.data[i] for i in model.data},label=metric.label)

	old = copier(model,metric,state,label)


	# Initial

	updates = Dictionary(state=True,data={i:True for i in model.data if model.data[i] is not None and not model.data[i].unitary},label=False)

	label.init(state=updates.state)	

	model.init(state=updates.state,data=updates.data)

	print("First")
	print(updates)	
	print(model.state,{attr:getattr(model,attr) for attr in ["N","D","ndim","locality","local"]})
	print(label.state,{attr:getattr(label,attr) for attr in ["N","D","ndim","locality","local"]})
	print(state.state,{attr:getattr(state,attr) for attr in ["N","D","ndim","locality","local"]})
	print({i: model.data[i] for i in model.data})
	print(model.state())
	print()

	init = Dictionary(state=state,data={i: model.data[i] for i in model.data},label=metric.label)
	

	# Update

	updates = Dictionary(state=False,data={i:False for i in model.data if model.data[i] is not None and not model.data[i].unitary},label=False)

	label.init(state=updates.state)	

	model.init(state=updates.state,data=updates.data)

	print("Second")
	print(updates)	
	print(model.state,{attr:getattr(model,attr) for attr in ["N","D","ndim","locality","local"]})
	print(label.state,{attr:getattr(label,attr) for attr in ["N","D","ndim","locality","local"]})
	print(state.state,{attr:getattr(state,attr) for attr in ["N","D","ndim","locality","local"]})
	print({i: model.data[i] for i in model.data})
	print(model.state())	
	print()

	metric.init(model=model,label=label)

	tmp = copier(model,metric,state,label)



	# Restore

	updates = Dictionary(state=defaults.state,data=defaults.data,label=False)

	label.init(state=updates.state)

	model.init(state=updates.state,data=updates.data)

	print("Third")
	print(updates)
	print(model.state,{attr:getattr(model,attr) for attr in ["N","D","ndim","locality","local"]})
	print(label.state,{attr:getattr(label,attr) for attr in ["N","D","ndim","locality","local"]})
	print(state.state,{attr:getattr(state,attr) for attr in ["N","D","ndim","locality","local"]})
	print({i: model.data[i] for i in model.data})
	print(model.state())
	print()

	metric.init(model=model,label=label)

	new = copier(model,metric,state,label)

	# print("--- COPY INFO ---")
	# old.model.info(verbose=verbose)
	# print()

	# print("--- TMP INFO ---")
	# tmp.model.info(verbose=verbose)
	# print()

	# print("--- NEW INFO ---")
	# new.model.info(verbose=verbose)
	# print()


	# print("State model (hermitian: %s, unitary: %s)"%(old.model.hermitian,old.model.unitary))
	# print(old.model.data)

	# print("State label (hermitian: %s, unitary: %s)"%(old.label.hermitian,old.label.unitary))
	# print(old.label.data)

	# print("State state (hermitian: %s, unitary: %s)"%(old.label.state.hermitian,old.label.state.unitary))
	# print(old.label.state())

	# print("Unitary model (hermitian: %s, unitary: %s)"%(tmp.model.hermitian,tmp.model.unitary))
	# print(tmp.model.data)

	# print("Unitary label (hermitian: %s, unitary: %s)"%(tmp.label.hermitian,tmp.label.unitary))
	# print(tmp.label.data)

	# print("State model (hermitian: %s, unitary: %s)"%(new.model.hermitian,new.model.unitary))
	# print(new.model.data)

	# print("State label (hermitian: %s, unitary: %s)"%(new.label.hermitian,new.label.unitary))
	# print(new.label.data)

	# print("State state (hermitian: %s, unitary: %s)"%(new.label.state.hermitian,new.label.state.unitary))
	# print(new.label.state())


	UpsiU = old.model.data
	U = tmp.model.data

	psi = old.state.data
	K = old.model.noise[-1].data if old.model.noise and old.model.noise[-1].parameters() else None
	VpsiV = old.label.data
	V = tmp.label.data

	if K is None:
		if psi is None:
			return
		elif psi.ndim == 1:
			UpsiUtmp = einsum("ij,j->i",U,psi,conjugate(U))
			VpsiVtmp = einsum("ij,j->i",V,psi,conjugate(V))
		elif psi.ndim == 2:
			UpsiUtmp = einsum("ij,jk,lk->il",U,psi,conjugate(U))
			VpsiVtmp = einsum("ij,jk,lk->il",V,psi,conjugate(V))		
	elif K is not None:
		#TODO: Implement test for multiple layers of noise 
		if psi is None:
			return
		elif psi.ndim == 1:
			return
		elif psi.ndim == 2 and model.M == 1:
			UpsiUtmp = einsum("uij,jk,kl,ml,unm->in",K,U,psi,conjugate(U),conjugate(K))
			VpsiVtmp = einsum("ij,jk,lk->il",V,psi,conjugate(V))		
		else:
			return


	print(UpsiUtmp)
	print(UpsiU)

	assert allclose(UpsiUtmp,UpsiU), "Incorrect model() re-initialization"
	assert allclose(VpsiVtmp,VpsiV), "Incorrect label() re-initialization"
	assert allclose(new.metric.data,old.metric.data), "Incorrect metric() re-initialization"
	
	print("Passed")
	
	return


def test_tensorproduct(*args,**kwargs):

	settings = Dict({
		"cls":{
			"model":"src.quantum.Operators",
			"state":"src.quantum.State",
			"operator":"src.quantum.Operator",
			"basis":"src.quantum.Basis"
		},
		"model":{
			"data":{
				# "cnot-hadamard":{
				# 	"operator":["S","CNOT","H"],"site":[3,2,0,5],"string":"cnot-hadamard",
				# 	"parameters":None,
				# 	"variable":False
				# },
				# "data":{
				# 	"operator":["CNOT"],"site":"<ij>","string":"cnot",
				# 	"parameters":None,
				# 	"variable":False
				# },
				# "xx":{
				# 	"operator":["X","X"],"site":"<ij>","string":"xx",
				# 	"parameters":0.5,
				# 	"variable":False
				# },
				"cnot":{
					"operator":["CNOT"],"site":"<ij>","string":"cnot",
					"parameters":None,
					"variable":False
				},								
				"hadamard":{
					"operator":["H"],"site":"i","string":"hadamard",
					"parameters":None,
					"variable":False
				}					
			},
			"N":4,"D":2,"ndim":2,"local":True,
			"system":{
				"seed":12345,"dtype":"complex",
				"architecture":None,"configuration":{"key":["site"]},
				}
		},	
		"state": {
			"data":None	,
			"operator":["plus","minus"],
			"site":None,
			"string":"psi",
			"parameters":None,
			"D":2,"ndim":2,"local":True,"variable":False,
			"system":{"seed":12345,"dtype":"complex","architecture":None}
			},
		"operator": {
			"operator":["CNOT"],"site":None,"string":"cnot",
			"parameters":None,
			"D":2,"local":True,"variable":False,"dtype":"complex"
		}
	})

	verbose = True

	model = load(settings.cls.model)
	state = load(settings.cls.state)
	basis = load(settings.cls.basis)

	model = model(**settings.model)
	state = state(**settings.state)
	# model.init(state=state)

	model.info(verbose=verbose)

	state.info(verbose=verbose)

	try:
		new = model @ model
		new.info(verbose=verbose)
		
		test = new(parameters=new.parameters(),state=new.state())

		_test = tensorprod([basis.get(j)(**settings.model) 
			for i in settings.model.data 
			for j in (
				settings.model.data[i].operator 
				if isinstance(settings.model.data[i].operator,iterables) else 
				[settings.model.data[i].operator])]*2)

		assert allclose(test,_test), "Incorrect model @ model"

	except NotImplementedError:
		pass


	for i in model.data:
		for j in model.data:
			
			try:
				new = model.data[i] @ model.data[j]
				# new.info(verbose=verbose)
				print(dict((map(tuple,i) for i in zip([model.data[i].site,model.data[j].site,new.site],[model.data[i].operator,model.data[j].operator,new.operator]))))

				# test = new(parameters=new.parameters(),state=new.state())

				# _test = tensorprod([basis.get(j)(**settings.model) 
				# 	for i in settings.model.data 
				# 	for j in (
				# 		settings.model.data[i].operator 
				# 		if isinstance(settings.model.data[i].operator,iterables) else 
				# 		[settings.model.data[i].operator])]*2)

				# assert allclose(test,_test), "Incorrect model @ model"
			except NotImplementedError:
				pass


	new = state @ state
	new.info(verbose=verbose)

	test = new(parameters=new.parameters(),state=new.state())

	_test = tensorprod([basis.get(i)(**settings.state)
				for i in (
					settings.state.operator
					if isinstance(settings.state.operator,iterables) else
					[settings.state.operator]
					)]*2)

	assert allclose(test,_test), "Incorrect state @ state"


	operators = [["plus","minus"]]*3

	options = Dict(**{
		"data":None	,
		"operator":None,
		"site":None,
		"string":"psi",
		"parameters":None,
		"D":2,"ndim":2,"local":True,"variable":False,
		"system":{"seed":12345,"dtype":"complex","architecture":None}
		})

	cls = load(settings.cls.state)


	states = [cls(**{**options,**dict(operator=operator)}) for operator in operators]
	site = [2*i for i in range(sum(len(operator) if isinstance(operator,iterables) else 1 for operator in operators))]

	new = states[0] 
	for state in states[1:]:
		new @= state

	new.init(site=site)

	new.info(verbose=verbose)

	test = new(parameters=new.parameters(),state=new.state())

	_test = tensorprod([basis.get(i)(**options) for operator in operators for i in (operator if isinstance(operator,iterables) else [operator])])

	_test_ = tensorprod([
			state(parameters=state.parameters(),state=state.state())
			for state in states])

	assert allclose(test,_test) and allclose(test,_test_) and allclose(_test,_test_), "Incorrect state @ state"




	operator = load(settings.cls.operator)
	operator = operator(**settings.operator)
	
	operator.info(verbose=verbose)


	new = operator @ 3
	new.info(verbose=verbose)
	
	# test = new(parameters=new.parameters(),state=new.state())

	# _test = tensorprod([basis.get(j)(**settings.operator) 
	# 	for i in (
	# 		settings.operator.operator 
	# 		if isinstance(settings.operator.operator,iterables) else 
	# 		[settings.operator.operator])]*2)

	# assert allclose(test,_test), "Incorrect operator @ operator"

	print("Passed")

	return


def test_random(*args,**kwargs):

	settings = Dict({
		"cls":{
			"state":"src.quantum.State",
			"operator":"src.quantum.Operator",
			"basis":"src.quantum.Basis"
		},
		"state": {
			"data":None	,
			"operator":"psi",
			"site":None,
			"string":"psi",
			"parameters":None,
			"N":3,"D":2,"ndim":2,
			"local":True,"variable":False,
			"seed":123,"dtype":"complex"
			},
		"operator": {
			"data":None,
			"operator":"U",
			"site":[0,2],
			"string":"U",
			"parameters":None,
			"N":3,"D":2,"ndim":2,
			"local":True,"variable":False,
			"seed":123,"dtype":"complex"
		}
	})

	options = Dictionary(verbose=True,precision=8)

	operator = load(settings.cls.operator)
	operator = operator(**settings.operator)
	operator.info(**options)

	parameters = operator.parameters()
	state = tensorprod([operator.basis.get(operator.default)(**operator)]*operator.N)
	kwargs = Dictionary(seed=seeder(123456789))

	obj = Dictionary()

	obj.data = operator.data
	obj.func = operator(parameters=parameters,state=state,**kwargs)

	key,kwargs.seed = rng.split(kwargs.seed)

	obj.tmp = operator(parameters=parameters,state=state,**kwargs)

	obj.norm = trace(dot(dagger(obj.data),obj.data))/operator.D**operator.locality

	for attr in obj:
		print(attr)
		print(getattr(obj,attr).round(options.precision))
		print()

	print()
	print("-------------------")
	print()

	state = load(settings.cls.state)
	state = state(**settings.state)
	state.info(**options)

	parameters = state.parameters()
	kwargs = Dictionary(seed=seeder(123))

	obj = Dictionary()

	obj.data = state.data
	obj.func = state(parameters=parameters,**kwargs)

	key,kwargs.seed = rng.split(kwargs.seed)

	obj.tmp = state(parameters=parameters,**kwargs)

	obj.norm = trace(obj.data)

	for attr in obj:
		print(attr)
		print(getattr(obj,attr).round(options.precision))
		print()

	return



def test_layout(*args,**kwargs):

	settings = Dict({
		"cls":{
			"model":"src.quantum.Operators",
			"state":"src.quantum.State"
		},
		"model":{
			"data":{
				"xx":{
					"operator":["X","X"],"site":"|ij|","string":"xx",
					"parameters":0.5,
					"variable":False
				},
				"noise":{
					"operator":["depolarize","depolarize"],"site":"|ij|","string":"noise",
					"parameters":{"data":[0,1,2,3,4,5,6]},
					"variable":False
				},	
				# "noise":{
				# 	"operator":["depolarize"],"site":"|i.j|","string":"noise",
				# 	"parameters":None,
				# 	"variable":False
				# }				
			},
			"N":6,"D":2,"ndim":2,"local":True,
			"system":{
				"seed":12345,"dtype":"complex",
				"architecture":None,"configuration":{
					"key":[lambda value,iterable: (
						# [tuple(j) for i in [*range(0,value.N,2),*range(1,value.N,2)] for j in [[i,i+1],[i],[i+1]]].index(tuple(value.site))
						value.site[0]%2,value.site[0],-value.locality,[id(iterable[i]) for i in iterable].index(id(value)),
						)],
					"sort":None,
					"reverse":False
					},
				}
		},	
		"state": {
			"data":None	,
			"operator":"zero",
			"site":None,
			"string":"psi",
			"parameters":None,
			"D":6,"ndim":2,"local":True,"variable":False,
			"system":{"seed":12345,"dtype":"complex","architecture":None}
			}
	})

	options = Dictionary(verbose=False,precision=4)

	model = load(settings.cls.model)
	model = model(**settings.model)
	model.info(**options)

	# state = load(settings.cls.state)
	# state = state(**settings.state)
	# state.info(**options)

	# model.init(state=state)

	attrs = ["string","parameters","N","locality","operator","site",]

	for i in model.data:
		print(
			[id(model.data[i]) for i in model.data].index(id(model.data[i])),
			{attr: getattr(model.data[i],attr) 
			if not callable(getattr(model.data[i],attr)) else 
			str(getattr(model.data[i],attr)().round(options.precision))
			for attr in attrs}
			)

	return


def test_measure(*args,**kwargs):

	kwargs = {
		"measure.operator":["pauli","tetrad"],
		"state.D":[4,4],
		}
	groups = [["measure.operator","state.D",]]

	for index,kwargs in enumerate(permuter(kwargs,groups=groups)):

		settings = Dict({
			"cls":{
				"measure":"src.quantum.Measure",
				"state":"src.quantum.State",
				"basis":"src.quantum.Basis",
			},
			"measure":{
				"data":None,
				"operator":"pauli",
				"string":"povm",
				"D":2,
				"dtype":"complex"
			},
			"state": {
				"data":"random"	,
				"operator":"state",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":4,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":"array"}
				}
			})

		setter(settings,kwargs,delimiter=delim,default=True)

		measure = load(settings.cls.measure)
		state = load(settings.cls.state)
		basis = load(settings.cls.basis)

		measure = measure(**settings.measure)
		state = state(**settings.state)

		measure.init(state=state)

		print(settings["measure"]["operator"])
		print(measure,len(measure),measure.D,state.D)
		print(measure.identity)
		print(measure.data)
		print(measure.inverse)
		print()


		assert allclose(sum(i for i in measure.basis),basis.I(D=measure.D,dtype=measure.dtype)), "Incorrect %r basis"%(measure)
		assert allclose(einsum("uw,vw->uv",measure.data,measure.inverse),basis.identity(D=len(measure),dtype=measure.dtype)), "Incorrect %r data"%(measure)

	print("Passed")

	return


def test_metric(path=None,tol=None,**kwargs):

	from src.utils import gradient
	from src.utils import allclose,trace,dot

	from src.iterables import getter,setter,permuter,equalizer,namespace

	from src.io import load,dump

	from src.optimize import Optimizer,Objective,Metric,Callback

	from src.system import Dict

	from src.parameters import Parameters

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

	func = None
	arguments = ()
	keywords = {}

	model = model(**{**settings.model,**dict(system=system)})

	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})

	label.init(state=state)
	model.init(state=state)

	metric = Metric(state=state,label=label,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system)


	print(state())
	print(label.parameters())
	print(label.data)

	out = metric(label())

	assert allclose(0,out), "Incorrect metric %0.5e"%(out)

	print("Passed")
	
	return

def test_namespace(*args,**kwargs):

	data = {}

	kwargs = {
		**{attr:[5] for attr in ["model.N"]},
		**{attr:[2] for attr in ["model.D"]},
		**{attr:[4] for attr in ["model.M"]},
		**{attr:[True] for attr in ["model.data.z.local","model.data.xx.local","model.data.noise.local"]},
		}
	groups = [["model.N"],["model.data.z.local","model.data.xx.local","model.data.noise.local"]]
	filters = None
	func = None

	for index,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters,func=func)):
		
		settings = Dict({
			"cls":{
				"model":"src.quantum.Operators",
				"state":"src.quantum.State",
				"label":"src.quantum.Label",
				"measure":"src.quantum.Measure"
			},
			"model":{
				"data":{
					"z":{
						"operator":["Z"],"site":"i","string":"Z",
						"parameters":0.5,"variable":True
					},
					"xx":{
						"operator":["X","X"],"site":"<ij>","string":"XX",
						"parameters":0.5,"variable":False
					},
					"noise":{
						"operator":["dephase","dephase"],"site":None,"string":"dephase",
						"parameters":1e-3,"variable":False
					}		
				},
				"space":"spin",
				"time":"linear",
				"lattice":"square",
				"architecture":"array"
				},
			"state": {
				"operator":"zero",
				"site":None,
				"string":"psi",
				"parameters":True,
				"ndim":2,
				"seed":123,
				"architecture":"array"
				},
			"label": {
				"operator":"X.X.X.X",
				"site":None,
				"string":"U",
				"parameters":0.5,
				"ndim":2,
				"seed":123
				},
			"measure":{
				"operator":"pauli",
				"architecture":"tensor"				
			},
			"system":{
				"dtype":"complex",
				"format":"array",
				"device":"cpu",
				"backend":None,
				"architecture":None,
				"base":None,
				"seed":123,
				"key":None,
				"instance":None,
				"cwd":"data",
				"path":None,
				"path":"data.hdf5",
				"conf":"logging.conf",
				"logger":None,
				"cleanup":False,
				"verbose":False
				}
		})

		setter(settings,kwargs,delimiter=delim,default=True)

		verbose = True
		ignore = "parameters"

		model = load(settings.cls.model)
		state = load(settings.cls.state)
		label = load(settings.cls.label)
		measure = load(settings.cls.measure)
		system = settings.system


		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(system=system)})
		label = label(**{**namespace(label,model),**settings.label,**dict(system=system)})
		measure = measure(**{**namespace(measure,model),**settings.measure,**dict(system=system)})

		print("Attributes",{attr: getattr(settings.model,attr,None) for attr in ["N","data"]})

		model.info(verbose=verbose)
		state.info(verbose=verbose)

		attributes = {"model":model,"state":state,"label":label}
		for attribute in attributes:
			print(attribute,{attr: getattr(attributes[attribute],attr) for attr in ["N","locality","site","hermitian","unitary","architecture"]},">>>>",namespace(attributes[attribute].__class__,model))
		print()

		model.init(state=state)
		label.init(state=state)

		attributes = {"model":model,"label":label}
		for attribute in attributes:
			print(attribute,{attr: getattr(attributes[attribute],attr) for attr in ["N","locality","site","hermitian","unitary","architecture"]},">>>>",namespace(attributes[attribute].__class__,model))
		print()

		print("Call")
		print(state())
		print(model(state=state()))
		print(label(state=state()))
		print()


		print("Measure")
		attributes = {"measure":measure}
		for attribute in attributes:
			print(attribute,{attr: getattr(attributes[attribute],attr) for attr in ["D","operator","ind","inds","tags","data","inverse","basis","architecture"]},">>>>",attributes[attribute].__class__,namespace(attributes[attribute].__class__,model))
		print()

		print()
		print()
		print()

		parameters = model.parameters()
		state = model.state()
		value = model(parameters,state)

		data[index] = value


	assert all(allclose(data[i],data[j]) for i in data for j in data if i != j), "Error - Inconsistent models"

	print("Passed")

	return


def test_objective(path=None,tol=None,**kwargs):

	from src.utils import gradient
	from src.utils import allclose,trace,dot

	from src.iterables import getter,setter,permuter,equalizer,namespace

	from src.io import load,dump

	from src.optimize import Optimizer,Objective,Metric,Callback

	from src.system import Dict

	from src.parameters import Parameters

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

	func = None
	arguments = ()
	keywords = {}

	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})
	callback = callback(**{**namespace(callback,model),**settings.callback,**dict(model=model,system=system)})

	label.init(state=state)
	model.init(state=state)

	parameters = model.parameters()
	state = model.state()
	label = model(parameters,state=state)

	print(state)
	print(label)

	metric = Metric(state=state,label=label,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system)
	func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system)

	out = func(parameters,state=state)

	assert allclose(0,out), "Incorrect objective %0.5e"%(out)

	print("Passed")

	return

def test_grad(path=None,tol=None,**kwargs):

	from src.utils import gradient
	from src.utils import allclose,trace,dot

	from src.iterables import getter,setter,permuter,equalizer,namespace

	from src.io import load,dump

	from src.optimize import Optimizer,Objective,Metric,Callback

	from src.system import Dict

	from src.parameters import Parameters

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

	func = None
	arguments = ()
	keywords = {}

	model = model(**{**settings.model,**dict(system=system)})
	state = state(**{**namespace(state,model),**settings.state,**dict(model=model,system=system)})
	label = label(**{**namespace(label,model),**settings.label,**dict(model=model,system=system)})

	label.init(state=state)
	model.init(state=state)

	parameters = model.parameters()
	state = model.state()

	# grad of unitary
	grad_automatic = model.grad_automatic
	grad_finite = model.grad_finite
	grad_analytical = model.grad_analytical

	index = slice(None)
	print("----- grad -----")	
	print(grad_automatic(parameters,state)[index])
	print()
	print("-----")
	print()
	print(grad_finite(parameters,state)[index])
	print()
	print("-----")
	print()	
	print(grad_analytical(parameters,state)[index])
	print()
	print("----- ratio -----")
	print()
	print(grad_automatic(parameters,state)[index]/grad_analytical(parameters,state)[index])
	print()
	print("-----")
	print()
	assert allclose(grad_automatic(parameters,state),grad_finite(parameters,state)), "JAX grad != Finite grad"
	assert allclose(grad_automatic(parameters,state),grad_analytical(parameters,state)), "JAX grad != Analytical grad"
	assert allclose(grad_finite(parameters,state),grad_analytical(parameters,state)), "Finite grad != Analytical grad"

	print("Passed")

	return


def test_module(*args,**kwargs):

	kwargs = {
		"module.N":[2],"module.M":[1],
		"model.N":[2],"model.D":[2],"model.M":[1],"model.ndim":[2],
		"state.N":[None],"state.D":[2],"state.ndim":[2],
		"measure.D":[2],"measure.architecture":["tensor","array"],
		}	

	groups = None
	filters = None
	func = None

	data = {}
	for index,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters,func=func)):

		settings = Dict({
		"cls":{
			"module":"src.quantum.Module",
			"measure":"src.quantum.Measure",
			"model":"src.quantum.Operators",
			"state":"src.quantum.State",
			"callback":"src.quantum.Callback"
			},
		"module":{
			"N":3,
			"M":1,
			"string":"module",
			"measure":{"string":"pauli","operator":"pauli","architecture":"tensor","options":{"cyclic":False}},
			"options":{"contract":"swap+split","max_bond":1000,"cutoff":0},
			"configuration":{
				"key":[lambda value,iterable: (
					value.site[0]%2,value.site[0],-value.locality,[id(iterable[i]) for i in iterable].index(id(value)),
					)],
				"sort":None,
				"reverse":False
				}			
		},
		"measure":{
			"operator":"pauli",
			"D":2,"dtype":"complex",
			"architecture":"tensor",
			"options":{"cyclic":False},
		},		
		"model":{
			"data":{
				# "xx":{
				# 	"operator":["X","X"],"site":"<ij>","string":"XX",
				# 	"parameters":0.5,"variable":False
				# },				
				# "noise":{
				# 	"operator":["dephase","dephase"],"site":"<ij>","string":"dephase",
				# 	"parameters":1e-8,"variable":False
				# },
				"unitary":{
					"operator":"haar","site":"|ij|","string":"unitary",
					"parameters":None,"variable":False,"ndim":2,"seed":123
				},				
				"noise":{
					"operator":["depolarize","depolarize"],"site":"|ij|","string":"depolarize",
					"parameters":1e-8,"variable":False,"ndim":3,"seed":123
				},				
			},
			"N":4,
			"D":2,
			"local":True,
			"space":"spin",
			"time":"linear",
			"lattice":"square",
			"architecture":"array",
			"configuration":{
				"key":[lambda value,iterable: (
					value.site[0]%2,value.site[0],-value.locality,[id(iterable[i]) for i in iterable].index(id(value)),
					)],
				"sort":None,
				"reverse":False
				}
			},
		"state": {
			"operator":"haar",
			"site":None,
			"string":"psi",
			"parameters":None,
			"D":2,
			"ndim":2,
			"local":False
			},
		"callback":{
			"attributes":{
				"N":"N","M":"N","d":"d","D":"state.D",
				"noise.parameters":"noise.parameters",
				"objective":"objective",
				"operator":"measure.operator"
				},
			"options":{"contract":False,"max_bond":None,"cutoff":0}
		},
		"system":{
			"dtype":"complex",
			"format":"array",
			"device":"cpu",
			"backend":None,
			"architecture":None,
			"base":None,
			"seed":1234567890,
			"key":None,
			"instance":None,
			"cwd":None,
			"path":"data.hdf5",
			"conf":"logging.conf",
			"logger":None,
			"cleanup":False,
			"verbose":False
			}
		})

		verbose = False
		precision = 8

		parse = lambda data: data.round(precision)

		# Settings
		setter(settings,kwargs,delimiter=delim,default="replace")
		system = settings.system



		# Model
		model = load(settings.cls.model)		
		model = model(**{**settings.model,**dict(system=system)})
		
		model.info(verbose=verbose)

		parameters = model.parameters()
		state = model.state() if model.state() is not None else model.identity
		kwargs = dict()

		if verbose:
			print(model(parameters=parameters,state=state,**kwargs))
			print()
			print()


		# State
		state = load(settings.cls.state)
		state = state(**{**namespace(state,model),**settings.state,**dict(system=system)})

		state.info(verbose=verbose)

		if verbose:
			print(state())
			print()
			print()

		obj = state

		tmp = tensorprod([obj()]*model.N)

		assert allclose(tmp,(state @ model.N)()),"Incorrect state tensor product"


		# Test

		obj = state
		
		data[index] = {}


		# Measure
		measure = load(settings.cls.measure)		
		measure = measure(**{**settings.measure,**dict(system=system)})


		# Probability
		parameters = measure.parameters()
		state = [obj]*settings.module.N
		kwargs = dict()

		probability = measure.probability(parameters=parameters,state=state,**kwargs)

		if verbose:
			print(probability)

		key = "probability"
		if settings.measure.architecture in ["array"]:
			value = array(probability)
		elif settings.measure.architecture in ["tensor"]:
			value = tensorprod(representation(probability))
		
		data[index][key] = value

		if verbose:
			print(settings.measure.architecture,parse(value),sum(value))

		# Amplitude
		parameters = measure.parameters()
		state = probability
		kwargs = dict()

		amplitude = measure.amplitude(parameters=parameters,state=state,**kwargs)

		key = "amplitude"
		if settings.measure.architecture in ["array"]:
			value = array(amplitude)
		elif settings.measure.architecture in ["tensor"]:
			value = array(amplitude)
		
		data[index][key] = value

		if verbose:
			print(settings.measure.architecture,parse(value),trace(value))

		tmp = tensorprod([obj()]*model.N).T

		assert allclose(tmp,value),"Incorrect probability <-> amplitude conversion"


		# Operator
		parameters = model.parameters()
		state = obj @ model.N

		model.init(state=state)


		parameters = model.parameters()
		state = [obj]*model.N
		kwargs = dict()

		state = measure.probability(parameters=parameters,state=state,**kwargs)

		where = model.site

		operator = measure.operation(parameters=parameters,state=state,model=model,where=where,**kwargs)

		key = "operator"
		if settings.measure.architecture in ["array"]:
			value = array(operator(parameters=parameters,state=state,**kwargs))
		elif settings.measure.architecture in ["tensor"]:
			value = representation(operator(parameters=parameters,state=state,**kwargs),to="tensor",contract=True)

		data[index][key] = value

		if verbose or True:
			print(settings.measure.architecture,parse(value))

		# tmp = model(parameters=model.parameters(),state=(obj @ model.N)())

		# print(tmp)
		# print(measure.amplitude(parameters=parameters,state=operator(parameters=parameters,state=state,**kwargs),**kwargs))

		# assert allclose(tmp,measure.amplitude(parameters=parameters,state=operator(parameters=parameters,state=state,**kwargs),**kwargs)), "Incorrect model <-> operator conversion"

		continue



		# Callback
		callback = load(settings.cls.callback)
		callback = callback(**{**settings.callback,**dict(system=system)})


		# Module
		module = load(settings.cls.module)

		model = model		
		state = obj
		callback = callback

		module = module(**{**settings.module,**dict(callback=callback,system=system)})

		module.init(model=model,state=state)

		module.info(verbose=verbose)
		exit()

		parameters = module.parameters()
		state = module.state()
		kwargs = dict()

		state = module(parameters,state)

		key = "module"
		value = module.measure.transform(parameters=parameters,state=state,transformation=False,**kwargs)
		data[index][key] = value

		# Init
		model = load(settings.cls.model)
		state = load(settings.cls.state)
		system = settings.system

		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(system=system)})

		model.init(state=state)

		parameters = model.parameters()
		state = model.state()

		key = "init"
		value = model(parameters,state)

		data[index][key] = value

		print({i:{attr: getattr(model.data[i],attr,None) for attr in ["string","operator","site"]} for i in model.data})

		if settings.module["options"]["contract"] is False or settings.module["options"]["cutoff"] <= 1e-16:
			assert allclose(data[i]["module"],data[i]["init"]), "Incorrect module() and model() ---\n%s\n%s\n%s"%(
				parse(data[i]["module"]),parse(data[i]["init"]),parse(data[i]["module"] - data[i]["init"]))

		parameters = module.parameters()
		state = module.state()
		options = {"contract":"swap+split","max_bond":8,"cutoff":1e-8}

		state,other = module(parameters,state,**options),module(parameters,state)


		attrs = ["norm_classical","norm_quantum","infidelity_classical","infidelity_quantum"]
		for attr in attrs:
			print(attr,getattr(module.measure,attr)(parameters,state,other=other))
		exit()

		key = "infidelity"
		value = abs(
			(module.measure.infidelity(parameters,state,other) -
			 module.measure.infidelity(parameters,state,state)) / 
			(module.measure.infidelity(parameters,state,state))
			)

		data[index][key] = value


		module.dump()

		module.load()


	assert all(equalizer(data[i],data[j]) for i in data for j in data if i != j), "Error - Inconsistent models"

	print("Passed")

	return



if __name__ == "__main__":

	arguments = {"path":"config/settings.json","tol":5e-8}
	args = argparser(arguments)

	# main(*args,**args)
	# test_basis(*args,**args)
	# test_operator(*args,**args)
	# test_data(*args,**args)
	# test_initialization(*args,**args)
	# test_tensorproduct(*args,**args)
	# test_random(*args,**args)
	# test_layout(*args,**args)
	# test_measure(*args,**args)
	# test_metric(*args,**args)
	# test_namespace(*args,**args)
	# test_objective(*args,**args)
	# test_grad(*args,**args)
	test_module(*args,**args)
