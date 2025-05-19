#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ["",".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,vmap,partial,array,zeros,ones,identity,empty,rand,haar,allclose,asscalar,is_array,is_nan,product
from src.utils import einsum,symbols,conjugate,dagger,dot,tensorprod,reshape,transpose,trace,real,imag,sqrtm,sqrt,cos,sin,abs2,log,log2,log10
from src.utils import shuffle,swap,seeder,rng,copy
from src.utils import arrays,tensors,iterables,scalars,integers,floats,pi,e,delim
from src.iterables import permutations
from src.io import load,dump,glob
from src.call import rm,echo
from src.system import Dict,Dictionary
from src.iterables import namespace,permuter,setter,getter
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
# logger = Logger()

from src.quantum import Basis as basis


def equalizer(a,b):
	try:
		if isinstance(a,arrays) and isinstance(b,arrays):
			return all(allclose(i,j) or (is_nan(i) or is_nan(j)) for i,j in zip(a.ravel(),b.ravel()))
		elif isinstance(a,dict) and isinstance(b,dict):
			return all(allclose(a[i],b[j]) or (is_nan(i) or is_nan(j)) for i,j in zip(a,b))
		elif isinstance(a,iterables) and isinstance(b,iterables):
			return all(allclose(i,j) or (is_nan(i) or is_nan(j)) for i,j in zip(a,b))
		else:
			return a==b
	except:
		return False

def test_basis(*args,**kwargs):

	D = 2
	N = 1
	L = 1
	K = 2
	ndim = 2
	shape = [D**L]*(K if ndim is None else ndim)
	operator="zero.depolarize.X"
	key = 123456789
	delim = "."
	dtype = "complex"

	options = Dict(D=D,N=N,ndim=ndim,shape=shape,operator=operator,key=key,dtype=dtype)

	operators = {
		"rand":Dict(localities=L,shapes={i:[options.D]*L for i in range(K if ndim is None else ndim)},dimensions=2),
		"X":Dict(localities=1,shapes={i:[options.D]*options.N for i in range(K)},dimensions=2),
		"depolarize":Dict(localities=1,shapes={**{i:[options.D**2]*options.N for i in range(1)},**{i:[options.D]*options.N for i in range(1,K+1)}},dimensions=3),
		"string":Dict(localities=len(operator.split(delim)),shapes={0:[1,options.D**2,1],1:[2,options.D,options.D],2:[options.D]*len(operator.split(delim))},dimensions=3),
		"pauli":Dict(localities=1,shapes={**{i:[options.D**2]*options.N for i in range(1)},**{i:[options.D]*options.N for i in range(1,K+1)}},dimensions=3),		
		}

	for operator in operators:
		data = basis.get(operator)(**options)

		print(operator)
		print(data)
		for attr in operators[operator]:
			print(attr,getattr(basis,attr)(operator,**options),operators[operator][attr])
			assert operators[operator][attr] == getattr(basis,attr)(operator,**options)
		print()


	args = ()
	kwargs = Dictionary(D=2,dtype='complex')

	operators = {
		'I':array([[1,0],[0,1]],dtype=kwargs.dtype),
		'X':array([[0,1],[1,0]],dtype=kwargs.dtype),
		'Y':array([[0,-1j],[1j,0]],dtype=kwargs.dtype),
		'Z':array([[1,0],[0,-1]],dtype=kwargs.dtype)
		}

	for operator in operators:

		data = getattr(basis,operator)(*args,**kwargs)

		print(operator)
		print(data.round(8))
		print()

		assert (kwargs.D > 2) or allclose(data,operators[operator]),"Incorrect %s"%(operator)


	print("Passed")

	return


def test_component(*args,**kwargs):

	settings = Dict({
			"cls":{
				"operator":"src.quantum.Operator",
				"state":"src.quantum.State",
				"basis":"src.quantum.Basis"
			},
			"operator":{
				"data":None,"operator":None,"where":None,"string":None,
				"N":2,"D":2,"ndim":2,"local":True,"variable":True,"constant":False,
				"system":{"seed":123,"dtype":"complex","architecture":None}				
			},	
			"state": {
				"data":"src.functions.state",
				"operator":"data",
				"where":None,
				"string":"psi",
				"parameters":None,
				"N":1,"D":2,"ndim":2,
				"system":{"seed":123,"dtype":"complex","architecture":None}
				},
		})

	verbose = True

	state = load(settings.cls.state)


	data = array([
		[ 0.19470377-0.j,-0.32788293+0.22200675j],
		[-0.32788293-0.22200675j,0.80529623+0.j]
		])

	settings.state.data = data

	state = state(**settings.state)

	basis = 'pauli'
	indices = {
		'I':(data[0,0]+data[1,1]),'X':(data[0,1]+data[1,0]),
		'Y':1j*(data[0,1]-data[1,0]),'Z':(data[0,0]-data[1,1])
		}

	state.info(verbose=verbose)

	assert allclose(data,state()),"Incorrect state item initialization"

	for index in indices:
		data = state.component(index=index,basis=basis)
		print(index,data)
		assert allclose(data,indices[index]),"Incorrect component"

	return


def test_null(*args,**kwargs):

	settings = Dict({
			"cls":{
				"model":"src.quantum.Operators",
				"operator":"src.quantum.Operator",
				"state":"src.quantum.State",
				"basis":"src.quantum.Basis"
			},
			"model":{
				"data":{
					"two":{
						"operator":"haar","where":"<ij>","string":"two",
						"parameters":None,"variable":False,"ndim":2,"seed":123456789
					},
					"one":{
						"operator":"haar","where":"ij","string":"one",
						"parameters":None,"variable":False,"ndim":2,"seed":123456789
					}															
				},
				"N":1,
				"D":2,
				"local":True,
				"space":"spin",
				"time":"linear",
				"lattice":"square",
				"architecture":"array",
				"configuration":{
					"key":"src.functions.key",
					"sort":None,
					"reverse":False
					}
			},			
			"operator":{
				"data":None,"operator":None,"where":None,"string":None,
				"N":2,"D":2,"ndim":2,"local":True,
				"system":{"seed":123,"dtype":"complex","architecture":None}				
			},	
			"state": {
				"data":"src.functions.state",
				"operator":"data",
				"where":None,
				"string":"psi",
				"parameters":None,
				"N":1,"D":2,"ndim":2,
				"system":{"seed":123,"dtype":"complex","architecture":None}
				},
		})

	verbose = True

	operator = load(settings.cls.operator)

	operator = operator(**settings.operator)

	operator.info(verbose=verbose)


	model = load(settings.cls.model)

	model = model(**settings.model)

	model.info(verbose=verbose)


	print('Passed')

	return


		
def test_operator(*args,**kwargs):

	data = {}

	kwargs = {
		"operator.N":[4],"state.N":[4],
		"operator.D":[2],"state.D":[2],
		"operator.ndim":[None],"state.ndim":[2,1],
		"operator.local":[True,False],
		"operator.tensor":[True,False],
		"state.tensor":[True,False],
		"operator.data":[None,None,None,None,None,None,None],
		"operator.operator":["CNOT.H",["X","Z"],"haar","haar",["U","U"],["u"],["depolarize","amplitude","dephase"]],
		"operator.where":[[3,0,1],[0,2],None,[1,2,0],[3,1],[0],[0,2,1]],
		"operator.string":["test","xz","Haar","haar","U","u","noise"],
		"operator.parameters":[None,0.25,None,None,None,None,1e-6],
		"operator.variable":[False,False,False,False,False,False,False,False],
		"operator.constant":[True,False,False,False,False,False,True,True],
		"state.local":[False],
		"state.data":[None],
		"state.operator":["state"],
		"state.where":[None],
		"state.string":["state"],
		"state.parameters":[None],
		"state.variable":[False],
		"state.constant":[True],


		}
	groups = [
		[
		"operator.data","operator.operator","operator.where",
		"operator.string","operator.parameters","operator.variable","operator.constant"
		],
		[
		"state.data","state.operator","state.where",
		"state.string","state.parameters","state.variable","state.constant"
		],
		[	
		"operator.tensor","state.tensor",
		],		
		]
	filters = lambda kwargs:[i for i in kwargs if all([
		# i['operator.string'] in ["noise","xz"],
		# i['state.ndim'] in [2],
		# i['operator.local'] in [False],
		# i['operator.tensor'] in [False],
		])]
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
				"data":None,"operator":None,"where":None,"string":None,
				"N":2,"D":2,"ndim":2,"tensor":False,"local":True,"variable":True,"constant":False,
				"system":{"seed":123,"dtype":"complex","architecture":None}				
			},	
			"state": {
				"data":None	,
				"operator":"state",
				"where":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":123,"dtype":"complex","architecture":None}
				},
		})

		setter(settings,kwargs,delimiter=delim,default=True)

		verbose = False


		# Class
		operator = load(settings.cls.operator)
		operator = operator(**settings.operator)

		state = load(settings.cls.state)
		state = state(**settings.state)

		operator.info(verbose=verbose)
		state.info(verbose=verbose)


		# Data
		N,D,L,d,l,where = state.N,state.D,operator.locality,state.ndim,2,operator.where
		axes = [operator.where.index(i) for i in sorted(operator.where)] if operator.local else [*operator.where,*(i for i in range(operator.N) if i not in operator.where)]
		shape = {**{i:[operator.shape[i]] for i in range(operator.ndim-2)},**{i:[operator.D]*(operator.locality if operator.local else operator.N) for i in range(operator.ndim-2,operator.ndim)}} if operator.ndim > 2 else {i:[operator.D]*(operator.locality if operator.local else operator.N) for i in range(operator.ndim-2,operator.ndim)}

		if operator.string in ["test"]:
			obj = [
				array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],**options),
				(1/sqrt(2))*array([[1,1],[1,-1]],**options),
				]
		elif operator.string in ["xz"]:
			obj = [
				array([[0,1],[1,0]],**options),
				array([[1,0],[0,-1]],**options),
				]
		elif operator.string in ["Haar"]:
			obj = [
				haar(shape=(2**4,)*2,seed=seeder(operator.seed),**options),
				]
		elif operator.string in ["haar"]:
			obj = [
				haar(shape=(2**3,)*2,seed=seeder(operator.seed),**options),
				]
		elif operator.string in ["U"]:
			obj = [
				haar(shape=(2**1,)*2,seed=seeder(operator.seed),**options),
				haar(shape=(2**1,)*2,seed=seeder(operator.seed),**options),
				]
		elif operator.string in ["u"]:
			obj = [
				haar(shape=(2,)*2,seed=seeder(operator.seed),**options),
				]
		elif operator.string in ["noise"]:
			obj = [
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

			shape.update({**{i:[operator.D**2,operator.D,operator.D,*([1]*(operator.N-operator.locality) if not operator.local else [])]
				for i in range(operator.ndim-2)},
			 **{i:[operator.D]*(operator.locality if operator.local else operator.N) for i in range(operator.ndim-2,operator.ndim)},
				})

		obj = [*obj,*[reshape(identity(operator.D),[*[1]*(operator.ndim-l),*[operator.D]*l])]*(operator.N-operator.locality)] if not operator.local else obj

		if not operator.local:
			obj = swap(tensorprod(obj),axes=axes,shape=shape)
		else:
			obj = tensorprod(obj)

		if operator.tensor:
			_shape = [*([-1] if operator.ndim>2 else []),*[operator.D]*(l*(operator.locality if operator.local else operator.N))]
			obj = reshape(obj,_shape)


		if not operator.constant:
			_identity = operator.identity(operator.locality if operator.local else operator.N)
			_parameters = operator.parameters(operator.parameters())
			obj = cos(_parameters)*_identity + -1j*sin(_parameters)*obj


		print(
			'operator',{attr:getattr(operator,attr) for attr in ['local','tensor','shape','N','locality','string']},
			'state',{attr:getattr(state,attr) for attr in ['shape']},
			'obj',{attr:getattr(obj,attr) for attr in ['shape']}
			)


		# Operator
		operator.init(N=operator.locality if operator.local else operator.N,where=list(range(operator.locality)) if operator.local else operator.where)


		data = operator(parameters=operator.parameters(),state=operator.identity(N=operator.locality if operator.local else operator.N,D=operator.D),**dict())

		if not operator.unitary:
			data = operator.data

		test = obj

		if verbose:
			print(operator.string)
			print("----- data ------")
			print(data)
			print("-----------------")
			print()
			print("----- _data ------")
			print(obj)
			print("-----------------")
			print()

		assert allclose(data,test), "Incorrect operator %r"%(operator)



		# State
		operator.init(state=state)

		data = operator(parameters=operator.parameters(),state=operator.state(),**dict())


		test = obj

		if operator.local:
			test = tensorprod([reshape(test,[*[-1]*(operator.ndim-l),*[D**L]*l]),reshape(identity(D**(N-L)),[*[1]*(operator.ndim-l),*[D**(N-L)]*l])])

			test = reshape(
					transpose(
					reshape(
						reshape(test,[*[-1]*(operator.ndim-l),*[D**N]*l]),
						[*[-1]*(operator.ndim-l),*[D]*(N*l)]),
						[*range(operator.ndim-l),*[operator.ndim-l+N*j+[*operator.where,*sorted(set(range(N))-set(operator.where))].index(i) for j in range(l) for i in range(N)]]),
						[*[-1]*(operator.ndim-l),*[D**N]*(l)]
					)
		elif operator.tensor:
			test = reshape(test,[*[-1]*(operator.ndim-l),*[D**N]*l])

		test = reshape(test,[*[-1]*(operator.ndim-l),*[D**N]*l])
		tmp = reshape(state(),[D**N]*state.ndim)

		if state.ndim == 2:
			if operator.ndim == 3:
				test = einsum('uij,jk,ulk->il',test,tmp,conjugate(test))
			elif operator.ndim == 2:
				test = einsum('ij,jk,lk->il',test,tmp,conjugate(test))
			else:
				raise NotImplementedError
		elif state.ndim == 1:
			if operator.ndim == 3:
				test = einsum('uij,j->i',test,tmp)
			elif operator.ndim == 2:
				test = einsum('ij,j->i',test,tmp)
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError			


		data = data.ravel()
		test = test.ravel()

		if verbose:
			print(operator.string)
			print("----- state ------")
			print(data)
			print("-----------------")
			print()
			print("----- _state ------")
			print(test)
			print("-----------------")
			print()


		assert allclose(data,test), "Incorrect operator(parameters,state) %r"%(operator)

	print("Passed")

	return	


def test_data(path,tol):

	def trotter(iterable=None,p=None,verbose=False):
		'''
		Trotterized iterable for order p or parameters for order p
		Args:
			iterable (iterable): Iterable
			p (int): Order of trotterization
			verbose (bool,int,str): Verbosity of function		
		Returns:
			iterables (iterable,scalar): Trotterized iterable for order p or parameters for order p if iterable is None
		'''
		P = 2
		if p is None and iterable is None:
			return p
		elif p is None:
			return iterable
		elif not isinstance(p,integers) or (p > P):
			raise NotImplementedError('p = %r !< %d Not Implemented'%(p,P))

		if iterable is None:
			options = {i:1/p for i in range(P+1)}
			iterables = options[p]
		else:
			options = {i:[slice(None,None,(-1)**i)] for i in range(P)}
			iterables = []        
			for i in range(p):
				for indices in options[i]:
					iterables += iterable[indices]

		return iterables


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
	tensor = model.tensor

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

	assert allclose(model.identity().ravel(),identity.ravel()), "Incorrect model identity"

	data = trotter(data,P)
	string = trotter(string,P)
	datas = trotter([model.data[i].data for i in model.data if model.data[i].unitary],P)
	where = trotter([model.data[i].where for i in model.data if model.data[i].unitary],P)

	for i,(s,d,D,index) in enumerate(zip(string,data,datas,where)):
		assert allclose(d.ravel(),D.ravel()), "data[%s,%d] incorrect"%(s,i)
	
	print("Passed")
	
	return


def test_copy(*args,**kwargs):

	kwargs = {
		"operator.N":[3],"operator.D":[2],"operator.local":[True],
		"state.N":[3],"state.D":[2],"state.ndim":[2],"state.local":[False],
		}	

	groups = None
	filters = None
	func = None

	data = {}
	for index,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters,func=func)):

		settings = Dict({
		"cls":{
			"operator":"src.quantum.Operator",
			"state":"src.quantum.State",
			},
		"operator":{
				"data":None,"operator":"haar","where":[0,2],"string":None,
				"N":3,"D":2,"ndim":2,"local":True,"variable":True,"constant":False,
				"parameters":0.5
			},		
		"state": {
			"operator":"haar",
			"where":None,
			"string":"psi",
			"parameters":None,
			"D":2,
			"ndim":2,
			"local":False
			},
		"system":{
			"dtype":"complex",
			"format":"array",
			"device":"cpu",
			"backend":None,
			"architecture":None,
			"base":None,
			"seed":123456789,
			"key":None,
			"instance":None,
			"cwd":"data",
			"path":"data.hdf5",
			"conf":"logging.conf",
			"logger":None,
			"cleanup":False,
			"verbose":False
			}
		})

		data[index] = {}

		verbose = True
		precision = 8

		parse = lambda data: data.round(precision)

		# Settings
		setter(settings,kwargs,delimiter=delim,default="replace")
		system = settings.system

		# State
		state = load(settings.cls.state)		
		state = state(**{**settings.state,**dict(system=system)})
		
		# Operator
		operator = load(settings.cls.operator)		
		operator = operator(**{**settings.operator,**dict(system=system)})
		operator.init(state=state)

		operator.info(verbose=verbose)

		parameters = operator.parameters()
		state = operator.state() if operator.state() is not None else operator.identity()
		kwargs = dict()

		tmp = operator(parameters=parameters,state=state)


		# State
		_state = load(settings.cls.state)		
		_state = _state(**{**settings.state,**dict(system=system)})
		
		# _Operator
		_operator = load(settings.cls.operator)		
		_operator = _operator(**operator)
		# _operator.init(state=_state)

		_operator.info(verbose=verbose)

		_parameters = _operator.parameters()
		_state = _operator.state() if _operator.state() is not None else _operator.identity()
		kwargs = dict()

		_tmp = _operator(parameters=_parameters,state=_state)

		print(parse(tmp))
		print(parse(_tmp))

		assert allclose(tmp,_tmp), "Incorrect Object copy"

	print('Passed')

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
				shape=model.shape,size=model.size,ndim=model.ndim,dtype=model.dtype,
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
				unitary=label.unitary,
				shape=label.shape,size=label.size,ndim=label.ndim,dtype=label.dtype),			
			state=Dictionary(
				func=state.__call__,
				data=state(),
				state=state,
				shape=state.shape,size=state.size,ndim=state.ndim,dtype=state.dtype,							
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

	shape = old.state.shape
	UpsiU = reshape(UpsiU,shape) if UpsiU is not None else None

	shape = old.model.shape
	U = reshape(U,shape) if U is not None else None

	shape = old.state.shape
	psi = reshape(psi,shape) if psi is not None else None

	shape = old.model.shape
	K = reshape(K,[-1,*shape]) if K is not None else None

	shape = old.model.shape
	VpsiV = reshape(VpsiV,shape) if VpsiV is not None else None

	shape = old.label.shape
	V = reshape(V,shape) if V is not None else None

	if K is None:
		if psi is None:
			return
		elif old.state.ndim == 1:
			UpsiUtmp = einsum("ij,j->i",U,psi,conjugate(U))
			VpsiVtmp = einsum("ij,j->i",V,psi,conjugate(V))
		elif old.state.ndim == 2:
			UpsiUtmp = einsum("ij,jk,lk->il",U,psi,conjugate(U))
			VpsiVtmp = einsum("ij,jk,lk->il",V,psi,conjugate(V))		
	elif K is not None:
		#TODO: Implement test for multiple layers of noise 
		if psi is None:
			return
		elif old.state.ndim == 1:
			return
		elif old.state.ndim == 2 and model.M == 1:
			UpsiUtmp = einsum("uij,jk,kl,ml,unm->in",K,U,psi,conjugate(U),conjugate(K))
			VpsiVtmp = einsum("ij,jk,lk->il",V,psi,conjugate(V))		
		else:
			return


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
				# 	"operator":["S","CNOT","H"],"where":[3,2,0,5],"string":"cnot-hadamard",
				# 	"parameters":None,
				# 	"variable":False
				# },
				# "data":{
				# 	"operator":["CNOT"],"where":"<ij>","string":"cnot",
				# 	"parameters":None,
				# 	"variable":False
				# },
				# "xx":{
				# 	"operator":["X","X"],"where":"<ij>","string":"xx",
				# 	"parameters":0.5,
				# 	"variable":False
				# },
				"cnot":{
					"operator":["CNOT"],"where":"<ij>","string":"cnot",
					"parameters":None,
					"variable":False
				},								
				"hadamard":{
					"operator":["H"],"where":"i","string":"hadamard",
					"parameters":None,
					"variable":False
				}					
			},
			"N":4,"D":2,"ndim":2,"local":True,
			"system":{
				"seed":123,"dtype":"complex",
				"architecture":None,"configuration":{"key":["where"]},
				}
		},	
		"state": {
			"data":None	,
			"operator":["plus","minus"],
			"where":None,
			"string":"psi",
			"parameters":None,
			"D":2,"ndim":2,"local":True,"variable":False,
			"system":{"seed":123,"dtype":"complex","architecture":None}
			},
		"operator": {
			"operator":["CNOT"],"where":None,"string":"cnot",
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
				print(dict((map(tuple,i) for i in zip([model.data[i].where,model.data[j].where,new.where],[model.data[i].operator,model.data[j].operator,new.operator]))))

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
		"where":None,
		"string":"psi",
		"parameters":None,
		"D":2,"ndim":2,"local":True,"variable":False,
		"system":{"seed":123,"dtype":"complex","architecture":None}
		})

	cls = load(settings.cls.state)


	states = [cls(**{**options,**dict(operator=operator)}) for operator in operators]
	where = [i for i in range(sum(len(operator) if isinstance(operator,iterables) else 1 for operator in operators))]

	new = states[0] 
	for state in states[1:]:
		new @= state

	new.init(where=where)

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
			"where":None,
			"string":"psi",
			"parameters":None,
			"N":3,"D":2,"ndim":2,
			"local":True,"variable":False,
			"seed":123456789,"dtype":"complex"
			},
		"operator": {
			"data":None,
			"operator":"U",
			"where":[0,2],
			"string":"U",
			"parameters":None,
			"N":3,"D":2,"ndim":2,
			"local":True,"variable":False,
			"seed":123456789,"dtype":"complex"
		}
	})

	options = Dictionary(verbose=True,precision=8)

	operator = load(settings.cls.operator)
	operator = operator(**settings.operator)
	operator.info(**options)

	parameters = operator.parameters()
	state = tensorprod([operator.basis.get(operator.default)(**operator)]*operator.N)
	kwargs = Dictionary(seed=seeder(1236789))

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
	kwargs = Dictionary(seed=seeder(123456789))

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


	print()
	print("-------------------")
	print()

	K = 5
	M = 4
	
	operator = load(settings.cls.operator)
	state = load(settings.cls.state)

	operator = operator(**settings.operator)
	state = state(**settings.state)
	
	model = [operator for i in range(K)]
	
	for i in range(K):
		model[i].init(state=state)

	parameters = [model[i].parameters() for i in range(K)]
	state = operator.state()

	kwargs = [Dictionary(seed=seeder(model[i].seed)) for i in range(K)]
	
	states = [None for i in range(M)]

	for j in range(M):
		state = operator.state()
		for i in range(K):
			state = model[i](parameters=parameters[i],state=state,**kwargs[i])
			seed,kwargs[i].seed = rng.split(kwargs[i].seed)
		states[j] = state

	assert all(not allclose(states[i],states[j]) for i in range(M) for j in range(M) if i!=j), "Incorrect seed updates"

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
					"operator":["X","X"],"where":"||ij||","string":"xx",
					"parameters":0.5,
					"variable":False
				},
				"noise":{
					"operator":["depolarize"],"where":"||i.j||","string":"noise",
					"parameters":{"data":[0,1,2,3,4,5,6]},
					"variable":False
				},	
				# "noise":{
				# 	"operator":["depolarize"],"where":"|i.j|","string":"noise",
				# 	"parameters":None,
				# 	"variable":False
				# }				
			},
			"N":6,"D":2,"ndim":2,"local":True,
			"system":{
				"seed":123,"dtype":"complex",
				"architecture":None,"configuration":{
					"key":"src.functions.brickwork",
					"sort":None,
					"reverse":False
					},
				}
		},	
		"state": {
			"data":None	,
			"operator":"zero",
			"where":None,
			"string":"psi",
			"parameters":None,
			"D":6,"ndim":2,"local":True,"variable":False,
			"system":{"seed":123,"dtype":"complex","architecture":None}
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

	attrs = ["string","parameters","N","locality","operator","where",]

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
				"operator":"tetrad",
				"string":"povm",
				"N":3,
				"D":2,
				"dtype":"complex"
			},
			"state": {
				"data":"random"	,
				"operator":"state",
				"where":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":4,"ndim":1,
				"system":{"seed":123,"dtype":"complex","architecture":"array"}
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
		print(measure.inverse)
		print()

		assert allclose(sum(i for i in measure.basis[measure.pointer]),basis.I(D=measure.D,dtype=measure.dtype)), "Incorrect %r basis"%(measure)

	print("Passed")

	return


def test_metric(path,tol):

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
						"operator":["Z"],"where":"i","string":"Z",
						"parameters":0.5,"variable":True
					},
					"xx":{
						"operator":["X","X"],"where":"<ij>","string":"XX",
						"parameters":0.5,"variable":False
					},
					"noise":{
						"operator":["dephase","dephase"],"where":None,"string":"dephase",
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
				"where":None,
				"string":"psi",
				"parameters":True,
				"ndim":2,
				"seed":123456789,
				"architecture":"array"
				},
			"label": {
				"operator":"X.X.X.X",
				"where":None,
				"string":"U",
				"parameters":0.5,
				"ndim":2,
				"seed":123456789
				},
			"measure":{
				"operator":"tetrad",
				"architecture":"array"				
			},
			"system":{
				"dtype":"complex",
				"format":"array",
				"device":"cpu",
				"backend":None,
				"architecture":None,
				"base":None,
				"seed":123456789,
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
			print(attribute,{attr: getattr(attributes[attribute],attr) for attr in ["N","locality","where","hermitian","unitary","architecture"]},">>>>",namespace(attributes[attribute].__class__,model))
		print()

		model.init(state=state)
		label.init(state=state)

		attributes = {"model":model,"label":label}
		for attribute in attributes:
			print(attribute,{attr: getattr(attributes[attribute],attr) for attr in ["N","locality","where","hermitian","unitary","architecture"]},">>>>",namespace(attributes[attribute].__class__,model))
		print()

		print("Call")
		print(state())
		print(model(state=state()))
		print(label(state=state()))
		print()


		print("Measure")
		attributes = {"measure":measure}
		for attribute in attributes:
			print(attribute,{attr: getattr(attributes[attribute],attr) for attr in ["D","operator","data","inverse","basis","architecture"]},">>>>",attributes[attribute].__class__,namespace(attributes[attribute].__class__,model))
		print()

		print()
		print()
		print()

		parameters = model.parameters()
		state = model.state()
		value = model(parameters,state)

		data[index] = value


	assert all(allclose(data[i],data[j]) for i in data for j in data if i < j), "Error - Inconsistent models"

	print("Passed")

	return


def test_objective(path,tol):

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
	label = partial(model,parameters=parameters,state=state)

	metric = Metric(state=state,label=label,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system)
	func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system)

	out = func(parameters,state=state)

	assert allclose(0,out), "Incorrect objective %0.5e"%(out)

	print("Passed")

	return

def test_grad(path,tol):

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
	print(grad_automatic(parameters,state).shape,grad_finite(parameters,state).shape,grad_analytical(parameters,state).shape)
	assert allclose(grad_automatic(parameters,state),grad_finite(parameters,state)), "JAX grad != Finite grad"
	assert allclose(grad_automatic(parameters,state),grad_analytical(parameters,state)), "JAX grad != Analytical grad"
	assert allclose(grad_finite(parameters,state),grad_analytical(parameters,state)), "Finite grad != Analytical grad"

	print("Passed")

	return

def test_module(*args,**kwargs):

	from importlib import reload
	import src

	os.environ['NUMPY_BACKEND'] = 'quimb'
	reload(src.utils)
	reload(src.quantum)

	from src.utils import representation_quimb,tensors_quimb,matrices_quimb,objects_quimb

	kwargs = {
		"module.N":[2],"module.M":[5],"module.measure.operator":["tetrad"],
		"model.N":[2],"model.D":[2],"model.M":[5],"model.ndim":[2],"model.local":[True],"model.tensor":[True],
		"state.N":[None],"state.D":[2],"state.ndim":[2],"state.local":[False],"state.tensor":[True],
		"measure.N":[2],"measure.D":[2],"measure.operator":["tetrad"],

		"module.measure.architecture":["tensor","tensor_quimb","array"],
		"measure.architecture":["tensor","tensor_quimb","array"],
		"module.options":[{"S":None,"scheme":"svd"},{"contract":"swap+split","max_bond":None,"cutoff":0},{"periodic":False}],
		"module.measure.options":[{"periodic":False},{"periodic":False},{"periodic":False}],
		"measure.options":[{"periodic":False},{"periodic":False},{"periodic":False}],
		"callback.options":[{"S":None,"scheme":"svd"},{"contract":True,"max_bond":None,"cutoff":0},{}],
		}	

	groups = ["module.measure.architecture","measure.architecture","module.options","module.measure.options","measure.options","callback.options"]
	filters = lambda kwargs:[i for i in kwargs if i['module.measure.architecture'] in ["tensor","tensor_quimb","array"]]
	func = None

	data = {}
	for index,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters,func=func)):

		print(kwargs)

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
			"measure":{"string":"tetrad","operator":"tetrad","D":2,"dtype":"complex","seed":13579,"architecture":"tensor","options":{"periodic":False}},
			"options":{},
			"configuration":{
				"key":"src.functions.brickwork",
				"sort":None,
				"reverse":False
				}			
		},
		"measure":{
			"operator":"tetrad",
			"D":2,"dtype":"complex",
			"architecture":"tensor",
			"options":{"periodic":False},
		},		
		"model":{
			"data":{
				# "local":{
				# 	"operator":"haar","where":"i","string":"local",
				# 	"parameters":None,"variable":False,"ndim":2,"seed":123456789
				# },
				"unitary":{
					"operator":["X","Z"],"where":"||ij||","string":"unitary",
					"parameters":0.25,"variable":True,"constant":None,"ndim":2,"seed":123456789
				},
				# "unitary":{
				# 	"operator":"haar","where":"||ij||","string":"unitary",
				# 	"parameters":None,"variable":False,"ndim":2,"seed":123456789
				# },				
				"noise":{
					"operator":["depolarize"],"where":"||i.j||","string":"noise",
					"parameters":1e-6,"variable":False,"ndim":3,"seed":123456789
				},
				# "unitary":{
				# 	"operator":["X","X"],"where":"||ij||","string":"unitary",
				# 	"parameters":"random","variable":True,"constant":False,"ndim":2,"seed":123456789
				# },				
				# "noise":{
				# 	"operator":["depolarize","depolarize"],"where":"||ij||","string":"noise",
				# 	"parameters":1e-6,"variable":False,"ndim":3,"seed":123456789
				# },					
				# "xx":{
				# 	"operator":["X","X"],"where":"<ij>","string":"xx",
				# 	"parameters":0.2464,"variable":False,"ndim":2,"seed":123456789
				# },												
			},
			"N":4,
			"D":2,
			"local":True,
			"space":"spin",
			"time":"linear",
			"lattice":"square",
			"architecture":"array",
			"configuration":{
				"key":"src.functions.brickwork",
				"sort":None,
				"reverse":False
				}
			},
		"state": {
			"operator":"haar",
			"where":None,
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
			"options":{}
		},
		"system":{
			"dtype":"complex",
			"format":"array",
			"device":"cpu",
			"backend":None,
			"architecture":None,
			"base":None,
			"seed":123456789,
			"key":None,
			"instance":None,
			"cwd":"data",
			"path":"data.hdf5",
			"conf":"logging.conf",
			"logger":None,
			"cleanup":False,
			"verbose":False
			}
		})

		data[index] = {}

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
		state = model.state() if model.state() is not None else model.identity()
		kwargs = dict()

		if verbose:
			print(model(parameters=parameters,state=state,**kwargs))
			print()
			print()

		# State
		state = load(settings.cls.state)
		settings.state = [
			{
			**settings.state,
			**dict(operator=settings.state.operator[i%len(settings.state.operator)] 
				if not isinstance(settings.state.operator,str) 
				else settings.state.operator)
			} for i in range(model.N)]
		
		state = [state(**{**settings.model,**i,**dict(system=system)})
				for i in settings.state]

		obj = state

		tmp = None
		for i in state:
			tmp = i if tmp is None else tmp @ i

		tmp = tmp().ravel()
		_tmp = tensorprod([i() for i in obj]).ravel()

		assert allclose(tmp,_tmp),"Incorrect state tensor product"

		# Test

		objs = state

		obj = None
		for i in objs:
			obj = i if obj is None else obj @ i

		if verbose and model.N in [1]:
			basis = 'tetrad'
			components = ['I','X','Y','Z']

			print(obj())

			for component in components:
				print(component,obj.component(basis=basis,index=component))
			
			model.init(state=obj)
			
			print(model())

			for component in components:
				print(component,model.component(basis=basis,index=component))
			
			model.init(state=False)

		
		# Measure
		measure = load(settings.cls.measure)		
		measure = measure(**{**settings.measure,**dict(system=system)})

		# Probability
		parameters = measure.parameters()
		state = [i for i in objs]
		kwargs = dict()

		probability = measure.probability(parameters=parameters,state=state,**kwargs)

		if verbose:
			print(probability)

		key = "probability"
		if measure.architecture in ["array"]:
			value = array(probability)
		elif measure.architecture in ["tensor"]:
			value = probability.array().ravel()
		elif measure.architecture in ["tensor_quimb"]:
			value = tensorprod(representation_quimb(probability))
		
		data[index][key] = value

		if verbose:
			print(measure.architecture,parse(value),sum(value))

		# Amplitude
		parameters = measure.parameters()
		state = probability
		kwargs = dict()

		amplitude = measure.amplitude(parameters=parameters,state=state,**kwargs)

		key = "amplitude"
		if measure.architecture in ["array"]:
			value = array(amplitude)
		elif measure.architecture in ["tensor"]:
			value = amplitude.matrix()
		elif measure.architecture in ["tensor_quimb"]:
			value = representation_quimb(amplitude,to=measure.architecture,contraction=True)

		data[index][key] = value

		if verbose:
			print(measure.architecture,parse(value),trace(value))

		tmp = value
		_tmp = tensorprod([i() for i in objs]) 

		if verbose:
			print(measure.architecture)
			print(parse(tmp))
			print(parse(_tmp))

		assert allclose(tmp,_tmp),"Incorrect probability <-> amplitude conversion"

		# Operator
		parameters = model.parameters()
		state = obj

		model.init(state=state)

		model.init(samples=[model.D**2]*model.locality,local=True,tensor=True)

		parameters = model.parameters()
		state = [i for i in objs]
		kwargs = dict()

		state = measure.probability(parameters=parameters,state=state,**kwargs)


		where = model.where
		options = dict(**settings.module.options)


		operator = measure.operation(parameters=parameters,state=state,model=model,where=where,options=options,**kwargs)


		key = "operator"
		if measure.architecture in ["array"]:
			value = array(operator(parameters=parameters,state=state,**kwargs))
		elif measure.architecture in ["tensor"]:
			value = operator(parameters=parameters,state=state,**kwargs).array().ravel()			
		elif measure.architecture in ["tensor_quimb"]:
			value = representation_quimb(operator(parameters=parameters,state=state,**kwargs),to=measure.architecture,contraction=True)

		data[index][key] = value

		if verbose:
			print(measure.architecture,parse(value))


		parameters = model.parameters()
		state = [i for i in objs]

		kwargs = dict()
		
		where = model.where
		options = dict(**settings.module.options)

		tmp = measure.amplitude(
			parameters=parameters,
			state=measure.operation(
				parameters=parameters,
				state=measure.probability(
					parameters=parameters,
					state=state,
					**kwargs),
				model=model,
				where=where,
				options=options,
				**kwargs)(
				parameters=parameters,
				state=measure.probability(
					parameters=parameters,
					state=state,
					**kwargs),
				**kwargs),
			**kwargs)

		model.init(samples=None,local=True,tensor=True)

		_tmp = model(parameters=parameters,state=obj())

		if measure.architecture in ['array']:
			tmp = array(tmp)
		elif measure.architecture in ['tensor']:
			tmp = tmp.matrix()
		elif measure.architecture in ['tensor_quimb']:
			tmp = representation_quimb(tmp,to=measure.architecture,contraction=True)

		if verbose:
			print(parse(tmp))
			print(parse(_tmp))

		tmp = tmp.ravel()
		_tmp = _tmp.ravel()

		assert allclose(tmp,_tmp), "Incorrect model <-> operator conversion"

		# Callback
		callback = load(settings.cls.callback)
		callback = callback(**{**settings.callback,**dict(system=system)})

		# Module
		module = load(settings.cls.module)
		model = load(settings.cls.model)		
		state = load(settings.cls.state)		
		callback = load(settings.cls.callback)		
		system = settings.system

		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**settings.state[0],**dict(system=system)})
		callback = callback(**{**settings.callback,**dict(system=system)})
	
		module = module(**{**settings.module,**dict(model=model,state=state,callback=callback,system=system)})

		module.info(verbose=verbose)
		model.info(verbose=verbose)

		parameters = module.parameters()
		state = module.state()
		kwargs = dict()


		state = module(parameters,state)

		state = module.measure.transform(parameters=parameters,state=state,transformation=False)
	
		if module.measure.architecture in ['array']:
			value = array(state)
		elif module.measure.architecture in ['tensor']:
			value = state.matrix()
		elif module.measure.architecture in ['tensor_quimb']:
			value = representation_quimb(state,to=module.measure.architecture,contraction=True)

		key = 'model'
		data[index][key] = value


		if verbose:
			print(measure.architecture,parse(value))

		
		tmp = value
		
		model.init(state=module.state @ module.N)
		_tmp = model(parameters=module.parameters(),state=model.state())

		if verbose:
			print(parse(tmp))
			print(parse(_tmp))

		tmp = tmp.ravel()
		_tmp = _tmp.ravel()

		assert allclose(tmp,_tmp),"Incorrect Module <-> Model conversion"


	print({i:list(data[i]) for i in data})

	assert all(equalizer(data[i],data[j]) for i in data for j in data if i < j), "Error - Inconsistent models"


	os.environ['NUMPY_BACKEND'] = 'jax'
	reload(src.utils)
	reload(src.quantum)


	print("Passed")

	return


def test_calculate(*args,**kwargs):

	from importlib import reload
	import src

	os.environ['NUMPY_BACKEND'] = 'quimb'
	reload(src.utils)
	reload(src.quantum)

	from src.utils import tensors
	from src.utils import representation_quimb,tensors_quimb,matrices_quimb,objects_quimb

	kwargs = {
		"module.N":[4],"module.M":[3],"module.seed":[123456789],
		"model.N":[4],"model.D":[2],"model.M":[1],"model.ndim":[2],"model.local":[True],"model.tensor":[True],"model.seed":[123456789],
		"model.data.unitary.where":["||ij||"],"model.data.unitary.parameters":[None],"model.data.unitary.seed":[123456789],
		"model.data.noise.where":["||i.j||"],"model.data.noise.parameters":[1e-3],"model.data.noise.seed":[None],
		"state.N":[None],"state.D":[2],"state.ndim":[2],"state.local":[False],"state.tensor":[True],

		"module.measure.D":[2],"module.measure.operator":[["povm","pauli","tetrad","povm","povm","pauli","tetrad","povm","povm","pauli","tetrad","povm"]],"module.measure.symmetry":[None],

		"module.measure.architecture":["tensor","tensor_quimb","array"],
		"module.options":[{"S":None,"scheme":"svd"},{"contract":"swap+split","max_bond":None,"cutoff":0},{"periodic":False}],
		"module.measure.options":[{"periodic":False},{"periodic":False},{"periodic":False}],
		"callback.options":[{"S":None,"scheme":"svd"},{"contract":"swap+split","max_bond":None,"cutoff":0},{}],
		}	

	groups = ["module.measure.architecture","module.options","module.measure.options","callback.options"]
	filters = None
	func = None

	data = {}
	for index,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters,func=func)):

		settings = Dict({
		"cls":{
			"module":"src.quantum.Module",
			"model":"src.quantum.Operators",
			"state":"src.quantum.State",
			"callback":"src.quantum.Callback"
			},
		"module":{
			"N":4,
			"M":5,
			"string":"module",
			"measure":{
				"operator":"tetrad",
				"D":2,"dtype":"complex","seed":13579,
				"architecture":"tensor",
				"options":{"periodic":False},
				},	
			"options":{"contract":"swap+split","max_bond":None,"cutoff":0},
			"configuration":{
				"key":"src.functions.brickwork",
				"sort":None,
				"reverse":False
				}
		},
		"model":{
			"data":{
				# "local":{
				# 	"operator":"haar","where":"i","string":"local",
				# 	"parameters":None,"variable":False,"ndim":2,"seed":123456789
				# },
				"unitary":{
					"operator":"haar","where":"||ij||","string":"unitary",
					"parameters":None,"variable":False,"ndim":2,"seed":13579
				},	
				"noise":{
					"operator":["depolarize"],"where":"||i.j||","string":"depolarize",
					"parameters":1e-3,"variable":False,"ndim":3,"seed":123456789
				},								
				# "xx":{
				# 	"operator":["X","X"],"where":"<ij>","string":"xx",
				# 	"parameters":0.2464,"variable":False,"ndim":2,"seed":123456789
				# },												
			},
			"N":4,
			"D":2,
			"local":True,
			"space":"spin",
			"time":"linear",
			"lattice":"square",
			"architecture":"array",
			"configuration":{
				"key":"src.functions.brickwork",
				"sort":None,
				"reverse":False
				}
			},
		"state": {
			"operator":"state",
			"where":None,
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
				"operator":"measure.operator",
				"trace":"trace",
				"vectorize":"vectorize",
				"measure":"measure",
				"norm_quantum":"norm_quantum",
				"norm_classical":"norm_classical",
				"norm_pure":"norm_pure",
				"infidelity_quantum":"infidelity_quantum",
				"infidelity_classical":"infidelity_classical",
				"infidelity_pure":"infidelity_pure",
				"entanglement_quantum":"entanglement_quantum",
				"entanglement_classical":"entanglement_classical",
				"entanglement_renyi":"entanglement_renyi",
				"entangling_quantum":"entangling_quantum",
				"entangling_classical":"entangling_classical",
				"entangling_renyi":"entangling_renyi",
				"mutual_quantum":"mutual_quantum",
				"mutual_measure":"mutual_measure",
				"mutual_classical":"mutual_classical",
				"mutual_renyi":"mutual_renyi",
				"discord_quantum":"discord_quantum",
				"discord_classical":"discord_classical",
				"discord_renyi":"discord_renyi",
				"spectrum_quantum":"spectrum_quantum",
				"spectrum_classical":"spectrum_classical",
				"rank_quantum":"rank_quantum",
				"rank_classical":"rank_classical",
				},
			"options":{}
		},
		"system":{
			"dtype":"complex",
			"format":"array",
			"device":"cpu",
			"backend":None,
			"architecture":None,
			"base":None,
			"seed":123456789,
			"key":None,
			"instance":None,
			"cwd":"data",
			"path":"data.hdf5",
			"conf":"logging.conf",
			"logger":None,
			"cleanup":False,
			"verbose":False
			}
		})

	
		attrs = [
			'trace',
			'vectorize',
			'measure',
			'square',
			'norm_quantum',
			'norm_classical',
			'norm_pure',
			'infidelity_quantum',
			'infidelity_classical',
			'infidelity_pure',
			'entanglement_quantum',
			'entanglement_classical',
			'entanglement_renyi',
			'entangling_quantum',
			'entangling_classical',
			'entangling_renyi',
			'mutual_quantum',
			'mutual_measure',
			'mutual_classical',
			'mutual_renyi',
			'discord_quantum',
			'discord_classical',
			'discord_renyi',
			'spectrum_quantum',
			'spectrum_classical',
			'rank_quantum',
			'rank_classical',
			]


		# Verbose
		verbose = False
		precision = 8

		parse = lambda data: data.round(precision)

		data[index] = {}

		# Settings
		setter(settings,kwargs,delimiter=delim,default="replace")
		system = settings.system

		# Class
		module = load(settings.cls.module)
		model = load(settings.cls.model)		
		state = load(settings.cls.state)		
		callback = load(settings.cls.callback)		
		system = settings.system

		# Model
		model = model(**{**settings.model,**dict(system=system)})

		# State
		state = state(**{**settings.state,**dict(system=system)})

		# Callback
		callback = callback(**{**settings.callback,**dict(system=system)})
	
		# Module
		module = module(**{**settings.module,**dict(model=model,state=state,callback=callback,system=system)})

		# Verbose
		model.info(verbose=verbose)
		module.info(verbose=verbose)

		# State
		parameters = module.parameters()
		state = module.state()
		kwargs = dict()

		state = module(parameters,state,**kwargs)


		# Test
		key = 'state'
		if module.measure.architecture in ['array']:
			value = array(state)
		elif module.measure.architecture in ['tensor']:
			value = state.array().ravel()
		elif module.measure.architecture in ['tensor_quimb']:
			value = representation_quimb(state.copy(),contraction=True).ravel()

		if verbose:
			print(parse(value))

		data[index][key] = value

		# Calculate
		for attr in attrs:

			if attr in [
				'infidelity_quantum','infidelity_classical','infidelity_pure',
				]:
				
				kwargs = dict(
					other=module(
						parameters=module.parameters(),
						state=module.state(),
						**dict(options=callback.options[attr])
						)
					)
				where = None

			elif attr in [
				'trace','vectorize',
				]:

				kwargs = dict()
				where = [i for i in range(model.N//4,3*model.N//4)]

			elif attr in [
				'entanglement_quantum','entanglement_classical','entanglement_renyi',
				'entangling_quantum','entangling_classical','entangling_renyi',
				'mutual_quantum','mutual_measure','mutual_classical','mutual_renyi',
				'discord_quantum','discord_classical','discord_renyi',
				'spectrum_quantum','spectrum_classical',
				'rank_quantum','rank_classical',
				]:

				kwargs = dict()
				where = 0.5

			elif attr in [
				'measure',
				]:

				kwargs = dict()
				where = {i:min(model.D**2-1,[1,4,3,2][j%4]) for j,i in enumerate(range(model.N//2,model.N)) if i < model.N}

			elif attr in [
				'square',
				]:

				kwargs = dict()
				where = None

			else:

				kwargs = dict()
				where = None

			obj = module.measure.calculate(attr,state=state,where=where,**kwargs)

			key = attr

			if module.measure.architecture in ['array']:
				value = array(obj).ravel()
			elif module.measure.architecture in ['tensor']:
				value = obj.array().ravel() if isinstance(obj,tensors) else array(obj).ravel()
			elif module.measure.architecture in ['tensor_quimb']:
				value = representation_quimb(obj,to=module.measure.architecture,contraction=True).ravel()

			if verbose or True:
				print(module.measure.architecture,attr,where,value.shape)
				print(parse(value))
				print()

			data[index][key] = value


	for i in data:
		for j in data:
			if i >= j:
				continue
			for k in data[i]:
				if not allclose(data[i][k],data[j][k]):
					print(i,j,k)

	assert all(equalizer(data[i],data[j]) for i in data for j in data if i < j), "Error - Inconsistent calculations"


	os.environ['NUMPY_BACKEND'] = 'jax'
	reload(src.utils)
	reload(src.quantum)


	print("Passed")

	return


def test_mps(*args,**kwargs):
	from importlib import reload
	import src

	from src.utils import array,allclose,seeder,permutations,tensorprod,inv

	def representation_quimb(data):
		return data.to_dense().ravel()

	def tensor(data,t=1):
		return array([tensorprod(i) for i in permutations(*[data]*t)])
	
	def init(N,D,S,L,architecture,**kwargs):
		
		if architecture in ['tensor']:
			os.environ['NUMPY_BACKEND'] = 'jax'
			reload(src.utils)
			reload(src.quantum)
			from src.quantum import MPS as mps
		elif architecture in ['tensor_quimb']:
			os.environ['NUMPY_BACKEND'] = 'quimb'
			reload(src.utils)
			reload(src.quantum)			
			from src.quantum import MPS_quimb as mps


		state = {i:{'data':data} 
			for i,data in enumerate(
				[data for i in range(N) for data in ['zero']],
				)
			}
		data = {i:{'data':data,'where':where} 
			for i,(data,where) in enumerate(zip(
			[data for i in range(N-1) for data in ['unitary','depolarize','depolarize']],
			[data for i in range(N-1) for data in [(i,i+1),(i,),(i+1,)]],
			))
			}

		measure = 'tetrad'

		arguments = tuple()
		keywords = dict(
			D=D,shape=(D,)*2,ndim=3,seed=seed,dtype=dtype,
			)
		measure = getattr(basis,measure)(*arguments,**keywords)
		inverse = inv(einsum('uij,vji->uv',measure,measure))

		arguments = tuple()
		keywords = dict(
			D=D,shape=(D,)*2,ndim=2,S=S,seed=seed,dtype=dtype,
			)
		state = {i:getattr(basis,state[i]['data'])(*arguments,**keywords) for i in state}

		state = {i:einsum('uij,ji->u',tensor(measure),state[i]) for i in state}

		state = mps(state,*arguments,**keywords)

		for i in data:
			
			value = data[i]['data']
			where = data[i]['where']
			
			arguments = tuple()
			keywords = dict(
				D=D**len(where),shape=(D**len(where),)*2,ndim=2,seed=seed,dtype=dtype,
				**kwargs
				)
			value = getattr(basis,value)(*arguments,**keywords)

			value = vmap(partial(basis.contract,data=value,where=where,**keywords))(tensor(measure,len(where)))

			value = einsum('uij,wji,wv->uv',tensor(measure,len(where)),value,tensor(inverse,len(where)))

			if architecture in ['tensor']:
				value = basis.shuffle(value,shape=[D**2]*len(where))
			elif architecture in ['tensor_quimb']:
				pass

			data[i] = {'data':value,'where':where}


		return state,data


	N = 4
	D = 2
	S = int(D**(N//1))
	M = 10
	L = 2
	T = 1
	architecture = 'tensor'
	# architecture = 'tensor_quimb'
	architecture = 'all'
	parameters = 1e-3,
	seed = 123456789
	seed = seeder(seed)
	dtype = 'complex'
	options = dict(scheme='svd',rank=S)
	options_quimb = dict(
		contract="swap+split",
		max_bond=S,
		cutoff=0
		)
	kwargs = dict(
		parameters=parameters,
		)
	state,state_quimb = None,None

	if architecture in ['tensor','all']:
		for i in range(T):
			state,data = init(N,D,S,L,architecture='tensor')
			for k in range(M):
				for i in data:
					value,where = data[i]['data'],data[i]['where']
					state = state(value,where=where,options=options)
		print(state)

	if architecture in ['tensor_quimb','all']:
		for i in range(T):
			state_quimb,data_quimb = init(N,D,S,L,architecture='tensor_quimb')
			for k in range(M):
				for i in data_quimb:
					value,where = data_quimb[i]['data'],data_quimb[i]['where']
					state_quimb = state_quimb.gate(value,where=where,**options_quimb)
		print(state_quimb)

	if state is not None and state_quimb is not None:
		tmp = state.array().ravel()
		tmp_quimb = representation_quimb(state_quimb).ravel()

		print(tmp.sum(),tmp_quimb.sum())

		assert allclose(tmp,tmp_quimb)

		print('Passed')

	return


def test_function(*args,**kwargs):
	D = [2]
	N = [2,4,8,16]
	values = {
		(2,2) : 0.12022458674074693,
		(2,4) : 0.6202836021067056,
		(2,8) : 0.8190755813771532,
		(2,16) : 0.9098304133895853,
		(2,32) : 0.9549157799634724,
		}
	for D,N in permutations(D,N):
		parameters = {'D':D,'N':N}
		
		data = (sum(1/i for i in range(int(parameters['D']**(parameters['N']/2))+1,int(parameters['D']**(parameters['N'])))) - 
			   ((int(parameters['D']**(parameters['N']/2))-1)/(2*int(parameters['D']**(parameters['N']/2)))))
		data = data*2/log(parameters['D']**(parameters['N']/2))/2

		string = f'D={D},N={N} : {data}'

		assert allclose(values[(D,N)],data), "Incorrect %s"%(string)

		print(string)

	print('Passed')

	return

def test_class(*args,**kwargs):

	kwargs = {
		"module.M":[10],"module.measure.operator":["tetrad"],
		"model.N":[8],"model.D":[2],"model.M":[None],"model.ndim":[2],"model.local":[True],"model.tensor":[True],
		"state.N":[None],"state.D":[2],"state.ndim":[2],"state.local":[False],"state.tensor":[True],

		"module.measure.architecture":["tensor","tensor_quimb","array"],
		"module.options":[
			{"S":None,"eps":1e-20,"iters":5e4,"parameters":None,"method":"mu","initialize":"nndsvda","scheme":"nmf","key":seeder(123)},
			# {"S":None,"eps":5e-6,"iters":1e5,"parameters":None,"method":"mu","initialize":"nndsvda","scheme":"nmf","key":seeder(123)},
			# {"S":None,"eps":1e-14,"iters":1e4,"parameters":0e-4,"method":"kl","initialize":"nndsvda","scheme":"nmf","key":seeder(123)},
			# {"S":None,"eps":1e-16,"iters":5e4,"parameters":1e-4,"method":"hals","initialize":"nndsvda","scheme":"nmf","key":seeder(123)},
			# {"S":None,"eps":5e-9,"iters":1e6,"parameters":1e-3,"method":"grad","initialize":"rand","scheme":"nmf","key":seeder(123)},
			# {"S":None,"eps":5e-9,"iters":1e6,"parameters":1e-3,"method":"div","initialize":"rand","scheme":"nmf","key":seeder(123)},
			# {"S":None,"eps":5e-9,"iters":1e3,"parameters":1e-6,"method":"als","initialize":"nndsvda","scheme":"nmf","key":seeder(123)},
			# {"S":None,"eps":1e5,"iters":1e7,"parameters":None,"method":"mu","initialize":"nndsvda","scheme":"svd","key":seeder(123)},
			{"contract":"swap+split","max_bond":None,"cutoff":0},
			{"periodic":False}
			],
		"module.measure.options":[{"periodic":False},{"periodic":False},{"periodic":False}],
		"callback.options":[{"S":None,"eps":None,"parameters":None,"method":None,"initialize":None,"scheme":"svd"},{"contract":True,"max_bond":None,"cutoff":0},{}],
		}	

	groups = ["module.measure.architecture","module.options","module.measure.options","callback.options"]
	filters = lambda kwargs:[i for i in kwargs if (
		i['module.measure.architecture'] in [
			"tensor",
			# "tensor_quimb",
			# "array",
			] 
		)
		]
	func = None

	data = {}
	for index,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters,func=func)):

		settings = Dict({
		"cls":{
			"module":"src.quantum.Module",
			"model":"src.quantum.Operators",
			"state":"src.quantum.State",
			"callback":"src.quantum.Callback"
			},
		"module":{
			"N":3,
			"M":1,
			"string":"module",
			"measure":{"string":"tetrad","operator":"tetrad","D":2,"dtype":"complex","seed":13579,"architecture":"tensor","options":{"periodic":False}},
			"options":{},
			"configuration":{
				"key":"src.functions.nearestneighbour",
				"sort":None,
				"reverse":False
				}			
		},
		"model":{
			"data":{
				# "unitary":{
				# 	"operator":"unitary","where":"||ij||","string":"unitary",
				# 	"parameters":None,"variable":False,"constant":None,"ndim":2,"seed":123456789
				# },
				"XX":{
					"operator":["X","X"],"where":"||ij||","string":"XX",
					"parameters":1e-6,"variable":False,"constant":None,"ndim":2,"seed":123456789
				},				
				# "II":{
				# 	"operator":["I","I"],"where":"||ij||","string":"II",
				# 	"parameters":0,"variable":False,"constant":None,"ndim":2,"seed":123456789
				# },								
				# "noise":{
				# 	"operator":["depolarize"],"where":"||i.j||","string":"noise",
				# 	"parameters":1e-6,"variable":False,"ndim":3,"seed":123456789
				# },
			},
			"N":4,
			"D":2,
			"local":True,
			"space":"spin",
			"time":"linear",
			"lattice":"square",
			"architecture":"array",
			"configuration":{
				"key":"src.functions.nearestneighbour",
				"sort":None,
				"reverse":False
				}
			},
		"state": {
			"operator":"haar",
			"where":None,
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
			"options":{}
		},
		"system":{
			"dtype":"complex",
			"format":"array",
			"device":"cpu",
			"backend":None,
			"architecture":None,
			"base":None,
			"seed":123456789,
			"key":None,
			"instance":None,
			"cwd":"data",
			"path":"data.hdf5",
			"conf":"logging.conf",
			"logger":None,
			"cleanup":False,
			"verbose":False
			}
		})

		data[index] = {}

		verbose = False
		precision = 8

		parse = lambda data: data.round(precision)

		test = False

		# Settings
		setter(settings,kwargs,delimiter=delim,default="replace")
		system = settings.system


		# Backend
		if settings.module.measure.architecture in ["array"]:
			pass
		elif settings.module.measure.architecture in ["tensor"]:
			pass
		elif settings.module.measure.architecture in ["tensor_quimb"]:
			from importlib import reload
			os.environ['NUMPY_BACKEND'] = 'quimb'
			import src
			reload(src.utils)
			reload(src.quantum)
			from src.utils import representation_quimb,tensors_quimb,matrices_quimb,objects_quimb

		# Classes
		module = load(settings.cls.module)
		model = load(settings.cls.model)		
		state = load(settings.cls.state)		
		callback = load(settings.cls.callback)		
		system = settings.system

		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**settings.model,**settings.state,**dict(system=system)})
		callback = callback(**{**settings.callback,**dict(system=system)})
	
		module = module(**{**settings.module,**dict(model=model,state=state,callback=callback,system=system)})

		model.info(verbose=verbose)
		module.info(verbose=verbose)

		parameters = module.parameters()
		state = module.state()
		kwargs = dict()

		state = module(parameters,state)

		if verbose or 1:

			value = module.measure.trace(parameters=parameters,state=state)

			if module.measure.architecture in ['array']:
				value = array(value)
			elif module.measure.architecture in ['tensor']:
				value = value.array().item()
			elif module.measure.architecture in ['tensor_quimb']:
				value = representation_quimb(value,to=module.measure.architecture,contraction=True)

			value = value.real - 1

			print(module.measure.architecture)
			print(state)
			print(value)


		# Value
		if test:
			
			value = module.measure.transform(parameters=parameters,state=state,transformation=False)
			if module.measure.architecture in ['array']:
				value = array(value)
			elif module.measure.architecture in ['tensor']:
				value = value.matrix()
			elif module.measure.architecture in ['tensor_quimb']:
				value = representation_quimb(value,to=module.measure.architecture,contraction=True)


			key = 'model'
			data[index][key] = value


			if verbose:
				print(module.measure.architecture)
				print(parse(value))


			model.init(state=module.state @ module.N)
			_value = model(parameters=module.parameters(),state=model.state())

			if verbose:
				print(parse(value))
				print(parse(_value))

			assert allclose(value,_value),"Incorrect Module <-> Model conversion"


		# Backend
		if module.measure.architecture in ["array"]:
			pass
		elif module.measure.architecture in ["tensor"]:
			pass
		elif module.measure.architecture in ["tensor_quimb"]:
			os.environ.pop('NUMPY_BACKEND')
			reload(src.utils)
			reload(src.quantum)
			del representation_quimb,tensors_quimb,matrices_quimb,objects_quimb


	assert all(equalizer(data[i],data[j]) for i in data for j in data if i < j), "Error - Inconsistent models"

	print("Passed")

	return


if __name__ == "__main__":

	arguments = {"path":"config/settings.json","tol":5e-8}
	args = argparser(arguments)

	# main(*args,**args)
	# test_function(*args,**args)
	# test_basis(*args,**args)
	# test_component(*args,**args)
	# test_operator(*args,**args)
	# test_null(*args,**args)
	# test_data(*args,**args)
	# test_copy(*args,**args)
	# test_initialization(*args,**args)
	# test_tensorproduct(*args,**args)
	# test_random(*args,**args)
	# test_layout(*args,**args)
	# test_measure(*args,**args)
	# test_metric(*args,**args)
	# test_namespace(*args,**args)
	# test_objective(*args,**args)
	# test_grad(*args,**args)
	# test_module(*args,**args)
	# test_calculate(*args,**args)
	# test_mps(*args,**args)
	test_class(*args,**args)
