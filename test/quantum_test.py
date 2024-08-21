#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ["",".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,array,zeros,ones,empty,allclose,product,spawn,einsum,conjugate,dot,tensorprod
from src.utils import arrays,iterables,scalars,integers,floats,pi,delim
from src.iterables import permutations
from src.io import load,dump,glob
from src.call import rm,echo
from src.system import Dict
from src.iterables import namespace,permuter,setter,getter
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
# logger = Logger()

def test_channel(*args,**kwargs):

	data = {}

	kwargs = {
		"cls.model":["src.quantum.Channel","src.quantum.Operators"],
		**{attr:[3] for attr in ["model.N","state.N"]},
		**{attr:["array","mps"] for attr in ["model.system.architecture","state.system.architecture"]},
		"model.M":[10],
		"model.ndim":[2],"state.ndim":[1],
		}
	groups = [["model.N","state.N"],["model.system.architecture","state.system.architecture"],]
	filters = lambda iterables: (iterable for iterable in iterables 
		if not (iterable['cls.model'] in ['src.quantum.Channel'] and 
			    iterable['model.system.architecture'] in ['mps'] and
			    iterable['state.system.architecture'] in ['mps'])
		)

	for i,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters)):
	
		settings = Dict({
			"cls":{
				"model":"src.quantum.Channel",
				"state":"src.quantum.State"
			},
			"model":{
				"data":{
					"x":{
						"operator":["X"],"site":"i","string":"x",
						"parameters":{"data":"random","seed":123},
						"variable":True
					},
					"y":{
						"operator":["Y"],"site":"i","string":"y",
						"parameters":{"data":"random","seed":123},
						"variable":True
					},
					"z":{
						"operator":["Z"],"site":"i","string":"z",
						"parameters":{"data":"random","seed":123},
						"variable":True
					},				
					"zz":{
						"operator":["Z","Z"],"site":"<ij>","string":"zz",
						"parameters":{"data":"random","seed":123},
						"variable":True
					},
					# "noise":{
					# 	"operator":"depolarize","site":None,"string":"noise",
					# 	"parameters":1e-12,
					# 	"ndim":3,
					# 	"variable":False
					# }
				},
				"N":2,"D":2,"ndim":2,"M":4,
				"system":{"seed":12345,"dtype":"complex","architecture":None}				
			},	
			"state": {
				"data":None	,
				"operator":"product",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":20,"D":2,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":None}
				},
		})

		setter(settings,kwargs,delimiter=delim,default=True)

		verbose = True
		ignore = "parameters"

		model = load(settings.cls.model)
		state = load(settings.cls.state)

		state = state(**settings.state)
		model = model(**settings.model,state=state)

		model.init(state=state)


		print('Settings: ',settings.cls.model,settings.cls.state)

		# Model

		value = product([model.data[i].data for i in model.data])

		if kwargs["model.system.architecture"] in ["array"]:
			value = array(value)
		elif kwargs["model.system.architecture"] in ["mps"]:
			value = array(value)

		print("--- model ---")
		model.info(ignore=ignore,verbose=verbose)
		print(model.data)
		print("------")


		# State

		value = state()
		
		if kwargs["model.system.architecture"] in ["array"]:
			value = array(value)
		elif kwargs["model.system.architecture"] in ["mps"]:
			value = value.to_dense().reshape(-1)

		print("--- state ---")
		state.info(ignore=ignore,verbose=verbose)
		print(value)
		print("------")


		# Value
		
		value = model(model.parameters(model.parameters()),model.state())

		if kwargs["model.system.architecture"] in ["array"]:
			value = array(value)
		elif kwargs["model.system.architecture"] in ["mps"]:
			value = value.to_dense().reshape(-1)

		print("--- value ---")
		print(value)
		print("------")

		print()
		print()
		print()

		data[i] = value


	assert all(allclose(data[i],data[j]) for i in data for j in data if i != j), "Error - Inconsistent models"

	print("Passed")

	return


def test_composite(*args,**kwargs):

	data = {}

	kwargs = {
		**{attr:[2] for attr in ["model.N","state.N"]},
		**{attr:["array"] for attr in ["model.system.architecture","state.system.architecture"]},
		"model.M":[1],
		"model.ndim":[2],"state.ndim":[1],
		"model.data":[{
			"operators":{
				"data":{
					"xx":{
						"operator":["X","X"],"site":"<ij>","string":"xx",
						"parameters":0.5,
						"variable":False
					},
				},
				"operator":"operators","site":None,"string":"operators",
				"N":3,"D":2,"ndim":2,
				"variable":False
			},
			"noise":{
				"operator":"depolarize","site":None,"string":"noise",
				"parameters":1e-12,
				"ndim":3,
				"variable":False
			},
		},
		{
			"xx":{
				"operator":["X","X"],"site":"<ij>","string":"xx",
				"parameters":0.5,
				"variable":False
			},
			"noise":{
				"operator":"depolarize","site":None,"string":"noise",
				"parameters":1e-12,
				"ndim":3,
				"variable":False
			},
		}
		]
		}
	groups = [["model.N","state.N"],["model.system.architecture","state.system.architecture"],]

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):
	
		settings = Dict({
			"cls":{
				"model":"src.quantum.Operators",
				"state":"src.quantum.State"
			},
			"model":{
				"data":{
					"channel":{
						"data":{
							"x":{
								"operator":["X"],"site":"i","string":"x",
								"parameters":{"data":"random","seed":123},
								"variable":True
							},
							"y":{
								"operator":["Y"],"site":"i","string":"y",
								"parameters":{"data":"random","seed":123},
								"variable":True
							},
							"z":{
								"operator":["Z"],"site":"i","string":"z",
								"parameters":{"data":"random","seed":123},
								"variable":True
							},				
							"zz":{
								"operator":["Z","Z"],"site":"i<j","string":"zz",
								"parameters":{"data":"random","seed":123},
								"variable":True
							},
						},
						"operator":"channel","site":None,"string":"channel",
						"N":3,"D":2,"ndim":2,
						"variable":True
					},
					"noise":{
						"operator":"depolarize","site":None,"string":"noise",
						"parameters":1e-12,
						"ndim":3,
						"variable":False
					}
				},
				"N":3,"D":2,"ndim":2,
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

		setter(settings,kwargs,delimiter=delim,default="replace")

		verbose = True

		model = load(settings.cls.model)
		state = load(settings.cls.state)

		model = model(**settings.model)

		state = state(**settings.state)

		model.init(state=state)

		# Model

		value = product([model.data[i].data for i in model.data])

		if kwargs["model.system.architecture"] in ["array"]:
			value = array(value)
		elif kwargs["model.system.architecture"] in ["mps"]:
			value = array(value)

		print("--- model ---")
		model.info(verbose=verbose)
		print(model.data)
		print("------")


		# State

		value = state()
		
		if kwargs["model.system.architecture"] in ["array"]:
			value = array(value)
		elif kwargs["model.system.architecture"] in ["mps"]:
			value = value.to_dense().reshape(-1)

		print("--- state ---")
		state.info(verbose=verbose)
		print(value)
		print("------")


		# Value
		
		value = model(model.parameters(model.parameters()),model.state())

		if kwargs["model.system.architecture"] in ["array"]:
			value = array(value)
		elif kwargs["model.system.architecture"] in ["mps"]:
			value = value.to_dense().reshape(-1)

		print("--- value ---")
		print(value)
		print("------")
		

		data[i] = value

	assert all(allclose(data[i],data[j]) for i in data for j in data if i != j), "Error - Inconsistent models"

	print("Passed")

	return

def test_contract(*args,**kwargs):

	print('Not Implemented')

	return

	architecture = "array"

	settings = Dict({
		"cls":{
				"model":"src.quantum.Operator",
				"state":"src.quantum.State"
			},		
		"model":{
			"operator":"X.X",
			"site":[0,1],
			"string":"operator",
			"parameters":0.5,
			"N":2,"D":2,"ndim":2,
			"system":{"architecture":None}
		},	
		"state": {
			"operator":"zero",
			"site":None,
			"string":"psi",
			"parameters":True,
			"N":2,"D":2,"ndim":1,
			"system":{"architecture":None}
			},
	})

	verbose = True

	model = load(settings.cls.model)
	state = load(settings.cls.state)

	model = model(**settings.model)
	state = state(**settings.state)

	print("--- state ---")
	state.info(verbose=verbose)
	print(state())
	print("------")

	model.init(state=state)

	print()

	print("--- model ---")
	model.info(verbose=verbose)
	print(model.data)
	print("------")

	print("--- contract ---")
	print(model(model.parameters(model.parameters()),model.state()))
	print("------")

	print()


	print("--- einsum ---",model.ndim,state.ndim)
	if model.ndim == 3 and state.ndim == 2:
		print(einsum("uij,jk,ulk",model.data,model.state(),conjugate(model.data)))
	elif model.ndim == 3 and state.ndim == 1:
		print(einsum("uij,jk",model.data,model.state()))
	elif model.ndim == 2 and state.ndim == 2:
		print(einsum("ij,jk,lk",model.data,model.state(),conjugate(model.data)))
	elif model.ndim == 2 and state.ndim == 1:
		print(einsum("ij,jk",model.data,model.state()))
	print("------")

	print()


	model.init(state=state,parameters=dict())

	print(model.data)
	print(model(model.parameters(model.parameters()),model.state()))
	print()

	return

def test_amplitude(*args,**kwargs):

	kwargs = {"state.ndim":[1,2]}
	groups = None

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):

		settings = Dict({
			"cls":{
				"state":"src.quantum.Amplitude"
			},
			"state": {
				"data":"011",
				"operator":"string",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":"array"}
				}
			})

		setter(settings,kwargs,delimiter=delim,default=True)

		state = load(settings.cls.state)

		state = state(**settings.state)

		print(settings["state"]["operator"],type(state))
		print(state.data)
		print(state(state.parameters(),state.state()))
		print(state.norm())
		print()

		assert allclose(state(),state.data), "Incorrect data for %r"%(settings.cls.state)
		assert allclose(state(state.parameters(),state.state()),state.data), "Incorrect state() for %r"%(settings.cls.state)
		assert allclose(state.norm(),1), "Incorrect normalization for %r"%(settings.cls.state)

	return

def test_probability(*args,**kwargs):

	kwargs = {"state.ndim":[1]}
	groups = None

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):

		settings = Dict({
			"cls":{
				"state":"src.quantum.Probability"
			},
			"state": {
				"data":[1/2]*2,
				"operator":"probability",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":"array"}
				}
			})

		setter(settings,kwargs,delimiter=delim,default=True)

		state = load(settings.cls.state)

		state = state(**settings.state)

		print(settings["state"]["operator"],type(state))
		print(state.data)
		print(state(state.parameters(),state.state()))
		print(state.norm())
		print()

		assert allclose(state(),state.data), "Incorrect data for %r"%(settings.cls.state)
		assert allclose(state(state.parameters(),state.state()),state.data), "Incorrect state() for %r"%(settings.cls.state)
		assert allclose(state.norm(),1), "Incorrect normalization for %r"%(settings.cls.state)

	return

def test_state(*args,**kwargs):

	kwargs = {
		"cls.state":["src.quantum.State","src.quantum.Operator"],
		"state.data":[
			"random",
			[1/2]*2
			],
		"state.operator":[
			"product",
			"probability"
			],
		"state.ndim":[1],
		}
	groups = [["state.data","state.operator"]]

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):

		settings = Dict({
			"cls":{
				"state":"src.quantum.State"
			},
			"state": {
				"data":"random",
				"operator":"probability",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":2,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":"array"}
				}
			})

		from src.quantum import State

		setter(settings,kwargs,delimiter=delim,default=True)

		state = load(settings.cls.state)

		state = state(**settings.state)

		print(settings["cls"]["state"],settings["state"]["operator"],type(state))
		print(state.data)
		print(state(state.parameters(),state.state()))
		print(state.norm())
		print()

		assert allclose(state(),state.data), "Incorrect data for %r"%(settings.cls.state)
		assert allclose(state(state.parameters(),state.state()),state.data), "Incorrect state() for %r"%(settings.cls.state)
		assert allclose(state.norm(),1), "Incorrect normalization for %r"%(settings.cls.state)

	return


def test_measure(*args,**kwargs):

	kwargs = {
		"model.base":["pauli","tetrad","standard"],
		"state.D":[4,4,3],
		}
	groups = [["model.base","state.D",]]

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):

		settings = Dict({
			"cls":{
				"model":"src.quantum.Measure",
				"state":"src.quantum.State",
				"basis":"src.quantum.Basis",
			},
			"model":{
				"data":None,
				"base":"pauli",
				"string":"povm",
				"D":2,
				"dtype":"complex"
			},
			"state": {
				"data":"random"	,
				"operator":"probability",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":4,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":"array","base":"pauli"}
				}
			})

		setter(settings,kwargs,delimiter=delim,default=True)

		model = load(settings.cls.model)
		state = load(settings.cls.state)
		basis = load(settings.cls.basis)

		model = model(**settings.model)
		state = state(**settings.state)

		model.init(state=state)

		print(settings["model"]["base"])
		print(model,len(model),model.D,state.D)
		print(model.identity)
		print(model.data)
		print(model.inverse)
		print(model.dot(model.data,model.inverse))
		print(model())
		print()

		assert allclose(sum(i for i in model.basis),basis.I(D=model.D,dtype=model.dtype)), "Incorrect %r basis"%(model)
		assert allclose(model.dot(model.data,model.inverse),basis.I(D=len(model),dtype=model.dtype)), "Incorrect %r data"%(model)

	print('Passed')

	return


def test_init(*args,**kwargs):
	def init(data,parameters=None,state=None):

		locality = max(len(data[operator].operator) if not isinstance(data[operator].operator,str) else 1 for operator in data if data[operator].operator is not None)
		basis = Basis
		lattice = Lattice(locality)

		funcs = []
		for operator in data:

			obj = None

			if isinstance(data[operator].operator,str):
				objs = []
				for i in range(locality):
					objs.append(getattr(basis,data[operator].operator)(parameters=data[operator].parameters))
				obj = array([tensorprod(i) for i in permutations(*objs)])
			else:
				obj = basis.string(data=data[operator].operator,parameters=data[operator].parameters)

			if obj is None:
				continue

			if state is None:
				einsummation = None
				def contract(parameters,state,data=obj,einsummation=einsummation):
					return data
				contract = lambda parameters,state,data=obj,einsummation=einsummation: data
			elif obj.ndim == 3 and state.ndim == 2:
				subscripts = 'uij,jk,ulk->il'
				shapes = (obj.shape,state.shape,obj.shape)
				einsummation = einsum(subscripts,*shapes)
				def contract(parameters,state,data=obj,einsummation=einsummation):
					return einsummation(data,state,conjugate(data))				
			elif obj.ndim == 2 and state.ndim == 2:
				subscripts = 'ij,jk,lk->il'
				shapes = (obj.shape,state.shape,obj.shape)
				einsummation = einsum(subscripts,*shapes)
				def contract(parameters,state,data=obj,einsummation=einsummation):
					return einsummation(data,state,conjugate(data))	
			
			def func(parameters,state,contract=contract):
				return contract(parameters,state)

			funcs.append(func)


		def func(parameters=None,state=None):
			for func in funcs:
				state = func(parameters,state)
			return state 


		return func


	# Modules
	from src.quantum import Operators,State	
	from src.quantum import Basis,Measure
	from src.system import Lattice
	
	# Settings
	default = {}
	wrapper = Dict
	# settings = load(settings,default=default,wrapper=wrapper)
	settings = {
		"model":{
			"data":{
				"xx":{
					"operator":["X","X"],"site":"<ij>","string":"XX",
					"parameters":0.25,"variable":False
				},
				"noise":{
					"operator":"dephase","site":None,"string":"dephase",
					"parameters":1e-3,"variable":False
				}		
			},
			"N":2,
			"D":2,
			"d":1,
			"M":1,
			"P":1,
			"S":4,
			"base":"pauli",
			"space":"spin",
			"time":"linear",
			"lattice":"square"
			},
		"state": {
			"operator":"zero",
			"site":None,
			"string":"psi",
			"parameters":True,
			"N":2,
			"D":2,
			"d":1,
			"M":1,
			"P":1,
			"S":4,	
			"ndim":2,
			"seed":123
			},
		"system":{
			"dtype":"complex",
			"format":"array",
			"device":"cpu",
			"backend":None,
			"architecture":"array",
			"seed":123,
			"key":None,
			"instance":None,
			"cwd":"data",
			"path":None,
			"path":"data.hdf5",
			"conf":"logging.conf",
			"logger":None,
			"cleanup":False,
			"verbose":"info"
			}
		}
	settings = wrapper(settings)


	# System
	N = settings.state.N
	D = settings.state.D
	d = settings.state.d
	ndim = settings.state.ndim
	data = settings.model.data
	state = settings.state
	base = settings.model.base
	lattice = settings.model.lattice
	dtype = settings.system.dtype

	# Initialize
	basis = Basis
	lattice = Lattice(N,d,lattice=lattice)
	measure = Measure(base,D=D)
	structure = '<ij>'

	# Model
	parameters = None
	state = basis.zero(N=N,D=D,ndim=ndim,dtype=dtype)
	model = init(data,parameters,state)

	check = model(parameters,state)

	print(check.round(8))

	# Basis
	parameters = measure.parameters()
	state = measure.probability(parameters,state)
	operator = measure.operator(parameters,state,model=model)

	print(measure.basis)
	print(measure.data)
	print(measure.inverse)

	state = dot(operator,state)
	test = measure(parameters,state)

	print(test.round(8))


	assert allclose(test,check), "Incorrect model() - measure() conversion"


	# Test
	# Function
	parameters = None
	state = basis.zero(N=N,D=D,ndim=ndim,dtype=dtype)

	value = model(parameters,state)

	# Model
	model = Operators(**settings.model)
	state = State(**settings.state)
	model.init(state=state)

	parameters = model.parameters()
	state = model.state()

	test = model(parameters,state)

	assert allclose(value,test), "Incorrect func and model"

	print('Passed')

	return	

def test_basis(*args,**kwargs):

	print('Not Implemented')

	return

	kwargs = {
		"model.operator":["pauli","tetrad","standard"],
		"state.D":[4,4,3],
		}
	groups = [["model.operator","state.D",]]

	for i,kwargs in enumerate(permuter(kwargs,groups=groups)):

		settings = Dict({
			"cls":{
				"model":"src.quantum.Measure",
				"state":"src.quantum.State",
				"basis":"src.quantum.Basis",
			},
			"model":{
				"data":None,
				"operator":"pauli",
				"string":"povm",
				"D":2,
				"dtype":"complex"
			},
			"state": {
				"data":"random"	,
				"operator":"probability",
				"site":None,
				"string":"psi",
				"parameters":True,
				"N":3,"D":4,"ndim":1,
				"system":{"seed":12345,"dtype":"complex","architecture":"array","base":"pauli"}
				}
			})

		setter(settings,kwargs,delimiter=delim,default=True)

		model = load(settings.cls.model)
		state = load(settings.cls.state)
		basis = load(settings.cls.basis)

		model = model(**settings.model)
		state = state(**settings.state)

		model.init(state=state)


		print(settings["model"]["operator"])
		print(model,len(model))
		print(model.identity)
		print(model.data)
		print(model.inverse)
		print(model.dot(model.data,model.inverse))
		print(model())
		print()

		assert allclose(sum(i for i in model.basis),basis.I(D=model.D,dtype=model.dtype)), "Incorrect %r basis"%(model)
		assert allclose(model.dot(model.data,model.inverse),basis.I(D=len(model),dtype=model.dtype)), "Incorrect %r data"%(model)

	return




if __name__ == "__main__":

	arguments = "path"
	args = argparser(arguments)

	# main(*args,**args)


	# test_channel(*args,**args)
	# test_amplitude(*args,**args)
	# test_probability(*args,**args)
	# test_state(*args,**args)
	# test_measure(*args,**args)
	# test_composite(*args,**args)
	# test_basis(*args,**args)
	test_init(*args,**args)
