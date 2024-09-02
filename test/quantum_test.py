#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ["",".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,array,zeros,ones,empty,allclose,product,spawn,einsum,conjugate,dot,tensorprod,trace,representation
from src.utils import arrays,iterables,scalars,integers,floats,pi,delim
from src.iterables import permutations
from src.io import load,dump,glob
from src.call import rm,echo
from src.system import Dict
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

def test_channel(*args,**kwargs):

	data = {}

	kwargs = {
		"cls.model":["src.quantum.Channel","src.quantum.Operators"],
		**{attr:[3] for attr in ["model.N","state.N"]},
		**{attr:["array"] for attr in ["model.system.architecture","state.system.architecture"]},
		"model.M":[10],
		"model.ndim":[2],"state.ndim":[1],
		}
	groups = [["model.N","state.N"],["model.system.architecture","state.system.architecture"],]
	filters = lambda iterables: (iterable for iterable in iterables 
		if not (iterable['cls.model'] in ['src.quantum.Channel'] and 
				iterable['model.system.architecture'] in ['mps'] and
				iterable['state.system.architecture'] in ['mps'])
		)
	func = None

	for i,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters,func=func)):
	
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
				"N":3,"D":2,"ndim":1,
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
				"D":2,"ndim":2,
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
						"D":2,"ndim":2,
						"variable":True
					},
					"noise":{
						"operator":"depolarize","site":None,"string":"noise",
						"parameters":1e-12,
						"ndim":3,
						"variable":False
					}
				},
				"D":2,"ndim":2,
				"system":{"seed":12345,"dtype":"complex","architecture":None}
			},	
			"state": {
				"data":None	,
				"operator":"product",
				"site":None,
				"string":"psi",
				"parameters":True,
				"D":2,"ndim":1,
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

def test_state(*args,**kwargs):

	kwargs = {
		"cls.state":["src.quantum.State","src.quantum.Operator"],
		"state.data":[
			"random",
			[1/2]*2
			],
		"state.operator":[
			"product",
			"state"
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
				"operator":"state",
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
		"model.base":["pauli","tetrad"],
		"state.D":[4,4],
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
				"operator":"state",
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
		print()


		assert allclose(sum(i for i in model.basis),basis.I(D=model.D,dtype=model.dtype)), "Incorrect %r basis"%(model)
		assert allclose(einsum('uw,vw->uv',model.data,model.inverse),basis.identity(D=len(model),dtype=model.dtype)), "Incorrect %r data"%(model)

	print('Passed')

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

	for i,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters,func=func)):
		
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
				"operator":'X.X.X.X',
				"site":None,
				"string":"U",
				"parameters":0.5,
				"ndim":2,
				"seed":123
				},
			"measure":{
				"base":"pauli",
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

		print('Attributes',{attr: getattr(settings.model,attr,None) for attr in ['N','data']})

		model.info(verbose=True)
		state.info(verbose=True)

		attributes = {'model':model,'state':state,'label':label}
		for attribute in attributes:
			print(attribute,{attr: getattr(attributes[attribute],attr) for attr in ['N','locality','site','hermitian','unitary','architecture']},'>>>>',namespace(attributes[attribute].__class__,model))
		print(i,state.local,{i:model.data[i].local for i in model.data})
		print()

		model.init(state=state)
		label.init(state=state)

		attributes = {'model':model,'label':label}
		for attribute in attributes:
			print(attribute,{attr: getattr(attributes[attribute],attr) for attr in ['N','locality','site','hermitian','unitary','architecture']},'>>>>',namespace(attributes[attribute].__class__,model))
		print()

		print('Call')
		print(state())
		print(model(state=state()))
		print(label(state=state()))
		print()


		print('Measure')
		attributes = {'measure':measure}
		for attribute in attributes:
			print(attribute,{attr: getattr(attributes[attribute],attr) for attr in ['D','base','ind','inds','tags','data','inverse','basis','architecture']},'>>>>',attributes[attribute].__class__,namespace(attributes[attribute].__class__,model))
		print()

		print()
		print()
		print()

		parameters = model.parameters()
		state = model.state()
		value = model(parameters,state)

		data[i] = value


	assert all(allclose(data[i],data[j]) for i in data for j in data if i != j), "Error - Inconsistent models"

	print("Passed")

	return



def test_module(*args,**kwargs):

	kwargs = {
		"module.N":[2],"module.M":[1],
		'model.N':[None],'model.D':[2],'model.ndim':[2],
		'state.N':[None],'state.D':[2],'state.ndim':[2],
		"model.local":[True],"state.local":[True],"model.options.shape":[[2,2,2]],
		"model.data.noise.operator":[["dephase","dephase"],"dephase"],
		"model.data.noise.site":["<ij>",None],
		"model.layout":[{"site":None},{"site":None}],
		"module.measure.base":["pauli","tetrad"],
		"measure.architecture":["array","tensor"]
		}

	kwargs = {
		"module.N":[4],"module.M":[3],
		'model.N':[None],'model.D':[2],'model.ndim':[2],
		'state.N':[None],'state.D':[2],'state.ndim':[2],
		"model.data.xx.parameters":[0.125],
		"model.data.noise.parameters":[1e-12],
		"model.data.xx.site":[None],"model.data.noise.site":[None],
		"model.local":[True],"state.local":[False],
		"model.layout":[{"site":None}],
		"model.options.shape":[[2,2,2]],
		"module.measure.string":["pauli","tetrad"],
		"module.measure.base":["pauli","tetrad"],
		"module.measure.architecture":["tensor"],
		"module.measure.options":[{"cyclic":False}],
		"module.lattice":[{"lattice":"square","structure":">ij<"}],
		"module.options":[{"contract":"swap+split","max_bond":4,"cutoff":1e-4}]
		}	


	groups = [["module.measure.string","module.measure.base"]]

	filters = None

	def func(dictionaries):
		for dictionary in dictionaries:
			setter(dictionary,{					
				# 'model.N':getter(dictionary,'module.N',delimiter=delim),
				# 'state.N':getter(dictionary,'module.N',delimiter=delim),
				'model.options.shape':[
					getter(dictionary,'state.D',delimiter=delim),
					getter(dictionary,'module.N',delimiter=delim),
					getter(dictionary,'state.ndim',delimiter=delim)],
				'model.data.xx.site':getter(dictionary,'module.lattice',delimiter=delim)['structure'],
				'model.data.noise.site':getter(dictionary,'module.lattice',delimiter=delim)['structure'],
					},
				delimiter=delim,default=None)
		return

	data = {}
	for i,kwargs in enumerate(permuter(kwargs,groups=groups,filters=filters,func=func)):

		settings = Dict({
		"cls":{
			"module":"src.quantum.Module",
			"measure":"src.quantum.Measure",
			"model":"src.quantum.Operators",
			"state":"src.quantum.State",
			"callback":"src.quantum.Callback"
			},
		"module":{
			"N":2,
			"M":1,
			"d":1,
			"string":"module",
			"lattice":"square",
			"measure":{"base":"tetrad","architecture":"tensor","options":{"cyclic":False}},
			"options":{"contract":"swap+split","max_bond":4,"cutoff":1e-2}		
		},
		"measure":{
			"base":"pauli",
			"architecture":"tensor",
			"options":{"cyclic":False}	
		},		
		"model":{
			"data":{
				# "z":{
				# 	"operator":["Z"],"site":"i","string":"Z",
				# 	"parameters":0.5,"variable":False
				# },
				"xx":{
					"operator":["X","X"],"site":"<ij>","string":"XX",
					"parameters":0.5,"variable":False
				},				
				"noise":{
					"operator":["dephase","dephase"],"site":"<ij>","string":"dephase",
					"parameters":1e-3,"variable":False
				},

			},
			"D":2,
			"local":True,
			"space":"spin",
			"time":"linear",
			"lattice":"square",
			"architecture":"array",
			"layout":None,
			},
		"state": {
			"operator":["zero"],
			"site":None,
			"string":"psi",
			"parameters":None,
			"D":2,
			"ndim":2,
			"local":True
			},
		"callback":{
			"attributes":{
				"N":"N","M":"N","d":"d","D":"state.D",
				"noise.parameters":"noise.parameters",
				"objective":"objective",
				"base":"measure.base"
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
			"seed":123,
			"key":None,
			"instance":None,
			"cwd":"data",
			"path":"data.hdf5",
			"conf":"logging.conf",
			"logger":"log.log",
			"cleanup":False,
			"verbose":False
			}
		})

		# Settings
		setter(settings,kwargs,delimiter=delim,default='replace')

		# Class
		module = load(settings.cls.module)
		measure = load(settings.cls.measure)
		model = load(settings.cls.model)
		state = load(settings.cls.state)
		callback = load(settings.cls.callback)
		system = settings.system

		# Model
		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(system=system)})
		callback = callback(**{**settings.callback,**dict(system=system)})


		# Test

		obj = state
		
		data[i] = {}


		# Measure
		measure = measure(**{**settings.measure,**dict(system=system)})


		# Probability
		parameters = measure.parameters()
		state = [obj()]*settings.module.N

		probability = measure.probability(parameters=parameters,state=state)

		key = 'probability'
		if settings.measure.architecture in ['array']:
			value = array(probability)
		elif settings.measure.architecture in ['tensor']:
			value = representation(probability,to='array',contract=False)
		
		data[i][key] = value


		# Amplitude
		parameters = measure.parameters()
		state = probability

		amplitude = measure.amplitude(parameters=parameters,state=state)

		key = 'amplitude'
		if settings.measure.architecture in ['array']:
			value = tensorprod(amplitude)
		elif settings.measure.architecture in ['tensor']:
			value = array(amplitude)
		
		data[i][key] = value


		# Operator
		parameters = model.parameters()
		state = [obj()]*model.locality
		where = list(range(model.locality))

		model.init(state=tensorprod(state))

		state = measure.probability(parameters=parameters,state=state)

		operator = measure.operator(parameters=parameters,state=state,model=model,where=where)

		key = 'operator'
		if settings.measure.architecture in ['array']:
			value = array(operator(parameters,state))
		elif settings.measure.architecture in ['tensor']:
			value = representation(operator(parameters=parameters,state=state),to='tensor',contract=True)

		data[i][key] = value

		# Module
		model = model		
		state = obj
		callback = callback

		module = module(**{**settings.module,**dict(callback=callback,system=system)})

		module.init(model=model,state=state)

		module.info(verbose=True)

		parameters = module.parameters()
		state = module.state()

		state = module(parameters,state)

		key = 'module'
		value = module.measure.transform(parameters=parameters,state=state,transformation=False)
		data[i][key] = value

		# Init
		model = load(settings.cls.model)
		state = load(settings.cls.state)
		system = settings.system

		model = model(**{**settings.model,**dict(N=module.N,M=module.M,options=dict(shape=(settings.state.D,settings.module.N,settings.state.ndim))),**dict(system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(N=module.N),**dict(system=system)})

		model.init(state=state)

		parameters = model.parameters()
		state = model.state()

		key = 'init'
		value = model(parameters,state)

		data[i][key] = value

		print({i:{attr: getattr(model.data[i],attr,None) for attr in ['string','operator','site']} for i in model.data})

		if settings.module['options']['contract'] is False or settings.module['options']['cutoff'] <= 1e-16:
			assert allclose(data[i]['module'],data[i]['init']), "Incorrect module() and model() ---\n%s\n%s\n%s"%(data[i]['module'].round(8),data[i]['init'].round(8),(data[i]['module'] - data[i]['init']).round(8))

		parameters = module.parameters()
		state = module.state()
		options = {"contract":"swap+split","max_bond":8,"cutoff":1e-8}

		state,other = module(parameters,state,**options),module(parameters,state)


		print(module.measure.infidelity_classical(parameters,state,other))
		exit()

		key = 'infidelity'
		value = abs(
			(module.measure.infidelity(parameters,state,other) -
			 module.measure.infidelity(parameters,state,state)) / 
			(module.measure.infidelity(parameters,state,state))
			)

		data[i][key] = value


		module.dump()

		module.load()

		continue


		# Model
		parameters = model.parameters()
		state = tensorprod([obj()]*module.N)

		model.init(state=tensorprod(state))

		state = measure.probability(parameters=parameters,state=state)

		operator = measure.operator(parameters=parameters,state=state,model=model)

		operator = operator(parameters,state)

		continue


		# Model
		module = load(settings.cls.module)
		measure = load(settings.cls.measure)
		model = load(settings.cls.model)
		state = load(settings.cls.state)
		system = settings.system

		N = settings.module.N
		model = model(**{**settings.model,**dict(N=N,local=True,system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(N=N,local=False,system=system)})

		model.init(state=state)

		parameters = model.parameters()
		state = model.state()

		key = 'model'
		value = model(parameters,state)

		data[i][key] = value

		continue	


		# Module
		module = module(**{**namespace(module,model),**namespace(module,state),**settings.module,**dict(model=model,state=state,system=system)})

	assert all(equalizer(data[i],data[j]) for i in data for j in data if i != j), "Error - Inconsistent models"

	print('Passed')

	return


if __name__ == "__main__":

	arguments = "path"
	args = argparser(arguments)

	# main(*args,**args)


	# test_channel(*args,**args)
	# test_state(*args,**args)
	# test_measure(*args,**args)
	# test_composite(*args,**args)
	# test_namespace(*args,**args)
	test_module(*args,**args)
