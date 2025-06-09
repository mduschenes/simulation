#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,array,allclose,delim,conjugate,prod
from src.io import load,glob
from src.call import rm
from src.system import Dict
from src.iterables import namespace
from src.logger import Logger

from src.train import train

# logger = Logger()


def test_call(path,*args,tol=None,**kwargs):

	args = ()
	kwargs = {}

	settings = {	
		"boolean":{
			"call":1,
			"optimize":0,
			"load":0,
			"dump":0
			},
		"seed":{
			"seed":492174,
			"size":24,
			"size":2,
			"reset":None,
			"group":None,
			"index":None
			},
		"permutations":{
			"permutations":{	
				"module.M":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,34,36,38,40,50],
				"module.M":[1,2],
				"model.N":[4,6,8],
				"model.N":[8],
				"model.N":[2],
				"model.M":[1],
				"model.D":[2],
				"model.d":[1],
				"model.local":[True],
				"model.tensor":[True],
				"model.configuration":[{
					"key":None,
					"options":{"layout":"brickwork","attribute":[{"where":"ij","unitary":True},{"where":"i","unitary":True},{"where":"j","unitary":True},{"where":"i","unitary":False},{"where":"j","unitary":False}]}
					}],
				
				"module.measure.string":["povm"],
				"module.measure.operator":["tetrad"],
				"module.measure.architecture":["tensor_quimb"],
				"module.measure.architecture":["tensor"],
				"module.measure.D":[2],		
				"module.measure.options":[{}],
				"module.options":[{"contract":"swap+split","max_bond":None,"cutoff":0}],
				"module.options":[{"scheme":"svd","eps":1e-16,"iters":5e6,"parameters":None,"method":"mu","initialize":"nndsvda","metric":"div","seed":None}],
				"module.options.S":[4,16,64,256],
				"module.options.S":[2,4,8,16,32,64,128,256],
				"module.options.S":[None],
				"module.configuration":[{
					"key":None,
					"options":{"layout":"brickwork","attribute":[{"where":"ij","unitary":True},{"where":"i","unitary":True},{"where":"j","unitary":True},{"where":"i","unitary":False},{"where":"j","unitary":False}]}
					}],

				"model.data.unitary.operator":["gate"],
				"model.data.unitary.parameters":[None],
				"model.data.unitary.where":["||ij||"],

				"model.data.local.operator":["gate"],
				"model.data.local.parameters":[None],
				"model.data.local.where":["||i.j||"],

				"model.data.noise.operator":[["depolarize"]],
				"model.data.noise.parameters":[1e-4,1e-2,5e-2,1e-1],
				"model.data.noise.parameters":[1e-4,1e-3,2.5e-3,5e-3,7.5e-3,1e-2,2.5e-2,5e-2,1e-1],
				"model.data.noise.parameters":[5e-3],
				"model.data.noise.where":["||i.j||"],

				"state.string":["psi"],
				"state.operator":["zero"],
				"state.local":[False],
				"state.tensor":[True],
				"state.D":[2],
				"state.ndim":[2],
				"state.parameters":[None],

				"callback.options":[{"contract":"swap+split","max_bond":None,"cutoff":0}],
				"callback.options":[{"scheme":"svd","S":None,"eps":None,"iters":None,"parameters":None,"method":None,"initialize":None,"metric":None,"seed":None}],

				},
			"groups":None,
			"filters":None,
			"func":None,
			"index":None
			},
		"cls":{
			"module":"src.quantum.Module",
			"model":"src.quantum.Operators",
			"state":"src.quantum.State",
			"callback":"src.quantum.Callback"	
			},
		"module":{
			"N":2,
			"M":2,
			"D":2,
			"d":1,
			"seed":None,
			"string":"module",
			"measure":{"string":"povm","operator":"tetrad","D":2,"seed":None,"architecture":"tensor","options":{}},
			"configuration":{},
			"options":{}		
		},
		"model":{
			"data":{
				"unitary":{
					"operator":"haar","where":"||ij||","string":"unitary",
					"parameters":None,"variable":False,"seed":None
				},				
				"local":{
					"operator":"haar","where":"||i.j||","string":"local",
					"parameters":None,"variable":False,"seed":None
				},						
				"noise":{
					"operator":["depolarize"],"where":"||i.j||","string":"noise",
					"parameters":1e-8,"variable":False
				}
			},
			"N":2,
			"M":1,
			"D":2,
			"d":1,
			"local":True,
			"tensor":True,
			"space":"spin",
			"time":"linear",
			"lattice":"square",
			"seed":None
			},
		"state": {
			"operator":"haar",
			"where":None,
			"string":"psi",
			"parameters":None,
			"N":None,
			"D":2,
			"ndim":2,
			"local":None,
			"tensor":True,	
			"seed":None	
			},
		"system":{
			"dtype":"complex",
			"format":"array",
			"device":"cpu",
			"backend":None,
			"architecture":None,
			"base":None,
			"seed":None,
			"key":None,
			"instance":None,
			"cwd":"tmp",
			"path":"data.hdf5",
			"lock":True,
			"backup":None,	
			"conf":None,
			"logger":None,
			"cleanup":False,
			"verbose":None
			},
		"callback":{
			"attributes":{
				"N":"N","M":"M","d":"d","D":"D",
				"key":"key","instance":"instance","timestamp":"timestamp",
				"seed":"seed","seeding":"seeding",
				"noise.parameters":"noise.parameters",
				"operator":"measure.operator",
				"S":"options.S",
				"scheme":"options.scheme",		
				"layout":"configuration.options.layout",		
				"periodic":"measure.options.periodic",
				
				"infidelity.quantum":None,
				"infidelity.classical":"measure.infidelity_classical",
				"infidelity.pure":None,
				"norm.quantum":None,
				"norm.classical":"measure.norm_classical",
				"norm.pure":None,

				"entanglement.quantum":None,
				"entanglement.classical":"measure.entanglement_classical",
				"entanglement.renyi":None,
				"entangling.quantum":None,
				"entangling.classical":None,
				"entangling.renyi":None,
				
				"mutual.quantum":None,
				"mutual.measure":None,
				"mutual.classical":"measure.mutual_classical",
				"mutual.renyi":None,
				"discord.quantum":None,
				"discord.classical":None,
				"discord.renyi":None,
				
				"spectrum.quantum":None,
				"spectrum.classical":"measure.spectrum_classical",
				"rank.quantum":None,
				"rank.classical":"measure.rank_classical"
			},
			"keywords": {
				"entanglement.quantum":{"where":0.5},"entanglement.classical":{"where":1.0},"entanglement.renyi":{"where":0.5},
				"entangling.quantum":{"where":0.5},"entangling.classical":{"where":0.5},"entangling.renyi":{"where":0.5},
				"mutual.quantum":{"where":0.5},"mutual.measure":{"where":0.5},"mutual.classical":{"where":0.5},"mutual.renyi":{"where":0.5},
				"discord.quantum":{"where":0.5},"discord.classical":{"where":0.5},"discord.renyi":{"where":0.5},
				"spectrum.quantum":{"where":0.5},"spectrum.classical":{"where":0.5},"rank.quantum":{"where":0.5},"rank.classical":{"where":0.5}
			},
			"options":{}	
		}
	}


	permutations = settings.get('permutations',{}).get('permutations',{})
	seeds = settings.get('seed',{}).get('size',None)

	size = prod(len(permutations[attr]) for attr in permutations)
	length = seeds if seeds is not None else 1

	indices = range(size*length)

	data = {}

	for index in indices:
		model,parameters,state,optimizer = train(settings,index=index,*args,**kwargs)

		value = model(parameters=model.parameters(),state=model.state())
		value = model.measure.amplitude(parameters=model.parameters(),state=value).matrix()

		print(index,model.seed)
		print(value)

		data[index] = value



	print("Passed")

	return

@pytest.mark.filterwarnings(r"ignore:Rounding errors prevent the line search from converging")
@pytest.mark.filterwarnings(r"ignore:The line search algorithm did not converge")
def test_train(path,*args,tol=None,**kwargs):

	path = 'config/settings.json'# if path is None else path
	tol = 5e-1 # if tol is None else tol

	settings = path
	args = ()
	kwargs = {}

	model,parameters,state,optimizer = train(settings,*args,**kwargs)

	paths = [optimizer.cwd]
	execute = True
	verbose = True
	for path in paths:
		rm(path,execute=execute,verbose=verbose)


	print(optimizer.data)
	assert all(not isinstance(optimizer.data[attr],list) or not optimizer.data[attr] for attr in optimizer.data) or (not optimizer.data.get('objective')) or (optimizer.data['objective'][-1] < tol), "Incorrect Optimization of %r"%(model)

	print("Passed")

	return


if __name__ == '__main__':

	arguments = 'path'
	args = argparser(arguments)

	test_call(*args,**args)
	# test_train(*args,**args)
