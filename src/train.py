#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,allclose,delim,spawn
from src.io import load,glob
from src.iterables import Dict,namespace,setter
from src.optimize import Optimizer,Objective,Metric,Callback
from src.logger import Logger
logger = Logger()

def setup(settings,*args,**kwargs):
	'''
	Setup settings
	Args:
		settings (dict,str): settings
		args (iterable): settings positional arguments
		kwargs (dict): settings keyword arguments
	Returns:
		settings (dict): settings
	'''

	default = {}
	wrapper = Dict
	defaults = Dict(
		boolean=dict(call=None,optimize=None,load=None,dump=None),
		cls=dict(module=None,model=None,state=None,label=None,callback=None),
		module=dict(),model=dict(),state=dict(),label=dict(),callback=dict(),
		optimize=dict(),seed=dict(),system=dict(),
		)

	if settings is None:
		settings = default
	elif isinstance(settings,str):
		settings = load(settings,default=default,wrapper=wrapper)

	setter(settings,kwargs,delimiter=delim,default=True)
	setter(settings,defaults,delimiter=delim,default=False)

	return settings


def call(settings,*args,**kwargs):
	'''
	Call model
	Args:
		settings (dict,str,iterable[str,dict]): settings
		args (iterable): settings positional arguments
		kwargs (dict): settings keyword arguments		
	Returns:
		model (object): Model instance
	'''

	settings = setup(settings,*args,**kwargs)

	module = load(settings.cls.module)
	model = load(settings.cls.model)
	state = load(settings.cls.state)
	callback = load(settings.cls.callback)
	system = settings.system

	if module is not None and model is not None and state is not None:
	
		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(system=system)})
		callback = callback(**{**settings.callback,**dict(system=system)})

		module = module(**{**settings.module,**dict(model=model,state=state,callback=callback,system=system)})

		module.init()

		model = module

	elif model is not None and state is not None:

		model = model(**{**settings.model,**dict(system=system)})
		state = state(**{**namespace(state,model),**settings.state,**dict(system=system)})

		model.init(state=state)

	elif model is not None:

		model = model(**{**settings.model,**dict(system=system)})

	else:

		model = None

	return model


def optimize(settings,*args,**kwargs):
	'''
	Optimize model
	Args:
		settings (dict,str,iterable[str,dict]): settings
		args (iterable): settings positional arguments
		kwargs (dict): settings keyword arguments		
	Returns:
		model (object): Model instance
		parameters (object): Model parameters
		state (object): Model state
		optimizer (object): Model optimizer
	'''	

	settings = setup(settings,*args,**kwargs)

	model = call(settings,*args,**kwargs)
	system = settings.system

	label = load(settings.cls.label)
	callback = load(settings.cls.callback)

	state = model.state
	label = label(**{**namespace(label,model),**settings.label,**dict(system=system)})
	callback = callback(**{**namespace(callback,model),**settings.callback,**dict(model=model,system=system)})

	label.init(state=state)

	func = model.parameters.constraints if hasattr(model.parameters,'constraints') else None
	seed = spawn(**settings.seed)
	hyperparameters = settings.optimize
	arguments = ()
	keywords = {}
	
	metric = Metric(state=state,label=label,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system)
	func = Objective(model,func=func,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system)
	callback = Callback(model,func=func,callback=callback,arguments=arguments,keywords=keywords,metric=metric,hyperparameters=hyperparameters,system=system)

	optimizer = Optimizer(func=func,arguments=arguments,keywords=keywords,callback=callback,hyperparameters=hyperparameters,system=system)

	parameters = model.parameters()
	state = model.state()

	parameters = optimizer(parameters,state=state)

	return model,parameters,state,optimizer


def train(settings,*args,**kwargs):
	'''
	Train model
	Args:
		settings (dict,str,iterable[str,dict]): settings
		args (iterable): settings positional arguments
		kwargs (dict): settings keyword arguments		
	Returns:
		model (object): Model instance
		parameters (object): Model parameters
		state (object): Model state
		optimizer (object): Model optimizer
	'''

	settings = setup(settings,*args,**kwargs)

	model = None
	parameters = None
	state = None
	optimizer = None

	if settings.boolean.load:

		model.load()

	if settings.boolean.call:

		model = call(settings)
	
	if settings.boolean.optimize:

		model,parameters,state,optimizer = optimize(settings)

	if settings.boolean.dump:	
	
		model.dump()

	return model,parameters,state,optimizer

def run(settings,*args,**kwargs):
	'''
	Run models
	Args:
		settings (dict,str,iterable[str,dict]): settings
		args (iterable): settings positional arguments
		kwargs (dict): settings keyword arguments		
	Returns:
		model (object): Model instance
	'''
	
	if settings is None:
		models = []
	elif isinstance(settings,str):
		models = list(glob(settings))
	elif isinstance(settings,dict):
		models = [settings]

	models = {name: Dict(settings=settings,model=None,parameters=None,state=None,optimizer=None)
		for name,settings in enumerate(models)}

	for name in models:
		
		settings = models[name].settings

		model,parameters,state,optimizer = train(settings,*args,**kwargs)
	
		models[name].model = model
		models[name].parameters = parameters
		models[name].state = state
		models[name].optimizer = optimizer

	model = models[name] if len(models) == 1 else models

	return model


def main(*args,**kwargs):

	run(*args,**kwargs)

	return


if __name__ == '__main__':

	arguments = 'settings'
	args = argparser(arguments)

	main(*args,**args)