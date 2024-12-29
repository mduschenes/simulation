#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,delim
from src.io import load,dump,join
from src.iterables import Dict,getter,setter
from src.process import process

def setup(settings,data,*args,**kwargs):
	'''
	Workflow setup
	Args:
		data (str,dict): Workflow data
		settings (str,dict): Workflow settings
		args (iterable): Workflow positional arguments 
		kwargs (dict): Workflow keyword arguments 
	Returns:
		settings (dict): Workflow settings		
		data (dict): Workflow data
	'''	

	default = {}
	if isinstance(settings,str):
		settings = load(settings,default=default)
	else:
		settings = default

	default = {}
	if isinstance(data,str):
		data = load(data,default=default)
	else:
		data = default

	iterables = {
		'system.path': (lambda attr,settings: join(
			getter(settings,delim.join(['system','cwd']),delimiter=delim),
			getter(settings,delim.join([*attr.split(delim)[:-1],'path']),delimiter=delim))
			),
		'process.data': (lambda attr,settings: join(
			getter(settings,delim.join(['system','cwd']),delimiter=delim),
			getter(settings,delim.join([*attr.split(delim)[:-1],'data']),delimiter=delim))
			),
	}

	for attr in iterables:
		iterables[attr] = iterables[attr](attr,settings)


	setter(settings,iterables,delimiter=delim)

	settings = Dict(settings)

	return settings,data

def model(settings,data,*args,**kwargs):
	'''
	Workflow model
	Args:
		settings (dict): Workflow settings
		data (dict): Workflow data
		args (iterable): Workflow positional arguments 
		kwargs (dict): Workflow keyword arguments 
	'''

	def setup(settings,data,*args,**kwargs):
		'''
		Workflow setup
		Args:
			settings (dict): Workflow settings
			data (dict): Workflow data
			args (iterable): Workflow positional arguments 
			kwargs (dict): Workflow keyword arguments 
		'''	

		iterables = {
			'system.func': (lambda attr,settings: load(getter(settings,attr,delimiter=delim))),
		}

		for attr in iterables:
			iterables[attr] = iterables[attr](attr,settings)

		setter(settings,iterables)

		return

	setup(settings,data,*args,**kwargs)

	attr = 'system.func'
	getter(settings,attr,delimiter=delim)(data,*args,**kwargs)

	return

def workflow(settings,data,*args,**kwargs):
	'''
	Workflow
	Args:
		settings (dict): Workflow settings
		data (dict): Workflow data
		args (iterable): Workflow positional arguments 
		kwargs (dict): Workflow keyword arguments 
	'''

	settings,data = setup(settings,data,*args,**kwargs)

	if settings.boolean.load:
		data = load(settings.system.path,default=data)

	if settings.boolean.model:
		model(settings,data,**settings.model)
		
	if settings.boolean.dump:
		dump(data,settings.system.path)

	if settings.boolean.process:
		process(**settings.process)

	return


def main(*args,**kwargs):
	'''
	Main
	Args:
		args (iterable): Positional arguments 
		kwargs (dict): Keyword arguments 
	'''

	workflow(*args,**kwargs)

	return

if __name__ == '__main__':

	arguments = ['settings','data']

	args = argparser(arguments)

	main(*args,**args)



	

