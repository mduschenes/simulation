#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

import pandas as pd

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,rand,allclose,scalars
from src.io import load,dump,join,split,edit,dirname,exists,glob,rm

# Logging
# from src.utils import logconfig
# conf = 'config/logging.conf'
# logger = logconfig(__name__,conf=conf)


def test_path(path='.tmp.tmp/data.hdf5'):
	new = edit(
			path=path,
			directory=None,
			file=(lambda directory,file,ext,delimiter: delimiter.join([*file.split(delimiter)[:]])),
			ext=None,
			delimiter='.'
			)

	assert new == path, "Incorrect path edit"

	return

def test_load(path=None):
	
	#TODO: Add data/ directory for test data to load 
	return

	kwargs = [
		{
		'path':'data/test/**/data.hdf5',
		'default':{},
		'wrapper':'df',
		'type':pd.DataFrame,
		},
		{
		'path':{'path':'data/test/-1/data.hdf5'},
		'default':None,
		'wrapper':'df',
		'type':None,		
		},
		{
		'path':{'path':'data/test/-1/data.hdf5'},
		'default':{},
		'wrapper':'df',
		'type':dict,		
		},		
		{
		'path':['data/test/**/data.hdf5'],
		'default':{},
		'wrapper':'df',
		'type':pd.DataFrame,		
		},
		{
		'path':['config/plot.json','config/process.json'],
		'default':{},
		'wrapper':None,
		'type':list,
		'subtype':dict,				
		},						
	]

	for kwarg in kwargs:
		typing = kwarg.pop('type')
		subtyping = kwarg.pop('subtype',None)
		data = load(**kwarg)
		print(kwarg,type(data))
		assert (typing is None and data is None) or isinstance(data,typing), "Incorrect loading %r"%(kwarg)
		if isinstance(data,(list,dict)):
			assert all(isinstance(datum,subtyping) for datum in data), "Incorrect multiple loading %r" %(kwarg)

	return





def test_dump(path=None):

	kwargs = [
		{
		'data':load('data/data/0/data.hdf5'),
		'path':'config/tmp/data.hdf5',
		'default':{},
		'wrapper':None,
		},
		{
		'data':load('data/data/0/data.hdf5'),
		'path':{'name':'config/tmp/data.hdf5','test':'config/tmp/tmp/test.hdf5'},
		'default':{},
		'wrapper':None,
		},		
		{
		'data':load('config/plot.json'),
		'path':['config/tmp/data.json','config/tmp/test.json'],
		'default':{},
		'wrapper':None,
		},				
	]



	for kwarg in kwargs:

		data = kwarg.pop('data',None)
		path = kwarg.get('path',None)
		dump(data,**kwarg)


		msg = "Incorrect dumping %r"%(kwarg)

		if isinstance(path,str):
			assert exists(path),msg
		elif isinstance(path,list):
			assert all(exists(subpath) for subpath in path),msg
		elif isinstance(path,dict):
			assert all(exists(path[subpath]) for subpath in path),msg
	
	path = 'config/tmp'
	rm(path,execute=True)

	return



def test_hdf5(path='.tmp.tmp/data.hdf5'):

	g = 3
	n = 2
	shape = (7,3)
	groups = ['%d'%(i) for i in range(g)]
	instances = ['%d'%(i) for i in range(n)]
	datasets = ['data','values','parameters']
	attributes = ['n','m','k']
	attrs = [*datasets,*attributes]
	data = {
		group: {
			instance:{
				**{attr: rand(shape) for attr in datasets},
				**{attr: rand() for attr in attributes}
				}
			for instance in instances
			}
		for group in groups
		}

	# Dump data
	wr = 'w'
	ext = 'hdf5'
	kwargs = {}

	dump(data,path,wr=wr,**kwargs)

	# Load data
	wr = 'r'
	ext = 'hdf5'
	kwargs = {}

	new = load(path,wr=wr,**kwargs)

	# Check dumped and loaded data are equal
	for group in groups:
		for instance in instances:
			for attr in attrs:
				msg = "group: %s, instance: %s, attr: %s Unequal"%(group,instance,attr)
				if isinstance(data[group][instance][attr],scalars):
					assertion = data[group][instance][attr] == new[group][instance][attr]
				else:
					assertion = allclose(data[group][instance][attr],new[group][instance][attr])
				assert assertion,msg


	path = dirname(path)

	rm(path,execute=True)

	return

def test_importlib(path=None,**kwargs):

	import os,sys,importlib

	objs = {"src.io":"load","src.quantum":"Object"}
	
	for attr in objs:
		
		obj = attr
		module = objs[attr]

		try:
			path = os.path.basename(obj).strip('.')
			data = getattr(importlib.import_module(path),module)
		except (SyntaxError,TypeError) as exception:
			logger.log(info,'Exception:\n%r\n%r'%(exception,traceback.format_exc()))
			exception = SyntaxError
			raise exception
		except Exception as exception:
			path = obj
			spec = importlib.util.spec_from_file_location(module,path)
			data = importlib.util.module_from_spec(spec)
			sys.modules[module] = data
			spec.loader.exec_module(data)
			data = getattr(data,module)

		print(data)

		assert data is not None

	print('Passed')

	return


def test_glob(path=None,**kwargs):

	directory = 'config'

	paths = {
		os.path.join(directory,'*.json'):
			[os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),directory,path))
			for path in  ['settings.json','test.json','process.json','plot.json']],
		os.path.join(directory,'{job.slurm,logging.conf}'):
			[os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),directory,path))
			for path in  ['job.slurm','logging.conf']],		
		}

	for path in paths:
		print(path)
		for i in glob(path):
			print(i)
			assert i in paths[path], "Incorrect glob(%r)"%(path)
		print()

	print('Passed')

	return


def test_lock(*args,**kwargs):

	key = str(args[0]) if args and args[0].isdigit() else None
	value = [1,2,3]

	path = './job/data.hdf5'
	kwargs = dict(
		default = {},
		lock = True
		)

	data = load(path,**kwargs)

	data.update({key:value})

	print(data)

	dump(data,path,**kwargs)

	if key == None:
		rm(path)

	# rm job/data.hdf5* -rf && parallel ./io_test.py ::: $(seq 1 10)
	return



if __name__ == '__main__':
	# test_load()
	# test_dump()
	# test_importlib()
	# test_glob()
	test_lock(*sys.argv[1:])

