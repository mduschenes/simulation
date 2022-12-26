#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime
from natsort import natsorted
from copy import deepcopy

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.call import submit,command,call,cp,rm,echo,sed,sleep,touch
from src.parallel import Parallelize,Pooler
from src.io import load,dump,glob


def func(path,default={},options={},**kwargs):
	value = load(path,default=default,**options)				
	value.update(kwargs['value'])
	dump(value,path)
	msg = 'Dumped: %s %d'%(path,len(value))
	# print(msg)
	return value

def callback(value,key,values,**kwargs):
	values.update(value)
	return

def test_parallelize(path):

	paths = [path]

	paths = natsorted(set((subpath for path in set(paths) for subpath in glob(path))))

	iterable = paths
	values = {}
	processes = -1
	args = ()
	kwds = {'value':{}}
	callback_args = ()
	callback_kwds = {'values':values}

	parallelize = Parallelize(processes)

	parallelize(
		iterable,func,
		callback=callback,
		args=args,kwds=kwds,callback_args=callback_args,callback_kwds=callback_kwds,
		)

	assert all(len(values[key])==len(kwds['value'][key]) for key in kwds['value'])

	return


def test_pooler(path):

	paths = [path]
	paths = natsorted(set((subpath for path in set(paths) for subpath in glob(path))))

	iterable = paths
	values = {}
	processes = -1
	args = ()
	kwds = {}
	callback_args = ()
	callback_kwds = {'values':values}

	pooler = Pooler(processes)

	pooler(
		iterable,func,
		callback=callback,
		args=args,kwds=kwds,callback_args=callback_args,callback_kwds=callback_kwds,
		)

	assert all(len(values[key])==len(kwds['value'][key]) for key in kwds['value'])

	return

def main(*args,**kwargs):

	# test_parallelize(*args,**kwargs)
	test_pooler(*args,**kwargs)

	return


if __name__ == "__main__":

	args = argparser('path')

	main(*args,**args)
	
