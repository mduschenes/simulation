#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import np,onp
from src.utils import arrays,scalars
from src.io import load,dump
from src.iterables import getter,setter,permuter,equalizer
from src.iterables import search,find,inserter,indexer

def test_equalizer(path=None,tol=None):
	a = {1:{2:[3,4],3:lambda x:x,4:{1:[],2:[{4:np.array([])}]}}}
	b = {1:{2:[3,4],3:lambda x:x,4:{1:[],2:[{4:onp.array([])}]}}}

	types = (dict,list,)
	exceptions = lambda a,b: any(any(e(a) for e in exception) and any(e(b) for e in exception) 
			for exception in [[callable],[lambda a: isinstance(a,arrays)]])

	x,y = a[1][4][2][0][4],b[1][4][2][0][4]

	try:
		equalizer(a,b,types=types,exceptions=exceptions)
	except AssertionError as e:
		print(e)
		raise

	return


def test_search(path=None,tol=None):

	iterable = [['hi','world',{'goodbye':[-1,5,{'no':[4,-1],'x':'a','y':'b','label':{'axis':[1,2,3]}}]}]]
	types = (list,dict,)

	print(iterable)
	for index,shape,item in search(iterable,returns=True,types=types):
		print(index,shape,item)

	print()

	item = -1
	index = find(item,iterable,types=types)
	print(item,list(index))

	print()

	index = [0,2,'goodbye',2,'label','axis',1]
	item = -2
	inserter(index,item,iterable,types=types)
	print(iterable)

	print()

	path = 'config/settings.json'
	iterable = load(path)
	items = ['seed']
	types = (list,dict)
	for index,shape,item in search(iterable,returns=True,items=items,types=types):
		print(index,shape,item)

	print()

	path = 'config/plot.json'
	iterable = load(path)
	items = ['x','y','label']
	types = (list,dict)
	for index,shape,item in search(iterable,returns=True,items=items,types=types):
		print(index,shape)
		print(item)
		print()


	return


def test_permuter(path=None,tol=None):

	iterable = {'hi':[1,2],'world':[-1,-2],'test':[True,False]}

	groupings = [
		None,
		[['hi','world']],
		[['hi','world']],
	]
	filterings = [
		None,
		None,
		lambda dictionaries: (dictionary for dictionary in dictionaries if dictionary['test'] is True)
	]
	results = [
		[
		{'hi':1,'world':-1,'test':True},{'hi':1,'world':-1,'test':False},
		{'hi':1,'world':-2,'test':True},{'hi':1,'world':-2,'test':False},
		{'hi':2,'world':-1,'test':True},{'hi':2,'world':-1,'test':False},
		{'hi':2,'world':-2,'test':True},{'hi':2,'world':-2,'test':False},
		],
		[
		{'hi':1,'world':-1,'test':True},{'hi':1,'world':-1,'test':False},
		{'hi':2,'world':-2,'test':True},{'hi':2,'world':-2,'test':False},
		],
		[
		{'hi':1,'world':-1,'test':True},{'hi':2,'world':-2,'test':True},
		]
	]

	for groups,filters,values in zip(groupings,filterings,results):
		
		iterables = permuter(iterable,groups=groups,filters=filters)

		assert all(i==j for i,j in zip(iterables,values)), "Incorrect permute for groups %r"%(groups)

	print('Passed')

	return




if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 

	# test_search(path=path,tol=tol)
	# test_equalizer(path=path,tol=tol)
	test_permuter(path=path,tol=tol)