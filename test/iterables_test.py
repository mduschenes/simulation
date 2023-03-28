#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy

import numpy as onp
import jax.numpy as np
	
# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import is_array,is_ndarray
from src.iterables import getter,setter,permuter,equalizer
from src.iterables import search,indexer,inserter

def test_equalizer(path=None,tol=None):
	a = {1:{2:[3,4],3:lambda x:x,4:{1:[],2:[{4:np.array([])}]}}}
	b = {1:{2:[3,4],3:lambda x:x,4:{1:[],2:[{4:onp.array([])}]}}}

	types = (dict,list,)
	exceptions = lambda a,b: any(any(e(a) for e in exception) and any(e(b) for e in exception) 
			for exception in [[callable],[is_array,is_ndarray]])

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
	for index,shape,item in search(iterable,types=types):
		print(index,shape,item)

	item = -1
	index = indexer(item,iterable,yielder=True,types=types)
	print(item,list(index))

	index = [0,2,'goodbye',2,'label','axis',1]
	item = -2
	inserter(index,item,iterable,types=types)
	print(iterable)

	return



if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 

	test_search(path=path,tol=tol)