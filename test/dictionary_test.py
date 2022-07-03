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

from src.dictionary import updater,getter,setter,permuter,equalizer

def test_equalizer():
	a = {1:{2:[3,4],3:3}}
	b = {1:{2:[3,4],3:lambda x:x}}
	exceptions = lambda a,b: callable(a) and callable(b)

	try:
		equalizer(a,b,exceptions=exceptions)
	except AssertionError as e:
		print(e)
		pass

	return