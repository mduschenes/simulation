#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools
from copy import deepcopy as deepcopy
	
# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.system import Lattice


def test_lattice(path,tol):

	N = 11
	d = 2
	lattice = 'square'
	vertices = ['i','ij','i<j','<ij>','>ij<']

	lattice = Lattice(N,d,lattice)

	for vertex in vertices:
		edges = lattice(vertex)

		print(vertex)
		for edge in edges:
			print(edge)
		print()

	return

if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	test_lattice(path,tol)