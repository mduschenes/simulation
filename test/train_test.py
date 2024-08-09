#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser,jit,array,allclose,delim,spawn,einsum,conjugate
from src.io import load,glob
from src.system import Dict
from src.iterables import namespace
from src.logger import Logger

from src.train import train

# logger = Logger()


def test_train(path,*args,**kwargs):

	model = train(path,*args,**kwargs)

	return


if __name__ == '__main__':

	arguments = 'path'
	args = argparser(arguments)

	test_train(*args,**args)
