#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.run import run

def main(*args,**kwargs):

	run(*args,**kwargs)

	return

if __name__ == '__main__':

	arguments = 'settings'

	args = argparser(arguments)

	main(*args,**args)
