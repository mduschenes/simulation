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

	arguments = {
		'--settings':{
			'help':'Settings',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--device':{
			'help':'Device',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--job':{
			'help':'Job',
			'type':str,
			'default':None,
			'nargs':'?'
		},	
		'--cmd':{
			'help':'Command',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--env':{
			'help':'Environment',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--dry-run':{
			'help':'Execute',
			'action':'store_true'
		},
		'--quiet':{
			'help':'Verbose',
			'action':'store_true'
		},										
		}		

	wrappers = {
		'execute': lambda kwarg,wrappers,kwargs: not kwargs.pop('dry-run',True),
		'verbose': lambda kwarg,wrappers,kwargs: not kwargs.pop('quiet',True),
		}

	args = argparser(arguments,wrappers)

	main(*args,**args)
