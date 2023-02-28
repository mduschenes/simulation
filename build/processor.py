#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.io import join,split
from src.utils import argparser
from src.process import process

def main(args):

	arguments = {
		'--data':{
			'help':'Process data files',
			'type':str,
			'default':[],
			'nargs':'*'
		},
		'--settings':{
			'help':'Process plot settings',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--hyperparameters':{
			'help':'Process process settings',
			'type':str,
			'default':None,
			'nargs':'?'
		},
		'--pwd':{
			'help':'Process pwd',
			'type':str,
			'default':None,
			'nargs':'?',
		},		
		'--cwd':{
			'help':'Process cwd',
			'type':str,
			'default':None,
			'nargs':'?',
		},						
		'--verbose':{
			'help':'Verbosity',
			'action':'store_true'
		},												

	}

	wrappers = {
		'settings':lambda kwarg,wrappers,kwargs: join(split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**',''),'plot.json') if kwargs.get(kwarg) is None else kwargs.get(kwarg),
		'hyperparameters':lambda kwarg,wrappers,kwargs: join(split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**',''),'process.json') if kwargs.get(kwarg) is None else kwargs.get(kwarg),
		'pwd':lambda kwarg,wrappers,kwargs: split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**','') if kwargs.get(kwarg) is None else kwargs.get(kwarg),
		'cwd':lambda kwarg,wrappers,kwargs: split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**','') if kwargs.get(kwarg) is None else kwargs.get(kwarg),
		'verbose': lambda kwarg,wrappers,kwargs: kwargs.get(kwarg),
		'data':lambda kwarg,wrappers,kwargs: join(split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**',''),'**/data.hdf5'),
	}


	args = argparser(arguments,wrappers)

	process(*args,**args)

	return

if __name__ == '__main__':
	main(sys.argv[1:])