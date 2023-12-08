#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.io import join,split,convert
from src.process import process

def main(args):

	arguments = {
		'--data':{
			'help':'Process data files',
			'type':str,
			'default':[],
			'nargs':'*'
		},
		'--path':{
			'help':'Convert data files',
			'type':str,
			'default':[],
			'nargs':'*'
		},		
		'--quiet':{
			'help':'Quiet',
			'action':'store_true'
		},												

	}

	wrappers = {
		'verbose': lambda kwarg,wrappers,kwargs: not kwargs.pop('quiet',True),
		'wrapper':lambda kwarg,wrappers,kwargs: 'pd.dict',
		'path':lambda kwarg,wrappers,kwargs: join(split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**',''),'**/data.hdf5'),
		'data':lambda kwarg,wrappers,kwargs: join(split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**',''),'**/data.tmp.hdf5'),
	}


	args = argparser(arguments,wrappers)

	convert(*args,**args)

	return

if __name__ == '__main__':
	main(sys.argv[1:])