#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.io import join,split,merge

def process(path,data,*args,**kwargs):

	merge(data,path,*args,**kwargs)

	return

def main(args):

	arguments = {
		'--path':{
			'help':'Data path',
			'type':str,
			'default':[],
			'nargs':'*'
		},
		'--data':{
			'help':'Data name',
			'type':str,
			'default':'data.hdf5',
			'nargs':'?'
		},
		'--quiet':{
			'help':'Quiet',
			'action':'store_true'
		},												

	}

	wrappers = {
		'verbose': lambda kwarg,wrappers,kwargs: not kwargs.pop('quiet',True),
		'data':lambda kwarg,wrappers,kwargs: join(split(kwargs['path'][-1] if kwargs['path'] else '.',directory=True).replace('/**','').replace('**',''),'**/%s'%(kwargs['data'])),
		'path':lambda kwarg,wrappers,kwargs: join(split(kwargs['path'][-1] if kwargs['path'] else '.',directory=True).replace('/**','').replace('**',''),(split(kwargs['data'],file_ext=True))),
	}

	args = argparser(arguments,wrappers)

	process(*args,**args)

	return

if __name__ == '__main__':
	main(sys.argv[1:])