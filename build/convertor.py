#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

os.environ['NUMPY_BACKEND'] = 'numpy'

from src.utils import argparser
from src.io import join,split,merge
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

	def wrapper(data):
		for key in list(data):
			if data[key] is None:
				data.pop(key);
				continue
			if any(data[key][attr].ndim>1 for attr in data[key]):
				index = data[key]['iteration']==data[key]['iteration.max']
				for attr in data[key]:
					data[key][attr] = data[key][attr][index]
		return data

	wrappers = {
		'verbose': lambda kwarg,wrappers,kwargs: not kwargs.pop('quiet',True),
		'wrapper':lambda kwarg,wrappers,kwargs: wrapper,
		'path':lambda kwarg,wrappers,kwargs: join(split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**',''),'data.hdf5'),
		'data':lambda kwarg,wrappers,kwargs: join(split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**',''),'*/data.hdf5'),
	}

	args = argparser(arguments,wrappers)

	merge(*args,**args)

	return

if __name__ == '__main__':
	main(sys.argv[1:])