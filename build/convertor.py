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

	def wrapper(data):

		import pandas as pd
		from src.utils import array,padding

		func = lambda data: data[data['iteration']==data['iteration.max']]
		data = func(data)

		options = {**{'orient':'list'}}
		data = data.to_dict(**options)

		types = (tuple,)
		for key in data:
			if all(isinstance(i,types) for i in data[key]):

				indices = {}
				for i,tmp in enumerate(data[key]):
					index = len(tmp)
					if index not in indices:
						indices[index] = []
					indices[index].append(i)

				tmp = {index: array([data[key][index] for index in indices[index]]) for index in indices}

				shape = tuple((max(tmp[index].shape[i] for index in tmp) for i in range(min(tmp[index].ndim for index in tmp))))

				tmp = {index: padding(tmp[index],shape=shape,random='zeros',dtype=tmp[index].dtype) for index in tmp}

				indices = {i:(index,indices[index].index(i)) for index in indices for i in indices[index]}
				indices = [indices[i] for i in range(len(indices))]

				data[key] = [tmp[i[0]][i[1]] for i in indices]

		return data

	wrappers = {
		'verbose': lambda kwarg,wrappers,kwargs: not kwargs.pop('quiet',True),
		'wrapper':lambda kwarg,wrappers,kwargs: ['pd',wrapper],
		'path':lambda kwarg,wrappers,kwargs: join(split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**',''),'**/data.hdf5'),
		'data':lambda kwarg,wrappers,kwargs: join(split(kwargs['data'][-1] if kwargs['data'] else '.',directory=True).replace('/**','').replace('**',''),'**/data.tmp.hdf5'),
	}


	args = argparser(arguments,wrappers)

	convert(*args,**args)

	return

if __name__ == '__main__':
	main(sys.argv[1:])