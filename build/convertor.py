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

	from src.utils import array,concatenate
	from src.io import load,dump,glob,cp,mkdir
	from src.call import call

	# size = 400
	# step = 100
	# cwd = 'gobi/test/eig'
	# for i in range(0,size,step):
	# 	mkdir(join(cwd,'tmp.%d'%(i)))
	# 	for j in range((i)*step,(i+1)*step):
	# 		cp(join(cwd,'%d'%(j)),join(cwd,'tmp.%d'%(i),'%d'%(j)))
	# 	call('./processor.py',join(cwd,'tmp.%d'%(i)))

	# mkdir gobi/test/eig/tmp.100 && for i in {0..100};do mv gobi/test/eig/$i gobi/test/eig/tmp.100/$i;done && ./processor.py gobi/test/eig/tmp.100	
	# mkdir gobi/test/eig/tmp.200 && for i in {100..200};do mv gobi/test/eig/$i gobi/test/eig/tmp.200/$i;done && ./processor.py gobi/test/eig/tmp.200
	# mkdir gobi/test/eig/tmp.300 && for i in {200..300};do mv gobi/test/eig/$i gobi/test/eig/tmp.300/$i;done && ./processor.py gobi/test/eig/tmp.300
	# mkdir gobi/test/eig/tmp.400 && for i in {300..400};do mv gobi/test/eig/$i gobi/test/eig/tmp.400/$i;done && ./processor.py gobi/test/eig/tmp.400

	# path = join(split(args['path'],directory=-1),'*','data',ext='hdf5')	
	# data = {}
	# for path in glob(path):

	# 	name = join(split(path,directory_file=True),ext='tmp.hdf5')
	# 	tmp = convert(path,name)
	# 	path = name

	# 	tmp = load(path,verbose=True)
	# 	# print(tmp)
	# 	for key in tmp:
	# 		if key not in data:
	# 			data[key] = tmp[key]
	# 		else:
	# 			data[key] = concatenate((data[key],tmp[key]))
	
	# path = join(split(args['path'],directory=-1),'data',ext='hdf5')	

	# for key in data:
	# 	print(key,data[key].shape)

	# dump(data,path,verbose=True)

	convert(*args,**args)

	return

if __name__ == '__main__':
	main(sys.argv[1:])