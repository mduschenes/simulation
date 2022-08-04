#!/usr/bin/env python

import os,sys,copy,itertools
import multiprocessing.dummy as mt
import numpy as np
import pandas as pd


if __name__ == "__main__":
	def func(data,labels):
		y = 1
		_label = ''
		
		for i,label in enumerate(labels):
			label = 'x_%d'%(label)
			_label += label
			x = np.expand_dims(data[label].values,-1) - data[label].values
			if i%2 == 0:
				y *= x
			else:
				y += x

		y = np.sum(y,axis=-1)

		return {_label:y}

	n_jobs = 4

	n = 10000
	p = 10
	data = np.random.randint(1,10,(n,p))
	columns = ['x_%d'%(i) for i in range(p)]
	data = pd.DataFrame(data=data,columns=columns)


	pool = mt.Pool(n_jobs)
	

	q = 8
	r = 2
	iters = np.random.randint(0,p,q).tolist()
	i = list(iters)
	iters = itertools.combinations(iters,r=r)

	print(len(i))
	# iters=  [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]

	iters = zip(itertools.repeat(data),iters)

	results = pool.starmap(func, iters)

	for result in results:
		for label in result:
			data[label] = result[label]

	print(data)
