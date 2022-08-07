#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy

import jax
import jax.numpy as np
import numpy as onp

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.io import load,dump,join,split,edit

# Logging
# from src.utils import logconfig
# conf = 'config/logging.conf'
# logger = logconfig(__name__,conf=conf)


def test_path(path='data/data.hdf5'):
	new = edit(
			path=path,
			directory=None,
			file=(lambda directory,file,ext,delimiter: delimiter.join([*file.split(delimiter)[:]])),
			ext=None,
			delimiter='.'
			)

	assert new == path, "Incorrect path edit"

	return

def test_hdf5(path='data/data.hdf5'):
	return
	# Create data
	def rand(shape=None):
		if shape is None:
			if onp.random.rand() < 0.5:
				return onp.random.randint(0,100)
			else:
				return 'sfdsgsdg'
		else:
			return np.array(onp.random.rand(*shape))
	g = 3
	n = 2
	shape = (7,3)
	groups = ['%d'%(i) for i in range(g)]
	instances = ['%d'%(i) for i in range(n)]
	datasets = ['data','values','parameters']
	attributes = ['n','m','k']
	attrs = [*datasets,*attributes]
	data = {
		group: {
			instance:{
				**{attr: rand(shape) for attr in datasets},
				**{attr: rand() for attr in attributes}
				}
			for instance in instances
			}
		for group in groups
		}

	# Dump data
	wr = 'w'
	ext = 'hdf5'
	kwargs = {}

	dump(data,path,wr=wr,**kwargs)

	# Load data
	wr = 'r'
	ext = 'hdf5'
	kwargs = {}

	new = load(path,wr=wr,**kwargs)



	# Check dumped and loaded data are equal
	for group in groups:
		for instance in instances:
			for attr in attrs:
				msg = "group: %s, instance: %s, attr: %s Unequal"%(group,instance,attr)
				if isinstance(data[group][instance][attr],(int,np.integer,float,np.floating,str)):
					assertion = data[group][instance][attr] == new[group][instance][attr]
				else:
					assertion = np.allclose(data[group][instance][attr],new[group][instance][attr])
				assert assertion,msg


	return