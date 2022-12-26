#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,rand,allclose,scalars
from src.io import load,dump,join,split,edit,dirname
from src.call import rm

# Logging
# from src.utils import logconfig
# conf = 'config/logging.conf'
# logger = logconfig(__name__,conf=conf)


def test_path(path='.tmp.tmp/data.hdf5'):
	new = edit(
			path=path,
			directory=None,
			file=(lambda directory,file,ext,delimiter: delimiter.join([*file.split(delimiter)[:]])),
			ext=None,
			delimiter='.'
			)

	assert new == path, "Incorrect path edit"

	return

def test_hdf5(path='.tmp.tmp/data.hdf5'):

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
				if isinstance(data[group][instance][attr],scalars):
					assertion = data[group][instance][attr] == new[group][instance][attr]
				else:
					assertion = allclose(data[group][instance][attr],new[group][instance][attr])
				assert assertion,msg


	path = dirname(path)

	rm(path,execute=True)

	return