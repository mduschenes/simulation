#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy,warnings

import numpy as np
import pandas as pd

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.process import parse,find
from src.utils import conditions,null,Null
from src.io import load,dump

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
	# log = file if hasattr(file,'write') else sys.stderr
	# traceback.print_stack(file=log)
	# log.write(warnings.formatwarning(message, category, filename, lineno, line))
	return
warnings.showwarning = warn_with_traceback


def _setup(args,kwargs):
	n = kwargs.get('n',10)
	
	verbose = kwargs.get('verbose',False)
	
	path = 'config/test.json'
	data = pd.DataFrame({
		'x':2*np.random.rand(n)-1,
		'y':2*np.random.rand(n)-1,
		'b':np.random.randint(0,n,n),
		'c':np.random.choice(['hello','world','goodbye'],n),
		'd':np.random.choice(['hello','world','goodbye'],n),
		'e':np.random.choice(['hello','world','goodbye'],n),
		'f':[tuple((2*np.random.rand(1)-1).tolist()) for i in range(n)],
		'g':np.random.choice(['first','second','third'],n),
		'h':np.random.choice(['first','second','third','four','five','six'],n),
	})
	axes = ['x','y']
	labels = ['label']
	properties = [*axes,*labels]
	
	dictionary = load(path)
	
	updates = {
		'path':path,'data':data,
		'axes':axes,'labels':labels,'properties':properties,'dictionary':dictionary,
		'verbose':verbose,
	}
	
	kwargs.update(updates)
	
	return



def test_conditions():
	args = ()
	kwargs = {}

	kwargs.update({'verbose':1})

	_setup(args,kwargs)
	
	df = kwargs['data']
	
	booleans = [df['b'].isin(df['b'].unique()[[0,1]]),df['c'] == 'hello']
	op = 'and'
	boolean = conditions(booleans,op)
	
	out = df[boolean]    
	_out = df[booleans[0] & booleans[1]]

	assert out.equals(_out), "Incorrect boolean conditions"

	return

def test_find():
	args = ()
	kwargs = {}

	kwargs.update({'verbose':1})

	_setup(args,kwargs)
	
	dictionary,properties,axes,labels = kwargs['dictionary'],kwargs['properties'],kwargs['axes'],kwargs['labels']
	
	keys = find(dictionary,properties)
	
	for name in keys:
		assert (
			all(isinstance(keys[name][prop],list) for prop in axes) and 
			all(isinstance(keys[name][prop],(list,dict)) for prop in labels) and
			all(isinstance(attr,str) and isinstance(keys[name][prop][attr],(str,Null)) for prop in labels for attr in keys[name][prop])
		), "Incorrect find key format"
	
	print(keys)
	
	return


def test_parse():
	args = ()
	kwargs = {}

	kwargs.update({'verbose':0})

	_setup(args,kwargs)
	
	data = kwargs['data']
	dictionary,properties,axes,labels = kwargs['dictionary'],kwargs['properties'],kwargs['axes'],kwargs['labels']
	verbose = kwargs['verbose']
	
	keys = find(dictionary,properties)

	print()
	print(data)
	print()
	
	
	for name in keys:        
		key = keys[name]
		for label in labels:
			for attr in key[label]:
				value = key[label][attr]
				boolean = parse(attr,value,data)
				
				if verbose:
					print(attr,value,data[attr].unique())
					print(boolean)
					print()

	return



if __name__ == '__main__':
	test_conditions()
	test_find()
	test_parse() 