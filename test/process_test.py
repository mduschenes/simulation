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

from src.process import parse,find,apply
from src.utils import conditions,null,delim,to_repr,to_eval
from src.iterables import setter,getter
from src.io import load,dump

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
	# log = file if hasattr(file,'write') else sys.stderr
	# traceback.print_stack(file=log)
	# log.write(warnings.formatwarning(message, category, filename, lineno, line))
	return
warnings.showwarning = warn_with_traceback


def _setup(args,kwargs):
	n = kwargs.get('n',10)
	
	path = kwargs.get('path')

	verbose = kwargs.get('verbose',False)
	
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

	path = path if path is not None else 'data/data/**/data.hdf5'
	df = load(path,default={},wrapper='df')

	axes = ['x','y']
	other = ['label']
	
	path = 'config/plot.test.json'
	settings = load(path)
	
	updates = {
		'data':data,'df':df,
		'axes':axes,'other':other,'settings':settings,
		'verbose':verbose,
	}
	
	kwargs.update(updates)
	
	return



def test_conditions(path=None):
	args = ()
	kwargs = {}

	kwargs.update({'path':path,'verbose':1})

	_setup(args,kwargs)
	
	df = kwargs['data']
	
	booleans = [df['b'].isin(df['b'].unique()[[0,1]]),df['c'] == 'hello']
	op = 'and'
	boolean = conditions(booleans,op)
	
	out = df[boolean]    
	_out = df[booleans[0] & booleans[1]]

	assert out.equals(_out), "Incorrect boolean conditions"

	return

def test_find(path=None):
	args = ()
	kwargs = {}

	kwargs.update({'path':path,'verbose':1})

	_setup(args,kwargs)
	
	settings,axes,other = kwargs['settings'],kwargs['axes'],kwargs['other']
	
	keys = [*axes]
	other = [*other]
	keys = find(settings,keys,*other)
	
	for name in keys:
		print(name)
		for attr in keys[name]:
			print(attr,keys[name][attr])
		print()
		assert all(isinstance(keys[name][prop],(str,dict)) for prop in keys[name]), "Incorrect find key format"
	
	return


def test_parse(path=None):
	args = ()
	kwargs = {}

	kwargs.update({'path':path,'verbose':1})

	_setup(args,kwargs)
	
	df = kwargs['df']
	settings,axes,other = kwargs['settings'],kwargs['axes'],kwargs['other']
	verbose = kwargs['verbose']
	
	keys = [*axes]
	other = [*other]
	keys = find(settings,keys,*other)

	print(df)
	print()
	
	
	for name in keys:  

		key = keys[name]

		for attr in other:
			for prop in key[attr][attr]:
				value = key[attr][attr][prop]
				boolean = parse(prop,value,df)
			
				if verbose and prop in df:
					print(prop,value,df[prop][boolean].unique())
					print()

	return


def test_groupby(path=None):


	def mean(group):
		return group.mean()

	def std(group):
		return group.std()

	args = ()
	kwargs = {}	

	kwargs.update({'path':path,'verbose':0})

	_setup(args,kwargs)
	
	df = kwargs['df']
	settings,axes,other = kwargs['settings'],kwargs['axes'],kwargs['other']

	keys = find(settings,axes,*other)

	data = {}

	for name in keys:      
		
		apply(name,keys,data,df)

	return




if __name__ == '__main__':
	path = 'data/test/**/data.hdf5'
	# test_conditions(path)
	# test_find(path)
	# test_parse(path) 
	# test_groupby(path)
