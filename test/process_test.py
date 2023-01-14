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
from src.utils import conditions,null,delim
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

	path = path if path is not None else 'config/data/**/data.hdf5'
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
		assert all(isinstance(keys[name][prop],dict) for prop in keys[name]), "Incorrect find key format"
	
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

	# Steps:
		# - Load data and settings
		# - Get data axes and labels based on branches of settings
		# - Iterate over all distinct data branches of settings
		# - Filter with booleans of all labels
		# - Group by non-null labels and independent
		# - Aggregate functions of dependent (mean,std) for each group
		# - Assign new labels for functions with label.function 
		# - Regroup with non-null labels
		
		# - Reshape each data into axes for [plot.type,plot.row,plot.col,plot.line=(plot.group,plot.function),plot.axis]
		# - Adjust settings based on data
		# - Plot data

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
	verbose = kwargs['verbose']

	keys = [*axes]
	other = [*other]
	keys = find(settings,keys,*other)

	data = {}

	print()
	print(df)
	print()

	for name in keys:      
		key = keys[name]

		data[name] = {}

		attr = other[0]
		label = key[attr].get('label',{})
		funcs = key[attr].get('func',{})

		if not funcs:
			funcs = {"stat":{"":"mean","err":"std"}}

		independent = [attr for axis in axes[:-1] for attr in key[axis] if attr in df]
		dependent = [attr for axis in axes[-1:] for attr in key[axis] if attr in df]
		labels = [attr for attr in label if attr in df and label[attr] is null]

		boolean = [parse(attr,label[attr],df) for attr in label]
		boolean = conditions(boolean,op='&')	

		by = [*labels,*independent]

		groupby = df[boolean].groupby(by=by,as_index=False)

		for func in funcs:

			print(independent,dependent,labels)

			agg = {
				**{attr : [(attr,'first')] for attr in df},
				**{attr : [(delim.join(((attr,function))),funcs[func][function]) for function in funcs[func]] for attr in df if attr in dependent},
			}
			droplevel = dict(level=0,axis=1)
			by = [*labels]


			data[name][func] = groupby.agg(agg).droplevel(**droplevel).groupby(by=by,as_index=False)

			for group in data[name][func].groups:
				value = data[name][func].get_group(group)
				print(group,value.shape)
			print()
			# 	# print(data[name][func])
			# print()
	return


if __name__ == '__main__':
	path = 'config/test/**/data.hdf5'
	# test_conditions(path)
	# test_find(path)
	# test_parse(path) 
	test_groupby(path)