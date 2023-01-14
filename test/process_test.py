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

	funcs = {
		'stat':{'':'mean','err':'std'}
	}

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
		
		prop = [attr for axis in axes for attr in key[axis]]
		independent = [attr for axis in axes[:-1] for attr in key[axis] if attr in df]
		dependent = [attr for axis in axes[-1:] for attr in key[axis] if attr in df]
		labels = {attr:key[label][attr] for attr in key[label] if attr in df and key[label][attr] is null}
		booleans = {attr:key[label][attr] for attr in key[label] if attr in df}

		boolean = [parse(attr,booleans[attr],df) for attr in booleans]	
		boolean = conditions(boolean,op='&')	



		by = [*labels,*independent]
		
		print(name,by,booleans,boolean.sum())
		groupby = df[boolean].groupby(by=by,as_index=False)

		# if name == '1':

		# 	for group in groupby.groups:
		# 		print(group)
		# 		print(groupby.get_group(group).shape)
		# 		print()
		# 	print()

		for func in funcs:
			agg,columns = {},{}
			by =  [*labels]

			for attr in df:
				agg[attr] = []
				if attr in dependent:
					functions = funcs[func]
					for function in functions:
						agg[attr].append(funcs[func][function])
						columns[(attr,funcs[func][function])] = delim.join([attr,function])
				else:
					functions = ['first']
					for function in functions:
						agg[attr].append(function)
						columns[(attr,function)] = attr

			print(func,prop,labels)
			value = groupby.agg(agg)
			print(value.columns.values)
			value.rename(columns=columns,inplace=True)
			print(value.columns.values)
			print(columns)
			# print(columns)
			# value.columns = [columns[label] for label in columns]
			# value.rename(columns,inplace=True)
			for i in zip(columns,value.columns.values):
				print(i[0],columns[i[0]],i[0]==i[1])
			continue

			for i,function in enumerate(funcs[func]):
				functions = {attr: funcs[func][function] for attr in dependent}
				names = [delim.join([attr,function]) for attr in dependent]

				agg.update(functions)
				value = groupby.agg(agg)

				print(function,value.shape)

				if not i:
					data[name][func] = value
				
				data[name][func][names] = value[dependent]


			data[name][func] = data[name][func].groupby(by=by)
			# print(data[name][func])
			# 	if not i:
			# 		data[name][func] = value

			# for i,func in enumerate(funcs[function]):
			# 	agg = {**{attr: funcs[function][func] for attr in axis},**{attr: 'first' for attr in df if attr not in axis}}
			# 	by =  [attr for attr in labels if attr in df and isinstance(labels[attr],Null)]
			# 	value = groupby.agg(agg).groupby(by=by)
			# 	if not i:
			# 		data[name][func] = value
			# 	data[name][func][[delim.join([attr,func]) for attr in axis]] = value[axis]

				# data[name][func] = groupby.pipe({**{attr: funcs[func] for attr in prop},**{attr: 'first' for attr in df if attr not in prop}}).groupby(by=[*labels])
					# data[name][func] = groupby.agg({**{attr: [funcs[func][prop] for prop in funcs[func]] for attr in axis},**{attr: 'first' for attr in df if attr not in axis}}).groupby(by=[attr for attr in labels if attr in df and isinstance(labels[attr],Null)])
				# data[name][func] = {group: data[name][func].get_group(group) for group in data[name][func].groups}
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
	test_find(path)
	test_parse(path) 
	# test_groupby(path)