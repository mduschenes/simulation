#!/usr/bin/env python

'''
Miscellaneous utility functions (for processing, plotting)
'''

# Import python modules
import os,sys,itertools,warnings,traceback
import numpy as np
import scipy as sp
import scipy.stats
import scipy.special
import pandas as pd
from pandas.api.types import is_float_dtype
from natsort import natsorted,realsorted
from math import prod

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,zeros,rand,random,randint,seeded,finfo,argparser
from src.utils import addition,multiply,divide,power,matmul,sqrt,floor,log10,abs
from src.utils import to_tuple,asscalar
from src.utils import maximum,minimum,abs,sort,log
from src.utils import arrays,scalars,nonzero

from src.iterables import permuter
from src.parallel import Parallel
from src.workflow import workflow

from src.io import load,dump

import sympy as sp

# Processing

def func_stat(data,func="min",stat='mean',**kwargs):
	default = lambda data: (lambda *args,data=data,**kwargs: data)
	stat = {'sem':lambda data,**kwargs:data.std()/np.sqrt(data.size)}.get(stat,stat) if isinstance(stat,str) else stat
	out = getattr(data,func,default(data))(**kwargs) if isinstance(func,str) else func(data,**kwargs)
	return getattr(out,stat,default(out))(**kwargs) if isinstance(stat,str) else stat(out,**kwargs)

def func_attr_stat(data,attr="objective",func="min",stat='mean',**kwargs):
	default = lambda data: (lambda *args,data=data,**kwargs: data)
	stat = {'sem':lambda data,**kwargs:data.std()/np.sqrt(data.size)}.get(stat,stat) if isinstance(stat,str) else stat
	attr = slice(None) if attr is None else attr
	out = getattr(data,func,default(data))(**kwargs) if isinstance(func,str) else func(data,**kwargs)
	return getattr(out,stat,default(out))(**kwargs) if isinstance(stat,str) else stat(out,**kwargs)

def func_MN(data):
	return data['M']/data['N']

def func_tau_unit(data):
	return data['tau']/data.get('unit',1)

def func_T_unit(data):
	return data['T']/data.get('unit',1)

def func_tau_noise_scale(data):
	return data['tau']/data.get('noise.scale',1)

def func_T_noise_scale(data):
	return data['T']/data.get('noise.scale',1)

def func_tau_J(data):
	return data['tau']*data.get('parameters.zz.parameters',1)

def func_T_J(data):
	return data['T']*data.get('parameters.zz.parameters',1)

def func_variables_relative_mean(data):
	out = np.array(data['variables.relative.mean'])
	return out/max(1,maximum(out))

def func_fisher_rank(data):
	out = np.array(list(data['fisher.eigenvalues']))
	out = sort(abs(out))
	out = out/maximum(out)
	out = asscalar(nonzero(out,axis=-1,eps=1e-13))
	return out

def func_fisher_eigenvalues(data):
	out = np.array(list(data['fisher.eigenvalues']))
	out = abs(out)
	out = out/maximum(out)
	out = to_tuple(out)
	return out

def func_hessian_rank(data):
	out = np.array(list(data['hessian.eigenvalues']))
	out = sort(abs(out))
	out = out/maximum(out)
	out = asscalar(nonzero(out,axis=-1,eps=1e-16))
	return out

def func_hessian_eigenvalues(data):
	out = np.array(list(data['hessian.eigenvalues']))
	out = abs(out)
	out = out/maximum(out)
	out = to_tuple(out)
	return out


def func_objective_func(data):
	out = data['N']*np.array(data['M'])*data['noise.parameters']*((data['D']**data['N']-1)/(data['D']**data['N']))
	return out	

def func_objective_func_err(data):
	out = 0*np.array(data['M'])
	return out	

def func_entropy(data):
	out = np.array(data['entropy'])/log(data['D']**data['N'])
	return out

def func_entropy_func(data):
	# Incorrect
	out = data['N']*np.array(data['M'])*data['noise.parameters']
	return out	

def func_entropy_func_err(data):
	out = 0*np.array(data['M'])
	return out	

def func_purity(data):
	out = np.array(data['purity'])
	return out	

def func_purity_func(data):
	out = 1-2*data['N']*np.array(data['M'])*data['noise.parameters']*((data['D']**data['N']-1)/(data['D']**data['N']))
	return out	

def func_purity_func_err(data):
	out = 0*np.array(data['M'])
	return out	

def func_similarity(data):
	out = 1-np.array(data['similarity'])
	return out

def func_similarity_func(data):
	out = 1-np.array(data['similarity'])	
	return out	

def func_similarity_func_err(data):
	out = 0*np.array(data['M'])
	return out

def func_divergence(data):
	out = np.array(data['divergence'])/log(data['D']**data['N'])
	return out

def func_divergence_func(data):
	# Incorrect	
	out = data['N']*np.array(data['M'])*data['noise.parameters']*((data['D']**data['N']-1)/(data['D']**data['N']))/log(data['D']**data['N'])
	return out	

def func_divergence_func_err(data):
	out = 0*np.array(data['M'])
	return out		

def label(string,label):
	strings = {
		'epsilon':{
			1e-7:"32~\\textrm{bit}~(\\varepsilon \\sim 10^{-7})",
			1e-16:"64~\\textrm{bit}~(\\varepsilon \\sim 10^{-16})",
			1e-19:"128~\\textrm{bit}~(\\varepsilon \\sim 10^{-19})",
		}
	}

	string = strings.get(label,{}).get(string,string)

	return string


def error(data,*args,**kwargs):
	'''
	Workflow function
	Args:
		data (str,dict): Workflow data
		args (iterable): Workflow positional arguments 
		kwargs (dict): Workflow keyword arguments 
	Returns:
		data (dict): Workflow data
	'''	

	def func(data,*args,**kwargs):

		def generator(iteration,shape,scale,seed=None,dtype=None):
			
			# seeded(seed)
			
			value = 2*random((2,*shape),dtype=dtype)-1
			value = (value[0] + 1j*value[1])/sqrt((value**2).sum(0))
			value = (scale/2)*value

			return value

		# def matmul(a,b,dtype=None):
		# 	c = zeros(a.shape,dtype=a.dtype)
		# 	for i in range(a.shape[0]):
		# 		for j in range(a.shape[1]):
		# 			c[i,j] = a[i].dot(b.T[j])
		# 	return c

		size = kwargs.get('size',2)
		ord = kwargs.get('ord',2)
		maxdtype = 'complex%d'%(int(kwargs.get('dtype').replace('complex',''))//1)
		maxftype = 'float%d'%(int(kwargs.get('dtype').replace('complex',''))//2)

		indexes = kwargs.get('indexes',[])
		iterations = range(
			max(2,(indexes.start if isinstance(indexes,range) else min(indexes))),
			max(1,(indexes.stop  if isinstance(indexes,range) else max(indexes)))+1
		)
		samples = range(kwargs.get('sample'))
		seeded(kwargs.get('seed'))
		seeds = {sample: rand(bounds=[0,1e12],random='randint') for sample in samples}
		
		bits = {sample:
				{
				**{types:
					{int(floor(log10(finfo('complex%d'%(bit)).eps))): bit 
					for bit in kwargs.get('bits',[])}
					for types in ['numerical']
				},
				**{types: {bit: bit for bit in kwargs.get('epsilon',[])} 
					for types in ['analytical','probabilistic']
				},
			}
			for sample in samples}
		eps = {sample: {types: {bit: 
			power(addition([power(power(
				1+(power(10,bit,dtype=maxftype)),
				size-i-1,dtype=maxftype)-1,
				ord,dtype=maxftype)
				for i in range(size)],dtype=maxftype),
				1/ord,dtype=maxftype) 
			for bit in bits[sample][types]}
			for types in bits[sample]}
			for sample in bits}



		for sample in bits:
			for types in bits[sample]:
				for bit in bits[sample][types]:

					print(sample,types,bit,eps[sample][types][bit])

		dtype = {sample: {
			**{types: {bit: 'complex%d'%(bits[sample][types][bit]) for bit in bits[sample][types]} for types in ['numerical']},
			**{types: {bit: maxdtype for bit in bits[sample][types]} for types in ['analytical','probabilistic']},
		} for sample in bits}
		ftype = {sample: {
			**{types: {bit: 'float%d'%(bits[sample][types][bit]//2) for bit in bits[sample][types]} for types in ['numerical']},
			**{types: {bit: maxftype for bit in bits[sample][types]} for types in ['analytical','probabilistic']},			
		}
		for sample in bits}

		V = sp.Matrix([[sp.exp(sp.Mul(sp.I,2*sp.pi,sp.Rational(i*j,size))) 
			for j in range(size)] 
			for i in range(size)])/sp.sqrt(size)

		S = [sp.Rational(randint(shape=None,bounds=[1,i]) if i>1 else 0,i) for i in randint(shape=size,bounds=[1,size**2])]
		D = lambda k=1: sp.diag(*(sp.exp(sp.Mul(sp.I,2*sp.pi,s,k)) for s in S))
	
		matrix = lambda k=1: V*D(k)*V.H
		norm = lambda A,bit=maxftype,ord=ord: ((((abs(A,dtype=bit))**ord).sum(dtype=bit))**(1/ord)).real
		numerical = lambda A,bit: array(sp.N(A,bit),dtype=maxdtype)

		A = {sample: {
			**{types: {bit: (lambda bit,types,sample,i=None,A=numerical(matrix(),bit=-bit): A)
				for bit in bits[sample][types]} for types in ['numerical']},
			**{types: {bit: (lambda bit,types,sample,i=None,A=1: A)
				for bit in bits[sample][types]} for types in ['analytical']},
			**{types: {bit: (lambda bit,types,sample,i=None,A=numerical(matrix(),bit=-bit): A)
				for bit in bits[sample][types]} for types in ['probabilistic']},
		} for sample in bits}
		B = {sample: {types: {bit: A[sample][types][bit](bit,types,sample) for bit in bits[sample][types]} for types in bits[sample]} for sample in bits}
		C = {sample: {
			**{types: {bit: 
				(lambda bit,types,sample,i=None,A=numerical(matrix(),bit=-bit): 0) 
				for bit in bits[sample][types]} for types in ['numerical']},
			**{types: {bit: 
				(lambda bit,types,sample,i=None,A=1: 0) 
				for bit in bits[sample][types]} for types in ['analytical']},
			**{types: {bit: 
				(lambda bit,types,sample,i=None,A=numerical(matrix(),bit=-bit): generator(iteration=i,shape=A.shape,scale=eps[sample][types][bit],seed=seeds[sample],dtype=maxftype))
				for bit in bits[sample][types]} for types in ['probabilistic']},
		} for sample in bits}		
		functions = {sample: {
			**{types: {bit: 
				(lambda bit,types,sample,i=None: matmul(A[sample][types][bit](bit,types,sample,i),B[sample][types][bit],dtype=dtype[sample][types][bit]) + C[sample][types][bit](bit,types,sample,i))
				for bit in bits[sample][types]} for types in ['numerical']},

			**{types: {bit: 
				(lambda bit,types,sample,i=None: multiply(A[sample][types][bit](bit,types,sample,i),B[sample][types][bit],dtype=dtype[sample][types][bit]) + C[sample][types][bit](bit,types,sample,i))
				for bit in bits[sample][types]} for types in ['analytical']},

			**{types: {bit: 
				(lambda bit,types,sample,i=None: matmul(A[sample][types][bit](bit,types,sample,i),B[sample][types][bit],dtype=dtype[sample][types][bit]) + C[sample][types][bit](bit,types,sample,i))
				for bit in bits[sample][types]} for types in ['probabilistic']},				
		} for sample in bits}
		values = {sample: {
			**{types: {bit: 
				(lambda bit,types,sample,i=None: norm(B[sample][types][bit] - numerical(matrix(i),bit=-bit))/norm(B[sample][types][bit]))
				for bit in bits[sample][types]} for types in ['numerical']},

			**{types: {bit: 
				(lambda bit,types,sample,i=None: multiply(
					divide(
						power(normalization[sample][types][bit],i,dtype=ftype[sample][types][bit]),
						norm(B[sample][types][bit]),dtype=ftype[sample][types][bit]),
						((power(1+eps[sample][types][bit],i,dtype=ftype[sample][types][bit]) - 1)),dtype=ftype[sample][types][bit]))
				for bit in bits[sample][types]} for types in ['analytical']},

			**{types: {bit: 
				(lambda bit,types,sample,i=None: norm(B[sample][types][bit] - numerical(matrix(i),bit=-bit))/norm(B[sample][types][bit]))
				for bit in bits[sample][types]} for types in ['probabilistic']},				
		} for sample in bits}

		normalization = {sample: {types: {bit: norm(A[sample][types][bit](bit,types,sample),ord=ord).real
			for bit in A[sample][types]} for types in A[sample]}
			for sample in bits}

		for i in iterations:

			for sample in bits:
				for types in bits[sample]:
					for bit in bits[sample][types]:

							B[sample][types][bit] = functions[sample][types][bit](bit,types,sample,i)

			if i in indexes:

				for sample in bits:
					for types in bits[sample]:
						for bit in bits[sample][types]:

							value = values[sample][types][bit](bit,types,sample,i)

							print(i,sample,types,bit,value,B[sample][types][bit].dtype,value.dtype)

							data['index'].append(i)
							data['value'].append(value)
							data['size'].append(size)
							data['epsilon'].append(10**(bit))
							data['type'].append(types)
							data['sample'].append(sample)
							data['seed'].append(seeds[sample])
			else:

				print(i)

		return


	permutations = {}
	groups = []
	data.update({attr:[] for attr in ['index','value','size','epsilon','type','sample','seed']})

	permutations = permuter(permutations,groups=groups) if permutations else [{}]

	for permutation in permutations:

		kwargs.update(permutation)

		func(data,**kwargs)

	return



# def func(data,*args,**kwargs):
# 	if data is None:
# 		data = {}
	
# 	default = {'index':[],'value':[]}

# 	data.update({attr: default[attr] for attr in default if attr not in data})

# 	data.update({attr: [*data[attr]] for attr in data})

# 	for i in kwargs.get('indexes'):
# 		print(i)
# 		data['index'].append(max(data['index'],default=-1)+1)
# 		data['value'].append(i)

# 	return
