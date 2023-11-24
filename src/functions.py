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

from src.utils import array
from src.utils import to_tuple,asscalar
from src.utils import maximum,minimum,abs,sort,log
from src.utils import arrays,scalars,nonzero

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

def func_entropy(data):
	out = np.array(data['entropy'])/log(data['D']**data['N'])
	return out

def func_purity(data):
	out = 1-np.array(data['purity'])
	return out	

def func_similarity(data):
	out = 1-np.array(data['similarity'])
	return out

def func_divergence(data):
	# out = np.array(data['divergence'])
	out = np.array(data['divergence'])/log(data['D']**data['N'])
	return out