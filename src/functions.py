#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings,traceback
from copy import deepcopy
from functools import partial,wraps
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


def func_objective_min_mean(data,attr="objective",func="min",stat='mean',**kwargs):
	default = lambda data: (lambda *args,data=data,**kwargs: data)
	attr = slice(None) if attr is None else attr
	out = getattr(data[attr],func,default(data[attr]))(**kwargs)
	return getattr(out,stat,default(out))()

def func_objective_min_sem(data,attr="objective",func="min",stat='sem',**kwargs):
	default = lambda data: (lambda *args,data=data,**kwargs: data)
	attr = slice(None) if attr is None else attr
	out = getattr(data[attr],func,default(data[attr]))(**kwargs)
	return getattr(out,stat,default(out))()	

def func_objective_max_mean(data,attr="objective",func="max",stat='mean',**kwargs):
	default = lambda data: (lambda *args,data=data,**kwargs: data)
	attr = slice(None) if attr is None else attr
	out = getattr(data[attr],func,default(data[attr]))(**kwargs)
	return getattr(out,stat,default(out))()

def func_objective_max_sem(data,attr="objective",func="max",stat='sem',**kwargs):
	default = lambda data: (lambda *args,data=data,**kwargs: data)
	attr = slice(None) if attr is None else attr
	out = getattr(data[attr],func,default(data[attr]))(**kwargs)
	return getattr(out,stat,default(out))()	

def func_MN(data):
	return data['M']/data['N']

def func_tau(data):
	return data['tau']/data.get('noise.scale',1)

def func_T(data):
	return data['T']/data.get('noise.scale',1)


