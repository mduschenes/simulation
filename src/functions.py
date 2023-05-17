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


def func_MN(data):
	return data['M']/data['N']

def func_tau(data):
	return data['tau']/data.get('T',1)

def func_T(data):
	return data['T']/data.get('T',1)

def func_tau_noise(data):
	return data['tau']/data.get('noise.scale',1)

def func_T_noise(data):
	return data['T']/data.get('noise.scale',1)