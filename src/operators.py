#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp
import jax
import jax.numpy as np
import jax.scipy as sp
import jax.example_libraries.optimizers
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'
# np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})

# Logging
import logging
logger = logging.getLogger(__name__)

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,gradient,gradient_finite,gradient_fwd
from src.utils import array,dictionary,ones,zeros,arange,eye,rand,identity,diag,PRNGKey
from src.utils import tensorprod,trace,broadcast_to,expand_dims,moveaxis,repeat,take,inner,outer
from src.utils import summation,exponentiation
from src.utils import inner_abs2,inner_real2,inner_imag2
from src.utils import gradient_expm,gradient_sigmoid,gradient_inner_abs2,gradient_inner_real2,gradient_inner_imag2
from src.utils import eigh,qr
from src.utils import maximum,minimum,abs,real,imag,cos,sin,arctan,sqrt,mod,ceil,floor,heaviside,sigmoid
from src.utils import concatenate,vstack,hstack,sort,norm,interpolate,unique,allclose,isclose,isnaninf
from src.utils import parse,to_str,to_number,datatype,_len_,_iter_
from src.utils import pi,e
from src.utils import itg,flt,dbl

from src.io import load,dump,path_join,path_split

def haar(shape,bounds,random,seed,dtype):
	'''
	Initialize haar random unitary operator
	Args:
		shape (iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	data = rand(shape,bounds=bounds,random=random,key=seed,dtype=dtype)
	data /= sqrt(2)

	Q,R = qr(data)
	R = diag(R)
	R = diag(R/abs(R))
	
	data = Q.dot(R)

	data = data.astype(dtype)

	return data



def cnot(shape,bounds,random,seed,dtype):
	'''
	Initialize cnot unitary operator
	Args:
		shape (iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = array([
		[1,0,0,0],
		[0,1,0,0],
		[0,0,0,1],
		[0,0,1,0]])

	data = data.astype(dtype)

	return data


def hadamard(shape,bounds,random,seed,dtype):
	'''
	Initialize hadamard unitary operator
	Args:
		shape (iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = array([
		[1,1,],
		[1,-1]])/sqrt(2)

	data = data.astype(dtype)

	return data	


def toffoli(shape,bounds,random,seed,dtype):
	'''
	Initialize toffoli unitary operator
	Args:
		shape (iterable[int]): Shape of operator
		bounds (iterable): Bounds on operator value
		random (str): Type of random value
		seed (int,key): Seed for random number generator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''
	
	data = array([
		[1,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0],
		[0,0,1,0,0,0,0,0],
		[0,0,0,1,0,0,0,0],
		[0,0,0,0,1,0,0,0],
		[0,0,0,0,0,1,0,0],
		[0,0,0,0,0,0,0,1],
		[0,0,0,0,0,0,1,0]])

	data = data.astype(dtype)

	return data	


def operatorize(data,shape,hyperparameters,index=None,dtype=None):
	'''
	Initialize operators
	Args:
		data (dict,str,array): Label or path or array of operator
		shape (iterable[int]): Shape of operator
		hyperparameters (dict): Dictionary of hyperparameters for operator
		index (int): Index to initialize operator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''

	# Dimension of data
	d = min(shape)

	# Delimiter for string
	delimiter = '_'

	# Properties for strings
	props = {
		**{string: {'func':haar,'locality':index} for string in ['random','U','haar']},
		**{string: {'func':hadamard,'locality':1} for string in ['hadamard','H']},
		**{string: {'func':cnot,'locality':2} for string in ['cnot','CNOT','C']},
		**{string: {'func':toffoli,'locality':3} for string in ['toffoli','TOFFOLI','T']},
		**{string: {'func':{2:cnot,3:toffoli}[index],'locality':index} for string in ['control']},
		None: {'func':haar,'locality':index},
		}


	if data is None:
		string = hyperparameters['string']
	elif isinstance(data,str):
		string = data
	elif isinstance(data,array):
		string = None

	if string is None or isinstance(string,str):

		if string is None:
			strings = [string]
			locality = index
		elif all(string in props for string in string.split(delimiter)):
			strings = string.split(delimiter)
			locality = sum(props[string]['locality'] for string in strings)
		else:
			strings = None
			locality = index			

		assert (index%locality == 0), 'Incorrect operator with locality %d !%% index %d'%(locality,index)

		if string is not None:
			data = tensorprod([
				props[string]['func'](shape,
					bounds=hyperparameters['bounds'],
					random=hyperparameters['random'],
					seed=hyperparameters['seed'],
					dtype=dtype
					)
				for string in strings
				]*(index//locality)
			)
		else:
			data = array(load(data))
	
	# Assert data is unitary
	assert allclose(eye(d),data.conj().T.dot(data))
	assert allclose(eye(d),data.dot(data.conj().T))

	# Set dtype of data
	data = data.astype(dtype=dtype)

	return data