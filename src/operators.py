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
PATHS = ['',".."]
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

def operatorize(data,shape,hyperparameters,index=None,dtype=None):
	'''
	Initialize operators
	Args:
		data (str,array): Label or path or array of operator
		shape (iterable[int]): Shape of operator
		hyperparameters (dict): Dictionary of hyperparameters for operator
		index (int): Index to initialize operator
		dtype (data_type): Data type of operator
	Returns:
		data (array): Array of operator
	'''

	# Dimension of data
	d = min(shape)

	if data is None:
		data = (rand(shape)+ 1j*rand(shape))/sqrt(2)
		data = sp.linalg.expm(-1j*(data + data.conj().T)/2.0/d)

	elif isinstance(data,str):
		
		if data == 'random':
			data = (rand(shape)+ 1j*rand(shape))/sqrt(2)
			data = sp.linalg.expm(-1j*(data + data.conj().T)/2.0/d)

			Q,R = qr(data);
			R = diag(diag(R)/abs(diag(R)));
			data = Q.dot(R)
			assert allclose(eye(d),data.conj().T.dot(data))
			assert allclose(eye(d),data.dot(data.conj().T))

		elif data == 'rank1':
			data = diag(rand(d))
			I = eye(d)
			r = 4*index
			k = rand(shape=(r,2),bounds=[0,d],random='randint')
			for j in range(r):
				v = outer(I[k[j,0]],I[k[j,1]].T)
				c = (rand()+ 1j*rand())/sqrt(2)
				v = (v + (v.T))
				v = (c*v + c.conj()*(v.T))
				data += v
			data = sp.linalg.expm(-1j*data)

		elif data == 'gate':
			data = {
				2: array([[1,0,0,0],
						   [0,1,0,0],
						   [0,0,0,1],
						   [0,0,1,0]]),
				# 2: tensorprod(((1/sqrt(2)))*array(
				# 		[[[1,1],
				# 		  [1,-1]]]*2)),
				# 2: array([[1,0,0,0],
				# 		   [0,1,0,0],
				# 		   [0,0,1,0],
				# 		   [0,0,0,1]]),					   		
				# 2: tensorprod(((1/sqrt(2)))*array(
				# 		[[[1,1],
				# 		  [1,-1]],
				# 		 [[1,1],
				# 		  [1,-1]],
				# 		  ])),
				3: tensorprod(((1/sqrt(2)))*array(
						[[[1,1],
						  [1,-1]]]*3)),
				3: array([[1,0,0,0,0,0,0,0],
						   [0,1,0,0,0,0,0,0],
						   [0,0,1,0,0,0,0,0],
						   [0,0,0,1,0,0,0,0],
						   [0,0,0,0,1,0,0,0],
						   [0,0,0,0,0,1,0,0],
						   [0,0,0,0,0,0,0,1],
						   [0,0,0,0,0,0,1,0]]),
				4: tensorprod(
					array(
						[tensorprod((1/sqrt(2))*array(
							[[[1,1],
							  [1,-1]]]*2)),
						[[1,0,0,0],
						[0,1,0,0],
						[0,0,0,1],
						[0,0,1,0]]
					])
					),					
				# 4: tensorprod(((1/sqrt(2)))*array(
				# 		[[[1,1],
				# 		  [1,-1]],
				# 		 [[1,1],
				# 		  [1,-1]],
				# 		  [[1,1],
				# 		  [1,-1]],
				# 		 [[1,1],
				# 		  [1,-1]],							  
				# 		 ])),	
				# 4: tensorprod(array([[[1,0,0,0],
				# 		   [0,1,0,0],
				# 		   [0,0,0,1],
				# 		   [0,0,1,0]]]*2)),	
				# 4: tensorprod(array([[[1,0,0,0],
				# 		   [0,1,0,0],
				# 		   [0,0,1,0],
				# 		   [0,0,0,1]]]*2)),						   		 
				}.get(index)
		else:
			try:
				data = array(load(data))
			except:
				data = (rand(shape)+ 1j*rand(shape))/sqrt(2)
				data = sp.linalg.expm(-1j*(data + data.conj().T)/2.0/d)					
	else:
		data = array(data)

	if data is None:
		data = (rand(shape)+ 1j*rand(shape))/sqrt(2)
		data = sp.linalg.expm(-1j*(data + data.conj().T)/2.0/d)					

	data = data.astype(dtype=dtype)


	return data