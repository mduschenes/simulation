#!/usr/bin/env python

# Import python modules
import os,sys
import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp
import jax.scipy as jsp

# Logging
import logging,logging.config
conf = "config/logging.conf"
try: 
	logging.config.fileConfig(conf,disable_existing_loggers=False) 
except: 
	pass
logger = logging.getLogger(__name__)


# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['',"..","../..","../../lib"]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from lib.utils.io import setup
from lib.utils.utils import dictionary,objs,array,asarray,ones,zeros,arange,eye,identity,hadamard,phasehadamard,cnot,toffoli
from lib.utils.utils import jit,grad,finitegrad,norm,expm,tensordot,tensorprod,multi_matmul,inner,abs,repeat,allclose,isclose,PRNG,gradinner,prod
from lib.utils.utils import trotter,trottergrad,cos,sin
from lib.utils.parallel import Pooler,Parallelize
from lib.utils.plot import plot

from src.quantum import Circuit,Gate,Operator,Unitary,Hamiltonian
from src.quantum import Pauli,X,Y,Z,I,Identity,Hadamard,PhaseHadamard,Cnot,Toffoli

from src.optimize import Model,Optimize,Gradient

def main(args):

	defaults = [
		{
			"io":{
				"verbose":"Info",
				"path":""
			},
			"logging":{
				"version": 1,
				"disable_existing_loggers":0,
				"formatters": {"file": {"format": "%(asctime)s :: %(message)s","datefmt":"%Y-%m-%d %H:%M:%S"},
							   "stdout": {"format": "%(message)s","datefmt":"%Y-%m-%d %H:%M:%S"}},
				"handlers": {"file": {"class": "logging.FileHandler",
										 "filename": "log.log",
										 "formatter": "file",
										 "level": "INFO",
										 "mode": "w"},
							"stdout": {"class": "logging.StreamHandler",
										  "formatter": "stdout",
										  "level": "INFO",
										  "stream": "ext://sys.stdout"}},
				"root": {"handlers": ["stdout", "file"], "level": "DEBUG"}
			},
			"plotting":{
				"names":["energy","order"],
				"path":"figures",
				"verbose":"info"
			},
			"model":{
				"N":3,
				"D":2,
				"d":1,
				"L":1,
				"M":10,
				"T":1,
				"p":1,
				"space":"spin",		
				"time":"linear",		
				"lattice":"square",
				"system":{
					"dtype":"complex",
					"format":"array",
					"device":"cpu",
					"verbose":"info"		
				},								
			},
			"gate":[
					[{"operator":["Z","Z"],"site":["i","j"],"string":"J","interaction":"<ij>","coefficient":1e-1}],
					[
					 {"operator":["X"],"site":["i"],"string":"g","interaction":"i","coefficient":1},
					 # {"operator":["Y"],"site":["i"],"string":"h","interaction":"i","coefficient":1},
					 # {"operator":["X","X"],"site":["i","j"],"string":"l","interaction":"i<j","coefficient":1}
					 ],
					[{"operator":["Y"],"site":["i"],"string":"h","interaction":"i","coefficient":1}],					
					# [{"operator":["Z"],"site":[1],"string":"k","interaction":"i","coefficient":1}],
				],				
			"target":"data/T.txt",
			"optimization":{
				"optimizer":"gd",
				"hyperparameters":{
					"iterations":1000,
					"alpha":1e-2,
					"eps":1e-3,
					"bounds":[0,2*np.pi],
					"constraints":[],
					"constants":{0:1},
					"boundaries":{0:0,-1:0},
					"shape":(),
					"interpolation":3,
					"y":[],
					},
				"system":{
					"dtype":"complex",
					"format":"array",
					"device":"cpu",
					"verbose":"info"		
					},					
			},
			"results":{
				"Sz_mean":[],
				"energy": []
			},
			"parameters":{
				"J": {"start":-2.0,"stop":2.0,"length":5,"type":"range"},
				"g": [1]
			}
		}
		]

	params = setup(args,defaults)

	log = lambda msg: logger.log(params['io']['verbose'],msg)

	theta = np.pi/12
	eps = 1e-8

	H = Hamiltonian(params['gate'],**params['model'])
	U = Unitary(H,**params['model'])
	C = Circuit(U,**params['model'])
	V = Gate(params['target'])

	optimizer = params['optimization']['optimizer']
	hyperparameters = params['optimization']['hyperparameters']
	system = params['optimization']['system']

	def func(parameters,labels,hyperparameters):

		shape = hyperparameters['shape']
		constants = hyperparameters['constants']

		x = zeros(shape)

		for d in range(x.ndim-parameters.ndim):
			parameters = parameters[...,None]
		
		k = 0
		for j,i in enumerate(range(shape[1])):
			if i in constants:
				p = constants[i]
				k += 1
			else:
				p = parameters[:,i-k]
			x = x.at[:,i].set(p)

		return C(-1j*x)

	def loss(x,y,hyperparameters):
		return -inner(func(x,y,hyperparameters),y)/y.shape[0]


	shape = C.shape
	hyperparameters['shape'] = shape
	x = zeros((shape[0],shape[1]-1))
	y = V().conj().T

	model = Model(optimizer,func,loss,hyperparameters,system)
	model.train(x,y,hyperparameters)
	exit()

	# import jax
	# import jax.numpy as jnp
	# import jax.scipy as jsp
	
	# k = 3
	# n = 10
	# p = 1
	# x = jnp.array([jnp.array(np.random.rand(1)) for i in range(k)]).reshape(k)
	# a = jnp.array([jnp.array(np.random.rand(n**2)) for i in range(k)]).reshape(k,n,n)
	# u = lambda x,p=p: jnp.array([jsp.linalg.expm(x[i]*a[i]/p) for i in range(k)])

	# v = lambda x,p=p: trotter(u(x,p),p).trace()
	# g = trottergrad(u(x),a,p).trace(0,1,2)
	# print('------------')	
	# h = grad(v,x)
	# print('------------')	
	# f = finitegrad(v,x,eps=1e-7)
	# print(g)
	# print(h)
	# print(f)

	# print('------------')

	# g = trottergrad(u(x),a,p).trace(0,1,2)
	# h = grad(v,x)
	# f = finitegrad(v,x,eps=1e-7)
	# print(g)
	# print(h)
	# print(f)

	# print(allclose(g,h),allclose(f,h),allclose(g,f))

	# exit()


	# Ha = Hadamard()
	# Pa = PhaseHadamard()
	# CN = Cnot()
	To = Toffoli()
	# h,p,c,t = Ha(),Pa(),CN(),To()
	t = To()
	_x,_y,_z,_i = X(),Y(),Z(),I()
	x,y,z,i = _x(),_y(),_z(),_i()
	# print(t,To,To.site,To.string)
	# print(x)
	# print(y)
	# print(z)
	# print(i)
	# print(h)
	# print(p)
	# print(c)


	# print(C,U,H,V)
	# print(H.shape,U.shape,C.shape,V.shape)

	# print(H,'---',U,'---',C)	
	print('1',U(-1j*theta))
	print('2',C(-1j*theta))
	print(np.allclose(U(-1j*theta),C(-1j*theta)))
	# print(inner(V(),U(theta)))
	print(grad(lambda theta: inner(V(),U(-1j*theta)))(theta))
	# print((inner(V(),U(-1j*(theta+eps))) - inner(V(),U(-1j*(theta-eps))))/(2*eps))
	# print(gradinner(V(),U(-1j*theta),H(-1j*theta),H(-1j*theta),theta,theta))
	# print(np.allclose(H(1j*theta),1j*H(theta).conj().T))
	# exit()

	zz = tensorprod([z,z])
	xi = tensorprod([x,i])
	ix = tensorprod([i,x])
	yi = tensorprod([y,i])
	iy = tensorprod([i,y])
	zi = tensorprod([z,i])

	h = [zz,xi+ix,yi+iy,zi]

	zzi = tensorprod([z,z,i])
	ziz = tensorprod([z,i,z])
	izz = tensorprod([i,z,z])
	xii = tensorprod([x,i,i])
	ixi = tensorprod([i,x,i])
	iix = tensorprod([i,i,x])
	yii = tensorprod([y,i,i])
	iyi = tensorprod([i,y,i])
	iiy = tensorprod([i,i,y])	
	zii = tensorprod([i,z,i])

	# h = [zzi+ziz+izz,xii+ixi+iix,yii+iyi+iiy,zii]
	h = [zzi+ziz+izz,xii+ixi+iix,yii+iyi+iiy]

	theta = -1j*np.pi/12

	u = H.expm(theta)
	v = multi_matmul(array([v.expm(theta) for v in H.data]))
	q = multi_matmul(array([expm(theta*i) for i in h]))
	s = U(theta)

	print([v for v in H.data],H,U)
	print(np.allclose(u,v),np.allclose(u,q),np.allclose(v,s),np.allclose(q,s))


	for j,u in enumerate(H.data):
		print(u,np.allclose(u(),h[j]))

	exit()



	# def func(x):
	# 	x[0] = 50

	# # @jit
	# # def loss(x,V,U,d):
	# # 	value = -jnp.absolute(jnp.trace(jnp.matmul(jnp.conjugate(V),U)/d))**2 
	# # 	return value

	# # @jit
	# # def regularization(x,V,U,d):
	# # 	value = jnp.absolute(jnp.trace(jnp.matmul(jnp.conjugate(V),U)/d))**2 
	# # 	return 		

	# # @jit
	# # def func(x,V,U,d):
	# # 	value = loss(x,V,U,d)
	# # 	return value



	# # x = [1,2,3]
	# # y = [1,2,9]
	# # settings = {
	# # 	'fig':{
	# # 		'savefig':{'fname':'data.pdf'}},
	# # 	'ax':{
	# # 		'plot':{'marker':'o','linestyle':''},
	# # 		'set_xlabel':{'xlabel':r'$x$','fontsize':20},
	# # 		'set_ylabel':{'ylabel':r'$y$','fontsize':20}
	# # 		}
	# # 	}
	# # plot(x,y,settings)

	return

if __name__ == '__main__':
	main(sys.argv[1:])

	