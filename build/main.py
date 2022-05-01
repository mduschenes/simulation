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
np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.quantum import main
from src.utils import array,tensorprod,sin,cos
from src.io import load,dump

# Logging
import logging,logging.config
logger = logging.getLogger(__name__)
conf = 'config/logging.conf'
try: 
	logging.config.fileConfig(conf,disable_existing_loggers=False) 
except:
	pass
logger = logging.getLogger(__name__)


if __name__ == '__main__':
	train = 1
	method = 'user'
	# method = 'random'
	local = 'local'
	# local = 'global'

	n = 1
	m = 0
	N = 4
	M = 1000
	iterations = 20

	scale = 1#100*1e-6
	if train:
		if train == 1:
			objective = []
			iteration = []
		elif train == -1:
			iteration = load('output/iteration_%s_%s_%d.npy'%(method,local,m)).tolist()
			objective = load('output/objective_%s_%s_%d.npy'%(method,local,m)).tolist()
		for i in range(n):
			print(i)
			N = N
			D = 2
			d = 1
			L = 1
			M = M
			# T = M*(4e-6)
			T = M	
			p = 1
			local = local
			if method == 'random':
				V = array(onp.random.rand(D**N,D**N)) + 1j*array(onp.random.rand(D**N,D**N))
				V = sp.linalg.expm(-1j*(V + V.conj().T)/2.0/D**N)

				# V = np.array(onp.random.randn(D**N,D**N) + 1j*onp.random.randn(D**N,D**N))/np.sqrt(2);
				# [Q,R] = np.linalg.qr(V);
				# R = np.diag(np.diag(R)/np.abs(np.diag(R)));
				# V = Q.dot(R)
				assert np.allclose(np.eye(D**N),V.conj().T.dot(V))
				assert np.allclose(np.eye(D**N),V.dot(V.conj().T))
			elif method == 'rank1':
				V = np.diag(array(onp.random.rand(D**N)))
				I = np.eye(D**N)
				r = 4*N
				k = onp.random.randint(D**N,size=(r,2))
				for j in range(r):
					v = np.outer(I[k[j,0]],I[k[j,1]].T)
					c = (onp.random.rand() + 1j*onp.random.rand())/np.sqrt(2)
					v = (v + (v.T))
					v = (c*v + np.conj(c)*(v.T))
					V += v
				V = sp.linalg.expm(-1j*V)
			elif method == 'user':
				V = {
					2: array([[1,0,0,0],
							   [0,1,0,0],
							   [0,0,0,1],
							   [0,0,1,0]]),
					# 2: array([[1,0,0,0],
					# 		   [0,1,0,0],
					# 		   [0,0,1,0],
					# 		   [0,0,0,1]]),					   		
					# 2: tensorprod(((1/np.sqrt(2)))*array(
					# 		[[[1,1],
					# 		  [1,-1]],
					# 		 [[1,1],
					# 		  [1,-1]],
					# 		  ])),
					3: tensorprod(((1/np.sqrt(2)))*array(
							[[[1,1],
							  [1,-1]],
							 [[1,1],
							  [1,-1]],
							  [[1,1],
							  [1,-1]],
							  ])),
					# 3: array([[1,0,0,0,0,0,0,0],
					# 		   [0,1,0,0,0,0,0,0],
					# 		   [0,0,1,0,0,0,0,0],
					# 		   [0,0,0,1,0,0,0,0],
					# 		   [0,0,0,0,1,0,0,0],
					# 		   [0,0,0,0,0,1,0,0],
					# 		   [0,0,0,0,0,0,0,1],
					# 		   [0,0,0,0,0,0,1,0]]),
					# 4: tensorprod(((1/np.sqrt(2)))*array(
					# 		[[[1,1],
					# 		  [1,-1]],
					# 		 [[1,1],
					# 		  [1,-1]],
					# 		  [[1,1],
					# 		  [1,-1]],
					# 		 [[1,1],
					# 		  [1,-1]],							  
					# 		 ])),	
					4: tensorprod(array([[[1,0,0,0],
							   [0,1,0,0],
							   [0,0,0,1],
							   [0,0,1,0]]]*2)),	
					# 4: tensorprod(array([[[1,0,0,0],
					# 		   [0,1,0,0],
					# 		   [0,0,1,0],
					# 		   [0,0,0,1]]]*2)),						   		 
					}.get(N)

				assert np.allclose(np.eye(D**N),V.conj().T.dot(V))
				assert np.allclose(np.eye(D**N),V.dot(V.conj().T))

			fig,ax = plt.subplots(2)
			plot0 = ax[0].imshow(V.real)
			plot1 = ax[1].imshow(V.imag)
			ax[0].set_title('Real')
			ax[1].set_title('Imag')
			plt.colorbar(plot0,ax=ax[0])
			plt.colorbar(plot1,ax=ax[1])
			fig.tight_layout()
			fig.savefig('output/V_%s_%s_%d.pdf'%(method,local,i))

			hyperparameters = {
				'model':{
					'N':N,
					'D':D,
					'd':d,
					'L':L,
					'M':M,
					'T':T,
					'p':p,
					'space':'spin',		
					'time':'linear',		
					'lattice':'square',
					'system':{
						'dtype':'complex',
						'format':'array',
						'device':'cpu',
						'verbose':'info'		
					},
				},			
				'data':{
					'operator': [['X'],['Y'],['Z'],['Z','Z']],
					'site': [['i'],['i'],['i'],['i','j']],
					'string': ['h','g','k','J'],
					'interaction': ['i','i','i','i<j'],
				},		
				'label': V,																
				'optimizer':'cg',
				'hyperparameters':{
					'iterations':iterations,
					'seed':0,#onp.random.randint(10000),		
					'interpolation':3,'smoothness':3,'init':[0,1],'random':'uniform',
					'c1':0.0001,'c2':0.9,'maxiter':50,'restart':iterations//4,'tol':1e-14,
					'bound':1e6,'alpha':5e-1,'beta':1e-1,'lambda':1*np.array([1e-6,1e-6,1e-2]),'eps':980e-3,
					},
				'track':{'log':1,'track':10,'size':0,
						 'iteration':[],'objective':[],
						 'value':[],'grad':[],'search':[],
						 'alpha':[],'beta':[],'lambda':[]
					},
				'value':None,
				'callback':{'objective': lambda parameters,hyperparameters: 
					hyperparameters['track']['objective'].append(-hyperparameters['track']['value'][-1] + constraints(parameters))
					},		
				'parameters':{
					'xy':{
						'name':'xy',
						'category':'variable',
						'locality':local,
						'parameters':None,
						'size':2,
						'group':[('h',),('g',)],
						'bounds':[0,1],
						'boundaries':{0:0,-1:0},
						'func': {
							**{group:(lambda parameters,parameter,hyperparameters: (	
								# 2*np.pi/4/(20e-6)*scale*
								1*(hyperparameters['parameters'][parameter]['bounds'][1]-hyperparameters['parameters'][parameter]['bounds'][0])*(
								cos(2*np.pi*parameters[:,hyperparameters['parameters']['xy']['slice'][group][1::2]])*parameters[:,hyperparameters['parameters']['xy']['slice'][group][0::2]])))
							for group in [('h',)]},
							**{group:(lambda parameters,parameter,hyperparameters: (
								# 2*np.pi/4/(20e-6)*scale*
								1*(hyperparameters['parameters'][parameter]['bounds'][1]-hyperparameters['parameters'][parameter]['bounds'][0])*(
								sin(2*np.pi*parameters[:,hyperparameters['parameters']['xy']['slice'][group][1::2]])*parameters[:,hyperparameters['parameters']['xy']['slice'][group][0::2]])))
							for group in [('g',)]},
						},
						'constraints': (lambda parameters,parameter,hyperparameters: (
							sum(hyperparameters['hyperparameters']['lambda'][0]*bound(hyperparameters['parameters'][parameter]['bounds'][0] - parameters[:,hyperparameters['parameters'][parameter]['slice'][group]],hyperparameters).sum()
							+hyperparameters['hyperparameters']['lambda'][1]*bound(-hyperparameters['parameters'][parameter]['bounds'][1] + parameters[:,hyperparameters['parameters'][parameter]['slice'][group]],hyperparameters).sum()
							+hyperparameters['hyperparameters']['lambda'][2]*sum(np.abs(parameters[i,hyperparameters['parameters'][parameter]['slice'][group]]-hyperparameters['parameters'][parameter]['boundaries'][i]).sum()
								for i in hyperparameters['parameters'][parameter]['boundaries']) for group in [('h',),('g',)])
							)),
					},
					'z':{
						'name':'z',						
						'category':'constant',
						'locality':'local',
						'parameters':array([
							# -2*np.pi/2*1000*scale,
							# 0*scale,
							# 2*np.pi/2*1000*scale,
							# 2*np.pi/2*500*scale,
							# *(2*np.pi/2*(2*onp.random.randint(2)-1)*1000*onp.random.rand(max(1,N-4))*scale)
							*(0.5*onp.arange(1,N+1))
							][:N]),
						'size':1,
						'group':[('k',)],
						'bounds':[-1,1],
						'boundaries':{0:None,-1:None},				
						'func': {('k',):(lambda parameters,parameter,hyperparameters: (hyperparameters['parameters'][parameter]['parameters'])),},
						'constraints': (lambda parameters,parameter,hyperparameters: (0)),	
					},
					'zz':{
						'name':'zz',						
						'category':'constant',
						'locality':'local',
						'parameters':array([
							# 2*np.pi/4*72.4*scale,
							# -2*np.pi/4*130*scale,
							# 2*np.pi/4*50.0*scale,
							# 2*np.pi/4*80.0*scale,
							# 2*np.pi/4*20.0*scale,
							# 2*np.pi/4*200.0*scale,
							# *(2*np.pi/4*(2*onp.random.randint(2)-1)*200*onp.random.rand(max(1,N**2))*scale)
							*(0.1*onp.arange(1,(N*(N-1))//2+1))							
							][:(N*(N-1))//2]),
						'size':1,
						'group':[('J',)],									
						'bounds':[-1,1],
						'boundaries':{0:None,-1:None},				
						'func': {('J',):(lambda parameters,parameter,hyperparameters: (hyperparameters['parameters'][parameter]['parameters'])),},						
						'constraints': (lambda parameters,parameter,hyperparameters: (0)),	

					},
				},
				}
			

			main(i,hyperparameters)

			iteration.append(hyperparameters['track']['iteration'][:])
			objective.append(hyperparameters['track']['objective'][:])

			dump(onp.array(iteration),'output/iteration_%s_%s_%d.npy'%(method,local,m+1+i))
			dump(onp.array(objective),'output/objective_%s_%s_%d.npy'%(method,local,m+1+i))

		iteration = onp.array(iteration)
		objective = onp.array(objective)

	else:

		iteration = load('output/iteration_%s_%s_%d.npy'%(method,local,m+n))
		objective = load('output/objective_%s_%s_%d.npy'%(method,local,m+n))

	n = min(len(iteration),len(objective))	
	slices = slice(1,None)
	iteration = iteration.mean(0)
	objective,error = objective.mean(0),objective.std(0)



	x,y,yerr = iteration[slices],objective[slices],error[slices]
	config = 'config/plot.mplstyle'

	with matplotlib.style.context(config):

		fig,ax = plt.subplots()


		ax.set_yscale(value='log',base=10)

		ax.set_xlabel(r'$\textrm{Iteration}$')
		ax.set_ylabel(r'$\textrm{Fidelity}$')
		# ax.set_ylim(6e-1,1.2e0)
		ax.set_ylim(1e-1,1e0)
		# ax.set_yticks([80e-2,85e-2,90e-2,95e-2,100e-2])
		# ax.set_yticks([10e-2,20e-2,30e-2,40e-2,50e-2])
		ax.tick_params(axis='y',which='major')
		ax.tick_params(axis='y',which='minor')
		ax.tick_params(axis='x',which='major')
		ax.tick_params(axis='x',which='minor')
		ax.grid(True)

		# ax.plot(x,y,'--o')
		ax.errorbar(x,y,yerr,fmt='--o',ecolor='k',elinewidth=1,capsize=1)


		fig.set_size_inches(6,6)
		fig.subplots_adjust()
		fig.tight_layout()
		fig.savefig('output/fidelity_%s_%s_%d_%d_%d.pdf'%(method,local,n,N,M))
