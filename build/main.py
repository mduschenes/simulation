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
np.set_printoptions(linewidth=1000)#,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.quantum import run
from src.utils import jit,array,tensorprod,sin,cos,sigmoid
from src.utils import pi,e
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


def bound(a,hyperparameters):
	# return 1/(1+np.exp(-eps*a))
	return sigmoid(a,hyperparameters['hyperparameters']['bound'])

def params(parameters,hyperparameters,parameter,group):

	if group in [('x',)]:
		param = hyperparameters['parameters'][parameter]['scale']*(
			parameters[:,hyperparameters['parameters'][parameter]['slice'][group][:len(hyperparameters['parameters'][parameter]['slice'][group])//2]])

	elif group in [('y',)]:
		param = hyperparameters['parameters'][parameter]['scale']*(
			parameters[:,hyperparameters['parameters'][parameter]['slice'][group][len(hyperparameters['parameters'][parameter]['slice'][group])//2:]])		
	

	return param

def grads(parameters,hyperparameters,parameter,group):

	if group in [('x',)]:
		param = hyperparameters['parameters'][parameter]['scale']*(
			parameters[:,hyperparameters['parameters'][parameter]['slice'][group][:len(hyperparameters['parameters'][parameter]['slice'][group])//2]])

	elif group in [('y',)]:
		param = hyperparameters['parameters'][parameter]['scale']*(
			parameters[:,hyperparameters['parameters'][parameter]['slice'][group][len(hyperparameters['parameters'][parameter]['slice'][group])//2:]])		
	

	return param	


def main(args):
	nargs = len(args)

	path = args[0] if nargs>0 else None

	settings = load(path)

	# method = 'user'
	# method = 'random'
	# locality = 'locality'
	# locality = 'global'

	train = settings['sys']['train']
	method = settings['hyperparameters']['method']
	locality = settings['hyperparameters']['locality']

	realizations = settings['hyperparameters']['realizations']
	prev_realizations = 0 #settings['hyperparameters']['prev_realizations']
	
	N = settings['model']['N']
	D = settings['model']['D']
	d = settings['model']['d']
	L = settings['model']['L']	
	M = settings['model']['M']
	p = settings['model']['p']
	T = settings['model']['T']
	tau = settings['model']['tau']
	delta = settings['model']['delta']
	
	iterations = settings['hyperparameters']['iterations']
	seed = settings['hyperparameters']['seed']

	onp.random.seed(seed)

	# scale = 100*1e-6
	scale = 1/(2*pi/4/(20e-6))
	if train:
		if train == 1:
			objective = []
			iteration = []
		elif train == -1:
			pass
			# iteration = load('output/iteration_%s_%s_%d.npy'%(method,locality,prev_realizations+1)).tolist()
			# objective = load('output/objective_%s_%s_%d.npy'%(method,locality,prev_realizations+1)).tolist()
		for i in range(realizations):
			print(i)
			N = N
			D = D
			d = d
			L = L
			M = M
			T = T
			tau = tau/scale
			delta = delta
			p = p
			locality = locality
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
					3: array([[1,0,0,0,0,0,0,0],
							   [0,1,0,0,0,0,0,0],
							   [0,0,1,0,0,0,0,0],
							   [0,0,0,1,0,0,0,0],
							   [0,0,0,0,1,0,0,0],
							   [0,0,0,0,0,1,0,0],
							   [0,0,0,0,0,0,0,1],
							   [0,0,0,0,0,0,1,0]]),
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

			# fig,ax = plt.subplots(2)
			# plot0 = ax[0].imshow(V.real)
			# plot1 = ax[1].imshow(V.imag)
			# ax[0].set_title('Real')
			# ax[1].set_title('Imag')
			# plt.colorbar(plot0,ax=ax[0])
			# plt.colorbar(plot1,ax=ax[1])
			# fig.tight_layout()
			# fig.savefig('output/V_%s_%s_%d.pdf'%(method,locality,i))

			hyperparameters = {
			**{
				'sys':{
					**settings['sys'],
				},
				'model':{
					'N':N,
					'D':D,
					'd':d,
					'L':L,
					'delta':delta,
					'M':M,
					'T':T,
					'tau':tau,
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
					**settings['model']
				},			
				'data':{
					'data':{
						'x':{'operator':['X'],'site':['i'],'string':'x','interaction':'i'},
						'y':{'operator':['Y'],'site':['i'],'string':'y','interaction':'i'},
						'z':{'operator':['Z'],'site':['i'],'string':'z','interaction':'i'},
						'zz':{'operator':['Z','Z'],'site':['i','j'],'string':'zz','interaction':'i<j'},
					},
					**settings['data']
				},		
				'label': V,																
				'hyperparameters':{
					'iterations':iterations,
					'realizations':realizations,
					'locality':locality,
					'class':'random',
					'optimizer':'cg',
					'seed':111,#onp.random.randint(10000),		
					'interpolation':3,'smoothness':2,'init':[0,1],'random':'uniform',
					'c1':0.0001,'c2':0.9,'maxiter':50,'restart':iterations//4,'tol':1e-14,
					'bound':1e4,'alpha':1,'beta':1e-1,'lambda':1*np.array([1e-6,1e-6,1e-2]),'eps':980e-3,
					'line_search':1,
					'track':{
						'track':{'log':1,'track':10,'callback':1},
						'iteration':[],'objective':[],
						'value':[],'grad':[],'search':[],
						'alpha':[],'beta':[],'lambda':[],
						'parameters':[],
					},
					**settings['hyperparameters']
				},
				'value':None,
				'parameters':{
					**{parameter:{
						'name':'xy',
						'category':'variable',
						'locality':locality,
						'parameters':None,
						'size':2,
						'group':[['x',],['y',]],
						'scale':1,
						# 'scale':2*pi/4/(20e-6)*scale,
						'bounds':[-1,1],
						'boundaries':{0:0,-1:0},
						'func': {
							**{group:(lambda parameters,hyperparameters,parameter=parameter,group=group,params=params: params(parameters,hyperparameters,parameter=parameter,group=group))
								for group in [('x',),('y',)]},
						},
						'constraints': {group: (lambda parameters,hyperparameters,parameter=parameter,group=group: (
							hyperparameters['hyperparameters']['lambda'][0]*bound(
								(hyperparameters['parameters'][parameter]['bounds'][0] - 
								(parameters[:,hyperparameters['parameters'][parameter]['slice'][group][:len(hyperparameters['parameters'][parameter]['slice'][group])//2]]**2+
								 parameters[:,hyperparameters['parameters'][parameter]['slice'][group][len(hyperparameters['parameters'][parameter]['slice'][group])//2:]]**2)**(1/2)),
								hyperparameters
								).sum()
							+hyperparameters['hyperparameters']['lambda'][1]*bound(
								(-hyperparameters['parameters'][parameter]['bounds'][1] + 
								(parameters[:,hyperparameters['parameters'][parameter]['slice'][group][:len(hyperparameters['parameters'][parameter]['slice'][group])//2]]**2+
								 parameters[:,hyperparameters['parameters'][parameter]['slice'][group][len(hyperparameters['parameters'][parameter]['slice'][group])//2:]]**2)**(1/2)),
								hyperparameters
								).sum()
							+hyperparameters['hyperparameters']['lambda'][2]*sum(((
								parameters[i,hyperparameters['parameters'][parameter]['slice'][group][:len(hyperparameters['parameters'][parameter]['slice'][group])//2]]-
								hyperparameters['parameters'][parameter]['boundaries'][i]
								)**2).sum()
								for i in hyperparameters['parameters'][parameter]['boundaries']) 							
							+hyperparameters['hyperparameters']['lambda'][2]*sum(((
								parameters[i,hyperparameters['parameters'][parameter]['slice'][group][len(hyperparameters['parameters'][parameter]['slice'][group])//2:]]-
								hyperparameters['parameters'][parameter]['boundaries'][i]
								)**2).sum()							
								for i in hyperparameters['parameters'][parameter]['boundaries'])
							)) 
							for group in [('x',),('y',)]
						},
						# 'func': {
						# 	**{group:(lambda parameters,hyperparameters,parameter=parameter,group=group: (	
						# 		hyperparameters['parameters'][parameter]['scale']*
						# 		(hyperparameters['parameters'][parameter]['bounds'][1]-hyperparameters['parameters'][parameter]['bounds'][0])*(
						# 		cos(2*pi*parameters[:,hyperparameters['parameters'][parameter]['slice'][group][1::2]])*parameters[:,hyperparameters['parameters'][parameter]['slice'][group][0::2]]) + (
						# 		hyperparameters['parameters'][parameter]['bounds'][0])								
						# 		))
						# 	for group in [('x',)]},
						# 	**{group:(lambda parameters,hyperparameters,parameter=parameter,group=group: (
						# 		hyperparameters['parameters'][parameter]['scale']*
						# 		(hyperparameters['parameters'][parameter]['bounds'][1]-hyperparameters['parameters'][parameter]['bounds'][0])*(
						# 		sin(2*pi*parameters[:,hyperparameters['parameters'][parameter]['slice'][group][1::2]])*parameters[:,hyperparameters['parameters'][parameter]['slice'][group][0::2]]) + (
						# 		hyperparameters['parameters'][parameter]['bounds'][0])								
						# 		))
						# 	for group in [('y',)]},
						# },
						# 'constraints': {group: (lambda parameters,hyperparameters,parameter=parameter,group=group: (
						# 	hyperparameters['hyperparameters']['lambda'][0]*bound(hyperparameters['parameters'][parameter]['bounds'][0] - parameters[:,hyperparameters['parameters'][parameter]['slice'][group]],hyperparameters).sum()
						# 	+hyperparameters['hyperparameters']['lambda'][1]*bound(-hyperparameters['parameters'][parameter]['bounds'][1] + parameters[:,hyperparameters['parameters'][parameter]['slice'][group]],hyperparameters).sum()
						# 	+hyperparameters['hyperparameters']['lambda'][2]*sum(((parameters[i,hyperparameters['parameters'][parameter]['slice'][group][0::2]]-hyperparameters['parameters'][parameter]['boundaries'][i])**2).sum()
						# 		for i in hyperparameters['parameters'][parameter]['boundaries']))) for group in [('x',),('y',)]
						# },
						'grad': {
							**{group:(lambda parameters,derivative,hyperparameters,parameter=parameter,group=group: (	
								hyperparameters['parameters'][parameter]['scale']*
								(hyperparameters['parameters'][parameter]['bounds'][1]-hyperparameters['parameters'][parameter]['bounds'][0])*(
								parameters[:,hyperparameters['parameters'][parameter]['slice'][group][:len(hyperparameters['parameters'][parameter]['slice'][group])//2]]) + (
								0*hyperparameters['parameters'][parameter]['bounds'][0])								
								))
							for group in [('x',)]},
							**{group:(lambda parameters,hyperparameters,parameter=parameter,group=group: (
								hyperparameters['parameters'][parameter]['scale']*
								(hyperparameters['parameters'][parameter]['bounds'][1]-hyperparameters['parameters'][parameter]['bounds'][0])*(
								parameters[:,hyperparameters['parameters'][parameter]['slice'][group][len(hyperparameters['parameters'][parameter]['slice'][group])//2:]]) + (
								0*hyperparameters['parameters'][parameter]['bounds'][0])								
								))
							for group in [('y',)]},
						},						
					} 
					for parameter in ['xy']},
					**{parameter:{
						'name':'z',						
						'category':'constant',
						'locality':'local',
						'parameters':array([
							-1, 
							0,
							1,
							1/2,
							*(2*onp.random.rand(N)-1)
							][:N]),
						'size':1,
						'group':[('z',)],
						'scale':1,
						# 'scale':2*pi/2*1000*scale,						
						'bounds':[-1,1],
						'boundaries':{0:None,-1:None},				
						'func': {group:(lambda parameters,hyperparameters,parameter=parameter,group=group: (
							hyperparameters['parameters'][parameter]['scale']*							
							hyperparameters['parameters'][parameter]['parameters']
							)) 
							for group in [('z',)]},
						'constraints': {group: (lambda parameters,hyperparameters,parameter=parameter,group=group: (0)) for group in [('z',)]},	
					}
					for parameter in ['z']},
					**{parameter:{
						'name':'zz',						
						'category':'constant',
						'locality':'local',
						'parameters':array([
							0.0724,
							-0.130,
							0.05,
							0.08,
							0.02,
							0.2,
							*(2*onp.random.rand(N**2)-1)							
							][:(N*(N-1))//2]),
						'size':1,
						'group':[('zz',)],				
						'scale':1,
						# 'scale':2*pi/4*1000*scale,
						'bounds':[-1,1],
						'boundaries':{0:None,-1:None},				
						'func': {group:(lambda parameters,hyperparameters,parameter=parameter,group=group: (
							hyperparameters['parameters'][parameter]['scale']*
							hyperparameters['parameters'][parameter]['parameters']
							)) 
							for group in [('zz',)]},						
						'constraints': {group: (lambda parameters,hyperparameters,parameter=parameter,group=group: (0)) for group in [('zz',)]},	
					}
					for parameter in ['zz']},					
				},
				},
			}

			

			run(i,hyperparameters)

			iteration.append(hyperparameters['hyperparameters']['track']['iteration'][:])
			objective.append(hyperparameters['hyperparameters']['track']['objective'][:])

			# dump(onp.array(iteration),'output/iteration_%s_%s_%d.npy'%(method,locality,prev_realizations+1+i))
			# dump(onp.array(objective),'output/objective_%s_%s_%d.npy'%(method,locality,prev_realizations+1+i))

		iteration = onp.array(iteration)
		objective = onp.array(objective)

	else:
		pass
		# iteration = load('output/iteration_%s_%s_%d.npy'%(method,locality,prev_realizations+realizations))
		# objective = load('output/objective_%s_%s_%d.npy'%(method,locality,prev_realizations+realizations))

	# realizations = min(len(iteration),len(objective))	
	# slices = slice(1,None)
	# iteration = iteration.mean(0)
	# objective,error = objective.mean(0),objective.std(0)



	# x,y,yerr = iteration[slices],objective[slices],error[slices]
	# config = 'config/plot.mplstyle'

	# with matplotlib.style.context(config):

	# 	fig,ax = plt.subplots()


	# 	ax.set_yscale(value='log',base=10)

	# 	ax.set_xlabel(r'$\textrm{Iteration}$')
	# 	ax.set_ylabel(r'$\textrm{Fidelity}$')
	# 	# ax.set_ylim(6e-1,1.2e0)
	# 	ax.set_ylim(1e-1,1e0)
	# 	# ax.set_yticks([80e-2,85e-2,90e-2,95e-2,100e-2])
	# 	# ax.set_yticks([10e-2,20e-2,30e-2,40e-2,50e-2])
	# 	ax.tick_params(axis='y',which='major')
	# 	ax.tick_params(axis='y',which='minor')
	# 	ax.tick_params(axis='x',which='major')
	# 	ax.tick_params(axis='x',which='minor')
	# 	ax.grid(True)

	# 	# ax.plot(x,y,'--o')
	# 	ax.errorbar(x,y,yerr,fmt='--o',ecolor='k',elinewidth=1,capsize=1)


	# 	fig.set_size_inches(6,6)
	# 	fig.subplots_adjust()
	# 	fig.tight_layout()
	# 	fig.savefig('output/fidelity_method%s_local%s_repeats%d__iterations%d_N%d_M%d.pdf'%(method,locality,realizations,iterations,N,M))



if __name__ == '__main__':
	main(sys.argv[1:])