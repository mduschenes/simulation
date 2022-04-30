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
PATHS = ['',".."]
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.optimize import Optimizer
from src.utils import jit,gradient
from src.utils import array,ones,zeros,arange,rand
from src.utils import matmul,tensordot,tensorprod,multi_tensorprod,trace,multi_expm,broadcast_to
from src.utils import maximum,minimum,abs,cos,sin,heaviside,sigmoid,inner,norm,interpolate

# Logging
import logging,logging.config
logger = logging.getLogger(__name__)
conf = 'config/logging.conf'
try: 
	logging.config.fileConfig(conf,disable_existing_loggers=False) 
except:
	pass
logger = logging.getLogger(__name__)


@jit
def unitary(x,A,I,string):
	'''
	x has shape (k,)
	A have shape (m,n,n)
	I has shape (n,n)
	Perform matrix exponentials of e^{-iA} as product of Euler form of exponentials of k arrays of shape (n,n)
	'''

	# c,s = cos(x)[:,None,None],sin(x)[:,None,None]
	# U = c*I - 1j*s*A
	# U = multi_matmul(U)

	U = multi_expm(x,A,I,string)

	return U

def distance(a,b):
	return 1-norm(a-b,axis=None,ord=2)/a.shape[0]
	# return 1-(np.real(trace((a-b).conj().T.dot(a-b))/a.size)/2 - np.imag(trace((a-b).conj().T.dot(a-b))/a.size)/2)/2
	# return 2*np.sqrt(1-np.abs(np.linalg.eigvals(a.dot(b))[0])**2)


def trotter(a,p):
	a = broadcast_to([v for u in [a[::i] for i in [1,-1,1,-1][:p]] for v in u],(p*a.shape[0],*a.shape[1:]))
	return a

# @jit
def bound(a,hyperparameters):
	# return 1/(1+np.exp(-eps*a))
	return sigmoid(a,hyperparameters['hyperparameters']['bound'])

def plot(x,y,**kwargs):
	if not all(isinstance(u,(tuple)) for u in [x,y]):
		x,y = [[x],],[[y],]
	k,n = len(x),min(len(u) for u in x)
	config = kwargs.get('config','config/plot.mplstyle')
	path = kwargs.get('path','output/plot.pdf')
	size = kwargs.get('size',(12,12))
	label = kwargs.get('label',[[None]*n]*k)
	legend = kwargs.get('legend',{'loc':(1.1,0.5)})
	xlabel = kwargs.get('xlabel',[[r'$\textrm{Time}$']*n]*k)
	ylabel = kwargs.get('ylabel',[[r'$\textrm{Amplitude}$']*n]*k)
	xscale = kwargs.get('xscale','linear')
	yscale = kwargs.get('yscale','linear')
	xlim = kwargs.get('xlim',[[None,None]]*n)
	ylim = kwargs.get('ylim',[[None,None]]*n)
	fig,ax = kwargs.get('fig',None),kwargs.get('ax',None)
	with matplotlib.style.context(config):
		if fig is None or ax is None:
			fig,ax = plt.subplots(n)
			ax = [ax] if n==1 else ax
		for i in range(n):
			for j in range(k):
				ax[i].plot(x[j][i],y[j][i],label=label[j][i])
			ax[i].set_xlabel(xlabel=xlabel[j][i])
			ax[i].set_ylabel(ylabel=ylabel[j][i])
			ax[i].set_xscale(value=xscale)
			ax[i].set_yscale(value=yscale)
			ax[i].set_xlim(xmin=xlim[i][0],xmax=xlim[i][1])
			ax[i].set_ylim(ymin=ylim[i][0],ymax=ylim[i][1])
			ax[i].grid(True)
			if label[j][i] is not None:
				ax[i].legend(**legend)
		fig.set_size_inches(*size)
		fig.subplots_adjust()
		fig.tight_layout()
		fig.savefig(path)

	return fig,ax



class Hamiltonian(object):

	def __init__(self,N,D=2,d=1,M=1,T=1.0,p=1,
		operator=[['X'],['Y'],['Z'],['Z','Z']],
		site=['i','i','i','i<j'],
		string=['h','g','k','J'],
		hyperparameters={}):
		'''
		Assemble operator data of k operators
		for N sites, D dimensional sites, d spatial dimensions
		for M time steps over time T, with trotterization of U = e^{-iA} of order p

		k Operators consist of sums of local terms across lattice, 
		with type of interaction sites, and k parameters x:
		data = sum_{l}^{k}(A_l) = sum_{l}^{k} (x_l sum_{s}^{ql} O_{ls})

		i.e) data = sum_{l}^{k}(A_l) = J*sum_{i<j}(Z_i*Z_j) + h*sum_{i}(X_i) + g*sum_{h}(Y_i)
			 with k=3 parameters {J,h,g}, and q_l = {N(N-1)/2, N, N}
		
		Resulting data is a list of shape (k,q_l,n,n), where q_l is the number of terms in each A_l
		and n is the space dimension D^N

		Resulting U = prod_{m}^{M} U_m, where U_m = e^{-iA} 
		will be a matrix product of M*sum_{l}^{k}(q_l) matrices

		Local basis operators all satisfy Euler identity and so identity I will also be kept for
		e^{-ixO} = cos(x)I - i sin(x) O

		Trotter error is p:
		p = 0: e^{iA} = e^{-i sum_{l}^{k}(A_l)}
		p = 1: e^{iA} = prod_{l}^{k} e^{-iA_l}
		p = 2: e^{iA} = prod_{l}^{k} e^{-iA_l/2} prod_{k}^{l} e^{-iA_l/2}

		From the Euler identity each e^{-iA_l} = prod_{s}^{q_l} (cos(x_l)I - i sin(x_l) O_{ls})

		There are therefore M*k parameters, and p*M*q matrix multiplications, where q = sum_{l}^{k}(q_l)
		some of which can be avoided if O_{ls} are strictly commuting and diagonal

		Sets up hyperparameters based on data
		'''

		self.__setup__(N,D,d,M,T,p,operator,site,string,hyperparameters)

		return

	def __setup__(self,N,D=2,d=1,M=1,T=1.0,p=1,
		operator=[['X'],['Y'],['Z'],['Z','Z']],
		site=['i','i','i','i<j'],
		string=['h','g','k','J'],
		hyperparameters={}):
		# Single site basis operators
		basis = {
			'I':array([[1,0],[0,1]]),
			'X':array([[0,1],[1,0]]),
			'Y':array([[0,-1j],[1j,0]]),
			'Z':array([[1,0],[0,-1]]),
		}

		# Basis operators that are diagonal
		diagonal = ['I','Z']

		# Site interactions on lattice
		lattice = {
			'i':[[i] for i in range(N)],
			'i<j':[[i,j] for i in range(N) for j in range(N) if i<j],
			}

		# Size of space
		n = D**N

		# Number of operators
		k = min([len(i) for i in [operator,site,string]])

		# Get identity operator I, to be maintained with same shape of data for Euler identities
		# with minimal redundant copying of data
		I = 'I'
		identity = [I]*N

		# Get basis operator string, operator, site
		strings = [string[l] for l in range(k) for i in lattice[site[l]]]
		operators = [[operator[l][i.index(j)] if j in i else I for j in range(N)] for l in range(k) for i in lattice[site[l]]]
		sites = [[j for j in i] for l in range(k) for i in lattice[site[l]]]


		# Form (size,n,n) shape operator from local strings for each data term
		data = array([multi_tensorprod([basis[j] for j in i]) for i in operators])
		identity = multi_tensorprod([basis[i] for i in identity])

		# Get size of data
		size = len(data)

		# Get Trotterized order of p copies of data for products of data
		data = trotter(data,p)

		# Reshape data to shape (p*size,n,n)
		data = data.reshape((p*size,n,n))

		# Get shape of parameters
		shape = (M,size)


		# Update indices of parameters within data and slices of parameters within parameters
		categories = list(set([hyperparameters['parameters'][parameter]['category'] for parameter in hyperparameters['parameters']]))
		indices = {category:{} for category in categories}
		slices = {category:{} for category in categories}
		for parameter in hyperparameters['parameters']:

			category = hyperparameters['parameters'][parameter]['category']
			locality = hyperparameters['parameters'][parameter]['locality']


			n = max([-1,*[slices[category][group][-1] for group in slices[category]]])+1
			for group in hyperparameters['parameters'][parameter]['group']:
				indices[category][group] = [i for i,s in enumerate(strings) 
					if any(g in group for g in [s,'_'.join([s,''.join(['%d'%j for j in sites[i]])])])] 
			
				m = len(indices[category][group]) if hyperparameters['parameters'][parameter]['locality'] in ['local'] else 1
				s = hyperparameters['parameters'][parameter]['size'] if hyperparameters['parameters'][parameter].get('size') is not None else 1
				slices[category][group] = [n+i*s+j for i in range(m) for j in range(s)]

			hyperparameters['parameters'][parameter]['index'] = {group: [j for j in indices[category][group]] for group in indices[category] if group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['slice'] = {group: [j for j in slices[category][group]] for group in slices[category]  if group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['site'] =  {group: [sites[j] for j in indices[category][group]] for group in indices[category]  if group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['string'] = {group: ['_'.join([strings[j],''.join(['%d'%(k) for k in sites[j]]),''.join(operators[j])]) for j in indices[category][group]] for group in indices[category]  if group in hyperparameters['parameters'][parameter]['group']}

		# Update shape of categories
		shapes = {
			category: (shape[0],
						sum([
							len(							
							set([i 
							for group in hyperparameters['parameters'][parameter]['slice']
							for i in hyperparameters['parameters'][parameter]['slice'][group]])) 
							for parameter in hyperparameters['parameters'] 
					   	   if hyperparameters['parameters'][parameter]['category'] == category]),
					   *shape[2:])
			for category in categories
		}


		# Update hyperparameters
		hyperparameters['data'] = data
		hyperparameters['identity'] = identity
		hyperparameters['size'] = size
		hyperparameters['shape'] = shape
		hyperparameters['shapes'] = shapes
		hyperparameters['string'] = strings
		hyperparameters['site'] = sites
		hyperparameters['operator'] = operators
		hyperparameters['N'] = N
		hyperparameters['M'] = M
		hyperparameters['D'] = D
		hyperparameters['d'] = d
		hyperparameters['n'] = n
		hyperparameters['p'] = p
		hyperparameters['coefficients'] = T/M

		return


	def __parameters__(self,hyperparameters):
		'''
		Initialize parameters and labels
		'''

		# Update hyperparameters

		# labels are hermitian conjugate of target matrix
		hyperparameters['value'] = zeros(hyperparameters['shape'])
		# hyperparameters['label'] = hyperparameters['label'].conj().T


		# Initialize parameters

		# Get shape of parameters of different category
		category = 'variable'
		shape = hyperparameters['shapes'][category]

		# parameters are initialized as interpolated random values between bounds
		factor = min(shape[0],hyperparameters['hyperparameters']['smoothness'])
		shape_interp = (shape[0]//factor + 1,*shape[1:])
		
		pts_interp = factor*arange(shape_interp[0])
		pts = arange(shape[0])
		
		key = jax.random.PRNGKey(hyperparameters['hyperparameters']['seed'])

		parameters_interp = zeros(shape_interp)

		parameters = zeros(shape)

		for parameter in hyperparameters['parameters']:
			if hyperparameters['parameters'][parameter]['category'] == category:
				key,subkey = jax.random.split(key)
				bounds = hyperparameters['parameters'][parameter]['bounds']
				for group in hyperparameters['parameters'][parameter]['slice']:
					parameters_interp = parameters_interp.at[:,hyperparameters['parameters'][parameter]['slice'][group]].set(rand(
						key,(shape_interp[0],len(hyperparameters['parameters'][parameter]['slice'][group]),*shape_interp[2:]),
						bounds=[bounds[0] + (bounds[1]-bounds[0])*hyperparameters['hyperparameters']['init'][0],bounds[1]**hyperparameters['hyperparameters']['init'][1]],
						random=hyperparameters['hyperparameters']['random']
						)
					)

			for i in hyperparameters['parameters'][parameter]['boundaries']:
				if hyperparameters['parameters'][parameter]['boundaries'][i] is not None and i < shape_interp[0]:
					parameters_interp = parameters_interp.at[i,:].set(hyperparameters['parameters'][parameter]['boundaries'][i])

		for parameter in hyperparameters['parameters']:
			if hyperparameters['parameters'][parameter]['category'] == category:

				for group in hyperparameters['parameters'][parameter]['slice']:
					hyperparameters['parameters'][parameter]['parameters'] = minimum(
						hyperparameters['parameters'][parameter]['bounds'][1], 
						maximum(
						hyperparameters['parameters'][parameter]['bounds'][0],
						interpolate(
						pts_interp,parameters_interp[:,hyperparameters['parameters'][parameter]['slice'][group]],
						pts,hyperparameters['hyperparameters']['interpolation'])
						)
						)

			for i in hyperparameters['parameters'][parameter]['boundaries']:
				if hyperparameters['parameters'][parameter]['boundaries'][i] is not None and i < shape[0]:
					hyperparameters['parameters'][parameter]['parameters'] = hyperparameters['parameters'][parameter]['parameters'].at[i,:].set(hyperparameters['parameters'][parameter]['boundaries'][i])

			if hyperparameters['parameters'][parameter]['category'] == category:
				for group in hyperparameters['parameters'][parameter]['slice']:
					parameters = parameters.at[:,hyperparameters['parameters'][parameter]['slice'][group]].set(hyperparameters['parameters'][parameter]['parameters'])


		# Get reshaped parameters

		parameters = parameters.ravel()

		return parameters


def plot_parameters(parameters,hyperparameters,**kwargs):
	'''
	Plot Parameters
	Args:
		parameters (array): Parameters
		hyperparameters (dict): Hyperparameters
	Returns:
		fig (object): Matplotlib figure object
		ax (object): Matplotlib axes object
	'''
	category = 'variable'

	localities = [hyperparameters['parameters'][parameter]['locality']
		for parameter in hyperparameters['parameters']
		if hyperparameters['parameters'][parameter]['category'] == category
		]
	localities = list(sorted(list(set(localities)),key = lambda i: localities.index(i)))

	# slices = [i
	# 	for parameter in hyperparameters['parameters']
	# 	for group in hyperparameters['parameters'][parameter]['group']
	# 	for i in hyperparameters['parameters'][parameter]['slice'][group]
	# 	if hyperparameters['parameters'][parameter]['category'] == category
	# 	]
	# slices = list(sorted(list(set(slices)),key = lambda i: slices.index(i)))

	groups = [group 
		for parameter in hyperparameters['parameters']
		for i,group in enumerate(hyperparameters['parameters'][parameter]['group'])
		if ((hyperparameters['parameters'][parameter]['category'] == category)
		and (hyperparameters['parameters'][parameter]['slice'][group] not in 
			[hyperparameters['parameters'][parameter]['slice'][g] 
			for g in hyperparameters['parameters'][parameter]['group'][:i]]))
		]
	groups = list(sorted(list(set(groups)),key = lambda i: groups.index(i)))


	print(groups)

	print([hyperparameters['parameters'][parameter]['site'][group] 
	for group in groups
	for parameter in hyperparameters['parameters'] 
	if ((hyperparameters['parameters'][parameter]['category'] == category) 
	and (i in hyperparameters['parameters'][parameter]['slice'][group]))])

	exit()

	shape = hyperparameters['shapes'][category]

	parameters = parameters.reshape(shape)

	fig,ax = plot(x=([arange(shape[0]) for i in range(shape[1])],),y=([u for u in parameters.T],),		
		label=[[r'${%s}_{%s}^{\textrm{%s}}$'%(
				[r'\alpha',r'\phi'][i%2],'',kwargs.get('string',''))
				for i in range(shape[1])]],
		xlabel=[[r'$\textrm{Time}$' for i in range(shape[1])]],
		ylabel=[[[r'$\textrm{Amplitude}$',r'$\textrm{Phase}$'][i%2] for i in range(shape[1])]],
		# ylim=[[hyperparameters['trainable'][category]['bounds'],
		yscale='linear',
		legend = {'loc':kwargs.get('loc',(0.78,0.1))},
		size= kwargs.get('size',(20,20)),
		fig=kwargs.get('fig'),ax=kwargs.get('ax'),
		path=kwargs.get('path','output/parameters.pdf')
		)

	return fig,ax


def main(index,N,D=2,d=1,M=1,T=1.0,p=1,
		operator=[['X'],['Y'],['Z','Z']],
		site=['i','i','i<j'],
		string=['h','g','J'],
		hyperparameters={}):


	def decorator(hyperparameters):
		def wrapper(func):
			@functools.wraps
			def wrapped(parameters):
				return func(parameters,hyperparameters)
			return wrapped
		return wrapper




	def params(parameters,hyperparameters):
		'''
		Set parameters
		'''

		# Set all parameters
		
		category = 'variable'
		value = hyperparameters['value']
		shape = hyperparameters['shapes'][category]

		parameters = parameters.reshape(shape)

		for parameter in hyperparameters['parameters']:

			if hyperparameters['parameters'][parameter]['category'] is category:
				for group in hyperparameters['parameters'][parameter]['slice']:
					hyperparameters['parameters'][parameter]['parameters'] = parameters[:,hyperparameters['parameters'][parameter]['slice'][group]]

				value = value.at[:,hyperparameters['parameters'][parameter]['index'][group]].set(
						hyperparameters['parameters'][parameter]['func'][group](parameters,parameter,hyperparameters))


		# Get Trotterized order of copies of parameters
		p = hyperparameters['p']
		parameters = trotter(value.T,p).T/p

		# Get coefficients (time step delta)
		coefficients = hyperparameters['coefficients']		
		parameters *= coefficients

		# Get reshaped parameters
		parameters = parameters.ravel()

		return parameters

	def constraints(parameters,hyperparameters):
		'''
		Set constraints
		'''
		penalty = 0

		category = 'variable'
		shape = hyperparameters['shapes'][category]

		parameters = parameters.reshape(shape)

		for parameter in hyperparameters['parameters']:
			penalty += hyperparameters['parameters'][parameter]['constraints'](parameters,parameter,hyperparameters)
		return penalty


	def func(parameters,hyperparameters):
		U = unitary(params(parameters),hyperparameters['data'],hyperparameters['identity'],hyperparameters['string'])
		return U


	def loss(parameters,hyperparameters):
		# return -(abs(inner(func(parameters),hyperparameters['label'])/hyperparameters['n'])**2)
		return -distance(func(parameters),hyperparameters['label'])
		# return -norm((func(parameters)-hyperparameters['label'].conj().T)**2/hyperparameters['n'],axis=None,ord=2)


	def loss_constraints(parameters,hyperparameters):
		return loss(parameters) + constraints(parameters)


	def step(iteration,state,optimizer,hyperparameters):

		state = optimizer.opt_update(iteration, state, hyperparameters)

		return state

	def objective(parameters,hyperparameters):
		return -hyperparameters['track']['value'][-1] + constraints(parameters)

	def log(parameters,hyperparameters):
		i = hyperparameters['track']['iteration'][-1]
		U = func(parameters)
		V = hyperparameters['label']
		UH = U.conj().T
		VH = V.conj().T


		# print(
		# 	'func',i,'\n',
		# 	'U-V',U-V,'\n',
		# 	'UV',U.dot(VH),'\n',
		# 	'UU',U.dot(UH),'\n',
		# 	'VV',V.dot(VH),'\n',
		# )

		return

	hamiltonian = Hamiltonian(N,D,d,M,T,p,operator,site,string,hyperparameters)
	parameters = hamiltonian.__parameters__(hyperparameters)


	func = jit(partial(func,hyperparameters=hyperparameters))
	loss = jit(partial(loss,hyperparameters=hyperparameters))
	params = jit(partial(params,hyperparameters=hyperparameters))
	constraints = jit(partial(constraints,hyperparameters=hyperparameters))
	loss_constraints = jit(partial(loss_constraints,hyperparameters=hyperparameters))

	hyperparameters['callback'] = {
		'objective': (lambda parameters,hyperparameters: 
			hyperparameters['track']['objective'].append(objective(parameters,hyperparameters))),
		'log': (lambda parameters,hyperparameters: log(parameters,hyperparameters)),
	}

	optimizer = Optimizer(func=loss_constraints,hyperparameters=hyperparameters)

	state = optimizer.opt_init(parameters)


	fig,ax = plot_parameters(optimizer.get_params(state),hyperparameters,
							 string='%d'%(-1+1),path='output/parameters_%d.pdf'%(index))

	for iteration in range(hyperparameters['hyperparameters']['iterations']):
		state = step(iteration,state,optimizer,hyperparameters)

	fig,ax = plot_parameters(optimizer.get_params(state),hyperparameters,
							 fig=fig,ax=ax,
							 string='%d'%(hyperparameters['track']['iteration'][-1]),
							 path='output/parameters_%d.pdf'%(index))

	plot(hyperparameters['track']['iteration'][::2],[v for v in hyperparameters['track']['objective']][::2],
		xlabel=[[r'$\textrm{Iteration}$']],
		ylabel=[[r'$\textrm{Fidelity}$']],
		size=(6,6),
		yscale='log',
		ylim=[[1e-1,1e0]],
		path='output/fidelity_%d.pdf'%(index)
		)


	plt.close();

	return