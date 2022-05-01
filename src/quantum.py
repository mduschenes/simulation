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
from src.utils import array,dictionary,ones,zeros,arange,rand,identity
from src.utils import matmul,tensordot,tensorprod,trace,expm,broadcast_to
from src.utils import maximum,minimum,abs,cos,sin,heaviside,sigmoid,inner,norm,interpolate,unique
from src.utils import parse

# Logging
import logging,logging.config
logger = logging.getLogger(__name__)
conf = 'config/logging.conf'
try: 
	logging.config.fileConfig(conf,disable_existing_loggers=False) 
except:
	pass
logger = logging.getLogger(__name__)


class System(dictionary):
	'''
	System attributes (dtype,format,device,seed,verbose,...)
	Args:
		dtype (str,data-type): Data type of class
		format (str): Format of array
		device (str): Device for computation
		seed (array,int): Seed for random number generation
		verbose (bool,str): Verbosity of class	
		args (dict,System): Additional system attributes
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,*args,**kwargs):


		updates = {
			'verbose':{
				'notset':0,'debug':10,'info':20,'warning':30,'error':40,'critical':50,
				'Notset':0,'Debug':10,'Info':20,'Warning':30,'Error':40,'Critical':50,
				'NOTSET':0,'DEBUG':10,'INFO':20,'WARNING':30,'ERROR':40,'CRITICAL':50,
				10:10,20:20,30:30,40:40,50:50,
				2:20,3:30,4:40,5:50,
				True:20,False:0,None:0,
				}
			}

		defaults = {
			'dtype':'complex',
			'format':'array',
			'device':'cpu',
			'seed':None,
			'verbose':False,
		}

		args = {k:v for a in args for k,v in ({} if a is None else a).items()}
		attrs = {**args,**kwargs}
		attrs.update({attr: defaults[attr] for attr in defaults if attrs.get(attr) is None})
		attrs.update({attr: updates.get(attr,{}).get(attrs[attr],attrs[attr]) for attr in attrs})

		super().__init__(**attrs)

		return


class Space(object):
	'''
	Hilbert space class for Operators with size n
	Args:
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		space (str,Space): Type of Hilbert space
		system (dict,System): System attributes
	'''
	def __init__(self,N,D,d,L,space,system):

		self.system = System(system)
		self.N = N if N is not None else 1
		self.D = D if D is not None else 2
		self.d = d if d is not None else 1
		self.L = L if L is not None else 1
		self.space = space		
		self.default = 'spin'

		self.__setup__()
		
		return

	def __setup__(self):
		'''
		Setup space attributes space,string,n
		'''
		if isinstance(self.space,Space):
			self.space = self.space.space
		if self.space is None:
			self.space = self.default
		self.__string__()
		self.__size__()
		return

	def __string__(self):
		self.string = self.space
		return

	def __size__(self):
		self.n = self.get_n()
		return 

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return str(self.string)

	def get_N(self,n):
		if self.space in ['spin']:
			try:
				return int(log(n)/log(self.D))
			except:
				return 1
		else:
			try:
				return int(log(n)/log(self.D))
			except:
				return 1			
		return 		

	def get_D(self,n):
		if self.space in ['spin']:
			return int(n**(1/self.N))
		else:
			return int(n**(1/self.N))
		return

	def get_n(self):
		if self.space in ['spin']:
			return self.D**self.N
		else:
			return self.D**self.N
		return		


class Time(object):
	'''
	Time evolution class for Operators with size n
	Args:
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		time (str,Time): Type of Time evolution space
		system (dict,System): System attributes
	'''
	def __init__(self,M,T,p,time,system):

		self.system = System(system)
		self.M = M if M is not None else 1
		self.T = T if T is not None else 1
		self.p = p if p is not None else 1
		self.time = time
		self.default = 'linear'

		self.__setup__()
		
		return

	def __setup__(self):
		'''
		Setup time evolution attributes tau
		'''
		if isinstance(self.time,Time):
			self.time = self.time.time
		if self.time is None:
			self.time = self.default
		self.__string__()
		self.__size__()
		return

	def __string__(self):
		self.string = self.time
		return
	def __size__(self):
		self.tau = self.get_tau()
		return 

	def __str__(self):
		return str(self.string)

	def __repr__(self):
		return str(self.string)

	def get_T(self,tau):
		if self.time in ['linear']:
			return tau*self.M
		else:
			return tau*self.M
		return 		

	def get_M(self,tau):
		if self.time in ['linear']:
			return self.T/tau
		else:
			return self.T/tau
		return

	def get_tau(self):
		if self.time in ['linear']:
			return self.T/self.M
		else:
			return self.T/self.M
		return	

class Lattice(object):
	'''
	Define a hyper lattice class
	Args:
		N (int): Lattice length along axis
		d (int): Dimension of lattice
		lattice (str,Lattice): Type of lattice, allowed strings in ['square','square-nearest']
		system (dict,System): System attributes (dtype,format,device,seed,verbose)		
	'''	
	def __init__(self,N=4,d=2,lattice='square',system=None):
		

		# Define lattice
		if isinstance(lattice,Lattice):
			lattice = lattice.lattice
		else:
			lattice = lattice

		# Define parameters of system        
		self.lattice = lattice
		self.N = N
		self.d = d

		# Define system
		self.system = System(system)
		self.dtype = self.system.dtype
		self.verbose = self.system.verbose

		self.dtype = self.dtype if self.dtype in ['int','Int32','Int64'] else int

		# Define linear size n and coordination number z	
		if self.lattice is None:
			N = 0
			d = 0
			n = 0
			z = 0
		elif self.lattice in ['square','square-nearest']:
			n = int(N**(1/d))
			z = 2*d
			assert n**d == N, "N != n^d for N=%d, d=%d, n=%d"%(N,d,n)
		else:
			n = int(N**(1/d))
			z = 2*d
			assert n**d == N, "N != n^d for N=%d, d=%d, n=%d"%(N,d,n)

		self.n = n
		self.z = z


		# Define attributes
		self.__size__()
		self.__shape__()
		self.__string__()

		# Define array of vertices
		self.vertices = np.arange(self.N)
		
		# n^i for i = 1:d array
		self.n_i = self.n**np.arange(self.d,dtype=self.dtype)
		
		# Arrays for finding coordinate and linear position in d dimensions
		self.I = np.eye(self.d)
		self.R = np.arange(1,max(2,np.ceil(self.n/2)),dtype=self.dtype)

		return


	def set(self,attr,value):
		'''	
		Set class attribute
		'''
		setattr(self,attr,value)
		return
		
	def get(self,attr,default=None):
		'''
		Get class attribute
		'''
		return getattr(self,attr,default)

	def log(self,msg):
		'''
		Log messages
		Args:
			msg (str): Message to log
		'''
		logger.log(self.verbose,msg)
		return

	def __call__(self,site=None):
		'''
		Get list of lists of sites of lattice
		Args:
			site (str,int): Type of sites, either int for unique site-length list of vertices, or string in allowed ["i","i,j","i<j"]
		Returns:
			sites (list): List of site-length lists of lattice
		'''

		# Unique site-length lists if site is int
		if isinstance(site,(int,np.integer)):
			k = site
			conditions = None
			sites = self.iterable(k,conditions)
		elif isinstance(site,(str)):
			if site in ["i"]:
				sites = [[i] for i in self.vertices]
			elif site in ["i,j"]:
				sites = [[i,j] for i in self.vertices for j in self.vertices]
			elif site in ["i<j"]:
				k = 2
				conditions = lambda i,k: all([i[j]<i[j+1] for j in range(k-1)])	
				sites = self.iterable(k,conditions)
			elif site in ["<ij>"]:
				if self.z > self.N:
					sites = []
				else:
					sites = [list(i) for i in unique(
						np.sort(
							np.vstack([
								np.repeat(np.arange(self.N),self.z),
								self.nearestneighbours(r=1)[0].ravel()
							]),
						axis=0),
						axis=1).T]

			elif site in ['i...j']:
				sites = [range(self.N) for i in range(self.N)]
		else:
			k = 2
			conditions = None
			sites = self.iterable(k,conditions)
		return sites


	def __string__(self):
		self.string = self.lattice if self.lattice is not None else 'null'
		return
		
	def __size__(self):
		self.size = self.N
		return 

	def __shape__(self):
		self.shape = (self.N,self.z)
		return

	def __str__(self):
		return self.string

	def __repr__(self):
		return self.string
		
	def position(self,site):
		'''
		Return position coordinates in d-dimensional n^d lattice 
		from given linear site position in 1d N^d length array
		i.e) [int(site/(self.n**(i))) % self.n for i in range(self.d)]
		Args:
			site (int,array): Linear site positions on lattice
		Returns:
			position (array): Position coordinates of linear site positions 
		'''
		isint = isinstance(site,(int,np.integer))

		if isint:
			site = np.array([site])
		position = np.mod(((site[:,None]/self.n_i)).
						astype(self.dtype),self.n)
		if isint:
			return position[0]
		else:
			return position
	
	def site(self,position):
		'''
		Return linear site position in 1d N^d length array 
		from given position coordinates in d-dimensional n^d lattice
		i.e) sum(position[i]*self.n**i for i in range(self.d))
		
		Args:
			position (array): Position coordinates of linear site positions 
		Returns:
			site (int,array): Linear site positions on lattice
		'''
		is1d = isinstance(position,(list,tuple)) or position.ndim < 2

		if is1d:
			position = np.array([position])
		
		site = position.dot(self.n_i).astype(self.dtype)

		if is1d:
			return site[0]
		else:
			return site


	def nearestneighbours(self,r=None,vertices=None):
		'''
		Return array of neighbouring spin vertices 
		for a given site and r-distance bonds
		i.e) [self.site(put(self.position(site),i,
						lambda x: mod(x + s*r,self.n))) 
						for i in range(self.d)for s in [1,-1]])
		Args:
			r (int,list): Radius of number of nearest neighbours away to compute nearest neighbours on lattice of shape (l,)
			vertices (array): Vertices to compute nearest neighbours on lattice of shape (N,)
		Returns:
			nearestneighbours (array): Array of shape (l,N,z) of nearest neighbours a manhattan distance r away
		'''
		if vertices is None:
			vertices = self.vertices
		
		sitepos = self.position(vertices)[:,None]
		
		if r is None:
			Rrange = self.R
		elif isinstance(r,list):
			Rrange = r
		else:
			Rrange = [r]
		return np.array([np.concatenate(
							(self.site(np.mod(sitepos+R*self.I,self.n)),
							 self.site(np.mod(sitepos-R*self.I,self.n))),1)
								for R in Rrange],dtype=self.dtype)                     


	def iterable(self,k,conditions=None):
		'''
		Return iterable of k-tuples of combinations of vertices
		Conditions limit generator to certain combinations of vertices
		Args:
			k (int): Number of vertices in lists of combinations of vertices
			conditions (callable): Conditions on allowed combinations of vertices k-lists i with signature conditons(i,k)
		Returns:
			iterable (list): list of k-lists of allowed combinations of vertices
		'''

		default = lambda i,k: any([i[j] != i[l] for j in range(k) for l in range(k) if j!=l])
		conditions = default if conditions is None else conditions
		iterable =  [list(i) for i in itertools.product(self.vertices,repeat=k) if conditions(i,k)]
		return iterable


class Object(object):
	'''
	Class for object
	Args:
		operator (iterable[str]): string names of operators
		site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		hyperparameters (dict) : class hyperparameters
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator,site,string,interaction,hyperparameters,N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):

		self.N = N
		self.D = D
		self.d = d
		self.L = L
		self.M = M
		self.T = T
		self.p = p
		self.space = space
		self.time = time
		self.lattice = lattice
		self.system = system

		self.hyperparameters = hyperparameters

		self.data = []
		self.size = getattr(self,'size',0)
		self.shape = (self.size,)
		self.locality = getattr(self,'locality',0)
		self.site = []
		self.string = []
		self.interaction = []

		self.delimiter = getattr(self,'delimiter',' ')
		self.basis = getattr(self,'basis',None)
		self.diagonal = []
		self._data = []
		self.funcs = lambda parameters: None
		self.expms = lambda parameters: None
		self.transform = []
		self.transformH = []
		self.index = arange(self.size)

		self.parameters = getattr(self,'parameters',None)
		self.dim = getattr(self,'dim',0)

		self.__system__()
		self.__space__()
		self.__time__()
		self.__lattice__()
		self.__setup__(operator,site,string,interaction,hyperparameters)
	
		return	


	def __parameters__(self,parameters=None):
		''' 
		Setup parameters
		Args:
			parameters (array): parameters
		'''
		self.parameters = parameters
		return parameters

	def __system__(self,system=None):
		'''
		Set system attributes
		Args:
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''
		system = self.system if system is None else system
		
		self.system = System(system)		
		self.dtype = self.system.dtype
		self.verbose = self.system.verbose

		return


	def __space__(self,N=None,D=None,d=None,L=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			d (int): Spatial dimension
			L (int,float): Scale in system
			space (str,Space): Type of Hilbert space
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''
		N = self.N if N is None else N
		D = self.D if D is None else D
		d = self.d if d is None else d
		L = self.L if L is None else L
		space = self.space if space is None else space
		system = self.system if system is None else system

		self.space = Space(N,D,d,L,space,system)
		self.N = self.space.N
		self.D = self.space.D
		self.d = self.space.d
		self.L = self.space.L
		self.n = self.space.n
		self.identity = identity(self.n)

		return


	def __time__(self,M=None,T=None,p=None,time=None,system=None):
		'''
		Set time attributes
		Args:
			M (int): Number of time steps
			T (int): Simulation Time
			p (int): Trotter order		
			time (str,Time): Type of Time evolution space						
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''
		M = self.M if M is None else M
		T = self.T if T is None else T
		p = self.p if p is None else p
		time = self.time if time is None else time
		system = self.system if system is None else system

		self.time = Time(M,T,p,time,system)		
		self.M = self.time.M
		self.T = self.time.T
		self.p = self.time.p
		self.tau = self.time.tau

		return


	def __lattice__(self,N=None,D=None,d=None,L=None,lattice=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Number of qudits
			D (int): Dimension of qudits
			d (int): Spatial dimension
			L (int,float): Scale in system
			lattice (str,Lattice): Type of lattice		
			system (dict,System): System attributes (dtype,format,device,seed,verbose)		
		'''		
		N = self.N if N is None else N
		D = self.D if D is None else D
		d = self.d if d is None else d
		L = self.L if L is None else L
		lattice = self.lattice if lattice is None else lattice
		system = self.system if system is None else system

		self.lattice = Lattice(N,d,lattice,system)	

		return


	def __str__(self):
		size = self.size
		multiple_time = (self.M>1)
		multiple_space = [size>1 and False for i in range(size)]
		return '%s%s%s%s'%(
				'{' if multiple_time else '',
				self.delimiter.join(['%s%s%s'%(
					'(' if multiple_space[i] else '',
					self.string[i],
					')' if multiple_space[i] else '',
					) for i in range(size)]),
				'}' if multiple_time else '',
				'^%d'%(self.M) if multiple_time else '')

	def __repr__(self):
		return self.__str__()

	def __len__(self):
		return self.size

	def log(self,msg):
		'''
		Log messages
		Args:
			msg (str): Message to log
		'''
		logger.log(self.verbose,msg)
		return	

	def dump(self,path=None,parameters=None,ext='txt'):
		'''
		Save class data		
		'''
		if path is None:
			path = '%s.%s'%(self,ext)
		data = self(parameters)
		dump(data,path)
		return


	def load(self,path=None,parameters=None,ext='txt'):
		'''
		Load class data		
		'''
		if path is None:
			path = '%s.%s'%(self,ext)
		data = array(load(path,dtype=self.dtype))
		string = basename(path)
		self.append(data,string=string)
		return



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


class Hamiltonian(Object):
	'''
	Hamiltonian class of Operators
	Args:
		operator (iterable[str]): string names of operators
		site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		hyperparameters (dict) : class hyperparameters
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator,site,string,interaction,hyperparameters={},N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site=site,string=string,interaction=interaction,hyperparameters=hyperparameters,
				 N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return


	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Return parameterized operator sum(parameters*operator)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			operator (array): Parameterized operator
		'''		
		parameters = self.__parameters__(parameters)

		return (parameters*self.data).sum(0)

	def __setup__(self,operator,site,string,interaction,hyperparameters={}):
		'''
		Setup class
		Args:
			operator (iterable[str]): string names of operators
			site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
			string (iterable[str]): string labels of operators
			interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
			hyperparameters (dict) : class hyperparameters
		'''

		# Get hyperparameters
		hyperparameters.update(self.hyperparameters)

		# Get number of operators
		size = min([len(i) for i in [operator,site,string,interaction]])

		# Lattice interactions
		interactions = {site: self.lattice(site) for site in ['i','i<j','<ij>','i...j']}	# interactions types on lattice
		locality = 2 # maximum number of body interactions
		sites = ['i','j'] # allowed symbolic sites


		# Get all sites from symbolic sites
		for i in range(size):
			_operator = operator.pop(0);
			_site = site.pop(0);
			_string = string.pop(0);
			_interaction = interaction.pop(0);
			if any(j in sites for j in site[i]):
				for s in interactions[interaction[i]]:
					_site = [dict(zip(sites,s if not isinstance(s,int) else [s])).get(j,parse(j,int)) for j in site[i]]

					operator.append(copy.deepcopy(_operator))
					site.append(copy.deepcopy(_site))
					string.append(copy.deepcopy(_string))
					interaction.append(copy.deepcopy(_interaction))
			else:
				operator.append(copy.deepcopy(_operator))
				site.append(copy.deepcopy(_site))
				string.append(copy.deepcopy(_string))
				interaction.append(copy.deepcopy(_interaction))


		print(operator)
		print(site)
		print(string)
		print(interaction)
		exit()

		# Get identity operator I, to be maintained with same shape of data for Euler identities
		# with minimal redundant copying of data
		I = 'I'
		identity = [I]*self.N


		# Form (size,n,n) shape operator from local strings for each data term
		data = array([multi_tensorprod([basis[j] for j in i]) for i in operator])
		identity = multi_tensorprod([basis[i] for i in identity])

		# Get size of data
		size = len(data)

		# Get Trotterized order of p copies of data for products of data
		data = trotter(data,self.p)

		# Reshape data to shape (p*size,n,n)
		data = data.reshape((self.p*size,self.n,self.n))

		# Get shape of parameters
		shape = (self.M,size)


		# Update indices of parameters within data and slices of parameters within parameters
		categories = list(set([hyperparameters['parameters'][parameter]['category'] for parameter in hyperparameters['parameters']]))
		indices = {category:{} for category in categories}
		slices = {category:{} for category in categories}
		for parameter in hyperparameters['parameters']:

			category = hyperparameters['parameters'][parameter]['category']
			locality = hyperparameters['parameters'][parameter]['locality']


			n = max([-1,*[slices[category][group][-1] for group in slices[category]]])+1
			for group in hyperparameters['parameters'][parameter]['group']:
				indices[category][group] = [i for i,s in enumerate(string) 
					if any(g in group for g in [s,'_'.join([s,''.join(['%d'%j for j in sites[i]])])])] 
			
				m = len(indices[category][group]) if hyperparameters['parameters'][parameter]['locality'] in ['local'] else 1
				s = hyperparameters['parameters'][parameter]['size'] if hyperparameters['parameters'][parameter].get('size') is not None else 1
				slices[category][group] = [n+i*s+j for i in range(m) for j in range(s)]

			hyperparameters['parameters'][parameter]['index'] = {group: [j for j in indices[category][group]] for group in indices[category] if group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['slice'] = {group: [j for j in slices[category][group]] for group in slices[category]  if group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['site'] =  {group: [sites[j] for j in indices[category][group]] for group in indices[category]  if group in hyperparameters['parameters'][parameter]['group']}
			hyperparameters['parameters'][parameter]['string'] = {group: ['_'.join([string[j],''.join(['%d'%(k) for k in sites[j]]),''.join(operator[j])]) for j in indices[category][group]] for group in indices[category]  if group in hyperparameters['parameters'][parameter]['group']}

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


		# Initialize parameters
		self.__init__parameters__()


		# Update hyperparameters
		hyperparameters['data'] = data
		hyperparameters['identity'] = identity
		hyperparameters['size'] = size
		hyperparameters['shape'] = shape
		hyperparameters['shapes'] = shapes
		hyperparameters['string'] = string
		hyperparameters['site'] = site
		hyperparameters['operator'] = operator
		hyperparameters['N'] = self.N
		hyperparameters['M'] = self.M
		hyperparameters['D'] = self.D
		hyperparameters['d'] = self.d
		hyperparameters['n'] = self.n
		hyperparameters['p'] = self.p
		hyperparameters['coefficients'] = self.T/self.M

		# Update class attributes
		self.data = data
		self.size = size
		self.string = string
		self.site = site
		self.operator = operator
		self.identity = identity
		self.hyperparameters = hyperparameters

		return


	def __parameters__(self,parameters):
		''' 
		Setup parameters
		Args:
			parameters (array): parameters
		'''		

		# Get parameters
		self.parameters = parameters

		# Get hyperparameters
		hyperparameters = self.hyperparameters

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


	def __init__parameters__(self,parameters=None):
		''' 
		Setup initial parameters
		Args:
			parameters (array): parameters
		'''

		# Update hyperparameters
		hyperparameters = self.hyperparameters

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

		# Update parameters
		self.parameters = parameters

		return parameters



class Unitary(Hamiltonian):
	'''
	Unitary class of Operators
	Args:
		operator (iterable[str]): string names of operators
		site (iterable[iterable[int,str]]): site of local operators, allowed strings in [["i"],["i","j"]]
		string (iterable[str]): string labels of operators
		interaction (iterable[str]): interaction types of operators type of interaction, i.e) nearest neighbour, allowed values in ["i","i,j","i<j","i...j"]
		hyperparameters (dict) : class hyperparameters
		N (int): Number of qudits
		D (int): Dimension of qudits
		d (int): Spatial dimension
		L (int,float): Scale in system
		M (int): Number of time steps
		T (int): Simulation Time
		p (int): Trotter order		
		space (str,Space): Type of Hilbert space
		time (str,Time): Type of Time evolution space						
		lattice (str,Lattice): Type of lattice		
		system (dict,System): System attributes (dtype,format,device,seed,verbose)
	'''

	def __init__(self,operator,site,string,interaction,hyperparameters={},N=None,D=None,d=None,L=None,M=None,T=None,p=None,space=None,time=None,lattice=None,system=None):
		super().__init__(operator,site=site,string=string,interaction=interaction,hyperparameters=hyperparameters,
				 N=N,D=D,d=d,L=L,M=M,T=T,p=p,space=space,time=time,lattice=lattice,system=system)
		return

	#@partial(jit,static_argnums=(0,))
	def __call__(self,parameters=None):
		'''
		Return parameterized operator sum(parameters*operator)
		Args:
			parameters (array): Parameters to parameterize operator
		Returns
			operator (array): Parameterized operator
		'''		
		parameters = self.__parameters__(parameters)

		return expm(parameters,self.data,self.identity)

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


def main(index,hyperparameters={}):


	def decorator(hyperparameters):
		def wrapper(func):
			@functools.wraps
			def wrapped(parameters):
				return func(parameters,hyperparameters)
			return wrapped
		return wrapper






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
		U = unitary(parameters)
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
	unitary = Unitary(**hyperparameters['object'],**hyperparameters['model'],hyperparameters=hyperparameters)
	parameters = unitary.__parameters__()


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