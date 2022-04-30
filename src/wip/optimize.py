#!/usr/bin/env python

# Import python modules
import os,sys,itertools
import logging,logging.config
import numpy as np
import scipy as sp

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

from lib.utils.utils import jit,grad,finitegrad,value_and_grad,value_and_finitegrad
from lib.utils.utils import System,array
from lib.utils.utils import norm
from lib.utils.plot import plot


class Gradient(object):
	r'''
	Gradient class
	Args:
		grad (str):  Method for computing gradients, allowed strings in ['grad',autograd','tensor']
	'''
	def __init__(self,gradient='autograd',system=None):
		
		self.system = System(system)
		self.verbose = self.system.verbose

		gradient = gradient if not isinstance(gradient,Gradient) else grad.gradient

		self.gradient = gradient

		if gradient in ['grad']:
			self.func = finitegrad
		elif gradient in ['autograd']:
			self.func = grad
		else:
			self.func = grad

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

	def __call__(self,func):
		'''	
		Perform optimization
		Args:
			func (callable): Function to compute gradient with signature func(x)
		Returns:
			grad (callable): Gradient of function
		'''
		return self.func(func)


class Value_and_Gradient(object):
	r'''
	Gradient class
	Args:
		grad (str):  Method for computing gradients, allowed strings in ['grad',autograd','tensor']
	'''
	def __init__(self,grad='autograd',system=None):
		
		self.system = System(system)
		self.verbose = self.system.verbose

		grad = grad if not isinstance(grad,Gradient) else grad.grad

		if grad in ['grad']:
			self.func = value_and_finitegrad
		elif grad in ['autograd']:
			self.func = value_and_grad
		else:
			self.func = value_and_grad

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

	def __call__(self,func):
		'''	
		Perform optimization
		Args:
			func (callable): Function to compute gradient with signature func(x)
		Returns:
			grad (callable): Gradient of function
		'''
		return self.func(func)

class Optimize(object):
	r'''
	Optimize matrix dynamics V = e^{i\int_0^T H(x)} for target matrix U, with generator H(x),
	through maximization of max_x func(x,V,U,d)
	Args:
		func (callable): Optimization function with signature func(x,y,hyperparameters)
		grad (str):  Method for computing gradients, allowed strings in ['grad',autograd','tensor']
		optimizer (str): Method for optimization, allowed strings in ['gd','adam','bfgs']
		hyperparameters (dict): Optimization hyperparameters
	'''
	def __init__(self,func=None,grad='autograd',optimizer='gd',hyperparameters={},system=None):
		
		self.system = System(system)
		self.verbose = self.system.verbose

		self.func = func
		self.grad = Gradient(grad,system=system)(func)
		self.value_and_grad = Value_and_Gradient(grad,system=system)(func)
		self.optimizer = optimizer if not isinstance(optimizer,Optimize) else optimizer.optimizer

		self.hyperparameters = {
			'alpha':1e-6,
			'eps':1e-3,
			'iterations':1000,
			'y':[],
		}
		self.hyperparameters.update(hyperparameters)

		self.x = None
		self.y = None


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

	def __call__(self,x,y,hyperparameters={}):
		'''	
		Perform optimization
		Args:
			x (array): Function parameters
			y (array): Function labels
			hyperparameters (dict): Optimization hyperparameters
		Returns:
			x (array): Optimal function parameters
		'''
		hyperparameters.update({k:self.hyperparameters[k] for k in self.hyperparameters if k not in hyperparameters})

		self.set_parameters(x,hyperparameters)
		self.set_labels(y,hyperparameters)

		for i in range(hyperparameters['iterations']):
			_y,g = self.value_and_grad(self.x,self.y,hyperparameters)
			self.x -= hyperparameters['alpha']*g
			hyperparameters['y'].append(_y)
			if i%100 == 0:
				self.log('%d : x = %0.4e, f(x) = %0.4e'%(i,norm(x),-_y))
		return x

	def set_parameters(self,parameters,hyperparameters={}):
		'''
		Set parameters
		Args:
			parameters (array): parameters
			hyperparameters (dict): Optimization hyperparameters			
		'''		
		self.x = parameters
		return

	def set_labels(self,labels,hyperparameters={}):
		'''
		Set parameters
		Args:
			labels (array): labels
			hyperparameters (dict): Optimization hyperparameters			
		'''		
		self.y = labels
		return


class Model(object):
	'''
	Optimization model
	Args:
		optimizer (str,Optimize): Optimizer class
		func (callable): Function to predict with signature func(x,y,hyperparameters)
		loss (callable): Function to optimize with signature loss(x,y,hyperparameters)
		hyperparameters (dict): Optimization hyperparameters		
	'''
	def __init__(self,optimizer,func,loss,hyperparameters={},system=None):

		self.system = System(system)
		self.verbose = self.system.verbose

		self.optimizer = Optimize(func=loss,optimizer=optimizer,hyperparameters=hyperparameters,system=system)
		self.func = func
		self.loss = loss
		
		self.hyperparameters = {
			'y':[],
		}
		self.hyperparameters.update(hyperparameters)

		self.x = None
		self.y = None
		self._y = None

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

	def train(self,x,y,hyperparameters={}):
		'''
		Call optimization function and update optimal x
		Args:
			x (array): Initial guess for optimization
			y (array): Data for optimization
			hyperparameters (dict): Optimization hyperparameters
			'''
		hyperparameters.update({k:self.hyperparameters[k] for k in self.hyperparameters if k not in hyperparameters})		
		
		self.set_parameters(x,hyperparameters)
		self.set_labels(y,hyperparameters)

		self.x = self.optimizer(self.x,self.y,hyperparameters=hyperparameters)
		self._y = self.predict(self.x,self.y,hyperparameters=hyperparameters)
		
		return self

	def predict(self,x,y=None,hyperparameters={}):
		'''
		Predict function
		Args:
			x (array): Data
			y (array): Data for optimization
			hyperparameters (dict): Optimization hyperparameters			
		Returns:
			prediction (object): Function value of x
		'''
		return self.func(x,y,hyperparameters)

	def set_parameters(self,parameters,hyperparameters={}):
		'''
		Set parameters
		Args:
			parameters (array): parameters
			hyperparameters (dict): Optimization hyperparameters			
		'''		

		hyperparameters.update({k:self.hyperparameters[k] for k in self.hyperparameters if k not in hyperparameters})				
		
		shape = hyperparameters['shape']
		constants = hyperparameters['constants']
		bounds = hyperparameters['bounds']
		boundaries = hyperparameters['boundaries']
		kind = hyperparameters['interpolation']

		n_interp = shape[0]//2 + 1
		n,m = shape[0],shape[1]-len(constants)
		
		pts_interp = 2*np.arange(n_interp)
		pts = np.arange(n)
		
		x_interp = (bounds[1]-bounds[0])*np.random.rand(n_interp,m)
		x = np.zeros((n,m))

		for b in boundaries:
			x_interp[b] = boundaries[b]

		for i in range(m):
			x[:,i] = (bounds[1]-bounds[0])*(1 - ((pts - (pts[-1]+pts[0])/2)/((pts[-1]-pts[0])/2))**2)
			# x[:,i] = sp.interpolate.interp1d(pts_interp,x_interp[:,i],kind)(pts)

		for b in boundaries:
			x[b] = boundaries[b]


		x_plt = (pts_interp,pts)*m 
		y_plt = tuple([u for i in range(m) for u in [x_interp[:,i],x[:,i]]])

		settings = {
		i: {						
			'fig':{'savefig':{'fname':'initial.pdf'},'set_size_inches':{'w':12,'h':12},'tight_layout':{},},
			'ax':{
				'plot':[{'x':pts_interp,'y':x_interp[:,i],'label':r'$\alpha_{%s}^{\textrm{initial}}$'%(['x','y'][i])},
						  {'x':pts,'y':x[:,i],'label':r'$\alpha_{%s}^{\textrm{interp}}$'%(['x','y'][i])}],
				'set_xlabel':{'xlabel':r'$\textrm{Time}$'},
				'set_ylabel':{'ylabel':r'$\textrm{Amplitude}$'},
				'legend':{'loc':(1.1,0.5)}},
			}
		for i in range(m)
		}

		plot(settings=settings)
			
		x = array(x)
		self.x = x

		return

	def set_labels(self,labels,hyperparameters={}):
		'''
		Set parameters
		Args:
			labels (array): labels
			hyperparameters (dict): Optimization hyperparameters			
		'''		
		hyperparameters.update({k:self.hyperparameters[k] for k in self.hyperparameters if k not in hyperparameters})				
		self.y = labels
		return