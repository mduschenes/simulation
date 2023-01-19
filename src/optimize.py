#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy,datetime
from functools import partial
import jax

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))


# Logging
import logging
logger = logging.getLogger(__name__)

# Import user modules
from src.utils import jit,value_and_gradient,gradient,conj,abs
from src.utils import is_unitary,is_hermitian,product,sqrt,asarray

from src.utils import inner_norm,inner_abs2,inner_real,inner_imag
from src.utils import gradient_inner_norm,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag

from src.utils import scalars,delim

from src.iterables import setter

from src.line_search import line_search,armijo

from src.io import dump,load,join,split,copy,exists

from src.system import System



class LineSearcher(System):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		
		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)


		defaults = {}		
		returns = ['alpha']
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		self.func = func
		self.grad = grad
		self.hyperparameters = hyperparameters
		self.system = system
		self.defaults = defaults
		self.returns = returns
		
		return

	def __call__(self,parameters,alpha,value,grad,search):
		'''
		Perform line search
		Args:
			parameters (array): Objective parameters
			alpha (iterable[array]): Previous alpha
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			alpha (array): Returned search value
		'''
		_alpha = alpha[-1]
		returns = (_alpha,)

		returns = self.__callback__(returns,parameters,alpha,value,grad,search)

		attr = 'alpha'
		alpha = returns[attr]

		return alpha

	def __callback__(self,returns,parameters,alpha,value,grad,search):
		'''
		Check return values of line search
		Args:
			returns (iterable): Iterable of returned values of line search
			parameters (array): Objective parameters
			alpha (iterable[array]): Previous alpha
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			returns (dict): Dictionary of returned values of line search
		'''
		
		returns = dict(zip(self.returns,returns))

		attr = 'alpha'
		if returns[attr] is None or (returns[attr] < self.hyperparameters['bounds'][attr][0]) or (returns[attr] > self.hyperparameters['bounds'][attr][1]):
			if len(alpha) > 1:
				returns[attr] = alpha[-1]*grad[-1].dot(search[-1])/grad[-2].dot(search[-2])
			else:
				returns[attr] = alpha[-1]

		return returns

class LineSearch(LineSearcher):
	'''
	Line Search class
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		hyperparameters (dict): line search hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
	'''
	def __new__(cls,func,grad,hyperparameters={},system=None):
	
		defaults = {'search':{'alpha':None}}
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		line_searches = {
			'line_search':Line_Search,
			'armijo':Armijo,
			None:Null_Line_Search
			}

		line_search = hyperparameters.get('search',{}).get('alpha',defaults['search']['alpha'])
		
		self = line_searches.get(line_search,line_searches[None])(func,grad,hyperparameters)

		return self


class Line_Search(LineSearcher):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		defaults = {'c1':0.0001,'c2':0.9,'maxiter':10,'old_old_fval':None}
		returns = ['alpha','nfunc','ngrad','value','_value','slope']
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		super().__init__(func,grad,hyperparameters=hyperparameters,system=system,**kwargs)

		self.defaults = defaults
		self.returns = returns

		return

	def __call__(self,parameters,alpha,value,grad,search):
		'''
		Perform line search
		Args:
			parameters (array): Objective parameters
			alpha (iterable[array]): Previous alpha
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			alpha (array): Returned search value
		'''
		self.defaults['old_old_fval'] = value[-2] if len(value)>1 else None
		
		returns = line_search(self.func,self.grad,
			parameters,search[-1],grad[-1],value[-1],
			**self.defaults)
		
		returns = self.__callback__(returns,parameters,alpha,value,grad,search)

		attr = 'alpha'
		alpha = returns[attr]

		return alpha


class Armijo(LineSearcher):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		defaults = {'c1':0.0001,'alpha0':1e-4}
		returns = ['alpha','nfunc','value']
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		super().__init__(func,grad,hyperparameters=hyperparameters,system=system,**kwargs)

		self.defaults = defaults
		self.returns = returns

		return

	def __call__(self,parameters,alpha,value,grad,search):
		'''
		Perform line search
		Args:
			parameters (array): Objective parameters
			alpha (iterable[array]): Previous alpha
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			alpha (array): Returned search value
		'''

		returns = armijo(self.func,self.grad,
			parameters,search[-1],grad[-1],value[-1],
			**self.defaults)

		returns = self.__callback__(returns,parameters,alpha,value,grad,search)
		
		return returns


class Null_Line_Search(LineSearcher):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)			
			kwargs (dict): Additional system attributes
		'''

		super().__init__(func,grad,hyperparameters=hyperparameters,system=system,**kwargs)

		return



class GradSearcher(System):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Grad search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		
		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)


		defaults = {}		
		returns = ['beta']
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		self.func = func
		self.grad = grad
		self.hyperparameters = hyperparameters
		self.system = system
		self.defaults = defaults
		self.returns = returns
		
		return

	def __call__(self,parameters,beta,value,grad,search):
		'''
		Perform grad search
		Args:
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			beta (array): Returned search value
		'''
		_beta = beta[-1]
		returns = (_beta,)

		returns = self.__callback__(returns,parameters,beta,value,grad,search)

		attr = 'beta'
		beta = returns[attr]

		return beta

	def __callback__(self,returns,parameters,beta,value,grad,search):
		'''
		Check return values of grad search
		Args:
			returns (iterable): Iterable of returned values of grad search
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			returns (dict): Dictionary of returned values of grad search
		'''
		
		returns = dict(zip(self.returns,returns))

		attr = 'beta'
		if returns[attr] is None or (returns[attr] < self.hyperparameters['bounds'][attr][0]) or (returns[attr] > self.hyperparameters['bounds'][attr][1]):
			if len(beta) > 1:
				returns[attr] = beta[-1]
			else:
				returns[attr] = beta[-1]

		return returns

class GradSearch(GradSearcher):
	'''
	Grad Search class
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		hyperparameters (dict): grad search hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
	'''
	def __new__(cls,func,grad,hyperparameters={},system=None):
	
		defaults = {'search':{'beta':None}}
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		grad_searches = {
			'grad_search':Fletcher_Reeves,'fletcher_reeves':Fletcher_Reeves,
			'polak_ribiere':Polak_Ribiere,'polak_ribiere_fletcher_reeves':Polak_Ribiere_Fletcher_Reeves,
			'hestenes_stiefel':Hestenes_Stiefel,'dai_yuan':Dai_Yuan,
			None:Null_Grad_Search,
			}

		grad_search = hyperparameters.get('search',{}).get('beta',defaults['search']['beta'])
		
		self = grad_searches.get(grad_search,grad_searches[None])(func,grad,hyperparameters)

		return self


class Fletcher_Reeves(GradSearcher):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,parameters,beta,value,grad,search):
		'''
		Perform grad search
		Args:
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			beta (array): Returned search value
		'''
		_beta = (grad[-1].dot(grad[-1]))/(grad[-2].dot(grad[-2])) # Fletcher-Reeves
		returns = (_beta,)

		returns = self.__callback__(returns,parameters,beta,value,grad,search)

		attr = 'beta'
		beta = returns[attr]

		return beta


class Polak_Ribiere(GradSearcher):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,parameters,beta,value,grad,search):
		'''
		Perform grad search
		Args:
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			beta (array): Returned search value
		'''
		_beta = max(0,(grad[-1].dot(grad[-1]-grad[-2]))/grad[-2].dot(grad[-2]))  # Polak-Ribiere
		returns = (_beta,)

		returns = self.__callback__(returns,parameters,beta,value,grad,search)

		attr = 'beta'
		beta = returns[attr]

		return beta


class Polak_Ribiere_Fletcher_Reeves(GradSearcher):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,parameters,beta,value,grad,search):
		'''
		Perform grad search
		Args:
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			beta (array): Returned search value
		'''
		_beta = [(grad[-1].dot(grad[-1]))/(grad[-2].dot(grad[-2])),max(0,(grad[-1].dot(grad[-1]-grad[-2]))/grad[-2].dot(grad[-2]))] # Polak-Ribiere-Fletcher-Reeves
		_beta = -_beta[0] if _beta[1] < -_beta[0] else _beta[1] if abs(_beta[1]) <= _beta[0] else _beta[0]
		returns = (_beta,)

		returns = self.__callback__(returns,parameters,beta,value,grad,search)

		attr = 'beta'
		beta = returns[attr]

		return beta


class Hestenes_Stiefel(GradSearcher):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,parameters,beta,value,grad,search):
		'''
		Perform grad search
		Args:
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			beta (array): Returned search value
		'''
		_beta = (grad[-1].dot(grad[-1]-grad[-2]))/(search[-1].dot(grad[-1]-grad[-2])) # Hestenes-Stiefel
		returns = (_beta,)

		returns = self.__callback__(returns,parameters,beta,value,grad,search)

		attr = 'beta'
		beta = returns[attr]

		return beta

class Dai_Yuan(GradSearcher):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,parameters,beta,value,grad,search):
		'''
		Perform grad search
		Args:
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
		Returns:
			beta (array): Returned search value
		'''
		_beta = (grad[-1].dot(grad[-1]))/(search[-1].dot(grad[-1]-grad[-2])) # Dai-Yuan https://doi.org/10.1137/S1052623497318992
		returns = (_beta,)

		returns = self.__callback__(returns,parameters,beta,value,grad,search)

		attr = 'beta'
		beta = returns[attr]

		return beta


class Null_Grad_Search(GradSearcher):
	def __init__(self,func,grad,hyperparameters,system=None,**kwargs):
		'''	
		Grad search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)			
			kwargs (dict): Additional system attributes
		'''

		super().__init__(func,grad,hyperparameters=hyperparameters,system=system,**kwargs)

		return


class Function(System):

	def __init__(self,model,func=None,grad=None,callback=None,metric=None,hyperparameters={},system=None,**kwargs):
		'''	
		Class for function
		Args:
			model (object): Model instance
			func (callable,iterable[callable]): Function function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			callback (callable): Callback of function with signature callback(parameters,track,optimizer,model,metric,func,grad)
			metric (str,callable): Function metric with signature metric(*operands)
			hyperparameters (dict): Function hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''

		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		if func is None:
			func = []
		if not callable(func):
			if len(func) == 1:
				function = func[0]
			elif func:
				@jit
				def function(*args,**kwargs):
					return sum(function(*args,**kwargs) for function in func)
			else:
				@jit
				def function(*args,**kwargs):
					return 0.
		else:
			function = func

		if grad is None:
			gradient = None
		elif not callable(grad):
			if len(grad) == 1:
				gradient = grad[0]
			elif grad:
				@jit
				def gradient(*args,**kwargs):
					return sum(gradient(*args,**kwargs) for gradient in grad)
			else:
				@jit
				def gradient(*args,**kwargs):
					return 0.
		else:
			gradient = grad

		if callback is None:
			def callback(parameters,track,optimizer,model,metric,func,grad):
				status = True
				return status

		self.value_and_gradient,self.function,self.gradient = value_and_gradient(function,gradient,returns=True)

		self.model = model
		self.callback = callback
		self.metric = metric
		self.hyperparameters = hyperparameters
		self.system = system

		return

	# @partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Function call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.function(parameters)

	# @partial(jit,static_argnums=(0,))
	def __grad__(self,parameters):
		'''
		Gradient call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.gradient(parameters)

	# @partial(jit,static_argnums=(0,))
	def __value_and_grad__(self,parameters):
		'''
		Function and gradient call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.value_and_gradient(parameters)

	# @partial(jit,static_argnums=(0,))
	def __callback__(self,parameters,track,optimizer):
		'''
		Callback call
		Args:
			parameters (array): parameters
			track (dict): callback track			
			optimizer (Optimizer): callback optimizer
		Returns:
			status (int): status of callback
		'''
		status = self.callback(parameters,track,optimizer,
			model=self.model,
			metric=self.metric,
			func=self.func,grad=self.grad)
		return status

	# @partial(jit,static_argnums=(0,))
	def func(self,parameters):
		'''
		Function call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.__call__(parameters)

	# @partial(jit,static_argnums=(0,))
	def grad(self,parameters):
		'''
		Gradient call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.__grad__(parameters)

	# @partial(jit,static_argnums=(0,))
	def value_and_grad(self,parameters):
		'''
		Function and gradient call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.__value_and_gradient__(parameters)


class Objective(Function):		
	def __init__(self,model,metric,func=None,grad=None,callback=None,hyperparameters={},system=None,**kwargs):
		'''	
		Objective class for metric + function
		Args:
			model (object): Model instance
			metric (str,callable): Objective metric with signature metric(*operands)
			func (callable,iterable[callable]): Objective function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			callback (callable): Callback of function with signature callback(parameters,track,optimizer,model,metric,func,grad)			
			hyperparameters (dict): Objective hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''

		super().__init__(model,func=func,grad=grad,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system,**kwargs)

		return

	# @partial(jit,static_argnums=(0,))
	def __call__(self,parameters):
		'''
		Function call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.metric(self.model(parameters)) + self.function(parameters)

	# @partial(jit,static_argnums=(0,))
	def func(self,parameters):
		'''
		Function call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.__call__(parameters)

	# @partial(jit,static_argnums=(0,))
	def __grad__(self,parameters):
		'''
		Gradient call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.metric.grad(self.model(parameters),self.model.grad(parameters)) + self.gradient(parameters)	

	# @partial(jit,static_argnums=(0,))
	def __grad_analytical__(self,parameters):
		'''
		Gradient call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.metric.grad_analytical(self.model(parameters),self.model.grad_analytical(parameters)) + self.gradient(parameters)	

	# @partial(jit,static_argnums=(0,))
	def grad_analytical(self,parameters):
		'''
		Gradient call
		Args:
			parameters (array): parameters
		Returns:
			out (object): Return of function
		'''
		return self.__grad_analytical__(parameters)


class Callback(Function):

	def __init__(self,model,callback,func=None,grad=None,metric=None,hyperparameters={},system=None,**kwargs):
		'''	
		Class for function
		Args:
			model (object): Model instance
			callback (callable): Callback of function with signature callback(parameters,track,optimizer,model,metric,func,grad)
			func (callable,iterable[callable]): Function function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			metric (str,callable): Callback metric with signature metric(*operands)
			hyperparameters (dict): Callback hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		
		super().__init__(model,func=func,grad=grad,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system,**kwargs)

		return

	# @partial(jit,static_argnums=(0,))
	def __call__(self,parameters,track,optimizer):
		''' 
		Callback
		Args:
			parameters (array): parameters
			track (dict): callback tracking
			optimizer (Optimizer): callback optimizer
		Returns:
			status (int): status of callback
		'''
		status = self.__callback__(parameters,track,optimizer)
		return status


class Metric(System):
	def __init__(self,metric=None,shapes=None,model=None,label=None,optimize=None,hyperparameters={},system=None,**kwargs):
		'''
		Metric class for distance between operands
		Args:
			metric (str,Metric): Type of metric
			shapes (iterable[tuple[int]]): Shapes of Operators
			model (object): Model instance	
			label (str,callable): Label			
			optimize (bool,str,iterable): Contraction type	
			hyperparameters (dict): Metric hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
			kwargs (dict): Additional system attributes
		'''

		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		self.metric = hyperparameters.get('metric',metric) if metric is None else metric
		self.label = hyperparameters.get('label',label) if label is None else label
		self.shapes = shapes
		self.model = model
		self.optimize = optimize
		self.hyperparameters = hyperparameters
		self.system = system

		self.__setup__()

		return

	def __setup__(self):
		'''
		Setup metric attributes metric,string
		'''
		if isinstance(self.metric,Metric):
			self.metric = self.metric.metric
		if self.metric is None:
			self.metric = None
		if self.shapes is None:
			self.shapes = ()

		self.__string__()
		self.__size__()

		self.get_metric()

		self.info()

		return

	def __string__(self):
		'''
		Class string
		'''
		self.string = str(self.metric)
		return

	def __size__(self):
		'''
		Class size
		'''
		if self.shapes:
			self.size = sum(int(product(shape)**(1/len(shape))) for shape in self.shapes)//len(self.shapes)
		else:
			self.size = 1
		return 

	# @partial(jit,static_argnums=(0,))
	def __call__(self,*operands):
		'''
		Function call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''
		return self.function(*operands)

	# @partial(jit,static_argnums=(0,))
	def __grad__(self,*operands):
		'''
		Gradient call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.gradient(*operands)

	# @partial(jit,static_argnums=(0,))
	def __grad_analytical__(self,*operands):
		'''
		Gradient call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.gradient_analytical(*operands)

	# @partial(jit,static_argnums=(0,))
	def func(self,*operands):
		'''
		Function call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.__call__(*operands)

	# @partial(jit,static_argnums=(0,))
	def grad(self,*operands):
		'''
		Gradient call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.__grad__(*operands)	

	# @partial(jit,static_argnums=(0,))
	def grad_analytical(self,*operands):
		'''
		Gradient call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.__grad_analytical__(*operands)	


	def __str__(self):
		'''
		Class string
		'''
		return str(self.string)

	def __repr__(self):
		'''
		Class representation
		'''
		return str(self.string)

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		
		msg = '%s'%('\n'.join([
			*['%s: %s'%(attr,getattr(self,attr)) 
				for attr in ['metric']
			],
			]
			))
		self.log(msg,verbose=verbose)
		return


	def get_metric(self):
		'''
		Setup metric	
		'''
		
		if callable(self.metric):
			metric = self.metric
			function = jit(metric)
			grad = jit(gradient(metric))
			# grad = gradient(function,mode='fwd',holomorphic=True,move=True)			
			gradient_analytical = jit(gradient(metric))
		else:

			if self.label is not None:
				if is_unitary(self.label) and self.metric in ['real','imag','norm']:
					self.metric = 'abs2'
				elif is_hermitian(self.label) and self.metric in ['abs2']:
					self.metric = 'real'

			if self.metric is None:

				function = inner_norm
				gradient_analytical = gradient_inner_norm

				def wrapper_function(out,*operands):
					return out/operands[0].shape[-1]/2

				def wrapper_gradient(out,*operands):
					return out/operands[0].shape[-1]/2

			elif self.metric in ['norm']:

				function = inner_norm
				gradient_analytical = gradient_inner_norm

				def wrapper_function(out,*operands):
					return out/operands[0].shape[-1]/2
				
				def wrapper_gradient(out,*operands):
					return out/operands[0].shape[-1]/2

			elif self.metric in ['abs2']:

				function = inner_abs2
				gradient_analytical = gradient_inner_abs2

				def wrapper_function(out,*operands):
					return 1 - out/(operands[0].shape[-1]*operands[0].shape[-2])

				def wrapper_gradient(out,*operands):
					return - out/(operands[0].shape[-1]*operands[0].shape[-2])

			elif self.metric in ['real']:

				function = inner_real
				gradient_analytical = gradient_inner_real

				def wrapper_function(out,*operands):
					return 1 - out

				def wrapper_gradient(out,*operands):
					return  - out

			elif self.metric in ['imag']:

				function = inner_imag
				gradient_analytical = gradient_inner_imag

				def wrapper_function(out,*operands):
					return 1 - out

				def wrapper_gradient(out,*operands):
					return - out

			else:

				function = inner_norm
				gradient_analytical = gradient_inner_norm

				def wrapper_function(out,*operands):
					return out/operands[0].shape[-1]/2

				def wrapper_gradient(out,*operands):
					return out/operands[0].shape[-1]/2


			shapes_function = (*self.shapes,) if self.shapes else ()
			optimize_function = self.optimize
			wrapper_function = jit(wrapper_function)

			shapes_gradient = (*self.shapes,(self.size**2,*self.shapes[0])) if self.shapes else ()
			optimize_gradient = self.optimize
			wrapper_gradient = jit(wrapper_gradient)

			if shapes_function:
				function = function(*shapes_function,optimize=optimize_function,wrapper=wrapper_function)
			else:
				function = partial(function,optimize=optimize_gradient,wrapper=wrapper_function)

			if shapes_gradient:
				gradient_analytical = gradient_analytical(*shapes_gradient,optimize=optimize_function,wrapper=wrapper_gradient)
			else:
				gradient_analytical = partial(gradient_analytical,optimize=optimize_gradient,wrapper=wrapper_gradient)

			grad = gradient_analytical
			# grad = gradient(function,mode='fwd',holomorphic=True,move=True)

			function = jit(function)
			grad = jit(grad)
			gradient_analytical = jit(gradient_analytical)

			if self.label is not None:

				label = conj(self.label)

				def function(*operands,function=function,label=label):
					return function(*operands[:1],label,*operands[1:])
				def grad(*operands,function=grad,label=label):
					return function(*operands[:1],label,*operands[1:])				
				def gradient_analytical(*operands,function=gradient_analytical,label=label):
					return function(*operands[:1],label,*operands[1:])

		function = jit(function)
		grad = jit(grad)
		gradient_analytical = jit(gradient_analytical)

		self.function = function
		self.gradient = grad
		self.gradient_analytical = gradient_analytical

		return



class Optimization(System):
	'''
	Base Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		model (object): model instance
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={},system=None,**kwargs):

		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		defaults = {
			'optimizer':None,
			'search':{'alpha':'line_search','beta':None},
			'eps':{'value':1e-4,'grad':1e-12,'alpha':1e-12,'beta':1e3},
			'alpha':0,
			'status':1,
			'clear':True,
			'cwd':None,
			'path':None,
			'modulo':{'log':None,'attributes':None,'callback':None,'restart':None,'dump':None},
			'length':{'log':None,'attributes':10,'callback':None,'restart':None,'dump':None},
			'attributes':{'iteration':[],'parameters':[],'value':[],'grad':[],'search':[],'alpha':[]},	
			'track':{},		
		}

		setter(hyperparameters,defaults,delimiter=delim,func=False)

		self.hyperparameters = hyperparameters

		self.value_and_grad,self.func,self.grad = value_and_gradient(func,grad,returns=True)

		self.alpha = LineSearch(self.func,self.grad,self.hyperparameters)

		if callback is None:
			def callback(parameters,track,optimizer):
				status = True
				return status
		self.callback = callback

		self.size = 0
		self.iteration = 0
		self.parameters = None
		self.optimizer = hyperparameters['optimizer']		
		self.status = hyperparameters['status']
		self.clear = hyperparameters['clear']
		self.modulo = hyperparameters['modulo']
		self.length = hyperparameters['length']
		self.attributes = hyperparameters['attributes']
		self.track = hyperparameters['track']
		self.iterations = range(int(hyperparameters['iterations']))
		self.sizes = hyperparameters['length'].get('attributes')		
		self.search = hyperparameters['search']
		self.eps = hyperparameters['eps']

		self.reset(clear=True)

		return

	def __call__(self,parameters):
		'''
		Iterate optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
		Returns:
			parameters (object): optimizer parameters
		'''

		iteration = self.iteration
		state = self.opt_init(parameters)
		iteration,state = self.load(iteration,state)
		
		self.info()

		for iteration in self.iterations:

			state = self.opt_update(iteration,state)

			self.dump(iteration,state)

			if not self.status:
				break

		parameters = self.get_params(state)

		self.parameters = parameters

		return parameters

	def opt_init(self,parameters):
		'''
		Initialize optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
		Returns:
			state (object): optimizer state
		'''
		state = parameters
		return state

	def opt_update(self,iteration,state):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			state (object): optimizer state
		'''

		def update(parameters,value,grad,search,optimizer):
			alpha = optimizer.hyperparameters['alpha']
			search = -grad
			parameters = parameters + alpha*search
			return parameters,search,alpha

		value,grad,parameters = self.opt_step(iteration,state)
		search = self.attributes['search'][-1] if self.size > 1 else 0

		parameters,search,alpha = update(parameters,value,grad,search,self)

		self.attributes['search'].append(search)
		self.attributes['alpha'].append(alpha)

		state = self.opt_init(parameters)
		parameters = self.get_params(state)
		track = self.track		
		optimizer = self
		self.status = self.callback(parameters,track,optimizer)

		return state

	def get_params(self,state):
		'''
		Get optimizer parameters with optimizer state
		Args:
			state (object): optimizer state
		Returns:
			parameters (object): optimizer parameters
		'''
		parameters = state
		return parameters

	def opt_step(self,iteration,state):
		'''
		Iterate optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			value (object): optimizer value
			grad (object): optimizer grad
		'''

		parameters = self.get_params(state)
		value,grad = self.value_and_grad(parameters)
		size = self.size

		if (self.sizes is not None) and (self.size > 0) and (self.size >= self.sizes):
			for attr in self.attributes:
				if self.attributes[attr]:
					self.attributes[attr].pop(min(self.size-2,0)+1)


		iteration += 1
		size += 1

		self.attributes['iteration'].append(iteration)
		self.attributes['parameters'].append(parameters)
		self.attributes['value'].append(value)
		self.attributes['grad'].append(grad)

		self.iteration = iteration
		self.size = size

		return value,grad,parameters


	def dump(self,iteration=None,state=None):
		'''
		Dump data
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		'''

		do = (self.path is not None) or (self.modulo['dump'] is None) or (iteration is None) or (iteration%self.modulo['dump'] == 0)

		if not do:
			return

		path = join(self.path,root=self.cwd)
		data = self.track
		dump(data,path)

		path = join(self.path,ext='ckpt',root=self.cwd)
		data = self.attributes
		dump(data,path)

		return

	def load(self,iteration=None,state=None):
		'''
		Load data
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			iteration (int): optimizer iteration
			state (object): optimizer state
		'''

		do = (self.path is not None)

		if not do:
			return iteration,state

		path = join(self.path,root=self.cwd)
		data = load(path)

		if data is not None:
			for attr in data:
				if attr in self.track:
					self.track[attr].extend(data[attr])


		path = join(self.path,ext='ckpt',root=self.cwd)
		data = load(path)

		if data is not None:
			for attr in data:
				if attr in self.attributes:
					self.attributes[attr].extend(data[attr])
				

		self.parameters = self.get_params(state)
		self.reset(clear=False)

		iteration = self.iteration
		state = self.opt_init(self.parameters)

		return iteration,state
		

	def reset(self,clear=None):
		'''
		Reset class attributes
		Args:
			clear (bool): clear attributes
		'''
		clear = self.clear if clear is None else clear

		for attr in list(self.attributes):
			value = self.attributes[attr]
			if ((not isinstance(value,list)) and (not value)):
				self.attributes.pop(attr)
			elif ((not isinstance(value,list)) and (value)) or clear:
				self.attributes[attr] = []

		for attr in list(self.track):
			value = self.track[attr]
			if ((not isinstance(value,list)) and (not value)):
				self.track.pop(attr)
			elif ((not isinstance(value,list)) and (value)) or clear:
				self.track[attr] = []
		
		self.size = min((len(self.attributes[attr]) for attr in self.attributes),default=self.size)

		while (self.sizes is not None) and (self.size > 0) and (self.size > self.sizes):
			for attr in self.attributes:
				if self.attributes[attr]:
					self.attributes[attr].pop(min(self.size-2,0)+1)
			self.size = min((len(self.attributes[attr]) for attr in self.attributes),default=self.size)

		if self.size:
			attr = 'iteration'	
			if attr in self.attributes:
				self.iteration = self.attributes[attr][-1]

			attr = 'parameters'		
			if attr in self.attributes:
				self.parameters = self.attributes[attr][-1]
		else:
			self.iteration = 0
	
		self.iterations = range(
			self.iteration,
			self.iterations.stop-self.iterations.start+self.iteration,
			self.iterations.step)				


		return

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		
		msg = '%s'%('\n'.join([
			*['%s: %s'%(attr,getattr(self,attr)) 
				for attr in ['optimizer','iterations','size','search','eps','modulo']
			],
			*['%s: %s'%(attr,{key: getattr(self,attr).get(key,[None])[-1] if isinstance(getattr(self,attr).get(key,[None])[-1],scalars) else ['...'] for key in getattr(self,attr)})
				for attr in ['track','attributes']
				if any(getattr(self,attr).get(key) for key in getattr(self,attr))
			],			
			]
			))
		self.log(msg,verbose=verbose)
		return

class Optimizer(Optimization):
	'''
	Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
	'''
	def __new__(cls,func,grad=None,callback=None,hyperparameters={},system=None):
	
		defaults = {'optimizer':None}
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		optimizers = {'adam':Adam,'cg':ConjugateGradient,'gd':GradientDescent,None:GradientDescent}

		optimizer = hyperparameters['optimizer']		
		
		self = optimizers.get(optimizer,optimizers[None])(func,grad,callback,hyperparameters=hyperparameters,system=system)

		return self
	

class GradientDescent(Optimization):
	'''
	Gradient Descent Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={},system=None,**kwargs):

		defaults = {'track':{'beta':False},'attributes':{'beta':False}}		
		setter(hyperparameters,defaults,delimiter=delim,func=True)

		super().__init__(func,grad,callback,hyperparameters=hyperparameters,system=system,**kwargs)

		defaults = {}
		setter(self.hyperparameters,defaults,delimiter=delim,func=False)

		return

	def opt_update(self,iteration,state):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			state (object): optimizer state
		'''

		def update(parameters,value,grad,search,optimizer):
			alpha = optimizer.hyperparameters['alpha']
			search = -grad
			parameters = parameters + alpha*search
			return parameters,search,alpha

		value,grad,parameters = self.opt_step(iteration,state)
		search = self.attributes['search'][-1] if self.size > 1 else 0

		parameters,search,alpha = update(parameters,value,grad,search,self)

		self.attributes['search'].append(search)
		self.attributes['alpha'].append(alpha)

		state = self.opt_init(parameters)
		parameters = self.get_params(state)
		track = self.track		
		optimizer = self
		self.status = self.callback(parameters,track,optimizer)

		return state


class ConjugateGradient(Optimization):
	'''
	Conjugate Gradient Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={},system=None,**kwargs):

		defaults = {'beta':0,'search':{'beta':None},'attributes':{'beta':[]}}
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		super().__init__(func,grad,callback,hyperparameters=hyperparameters,system=system,**kwargs)

		defaults = {}
		setter(self.hyperparameters,defaults,delimiter=delim,func=False)

		self.beta = GradSearch(self.func,self.grad,self.hyperparameters)

		return

	def opt_update(self,iteration,state):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			state (object): optimizer state
		'''

		def update(parameters,value,grad,search,optimizer):
			alpha = optimizer.alpha(
				parameters,
				optimizer.attributes['alpha'],
				optimizer.attributes['value'],
				optimizer.attributes['grad'],
				optimizer.attributes['search'])

			parameters = parameters + alpha*search

			state = optimizer.opt_init(parameters)

			_value,_grad,parameters = optimizer.opt_step(iteration,state)
			
			beta = optimizer.beta(
				parameters,
				optimizer.attributes['beta'],
				optimizer.attributes['value'],
				optimizer.attributes['grad'],
				optimizer.attributes['search'])

			search = -_grad + beta*search

			return parameters,search,alpha,beta


		steps = self.size == 0

		for step in range(steps+1):

			init = self.size == 0

			if not init:
				parameters = self.get_params(state)
				value = self.attributes['value'][-1]
				grad = self.attributes['grad'][-1]
				search = self.attributes['search'][-1]

				parameters,search,alpha,beta = update(parameters,value,grad,search,self)

			else:
				value,grad,parameters = self.opt_step(iteration-init,state)

				alpha = self.hyperparameters['alpha']
				beta = self.hyperparameters['beta']
				search = -grad

			self.attributes['search'].append(search)
			self.attributes['alpha'].append(alpha)
			self.attributes['beta'].append(beta)
		
			state = self.opt_init(parameters)
			parameters = self.get_params(state)
			track = self.track
			optimizer = self
			self.status = self.callback(parameters,track,optimizer)

		return state


class Adam(Optimization):
	'''
	Adam Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={},system=None,**kwargs):

		defaults = {'track':{'beta':False},'attributes':{'beta':False}}		
		setter(hyperparameters,defaults,delimiter=delim,func=True)

		super().__init__(func,grad,callback,hyperparameters=hyperparameters,system=system,**kwargs)

		defaults = {}
		setter(self.hyperparameters,defaults,delimiter=delim,func=False)

		self._optimizer = getattr(jax.example_libraries.optimizers,self.optimizer)

		self._opt_init,self._opt_update,self._get_params = self._optimizer(self.hyperparameters['alpha'])

		return

	def opt_init(self,parameters):
		'''
		Initialize optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
		Returns:
			state (object): optimizer state
		'''

		state = self._opt_init(parameters)

		return state

	def opt_update(self,iteration,state):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			state (object): optimizer state
		'''

		def update(parameters,value,grad,search,optimizer):

			alpha = optimizer.hyperparameters['alpha']
			search = -grad
			
			state = self.opt_init(parameters)
			state = self._opt_update(iteration,grad,state)
			parameters = self.get_params(state)			

			return parameters,search,alpha

		value,grad,parameters = self.opt_step(iteration,state)
		search = self.attributes['search'][-1] if self.size > 1 else 0

		parameters,search,alpha = update(parameters,value,grad,search,self)

		self.attributes['alpha'].append(alpha)
		self.attributes['search'].append(search)

		state = self.opt_init(parameters)
		parameters = self.get_params(state)
		track = self.track		
		optimizer = self
		self.status = self.callback(parameters,track,optimizer)

		return state

	def get_params(self,state):
		'''
		Get optimizer parameters with optimizer state
		Args:
			state (object): optimizer state
		Returns:
			parameters (object): optimizer parameters
		'''

		parameters = self._get_params(state)

		return parameters