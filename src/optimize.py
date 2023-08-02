#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))


# Import user modules
from src.utils import jit,partial,copy,value_and_gradient,gradient,hessian,abs,dot,lstsq,inv,norm,einsum
from src.utils import metrics,contraction,gradient_contraction,optimizer_libraries
from src.utils import is_unitary,is_hermitian,is_naninf
from src.utils import scalars,delim,nan

from src.iterables import setter,getattrs,hasattrs,iterate

from src.line_search import line_search,armijo

from src.io import dump,load,join,split,exists

from src.system import System,Dict


# Logging
from src.logger	import Logger
logger = Logger()


class LineSearcher(System):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function			
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		
		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		if hyperparameters is None:
			hyperparameters = {}
		defaults = {}		
		returns = ['alpha']
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		arguments = () if arguments is None else arguments
		keywords = {} if keywords is None else keywords

		self.func = func
		self.grad = grad
		self.arguments = arguments
		self.keywords = keywords
		self.hyperparameters = hyperparameters
		self.system = system
		self.defaults = defaults
		self.returns = returns

		return

	def __call__(self,iteration,parameters,alpha,value,grad,search,*args,**kwargs):
		'''
		Perform line search
		Args:
			iteration (int): Objective iteration
			parameters (array): Objective parameters
			alpha (iterable[array]): Previous alpha
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			alpha (array): Returned search value
		'''
		
		_alpha = alpha[-1]
		returns = (_alpha,)

		returns = self.__callback__(returns,iteration,parameters,alpha,value,grad,search,*args,**kwargs)

		attr = 'alpha'
		alpha = returns[attr]

		return alpha

	def __callback__(self,returns,iteration,parameters,alpha,value,grad,search,*args,**kwargs):
		'''
		Check return values of line search
		Args:
			returns (iterable): Iterable of returned values of line search
			iteration (int): Objective iteration			
			parameters (array): Objective parameters
			alpha (iterable[array]): Previous alpha
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function				
		Returns:
			returns (dict): Dictionary of returned values of line search
		'''
		returns = dict(zip(self.returns,returns))

		attr = 'alpha'
		if (returns[attr] is None) or (is_naninf(returns[attr])) or (returns[attr] < self.hyperparameters['bounds'][attr][0]) or (returns[attr] > self.hyperparameters['bounds'][attr][1]):		
			if len(alpha) > 1:
				returns[attr] = alpha[-1]#*dot(grad[-1],search[-1])/dot(grad[-2],search[-2])
			else:
				returns[attr] = alpha[-1]		
		elif (self.hyperparameters['modulo'].get(attr) is not None) and ((iteration+1)%(self.hyperparameters['modulo'][attr]) == 0):
			if len(alpha) > 1:
				returns[attr] = alpha[-1]
			else:
				returns[attr] = self.hyperparameters.get(attr,alpha[-1])

		return returns

class LineSearch(LineSearcher):
	'''
	Line Search class
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		arguments (iterable[object]): Position arguments for function
		keywords (dict[str,object]): Keyword arguments for function		
		hyperparameters (dict): line search hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
		kwargs (dict): Additional system attributes
	'''
	def __new__(cls,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		
		hyperparameters = {} if hyperparameters is None else hyperparameters

		defaults = {'search':{'alpha':None}}
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		line_searches = {
			'line_search':Line_Search,
			'armijo':Armijo,
			None:Null_Line_Search
			}

		line_search = hyperparameters.get('search',{}).get('alpha',defaults['search']['alpha'])
		
		self = line_searches.get(line_search,line_searches[None])(func,grad,
			arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,
			**kwargs)

		return self


class Line_Search(LineSearcher):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function					
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''

		hyperparameters = {} if hyperparameters is None else hyperparameters

		defaults = {'c1':0.0001,'c2':0.9,'maxiter':10,'old_old_fval':None,'args':()}
		returns = ['alpha','nfunc','ngrad','value','_value','slope']
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		super().__init__(func,grad,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		self.defaults = defaults
		self.returns = returns

		return

	def __call__(self,iteration,parameters,alpha,value,grad,search,*args,**kwargs):
		'''
		Perform line search
		Args:
			iteration (int): Objective iteration
			parameters (array): Objective parameters
			alpha (iterable[array]): Previous alpha
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			alpha (array): Returned search value
		'''

		self.defaults['args'] = (*args,*(kwargs[kwarg] for kwarg in kwargs))
		self.defaults['old_old_fval'] = value[-2] if len(value)>1 else None
		
		returns = line_search(self.func,self.grad,
			parameters,search[-1],grad[-1],value[-1],
			**self.defaults)
		
		returns = self.__callback__(returns,iteration,parameters,alpha,value,grad,search,*args,**kwargs)

		attr = 'alpha'
		alpha = returns[attr]

		return alpha


class Armijo(LineSearcher):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function				
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		
		hyperparameters = {} if hyperparameters is None else hyperparameters

		defaults = {'c1':0.0001,'alpha0':hyperparameters.get('alpha',1e-4),'args':()}
		returns = ['alpha','nfunc','value']
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		super().__init__(func,grad,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		self.defaults = defaults
		self.returns = returns

		return

	def __call__(self,iteration,parameters,alpha,value,grad,search,*args,**kwargs):
		'''
		Perform line search
		Args:
			iteration (int): Objective iteration
			parameters (array): Objective parameters
			alpha (iterable[array]): Previous alpha
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			alpha (array): Returned search value
		'''

		self.defaults['args'] = (*args,*(kwargs[kwarg] for kwarg in kwargs))

		returns = armijo(self.func,parameters,search[-1],grad[-1],value[-1],**self.defaults)

		returns = self.__callback__(returns,iteration,parameters,alpha,value,grad,search,*args,**kwargs)

		attr = 'alpha'
		alpha = returns[attr]

		return alpha

class Null_Line_Search(LineSearcher):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function	
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)			
			kwargs (dict): Additional system attributes
		'''

		super().__init__(func,grad,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		return



class GradSearcher(System):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Grad search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function						
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		
		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		hyperparameters = {} if hyperparameters is None else hyperparameters

		defaults = {}		
		returns = ['beta']
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		arguments = () if arguments is None else arguments
		keywords = {} if keywords is None else keywords

		self.func = func
		self.grad = grad
		self.arguments = arguments
		self.keywords = keywords
		self.hyperparameters = hyperparameters
		self.system = system
		self.defaults = defaults
		self.returns = returns
		
		return

	def __call__(self,iteration,parameters,beta,value,grad,search,*args,**kwargs):
		'''
		Perform grad search
		Args:
			iteration (int): Objective iteration
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			beta (array): Returned search value
		'''

		_beta = beta[-1]
		returns = (_beta,)

		returns = self.__callback__(returns,iteration,parameters,beta,value,grad,search,*args,**kwargs)

		attr = 'beta'
		beta = returns[attr]

		return beta

	def __callback__(self,returns,iteration,parameters,beta,value,grad,search,*args,**kwargs):
		'''
		Check return values of grad search
		Args:
			returns (iterable): Iterable of returned values of grad search
			iteration (int): Objective iteration			
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function				
		Returns:
			returns (dict): Dictionary of returned values of grad search
		'''
		
		returns = dict(zip(self.returns,returns))

		attr = 'beta'
		if (returns[attr] is None) or (is_naninf(returns[attr])) or (returns[attr] < self.hyperparameters['bounds'][attr][0]) or (returns[attr] > self.hyperparameters['bounds'][attr][1]):
			if len(beta) > 1:
				returns[attr] = 0
			else:
				returns[attr] = beta[0]
		elif (self.hyperparameters['eps'].get('grad.dot') is not None) and (len(grad)>1) and ((abs(dot(grad[-1],grad[-2]))/(dot(grad[-1],grad[-1]))) >= self.hyperparameters['eps']['grad.dot']):
			returns[attr] = 0			
		elif (self.hyperparameters['modulo'].get(attr) is not None) and ((iteration+1)%(self.hyperparameters['modulo'][attr]) == 0):
			if len(beta) > 1:
				returns[attr] = beta[0]
			else:
				returns[attr] = self.hyperparameters.get(attr,beta[-1])

		return returns

class GradSearch(GradSearcher):
	'''
	Grad Search class
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		arguments (iterable[object]): Position arguments for function
		keywords (dict[str,object]): Keyword arguments for function		
		hyperparameters (dict): grad search hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
		kwargs (dict): Additional system attributes		
	'''
	def __new__(cls,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
	
		hyperparameters = {} if hyperparameters is None else hyperparameters

		defaults = {'search':{'beta':None}}
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		grad_searches = {
			'grad_search':Fletcher_Reeves,'fletcher_reeves':Fletcher_Reeves,
			'polak_ribiere':Polak_Ribiere,'polak_ribiere_fletcher_reeves':Polak_Ribiere_Fletcher_Reeves,
			'hestenes_stiefel':Hestenes_Stiefel,'dai_yuan':Dai_Yuan,'hager_zhang':Hager_Zhang,
			None:Null_Grad_Search,
			}

		grad_search = hyperparameters.get('search',{}).get('beta',defaults['search']['beta'])
		
		self = grad_searches.get(grad_search,grad_searches[None])(func,grad,
			arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,
			**kwargs)

		return self


class Fletcher_Reeves(GradSearcher):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function			
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,iteration,parameters,beta,value,grad,search,*args,**kwargs):
		'''
		Perform grad search
		Args:
			iteration (int): Objective iteration
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			beta (array): Returned search value
		'''

		_beta = (dot(grad[-1],grad[-1]))/(dot(grad[-2],grad[-2])) # Fletcher-Reeves
		returns = (_beta,)

		returns = self.__callback__(returns,iteration,parameters,beta,value,grad,search,*args,**kwargs)

		attr = 'beta'
		beta = returns[attr]

		return beta


class Polak_Ribiere(GradSearcher):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function			
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,iteration,parameters,beta,value,grad,search,*args,**kwargs):
		'''
		Perform grad search
		Args:
			iteration (int): Objective iteration
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			beta (array): Returned search value
		'''

		_beta = max(0,(dot(grad[-1],grad[-1]-grad[-2]))/(dot(grad[-2],grad[-2])))  # Polak-Ribiere
		returns = (_beta,)

		returns = self.__callback__(returns,iteration,parameters,beta,value,grad,search,*args,**kwargs)

		attr = 'beta'
		beta = returns[attr]

		return beta


class Polak_Ribiere_Fletcher_Reeves(GradSearcher):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function			
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,iteration,parameters,beta,value,grad,search,*args,**kwargs):
		'''
		Perform grad search
		Args:
			iteration (int): Objective iteration
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			beta (array): Returned search value
		'''

		_beta = [(dot(grad[-1],grad[-1]))/(dot(grad[-2],grad[-2])),max(0,(dot(grad[-1],grad[-1]-grad[-2]))/(dot(grad[-2],grad[-2])))] # Polak-Ribiere-Fletcher-Reeves
		_beta = -_beta[0] if _beta[1] < -_beta[0] else _beta[1] if abs(_beta[1]) <= _beta[0] else _beta[0]
		returns = (_beta,)

		returns = self.__callback__(returns,iteration,parameters,beta,value,grad,search,*args,**kwargs)

		attr = 'beta'
		beta = returns[attr]

		return beta


class Hestenes_Stiefel(GradSearcher):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function			
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,iteration,parameters,beta,value,grad,search,*args,**kwargs):
		'''
		Perform grad search
		Args:
			iteration (int): Objective iteration
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			beta (array): Returned search value
		'''

		_beta = (dot(grad[-1],grad[-1]-grad[-2]))/(dot(search[-1],grad[-1]-grad[-2])) # Hestenes-Stiefel
		returns = (_beta,)

		returns = self.__callback__(returns,iteration,parameters,beta,value,grad,search,*args,**kwargs)

		attr = 'beta'
		beta = returns[attr]

		return beta

class Dai_Yuan(GradSearcher):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function			
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,iteration,parameters,beta,value,grad,search,*args,**kwargs):
		'''
		Perform grad search
		Args:
			iteration (int): Objective iteration
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			beta (array): Returned search value
		'''

		_beta = (dot(grad[-1],grad[-1]))/(dot(search[-1],grad[-1]-grad[-2])) # Dai-Yuan https://doi.org/10.1137/S1052623497318992
		returns = (_beta,)

		returns = self.__callback__(returns,iteration,parameters,beta,value,grad,search,*args,**kwargs)

		attr = 'beta'
		beta = returns[attr]

		return beta

class Hager_Zhang(GradSearcher):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function		
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		super().__init__(func,grad,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)
		return

	def __call__(self,iteration,parameters,beta,value,grad,search,*args,**kwargs):
		'''
		Perform grad search
		Args:
			iteration (int): Objective iteration
			parameters (array): Objective parameters
			beta (iterable[array]): Previous beta
			value (iterable[array]): Previous objective values
			grad (iterable[array]): Previous objective gradients
			search (iterable[array]): Previous objective search directions
			args (iterable[object]): Position arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			beta (array): Returned search value
		'''

		_beta = grad[-1]-grad[-2]
		_beta = dot((_beta - 2*((dot(_beta,_beta))/(dot(_beta,search[-1])))*search[-1]),grad[-1]/(dot(_beta,search[-1]))) # Hager-Zhang https://doi.org/10.1137/030601880
		returns = (_beta,)

		returns = self.__callback__(returns,iteration,parameters,beta,value,grad,search,*args,**kwargs)

		attr = 'beta'
		beta = returns[attr]

		return beta


class Null_Grad_Search(GradSearcher):
	def __init__(self,func,grad,arguments=None,keywords=None,hyperparameters=None,system=None,**kwargs):
		'''	
		Grad search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function			
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)			
			kwargs (dict): Additional system attributes
		'''

		super().__init__(func,grad,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		return


class Function(System):

	def __init__(self,model,func=None,grad=None,callback=None,metric=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):
		'''	
		Class for function
		Args:
			model (object): Model instance
			func (callable,iterable[callable]): Function function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			callback (callable): Callback of function with signature callback(parameters,track,optimizer,model,metric,func,grad)
			metric (str,callable): Function metric with signature metric(*operands)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function
			hyperparameters (dict): Function hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		if hyperparameters is not None and system is not None:
			kwargs.update({attr: hyperparameters.get(attr) for attr in (system if system is not None else ()) if attr in hyperparameters})

		setter(kwargs,system,delimiter=delim,func=False)
		
		super().__init__(**kwargs)

		if func is None:
			func = []
		if not callable(func):
			func = [i for i in func if i is not None]
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
			grad = [i for i in grad if i is not None]
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

		arguments = () if arguments is None else arguments
		keywords = {} if keywords is None else keywords

		if callback is None:
			def callback(parameters,track,optimizer,model,metric,func,grad):
				status = True
				return status

		self.value_and_gradient,self.function,self.gradient = value_and_gradient(function,gradient,returns=True)

		self.model = model
		self.callback = callback
		self.metric = metric
		self.arguments = arguments
		self.keywords = keywords
		self.hyperparameters = hyperparameters
		self.system = system

		return

	# @partial(jit,static_argnums=(0,))
	def __call__(self,parameters,*args,**kwargs):
		'''
		Function call
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function
		Returns:
			out (object): Return of function
		'''
		return self.function(parameters,*args,**kwargs)

	# @partial(jit,static_argnums=(0,))
	def func(self,parameters,*args,**kwargs):
		'''
		Function call
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function			
		Returns:
			out (object): Return of function
		'''
		return self.__call__(parameters,*args,**kwargs)


	# @partial(jit,static_argnums=(0,))
	def grad(self,parameters,*args,**kwargs):
		'''
		Gradient call
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function
		Returns:
			out (object): Return of function
		'''
		return self.gradient(parameters,*args,**kwargs)

	# @partial(jit,static_argnums=(0,))
	def value_and_grad(self,parameters,*args,**kwargs):
		'''
		Function and gradient call
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function			
		Returns:
			out (object): Return of function
		'''
		return self.value_and_gradient(parameters,*args,**kwargs)

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



class Objective(Function):		
	def __init__(self,model,metric,func=None,grad=None,callback=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):
		'''	
		Objective class for metric + function
		Args:
			model (object): Model instance
			metric (str,callable): Objective metric with signature metric(*operands)
			func (callable,iterable[callable]): Objective function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			callback (callable): Callback of function with signature callback(parameters,track,optimizer,model,metric,func,grad)			
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function
			hyperparameters (dict): Objective hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''

		super().__init__(model,func=func,grad=grad,callback=callback,metric=metric,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		grad_automatic = gradient(self,mode='rev',move=True)
		grad_finite = gradient(self,mode='finite',move=True)
		def grad_analytical(parameters):
			return self.metric.grad_analytical(self.model(parameters),self.model.grad_analytical(parameters)) + self.gradient(parameters)	

		self.gradient_automatic = grad_automatic
		self.gradient_finite = grad_finite
		self.gradient_analytical = grad_analytical

		return

	# @partial(jit,static_argnums=(0,))
	def __call__(self,parameters,*args,**kwargs):
		'''
		Function call
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function	
		Returns:
			out (object): Return of function
		'''
		return self.func(parameters,*args,**kwargs) + self.function(parameters,*args,**kwargs)

	# @partial(jit,static_argnums=(0,))
	def func(self,parameters,*args,**kwargs):
		'''
		Function call
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function		
		Returns:
			out (object): Return of function
		'''
		return self.metric(self.model(parameters,*args,**kwargs))

	# @partial(jit,static_argnums=(0,))
	def grad(self,parameters,*args,**kwargs):
		'''
		Gradient call
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function			
		Returns:
			out (object): Return of function
		'''
		return self.metric.grad(self.model(parameters,*args,**kwargs),self.model.grad(parameters,*args,**kwargs)) + self.gradient(parameters,*args,**kwargs)	

	# @partial(jit,static_argnums=(0,))
	def grad_automatic(self,parameters,*args,**kwargs):
		'''
		Gradient call
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function		
		Returns:
			out (object): Return of function
		'''
		return self.gradient_automatic(parameters,*args,**kwargs)

	# @partial(jit,static_argnums=(0,))
	def grad_analytical(self,parameters,*args,**kwargs):
		'''
		Gradient call
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function			
		Returns:
			out (object): Return of function
		'''
		return self.metric.grad_analytical(self.model(parameters,*args,**kwargs),self.model.grad_analytical(parameters,*args,**kwargs)) + self.gradient(parameters,*args,**kwargs)	

	# @partial(jit,static_argnums=(0,))
	def grad_finite(self,parameters,*args,**kwargs):
		'''
		Gradient call
		Args:
			parameters (array): parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function		
		Returns:
			out (object): Return of function
		'''
		return self.gradient_finite(parameters,*args,**kwargs)


class Callback(Function):

	def __init__(self,model,callback,func=None,grad=None,metric=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):
		'''	
		Class for function
		Args:
			model (object): Model instance
			callback (callable): Callback of function with signature callback(parameters,track,optimizer,model,metric,func,grad)
			func (callable,iterable[callable]): Function function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			metric (str,callable): Callback metric with signature metric(*operands)
			arguments (iterable[object]): Position arguments for function
			keywords (dict[str,object]): Keyword arguments for function
			hyperparameters (dict): Callback hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		'''
		
		super().__init__(model,func=func,grad=grad,callback=callback,metric=metric,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

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
	def __init__(self,metric=None,shapes=None,model=None,state=None,label=None,weights=None,optimize=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):
		'''
		Metric class for distance between operands
		Args:
			metric (str,Metric): Type of metric
			shapes (iterable[tuple[int]]): Shapes of Operators
			model (object): Model instance	
			state (array,callable): State			
			label (array,callable): Label			
			weights (array): Weights
			optimize (bool,str,iterable): Contraction type	
			arguments (iterable[object]): Position arguments for metric
			keywords (dict[str,object]): Keyword arguments for metric			
			hyperparameters (dict): Metric hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
			kwargs (dict): Additional system attributes
		'''
		if hyperparameters is not None and system is not None:
			kwargs.update({attr: hyperparameters.get(attr) for attr in (system if system is not None else ()) if attr in hyperparameters})

		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		self.metric = hyperparameters.get('metric',metric) if metric is None else metric
		self.state = hyperparameters.get('state',state) if state is None else state
		self.label = hyperparameters.get('label',label) if label is None else label
		self.weights = hyperparameters.get('weights',weights) if weights is None else weights
		self.shapes = getattr(label,'shape') if shapes is None else shapes
		self.model = model
		self.optimize = optimize
		self.arguments = arguments if arguments is not None else ()
		self.keywords = keywords if keywords is not None else {}
		self.hyperparameters = hyperparameters
		self.system = system

		self.string = str(self.metric)

		self.__setup__()

		return

	def __setup__(self,metric=None,shapes=None,model=None,state=None,label=None,weights=None,optimize=None):
		'''
		Setup metric attributes metric,string
		Args:
			metric (str,Metric): Type of metric
			shapes (iterable[tuple[int]]): Shapes of Operators
			model (object): Model instance	
			state (array,callable): State			
			label (array,callable): Label			
			weights (array): Weights
			optimize (bool,str,iterable): Contraction type	
		'''

		self.metric = self.metric if metric is None else metric
		self.shapes = self.shapes if shapes is None else shapes
		self.model = self.model if model is None else model
		self.state = self.state if state is None else state
		self.label = self.label if label is None else label
		self.weights = self.weights if weights is None else weights
		self.optimize = self.optimize if optimize is None else optimize

		if isinstance(self.metric,Metric):
			self.metric = self.metric.metric
		if self.metric is None:
			self.metric = None
		if self.shapes is None:
			self.shapes = ()

		self.__initialize__()

		self.info()

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
		return self.gradient(*operands)

	# @partial(jit,static_argnums=(0,))
	def grad_automatic(self,*operands):
		'''
		Gradient call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.gradient_automatic(*operands)

	# @partial(jit,static_argnums=(0,))
	def grad_analytical(self,*operands):
		'''
		Gradient call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.gradient_analytical(*operands)

	# @partial(jit,static_argnums=(0,))
	def grad_finite(self,*operands):
		'''
		Gradient call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.gradient_finite(*operands)

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		

		msg = []

		for attr in ['metric']:
			string = '%s %s: %s'%(self.__class__.__name__,attr,getattr(self,attr))
			msg.append(string)

		msg = '\n'.join(msg)

		self.log(msg,verbose=verbose)
		return


	def __initialize__(self,metric=None,shapes=None,model=None,state=None,label=None,weights=None,optimize=None):
		'''
		Setup metric
		Args:
			metric (str,Metric): Type of metric
			shapes (iterable[tuple[int]]): Shapes of Operators
			model (object): Model instance	
			state (array,callable): State			
			label (array,callable): Label			
			weights (array): Weights
			optimize (bool,str,iterable): Contraction type	
		'''

		self.metric = self.metric if metric is None else metric
		self.shapes = self.shapes if shapes is None else shapes
		self.model = self.model if model is None else model
		self.state = self.state if state is None else state
		self.label = self.label if label is None else label
		self.weights = self.weights if weights is None else weights
		self.optimize = self.optimize if optimize is None else optimize

		if callable(self.label):
			label = self.label()
		else:
			label = self.label

		if isinstance(self.metric,str):

			if label is None:
				pass
			elif label.ndim == 1:
				if self.metric in ['real','imag','norm','abs2']:
					self.metric = 'abs2'
			elif label.ndim == 2:
				if is_unitary(label) and self.metric in ['real','imag','norm','abs2']:
					self.metric = 'abs2'
				elif is_hermitian(label) and self.metric in ['real','imag','norm','abs2']:
					self.metric = 'real'

		if label is not None:
			if all(isinstance(i,int) for i in self.shapes) or (len(self.shapes) == 1):
				self.shapes = label.shape
			else:
				self.shapes = [label.shape]*len(self.shapes)
		
		if all(isinstance(i,int) for i in self.shapes) or (len(self.shapes) == 1):
			self.shapes = [self.shapes,]*2

		func,grad,grad_analytical = metrics(
			metric=self.metric,shapes=self.shapes,
			label=self.label,weights=self.weights,
			optimize=self.optimize,
			returns=True)

		grad_automatic = gradient(self,mode='fwd',move=True)
		grad_finite = gradient(self,mode='finite',move=True)

		self.function = func
		self.gradient = grad
		self.gradient_automatic = grad_automatic
		self.gradient_finite = grad_finite
		self.gradient_analytical = grad_analytical

		return



class Optimization(System):
	'''
	Base Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		model (object): model instance
		arguments (iterable[object]): Position arguments for function
		keywords (dict[str,object]): Keyword arguments for function
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,func,grad=None,callback=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):

		if hyperparameters is not None and system is not None:
			kwargs.update({attr: hyperparameters.get(attr) for attr in (system if system is not None else ()) if attr in hyperparameters})

		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		defaults = {
			'iterations':None,
			'optimizer':None,
			'search':{'alpha':'line_search','beta':None},
			'eps':{'value':1e-16,'grad':1e-18,'value.difference':1e-18,'grad.difference':1e-18,'value.increase':1e1,'iteration':1,'alpha':1e-7,'beta':1e3},
			'value':{'value':1,'grad':1,'value.difference':1,'grad.difference':1,'value.increase':1,'iteration':100,'alpha':0,'beta':0},
			'bounds':{'value':[-1e20,1e20],'grad':[-1e12,1e12],'alpha':[1e-20,1e6],'beta':[-1e10,1e10]},	
			'kwargs':{},
			'alpha':0,
			'status':1,
			'clear':True,
			'cwd':None,
			'path':None,
			'modulo':{'log':None,'buffer':None,'attributes':None,'callback':None,'alpha':None,'beta':None,'dump':None},
			'length':{'log':None,'buffer':1,'attributes':5,'callback':None,'alpha':None,'beta':None,'dump':None},
			'attributes':{'iteration':[],'parameters':[],'value':[],'grad':[],'search':[],'alpha':[]},	
			'track':{},		
		}

		setter(hyperparameters,defaults,delimiter=None,func=False)

		arguments = () if arguments is None else arguments
		keywords = {} if keywords is None else keywords

		self.arguments = arguments
		self.keywords = keywords

		self.hyperparameters = Dict(hyperparameters)
		self.system = system

		self.value_and_grad,self.func,self.grad = value_and_gradient(func,grad,returns=True)
		self.hess = hessian(func)

		self.alpha = LineSearch(self.func,self.grad,
			arguments=self.arguments,keywords=self.keywords,hyperparameters=self.hyperparameters,
			system=self.system)

		if callback is None:
			def callback(parameters,track,optimizer):
				status = True
				return status

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
		self.iterations = hyperparameters['iterations']
		self.sizes = {attr: hyperparameters['length'].get(attr) if hyperparameters['length'].get(attr) else 1 for attr in ['buffer','attributes']}
		self.search = hyperparameters['search']
		self.eps = hyperparameters['eps']
		self.bounds = hyperparameters['bounds']
		self.value = hyperparameters['value']
		self.kwargs = hyperparameters['kwargs']

		self.paths = {'track':join(self.path,ext=None,root=self.cwd),'attributes':join(self.path,ext='ckpt',root=self.cwd)} if self.path is not None else None

		self.callback = callback

		self.reset(clear=True,opt=None)

		return

	def __call__(self,parameters,*args,**kwargs):
		'''
		Iterate optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function				
		Returns:
			parameters (object): optimizer parameters
		'''

		iteration = self.iteration
		opt = self.opt_init(parameters)
		iteration,opt = self.load(iteration,opt)
		
		args = (*args,*self.arguments[len(args):])
		kwargs = {**self.keywords,**kwargs}

		self.info()

		self.init(iteration,opt)

		for iteration in self.iterations:

			opt = self.opt_update(iteration,opt,*args,**kwargs)

			self.dump(iteration,opt)

			if not self.status:
				break

		parameters = self.get_params(opt)

		self.parameters = parameters

		return parameters


	def update(self,iteration,parameters,value,grad,search,*args,**kwargs):
		'''
		Update parameters
		Args:
			iteration (int): Iteration index
			parameters (array): Parameters
			value (array): Optimization value
			grad (array): Optimization gradient
			search (array): Optimization search direction
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function			
		Returns:
			parameters (array): Updated parameters
			search (array): Updated optimization search direction			
			alpha (array): Search rate
		'''
		
		optimizer = self

		alpha = optimizer.alpha(
				iteration,
				parameters,
				optimizer.attributes['alpha'],
				optimizer.attributes['value'],
				optimizer.attributes['grad'],
				optimizer.attributes['search']
				*args,**kwargs) if optimizer.size > 1 else optimizer.hyperparameters['alpha']
		search = -grad
		# search = search/norm(search) if optimizer.kwargs.get('normalize') else search
		parameters = parameters + alpha*search
		return parameters,search,alpha

	def opt_init(self,parameters):
		'''
		Initialize optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
		Returns:
			opt (object): optimizer state
		'''
		opt = parameters
		return opt

	def opt_update(self,iteration,opt,*args,**kwargs):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			opt (object): optimizer state
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function
		Returns:
			opt (object): optimizer state
		'''


		value,grad,parameters = self.opt_step(iteration,opt,*args,**kwargs)
		search = self.attributes['search'][-1] if self.size > 1 else 0

		parameters,search,alpha = self.update(iteration,parameters,value,grad,search,*args,**kwargs)

		self.attributes['search'].append(search)
		self.attributes['alpha'].append(alpha)

		opt = self.opt_init(parameters)
		parameters = self.get_params(opt)
		track = self.track		
		optimizer = self
		self.status = self.callback(parameters,track,optimizer)

		return opt

	def get_params(self,opt):
		'''
		Get optimizer parameters with optimizer state
		Args:
			opt (object): optimizer state
		Returns:
			parameters (object): optimizer parameters
		'''
		parameters = opt
		return parameters

	def opt_step(self,iteration,opt,*args,**kwargs):
		'''
		Iterate optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			opt (object): optimizer state
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function
		Returns:
			value (object): optimizer value
			grad (object): optimizer grad
		'''

		parameters = self.get_params(opt)
		value,grad = self.value_and_grad(parameters,*args,**kwargs)

		size = self.size

		if (self.sizes) and (self.size > 0) and (self.size >= sum(self.sizes[attr] for attr in self.sizes)):
			for attr in self.attributes:
				if self.attributes[attr]:
					self.attributes[attr].pop(self.sizes['buffer'])

		iteration += 1
		size += 1

		self.attributes['iteration'].append(iteration)
		self.attributes['parameters'].append(parameters)
		self.attributes['value'].append(value)
		self.attributes['grad'].append(grad)

		self.iteration = iteration
		self.size = size

		return value,grad,parameters


	def dump(self,iteration=None,opt=None):
		'''
		Dump data
		Args:
			iteration (int): optimizer iteration
			opt (object): optimizer state
		'''

		do = (self.paths is not None) and ((not self.status) or (self.modulo['dump'] is None) or (self.modulo['dump'] == -1) or (iteration is None) or (iteration%self.modulo['dump'] == 0) or (iteration==(self.iterations.stop-1)))

		if not do:
			return

		path = self.paths['track']
		data = self.track
		dump(data,path)

		path = self.paths['attributes']
		data = self.attributes
		dump(data,path)

		return

	def load(self,iteration=None,opt=None):
		'''
		Load data
		Args:
			iteration (int): optimizer iteration
			opt (object): optimizer state
		Returns:
			iteration (int): optimizer iteration
			opt (object): optimizer state
		'''

		do = (self.paths is not None)

		if not do:
			self.reset(clear=False,opt=opt)
			return iteration,opt

		path = self.paths['track']
		data = load(path)

		if data is not None:
			length = min(len(data[attr]) for attr in data)
			for attr in data:
				if attr not in self.track:
					continue
				self.track[attr] = [*data[attr],*self.track[attr]]

		size = max((len(self.track[attr]) for attr in self.track),default=0)
		default = nan

		for attr in self.track:
			data = [default for i in range(size-len(self.track[attr]))]
			self.track[attr] = [*self.track[attr],*data]

		path = self.paths['attributes']
		data = load(path)

		if data is not None:
			for attr in data:
				if attr not in self.attributes:
					continue
				self.attributes[attr] = [*data[attr],*self.attributes[attr]]

		size = max((len(self.attributes[attr]) for attr in self.attributes),default=0)
		default = nan

		for attr in self.attributes:
			data = [default for i in range(size-len(self.attributes[attr]))]
			self.attributes[attr] = [*self.attributes[attr],*data]

		self.parameters = self.get_params(opt)
		self.reset(clear=False,opt=opt)

		iteration = self.iteration
		opt = self.opt_init(self.parameters)

		return iteration,opt
		

	def reset(self,clear=None,opt=None):
		'''
		Reset class attributes
		Args:
			clear (bool): clear attributes
			opt (object): optimizer state
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

		
		objs = {'func.model':False,'hyperparameters':True}
		for attr in list(self.track):

			if attr in self.attributes:
				continue
			
			value = self.track.pop(attr)
			
			for name in objs:

				if not hasattrs(self,name,delimiter=delim):
					continue

				if objs[name]:
					pattern = delim.join(attr.split(delim)[1:]) if attr.split(delim)[0] == name else attr
				else:
					pattern = attr

				obj = getattrs(self,name,delimiter=delim)
				attribute = None

				for attribute in iterate(obj,pattern):

					if objs[name]:
						attribute = delim.join([name,attribute])

					self.track[attribute] = [*copy(value)]

				if attribute is not None:
					break
				else:
					self.track[attr] = [*copy(value)]


		self.size = min((len(self.attributes[attr]) for attr in self.attributes),default=self.size)


		while (self.sizes) and (self.size > 0) and (self.size >= sum(self.sizes[attr] for attr in self.sizes)):
			for attr in self.attributes:
				if self.attributes[attr]:
					self.attributes[attr].pop(self.sizes['buffer'])
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

	
		if not clear:
			if isinstance(self.iterations,int):
				self.iterations = range(self.iteration,self.iterations+self.iteration)
			elif isinstance(self.iterations,range):
				self.iterations = range(
					self.iteration,
					self.iterations.stop-self.iterations.start+self.iteration,
					self.iterations.step)				
			else:
				self.iterations = range(self.iteration,self.iterations[1],*self.iterations[2:])

		return


	def init(self,iteration,opt,*args,**kwargs):
		'''
		Update attributes
		Args:
			iteration (int): optimizer iteration
			opt (object): optimizer state
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function
		'''

		do = (opt is not None) and (
		   (isinstance(self.iterations,int) and (self.iteration == 0)) or 
		   (isinstance(self.iterations,range) and (self.iterations.start == 0) and (self.iterations.stop == 0)) or 
		   ((not isinstance(self.iterations,(int,range))) and (self.iterations[0] == 0) and (self.iterations[1] == 0)))

		if not do:
			return
			
		value,grad,parameters = self.opt_step(iteration-1,opt,*args,**kwargs)
		search = -grad
		alpha,beta = self.hyperparameters.get('alpha'),self.hyperparameters.get('beta')

		attrs = {'search':search,'alpha':alpha,'beta':beta}
		for attr in attrs:
			if attr in self.attributes:
				self.attributes[attr].append(attrs[attr])
	
		opt = self.opt_init(parameters)
		parameters = self.get_params(opt)
		track = self.track
		optimizer = self
		self.status = self.callback(parameters,track,optimizer)

		self.dump(iteration,opt)

		return

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		
		
		msg = []

		for attr in ['optimizer','iterations','size','search','eps','modulo','kwargs']:
			string = '%s %s: %s'%('Optimizer',attr,getattr(self,attr))
			msg.append(string)

		for attr in ['dtype']:
			string = []
			for subattr in ['func.model','func.metric','func.model.parameters','func.model.state','func.metric.state','func.metric.label']:
				substring = '%s: %s'%(subattr,getattrs(self,delim.join([subattr,attr]),delimiter=delim) if getattrs(self,subattr,delimiter=delim) is not None else None)
				string.append(substring)
			string = '%s %s: %s'%('Optimizer',attr,', '.join(string))

			msg.append(string)

		for attr in ['track','attributes']:
			string = '%s %s: %s'%('Optimizer',attr,
				{key: getattr(self,attr).get(key,[None])[-1] 
				if isinstance(getattr(self,attr).get(key,[None])[-1],scalars) else ['...'] 
				for key in getattr(self,attr)} if any(getattr(self,attr).get(key) for key in getattr(self,attr)) else
				[attr for attr in getattr(self,attr)]
				)
			msg.append(string)


		msg = '\n'.join(msg)

		self.log(msg,verbose=verbose)
		return

class Optimizer(Optimization):
	'''
	Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		arguments (iterable[object]): Position arguments for function
		keywords (dict[str,object]): Keyword arguments for function
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
		kwargs (dict): Additional system attributes		
	'''
	def __new__(cls,func,grad=None,callback=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):
	
		defaults = {'optimizer':None}
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		# optimizers = {'adam':Adam,'cg':ConjugateGradient,'gd':GradientDescent,'ls':LineSearchDescent,'hd':HessianDescent,None:GradientDescent}
		optimizers = {'adam':GradientDescent,'cg':ConjugateGradient,'gd':GradientDescent,'ls':LineSearchDescent,'hd':HessianDescent,None:GradientDescent}

		optimizer = hyperparameters.get('optimizer')
		
		self = optimizers.get(optimizer,optimizers[None])(func,grad,callback,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		return self
	

class GradientDescent(Optimization):
	'''
	Gradient Descent Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		arguments (iterable[object]): Position arguments for function
		keywords (dict[str,object]): Keyword arguments for function		
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,func,grad=None,callback=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):

		defaults = {'track':{'beta':False},'attributes':{'beta':False}}		
		setter(hyperparameters,defaults,delimiter=delim,func=True)

		super().__init__(func,grad,callback,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		defaults = {}
		setter(self.hyperparameters,defaults,delimiter=delim,func=False)

		return

	def opt_update(self,iteration,opt,*args,**kwargs):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			opt (object): optimizer state
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function
		Returns:
			opt (object): optimizer state
		'''

		value,grad,parameters = self.opt_step(iteration,opt,*args,**kwargs)
		search = self.attributes['search'][-1] if self.size > 1 else 0

		parameters,search,alpha = self.update(iteration,parameters,value,grad,search,*args,**kwargs)

		self.attributes['search'].append(search)
		self.attributes['alpha'].append(alpha)

		opt = self.opt_init(parameters)
		parameters = self.get_params(opt)
		track = self.track		
		optimizer = self
		self.status = self.callback(parameters,track,optimizer)

		return opt


class LineSearchDescent(Optimization):
	'''
	Gradient Descent with Line Search Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		arguments (iterable[object]): Position arguments for function
		keywords (dict[str,object]): Keyword arguments for function		
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,func,grad=None,callback=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):

		defaults = {'track':{'beta':False},'attributes':{'beta':False}}		
		setter(hyperparameters,defaults,delimiter=delim,func=True)

		super().__init__(func,grad,callback,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		defaults = {}
		setter(self.hyperparameters,defaults,delimiter=delim,func=False)

		return

	def opt_update(self,iteration,opt,*args,**kwargs):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			opt (object): optimizer state
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function			
		Returns:
			opt (object): optimizer state
		'''

		value,grad,parameters = self.opt_step(iteration,opt,*args,**kwargs)
		search = self.attributes['search'][-1] if self.size > 1 else 0

		parameters,search,alpha = self.update(iteration,parameters,value,grad,search,*args,**kwargs)

		self.attributes['search'].append(search)
		self.attributes['alpha'].append(alpha)

		opt = self.opt_init(parameters)
		parameters = self.get_params(opt)
		track = self.track		
		optimizer = self
		self.status = self.callback(parameters,track,optimizer)

		return opt

class HessianDescent(Optimization):
	'''
	Gradient Descent with Hessian Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		arguments (iterable[object]): Position arguments for function
		keywords (dict[str,object]): Keyword arguments for function
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,func,grad=None,callback=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):

		defaults = {'track':{'beta':False},'attributes':{'beta':False}}		
		setter(hyperparameters,defaults,delimiter=delim,func=True)

		super().__init__(func,grad,callback,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		defaults = {}
		setter(self.hyperparameters,defaults,delimiter=delim,func=False)

		return

	def update(self,iteration,parameters,value,grad,search,*args,**kwargs):
		'''
		Update parameters
		Args:
			iteration (int): Iteration index
			parameters (array): Parameters
			value (array): Optimization value
			grad (array): Optimization gradient
			search (array): Optimization search direction
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function
		Returns:
			parameters (array): Updated parameters
			search (array): Updated optimization search direction			
			alpha (array): Search rate
		'''
		optimizer = self
		alpha = optimizer.hyperparameters['alpha']
		search = -grad
		# search = search/norm(search) if optimizer.kwargs.get('normalize') else search
		hess = optimizer.hess(parameters)		
		parameters = parameters + alpha*lstsq(hess,search)
		return parameters,search,alpha

	def opt_update(self,iteration,opt,*args,**kwargs):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			opt (object): optimizer state
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function			
		Returns:
			opt (object): optimizer state
		'''

		value,grad,parameters = self.opt_step(iteration,opt,*args,**kwargs)
		search = self.attributes['search'][-1] if self.size > 1 else 0

		parameters,search,alpha = self.update(iteration,parameters,value,grad,search,*args,**kwargs)

		self.attributes['search'].append(search)
		self.attributes['alpha'].append(alpha)

		opt = self.opt_init(parameters)
		parameters = self.get_params(opt)
		track = self.track		
		optimizer = self
		self.status = self.callback(parameters,track,optimizer)

		return opt

class ConjugateGradient(Optimization):
	'''
	Conjugate Gradient Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		arguments (iterable[object]): Position arguments for function
		keywords (dict[str,object]): Keyword arguments for function		
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,func,grad=None,callback=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):

		defaults = {'beta':0,'search':{'beta':None},'attributes':{'beta':[]}}
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		super().__init__(func,grad,callback,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		defaults = {}
		setter(self.hyperparameters,defaults,delimiter=delim,func=False)

		self.beta = GradSearch(self.func,self.grad,
			arguments=self.arguments,keywords=self.keywords,hyperparameters=self.hyperparameters,
			system=self.system)

		return


	def update(self,iteration,parameters,value,grad,search,*args,**kwargs):
		'''
		Update parameters
		Args:
			iteration (int): Iteration index
			parameters (array): Parameters
			value (array): Optimization value
			grad (array): Optimization gradient
			search (array): Optimization search direction
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function			
		Returns:
			parameters (array): Updated parameters
			search (array): Updated optimization search direction			
			alpha (array): Search rate
			beta (array): Conjugate rate
		'''

		optimizer = self

		alpha = optimizer.alpha(
			iteration,
			parameters,
			optimizer.attributes['alpha'],
			optimizer.attributes['value'],
			optimizer.attributes['grad'],
			optimizer.attributes['search'],
			*args,**kwargs)

		parameters = parameters + alpha*search

		opt = optimizer.opt_init(parameters)

		value,grad,parameters = optimizer.opt_step(iteration,opt,*args,**kwargs)
		
		beta = optimizer.beta(
			iteration,
			parameters,
			optimizer.attributes['beta'],
			optimizer.attributes['value'],
			optimizer.attributes['grad'],
			optimizer.attributes['search'])

		search = -grad + beta*search
		return parameters,search,alpha,beta

	def opt_update(self,iteration,opt,*args,**kwargs):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			opt (object): optimizer state
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function
		Returns:
			opt (object): optimizer state
		'''


		if self.size == 0:
			
			value,grad,parameters = self.opt_step(iteration-1,opt,*args,**kwargs)
			search = -grad
			alpha,beta = self.hyperparameters['alpha'],self.hyperparameters['beta']

			attrs = {'search':search,'alpha':alpha,'beta':beta}
			for attr in attrs:
				if attr in self.attributes:
					self.attributes[attr].append(attrs[attr])
		
			opt = self.opt_init(parameters)
			parameters = self.get_params(opt)
			track = self.track
			optimizer = self
			self.status = self.callback(parameters,track,optimizer)

		parameters = self.get_params(opt)

		value = self.attributes['value'][-1]
		grad = self.attributes['grad'][-1]
		search = self.attributes['search'][-1]

		parameters,search,alpha,beta = self.update(iteration,parameters,value,grad,search,*args,**kwargs)

		self.attributes['search'].append(search)
		self.attributes['alpha'].append(alpha)
		self.attributes['beta'].append(beta)
	
		opt = self.opt_init(parameters)
		parameters = self.get_params(opt)
		track = self.track
		optimizer = self
		self.status = self.callback(parameters,track,optimizer)

		return opt


class Adam(Optimization):
	'''
	Adam Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,optimizer) and returns status of optimization
		arguments (iterable[object]): Position arguments for function
		keywords (dict[str,object]): Keyword arguments for function		
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
		kwargs (dict): Additional system attributes
	'''
	def __init__(self,func,grad=None,callback=None,arguments=None,keywords=None,hyperparameters={},system=None,**kwargs):

		defaults = {'track':{'beta':False},'attributes':{'beta':False}}		
		setter(hyperparameters,defaults,delimiter=delim,func=True)

		super().__init__(func,grad,callback,arguments=arguments,keywords=keywords,hyperparameters=hyperparameters,system=system,**kwargs)

		defaults = {}
		setter(self.hyperparameters,defaults,delimiter=delim,func=False)

		self._optimizer = getattr(optimizer_libraries,self.optimizer)

		self._opt_init,self._opt_update,self._get_params = self._optimizer(self.hyperparameters['alpha'])

		return

	def opt_init(self,parameters):
		'''
		Initialize optimizer state with parameters
		Args:
			parameters (object): optimizer parameters
		Returns:
			opt (object): optimizer state
		'''

		opt = self._opt_init(parameters)

		return opt

	def update(self,iteration,parameters,value,grad,search,*args,**kwargs):
		'''
		Update parameters
		Args:
			iteration (int): Iteration index
			parameters (array): Parameters
			value (array): Optimization value
			grad (array): Optimization gradient
			search (array): Optimization search direction
		Returns:
			parameters (array): Updated parameters
			search (array): Updated optimization search direction			
			alpha (array): Search rate
		'''
		optimizer = self

		alpha = optimizer.hyperparameters['alpha']
		search = -grad
		
		opt = optimizer.opt_init(parameters)
		opt = optimizer._opt_update(iteration,grad,opt)
		parameters = optimizer.get_params(opt)

		return parameters,search,alpha


	def opt_update(self,iteration,opt,*args,**kwargs):
		'''
		Update optimizer state with parameters
		Args:
			iteration (int): optimizer iteration
			opt (object): optimizer state
			args (iterable[object]): Positional arguments for function
			kwargs (dict[str,object]): Keyword arguments for function
		Returns:
			opt (object): optimizer state
		'''

		value,grad,parameters = self.opt_step(iteration,opt,*args,**kwargs)
		search = self.attributes['search'][-1] if self.size > 1 else 0

		parameters,search,alpha = self.update(iteration,parameters,value,grad,search,*args,**kwargs)

		self.attributes['alpha'].append(alpha)
		self.attributes['search'].append(search)

		opt = self.opt_init(parameters)
		parameters = self.get_params(opt)
		track = self.track		
		optimizer = self
		self.status = self.callback(parameters,track,optimizer)

		return opt

	def get_params(self,opt):
		'''
		Get optimizer parameters with optimizer state
		Args:
			opt (object): optimizer state
		Returns:
			parameters (object): optimizer parameters
		'''

		parameters = self._get_params(opt)

		return parameters



class Covariance(System):
	def __init__(self,func,grad=None,shapes=None,state=None,label=None,weights=None,optimize=None,metric=None,hyperparameters={},system=None,**kwargs):
		'''
		Compute covariance of function (with Cramer Rao bound)
		Args:
			func (callable): Function to compute
			grad (callable): Gradient to compute
			shapes (iterable[tuple[int]]): Shapes of functions		
			state (array, callable): state data for function
			label (array, callable): label data for function
			weights (array): weights data for function
			optimize (bool,str,iterable): Contraction type
			metric (str,Metric): Type of distribution, allowed ['lstsq','mse','normal','gaussian']
			hyperparameters (dict): Function hyperparameters			
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
			kwargs (dict): Additional system attributes
		Returns:
			cov (callable): Covariance of function
		'''

		if hyperparameters is not None and system is not None:
			kwargs.update({attr: hyperparameters.get(attr) for attr in (system if system is not None else ()) if attr in hyperparameters})

		setter(kwargs,system,delimiter=delim,func=False)

		super().__init__(**kwargs)

		if shapes is None:
			try:
				shapes = label.shape
			except:
				pass

		if all(isinstance(i,int) for i in shapes) or (len(shapes) == 1):
			shapes = [shapes]*2

		if label is None:
			shapes = (*shapes[:1],*shapes[2:])				
		else:
			shapes = (*shapes[:1],label.shape,*shapes[2:])						

		if weights is None:
			shapes = (*shapes[:2],None)
		else:
			shapes = (*shapes[:2],weights.shape)		

		if isinstance(metric,Metric):
			metric = metric.metric

		function = func

		metric = metrics(metric,shapes=shapes,label=label,weights=weights)

		@jit
		def func(parameters,*args,**kwargs):
			return metric(function(parameters,*args,**kwargs))

		self.func = func
		self.metric = metric
		self.function = function
		self.hess = hessian(func)

		self.string = str(self.metric)

		self.optimize = optimize
		self.hyperparameters = hyperparameters
		self.system = system
		

		return

	def __call__(self,parameters,*args,**kwargs):
		return inv(self.hess(parameters,*args,**kwargs))

	def info(self,verbose=None):
		'''
		Log class information
		Args:
			verbose (int,str): Verbosity of message			
		'''		
		
		msg = []

		for attr in ['metric']:
			string = '%s %s: %s'%(self.__class__.__name__,attr,getattr(self,attr))
			msg.append(string)

		msg = '\n'.join(msg)
		
		self.log(msg,verbose=verbose)
		return