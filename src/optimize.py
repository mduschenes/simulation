#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy,datetime
from functools import partial

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))


# Logging
import logging
logger = logging.getLogger(__name__)

# Import user modules
from src.utils import jit,value_and_gradient,gradient
from src.utils import is_naninf,product,sqrt,asarray

from src.utils import inner_norm,inner_abs2,inner_real,inner_imag
from src.utils import gradient_inner_norm,gradient_inner_abs2,gradient_inner_real,gradient_inner_imag

from src.utils import itg,dbl,flt,delim,nan,Null

from src.dictionary import setter

from src.line_search import line_search,armijo

from src.io import dump,load,join,split,copy,exists

from src.system import Class



class LineSearchBase(Class):
	def __init__(self,func,grad,hyperparameters,system=None):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
		'''
		defaults = {}		
		returns = ['alpha']
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		self.func = func
		self.grad = grad
		self.hyperparameters = hyperparameters
		self.defaults = defaults
		self.returns = returns

		super().__init__(hyperparameters=hyperparameters,system=system)

		return

	def __call__(self,parameters,alpha,value,gradient,search):
		'''
		Perform line search
		Args:
			parameters (array): Objective parameters
			alpha (iterable): Previous alpha
			value (iterable): Previous objective values
			gradient (array): Previous objective gradients
			search (array): Previous objective search directions
		Returns:
			alpha (array): Returned search value
		'''
		returns = (alpha[-1],)

		returns = self.__callback__(returns,parameters,alpha,value,gradient,search)

		attr = 'alpha'
		alpha = returns[attr]

		return alpha

	def __callback__(self,returns,parameters,alpha,value,gradient,search):
		'''
		Check return values of line search
		Args:
			returns (iterable): Iterable of returned values of line search
			parameters (array): Objective parameters
			alpha (iterable): Previous alpha
			value (iterable): Previous objective values
			gradient (array): Previous objective gradients
			search (array): Previous objective search directions
		Returns:
			returns (dict): Dictionary of returned values of line search
		'''
		
		returns = dict(zip(self.returns,returns))

		attr = 'alpha'
		if returns[attr] is None or (returns[attr] < self.hyperparameters['bounds'][attr][0]) or (returns[attr] > self.hyperparameters['bounds'][attr][1]):
			if len(alpha) > 1:
				returns[attr] = alpha[-1]*gradient[-1].dot(search[-1])/gradient[-2].dot(search[-2])
			else:
				returns[attr] = alpha[-1]

		return returns

class LineSearch(LineSearchBase):
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

		line_searches = {'line_search':Line_Search,'armijo':Armijo,None:Null_Search}

		line_search = hyperparameters.get('search',{}).get('alpha',defaults['search']['alpha'])
		
		self = line_searches.get(line_search,line_searches[None])(func,grad,hyperparameters)

		return self


class Line_Search(LineSearchBase):
	def __init__(self,func,grad,hyperparameters,system=None):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
		'''
		defaults = {'c1':0.0001,'c2':0.9,'maxiter':10,'old_old_fval':None}
		returns = ['alpha','nfunc','ngrad','value','_value','slope']
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		super().__init__(func,grad,hyperparameters=hyperparameters,system=system)

		self.defaults = defaults
		self.returns = returns

		return

	def __call__(self,parameters,alpha,value,gradient,search):
		'''
		Perform line search
		Args:
			parameters (array): Objective parameters
			alpha (iterable): Previous alpha
			value (iterable): Previous objective values
			gradient (array): Previous objective gradients
			search (array): Previous objective search directions
		Returns:
			alpha (array): Returned search value
		'''
		self.defaults['old_old_fval'] = value[-2] if len(value)>1 else None
		
		returns = line_search(self.func,self.grad,
			parameters,search[-1],gradient[-1],value[-1],
			**self.defaults)
		
		returns = self.__callback__(returns,parameters,alpha,value,gradient,search)

		attr = 'alpha'
		alpha = returns[attr]

		return alpha


class Armijo(LineSearchBase):
	def __init__(self,func,grad,hyperparameters,system=None):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
		'''
		defaults = {'c1':0.0001,'alpha0':1e-4}
		returns = ['alpha','nfunc','value']
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		defaults.update({attr: hyperparameters.get(attr,defaults[attr]) for attr in defaults})

		super().__init__(func,grad,hyperparameters=hyperparameters,system=system)

		self.defaults = defaults
		self.returns = returns

		return

	def __call__(self,parameters,alpha,value,gradient,search):
		'''
		Perform line search
		Args:
			parameters (array): Objective parameters
			alpha (iterable): Previous alpha
			value (iterable): Previous objective values
			gradient (array): Previous objective gradients
			search (array): Previous objective search directions
		Returns:
			alpha (array): Returned search value
		'''

		returns = armijo(self.func,self.grad,
			parameters,search[-1],gradient[-1],value[-1],
			**self.defaults)

		returns = self.__callback__(returns,parameters,alpha,value,gradient,search)
		
		return returns


class Null_Search(LineSearchBase):
	def __init__(self,func,grad,hyperparameters,system=None):
		'''	
		Line search class
		Args:
			func (callable): objective function with signature func(parameters)
			grad (callable): gradient of function to optimize, with signature grad(parameters)
			hyperparameters (dict): Line search hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)			
		'''

		super().__init__(func,grad,hyperparameters=hyperparameters,system=system)

		return


class FuncBase(Class):

	def __init__(self,model,func=None,grad=None,callback=None,metric=None,hyperparameters={},system=None):
		'''	
		Class for function
		Args:
			model (object): Model instance
			func (callable,iterable[callable]): Function function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			callback (callable): Callback of function with signature callback(parameters,track,attributes,model,metric,func,grad,hyperparameters)
			metric (str,callable): Function metric with signature metric(*operands)
			hyperparameters (dict): Function hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
		'''

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
			def callback(parameters,track,attributes,model,func,grad,hyperparameters):
				status = True
				return status

		self.value_and_gradient,self.function,self.gradient = value_and_gradient(function,gradient,returns=True)

		self.model = model
		self.callback = callback
		self.metric = metric
		self.hyperparameters = hyperparameters

		super().__init__(hyperparameters=hyperparameters,system=system)

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
	def __callback__(self,parameters,track,attributes,hyperparameters):
		'''
		Callback call
		Args:
			parameters (array): parameters
			track (dict): callback track			
			attributes (dict): callback attributes			
			hyperparameters (dict): callback hyperparameters			
		Returns:
			status (int): status of callback
		'''
		status = self.callback(parameters,track,attributes,
			hyperparameters=hyperparameters,
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


class Objective(FuncBase):		
	def __init__(self,model,metric,func=None,grad=None,callback=None,hyperparameters={},system=None):
		'''	
		Objective class for metric + function
		Args:
			model (object): Model instance
			metric (str,callable): Objective metric with signature metric(*operands)
			func (callable,iterable[callable]): Objective function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			callback (callable): Callback of function with signature callback(parameters,track,attributes,model,metric,func,grad,hyperparameters)			
			hyperparameters (dict): Objective hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
		'''

		super().__init__(model,func=func,grad=grad,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system)

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


class Callback(FuncBase):

	def __init__(self,model,callback,func=None,grad=None,metric=None,hyperparameters={},system=None):
		'''	
		Class for function
		Args:
			model (object): Model instance
			callback (callable): Callback of function with signature callback(parameters,track,attributes,model,metric,func,grad,hyperparameters)
			func (callable,iterable[callable]): Function function with signature func(parameters), or iterable of functions to sum
			grad (callable,iterable[callable]): Gradient of function with signature grad(parameters), or iterable of functions to sum
			metric (str,callable): Callback metric with signature metric(*operands)
			hyperparameters (dict): Callback hyperparameters
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)
		'''
		
		super().__init__(model,func=func,grad=grad,callback=callback,metric=metric,hyperparameters=hyperparameters,system=system)

		return

	# @partial(jit,static_argnums=(0,))
	def __call__(self,parameters,track,attributes,hyperparameters):
		''' 
		Callback
		Args:
			parameters (array): parameters
			track (dict): callback tracking
			attributes (dict): Callback attributes
			hyperparameters(dict): Callback hyperparameters
		Returns:
			status (int): status of callback
		'''
		status = self.__callback__(parameters,track,attributes,hyperparameters)
		return status


class Metric(Class):
	def __init__(self,metric=None,shapes=None,model=None,label=None,optimize=None,hyperparameters={},system=None):
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
		'''

		self.metric = hyperparameters.get('metric',metric) if metric is None else metric
		self.label = hyperparameters.get('label',label) if label is None else label
		self.shapes = shapes
		self.model = model
		self.optimize = optimize
		self.hyperparameters = hyperparameters

		super().__init__(hyperparameters=hyperparameters,system=system)
		self.__setup__()

		self.info()

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

	@partial(jit,static_argnums=(0,))
	def __call__(self,*operands):
		'''
		Function call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''
		return self.function(*operands)

	@partial(jit,static_argnums=(0,))
	def __grad__(self,*operands):
		'''
		Gradient call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.gradient(*operands)

	@partial(jit,static_argnums=(0,))
	def __grad_analytical__(self,*operands):
		'''
		Gradient call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.gradient_analytical(*operands)

	@partial(jit,static_argnums=(0,))
	def func(self,*operands):
		'''
		Function call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.__call__(*operands)

	@partial(jit,static_argnums=(0,))
	def grad(self,*operands):
		'''
		Gradient call
		Args:
			operands (array): operands
		Returns:
			out (object): Return of function
		'''		
		return self.__grad__(*operands)	

	@partial(jit,static_argnums=(0,))
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
			gradient_analytical = jit(gradient(metric))

		elif self.metric is None:

			def wrapper(out,*operands):
				return out/2/operands[0].shape[0]

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				function = inner_norm(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				function = partial(inner_norm,optimize=optimize,wrapper=wrapper)


			def wrapper(out,*operands):
				return out/2/operands[0].shape[0]

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				gradient_analytical = gradient_inner_norm(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				gradient_analytical = partial(gradient_inner_norm,optimize=optimize,wrapper=wrapper)

		elif self.metric in ['norm']:

			def wrapper(out,*operands):
				return out/2/operands[0].shape[0]

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				function = inner_norm(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				function = partial(inner_norm,optimize=optimize,wrapper=wrapper)


			def wrapper(out,*operands):
				return out/2/operands[0].shape[0]

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				gradient_analytical = gradient_inner_norm(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				gradient_analytical = partial(gradient_inner_norm,optimize=optimize,wrapper=wrapper)


		elif self.metric in ['abs2']:

			def wrapper(out,*operands):
				return 1 - out/(operands[0].size)

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				function = inner_abs2(*shapes,optimize=optimize,wrapper=wrapper)
				print('DOing function',function)
			else:
				shapes = ()
				optimize = self.optimize
				function = partial(inner_abs2,optimize=optimize,wrapper=wrapper)


			def wrapper(out,*operands):
				return - out/(operands[0].size)

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				gradient_analytical = gradient_inner_abs2(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				gradient_analytical = partial(gradient_inner_abs2,optimize=optimize,wrapper=wrapper)


		elif self.metric in ['real']:

			def wrapper(out,*operands):
				return 1 - out/(operands[0].shape[0])

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				function = inner_real(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				function = partial(inner_real,optimize=optimize,wrapper=wrapper)


			def wrapper(out,*operands):
				return  - out/(operands[0].shape[0])

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				gradient_analytical = gradient_inner_real(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				gradient_analytical = partial(gradient_inner_real,optimize=optimize,wrapper=wrapper)


		elif self.metric in ['imag']:

			def wrapper(out,*operands):
				return 1 - out/(operands[0].shape[0])

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				function = inner_imag(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				function = partial(inner_imag,optimize=optimize,wrapper=wrapper)


			def wrapper(out,*operands):
				return - out/(operands[0].shape[0])

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				gradient_analytical = gradient_inner_imag(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				gradient_analytical = partial(gradient_inner_imag,optimize=optimize,wrapper=wrapper)

		else:

			def wrapper(out,*operands):
				return out/2/operands[0].shape[0]

			if self.shapes:
				shapes = (*self.shapes,)
				optimize = self.optimize
				function = inner_norm(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				function = partial(inner_norm,optimize=optimize,wrapper=wrapper)


			def wrapper(out,*operands):
				return out/2/operands[0].shape[0]

			if self.shapes:
				shapes = (*self.shapes,(self.size**2,*self.shapes[0]))
				optimize = self.optimize
				gradient_analytical = gradient_inner_norm(*shapes,optimize=optimize,wrapper=wrapper)
			else:
				shapes = ()
				optimize = self.optimize
				gradient_analytical = partial(gradient_inner_norm,optimize=optimize,wrapper=wrapper)

		_function = jit(function)
		_grad = jit(gradient_analytical)
		# _grad = jit(gradient(function,mode='fwd',holomorphic=True,move=True))
		_gradient_analytical = jit(gradient_analytical)

		if self.label is not None and self.metric in [None,'norm','abs2','real','imag']:
			def function(*operands):
				return _function(*operands[:1],self.label,*operands[1:])
			def grad(*operands):
				return _grad(*operands[:1],self.label,*operands[1:])				
			def gradient_analytical(*operands):
				return _gradient_analytical(*operands[:1],self.label,*operands[1:])
		else:
			def function(*operands):
				return _function(*operands)
			def grad(*operands):
				return _grad(*operands)
			def gradient_analytical(*operands):
				return _gradient_analytical(*operands)

		function = jit(function)
		grad = jit(grad)
		gradient_analytical = jit(gradient_analytical)


		self.function = function
		self.gradient = grad
		self.gradient_analytical = gradient_analytical

		return



class OptimizerBase(Class):
	'''
	Base Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,attributes,hyperparameters) and returns status of optimization
		model (object): model instance
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={},system=None):

		defaults = {
			'optimizer':None,
			'search':{'alpha':'line_search','beta':None},
			'eps':{'value':1e-4,'grad':1e-12,'alpha':1e-12,'beta':1e3},
			'alpha':0,
			'status':1,
			'cwd':None,
			'path':None,
			'modulo':{'log':None,'attributes':None,'callback':None,'restart':None,'dump':None},
			'length':{'log':None,'attributes':10,'callback':None,'restart':None,'dump':None},
			'attributes':{'iteration':[],'parameters':[],'value':[],'grad':[],'search':[],'alpha':[]},	
			'track':{},		
		}
		setter(hyperparameters,defaults,delimiter=delim,func=False)
		hyperparameters.update({attr: updates.get(attr,{}).get(hyperparameters[attr],hyperparameters[attr]) 
			if attr in updates else hyperparameters[attr] for attr in hyperparameters})

		self.hyperparameters = hyperparameters

		self.value_and_grad,self.func,self.grad = value_and_gradient(func,grad,returns=True)

		self.alpha = LineSearch(self.func,self.grad,self.hyperparameters)

		if callback is None:
			def callback(parameters,track,attributes,hyperparameters):
				status = True
				return status
		self.callback = callback

		self.size = 0
		self.iteration = -1
		self.parameters = None
		self.optimizer = hyperparameters['optimizer']		
		self.modulo = hyperparameters['modulo']
		self.length = hyperparameters['length']
		self.attributes = hyperparameters['attributes']
		self.track = hyperparameters['track']
		self.iterations = range(int(hyperparameters['iterations']))
		self.sizes = hyperparameters['length'].get('attributes')
		self.status = hyperparameters['status']
		self.search = hyperparameters['search']
		self.eps = hyperparameters['eps']
		self.path = join(hyperparameters['path'],root=hyperparameters['cwd'])

		for attr in list(self.attributes):
			value = self.attributes[attr]
			if ((not isinstance(value,list)) and (not value)):
				self.attributes.pop(attr)
			elif ((isinstance(value,list)) and (value)):
				self.attributes[value] = []

		for attr in list(self.track):
			value = self.track[attr]
			if ((not isinstance(value,list)) and (not value)):
				self.track.pop(attr)
			elif ((isinstance(value,list)) and (value)):
				self.track[value] = []		

		super().__init__(hyperparameters=hyperparameters,system=system)

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

		for iteration in self.iterations:

			iteration += 1
			
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
			parameters = parameters + alpha*search
			alpha = optimizer.hyperparameters['alpha']
			search = -grad
			return parameters,search,alpha

		steps = self.size == 0

		for step in range(steps+1):

			init = self.size == 0

			value,grad,parameters = self.opt_step(iteration-init,state)

			if not init:
				parameters,search,alpha = update(parameters,value,grad,search,self)
			else:
				parameters = parameters
				alpha = self.hyperparameters['alpha']
				search = -grad

			self.attributes['search'].append(search)
			self.attributes['alpha'].append(alpha)

			state = self.opt_init(parameters)
			parameters = self.get_params(state)
			track = self.track		
			attributes = self.attributes
			hyperparameters = self.hyperparameters			
			self.status = self.callback(parameters,track,attributes,hyperparameters)

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

		if (self.sizes is not None) and (self.size > 0) and (self.size >= self.sizes):
			for attr in self.attributes:
				if self.attributes[attr]:
					self.attributes[attr].pop(0)

		self.iteration = iteration
		
		self.attributes['iteration'].append(iteration)
		self.attributes['parameters'].append(parameters)
		self.attributes['value'].append(value)
		self.attributes['grad'].append(grad)

		self.size += 1
		self.iteration += 1

		return value,grad,parameters


	def dump(self,iteration,state):
		'''
		Dump data
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		'''

		if self.path is None:
			return

		def parse(iteration,labels):
			string = delim.join([*(str(i) for i in labels if i is not None),*([str(i) for i in [iteration]])])
			return string

		start = (self.size==1) and (iteration<self.iterations.stop)
		done = (self.size>0) and (iteration == self.iterations.stop)
		other = (self.size == 0) or (self.modulo['dump'] is None) or (iteration%self.modulo['dump'] == 0)

		if not ((not self.status) or done or start or other):
			return

		path = self.path
		track = self.track
		attributes = self.attributes

		labels = [self.timestamp,self.key]
		size = min((len(track[attr]) for attr in track if len(track[attr])>0),default=0)
		iterations = range(size)
		default = nan

		data = {}
		for iteration in iterations:
			
			string = parse(iteration,labels)
			
			value = {}
			value.update({attr: attributes[attr][-1] if ((not self.status) or done or start) else default for attr in attributes})
			value.update({attr: track[attr][iteration] for attr in track})

			data[string] = value

		dump(data,path)

		return

	def load(self,iteration,state):
		'''
		Load data
		Args:
			iteration (int): optimizer iteration
			state (object): optimizer state
		Returns:
			iteration (int): optimizer iteration
			state (object): optimizer state
		'''

		if self.path is None:
			return

		def parse(string):
			iteration = int(string.split(delim)[-1])
			labels = string.split(delim)[:-1]
			return iteration,labels

		path = self.path
		default = {}
		data = load(path,default=default)

		track = {}
		for string in data:
			iteration,labels = parse(string)

			for attr in data[string]:
				if attr not in track:
					track[attr] = []
				value = data[string][attr]
				track[attr].append(value)

		for attr in self.track:
			if attr in track:
				self.track[attr].extend(track[attr])

		for attr in self.attributes:
			if attr in track:
				self.attributes[attr].extend(track[attr])

		self.size = min((len(self.track[attr]) for attr in self.track),default=self.size)

		if self.size:
			attr = 'iteration'	
			if attr in track:
				self.iteration = self.track[attr][-1]

			attr = 'parameters'		
			if attr in track:
				self.parameters = self.track[attr][-1]
		else:
			self.iteration = iteration
			self.parameters = self.get_params(state)
	
		self.iteration = max(iteration,self.iteration-1,-1)
		self.iterations = range(
			self.iterations.start+self.iteration+1,
			self.iterations.stop+self.iteration+1,
			self.iterations.step)				

		iteration = self.iteration
		state = self.opt_init(self.parameters)

		return iteration,state


class Optimizer(OptimizerBase):
	'''
	Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,attributes,hyperparameters) and returns status of optimization
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
	

class GradientDescent(OptimizerBase):
	'''
	Gradient Descent Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,attributes,hyperparameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={},system=None):

		defaults = {'attributes':{'beta':False}}		
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		super().__init__(func,grad,callback,hyperparameters=hyperparameters,system=system)

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
			parameters = parameters + alpha*search
			alpha = optimizer.hyperparameters['alpha']
			search = -grad
			return parameters,search,alpha

		steps = self.size == 0

		for step in range(steps+1):

			init = self.size == 0

			value,grad,parameters = self.opt_step(iteration-init,state)

			if not init:
				parameters,search,alpha = update(parameters,value,grad,search,self)
			else:
				parameters = parameters
				alpha = self.hyperparameters['alpha']
				search = -grad

			self.attributes['search'].append(search)
			self.attributes['alpha'].append(alpha)

			state = self.opt_init(parameters)
			parameters = self.get_params(state)
			track = self.track		
			attributes = self.attributes
			hyperparameters = self.hyperparameters			
			self.status = self.callback(parameters,track,attributes,hyperparameters)

		return state


class ConjugateGradient(OptimizerBase):
	'''
	Conjugate Gradient Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,attributes,hyperparameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={},system=None):

		defaults = {'beta':0,'search':{'alpha':'line_search','beta':None},'attributes':{'beta':[]}}
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		super().__init__(func,grad,callback,hyperparameters=hyperparameters,system=system)

		defaults = {}
		setter(self.hyperparameters,defaults,delimiter=delim,func=False)

		beta = self.search.get('beta')
		if beta is None:
			def beta(_grad,grad,search):
				beta = (_grad.dot(_grad))/(grad.dot(grad)) # Fletcher-Reeves
				return beta
		
		elif beta in ['fletcher_reeves']:
			def beta(_grad,grad,search):
				beta = (_grad.dot(_grad))/(grad.dot(grad)) # Fletcher-Reeves
				return beta
		
		elif beta in ['polak_ribiere']:
			def beta(_grad,grad,search):
				return max(0,(_grad.dot(_grad-grad))/grad.dot(grad)) # Polak-Ribiere
		
		elif beta in ['polak_ribiere_fletcher_reeves']:
			def beta(_grad,grad,search):
				beta = [(_grad.dot(_grad))/(grad.dot(grad)),max(0,(_grad.dot(_grad-grad))/grad.dot(grad))] # Polak-Ribiere-Fletcher-Reeves
				beta = -beta[0] if beta[1] < -beta[0] else beta[1] if abs(beta[1]) <= beta[0] else beta[0]
				return beta			

		elif beta in ['hestenes_stiefel']:
			def beta(_grad,grad,search):
				beta = (_grad.dot(_grad-grad))/(search.dot(_grad-grad)) # Hestenes-Stiefel
				return beta

		elif beta in ['dai_yuan']:
			def beta(_grad,grad,search):
				beta = (_grad.dot(_grad))/(search.dot(_grad-grad)) # Dai-Yuan https://doi.org/10.1137/S1052623497318992
				return beta								
					
		else:
			def beta(_grad,grad,search):
				beta = (_grad.dot(_grad))/(grad.dot(grad)) # Fletcher-Reeves
				return beta			

		self.beta = beta

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
			
			beta = optimizer.beta(_grad,grad,search)
			
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
		attributes = self.attributes
		hyperparameters = self.hyperparameters		
		self.status = self.callback(parameters,track,attributes,hyperparameters)

		return state


class Adam(OptimizerBase):
	'''
	Adam Optimizer class, with numpy optimizer API
	Args:
		func (callable): function to optimize, with signature function(parameters)
		grad (callable): gradient of function to optimize, with signature grad(parameters)
		callback (callable): callback function with signature callback(parameters,track,attributes,hyperparameters) and returns status of optimization
		hyperparameters (dict): optimizer hyperparameters
		system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,logconf,logging,cleanup,verbose)	
	'''
	def __init__(self,func,grad=None,callback=None,hyperparameters={},system=None):

		defaults = {'attributes':{'beta':False}}		
		setter(hyperparameters,defaults,delimiter=delim,func=False)

		super().__init__(func,grad,callback,hyperparameters=hyperparameters,system=system)

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


		if self.size == 0:

			value,grad,parameters = self.opt_step(iteration-1,state)

			alpha = self.hyperparameters['alpha']
			search = -grad

			self.attributes['alpha'].append(alpha)
			self.attributes['search'].append(search)

			state = self.opt_init(parameters)
			parameters = self.get_params(state)
			track = self.track		
			attributes = self.attributes
			hyperparameters = self.hyperparameters			
			self.status = self.callback(parameters,track,attributes,hyperparameters)

		value,grad,parameters = self.opt_step(iteration,state)

		alpha = self.attributes['alpha'][-1]
		search = -grad

		state = self._opt_update(iteration,grad,state)

		self.attributes['alpha'].append(alpha)
		self.attributes['search'].append(search)

		parameters = self.get_params(state)
		track = self.track		
		attributes = self.attributes
		hyperparameters = self.hyperparameters		
		self.status = self.callback(parameters,track,attributes,hyperparameters)

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