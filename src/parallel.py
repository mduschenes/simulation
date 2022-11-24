#!/usr/bin/env python

# Import python modules
import os,sys,copy,warnings,functools,itertools,inspect,timeit
import numpy as np

# import multiprocess as mp
# import multithreading as mt
import joblib
import multiprocessing as multiprocessing
import multiprocessing.dummy as multithreading
from progress.bar import Bar

# warnings.simplefilter("ignore", (UserWarning,DeprecationWarning,FutureWarning))
# warnings.simplefilter("ignore", (sp.sparse.SparseEfficiencyWarning))
# warnings.filterwarnings('error',category=sp.sparse.SparseEfficiencyWarning)

DELIMITER='__'
MAX_PROCESSES = 8

# Python Modules
from src.utils import mapping

# Logging
from src.system	 import Logger
name = __name__
path = os.getcwd()
file = 'logging.conf'
conf = os.path.join(path,file)
file = None #'log.log'
logger = Logger(name,conf,file=file)


def timing(verbose):
	''' 
	Timing function wrapper
	'''
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			if verbose:
				time = 0
				time = timeit.default_timer() - time
				value = func(*args,**kwargs)
				time = timeit.default_timer() - time
				logger.log(verbose,'%r: %r s'%(repr(func),time))
			else:
				value = func(*args,**kwargs)				
			return value
		return wrapper
	return decorator


# Wrapper class for function, with
# class args and kwargs assigned after
# classed args and kwrags
class Wrapper(object):
	def __init__(self,_func,*args,**kwargs):
		self.func = _func
		self.args = args
		self.kwargs = kwargs
		functools.update_wrapper(self, _func)
		return

	def __call__(self,*args,**kwargs):
		args = [*args,*self.args]
		kwargs = {**self.kwargs,**kwargs}
		return self.func(*args,**kwargs)

	def __repr__(self):
		return self.func.__repr__()

	def __str__(self):
		return self.func.__str__()


# Decorator with optional additional arguments
def decorator(*ags,**kwds):
	def wrapper(func):
		@functools.wraps(func)
		def function(*args,**kwargs):
			args = list(args)
			args.extend(ags)
			kwargs.update(kwds)
			return func(*args,**kwargs)
		return function
	return wrapper


# Context manager
class context(object):
	def __init__(self,func,*args,**kwargs):
		self.obj = func(*args,**kwargs)		
	def __enter__(self):
		return self.obj
	def __exit__(self, type, value, traceback):
		self.obj.__exit__(type,value,traceback)

# Empty Context manager
class emptycontext(object):
	def __init__(self,func,*args,**kwargs):
		self.obj = func	
	def __call__(self,*args,**kwargs):
		return self
	def __enter__(self,*args,**kwargs):
		return self.obj
	def __exit__(self, type, value, traceback):
		try:
			self.obj.__exit__(type,value,traceback)
		except:
			pass
		return


# Null Context manager
class nullcontext(object):
	def __init__(self,*args,**kwargs):
		return
	def __call__(self,*args,**kwargs):
		return self
	def __enter__(self,*args,**kwargs):
		return self
	def __exit__(self, type, value, traceback):
		return


def nullfunc(*args,**kwargs):
	return


class nullclass(object):
	pass


# Call function with proper signature
def call(cls,func,*args,**kwargs):
	try:
		func = getattr(cls,func,func)	
	except:
		pass
	assert callable(func), "Error - cls.func or func not callable"

	params = inspect.signature(func).parameters.values()
	arguments = []
	keywords = {}    

	for param in params:
		name = param.name
		default = param.default
		kind = str(param.kind)
		if kind in ['VAR_POSITIONAL']:
			if name in kwargs:
				keywords[name] = kwargs.get(name,default)
			arguments.extend(args)
		elif kind in ['VAR_KEYWORD']:
			keywords.update(kwargs)
		elif kind not in ['POSITIONAL_OR_KEYWORD'] and default is param.empty:
			pass
		else:
			keywords[name] = kwargs.get(name,default)

	return func(*arguments,**keywords)


def empty(obj,*attrs):
	class Empty(obj.__class__):
		def __init__(self): pass
	newobj = Empty()
	newobj.__class__ = obj.__class__
	for attr in inspect.getmembers(obj):
		attr = attr[0]
		if attr in attrs:
			setattr(newobj,attr,copy.deepcopy(getattr(obj,attr)))
	newobj.__dict__.update({attr: obj.__dict__.get(attr) 
						   for attr in attrs if not getattr(newobj,attr,False)})
	return newobj


# Pool class, similar to multiprocessing.Pool
class Pooler(object):
	def __init__(self,processes,pool=None,initializer=None,initargs=(),maxtasksperchild=None,context=None,verbose=None):
		'''
		Parallelize function
		Args:
			processes (int): Number of processes,default MAX_PROCESSES
			pool (str): Parallelization backend, Pool
			initializer (callable): Initializer of arguments
			initargs (iterable[object]): Initial arguments
			maxtasksperchild (int): Maximum tasks per child process
			context (context): Context manager class
			verbose (str,int,bool): Verbosity of parallelization 
		'''				
		self.set_processes(processes)
		self.set_pool(pool)
		self.set_initializer(initializer)
		self.set_initargs(initargs)
		self.set_maxtasksperchild(maxtasksperchild)
		self.set_context(context)
		self.set_verbose(verbose)
		return

	@timing(False)	
	def __call__(self,module,func,iterable,args=(),kwds={},callback_args=(),callback_kwds={},callback=nullfunc,error_callback=nullfunc):
		'''
		Call function in parallel
		Args:
			module (str): Type of parallelization to call i.e) apply_async
			func (callable): Function to call in parallel, with signature func(*args,**kwargs)
			iterable (int,iterable[object],dict[str,iterable[object]]): Iterable of arguments to iterate in parallel, 
				either integer to call range(iterable) -> func(i), or iterable of objects [i] -> func(i), 
				or dictionary of combinations of arguments to iterate over in all combinations
			args (iterable[object]): Positional arguments to pass to func
			kwds (dict[str,object]): Keyword arguments to pass to func
			callback_args (iterable[object]): Positional arguments to pass to callback
			callback_kwds (dict[str,object]): Keyword arguments to pass to callback
			callback (callable): Callback function to collect parallel outputs, with signature callback(value,*callback_args,**callback_kwds)
			error_callback (callable): Error callback function if errors occur, with signature callback(value,*callback_args,**callback_kwds)
		'''				
		with self.get_pool()(
			processes=self.get_processes(),
			initializer=self.get_initializer(),
			initargs=self.get_initargs(),
			maxtasksperchild=self.get_maxtasksperchild(),
			context=self.get_context()) as pool:

			self.set_iterable(iterable)
			jobs = (getattr(pool,module)(
					func=Wrapper(func,*args,**{**kwds,**i}),
					**(dict(callback=Wrapper(callback,*i,*callback_args,**{**callback_kwds,**i}),
					error_callback=Wrapper(error_callback,*i,*callback_args,**{**callback_kwds,**i}))
					if 'async' in module else dict()))
					for i in self.get_iterable())
						

			start = timeit.default_timer()	
			pool.close()
			pool.join()
			end = timeit.default_timer()

			if not self.get_null():
				logger.log(self.get_verbose(),"processes: %d, time: %0.3e"%(self.get_processes(),end-start))							

		return

	def set_pool(self,pool):
		attr = 'pool'  
		self.set_null()
		if self.get_null():
			value = nullPool
		elif pool in [False]:
			value = nullPool
		elif pool in [None,True]:
			value = Pool 
		elif callable(pool):
			value = emptycontext(pool)	
		else:
			value = pool		
		setattr(self,attr,value)
		return 
	def get_pool(self,default=None):
		attr = 'pool'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_processes(self,processes):
		attr = 'processes'
		default = MAX_PROCESSES
		processes = default if processes is None else processes
		processes = min(processes,MAX_PROCESSES-1)
		setattr(self,attr,processes)
		self.set_null()
		return
	def get_processes(self,default=None):
		attr = 'processes'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_initializer(self,initializer):
		attr = 'initializer'
		value = initializer
		setattr(self,attr,value)
		return
	def get_initializer(self,default=None):
		attr = 'initializer'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_initargs(self,initargs):
		attr = 'initargs'
		value = initargs
		setattr(self,attr,value)
		return
	def get_initargs(self,default=None):
		attr = 'initargs'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_maxtasksperchild(self,maxtasksperchild):
		attr = 'maxtasksperchild'
		value = maxtasksperchild
		setattr(self,attr,value)
		return

	def get_maxtasksperchild(self,default=None):
		attr = 'maxtasksperchild'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_context(self,context):
		attr = 'context'
		value = context
		setattr(self,attr,value)
		return
	def get_context(self):
		attr = 'context'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_verbose(self,verbose):  
		attr = 'verbose'
		if verbose is None:
			value = False 
		else:
			value = verbose
		setattr(self,attr,value)
		return 
	def get_verbose(self,default=None):
		attr = 'verbose'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_iterable(self,iterable):
		attr = 'iterable'
		if isinstance(iterable,dict) and all(isinstance(i,list) for i in iterable):
			value = (mapping(**dict(zip(iterable,values)))
							for values in itertools.product(*[iterable[key] 
															for key in iterable]))
		elif all(isinstance(i,dict) for i in iterable):
			value = (mapping(**i) for i in iterable)
		elif isinstance(iterable,int):
			value = (mapping(i) for i in range(iterable))
		else:
			value = (mapping(i) for i in iterable)
		setattr(self,attr,value)
		return
	def get_iterable(self,default=None):
		attr = 'iterable'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value		

	def set_null(self):
		attr = 'null'
		min_processes = 2
		value = self.get_processes() < min_processes
		setattr(self,attr,value)
		return
	def get_null(self,default=None):
		attr = 'null'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

# nullPool class, similar to multiprocessing.Pool
class Pool(multiprocessing.pool.Pool):
	pass

# nullPool class, similar to multiprocessing.Pool
class nullPool(object):
	def __init__(self,processes=None,initializer=None,initargs=(),maxtasksperchild=None,context=None,verbose=None):
		'''
		Parallelize function
		Args:
			processes (int): Number of processes,default MAX_PROCESSES
			pool (str): Parallelization backend, Pool
			initializer (callable): Initializer of arguments
			initargs (iterable[object]): Initial arguments
			maxtasksperchild (int): Maximum tasks per child process
			context (context): Context manager class
			verbose (str,int,bool): Verbosity of parallelization 
		'''		
		self.set_processes(processes)
		self.set_verbose(verbose)		
		return

	def __enter__(self,*args,**kwargs):
		return self
	def __exit__(self, type, value, traceback):
		return 

	@timing(False)	
	def apply(self,func,args=(),kwds={}):
		return func(*args,**kwds)

	@timing(False)
	def apply_async(self,func,args=(),kwds={},callback=nullfunc,error_callback=nullfunc):
		try:
			callback(func(*args,**kwds))
		except:
			error_callback(func(*args,**kwds))
		return
	@timing(False)	
	def map(self,func,iterable,chunksize=None):
		return list(map(func,iterable))
	@timing(False)		
	def map_async(self,func,iterable,chunksize=None,callback=nullfunc,error_callback=nullfunc):
		try:
			map(callback,list(map(func,iterable)))
		except:
			map(error_callback,list(map(func,iterable)))
		return 
	@timing(False)		
	def imap(self,func,iterable,chunksize=None):
		return list(map(func,iterable))
	@timing(False)	
	def imap_unordered(self,func,iterable,chunksize=None):
		return list(map(func,iterable))
	@timing(False)	
	def starmap(self,func,iterable,chunksize=None):
		return list(map(func,iterable))
	@timing(False)	
	def starmap_async(self,func,iterable,chunksize=None,callback=nullfunc,error_callback=nullfunc):
		try:
			map(callback,list(map(func,iterable)))
		except:
			map(error_callback,list(map(func,iterable)))
		return 		
	def close(self):
		pass
	def join(self):
		pass

	def set_processes(self,processes):
		attr = 'processes'
		default = 1
		processes = default if processes is None else processes
		processes = min(processes,MAX_PROCESSES-1)
		setattr(self,attr,processes)
		self.set_null()
		return
	def get_processes(self,default=None):
		attr = 'processes'
		if not hasattr(self,attr):
			self.set_processes(default)
		return getattr(self,attr)

	def set_verbose(self,verbose):  
		attr = 'verbose'
		if verbose is None:
			value = False 
		else:
			value = verbose
		setattr(self,attr,value)
		return 
	def get_verbose(self,default=None):
		attr = 'verbose'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_iterable(self,iterable):
		attr = 'iterable'
		if isinstance(iterable,dict) and all(isinstance(i,list) for i in iterable):
			value = (mapping(**dict(zip(iterable,values)))
							for values in itertools.product(*[iterable[key] 
															for key in iterable]))
		elif all(isinstance(i,dict) for i in iterable):
			value = (mapping(**i) for i in iterable)
		elif isinstance(iterable,int):
			value = (mapping(i) for i in range(iterable))
		else:
			value = (mapping(i) for i in iterable)
		setattr(self,attr,value)
		return
	def get_iterable(self,default=None):
		attr = 'iterable'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_null(self):
		attr = 'null'
		min_processes = 2
		value = self.get_processes() < min_processes
		setattr(self,attr,value)
		return
	def get_null(self,default=None):
		attr = 'null'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	


# Parallelize iterations, similar to joblib
class Parallelize(object):
	def __init__(self,n_jobs,backend=None,parallel=None,delayed=None,prefer=None,verbose=False):
		'''
		Parallelize function
		Args:
			n_jobs (int): Number of processes,default MAX_PROCESSES
			backend (str): Parallelization backend, default 'loky'
			delayed (callable): Function for delayed execution, with signature delayed(func)
			prefer (str): Type of parallelization preference, default 'threads'
			verbose (str,int,bool): Verbosity of parallelization 
		'''
		self.set_n_jobs(n_jobs)
		self.set_backend(backend)
		self.set_parallel(parallel)
		self.set_delayed(delayed)
		self.set_prefer(prefer)
		self.set_verbose(verbose)
		return
	@timing(False)
	def __call__(self,func,iterable,values,args=(),kwds={}):
		'''
		Call function in parallel
		Args:
			func (callable): Function to call in parallel, with signature func(*args,**kwargs)
			iterable (int,iterable[object],dict[str,iterable[object]]): Iterable of arguments to iterate in parallel, 
				either integer to call range(iterable) -> func(i), or iterable of objects [i] -> func(i), 
				or dictionary of combinations of arguments to iterate over in all combinations
			values (list): List to append in place parallel outputs to (in order of arguments passed in iterable)
			args (iterable[object]): Positional arguments to pass to func
			kwds (dict[str,object]): Keyword arguments to pass to func
		'''		
		with self.get_parallel()(n_jobs=self.get_n_jobs(),backend=self.get_backend(),prefer=self.get_prefer()) as parallel:           

			self.set_iterable(iterable)

			jobs = (self.get_delayed()(func)(*i,*args,**{**kwds,**i}) 
					for i in self.get_iterable())

			start = timeit.default_timer()	
			values.extend(parallel(jobs))
			end = timeit.default_timer()

			if not self.get_null():
				logger.log(self.get_verbose(),"n_jobs: %d, time: %0.3e"%(self.get_n_jobs(),end-start))
		return 
	def __enter__(self,*args,**kwargs):
		return self
	def __exit__(self, type, value, traceback):
		return 

	def set_n_jobs(self,n_jobs):  
		attr = 'n_jobs'
		if n_jobs is None:
			n_jobs = 1  
		value = max(1,joblib.effective_n_jobs(n_jobs))
		setattr(self,attr,value)
		self.set_null()		
		return 
	def get_n_jobs(self,default=None):
		attr = 'n_jobs'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_backend(self,backend):  
		attr = 'backend'
		if backend is None:
			value = 'loky'  
		else:
			value = backend
		setattr(self,attr,value)
		return 
	def get_backend(self,default=None):
		attr = 'backend'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_parallel(self,parallel):
		attr = 'parallel'  
		self.set_null()
		if self.get_null():
			value = nullParallel
		elif parallel in [False]:
			value = nullParallel
		elif parallel in [None,True]:
			value = Parallel 
		elif callable(parallel):
			value = emptycontext(parallel)	
		else:
			value = parallel
		setattr(self,attr,value)
		return 
	def get_parallel(self,default=None):
		attr = 'parallel'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_delayed(self,delayed):
		attr = 'delayed'  
		if delayed is None:
			value = Delayed 
		else:
			value = delayed
		setattr(self,attr,value)
		return 
	def get_delayed(self,default=None):
		attr = 'delayed'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_prefer(self,prefer):  
		attr = 'prefer'
		if prefer is None:
			value = None 
		else:
			value = prefer
		setattr(self,attr,value)
		return 
	def get_prefer(self,default=None):
		attr = 'prefer'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value			

	def set_verbose(self,verbose):  
		attr = 'verbose'
		if verbose is None:
			value = False 
		else:
			value = verbose
		setattr(self,attr,value)
		return 
	def get_verbose(self,default=None):
		attr = 'verbose'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value			

	def set_iterable(self,iterable):
		attr = 'iterable'
		if isinstance(iterable,dict) and all(isinstance(i,list) for i in iterable):
			value = (mapping(**dict(zip(iterable,values)))
							for values in itertools.product(*[iterable[key] 
															for key in iterable]))
		elif all(isinstance(i,dict) for i in iterable):
			value = (mapping(**i) for i in iterable)
		elif isinstance(iterable,int):
			value = (mapping(i) for i in range(iterable))
		else:
			value = (mapping(i) for i in iterable)
		setattr(self,attr,value)
		return
	def get_iterable(self,default=None):
		attr = 'iterable'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_null(self):
		attr = 'null'
		min_n_jobs = 2
		value = self.get_n_jobs() < min_n_jobs
		setattr(self,attr,value)
		return
	def get_null(self,default=None):
		attr = 'null'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value			


# Parallel class using joblib
class Parallel(joblib.Parallel):
	pass


# null Parallel class using joblib
class nullParallel(joblib.Parallel):
	@timing(False)
	def __call__(self,jobs):
		return [func(*args,**kwargs) for func,args,kwargs in jobs]


# Delayed function call for parallelization using joblib
class Delayed(object):
	def __init__(self,function, check_pickle=None):
		self.function = joblib.delayed(function)
		return
	def __call__(self,*args,**kwargs):
		return self.function(*args,**kwargs)



# TODO: Modify to match class API of concurrent futures 
# (can be also similar to structure of __call__ and get_exucator_pool in place of get_pool, with a Future() and nullFuture() class as the context)
# Futures class, similar to concurrent.futures
class Futures(object):
	def __init__(self,processes,pool=None,initializer=None,initargs=(),maxtasksperchild=None,context=None,verbose=None):
		'''
		Parallelize function
		Args:
			processes (int): Number of processes,default MAX_PROCESSES
			pool (str): Parallelization backend, Pool
			initializer (callable): Initializer of arguments
			initargs (iterable[object]): Initial arguments
			maxtasksperchild (int): Maximum tasks per child process
			context (context): Context manager class
			verbose (str,int,bool): Verbosity of parallelization 
		'''		
		self.set_processes(processes)
		self.set_pool(pool)
		self.set_initializer(initializer)
		self.set_initargs(initargs)
		self.set_maxtasksperchild(maxtasksperchild)
		self.set_context(context)
		self.set_verbose(verbose)
		return
	@timing(False)	
	def __call__(self,module,func,iterable,args=(),kwds={},callback_args=(),callback_kwds={},callback=nullfunc,error_callback=nullfunc):
		'''
		Call function in parallel
		Args:
			module (str): Type of parallelization to call i.e) apply_async
			func (callable): Function to call in parallel, with signature func(*args,**kwargs)
			iterable (int,iterable[object],dict[str,iterable[object]]): Iterable of arguments to iterate in parallel, 
				either integer to call range(iterable) -> func(i), or iterable of objects [i] -> func(i), 
				or dictionary of combinations of arguments to iterate over in all combinations
			args (iterable[object]): Positional arguments to pass to func
			kwds (dict[str,object]): Keyword arguments to pass to func
			callback_args (iterable[object]): Positional arguments to pass to callback
			callback_kwds (dict[str,object]): Keyword arguments to pass to callback
			callback (callable): Callback function to collect parallel outputs, with signature callback(value,*callback_args,**callback_kwds)
			error_callback (callable): Error callback function if errors occur, with signature callback(value,*callback_args,**callback_kwds)
		'''				

		with self.get_pool()(
			processes=self.get_processes(),
			initializer=self.get_initializer(),
			initargs=self.get_initargs(),
			maxtasksperchild=self.get_maxtasksperchild(),
			context=self.get_context()) as pool:

			self.set_iterable(iterable)
			jobs = (getattr(pool,module)(
					func=Wrapper(func,*args,**{**kwds,**i}),
					**(dict(callback=Wrapper(callback,*i,*callback_args,**{**callback_kwds,**i}),
					error_callback=Wrapper(error_callback,*i,*callback_args,**{**callback_kwds,**i}))
					if 'async' in module else dict()))
					for i in self.get_iterable())
						

			start = timeit.default_timer()	
			pool.close()
			pool.join()
			end = timeit.default_timer()

			logger.log(self.get_verbose(),"processes: %d, time: %0.3e"%(self.get_processes(),end-start))							

		return

	def set_pool(self,pool):
		attr = 'pool'  
		self.set_null()
		if self.get_null():
			value = nullPool
		elif value in [False]:
			value = nullPool
		elif pool in [None,True]:
			value = Pool 
		elif callable(pool):
			value = emptycontext(pool)	
		else:
			value = pool		
		setattr(self,attr,value)
		return 
	def get_pool(self,default=None):
		attr = 'pool'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_processes(self,processes):
		attr = 'processes'
		default = 1
		processes = default if processes is None else processes
		processes = min(processes,MAX_PROCESSES-1)
		setattr(self,attr,processes)
		self.set_null()
		return
	def get_processes(self,default=None):
		attr = 'processes'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_initializer(self,initializer):
		attr = 'initializer'
		value = initializer
		setattr(self,attr,value)
		return
	def get_initializer(self,default=None):
		attr = 'initializer'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_initargs(self,initargs):
		attr = 'initargs'
		value = initargs
		setattr(self,attr,value)
		return
	def get_initargs(self,default=None):
		attr = 'initargs'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_maxtasksperchild(self,maxtasksperchild):
		attr = 'maxtasksperchild'
		value = maxtasksperchild
		setattr(self,attr,value)
		return
	def get_maxtasksperchild(self,default=None):
		attr = 'maxtasksperchild'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_context(self,context):
		attr = 'context'
		value = context
		setattr(self,attr,value)
		return
	def get_context(self):
		attr = 'context'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_verbose(self,verbose):  
		attr = 'verbose'
		if verbose is None:
			value = False 
		else:
			value = verbose
		setattr(self,attr,value)
		return 
	def get_verbose(self,default=None):
		attr = 'verbose'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_iterable(self,iterable):
		attr = 'iterable'
		if isinstance(iterable,dict) and all(isinstance(i,list) for i in iterable):
			value = (mapping(**dict(zip(iterable,values)))
							for values in itertools.product(*[iterable[key] 
															for key in iterable]))
		elif all(isinstance(i,dict) for i in iterable):
			value = (mapping(**i) for i in iterable)
		elif isinstance(iterable,int):
			value = (mapping(i) for i in range(iterable))
		else:
			value = (mapping(i) for i in iterable)
		setattr(self,attr,value)
		return
	def get_iterable(self,default=None):
		attr = 'iterable'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value		

	def set_null(self):
		attr = 'null'
		min_processes = 2
		value = self.get_processes() < min_processes
		setattr(self,attr,value)
		return
	def get_null(self,default=None):
		attr = 'null'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value




def catch(update,exceptions,raises,iterations=1000):
	'''
	Wrapper to loop through function and catch exceptions, updating args and kwargs until no exceptions
	Args:
		update (callable): function with signature update(exception,*args,**kwargs) to update *args and **kwargs after exceptions
		exceptions (tuple): exceptions that invoke updating of *args and **kwargs
		raises (tuple): exceptions that raise exception and do not update *args and **kwargs
		iterations (int): maximum number of iterations before exiting
	Returns:
		func (callable): wrapped function for catching exceptions
	'''
	def wrap(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			result = None
			exception = Exception
			iteration = 0
			while (exception is not None) and (iteration < iterations):
				try:
					result = func(*args,**kwargs)
					exception = None
				except Exception as e:
					exception = e
					if isinstance(exception,exceptions):
						update(exception,*args,**kwargs)
					elif isinstance(exception,raises):
						raise exception
				iteration += 1
			if exception is not None:
				raise exception
			return result
		return wrapper
	return wrap

