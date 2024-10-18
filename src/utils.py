#!/usr/bin/env python

# Import python modules
import os,sys,itertools,ast,operator
from copy import deepcopy as copy
from string import ascii_lowercase as characters
from math import prod

from functools import partial,wraps
from natsort import natsorted
import hashlib
import argparse

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
	# log = file if hasattr(file,'write') else sys.stderr
	# traceback.print_stack(file=log)
	# log.write(warnings.formatwarning(message, category, filename, lineno, line))
	return
warnings.showwarning = warn_with_traceback

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp
import pandas as pd
import scipy.special as ospsp

_partial = partial
partial = lambda func,*args,**kwargs: wraps(func)(_partial(func,*args,**kwargs))

# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

ENVIRON = 'NUMPY_BACKEND'
DEFAULT = 'jax'
BACKENDS = ['jax','autograd','jax.autograd','numpy']

backend = os.environ.get(ENVIRON,DEFAULT).lower()

assert backend in BACKENDS, "%s=%s not in allowed %r"%(ENVIRON,backend,BACKENDS)


if backend in ['jax','jax.autograd']:
	
	envs = {
		'JAX_DISABLE_JIT':False,
		'JAX_PLATFORMS':'',
		'JAX_PLATFORM_NAME':'',
		'TF_CPP_MIN_LOG_LEVEL':5,
		'JAX_TRACEBACK_FILTERING':'off',
		# "XLA_FLAGS":(
		# 	"--xla_cpu_multi_thread_eigen=false "
		# 	"intra_op_parallelism_threads=1"),
	}
	for var in envs:
		os.environ[var] = str(envs[var])


	import jax
	import jax.numpy as np
	import jax.scipy as sp

	import jax.example_libraries.optimizers
	from jax.tree_util import register_pytree_node_class as tree_register
	from jax.tree_util import tree_map as tree_map

	import quimb as qu
	import quimb.tensor as qtn

	import absl.logging
	absl.logging.set_verbosity(absl.logging.INFO)

	configs = {
		'jax_disable_jit':False,
		'jax_platforms':'',
		'jax_platform_name':'',
		'jax_enable_x64': True,
		}
	for name in configs:
		jax.config.update(name,configs[name])

	mapper = tree_map

elif backend in ['autograd']:

	import autograd
	import autograd.numpy as np
	import autograd.scipy as sp
	import autograd.scipy.linalg

	import quimb as qu
	import quimb.tensor as qtn

	def tree_map(func,*trees,is_leaf=None,**kwargs):
		'''
		Perform function on trees
		Args:
			func (callable): Callable function with signature func(*trees,**kwargs)
			trees (iterable[pytree]): Pytrees of identical structure for function
			is_leaf (type,iterable[type],callable): Boolean whether tree nodes are leaves
			kwargs (dict): Additional keyword arguments for function
		Returns:
			tree (pytree): Tree with mapped function
		'''
		if not callable(is_leaf):
			types = (dict,tuple,list,) if is_leaf is None else (*is_leaf,) if isinstance(is_leaf,iterables) else (is_leaf,)
			is_leaf = lambda tree,types=types: isinstance(tree,types)	

		if not callable(func):
			return

		def mapper(*trees,func=None,is_leaf=None,**kwargs):
			if not trees:
				return
			tree = trees[0]
			if is_leaf(tree):
				for key in tree:
					node = tree[key] if isinstance(tree,dict) else key
					nodes = (tree[key] if isinstance(tree,dict) else tree for tree in trees)
					if is_leaf(node):
						mapper(*nodes,func=func,is_leaf=is_leaf,**kwargs)
					else:
						leaf = func(*nodes,**kwargs)
						if isinstance(tree,dict):
							tree[key] = leaf
						else:
							tree[tree.index(key)] = leaf
			return

		trees = (*deepcopy(trees[:1]),*trees[1:])
		mapper(*trees,func=func,is_leaf=is_leaf,**kwargs)
		tree = trees[0]

		return tree

	mapper = tree_map

elif backend in ['numpy']:

	import numpy as np
	import scipy as sp
	import pandas as pd
	import scipy.special as spsp

	import quimb as qu
	import quimb.tensor as qtn

	mapper = map


np.set_printoptions(linewidth=1000,formatter={**{dtype: (lambda x: format(x, '0.6e')) for dtype in ['float','float64',np.float64,np.float32]}})
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Constants

class Null(object):
	def __str__(self):
		return 'Null'
	def __repr__(self):
		return self.__str__()

class none(object):
	def __init__(self,default=0,*args,**kwargs):
		self.default = default
		return
	def __call__(self,*args,**kwargs):
		return self.default

null = Null()

class structure(object):
	'''
	array placeholder class
	Args:
		args (iterable): Dataframe arguments
		kwargs (dict): Dataframe keyword arguments
	Returns:
		out (array): dataframe
	'''
	def __init__(self,shape=None,dtype=None,**kwargs):
		self._shape = () if shape is None else (shape,) if not isinstance(shape,iterables) else (*shape,)
		self._dtype = float if dtype is None else dtype
		self._data = asscalar(asarray(0,dtype=self.dtype))
		return 

	@property
	def shape(self):
		return self._shape
	
	@property
	def size(self):
		return prod(self.shape)
	
	@property
	def ndim(self):
		return len(self.shape)

	@property
	def dtype(self):
		return self._dtype

	@property
	def data(self):
		return self._data

	@property
	def nbytes(self):
		return sys.getsizeof(self)

	def __len__(self):
		return self.shape[0]

	def __iter__(self):
		for i in range(len(self)):
			yield self.data

	def __getitem__(self,item):
		return self.data

	def __setitem__(self,item,value):
		self.data = self.dtype(value)
		return




# Types


if backend in ['jax','jax.autograd']:

	itg = np.integer
	flt = np.float32
	dbl = np.float64
	uint = np.uint32

	pi = np.pi
	e = np.exp(1)

	nan = np.nan
	inf = np.inf
	integers = (int,np.integer,getattr(onp,'int',int),onp.integer)
	floats = (float,np.floating,getattr(onp,'float',float),onp.floating)
	scalars = (*integers,*floats,str,type(None))
	arrays = (np.ndarray,onp.ndarray,structure)
	tensors = (qtn.Tensor,qtn.TensorNetwork,qtn.Gate,qtn.MatrixProductState)

	iterables = (*arrays,list,tuple,set)
	nulls = (Null,)
	delim = '.'
	separ = '_'


	def disp(*args,**kwargs):
		with jax.disable_jit():
			print(*args,**kwargs)
		return

elif backend in ['autograd']:

	itg = np.integer
	flt = np.float32
	dbl = np.float64
	uint = np.uint32

	pi = np.pi
	e = np.exp(1)

	nan = np.nan
	inf = np.inf
	integers = (int,np.integer,getattr(onp,'int',int),onp.integer)
	floats = (float,np.floating,getattr(onp,'float',float),onp.floating)
	scalars = (*integers,*floats,str,type(None))	
	arrays = (np.ndarray,onp.ndarray,np.numpy_boxes.ArrayBox,structure)
	tensors = (qtn.Tensor,qtn.TensorNetwork)

	iterables = (*arrays,list,tuple,set)
	nulls = (Null,)
	delim = '.'
	separ = '_'	

	def disp(*args,**kwargs):
		print(*args,**kwargs)
		return


elif backend in ['numpy']:

	itg = np.integer
	flt = np.float32
	dbl = np.float64
	uint = np.uint32

	pi = np.pi
	e = np.exp(1)

	nan = np.nan
	inf = np.inf
	integers = (int,np.integer,getattr(onp,'int',int),onp.integer)
	floats = (float,np.floating,getattr(onp,'float',float),onp.floating)
	scalars = (*integers,*floats,str,type(None))	
	arrays = (np.ndarray,onp.ndarray,)
	tensors = (qtn.Tensor,qtn.TensorNetwork)


	iterables = (*arrays,list,tuple,set)
	nulls = (Null,)
	delim = '.'
	separ = '_'	

	def disp(*args,**kwargs):
		print(*args,**kwargs)
		return


# Libraries
if backend in ['jax','jax.autograd']:

	optimizer_libraries = jax.example_libraries.optimizers

elif backend in ['autograd','numpy']:

	optimizer_libraries = []

class mapping(dict):
	def __init__(self,*args,**kwargs):
		'''
		Mapping for args and kwargs
		Args:
			args (tuple[object]): Positional arguments
			kwargs (dict[str,object]): Keyword arguments
		'''
		self.args = list(args)
		self.kwargs = dict(kwargs)
		return

	def __iter__(self):
		return self.args.__iter__()

	def __getitem__(self,item):
		return self.kwargs[item]

	def __setitem__(self,item,value):
		if isinstance(item,integers):
			self.args[item] = value
		else:
			self.kwargs[item] = value
		return

	def __len__(self):
		return len(self.args)+len(self.kwargs)

	def __str__(self):
		return str(self.args) + ' , ' + str(self.kwargs)

	def keys(self):
		return self.kwargs.keys()

	def values(self):
		return self.kwargs.values()

class argparser(argparse.ArgumentParser):
	def __init__(self,arguments=None,wrappers=None):
		'''
		Parse command line arguments
		Args:
			arguments (str,iterable[str],dict[str,dict[str,object]]): Command line arguments {argument:{option:value}}
			wrappers (dict[str,[str,callable]]: Wrappers of arguments, either string for argument name, or callable(kwarg,wrappers,kwargs)
		'''

		# TODO: Allow for non-string types of iterable of values parsed by action (comma-separated values)

		class action(argparse.Action):
			def __call__(self, parser, namespace, values, option_string=None):
				_values = []
				delimiter = ','
				iterable = isinstance(values,list)
				if not iterable:
					values = [values]
				for value in values:
					for val in str(value).split(delimiter):
						_values.append(self.type(val))
					if iterable:
						setattr(namespace,self.dest,_values)
				if not iterable:
					setattr(namespace,self.dest,_values[-1])

				return

		defaults = {
			'action':action
		}

		nulls = {
			'action':['type','nargs','default']
		}

		default = lambda argument: ({
			'help':argument.replace('--','').capitalize(),
			'type':str,
			'nargs':'?',
			'default':None,
			})

		if arguments is None:
			arguments = '--args'
		if isinstance(arguments,str):
			arguments = [arguments]
		if not isinstance(arguments,dict):
			arguments = {
				'--%s'%(argument.replace('--','')):{**default(argument)}
				for argument in arguments
			}
		else:
			arguments = {
				'--%s'%(argument.replace('--','')):{
					**default(argument),
					**(arguments[argument] if isinstance(arguments[argument],dict) else 
					   dict(default=arguments[argument]))
					}
				for argument in arguments
			}

		if wrappers is None:
			wrappers = {}

		super().__init__()

		for i,argument in enumerate(arguments):


			name = '%s'%(argument.replace('--',''))
			options = {option: arguments[argument][option] for option in arguments[argument]}

			if options.get('action') is None:
				for null in nulls:
					if null in options:
						for option in nulls[null]:
							options.pop(option,None);

				options.update({option: options.get(option,defaults[option]) for option in defaults if option not in options})
				options.update({
					**{option:'?' if options.get(option) not in ['*','+'] or i>0 else '*' for option in ['nargs'] if option in options},
					**{option: argparse.SUPPRESS for option in ['default'] if option in options}
					})
				names = [name]
				self.add_argument(*names,**options)

			name = '--%s'%(argument.replace('--',''))
			options = {option: arguments[argument][option] for option in arguments[argument]}
			
			for null in nulls:
				if null in options:
					for option in nulls[null]:
						options.pop(option,None);

			options.update({option: options.get(option,defaults[option]) for option in defaults if option not in options})
			options.update({'dest':options.get('dest',argument.replace('--',''))})
			names = [name]
			self.add_argument(*names,**options)

		kwargs,args = self.parse_known_args()

		kwargs = {**dict(**vars(kwargs))}

		for kwarg in wrappers:
			name = kwarg.replace('--','')
			func = wrappers[kwarg] if callable(wrappers[kwarg]) else lambda kwarg,wrappers,kwargs: kwargs[wrappers[kwarg].replace('--','')]
			kwargs[name] = func(kwarg,wrappers,kwargs)

		self.args = args
		self.kwargs = kwargs

		return

	def __iter__(self):
		return self.args.__iter__()

	def __getitem__(self,item):
		return self.kwargs[item]

	def __len__(self):
		return len(self.args)+len(self.kwargs)

	def keys(self):
		return self.kwargs.keys()

	def values(self):
		return self.kwargs.values()


def insertion(obj,index,value):
	'''
	Insert value into obj at index
	Args:
		obj (iterable[object],dict): Object to insert into
		index (int): Index to insert into
		value (object,dict): Value to insert (if obj is a dict then value if value is a dict, the obj will be updated with value)
	Returns:
		obj (iterable[object],dict): Object with inserted value at index
	'''

	if isinstance(obj,dict):
		index = len(obj)+index if index < 0 else index
		obj = type(obj)({
			**{key: obj[key] for i,key in enumerate(obj) if i < index},
			**(value if isinstance(value,dict) else {index:value}),
			**{key: obj[key] for i,key in enumerate(obj) if i >= index}
			})
	else:
		obj.insert(index,value)

	return obj


def removal(obj,index):
	'''
	Remove value of obj at index
	Args:
		obj (iterable[object],dict): Object to insert into
		index (int): Index to insert into
	Returns:
		obj (iterable[object],dict): Object with inserted value at index
	'''

	if isinstance(obj,dict):
		if index < 0:
			return obj
		obj = type(obj)({
			**{key: obj[key] for i,key in enumerate(obj) if i < index},
			**{key: obj[key] for i,key in enumerate(obj) if i > index}
			})
	else:
		obj.remove(index,None)

	return obj


def finfo(dtype=float):
	'''	
	Get machine precision information for dtype
	Args:
		dtype (datatype): Datatype to get machine precision
	Returns:
		finfo (finfo): Machine precision information
	'''	
	return np.finfo(dtype)

def epsilon(dtype=float,eps=None):
	'''	
	Get machine precision epsilon for dtype
	Args:
		dtype (datatype): Datatype to get machine precision
		eps (float): Relative machine precision multiplier
	Returns:
		eps (array): Machine precision
	'''
	eps = 1 if eps is None else eps
	try:
		dtype = finfo(dtype).eps
	except:
		dtype = float
		dtype = finfo(dtype).eps

	eps = eps*dtype

	return eps

if backend in ['jax','jax.autograd']:
	
	def inplace(obj,index,item,op=None,**kwargs):
		'''
		Apply operation with item at index of object
		Args:
			obj (object): Object to apply operation
			index (object): Index to apply operation item
			item (object): Item to operate with
			op (str): Operation to apply at index, allowed strings in ['set','get','apply','add','multiply','divide','power','min','max']
			kwargs (dict): Additional keyword arguments for operation
		Returns:
			obj (object): Object with applied item at index
		'''
		
		# TODO merge indexing for different numpy backends (jax vs autograd)

		op = 'set' if op is None else op

		obj = getattr(obj.at[index],op)(item,**kwargs)

		return obj

	def index(obj,index):
		'''
		Index object
		Args:
			obj (object): Object to apply operation
			index (object): Index to apply operation item
		Returns:
			item (object): Item at index
		'''
		return obj[index]

elif backend in ['autograd','numpy']:
	
	def inplace(obj,index,item,op=None,**kwargs):
		'''
		Apply operation with item at index of object
		Args:
			obj (object): Object to apply operation
			index (object): Index to apply operation item
			item (object): Item to operate with
			op (str): Operation to apply at index, allowed strings in ['set','get','apply','add','multiply','divide','power','min','max']
			kwargs (dict): Additional keyword arguments for operation
		Returns:
			obj (object): Object with applied item at index
		'''
		
		# TODO merge indexing for different numpy backends (jax vs autograd)

		op = 'set' if op is None else op

		if op in ['set']:
			obj[index] = item
		elif op in ['get']:
			obj = obj[index]
		elif op in ['apply']:
			obj = item.at(obj,index)
		elif op in ['add']:
			obj[index] += item
		elif op in ['multiply']:
			obj[index] *= item
		elif op in ['divide']:
			obj[index] /= item						
		elif op in ['power']:
			obj[index] **= item						
		elif op in ['min']:
			obj[index] = min(obj[index],item)
		elif op in ['max']:
			obj[index] = max(obj[index],item)

		return obj

	def index(obj,index):
		'''
		Index object
		Args:
			obj (object): Object to apply operation
			index (object): Index to apply operation item
		Returns:
			item (object): Item at index
		'''
		return a[index]

if backend in ['jax','jax.autograd']:

	def jit(func,*,static_argnums=None,**kwargs):
		'''
		Just-in-time compile function
		Args:
			func (callable): Function to compile
			static_argnums (dict): Arguments to statically compile
			kwargs (dict): Additional (non-jittable) arguments to keep constant when jitting
		Returns:
			func (callable): Compiled function
		'''

		# TODO merge jit for different numpy backends (jax vs autograd)

		return wraps(func)(jax.jit(partial(func,**kwargs),static_argnums=static_argnums))
		# return wraps(func)(partial(func,**kwargs))

elif backend in ['autograd','numpy']:

	def jit(func,*,static_argnums=None,**kwargs):
		'''
		Just-in-time compile function
		Args:
			func (callable): Function to compile
			static_argnums (dict): Arguments to statically compile
			kwargs (dict): Additional (non-jittable) arguments to keep constant when jitting
		Returns:
			func (callable): Compiled function
		'''

		# TODO merge jit for different numpy backends (jax vs autograd)

		# return wraps(func)(jax.jit(partial(func,**kwargs),static_argnums=static_argnums))
		return wraps(func)(partial(func,**kwargs))		


if backend in ['jax','jax.autograd']:

	# @partial(jit,static_argnums=(2,))	
	def vmap(func,in_axes=0,out_axes=0,axis_name=None,**kwargs):	
		'''
		Vectorize function over input axis of iterables
		Args:
			func (callable): Function that acts on single elements of iterables
			in_axes (int,iterable): Input axis of iterables
			out_axes (int,interable): Output axis of func return
			axis_names (object): hashable Python object used to identify the mapped
				axis so that parallel collectives can be applied.
			kwargs (dict): Additional keyword arguments for func
		Returns:
			vfunc (callable): Vectorized function with signature vfunc(*iterables) = [func(*iterables[axes_in][0]),...,func(*iterables[axes_in][n-1])]
		'''

		# TODO merge vmap for different numpy backends (jax vs autograd)

		func = jit(func,**kwargs)

		vfunc = jax.vmap(func,in_axes=in_axes,out_axes=out_axes,axis_name=axis_name)

		return vfunc

		# in_axes = [in_axes] if in_axes is None or isinstance(in_axes,integers) else in_axes
		# out_axes = [out_axes] if out_axes is None or isinstance(out_axes,integers) else out_axes
		# axis_name = [axis_name] if axis_name is None or isinstance(axis_name,integers) else axis_name

		# def vfunc(*args,**kwargs):
		# 	args = itertools.product(*(arg if (i in in_axes) and ((len(in_axes)<len(args)) or (in_axes[i] is not None)) else [arg] for i,arg in enumerate(args)))
		# 	# TODO arbitrary in_axes, out_axes
		# 	return array([func(*arg,**kwargs) for arg in args])

		# return vfunc


elif backend in ['autograd','numpy']:

	# @partial(jit,static_argnums=(2,))	
	def vmap(func,in_axes=0,out_axes=0,axis_name=None,**kwargs):	
		'''
		Vectorize function over input axis of iterables
		Args:
			func (callable): Function that acts on single elements of iterables
			in_axes (int,iterable): Input axis of iterables
			out_axes (int,interable): Output axis of func return
			axis_names (object): hashable Python object used to identify the mapped
				axis so that parallel collectives can be applied.
			kwargs (dict): Additional keyword arguments for func
		Returns:
			vfunc (callable): Vectorized function with signature vfunc(*iterables) = [func(*iterables[axes_in][0]),...,func(*iterables[axes_in][n-1])]
		'''

		# TODO merge vmap for different numpy backends (jax vs autograd)

		# func = jit(func,**kwargs)

		# vfunc = jax.vmap(func,in_axes=in_axes,out_axes=out_axes,axis_name=axis_name)

		# return vfunc

		in_axes = [in_axes] if in_axes is None or isinstance(in_axes,integers) else in_axes
		out_axes = [out_axes] if out_axes is None or isinstance(out_axes,integers) else out_axes
		axis_name = [axis_name] if axis_name is None or isinstance(axis_name,integers) else axis_name

		def vfunc(*args,**kwargs):
			args = itertools.product(*(arg if (i in in_axes) and ((len(in_axes)<len(args)) or (in_axes[i] is not None)) else [arg] for i,arg in enumerate(args)))
			# TODO arbitrary in_axes, out_axes
			return array([func(*arg,**kwargs) for arg in args])

		return vfunc


if backend in ['jax','jax.autograd']:

	# @partial(jit,static_argnums=(2,))	
	def pmap(func,in_axes=0,out_axes=0,axis_name=None,**kwargs):	
		'''
		Vectorize function over input axis of iterables
		Args:
			func (callable): Function that acts on single elements of iterables
			in_axes (int,iterable): Input axis of iterables
			out_axes (int,interable): Output axis of func return
			axis_names (object): hashable Python object used to identify the mapped
				axis so that parallel collectives can be applied.
			kwargs (dict): Additional keyword arguments for func
		Returns:
			pfunc (callable): Vectorized function with signature vfunc(*iterables) = [func(*iterables[axes_in][0]),...,func(*iterables[axes_in][n-1])]
		'''

		# TODO merge pmap for different numpy backends (jax vs autograd)

		func = jit(func,**kwargs)

		pfunc = jax.pmap(func,in_axes=in_axes,out_axes=out_axes,axis_name=axis_name)

		return pfunc

elif backend in ['autograd','numpy']:

	# @partial(jit,static_argnums=(2,))	
	def pmap(func,in_axes=0,out_axes=0,axis_name=None,**kwargs):	
		'''
		Vectorize function over input axis of iterables
		Args:
			func (callable): Function that acts on single elements of iterables
			in_axes (int,iterable): Input axis of iterables
			out_axes (int,interable): Output axis of func return
			axis_names (object): hashable Python object used to identify the mapped
				axis so that parallel collectives can be applied.
			kwargs (dict): Additional keyword arguments for func
		Returns:
			pfunc (callable): Vectorized function with signature vfunc(*iterables) = [func(*iterables[axes_in][0]),...,func(*iterables[axes_in][n-1])]
		'''

		# TODO merge pmap for different numpy backends (jax vs autograd)

		func = jit(func,**kwargs)

		pfunc = jax.pmap(func,in_axes=in_axes,out_axes=out_axes,axis_name=axis_name)

		return pfunc


if backend in ['jax','jax.autograd']:

	# @partial(jit,static_argnums=(2,))
	def vfunc(funcs,in_axes=0,out_axes=0,axis_name=None,**kwargs):	
		'''
		Vectorize indexed functions over operands
		Args:
			funcs (iterable[callable]): Functions that act on that acts on single elements of iterables
			in_axes (int,iterable): Input axis of iterables
			out_axes (int,interable): Output axis of func return
			axis_names (object): hashable Python object used to identify the mapped
				axis so that parallel collectives can be applied.
			kwargs (dict): Additional keyword arguments for func	
		Returns:
			vfunc (callable): Vectorized function with signature vfunc(*iterables) = [func(*iterables[axes_in][0]),...,func(*iterables[axes_in][n-1])]
		'''

		funcs = [jit(func,**kwargs) for func in funcs]

		func = lambda index,*args: switch(index,funcs,*args)
		vfunc = vmap(func,in_axes=in_axes,out_axes=out_axes,axis_name=axis_name)

		# func = lambda index,*args,funcs=funcs: switch(index,funcs,*args[index])
		# vfunc = lambda *args,funcs: array([func(index,*args) for index in range(len(funcs))])

		return vfunc

elif backend in ['autograd','numpy']:

	# @partial(jit,static_argnums=(2,))
	def vfunc(funcs,in_axes=0,out_axes=0,axis_name=None,**kwargs):	
		'''
		Vectorize indexed functions over operands
		Args:
			funcs (iterable[callable]): Functions that act on that acts on single elements of iterables
			in_axes (int,iterable): Input axis of iterables
			out_axes (int,interable): Output axis of func return
			axis_names (object): hashable Python object used to identify the mapped
				axis so that parallel collectives can be applied.
			kwargs (dict): Additional keyword arguments for func	
		Returns:
			vfunc (callable): Vectorized function with signature vfunc(*iterables) = [func(*iterables[axes_in][0]),...,func(*iterables[axes_in][n-1])]
		'''

		funcs = [jit(func,**kwargs) for func in funcs]

		# func = lambda index,*args: switch(index,funcs,*args)
		# vfunc = vmap(func,in_axes=in_axes,out_axes=out_axes,axis_name=axis_name)

		func = lambda index,*args,funcs=funcs: switch(index,funcs,*args[index])
		vfunc = lambda *args,funcs: array([func(index,*args) for index in range(len(funcs))])

		return vfunc


if backend in ['jax','jax.autograd']:
	
	def switch(index,funcs,*args):
		'''
		Switch between indexed functions over operands
		Args:
			index (int): Index for function
			funcs (iterable[callable]): Functions that act on that acts on single elements of iterables
			args (tuple): Arguments for function
		Returns:
			out (object): Return of function
		'''	

		# TODO merge switch for different numpy backends (jax vs autograd)

		return jax.lax.switch(index,funcs,*args)
		# return funcs[index](*args)

elif backend in ['autograd','numpy']:

	def switch(index,funcs,*args):
		'''
		Switch between indexed functions over operands
		Args:
			index (int): Index for function
			funcs (iterable[callable]): Functions that act on that acts on single elements of iterables
			args (tuple): Arguments for function
		Returns:
			out (object): Return of function
		'''	

		# TODO merge switch for different numpy backends (jax vs autograd)

		# return jax.lax.switch(index,funcs,*args)
		return funcs[index](*args)


if backend in ['jax','jax.autograd']:
	
	# @partial(jit,static_argnums=(2,))	
	def forloop(start,end,func,out):	
		'''
		Perform loop of func from start to end indices
		Args:
			start (int): Start index of loop
			end (int): End index of loop
			func (callable): Function that acts on iterables with signature func(i,out)
			out (array): Initial value of loop
		Returns:
			out (array): Return of loop
		'''

		# TODO merge forloop for different numpy backends (jax vs autograd)

		return jax.lax.fori_loop(start,end,func,out)


elif backend in ['autograd','numpy']:

	# @partial(jit,static_argnums=(2,))	
	def forloop(start,end,func,out):	
		'''
		Perform loop of func from start to end indices
		Args:
			start (int): Start index of loop
			end (int): End index of loop
			func (callable): Function that acts on iterables with signature func(i,out)
			out (array): Initial value of loop
		Returns:
			out (array): Return of loop
		'''

		# TODO merge forloop for different numpy backends (jax vs autograd)

		for i in range(start,end):
			out = func(i,out)
		return out		


if backend in ['jax','jax.autograd']:
	
	# @partial(jit,static_argnums=(2,))	
	def cond(pred,true_fun,false_fun,*operands):	
		'''
		Conditionally evaluate functions
		Args:
			pred (bool): Conditional to choose function to evaluate
			true_fun (callable): Function to evaluate if pred is True, with signature true_func(*operands)
			false_fun (callable): Function to evaluate if pred is False, with signature false_fun(*operands)
			operands (iterable[object]): Arguments for functions
		Returns:
			out (array): Return of function
		'''
		return jax.lax.cond(pred,true_fun,false_fun,*operands)


elif backend in ['autograd','numpy']:

	# @partial(jit,static_argnums=(2,))	
	def cond(pred,true_fun,false_fun,*operands):	
		'''
		Conditionally evaluate functions
		Args:
			pred (bool): Conditional to choose function to evaluate
			true_fun (callable): Function to evaluate if pred is True, with signature true_func(*operands)
			false_fun (callable): Function to evaluate if pred is False, with signature false_fun(*operands)
			operands (iterable[object]): Arguments for functions
		Returns:
			out (array): Return of function
		'''
		if pred:
			return true_fun(*operands)
		else:
			return false_fun(*operands)

def value_and_gradient(func,grad=None,returns=False):
	'''
	Compute value and gradient of function
	Args:
		func (callable): Function to differentiate
		grad (callable): Gradient of function
		returns (bool): Return function and gradient
	Returns:
		value_and_grad (callable): Value and Gradient of function
		func (callable): Function
		grad (callable): Gradient
	'''	
	def _value_and_grad(func,grad):
		def _func_and_grad(*args,**kwargs):
			return func(*args,**kwargs),grad(*args,**kwargs)
		return _func_and_grad

	if grad is None:
		grad = gradient(func)
		# value_and_grad = jit(jax.value_and_grad(func))
		value_and_grad = _value_and_grad(func,grad)
	else:
		value_and_grad = _value_and_grad(func,grad)

	if returns:
		return value_and_grad,func,grad
	else:
		return value_and_grad

def gradient(func,mode=None,argnums=0,holomorphic=False,**kwargs):
	'''
	Compute gradient of function
	Args:
		func (callable): Function to differentiate
		mode (str): Type of gradient, allowed ['grad','finite','shift','fwd','rev'], defaults to 'grad'
		argnums (int,iterable[int]): Arguments of func to derive with respect to
		holomorphic (bool): Whether function is holomorphic
		kwargs : Additional keyword arguments for gradient mode:
			'finite': tol (float): Finite difference tolerance
			'shift': shifts (int): Number of eigenvalues of shifted values
			'fwd': move (bool): Move differentiated axis to beginning of dimensions
			'rev': move (bool): Move differentiated axis to beginning of dimensions
	Returns:
		grad (callable): Gradient of function
	'''

	if mode in ['finite']:
		grad = gradient_finite(func,argnums=argnums,holomorphic=holomorphic,**kwargs)
	elif mode in ['shift']:
		grad = gradient_shift(func,argnums=argnums,holomorphic=holomorphic,**kwargs)
	elif mode in ['fwd']:
		grad = gradient_fwd(func,argnums=argnums,holomorphic=holomorphic,**kwargs)
	elif mode in ['rev']:
		grad = gradient_rev(func,argnums=argnums,holomorphic=holomorphic,**kwargs)
	elif mode in ['grad']:
		grad = gradient_grad(func,argnums=argnums,holomorphic=holomorphic,**kwargs)
	else:
		grad = gradient_grad(func,argnums=argnums,holomorphic=holomorphic,**kwargs)

	return grad


def gradient_finite(func,tol=1e-6,argnums=0,holomorphic=False,**kwargs):
	'''
	Calculate finite difference second order derivative of function
	Args:
		func (callable): Function to derive, with signature func(*args,**kwargs) and output shape
		tol (float): Finite difference tolerance
		argnums (int,iterable[int]): Arguments of func to derive with respect to
		holomorphic (bool): Whether function is holomorphic		
		kwargs : Additional keyword arguments	
	Returns:
		grad (callable): Gradient of function
	'''
	@jit
	def grad(*args,**kwargs):
		if args:
			x,args = args[argnums],(args[:argnums],args[argnums+1:])
		else:
			x,args = kwargs.pop(list(kwargs)[argnums]),((),args) 
		size,shape = x.size,x.shape
		vectors = eye(size).reshape((size,*shape))

		out = vmap(lambda v,tol=tol: (func(*args[0],x+tol*v,*args[1],**kwargs)-func(*args[0],x-tol*v,*args[1],**kwargs))/(2*tol))(vectors)
		out = out.reshape((*shape,*out.shape[1:]))
		return out

	return grad


def gradient_shift(func,shifts=2,argnums=0,holomorphic=False,**kwargs):
	'''
	Calculate shift-rules derivative of function
	Args:
		func (callable): Function to derive, with signature func(*args,**kwargs) and output shape
		shifts (int): Number of eigenvalues of shifted values
		argnums (int,iterable[int]): Arguments of func to derive with respect to
		holomorphic (bool): Whether function is holomorphic				
		kwargs : Additional keyword arguments
	Returns:
		grad (callable): Gradient of function
	'''

	shifts = -(shifts-1)/2 + arange(shifts)

	@jit
	def grad(*args,**kwargs):
		if args:
			x,args = args[argnums],(args[:argnums],args[argnums+1:])
		else:
			x,args = kwargs.pop(list(kwargs)[argnums]),((),args) 
		size,shape = x.size,x.shape
		vectors = eye(size).reshape((size,*shape))
		out = vmap(vmap(lambda v,s: s*func(*args[0],x+pi/4/s*v,*args[1],**kwargs),in_axes=(0,None)),in_axes=(None,0))(vectors,shifts).sum(0)
		out = out.reshape((*shape,*out.shape[1:]))
		return out

	return grad


if backend in ['jax','jax.autograd']:

	def gradient_grad(func,move=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute gradient of function
		Args:
			func (callable): Function to differentiate
			move (bool): Move differentiated axis to beginning of dimensions
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic	
			kwargs : Additional keyword arguments
		Returns:
			grad (callable): Gradient of function
		'''

		# TODO merge grad for different numpy backends (jax vs autograd)

		_grad = jit(jax.grad(func,argnums=argnums,holomorphic=holomorphic))
		# argnum = argnums
		# if holomorphic:
		# 	_grad = jit(autograd.grad(func,argnum=argnum))
		# else:
		# 	_grad = jit(autograd.grad(func,argnum=argnum))

		if move:
			grad = _grad
		else:
			grad = _grad

		return grad

elif backend in ['autograd']:

	def gradient_grad(func,move=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute gradient of function
		Args:
			func (callable): Function to differentiate
			move (bool): Move differentiated axis to beginning of dimensions
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic	
			kwargs : Additional keyword arguments
		Returns:
			grad (callable): Gradient of function
		'''

		# TODO merge grad for different numpy backends (jax vs autograd)

		# _grad = jit(jax.grad(func,argnums=argnums,holomorphic=holomorphic))
		argnum = argnums
		if holomorphic:
			_grad = jit(autograd.grad(func,argnum=argnum))
		else:
			_grad = jit(autograd.grad(func,argnum=argnum))

		if move:
			grad = _grad
		else:
			grad = _grad

		return grad

elif backend in ['numpy']:

	def gradient_grad(func,move=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute gradient of function
		Args:
			func (callable): Function to differentiate
			move (bool): Move differentiated axis to beginning of dimensions
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic	
			kwargs : Additional keyword arguments
		Returns:
			grad (callable): Gradient of function
		'''

		# TODO merge grad for different numpy backends (jax vs autograd)

		raise NotImplementedError
		return

if backend in ['jax','jax.autograd']:

	def gradient_fwd(func,move=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute forward gradient of function
		Args:
			func (callable): Function to differentiate
			move (bool): Move differentiated axis to beginning of dimensions
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic	
			kwargs : Additional keyword arguments
		Returns:
			grad (callable): Gradient of function
		'''

		# TODO merge grad for different numpy backends (jax vs autograd)

		_grad = jit(jax.jacfwd(func,argnums=argnums,holomorphic=holomorphic))
		# argnum = argnums
		# if holomorphic:
		# 	_grad = jit(autograd.jacobian(func,argnum=argnum))
		# else:
		# 	_grad = jit(autograd.jacobian(func,argnum=argnum))

		if move:
			@jit
			def grad(*args,**kwargs):
				if args:
					x,args = args[argnums],(args[:argnums],args[argnums+1:])
				else:
					x,args = kwargs.pop(list(kwargs)[argnums]),((),args) 
				ndim = x.ndim
				return moveaxis(_grad(*args[0],x,*args[1],**kwargs),range(-1,-ndim-1,-1),range(ndim-1,-1,-1))
		else:
			grad = _grad

		return grad


elif backend in ['autograd']:

	def gradient_fwd(func,move=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute forward gradient of function
		Args:
			func (callable): Function to differentiate
			move (bool): Move differentiated axis to beginning of dimensions
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic	
			kwargs : Additional keyword arguments
		Returns:
			grad (callable): Gradient of function
		'''
		# TODO merge grad for different numpy backends (jax vs autograd)

		# _grad = jit(jax.jacfwd(func,argnums=argnums,holomorphic=holomorphic))
		argnum = argnums

		if not isinstance(argnum,integers):
			
			move = False
			def _grad(*args,**kwargs):
				grads = [[None for axis in argnum] for axis in argnum]
				for i,axis in enumerate(argnum):
					_grad = jit(autograd.jacobian(func,argnum=axis))
					grads[i][i] = _grad(*args,**kwargs)
				return grads
		else:
			if holomorphic:
				_grad = jit(autograd.jacobian(func,argnum=argnum))
			else:
				_grad = jit(autograd.jacobian(func,argnum=argnum))

		if move:
			@jit
			def grad(*args,**kwargs):
				if args:
					x,args = args[argnums],(args[:argnums],args[argnums+1:])
				else:
					x,args = kwargs.pop(list(kwargs)[argnums]),((),args) 
				ndim = x.ndim
				return moveaxis(_grad(*args[0],x,*args[1],**kwargs),range(-1,-ndim-1,-1),range(ndim-1,-1,-1))
		else:
			grad = _grad

		return grad

elif backend in ['numpy']:

	def gradient_fwd(func,move=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute forward gradient of function
		Args:
			func (callable): Function to differentiate
			move (bool): Move differentiated axis to beginning of dimensions
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic	
			kwargs : Additional keyword arguments
		Returns:
			grad (callable): Gradient of function
		'''
		# TODO merge grad for different numpy backends (jax vs autograd)

		# _grad = jit(jax.jacfwd(func,argnums=argnums,holomorphic=holomorphic))

		raise NotImplementedError
		return

if backend in ['jax','jax.autograd']:

	def gradient_rev(func,move=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute reverse gradient of function
		Args:
			func (callable): Function to differentiate
			move (bool): Move differentiated axis to beginning of dimensions
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic		
			kwargs : Additional keyword arguments		
		Returns:
			grad (callable): Gradient of function
		'''

		# TODO merge grad for different numpy backends (jax vs autograd)

		_grad = jit(jax.jacrev(func,argnums=argnums,holomorphic=holomorphic))
		# argnum = argnums
		# if holomorphic:
		# 	_grad = jit(autograd.grad(func,argnum=argnum))
		# else:
		# 	_grad = jit(autograd.grad(func,argnum=argnum))	

		if move:
			@jit
			def grad(*args,**kwargs):
				if args:
					x,args = args[argnums],(args[:argnums],args[argnums+1:])
				else:
					x,args = kwargs.pop(list(kwargs)[argnums]),((),args) 
				ndim = x.ndim
				return moveaxis(_grad(*args[0],x,*args[1],**kwargs),range(-1,-ndim-1,-1),range(ndim-1,-1,-1))		
		else:
			grad = _grad

		return grad


elif backend in ['autograd']:

	def gradient_rev(func,move=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute reverse gradient of function
		Args:
			func (callable): Function to differentiate
			move (bool): Move differentiated axis to beginning of dimensions
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic		
			kwargs : Additional keyword arguments		
		Returns:
			grad (callable): Gradient of function
		'''

		# TODO merge grad for different numpy backends (jax vs autograd)

		# _grad = jit(jax.jacrev(func,argnums=argnums,holomorphic=holomorphic))
		argnum = argnums
		if holomorphic:
			_grad = jit(autograd.grad(func,argnum=argnum))
		else:
			_grad = jit(autograd.grad(func,argnum=argnum))	

		if move:
			@jit
			def grad(*args,**kwargs):
				x,args = args[0],args[1:]
				ndim = x.ndim
				return moveaxis(_grad(x,*args,**kwargs),range(-1,-ndim-1,-1),range(ndim-1,-1,-1))		
		else:
			grad = _grad

		return grad


elif backend in ['numpy']:

	def gradient_rev(func,move=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute reverse gradient of function
		Args:
			func (callable): Function to differentiate
			move (bool): Move differentiated axis to beginning of dimensions
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic		
			kwargs : Additional keyword arguments		
		Returns:
			grad (callable): Gradient of function
		'''

		raise NotImplementedError
		return

if backend in ['jax','jax.autograd']:

	def hessian(func,mode=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute hessian of function
		Args:
			func (callable): Function to differentiate
			mode (str): Type of gradient, allowed ['grad','finite','shift','fwd','rev'], defaults to 'grad'
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic
			kwargs : Additional keyword arguments for gradient mode:
				'finite': tol (float): Finite difference tolerance
				'shift': shifts (int): Number of eigenvalues of shifted values
				'fwd': move (bool): Move differentiated axis to beginning of dimensions
				'rev': move (bool): Move differentiated axis to beginning of dimensions
		Returns:
			grad (callable): Hessian of function
		'''
		
		# TODO merge grad for different numpy backends (jax vs autograd)
		
		grad = jit(jax.hessian(func,argnums=argnums,holomorphic=holomorphic))
		# argnum = argnums
		# if holomorphic:
		# 	grad = jit(autograd.hessian(func,argnum=argnum))
		# else:
		# 	grad = jit(autograd.hessian(func,argnum=argnum))	

		return grad


elif backend in ['autograd']:

	def hessian(func,mode=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute hessian of function
		Args:
			func (callable): Function to differentiate
			mode (str): Type of gradient, allowed ['grad','finite','shift','fwd','rev'], defaults to 'grad'
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic
			kwargs : Additional keyword arguments for gradient mode:
				'finite': tol (float): Finite difference tolerance
				'shift': shifts (int): Number of eigenvalues of shifted values
				'fwd': move (bool): Move differentiated axis to beginning of dimensions
				'rev': move (bool): Move differentiated axis to beginning of dimensions
		Returns:
			grad (callable): Hessian of function
		'''
		
		# TODO merge grad for different numpy backends (jax vs autograd)
		
		# grad = jit(jax.hessian(func,argnums=argnums,holomorphic=holomorphic))
		argnum = argnums
		if holomorphic:
			grad = jit(autograd.hessian(func,argnum=argnum))
		else:
			grad = jit(autograd.hessian(func,argnum=argnum))	

		return grad		

elif backend in ['numpy']:

	def hessian(func,mode=None,argnums=0,holomorphic=False,**kwargs):
		'''
		Compute hessian of function
		Args:
			func (callable): Function to differentiate
			mode (str): Type of gradient, allowed ['grad','finite','shift','fwd','rev'], defaults to 'grad'
			argnums (int,iterable[int]): Arguments of func to derive with respect to
			holomorphic (bool): Whether function is holomorphic
			kwargs : Additional keyword arguments for gradient mode:
				'finite': tol (float): Finite difference tolerance
				'shift': shifts (int): Number of eigenvalues of shifted values
				'fwd': move (bool): Move differentiated axis to beginning of dimensions
				'rev': move (bool): Move differentiated axis to beginning of dimensions
		Returns:
			grad (callable): Hessian of function
		'''

		raise NotImplementedError
		return

def fisher(func,grad=None,shapes=None,optimize=None,mode=None,hermitian=None,unitary=None,**kwargs):
	'''
	Compute fisher information of function
	Args:
		func (callable,model instance): Function to compute
		grad (callable): Gradient to compute
		shapes (iterable[tuple[int]]): Shapes of func and grad arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type
		mode (str): Type of gradient, allowed ['grad','finite','shift','fwd','rev'], defaults to 'fwd'
		hermitian (bool): function is hermitian
		unitary (bool): function is unitary
	Returns:
		fisher (callable): Fisher information of function
	'''
	if mode is None:
		mode = 'fwd'

	if shapes is None:
		ndim = None
	else:
		ndim = min((len(shape) for shape in shapes),default=2)

	if grad is None:
		grad = gradient(func,mode=mode,move=True)

	size = min((prod(shape[:len(shape)-ndim] for shape in shapes if len(shape)>ndim)),default=0)
	dtype = getattr(func,'dtype',None)

	if not size:
		def fisher(*args,**kwargs):
			return None
		return fisher

	if hermitian:

		func = spectrum(func,compute_v=True,hermitian=hermitian)

		if ndim == 1:
			raise NotImplementedError("Hermitian Fisher Information Not Implemented for ndim = %r"%(ndim))			
		elif ndim == 2:
			shapes = [[shapes[0],shapes[1],shapes[0]],[shapes[1],shapes[1],shapes[0]]]
			subscripts = ['ni,unm,mj->uij','uij,vij,ij->uv']
			wrappers = [lambda out,*operands: out, lambda out,*operands: 2*real(out)]
		else:
			raise NotImplementedError("Hermitian Fisher Information Not Implemented for ndim = %r"%(ndim))
	

		if shapes is not None:
			einsummations = [
				lambda *operands,einsummation=einsum(subscript,*shape,optimize=optimize,wrapper=wrapper): einsummation(*operands)
					for subscript,shape,wrapper in zip(subscripts,shapes,wrappers)
				]
		else:
			einsummations = [
				lambda *operands,subscript=subscript,shape=shape,optimize=optimize,wrapper=wrapper: einsum(subscripts,*operands,optimize=optimize,wrapper=wrapper)
					for subscript,shape,wrapper in zip(subscripts,shapes,wrappers)
				]

		# @jit
		def fisher(*args,**kwargs):
			
			function = func(*args,**kwargs)
			gradient = grad(*args,**kwargs)

			eigenvalues,eigenvectors = function

			eps = None
			n = eigenvalues.size
			d = nonzero(eigenvalues,eps=eps)
			indices,zeros = slice(n-d,n),slice(0,n-d)
			indexes = ((indices,indices),(indices,zeros),(zeros,indices)) if d<n else ((indices,indices),)
			out = 0

			for i,j in indexes:
				tmp = einsummations[0](conjugate(eigenvectors[:,i]),gradient,eigenvectors[:,j])
				out += einsummations[1](tmp,conjugate(tmp),1/(eigenvalues[i,None] + eigenvalues[None,j]))

			out = real(out)

			return out			


	elif unitary:
		if ndim == 1:
			shapes = [[shapes[1],shapes[0]],[[shapes[1][0]],[shapes[1][0]]],[shapes[1],shapes[1]]]
			subscripts = ['ui,i->u','u,v->uv','ui,vi->uv']
			wrappers = [lambda out,*operands: out/sqrt(operands[0].shape[-1]**0),lambda out,*operands: -2*out,lambda out,*operands: 2*out/sqrt(operands[0].shape[-1]**0)]
		elif ndim == 2:
			shapes = [[shapes[1],shapes[0]],[[shapes[1][0]],[shapes[1][0]]],[shapes[1],shapes[1]]]
			subscripts = ['uij,ij->u','u,v->uv','uij,vij->uv']
			wrappers = [lambda out,*operands: out/sqrt(operands[0].shape[-1]**2),lambda out,*operands: -2*out,lambda out,*operands: 2*out/sqrt(operands[0].shape[-1]**2)]
		else:
			shapes = None
			subscripts = ['uij,ij->u','u,v->uv','uij,vij->uv']
			wrappers = [lambda out,*operands: out/sqrt(operands[0].shape[-1]**2),lambda out,*operands: -2*out,lambda out,*operands: 2*out/sqrt(operands[0].shape[-1]**2)]			
		
		if shapes is not None:
			einsummations = [
				lambda *operands,einsummation=einsum(subscript,*shape,optimize=optimize,wrapper=wrapper): einsummation(*operands)
					for subscript,shape,wrapper in zip(subscripts,shapes,wrappers)
				]
		else:
			einsummations = [
				lambda *operands,subscript=subscript,shape=shape,optimize=optimize,wrapper=wrapper: einsum(subscripts,*operands,optimize=optimize,wrapper=wrapper)
					for subscript,shape,wrapper in zip(subscripts,shapes,wrappers)
				]	

		@jit
		def fisher(*args,**kwargs):
			function = func(*args,**kwargs)
			gradient = grad(*args,**kwargs)

			out = 0

			tmp = einsummations[0](conjugate(gradient),function)
			out += einsummations[1](conjugate(tmp),tmp)

			tmp = einsummations[2](conjugate(gradient),gradient)
			out += tmp
			
			out = real(out)

			return out
	else:

		raise NotImplementedError("Not Hermitian/Unitary Fisher Information Not Implemented for ndim = %r"%(ndim))

	return fisher


def entropy(func,shape=None,hermitian=None,unitary=None,**kwargs):
	'''
	Compute entropy of function
	Args:
		func  (callable): Function to compute entropy
		shape (iterable[int]): Shape of function
		hermitian (bool): function is hermitian
		unitary (bool): function is unitary
	Returns:
		entropy (callable): Entropy of function
	'''

	if shape is None:
		shape = getattr(func,'shape',None)

	if hermitian is None:
		hermitian = getattr(func,'hermitian',None)

	if unitary is None:
		unitary = getattr(func,'unitary',None)

	ndim = len(shape) if shape is not None else None
	d = max(shape) if shape is not None else None

	if ndim is not None and ndim < 2:
		def entropy(*args,**kwargs):
			return 0
	else:
		def entropy(*args,**kwargs):
			out = func(*args,**kwargs)
			
			out = eig(out,compute_v=False,hermitian=hermitian)

			out = abs(out)

			out = -addition(out*log(out))

			return out

	return entropy


def purity(func,shape=None,hermitian=None,unitary=None,**kwargs):
	'''
	Compute purity of function
	Args:
		func  (callable): Function to compute purity
		shape (iterable[int]): Shape of function
		hermitian (bool): function is hermitian
		unitary (bool): function is unitary
	Returns:
		purity (callable): purity of function
	'''

	if shape is None:
		shape = getattr(func,'shape',None)

	if hermitian is None:
		hermitian = getattr(func,'hermitian',None)

	if unitary is None:
		unitary = getattr(func,'unitary',None)

	ndim = len(shape) if shape is not None else None
	d = max(shape) if shape is not None else None

	if ndim is not None and ndim < 2:
		def purity(*args,**kwargs):
			return 1
	else:
		def purity(*args,**kwargs):
			out = func(*args,**kwargs)
			
			out = real(einsum('ij,ij->',out,conjugate(out)))

			return out

	return purity


def similarity(func,label,shape=None,hermitian=None,unitary=None,**kwargs):
	'''
	Compute similarity of function
	Args:
		func  (callable): Function to compute similarity
		label  (callable,array): Label to compute similarity
		shape (iterable[int]): Shape of function
		hermitian (bool): function is hermitian
		unitary (bool): function is unitary
	Returns:
		similarity (callable): similarity of function
	'''

	if shape is None:
		shape = getattr(func,'shape',None)

	if hermitian is None:
		hermitian = getattr(func,'hermitian',None)

	if unitary is None:
		unitary = getattr(func,'unitary',None)

	if callable(label):
		label = label()

	labels = einsum('ij,ji->',label,label)

	ndim = len(shape) if shape is not None else None
	d = max(shape) if shape is not None else None

	if ndim is not None and ndim < 2:
		def similarity(*args,**kwargs):
			return 0
	else:
		def similarity(*args,**kwargs):
			out = func(*args,**kwargs)

			outs,out = einsum('ij,ji->',out,out),einsum('ij,ji->',out,label)
			
			out = abs((d*(out)-1)/sqrt((d*(outs)-1)*(d*(labels)-1)))

			return out

	return similarity

def divergence(func,label,shape=None,hermitian=None,unitary=None,**kwargs):
	'''
	Compute divergence of function
	Args:
		func  (callable): Function to compute divergence
		label  (callable,array): Label to compute divergence
		shape (iterable[int]): Shape of function
		hermitian (bool): function is hermitian
		unitary (bool): function is unitary
	Returns:
		divergence (callable): divergence of function
	'''

	if shape is None:
		shape = getattr(func,'shape',None)

	if hermitian is None:
		hermitian = getattr(func,'hermitian',None)

	if unitary is None:
		unitary = getattr(func,'unitary',None)

	if callable(label):
		label = label()
	labels = eig(label,compute_v=False,hermitian=hermitian)
	labels = abs(labels)
	labels = addition(log(labels**labels))

	ndim = len(shape) if shape is not None else None
	d = max(shape) if shape is not None else None

	if ndim is not None and ndim < 2:
		def divergence(*args,**kwargs):
			return 0
	else:
		def divergence(*args,**kwargs):
			out = func(*args,**kwargs)

			outs,out = eig(out,compute_v=True,hermitian=hermitian)

			outs = abs(outs)

			out = real(labels - einsum('ij,jk,k,ik->',label,out,log(outs),conjugate(out)))

			return out

	return divergence


def nullfunc(obj,*args,**kwargs):
	'''
	Null function
	Args:
		obj (object): Object to return
		args (iterable): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		obj (object): Object to return
	'''
	return obj

def zerofunc(obj,*args,**kwargs):
	'''
	Zero function
	Args:
		obj (object): Object to return
		args (iterable): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		obj (object): Object to return
	'''
	return 0

def astype(a,dtype):
	'''
	Convert array to dtype
	Args:
		a (object): array
		dtype (str,datatype): Datatype
	Returns:
		out (object): array of type dtype
	'''
	return a.astype(dtype)

def datatype(dtype):
	'''
	Get underlying data type of dtype
	Args:
		dtype (str,datatype): Datatype
	Returns:
		dtype (datatype): Underlying datatype
	'''
	
	return real(array([],dtype=dtype)).dtype

class Array(onp.ndarray):
	'''
	Numpy array subclass, subclass of nd.ndarray with array of data of shape
	
	Classes that inherit Array must have a __setup__() in place of __init__() to properly initialize
	subclassed ndarray, that is initalized through Array's __new__() and an array.view(cls)
	
	Classes that inherit Array's __setup__() create the following attributes:
		self.data (array)
		self.string (str)

	Classes that inherit Array's __setup__() must also append newly institated attributes to the 
	self.attrs attribute dictionary to allow for proper re-instantiation of the inherited class 
	upon numpy-like views, slices etc.

	Args:
		data (array,list,tuple): array data if array or list, array shape if tuple
		args (iterable): class attributes
		kwargs (dict): class keyword attributes
	'''
	def __new__(cls,*args,**kwargs):
		attrs = {}
		clsattrs = {}
		for attr,value in kwargs.items():
			if attr in onp.ndarray.__dict__:
				clsattrs[attr] = value
			else:
				attrs[attr] = value

		field,value = 'attrs',attrs
		setattr(cls,field,value)

		cls = cls.__new__(cls,*args,**kwargs)

		obj = onp.asarray(cls.data,**clsattrs).view(cls)

		field,value = 'attrs',attrs
		setattr(obj,field,value)

		for attr,value in attrs.items():
			setattr(obj,attr,getattr(obj,attr,value))

		return obj

	def __array_finalize__(self, obj):
		field = 'attrs'
		attrs = getattr(obj,field,getattr(self,field,None))
		if obj is None or attrs is None: 
			return
		for attr in attrs:
			setattr(self,attr,getattr(obj,attr,getattr(self,attr,None)))
		return

	
	def __setup__(self,*args,**kwargs):
		
		field = 'attrs'
		getattr(self,field).update({attr:getattr(self,attr,None) for attr in kwargs if attr not in getattr(self,field)})
		
		data = args[0] if len(args)>0 else None
		if data is None:
			data = onp.array()
		elif isinstance(data,tuple):
			data = onp.zeros(data)
		else:
			data = onp.asarray(data)
		self.data = data
		self.string = data.__str__()
		return

	def __repr__(self):
		return self.string

	def to_numpy(self):
		return onp.array(self)

	def to_jax(self):
		return np.array(self)

class String(str):
	'''
	String class to represent concatenated and delimited strings, subclass of str
	Args:
		string (str,tuple,list): string, tuple of strings, or list of tuple of strings or strings, where strings in tuple are concatenated and strings in list are delimited		
	'''
	def __new__(cls,string,deliminate=' + ',concatenate='*',fmt=None):
		try:
			string = [tuple([substring]) if not isinstance(substring,tuple) else substring 
						for substring in ([string] if not isinstance(string,list) else string)]
			string = deliminate.join([concatenate.join(substring) for substring in string])
		except:
			string = ''
		string = fmt%string if fmt is not None else string
		obj = super().__new__(cls,string)
		obj.deliminate = deliminate
		obj.concatenate = concatenate			
		return obj

	def __getslice__(self,item):
		try:
			obj = self.deliminate.join(self.split(self.deliminate)[item])
		except:
			obj = self
		return obj

	def __getitem__(self,item):
		try:
			obj = self.split(self.deliminate)[item]
		except:
			obj = self
		return obj



# @tree_register
# class Parameters(dict):
# 	'''
# 	Class for pytree subclassed dict dictionary of parameters, with children and auxiliary keys
# 	Args:
# 		parameters (dict): Dictionary of parameters
# 		children (iterable): Iterable of tree leaf children keys
# 		auxiliary (iterable): Iterable of tree leaf auxiliary keys
# 	'''
# 	def __init__(self,parameters,children=None,auxiliary=None):
# 		super().__init__(parameters)

# 		if children is None:
# 			children = [parameter for parameter in parameters]
# 		if auxiliary is None:
# 			auxiliary = []
# 		else:
# 			auxiliary = [parameter for parameter in parameters if parameters not in children]

# 		self.children = children
# 		self.auxiliary = auxiliary
# 		return

# 	def tree_flatten(self):
# 		keys = (self.children,self.auxiliary,)
# 		children = (*(self[parameter] for parameter in self.children),)
# 		auxiliary = (*keys,*(self[parameter] for parameter in self.auxiliary),)
# 		return (children,auxiliary)

# 	@classmethod
# 	def tree_unflatten(cls,auxiliary,children):
# 		keys,auxiliary = auxiliary[:2],auxiliary[2:]
# 		parameters = {
# 			**dict(zip(keys[0],children)),
# 			**dict(zip(keys[1],auxiliary))
# 			}
# 		return cls(parameters)

if backend in ['jax']:

	def tree_ravel(tree,is_leaf=None):
		'''
		Flatten tree
		Args:
			tree (pytree): Tree to flatten
			is_leaf (type,iterable[type],callable): Boolean whether tree nodes are leaves			
		Yields:
			node (object): Nodes of tree
		'''
		yield from jax.flatten_util.ravel_pytree(tree)

	def tree_flatten(tree,is_leaf=None):
		'''
		Flatten tree
		Args:
			tree (pytree): Tree to flatten
			is_leaf (type,iterable[type],callable): Boolean whether tree nodes are leaves			
		Returns:
			flat (array): Flattened tree
		'''
		return array([tree for tree in tree_ravel(tree,is_leaf=is_leaf)])

	def tree_func(func):
		'''
		Perform function on trees
		Args:
			func (callable): Callable function with signature func(*trees,**kwargs)
		Returns:
			tree_func (callable): Function that returns tree_map pytree of function call with signature tree_func(*trees,**kwargs)
		'''
		@jit
		def tree_func(*trees,is_leaf=None,**kwargs):
			return tree_map(partial(func,**kwargs),*trees,is_leaf=is_leaf)
		return tree_func

	@tree_func
	def tree_dot(a,b):
		'''
		Perform dot product function on trees a and b
		Args:
			a (pytree): Pytree object to perform function
			b (pytree): Pytree object to perform function
		Returns:
			tree_map (pytree): Return pytree of function call
		'''	
		return dot(a.ravel(),b.ravel())

	@tree_func
	def tree_add(a,b):
		'''
		Perform add function on trees a and b
		Args:
			a (pytree): Pytree object to perform function
			b (pytree): Pytree object to perform function
		Returns:
			tree_map (pytree): Return pytree of function call
		'''
		return add(a,b)

	@tree_func
	def tree_index(a,index=None):
		'''
		Perform index function on tree a
		Args:
			a (pytree): Pytree object to perform function
			index (object): Index for pytree
		Returns:
			tree_map (pytree): Return pytree of function call
		'''
		return a[index]


elif backend in ['jax.autograd','autograd','numpy']:
	
	def tree_ravel(tree,is_leaf=None):
		'''
		Flatten tree
		Args:
			tree (pytree): Tree to flatten
			is_leaf (type,iterable[type],callable): Boolean whether tree nodes are leaves
		Yields:
			node (object): Nodes of tree
		'''
		
		if not callable(is_leaf):
			types = (dict,tuple,list,) if is_leaf is None else (*is_leaf,) if isinstance(is_leaf,iterables) else (is_leaf,)
			is_leaf = lambda tree,types=types: isinstance(tree,types)			

		if is_leaf(tree):
			for key in tree:
				node = tree[key] if isinstance(tree,dict) else key
				yield from tree_ravel(node,is_leaf=is_leaf)
		else:
			try:
				yield from tree.ravel()
			except:
				yield tree

	def tree_flatten(tree,is_leaf=None):
		'''
		Flatten tree
		Args:
			tree (pytree): Tree to flatten
			is_leaf (type,iterable[type],callable): Boolean whether tree nodes are leaves
		Returns:
			flat (array): Flattened tree
		'''
		return array([tree for tree in tree_ravel(tree,is_leaf=is_leaf)])

	def tree_func(func):
		'''
		Perform function on trees
		Args:
			func (callable): Callable function with signature func(*trees,**kwargs)
		Returns:
			tree_func (callable): Function that returns tree_map pytree of function call with signature tree_func(*trees,**kwargs)
		'''
		@jit
		def tree_func(*trees,is_leaf=None,**kwargs):
			return tree_map(partial(func,**kwargs),*trees,is_leaf=is_leaf)
		return tree_func

	@tree_func
	def tree_dot(a,b):
		'''
		Perform dot product function on trees a and b
		Args:
			a (pytree): Pytree object to perform function
			b (pytree): Pytree object to perform function
		Returns:
			tree_map (pytree): Return pytree of function call
		'''	
		return dot(a.ravel(),b.ravel())

	@tree_func
	def tree_add(a,b):
		'''
		Perform add function on trees a and b
		Args:
			a (pytree): Pytree object to perform function
			b (pytree): Pytree object to perform function
		Returns:
			tree_map (pytree): Return pytree of function call
		'''
		return add(a,b)

	@tree_func
	def tree_index(a,index=None):
		'''
		Perform index function on tree a
		Args:
			a (pytree): Pytree object to perform function
			index (object): Index for pytree
		Returns:
			tree_map (pytree): Return pytree of function call
		'''
		return a[index]	

def decorator(*arguments,function=None,**keywords):
	'''
	Wrap function with args and kwargs
	Args:
		arguments (iterable): Decorator arguments
		keywords (dict): Decorator keyword arguments

	'''
	def wrapper(func):
		@wraps
		def wrapped(*args,**kwargs):
			args = (*args,*arguments)
			kwargs = {**keywords,**kwargs}
			return func(*args,**kwargs)
		return wrapped
	return wrapper	


def wrapper(function,*arguments,**keywords):
	'''
	Wrap func args and kwargs with function
	Args:
		function (callable): Function to wrap arguments with signature function(args,kwargs,*arguments,**keywords) -> args,kwargs
		arguments (iterable): Function arguments
		keywords (dict): Function keyword arguments

	'''
	if function is None:
		def function(args,kwargs,*arguments,**keywords):
			return args,kwargs
	function = partial(function,*arguments,**keywords)
	def wrapper(func):
		@wraps
		def wrapped(*args,**kwargs):
			args,kwargs = function(args,kwargs)
			return func(*args,**kwargs)
		return wrapped
	return wrapper	


class Array(onp.ndarray):
	'''
	array subclass (TODO: Views do not copy attrs)
	'''
	def __new__(cls,obj,attrs={},**kwargs):
		self = onp.asarray(obj,**kwargs).view(cls)
		self.attrs = attrs
		for attr in self.attrs:
			setattr(self,attr,attrs[attr])
		return self

	def __array_finalize__(self, obj):
		if obj is None:
			return
		try:
			self.attrs = obj.attrs
			for attr in self.attrs:
				setattr(self,attr,getattr(obj,attr,None))
		except:
			pass
		return

class array(np.ndarray):
	'''
	array class
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return np.array(*args,**kwargs)
		# return super().__init__(cls,*args,**kwargs)

class nparray(onp.ndarray):
	'''
	array class
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return onp.array(*args,**kwargs)
		# return super().__init__(cls,*args,**kwargs)

class asndarray(onp.ndarray):
	'''
	array class
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return onp.asarray(*args,**kwargs)

class asarray(np.ndarray):
	'''
	array class
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return np.asarray(*args,**kwargs)

class asscalar(np.ndarray):
	'''
	array class
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,a,*args,**kwargs):
		try:
			return a.item()
		except (AttributeError,ValueError,TypeError):
			try:
				return onp.asscalar(a,*args,**kwargs)
			except:
				return a


class objs(onp.ndarray):
	'''
	array class of objects
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return onp.array(*args,**kwargs)


class asobjs(onp.ndarray):
	'''
	array class of objects
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return asndarray(*args,**kwargs)

class ones(array):
	'''
	array class of ones
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return np.ones(*args,**kwargs)


class zeros(array):
	'''
	array class of zeros
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return np.zeros(*args,**kwargs)

class empty(array):
	'''
	array class of empty
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return np.empty(*args,**kwargs)

class full(array):
	'''
	array class of full
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return np.full(*args,**kwargs)

class eye(array):
	'''
	array class of eye
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return np.eye(*args,**kwargs)

class arange(array):
	'''
	array class of arange
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return np.arange(*args,**kwargs)

class linspace(array):
	'''
	array class of linspace
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return np.linspace(*args,**kwargs)

class logspace(array):
	'''
	array class of logspace
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return np.logspace(*args,**kwargs)


class dataframe(pd.DataFrame):
	'''
	dataframe class
	Args:
		args (iterable): Dataframe arguments
		kwargs (dict): Dataframe keyword arguments
	Returns:
		out (array): dataframe
	'''
	def __new__(cls,*args,**kwargs):
		return pd.DataFrame(*args,**kwargs)
		# return super().__init__(cls,*args,**kwargs)


class identity(array):
	'''
	array class of identity
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,n,*args,**kwargs):
		return np.eye(*((n,) if isinstance(n,integers) else n),*args,**kwargs)


class hadamard(array):
	'''
	array class of hadamard
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,n,*args,**kwargs):
		if n == 1:
			out = array(1,*args,**kwargs)
			return out
		else:
			out = hadamard(n-1,*args,**kwargs)
			return 1/sqrt(2)*array([[out,out],[out,-out]],*args,**kwargs)


class phasehadamard(array):
	'''
	array class of phaseshadamard
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,n,*args,**kwargs):
		assert n == 2, "phasehadamard only defined for n = 2"
		if n == 1:
			out = array(1,*args,**kwargs)
			return out
		elif n == 2:
			out = 1/sqrt(2)*array([[1,1],[1j,-1j]],*args,**kwargs)
			return out
		else:
			out = phasehadamard(n-1,*args,**kwargs)
			return 1/sqrt(2)*array([[out,out],[out,-out]],*args,**kwargs)


class cnot(array):
	'''
	array class of cnot
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,n,*args,**kwargs):
		assert n == 4, "cnot only defined for n = 4"
		if n == 1:
			out = array(1,*args,**kwargs)
			return out
		elif n == 4:
			out = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],*args,**kwargs)
			return out
		else:
			out = cnot(n-1,*args,**kwargs)
			return array([[out,out],[out,-out]],*args,**kwargs)


class toffoli(array):
	'''
	array class of toffoli
	Args:
		args (iterable): Array positional arguments
		kwargs (dict): Array keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,n,*args,**kwargs):
		assert n == 8, "toffoli only defined for n = 8"
		if n == 1:
			out = array(1,*args,**kwargs)
			return out
		elif n == 8:
			out = array([
				[1,0,0,0,0,0,0,0],
				[0,1,0,0,0,0,0,0],
				[0,0,1,0,0,0,0,0],
				[0,0,0,1,0,0,0,0],
				[0,0,0,0,1,0,0,0],
				[0,0,0,0,0,1,0,0],
				[0,0,0,0,0,0,0,1],
				[0,0,0,0,0,0,1,0]]
				,*args,**kwargs)
			return out
		else:
			out = toffoli(n-1,*args,**kwargs)
			return array([[out,out],[out,-out]],*args,**kwargs)




class tensor(qtn.Tensor):
	'''
	tensor class
	Args:
		args (iterable): Tensor arguments
		kwargs (dict): Tensor keyword arguments
	Returns:
		self (object): class instance
	'''
	def __new__(cls,*args,**kwargs):
		return qtn.Tensor(*args,**kwargs)
		# return super().__init__(cls,*args,**kwargs)


class tensornetwork(qtn.TensorNetwork):
	'''
	tensornetwork class
	Args:
		args (iterable): Tensor arguments
		kwargs (dict): Tensor keyword arguments
	Returns:
		self (object): class instance
	'''
	def __new__(cls,*args,**kwargs):
		return qtn.TensorNetwork(*args,**kwargs)
		# return super().__init__(cls,*args,**kwargs)


class mps(qtn.MatrixProductState):
	'''
	matrix product state class
	Args:
		data (iterable,int,str,callable,array,tensor,object): Tensor data
		args (iterable): Tensor arguments
		kwargs (dict): Tensor keyword arguments
	Returns:
		self (object): class instance
	'''
	def __new__(cls,data,*args,**kwargs):

		if isinstance(data,tensors):
			self = data
		elif isinstance(data,arrays):
			kwargs.update(dict(arrays=data))
			self = qtn.MPS_product_state(*args,**kwargs)
		elif isinstance(data,iterables):
			kwargs.update(dict(arrays=asarray(data)))
			self = qtn.MPS_product_state(*args,**kwargs)
		elif isinstance(data,str):
			kwargs.update(dict(binary=data))
			self = qtn.MPS_computational_state(*args,**kwargs)
		elif isinstance(data,integers):
			kwargs.update(dict(L=data))
			self = qtn.MPS_rand_state(*args,**kwargs)
		elif callable(data):
			kwargs.update(dict(fill_fn=data))
			self = qtn.MatrixProductState.from_fill_fn(*args,**kwargs)
		else:
			self = qtn.MatrixProductState(data,*args,**kwargs)
		return self
		# return super().__init__(cls,*args,**kwargs)


class gate(qtn.Gate):
	'''
	gate class
	Args:
		args (iterable): Gate arguments
		kwargs (dict): Gate keyword arguments
	Returns:
		out (array): array
	'''
	def __new__(cls,*args,**kwargs):
		return qtn.Gate(*args,**kwargs)
		# return super().__init__(cls,*args,**kwargs)



def representation(obj,to=True,contract=None,func=None,**kwargs):
	'''
	Get data of object
	Args:
		obj (tensor): object
		to (str): Return data as type, defaults as array, allowed strings in ['data','structure','array','tensor']
		contract (bool): Contract data
		func (callable): Wrapper function for data with signature func(obj)
		kwargs (dict): Additional keyword arguments for data
	Returns:
		obj (object): data of object
	'''
	
	if contract:
		obj = obj.contract(**kwargs)

	if to in ['array']:
		obj,structure = qtn.pack(obj)
		obj = array([obj[i].ravel() for i in obj]) if not isinstance(obj,arrays) else obj

	elif to in ['tensor']:
		args = tuple(sorted(set((tuple(sorted(i for i in obj.inds if i.startswith(string))) for string in tuple(sorted(set(i[0] for i in obj.inds)))))))
		kwargs = dict()
		obj = obj.to_dense(*args,**kwargs)

	elif to in ['data']:
		obj,structure = qtn.pack(obj)

	elif to in ['structure']:
		data,obj = qtn.pack(obj)

	elif to:
		obj,structure = qtn.pack(obj)
		obj = array([obj[i].ravel() for i in obj]) if not isinstance(obj,arrays) else obj

	if func is not None:
		obj = func(obj)

	return obj

if backend in ['jax']:

	def seeder(seed=None,size=None,split=False,data=False,**kwargs):
		'''
		Generate prng key
		Args:
			seed (int,array,Key): Seed for random number generation or random key for future seeding
			size(bool,int): Number of splits of random key
			split(bool): Split key to generate new key
			data (bool): Return key data
			kwargs (dict): Additional keyword arguments for seeding
		Returns:
			key (key,list[key]): Random key
		'''	

		# TODO merge random seeding for different numpy backends (jax vs autograd)

		generator = jax.random

		if is_key(seed):
			return seed

		if seed is None or isinstance(seed,integers):
			seed = seeded(seed)
		else:
			seed = asndarray(seed,dtype=uint)

		if size:
			key = generator.split(seed,num=size)
		elif split:
			seed,key = generator.split(seed)
		else:
			key = seed

		if data:
			key = generator.key_data(key).tolist()

		return key

	def seeded(seed=None):
		'''
		Set random seed
		Args:
			seed (int,array,Key): Seed for random number generation or random key for future seeding
		Returns:
			key (key): Random key
		'''

		if seed is None:
			bounds = [0,2**32]
			seed = onp.random.randint(*bounds)
		else:
			onp.random.seed(seed)
		
		key = jax.random.key(seed)

		return key


	class spawn(object):
		'''
		Spawn seeds class
		Args:
			seed (int,Key): Seed or Key
		'''

		def __init__(self,seed):
			self.seed = None
			self.key = None			
			self.init(seed)
			return

		def init(self,seed=None):
			if seed is not None:
				self.seed = seed
				self.key = jax.random.key(self.seed) if not is_key(self.seed) else self.seed
			return

		def split(self,shape=None,seed=None):
			self.init(seed)
			if shape is None:
				return self.key
			if not is_key(self.key):
				keys = onp.full(shape,self.key,dtype=object)
			else:
				keys = jax.random.split(self.key,shape)
			return keys

		def fold(self,folds=None,seed=None):
			self.init(seed)			
			if folds is None:
				return self.key
			key = jax.random.fold_in(self.key,self.hashes(folds))
			return key

		def hashes(self,hashes=None):
			'''
			Get hashes from data
			Args:
				hashes (iterable[str,int]): Hashing data
			Returns:
				hashes (int): Hashed integer
			'''
			if hashes is None:
				return hashes
			
			if isinstance(hashes,integers):
				hashes = (hashes,)

			keys = hashlib.sha1()
			
			for key in hashes:
				if isinstance(key, str):
				  keys.update(key.encode('utf-8'))
				elif isinstance(key, int):
				  keys.update(key.to_bytes((key.bit_length() + 7) // 8, byteorder='big'))
				else:
				  raise ValueError(f'Expected int or string, got: {key}')
			
			keys = keys.digest()
			hashes = int.from_bytes(keys[:4], byteorder='big')
			hashes = np.uint32(hashes)

			return hashes

		def __call__(self,shape=None,seed=None,wrapper=None):
			keys = self.split(shape)
			if callable(wrapper):
				keys = wrapper(keys)
			return keys


elif backend in ['jax.autograd','autograd','numpy']:

	def seeder(seed=None,size=None,data=None,**kwargs):
		'''
		Generate prng key
		Args:
			seed (int,array,Key): Seed for random number generation or random key for future seeding
			size(bool,int): Number of splits of random key
			data (bool): Return key data
			kwargs (dict): Additional keyword arguments for seeding
		Returns:
			key (key,list[key]): Random key
		'''	

		# TODO merge random seeding for different numpy backends (jax vs autograd)

		if seed is None or isinstance(seed,integers):
			seed = seeded(seed)
		else:
			seed = asndarray(seed,dtype=uint)

		if size:
			key = generator.randint(*bounds,size=size)
		else:
			key = seed

		if data:
			key = key.tolist()

		return key

	def seeded(seed=None):
		'''
		Set random seed
		Args:
			seed (int,array,Key): Seed for random number generation or random key for future seeding
		Returns:
			key (key): Random key
		'''

		if seed is None:
			bounds = [0,2**32]
			seed = np.random.randint(*bounds)

		np.random.seed(seed)
		
		key = seed

		return key

	class spawn(object):
		'''
		Spawn seeds class
		Args:
			seed (int,Key): Seed or Key
		'''

		def __init__(self,seed):
			self.seed = None
			self.key = None
			self.init(seed)
			return

		def init(self,seed=None):
			if seed is not None:
				onp.random.seed(seed)
				self.seed = seed
				self.key = seed
			return

		def split(self,shape=None,seed=None):
			self.init(seed)
			if shape is None:
				return self.key
			if not is_key(self.key):
				keys = onp.full(shape,self.key,dtype=object)
			else:
				keys = onp.full(shape,self.key,dtype=object)
			return keys

		def fold(self,folds=None,seed=None):
			self.init(seed)			
			if folds is None:
				return self.key
			key = self.key
			return key

		def hashes(self,hashes=None):
			'''
			Get hashes from data
			Args:
				hashes (iterable[str,int]): Hashing data
			Returns:
				hashes (int): Hashed integer
			'''
			if hashes is None:
				return hashes
			
			if isinstance(hashes,integers):
				hashes = (hashes,)

			keys = hashlib.sha1()
			
			for key in hashes:
				if isinstance(key, str):
				  keys.update(key.encode('utf-8'))
				elif isinstance(key, int):
				  keys.update(key.to_bytes((key.bit_length() + 7) // 8, byteorder='big'))
				else:
				  raise ValueError(f'Expected int or string, got: {key}')
			
			keys = keys.digest()
			hashes = int.from_bytes(keys[:4], byteorder='big')
			hashes = onp.uint32(hashes)

			return hashes

		def __call__(self,shape=None,seed=None,wrapper=None):
			keys = self.split(shape)
			if callable(wrapper):
				keys = wrapper(keys)
			return keys

if backend in ['jax']:

	def rand(shape=None,bounds=[0,1],key=None,seed=None,random='random',scale=None,mesh=None,dtype=None,**kwargs):
		'''
		Get random array
		Args:
			shape (int,iterable): Size or Shape of random array
			key (PRNGArrayKey,iterable[int],int): PRNG key or seed
			seed (PRNGArrayKey,iterable[int],int): PRNG key or seed
			bounds (iterable): Bounds on array
			random (str): Type of random distribution, allowed strings in ['random','rand','uniform','random','randint','randn','constant','gaussian','normal','haar','hermitian','symmetric','zero','one','plus','minus','zeros','ones','linspace','logspace']
			scale (int,float,str): Scale output, either number, or normalize with L1,L2 norms, allowed strings in ['normalize','1','2']
			mesh (int): Get meshgrid of array for mesh dimensions
			dtype (datatype): Datatype of array		
			kwargs (dict): Additional keyword arguments for random
		Returns:
			out (array): Random array
		'''	

		# TODO merge random seeding for different numpy backends (jax vs autograd)

		if shape is None:
			shape = 1
		if isinstance(shape,integers):
			shape = (shape,)

		key = seed if seed is not None else key
		
		key = seeder(key)

		generator = jax.random

		bounds = bounding(bounds,dtype=dtype)

		b = len(bounds)
		for i in range(b):
			if isinstance(bounds[i],str):
				if random in ['gaussian','randn','normal']:
					bounds[i] = int(((b-b%2)/(b-1))*i)-b//2
				else:
					bounds[i] = float(bounds)

		if random in ['random','rand','uniform']:
			assert not any(i in [float(j) for j in ["-inf","inf"]] for i in bounds), "'%s' random type must have finite bounds != [-inf,inf]"%(random)

		subrandoms = ['haar','hermitian','symmetric','one','zero','plus','minus']
		samplerandoms = ['choice','constant']
		complex = is_complexdtype(dtype) and (random not in subrandoms) and (random not in samplerandoms)
		_dtype = dtype
		dtype = datatype(dtype)

		if complex:
			shape = (2,*shape)

		if random in ['random','rand','uniform']:
			def func(key,shape,bounds,dtype):
				out = generator.uniform(key,shape,minval=bounds[0],maxval=bounds[1],dtype=dtype)
				# out = asarray(generator.uniform(low=bounds[0],high=bounds[1],size=shape).astype(dtype),dtype=dtype)
				return out
		elif random in ['randint']:
			def func(key,shape,bounds,dtype):		
				out = generator.randint(key,shape,minval=bounds[0],maxval=bounds[1]).astype(dtype)		
				# out = asarray(generator.randint(low=bounds[0],high=bounds[1],size=shape).astype(dtype),dtype=dtype)		
				return out
		elif random in ['gaussian','randn','normal']:
			def func(key,shape,bounds,dtype):
				out = (bounds[1]+bounds[0])/2 + sqrt((bounds[1]-bounds[0])/2)*generator.normal(key,shape,dtype=dtype)				
				# out = asarray((bounds[1]+bounds[0])/2 + sqrt((bounds[1]-bounds[0])/2)*generator.normal(size=shape).astype(dtype),dtype=dtype)
				return out
		elif random in ['constant']:
			def func(key,shape,bounds,dtype):
				out = (1/prod(shape))*ones(shape,dtype=datatype(dtype))
				return out			
		elif random in ['choice']:
			def func(key,shape,bounds,dtype):
				kwds = {'array':'a','weights':'p','replace':'replace','axis':'axis'}
				kwds = {**dict(key=key,shape=shape),
						**{kwds[attr]:kwargs.get(attr,None) for attr in kwds}
					}
				out = generator.choice(**kwds)
				return out			
		elif random in ['haar']:
			def func(key,shape,bounds,dtype):

				bounds = [-1,1]
				subrandom = 'gaussian'
				subdtype = 'complex'
				ndim = len(shape)
				shapes = shape

				if ndim < 2:
					shape = [*shape]*2
				elif shape[-1] != shape[-2]:
					shape = [*shape[:-1],*shape[-1:]*2]

				out = rand(shape,bounds=bounds,key=key,random=subrandom,dtype=subdtype,**kwargs)

				if out.ndim < 4:
					reshape = (*(1,)*(4-out.ndim),*out.shape)
				else:
					reshape = out.shape

				out = out.reshape(reshape)

				for i in range(out.shape[0]):
					for j in range(out.shape[1]):

						Q,R = qr(out[i,j])
						R = diag(R)
						R = diag(R/abs(R))
						
						out = inplace(out,(i,j),dot(Q,R))

				out = out.reshape(shape)

				assert allclose(1,real(einsum('...ij,...ij->...',out,conjugate(out)))/out.shape[-1])

				# Create random matrices versus vectors
				shape = shapes
				if ndim == 1: # Random vector
					out = out[...,0] 
				else:
					out = out[:,:]

				return out


		elif random in ['hermitian','symmetric']:
			def func(key,shape,bounds,dtype):
			
				bounds = [-1,1]
				subrandom = 'gaussian'
				subdtype = 'complex'
				ndim = len(shape)

				if ndim == 1:
					shape = [*shape]*2
					ndim = len(shape)

				out = rand(shape,bounds=bounds,key=key,random=subrandom,dtype=subdtype,**kwargs)	

				out = (out + conjugate(moveaxis(out,(-1,-2),(-2,-1))))/2


				if ndim == 1:
					out = diag(out)

				return out

		elif random in ['zero']:
			def func(key,shape,bounds,dtype):
				out = zeros(shape[-1],dtype=dtype)
				out = inplace(out,0,1)
				ndim = len(shape)
				if ndim == 1:
					pass
				elif ndim == 2:
					out = outer(out,out)
				elif ndim == 3:
					out = array([[out]*shape[1]]*shape[0])
				elif ndim == 4:
					out = outer(out,out)
					out = array([[out]*shape[1]]*shape[0])
				return out
		elif random in ['one']:
			def func(key,shape,bounds,dtype):
				out = zeros(shape[-1],dtype=dtype)
				out = inplace(out,-1,1)
				ndim = len(shape)
				if ndim == 1:
					pass
				elif ndim == 2:
					out = outer(out,out)
				elif ndim == 3:
					out = array([[out]*shape[1]]*shape[0])
				elif ndim == 4:
					out = outer(out,out)
					out = array([[out]*shape[1]]*shape[0])
				return out			
		elif random in ['plus']:
			def func(key,shape,bounds,dtype):
				out = zeros(shape[-1],dtype=dtype)
				out = inplace(out,slice(None),1/sqrt(shape[-1]))
				ndim = len(shape)
				if ndim == 1:
					pass
				elif ndim == 2:
					out = outer(out,out)
				elif ndim == 3:
					out = array([[out]*shape[1]]*shape[0])
				elif ndim == 4:
					out = outer(out,out)
					out = array([[out]*shape[1]]*shape[0])
				return out	
		elif random in ['minus']:
			def func(key,shape,bounds,dtype):
				out = zeros(shape[-1],dtype=dtype)
				out = inplace(out,slice(0,None,2),1/sqrt(shape[-1]))
				out = inplace(out,slice(1,None,2),-1/sqrt(shape[-1]))
				ndim = len(shape)
				if ndim == 1:
					pass
				elif ndim == 2:
					out = outer(out,out)
				elif ndim == 3:
					out = array([[out]*shape[1]]*shape[0])
				elif ndim == 4:
					out = outer(out,out)
					out = array([[out]*shape[1]]*shape[0])
				return out				
		elif random in ['zeros']:
			def func(key,shape,bounds,dtype):
				out = zeros(shape,dtype=dtype)
				return out
		elif random in ['ones']:
			def func(key,shape,bounds,dtype):
				out = ones(shape,dtype=dtype)
				return out	
		elif random in ['linspace']:
			def func(key,shape,bounds,dtype):
				num = shape[0] if not isinstance(shape,integers) else shape
				out = linspace(*bounds,num,dtype=dtype)
				return out
		elif random in ['logspace']:
			def func(key,shape,bounds,dtype):
				num = shape[0] if not isinstance(shape,integers) else shape
				out = logspace(*bounds,num,dtype=dtype)
				return out
		else:
			def func(key,shape,bounds,dtype):
				out = generator.uniform(key,shape,minval=bounds[0],maxval=bounds[1],dtype=dtype)
				# out = asarray(generator.uniform(low=bounds[0],high=bounds[1],size=shape).astype(dtype),dtype=dtype)
				return out

		if mesh is not None:
			out = array([out.reshape(-1) for out in np.meshgrid(*[func(key,shape,bounds,dtype) for i in range(mesh)])])
		else:
			out = func(key,shape,bounds,dtype)


		if scale in ['normalize']:
			out = out/out.sum()
		elif scale in ['1']:
			out = out/out.sum()
		elif scale in ['2']:
			out = out/sqrt(addition(conjugate(out)*out))
		elif scale is not None:
			out = out*scale

		if complex:
			out = out[0] + 1j*out[1]

		dtype = _dtype if _dtype is not None else out.dtype

		out = array(out,dtype=dtype)

		return out


	def random(shape=(),random='uniform',seed=None,key=None,dtype=None):
		'''
		Get random array
		Args:
			shape (int,iterable): Size or Shape of random array
			random (str): Type of random distribution, allowed strings in ['uniform','normal']
			seed (PRNGArrayKey,iterable[int],int): PRNG key or seed
			key (PRNGArrayKey,iterable[int],int): PRNG key or seed
			dtype (datatype): Datatype of array		
		Returns:
			out (array): Random array
		'''	

		key = seed if key is None else key
		generator = getattr(jax.random,random)

		return astype(generator(key,shape=shape),dtype=dtype)

	def randint(shape=(),bounds=[0,1],seed=None,key=None,dtype=None):
		'''
		Get random integer array
		Args:
			shape (int,iterable): Size or Shape of random array
			bounds (iterable): Bounds on array
			seed (PRNGArrayKey,iterable[int],int): PRNG key or seed
			key (PRNGArrayKey,iterable[int],int): PRNG key or seed
			dtype (datatype): Datatype of array		
		Returns:
			out (array): Random array
		'''	

		key = seed if key is None else key
		generator = jax.random.randint

		return astype(generator(key,shape=shape,minval=bounds[0],maxval=bounds[1]),dtype=dtype)

	def haar(shape=(),seed=None,key=None,dtype=None):
		'''
		Get random haar array
		Args:
			shape (int,iterable): Size or Shape of random array
			seed (PRNGArrayKey,iterable[int],int): PRNG key or seed
			key (PRNGArrayKey,iterable[int],int): PRNG key or seed
			dtype (datatype): Datatype of array		
		Returns:
			out (array): Random array
		'''	

		key = seed if key is None else key

		kwargs = dict(
			shape = (shape,)*2 if isinstance(shape,integers) else shape,
			seed = seed,
			key = key,
			random = 'normal',
			dtype=dtype
		)

		out = astype(random(**kwargs),dtype=dtype)

		out = out + 1j*out

		Q,R = qr(out)
		R = diag(R)
		R = diag(R/abs(R))
		
		out = dot(Q,R)

		return out


elif backend in ['jax.autograd','autograd','numpy']:

	def rand(shape=None,bounds=[0,1],key=None,seed=None,random='random',scale=None,mesh=None,dtype=None,**kwargs):
		'''
		Get random array
		Args:
			shape (int,iterable): Size or Shape of random array
			key (PRNGArrayKey,iterable[int],int): PRNG key or seed
			seed (PRNGArrayKey,iterable[int],int): PRNG key or seed
			bounds (iterable): Bounds on array
			random (str): Type of random distribution, allowed strings in ['random','rand','uniform','randint','randn','constant','gaussian','normal','haar','hermitian','symmetric','zero','one','plus','minus','zeros','ones','linspace','logspace']		
			scale (int,float,str): Scale output, either number, or normalize with L1,L2 norms, allowed strings in ['normalize','1','2']
			mesh (int): Get meshgrid of array for mesh dimensions
			dtype (datatype): Datatype of array		
			kwargs (dict): Additional keyword arguments for random
		Returns:
			out (array): Random array
		'''	

		# TODO merge random seeding for different numpy backends (jax vs autograd)

		if shape is None:
			shape = 1
		if isinstance(shape,integers):
			shape = (shape,)

		key = seed if seed is not None else key
		
		key = seeder(key)

		generator = onp.random.RandomState(key)

		bounds = bounding(bounds,dtype=dtype)

		if random in ['random','rand','uniform']:
			assert not any(i in ["-inf","inf"] for i in bounds), "'%s' random type must have finite bounds != [-inf,inf]"%(random)

		b = len(bounds)
		for i in range(b):
			if isinstance(bounds[i],str):
				if random in ['gaussian','randn','normal']:
					bounds[i] = int(((b-b%2)/(b-1))*i)-b//2
				elif random in ['constant']:
					bounds[i] = [0,1][i]
				else:
					bounds[i] = float(bounds)
		
		subrandoms = ['haar','hermitian','symmetric','one','zero','plus','minus']
		samplerandoms = ['choice','constant']		
		complex = is_complexdtype(dtype) and (random not in subrandoms) and (random not in samplerandoms)
		_dtype = dtype
		dtype = datatype(dtype)

		if complex:
			shape = (2,*shape)

		if random in ['random','rand','uniform']:
			def func(key,shape,bounds,dtype):
				# out = generator.uniform(key,shape,minval=bounds[0],maxval=bounds[1],dtype=dtype)
				out = generator.uniform(low=bounds[0],high=bounds[1],size=shape).astype(dtype)
				return out
		elif random in ['randint']:
			def func(key,shape,bounds,dtype):		
				# out = generator.randint(key,shape,minval=bounds[0],maxval=bounds[1],dtype=dtype)		
				out = generator.randint(low=bounds[0],high=bounds[1],size=shape).astype(dtype)		
				return out
		elif random in ['gaussian','randn','normal']:
			def func(key,shape,bounds,dtype):
				# out = (bounds[1]+bounds[0])/2 + sqrt((bounds[1]-bounds[0])/2)*generator.normal(key,shape,dtype=dtype)				
				out = (bounds[1]+bounds[0])/2 + sqrt((bounds[1]-bounds[0])/2)*generator.normal(size=shape).astype(dtype)				
				return out
		elif random in ['constant']:
			def func(key,shape,bounds,dtype):
				out = (1/prod(shape))*ones(shape,dtype=datatype(dtype))
				return out			
		elif random in ['choice']:
			def func(key,shape,bounds,dtype):
				kwds = {'array':'a','weights':'p','replace':'replace','axis':'axis'}
				kwds = {**dict(key=key,shape=shape),
						**{kwds[attr]:kwargs.get(attr,None) for attr in kwds}
					}
				out = generator.choice(**kwds)
				return out					
		elif random in ['haar']:
			def func(key,shape,bounds,dtype):

				bounds = [-1,1]
				subrandom = 'normal'
				subdtype = 'complex'
				ndim = len(shape)
				shapes = shape

				if ndim < 2:
					shape = [*shape]*2
				elif shape[-1] != shape[-2]:
					shape = [*shape[:-1],*shape[-1:]*2]

				out = rand(shape,bounds=bounds,key=key,random=subrandom,dtype=subdtype,**kwargs)

				if out.ndim < 4:
					reshape = (*(1,)*(4-out.ndim),*out.shape)
				else:
					reshape = out.shape


				out = out.reshape(reshape)

				for i in range(out.shape[0]):
					for j in range(out.shape[1]):

						Q,R = qr(out[i,j])
						R = diag(R)
						R = diag(R/abs(R))
						
						out = inplace(out,(i,j),dot(Q,R))

				out = out.reshape(shape)

				assert allclose(1,real(einsum('...ij,...ij->...',out,conjugate(out)))/out.shape[-1])

				# Create random matrices versus vectors
				shape = shapes
				if ndim == 1: # Random vector
					out = out[...,0] 
				else: # Random matrix
					out = out

				return out

		elif random in ['hermitian','symmetric']:
			def func(key,shape,bounds,dtype):
			
				bounds = [-1,1]
				subrandom = 'normal'
				subdtype = 'complex'
				ndim = len(shape)

				if ndim == 1:
					shape = [*shape]*2
					ndim = len(shape)

				out = rand(shape,bounds=bounds,key=key,random=subrandom,dtype=subdtype,**kwargs)	

				out = (out + conjugate(moveaxis(out,(-1,-2),(-2,-1))))/2


				if ndim == 1:
					out = diag(out)

				return out

		elif random in ['zero']:
			def func(key,shape,bounds,dtype):
				out = zeros(shape[-1],dtype=dtype)
				out = inplace(out,0,1)
				ndim = len(shape)
				if ndim == 1:
					pass
				elif ndim == 2:
					out = outer(out,out)
				elif ndim == 3:
					out = array([[out]*shape[1]]*shape[0])
				elif ndim == 4:
					out = outer(out,out)
					out = array([[out]*shape[1]]*shape[0])
				return out
		elif random in ['one']:
			def func(key,shape,bounds,dtype):
				out = zeros(shape[-1],dtype=dtype)
				out = inplace(out,-1,1)
				ndim = len(shape)
				if ndim == 1:
					pass
				elif ndim == 2:
					out = outer(out,out)
				elif ndim == 3:
					out = array([[out]*shape[1]]*shape[0])
				elif ndim == 4:
					out = outer(out,out)
					out = array([[out]*shape[1]]*shape[0])
				return out			
		elif random in ['plus']:
			def func(key,shape,bounds,dtype):
				out = zeros(shape[-1],dtype=dtype)
				out = inplace(out,slice(None),1/sqrt(shape[-1]))
				ndim = len(shape)
				if ndim == 1:
					pass
				elif ndim == 2:
					out = outer(out,out)
				elif ndim == 3:
					out = array([[out]*shape[1]]*shape[0])
				elif ndim == 4:
					out = outer(out,out)
					out = array([[out]*shape[1]]*shape[0])
				return out	
		elif random in ['minus']:
			def func(key,shape,bounds,dtype):
				out = zeros(shape[-1],dtype=dtype)
				out = inplace(out,slice(0,None,2),1/sqrt(shape[-1]))
				out = inplace(out,slice(1,None,2),-1/sqrt(shape[-1]))
				ndim = len(shape)
				if ndim == 1:
					pass
				elif ndim == 2:
					out = outer(out,out)
				elif ndim == 3:
					out = array([[out]*shape[1]]*shape[0])
				elif ndim == 4:
					out = outer(out,out)
					out = array([[out]*shape[1]]*shape[0])
				return out				
		elif random in ['zeros']:
			def func(key,shape,bounds,dtype):
				out = zeros(shape,dtype=dtype)
				return out
		elif random in ['ones']:
			def func(key,shape,bounds,dtype):
				out = ones(shape,dtype=dtype)
				return out	
		elif random in ['linspace']:
			def func(key,shape,bounds,dtype):
				num = shape[0] if not isinstance(shape,integers) else shape
				out = linspace(*bounds,num,dtype=dtype)
				return out					
		elif random in ['logspace']:
			def func(key,shape,bounds,dtype):
				num = shape[0] if not isinstance(shape,integers) else shape
				out = logspace(*bounds,num,dtype=dtype)
				return out								
		else:
			def func(key,shape,bounds,dtype):
				# out = generator.uniform(key,shape,minval=bounds[0],maxval=bounds[1],dtype=dtype)
				out = generator.uniform(low=bounds[0],high=bounds[1],size=shape).astype(dtype)
				return out

		if mesh is not None:
			out = array([out.reshape(-1) for out in np.meshgrid(*[func(key,shape,bounds,dtype) for i in range(mesh)])])
		else:
			out = func(key,shape,bounds,dtype)


		if scale in ['normalize']:
			out = out/out.sum()
		elif scale in ['1']:
			out = out/out.sum()
		elif scale in ['2']:
			out = out/sqrt(addition(conjugate(out)*out))
		elif scale is not None:
			out = out*scale

		if complex:
			out = out[0] + 1j*out[1]

		dtype = _dtype if _dtype is not None else out.dtype

		out = array(out,dtype=dtype)

		return out


	def random(shape=(),random='uniform',seed=None,key=None,dtype=None):
		'''
		Get random array
		Args:
			shape (int,iterable): Size or Shape of random array
			random (str): Type of random distribution, allowed strings in ['uniform','normal']
			seed (PRNGArrayKey,iterable[int],int): PRNG key or seed
			key (PRNGArrayKey,iterable[int],int): PRNG key or seed
			dtype (datatype): Datatype of array		
		Returns:
			out (array): Random array
		'''	

		key = seed if key is None else key
		generator = getattr(np.random,random)

		return astype(generator(size=shape),dtype=dtype)

	def randint(shape=(),bounds=[0,1],seed=None,key=None,dtype=None):
		'''
		Get random integer array
		Args:
			shape (int,iterable): Size or Shape of random array
			bounds (iterable): Bounds on array
			seed (PRNGArrayKey,iterable[int],int): PRNG key or seed
			key (PRNGArrayKey,iterable[int],int): PRNG key or seed
			dtype (datatype): Datatype of array		
		Returns:
			out (array): Random array
		'''	

		key = seed if key is None else key
		generator = np.random.randint

		return astype(generator(*bounds,size=shape),dtype=dtype)

	def haar(shape=(),seed=None,key=None,dtype=None):
		'''
		Get random haar array
		Args:
			shape (int,iterable): Size or Shape of random array
			seed (PRNGArrayKey,iterable[int],int): PRNG key or seed
			key (PRNGArrayKey,iterable[int],int): PRNG key or seed
			dtype (datatype): Datatype of array		
		Returns:
			out (array): Random array
		'''	

		key = seed if key is None else key

		kwargs = dict(
			shape = (shape,)*2 if isinstance(shape,integers) else shape,
			seed = seed,
			key = key,
			random = 'normal',
			dtype=dtype
		)

		out = astype(random(**kwargs),dtype=dtype)

		out = out + 1j*out

		Q,R = qr(out)
		R = diag(R)
		R = diag(R/abs(R))
		
		out = dot(Q,R)

		return out


def _svd(A,k=None):
	'''
	Perform SVD on array, possibly reduced rank k
	Args:
		A (array): array of shape (n,m)
		k (int): reduced rank of SVD, defaults to max(n,m) if None
	Returns
		U (array): unitary of left eigenvectors of A of shape (n,m)
		S (array): array of singular values of shape (m,)
		V (array): conjugate of unitary of right eigenvectors of A of shape (m,m)
	'''

	n,m = A.shape

	U,S,V = onp.linalg.svd(A)

	p,r,q = U.shape[0],S.size,V.shape[0]

	if k > p:
		U = onp.concatenate((U,onp.zeros((n,k-p))),axis=1)
	else:
		U = U[:,:k]
	if k > r:
		S = onp.concatenate((S,onp.zeros(k-r)),axis=0)
	else:
		S = S[:k]		
	if k > q:
		V = onp.concatenate((V,onp.zeros((k-q,m))),axis=0)
	else:
		V = V[:k,:]


	S = onp.diag(S)

	return U,S,V

def eig(a,compute_v=False,hermitian=False):
	'''
	Compute eigenvalues and eigenvectors
	Args:
		a (array): Array to compute eigenvalues and eigenvectors of shape (...,n,n)
		compute_v (bool): Compute V eigenvectors in addition to eigenvalues
		hermitian (bool): Whether array is Hermitian
	Returns:
		eigenvalues (array): Array of eigenvalues of shape (...,n)
		eigenvectors (array): Array of normalized eigenvectors of shape (...,n,n)
	'''
	if compute_v:
		if hermitian:
			_eig = np.linalg.eigh
		else:
			_eig = sp.linalg.eig
	else:
		if hermitian:
			_eig = np.linalg.eigvalsh
		else:
			_eig = np.linalg.eigvals
	return _eig(a)

def schur(a,compute_v=False,output=None):
	'''
	Compute schur decomposition of array
	Args:
		a (array): Array to compute schur decmposition of shape (...,n,n)
		compute_v (bool): Compute unitary transformation of decomposition
		output (str): Return real or complex decomposition, allowed strings in ['real','complex']
	Returns:
		triangular (array): Array of triangular similar array (...,n,n)
		unitary (array): Array of unitary transformation of decomposition of shape (...,n,n)
	'''
	_schur = sp.linalg.schur
	output = {True:'complex',False:'real'}[is_complexdtype(a.dtype)] if output is None else output
	
	triangular,unitary = _schur(a,outut=output)

	if compute_v:
		return triangular,unitary
	else:
		return triangular

def svd(a,full_matrices=True,compute_uv=False,hermitian=False):
	'''
	Compute singular values of an array
	Args:
		a (array): Array to compute eigenvalues of shape (...,n,n)
		full_matrices (bool): Compute full matrices of right,left singular values
		compute_uv (bool): Compute U,V in addition to singular values
		hermitian (bool): Whether array is Hermitian				
	Returns:
		singular (array): Array of singular values of shape (...,n)
		rightvectors (array): Array of right singular vectors of shape (...,n,n)
		leftvectors (array): Array of left singular vectors of shape (...,n,n)
	'''
	return np.linalg.svd(a,full_matrices=full_matrices,compute_uv=compute_uv,hermitian=hermitian)

def qr(a):
	'''
	Compute QR decomposition of array
	Args:
		a (array): Array to compute QR decomposition of shape (...,n,n)
	Returns:
		Q (array): Q factor of shape (...,n,n)
		R (array): R factor of shape (...,n,n)
	'''
	return np.linalg.qr(a)


def cholesky(a):
	'''
	Compute cholesky decomposition of array
	Args:
		a (array): Array to compute cholesky decomposition of shape (...,n,n)
	Returns:
		L (array): Cholesky factor of shape (...,n,n)
	'''
	return np.linalg.cholesky(a)

def lstsq(x,y):
	'''
	Compute least squares fit between x and y
	Args:
		x (array): Array of input data
		y (array): Array of output data
	Returns:
		out (array): Least squares fit
	'''
	out = np.linalg.lstsq(x,y)[0] + 0.0
	return out


@jit
def inv(a):
	'''
	Compute inverse of a
	Args:
		a (array): Array to compute inverse
	Returns:
		out (array): Inverse
	'''
	return np.linalg.inv(a)

def spectrum(func,compute_v=False,hermitian=False):
	'''
	Compute eigenvalues and eigenvectors of a function
	Args:
		func (callable): Function to compute eigenvalues and eigenvectors of shape (...,n,n)
		compute_v (bool): Compute V eigenvectors in addition to eigenvalues
		hermitian (bool): Whether array is Hermitian
	Returns:
		wrapper (callable): Returns:
			eigenvalues (array): Array of eigenvalues of shape (...,n)
			eigenvectors (array): Array of normalized eigenvectors of shape (...,n,n)
	'''

	@jit
	def wrapper(*args,**kwargs):
		return eig(func(*args,**kwargs),compute_v=compute_v,hermitian=hermitian)

	return wrapper

@partial(jit,static_argnums=(1,))
def mean(a,axis=None):
	'''
	Compute mean of array along axis
	Args:
		a (array): array to compute mean
		axis (int): axis to compute over. Flattens array if None.
	Returns:
		out (array): mean of array
	'''
	return np.mean(a,axis=axis)

@partial(jit,static_argnums=(1,2,))
def std(a,axis=None,ddof=None):
	'''
	Compute std of array along axis
	Args:
		a (array): array to compute std
		axis (int): axis to compute over. Flattens array if None.
		ddof (int): Number of degrees of freedom
	Returns:
		out (array): std of array
	'''
	return np.std(a,axis=axis,ddof=ddof)


@partial(jit,static_argnums=(1,2,))
def sem(a,axis=None,ddof=None):

	'''
	Compute standard error of mean
	Args:
		a (array): array to compute sem
		axis (int): axis to compute over. Flattens array if None.
		ddof (int): Number of degrees of freedom
	Returns:
		out (array): sem of array
	'''
	if axis is None:
		size = a.size
	elif isinstance(axis,integers):
		size = a.shape[axis]
	else:
		size = int(product([a.shape[ax] for ax in axis]))
	return std(a,axis=axis,ddof=ddof)/np.sqrt(size)

@partial(jit,static_argnums=(1,))
def nanmean(a,axis=None):
	'''
	Compute nanmean of array along axis
	Args:
		a (array): array to compute nanmean
		axis (int): axis to compute over. Flattens array if None.
	Returns:
		out (array): nanmean of array
	'''
	return np.nanmean(a,axis=axis)

@partial(jit,static_argnums=(1,2,))
def nanstd(a,axis=None,ddof=None):
	'''
	Compute nanstd of array along axis
	Args:
		a (array): array to compute nanstd
		axis (int): axis to compute over. Flattens array if None.
		ddof (int): Number of degrees of freedom
	Returns:
		out (array): nanstd of array
	'''
	return np.nanstd(a,axis=axis,ddof=ddof)


@partial(jit,static_argnums=(1,2,))
def nansem(a,axis=None,ddof=None):

	'''
	Compute nan standard error of mean
	Args:
		a (array): array to compute sem
		axis (int): axis to compute over. Flattens array if None.
		ddof (int): Number of degrees of freedom
	Returns:
		out (array): sem of array
	'''
	if axis is None:
		size = a.size
	elif isinstance(axis,integers):
		size = a.shape[axis]
	else:
		size = int(product([a.shape[ax] for ax in axis]))
	return nanstd(a,axis=axis,ddof=ddof)/np.sqrt(size)

# @partial(jit,static_argnums=(2,3,4,))
def bootstrap(a,size=None,shape=(),axis=None,replace=True,weights=None,key=None,seed=None):
	'''
	Compute bootstrapped data
	Args:
		a (array): array to compute bootstrapped sampling
		size (int): Number of bootstrap samplings, defaults to first element of shape if None
		shape (int,iterable): Size or shape of sample
		axis (int): axis to compute over. Flattens array if None.
		replace (bool): Sample with replacement
		weights (array): Weights of sampling of length a.shape[axis]
		key (PRNGArrayKey,iterable[int],int): PRNG key or seed
		seed (PRNGArrayKey,iterable[int],int): PRNG key or seed
	Returns:
		out (array): bootstrapped sample of array of shape [*a.shape[:axis],*shape,*a.shape[axis+1:]]
	'''

	if seed is not None:
		key = seed

	if isinstance(shape,integers):
		shape = (size,shape)
	elif size is not None:
		shape = (size,*shape)

	if is_dataframe(a):
		return a.sample(shape,replace=replace,weights=weights,random_state=key)
	else:
		return rand(shape=shape,key=key,array=a,axis=axis,replace=replace,weights=weights)

@jit
def nansqrt(a):
	'''
	Compute nansqrt
	Args:
		a (array): array to compute nansqrt
	Returns:
		out (array): nansqrt of array
	'''
	return np.sqrt(a)


def factorial(n,exact=True):
	'''
	Compute factorial n!
	Args:
		n (int): Number to compute factorial
		exact (bool): Compute factorial exactly
	Returns:
		n (int,float): Factorial of n
	'''
	n = ospsp.factorial(n,exact=exact)
	return n

if backend in ['jax','jax.autograd']:

	@partial(jit,static_argnums=(1,2,3,))
	def norm(a,axis=None,ord=2,keepdims=False):
		'''
		Norm of array
		Args:
			a (array): array to be normalized
			axis (int,iterable[int]): axis to normalize over. Flattens array if None.
			ord (int,str): order of normalization
			keepdims (bool): Keep axis of size 1 along normalization
		Returns:
			out (array): Norm of array
		'''

		# TODO merge norm for different numpy backends (jax vs autograd)

		out = np.linalg.norm(a,axis=axis,ord=ord,keepdims=keepdims)

		return out

elif backend in ['autograd','numpy']:

	@partial(jit,static_argnums=(1,2,3,))
	def norm(a,axis=None,ord=2,keepdims=False):
		'''
		Norm of array
		Args:
			a (array): array to be normalized
			axis (int,iterable[int]): axis to normalize over. Flattens array if None.
			ord (int,str): order of normalization
			keepdims (bool): Keep axis of size 1 along normalization
		Returns:
			out (array): Norm of array
		'''

		# TODO merge norm for different numpy backends (jax vs autograd)

		out = addition(a**ord,axis=axis)**(1/ord)

		return out		


@jit
def norm2(a,b=None):
	'''
	2-Norm squared of array
	Args:
		a (array): array to be normalized
		b (array): array to weight normalization
	Returns:
		out (array): Norm of array
	'''
	if b is None:
		out = dot(conjugate(a),a)
	elif b.ndim == 1:
		out = dot(conjugate(a),a*b)
	elif b.ndim == 2:
		out = dot(dot(conjugate(a),b),a)
	else:
		out = dot(conjugate(a),a)

	return out



def contraction(data=None,state=None,where=None,**kwargs):
	'''
	Contract data and state
	Args:
		data (array,tensor): data
		state (array,tensor): state
		where (int,str,iterable[int,str]): indices of contraction
		kwargs (dict): Additional keyword arguments for contraction
	Returns:
		func (callable): contracted data and state with signature func(data,state,where=where)
	'''

	def func(data,state,where=where):
		return data

	if state is None:
		wrapper = jit
	elif isinstance(state,arrays):
		wrapper = jit
	elif isinstance(state,tensors):
		wrapper = None
	else:
		wrapper = None

	if data is None:

		if state is None:
		
			def func(data,state,where=where):
				return data

		elif isinstance(state,arrays):
			
			if state.ndim == 1:

				def func(data,state,where=where):
					return data
	
			elif state.ndim == 2:
				
				def func(data,state,where=where):
					return data
	
		elif isinstance(state,tensors):

			def func(data,state,where=where):
				return data

	elif callable(data) and not isinstance(data,(*arrays,*tensors)):
	
		def func(data,state,where=where):
			return data(state)

	elif isinstance(data,arrays):

		if data.ndim == 0:

			if state is None:
			
				def func(data,state,where=where):
					return data
			
			elif isinstance(state,arrays):

				if state.ndim == 1:

					def func(data,state,where=where):
						return data

				elif state.ndim == 2:
					
					def func(data,state,where=where):
						return data

			elif isinstance(state,tensors):

				def func(data,state,where=where):
					raise NotImplementedError("Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

		elif data.ndim == 1:
			
			if state is None:
			
				def func(data,state,where=where):
					return data

			elif isinstance(state,arrays):

				if state.ndim == 1:

					def func(data,state,where=where):
						return data

				elif state.ndim == 2:
					
					def func(data,state,where=where):
						return data

			elif isinstance(state,tensors):

				raise NotImplementedError("Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

		elif data.ndim == 2:

			if state is None:

				state = data

				if where is None:
					subscripts = 'ij,jk->ik'
					shapes = (data.shape,state.shape)
					einsummation = einsum(subscripts,*shapes)
					
					def func(data,state,where=where):
						return einsummation(data,state)

				else:
					subscripts = 'ij,jk...->ik...'
					shapes = (data.shape,data.shape[(data.ndim-state.ndim):])
					einsummation = einsum(subscripts,*shapes)
					
					swapper = swap(state,**kwargs,transform=True,execute=False)
					_swapper = swap(state,**kwargs,transform=False,execute=False)
					
					def func(data,state,where=where):
						return _swapper(einsummation(data,swapper(state)))


			elif isinstance(state,arrays):

				if state.ndim == 1:
					
					if where is None:
						subscripts = 'ij,j->i'
						shapes = (data.shape,state.shape)
						einsummation = einsum(subscripts,*shapes)
						
						def func(data,state,where=where):
							return einsummation(data,state)
					else:
						subscripts = 'ij,j...->i...'
						shapes = (data.shape,data.shape[(data.ndim-state.ndim):])
						einsummation = einsum(subscripts,*shapes)
						
						swapper = swap(state,**kwargs,transform=True,execute=False)
						_swapper = swap(state,**kwargs,transform=False,execute=False)
						
						def func(data,state,where=where):
							return _swapper(einsummation(data,swapper(state)))


				elif state.ndim == 2:
				
					if where is None:
						subscripts = 'ij,jk,lk->il'
						shapes = (data.shape,state.shape,data.shape)
						einsummation = einsum(subscripts,*shapes)
						def func(data,state,where=where):
							return einsummation(data,state,conjugate(data))
					else:
						subscripts = 'ij,jk...,lk->il...'
						shapes = (data.shape,data.shape[(data.ndim-state.ndim):],data.shape)
						einsummation = einsum(subscripts,*shapes)
						
						swapper = swap(state,**kwargs,transform=True,execute=False)
						_swapper = swap(state,**kwargs,transform=False,execute=False)
						
						def func(data,state,where=where):
							return _swapper(einsummation(data,swapper(state),conjugate(data)))							

			elif isinstance(state,tensors):
				
				def func(data,state,where=where):
					return state.gate(data,where=where)

		elif data.ndim == 3:

			if state is None:

				state = data
				
				if where is None:
					subscripts = 'uij,...j->i...'
					shapes = (data.shape,state.shape)
					einsummation = einsum(subscripts,*shapes)
					
					def func(data,state,where=where):
						return einsummation(data,state)

				else:
					subscripts = 'uij,...j->i...'
					shapes = (data.shape,state.shape)
					einsummation = einsum(subscripts,*shapes)
					
					def func(data,state,where=where):
						return einsummation(data,state)


			elif isinstance(state,arrays):

				if state.ndim == 1:
					
					if where is None:
						subscripts = 'uij,j->i'
						shapes = (data.shape,state.shape)
						einsummation = einsum(subscripts,*shapes)
					
						def func(data,state,where=where):
							return einsummation(data,state)
					else:
						subscripts = 'uij,j...->i...'
						shapes = (data.shape,data.shape[(data.ndim-state.ndim):])
						einsummation = einsum(subscripts,*shapes)
						
						swapper = swap(state,**kwargs,transform=True,execute=False)
						_swapper = swap(state,**kwargs,transform=False,execute=False)
						
						def func(data,state,where=where):
							return _swapper(einsummation(data,swapper(state)))


				elif state.ndim == 2:

					if where is None:					
						subscripts = 'uij,jk,ulk->il'
						shapes = (data.shape,state.shape,data.shape)
						einsummation = einsum(subscripts,*shapes)

						def func(data,state,where=where):
							return einsummation(data,state,conjugate(data))

					else:
						subscripts = 'uij,jk...,ulk->il...'
						shapes = (data.shape,data.shape[(data.ndim-state.ndim):],data.shape)
						einsummation = einsum(subscripts,*shapes)

						swapper = swap(state,**kwargs,transform=True,execute=False)
						_swapper = swap(state,**kwargs,transform=False,execute=False)
						
						def func(data,state,where=where):
							return _swapper(einsummation(data,swapper(state),conjugate(data)))


			elif isinstance(state,tensors):

				raise NotImplementedError("Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

	elif isinstance(data,tensors):

		if state is None:
			def func(data,state,where=where):
				return data
		
		elif isinstance(state,arrays):

			if state.ndim == 1:
				raise NotImplementedError("Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))
			elif state.ndim == 2:
				raise NotImplementedError("Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

		elif isinstance(state,tensors):

			def func(data,state,where=where):
				return state.gate(data,where=where)

	func = wrapper(func) if wrapper is not None else func

	return func


def gradient_contraction(data=None,state=None,where=None,**kwargs):
	'''
	Contract grad, data and state
	Args:
		data (array,tensor): data
		state (array,tensor): state
		where (int,str,iterable[int,str]): indices of contraction
		kwargs (dict): Additional keyword arguments for contraction		
	Returns:
		func (callable): contracted data and state with signature func(grad,data,state,where=where)
	'''

	def func(grad,data,state,where=where):
		return 0

	if state is None:
		wrapper = jit
	elif isinstance(state,arrays):
		wrapper = jit
	elif isinstance(state,tensors):
		wrapper = None
	else:
		wrapper = None

	if data is None:
		
		if state is None:
		
			def func(grad,data,state,where=where):
				return grad

		elif isinstance(state,arrays):

			if state.ndim == 1:

				def func(grad,data,state,where=where):
					return grad

			elif state.ndim == 2:
				
				def func(grad,data,state,where=where):
					return grad
	
		elif isinstance(state,tensors):

			raise NotImplementedError("Gradient Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

	elif callable(data) and not isinstance(data,(*arrays,*tensors)):
	
		raise NotImplementedError("Gradient Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

	elif isinstance(data,arrays):

		if data.ndim == 0:

			if state is None:
			
				def func(grad,data,state,where=where):
					return grad
			
			elif isinstance(state,arrays):

				if state.ndim == 1:

					def func(grad,data,state,where=where):
						return grad

				elif state.ndim == 2:
					
					def func(grad,data,state,where=where):
						return grad

			elif isinstance(state,tensors):

				raise NotImplementedError("Gradient Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

		elif data.ndim == 1:
			
			if state is None:
			
				def func(grad,data,state,where=where):
					return grad

			elif isinstance(state,arrays):

				if state.ndim == 1:

					def func(grad,data,state,where=where):
						return grad

				elif state.ndim == 2:
					
					def func(grad,data,state,where=where):
						return grad

			elif isinstance(state,tensors):

				raise NotImplementedError("Gradient Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

		elif data.ndim == 2:

			if state is None:

				state = data

				if where is None:
					subscripts = 'ij,jk->ik'
					shapes = (data.shape,state.shape)
					einsummation = einsum(subscripts,*shapes)
					
					def func(grad,data,state,where=where):
						return einsummation(grad,state)

				else:
					subscripts = 'ij,jk...->ik...'
					shapes = (data.shape,data.shape[(data.ndim-state.ndim):])
					einsummation = einsum(subscripts,*shapes)
					
					swapper = swap(state,**kwargs,transform=True,execute=False)
					_swapper = swap(state,**kwargs,transform=False,execute=False)
					
					def func(grad,data,state,where=where):
						return _swapper(einsummation(grad,swapper(state)))					

			elif isinstance(state,arrays):

				if state.ndim == 1:

					if where is None:
						subscripts = 'ij,j->i'
						shapes = (data.shape,state.shape)
						einsummation = einsum(subscripts,*shapes)
						
						def func(grad,data,state,where=where):
							return einsummation(grad,state)

					else:
						subscripts = 'ij,j...->i...'
						shapes = (data.shape,data.shape[(data.ndim-state.ndim):])
						einsummation = einsum(subscripts,*shapes)

						swapper = swap(state,**kwargs,transform=True,execute=False)
						_swapper = swap(state,**kwargs,transform=False,execute=False)
						
						def func(grad,data,state,where=where):
							return _swapper(einsummation(grad,swapper(state)))


				elif state.ndim == 2:
					
					if where is None:
						subscripts = 'ij,jk,lk->il'
						shapes = (data.shape,state.shape,data.shape)
						einsummation = einsum(subscripts,*shapes)
						
						def func(grad,data,state,where=where):
							out = einsummation(grad,state,conjugate(data))
							return out + dagger(out)

					else:
						subscripts = 'ij,jk...,lk->il...'
						shapes = (data.shape,data.shape[(data.ndim-state.ndim):],data.shape)
						einsummation = einsum(subscripts,*shapes)

						swapper = swap(state,**kwargs,transform=True,execute=False)
						_swapper = swap(state,**kwargs,transform=False,execute=False)
						
						def func(grad,data,state,where=where):
							out = _swapper(einsummation(grad,swapper(state),conjugate(data)))
							return out + dagger(out)				

			elif isinstance(state,tensors):

				raise NotImplementedError("Gradient Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

		elif data.ndim == 3:

			if state is None:

				state = data

				if where is None:
					subscripts = 'uij,...j->i...'
					shapes = (data.shape,state.shape)
					einsummation = einsum(subscripts,*shapes)
					
					def func(grad,data,state,where=where):
						return einsummation(grad,state)

				else:
					subscripts = 'uij,...j->i...'
					shapes = (data.shape,state.shape)
					einsummation = einsum(subscripts,*shapes)
					
					def func(grad,data,state,where=where):
						return einsummation(grad,state)


			elif isinstance(state,arrays):

				if state.ndim == 1:
					
					if where is None:
						subscripts = 'uij,j->i'
						shapes = (data.shape,state.shape)
						einsummation = einsum(subscripts,*shapes)
						
						def func(grad,data,state,where=where):
							return einsummation(grad,state)

					else:
						subscripts = 'uij,j...->i...'
						shapes = (data.shape,data.shape[(data.ndim-state.ndim):])
						einsummation = einsum(subscripts,*shapes)

						swapper = swap(state,**kwargs,transform=True,execute=False)
						_swapper = swap(state,**kwargs,transform=False,execute=False)
						
						def func(grad,data,state,where=where):
							return _swapper(einsummation(grad,swapper(state)))

				elif state.ndim == 2:
					
					if where is None:
						subscripts = 'uij,jk,ulk->il'
						shapes = (data.shape,state.shape,data.shape)
						einsummation = einsum(subscripts,*shapes)
						
						def func(grad,data,state,where=where):
							out = einsummation(grad,state,conjugate(data))
							return out + dagger(out)

					else:
						subscripts = 'uij,jk...,ulk->il...'
						shapes = (data.shape,data.shape[(data.ndim-state.ndim):],data.shape)
						einsummation = einsum(subscripts,*shapes)

						swapper = swap(state,**kwargs,transform=True,execute=False)
						_swapper = swap(state,**kwargs,transform=False,execute=False)
						
						def func(grad,data,state,where=where):
							out = _swapper(einsummation(grad,swapper(state),conjugate(data)))
							return out + dagger(out)						

			elif isinstance(state,tensors):

				raise NotImplementedError("Gradient Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

	elif isinstance(data,tensors):

		raise NotImplementedError("Gradient Contraction Not Implemented for data: %r , state: %r"%(type(data),type(state)))

	func = wrapper(func) if wrapper is not None else func

	return func

def metrics(metric,shapes=None,label=None,weights=None,optimize=None,returns=None):
	'''
	Setup metrics
	Args:
		metric (str,callable): Type of metric
		shapes (iterable[tuple[int]]): Shapes of Operators
		label (array,callable): Label			
		weights (array): Weights
		optimize (bool,str,iterable): Contraction type			
		returns (bool): Return metric gradients
	Returns:
		func (callable): Metric function with signature func(*operands,label,weights)
		grad (callable): Metric gradient with signature grad(*operands,label,weights,*gradients)
		grad_analytical (callable): Metric analytical gradient with signature grad_analytical(*operands,label,weights,*gradients)
	'''
	
	if shapes:
		size = sum(int(product(shape)**(1/len(shape))) for shape in shapes[:2] if shape is not None)//len(shapes[:2])
		ndim = min([len(shape) for shape in shapes[:2] if shape is not None])
	else:
		size = 1
		ndim = None


	if callable(metric):
			metric = metric
			func = jit(metric)
			grad = jit(gradient(metric))
			# grad = gradient(func,mode='fwd',holomorphic=True,move=True)			
			grad_analytical = jit(gradient(metric))
	
	elif metric is None:

		func = inner_norm
		grad_analytical = gradient_inner_norm

		def wrapper_func(out,*operands):
			return out/2

		def wrapper_grad(out,*operands):
			return out/2

	elif metric in ['lstsq']:
		func = mse
		grad_analytical = gradient_mse

		def wrapper_func(out,*operands):
			return out/2

		def wrapper_grad(out,*operands):
			return out/2	

	elif metric in ['mse']:
		func = mse
		grad_analytical = gradient_mse

		def wrapper_func(out,*operands):
			return out/operands[0].size/2

		def wrapper_grad(out,*operands):
			return out/operands[0].size/2					

	elif metric in ['norm']:

		func = inner_norm
		grad_analytical = gradient_inner_norm

		def wrapper_func(out,*operands):
			return out/operands[0].shape[-1]/2
		
		def wrapper_grad(out,*operands):
			return out/operands[0].shape[-1]/2

	elif metric in ['abs2']:

		func = inner_abs2
		grad_analytical = gradient_inner_abs2

		if ndim is not None:
			if ndim == 1:
				def wrapper_func(out,*operands):
					return 1 - out

				def wrapper_grad(out,*operands):
					return - out

			elif ndim == 2:
				def wrapper_func(out,*operands):
					return 1 - out/((operands[0].shape[-1]*operands[0].shape[-2]))

				def wrapper_grad(out,*operands):
					return - out/((operands[0].shape[-1]*operands[0].shape[-2]))
			else:
				def wrapper_func(out,*operands):
					return 1 - out/((operands[0].shape[-1]*operands[0].shape[-2]))

				def wrapper_grad(out,*operands):
					return - out/((operands[0].shape[-1]*operands[0].shape[-2]))

		else:
			def wrapper_func(out,*operands):
				return 1 - out/((operands[0].shape[-1]*operands[0].shape[-2]) if operands[0].ndim > 1 else 1)

			def wrapper_grad(out,*operands):
				return - out/((operands[0].shape[-1]*operands[0].shape[-2]) if operands[0].ndim > 1 else 1)

	elif metric in ['real']:

		func = inner_real
		grad_analytical = gradient_inner_real

		def wrapper_func(out,*operands):
			return 1 - out

		def wrapper_grad(out,*operands):
			return  - out

	elif metric in ['imag']:

		func = inner_imag
		grad_analytical = gradient_inner_imag

		def wrapper_func(out,*operands):
			return 1 - out

		def wrapper_grad(out,*operands):
			return - out

	else:

		func = inner_norm
		grad_analytical = gradient_inner_norm

		def wrapper_func(out,*operands):
			return out/operands[0].shape[-1]/2

		def wrapper_grad(out,*operands):
			return out/operands[0].shape[-1]/2


	shapes_func = (*(shape for shape in shapes if shape is not None),) if shapes else ()
	optimize_func = optimize
	wrapper_func = jit(wrapper_func)

	shapes_grad = (*(shape for shape in shapes if shape is not None),(size**2,*shapes[0]),) if shapes else ()
	optimize_grad = optimize
	wrapper_grad = jit(wrapper_grad)

	if shapes_func:
		func = func(*shapes_func,optimize=optimize_func,wrapper=wrapper_func)
	else:
		func = partial(func,optimize=optimize_grad,wrapper=wrapper_func)

	if shapes_grad:
		grad_analytical = grad_analytical(*shapes_grad,optimize=optimize_func,wrapper=wrapper_grad)
	else:
		grad_analytical = partial(grad_analytical,optimize=optimize_grad,wrapper=wrapper_grad)
	# grad_analytical = gradient(func,mode='fwd',holomorphic=True,move=True)

	grad = grad_analytical

	func = jit(func)
	grad = jit(grad)
	grad_analytical = jit(grad_analytical)

	if (label is not None) and (weights is not None):

		if callable(label):
			label = label()

		if label is not None and metric in ['abs2','real']:
			label = conjugate(label)

		weights = inv(weights) if weights.ndim>1 else 1/weights**2

		def func(*operands,func=func,label=label,weights=weights):
			return func(*operands[:1],label,weights,*operands[1:])
		def grad(*operands,func=grad,label=label,weights=weights):
			return func(*operands[:1],label,weights,*operands[1:])				
		def grad_analytical(*operands,func=grad_analytical,label=label,weights=weights):
			return func(*operands[:1],label,weights,*operands[1:])
	
	elif (label is not None):

		if callable(label):
			label = label()

		if label is not None and metric in ['abs2','real']:
			label = conjugate(label)	

		def func(*operands,func=func,label=label):
			return func(*operands[:1],label,*operands[1:])
		def grad(*operands,func=grad,label=label):
			return func(*operands[:1],label,*operands[1:])				
		def grad_analytical(*operands,func=grad_analytical,label=label):
			return func(*operands[:1],label,*operands[1:])

	elif (weights is not None):

		weights = inv(weights) if weights.ndim>1 else 1/weights**2

		def func(*operands,func=func,weights=weights):
			return func(*operands[:2],weights,*operands[2:])
		def grad(*operands,func=grad,weights=weights):
			return func(*operands[:2],weights,*operands[2:])				
		def grad_analytical(*operands,func=grad_analytical,weights=weights):
			return func(*operands[:2],weights,*operands[2:])

	func = jit(func)
	grad = jit(grad)
	grad_analytical = jit(grad_analytical)

	if not returns:
		return func
	else:
		return func,grad,grad_analytical




def mse(*operands,optimize=True,wrapper=None):
	'''
	Calculate square inner product of arrays
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)
	length = len([shape for shape in shapes if shape is not None])

	if ndim == 1:
		if length == 2:
			subscripts = 'i,i->'
		elif length == 3:
			if len(shapes[2]) == 1:
				subscripts = 'i,i,i->'
			elif len(shapes[2]) == 2:
				subscripts = 'i,j,ij->'			
			else:
				subscripts = 'i,i->'
		else:
			subscripts = 'i,i->'
	elif ndim == 2:
		if length == 2:
			subscripts = 'ij,ij->'
		elif length == 3:
			if len(shapes[2]) == 1:
				subscripts = 'ij,ij,j->'			
			elif len(shapes[2]) == 2:
				subscripts = 'ij,ik,jk->'			
			else:
				subscripts = 'ij,ij->'
		else:
			subscripts = 'ij,ij->'
	else:
		if length == 2:
			subscripts = '...ij,...ij->...'
		elif length == 3:
			if len(shapes[2]) == 1:
				subscripts = '...ij,...ij,j->...'
			elif len(shapes[2]) == 2:
				subscripts = '...ij,...ik,jk->...'
			else:
				subscripts = '...ij,...ij->...'
		else:
			subscripts = '...ij,...ij->...'

	if length == 2:
		shapes = (shapes[0],shapes[1])
	elif length == 3:
		shapes = (shapes[0],shapes[1],shapes[2])
	else:
		shapes = (shapes[0],shapes[1])

	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	@jit
	def func(*operands):
		out = operands[0]-operands[1]
		out = real(einsummation(out,out,*operands[2:]))
		# out = real(einsummation(out,conjugate(out),*operands[2:])/
		# 		   einsummation(operands[1],conjugate(operands[1]),*operands[2:]))
		return wrapper(out,*operands)

	if isarray:
		out = func(*operands)
	else:
		out = func

	return out

def gradient_mse(*operands,optimize=True,wrapper=None):
	'''
	Calculate gradient of square inner product of arrays
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)
	length = len([shape for shape in shapes if shape is not None])

	if ndim == 1:
		if length == 3:
			subscripts = '...i,i->'
		elif length == 4:
			if len(shapes[2]) == 1:
				subscripts = '...i,i,i->'
			elif len(shapes[2]) == 2:
				subscripts = '...i,j,ij->'			
			else:
				subscripts = '...i,i->'
		else:
			subscripts = '...i,i->'
	elif ndim == 2:
		if length == 3:
			subscripts = '...ij,ij->'
		elif length == 4:
			if len(shapes[2]) == 1:
				subscripts = '...ij,ij,j->'			
			elif len(shapes[2]) == 2:
				subscripts = '...ij,ik,jk->'			
			else:
				subscripts = '...ij,ij->'
		else:
			subscripts = '...ij,ij->'
	else:
		if length == 3:
			subscripts = '...ij,...ij->...'
		elif length == 4:
			if len(shapes[2]) == 1:
				subscripts = '...ij,...ij,j->...'
			elif len(shapes[2]) == 2:
				subscripts = '...ij,...ik,jk->...'
			else:
				subscripts = '...ij,...ij->...'
		else:
			subscripts = '...ij,...ij->...'

	if length == 3:
		shapes = (shapes[2],shapes[1])
	elif length == 4:
		shapes = (shapes[3],shapes[1],shapes[2])
	else:
		shapes = (shapes[2],shapes[1])

	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	if length == 3:
		@jit
		def func(*operands):
			out = operands[0]-operands[1]
			out = 2*real(einsummation(operands[2],conjugate(out))/
				   einsummation(operands[1],conjugate(operands[1]))
				)
			return wrapper(out,*operands)
	elif length == 4:
		@jit
		def func(*operands):
			out = operands[0]-operands[1]			
			out = 2*real(einsummation(operands[3],conjugate(out),operands[2])/
				   einsummation(operands[1],conjugate(operands[1]))
				   )
			return wrapper(out,*operands)
	else:
		@jit
		def func(*operands):
			out = operands[0]-operands[1]
			out = 2*real(einsummation(operands[2],conjugate(out))/
				   einsummation(operands[1],conjugate(operands[1])))
			return wrapper(out,*operands)			

	if isarray:
		out = func(*operands)
	else:
		out = func

	return out



def inner_prod(*operands,optimize=True,wrapper=None):
	'''
	Calculate inner product of arrays
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)
	length = len([shape for shape in shapes if shape is not None])

	if ndim == 1:
		if length == 2:
			subscripts = 'i,i->'
		elif length == 3:
			if len(shapes[2]) == 1:
				subscripts = 'i,i,i->'
			elif len(shapes[2]) == 2:
				subscripts = 'i,j,ij->'			
			else:
				subscripts = 'i,i->'
		else:
			subscripts = 'i,i->'
	elif ndim == 2:
		if length == 2:
			subscripts = 'ij,ij->'
		elif length == 3:
			if len(shapes[2]) == 1:
				subscripts = 'ij,ij,j->'			
			elif len(shapes[2]) == 2:
				subscripts = 'ij,ik,jk->'			
			else:
				subscripts = 'ij,ij->'
		else:
			subscripts = 'ij,ij->'
	else:
		if length == 2:
			subscripts = '...ij,...ij->...'
		elif length == 3:
			if len(shapes[2]) == 1:
				subscripts = '...ij,...ij,j->...'
			elif len(shapes[2]) == 2:
				subscripts = '...ij,...ik,jk->...'
			else:
				subscripts = '...ij,...ij->...'
		else:
			subscripts = '...ij,...ij->...'

	if length == 2:
		shapes = (shapes[0],shapes[1])
	elif length == 3:
		shapes = (shapes[0],shapes[1],shapes[2])
	else:
		shapes = (shapes[0],shapes[1])

	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	@jit
	def func(*operands):
		out = real(einsummation(*operands[:length]))
		return wrapper(out,*operands)

	if isarray:
		out = func(*operands)
	else:
		out = func

	return out

def gradient_inner_prod(*operands,optimize=True,wrapper=None):
	'''
	Calculate gradient of inner product of arrays
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)
	length = len([shape for shape in shapes if shape is not None])

	if ndim == 1:
		if length == 3:
			subscripts = '...i,i->'
		elif length == 4:
			if len(shapes[2]) == 1:
				subscripts = '...i,i,i->'
			elif len(shapes[2]) == 2:
				subscripts = '...i,j,ij->'			
			else:
				subscripts = '...i,i->'
		else:
			subscripts = '...i,i->'
	elif ndim == 2:
		if length == 3:
			subscripts = '...ij,ij->'
		elif length == 4:
			if len(shapes[2]) == 1:
				subscripts = '...ij,ij,j->'			
			elif len(shapes[2]) == 2:
				subscripts = '...ij,ik,jk->'			
			else:
				subscripts = '...ij,ij->'
		else:
			subscripts = '...ij,ij->'
	else:
		if length == 3:
			subscripts = '...ij,...ij->...'
		elif length == 4:
			if len(shapes[2]) == 1:
				subscripts = '...ij,...ij,j->...'
			elif len(shapes[2]) == 2:
				subscripts = '...ij,...ik,jk->...'
			else:
				subscripts = '...ij,...ij->...'
		else:
			subscripts = '...ij,...ij->...'

	if length == 3:
		shapes = (shapes[2],shapes[1])
	elif length == 4:
		shapes = (shapes[3],shapes[1],shapes[2])
	else:
		shapes = (shapes[2],shapes[1])


	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	if length == 3:
		@jit
		def func(*operands):
			out = real(einsummation(operands[2],operands[1]))
			return wrapper(out,*operands)
	elif length == 4:
		@jit
		def func(*operands):
			out = real(einsummation(operands[3],operands[1],operands[2]))
			return wrapper(out,*operands)
	else:
		@jit
		def func(*operands):
			out = real(einsummation(operands[2],operands[1]))
			return wrapper(out,*operands)			

	if isarray:
		out = func(*operands)
	else:
		out = func

	return out


def inner_norm(*operands,optimize=True,wrapper=None):
	'''
	Calculate norm squared of arrays a and b, with einsum if shapes supplied
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)		
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	

	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)

	if ndim == 1:
		subscripts = 'i->'
	elif ndim == 2:
		subscripts = 'ij->'
	else:
		subscripts = '...ij->...'

	shapes = (shapes[0],)

	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	@jit
	def func(*operands):
		out = einsummation(abs2(operands[0]-conjugate(operands[1])))
		return wrapper(out,*operands)
	
	if isarray:
		out = func(*operands)
	else:
		out = func

	return out

def gradient_inner_norm(*operands,optimize=True,wrapper=None):
	'''
	Calculate norm squared of arrays a and b, with einsum if shapes supplied
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)

	if ndim == 1:
		subscripts = '...i,i->...'
	elif ndim == 2:
		subscripts = '...ij,ij->...'
	else:
		subscripts = '...ij,...ij->...'

	shapes = (shapes[2],shapes[1])

	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	@jit
	def func(*operands):
		out = conjugate(operands[0])-operands[1]
		out = 2*real(einsummation(operands[2],out))
		return wrapper(out,*operands)
	
	if isarray:
		out = func(*operands)
	else:
		out = func

	return out


def inner_abs2(*operands,optimize=True,wrapper=None):
	'''
	Calculate absolute square inner product of arrays
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)

	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)

	if ndim == 1:
		subscripts = 'i,i->'
	elif ndim == 2:
		subscripts = 'ij,ij->'
	else:
		subscripts = '...ij,...ij->...'
	
	shapes = (shapes[0],shapes[1])

	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	@jit
	def func(*operands):
		out = abs2(einsummation(*operands))#/real(einsummation(operands[0],conjugate(operands[0]))*einsummation(operands[1],conjugate(operands[1])))
		return wrapper(out,*operands)
	
	if isarray:
		out = func(*operands)
	else:
		out = func

	return out



def gradient_inner_abs2(*operands,optimize=True,wrapper=None):
	'''
	Calculate gradient of absolute square of inner product of arrays
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)

	if ndim == 1:
		subscripts_func = 'i,i->'
	elif ndim == 2:
		subscripts_func = 'ij,ij->'
	else:
		subscripts_func = '...ij,...ij->...'

	shapes_func = (shapes[0],shapes[1])

	einsummation_func = einsum(subscripts_func,*shapes_func,optimize=optimize,wrapper=None)

	if ndim == 1:
		subscripts_grad = '...i,i->...'
	elif ndim == 2:
		subscripts_grad = '...ij,ij->...'
	else:
		subscripts_grad = '...ij,...ij->...'

	shapes_grad = (shapes[2],shapes[1])

	einsummation_grad = einsum(subscripts_grad,*shapes_grad,optimize=optimize,wrapper=None)

	@jit
	def func(*operands):
		out = 2*real(conjugate(einsummation_func(operands[0],operands[1]))*einsummation_grad(operands[2],operands[1]))
		return wrapper(out,*operands)

	if isarray:
		out = func(*operands)
	else:
		out = func

	return out


def inner_real(*operands,optimize=True,wrapper=None):
	'''
	Calculate real inner product of arrays
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)

	if ndim == 1:
		subscripts = 'i,i->'
	elif ndim == 2:
		subscripts = 'ij,ij->'
	else:
		subscripts = '...ij,...ij->...'

	shapes = (shapes[0],shapes[1])

	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	@jit
	def func(*operands):
		out = real(einsummation(*operands))
		return wrapper(out,*operands)

	if isarray:
		out = func(*operands)
	else:
		out = func

	return out


def gradient_inner_real(*operands,optimize=True,wrapper=None):
	'''
	Calculate gradient of real inner product of arrays
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)

	if ndim == 1:
		subscripts = '...i,i->...'
	elif ndim == 2:
		subscripts = '...ij,ij->...'
	else:
		subscripts = '...ij,...ij->...'

	shapes = (shapes[2],shapes[1])

	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	@jit
	def func(*operands):
		out = real(einsummation(operands[2],operands[1]))
		return wrapper(out,*operands)

	if isarray:
		out = func(*operands)
	else:
		out = func

	return out


def inner_imag(*operands,optimize=True,wrapper=None):
	'''
	Calculate imag inner product of arrays
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)

	if ndim == 1:
		subscripts = 'i,i->'
	elif ndim == 2:
		subscripts = 'ij,ij->'
	else:
		subscripts = '...ij,...ij->...'

	shapes = (shapes[0],shapes[1])

	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	@jit
	def func(*operands):
		out = einsummation(*operands).imag
		return wrapper(out,*operands)

	if isarray:
		out = func(*operands)
	else:
		out = func

	return out


def gradient_inner_imag(*operands,optimize=True,wrapper=None):
	'''
	Calculate gradient of imag inner product of arrays
	Args:
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)				
	Returns:
		out (callable,array): Summation, callable if shapes supplied, otherwise out array
	'''	
	isarray = all(isinstance(operand,arrays) for operand in operands)
	wrapper = jit(wrapper) if wrapper is not None else jit(nullfunc)
	
	if isarray:
		shapes = [operand.shape for operand in operands]
	else:
		shapes = [operand for operand in operands]
	
	ndim = min(len(shape) for shape in shapes if shape is not None)

	if ndim == 1:
		subscripts = '...i,i->...'
	elif ndim == 2:
		subscripts = '...ij,ij->...'
	else:
		subscripts = '...ij,...ij->...'

	shapes = (shapes[2],shapes[1])

	einsummation = einsum(subscripts,*shapes,optimize=optimize,wrapper=None)

	@jit
	def func(*operands):
		out = einsummation(operands[2],operands[1]).imag
		return wrapper(out,*operands)

	if isarray:
		out = func(*operands)
	else:
		out = func

	return out



@jit
def dot(a,b):
	'''
	Calculate dot product of arrays a and b
	Args:
		a (array): Array to calculate dot product
		b (array): Array to calculate dot product
	Returns:
		out (array): Dot product
	'''	
	return np.dot(a,b)


def dots(*a):
	'''
	Calculate dot product of arrays a and b
	Args:
		a (iterable[array]): Arrays to calculate dot product
	Returns:
		out (array): Dot product
	'''	

	if not len(a):
		out = None
	else:
		out = a[0]
		for i in a[1:]:
			out = dot(out,i)
	
	return out
	

@jit
def inner(a,b):
	'''
	Calculate inner product of arrays a and b
	Args:
		a (array): Array to calculate inner product
		b (array): Array to calculate inner product
	Returns:
		out (array): Inner product
	'''	
	return np.inner(dagger(a),b)

@jit
def outer(a,b):
	'''
	Calculate outer product of arrays a and b
	Args:
		a (array): Array to calculate outer product
		b (array): Array to calculate outer product
	Returns:
		out (array): Outer product
	'''	
	return np.outer(a,dagger(b))


@jit
def multiply(a,b):
	'''
	Multiply arrays elementwise
	Args:
		a (array): Array to multiply
		b (array): Array to multiply
	Returns:
		out (array): Elementwise multiplication of arrays
	'''
	return np.multiply(a,b)

@jit
def divide(a,b):
	'''
	divide arrays elementwise
	Args:
		a (array): Array to divide
		b (array): Array to divide
	Returns:
		out (array): Elementwise division of arrays
	'''
	return np.divide(a,b)


@jit
def multiplication(a):
	'''
	Multiply list of arrays elementwise
	Args:
		a(array): Arrays to multiply
	Returns:
		out (array): Elementwise multiplication of arrays
	'''
	out = a[0]

	sparse = is_sparse(out)

	for b in a[1:]:
		if not sparse:
			out = out*b
		else:
			out = out.multiply(b)
	
	return out

@jit
def add(a,b):
	'''
	Add arrays elementwise
	Args:
		a (array): Array to add
		b (array): Array to add
	Returns:
		out (ndarray): Elementwise addition of arrays
	'''
	return np.add(a,b)


@jit
def _addition(a,b):
	'''
	Add arrays elementwise
	Args:
		a (array): Array to add
		b (array): Array to add
	Returns:
		out (ndarray): Elementwise addition of arrays
	'''
	return np.add(a,b)

@jit
def addition(a,axis=None):
	'''
	Add list of arrays elementwise
	Args:
		a (iterable): Arrays to add
		axis (int): axis to perform addition		
	Returns:
		out (ndarray) if out argument is not None
	'''
	# return forloop(1,len(a),lambda i,out: _add(out,a[i]),a[0])
	return np.sum(a,axis=axis)

def product(a):
	'''
	Get product of elements in iterable
	Args:
		a (iterable): Array to compute product of elements
	Returns:
		out (array): Reduced array of product of elements
	'''
	try:
		out = onp.prod(a)
	except:
		out = 0
	return out


def where(conditions,x=None,y=None):
	'''
	Indices where conditions are True
	Args:
		conditions (array): conditions
	Returns:
		out (array): Indices of conditions
	'''
	if x is not None or y is not None:
		return np.where(conditions,x,y)
	else:
		return np.where(conditions)

def conditions(booleans,op):
	'''
	Compute multiple conditions with boolean operator
	Args:
		booleans (iterable[bool]): Boolean conditions
		op (str,iterable[str]): Boolean operators, ['and','or','lt','gt','eq','le','ge','ne','in'] 
	Returns:
		out (bool): Boolean of conditions
	'''

	updates = {
		'&':'and','&&':'and',
		'|':'or','||':'or',
		'=':'eq','==':'eq',
		'<':'lt','<=':'le',
		'>':'gt','ge':'ge',
		'!=':'ne',
		'isin':'in'
		}

	funcs = {
		**{op: getattr(operator,'__%s__'%(op)) for op in ['and','or','lt','gt','eq','le','ge','ne']},
		**{op: lambda a,b: onp.isin(a,b) for op in ['in']},
		**{None: lambda a,b:a},
	}

	if isinstance(op,str):
		ops = [op]*len(booleans)
	else:
		ops = op

	ops = [updates.get(op,op) for op in ops]

	op = ops[0] if ops else None
	
	if op is None:
		out = None
	elif op in ['or']:
		out = False
	elif op in ['and','lt','gt','eq','le','ge','ne']:
		out = True
	elif op in ['in']:		
		out = booleans[0]
		ops = ops[1:]
		booleans = booleans[1:]

	for op,boolean in zip(ops,booleans):
		if boolean is None:
			continue
		
		func = funcs.get(op,funcs[None])
		
		out = func(out,boolean)
	
	return out



@jit
def matmul(a,b,dtype=None):
	'''
	Calculate matrix product arrays a and b
	Args:
		a (array): Array to calculate matrix product
		b (array): Array to calculate matrix product
		dtype (datatype): Datatype of array		
	Returns:
		out (array): Matrix product
	'''	
	return np.matmul(a,b,dtype=dtype)

@jit
def multi_matmul(a,dtype=None):
	'''
	Get matrix product of iterable of arrays [a_i], where a_{i}.shape[-1] == a_{i+1}.shape[0]
	Args:
		a (iterable): Arrays to compute product of arrays
		dtype (datatype): Datatype of array		
	Returns:
		out (array): Reduced array of matrix product of arrays
	'''
	return forloop(1,len(a),lambda i,out: matmul(out,a[i],dtype=dtype),a[0])

@jit
def multi_dot(a,dtype=None):
	'''
	Get matrix product of iterable of arrays [a_i], where a_{i}.shape[-1] == a_{i+1}.shape[0] with optimized ordering of products
	Args:
		a (iterable): Arrays to compute product of arrays
		dtype (datatype): Datatype of array		
	Returns:
		out (array): Reduced array of matrix product of arrays
	'''
	return np.linalg.multi_dot(a,dtype=dtype)

@partial(jit,static_argnums=(1,))
def average(a,axis=0,weights=None):
	'''
	Get average of elements in array along axis
	Args:
		a (array): Array to compute product of elements
		axis (int): axis to perform product
		weights (array): Weights to compute average with respect to
	Returns:
		out (array): Reduced array of average of elements along axis
	'''
	return np.average(a,axis,weights)



@partial(jit,static_argnums=(2,))
def vtensordot(a,b,axis=0):
	'''
	Tensordot product of arrays a and b
	Args:
		a (array): Array to multiply
		b (array): Array to multiply
		axis (int,iterable): axis to perform tensordot 
	Returns:
		out (array): Dot product of array along axis
	'''
	return vmap(lambda a: tensordot(a,b,axis))(a)

@partial(jit,static_argnums=(2,))
def tensordot(a,b,axis=0):
	'''
	Tensordot product of arrays a and b
	Args:
		a (array): Array to multiply
		b (array): Array to multiply
		axis (int,iterable): axis to perform tensordot 
	Returns:
		out (array): Dot product of array along axis
	'''
	return np.tensordot(a,b,axis)


@jit
def _tensorprod(a,b):
	'''
	Tensor (kronecker) product of arrays a and b	
	Args:
		a (array): Array to tensorproduct
		b (array): Array to tensorproduct
	Returns:
		out (array): Tensorproduct product of arrays
	'''
	return np.kron(a,b)


@jit
def tensorprod(a):
	'''
	Tensor (kronecker) product of arrays a
	Args:
		a (iterable): Array to perform kronecker product
	Returns:
		out (array): Kronecker product of array
	'''
	out = a[0]
	for i in range(1,len(a)):
		out = _tensorprod(out,a[i])
	return out
	# return forloop(1,len(a),lambda i,out: _tensorprod(out,a[i]),a[0])	

@jit
def vtensorprod(a):
	'''
	Tensor (kronecker) product of arrays a
	Args:
		a (iterable): Array to perform kronecker product
	Returns:
		out (array): Kronecker product of array
	'''
	return vmap(tensorprod)(a)



@jit
def ntensorprod(a,n):
	'''
	Tensor (kronecker) product of arrays a
	Args:
		a (iterable): Array to perform kronecker product
		n (int): Perform kronecker product n times
	Returns:
		out (array): Kronecker product of array
	'''
	out = a
	for i in range(1,n):
		out = _tensorpod(out,a)
	return out
	# return forloop(1,n,lambda i,out: _tensorprod(out,a),a)	

@jit
def vntensorprod(a,n):
	'''
	Tensor (kronecker) product of arrays a
	Args:
		a (iterable): Array to perform kronecker product
		n (int): Perform kronecker product n times
	Returns:
		out (array): Kronecker product of array
	'''
	return vmap(lambda a: ntensorprod(a,n))(a)


def einsum(subscripts,*operands,optimize=True,wrapper=None):
	'''
	Get optimal summation of axis in array denoted by subscripts
	Args:
		subscripts (str): operations to perform for summation
		operands (iterable[iterable[int],array]): Shapes of arrays or arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type	
		wrapper (callable): Wrapper for einsum with signature wrapper(out,*operands)	
	Returns:
		einsummation (callable,array): Optimal einsum operator or array of optimal einsum
	'''

	noperands = subscripts.count(',')+1
	operands = operands[:noperands]

	isarray = all(isinstance(operand,arrays) for operand in operands)

	if wrapper is not None:
		wrapper = jit(wrapper)

	if isarray:
		shapes = [tuple(operand.shape) for operand in operands]
	else:
		shapes = [tuple(operand) for operand in operands if operand is not None]

	optimize = einsum_path(subscripts,*shapes,optimize=optimize)	

	if wrapper is not None:
		@jit
		def einsummation(*operands,subscripts=subscripts,optimize=optimize,wrapper=wrapper):
			return wrapper(np.einsum(subscripts,*operands,optimize=optimize),*operands)
	else:
		@jit
		def einsummation(*operands,subscripts=subscripts,optimize=optimize):
			return np.einsum(subscripts,*operands,optimize=optimize)

	if isarray:
		return einsummation(*operands)
	else:
		return einsummation


def einsum_path(subscripts,*shapes,optimize=True):
	'''
	Get optimal summation path of axis of shapes
	Args:
		subscripts (str): operations to perform for summation	
		shapes (iterable): Shapes of arrays to compute summation of elements
		optimize (bool,str,iterable): Contraction type
	Returns:
		optimize (list[tuple[int]]): Optimal einsum path of shapes
	'''

	optimizers = {True:'optimal',False:'auto',None:'optimal'}
	optimize = optimizers.get(optimize,optimize)

	operands = (zeros(shape) for shape in shapes)

	optimize,string = np.einsum_path(subscripts,*operands,optimize=optimize)

	return optimize



@jit
def distance(a,b):
	'''
	Calculate distance between two objects a,b
	Args:
		a (array): Object a
		b (array): Object b
	Returns:
		out (array): Distance between objects a,b
	'''	
	return norm(a-b,ord=2)


if backend in ['jax','jax.autograd']:

	def slicing(a,start,size):
		'''
		Get slice of array
		Args:
			a (array): Array to be sliced along first dimension
			start (int): Start index to slice
			size (int): Length of slice
		Returns:
			a (array): Sliced array
		'''

		# TODO merge slicing for different numpy backends (jax vs autograd)

		return jax.lax.dynamic_slice(a,(start,*[0]*(a.ndim-1),),(size,*a.shape[1:]))
		# return a[start:start+size]

elif backend in ['autograd','numpy']:

	def slicing(a,start,size):
		'''
		Get slice of array
		Args:
			a (array): Array to be sliced along first dimension
			start (int): Start index to slice
			size (int): Length of slice
		Returns:
			a (array): Sliced array
		'''

		# TODO merge slicing for different numpy backends (jax vs autograd)

		# return jax.lax.dynamic_slice(a,(start,*[0]*(a.ndim-1),),(size,*a.shape[1:]))
		return a[start:start+size]		




def slice_size(*slices):
	'''
	Get length of merged slices
	Args:
		slice (slice): Slice to size
	Returns:
		size (int): Length of merged slices
	'''
	slices = slice_merge(*slices)
	size = (slices.stop-slices.start)//(slices.step if slices.step is not None else 1)
	return size	

def slice_merge(*slices):
	'''
	Merge slices
	Args:
		slices (slice): Slices to merge
	Returns:
		slices (slice): Merged slice
	'''
	start = [s.start for s in slices if s is not None]
	stop = [s.stop for s in slices if s is not None]
	step = [s.step for s in slices if s is not None]

	assert len(set(step)) <= 1, "All step sizes must be equal for merged slices"

	if start == []:
		start = None
	else:
		start = min(start)

	if stop == []:
		stop = None
	else:
		stop = sum(stop)

	if step == []:
		step = None
	else:
		step = min(step)

	slices = slice(start,stop,step)

	return slices


def slice_slice(*slices,index=None):
	'''
	Get new slice indices of slices[index] within merged slices
	Args:
		slices (slice): Slices to merge
		index (int): Index of slices to re-index within merged slices
	Returns:
		slices (slice,iterable[slice]): Re-indexed slice of slices[index], or all re-indexed slices if index is None
	'''

	isint = isinstance(index,integers)

	if index is None:
		index = range(len(slices))
	elif isint:
		index = [index]

	length = len(slices)
	merged = slice_merge(*slices)
	size = slice_size(merged)

	sliced = []
	for i in index:
		if i == 0:
			sliced.append(slices[i])
		else:
			submerged = slice_merge(*slices[:i])
			
			start = submerged.stop + slices[i].start
			stop = submerged.stop + slices[i].stop
			step = slices[i].step
			
			subsliced = slice(start,stop,step)
			
			sliced.append(subsliced)

	if isint:
		slices = sliced[0]
	else:
		slices = sliced

	return slices


def nonzero(a,axis=None,eps=None):
	'''
	Count non-zero elements of array, with eps tolerance
	Args:
		a (array): Array to count non-zero elements
		axis (int,iterable[int]): Axis to compute non-zero elements
		eps (scalar): Epsilon tolerance, defaults to epsilon precision of array dtype
	Returns:
		n (int): Number of non-zero entries
	'''
	eps = epsilon(a.dtype,eps=eps) if eps is None or isinstance(eps,integers) else eps
	n = np.count_nonzero(abs(a)>=eps,axis=axis)
	return n

def _len_(obj):
	'''
	Get length of object
	Args:
		obj (object): Object to be sized
	Returns:
		length (int): Length of object
	'''
	try:
		length = len(obj)
	except:
		if isinstance(obj,slice):
			start = obj.start
			stop = obj.stop
			step = obj.step
			
			assert all(s is not None for s in [start,stop]), "Slice must have static start and stop to be sized"
			step = 1 if step is None else step

			length = (stop-start)//step
		else:
			length = len(obj)
	
	return length


def _max_(obj):
	'''
	Get maximum of object
	Args:
		obj (object): Object to be maximized
	Returns:
		maximum (int): Maximum of object
	'''
	try:
		maximum = max(obj)
	except:
		if isinstance(obj,slice):
			start = obj.start
			stop = obj.stop
			step = obj.step
			
			assert all(s is not None for s in [start,stop]), "Slice must have static start and stop to be sized"

			step = 1 if step is None else step

			maximum = stop-step
		else:
			maximum = max(obj)
	
	return maximum


def _min_(obj):
	'''
	Get minimum of object
	Args:
		obj (object): Object to be minimized
	Returns:
		minimum (int): Minimum of object
	'''
	try:
		minimum = min(obj)
	except:
		if isinstance(obj,slice):
			start = obj.start
			stop = obj.stop
			step = obj.step
			
			assert all(s is not None for s in [start,stop]), "Slice must have static start and stop to be sized"
			step = 1 if step is None else step

			minimum = start
		else:
			minimum = min(obj)
	
	return minimum


def _iter_(obj):
	'''
	Get iterator of object
	Args:
		obj (object): Object to be iterated
	Returns:
		iterator (int): Iterator of object
	'''
	try:
		iterator = obj.__iter__()
		iterator = obj
	except:
		if isinstance(obj,slice):
			start = obj.start
			stop = obj.stop
			step = obj.step
			
			assert all(s is not None for s in [start,stop]), "Slice must have static start and stop to be sized"
			step = 1 if step is None else step

			iterator = range(start,stop,step)
		else:
			iterator = obj
	
	return iterator



def powerset(iterable,probability=1):
	'''
	Get power set of iterable
	Args:
		iterable (iterable): iterable to get powerset of
		probability (float): probability of returning element from power set
	Yields:
		iteration (iterable): iteration of powerset of iterable
	'''

	for number,iteration in enumerate(itertools.product(*[[[], [i]] for i in iterable])):
		iteration = (j for i in iteration for j in i)
		if (probability==1) or (rand() >= probability):
			yield iteration
		else:
			continue


@jit
def commutator(a,b):
	'''
	Calculate commutator of ab - ba
	Args:
		a (array): Array to calculate commutator
		b (array): Array to calculate commutator
	Returns:
		out (array): commutator
	'''	
	return tensordot(a,b,1) - tensordot(b,a,1)


@jit
def commutes(a,b):
	'''
	Calculate whether a and b commutate
	Args:
		a (array): Array to calculate commutator
		b (array): Array to calculate commutator
	Returns:
		commutes (bool): Boolean of a and b commuting
	'''	
	return bool(~(commutator(a,b).any()))

@jit
def anticommutator(a,b):
	'''
	Calculate anticommutator of ab + ba
	Args:
		a (array): Array to calculate anticommutator
		b (array): Array to calculate anticommutator
	Returns:
		out (array): anticommutator
	'''	
	return tensordot(a,b,1) + tensordot(b,a,1)


@jit
def anticommutes(a,b):
	'''
	Calculate whether a and b anticommutate
	Args:
		a (array): Array to calculate anticommutator
		b (array): Array to calculate anticommutator
	Returns:
		commutes (bool): Boolean of a and b anticommuting
	'''	
	return bool(~(anticommutator(a,b).any()))


@partial(jit,static_argnums=(1,))
def trace(a,axis=(0,1)):
	'''
	Calculate trace of array
	Args:
		a (array): Array to calculate trace
		axis (iterable): Axes to compute trace with respect to
	Returns:
		out (array): Trace of array
	'''	
	return np.trace(a,axis1=axis[0],axis2=axis[1])

@jit
def rank(a,tol=None,hermitian=False):
	'''
	Calculate rank of array
	Args:
		a (array): Array to calculate rank
		tol (float): Tolerance of rank computation
		hermitian (bool): Whether array is hermitian
	Returns:
		out (array): rank of array
	'''		
	try:
		return np.linalg.matrix_rank(a,tol=tol,hermitian=hermitian)
	except:
		return 0


def tr(obj,axis=None,shape=None,size=None):
	'''
	Calculate partial trace of object at axis, as per shape
	Args:
		obj (array): Array to compute partial trace
		axis (int,iterable[int]): axis of array to trace over, as per shape
		shape (iterable[int]): Shape to reshape array for partial trace, or power of size if size is not None
		size (int,iterable[int]): Base dimensions of each reshaped axis
	Returns:
		obj (array): Partially traced object
	'''
	
	if size is not None:
		if shape is not None:
			if isinstance(size,integers):
				size = [size for i in shape]
			shape = [s**i for s,i in zip(size,shape)]
  

	ndim = obj.ndim
	shape = shape*ndim
	
	if shape is not None:
		obj = obj.reshape(shape)

	dim = obj.ndim//ndim        
	shape = obj.shape[:dim]
		
	if axis is None:
		axis = range(dim)
	elif isinstance(axis,integers):
		axis = [axis]        
	axis = [dim+i if i<0 else i for i in axis]
	
	shape = [prod([shape[i] for i in range(dim) if i not in axis])]*ndim
	
	for i in axis:
		obj = trace(obj,axis=[j*dim+i for j in range(ndim)])

	obj = obj.reshape(shape)
		
#     subscripts = [i for i in characters[:dim*ndim]]
#     for i in axis:
#         for j in range(ndim-1,0,-1):
#             subscripts[j*dim+i] = subscripts[i]
#     subscripts = ''.join(subscripts)
			
#     obj = einsum(subscripts,obj)
	
#     obj = obj.reshape(shape)
	
	return obj

@jit
def abs(a):
	'''
	Calculate absolute value of array
	Args:
		a (array): Array to calculate absolute value
	Returns:
		out (array): Absolute value of array
	'''	
	return np.abs(a)

@jit
def abs2(a):
	'''
	Calculate absolute value squared of array
	Args:
		a (array): Array to calculate absolute value
	Returns:
		out (array): Absolute value squared of array
	'''	
	return abs(a)**2

@jit
def real(a):
	'''
	Calculate real value of array
	Args:
		a (array): Array to calculate real value
	Returns:
		out (array): Real value of array
	'''	
	return np.real(a)


@jit
def imag(a):
	'''
	Calculate imaginary value of array
	Args:
		a (array): Array to calculate imaginary value
	Returns:
		out (array): Imaginary value of array
	'''	
	return np.imag(a)


def transpose(a,axes=None):
	'''
	Calculate transpose of array a
	Args:
		a (array): Array to calculate transpose
		axes (iterable[int]): Order of axis to permute
	Returns:
		out (array): Transpose
	'''	
	return np.transpose(a,axes=axes)


@jit
def conjugate(a):
	'''
	Calculate conjugate of array a
	Args:
		a (array): Array to calculate conjugate
	Returns:
		out (array): Conjugate
	'''	
	return np.conj(a)

@jit
def dagger(a):
	'''
	Calculate conjugate transpose of array a
	Args:
		a (array): Array to calculate conjugate transpose
		conj (bool): Conjugate of array
	Returns:
		out (array): Conjugate transpose
	'''	
	return conjugate(transpose(a))

@partial(jit,static_argnums=(1,))
def sqrtm(a,hermitian=False):
	'''
	Calculate matrix square-root of array a
	Args:
		a (array): Array to compute square root
		hermitian (bool): Whether array is Hermitian						
	Returns:
		out (array): Square root of array
	'''

	eigenvalues,eigenvectors = eig(a,hermitian=hermitian,compute_v=True)

	eigenvalues = eigenvalues.astype(a.dtype) if hermitian else eigenvalues

	return dot(eigenvectors*sqrt(eigenvalues),dagger(eigenvectors))


@jit
def sqrt(a):
	'''
	Calculate square-root of array a
	Args:
		a (array): Array to compute square root
	Returns:
		out (array): Square root of array
	'''
	return np.sqrt(a)

@jit
def sqr(a):
	'''
	Calculate square of array a
	Args:
		a (array): Array to compute square root
	Returns:
		out (array): Square root of array
	'''
	return a**2

@jit
def log10(a):
	'''
	Calculate log base 10 of array a
	Args:
		a (array): Array to compute log
	Returns:
		out (array): Base 10 log of array
	'''
	return np.log10(a)


@jit
def log(a):
	'''
	Calculate natural log of array a
	Args:
		a (array): Array to compute log
	Returns:
		out (array): Natural log of array
	'''
	return np.log(a)


# @jit
def logm(a):
	'''
	Calculate matrix log of array a
	Args:
		a (array): Array to compute log
	Returns:
		out (array): Matrix log of array
	'''
	return osp.linalg.logm(a)

@jit
def exp10(a):
	'''
	Calculate element-wise base 10 exponential of array a
	Args:
		a (array): Array to compute element-wise exponential
	Returns:
		out (array): Element-wise exponential of array
	'''
	return np.power(10,a)

@jit
def exp(a):
	'''
	Calculate element-wise exponential of array a
	Args:
		a (array): Array to compute element-wise exponential
	Returns:
		out (array): Element-wise exponential of array
	'''
	return np.exp(a)

@jit
def power(a,b,dtype=None):
	'''
	Calculate power of array a
	Args:
		a (array): Array to power
		b (int): Power
		dtype (datatype): Datatype of array		
	Returns:
		out (array): Power of array
	'''
	return np.power(a,b,dtype=dtype)

@jit
def matrix_power(a,n,dtype=None):
	'''
	Calculate matrix power of array a
	Args:
		a (array): Array to power
		n (int): Power
		dtype (datatype): Datatype of array		
	Returns:
		out (array): Matrix power of array
	'''
	return np.linalg.matrix_power(a,n,dtype=dtype)


@jit
def _expm(x,A,I,n=2):
	'''
	Calculate matrix exponential of parameters times data
	Args:
		x (array): parameters of shape (1,) or (n,) or (n,n)
		A (array): Array of data to matrix exponentiate of shape (n,n)
		I (array): Array of data identity of shape (n,n)
		n (int): Number of eigenvalues of matrix
	Returns:
		out (array): Matrix exponential of A of shape (n,n)
	'''	
	return cosh(x)*I + sinh(x)*A


@jit
def expm(x,A,I):
	'''
	Calculate matrix exponential of parameters times data
	Args:
		x (array): parameters of shape (m,) or (m,n,) or (m,n,n)		
		A (array): Array of data to matrix exponentiate of shape (d,n,n)
		I (array): Array of data identity of shape (n,n)
	Returns:
		out (array): Matrix exponential of A of shape (n,n)
	'''		
	m = x.shape[0]
	d = A.shape[0]

	if m == 0:
		return I

	subscripts = 'ij,jk->ik'
	shapes = (I.shape,I.shape)
	einsummation = einsum

	def func(i,out):
		U = _expm(x[i],A[i%d],I)
		return einsummation(subscripts,U,out)

	return forloop(0,m,func,I)

@jit
def gradient_expm(x,A,I):
	'''
	Calculate gradient of matrix exponential of parameters times data
	Args:
		x (array): parameters of shape (m,) or (m,n,) or (m,n,n)		
		A (array): Array of data to matrix exponentiate of shape (d,n,n)
		I (array): Array of data identity of shape (n,n)
	Returns:
		out (array): Gradient of matrix exponential of A of shape (m,n,n)
	'''			

	m = x.shape[0]
	d = A.shape[0]

	subscripts = 'ij,jk,kl->il'
	shapes = (I.shape,I.shape,I.shape)	
	einsummation = einsum

	def grad(i):
		y = slicing(x,0,i)
		z = slicing(x,i,m-i)
		U = expm(y,A,I)
		V = expm(z,roll(A,-(i%d)),I)
		return einsummation(subscripts,V,A[i%d],U)		

	return array([grad(i) for i in range(m)])

@jit
def expmc(x,A,I,B):
	'''
	Calculate matrix exponential of parameters times data, multiplied with constant matrix
	Args:
		x (array): parameters of shape (m,) or (m,n,) or (m,n,n)		
		A (array): Array of data to matrix exponentiate of shape (d,n,n)
		I (array): Array of data identity of shape (n,n)
		B (array): Array of data to constant multiply with each matrix exponential of shape (n,n)
	Returns:
		out (array): Matrix exponential of A of shape times vector of shape (n,)
	'''		
	m = x.shape[0]
	d = A.shape[0]

	subscripts = 'ij,jk,kl->il'
	shapes = (B.shape,I.shape,I.shape)
	einsummation = einsum

	def func(i,out):
		U = _expm(x[i],A[i%d],I)
		return einsummation(subscripts,B,U,out)

	return forloop(0,m,func,I)



@jit
def expmv(x,A,I,v):
	'''
	Calculate matrix exponential of parameters times data, multiplied with vector
	Args:
		x (array): parameters of shape (m,) or (m,n,) or (m,n,n)		
		A (array): Array of data to matrix exponentiate of shape (d,n,n)
		I (array): Array of data identity of shape (n,n)
		v (array): Array of data to multiply with matrix exponentiate of shape (n,)
	Returns:
		out (array): Matrix exponential of A of shape times vector of shape (n,)
	'''		
	m = x.shape[0]
	d = A.shape[0]

	subscripts = 'ij,j->i'
	shapes = (I.shape,v.shape)
	einsummation = einsum

	def func(i,out):
		U = _expm(x[i],A[i%d],I)
		return einsummation(subscripts,U,out)

	return forloop(0,m,func,v)


@jit
def expmm(x,A,I,v):
	'''
	Calculate matrix exponential of parameters times data, multiplied with matrix
	Args:
		x (array): parameters of shape (m,) or (m,n,) or (m,n,n)		
		A (array): Array of data to matrix exponentiate of shape (d,n,n)
		I (array): Array of data identity of shape (n,n)
		v (array): Array of data to multiply with matrix exponentiate of shape (n,n)
	Returns:
		out (array): Matrix exponential of A of shape times vector of shape (n,)
	'''		
	m = x.shape[0]
	d = A.shape[0]

	subscripts = 'ij,jk,lk->il'
	shapes = (I.shape,v.shape,I.shape)
	einsummation = einsum

	def func(i,out):
		U = _expm(x[i],A[i%d],I)
		return einsummation(subscripts,U,out,conjugate(U))

	return forloop(0,m,func,v)


@jit
def expmvc(x,A,I,v,B):
	'''
	Calculate matrix exponential of parameters times data, multiplied with vector, and constant matrix
	Args:
		x (array): parameters of shape (m,) or (m,n,) or (m,n,n)		
		A (array): Array of data to matrix exponentiate of shape (d,n,n)
		I (array): Array of data identity of shape (n,n)
		v (array): Array of data to multiply with matrix exponentiate of shape (n,)
		B (array): Array of data to constant multiply with each matrix exponential of shape (n,n)
	Returns:
		out (array): Matrix exponential of A of shape times vector of shape (n,)
	'''		
	m = x.shape[0]
	d = A.shape[0]

	subscripts = 'ij,jk,k->i'
	shapes = (B.shape,I.shape,v.shape)
	einsummation = einsum

	def func(i,out):
		y = slicing(x,i*d,d)
		U = expm(y,A,I)
		return einsummation(subscripts,B,U,out)		

	return forloop(0,m//d,func,v)


@jit
def expmmc(x,A,I,v,B):
	'''
	Calculate matrix exponential of parameters times data, multiplied with matrix, and constant matrix
	Args:
		x (array): parameters of shape (m,) or (m,n,) or (m,n,n)		
		A (array): Array of data to matrix exponentiate of shape (d,n,n)
		I (array): Array of data identity of shape (n,n)
		v (array): Array of data to multiply with matrix exponentiate of shape (n,n)
		B (array): Array of data to constant multiply with each matrix exponential of shape (n,n)
	Returns:
		out (array): Matrix exponential of A of shape times vector of shape (n,)
	'''		
	m = x.shape[0]
	d = A.shape[0]

	subscripts = 'ij,jk,kl,ml,nm->in'
	shapes = (B.shape,I.shape,v.shape,I.shape,B.shape)
	einsummation = einsum

	def func(i,out):
		y = slicing(x,i*d,d)
		U = expm(y,A,I)
		return einsummation(subscripts,B,U,out,conjugate(U),conjugate(B))

	return forloop(0,m//d,func,v)

@jit
def expmmn(x,A,I,v,B):
	'''
	Calculate matrix exponential of parameters times data, multiplied with matrix, and constant matrix
	Args:
		x (array): parameters of shape (m,) or (m,n,) or (m,n,n)		
		A (array): Array of data to matrix exponentiate of shape (d,n,n)
		I (array): Array of data identity of shape (n,n)
		v (array): Array of data to multiply with matrix exponentiate of shape (n,n)
		B (array): Array of data to constant multiply with each matrix exponential of shape (k,n,n)
	Returns:
		out (array): Matrix exponential of A of shape times vector of shape (n,)
	'''		
	m = x.shape[0]
	d = A.shape[0]

	subscripts = 'uij,jk,kl,ml,unm->in'
	shapes = (B.shape,I.shape,v.shape,I.shape,B.shape)
	einsummation = einsum

	def func(i,out):
		y = slicing(x,i*d,d)
		U = expm(y,A,I)
		return einsummation(subscripts,B,U,out,conjugate(U),conjugate(B))

	return forloop(0,m//d,func,v)


@jit
def expmmcn(x,A,I,v,B,C):
	'''
	Calculate matrix exponential of parameters times data, multiplied with matrix, and constant matrix
	Args:
		x (array): parameters of shape (m,) or (m,n,) or (m,n,n)		
		A (array): Array of data to matrix exponentiate of shape (d,n,n)
		I (array): Array of data identity of shape (n,n)
		v (array): Array of data to multiply with matrix exponentiate of shape (n,n)
		B (array): Array of data to constant multiply with each matrix exponential of shape (n,n)
		C (array): Array of data to constant multiply with each matrix exponential of shape (k,n,n)
	Returns:
		out (array): Matrix exponential of A of shape times vector of shape (n,)
	'''		
	m = x.shape[0]
	d = A.shape[0]

	subscripts = 'uij,jk,kl,lm,nm,on,upo->ip'
	shapes = (C.shape,B.shape,I.shape,v.shape,I.shape,B.shape,C.shape)
	einsummation = einsum

	def func(i,out):
		y = slicing(x,i*d,d)
		U = expm(y,A,I)
		return einsummation(subscripts,C,B,U,out,conjugate(U),conjugate(B),conjugate(C))

	return forloop(0,m//d,func,v)


@jit
def sign(a):
	'''
	Calculate sign of array a
	Args:
		a (array): Array to compute sign
	Returns:
		out (array): Sign of array
	'''
	return np.sign(a)


@jit
def expmat(a,dtype=None):
	'''
	Calculate matrix exponential of array a
	Args:
		a (array): Array to compute matrix exponential
		dtype (datatype): Datatype of array		
	Returns:
		out (array): Matrix exponential of array
	'''
	return sp.linalg.expm(a,dtype=dtype)


@jit
def sin(a):
	'''
	Calculate sine of array a
	Args:
		a (array): Array to compute sine
	Returns:
		out (array): Sine of array
	'''
	return np.sin(a)


@jit
def cos(a):
	'''
	Calculate cosine of array a
	Args:
		a (array): Array to compute cosine
	Returns:
		out (array): Cosine of array
	'''
	return np.cos(a)

@jit
def tan(a):
	'''
	Calculate tan of array a
	Args:
		a (array): Array to compute tan
	Returns:
		out (array): Tan of array
	'''
	return np.tan(a)

@jit
def sinh(a):
	'''
	Calculate sinh of array a
	Args:
		a (array): Array to compute sinh
	Returns:
		out (array): Sinh of array
	'''
	return np.sinh(a)


@jit
def cosh(a):
	'''
	Calculate cosinh of array a
	Args:
		a (array): Array to compute cosinh
	Returns:
		out (array): Cosinh of array
	'''
	return np.cosh(a)

@jit
def tanh(a):
	'''
	Calculate tanh of array a
	Args:
		a (array): Array to compute tanh
	Returns:
		out (array): Tanh of array
	'''
	return np.tanh(a)


@jit
def arcsin(a):
	'''
	Calculate inverse sine of array a
	Args:
		a (array): Array to compute inverse sine
	Returns:
		out (array): Inverse Sine of array
	'''
	return np.arcsin(a)


@jit
def arccos(a):
	'''
	Calculate inverse cosine of array a
	Args:
		a (array): Array to compute inverse cosine
	Returns:
		out (array): Inverse cosine of array
	'''
	return np.arccos(a)


# @partial(jit,static_argnums=(1,))
@jit
def arctan(a,b=None):
	'''
	Calculate inverse tan of array a or a/b
	Args:
		a (array): Array to compute inverse tan a or a/b
		b (array): Array to compute inverse tan a/b
	Returns:
		out (array): Inverse tan of array
	'''
	if b is None:
		return np.arctan(a)
	else:
		return np.arctan2(a,b)


@jit
def arcsinh(a):
	'''
	Calculate inverse sinh of array a
	Args:
		a (array): Array to compute inverse sinh
	Returns:
		out (array): Inverse sinh of array
	'''
	return np.arcsinh(a)


@jit
def arccosh(a):
	'''
	Calculate inverse cosinh of array a
	Args:
		a (array): Array to compute inverse cosinh
	Returns:
		out (array): Inverse cosinh of array
	'''
	return np.arccosh(a)

@jit
def arctanh(a):
	'''
	Calculate inverse tanh of array a
	Args:
		a (array): Array to compute inverse tanh
	Returns:
		out (array): Inverse tanh of array
	'''
	return np.arctanh(a)


@jit
def ceil(a):
	'''
	Calculate ceiling of array
	Args:
		a (array): Array to compute ceiling
	Returns:
		out (array): Ceiling of array
	'''
	return np.ceil(a)


@jit
def floor(a):
	'''
	Calculate floor of array
	Args:
		a (array): Array to compute floor
	Returns:
		out (array): Floor of array
	'''
	return np.floor(a)

@partial(jit,static_argnums=(1,))
def argmax(a,axis=None):
	'''
	Calculate index of maximum of array a
	Args:
		a (array): Array to compute maximum
		axis (int): Axis to compute maximum		
	Returns:
		out (int): Index of maximum of array a
	'''
	return np.argmax(a)


@partial(jit,static_argnums=(1,))
def argmin(a,axis=None):
	'''
	Calculate index of minimum of array a
	Args:
		a (array): Array to compute minimum
		axis (int): Axis to compute minimum
	Returns:
		out (int): Index of minimum of array a
	'''
	return np.argmin(a,axis=axis)


@jit
def maximum(a):
	'''
	Calculate maximum of array a
	Args:
		a (array): Array to compute maximum
	Returns:
		out (array): Maximum of array a
	'''
	return np.max(a)

@jit
def minimum(a):
	'''
	Calculate maximum of array a
	Args:
		a (array): Array to compute maximum
	Returns:
		out (array): Maximum of array a
	'''
	return np.min(a)


@jit
def maximums(a,b):
	'''
	Calculate maximum of array a and b
	Args:
		a (array): Array to compute maximum
		b (array): Array to compute maximum
	Returns:
		out (array): Maximum of array a and b
	'''
	return np.maximum(a,b)


@jit
def minimums(a,b):
	'''
	Calculate minimum of array a and b
	Args:
		a (array): Array to compute minimum
		b (array): Array to compute minimum
	Returns:
		out (array): Minimum of array a and b
	'''
	return np.minimum(a,b)

def natsort(a):
	'''
	Natural sort iterable
	Args:
		a (iterable): Iterable to sort
	Returns:
		out (array): Sorted iterable
	'''
	return natsorted(a)

@partial(jit,static_argnums=(1,))
def sort(a,axis=0):
	'''
	Sort array along axis
	Args:
		a (array): Array to sort
		axis (int): Axis to sort array
	Returns:
		out (array): Sorted array
	'''
	return np.sort(a,axis)

@partial(jit,static_argnums=(1,))
def argsort(a,axis=0):
	'''
	Argsort array along axis
	Args:
		a (array): Array to sort
		axis (int): Axis to sort array
	Returns:
		out (array): Sorted array
	'''
	return np.argsort(a,axis)


@partial(jit,static_argnums=(1,))
def concatenate(a,axis=0):
	'''
	Concatenate iterables along axis
	Args:
		a (iterable): Iterables to concatenate
		axis (int): Axis to concatenate arrays
	Returns:
		out (iterable): Concatenation row-wise
	'''
	return np.concatenate(a,axis)


@jit
def hstack(a):
	'''
	Concatenate iterables horizontally
	Args:
		a (iterable): Iterables to concatenate
	Returns:
		out (iterable): Concatenation column-wise
	'''
	return np.hstack(a)

@jit
def vstack(a):
	'''
	Concatenate iterables vertically
	Args:
		a (iterable): Iterables to concatenate
	Returns:
		out (iterable): Concatenation row-wise
	'''
	return np.vstack(a)

def roll(a,shift,axis=0):
	'''
	Shift array along axis (periodically)
	Args:
		a (array): Array to shift
		shift (int,iterable[int]): Shift along axis
		axis (int,iterable[int]): Axis to shift along
	Returns:
		out (array): Shifted array
	'''
	return np.roll(a,shift,axis=axis)


def shift(iterable,shift,axis=None):
	'''
	Shift iterable along axis (periodically)
	Args:
		iterable (iterable): Iterable to shift
		shift (int,iterable[int]): Shift along axis
		axis (int,iterable[int]): Axis to shift along
	Returns:
		out (iterable): Shifted iterable
	'''	
	shift = shift % len(iterable)
	return iterable[-shift:] + iterable[:-shift]

def cumsum(a,axis=None,dtype=None):
	'''
	Cumulative sum along axis
	Args:
		a (array): iterable to sum
		axis (int) : axis to sum
		dtype (datatype): Datatype of array		
	Returns:
		out (array): Summed iterable
	'''
	return np.cumsum(asarray(a),axis=axis,dtype=dtype)

def split(a,condition):
	'''
	Split array along axis
	Args:
		a (array): iterable to split
		condition (array): condition of where to split
	Returns:
		out (array): Split iterable
	'''
	return np.split(a,condition)

def splitter(a):
	'''
	Split array into non-adjacent elements
	Args:
		a (iterable): iterable to split
	Returns:
		out (iterable[iterable]): Split iterable
	'''

	a = array([i for i in a])

	condition = where(difference(a) != 1)[0] + 1

	a = split(a,condition)

	a = [[asscalar(j) for j in i] for i in a]

	return a

def interleaver(*iterable):
	'''
	Interleave iterables
	Args:
		iterable (iterable): iterables to interleave
	Returns:
		iterable (iterable): Interleaved iterable
	'''

	n = max((len(i) for i in iterable),default=0)

	iterable = [i[j] for j in range(n) for i in iterable if j<len(i)]

	return iterable


@jit
def difference(a,n=1,axis=-1):
	'''
	Get difference of array elements
	Args:
		a (array): Array to get difference
		n (int): Step of differences
		axis (int): Axis of differences
	'''
	return np.diff(a,n=n,axis=axis)

@jit
def diag(a):
	'''
	Get diagonal of array
	Args:
		a (array): Array to get diagonal
	Returns:
		out (array): Diagonal of array
	'''
	return np.diag(a)

@jit
def mod(a,b):
	'''
	Element wise modular division of a mod b
	Args:
		a (array): Array to compare
		b (array): Array to divide by
	Returns:
		out (array): Modular division of a mod b
	'''
	try:
		return np.mod(a,b)
	except TypeError:
		return mod(real(a),b) + 1j*mod(imag(a),b)

# @partial(jit,static_argnums=(1,))
def unique(a,axis=None):
	'''
	Find unique elements of array
	Args:
		a (array): Array to search for unique elements
		axis (int): Axis to search for unique elements
	Returns:
		out (array): Unique elements of array
	'''
	return onp.unique(a,axis=axis)

def uniqueobjs(a,axis=None):
	'''
	Find unique elements of array
	Args:
		a (array): Array to search for unique elements
		axis (int): Axis to search for unique elements
	Returns:
		out (array): Unique elements of array
	'''
	return onp.unique(a,axis=axis)


def repeat(a,repeats,axis):
	'''
	Repeat array repeats-times along axis
	Concatenate iterables row-wise
	Args:
		a (array): Array to repeat
		repeats (int): Number of times to repeat
		axis (int): Axis along which to repeat
	Returns:
		out (array): Repeated array
	'''
	return np.repeat(a,repeats,axis)


def repeats(a,repeats,axis):
	'''
	Repeat array repeats-times along axis
	Concatenate iterables row-wise
	Args:
		a (array): Array to repeat
		repeats (int,iterable[int]): Number of times to repeat along each axis
		axis (int,iterable[int]): Axes along which to repeat
	Returns:
		out (array): Repeated array
	'''
	if isinstance(repeats,integers):
		repeats = (repeats,)
	if isinstance(axis,integers):
		axis = (axis,)		
	a = expand_dims(a,range(a.ndim,max(axis)+1))
	for rep,ax in zip(repeats,axis):
		a = repeat(a,rep,ax)
	return a


def take(a,indices,axis):
	'''
	Take slices from array
	Args:
		a (array): Array to take
		indices (iterable,iterable[iterable]): Indices, or iterable of indices to slice
		axis (int,interable[int]): Axis or axis corresponding to indices to slice
	Returns:
		out (array): Sliced array
	'''
	if isinstance(axis,integers):
		axis = [axis]
		indices = [indices]

	shape = a.shape

	for axis,indices in zip(axis,indices):
		if isinstance(indices,integers):
			indices = array(range(indices))
		else:
			indices = array(_iter_(indices))
		indices = minimums(shape[axis]-1,indices)[:shape[axis]]
		a = np.take(a,indices,axis)
	return a


def put(a,values,indices,axis):
	'''
	Put array to slices array
	Args:
		a (array): Array to put
		values (array): Array to take
		indices (iterable,iterable[iterable]): Indices, or iterable of indices to slice
		axis (int,interable[int]): Axis or axis corresponding to indices to slice
	Returns:
		out (array): Put array
	'''
	if isinstance(axis,integers):
		axis = [axis]
		indices = [indices]
		values = [values]

	# TODO merge put_along_axis for different numpy backends (jax vs autograd)

	for axis,indices,values in zip(axis,indices,values):

		axis = axis % a.ndim
		indices = array(indices)

		# if values.ndim < a.ndim:
		# 	values = values.reshape(*(1,)*(axis),*values.shape,*(1,)*(a.ndim-values.ndim-axis))
		# if indices.ndim < a.ndim:
		# 	indices = indices.reshape(*(1,)*(axis),*indices.shape,*(1,)*(a.ndim-indices.ndim-axis))

		# np.put_along_axis(a,indices,values,axis=axis)

		if axis in [a.ndim-1]:
			a = inplace(a,(Ellipsis,indices),values)
		else:
			raise ValueError("Not Implemented for axis %d"%(axis))

		# Ni, M, Nk = a.shape[:axis], a.shape[axis], a.shape[axis+1:]
		# size = indices.shape[axis]

		# for i in np.ndindex(Ni):
		# 	for k in np.ndindex(Nk):
		# 		a_1d = a[i + np.s_[:,] + k]
		# 		indices_1d = indices[i + np.s_[:,] + k]
		# 		values_1d  = values[i + np.s_[:,] + k]
		# 		for j in range(size):
		# 			a_1d[indices_1d[j]] = values_1d[j]

	return a



def permutations(*iterables,repeat=None):
	'''
	Get product of permutations of iterables
	Args:
		iterables (iterable[iterables],iterable[int]): Iterables to permute, or iterable of int to get all permutations of range(int)
	Returns:
		iterables (generator[tuple]): Generator of tuples of all permutations of iterables
	'''
	
	if all(isinstance(i,integers) for i in iterables):
		iterables = (range(i) for i in iterables)
	
	if repeat is None:
		repeat = 1

	return itertools.product(*iterables,repeat=repeat)

def sortby(dictionary,key=None):
	'''
	Sort dictionary by keys
	Args:
		dictionary (dict): dictionary to be sorted
		key (object,iterable[object],callable,iterable[callable]): sort iterable by key, iterable of keys, callable, or iterable of callables, with signature key(value)
	Returns:
		dictionary (dict[key,value]): Sorted dictionary
	'''

	def parse(value):
		if isinstance(value,iterables):
			try:
				return tuple((parse(i) for i in value))
			except:
				return asscalar(value)
		else:
			return value

	def get(value,key):
		value = getattr(value,key,value.get(key)) if not callable(key) else key(value)
		return value

	if key is None:
		key = None
	elif callable(key) or not isinstance(key,iterables):
		key = lambda value,key=key: parse(get(dictionary[value],key))
	elif isinstance(key,iterables):
		key = lambda value,key=key: parse([get(dictionary[value],item) for item in key])
	else:
		key = None

	dictionary = {value: dictionary[value] for value in sorted(dictionary,key=key)}

	return dictionary

def groupby(iterable,key=None,sort=None):
	'''
	Group iterable by keys, after sorting by key or sort
	Args:
		iterable (iterable): Iterable to be grouped
		key (object,iterable[object],iterable[callable],callable): group iterable by key, iterable of keys, callable, or iterable of callables, with signature key(value)
		sort (object,iterable[object],callable,iterable[callable]): sort iterable by key, iterable of keys, callable, or iterable of callables, with signature sort(value)
	Returns:
		iterable (iterable[group]): Grouped iterable with group keys and iterable values 
	'''

	def parse(value):
		if isinstance(value,iterables):
			try:
				return tuple((parse(i) for i in value))
			except:
				return asscalar(value)
		else:
			return value

	def get(value,key):
		value = getattr(value,key,value.get(key)) if not callable(key) else key(value)
		return value

	if key is None:
		key = None
	elif callable(key) or not isinstance(key,iterables):
		key = lambda value,key=key: parse(get(value,key))
	elif isinstance(key,iterables):
		key = lambda value,key=key: parse([get(value,item) for item in key])
	else:
		key = None

	if sort is None:
		sort = None
	elif callable(sort) or not isinstance(sort,iterables):
		sort = lambda value,sort=sort: parse(get(value,sort))
	elif isinstance(sort,iterables):
		sort = lambda value,sort=sort: parse([get(value,item) for item in sort])
	else:
		sort = None

	sort = key if sort is None else sort

	iterable = sorted(iterable,key=sort)

	if key is not None:
		iterable = itertools.groupby(iterable,key=key)
	else:
		iterable = ((index,(value,)) for index,value in enumerate(iterable))

	return iterable





def convert(iterable,type=list,types=(list,),default=None):
	'''
	Convert iterable to type
	Args:
		iterable (iterable): Nested iterable to flatten
		type (type): Type of iterable to return
		types (type,iterable[type]): Types to flatten
		default (type): Default type of elements of nested iterable
	Returns:
		iterable (type): Flattened iterable
	'''

	if not isinstance(iterable,types):
		if default is not None:
			try:
				return default(iterable)
			except Exception as exception:
				return iterable
	else:
		try:
			return type([convert(i,type=type,types=types,default=default) for i in iterable])
		except TypeError:
			if default is not None:
				try:
					return default(iterable)
				except Exception as exception:
					return iterable
			else:
				return iterable

def shape(a):
	'''
	Shape of array
	Args:
		a (array): Array to shape
	Returns:
		shape (iterable[int]): Shape of array
	'''
	return np.shape(a)

def size(a):
	'''
	Size of array
	Args:
		a (array): Array to size
	Returns:
		size (int): Size of array
	'''
	return np.size(a)

def ndim(a):
	'''
	Number of dimensions of array
	Args:
		a (array): Array to dimension
	Returns:
		ndim (int): Number of dimensions of array
	'''
	return np.ndim(a)

def reshape(a,shape=-1,order='C'):
	'''
	Reshape array to shape, with ordering order
	Args:
		a (array): Array to reshape
		shape (iterable[int]): Shape
		order (str): Ordering of elements during reshaping, allowed strings in ['C','F','A']
	Returns:
		out (array): Reshaped array
	'''
	return np.reshape(a,shape,order=order)

def ravel(a,order='C'):
	'''
	Flatten array, with ordering order
	Args:
		a (array): Array to reshape
		order (str): Ordering of elements during reshaping, allowed strings in ['C','F','A']
	Returns:
		out (array): Reshaped array
	'''
	return a.ravel()

def swapaxis(a,axis1,axis2):
	'''
	Swap axis of array
	Args:
		a (array): Array to be swaped
		axis1 (int): Axis to swap
		axis2 (int): Axis to swap
	Returns:
		out (array): Array with swaped axis
	'''
	return np.swapaxis(a,axis1,axis2)

def moveaxis(a,source,destination):
	'''
	Move axis of array
	Args:
		a (array): Array to be moved
		source (int,iterable[int]): Initial axis
		destination (int,interable[int]): Final axis
	Returns:
		out (array): Array with moved axis
	'''

	return np.moveaxis(a,source,destination)


def swap(a=None,axes=None,shape=None,transform=None,execute=True):
	'''
	Split and swap, and group axis
	
	Array has shape, up to reordering of axis,
		shape: (*(dimension_ij for j in range(n)) for i in range(dim)),*(dimensions_i for i in range(dims)))
	and axes are grouped by n components
		axes: [[i,j,...],...,[k,l,...]] for i,j,k,l,... in [n]
	
	If transform: array gets split with shape to
		shape -> (*(dimension_ij for j in range(n) for i in range(dim)),*(dimensions_i for i in range(dims)))
	and grouped with axes to
		shape -> (*(prod(dimension_ij for j in axis) for axis in axes for i in range(dim) ),*(dimensions_i for i in range(dims)))
	
	If not transform, shape is assumed to be previously split and grouped, and is reverse grouped and split
	
	Example:
		shape(a) = (s,xyz,uvw) with shape = {1:(x,y,z),2:(u,v,w),0:s}, axes = ((2,0))
		parse(axes,shape):
			axes: ((2,0),(1))
			dimension: {0:(x,y,z),1:(u,v,w)}
			dimensions: {2: s}
			dim: 2
			dims: 1
			n: 2
			sort: [1,2,0]
		
		transform: (s,xyz,uvw) ->
			split.func-> (xyz,uvw,s)
			split.reshape-> (x,y,z,u,v,w,s)
		 	split.transpose-> (x,u,y,v,z,w,s)
		 	group.func-> (x,u,y,v,z,w,s)
		 	group.transpose-> (z,x,w,u,y,v,s)
		 	group.reshape-> (zx,wu,y,v,s)

		~transform: (zx,wu,y,v,s) ->
			group.reshape-> (z,x,w,u,y,v,s)
		 	group.transpose-> (x,u,y,v,z,w,s)
		 	group.func-> (x,u,y,v,z,w,s)
		 	split.transpose-> (x,y,z,u,v,w,s)
		 	split.reshape-> (xyz,uvw,s)
			split.func-> (s,xyz,uvw)
				 					 	

	Args:
		a (array): array to reshape into subspaces
		axes (iterable[iterable[int] or int]): order of n composite subspaces axis to permute and group ((i,j,...),(l,m,...),...) , i,j,l,m in [n]		
		shape (dict[int,iterable[int] or int]): shape of array of form 
				{axis_i: (dimension_ij for j in [n]) for i in [dim], axis_i: dimensions_i for i in [dims]}
				for dim composite axes of n dimensions dimension = ((dimension_ij for j in [n]) for i in [dim]), 
				where each dimension j has n components i with size dimension_ij, 
				and remaining dims non-composite axes of dimension dimensions = (dimensions_i for i in [dims]), 
				such that up to rearrangements according of axis_i ordering, the array has properties 
				shape: (*(prod(dimension_ij for j in [n]) for i in [dim]),*(dimensions_i for i in [dims]),
				size: prod(dimension)*prod(dimensions),
				ndim: len(dimension)+len(dimensions)		
		transform (bool): transformation of array options
			True,None: split and swap and group axes
			False: reverse split, and swap, and group
		execute (bool): Execute transformations or return function with precomputed axes and shapes
	Returns:
		a (array): reordered array
	'''

	def parse(axes,shape):
		'''
		Parse axes and shape
		Args:
			axes (iterable[iterable[int] or int]): order of n composite subspaces axis to permute and group ((i,j,...),(l,m,...),...) , i,j,l,m in [n]		
			shape (dict[int,iterable[int] or int]): shape of array of form 
				{axis_i: (dimension_ij for j in [n]) for i in [dim], axis_i: dimensions_i for i in [dims]}	
		Returns:
			axes (iterable[iterable[int]]): order of n composite subspaces axis to permute and group ((i,j,...),(l,m,...),...) , i,j,l,m in [n]		
			dimension (dict[int,iterable[int]]): dimension of composite axis of form {axis_i: (dimension_ij for j in [n]) for i in [dim]}
			dimensions (dict[int,int]): dimension of non-composite axis of form {axis_i: dimensions_i for i in [dims]}
			dim (int): number of axis of composite space
			dims (int): number of axis of non-composite space
			n (int): number of composite dimensions
			sort (iterable[int]): order of axis, with composite then non-composite axis
		'''

		dimension = {axis: shape[axis] for axis in shape if not isinstance(shape[axis],integers)}
		dimensions = {axis: shape[axis] for axis in shape if isinstance(shape[axis],integers)}

		dim = len(dimension)
		dims = len(dimensions)

		n = max(len(dimension[axis]) for axis in dimension)
		sort = [*[axis for axis in dimension],*[axis for axis in dimensions]]

		dimension = {i: dimension[axis] for i,axis in enumerate(dimension)}
		dimensions = {dim+i: dimensions[axis] for i,axis in enumerate(dimensions)}

		axes = [[i] if isinstance(i,integers) else [*i] for i in axes] if axes is not None else [[i] for i in range(n)]
		axes = [list(sorted(set(axis),key=lambda i: axis.index(i))) for axis in axes if axis]
		axes = [*[[i for i in axis if i in range(n)] for axis in axes],*[[i] for i in range(n) if all(i not in axis for axis in axes)]]
	
		return axes,dimension,dimensions,dim,dims,n,sort


	if execute:

		def split(a,axes,dimension,dimensions,dim,dims,n,sort):
			shape = [*[dimension[axis][i] for axis in dimension for i in range(n)],*[dimensions[axis] for axis in dimensions]]
			axes = [*[i+j*n for i in range(n) for j in range(dim)],*[i for i in range(n*dim,n*dim+dims)]]
			func = lambda a: reshape(transpose(a,sort))
			return transpose(reshape(func(a),shape),axes)

		def group(a,axes,dimension,dimensions,dim,dims,n,sort):
			shape = [*[prod(dimension[i][j] for j in axis) for axis in axes for i in range(dim)],*[dimensions[axis] for axis in dimensions]]
			axes = [*[j*dim+i for axis in axes for i in range(dim) for j in axis],*[i for i in range(n*dim,n*dim+dims)]]
			func = lambda a: a
			return reshape(transpose(func(a),axes),shape)

		def _split(a,axes,dimension,dimensions,dim,dims,n,sort):
			shape = [*[prod(dimension[axis][i] for i in range(n)) for axis in dimension ],*[dimensions[axis] for axis in dimensions]]
			axes = [*[i+j*dim for i in range(dim) for j in range(n)],*[i for i in range(n*dim,n*dim+dims)]]
			func = lambda a: transpose(a,[sort.index(i) for i in range(len(sort))])
			return func(reshape(transpose(a,axes),shape))

		def _group(a,axes,dimension,dimensions,dim,dims,n,sort):
			shape = [*[dimension[i][j] for axis in axes for i in range(dim) for j in axis],*[dimensions[axis] for axis in dimensions]]
			axes = [*[[j*dim+i for axis in axes for i in range(dim) for j in axis].index(i) for i in range(n*dim)],*[i for i in range(n*dim,n*dim+dims)]]
			func = lambda a: a
			return func(transpose(reshape(a,shape),axes))

	else:
	

		def split(obj,axes,dimension,dimensions,dim,dims,n,sort):
			shape = [*[dimension[axis][i] for axis in dimension for i in range(n)],*[dimensions[axis] for axis in dimensions]]
			axes = [*[i+j*n for i in range(n) for j in range(dim)],*[i for i in range(n*dim,n*dim+dims)]]
			func = lambda a: reshape(transpose(a,sort))
			return (lambda a,axes=axes,shape=shape,func=func,obj=obj: (transpose(reshape(func(a),shape),axes)))

		def group(obj,axes,dimension,dimensions,dim,dims,n,sort):
			shape = [*[prod(dimension[i][j] for j in axis) for axis in axes for i in range(dim)],*[dimensions[axis] for axis in dimensions]]
			axes = [*[j*dim+i for axis in axes for i in range(dim) for j in axis],*[i for i in range(n*dim,n*dim+dims)]]
			func = lambda a: a
			return (lambda a,axes=axes,shape=shape,func=func,obj=obj: (reshape(transpose(func(obj(a)),axes),shape)))

		def _split(obj,axes,dimension,dimensions,dim,dims,n,sort):
			shape = [*[prod(dimension[axis][i] for i in range(n)) for axis in dimension ],*[dimensions[axis] for axis in dimensions]]
			axes = [*[i+j*dim for i in range(dim) for j in range(n)],*[i for i in range(n*dim,n*dim+dims)]]
			func = lambda a: transpose(a,[sort.index(i) for i in range(len(sort))])
			return (lambda a,axes=axes,shape=shape,func=func,obj=obj: (func(reshape(transpose(obj(a),axes),shape))))

		def _group(obj,axes,dimension,dimensions,dim,dims,n,sort):
			shape = [*[dimension[i][j] for axis in axes for i in range(dim) for j in axis],*[dimensions[axis] for axis in dimensions]]
			axes = [*[[j*dim+i for axis in axes for i in range(dim) for j in axis].index(i) for i in range(n*dim)],*[i for i in range(n*dim,n*dim+dims)]]
			func = lambda a: a
			return (lambda a,axes=axes,shape=shape,func=func,obj=obj: (func(transpose(reshape(obj(a),shape),axes))))

	axes,dimension,dimensions,dim,dims,n,sort = parse(axes,shape)

	if transform is None or transform is True:

		return group(split(a,axes,dimension,dimensions,dim,dims,n,sort),axes,dimension,dimensions,dim,dims,n,sort)
		
	elif transform is False:
		
		return _split(_group(a,axes,dimension,dimensions,dim,dims,n,sort),axes,dimension,dimensions,dim,dims,n,sort)

def shuffle(a=None,axes=None,shape=None,execute=True):
	'''
	Shuffle axes of array of shape (d**n,)*k	
	Args:
		a (array): array to reshape into subspaces
		axes (iterable[int],iterable[iterable[int]]): (i,j,k,...), order of to be shuffled axes of array, currently (i,j,k,...) to (0,1,2,...,n-1) , i,j,k in [n]		
		shape (iterable[int]): (d,n,k), dimension of subspaces d, number of subspaces n, and number of dimensions of subspaces k, (d,...,n,k)
		execute (bool): Execute transformations or return function with precomputed axes and shapes
	Returns:
		a (array): shuffled array
	'''
	def permute(axes,shape):
		'''
		Permute axes
		Assume array is initially with composite axis ordering
			(i,j,k,..0,1,...,i-1,i+1,...,j-1,j+1,...,k-1,k+1,...)
		and will be permuted to composite axis ordering
			(0,1,...i,..,j,...,k,...)
		Args:
			axes (iterable[iterable[int] or int]): order of n composite subspaces axis to permute and group ((i,j,...),(l,m,...),...) , i,j,l,m in [n]		
			shape (dict[int,iterable[int] or int]): shape of array of form 
				{axis_i: (dimension_ij for j in [n]) for i in [dim], axis_i: dimensions_i for i in [dims]}	
		Returns:
			axes (iterable[iterable[int] or int]): order of n composite subspaces axis to permute and group ((i,j,...),(l,m,...),...) , i,j,l,m in [n]		
			shape (dict[int,iterable[int] or int]): shape of array of form 
				{axis_i: (dimension_ij for j in [n]) for i in [dim], axis_i: dimensions_i for i in [dims]}	
			_axes (iterable[iterable[int] or int]): order of n composite subspaces axis to permute and group ((i,j,...),(l,m,...),...) , i,j,l,m in [n]		
			_shape (dict[int,iterable[int] or int]): shape of array of form 
				{axis_i: (dimension_ij for j in [n]) for i in [dim], axis_i: dimensions_i for i in [dims]}	

		'''

		# TODO: Allow inverse permutations of axes for grouped axes
		assert all(isinstance(axis,integers) or len(axis) == 1 for axis in axes), "Inverse permutations not implemented for grouped axes"
		
		axes = [i for axis in axes for i in (axis if not isinstance(axis,integers) else [axis])]

		n = max(len(shape[axis]) for axis in shape if not isinstance(shape[axis],integers))

		shape = {axis: [i for i in shape[axis]] if not isinstance(shape[axis],integers) else shape[axis] for axis in shape}

		_shape = {axis: [shape[axis][axes.index(i)] if i in axes else shape[axis][len(axes)+[i for i in range(n) if i not in axes].index(i)] for i in range(n)] if not isinstance(shape[axis],integers) else shape[axis] for axis in shape}

		axes = [[axes.index(i)] if i in axes else [len(axes)+[i for i in range(n) if i not in axes].index(i)] for i in range(n)]

		_axes = [[i] for i in range(n)]		

		return axes,shape,_axes,_shape	

	axes,shape,_axes,_shape = permute(axes,shape)

	return swap(swap(a,axes=axes,shape=shape,transform=True,execute=execute),axes=_axes,shape=_shape,transform=False,execute=execute)


def broadcast_to(a,shape):
	'''
	Broadcast array to shape
	Args:
		a (array): Array to broadcast
		shape (iterable): Shape to broadcast to
	Returns:
		out (array): Broadcasted array
	'''
	return np.broadcast_to(a,shape)


def expand_dims(a,axis):
	'''
	Expand axis of array
	Args:
		a (array): Array to be expanded
		axis (int,iterable[int]): Axes to expand to
	Returns:
		out (array): Array with expanded axis
	'''
	if isinstance(axis,range):
		axis = list(axis)
	return np.expand_dims(a,axis)


def bounding(bounds,dtype=None):
	'''
	Set bounds of data
	Args:
		bounds (iterable[object]): Bounds of data
		dtype (datatype): Datatype of array		
	Returns:
		bounds (iterable[object]): Bounds of data
	'''

	if bounds is None:
		bounds = ["-inf","inf"]
	elif isinstance(bounds,scalars):
		bounds = [0,bounds]
	elif len(bounds)==0:
		bounds = ["-inf","inf"]

	bounds = [to_number(i,dtype) for i in bounds]

	return bounds


def edging(data,constants=None,dtype=None,**kwargs):
	'''
	Set edges of data
	Args:
		data (array): Array of data
		constants (dict[int,dict[int,object]]): Dictionary of axis of data with indices and values of constants to set 
		dtype (datatype): Datatype of array			
		kwargs (dict): Additional keyword arguments for data
	Returns:
		data (array): Array of data
	'''

	if constants is None or not isinstance(data,arrays) or not data.ndim:
		constants = {}
	elif not all(isinstance(constants[i],dict) for i in constants):
		axis = -1
		constants = {axis:constants}
	for axis in constants:
		indices = array([int(i) for i in constants[axis]])
		values = array([constants[axis][i] for i in constants[axis]],dtype=dtype)
		axis = int(axis)

		data = put(data,values,indices,axis=axis)

	return data



def padding(data,shape,key=None,bounds=None,random=None,dtype=None,**kwargs):
	'''
	Ensure array is shape and pad with values
	Args:
		data (array): Array to be padded
		shape (int,iterable[int]): Size or shape of array
		key (key,int): PRNG key or seed
		bounds (iterable): Bounds on array
		random (str): Type of random distribution, allowed strings in ['uniform','rand','randint','randn','constant','gaussian','normal','haar','hermitian','symmetric','zero','one','plus','minus','zeros','ones','linspace','logspace']		
		dtype (datatype): Datatype of array	
		kwargs (dict): Additional keyword arguments for padding	
	Returns:
		data (array): Padded array
	'''

	if shape is None:
		return data

	if data is None:
		data = zeros(shape,dtype=dtype)
	else:
		data = array(data,dtype=dtype)

	if isinstance(shape,integers):
		shape = [shape]

	ndim = len(shape)

	diff = max(0,ndim - data.ndim)
	reshape = data.shape

	data = data.reshape((*data.shape,*(1,)*diff))

	for axis in range(ndim-diff,ndim):
		data = repeat(data,shape[axis],axis)	

	data = take(data,shape,range(ndim))

	if random is not None:
		ax = 0
		reshape = [data.shape[axis] for axis in range(ndim)]
		diff = [shape[axis] - reshape[axis] for axis in range(ndim)]

		for axis in range(ndim-1,-1,-1):
			if diff[axis] > 0:

				reshape[axis] = diff[axis] 
				pad = rand(reshape,key=key,bounds=bounds,random=random)
				reshape[axis] = shape[axis]

				data = moveaxis(data,axis,ax)
				pad = moveaxis(pad,axis,ax)

				data = array([*data,*pad])

				data = moveaxis(data,ax,axis)	

	return data


def isinstances(obj,instances,reverse=False,args=(),kwargs={}):
	'''
	Check whether callable object return is instance
	Args:
		obj (object): Callable object
		instances (type,iterable[type]): Instances of object return
		reverse (bool): Reverse whether object is instance
		args (iterable): Positional arguments for object
		kwargs (dict): Keyword arguments for object
	Returns:
		boolean (bool): Whether object return is instance
	'''
	instances = tuple(instances) if isinstance(instances,iterables) else (instances,)

	boolean = callable(obj)
	if boolean:
		try:
			boolean = isinstance(obj(*args,**kwargs),instances)
			boolean = not reverse and boolean
		except:
			boolean = reverse
			pass
	return boolean


# @partial(jit,static_argnums=(2,3,4,))
def allclose(a,b,rtol=1e-05,atol=1e-08,equal_nan=False):
	'''
	Check if arrays a and b are all close within tolerance
	Args:
		a (array): Array to compare with
		b (array): Array to compare
		rtol (float): Relative tolerance of arrays
		atol (float): Absolute tolerance of arrays
		equal_nan (bool): Compare nan's as equal
	Returns:
		out (bool): Boolean of whether a and b are all close
	'''
	return np.allclose(a,b,rtol,atol,equal_nan)	


@partial(jit,static_argnums=(2,3,4,))
def isclose(a,b,rtol=1e-05,atol=1e-08,equal_nan=False):
	'''
	Check if arrays a and b are close within tolerance
	Args:
		a (array): Array to compare with
		b (array): Array to compare
		rtol (float): Relative tolerance of arrays
		atol (float): Absolute tolerance of arrays
		equal_nan (bool): Compare nan's as equal
	Returns:
		out (array): Boolean of whether a and b are close
	'''
	return np.isclose(a,b,rtol,atol,equal_nan)	


def is_equal(a,b,rtol=1e-05,atol=1e-08,equal_nan=False):
	'''
	Check if object a and b are equal
	Args:
		a (object): object to be tested
		b (object): object to be tested
		rtol (float): Relative tolerance of arrays
		atol (float): Absolute tolerance of arrays
		equal_nan (bool): Compare nan's as equal		
	Returns:
		out (bool): whether objects are equals
	'''
	try:
		return a is b
	except ValueError:
		return allclose(a,b,rtol=rtol,atol=atol,equal_nan=equal_nan)

def is_iterable(obj,exceptions=()):
	'''
	Check if object is iterable
	Args:
		obj (object): object to be tested
		exceptions (iterable[type]): exceptions to iterables
	Returns:
		out (bool): whether object is iterable
	'''
	return hasattr(obj,'__iter__') and not isinstance(obj,exceptions)


@jit
def is_diag(a):
	'''
	Check if array is diagonal
	Args:
		a (array): Possible diagonal array of shape (n,n)
	Returns:
		out (bool): whether object is iterable
	'''	
	n,m = a.shape
	assert n == m
	return ~a.ravel()[:-1].reshape(n-1,m+1)[:,1:].any()



def is_sparse(a,*args,**kwargs):
	'''
	Check if array is sparse
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is sparse
	'''
	return is_sparsematrix(a) or is_sparsearray(a)

def is_sparsematrix(a,*args,**kwargs):
	'''
	Check if array is sparse matrix
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is sparse matrix
	'''
	return osp.sparse.issparse(a)

def is_sparsearray(a,*args,**kwargs):
	'''
	Check if array is sparse array
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is sparse array
	'''
	return osp.sparse.issparse(a)
	# return isinstance(a,sparray.SparseArray)

def is_ndarray(a,*args,**kwargs):
	'''
	Check if array is numpy array
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is numpy array
	'''
	return isinstance(a,(onp.ndarray))

def is_dataframe(a,*args,**kwargs):
	'''
	Check if array is pandas dataframe
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is pandas dataframe
	'''
	return isinstance(a,(pd.DataFrame,))

def is_array(a,*args,**kwargs):
	'''
	Check if array is array
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is array
	'''
	return isinstance(a,np.ndarray) #isndarray(a) or is_sparse(a)

def is_scalar(a,*args,**kwargs):
	'''
	Check if array is scalar
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is scalar
	'''
	return isinstance(a,scalars) or (not isinstance(a,arrays) and not is_listtuple(a)) or (isinstance(a,arrays) and (a.ndim<1) and (a.size<2))


def is_int(a,*args,**kwargs):
	'''
	Check if object is an integer number
	Args:
		a (object): Object to be checked as int
	Returns:
		out (boolean): If object is an int
	'''
	try:
		return float(a) == int(a)
	except:
		return False

def is_float(a,*args,**kwargs):
	'''
	Check if object is a float number
	Args:
		a (object): Object to be checked as float
	Returns:
		out (boolean): If object is a float
	'''
	try:
		a = float(a)
		return True
	except:
		return False

def is_number(a,*args,**kwargs):
	'''
	Check if object is an integer float number
	Args:
		a (object): Object to be checked as number
	Returns:
		out (boolean): If object is a number
	'''
	return is_int(a,*args,**kwargs) or is_float(a,*args,**kwargs)


def is_numeric(a,*args,**kwargs):
	'''
	Check if object is numeric type
	Args:
		a (object): Object to be checked as numeric
	Returns:
		out (boolean): If object is a numeric
	'''
	dtype = getattr(a,'dtype',type(a))
	return np.issubdtype(dtype, np.number)

def is_null(a,*args,**kwargs):
	'''
	Check if object is Null
	Args:
		a (object): Object to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If object is Null
	'''
	return isinstance(a,Null)

def is_none(a,*args,**kwargs):
	'''
	Check if object is None
	Args:
		a (object): Object to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If object is None
	'''
	if a is None:
		return True
	elif isinstance(a,str) and a in ['none','None','null','Null']:
		return True
	else:
		return False

def is_naninf(a,*args,**kwargs):
	'''
	Check if array is nan or inf
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is nan or inf
	'''
	return is_nan(a,*args,**kwargs) | is_inf(a,*args,**kwargs)

def is_inf(a,*args,**kwargs):
	'''
	Check if array is inf
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is inf
	'''
	return is_numeric(a) and np.isinf(a)

def is_nan(a,*args,**kwargs):
	'''
	Check if array is nan
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is nan
	'''
	return is_numeric(a) and np.isnan(a)

def is_zero(a,*args,**kwargs):
	'''
	Check if array is zeros
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is zeros
	'''
	return allclose(a,zeros(a.shape,dtype=a.dtype))


def is_realdtype(dtype,*args,**kwargs):
	'''
	Check if dtype is real
	Args:
		dtype (datatype): Datatype to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If dtype is real
	'''
	return is_intdtype(dtype,*args,**kwargs) or isfloatdtype(dtype,*args,**kwargs)

def is_intdtype(dtype,*args,**kwargs):
	'''
	Check if dtype is integer
	Args:
		dtype (datatype): Datatype to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If dtype is integer
	'''
	return np.issubdtype(dtype, np.integer)

def is_floatdtype(dtype,*args,**kwargs):
	'''
	Check if dtype is floating
	Args:
		dtype (datatype): Datatype to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If dtype is floating
	'''
	return np.issubdtype(dtype, np.floating)

def is_complexdtype(dtype,*args,**kwargs):
	'''
	Check if dtype is complex
	Args:
		dtype (datatype): Datatype to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If dtype is complex
	'''
	return np.issubdtype(dtype, np.complexfloating)


def is_key(a,*args,**kwargs):
	'''
	Check if array is Key
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is nan
	'''
	return hasattr(a,'dtype') and jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key)


def is_hermitian(obj,*args,**kwargs):
	'''
	Check if object is hermitian
	Args:
		obj (array): Object to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If object is hermitian
	'''
	try:
		# out = cholesky(obj)
		# out = (True and not is_naninf(out).any()) or allclose(obj,dagger(obj))
		out = allclose(obj,dagger(obj))
	except:
		out = False
	return out


def is_unitary(obj,*args,**kwargs):
	'''
	Check if object is unitary
	Args:
		obj (array): Object to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If object is unitary
	'''
	try:
		if obj.ndim == 1:
			out = allclose(ones(1,dtype=obj.dtype),dot(obj,dagger(obj)))
		else:
			out = allclose(identity(obj.shape,dtype=obj.dtype),dot(obj,dagger(obj)))
	except:
		out = False
	return out

def is_list(a,*args,**kwargs):
	'''
	Check if array is list
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is list
	'''
	return isinstance(a,(list))

def is_tuple(a,*args,**kwargs):
	'''
	Check if array is tuple
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is tuple
	'''
	return isinstance(a,(tuple))

def is_listtuple(a,*args,**kwargs):
	'''
	Check if array is list or tuple
	Args:
		a (array): Array to check
		args (tuple): Additional arguments
		kwargs (dict): Additional keyword arguments
	Returns:
		out (bool): If array is list or tuple
	'''
	return is_list(a) or is_tuple(a)

def parse(string,dtype):
	'''
	Parse string as numerical type
	Args:
		string (str): String to aprse
		dtype (callable): Type to convert to
	Returns:
		parsed (object): Converted string
	'''
	parsed = string
	try:
		parsed = dtype(string)
	except:
		pass
	return parsed


def returnargs(returns):
	'''
	Return tuple of returns, returning first element if tuple and length 1 iterable
	Args:
		returns (iterable): Iterable of return values
	Returns:
		returns (iterable,object): Return values
	'''

	if isinstance(returns,tuple) and len(returns) == 1:
		return returns[0]
	else:
		return returns


def generator(stop=None):
	'''
	Generator wrapper to restart stop number of times
	'''
	def wrap(func):
		def set(*args,**kwargs):
			return func(*args,*kwargs)
		@wraps(func)
		def wrapper(*args,stop=stop,**kwargs):
			generator = set(*args,**kwargs)
			while stop:
				try:
					yield next(generator)
				except StopIteration:
					stop -= 1
					generator = set(*args,**kwargs)
					yield next(generator)
		return wrapper
	return wrap


def relsort(iterable,relative):
	'''
	Sort iterable relative to other iterable
	Args:
		iterable (iterable): iterable to sort
		relative (iterable): relative iterable
	Returns:
		sort (iterable): sorted iterable
	'''

	key = lambda item: list(relative).index(item) if item in relative else len(relative) + list(iterable).index(item)
	sort = natsorted(iterable,key=key)

	if isinstance(iterable,dict):
		sort = {item: iterable[item] for item in sort}
	else:
		sort = type(iterable)(sort)

	return sort


def union(*iterables,sort=False):
	'''
	Get union of elements in iterables
	Args:
		iterables (iterable[iterable]): Iterables
		sort (bool): Sort elements as per order in iterables, or naturalsort if None
	Returns:
		union (set,list): Union of iterables, set if unsorted else list
	'''

	union = set().union(*iterables)

	if sort is None:
		iterables = tuple((tuple(natsorted(tuple(iterable)))
				for iterable in iterables))
		n = max(len(iterable) for iterable in iterables)
		key = lambda i: min(iterable.index(i) if i in iterable else n
				for iterable in iterables)
		union = natsorted(union,key=key,reverse=False)
	elif sort is True:
		union = sorted(union,key=lambda i: tuple(list(iterable).index(i) for iterable in iterables))

	return union

def intersection(*iterables,sort=False):
	'''
	Get intersection of elements in iterables
	Args:
		iterables (iterable[iterable]): Iterables
		sort (bool): Sort elements as per order in iterables, or naturalsort if None
	Returns:
		intersection (set,list): Intersection of iterables, set if unsorted else list
	'''
	intersection = union(*iterables)

	for iterable in iterables:
		intersection = intersection.intersection(set(iterable))

	if sort is None:
		iterables = tuple((tuple(natsorted(tuple(iterable)))
				for iterable in iterables))
		n = max(len(iterable) for iterable in iterables)
		key = lambda i: min(iterable.index(i) if i in iterable else n
				for iterable in iterables)
		intersection = natsorted(intersection,key=key,reverse=False)
	elif sort is True:
		intersection = sorted(intersection,key=lambda i: tuple(list(iterable).index(i) for iterable in iterables))

	return intersection

def exclusion(*iterables,sort=False):
	'''
	Get exclusion of elements in iterables
	Args:
		iterables (iterable[iterable]): Iterables
		sort (bool): Sort elements as per order in iterables, or naturalsort if None
	Returns:
		exclusion (set,list): Exclusion of iterables, set if unsorted else list
	'''

	exclusion = union(*iterables) - intersection(*iterables)

	if sort is None:
		iterables = tuple((tuple(natsorted(tuple(iterable)))
				for iterable in iterables))
		n = max(len(iterable) for iterable in iterables)
		key = lambda i: min(iterable.index(i) if i in iterable else n
				for iterable in iterables)
		exclusion = natsorted(exclusion,key=key,reverse=False)
	elif sort is True:
		exclusion = sorted(exclusion,key=lambda i: tuple(list(iterable).index(i) for iterable in iterables))

	return exclusion


def accumulate(iterable):
	'''
	Cumulative summation of iterable
	Args:
		iterable (iterable): iterable to sum of length size
	Returns:
		accumulation (iterable): cumulative summation of elements of length size
	'''
	return itertools.accumulate(iterable)

def copier(key,value,_copy):
	'''
	Copy value based on associated key 

	Args:
		key (string): key associated with value to be copied
		value (python object): data to be copied
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
	Returns:
		Copy of value
	'''

	# Check if _copy is a dictionary and key is in _copy and is True to copy value
	if ((not _copy) or (isinstance(_copy,dict) and (not _copy.get(key)))):
		return value
	else:
		return copy(value)

def permute(dictionary,_copy=False,_groups=None,_ordered=True):
	'''
	Get all combinations of values of dictionary of lists

	Args:
		dictionary (dict): dictionary of keys with lists of values to be combined in all combinations across lists
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		_groups (list,None): List of lists of groups of keys that should not have their values permuted in all combinations, but should be combined in sequence element wise. For example groups = [[key0,key1]], where dictionary[key0] = [value_00,value_01,value_02],dictionary[key1] = [value_10,value_11,value_12], then the permuted dictionary will have key0 and key1 keys with only pairwise values of [{key0:value_00,key1:value_10},{key0:value_01,key1:value_11},{key0:value_02,key1:value_12}].
		_ordered (bool): Boolean on whether to return dictionaries with same ordering of keys as dictionary

	Returns:
		List of dictionaries with all combinations of lists of values in dictionary
	'''		
	def indexer(keys,values,_groups):
		'''
		Get lists of values for each group of keys in _groups
		'''
		_groups = copy(_groups)
		if _groups is not None:
			inds = [[keys.index(k) for k in g if k in keys] for g in _groups]
		else:
			inds = []
			_groups = []
		N = len(_groups)
		_groups.extend([[k] for k in keys if all([k not in g for g in _groups])])
		inds.extend([[keys.index(k) for k in g if k in keys] for g in _groups[N:]])
		values = [[values[j] for j in i ] for i in inds]
		return _groups,values

	def zipper(keys,values,_copy): 
		'''
		Get list of dictionaries with keys, based on list of lists in values, retaining ordering in case of grouped values
		'''
		return [{k:copier(k,u,_copy) for k,u in zip(keys,v)} for v in zip(*values)]

	def unzipper(dictionary):
		'''
		Zip keys of dictionary of list, and values of dictionary as list
		'''
		keys, values = zip(*dictionary.items())	
		return keys,values

	def permuter(dictionaries): 
		'''
		Get all list of dictionaries of all permutations of sub-dictionaries
		'''
		return [{k:d[k] for d in dicts for k in d} for dicts in itertools.product(*dictionaries)]

	def nester(keys,values):
		'''
		Get values of permuted nested dictionaries in values.
		Recurse permute until values are lists and not dictionaries.
		'''
		keys,values = list(keys),list(values)
		for i,(key,value) in enumerate(zip(keys,values)):
			if isinstance(value,dict):
				if isinstance(_groups,dict):
					_group = _groups.get(key,_group)
				else:
					_group = _groups
				values[i] = permute(value,_copy=_copy,_groups=_group)
		return keys,values


	if dictionary in [None,{}]:
		return [{}]

	# Get list of all keys from dictionary, and list of lists of values for each key
	keys,values = unzipper(dictionary)


	# Get values of permuted nested dictionaries in values
	keys,values = nester(keys,values)

	# Retain ordering of keys in dictionary
	keys_ordered = keys
	
	# Get groups of keys based on _groups and get lists of values for each group
	keys,values = indexer(keys,values,_groups)

	# Zip keys with lists of lists in values into list of dictionaries
	dictionaries = [zipper(k,v,_copy) for k,v in zip(keys,values)]


	# Get all permutations of list of dictionaries into one list of dictionaries with all keys
	dictionaries = permuter(dictionaries)


	# Retain original ordering of keys if _ordered is True
	if _ordered:
		for i,d in enumerate(dictionaries):
			dictionaries[i] = {k: dictionaries[i][k] for k in keys_ordered}
	return dictionaries


@partial(jit,static_argnums=(1,2,))
def binary(a,n,function):
	'''
	Perform binary function recursively on n-fold reduction of n arguments f(1(f(2,...f(n-1,n))
	Args:
		a (iterable): iterable of length n of arguments
		n (int): Number of reductions to perform
		function (callable): Binary function with signature function(x,y)
	Returns:
		out (object): Reduced output of binary function
	'''
	if n == 1:
		return a[0]
	elif n == 2:
		return function(a[0],a[1])
	else:
		m = n%2
		n = n//2
		if m:
			return function(a[0],
				function(binary(a[m:n+m],n,function),
				binary(a[n+m:],n,function)))
		else:
			return function(binary(a[m:n+m],n,function),
				binary(a[n+m:],n,function))



def interp(x,y,**kwargs):
	'''
	Interpolate array at new points
	Args:
		x (array): Interpolation points
		y (array): Interpolation values
		kwargs (dict): Additional keyword arguments for interpolation
			kind (int): Order of interpolation
			smooth (int,float): Smoothness of fit
			der (int): order of derivative to estimate
			bounds (iterable[object]): Bounds on points
	Returns:
		func (callable): Interpolation function with signature func(x,*args,**kwargs)
	'''	

	def _interpolate(x,y,**kwargs):
		n = len(x)
		kinds = {'linear':1,'quadratic':2,'cubic':3,'quartic':4,'quintic':5,None:3}
		kind = kwargs.get('k',kwargs.get('kind'))
		bounds = kwargs.get('bounds')
		dtype = kwargs.get('dtype')

		bounds = bounding(bounds,dtype=dtype)

		def wrapper(obj):
			return minimums(bounds[1],maximums(bounds[0],obj))

		if n == 1:
			k = None
		elif n <= kinds.get(kind):
			k = [kinds[k] for k in kinds if kinds[k]==(n-1)][-1]
		else:
			k = kinds.get(kind)

		s = kwargs.get('s',kwargs.get('smooth'))
		der = kwargs.get('der')

		if n == 1:
			_func = lambda x,y=y: onp.linspace(abs(y.min()),abs(y.max()),x.size)
			def func(x,y=y,_func=_func):
				return wrapper(_func(x))
		elif der:
			spline = osp.interpolate.splrep(x,y,k=k,s=s)
			_func = lambda x: osp.interpolate.splev(x,spline,der=der)
			def func(x,y=y,_func=_func):
				return wrapper(_func(x))
		else:
			_func = osp.interpolate.UnivariateSpline(x,y,k=k,s=s)
			def func(x,y=y,_func=_func):
				x = onp.asarray(x)
				return wrapper(_func(x))
			# func = osp.interpolate.interp1d(x,y,kind)
		return func

	return _interpolate(x,y,**kwargs)


def interpolate(x,y,_x,**kwargs):
	'''
	Interpolate array at new points
	Args:
		x (array): Interpolation points
		y (array): Interpolation values
		_x (array): New points
		kwargs (dict): Additional keyword arguments for interpolation
				kind (int): Order of interpolation
				smooth (int,float): Smoothness of fit
				der (int): order of derivative to estimate
				bounds (iterable[object]): Bounds on points
	Returns:
		out (array): Interpolated values at new points
	'''		
	is1d = (y.ndim == 1)

	if is1d:
		y = [y]

	out = array([interp(x,u,**kwargs)(_x) for u in y])

	if is1d:
		out = out[0]

	return out



# @partial(jit,static_argnums=(0,))
def piecewises(func,shape,include=None,**kwargs):
	'''
	Compute piecewise curve from func
	Args:
		func (callable,iterable[callable]): Functions to fit with signature func(x,*args,**kwargs)
		shape (iterable[int]): Piecewise parameters shape
		include (bool): Include piecewise indices of coefficients
		kwargs (dict[str,object]): Additional keyword arguments for fitting		
	Returns:
		func (callable): Piecewise function with signature func(x,*args,**kwargs)
		funcs (iterable[callable]): Piecewise functions with signature func(x,*args,**kwargs)
		indices (array): indices of coefficients for piecewise domains
	'''

	if callable(func):
		funcs = [func]
	else:
		funcs = func

	n = len(funcs)
	indices = [slice(sum(shape[:i-1]),sum(shape[:i])) for i in range(1,n+2)]

	def func(x,parameters):

		bounds,parameters = parameters[indices[0]],[parameters[index] for index in indices[1:]]
		n = len(funcs)

		func = [lambda x,parameters,i=i: funcs[i](x,parameters[i]) for i in range(n)]

		function,conditions = piecewise(func,bounds)

		return function(x,parameters)
	
	if include:
		return func,funcs,indices
	else:
		return func


def piecewise(func,bounds,**kwargs):
	'''
	Compute piecewise curve from func
	Args:
		func (iterable[callable]): Functions to fit with signature func(x,*args,**kwargs)
		bounds (iterable[object]): Bounds for piecewise domains
		kwargs (dict): Additional keyword arguments
	Returns:
		func (callable): Piecewise function with signature func(x,*args,**kwargs)
		conditions (callable): Conditions for piecewise domains with signature conditions(x) -> iterable[bool]
	'''

	if callable(func) or isinstance(func,str):
		func = [func]
	else:
		func = func

	n = len(func)

	if bounds is None and n>1:
		raise ValueError("TODO: Allow for bounds to be fit")
	elif isinstance(bounds,scalars):
		bounds = [True for i in range(n+1)]
	elif len(bounds) == (n-1):
		bounds = [*bounds,True,True]

	function = func

	def conditions(x,*args,**kwargs):
		n = len(bounds)-1
		if x.ndim > 1:
			axis,ord,r = 1,2,x.reshape(*x.shape[:1],-1)
			r = norm(r,axis=axis,ord=ord)
		else:
			r = x
		conditions = [(
			(bool(bounds[i-1])*ones(r.shape,dtype=bool) if (bounds[i-1] is None or isinstance(bounds[i-1],bool)) else r>=bounds[i-1]) & 
			(bool(bounds[i])*ones(r.shape,dtype=bool) if (bounds[i] is None or isinstance(bounds[i],bool)) else r<=bounds[i])
			)
			for i in range(n)]
		return conditions

	def func(x,*args,**kwargs):
		func = function
		return np.piecewise(x,conditions(x,*args,**kwargs),func,*args,**kwargs)

	return func,conditions


def extrema(x,y,_x=None,**kwargs):
	'''
	Get extreme points of array
	Args:
		x (array): Interpolation points
		y (array): Interpolation values
		_x (array): New points		
		kwargs (dict): Additional keyword arguments for interpolation
			kind (int): Order of interpolation
			smooth (int,float): Smoothness of fit
			der (int): order of derivative to estimate
	Returns:
		indices (array): Indices of extreme points
	'''	

	defaults = {'kind':1,'smooth':0,'der':2}

	if _x is None:
		_x = x

	kwargs.update({kwarg: kwargs.get(kwarg,defaults[kwarg]) for kwarg in defaults})

	indices = argsort(abs(interp(x,y,**kwargs)(_x)))

	return indices



@jit
def heaviside(a):
	'''
	Calculate heaviside function
	Args:
		a (array): Array to calculate heaviside
	Returns:
		out (array): Heaviside
	'''		
	return np.heaviside(a,0)


@jit
def sigmoid(a,scale=1):
	'''
	Calculate sigmoid function with scale
	Args:
		a (array): Array to calculate sigmoid
		scale (float): scale of sigmoid
	Returns:
		out (array): Sigmoid
	'''		
	return (tanh(a*scale/2)+1)/2
	# return sp.special.expit(scale*a)

@jit
def bound(a,scale=1,**kwargs):
	'''
	Bound array
	Args:
		a (array): Array to bound
		scale (float): scale of bound
		kwargs (dict): Keyword arguments for bounds
	Returns:
		out (array): Bounded array
	'''
	return 2*sigmoid(a,scale) - 1


@jit
def gradient_bound(a,scale=1,**kwargs):
	'''
	Bound gradient array
	Args:
		a (array): Array to bound
		scale (float): scale of bound
		kwargs (dict): Keyword arguments for bounds
	Returns:
		out (array): Bounded array
	'''
	return 2*gradient_sigmoid(a,scale)


@jit
def nullbound(a,scale=1,**kwargs):
	'''
	Null bound array
	Args:
		a (array): Array to bound
		scale (float): scale of bound
		kwargs (dict): Keyword arguments for bounds
	Returns:
		out (array): Bounded array
	'''
	return a

@jit
def gradient_sigmoid(a,scale=1):
	'''
	Calculate gradient of sigmoid function with scale
	Args:
		a (array): Array to calculate sigmoid
		scale (float): scale of sigmoid
	Returns:
		out (array): Gradient of sigmoid
	'''
	return scale*sigmoid(a,scale)*sigmoid(-a,scale)


def replace(iterable,elements):
	'''
	Find and replace old elements in iterable with new elements

	Args:
		iterable (iterable): Iterable to be searched
		elements (dict): New and old elements
	Returns:
		iterable (iterable): Updated iterable
	'''	

	# Recursively find where nested iterable elements exist, and replace or append in-place with replacement elements
	oldtype = type(iterable)
	newtype = list
	iterable = to_iterable(iterable,newtype)
	for old in elements:
		new = elements[old]
		try:
			for i,value in enumerate(iterable):
				if value == old:
					iterable[i] = new
				else:
					iterable[i] = replace(value,elements)
		except Exception as e:
			if iterable == old:
				iterable = new

	iterable = to_iterable(iterable,oldtype)

	return iterable


def to_eval(a,represent=True):
	'''
	Convert string to python object
	Args:
		a (str): Object to convert to python object
		represent (bool): Representation of objects		
	Returns:
		object (object): Python object representation of string
	'''
	try:
		replacements = {'nan':"'nan'"}
		for replacement in replacements:
			a = a.replace(replacement,replacements[replacement])
		a = ast.literal_eval(a) if represent else a
		try:
			replacements = {'nan':nan}
			a = replace(a,replacements)
		except:
			pass
	except Exception as e:
		pass
	return a

def to_repr(a,represent=True):
	'''
	Convert python object to string representation
	Args:
		a (object): Object to convert to string representation
		represent (bool): Representation of objects				
	Returns:
		string (str): String representation of Python object
	'''
	return repr(a) if represent else a


def to_iterable(a,dtype=None,exceptions=(str,),**kwargs):
	'''
	Convert iterable to iterable type
	Args:
		a (iterable): Iterable to convert to iterable
		dtype (datatype): Type of iterable
		exceptions (tuple[datatype]): Exception types not to update
		kwargs (dict): Additional keyword arguments
	Returns:
		out (iterable): Iterable representation of iterable
	'''
	if exceptions is not None and isinstance(a,exceptions):
		return a

	try:
		a = list(a)
	except Exception as e:
		pass
	try:
		for i,value in enumerate(a):
			a[i] = to_iterable(value,dtype=dtype,exceptions=exceptions,**kwargs)
	except Exception as e:
		pass
	try:
		a = dtype(a)
	except Exception as e:
		pass
	return a

def to_list(a,dtype=None,**kwargs):
	'''
	Convert iterable to list
	Args:
		a (iterable): Iterable to convert to list
		dtype (datatype): Datatype of number
	Returns:
		out (list): List representation of iterable
	'''
	
	try:
		return a.tolist()
	except:
		try:
			return [to_list(i,dtype=dtype,**kwargs) for i in a]
		except TypeError:
			return a

def to_tuple(a,dtype=None,**kwargs):
	'''
	Convert iterable to tuple
	Args:
		a (iterable): Iterable to convert to list
		dtype (datatype): Datatype of number
	Returns:
		out (tuple): List representation of iterable
	'''
	try:
		return tuple(to_tuple(i,dtype=dtype,**kwargs) for i in a)
	except:
		return a

def to_number(a,dtype=None,**kwargs):
	'''
	Convert object to number
	Args:
		a (int,float,str): Object to convert to number
		dtype (datatype): Datatype of number
	Returns:
		number (object): Number representation of object
	'''
	prefixes = {'-':-1}
	dtypes = {'int':int,'float':float}

	coefficient = 1
	number = a
	dtype = dtypes.get(dtype,dtype)
	if isinstance(a,str):
		for prefix in prefixes:
			if a.startswith(prefix):
				a = prefix.join(a.split(prefix)[1:])
				coefficient *= prefixes[prefix]
		if is_int(a):
			dtype = int
		elif is_float(a):
			dtype = float
		if is_number(a):
			number = coefficient*float(a)
	return number

def to_int(a,**kwargs):
	'''
	Convert object to int
	Args:
		a (object): Object to represent
		kwargs (dict): Additional keyword formatting options
	Returns:
		integer (int): int representation of object
	'''

	try:
		integer = int(a)
	except:
		integer = a
	return integer


def to_str(a,**kwargs):
	'''
	Convert object to string
	Args:
		a (object): Object to represent
		kwargs (dict): Additional keyword formatting options
	Returns:
		string (str): String representation of object
	'''

	try:
		string = str(a)
	except:
		string = a
	return string

def to_string(a,**kwargs):
	'''
	Convert array to string representation
	Args:
		a (array): Array to represent
		kwargs (dict): Additional keyword formatting options
	Returns:
		string (str): String representation of array
	'''

	if a is not None:
		string = np.array_str(a,**kwargs)#.replace('[[',' [').replace(']]','] ')
	else:
		string = None

	return string

def to_key_value(string,delimiter='=',default=None,**kwargs):
	'''
	Parse strings for specific values with key=value
	Args:
		string (str): String to parse
		delimiter (str): Delimiter separating key and value
		default (object): Default value
		kwargs (dict): Additional keyword formatting options
	Returns:
		key (str): Key of string
		value (int,float,bool,None): Value of string 
	'''
	if not isinstance(string,str):
		key = string
		value = default
	else:
		string = string.split(delimiter)
		if len(string) == 1:
			key = delimiter.join(string)
			value = default
		else:
			key = string[0]
			value = delimiter.join(string[1:])
			if is_number(value):
				value = to_number(value)
			elif is_none(value):
				value = None
			else:
				value = value
	return key,value

def to_position(index,shape):
	'''
	Convert linear index to dimensional position
	Args:
		index (int): Linear index
		shape (iterable[int]): Dimensions of positions
	Returns:
		position (iterable[int]): Dimensional positions
	'''
	position = [index//(prod(shape[i+1:]))%(shape[i]) for i in range(len(shape))]
	return position

def to_index(position,shape):
	'''
	Convert dimensional position to linear index
	Args:
		position (iterable[int]): Dimensional positions
		shape (iterable[int]): Dimensions of positions
	Returns:
		index (int): Linear index
	'''	
	index = sum((position[i]*(prod(shape[i+1:])) for i in range(len(shape))))
	return index


def scinotation(number,decimals=1,base=10,order=20,zero=True,one=False,scilimits=[-1,1],error=None,usetex=False):
	'''
	Put number into scientific notation string
	Args:
		number (str,int,float): Number to be processed
		decimals (int): Number of decimals in base part of number (including leading ones digit)
		base (int): Base of scientific notation
		order (int): Max power of number allowed for rounding
		zero (bool): Make numbers that equal 0 be the int representation
		one (bool): Make numbers that equal 1 be the int representation, otherwise ''
		scilimits (iterable[int]): Limits on where not to represent with scientific notation
		error (str,int,float): Error of number to be processed
		usetex (bool): Render string with Latex
	Returns:
		string (str): String with scientific notation format for number
	'''

	if decimals is None:
		decimals = 1

	if scilimits is None:
		scilimits = [-1,1]

	if not is_number(number):
		return str(number)

	try:
		number = int(number) if is_int(number) else float(number)
	except:
		string = number
		return string

	try:
		error = int(error) if is_int(error) else float(error)
	except:
		error = None

	maxnumber = base**order
	if number > maxnumber:
		number = number/maxnumber
		if int(number) == number:
			number = int(number)
		string = str(number)

	if error is not None and is_naninf(error):
		# error = r'$\infty$'
		error = None

	if zero and number == 0:
		string = r'%d%%s%%s%%s'%(number)

	elif is_int(number) and ((number >= base**scilimits[0]) and (number <= base**scilimits[1])):
		string = r'%s%%s%%s%%s'%(str(number))

	elif is_naninf(number):
		string = r'%s%%s%%s%%s'%(str(0))

	elif isinstance(number,(float,dbl,int,itg)):		
		string = '%0.*e'%(decimals-1,number)
		string = string.split('e')
		basechange = log(10)/log(base)
		basechange = int(basechange) if int(basechange) == basechange else basechange
		flt = string[0]
		exp = str(int(string[1])*basechange)

		if int(exp) in range(*scilimits):
			flt = '%d'%(ceil(int(flt)*base**(int(exp)))) if is_int(flt) else '%0.*f'%(decimals-1,float(flt)/(base**(-int(exp)))) if (one or (float(flt) != 1.0)) else ''
			string = r'%s%%s%%s%%s'%(flt)
		else:
			string = r'%s%s%s%%s%%s%%s'%('%0.*f'%(decimals-1,float(flt)) if (one or (float(flt) != 1.0)) else '',
				r'\cdot' if ((one or (float(flt) != 1.0)) and (int(exp)!=0)) else '',
				'%d^{%s}'%(base,exp) if (int(exp)!=0) else ''
				)
	
		if error is not None and not isinstance(error,str):
			if int(exp) in range(*scilimits):
				error = '%d'%(ceil(int(error))) if is_int(error) else '%0.*f'%(decimals-1,float(error))
			else:
				error = r'%s%s%s'%(
					'%0.*f'%(decimals-1,float(error)/(base**(int(exp)))),
					r'\cdot' if ((one or (float(flt) != 1.0)) and (int(exp)!=0)) else '',
					'%d^{%s}'%(base,exp) if (int(exp)!=0) else ''
					)

	if error is None:
		error = ''
		prefix = ''
		postfix = ''
	else:
		error = str(error)
		prefix = r'~\pm~'
		postfix = ''

	string = string%(prefix,error,postfix)

	if usetex:
		string = r'%s'%(string.replace('$',''))
	else:
		string = string.replace('$','')
	return string


def texify(string,usetex=False):
	'''
	Put string into latex format
	Args:
		string (object): Object to be processed
		usetex (bool): Render string with Latex
	Returns:
		string (str): String with latex format
	'''

	if isinstance(string,str):
		string = '\\textrm{%s}'%(string)
	else:
		string = str(string)

	string = string.replace('$','')

	if usetex:
		string = '$%s$'%(string)

	return string

def uncertainty_propagation(x,y,xerr,yerr,operation):
	'''
	Calculate uncertainty of binary operations
	Args:
		x (array): x array
		y (array): y array
		xerr (array): error in x
		yerr (array): error in y
		operation (str): Binary operation between x and y, allowed strings in ['+','-','*','/','plus','minus','times','divides']
	Returns:
		out (array): Result of binary operation
		err (array): Error of binary operation
	'''

	operations = ['+','-','*','/','plus','minus','times','divides']
	assert operation in operations, "operation: %s not in operations %r"%(operation,operations)

	if operation in ['+','plus']:
		func = lambda x,y: x+y
		error = lambda x,y: sqrt((xerr*1)**2+(yerr*1)**2)
	elif operation in ['-','minus']:
		func = lambda x,y: x-y
		error = lambda x,y: sqrt((xerr*1)**2+(yerr*-1)**2)
	elif operation in ['*','times']:
		func = lambda x,y: x*y
		error = lambda x,y: sqrt((xerr*y)**2+(yerr*x)**2)
	elif operation in ['/','divides']:
		func = lambda x,y: x+y
		error = lambda x,y: sqrt((xerr/y)**2+(yerr*x/-y**2)**2)

	out,err = func(x,y),error(x,y)

	return out,err


def padder(strings,padding=' ',delimiter=None,justification='left'):
	'''
	Pad strings to all be length of largest string (or substring within string)
	Args:
		strings (iterable[str]): Strings to pad
		padding (str): Padding of string
		delimiter (str): Delimiter to split each string and ensure each substring is equal length
		justification (str): Type of justification of string, where to pad, allowed strings in ['left','right','center']
	Returns:
		padded (iterable[str]): Padded strings
	'''
	
	def justifier(string,padding,length,justification):
		padded = '%s%s%s'
		if justification in ['left']:
			padding = padding*length
			padded = padded%('',string,padding)
		elif justification in ['right']:
			padding = padding*length
			padded = padded%(padding,string,'')
		elif justification in ['center']:
			padding = [padding*(length//2),padding*(length - padding//2)]
			padded = padded%(padding[0],string,padding[1])
		else:
			padded = string
		return padded
	
	# Default padded
	padded = strings
	
	# Check if string in strings
	if len(strings) == 0 or all(string is None for string in strings):
		return padded

	# Get delimited substrings of string in strings
	if delimiter is None:
		delimiter = ''
		strings = [[string] if string is not None else None for string in strings]
	else:
		strings = [string.split(delimiter) if string is not None else None for string in strings]
	
	# Get size: Min number of delimited substrings per string in strings
	# Get length: Max length of delimited substring per string in strings
	size = min(len(string) for string in strings if string is not None)
	length = [max(len(string[i]) for string in strings if string is not None)
			for i in range(size)]
	
	# Get justified delimited substrings based on length difference to length
	strings = [[justifier(string[i],padding,length[i] - len(string[i]),justification) for i in range(size)] if string is not None else None
				for string in strings]
	
	# Join delimited substrings per string in strings
	padded = [delimiter.join(string) if string is not None else None for string in strings]
	
	return padded


def initialize(data,shape,random=None,bounds=None,dtype=None,**kwargs):
	'''
	Initialize data
	Args:
		data (array,str): data array or random type or path to load data
		shape (iterable): shape of data
		random (str,dict,callable): random type of initialization, 
			dictionary of attributes {'interpolation' (dict): keyword arguments for interp(), 'size' (int): Number of interpolation points via shape[-1]//size}, or 
			allowed strings in ['uniform','ones','zeros','random','pad'], or 
			callable function with signature random(data,shape,bounds,random,dtype,**kwargs)
		bounds (iterable[object]): bounds of data
		dtype (str,datatype): data type of data		
		kwargs (dict): Additional keyword arguments for initialization
	Returns:
		data (array): data
	'''	

	default = None
	if data is None:
		data = default
	elif exists(data):
		data = load(data,default=default)
	elif isinstance(data,str):
		data,shape,random = default,(),data
	else:
		data = data

	if shape is None:
		shape = data.shape if data is not None else None

	if dtype is None:
		dtype = data.dtype if data is not None else None

	default = 'pad'
	shape = (shape,) if isinstance(shape,integers) else tuple(shape) if shape is not None and len(shape) else () if shape is not None else None
	random = random if random is not None else None
	bounds = bounding(bounds,dtype=dtype)
	dtype = dtype if dtype is not None else None

	if data is None and shape is None and random is None:
		data = data
		shape = shape
		random = random
	elif data is not None and shape is None and random is None:
		data = data
		shape = None
		random = default
	elif data is None and shape is not None and random is None:
		data = data
		shape = shape
		random = default
	elif data is not None and shape is not None and random is None:
		data = data
		shape = shape
		random = default		
	elif data is None and shape is None and random is not None:
		data = data
		shape = ()
		random = random
	elif data is not None and shape is None and random is not None:
		data = data
		shape = None
		random = default
	elif data is None and shape is not None and random is not None:
		data = data
		shape = shape
		random = random
	elif data is not None and shape is not None and random is not None:
		data = data
		shape = shape
		random = default

	if isinstance(random,dict):

		interpolation = random.get('interpolation',{})
		size = max(1,min(shape[-1]//2,random.get('size',1)))
		shape_interp = (*shape[:-1],shape[-1]//size+2)
		pts_interp = size*arange(shape_interp[-1])
		pts = arange(shape[-1])

		data_interp = rand(shape_interp,bounds=bounds,dtype=dtype,**kwargs)

		try:
			data = interpolate(pts_interp,data_interp,pts,bounds=bounds,**interpolation)
		except:
			data = rand(shape,bounds=bounds,dtype=dtype,**kwargs)

	elif isinstance(random,str):
		
		if random in ['uniform']:
			data = ((bounds[0]+bounds[1])/2)*ones(shape,dtype=dtype)
		
		elif random in ['ones']:
			data = ones(shape,dtype=dtype)
		
		elif random in ['zeros']:
			data = zeros(shape,dtype=dtype)

		elif random in ['pad']:
			data = padding(data,shape,random=random,bounds=bounds,dtype=dtype,**kwargs)

		else:
			data = rand(shape,bounds=bounds,random=random,dtype=dtype,**kwargs)

	elif callable(random):
		data = random(data,shape,bounds=bounds,random=random,dtype=dtype,**kwargs)

	data = edging(data,**kwargs)

	data = data.astype(dtype) if data is not None else None

	return data


def nester(iterable,keys,func=None):
	'''
	Apply function to values of nested iterable
	Args:
		iterable (dict): iterable
		keys (iterable[str]): keys to apply function to
		func (callable): function to load
	Returns:
		iterable (dict): loaded iterable
	'''
	if keys is None:
		keys = None
	elif not isinstance(keys,iterables):
		keys = [keys]

	if func is None:
		return
	
	for key in iterable:
		if keys is None or key in keys:
			iterable[key] = func(iterable[key])
		elif isinstance(iterable[key],type(iterable)):
			iterable[key] = nester(iterable[key],keys,func=func)
	
	return iterable


def projector(i,shape):
	'''
	Create projector at indices i
	Args:
		i (iterable[int],int): Indices of projector
		shape (iterable[int],int): Shape of projector
	Returns:
		projector (array): array of projector
	'''
	if isinstance(i,integers):
		i = (i,i)
	if isinstance(shape,integers):
		shape = (shape,shape)
	i = tuple(i)
	shape = tuple(shape)

	projector = zeros(shape,dtype=int)
	projector = inplace(projector,i,1)

	return projector

def bloch(state,path=None):
	'''
	Plot state on Bloch Sphere
	Args:
		state (array): States of shape (d,) or (n,d) or (n,d,d) for n, d dimensional states
		path (str,boolean): Path to save plot, or boolean to save
	Returns:
		fig (matplotlib.figure): Figure of plots
		ax (matplotlib.axes): Axes of plots
	'''
	
	def coordinates(state):
		'''
		Convert state vector to Bloch vector
		Args:
			state (array): States of shape (d,) or (n,d) or (n,d,d)  for n, d dimensional states
		Returns:
			state (array): States of shape (1,d^2-1) or (n,d^2-1) or (n,d^2-1) for n, d dimensional states
		'''

		basis = array([
			[[0, 1], [1, 0]],
			[[0, -1j], [1j, 0]],
			[[1, 0], [0, -1]]
			])

		ndim = state.ndim

		if ndim == 1:
			state = einsum('i,aij,j->a',conjugate(state),basis,state)
		elif ndim == 2:
			state = einsum('ui,aij,uj->ua',conjugate(state),basis,state)
		elif ndim == 3:
			state = einsum('uij,aij->ua',state,conjugate(basis))
		else:
			pass
		state = real(state)
		return state

	root = os.path.dirname(os.path.abspath(__file__))
	mplstyle = 'plot.mplstyle'
	mplstyle = os.path.join(root,mplstyle)

	with matplotlib.style.context(mplstyle):

		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d')
		fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

		ax.grid(False)
		ax.set_axis_off()
		ax.view_init(30, 45)
		ax.dist = 7

		x, y, z = array([[-1.5,0,0], [0,-1.5,0], [0,0,-1.5]],dtype=float)
		u, v, w = array([[3,0,0], [0,3,0], [0,0,3]],dtype=float)
		ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05, color='black', linewidth=1.2)

		ax.text(0, 0, 1.7, r'$\ket{0}$', color='black', fontsize=18)
		ax.text(0, 0, -1.9, r'$\ket{1}$', color='black', fontsize=18)
		ax.text(1.9, 0, 0, r'$\ket{+i}$', color='black', fontsize=18)
		ax.text(-1.7, 0, 0, r'$\ket{-i}$', color='black', fontsize=18)
		ax.text(0, 1.7, 0, r'$\ket{+}$', color='black', fontsize=18)
		ax.text(0,-1.9, 0, r'$\ket{-}$', color='black', fontsize=18)

		state = coordinates(state)

		ax.scatter(state[:,0], state[:,1], state[:, 2], color=getattr(plt.cm,'viridis')(0.5),s=12,alpha=0.6)

		if path:
			if not isinstance(path,str):
				path = 'bloch.pdf'
			fig.savefig(path,pad_inches=0.5)

	return fig,ax



from src.io import load,dump,exists
