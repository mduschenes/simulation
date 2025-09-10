#!/usr/bin/env python

# Import python modules
import os,sys
import traceback

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,partial,wraps,copy,debug,vmap,vfunc,switch,forloop,cond,slicing,gradient,hessian,fisher,entropy,purity,similarity,divergence
from src.utils import array,empty,identity,ones,zeros,rand,random,haar,choice,arange
from src.utils import tensor,matrix,network,mps,context
from src.utils import contraction,gradient_contraction
from src.utils import inplace,reduce,reshape,transpose,tensorprod,conjugate,dagger,einsum,einsummand,dot,dots,inner,outer,trace,norm,eig,svd,diag,inv,sqrtm,addition,product,ravel
from src.utils import maximum,minimum,argmax,argmin,nonzero,difference,unique,shift,sort,relsort,prod,product
from src.utils import real,imag,absolute,abs2,mod,sign,reciprocal,sqr,sqrt,log,log10,sin,cos,exp
from src.utils import insertion,shuffle,swap,groupby,sortby,union,intersection,accumulate,interleaver,splitter,seeder,rng
from src.utils import to_index,to_position,to_string,allclose,is_hermitian,is_unitary
from src.utils import backend,pi,e,nan,null,delim,scalars,arrays,tensors,objects,nulls,integers,floats,strings,iterables,dicts,symbols,character,epsilon,datatype

from src.iterables import Dict,Dictionary,setter,getter,getattrs,hasattrs,namespace,permutations

from src.io import load,dump,join,split,exists

from src.system import System,Space,Time,Lattice

from src.parameters import Parameters,Parameter

from src.optimize import Objective,Metric

delim = '.'
separ = '_'
memory = lambda size:size<2**20

class Basis(Dict):
	'''
	Basis Class of operators
	Args:
		D (int): Local dimension of system
		basis (str): Type of basis
		args (iterable): Additional class positional arguments
		kwargs (dict): Additional class keyword arguments
	'''

	def decorator(func):
		@wraps(func)
		def wrapper(cls,*args,system=None,**kwargs):
			# super().__init__(*args,system=system,**kwargs)

			setter(kwargs,dict(system=system),delimiter=delim,default=False)
			setter(kwargs,system,delimiter=delim,default=False)
			setter(kwargs,cls.defaults,delimiter=delim,default='none')

			out = func(cls,*args,**kwargs)

			return out

		return wrapper

	dimension = 2

	defaults = dict(
		D = 2,
		N = 1,

		shape = None,
		size = None,
		ndim = None,
		dtype = None,

		architecture = None,

		seed = None,
		random = None,
		bounds = [-1,1],

		data = None,
		operator = None,
		parameters = None,
	)

	@classmethod
	def get(cls,attr):
		for item in cls.__dict__:
			if cls.parse(item) == cls.parse(attr):
				return getattr(cls,item)
		return getattr(cls,attr)


	@classmethod
	def set(cls,attr,value):
		for item in cls.__dict__:
			if cls.parse(item) == cls.parse(attr):
				setattr(cls,item,value)
				return
		setattr(cls,attr,value)
		return

	@classmethod
	def parse(cls,attr):
		if attr is None:
			attr = None
		elif isinstance(attr,str):
			attr = attr
		else:
			attr = attr.__name__
		return attr

	@classmethod
	def dot(cls,data,other):
		return inner(ravel(data),ravel(other))

	@classmethod
	def contract(cls,state,data,where=None,options=None,**kwargs):
		'''
		Contract state and data
		Args:
			state (array,dict): Class state
			data (array,dict): Class data
			shape (iterable[int]): Shape of data
			where (float,int,iterable[int]): indices of class
			options (dict): Class options
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (array): Class data
		'''
		if state.ndim == 1:
			if data.ndim == 2:
				data = einsum('ij,j->i',data,state)
			else:
				raise NotImplementedError(f"Not Implemented {data}")
		elif state.ndim > 1:
			if data.ndim == 2:
				data = einsum('ij,jk...,lk->il...',data,state,conjugate(data))
			elif data.ndim == 3:
				data = einsum('uij,jk...,ulk->il...',data,state,conjugate(data))
			else:
				raise NotImplementedError(f"Not Implemented {data}")
		else:
			raise NotImplementedError(f"Not Implemented {state}")

		return data

	@classmethod
	def shuffle(cls,data,shape,where=None,transform=True,**kwargs):
		'''
		Shuffle class data
		Args:
			data (array,dict): Class data
			shape (iterable[int]): Shape of data
			where (float,int,iterable[int]): indices of class
			transform (bool): Forward or backward transform data
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (array): Class data
		'''

		if transform:
			n,d = len(shape),data.ndim
			where = [range(n)] if where is None else [where,sorted(set(range(n))-set(where))]
			shape = [*shape]*d
			axes = [j*n+i for indices in where for j in range(d) for i in indices]
			data = transpose(reshape(data,shape),axes)
		else:
			n,d = len(shape),data.ndim//len(shape)
			where = [j*n+i for indices in ([range(n)] if where is None else [where,sorted(set(range(n))-set(where))]) for j in range(d) for i in indices]
			shape = [prod(shape)]*d
			axes = [where.index(j*n+i) for j in range(d) for i in range(n)]
			data = reshape(transpose(data,axes),shape)
		return data

	@classmethod
	@decorator
	def opts(cls,attr,options,*args,**kwargs):
		'''
		Default options for class
		Args:
			attr (str,callable): Name of operator, either string or cls method corresponding to operator
			options (dict): Class options
			args (iterable): Additional cls operator positional arguments
			kwargs (iterable): Additional cls operator keyword arguments
		Returns:
			options (dict): Class options
		'''

		attr = cls.parse(attr)

		if attr in ['rand']:
			options.update(dict(
				shape=(options.shape if (getattr(options,'shape',None) is not None or any(obj is None for obj in (options.D,options.N,options.ndim,))) else
					  (options.D**options.N,)*options.ndim)
				))
		elif attr in ['unitary']:
			options.update(dict(
				shape=(options.shape if (getattr(options,'shape',None) is not None or any(obj is None for obj in (options.D,options.N,))) else
					  (options.D**options.N,)*cls.dimensions(attr,**options))
				))
		elif attr in ['gate']:
			options.update(dict(
				shape=(options.shape if (getattr(options,'shape',None) is not None or any(obj is None for obj in (options.D,options.N,))) else
					  (options.D**options.N,)*cls.dimensions(attr,**options))
				))
		elif attr in ['state']:
			options.update(dict(
				shape=(options.shape if (getattr(options,'shape',None) is not None or any(obj is None for obj in (options.D,options.N,options.ndim))) else
					  (options.D**options.N,)*options.ndim)
				))
		elif attr in ['ghz']:
			options.update(dict(
				shape=(options.shape if (getattr(options,'shape',None) is not None or any(obj is None for obj in (options.D,options.N,options.ndim))) else
					  (options.D**options.N,)*options.ndim)
				))
		return options

	@classmethod
	@decorator
	def localities(cls,attr,*args,**kwargs):
		'''
		Locality of composite operators
		Args:
			attr (str,callable): Name of operator, either string or cls method corresponding to operator
			args (iterable): Additional cls operator positional arguments
			kwargs (iterable): Additional cls operator keyword arguments
		Returns:
			locality (int): Locality of operator
		'''

		attr = cls.parse(attr)

		kwargs = Dictionary(**kwargs)

		options = Dictionary(
			D=kwargs.D if kwargs.D is not None else None,
			N=kwargs.N if kwargs.N is not None else None,
			ndim=cls.dimensions(attr,*args,**kwargs),
			operator=((
				[cls.parse(i) for i in kwargs.operator] if isinstance(kwargs.operator,iterables) else
				[i for i in cls.parse(kwargs.operator)] if isinstance(kwargs.operator,str) and not kwargs.operator.count(delim) else
				[cls.parse(i) for i in kwargs.operator.split(delim)] if isinstance(kwargs.operator,str) and kwargs.operator.count(delim) else
				None
				) if kwargs.operator is not None else None),
			shape=kwargs.shape,
			)

		if attr is None or not hasattr(cls,attr):
			locality = None
		elif attr in ['string']:
			locality = sum(cls.localities(i,*args,**kwargs) for i in options.operator if i is not None and hasattr(cls,i)) if options.operator is not None else None
		elif attr in ['data']:
			data = getattr(cls,attr)(*args,**kwargs)
			locality = int(round(log(data.size)/log(options.D)/data.ndim))
		elif attr in ['identity']:
			locality = 1
		elif attr in ['rand']:
			locality = options.N
		elif attr in ['I','X','Y','Z']:
			locality = 1
		elif attr in ['H','S']:
			locality = 1
		elif attr in ['CNOT']:
			locality = 2
		elif attr in ['gate','clifford']:
			locality = options.N
		elif attr in ['unitary']:
			locality = options.N
		elif attr in ['state']:
			locality = options.N
		elif attr in ['zero','one','plus','minus','plusi','minusi']:
			locality = 1
		elif attr in ['ghz']:
			locality = options.N
		elif attr in ['element']:
			locality = 1
		elif attr in ['projector']:
			locality = 1
		elif attr in ['dephase','bitflip','phaseflip','amplitude']:
			locality = 1
		elif attr in ['depolarize']:
			locality = 1
		elif attr in ['pauli','tetrad','trine','standard']:
			locality = 1
		else:
			locality = None

		return locality


	@classmethod
	@decorator
	def dimensions(cls,attr,*args,**kwargs):
		'''
		Number of dimensions of composite operators
		Args:
			attr (str,callable): Name of operator, either string or cls method corresponding to operator
			args (iterable): Additional cls operator positional arguments
			kwargs (iterable): Additional cls operator keyword arguments
		Returns:
			dimension (int): Number of dimensions of operator
		'''

		attr = cls.parse(attr)

		kwargs = Dictionary(**kwargs)

		options = Dictionary(
			D=kwargs.D if kwargs.D is not None else None,
			N=kwargs.N if kwargs.N is not None else None,
			ndim=kwargs.ndim if kwargs.ndim is not None else None,
			operator=((
				[cls.parse(i) for i in kwargs.operator] if isinstance(kwargs.operator,iterables) else
				[i for i in cls.parse(kwargs.operator)] if isinstance(kwargs.operator,str) and not kwargs.operator.count(delim) else
				[cls.parse(i) for i in kwargs.operator.split(delim)] if isinstance(kwargs.operator,str) and kwargs.operator.count(delim) else
				None
				) if kwargs.operator is not None else None),
			shape=kwargs.shape,
			)


		if attr is None or not hasattr(cls,attr):
			dimension = None
		elif attr in ['string']:
			dimension = max(cls.dimensions(i,*args,**kwargs) for i in options.operator if i is not None and hasattr(cls,i)) if options.operator is not None else None
		elif attr in ['data']:
			data = getattr(cls,attr)(*args,**kwargs)
			dimension = data.ndim
		elif attr in ['identity']:
			dimension = 2
		elif attr in ['rand']:
			dimension = options.ndim
		elif attr in ['I','X','Y','Z']:
			dimension = 2
		elif attr in ['H','S']:
			dimension = 2
		elif attr in ['CNOT']:
			dimension = 2
		elif attr in ['gate','clifford']:
			dimension = 2
		elif attr in ['unitary']:
			dimension = 2
		elif attr in ['state']:
			dimension = options.ndim
		elif attr in ['zero','one','plus','minus','plusi','minusi']:
			dimension = max(
				options.ndim if options.ndim is not None else 0,
				len(options.shape) if options.shape is not None and not isinstance(options.shape,int) else 0
				) if options.ndim is not None or options.shape is not None else 1
		elif attr in ['ghz']:
			dimension = max(
				options.ndim if options.ndim is not None else 0,
				len(options.shape) if options.shape is not None and not isinstance(options.shape,int) else 0
				) if options.ndim is not None or options.shape is not None else 1
		elif attr in ['element']:
			dimension = len(options.operator) if options.operator is not None else 1
		elif attr in ['projector']:
			dimension = 2
		elif attr in ['dephase','bitflip','phaseflip','amplitude']:
			dimension = 3
		elif attr in ['depolarize']:
			dimension = 3
		elif attr in ['pauli','tetrad','trine','standard']:
			dimension = 3
		else:
			dimension = None

		return dimension


	@classmethod
	@decorator
	def shapes(cls,attr,*args,**kwargs):
		'''
		Shape of composite operators
		Args:
			attr (str,callable): Name of operator, either string or cls method corresponding to operator
			args (iterable): Additional cls operator positional arguments
			kwargs (iterable): Additional cls operator keyword arguments
		Returns:
			shape (dict[int,iterable[int]]): Composite shape {axis_i: [D_i0,...D_iN-1]} of each ndim axis of operator
		'''

		attr = cls.parse(attr)

		kwargs = Dictionary(**kwargs)

		options = Dictionary(
			D=kwargs.D if kwargs.D is not None else None,
			N=cls.localities(attr,*args,**kwargs),
			ndim=cls.dimensions(attr,*args,**kwargs),
			operator=((
				[cls.parse(i) for i in kwargs.operator] if isinstance(kwargs.operator,iterables) else
				[i for i in cls.parse(kwargs.operator)] if isinstance(kwargs.operator,str) and not kwargs.operator.count(delim) else
				[cls.parse(i) for i in kwargs.operator.split(delim)] if isinstance(kwargs.operator,str) and kwargs.operator.count(delim) else
				None
				) if kwargs.operator is not None else None),
			shape=kwargs.shape,
			)

		if attr is None or not hasattr(cls,attr):
			shape = None
		elif attr in ['string']:
			shape = {key: cls.shapes(data,*args,**kwargs) for key,data in enumerate(options.operator)} if options.operator is not None else None

			D = max(i for key in shape if shape[key] is not None for i in shape[key]) if shape is not None else None
			N = sum(len(shape[key]) for key in shape if shape[key] is not None) if shape is not None else None
			ndim = max(len(shape[key]) for key in shape if shape[key] is not None) if shape is not None else None

			shape = {axis: [i
				for key in shape if shape[key] is not None
				for i in (
					shape[key][axis-ndim+len(shape[key])]
					if (axis >= (ndim-len(shape[key]))) else
					[1]*max(len(shape[key][i]) for i in shape[key])
					)
				] for axis in range(ndim)} if shape is not None else None

		elif attr in ['data']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['identity']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['rand']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['I','X','Y','Z']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['H','S']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['CNOT']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['gate','clifford']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['unitary']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['state']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['zero','one','plus','minus','plusi','minusi']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['ghz']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['element']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['projector']:
			shape = {i: [options.D]*options.N for i in range(options.ndim)}
		elif attr in ['dephase','bitflip','phaseflip','amplitude']:
			shape = {**{i: [2]*options.N for i in range(1)},**{i: [options.D]*options.N for i in range(1,options.ndim)}}
		elif attr in ['depolarize']:
			shape = {**{i: [options.D**2]*options.N for i in range(1)},**{i: [options.D]*options.N for i in range(1,options.ndim)}}
		elif attr in ['pauli','tetrad','trine','standard']:
			shape = {**{i: [options.D**2]*options.N for i in range(1)},**{i: [options.D]*options.N for i in range(1,options.ndim)}}
		else:
			shape = None

		return shape

	# Basis
	@classmethod
	@decorator
	def basis(cls,attr,*args,**kwargs):
		'''
		Shape of composite operators
		Args:
			attr (str,callable): Name of operator, either string or cls method corresponding to operator
			args (iterable): Additional cls operator positional arguments
			kwargs (iterable): Additional cls operator keyword arguments
		Returns:
			basis (dict[iterable[str],object]): Basis elements
		'''

		attr = cls.parse(attr)

		kwargs = Dictionary(**kwargs)

		options = Dictionary(
			**{
			**kwargs,
			**dict(
				D=kwargs.D if kwargs.D is not None else None,
				N=kwargs.N if kwargs.N is not None else 1,
				)
			}
			)

		if attr in ['pauli']:
			basis = {index: tensorprod([getattr(cls,i)(**options) for i in index]) for index in permutations(['I','X','Y','Z'],repeat=options.N)}
		else:
			basis = None

		basis = {i: basis[i]/sqrt(cls.dot(basis[i],basis[i])) for i in basis} if basis is not None else None

		return basis



	# General
	@classmethod
	@decorator
	def string(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = kwargs.data if kwargs.data is not None else None
		data = None if data is None else data.split(delim) if isinstance(data,str) else [*data]
		data = [getattr(cls,i)(**kwargs) for i in data] if data is not None else None
		data = tensorprod(data) if data is not None else None
		data = array(data,dtype=kwargs.dtype) if data is not None else None
		return data

	@classmethod
	@decorator
	def data(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = kwargs.data if kwargs.data is not None else None
		data = load(data) if isinstance(data,str) else data
		data = data(*args,**kwargs) if callable(data) else data
		return data

	@classmethod
	@decorator
	def identity(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = identity(kwargs.D,dtype=kwargs.dtype)
		return data


	# Random
	@classmethod
	@decorator
	def rand(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = rand(
			shape=kwargs.shape,
			random=kwargs.random,
			bounds=kwargs.bounds,
			seed=kwargs.seed,
			dtype=kwargs.dtype)
		if data is not None and data.ndim == 1:
			data /= sqrt(inner(data,data))
		elif data is not None and data.ndim == 2:
			data = (data + dagger(data))/2
			data /= trace(data)
		return data

	# Pauli
	@classmethod
	@decorator
	def I(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.D is None:
			data = array([[1,0],[0,1]],dtype=kwargs.dtype)
		else:
			data = sum(cls.element(D=kwargs.D,data='%d%d'%(i,i),dtype=kwargs.dtype) for i in range(kwargs.D))
		return data

	@classmethod
	@decorator
	def X(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.D is None:
			data = array([[0,1],[1,0]],dtype=kwargs.dtype)
		else:
			data = sum(cls.element(D=kwargs.D,data='%d%d'%((i+1)%kwargs.D,i),dtype=kwargs.dtype) for i in range(kwargs.D))
		return data

	@classmethod
	@decorator
	def Y(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.D is None:
			data = array([[0,-1j],[1j,0]],dtype=kwargs.dtype)
		else:
			data = -1j*dot(cls.Z(*args,**kwargs),cls.X(*args,**kwargs))
		return data

	@classmethod
	@decorator
	def Z(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.D is None:
			data = array([[1,0],[0,-1]],dtype=kwargs.dtype)
		else:
			data = sum((e**(1j*2*pi*i/kwargs.D))*cls.element(D=kwargs.D,data='%d%d'%(i,i),dtype=kwargs.dtype) for i in range(kwargs.D))
		return data

	# Gate
	@classmethod
	@decorator
	def H(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/sqrt(2))*array([[1,1],[1,-1]],dtype=kwargs.dtype)
		return data

	@classmethod
	@decorator
	def S(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([[1,0,],[0,1j]],dtype=kwargs.dtype)
		return data

	@classmethod
	@decorator
	def T(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([[1,0,],[0,(1+1j)/sqrt(2)]],dtype=kwargs.dtype)
		return data

	@classmethod
	@decorator
	def CNOT(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=kwargs.dtype)
		return data

	@classmethod
	# @decorator
	def gate(cls,*args,**kwargs):
		data = {1:[cls.I,cls.H,cls.S,cls.T],2:[cls.identity,cls.CNOT]}[kwargs['N']]
		data = [jit(func,D=kwargs['D']**kwargs['N'],dtype=kwargs['dtype']) for func in data]
		index = choice(
			data=len(data),
			shape=(),
			seed=kwargs['seed'],
			dtype=int
			)
		data = switch(index,data)
		return data

	@classmethod
	# @decorator
	def clifford(cls,*args,**kwargs):
		data = {1:[cls.I,cls.H,cls.S],2:[cls.identity,cls.CNOT]}[kwargs['N']]
		data = [jit(func,D=kwargs['D']**kwargs['N'],dtype=kwargs['dtype']) for func in data]
		index = choice(
			data=len(data),
			shape=(),
			seed=kwargs['seed'],
			dtype=int
			)
		data = switch(index,data)
		return data

	# Unitary
	@classmethod
	# @decorator
	def unitary(cls,*args,**kwargs):
		# kwargs = Dictionary(**kwargs)
		# data = haar(
		# 	shape=kwargs.shape,
		# 	seed=kwargs.seed,
			# dtype=kwargs.dtype)
		data = haar(
			shape=kwargs['shape'],
			seed=kwargs['seed'],
			dtype=kwargs['dtype']
			)
		return data

	# State
	@classmethod
	@decorator
	def state(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = haar(
			shape=kwargs.shape if isinstance(kwargs.shape,iterables) and len(kwargs.shape) == 2 else kwargs.shape*2 if isinstance(kwargs.shape,iterables) else (kwargs.shape,)*2 if isinstance(kwargs.shape,integers) else (kwargs.D,)*2,
			seed=kwargs.seed,
			dtype=kwargs.dtype)[0]
		if data is not None and data.ndim < max(
			kwargs.ndim if kwargs.ndim is not None else 0,
			len(kwargs.shape) if not isinstance(kwargs.shape,int) else 0
			):
			data = outer(data,data)
		return data

	@classmethod
	@decorator
	def zero(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([1,*[0]*(kwargs.D-1)],dtype=kwargs.dtype)
		if kwargs.architecture is None or kwargs.architecture in ['array']:
			if data is not None and data.ndim < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = outer(data,data)
		elif kwargs.architecture in ['tensor']:
			data = [data[i]*cls.element(D=kwargs.D,data=[i],dtype=kwargs.dtype) for i in range(kwargs.D)]
			data = array([outer(i,i) for i in data])
			if data is not None and (data.ndim-2) < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = tensorprod([data,conjugate(data)])
			data = transpose(reshape(data,(*[kwargs.D]*kwargs.ndim,*data.shape[-2:])),(-2,*range(kwargs.ndim),-1))
		return data

	@classmethod
	@decorator
	def one(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = array([*[0]*(kwargs.D-1),1],dtype=kwargs.dtype)
		if kwargs.architecture is None or kwargs.architecture in ['array']:
			if data is not None and data.ndim < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = outer(data,data)
		elif kwargs.architecture in ['tensor']:
			data = [data[i]*cls.element(D=kwargs.D,data=[i],dtype=kwargs.dtype) for i in range(kwargs.D)]
			data = array([outer(i,i) for i in data])
			if data is not None and (data.ndim-2) < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = tensorprod([data,conjugate(data)])
			data = transpose(reshape(data,(*[kwargs.D]*kwargs.ndim,*data.shape[-2:])),(-2,*range(kwargs.ndim),-1))
		return data

	@classmethod
	@decorator
	def plus(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/sqrt(kwargs.D))*array([1,1],dtype=kwargs.dtype)
		if kwargs.architecture is None or kwargs.architecture in ['array']:
			if data is not None and data.ndim < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = outer(data,data)
		elif kwargs.architecture in ['tensor']:
			data = [data[i]*cls.element(D=kwargs.D,data=[i],dtype=kwargs.dtype) for i in range(kwargs.D)]
			data = array([outer(i,i) for i in data])
			if data is not None and (data.ndim-2) < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = tensorprod([data,conjugate(data)])
			data = transpose(reshape(data,(*[kwargs.D]*kwargs.ndim,*data.shape[-2:])),(-2,*range(kwargs.ndim),-1))
		return data

	@classmethod
	@decorator
	def minus(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/sqrt(kwargs.D))*array([1,-1],dtype=kwargs.dtype)
		if kwargs.architecture is None or kwargs.architecture in ['array']:
			if data is not None and data.ndim < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = outer(data,data)
		elif kwargs.architecture in ['tensor']:
			data = [data[i]*cls.element(D=kwargs.D,data=[i],dtype=kwargs.dtype) for i in range(kwargs.D)]
			data = array([outer(i,i) for i in data])
			if data is not None and (data.ndim-2) < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = tensorprod([data,conjugate(data)])
			data = transpose(reshape(data,(*[kwargs.D]*kwargs.ndim,*data.shape[-2:])),(-2,*range(kwargs.ndim),-1))
		return data

	@classmethod
	@decorator
	def plusi(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/sqrt(kwargs.D))*array([1,1j],dtype=kwargs.dtype)
		if kwargs.architecture is None or kwargs.architecture in ['array']:
			if data is not None and data.ndim < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = outer(data,data)
		elif kwargs.architecture in ['tensor']:
			data = [data[i]*cls.element(D=kwargs.D,data=[i],dtype=kwargs.dtype) for i in range(kwargs.D)]
			data = array([outer(i,i) for i in data])
			if data is not None and (data.ndim-2) < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = tensorprod([data,conjugate(data)])
			data = transpose(reshape(data,(*[kwargs.D]*kwargs.ndim,*data.shape[-2:])),(-2,*range(kwargs.ndim),-1))
		return data

	@classmethod
	@decorator
	def minusi(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = (1/sqrt(kwargs.D))*array([1,-1j],dtype=kwargs.dtype)
		if kwargs.architecture is None or kwargs.architecture in ['array']:
			if data is not None and data.ndim < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = outer(data,data)
		elif kwargs.architecture in ['tensor']:
			data = [data[i]*cls.element(D=kwargs.D,data=[i],dtype=kwargs.dtype) for i in range(kwargs.D)]
			data = array([outer(i,i) for i in data])
			if data is not None and (data.ndim-2) < (kwargs.ndim if kwargs.ndim is not None else 0):
				data = tensorprod([data,conjugate(data)])
			data = transpose(reshape(data,(*[kwargs.D]*kwargs.ndim,*data.shape[-2:])),(-2,*range(kwargs.ndim),-1))
		return data

	@classmethod
	@decorator
	def ghz(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = [ravel(cls.element(D=kwargs.D,data=[i]*kwargs.N,dtype=kwargs.dtype)) for i in range(kwargs.D)]
		if kwargs.architecture is None or kwargs.architecture in ['array']:
			data = (1/sqrt(kwargs.D))*sum(data)
			if data is not None and data.ndim < max(
				kwargs.ndim if kwargs.ndim is not None else 0,
				len(kwargs.shape) if not isinstance(kwargs.shape,int) else 0
				):
				data = outer(data,data)
		elif kwargs.architecture in ['tensor']:
			data = array([outer(i,i) for i in data])
			if data is not None and 1 < max(
				kwargs.ndim if kwargs.ndim is not None else 0,
				len(kwargs.shape) if not isinstance(kwargs.shape,int) else 0
				):
				data = tensorprod([data,conjugate(data)])
			data = transpose(reshape(data,(*[kwargs.D]*kwargs.ndim,*data.shape[-2:])),(-2,*range(kwargs.ndim),-1))
		return data

	# Operator
	@classmethod
	@decorator
	def element(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		index = tuple(map(int,kwargs.data)) if kwargs.data is not None else None
		data = zeros((kwargs.D,)*(len(index) if index is not None else 1),dtype=kwargs.dtype)
		data = inplace(data,index,1) if index is not None else data
		return data

	@classmethod
	@decorator
	def projector(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		data = []
		tmp = zeros((kwargs.D,kwargs.D),dtype=kwargs.dtype)
		for i in range(size):
			for j in range(size):
				obj = inplace(tmp,(i,j),1)
				data.append(obj)
		return data

	# Noise
	@classmethod
	@decorator
	def dephase(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.parameters is None:
			kwargs.parameters = 0
		data = array([
			sqrt(1-kwargs.parameters)*cls.I(D=kwargs.D,dtype=kwargs.dtype),
			sqrt(kwargs.parameters)*cls.Z(D=kwargs.D,dtype=kwargs.dtype)
			],dtype=kwargs.dtype)
		return data

	@classmethod
	@decorator
	def bitflip(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.parameters is None:
			kwargs.parameters = 0
		data = array([
			sqrt(1-kwargs.parameters)*cls.I(D=kwargs.D,dtype=kwargs.dtype),
			sqrt(kwargs.parameters)*cls.X(D=kwargs.D,dtype=kwargs.dtype)
			],dtype=kwargs.dtype)
		return data

	@classmethod
	@decorator
	def phaseflip(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.parameters is None:
			kwargs.parameters = 0
		data = array([
			sqrt(1-kwargs.parameters)*cls.I(D=kwargs.D,dtype=kwargs.dtype),
			sqrt(kwargs.parameters)*cls.Y(D=kwargs.D,dtype=kwargs.dtype)
			],dtype=kwargs.dtype)
		return data

	@classmethod
	@decorator
	def depolarize(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.parameters is None:
			kwargs.parameters = 0
		data = array([
				sqrt(1-(kwargs.D**2-1)*kwargs.parameters/(kwargs.D**2))*cls.I(D=kwargs.D,dtype=kwargs.dtype),
				sqrt(kwargs.parameters/(kwargs.D**2))*cls.X(D=kwargs.D,dtype=kwargs.dtype),
				sqrt(kwargs.parameters/(kwargs.D**2))*cls.Y(D=kwargs.D,dtype=kwargs.dtype),
				sqrt(kwargs.parameters/(kwargs.D**2))*cls.Z(D=kwargs.D,dtype=kwargs.dtype)
				],dtype=kwargs.dtype)
		return data

	@classmethod
	@decorator
	def amplitude(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)
		if kwargs.parameters is None:
			kwargs.parameters = 0
		data = array([
			cls.element(D=kwargs.D,data='00',dtype=kwargs.dtype) +
				sqrt(1-kwargs.parameters)*cls.element(D=kwargs.D,data='11',dtype=kwargs.dtype),
			sqrt(kwargs.parameters)*cls.element(D=kwargs.D,data='01',dtype=kwargs.dtype)
			],dtype=kwargs.dtype)
		return data

	# POVM
	@classmethod
	@decorator
	def pauli(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)

		if kwargs.ndim is None or kwargs.ndim < 2:
			kwargs.ndim = 2

		data = (1/(kwargs.D**2-1))*array([
				cls.zero(*args,**kwargs),
				cls.plus(*args,**kwargs),
				cls.plusi(*args,**kwargs),
			   (cls.one(*args,**kwargs)+
				cls.minus(*args,**kwargs)+
				cls.minusi(*args,**kwargs)),
			],dtype=kwargs.dtype)

		return data


	@classmethod
	@decorator
	def tetrad(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)

		if kwargs.ndim is None or kwargs.ndim < 2:
			kwargs.ndim = 2

		data = (1/(kwargs.D**2))*array([
			sum(i*operator(*args,**kwargs)
				for i,operator in
				zip(parameters,(cls.I,cls.X,cls.Y,cls.Z)))
			for parameters in [
			(1,0,0,1),
			(1,2*sqrt(2)/3,0,-1/3),
			(1,-sqrt(2)/3,sqrt(2/3),-1/3),
			(1,-sqrt(2)/3,-sqrt(2/3),-1/3)
			]
			],dtype=kwargs.dtype)

		return data

	@classmethod
	@decorator
	def povm(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)

		if kwargs.ndim is None or kwargs.ndim < 2:
			kwargs.ndim = 2

		arguments = (*args,)
		keywords = {**kwargs,**dict(shape=kwargs.shape if isinstance(kwargs.shape,iterables) and len(kwargs.shape) == 2 else kwargs.shape*2 if isinstance(kwargs.shape,iterables) else (kwargs.shape,)*2 if isinstance(kwargs.shape,integers) else (kwargs.D,)*2,)}

		data = 'unitary'
		unitary = getattr(cls,data)(*arguments,**keywords)

		data = 'tetrad'
		data = getattr(cls,data)(*args,**kwargs)

		data = array([dot(unitary,dot(i,dagger(unitary))) for i in data])

		return data

	@classmethod
	@decorator
	def trine(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)

		if kwargs.ndim is None or kwargs.ndim < 2:
			kwargs.ndim = 2

		data = (1/(kwargs.D**2-1))*array([
										cls.I(D=kwargs.D,dtype=kwargs.dtype) +
			cos(i*2*pi/(kwargs.D**2-1))*cls.Z(D=kwargs.D,dtype=kwargs.dtype) +
			sin(i*2*pi/(kwargs.D**2-1))*cls.X(D=kwargs.D,dtype=kwargs.dtype)
			for i in range(kwargs.D**2-1)
			],dtype=kwargs.dtype)

		raise ValueError('Not Informationally Complete POVM <%s>'%(sys._getframe().f_code.co_name))

		return data

	@classmethod
	@decorator
	def standard(cls,*args,**kwargs):
		kwargs = Dictionary(**kwargs)

		if kwargs.ndim is None or kwargs.ndim < 2:
			kwargs.ndim = 2

		data = zeros((kwargs.D**2,kwargs.D**2),dtype=kwargs.dtype)
		for i in range(kwargs.D**2):
			data = inplace(data,(i,i),1)
		data = reshape(data,shape=(kwargs.D**2,kwargs.D,kwargs.D))

		raise ValueError('Non-Normalized POVM <%s>'%(sys._getframe().f_code.co_name))

		return data


class Measure(System):

	N = None
	D = None

	basis = None

	defaults = dict(
		data=None,operator=None,string=None,system=None,
		shape=None,size=None,ndim=None,dtype=None,
		basis=None,inverse=None,structure=None,ones=None,zeros=None,pointer=None,
		parameters=None,variable=None,constant=None,symmetry=None,hermitian=None,unitary=None,
		func=None,gradient=None,
		)

	def __init__(self,data=None,operator=None,string=None,system=None,**kwargs):
		'''
		Measure Class
		Args:
			data (str,array,tensor,mps,Measure): data of measure
			operator (str): name of measure basis
			string (str): string label of measure
			system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
			args (iterable): Additional class positional arguments
			kwargs (dict): Additional class keyword arguments
		'''

		setter(kwargs,dict(data=data,operator=operator,string=string,system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)

		self.init()

		return

	def init(self,data=None,parameters=None,**kwargs):
		'''
		Initialize measure
		Args:
			data (str,array,tensor,mps,Measure): data of measure
			parameters (array,dict): parameters of class
			kwargs (dict): Additional class keyword arguments
		'''

		data = self.data if data is None else data
		parameters = self.parameters if parameters is None else parameters

		if parameters is None or not callable(parameters):
			def parameters(parameters=parameters):
				return parameters

		self.parameters = parameters

		for kwarg in kwargs:
			if hasattr(self,kwarg):
				setattr(self,kwarg,kwargs[kwarg])

		self.setup(data=data)

		return

	def setup(self,data=None,operator=None,string=None,**kwargs):
		'''
		Setup measure
		Args:
			data (str,array,tensor,mps,Measure): data of measure
			operator (str): name of measure basis
			string (str): string label of measure
			kwargs (dict): Additional class keyword arguments
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		string = self.string if string is None else string

		operator = data if operator is None and isinstance(data,str) else operator if data is None else operator
		invariant = isinstance(operator,str)
		symmetry = self.symmetry is None
		N = min((i for i in (self.N,len(operator) if not invariant else self.N if self.N is not None else None) if i is not None),default=None)
		D = self.D
		seed = self.seed
		dtype = self.dtype

		where = range(N) if N is not None else None
		pointer = min(where) if where is not None else 0 if N is not None else None

		operator = [operator for i in where] if invariant else [operator[i] for i in where]
		seed = [seeder(seed)]*N if symmetry else seeder(seed,size=N)
		options = [dict(D=D,dtype=dtype,seed=seed[i],system=self.system) for i in where]

		basis = [getattr(Basis,operator[pointer])(**options[pointer])]*N if symmetry and invariant else [getattr(Basis,operator[i])(**options[i]) for i in where]
		inverse = [inv(einsum('uij,vji->uv',basis[pointer],basis[pointer]))]*N if symmetry else [inv(einsum('uij,vji->uv',basis[i],basis[i])) for i in where]
		structure = [D*(D+1)*einsum('uij,vjk,wki->uvw',basis[pointer],basis[pointer],basis[pointer])-einsum('uij,vji,w->uvw',basis[pointer],basis[pointer],array([1 for i in range(len(basis[pointer]))],dtype=dtype))]*N if symmetry else [D*(D+1)*einsum('uij,vjk,wki->uvw',basis[pointer],basis[pointer],basis[pointer])-einsum('uij,vji,w->uvw',basis[i],basis[i],array([1 for i in range(len(basis[i]))],dtype=dtype)) for i in where]

		ones = [array([1 for i in range(len(basis[i]))],dtype=dtype) for i in where]
		zeros = [array([0 for i in range(len(basis[i]))],dtype=dtype) for i in where]
		pointer = pointer if pointer is not None else None

		K = max(len(basis[i]) for i in where)
		shape = [min(data.shape[axis] for i in where for data in basis[i]) for axis in range(min(len(data.shape) for i in where for data in basis[i]))]
		size = prod(shape)
		ndim = len(shape)
		dtype = dtype

		if self.architecture is None or self.architecture in ['array']:

			cls = array

			kwargs = dict(dtype=dtype)

			basis = [cls(basis[pointer],**kwargs)]*N if symmetry else [cls(basis[i],**kwargs) for i in where]
			inverse = [cls(inverse[pointer],**kwargs)]*N if symmetry else [cls(inverse[i],**kwargs) for i in where]
			structure = [cls(structure[pointer],**kwargs)]*N if symmetry else [cls(structure[i],**kwargs) for i in where]

			ones = [cls(ones[pointer],**kwargs)]*N if symmetry else [cls(ones[i],**kwargs) for i in where]
			zeros = [cls(zeros[pointer],**kwargs)]*N if symmetry else [cls(zeros[i],**kwargs) for i in where]

		elif self.architecture in ['tensor']:

			self.ind = 'u{}'
			self.inds = ('u{}','v{}','w{}',)
			self.indices = ('i{}','j{}',)
			self.symbol = ('x{}','y{}',)
			self.symbols = ('k{}','l{}','m{}','n{}',)

			cls = tensor

			kwargs = dict(indices=[*self.inds[:1],*self.indices[:2]])
			basis = [cls(basis[pointer],**kwargs)]*N if symmetry else [cls(basis[i],**kwargs) for i in where]

			kwargs = dict(indices=[*self.inds[:2]])
			inverse = [cls(inverse[pointer],**kwargs)]*N if symmetry else [cls(inverse[i],**kwargs) for i in where]

			kwargs = dict(indices=[*self.inds[:3]])
			structure = [cls(structure[pointer],**kwargs)]*N if symmetry else [cls(structure[i],**kwargs) for i in where]

			kwargs = dict(indices=[*self.inds[:1]])
			ones = [cls(ones[pointer],**kwargs)]*N if symmetry else [cls(ones[i],**kwargs) for i in where]

			kwargs = dict(indices=[*self.inds[:1]])
			zeros = [cls(zeros[pointer],**kwargs)]*N if symmetry else [cls(zeros[i],**kwargs) for i in where]

		elif self.architecture in ['tensor_quimb']:

			self.ind = 'u{}'
			self.inds = ('u{}','v{}','w{}',)
			self.indices = ('i{}','j{}',)
			self.tag = 'I{}'
			self.tags = ()
			self.symbol = ('x{}','y{}',)
			self.symbols = ('k{}','l{}','m{}','n{}',)

			cls = tensor_quimb

			kwargs = dict(inds=(*self.inds[:1],*self.indices[:2],),tags=(self.tag,*self.tags,))
			basis = [cls(basis[pointer],**kwargs)]*N if symmetry else [cls(basis[i],**kwargs) for i in where]

			kwargs = dict(inds=(*self.inds[:2],),tags=(self.tag,*self.tags,))
			inverse = [cls(inverse[pointer],**kwargs)]*N if symmetry else [cls(inverse[i],**kwargs) for i in where]

			kwargs = dict(inds=(*self.inds[:3],),tags=(self.tag,*self.tags,))
			structure = [cls(structure[pointer],**kwargs)]*N if symmetry else [cls(structure[i],**kwargs) for i in where]

			kwargs = dict(inds=(*self.inds[:1],),tags=(self.tag,*self.tags,))
			ones = [cls(ones[pointer],**kwargs)]*N if symmetry else [cls(ones[i],**kwargs) for i in where]

			kwargs = dict(inds=(*self.inds[:1],),tags=(self.tag,*self.tags,))
			zeros = [cls(zeros[pointer],**kwargs)]*N if symmetry else [cls(zeros[i],**kwargs) for i in where]

		basis = [basis]*N if isinstance(basis,objects) else basis
		inverse = [inverse]*N if isinstance(inverse,objects) else inverse
		structure = [structure]*N if isinstance(structure,objects) else structure
		ones = [ones]*N if isinstance(ones,objects) else ones
		zeros = [zeros]*N if isinstance(zeros,objects) else zeros

		if self.parameters is not None and self.parameters() is not None:
			variable = True
			constant = False
			symmetry = None
		else:
			variable = False
			constant = True
			symmetry = None

		hermitian = True
		unitary = False

		self.N = N
		self.D = D

		self.operator = operator
		self.string = string

		self.basis = basis
		self.inverse = inverse
		self.structure = structure
		self.ones = ones
		self.zeros = zeros
		self.pointer = pointer

		self.shape = shape
		self.size = size
		self.ndim =  ndim
		self.dtype = dtype

		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary

		if self.architecture is None or self.architecture in ['array']:
			subscripts = '...u,uv,vij->...ij'
			shapes = ((self.K,),self.inverse[self.pointer].shape,self.basis[self.pointer].shape)
			einsummation = einsummand(subscripts,*shapes)
			def func(parameters=None,state=None,**kwargs):
				return einsummation(state,self.inverse[self.pointer],self.basis[self.pointer])

			def gradient(parameters=None,state=None,**kwargs):
				return 0

		elif self.architecture in ['tensor']:

			def func(parameters=None,state=None,**kwargs):
				N = state.N
				for i in range(N):
					with context(self.basis[i],self.inverse[i],formats=i,indices=[{self.inds[0]:self.inds[1]},None]):
						state &= self.inverse[i] & self.basis[i]
				return state
			def gradient(parameters=None,state=None,**kwargs):
				return 0

		elif self.architecture in ['tensor_quimb']:
			def func(parameters=None,state=None,**kwargs):
				N = state.L
				for i in range(N):
					with context_quimb(self.basis[i],self.inverse[i],key=i,formats=dict(inds=[{self.inds[0]:self.inds[1]},{index:index for index in self.inds}],tags=None)):
						state &= self.inverse[i] & self.basis[i]
				return state

			def gradient(parameters=None,state=None,**kwargs):
				return 0

		self.func = func
		self.gradient = gradient

		parameters = self.parameters()
		wrapper = partial
		kwargs = {}

		self.func = wrapper(self.func,parameters=parameters,**kwargs)
		self.gradient = wrapper(self.gradient,parameters=parameters,**kwargs)

		return

	def __call__(self,parameters=None,state=None,**kwargs):
		'''
		Call class for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class of Probability state of shape (N,self.K) or (self.K**N,)
			kwargs (dict): Additional class keyword arguments
		'''
		parameters = self.parameters() if parameters is None else parameters() if callable(parameters) else parameters

		return self.func(parameters=parameters,state=state,**kwargs)

	def __len__(self):
		return self.basis[self.pointer].shape[0]

	def __str__(self):
		if isinstance(self.string,str):
			string = self.string
		else:
			string = self.__class__.__name__
		return string

	def __repr__(self):
		return str(self)

	def get(self,attr):
		return getattr(self,attr)

	def set(self,attr,value):
		setattr(self,attr,value)
		return

	@property
	def K(self):
		return len(self)

	def info(self,display=None,ignore=None,verbose=None,**kwargs):
		'''
		Log class information
		Args:
			display (str,iterable[str]): Show attributes
			ignore (str,iterable[str]): Do not show attributes
			verbose (bool,int,str): Verbosity of message
			kwargs (dict): Additional logging keyword arguments
		'''

		if not verbose and not self.verbose:
			return

		msg = []

		options = dict(
			align=kwargs.get('align','<'),
			space=kwargs.get('space',1),
			width=kwargs.get('width',2)
			)

		precision = kwargs.get('precision',8)

		parse = lambda obj: str(obj.round(precision)) if isinstance(obj,arrays) else str(obj)

		display = None if display is None else [display] if isinstance(display,str) else display
		ignore = None if ignore is None else [ignore] if isinstance(ignore,str) else ignore

		for attr in [None,'string','key','seed','instance','instances','backend','architecture','timestamp','operator','N','D','K','data']:

			obj = attr
			if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
				continue

			if attr is None:
				attr = 'cls'
				substring = str(self)
			else:
				substring = getattrs(self,attr,delimiter=delim,default=None)

			if isinstance(substring,objects):
				string = '%s:\n%s'%(attr,parse(substring))
			else:
				string = '%s: %s'%(attr,parse(substring))

			msg.append(string)

		msg = [i if isinstance(i,str) else str(i) for i in msg]

		msg = '\n'.join(msg)

		self.log(msg,verbose=verbose)

		return

	def transform(self,parameters=None,state=None,model=None,transformation=None,where=None,func=None,options=None,**kwargs):
		'''
		Probability for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (str,iterable[str],array,tensor,network): state of class of shape (N,self.D,self.D) or (self.D**N,self.D**N)
			model (callable): model of operator with signature model(parameters,state,**kwargs) -> data
			transformation (bool,str): Type of transformation, True for amplitude -> probability or model to fun, or False for probability -> amplitude, allowed strings in ['probability','amplitude','operator','state','function','model'], default of amplitude -> probability
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			state (array,tensor,network): state of class of Probability state of shape (N,self.K) or (self.K**N,) or (self.D**N,self.D**N)
			func (callable): operator with signature func(parameters,state,where,**kwargs) -> data (array) POVM operator of shape (self.K**N,self.K**N)
		'''

		if transformation in [None,True,'probability','state'] and model is None:

			return self.probability(parameters=parameters,state=state,where=where,func=func,options=options,**kwargs)

		elif transformation in [False,'amplitude','function']:

			return self.amplitude(parameters=parameters,state=state,where=where,func=func,options=options,**kwargs)

		elif transformation in [None,True,'probability','state'] and model is not None:

			state = self.transform(parameters=parameters,state=state,transformation=transformation,where=where,func=func,options=options,**kwargs)

			return self.operation(parameters=parameters,state=state,model=model,where=where,func=func,options=options,**kwargs)

		else:

			return state

	def probability(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Probability for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (iterable[array,tensor,callable]): state of class of shape (N,self.D,self.D)
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			state (array,tensor,network): state of class of Probability state of shape (N,self.K) or (self.K**N)
		'''

		state = [*state] if isinstance(state,iterables) else [state] if state is not None else state
		parameters = self.parameters() if parameters is None else parameters() if callable(parameters) else parameters

		if state is None:
			return state


		if self.architecture is None or self.architecture in ['array']:

			N = len(state)
			D = self.D
			ndim = 2

			cls = array

			for i in range(N):

				data = state[i] if not callable(state[i]) else state[i]()

				data = einsum('uij,...ji->...u',self.basis[i],data)

				data = cls(data)

				state[i] = data

			state = tensorprod(state)

		elif self.architecture in ['tensor']:

			N = len(state)
			D = self.D
			ndim = 2

			size = None

			cls = tensor

			for i in range(N):

				data = state[i] if not callable(state[i]) else state[i]()
				size = max(0,(data.ndim-ndim)//2)

				if size:
					indices = [*[self.symbols[i] for i in range(size)],*self.indices[:ndim][::-1],*[self.symbols[size+i] for i in range(size)]]
				else:
					indices = [*self.indices[:ndim][::-1]]

				data = cls(data=data,indices=indices)

				data &= self.basis[i]

				data.format(i)

				if size:
					data.transform(axes=[*[i for i in range(size)],-1,*[size+i for i in range(size)]])

				state[i] = data


			if size:
				options = Dictionary(constant=addition(dots(*(state[i]() for i in range(N)))))
				for i in [0,N-1]:
					if i in [0]:
						state[i].transform(func=lambda data,options=options:(1/sqrt(options.constant))*addition(data(),axis=0),shape=lambda data:{data.indices[0]:1,**dict(zip(data.indices[1:],data.shape[:]))})
					elif i in [N-1]:
						state[i].transform(func=lambda data,options=options:(1/sqrt(options.constant))*addition(data(),axis=-1),shape=lambda data:{**dict(zip(data.indices[:-1],data.shape[:])),data.indices[-1]:1})


			options = {**(self.options if self.options is not None else {}),**{}}

			state = mps(state,**options)

		elif self.architecture in ['tensor_quimb']:

			N = len(state)
			D = self.D
			ndim = 2

			cls = tensor_quimb

			for i in range(N):

				data = state[i] if not callable(state[i]) else state[i]()
				inds = (*self.indices[:ndim][::-1],)
				tags = (self.tag,*self.tags,)

				data = cls(data=data,inds=inds,tags=tags)

				with context_quimb(data,self.basis[i],key=i):
					data &= self.basis[i]

				data = representation_quimb(data,contraction=True)

				state[i] = data

			options = {**dict(site_ind_id=self.ind,site_tag_id=self.tag),**dict(cyclic=self.options.get('periodic',self.options.get('cyclic',None)) if self.options is not None else None)}

			state = mps_quimb(state,**options)

		return state

	def amplitude(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Amplitude for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			state (array,tensor,network): state of class of Probability state of shape (self.D**N,self.D**N)
		'''

		parameters = self.parameters() if parameters is None else parameters() if callable(parameters) else parameters
		state = state if state is not None else state

		if state is None:
			return state

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			if L:
				basis = array([tensorprod(i) for i in permutations(*[self.basis[i] for i in where])],dtype=self.dtype)
				inverse = array([tensorprod(i) for i in permutations(*[self.inverse[i] for i in where])],dtype=self.dtype)
			else:
				basis = self.basis[self.pointer]
				inverse = self.inverse[self.pointer]

			state = einsum('u...,uv,vij->ij...',state,inverse,basis)

		elif self.architecture in ['tensor']:

			state = state.copy()

			for i in where:
				with context(self.basis[i],self.inverse[i],formats=i,indices=[{self.inds[0]:self.inds[1]},None]):
					state &= self.inverse[i] & self.basis[i]

		elif self.architecture in ['tensor_quimb']:

			state = state.copy()

			for i in where:
				with context_quimb(self.basis[i],self.inverse[i],key=i,formats=dict(inds=[{self.inds[0]:self.inds[1]},{index:index for index in self.inds}],tags=None)):
					state &= self.inverse[i] & self.basis[i]


		return state

	def operation(self,parameters=None,state=None,model=None,where=None,func=None,options=None,**kwargs):
		'''
		Operator for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class of Probability state of shape (N,self.K) or (self.K**N,)
			model (callable): model of operator with signature model(parameters,state,**kwargs) -> data
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			func (callable): operator with signature func(parameters,state,where,**kwargs) -> data (array) POVM operator of shape (self.K**N,self.K**N)
		'''

		parameters = self.parameters() if parameters is None else parameters() if callable(parameters) else parameters
		state = state if state is not None else state

		default = tuple()
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		K = self.K
		D = self.D
		ndim = 2
		letters = {letter:f'{letter}{{}}' for letter in ['i','j','k','u','v','w']}

		if L:

			if self.architecture is None or self.architecture in ['array']:
				basis = [self.basis[i] for i in where]
				inverse = [self.inverse[i] for i in where]
			elif self.architecture in ['tensor']:
				basis = [self.basis[i].array() for i in where]
				inverse = [self.inverse[i].array() for i in where]
			elif self.architecture in ['tensor_quimb']:
				basis = [representation_quimb(self.basis[i]) for i in where]
				inverse = [representation_quimb(self.inverse[i]) for i in where]

			basis = array([tensorprod(i) for i in permutations(*[i for i in basis])],dtype=self.dtype)
			inverse = array([tensorprod(i) for i in permutations(*[i for i in inverse])],dtype=self.dtype)

		else:

			if self.architecture is None or self.architecture in ['array']:
				basis = self.basis[self.pointer]
				inverse = self.inverse[self.pointer]
			elif self.architecture in ['tensor']:
				basis = self.basis[self.pointer].array()
				inverse = self.inverse[self.pointer].array()
			elif self.architecture in ['tensor_quimb']:
				basis = representation_quimb(self.basis[self.pointer])
				inverse = representation_quimb(self.inverse[self.pointer])

		if where is not None:
			samples = [K]*L

			shape = (*samples,*[*[D]*L]*2)
			basis = reshape(basis,shape)

			shape = (*samples,)*2
			inverse = reshape(inverse,shape)

			subscripts = (
				(*[letters['u'].format(i) for i in range(N) if i in where],*(letters[j].format(i) for j in ['i','j'][:ndim] for i in range(N) if i in where)),
				(*[letters['w'].format(i) for i in range(N) if i in where],*(letters[j].format(i) for j in ['j','i'][:ndim] for i in range(N) if i in where)),
				(*[letters['w'].format(i) for i in range(N) if i in where],*[letters['v'].format(i) for i in range(N) if i in where],),
				(*[letters['u'].format(i) for i in range(N) if i in where],*[letters['v'].format(i) for i in range(N) if i in where],),
				)
			shapes = ((*samples,*[*[D]*L]*2),(*samples,*[*[D]*L]*2),(*samples,)*2,)

			einsummation = einsummand(subscripts,*shapes)

		else:

			einsummation = None

		if self.architecture is None or self.architecture in ['array']:

			options = {**{option:self.options[option] for option in self.options if option not in []},**(options if options is not None else {})}

			if model is not None and where:

				shuffler = lambda state,K=K,N=N: reshape(state,[K]*N)
				_shuffler = lambda state,K=K,N=N: reshape(state,[K**N])

				subscripts = (
					(*[letters['u'].format(i) for i in range(N) if i in where],*[letters['v'].format(i) for i in range(N) if i in where]),
					(*[letters['v'].format(i) for i in range(N)],),
					(*[letters['u' if i in where else 'v'].format(i) for i in range(N)],),
					)
				shapes = ((*[K]*L,*[K]*L,),(*[K]*N,),)

				function = einsummand(subscripts,*shapes)

				options = options if options is not None else {}

				def func(parameters,state,where=where,model=model,basis=basis,inverse=inverse,einsummation=einsummation,shuffler=shuffler,_shuffler=_shuffler,function=function,options=options,**kwargs):
					return _shuffler(function(einsummation(basis,model(parameters=parameters,state=basis,**kwargs),inverse),shuffler(state)))

			else:

				options = options if options is not None else {}

				def func(parameters,state,where=where,model=model,basis=basis,inverse=inverse,options=options,**kwargs):
					return None

		elif self.architecture in ['tensor']:

			options = {**(self.options if self.options is not None else {}),**(options if options is not None else {})}

			if model is not None and where:

				options = options if options is not None else {}

				def func(parameters,state,where=where,model=model,basis=basis,inverse=inverse,einsummation=einsummation,options=options,**kwargs):
					return state(einsummation(basis,model(parameters=parameters,state=basis,**kwargs),inverse),where=where,options=options)

			else:

				options = options if options is not None else {}

				def func(parameters,state,where=where,model=model,basis=basis,inverse=inverse,options=options,**kwargs):
					return None

		elif self.architecture in ['tensor_quimb']:

			options = {**{option:self.options[option] for option in self.options if option not in ['periodic','cyclic']},**(options if options is not None else {})}

			if model is not None and where:

				options = options if options is not None else {}

				shuffler = lambda state,K=K,L=L,d=2: reshape(state,[K**L]*d)

				def func(parameters,state,where=where,model=model,basis=basis,inverse=inverse,einsummation=einsummation,shuffler=shuffler,options=options,**kwargs):
					options.update(dict(max_bond=options.pop('S',options.get('max_bond'))))
					return state.gate((einsummation(basis,model(parameters=parameters,state=basis,**kwargs),inverse)),where=where,**options)

			else:

				options = options if options is not None else {}

				def func(parameters,state,where=where,model=model,basis=basis,inverse=inverse,options=options,**kwargs):
					return None

		parameters = self.parameters() if parameters is None else parameters
		wrapper = partial
		kwargs = {}

		func = wrapper(func,parameters=parameters,where=where,model=model,basis=basis,inverse=inverse,options=options,**kwargs)

		return func

	def calculate(self,attribute=None,function=None,settings=None,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Calculate data for POVM probability measure
		Args:
			attribute (str,callable): attribute for calculation of data
			function (callable): function of data
			settings (dict): settings of data
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		if isinstance(attribute,str):
			if hasattr(self,attribute):
				attribute = getattr(self,attribute)
		elif not callable(attribute):
			attribute = state

		if isinstance(function,str):
			if hasattr(self,function):
				function = getattr(self,function)
			else:
				try:
					function = load(function)
				except:
					function = None
		elif not callable(function):
			function = None

		settings = {} if settings is None else settings
		settings.update({attr:getattr(self,attr) for attr in self if attr not in settings and not callable(getattr(self,attr))})

		if callable(attribute):
			data = attribute(parameters=parameters,state=state,where=where,func=func,options=options,**kwargs)
		else:
			data = attribute

		if callable(function):
			data = function(data,**settings)

		return data

	def where(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Indices of function
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			where (iterable[int],dict[int,int]): indices of function
			L (int): size of indices
			N (int): size of state
		'''

		if self.architecture is None or self.architecture in ['array']:
			N = int(round(log(state.size)/log(self.K)/state.ndim)) if isinstance(state,arrays) else len(state) if isinstance(state,iterables) else len(where) if where is not None else None
		elif self.architecture in ['tensor']:
			N = state.N if isinstance(state,tensors) else int(round(log(state.size)/log(self.D)/state.ndim)) if isinstance(state,arrays) else len(where) if where is not None else None
		elif self.architecture in ['tensor_quimb']:
			N = state.L if isinstance(state,tensors_quimb) else int(round(log(state.size)/log(self.D)/state.ndim)) if isinstance(state,arrays) else None
		else:
			N = None

		where = func(N) if callable(func) and where is None else func if where is None else where

		if where is None:
			where = None
		elif where is True:
			where = tuple(range(N))
		elif where is False:
			where = tuple()
		elif isinstance(where,integers):
			where = tuple(range(where))
		elif isinstance(where,floats):
			where = tuple(range(int(where*N)))
		elif isinstance(where,dicts):
			where = dict(where)
		elif isinstance(where,iterables):
			where = tuple(where)
		else:
			where = tuple(range(N))

		L = len(where) if where is not None else None

		return where,L,N

	def eig(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Eigenvalues for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (array): Eigenvalues, sorted largest to smallest, of shape (self.D**L,) or (self.K**L,)
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(sort(data)[::-1])

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		where = where if where is not None and L else None

		if isinstance(state,arrays):

			where = tuple(where) if where is not None else None

			data = eig(state,**kwargs) if where is not None else array([])

		elif isinstance(state,tensors):

			raise NotImplementedError

		elif isinstance(state,matrices_quimb):

			where = min(N-2,max(1,(min(where)-1) if min(where) > 0 else (max(where)+1))) if where is not None else None

			data = state.singular_values(where) if where is not None else array([])

			data = sqr(data)

		elif isinstance(state,tensors_quimb):

			where = tuple(self.ind.format(i) for i in where) if where is not None else None

			data = state.singular_values(where) if where is not None else array([])

			data = sqr(data)

		else:

			where = tuple(where) if where is not None else None

			data = state

		data = func(data)

		return data

	def svd(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Singular values for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (array): Singular values, sorted largest to smallest, of shape (self.D**L,) or (self.K**L,)
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(data)

		default = tuple()
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		where = where if where is not None else None

		if isinstance(state,arrays):

			where = tuple(where) if where is not None and L else None

			data = svd(state,**kwargs) if where is not None else array([])

		elif isinstance(state,tensors):

			data = state.array()

		elif isinstance(state,matrices_quimb):

			where = min(N-2,max(1,(min(where)-1) if min(where) > 0 else (max(where)+1))) if where is not None and (L) and (N-L) else None
			data = state.singular_values(where) if where is not None else array([])

		elif isinstance(state,tensors_quimb):

			where = tuple(self.ind.format(i) for i in where) if where is not None else None

			data = state.singular_values(where) if where is not None else array([])

		else:

			where = tuple(where) if where is not None else None

			data = state

		data = func(data)

		return data

	def rank(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Rank for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (array): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(nonzero(real(data)/maximum(absolute(real(data))),**kwargs))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		data = func(state)

		return data

	def entropy(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Entropy for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class of Probability of shape (N,self.D) or (self.D**N,) or (N,self.K) or (self.K**N,)
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (array,tensor,network): state of class of Probability of shape (L,self.D) or (self.D**L,) or (L,self.K) or (self.K**L,)
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(data)

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if isinstance(state,arrays):

			data = state

			data = absolute(data)
			data /= addition(data)

			data = -addition(data*log(data))

		elif isinstance(state,tensors):

			data = state.array()

			data = absolute(data)
			data /= addition(data)

			data = -addition(data*log(data))

		elif isinstance(state,matrices_quimb):

			where = min(N-2,max(1,(min(where)-1) if min(where) > 0 else (max(where)+1))) if where is not None else None

			data = state.entropy(where)

		elif isinstance(state,tensors_quimb):

			where = tuple(self.ind.format(i) for i in where) if where is not None else None

			data = state.entropy(where)

		else:

			where = tuple(where) if where is not None else None

			data = state

		data = func(data)

		return data

	def sample(self,attribute=None,function=None,settings=None,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Class sample
		Args:
			attribute (str,callable): attribute for calculation of data
			function (callable): function of data
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		if isinstance(attribute,str):
			if hasattr(self,attribute):
				attribute = getattr(self,attribute)
		elif not callable(attribute):
			attribute = self.array

		if isinstance(function,str):
			if hasattr(self,function):
				function = getattr(self,function)
			else:
				try:
					function = load(function)
				except:
					function = None
		elif not callable(function):
			function = None

		settings = {} if settings is None else settings
		settings.update({attr:getattr(self,attr) for attr in self if attr not in settings and not callable(getattr(self,attr))})

		if callable(attribute):
			data = attribute(parameters=parameters,state=state,where=where,func=func,options=options,**kwargs)
		else:
			data = attribute

		if callable(function):
			data = function(data,**settings)

		return data

	def array(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Class data
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(ravel(data)))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		where = tuple(i for i in range(N) if i not in where)

		state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

		where = tuple(i for i in range(N) if i not in where)

		if self.architecture is None or self.architecture in ['array']:

			data = state

		elif self.architecture in ['tensor']:

			data = state.array()

		elif self.architecture in ['tensor_quimb']:

			data = representation_quimb(contract_quimb(state),contraction=True)

		data = func(data)

		return data

	def state(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Class state
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(diag(data)))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		where = tuple(i for i in range(N) if i not in where)

		state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

		where = tuple(i for i in range(N) if i not in where)

		settings = dict(transformation=False)
		state = self.transform(parameters=parameters,state=state,where=where,options=options,**{**settings,**kwargs})

		if self.architecture is None or self.architecture in ['array']:

			data = array(state)

		elif self.architecture in ['tensor']:

			data = state.matrix()

		elif self.architecture in ['tensor_quimb']:

			data = representation_quimb(state,**{**dict(to=self.architecture,contraction=True),**kwargs})

		data = func(data)

		return data

	def trace(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Trace for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			state (array,tensor,network): state of class
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(data)

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = state

			K = self.K
			ndim = data.ndim

			settings = dict(
				axes = [[i] for i in range(N)],
				shape = [K,N,ndim],
				transformation=True,
				)
			_settings = dict(
				axes = [[i] for i in range(N-L)],
				shape = [K,N-L,ndim],
				transformation=False,
				)

			function = lambda data: addition(data,axis=where)

			data = shuffle(function(shuffle(data,**settings)),**_settings)

		elif self.architecture in ['tensor']:

			data = state.copy()

			for i in where:
				with context(self.ones[i],formats=i):
					data &= self.ones[i]

		elif self.architecture in ['tensor_quimb']:

			data = state.copy()

			for i in where:
				with context_quimb(self.ones[i],key=i):
					data &= self.ones[i]

		data = func(data)

		return data

	def measure(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Measure probability for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			state (array,tensor,network): state of class
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(data)

		default = {}
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = state

			K = self.K
			ndim = data.ndim

			settings = dict(
				axes = [[i] for i in range(N)],
				shape = [K,N,ndim],
				transformation=True,
				)
			_settings = dict(
				axes = [[i] for i in range(N-L)],
				shape = [K,N-L,ndim],
				transformation=False,
				)

			function = lambda data: data[tuple(slice(None) if i not in where else where[i] for i in range(N))]

			data = shuffle(function(shuffle(data,**settings)),**_settings)

		elif self.architecture in ['tensor']:

			data = state.copy()

			for i in where:
				self.zeros[i].set(array([1 if k==where[i] else 0 for k in range(self.K)],dtype=self.dtype))
				with context(self.zeros[i],formats=i):
					data &= self.zeros[i]
				self.zeros[i].set(array([0 if k==where[i] else 0 for k in range(self.K)],dtype=self.dtype))

		elif self.architecture in ['tensor_quimb']:

			data = state.copy()

			for i in where:
				self.zeros[i].modify(data=array([1 if k==where[i] else 0 for k in range(self.K)],dtype=self.dtype))
				with context_quimb(self.zeros[i],key=i):
					data &= self.zeros[i]
				self.zeros[i].modify(data=array([0 if k==where[i] else 0 for k in range(self.K)],dtype=self.dtype))

		data = func(data)

		return data

	def square(self,parameters=None,state=None,other=None,where=None,func=None,options=None,**kwargs):
		'''
		Trace of Square for POVM probability measure with respect to other POVM
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			other (array,tensor,network): state of class of Probability of shape (N,self.K) or (self.K**N,)
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(data)

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			other = state if other is None else other

			inverse = array([tensorprod(i) for i in permutations(*[self.inverse[i] for i in where])],dtype=self.dtype)

			data = einsum('...u,uv,...v->...',state,inverse,other)

		elif self.architecture in ['tensor']:

			other = state if other is None else other

			state = state.copy()
			other = other.copy()

			for i in where:
				with context(self.inverse[i],formats=i):
					state &= self.inverse[i]

			indices = {self.inds[0].format(i):self.inds[1].format(i) for i in range(N)}

			other.transform(indices={index:indices for index in other})

			state &= other

			data = state

		elif self.architecture in ['tensor_quimb']:

			other = state if other is None else other

			state = state.copy()
			other = other.copy()

			for i in where:
				with context_quimb(self.inverse[i],key=i):
					state &= self.inverse[i]

			with context_quimb(state,other,formats=dict(sites=[{self.inds[1]:self.inds[1]},{self.inds[0]:self.inds[1]}],tags=None)):

				state &= other

				data = state

		data = func(data)

		return data

	def vectorize(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Partial Trace of Vectorized Operator for POVM probability measure with respect to other POVM
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(data)

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = state

			K = self.K
			ndim = data.ndim

			settings = dict(
				axes = [[i for i in range(N) if i not in where],[i for i in range(N) if i in where]],
				shape = [K,N,ndim],
				transformation=True,
				)

			inverse = array([tensorprod(i) for i in permutations(*[self.inverse[i] for i in where])],dtype=self.dtype)

			data = shuffle(data,**settings)

			data = reshape(data,shape=(1,*data.shape)) if not (N-L) else data

			data = einsum('us,sp,vp->uv',data,inverse,data)

		elif self.architecture in ['tensor']:

			state = state.copy()
			other = state.copy()

			for i in where:
				with context(self.inverse[i],formats=i):
					state &= self.inverse[i]

			indices = {self.inds[0].format(i):self.inds[1].format(i) for i in range(N)}

			other.transform(indices={index:indices for index in other})

			state &= other

			data = state

		elif self.architecture in ['tensor_quimb']:

			state = state.copy()
			other = state.copy()

			for i in where:
				with context_quimb(self.inverse[i],key=i):
					state &= self.inverse[i]

			with context_quimb(state,other,formats=dict(sites=[{self.inds[1]:self.inds[1]},{self.inds[0]:self.inds[1]}],tags=None)):

				state &= other

				data = state

		data = func(data)

		return data

	def infidelity(self,parameters=None,state=None,other=None,where=None,func=None,options=None,**kwargs):
		'''
		Infidelity for POVM probability measure with respect to other POVM
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			other (array,tensor,network): state of class of Probability of shape (N,self.K) or (self.K**N,)
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		attr = 'infidelity_classical'

		if hasattr(self,attr):
			data = getattr(self,attr)(parameters=parameters,state=state,other=other,where=where,func=func,options=options,**kwargs)
		else:
			data = state

		return data

	def infidelity_quantum(self,parameters=None,state=None,other=None,where=None,func=None,options=None,**kwargs):
		'''
		Infidelity (Quantum) for POVM probability measure with respect to other POVM
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			other (array,tensor,network): state of class of Probability of shape (N,self.K) or (self.K**N,)
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(1 - real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			settings = dict(transformation=False)
			state = self.transform(parameters=parameters,state=state,where=where,options=options,**{**settings,**kwargs})
			other = self.transform(parameters=parameters,state=other,where=where,options=options,**{**settings,**kwargs})

			state = dot(state,other)
			data = self.eig(parameters=parameters,state=state,options=options,**kwargs)

			data = addition(sqrt(absolute(data)))

		elif self.architecture in ['tensor']:

			settings = dict(transformation=False)
			state = self.transform(parameters=parameters,state=state,where=where,options=options,**{**settings,**kwargs})
			other = self.transform(parameters=parameters,state=other,where=where,options=options,**{**settings,**kwargs})

			state = state.matrix()
			other = other.matrix()

			state = dot(state,other)
			data = self.eig(parameters=parameters,state=state,options=options,**kwargs)

			data = addition(sqrt(absolute(data)))

		elif self.architecture in ['tensor_quimb']:

			settings = dict(transformation=False)
			state = self.transform(parameters=parameters,state=state,where=where,options=options,**{**settings,**kwargs})
			other = self.transform(parameters=parameters,state=other,where=where,options=options,**{**settings,**kwargs})

			settings = dict(to=self.architecture,contraction=True)
			state = representation_quimb(state,**{**settings,**kwargs})
			other = representation_quimb(other,**{**settings,**kwargs})

			state = dot(state,other)
			data = self.eig(parameters=parameters,state=state,options=options,**kwargs)

			data = addition(sqrt(absolute(data)))

		data = func(data)

		return data

	def infidelity_classical(self,parameters=None,state=None,other=None,where=None,func=None,options=None,**kwargs):
		'''
		Infidelity (Classical) for POVM probability measure with respect to other POVM
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			other (array,tensor,network): state of class of Probability of shape (N,self.K) or (self.K**N,)
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(1 - real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			function = sqrt

			state = function(state)
			other = function(other)

			data = einsum('...u,...u->...',state,other)

		elif self.architecture in ['tensor']:

			function = sqrt
			kwargs = {}

			state = function(ravel(state.array()),**kwargs)
			other = function(ravel(other.array()),**kwargs)

			data = dot(state,other)

		elif self.architecture in ['tensor_quimb']:

			function = sqrt
			kwargs = {}

			state = contract_quimb(state,**kwargs)
			state.modify(apply=function)

			other = contract_quimb(other,**kwargs)
			other.modify(apply=function)

			data = representation_quimb(state & other,contraction=True,**kwargs)

		data = func(data)

		return data

	def infidelity_pure(self,parameters=None,state=None,other=None,where=None,func=None,options=None,**kwargs):
		'''
		Infidelity (Pure) for POVM probability measure with respect to other POVM
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			other (array,tensor,network): state of class of Probability of shape (N,self.K) or (self.K**N,)
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(1 - sqrt(absolute(real(data))))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = self.square(parameters=parameters,state=state,other=other,where=where,options=options,**kwargs)
			state = self.square(parameters=parameters,state=state,other=state,where=where,options=options,**kwargs)
			other = self.square(parameters=parameters,state=other,other=other,where=where,options=options,**kwargs)

			data = data/sqrt(state*other)

		elif self.architecture in ['tensor']:

			data = self.square(parameters=parameters,state=state,other=other,where=where,options=options,**kwargs)
			state = self.square(parameters=parameters,state=state,other=state,where=where,options=options,**kwargs)
			other = self.square(parameters=parameters,state=other,other=other,where=where,options=options,**kwargs)

			data = (data.array()/sqrt(state.array()*other.array())).item()

		elif self.architecture in ['tensor_quimb']:

			settings = dict(contraction=True)

			data = self.square(parameters=parameters,state=state,other=other,where=where,options=options,**kwargs)
			state = self.square(parameters=parameters,state=state,other=state,where=where,options=options,**kwargs)
			other = self.square(parameters=parameters,state=other,other=other,where=where,options=options,**kwargs)

			data = representation_quimb(data,**settings)/sqrt(representation_quimb(state,**settings)*representation_quimb(other,**settings))

		data = func(data)

		return data

	def norm(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Norm for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		attr = 'norm_classical'

		if hasattr(self,attr):
			data = getattr(self,attr)(parameters=parameters,state=state,where=where,func=func,options=options,**kwargs)
		else:
			data = state

		return data

	def norm_quantum(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Norm (quantum) for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(1 - real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = data.item()

		elif self.architecture in ['tensor']:

			data = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = data.array().item()

		elif self.architecture in ['tensor_quimb']:

			settings = dict(contraction=True)

			data = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = representation_quimb(data,**settings)

		data = func(data)

		return data

	def norm_classical(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Norm (Classical) for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(1 - real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = data.item()

		elif self.architecture in ['tensor']:

			data = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = data.array().item()

		elif self.architecture in ['tensor_quimb']:

			settings = dict(contraction=True)

			data = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = representation_quimb(data,**settings)

		data = func(data)

		return data

	def norm_pure(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Norm (Pure) for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(1 - sqrt(absolute(real(data))))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = self.square(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = data.item()

		elif self.architecture in ['tensor']:

			data = self.square(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = data.array().item()

		elif self.architecture in ['tensor_quimb']:

			settings = dict(contraction=True)

			data = self.square(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = representation_quimb(data,**settings)

		data = func(data)

		return data


	def entanglement(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Entanglement Entropy for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		attr = 'entanglement_classical'

		if hasattr(self,attr):
			data = getattr(self,attr)(parameters=parameters,state=state,where=where,func=func,options=options,**kwargs)
		else:
			data = state

		return data

	def entanglement_quantum(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Entanglement Entropy (Quantum) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(data)/log(self.D**L))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			settings = dict(transformation=False)
			state = self.transform(parameters=parameters,state=state,where=where,options=options,**{**settings,**kwargs})

			state = array(state)

			data = self.eig(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = self.entropy(parameters=parameters,state=data,where=where,options=options,**kwargs)

			data = data.item()

		elif self.architecture in ['tensor']:

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			settings = dict(transformation=False)
			state = self.transform(parameters=parameters,state=state,where=where,options=options,**{**settings,**kwargs})

			state = state.matrix()

			data = self.eig(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = self.entropy(parameters=parameters,state=data,where=where,options=options,**kwargs)

			data = data.item()

		elif self.architecture in ['tensor_quimb']:

			state = state.copy()

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			settings = dict(transformation=False)
			state = self.transform(parameters=parameters,state=state,where=where,options=options,**{**settings,**kwargs})

			settings = dict(to=self.architecture,contraction=True)
			state = representation_quimb(state,**{**settings,**kwargs})

			data = self.eig(parameters=parameters,state=state,options=options,**kwargs)

			data = self.entropy(parameters=parameters,state=data,where=where,options=options,**kwargs)

		data = func(data)

		return data

	def entanglement_classical(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Entanglement Entropy (Classical) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(data)/log(self.K**L))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = self.entropy(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = data.item()

		elif self.architecture in ['tensor']:

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			state = state.array().ravel()

			data = self.entropy(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = data.item()

		elif self.architecture in ['tensor_quimb']:

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			state = ravel(representation_quimb(state,contraction=True))

			data = self.entropy(parameters=parameters,state=state,where=where,options=options,**kwargs)

		data = func(data)

		return data

	def entanglement_renyi(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Entanglement Entropy (Renyi) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(1 - real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			data = self.square(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = data.item()

		elif self.architecture in ['tensor']:

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			data = self.square(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = data.array().item()

		elif self.architecture in ['tensor_quimb']:

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			settings = dict(contraction=True)

			data = self.square(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = representation_quimb(data,**settings)

		data = func(data)

		return data

	def entangling(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Entangling Power/Operator Entanglement Entropy for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		attr = 'entangling_renyi'

		if hasattr(self,attr):
			data = getattr(self,attr)(parameters=parameters,state=state,where=where,func=func,options=options,**kwargs)
		else:
			data = state

		return data

	def entangling_quantum(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Entangling Power/Operator Entanglement Entropy (Quantum) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func((1/2)*real(data)/log(self.D**L))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			where = tuple(i for i in range(N) if i not in where)

			data = self.vectorize(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			basis = array([tensorprod(i) for i in permutations(*[self.basis[i] for i in where])],dtype=self.dtype)
			inverse = array([tensorprod(i) for i in permutations(*[self.inverse[i] for i in where])],dtype=self.dtype)

			basis = reshape(basis,shape=(self.K**L,-1))

			data = einsum('uv,us,vp,si,pj->ij',data,inverse,inverse,basis,conjugate(basis))

			data /= self.vectorize(parameters=parameters,state=state,options=options,**kwargs)

			data = self.eig(parameters=parameters,state=data,**kwargs)

			data = self.entropy(parameters=parameters,state=data,where=where,options=options,**kwargs)

			data = data.item()

		elif self.architecture in ['tensor']:

			where = tuple(i for i in range(N) if i not in where)

			data = self.vectorize(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			for i in where:
				with context(self.inverse[i],self.basis[i],formats=i,indices=[{self.inds[0]:self.inds[0],self.inds[1]:self.symbol[0]},{self.inds[0]:self.symbol[0],self.indices[0]:self.symbols[0],self.indices[1]:self.symbols[1]}]):
					data &= self.inverse[i] & self.basis[i]
				with context(self.inverse[i],self.basis[i],formats=i,indices=[{self.inds[0]:self.inds[1],self.inds[1]:self.symbol[1]},{self.inds[0]:self.symbol[1],self.indices[0]:self.symbols[2],self.indices[1]:self.symbols[3]}]):
					data &= self.inverse[i] & self.basis[i].transform(conj=True)

			indices = [[symbol.format(i) for i in where for symbol in symbols] for symbols in [self.symbols[:2],self.symbols[2:4]]]
			data = data.matrix(indices=indices)

			data /= self.vectorize(parameters=parameters,state=state,options=options,**kwargs).array()

			data = self.eig(parameters=parameters,state=data,**kwargs)

			data = self.entropy(parameters=parameters,state=data,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor_quimb']:

			where = tuple(i for i in range(N) if i not in where)

			data = self.vectorize(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			for i in where:
				with context_quimb(self.inverse[i],self.basis[i],key=i,formats=dict(inds=[{self.inds[0]:self.inds[0],self.inds[1]:self.symbol[0]},{self.inds[0]:self.symbol[0],self.indices[0]:self.symbols[0],self.indices[1]:self.symbols[1]}],tags=None)):
					data &= self.inverse[i] & self.basis[i]
				with context_quimb(self.inverse[i],self.basis[i],key=i,formats=dict(inds=[{self.inds[0]:self.inds[1],self.inds[1]:self.symbol[1]},{self.inds[0]:self.symbol[1],self.indices[0]:self.symbols[2],self.indices[1]:self.symbols[3]}],tags=None)):
					data &= self.inverse[i] & self.basis[i].conj()

			settings = {}
			data = contract_quimb(data,**settings)

			settings = dict(where={self.symbols[j].format(j):(*(symbol.format(i) for i in where for symbol in self.symbols[2*j:2*(j+1)]),) for j in range(2)})
			data = fuse_quimb(data,**settings)

			settings = dict(contraction=True)
			data = representation_quimb(data,**settings)

			settings = {}
			data /= contract_quimb(self.vectorize(parameters=parameters,state=state,options=options,**kwargs),**settings)

			data = self.eig(parameters=parameters,state=data,**kwargs)

			data = self.entropy(parameters=parameters,state=data,where=where,options=options,**kwargs)

		data = func(data)

		return data

	def entangling_classical(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Entangling Power/Operator Entanglement Entropy (Classical) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func((1/2)*real(data)/log(self.K**L))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			where = tuple(i for i in range(N) if i not in where)

			data = self.vectorize(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			data /= self.vectorize(parameters=parameters,state=state,options=options,**kwargs)

			data = self.eig(parameters=parameters,state=data,**kwargs)

			data = self.entropy(parameters=parameters,state=data,where=where,options=options,**kwargs)

			data = data.item()

		elif self.architecture in ['tensor']:

			where = tuple(i for i in range(N) if i not in where)

			data = self.vectorize(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			data = data.matrix()

			data /= self.vectorize(parameters=parameters,state=state,options=options,**kwargs).array()

			data = self.eig(parameters=parameters,state=data,**kwargs)

			data = self.entropy(parameters=parameters,state=data,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor_quimb']:

			where = tuple(i for i in range(N) if i not in where)

			data = self.vectorize(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			settings = {}
			data = contract_quimb(data,**settings)

			settings = dict(where={self.inds[j]:(*(self.inds[j].format(i) for i in where),) for j in range(2)})
			data = fuse_quimb(data,**settings)

			settings = dict(contraction=True)
			data = representation_quimb(data,**settings)

			settings = {}
			data /= contract_quimb(self.vectorize(parameters=parameters,state=state,options=options,**kwargs),**settings)

			data = self.eig(parameters=parameters,state=data,**kwargs)

			data = self.entropy(parameters=parameters,state=data,where=where,options=options,**kwargs)

		data = func(data)

		return data

	def entangling_renyi(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Entangling Power/Operator Entanglement Entropy (Renyi) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(1 - real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			where = tuple(i for i in range(N) if i not in where)

			data = self.vectorize(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			inverse = array([tensorprod(i) for i in permutations(*[self.inverse[i] for i in where])],dtype=self.dtype)

			data = einsum('uv,up,vs,sp->',data,inverse,inverse,data)

			data /= self.vectorize(parameters=parameters,state=state,options=options,**kwargs)**2

			data = data.item()

		elif self.architecture in ['tensor']:

			where = tuple(i for i in range(N) if i not in where)

			data = self.vectorize(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			other = data.copy()

			indices = {**{self.inds[0].format(i):self.symbol[0].format(i) for i in range(N)},**{self.inds[1].format(i):self.symbol[1].format(i) for i in range(N)}}

			other.transform(indices={index:indices for index in other})

			for i in where:
				with context(self.inverse[i],formats=i,indices={self.inds[1]:self.symbol[1],self.inds[0]:self.inds[0]}):
					data &= self.inverse[i]
				with context(self.inverse[i],formats=i,indices={self.inds[1]:self.symbol[0],self.inds[0]:self.inds[1]}):
					other &= self.inverse[i]

			data &= other

			data = data.array().item()

			data /= self.vectorize(parameters=parameters,state=state,options=options,**kwargs).array().item()**2

		elif self.architecture in ['tensor_quimb']:

			where = tuple(i for i in range(N) if i not in where)

			data = self.vectorize(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			settings = {}
			data = contract_quimb(data,**settings)

			other = data.copy()

			with context_quimb(data,other,key=where,formats=dict(inds=[{self.inds[0]:self.inds[0],self.inds[1]:self.inds[1]},{self.inds[0]:self.symbol[0],self.inds[1]:self.symbol[1]}],tags=None)):

				for i in where:
					with context_quimb(self.inverse[i],key=i,formats=dict(inds=[{self.inds[0]:self.inds[0],self.inds[1]:self.symbol[1]}],tags=None)):
						data &= self.inverse[i]
					with context_quimb(self.inverse[i],key=i,formats=dict(inds=[{self.inds[0]:self.inds[1],self.inds[1]:self.symbol[0]}],tags=None)):
						other &= self.inverse[i]

				data &= other

				settings = dict(contraction=True)
				data = representation_quimb(data,**settings)

				settings = {}
				data /= contract_quimb(self.vectorize(parameters=parameters,state=state,options=options,**kwargs),**settings)**2

		data = func(data)

		return data

	def mutual(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Mutual Information for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		attr = 'mutual_renyi'

		if hasattr(self,attr):
			data = getattr(self,attr)(parameters=parameters,state=state,where=where,func=func,options=options,**kwargs)
		else:
			data = state

		return data

	def mutual_quantum(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Mutual Information (Quantum) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(data)/log(self.D**(2*N)))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			L = len(where)
			tmp = self.entanglement_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.D**L)
			data += tmp

			where = tuple(i for i in range(N) if i not in where)
			L = len(where)
			tmp = self.entanglement_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.D**L)
			data += tmp

			where = tuple(i for i in range(N))
			L = len(where)
			tmp = self.entanglement_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.D**L)
			data -= tmp

			data = data.item()

		elif self.architecture in ['tensor']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			L = len(where)
			tmp = self.entanglement_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.D**L)
			data += tmp

			where = tuple(i for i in range(N) if i not in where)
			L = len(where)
			tmp = self.entanglement_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.D**L)
			data += tmp

			where = tuple(i for i in range(N))
			L = len(where)
			tmp = self.entanglement_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.D**L)
			data -= tmp

		elif self.architecture in ['tensor_quimb']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			L = len(where)
			tmp = self.entanglement_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.D**L)
			data += tmp

			where = tuple(i for i in range(N) if i not in where)
			L = len(where)
			tmp = self.entanglement_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.D**L)
			data += tmp

			where = tuple(i for i in range(N))
			L = len(where)
			tmp = self.entanglement_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.D**L)
			data -= tmp

		data = func(data)

		return data

	def mutual_measure(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Mutual Information (Measure) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(data)/log(self.D**(2*N)))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			tmp = state
			data += self.entanglement_quantum(parameters=parameters,state=tmp,where=where,options=options,**kwargs)*log(self.D**L)

			where = tuple(i for i in range(N) if i not in where)

			for index in permutations(*[range(self.K)]*(N-L)):

				index = dict(zip(where,index))
				tmp = self.measure(parameters=parameters,state=state,where=index,**kwargs)

				index = range(L)
				norm = self.trace(parameters=parameters,state=tmp,where=index,**kwargs)

				index = range(L)
				settings = dict(transformation=False)
				tmp = self.transform(parameters=parameters,state=tmp,where=index,**{**settings,**kwargs})

				tmp /= norm

				tmp = self.eig(parameters=parameters,state=tmp,**kwargs)

				index = tuple(i for i in range(N) if i not in where)
				tmp = self.entropy(parameters=parameters,state=tmp,where=index,**kwargs)

				data -= norm*tmp

			data = data.item()

		elif self.architecture in ['tensor']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			tmp = state
			data += self.entanglement_quantum(parameters=parameters,state=tmp,where=where,options=options,**kwargs)*log(self.D**L)

			where = tuple(i for i in range(N) if i not in where)

			for index in permutations(*[range(self.K)]*(N-L)):

				index = dict(zip(where,index))
				tmp = self.measure(parameters=parameters,state=state,where=index,**kwargs)

				index = tuple(i for i in range(N) if i not in where)
				norm = self.trace(parameters=parameters,state=tmp,where=index,**kwargs)

				index = tuple(i for i in range(N) if i not in where)
				settings = dict(transformation=False)
				tmp = self.transform(parameters=parameters,state=tmp,where=index,**{**settings,**kwargs})

				tmp = tmp.matrix()

				norm = norm.array()

				tmp /= norm

				tmp = self.eig(parameters=parameters,state=tmp,**kwargs)

				index = tuple(i for i in range(N) if i not in where)
				tmp = self.entropy(parameters=parameters,state=tmp,where=index,**kwargs)

				data -= norm*tmp

			data = data.item()

		elif self.architecture in ['tensor_quimb']:

			data = 0

			index = tuple(i for i in range(N) if i in where)
			tmp = state
			data += self.entanglement_quantum(parameters=parameters,state=tmp,where=index,**kwargs)*log(self.D**L)

			where = tuple(i for i in range(N) if i not in where)

			for index in permutations(*[range(self.K)]*(N-L)):

				index = dict(zip(where,index))
				tmp = self.measure(parameters=parameters,state=state,where=index,**kwargs)

				index = tuple(i for i in range(N) if i not in where)
				norm = self.trace(parameters=parameters,state=tmp,where=index,**kwargs)

				index = tuple(i for i in range(N) if i not in where)
				settings = dict(transformation=False)
				tmp = self.transform(parameters=parameters,state=tmp,where=index,**{**settings,**kwargs})

				settings = dict(to=self.architecture,contraction=True)
				tmp = representation_quimb(tmp,**{**settings,**kwargs})

				settings = dict(to=self.architecture,contraction=True)
				norm = representation_quimb(norm,**{**settings,**kwargs})

				tmp /= norm

				tmp = self.eig(parameters=parameters,state=tmp,**kwargs)

				tmp = self.entropy(parameters=parameters,state=tmp,where=index,**kwargs)

				data -= norm*tmp

		data = func(data)

		return data

	def mutual_classical(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Mutual Information (Classical) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(data)/log(self.K**L))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			L = len(where)
			tmp = self.entanglement_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.K**L)
			data += tmp

			where = tuple(i for i in range(N) if i not in where)
			L = len(where)
			tmp = self.entanglement_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.K**L)
			data += tmp

			where = tuple(i for i in range(N))
			L = len(where)
			tmp = self.entanglement_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.K**L)
			data -= tmp

			data = data.item()

		elif self.architecture in ['tensor']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			L = len(where)
			tmp = self.entanglement_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.K**L)
			data += tmp

			where = tuple(i for i in range(N) if i not in where)
			L = len(where)
			tmp = self.entanglement_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.K**L)
			data += tmp

			where = tuple(i for i in range(N))
			L = len(where)
			tmp = self.entanglement_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.K**L)
			data -= tmp

			data = data.item()

		elif self.architecture in ['tensor_quimb']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			L = len(where)
			tmp = self.entanglement_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.K**L)
			data += tmp

			where = tuple(i for i in range(N) if i not in where)
			L = len(where)
			tmp = self.entanglement_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.K**L)
			data += tmp

			where = tuple(i for i in range(N))
			L = len(where)
			tmp = self.entanglement_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)*log(self.K**L)
			data -= tmp

		data = func(data)

		return data

	def mutual_renyi(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Mutual Information (Renyi) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func((1/2)*real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			L = len(where)
			tmp = self.entanglement_renyi(parameters=parameters,state=state,where=where,options=options,**kwargs)
			data += tmp

			where = tuple(i for i in range(N) if i not in where)
			L = len(where)
			tmp = self.entanglement_renyi(parameters=parameters,state=state,where=where,options=options,**kwargs)
			data += tmp

			where = tuple(i for i in range(N))
			L = len(where)
			tmp = self.entanglement_renyi(parameters=parameters,state=state,where=where,options=options,**kwargs)
			data -= tmp

			data = data.item()

		elif self.architecture in ['tensor']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			L = len(where)
			tmp = self.entanglement_renyi(parameters=parameters,state=state,where=where,options=options,**kwargs)
			data += tmp

			where = tuple(i for i in range(N) if i not in where)
			L = len(where)
			tmp = self.entanglement_renyi(parameters=parameters,state=state,where=where,options=options,**kwargs)
			data += tmp

			where = tuple(i for i in range(N))
			L = len(where)
			tmp = self.entanglement_renyi(parameters=parameters,state=state,where=where,options=options,**kwargs)
			data -= tmp

			data = data.item()

		elif self.architecture in ['tensor_quimb']:

			data = 0

			where = tuple(i for i in range(N) if i in where)
			L = len(where)
			tmp = self.entanglement_renyi(parameters=parameters,state=state,where=where,options=options,**kwargs)
			data += tmp

			where = tuple(i for i in range(N) if i not in where)
			L = len(where)
			tmp = self.entanglement_renyi(parameters=parameters,state=state,where=where,options=options,**kwargs)
			data += tmp

			where = tuple(i for i in range(N))
			L = len(where)
			tmp = self.entanglement_renyi(parameters=parameters,state=state,where=where,options=options,**kwargs)
			data -= tmp

		data = func(data)

		return data

	def discord(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Discord for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		attr = 'discord_quantum'

		if hasattr(self,attr):
			data = getattr(self,attr)(parameters=parameters,state=state,where=where,func=func,options=options,**kwargs)
		else:
			data = state

		return data

	def discord_quantum(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Discord (Quantum) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = self.mutual_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs) - self.mutual_measure(parameters=parameters,state=state,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor']:

			data = self.mutual_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs) - self.mutual_measure(parameters=parameters,state=state,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor_quimb']:

			data = self.mutual_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs) - self.mutual_measure(parameters=parameters,state=state,where=where,options=options,**kwargs)

		data = func(data)

		return data

	def discord_classical(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Discord (Classical) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = 0

		elif self.architecture in ['tensor']:

			data = 0

		elif self.architecture in ['tensor_quimb']:

			data = 0

		data = func(data)

		return data

	def discord_renyi(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Discord (Renyi) for POVM probability measure with respect to where
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = 0

		elif self.architecture in ['tensor']:

			data = 0

		elif self.architecture in ['tensor_quimb']:

			data = 0

		data = func(data)

		return data

	def spectrum(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Spectrum for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		attr = 'spectrum_classical'

		if hasattr(self,attr):
			data = getattr(self,attr)(parameters=parameters,state=state,other=other,where=where,func=func,options=options,**kwargs)
		else:
			data = state

		return data

	def spectrum_quantum(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Spectrum (Quantum) for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			settings = dict(transformation=False)
			state = self.transform(parameters=parameters,state=state,where=where,options=options,**{**settings,**kwargs})

			where = tuple(i for i in range(N) if i in where)

			data = self.eig(parameters=parameters,state=state,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor']:

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			settings = dict(transformation=False)
			state = self.transform(parameters=parameters,state=state,where=where,options=options,**{**settings,**kwargs})

			state = state.matrix()

			where = tuple(i for i in range(N) if i in where)

			data = self.eig(parameters=parameters,state=state,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor_quimb']:

			state = state.copy()

			where = tuple(i for i in range(N) if i not in where)

			state = self.trace(parameters=parameters,state=state,where=where,options=options,**kwargs)

			where = tuple(i for i in range(N) if i not in where)

			settings = dict(transformation=False)
			state = self.transform(parameters=parameters,state=state,where=where,options=options,**{**settings,**kwargs})

			settings = dict(to=self.architecture,contraction=True)
			state = representation_quimb(state,**{**settings,**kwargs})

			where = tuple(i for i in range(N) if i in where)

			data = self.eig(parameters=parameters,state=state,where=where,options=options,**kwargs)

		data = func(data)

		return data

	def spectrum_classical(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Spectrum (Classical) for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(real(data))

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			K = self.K
			ndim = state.ndim

			settings = dict(
				axes = [[i for i in range(N) if i in where],[i for i in range(N) if i not in where]],
				shape = [K,N,ndim],
				transformation=True,
				)

			state = shuffle(state,**settings)

			data = self.svd(parameters=parameters,state=state,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor']:

			state = state.array().ravel()

			K = self.K
			ndim = state.ndim

			settings = dict(
				axes = [[i for i in range(N) if i in where],[i for i in range(N) if i not in where]],
				shape = [K,N,ndim],
				transformation=True,
				)

			state = shuffle(state,**settings)

			data = self.svd(parameters=parameters,state=state,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor_quimb']:

			data = self.svd(parameters=parameters,state=state,where=where,options=options,**kwargs)

		data = func(data)

		return data

	def rank_quantum(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Rank (Quantum) for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(data)

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = self.spectrum_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = self.rank(parameters=parameters,state=data,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor']:

			data = self.spectrum_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = self.rank(parameters=parameters,state=data,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor_quimb']:

			data = self.spectrum_quantum(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = self.rank(parameters=parameters,state=data,where=where,options=options,**kwargs)

		data = func(data)

		return data

	def rank_classical(self,parameters=None,state=None,where=None,func=None,options=None,**kwargs):
		'''
		Rank (Classical) for POVM probability measure
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			where (float,int,iterable[int]): indices of function
			func (callable): function of function
			options (dict): options of function
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		func = (lambda data:data) if not callable(func) else func
		func = lambda data,func=func: func(data)

		default = range
		where,L,N = self.where(parameters=parameters,state=state,where=where,func=default,options=options)

		if self.architecture is None or self.architecture in ['array']:

			data = self.spectrum_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = self.rank(parameters=parameters,state=data,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor']:

			data = self.spectrum_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = self.rank(parameters=parameters,state=data,where=where,options=options,**kwargs)

		elif self.architecture in ['tensor_quimb']:

			data = self.spectrum_classical(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data = self.rank(parameters=parameters,state=data,where=where,options=options,**kwargs)

		data = func(data)

		return data


class MPS(mps):
	'''
	Matrix Product State class
	Args:
		data (iterable,int,str,callable,array,object): Tensor data
		parameters (array,dict): Tensor parameters
		N (int): Tensor system size
		D (int): Tensor physical bond dimension
		S (int): Tensor virtual bond dimension
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional keyword arguments
	'''

	N = None
	D = None
	S = None

	def __init__(self,data=None,parameters=None,N=None,D=None,S=None,system=None,**kwargs):

		defaults = dict()

		setter(kwargs,dict(system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,defaults,delimiter=delim,default=False)

		if isinstance(data,(*strings,*dicts)):
			basis = {
				**{attr: Basis.state for attr in ['state','psi']},
				**{attr: Basis.state for attr in ['haar']},
				**{attr: Basis.rand for attr in ['random','rand']},
				**{attr: Basis.zero for attr in ['zero','zeros','0']},
				**{attr: Basis.one for attr in ['one','ones','1']},
				**{attr: Basis.plus for attr in ['plus','+']},
				**{attr: Basis.minus for attr in ['minus','-']},
				**{attr: Basis.plusi for attr in ['plusi','+i']},
				**{attr: Basis.minusi for attr in ['minusi','-i']},
				**{attr: Basis.ghz for attr in ['ghz']},
			}
			options = dict(D=D,**kwargs)
			data = [data]*N if isinstance(data,str) else [data[i] for i in data] if isinstance(data,dicts) else [i for i in data]
			data = [basis.get(i)(**Basis.opts(basis.get(i),options)) if isinstance(i,str) else i for i in data]

		super().__init__(data=data,parameters=parameters,N=N,D=D,S=S,**kwargs)

		return


if backend in ['quimb']:

	from src.utils import tensor_quimb,network_quimb,mps_quimb,representation_quimb,contract_quimb,fuse_quimb,context_quimb
	from src.utils import tensors_quimb,matrices_quimb,objects_quimb

	objects = (*objects,*objects_quimb)

	class MPS_quimb(mps_quimb):
		'''
		Matrix Product State class
		Args:
			data (iterable,int,str,callable,array,object): Tensor data
			parameters (array,dict): Tensor parameters
			N (int): Tensor system size
			D (int): Tensor physical bond dimension
			S (int): Tensor virtual bond dimension
			kwargs (dict): Tensor keyword arguments
		'''
		def __new__(cls,data,parameters=None,N=None,D=None,S=None,**kwargs):

			updates = {
				'periodic':(
					(lambda attr,value,kwargs:'cyclic'),
					(lambda attr,value,kwargs: (value is True) and N is not None and N>2)
					),
				}

			kwargs.update(dict(data=data,L=N))
			for attr in updates:
				if attr not in kwargs:
					continue
				attrs,values = updates[attr]
				attr,value = attr,kwargs.pop(attr)
				attr,value = attrs(attr,value,kwargs),values(attr,value,kwargs)
				if value is None:
					continue
				kwargs[attr] = value

			if data is None:
				kwds = {attr: kwargs.get(attr) for attr in ['random','seed','bounds','dtype']}
				def data(shape,*args,**kwargs):
					data = Basis.state(shape=shape,**kwds)
					return
				kwargs.update(dict(phys_dim=D,bond_dim=S))
				kwargs = {attr: kwargs.get(attr) for attr in ['L','phys_dim','bond_dim','cyclic'] if attr in kwargs}
			elif isinstance(data,(str,*dicts,*iterables)):
				basis = {
					**{attr: Basis.state for attr in ['state','psi']},
					**{attr: Basis.state for attr in ['haar']},
					**{attr: Basis.rand for attr in ['random','rand']},
					**{attr: Basis.zero for attr in ['zero','zeros','0']},
					**{attr: Basis.one for attr in ['one','ones','1']},
					**{attr: Basis.plus for attr in ['plus','+']},
					**{attr: Basis.minus for attr in ['minus','-']},
					**{attr: Basis.plusi for attr in ['plusi','+i']},
					**{attr: Basis.minusi for attr in ['minusi','-i']},
					**{attr: Basis.ghz for attr in ['ghz']},
				}
				options = dict(D=D,**kwargs)
				data = [data]*N if isinstance(data,str) else [data[i] for i in data] if isinstance(data,dicts) else [i for i in data]
				data = [basis.get(i)(**Basis.opts(basis.get(i),options)) if isinstance(i,str) else i for i in data]
				kwargs = {attr: kwargs.get(attr) for attr in ['L','cyclic'] if attr in kwargs}
			elif isinstance(data,integers):
				kwargs.update(dict(phys_dim=D,bond_dim=S))
				kwargs = {attr: kwargs.get(attr) for attr in ['L','cyclic','dtype'] if attr in kwargs}
			else:
				kwargs.update(dict(phys_dim=D,bond_dim=S))
				kwargs = {attr: kwargs.get(attr) for attr in ['L','cyclic','dtype'] if attr in kwargs}

			kwargs.update(dict(data=data))

			self = super().__new__(cls,**kwargs)

			return self

def trotter(iterable=None,p=None,verbose=False):
	'''
	Trotterized iterable for order p or parameters for order p
	Args:
		iterable (iterable): Iterable
		p (int): Order of trotterization
		verbose (bool,int,str): Verbosity of function
	Returns:
		iterables (iterable,scalar): Trotterized iterable for order p or parameters for order p if iterable is None
	'''
	P = 2
	if p is None and iterable is None:
		return p
	elif p is None:
		return iterable
	elif not isinstance(p,integers) or (p > P):
		raise NotImplementedError('p = %r !< %d Not Implemented'%(p,P))

	if iterable is None:
		options = {i:1/p for i in range(P+1)}
		iterables = options[p]
	else:
		options = {i:[slice(None,None,(-1)**i)] for i in range(P)}
		iterables = []
		for i in range(p):
			for indices in options[i]:
				iterables += iterable[indices]

	return iterables


def compile(data,state=None,conj=False,size=None,compilation=None,verbose=False):
	'''
	Compile data and state
	Args:
		data (iterable[Object],dict[int,Object]): data of shape (length,)
		state (array,Object): state of shape (n,) or (n,n)
		conj (bool): conjugate
		size (int): Number of contractions of data and state
		compilation (dict[str,int,bool]): Compilation options, allowed keys
			trotter (int): Number of cycles of each contraction of data and state
			simplify (bool): Simplification of data to most basic form
		verbose (bool,int,str): Verbosity of function
	Returns:
		data (iterable[Object]): Compiled data of shape (length,)
	'''

	data = {j: data[i] if isinstance(data,dict) else i for j,i in enumerate(data)} if data is not None else {}
	state = state() if callable(state) else state
	conj = conj if conj is None else False
	size = size if size is not None else 1
	compilation = Dict({**dict(trotter=None,simplify=False),**(compilation if isinstance(compilation,dict) else {})})

	# Update data
	for i in data:

		if data[i] is None:
			continue

		kwargs = dict(
			parameters=dict(parameters=dict(trotter=trotter(p=compilation.trotter)) if (data[i].unitary) else None),
			)
		data[i].init(**kwargs)

	# Filter None data
	boolean = lambda i=None,data=None: (data is not None) and (data[i] is not None) and (not data[i].null())

	data = {i: data[i] for i in data if boolean(i,data)}

	# Filter constant data
	if compilation.simplify:

		boolean = lambda i=None,data=None: (data[i] is not None) and (not data[i].null()) and (not data[i].variable) and (data[i].unitary)

		obj = {i: data[i] for i in data if boolean(i,data)}

		if len(obj)>1:

			for obj in splitter(obj):
				if len(obj) < 2:
					continue

				j = min(obj)

				for i in obj:
					if i == j:
						continue
					data[j] @= data.pop(i)

		data = {j: data[i] if isinstance(data,dict) else i for j,i in enumerate(data)} if data is not None else {}

	# Filter trotterized data
	boolean = lambda i=None,data=None: (data[i] is not None) and (not data[i].null()) and (data[i].unitary)

	obj = [j
		for i in interleaver(
		[trotter(i,p=compilation.trotter) for i in splitter([i for i in data if boolean(i,data)])],
		[i for i in splitter([i for i in data if not boolean(i,data)])])
		for j in i]

	data = [data[i] for i in obj]

	return data

def variables(data,state=None,conj=False,size=None,compilation=None,verbose=False):
	'''
	Get indexes of variable parameters of data and state,
	with positive integers representing parameter indexes for that data element,
	and negative integers representing no parameter indexes for that data element
	Args:
		data (iterable[Object],dict[int,Object]): data of shape (length,)
		state (array,Object): state of shape (n,) or (n,n)
		conj (bool): conjugate
		size (int): Number of contractions of data and state
		compilation (dict[str,int,bool]): Compilation options, allowed keys
			trotter (int): Number of cycles of each contraction of data and state
			simplify (bool): Simplification of data to most basic form
		verbose (bool,int,str): Verbosity of function
	Returns:
		indexes (iterable[int]): Indices of variable parameters data and state of shape (length,)
		shape (iterable[int]): Shape of variable parameters
	'''

	data = {j: data[i] if isinstance(data,dict) else i for j,i in enumerate(data)} if data is not None else {}
	state = state() if callable(state) else state
	conj = conj if conj is None else False
	size = size if size is not None else 1
	compilation = compilation if compilation is not None else None

	boolean = lambda i=None,data=None: (data[i] is not None) and (data[i].variable) and (data[i].parameters is not None) and (data[i].parameters.indices is not None)
	default = -1

	length = len(set(([data[i].parameters.indices for i in data if boolean(i,data)])))

	shape = (size*length,)

	indexes = [j*length + data[i].parameters.indices if boolean(i,data) else default for j in range(size) for i in data]

	return indexes,shape




def scheme(data,parameters=None,state=None,conj=False,size=None,compilation=None,verbose=False):
	'''
	Contract data and state
	Args:
		data (iterable[Object],dict[int,Object]): data of shape (length,)
		parameters (array,Parameters): parameters of data
		state (array,Object): state of shape (n,) or (n,n)
		conj (bool): conjugate
		size (int): Number of contractions of data and state
		compilation (dict[str,int,bool]): Compilation options, allowed keys
			trotter (int): Number of cycles of each contraction of data and state
			simplify (bool): Simplification of data to most basic form
		verbose (bool,int,str): Verbosity of function
	Returns:
		func (callable): contract data with signature func(parameters,state,indices,**kwargs)
	'''

	state = state() if callable(state) else state
	parameters = parameters() if callable(parameters) else parameters
	conj = conj if conj is None else False
	size = size if size is not None else 1
	compilation = compilation if compilation is not None else None

	length = len(data) if data is not None else 1
	indices = (0,size*length)

	D = min(data[i].D for i in data if data[i] is not None) if data else None
	locality = len(set(j for i in data if data[i] is not None for j in data[i].where)) if data else None
	dtype = data[0].dtype if data else None

	if parameters is not None and len(parameters):
		def function(parameters,state=state,indices=indices,**kwargs):
			return switch(indices%length,data,parameters[indices//length],state,**kwargs)
	else:
		def function(parameters,state=state,indices=indices,**kwargs):
			return switch(indices%length,data,parameters,state,**kwargs)

	obj = state() if state is not None and state() is not None else state.identity() if data else None

	data = compile(data,state=state,conj=conj,size=size,compilation=compilation,verbose=verbose)

	length = len(data)

	wrapper = jit
	kwargs = {}

	data = [wrapper(data[i],**kwargs) for i in range(length)] # TODO: Time/M-dependent constant data/parameters

	def func(parameters,state=state,indices=indices,**kwargs):

		def func(i,out):
			return function(parameters,out,indices=i,**kwargs)

		state = obj if state is None else state
		return forloop(*indices,func,state,**kwargs)

	func = wrapper(func,**kwargs)

	return func




def gradient_scheme(data,parameters=None,state=None,conj=False,size=None,compilation=None,verbose=False):
	'''
	Contract gradient of data and state
	Args:
		data (iterable[Object],dict[int,Object]): data of shape (length,)
		parameters (array,Parameters): parameters of data
		state (array,Object): state of shape (n,) or (n,n)
		conj (bool): conjugate
		size (int): Number of contractions of data and state
		compilation (dict[str,int,bool]): Compilation options, allowed keys
			trotter (int): Number of cycles of each contraction of data and state
			simplify (bool): Simplification of data to most basic form
		verbose (bool,int,str): Verbosity of function
	Returns:
		func (callable): contract gradient with signature func(parameters,state,indices,**kwargs)
	'''

	state = state() if callable(state) else state
	parameters = parameters() if callable(parameters) else parameters
	conj = conj if conj is None else False
	size = size if size is not None else 1
	compilation = compilation if compilation is not None else None

	length = len(data)
	indices = (0,size*length)

	D = min(data[i].D for i in data if data[i] is not None) if data else None
	locality = len(set(j for i in data if data[i] is not None for j in data[i].where)) if data else None
	dtype = data[0].dtype if data else None

	function = scheme(data,parameters=parameters,state=state,conj=conj,size=size,compilation=compilation)

	if parameters is not None and len(parameters):
		def gradient(parameters,state=state,indices=indices,**kwargs):
			return switch(indices%length,grad,parameters[indices//length],state,**kwargs)
	else:
		def gradient(parameters,state=state,indices=indices,**kwargs):
			return switch(indices%length,grad,parameters,state,**kwargs)

	obj = state() if state is not None and state() is not None else state.identity() if data else None

	data = compile(data,state=state,conj=conj,size=size,compilation=compilation)

	indexes,shape = variables(data,state=state,conj=conj,size=size,compilation=compilation)
	length = len(data)
	indices = (0,size*length)

	wrapper = jit
	kwargs = {}

	data,grad,indexes = (
		[wrapper(data[i],**kwargs) for i in range(length)],
		[wrapper(data[i].grad,**kwargs) for i in range(length)],
		array(indexes)
		)

	def true(i,out,parameters,state,**kwargs):

		obj = function(parameters,state,indices=(0,i),**kwargs)

		obj = gradient(parameters,obj,indices=i,**kwargs)

		obj = function(parameters,obj,indices=(i+1,size*length),**kwargs)

		out = inplace(out,indexes[i],obj,'add')

		return out

	def false(i,out,parameters,state,**kwargs):
		return out

	true = wrapper(true,**kwargs)
	false = wrapper(false,**kwargs)

	def func(parameters,state=state,indices=indices,**kwargs):

		def func(i,out):
			return cond(indexes[i]>=0,true,false,i,out,parameters,state,**kwargs)

		state = obj if state is None else state

		out = zeros((*shape,*state.shape),dtype=state.dtype)

		return forloop(*indices,func,out)

	func = wrapper(func,**kwargs)

	return func


class Object(System):
	'''
	Base class for Quantum Objects
	Args:
		data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) N-length delimiter-separated string of operators 'X_Y_Z' or N-length iterable of operator strings['X','Y','Z']
		where (iterable[int]): location of local operators, i.e) nearest neighbour
		string (str): string label of operator
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional system keyword arguments
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		}

	defaults = dict(
		data=None,operator=None,where=None,string=None,system=None,
		state=None,parameters=None,conj=False,
		tensor=None,local=None,locality=None,number=None,variable=None,constant=None,symmetry=None,hermitian=None,unitary=None,
		shape=None,size=None,ndim=None,dtype=None,
		identity=None,
		space=None,time=None,lattice=None,
		func=None,gradient=None,
		contract=None,gradient_contract=None
		)

	def __init__(self,data=None,operator=None,where=None,string=None,system=None,**kwargs):

		setter(kwargs,dict(data=data,operator=operator,where=where,string=string,system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)

		if isinstance(data,self.__class__) or isinstance(operator,self.__class__):
			if isinstance(data,self.__class__):
				for attr in data:
					if attr not in kwargs:
						setattr(self,attr,getattr(data,attr))
			if isinstance(operator,self.__class__):
				for attr in operator:
					if attr not in kwargs:
						setattr(self,attr,getattr(operator,attr))

			return

		# Set attributes
		#	N: size of system acted on by locality number of non-local operators with indices where
		#	tensor (bool,callable): whether operator has tensor shape, or callable with signature tensor(data,N,D,d)
		#	local (bool): whether operator acts locally
		# 	locality (int): number of indices acted on locally, non-trivially by operator
		# 	number (int): number of operators within class
		#	where (iterable[int]): indices of local action within space of size N


		operator = self.operator if self.operator is not None else None
		where = self.where if self.where is None or not isinstance(self.where,integers) else [self.where]
		string = self.string

		tensor = self.tensor if self.tensor is not None else None
		local = self.local if self.local is not None else None
		locality = self.locality if self.locality is not None else None
		number = self.number if self.number is not None else None

		variable = self.variable if self.variable is not None else None
		constant = self.constant if self.constant is not None else None
		symmetry = self.symmetry if self.symmetry is not None else None
		hermitian = self.hermitian
		unitary = self.unitary

		N = self.N if self.N is not None else None
		D = self.D if self.D is not None else None

		shape = self.shape if self.shape is not None else None
		size = self.size if self.size is not None else None
		ndim = self.ndim if self.ndim is not None else None
		dtype = self.dtype if self.dtype is not None else None

		identity = self.identity
		basis = self.basis
		default = self.default

		parameters = self.parameters() if callable(self.parameters) else self.parameters if self.parameters is not None else None
		system = self.system if self.system is not None else None

		options = Dictionary(
			parameters=parameters,
			D=D,ndim=ndim,
			data=self.data,
			random=self.random,seed=seeder(self.seed),
			index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system
			) if not self.null() else None

		number = Basis.localities(attr=Basis.string,operator=[basis.get(i) for i in (operator if isinstance(operator,iterables) else operator.split(delim))],**options) if operator is not None else None if not self.null() else None

		options = Dictionary(
			parameters=parameters,
			D=D,
			N=((locality if isinstance(locality,integers) else len(where) if isinstance(where,iterables) else 1)//
			   (number if isinstance(number,integers) else 1)),
			ndim=ndim,
			data=self.data,
			random=self.random,seed=seeder(self.seed),
			index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system
			) if not self.null() else None

		# Set tenor, local, locality, where

		tensor = tensor

		local = local

		if locality is not None:
			locality = locality
		elif isinstance(where,iterables):
			locality = len(where)
		elif isinstance(operator,iterables):
			locality = Basis.localities(attr=Basis.string,operator=[basis.get(i) for i in operator],**options)
		elif isinstance(operator,str) and operator.count(delim) > 0:
			locality = Basis.localities(attr=Basis.string,operator=[basis.get(i) for i in operator.split(delim)],**options)
		elif N is not None:
			locality = N
		elif not self.null():
			locality = 1
		else:
			locality = None

		if where is not None:
			where = where
		else:
			where = None

		# Set N
		N = locality if local and N is None else N

		# Set where,locality,operator
		if where is None:
			if operator is None:
				locality = locality
				where = where
				operator = operator
			elif isinstance(operator,str):
				if operator in [default]:
					locality = locality
					where = [i for i in range(locality)]
					operator = operator
				elif operator.count(delim):
					locality = locality
					where = [i for i in range(locality)]
					operator = [i for i in operator.split(delim)]
				else:
					locality = locality
					where = [i for i in range(locality)]
					operator = operator
			elif not isinstance(operator,str) and not isinstance(operator,arrays) and not callable(operator):
				locality = locality
				where = [i for i in range(locality)]
				operator = [i for i in operator]
			else:
				locality = locality
				where = [i for i in range(locality)]
				operator = operator
		else:
			if operator is None:
				locality = locality
				where = [i for i in where] if isinstance(where,iterables) else where
				operator = operator
			elif isinstance(operator,str):
				if operator in [default]:
					locality = locality
					where = [i for i in where] if isinstance(where,iterables) else where
					operator = operator
				elif operator.count(delim):
					locality = locality
					where = [i for i in where] if isinstance(where,iterables) else where
					operator = [i for i in operator.split(delim)]
				else:
					locality = locality
					where = [i for i in where] if isinstance(where,iterables) else where
					operator = operator
			elif not isinstance(operator,str) and not isinstance(operator,arrays) and not callable(operator):
				locality = locality
				where = [i for i in where] if isinstance(where,iterables) else where
				operator = [i for i in operator]
			else:
				locality = locality
				where = [i for i in where] if isinstance(where,iterables) else where
				operator = operator

		N = max((i for i in (N if N is not None else None,locality if locality is not None else None,) if i is not None),default=None) if N is not None or locality is not None else None
		D = D if D is not None else None

		tensor = tensor
		local = local
		locality = min((i for i in (locality if locality is not None else None,sum(i not in [default] for i in where) if isinstance(where,iterables) else None,locality if local else N) if i is not None),default=None) if locality is not None or isinstance(where,iterables) else None
		number = number

		variable = variable
		constant = constant
		symmetry = symmetry
		hermitian = hermitian
		unitary = unitary

		operator = operator[:locality] if operator is not None and not isinstance(operator,str) and not isinstance(operator,arrays) and not callable(operator) else operator
		where = where[:locality] if isinstance(where,iterables) else where
		string = string if string is not None else None

		tensor = tensor if tensor is not None else None
		local = local if local is not None else None
		locality = max(locality,len(where)) if isinstance(where,iterables) else locality
		number = number if number is not None else None

		shape = self.shape if self.shape is not None else None
		size = self.size if self.size is not None else None
		ndim = self.ndim if self.ndim is not None else None
		dtype = self.dtype if self.dtype is not None else None

		identity = identity if self.identity is not None else lambda self=self:None

		system = self.system if self.system is not None else None

		# Check attributes

		options = Dictionary(
			parameters=parameters,
			D=D,N=locality//number,ndim=ndim,
			data=self.data,
			random=self.random,seed=seeder(self.seed),
			index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system
			) if not self.null() else None

		assert ( self.null() or (operator is None and where is None and not locality) or (
				(len(where) == locality) and (
				(isinstance(operator,iterables) and (
					any(i not in basis for i in operator) or
					(Basis.localities(attr=Basis.string,operator=[basis.get(i) for i in operator],**options) == locality))) or
				(isinstance(operator,str) and operator.count(delim) and (
					any(i not in basis for i in operator.split(delim)) or
					(Basis.localities(attr=Basis.string,operator=[basis.get(i) for i in operator.split(delim)],**options) == locality))) or
				(isinstance(operator,str) and not operator.count(delim) and (
					(operator not in basis) or
					((Basis.localities(basis.get(operator),**options)==0) or ((locality % Basis.localities(basis.get(operator),**options)) == 0))))
				))
			),"Inconsistent operator %r, where %r: locality != %d"%(operator,where,locality)


		assert ( self.null() or (operator is None and where is None and not locality) or (
				(isinstance(operator,iterables) and (
					any(i not in basis for i in operator) or
					(len(set((Basis.dimensions(basis.get(i),**options)for i in operator))) == 1))) or
				(isinstance(operator,str) and operator.count(delim) and (
					any(i not in basis for i in operator.split(delim)) or
					(len(set((Basis.dimensions(basis.get(i),**options)for i in operator))) == 1))) or
				(isinstance(operator,str) and not operator.count(delim))
				)
			),"Inconsistent operator %r, dimension %r"%(operator,[Basis.dimensions(basis.get(i),**options) for i in (operator if isinstance(operator,iterables) else [operator])])


		# Set attributes
		self.data = data if data is not None else None
		self.operator = operator if operator is not None else None
		self.where = where if where is not None else None
		self.string = string if string is not None else None

		self.tensor = tensor
		self.local = local
		self.locality = locality
		self.number = number

		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary

		self.N = N
		self.D = D

		self.shape = shape
		self.size = size
		self.ndim = ndim
		self.dtype = dtype

		self.init(data=self.data,state=self.state,parameters=self.parameters,conj=self.conj)

		self.info()

		return

	def init(self,data=None,state=None,parameters=None,conj=False,**kwargs):
		'''
		Initialize operator
		Args:
			data (bool,dict): data of class, or boolean to retain current data attribute or initialize as None
			state (bool,dict,array,Object): state of class
			parameters (array,dict): parameters of class
			conj (bool): conjugate
			kwargs (dict): Additional class keyword arguments
		'''

		if data is None:
			data = self.data
		elif data is True:
			data = self.data
		elif data is False:
			data = None

		if state is None:
			state = self.state
		elif state is True:
			state = self.state
		elif state is False:
			state = None

		if conj is None:
			conj = False

		for kwarg in kwargs:
			if hasattr(self,kwarg):
				setattr(self,kwarg,kwargs[kwarg])

		self.data = data
		self.state = state
		self.conj = conj

		self.spaces()

		if parameters is None:
			parameters = None
		elif parameters is True:
			parameters = None
		elif parameters is False:
			parameters = None

		cls = Parameter
		if not isinstance(self.parameters,cls) and not isinstance(parameters,cls):
			defaults = {}
			parameters = parameters if parameters is not None else self.parameters
			keywords = dict(data=parameters) if not isinstance(parameters,dicts) else parameters
			setter(keywords,{attr: getattr(self,attr) for attr in {**self,**kwargs} if hasattr(self,attr) and not callable(getattr(self,attr)) and attr not in cls.defaults and attr not in dict(data=None,local=None)},delimiter=delim,default=False)
			setter(keywords,dict(string=self.string,variable=self.variable,constant=self.constant,system=self.system),delimiter=delim,default=True)
			setter(keywords,defaults,delimiter=delim,default=False)
			setter(keywords,dict(self.parameters if isinstance(self.parameters,dicts) else {}),delimiter=delim,default=False)
			setter(keywords,{attr: getattr(self,attr) for attr in (self.system if isinstance(self.system,dict) else {}) if (isinstance(self.parameters,dicts) and attr not in self.parameters)},delimiter=delim,default=True)

			self.parameters = cls(**keywords)

		elif isinstance(self.parameters,cls):
			defaults = {}
			parameters = parameters if parameters is not None else parameters
			keywords = parameters if isinstance(parameters,dicts) else dict(data=parameters) if parameters is not None else {}
			setter(keywords,{attr: getattr(self,attr) for attr in {**self,**kwargs} if hasattr(self,attr) and not callable(getattr(self,attr)) and attr not in cls.defaults and attr not in dict(data=None,local=None)},delimiter=delim,default=False)
			setter(keywords,dict(string=self.string,variable=self.variable,constant=self.constant,system=self.system),delimiter=delim,default=False)
			setter(keywords,defaults,delimiter=delim,default=False)
			setter(keywords,dict(self.parameters if isinstance(self.parameters,dicts) else {}),delimiter=delim,default=False)

			self.parameters.init(**keywords)

		else:
			self.parameters = parameters


		if state is None or not callable(state):
			def state(parameters=None,state=state):
				return state

		self.state = state

		N = self.N
		D = self.D

		operator = self.operator
		where = self.where
		string = self.string

		tensor = self.tensor
		local = self.local
		locality = self.locality
		number = self.number

		variable =  self.variable
		constant =  self.constant
		symmetry =  self.symmetry
		hermitian =  self.hermitian
		unitary =  self.unitary

		identity = self.identity

		N = max((i for i in (max((i for i in (locality,len(where) if where is not None else None,N) if i is not None),default=None),) if i is not None),default=None)
		D = D if D is not None else None

		operator = operator[:locality] if operator is not None and not isinstance(operator,str) and not isinstance(operator,arrays) and not callable(operator) else operator
		where = where[:locality] if isinstance(where,iterables) else where
		string = string if string is not None else None

		tensor = tensor if tensor is not None else None
		local = local if local is not None else None
		locality = max(locality,len(where)) if isinstance(where,iterables) else locality
		number = number if number is not None else None

		assert self.null() or (N is None and locality is None and where is None) or ((locality <= N) and (locality == len(where)) and all(i in range(N) for i in where)), "Inconsistent N %d, locality %d, where %r"%(N,locality,where)

		self.N = N
		self.D = D

		self.operator = operator
		self.where = where
		self.string = string

		self.tensor = tensor
		self.local = local
		self.locality = locality
		self.number = number

		self.variable =  variable
		self.constant =  constant
		self.symmetry =  symmetry
		self.hermitian =  hermitian
		self.unitary =  unitary

		options = Dictionary(
			parameters=self.parameters(),
			D=self.D,N=self.locality//self.number,ndim=self.ndim,
			data=self.data,
			random=self.random,seed=seeder(self.seed),
			index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system
			) if not self.null() else None

		if tensor not in [None,False]:
			if not callable(tensor):
				def tensor(data,N,D,d):
					return reshape(data,[*data.shape[:max(0,d-2)],*[D]*(N*min(d,2))])
		else:
			tensor = None
		self.tensor = tensor

		def identity(N=None,D=None,self=self):

			data = None

			if self.null():
				return data

			cls = Basis.identity
			N = self.N if N is None else N
			D = self.D if D is None else D
			d = Basis.dimensions(cls)
			options = dict(index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system)

			if not memory(D**N):
				return data

			data = tensorprod([cls(D=D,**options)]*N)
			data = self.tensor(data,N=N,D=D,d=d) if self.tensor is not None else data

			return data

		self.identity = identity

		if ( (not self.null()) and ((not isinstance(self.data,arrays)) and not callable(self.data)) and (
			((isinstance(self.operator,str) and self.operator in self.basis) or
			(isinstance(self.operator,iterables) and all(i in self.basis for i in self.operator)))
			)):

			assert (
					(isinstance(self.operator,iterables) and (
						any(i not in self.basis for i in self.operator) or
						(Basis.localities(attr=Basis.string,operator=[self.basis.get(i) for i in self.operator],**options) == self.locality))) or
					(isinstance(self.operator,str) and self.operator.count(delim) and (
						any(i not in self.basis for i in self.operator.split(delim)) or
						(Basis.localities(attr=Basis.string,operator=[self.basis.get(i) for i in self.operator.split(delim)],**options) == self.locality))) or
					(isinstance(self.operator,str) and not self.operator.count(delim) and (
						(self.operator not in self.basis) or
						((Basis.localities(self.basis.get(self.operator),**options)==0) or ((self.locality % Basis.localities(self.basis.get(self.operator),**options)) == 0))))
				),"Inconsistent operator %r, where %r: locality != %d"%(self.operator,self.where,self.locality)


			assert (
					(isinstance(self.operator,iterables) and (
						any(i not in self.basis for i in self.operator) or
						(len(set((Basis.dimensions(self.basis.get(i),**options)for i in self.operator))) == 1))) or
					(isinstance(self.operator,str) and self.operator.count(delim) and (
						any(i not in self.basis for i in self.operator.split(delim)) or
						(len(set((Basis.dimensions(self.basis.get(i),**options)for i in self.operator))) == 1))) or
					(isinstance(self.operator,str) and not self.operator.count(delim))
				),"Inconsistent operator %r, dimension %r"%(self.operator,[Basis.dimensions(self.basis.get(i),**options) for i in (self.operator if isinstance(self.operator,iterables) else [self.operator])])

			data = [i for i in self.operator] if isinstance(self.operator,iterables) else [self.operator]*(self.locality//Basis.localities(self.basis.get(self.operator),**options)) if isinstance(self.operator,str) else None
			_data = [] if self.local else [self.default]*(self.N-self.locality) if data is not None else None

			shape = Basis.shapes(attr=Basis.string,operator=[self.basis.get(i) for i in [*data,*_data]],**options) if data is not None else None
			axes = [*self.where,*(() if self.local else set(range(self.N))-set(self.where))] if data is not None else None
			ndim = Basis.dimensions(attr=Basis.string,operator=[self.basis.get(i) for i in [*data,*_data]],**options) if data is not None else None
			dtype = self.dtype

			shape = {axis: [shape[axis][axes.index(i)] for i in range(max(axes)+1) if i in axes] for axis in shape} if data is not None else None
			axes = [[i] for i in axes] if data is not None else None
			ndim = ndim if data is not None else None
			dtype = dtype

			local = self.local
			tensor = partial(self.tensor,N=self.locality if self.local else self.N,D=self.D,d=ndim) if self.tensor is not None else None

			data = [self.basis.get(i)(**Basis.opts(self.basis.get(i),options)) for i in data] if data is not None else None
			_data = [self.basis.get(i)(**Basis.opts(self.basis.get(i),options)) for i in _data] if data is not None else None

			data = [*data,*_data] if not self.local else data

			if self.architecture is None or self.architecture in ['array']:
				if self.local:
					data = tensorprod(data) if data is not None else None
				else:
					data = swap(tensorprod(data),axes=axes,shape=shape) if data is not None else None
				data = array(data,dtype=dtype) if data is not None else None
			elif self.architecture in ['tensor']:
				data = tensorprod(data) if data is not None else None
			elif self.architecture in ['tensor_quimb']:
				data = tensorprod(data) if data is not None else None

		else:

			data = self.data


		if (((self.data is not None) or (self.operator is not None))):

			self.setup(data=data,operator=self.operator,where=self.where,string=self.string)

		if (self.parameters() is None) and (not isinstance(self.data,arrays)) and (not callable(self.data)):

			data = None

		elif isinstance(self.data,arrays) or callable(self.data):

			data = self.data

		elif isinstance(self.operator,arrays) or callable(self.operator):

			data,self.operator = self.operator,self.string

		elif self.operator is None:

			data = None

		else:

			data = self.data

		self.N = max((i for i in (self.N if self.N is not None else None,self.locality if self.locality is not None else None,) if i is not None),default=None) if self.local and (self.N is not None or self.locality is not None) else self.N if self.N is not None else None
		self.D = self.D if self.D is not None else None

		self.shape = [prod(i) for i in Basis.shapes(
				attr=Basis.string,
				operator=[self.basis.get(i)
					for i in [
						*(self.operator if isinstance(self.operator,iterables) else [self.operator]*(self.locality//Basis.localities(self.basis.get(self.operator),**options))),
						*([] if self.local else [self.default]*(self.N-self.locality))]
						],
				**options).values()] if all(obj is not None for obj in [self.operator,self.N,self.D,self.locality]) else ()
		self.size = prod(self.shape) if self.shape is not None else None
		self.ndim = len(self.shape) if self.size is not None else None
		self.dtype = self.dtype if self.dtype is not None else None

		if self.architecture is None or self.architecture in ['array']:
			data = self.tensor(data,N=self.locality if self.local else self.N,D=self.D,d=self.ndim) if self.tensor is not None else data
		elif self.architecture in ['tensor']:
			data = data
		elif self.architecture in ['tensor_quimb']:
			data = data

		self.data = data

		if self.func is None:
			def func(parameters=None,state=None,**kwargs):
				return self.data
		else:
			func = self.func

		if self.gradient is None:
			def gradient(parameters=None,state=None,**kwargs):
				return 0*self.data
		else:
			gradient = self.gradient

		data = self.data
		state = self.state() if callable(self.state) else self.state

		# TODO: Add self.where,self.state.where interdependency
		# for subspace evolution within where of state
		if self.null():
			kwargs = dict(
				where = None,
				samples = None,
				attributes = None,
				local = None,
				tensor = None,
				)
		else:
			if self.state is not None and self.state() is not None:
				if all(i in self.state.where for i in self.where) or (self.state.locality >= self.locality):
					kwargs = dict(
						where = self.where,
						attributes = Dictionary(N=self.state.N,D=self.state.D,d=self.ndim,s=self.state.ndim,samples=self.state.samples if self.state.samples is not None else self.samples,),
						local = self.local is True,
						tensor = self.tensor is not None,
						)
				else:
					raise NotImplementedError("Incorrect state %r for locality %d, where %r"%(self.state,self.locality,self.where))
			else:
				kwargs = dict(
					where = self.where,
					attributes = Dictionary(N=self.N,D=self.D,d=self.ndim,s=Basis.dimension,samples=self.samples),
					local = self.local is True,
					tensor = self.tensor is not None,
					)

		kwargs = dict(**{**kwargs,**{attr: self.options[attr] for attr in self.options if attr not in kwargs}}) if self.options is not None else kwargs

		try:
			contract = contraction(data,state,**kwargs) if self.contract is None else self.contract
		except NotImplementedError as exception:
			def contract(data,state,**kwargs):
				return state
			raise exception

		try:
			grad_contract = gradient_contraction(data,state,**kwargs) if self.gradient_contract is None else self.gradient_contract
		except NotImplementedError as exception:
			def grad_contract(grad,data,state,**kwargs):
				return 0

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = grad_contract

		parameters = self.parameters()
		state = self.state() if self.state is not None and self.state() is not None else self.identity()
		where = self.where
		wrapper = jit
		kwargs = {}

		self.func = wrapper(self.func,parameters=parameters,state=state,**kwargs)
		self.gradient = wrapper(self.gradient,parameters=parameters,state=state,**kwargs)
		self.contract = wrapper(self.contract,state=state,where=where,**kwargs)
		self.gradient_contract = wrapper(self.gradient_contract,state=state,where=where,**kwargs)

		return

	def setup(self,data=None,operator=None,where=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			where (iterable[int]): location of local operators, i.e) nearest neighbour
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''

		return

	def __call__(self,parameters=None,state=None,**kwargs):
		'''
		Call class
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional operator keyword arguments
		Returns:
			data (array): data
		'''
		return self.contract(self.func(parameters=parameters,state=state,**kwargs),state=state)

	def grad(self,parameters=None,state=None,**kwargs):
		'''
		Call class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional operator keyword arguments
		Returns:
			data (array): data
		'''
		return self.gradient_contract(self.gradient(parameters=parameters,state=state,**kwargs),self.func(parameters=parameters,state=state,**kwargs),state=state)


	def spaces(self,N=None,D=None,space=None,system=None):
		'''
		Set space attributes
		Args:
			N (int): Size of system
			D (int): Local dimension of system
			space (str,dict,Space): Type of local space
			system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		'''

		space = self.space if space is None else space

		defaults = dict(
			N = self.N if N is None else N,
			D = self.D if D is None else D,
			space = self.space if space is None else space,
			system = self.system if system is None else system,
		)

		if space is None:
			space = dict(space=space)
		elif not isinstance(space,dicts):
			space = dict(space=space)
		else:
			space = dict(**space)

		setter(space,defaults,delimiter=delim,default=True)

		self.space = Space(**space)

		self.N = self.space.N
		self.D = self.space.D

		return


	def times(self,M=None,T=None,tau=None,P=None,time=None,system=None):
		'''
		Set time attributes
		Args:
			M (int): Duration of system
			T (int): Simulation time
			tau (float): Simulation time scale
			P (int): Trotter order
			time (str,dict,Time): Type of time evolution
			system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		'''

		time = self.time if time is None else time

		defaults = dict(
			M = self.M if M is None else M,
			T = self.T if T is None else T,
			tau = self.tau if tau is None else tau,
			P = self.P if P is None else P,
			time = self.time if time is None else time,
			system = self.system if system is None else system,
		)

		if time is None:
			time = dict(time=time)
		elif not isinstance(time,dicts):
			time = dict(time=time)
		else:
			time = dict(**time)

		setter(time,defaults,delimiter=delim,default=True)

		self.time = Time(**time)

		self.M = self.time.M
		self.T = self.time.T
		self.P = self.time.P
		self.tau = self.time.tau

		return


	def lattices(self,N=None,d=None,lattice=None,system=None):
		'''
		Set lattice attributes
		Args:
			N (int): Size of system
			d (int): Spatial dimension of system
			lattice (str,dict,Lattice): Type of lattice
			system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		'''

		lattice = self.lattice if lattice is None else lattice

		defaults = dict(
			N = self.N if N is None else N,
			d = self.d if d is None else d,
			lattice = self.lattice if lattice is None else lattice,
			system = self.system if system is None else system,
		)

		if lattice is None:
			lattice = dict(lattice=lattice)
		elif not isinstance(lattice,dicts):
			lattice = dict(lattice=lattice)
		else:
			lattice = dict(**lattice)

		setter(lattice,defaults,delimiter=delim,default=True)

		self.lattice = Lattice(**lattice)

		self.N = self.lattice.N
		self.d = self.lattice.d

		return

	def sample(self,parameters=None,state=None,**kwargs):
		'''
		Class sample
		Args:
			parameters (array): parameters of class
			state (array,tensor,network): state of class
			kwargs (dict): Additional class keyword arguments
		Returns:
			data (object): data
		'''

		data = self.array(parameters=parameters,state=state,**kwargs)

		return data

	def array(self,parameters=None,state=None,**kwargs):
		'''
		Class array
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional operator keyword arguments
		Returns:
			data (array): data
		'''

		data = self(parameters=parameters,state=state,**kwargs)

		return data

	def __str__(self):
		if isinstance(self.string,str):
			string = self.string
		elif isinstance(self.operator,str):
			string = self.operator
		elif self.operator is not None and not isinstance(self.operator,arrays) and not callable(self.operator):
			string = '%s'%(delim.join(self.operator))
		elif self.string:
			string = self.string
		else:
			string = self.__class__.__name__
		return string

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return (
			hash(self.string) ^
			hash(tuple(self.operator) if not isinstance(self.operator,str) else self.operator) ^
			hash(tuple(self.where)) ^
			hash(id(self))
			)

	def __key__(self):
		attrs = [self.string,self.operator,self.where,id(self)]
		key = []
		for attr in attrs:
			if attr is None:
				continue
			if isinstance(attr,scalars):
				key.append(attr)
			else:
				key.extend(attr)
		key = tuple(key)
		return key

	def __len__(self):
		return len(self.data)

	def __matmul__(self,other):
		'''
		Tensor product operation on class

		Objects with disjoint locality are tensored within the larger of the self,other locality
		i.e) self.where = [0,1], other.where = [2,3], self.N = 3, other.N = 4
		-> instance.where = [*self.where,*other.where] = [0,1,2,3] ,  instance.N = max(self.N,other.N) = 4

		Objects with non-disjoint locality are tensored within the product of the self,other locality
		i.e) self.where = [0,1], other.where = [1,3], self.N = 3, other.N = 4
		-> instance.where = [*self.where,self.N+*other.where] = [0,1,4,6] ,  instance.N = max(self.N,other.N) = 7

		Args:
			other (class,int): class instance or integer for tensor product
		Returns:
			instance (class): new class instance with tensor product of instance and other
		'''

		support = self.where if isinstance(self.where,iterables) else None
		attributes = ['D','ndim']

		if other is self or isinstance(other,integers):

			support = self.where if support is None else support
			attributes = [attr for attr in attributes if hasattr(self,attr)]

			if other is self:
				other = 2

			if not self.constant:
				raise NotImplementedError("<%r> @ <%r> Not Implemented - Constant classes required"%(self,other))

			if (self.parameters is not None and self.parameters() is not None):
				raise NotImplementedError("<%r> @ <%r> Not Implemented - Non-parameterized distinct classes required"%(self,other))

			data = None
			operator = ([*self.operator] if isinstance(self.operator,iterables) else [self.operator])*other
			where = [i+self.N*j for j in range(other) for i in self.where]
			string = delim.join((self.string,)*other) if self.string is not None else None

			tensor = self.tensor
			local = self.local
			locality = self.locality*other
			number = self.number
			variable = self.variable
			constant = self.constant
			symmetry = self.symmetry

			N = self.N*other
			D = self.D

			shape = None
			size = None
			ndim = self.ndim

			if self.state is not None and self.state() is not None:
				try:
					state = self.state @ other
				except:
					state = None
			else:
				state = None

			parameters = self.parameters() if self.parameters is not None and self.parameters() is not None else self.parameters() if self.parameters is not None else None

			conj = self.conj

		elif isinstance(other,type(self)):

			support = intersection(*(obj.where for obj in (self,other) if obj is not None and obj.where is not None))
			attributes = [attr for attr in attributes if all(hasattr(obj,attr) for obj in (self,other))]

			if (not self.constant) or (not other.constant):
				raise NotImplementedError("<%r> @ <%r> Not Implemented - Constant classes required"%(self,other))

			if (self.parameters is not None and self.parameters() is not None) and (other.parameters is not None and other.parameters() is not None) and (self.operator != other.operator):
				raise NotImplementedError("<%r> @ <%r> Not Implemented - Non-parameterized distinct classes required"%(self,other))

			if any(getattr(self,attr) != getattr(other,attr) for attr in attributes):
				raise NotImplementedError("<%r> @ <%r> Not Implemented - Identical attributes required"%(self,other))

			data = None
			operator = [*(self.operator if isinstance(self.operator,iterables) else [self.operator]),
						*(other.operator if isinstance(other.operator,iterables) else [other.operator])]
			where = [*([i for i in self.where] if isinstance(self.where,iterables) else [self.where]),
					*([i + self.N*(len(support) > 0) for i in other.where] if isinstance(other.where,iterables) else [other.where])]
			string = delim.join((self.string,other.string)) if self.string is not None and other.string is not None else self.string if self.string is not None else other.string if other.string is not None else None

			tensor = self.tensor if self.tensor is not None else other.tensor if other.tensor is not None else None
			local = all((self.local,other.local))
			locality = sum((self.locality,other.locality))
			number = max((self.number,other.number))
			variable = all((self.variable,other.variable))
			constant = all((self.constant,other.constant))
			symmetry = self.symmetry

			N = max(self.N,other.N) if (len(support) == 0) else (self.N + other.N)
			D = max(self.D,other.D)

			shape = None
			size = None
			ndim = self.ndim

			if self.state is not None and other.state is not None and self.state() is not None and other.state() is not None:
				try:
					state = self.state @ other.state
				except:
					state = None
			elif self.state is not None:
				state = self.state
			elif other.state is not None:
				state = other.state
			else:
				state = None

			parameters = self.parameters()*other.parameters() if self.parameters is not None and self.parameters() is not None and other.parameters is not None and other.parameters() is not None else self.parameters() if self.parameters is not None else other.parameters() if other.parameters is not None else None

			conj = all((self.conj,other.conj))

		else:

			raise NotImplementedError("<%r> @ <%r> Not Implemented - Identical classes required"%(self,other))


		kwargs = {
			**{attr: getattr(self,attr) for attr in self if not callable(getattr(self,attr))},
			**dict(
				data=data,operator=operator,where=where,string=string,
				tensor=tensor,local=local,locality=locality,number=number,variable=variable,constant=constant,symmetry=symmetry,
				N=N,D=D,shape=shape,size=size,ndim=ndim,
				state=state,parameters=parameters,conj=conj
				)
			}


		instance = self.__class__(**kwargs)


		return instance

	def null(self):
		'''
		Null status of class
		Returns:
			null (bool): Null status of class, depending on class attributes
		'''

		null = all(getattr(self,attr,default) is None
			for attr,default in dict(operator=None,where=None,string=None).items())

		return null


	def component(self,parameters=None,state=None,index=None,basis=None,**kwargs):
		'''
		Get components of objects with respect to basis
		Args:
			parameters (array): parameters
			state (obj): state
			index (int,str,iterable[str]): Index of basis operator for component
			basis (str): basis for operators, allowed strings in ['pauli']
			kwargs (dict): Additional operator keyword arguments
		Returns:
			data (array): Component of basis of class with respect to string
		'''

		index = tuple(index.split(delim) if isinstance(index,str) else index) if not isinstance(index,integers) else index

		basis = 'pauli' if basis is None else basis

		options = Dictionary(D=self.D,N=self.locality//self.number,ndim=self.ndim,index=self.index,architecture=self.architecture,dtype=self.dtype,seed=seeder(self.seed),system=self.system)

		basis = Basis.basis(basis,**options) if basis is not None else None

		index = (list(basis)[index] if isinstance(index,integers) else index) if basis is not None else None

		data = basis[index] if basis is not None else None

		data = self.dot(parameters=parameters,state=state,data=data) if basis is not None else None

		data = real(data*sqrt(self.D**self.locality))

		return data

	def dot(self,parameters=None,state=None,data=None,**kwargs):
		'''
		Get inner product of class with data
		Args:
			parameters (array): parameters
			state (obj): state
			data (array): Data for inner product
			kwargs (dict): Additional operator keyword arguments
		Returns:
			data (array): Inner product of class with data
		'''

		cls = self(parameters=parameters,state=state,**kwargs)

		data = self.inner(cls,data)

		return data

	def inner(self,data=None,other=None,**kwargs):
		'''
		Get inner product of data and other
		Args:
			data (array): Data for inner product
			other (array): Data for inner product
			kwargs (dict): Additional operator keyword arguments
		Returns:
			data (array): Inner product of data and other
		'''

		if data is None or other is None:
			data = data
		else:
			data = inner(ravel(data),ravel(other))

		return data


	def info(self,display=None,ignore=None,verbose=None,**kwargs):
		'''
		Log class information
		Args:
			display (str,iterable[str]): Show attributes
			ignore (str,iterable[str]): Do not show attributes
			verbose (bool,int,str): Verbosity of message
			kwargs (dict): Additional logging keyword arguments
		'''

		if not verbose and not self.verbose:
			return

		msg = []

		options = dict(
			align=kwargs.get('align','<'),
			space=kwargs.get('space',1),
			width=kwargs.get('width',2)
			)

		precision = kwargs.get('precision',8)

		parse = lambda obj: str(obj.round(precision)) if isinstance(obj,arrays) else str(obj)

		display = None if display is None else [display] if isinstance(display,str) else display
		ignore = None if ignore is None else [ignore] if isinstance(ignore,str) else ignore

		for attr in [None,'string','key','seed','instance','instances','backend','architecture','timestamp','N','D','d','data','shape','size','ndim','dtype','cwd','path','conf','logger','cleanup']:

			obj = attr
			if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
				continue

			if attr is None:
				attr = 'cls'
				substring = str(self)
			else:
				substring = getattr(self,attr,None)

			if getattr(self,attr,None) is not None:
				if attr in ['data']:
					substring = [substring[i] for i in substring if (substring[i] is not None) and (substring[i].data is not None)] if isinstance(substring,dict) else substring if not isinstance(substring,arrays) else '\n%s'%(parse(substring))
				else:
					substring = [substring[i] for i in substring] if isinstance(substring,dict) else substring
				string = '%s: %s'%(attr,parse(substring))
			else:
				string = '%s: %s'%(attr,parse(substring))

			msg.append(string)

		for attr in ['operator','where','locality','tensor','local','variable','constant','symmetry']:

			obj = attr
			if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
				continue

			if getattr(self,attr,None) is not None:
				try:
					string = '%s: %s'%(attr,getattr(self,attr)() if callable(getattr(self,attr)) else getattr(self,attr))
				except:
					string = '%s: %s'%(attr,getattr(self,attr) is not None if callable(getattr(self,attr)) else getattr(self,attr))
			else:
				string = '%s: %s'%(attr,parse(substring))

			msg.append(string)

		if isinstance(self,Objects):

			for attr in (self.data if self.data is not None else []):

				obj = 'parameters'
				if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
					continue

				if self.data.get(attr) is None:
					continue

				string = []
				for subattr in [None,'operator','variable','method','indices','local','tensor','where','shape','parameters']:

					obj = subattr
					if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
						continue

					if subattr is None:
						subattr = 'data.mean'
						if self.parameters is None or self.parameters() is None or not self.data[attr].parameters.variable:
							try:
								substring = self.data[attr].parameters(self.data[attr].parameters())
							except:
								if self.parameters.indices is not None:
									substring = (self.data[attr].parameters({self.data[attr].parameters.indices:self.data[attr].parameters()}),)
								else:
									substring = self.data[attr].parameters(self.data[attr].parameters())
						else:
							substring = (self.data[attr].parameters(parameters) for parameters in (
								self.parameters(self.parameters()) if self.parameters.size>1
								else (self.parameters(self.parameters()),)))
							substring = array([i for i in substring if i is not None])
							if substring.size:
								substring = norm(substring)/sqrt(substring.size)
							else:
								substring = None
						if substring is not None and substring.size:
							substring = '%0.4e'%(substring)
						else:
							substring = parse(substring)
					elif subattr in ['tensor']:
						substring = getattrs(self.data[attr],subattr,default=None,delimiter=delim)
						if callable(substring):
							try:
								substring = parse(substring())
							except:
								substring = parse(substring is not None)
						else:
							substring = parse(substring is not None)
					else:
						substring = getattrs(self.data[attr].parameters,subattr,default=None,delimiter=delim)
						if callable(substring):
							try:
								substring = parse(substring())
							except:
								substring = parse(substring is not None)
						elif isinstance(substring,(str,int,list,tuple,bool,*arrays)):
							substring = parse(substring)
						elif isinstance(substring,dict):
							substring = ', '.join(['%s: %s'%(prop,substring[prop]) for prop in substring])
						elif substring is not None:
							substring = '%0.4e'%(substring)
						else:
							substring = parse(substring)

					substring = '%s : %s'%(subattr,'{:{align}{space}{width}}'.format(substring,**options))

					string.append(substring)

				string = 'parameters.%s\n\t%s'%(self.data[attr],'\n\t'.join([i for i in string if i is not None]))

				msg.append(string)

			for attr in ['parameters']:

				obj = attr
				if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
					continue
				string = []
				for subattr in [None,'shape']:
					if subattr is None:
						subattr = attr
						substring = ' '.join([str(i) for i in getattr(self,attr,[]) if getattr(self,attr)[i].variable])
						substring = None if not len(substring) else substring
					else:
						substring = getattr(self.parameters,subattr)
						substring = None if substring is None or (0 in substring) else substring

					if isinstance(substring,(str,int,list,tuple,*arrays)):
						substring = '%s'%(substring,)
					elif substring is not None:
						substring = '%0.4e'%(substring)
					else:
						substring = parse(substring)

					if len(substring):
						substring = '%s : %s'%(subattr,substring)
					else:
						substring = None

					string.append(substring)

				string = ', '.join([i for i in string if i is not None])

				msg.append(string)


		elif isinstance(self,Object):

			string = []
			for attr in [None,'variable','method','indices','local','where','shape','parameters']:

				obj = 'parameters'
				if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
					continue

				if attr is None:
					attr = 'data.mean'
					if self.parameters is None or self.parameters() is None or not self.parameters.variable:
						substring = self.parameters()
					else:
						try:
							substring = (self.parameters(parameters) for parameters in (
									self.parameters(self.parameters()) if self.parameters().size>1
									else (self.parameters(self.parameters()),)))
						except:
							if self.parameters.indices is not None:
								substring = (self.parameters({self.parameters.indices:self.parameters()}),)
							else:
								substring = self.parameters(self.parameters())
						substring = array([i for i in substring if i is not None])
						if substring.size:
							substring = norm(substring)/sqrt(substring.size)
						else:
							substring = None
					if substring is not None:
						substring = '%0.4e'%(substring)
					else:
						substring = parse(substring)
				else:
					substring = getattrs(self.parameters,attr,default=None,delimiter=delim)
					if isinstance(substring,(str,int,list,tuple,bool,*arrays)):
						substring = parse(substring)
					elif isinstance(substring,dict):
						substring = ', '.join(['%s: %s'%(prop,substring[prop]) for prop in substring])
					elif substring is not None:
						substring = '%0.4e'%(substring)
					else:
						substring = parse(substring)

				substring = '%s : %s'%(attr,'{:{align}{space}{width}}'.format(substring,**options))

				string.append(substring)

			string = 'parameters.%s\n\t%s'%(self,'\n\t'.join(string))

			msg.append(string)


		msg = [i if isinstance(i,str) else str(i) for i in msg]

		msg = '\n'.join(msg)

		self.log(msg,verbose=verbose)

		return

class Data(Object):
	'''
	Data class for Quantum Objects
	Args:
		data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
		where (iterable[int]): location of local operators, i.e) nearest neighbour
		string (str): string label of operator
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional system keyword arguments
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.data for attr in ['data']},
		}

	def setup(self,data=None,operator=None,where=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			where (iterable[int]): location of local operators, i.e) nearest neighbour
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		where = self.where if where is None else where
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		functions = []

		do = not self.null()

		if do:

			if ((isinstance(self.operator,str) and (self.operator in functions)) or
				(isinstance(self.operator,iterables) and any(i in self.operator for i in functions))):

				options = Dictionary(D=self.D,N=self.locality//self.number,ndim=self.ndim,index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system)

				seed = seeder(self.seed)

				data = [i for i in self.operator] if isinstance(self.operator,iterables) else [self.operator]*(self.locality//Basis.localities(self.basis.get(self.operator),**options)) if isinstance(self.operator,str) else None

				def function(parameters,state,options=options,**kwargs):
					return None

				def func(parameters=None,state=None,**kwargs):
					return function(parameters=parameters,state=state,**kwargs)

				def gradient(parameters=None,state=None,**kwargs):
					return None

				data = func(parameters=self.parameters(),state=self.state())

			else:

				data = self.data if data is None else data

		else:

			data = None

		variable = self.variable if self.variable is not None else None
		constant = True
		symmetry = None

		hermitian = False
		unitary = True

		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.where = where if where is not None else self.where
		self.string = string if string is not None else self.string

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = self.gradient_contract

		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary

		return

class Gate(Object):
	'''
	Gate class for Quantum Objects
	Args:
		data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
		where (iterable[int]): location of local operators, i.e) nearest neighbour
		string (str): string label of operator
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional system keyword arguments
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.I for attr in ['i']},
		**{attr: Basis.X for attr in ['x']},
		**{attr: Basis.Y for attr in ['y']},
		**{attr: Basis.Z for attr in ['z']},
		**{attr: Basis.H for attr in ['HADAMARD','H']},
		**{attr: Basis.S for attr in ['PHASE','S']},
		**{attr: Basis.T for attr in ['TEE','T']},
		**{attr: Basis.CNOT for attr in ['CNOT','C','cnot']},
		**{attr: Basis.gate for attr in ['gate']},
		**{attr: Basis.clifford for attr in ['clifford']},
		**{attr: Basis.identity for attr in ['IDENTITY','identity']},
		}

	def setup(self,data=None,operator=None,where=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			where (iterable[int]): location of local operators, i.e) nearest neighbour
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		where = self.where if where is None else where
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		functions = ['gate','clifford']

		do = not self.null()

		if do:

			if ((isinstance(self.operator,str) and (self.operator in functions)) or
				(isinstance(self.operator,iterables) and any(i in self.operator for i in functions))):

				options = dict(D=self.D,N=self.locality//self.number,ndim=self.ndim,index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system)

				data = [i for i in self.operator] if isinstance(self.operator,iterables) else [self.operator]*(self.locality//Basis.localities(self.basis.get(self.operator),**options)) if isinstance(self.operator,str) else None
				_data = [] if self.local else [self.default]*(self.N-self.locality) if data is not None else None

				shape = Basis.shapes(attr=Basis.string,operator=[self.basis.get(i) for i in [*data,*_data]],**options) if data is not None else None
				axes = [*self.where,*(() if self.local else set(range(self.N))-set(self.where))] if data is not None else None
				ndim = Basis.dimensions(attr=Basis.string,operator=[self.basis.get(i) for i in [*data,*_data]],**options) if data is not None else None
				dtype = self.dtype

				shape = {axis: [shape[axis][axes.index(i)] for i in range(max(axes)+1) if i in axes] for axis in shape} if data is not None else None
				axes = [[i] for i in axes] if data is not None else None
				ndim = ndim if data is not None else None
				dtype = dtype

				local = self.local
				tensor = partial(self.tensor,N=self.locality if self.local else self.N,D=self.D,d=ndim) if self.tensor is not None else None

				data = [*data,*_data] if not self.local else data

				seed = seeder(self.seed)

				if self.architecture is None or self.architecture in ['array']:
					if self.local:
						options = Dictionary(
							D=self.D,N=self.locality//self.number,ndim=ndim,
							local=local,tensor=tensor,
							random=self.random,seed=seed,
							index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system,
							data=self.data,operator=data,
							basis=self.basis,axes=axes,shapes=shape,
							)
						data = options.operator
						if len(data)>1:
							options = {index: Dictionary(Basis.opts(options.basis.get(i),options)) for index,i in enumerate(data)}
							for index,i in zip(options,data):
								options[index].basis = options[index].basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options[list(options)[0]].tensor(tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options]))
							else:
								def function(parameters,state,options=options,**kwargs):
									return tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options])
						else:
							for i in data:
								options = Dictionary(Basis.opts(options.basis.get(i),options))
								options.basis = options.basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options.tensor(options.basis(**{**options,**kwargs}))
							else:
								def function(parameters,state,options=options,**kwargs):
									return options.basis(**{**options,**kwargs})
					else:
						options = Dictionary(
							D=self.D,N=self.locality//self.number,ndim=ndim,
							local=local,tensor=tensor,
							random=self.random,seed=seed,
							index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system,
							data=self.data,operator=data,
							basis=self.basis,axes=axes,shapes=shape,
							)
						data = options.operator
						if len(data)>1:
							options = {index: Dictionary(Basis.opts(options.basis.get(i),options)) for index,i in enumerate(data)}
							for index,i in zip(options,data):
								options[index].basis = options[index].basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options[list(options)[0]].tensor(swap(tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options]),axes=options[list(options)[0]].axes,shape=options[list(options)[0]].shapes))
							else:
								def function(parameters,state,options=options,**kwargs):
									return swap(tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options]),axes=options[list(options)[0]].axes,shape=options[list(options)[0]].shapes)
						else:
							for i in data:
								options = Dictionary(Basis.opts(options.basis.get(i),options))
								options.basis = options.basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options.tensor(options.basis(**{**options,**kwargs}))
							else:
								def function(parameters,state,options=options,**kwargs):
									return options.basis(**{**options,**kwargs})
				else:
					raise NotImplementedError

				def func(parameters=None,state=None,**kwargs):
					return function(parameters=parameters,state=state,**kwargs)

				def gradient(parameters=None,state=None,**kwargs):
					return None

				data = func(parameters=self.parameters(),state=self.state())

			else:

				data = self.data if data is None else data

		else:

			data = None

		variable = self.variable if self.variable is not None else None
		constant = True
		symmetry = None

		hermitian = False
		unitary = True

		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.where = where if where is not None else self.where
		self.string = string if string is not None else self.string

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = self.gradient_contract

		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary

		return

class Pauli(Object):
	'''
	Pauli class for Quantum Objects
	Args:
		data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
		where (iterable[int]): location of local operators, i.e) nearest neighbour
		string (str): string label of operator
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional system keyword arguments
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.I for attr in ['I']},
		**{attr: Basis.X for attr in ['X']},
		**{attr: Basis.Y for attr in ['Y']},
		**{attr: Basis.Z for attr in ['Z']},
			}

	def setup(self,data=None,operator=None,where=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			where (iterable[int]): location of local operators, i.e) nearest neighbour
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		where = self.where if where is None else where
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		functions = []

		do = (not self.null()) and (self.parameters is not None) and (self.parameters() is not None)

		if do:

			if ((isinstance(self.operator,str) and (self.operator in functions)) or
				(isinstance(self.operator,iterables) and any(i in self.operator for i in functions))):

				options = Dictionary(D=self.D,N=self.locality//self.number,ndim=self.ndim,index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system)

				seed = seeder(self.seed)

				data = [i for i in self.operator] if isinstance(self.operator,iterables) else [self.operator]*(self.locality//Basis.localities(self.basis.get(self.operator),**options)) if isinstance(self.operator,str) else None

				def function(parameters,state,options=options,**kwargs):
					return None

				def func(parameters=None,state=None,**kwargs):
					return function(parameters=parameters,state=state,**kwargs)

				def gradient(parameters=None,state=None,**kwargs):
					return None

				data = func(parameters=self.parameters(),state=self.state())

			else:

				data = self.data if data is None else data

		else:

			data = None

		variable = self.variable if self.variable is not None else None
		constant = self.constant if self.constant is not None else None
		symmetry = None

		hermitian = False
		unitary = True

		self.parameters.init(parameters=dict(scale=pi/2))

		identity = self.identity(N=self.locality if self.local else self.N,D=self.D)

		if self.parameters() is not None:

			def func(parameters=None,state=None,**kwargs):
				parameters = self.parameters(parameters,**kwargs) if parameters is not None else self.parameters(self.parameters(),**kwargs)
				return cos(parameters)*identity + -1j*sin(parameters)*self.data

			def gradient(parameters=None,state=None,**kwargs):
				grad = self.parameters.grad(parameters,**kwargs)
				parameters = self.parameters(parameters,**kwargs) if parameters is not None else self.parameters(self.parameters(),**kwargs)
				return grad*(-sin(parameters)*identity + -1j*cos(parameters)*self.data)

		elif self.parameters() is None:

			def func(parameters=None,state=None,**kwargs):
				parameters = self.parameters(parameters,**kwargs) if parameters is not None else self.parameters(self.parameters(),**kwargs)
				return cos(parameters)*identity + -1j*sin(parameters)*self.data

			def gradient(parameters=None,state=None,**kwargs):
				parameters = self.parameters(parameters,**kwargs) if parameters is not None else self.parameters(self.parameters(),**kwargs)
				return (-sin(parameters)*identity + -1j*cos(parameters)*self.data)


		contract = None
		gradient_contract = None

		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.where = where if where is not None else self.where
		self.string = string if string is not None else self.string

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = gradient_contract

		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary

		return

class Haar(Object):
	'''
	Haar class for Quantum Objects
	Args:
		data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
		where (iterable[int]): location of local operators, i.e) nearest neighbour
		string (str): string label of operator
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional system keyword arguments
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.identity for attr in ['I']},
		**{attr: Basis.unitary for attr in ['unitary','haar','U','u']},
		}

	def setup(self,data=None,operator=None,where=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			where (iterable[int]): location of local operators, i.e) nearest neighbour
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		where = self.where if where is None else where
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		functions = ['unitary','haar','U','u']

		do = not self.null()

		if do:

			if ((isinstance(self.operator,str) and (self.operator in functions)) or
				(isinstance(self.operator,iterables) and any(i in self.operator for i in functions))):

				options = dict(D=self.D,N=self.locality//self.number,ndim=self.ndim,index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system)

				data = [i for i in self.operator] if isinstance(self.operator,iterables) else [self.operator]*(self.locality//Basis.localities(self.basis.get(self.operator),**options)) if isinstance(self.operator,str) else None
				_data = [] if self.local else [self.default]*(self.N-self.locality) if data is not None else None

				shape = Basis.shapes(attr=Basis.string,operator=[self.basis.get(i) for i in [*data,*_data]],**options) if data is not None else None
				axes = [*self.where,*(() if self.local else set(range(self.N))-set(self.where))] if data is not None else None
				ndim = Basis.dimensions(attr=Basis.string,operator=[self.basis.get(i) for i in [*data,*_data]],**options) if data is not None else None
				dtype = self.dtype

				shape = {axis: [shape[axis][axes.index(i)] for i in range(max(axes)+1) if i in axes] for axis in shape} if data is not None else None
				axes = [[i] for i in axes] if data is not None else None
				ndim = ndim if data is not None else None
				dtype = dtype

				local = self.local
				tensor = partial(self.tensor,N=self.locality if self.local else self.N,D=self.D,d=ndim) if self.tensor is not None else None

				data = [*data,*_data] if not self.local else data

				seed = seeder(self.seed)

				if self.architecture is None or self.architecture in ['array']:
					if self.local:
						options = Dictionary(
							D=self.D,N=self.locality//self.number,ndim=ndim,
							local=local,tensor=tensor,
							random=self.random,seed=seed,
							index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system,
							data=self.data,operator=data,
							basis=self.basis,axes=axes,shapes=shape,
							)
						data = options.operator
						if len(data)>1:
							options = {index: Dictionary(Basis.opts(options.basis.get(i),options)) for index,i in enumerate(data)}
							for index,i in zip(options,data):
								options[index].basis = options[index].basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options[list(options)[0]].tensor(tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options]))
							else:
								def function(parameters,state,options=options,**kwargs):
									return tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options])
						else:
							for i in data:
								options = Dictionary(Basis.opts(options.basis.get(i),options))
								options.basis = options.basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options.tensor(options.basis(**{**options,**kwargs}))
							else:
								def function(parameters,state,options=options,**kwargs):
									return options.basis(**{**options,**kwargs})
					else:
						options = Dictionary(
							D=self.D,N=self.locality//self.number,ndim=ndim,
							local=local,tensor=tensor,
							random=self.random,seed=seed,
							index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system,
							data=self.data,operator=data,
							basis=self.basis,axes=axes,shapes=shape,
							)
						data = options.operator
						if len(data)>1:
							options = {index: Dictionary(Basis.opts(options.basis.get(i),options)) for index,i in enumerate(data)}
							for index,i in zip(options,data):
								options[index].basis = options[index].basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options[list(options)[0]].tensor(swap(tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options]),axes=options[list(options)[0]].axes,shape=options[list(options)[0]].shapes))
							else:
								def function(parameters,state,options=options,**kwargs):
									return swap(tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options]),axes=options[list(options)[0]].axes,shape=options[list(options)[0]].shapes)
						else:
							for i in data:
								options = Dictionary(Basis.opts(options.basis.get(i),options))
								options.basis = options.basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options.tensor(options.basis(**{**options,**kwargs}))
							else:
								def function(parameters,state,options=options,**kwargs):
									return options.basis(**{**options,**kwargs})
				else:
					raise NotImplementedError

				def func(parameters=None,state=None,**kwargs):
					return function(parameters=parameters,state=state,**kwargs)

				def gradient(parameters=None,state=None,**kwargs):
					return None

				data = func(parameters=self.parameters(),state=self.state())

			else:

				data = self.data if data is None else data

		else:

			data = None


		variable = self.variable if self.variable is not None else None
		constant = True
		symmetry = None

		hermitian = False
		unitary = True

		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.where = where if where is not None else self.where
		self.string = string if string is not None else self.string

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = gradient_contract

		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary

		return


class Noise(Object):
	'''
	Noise class for Quantum Objects
	Args:
		data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
		where (iterable[int]): location of local operators, i.e) nearest neighbour
		string (str): string label of operator
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional system keyword arguments
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.identity for attr in ['noise','rand','eps']},
		**{attr: Basis.depolarize for attr in ['depolarize']},
		**{attr: Basis.amplitude for attr in ['amplitude']},
		**{attr: Basis.element for attr in ['element']},
		**{attr: Basis.bitflip for attr in ['flip','bitflip']},
		**{attr: Basis.phaseflip for attr in ['phaseflip','flipphase']},
		**{attr: Basis.dephase for attr in ['phase','dephase']}
		}

	def __init__(self,data=None,operator=None,where=None,string=None,system=None,**kwargs):

		setter(kwargs,dict(data=data,operator=operator,where=where,string=string,system=system),delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)

		return

	def setup(self,data=None,operator=None,where=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			where (iterable[int]): location of local operators, i.e) nearest neighbour
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		where = self.where if where is None else where
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		functions = ['noise','rand','eps']

		do = (not self.null()) and (self.parameters is not None) and (self.parameters() is not None)

		if do:

			if ((isinstance(self.operator,str) and (self.operator in functions)) or
				(isinstance(self.operator,iterables) and any(i in self.operator for i in functions))):

				options = Dictionary(D=self.D,N=self.locality//self.number,ndim=self.ndim,index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system)

				seed = seeder(self.seed)

				data = [i for i in self.operator] if isinstance(self.operator,iterables) else [self.operator]*(self.locality//Basis.localities(self.basis.get(self.operator),**options)) if isinstance(self.operator,str) else None

				if all(i in ['noise','rand'] for i in data):
					options = Dictionary(
						D=self.D,N=self.locality//self.number,ndim=self.ndim,
						random=self.random,bounds=[-1,1],seed=seed,
						index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system,
						data=self.data,operator=None,
						basis=self.basis,axes=axes,shapes=shape,
						)
					def function(parameters,state,options=options,**kwargs):
						return state + parameters*rand(**{**options,**dict(shape=state.shape,seed=options.seed,dtype=state.dtype)})/2
				elif all(i in ['eps'] for i in data):
					options = Dictionary(
						D=self.D,N=self.locality//self.number,ndim=self.ndim,
						random=self.random,bounds=[-1,1],seed=seed,
						index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system,
						data=self.data,operator=tensorprod([diag((1+self.parameters())**(arange(D)+2) - 1) for i in range(self.locality if self.local else self.N)]),
						basis=self.basis,axes=axes,shapes=shape,
						)
					def function(parameters,state,options=options,**kwargs):
						return options['data']
				else:
					options = Dictionary()
					def function(parameters,state,options=options,**kwargs):
						return None

				def func(parameters=None,state=None,**kwargs):
					return function(parameters=parameters,state=state,**kwargs)

				def gradient(parameters=None,state=None,**kwargs):
					return None

				def contract(data=None,state=None,where=None,**kwargs):
					return data

				def gradient_contract(grad=None,data=None,state=None,where=None,**kwargs):
					return grad

				data = func(parameters=self.parameters(),state=self.state())

			else:

				data = self.data if data is None else data

		else:

			data = None

		variable = self.variable if self.variable is not None else None
		constant = True
		symmetry = None

		hermitian = False
		unitary = False

		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.where = where if where is not None else self.where
		self.string = string if string is not None else self.string

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = gradient_contract

		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary

		return


class State(Object):
	'''
	State class for Quantum Objects
	Args:
		data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
		where (iterable[int]): location of local operators, i.e) nearest neighbour
		string (str): string label of operator
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional system keyword arguments
	'''

	N = None
	D = None

	default = 'I'
	basis = {
		**{attr: Basis.identity for attr in [default]},
		**{attr: Basis.identity for attr in ['I']},
		**{attr: Basis.data for attr in ['data']},
		**{attr: Basis.state for attr in ['state','psi']},
		**{attr: Basis.state for attr in ['haar']},
		**{attr: Basis.rand for attr in ['random','rand']},
		**{attr: Basis.zero for attr in ['zero','zeros','0']},
		**{attr: Basis.one for attr in ['one','ones','1']},
		**{attr: Basis.plus for attr in ['plus','+']},
		**{attr: Basis.minus for attr in ['minus','-']},
		**{attr: Basis.plusi for attr in ['plusi','+i']},
		**{attr: Basis.minusi for attr in ['minusi','-i']},
		**{attr: Basis.ghz for attr in ['ghz']},
		}

	def setup(self,data=None,operator=None,where=None,string=None,**kwargs):
		'''
		Setup operator
		Args:
			data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			where (iterable[int]): location of local operators, i.e) nearest neighbour
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''

		data = self.data if data is None else data
		operator = self.operator if operator is None else operator
		where = self.where if where is None else where
		string = self.string if string is None else string

		func = self.func
		gradient = self.gradient
		contract = None
		gradient_contract = None

		functions = ['state','psi','haar','random','rand']

		do = not self.null()

		if do:

			if ((isinstance(self.operator,str) and (self.operator in functions)) or
				(isinstance(self.operator,iterables) and any(i in self.operator for i in functions))):

				options = dict(D=self.D,N=self.locality//self.number,ndim=self.ndim,index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system)

				data = [i for i in self.operator] if isinstance(self.operator,iterables) else [self.operator]*(self.locality//Basis.localities(self.basis.get(self.operator),**options)) if isinstance(self.operator,str) else None
				_data = [] if self.local else [self.default]*(self.N-self.locality) if data is not None else None

				shape = {axis: shape if self.local else [*shape] for axis,shape in Basis.shapes(attr=Basis.string,operator=[self.basis.get(i) for i in [*data,*_data]],**options).items()} if data is not None else None
				axes = [i for i in range(self.locality)] if data is not None else None
				ndim = Basis.dimensions(attr=Basis.string,operator=[self.basis.get(i) for i in [*data,*_data]],**options) if data is not None else None
				dtype = self.dtype

				shape = {axis: [shape[axis][axes.index(i)] for i in range(max(axes)+1) if i in axes] for axis in shape} if data is not None else None
				axes = [[i] for i in axes] if data is not None else None
				ndim = ndim if data is not None else None
				dtype = dtype

				local = self.local
				tensor = partial(self.tensor,N=self.locality if self.local else self.N,D=self.D,d=ndim) if self.tensor is not None else None

				data = [*data,*_data] if not self.local else data

				seed = seeder(self.seed)

				if self.architecture is None or self.architecture in ['array']:
					if self.local:
						options = Dictionary(
							D=self.D,N=self.locality//self.number,ndim=ndim,
							local=local,tensor=tensor,
							random=self.random,seed=seed,
							index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system,
							data=self.data,operator=data,
							basis=self.basis,axes=axes,shapes=shape,
							)
						data = options.operator
						if len(data)>1:
							options = {index: Dictionary(Basis.opts(options.basis.get(i),options)) for index,i in enumerate(data)}
							for index,i in zip(options,data):
								options[index].basis = options[index].basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options[list(options)[0]].tensor(tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options]))
							else:
								def function(parameters,state,options=options,**kwargs):
									return tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options])
						else:
							for i in data:
								options = Dictionary(Basis.opts(options.basis.get(i),options))
								options.basis = options.basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options.tensor(options.basis(**{**options,**kwargs}))
							else:
								def function(parameters,state,options=options,**kwargs):
									return options.basis(**{**options,**kwargs})
					else:
						options = Dictionary(
							D=self.D,N=self.locality//self.number,ndim=ndim,
							local=local,tensor=tensor,
							random=self.random,seed=seed,
							index=self.index,architecture=self.architecture,dtype=self.dtype,system=self.system,
							data=self.data,operator=data,
							basis=self.basis,axes=axes,shapes=shape,
							)
						data = options.operator
						if len(data)>1:
							options = {index: Dictionary(Basis.opts(options.basis.get(i),options)) for index,i in enumerate(data)}
							for index,i in zip(options,data):
								options[index].basis = options[index].basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options[list(options)[0]].tensor(swap(tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options]),axes=options[list(options)[0]].axes,shape=options[list(options)[0]].shapes))
							else:
								def function(parameters,state,options=options,**kwargs):
									return swap(tensorprod([options[i].basis(**{**options[i],**kwargs}) for i in options]),axes=options[list(options)[0]].axes,shape=options[list(options)[0]].shapes)
						else:
							for i in data:
								options = Dictionary(Basis.opts(options.basis.get(i),options))
								options.basis = options.basis.get(i)
							if self.tensor is not None:
								def function(parameters,state,options=options,**kwargs):
									return options.tensor(options.basis(**{**options,**kwargs}))
							else:
								def function(parameters,state,options=options,**kwargs):
									return options.basis(**{**options,**kwargs})
				else:
					raise NotImplementedError

				def func(parameters=None,state=None,**kwargs):
					return function(parameters=parameters,state=state,**kwargs)

				def gradient(parameters=None,state=None,**kwargs):
					return None

				def contract(data=None,state=None,where=None):
					return data

				def gradient_contract(grad=None,data=None,state=None,where=None):
					return grad

				data = func(parameters=self.parameters(),state=self.state())

			else:

				data = self.data if data is None else data

		else:

			data = None

		variable = self.variable if self.variable is not None else None
		constant = True
		symmetry = None

		if self.ndim is None:
			hermitian = True
			unitary = False
		elif self.ndim == 1:
			hermitian = False
			unitary = True
		elif self.ndim == 2:
			hermitian = True
			unitary = False
		else:
			hermitian = True
			unitary = False


		self.data = data

		self.operator = operator if operator is not None else self.operator
		self.where = where if where is not None else self.where
		self.string = string if string is not None else self.string

		self.func = func
		self.gradient = gradient
		self.contract = contract
		self.gradient_contract = gradient_contract

		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary

		return

	def __call__(self,parameters=None,state=None,**kwargs):
		'''
		Call class
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional operator keyword arguments
		Returns:
			data (array): data
		'''
		if state is None:
			return self.func(parameters=parameters,state=state,**kwargs)
		else:
			return self.contract(self.func(parameters=parameters,state=state,**kwargs),state=state)

	def grad(self,parameters=None,state=None,**kwargs):
		'''
		Call class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional operator keyword arguments
		Returns:
			data (array): data
		'''
		if state is None:
			return self.gradient(parameters=parameters,state=state,**kwargs)
		else:
			return self.gradient_contract(self.gradient(parameters=parameters,state=state,**kwargs),self.func(parameters=parameters,state=state,**kwargs),state=state)


class Operator(Object):
	'''
	Class for Operator
	Args:
		data (str,array,tensor,mps,iterable[str,array,tensor,mps],dict): data of operator
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
		where (iterable[int]): location of local operators, i.e) nearest neighbour
		string (str): string label of operator
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional system keyword arguments
	'''

	N = None
	D = None

	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}}

	def __new__(cls,data=None,operator=None,where=None,string=None,system=None,**kwargs):

		# TODO: Allow multiple different classes to be part of one operator

		self = None

		setter(kwargs,dict(data=data,operator=operator,where=where,string=string,system=system),delimiter=delim,default=False)

		classes = [Data,Gate,Pauli,Haar,Noise,State,Channel,Operators,Unitary,Hamiltonian,Object]

		for subclass in classes:

			if any(isinstance(obj,subclass) for obj in [data if data is not None else operator if operator is not None else None]):

				self = subclass(**kwargs)

				break

			if not all(
				j in subclass.basis for obj in [operator if operator is not None else data if data is not None else None]
				if isinstance(obj,(str,*iterables))
				for k in (obj if isinstance(obj,iterables) else [obj])
				for j in ([k] if k in subclass.basis else k.split(delim) if all(j in subclass.basis for j in k.split(delim)) else [None])):
				continue

			if cls in [Operator]:

				self = subclass(**kwargs)

			else:

				for attr in subclass.__dict__:
					setattr(cls,attr,getattr(subclass,attr))

				self = subclass.__new__(cls,**kwargs)

				subclass.__init__(self,**kwargs)

			break

		assert (self is not None),'TODO: All operators not in same class'

		return self

class Objects(Object):
	'''
	Class for Operators
	Args:
		data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,where,string dictionary for operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			where (iterable[str,iterable[int,str]]): location of local operators
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional operator keyword arguments
		operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
		where (iterable[str,iterable[int,str]]): location of local operators
		string (iterable[str]): string labels of operators
		N (int): Size of system
		M (int): Duration of system
		D (int): Local dimension of system
		d (int): Spatial dimension of system
		T (int): Simulation time
		tau (float): Simulation time scale
		P (int): Trotter order
		space (str,dict,Space): Type of local space
		time (str,dict,Time): Type of time evolution
		lattice (str,dict,Lattice): Type of lattice
		parameters (iterable[str],dict,Parameters): Type of parameters of operators
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional system keyword arguments
	'''

	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}, **{attr: Basis.identity for attr in ['operators']}}

	def __init__(self,data=None,operator=None,where=None,string=None,
		N=None,M=None,D=None,d=None,T=None,tau=None,P=None,
		space=None,time=None,lattice=None,parameters=None,system=None,**kwargs):

		setter(kwargs,dict(
			data=data,operator=operator,where=where,string=string,
			N=N,M=M,D=D,d=d,T=T,tau=tau,P=P,
			space=space,time=time,lattice=lattice,parameters=parameters,system=system),
			delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)

		return

	def init(self,data=None,state=None,parameters=None,conj=False,**kwargs):
		'''
		Setup class functions
		Args:
			data (bool,dict): data of class, or boolean to retain current data attribute or initialize as None
			state (bool,array,Object): state of class
			parameters (dict,array,Parameters): parameters of class
			conj (bool): conjugate
			kwargs (dict): Additional class keyword arguments
		'''

		# Set attributes
		self.spaces()
		self.times()
		self.lattices()

		self.shape = () if self.D is None or self.locality is None else (self.D**self.locality,)*2
		self.size = prod(self.shape) if self.shape is not None else None
		self.ndim = len(self.shape) if self.shape is not None else None
		self.dtype = self.dtype if self.dtype is not None else None

		self.setup(data,where=self.where,operator=self.operator,string=self.string)

		# Set data
		if data is None:
			data = {}
		elif data is True:
			data = {}
		elif data is False:
			data = {i: None for i in self.data}
		elif isinstance(data,dicts):
			data = {i: data[i] for i in data if i in self.data}

		for i in data:
			if data[i] is True:
				self.data[i] = self.data[i]
			elif data[i] is None or data[i] is False:
				self.data[i] = None
			elif self.data[i] is None:
				self.data[i] = data[i]
			elif isinstance(data[i],type(self.data[i])):
				self.data[i] = data[i]
			elif isinstance(data[i],dicts):
				self.data[i].init(**data[i])
			else:
				self.data[i].init(data=data[i])


		# Set state
		if state is None:
			state = self.state
		elif state is True:
			state = self.state
		elif state is False:
			state = None

		if state is None or not callable(state):
			def state(parameters=None,state=state):
				return state

		self.state = state


		# Set conjugate
		if conj is None:
			conj = False
		elif conj is True:
			conj = True
		elif conj is False:
			conj = False

		self.conj = conj


		# Set parameters
		if parameters is None:
			parameters = None
		elif parameters is True:
			parameters = None
		elif parameters is False:
			parameters = None

		cls = Parameters
		keywords = dict(
			parameters={i:self.data[i].parameters
				for i in self.data
				if ((self.data[i] is not None) and
					(not self.data[i].null()))
				} if parameters is None else parameters if not isinstance(parameters,dicts) else None,
			system=self.system
		)

		parameters = cls(**keywords)

		self.parameters = parameters


		# Set kwargs
		for kwarg in kwargs:
			if hasattr(self,kwarg):
				setattr(self,kwarg,kwargs[kwarg])


		# Set identity
		def identity(N=None,D=None,self=self):
			data = None
			for i in self.data:
				if self.data[i].null():
					continue
				else:
					data = self.data[i].identity(N=N,D=D)
					break
			return data

		self.identity = identity

		# Set data
		for i in self.data:

			if self.data[i] is None:
				continue

			keywords = {**dict(state=self.state)}

			self.data[i].init(**keywords)

		# Set attributes
		self.update()

		# Set functions
		def func(parameters=None,state=None,**kwargs):
			state = state if state is not None else self.state() if self.state is not None and self.state() is not None else self.identity()
			shape = (self.M,len([i for i in self.data if self.data[i] is not None]))
			indices = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None]
			size = shape[1]
			kwargs = [Dictionary(**{**dict(index=self.data[i].index,seed=self.data[i].seed),**kwargs}) for i in self.data if self.data[i] is not None]
			for i in range(size):
				kwargs[i].seed = seeder(seed=kwargs[i].seed,size=size)[i]
			out = state
			if parameters is not None and len(parameters):
				for i in indices:
					kwargs[i%size].index = i
					out = self.data[i%size](parameters=parameters[i//size],state=out,**kwargs[i%size])
					seed,kwargs[i%size].seed = rng.split(kwargs[i%size].seed)
			else:
				for i in indices:
					kwargs[i%size].index = i
					out = self.data[i%size](parameters=parameters,state=out,**kwargs[i%size])
					seed,kwargs[i%size].seed = rng.split(kwargs[i%size].seed)
			return out

		def grad(parameters=None,state=None,**kwargs):
			state = state if state is not None else self.state() if self.state is not None and self.state() is not None else self.identity()
			grad = zeros((parameters.size,*state.shape),dtype=state.dtype)
			shape = (self.M,len([i for i in self.data if self.data[i] is not None]))
			indices = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None]
			indexes = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None and self.data[i].variable]
			size = shape[1]
			kwargs = [Dictionary(**{**dict(index=self.data[i].index,seed=self.data[i].seed),**kwargs}) for i in self.data if self.data[i] is not None]
			for i in range(size):
				kwargs[i].seed = seeder(seed=kwargs[i].seed,size=size)[i]
			if parameters is not None and len(parameters):
				for i in indexes:
					kwargs[i%size].index = i
					out = state
					for j in (j for j in indices if j<i):
						out = self.data[j%size](parameters=parameters[j//size],state=out,**kwargs[i%size])
					out = self.data[i%size].grad(parameters=parameters[i//size],state=out,**kwargs[i%size])
					for j in (j for j in indices if j>i):
						out = self.data[j%size](parameters=parameters[j//size],state=out,**kwargs[i%size])
					grad = inplace(grad,indexes.index(i),out,'add')
					seed,kwargs[i%size].seed = rng.split(kwargs[i%size].seed)
			else:
				for i in indexes:
					kwargs[i%size].index = i
					out = state
					for j in (j for j in indices if j<i):
						out = self.data[j%size](parameters=parameters,state=out,**kwargs[i%size])
					out = self.data[i%size].grad(parameters=parameters,state=out,**kwargs[i%size])
					for j in (j for j in indices if j>i):
						out = self.data[j%size](parameters=parameters,state=out,**kwargs[i%size])
					grad = inplace(grad,indexes.index(i),out,'add')
					seed,kwargs[i%size].seed = rng.split(kwargs[i%size].seed)
			return grad

		grad_automatic = gradient(self,mode='fwd',move=True)
		grad_finite = gradient(self,mode='finite',move=True)
		grad_analytical = grad

		grad = grad_automatic

		self.func = func
		self.gradient = grad
		self.gradient_automatic = grad_automatic
		self.gradient_finite = grad_finite
		self.gradient_analytical = grad_analytical


		# Set wrapper
		parameters = self.parameters(self.parameters())
		state = self.state() if self.state is not None and self.state() is not None else self.identity()
		wrapper = jit
		kwargs = {}

		self.func = wrapper(self.func,parameters=parameters,state=state,**kwargs)
		self.gradient = wrapper(self.gradient,parameters=parameters,state=state,**kwargs)
		self.gradient_automatic = wrapper(self.gradient_automatic,parameters=parameters,state=state,**kwargs)
		self.gradient_finite = wrapper(self.gradient_finite,parameters=parameters,state=state,**kwargs)
		self.gradient_analytical = wrapper(self.gradient_analytical,parameters=parameters,state=state,**kwargs)

		return

	def setup(self,data=None,operator=None,where=None,string=None,**kwargs):
		'''
		Setup class
		Args:
			data (dict[str,dict],iterable[Operator]): data for operators with key,values of operator name and operator,where,string dictionary for operator
				operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
				where (iterable[str,iterable[int,str]]): location of local operators
				string (iterable[str]): string labels of operators
				kwargs (dict): Additional operator keyword arguments
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			where (iterable[str,iterable[int,str]]): location of local operators
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional class keyword arguments
		'''

		# Get status of data
		if self.status(data):
			return

		# Get operator,where,string from data
		objs = Dictionary(operator=operator,where=where,string=string)

		# Get data and kwargs
		if (isinstance(data,dict) or data is None) and not all(isinstance(objs[obj],list) or objs[obj] is None for obj in objs):

			if data is None:
				data = {None:{}}

			for obj in objs:
				for name in data:
					if obj not in data[name] and objs[obj] is not None:
						data[name][obj] = objs[obj]

				objs[obj] = []

		for obj in objs:
			objs[obj] = [] if not isinstance(objs[obj],list) else objs[obj]


		if isinstance(data,dict):
			for name in data:
				if isinstance(data[name],dict):
					for obj in objs:
						if obj not in data[name]:
							data[name][obj] = None
			for obj in objs:
				objs[obj].extend([data[name].get(obj) for name in data if data[name] is not None])

			kwargs.update({kwarg: [data[name][kwarg] if kwarg in data[name] else null for name in data if data[name] is not None]
				for kwarg in set(kwarg for name in data if data[name] is not None for kwarg in data[name] if kwarg not in objs)
				})

		data = objs

		# Lattice
		lattice = self.lattice
		locality = self.N
		indices = self.lattice.structures

		# Get number of operators
		size = min([len(data[obj]) for obj in data if data[obj] is not None],default=0)

		# Get attribute of symbolic indices
		attrs = dict(
			where = lambda attr,value,values,indices: [dict(zip(indices,
								value if not isinstance(value,integers) else (value,))
							).get(i,int(i) if not isinstance(i,str) else i)
							for i in values[attr]]
			)

		# Get data
		for index in range(size):

			key = None
			tmp = {obj: copy(data[obj].pop(0)) for obj in data}
			tmps = {kwarg: copy(kwargs[kwarg].pop(0)) for kwarg in kwargs}

			for attr in attrs:
				if tmp[attr] is None:
					key = True
				elif isinstance(tmp[attr],str) and tmp[attr] in indices:
					if len(indices[tmp[attr]]) > locality:
						key = False
						tmp[attr] = None
					else:
						key = tmp[attr]
						tmp[attr] = indices[tmp[attr]]
				elif isinstance(tmp[attr],str) and tmp[attr] not in indices:
					raise NotImplementedError("Index %s Not Implemented"%(tmp[attr]))
				elif isinstance(tmp[attr],scalars):
					if tmp[attr] > locality:
						key = False
						tmp[attr] = None
					else:
						tmp[attr] = [tmp[attr]]
				elif not isinstance(tmp[attr],scalars):
					if any((i in indices and len(indices[tmp[attr]]) > locality) or (i > locality) for i in tmp[attr]):
						key = False
						tmp[attr] = None
					else:
						for i in tmp[attr]:
							if i in indices:
								key = i
								tmp[attr][tmp[attr].index(i)] = indices[i][tmp[attr].index(i)]
			if key is True:
				for obj in data:
					data[obj].append(tmp[obj])

				for kwarg in kwargs:
					kwargs[kwarg].append(copy(tmps[kwarg]))
			elif key is False:
				for obj in data:
					data[obj].append(None)

				for kwarg in kwargs:
					kwargs[kwarg].append(None)
			elif key is not None:
				for i,index in enumerate(lattice(key)):
					value = {}
					for obj in data:

						if obj in attrs:
							value[obj] = attrs[obj](obj,index,tmp,indices[key])
						else:
							value[obj] = tmp[obj]

					for obj in data:
						data[obj].append(value[obj])

					for kwarg in kwargs:
						kwargs[kwarg].append(copy(tmps[kwarg]))

			else:
				for obj in data:
					data[obj].append(tmp[obj])

				for kwarg in kwargs:
					kwargs[kwarg].append(copy(tmps[kwarg]))


		# Set class dependent attributes
		# i.e) set parameters data with where-dependent data
		# i.e) set parameters shape with depth-M-dependent shape
		attributes = {}
		def decorator(func):
			def wrapper(i,attr,attrs,data,kwargs):
				if kwargs.get(attr) is None:
					kwargs[attr] = [None for j in range(i)]
				if len(kwargs[attr]) < i:
					kwargs[attr] = [*kwargs[attr],*[None for j in range(1+i-len(kwargs[attr]))]]
				func(i,attr,attrs,data,kwargs)
				return
			return wrapper

		attribute = 'parameters.data'
		@decorator
		def func(i,attr,attrs,data,kwargs):

			if not isinstance(kwargs[attr][i],dict):
				tmp = None if isinstance(kwargs[attr][i],nulls) else kwargs[attr][i]
				kwargs[attr][i] = dict(data=tmp)

			size = len(kwargs.get(attr,[]))
			index = [j for j in range(size) if data.string[j] == data.string[i]].index(i)
			obj = getter(kwargs[attr][i],attrs,delimiter=delim)
			default = None

			if obj is None:
				obj = obj
			elif isinstance(obj,dict):
				obj = obj.get(
					tuple(data.where[i]),obj.get(str(data.string[i]),
					obj.get(str(index),obj.get(int(index),default))))
			elif isinstance(obj,iterables):
				obj = obj[index%len(obj)] if len(obj) else default
			else:
				obj = obj

			setter(kwargs[attr][i],{attrs:obj},delimiter=delim,default=True)
			return
		attributes[attribute] = func

		attribute = 'parameters.axis'
		@decorator
		def func(i,attr,attrs,data,kwargs):

			if not isinstance(kwargs[attr][i],dict):
				tmp = None if isinstance(kwargs[attr][i],nulls) else kwargs[attr][i]
				kwargs[attr][i] = dict(data=tmp)

			size = len(kwargs.get(attr,[]))
			index = [j for j in range(size) if data.string[j] == data.string[i]].index(i)
			obj = getter(kwargs[attr][i],attrs,delimiter=delim)
			default = ['M']

			if obj is None:
				obj = [j for j in default]
			elif isinstance(obj,iterables):
				obj = [*obj,*(j for j in default if j not in obj)]
			else:
				obj = obj

			setter(kwargs[attr][i],{attrs:obj},delimiter=delim,default=True)
			return
		attributes[attribute] = func

		for attribute in attributes:
			attr,attrs = attribute.split(delim)[0] if attribute.count(delim)>=0 else None,delim.join(attribute.split(delim)[1:]) if attribute.count(delim)>0 else None
			size = len(kwargs.get(attr,[]))

			for i in range(size):
				if isinstance(kwargs[attr][i],nulls):
					continue
				attributes[attribute](i,attr,attrs,data,kwargs)


		# Set class attributes
		self.extend(**data,kwargs=kwargs)

		return


	def __call__(self,parameters=None,state=None,**kwargs):
		'''
		Class function
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional function keyword arguments
		Returns
			out (array): Return of function
		'''

		parameters = self.parameters(parameters) if parameters is not None else self.parameters(self.parameters())

		return self.func(parameters=parameters,state=state,**kwargs)


	def grad(self,parameters=None,state=None,**kwargs):
		'''
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional function keyword arguments
		Returns:
			out (array): Return of function
		'''
		return self.gradient(parameters=parameters,state=state,**kwargs)


	def grad_automatic(self,parameters=None,state=None,**kwargs):
		'''
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional function keyword arguments
		Returns:
			out (array): Return of function
		'''
		return self.gradient_automatic(parameters=parameters,state=state,**kwargs)


	def grad_finite(self,parameters=None,state=None,**kwargs):
		'''
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional function keyword arguments
		Returns:
			out (array): Return of function
		'''
		return self.gradient_finite(parameters=parameters,state=state,**kwargs)


	def grad_analytical(self,parameters=None,state=None,**kwargs):
		'''
		Class gradient
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional function keyword arguments
		Returns:
			out (array): Return of function
		'''

		parameters = self.parameters(parameters) if parameters is not None else self.parameters(self.parameters())

		return self.gradient_analytical(parameters=parameters,state=state,**kwargs)


	def __len__(self):
		return len(self.data)

	def __str__(self):
		if self.data is not None:
			size = len(self) if self.data is not None else 0
			delimiter = ' '
			multiple_time = (self.M>1) if self.M is not None else None
			multiple_space = [size>1 and False for i in range(size)]
			string = '%s%s%s%s'%(
					'{' if multiple_time else '',
					delimiter.join(['%s%s%s'%(
						'(' if multiple_space[i] else '',
						self.data[i].string,
						')' if multiple_space[i] else '',
						) for i in range(size) if self.data[i] is not None]),
					'}' if multiple_time else '',
					'%s'%('^%s'%(self.M) if multiple_time else '') if multiple_time else '')
		else:
			string = self.__class__.__name__
		return string


	def status(self,data=None):
		'''
		Check status of class data
		Args:
			data (dict[Object]): class data
		Returns:
			status (bool): Status of data
		'''

		data = self.data if data is None else data

		cls = Object
		status = data is not None and all(isinstance(data[i],cls) or data[i] is None or data[i] is True or data[i] is False for i in data)

		return status


	def set(self,data=None,index=None):
		'''
		Set data of class
		Args:
			data (dict): data for class
			index (object): index of data for class
		'''

		if not self.status():
			self.clear()

		if index is not None and data is not None:

			index = len(self)+1+index if index < 0 else index

			self.data = insertion(self.data,index,{index:data})

		elif data is not None:

			self.data = type(self.data)({index:data[i] for index,i in enumerate(data)})

		return

	def get(self,index=None):
		'''
		Get data of class
		Args:
			index (object): index of data of class
		Returns:
			data (object): data of class
		'''

		if index is not None:

			index = len(self) + index if index < 0 else index

			data = self.data.get(index)

		else:

			data = self.data

		return data

	def clear(self):
		'''
		Clear data of class
		'''

		self.data = Dictionary()

		return

	def layout(self,configuration=None):
		'''
		Sort data of class
		Args:
			configuration (dict): configuration options for layout
				key (object,iterable[object],iterable[callable],callable): group iterable by key, iterable of keys, callable, or iterable of callables, with signature key(value)
		'''

		self.set()

		data = self.get()

		configuration = self.configuration if configuration is None else configuration

		options = {attr: configuration.get(attr,default) for attr,default in dict(key=None,options=None).items()} if configuration is not None else {}

		data = {index: data[i] for index,i in enumerate(sortby(data,**options))}

		self.set(data)

		return

	def extend(self,data=None,operator=None,where=None,string=None,kwargs=None):
		'''
		Extend to class
		Args:
			data (iterable[str,Operator]): data of operator
			operator (str,iterable[str]): name of operator, i.e) locality-length delimiter-separated string of operators 'X_Y_Z' or locality-length iterable of operator strings['X','Y','Z']
			where (iterable[str,iterable[int,str]]): location of local operators
			string (iterable[str]): string labels of operators
			kwargs (dict): Additional operator keyword arguments
		'''

		size = min([len(i) for i in (data,operator,where,string) if i is not None and not all(j is None for j in i)],default=0)

		length = min([len(i) for i in (kwargs[kwarg] for kwarg in kwargs) if i is not null],default=size) if kwargs is not None else None
		kwargs = [{kwarg: kwargs[kwarg][i] for kwarg in kwargs} for i in range(length)] if kwargs is not None else None

		if not size:
			self.set()
			return

		if data is None:
			data = [None]*size
		if operator is None:
			operator = [None]*size
		if where is None:
			where = [None]*size
		if string is None:
			string = [None]*size
		if kwargs is None:
			kwargs = [None]*size

		for _data,_operator,_where,_string,_kwargs in zip(data,operator,where,string,kwargs):

			self.append(_data,_operator,_where,_string,_kwargs)

		return


	def append(self,data=None,operator=None,where=None,string=None,kwargs=None):
		'''
		Append to class
		Args:
			data (str,Operator): data of operator
			operator (str): string name of operator
			where (int): where of local operator
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''
		index = -1
		self.insert(index,data,operator,where,string,kwargs)
		return


	def insert(self,index,data,operator,where,string,kwargs=None):
		'''
		Insert to class
		Args:
			index (int): index to insert operator
			data (str,Operator): data of operator
			operator (str): string name of operator
			where (int): where of local operator
			string (str): string label of operator
			kwargs (dict): Additional operator keyword arguments
		'''

		status = self.status()

		if all(obj is None for obj in (data,operator,where,string)):
			self.set()
			return

		cls = Operator
		defaults = {}
		kwargs = {kwarg: kwargs[kwarg] for kwarg in kwargs if not isinstance(kwargs[kwarg],nulls)} if kwargs is not None else defaults

		setter(kwargs,{attr: getattr(self,attr) for attr in self if attr not in cls.defaults and attr not in ['N','local','locality'] and attr not in ['data','operator','where','string']},delimiter=delim,default=False)
		setter(kwargs,dict(N=self.N,D=self.D,local=self.local,tensor=self.tensor,samples=self.samples),delimiter=delim,default=False)
		setter(kwargs,dict(state=self.state,system=self.system),delimiter=delim,default=True)
		setter(kwargs,dict(verbose=False),delimiter=delim,default=True)
		setter(kwargs,defaults,default=False)

		data = cls(**{**dict(data=data,operator=operator,where=where,string=string),**kwargs})

		self.set(data,index=index)

		self.layout()

		self.update(status)

		return


	def update(self,status=None,data=None):
		'''
		Update class attributes
		Args:
			status (bool): Class status
			data (dict): Class data
		'''

		cls = Object

		status = self.status() if status is None else status
		data = self.data if data is None else data
		state = self.state if isinstance(self.state,cls) else self.state() if callable(self.state) else self.state

		boolean = lambda i=None,data=None: ((data is not None) and (data[i] is not None) and (not data[i].null()))

		where = [j for i in data if boolean(i,data) and isinstance(data[i].where,iterables) for j in data[i].where]
		where = sorted(set(where),key=lambda i:where.index(i))

		operator = separ.join([
					delim.join(data[i].operator) if isinstance(data[i].operator,iterables) else data[i].operator
					for i in data if boolean(i,data) and data[i].operator is not None])

		string = separ.join([data[i].string for i in data if boolean(i,data)]) if data is not None else None

		tensor = None
		for i in data:
			tensor = data[i].tensor if boolean(i,data) and data[i].tensor is not None else None

		local = all(data[i].local for i in data if boolean(i,data)) if data is not None else None

		locality = len(where) if where is not None else None

		number = max((data[i].number for i in data if boolean(i,data)),default=None) if data is not None else None

		variable = any(data[i].variable for i in data) if data is not None else False
		constant = all(data[i].constant for i in data) if data is not None else False
		symmetry = [data[i].symmetry for i in data if data[i] and data[i].symmetry is not None][0] if data is not None and any(data[i].symmetry for i in data if data[i]) else None

		if state is None:
			hermitian = all(data[i].hermitian for i in data if boolean(i,data))
			unitary = all(data[i].unitary for i in data if boolean(i,data))
		elif state.ndim == 1:
			hermitian = False
			unitary = True
		elif state.ndim == 2:
			hermitian = True
			unitary = False

		N = max((i for i in (
			self.N if self.N is not None else None,
			max([data[i].N for i in data if boolean(i,data) and data[i].N is not None],default=None),
			) if i is not None),default=self.N) if data is not None else None

		D = max((i for i in (
			max([data[i].D for i in data if boolean(i,data) and data[i].D is not None],default=None),
			) if i is not None),default=self.D) if data is not None else None

		self.data = data

		self.operator = operator
		self.where = where
		self.string = string

		self.tensor = tensor
		self.local = local
		self.locality = locality
		self.number = number
		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary

		self.N = N
		self.D = D

		self.spaces()
		self.times()
		self.lattices()

		shape = () if self.D is None or self.locality is None else (self.D**self.locality,)*2
		size = prod(self.shape)
		ndim = len(self.shape)
		dtype = self.dtype if self.dtype is not None else None

		self.shape = shape
		self.size = size
		self.ndim = ndim
		self.dtype = dtype

		return


	def dump(self,path=None):
		'''
		Save class data
		Args:
			path (str,dict[str,(str,bool)]): Path to dump class data, either path or boolean to dump
		'''

		# Set path
		paths = {}
		if path is None:
			paths.update({attr: True for attr in data})
		elif not isinstance(path,dict):
			paths.update({attr: path for attr in data if path})
		else:
			paths.update({attr: path[attr] for attr in path if path[attr]})

		paths.update({attr: paths.get(attr) if isinstance(paths.get(attr),str) else self.cwd for attr in data if paths.get(attr)})

		# Set data
		data = {}

		# Set options
		options = dict(lock=self.lock,backup=self.backup)

		# Dump data
		for attr in paths:
			root,file = split(paths[attr],directory=True,file_ext=True)
			file = file if file is not None else self.path
			path = join(file,root=root)
			dump(data[attr],path,**options)

		return

	def load(self,path=None):
		'''
		Load class data
		Args:
			path (str,dict[str,(str,bool)]): Path to load class data, either path or boolean to load
		Returns:
			data (object): Class data
		'''

		# Set data
		data = {}

		# Set path
		paths = {}
		if path is None:
			paths.update({attr: True for attr in data})
		elif not isinstance(path,dict):
			paths.update({attr: path for attr in data if path})
		else:
			paths.update({attr: path[attr] for attr in path if path[attr]})

		paths.update({attr: paths.get(attr) if isinstance(paths.get(attr),str) else self.cwd for attr in data if paths.get(attr)})

		# Load data
		for attr in paths:
			root,file = split(paths[attr],directory=True,file_ext=True)
			file = file if file is not None else self.path
			path = join(file,root=root)
			func = (list,)
			default = data[attr]
			data[attr] = load(path,default=default)
			setter(default,data[attr],default=func)

		return data

class Channel(Objects):
	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}, **{attr: Basis.identity for attr in ['Channel']}}

	def init(self,data=None,state=None,parameters=None,conj=False,**kwargs):
		'''
		Setup class functions
		Args:
			data (bool,dict): data of class, or boolean to retain current data attribute or initialize as None
			state (bool,array,Object): state of class
			parameters (dict,array,Parameters): parameters of class
			conj (bool): conjugate
		'''

		super().init(data=data,state=state,parameters=parameters,conj=conj,**kwargs)

		# Set data
		for i in self.data:

			if self.data[i] is None:
				continue

			keywords = {
				**dict(
					parameters=dict(parameters=dict(tau=self.tau) if self.data[i].unitary else None),
					state=self.state,
					),
				**kwargs}
			self.data[i].init(**keywords)


		# Set attributes
		self.update()

		# Set functions
		func = scheme(data=self.data,parameters=self.parameters,state=self.state,conj=self.conj,size=self.M,compilation=dict(trotter=self.P,**self),verbose=self.verbose)
		grad = gradient_scheme(data=self.data,parameters=self.parameters,state=self.state,conj=self.conj,size=self.M,compilation=dict(trotter=self.P,**self),verbose=self.verbose)

		grad_automatic = gradient(self,mode='fwd',move=True)
		grad_finite = gradient(self,mode='finite',move=True)
		grad_analytical = grad

		grad = grad_automatic

		self.func = func
		self.gradient = grad
		self.gradient_automatic = grad_automatic
		self.gradient_finite = grad_finite
		self.gradient_analytical = grad_analytical


		# Set wrapper
		parameters = self.parameters(self.parameters())
		state = self.state() if self.state is not None and self.state() is not None else self.identity()
		wrapper = jit
		kwargs = {}

		self.func = wrapper(self.func,parameters=parameters,state=state,**kwargs)
		self.gradient = wrapper(self.gradient,parameters=parameters,state=state,**kwargs)
		self.gradient_automatic = wrapper(self.gradient_automatic,parameters=parameters,state=state,**kwargs)
		self.gradient_finite = wrapper(self.gradient_finite,parameters=parameters,state=state,**kwargs)
		self.gradient_analytical = wrapper(self.gradient_analytical,parameters=parameters,state=state,**kwargs)

		return

class Operators(Objects):

	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}, **{attr: Basis.identity for attr in ['operators']}}

	def init(self,data=None,state=None,parameters=None,conj=False,**kwargs):
		'''
		Setup class functions
		Args:
			data (bool,dict): data of class, or boolean to retain current data attribute or initialize as None
			state (bool,array,Object): state of class
			parameters (dict,array,Parameters): parameters of class
			conj (bool): conjugate
			kwargs (dict): Additional class keyword arguments
		'''

		super().init(data=data,state=state,parameters=parameters,conj=conj,**kwargs)

		# Set data
		for i in self.data:

			if self.data[i] is None:
				continue

			keywords = {**dict(state=self.state),**kwargs}

			self.data[i].init(**keywords)


		# Set attributes
		self.update()

		# Set functions
		def func(parameters=None,state=None,**kwargs):
			state = state if state is not None else self.state() if self.state is not None and self.state() is not None else self.identity()
			shape = (self.M,len([i for i in self.data if self.data[i] is not None]))
			indices = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None]
			size = shape[1]
			kwargs = [Dictionary(**{**dict(index=self.data[i].index,seed=self.data[i].seed),**kwargs}) for i in self.data if self.data[i] is not None]
			for i in range(size):
				kwargs[i].seed = seeder(seed=kwargs[i].seed,size=size)[i]
			out = state
			if parameters is not None and len(parameters):
				for i in indices:
					kwargs[i%size].index = i
					out = self.data[i%size](parameters=parameters[i//size],state=out,**kwargs[i%size])
					seed,kwargs[i%size].seed = rng.split(kwargs[i%size].seed)
			else:
				for i in indices:
					kwargs[i%size].index = i
					out = self.data[i%size](parameters=parameters,state=out,**kwargs[i%size])
					seed,kwargs[i%size].seed = rng.split(kwargs[i%size].seed)
			return out

		def grad(parameters=None,state=None,**kwargs):
			state = state if state is not None else self.state() if self.state is not None and self.state() is not None else self.identity()
			grad = zeros((parameters.size,*state.shape),dtype=state.dtype)
			shape = (self.M,len([i for i in self.data if self.data[i] is not None]))
			indices = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None]
			indexes = [j*shape[1]+i for j in range(shape[0]) for i in self.data if self.data[i] is not None and self.data[i].variable]
			size = shape[1]
			kwargs = [Dictionary(**{**dict(index=self.data[i].index,seed=self.data[i].seed),**kwargs}) for i in self.data if self.data[i] is not None]
			for i in range(size):
				kwargs[i].seed = seeder(seed=kwargs[i].seed,size=size)[i]
			if parameters is not None and len(parameters):
				for i in indexes:
					kwargs[i%size].index = i
					out = state
					for j in (j for j in indices if j<i):
						out = self.data[j%size](parameters=parameters[j//size],state=out,**kwargs[i%size])
					out = self.data[i%size].grad(parameters=parameters[i//size],state=out,**kwargs[i%size])
					for j in (j for j in indices if j>i):
						out = self.data[j%size](parameters=parameters[j//size],state=out,**kwargs[i%size])
					grad = inplace(grad,indexes.index(i),out,'add')
					seed,kwargs[i%size].seed = rng.split(kwargs[i%size].seed)
			else:
				for i in indexes:
					kwargs[i%size].index = i
					out = state
					for j in (j for j in indices if j<i):
						out = self.data[j%size](parameters=parameters,state=out,**kwargs[i%size])
					out = self.data[i%size].grad(parameters=parameters,state=out,**kwargs[i%size])
					for j in (j for j in indices if j>i):
						out = self.data[j%size](parameters=parameters,state=out,**kwargs[i%size])
					grad = inplace(grad,indexes.index(i),out,'add')
					seed,kwargs[i%size].seed = rng.split(kwargs[i%size].seed)
			return grad

		grad_automatic = gradient(self,mode='fwd',move=True)
		grad_finite = gradient(self,mode='finite',move=True)
		grad_analytical = grad

		grad = grad_automatic

		self.func = func
		self.gradient = grad
		self.gradient_automatic = grad_automatic
		self.gradient_finite = grad_finite
		self.gradient_analytical = grad_analytical

		# Set wrapper
		parameters = self.parameters(self.parameters())
		state = self.state() if self.state is not None and self.state() is not None else self.identity()
		wrapper = partial
		kwargs = {}

		self.func = wrapper(self.func,parameters=parameters,state=state,**kwargs)
		self.gradient = wrapper(self.gradient,parameters=parameters,state=state,**kwargs)
		self.gradient_automatic = wrapper(self.gradient_automatic,parameters=parameters,state=state,**kwargs)
		self.gradient_finite = wrapper(self.gradient_finite,parameters=parameters,state=state,**kwargs)
		self.gradient_analytical = wrapper(self.gradient_analytical,parameters=parameters,state=state,**kwargs)

		return

class Hamiltonian(Channel):
	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}, **{attr: Basis.identity for attr in ['Hamiltonian']}}

class Unitary(Channel):
	default = 'I'
	basis = {**{attr: Basis.identity for attr in [default]}, **{attr: Basis.identity for attr in ['Unitary']}}


class Module(System):
	'''
	Class for Module
	Args:
		model (Object,iterable[Object],dict[str,Object): model for module, iterable of models or dictionary of models
		N (int): Size of system
		M (int): Duration of system
		state (array,State): state for module
		parameters (iterable[str],dict,Parameters): Type of parameters of operators
		system (dict,System): System attributes (string,dtype,format,device,backend,architecture,configuration,key,index,seed,seeding,random,instance,instances,samples,base,unit,cwd,path,lock,backup,timestamp,conf,logger,cleanup,verbose,options)
		kwargs (dict): Additional system keyword arguments
	'''

	N = None
	D = None
	d = None

	defaults = dict(
		model=None,
		N=None,M=None,
		state=None,parameters=None,
		variable=None,constant=None,symmetry=None,hermitian=None,unitary=None,
		data=None,measure=None,callback=None,
		func=None,gradient=None,
		system=None,
		)

	def __init__(self,model=None,N=None,M=None,
		state=None,parameters=None,system=None,**kwargs):

		setter(kwargs,dict(
			model=model,N=N,M=M,
			state=state,parameters=parameters,system=system),
			delimiter=delim,default=False)
		setter(kwargs,system,delimiter=delim,default=False)
		setter(kwargs,self.defaults,delimiter=delim,default=False)

		super().__init__(**kwargs)

		self.init()

		self.info()

		return

	def init(self,model=None,state=None,parameters=None,**kwargs):
		'''
		Setup class functions
		Args:
			model (Object,iterable[Object],dict[str,Object): model for module, iterable of models or dictionary of models
			state (State): state for module
			parameters (dict,array,Parameters): parameters of class
			kwargs (dict): Additional class keyword arguments
		'''

		# Set model
		model = self.model if model is None else model

		self.model = model

		# Set parameters
		parameters = self.parameters if parameters is None else parameters
		if parameters is None or not callable(parameters):
			try:
				parameters = self.model.parameters
			except:
				parameters = parameters
		if parameters is None or not callable(parameters):
			def parameters(parameters=parameters):
				return parameters

		self.parameters = parameters


		# Set state
		state = self.state if state is None else state

		if state is None or not callable(state):
			def state(parameters=None,state=state):
				return state

		self.state = state


		# Set kwargs
		for kwarg in kwargs:
			if hasattr(self,kwarg):
				setattr(self,kwarg,kwargs[kwarg])


		# Setup
		self.setup()

		return

	def setup(self,model=None,state=None,parameters=None,**kwargs):
		'''
		Setup class
		Args:
			model (Object,iterable[Object],dict[str,Object): model for module, iterable of models or dictionary of models
			state (State): state for module
			parameters (dict,array,Parameters): parameters of class
			kwargs (dict): Additional class keyword arguments
		'''

		# Settings
		model = self.model if model is None else model
		state = self.state if state is None else state
		parameters = self.parameters if parameters is None else parameters

		self.model = model
		self.state = state
		self.parameters = parameters

		# Attributes
		self.set(model=model)
		self.layout()

		options = self.options if self.options is not None else {}
		boolean = lambda model: ((model is not None) and (not model.null()))

		N = max((model.N for index in self.model for model in self.model[index] if boolean(model) and model.N is not None),default=self.N)
		D = max((model.D for index in self.model for model in self.model[index] if boolean(model) and model.D is not None),default=None)

		self.N = N

		# Measure
		cls = Measure
		measure = self.measure if isinstance(self.measure,dicts) else {}
		measure = {**namespace(cls,self),**{attr: getattr(self,attr) for attr in (self.system if isinstance(self.system,dicts) else {}) if hasattr(self,attr)},**measure,**dict(N=N,D=D,system=self.system)}
		measure = cls(**measure)

		self.measure = measure

		self.measure.info()

		# Data

		self.set(model=model)

		self.layout()

		data = []
		wrapper = jit

		for index in self.model:

			if not self.model[index]:
				continue

			where = [i for model in self.model[index] if boolean(model) and isinstance(model.where,iterables) for i in model.where]
			where = sorted(set(where),key=lambda i:where.index(i))

			N = max((model.N for model in self.model[index] if boolean(model) and model.N is not None),default=len(where))
			D = max((model.D for model in self.model[index] if boolean(model) and model.D is not None),default=measure.D)
			locality = len(where)

			cls = self.state.__class__
			keywords = dict(local=False,tensor=True,verbose=False)
			state = cls(**{**self.state,**keywords})

			cls = {model:model.__class__ for model in self.model[index]}
			keywords = {model:dict(
				state=state @ locality,
				where=[where.index(i) for i in model.where],
				samples=[D**2]*locality,
				local=True,
				tensor=True,
				verbose=False,
				) for model in self.model[index]}
			model = [wrapper(cls[model](**{**model,**keywords[model]})) for model in self.model[index]]

			if len(model) > 1:
				def model(parameters,state,model=model,**kwargs):
					for func in model:
						state = func(parameters=parameters,state=state,**kwargs)
					return state
			else:
				model, = model

			parameters = measure.parameters()
			state = [self.state]*N

			model = measure.transform(parameters=parameters,state=state,model=model,where=where,options=options,**kwargs)

			def func(parameters,state,where=where,model=model,options=options,**kwargs):
				return model(parameters=parameters,state=state,where=where,options=options,**kwargs)

			data.append(func)

		self.data = data


		if self.parameters is not None and self.parameters() is not None:
			variable = True
			constant = False
			symmetry = None
		else:
			variable = False
			constant = True
			symmetry = None

		hermitian = True
		unitary = False

		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary


		# Functions

		data = self.data
		options = self.options

		def func(parameters,state,options=options,**kwargs):
			M,N,index,size = self.M,self.N,None,len(self.data)
			kwargs = [Dictionary(**{**dict(index=index,seed=self.seed,options=options),**kwargs}) for i in range(len(self.data))]
			attrs = copy(kwargs)
			for i in range(size):
				kwargs[i].seed = seeder(seed=kwargs[i].seed,size=size)[i]
			state = [state]*N if isinstance(state,arrays) or not isinstance(state,iterables) else state
			state = self.measure.transform(parameters=parameters,state=state)

			parameters = array([parameters]*M) if isinstance(parameters,scalars) or isinstance(parameters,arrays) and parameters.ndim == 1 else parameters
			parameters = parameters if parameters is not None else parameters

			for l in range(M):
				for i,data in enumerate(self.data):

					# print('index',l,i)
					kwargs[i].index = (l,i)
					state = data(parameters=parameters[l],state=state,**kwargs[i])
					seed,kwargs[i].seed = rng.split(kwargs[i].seed)
					# spectrum = self.measure.spectrum_quantum(parameters=parameters,state=state)
					# ratio = -addition(spectrum[spectrum<0])/addition(spectrum[spectrum>0])
					# trace = self.measure.trace(parameters=parameters,state=state)

					# if self.measure.architecture in ['tensor']:
					# 	trace = trace.array().item()
					# elif self.measure.architecture in ['tensor_quimb']:
					# 	trace = representation_quimb(trace,to=self.measure.architecture,contraction=True)
					# trace = trace.real-1
					# data = state
					# print('spectrum',ratio,spectrum[0],spectrum[1],spectrum[-2],spectrum[-1])
					# print('trace',trace)
					# # where = [i,i+1]
					# # for i in data:
					# # 	if i < min(where):
					# # 		print(i,addition(data[i].data,(0,1)))
					# # 	elif i > max(where):
					# # 		print(i,addition(data[i].data,(-2,-1)))
					# # print(where,addition(dot(data[min(where)].data,data[max(where)].data)))
					# # print('data',data)
					# print()

			return state

		def grad(parameters=None,state=None,options=options,**kwargs):
			return None

		self.func = func
		self.gradient = grad


		# Wrapper
		parameters = self.parameters()
		state = self.state()
		wrapper = partial
		kwargs = {}

		self.func = wrapper(self.func,parameters=parameters,state=state,**kwargs)
		self.gradient = wrapper(self.gradient,parameters=parameters,state=state,**kwargs)

		return

	def status(self,model=None):
		'''
		Check status of class model
		Args:
			model (dict[Object]): class model
		Returns:
			status (bool): Status of model
		'''

		model = self.model if model is None else model

		cls = Object

		status = model is not None and isinstance(model,dicts) and not isinstance(model,cls) and all(isinstance(instance,cls) for index in model for instance in model[index])

		return status


	def set(self,model=None,index=None):
		'''
		Set model of class
		Args:
			model (dict): model for class
			index (object): index of model for class
		'''

		if not self.status():
			self.clear()

		if index is not None and model is not None:

			index = len(self)+1+index if index < 0 else index

			self.model = insertion(self.model,index,{index:[model]})

		elif model is not None:

			cls = Object
			if isinstance(model,cls) and isinstance(model.data,Dictionary):
				model = {index:[model.data[key]] for index,key in enumerate(model.data)}
			elif isinstance(model,cls) and not isinstance(model.data,dicts):
				model = {None:[model]}
			elif isinstance(model,dicts) and all(isinstance(model[key],cls) for key in model):
				model = {index:[model[key]] for index,key in enumerate(model)}
			elif isinstance(model,iterables) and all(isinstance(instance,cls) for instance in model):
				model = {index:[instance] for index,instance in enumerate(model)}
			elif isinstance(model,dicts) and all(not isinstance(model[key],cls) and all(isinstance(instance,cls) for instance in model[key]) for key in model):
				model = {index:[instance for instance in model[key]] for index,key in enumerate(model)}

			else:
				raise NotImplementedError("Incorrect model %r"%(model))

			self.model = type(self.model)(model)

		return

	def get(self,index=None):
		'''
		Get model of class
		Args:
			index (object): index of model of class
		Returns:
			model (object): model of class
		'''

		if index is not None:

			index = len(self) + index if index < 0 else index

			model = self.model.get(index)

		else:

			model = self.model

		return model

	def clear(self):
		'''
		Clear model of class
		'''

		self.model = Dictionary()

		return

	def layout(self,configuration=None):
		'''
		Sort models of class
		Args:
			configuration (dict): configuration options for layout
				key (str,callable): group iterable, with signature key(iterable,group=True,sort=True) -> callable(key) -> sortable object i.e) int,float,str,tuple
		'''

		self.set()

		model = self.get()

		configuration = self.configuration if configuration is None else configuration

		model = {index: model for index,model in enumerate(model for index in self.model for model in self.model[index])}
		options = {attr: configuration.get(attr,default) for attr,default in dict(key=None,options=None).items()} if configuration is not None else {}

		model = {index: [model[i] for i in group] for index,group in enumerate(groupby(model,**options))}

		self.set(model)

		return

	def __call__(self,parameters=None,state=None,**kwargs):
		'''
		Class function
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): class keyword arguments
		Returns
			out (array): Return of function
		'''

		parameters = self.parameters(parameters) if parameters is not None else self.parameters(self.parameters())
		state = state if state is not None else state

		return self.func(parameters=parameters,state=state,**kwargs)

	def info(self,display=None,ignore=None,verbose=None,**kwargs):
		'''
		Log class information
		Args:
			display (str,iterable[str]): Show attributes
			ignore (str,iterable[str]): Do not show attributes
			verbose (bool,int,str): Verbosity of message
			kwargs (dict): Additional logging keyword arguments
		'''

		if not verbose and not self.verbose:
			return

		msg = []

		options = dict(
			align=kwargs.get('align','<'),
			space=kwargs.get('space',1),
			width=kwargs.get('width',2)
			)

		precision = kwargs.get('precision',8)

		parse = lambda obj: str(obj.round(precision)) if isinstance(obj,arrays) else str(obj)

		display = None if display is None else [display] if isinstance(display,str) else display
		ignore = None if ignore is None else [ignore] if isinstance(ignore,str) else ignore

		for attr in [None,'string','key','seed','instance','instances','backend','architecture','timestamp','N','M','measure','model','data']:

			obj = attr
			if (display is not None and obj not in display) or (ignore is not None and obj in ignore):
				continue

			if attr is None:
				attr = 'cls'
				substring = str(self)
			elif attr in ['model']:
				substring = getattrs(self,attr,delimiter=delim,default=None)
			elif attr in ['measure']:
				substring = getattrs(self,attr,delimiter=delim,default=None).info(verbose=verbose)
				continue
			else:
				substring = getattrs(self,attr,delimiter=delim,default=None)

			if isinstance(substring,arrays):
				string = '%s:\n%s'%(attr,parse(substring))
			else:
				string = '%s: %s'%(attr,parse(substring))

			msg.append(string)

		msg = [i if isinstance(i,str) else str(i) for i in msg]

		msg = '\n'.join(msg)

		self.log(msg,verbose=verbose)

		return

	def dump(self,path=None):
		'''
		Save class data
		Args:
			path (str): Path to dump class data
		'''

		# Set path
		path = join(self.path,root=self.cwd) if path is None else path

		# Set data
		data = {}

		# Set options
		options = dict(lock=self.lock,backup=self.backup)

		# Set key
		key = self.key if self.key is not None else None

		# Set callback
		callback = self.callback if self.callback is not None else None

		# Set do
		do = (path is not None) and (callback is not None) and ((not exists(path)) or (self.load(path) is None))

		# Dump data
		if do:
			parameters = self.parameters()
			state = self.state()
			model = self
			data = data
			kwargs = {}

			status = callback(parameters=parameters,state=state,model=model,data=data,**kwargs)

			data = {key:data} if key is not None else data

			dump(data,path,**options)

		return

	def load(self,path=None):
		'''
		Load class data
		Args:
			path (str,dict[str,(str,bool)]): Path to load class data, either path or boolean to load
		Returns:
			data (object): Class data
		'''

		# Set path
		path = join(self.path,root=self.cwd) if path is None else path

		# Set data
		data = {}

		# Set options
		options = dict(default=data,lock=self.lock,backup=self.backup)

		# Set key
		key = self.key if self.key is not None else None

		# Set do
		do = (path is not None) and (exists(path))

		# Load data
		data = load(path,**options)

		data = data.get(key) if key is not None else data if data else None

		return data

class Label(Operator):

	N = None
	D = None

	default = 'I'

	def __new__(cls,*args,**kwargs):

		self = super().__new__(cls,*args,**kwargs)

		return self

	def __init__(self,*args,**kwargs):
		return

	def init(self,data=None,state=None,parameters=None,conj=False,**kwargs):
		'''
		Initialize class
		Args:
			data (bool,dict): data of class, or boolean to retain current data attribute or initialize as None
			state (bool,dict,array,Object): state of class
			parameters (array,dict): parameters of class
			conj (bool): conjugate
			kwargs (dict): Additional class keyword arguments
		'''

		super().init(data=data,state=state,parameters=parameters,conj=conj,**kwargs)

		variable = self.variable
		constant = self.constant
		symmetry = self.symmetry

		if self.state is None or self.state() is None:
			hermitian = self.hermitian
			unitary = self.unitary
		elif self.state.ndim == 1:
			hermitian = False
			unitary = True
		elif self.state.ndim == 2:
			hermitian = True
			unitary = False

		self.variable = variable
		self.constant = constant
		self.symmetry = symmetry
		self.hermitian = hermitian
		self.unitary = unitary

		return

	def __call__(self,parameters=None,state=None,**kwargs):
		'''
		Call class
		Args:
			parameters (array): parameters
			state (obj): state
			kwargs (dict): Additional operator keyword arguments
		Returns:
			data (array): data
		'''

		parameters = self.parameters() if parameters is None and self.parameters is not None else parameters if parameters is not None else None
		state = self.state() if state is None and self.state is not None else state if state is not None else self.identity()

		if state is None:
			return self.func(parameters=parameters,state=state,**kwargs)
		else:
			return self.contract(self.func(parameters=parameters,state=state,**kwargs),state=state)

	def grad(self,parameters=None,state=None,**kwargs):
		'''
		Call class gradient
		Args:
			parameters (array): parameters
			state (obj): state
		Returns:
			data (array): data
		'''
		state = self.state() if state is None and self.state is not None else state if state is not None else self.identity()

		if state is None:
			return self.gradient(parameters=parameters,state=state,**kwargs)
		else:
			return self.gradient_contract(self.gradient(parameters=parameters,state=state,**kwargs),self.func(parameters=parameters,state=state,**kwargs),state=state)




class Callback(System):
	def __init__(self,*args,**kwargs):
		'''
		Class for callback
		Args:
			args (tuple): Class arguments
			kwargs (dict): Class keyword arguments
		'''

		super().__init__(*args,**kwargs)

		self.defaults = {
			'iteration':[],
			'parameters':[],'grad':[],'search':[],
			'value':[],'objective':[],
			'alpha':[],'beta':[],

			'iteration.max':[],'iteration.min':[],
			'parameters.relative':[],'parameters.relative.mean':[],
			'variables':[],
			'variables.relative':[],'variables.relative.mean':[],
			'parameters.norm':[],'grad.norm':[],'search.norm':[],
			'objective.ideal.noise':[],'objective.diff.noise':[],'objective.rel.noise':[],
			'objective.ideal.state':[],'objective.diff.state':[],'objective.rel.state':[],
			'objective.ideal.operator':[],'objective.diff.operator':[],'objective.rel.operator':[],
			'hessian':[],'fisher':[],
			'hessian.eigenvalues':[],'fisher.eigenvalues':[],
			'hessian.rank':[],'fisher.rank':[],
			'entropy':[],'purity':[],'similarity':[],'divergence':[],

			'N':[],'D':[],'d':[],'M':[],'T':[],'tau':[],'P':[],
			'space':[],'time':[],'lattice':[],'backend':[],'architecture':[],'timestamp':[],

			'noise.string':[],'noise.ndim':[],'noise.locality':[],'noise.method':[],'noise.scale':[],'noise.tau':[],'noise.initialization':[],
			'noise.parameters':[],

			'state.string':[],'state.ndim':[],'label.string':[],'label.ndim':[],

			'hyperparameters.c1':[],'hyperparameters.c2':[],

		}

		return

	def __call__(self,parameters,track,optimizer,model,metric,func,grad):
		'''
		Callback
		Args:
			parameters (array): parameters
			track (dict): data tracking
			optimizer (Optimizer): optimizer
			model (object): Model instance
			metric (str,callable): metric
			func (callable): Objective function with signature func(parameters)
			grad (callable): Objective gradient with signature grad(parameters)
		Returns:
			status (int): status of callback
		'''

		attributes = optimizer.attributes
		iterations = optimizer.iterations
		hyperparameters = optimizer.hyperparameters

		init = (len(attributes['iteration'])==1) and ((attributes['iteration'][-1]==0) or (attributes['iteration'][-1] != (iterations.stop)))

		done = ((len(attributes['iteration'])>1) and (attributes['iteration'][-1] == (iterations.stop)))

		status = (
			((len(attributes['value']) >= 1) and
			 (attributes['iteration'][-1] <= max(1,
				hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) or
			(
			(absolute(attributes['value'][-1]) >
				(hyperparameters['eps']['value']*hyperparameters['value']['value'])) and
			(log10(absolute(attributes['value'][-1] - attributes['value'][-2])) >
				(log10(absolute(hyperparameters['eps']['value.difference'])))) and
			(norm(attributes['grad'][-1])/(attributes['grad'][-1].size) >
				  (hyperparameters['eps']['grad']*hyperparameters['value']['grad'])) and
			(norm(attributes['grad'][-1] - attributes['grad'][-2])/(attributes['grad'][-2].size) >
				  (hyperparameters['eps']['grad.difference']*norm(attributes['grad'][-2])/(attributes['grad'][-2].size)))
			)
			)


		stop = (
			(
			(hyperparameters['eps'].get('value.increase') is not None) and
			(hyperparameters['eps'].get('value.increase') > 0) and
			((len(attributes['value']) > 1) and
			 (attributes['iteration'][-1] >= max(1,
				hyperparameters['value']['iteration'] if hyperparameters['value'].get('iteration') is not None else 1))) and
			((attributes['value'][-1] > attributes['value'][-2]) and
			(log10(attributes['value'][-1] - attributes['value'][-2]) >
			(log10(hyperparameters['eps']['value.increase']*attributes['value'][-1]))))
			) or
			((iterations.start == iterations.stop))
			)

		none = (iterations.start == 0) and (iterations.stop == 0)

		status = ((status) and (not stop) and (not none))

		other = (
			(len(attributes['iteration']) == 1) or
			(hyperparameters['modulo']['track'] is None) or
			((hyperparameters['modulo']['track'] == -1) and (not status)) or
			((hyperparameters['modulo']['track'] > 0) and (attributes['iteration'][-1]%hyperparameters['modulo']['track'] == 0))
			)

		tracking = ((not status) or done or init or other)

		updates = {
			**{attr: lambda i,attr,track,default: (track[attr][-1]) for attr in ['iteration.max','iteration.min']},
			**{attr: lambda i,attr,track,default: (track[attr][i])
				for attr in [
					'parameters','grad','search']},
			**{attr: lambda i,attr,track,default: (empty(track[attr][-1].shape) if ((i>0) and i<(len(track[attr])-1)) else track[attr][i])
				for attr in [
					'parameters','grad','search',
					'parameters.relative',
					'variables','variables.relative'
					'hessian','fisher',
					'hessian.eigenvalues','fisher.eigenvalues']},
			**{attr: None for attr in [
				'parameters.norm','grad.norm','search.norm',
				]},
			**{attr: lambda i,attr,track,default: (default if ((i>0) and (i<(len(track[attr])-1))) else track[attr][i])
				for attr in [
				'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
				'objective.ideal.state','objective.diff.state','objective.rel.state',
				'objective.ideal.operator','objective.diff.operator','objective.rel.operator',
				'hessian.rank','fisher.rank'
				]
			},
			}

		does = {
			'parameters':True,
			'grad':True,
			'search':True,
			'parameters.relative':True,'parameters.relative.mean':True,
			'variables':True,'variables.relative':True,'variables.relative.mean':True,
			'entropy':True,'purity':True,'similarity':True,'divergence':True
		}

		attrs = relsort(track,attributes)
		size = min(len(track[attr]) for attr in track)
		does = {**{attr: False for attr in attrs},**does,**hyperparameters.get('do',{})}


		if tracking:

			for attr in attrs:

				if ((hyperparameters['length']['track'] is not None) and
					(len(track[attr]) > hyperparameters['length']['track'])
					):
					_value = track[attr].pop(0)


				index = -1 if ((not stop) or other) else -2
				parameters = attributes['parameters'][index]
				state = metric.state()
				kwargs = {}

				if attr in [
					'parameters','grad','search',
					'parameters.relative',
					'variables','variables.relative'
					'hessian','fisher',
					'hessian.eigenvalues','fisher.eigenvalues']:
					default = empty(track[attr][-1].shape) if ((len(track[attr])>0) and isinstance(track[attr][-1],arrays)) else nan
				else:
					default = nan

				do = (not ((status) and (not done) and (not init))) or does[attr]

				value = default

				if attr in attributes:
					value = attributes[attr][index]

				if ((not stop) or other):
					track[attr].append(value)

				if attr in ['iteration.max']:
					value = int(track['iteration'][-1])

				elif attr in ['iteration.min']:
					value = int(track['iteration'][argmin(absolute(array(track['objective'],dtype=model.dtype)))])

				elif attr in ['value']:
					value = absolute(attributes[attr][index])

				elif attr in ['parameters','grad','search'] and (not do):
					value = default

				elif attr in ['parameters','grad','search'] and (do):
					value = attributes[attr][index]

				elif attr in ['parameters.norm','grad.norm','search.norm']:
					value = attr.split(delim)[0]
					value = attributes[value][index]
					value = norm(value)/(value.size)

				elif attr in ['parameters.relative','parameters.relative.mean',
					] and (not do):
					value = default

				elif attr in [
					'parameters.relative','parameters.relative.mean',
					] and (do):
					eps = 1e-20
					if attr in ['parameters.relative']:
						value = parameters
						_value = attributes['parameters'][0]
						value = absolute((value-_value)/(_value+eps))
					elif attr in ['parameters.relative.mean']:
						value = parameters
						_value = attributes['parameters'][0]
						value = norm((value-_value)/(_value+eps))/(value.size)

				elif attr in ['variables.relative','variables.relative.mean',
					] and (not do):
					value = default

				elif attr in [
					'variables.relative','variables.relative.mean',
					] and (do):
					eps = 1e-20
					if attr in ['variables.relative']:
						value = model.parameters(parameters)
						value = array([model.data[i].parameters(j) for j in value for i in model.data if model.data[i].variable])
						_value = model.parameters(attributes['parameters'][0])
						_value = array([model.data[i].parameters(j) for j in _value for i in model.data if model.data[i].variable])
						value = absolute((value-_value)/(_value+eps))
					elif attr in ['variables.relative.mean']:
						value = model.parameters(parameters)
						value = array([model.data[i].parameters(j) for j in value for i in model.data if model.data[i].variable])
						_value = model.parameters(attributes['parameters'][0])
						_value = array([model.data[i].parameters(j) for j in _value for i in model.data if model.data[i].variable])
						value = norm((value-_value)/(_value+eps))/(value.size)

				elif attr in ['objective']:
					value = absolute(metric(model(parameters=parameters,state=state,**kwargs)))

				elif attr in [
					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
					'objective.ideal.state','objective.diff.state','objective.rel.state',
					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and ((status) and (not done)):
					value = default


				elif attr in [
					'objective.ideal.noise','objective.diff.noise','objective.rel.noise',
					'objective.ideal.state','objective.diff.state','objective.rel.state',
					'objective.ideal.operator','objective.diff.operator','objective.rel.operator'] and (not ((status) and (not done))):


					defaults = Dictionary(
						data={i: model.data[i] for i in model.data},
						state=metric.state,
						label=metric.label)

					tmp = Dictionary(
						data={i: model.data[i] for i in model.data if (not model.data[i].unitary)},
						state=metric.state,
						label=metric.label)

					if attr in ['objective.ideal.noise','objective.diff.noise','objective.rel.noise']:
						tmp.update(dict(data={i:True for i in tmp.data},state=tmp.state,label=tmp.label))
					elif attr in ['objective.ideal.state','objective.diff.state','objective.rel.state']:
						tmp.update(dict(data={i:False for i in tmp.data},state=tmp.state,label=tmp.label))
					elif attr in ['objective.ideal.operator','objective.diff.operator','objective.rel.operator']:
						tmp.update(dict(data={i:False for i in tmp.data},state=False,label=tmp.label))

					data = tmp.data
					state = metric.state
					label = metric.label

					state.init(data=tmp.state)
					label.init(state=state)

					model.init(data=data,state=state)

					metric.init(model=model,state=state,label=label)


					if attr in ['objective.ideal.noise','objective.ideal.state','objective.ideal.operator']:
						value = absolute(metric(model(parameters=parameters,state=state,**kwargs)))
					elif attr in ['objective.diff.noise','objective.diff.state','objective.diff.operator']:
						value = absolute((track['objective'][-1] - metric(model(parameters=parameters,state=state,**kwargs))))
					elif attr in ['objective.rel.noise','objective.rel.state','objective.rel.operator']:
						value = absolute((track['objective'][-1] - metric(model(parameters=parameters,state=state,**kwargs)))/(track['objective'][-1]))


					data = defaults.data
					state = state
					label = label

					state.init(data=default.state)
					label.init(state=state)

					model.init(data=data,state=state)

					metric.init(model=model,state=state,label=label)

					state = state()

				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (not do):
					value = default

				elif attr in ['hessian','fisher','hessian.eigenvalues','fisher.eigenvalues','hessian.rank','fisher.rank'] and (do):

					shape = model.shape if ((metric.state is None) or (metric.state.shape is None)) else metric.state.shape
					grad = model.grad_analytical
					wrapper = jit

					if attr in ['hessian','hessian.eigenvalues','hessian.rank']:
						function = hessian(wrapper(lambda parameters,state,**kwargs: metric(model(parameters=parameters,state=state,**kwargs))))
					elif attr in ['fisher','fisher.eigenvalues','fisher.rank']:
						function = fisher(model,grad,shapes=(shape,(*parameters.shape,*shape)),hermitian=metric.state.hermitian,unitary=model.unitary)

					if attr in ['hessian','fisher']:
						value = function(parameters=parameters,state=state,**kwargs)

					elif attr in ['hessian.eigenvalues','fisher.eigenvalues']:
						value = sort(absolute(eig(function(parameters=parameters,state=state,**kwargs),hermitian=True)))[::-1]
						value = value/maximum(value)
					elif attr in ['hessian.rank','fisher.rank']:
						value = sort(absolute(eig(function(parameters=parameters,state=state,**kwargs),hermitian=True)))[::-1]
						value = value/maximum(value)
						value = nonzero(value,eps=1e-12)
						# value = (argmax(absolute(difference(value)/value[:-1]))+1) if value.size > 1 else 1

				elif attr in ['entropy'] and (not do):
					value = default

				elif attr in ['entropy'] and (do):
					shape = model.shape if ((metric.state is None) or (metric.state.shape is None)) else metric.state.shape

					function = entropy(model,shape=shape,hermitian=metric.state.hermitian,unitary=model.unitary)

					value = function(parameters=parameters,state=state,**kwargs)

				elif attr in ['purity'] and (not do):
					value = default

				elif attr in ['purity'] and (do):
					shape = model.shape if ((metric.state is None) or (metric.state.shape is None)) else metric.state.shape

					function = purity(model,shape=shape,hermitian=metric.state.hermitian,unitary=model.unitary)

					value = function(parameters=parameters,state=state,**kwargs)

				elif attr in ['similarity'] and (not do):
					value = default

				elif attr in ['similarity'] and (do):
					shape = model.shape if ((metric.state is None) or (metric.state.shape is None)) else metric.state.shape

					function = similarity(model,metric.label,shape=shape,hermitian=metric.state.hermitian,unitary=model.unitary)

					value = function(parameters=parameters,state=state,**kwargs)

				elif attr in ['divergence'] and (not do):
					value = default

				elif attr in ['divergence'] and (do):
					shape = model.shape if ((metric.state is None) or (metric.state.shape is None)) else metric.state.shape

					function = divergence(model,metric.label,shape=shape,hermitian=metric.state.hermitian,unitary=model.unitary)

					value = function(parameters=parameters,state=state,**kwargs)

				elif attr in [
					'state.string','state.ndim',
					'label.string','label.ndim',
					]:
					value = getattrs(metric,attr,default=default,delimiter=delim)

				elif attr in [
					'noise.string','noise.ndim','noise.locality',
					'noise.method','noise.scale','noise.tau','noise.initialization'
					]:
					for i in model.data:
						if model.data[i].string == delim.join(attr.split(delim)[:1]):
							value = getattrs(model.data[i],delim.join(attr.split(delim)[1:]),default=default,delimiter=delim)
							break

				elif attr in ['noise.parameters']:
					for i in model.data:
						if model.data[i].string == delim.join(attr.split(delim)[:1]):
							value = getattrs(model.data[i],delim.join(attr.split(delim)[1:]),default=default,delimiter=delim)
							value = value(value())
							break

				elif attr in []:
					value = [attr.split(delim)[0],delim.join(attr.split(delim)[1:])]
					value = [getattrs(model,i,default=default,delimiter=delim) for i in value]
					value = value[0]/value[1] if value[1] else value[0]

				elif attr not in attributes and hasattrs(optimizer,attr,delimiter=delim):
					value = getattrs(optimizer,attr,default=default,delimiter=delim)

				elif attr not in attributes and hasattrs(model,attr,delimiter=delim):
					value = getattrs(model,attr,default=default,delimiter=delim)

				track[attr][-1] = value

				if (not does[attr]) and (updates.get(attr) is not None):
					for i in range(len(track[attr])):
						track[attr][i] = updates[attr](i,attr,track,default)


		logging = ((len(attributes['iteration']) == 1) or
			(hyperparameters['modulo']['log'] is None) or
			(attributes['iteration'][-1]%hyperparameters['modulo']['log'] == 0)
			)

		if logging:

			msg = '\n'.join([
				'%d f(x) = %0.4e'%(
					attributes['iteration'][-1],
					# absolute(metric(model(attributes['parameters'][-1],metric.state()))),
					attributes['value'][-1],
				),
				'|x| = %0.4e\t\t|grad(x)| = %0.4e'%(
					norm(attributes['parameters'][-1])/
						 (attributes['parameters'][-1].size),
					norm(attributes['grad'][-1])/
						 (attributes['grad'][-1].size),
				),
				'\t\t'.join([
					'%s = %0.4e'%(attr,attributes[attr][-1])
					for attr in ['alpha','beta']
					if attr in attributes and len(attributes[attr])>0
					]),
				# 'attributes: \t %s'%({attr: attributes[attr][-1].dtype
				# 	if isinstance(attributes[attr][-1],arrays) else type(attributes[attr][-1])
				# 	for attr in attributes}),
				# 'track: \t %s'%({attr: attributes[attr][-1].dtype
				# 	if isinstance(attributes[attr][-1],arrays) else type(attributes[attr][-1])
				# 	for attr in attributes}),
				# 'x\n%s'%(to_string(parameters.round(4))),
				# 'theta\n%s'%(to_string(model.parameters(parameters).round(4))),
				# 'U\n%s\nV\n%s'%(
				# 	to_string((model(parameters=parameters,state=state,**kwargs)).round(4)),
				# 	to_string((metric.label()).round(4))),
				])


			model.log(msg)

		return status



class Callback(System):
	def __init__(self,*args,attributes=None,arguments=None,keywords=None,options=None,**kwargs):
		'''
		Class for callback
		Args:
			attributes (dict): Attributes for callback
			args (tuple): Class arguments
			arguments (dict,iterable): Class positional arguments
			keywords (dict): Class keyword arguments
			options (dict): Class keyword arguments
			kwargs (dict): Class keyword arguments
		'''

		defaults = [
			'objective','infidelity','norm','entanglement','entangling','trace',
			'array','state',
			'sample.array.linear','sample.array.log','sample.state.linear','sample.state.log',
			'sample.array.information','sample.state.information',
			'infidelity.quantum','infidelity.classical','infidelity.pure',
			'norm.quantum','norm.classical','norm.pure',
			'entanglement.quantum','entanglement.classical','entanglement.renyi',
			'entangling.quantum','entangling.classical','entangling.renyi',
			'mutual.quantum','mutual.measure','mutual.classical','mutual.renyi',
			'discord.quantum','discord.classical','discord.renyi',
			'spectrum.quantum','spectrum.classical',
			'rank.quantum','rank.classical',
			'noise.parameters',
			]

		if attributes is None:
			attributes = {attr:attr for attr in defaults}
		elif not isinstance(attributes,dict):
			attributes = {attr:attr for attr in attributes}
		else:
			attributes = {attr:attributes[attr] for attr in attributes if attributes[attr]}

		if arguments is None:
			arguments = {attr: () for attr in attributes}
		elif any(attr in arguments for attr in [*attributes,*defaults]):
			arguments = {attr: arguments.get(attr,()) for attr in attributes}
		else:
			arguments = {attr: (*arguments,) for attr in attributes}

		if keywords is None:
			keywords = {attr: {} for attr in attributes}
		elif any(attr in keywords for attr in [*attributes,*defaults]):
			keywords = {attr: keywords.get(attr,{}) for attr in attributes}
		else:
			keywords = {attr: {**keywords} for attr in attributes}

		if options is None:
			options = {attr: {} for attr in attributes}
		elif all(attr in options for attr in [*attributes,*defaults]):
			options = {attr: options.get(attr,{}) for attr in attributes}
		else:
			options = {attr: {**options} for attr in attributes}

		for attr in attributes:
			if isinstance(attributes[attr],dict) and all(prop in attributes[attr] for prop in ['func','args','kwargs']):
				for prop in attributes[attr]:
					if prop in ['args'] and arguments.get(attr):
						arguments[attr] = [*arguments[attr],*attributes[attr][prop]]
					elif prop in ['kwargs'] and keywords.get(attr):
						keywords[attr] = {**keywords[attr],**attributes[attr][prop]}
				prop = 'func'
				if prop in attributes[attr]:
					attributes[attr] = attributes[attr][prop]

		setter(kwargs,dict(attributes=attributes,arguments=arguments,keywords=keywords,options=options),delimiter=delim,default=False)

		super().__init__(*args,**kwargs)

		return

	def __call__(self,parameters=None,state=None,model=None,data=None,optimizer=None,**kwargs):
		'''
		Callback
		Args:
			parameters (array): parameters
			state (array): state
			model (object): Model instance
			data (dict): data
			optimizer (Optimizer): optimizer
			kwargs (dict): additional keyword arguments for callback
		Returns:
			status (int): status of callback
		'''

		attributes = {attr:self.attributes[attr]
			for attr in self.attributes
			if hasattrs(model,self.attributes[attr],delimiter=delim) or hasattrs(optimizer,self.attributes[attr],delimiter=delim) or attr in ['noise.parameters']
			}

		status = True

		logging = True


		attr = list(attributes)[0] if attributes else None
		arguments = self.arguments.get(attr,())
		keywords = self.keywords.get(attr,{})

		options = {
			**{key: model.options[key] for key in model.options}
			} if model.options is not None else {}
		_options = {
			**options,
			**{key: self.options.get(attr,{}).get(key) for key in options if key in self.options.get(attr,{})},
			**{key: kwargs.get(key) for key in kwargs if key in options},
			}

		obj = model(parameters=parameters,state=state,options=options)

		_obj = model(parameters=parameters,state=state,options=_options)

		for attr in attributes:

			arguments = self.arguments.get(attr,())
			keywords = self.keywords.get(attr,{})

			key = attr

			if attr in [
				'objective','infidelity','norm','entanglement','entangling','trace',
				'array','state',
				'sample.array.linear','sample.array.log','sample.state.linear','sample.state.log',
				'sample.array.information','sample.state.information',
				'infidelity.quantum','infidelity.classical','infidelity.pure',
				'norm.quantum','norm.classical','norm.pure',
				'entanglement.quantum','entanglement.classical','entanglement.renyi',
				'entangling.quantum','entangling.classical','entangling.renyi',
				'mutual.quantum','mutual.measure','mutual.classical','mutual.renyi',
				'discord.quantum','discord.classical','discord.renyi',
				'spectrum.quantum','spectrum.classical',
				'rank.quantum','rank.classical',
				]:

				if attr in [
					'objective','infidelity',
					'infidelity.quantum','infidelity.classical','infidelity.pure',
					]:
					value = getattrs(model,attributes[attr],delimiter=delim)(
						parameters=parameters,
						state=obj,
						other=_obj,
						**keywords)
				elif attr in [
					'norm','entanglement','entangling','trace',
					'array','state',
					'norm.quantum','norm.classical','norm.pure',
					'entanglement.quantum','entanglement.classical','entanglement.renyi',
					'entangling.quantum','entangling.classical','entangling.renyi',
					'mutual.quantum','mutual.measure','mutual.classical','mutual.renyi',
					'discord.quantum','discord.classical','discord.renyi',
					'spectrum.quantum','spectrum.classical',
					'rank.quantum','rank.classical',
					]:
					value = getattrs(model,attributes[attr],delimiter=delim)(
						parameters=parameters,
						state=obj,
						**keywords)

				elif attr in ['sample.array.linear','sample.array.log','sample.state.linear','sample.state.log']:

					key = '{attr}.{i}'

					value = getattrs(model,attributes[attr],delimiter=delim)(
						parameters=parameters,
						state=obj,
						**keywords)

					if isinstance(value,dict):
						key = [key.format(attr=attr,i=i) for i in value]
						value = [value[i] for i in value]
					else:
						key = attr
						value = value

				elif attr in ['sample.array.information','sample.state.information']:

					key = '{attr}{i}'

					value = getattrs(model,attributes[attr],delimiter=delim)(
						parameters=parameters,
						state=obj,
						**keywords)

					if isinstance(value,dict):
						key = [key.format(attr=attr,i=i) for i in value]
						value = [value[i] for i in value]
					else:
						key = attr
						value = value

			elif attr in ['noise.parameters']:

				if hasattr(model,'model'):
					value = getattr(model,'model')
					value = [model for index in value for model in value[index]]
				else:
					value = model.data
					value = [value[index] for index in value]

				value = [model.parameters() for model in value if not model.unitary and not model.hermitian]

				value = value[0] if value else None

			elif hasattrs(model,attributes[attr],delimiter=delim):

				value = getattrs(model,attributes[attr],delimiter=delim)

				if callable(value):
					value = value(parameters=parameters,state=obj,**kwargs)

			elif hasattrs(optimizer,attributes[attr],delimiter=delim):

				value = getattrs(optimizer,attributes[attr],delimiter=delim)

				if callable(value):
					value = value(parameters=parameters,state=obj,**kwargs)

			else:

				value = None

			if isinstance(key,str):
				key = [key]
				value = [value]

			for key,value in zip(key,value):
				if not data.get(key) and not isinstance(data.get(key),list):
					data[key] = []
				data[key].append(value)

		if logging:

			msg = '\n'.join([
				'%r f(x) = %s'%(optimizer.iteration if optimizer is not None else None,'%0.4e'%(data['objective'][-1]) if data.get('objective') else None),
				'|x| = %s'%('%0.4e'%(norm(parameters)) if parameters is not None else None),
				])


			model.log(msg)

		return status