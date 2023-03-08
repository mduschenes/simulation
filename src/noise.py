#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from copy import deepcopy
from functools import partial
import time
from time import time as timer

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,rand,eye
from src.utils import einsum,tensorprod,sqrt,allclose,exp
from src.system import Object


class Noise(Object):
	def __init__(self,data,shape,size=None,ndim=None,dims=None,system=None,**kwargs):
		'''
		Initialize data of attribute based on shape, with highest priority of arguments of: kwargs,args,data,system
		Args:
			data (dict,str,array,Noise): Data corresponding to noise
			shape (int,iterable[int]): Shape of each data
			size (int,iterable[int]): Number of data
			ndim (int): Number of dimensions of data
			dims (iterable[int]): Dimensions of N, D-dimensional sites [N,D]
			system (dict,System): System attributes (dtype,format,device,backend,architecture,unit,seed,key,timestamp,cwd,path,conf,logger,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		super().__init__(data,shape,size=size,ndim=ndim,dims=dims,system=system,**kwargs)

		return

	def __setup__(self,**kwargs):
		'''
		Setup attribute
		Args:
			kwargs (dict): Additional keyword arguments
		'''

		
		# Size of data
		size = None
		self.size = size
		self.length = len(self.size) if self.size is not None else None

		# Set basis
		basis = {
				'I': array([[1,0],[0,1]]),
				'X': array([[0,1],[1,0]]),
				'Y': array([[0,-1j],[1j,0]]),
				'Z': array([[1,0],[0,-1]]),
				'00':array([[1,0],[0,0]]),
				'01':array([[0,1],[0,0]]),
				'10':array([[0,0],[1,0]]),
				'11':array([[0,0],[0,1]]),
			}
		basis = {string: basis[string].astype(self.dtype) for string in basis}

		if (self.parameters is not None):
			if (self.initialization in ['time']):
				if (self.cls.get('tau') is not None):
					self.scale = (1 - exp(-self.cls['tau']/self.parameters))/2

		assert (self.scale >= 0) and (self.scale <= 1), "Noise scale %r not in [0,1]"%(self.scale)

		if self.scale > 0:
			if self.string is None:
				data = [basis['I']]
			elif self.string in ['phase']:
				data = [sqrt(1-self.scale)*basis['I'],
						sqrt(self.scale)*basis['Z']]
			elif self.string in ['amplitude']:
				data = [basis['00'] + sqrt(1-self.scale)*basis['11'],
						sqrt(self.scale)*basis['01']]
			elif self.string in ['depolarize']:
				data = [sqrt(1-self.scale)*basis['I'],
						sqrt(self.scale/3)*basis['X'],
						sqrt(self.scale/3)*basis['Y'],
						sqrt(self.scale/3)*basis['Z']]
			else:
				data = [basis['I']]
		else:
			data = [basis['I']]
	
		data = array([
			tensorprod(i)
			for i in itertools.product(data,repeat=self.N)
			],dtype=self.dtype)
			
		# Assert data is normalized
		if data.ndim == 3:
			normalization = einsum('...uji,...ujk->...ik',data.conj(),data)
		else:
			normalization = einsum('...uji,...ujk->...ik',data.conj(),data)

		assert allclose(eye(self.n),normalization), "Incorrect normalization data%r: %r"%(data.shape,normalization)

		self.data = data
		self.shape = self.data.shape if self.data is not None else None
		self.ndim = self.data.ndim if self.data is not None else None

		return