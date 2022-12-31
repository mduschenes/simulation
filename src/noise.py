#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from copy import deepcopy
from functools import partial
import time
from time import time as timer

# Logging
import logging
logger = logging.getLogger(__name__)

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,rand
from src.utils import tensorprod,sqrt
from src.system import Object


class Noise(Object):
	def __init__(self,data,shape,size=None,dims=None,samples=None,system=None,**kwargs):
		'''
		Initialize data of attribute based on shape, with highest priority of arguments of: args,data,system,kwargs
		Args:
			data (dict,str,array,Noise): Data corresponding to noise
			shape (int,iterable[int]): Shape of each data
			size (int,iterable[int]): Number of data
			dims (iterable[int]): Dimensions of N, D-dimensional sites [N,D]
			samples (bool,array): Weight samples (create random weights, or use samples weights)
			system (dict,System): System attributes (dtype,format,device,backend,architecture,seed,key,timestamp,cwd,path,conf,logging,cleanup,verbose)			
			kwargs (dict): Additional system keyword arguments
		'''

		super().__init__(data,shape,size=size,dims=dims,samples=samples,system=system,**kwargs)

		return

	def __setup__(self,**kwargs):
		'''
		Setup attribute
		Args:
			kwargs (dict): Additional keyword arguments
		'''

		# Basis
		operators = {
			attr: self.basis[attr].astype(self.dtype)
			for attr in self.basis
			}

		assert (self.scale >= 0) and (self.scale <= 1), "Noise scale %r not in [0,1]"%(self.scale)

		if self.string is None:
			data = [self.basis['I']]
		elif self.string in ['phase']:
			data = [sqrt(1-self.scale)*self.basis['I'],
					sqrt(self.scale)*self.basis['Z']]
		elif self.string in ['amplitude']:
			data = [self.basis['00'] + sqrt(1-self.scale)*self.basis['11'],
					sqrt(self.scale)*self.basis['01']]
		elif self.string in ['depolarize']:
			data = [sqrt(1-self.scale)*self.basis['I'],
					sqrt(self.scale/3)*self.basis['X'],
					sqrt(self.scale/3)*self.basis['Y'],
					sqrt(self.scale/3)*self.basis['Z']]
		else:
			data = [self.basis['I']]

		data = array([
			tensorprod(i)
			for i in itertools.product(data,repeat=self.N)
			])
			
		self.data = data

		return