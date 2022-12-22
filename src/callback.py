#!/usr/bin/env python

# Import python modules
import os,sys
from copy import deepcopy as deepcopy

# Logging
import logging

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,abs,argmin,argmax,min,max,eig,norm,gradient,nan,inf
from src.io import load
from src.dictionary import getter

from src.noise import noiseize
from src.states import stateize


class Callback(object):

	def __init__(self,model,func,grad=None,funcs=None,grads=None,hyperparameters):
		''' 
		Setup callback and logging
		Args:
			model (object): Model instance			
			func (callable): Objective function with signature func(parameters)
			grad (callable): Objective gradient with signature func(parameters)
			funcs (iterable[callable]): Iterable of functions to sum
			grads (iterable[callable]): Iterable of gradients to sum
			hyperparameters(dict): Callback hyperparameters
		'''	
		if funcs is None:
			funcs = [func]
		if grad is None:
			grad = gradient(func)
		if grads is None:
			grads = [grad]


		self.model = model

		self.func = func
		self.grad = grad
		self.funcs = funcs
		self.grads = grads

		self.hyperparameters = hyperparameters

		return

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

		start = (len(attributes['iteration'])==1) and (attributes['iteration'][-1]<hyperparameters['iterations'])
		
		done = (len(attributes['iteration'])>0) and (attributes['iteration'][-1]==hyperparameters['iterations'])
		
		status = (
			(abs(attributes['value'][-1]) > 
				(hyperparameters['eps']['value']*hyperparameters['value']['value'])) and
			((len(attributes['value'])==1) or 
			 ((len(attributes['value'])>1) and 
			 (abs(attributes['value'][-1] - attributes['value'][-2]) > 
				(hyperparameters['eps']['difference']*attributes['value'][-2])))) and
			((len(attributes['value'])==1) or 			
			 ((len(attributes['grad'])>1) and
			(norm(attributes['grad'][-1] - attributes['grad'][-2])/attributes['grad'][-2].size > 
				  (hyperparameters['eps']['grad']*norm(attributes['grad'][-2])/attributes['grad'][-2].size))))
			)

		default = nan

		if (((not status) or done or start) or 
			(len(attributes['iteration']) == 0) or 
			(hyperparameters['modulo']['track'] is None) or 
			(attributes['iteration'][-1]%hyperparameters['modulo']['track'] == 0)
			):	

			for attr in model.__dict__
				if ((not callable(getattr(model,attr))) and
					(getattr(model,attr) is None or isinstance(getattr(model,attr),scalars))):
					_attr = attr
					_value = getattr(model,attr)
					track[_attr].append(_value)

			for attr in track:
				if attr not in ['iteration','parameters','value','grad','search','alpha','beta','objective','hessian','fisher']:
					_attr = attr
					_value = getter(model.hyperparameters,attr.split(delim)) 
					track[_attr].append(_value)


			# TODO: Ensure all attributes are assigned value, either _value or default for all iterations
			for attr in ['iteration','parameters','value','grad','search','alpha','beta','objective','hessian','fisher']:

				if ((hyperparameters['length']['track'] is not None) and 
					(len(track[attr]) > hyperparameters['length']['track'])
					):
					_value = track[attr].pop(0)

				if attr in ['iteration','value','grad','search','alpha','beta'] and attr in attributes:
					_attr = attr
					if _attr in track:
						_value = attributes[attr][-1]
						track[_attr].append(_value)

				value = track[attr][-1]

				if attr in ['iteration']:

					if ((not status) or done):
						_attr = '%s.max'%(attr)
						if _attr in track:
							_value = value
							track[_attr].append(_value)

						_attr = '%s.min'%(attr)
						if _attr in track:
							_value = track[attr][argmin(array(track['objective']))]
							track[_attr].append(_value)
					else:

				elif attr in ['parameters'] and ((not status) or done or start):
		
					_attr = attr
					if _attr in track:
						_value = value
						track[_attr].append(_value)

					_attr = 'features'
					if _attr in track:

						layer = 'features'
						attrs = model.attributes
						indices = tuple([(
							slice(
							min(attrs['index'][layer][parameter][group][axis].start
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]),
							max(attrs['index'][layer][parameter][group][axis].stop
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]),
							min(attrs['index'][layer][parameter][group][axis].step
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]))
							if all(isinstance(attrs['index'][layer][parameter][group][axis],slice)
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]) else
							list(set(i 
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter] 
								for i in attrs['index'][layer][parameter][group][axis]))
							)
								for axis in range(min(len(attrs['index'][layer][parameter][group]) 
												for parameter in attrs['index'][layer] 
												for group in attrs['index'][layer][parameter]))
							])

						_value = model.__layers__(value,layer)[indices]
						track[_attr].append(_value)


					_attr = '%s.relative'%(attr)
					if _attr in track:

						layer = 'features'
						attrs = model.attributes
						indices = tuple([(
							slice(
							min(attrs['index'][layer][parameter][group][axis].start
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]),
							max(attrs['index'][layer][parameter][group][axis].stop
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]),
							min(attrs['index'][layer][parameter][group][axis].step
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]))
							if all(isinstance(attrs['index'][layer][parameter][group][axis],slice)
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]) else
							list(set(i 
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter] 
								for i in attrs['index'][layer][parameter][group][axis]))
							)
								for axis in range(min(len(attrs['index'][layer][parameter][group]) 
												for parameter in attrs['index'][layer] 
												for group in attrs['index'][layer][parameter]))
							])

						_value = abs((model.__layers__(parameters,layer)[indices] - 
							model.__layers__(track[attr][0],layer)[indices] + 1e-20)/(
							model.__layers__(track[attr][0],layer)[indices] + 1e-20))

						track[_attr].append(_value)

					_attr = '%s.relative.mean'%(attr)
					if _attr in track:

						layer = 'features'
						attrs = model.attributes
						indices = tuple([(
							slice(
							min(attrs['index'][layer][parameter][group][axis].start
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]),
							max(attrs['index'][layer][parameter][group][axis].stop
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]),
							min(attrs['index'][layer][parameter][group][axis].step
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]))
							if all(isinstance(attrs['index'][layer][parameter][group][axis],slice)
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter]) else
							list(set(i 
								for parameter in attrs['index'][layer] 
								for group in attrs['index'][layer][parameter] 
								for i in attrs['index'][layer][parameter][group][axis]))
							)
								for axis in range(min(len(attrs['index'][layer][parameter][group]) 
												for parameter in attrs['index'][layer] 
												for group in attrs['index'][layer][parameter]))
							])

						_value = abs((model.__layers__(parameters,layer)[indices] - 
							model.__layers__(track[attr][0],layer)[indices] + 1e-20)/(
							model.__layers__(track[attr][0],layer)[indices] + 1e-20)).mean(-1)

						track[_attr].append(_value)



				elif attr in ['objective']:

					_attr = attr
					if _attr in track:
						_value = getattr(model,'__%s__'%(attr))(parameters)
						track[_attr].append(_value)

					if model.state is None:

						data = None
						shape = model.dims
						hyperparams = deepcopy(model.hyperparameters['state'])
						hyperparams['scale'] = 1 if hyperparams.get('scale') is None else hyperparams.get('scale')
						size = model.N
						samples = True
						seed = model.seed		
						dtype = model.dtype
						cls = model

						state = stateize(data,shape,hyperparams,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)

					else:
						state = model.state

					if model.noise is None:

						data = None
						shape = model.dims
						hyperparams = deepcopy(model.hyperparameters['noise'])
						hyperparams['scale'] = 1 if hyperparams.get('scale') is None else hyperparams.get('scale')
						size = model.N
						samples = None
						seed = model.seed		
						cls = model
						dtype = model.dtype

						noise = noiseize(data,shape,hyperparams,size=size,samples=samples,seed=seed,cls=cls,dtype=dtype)

					else:
						noise = model.noise

					model.__functions__(state=state,noise=noise,label=True,metric='infidelity.norm')

					_attr = '%s.ideal.noise'%(attr)
					if _attr in track:
						_value = model.__objective__(parameters)
						track[_attr].append(_value)

					_attr = '%s.diff.noise'%(attr)
					if _attr in track:
						_value = abs(track[attr][-1] - model.__objective__(parameters))
						track[_attr].append(_value)

					_attr = '%s.rel.noise'%(attr)
					if _attr in track:
						_value = abs((track[attr][-1] - model.__objective__(parameters))/track[attr][-1])
						track[_attr].append(_value)


					model.__functions__(state=state,noise=False,label=True,metric='infidelity.norm')

					_attr = '%s.ideal.state'%(attr)
					if _attr in track:
						_value = model.__objective__(parameters)
						track[_attr].append(_value)

					_attr = '%s.diff.state'%(attr)
					if _attr in track:
						_value = abs(track[attr][-1] - model.__objective__(parameters))
						track[_attr].append(_value)

					_attr = '%s.rel.state'%(attr)
					if _attr in track:
						_value = abs((track[attr][-1] - model.__objective__(parameters))/track[attr][-1])
						track[_attr].append(_value)


					model.__functions__(state=False,noise=False,label=True,metric='infidelity.abs')

					_attr = '%s.ideal.operator'%(attr)
					if _attr in track:
						_value = model.__objective__(parameters)
						track[_attr].append(_value)

					_attr = '%s.diff.operator'%(attr)
					if _attr in track:
						_value = abs(track[attr][-1] - model.__objective__(parameters))
						track[_attr].append(_value)

					_attr = '%s.rel.operator'%(attr)
					if _attr in track:
						_value = abs((track[attr][-1] - model.__objective__(parameters))/track[attr][-1])
						track[_attr].append(_value)
				
				
				elif attr in ['hessian','fisher'] and ((not status) or done):

					_attr = attr
					if _attr in track:
						_value = getattr(model,'__%s__'%(attr))(parameters)
						track[_attr].append(_value)


					_attr = '%s.eigenvalues'%(attr)
					if _attr in track:
						_value = sort(abs(eig(getattr(model,'__%s__'%(attr))(parameters),compute_v=False,hermitian=True)))[::-1]
						_value = _value/max(1,maximum(_value))
						track[_attr].append(_value)

					_attr = '%s.rank'%(attr)
					if _attr in track:
						_value = sort(abs(eig(getattr(model,'__%s__'%(attr))(parameters),compute_v=False,hermitian=True)))[::-1]
						_value = argmax(abs(difference(_value)/_value[:-1]))+1						
						track[_attr].append(_value)

				else:
					track[attr].append(default)

		if ((len(attributes['iteration']) == 0) or 
			(hyperparameters['modulo']['log'] is None) or 
			(attributes['iteration'][-1]%hyperparameters['modulo']['log'] == 0)
			):

			msg = '\n'.join([
				'%d f(x) = %0.4e'%(
					attributes['iteration'][-1],
					track['objective'][-1],
				),
				'|x| = %0.4e\t\t|grad(x)| = %0.4e'%(
					norm(attributes['parameters'][-1])/
						 max(1,attributes['parameters'][-1].size),
					norm(attributes['grad'][-1])/
						 max(1,attributes['grad'][-1].size),
				),
				'\t\t'.join([
					'%s = %0.4e'%(attr,attributes[attr][-1])
					for attr in ['alpha','beta']
					if attr in attributes and len(attributes[attr])>0
					]),
				# 'x\n%s'%(to_string(parameters.round(4))),
				'U\n%s\nV\n%s\n'%(
				to_string(abs(model(parameters)).round(4)),
				to_string(abs(model.labels).round(4))),
				# 'U: %0.4e\tV: %0.4e\n'%(
				# 	trace(model(parameters)).real,
				# 	trace(model.labels).real
				# 	),				
				])


			model.log(msg)


			# print(parameters.reshape(-1,model.M))
			# print(model.__layers__(parameters,'variables').T.reshape(model.M,-1))



		return status