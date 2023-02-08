#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy,warnings

import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.io import load,dump,join,split,edit
from src.utils import array,ones,zeros,rand,logspace,gradient,sort,norm,allclose,log10,exp10,abs,inf
from src.fit import fit,cov

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
	# log = file if hasattr(file,'write') else sys.stderr
	# traceback.print_stack(file=log)
	# log.write(warnings.formatwarning(message, category, filename, lineno, line))
	return
warnings.showwarning = warn_with_traceback



def test_err(path=None,tol=None):

	scale = 3
	def model(parameters,x):
		y = parameters[0] + parameters[1]*x
		# y = scale*log10(parameters[0] + parameters[1]*log10(x))
		# y = scale*(parameters[0] + parameters[1]*log10(x))
		return y

	n = 100
	d = 2
	sigma = 2e-1
	key = {'x':25643632235,'parameters':[23512313,123],'parameters_':[924254,1047324],'yerr':1313}
	shapes = ((n,),(n,),(n,))

	x = sort(rand((n,),bounds=[0,1],key=key['x']))
	parameters = array([rand(bounds=[0,1],key=key['parameters_'][0]),rand(bounds=[0,1],key=key['parameters_'][1])])[:d].ravel()
	y = model(parameters,x) 
	xerr = None
	yerr = sigma*rand(n,bounds=[-1,1],key=key['yerr']) if (sigma is not None and sigma>0) else None
	yerr = sigma*ones(n) if (sigma is not None and sigma>0) else None

	y = y#+yerr if yerr is not None else y
	yerr = abs(yerr) if yerr is not None else yerr

	x_ = x
	y_ = y
	xerr_ = xerr
	yerr_ = yerr
	parameters_ = parameters
	_cov_ = cov(model,shapes=shapes,label=y_,weights=yerr_)(parameters_,x_)

	def func(parameters,x):
		y = parameters[0] + parameters[1]*x
		return y


	_n = n*10
	_x = logspace(int(log10(x.min()))-2,int(log10(x.max())),_n)
	_n = n
	_x = x
	_y = zeros(_n)

	parameters = array([rand(bounds=[0,1],key=key['parameters'][0]),rand(bounds=[0,1],key=key['parameters'][1])])[:d].ravel()
	kwargs = {
		'process':False,
		'standardize':False,
	}
	
	preprocess = None
	postprocess = None

	preprocess = lambda x,y,parameters: (log10(x) if x is not None else None,exp10(y/scale) if y is not None else None,parameters if parameters is not None else None)
	postprocess = lambda x,y,parameters: (exp10(x) if x is not None else None,scale*log10(y) if y is not None else None,parameters if parameters is not None else None)
	
	# preprocess = lambda x,y,parameters: (log10(x) if x is not None else None,(y/scale) if y is not None else None,parameters if parameters is not None else None)
	# postprocess = lambda x,y,parameters: (exp10(x) if x is not None else None,scale*(y) if y is not None else None,parameters if parameters is not None else None)
	
	_func,_y,_parameters,_yerr,_parameterserr,_other = fit(
		x,y,
		_x=_x,_y=_y,
		func=func,parameters=parameters,
		yerr=yerr,
		xerr=xerr,
		preprocess=preprocess,postprocess=postprocess,
		kwargs=kwargs)


	cov_ = cov(_func,shapes=shapes,label=y_,weights=yerr)(_parameters,x)
	_cov = _parameterserr

	print(sigma)
	print('----')
	print(parameters_)
	print(_parameters)
	print()
	print(_cov_)
	print(cov_)
	print(_cov)

	fig,ax = plt.subplots()
	ax.plot(x_,y_,label='orig',marker='o',linestyle='')
	ax.plot(_x,_y,label='pred',marker='*',linestyle='-')
	ax.plot(_x,(model(parameters_,(_x))),label='func')
	ax.plot(_x,(_func(parameters_,(_x))),label='$\_$func',linestyle='--')
	ax.legend();
	fig.savefig('plot.pdf')

	tol = 1e-7
	
	s,a,b = 'parameters',_parameters,parameters_
	eps = norm(a-b)/norm(b)
	assert eps < tol,'%s: %r - %r = %0.3e < %0.1e'%(s,a,b,eps,tol)

	s,a,b = 'covar',(_cov),(cov_)
	eps = norm(a-b)/norm(b)
	assert eps < tol,'%s: %r - %r = %0.3e < %0.1e'%(s,a,b,eps,tol)

	return


if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 
	# test_objective(path,tol)
	# test_optimizer(path,tol)
	# test_getter(path,tol)
	# test_setter(path,tol)
	# test_scinotation(path,tol)
	# test_gradient(path,tol)
	test_err(path,tol)