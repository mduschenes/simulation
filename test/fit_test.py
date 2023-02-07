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

from src.utils import array,ones,zeros,rand,logspace,gradient,rao,sort,norm,allclose,log10,exp10,abs,inf
from src.fit import fit

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
	# log = file if hasattr(file,'write') else sys.stderr
	# traceback.print_stack(file=log)
	# log.write(warnings.formatwarning(message, category, filename, lineno, line))
	return
warnings.showwarning = warn_with_traceback



def test_err(path=None,tol=None):

	scale = 3
	scale = 1
	def model(coef,x):
		# y = scale*log10(coef[0] + coef[1]*log10(x))
		y = scale*(coef[0] + coef[1]*(x))
		return y

	n = 100
	d = 2
	sigma = None
	key = {'x':19354,'coef':[1923,2424],'yerr':25215}


	x = sort(rand((n,),bounds=[1e-6,3],key=key['x']))
	coef = array([rand(bounds=[0,7],key=key['coef'][0]),rand(bounds=[-1,0],key=key['coef'][1])]).ravel()
	y = model(coef,x) 
	xerr = None
	yerr = sigma*rand(n,bounds=[-1,1],key=key['yerr']) if (sigma is not None and sigma>0) else None
	yerr = sigma*ones(n) if (sigma is not None and sigma>0) else None

	y = y#+yerr if yerr is not None else y
	yerr = abs(yerr) if yerr is not None else yerr

	x_ = x
	y_ = y
	coef_ = coef
	_rao_ = rao(model,label=y,error=yerr)(coef_,x)

	def func(coef,x):
		y = coef[0] + coef[1]*x
		return y


	_n = n*10
	_x = logspace(int(log10(x.min()))-2,int(log10(x.max())),_n)
	_n = n
	_x = x
	_y = zeros(_n)

	coef = rand(coef.shape)
	kwargs = {
		'maxfev':200000,
		'absolute_sigma':True,
	}
	preprocess = lambda x,y,coef: (log10(x) if x is not None else None,exp10(y/scale) if y is not None else None,coef if coef is not None else None)
	postprocess = lambda x,y,coef: (exp10(x) if x is not None else None,scale*log10(y) if y is not None else None,coef if coef is not None else None)

	preprocess = lambda x,y,coef: ((x) if x is not None else None,(y/scale) if y is not None else None,coef if coef is not None else None)
	postprocess = lambda x,y,coef: ((x) if x is not None else None,scale*(y) if y is not None else None,coef if coef is not None else None)

	_func,_y,_coef,_yerr,_coeferr,_r,_other = fit(
		x,y,
		_x=_x,_y=_y,
		func=func,coef=coef,
		yerr=yerr,
		xerr=xerr,
		preprocess=preprocess,postprocess=postprocess,
		kwargs=kwargs)



	rao_ = rao(_func,label=y,error=yerr)(_coef,x)
	_rao = _coeferr

	print(coef_)
	print(coef)
	print(_coef)
	print()
	print(_rao_)
	print(rao_)
	print(_rao)

	fig,ax = plt.subplots()
	ax.plot(x_,y_,label='orig',marker='o',linestyle='')
	ax.plot(_x,_y,label='pred',marker='*',linestyle='-')
	ax.plot(_x,(model(coef_,(_x))),label='func')
	ax.plot(_x,(_func(coef_,(_x))),label='$\_$func',linestyle='--')
	ax.legend();
	fig.savefig('plot.pdf')

	tol = 1e-7
	
	s,a,b = 'coef',_coef,coef_
	eps = norm(a-b)/norm(b)
	assert eps < tol,'%s: %r - %r = %0.3e < %0.1e'%(s,a,b,eps,tol)

	s,a,b = 'covar',(_rao),(rao_)
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