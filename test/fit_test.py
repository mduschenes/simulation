#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy,warnings

import jax
import jax.numpy as np
import numpy as onp

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

	def model(coef,x):
		y = 2*log10(coef[0] + coef[1]*log10(x))
		return y

	n = 100
	d = 2
	sigma = 1e-3
	key = 12345


	x = sort(rand((n,),bounds=[1e-6,1],key=key))
	coef = array([rand(bounds=[0,4]),rand(bounds=[-2,0])]).ravel()
	y = model(coef,x) 
	xerr = None
	yerr = sigma*rand(n,bounds=[-1,1]) if (sigma is not None and sigma>0) else None
	yerr = sigma*ones(n) if (sigma is not None and sigma>0) else None

	y = y#+yerr if yerr is not None else y
	yerr = abs(yerr) if yerr is not None else yerr

	x_ = x
	y_ = y
	coef_ = coef

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
	preprocess = lambda x,y,coef: (log10(x) if x is not None else None,exp10(y/2) if y is not None else None,coef if coef is not None else None)
	postprocess = lambda x,y,coef: (exp10(x) if x is not None else None,2*log10(y) if y is not None else None,coef if coef is not None else None)
	_func,_y,_coef,_yerr,_coeferr,_r = fit(
		x,y,
		_x=_x,_y=_y,
		func=func,coef=coef,
		yerr=yerr,
		xerr=xerr,
		preprocess=preprocess,postprocess=postprocess,
		kwargs=kwargs)

	_rao = _coeferr
	rao_ = rao(model,label=y,error=yerr)(coef_,x)

	print(coef_)
	print(coef)
	print(_coef)
	print()
	# print(yerr)
	print(_coeferr)
	print(rao_)

	# fig,ax = plt.subplots()
	# ax.plot(x_,y_,label='orig')
	# ax.plot(_x,_y,label='pred')
	# ax.plot(_x,(model(coef_,(_x))),label='func')
	# ax.legend();

	tol = 1e-2
	
	s,a,b = 'coef',_coef,coef_
	eps = norm(a-b)/norm(b)
	assert eps < tol,'%s: %r - %r = %0.3e < %0.1e'%(s,a,b,eps,tol)

	s,a,b = 'covar',abs(_rao),abs(rao_)
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