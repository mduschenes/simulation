#!/usr/bin/env python

# Import python modules
import pytest
import os,sys
import itertools,functools,copy,warnings

import matplotlib
import matplotlib.pyplot as plt

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','.','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import np,onp,backend,context
from src.io import load,dump,join,split,edit
from src.utils import array,ones,zeros,rand,linspace,logspace,gradient,sort,norm,allclose,log10,exp10,absolute,sqrt,inf,mean,std,sem,bootstrapper,scinotation,delim,separ,blank
from src.fit import fit,cov

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
	# log = file if hasattr(file,'write') else sys.stderr
	# traceback.print_stack(file=log)
	# log.write(warnings.formatwarning(message, category, filename, lineno, line))
	return
# warnings.showwarning = warn_with_traceback



def test_err(path=None,tol=None):

	if backend in ['autograd']:
		return

	scale = 3
	def model(parameters,x):
		y = parameters[0] + parameters[1]*x
		# y = scale*log10(parameters[0] + parameters[1]*log10(x))
		return y

	n = 20
	d = 2
	sigma = 3e-3
	key = {'x':18212,'parameters':[23512313,123],'parameters_':[924254,1047324],'yerr':1313}
	shapes = ((n,),(n,),(n,))
	metric = 'lstsq'

	x = sort(rand((n,),bounds=[0,1],key=key['x']))
	parameters = array([rand(bounds=[0,1],key=key['parameters_'][0]),rand(bounds=[-1,0],key=key['parameters_'][1])])[:d].ravel()
	y = model(parameters,x) 

	xerr = None
	yerr = sigma*rand(n,bounds=[-1,1],key=key['yerr']) if (sigma is not None and sigma>0) else None
	yerr = sigma*ones(n) if (sigma is not None and sigma>0) else None

	y = y#+yerr if yerr is not None else y
	yerr = absolute(yerr) if yerr is not None else yerr

	x_ = x
	y_ = y
	xerr_ = xerr
	yerr_ = yerr
	parameters_ = parameters
	_cov_ = cov(model,shapes=shapes,label=y_,weights=yerr_,metric=metric)(parameters_,x_)

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
		'standardize':True,
		'iterations':1500,
		'alpha':1e-10,'beta':1e-10,
		"c1":0.0001,"c2":0.9,"maxiter":2000,

	}
	
	preprocess = lambda x,y,parameters: (log10(x) if x is not None else None,exp10(y/scale) if y is not None else None,parameters if parameters is not None else None)
	postprocess = lambda x,y,parameters: (exp10(x) if x is not None else None,scale*log10(y) if y is not None else None,parameters if parameters is not None else None)
	
	_func,_y,_parameters,_yerr,_covariance,_other = fit(
		x,y,
		_x=_x,_y=_y,
		func=func,parameters=parameters,
		yerr=yerr,
		xerr=xerr,
		preprocess=preprocess,postprocess=postprocess,
		kwargs=kwargs)

	cov_ = cov(_func,shapes=shapes,label=y_,weights=yerr,metric=metric)(_parameters,x_)
	_cov = _covariance
	print(sigma)
	print('----')
	print(parameters_)
	print(_parameters)
	print()
	print(_cov_)
	print(cov_)
	print(_cov)

	# fig,ax = plt.subplots()
	# ax.plot(x_,y_,label='orig',marker='o',linestyle='')
	# ax.plot(_x,_y,label='pred',marker='*',linestyle='-')
	# ax.plot(_x,(model(parameters_,(_x))),label='func')
	# ax.plot(_x,(_func(parameters_,(_x))),label='$\_$func',linestyle='--')
	# ax.legend();
	# fig.savefig('plot.pdf')

	tol = 1e-7
	
	s,a,b = 'parameters',_parameters,parameters_
	eps = norm(a-b)/norm(b)
	assert eps < tol,'%s: %r - %r = %0.3e < %0.1e'%(s,a,b,eps,tol)

	s,a,b = 'covar',(_cov),(cov_)
	eps = norm(a-b)/norm(b)
	assert eps < tol,'%s: %r - %r = %0.3e < %0.1e'%(s,a,b,eps,tol)

	print('Passed')

	return


def test_fit(path=None,tol=None):


	def plot(x,y,xerr=None,yerr=None,fig=None,ax=None,options=None,**kwargs):

		def setup(options):

			options = {} if options is None else options
			for option in options:
				if option in ['color','ecolor']:
					if isinstance(options[option],str):
						value = options[option].split(separ) if options[option].count(separ) else (options[option],0.5)
						value = getattr(plt.cm,str(value[0]))(float(value[1]))
				else:
					value = options[option]
				options[option] = value

			settings = {}
			settings['path'] = options.pop('path') if options.get('path') else None
			settings['mplstyle'] = options.pop('mplstyle') if options.get('mplstyle') else 'config/plot.mplstyle'
			settings['fill_between'] = options.pop('fill_between') if options.get('fill_between') else None

			return options,settings


		options,settings = setup(options)

		with matplotlib.style.context(settings.get('mplstyle')) if settings.get('mplstyle') else context(settings.get('mplstyle')):

			fig,ax = plt.subplots() if fig is None or ax is None else (fig,ax)

			if settings.get('fill_between'):
				arguments = [x,y]
			else:
				arguments = [x,y,yerr,xerr]

			ax.errorbar(*arguments,**options)

			ax.set_ylabel(ylabel="$y$")
			ax.set_xlabel(xlabel="$1/x$")

			ax.set_xscale(value="log",base=2)
			ax.set_yscale(value="log",base=10)
			ax.set_xlim(xmin=2**(-5),xmax=2**(0))
			ax.set_ylim(ymin=5e-2,ymax=5e0)
			ax.set_xticks(ticks=[2**i for i in [-4,-3,-2,-1]])
			ax.set_xticklabels(labels=['$%d$'%(2**i) if i not in [0,1] else '$2$' if i not in [0] else '$1$' for i in [4,3,2,1]])
			ax.set_yticks(ticks=[10**i for i in [-1,0,1]])
			ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [-1,0,1]])
			ax.tick_params(**{"axis":"y","which":"major","length":8,"width":1})
			ax.tick_params(**{"axis":"y","which":"minor","length":4,"width":0})
			ax.tick_params(**{"axis":"x","which":"major","length":8,"width":1})
			ax.tick_params(**{"axis":"x","which":"minor","length":4,"width":0})

			if settings.get('fill_between'):
				ax.fill_between(x,y,y-yerr,**settings.get('fill_between'))
				ax.fill_between(x,y+yerr,y,**settings.get('fill_between'))

			ax.grid(visible=True)

			ax.legend(
				loc="lower right",
				ncol=1,
				title_fontsize=22,
				prop={"size":20},
				markerscale=1,
				handlelength=3
			)

			if settings.get('path'):
				fig.set_size_inches(w=20,h=12)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname=settings.get('path'))

		return fig,ax

	def model(*args,**kwargs):
		data = load('examples/stats/data.json')

		def func(parameters,x):
			y = parameters[0] - parameters[1]*x
			return y

		data = {attr:array(data[list(data)[0]]['value'][attr]) if data[list(data)[0]]['value'].get(attr) is not None else None for attr in ['x','y','xerr','yerr']}

		settings = dict(
			func='lstsq',
			# func=func,
			parameters=array([2.,1.]),
			intercept=True,
			# intercept=False,
			preprocess="log10.log10",
			postprocess="exp10.exp10",
			bootstrap=dict(size=3,random='gaussian',seed=123),
			_x=logspace(-20,0,100),_y = None,
			)

		return data,settings

	# Model

	args = ()
	kwargs = {}

	data,settings = model(*args,**kwargs)


	# Plot
	fig,ax = None,None
	options = dict(
		path='examples/stats/plot.pdf',mplstyle=None,
		label='$\\textrm{Data}$',
		color='viridis_0.5',alpha=0.8,
		marker='o',linestyle=':',
		markersize=7,
		linewidth=4,
		elinewidth=3,
		capsize=4
	)

	fig,ax = plot(**data,fig=fig,ax=ax,options=options)

	# Fit

	kwargs = {
		'process':True,
		'standardize':True,
		'metric':'lstsq',
		'optimizer':'cg',
		'iterations':3,
		'eps':{'value':1e-20},
		'value':{'value':1},
		'uncertainty':True,
		'alpha':1e-4,'beta':1e-6,
		"c1":1e-2,"c2":1e-1,"maxiter":2000,
		'path':None,
		'verbose':False,
	}

	_func,_y,_parameters,_yerr,_covariance,_other = fit(
		**data,
		**settings,
		kwargs=kwargs,
		)

	# Plot

	x = settings['_x']
	y = _func(_parameters,x)
	yerr = _yerr

	data = dict(x=x,y=y,yerr=yerr)

	options = dict(
		path='examples/stats/plot.pdf',mplstyle='examples/stats/plot.mplstyle',
		label='$\\textrm{Model}~~%s$'%('\t'.join(['\\overline{%s}: %s'%(stat,scinotation(_other[stat],decimals=3)) for stat in _other])),
		fill_between=dict(
			interpolate=True,
			where=None,
			zorder=0,
			color='gray',alpha=0.8
			),
		color='viridis_0.25',alpha=0.8,
		marker='',linestyle='--',
		markersize=7,
		linewidth=4,
		elinewidth=3,
		capsize=4
	)

	fig,ax = plot(**data,fig=fig,ax=ax,options=options)

	print('Passed')

	return


if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8
	# test_err(path,tol)
	test_fit(path,tol)