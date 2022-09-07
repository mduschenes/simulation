#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,copy
from functools import partial,wraps

import matplotlib
import matplotlib.pyplot as plt

import numpy as onp
import scipy as osp
import jax
import jax.numpy as np
import jax.scipy as sp
import jax.example_libraries.optimizers
np.set_printoptions(linewidth=1000)#,formatter={**{dtype: (lambda x: format(x, '0.2e')) for dtype in ['float','float64',np.float64,np.float32]}})
jax.config.update('jax_platform_name','cpu')
jax.config.update('jax_enable_x64', True)
# jax.set_cpu_device_count(8)
# os.env['XLA_FLAGS'] ='--xla_force_host_platform_device_count=8'


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.utils import rank,eigvalsh,rand,scinotation
from src.io import load,dump,join,split
from src.quantum import Unitary

def plot(root,nsamples=None):

	file = 'plot.mplstyle'
	mplstyle = join(root,file)

	if nsamples is None:
		nsamples = 5

	with matplotlib.style.context(mplstyle):
		for directory in glob(root):
			print(directory)
			continue

		    directory = join(directory,root=root)
		    
		    file = 'settings.json'
		    path = join(directory,file)
		    print('Loading: ',path)
		    hyperparameters = load(path)

		    U = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)

		    parameters = U.parameters

		    func = U.__func__
		    grad = U.__grad__
		    hess = U.__hessian__
		    fisher = U.__fisher__

		    path = join(directory)
		    U.load(path)

		    parameters = U.hyperparameters['optimize']['track']['parameters'][-1]

		    shape = parameters.shape
		    bounds = [-1,1]
		    seed = None
		    random = 'uniform'

		    fig,axes = plt.subplots(1,2)

		    funcs = [hess,fisher]

		    for j,func in enumerate(funcs):
		            ax = axes[j]
		            tol = 1e-4
		            h = func(parameters)
		            e = eigvalsh(h)
		            r = (e**2>= tol).sum()
		            e = np.abs(e)
		            e = e/np.max(e)
		            ax.plot(sorted(abs(e))[:-1],label=r'$\textrm{Optimal}, r = %d$'%(r),marker='o',linestyle='-',linewidth=3)


		    for i in range(nsamples):

		        x = rand(shape=shape,bounds=bounds,key=seed,random=random)
		        x = x.ravel()

		        for j,func in enumerate(funcs):
		            ax = axes[j]
		            tol = 1e-4
		            h = func(x)
		            e = eigvalsh(h)
		            r = (e**2>= tol).sum()
		            e = np.abs(e)
		            e = e/np.max(e)
		            ax.plot(sorted(abs(e))[:-1],label=r'$%d, r = %d$'%(i,r),linestyle='--',linewidth=2)

		    for i,ax in enumerate(axes):
		        ax.set_yscale('log');
		        
		        ax.set_title(r'$M = %d~~~~\epsilon = %s$'%(U.M,scinotation(U.hyperparameters['optimize']['track']['value'][-1],0)));
		        
		        if i == 0:
		            ax.set_ylabel(r'$\left|{\frac{\lambda}{\lambda_{\textrm{max}}}}\right|$')

		        ax.set_xlabel(r'$\textrm{%s Index}$'%(['Hessian','Fisher'][i]))
		#         if i == (len(axes)-1):
		        ax.legend(title=r'$\textrm{Sample, Rank}$',prop={'size':25},loc='lower right');
		        for tickparams in [{"axis":"y","which":"major","length":8,"width":1},
		                {"axis":"y","which":"minor","length":4,"width":0.5},
		                {"axis":"x","which":"major","length":8,"width":1},
		                {"axis":"x","which":"minor","length":4,"width":0.5}]:
		            ax.tick_params(**tickparams)
		        ax.grid(**{"visible":True,"which":"both","axis":"both"})

		    fig.set_size_inches(20,10)
		    
		    file = 'eig.pdf'
		    path = join(directory,file)
		    print(path)
		    print()
		    fig.savefig(path,**{"bbox_inches":"tight","pad_inches":0.2})

def main(*args,**kwargs):

	plot(*args,**kwargs)

	return

if __name__ == '__main__':

	arguments = ['root','nsamples']

	args = argparser(arguments)

	main(*args,**args)

		