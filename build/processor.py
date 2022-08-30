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

from src.io import join,split
from src.process import process

def main(args):

	kwargs = {
		'data':'**/data.hdf5',
		'settings':'config/plot.json',
		'hyperparameters':'config/process.json',
		'cwd':None
	}
	nkwargs = len(kwargs)
	nargs = len(args)

	if nargs < 1:
		return

	path = args[0]

	args.extend([path]*(nkwargs-nargs))

	kwargs.update({kwarg: join(arg,kwargs[kwarg]) if kwargs[kwarg] is None or kwargs[kwarg].startswith('*') else kwargs[kwarg] 
		for arg,kwarg in zip(args,kwargs)})

	process(**kwargs)

	return

if __name__ == '__main__':
	main(sys.argv[1:])