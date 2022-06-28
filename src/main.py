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

from src.utils import logconfig

conf = 'config/logging.conf'
logger = logconfig(__name__,conf=conf)

from src.quantum import run
from src.functions import functions

def main(args):

	nargs = len(args)

	path = args[0] if nargs>0 else None

	hyperparameters = functions(path)

	run(hyperparameters)

	return

if __name__ == '__main__':
	main(sys.argv[1:])