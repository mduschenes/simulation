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

from src.utils import array,zeros,rand,eye
from src.utils import abs2,trace,dot,allclose,product,gradient
from src.utils import inner_abs2,inner_real
from src.utils import gradient_inner_abs2,gradient_inner_real
from src.utils import inner_abs2_einsum,inner_real_einsum
from src.utils import gradient_inner_abs2_einsum,gradient_inner_real_einsum


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    # log = file if hasattr(file,'write') else sys.stderr
    # traceback.print_stack(file=log)
    # log.write(warnings.formatwarning(message, category, filename, lineno, line))
    return
warnings.showwarning = warn_with_traceback

def _setup(args,kwargs):
    
    n,d = kwargs['n'],kwargs['d']
    
    shape = (n,n)
    dtype = kwargs['dtype']

    shapes = (shape[:d],shape[:d],(int(product(shape[:d])),*shape[:d]))
    a = rand(shapes[0],dtype=dtype)
    b = rand(shapes[1],dtype=dtype)
    da = eye(shapes[2][0]).reshape((*shape[:d],*shape[:d]))

    updates = {'a':a,'b':b,'da':da,'shapes':shapes}
    
    
    kwargs.update(updates)
    
    return



def test_dot():

    def func(*args,**kwargs):

        a,b,da,shapes,optimize,metric = kwargs['a'],kwargs['b'],kwargs['da'],kwargs['shapes'],kwargs['optimize'],kwargs['metric']

        boolean = True

        if metric in ['real.einsum']:
            f = inner_real_einsum
            g = gradient_inner_real_einsum
            if d == 1:
                _f = lambda *shapes,optimize: (lambda a,b: ((dot(a,b.T))).real)
                _g = lambda *shapes,optimize: (lambda a,b,da: ((dot(da,b.T))).real)
            elif d == 2:
                _f = lambda *shapes,optimize: (lambda a,b: (trace(dot(a,b.T))).real)
                _g = lambda *shapes,optimize: (lambda a,b,da: (trace(dot(da,b.T),axes=(-2,-1))).real)
            else:
                _f = lambda *shapes,optimize: (lambda a,b: (trace(dot(a,b.T))).real)
                _g = lambda *shapes,optimize: (lambda a,b,da: (trace(dot(da,b.T),axes=(-2,-1))).real)
        elif metric in ['abs.einsum']:
            f = inner_abs2_einsum
            g = gradient_inner_abs2_einsum
            if d == 1:
                _f = lambda *shapes,optimize: (lambda a,b: abs2((dot(a,b.T))))
                _g = lambda *shapes,optimize: (lambda a,b,da: ((2*((dot(a,b.T)).conj()))*dot(da,b.T)).real)
            elif d == 2:
                _f = lambda *shapes,optimize: (lambda a,b: abs2(trace(dot(a,b.T))))
                _g = lambda *shapes,optimize: (lambda a,b,da: ((2*((trace(dot(a,b.T),axes=(-2,-1))).conj()))*trace(dot(da,b.T),axes=(-2,-1))).real)
            else:
                _f = lambda *shapes,optimize: (lambda a,b: abs2(trace(dot(a,b.T))))
                _g = lambda *shapes,optimize: (lambda a,b,da: ((2*((trace(dot(a,b.T),axes=(-2,-1))).conj()))*trace(dot(da,b.T),axes=(-2,-1))).real)
        elif metric in ['real']:
            f = lambda *shapes,optimize: inner_real
            g = lambda *shapes,optimize: gradient_inner_real
            if d == 1:
                _f = lambda *shapes,optimize: (lambda a,b: ((dot(a,b.T))).real)
                _g = lambda *shapes,optimize: (lambda a,b,da: ((dot(da,b.T))).real)
            elif d == 2:
                _f = lambda *shapes,optimize: (lambda a,b: (trace(dot(a,b.T))).real)
                _g = lambda *shapes,optimize: (lambda a,b,da: (trace(dot(da,b.T),axes=(-2,-1))).real)
            else:
                _f = lambda *shapes,optimize: (lambda a,b: (trace(dot(a,b.T))).real)
                _g = lambda *shapes,optimize: (lambda a,b,da: (trace(dot(da,b.T),axes=(-2,-1))).real)
        elif metric in ['abs']:
            f = lambda *shapes,optimize: inner_abs2
            g = lambda *shapes,optimize: gradient_inner_abs2            
            if d == 1:
                _f = lambda *shapes,optimize: (lambda a,b: abs2((dot(a,b.T))))
                _g = lambda *shapes,optimize: (lambda a,b,da: ((2*((dot(a,b.T)).conj()))*dot(da,b.T)).real)
            elif d == 2:
                _f = lambda *shapes,optimize: (lambda a,b: abs2(trace(dot(a,b.T))))
                _g = lambda *shapes,optimize: (lambda a,b,da: ((2*((trace(dot(a,b.T),axes=(-2,-1))).conj()))*trace(dot(da,b.T),axes=(-2,-1))).real)
            else:
                _f = lambda *shapes,optimize: (lambda a,b: abs2(trace(dot(a,b.T))))
                _g = lambda *shapes,optimize: (lambda a,b,da: ((2*((trace(dot(a,b.T),axes=(-2,-1))).conj()))*trace(dot(da,b.T),axes=(-2,-1))).real)             
        else:
            f = inner_abs2_einsum
            g = gradient_inner_abs2_einsum
            if d == 1:
                _f = lambda *shapes,optimize: (lambda a,b: abs2((dot(a,b.T))))
                _g = lambda *shapes,optimize: (lambda a,b,da: ((2*((dot(a,b.T)).conj()))*dot(da,b.T)).real)
            elif d == 2:
                _f = lambda *shapes,optimize: (lambda a,b: abs2(trace(dot(a,b.T))))
                _g = lambda *shapes,optimize: (lambda a,b,da: ((2*((trace(dot(a,b.T),axes=(-2,-1))).conj()))*trace(dot(da,b.T),axes=(-2,-1))).real)
            else:
                _f = lambda *shapes,optimize: (lambda a,b: abs2(trace(dot(a,b.T))))
                _g = lambda *shapes,optimize: (lambda a,b,da: ((2*((trace(dot(a,b.T),axes=(-2,-1))).conj()))*trace(dot(da,b.T),axes=(-2,-1))).real)

        func = f(*shapes,optimize=optimize)
        _func = _f(*shapes,optimize=optimize)
        
        out = func(a,b)
        _out = _func(a,b)
        
        boolean &= allclose(out,_out)
        
        grad = g(*shapes,optimize=optimize)
        _grad = _g(*shapes,optimize=optimize)
        _grad_ = gradient(func)
        
        out = grad(a,b,da)
        _out = _grad(a,b,da)
        _out_ = _grad_(a,b).real
        
        boolean &= allclose(out,_out) and allclose(out,_out_) and allclose(_out,_out_)

        return boolean

    n = 20
    dims = [1,2]
    metrics = ['real.einsum',
       'abs.einsum',
        'real','abs',
        ]

    for d in dims:
        for metric in metrics:
            
            args = ()
            kwargs = {}
            
            kwargs.update({
                    'n': n,
                    'd': d,
                    'optimize':None,
                    'metric':metric,
                    'dtype':'complex',
                })

            _setup(args,kwargs)


            boolean = func(*args,**kwargs)

            assert boolean, "%s [dim = %d] metric function error"%(metric,d)
   

    return