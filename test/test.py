#!/usr/bin/env python

import os,sys,subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import jit,array,zeros,tensorprod,trotter
from src.utils import allclose,cosh,sinh,real,abs,rand,pi
from src.utils import gradient,hessian,gradient_fwd,gradient_shift
from src.operators import haar
from functools import partial

from jax import config
configs = {'jax_disable_jit':False}
for name in configs:
    config.update(name,configs[name])

N = 2
p = 1
M = 5
seed = None
bounds = [-1,1]
random = 'gaussian'
dtype = 'complex'


n = 2**N

I = array([[1,0],[0,1]],dtype=dtype)
X = array([[0,1],[1,0]],dtype=dtype)
Y = array([[0,-1j],[1j,0]],dtype=dtype)
Z = array([[1,0],[0,-1]],dtype=dtype)
data = [
    *[tensorprod(array([X if k in [i] else I for k in range(N)])) for i in range(N)],
    *[tensorprod(array([Y if k in [i] else I for k in range(N)])) for i in range(N)],
    *[tensorprod(array([Z if k in [i] else I for k in range(N)])) for i in range(N)],
    *[tensorprod(array([Z if k in [i,j] else I for k in range(N)])) for i in range(N) for j in range(N) if i<j],
    ]

K = len(data)

data = array(trotter(data,p))
identity = tensorprod(array([I]*N))

V = haar(shape=(n,n),bounds=bounds,random=random,seed=seed,dtype=dtype)

k = 2*N
slices = (slice(M),slice(None,k))

k = (N*(N-1))//2
slices = (slice(M),slice(-k,None))
          
k = K
slices = (slice(M),slice(k))
          
shape = (M,K)
subshape = (M,k)


X = rand(shape=shape,bounds=bounds,key=seed,random=random)
X = X.ravel()

x = rand(shape=subshape,bounds=bounds,key=seed,random=random)
x = x.ravel()


def model(x,X,data,identity,M,K,k,n,p):
    def _func(x,data,identity):
        return cosh(x)*identity + sinh(x)*data

    D = M*K
    d = M*k
    
    X = X.reshape((M,K))
    x = x.reshape((M,k))
    
    X = X.at[slices].set(x)

    X = X.ravel()
    
    coefficient = -1j/p
    
    out = identity
    for i in range(D):
        out = out.dot(_func(coefficient*X[i],data[i%K],identity))
    return out

model = jit(partial(model,X=X,data=data,identity=identity,M=M,K=K,k=k,n=n,p=p))

def func(x,V,n):
    U = model(x)
#     out = (abs((V.conj().T.dot(U)).trace())/n)**2
    out = (real((V.conj().T.dot(U)).trace())/n)
    return out 

func = jit(partial(func,V=V,n=n))
_grad = jit(gradient_shift(func)) # For p=1,unconstrained features

grad = gradient(func)
hess = hessian(func)

print(allclose(grad(x),_grad(x)))