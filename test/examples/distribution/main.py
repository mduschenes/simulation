#!/usr/bin/env python

# Import python modules
import os,sys
import itertools,functools,warnings,traceback

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','.','..','../../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))


os.environ['NUMPY_BACKEND'] = 'JAX'

from src.utils import array,rand,asscalar,tensorprod,concatenate,meshgrid,linspace,logspace,inplace,partial,cache,scan,vmap,callback,allclose,vtype,copy,exponentiate
from src.utils import exp,log,log1p,sign,gammaln
from src.utils import pi,nan,fltmin,fltmax,delim,separ,epsilon,iterables,arrays
from src.utils import where,real,imag,nonzero,unique,sort,minimum,maximum,minimums,maximums
from src.utils import eig,addition,prod,permutations,partitions,multinomial,permute,distribution
from src.utils import dataframe
from src.utils import scinotation
from src.utils import integral

from src.iterables import Dict

from src.quantum import Basis as basis

from src.io import load,dump,exists,join,split

from src.logger import Logger

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects

from natsort import natsorted

import mpmath as mp
mp.dps = 100

def function(parameters,x):
	x = (x-parameters[0])/(parameters[1]-parameters[0])
	x = where((x>0)*(x<1),(parameters[2]*(1/(parameters[1]-parameters[0]))*exp((parameters[6]*parameters[7]-1)*log(x) + (((parameters[8]-parameters[6])*parameters[7]-1)/2)*log1p(-2*parameters[4]*x + parameters[5]*x**2) - log(parameters[3]) - log(parameters[9]))),0)
	return x

def functions(parameters,x):
	func = lambda y,parameters: y+function(parameters,x)
	y = 0*x
	y = scan(parameters,y,func)
	return y

def Function(parameters,x):
	bounds = logspace(-20,0,50)
	@vmap
	def func(z):
		return z*integral(lambda x,z=z:function(parameters,z*x),bounds)
	y = func(x)
	return y

def Functions(parameters,x):
	bounds = logspace(-20,0,50)
	@vmap
	def func(z):
		return z*integral(lambda x,z=z:functions(parameters,z*x),bounds)
	y = func(x)
	return y

def parameterize(z,d=None,s=None,w=None):

	from mpmath import exp,log,log1p,sign
	from mpmath import quad as integral,linspace,sqrt,mpmathify

	def func(x,parameters):
		x = (x-parameters[0])/(parameters[1]-parameters[0])
		x = (parameters[2]*(1/(parameters[1]-parameters[0]))*exp((parameters[6]*parameters[7]-1)*log(x) + (((parameters[8]-parameters[6])*parameters[7]-1)/2)*log1p(-2*parameters[4]*x + parameters[5]*x**2) - log(parameters[3]) - log(parameters[9]))) if ((x>0)*(x<1)) else 0
		return x

	if z is None:
		parameters = None
		return parameters

	d = len(z) if d is None else d
	s = 1 if s is None else s
	w = 1 if w is None else w

	eps = 0
	args,kwargs = tuple((-32,0,50,)),dict(endpoint=True)
	bounds = exponentiate(linspace(*args,**kwargs))

	u,v = asscalar(minimum(z)),asscalar(maximum(z))

	z = (z-u)/(v-u)

	z = z[(z>eps)*(z<=1)]

	a,b = asscalar(addition(1/z)/z.size),asscalar(addition(1/z**2)/z.size)
	l,s,d = z.size,s,d
	w = w

	alpha = ((a**2)/(b)) if (b>0) else 1
	beta = (2*((l*s)-1)/((d-l)*s-1)) if (((d-l)*s-1)>0) else 0
	gamma = ((beta*(beta+2))/((beta*(beta+2))+1)/alpha) if (alpha>0) else 0
	delta = (a/b) if (b>0) else 0
	optima = [1,*((delta*((beta+1)/(beta+2))*(1+sign*sqrt(1-gamma))) for sign in [1,-1])] if gamma <= 1 else [1]

	params = dict(u=0,v=1,w=1,p=1,a=a,b=b,l=l,s=s,d=d,c=1)

	c = max(func(i,parameters=[params[i] for i in params]) for i in optima)
	c = c if (abs(c)>eps) else 1

	params.update(dict(c=c))
	f = partial(func,parameters=[params[i] for i in params])
	p = integral(f,bounds)

	# params.update(dict(p=p))
	# f = partial(func,parameters=[params[i] for i in params])
	# q = integral(f,bounds)

	params.update({i:j for i,j in dict(p=p,u=u,v=v,w=w).items()})

	parameters = [float(max(min(params[i],fltmax),fltmin)) for i in params]

	return parameters



import sympy as sp
from sympy.combinatorics.named_groups import SymmetricGroup
def _function(parameters,x):

	def trace(x,z,p):
		return addition((1-(x[:,None]/z[None,:]))**p,axis=-1)

	n = 2
	k = 4

	z,l,(d,u,v,w) = parameters[:-k-n],parameters[-k-n:-k],parameters[-k:]

	z,l,d,u,v,w = z.astype(float),l.astype(int),int(d),float(u),float(v),float(w)

	n = len(l)

	l = [int(i) for i in l]

	z = [z[sum(l[:i]):sum(l[:i+1])] for i in range(n)]

	y = 0

	for i,(z,l) in enumerate(zip(z,l)):

		if i == 0:
			x = (x-u)/(v-u)
		else:
			x = (v-x)/(v-u)

		t = d-l-1

		s = 0

		if t>0:
			G = list(SymmetricGroup(t).generate_schreier_sims())
			for g in G:
				k = [len(p) for p in g.full_cyclic_form]
				k = {p:k.count(p) for p in set(k)}
				s += prod(trace(x,z,p)**k[p] for p in k)
		else:
			s += exp(gammaln(l+t)-gammaln(l))

		s *= sign(w)*exp(log(abs(w))+(gammaln(d)-gammaln(l)-gammaln(d-l))-(gammaln(l+t)-gammaln(l))-sum(log(z))+(l-1)*log(x))

		y += s

	y /= n

	return y

def _functions(parameters,x):
	func = lambda y,parameters: y+_function(parameters,x)
	y = 0*x
	for params in parameters:
		y = func(y,params)
	return y

def _Function(parameters,x):
	bounds = logspace(-20,0,50)
	@vmap
	def func(z):
		return z*integral(lambda x,z=z:_function(parameters,z*x),bounds)
	y = func(x)
	return y

def _Functions(parameters,x):
	bounds = logspace(-20,0,50)
	@vmap
	def func(z):
		return z*integral(lambda x,z=z:_functions(parameters,z*x),bounds)
	y = func(x)
	return y

def _parameterize(z,d=None,s=None,w=None):

	if z is None:
		parameters = None
		return parameters

	d = len(z) if d is None else d
	s = 1 if s is None else s
	w = 1 if w is None else w

	eps = 0

	u,v = asscalar(minimum(z)),asscalar(maximum(z))

	z = (z-u)/(v-u),(v-z)/(v-u)

	z = [i[(i>eps)*(i<=1)] for i in z]

	l,s,d = [i.size for i in z],s,d
	w = w

	params = [*[j for i in z for j in i],*l,d,u,v,w]

	parameters = [i for i in params]

	return parameters


vectorize = 0

if vectorize:

	def functional(parameters,x):

		z,l,w = parameters

		z,l,w = z.astype(float),l.astype(int),w.astype(float)

		z,l,w = z[l>0],l[l>0],w

		n,d,u,v = z.size,addition(l),minimum(z),maximum(z)

		z = (z-u)/(v-u)

		x = (x-u)/(v-u)

		y = sum(
			(
			sign(w[sum(l[:i])+t])
			*
			sign(z[i]-x)**((d-l[i]+t)%2)
			*
			exp((d-l[i]+t-1)*log(abs(z[i]-x)) + log(abs(w[sum(l[:i])+t])))
			)
			for i in range(n)
			for t in range(l[i])
		)

		y = where((x>0)*(x<1),y,0)

		return y

	def functionals(parameters,x):

		func = lambda y,parameters: y + functional(parameters,x)
		y = 0*x

		for params in parameters:
			y = func(y,params)

		return y

	def Functional(parameters,x):

		z,l,w = parameters

		z,l,w = z.astype(float),l.astype(int),w.astype(float)

		z,l,w = z[l>0],l[l>0],w

		n,d,u,v = z.size,addition(l),minimum(z),maximum(z)

		y = sum(
			(
			sign(w[sum(l[:i])+t])
			*
			(
			(
			-sign(z[i]-x)**((d-l[i]+t+1)%2)
			*
			exp((d-l[i]+t)*log(abs(z[i]-x)) + log(abs(w[sum(l[:i])+t])) - log(d-l[i]+t))
			)
			+
			(
			exp((d-l[i]+t)*log(abs(z[i]-u)) + log(abs(w[sum(l[:i])+t])) - log(d-l[i]+t))
			)
			)
			)
			for i in range(n)
			for t in range(l[i])
		)

		y = where((x>u)*(x<v),y,0)

		return y

	def Functionals(parameters,x):

		func = lambda y,parameters: y+Functional(parameters,x)
		y = 0*x

		for params in parameters:
			y = func(y,params)

		return y

	def Functionals(parameters,x):

		func = lambda y,parameters: y+Functional(parameters,x)
		y = 0*x

		for params in parameters:
			y = func(y,params)

		return y

	def parameterization(z,d=None,s=None,w=None):

		@cache
		def factorial(l,t):
			return gammaln(d) - gammaln(d-l+t) - gammaln(l-t)

		@cache
		def factorials(l,t):
			return gammaln(l+t) - gammaln(l) - gammaln(t+1)

		d = 1 if d is None else d
		s = 1 if s is None else s
		w = 1 if w is None else w

		eps = 0
		opts = dict(return_counts=True)

		z,l = unique(z,**opts)

		w = [w]*len(z) if not isinstance(w,iterables) else [*w]

		l *= s

		n,d,u,v = z.size,addition(l),minimum(z),maximum(z)

		z = (z-u)/(v-u)

		w = array([
			(
			(1/2)*((-1)**(t%2))*sign(w[i])
			*
			(1/(v-u))
			*
			exp(log(abs(w[i]))+factorial(l[i],t))
			*
			sum(
			(
			exp(sum(factorials(l[j],p[j])-((l[j]+p[j])*log(abs(z[i]-z[j])))
				for j in range(n) if (j!=i)))
			*
			prod(sign(z[i]-z[j])**((l[j]+p[j])%2)
				for j in range(n) if (j!=i))
			)
			for p in permutations(range(t+1),repeat=n)
			if ((p[i]==0) and (sum(p)==t))
			)
			)
			for i in range(n)
			for t in range(l[i])
		])

		z = (v-u)*z + u

		params = [z,l,w]

		parameters = params

		return parameters

else:

	def functional(parameters,x):

		from mpmath import exp,log,sign

		z,l,w = parameters

		n,d,u,v = len(z),sum(l),min(z),max(z)

		z = [(z[i]-u)/(v-u) for i in range(n)]

		x = [(y-u)/(v-u) for y in x]

		y = [sum(
			(
			w[sum(l[:i])+t]
			*
			sign(z[i]-y)
			*
			((z[i]-y)**(d-l[i]+t-1))
			)
			for i in range(n)
			for t in range(l[i])
			) if (y>u)*(y<=v) else 0
			for y in x]

		return y

	def functionals(parameters,x):

		func = lambda y,parameters: [i+j for i,j in zip(y,functional(parameters,x))]
		y = [0 for i in x]

		for params in parameters:
			y = func(y,params)

		return y

	def Functional(parameters,x):

		from mpmath import exp,log,sign

		z,l,w = parameters

		n,d,u,v = len(z),sum(l),min(z),max(z)

		z = [(z[i]-u)/(v-u) for i in range(n)]

		x = [(y-u)/(v-u) for y in x]

		y = [sum(
			(
			w[sum(l[:i])+t]
			*
			(1/(d-l[i]+t))
			*
			(
			(-(sign(z[i]-y)*((z[i]-y)**(d-l[i]+t))))
			+
			((z[i]-u)**(d-l[i]+t))
			)
			)
			for i in range(n)
			for t in range(l[i])
			) if (y>u)*(y<v) else 0 if (y<=u) else 1 if (y>=u) else 0
			for y in x]

		return y

	def Functionals(parameters,x):

		func = lambda y,parameters: [i+j for i,j in zip(y,Functional(parameters,x))]
		y = [0 for i in x]

		for params in parameters:
			y = func(y,params)

		return y

	def parameterization(z,d=None,s=None,w=None):

		from mpmath import exp,log,sign,gammaprod
		from itertools import product
		from math import prod

		@cache
		def factorial(l,t):
			return gammaprod([d],[d-l+t,l-t])

		@cache
		def factorials(l,t):
			return gammaprod([l+t],[t+1,l])

		d = 1 if d is None else d
		s = 1 if s is None else s
		w = 1 if w is None else w

		eps = 0

		z = list(map(lambda i: float(asscalar(i)),z))

		z,l = list(set(z)),[z.count(i) for i in set(z)]

		w = [w]*len(z) if not isinstance(w,iterables) else [*w]

		l = [i*s for i in l]

		n,d,u,v = len(z),sum(l),min(z),max(z)

		z = [(z[i]-u)/(v-u) for i in range(n)]

		w = [
			(
			(1/2)*((-1)**(t%2))
			*
			w[i]
			*
			(1/(v-u))
			*
			factorial(l[i],t)
			*
			sum(
			(
			prod(factorials(l[j],p[j])*(abs(z[i]-z[j])**(l[j]+p[j]))
				for j in range(n) if (j!=i))
			*
			prod(sign(z[i]-z[j])**((l[j]+p[j])%2)
				for j in range(n) if (j!=i))
			)
			for p in product(range(t+1),repeat=n)
			if ((p[i]==0) and (sum(p)==t))
			)
			)
			for i in range(n)
			for t in range(l[i])
			]

		z = [(z[i]-u)/(v-u) for i in range(n)]

		params = [z,l,w]

		parameters = params

		return parameters

def run(settings,options,*args,**kwargs):

	for index,setting in enumerate(permute(settings)):

		attr = setting['attr']
		D = setting['D']
		N = setting['N']
		M = setting['M']
		L = setting['L']

		attribute = options['attribute'](setting,options)

		path = options['data'](setting,options)
		key = options['key'](setting,options)
		io = options['io'](setting,options)

		logger = options['logger'](setting,options)

		do = options['do'](setting,options)

		data = load(path)

		if not do:
			continue

		logger(setting)

		data = attribute['data']()

		parameters = []

		for number,partition in enumerate(partitions(N,D**2)):

			try:

				z = tensorprod([obj for i,j in enumerate(partition) for obj in [data[i]]*j])

				z = attribute['func'](data=z)

				d = D**N
				s = M+1
				w = multinomial(partition)/(d**2)

				if attribute['method']() in ['func']:

					params = parameterize

				elif attribute['method']() in ['functional']:

					params = parameterization

				elif attribute['method']() in ['_func']:

					params = _parameterize

				params = params(z,d=d,s=s,w=w)

				parameters.append(params)

			except Exception as exception:

				logger('Exception:\n%r\n%r'%(exception,traceback.format_exc()))


		if vectorize:

			if attribute['method']() in ['func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

			elif attribute['method']() in ['functional']:

				size = max(len(i) for params in parameters for i in params)
				parameters = [[[*i,*[0]*(size-len(i))] for i in params] for params in parameters]
				parameters = array(parameters)

			elif attribute['method']() in ['_func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

		else:

			if attribute['method']() in ['func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

			elif attribute['method']() in ['functional']:

				size = max(len(i) for params in parameters for i in params)
				parameters = [params for params in parameters]
				parameters = parameters

			elif attribute['method']() in ['_func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

		value = {key:dict(parameters=parameters)}

		data = load(path,**io)

		data.update(value)

		dump(data,path,**io)

	return

def process(settings,options,*args,**kwargs):

	def plot(x,y,xerr=None,yerr=None,fig=None,ax=None,options=None,**kwargs):

		def setup(options):

			options = {} if options is None else options
			for option in options:
				if option in ['color','ecolor']:
					if isinstance(options[option],str):
						value = options[option].split('_') if options[option].count('_') else (options[option],0.5)
						value = getattr(plt.cm,str(value[0]))(float(value[1])) if hasattr(plt.cm,value[0]) else value[0]
				else:
					value = options[option]
				options[option] = value

			settings = {}
			settings['path'] = options.pop('path') if options.get('path') else None
			settings['mplstyle'] = options.pop('mplstyle') if options.get('mplstyle') else None

			return options,settings

		options,settings = setup(options)

		with matplotlib.style.context(settings.get('mplstyle')) if settings.get('mplstyle') else context(settings.get('mplstyle')):

			fig,ax = plt.subplots() if fig is None or ax is None else (fig,ax)

			ax.errorbar(x,y,yerr,xerr,**options)

			ax.set_xlabel(xlabel='$p$',size=60)
			ax.set_ylabel(ylabel='$P(p)$',size=60)

			# ax.set_xscale(value='log',base=10)
			# ax.set_yscale(value='log',base=10)
			# ax.set_xlim(xmin=1e-22,xmax=1e2)
			# ax.set_ylim(ymin=1e-21,ymax=1e21)
			# ax.set_xticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1])
			# ax.set_xticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-20,-16,-12,-8,-4,0]],size=60)
			# ax.set_yticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16,1e20])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-20,-16,-12,-8,-4,0,4,8,12,16,20]],size=60)

			# ax.set_xscale(value='log',base=4)
			# ax.set_yscale(value='log',base=10)
			# ax.set_xlim(xmin=2**(-11),xmax=2**(2))
			# ax.set_ylim(ymin=1e-129,ymax=1e9)
			# ax.set_xticks(ticks=[2**(-2*i) for i in [10,8,6,4,2,0]])
			# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [10,8,6,4,2,0]],size=60)
			# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-128,-64,-32,-16,-12,-8,-4,-0,4,8]],size=60)

			ax.set_xscale(value='log',base=4)
			ax.set_yscale(value='log',base=10)
			ax.set_xlim(xmin=2**(-2*17),xmax=2**(2))
			ax.set_ylim(ymin=1e-129,ymax=1e129)
			ax.set_xticks(ticks=[2**(-2*i) for i in [16,14,12,10,8,6,4,2,0]])
			ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [16,14,12,10,8,6,4,2,0]],size=60)
			ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16,1e32,1e64,1e128])
			ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-128,-64,-32,-16,-12,-8,-4,0,4,8,12,16,32,64,128]],size=60)

			ax.tick_params(**{'axis':'y','which':'major','length':6,'width':1,'pad':10})
			ax.tick_params(**{'axis':'y','which':'minor','length':4,'width':0})
			ax.tick_params(**{'axis':'x','which':'major','length':6,'width':1,'pad':10})
			ax.tick_params(**{'axis':'x','which':'minor','length':4,'width':0})

			ax.grid(visible=True)

			handles,labels = ax.get_legend_handles_labels()
			handles,labels = [copy(handle) for handle in handles],[copy(label) for label in labels]
			for handle,label in zip(handles,labels):
				handle[0].set_linewidth(12)
			indices = [labels.index(label) for label in natsorted(labels)]
			handles,labels = [handles[i] for i in indices],[labels[i] for i in indices]

			legend = ax.legend(
				handles,labels,
				title='$k$',
				loc='upper right',
				ncol=3,
				title_fontsize=50,
				prop={'size':50},
				markerscale=6,
				handlelength=2.5
			)

			if settings.get('path'):
				fig.set_size_inches(w=48,h=48)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname=settings.get('path'))

		return fig,ax


	fig,ax = {},{}

	for index,setting in enumerate(permute(settings)):

		attrs = options['attrs'](setting,options)
		method = options['method'](setting,options)
		attribute = options['attribute'](setting,options)

		path = options['data'](setting,options)
		key = options['key'](setting,options)
		io = options['io'](setting,options)
		plots = options['plot'](setting,options)

		logger = options['logger'](setting,options)

		data = load(path,**io)

		do = ((data is not None) and (data.get(key) is not None) and (data[key].get('parameters') is not None))

		if not do:
			continue

		logger(setting)

		parameters = data[key]['parameters']

		attr = tuple(setting[attr] if not isinstance(setting[attr],str) else setting[attr].split(delim)[-1] for attr in attrs)

		args,kwargs = tuple((-32,0,1000,)),dict(endpoint=True)

		if (attr not in fig) or (attr not in ax):
			fig[attr],ax[attr] = None,None

		# if vectorize:

		# 	if attribute['method']() in ['func']:

		# 		x = logspace(*args,**kwargs)

		# 		func = {'pdf':functions,'cdf':Functions}[method]

		# 		plts = dict(
		# 			label='$\\textrm{Conjecture}~:~{%s}$'%(setting['M']),
		# 			color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
		# 			marker='',
		# 			linestyle='-',
		# 			alpha=0.5,
		# 			)

		# 	elif attribute['method']() in ['functional']:

		# 		x = logspace(*args,**kwargs)

		# 		func = {'pdf':functionals,'cdf':Functionals}[method]

		# 		plts = dict(
		# 			label='$\\textrm{Analytical}~:~{%s}$'%(setting['M']),
		# 			color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
		# 			marker='',
		# 			linestyle='--',
		# 			alpha=0.8,
		# 			)

		# 	elif attribute['method']() in ['_func']:

		# 		x = logspace(*args,**kwargs)

		# 		func = {'pdf':_functions,'cdf':_Functions}[method]

		# 		plts = dict(
		# 			label='$\\textrm{Symmetry}~:~{%s}$'%(setting['M']),
		# 			color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
		# 			marker='',
		# 			linestyle='-',
		# 			alpha=0.25,
		# 			)

		# else:

		# 	if attribute['method']() in ['func']:

		# 		x = logspace(*args,**kwargs)

		# 		func = {'pdf':functions,'cdf':Functions}[method]

		# 		plts = dict(
		# 			label='$\\textrm{Conjecture}~:~{%s}$'%(setting['M']),
		# 			color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
		# 			marker='',
		# 			linestyle='-',
		# 			alpha=0.5,
		# 			)

		# 	elif attribute['method']() in ['functional']:

		# 		from mpmath import linspace

		# 		x = exponentiate(linspace(*args,**kwargs))

		# 		func = {'pdf':functionals,'cdf':Functionals}[method]

		# 		plts = dict(
		# 			label='$\\textrm{Analytical}~:~{%s}$'%(setting['M']),
		# 			color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
		# 			marker='',
		# 			linestyle='--',
		# 			alpha=0.8,
		# 			)

		# 	elif attribute['method']() in ['_func']:

		# 		x = logspace(*args,**kwargs)

		# 		func = {'pdf':_functions,'cdf':_Functions}[method]

		# 		plts = dict(
		# 			label='$\\textrm{Symmetry}~:~{%s}$'%(setting['M']),
		# 			color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
		# 			marker='',
		# 			linestyle='-',
		# 			alpha=0.5,
		# 			)


		# y = func(parameters,x)

		# x,y = array([float(max(min(i,fltmax),fltmin)) for i in x]),array([float(max(min(i,fltmax),fltmin)) for i in y])

		# x,y = x[y>=0],y[y>=0]

		# fig[attr],ax[attr] = plot(x,y,fig=fig[attr],ax=ax[attr],options={**plots,**plts})



		if attribute['method']() in ['func']:

			d = setting['D']**setting['N']
			s = setting['M']+1
			l = setting['L']
			m = setting['M']
			parameters = setting['parameters']

			parameters = 1 - ((1-parameters)**(m))

			params = dict(
				a = l*s,
				b = (d-l)*s,
				loc = parameters/d/d,
				scale = 1/(1-parameters),
			)

			plts = dict(
				label='$\\textrm{Distribution}~:~{%s}$'%(setting['M']),
				color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
				marker='',
				linestyle=':',
				alpha=1,
				)

			opts = dict(function=f'beta.{method}',**params)

			x = logspace(*args,**kwargs)

			y = distribution(x,**opts)

			fig[attr],ax[attr] = plot(x,y,fig=fig[attr],ax=ax[attr],options={**plots,**plts})

	return


def test(settings,options,*args,**kwargs):

	def plot(x,y,xerr=None,yerr=None,fig=None,ax=None,options=None,**kwargs):

		def setup(options):

			options = {} if options is None else options
			for option in options:
				if option in ['color','ecolor']:
					if isinstance(options[option],str):
						value = options[option].split('_') if options[option].count('_') else (options[option],0.5)
						value = getattr(plt.cm,str(value[0]))(float(value[1])) if hasattr(plt.cm,value[0]) else value[0]
				else:
					value = options[option]
				options[option] = value

			settings = {}
			settings['path'] = options.pop('path') if options.get('path') else None
			settings['mplstyle'] = options.pop('mplstyle') if options.get('mplstyle') else None

			return options,settings

		options,settings = setup(options)

		with matplotlib.style.context(settings.get('mplstyle')) if settings.get('mplstyle') else context(settings.get('mplstyle')):

			fig,ax = plt.subplots() if fig is None or ax is None else (fig,ax)

			ax.errorbar(x,y,yerr,xerr,**options)

			ax.set_xlabel(xlabel='$x$',size=60)
			ax.set_ylabel(ylabel='$f(x)$',size=60)

			ax.set_xscale(value='log',base=10)
			ax.set_yscale(value='log',base=10)
			ax.set_xlim(xmin=1e-5,xmax=1e1)
			ax.set_ylim(ymin=1e-17,ymax=1e17)
			ax.set_xticks(ticks=[1e-4,1e-3,1e-2,1e-1,1])
			ax.set_xticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-4,-3,-2,-1,0]],size=60)
			ax.set_yticks(ticks=[1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16])
			ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-16,-12,-8,-4,0,4,8,12,16]],size=60)

			# ax.set_xscale(value='log',base=4)
			# ax.set_yscale(value='log',base=10)
			# ax.set_xlim(xmin=2**(-11),xmax=2**(2))
			# ax.set_ylim(ymin=1e-129,ymax=1e9)
			# ax.set_xticks(ticks=[2**(-2*i) for i in [10,8,6,4,2,0]])
			# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [10,8,6,4,2,0]],size=60)
			# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-128,-64,-32,-16,-12,-8,-4,0,4,8]],size=60)

			# ax.set_xscale(value='log',base=4)
			# ax.set_yscale(value='log',base=10)
			# ax.set_xlim(xmin=2**(-2*17),xmax=2**(2))
			# ax.set_ylim(ymin=1e-129,ymax=1e9)
			# ax.set_xticks(ticks=[2**(-2*i) for i in [16,14,12,10,8,6,4,2,0]])
			# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [16,14,12,10,8,6,4,2,0]],size=60)
			# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-128,-64,-32,-16,-12,-8,-4,0,4,8]],size=60)

			ax.tick_params(**{'axis':'y','which':'major','length':6,'width':1,'pad':10})
			ax.tick_params(**{'axis':'y','which':'minor','length':4,'width':0})
			ax.tick_params(**{'axis':'x','which':'major','length':6,'width':1,'pad':10})
			ax.tick_params(**{'axis':'x','which':'minor','length':4,'width':0})

			ax.grid(visible=True)

			handles,labels = ax.get_legend_handles_labels()
			handles,labels = [copy(handle) for handle in handles],[copy(label) for label in labels]
			for handle,label in zip(handles,labels):
				handle[0].set_linewidth(12)
			indices = [labels.index(label) for label in natsorted(labels)]
			handles,labels = [handles[i] for i in indices],[labels[i] for i in indices]

			legend = ax.legend(
				handles,labels,
				title='$k$',
				loc='upper right',
				ncol=2,
				title_fontsize=50,
				prop={'size':50},
				markerscale=1.5,
				handlelength=2.5
			)

			if settings.get('path'):
				fig.set_size_inches(w=48,h=30)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname=settings.get('path'))

		return fig,ax


	settings = dict(
		attr=['test.pauli','check.pauli'],
		D=[2],
		N=[3],
		M=[0],
		L=[2],
		parameters=[0],
		)

	fig,ax = None,None

	for index,setting in enumerate(permute(settings)):

		attr = setting['attr']
		D = setting['D']
		N = setting['N']
		M = setting['M']
		L = setting['L']

		method = options['method'](setting,options)
		attribute = options['attribute'](setting,options)

		path = options['path'](setting,options)
		plots = options['plot'](setting,options)
		logger = options['logger'](setting,options)

		args,kwargs = tuple((-32,0,1000,)),dict(endpoint=False)

		d = D**N
		s = M+1
		w = 1

		l = L
		z = {2.3433e-2:3,5.4553e-2:3,7.8291e-2:1}; z = {**z,**{1.2954e-2:d-sum(z[i] for i in z)}}; z = {i:z[i] for i in z if z[i]>0}
		# z = {1:l}; z = {**z,**{0:d-sum(z[i] for i in z)}}; z = {i:z[i] for i in z if z[i]>0}
		# z = {(i+1)/d:1 for i in range(d)}; z = {**z,**{0:d-sum(z[i] for i in z)}}; z = {i:z[i] for i in z if z[i]>0}

		z = array([j for i in z for j in [i]*z[i]])

		logger(setting)

		parameters = []

		try:

			if attribute['method']() in ['func']:

				func = parameterize

			elif attribute['method']() in ['functional']:

				func = parameterization

			elif attribute['method']() in ['_func']:

				func = _parameterize

			params = func(z,d=d,s=s,w=w)

			parameters.append(params)

		except Exception as exception:

			logger('Exception:\n%r\n%r'%(exception,traceback.format_exc()))

		if vectorize:

			if attribute['method']() in ['func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

				x = logspace(*args,**kwargs)

				func = {'pdf':functions,'cdf':Functions}[method]

				plts = dict(
					label='$\\textrm{Incorrect}~:~{%s}$'%(setting['M']),
					color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
					marker='',
					linestyle='-',
					alpha=0.5,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

			elif attribute['method']() in ['functional']:

				size = max(len(i) for params in parameters for i in params)
				parameters = [[[*i,*[0]*(size-len(i))] for i in params] for params in parameters]
				parameters = array(parameters)

				x = logspace(*args,**kwargs)

				func = {'pdf':functionals,'cdf':Functionals}[method]

				plts = dict(
					label='$\\textrm{Zanardi}~:~{%s}$'%(setting['M']),
					color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
					marker='',
					linestyle='--',
					alpha=0.8,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

			elif attribute['method']() in ['_func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

				x = logspace(*args,**kwargs)

				func = {'pdf':_functions,'cdf':_Functions}[method]

				plts = dict(
					label='$\\textrm{Matt}~:~{%s}$'%(setting['M']),
					color='black',
					marker='o',
					linestyle='-',
					markersize=45,
					alpha=0.25,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

		else:

			if attribute['method']() in ['func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

				x = logspace(*args,**kwargs)

				func = {'pdf':functions,'cdf':Functions}[method]

				plts = dict(
					label='$\\textrm{Incorrect}~:~{%s}$'%(setting['M']),
					color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
					marker='',
					linestyle='-',
					alpha=0.5,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

			elif attribute['method']() in ['functional']:

				from mpmath import linspace

				size = max(len(i) for params in parameters for i in params)
				parameters = [params for params in parameters]
				parameters = parameters

				x = exponentiate(linspace(*args,**kwargs))

				func = {'pdf':functionals,'cdf':Functionals}[method]

				plts = dict(
					label='$\\textrm{Zanardi}~:~{%s}$'%(setting['M']),
					color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
					marker='',
					linestyle='--',
					alpha=0.8,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

			elif attribute['method']() in ['_func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

				x = logspace(*args,**kwargs)

				func = {'pdf':_functions,'cdf':_Functions}[method]

				plts = dict(
					label='$\\textrm{Matt}~:~{%s}$'%(setting['M']),
					color='black',
					marker='o',
					linestyle='-',
					markersize=45,
					alpha=0.25,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

		y = func(parameters,x)

		fig,ax = plot(x,y,fig=fig,ax=ax,options={**plots,**plts})

	# plts = dict(
	# 	label='$\\textrm{Beta}~:~{%s}$'%(setting['M']),
	# 	color='viridis_%f'%((settings['M'].index(setting['M']))/(len(settings['M'])+1)),
	# 	marker='',
	# 	linestyle=':',
	# 	alpha=1,
	# 	path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
	# 	)

	# opts = dict(function=f'beta.{method}',a=l*s,b=(d-l)*s)

	# x = logspace(*args,**kwargs)

	# y = distribution(x,**opts)

	# fig,ax = plot(x,y,fig=fig,ax=ax,options={**plots,**plts})

	return





def draw(settings,options,*args,**kwargs):

	def plot(x,y,xerr=None,yerr=None,fig=None,ax=None,options=None,**kwargs):

		def setup(options):

			options = {} if options is None else options
			for option in options:
				if option in ['color','ecolor']:
					if isinstance(options[option],str):
						value = options[option].split('_') if options[option].count('_') else (options[option],0.5)
						value = getattr(plt.cm,str(value[0]))(float(value[1])) if hasattr(plt.cm,value[0]) else value[0]
				else:
					value = options[option]
				options[option] = value

			settings = {}
			settings['path'] = options.pop('path') if options.get('path') else None
			settings['mplstyle'] = options.pop('mplstyle') if options.get('mplstyle') else None
			settings['fontsize'] = 200
			settings['legend.fontsize'] = 110
			settings['n'] = options.pop('n') if options.get('n') else None

			return options,settings

		options,settings = setup(options)

		with matplotlib.style.context(settings.get('mplstyle')) if settings.get('mplstyle') else context(settings.get('mplstyle')):

			fig,ax = plt.subplots() if fig is None or ax is None else (fig,ax)

			ax.errorbar(x,y,yerr,xerr,**options)

			ax.set_xlabel(xlabel='$\\textrm{Expectation Value}~~x/(1/d)$',size=settings['fontsize'])
			ax.set_ylabel(ylabel='$P_{\\Pi}(x) ~\\sim~ x^{ls-1}~(1-x)^{(d-l)s-1}$',size=settings['fontsize'])# ~~\\to~~ P_{\\gamma} = \\frac{1}{1-\\gamma}P(\\frac{x-\\gamma/d}{1-\\gamma})$',size=settings['fontsize'])

			# ax.set_xscale(value='log',base=10)
			# ax.set_yscale(value='log',base=10)
			# ax.set_xlim(xmin=1e-5,xmax=1e1)
			# ax.set_ylim(ymin=1e-17,ymax=1e17)
			# ax.set_xticks(ticks=[1e-4,1e-3,1e-2,1e-1,1])
			# ax.set_xticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-4,-3,-2,-1,0]],size=settings['fontsize'])
			# ax.set_yticks(ticks=[1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-16,-12,-8,-4,0,4,8,12,16]],size=settings['fontsize'])

			# ax.set_xscale(value='log',base=4)
			# ax.set_yscale(value='log',base=10)
			# ax.set_xlim(xmin=2**(-11),xmax=2**(2))
			# ax.set_ylim(ymin=1e-129,ymax=1e9)
			# ax.set_xticks(ticks=[2**(-2*i) for i in [10,8,6,4,2,0]])
			# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [10,8,6,4,2,0]],size=settings['fontsize'])
			# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-128,-64,-32,-16,-12,-8,-4,0,4,8]],size=settings['fontsize'])

			# ax.set_xscale(value='log',base=4)
			# ax.set_yscale(value='log',base=10)
			# ax.set_xlim(xmin=2**(-2*17),xmax=2**(2))
			# ax.set_ylim(ymin=1e-129,ymax=1e9)
			# ax.set_xticks(ticks=[2**(-2*i) for i in [16,14,12,10,8,6,4,2,0]])
			# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [16,14,12,10,8,6,4,2,0]],size=settings['fontsize'])
			# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-128,-64,-32,-16,-12,-8,-4,0,4,8]],size=settings['fontsize'])


			ax.set_xscale(value='log',base=4)
			ax.set_yscale(value='log',base=10)
			ax.set_xlim(xmin=2**(-2*17),xmax=2**(2))
			ax.set_ylim(ymin=1e-17,ymax=1e9)
			ax.set_xticks(ticks=[2**(-2*i) for i in [16,12,8,4,0]])
			ax.set_xticklabels(labels=['$10^{%d}$'%(8-2*i) if (8-2*i) not in [0] else '$1$' for i in [16,12,8,4,0]],size=settings['fontsize'])
			ax.set_yticks(ticks=[1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
			ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-16,-12,-8,-4,0,4,8]],size=settings['fontsize'])


			ax.tick_params(**{'axis':'y','which':'major','length':6,'width':1,'pad':30})
			ax.tick_params(**{'axis':'y','which':'minor','length':4,'width':0})
			ax.tick_params(**{'axis':'x','which':'major','length':6,'width':1,'pad':50})
			ax.tick_params(**{'axis':'x','which':'minor','length':4,'width':0})

			ax.grid(visible=True)

			if (settings.get('n') is not None) and (len(ax.get_legend_handles_labels()[1]) == settings['n']):

				n = len(set([i.replace('$','').split('~,~')[0] for i in ax.get_legend_handles_labels()[1]]))

				opts = dict(location='right',fraction=0.15,shrink=1,aspect=40,pad=0.02,anchor=(1.25,0.5))

				colors = [plt.get_cmap('viridis')((i+1)/(n+1)) for i in range(n)]

				cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name=None,colors=colors,N=100*len(colors))

				cax,opts = matplotlib.colorbar.make_axes([ax for ax in fig.axes],**opts)

				opts = {**dict(cmap=cmap,orientation='vertical')}
				cbar = matplotlib.colorbar.ColorbarBase(cax,**opts)

				cbar.ax.set_ylabel(ylabel='$\\textrm{Depth}~~k$',size=settings['fontsize'])
				cbar.ax.set_yticks(ticks=[(i)/(n-1) for i in range(n)])
				cbar.ax.set_yticklabels(labels=['$%s$'%(i.replace('$','').split('~,~')[0]) for i in ax.get_legend_handles_labels()[1][:n]],size=settings['fontsize'])

				cbar.ax.yaxis.set_tick_params(pad=20)
				cbar.ax.tick_params(labelsize=settings['fontsize'],which='major',pad=20,size=15,length=15,width=1)

				handles,labels = ax.get_legend_handles_labels()
				handles,labels = [copy(handle) for handle in handles],[copy(label) for label in labels]
				for i,(handle,label) in enumerate(zip(handles,labels)):
					handle[0].set_linewidth(16)
					# handle[0].set_color('gray')
					labels[i] = '$%s~,~%s$'%(labels[i].replace('$','').split('~,~')[1],('10^{%s}'%(labels[i].replace('$','').split('~,~')[2]) if int(labels[i].replace('$','').split('~,~')[2]) != 0 else '0'))
				indices = [i[len(i)//2] for i in [[i for i,obj in enumerate(labels) if obj==label] for label in natsorted(set(labels))]]
				handles,labels = [handles[i] for i in indices],[labels[i] for i in indices]

				legend = ax.legend(
					handles,labels,
					title='$\\textrm{Rank}~~l ~,~ \\textrm{Noise}~~\\gamma$',
					loc='upper center',
					ncol=4,
					title_fontsize=settings['legend.fontsize'],
					prop={'size':settings['legend.fontsize']},
					markerscale=6,
					handlelength=3
				)

			else:
				pass
				# handles,labels = ax.get_legend_handles_labels()
				# handles,labels = [copy(handle) for handle in handles],[copy(label) for label in labels]
				# for handle,label in zip(handles,labels):
				# 	handle[0].set_linewidth(12)
				# indices = [labels.index(label) for label in natsorted(labels)]
				# handles,labels = [handles[i] for i in indices],[labels[i] for i in indices]


			if settings.get('path'):
				fig.set_size_inches(w=65,h=45)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname=settings.get('path'),bbox_inches='tight',pad_inches=0.5)

		return fig,ax


	settings = dict(
		# attr=['test.pauli','check.pauli','pauli'],
		attr=['distribution'],
		D=[2],
		N=[8],
		L=[1,(1,2)],
		parameters=[0,-2],
		M=[0,2,4,8,16,32],
		)

	fig,ax = None,None

	for index,setting in enumerate(permute(settings)):

		attr = setting['attr']
		D = setting['D']
		N = setting['N']
		M = setting['M']
		L = setting['L']
		parameters = setting['parameters']

		method = options['method'](setting,options)
		attribute = options['attribute'](setting,options)

		path = options['path'](setting,options)
		plots = options['plot'](setting,options)
		logger = options['logger'](setting,options)

		args,kwargs = tuple((-32,0,1000,)),dict(endpoint=False)

		d = D**N
		s = M+1
		w = 1

		l = L if (isinstance(L,int) and (L>=0)) else d+L if (isinstance(L,int) and (L<0)) else int(L*d) if isinstance(L,float) else int(L[0]*d/L[-1]) if isinstance(L,tuple) else 1
		# z = {2.3433e-2:3,5.4553e-2:3,7.8291e-2:1}; z = {**z,**{1.2954e-2:d-sum(z[i] for i in z)}}; z = {i:z[i] for i in z if z[i]>0}
		z = {1:l}; z = {**z,**{0:d-sum(z[i] for i in z)}}; z = {i:z[i] for i in z if z[i]>0}
		# z = {(i+1)/d:1 for i in range(d)}; z = {**z,**{0:d-sum(z[i] for i in z)}}; z = {i:z[i] for i in z if z[i]>0}

		z = array([j for i in z for j in [i]*z[i]])

		logger(setting)

		parameters = []

		try:

			if attribute['method']() in ['func']:

				func = parameterize

			elif attribute['method']() in ['functional']:

				func = parameterization

			elif attribute['method']() in ['_func']:

				func = _parameterize

			else:

				func = None

			params = func(z,d=d,s=s,w=w) if func is not None else None

			parameters.append(params)

		except Exception as exception:

			logger('Exception:\n%r\n%r'%(exception,traceback.format_exc()))

		if vectorize:

			if attribute['method']() in ['func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

				x = logspace(*args,**kwargs)

				func = {'pdf':functions,'cdf':Functions}[method]

				plts = dict(
					label='$\\textrm{Conjecture}~:~{%s}$'%(setting['M']),
					color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
					marker='',
					linestyle='-',
					alpha=0.5,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

			elif attribute['method']() in ['functional']:

				size = max(len(i) for params in parameters for i in params)
				parameters = [[[*i,*[0]*(size-len(i))] for i in params] for params in parameters]
				parameters = array(parameters)

				x = logspace(*args,**kwargs)

				func = {'pdf':functionals,'cdf':Functionals}[method]

				plts = dict(
					label='$\\textrm{Analytical}~:~{%s}$'%(setting['M']),
					color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
					marker='',
					linestyle='--',
					alpha=0.8,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

			elif attribute['method']() in ['_func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

				x = logspace(*args,**kwargs)

				func = {'pdf':_functions,'cdf':_Functions}[method]

				plts = dict(
					label='$\\textrm{Symmetry}~:~{%s}$'%(setting['M']),
					color='black',
					marker='o',
					linestyle='-',
					markersize=45,
					alpha=0.25,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

			else:

				parameters = None
				x = None
				func = None
				plts = None

		else:

			if attribute['method']() in ['func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

				x = logspace(*args,**kwargs)

				func = {'pdf':functions,'cdf':Functions}[method]

				plts = dict(
					label='$\\textrm{Conjecture}~:~{%s}$'%(setting['M']),
					color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
					marker='',
					linestyle='-',
					alpha=0.5,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

			elif attribute['method']() in ['functional']:

				from mpmath import linspace

				size = max(len(i) for params in parameters for i in params)
				parameters = [params for params in parameters]
				parameters = parameters

				x = exponentiate(linspace(*args,**kwargs))

				func = {'pdf':functionals,'cdf':Functionals}[method]

				plts = dict(
					label='$\\textrm{Analytical}~:~{%s}$'%(setting['M']),
					color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
					marker='',
					linestyle='--',
					alpha=0.8,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

			elif attribute['method']() in ['_func']:

				size = max(len(params) for params in parameters)
				parameters = [params for params in parameters]
				parameters = array(parameters)

				x = logspace(*args,**kwargs)

				func = {'pdf':_functions,'cdf':_Functions}[method]

				plts = dict(
					label='$\\textrm{Symmetry}~:~{%s}$'%(setting['M']),
					color='black',
					marker='o',
					linestyle='-',
					markersize=45,
					alpha=0.25,
					path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
					)

			else:

				parameters = None
				x = None
				func = None
				plts = None

		if attribute['method']() not in ['distribution']:

			y = func(parameters,x)

		else:

			plts = dict(
				# label='$\\textrm{Distribution}~:~{%s}$'%(setting['M']),
				label='$%s~,~%s~,~%s$'%(
					setting['M'],
					'%s'%(str(L)) if (isinstance(L,int) and (L>=0)) else 'd%s'%(str(L)) if (isinstance(L,int) and (L<0)) else '%sd'%(str(L)) if isinstance(L,float) else '%sd/%s'%(str(L[0]) if L[0]!=1 else '',str(L[-1])) if isinstance(L,tuple) else str(1),
					setting['parameters'],
					),
				color='%s_%f'%('viridis' if setting['L'] == 1 else 'magma',(settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
				marker='' if setting['parameters']==0 else '',
				markersize=None if setting['parameters']==0 else 10,
				linestyle='-' if (setting['parameters']==0) else '--' if (isinstance(setting['L'],int)) else '--',
				linewidth=20 if (isinstance(setting['L'],int)) else 16,
				alpha=0.8,
				path=join(path,'plot','plot.distribution.%s.L.1.d'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
				n=prod(len(settings[i]) for i in settings if i not in ['N']),
				)

			def func(x,settings):

				D = setting['D']
				N = setting['N']
				M = setting['M']
				L = setting['L']
				parameters = setting['parameters']

				d = D**N
				s = M+1
				l = L if (isinstance(L,int) and (L>=0)) else d+L if (isinstance(L,int) and (L<0)) else int(L*d) if isinstance(L,float) else int(L[0]*d/L[-1]) if isinstance(L,tuple) else 1
				parameters = 1 - ((1-((10**parameters) if (parameters != 0) else 0))**M)
				constant = 1

				u,v = 0,constant
				a,b = 1-parameters,parameters*constant/d

				options = dict(
					function='beta.pdf',
					a=l*s,
					b=(d-l)*s,
					loc=a*u+b,
					scale=a*(v-u)
					)

				y = distribution(x,**options)

				# y = where((x>(a*u+b))*(x<(a*v+b)),y,0)

				return y

			x = logspace(*args,**kwargs)

			y = func(x,setting)

			fig,ax = (None,None) if all(settings[i].index(setting[i]) == 0 for i in ['M','L','parameters']) else (fig,ax)

		fig,ax = plot(x,y,fig=fig,ax=ax,options={**plots,**plts})

	return


def plot(settings,options,*args,**kwargs):

	def setup(path=None,**kwargs):

		ext = 'hdf5'

		default = join('data','stats',root=path,ext=ext)
		opts = dict(wrapper='pd')

		path = join('data','data',root=path,ext=ext)

		if exists(default):
			path = default
			data = load(path,**opts)
			return data

		def wrapper(data):
			keys = {'none':None}
			for key in keys:
				for attr in data:
					if all(isinstance(i,str) for i in data[attr]):
						data[attr][data[attr]==key] = keys[key]
			return data

		data = load(path,wrapper=['df',wrapper],iterates=['x','y','parameters'])

		if data is None:
			return data

		attrs = [attr for attr in data]

		def boolean(data):
			keys = {'function':['array']}
			boolean = True
			for key in keys:
				boolean = boolean & data[key].isin(keys[key])
			return boolean

		by = ['N','M','noise.parameters','function','operator','sample']

		options = dict(as_index=False,dropna=False)

		data = data[boolean(data)].groupby(by=by,**options)

		def func(data):
			if data.dtype in ['object'] and all(isinstance(i,tuple) for i in data):
				data = tuple(array([[*i] for i in data]).mean(axis=0))
			elif data.dtype in ['object'] and all(isinstance(i,str) for i in data):
				data = data.iloc[0]
			else:
				data = data.iloc[0]
			return data

		agg = {attr:func for attr in attrs}
		options = dict()

		data = data.agg(agg,**options)

		path = default
		dump(data,path,**opts)

		data = load(path,**opts)

		return data

	def plot(options,fig=None,ax=None,index=None):

		def setter(obj,key,value):
			def func(obj,key,value):
				value = copy(value)
				options = []
				if key in ['plot']:
					options = ['x','y']
				elif key in ['errorbar']:
					options = ['x','y','yerr','xerr']
				elif key in ['set_xscale','set_yscale']:
					if value.get('value') in ['linear']:
						value.pop('base');

				args = [value.get(i) for i in options]
				kwargs = {option:value[option] for option in value if option not in options}

				for option in kwargs:
					if option in ['color','ecolor']:
						try:
							kwargs[option] = plt.get_cmap(kwargs[option].split(separ)[0])(float(kwargs[option].split(separ)[1]))
						except:
							if isinstance(kwargs[option],(str,list,tuple,*arrays)):
								kwargs[option] = kwargs[option]
							else:
								kwargs[option] = None

						keyword = 'alpha'
						if kwargs.get(keyword) is not None:
							if isinstance(kwargs[option],(list,tuple)):
								kwargs[option] = list(kwargs[option])
								kwargs[option][-1] = kwargs[keyword]
							elif isinstance(kwargs[option],arrays):
								kwargs[option][:,-1] = kwargs[keyword]

				return args,kwargs

			if isinstance(value,dict):
				args,kwargs = func(obj,key,value)
				getattr(obj,key)(*args,**kwargs)
			elif isinstance(value,list):
				for val in value:
					args,kwargs = func(obj,key,val)
					getattr(obj,key)(*args,**kwargs)
			return

		options = copy(options)

		with matplotlib.style.context(options['options'].get('mplstyle')) if options['options'].get('mplstyle') else context(options['options'].get('mplstyle')):

			figs,axes = plt.subplots(*options['options']['layout']) if ((fig is None) or (ax is None)) else (fig,ax)

			fig = figs
			ax = axes.flatten()[index%axes.size] if (isinstance(axes,arrays) and index is not None) else axes


			attr = 'ax'
			obj = ax

			if isinstance(options.get(attr),dict) and len(options.get(attr)):
				for key in options[attr]:
					setter(obj,key,options[attr][key])


			attr = 'colorbar'

			if isinstance(options.get(attr),dict) and len(options.get(attr)):

				try:
					options[attr]['cmap']['colors'] = [plt.get_cmap(i.split(separ)[0])(float(i.split(separ)[1])) for i in options[attr]['cmap']['colors']]
				except:
					options[attr]['cmap']['colors'] = [i for i in options[attr]['cmap']['colors']]

				cmap = matplotlib.colors.LinearSegmentedColormap.from_list(**options[attr]['cmap'])

				cax,opts = matplotlib.colorbar.make_axes([ax for ax in fig.axes],**options[attr]['cax'])

				cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,**options[attr]['cbar'])

				obj = cbar.ax
				for key in options[attr]['ax']:
					setter(obj,key,options[attr]['ax'][key])


			attr = 'legend'
			obj = ax

			if isinstance(options.get(attr),dict) and len(options.get(attr)):

				handles,labels = obj.get_legend_handles_labels()
				handles,labels = [copy(handle) for handle in handles],[copy(label) for label in labels]
				keywords = {keyword:options[attr].pop(keyword) for keyword in ['set_linewidth','set_alpha','set_color'] if keyword in options[attr]}
				for i,(handle,label) in enumerate(zip(handles,labels)):
					for keyword in keywords:
						setter(handle[0],keyword,keywords[keyword])

				indices = range(min(len(handles),len(labels)))
				handles,labels = [handles[i] for i in indices],[labels[i] for i in indices]

				legend = obj.legend(handles,labels,**options[attr])


			attr = 'fig'
			obj = fig

			if isinstance(options.get(attr),dict) and len(options.get(attr)):
				for key in options[attr]:
					setter(obj,key,options[attr][key])

		fig = figs
		ax = axes

		return fig,ax

	def process(path):

		data = setup(path)

		if data is None:
			return

		settings = [
			Dict(
				variables = dict(
					x = 'M',
					y = 'parameters',
					colorbar = 'noise.parameters',
					legend = 'sample',
					sort = ['N','operator']
					),
				data = ['noise','env'],
				boolean = lambda data: data['M'].isin([2,4,16,32]) & data['N'].isin([10]),
				options = dict(
					groupby=dict(as_index=False,dropna=False)
					),
				fig = {},
				ax = {},
				index = lambda index=None,group=None,groupby=None,**kwargs:None,
				),
			Dict(
				variables = dict(
					x = 'noise.parameters',
					y = 'parameters',
					colorbar = 'M',
					legend = 'sample',
					sort = ['N','operator']
					),
				data = ['noise','env'],
				boolean = lambda data: data['M'].isin([2,4,16,32]) & data['N'].isin([10]),
				options = dict(
					groupby=dict(as_index=False,dropna=False)
					),
				fig = {},
				ax = {},
				index = lambda number=None,group=None,groupby=None,**kwargs:None,
				),
			# Dict(
			# 	variables = dict(
			# 		x = 'M',
			# 		y = 'parameters',
			# 		colorbar = 'noise.parameters',
			# 		legend = 'N',
			# 		sort = ['sample','operator']
			# 		),
			# 	data = ['noise','env'],
			# 	boolean = lambda data: data['M'].isin([2,4,16,32]) & data['sample'].isin([1.0]),
			# 	options = dict(
			# 		groupby=dict(as_index=False,dropna=False)
			# 		),
			# 	fig = {},
			# 	ax = {},
			# 	index = lambda index=None,group=None,groupby=None,**kwargs:None,
			# 	),
			# Dict(
			# 	variables = dict(
			# 		x = 'noise.parameters',
			# 		y = 'parameters',
			# 		colorbar = 'M',
			# 		legend = 'N',
			# 		sort = ['sample','operator']
			# 		),
			# 	data = ['noise','env'],
			# 	boolean = lambda data: data['M'].isin([2,4,16,32]) & data['sample'].isin([1.0]),
			# 	options = dict(
			# 		groupby=dict(as_index=False,dropna=False)
			# 		),
			# 	fig = {},
			# 	ax = {},
			# 	index = lambda number=None,group=None,groupby=None,**kwargs:None,
			# 	),
			]

		for setting in settings:

			bys = [attr for attr in setting.variables.sort if attr in data]
			groupbys = data[setting.boolean(data)].groupby(by=bys,**setting.options.groupby)

			groupoids = groupbys[bys].agg('first')
			groupoids = {attr:groupoids[attr].unique().tolist() for attr in bys}

			for groupby in groupbys.groups:

				objs = groupbys.get_group(groupby)

				groupoid = dict(zip(bys,groupby))

				by = [attr for attr in [setting.variables.colorbar,setting.variables.legend] if attr in data]
				groups = objs.groupby(by=by,**setting.options.groupby)

				groupings = groups[by].agg('first')
				groupings = {attr: groupings[attr].unique().tolist() for attr in by}

				for group in groups.groups:

					obj = groups.get_group(group)

					grouping = dict(zip(by,group))

					y = array([list(i) if isinstance(i,iterables) else i for i in obj[setting.variables.y]]).T
					x = [array([i for i in obj[setting.variables.x]])]*len(y)

					for number,(x,y) in enumerate(zip(x,y)):

						key = (*groupby,number)

						options = {}

						options.update({

							'options':{
								'path':join(path,'plot'),
								'font':{'text':200,'legend':200,'colorbar':200},
								'color':{'plot':'viridis','colorbar':'viridis'},
								'layout':[],
								},

							})

						options.update({

							'options': {
								**options['options'],
								**{'mplstyle': join(options['options']['path'],'plot',ext='mplstyle'),}
								},

							'ax':{
								'errorbar': {
									'x':x,
									'y':y,

									**({
										(True,'sample'):{
											'label':'$%d$'%(int(128*obj[setting.variables.legend].iloc[0])),
											},
										(True,'N'):{
											'label':'$%d$'%(int(obj[setting.variables.legend].iloc[0])),
											},
										}.get((all((groupings[attr].index(grouping[attr])==(len(groupings[attr])-1)) for attr in [setting.variables.colorbar]),
											setting.variables.legend),{})
										),

									**([
										{
										'marker': 'o'
										},
										{
										'marker': '^'
										},
										{
										'marker': 's'
										},
										{
										'marker': 'P'
										},
										{
										'marker': 'h'
										},
										{
										'marker': 'x'
										},
										][groupings[setting.variables.legend].index(grouping[setting.variables.legend])]
										),

									'color':'%s_%s'%(
										options['options']['color']['plot'],
										(groupings[setting.variables.colorbar].index(grouping[setting.variables.colorbar]))/(len(groupings[setting.variables.colorbar])-1)),
									'markersize':60,
									'linestyle':'--',
									'linewidth':30,
									'alpha':0.5,
									},

								**({
									('M','noise'):
										{
										'set_title':None,
										'set_xlabel':{'xlabel':'$\\textrm{Depth}~~k$','size':options['options']['font']['text']},
										'set_ylabel':{'ylabel':'$\\textrm{Effective Noise}~~\\tilde{\\gamma}$','size':options['options']['font']['text']},

										'set_xscale':{'value':'linear','base':10},
										'set_xlim':{'xmin':-2,'xmax':34},
										'set_xticks':{'ticks':[0,2,4,8,16,32]},
										'set_xticklabels':{'labels':['$%d$'%(i) for i in [0,2,4,8,16,32]],'size':options['options']['font']['text']},

										'set_yscale':{'value':'log','base':10},
										'set_ylim':{'ymin':5e-5,'ymax':2e0},
										'set_yticks':{'ticks':[1e-4,1e-3,1e-2,1e-1,1e0]},
										'set_yticklabels':{'labels':['$10^{%d}$'%(i) if i!=0 else '$1$' for i in [-4,-3,-2,-1,0]],'size':options['options']['font']['text']},
										},
									('M','env'):
										{
										'set_title':None,
										'set_xlabel':{'xlabel':'$\\textrm{Depth}~~k$','size':options['options']['font']['text']},
										'set_ylabel':{'ylabel':'$\\textrm{Effective Environment}~~\\tilde{s}$','size':options['options']['font']['text']},

										'set_xscale':{'value':'linear','base':10},
										'set_xlim':{'xmin':-2,'xmax':34},
										'set_xticks':{'ticks':[0,2,4,8,16,32]},
										'set_xticklabels':{'labels':['$%d$'%(i) for i in [0,2,4,8,16,32]],'size':options['options']['font']['text']},

										'set_yscale':{'value':'log','base':2},
										'set_ylim':{'ymin':0.5,'ymax':1500},
										'set_yticks':{'ticks':[1,4,16,64,256,1024]},
										'set_yticklabels':{'labels':['$2^{%d}$'%(i) if i!=0 else '$1$' for i in [0,2,4,6,8,10]],'size':options['options']['font']['text']},
										},
									('noise.parameters','noise'):
										{
										'set_title':None,
										'set_xlabel':{'xlabel':'$\\textrm{Noise}~~\\gamma$','size':options['options']['font']['text']},
										'set_ylabel':{'ylabel':'$\\textrm{Effective Noise}~~\\tilde{\\gamma}$','size':options['options']['font']['text']},

										'set_xscale':{'value':'log','base':10},
										'set_xlim':{'xmin':5e-5,'xmax':2e0},
										'set_xticks':{'ticks':[1e-4,1e-3,1e-2,1e-1,1e0]},
										'set_xticklabels':{'labels':['$10^{%d}$'%(i) if i!=0 else '$1$' for i in [-4,-3,-2,-1,0]],'size':options['options']['font']['text']},

										'set_yscale':{'value':'log','base':10},
										'set_ylim':{'ymin':5e-5,'ymax':2e0},
										'set_yticks':{'ticks':[1e-4,1e-3,1e-2,1e-1,1e0]},
										'set_yticklabels':{'labels':['$10^{%d}$'%(i) if i!=0 else '$1$' for i in [-4,-3,-2,-1,0]],'size':options['options']['font']['text']},
										},
									('noise.parameters','env'):
										{
										'set_title':None,
										'set_xlabel':{'xlabel':'$\\textrm{Noise}~~\\gamma$','size':options['options']['font']['text']},
										'set_ylabel':{'ylabel':'$\\textrm{Effective Environment}~~\\tilde{s}$','size':options['options']['font']['text']},

										'set_xscale':{'value':'log','base':10},
										'set_xlim':{'xmin':5e-5,'xmax':2e0},
										'set_xticks':{'ticks':[1e-4,1e-3,1e-2,1e-1,1e0]},
										'set_xticklabels':{'labels':['$10^{%d}$'%(i) if i!=0 else '$1$' for i in [-4,-3,-2,-1,0]],'size':options['options']['font']['text']},

										'set_yscale':{'value':'log','base':2},
										'set_ylim':{'ymin':0.5,'ymax':1500},
										'set_yticks':{'ticks':[1,4,16,64,256,1024]},
										'set_yticklabels':{'labels':['$2^{%d}$'%(i) if i!=0 else '$1$' for i in [0,2,4,6,8,10]],'size':options['options']['font']['text']},
										},
									}.get((setting.variables.x,setting.data[number]),{})
									),

								'tick_params':[
									{'axis':'y','which':'major','length':6,'width':1,'size':15,'pad':30},
									{'axis':'y','which':'minor','length':4,'width':0},
									{'axis':'x','which':'major','length':6,'width':1,'size':15,'pad':50},
									{'axis':'x','which':'minor','length':4,'width':0}
									],

								'grid':{'visible':True},

								},

							'colorbar': {
								**({
								True:{
									'cmap': {
										'name':None,
										'colors':[separ.join([options['options']['color']['colorbar'],str((i)/(len(groupings[setting.variables.colorbar])-1))])
											for i in range(len(groupings[setting.variables.colorbar]))],
										'N':100*len(groupings[setting.variables.colorbar]),
										},
									'cax': {'location':'right','fraction':0.15,'shrink':1,'aspect':40,'pad':0.02,'anchor':(1.25,0.5)},
									'cbar': {'orientation':'vertical'},
									**({
										'noise.parameters':{
											'ax': {
												'set_ylabel':{'ylabel':'$\\textrm{Noise}~~\\gamma$','size':options['options']['font']['colorbar']},
												'set_yticks':{'ticks':[(i)/(len(groupings[setting.variables.colorbar])-1) for i in range(len(groupings[setting.variables.colorbar]))]},
												'set_yticklabels':{'labels':['$%s$'%(scinotation(i,scilimits=[0,0],one=False,zero=True,integral=True,usetex=False)) for i in groupings[setting.variables.colorbar]],'size':options['options']['font']['colorbar']},
												'tick_params':[
													{'axis':'y','which':'major','length':15,'width':1,'size':15,'pad':30},
													],
												},
											},
										'M':{
											'ax': {
												'set_ylabel':{'ylabel':'$\\textrm{Depth}~~k$','size':options['options']['font']['colorbar']},
												'set_yticks':{'ticks':[(i)/(len(groupings[setting.variables.colorbar])-1) for i in range(len(groupings[setting.variables.colorbar]))]},
												'set_yticklabels':{'labels':['$%s$'%(scinotation(i,scilimits=[0,3],one=True,zero=True,integral=True,usetex=False)) for i in groupings[setting.variables.colorbar]],'size':options['options']['font']['colorbar']},
												'tick_params':[
													{'axis':'y','which':'major','length':15,'width':1,'size':15,'pad':30},
													],
												},
											},
										}.get(setting.variables.colorbar,{})
									),
									},
								}.get(all((groupings[attr].index(grouping[attr])==(len(groupings[attr])-1)) for attr in grouping),{})
								),
								},
							'legend': {
								**({
								(True,'sample'):{
									'title': '$\\textrm{Samples}~~m$',
									'loc': {'noise':'lower right','env':'upper left'}.get(setting.data[number],'upper right'),
									'ncol':1,
									'title_fontsize': options['options']['font']['legend'],
									'prop':{'size': options['options']['font']['legend'],},
									'markerscale':1.25,
									'handlelength':3,
									'framealpha':1,
									'set_alpha':{'alpha':0.5},
									'set_color':{'color':'gray'},
									'set_linewidth':{'w':16},
									},
								(True,'N'):{
									'title': '$\\textrm{Size}~~n$',
									'loc':'upper right',
									'ncol':1,
									'title_fontsize': options['options']['font']['legend'],
									'prop':{'size': options['options']['font']['legend'],},
									'markerscale':1.25,
									'handlelength':3,
									'framealpha':1,
									'set_alpha':{'alpha':0.5},
									'set_color':{'color':'gray'},
									'set_linewidth':{'w':16},
									},
								}.get((all((groupings[attr].index(grouping[attr])==(len(groupings[attr])-1)) for attr in grouping),setting.variables.legend),{})
								),
								},

							'fig':{
								'set_size_inches':{'w':65,'h':45},
								'subplots_adjust':{},
								'tight_layout':{},
								'savefig':{
									'fname':join(
										options['options']['path'],
										delim.join([
											'plot',
											setting.variables.x,
											setting.variables.y,
											setting.data[number],
											setting.variables.colorbar,
											setting.variables.legend,
											delim.join([str(i) for attr in groupoid for i in [attr,groupoid[attr]]])
											]),
										ext='pdf'),
									'bbox_inches':'tight',
									'pad_inches':0.5
									},
							}
						})

						fig,ax = setting.fig.get(key),setting.ax.get(key)
						index = setting.index(number,group,groupby)

						fig,ax = plot(options,fig=fig,ax=ax,index=index)

						setting.fig[key] = fig
						setting.ax[key] = ax

						if all((groupings[attr].index(grouping[attr])==(len(groupings[attr])-1)) for attr in grouping):
							logger(options['fig']['savefig']['fname'])

		return

	path = sys.argv[1] if len(sys.argv[1:]) else None

	logger = Logger(file=None,verbose='info')

	if path is None:
		return

	process(path)

	return


def setup(settings,options,*args,**kwargs):

	boolean = options['boolean'](settings,options)
	logger = options['logger'](settings,options)

	logger(boolean)

	if boolean.get('run'):

		run(settings,options,*args,**kwargs)

	if boolean.get('process'):

		process(settings,options,*args,**kwargs)

	if boolean.get('test'):

		test(settings,options,*args,**kwargs)

	if boolean.get('draw'):

		draw(settings,options,*args,**kwargs)

	if boolean.get('plot'):

		plot(settings,options,*args,**kwargs)

	return

def main(*args,**kwargs):

	attribute = {}

	def func(settings,options,*args,**kwargs):
		attr = settings['attr']
		obj = {
			**{attr:'func' for attr in ['tetrad','pauli']},
			**{attr:'functional' for attr in ['test.tetrad','test.pauli']},
			**{attr:'_func' for attr in ['check.tetrad','check.pauli']},
			**{attr:'distribution' for attr in ['distribution']},
			}[attr]
		return obj
	attribute['method'] = func

	def func(settings,options,*args,**kwargs):
		attr = settings['attr'].split(delim)[-1]
		obj = real(eig(getattr(basis,attr)(D=settings['D'])))
		return obj
	attribute['data'] = func

	def func(settings,options,data,*args,**kwargs):
		parameters = settings['parameters']
		# parameters = 1 - ((1-parameters)**(settings['M']))
		obj = (1-parameters)*data + parameters*addition(data)/data.size
		return obj
	attribute['func'] = func


	keywords = dict(attribute=attribute)


	settings = dict(
		# attr=['pauli'],
		# attr=['tetrad'],
		# attr=['tetrad','pauli'],
		# attr=['test.pauli','test.tetrad'],
		attr=['test.tetrad','tetrad'],
		D=[2],
		# N=[2,3,4,5,6,7,8],
		# M=[0,2,4,8,16,32],
		# N=[3],
		# M=[0,2,4,8,16,32],
		N=[3],
		# M=[0,2,4,8,16,32],
		# M=[0,2,4,8,16,32],
		M=[0],
		L=[1],
		parameters=[0],
		)

	options = dict(
		boolean = (lambda settings={},options={},keywords=keywords: {
			'run':0,
			'process':0,
			'test':0,
			'draw':0,
			'plot':1,
			}),
		path   = (lambda settings={},options={},keywords=keywords: '~/scratch/probability/distribution'),
		io     = (lambda settings={},options={},keywords=keywords: dict(wr='a',default={})),
		do     = (lambda settings={},options={},keywords=keywords: True or (not exists(options['data'](settings,options))) or (load(options['data'](settings,options),**options['io'](settings,options)) is None) or (options['key'](settings,options) not in load(options['data'](settings,options),**options['io'](settings,options)))),
		key    = (lambda settings={},options={},keywords=keywords: 'operator.{attr}.N.{N}.M.{M}'.format(**settings)),
		attrs  = (lambda settings={},options={},keywords=keywords: ('attr','N')),
		method  = (lambda settings={},options={},keywords=keywords: 'pdf'),
		attribute = (lambda settings={},options={},keywords=keywords:{attr:partial(keywords['attribute'][attr],settings=settings,options=options) for attr in keywords['attribute']}),
		data   = (lambda settings={},options={},keywords=keywords: join(options['path'](settings,options),'data','test',ext='hdf5' if vectorize else 'pkl')),
		logger = (lambda settings={},options={},keywords=keywords: Logger(file=join(options['path'](settings,options),'log','log.log'),verbose='info')),
		plot   =  (lambda settings={},options={},keywords=keywords: dict(
			path=join(options['path'](settings,options),'plot','plot.test.%s'%('.'.join([str(i) if not isinstance(i,str) else i.split(delim)[-1] for attr in options['attrs'](settings,options) for i in [attr,settings[attr]]])),ext='pdf'),
			mplstyle=join(options['path'](settings,options),'plot','plot.mplstyle'),
			markersize=9,
			linewidth=16,
			alpha=0.8,
				)
			)
		)

	args = ()

	kwargs = {}

	setup(settings,options,*args,**kwargs)

	return

if __name__ == '__main__':

	args = ()

	kwargs = {}

	main(*args,**kwargs)
