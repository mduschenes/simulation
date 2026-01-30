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
from src.utils import pi,nan,fltmin,fltmax,delim,epsilon,iterables
from src.utils import where,real,imag,nonzero,unique,sort,minimum,maximum,minimums,maximums
from src.utils import eig,addition,prod,permutations,partitions,multinomial,permute,distribution
from src.utils import integral

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

	z,(d,l,u,v,w) = parameters[:-5],parameters[-5:]

	z,d,l,u,v,w = z.astype(float),int(d),int(l),float(u),float(v),float(w)

	x = (x-u)/(v-u)

	t = d-l-1

	if t>0:
		G = list(SymmetricGroup(t).generate_schreier_sims())
		y = 0
		for g in G:
			k = [len(p) for p in g.full_cyclic_form]
			k = {p:k.count(p) for p in set(k)}
			y += prod(trace(x,z,p)**k[p] for p in k)
	else:
		y = exp(gammaln(l+t)-gammaln(l))

	y *= sign(w)*exp(log(abs(w))-log(v-u)+(gammaln(d)-gammaln(l)-gammaln(d-l))-(gammaln(l+t)-gammaln(l))-sum(log(z))+(l-1)*log(x))

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

	z = (z-u)/(v-u)

	z = z[(z>eps)*(z<=1)]

	l,s,d = z.size,s,d
	w = w

	params = [*z,d,l,u,v,w]

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

		ax.set_xlabel(xlabel="$p$",size=60)
		ax.set_ylabel(ylabel="$P(p)$",size=60)

		# ax.set_xscale(value="log",base=10)
		# ax.set_yscale(value="log",base=10)
		# ax.set_xlim(xmin=1e-22,xmax=1e2)
		# ax.set_ylim(ymin=1e-21,ymax=1e21)
		# ax.set_xticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1])
		# ax.set_xticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-20,-16,-12,-8,-4,0]],size=60)
		# ax.set_yticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16,1e20])
		# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-20,-16,-12,-8,-4,0,4,8,12,16,20]],size=60)

		# ax.set_xscale(value="log",base=4)
		# ax.set_yscale(value="log",base=10)
		# ax.set_xlim(xmin=2**(-11),xmax=2**(2))
		# ax.set_ylim(ymin=1e-129,ymax=1e9)
		# ax.set_xticks(ticks=[2**(-2*i) for i in [10,8,6,4,2,0]])
		# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [10,8,6,4,2,0]],size=60)
		# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
		# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-128,-64,-32,-16,-12,-8,-4,-0,4,8]],size=60)

		ax.set_xscale(value="log",base=4)
		ax.set_yscale(value="log",base=10)
		ax.set_xlim(xmin=2**(-2*17),xmax=2**(2))
		ax.set_ylim(ymin=1e-129,ymax=1e129)
		ax.set_xticks(ticks=[2**(-2*i) for i in [16,14,12,10,8,6,4,2,0]])
		ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [16,14,12,10,8,6,4,2,0]],size=60)
		ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16,1e32,1e64,1e128])
		ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-128,-64,-32,-16,-12,-8,-4,0,4,8,12,16,32,64,128]],size=60)

		ax.tick_params(**{"axis":"y","which":"major","length":6,"width":1,"pad":10})
		ax.tick_params(**{"axis":"y","which":"minor","length":4,"width":0})
		ax.tick_params(**{"axis":"x","which":"major","length":6,"width":1,"pad":10})
		ax.tick_params(**{"axis":"x","which":"minor","length":4,"width":0})

		ax.grid(visible=True)

		handles,labels = ax.get_legend_handles_labels()
		handles,labels = [copy(handle) for handle in handles],[copy(label) for label in labels]
		for handle,label in zip(handles,labels):
			handle[0].set_linewidth(12)
		indices = [labels.index(label) for label in natsorted(labels)]
		handles,labels = [handles[i] for i in indices],[labels[i] for i in indices]

		legend = ax.legend(
			handles,labels,
			title="$k$",
			loc="upper right",
			ncol=3,
			title_fontsize=50,
			prop={"size":50},
			markerscale=6,
			handlelength=2.5
		)

		if settings.get('path'):
			fig.set_size_inches(w=48,h=48)
			fig.subplots_adjust()
			fig.tight_layout()
			fig.savefig(fname=settings.get('path'))

	return fig,ax

def process(settings,options,*args,**kwargs):

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
			l = 1
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

			ax.set_xlabel(xlabel="$x$",size=60)
			ax.set_ylabel(ylabel="$f(x)$",size=60)

			ax.set_xscale(value="log",base=10)
			ax.set_yscale(value="log",base=10)
			ax.set_xlim(xmin=1e-5,xmax=1e1)
			ax.set_ylim(ymin=1e-17,ymax=1e17)
			ax.set_xticks(ticks=[1e-4,1e-3,1e-2,1e-1,1])
			ax.set_xticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-4,-3,-2,-1,0]],size=60)
			ax.set_yticks(ticks=[1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16])
			ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-16,-12,-8,-4,0,4,8,12,16]],size=60)

			# ax.set_xscale(value="log",base=4)
			# ax.set_yscale(value="log",base=10)
			# ax.set_xlim(xmin=2**(-11),xmax=2**(2))
			# ax.set_ylim(ymin=1e-129,ymax=1e9)
			# ax.set_xticks(ticks=[2**(-2*i) for i in [10,8,6,4,2,0]])
			# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [10,8,6,4,2,0]],size=60)
			# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-128,-64,-32,-16,-12,-8,-4,0,4,8]],size=60)

			# ax.set_xscale(value="log",base=4)
			# ax.set_yscale(value="log",base=10)
			# ax.set_xlim(xmin=2**(-2*17),xmax=2**(2))
			# ax.set_ylim(ymin=1e-129,ymax=1e9)
			# ax.set_xticks(ticks=[2**(-2*i) for i in [16,14,12,10,8,6,4,2,0]])
			# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [16,14,12,10,8,6,4,2,0]],size=60)
			# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i not in [0] else '$1$' for i in [-128,-64,-32,-16,-12,-8,-4,0,4,8]],size=60)

			ax.tick_params(**{"axis":"y","which":"major","length":6,"width":1,"pad":10})
			ax.tick_params(**{"axis":"y","which":"minor","length":4,"width":0})
			ax.tick_params(**{"axis":"x","which":"major","length":6,"width":1,"pad":10})
			ax.tick_params(**{"axis":"x","which":"minor","length":4,"width":0})

			ax.grid(visible=True)

			handles,labels = ax.get_legend_handles_labels()
			handles,labels = [copy(handle) for handle in handles],[copy(label) for label in labels]
			for handle,label in zip(handles,labels):
				handle[0].set_linewidth(12)
			indices = [labels.index(label) for label in natsorted(labels)]
			handles,labels = [handles[i] for i in indices],[labels[i] for i in indices]

			legend = ax.legend(
				handles,labels,
				title="$k$",
				loc="upper right",
				ncol=2,
				title_fontsize=50,
				prop={"size":50},
				markerscale=6,
				handlelength=2.5
			)

			if settings.get('path'):
				fig.set_size_inches(w=48,h=30)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname=settings.get('path'))

		return fig,ax


	settings = dict(
		attr=['test.pauli','check.pauli','pauli'],
		D=[2],
		N=[3],
		M=[0],
		parameters=[0],
		)

	fig,ax = None,None

	for index,setting in enumerate(permute(settings)):

		attr = setting['attr']
		D = setting['D']
		N = setting['N']
		M = setting['M']

		method = options['method'](setting,options)
		attribute = options['attribute'](setting,options)

		path = options['path'](setting,options)
		plots = options['plot'](setting,options)
		logger = options['logger'](setting,options)

		args,kwargs = tuple((-32,0,1000,)),dict(endpoint=False)

		d = D**N
		s = M+1
		w = 1

		l = d//2
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

		y = func(parameters,x)

		fig,ax = plot(x,y,fig=fig,ax=ax,options={**plots,**plts})

	plts = dict(
		label='$\\textrm{Distribution}~:~{%s}$'%(setting['M']),
		color='viridis_%f'%((settings['M'].index(setting['M']))/(len(settings['M'])+1)),
		marker='',
		linestyle=':',
		alpha=1,
		path=join(path,'plot','plot.test.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
		)

	opts = dict(function=f'beta.{method}',a=l*s,b=(d-l)*s)

	x = logspace(*args,**kwargs)

	y = distribution(x,**opts)

	fig,ax = plot(x,y,fig=fig,ax=ax,options={**plots,**plts})

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

	return

def main(*args,**kwargs):

	attribute = {}

	def func(settings,options,*args,**kwargs):
		attr = settings['attr']
		obj = {
			**{attr:'func' for attr in ['tetrad','pauli']},
			**{attr:'functional' for attr in ['test.tetrad','test.pauli']},
			**{attr:'_func' for attr in ['check.tetrad','check.pauli']},
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
		parameters = 1 - ((1-parameters)**(settings['M']))
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
		N=[2],
		# M=[0,2,4,8,16,32],
		# M=[0,2,4,8,16,32],
		M=[0,2,4],
		parameters=[1e-3],
		)

	options = dict(
		boolean = (lambda settings={},options={},keywords=keywords: {
			'run':0,
			'process':0,
			'test':1,
			}),
		path   = (lambda settings={},options={},keywords=keywords: '~/scratch/probability/distribution'),
		io     = (lambda settings={},options={},keywords=keywords: dict(wr='a',default={})),
		do     = (lambda settings={},options={},keywords=keywords: True or (not exists(options['data'](settings,options))) or (load(options['data'](settings,options),**options['io'](settings,options)) is None) or (options['key'](settings,options) not in load(options['data'](settings,options),**options['io'](settings,options)))),
		key    = (lambda settings={},options={},keywords=keywords: 'operator.{attr}.N.{N}.M.{M}'.format(**settings)),
		attrs  = (lambda settings={},options={},keywords=keywords: ('attr','N')),
		method  = (lambda settings={},options={},keywords=keywords: 'pdf'),
		attribute = (lambda settings={},options={},keywords=keywords:{attr:partial(keywords['attribute'][attr],settings=settings,options=options) for attr in keywords['attribute']}),
		data   = (lambda settings={},options={},keywords=keywords: join(options['path'](settings,options),'data','data',ext='hdf5' if vectorize else 'pkl')),
		logger = (lambda settings={},options={},keywords=keywords: Logger(file=join(options['path'](settings,options),'log','log.log'),verbose='info')),
		plot   =  (lambda settings={},options={},keywords=keywords: dict(
			path=join(options['path'](settings,options),'plot','plot.distribution.%s'%('.'.join([str(i) if not isinstance(i,str) else i.split(delim)[-1] for attr in options['attrs'](settings,options) for i in [attr,settings[attr]]])),ext='pdf'),
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
