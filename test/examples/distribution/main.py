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

from src.utils import array,rand,asscalar,tensorprod,concatenate,meshgrid,linspace,logspace,inplace,partial,cache,scan,vmap,callback,vectorize,allclose,vtype,copy,padding
from src.utils import exp,log,log1p,sign,gammaln
from src.utils import nan,fltmin,fltmax,delim,epsilon,iterables
from src.utils import where,real,imag,nonzero,unique,sort,minimum,maximum,minimums,maximums
from src.utils import eig,addition,prod,permutations,partitions,multinomial,permute,distribution
from src.utils import integral as integrate

from src.quantum import Basis as basis

from src.io import load,dump,exists,join,split

from src.logger import Logger

from mpmath import quad as integral,linspace as linearspace,mpmathify,workdps,sqrt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects

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
		return z*integrate(lambda x,z=z:function(parameters,z*x),bounds)
	y = func(x)
	return y

def Functions(parameters,x):
	bounds = logspace(-20,0,50)
	@vmap
	def func(z):
		return z*integrate(lambda x,z=z:functions(parameters,z*x),bounds)
	y = func(x)
	return y

def parameter(z,d=None,s=None,w=None):

	from mpmath import exp,log,log1p
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

	eps = 1e-12
	bounds = list(map(lambda i:10**i,linearspace(-32,0,1000,endpoint=False)))

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

	params.update(dict(p=p))
	f = partial(func,parameters=[params[i] for i in params])
	q = integral(f,bounds)

	params.update({i:j for i,j in dict(p=p,u=u,v=v,w=w).items()})

	parameters = [params[i] for i in params]

	parameters = [float(max(min(params,fltmax),fltmin)) for params in parameters]

	parameters = array(parameters)

	return parameters


def functional(parameters,x):

	z,l,w = parameters[0].astype(float),parameters[1].astype(int),parameters[2].astype(float)

	i = (l>0)
	z,l = z[i],l[i]
	w = w

	n,d,u,v = z.size,addition(l),minimum(z),maximum(z)

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

	y = where((x>u)*(x<=v),y,0)

	return y

def functional(parameters,x):

	z,l,w = parameters[0].astype(float),parameters[1].astype(int),parameters[2].astype(float)

	i = (l>0)
	z,l = z[i],l[i]
	w = w

	n,d,u,v = z.size,addition(l),minimum(z),maximum(z)

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

	y = where((x>u)*(x<=v),y,0)

	return y

def functionals(parameters,x):
	func = lambda y,parameters: y+functional(parameters,x)
	y = 0*x
	for params in parameters:
		y = func(y,params)
	return y

def Functional(parameters,x):

	z,l,w = parameters[0].astype(float),parameters[1].astype(int),parameters[2].astype(float)

	i = (l>0)
	z,l = z[i],l[i]
	w = w

	n,d,u,v = z.size,addition(l),minimum(z),maximum(z)

	y = sum(
		(
		sign(w[sum(l[:i])+t])
		*
		(
		(-(sign(z[i]-x)**((d-l[i]+t+1)%2))*exp((d-l[i]+t)*log(abs(z[i]-x)) + log(abs(w[sum(l[:i])+t])) - log(d-l[i]+t)))
		+
		( (sign(z[i]-u)**((d-l[i]+t)%2))*exp((d-l[i]+t)*log(abs(z[i]-u)) + log(abs(w[sum(l[:i])+t])) - log(d-l[i]+t)))
		)
		)
		for i in range(n)
		for t in range(l[i])
		)

	y = where((x>=v),1,where((x<=u),0,y))

	return y

def Functionals(parameters,x):
	func = lambda y,parameters: y+Functional(parameters,x)
	y = 0*x
	for params in parameters:
		y = func(y,params)
	return y

def parameterizations(z,d=None,s=None,w=None):

	from src.utils import exp,log,log1p,sign,gammaln

	@cache
	def factorial(l,t):
		return gammaln(d)-gammaln(d-l+t)-gammaln(l-t)

	@cache
	def factorials(l,t):
		return gammaln(l+t)-gammaln(t+1)-gammaln(l)

	if z is None:
		parameters = None
		return parameters

	d = 1 if d is None else d
	s = 1 if s is None else s
	w = 1 if w is None else w

	eps = 1e-12

	z,l = unique(z,return_counts=True)

	i = (l>0)
	z,l = z[i],l[i]
	w = [w]*len(z) if not isinstance(w,iterables) else w

	l *= s

	n,d,u,v = z.size,addition(l),minimum(z),maximum(z)

	w = array([
		(
		(1/2)*((-1)**(t%2))*
		exp(log(w[i])+factorial(l[i],t))
		*
		sum(
		(
		exp(sum(factorials(l[j],p[j])-((l[j]+p[j])*log(abs(z[i]-z[j])))
			for j in range(n) if (j!=i)))
		*
		prod(sign(z[i]-z[j])**((l[j]+p[j])%2)
			for j in range(n) if (j!=i))
		)
		for p in permutations(t+1,repeat=n)
		if ((p[i]==0) and (sum(p)==t))
		)
		)
		for i in range(n)
		for t in range(l[i])
		])

	params = [z,l,w]

	parameters = [i for i in params]

	parameters = padding(parameters)

	parameters = array(parameters)

	return parameters

def run(settings,options,*args,**kwargs):

	for index,setting in enumerate(permute(settings)):

		attr = setting['attr']
		D = setting['D']
		N = setting['N']
		M = setting['M']

		path = options['data'](setting,options)
		key = options['key'](setting,options)
		io = options['io'](setting,options)

		attribute = options['attribute'](setting,options)
		logger = options['logger'](setting,options)

		do = options['do'](setting,options)

		if not do:
			continue

		logger(setting)

		data = attribute['data']()

		parameters = []

		for number,partition in enumerate(partitions(N,D**2)):

			try:

				z = tensorprod([obj for i,j in enumerate(partition) for obj in [data[i]]*j])

				z = where(z>epsilon(),z,0)
				d = D**N
				s = M+1
				w = multinomial(partition)/d

				if attribute['func']() in ['func']:

					params = parameter

				elif attribute['func']() in ['functional']:

					params = parameterizations

				params = params(z,d=d,s=s,w=w)

				parameters.append(params)

				logger(f'{number}'+'\t'+'\t'.join([f'{i}' for i in params]))

			except Exception as exception:

				logger('Exception:\n%r\n%r'%(exception,traceback.format_exc()))

		if attribute['func']() in ['func']:

			parameters = array(parameters)

		elif attribute['func']() in ['functional']:

			parameters = padding(parameters)

		data = {key:dict(parameters=parameters)}

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

		legend = ax.legend(
			handles,labels,
			title="$k$",
			loc="upper right",
			ncol=1,
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

		path = options['data'](setting,options)
		key = options['key'](setting,options)
		attrs = options['attrs'](setting,options)
		plots = options['plot'](setting,options)

		attribute = options['attribute'](setting,options)
		logger = options['logger'](setting,options)

		data = load(path)

		do = ((data is not None) and (data.get(key) is not None) and (data[key].get('parameters') is not None))

		if not do:
			continue

		logger(setting)

		parameters = data[key]['parameters']

		attr = tuple(setting[attr] for attr in attrs)

		if (attr not in fig) or (attr not in ax):
			fig[attr],ax[attr] = None,None

		if attribute['func']() in ['func']:

			func = functions

		elif attribute['func']() in ['functional']:

			func = functionals

		x = logspace(start=-20,stop=0,num=100)

		y = func(parameters,x)

		opts = dict(
			label='$%s$'%('~,~'.join(['{value}'.format(key=key,value=setting[key]) for key in ['M']])),
			color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
			marker='',
			linestyle='-',
			)

		fig[attr],ax[attr] = plot(x,y,fig=fig[attr],ax=ax[attr],options={**plots,**opts})

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

			legend = ax.legend(
				handles,labels,
				title="$k$",
				loc="upper right",
				ncol=1,
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
		attr=['test.pauli','pauli'],
		D=[2],
		N=[4],
		M=[0],
		)

	fig,ax = None,None

	for index,setting in enumerate(permute(settings)):

		attr = setting['attr']
		D = setting['D']
		N = setting['N']
		M = setting['M']

		path = options['path'](setting,options)
		plots = options['plot'](setting,options)
		attribute = options['attribute'](setting,options)
		logger = options['logger'](setting,options)

		d = D**N
		s = M+1
		w = 1

		z = {2.3433e-2:3,5.4553e-2:3,7.8291e-2:1}; z = {**z,**{1.2954e-2:d-sum(z[i] for i in z)}}; z = {i:z[i] for i in z if z[i]>0}
		z = {1:3}; z= {**z,**{0:d-sum(z[i] for i in z)}}; z = {i:z[i] for i in z if z[i]>0}
		# z = {(i+1)/d:1 for i in range(d)}; z= {**z,**{0:d-sum(z[i] for i in z)}}; z = {i:z[i] for i in z if z[i]>0}

		z = array([j for i in z for j in [i]*z[i]])

		logger(setting)

		try:

			if attribute['func']() in ['func']:

				func = parameter

			elif attribute['func']() in ['functional']:

				func = parameterizations

			parameters = func(z,d=d,s=s,w=w)

		except Exception as exception:

			logger('Exception:\n%r\n%r'%(exception,traceback.format_exc()))

		if attribute['func']() in ['func']:

			func = function

			opts = dict(
				label='$\\textrm{Conjecture}~:~{%s}$'%(setting['M']),
				color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
				marker='',
				linestyle='-',
				alpha=0.5,
				path=join(path,'plot','plot.test.%s.pdf'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]]))),
				)

		elif attribute['func']() in ['functional']:

			func = functional

			opts = dict(
				label='$\\textrm{Analytical}~:~{%s}$'%(setting['M']),
				color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
				marker='',
				linestyle='--',
				alpha=0.8,
				path=join(path,'plot','plot.test.%s.pdf'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]]))),
				)

		x = logspace(start=-20,stop=0,num=1000)

		y = func(parameters,x)

		fig,ax = plot(x,y,fig=fig,ax=ax,options={**plots,**opts})

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

	settings = dict(
		# attr=['pauli'],
		# attr=['tetrad'],
		# attr=['tetrad','pauli'],
		attr=['test.pauli','test.tetrad'],
		# attr=['test.pauli'],
		D=[2],
		# N=[2,3,4,5,6,7,8],
		# M=[0,2,4,8,16,32],
		# N=[3],
		# M=[0,2,4,8,16,32],
		N=[4],
		# M=[0,2,4,8,16,32],
		M=[0],
		)

	options = dict(
		boolean = (lambda settings={},options={}: {
			'run':0,
			'process':0,
			'test':1,
			}),
		path   = (lambda settings={},options={}: '~/scratch/probability/distribution'),
		io     = (lambda settings={},options={}: dict(wr='a')),
		do     = (lambda settings={},options={}: (not exists(options['data'](settings,options))) or (options['key'](settings,options) not in load(options['data'](settings,options)))),
		key    = (lambda settings={},options={}: 'operator.{attr}.N.{N}.M.{M}'.format(**settings)),
		attrs  = (lambda settings={},options={}: ('attr','N')),
		attribute = (lambda settings={},options={}:{
			**{i:dict(func=lambda attr=i:'func',data=lambda attr=i:real(eig(getattr(basis,attr)(D=settings['D'])))) for i in ['tetrad','pauli']},
			**{i:dict(func=lambda attr=i:'functional',data=lambda attr=i:real(eig(getattr(basis,attr.split(delim)[-1])(D=settings['D'])))) for i in ['test.tetrad','test.pauli']},
			}.get(settings['attr'])
			),
		data   = (lambda settings={},options={}: join(options['path'](settings,options),'data','data.hdf5')),
		logger = (lambda settings={},options={}: Logger(file=join(options['path'](settings,options),'log','log.log'),verbose='info')),
		plot   =  (lambda settings={},options={}: dict(
			path=join(options['path'](settings,options),'plot','plot.distribution.%s.pdf'%('.'.join([str(i) for attr in options['attrs'](settings,options) for i in [attr,settings[attr]]]))),
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
