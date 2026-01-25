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

from src.utils import array,rand,asscalar,tensorprod,concatenate,meshgrid,linspace,logspace,inplace,partial,scan,vmap,callback,vectorize,allclose,vtype,copy
from src.utils import exp,log,log1p
from src.utils import log10,real,nan,is_naninf,epsilon,delim
from src.utils import where,nonzero,sign,unique,sort,minimum,maximum,minimums,maximums
from src.utils import eig,addition,prod,permutations,partitions,comb,factorial,gammaln,multinomial,permute,distribution
from src.utils import integral as integrate

from src.quantum import Basis as basis

from src.io import load,dump,exists,join,split

from src.logger import Logger

from mpmath import quad as integral,linspace as linearspace,mpmathify,workdps,sqrt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects

from mpmath import exp,log,log1p
def func(x,parameters,exp=exp,log=log,log1p=log1p):
	x = (x-parameters[0])/(parameters[1]-parameters[0])
	x = (parameters[2]*(1/(parameters[1]-parameters[0]))*exp((parameters[6]*parameters[7]-1)*log(x) + (((parameters[8]-parameters[6])*parameters[7]-1)/2)*log1p(-2*parameters[4]*x + parameters[5]*x**2) - log(parameters[3]) - log(parameters[9]))) if ((x>=0)*(x<=1)) else 0
	return x

from src.utils import exp,log,log1p
def function(parameters,x,exp=exp,log=log,log1p=log1p):
	x = (x-parameters[0])/(parameters[1]-parameters[0])
	x = where((x>0)*(x<=1),(parameters[2]*(1/(parameters[1]-parameters[0]))*exp((parameters[6]*parameters[7]-1)*log(x) + (((parameters[8]-parameters[6])*parameters[7]-1)/2)*log1p(-2*parameters[4]*x + parameters[5]*x**2) - log(parameters[3]) - log(parameters[9]))),0)
	return x

from src.utils import exp,log,log1p
def functional(parameters,x,exp=exp,log=log,log1p=log1p):
	z,l,w = parameters[0].astype(float),parameters[1].astype(int),parameters[2].astype(float)
	z,l,w = z[l>0],l[l>0],w[l>0]
	n,d,u,v = z.size,addition(l),minimum(z),maximum(z)
	y = sum((w[i]*(sign(z[i]-x)**((d-l[i]+k)%2))*exp((d-l[i]+k-1)*log(abs(z[i]-x)))) for i in range(n) if ((w[i]!=0) and (l[i]!=0)) for k in range(l[i]))
	y = where((x>u)*(x<v),y,0)
	return y

def draw(*args,**kwargs):

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

			fig,axes = plt.subplots(2,1) if fig is None or ax is None else (fig,ax)

			for index,ax in enumerate(axes):

				ax.errorbar(x,y,yerr,xerr,**options)

				ax.set_xlabel(xlabel="$x$",size=60)
				ax.set_ylabel(ylabel="$f(x)$",size=60)

				if index == 0:

					ax.set_xscale(value="linear")
					ax.set_yscale(value="log",base=10)
					ax.set_xlim(xmin=-0.1,xmax=1.1)
					ax.set_ylim(ymin=1e-21,ymax=1e21)
					ax.set_xticks(ticks=[0,0.2,0.4,0.6,0.8,1])
					ax.set_xticklabels(labels=['$%s$'%(str(i)) if i not in [0,1] else f'${i}$' for i in [0,0.2,0.4,0.6,0.8,1]],size=60)
					ax.set_yticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16,1e20])
					ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [20,16,12,8,4,0,-4,-8,-12,-16,-20]],size=60)

				elif index == 1:

					ax.set_xscale(value="log",base=10)
					ax.set_yscale(value="log",base=10)
					ax.set_xlim(xmin=1e-22,xmax=1e2)
					ax.set_ylim(ymin=1e-21,ymax=1e21)
					ax.set_xticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1])
					ax.set_xticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [20,16,12,8,4,0]],size=60)
					ax.set_yticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16,1e20])
					ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [20,16,12,8,4,0,-4,-8,-12,-16,-20]],size=60)

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
				# title="$$",
				loc="lower left",
				ncol=2,
				title_fontsize=50,
				prop={"size":50},
				markerscale=6,
				handlelength=2.5
			)

			if settings.get('path'):
				fig.set_size_inches(w=36,h=36)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname=settings.get('path'))

			ax = axes

		return fig,ax

	def f(x,parameters):

		def f(x,parameters,functions):
			x = (x-parameters['u'])/(parameters['v']-parameters['u'])
			x = functions['x'](x)
			x = parameters['w']*(1/(parameters['v']-parameters['u']))*functions['exp'](parameters['d']*(parameters['a']*functions['log'](x) + functions['log1p'](-2*parameters['b']*(x/parameters['c']) + parameters['b']*(x/parameters['c'])**2) - functions['log'](parameters['p'])))
			return x

		from mpmath import exp,log,log1p
		functions = dict(exp=exp,log=log,log1p=log1p,x=lambda x:x if ((x>=0)*(x<=1)) else 0)
		func = partial(f,parameters=parameters,functions=functions)

		from src.utils import exp,log,log1p
		functions = dict(exp=exp,log=log,log1p=log1p,x=lambda x:where(x<1,where(x>0,x,0),0))
		function = partial(f,parameters=parameters,functions=functions)

		parameters['p'] = float(integral(func,linearspace(0,1,100)))

		y = function(x,parameters=parameters)

		return y

	settings = [
		{'u':0,'v':1,'w':1,'p':1,'a':0,'b':1,'c':1,'d':2,
			'options': dict(
				label='$\\alpha = 1 ~,~\\beta = 0$',
				color='k',
				marker='',
				linestyle='-',
				),
		},
		{'u':0,'v':1,'w':1,'p':1,'a':2*((1)*2-1)/((10-1)*2-1),'b':1-((1-(2*((1)*2-1)/((10-1)*2-1)*((2*((1)*2-1)/((10-1)*2-1)) + 2))/(((2*((1)*2-1)/((10-1)*2-1)*((2*((1)*2-1)/((10-1)*2-1)) + 2)))+1))/1000),'c':1,'d':((10-1)*2-1)/2,
			'options': dict(
				label='$\\frac{\\beta(\\beta+2)}{\\beta(\\beta+2)+1} < \\alpha < 1 ~,~\\beta \\to 0$',
				color='viridis_%f'%(0.1),
				marker='',
				linestyle='-',
				),
		},
		{'u':0,'v':1,'w':1,'p':1,'a':2*((1)*2-1)/((10-1)*2-1),'b':1/5,'c':1,'d':((10-1)*2-1)/2,
			'options': dict(
				label='$\\alpha < \\frac{\\beta(\\beta+2)}{\\beta(\\beta+2)+1} < 1 ~,~\\beta \\to 0$',
				color='viridis_%f'%(0.25),
				marker='',
				linestyle='-',
				),
		},
		{'u':0,'v':1,'w':1,'p':1,'a':2*((9)*2-1)/((10-9)*2-1),'b':1/5,'c':1,'d':((10-9)*2-1)/2,
			'options': dict(
				label='$\\alpha < \\frac{\\beta(\\beta+2)}{\\beta(\\beta+2)+1} < 1 ~,~\\beta \\to \\infty$',
				color='viridis_%f'%(0.5),
				marker='',
				linestyle='-',
				),
		},
		{'u':0,'v':1,'w':1,'p':1,'a':2*((9)*2-1)/((10-9)*2-1),'b': 1-((1-(2*((9)*2-1)/((10-9)*2-1)*((2*((9)*2-1)/((10-9)*2-1)) + 2))/(((2*((9)*2-1)/((10-9)*2-1)*((2*((9)*2-1)/((10-9)*2-1)) + 2)))+1))/1000),'c':1,'d':((10-9)*2-1)/2,
			'options': dict(
				label='$\\frac{\\beta(\\beta+2)}{\\beta(\\beta+2)+1} < \\alpha < 1 ~,~\\beta \\to \\infty$',
				color='viridis_%f'%(0.75),
				marker='',
				linestyle='-',
				),
		},
	]

	x = logspace(start=-20,stop=0,num=100000)

	fig,ax = None,None
	options = dict(
		path=join('~/scratch/probability/distribution','plot','plot.distribution.pdf'),
		mplstyle=join('~/scratch/probability/distribution','plot','plot.mplstyle'),
		markersize=9,
		linewidth=16,
		alpha=0.8,
	)

	for parameters in settings:

		y = f(x,parameters)

		fig,ax = plot(x,y,fig=fig,ax=ax,options={**options,**parameters['options']})

	y = sum(f((x-(i/len(settings)))/(1-(i/len(settings))),parameters) for i,parameters in enumerate(settings))/len(settings)

	fig,ax = plot(x,y,fig=fig,ax=ax,options={**options,** dict(
				label='$\\sum \\alpha ~,~ \\beta $',
				color='viridis_%f'%(0.9),
				marker='',
				linestyle='-',
				)})

	return

def run(settings,options,*args,**kwargs):

	for index,setting in enumerate(permute(settings)):

		attr = setting['attr']
		D = setting['D']
		N = setting['N']
		M = setting['M']

		path = options['data'](setting,options)
		key = options['key'](setting,options)
		io = options['io'](setting,options)

		logger = options['logger'](setting,options)

		attribute = options['attribute'](setting,options)
		eps = options['eps'](setting,options)
		bounds = options['bounds'](setting,options)

		do = options['do'](setting,options)

		if not do:
			continue

		logger(setting)

		data = attribute['data']()

		parameters = []

		for number,partition in enumerate(partitions(N,D**2)):

			try:

				z = tensorprod([obj for i,j in enumerate(partition) for obj in [data[i]]*j])

				if attribute['func']() in ['func']:

					u,v = asscalar(minimum(z)),asscalar(maximum(z))

					z = (z-u)/(v-u)
					z = z[(z>eps)*(z<=1)]

					a,b = asscalar(addition(1/z)/z.size),asscalar(addition(1/z**2)/z.size)
					l,s,d = z.size,M+1,D**N
					w = multinomial(partition)/D**(2*N)

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

					if number == 0:
						logger(f'#/{len(list(partitions(N,D**2)))}'+'\t'+'\t'.join([f'{i:8}' for i in ['c','p','q']]))
					with workdps(8):
						logger(f'{number}'+'\t'+'\t\t'.join([f'{i}' if i != 1 else f'{i}' for i in [c,p,q]]))

					parameters.append([float(max(min(params[i],sys.float_info.max),sys.float_info.min)) for i in params])

				elif attribute['func']() in ['functional']:

					z = where(z>eps,z,0)

					opts = dict(return_counts=True)
					z,l = unique(z,**opts)
					n,d = len(z),addition(l)
					w = [multinomial(partition)/D**(2*N)]*n

					w = array([
						(
						((-1)**(k))*
						exp(
						log(w[i])+
						gammaln(d-1)-
						log(2)-
						gammaln(d-l[i]+k)-
						gammaln(l[i]-k)
						)
						+
						sum(
						(
						exp(
						sum(
						(
						gammaln(l[j]+p[j])-
						gammaln(p[j]+1)-
						gammaln(l[j])-
						(l[j]+p[j])*log(abs(z[i]-z[j]))
						)
						for j in range(n)
						if ((j!=i) and (l[j]>0))
						)
						)
						*
						prod(
						(
						sign(z[i]-z[j])**((l[j]+p[j])%2)
						)
						for j in range(n)
						if ((j!=i) and (l[j]>0))
						)
						)
						for p in permutations(k+1,repeat=n)
						if ((p[i]==0) and all(((l[j]>0)or(p[j]==0)) for j in range(n)) and (sum(p)==k))
						)
						) if ((w[i]>0) and (l[i]>0)) else 0
						for i in range(n)
						for k in range(l[i])
						])

					logger(f'{number}'+'\t'+'\t'.join([f'{i}' for i in [z,l,w]]))

					parameters.append([z,l,w])

			except Exception as exception:

				logger('Exception:\n%r\n%r'%(exception,traceback.format_exc()))


		if attribute['func']() in ['func']:

			parameters = array(parameters)

		elif attribute['func']() in ['functional']:

			n = max(len(i) for params in parameters for i in params)
			parameters = array([[[*i,*[0]*(n-len(i))] for i in params] for params in parameters])

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
		# ax.set_xticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [20,16,12,8,4,0]],size=60)
		# ax.set_yticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16,1e20])
		# ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [20,16,12,8,4,0,-4,-8,-12,-16,-20]],size=60)

		# ax.set_xscale(value="log",base=4)
		# ax.set_yscale(value="log",base=10)
		# ax.set_xlim(xmin=2**(-11),xmax=2**(2))
		# ax.set_ylim(ymin=1e-129,ymax=1e9)
		# ax.set_xticks(ticks=[2**(-2*i) for i in [10,8,6,4,2,0]])
		# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [10,8,6,4,2,0]],size=60)
		# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
		# ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [128,64,32,16,12,8,4,0,-4,-8]],size=60)

		ax.set_xscale(value="log",base=4)
		ax.set_yscale(value="log",base=10)
		ax.set_xlim(xmin=2**(-2*17),xmax=2**(2))
		ax.set_ylim(ymin=1e-129,ymax=1e9)
		ax.set_xticks(ticks=[2**(-2*i) for i in [16,14,12,10,8,6,4,2,0]])
		ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [16,14,12,10,8,6,4,2,0]],size=60)
		ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
		ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [128,64,32,16,12,8,4,0,-4,-8]],size=60)

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

			def f(x):
				func = lambda y,parameters: y+function(parameters,x)
				y = 0*x
				y = scan(parameters,y,func)
				return y
		elif attribute['func']() in ['functional']:

			def f(x):
				func = lambda y,parameters: y+functional(parameters,x)
				y = 0*x
				for params in parameters:
					y = func(y,params)
				return y

		x = logspace(start=-20,stop=0,num=100)

		y = f(x)

		opts = dict(
			label='$%s$'%('~,~'.join(['{value}'.format(key=key,value=setting[key]) for key in ['M']])),
			color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
			marker='',
			linestyle='-',
			)

		fig[attr],ax[attr] = plot(x,y,fig=fig[attr],ax=ax[attr],options={**plots,**opts})


		# parameters = parameters[0]

		# def f(x):

		# 	params = dict(
		# 		function='beta.pdf',
		# 		a=(parameters[6]*parameters[7]),
		# 		b=((parameters[8]-parameters[6])*parameters[7]),
		# 		loc=0,
		# 		scale=1/(parameters[8]),
		# 		)

		# 	y = distribution(x,**params)

		# 	return y

		# x = x
		# y = f(x)

		# opts = dict(
		# 	label=None,
		# 	color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
		# 	marker='',
		# 	linestyle='--',
		# 	alpha=1,
		# 	zorder=100,
		# 	path_effects=[matplotlib.patheffects.Stroke(linewidth=20,foreground='k'),matplotlib.patheffects.Normal()],
		# 	)

		# fig[attr],ax[attr] = plot(x,y,fig=fig[attr],ax=ax[attr],options={**plots,**opts})

	return

def test(settings,options,*args,**kwargs):

	from src.utils import array,asscalar
	from src.utils import addition,minimum,maximum
	from src.utils import prod

	from mpmath import sqrt


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
			ax.set_ylim(ymin=1e-17,ymax=1e5)
			ax.set_xticks(ticks=[1e-4,1e-3,1e-2,1e-1,1])
			ax.set_xticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [4,3,2,1,0]],size=60)
			ax.set_yticks(ticks=[1e-16,1e-12,1e-8,1e-4,1,1e4])
			ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [16,12,8,4,0,-4]],size=60)

			# ax.set_xscale(value="log",base=4)
			# ax.set_yscale(value="log",base=10)
			# ax.set_xlim(xmin=2**(-11),xmax=2**(2))
			# ax.set_ylim(ymin=1e-129,ymax=1e9)
			# ax.set_xticks(ticks=[2**(-2*i) for i in [10,8,6,4,2,0]])
			# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [10,8,6,4,2,0]],size=60)
			# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [128,64,32,16,12,8,4,0,-4,-8]],size=60)

			# ax.set_xscale(value="log",base=4)
			# ax.set_yscale(value="log",base=10)
			# ax.set_xlim(xmin=2**(-2*17),xmax=2**(2))
			# ax.set_ylim(ymin=1e-129,ymax=1e9)
			# ax.set_xticks(ticks=[2**(-2*i) for i in [16,14,12,10,8,6,4,2,0]])
			# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$2^{-2}$' if i in [1] else '$1$' for i in [16,14,12,10,8,6,4,2,0]],size=60)
			# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
			# ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0] else '$1$' for i in [128,64,32,16,12,8,4,0,-4,-8]],size=60)

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
				# title="$$",
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


	setting = dict(
		attr='test.pauli',
		D=2,
		N=3,
		M=0,
		)

	fig,ax = None,None

	d = setting['D']**setting['N']
	s = setting['M']+1
	# o = {2.3433e-2:1,5.4553e-2:1,7.8291e-2:1,1.2954e-2:1,2.8291e-2:1,6.2954e-2:1,7.8291e-3:1,5.2954e-2:1}
	o = {2.3433e-2:3,5.4553e-2:3,7.8291e-2:1,1.2954e-2:1}
	o = {1:5,0:3}

	z = array([k for i in o for k in [i]*(o[i]*s)])


	logger = options['logger'](setting,options)

	plots = options['plot'](setting,options)

	attribute = options['attribute'](setting,options)
	eps = options['eps'](setting,options)
	bounds = options['bounds'](setting,options)

	logger(setting)

	try:

		if attribute['func']() in ['func']:

			u,v = asscalar(minimum(z)),asscalar(maximum(z))

			z = (z-u)/(v-u)
			z = z[(z>eps)*(z<=1)]

			a,b = asscalar(addition(1/z)/z.size),asscalar(addition(1/z**2)/z.size)
			l,s,d = z.size,M+1,D**N
			w = multinomial(partition)/D**(2*N)

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

			parameters = [float(max(min(params[i],sys.float_info.max),sys.float_info.min)) for i in params]

		elif attribute['func']() in ['functional']:

			opts = dict(return_counts=True)
			z,l = unique(z,**opts)
			n,d = len(z),addition(l)
			w = [1]*n

			w = array([
				(
				((-1)**(k))*
				exp(
				log(w[i])+
				gammaln(d-1)-
				log(2)-
				gammaln(d-l[i]+k)-
				gammaln(l[i]-k)
				)
				+
				sum(
				(
				exp(
				sum(
				(
				gammaln(l[j]+p[j])-
				gammaln(p[j]+1)-
				gammaln(l[j])-
				(l[j]+p[j])*log(abs(z[i]-z[j]))
				)
				for j in range(n)
				if ((j!=i) and (l[j]>0))
				)
				)
				*
				prod(
				(
				sign(z[i]-z[j])**((l[j]+p[j])%2)
				)
				for j in range(n)
				if ((j!=i) and (l[j]>0))
				)
				)
				for p in permutations(k+1,repeat=n)
				if ((p[i]==0) and all(((l[j]>0)or(p[j]==0)) for j in range(n)) and (sum(p)==k))
				)
				) if ((w[i]>0) and (l[i]>0)) else 0
				for i in range(n)
				for k in range(l[i])
				])

			parameters = [z,l,w]

	except Exception as exception:

		logger('Exception:\n%r\n%r'%(exception,traceback.format_exc()))


	if attribute['func']() in ['func']:

		parameters = array(parameters)

		f = function

		opts = dict(
			label='$\\textrm{Analytical}$',
			color='viridis_%f'%(0.5),
			marker='',
			linestyle='-',
			)

	elif attribute['func']() in ['functional']:

		n = max(len(i) for i in parameters)
		parameters = array([[*i,*[0]*(n-len(i))] for i in parameters])

		f = functional

		opts = dict(
			label='$\\textrm{Theory}$',
			color='viridis_%f'%(0.25),
			marker='',
			linestyle='--',
			)


	x = logspace(start=-20,stop=0,num=1000)

	y = f(parameters,x)

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

	if boolean.get('draw'):

		draw(settings,options,*args,**kwargs)

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
		N=[2,3,4],
		M=[0,2,4,8,16,32],
		)

	options = dict(
		boolean = (lambda settings={},options={}: {
			'run':0,
			'process':0,
			'test':1,
			'draw':0
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
		eps    = (lambda settings={},options={}: 1e-12),#epsilon()),
		bounds = (lambda settings={},options={}: linearspace(0,1,500)),
		data   = (lambda settings={},options={}: join(options['path'](settings,options),'data','data.hdf5')),
		logger = (lambda settings={},options={}: Logger(file=join(options['path'](settings,options),'log','log.log'),verbose='info')),
		plot   =  (lambda settings={},options={}: dict(
			path=join(options['path'](settings,options),'plot','plot.test.%s.pdf'%('.'.join([str(i) for attr in options['attrs'](settings,options) for i in [attr,settings[attr]]]))),
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
