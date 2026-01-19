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

from src.utils import array,asscalar,tensorprod,concatenate,meshgrid,linspace,logspace,inplace,partial,scan,vmap,callback,vectorize,allclose,vtype,copy
from src.utils import exp,log,log1p
from src.utils import log10,real,nan,is_naninf,epsilon
from src.utils import where,nonzero,unique,sort,minimum,maximum,minimums,maximums
from src.utils import eig,product,addition,permutations,partitions,products,comb,factorial,multinomial,permute,distribution
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
					ax.set_xticklabels(labels=['$%s$'%(str(i)) if i not in [0,1] else '$1$' if i not in [0] else '$0$' for i in [0,0.2,0.4,0.6,0.8,1]],size=60)
					ax.set_yticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16,1e20])
					ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [20,16,12,8,4,0,-4,-8,-12,-16,-20]],size=60)

				elif index == 1:

					ax.set_xscale(value="log",base=10)
					ax.set_yscale(value="log",base=10)
					ax.set_xlim(xmin=1e-22,xmax=1e2)
					ax.set_ylim(ymin=1e-21,ymax=1e21)
					ax.set_xticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1])
					ax.set_xticklabels(labels=['$10^{%d}$'%(-i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [20,16,12,8,4,0]],size=60)
					ax.set_yticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16,1e20])
					ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [20,16,12,8,4,0,-4,-8,-12,-16,-20]],size=60)

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

	def func(x,parameters):

		def functional(x,parameters,functions):
			x = (x-parameters['u'])/(parameters['v']-parameters['u'])
			x = functions['x'](x)
			x = parameters['w']*(1/(parameters['v']-parameters['u']))*functions['exp'](parameters['d']*(parameters['a']*functions['log'](x) + functions['log1p'](-2*parameters['b']*(x/parameters['c']) + parameters['b']*(x/parameters['c'])**2) - functions['log'](parameters['p'])))
			return x

		from mpmath import exp,log,log1p
		functions = dict(exp=exp,log=log,log1p=log1p,x=lambda x:x if ((x>=0)*(x<=1)) else 0)
		func = partial(functional,parameters=parameters,functions=functions)

		from src.utils import exp,log,log1p
		functions = dict(exp=exp,log=log,log1p=log1p,x=lambda x:where(x<1,where(x>0,x,0),0))
		function = partial(functional,parameters=parameters,functions=functions)

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

		y = func(x,parameters)

		fig,ax = plot(x,y,fig=fig,ax=ax,options={**options,**parameters['options']})

	y = sum(func((x-(i/len(settings)))/(1-(i/len(settings))),parameters) for i,parameters in enumerate(settings))/len(settings)

	fig,ax = plot(x,y,fig=fig,ax=ax,options={**options,** dict(
				label='$\\sum \\alpha ~,~ \\beta $',
				color='viridis_%f'%(0.9),
				marker='',
				linestyle='-',
				)})

	return

def analyse(settings,options,*args,**kwargs):

	# def f(x):
	# 	func = lambda y,parameters: y+function(parameters,x)
	# 	y = 0*x
	# 	y = scan(parameters,y,func)
	# 	return y


	# def i(x):
	# 	def func(x,z):
	# 		return f(z*x)/z
	# 	# func = lambda x,z: float(f(float(z)*x)/float(z))
	# 	# bounds = linearspace(0,1,10)
	# 	bounds = [0,1]

	# 	def func(x):
	# 		return f(x)

	# 	# z = z.astype(jnp.result_type(float, z.dtype))
	# 	# function = callback(lambda z: float(vtype(integral(partial(func,z=asscalar(vtype(z,float))),bounds),float)),shape=(),dtype=float)
	# 	# function = callback(lambda z: integral(partial(func,z=z),bounds),shape=(),dtype=float)
	# 	# function = lambda z: integral(partial(func,z=z),bounds)
	# 	function = lambda z,l=minimum(x): integrate(func,linspace(l,z,100),weights=10000,method='sinh_tanh')
	# 	# function = lambda x: vtype(integral(partial(func,z=mpmathify(vtype(x,float))),bounds),float)
	# 	y = vmap(function)(x)
	# 	# function = lambda x: float(integral(lambda x,z=x:func(z*x)/z,linearspace(0,1,100)))
	# 	# y = vectorize(function)(x)
	# 	return y

	# z = f(x)
	# print(allclose(z,y))

	# x = logspace(log10(minimum(parameters[:,0])),log10(maximum(parameters[:,1])),10)

	# z = i(x)

	# print(parameters)
	# print(x)
	# print(z)

	return

def run(settings,options,*args,**kwargs):

	permutations = permute(settings)

	for index,setting in enumerate(permutations):

		attr = setting['attr']
		D = setting['D']
		N = setting['N']
		M = setting['M']

		path = options['data'](setting,options)
		key = options['key'](setting,options)
		io = options['io'](setting,options)

		logger = options['logger'](setting,options)

		eps = options['eps'](setting,options)
		bounds = options['bounds'](setting,options)

		do = options['do'](setting,options)

		if not do:
			continue

		logger(setting)

		operator = real(eig(getattr(basis,attr)(D=D)))
		parameters = []

		for number,partition in enumerate(partitions(N,D**2)):

			try:

				z = tensorprod([obj for i,j in enumerate(partition) for obj in [operator[i]]*j])
				u,v = asscalar(product(minimum(z,axis=-1))),asscalar(product(maximum(z,axis=-1)))

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
					logger(f'{number}'+'\t'+'\t'.join([f'{i}' if i != 1 else f'{i}' for i in [c,p,q]]))

				parameters = array([*parameters,[float(min(params[i],sys.float_info.max)) for i in params]])

			except Exception as exception:
				print('----')
				logger('Exception:\n%r\n%r'%(exception,traceback.format_exc()))


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
		# ax.set_xticklabels(labels=['$10^{%d}$'%(-i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [20,16,12,8,4,0]],size=60)
		# ax.set_yticks(ticks=[1e-20,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8,1e12,1e16,1e20])
		# ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [20,16,12,8,4,0,-4,-8,-12,-16,-20]],size=60)

		# ax.set_xscale(value="log",base=4)
		# ax.set_yscale(value="log",base=10)
		# ax.set_xlim(xmin=2**(-11),xmax=2**(2))
		# ax.set_ylim(ymin=1e-129,ymax=1e9)
		# ax.set_xticks(ticks=[2**(-2*i) for i in [10,8,6,4,2,0]])
		# ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [10,8,6,4,2,0]],size=60)
		# ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
		# ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [128,64,32,16,12,8,4,0,-4,-8]],size=60)

		ax.set_xscale(value="log",base=4)
		ax.set_yscale(value="log",base=10)
		ax.set_xlim(xmin=2**(-2*17),xmax=2**(2))
		ax.set_ylim(ymin=1e-129,ymax=1e9)
		ax.set_xticks(ticks=[2**(-2*i) for i in [16,14,12,10,8,6,4,2,0]])
		ax.set_xticklabels(labels=['$2^{-2\\cdot%d}$'%(i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [16,14,12,10,8,6,4,2,0]],size=60)
		ax.set_yticks(ticks=[1e-128,1e-64,1e-32,1e-16,1e-12,1e-8,1e-4,1,1e4,1e8])
		ax.set_yticklabels(labels=['$10^{%d}$'%(-i) if i not in [0,1] else '$10$' if i not in [0] else '$1$' for i in [128,64,32,16,12,8,4,0,-4,-8]],size=60)

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
			title="$M$",
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

	permutations = permute(settings)

	fig,ax = {},{}

	for index,setting in enumerate(permutations):

		path = options['data'](setting,options)
		key = options['key'](setting,options)
		attrs = options['attrs'](setting,options)
		plots = options['plot'](setting,options)

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

		def f(x):
			func = lambda y,parameters: y+function(parameters,x)
			y = 0*x
			y = scan(parameters,y,func)
			return y

		x = logspace(start=-20,stop=0,num=10000)

		y = f(x)

		opts = dict(
			label='$%s$'%('~,~'.join(['{value}'.format(key=key,value=setting[key]) for key in ['M']])),
			color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
			marker='',
			linestyle='-',
			)

		fig[attr],ax[attr] = plot(x,y,fig=fig[attr],ax=ax[attr],options={**plots,**opts})



		parameters = parameters[0]

		def f(x):

			params = dict(
				function='beta.pdf',
				a=(parameters[6]*parameters[7]),
				b=((parameters[8]-parameters[6])*parameters[7]),
				loc=0,
				scale=1/(parameters[8]),
				)

			y = distribution(x,**params)

			return y

		x = x
		y = f(x)

		opts = dict(
			label=None,
			color='viridis_%f'%((settings['M'].index(setting['M'])+1)/(len(settings['M'])+1)),
			marker='',
			linestyle='--',
			alpha=1,
			zorder=100,
			path_effects=[matplotlib.patheffects.Stroke(linewidth=20,foreground='k'),matplotlib.patheffects.Normal()],
			)

		fig[attr],ax[attr] = plot(x,y,fig=fig[attr],ax=ax[attr],options={**plots,**opts})

	return

def setup(settings,options,*args,**kwargs):

	boolean = options['boolean'](settings,options)
	logger = options['logger'](settings,options)

	logger(boolean)

	if boolean.get('run'):

		run(settings,options,*args,**kwargs)

	if boolean.get('process'):

		process(settings,options,*args,**kwargs)

	if boolean.get('analyse'):

		draw(settings,options,*args,**kwargs)

	if boolean.get('draw'):

		draw(settings,options,*args,**kwargs)

	return

def main(*args,**kwargs):

	settings = dict(
		# attr=['pauli'],
		# attr=['tetrad'],
		attr=['tetrad','pauli'],
		D=[2],
		# N=[2,3,4,5,6,7,8],
		# M=[0,2,4,8,16,32],
		# N=[3],
		# M=[0,2,4,8,16,32],
		N=[2,3,4,5,6,7,8],
		M=[0,2,4,8,16,32],
		)

	options = dict(
		boolean = (lambda settings={},options={}: {
			'run':1,
			'process':1,
			'analyse':0,
			'draw':0
			}),
		path   = (lambda settings={},options={}: '~/scratch/probability/distribution'),
		io     = (lambda settings={},options={}: dict(wr='a')),
		do     = (lambda settings={},options={}: (not exists(options['data'](settings,options))) or (options['key'](settings,options) not in load(options['data'](settings,options)))),
		key    = (lambda settings={},options={}: 'operator.{attr}.N.{N}.M.{M}'.format(**settings)),
		attrs  = (lambda settings={},options={}: ('attr','N')),
		eps    = (lambda settings={},options={}: 1e-12),#epsilon()),
		bounds = (lambda settings={},options={}: linearspace(0,1,500)),
		data   = (lambda settings={},options={}: join(options['path'](settings,options),'data','data.hdf5')),
		logger = (lambda settings={},options={}: Logger(file=join(options['path'](settings,options),'log','log.log'),verbose='info')),
		plot   =  (lambda settings={},options={}: dict(
			path=join(options['path'](settings,options),'plot','plot.process.%s.pdf'%('.'.join([str(i) for attr in options['attrs'](settings,options) for i in [attr,settings[attr]]]))),
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