#!/usr/bin/env python

import os,sys

def plot(settings,options,*args,**kwargs):

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
			settings['path'] = join(options.pop('path') if options.get('path') else None,options.pop('name'))
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

				caxes = {cax:None for cax in ['viridis','magma']}

				for color in caxes:

					colors = [plt.get_cmap(color)((i+1)/(n+1)) for i in range(n)]
					cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name=None,colors=colors,N=100*len(colors))

					opts = dict(location='right',fraction=0.15,shrink=1,aspect=40,pad=0.02,anchor={'viridis':(1.25,0.5),'magma':(1.5,0.5)}[color])
					cax,opts = matplotlib.colorbar.make_axes([ax for ax in fig.axes],**opts)

					# opts = dict(position='right',size='2%',pad={'viridis':0.05,'magma':0.5}[color])
					# cax = make_axes_locatable(ax).append_axes(**opts)

					opts = {**dict(cmap=cmap,orientation='vertical')}
					cbar = matplotlib.colorbar.ColorbarBase(cax,**opts)

					if color in ['viridis']:
						cbar.ax.set_yticks(ticks=[(i)/(n-1) for i in range(n)])
						cbar.ax.set_yticklabels(labels=[])

						cbar.ax.yaxis.set_tick_params(pad=20)
						cbar.ax.tick_params(labelsize=settings['fontsize'],which='major',pad=20,size=15,length=15,width=1)

					elif color in ['magma']:
						cbar.ax.set_ylabel(ylabel='$\\textrm{Depth}~~k$',size=settings['fontsize'])
						cbar.ax.set_yticks(ticks=[(i)/(n-1) for i in range(n)])
						cbar.ax.set_yticklabels(labels=['$%s$'%(i.replace('$','').split('~,~')[0]) for i in ax.get_legend_handles_labels()[1][:n]],size=settings['fontsize'])

						cbar.ax.yaxis.set_tick_params(pad=20)
						cbar.ax.tick_params(labelsize=settings['fontsize'],which='major',pad=20,size=15,length=15,width=1)

					caxes[color] = cax

				for color in caxes:
					cax = caxes[color]
					if color in ['viridis']:
						cax.set_position([cax.get_position().x0 + 0.1675,cax.get_position().y0,cax.get_position().width,cax.get_position().height])
					elif color in ['magma']:
						cax.set_position([cax.get_position().x0 + 0,cax.get_position().y0,cax.get_position().width,cax.get_position().height])

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


			if settings.get('path'):
				fig.set_size_inches(w=65,h=45)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname=settings.get('path'),bbox_inches='tight',pad_inches=0.5)

		return fig,ax


	settings = dict(
		attr=['distribution'],
		D=[2],
		N=[8],
		L=[1,(1,2)],
		parameters=[0,-2],
		M=[0,2,4,8,16,32],
		)

	fig,ax = None,None

	data = {}
	file = options['data']

	path = options['path']
	logger = options['logger']

	logger(path)

	if not exists(file):

		for index,setting in enumerate(permute(settings)):

			attr = setting['attr']
			D = setting['D']
			N = setting['N']
			M = setting['M']
			L = setting['L']
			parameters = setting['parameters']

			args,kwargs = tuple((-32,0,1000,)),dict(endpoint=False)

			d = D**N
			s = M+1

			logger(setting)

			plots = dict(
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
				name=join('plot.distribution.%s'%('.'.join([str(i) for attr in ['N'] for i in [attr,setting[attr]]])),ext='pdf'),
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

				return y

			x = logspace(*args,**kwargs)

			y = func(x,setting)

			fig,ax = (None,None) if all(settings[i].index(setting[i]) == 0 for i in ['M','L','parameters']) else (fig,ax)

			fig,ax = plot(x,y,fig=fig,ax=ax,options={**plots,**options['plot']})


			key = str(index)
			value = dict(x=x,y=y,options=plots)
			data[key] = value

			dump(data,file)

	else:

		def parse(data):
			for key in data:
				if isinstance(data[key],dict):
					parse(data[key])
				else:
					if isinstance(data[key],str) and (data[key] in  ['none']):
						data[key] = None
			return

		data = load(file)

		parse(data)

		for key in natsorted(data):
			x,y,plots = data[key]['x'],data[key]['y'],data[key]['options']
			fig,ax = plot(x,y,fig=fig,ax=ax,options={**plots,**options['plot']})

		lines = [
			[0],[0,2],[0,2,5],[0,2,5,6],[0,2,5,6,8],[0,2,5,6,8,11],
			[0,2,5,6,8,11,12],[0,2,5,6,8,11,12,16],[0,2,5,6,8,11,12,16,18],[0,2,5,6,8,11,12,16,18,23]
			]
		for number,lines in enumerate(lines):
			for index,line in enumerate(ax.get_lines()):
				if index not in lines:
					line.set_visible(False)

			path = join(options.get('path') if options.get('path') else None,join('.'.join([split(plots.get('name'),directory_file=True),str(number)]),ext=split(plots.get('name'),ext=True)))

			fig.set_size_inches(w=65,h=45)
			fig.subplots_adjust()
			fig.tight_layout()
			fig.savefig(fname=path,bbox_inches='tight',pad_inches=0.5)

			for index,line in enumerate(ax.get_lines()):
				line.set_visible(True)

	return


def main(*args,**kwargs):

	path = str(args[0]) if len(args)>0 else '~/scratch/probability/distribution'

	settings = dict()

	options = dict(
		path   = path,
		data   = join(path,'data',ext='hdf5'),
		logger = Logger(file=None,verbose='info'),
		plot = dict(
			path = path,
			mplstyle = join(path,'plot',ext='mplstyle'),
			)
		)

	args = ()

	kwargs = {}

	plot(settings,options,*args,**kwargs)

	return

if __name__ == '__main__':

	build = sys.argv.pop(1) if len(sys.argv[1:]) > 1 else '~/code/tensor'
	paths = ['.','..',os.path.dirname(os.path.abspath(__file__)),os.path.abspath(os.path.expandvars(os.path.expanduser(build)))]
	sys.path.extend(paths)

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
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	from natsort import natsorted

	import mpmath as mp
	mp.dps = 100

	args = sys.argv[1:]

	kwargs = {}

	main(*args,**kwargs)
