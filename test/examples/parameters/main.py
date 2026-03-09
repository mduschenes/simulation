#!/usr/bin/env python

import os,sys

def setup(path=None,**kwargs):

	if path is None:
		return

	ext = 'hdf5'

	default = join('data.tmp',root=path,ext=ext)
	opts = dict(wrapper='pd')

	path = join('data',root=path,ext=ext)

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

def plot(path,*args,**kwargs):

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

	def process(data):

		settings = [
			Dict(
				name = 'M.parameters.noise.parameters.sample',
				variables = dict(
					x = 'M',
					y = 'parameters',
					colorbar = 'noise.parameters',
					legend = 'sample',
					sort = ['N','operator']
					),
				data = ['noise','env'],
				boolean = lambda data: data['M'].isin([2,4,8,16,32]) & data['N'].isin([10]),
				options = dict(
					groupby=dict(as_index=False,dropna=False)
					),
				fig = {},
				ax = {},
				index = lambda index=None,group=None,groupby=None,**kwargs:None,
				),
			Dict(
				name = 'noise.parameters.parameters.M.sample',
				variables = dict(
					x = 'noise.parameters',
					y = 'parameters',
					colorbar = 'M',
					legend = 'sample',
					sort = ['N','operator']
					),
				data = ['noise','env'],
				boolean = lambda data: data['M'].isin([2,4,8,16,32]) & data['N'].isin([10]),
				options = dict(
					groupby=dict(as_index=False,dropna=False)
					),
				fig = {},
				ax = {},
				index = lambda number=None,group=None,groupby=None,**kwargs:None,
				),
			Dict(
				name = 'M.parameters.noise.parameters.N',
				variables = dict(
					x = 'M',
					y = 'parameters',
					colorbar = 'noise.parameters',
					legend = 'N',
					sort = ['sample','operator']
					),
				data = ['noise','env'],
				boolean = lambda data: data['M'].isin([2,4,8,16,32]) & data['sample'].isin([1.0]),
				options = dict(
					groupby=dict(as_index=False,dropna=False)
					),
				fig = {},
				ax = {},
				index = lambda index=None,group=None,groupby=None,**kwargs:None,
				),
			Dict(
				name = 'noise.parameters.parameters.M.N',
				variables = dict(
					x = 'noise.parameters',
					y = 'parameters',
					colorbar = 'M',
					legend = 'N',
					sort = ['sample','operator']
					),
				data = ['noise','env'],
				boolean = lambda data: data['M'].isin([2,4,8,16,32]) & data['sample'].isin([1.0]),
				options = dict(
					groupby=dict(as_index=False,dropna=False)
					),
				fig = {},
				ax = {},
				index = lambda number=None,group=None,groupby=None,**kwargs:None,
				),
			]

		names = ['M.parameters.noise.parameters.sample','noise.parameters.parameters.M.sample']

		for setting in settings:

			if setting.name not in names:
				continue

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
								'font':{'text':130,'legend':95,'colorbar':130},
								'color':{'plot':'viridis','colorbar':'viridis'},
								'layout':[],
								},

							})

						options.update({

							'options': {
								**options['options'],
								**{'mplstyle': join(path,'plot',ext='mplstyle'),}
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
										'marker': 'o',
										'alpha':0.8,

										},
										{
										'marker': '^',
										'alpha':0.6,
										},
										{
										'marker': 's',
										'alpha':0.5,
										},
										{
										'marker': 'P',
										'alpha':0.4,
										},
										{
										'marker': 'h',
										'alpha':0.3,
										},
										{
										'marker': 'x',
										'alpha':0.2,
										},
										][len(groupings[setting.variables.legend])-1-groupings[setting.variables.legend].index(grouping[setting.variables.legend])]
										),

									'color':'%s_%s'%(
										options['options']['color']['plot'],
										(groupings[setting.variables.colorbar].index(grouping[setting.variables.colorbar]))/(len(groupings[setting.variables.colorbar])-1)),
									'markersize':60,
									'linestyle':'--',
									'linewidth':30,
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
									'loc': {
										('M','noise'):'lower right',
										('M','env'):'upper right',
										('noise.parameters','noise'):'lower right',
										('noise.parameters','env'):'upper left',
										}.get((setting.variables.x,setting.data[number]),'upper right'),
									'ncol':3,
									'title_fontsize': options['options']['font']['legend'],
									'prop':{'size': options['options']['font']['legend'],},
									'markerscale':1.25,
									'handlelength':3,
									'framealpha':1,
									'set_color':{'color':'gray'},
									'set_linewidth':{'w':16},
									},
								(True,'N'):{
									'title': '$\\textrm{Size}~~n$',
									'loc': {
										('M','noise'):'lower right',
										('M','env'):'upper left',
										('noise.parameters','noise'):'lower right',
										('noise.parameters','env'):'upper left',
										}.get((setting.variables.x,setting.data[number]),'upper right'),
									'ncol':1,
									'title_fontsize': options['options']['font']['legend'],
									'prop':{'size': options['options']['font']['legend'],},
									'markerscale':1.25,
									'handlelength':3,
									'framealpha':1,
									'set_color':{'color':'gray'},
									'set_linewidth':{'w':16},
									},
								}.get((all((groupings[attr].index(grouping[attr])==(len(groupings[attr])-1)) for attr in grouping),setting.variables.legend),{})
								),
								},

							'fig':{
								'set_size_inches':{'w':65,'h':45},
								'set_size_inches':{'w':45,'h':20},
								'subplots_adjust':{},
								'subplots_adjust': {'wspace': 0.5,'hspace': 0.1},
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
									'pad_inches':0.5,
									'pad_inches':0.2,
									},
							}
						})

						fig,ax = setting.fig.get(key),setting.ax.get(key)
						index = setting.index(number,group,groupby)

						fig,ax = plot(options,fig=fig,ax=ax,index=index)

						setting.fig[key] = fig
						setting.ax[key] = ax

						if all((groupings[attr].index(grouping[attr])==(len(groupings[attr])-1)) for attr in grouping):
							logger = Logger(file=None,verbose='info')
							logger(options['fig']['savefig']['fname'])

		return

	data = setup(path)

	if data is None:
		return

	process(data)

	return


def main(*args,**kwargs):

	path = str(args[0]) if len(args)>0 else '~/scratch/probability/distribution'

	args = ()

	kwargs = {}

	plot(path,*args,**kwargs)

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

	from natsort import natsorted

	args = sys.argv[1:]

	kwargs = {}

	main(*args,**kwargs)
