#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools
from copy import deepcopy
from functools import partial,wraps
import traceback

import matplotlib
import matplotlib.pyplot as plt


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.utils import gradient,array,zeros,ones,arange,linspace,logspace,rand,where,sort,eig,piecewise,interp,uncertainty 
from src.utils import mean,std,sem,argmax,argmin,maximum,minimum,difference,rand,scinotation,exp,exp10,log,log10,sqrt
from src.utils import is_naninf
from src.utils import nan,delim,null
from src.iterables import setter,getter,flatten
from src.fit import fit
from src.io import load,dump,join,split,glob,cd,exists,dirname

from src.plot import plot,AXIS,VARIANTS,FORMATS,ALL,OTHER,PLOTS

from src.quantum import Unitary



defaults = {
'None.eigenvalues': {
	"fig":{
		"set_size_inches":{"w":20,"h":10},
		"subplots_adjust":{},
		"tight_layout":{},
		"savefig":{"fname":None,"bbox_inches":"tight","pad_inches":0.2},
		"close":{}
		},
	"ax":{
		"plot":{
			"x":"",
			"y":"fisher.eigenvalues",
			"label":["N","architecture","M","iteration=@iteration.max@","r=@fisher.rank@"],
			"marker":"o",
			"markersize":4,
			"linestyle":"--",
			"linewidth":4,
			"color":"viridis",
			},
		"set_ylabel":{"ylabel":r'$\left|{\frac{\lambda}{\lambda_{\textrm{max}}}}\right|$'},
		"set_ylabel":{"ylabel":r"$\abs{\frac{\lambda}{\lambda_{\textrm{max}}}}$"},
		"set_xlabel":{"xlabel":r"$\textrm{Index}$"},
		"yaxis.offsetText.set_fontsize":{"fontsize":20},											
		"set_xnbins":{"nbins":6},
		"set_ynbins":{"nbins":4},		
		"set_yscale":{"value":"log","base":10},
		"yaxis.set_major_formatter":{"ticker":{"LogFormatterMathtext":{}}},
		"yaxis.set_minor_locator":{"ticker":{"LogLocator":{"base":10.0,"subs":[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"numticks":100}}},
		"yaxis.set_minor_formatter":{"ticker":{"NullFormatter":{}}},		
		"tick_params":[
			{"axis":"y","which":"major","length":8,"width":1},
			{"axis":"y","which":"minor","length":4,"width":0.5},
			{"axis":"x","which":"major","length":8,"width":1},
			{"axis":"x","which":"minor","length":4,"width":0.5}
			],
		"set_aspect":{"aspect":"auto"},
		"grid":{"visible":True,"which":"both","axis":"both","zorder":0},
		"legend":{
			"title":r'$\textrm{Sample, Rank}$',
			"title_fontsize": 20,
			"get_title":{"ha":"center"},	
			"prop": {"size": 20},
			"markerscale": 2,
			"handlelength": 3,
			"framealpha": 0.8,
			"loc": "lower right",
			"ncol": 1,
			"set_zorder":{"level":100}
			}			
		},
	"style":{
		"texify":None,
		"mplstyle":"plot.mplstyle",	
		"rcParams":{"font.size":35},
		"layout":{"nrows":1,"ncols":1,"index":1},
		"share": {
			"ax":{
				"legend":{"title":True,"handles":True,"set_label":True},
					"set_ylabel":{"ylabel":"left"}
				}
			}
		}

	},
'noise.scale.M.min': {
	"fig":{
		"set_size_inches":{"w":12,"h":12},
		"subplots_adjust":{},
		"tight_layout":{},
		"savefig":{"fname":None,"bbox_inches":"tight","pad_inches":0.2},
		"close":{}
		},
	"ax":{
		"errorbar":{
			"x":"noise.scale",
			"y":"M",
			"label":None,
			"marker":"o",
			"markersize":10,
			"linestyle":"--",
			"linewidth":4,
			"capsize":4,			
			"color":"viridis",
			},
		"fill_between":{
			"x":"noise.scale",
			"y":"M",
			"alpha":0.5,
			"color":'viridis',
			},
		"set_ylabel":{"ylabel":r'$M_{\gamma}$'},
		"set_xlabel":{"xlabel":r"$\gamma$"},
		"yaxis.offsetText.set_fontsize":{"fontsize":20},											
		"set_xscale":{"value":"log","base":10},
		"set_xnbins":{"nbins":6},
		"set_xlim": {"xmin": 5e-13,"xmax":5e0},
		"set_xlim": {"xmin": 5e-21,"xmax":5e0},
		"set_xlim": {"xmin": 5e-13,"xmax":5e0},
		"set_xticks":{"ticks":[1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0]},
		"set_xticks":{"ticks":[1e-20,1e-16,1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0]},
		"set_xticks":{"ticks":[1e-20,1e-16,1e-12,1e-8,1e-4,1e0]},
		"set_xticks":{"ticks":[1e-12,1e-8,1e-4,1e0]},
		"xaxis.set_major_formatter":{"ticker":{"LogFormatterMathtext":{}}},
		"xaxis.set_minor_locator":{"ticker":{"LogLocator":{"base":10.0,"subs":[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"numticks":100}}},
		"xaxis.set_minor_formatter":{"ticker":{"NullFormatter":{}}},		
		"set_yscale":{"value":"linear"},
		"set_ynbins":{"nbins":7},
		"set_ylim": {
				"ymin": -100,
				"ymin": -500,
				"ymax": 5100,
				"ymax": 2500
			},
		"set_yticks":{"ticks":[0,1000,2000,3000,4000,5000]},		
		"set_yticks":{"ticks":[0,1000,2000]},		
		"tick_params":[
			{"axis":"y","which":"major","length":8,"width":1},
			{"axis":"y","which":"minor","length":4,"width":0.5},
			{"axis":"x","which":"major","length":8,"width":1},
			{"axis":"x","which":"minor","length":4,"width":0.5}
			],
		"set_aspect":{"aspect":"auto"},
		"grid":{"visible":True,"which":"both","axis":"both","zorder":0},
		"legend":{
			"title_fontsize": 12,
			"get_title":{"ha":"center"},
			"get_texts":{"va":"center","ha":"center","position":[0,45]},
			"prop": {"size": 12},
			"markerscale": 1.2,
			"handlelength": 3,
			"framealpha": 0.8,
			"loc": [0.52,0.80],
			"ncol": 1,
			"set_zorder":{"level":100},
			"set_label":True,
			}
		},
	"style":{
		"texify":None,
		"mplstyle":"plot.mplstyle",	
		"rcParams":{"font.size":20},
		"layout":{"nrows":1,"ncols":1,"index":1},
		"share": {
			"ax":{
				"legend":{"title":True,"handles":True,"set_label":True},
					"set_ylabel":{"ylabel":"left"}
				}
			}
		}

	},
'M.objective.noise.scale': {
	"fig":{
		"set_size_inches":{"w":16,"h":9.5},
		"subplots_adjust":{},
		"tight_layout":{},
		"savefig":{"fname":None,"bbox_inches":"tight","pad_inches":0.2},
		"close":{}
		},
	"ax":{
		"errorbar":{
			"x":"M",
			"y":"objective",
			"label":None,
			"marker":"o",
			"markersize":10,
			"linestyle":"--",
			"capsize":4,			
			"linewidth":4,
			"color":"viridis",
			},
		"fill_between":{
			"x":"noise.scale",
			"alpha":0.5,
			"color":'viridis',
			},			
		"set_ylabel":{"ylabel":r'$\textrm{Infidelity}$'},
		"set_xlabel":{"xlabel":r"$M$"},
		"yaxis.offsetText.set_fontsize":{"fontsize":20},											
		"set_xscale":{"value":"linear"},
		"set_xnbins":{"nbins":9},
		"set_xlim": {"xmin": 0,"xmax": 6100},
		"set_xticks":{"ticks":[0,1000,2000,3000,4000,5000,6000]},
		"set_yscale":{"value":"linear"},
		"set_yscale":{"value":"log","base":10},
		"set_ylim": {"ymin": 1e-5,"ymax": 1e-1},
		"set_ylim": {"ymin": 1e-13,"ymax": 5e2},
		"set_ylim": {"ymin": 5e-9,"ymax": 5e2},
		"set_ynbins":{"nbins":5},
		"set_yticks":{"ticks":[1e-4,1e-3,1e-2,1e-1]},
		"set_yticks":{"ticks":[1e-12,1e-8,1e-6,1e-4,1e-2,1e0,1e2]},
		"set_yticks":{"ticks":[1e-8,1e-6,1e-4,1e-2,1e0,1e2]},
		"yaxis.set_major_formatter":{"ticker":{"LogFormatterMathtext":{}}},
		"yaxis.set_minor_locator":{"ticker":{"LogLocator":{"base":10.0,"subs":[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"numticks":100}}},
		"yaxis.set_minor_formatter":{"ticker":{"NullFormatter":{}}},		
		"tick_params":[
			{"axis":"y","which":"major","length":8,"width":1},
			{"axis":"y","which":"minor","length":4,"width":0.5},
			{"axis":"x","which":"major","length":8,"width":1},
			{"axis":"x","which":"minor","length":4,"width":0.5}
			],
		"set_aspect":{"aspect":"auto"},
		"grid":{"visible":True,"which":"both","axis":"both","zorder":0},
		"legend":{
			"title_fontsize": 20,
			"set_title":r"$\gamma$",
			"prop": {"size": 20},
			"markerscale": 1.2,
			"handlelength": 3,
			"framealpha": 0.8,
			"loc": [0.125,0.82],
			"ncol": 5,
			"set_zorder":{"level":100},
			"set_label":None,
			}
		},
	"style":{
		"texify":None,
		"mplstyle":"plot.mplstyle",	
		"rcParams":{"font.size":20},
		"layout":{"nrows":1,"ncols":1,"index":1},
		"share": {
			"ax":{
				"legend":{"title":True,"handles":True,"set_label":True},
					"set_ylabel":{"ylabel":"left"}
				}
			}
		}

	}
}

def postprocess(path,**kwargs):
	'''
	Postprocess data
	Args:
		path (str): Path to postprocess
		kwargs (dict): Additional keyword arguments
	'''

	plots = [
		'noise.scale.M.min',
		# 'None.eigenvalues',
		]
	

	for name in plots:
		print('Plotting :',name)		

		if name in ['noise.scale.M.min']:

			with cd(path):

				file = 'metadata.json'
				hyperparameters = load(file)
					
				if hyperparameters is None:
					continue

				data = {}
				
				key = ['M.objective.noise.scale','None','ax','errorbar']
				label = {'x':'noise.scale','y':'M','z':'objective'}
				axes = AXIS
				other = OTHER
				values = list(flatten(getter(hyperparameters,key)))
				slices = slice(None,None,None)

				for axis in label:
					ax = [ax for ax in axes if all(((ax in value[other]) and (value[other][ax]['axis'] == label[axis])) for value in values)]

					if ax:
						ax = ax[0]
						data[label[axis]] = [value[ax][slices] for value in values]
						data['%serr'%(label[axis])] = [value['%serr'%(ax)][slices] if '%serr'%(ax) in value else None for value in values]
					else:
						data[label[axis]] = [value[other][label[axis]] for value in values]
						data['%serr'%(label[axis])] = [None for value in values]

					# data[label[axis]] = [value if value not in ['None',None,nan] else 1e-20 for value in data[label[axis]]]

					try:
						data[label[axis]] = array(data[label[axis]])
						data['%serr'%(label[axis])] = array(data['%serr'%(label[axis])])

					except Exception as exception:
						pass

				slices = range(4,len(data[label['y']])-5)
				slices = [4,6,8,9]#range(4,len(data[label['y']])-5)
				# slices = range(len(data[label['y']])-3)

				X = array([data['%s'%(label['x'])][i] for i in slices])
				Y = array([data['%s'%(label['y'])][i] for i in slices])
				Z = array([data['%s'%(label['z'])][i] for i in slices])


				Xerr = array([data['%serr'%(label['x'])][i] for i in slices])
				Yerr = array([data['%serr'%(label['y'])][i] for i in slices])
				Zerr = array([data['%serr'%(label['z'])][i] for i in slices])

				_X,_Y,_Z = [],[],[]
				_Xerr,_Yerr,_Zerr = [],[],[]

				try:
					x,y,z,xerr,yerr,zerr = [],[],[],None,[],[]
					coefs,coeferrs,rs = [],[],[]
					indices,indexes,slices = [],[],[]
					for i,(x_,y_,z_,yerr_,zerr_) in enumerate(zip(X,Y,Z,Yerr,Zerr)):

						indices.append(i)

						slices = slice(0,None,None)
						
						y_ = y_[slices] if y_ is not None else None
						z_ = z_[slices] if z_ is not None else None

						yerr_ = yerr_[slices] if yerr_ is not None and not is_naninf(yerr_).all() else None
						zerr_ = zerr_[slices] if zerr_ is not None and not is_naninf(zerr_).all() else None

						_x = x_
						_n = y_.size*20
						_y = linspace(y_.min(),y_.max(),_n)
						_z = zeros(_n)
						_yerr = zeros(_n)
						_zerr = zeros(_n)

						func = [
								(lambda coef,x: coef[0] + coef[1]*x),
								(lambda coef,x: coef[0] + coef[1]*x),
								]
						coef = [array([1.0,1.0]),array([1.0,1.0])]
						bounds = [y_[argmin(z_)]]
						kwargs = {}
						
						preprocess = [
							lambda x,y,coef: (x if x is not None else None,log(y) if y is not None else None,coef if coef is not None else None),
							lambda x,y,coef: (log(x) if x is not None else None,log(y) if y is not None else None,coef if coef is not None else None),
							]
						postprocess = [
							lambda x,y,coef: (x if x is not None else None,exp(y) if y is not None else None,coef if coef is not None else None),
							lambda x,y,coef: (exp(x) if x is not None else None,exp(y) if y is not None else None,coef if coef is not None else None),
							]

						# func = 'linear'
						# coef = None
						# kwargs = {'smooth':0}
						# preprocess = None
						# postprocess = None
						print(coef)

						_func,_z,_coef,_zerr,_coeferr,_r = fit(
							y_,z_,
							_x=_y,_y=_z,
							func=func,
							xerr=yerr_,yerr=zerr_,
							coef=coef,coefframe=True,
							preprocess=preprocess,postprocess=postprocess,
							bounds=bounds,kwargs=kwargs)	

						# _z,_coef,_zerr,_coeferr,_r = z_,coef,zerr_[slices],None,1
						index = argmin(_z)
						indexerr = (argmin(_z+_zerr) + argmin(_z-_zerr))//2

						_yerr = _yerr.at[index].set(sum(abs(_y[argmin(_z)+k] - _y[argmin(_z)]) for k in [1,-1])/2)

						print(coef)
						print(_coef)
						print(_x,_r)
						print()

						_X.append(_x)
						_Y.append(_y)
						_Yerr.append(_yerr)
						_Z.append(_z)
						_Zerr.append(_zerr)

						x.append(_x)
						y.append(_y[index])
						z.append(_z[index])
						# yerr.append(_yerr[index])
						yerr.append(sum(abs(_y[argmin(_z)+k] - _y[argmin(_z)]) for k in [1,-1])/2)
						zerr.append(_zerr[index])
						indexes.append(index)
						coefs.append(_coef)
						coeferrs.append(_coeferr)
						rs.append(_r)

					fig,ax = None,None

					settings = deepcopy(defaults[key[0]])
					options = {
						'fig':{
							'savefig':{
								**settings['fig']['savefig'],
								'fname':join(delim.join(['plot',name]),ext='pdf'),
								}
						},
						'ax':{
							'errorbar':[
								*[
								{
								**settings['ax']['errorbar'],
								'x':Y[i],
								'y':Z[i],
								'xerr':Yerr[i] if Yerr.ndim == 1 else Yerr[i],
								'yerr':Zerr[i],	
								'color': getattr(plt.cm,defaults[key[0]]['ax']['errorbar']['color'])(i/len(indices)),	
								'label':scinotation(X[i],decimals=1,scilimits=[0,3]),
								'marker':'o',
								'linestyle':'',
								'alpha':0.7,
								} for i in indices
								],
								*[
								{
								**settings['ax']['errorbar'],
								'x':_Y[i],
								'y':_Z[i],
								# 'yerr':[(_Z[i]*(1 - (_Z[i]/(_Z[i]+_Zerr[i])))),_Zerr[i]],							
								'color': 'k',#getattr(plt.cm,defaults[key[0]]['ax']['errorbar']['color'])(i/len(indices)),	
								'marker':'',
								'linestyle':'-',
								'linewidth':2,
								'alpha':0.8,
								} for i in indices
								],
								*[
								{
								'x':[y[i] for i in indices],
								'y':[z[i] for i in indices],
								'xerr':[yerr[i] for i in indices],
								'yerr':[zerr[i] for i in indices],
								# 'yerr':[(_Z[i]*(1 - (_Z[i]/(_Z[i]+_Zerr[i])))),_Zerr[i]],							
								'color': 'k',#getattr(plt.cm,defaults[key[0]]['ax']['errorbar']['color'])(i/len(indices)),	
								'ecolor':'viridis',
								'marker':'o',
								'markersize':20,
								'markerfacecolor':"None",
								"linestyle":"--",
								"capsize":4,			
								"linewidth":4,
								'alpha':0.8,
								}
								],								
							],
							'fill_between':[
								# *[
								# {
								# **settings['ax']['fill_between'],	
								# 'x':Y[i],
								# 'y':Z[i],
								# 'yerr':[(Z[i]*(1 - (Z[i]/(Z[i]+Zerr[i])))),Zerr[i]],
								# 'color': getattr(plt.cm,defaults[key[0]]['ax']['fill_between']['color'])(i/len(indices)),	
								# } for i in indices
								# ],
								*[
								{
								**settings['ax']['fill_between'],	
								'x':_Y[i],
								'y':_Z[i],
								'yerr':_Zerr[i],
								'color': getattr(plt.cm,defaults[key[0]]['ax']['fill_between']['color'])(i/len(indices)),	
								'alpha':0.4,
								} for i in indices
								],
								]
							},
						}

					setter(settings,options)

					fig,ax = plot(settings=settings,fig=fig,ax=ax)

				except Exception as exception:
					raise


					_x = X
					_y = Y
					_z = Z

					shape = _y.shape
					ndim = _y.ndim
					axis = 1
					indices = arange(len(_x))
					slices = slice(None,None,None)
					slices = tuple((slices if ax == axis else arange(shape[ax]) for ax in range(ndim)))
					slices = argmin(_z[slices],axis=axis)
					slices = tuple((slices if ax == axis else arange(shape[ax]) for ax in range(ndim)))
					x = _x
					y = _y[slices]

				
				x = array(x)
				y = array(y)
				xerr = array(xerr) if xerr is not None and not all([z is None for z in xerr]) else None
				yerr = array(yerr) if yerr is not None and not all([z is None for z in yerr]) else None
				slices = arange(len(x))[(x>=1e-28) & (x<=1e3) & (x != 1e0)]

				def func(coef,x):
					y = coef[0] + coef[1]*(x)
					return y

				_n = x.size*10
				_x = logspace(int(log10(x.min()))-2,int(log10(x.max())),_n)
				_y = zeros(_n)
				p = 2
				coef = array([1.0,-1.0])[:p]
				kwargs = {
					'maxfev':200000,
				}
				preprocess = lambda x,y,coef: (log10(x) if x is not None else None,y if y is not None else None,coef if coef is not None else None)
				postprocess = lambda x,y,coef: (exp10(x) if x is not None else None,y if y is not None else None,coef if coef is not None else None)

				_func,_y,_coef,_yerr,_coeferr,_r = fit(
					x[slices],y[slices],
					_x=_x,_y=_y,
					func=func,coef=coef,
					yerr=yerr[slices] if yerr is not None else yerr,
					xerr=xerr[slices] if xerr is not None else xerr,
					preprocess=preprocess,postprocess=postprocess,
					coefframe=True,kwargs=kwargs)

				print(_y)
				print(func(_coef,log10(_x)))
				print(_coef[0]+_coef[1]*log10(_x))

				fig,ax = None,None

				settings = deepcopy(defaults[name])

				options = {
					'fig':{
						'savefig':{
							**settings['fig']['savefig'],
							'fname':join(delim.join(['plot',name,'fit']),ext='pdf'),
							}
						},
					'ax':{
						'errorbar':[
							{
							**settings['ax']['errorbar'],
							'x':x,
							'y':y,
							'xerr':xerr,
							'yerr':yerr,
							'legend':None,
							'color': getattr(plt.cm,defaults[name]['ax']['errorbar']['color'])(0.5),	
							'marker':'o',
							'linestyle':'',
							'alpha':0.7,
							},
							{
							**settings['ax']['errorbar'],						
							'x':_x,
							'y':_y,
							'yerr':_yerr,
							# 'label':r'$\quad~~ M_{\gamma} = \alpha\log{\gamma} + \beta$'+'\n'+r'$%s$'%(',~'.join([
							# 'label':r'$\quad~~ M_{\gamma} = \alpha{\gamma}^{-\chi} + \beta$'+'\n'+r'$%s$'%(',~'.join([
							# 'label':r'$\quad~~ M_{\gamma} = {\gamma}^{-\alpha}$'+'\n'+r'$%s$'%(',~'.join([
							# 'label':r'$\quad~~ M_{\gamma} = {(\gamma-\beta)}^{-\alpha}$'+'\n'+r'$%s$'%(',~'.join([
							# 'label':r'$\quad~~ M_{\gamma} = {\gamma}^{-\alpha}$'+'\n'+r'$%s$'%(',~'.join([
							'label':(
								r'$\quad~~ M_{\gamma} = \beta\log{\gamma} + {\alpha}$' + '\n' + 
								r'$%s$'%('\n'.join([
								'%s = %s'%(z,scinotation(_coef[i],decimals=2,scilimits=[-1,4],error=sqrt(_coeferr[i][i]) if _coeferr is not None else None)) 
									for i,z in enumerate([r'\alpha',r'\beta',r'\chi',r'\eta'][:len(_coef)])])) + '\n' +
								r"$\gamma_{0} = 10^{-\alpha/\beta} = 10^{-%s}"%(scinotation(log10(exp(1))*(_coef[0]/_coef[1]),decimals=2,scilimits=[-1,4],error=log10(exp(1))*uncertainty(*(_coef[i] for i in [0,1]),*(sqrt(_coeferr[i][i]) for i in [0,1]),'/')[1] if _coeferr is not None else None)) + '\n' +
								r'$%s$'%('r^2 = %s'%(scinotation(_r,decimals=4,scilimits=[-1,4])))
								),
							'color': getattr(plt.cm,defaults[name]['ax']['errorbar']['color'])(0.25),	
							'marker':None,
							'linestyle':'--',
							'zorder':-1,
							},												
							],
						'fill_between':{
							**settings['ax']['fill_between'],	
							'x':_x,
							'y':_y,
							'yerr':_yerr,
							'color': getattr(plt.cm,defaults[name]['ax']['fill_between']['color'])(0.25),	
							}
						},
					}

				setter(settings,options)

				fig,ax = plot(settings=settings,fig=fig,ax=ax)





		elif name in ['None.eigenvalues']:

			raise("TODO: Fix model init")

			files = {'hyperparameters':'settings.json','data':'data.hdf5','model':'model.pkl'}
			paths = glob(path,include='directory',recursive='**')
			paths = [subpath for subpath in paths if all(exists(join(subpath,files[file])) for file in files)]

			for path in paths:
				with cd(path):
				
					hyperparameters = load(files['hyperparameters'])

					if hyperparameters is None:
						continue

					hyperparameters['sys']['path']['data']['log'] = None
					U = Unitary(**hyperparameters['data'],**hyperparameters['model'],hyperparameters=hyperparameters)
					U.load()

					func = U.__func__
					grad = U.__grad__
					hess = U.__hessian__
					fisher = U.__fisher__
					parameters = U.parameters

					shape = parameters.shape
					bounds = [-1,1]
					seed = 123321
					random = 'uniform'

					funcs = [fisher]
					n = 5
					m = len(funcs)
					params = [parameters,*rand(shape=(n,*shape),bounds=bounds,key=seed,random=random)]

					fig,ax = None,None
					settings = {i: deepcopy(defaults[name]) for i in range(m)}

					for i in range(m):

						func = funcs[i]
						options = deepcopy(defaults[name])

						for p,parameters in enumerate(params):

							y = func(parameters)
							y = sort(abs(eig(y,hermitian=True)))[::-1]
							y = y/maximum(y)
							x = arange(y.size)

							attrs = {
								'rank': argmax(abs(difference(y)/y[:-1]))+1 if y[argmax(abs(difference(y)/y[:-1]))+1]<1e-4 else y.size
								}

							y = y[:U.g*2]
							x = x[:U.g*2]

							options = {
								'fig':{
									'suptitle':{
										't': r'$M = %d~~~~\epsilon = %s$'%(U.M,scinotation(U.hyperparameters['optimize']['track']['value'][-1],decimals=1)),
										},
									'savefig':{
										**defaults[name]['fig']['savefig'],
										'fname':join(delim.join(['plot',name]),ext='pdf'),
									}
									},
								'ax':{
									'plot':[
										*([] if isinstance(options['ax']['plot'],dict) else options['ax']['plot']), 
										{
										**defaults[name]['ax']['plot'],
										'x':x,
										'y':y,
										'label': r'$%s, r = %d$'%('*' if p==0 else r'%d'%(p),attrs['rank']),
										'alpha':0.8 if p==0 else 0.8,
										'marker': 'o' if p==0 else 'o',
										'markersize':7 if p==0 else 5,
										'linestyle': '-' if p==0 else '--',
										'linewidth': 5 if p==0 else 4,
										'color': 'k' if p==0 else getattr(plt.cm,defaults[name]['ax']['plot']['color'])((p-1)/(len(params)-1)),
										'zorder':3 if p==0 else 4,
										}],
									'set_xlabel': {
										'xlabel': r'$\textrm{%s Index}$'%(['Fisher','Hessian'][i]),
										},
									"set_ylim":{"ymax":1e1},
									"set_ynbins":{"nbins":6},
									},
								'style': {
									"layout":{"nrows":1,"ncols":m,"index":i+1},
									}
								}

						setter(settings[i],options)

					fig,ax = plot(settings=settings,fig=fig,ax=ax)

	return


def main(*args,**kwargs):

	postprocess(*args,**kwargs)

	return

if __name__ == '__main__':

	arguments = ['path']

	args = argparser(arguments)

	main(*args,**args)

		