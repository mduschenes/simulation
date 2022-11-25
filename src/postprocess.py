#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools
from copy import deepcopy
from functools import partial,wraps

import matplotlib
import matplotlib.pyplot as plt


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.utils import array,zeros,ones,arange,linspace,rand,sort,eig,argmax,argmin,maximum,difference,rand,scinotation,log,sqrt
from src.utils import is_naninf
from src.utils import nan
from src.dictionary import updater,getter
from src.fit import fit
from src.io import load,dump,join,split,glob,cd,exists,dirname

from src.plot import plot

from src.quantum import Unitary



defaults = {
'plot.None.eigenvalues.pdf': {
	"fig":{
		"set_size_inches":{"w":20,"h":10},
		"subplots_adjust":{},
		"tight_layout":{},
		"savefig":{"fname":"plot.None.eigenvalues.pdf","bbox_inches":"tight","pad_inches":0.2},
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
				"legend":{"title":True,"handles":True,"labels":True},
					"set_ylabel":{"ylabel":"left"}
				}
			}
		}

	},
'plot.noise.scale.M.min.pdf': {
	"fig":{
		"set_size_inches":{"w":9.5,"h":9.5},
		"subplots_adjust":{},
		"tight_layout":{},
		"savefig":{"fname":"plot.noise.scale.M.min.pdf","bbox_inches":"tight","pad_inches":0.2},
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
			"color":"viridis",
			},
		"fill_between":{
			"x":"noise.scale",
			"y1":"M",
			"y2":"M",
			"alpha":0.5,
			"color":'viridis',
			},
		"set_ylabel":{"ylabel":r'$M_{\gamma}$'},
		"set_xlabel":{"xlabel":r"$\gamma$"},
		"yaxis.offsetText.set_fontsize":{"fontsize":20},											
		"set_xscale":{"value":"log","base":10},
		"set_xnbins":{"nbins":6},
		"set_xticks":{"ticks":[1e-6,1e-5,1e-4,1e-3,1e-2]},
		"xaxis.set_major_formatter":{"ticker":{"LogFormatterMathtext":{}}},
		"xaxis.set_minor_locator":{"ticker":{"LogLocator":{"base":10.0,"subs":[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"numticks":100}}},
		"xaxis.set_minor_formatter":{"ticker":{"NullFormatter":{}}},		
		"set_yscale":{"value":"linear"},
		"set_ynbins":{"nbins":7},
		"set_ylim": {
				"ymin": -100,
				"ymax": 1200
			},
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
			"get_texts":{"va":"center","ha":"center","position":[0,15]},
			"prop": {"size": 12},
			"markerscale": 1.2,
			"handlelength": 3,
			"framealpha": 0.8,
			"loc": [0.12,0.87],
			"ncol": 1,
			"set_zorder":{"level":100},
			"set_label":True,
			}
		},
	"style":{
		"texify":None,
		"mplstyle":"config/plot.mplstyle",	
		"rcParams":{"font.size":20},
		"layout":{"nrows":1,"ncols":1,"index":1},
		"share": {
			"ax":{
				"legend":{"title":True,"handles":True,"labels":True},
					"set_ylabel":{"ylabel":"left"}
				}
			}
		}

	},
'M.objective.noise.scale': {
	"fig":{
		"set_size_inches":{"w":9.5,"h":9.5},
		"subplots_adjust":{},
		"tight_layout":{},
		"savefig":{"fname":"plot.M.objective.noise.scale.fit.pdf","bbox_inches":"tight","pad_inches":0.2},
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
		"set_xnbins":{"nbins":6},
		"set_xlim": {"xmin": 0,"xmax": 400},
		"set_yscale":{"value":"log","base":10},
		"set_ylim": {"ymin": 1e-5,"ymax": 1e1},
		"set_ynbins":{"nbins":5},
		"set_yticks":{"ticks":[1e-4,1e-2,1e0]},
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
			"loc": [0.7,0.02],
			"ncol": 1,
			"set_zorder":{"level":100},
			"set_label":None,
			}
		},
	"style":{
		"texify":None,
		"mplstyle":"config/plot.mplstyle",	
		"rcParams":{"font.size":20},
		"layout":{"nrows":1,"ncols":1,"index":1},
		"share": {
			"ax":{
				"legend":{"title":True,"handles":True,"labels":True},
					"set_ylabel":{"ylabel":"left"}
				}
			}
		}

	}
}

def process(path):

	plots = [
		'plot.noise.scale.M.min.pdf',
		# 'plot.None.eigenvalues.pdf',
		]
	

	for name in plots:
		print('Plotting :',name)		

		if name in ['plot.noise.scale.M.min.pdf']:

			with cd(path):

				file = 'metadata.json'
				hyperparameters = load(file)
					
				if hyperparameters is None:
					continue

				data = {}
				
				key = ['M.objective.noise.scale',-1]
				label = {'x':'noise.scale','y':'M','z':'objective'}
				values = getter(hyperparameters,key)

				attrs = set((attr for value in values for attr in value))
				for attr in attrs:
					data[attr] = [value[attr] for value in values]
					try:
						data[attr] = array(data[attr])
					except:
						pass


				X = data[label['x']]
				Y = data[label['y']]
				Z = data[label['z']]

				Xerr = data.get('%serr'%(label['x'])) if data.get('%serr'%(label['x'])) is not None else [None]*len(X)
				Yerr = data.get('%serr'%(label['y'])) if data.get('%serr'%(label['y'])) is not None else [None]*len(Y)
				Zerr = data.get('%serr'%(label['z'])) if data.get('%serr'%(label['z'])) is not None else [None]*len(Z)

				_X,_Y,_Z = [],[],[]
				_Xerr,_Yerr,_Zerr = [],[],[]

				indices = arange(len(X))[X>=1e-8]
				slices = slice(0,30,None)

				interpolate = 1

				try:
					x,y = [],[]
					for i,(x_,y_,z_,yerr_,zerr_) in enumerate(zip(X,Y,Z,Yerr,Zerr)):

						y_ = y_[slices]
						z_ = z_[slices]
						yerr_ = yerr_ if yerr_ is not None and not is_naninf(yerr_).all() else None
						zerr_ = zerr_[slices] if zerr_ is not None and not is_naninf(zerr_).all() else None

						_y = linspace(y_.min(),y_.max(),y_.size*20)
						_yerr = zeros(_y.shape)
						func = 'cubic'
						coef0 = None
						kwargs = {}

						_z,coef,_zerr,coefferr = fit(y_,z_,_x=_y,func=func,yerr=zerr_,coef0=coef0,uncertainty=True,**kwargs)	

						_Y.append(_y)
						_Yerr.append(_yerr)
						_Z.append(_z)
						_Zerr.append(_zerr)

						index = argmin(_z-_zerr)
						x.append(x_)
						y.append(_y[index])

					
					fig,ax = None,None

					settings = deepcopy(defaults[key[0]])

					options = {
						'ax':{
							'errorbar':[
								*[
								{
								**settings['ax']['errorbar'],
								'x':Y[i][slices],
								'y':Z[i][slices],
								'xerr':Yerr[i],
								'yerr':[(Z[i]*(1 - (Z[i]/(Z[i]+Zerr[i]))))[slices],Zerr[i][slices]],							
								'color': getattr(plt.cm,defaults[key[0]]['ax']['errorbar']['color'])(i/len(Z)),	
								'label':scinotation(X[i],decimals=1,scilimits=[-1,3]),
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
								'color': 'k',#getattr(plt.cm,defaults[key[0]]['ax']['errorbar']['color'])(i/len(Z)),	
								'marker':'',
								'linestyle':'-',
								'linewidth':2,
								'alpha':0.8,
								} for i in indices
								],
							],
							'fill_between':[
								# *[
								# {
								# **settings['ax']['fill_between'],	
								# 'x':Y[i][slices],
								# 'y':Z[i][slices],
								# 'yerr':[(Z[i]*(1 - (Z[i]/(Z[i]+Zerr[i]))))[slices],Zerr[i][slices]],
								# 'color': getattr(plt.cm,defaults[key[0]]['ax']['fill_between']['color'])(i/len(Z)),	
								# } for i in indices
								# ],
								*[
								{
								**settings['ax']['fill_between'],	
								'x':_Y[i],
								'y':_Z[i],
								'yerr':[(_Z[i]*(1 - (_Z[i]/(_Z[i]+_Zerr[i])))),_Zerr[i]],														
								'color': getattr(plt.cm,defaults[key[0]]['ax']['fill_between']['color'])(i/len(Z)),	
								'alpha':0.4,
								} for i in indices
								],
								]
							},
						}

					updater(settings,options)

					fig,ax = plot(settings=settings,fig=fig,ax=ax)

				except Exception as exception:
					print(exception)
					_x = X
					_y = Y
					_z = Z

					shape = _y.shape
					ndim = _y.ndim
					axis = 1
					slices = tuple((slices if ax == axis else arange(shape[ax]) for ax in range(ndim)))
					slices = argmin(_z[slices],axis=axis)
					slices = tuple((slices if ax == axis else arange(shape[ax]) for ax in range(ndim)))
					x = _x
					y = _y[slices]
				
				x = array(x)
				y = array(y)

				x = x[indices]
				y = y[indices]
				xerr = None
				yerr = None

				def func(x,*coef):
					y = coef[0]*(log(x))**1 + coef[1]
					return y

				_x = linspace(0.5*x.min(),1.75*x.max(),x.size*100)
				p = 2
				coef0 = array([-50,-50,1],dtype=float)[:p]
				kwargs = {
					'maxfev':2000,
					# 'bounds':array([[-100,-100,1][:p],[-20,-20,2][:p]],dtype=float)
				}

				_y,coef,_yerr,coefferr = fit(x,y,_x=_x,func=func,coef0=coef0,uncertainty=True,**kwargs)

				fig,ax = None,None

				settings = deepcopy(defaults[name])

				options = {
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
							# 'yerr':_yerr,
							'label':r'$\quad~~ M_{\gamma} = \alpha\log^{}{\gamma} + \beta$'+'\n'+r'$%s$'%(',~'.join([
								'%s = %s'%(z,scinotation(coef[i],decimals=2,scilimits=[-1,3],error=sqrt(coefferr[i][i]))) 
									for i,z in enumerate([r'\alpha',r'\beta',r'\chi'][:len(coef)])])),
							'color': getattr(plt.cm,defaults[name]['ax']['errorbar']['color'])(0.25),	
							'marker':None,
							'linestyle':'--',
							'zorder':-1,
							},												
							],
						'fill_between':{
							**settings['ax']['fill_between'],	
							'x':_x,
							'y1':_y - _yerr,
							'y2':_y + _yerr,
							'color': getattr(plt.cm,defaults[name]['ax']['fill_between']['color'])(0.25),	
							}
						},
					}

				updater(settings,options)

				fig,ax = plot(settings=settings,fig=fig,ax=ax)





		elif name in ['plot.None.eigenvalues.pdf']:

			files = {'hyperparameters':'settings.json','data':'data.hdf5','model':'model.pkl'}
			paths = glob(path,include='directory',recursive='**')
			paths = [subpath for subpath in paths if all(exists(join(subpath,files[file])) for file in files)]

			for path in paths:
				with cd(path):
				
					print('Loading: ',path,files)
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
										'fname':name,
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

						updater(settings[i],options)

					fig,ax = plot(settings=settings,fig=fig,ax=ax)

	return


def main(*args,**kwargs):

	process(*args,**kwargs)

	return

if __name__ == '__main__':

	arguments = ['path']

	args = argparser(arguments)

	main(*args,**args)

		