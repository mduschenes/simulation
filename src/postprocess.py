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
		"set_xticks":{"ticks":[1e-7,1e-6,1e-5,1e-4,1e-3]},
		"xaxis.set_major_formatter":{"ticker":{"LogFormatterMathtext":{}}},
		"xaxis.set_minor_locator":{"ticker":{"LogLocator":{"base":10.0,"subs":[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"numticks":100}}},
		"xaxis.set_minor_formatter":{"ticker":{"NullFormatter":{}}},		
		"set_yscale":{"value":"linear"},
		"set_ynbins":{"nbins":7},
		"set_ylim": {
                "ymin": 0,
                "ymax": 350
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
			"title_fontsize": 20,
			"get_title":{"ha":"center"},
			"get_texts":{"va":"center","ha":"center","position":[0,15]},
			"prop": {"size": 20},
			"markerscale": 1.2,
			"handlelength": 3,
			"framealpha": 0.8,
			"loc": [0.02,0.01],
			"ncol": 1,
			"set_zorder":{"level":100},
			"set_label":True,
			}
		},
	"style":{
		"texify":None,
		"mplstyle":"config/plot.mplstyle",	
		"rcParams":{"font.size":35},
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

			file = 'plot.settings.json'
			path = join(path,file)
			hyperparameters = load(path)
			data = {}
			
			keys = [
				['M.objective.noise.scale',(None,0,0),'ax','errorbar'],
				['M.objective.noise.scale',(None,0,0),'__property__']
				]
			for key in keys:
				values = getter(hyperparameters,key)
				attrs = set((attr for value in values for attr in value))
				for attr in attrs:
					data[attr] = [value[attr] for value in values]
					try:
						data[attr] = array(data[attr])
					except:
						pass

			shape = data['y'].shape
			ndim = data['y'].ndim
			axis = 1

			indices = argmin(data['y'],axis=axis)
			indices = tuple((indices if ax == axis else arange(shape[ax]) for ax in range(ndim)))

			x = data['noise.scale']
			y = data['x'][indices]
			xerr = None
			yerr = None

			# def func(x,*coef):
			def func(x,a,b):
				y = a*log(x) + b
				return y

			_x = linspace(0.75*x.min(),1.5*x.max(),x.size*20)
			coef0 = None

			_y,coef,_yerr,coefferr = fit(x,y,_x=_x,func=func,coef0=coef0,uncertainty=True)

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
						'label':None,
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
						'label':r'$\quad~~ M_{\gamma} = \alpha\log{\gamma} + \beta$'+'\n'+r'$\alpha = %s~,~\beta = %s$'%(
								tuple((scinotation(coef[i],decimals=2,scilimits=[-1,3],error=sqrt(coefferr[i][i])) for i in range(len(coef))))),						
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

		