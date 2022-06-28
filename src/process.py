#!/usr/bin/env python

# Import python modules
import os,sys,itertools,warnings
import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Logging
import logging
conf = 'logging.conf'
try: 
	logging.config.fileConfig(conf,disable_existing_loggers=False) 
except: 
	pass
logger = logging.getLogger(__name__)


# Import user modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..','../../lib']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from lib.utils.io import setup,load,dump,_load
from lib.utils.plot import plot

def texify(string,usetex=True):
	strings = {
		'N':r'$N$'%(),
		'h':r'$h/J$'%(),
		'U':r'$U$'%(),
		'order':r'$\sigma/NJ$'%(),
		# 'energy':r'$\epsilon - \epsilon_0 / N$'%(),
		'energy':r'$\epsilon/NJ$'%(),
		'gap':r'$\Delta/J$'%(),
		'entanglement':r'$S$'%(),
		'partition':r'$L/N$'%(),
	}
	if not isinstance(string,str):
		string = str(string)

	default = r'$%s$'%(string.replace('$',''))

	string = strings.get(string,default)

	if not usetex:
		string = string.replace('$','')

	return string


def fit(x,y,_x=None,func=None,wrapper=None,coef0=None,intercept=True):

	x[np.isnan(x) | np.isinf(x)] = 0

	if wrapper is None:
		wrapper = lambda x,y,*coef: y

	if func is None:
		if intercept:
			x = np.array([x,np.ones(x.size)]).T
		else:
			x = np.array([x]).T
		if _x is None:
			_x = x
		elif intercept:
			_x = np.array([_x,np.ones(_x.size)]).T
		else:
			_x = np.array([_x]).T
		try:
			coef = np.linalg.lstsq(x,y)[0] + 0.0
			_y = _x.dot(coef)
		except:
			_y = y
			coef = np.zeros(_x.shape[1])
	else:
		if _x is None:
			_x = x
		try:
			coef = sp.optimize.curve_fit(func,x,y,p0=coef0)[0] + 0.0
			_y = func(_x,*coef)
		except:
			coef = coef0
			_y = np.zeros(_x.shape[0])

	if coef is not None:
		_y = wrapper(_x,_y,*coef)
	else:
		coef = np.zeros(3)
	return _y,coef


def main(args):

	path = args[0]
	directory = os.path.dirname(path)
	wr = 'r'
	default = {}
	kwargs = {}

	data = load(path,wr=wr,default=default,**kwargs)

	attrs = list(set([attr for name in data for attr in data[name]]))

	parameters = {attr: list(sorted(set([data[name][attr] 
					for name in data if attr in data[name] and isinstance(data[name][attr],(int,np.integer,float,np.float))]))) 
			for attr in attrs}
	values = {attr: [data[name][attr] for name in data if attr in data[name] and not isinstance(data[name][attr],(int,np.integer,float,np.float))] 
			for attr in attrs}
	parameters = {attr: parameters[attr] for attr in parameters if len(parameters[attr])>0}
	values = {attr: values[attr] for attr in values if len(values[attr])>0}
	
	sizes = {attr: min([data[name][attr].shape[0] for name in data]) if attr in values else 1 for attr in attrs}

	keys = [
		('h','energy',('N',)),
		('h','order',('N',)),
		('N','gap',('h','J',)),
		('partition','entanglement',('N','h',)),
		# ('N','energy',('h','J',)),
		# ('N','entanglement',('h','J')),
	]

	variables = {}

	for key in keys:
		variables[key] = {}
		for index in range(sizes[key[1]]) if key[1] in sizes else []:
			variables[key][index] = {}
			for i in itertools.product(*[parameters[l] for l in key[2]]):
				params = dict(zip(key[2],i))
				names = [name for name in data if all(data[name][p] == params[p] for p in params)]
				unique = {p: [name for name in names if all([data[name][k] == j for k,j in zip(parameters,p)])]
						  for p in itertools.product(*[parameters[k] for k in parameters])
						  if all([params[k] == dict(zip(parameters,p))[k] for k in params]) and len([name for name in names if all([data[name][k] == j for k,j in zip(parameters,p)])]) > 0
						  }
				variables[key][index][i] = {}

				if key == ('N','entanglement',('h','J',)):
					attr = 'partition'
					name = names[0]
					indexes = {name: np.where(data[name][attr][index]==0.5)[0][0] for name in names}
					xy = {
						'x':np.array([np.array([data[name][key[0]] for name in unique[p]]).mean(0) for p in unique]),
						'y':np.array([np.array([data[name][key[1]][:,indexes[name]] for name in unique[p]]).mean(0) for p in unique]),
						'xerr':np.array([np.array([data[name][key[0]] for name in unique[p]]).std(0) for p in unique]),
						'yerr':np.array([np.array([data[name][key[1]][:,indexes[name]] for name in unique[p]]).std(0) for p in unique]),
						}											
				else:
					xy = {
						'x':np.array([np.array([data[name][key[0]] for name in unique[p]]).mean(0) for p in unique]),
						'y':np.array([np.array([data[name][key[1]] for name in unique[p]]).mean(0) for p in unique]),
						'xerr':np.array([np.array([data[name][key[0]] for name in unique[p]]).std(0) for p in unique]),
						'yerr':np.array([np.array([data[name][key[1]] for name in unique[p]]).std(0) for p in unique]),
						}



				variables[key][index][i]['argsort'] = np.argsort(xy['x'] if xy['x'].ndim < 2 else xy['x'][:,index,0] if key[0] in attrs else np.arange(len(names)))
				variables[key][index][i]['x'] = ((xy['x'] if xy['x'].ndim < 2 else xy['x'][:,index] if key[0] in attrs else np.arange(len(names)))[variables[key][index][i]['argsort']]).reshape(-1)
				variables[key][index][i]['y'] = ((xy['y'] if xy['y'].ndim < 2 else xy['y'][:,index] if key[1] in attrs else np.arange(len(names)))[variables[key][index][i]['argsort']]).reshape(-1)
				variables[key][index][i]['xerr'] = ((xy['xerr'] if xy['xerr'].ndim < 2 else xy['xerr'][:,index] if key[0] in attrs else np.arange(len(names)))[variables[key][index][i]['argsort']]).reshape(-1)
				variables[key][index][i]['yerr'] = ((xy['yerr'] if xy['yerr'].ndim < 2 else xy['yerr'][:,index] if key[1] in attrs else np.arange(len(names)))[variables[key][index][i]['argsort']]).reshape(-1)#/variables[key][index][i]['x']


				variables[key][index][i]['xfunc'] = ({
						('h','energy',('N',)): variables[key][index][i]['x'] if (key == ('h','energy',('N',))) else variables[key][index][i]['x'],
						('h','order',('N',)): variables[key][index][i]['x'] if (key == ('h','order',('N',))) else variables[key][index][i]['x'],
						('N','gap',('h','J',)): 1/variables[key][index][i]['x'] if (key == ('N','gap',('h','J',))) else variables[key][index][i]['x'],
						('partition','entanglement',('N','h',)): variables[key][index][i]['x'] if (key == ('partition','entanglement',('N','h',))) else variables[key][index][i]['x'],
						('N','energy',('h','J',)): variables[key][index][i]['x'] if (key == ('N','energy',('h','J',))) else variables[key][index][i]['x'],
						('N','entanglement',('h','J',)): variables[key][index][i]['x'] if (key == ('N','entanglement',('h','J',))) else variables[key][index][i]['x'],
						}[key])

				variables[key][index][i]['yfunc'] = ({
						('h','energy',('N',)): variables[key][index][i]['y'] if (key == ('h','energy',('N',))) else variables[key][index][i]['y'],
						('h','order',('N',)): variables[key][index][i]['y'] if (key == ('h','order',('N',))) else variables[key][index][i]['y'],
						('N','gap',('h','J',)): variables[key][index][i]['y'] if (key == ('N','gap',('h','J',))) else variables[key][index][i]['y'],
						('partition','entanglement',('N','h',)): variables[key][index][i]['y'] if (key == ('partition','entanglement',('N','h',))) else variables[key][index][i]['y'],
						('N','energy',('h','J',)): variables[key][index][i]['y'] if (key == ('N','energy',('h','J',))) else variables[key][index][i]['y'],
						('N','entanglement',('h','J',)): variables[key][index][i]['y'] if (key == ('N','entanglement',('h','J',))) else variables[key][index][i]['y'],
						}[key])				

				variables[key][index][i]['xfit'] = ({
						('h','energy',('N',)): variables[key][index][i]['xfunc'] if (key == ('h','energy',('N',))) else variables[key][index][i]['xfunc'],
						('h','order',('N',)): variables[key][index][i]['xfunc'] if (key == ('h','order',('N',))) else variables[key][index][i]['xfunc'],
						('N','gap',('h','J',)): np.linspace(min(variables[key][index][i]['xfunc']),max(variables[key][index][i]['xfunc']),10) if (key == ('N','gap',('h','J',))) else variables[key][index][i]['xfunc'],
						('partition','entanglement',('N','h',)): variables[key][index][i]['xfunc'] if (key == ('partition','entanglement',('N','h',))) else variables[key][index][i]['xfunc'],
						('N','energy',('h','J',)): variables[key][index][i]['xfunc'] if (key == ('N','energy',('h','J',))) else variables[key][index][i]['xfunc'],
						('N','entanglement',('h','J',)): variables[key][index][i]['xfunc'] if (key == ('N','entanglement',('h','J',))) else variables[key][index][i]['xfunc'],
						}[key])
				variables[key][index][i]['yfit'] = ({
						('h','energy',('N',)): -2/np.pi*np.abs(1-variables[key][index][i]['xfunc'])*sp.special.ellipe(
												((-4*variables[key][index][i]['xfunc'])/(1-variables[key][index][i]['xfunc'])**2)) if (key == ('h','energy',('N',))) else variables[key][index][i]['yfunc'],
						('h','order',('N',)): variables[key][index][i]['yfunc'] if (key == ('h','order',('N',))) else variables[key][index][i]['yfunc'],
						# ('N','gap',('h','J',)): np.exp(fit(
						# 	 	np.log(variables[key][index][i]['xfunc']),
						# 	 	np.log(variables[key][index][i]['yfunc']/variables[key][index][i]['xfunc']),
						# 	 	variables[key][index][i]['xfit'],
						# 	 	intercept=True)[0]) if (key == ('N','gap',('h','J',))) else variables[key][index][i]['yfunc'],						
						('N','gap',('h','J',)): fit(
							 	variables[key][index][i]['xfunc'],
							 	(variables[key][index][i]['yfunc']/variables[key][index][i]['xfunc']),
							 	variables[key][index][i]['xfit'],			
							 	lambda x,a,b,c: b*x**a + c,		
							 	lambda x,y,a,b,c:	x*y,	 	
							 	[1,1,0],
							 	intercept=False)[0] if (key == ('N','gap',('h','J',))) else variables[key][index][i]['yfunc'],
						# ('N','gap',('h','J',)): fit(
						# 	 	variables[key][index][i]['xfunc'],
						# 	 	(variables[key][index][i]['yfunc']/variables[key][index][i]['xfunc']),
						# 	 	variables[key][index][i]['xfit'],							 	
						# 	 	intercept=True)[0]*variables[key][index][i]['xfit'] if (key == ('N','gap',('h','J',))) else variables[key][index][i]['yfunc'],
						('partition','entanglement',('N','h',)): fit(
							 	np.log(params['N']/np.pi*np.sin(np.pi*variables[key][index][i]['xfunc']))/3,
							 	variables[key][index][i]['yfunc'],
							 	intercept=True)[0] if (key == ('partition','entanglement',('N','h',))) else variables[key][index][i]['yfunc'],
						('N','energy',('h','J',)): variables[key][index][i]['yfunc'] if (key == ('N','energy',('h','J',))) else variables[key][index][i]['yfunc'],
						('N','entanglement',('h','J',)): variables[key][index][i]['yfunc'] if (key == ('N','entanglement',('h','J',))) else variables[key][index][i]['yfunc'],
						}[key])
	
				variables[key][index][i]['xcoef'] = ({
						('h','energy',('N',)): variables[key][index][i]['xfunc'] if (key == ('h','energy',('N',))) else variables[key][index][i]['xfunc'],
						('h','order',('N',)): variables[key][index][i]['xfunc'] if (key == ('h','order',('N',))) else variables[key][index][i]['xfunc'],
						('N','gap',('h','J',)): variables[key][index][i]['xfit'] if (key == ('N','gap',('h','J',))) else variables[key][index][i]['xfunc'],
						('partition','entanglement',('N','h',)): variables[key][index][i]['xfunc'] if (key == ('partition','entanglement',('N','h',))) else variables[key][index][i]['xfunc'],
						('N','energy',('h','J',)): variables[key][index][i]['xfunc'] if (key == ('N','energy',('h','J',))) else variables[key][index][i]['xfunc'],
						('N','entanglement',('h','J',)): variables[key][index][i]['xfunc'] if (key == ('N','entanglement',('h','J',))) else variables[key][index][i]['xfunc'],
						}[key])

				variables[key][index][i]['ycoef'] = ({
						('h','energy',('N',)): variables[key][index][i]['yfunc'] if (key == ('h','energy',('N',))) else variables[key][index][i]['yfunc'],
						('h','order',('N',)): variables[key][index][i]['yfunc'] if (key == ('h','order',('N',))) else variables[key][index][i]['yfunc'],
						('N','gap',('h','J',)): fit(
							 	variables[key][index][i]['xfunc'],
							 	(variables[key][index][i]['yfunc']/variables[key][index][i]['xfunc']),
							 	variables[key][index][i]['xfit'],			
							 	lambda x,a,b,c: b*x**a + c,				 	
							 	intercept=False)[1] if (key == ('N','gap',('h','J',))) else variables[key][index][i]['yfunc'],
						# ('N','gap',('h','J',)): fit(
						# 	 	variables[key][index][i]['xfunc'],
						# 	 	(variables[key][index][i]['yfunc']/variables[key][index][i]['xfunc']),
						# 	 	variables[key][index][i]['xfit'],							 	
						# 	 	intercept=True)[1] if (key == ('N','gap',('h','J',))) else variables[key][index][i]['yfunc'],						
						('partition','entanglement',('N','h',)): fit(
							 	np.log(params['N']/np.pi*np.sin(np.pi*variables[key][index][i]['xfunc']))/3,
							 	variables[key][index][i]['yfunc'],
							 	intercept=True)[1] if (key == ('partition','entanglement',('N','h',))) else variables[key][index][i]['yfunc'],
						('N','energy',('h','J',)): variables[key][index][i]['yfunc'] if (key == ('N','energy',('h','J',))) else variables[key][index][i]['yfunc'],
						('N','entanglement',('h','J',)): variables[key][index][i]['yfunc'] if (key == ('N','entanglement',('h','J',))) else variables[key][index][i]['yfunc'],
						}[key])



	settings = {
		**{
		key:{
		index: {
			'ax': {
				'plot':[
				*[{
					'x': variables[key][index][i]['xfunc'],
					'y': variables[key][index][i]['yfunc'],
					'marker':'o',	
					'linestyle':'--',									
					# 'marker':{0:'',1:'o',2:'^'}.get(index,'o'),					
					# 'color':getattr(plt.cm,'tab10')(l),
					'color':getattr(plt.cm,'viridis')((len(variables[key][index]) - 1 - l)/len(variables[key][index])),
					'label':texify(dict(zip(key[2],i))[key[2][0]])if index==(0) else None,
					# 'alpha':0.5,
					}
					for l,i in enumerate(variables[key][index])
					# if ((dict(zip(key[2],i)).get('h',-1) > 0.9 and dict(zip(key[2],i)).get('h',-1) < 1.1) and
					#     (dict(zip(key[2],i)).get('N',1000) >= 8))
				],
				*[{
					'x': variables[key][index][i]['xfit'],
					'y': variables[key][index][i]['yfit'],
					'marker':'',					
					# 'marker':{0:'',1:'o',2:'^'}.get(index,'o'),					
					# 'color':getattr(plt.cm,'tab10')(l),
					'color':'gray',
					'linestyle':'--',
					'linewidth':10,
					'zorder':100,
					'label':r'$N \to \infty$',
					}
					for l,i in enumerate(variables[key][index])
					if index == 0 and l == 0
					# if ((dict(zip(key[2],i)).get('h',-1) > 0.9 and dict(zip(key[2],i)).get('h',-1) < 1.1) and
					#     (dict(zip(key[2],i)).get('N',1000) >= 8))
				],
				],				
				'set_title':{'label':r'$\textrm{Level %d}$'%(index)},
				'set_xlabel':{'xlabel':texify(key[0])},
				'set_ylabel':{'ylabel':texify(key[1])},
				'set_xscale':{'value':'linear'},				
				'set_ylim':{'ymin':-2.5,'ymax':0.5},				
				'set_xticks':{'ticks':np.linspace(
					np.min([u for i in variables[key][index] for u in variables[key][index][i]['xfunc']]),
					np.max([u for i in variables[key][index] for u in variables[key][index][i]['xfunc']]),
					3)},
				'legend':{'title':texify(key[2][0]),'ncol':1,'set_linewidth':{'w':3}},
				'grid':{'visible':True},
			},	
			'fig':{
				'set_size_inches':{'w':18,'h':12},
				'savefig':{'fname':os.path.join(directory,'%s__%s__%s.pdf'%(key[1],key[0],key[2][0]))},
				# 'close':{'fig':True},
			},
			'style':{
				'layout':{
					'nrows':1,
					'ncols':sizes[key[1]],
					'index':index+1,
					# 'ncols':1,
					# 'index':1,
				},
			'rcParams':{
					'font.size':20
				},				

			},			
		}
		for index in range(sizes[key[1]])}
		for key in [('h','energy',('N',))]
		if key in keys
		},
		**{
		key:{
		index: {
			'ax': {
				'plot':[{
					'x': variables[key][index][i]['xfunc'],
					'y': variables[key][index][i]['yfunc'],
					'marker':'o',					
					# 'marker':{0:'',1:'o',2:'^'}.get(index,'o'),					
					# 'color':getattr(plt.cm,'tab10')(l),
					'color':getattr(plt.cm,'viridis')((len(variables[key][index]) - 1 - l)/len(variables[key][index])),
					'label':texify(dict(zip(key[2],i))[key[2][0]])if index==(0) else None,
					}
					for l,i in enumerate(variables[key][index])
					# if ((dict(zip(key[2],i)).get('h',-1) > 0.9 and dict(zip(key[2],i)).get('h',-1) < 1.1) and
					#     (dict(zip(key[2],i)).get('N',1000) >= 8))
					],
				'set_title':{'label':r'$\textrm{Level %d}$'%(index)},
				'set_xlabel':{'xlabel':texify(key[0])},
				'set_ylabel':{'ylabel':texify(key[1])},
				'set_xscale':{'value':'linear'},				
				'set_ylim':{'ymin':0,'ymax':1.1},								
				'set_xticks':{'ticks':np.linspace(
					np.min([u for i in variables[key][index] for u in variables[key][index][i]['xfunc']]),
					np.max([u for i in variables[key][index] for u in variables[key][index][i]['xfunc']]),
					3)},
				'legend':{'title':texify(key[2][0]),'ncol':1,'set_linewidth':{'w':3}},
				'grid':{'visible':True},
			},	
			'fig':{
				'set_size_inches':{'w':18,'h':12},
				'savefig':{'fname':os.path.join(directory,'%s__%s__%s.pdf'%(key[1],key[0],key[2][0]))},
				# 'close':{'fig':True},
			},
			'style':{
				'layout':{
					'nrows':1,
					'ncols':sizes[key[1]],
					'index':index+1,
					# 'ncols':1,
					# 'index':1,
				},
			'rcParams':{
					'font.size':20
				},				
			},			
		}
		for index in range(sizes[key[1]])}
		for key in [('h','order',('N',))]
		if key in keys
		},
		**{
		key:{
		index: {
			'ax': {
				'plot':[{
					'x': variables[key][index][i]['xfunc'],
					'y': (variables[key][index][i]['yfunc']-variables[key][0][i]['yfunc'])*variables[key][index][i]['xfunc'],
					'marker':'o',					
					# 'marker':{0:'',1:'o',2:'^'}.get(index,'o'),					
					# 'color':getattr(plt.cm,'tab10')(l),
					'color':getattr(plt.cm,'viridis')((len(variables[key][index]) - 1 - l)/len(variables[key][index])),
					'label':texify(dict(zip(key[2],i))[key[2][0]])if index==(0) else None,
					}
					for l,i in enumerate(variables[key][index])
					if ((dict(zip(key[2],i)).get('h',-1) > 0.9 and dict(zip(key[2],i)).get('h',-1) < 1.1) and
					    1 #(dict(zip(key[2],i)).get('N',1000) >= 8)
					    )
					],
				# 'set_title':{'label':r'$\textrm{Level %d}$'%(index)},
				'set_xlabel':{'xlabel':texify(key[0])},
				'set_ylabel':{'ylabel':texify(key[1])},
				'set_xscale':{'value':'linear'},				
				'set_xticks':{'ticks':np.linspace(
					np.min([u for i in variables[key][index] for u in variables[key][index][i]['xfunc']]),
					np.max([u for i in variables[key][index] for u in variables[key][index][i]['xfunc']]),
					3)},
				# 'legend':{'title':texify(key[2][0]),'ncol':1,'set_linewidth':{'w':3}},
				'grid':{'visible':True},
				'set_colorbar':{
					'values':[i for i in np.linspace(
						min([dict(zip(key[2],i))['h'] for i in variables[key][index]]),
						max([dict(zip(key[2],i))['h'] for i in variables[key][index]]),
						4*len(variables[key][index]))],
					'colors':[getattr(plt.cm,'viridis')(i/(4*len(variables[key][index]))) for i in range(4*len(variables[key][index]))],
					'size':'3%',
					'pad':0.1,
					'set_ylabel':{'ylabel':texify(key[2][0])},
					'set_yticks':{'ticks':np.linspace(min([dict(zip(key[2],i))['h'] for i in variables[key][index]]),max([dict(zip(key[2],i))['h'] for i in variables[key][index]]),5)},
					# }  if index == (sizes[key[1]]-1) else {},							
					}  if index == (1) else {},							
			},	
			'fig':{
				'set_size_inches':{'w':18,'h':12},
				'savefig':{'fname':os.path.join(directory,'%s__%s__%s.pdf'%(key[1],key[0],key[2][0]))},
				# 'close':{'fig':True},
			},
			'style':{
				'layout':{
					'nrows':1,
					'ncols':sizes[key[1]],
					'index':index+1,
					'ncols':1,
					'index':1,
				},
			'rcParams':{
					'font.size':20
				},				
			},			
		}
		for index in [1]} #range(1,sizes[key[1]])}
		for key in [('N','energy',('h','J',))]
		if key in keys
		},		
		**{
		key:{
		index: {
			'ax': {
				'errorbar':[{
					'x': variables[key][index][i]['xfunc'],
					'y': variables[key][index][i]['yfunc']/variables[key][index][i]['xfunc'],
					**({
					# 'xerr': variables[key][index][i]['xerr'],
					'yerr': variables[key][index][i]['yerr']/variables[key][index][i]['xfunc'],
					'capsize':5,
					'elinewidth':2,
					} if variables[key][index][i]['yerr'].sum()>0 else {}),
					'marker':'o',
					# 'color':getattr(plt.cm,'tab10')(l),
					'color':getattr(plt.cm,'viridis')((len(variables[key][index]) - 1 - l)/len(variables[key][index])),					
					'label':r'$%s$'%('~'.join(['%s: %s'%(texify(k,usetex=False),texify(j,usetex=False)) for k,j in dict(zip(key[2],i)).items()])),
					}
					for l,i in enumerate(variables[key][index])
					# if (dict(zip(key[2],i)).get('h',0.0) == 0.0)
					# if (dict(zip(key[2],i)).get('h',0.5) == 0.5)
					# if (dict(zip(key[2],i)).get('h',-1) == -1 and
					#     dict(zip(key[2],i)).get('J',-1) == -1)
					],
				'set_title':{'label':r'$\textrm{Level %d}$'%(index)},
				'set_xlabel':{'xlabel':texify(key[0])},
				'set_ylabel':{'ylabel':texify(key[1])},
				'set_xscale':{'value':'linear'},				
				'set_yscale':{'value':'log'},				
				'set_xticks':{'ticks':np.arange(
					np.min([u for i in variables[key][index] for u in variables[key][index][i]['xfunc']]),
					np.max([u for i in variables[key][index] for u in variables[key][index][i]['xfunc']])+2,
					2)
					},
				'legend':{
					'title':r'$%s$'%(',~'.join(['%s'%(texify(k,usetex=False)) for k,j in dict(zip(key[2],i)).items()])),
					'ncol':1,
					'set_errorbar':False},
				'grid':{'visible':True},
			},	
			'fig':{
				'set_size_inches':{'w':18,'h':12},
				'savefig':{'fname':os.path.join(directory,'%s__%s__%s.pdf'%(key[1],key[0],'__'.join(key[2])))},
				# 'close':{'fig':True},
			},
			'style':{
				'layout':{
					'nrows':1,
					'ncols':sizes[key[1]],
					'index':index+1,
				},
			'rcParams':{
					'font.size':20
				},				
			},			
		}
		for index in range(sizes[key[1]])}
		for key in [('N','entanglement',('h','J'))]
		if key in keys
		},		
		**{
		key:{
		index: {
			'ax': {
				'plot':[
				*[{
					'x': variables[key][index][i]['xfunc'],
					# 'y': variables[key][index][i]['yfunc'],
					'y': variables[key][index][i]['yfunc']/variables[key][index][i]['xfunc'],
					'marker':'o',	
					'linestyle':'--',									
					# 'marker':{0:'',1:'o',2:'^'}.get(index,'o'),					
					# 'color':getattr(plt.cm,'tab10')(l),
					# 'color':getattr(plt.cm,'viridis')((len(variables[key][index]) - 1 - l)/len(variables[key][index])),
					'color':getattr(plt.cm,'twilight')(
						((dict(zip(key[2],i))['h']-0.9)/(1.1-0.9))),
					# 'color':getattr(plt.cm,'twilight')(
					# 	((dict(zip(key[2],i))['h']-min([dict(zip(key[2],i))['h'] for i in variables[key][index]]))/
					# 	 max(1e-30,(max([dict(zip(key[2],i))['h'] for i in variables[key][index]]) - min([dict(zip(key[2],i))['h'] for i in variables[key][index]]))))),
					# 'label':texify(dict(zip(key[2],i))[key[2][0]])if index==(0) else None,					
					# 'alpha':0.5,
					}
					for l,i in enumerate(variables[key][index])
					if ((dict(zip(key[2],i)).get('h',-1) > 0.9 and dict(zip(key[2],i)).get('h',-1) < 1.1) and
					    1
					    # (dict(zip(key[2],i)).get('N',1000) >= 8))
						)
				],
				*[{
					'x': variables[key][index][i]['xfit'],
					# 'y': variables[key][index][i]['yfit'],
					'y': variables[key][index][i]['yfit']/variables[key][index][i]['xfit'],
					'marker':'',					
					# 'marker':{0:'',1:'o',2:'^'}.get(index,'o'),					
					# 'color':getattr(plt.cm,'tab10')(l),
					# 'color':'gray',
					'color':getattr(plt.cm,'twilight')(
						((dict(zip(key[2],i))['h']-0.9)/(1.1-0.9))),					
					'linestyle':'-',
					'linewidth':4,
					'zorder':-1,
					'alpha':0.6,
					'label':r'$\tilde{\Delta} \sim \tilde{\gamma}N^{-\tilde{z}} + \tilde{\delta} \quad \quad \tilde{h}/J = %0.3f,~ \tilde{z} = %0.3f,~ \tilde{\gamma} = %0.3f, ~ \tilde{\delta} = %0.3f$'%(
						dict(zip(key[2],i)).get('h',-1) ,*tuple(variables[key][index][i]['ycoef'])) if index==(0) else None,					
					}
					for l,i in enumerate(variables[key][index])
					# if index == 0 and l == 0
					if ((dict(zip(key[2],i)).get('h',-1) > 0.9 and dict(zip(key[2],i)).get('h',-1) < 1.1) and
						# (dict(zip(key[2],i)).get('h',-1) == 1) and
						# 1
						l == [np.abs(variables[key][index][j]['ycoef'][-1]) 
								for q,j in enumerate(variables[key][index])].index(
							  np.min([np.abs(variables[key][index][j]['ycoef'][-1]) 
							  			for j in variables[key][index] 
							  		if (dict(zip(key[2],j)).get('h',-1) > 0.9 and dict(zip(key[2],j)).get('h',-1) < 1.1)
							  		])) 
					    # (dict(zip(key[2],i)).get('N',1000) >= 8))
					    # 1
						)					
				],
				],
				# 'set_title':{'label':r'$\textrm{Level %d}$'%(index)},
				'set_xlabel':{'xlabel':texify(r'1/%s'%(key[0]))},
				'set_ylabel':{'ylabel':texify(key[1])},
				'set_xscale':{'value':'linear'},				
				'set_yscale':{'value':'linear'},
				'set_xticks':{'ticks':variables[key][index][list(variables[key][index])[0]]['xfunc']},
				'set_xticklabels':{'labels':[r'$%d$'%(int(j)) for j in 1/variables[key][index][list(variables[key][index])[0]]['xfunc']]},
				# 'set_xscale':{'value':'log'},				
				# 'set_yscale':{'value':'log'},				
				'legend':{'loc':(0.42,0.05)},
				'grid':[{'visible':True,'which':'major'},{'visible':True,'which':'minor'},],									
				'set_colorbar':{
					'values':[i for i in np.linspace(0.9,1.1,100)],
					'colors':[getattr(plt.cm,'twilight')((i-0.9)/(1.1-0.9)) for i in np.linspace(0.9,1.1,100)],
					# 'values':[i for i in np.linspace(
					# 	min([dict(zip(key[2],i))['h'] for i in variables[key][index]]),
					# 	max([dict(zip(key[2],i))['h'] for i in variables[key][index]]),
					# 	4*len(variables[key][index]))],
					# 'colors':[getattr(plt.cm,'twilight')(i/(4*len(variables[key][index]))) for i in range(4*len(variables[key][index]))],
					'size':'3%',
					'pad':0.1,
					'set_ylabel':{'ylabel':texify(key[2][0])},
					# 'set_yticks':{'ticks':np.linspace(min([dict(zip(key[2],i))['h'] for i in variables[key][index]]),max([dict(zip(key[2],i))['h'] for i in variables[key][index]]),5)},
					'set_yticks':{'ticks':np.linspace(0.9,1.1,5)},
					}  if index == 0 else {},			
			},	
			'fig':{
				'set_size_inches':{'w':18,'h':12},
				'savefig':{'fname':os.path.join(directory,'%s__%s__%s.pdf'%(key[1],key[0],key[2][0]))},
				# 'close':{'fig':True},
			},
			'style':{
				'layout':{
					'nrows':1,
					'ncols':1,
					'index':1,
				},
			'rcParams':{
					'font.size':20
				},				
			},			
		}
		for index in [0]} #range(sizes[key[1]]-1)}
		for key in [('N','gap',('h','J',))]
		if key in keys
		},	
	**{
		key:{
		(index,j): {
			'ax': {
				'errorbar':[
					*[{
					'x': variables[key][index][i]['xfunc'],
					'y': variables[key][index][i]['yfunc'],
					**({
					# 'xerr': variables[key][index][i]['xerr'],
					'yerr': variables[key][index][i]['yerr'],
					} if variables[key][index][i]['yerr'].sum()>0 else {}),					
					'marker':{0:'^',1:'o',2:'*'}.get(dict(zip(key[2],i))['h'],'o'),
					'linestyle':'--',
					# 'alpha':{0:0.3,1:0.6,2:0.9}.get(dict(zip(key[2],i))['h'],1),
					# 'color':getattr(plt.cm,'tab10')(parameters['N'].index(dict(zip(key[2],i))['N'])),					
					'color':getattr(plt.cm,'viridis')((len(parameters['N'])-parameters['N'].index(dict(zip(key[2],i))['N']))/len(parameters['N'])),
					# 'label':'$%s%s ~ c = %0.3f,~ C = %0.3f$'%(
					# 	'~~'.join(['%s: %s'%(k,texify(j,usetex=False)) for k,j in dict(zip(key[2],i)).items() if k not in ['h']]),
					# 	r'~' if dict(zip(key[2],i))['N']<10 else '',
					# 	*([v+0 if v>1e-4 else 0  for v in variables[key][index][i]['ycoef']])) if index==(0) else None,					
					}
					for l,i in enumerate(sorted(variables[key][index],key=lambda i: dict(zip(key[2],i))['h']))
					if all([dict(zip(key[2],i))[attr] == dict(zip(key[2],j))[attr] for attr in key[2]])
					# if dict(zip(key[2],i))['h'] == 1					
					],
					*[{
					'x': variables[key][index][i]['xfit'],
					'y': variables[key][index][i]['yfit'],
					'linestyle':'-',					
					# 'alpha':{0:0.3,1:0.6,2:0.9}.get(dict(zip(key[2],i))['h'],1),					
					# 'color':getattr(plt.cm,'tab10')(parameters['N'].index(dict(zip(key[2],i))['N'])),
					'color':getattr(plt.cm,'viridis')((len(parameters['N'])-parameters['N'].index(dict(zip(key[2],i))['N']))/len(parameters['N'])),				
					'label':'$%s%s ~ c = %0.3f,~ C = %0.3f$'%(
						'~~'.join(['%s: %s'%(k,texify(j,usetex=False)) for k,j in dict(zip(key[2],i)).items() if k not in ['h']]),
						r'~' if dict(zip(key[2],i))['N']<10 else '',
						*([v+0 if v>1e-4 else 0  for v in variables[key][index][(dict(zip(key[2],i))['N'],1)]['ycoef']])) if index==(0) else None,										
					}
					for l,i in enumerate(sorted(variables[key][index],key=lambda i: dict(zip(key[2],i))['h']))
					if all([dict(zip(key[2],i))[attr] == dict(zip(key[2],j))[attr] for attr in key[2]])
					# if dict(zip(key[2],i))['h'] == 1					
					]]
					,
				'set_title':{'label':r'$h/J = %0.1f$'%(dict(zip(key[2],j))['h'])},
				'set_xlabel':{'xlabel':texify(r'%s'%(key[0]))},
				'set_ylabel':{'ylabel':texify(key[1])},
				'set_xscale':{'value':'linear'},				
				'set_xlim':{'xmin':0,'xmax':1},				
				'set_ylim':{'ymin':0,'ymax':1},				
				'set_xticks':{'ticks':[0,0.25,0.5,0.75,1]},				
				'legend':{'title':texify(key[2][0]),'ncol':1,'loc':'best','set_errorbar':False} if dict(zip(key[2],j))['h'] == 1 else None,
				'grid':{'visible':True},
			},	
			'fig':{
				'set_size_inches':{'w':18,'h':12},
				'subplots_adjust':{},				
				'tight_layout':{},
				'savefig':{'fname':os.path.join(directory,'%s__%s__%s.pdf'%(key[1],key[0],key[2][0]))},
				# 'close':{'fig':True},
			},
			'style':{
				'layout':{
					# 'nrows':1,
					# 'ncols':1,
					# 'index':1,
					'nrows':1,
					'ncols':len(set([dict(zip(key[2],k))['h'] for k in variables[key][index]])) ,
					'index':{0:1,1:2,2:3}[dict(zip(key[2],j))['h']],
				},
			'rcParams':{
					'font.size':20
				},
			},			
		}
		for q,j in enumerate(variables[key][index])
		for index in [0]} #range(sizes[key[1]]-1)}
		for key in [('partition','entanglement',('N','h',))]
		if key in keys
		},			
	}

	# key = ('partition','entanglement',('N','h',))
	# index = 0
	# i = (parameters['N'][-1],1)
	# print(variables[key][index][i]['yfunc'])

	for k,key in enumerate(settings):
		plot(settings=settings[key])


if __name__ == '__main__':
	main(sys.argv[1:])