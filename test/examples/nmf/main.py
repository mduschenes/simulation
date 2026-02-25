#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','.','..','../../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,ones,zeros,rand,random,stochastic
from src.utils import addition,abs2,log10,reciprocal,einsum,reshape,dot,dotr,dotl,condition_number
from src.utils import copy,seeder,delim,scinotation
from src.utils import nmf

from src.iterables import permuter,setter,getter
from src.io import load,dump,join,split,exists

import jax
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from random import choices,seed	as seeds

def main(*args,**kwargs):

	path = kwargs.get('path',str(args[0]) if len(args)>0 else None)

	n = kwargs.get('n',int(args[1]) if len(args)>1 else 6)
	d = kwargs.get('d',int(args[2]) if len(args)>2 else 2)
	x = kwargs.get('x',int(args[3]) if len(args)>3 else None)

	print(path)

	mplstyle = join(path,'plot',ext='mplstyle')

	path = join(path,f'data.n.{n}.x.{x}',ext='pkl')

	length = n//2 + n%2 -1
	locality = 2

	d = d**2
	x = (d**((length))) if x is None else x

	seed = 123
	size = 5
	seeds(seed)
	seed = choices(range(int(2**32)),k=int(size))

	kwargs = {
		'method':[
			'mu',
			'kl',
			'hals'
			# ('mu','hals','kl','hals')
			],
		'initialize':[
			# 'rand',
			# 'nndsvd',
			# 'nndsvda',
			'nndsvdr',
			],
		'metric':[
			'norm',
			'div',
			# 'norm',
			# 'abs',
			],
		'size':[None],
		'eps':[1e-14],
		'iters':[
			1e2
			# 1e1,
			# 1e3,5e1			
			# [1e3,1e3,1e3,1e3],
			],
		'parameters':[0],
		'seed':[i for i in seed],
		'function':[
			'nmf.marginal',
			'nmf.joint',
			],
		'n':[n],'d':[d],'x':[x],'locality':[locality],'length':[length],
		'shapes':[{
			'u':[min(d**(n//2),x),d,min(d**(n//2),x)],
			'v':[min(d**(n//2),x),d,min(d**(n//2),x)],
			'w':[d**(locality)]*(2),
			'x':[d**(length),min(d**(n//2),x)],
			'y':[min(d**(n//2),x),d**(n-length-locality)],
			}]
		}

	def boolean(index,data,options,opts):
		return any(options==data[i]['options'] for i in data)

	def filters(kwargs):

		return not any([
			(kwargs['method'] in ['mu']) and (kwargs['metric'] in ['div']),
			(kwargs['method'] in ['kl']) and (kwargs['metric'] in ['norm']),
			(kwargs['method'] in ['hals']) and (kwargs['metric'] in ['div']),
			(kwargs['method'] in ['hals']) and (kwargs['iters'] >= 1000)
			]
			)

		return True


	booleans = dict(
		run = 1,
		load = 0,
		dump = 1,
		plot = 1,
		)

	booleans = dict(
		run = 0,
		load = 1,
		dump = 0,
		plot = 1,
		)

	if booleans['load']:
		data = load(path,default={})
	else:
		data = {}

	if booleans['run']:

		print('Run',kwargs)

		for index,kwargs in enumerate(permuter(kwargs)):

			if not filters(kwargs):
				continue

			options = {
				'size': None,
				'eps': 5e-9,
				'iters':1e3,
				'parameters': 1e-3,
				'method': 'kl',
				'initialize': 'rand',
				'metric':'norm',
				'seed': 123,
				}
			def init(index,data,options):

				function = nmf

				options['key'] = seeder(options['seed'])
				options['keys'] = seeder(options['seed'],size=len(options['shapes']))

				options['architecture'] = options['function'].split(delim)[-1] if options['function'] and options['function'].count(delim) else None

				opts = dict(
					n = options.pop('n'),
					d = options.pop('d'),
					x = options.pop('x'),
					length = options.pop('length'),
					locality = options.pop('locality'),
					shapes = options.pop('shapes'),
					keys = options.pop('keys'),
					function = options.pop('function'),
				)

				if opts['function'] in ['nmf.marginal']:

					u,v,w = random(opts['shapes']['u'],key=opts['keys'][0]),random(opts['shapes']['v'],key=opts['keys'][1]),reshape(stochastic(opts['shapes']['w'],key=opts['keys'][2]),(d,)*(2*locality))
					x,y = random(opts['shapes']['x'],key=opts['keys'][-2]),random(opts['shapes']['y'],key=opts['keys'][-1])
					
					x,y = addition(x,0),addition(y,-1)
					
					p,q = addition(x,range(0,x.ndim-1)),addition(y,range(1,y.ndim))
					x,y = dotr(x,reciprocal(p)),dotl(y,reciprocal(q))
					u,v = dotl(u,p),dotr(v,q)
					a = einsum('awg,gzb,uvwz->auvb',u,v,w)
					
					a = dotr(dotl(a,x),y)
					c = a

					a /= addition(a)
					objects = a,u,v,(x,y)

				elif opts['function'] in ['nmf.joint']:
				
					u,v,w = random(opts['shapes']['u'],key=opts['keys'][0]),random(opts['shapes']['v'],key=opts['keys'][1]),reshape(stochastic(opts['shapes']['w'],key=opts['keys'][2]),(d,)*(2*locality))
					x,y = random(opts['shapes']['x'],key=opts['keys'][-2]),random(opts['shapes']['y'],key=opts['keys'][-1])
					
					p,q = addition(x,range(0,x.ndim-1)),addition(y,range(1,y.ndim))
					x,y = dotr(x,reciprocal(p)),dotl(y,reciprocal(q))
					u,v = dotl(u,p),dotr(v,q)
					a = einsum('awg,gzb,uvwz->auvb',u,v,w)
					
					a = a
					c = dot(x,dot(a,y))

					a /= addition(c)
					objects = a,u,v,(x,y)

				return function,objects,options,opts

			def process(index,data,stats,options,opts):
				if boolean(index,data,options,opts):
					for i in data:
						if boolean(i,{i:data[i]},options,opts):
							index = i
							break
				else:
					index = len(data)
				if not isinstance(data.get(index),dict):
					data[index] = {}
				data[index].update({**dict(options={**options,**opts}),**stats})
				return

			def func(function,objects,options):
				u,v,s,stats = function(*objects,**options)
				return stats

			print(kwargs)

			setter(options,kwargs,delimiter=delim,default='replace')

			kwargs = copy(options)

			function,objects,options,opts = init(index,data,options)

			if boolean(index,data,options,opts):
				continue

			stats = func(function,objects,options)

			process(index,data,stats,kwargs,opts)

			print()

	if booleans['dump']:
		
		dump(data,path)

	if booleans['plot']:

		print('Plot',path)

		data = load(path)

		if data is None:
			return

		attrs = {
			**{attr:dict(
				x='iteration',
				y=attr,
				label=['function','method','metric','seed'])
			for attr in set(i for index in data for i in data[index] 
				if i not in ['options'] and i in ['error'])
			},
		}

		def texify(string,default=None):

			texify = {
				'method':'$\\textnormal{NMF Update}$',
				'initialize':'$\\textnormal{Initialize}$',
				'metric':'$\\textnormal{Objective}$',
				'seed':'$\\textnormal{Instance}$',
				'function':'$\\textnormal{NMF Algorithm Method}$',
				'iteration':'$\\textnormal{NMF Iteration}$',
				'error':'$\\textnormal{NMF Objective}$',
				'rank':'$\\textnormal{Rank}~~~\\textnormal{max}\\left\\{\\textnormal{rank}(U),\\textnormal{rank}(V)\\right\\}$',
				'rank':'$\\textnormal{Time}~~~[s]$',
				'cond.u':'$\\textnormal{Condition Number}~\\kappa(U)$',
				'cond.v':'$\\textnormal{Condition Number}~\\kappa(V)$',
				'nmf':'$\\textnormal{NMF}$',
				# **{i:'$\\textcolor[HTML]{22A884}{\\textnormal{Marginal}}~,~\\textcolor[HTML]{EA4F88}{\\textnormal{Joint}}$' for i in ['nmf.marginal','pnmf']},
				**{i:'$\\textnormal{NMF Method}$'+'\n'+'$\\textnormal{Marginal}~,~\\textnormal{Joint}$' for i in ['nmf.joint','xnmf']},
				'mu':'$\\textnormal{MU}~~~$',
				'kl':'$\\textnormal{KL}~~~~$',
				'hals':'$\\textnormal{HALS}$',
				'gd':'$\\textnormal{GD}$',
				'kld':'$\\textnormal{KL-GD}$',
				**{('norm',i):'$\\textnormal{Norm}$' for i in ['nmf.marginal','pnmf']},
				**{('norm',i):'$\\textnormal{Norm}$' for i in ['nmf.joint','xnmf']},
				**{('abs',i):'$\\textnormal{Abs}$' for i in ['nmf.marginal','pnmf']},
				**{('abs',i):'$\\textnormal{Abs}$' for i in ['nmf.joint','xnmf']},
				**{('div',i):'$\\textnormal{KL}' for i in ['nmf.marginal','pnmf']},
				**{('div',i):'$\\textnormal{KL}' for i in ['nmf.joint','xnmf']},
				}

			if string in texify:
				value = texify.get(string,default)
			elif not isinstance(string,str):
				value = '$%s$'%('-'.join([texify.get(i,i) for i in string]).replace('$',''))
			else:
				value = default
			return value

		with matplotlib.style.context(mplstyle):
			for attr in attrs:

				fig,ax = plt.subplots()

				def boolean(data,index=None,wrapper=None,attr=None,):
					if index is None:
						size = max(len(data[i][attrs[attr]['y']]) for i in data)	
					else:
						size = len(data[index][attrs[attr]['y']])
					step = 5
					indices = slice(0,size,step if step < size else 1)
					if wrapper is not None:
						indices = wrapper(indices.start,int(data[index]['options']['iters']),int(data[index]['options']['iters'])//size)
					return indices

				def filters(index,data):
					return not any([
						(data[index]['options']['method'] in ['mu']) and (data[index]['options']['metric'] in ['div']),
						(data[index]['options']['method'] in ['kl']) and (data[index]['options']['metric'] in ['norm']),
						(data[index]['options']['method'] in ['hals']) and (data[index]['options']['metric'] in ['div']),
						]
						)
					return True

				def statistics(data):

					stats = {}

					key = 'indices'
					value = sorted(list(set(data[i]['options'][attrs[attr]['label'][-1]] for i in data)),key=lambda i:[data[i]['options'][attrs[attr]['label'][-1]] for i in data].index(i))
					stats[key] = value

					key = 'time'
					value = ['metric','method','function']
					value = [dict(zip(value,i)) for i in set(tuple(data[i]['options'][option] for option in value) for i in data)]
					value = sorted(value,key=lambda i:['nmf.marginal','nmf.joint'].index(i['function']))
					value = [{**val,**dict(data=[data[i][key][-1] for i in data if all(data[i]['options'][option]==val[option] for option in val)])} for val in value]
					value = [{**val,**dict(data=sum(val['data'])/len(val['data']))} for val in value]
					value = [[val['data'] for val in value if all(val[option]==data[i]['options'][option] for option in val if option not in ['function','data'])] for i in data]
					stats[key] =  value

					return stats

				values = {index:data[index] for index in data if filters(index,data)}					

				options = dict()

				stats = statistics(values)

				x = {index:values[index][attrs[attr]['x']][boolean(values,index=index,attr=attr)] for index in values}
				y = {index:values[index][attrs[attr]['y']][boolean(values,index=index,attr=attr)] for index in values}

				options = {index:{**options,**dict(
					label='$%s~~~%s$'%(
						'~,~'.join(str(texify(values[index]['options'][label] if label not in ['metric'] else (values[index]['options'][label],values[index]['options']['function']),values[index]['options'][label])) for label in attrs[attr]['label'][:-1] if label not in ['metric'] and label in ['method']).replace('$',''),
						'~,~'.join(scinotation(i,scilimits=[-2,4],decimals=4 if i < 1 else 4 if i < 1000 else 2,zero=True,strip=False) for i in stats['time'][index]),
						),
					color=plt.get_cmap({**{i:'viridis' for i in ['nmf.marginal','pnmf']},**{i:'magma' for i in ['nmf.joint','xnmf']}}.get(values[index]['options']['function']))((stats['indices'].index(values[index]['options'][attrs[attr]['label'][-1]])+1)/(len(stats['indices'])+1)),
					alpha=0.6,
					# marker={'norm':'o','abs':'s','div':'^'}.get(values[index]['options']['metric']),
					# linestyle={'mu':'-','kl':'--','hals':':'}.get(values[index]['options']['method']),
					marker={'mu':'o','kl':'s','hals':'X',('kl','hals'):'d'}.get(values[index]['options']['method']),
					linestyle={'norm':'-','div':'-','abs':':'}.get(values[index]['options']['metric']),
					markersize={'mu':14,'kl':12,'hals':14,('kl','hals'):12}.get(values[index]['options']['method']),
					linewidth=7
					)} for index in values}
				plot = {}
				for index in values:
					plot[index] = ax.plot(x[index],y[index],**options[index])

				options = dict(position='right',size="3%",pad=-0.545535)
				number = 6
				functions = sorted(set(values[i]['options']['function'] for i in values),key=lambda i:['nmf.marginal','nmf.joint'].index(i))
				for i,function in enumerate(functions):
					colors = [plt.get_cmap({**{i:'viridis' for i in ['nmf.marginal','pnmf']},**{i:'magma' for i in ['nmf.joint','xnmf']}}.get(function))((i+1)/(len(stats['indices'])+3)) for i in range(len(stats['indices'])+2)]
					# if len(functions)>1:
					# 	opts = {**options,**dict(pad=options['pad']+i*0.065)}
					# 	cax,opts = fig.add_axes([
					# 		ax.get_position().x1+opts['pad'],
					# 		ax.get_position().y0,
					# 		0.01,
					# 		(ax.get_position().y1-ax.get_position().y0)*1.1075
					# 		]),dict()
					# else:
					# 	opts = {**options,**dict(pad=0.1)}
					# 	cax,opts = make_axes_locatable(ax).append_axes(**opts),dict()

					# options = {option:kwargs[attr].get(option) for option in ['location','fraction','shrink','aspect','pad','anchor','panchor'] if option in kwargs[attr]}

					# opts = {**options,**dict(pad=0.1)}
					# cax,opts = make_axes_locatable(cax).append_axes(**opts),dict()

					# opts = {**dict(location='right',fraction=0.15,shrink=1.05 if i==0 else 1,aspect=50,pad=0.2)}
					# cax,opts = matplotlib.colorbar.make_axes([ax for ax in fig.axes],**opts)
					opts = {**dict(position='right',size='6%' if i == 0 else '3%',pad=2.5)}
					cax,opts = make_axes_locatable(ax).append_axes(**opts),dict()

					cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name=None,colors=colors,N=100*len(colors))
					opts = {**opts,**dict(cmap=cmap,orientation='vertical')}
					cbar = matplotlib.colorbar.ColorbarBase(cax,**opts)
					if i == (len(functions)-1):
						cbar.ax.set_ylabel(ylabel=texify(attrs[attr]['label'][-1],attrs[attr]['label'][-1]),size=55)
						cbar.ax.set_yticks(ticks=[(i+1)/(len(stats['indices'])+1) for i,obj in enumerate(stats['indices'])][::max(1,len(stats['indices'])//number)])
						cbar.ax.set_yticklabels(labels=['$%s$'%(i) for i,obj in enumerate(stats['indices'])][::max(1,len(stats['indices'])//number)],size=55)
					else:
						# cbar.ax.set_yticks(ticks=[(i+1)/(len(stats['indices'])+1) for i,obj in enumerate(stats['indices'])][::max(1,len(stats['indices'])//number)])
						# cbar.ax.set_yticklabels(labels=['$%s$'%(i) for i,obj in enumerate(stats['indices'])][::max(1,len(stats['indices'])//number)])
						cbar.ax.set_yticks(ticks=[])
						cbar.ax.set_yticklabels(labels=[],size=55)

					if (i==(len(functions)-1)) and (len(functions)>1):
						cbar.ax.set_xlabel(xlabel=texify(function),x=-0.8,size=55)


				options = dict(size=45)
				# ax.set_title(label="$%s$"%(" ~,~ ".join(["%s = %s"%(i,j) for i,j in [
				# 	("N",max((values[index]['options'].get('n') for index in values if values[index]['options'].get('n')),default=None)),
				# 	("D",max((values[index]['options'].get('d') for index in values if values[index]['options'].get('d')),default=None)),
				# 	# ("L",max((values[index]['options'].get('locality') for index in values if values[index]['options'].get('locality')),default=None)),
				# 	("A","(D^{N/2},D,D,D^{N/2})"),
				# 	] if i and j])
				# 	),**options)
				ax.set_xlabel(xlabel=texify(attrs[attr]['x']),**options)
				ax.set_ylabel(ylabel=texify(attrs[attr]['y']),**options)


				if attr in ['error']:
					options = dict(x=[int(min(min((x[index])) for index in x)),int(max(max((x[index])) for index in x))],y=[int(min(min(log10(y[index])) for index in y)),int(max(max(log10(y[index])) for index in y))])
					number = 6
					# ax.set_xlim(xmin=(min(max(1,int(options['x'][0]*0.1)),-int(options['x'][-1]*0.05))),xmax=(max(int(options['x'][-1]*1.1),1)))
					# ax.set_xticks(ticks=range(options['x'][0],options['x'][-1],max(1,(options['x'][-1]-options['x'][0])//number)))
					# ax.tick_params(**{"axis":"x","which":"minor","length":0,"width":0})
					# ax.set_xscale(value='linear')
					# ax.set_ylim(ymin=5*10**(options['y'][0]-2),ymax=2*10**(options['y'][-1]+1))
					# ax.set_yticks(ticks=[10**(i) for i in range(options['y'][0]-1,options['y'][-1]+1,2)])
					# ax.tick_params(**{"axis":"y","which":"minor","length":0,"width":0})
					# ax.set_yscale(value='log')
					ax.set_xscale(value='linear')
					ax.set_xlim(xmin=-5,xmax=105)
					ax.set_xticks(ticks=[0,20,40,60,80,100])
					ax.set_xticklabels(labels=['$%d$'%(i) for i in [0,20,40,60,80,100]],size=55)
					ax.tick_params(**{"axis":"x","which":"minor","length":0,"width":0})

					ax.set_yscale(value='log')
					ax.set_ylim(ymin=1e-17,ymax=1e1)
					ax.set_yticks(ticks=[1e-16,1e-12,1e-8,1e-4,1e0])
					ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i != 0 else '$1$' for i in [-16,-12,-8,-4,0]],size=55)
					ax.tick_params(**{"axis":"y","which":"minor","length":0,"width":0})
					ax.grid(visible=True)
				elif attr in ['rank']:
					options = dict(x=[int(min(min((x[index])) for index in x)),int(max(max((x[index])) for index in x))],y=[int(min(min((y[index])) for index in y)),int(max(max((y[index])) for index in y))])
					number = 6
					ax.set_xlim(xmin=(min(max(1,int(options['x'][0]*0.1)),-int(options['x'][-1]*0.05))),xmax=(max(int(options['x'][-1]*1.1),1)))
					ax.set_xticks(ticks=range(options['x'][0],options['x'][-1],max(1,(options['x'][-1]-options['x'][0])//number)))
					ax.tick_params(**{"axis":"x","which":"minor","length":0,"width":0})
					ax.set_xscale(value='linear')
					ax.set_ylim(ymin=(options['y'][0]-1),ymax=(options['y'][-1]+1))
					ax.set_yticks(ticks=[i for i in range(options['y'][0]-1,options['y'][-1]+1,2)])
					ax.tick_params(**{"axis":"y","which":"minor","length":0,"width":0})
					ax.set_yscale(value='linear')									
				elif attr in ['time']:
					options = dict(x=[int(min(min((x[index])) for index in x)),int(max(max((x[index])) for index in x))],y=None)
					number = 6
					ax.set_xlim(xmin=(min(max(1,int(options['x'][0]*0.1)),-int(options['x'][-1]*0.05))),xmax=(max(int(options['x'][-1]*1.1),1)))
					ax.set_xticks(ticks=range(options['x'][0],options['x'][-1],max(1,(options['x'][-1]-options['x'][0])//number)))
					ax.tick_params(**{"axis":"x","which":"minor","length":0,"width":0})
					ax.set_xscale(value='linear')
					ax.set_ylim(ymin=1e-7,ymax=1e3)
					ax.set_yticks(ticks=[1e-6,1e-4,1e-2,1e0,1e2])
					ax.set_yticklabels(labels=['$10^{%d}$'%(i) if i != 0 else '$1$' for i in [-6,-4,-2,0,2]])
					ax.tick_params(**{"axis":"y","which":"minor","length":0,"width":0})
					ax.set_yscale(value='log')

				options = dict(
					title=(
						'$%s~,~%s$'%(
						'~,~'.join(texify(label,label) for label in attrs[attr]['label'][:-1] if label not in ['metric'] and label in ['method']).replace('$',''),
						'\\textnormal{Time}~~[s]',
						)),
					ncol=1,
					title_fontsize=45,
					fontsize=45,
					markerscale=1.5,
					handlelength=3,
					loc= 'lower right',#(1.4175,0.1685) if len(functions)>1 else (1.1,0.45),
					)
				handles_labels = [getattr(axes,'get_legend_handles_labels')() for axes in ax.get_figure().axes]
				handles,labels = [sum(i, []) for i in zip(*handles_labels)]
				handles,labels = (
					[handle[0] if isinstance(handle, matplotlib.container.ErrorbarContainer) else handle for handle,label in zip(handles,labels)],
					[label if isinstance(handle, matplotlib.container.ErrorbarContainer) else label for handle,label in zip(handles,labels)]
					)
				indexes,unique = [[i for i,label in enumerate(labels) if label==value] for value in sorted(set(labels),key=lambda i:labels.index(i))],[len([j for j in labels if j==i])//2 for i in sorted(set(labels),key=lambda i:labels.index(i))]
				handles,labels = [copy(handles[i[j]]) for i,j in zip(indexes,unique)],[labels[i[j]] for i,j in zip(indexes,unique)]
				for handle in handles:
					handle.set_color('gray')
				leg = ax.legend(handles,labels,**options)

				options = dict(
					w=20,
					h=12
					)
				fig.set_size_inches(**options)
				fig.subplots_adjust()
				fig.tight_layout()
				options = dict(fname=join(
					split(path,directory=True),
					'%s.%s.%s.%s'%(
						'plot',
						attrs[attr]['x'],
						attrs[attr]['y'],
						'.'.join(split(path,directory_file=True).split('.')[1:]),
						),ext='pdf'),bbox_inches='tight',pad_inches=0.6)
				print(options)
				fig.savefig(**options)

	return


if __name__ == '__main__':
	args = []
	kwargs = {}

	args.extend([])
	kwargs.update({})

	settings = {'path':sys.argv[1:2] if sys.argv[1:] else ['~/scratch/nmf'],'n':[6],'d':[2],'x':[2,4,8,16,32,64]}
	for setting in permuter(settings):

		kwargs.update(setting)

		main(*args,**kwargs)