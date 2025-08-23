#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','.','..','../../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))


os.environ['NUMPY_BACKEND'] = 'JAX'

from src.utils import array,ones,zeros,rand,random,stochastic
from src.utils import addition,abs2,log10,reciprocal,einsum,reshape,dot,dotr,dotl,condition_number
from src.utils import copy,seeder,delim
from src.utils import nmf

from src.iterables import permuter,setter,getter
from src.io import load,dump,join,exists

import jax
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from random import choices,seed	as seeds

def main(*args,**kwargs):

	n = int(args[0] if len(args)>0 else 5)
	d = int(args[1] if len(args)>1 else 2)
	l = int(args[2] if len(args)>2 else 2)

	directory = 'scratch/nmf/data'
	file = 'data'
	mplstyle = 'config/plot.mplstyle'
	path = join(directory,file,ext='pkl')

	q = n//2 + n%2
	k = d**2

	seed = 123
	size = 1
	seeds(seed)
	seed = choices(range(int(2**32)),k=int(size))

	kwargs = {
		'method':[
			# 'mu',
			# 'kl',
			# 'hals'
			('mu','hals','kl','hals')
			],
		'initialize':[
			# 'rand',
			# 'nndsvd',
			# 'nndsvda',
			'nndsvdr',
			],
		'metric':[
			# 'norm',
			'div',
			# 'abs',
			],
		'size':[None],
		'eps':[0],
		'iters':[
			# 1e1,
			# 1e3,5e1			
			[1e3,1e3,1e3,1e3],
			],
		'parameters':[0],
		'seed':[i for i in seed],
		'function':[
			'nmf.marginal',
			'nmf.joint',
			],
		'n':[n],'d':[d],'l':[l],'q':[q],'k':[k],
		'shapes':[[
			[k**(q),k,k**(q+1)],
			[k**(q+1),k,k**(q)],
			[k**(l)]*(2),
			[k**(q-q+1),k**(q)],
			[k**(q),k**(q-q+1)]
			]]
		}

	def boolean(index,data,options,opts):
		return any(options==data[i]['options'] for i in data)

	def filters(kwargs):
		if kwargs['method'] in ['hals'] and kwargs['iters'] >= 1000:
			return False

		return True


	booleans = dict(
		run = 1,
		load = 0,
		dump = 1,
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
					l = options.pop('l'),
					q = options.pop('q'),
					k = options.pop('k'),
					shapes = options.pop('shapes'),
					keys = options.pop('keys'),
					function = options.pop('function'),
				)

				if opts['function'] in ['nmf.marginal']:

					u,v,d = random(opts['shapes'][0],key=opts['keys'][0]),random(opts['shapes'][1],key=opts['keys'][1]),reshape(stochastic(opts['shapes'][2],key=opts['keys'][2]),(k,)*(2*l))
					x,y = random(opts['shapes'][-2],key=opts['keys'][-2]),random(opts['shapes'][-1],key=opts['keys'][-1])
					
					x,y = addition(x,0),addition(y,-1)
					
					p,q = addition(x,range(0,x.ndim-1)),addition(y,range(1,y.ndim))
					x,y = dotr(x,reciprocal(p)),dotl(y,reciprocal(q))
					u,v = dotl(u,p),dotr(v,q)
					a = einsum('awg,gzb,uvwz->auvb',u,v,d)
					
					a = dotr(dotl(a,x),y)
					c = a

					a /= addition(a)
					objects = a,u,v,(x,y)

				elif opts['function'] in ['nmf.joint']:
				
					u,v,d = random(opts['shapes'][0],key=opts['keys'][0]),random(opts['shapes'][1],key=opts['keys'][1]),reshape(stochastic(opts['shapes'][2],key=opts['keys'][2]),(k,)*(2*l))
					x,y = random(opts['shapes'][-2],key=opts['keys'][-2]),random(opts['shapes'][-1],key=opts['keys'][-1])
					
					p,q = addition(x,range(0,x.ndim-1)),addition(y,range(1,y.ndim))
					x,y = dotr(x,reciprocal(p)),dotl(y,reciprocal(q))
					u,v = dotl(u,p),dotr(v,q)
					a = einsum('awg,gzb,uvwz->auvb',u,v,d)
					
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

		attrs = {
			**{attr:dict(
				x='iteration',
				y=attr,
				label=['function','method','metric','seed'])
			for attr in set(i for index in data for i in data[index] 
				if i not in ['options'] and i in ['error','rank'])
			},
		}
		def texify(string,default=None):

			texify = {
				'method':'$\\textnormal{Method}$',
				'initialize':'$\\textnormal{Initialize}$',
				'metric':'$\\textnormal{Metric}$',
				'seed':'$\\textnormal{Seed}$',
				'function':'$\\textnormal{Function}$',
				'iteration':'$\\textnormal{Iteration}$',
				'error':'$\\textnormal{Error}~\\mathcal{L}(A,UV)$',
				'rank':'$\\textnormal{Rank}~~~\\textrm{max}\\left\\{\\textrm{rank}(U),\\textrm{rank}(V)\\right\\}$',
				'cond.u':'$\\textnormal{Condition Number}~\\kappa(U)$',
				'cond.v':'$\\textnormal{Condition Number}~\\kappa(V)$',
				'nmf':'$\\textnormal{NMF}$',
				'nmf.marginal':'$\\textnormal{Marginal-NMF}$',
				'nmf.joint':'$\\textnormal{Joint-NMF}$',
				'mu':'$\\textnormal{MU}$',
				'kl':'$\\textnormal{KL}$',
				'hals':'$\\textnormal{H-ALS}$',
				'gd':'$\\textnormal{GD}$',
				'kld':'$\\textnormal{KL-GD}$',
				('norm','nmf.marginal'):'$\\norm{A-UV}/\\norm{A}',
				('norm','nmf.joint'):'$\\norm{A-UV}/\\norm{A}',
				('abs','nmf.marginal'):'$\\norm{X_{\\alpha}\\abs{A_{\\alpha\\mu\\nu\\beta}-U_{\\alpha\\mu\\gamma}V_{\\gamma\\nu\\beta}}Y_{\\beta}}/\\norm{A}',
				('abs','nmf.joint'):'$\\norm{X_{x \\alpha}\\abs{A_{\\alpha\\mu\\nu\\beta}-U_{\\alpha\\mu\\gamma}V_{\\gamma\\nu\\beta}}Y_{\\beta y}}/\\norm{A}',
				('div','nmf.marginal'):'$\\mathcal{D}(A,UV)',
				('div','nmf.joint'):'$\\mathcal{D}(A,UV)',
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

				def boolean(data,index=None,wrapper=None):
					if index is None:
						size = max(len(data[i][attrs[attr]['y']]) for i in data)	
					else:
						size = len(data[index][attrs[attr]['y']])
					step = max(20,(size//100)) 
					indices = slice(0,size,step if step < size else 1)
					if wrapper is not None:
						indices = wrapper(indices.start,int(data[index]['options']['iters']),int(data[index]['options']['iters'])//size)
					return indices

				def filters(index,data):
					return True

				values = {index:data[index] for index in data if filters(index,data)}					

				options = dict()
				indices = sorted(list(set(values[i]['options'][attrs[attr]['label'][-1]] for i in values)),key=lambda i:[values[i]['options'][attrs[attr]['label'][-1]] for i in values].index(i))
				
				x = {index:values[index][attrs[attr]['x']][boolean(values,index=index)] for index in values}
				y = {index:values[index][attrs[attr]['y']][boolean(values,index=index)] for index in values}

				options = {index:{**options,**dict(
					label='$%s$'%('~,~'.join(str(texify(values[index]['options'][label] if label not in ['metric'] else (values[index]['options'][label],values[index]['options']['function']),values[index]['options'][label])) for label in attrs[attr]['label'][:-1]).replace('$','')),
					color=plt.get_cmap({'nmf.marginal':'viridis','nmf.joint':'magma'}.get(values[index]['options']['function']))((indices.index(values[index]['options'][attrs[attr]['label'][-1]])+1)/(len(indices)+1)),
					alpha=0.6,
					# marker={'norm':'o','abs':'s','div':'^'}.get(values[index]['options']['metric']),
					# linestyle={'mu':'-','kl':'--','hals':':'}.get(values[index]['options']['method']),
					marker={'mu':'o','kl':'s','hals':'^',('kl','hals'):'d'}.get(values[index]['options']['method']),
					linestyle={'norm':'-','div':'--','abs':':'}.get(values[index]['options']['metric']),
					markersize=8,
					linewidth=3
					)} for index in values}
				plot = {}
				for index in values:
					plot[index] = ax.plot(x[index],y[index],**options[index])

				options = dict(position='right',size="3%",pad=-0.545535)
				number = 6
				functions = sorted(set(values[i]['options']['function'] for i in values))
				for i,function in enumerate(functions):
					colors = [plt.get_cmap({'nmf.marginal':'viridis','nmf.joint':'magma'}.get(function))((i+1)/(len(indices)+3)) for i in range(len(indices)+2)]
					if len(functions)>1:
						opts = {**options,**dict(pad=options['pad']+i*0.065)}
						cax,opts = fig.add_axes([
							ax.get_position().x1+opts['pad'],
							ax.get_position().y0,
							0.01,
							(ax.get_position().y1-ax.get_position().y0)*1.1075
							]),dict()
					else:
						opts = {**options,**dict(pad=0.1)}
						cax,opts = make_axes_locatable(ax).append_axes(**opts),dict()

					cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name=None,colors=colors,N=100*len(colors))
					opts = {**opts,**dict(cmap=cmap,orientation='vertical')}
					cbar = matplotlib.colorbar.ColorbarBase(cax,**opts)
					if i == (len(functions)-1):
						cbar.ax.set_ylabel(ylabel=texify(attrs[attr]['label'][-1],attrs[attr]['label'][-1]))
						cbar.ax.set_yticks(ticks=[(i+1)/(len(indices)+1) for i,obj in enumerate(indices)][::max(1,len(indices)//number)])
						cbar.ax.set_yticklabels(labels=['$%s$'%(i) for i,obj in enumerate(indices)][::max(1,len(indices)//number)])
					else:
						cbar.ax.set_yticks(ticks=[(i+1)/(len(indices)+1) for i,obj in enumerate(indices)][::max(1,len(indices)//number)])
						cbar.ax.set_yticklabels(labels=['$%s$'%(i) for i,obj in enumerate(indices)][::max(1,len(indices)//number)])						
						# cbar.ax.set_yticks(ticks=[])
						# cbar.ax.set_yticklabels(labels=[])

					if len(functions)>1:
						cbar.ax.set_xlabel(xlabel=texify(function))


				options = dict()
				ax.set_title(label="$%s$"%(" ~,~ ".join(["%s = %s"%(i,j) for i,j in [
					("N",max((values[index]['options'].get('n') for index in values if values[index]['options'].get('n')),default=None)),
					("D",max((values[index]['options'].get('k') for index in values if values[index]['options'].get('k')),default=None)),
					# ("L",max((values[index]['options'].get('l') for index in values if values[index]['options'].get('l')),default=None)),
					("A","(D^{N/2},D,D,D^{N/2})"),
					] if i and j])
					),**options)
				ax.set_xlabel(xlabel=texify(attrs[attr]['x']),**options)
				ax.set_ylabel(ylabel=texify(attrs[attr]['y']),**options)


				if attr in ['error']:
					options = dict(x=[int(min(min((x[index])) for index in x)),int(max(max((x[index])) for index in x))],y=[int(min(min(log10(y[index])) for index in y)),int(max(max(log10(y[index])) for index in y))])
					number = 6
					ax.set_xlim(xmin=(min(max(1,int(options['x'][0]*0.1)),-int(options['x'][-1]*0.05))),xmax=(max(int(options['x'][-1]*1.1),1)))
					ax.set_xticks(ticks=range(options['x'][0],options['x'][-1],max(1,(options['x'][-1]-options['x'][0])//number)))
					ax.tick_params(**{"axis":"x","which":"minor","length":0,"width":0})
					ax.set_xscale(value='linear')
					ax.set_ylim(ymin=5*10**(options['y'][0]-2),ymax=2*10**(options['y'][-1]+1))
					ax.set_yticks(ticks=[10**(i) for i in range(options['y'][0]-1,options['y'][-1]+1,2)])
					ax.tick_params(**{"axis":"y","which":"minor","length":0,"width":0})
					ax.set_yscale(value='log')
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

				options = dict(
					title=(
						'$%s ~:~ %s$'%(
						'~,~'.join(texify(label,label) for label in attrs[attr]['label'][:-1]).replace('$',''),
						{
							'nmf.marginal':'$A_{\\mu\\nu} = X_{\\alpha}A_{\\alpha\\mu\\nu\\beta}Y_{\\beta} \\approx X_{\\alpha}U_{\\alpha\\mu\\gamma}V_{\\gamma\\nu\\beta}Y_{\\beta} = U_{\\mu\\gamma}V_{\\gamma\\nu}$'.replace('$',''),
							'nmf.joint':'$A_{x \\mu\\nu y} = X_{x \\alpha}A_{\\alpha\\mu\\nu\\beta}Y_{\\beta y} \\approx X_{x \\alpha}U_{\\alpha\\mu\\gamma}V_{\\gamma\\nu\\beta}Y_{\\beta y} = U_{x \\mu\\gamma}V_{\\gamma\\nu y}$'.replace('$','')
						}.get(values[index]['options']['function'])
						)),
					ncol=1,
					loc=(1.4175,0.1685) if len(functions)>1 else (1.1,0.45),
					)
				handles_labels = [getattr(axes,'get_legend_handles_labels')() for axes in ax.get_figure().axes]
				handles,labels = [sum(i, []) for i in zip(*handles_labels)]
				handles,labels = (
					[handle[0] if isinstance(handle, matplotlib.container.ErrorbarContainer) else handle for handle,label in zip(handles,labels)],
					[label if isinstance(handle, matplotlib.container.ErrorbarContainer) else label for handle,label in zip(handles,labels)]
					)
				indexes,unique = [[i for i,label in enumerate(labels) if label==value] for value in sorted(set(labels),key=lambda i:labels.index(i))],[len([j for j in labels if j==i])//2 for i in sorted(set(labels),key=lambda i:labels.index(i))]
				handles,labels = [copy(handles[i[j]]) for i,j in zip(indexes,unique)],[labels[i[j]] for i,j in zip(indexes,unique)]
				# for handle in handles:
				# 	handle.set_color('gray')
				leg = ax.legend(handles,labels,**options)

				options = dict(
					w=43.5,
					h=14
					)
				fig.set_size_inches(**options)
				fig.subplots_adjust()
				fig.tight_layout()
				options = dict(fname=join(directory,'%s.%s.%s'%(file,attrs[attr]['x'],attrs[attr]['y']),ext='pdf'),bbox_inches='tight',pad_inches=0.2)
				fig.savefig(**options)

	return


if __name__ == '__main__':
	args = []
	kwargs = {}

	args.extend(sys.argv[1:])
	kwargs.update({})

	main(*args,**kwargs)