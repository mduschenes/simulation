#!/usr/bin/env python

# Import python modules
import os,sys,copy,warnings,itertools,inspect
from copy import deepcopy
import json,glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import user modules
paths = set([os.getcwd(),os.path.abspath(os.path.dirname(__file__)),os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))])
sys.path.extend(paths)
from texify import Texify

warnings.simplefilter('ignore', (UserWarning,DeprecationWarning,FutureWarning))


# Global Variables
DELIMITER='__'

# Update nested elements
def updater(iterable,elements,_copy=False,_clear=True,_func=None):
	if not callable(_func):
		_func = lambda key,iterable,elements: elements[key]
	if _clear and elements == {}:
		iterable.clear()
	if not isinstance(elements,(dict)):
		iterable = elements
		return
	for e in elements:
		if isinstance(iterable.get(e),dict):
			if e not in iterable:
				iterable.update({e: elements[e]})
			else:
				updater(iterable[e],elements[e],_copy=_copy,_clear=_clear,_func=_func)
		else:
			iterable.update({e:elements[e]})
	return


def flatten(iterable,types=(list,)):
	'''	
	Flatten nested iterable
	Args:
		iterable (iterable): Nested iterable
		types (type,tuple[type]): Allowed types to be flattened
	Yields:
		elements (object): Flattened iterable elements
	'''
	if not isinstance(iterable,types):
		yield iterable
	else:
		for element in iterable:
			if isinstance(iterable,dict):
				element = iterable[element]
			yield from flatten(element,types=types)

	return


# Load from path
def load(path):
	with open(path,'r') as f:
		data = json.load(f)
	return data


# List from generator
def list_from_generator(generator,field=None):
	item = next(generator)
	if field is not None:
		item = item[field]    
	items = [item]
	for item in generator:
		if field is not None:
			item = item[field]
		if item == items[0]:
			break
		items.append(item)

	# Reset iterator state:
	for item in generator:
		if field is not None:
			item = item[field]
		if item == items[-1]:
			break
	return items


# Check if obj is number
def is_number(obj):
	try:
		obj = float(obj)
		return True
	except:
		try:
			obj = int(obj)
			return True
		except:
			return False

def is_int(a,*args,**kwargs):
	'''
	Check if object is an integer number
	Args:
		a (object): Object to be checked as int
	Returns:
		out (boolean): If object is an int
	'''
	try:
		return float(a) == int(a)
	except:
		return False

# Check if obj is nan
def is_nan(obj):
	try:
		return np.isnan(obj).all()
	except:
		return False

def scinotation(number,decimals=1,base=10,order=20,zero=True,one=False,scilimits=[-1,1],error=None,usetex=False):
	'''
	Put number into scientific notation string
	Args:
		number (str,int,float): Number to be processed
		decimals (int): Number of decimals in base part of number (including leading digit)
		base (int): Base of scientific notation
		order (int): Max power of number allowed for rounding
		zero (bool): Make numbers that equal 0 be the int representation
		one (bool): Make numbers that equal 1 be the int representation, otherwise ''
		scilimits (list): Limits on where not to represent with scientific notation
		error (str,int,float): Error of number to be processed
		usetex (bool): Render string with Latex
	
	Returns:
		String with scientific notation format for number

	'''
	if not is_number(number):
		return str(number)

	try:
		number = int(number) if is_int(number) else float(number)
	except:
		string = number
		return string

	try:
		error = int(error) if is_int(error) else float(error)
	except:
		error = None

	maxnumber = base**order
	if number > maxnumber:
		number = number/maxnumber
		if int(number) == number:
			number = int(number)
		string = str(number)

	if error is not None and (np.isnan(error) or np.isinf(error)):
		# error = r'$\infty$'
		error = None
	
	if zero and number == 0:
		string = r'%d%%s%%s%%s'%(number)

	elif is_int(number):
		string = r'%s%%s%%s%%s'%(str(number))

	elif isinstance(number,(float,np.float64)):		
		string = '%0.*e'%(decimals-1,number)
		string = string.split('e')
		basechange = np.log(10)/np.log(base)
		basechange = int(basechange) if int(basechange) == basechange else basechange
		flt = string[0]
		exp = str(int(string[1])*basechange)

		if int(exp) in range(*scilimits):
			flt = '%d'%(np.ceil(int(flt)*base**(int(exp)))) if is_int(flt) else '%0.*f'%(decimals-1,float(flt)/(base**(-int(exp)))) if (one or (float(flt) != 1.0)) else ''
			string = r'%s%%s%%s%%s'%(flt)
		else:
			string = r'%s%s%s%%s%%s%%s'%('%0.*f'%(decimals-1,float(flt)) if (one or (float(flt) != 1.0)) else '',
				r'\cdot' if (one or (float(flt) != 1.0)) else '',
				'%d^{%s}'%(base,exp) if exp!= '0' else ''
				)
	
		if error is not None and not isinstance(error,str):
			if int(exp) in range(*scilimits):
				error = '%d'%(np.ceil(int(error))) if is_int(error) else '%0.*f'%(decimals-1,float(error))
			else:
				error = r'%s%s%s'%(
					'%0.*f'%(decimals-1,float(error)/(base**(int(exp)))),
					r'\cdot' if (one or (float(flt) != 1.0)) else '',
					'%d^{%s}'%(base,exp) if exp!= '0' else ''
					)

	if error is None:
		error = ''
		prefix = ''
		postfix = ''
	else:
		error = str(error)
		prefix = r'~\pm~'
		postfix = ''

	string = string%(prefix,error,postfix)

	if usetex:
		string = r'$%s$'%(string.replace('$',''))
	else:
		string = string.replace('$','')
	return string



def plot(x=None,y=None,z=None,settings={},fig=None,ax=None,mplstyle=None,texify=None):
	'''
	Plot x,y,z with settings
	settings are of the form of keys for subplots
	x,y,z data may be passed explicitly, or within settings
	Args:
		x (dict,array): x variable to plot
		y (dict,array): y variable to plot
		z (dict,array): z variable to plot
		settings (dict): Plot settings for subplot keys {key:{'fig':{},'ax':{},'style':{}}}
		fig (dict,matplotlib.figure): Existing figure or dictionary of subplots of figures to plot to {key: figure}
		ax (dict,matplotlib.axes): Existing axes or dictionary of subplots of axes to plot to {key: axes}
		mplstyle (str): Path to mplstyle file
		texify (dict,callable): Dictionary to initialize Texify class, or function to return texified string texify(string)
	Returns:
		fig (dict): dictionary of subplots of figures of plots {key: figure}
		ax (dict): dictionary of subplots of axes of plots {key: figure}
	'''
	AXIS = ['x','y','z']
	WHICH = ['major','minor']
	FORMATTER = ['formatter','locator']
	AXES = ['colorbar']
	LAYOUT = ['nrows','ncols','index','left','right','top','bottom','hspace','wspace','width_ratios','height_ratios','pad']
	NULLLAYOUT = ['index','pad']
	DIM = 2
	PATHS = {
		'plot':os.path.join(os.path.dirname(os.path.abspath(__file__)),'plot.json'),
		'mplstyle':os.path.join(os.path.dirname(os.path.abspath(__file__)),'plot.mplstyle'),		
		'mplstyle.notex':os.path.join(os.path.dirname(os.path.abspath(__file__)),'plot.notex.mplstyle'),
		}
	def _layout(settings):
		if isinstance(settings,(list,tuple)):
			return dict(zip(LAYOUT,settings))
		_layout_ = {}
		if all([k in settings for k in ['pos']]):
			pos = settings.pop('pos')
			if pos not in [None]:
				pos = str(pos)
				_layout_ = {k: int(pos[i]) for i,k in zip(range(len(pos)),LAYOUT)}
		elif all([k in settings and settings.get(k) not in [None] for k in LAYOUT]):
			_layout_ = {k: settings[k] for k in LAYOUT}
		else:
			_layout_ = {k: settings[k] for k in settings}
		if _layout_ != {}:
			settings.update(_layout_)
		else:
			settings.clear()
		return _layout_

	def _position(layout):
		if all([kwarg == _kwarg for kwarg,_kwarg in zip(LAYOUT,['nrows','ncols'])]):
			position = ((((layout['index']-1)//layout['ncols'])%layout['nrows'])+1,((layout['index']-1)%layout['ncols'])+1)
		else:
			position = (1,1)
		return position

	def _positions(layout):
		if all([kwarg == _kwarg for kwarg,_kwarg in zip(LAYOUT,['nrows','ncols'])]):
			positions = {
				'top':(1,None),'bottom':(layout['nrows'],None),
				'left':(None,1),'right':(None,layout['ncols']),
				'top_left':(1,1),'bottom_right':(layout['nrows'],layout['ncols']),
				'top_right':(1,layout['ncols']),'bottom_left':(layout['nrows'],1),
				}
		else:
			positions = {
				'top':(1,None),'bottom':(1,None),
				'left':(None,1),'right':(None,1),
				'top_left':(1,1),'bottom_right':(1,1),
				'top_right':(1,1),'bottom_left':(1,1),
				}
		return positions


	def layout(key,fig,ax,settings):
		if all([key in obj for obj in [fig,ax]]):
			return
		_layout_ = _layout(settings[key]['style']['layout'])
		add_subplot = True and (_layout_ != {})
		other = {'%s_%s'%(key,k):settings[key]['style'].get(k) for k in AXES if isinstance(settings[key]['style'].get(k),dict)}
		for k in ax:
			__layout__ = _layout(settings.get(k,{}).get('style',{}).get('layout',ax[k].get_geometry()))
			if all([_layout_[kwarg]==__layout__[kwarg] for kwarg in _layout_]):
				ax[key] = ax[k]
				add_subplot = False
				break

		if fig.get(key) is None:
			if (fig == {} or settings[key]['style'].get('unique_fig',False)) and not hasattr(ax.get(key),'figure'):
				fig[key] = plt.figure()
			elif hasattr(ax.get(key),'figure'):
				fig[key] = getattr(ax.get(key),'figure')
			else:
				k = list(fig)[0]
				fig[key] = fig[k]

		if add_subplot:					

			kwargs = {kwarg: _layout_.get(kwarg) for kwarg in LAYOUT if kwarg not in NULLLAYOUT}
			nullkwargs = {kwarg: _layout_.get(kwarg) for kwarg in LAYOUT if kwarg in NULLLAYOUT}

			for kwarg in kwargs:
				if kwarg in ['left','right','top','bottom'] and kwargs.get(kwarg) is not None and nullkwargs['pad'] is not None:
					if kwarg in ['right','top']:
						kwargs[kwarg] = max(0,kwargs[kwarg]-nullkwargs['pad'])
					elif kwarg in ['left','bottom']:
						kwargs[kwarg] = min(1,kwargs[kwarg]+nullkwargs['pad'])
				else:
					kwargs[kwarg] = kwargs[kwarg]

			gs = gridspec.GridSpec(**kwargs)

			for index,g in enumerate(gs):
				index += 1
				if index == nullkwargs['index']:
					ax[key] = fig[key].add_subplot(g)

			for k in other:
				ax[k] = fig[key].add_axes(**other[k])
		return

	def attr_texify(string,attr,kwarg,texify,**kwargs):
		def _texify(string):

			string = str(string)
				
			substring = '\n'.join(['%s'%(substring.replace('$','')) for substring in string.split('\n')])

			if not any([t in substring for t in [r'\textrm','_','^','\\']]):
				pass
				# substring = r'\textrm{%s}'%(subtring)
			# for t in ['_','^']:
			# 	substring = substring.split(t)
			# 	substring = [r'\textrm{%s}'%i  if (not (is_number(i) or any([j in i for j in ['$','textrm','_','^','\\','}','{']]))) else i for i in substring]
			# 	substring = t.join(['{%s}'%i for i in substring])
			substring = '\n'.join(['$%s$'%(substring.replace('$','')) for substring in string.split('\n')])

			if len(substring) == 0:
				substring = substring.replace('$','')
			return substring
		attrs = {
			**{'set_%slabel'%(axis):['%slabel'%(axis)]
				for axis in AXIS},
			# **{'set_%sticks'%(axis):['ticks']
			# 	for axis in AXIS},				
			**{'set_%sticklabels'%(axis):['labels']
				for axis in AXIS},	
			**{k:['label'] for k in ['plot','scatter','errorbar','axvline','axhline','vlines','hlines','plot_surface']},								
			**{'set_title':['label'],'suptitle':['t'],
			'annotate':['s'],
			'legend':['title','set_title']},
		}

		if texify is None:
			texify = _texify
		elif isinstance(texify,dict):
			Tex = Texify(**texify)
			texify = Tex.texify
			texify = lambda string,texify=texify: _texify(texify(string))
		elif callable(texify):
			pass

		if attr in attrs and kwarg in attrs[attr]:
			if attr in ['set_%sticklabels'%(axis) for axis in AXIS]:
				string = [scinotation(substring,decimals=1,usetex=True) for substring in string]
			elif isinstance(string,(str,tuple,int,float,np.integer,np.floating)):
				string = texify(string)
				if len(string.replace('$','')) == 0:
					string = ''
			elif isinstance(string,list):
				string = [texify(substring) for substring in string]
				string = ['' if len(substring.replace('$','')) == 0 else substring for substring in string]
		if isinstance(string,(str,tuple,int,float,np.integer,np.floating)):
			if isinstance(string,str) and len(string.replace('$','')) == 0:
				string = ''
		elif isinstance(string,list):
			string = ['' if isinstance(substring,str) and len(substring.replace('$','')) == 0 else substring for substring in string]

		return string


	def attr_share(value,attr,kwarg,share,**kwargs):
		
		attrs = {
			**{'set_%s'%(key):['%s'%(label)]
				for axis in AXIS 
				for key,label in [('%slabel'%(axis),'%slabel'%(axis)),
								  ('%sticks'%(axis),'ticks'),
								  ('%sticklabels'%(axis),'labels')]},
			**{k:['label'] for k in ['plot','scatter','errorbar','axvline','axhline','vlines','hlines','plot_surface']},	
			**{
				'set_title':['label'],
				'suptitle':['t'],
				'annotate':['s'],
				'legend':['handles','labels','title','set_title']},
		}					
		if ((attr in attrs) and (attr in share) and (kwarg in attrs[attr]) and (kwarg in share[attr])):
			share = share[attr][kwarg]
			if ((share is None) or 
				(not all([(k in kwargs and kwargs[k] is not None) 
					for k in ['layout']]))):
				return value
			elif isinstance(share,bool) and (not share) and (share is not None):
				if isinstance(value,list):
					return []
				else:
					return None     
			elif isinstance(share,bool) and share:
				_position_ = _position(kwargs['layout']) 
				position = _position(kwargs['layout'])
				if all([((_position_[i] is None) or (position[i]==_position_[i])) for i in range(DIM)]):
					return value
				else:
					if isinstance(value,list):
						return []
					else:
						return None     
			else:
				_position_ = _positions(kwargs['layout']).get(share,share)
				position = _position(kwargs['layout'])
				if all([((_position_[i] is None) or (position[i]==_position_[i])) for i in range(DIM)]):
					return value
				else:
					if isinstance(value,list):
						return []
					else:
						return None     						

		else:
			return value
		return

	def attr_wrap(obj,attr,settings,**kwargs):

		def attrs(obj,attr,_attr,index,_kwargs,kwargs):
			call = True
			args = []
			kwds = {}
			_args = []
			_kwds = {}

			attr_ = attr
			kwargs = deepcopy(kwargs)
			nullkwargs = []				


			if attr in ['legend']:
				handles,labels = getattr(obj,'get_legend_handles_labels')()
				handles,labels = (
					[handle[0] if isinstance(handle, matplotlib.container.ErrorbarContainer) else handle for handle,label in zip(handles,labels)],
					[label if isinstance(handle, matplotlib.container.ErrorbarContainer) else label for handle,label in zip(handles,labels)]
					)

				if len(handles)>0 and len(labels)>0:
					handles,labels = zip(
						*((handle,attr_share(attr_texify(label,attr,handle,**{**kwargs[attr],**_kwargs}),attr,handle,**{**kwargs[attr],**_kwargs})) 
							for handle,label in zip(handles,labels))
						)

				if kwargs[attr].get('update') is not None:
					update = kwargs[attr].get('update')
					if isinstance(update,str) and update.count('%s'):
						labels = [update%(label) for label in labels]
					else:
						labels = [string%(label) for stirng,label in zip(update,labels)]

				if kwargs[attr].get('join') is not None:
					n = min(len(handles),len(labels))
					k = kwargs[attr].get('join',1)
					handles = list(zip(*(handles[i*n//k:(i+1)*n//k] for i in range(k))))
					labels = labels[:n//k]
					handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None,pad=0.5)}
				else:
					handler_map = None

				if kwargs[attr].get('flip') is True:
					flip = kwargs[attr].get('flip',None)
					ncol = kwargs[attr].get('ncol',1)
					flip = lambda items,n: list(itertools.chain(*[items[i::n] for i in range(n)]))
					handles,labels = flip(handles,ncol),flip(labels,ncol)

				kwargs[attr].update(dict(zip(['handles','labels','handler_map'],[handles,labels,handler_map])))

				_kwds.update({
					'set_zorder':kwargs[attr].get('set_zorder',{'level':100}),
					'set_title':{
						**({'title': kwargs[attr].get('set_title',kwargs[attr].get('title',None)),
							'prop':{'size':kwargs[attr].get('prop',{}).get('size')},
							} 
									if 'set_title' in kwargs[attr] or 'title' in kwargs[attr] else {'title':None})},
					**{subattr: {**kwargs[attr].get(subattr,{})} for subattr in ['get_title','get_texts']},
					})


				call = (not (					
					(kwargs[attr]['handles'] == [] or kwargs[attr]['labels'] == []) or 
					(min(len(kwargs[attr]['handles']),len(kwargs[attr]['labels']))==1) or 
					all([kwargs[attr][k] is None for k in kwargs[attr]]))
					or
					('set_label' in kwargs[attr]) and (kwargs[attr].get('set_label',None) is True)
					)

				nullkwargs.extend(['prop','join','flip','update','set_zorder','get_zorder','set_title','title','get_title','get_texts','set_label'])
			
			elif attr in ['plot','axvline','axhline']:

				args.extend([kwargs[attr].get(k) for k in ['x','y'] if kwargs[attr].get(k) is not None])

				nullkwargs.extend(['x','y','z','xerr','yerr'])

				call = len(args)>0			


			elif attr in ['errorbar']:

				subattrs = 'set_%sscale'
				props ='%s'
				subprops = '%serr'
				for axis in AXIS:
					prop = props%(axis)
					subprop = subprops%(axis)
					subattr = subattrs%(axis)

					if (
						(kwargs[attr].get(prop) is not None) and
						(kwargs[attr].get(subprop) is not None) and
						(kwargs.get(subattr) is not None) and
						(kwargs.get(subattr,{}).get('value') in ['log'])
						):
						if np.array(kwargs[attr][subprop]).ndim == 1:
							kwargs[attr][subprop] = np.array([[k,k] for k in kwargs[attr][subprop]]).T
						else:
							kwargs[attr][subprop] = np.array([k for k in kwargs[attr][subprop]])
						
						kwargs[attr][prop] = np.array(kwargs[attr][prop])
						
						kwargs[attr][subprop] = np.array([
							kwargs[attr][prop]*(1-(kwargs[attr][prop]/(kwargs[attr][prop]+kwargs[attr][subprop][0]))),
							kwargs[attr][subprop][1]
							])

				args.extend([kwargs[attr].get(k) for k in ['x','y','yerr','xerr'] if (
					(k in kwargs[attr]) and (kwargs[attr].get(k) is not None) and True #(not all(is_nan(kwargs[attr].get(k)) for k in ['x','y','yerr','xerr']))
					)])

				nullkwargs.extend(['x','y','z','xerr','yerr'])

				call = len(args)>0			

			elif attr in ['fill_between']:

				if kwargs[attr].get('y1') is not None and kwargs[attr].get('y2') is not None:
					call = True
					args.extend([kwargs[attr].get('x'),kwargs[attr].get('y1'),kwargs[attr].get('y2')])					
				elif kwargs[attr].get('yerr') is None:
					call = False
					args.extend([kwargs[attr].get('x'),kwargs[attr].get('y'),kwargs[attr].get('y')])
				else:
					call = True
					if np.ndim(kwargs[attr].get('yerr')) == 2 and len(kwargs[attr].get('yerr'))==2:
						args.extend([kwargs[attr].get('x'),kwargs[attr].get('y')-kwargs[attr].get('yerr')[0],kwargs[attr].get('y')+kwargs[attr].get('yerr')[1]])
					else:
						args.extend([kwargs[attr].get('x'),kwargs[attr].get('y')-kwargs[attr].get('yerr'),kwargs[attr].get('y')+kwargs[attr].get('yerr')])

				nullkwargs.extend(['x','y','z','xerr','yerr','y1','y2','label'])

			elif attr in ['plot_surface','contour','contourf','scatter']:
				args.extend([kwargs[attr].get(k) for k in ['x','y','z'] if kwargs[attr].get(k) is not None])
				call = True

				nullkwargs.extend(['x','y','z','xerr','yerr','zerr'])

			elif attr in ['imshow']:
				fields = ['X','y','x']
				for field in fields:
					if field in kwargs[attr]:
						args.append(kwargs[attr].get(field))
						break

				call = True

				nullkwargs.extend(['X','x','y','xerr','yerr'])


			elif attr in ['%saxis.set_%s_%s'%(axis,which,formatter) for axis in AXIS for which in WHICH for formatter in FORMATTER]:
				axis = attr.split('.')[0].replace('axis','')
				which = attr.split('.')[1].replace('set_','').replace('_%s'%(attr.split('_')[-1]),'')
				formatter = attr.split('_')[-1]
				for k in kwargs[attr]:
					for a in kwargs[attr][k]:
						getattr(getattr(obj,'%saxis'%(axis)),'set_%s_%s'%(which,formatter))(
							getattr(getattr(matplotlib,k),a)(**kwargs[attr][k][a]))					
				call = False


			elif attr in ['set_%sbreak'%(axis) for axis in AXIS]:

				fields = ['transform']
				for field in fields:
					if field in ['transform']:
						kwargs[attr][field] = getattr(obj,kwargs[attr].get(field))

				args.extend([kwargs[attr].get(k) for k in ['x','y'] if (
				(k in kwargs[attr]) and (kwargs[attr].get(k) is not None) and True #(not all(is_nan(kwargs[attr].get(k)) for k in ['x','y','yerr','xerr']))
				)])

				nullkwargs.extend(['x','y','z','xerr','yerr','transform'])

				attr_ = 'plot'

				call = len(args)>0		

			elif attr in ['set_%snbins'%(axis) for axis in AXIS]:
				axis = attr.replace('set_','').replace('nbins','')
				which = 'major'
				formatter = 'locator'
				k = 'ticker'
				try:
					a = 'MaxNLocator'
					getattr(getattr(obj,'%saxis'%(axis)),'set_%s_%s'%(which,formatter))(
							getattr(getattr(matplotlib,k),a)(**kwargs[attr]))
				except:
					a = 'LogLocator'
					getattr(getattr(obj,'%saxis'%(axis)),'set_%s_%s'%(which,formatter))(
							getattr(getattr(matplotlib,k),a)(**kwargs[attr]))
				call = False

			# elif attr in ['%saxis.offsetText.set_fontsize'%(axis) for axis in AXIS]:
			# 	axis = attr.split('.')[0].replace('axis','')
			# 	getattr(getattr(getattr(obj,'%saxis'%(axis)),'offsetText'),'set_fontsize')(**kwargs[attr])
			# 	call = False

			elif attr in ['set_colorbar']:
				values = kwargs[attr].get('values')
				colors = kwargs[attr].get('colors')
				norm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values))  
				normed_values = norm(values)
				cmap = matplotlib.colors.LinearSegmentedColormap.from_list('colorbar', list(zip(normed_values,colors)), N=len(normed_vals)*10)  
				colorbar = matplotlib.colorbar.ColorbarBase(cax=obj, cmap=cmap, norm=norm, orientation='vertical')
				obj = colorbar
				call = True


			elif attr in ['savefig']:
				path = kwargs[attr].get('fname')
				dirname = os.path.abspath(os.path.dirname(path))
				if not os.path.exists(dirname):
					os.makedirs(dirname)
				call = True


			elif attr in ['close']:
				try:
					plt.close(obj,**kwargs[attr])
				except:
					plt.close(obj)
				call = False
			
			for kwarg in nullkwargs:
				kwargs[attr].pop(kwarg,None)


			if not call:	
				return

			fields = ['color','ecolor']
			for field in fields:
				if kwargs[attr].get(field) == '__cycle__':
					try:
						_obj = _attr[-1]
					except:
						_obj = _attr
					values = list_from_generator(getattr(getattr(obj,'_get_lines'),'prop_cycler'),field)
					kwargs[attr][field] = values[-1]
				
				elif kwargs[attr].get(field) == '__lines__':
					_obj = getattr(obj,'get_lines')()[-1]
					kwargs[attr][field] = getattr(_obj,'get_%s'%(field))()
			
				elif isinstance(kwargs[attr].get(field),str):
					value = kwargs[attr].get(field)
					i = index/(len(kwargs.get((attr,index)))) if (isinstance(kwargs.get((attr,index)),list) and len(kwargs.get((attr,index)))>1) else 0.5
					kwargs[attr][field] = getattr(plt.cm,value)(i)
				
				else:
					continue




			_obj = obj
			for a in attr_.split('.'):
				try:
					_obj = getattr(_obj,a)
				except:
					break			

			try:
				if args != []:
					_attr = _obj(*args,**kwargs[attr])
				else:
					_attr = _obj(**kwargs[attr])
			except Exception as e:
				if not isinstance(e,AttributeError):
					print(e,attr,_obj,args,kwargs[attr],[a.dtype if isinstance(a,np.ndarray) else [type(i) for i in a] for a in args])

			for k in _kwds:
				_attr_ = _attr
				for a in k.split('.')[:-1]:
					try:
						_attr_ = getattr(_attr_,a)()
					except:
						_attr_ = getattr(_attr_,a)
				a = k.split('.')[-1]

				try:
					getattr(_attr_,a)(**_kwds[k])
				except:
					try:
						plt.setp(getattr(_attr_,a)(),**_kwds[k])
					except:
						try:
							plt.setp(getattr(_attr_,a),**_kwds[k])
						except:
							try:
								for _subattr_ in getattr(_attr_,a)():
									for l in _kwds[k]:
										getattr(_subattr_,l)(**_kwds[k][l])
							except:
								pass
				

			# except:
			# 	_kwargs = inspect.getfullargspec(getattr(obj,attr))[0]
			# 	args.extend([kwargs[attr][k] for k in kwargs[attr] if k not in _kwargs])
			# 	kwargs[attr] = {k:kwargs[attr][k] for k in kwargs[attr] if k in _kwargs}
			# 	try:
			# 		getattr(obj,attr)(*args,**kwargs[attr])
			# 	except:
			# 		pass
			return _attr

		_wrapper = lambda kwarg,attr,kwargs,settings,index:{
			**kwarg,
			attr: {k: attr_share(attr_texify(kwarg[attr][k],attr,k,**kwargs),attr,k,**kwargs) for k in kwarg[attr]},
			(attr,index):settings[attr],
			}
		_attr = None

		# Convert settings (dict,nested lists of dict) to list of dicts
		if not isinstance(settings[attr],(dict,list)):
			return
			
		_kwargs = [{**settings,attr:setting} for setting in flatten(settings[attr],type=(list,))]
		
		for index,_kwarg in enumerate(_kwargs):
			_attr = attrs(obj,attr,_attr,index,kwargs,_wrapper(_kwarg,attr,kwargs,settings,index))
		return

	def obj_wrap(attr,key,fig,ax,settings):
		attr_kwargs = lambda attr,key,settings:{
			'texify':settings[key]['style'].get('texify'),
			'share':settings[key]['style'].get('share',{}).get(attr,{}),
			'layout':_layout(settings[key]['style'].get('layout',{})),
			}	
		
		matplotlib.rcParams.update(settings[key]['style'].get('rcParams',{}))

		objs = lambda attr,key,fig,ax: {'fig':fig.get(key),'ax':ax.get(key),**{'%s_%s'%('ax',k):ax.get('%s_%s'%(key,k)) for k in AXES}}[attr]
		obj = objs(attr,key,fig,ax)

		exceptions = {
			**{
				prop: {
					'settings':{'set_%sscale'%(AXIS[-1]):{'value':'log'}},
					'kwargs':{kwarg: (lambda settings,prop=prop,kwarg=kwarg,obj=obj: (np.log10(settings[prop][kwarg]))) 
														for kwarg in ['z'] if settings.get(prop,{}).get(kwarg) is not None},
					'pop':False,
					}
				for prop in ['plot_surface'] if settings.get(prop) is not None
				},
			**{
				prop: {
					'settings':{'set_%sscale'%(AXIS[-2]):{'value':'log'}},
					'kwargs':{kwarg: (lambda settings,prop=prop,kwarg=kwarg,obj=obj: ((settings[prop][kwarg]))) 
														for kwarg in ['yerr'] if settings.get(prop,{}).get(kwarg) is not None},
					'pop':False,
					}
				for prop in ['errorbar'] if settings.get(prop) is not None
				},				
			**{
				prop: {
					'settings':{'set_%sscale'%(AXIS[-1]):{'value':'log'}},
					'kwargs':{kwarg: (lambda settings,prop=prop,kwarg=kwarg,obj=obj: ([r'$10^{%d}$'%(round(t,-1)) 
														for t in (settings['set_%sticks'%(AXIS[-1])]['ticks'] if (
														'set_%sticks'%(AXIS[-1]) in settings) else (
														getattr(obj,('set_%sticks'%(AXIS[-1])).replace('set','get'))() if (
														hasattr(obj,'set_%sticks'%(AXIS[-1]).replace('set','get'))) else [0]))])) 
														for kwarg in ['labels'] if settings.get(prop,{}).get(kwarg) is not None},
					'pop':False,
					}
				for prop in ['set_%sticklabels'%(AXIS[-1])] if settings.get(prop) is not None
				},				
			**{
				prop: {
					'settings':{'set_%sscale'%(AXIS[-1]):{'value':'log'}},
					'kwargs':{},
					'pop':True,
					}
				for prop in ['set_%sscale'%(AXIS[-1])] if settings.get(prop) is not None
				},	

			}

		ordering = {'close':-1,'savefig':-2}


		if obj is not None:
			props = list(settings[key][attr])
			for prop in ordering:
				if prop in settings[key][attr]:
					if ordering[prop] == -1:
						ordering[prop] = len(props)
					elif ordering[prop] < -1:
						ordering[prop] += 1
					props.insert(ordering[prop],props.pop(props.index(prop)))

			for prop in props:
				kwargs = attr_kwargs(attr,key,settings)
				# if prop in exceptions and all([((settings[key][attr][k][l] if (k in settings[key][attr]) else (
				# 										getattr(obj,k.replace('set','get'))() if (
				# 										hasattr(obj,k.replace('set','get'))) else None))==exceptions[prop]['settings'][k][l]) 
				# 										for k in exceptions[prop]['settings'] 
				# 										for l in exceptions[prop]['settings'][k]]):
				# 	for kwarg in exceptions[prop]['kwargs']:
				# 		if isinstance(settings[key][attr][prop],dict):
				# 			settings[key][attr][prop][kwarg] = exceptions[prop]['kwargs'][kwarg](settings[key][attr])
				# 		else:
				# 			for i in range(len(settings[key][attr][prop])):
				# 				settings[key][attr][prop][i][kwarg] = exceptions[prop]['kwargs'][kwarg](
				# 					{_prop: settings[key][attr][_prop] if _prop !=prop else settings[key][attr][_prop][i] 
				# 						for _prop in settings[key][attr]})
				# 	if exceptions[prop]['pop']:
				# 		continue
				attr_wrap(obj,prop,settings[key][attr],**kwargs)
		return
		
		
	def context(x,y,z,settings,fig,ax,mplstyle,texify):
		with matplotlib.style.context(mplstyle):
			settings,fig,ax = setup(x,y,z,settings,fig,ax,mplstyle,texify)
			for key in settings:
				for attr in ['ax',*['%s_%s'%('ax',k) for k in AXES],'fig']:
					obj_wrap(attr,key,fig,ax,settings)

		return fig,ax

	def setup(x,y,z,settings,fig,ax,mplstyle,texify):

		def _index(i,N,method='row'):
			
			if method == 'row':
				return [1,N,i+1]
			if method == 'col':
				return [N,1,i+1]				
			elif method == 'grid':
				M = int(np.sqrt(N))+1 if N > 1 else 1
				return [M,M,i+1]
			else:
				return [1,N,i+1]


		_defaults = {
			'ax':{},
			'fig':{},
			'style':{
				'layout':{
					'nrows':1,'ncols':1,'index':1,
					'left':None,'right':None,'top':None,'bottom':None,
					'hspace':None,'wspace':None,
					'width_ratios':None,'height_ratios':None,
					'pad':0,
					}
				}
			}
		defaults = {'ax':{},'fig':{},'style':{}}

		if isinstance(settings,str):
			settings = load(settings)

		if settings == {}:
			settings.update({None:{}})

		update = y is not None

		if any([key in settings for key in defaults]):
			settings = {key:deepcopy(settings) for key in (y if update and isinstance(y,dict) else [None])}


		if not isinstance(y,dict):
			if not isinstance(y,tuple):
				y = (y,)
			y = {key: y for key in settings}

		if not isinstance(x,dict):
			if not isinstance(x,tuple):
				x = (x,)
			x = {key: x for key in settings}

		if not isinstance(z,dict):
			if not isinstance(z,tuple):
				z = (z,)
			z = {key: z for key in settings}

		for key in settings:
			settings[key].update({k:deepcopy(defaults[k])
				for k in defaults if k not in settings[key]})

		for i,key in enumerate(y):
			if not isinstance(settings[key]['style'].get('layout'),dict):
				settings[key]['style']['layout'] = {}
			if not all([kwarg in settings[key]['style']['layout'] for kwarg in LAYOUT[:DIM+1]]):
				settings[key]['style']['layout'].update(dict(zip([*LAYOUT[:DIM],LAYOUT[DIM]],_index(i,len(y),'row'))))
		
		for key in y:

			_settings = load(PATHS['plot'])
			updater(_settings,settings[key])

			_settings['style'].update({
				'layout':{kwarg:settings[key]['style'].get('layout',{}).get(kwarg,_defaults['style']['layout'][kwarg])
							if settings[key]['style'].get('layout',{}).get(kwarg) is None else settings[key]['style'].get('layout',{}).get(kwarg) 
							for kwarg in LAYOUT}
				})
			if update:
				plotsettings = settings[key].get('ax',{}).pop('plot',{})				
				_settings['ax'].update({
					**{'plot':[{'x':_x,'y':_y,'z':_z,**(plotsettings if isinstance(plotsettings,dict) else plotsettings[_i])} 
								for _i,(_x,_y,_z) in enumerate(zip(x.get(key,[None]*len(y[key])),y[key],z[key]))]
								},
					**settings[key].get('ax',{}), 
					})

			for attr in settings[key]:
				if attr in _settings:
					_settings[attr].update(settings[key][attr])
				# if attr in _settings:
				# 	for kwarg in list(_settings[attr]):
				# 		if kwarg not in settings[key][attr]:
				# 			_settings[attr].pop(kwarg)

			updater(settings[key],_settings)

		for key in settings:
			settings[key].update({k:defaults[k] 
				for k in defaults if k not in settings[key]})

		if fig in [None]:
			fig = {}

		if ax in [None]:
			ax = {}

		for key in settings:
			attr = 'layout'
			layout(key,fig,ax,settings)

			attr = 'style'
			for prop,obj in zip(['mplstyle','texify'],[mplstyle,texify]):
				settings[key][attr][prop] = settings[key][attr].get(prop,obj)

		return settings,fig,ax

	attr = 'style'
	prop = 'mplstyle'
	mplstyles = [*[settings[key].get(attr,{}).get(prop) for key in settings],
				settings.get(attr,{}).get(prop),
				mplstyle,PATHS[prop],matplotlib.matplotlib_fname()]

	_mplstyles = [mplstyle,PATHS['mplstyle.notex'],matplotlib.matplotlib_fname()]
	for mplstyle in mplstyles:
		if mplstyle is not None and os.path.isfile(mplstyle):
			break
	for _mplstyle in _mplstyles:
		if _mplstyle is not None and os.path.isfile(_mplstyle):
			break			

	settingss = [settings,PATHS['plot'],{}]
	for settings in settingss:
		if ((settings is not None) or (isinstance(settings,str) and os.path.isfile(settings))):
			break

	try:
		fig,ax = context(x,y,z,settings,fig,ax,mplstyle,texify)
	except:
		rc_params = {'text.usetex': False}
		matplotlib.rcParams.update(rc_params)
		matplotlib.use('pdf') 
		fig,ax = context(x,y,z,settings,fig,ax,_mplstyle,texify)

	return fig,ax



if __name__ == '__main__':
	if len(sys.argv)<2:
		exit()
	data = sys.argv[1]
	path = sys.argv[2]
	settings = sys.argv[3]
	mplstyle = sys.argv[4]
	Y = sys.argv[5].split(' ')
	X = sys.argv[6].split(' ')
	Z = sys.argv[7].split(' ')



	df = pd.concat([pd.read_csv(d) for d in glob.glob(data)],
					axis=0,ignore_index=True)

	_settings = load(settings)

	settings = {}

	for i,(x,y,z) in enumerate(zip(X,Y,Z)):
		key = y

		settings[key] = deepcopy(_settings)

		settings[key]['ax']['plot']['x'] = df[x].values if x in df else df.index.values
		settings[key]['ax']['plot']['y'] = df[y].values if y in df else df.index.values
		settings[key]['ax']['plot']['z'] = df[z].values if z in df else df.index.values
		settings[key]['ax']['set_xlabel'] = {'xlabel':x.capitalize() if x in df else None}
		settings[key]['ax']['set_ylabel'] = {'ylabel':y.capitalize() if y in df else None }
		settings[key]['ax']['set_zlabel'] = {'zlabel':z.capitalize() if y in df else None }
		settings[key]['style']['layout'] = {'ncols':len(Y),'nrows':1,'index':i}
		settings[key]['fig']['savefig'] = {'fname':path,'bbox_inches':'tight'}

	fig,ax = plot(settings=settings,mplstyle=mplstyle) 
