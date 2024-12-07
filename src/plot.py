#!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,inspect,datetime
import traceback
from copy import copy,deepcopy
from math import prod
import json,glob
import numpy as np
import pandas as pd
from natsort import natsorted

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
# improt matplotlib.lines import Line2D

# Logging
from src.logger	import Logger
logger = Logger()
info = 100	
debug = 100

# Import user modules
paths = set([os.getcwd(),os.path.abspath(os.path.dirname(__file__)),os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))])
sys.path.extend(paths)
from texify import Texify

warnings.simplefilter('ignore', (UserWarning,DeprecationWarning,FutureWarning))


# Global Variables
DIM = 2
LAYOUTDIM = 2
AXES = ['x','y','z']
VARIANTS = ['','err','1','2']
FORMATS = ['lower','upper']
ALL = ['%s%s'%(getattr(axes,fmt)(),variant) for axes in AXES for variant in VARIANTS for fmt in FORMATS]
VARIABLES = {ax: [axes for axes in ALL if axes.lower().startswith(ax.lower())] for ax in AXES}
OTHER = 'label'
WHICH = ['major','minor']
FORMATTER = ['formatter','locator']
CAXES = ['colorbar']
PLOTS = ['plot','scatter','errorbar','histogram','fill_between','axvline','axhline','vlines','hlines','plot_surface','contour','contourf','tricontour','tricontourf','imshow','matshow']
LAYOUT = ['nrows','ncols','index','left','right','top','bottom','hspace','wspace','width_ratios','height_ratios','pad']
NULLLAYOUT = ['index','pad']
OBJS = ['ax','fig']
OBJ = 'ax'
PATHS = {
	'plot':os.path.join(os.path.dirname(os.path.abspath(__file__)),'plot.json'),
	'mplstyle':os.path.join(os.path.dirname(os.path.abspath(__file__)),'plot.mplstyle'),		
	'mplstyle.notex':os.path.join(os.path.dirname(os.path.abspath(__file__)),'plot.notex.mplstyle'),
	}

scalars = (int,np.integer,float,np.floating,str,type(None))
nan = np.nan

def setter(iterable,elements,delimiter=False,copy=False,reset=False,clear=False,default=None):
	'''
	Set nested value in iterable with nested elements keys
	Args:
		iterable (dict): dictionary to be set in-place with value
		elements (dict): Dictionary of keys of delimiter separated strings, or tuple of string for nested keys, and values to set 
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		reset (bool): boolean on whether to replace value at key with value, or update the nested dictionary
		clear (bool): boolean of whether to clear iterable when the element's value is an empty dictionary
		default(callable,None,bool,iterable): Callable function with signature default(key_iterable,key_elements,iterable,elements) to modify value to be updated based on the given dictionaries, or True or False to default to elements or iterable values, or iterable of allowed types
	'''

	if (not isinstance(iterable,(dict,list))) or (not isinstance(elements,dict)):
		return

	# Setup default as callable
	if default is None:
		func = lambda key_iterable,key_elements,iterable,elements: elements.get(key_elements)
	elif default is True:
		func = lambda key_iterable,key_elements,iterable,elements: elements.get(key_elements)
	elif default is False:
		func = lambda key_iterable,key_elements,iterable,elements: iterable.get(key_iterable,elements.get(key_elements))
	elif default in ['none','None']:
		func = lambda key_iterable,key_elements,iterable,elements: elements.get(key_elements) if elements.get(key_elements) is not None else iterable.get(key_iterable,elements.get(key_elements))
	elif not callable(default):
		types = tuple(default)
		def func(key_iterable,key_elements,iterable,elements,types=types): 
			i = iterable.get(key_iterable,elements.get(key_elements))
			e = elements.get(key_elements,i)
			return e if isinstance(e,types) else i
	else:
		func = default

	# Clear iterable if clear and elements is empty dictionary
	if clear and elements == {}:
		iterable.clear()

	# Set nested elements
	for element in elements:

		# Get iterable, and index of tuple of nested element key
		i = iterable
		index = 0

		# Convert string instance of elements to list, splitting string based on delimiter delimiter
		try:
			if isinstance(element,str) and delimiter:
				e = tuple(element.split(delimiter))
			elif isiterable(element,exceptions=scalars):
				e = tuple(element)
			else:
				e = tuple((element,))

			# Update iterable with elements 
			while index<(len(e)-1):
				if isinstance(i,list):
					if (e[index] >= len(i)):
						i.extend([[] if isinstance(e[index+1],int) else {} for j in range(e[index]-len(i)+1)])
				elif (isinstance(i,dict) and (not isinstance(i.get(e[index]),(dict,list)))):
					i[e[index]] = [] if isinstance(e[index+1],int) else {}
				i = i[e[index]]
				index+=1

			value = copier(element,func(e[index],element,i,elements),copy)

			if isinstance(i,list) and (e[index] >= len(i)):
				i.extend([{} for j in range(e[index]-len(i)+1)])

			if reset:
				i[e[index]] = value
			elif e[index] not in i or not isinstance(i[e[index]],(dict,list)):
				i[e[index]] = value
			elif isinstance(elements[element],dict):
				setter(i[e[index]],elements[element],delimiter=delimiter,copy=copy,reset=reset,clear=clear,default=default)
			else:
				i[e[index]] = value
		except Exception as exception:
			pass

	return


def isiterable(obj,exceptions=()):
	'''
	Check if object is iterable
	Args:
		obj (object): object to be tested
		exceptions (tuple[type]): Exceptional iterable types to exclude
	Returns:
		iterable (bool): whether object is iterable
	'''
	return hasattr(obj,'__iter__') and not isinstance(obj,exceptions)


def copier(obj,copy):
	'''
	Copy object based on copy

	Args:
		obj (object): object to be copied
		copy (bool): boolean or None whether to copy value
	Returns:
		Copy of value
	'''

	if copy:
		return deepcopy(obj)
	else:
		return obj

def to_position(index,shape):
	'''
	Convert linear index to dimensional position
	Args:
		index (int): Linear index
		shape (iterable[int]): Dimensions of positions
	Returns:
		position (iterable[int]): Dimensional positions
	'''
	position = [index//(prod(shape[i+1:]))%(shape[i]) for i in range(len(shape))]
	return position

def to_index(position,shape):
	'''
	Convert dimensional position to linear index
	Args:
		position (iterable[int]): Dimensional positions
		shape (iterable[int]): Dimensions of positions
	Returns:
		index (int): Linear index
	'''	
	index = sum((position[i]*(prod(shape[i+1:])) for i in range(len(shape))))
	return index


def getter(iterable,elements,default=None,delimiter=False,copy=False):
	'''
	Get nested value in iterable with nested elements keys

	Args:
		iterable (dict): dictionary of values
		elements (str,iterable[str]): delimiter separated string or list to nested keys of location to get value
		default (object): default data to return if elements not in nested iterable
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
	Returns:
		value (object): Value at nested keys elements of iterable
	'''	

	# Convert string instance of elements to list, splitting string based on delimiter delimiter
	if isinstance(elements,str) and delimiter:
		elements = elements.split(delimiter)

	# Get nested element if iterable, based on elements
	if not isinstance(elements,(list,tuple)):
		# elements is object and value is to be got from iterable at first level of nesting
		try:
			return copier(elements,iterable[elements],copy)
		except:
			return default
	elif not elements:
		return copier(elements,iterable,copy)
	else:
		# elements is list of nested keys and the nested values are to be extracted from iterable
		try:
			i = iterable
			e = 0
			while e<len(elements):
				i = i[elements[e]]
				e+=1			
			return copier(elements[e-1],i,copy)
		except:
			return default

	return default



def search(iterable,index=[],shape=[],returns=None,items=None,types=(list,),exceptions=()):
	'''
	Search of iterable, returning keys and indices of keys
	Args:
		iterable (iterable): Nested iterable
		index (iterable[int,str]): Index of key
		shape (iterable[int]): Shape of iterable
		returns (bool,str): Returns of search, 
			None returns item, True returns index,shape,item, False returns None, 
			allowed strings (.delimited) for combinations of ['index','shape','item']
		types (type,tuple[type]): Allowed types to be searched
		exceptions (type,tuple[type]): Disallowed types to be searched
	Yields:
		index (iterable[int,str]): Index of item
		shape (iterable[iterable[int]]): Shape of iterable at index
		item (iterable): Iterable key
	'''
	def returner(index,shape,item,returns=None):
		if returns is None:
			yield item
		elif returns is True:
			yield (index,shape,item)
		elif returns is False:
			return None
		elif returns in ['index']:
			yield index
		elif returns in ['shape']:
			yield shape
		elif returns in ['item']:
			yield item
		elif returns in ['index.shape']:
			yield (index,shape)
		elif returns in ['index.item']:
			yield (index,item)
		elif returns in ['shape.item']:
			yield (shape,item)
		elif returns in ['index.shape.item']:
			yield (index,shape,item)

	dictionaries = (dict,)
	items = [items] if (items is not None) and isinstance(items,scalars) else items
	if (not isinstance(iterable,types)) or (isinstance(iterable,exceptions)) or (items and isinstance(iterable,types) and all(item in iterable for item in items)):
		
		if items:
			if (not isinstance(iterable,types)) or (isinstance(iterable,exceptions)):
				return
			elif isinstance(iterable,dictionaries):
				item = [iterable[item] for item in items]
			else:
				item = items
		else:
			item = iterable

		yield from returner(index,shape,item,returns=returns)


	if (isinstance(iterable,types)) and (not isinstance(iterable,exceptions)):
		for i,item in enumerate(iterable):
			if isinstance(iterable,dictionaries):
				i,item = item,iterable[item]
			size = len(iterable)					
			yield from search(item,index=[*index,i],shape=[*shape,size],
				returns=returns,items=items,types=types,exceptions=exceptions)


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

def is_float(a,*args,**kwargs):
	'''
	Check if object is a float number
	Args:
		a (object): Object to be checked as float
	Returns:
		out (boolean): If object is a float
	'''
	try:
		a = float(a)
		return True
	except:
		return False

# Check if obj is nan
def is_nan(obj):
	try:
		return np.isnan(obj).all()
	except:
		return False

# Check if obj is inf
def is_inf(obj):
	try:
		return np.isinf(obj).all()
	except:
		return False	

# Check if obj is nan or inf
def is_naninf(obj):
	return is_nan(obj) or is_inf(obj)


def is_number(a,*args,**kwargs):
	'''
	Check if object is an integer float number
	Args:
		a (object): Object to be checked as number
	Returns:
		out (boolean): If object is a number
	'''
	return is_int(a,*args,**kwargs) or is_float(a,*args,**kwargs)

def to_number(a,dtype=None,**kwargs):
	'''
	Convert object to number
	Args:
		a (int,float,str): Object to convert to number
		dtype (data_type): Datatype of number
	Returns:
		number (object): Number representation of object
	'''
	prefixes = {'-':-1}
	dtypes = {'int':int,'float':float}

	coefficient = 1
	number = a
	dtype = dtypes.get(dtype,dtype)
	if isinstance(a,str):
		for prefix in prefixes:
			if a.startswith(prefix):
				a = prefix.join(a.split(prefix)[1:])
				coefficient *= prefixes[prefix]
		if is_int(a):
			dtype = int
		elif is_float(a):
			dtype = float
		if is_number(a):
			number = coefficient*float(a)
	return number


def allclose(a,b,rtol=1e-05,atol=1e-08,equal_nan=False):
	'''
	Check if arrays a and b are all close within tolerance
	Args:
		a (array): Array to compare with
		b (array): Array to compare
		rtol (float): Relative tolerance of arrays
		atol (float): Absolute tolerance of arrays
		equal_nan (bool): Compare nan's as equal
	Returns:
		out (bool): Boolean of whether a and b are all close
	'''
	return np.allclose(a,b,rtol,atol,equal_nan)	




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

	if error is not None and (is_nan(error) or is_inf(error)):
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



def set_color(value=None,color=None,values=[],norm=None,scale=None,alpha=None,**kwargs):
	'''
	Set color
	Args:
		value (int,float,iterable[int,float]): Value to process color
		color (str): color
		values (iterable[int,float]): Values to process colors
		norm (iterable[int,float],dict[str,[int,float]]): Range of values, either iterable [vmin,vmax] or dictionary {'vmin':vmin,'vmax':vmax}
		scale (str): Scale type for normalization, allowed strings in ['linear','log','symlog']
		alpha (int,float,iterable[int,float]): Alpha of color
		kwargs (dict): Additional keyword arguments
	Returns:
		value (int,float,iterable[int,float]): Normalized values corresponding to color
		color (str,tuple,array): colors of value
		values (int,float,iterable[int,float]): Normalized values corresponding to color
		colors (str,tuple,array): colors of values
		norm (callable): Normalization function, with signature norm(value)
	'''

	if isinstance(value,str):
		value,color,values,colors,norm = None,None,None,None,None
		return value,color,values,colors,norm

	if value is None:
		value = values
	if color is None:
		color = 'viridis'

	if value is None:
		value = []

	if isinstance(values,scalars):
		values = [values]

	values = [i for i in values if not ((i is None) or is_naninf(i))]
	norm = ({**norm,**{
				 'vmin':norm.get('vmin',min(values,default=0)),
				 'vmax':norm.get('vmax',max(values,default=1))}} if isinstance(norm,dict) else 
				{'vmin':norm[0],
				 'vmax':norm[1]} if norm is not None else
				{'vmin':min(values,default=0),
				 'vmax':max(values,default=1)})

	values = [i for i in natsorted(set([*values,*[norm['vmin'],norm['vmax']]])) if is_number(i)]
	norm.update(dict(zip(['vmin','vmax'],[min(values,default=0),max(values,default=1)])))

	if not isinstance(value,scalars):
		value = list(natsorted(set([*value])))

	if scale in ['linear',None]:
		norm = matplotlib.colors.Normalize(**norm)  
	elif scale in ['log','symlog']:
		values = [i for i in values if is_number(i) and i>0]
		norm.update(dict(zip(['vmin','vmax'],[min(values,default=0),max(values,default=1)])) if values else {})
		norm = {i:norm[i] if norm[i]>0 else 1e-20 for i in norm}
		norm = matplotlib.colors.LogNorm(**norm)  
	else:
		norm = matplotlib.colors.Normalize(**norm)					

	try:
		value = norm(value)
		values = norm(values)
	except:
		value = None
		values = None

	if hasattr(plt.cm,color):
		try:
			colors = getattr(plt.cm,color)(values)
			color = getattr(plt.cm,color)(value)
		except:
			colors = color
			color = color
		
		if isinstance(color,tuple):
			color = list(color)
			color[-1] = alpha
			color = tuple(color)
		elif isinstance(color,np.ndarray):
			color[:,-1] = alpha

		if isinstance(colors,tuple):
			colors = list(colors)
			colors[-1] = alpha
			colors = tuple(colors)
		elif isinstance(colors,np.ndarray):
			colors[:,-1] = alpha	

	else:
		colors = color
		color = color		

	return value,color,values,colors,norm

def set_data(data=None,scale=None,**kwargs):
	'''
	Set data
	Args:
		data (int,float,iterable[int,float]): Data
		scale (str): Scale type for normalization, allowed strings in ['linear','log','symlog']
		kwargs (dict): Additional keyword arguments
	Returns:
		data (array): Data
	'''

	if isinstance(scale,str):
		scale = [scale]

	if ((data is None) or 
	   (scale is None)):
	
	   data = None

	elif ((scale is None) or
		  (not any(i in ['log','symlog'] for i in scale))):
	
		data = data
	
	elif ((not isinstance(scale,str) and any(i in ['log','symlog'] for i in scale))):

		data = np.array(data)
		data[data==0] = np.nan

	return data


def set_err(err=None,value=None,scale=None,**kwargs):
	'''
	Set error
	Args:
		err (int,float,iterable[int,float],iterable[iterable[int,float]]): Error, either scalar, or iterable of scalars for equal +- errors, or iterable of 2 iterables for independent +- errors
		value (int,float,iterable[int,float]): Value to process error
		scale (str): Scale type for normalization, allowed strings in ['linear','log','symlog']
		kwargs (dict): Additional keyword arguments
	Returns:
		err (iterable[iterable[int,float]]): Errors normalized as per value and scale
	'''

	if isinstance(scale,str):
		scale = [scale]

	if ((err is None) or (value is None) or 
	   (scale is None)):
	
	   err = None

	elif ((scale is None) or
		  (not any(i in ['log','symlog'] for i in scale))):
	
		if allclose(err,0):
			err = None
		else:
			err = err
	
	elif ((not isinstance(scale,str) and any(i in ['log','symlog'] for i in scale))):		
		if isinstance(err,scalars):
			err = [err]*2
		elif is_naninf(err):
			err = [err]*2
		else:
			err = np.array(err)
			value = np.array(value)
			if err.size == 1:
				err = [err]*2
			elif err.ndim == 1:
				if err.size == value.size:
					err = [[i if i is not None and not is_naninf(i) else nan for i in err]]*2
				else:
					err = [i if i is not None and not is_naninf(i) else nan for i in err]
			elif err.ndim == 2:
				err = [[j if j is not None and not is_naninf(j) else nan for j in i] for i in err]

		err = np.array(err)
		value = np.array(value)

		if allclose(err,0):
			err = None
		else:
			err = np.array([value*(1-(value/(value+err[1]))),np.ones(value.shape)*err[1]])
	else:
	
		err = None

	if err is not None:
		err = np.abs(err)

	return err

def get_obj(obj,attr=None):
	'''
	Return object relative to to obj
	Args:
		obj (object): Object instance
		attr (str): attribute
	Returns:
		instance (object): Object instance
	'''

	if attr is None:
		instance = obj
	elif attr in ['twin%s'%(axes) for axes in AXES]:
		instance = None
		axes = attr[-1]
		siblings = getattr(obj,"get_shared_%s_axes"%(axes))().get_siblings(obj)
		for sibling in siblings:
			if sibling.bbox.bounds == obj.bbox.bounds and sibling is not obj:
				instance = sibling
				break
		if instance is None:
			instance = getattr(obj,attr)()
	else:
		instance = obj

	return instance


def get_children(obj,attr):
	'''
	Return all children of attribute from obj
	Args:
		obj (object): Object instance
		attr (str): attribute
	Yields:
		children (object): Children instances
	'''
	if attr in ['legendHandles']:

		try:
			tree = obj._legend_box.get_children()[1]
			for column in tree.get_children():
				for row in column.get_children():
					for i in row.get_children()[0].get_children():
						yield i
		except:
			tree = getattr(obj,attr,[])
			for i in tree:
				yield i
	else:
		tree = getattr(obj,attr,[])
		for i in tree:
			yield i


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

	def _positions(layout=None):
	
		attr = 'indices'

		if layout is None:
			positions = {
				'top':(1,None),'bottom':(1,None),'middle':(None,None),
				'left':(None,1),'right':(None,1),'centre':(None,None),
				'top_left':(None,None),'bottom_left':(None,None),'middle_left':(None,None),
				'top_right':(None,None),'bottom_right':(None,None),'middle_right':(None,None),
				'top_centre':(None,None),'bottom_centre':(None,None),'middle_centre':(None,None),
				}
		elif all([kwarg == _kwarg and _kwarg in layout for kwarg,_kwarg in zip(LAYOUT,['nrows','ncols'])]):
			
			if layout.get(attr):
				positions = [((((i-1)//layout['ncols'])%layout['nrows'])+1,((i-1)%layout['ncols'])+1) for i in layout[attr]]
				positions = [{index:list(set(tuple(i[axis] for i in positions 
					if all(i[j]==index[k] for k,j in enumerate(axes)))))
					for index in sorted(set(tuple(i[j] for j in axes) for i in positions))}
					for axis,axes in {i:[j for j in range(LAYOUTDIM) if j not in [i]] for i in range(LAYOUTDIM)}.items()]
				positions = {
					'top':[[*i[:0],min(positions[0][i]),*i[0:]] for i in positions[0]],
					'bottom':[[*i[:0],max(positions[0][i]),*i[0:]] for i in positions[0]],
					'middle':[[*i[:0],positions[0][i][len(positions[0][i])//2],*i[0:]] for i in positions[0]],
					'left':[[*i[:1],min(positions[1][i]),*i[1:]] for i in positions[1]],
					'right':[[*i[:1],max(positions[1][i]),*i[1:]] for i in positions[1]],
					'centre':[[*i[:1],positions[1][i][len(positions[1][i])//2],*i[1:]] for i in positions[1]],					
					}
				positions = {
					**{index: positions[index] for index in ['top','bottom','middle']},
					**{index: [i for i in positions[index] if i[1] == min(i[1] for i in positions[index])] for index in ['left']},
					**{index: [i for i in positions[index] if i[1] == max(i[1] for i in positions[index])] for index in ['right']},
					**{index: positions[index] for index in ['centre']},
					}					
				positions = {
					**{index: positions[index] for index in positions},
					**{'%s_%s'%(row,col):[i for i in positions[row]+positions[col] if i in positions[row] and i in positions[col]]
						for row in ['top','bottom','middle'] for col in ['left','right','centre']},
				}

			else:
				positions = {
					'top':(1,None),'bottom':(layout['nrows'],None),'middle':(layout['nrows']//2+layout['nrows']%2,None),
					'left':(None,1),'right':(None,layout['ncols']),'centre':(None,layout['ncols']//2+layout['ncols']%2),
					'top_left':(1,1),'bottom_left':(layout['nrows'],1),'middle_left':(layout['nrows']//2+layout['nrows']%2,1),
					'top_right':(1,layout['ncols']),'bottom_right':(layout['nrows'],layout['ncols']),'middle_right':(layout['nrows']//2+layout['nrows']%2,layout['ncols']),
					'top_centre':(1,layout['ncols']//2+layout['ncols']%2),'bottom_centre':(layout['nrows'],layout['ncols']//2+layout['ncols']%2),'middle_center':(layout['nrows']//2+layout['nrows']%2,layout['ncols']//2+layout['ncols']%2),
					}
				
		else:
			positions = {
				'top':(1,None),'bottom':(1,None),'middle':(None,None),
				'left':(None,1),'right':(None,1),'centre':(None,None),
				'top_left':(None,None),'bottom_left':(None,None),'middle_left':(None,None),
				'top_right':(None,None),'bottom_right':(None,None),'middle_right':(None,None),
				'top_centre':(None,None),'bottom_centre':(None,None),'middle_centre':(None,None),
				}
		return positions


	def layout(key,fig,ax,settings):
		if all([key in obj for obj in [fig,ax]]):
			return
		_layout_ = _layout(settings[key]['style']['layout'])
		add_subplot = True and (_layout_ != {})
		other = {'%s_%s'%(key,k):settings[key]['style'].get(k) for k in CAXES if isinstance(settings[key]['style'].get(k),dict)}
		for k in ax:
			try:
				geometry = ax[k].get_geometry()
			except Exception as exception:
				geometry = ax[k].get_subplotspec().get_geometry()
				geometry = [*geometry[:LAYOUTDIM],to_index(geometry[LAYOUTDIM:],geometry[:LAYOUTDIM][::-1])]
			__layout__ = _layout(settings.get(k,{}).get('style',{}).get('layout',geometry))
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

	def attr_texify(string,attr,kwarg,texify=None,**kwargs):
		def _texify(string):

			string = str(string)
				
			substring = '\n'.join(['%s'%(substring.replace('$','')) if (len(substring.replace('$',''))>0) else r'~' for substring in string.split('\n')])

			if not any([t in substring for t in [r'\textrm','_','^','\\']]):
				pass
				# substring = r'\textrm{%s}'%(subtring)
			# for t in ['_','^']:
			# 	substring = substring.split(t)
			# 	substring = [r'\textrm{%s}'%i  if (not (is_number(i) or any([j in i for j in ['$','textrm','_','^','\\','}','{']]))) else i for i in substring]
			# 	substring = t.join(['{%s}'%i for i in substring])
			substring = '\n'.join(['$%s$'%(substring.replace('$','')) if (len(substring.replace('$',''))>0) else r'~' for substring in string.split('\n')])

			if len(substring) == 0:
				substring = substring.replace('$','')
			return substring
		attrs = {
			**{'set_%slabel'%(axes):['%slabel'%(axes)]
				for axes in AXES},
			# **{'set_%sticks'%(axes):['ticks']
			# 	for axes in AXES},				
			**{'set_%sticklabels'%(axes):['labels']
				for axes in AXES},	
			**{k:[OTHER] for k in PLOTS},								
			**{'set_title':[OTHER],'suptitle':['t'],
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

		if ((attr in attrs) and 
			((isinstance(attrs[attr],list) and (kwarg in attrs[attr])) or 
			 (isinstance(attrs[attr],dict) and (kwarg in attrs[attr])))):
			if attr in ['set_%sticklabels'%(axes) for axes in AXES]:
				string = [scinotation(substring,decimals=1,usetex=True) if '$' not in substring else substring for substring in string]
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
				for axes in AXES 
				for key,label in [('%slabel'%(axes),'%slabel'%(axes)),
								  ('%sticks'%(axes),'ticks'),
								  ('%sticklabels'%(axes),'labels')]},
			**{k:[OTHER] for k in PLOTS},	
			**{
				'set_title':['label'],
				'suptitle':['t'],
				'annotate':['s'],
				'set_colorbar':['value','values'],
				"tick_params":["axis","which","length","width"],
				'legend':['handles','labels','title','set_title']
				},
			}

		def returns(value,attr,kwarg):
			if attr in ['set_%sticklabels'%(axes) for axes in AXES]:
				if kwarg in ['labels']:
					return []
				else:
					return None
			elif isinstance(value,list):
				return []
			else:
				return None			

		if ((attr in attrs) and (kwarg not in attrs[attr])):
			return value

		if ((attr in attrs) and (attr in share) and (share.get(attr) is not None) and (isinstance(share.get(attr),(bool,str,list,tuple))) or ((kwarg in attrs.get(attr,[])) and (kwarg in share.get(attr,[])))):

			if isinstance(share[attr],dict):
				share = share[attr][kwarg]
			else:
				share = share[attr]
			if ((share is None) or 
				(not all([(k in kwargs and kwargs[k] is not None) 
					for k in ['layout']]))):
				return value
			if isinstance(share,bool) and (not share) and (share is not None):
				return returns(value,attr,kwarg)
			elif isinstance(share,bool) and share:
				_position_ = _position(kwargs['layout']) 
				position = _position(kwargs['layout'])
				if all([((_position_[i] is None) or (position[i]==_position_[i])) for i in range(LAYOUTDIM)]):
					return value
				else:
					return returns(value,attr,kwarg)
			elif (isinstance(share,str) and share in _positions()) or isinstance(share,(list,tuple)):
				_position_ = _positions(kwargs['layout']).get(share,share) if isinstance(share,str) else share
				position = _position(kwargs['layout'])
				_position_ = (_position_,) if all(i is None or isinstance(i,int) for i in _position_) else _position_
				if any(all(pos[i] is None or pos[i] == position[i] for i in range(LAYOUTDIM)) for pos in _position_):
					return value
				else:
					return returns(value,attr,kwarg)
			else:
				if value == share:
					return value
				else:
					return returns(value,attr,kwarg)
		else:
			return value
		return

	def attr_wrap(obj,attr,objs,settings,**kwargs):

		def attrs(obj,attr,objs,index,indices,shape,count,_kwargs,kwargs):
			call = True
			args = []
			kwds = {}
			_args = []
			_kwds = {}


			fields = {
				'set_colorbar':slice(-2,None),
				'color':slice(-2,None),
				'ecolor':slice(-2,None),
				'c':slice(-2,None),
				'marker':slice(-4,-3),
				'linestyle':slice(-5,-4),
				'alpha':slice(-6,-5),
				}

			attr_ = attr
			kwargs = deepcopy(kwargs)
			nullkwargs = []				
			nullkwarg = ['call']


			attribute = kwargs[attr].pop('obj',None)
			obj = get_obj(obj,attribute)

			if attr in ['legend']:

				if kwargs[attr].get('merge') is not None:
					handles_labels = [getattr(ax,'get_legend_handles_labels')() for ax in obj.get_figure().axes]
					handles_labels = [sum(i, []) for i in zip(*handles_labels)]
				else:
					handles_labels = getattr(obj,'get_legend_handles_labels')()

				handles,labels = handles_labels
				handles,labels = (
					[handle[0] if isinstance(handle, matplotlib.container.ErrorbarContainer) else handle for handle,label in zip(handles,labels)],
					[label if isinstance(handle, matplotlib.container.ErrorbarContainer) else label for handle,label in zip(handles,labels)]
					)
				handler_map = {}

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
						labels = [string%(label) for string,label in zip(update,labels)]


				if kwargs[attr].get('handlers') is not None:
					handlers = kwargs[attr].get('handlers',{})
					if not isinstance(handlers,dict):
						handlers = {handler:{} for handler in handlers}
					funcs = {
						**{
						container.lower():{
							getattr(matplotlib.container,'%sContainer'%(container.capitalize())): getattr(matplotlib.legend_handler,'Handler%s'%(container.capitalize()))
							}
						for container in ['Errorbar']
						},
						**{
						'Errorbar'.lower():{
							getattr(matplotlib.container,'%sContainer'%('Errorbar'.capitalize())): getattr(matplotlib.legend_handler,'Handler%s'%('Errorbar'.capitalize()))
							}
						for container in ['cmap']
						},						
						}
					for handler in handlers:
						for _obj in [i for i in objs if i is not None]:
							func = funcs.get(handler)
							if func is not None:
								handler_map.update({types: func[types](**handlers[handler]) for types in func if isinstance(_obj,types)})
				if kwargs[attr].get('join') is not None:
					n = min(len(handles),len(labels))
					k = kwargs[attr].get('join',1)
					handles = list(zip(*(handles[i*n//k:(i+1)*n//k] for i in range(k))))
					labels = labels[:n//k]
					handler_map.update({tuple: matplotlib.legend_handler.HandlerTuple(None,pad=0.5)})

				if kwargs[attr].get('flip') is True:
					flip = kwargs[attr].get('flip',None)
					ncol = kwargs[attr].get('ncol',1)
					flip = lambda items,n: list(itertools.chain(*[items[i::n] for i in range(n)]))
					handles,labels = flip(handles,ncol),flip(labels,ncol)
				if kwargs[attr].get('keep') is not None:
					keep = kwargs[attr]['keep']
					unique = list(sorted(set(labels),key=lambda i: labels.index(i)))
					if unique:
						indexes = [[i for i,label in enumerate(labels) if label==value] for value in unique]
						keep = [keep]*len(indexes) if isinstance(keep,(str,int)) else keep
						elements = []
						for k,i in zip(keep,indexes):

							if k in ['first']:
								elements.append(0)
							elif k in ['middle']:
								elements.append(len(i)//2)
							elif k in ['last']:
								elements.append(-1)
							elif isinstance(k,int):
								elements.append(k)
							else:
								elements.append(k)
						if elements is not None:
							labels,handles = [labels[i[j]] for i,j in zip(indexes,elements)],[handles[i[j]] for i,j in zip(indexes,elements)]
				if kwargs[attr].get('multiline') is True:
					pass
				if kwargs[attr].get('sort') is not None:

					if isinstance(kwargs[attr].get('sort'),dict): 

						funcs = {
							**{
							container.lower():{
								getattr(matplotlib.container,'%sContainer'%(container.capitalize())): getattr(matplotlib.legend_handler,'Handler%s'%(container.capitalize()))
								}
							for container in ['Errorbar']
							},
						}

						indexes = [_obj['count'] for i,_obj in enumerate(objs) 
							if ((_obj is not None) and any(isinstance(_obj['obj'],types) and 
								(getattr(_obj['obj'],'get_label',lambda:None)() is not None)
								for handler in funcs for types in funcs[handler]))]
						indexes = [indexes.index(kwargs[attr].get('sort').get(i)) for i in kwargs[attr].get('sort') if kwargs[attr].get('sort').get(i) in indexes]

						handles,labels = [handles[i] for i in indexes],[labels[i] for i in indexes]

				if kwargs[attr].get('set_color') is not None:

					if isinstance(kwargs[attr].get('set_color'),dict): 

						funcs = {
							**{
							container.lower():{
								getattr(matplotlib.container,'%sContainer'%(container.capitalize())): getattr(matplotlib.legend_handler,'Handler%s'%(container.capitalize()))
								}
							for container in ['Errorbar']
							},
						}

						indexes = [_obj['count'] for i,_obj in enumerate(objs) 
							if ((_obj is not None) and any(isinstance(_obj['obj'],types) and 
								(getattr(_obj['obj'],'get_label',lambda:None)() is not None)
								for handler in funcs for types in funcs[handler]))]
						indexes = [kwargs[attr].get('set_color').get(i) for i in kwargs[attr].get('set_color') if kwargs[attr].get('set_color').get(i) in indexes]

						kwargs[attr]['set_color'] = [
							(set_color(**kwargs[attr].get('set_color')[i])[1] 
							if isinstance(kwargs[attr].get('set_color')[i],dict) else kwargs[attr].get('set_color')[i])
							if not isinstance(kwargs[attr].get('set_color'),scalars) 
							else kwargs[attr].get('set_color')
							for i in indexes
							]

				if ('handles' in kwargs[attr]) and (not kwargs[attr]['handles']):
					handles = []
				if ('labels' in kwargs[attr]) and (not kwargs[attr]['labels']):
					labels = []
				kwargs[attr].update(dict(zip(['handles','labels','handler_map'],[handles,labels,handler_map])))
				_kwds.update({
					'set_zorder':kwargs[attr].get('set_zorder',{'level':100}),
					'set_title':{
						**({'title': kwargs[attr].get('set_title',kwargs[attr].get('title',None)),
							'prop':{'size':kwargs[attr].get('prop',{}).get('size')},
							} 
									if 'set_title' in kwargs[attr] or 'title' in kwargs[attr] else {'title':None})},
					**{subattr: {**kwargs[attr].get(subattr,{})} for subattr in ['get_title','get_texts']},
					**({'legendHandles': {
							'set_%s'%(prop): [kwargs[attr]['set_%s'%(prop)]]*len(handles) if isinstance(kwargs[attr]['set_%s'%(prop)],scalars) else kwargs[attr]['set_%s'%(prop)]
							for prop in ['alpha','color']
							if kwargs[attr].get('set_%s'%(prop)) is not None}} if any(kwargs[attr].get('set_%s'%(prop)) is not None for prop in ['alpha','color']) else {})
					})

				call = (
					(not (
					(( (not kwargs[attr]['handles']) or (not kwargs[attr]['labels'])) or 
					(all([kwargs[attr][k] is None for k in kwargs[attr]]))) or
					((min(len(kwargs[attr]['handles']),len(kwargs[attr]['labels']))>=1) and
					(('set_label' in kwargs[attr]) and (kwargs[attr].get('set_label',None) is False)))
					)) or (
					(min(len(kwargs[attr]['handles']),len(kwargs[attr]['labels']))==1) and 
					(kwargs[attr].get('set_label',True)))
					)

				nullkwargs.extend(['prop','join','merge','flip','update','keep','sort','multiline','texify','handlers','set_zorder','get_zorder','set_title','set_alpha','set_color','title','get_title','get_texts','set_label'])

			elif attr in ['plot']:
				dim = 2
		
				props = '%s'
				subattrs = 'set_%sscale'
				for axes in AXES[:dim]:
					prop = props%(axes)
					subattr = subattrs%(axes)
					
					if kwargs[attr].get(prop) is None:
						continue

					data = kwargs[attr].get(prop)
					scale = [tmp[-1].get('value') for tmp in search(kwargs.get(subattr),returns=True) if tmp is not None and tmp[-1] is not None]
					
					data = set_data(data=data,scale=scale)

					kwargs[attr][prop] = data

				args.extend([kwargs[attr].get('%s%s'%(k,s)) for s in VARIANTS[:1] for k in AXES[:dim] if ((kwargs[attr].get('%s%s'%(k,s)) is not None))])

				nullkwargs.extend([*['%s%s'%(k,s) for s in VARIANTS[:2] for k in AXES],*[]])

				call = len(args)>0		

			elif attr in ['axvline','axhline']:
				dim = 1
		
				props = '%s'
				subattrs = 'set_%sscale'
				for axes in AXES[:dim]:
					prop = props%(axes)
					subattr = subattrs%(axes)
					
					if kwargs[attr].get(prop) is None:
						continue

					data = kwargs[attr].get(prop)
					scale = [tmp[-1].get('value') for tmp in search(kwargs.get(subattr),returns=True) if tmp is not None and tmp[-1] is not None]
					
					data = set_data(data=data,scale=scale)

					kwargs[attr][prop] = data

				if attr in ['axvline']:
					args.extend([kwargs[attr].get(k) for k in [AXES[0]] if k in kwargs[attr]])
				elif attr in ['axhline']:
					args.extend([kwargs[attr].get(k) for k in [AXES[1]] if k in kwargs[attr]])

				args = [arg if isinstance(arg,scalars) else np.mean(arg) for arg in args if arg is not None and (isinstance(arg,scalars) or len(arg))]

				nullkwargs.extend([*['%s%s'%(k.upper(),s) for s in VARIANTS[:] for k in AXES[:]],*['%s%s'%(k,s) for s in VARIANTS[:] for k in AXES],*[]])

				call = len(args)>0			


			elif attr in ['errorbar']:
				dim = 2
			
				props = '%s'
				subattrs = 'set_%sscale'
				for axes in AXES[:dim]:
					prop = props%(axes)
					subattr = subattrs%(axes)

					if kwargs[attr].get(prop) is None:
						continue

					data = kwargs[attr].get(prop)
					scale = [tmp[-1].get('value') for tmp in search(kwargs.get(subattr),returns=True) if tmp is not None and tmp[-1] is not None]
					
					data = set_data(data=data,scale=scale)

					kwargs[attr][prop] = data


				props = '%serr'
				subprops ='%s'
				subattrs = 'set_%sscale'
				for axes in AXES[:dim]:
					prop = props%(axes)
					subprop = subprops%(axes)
					subattr = subattrs%(axes)

					err = kwargs[attr].get(prop)
					value = kwargs[attr].get(subprop)
					scale = [tmp[-1].get('value') for tmp in search(kwargs.get(subattr),returns=True) if tmp is not None and tmp[-1] is not None]
					
					err = set_err(err=err,value=value,scale=scale)

					kwargs[attr][prop] = err



				args.extend([kwargs[attr].get('%s%s'%(k,s)) for s in VARIANTS[:2] for k in AXES[:dim] if ((kwargs[attr].get('%s%s'%(k,s)) is not None))])

				nullkwargs.extend([*['%s%s'%(k,s) for s in VARIANTS[:2] for k in AXES],*[]])
				
				call = len(args)>0			

			elif attr in ['fill_between']:

				dim = 2
				
				props = '%s'
				subattrs = 'set_%sscale'
				for axes in AXES[:dim]:
					prop = props%(axes)
					subattr = subattrs%(axes)
					
					if kwargs[attr].get(prop) is None:
						continue

					data = kwargs[attr].get(prop)
					scale = [tmp[-1].get('value') for tmp in search(kwargs.get(subattr),returns=True) if tmp is not None and tmp[-1] is not None]
					
					data = set_data(data=data,scale=scale)

					kwargs[attr][prop] = data

				props = '%serr'
				subprops ='%s'
				subattrs = 'set_%sscale'
				for axes in AXES[:dim]:
					prop = props%(axes)
					subprop = subprops%(axes)
					subattr = subattrs%(axes)

					err = kwargs[attr].get(prop)
					value = kwargs[attr].get(subprop)
					scale = [tmp[-1].get('value') for tmp in search(kwargs.get(subattr),returns=True) if tmp is not None and tmp[-1] is not None]

					err = set_err(err=err,value=value,scale=scale)

					kwargs[attr][prop] = err

				if ((kwargs[attr].get('y1') is not None) and (len(kwargs[attr].get('y1'))) and 
					(kwargs[attr].get('y2') is not None) and (len(kwargs[attr].get('y2')))):
					call = True
					x = np.array(kwargs[attr].get('x')) if kwargs[attr].get('x') is not None else kwargs[attr].get('x')
					y = np.array(kwargs[attr].get('y'))
					yerr = [y-kwargs[attr].get('y1'),kwargs[attr].get('y2')-y]
					args.extend([x,y-yerr[0],y+yerr[1]])
				elif ((kwargs[attr].get('yerr') is not None) and (len(kwargs[attr].get('yerr')))):
					call = False
					x = np.array(kwargs[attr].get('x')) if kwargs[attr].get('x') is not None else kwargs[attr].get('x')
					y = np.array(kwargs[attr].get('y'))
					yerr = [0]*2
					args.extend([x,y-yerr[0],y+yerr[1]])
				elif ((kwargs[attr].get('yerr') is not None) and (len(kwargs[attr].get('yerr'))) and
					  (kwargs[attr].get('y') is not None) and (len(kwargs[attr].get('y')))):
					call = True
					x = np.array(kwargs[attr].get('x')) if kwargs[attr].get('x') is not None else kwargs[attr].get('x')
					y = np.array(kwargs[attr].get('y'))
					yerr = np.array(kwargs[attr].get('yerr'))
					if ((yerr.ndim == 2) and (yerr.shape[0] == 2)):
						args.extend([x,y-yerr[0],y+yerr[1]])
					else:
						args.extend([kwargs[attr].get('x'),y-yerr,y+yerr])
				else:
					args.extend([])
					call = False
				nullkwargs.extend([*['%s%s'%(k,s) for s in VARIANTS for k in AXES],*[OTHER]])


			elif attr in ['scatter']:

				dim = 2

				props = '%s'
				subattrs = 'set_%sscale'
				for axes in AXES[:dim]:
					prop = props%(axes)
					subattr = subattrs%(axes)

					if kwargs[attr].get(prop) is None:
						continue

					data = kwargs[attr].get(prop)
					scale = [tmp[-1].get('value') for tmp in search(kwargs.get(subattr),returns=True) if tmp is not None and tmp[-1] is not None]
					
					data = set_data(data=data,scale=scale)

					kwargs[attr][prop] = data

				args.extend([kwargs[attr].get('%s%s'%(k,s)) for s in VARIANTS[:1] for k in AXES[:dim] if ((kwargs[attr].get('%s%s'%(k,s)) is not None))])

				replacements = {'color':'c','markersize':'s'}
				for replacement in replacements:
					if replacement in kwargs[attr]:
						kwargs[attr][replacements[replacement]] = kwargs[attr][replacement]

				nullkwargs.extend([*['%s%s'%(k,s) for s in VARIANTS[:2] for k in AXES],*[]])
				nullkwargs.extend([i for i in ['label', 'alpha', 'marker','markersize','linestyle','linewidth','elinewidth','capsize','color', 'ecolor']])

				call = True

			elif attr in ['plot_surface','contour','contourf','tricontour','tricontourf']:

				dim = 3
				args.extend([kwargs[attr].get('%s%s'%(k,s)) for s in VARIANTS[:1] for k in AXES[:dim] if ((kwargs[attr].get('%s%s'%(k,s)) is not None))])

				props = ['color']
				for prop in props:
					if prop not in kwargs[attr]:
						continue
					if isinstance(kwargs[attr][prop],dict):

						kwds = ['value','values','color','norm','scale','alpha']
					
						kwds = {kwd: kwargs[attr][prop].get(kwd,None) for kwd in kwds}

						if isinstance(kwds.get('value'),dict):
							kwds.update(kwds.pop('value',{}))

						value = 'values'
						values = 'value'
						if kwds.get(value) is None and kwds.get(values) is None:
							kwds[value] = [i/max(1,prod(shape[fields[prop]])-1) for i in range(prod(shape[fields[prop]]))]
							kwds[values] = [i/max(1,prod(shape[fields[prop]])-1) for i in range(prod(shape[fields[prop]]))]
						elif kwds.get(value) is None and kwds.get(values) is not None:
							kwds[value] = kwds[values]
						elif kwds.get(value) is not None and kwds.get(values) is None:
							kwds[values] = kwds[value]
						elif kwds.get(value) is not None and kwds.get(values) is not None:
							pass

						value,color,values,colors,norm = set_color(**kwds)

						kwargs[attr]['cmap'] = kwds['color']
						kwargs[attr]['norm'] = norm
						kwargs[attr]['vmin'] = norm.vmin
						kwargs[attr]['vmax'] = norm.vmax

				nullkwargs.extend([*['%s%s'%(k,s) for s in VARIANTS[:2] for k in AXES],*[]])
				nullkwargs.extend([i for i in ['label', 'alpha', 'marker','markersize','linestyle','linewidth','elinewidth','capsize','color', 'ecolor']])

				call = True


			elif attr in ['imshow','matshow']:
				dim = 2

				if any(prop in kwargs[attr] for prop in ['%s%s'%(k.upper(),s) for s in VARIANTS[:1] for k in AXES[:1]]):
					args.append(kwargs[attr].get(prop))
				elif all(prop in kwargs[attr] for prop in AXES[:dim+1][::-1]):
					for prop in [AXES[dim]]:
						shape = (*(len(set(kwargs[attr].get(prop))) for prop in AXES[:dim]),)
						args.append(kwargs[attr].get(prop).reshape(shape))

					props = {
						# 'extent':[value for prop in AXES[:dim] 
						# 	for value in [
						# 		min(kwargs[attr].get(prop),default=0),
						# 		max(kwargs[attr].get(prop),default=0)]
						# 	],
						'origin':'lower',
						'interpolation':'nearest',
						'aspect':1
						}
					for prop in props:
						if prop not in kwargs[attr]:
							kwargs[attr][prop] = props[prop]


					props = ['color']
					for prop in props:
						if prop not in kwargs[attr]:
							continue
						if isinstance(kwargs[attr][prop],dict):

							kwds = ['value','values','color','norm','scale','alpha']
						
							kwds = {kwd: kwargs[attr][prop].get(kwd,None) for kwd in kwds}

							if isinstance(kwds.get('value'),dict):
								kwds.update(kwds.pop('value',{}))
							
							value = 'values'
							values = 'value'
							if kwds.get(value) is None and kwds.get(values) is None:
								kwds[value] = [i/max(1,prod(shape[fields[prop]])-1) for i in range(prod(shape[fields[prop]]))]
								kwds[values] = [i/max(1,prod(shape[fields[prop]])-1) for i in range(prod(shape[fields[prop]]))]
							elif kwds.get(value) is None and kwds.get(values) is not None:
								kwds[value] = kwds[values]
							elif kwds.get(value) is not None and kwds.get(values) is None:
								kwds[values] = kwds[value]
							elif kwds.get(value) is not None and kwds.get(values) is not None:
								pass


							# value = 'norm'
							# values = 'values'
							# if kwds.get(value) is not None:
							# 	kwds[values] = None

							value = 'norm'
							values = 'value'
							if kwds.get(value) is not None:
								kwds[values] = None	

							value,color,values,colors,norm = set_color(**kwds)

							if colors is not None and len(args):
								args[0] = np.array([[*j] for i,j in zip(args[0].ravel(),colors)]).reshape((*args[0].shape,-1))
								kwargs[attr]['cmap'] = kwds['color']
								kwargs[attr]['norm'] = norm
								# kwargs[attr]['vmin'] = norm.vmin
								# kwargs[attr]['vmax'] = norm.vmax
							else:
								kwargs[attr]['cmap'] = kwds['color']
								kwargs[attr]['norm'] = norm
								kwargs[attr]['vmin'] = norm.vmin
								kwargs[attr]['vmax'] = norm.vmax


				nullkwargs.extend([*['%s%s'%(k.upper(),s) for s in VARIANTS[:2] for k in AXES[:]],*['%s%s'%(k,s) for s in VARIANTS[:2] for k in AXES],*[]])
				nullkwargs.extend(['value','color','ecolor','label','alpha','marker','markersize','linestyle','linewidth','elinewidth','capsize'])

				call = True


			elif attr in ['%saxis.set_%s_%s'%(axes,which,formatter) for axes in AXES for which in WHICH for formatter in FORMATTER]:
				
				class LogFormatterCustom(matplotlib.ticker.LogFormatterMathtext):
					def __call__(self, x, pos=None):
						if not (self._base**options['scilimits'][0] <= x <= self._base**options['scilimits'][-1]):
							return matplotlib.ticker.LogFormatterMathtext.__call__(self,x,pos=pos)
						else:
							return "${x:g}$".format(x=x)

				formatters = {"LogFormatterCustom":LogFormatterCustom}

				axes = attr.split('.')[0].replace('axis','')
				which = attr.split('.')[1].replace('set_','').replace('_%s'%(attr.split('_')[-1]),'')
				formatter = attr.split('_')[-1]
				for k in kwargs[attr]:
					for a in kwargs[attr][k]:
						options = {i:kwargs[attr][k][a].pop(i,default)
							for i,default in {'scilimits':[0,1]}.items()}
						if a in formatters:
							Formatter = formatters.get(a)
						elif hasattr(getattr(matplotlib,k),a):
							Formatter = getattr(getattr(matplotlib,k),a)
						else:
							continue
						getattr(getattr(obj,'%saxis'%(axes)),'set_%s_%s'%(which,formatter))(
							Formatter(**kwargs[attr][k][a]))
				call = False


			elif attr in ['set_%sscale'%(axes) for axes in AXES]:
				replacements = {
					'base':lambda axes,key,attr,kwargs:('%s%s'%(key,axes) if (kwargs[attr].get('value') not in ['linear']) else None),
					'base':lambda axes,key,attr,kwargs:('%s'%(key) if (kwargs[attr].get('value') not in ['linear']) else None),
					}
				for axes in AXES:
					if attr == 'set_%sscale'%(axes):
						for k in kwargs[attr]:
							if k in replacements:
								k,v = replacements[k](axes,k,attr,kwargs),kwargs[attr].pop(k)
								
								if k is not None:
									kwargs[attr][k] = v

								break

			elif attr in ['set_%sticks'%(axes) for axes in AXES]:
				pass

			elif attr in ['set_%sticks'%(axes) for axes in AXES]:
				pass				

			elif attr in ['set_%sbreak'%(axes) for axes in AXES]:

				props = ['transform']
				for prop in props:
					if prop in ['transform']:
						kwargs[attr][prop] = getattr(obj,kwargs[attr].get(prop))

				dim = 2
				args.extend([kwargs[attr].get('%s%s'%(k,s)) for s in VARIANTS[:1] for k in AXES[:dim] if ((kwargs[attr].get('%s%s'%(k,s)) is not None))])

				nullkwargs.extend([*[],*['%s%s'%(k,s) for s in VARIANTS[:2] for k in AXES],*['transform']])

				attr_ = 'plot'

				call = len(args)>0		

			elif attr in ['set_%snbins'%(axes) for axes in AXES]:
				axes = attr.replace('set_','').replace('nbins','')
				which = 'major'
				formatter = 'locator'
				prop = 'ticker'
				try:
					subprop = 'MaxNLocator'
					getattr(getattr(obj,'%saxis'%(axes)),'set_%s_%s'%(which,formatter))(
							getattr(getattr(matplotlib,prop),subprop)(**kwargs[attr]))

				except Exception as exception:
					a = 'LogLocator'
					getattr(getattr(obj,'%saxis'%(axes)),'set_%s_%s'%(which,formatter))(
							getattr(getattr(matplotlib,prop),subprop)(**kwargs[attr]))
				call = False

			# elif attr in ['%saxis.offsetText.set_fontsize'%(axes) for axes in AXES]:
			# 	axes = attr.split('.')[0].replace('axis','')
			# 	getattr(getattr(getattr(obj,'%saxis'%(axes)),'offsetText'),'set_fontsize')(**kwargs[attr])
			# 	call = False

			elif attr in ['set_colorbar']:

				nullkwargs.extend(['value','color','norm','scale','alpha',
					'segments','size','pad','padding','orientation','position','set_yscale','set_xscale','normed_values','share'])
				call = False

				kwds = ['value','values','color','norm','scale','alpha']
	
				subcall = not all(prop not in kwargs[attr] or kwargs[attr][prop] is None or kwargs[attr][prop] == [] for prop in ['value'])

				if subcall:
					kwds = {prop: kwargs[attr].get(prop,None) for prop in kwds}

					if isinstance(kwds.get('value'),dict):
						kwds.update(kwds.pop('value',{}))

					value = 'values'
					values = 'value'
					
					if kwds.get(value) is None and kwds.get(values) is None:
						kwds[value] = [i/max(1,prod(shape[fields[attr]])-1) for i in range(prod(shape[fields[attr]]))]
						kwds[values] = [i/max(1,prod(shape[fields[attr]])-1) for i in range(prod(shape[fields[attr]]))]
					elif kwds.get(value) is None and kwds.get(values) is not None:
						kwds[value] = kwds[values]
					elif kwds.get(value) is not None and kwds.get(values) is None:
						kwds[values] = kwds[value]
					elif kwds.get(value) is not None and kwds.get(values) is not None:
						pass

					value,color,values,colors,norm = set_color(**kwds)
					

					name = 'colorbar'				
					colors = list([list(i) for i in zip([i for i in values],[tuple(i) for i in colors])])
					N = len(colors)

					segments = kwargs[attr].get('segments',1)
					sizing = kwargs[attr].get('size','5%')
					padding = kwargs[attr].get('padding',kwargs[attr].get('pad',0.05))
					orientation = kwargs[attr].get('orientation','vertical')
					position = kwargs[attr].get('position','right')
					relative = sizing if isinstance(sizing,(int,np.integer,float,np.floating)) else float(sizing.replace('%',''))/100
					share = kwargs[attr].get('share')

					options = {option:kwargs[attr].get(option) for option in ['fraction','shrink','aspect','pad'] if option in kwargs[attr]}

					nullkwargs.extend(options)

					for axes in ['',*AXES]:
						prop = 'set_%slabel'%(axes)
						subprop = '%slabel'%(axes)
						if prop in kwargs[attr]:
							kwargs[attr][prop][subprop] = attr_texify(kwargs[attr][prop][subprop],prop,subprop)

					if N > 1:

						cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name=name,colors=colors,N=N*segments)

						if share:
							cax,options = matplotlib.colorbar.make_axes([ax for ax in obj.get_figure().axes],**options)
						else:
							pos = obj.get_position()
							divider = make_axes_locatable(obj)
							cax,options = divider.append_axes(position,size=sizing,pad=padding),dict()
						
							# pos = [pos.x0+padding, pos.y0, pos.width*relative, pos1.height] 
							# cax = plt.add_axes()
							# cax.set_position(pos)
						
						options = {**options,**dict(cmap=cmap,norm=norm,orientation=orientation,
							)}

						colorbar = matplotlib.colorbar.ColorbarBase(cax,**options)

						obj = colorbar	

						for kwarg in kwargs[attr]:
							
							if kwarg in nullkwargs:
								continue

							_obj = obj
							
							for _kwarg in kwarg.split('.'):
								try:
									_obj = getattr(_obj,_kwarg)
								except Exception as exception:
									break
							if isinstance(kwargs[attr][kwarg],dict):
								try:
									if _kwarg.startswith('get_'):
										for i in _obj():
											getattr(i,_kwarg)(kwargs[attr][kwarg])
									else:
										_obj(**kwargs[attr][kwarg])
								except Exception as exception:
									continue
							else:
								try:
									getattr(_obj,'set_%s'%(kwarg))(kwargs[attr][kwarg])
								except Exception as exception:
									continue									
				

			elif attr in ['savefig']:
				path = kwargs[attr].get('fname')
				if path is not None:
					dirname = os.path.abspath(os.path.dirname(path))
					if not os.path.exists(dirname):
						os.makedirs(dirname)		
					kwargs[attr]['fname'] = os.path.abspath(os.path.expanduser(path))
					call = True
				else:
					call = False

				call = call and (len(obj.get_figure().axes)>0)

			elif attr in ['close']:
				try:
					plt.close(obj,**kwargs[attr])
				except:
					plt.close(obj)
				call = False


			if any(((kwargs[attr].get(kwarg) is not None) and (not kwargs[attr][kwarg]))
				for kwarg in nullkwarg):

				call = False

			nullkwargs.extend(nullkwarg)

			try:
				_kwargs_ = deepcopy(kwargs[attr])
			except:
				_kwargs_ = copy(kwargs[attr])

			for kwarg in _kwargs_:
				if kwarg in ['linestyle']:
					if not isinstance(_kwargs_[kwarg],str):
						_kwargs_[kwarg] = tuple((i if isinstance(i,int) else tuple(i) for i in _kwargs_[kwarg]))

			for kwarg in nullkwargs:
				_kwargs_.pop(kwarg,None)


			if not call:	
		
				_obj = None

				objs.append(_obj)

				return

			# Set fields as per index
			# color,ecolor: index[-3],
			# marker index[-4]
			# linestyle index[-5]
			# linestyle index[-6]
			for field in fields:
				
				value = _kwargs_.get(field)

				if value is None:
					continue

				if value == '__cycle__':
					try:
						_obj = [i for i in objs if i is not None][-1]['obj'][-1]
					except:
						try:
							_obj = [i for i in objs if i is not None][-1]['obj']
						except:
							_obj = objs
					values = list_from_generator(getattr(getattr(obj,'_get_lines'),'prop_cycler'),field)
					_kwargs_[field] = values[-1]
				
				elif value == '__lines__':
					_obj = getattr(obj,'get_lines')()[-1]
					_kwargs_[field] = getattr(_obj,'get_%s'%(field))()
			
				elif isinstance(value,(dict,str)):

					if field in ['color','ecolor','c']:

						kwds = ['value','values','color','norm','scale','alpha']

						kwds = {prop: None	for prop in kwds}

						if isinstance(value,dict):
							kwds.update(value)
						elif isinstance(value,str):
							kwds.update({'color':value})

						if isinstance(kwds.get('value'),dict):
							kwds.update(kwds.pop('value',{}))

						value = 'values'
						values = 'value'
						if kwds.get(value) is None and kwds.get(values) is None:
							kwds[value] = to_index(index[fields[field]],shape[fields[field]])/max(1,prod(shape[fields[field]])-1)
							kwds[values] = [i/max(1,prod(shape[fields[field]])-1) for i in range(prod(shape[fields[field]]))]
						elif kwds.get(value) is None and kwds.get(values) is not None:
							kwds[value] = kwds[values]
						elif kwds.get(value) is not None and kwds.get(values) is None:
							kwds[values] = kwds[value]
						elif kwds.get(value) is not None and kwds.get(values) is not None:
							pass
						value,color,values,colors,norm = set_color(**kwds)

						value = color

					replacements = {'c':'color'}
					if field in replacements:
						if not isinstance(value,scalars) or isinstance(value,tuple) or len(value) == 1:
							_kwargs_.pop(field);
							field = replacements[field]

					_kwargs_[field] = value
				
				else:
					continue


			_obj = obj
			for a in attr_.split('.'):
				try:
					_obj = getattr(_obj,a)
				except:
					break			


			# try:
			if args != []:
				_attr = _obj(*args,**_kwargs_)
			else:
				try:
					_attr = _obj(**_kwargs_)
				except:
					try:
						_kwargs_ = {_kwarg_:_kwargs_[_kwarg_] for _kwarg_ in _kwargs_ if _kwargs_[_kwarg_] is not None}
						if _kwargs_:
							_attr = _obj(**_kwargs_)
						else:
							_attr = None
					except Exception as exception:
						logger.log(debug,'%r'%(exception))
						_attr = None
						# exit()
			# except Exception as exception:
			# 	_attr = None
			# 	if not isinstance(exception,AttributeError):
			# 		logger.log(debug,'%r %r %s %r %r'%(exception,_obj,attr,args,_kwargs_))
			# 		logger.log(debug,'%r'%(traceback.format_exc()))
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
								try:
									for j,_subattr_ in enumerate(list(get_children(_attr,a))):
										for l in _kwds[k]:
											getattr(_subattr_,l)(_kwds[k][l][j%len(_kwds[k][l])])
								except Exception as exception:
									try:
										for j,_subattr_ in enumerate(getattr(_attr_,a)):
											for i,l in enumerate(_kwds[k]):
												if _kwds[k][l] is not None:
													v = getattr(_subattr_,'get_%s'%(l.replace('set_','')))()
													v = _kwds[k][l][j%len(_kwds[k][l])]
													getattr(_subattr_,'set_%s'%(l.replace('set_','')))(v)
												else:
													pass
										else:
											pass
									except Exception as exception:
										pass	
									pass							
				

			# except:
			# 	_kwargs = inspect.getfullargspec(getattr(obj,attr))[0]
			# 	args.extend([_kwargs_[k] for k in _kwargs_ if k not in _kwargs])
			# 	_kwargs_ = {k:_kwargs_[k] for k in _kwargs_ if k in _kwargs}
			# 	try:
			# 		getattr(obj,attr)(*args,**_kwargs_)
			# 	except:
			# 		pass
			_obj = {'obj':_attr,'attr':attr,'index':index,'indices':indices,'shape':shape,'count':count}
			
			objs.append(_obj)

			return

		def attr_kwargs(kwarg,attr,kwargs,settings,index):
			updates = {}
			if attr in kwargs.get('share',{}):
				updates.update({
						'legend': {'handles':True,'labels':True}
						}.get(attr,{}))
			kwarg[attr].update(updates)
			kwargs = {
				**kwarg,
				attr: {
					**{k: attr_share(attr_texify(kwarg[attr][k],attr,k,**kwargs),attr,k,**kwargs) for k in kwarg[attr]},
					},
				(attr,*index):settings[attr],
				}
			return kwargs 

		# Flatten nested settings
		if not isinstance(settings[attr],(dict,list)):
			return

		finds = [(index,[1] if not shape else shape,{**settings,attr:setting} if setting else None) for index,shape,setting in search(settings[attr],types=(list,),returns=True)]
		indices = [index for index,shape,kwarg in finds]

		for count,(index,shape,kwarg) in enumerate(finds):
			
			if not kwarg:
				continue

			attrs(obj,attr,objs,index,indices,shape,count,kwargs,attr_kwargs(kwarg,attr,kwargs,settings,index))

		return

	def obj_wrap(attr,key,fig,ax,settings):

		defaults = {
			'texify':None,
			'share': {},
			'layout':{
				**{attrs:[settings[k]['style'].get('layout',{}).get(attr) 
					for k in settings 
					if (attr in settings[k]['style'].get('layout',{}) and 
						any((data is not None and any(data.get(i) is not None and len(data.get(i))  for i in ALL))
						for prop in PLOTS if prop in settings[k][OBJ] 
						for data in search(settings[k][OBJ][prop])))] 
					for attr,attrs in {'index':'indices'}.items()},
				}
			}

		def attr_kwargs(attr,key,settings):
			kwargs = {
				'texify':settings[key]['style'].get('texify',defaults['texify']),
				'share':settings[key]['style'].get('share',{}).get(attr,defaults['share']),
				'layout':_layout({
					**{attr:settings[key]['style'].get('layout',{}).get(attr) for attr in LAYOUT},
					**defaults['layout']
					}),
				}
			return kwargs

		matplotlib.rcParams.update(settings[key]['style'].get('rcParams',{}))

		obj = lambda attr,key,fig,ax: {'fig':fig.get(key),'ax':ax.get(key),**{'%s_%s'%('ax',k):ax.get('%s_%s'%(key,k)) for k in CAXES}}[attr]
		obj = obj(attr,key,fig,ax)

		objs = []

		try:
			axes = [ax for ax in obj.get_figure().axes]
		except:
			axes = None

		if obj is not None:
			
			props = list(settings[key][attr])

			ordering = {'close':-1,'savefig':-2}
			for prop in ordering:
				if prop in settings[key][attr]:
					if ordering[prop] == -1:
						ordering[prop] = len(props)
					elif ordering[prop] < -1:
						ordering[prop] += 1
					props.insert(ordering[prop],props.pop(props.index(prop)))

			modify = {'savefig':-1}
			for prop in modify:
				if prop in settings[key][attr]:
					if modify[prop] == -1:
						modify[prop] = len(settings)-1
					if list(settings).index(key) != modify[prop]:
						props.pop(props.index(prop))

			for prop in props:

				kwargs = attr_kwargs(attr,key,settings)

				attr_wrap(obj,prop,objs,settings[key][attr],**kwargs)

		try:
			axes = [ax for ax in obj.get_figure().axes if ax not in axes]
			if not any(obj['attr'] in PLOTS for obj in objs if obj is not None):
				for ax in [*axes,obj]:
					obj.get_figure().delaxes(ax)
		except:
			pass

		return
		
		
	def context(x,y,z,settings,fig,ax,mplstyle,texify):
		with matplotlib.style.context(mplstyle):
			settings,fig,ax = setup(x,y,z,settings,fig,ax,mplstyle,texify)
			for key in settings:
				for attr in ['ax',*['%s_%s'%('ax',k) for k in CAXES],'fig']:
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
			if not all([kwarg in settings[key]['style']['layout'] for kwarg in LAYOUT[:LAYOUTDIM+1]]):
				settings[key]['style']['layout'].update(dict(zip([*LAYOUT[:LAYOUTDIM],LAYOUT[LAYOUTDIM]],_index(i,len(y),'row'))))
		
		for key in y:

			_settings = load(PATHS['plot'])
			setter(_settings,settings[key],default=True)

			_settings['style'].update({
				'layout':{kwarg:settings[key]['style'].get('layout',{}).get(kwarg,_defaults['style']['layout'][kwarg])
							if settings[key]['style'].get('layout',{}).get(kwarg) is None else settings[key]['style'].get('layout',{}).get(kwarg) 
							for kwarg in LAYOUT}
				})
			if update:
				plotsettings = settings[key].get('ax',{}).pop('plot',{})				
				_settings[OBJ].update({
					**{'plot':[{'x':_x,'y':_y,'z':_z,**(plotsettings if isinstance(plotsettings,dict) else plotsettings[_i])} 
								for _i,(_x,_y,_z) in enumerate(zip(x.get(key,[None]*len(y[key])),y[key],z[key]))]
								},
					**settings[key].get(OBJ,{}), 
					})

			for attr in settings[key]:
				if attr in _settings:
					_settings[attr].update(settings[key][attr])

			setter(settings[key],_settings,default=True)


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

	attr = 'style'
	prop = 'use'
	default = 'pdf'
	uses = [*[settings[key].get(attr,{}).get(prop) for key in settings],
				settings.get(attr,{}).get(prop),
				default]
	uses = [use for use in uses if use is not None]
	use = uses[0] if uses else default

	options = [settings,PATHS['plot'],{}]
	for settings in options:
		if ((settings is not None) or (isinstance(settings,str) and os.path.isfile(settings))):
			break

	try:
		matplotlib.use(use)		
		fig,ax = context(x,y,z,settings,fig,ax,mplstyle,texify)
	except Exception as exception:
		rc_params = {'text.usetex': False}
		matplotlib.rcParams.update(rc_params)
		matplotlib.use(use) 
		fig,ax = context(x,y,z,settings,fig,ax,_mplstyle,texify)

	return fig,ax



if __name__ == '__main__':

	data = sys.argv[1]
	path = sys.argv[2]
	settings = sys.argv[3]
	mplstyle = sys.argv[4]
	Y = sys.argv[5].split(' ')
	X = sys.argv[6].split(' ')
	Z = sys.argv[7].split(' ')



	df = pd.concat([pd.read_csv(d) for d in glob.glob(data)],
					axes=0,ignore_index=True)

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
