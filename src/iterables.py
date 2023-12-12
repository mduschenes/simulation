 #!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,re
from copy import deepcopy
import traceback

import numpy as np

warnings.simplefilter("ignore", (UserWarning,DeprecationWarning,FutureWarning))


class Null(object):
	def __str__(self):
		return 'Null'
	def __repr__(self):
		return self.__str__()

class none(object):
	def __init__(self,default=0,*args,**kwargs):
		self.default = default
		return
	def __call__(self,*args,**kwargs):
		return self.default

null = Null()

scalars = (int,np.integer,float,np.floating,str,type(None))
nulls = (Null,)

def namespace(cls,signature=None,init=False,**kwargs):
	'''
	Get namespace of attributes of class instance
	Args:
		cls (class): Class to get attributes
		signature (dict): Dictionary to get only attributes in cls
		init (bool): Initialize class for all attributes
		kwargs (dict): Additional keyword arguments for cls
	Returns:
		attrs (iterable,dict): Attributes of cls
	'''
	
	if init:
		attrs = dir(cls(**kwargs))
	else:
		try:
			attrs = cls.__dict__
		except:
			attrs = None

	if signature is None:
		return attrs
	elif attrs is not None:
		return {attr: signature[attr] for attr in signature if attr in attrs}
	else:
		return attrs


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

def getattrs(obj,attr,default=None,delimiter=None):
	'''
	Get nested attribute of object
	Args:
		obj (object): Object of attributes
		attr (str): Nested attribute of object
		default (object): Default nested attribute of object
		delimiter (str): Delimiter to split nested attributes
	Returns:
		obj (object): Nested attribute
	'''

	if delimiter is None:
		return getattr(obj,attr,default)

	attrs = attr.split(delimiter)
	
	n = len(attrs)

	for i in range(n):
		attr = delimiter.join(attrs[:i+1])
		
		if hasattr(obj,attr):
			if i == (n-1):
				return getattr(obj,attr)
			else:
				attribute = delimiter.join(attrs[i+1:])
				return getattrs(getattr(obj,attr),attribute,default=default,delimiter=delimiter)
	
	return default

def setattrs(obj,attr,value,delimiter=None):
	'''
	Set nested attribute of object
	Args:
		obj (object): Object of attributes
		attr (str): Nested attribute of object
		value (object): Nested value of object
		delimiter (str): Delimiter to split nested attributes
	'''

	raise NotImplementedError


	if delimiter is None:
		return setattr(obj,attr,value)

	attrs = attr.split(delimiter)
	
	n = len(attrs)

	for i in range(n):
		attr = delimiter.join(attrs[:i+1])
		
		if hasattr(obj,attr):
			if i == (n-1):
				setattr(obj,attr,value)
				return True
			else:
				attribute = delimiter.join(attrs[i+1:])
				setattrs(getattr(obj,attr),attribute,value=value,delimiter=delimiter)
				return

	attr = attrs[-1]
	setattr(obj,attr,value)
		
	return


def hasattrs(obj,attr,default=None,delimiter=None):
	'''
	Check existence of nested attribute of object
	Args:
		obj (object): Object of attributes
		attr (str): Nested attribute of object
		default (object): Default nested attribute of object
		delimiter (str): Delimiter to split nested attributes
	Returns:
		has (bool): Nested attribute existence
	'''

	if delimiter is None:
		return hasattr(obj,attr)

	attrs = attr.split(delimiter)
	
	n = len(attrs)
	
	for i in range(n):
		attr = delimiter.join(attrs[:i+1])
		if hasattr(obj,attr):
			if i == (n-1):
				return True
			else:
				attribute = delimiter.join(attrs[i+1:])
				return hasattrs(getattr(obj,attr),attribute,default=default,delimiter=delimiter)
	
	return False

def contains(string,pattern):
	'''
	Search for pattern in string
	Args:
		string (str): String to search
		pattern (str): Pattern to search
	Returns:
		boolean (bool): String contains pattern
	'''

	string = str(string)
	pattern = str(pattern)

	replacements = {'\\':'\\\\','.':'\\.','*':'.*',}
	for replacement in replacements:
		pattern = pattern.replace(replacement,replacements[replacement])
		
	boolean = re.fullmatch(pattern,string) is not None

	return boolean


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

def setter(iterable,keys,delimiter=None,default=None,copy=False):
	'''
	Set nested value in iterable with nested keys
	Args:
		iterable (dict): dictionary to be set in-place with value
		keys (dict,tuple): Dictionary of keys of delimiter separated strings, or tuple of string for nested keys, and values to set 
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string keys into list of nested keys
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		reset (bool): boolean on whether to replace value at key with value, or update the nested dictionary
		default(callable,None,bool,iterable): Callable function with signature default(key_iterable,key_keys,iterable,keys) to modify value to be updated based on the given dictionaries, or True or False to default to keys or iterable values, or iterable of allowed types
	'''

	types = (dict,)

	if (not isinstance(iterable,types)) or (not isinstance(keys,types)):
		return

	# Setup default func as callable
	if default is None:
		func = lambda key_iterable,key_keys,iterable,keys: keys.get(key_keys)
	elif default is True:
		func = lambda key_iterable,key_keys,iterable,keys: keys.get(key_keys)
	elif default is False:
		func = lambda key_iterable,key_keys,iterable,keys: iterable.get(key_iterable,keys.get(key_keys))
	elif default in ['none','None']:
		func = lambda key_iterable,key_keys,iterable,keys: keys.get(key_keys) if iterable.get(key_iterable,keys.get(key_keys)) is None else iterable.get(key_iterable,keys.get(key_keys))
	elif not callable(default):
		instances = tuple(default)
		def func(key_iterable,key_keys,iterable,keys,instances=instances): 
			i = iterable.get(key_iterable,keys.get(key_keys))
			e = keys.get(key_keys,i)
			return e if isinstance(e,instances) else i
	else:
		func = default


	for key in keys:

		if (isinstance(key,str) and (delimiter is not None) and (key not in iterable)):
			index = key.split(delimiter)
		elif (key in iterable):
			index = (key,)
		elif isinstance(key,scalars):
			index = (key,)
		else:
			index = (*key,)

		if len(index)>1 and (delimiter is not None):
			index,other = index[0],delimiter.join(index[1:])
		else:
			index,other = index[0],null

		if index in iterable:
			if not isinstance(other,nulls):
				setter(iterable[index],{other:keys[key]},delimiter=delimiter,default=default,copy=copy)
			else:
				if isinstance(keys[key],types) and isinstance(iterable[index],types):
					setter(iterable[index],keys[key],delimiter=delimiter,default=default,copy=copy)
				else:
					iterable[index] = copier(func(index,key,iterable,keys),copy=copy)

		else:
			if not isinstance(other,nulls):
				iterable[index] = {}
				setter(iterable[index],{other:keys[key]},delimiter=delimiter,default=default,copy=copy)
			else:
				iterable[index] = copier(func(index,key,iterable,keys),copy=copy)

	return




def getter(iterable,keys,delimiter=None,default=None,copy=False):
	'''
	Get nested value in iterable with nested keys
	Args:
		iterable (dict): dictionary to get with keys
		keys (str,dict,tuple,list): Dictionary of keys of delimiter separated strings, or tuple of string for nested keys
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string keys into list of nested keys
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		reset (bool): boolean on whether to replace value at key with value, or update the nested dictionary
		default(callable,None,bool,iterable): Callable function with signature default(key_iterable,key_keys,iterable,keys) to modify value to be updated based on the given dictionaries, or True or False to default to keys or iterable values, or iterable of allowed types
	'''

	types = (dict,)

	if (not isinstance(iterable,types)) or (not isinstance(keys,(str,tuple,list))):
		return copier(iterable,copy=copy)

	key = keys

	if (isinstance(key,str) and (delimiter is not None) and (key not in iterable)):
		index = key.split(delimiter)
	elif isinstance(key,scalars):
		index = (key,)
	else:
		index = (*key,)

	if len(index)>1 and (delimiter is not None):
		index,other = index[0],delimiter.join(index[1:])
	else:
		index,other = index[0],null

	if index in iterable:
		if not isinstance(other,nulls):
			return getter(iterable[index],other,delimiter=delimiter,default=default,copy=copy)
		else:
			return copier(iterable[index],copy=copy)
	else:
		return copier(iterable,copy=copy)


def permutations(*iterables,repeat=None):
	'''
	Get product of permutations of iterables
	Args:
		iterables (iterable[iterables],iterable[int]): Iterables to permute, or iterable of int to get all permutations of range(int)
	Returns:
		iterables (generator[tuple]): Generator of tuples of all permutations of iterables
	'''
	
	if all(isinstance(i,int) for i in iterables):
		iterables = (range(i) for i in iterables)
	
	if repeat is None:
		repeat = 1

	return itertools.product(*iterables,repeat=repeat)

def permuter(dictionary,copy=False,groups=None,ordered=True):
	'''
	Get all combinations of values of dictionary of lists

	Args:
		dictionary (dict): dictionary of keys with lists of values to be combined in all combinations across lists
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		groups (list,None): List of lists of groups of keys that should not have their values permuted in all combinations, 
			but should be combined in sequence key wise. 
			For example groups = [[key0,key1]], where 
			dictionary[key0] = [value_00,value_01,value_02],
			dictionary[key1] = [value_10,value_11,value_12], 
			then the permuted dictionary will have key0 and key1 keys with only pairwise values of 
			[{key0:value_00,key1:value_10},{key0:value_01,key1:value_11},{key0:value_02,key1:value_12}].
		ordered (bool): Boolean on whether to return dictionaries with same ordering of keys as dictionary

	Returns:
		List of dictionaries with all combinations of lists of values in dictionary
	'''		
	def indexer(keys,values,groups):
		'''
		Get lists of values for each group of keys in groups
		'''
		groups = deepcopy(groups)
		if groups is not None:
			inds = [[keys.index(k) for k in g if k in keys] for g in groups]
		else:
			inds = []
			groups = []
		N = len(groups)
		groups.extend([[k] for k in keys if all([k not in g for g in groups])])
		inds.extend([[keys.index(k) for k in g if k in keys] for g in groups[N:]])
		values = [[values[j] for j in i ] for i in inds]
		return groups,values

	def zipper(keys,values,copy): 
		'''
		Get list of dictionaries with keys, based on list of lists in values, retaining ordering in case of grouped values
		'''
		return [{k:copier(u,copy=copy) for k,u in zip(keys,v)} for v in zip(*values)]

	def unzipper(dictionary):
		'''
		Zip keys of dictionary of list, and values of dictionary as list
		'''
		keys, values = zip(*dictionary.items())	
		return keys,values

	def permute(dictionaries): 
		'''
		Get all list of dictionaries of all permutations of sub-dictionaries
		'''
		return [{k:d[k] for d in dicts for k in d} for dicts in permutations(*dictionaries)]

	def retriever(keys,values):
		'''
		Get values of permuted nested dictionaries in values.
		Recurse permute until values are lists and not dictionaries.
		'''
		keys,values = list(keys),list(values)
		for i,(key,value) in enumerate(zip(keys,values)):
			if isinstance(value,dict):
				if isinstance(groups,dict):
					group = groups.get(key,group)
				else:
					group = groups
				values[i] = permuter(value,copy=copy,groups=group) 
		return keys,values


	if dictionary in [None,{}]:
		return [{}]

	# Get list of all keys from dictionary, and list of lists of values for each key
	keys,values = unzipper(dictionary)


	# Get values of permuted nested dictionaries in values
	keys,values = retriever(keys,values)

	# Retain ordering of keys in dictionary
	keys_ordered = keys
	
	# Get groups of keys based on groups and get lists of values for each group
	keys,values = indexer(keys,values,groups)

	# Zip keys with lists of lists in values into list of dictionaries
	dictionaries = [zipper(k,v,copy) for k,v in zip(keys,values)]


	# Get all permutations of list of dictionaries into one list of dictionaries with all keys
	dictionaries = permute(dictionaries)


	# Retain original ordering of keys if ordered is True
	if ordered:
		for i,d in enumerate(dictionaries):
			dictionaries[i] = {k: dictionaries[i][k] for k in keys_ordered}
	return dictionaries


def equalizer(a,b,types=(dict,),exceptions=None):
	'''
	Check if nested iterables have equal keys and values
	Args:
		a (dict): Iterable to check
		b (dict): Iterable to check
		types (tuple[type]): Allowed nested types of iterable values		
		exceptions (callable): Check for exceptions of value equality, returning True for exception, with signature exceptions(a,b)
	Raises:
		Assertion Error if iterables are not equal for given key and value
	'''
	if exceptions is None:
		exceptions = lambda a,b: False

	if (not isinstance(a,types)) and (not isinstance(b,types)): 
		assert exceptions(a,b) or (a == b),"%r (%r) != %r (%r)"%(a,b,type(a),type(b)) 
		return
	elif exceptions(a,b):
		return

	assert isinstance(a,types) and isinstance(b,types), "iterables %r,%r, are not of type %r"%(type(a),type(b),types,)
	assert len(a) == len(b), "iterables are not equal lengths\n%r\n%r"%(a,b)

	for i,item in enumerate(a):
		if not isinstance(item,types):
			assert item in b, "%r not in both iterables"%(item)	
		if isinstance(a,dict) and isinstance(b,dict):
			key = item
		else:
			key = i
		equalizer(a[key],b[key],types=types,exceptions=exceptions)

	return


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

def find(item,iterable,types=(list,),exceptions=()):
	'''
	Search for index of item in iterable
	Args:
		item (object): Item to search
		iterable (iterable): Nested iterable
		types (type,tuple[type]): Allowed types to be searched
		exceptions (type,tuple[type]): Disallowed types to be searched
	Returns:
		index (iterable[int,str]): Index of item
	'''	
	for index,shape,key in search(iterable,returns=True,types=types,exceptions=exceptions):
		if key == item:
			return index
	return None

def indexer(index,iterable,types=(list,),exceptions=()):
	'''
	Get item at index in iterable
	Args:
		index (iterable[int,str]): Index of key
		iterable (iterable): Nested iterable
		types (type,tuple[type]): Allowed types to be searched
		exceptions (type,tuple[type]): Disallowed types to be searched
	Returns:
		item (object): Item at index
	'''	
	item = iterable
	for i in index:
		item = item[i]

	return item


def inserter(index,item,iterable,types=(list,),exceptions=()):
	'''
	Insert item at index into iterable
	Args:
		index (iterable): Index of item
		item (object): Item to search
		iterable (iterable): Nested iterable
		types (type,tuple[type]): Allowed types to be searched
		exceptions (type,tuple[type]): Disallowed types to be searched
	'''
	dictionaries = (dict,)

	if isinstance(index,scalars):
		index = [index]

	for j,i in enumerate(index):
		default = None if (j==(len(index)-1)) else {} if isinstance(index[j+1],str) else []
		if isinstance(iterable,dictionaries) and i not in iterable:
			iterable[i] = default
		elif not isinstance(iterable,dictionaries) and isinstance(i,int) and (len(iterable) <= i):
			iterable.extend((default for j in range(i+1-len(iterable))))
		
		if j < (len(index)-1):
			iterable = iterable[i]
		else:
			iterable[i] = item

	return


def iterate(obj,attr,attributes=[],delimiter=None):
	'''
	Get nested attributes of object
	Args:
		obj (object): Object of object
		attr (str): Nested attribute of object
		attributes (iterable[str]): Nested attributes of object
		delimiter (str): Delimiter to split nested attributes
	Yields:
		attr (str): Nested attributes
	'''

	delimiter = delimiter if delimiter is not None else '.'

	attrs = attr.split(delimiter)

	n = len(attrs)

	exists = False

	if attrs and namespace(obj) is not None:

		for i in range(n):
			attr = delimiter.join(attrs[:i+1])
			for attribute in namespace(obj):
				if contains(attribute,attr):
					if attrs[i+1:]:
						yield from iterate(getattr(obj,attribute),delimiter.join(attrs[i+1:]),attributes=[*attributes,attribute],delimiter=delimiter)
					else:
						yield delimiter.join([*attributes,attribute])

					exists = True
			
			if exists:
				break