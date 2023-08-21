 #!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,re
from copy import deepcopy as copy
import traceback

import numpy as np

warnings.simplefilter("ignore", (UserWarning,DeprecationWarning,FutureWarning))

scalars = (int,np.integer,float,np.floating,str,type(None))


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
		return getattr(obj,attr,default)

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
	replacements = {'\\':'\\\\','.':'\\.','*':'.*',}
	for replacement in replacements:
		pattern = pattern.replace(replacement,replacements[replacement])
		
	boolean = re.fullmatch(pattern,string) is not None

	return boolean


def copier(key,value,copy):
	'''
	Copy value based on associated key 

	Args:
		key (string): key associated with value to be copied
		value (object): data to be copied
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
	Returns:
		Copy of value
	'''

	# Check if copy is a dictionary and key is in copy and is True to copy value
	if ((not copy) or (isinstance(copy,dict) and (not copy.get(key)))):
		return value
	else:
		return copy(value)



def clone(iterable,twin,copy=False):
	'''
	Shallow in-place copy of iterable to twin

	Args:
		iterable (dict): dictionary to be copied
		twin (dict): dictionary to be modified in-place with copy of iterable
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
	'''	

	# Iterate through iterable and copy values in-place to twin dictionary
	for key in iterable:
		if isinstance(iterable[key],dict):
			if twin.get(key) is None:
				twin[key] = {}
			clone(iterable[key],twin[key],copy)
		else:
			twin[key] = copier(key,iterable[key],copy)
	return


def setter(iterable,elements,delimiter=False,copy=False,reset=False,clear=False,func=None):
	'''
	Set nested value in iterable with nested elements keys
	Args:
		iterable (dict): dictionary to be set in-place with value
		elements (dict): Dictionary of keys of delimiter separated strings, or tuple of string for nested keys, and values to set 
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		reset (bool): boolean on whether to replace value at key with value, or update the nested dictionary
		clear (bool): boolean of whether to clear iterable when the element's value is an empty dictionary
		func(callable,None,bool,iterable): Callable function with signature func(key_iterable,key_elements,iterable,elements) to modify value to be updated based on the given dictionaries, or True or False to default to elements or iterable values, or iterable of allowed types
	'''

	if (not isinstance(iterable,(dict,list))) or (not isinstance(elements,dict)):
		return

	# Setup func as callable
	if func is None:
		function = lambda key_iterable,key_elements,iterable,elements: elements.get(key_elements)
	elif func is True:
		function = lambda key_iterable,key_elements,iterable,elements: elements.get(key_elements)
	elif func is False:
		function = lambda key_iterable,key_elements,iterable,elements: iterable.get(key_iterable,elements.get(key_elements))
	elif func in ['none','None']:
		function = lambda key_iterable,key_elements,iterable,elements: elements.get(key_elements) if iterable.get(key_iterable,elements.get(key_elements)) is None else iterable.get(key_iterable,elements.get(key_elements))
	elif not callable(func):
		types = tuple(func)
		def function(key_iterable,key_elements,iterable,elements,types=types): 
			i = iterable.get(key_iterable,elements.get(key_elements))
			e = elements.get(key_elements,i)
			return e if isinstance(e,types) else i
	else:
		function = func

	# Clear iterable if clear and elements is empty dictionary
	if clear and elements == {}:
		iterable.clear()

	# Set nested elements
	for element in list(elements):

		# Get iterable, and index of tuple of nested element key
		i = iterable
		index = 0

		# Convert string instance of elements to list, splitting string based on delimiter delimiter
		try:
			if (
				(isinstance(element,str) and delimiter) and 
				(element not in iterable)):
				#((element.count(delimiter)>0) and ((element not in iterable)) or (element.split(delimiter)[0] in iterable))):
				# e = element.split(delimiter)

				e = []
				_element = element.split(delimiter)
				_iterable = iterable
				while _element and isinstance(_iterable,dict):
					for l in range(len(_element),-1,-1):
						_e = delimiter.join(_element[:l])

						if _e in _iterable:
							_iterable = _iterable.get(_e)
							e.append(_e)
							_element = _element[l:]
							break
					if l == 0:
						e.extend(_element)
						break
				e = tuple(e)

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

			# try:
			value = copier(element,function(e[index],element,i,elements),copy)

			if isinstance(i,list) and (e[index] >= len(i)):
				i.extend([{} for j in range(e[index]-len(i)+1)])

			if reset:
				i[e[index]] = value
			elif e[index] not in i or not isinstance(i[e[index]],(dict,list)):
				i[e[index]] = value
			elif isinstance(elements[element],dict):
				setter(i[e[index]],elements[element],delimiter=delimiter,copy=copy,reset=reset,clear=clear,func=func)
			else:
				i[e[index]] = value
		except Exception as exception:
			print(traceback.format_exc())
			pass

	return


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
	if isinstance(elements,str):
		if delimiter and (elements not in iterable):
			elements = elements.split(delimiter)
		else:
			elements = [elements]

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

def popper(iterable,elements,default=None,delimiter=False,copy=False):
	'''
	Pop nested value in iterable with nested elements keys

	Args:
		iterable (dict): dictionary to be popped in-place
		elements (str,iterable[str]): delimiter separated string or list to nested keys of location to pop value
		default (object): default data to return if elements not in nested iterable
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value

	Returns:
		Value at nested keys elements of iterable
	'''		
	
	i = iterable
	e = 0

	# Convert string instance of elements to list, splitting string based on delimiter delimiter	
	if isinstance(elements,str) and delimiter:
		elements = elements.split(delimiter)

	if not isinstance(elements,list):
		# elements is object and value is to be got from iterable at first level of nesting		
		try:
			return i.pop(elements)
		except:
			return default
	else:
		# elements is list of nested keys and the nested values are to be extracted from iterable		
		try:
			while e<(len(elements)-1):
				i = i[elements[e]]
				e+=1			
		except:
			return default

	return copier(e,i.pop(elements[e],default),copy)

def hasser(iterable,elements,delimiter=False):
	'''
	Check if nested iterable has nested elements keys

	Args:
		iterable (dict): dictionary to be searched
		elements (str,iterable[str]): delimiter separated string or list to nested keys of location to set value
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys

	Returns:
		Boolean value if nested keys elements are in iterable
	'''		

	i = iterable
	e = 0

	# Convert string instance of elements to list, splitting string based on delimiter delimiter	
	if isinstance(elements,str) and delimiter:
		elements = elements.split(delimiter)
	try:
		if not isinstance(elements,list):
			# elements is object and value is to be got from iterable at first level of nesting				
			i = i[element]
		else:
			# elements is list of nested keys and the nested values are to be extracted from iterable		
			while e<len(elements):
				i = i[elements[e]]
				e+=1			
		return True
	except:
		return False



finder = lambda key,value,delimiter=None: (
	finder(delimiter.join(key.split(delimiter)[1:]),value.get(key.split(delimiter)[0])) if (
		(isinstance(delimiter,str)) and (len(key)) and (
		(isinstance(value,dict) and (key.split(delimiter)[0] in value))))
		else value)


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
			but should be combined in sequence element wise. 
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
		groups = copy(groups)
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
		return [{k:copier(k,u,copy) for k,u in zip(keys,v)} for v in zip(*values)]

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
	Search of iterable, returning elements and indices of elements
	Args:
		iterable (iterable): Nested iterable
		index (iterable[int,str]): Index of element
		shape (iterable[int]): Shape of iterable
		returns (bool,str): Returns of search, 
			None returns item, True returns index,shape,item, False returns None, 
			allowed strings (.delimited) for combinations of ['index','shape','item']
		types (type,tuple[type]): Allowed types to be searched
		exceptions (type,tuple[type]): Disallowed types to be searched
	Yields:
		index (iterable[int,str]): Index of item
		shape (iterable[iterable[int]]): Shape of iterable at index
		item (iterable): Iterable element
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
	for index,shape,element in search(iterable,returns=True,types=types,exceptions=exceptions):
		if element == item:
			return index
	return None

def indexer(index,iterable,types=(list,),exceptions=()):
	'''
	Get item at index in iterable
	Args:
		index (iterable[int,str]): Index of element
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