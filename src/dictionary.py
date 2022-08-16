 #!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools
import copy as copying

warnings.simplefilter("ignore", (UserWarning,DeprecationWarning,FutureWarning))


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
		return copying.deepcopy(value)



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




def setter(iterable,elements,delimiter=False,copy=False,reset=True):
	'''
	Set nested value in iterable with nested elements keys

	Args:
		iterable (dict): dictionary to be set in-place with value
		elements (dict): Dictionary of keys of delimiter separated strings, or tuple of string for nested keys, and values to set 
		delimiter (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		reset (bool): boolean on whether to replace value at key with value, or update the nested dictionary
	'''

	assert isinstance(iterable,dict), "Error - iterable is not dictionary"
	assert isinstance(elements,dict), "Error - elements is not dictionary"

	for element in elements:
	
		# Get copy of value in elements
		value = copier(element,elements[element],copy)


		# Get iterable, and index of tuple of nested element key
		i = iterable
		e = 0

		# Convert string instance of elements to list, splitting string based on delimiter delimiter
		if isinstance(element,str) and delimiter:
			element = tuple(element.split(delimiter))
		elif not isinstance(element,str):
			element = tuple(element)

		# Boolean whether element is a tuple, otherwise is object that is explicit key in dictionary
		istuple = isinstance(element,tuple)

		# Update iterable with elements 
		if not istuple:
			# elements is object and iterable is to be updated at first level of nesting
			isdict = not reset and isinstance(i.get(element),dict) and isinstance(value,dict)
			if isdict:
				i[element].update(value)
			else:
				i[element] = value
		else:
			# elements is list of nested keys and the nested values are to be extracted from iterable and set with value
			try:
				while e<len(element)-1:
					if i.get(element[e]) is None:
						i[element[e]] = {}
					i = i[element[e]]
					e+=1
				isdict = not reset and isinstance(i.get(element[e]),dict) and isinstance(value,dict)
				if isdict:
					i[element[e]].update(value)
				else:
					i[element[e]] = value
			except:
				pass

	return

def getter(iterable,elements,default=None,delimiter=False,copy=False):
	'''
	Get nested value in iterable with nested elements keys

	Args:
		iterable (dict): dictionary of values
		elements (str,list): DELIMITER separated string or list to nested keys of location to get value
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

	# Get nested element if iterable, based on elements
	if not isinstance(elements,list):
		# elements is object and value is to be got from iterable at first level of nesting
		try:
			return i[elements]
		except:
			return default
	else:
		# elements is list of nested keys and the nested values are to be extracted from iterable
		try:
			while e<len(elements):
				i = i[elements[e]]
				e+=1			
		except:
			return default

	return copier(elements[e-1],i,copy)

def popper(iterable,elements,default=None,delimiter=False,copy=False):
	'''
	Pop nested value in iterable with nested elements keys

	Args:
		iterable (dict): dictionary to be popped in-place
		elements (str,list): DELIMITER separated string or list to nested keys of location to pop value
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
		elements (str,list): DELIMITER separated string or list to nested keys of location to set value
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

def plant(old,new,key):
	'''
	Transfer leaf keys and values corresponding to key of old dictionary to new dictionary
	Args:
		old (dict): Old dictionary to retrieve leaves corresponding to key
		new (dict): New dictionary to place leaves corresponding to key
		key (object): Key of leaves to retrieve
	'''

	for branch,leaf in leaves(old,key,types=(dict,),returns='both'):
		grow(new,branch,leaf)
	return


def grow(dictionary,branch,leaf):
	'''
	Insert nested branch into dictionary in-place
	Args:
		dictionary (dict): Dictionary to insert nested branch
		branch (iterable): Iterable of nested branch
		leaf (object): Object to insert at leaf of nested branch
	'''
	temporary = dictionary
	depth = len(branch)
	for length,key in enumerate(branch):
		nest = length < (depth-1)
		old = temporary.get(key)
		if nest:
			if not isinstance(old,dict):
				new = {}
			else:
				new = old
		else:
			new = leaf
		temporary[key] = new
		if nest:
			temporary = temporary[key]
	return


def leaves(iterable,key,types=(dict,),returns='value'):
	'''
	Find and yield branch of key in nested iterable
	Args:
		iterable (iterable): Iterable of nested keys
		key (object): Key in iterable
		types (tuple[type]): Allowed nested types of iterable values		
		returns (str): Return of either {'value',key','both'}
	Yields:
		value (tuple[object]): Found path of branch in iterable
	'''	
	try:
		if not isinstance(iterable,types):
			raise
		for item in iterable:
			if isinstance(iterable,dict):
				value = iterable[item]
			else:
				value = item
			if item == key:
				if returns == 'value':
					yield value
				elif returns == 'key':
					yield (item,)				
				elif returns == 'both':
					yield ((item,),value)				
			for value in leaves(value,key,types=types,returns=returns):
				if returns == 'value':
					yield value
				elif returns == 'key':
					yield (item,*value)				
				elif returns == 'both':
					yield ((item,*value[0]),value[1])				
	except:
		pass
	return


def branches(iterable,keys,types=(dict,),returns='value',exceptions=(str,)):
	'''
	Find and yield branch of set of keys at same depth in nested iterable
	Args:
		iterable (iterable): Iterable of nested keys
		keys (iterable[object],object): Keys in iterable
		types (tuple[type]): Allowed nested types of iterable values		
		returns (str): Return of either {'value',key','both'}
		exceptions (tuple[type]): Exceptional iterable key types to exclude		
	Yields:
		value (tuple[object]): Found path of branch in iterable
	'''	
	
	# TODO: Fix bug of returning expanded keys in groups of keys instead of separate yields for each key
	
	assert returns in ['value'], 'TODO: returns "%s" not implemented'%(returns)

	if not isiterable(keys,exceptions=exceptions):		
		keys = (keys,)
		slices = 0
	else:
		slices = slice(None)
	try:
		if not isinstance(iterable,types):
			raise
		if all(key in iterable for key in keys):
			if isinstance(iterable,dict):
				values = (iterable[key] for key in keys)
			else:
				values = keys	
			if returns == 'value':
				yield tuple(values)[slices]
			elif returns == 'key':
				for key in keys:
					yield tuple((key,))
			elif returns == 'both':
				for key,value in zip(keys,values):
					yield tuple(((key,),value))
		for item in iterable:
			if isinstance(iterable,dict):
				value = iterable[item]
			else:
				value = item
			for values in branches(value,keys,types=types,returns=returns,exceptions=exceptions):
				if returns == 'value':
					yield values
				elif returns == 'key':
					yield (item,*values)
				elif returns == 'both':
					yield ((item,*values[0]),values[1])
	except:
		pass
	return	


def counts(iterable,types=(dict,)):
	'''
	Count number of leaves in nested iterable
	Args:
		iterable (iterable): Iterable of nested keys
		types (tuple[type]): Allowed nested types of iterable values
	Returns:
		count (int): Number of leaves in iterable
	'''
	count = 0
	if isinstance(iterable,types):
		for item in iterable:
			if isinstance(iterable,dict):
				value = iterable[item]
			else:
				value = item
			count += counts(value)
	else:
		count = 1
	return count

def updater(iterable,elements,copy=False,clear=False,func=None):
	'''
	Update nested iterable with elements

	Args:
		iterable (dict): dictionary to be updated in-place
		elements (dict): dictionary of nested values to update iterable
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		clear (bool): boolean of whether to clear iterable when the element's value is an empty dictionary
		func(callable,None): Callable function that accepts key,iterable,elements arguments to modify value to be updated based on the given dictionaries
	'''		

	# Setup func as callable
	if not callable(func):
		func = lambda key,iterable,elements: elements[key]

	# Clear iterable if clear and elements is empty dictionary
	if clear and elements == {}:
		iterable.clear()

	if not isinstance(elements,(dict)):
		# elements is object and iterable is directly set as elements
		key = None
		iterable = copier(key,elements,copy)
		return

	# Recursively update iterable with elements
	for key in elements:
		if isinstance(iterable.get(key),dict):
			if key not in iterable:
				iterable.update({key: copier(key,func(key,iterable,elements),copy)})
			elif not isinstance(elements[key],dict):
				iterable[key] = copier(key,func(key,iterable,elements),copy)
			else:
				updater(iterable[key],elements[key],copy=copy,clear=clear,func=func)
		else:
			iterable.update({key:copier(key,func(key,iterable,elements),copy)})
	return

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
		groups = copying.deepcopy(groups)
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
		return [{k:d[k] for d in dicts for k in d} for dicts in itertools.product(*dictionaries)]

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



def finder(iterable,key):
	'''
	Find and yield key in nested iterable

	Args:
		iterable (dict): dictionary to search
		key (object): key to find in iterable dictionary

	Yields:
		value (object): Found values with key in iterable
	'''	

	# Recursively find and yield value associated with key in iterable		
	try:
		if not isinstance(iterable,dict):
			raise
		for k in iterable:
			if k == key:
				yield iterable[k]
			for v in finder(iterable[k],key):
				yield v
	except:
		pass
	return
				
def replacer(iterable,key,replacement,append=False,copy=True,values=False):
	'''
	Find and replace key in-place in iterable with replacement key

	Args:
		iterable (dict): dictionary to be searched
		key (object): key to be replaced with replacement key
		replacement (object): dictionary key to replace key
		append (bool): boolean on whether to append replacement key to dictionary with value associated with key
		copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		values (bool): boolean of whether to replace any values that equal key with replacement in the iterable 
	'''	

	# Recursively find where nested iterable keys exist, and replace or append in-place with replacement key
	try:
		keys = list(iterable)
		for k in keys:
			if k == key:
				if append:
					iterable[replacement] = copier(replacement,iterable.get(key),copy)
					k = replacement
				else:
					iterable[replacement] = copier(replacement,iterable.pop(key),copy)
					k = replacement  
			if values and iterable[k] == key:
				iterable[k] = copier(k,replacement,copy)
			replacer(iterable[k],key,replacement,append=append,copy=copy,values=values)
	except Exception as e:
		pass
	return



def formatstring(key,iterable,elements,*args,**kwargs):

	'''
	Format values in iterable based on key and elements

	Args:
		key (object): key to index iterable for formatting
		iterable (dict): dictionary with values to be formatted
		elements (dict): dictionary of elements to format iterable values

	Returns:
		Formatted value based on key,iterable, and elements
	'''	


	# Get value associated with key for iterable and elements dictionaries
	try:
		i = iterable[key]
	except:
		i = None
	e = elements[key]
	n = 0
	m = 0


	# Return elements[key] if kwargs[key] not passed to function, or elements[key] is not a type to be formatted
	if key not in kwargs or not isinstance(e,(str,tuple,list)):
		return e

	# Check for different cases of types of iterable[key] and elements[key] to be formatted

	# If iterable[key] is not a string, or iterable tuple or list, return value based on elements[key]
	if not isinstance(i,(str,tuple,list)):

		# If elements[key] is a string, string format elements[key] with args and kwargs and return the formatted value
		if isinstance(e,str):
			m = e.count('%')
			if m == 0:
				return e
			else:
				return e%(tuple((*args,*kwargs[key]))[:m])

		# If elements[key] is an iterable tuple or list, string format each element of elements[key] with args and kwargs and return the formatted value as a tuple
		elif isinstance(e,(tuple,list)):
			m = 0
			e = [x for x in e]
			c = [j for j,x in enumerate(e) if isinstance(x,str) and x.count('%')>0]
			for j,x in enumerate(e):
				if not isinstance(x,str):
					continue
				m = x.count('%')
				if m > 0:
					k = c.index(j)
					e[j] = x%(tuple((*args,*kwargs[key]))[k:m+k])
			e = tuple(x for x in e)
			return e

		# If elements[key] is other object, return elements[key]
		else:
			return e

	# If iterable[key] is a string, format iterable[key] based on elements[key]
	elif isinstance(i,str):

		# Get number of formatting elements in iterable[key] string to be formatted
		n = i.count('%')
		if n == 0:
			# If iterable[key] has no formatting elements, return based on elements[key]

			# If elements[key] is a string, string format elements[key] with args and kwargs and return the formatted value
			if isinstance(e,str):
				m = e.count('%')
				if m == 0:
					return e
				else:
					return e%(tuple((i,*args,*kwargs[key]))[:m])

			# If elements[key] is an iterable tuple or list, string format each element of elements[key] with args and kwargs and return the formatted value as a tuple
			elif isinstance(e,(tuple,list)):
				m = 0
				e = [x for x in e]
				c = [j for j,x in enumerate(e) if isinstance(x,str) and x.count('%')>0]	
				for j,x in enumerate(e):
					if not isinstance(x,str):
						continue
					m = x.count('%')
					if m > 0:
						k = c.index(j)
						if isinstance(i,str):
							e[j] = x%(tuple((i,*args,*kwargs[key]))[k:m+k])
						else:
							e[j] = x%(tuple((*i,*args,*kwargs[key]))[k:m+k])										
				e = tuple(x for x in e)
				return e

			# If elements[key] is other object, return elements[key]
			else:
				return e
		# If iterable[key] string has non-zero formatting elements, format iterable[key] string with elements[key], args, and kwargs
		else:
			if isinstance(e,str):
				return i%(tuple((e,*args,*kwargs[key]))[:n])
			elif isinstance(e,(tuple,list)):
				return i%(tuple((*e,*args,*kwargs[key]))[:n])
			else:
				return e

	# If iterable[key] is an iterable tuple or list, string format each element of iterable[key] with elements[key],args and kwargs and return the formatted value as a tuple
	elif isinstance(i,(tuple,list)):
		i = [str(x) for x in i]
		n = 0
		c = [j for j,x in enumerate(i) if isinstance(x,str) and x.count('%')>0]	
		for j,x in enumerate(i):
			n = x.count('%')
			if n > 0:
				k = c.index(j)				
				if isinstance(e,str):
					i[j] = x%(tuple((e,*args,*kwargs[key]))[k:n+k])
				else:
					i[j] = x%(tuple((*e,*args,*kwargs[key]))[k:n+k])										

		if n == 0:
			if isinstance(e,str):
				m = e.count('%')
				if m == 0:
					return e
				else:
					return e%(tuple((i,*args,*kwargs[key]))[:m])
			elif isinstance(e,(tuple,list)):
				m = 0
				e = [x for x in e]
				c = [j for j,x in enumerate(e) if isinstance(x,str) and x.count('%')>0]					
				for j,x in enumerate(e):
					if not isinstance(x,str):
						continue
					m = x.count('%')
					if m > 0:
						k = c.index(j)				
						if isinstance(i,str):
							e[j] = x%(tuple((i,*args,*kwargs[key]))[k:m+k])
						else:
							e[j] = x%(tuple((*i,*args,*kwargs[key]))[k:m+k])										
				e = tuple(x for x in e)
				return e
			else:
				return e			
			return e
		else:
			i = tuple(x for x in i)
			return i
	else:
		return e