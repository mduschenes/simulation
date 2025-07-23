#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

import pandas as pd

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','.','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,rand,allclose,arrays,iterables,scalars,seeder,prod,nan,is_naninf,is_scalar
from src.io import load,dump,merge,join,split,edit,dirname,exists,glob,rm,mkdir,cd

# Logging
# from src.utils import logconfig
# conf = 'config/logging.conf'
# logger = logconfig(__name__,conf=conf)

def equalizer(a,b):

	if isinstance(a,dict) and isinstance(b,dict):
		return all(equalizer(a[i],b[j]) for i,j in zip(a,b) if i==j)
	elif isinstance(a,iterables) and isinstance(b,iterables):
		return all(equalizer(i,j) for i,j in zip(a,b))
	elif (isinstance(a,scalars) or is_scalar(a)) and (isinstance(b,scalars) or is_scalar(b)):
		return a == b
	else:
		return False


def test_path(path='.tmp.tmp/data.hdf5'):
	new = edit(
			path=path,
			directory=None,
			file=(lambda directory,file,ext,delimiter: delimiter.join([*file.split(delimiter)[:]])),
			ext=None,
			delimiter='.'
			)

	assert new == path, "Incorrect path edit"

	return

def test_load_dump(path=None):

	folder = '.tmp.tmp'
	options = dict(verbose=True)
	
	mkdir(folder)

	with cd(folder):	
	

		# module
		path = 'src.functions.test'

		obj = load(path,**options)

		args = (1,2,3)
		kwargs = dict(test='test')
		assert callable(obj) and equalizer(obj(*args,**kwargs),(args,kwargs))


		# json
		path =  'settings.json'

		obj = {'hi':[1.23e-12,False,None,'test string',[1,2,3],{'bye':array([1,2,3.])}],'other':[array([[1,-2,3]])]}

		dump(obj,path,**options)

		tmp = load(path,**options)

		assert equalizer(tmp,obj)

		rm(path)


		# hdf5
		path =  'data.hdf5'

		obj = {'hi':rand((1,2,3)),'by':[array([[1,-2,3]])]}

		dump(obj,path,**options)

		tmp = load(path,**options)

		assert equalizer(tmp,obj)

		rm(path)


	rm(folder)

	print('Passed')

	return



def test_load_dump_merge(path='.tmp.tmp'):


	n = 3
	g = 4
	attrs = ['data','values','parameters']
	shape = (2,3)
	directory = '.tmp.tmp'

	path = 'data.hdf5'
	data = {join(directory,i,path):{
			f'{i}.{j}':{attr:rand(shape) for attr in attrs}
			for j in range(g)
			}
		for i in range(n)
	}

	for i in data:
		dump(data[i],i)


	tmp = {}
	for i in data:
		tmp[i] = load(i)

		assert equalizer(data[i],tmp[i])


	path = 'settings.json'
	data = {join(directory,i,path):{
			f'{i}.{j}':{attr:rand(shape) for attr in attrs}
			for j in range(g)
			}
		for i in range(n)
	}

	for i in data:
		dump(data[i],i)


	tmp = {}
	for i in data:
		tmp[i] = load(i)

		assert equalizer(data[i],tmp[i])

	return


def test_hdf5(path='.tmp.tmp/data.hdf5'):

	g = 3
	n = 2
	shape = (2,3)
	groups = ['%d'%(i) for i in range(g)]
	instances = ['%d'%(i) for i in range(n)]
	datasets = ['data','values','parameters']
	attributes = {'scalar':0.12321312,'integer':123,'string':'hello world --- goodbye','bool':True,'None':nan}
	attrs = [*datasets,*attributes]
	data = {
		group: {
			instance:{
				**{attr: rand(shape) for attr in datasets},
				**{attr: attributes[attr] for attr in attributes}
				}
			for instance in instances
			}
		for group in groups
		}

	# Dump data
	wr = 'w'
	ext = 'hdf5'
	kwargs = {}

	dump(data,path,wr=wr,**kwargs)

	# Load data
	wr = 'r'
	ext = 'hdf5'
	kwargs = {}

	new = load(path,wr=wr,**kwargs)

	# Check dumped and loaded data are equal
	for group in groups:
		for instance in instances:
			for attr in attrs:
				msg = "group: %s, instance: %s, attr: %s Unequal"%(group,instance,attr)
				if isinstance(data[group][instance][attr],scalars):
					if is_naninf(data[group][instance][attr]):
						assertion = is_naninf(data[group][instance][attr]) and is_naninf(new[group][instance][attr])
					else:
						assertion = data[group][instance][attr] == new[group][instance][attr]
				else:
					assertion = allclose(data[group][instance][attr],new[group][instance][attr])
				assert assertion,msg


	path = dirname(path)

	rm(path)

	print('Passed')

	return

def test_pd(path='.tmp.tmp/data.hdf5'):

	directory = split(path,directory=True)
	file = split(path,file_ext=True)

	shape = (2,3)
	paths = 2
	indices = 4
	copies = 3

	options = dict(wr='a')

	data = {}

	for k in range(copies):

		data[k] = {
			join(directory,str(i),file):{
				str(k*(paths*indices)+i*indices+j):dict(data=[rand(shape,key=seeder(123+k))],scalar=k*10+j+0.34343,string='hello worlds',boolean=j%2==0,none=nan)
				for j in range(indices)
				}
			for i in range(paths)
			}

		for string in data[k]:
			dump(data[k][string],string,**options)

	string = join(directory,'**',file)

	options = dict(wrapper='df')

	new = load(string,**options)

	print(new)

	new = {attr: [*new[attr]] if not any(isinstance(i,tuple) for i in new[attr]) else [[array([[k for k in j] if isinstance(j,tuple) else j for j in i])] for i in new[attr]] for attr in new}
	old = {attr: [data[k][i][j][attr] for k in data for i in data[k] for j in data[k][i] if attr in data[k][i][j]] for attr in set(attr for k in data for i in data[k] for j in data[k][i] for attr in data[k][i][j])}


	for attr in old:
		assert all((not isinstance(i,arrays) and not isinstance(j,arrays)) or (is_naninf(i) and is_naninf(j)) for i,j in zip(old[attr],new[attr])) or (old[attr]==new[attr]),"\n%r\n%r"%(old[attr],new[attr])


	key = str((copies//2)*(paths*indices)+(paths//2)*indices+(indices//2))

	assert exists(join(directory,str(paths//2),file)) and key in load(join(directory,str(paths//2),file))

	rm(directory)

	print('Passed')

	return


def test_parallel(path='.tmp.parallel/data.hdf5'):

	directory = split(path,directory=True)
	mkdir(directory)
	rm(path)

	n = 24
	j = n//3
	time = 2

	file = join(directory,'func.py')
	text = '\n'.join([
		"#!/usr/bin/env python",
		"import os,sys,time,datetime",
		"",
		"ROOT = os.path.dirname(os.path.abspath(__file__))",
		"PATHS = ['','.','..','../..']",
		"for PATH in PATHS:",
		"\t"+"sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))",
		"from src.utils import rand",
		"from src.io import dump",
		"",
		"index = sys.argv[1]",
		"",
		f"path = '{path}'",
		"func = lambda index: float(index)",
		"timestamp = datetime.datetime.now().strftime('%d.%M.%Y.%H.%M.%S.%f')",
		"",
		"key = index",
		"value = dict(data=[rand((3,4))+int(index)*10],scalar=float(index),string=timestamp,boolean=True)",
		"",
		"data = {key:value}",
		"options = dict(lock=True,backup=True)",
		"",
		"dump(data,path,**options)",
		"",
		"print(key,timestamp)",
		"",
		f"time.sleep({time})"
		])
	string = f'echo "{text}" > {file}'
	os.system(string)

	string = f'chmod +x {file}'
	os.system(string)


	string = f'parallel -j {j} ./{file} {{}} ::: $(seq 0 {n-1})'

	os.system(string)


	options = dict(lock=True,backup=True)
	data = load(path,**options)

	assert (
		(len(data) == n) and 
		all(int(i) in range(n) for i in data) and
		all(data[i]['scalar']==float(i) for i in data)
		)

	rm(path)
	rm(file)
	rm(directory)

	return


def test_importlib(path=None,**kwargs):

	import os,sys,importlib

	objs = {"src.io":"load","src.quantum":"Object"}
	
	for attr in objs:
		
		obj = attr
		module = objs[attr]

		try:
			path = os.path.basename(obj).strip('.')
			data = getattr(importlib.import_module(path),module)
		except (SyntaxError,TypeError) as exception:
			logger.log(info,'Exception:\n%r\n%r'%(exception,traceback.format_exc()))
			exception = SyntaxError
			raise exception
		except Exception as exception:
			path = obj
			spec = importlib.util.spec_from_file_location(module,path)
			data = importlib.util.module_from_spec(spec)
			sys.modules[module] = data
			spec.loader.exec_module(data)
			data = getattr(data,module)

		print(data)

		assert data is not None

	print('Passed')

	return


def test_glob(path=None,**kwargs):

	directory = 'config'
	variables = {'TEST_VARIABLE':'test'}

	for variable in variables:
		os.environ[variable] = variables[variable]

	paths = {
		os.path.join(directory,'*.json'):
			[os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),directory,path))
			for path in  ['settings.json','test.json','process.json','plot.json']],
		os.path.join(directory,'{job.slurm,logging.conf}'):
			[os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),directory,path))
			for path in  ['job.slurm','logging.conf']],		
		**{os.path.join(directory,f'${variable}.json'):[os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),directory,f'{variables[variable]}.json'))] for variable in variables}
		}

	for path in paths:
		print(path)
		for i in glob(path):
			print(i)
			assert i in paths[path], "Incorrect glob(%r)"%(path)
		print()


	for variable in variables:
		del os.environ[variable]


	print('Passed')

	return





if __name__ == '__main__':
	test_load_dump()
	# test_load_dump_merge()
	# test_importlib()
	# test_glob()
	# test_hdf5()
	# test_pd()
	# test_parallel()
