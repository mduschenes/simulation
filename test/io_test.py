#!/usr/bin/env python

# Import python modules
import pytest
import os,sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','.','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import array,nparray,rand,allclose,linspace,logspace,absolute,difference,maximums,addition,inplace,log,arrays,dicts,iterables,scalars,seeder,prod,nan,is_naninf,is_scalar,nan
from src.io import load,dump,merge,join,split,edit,dirname,exists,glob,cp,rm,mkdir,cd

# Logging
# from src.utils import logconfig
# conf = 'config/logging.conf'
# logger = logconfig(__name__,conf=conf)

def equalizer(a,b):

	if isinstance(a,dicts) and isinstance(b,dicts):
		return all(equalizer(a[i],b[j]) for i,j in zip(a,b) if i==j)
	elif isinstance(a,iterables) and isinstance(b,iterables):
		return all(equalizer(i,j) for i,j in zip(a,b))
	elif (isinstance(a,scalars) or is_scalar(a)) and (isinstance(b,scalars) or is_scalar(b)):
		return a == b
	else:
		return False


def test_path(path='.tmp/data.hdf5'):
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

	directory = '.tmp'
	options = dict(verbose=True)
	
	mkdir(directory)

	with cd(directory):	
	

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


	rm(directory)

	print('Passed')

	return



def test_load_dump_merge(path='.tmp'):

	n = 3
	g = 4
	l = 3
	attrs = ['data','values','parameters'][:l]
	shape = (2,3)
	key = seeder(123,size=(n,g,l))
	directory = '.tmp'
	paths = ['settings.json','data.hdf5']
	options = dict(verbose=True)

	mkdir(directory)

	with cd(directory):

		for path in paths:
			
			obj = {join(i,path):{
					f'{i}.{j}':{attr:rand(shape,key=key[i][j][k]) for k,attr in enumerate(attrs)}
					for j in range(g)
					}
				for i in range(n)
			}

			for i in obj:
				dump(obj[i],i,**options)
				tmp = load(i,**options)
				assert equalizer(obj[i],tmp)


			data = join('*',path)
			merge(data,path,**options)

			data = load(path,**options)

			data = {j:data[i] for i,j in zip(data,obj)}

			assert equalizer(data,obj)

	rm(directory)
	
	print('Passed')

	return


def test_load_dump_df(path='.tmp'):

	n = 6
	g = 4
	l = 2

	scale = 'log'
	scale = 'linear'
	base = 10
	number = 5
	density = 'probability'

	attrs = dict(
		data=lambda index,shape,key:rand(shape,random='choice',array=array({'linear':linspace(0,1,number),'log':logspace(-20,0,number)}[scale]),key=key,dtype=float),
		parameters=lambda index,shape,key:rand(shape,random='randint',bounds=[0,g],key=key,dtype=int)
		)
	shape = dict(data=(3,100),parameters=(3,))
	key = seeder(123,size=(n,g,l))
	directory = '.tmp'
	paths = ['data.hdf5']
	options = dict(verbose=False)
	opts = dict(chunk=2,wrapper='df')

	func = load('src.functions.func_hist')
	arguments = tuple()
	keywords = dict(bins=2*number,range={'linear':[0,1],'log':[1e-20,1e0]}[scale],scale=scale,base=base,density=density)
	variables = dict(x='data',label='parameters')

	mkdir(directory)

	with cd(directory):

		for path in paths:
			
			obj = {join(i,path):{
					f'{i}.{j}':{attr:attrs[attr](i,shape=shape[attr],key=key[i][j][k]) for k,attr in enumerate(attrs)}
					for j in range(g)
					}
				for i in range(n)
			}

			for i in obj:
				dump(obj[i],i,**options)

			plots = {}
			for data in load(join('*',path),**{**options,**opts}):
				groups = data.groupby(variables['label'])
				for group in groups.groups:
					key = group
					value = groups.get_group(group)
					x,y = func(value[variables['x']],*arguments,**keywords)
					if key not in plots:
						plots[key] = {}
						plots[key]['x'] = x
						plots[key]['y'] = y
						plots[key]['label'] = key
					plots[key]['x'] = x
					plots[key]['y'] += y
				del data,groups

			mplstyle = '../config/plot.mplstyle'
			with matplotlib.style.context(mplstyle):

				fig,ax = plt.subplots()

				for index,key in enumerate(plots):

					x = plots[key]['x']
					y = plots[key]['y']

					size = len(plots)
					length = size//2

					if size > 1:
						if scale in ['linear']:
							z = array([*(2*x[:1]-x[1:2]),*x,*(2*x[-2:-1]-x[-1:])])
							w = 1/(size+2*length)
							
							diff = difference(z[:-1])
							step = diff*(-1/2 + (index+length)*w)
							
							x += step
							width = diff*w
						
						elif scale in ['log']:
							z = array([*log(x[:1]**2/x[1:2]),*log(x),*log(x[-2:-1]**2/x[-1:])])/log(base)

							w = 1/(size+2*length)

							diff = difference(z[:-1])
							step = base**(diff*(-1/2 + (index+length)*w))
							
							x *= step

							z = array([*log(x[:1]**2/x[1:2]),*log(x),*log(x[-1:]**2/x[-2:-1])])/log(base)

							width = base**(z[1:-1]*(1-w/2) + z[2:]*(w/2)) - base**(z[:-2]*(w/2) + z[1:-1]*(1-w/2))

					if density is None:
						pass
					elif density in ['probability']:
						y /= maximums(addition(y),1)
					y = inplace(y,y==0,nan)

					args = (x,y)
					kwargs = dict(
						label='$%s$'%(plots[key]['label']),
						color=getattr(plt.cm,'viridis')((index+1)/(len(plots)+1)),
						alpha=0.7,
						width=width,
						linewidth=2,
						edgecolor=matplotlib.colors.colorConverter.to_rgba('black', alpha=0.6),
						align='center'
						)
					ax.bar(*args,**kwargs)

				ax.set_xlabel(xlabel='$\\textrm{Probability}$')
				ax.set_ylabel(ylabel='$\\textrm{Count}$')

				if scale in ['linear']:
					ax.set_xscale(value=scale)
					ax.set_xlim(xmin=-0.05,xmax=1.05)
					ax.set_xticks(ticks=[i for i in [0,0.2,0.4,0.6,0.8,1]])
					ax.set_xticklabels(labels=['$%s$'%(i) for i in [0,0.2,0.4,0.6,0.8,1]])
				elif scale in ['log']:
					ax.set_xscale(value=scale)
					ax.set_xlim(xmin=1e-21,xmax=5e0)
					ax.set_xticks(ticks=[i for i in [1e-20,1e-16,1e-12,1e-8,1e-4,1e0]])
					ax.set_xticklabels(labels=['$%s^{%s}$'%(base,i) for i in [-20,-16,-12,-8,-4,0]])
				ax.set_ylim(ymin=-0.05,ymax=1.05)
				ax.set_yticks(ticks=[0,0.2,0.4,0.6,0.8,1])
				ax.set_yticklabels(labels=['$%s$'%(i) for i in [0,0.2,0.4,0.6,0.8,1]])					

				ax.grid(visible=True,which='major',zorder=0)
				ax.set_axisbelow(b=True)

				ax.legend(title='$\\textrm{Parameter}$',ncol=1,loc='upper right')

				fig.set_size_inches(w=20,h=20)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname='plot.bar.pdf',bbox_inches='tight',pad_inches=0.5)


	rm(directory)
	
	print('Passed')

	return


def test_hdf5(path='.tmp/data.hdf5'):

	g = 3
	n = 2
	shape = (2,3)
	key = seeder(123)
	groups = ['%d'%(i) for i in range(g)]
	instances = ['%d'%(i) for i in range(n)]
	datasets = ['data','values','parameters']
	attributes = {'scalar':0.12321312,'integer':123,'string':'hello world --- goodbye','bool':True,'None':nan}
	attrs = [*datasets,*attributes]
	data = {
		group: {
			instance:{
				**{attr: rand(shape,key=key) for attr in datasets},
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

def test_pd(path='.tmp/data.hdf5'):

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
	# test_load_dump()
	# test_load_dump_merge()
	test_load_dump_df()
	# test_importlib()
	# test_glob()
	# test_hdf5()
	# test_pd()
	# test_parallel()
