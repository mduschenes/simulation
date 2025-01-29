#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime
from copy import deepcopy

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.workflow import Job,Task,Dict
from src.workflow import call,timeout,permuter,sleep,scalars,iterables


def test_call(*args,**kwargs):

	path = 'text.tmp'
	string = 'Test Call\nHello World'

	kwargs = dict(
		path=None,wrapper=None,env=None,
		stdin=None,stdout=None,stderr=None,shell=None,time=None,
		execute=True,verbose=True
		)

	args = [f'echo {string} > {path}']
	result = call(*args,**kwargs)
	print('-----\n%s\n'%(result))

	args = [f'xargs echo | sed "s/World/Python/g" < {path}']
	result = call(*args,**kwargs)
	print('-----\n%s\n'%(result))

	args = [f'sed -i "s/l/L/g" {path}']
	result = call(*args,**kwargs)
	print('-----\n%s\n'%(result))

	args = [f'cat {path}']
	result = call(*args,**kwargs)
	print('-----\n%s\n'%(result))

	args = [f'rm -v {path}']
	result = call(*args,**kwargs)
	print('-----\n%s\n'%(result))

	return

def test_parse(*args,**kwargs):

	def wrapper(data):

		if data is None:
			return data

		funcs = dict()

		func = 'jobid'
		def function(index,data,attrs):

			if data[index].split(splitter)[0].isdigit():
				value = int(data[index].split(splitter)[0])
				attrs['identity'] = value
			
			if data[index].count(splitter):
				if (parser not in data[index]) and (delimiter not in data[index]):
					value = data[index].split(splitter)[-1].replace('[','').replace(']','')
					value = int(value) if value.isdigit() else False
				else:
					value = [j
						for i in data[index].split(splitter)[-1].replace('[','').replace(']','').split(separator)[0].split(delimiter)
						if i
						for j in ([int(i) if i.isdigit() else False] if not i.count(parser) else (range(*(int(j)+k for k,j in enumerate(i.split(parser)))) if all(j.isdigit() for j in i.split(parser)) else [False]))
						]
					value = value if value and all(i is not None for i in value) else None
				attrs['index'] = value

			if isinstance(attrs['index'],iterables) and any(i is False for i in attrs['index']) or attrs['index'] is False:
				attrs.update({attr:None for attr in attrs})

			return
		funcs[func] = function

		func = 'jobname'
		def function(index,data,attrs):
			attrs.update(dict(name=str(data[index])))
			return
		funcs[func] = function

		func = 'state'
		def function(index,data,attrs):
			attrs.update(dict(state=str(data[index])))
			return
		funcs[func] = function

		separator,delimiter,splitter,parser = '%',',','_','-'
		
		if not any(i for i in data):
			data = []
		else:
			for i in range(len(data)-1,-1,-1):
				attrs = Dict()
				for index,attr in enumerate(funcs):
					funcs[attr](index,data[i],attrs)
				data[i] = attrs
				if not data[i] or all(data[i][attr] is None for attr in data[i]):
					data.pop(i)
				elif data[i].identity is None:
					data.pop(i)
				elif any(isinstance(data[i][attr],iterables) for attr in data[i]):
					data.extend((
						Dict({**{attr:data[i][attr] for attr in data[i] if not isinstance(data[i][attr],iterables)},**attrs})
						for attrs in permuter({attr:data[i][attr] for attr in data[i] if isinstance(data[i][attr],iterables)})
						))
					data.pop(i)
		
		boolean = lambda data,obj=max(i.identity for i in data) if data else None:data.identity==obj
		
		data = [i for i in data if boolean(i)]
		
		return data

	def equal(x,y):
		return all(i[attr]==j[attr] for i,j in zip(x,y) for attr in ['identity','name','index','state'])

	data = [['123_[2-3,5,9-10]','job','run']]
	true = [{'identity': 123, 'name': 'job', 'state': 'run', 'index': 2}, {'identity': 123, 'name': 'job', 'state': 'run', 'index': 3}, {'identity': 123, 'name': 'job', 'state': 'run', 'index': 5}, {'identity': 123, 'name': 'job', 'state': 'run', 'index': 9}, {'identity': 123, 'name': 'job', 'state': 'run', 'index': 10}]

	data = wrapper(data)

	assert equal(data,true)

	data = [['123_[2-3,5+,9-10]','job','run']]
	true = []

	data = wrapper(data)

	assert equal(data,true)

	return


def test_timeout(*args,**kwargs):
	time = 3
	pause = 0.5
	i = 1
	with timeout(time=time):
		while i:
			sleep(pause)
			print('Sleeping',i)
			i += 1

	return

def test_submit(*args,**kwargs):

	def wrapper(stdout,stderr,returncode,env=None,shell=None,time=None,verbose=None):
		def wrapper(string):
			delimiter = ' '
			types = (int,str)
			if string.count(delimiter) > 1:
				string = [i.strip() for i in string.split(delimiter) if i]
			else:
				for type in types:
					try:
						string = type(string)
						break
					except:
						pass
			return string

		delimiter = '\n'


		result = [wrapper(i) for i in stdout.split(delimiter) if i] if stdout and any(stdout.split(delimiter)) is not None else None

		result = result[0] if result is not None and len(result)==1 else result

		return result


	args = ['sbatch < job.slurm']
	kwargs = {'path': './job', 'wrapper': wrapper, 'time':0.5, 'execute': True, 'verbose': False}

	data = None
	data = 14666390

	if data is None:
		data = call(*args,**kwargs)

	identity = data

	args = [f'sacct --noheader --allocations --user=mduschen --jobs={identity} --name=job --format=jobid,jobname,state --start=now-12hours']
	kwargs = {'path': './job', 'wrapper': wrapper,'time':0.1, 'execute': True, 'verbose': False}

	time = 10
	pause = time/1000
	data = None

	with timeout(time=time):
		while data is None:
			data = call(*args,**kwargs)
			sleep(pause)
	
	stats = data

	print(identity,stats)

	return


def test_job(*args,**kwargs):

	verbose = True

	args = tuple()
	kwargs = dict(
		name='job',
		identity=None,
		device='slurm',		
		jobs=[],
		path='./job',
		data={'./job/job.slurm':'job.slurm'},
		file='job.sh',
		env={},
		options={
			'partition':'cpu',
			'time':'1:00:00',
			'mem':'1G',
			'cpus-per-task':1,
			'array':'0,3-4:1%100',
			'dependency':'afterany:',
			'output':'%x.%A.stdout',
			'error':'%x.%A.stderr',
			'get-user-env':False,
			'export':'JOB_CMD=main.py,JOB_ARGS=settings.json',
			},
		time=None,execute=True,verbose=False
		)


	job = Job(*args,**kwargs)

	job.info(verbose=verbose)

	job.setup()

	identity = job.submit()

	status = job.status()

	for state in status:
		print(state)
		for attr in status[state]:
			print('\t',attr,status[state][attr])
	print(identity,job.identity,job.jobs)

	return


def test_task(*args,**kwargs):

	verbose = True

	n = 3
	jobs = []

	for i in range(n):
		args = tuple()
		kwargs = dict(
			name='job.{i}'.format(i=i),
			identity=None,
			device='slurm',		
			jobs=['job.{i}'.format(i=i-1)] if i>0 else [],
			path='./job/{i}'.format(i=i),
			data={'./job/job.slurm':'job.slurm'},
			file='job.sh',
			env={},
			options={
				'partition':'cpu',
				'time':'1:00:00',
				'mem':'1G',
				'cpus-per-task':1,
				'array':'0,3-4:1%100',
				'dependency':'afterany:',
				'output':'%x.%A.stdout',
				'error':'%x.%A.stderr',
				'get-user-env':False,
				'export':'JOB_CMD=main.py,JOB_ARGS=settings.json',
				},
			execute=True,verbose=True
			)

		# job = Job(*args,**kwargs)

		job = kwargs

		jobs.append(job)

	args = tuple()
	kwargs = dict(
		name='job',
		identity=None,
		device='slurm',		
		jobs=jobs,
		path='./job',
		data={'./job/job.slurm':'job.slurm'},
		file='job.sh',
		env={},
		options={
			},
		execute=True,verbose=True
		)


	task = Task(*args,**kwargs)

	task.info(verbose=verbose)

	task.setup()

	print(task.identity)

	identity = task.submit()

	status = task.status()

	for state in status:
		print(state)
		for attr in status[state]:
			print('\t',attr,status[state][attr])
	print(identity,task.identity)

	return

if __name__ == '__main__':

	args = tuple()
	kwargs = dict()

	# test_call(*args,**kwargs)
	test_timeout(*args,**kwargs)
	# test_parse(*args,**kwargs)
	# test_submit(*args,**kwargs)
	# test_job(*args,**kwargs)
	# test_task(*args,**kwargs)
