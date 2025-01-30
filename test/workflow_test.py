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
		job = Job(device='slurm')
		return job.index(data)

	def equal(x,y):
		return all(i==j for i,j in zip(x,y))

	data = None
	true = (None,None)
	data = wrapper(data)
	assert equal(data,true)
	
	data = 123
	true = (123,None)
	data = wrapper(data)
	assert equal(data,true)

	data = '2'
	true = (2,None)
	data = wrapper(data)
	assert equal(data,true)

	data = '0,3-4:1%100'
	true = (None,[0,3,4])
	data = wrapper(data)
	assert equal(data,true)

	data = '123_[2-3,5,9-11]'
	true = (123,[2,3,5,9,10,11])
	data = wrapper(data)
	assert equal(data,true)

	data = '123_[2-3,5+,9-11]'
	true = (None,None)
	data = wrapper(data)
	assert equal(data,true)

	return


def test_timeout(*args,**kwargs):

	def func(i):
		i = sum(range(i))
		return

	time = 1
	pause = 0.5
	i = 1
	try:
		with timeout(time=time):
			while i:
				func(i)
				i += 1
	except timeout.error:
		print('Timed Out')
		pass

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
			'parallel':'0,3-4:1%100',
			'jobs':'afterany:',
			'stdout':'%x.%A.stdout',
			'stderr':'%x.%A.stderr',
			'get-user-env':False,
			'export':'JOB_CMD=main.py,JOB_ARGS=settings.json',
			},
		execute=True,verbose=False
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
	print(job.identity,job.jobs)
	print('---------')

	sleep(10)
	job.cleanup(verbose=True)
	print('---------')

	return


def test_task(*args,**kwargs):

	verbose = True

	def status(task):

		print('Status',task.state)
		status = task.status()

		for state in status:
			print(state)
			for attr in status[state]:
				print('\t',attr,status[state][attr])
		print(identity,task.identity)
		task.cleanup()
		print('State',task.state)
		print('---------')

		return

	n = 3
	s = 3
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
			env={'TEST_ARGS':"Hello World"},
			options={
				'partition':'cpu',
				'time':'1:00:00',
				'mem':'1G',
				'cpus-per-task':1,
				'parallel':'0,3-4:1%100',
				'jobs':'afterany:',
				'stdout':'%x.%A.stdout',
				'stderr':'%x.%A.stderr',
				'get-user-env':False,
				'export':'JOB_CMD=main.py,JOB_ARGS=settings.json',
				},
			execute=True,verbose=False
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
		execute=True,verbose=False
		)


	task = Task(*args,**kwargs)

	task.info(verbose=verbose)

	task.setup()

	identity = task.submit()

	for i in range(s):
		status(task)

	sleep(25)
	task.cleanup(verbose=True)
	print('---------')

	return

if __name__ == '__main__':

	args = tuple()
	kwargs = dict()

	# test_call(*args,**kwargs)
	# test_timeout(*args,**kwargs)
	# test_parse(*args,**kwargs)
	# test_submit(*args,**kwargs)
	# test_job(*args,**kwargs)
	test_task(*args,**kwargs)
