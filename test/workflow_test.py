#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime
from copy import deepcopy

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.workflow import Job,Task
from src.workflow import call,scalars,iterables


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
	print('-----\n%s'%(result))

	args = [f'xargs echo | sed "s/World/Python/g" < {path}']
	result = call(*args,**kwargs)
	print('-----\n%s'%(result))

	args = [f'sed -i "s/l/L/g" {path}']
	result = call(*args,**kwargs)
	print('-----\n%s'%(result))

	args = [f'cat {path}']
	result = call(*args,**kwargs)
	print('-----\n%s'%(result))

	args = [f'rm -rfv {path}']
	result = call(*args,**kwargs)
	print('-----\n%s'%(result))

	return

def test_job(*args,**kwargs):

	verbose = True

	args = tuple()
	kwargs = dict(
		name='job',
		identity=None,
		device='slurm',		
		jobs=[123],
		path='./job',
		data={'./job/job.slurm':'job.slurm'},
		file='job.sh',
		env={},
		options={
			'account':'mduschenes',
			'partition':'cpu',
			'time':'3:00:00',
			'mem':'64G',
			'cpus-per-task':4,
			'array':'1-100:1%100',
			'dependency':'afterany:',
			'output':'%x.%A.stdout',
			'error':'%x.%A.stderr',
			'get-user-env':False,
			'export':'JOB_CMD=main.py,JOB_ARGS=settings.json',
			},
		execute=True,verbose=True
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
	print(identity,job.identity)

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
			jobs=[],
			path='./job/{i}'.format(i=i),
			data={'./job/job.slurm':'job.slurm'},
			file='job.sh',
			env={},
			options={
				'account':'mduschenes',
				'partition':'cpu',
				'time':'3:00:00',
				'mem':'64G',
				'cpus-per-task':4,
				'array':'1-100:1%100',
				'dependency':'afterany:',
				'output':'%x.%A.stdout',
				'error':'%x.%A.stderr',
				'get-user-env':False,
				'export':'JOB_CMD=main.py,JOB_ARGS=settings.json',
				},
			execute=True,verbose=True
			)

		job = Job(*args,**kwargs)

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
			'account':'mduschenes',
			'partition':'cpu',
			'time':'3:00:00',
			'mem':'64G',
			'cpus-per-task':4,
			'array':'1-100:1%100',
			'dependency':'afterany:',
			'output':'%x.%A.stdout',
			'error':'%x.%A.stderr',
			'get-user-env':False,
			'export':'JOB_CMD=main.py,JOB_ARGS=settings.json',
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

	test_call(*args,**kwargs)
	# test_job(*args,**kwargs)
	# test_task(*args,**kwargs)
