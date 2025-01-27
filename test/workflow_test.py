#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime
from copy import deepcopy

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.workflow import Job

def test_job(*args,**kwargs):

	verbose = True

	args = tuple()
	kwargs = dict(
		name='job',
		identity=None,
		device='slurm',		
		jobs=[123],
		path='./job',
		data={'./config/job.slurm':'job.slurm'},
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
		execute=True,verbose=True,
		**kwargs
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

if __name__ == '__main__':

	args = tuple()
	kwargs = dict()

	test_job(*args,**kwargs)
