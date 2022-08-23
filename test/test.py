#!/usr/bin/env python

import os,sys,subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.call import search,sed

path = 'job.slurm'

options = {'nodes':1,'test':2,'array':'[0-100]'}



patterns = {
	'nodes':2,
	'test':2,
	'array':'1-100%20'
}

update(path,patterns,default)
