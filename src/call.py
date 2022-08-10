#!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,inspect,traceback
import subprocess
import shutil
import glob as globber
import importlib
import json,jsonpickle,h5py,pickle,dill
import numpy as np
import pandas as pd

from natsort import natsorted, ns,index_natsorted,order_by_index

# Logging
import logging
logger = logging.getLogger(__name__)
debug = 0

# Import user modules
from src.io import mkdir


def call(*args,path='.',exe=True):
	'''
	Submit call to command line
	Args:
		args (iterable[str]): Arguments to pass to command line
		path (str): Path to call from
		exe (boolean): Boolean whether to issue commands
	Returns:
		stdout (str): Return of commands
	'''

	with cd(path):
		if exe:
			args = [' '.join(args)]
			# stdout = os.system(args)
			stdout = subprocess.check_output(args,shell=True).strip().decode("utf-8")
		else:
			args = ' '.join(args)			
			print(args)
			stdout = args
	return stdout

def copy(source,destination,**kwargs):
	'''
	Copy objects from source to destination
	Args:
		source (str): Path of source object
		destination (str): Path of destination object
		kwargs (dict): Additional copying keyword arguments
	'''
	assert os.path.exists(source), "source %s does not exist"%(source)

	mkdir(destination)

	args = ['cp','-rf',source,destination]

	stdout = call(*args)

	# shutil.copy2(source,destination)

	return



def sed(path,patterns):
	for pattern in patterns:
		args=[
			'sed','-i',
			("s,%s,%s,g"%(patterns[pattern]['pattern'],patterns[pattern]['replacement'])).replace(r'-',r"\-").replace(r' ',r'\ '),
			path]
		call(*args)
	return


class cd(object):
	'''
	Class to safely change paths and return to previous path
	Args:
		path (str): Path to change to
	'''
	def __init__(self,path):
		self.path = path
	def __enter__(self):
		self.cwd = os.getcwd()
		os.chdir(self.path)
	def __exit__(self,etype, value, traceback):
		os.chdir(self.cwd)



def submit_pc(*args):
	job,cmd,path,args = os.path.abspath('%s'%(args[0])),os.path.abspath('%s'%(args[1])),os.path.abspath('%s'%(args[2])),' '.join(args[3:])

	patterns = {
		'cmd':{'pattern':r'CMD=.*','replacement':'CMD=%s'%(cmd)},
		'args':{'pattern':r'ARGS=.*','replacement':'ARGS=%s'%(args)},
		'stdout':{'pattern':r'#SBATCH --output=.*','replacement':r'#SBATCH --output=%s/%x.%A.stdout'},
		'stderr':{'pattern':r'#SBATCH --error=.*','replacement':r'#SBATCH --error=%s/%x.%A.stderr'},
		}
	sed(job,patterns)

	exe = [job]	
	return exe

def submit_slurm(*args):
	job,cmd,path,args = os.path.abspath('%s'%(args[0])),os.path.abspath('%s'%(args[1])),os.path.abspath('%s'%(args[2])),' '.join(args[3:])

	patterns = {
		'cmd':{'pattern':r'CMD=.*','replacement':'CMD=%s'%(cmd)},
		'args':{'pattern':r'ARGS=.*','replacement':'ARGS=%s'%(args)},
		'stdout':{'pattern':'#SBATCH --output=.*','replacement':'#SBATCH --output=%s/%%x.%%A.stdout'%(path)},
		'stderr':{'pattern':'#SBATCH --error=.*','replacement':'#SBATCH --error=%s/%%x.%%A.stderr'%(path)},
		}
	sed(job,patterns)

	exe = ['sbatch','<',job]
	return exe

def submit_null(*args):
	exe = []
	return exe


def submit(args,path='.',device='pc',exe=True):
	'''
	Submit commands to command line
	Args:
		args (iterable[iterable[str]]): Arguments to pass to command line
		path (str,iterable[str]): Path to issue commands to command line
		device (str): Name of device to submit to
		exe (boolean): Boolean whether to issue commands
	Returns:
		stdouts (str,iterable[str]): Return of commands
	'''

	devices = {
		'pc': submit_pc,
		'slurm': submit_slurm,
		None: submit_null,
		}

	assert device in devices, 'device: "%s" not in allowed %r'%(device,list(devices))
	
	if len(args) < 1:
		args = [[]]

	single = (len(args) == 1)

	if isinstance(args[0],str):
		args = [args]
	else:
		args = args
	if isinstance(path,str):
		paths = [path]*len(args)
	else:
		paths = path

	stdouts = []
	for arg,path in zip(args,paths):
		cmd = devices[device](*arg)
		stdout = call(*cmd,path=path,exe=exe)
		stdouts.append(stdout)

	if single:
		stdouts = stdouts[0]

	return stdouts


