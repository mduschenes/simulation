#!/usr/bin/env python

# Import python modules
import os,sys,warnings,itertools,inspect,traceback
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
from src.io import cd

def submit_pc(*args):
	job,cmd,args = os.path.abspath('%s'%(args[0])),os.path.abspath('%s'%(args[1])),' '.join(args[2:])

	updates = {'CMD':cmd,'ARGS':args}
	replace(job,updates)

	exe = [job]	
	return exe

def submit_slurm(*args):
	job,cmd,args = os.path.abspath('%s'%(args[0])),os.path.abspath('%s'%(args[1])),' '.join(args[2:])

	updates = {'CMD':cmd,'ARGS':args}
	replace(job,updates)

	exe = ['sbatch','<',job]
	return exe

def submit_null(*args):
	exe = []
	return exe


def replace(path,updates):
	for update in updates:
		update=['sed','-i','s%%%s=.*%%%s=%r%%g'%(update,update,updates[update]),path,]
		call(*update)
	return

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
			args = ' '.join(args)
			stdout = os.system(args)
		else:
			args = ' '.join(args)			
			os.system('cat %s;echo;echo;echo;echo;echo'%(args))
			stdout = args
	return stdout



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
