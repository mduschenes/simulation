#!/usr/bin/env python

import time,gc
import Queue
import numpy as np
import pandas as pd
import multiprocessing as mp
from random import randint

#==========================================================
# DATA HANDLER
#==========================================================

done = '__DONE__'

class multiprocess(object):
	def __init__(self,**kwargs):
		self.done = '__DONE__'
		for kwarg in kwargs:
			setattr(self,kwarg,kwargs[kwarg])
		return

	def data_handler(self,data,queue_call, queue_receive, queue_data, n_processes,**kwargs):

		# Handle data requests
		finished = 0
		done = self.done
		while finished < n_processes:

			try:
				# Get the label we sent in
				labels = queue_call.get(False)
			except Queue.Empty:
				continue
			else:
				if labels == done:
					finished += 1
				else:
					try:
						values = {label:data[label].values for label in labels}
						queue_receive.put(values)
					except:
						pass    

	#==========================================================
	# PROCESS DATA
	#==========================================================

	def process_data(self,queue_call, queue_receive, queue_data,func,*args,**kwargs):

		results = {}

		indices = range(kwargs.get('indices'))

		for i in indices:

			# Send an index value
			queue_call.put(labels)

			# Handle the return call
			while True:
				try:
					values = queue_receive.get(False)
				except Queue.Empty:
					continue
				else:
					results.append(func(values))
					break

		queue_call.put(self.done)
		queue_data.put(results)   

	#==========================================================
	# START MULTIPROCESSING
	#==========================================================

	def multiprocess(self,data,n_processes):

		processes = []
		results  = []

		# Create queues
		queue_call = multiprocessing.Queue()
		queue_receive = multiprocessing.Queue()
		queue_data = multiprocessing.Queue()

		for process in range(n_processes): 

			if process == 0:

					p = multiprocessing.Process(
						target = data_handler,
						args=(data,queue_call, queue_receive, queue_data, n_processes))
					processes.append(p)
					p.start()

			p = multiprocessing.Process(
					target = process_data,
					args=(queue_call, queue_receive, queue_data))
			processes.append(p)
			p.start()

		for process in range(n_processes):
			value = queue_data.get()    
			results.append(value)

		for p in processes:
			p.join()    

		print(resultsvpn-)



if __name__ == "__main__":
	n = 10000
	p = 1000
	data = np.random.rand(n,p)
	columns = ['x_%d'%(d) for in range(m)]
	data = pd.DataFrame(data=data,columns=columns)
	multiprocess(data,n_processes = 4)
