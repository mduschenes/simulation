#!/usr/bin/env python

import os,sys
import matplotlib.pyplot as plt
import numpy as np


def plot_bar(path=None):

	name = 'plot.bar.pdf'
	path = os.path.join(path,name) if path is not None else None

	n = 4
	d = 3
	def func(x,y):
		z = x+y
		z[((x==0)*(y==0))+((x==(n-1))*(y==0))+((x==0)*(y==(n-1)))+((x==(n-1))*(y==(n-1)))] = 1
		return z
	x,y,z = np.arange(n),np.arange(n),np.zeros(n**(d-1))
	x,y = map(lambda i:i.ravel(),np.meshgrid(x,y))
	u = func(x,y)

	options = dict(projection='3d')
	fig = plt.figure()
	ax = fig.add_subplot(**options)

	options = dict(dx=1,dy=1,dz=u,color=plt.cm.viridis(u.flatten()/u.max()),alpha=0.8,shade=True)
	ax.bar3d(x,y,z,**options)

	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])

	ax.set_axis_off()

	if path is not None:

		options = dict(w=8,h=3)
		fig.set_size_inches(**options)

		options = dict()
		fig.subplots_adjust(**options)

		options = dict()
		fig.tight_layout(**options)

		options = dict(fname=path,bbox_inches='tight',pad_inches=0,transparent=True)
		fig.savefig(**options)

	return

def plot_curve(path=None):

	name = 'plot.curve.pdf'
	path = os.path.join(path,name) if path is not None else None

	def func(f,a=0,b=0,c=0,d=0):
		x,y,z = f.copy()
		f[0] += (a*(y - x))*d
		f[1] += (b*x - y - x*z)*d
		f[2] += (x*y - c*z)*d
		return f

	n = 10000
	d = 3
	x = (0., 1., 1.05)
	parameters = dict(a=10,b=28,c=2.667,d=0.01)

	f = np.zeros((n+1,d))
	f[0] = x
	for i in range(n):
		f[i+1] = func(f[i],**parameters)

	options = dict(projection='3d')
	fig = plt.figure()
	ax = fig.add_subplot(**options)

	options = dict(marker='')
	x,y,z = f.T
	# ax.scatter(x,y,z,**options)
	for i in range(n-1):
		ax.plot(x[i:i+2],y[i:i+2],z[i:i+2],color=plt.cm.viridis(i/n))

	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])

	ax.set_axis_off()

	if path is not None:

		options = dict(w=8,h=8)
		fig.set_size_inches(**options)

		options = dict()
		fig.subplots_adjust(**options)

		options = dict()
		fig.tight_layout(**options)

		options = dict(fname=path,bbox_inches='tight',pad_inches=0,transparent=True)
		fig.savefig(**options)


if __name__ == "__main__":

	path = sys.argv[1] if len(sys.argv)>1 else '.'

	plot_bar(path)
	plot_curve(path)
