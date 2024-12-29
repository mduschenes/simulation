#!/usr/bin/env python

# Import python modules
import os,sys,datetime,shutil
import atexit

# Logging
import logging
import logging.config,configparser


def config(name=None,conf=None,**kwargs):
	'''
	Configure logging
	Args:
		name (str): File name for logger
		conf (str): Path for logging config
		kwargs (dict): Additional keywork arguments to overwrite config
	Returns:
		logger (logger): Configured logger
	'''

	name = __name__ if name is None else name

	logger = logging.getLogger(name)

	default = 'logging.conf'
	conf = default if conf is None else conf
	file = kwargs.get('file')
	path = os.path.abspath(os.path.expandvars(os.path.expanduser(file))) if file is not None else None
	delim = '.'
	ext = '%s.tmp'%(datetime.datetime.now().strftime('%d.%M.%Y.%H.%M.%S.%f'))
	existing = os.path.exists(conf)

	props = ['loggers','handlers','formatters','keys']
	keys  = {'class':'FileHandler'}
	separator = '_'

	if not existing:
		source = os.path.join(os.path.abspath(os.path.dirname(__file__)),default)
		destination = os.path.join(os.path.abspath(os.path.dirname(conf)) if conf is not None else os.getcwd(),delim.join([default,ext]))
		directory = os.path.abspath(os.path.dirname(destination))
		if directory not in ['',None] and not os.path.exists(directory):
			os.makedirs(directory)
		shutil.copy2(source,destination)
		conf = destination
	else:
		source = conf
		destination = delim.join([default,ext])
		directory = os.path.abspath(os.path.dirname(destination))
		if directory not in ['',None] and not os.path.exists(directory):
			os.makedirs(directory)
		shutil.copy2(source,destination)
		conf = destination

	if conf is not None:
		try:
			config = configparser.ConfigParser()
			config.read(conf)

			sections = [section for section in config if all(config[section].get(key) == keys[key] for key in keys)]
			for section in sections:

				name = section.split(separator)[-1]

				if file is None:

					sections = list(config)
					for section in sections:
						for prop in props:
							if prop not in config[section]:
								continue
							if name in config[section][prop]:
								names = config[section][prop].split(',')
								config[section][prop] = ','.join([
									i for i in names
									if i not in [name]])
				
					sections = list(config)				
					for section in sections:
						subname = section.split(separator)[-1]
						if (subname == name):
							config.pop(section);
							continue
					
				else:

					directory = os.path.dirname(path) if path is not None else None
					if (directory is not None) and (not os.path.exists(directory)):
						os.makedirs(directory)

					prop = 'args'
					config[section][prop] = '(%s)'%(','.join([
						'"%s"'%(path),
						# '"%s"'%('"a"'),
						*(config[section][prop][1:-1].split(',')[1:] 
						if not os.path.exists(path) else ['"a"'])]
						))

			sections = list(config)				
			for section in sections:						
				for prop in props:
					if prop not in config[section]:
						continue

					if not config[section][prop]:
						config.pop(section);
						break


			sections = list(config)				
			for section in sections:						
				for prop in props:
					if prop not in config[section]:
						continue

					names = config[section][prop].split(',')
					for name in names:
						if not any(name in section.split(separator)[-1] for section in config if separator in section):
							config[section][prop] = ','.join([
								i for i in names
								if i not in [name]])
					if not config[section][prop]:
						config.pop(section);
						break

			sections = list(config)				
			for section in sections:						
				for prop in props:
					if prop not in config[section]:
						continue

					if not config[section][prop]:
						config.pop(section);
						break

			with open(conf, 'w') as configfile:
				config.write(configfile)
			
			try:
				logging.config.fileConfig(conf,disable_existing_loggers=False,defaults={'__name__':datetime.datetime.now().strftime('%d.%M.%Y.%H.%M.%S.%f')}) 	
			except KeyError:
				pass

		except Exception as exception:
			pass

		logger = logging.getLogger(name)


	try:
		os.remove(conf)
	except:
		pass

	return logger


class Logger(object):
	def __init__(self,name=None,conf=None,file=None,cleanup=None,verbose=None,**kwargs):
		'''
		Logger class
		Args:
			name (str,logger): Name of logger or Python logging logger
			conf (str): Path to configuration
			file (str): Path to log file
			cleanup (bool): Cleanup log files upon exit
			verbose (int,str,bool): Verbosity
			kwargs (dict): Additional keyword arguments
		'''
		if (name is None) or isinstance(name,str):
			try:
				self.logger = config(name=name,conf=conf,file=file,**kwargs)
			except Exception as exception:
				self.logger = logging.getLogger(name)
		else:
			self.logger = name

		self.name = name
		self.conf = conf
		self.file = file

		self.string = os.path.basename(str(self.file)) if self.file is not None else 'stdout'

		self.cleanup = cleanup
		self.__clean__()

		self.verbosity = {
			'notset':0,'debug':10,'info':20,'warning':30,'error':40,'critical':50,
			'Notset':0,'Debug':10,'Info':20,'Warning':30,'Error':40,'Critical':50,
			'NOTSET':0,'DEBUG':10,'INFO':20,'WARNING':30,'ERROR':40,'CRITICAL':50,
			**{i:i for i in range(0,100+1,1)},
			-1:50,
			True:100,False:0,None:0,
			}
		self.verbose = self.verbosity.get(verbose,verbose)
		
		self.verbosity.pop(None);

		return
	
	def log(self,verbose=None,msg=None):
		'''
		Log messages
		Args:
			verbose (int,str,bool): Verbosity of message
			msg (str): Message to log
		'''
		verbose = self.verbosity.get(verbose,self.verbose)
		self.logger.log(verbose,msg)
		return

	def __clean__(self,cleanup=None):
		'''
		Set cleanup state of class
		Args:
			cleanup (bool): Cleanup log files upon exit	
		'''

		cleanup = self.cleanup if cleanup is None else cleanup

		if cleanup:
			atexit.register(self.__atexit__)
		else:
			atexit.unregister(self.__atexit__)

		return


	def __atexit__(self):
		'''
		Cleanup log files upon class exit
		'''

		loggers = [logging.getLogger(),self.logger,*logging.Logger.manager.loggerDict.values()]
		loggers = [handler.baseFilename for logger in loggers for handler in getattr(logger,'handlers',[]) if isinstance(handler,logging.FileHandler)]
		loggers = list(set(loggers))

		for logger in loggers:
			try:
				os.remove(logger)
				os.rmdir(os.path.dirname(logger))
			except:
				pass

		return

	def __str__(self):
		return self.string

	def __repr__(self):
		return self.__str__()