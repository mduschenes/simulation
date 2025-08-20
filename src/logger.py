#!/usr/bin/env python

# Import python modules
import os,sys,datetime,shutil
import atexit

# Logging
import logging
import logging.config,configparser


def timestamp(format=None,time=None):
	'''
	Get timestamp
	Args:
		format (str): Format for timestamp
		time (datetime): datetime
	Returns:
		timestamp (str): timestamp
	'''
	if format is None:
		format = '%d.%M.%Y.%H.%M.%S.%f'
	
	if time is None:
		time = datetime.datetime.now()

	timestamp = time.strftime(format)
	
	return timestamp

def config(name=None,conf=None,file=None,write=None,**kwargs):
	'''
	Configure logging
	Args:
		name (str): File name for logger
		conf (str,dict): Path or dictionary for logging config
		file (str): Path to log file		
		write (str):  Write type of conf file
		kwargs (dict): Additional keywork arguments to overwrite config
	Returns:
		logger (logger): Configured logger
	'''

	name = __name__ if name is None else name

	logger = logging.getLogger(name)

	default = 'logging.conf'
	conf = default if conf is None else conf
	existing = os.path.exists(conf) if isinstance(conf,str) else None

	path = os.path.abspath(os.path.expandvars(os.path.expanduser(file))) if file is not None else None
	time = timestamp()
	delim = '.'
	ext = '%s.tmp'%(time) if write else None

	props = ['loggers','handlers','formatters','keys']
	keys  = {'class':'logging.FileHandler'}
	args = {'stream':{'sys.stdout':'ext://sys.stdout','sys.stderr':'ext://sys.stderr'}}
	options = dict(disable_existing_loggers=False)
	defaults = {'__name__':time}
	separator = '_'

	if not existing:
		if write:
			source = os.path.join(os.path.abspath(os.path.dirname(__file__)),os.path.basename(default))
			destination = os.path.join(os.path.abspath(os.path.dirname(conf)) if conf is not None else os.getcwd(),delim.join([os.path.basename(default),ext]))
			directory = os.path.abspath(os.path.dirname(destination))
			if directory not in ['',None] and not os.path.exists(directory):
				os.makedirs(directory)
			shutil.copy2(source,destination)
			conf = destination
		else:
			conf = os.path.join(os.path.abspath(os.path.dirname(__file__)),os.path.basename(default))
	elif isinstance(conf,str):
		if write:
			source = conf
			destination = delim.join([os.path.basename(default),ext])
			directory = os.path.abspath(os.path.dirname(destination))
			if directory not in ['',None] and not os.path.exists(directory):
				os.makedirs(directory)
			shutil.copy2(source,destination)
			conf = destination
		else:
			conf = str(conf)
	else:
		conf = dict(conf)

	if conf is not None:
		try:
			if isinstance(conf,str):
				config = configparser.ConfigParser()
				config.read(conf)
			else:
				config = logging.config.dictConfig(conf)

			sections = [section for section in config if all(config[section].get(key) == keys[key] for key in keys)]
			for section in sections:

				string = section.split(separator)[-1]

				if file is None:

					sections = list(config)
					for section in sections:
						for prop in props:
							if prop not in config[section]:
								continue
							if string in config[section][prop]:
								strings = config[section][prop].split(',')
								config[section][prop] = ','.join([
									i for i in strings
									if i not in [string]])
				
					sections = list(config)				
					for section in sections:
						subname = section.split(separator)[-1]
						if (subname == string):
							config.pop(section);
							continue
					
				else:
					directory = os.path.dirname(path) if path is not None else None
					if (directory is not None) and (not os.path.exists(directory)):
						os.makedirs(directory)

					prop = 'args'
					config[section][prop] = '(%s)'%(','.join([
						'"%s"'%(path),
						*(config[section][prop][1:-1].split(',')[1:] 
						if not os.path.exists(path) else ['"a"'])
						]
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
					strings = config[section][prop].split(',')
					for string in strings:
						if not any(string in section.split(separator)[-1] for section in config if separator in section):
							config[section][prop] = ','.join([
								i for i in strings
								if i not in [string]])
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


			if write:

				with open(conf, 'w') as configfile:
					config.write(configfile)
				
				options.update(dict())

				try:
					logging.config.fileConfig(conf,**options) 	
				except KeyError:
					pass

			else:

				conf = {}
				sections = set(section for section in ['%ss'%(section.split(separator)[0]) if separator in section else section for section in config.sections()] if section in props)
				strings = {'root':''}
				for section in sections:
					conf[section] = {}
					for string in config.sections():
						attr = string.split(separator)[-1]
						item = '%ss'%(string.split(separator)[0])
						if separator not in string or item != section:
							continue
						attr = strings.get(attr,attr)
						conf[section][attr] = {}
						for key, val in config.items(string):

							if section in ['handlers'] and key in ['args']:
								key,val = None,[i.strip() for i in val.replace('(','').replace(')','').replace('"','').split(',') if i]
								for arg in args:
									if all(v in args[arg] for v in val):
										key,val = [arg],[args[arg].get(v,v) for v in val]
										break
								if key is None:
									key,val = ['filename','mode'],[*val]
							else:
								key = [key]
								val = [val]
							for k,v in zip(key,val):
								conf[section][attr][k] = v

				for section in conf:
					for attr in conf[section]:
						for key in conf[section][attr]:
							if section in ['loggers'] and key in ['handlers']:
								conf[section][attr][key] = conf[section][attr][key].split(',') if isinstance(conf[section][attr][key],str) else [*conf[section][attr][key]] 
						if any(f'%({default})' in conf[section][attr][key] for default in defaults for key in conf[section][attr] if isinstance(conf[section][attr][key],str)):
							conf[section][attr]['defaults'] = defaults

				options.update(dict(version=1))
				conf.update(options)

				try:
					logging.config.dictConfig(conf) 	
				except KeyError:
					pass

		except Exception as exception:
			print(exception)
			pass

		logger = logging.getLogger(name)

	if write:
		try:
			os.remove(conf)
		except:
			pass
		if (conf is not None) and ((file is None) or (os.path.dirname(conf) != os.path.dirname(file))) and (os.path.exists(os.path.dirname(conf))) and (not os.listdir(os.path.dirname(conf))):
			try:
				os.rmdir(os.path.dirname(conf))
			except:
				pass

	return logger


class Logger(object):
	def __init__(self,name=None,conf=None,file=None,write=None,cleanup=None,verbose=None,**kwargs):
		'''
		Logger class
		Args:
			name (str,logger): Name of logger or Python logging logger
			conf (str,dict): Path or dictionary to configuration
			file (str): Path to log file
			write (str):  Write type of conf file
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
	
	def __call__(self,msg=None,verbose=None,**kwargs):
		'''		
		Call logger
		'''
		return self.log(verbose=verbose,msg=msg)

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