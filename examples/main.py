# Import packages
import os,sys
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.io import load
from src.system import Dict
from src.iterables import namespace
from src.optimize import Optimizer,Objective,Metric,Callback

# Parse command line arguments (default: settings.json)
arguments = {'--settings':'settings.json'}
arguments = argparser(arguments)
settings = arguments['settings']

# Load settings file (wrap in Dict class for object-style key-value pair attributes)
settings = load(settings,wrapper=Dict)

# Load classes (from path of class i.e) src.quantum.py)
Model = load(settings.cls.model)
State = load(settings.cls.state)
Label = load(settings.cls.label)
Callback = load(settings.cls.callback)

# Get optimizer and system settings
hyperparameters = settings.optimize
system = settings.system

# Initialize model classes
model = Model(**{**settings.model,**dict(system=system)})
state = State(**{
	**namespace(state,model),
	**settings.state,**dict(model=model,system=system)
	})
label = Label(**{
	**namespace(label,model),
	**settings.label,
	**dict(model=model,system=system)
	})

# Initialize label and model with state
label.__initialize__(state=state)
model.__initialize__(state=state)

# Set optimizer arguments
func = model.parameters.constraints
callback = Callback(**{
	**namespace(callback,model),
	**settings.callback,
	**dict(model=model,system=system)
	})
arguments = ()
keywords = {}

# Initialize optimizer classes
metric = Metric(state=state,label=label,
	arguments=arguments,keywords=keywords,
	hyperparameters=hyperparameters,system=system)
func = Objective(model,
	func=func,callback=callback,metric=metric,
	hyperparameters=hyperparameters,system=system)
callback = Callback(model,
	func=func,callback=callback,metric=metric,
	arguments=arguments,keywords=keywords,
	hyperparameters=hyperparameters,system=system)
optimizer = Optimizer(func=func,callback=callback,
	arguments=arguments,keywords=keywords,
	hyperparameters=hyperparameters,system=system)

# Get model parameters and state
parameters = model.parameters()
state = model.state()

# Run optimizer
parameters = optimizer(parameters,state=state)