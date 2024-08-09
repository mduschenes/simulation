# Tensor Network Methods for Efficient Classical Simulation of Non-Unitary Quantum Processes
A pure JAX based quantum circuit simulator library, including automatic differentiation of arbitrary quantum circuits, or Trotterized Hamiltonians, for use in quantum state preparation, unitary compilation, or optimization of arbitrary quantum channels with noise. 

Pre-processing, hyperparameter searches, and inter-dependent, parallelized job submission scripts, and post-processing with statistical analysis, and plotting are also included in the library.

<!-- This library is used in the preparation of the work *Characterization of Overparameterization in Simulation of Realistic Quantum Systems*, found on [arXiv](https://arxiv.org/abs/2401.05500) , or [PRA](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.062607) and data can be found in the [Zenodo repository](https://zenodo.org/records/10884844). -->

## Install
After cloning the repository, please run 
```sh
conda create --name <env>
conda activate <env>
conda install --channel conda-forge --file requirements.txt
```
which installs a Python environment with name `env` with all necessary packages, including JAX.

## Setup
Under `build`, please modify the `settings.json` file, with all model parameters, hyperparameter combinations, job settings, and plot settings. 

To configure jobs scripts, plot and processing settings, that follow the `matplotlib` API, please modify the files under `config`, or under the local working directory 
- `settings.json` : model and job settings
- `job.slurm` : job script
- `logging.conf` : logger configuration
- `process.json` : data analysis settings 
- `plot.json` : plots axes and figures 
- `plot.mplstyle` : matplotlib style

The numpy backend can be set with the environment variable (with any case)

`NUMPY_BACKEND=<jax,autograd,numpy>`

where the default backend is `jax`, and the `numpy` backend does not offer automatic differentiation.

## Examples
Examples are found in `examples`.

Example workflow `main.py`

```python
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
Call = load(settings.cls.callback)

# Get optimizer and system settings
hyperparameters = settings.optimize
system = settings.system

# Initialize model classes (getting attributes common to previous model namespaces)
model = Model(**{**settings.model,**dict(system=system)})
state = State(**{
    **namespace(State,model),
    **settings.state,**dict(model=model,system=system)
    })
label = Label(**{
    **namespace(Label,model),
    **settings.label,
    **dict(model=model,system=system)
    })

# Initialize label and model with state
label.init(state=state)
model.init(state=state)

# Set optimizer arguments
func = model.parameters.constraints
callback = Call(**{
    **namespace(Call,model),
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
```

Example settings `settings.json`

```python
{
"cls":{
    "model":"src.quantum.Channel",
    "label":"src.quantum.Label",
    "state":"src.quantum.State",
    "callback":"src.quantum.Callback"
    },
"model":{
    "data":{
        "x":{
            "operator":["X"],"site":"i","string":"x",
            "parameters":{"data":0,"random":"random","seed":123,"axis":["M"],"group":["x"]},
            "variable":true
        },
        "y":{
            "operator":["Y"],"site":"i","string":"y",
            "parameters":{"data":0,"random":"random","seed":123,"axis":["M"],"group":["y"]},
            "variable":true
        },      
        "zz":{
            "operator":["Z","Z"],"site":"i<j","string":"zz",
            "parameters":{"data":0,"random":"random","seed":123,"axis":["M"],"group":["zz"]},
            "variable":true
        },
        "noise":{
            "operator":"depolarize","site":null,"string":"noise",
            "parameters":{"data":1e-12},
            "variable":false
        }
    },
    "N":2,
    "D":2,
    "d":1,
    "M":500,
    "tau":1,
    "P":1,
    "space":"spin",
    "time":"linear",
    "lattice":"square"
    },
"system":{
    "dtype":"complex",
    "format":"array",
    "device":"cpu",
    "backend":null,
    "architecture":null,
    "seed":12345,
    "key":null,
    "instance":null,
    "cwd":".",
    "path":null,
    "path":"data.hdf5",
    "conf":"logging.conf",
    "logger":"log.log",
    "cleanup":false,
    "verbose":"info"
    },
"optimize":{
    "iterations":[0,25],
    "optimizer":"cg",
    "metric":"abs2",    
    "alpha":1e-4,"beta":1e-4,
    "search":{"alpha":"line_search","beta":"hestenes_stiefel"},
    "track":{
        "iteration":[],"objective":[],
        "alpha":[],"beta":[],
        "purity":[],
        "N":[],"D":[],"d":[],"M":[],"tau":[],"P":[],
        "noise.parameters":[]
        }   
    },
"label": {
    "operator":"haar",
    "site":null,
    "string":"U",
    "parameters":1,
    "ndim":2,
    "seed":null
    },
"state": {
    "operator":"zero",
    "site":null,
    "string":"psi",
    "parameters":true,
    "ndim":2,
    "samples":1,
    "seed":null
    },
"callback":{}
}
```

Example data `data.hdf5`

```python
data = {
        'iteration':[0,1,2],
        'parameters':[array([...]),array([...]),array([...])],
        'value': [1e-1,1e-2,1e-3]
    }
```

## Run
Under `build`, please run 
```sh
python main.py settings.json 
```
to run all model permutations, either in serial, (GNU) parallel, or with interdependent job arrays on an HPC cluster. 

## Plot
Plotting and post-processing can be performed, with plot and processing files, and with saving figures to an output directory.  

Any files stored as attribute-iterable format, i.e) `.hdf5` or `.json` files may be imported and processed within the `pandas` and `matplotlib` API frameworks. 

Under `build`, please run
```sh
python processor.py <path/to/data.hdf5> <path/to/plot.json> <path/to/process.json> <path/to/plots>
```
An example plot for optimization convergence is
<!-- <object data="https://github.com/mduschenes/tensor/blob/master/plot.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/mduschenes/tensor/blob/master/plot.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/mduschenes/tensor/blob/master/plot.pdf">Download PDF</a>.</p>
    </embed>
</object> -->
![alt text](https://github.com/mduschenes/tensor/blob/dev/.data/plot.jpg?raw=true)

## Test
Under `test`, to run unit tests (with pytest API), please run
```sh
. pytest.sh
```

## Classes
The hierarchy of inheritance of classes is as follows

`Channel` &larr; `Unitary` &larr; `Hamiltonian` &larr; `Operators` &larr; `Object` &larr; `System` &larr; `Dictionary`

`Operators` &larr; `Object` &larr; `System` &larr; `Dictionary`

`Operator` &larr; `Object` &larr; `System` &larr; `Dictionary`

`State,Noise,Pauli,Gate` &larr; `Object` &larr; `System` &larr; `Dictionary`

`Label` &larr; `Operator` &larr; `Object` &larr; `System` &larr; `Dictionary`

`Parameters` &larr; `System` &larr; `Dictionary`

`Parameter` &larr; `System` &larr; `Dictionary`

`Space,Time,Lattice` &larr; `System` &larr; `Dictionary`

`Objective,Callback` &larr; `Function` &larr; `System` &larr; `Dictionary`

`Metric` &larr; `System` &larr; `Dictionary`

`Optimizer` &larr; `Optimization` &larr; `System` &larr; `Dictionary`

## Settings
Settings files `settings.json` are used to configure model, optimization, and job settings, and to define all permutations of settings intended to be run. The files should have the following fields:
- `cls` : paths to classes for model (`model`,`label`,`state`,`callback`,...)
- `boolean` : booleans for training, optimization, and saving, loading of models (`train`,`load`,`dump`)
- `seed` : random seed settings (`seed`,`size`,...)
- `permutations` : permutations of model settings
- `model` : model instance settings (`data`,`N`,`M`,`D`,...)
- `system` : system settings (`dtype`,`device`,`backend`,`path`,`logger`,...)
- `optimize` : optimization settings (`iterations`,`optimizer`,`metric`,`track`,...)
- `label` : label settings for optimization (`operator`,`site`,`parameters`,...)
- `state` : state settings for optimization (`operator`,`site`,`parameters`,...)
- `callback` : callback settings for optimization (`args`,`kwargs`,...)
- `process` : postprocessing settings (`load`,`dump`,`plot`,`instance`,`texify`,...)
- `plot` : plotting settings (`fig`,`ax`,`style`,...)
- `jobs` : job settings for multiple model instances (`args`,`paths`,`patterns`,...)

If a single model instance is to be run with `src/quantum.py`, then `cls`,`model`,`system` fields are required.

If a single model instance is to be optimized with `src/train.py`, then `optimize`,`label`,`state`,`callback` fields are also required.

If multiple model instances are to be run with `src/run.py`, then the `jobs` fields are also required. 

If postprocessing is to be run with `src/process.py`, then `process.json` and `plot.json` are required, or loaded settings from the `process`,`plot` fields of `settings.json`.


## File Formats
All settings and data are generally stored as key-value pairs, allowing for simplified loading and dumping as nested dictionaries. 

Settings are generally loaded as `.json` format. Settings are formatted as nested keyword arguments, to initialize classes. 

Data are generally saved as `.hdf5` format. Data are formatted as attribute-iterable datasets, corresponding to data at optimization iterations.

Optimization checkpoints are generally saved as `.hdf5.ckpt` format. Any checkpoint files present in cwd will be resumed by the optimizer at the last checkpointed iteration.

Log files are generally saved as `.log` format. All logging across classes is configured with `logging.conf` files, to print `stdout` and `stderr` to the terminal and a log file.


