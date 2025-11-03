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
Module = load(settings.cls.module)
Model = load(settings.cls.model)
State = load(settings.cls.state)
Callback = load(settings.cls.callback)

# Get system settings
system = settings.system

# Initialize model classes (getting attributes common to previous model namespaces)
if Module is not None and Model is not None and State is not None:

	model = Model(**{**settings.model,**dict(system=system)})
	state = State(**{**namespace(State,model),**settings.state,**dict(system=system)})
	callback = Callback(**{**settings.callback,**dict(system=system)})

	module = Module(**{**settings.module,**namespace(Module,model),**dict(model=model,state=state,callback=callback,system=system)})

	module.init()

	model = module

elif Model is not None and State is not None:

	model = Model(**{**settings.model,**dict(system=system)})
	state = State(**{**namespace(State,model),**settings.state,**dict(system=system)})

	model.init(state=state)

elif Model is not None:

	model = Model(**{**settings.model,**dict(system=system)})

	model.init()

else:

	model = None

# Dump model
if model is not None:
	model.dump()
```

Example settings `settings.json`

```python
{
"cls":{
	"module":"src.quantum.Module",
	"model":"src.quantum.Operators",
	"state":"src.quantum.State",
	"callback":"src.quantum.Callback"
	},
"module":{
	"N":4,
	"M":4,
	"D":2,
	"d":1,
	"seed":123,
	"string":"module",
	"measure":{"string":"povm","operator":"tetrad","D":2,"seed":null,"architecture":"tensor","options":{}},
	"configuration":{},
	"options":{}
},
"model":{
	"data":{
		"unitary":{
			"operator":"haar","where":"||ij||","string":"unitary",
			"parameters":null,"variable":false,"seed":null
		},
		"noise":{
			"operator":["depolarize"],"where":"||i.j||","string":"noise",
			"parameters":1e-8,"variable":false
		}
	},
	"N":4,
	"M":1,
	"D":2,
	"d":1,
	"local":true,
	"tensor":true,
	"space":"spin",
	"time":"linear",
	"lattice":"square",
	"seed":null
	},
"state": {
	"operator":"haar",
	"where":null,
	"string":"psi",
	"parameters":null,
	"N":null,
	"D":2,
	"ndim":2,
	"local":null,
	"tensor":true,
	"architecture":null,
	"seed":null
	},
"system":{
	"dtype":"complex",
	"format":"array",
	"device":"cpu",
	"backend":null,
	"architecture":null,
	"base":null,
	"seed":null,
	"key":null,
	"instance":null,
	"cwd":"data",
	"path":"data.hdf5",
	"lock":true,
	"backup":true,
	"conf":"logging.conf",
	"logger":null,
	"cleanup":false,
	"verbose":"info"
	},
"callback":{
	"attributes":{
		"N":"N","M":"M","d":"d","D":"D",
		"key":"key","instance":"instance","timestamp":"timestamp",
		"seed":"seed","seeding":"seeding",
		"noise.parameters":"noise.parameters",
		"operator":"measure.operator",
		"S":"options.S",
		"scheme":"options.scheme",
		"layout":"configuration.options.layout",
		"periodic":"measure.options.periodic",

		"samples":"samples",

		"array":"measure.array",
		"state":"measure.state",

		"sample.array.linear":"measure.sample",
		"sample.array.log":"measure.sample",
		"sample.state.linear":"measure.sample",
		"sample.state.log":"measure.sample",
		"sample.array.process":"measure.sample",
		"sample.state.process":"measure.sample",
		"sample.array.information":"measure.sample",
		"sample.state.information":"measure.sample",

		"infidelity.quantum":"measure.infidelity_quantum",
		"infidelity.classical":"measure.infidelity_classical",
		"infidelity.pure":"measure.infidelity_pure",
		"norm.quantum":"measure.norm_quantum",
		"norm.classical":"measure.norm_classical",
		"norm.pure":"measure.norm_pure",

		"entanglement.quantum":"measure.entanglement_quantum",
		"entanglement.classical":"measure.entanglement_classical",
		"entanglement.renyi":"measure.entanglement_renyi",
		"entangling.quantum":"measure.entangling_quantum",
		"entangling.classical":"measure.entangling_classical",
		"entangling.renyi":"measure.entangling_renyi",

		"mutual.quantum":"measure.mutual_quantum",
		"mutual.measure":"measure.mutual_measure",
		"mutual.classical":"measure.mutual_classical",
		"mutual.renyi":"measure.mutual_renyi",
		"discord.quantum":"measure.discord_quantum",
		"discord.classical":"measure.discord_classical",
		"discord.renyi":"measure.discord_renyi",

		"spectrum.quantum":"measure.spectrum_quantum",
		"spectrum.classical":"measure.spectrum_classical",
		"rank.quantum":"measure.rank_quantum",
		"rank.classical":"measure.rank_classical"
	},
	"keywords": {
		"sample.array.linear":{"attribute":"array","function":"src.functions.func_histogram","settings":{"bins":1000,"scale":"linear","base":10,"range":[0,1]}},
		"sample.array.log":{"attribute":"array","function":"src.functions.func_histogram","settings":{"bins":1000,"scale":"log","base":10,"range":[1e-20,1e0]}},
		"sample.state.linear":{"attribute":"state","function":"src.functions.func_histogram","settings":{"bins":1000,"scale":"linear","base":10,"range":[0,1]}},
		"sample.state.log":{"attribute":"state","function":"src.functions.func_histogram","settings":{"bins":1000,"scale":"log","base":10,"range":[1e-20,1e0]}},
		"sample.array.process":{"attribute":"array","function":"src.functions.func_process","settings":{}},
		"sample.state.process":{"attribute":"state","function":"src.functions.func_process","settings":{}},
		"sample.array.information":{"attribute":"array","function":"src.functions.func_information","settings":{}},
		"sample.state.information":{"attribute":"state","function":"src.functions.func_information","settings":{}},
		"entanglement.quantum":{"where":0.5},"entanglement.classical":{"where":0.5},"entanglement.renyi":{"where":0.5},
		"entangling.quantum":{"where":0.5},"entangling.classical":{"where":0.5},"entangling.renyi":{"where":0.5},
		"mutual.quantum":{"where":0.5},"mutual.measure":{"where":0.5},"mutual.classical":{"where":0.5},"mutual.renyi":{"where":0.5},
		"discord.quantum":{"where":0.5},"discord.classical":{"where":0.5},"discord.renyi":{"where":0.5},
		"spectrum.quantum":{"where":0.5},"spectrum.classical":{"where":0.5},"rank.quantum":{"where":0.5},"rank.classical":{"where":0.5}
	},
	"options":{}
	}
}
```

Example data `data.hdf5`

```python
data = {
		'N': 4,
		'D': 2,
		'd': 1,
		'M': 10,
		'S' : None,
		'infidelity.classical': 1.1879386363489175e-14,
		'entanglement.quantum': 0.45686812044843,
		'mutual.measure': 0.11421606311097962
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
An example plot for operator entanglement scaling is
<!-- <object data="https://github.com/mduschenes/tensor/blob/master/plot.pdf" type="application/pdf" width="700px" height="700px">
	<embed src="https://github.com/mduschenes/tensor/blob/master/plot.pdf">
		<p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/mduschenes/tensor/blob/master/plot.pdf">Download PDF</a>.</p>
	</embed>
</object> -->
![alt text](https://github.com/mduschenes/simulation/blob/tensor/.data/plot.jpg?raw=true)

## Test
Under `test`, to run unit tests (with pytest API), please run
```sh
. pytest.sh
```

## Classes
The hierarchy of inheritance of classes is as follows

`Channel` &larr; `Unitary` &larr; `Hamiltonian` &larr; `Objects` &larr; `Operator` &larr; `Object` &larr; `System` &larr; `Dictionary`

`Operators` &larr; `Objects` &larr; `Operator` &larr; `Object` &larr; `System` &larr; `Dictionary`

`Objects` &larr; `Operator` &larr; `Object` &larr; `System` &larr; `Dictionary`

`Operator` &larr; `Object` &larr; `System` &larr; `Dictionary`

`State` &larr; `Amplitude,Probability` &larr; `System` &larr; `Dictionary`

`State,Noise,Pauli,Gate` &larr; `Object` &larr; `System` &larr; `Dictionary`

`Modules` &larr; `Objects` &larr; `Operator` &larr; `Object` &larr; `System` &larr; `Dictionary`

`Module` &larr; `Operator` &larr; `Object` &larr; `System` &larr; `Dictionary`

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
- `boolean` : booleans for calling, training, optimization, and saving, loading of models (`call`,`train`,`load`,`dump`)
- `seed` : random seed settings (`seed`,`size`,...)
- `permutations` : permutations of model settings
- `model` : model instance settings (`data`,`N`,`M`,`D`,...)
- `system` : system settings (`dtype`,`device`,`backend`,`path`,`logger`,...)
- `optimize` : optimization settings (`iterations`,`optimizer`,`metric`,`track`,...)
- `label` : label settings for optimization (`operator`,`where`,`parameters`,...)
- `state` : state settings for optimization (`operator`,`where`,`parameters`,...)
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


