# Simulator for Quantum Systems
A pure JAX based quantum circuit simulator library, including automatic differentiation of arbitrary quantum circuits, or Trotterized Hamiltonians, for use in quantum state preparation, unitary compilation, or optimization of arbitrary quantum channels with noise. 

Pre-processing, hyperparameter searches, and inter-dependent, parallelized job submission scripts, and post-processing with statistical analysis, and plotting are also included in the library.

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

## Install
After cloning the repository, under `setup`, please run 
```sh
. setup.sh <env>
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
Settings are generally loaded as `.json` format. 

Data is generally saved as `.hdf5` format.

All settings and data are generally stored as key-value pairs, allowing for simplified loading and dumping in library as nested dictionaries
i.e) `settings = {
        'cls': {'model':Channel,'label':Label},
        'model': {'N':4,'D':2,M':10},
        'optimize': {'optimizer':'cg','iterations':[0,100]}
    }`
i.e) `data = {
        'iteration':[0,1,2],
        'parameters':[array([...]),array([...]),array([...])],
        'value': [1e-1,1e-1,1e-3]
    }`

## Run
Under `build`, please run 
```sh
python main.py settings.json 
```
to run all model permutations, either in serial, (GNU) parallel, or with interdependent job arrays on an HPC cluster. 

## Optimization
Optimize models with `src/optimize.py`, yielding `data.hdf5` files with data attributes and datasets for data at optimization iterations. 

## Plot
Plotting and post-processing can be performed, with plot and processing files, and with saving figures to an output directory. Any files stored as attribute-list format i.e) .hdf5,.json may be imported and processed within the `pandas` and `matplotlib` API frameworks. Under `build`, please run
```sh
python processor.py <path/to/data.hdf5> <path/to/plot.json> <path/to/process.json> <path/to/plots>
```
An example plot for optimization convergence is
<!-- <object data="https://github.com/mduschenes/simulation/blob/master/plot.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/mduschenes/simulation/blob/master/plot.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/mduschenes/simulation/blob/master/plot.pdf">Download PDF</a>.</p>
    </embed>
</object> -->
![alt text](https://github.com/mduschenes/simulation/blob/master/setup/plot.jpg?raw=true)

## Test
Under `test`, to run unit tests (with pytest API), please run
```sh
. pytest.sh
```