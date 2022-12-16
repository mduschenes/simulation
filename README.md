### Simulator for Quantum Systems
A pure JAX based quantum circuit simulator library, including automatic differentiation of arbitrary quantum circuits, or Trotterized Hamiltonians, for use in quantum state preparation, unitary compilation, or optimization of arbitrary quantum channels with noise. Pre-processing, hyperparameter searches, and inter-dependent, parallelized job submission scripts, and post-processing with statistical analysis, and plotting are also included in the library.

# Install
After cloning the repository, under `setup`, please run 
```sh
	. setup.sh <env>
```
which installs a Python environment with name `env` with all necessary packages, including JAX.

# Setup
Under `build`, please modify the `settings.json` file, with all model parameters, hyperparameter combinations, job settings, and plot settings.

# Run
Under `build`, please run 
```sh
	python main.py settings.json 
```
to run all model configurations, either in serial, (GNU) parallel, or with interdependent job arrays on an HPC cluster.