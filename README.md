# Simulator for Quantum Systems
A pure JAX based quantum circuit simulator library, including automatic differentiation of arbitrary quantum circuits, or Trotterized Hamiltonians, for use in quantum state preparation, unitary compilation, or optimization of arbitrary quantum channels with noise. 

Pre-processing, hyperparameter searches, and inter-dependent, parallelized job submission scripts, and post-processing with statistical analysis, and plotting are also included in the library.

The hierarchy of inheritance of classes is as follows

`Unitary <- Hamiltonian <- Observable <- System <- Dictionary`

`Operator <- System <- Dictionary`

`State,Noise,Gate <- Object <- System <- Dictionary`

`Space,Time,Lattice <- System <- Dictionary`

`Objective,Callback <- Function <- System <- Dictionary`

`Metric <- System <- Dictionary`

`Optimizer <- Optimization <- System <- Dictionary`

## Install
After cloning the repository, under `setup`, please run 
```sh
. setup.sh <env>
```
which installs a Python environment with name `env` with all necessary packages, including JAX.

## Setup
Under `build`, please modify the `settings.json` file, with all model parameters, hyperparameter combinations, job settings, and plot settings. 

To configure jobs scripts, plot and processing settings, that follow the matplotlib API, please modify the files under `config`. 
- `job.slurm` : job script
- `logging.conf` : logger configuration
- `process.json` : data analysis settings 
- `plot.json` : plots axes and figures 
- `plot.mplstyle` : matplotlib style

## Run
Under `build`, please run 
```sh
python main.py settings.json 
```
to run all model configurations, either in serial, (GNU) parallel, or with interdependent job arrays on an HPC cluster.

## Plot
Plotting and post-processing can be performed, with plot and processing files, and with saving figures to an output directory. Under `build`, please run
```sh
python processor.py <path/to/data> <path/to/plot.json> <path/to/process.json> <path/to/plots>
```
An example plot for optimization convergence is
<!-- <object data="https://github.com/mduschenes/simulation/blob/master/plot.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/mduschenes/simulation/blob/master/plot.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/mduschenes/simulation/blob/master/plot.pdf">Download PDF</a>.</p>
    </embed>
</object> -->
![alt text](https://github.com/mduschenes/simulation/blob/master/build/config/plot.jpg?raw=true)

## Test
Under `test`, to run unit tests (with pytest API), please run
```sh
. pytest.sh
```