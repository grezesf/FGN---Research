# Finite Gaussian Neurons for Adversarial Networks

## Proposal by Felix Grezes

### Directory Organization

```
----\
    |
    |---\Finite_Gaussian_Network_lib
        # functional library to run FGNs
        |
        |---\fgn_helper_lib
            # useful but unrelated functions
        |
        |---\tests
            # tests for the library functions
    |
    |---\Notebooks
        # notebooks to plot results, visualize data, etc...
    |
    |---\Experiments
        # contains scripts to run experiments and the results
    |
    |---\dev
        # development work
    |
    |---\old
        # old work
```

#### [Finite_Gaussian_Network_lib](./Finite_Gaussian_Network_lib)
A collection of functions related to Finite Gaussian Networks.
* Matlab style: one function per file. Open a file to see it's definition, parameters, etc...
* the [fgn_helper_lib](./Finite_Gaussian_Network_lib/fgn_helper_lib) directory contains randoms useful functions, but that don't directly relate to FGNs.
* the [tests](./Finite_Gaussian_Network_lib/tests) directory contains tests for functions in the library

#### [Notebooks](./Notebooks)
A collection of Jupyter Notebooks used for data visualization, results plotting, experiments analysis, etc...
Loosely follows the narrative of the proposal.

#### [Experiments](./Experiments)
A collection of tiny scripts that run experiments, and folders containing the results.
The scripts should be tiny, only creating the folders, setting the parameters and calling the library function.
Convention: scripts should create timestamped folders for the results each run.
`mnist_fgn_train.py` should create `/res-mnist_fgn_train-time:stamp'

#### [Dev](./dev)
collection of notebooks used to develop the FGN library functions, the scripts.
