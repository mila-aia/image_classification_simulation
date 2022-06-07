# image_classification_simulation


Simulation project to classify user images into a catalog of existing classes.


* Free software: MIT license



## Instructions to setup the project

### Install the dependencies:
(remember to activate the virtual env if you want to use one)
Add new dependencies (if needed) to setup.py.

    pip install -e .

### Add git:

    git init

### Setup pre-commit hooks:
These hooks will:
* validate flake8 before any commit
* check that jupyter notebook outputs have been stripped

    cd .git/hooks/ && ln -s ../../config/hooks/pre-commit . && cd -

### Commit the code

    git add .
    git commit -m 'first commit'

### Link github to your local repository
Go on github and follow the instructions to create a new project.
When done, do not add any file, and follow the instructions to
link your local git to the remote project, which should look like this:
(PS: these instructions are reported here for your convenience.
We suggest to also look at the GitHub project page for more up-to-date info)

    git remote add origin git@github.com:alzaia/image_classification_simulation.git
    git branch -M main
    git push -u origin main

### Setup Continuous Integration

Continuous integration will run the following:
- Unit tests under `tests`.
- End-to-end test under `exmaples/local`.
- `flake8` to check the code syntax.
- Checks on documentation presence and format (using `sphinx`).

We support the following Continuous Integration providers.
Check the following instructions for more details.

#### GitHub Actions

Github actions are already configured in `.github/workflows/tests.yml`.
Github actions are already enabled by default when using Github, so, when
pushing to github, they will be executed automatically for pull requests to
`main` and to `develop`.

#### Travis

Travis is already configured in (`.travis.yml`).

To enable it server-side, just go to https://travis-ci.com/account/repositories and click
` Manage repositories on GitHub`. Give the permission to run on the git repository you just created.

Note, the link for public project may be https://travis-ci.org/account/repositories .

#### Azure

Azure Continuous Integration is already configured in (`.azure_pipeline.yml`).

To enable it server-side, just in azure and select `.azure_pipeline.yml` as the 
configuration one for Continuous Integration.

## Running the code

### Run the tests
Just run (from the root folder):

    pytest

### Run the code/examples.
Note that the code should already compile at this point.

Running examples can be found under the `examples` folder.

In particular, you will find examples for:
* local machine (e.g., your laptop).
* a slurm cluster.

For both these cases, there is the possibility to run with or without Orion.
(Orion is a hyper-parameter search tool - see https://github.com/Epistimio/orion -
that is already configured in this project)

#### Run locally

For example, to run on your local machine without Orion:

    cd examples/local
    sh run.sh

This will run a simple MLP on a simple toy task: sum 5 float numbers.
You should see an almost perfect loss of 0 after a few epochs.

Note you have two new folders now:
* output: contains the models and a summary of the results.
* mlruns: produced by mlflow, contains all the data for visualization.
You can run mlflow from this folder (`examples/local`) by running
`mlflow ui`.

#### Run on a remote cluster (with Slurm)

First, bring you project on the cluster (assuming you didn't create your
project directly there). To do so, simply login on the cluster and git
clone your project:

    git clone git@github.com:alzaia/image_classification_simulation.git

Then activate your virtual env, and install the dependencies:

    cd image_classification_simulation
    pip install -e .

To run with Slurm, just:

    cd examples/slurm
    sh run.sh

Check the log to see that you got an almost perfect loss (i.e., 0).

#### Run with Orion on the Slurm cluster

This example will run orion for 2 trials (see the orion config file).
To do so, go into `examples/slurm_orion`.
Here you can find the orion config file (`orion_config.yaml`), as well as the config
file (`config.yaml`) for your project (that contains the hyper-parameters).

In general, you will want to run Orion in parallel over N slurm jobs.
To do so, simply run `sh run.sh` N times.

When Orion has completed the trials, you will find the orion db file and the
mlruns folder (i.e., the folder containing the mlflow results).

You will also find the output of your experiments in `orion_working_dir`, which
will contain a folder for every trial.
Inside these folders, you can find the models (the best one and the last one), the config file with
the hyper-parameters for this trial, and the log file.

You can check orion status with the following commands:
(to be run from `examples/slurm_orion`)

    export ORION_DB_ADDRESS='orion_db.pkl'
    export ORION_DB_TYPE='pickleddb'
    orion status
    orion info --name my_exp

### Building docs:

To automatically generate docs for your project, cd to the `docs` folder then run:

    make html

To view the docs locally, open `docs/_build/html/index.html` in your browser.


## YOUR PROJECT README:

* __TODO__
