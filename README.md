# AutoML for Text Classification

This project implements an AutoML system for text classification, developed as part of the AutoML course (SS25) at the University of Freiburg.
The initial template was provided by the course instructors, and we extended it into a working AutoML pipeline as part of the final exam project.
This project consists of a solution inspired by evolutionary algorithms to speed up Hyperband while using Bayesian Optimization with a layer system to improve performance.

## Motivation
Our hyperparameter search space contains both continuous and categorical variables so the best fitting Bayesian Optimization approach would be to use a Gaussian Process with a suitable kernel that can handle both types of variables. Also since training transformers can be very expensive we wanted go with some multi fidelity approaches. Putting all of these criteria together to try BOHB out.

BOHB code was done with RayTune and it is working. It can be run by
python -m automl.automl_methods.hpo.bohb

BOHB ended up being too time intensive and it took a long time to leverage TPEs abilities so with a low budget it was just working as hyperband. So we came up with the idea for the project. 

## Project Idea
There are two parts to this idea:

### 1- Ensemble Hyperband

This idea came from evolutionary algorithms. In evolutionary algorithms at every step, a population of models are evaluated and the best performing models are selected for the next iteration. Then in the next step, new configurations (children) are generated and they are evaluated all together. So we decided to evaluate the n amount of models at budget point and only allow the best n/2*k (k being the downsampling rate for hyperband) models for the next budget point along with the newly created n/2*k models. So the next step includes half the best models from the last layer and half the newly created ones.

### 2- Leveraging TPE Sooner

For each layer there is a TPE and until that layers TPE matures it uses the preivous layers TPE. It is only limited to the previous layer because we dont want it to converge to a local minima.

## Project Constraints

In this project we had a 24 hour run time constraint. We decided to use transformers and fine tune them. We started with the idea of a nested NAS and HPO model but after noticing how time intensive fine tuning transformer based morels are we switched to an only HPO pipeline. 

## Hyperparameters
In this project we have done hyperparameter optimization for the following hyperparameters:
Learning rate: Log-uniform, [1e-6, 1e-2]
Token length: Categorical, {64, 128, 256}
Weight decay: Log-uniform, [1e-6, 0.1]
Layers to finetune: Integer, [0–6]
Dropout rate: [0.0, 0.5]

There are some NAS related hyperparameters for the classification head that can be added with minor changes in the code. These are:
hidden_dim = Categorical, [64, 128, 256]
activation = Categorical, ["ReLU", "GELU", "LeakyReLU"]
hidden_layer = Integer, [1, 4]
use_layer_norm = Categorical, [True, False]

## The parts provided by the university

They provided the following:

* `run.py`: Trains an AutoML system on a dataset and generates predictions for the test split.
* `automl`: A Python package containing the AutoML system implementation.

They have been almost completely changed but the backbone still remains.

## Installation

To install the repository, first create an environment of your choice and activate it. 

For example, using `venv`:

You can change the python version here to the version you prefer.

**Virtual Environment**

```bash
python3 -m venv automl-text-env
source automl-text-env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-text-env python=3.10
conda activate automl-text-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

*NOTE*: this is an editable install which allows you to edit the package code without requiring re-installations.

You can test that the installation was successful by running the following command:

```bash
python -c "import automl; print(automl.__file__)"
# this should print the full path to your cloned install of this repo
```





## Notice

This project is more of a proof of concept demonstrating feasibility, it is not a flexible implementation due to time constraints.
