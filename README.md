# AutoML for Text Classification

This project implements an AutoML system for text classification, developed as part of the AutoML course (SS25) at the University of Freiburg.
The initial template was provided by the course instructors, and I extended it into a working AutoML pipeline as part of the final exam project.
This project consists of a creative solution to budget extensive AutoML methods while maintaining high performance.

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

## The code provided by the university

They provided the following:

* `run.py`: A script that trains an _AutoML-System_ on the training split of a given dataset and 
  then generates predictions for the test split, saving those predictions to a file. 
  For the training datasets, the test splits will contain the ground truth labels, but for the 
  test dataset which we provide later the labels of the test split will not be available. 

* `automl`: This is a python package that will be installed above and contain your source code for whatever
  system you would like to build. We have provided a dummy `AutoML` class to serve as an example.

They have been almost completely changed but the backbone still remains.

## Project Constraints

In this project we had a 24 hour time constraint. Since we decided to use only transformers and fine tune some models it was very time consuming. We started with the idea of a nested NAS and HPO model but after noticing how time intensive it was we switched to an only HPO pipeline. 

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

## How we got here
Since we have a mixed Hyperparameter search space that consists of both numerical and categorical values the best fitting Bayesian Optimization approach would be to use a Gaussian Process with a suitable kernel that can handle both types of variables. Also since training trasnformers can be very expensive we wanted go with some multi fidelity approaches. Putting all of these criteri together and with the fact that our lecturer was the creator of BOHB (Bayesian Optimization and Hyperband) we decided to try it out.

BOHB code was done with RayTune and it is working. It can be run by
python -m automl.automl_methods.hpo.bohb

BOHB ended up being too time intensive and it takes a few runs of succesive halving to leverage TPEs abilities so with out low bodget it was just working as hyperband. So we came up with the ide for the project. 

## Project Idea

THe probelm we had with bohb was that it took a long time to leverage TPEs abilities. Also in general multi fidelity approaches also took more time then we anticipated.

There are two parts of this idea so lets start with the ensemble hyperband part

## 1- Ensemble Hyperband

This idea came from evolutionary algorithms. In evolutionary algorithms at every step a population of models is evaluated and the best performing models are selected for the next iteration. Then in the next step new configurations (children) are generated and they are evaluated all together. So we though why not do this with succesive halving. 

## 2- leveraging TPE sooner

For each layer there is a TPE and untill the layers tpe matures it uses the preivous layers TPE. It is only limited to the previous layer because we dont want it to converge to a lcal minima.

## Notice

This project is more of an idea, it is not a felixble implementation due to time constraints. I will keep wokring on it further with tabular data for faster experiments. The idea is not explained very clearly because I believe its promising and I would like to prepare solid outputs and share it with my instructures before I make it public.