
This project implements an AutoML system for text classification, developed as part of the AutoML course (SS25) at the University of Freiburg.
The initial template was provided by the course instructors, and we extended it into a working AutoML pipeline as part of the final exam project.
This project consists of a solution inspired by evolutionary algorithms to speed up Hyperband while using Bayesian Optimization with a layer system to improve performance.

## Motivation
Our hyperparameter search space contains both continuous and categorical variables so the best fitting Bayesian Optimization approach would be to use a Gaussian Process with a suitable kernel that can handle both types of variables. Also since training transformers can be very expensive we wanted go with some multi fidelity approaches. Putting all of these criteria together we decided to try BOHB out.

BOHB code was done with RayTune and it is working. It can be run by
python -m automl.automl_methods.hpo.bohb

BOHB ended up being too time intensive and it took a long time to leverage TPEs abilities so with a low budget it was just working as hyperband. So we came up with the idea for the project. 

## Project Idea
There are two parts to this idea:

### 1- Ensemble Hyperband

This idea came from evolutionary algorithms. We decided to evaluate the n amount of configs at a budget point and only allow the best n/2*k configurations (k being the downsampling rate for hyperband) for the next budget point. So the next step includes; one half the best configurations from the last layer and one half the newly compiled ones (children).

- Below is a visual demonstrating the layer parameter selection
<img width="827" height="490" alt="ensemble hb drawio (1)" src="https://github.com/user-attachments/assets/38bd8b8d-ec5e-4b64-b529-3adbbc1871ba" />



- Below is a visual demonstrating the time gain compared to classical hyperband. This results in a %33 percent run-time improvement.
<img width="701" height="362" alt="epoch_gains drawio" src="https://github.com/user-attachments/assets/d8a14041-5487-4355-a83c-58c57dac5e04" />

### 2- Leveraging TPE Sooner

For each layer there is a TPE and until the TPE of that layer matures it uses the TPE of the previous layer. The new configurations use the TPE if it is mature, if not they use random sampling. The usage of using the TPE of another layer is only limited to the previous layer because we dont want it to converge to a local minima. 

- Below, it can be seen that the parameters at the second layer, which was the only layer where TPE was active in this short experiment, consistently outperformed the randomly generated configs at other layers. This can be judged by the configs that made it to the last layer being mostly blue (from the second layers, compiled with TPE). This has been tested 4 times andd each time it resulted similarly. 
<img width="3600" height="2100" alt="final_plot_seaborn" src="https://github.com/user-attachments/assets/6996bcad-d5bd-4a78-94ad-60d7750473a1" />

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

This project is more of a proof of concept demonstrating feasibility, it is not a flexible implementation due to time constraints. Also this project has been tested with a limit of 24 hours run-time.
