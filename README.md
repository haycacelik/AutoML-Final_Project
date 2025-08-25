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

We make no restrictions on the python library or version you use, but we recommend using python 3.10 or higher.

## Code

We provide the following:

* `run.py`: A script that trains an _AutoML-System_ on the training split of a given dataset and 
  then generates predictions for the test split, saving those predictions to a file. 
  For the training datasets, the test splits will contain the ground truth labels, but for the 
  test dataset which we provide later the labels of the test split will not be available. 
  You will be expected to generate these labels yourself and submit them to us through GitHub classrooms.

* `automl`: This is a python package that will be installed above and contain your source code for whatever
  system you would like to build. We have provided a dummy `AutoML` class to serve as an example.

*You are completely free to modify, install new libraries, make changes and in general do whatever you want with the code.* 
The *only requirement* for the exam will be that you can generate predictions for the test splits of our datasets in a `.npy` file that we can then use to give you a test score through GitHub classrooms.


## Data

We selected 4 different text-classification datasets which you can use to develop your AutoML system and we will provide you with 
a test dataset to evaluate your system at a later point in time. 

The dataset can be automatically or programatically downloaded and extracted from: [https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-text/text-phase1.zip](https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-text/text-phase1.zip)

The downloaded datasets should have the following structure:
```bash
<target-folder>
├── ag_news
│   ├── train.csv
│   ├── test.csv
├── amazon
│   ├── train.csv
│   ├── test.csv
├── imdb
│   ├── train.csv
│   ├── test.csv
├── dbpedia
│   ├── train.csv
│   ├── test.csv
```


## Running an initial test

After having downloaded and extracted the data at a suitable location, this is the parent data directory. \\
To run a quick test:

```bash
python run.py \
  --data-path <path-to-data-parent> \
  --dataset amazon \
  --epochs 1 \
  --data-fraction 0.2
```
*TIP*: play with the batch size and different approaches for an epoch (or few mini-batches) to estimate compute requirements given your hardware availability.

You are free to modify these files and command line arguments as you see fit.