# AutoML Exam - SS25 (Text Data)

This repo serves as a template for the exam assignment of the AutoML SS25 course
at the university of Freiburg.

The aim of this repo is to provide a minimal installable template to help you get up and running.

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

### Meta-data for datasets:

The following table will provide you an overview of their characteristics and also a reference value for the test accuracy.
*NOTE*: These scores were obtained through a rather simple HPO on a crudely constructed search space, for an undisclosed HPO budget and compute resources.

| Dataset Name | Labels | Rows | Seq. Length: `min` | Seq. Length: `max` | Seq. Length: `mean` | Seq. Length: `median` | Reference Accuracy |
| --- | --- |  --- |  --- |  --- | --- | --- | --- |
| amazon | 3 | 24985 | 4 | 15521 | 512 | 230 | 81.799% |
| imdb | 2 | 25000 | 52 | 13584 | 1300 | 962 | 86.993% |
| ag_news | 4 | 120000 | 99 | 1012 | 235 | 231 | 90.265% |
| dbpedia | 14 | 560000 | 11 | 13573 | 300 | 301 | 97.882% |
| *final\_exam\_dataset* | TBA | TBA | TBA | TBA | TBA | TBA | TBA |

*NOTE*: sequence length calculated at the raw character level

We will add the test dataset later in the final Github Classroom template code that will be released.
 <!-- by pushing its class definition to the `datasets.py` file.  -->
The test dataset will be in the same format as the training datasets, but `test.csv` will only contain `nan`'s for labels.


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

<!-- ## Final submission

The final test predictions should be uploaded in a file `final_test_preds.npy`, with each line containing the predictions for the input in the exact order of `X_test` given.

Upload your poster as a PDF file named as `final_poster_text_<team-name>.pdf`, following the template given [here](https://docs.google.com/presentation/d/1T55GFGsoon9a4T_oUm4WXOhW8wMEQL3M/edit?usp=sharing&ouid=118357408080604124767&rtpof=true&sd=true). -->

## Tips

* If you need to add dependencies that you and your teammates are all on the same page, you can modify the
  `pyproject.toml` file and add the dependencies there. This will ensure that everyone has the same dependencies

* Please feel free to modify the `.gitignore` file to exclude files generated by your experiments, such as models,
  predictions, etc. Also, be friendly teammate and ignore your virtual environment and any additional folders/files
  created by your IDE.
