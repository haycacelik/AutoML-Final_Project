### you can use the following command to create a new conda environment
conda create -n automl-text-env python=3.10
conda activate automl-text-env

### install the dependencies
pip install -e .

### to run the code use the following command
python -m run

it will result in a results/dataset={dataset}/seed={seed}/optuna_{number} folder, in the corresponding optuna folder you can find all trials, their most current model, and their best model. Once the training is done it will output:
In the optuna folder, a yaml file that contains the best trial, which includes the parameters and the path to the best model
In the parent folder, a yaml file that contains the training time, validation error and test error if labels are available
In the parent folder, a numpy file that contains the predictions on the test set if labels are not avaliable

### There is a manual_run.py file that can be used to train the best trial on the full dataset, it will use the best configuration found by the model and train the model on the full dataset, it will output the validation error and test predictions if labels are not available. It requires the path to the best trial yaml file. This is what we used after we found the best configuration. You can run it similar to as follows:
python -m automl.automl_methods.hpo.manual_run --config_path /work/dlclarge2/celikh-nr1-ayca/automl-exam-ss25-text-dasauto/results/dataset=yelp/seed=42/optuna_4/best_trial.yaml
(please adjust the path to the best_trial.yaml file according to your setup)

# EXTRA 
### Previously we have also implemented bohb, if you also want to check out our bohb code, you need to install the following
pip install ray[tune]
### and then run the code using the following command
python -m automl.automl_methods.hpo.bohb

## CAUTION
### We have tried to run the code on a brand new environment following our instructions, and it worked without any issues, but just in case we have also created a requirements txt file from our main environment, so if you have any issues with the dependencies, you can use the requirements.txt file to install the dependencies using pip.
### I would like to add that when I was preparing this file I made some tiny changes to the code, I tested it with dummy data and it works, it should not break it but if you have any issues, please let me know, and I will try to help you out.
### Also I have run it at the same time from 2 computers at once for the first time and although we have seeds for both the tpe sampler and for numpy, it does not result in the exact same results. But it is always close.

### IMPORTANT ###
# For some reason we could not understand our precision files score horribly. We have added all_trials.yaml file to show that the code works.
