# Sentencing Under Fairness Constraints: Finding a Fair Policy with Offline Contextual Bandit

## Final Report

For details about our experiments on the NODA dataset, please refer to [here](https://github.com/CoraJung/bandit-under-fairness-constraints/blob/master/%5BFinal%20Paper%5D%20Sentencing%20Under%20Fairness%20Constraints.pdf).

## Disclaimer

Reproduction can only be done using ProPublica (located in `dataset/propublica`) as NODA cannot be published for confidential reasons. Our code for data processing on NODA cannot be shared for the same reason.

## Installation

This code has been tested on Ubuntu.

First, install Python 3.x, Numpy (1.16+), and Cython (0.29+).

The remaining dependencies can be installed by executing the following command from the Python directory of : 

	pip install -r requirements.txt

## Usage

The experiments presented in the report can be executed by running the following line in the command:

     python -m experiments.bandit.recidivism recidivism_all --n_trials 5 --definition GroupFairness --e 0.1 \
     --d 0.05 --ci_type ttest --n_iters 2000 --n_jobs 15  --r_train_v_test 0.4 --r_cand_v_safe 0.4 --rwd_recid -1.0 \
     --rwd_nonrecid 1.0 --use_score_text --data_pct 1.0 --add_info all
     
* `recidivism_all`: folder name to save the results
* `--add_info`: covariate sets to use ('none', 'all', 'judge', 'screen_ada', 'trial_ada') - only applicable to NODA dataset

## Reading Experiment Outputs

Model output comes in .h5 file format. `reading robinhood output.ipynb` in `ipynb` folder will transform the result to a well-formated dataframe. 

## Acknowledgement

Original code comes from https://github.com/sgiguere/RobinHood-NeurIPS-2019, and it has been modified to be applicable for our experiments on NODA.

## License

Code for RobinHood is released under the MIT license, with the exception of the code for FairMachineLearning (located in `Python/baselines/fairml.py`), which are released under their licences assigned by their respective authors.
