# Sentencing Under Fairness Constraints: Finding a Fair Policy with Offline Contextual Bandit

# Final Report

Final report including experiment results can be found [here](https://docs.google.com/document/d/17uofC2CaA0BKe8DIyfa38uqWqWOK_pCggPaW2asECJI/edit#).

# Installation

This code has been tested on Ubuntu.

First, install Python 3.x, Numpy (1.16+), and Cython (0.29+).

The remaining dependencies can be installed by executing the following command from the Python directory of : 

	pip install -r requirements.txt

# Usage

The experiments presented in the report can be executed by running the following line in the command:

     python -m experiments.bandit.recidivism recidivism_all --n_trials 5 --definition GroupFairness --e 0.1 \
     --d 0.05 --ci_type ttest --n_iters 2000 --n_jobs 15  --r_train_v_test 0.4 --r_cand_v_safe 0.4 --rwd_recid -1.0 \
     --rwd_nonrecid 1.0 --use_score_text --data_pct 1.0 --add_info all
     
You can expeirment with different covariate sets by changing the `--add_info` parameter ('none', 'all', 'judge', 'screen_ada', 'trial_ada').

# Acknowledgement

Original code immplementation comes from https://github.com/sgiguere/RobinHood-NeurIPS-2019.

# License

Code for RobinHood is released under the MIT license, with the exception of the code for FairMachineLearning (located in `Python/baselines/fairml.py`), which are released under their licences assigned by their respective authors.
