# Sentencing Under Fairness Constraints: \ Finding a Fair Policy with Offline Contextual Bandit

Original code immplementation from https://github.com/sgiguere/RobinHood-NeurIPS-2019

# Installation

This code has been tested on Ubuntu.

First, install Python 3.x, Numpy (1.16+), and Cython (0.29+).

The remaining dependencies can be installed by executing the following command from the Python directory of : 

	pip install -r requirements.txt

# Usage

The experiments featured in the paper can be executed by running the provided batch file from the Python directory, as follows:

     ./experiments/scripts/bandit_experiments.bat

# License

Code for RobinHood is released under the MIT license, with the exception of the code for FairMachineLearning (located in `Python/baselines/fairml.py`), which are released under their licences assigned by their respective authors.
