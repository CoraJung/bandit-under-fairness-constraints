import numpy as np
import pandas as pd
import os.path

import joblib
from datasets.dataset import BanditDataset
from utils import keyboard



DATA_PATH = "/misc/vlgscratch5/PichenyGroup/s2i-common/bandit-under-fairness-constraints/Python/datasets/noda/noda_trial.csv"

# Need to determine X
CAT_TO_KEEP = ['DISP_CODE_SENT', 'SENTENCE_TYPE', 'HABITUAL_OFFENDER_FLAG_SENT', 'SENTENCE_LOCATION', \
                 'CUSTODY_CODE', 'CRIMINAL_FLAG', 'SEX_DFDN', 'RACE_DFDN', 'JUVENILE_INVOLVED_FLAG', 'RACE_TRIAL_ADA', \
                'SEX_TRIAL_ADA', 'PARTY_TRIAL_ADA', 'SEX_SCREEN_ADA', 'PARTY_SCREEN_ADA', 'BOND_MADE_FLAG', \
                'CHARGE_CAT', 'BOND_TYPE_CODE', 'SEX_JUDGE', 'RACE_JUDGE','PARTY_JUDGE', 'RACE_SCREEN_ADA' ]
NUM_TO_KEEP = ['FINE_AMOUNT', 'NBR_OF_DFDN', 'AGE_DFDN', 'AGE_DFDN_ISNA', 'CASE_CLASS_ISNA', \
                'BOND_MADE_AMOUNT', 'BOND_SET_AMOUNT', 'CHARGE_CLASS', 'AGE_JUDGE', 'DOB_SCREEN_ADA', 'DOB_TRIAL_ADA', \
                'AGE_JUDGE_ISNA', 'AGE_SCREEN_ADA_ISNA', 'AGE_TRIAL_ADA_ISNA', 'CHARGE_CLASS_ISNA']

LABELS_TO_KEEP = CAT_TO_KEEP + NUM_TO_KEEP

def load(r_train=0.4, r_candidate=0.2, T0='W', T1='B', seed=None, include_T=False, include_intercept=True, use_pct=1.0, use_score_text=False, rwd_recid=-1.0, rwd_nonrecid=1.0, use_cached_gps=False):
	random = np.random.RandomState(seed)
	scores = pd.read_pickle(DATA_PATH)
	# Generate the full dataset
	#S = scores[scores['RACE_DFDN'].isin([T0,T1])].copy()
	S = scores[np.logical_or(scores['RACE_DFDN']==T0, scores['RACE_DFDN']==T1)].copy() 
	#A is the action, A = df['ACTION']
	A = S['ACTION'].astype(int)
	A = A.values 

	n_actions = len(np.unique(A))
	
	R = np.sign(S['RECIDIVIZE_FLAG'].values-0.5) #orig: two_year_recid
	R = (R==-1)*rwd_nonrecid + (R==1)*rwd_recid

	S = S[LABELS_TO_KEEP]

	# Turn all cat covariates to one-hot encoding -> do race separately
	for col in CAT_TO_KEEP:
		if col != 'RACE_DFDN':
			S = with_dummies(S, col)

	T = 1 * (S['RACE_DFDN']==T1).values #One-hot encoding for race T1
	del S['RACE_DFDN']
	L = np.array(S.columns, dtype=str)
	S = S.values
	
	if include_intercept:
		S = np.hstack((S, np.ones((len(S),1)))) 
		L = np.hstack((L, ['intercept'])) 
	if include_T:
		S = np.hstack((T[:,None], S))
		L = np.hstack((['type'], L))	


	n_keep = int(np.ceil(len(S) * use_pct))
	I = np.arange(len(S))
	random.shuffle(I)
	I = I[:n_keep]
	S = S[I]
	A = A[I]
	R = R[I]
	T = T[I]	
	
	# Compute split sizes
	n_samples   = len(S)
	n_train     = int(r_train*n_samples)
	n_test      = n_samples - n_train
	n_candidate = int(r_candidate*n_train)
	n_safety    = n_train - n_candidate
	max_reward = max(rwd_recid, rwd_nonrecid)
	min_reward = min(rwd_recid, rwd_nonrecid)

	# Load cached GPs if requested
	# if use_cached_gps:
	# 	if use_score_text:
	# 		proba_gp_path = os.path.join(BASE_URL, '%s_score_text_proba_gp.joblib' % dset_type)
	# 		rwd_gp_path   = os.path.join(BASE_URL, '%s_score_text_rwd_gp.joblib' % dset_type)
	# 	else:
	# 		proba_gp_path = os.path.join(BASE_URL, '%s_decile_score_proba_gp.joblib' % dset_type)
	# 		rwd_gp_path   = os.path.join(BASE_URL, '%s_decile_score_rwd_gp.joblib' % dset_type)
	# 	proba_gp  = joblib.load(proba_gp_path)
	# 	return_gp = joblib.load(rwd_gp_path)
	# 	X = np.hstack((S,T[:,None]))
	# 	Ps = proba_gp.predict_proba(X)
	# 	P = np.array([ Ps[i,a] for i,a in enumerate(A) ])
	# else: P = None

	P = None

	dataset = BanditDataset(S, A, R, n_actions, n_candidate, n_safety, n_test, min_reward, max_reward, seed=seed, P=P, T=T)
	dataset.X_labels = L
	dataset.T0_label = T0
	dataset.T1_label = T1

	# Store the GPs that were loaded, if using cached GPs
	# if use_cached_gps:
	# 	dataset._proba_gp  = proba_gp
	# 	dataset._return_gp = return_gp

	return dataset

def with_dummies(dataset, column, label=None, keep_orig=False, zero_index=True):
	dataset = dataset.copy()
	assert column in dataset.columns, 'with_dummies(): column %r not found in dataset.'%column
	if label is None:
		label = column
	dummies = pd.get_dummies(dataset[column], prefix=label, prefix_sep=':')
	for i,col in enumerate(dummies.columns):
		col_name = col
		if zero_index and (len(dummies.columns) > 1):
			if i > 0:
				name, val = col.split(':',1)
				col_name = ':'.join([name, 'IS_'+val])
				dataset[col_name] = dummies[col]
		else:
			dataset[col] = dummies[col]
	return dataset if keep_orig else dataset.drop(column,1)
