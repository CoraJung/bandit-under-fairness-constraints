#!/bin/bash
python -m experiments.bandit.recidivism recidivism_gf --n_trials 1 --definition GroupFairness --e 0.1 --d 0.05 --ci_type ttest --n_iters 10000 --n_jobs 4  --r_train_v_test 0.4 --r_cand_v_safe 0.4 --rwd_recid -1.0 --rwd_nonrecid 1.0 --use_score_text --data_pct 0.01 0.012742749857 0.0206913808111 0.0335981828628 0.0545559478117 0.088586679041 0.143844988829 0.233572146909 0.379269019073 0.615848211066 1.0

# python -m experiments.bandit.credit_assignment credt_assignment_di --n_trials 50 --definition DisparateImpact --e -0.8 --d 0.05 --ci_type ttest --n_iters 2000 --n_jobs 4 --r_train_v_test 0.4 --r_cand_v_safe 0.4 --data_pct 0.233572146909 0.379269019073 0.615848211066 1.0
# python -m experiments.bandit.credit_assignment credt_assignment_gf --n_trials 50 --definition GroupFairness   --e 0.23 --d 0.05 --ci_type ttest --n_iters 2000 --n_jobs 4 --r_train_v_test 0.4 --r_cand_v_safe 0.4 --data_pct 0.233572146909 0.379269019073 0.615848211066 1.0

# python -m experiments.bandit.credit_assignment_online credit_assignment_online_di --n_trials 50 --definition DisparateImpact --e -0.8 --d 0.05 --ci_type ttest --n_iters 2000 --n_jobs 4 --r_train_v_test 0.4 --r_cand_v_safe 0.4 --data_pct 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0

# python -m experiments.bandit.tutoring tutoring      --n_trials 50 --e_m 0.5 --e_f 0.0 --ci_type bootstrap --n_iters 2000 --n_jobs 4 --r_train_v_test 0.4 --r_cand_v_safe 0.4 --remove_biased_tutorial          --data_pct 0.01 0.012742749857 0.0206913808111 0.0335981828628 0.0545559478117 0.088586679041 0.143844988829 0.233572146909 0.379269019073 0.615848211066 1.0
# python -m experiments.bandit.tutoring tutoring_skew --n_trials 50 --e_m 2.5 --e_f 0.0 --ci_type bootstrap --n_iters 2000 --n_jobs 4 --r_train_v_test 0.4 --r_cand_v_safe 0.4 --simulated_female_proportion 0.8 --data_pct 0.01 0.012742749857 0.0206913808111 0.0335981828628 0.0545559478117 0.088586679041 0.143844988829 0.233572146909 0.379269019073 0.615848211066 1.0