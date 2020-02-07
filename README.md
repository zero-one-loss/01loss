Robust binary classification with the 01 loss
=

Contents
-
1. SCD 01 loss algorithm implementations are in `core` directory.
2. Training scripts for each data set are in their directory named by their
dataset's name. Parameters setting are saved in shell scripts in job.  
    - job1.sh  training scd01 loss
    - job2.sh  training svm and mlp
    - job3.sh  black-box attack for scd model
    - job4.sh  black-box attack for svm and mlp
3. black box attack results will be saved in `results`.
4. Create `model` and `temp_data` under each 'dataset' folder
