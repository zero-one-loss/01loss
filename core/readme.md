01 loss Stochastic Coordinate Descent
=

Paper
-

Contents
-
-   `scd_v9_gpu.py` defines 01 SCD classes.
    -   `--nrows` (IMPORTANT) The ratio of training data in each iteration. 
                    `0 ~ 1.0`
    -   `--nfeatures` The ratio of features used in each iteration `0 ~ 1.0`
    -   `--w_inc` The initial step size of w. Default is `0.17, 0.1 ~ 0.2`
    -   `--tol` The stop threshold. Default is `1e-4`
    -   `--local_iter` The maximum number of updating in each iteration. Default
                        is `100`
    -   `--num_iters` (IMPORTANT) The total number of iterations. Default 
                        is `5000`
    -   `--round`   The number of random restart. Used to do majority vote.
    -   `--interval` K neighbours considered in best `bias` searching. Default 
                        is `10`.
    -   `--adaptive_inc` (USELESS). Adaptive step size. Please set False now.
    -   `--updated_features` (IMPORTANT) The number of coordinates will be 
                            considered to update in each inner loop. 
                            The algorithm will pick the best
                            coordinate to update by step size. Default is `128`
    -   `--n_jobs` The number of processes in CPU. Default is `10`
    -   `--num_gpus` The number of GPUs. Default is the number of GPUs available
                        in the node. 

-   `scd_v11_gpu.py` Adversarial training.
-   `scd_v12_gpu.py` Adversarial training.
-   `scd_v15_gpu.py` MLP01 loss.
