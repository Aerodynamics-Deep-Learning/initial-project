Here, we conduct our exploratory analyses.

1- MLP & Pairwise NIF, Size Analysis

    - MLPs with sizes reflecting that of NeuralFoil:
        - xxsmall
        - xsmall
        - small
        - medium
        - large
        - xlarge
        - xxlarge
        - xxxlarge
        - sub_enormous
        - enormous

    - Pairwise NIF models with the following sizes (paramnet_shapenet):
        - xxsmall_xsmall
        - xsmall_small
        - small_medium
        - medium_large
        - large_xlarge
        - xlarge_xxlarge
        - xxlarge_xxxlarge
        - xxxlarge_sub_enormous
        - sub_enormous_enormous

    - these sizes are:
        - xxsmall = 2 layers, 32 neurons
        - xsmall =  3 layers, 32 neurons
        - small = 3 layers, 48 neurons 
        - medium = 4 layers, 64 neurons
        - large = 4 layers, 128 neurons 
        - xlarge = 4 layers, 256 neurons
        - xxlarge = 5 layers, 256 neurons
        - xxxlarge = 5 layers, 512 neurons     
        - sub_enormous = 5 layers, 1024 neurons
        - enormous = 5 layers, 2048 neurons

    - with the following hyperparams/datasets:

        - AdamW optimizer:
            - constant lr = 1e-4
            - constant wd = 1e-2

        - airfoil_dataset_8bern.csv
            - 9 bernstein coefficients * 2 + trailing edge * 2
            - set mach number = 0.1
            - reynolds: [3m, 6m, 9m]
            - aoa: [-4, -3, -2, -1, 0 , 1, ..., 20]

