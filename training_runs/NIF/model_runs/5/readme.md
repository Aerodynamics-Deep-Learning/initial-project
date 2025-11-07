Author: @Ege

In this folder, I have made major tests. These are:

1: Simple NIF training
    - This training is on the 137250 datapoint dataset, we had insane model collapse, for more detail check the notebook.
    - It used a Pairwise NIF, had some weight decay, and used a MSE loss, it had a relatively large size.
2: MLP training
    - This training is on the same relatively large dataset, we also had insane model collapse with this one.
    - It used a simple MLP, that is relatively deep.
3: Simple test
    - In this training, I tried to see if the model was working at all, turns out it is
4: Retrain NIF
    - In this one, I have realized a MAJOR flaw in our data inputs
    - Reynolds column is on the order of millions... this means, it dominates over all other inputs, and was the primary reason of BS results
    - This was on the same Pair NIF model as the 1st training
    - It had simple MSE
    - It had some weight decay

Next steps:
    - We can finally try testing different models
    - Maybe some drop-out
    - The first one I really want to try out is the Full Paper NIF