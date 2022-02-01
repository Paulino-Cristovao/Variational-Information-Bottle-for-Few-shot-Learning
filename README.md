# Variational-Information-Bottle-for-Few-shot-Learning

This work introduces the variational Information Bottleneck (VIB) loss function for imprinted models.
We find that this objective function gives us relevant information for generalizing on novel classes.
We intend to show that cross entropy learning objective alone is not enough for Few-shot learning.

We trained the imprinted model with VIB. 
The dataset used is CUB-200-2011. We splitted the base classes (train) [0-99] claases and novel [100-199] classes.

Download the dataset and store in folder "Data".
You can tune some of the parameter in pre-train.py and run.sh file.
Training the modelf for 50 epochs may last around 1h in GPU. We recommend training on GPU.
We give credit to https://github.com/YU1ut.
Parts of the code is taking from his repository.
