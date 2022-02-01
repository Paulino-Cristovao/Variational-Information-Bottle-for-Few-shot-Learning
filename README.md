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



## Preparation
Download [CUB_200_2011 Dataset](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz).

Unzip and locate it in this directory.

The whole directory should be look like this:
```
imprinted-weights
│   README.md
│   pretrain.py
│   models.py
│   loader.py
│   imprint.py
│   imprint_ft.py
│   alljoint.py
|
└───utils
│   
└───CUB_200_2011
    │   images.txt
    │   image_class_labels.txt
    │   train_test_split.txt
    │
    └───images
        │   001.Black_footed_Albatross
        │   002.Laysan_Albatross
        │   ...
```



## References
- [1]: H. Qi, M. Brown and D. Lowe. "Low-Shot Learning with Imprinted Weights", in CVPR, 2018.
- [2]: A. A. Alemi et al. "Deep Variational Information Bottleneck" ICLR, 2017
