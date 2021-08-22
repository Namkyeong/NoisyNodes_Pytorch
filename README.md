# NoisyNodes_Pytorch
Implementation of Very Deep Graph Neural Networks Via Noise Regularisation

A PyTorch implementation of "<a href="https://arxiv.org/abs/2106.07971">Very Deep Graph Neural Networks Via Noise Regularisation</a>" paper, worked as base model of KDD cup 2021 3rd place team Quantum (DeepMind).


## Model Description
While Deep GNNs should have greater expressivity and ability to capture complex functions, it has been proposed that in practice **Oversmoothing** and **bottleneck effects** limit the potential of deep GNNs. With noise regularisation, this paper overcomes the limitation. Specifically, noise regularisation approach corrupts the input graph's nodes with noise, and then adds an autoencoding loss if a node prediction task is not defined. The node-level task is the key framework for preventing oversmoothing problem.  


In my implementation, I corrupted atom position with Gaussian noise for every epoch and tried to predict original coordinate of the atom for node-level task.

<img src="img/image.png" width="700px"></img>


## Hyperparameters for training NoisyNodes
Following Options can be passed to `exec.py`  


`--M:`
Number of Message passing steps for each block. Default is 10.  
usage example :`--M 10`  


`--N:`
Number of Block iteration in Process step. Default is 2.  
usage example :`--N 2`  


`--noise_std:`
Standard deviation for Gaussian Noise corrupting the atomic position (coordinate). Default is 0.02.  
usage example :`--noise_std 0.02`



## Notice
Detailed implementation has not been fully explored since there is no available official code yet. Please feel free to come up with any issues or any implementation details. Thanks! 