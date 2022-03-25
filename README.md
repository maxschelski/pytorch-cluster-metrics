# pytorch-cluster-metrics
Pytorch implementation of standard metrics for clustering.
Test on PyTorch = 1.10.1 / cuda11.3_cudnn8_0

So far only the Silhouette score is implemented.
Code for the Silhouette score was developed in NumPy by Alexandre Abraham:
https://gist.github.com/AlexandreAbraham/5544803

# Installation

1. Open a terminal, navigate to the folder where you want to put the repository and clone it:
> git clone https://github.com/maxschelski/pytorch-cluster-metrics.git
2. Navigate into the folder of the repository (pytorch-cluster-metrics):
> cd pytorch-cluster-metrics
3. Install torchclustermetrics locally using pip:
> pip install -e .

For any questions feel free to contact me via E-Mail to max.schelski@googlemail.com.

# Usage

> from torchclustermetrics import silhouette
> 
> score = silhouette.score(X, labels)
