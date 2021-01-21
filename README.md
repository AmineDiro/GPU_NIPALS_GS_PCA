## NIPALS-PCA Algorithm

- The purpose of considering this algorithm here is three-fold:
    - it gives additional insight into what the loadings and scores mean;
    - it shows how each component is independent of (orthogonal to) the other components,
    - it shows how the algorithm can handle missing data.
- The algorithm extracts each component sequentially, starting with the first component, direction of greatest variance, and then the second component, and so on.

[Checkout this link](https://learnche.org/pid/latent-variable-modelling/principal-component-analysis/algorithms-to-calculate-build-pca-models#lvm-eigenvalue-decomposition)  for a more detailed view of the algorithm. 

The purpose is to implement the paper :[link](https://arxiv.org/abs/0811.1081) on GPU 

## Install requirements

## Todo list 
- Implement CPu GS-PCA
- Writee Unitary tests
- Implement GPU NIPALS-PCA
- Implement GPU GS-PCA