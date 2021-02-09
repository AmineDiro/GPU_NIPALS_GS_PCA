## NIPALS-PCA Algorithm

- The purpose of considering this algorithm here is three-fold:
    - it gives additional insight into what the loadings and scores mean;
    - it shows how each component is independent of (orthogonal to) the other components,
    - it shows how the algorithm can handle missing data.
- The algorithm extracts each component sequentially, starting with the first component, direction of greatest variance, and then the second component, and so on.

[Checkout this link](https://learnche.org/pid/latent-variable-modelling/principal-component-analysis/algorithms-to-calculate-build-pca-models#lvm-eigenvalue-decomposition)  and [this link](https://cran.r-project.org/web/packages/nipals/vignettes/nipals_algorithm.html) for a more detailed view of the algorithm. 

The purpose is to implement the paper :[link](https://arxiv.org/abs/0811.1081) on GPU using pyCuda. 

## Install requirements

## Todo Amine 
- Implement CPU NIPALS-PCA => OK
- Write Unitary tests vs sklearn  PCA  => OK
- Implement GPU NIPALS-PCA
    - Implemented onstep compt => BUG MATRIX MULT 
    - Implement for loop
- Test CPU vs GPU nipals


# Report 
- PCA pseudo code : nipals gram schmidt 
- Pycuda + kernels 
- Challenges : 
    - Cuda 
    - version normal
    - matrice carré
    - version optimisé : shared memory , tiles.. 
- Speedup test (non optimal) and vs numpy
- Conclusion : 
    - what do ? coallesce , shared memory , 