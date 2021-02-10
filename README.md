# Parallel GPU Implementation of IterativePCA Algorithms
Principal component analysis (PCA) is one of the most valuable results from applied linear algebra, and probably the most popular method used for compacting higher dimensional data sets into lower dimensional ones for data analysis, visu-alization, feature extraction, or data compression
- The purpose of considering this algorithm here is three-fold:
    - it gives additional insight into what the loadings and scores mean;
    - it shows how each component is independent of (orthogonal to) the other components,
    - it shows how the algorithm can handle missing data.
- The algorithm extracts each component sequentially, starting with the first component, direction of greatest variance, and then the second component, and so on.

[Checkout this link](https://learnche.org/pid/latent-variable-modelling/principal-component-analysis/algorithms-to-calculate-build-pca-models#lvm-eigenvalue-decomposition)  and [this link](https://cran.r-project.org/web/packages/nipals/vignettes/nipals_algorithm.html) for a more detailed view of the algorithm. 

The purpose is to implement the paper :[link](https://arxiv.org/abs/0811.1081) on GPU using pyCuda. 

## Install requirements

You'll need Numpy, Sklearn for running and testing the CPU verion.

For running the GPU version, you'll need a Linux or Windows 10 PC with a modern NVIDIA GPU (>=2016) is required, with all necessary GPU drivers and the CUDA Toolkit (10.0 onward) installed. 
A suitable Python (>=3) with the PyCUDA module is also required. 

## Testings

For testing the kernel functions in `tests/`, run the following:
``` 
pip install -e .
python -m unittest -v tests/test_nipals_gpu.py
 ```


The  kernel functions in `test_kernels/` are an optimized version of the operations in  `nipals/kernels.py` but are not stable yet.