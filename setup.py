import os
from setuptools import setup


setup(name='pca_gpu',
      version='0.0.1',
      description='Implementation of Nipals and GC PCA on GPU',
      author='Amine Wissal',
      license='MIT',
      packages=['nipals'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License"
      ],
      install_requires=['numpy'],
      python_requires='>=3.7',
    #   extras_require={
    #       'gpu': ["pyopencl", "six"],
    #       'testing': [
    #           "pytest",
    #           "torch",
    #           "tqdm",
    #       ],
    #   },
      include_package_data = True)
