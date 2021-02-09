import os
from setuptools import setup


setup(name='pca_gpu',
      version='0.0.1',
      description='Implementation of Nipals and GC PCA on GPU',
      author='Amine',
      license='MIT',
      packages=['nipals'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License"
      ],
      install_requires=['numpy','sklearn','pycuda'],
      python_requires='>=3',
      extras_require={
          'gpu': ["pycuda"],
          'testing': [
              "sklearn",
              "tqdm",
          ],
      },
      include_package_data = True)
