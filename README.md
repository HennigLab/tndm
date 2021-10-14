# Targeted Neural Dynamical Modeling

[![Python Tests](https://github.com/HennigLab/tndm/actions/workflows/python-tests.yml/badge.svg)](https://github.com/uoe-neuro/tndm/actions/workflows/python-tests.yml) [![codecov](https://codecov.io/gh/HennigLab/tndm/branch/main/graph/badge.svg?token=EDXVU3YSEL)](https://codecov.io/gh/HennigLab/tndm) ![TensorFlow Requirement: 2.x](https://img.shields.io/badge/TensorFlow%20Requirement-2.x-brightgreen)


Latent dynamics models have emerged as powerful tools for modeling and interpreting neural population activity. Recently, there has been a focus on incorporating simultaneously measured behaviour into these models to further disentangle sources of neural variability in their latent space. These approaches, however, are limited in their ability to capture the underlying neural dynamics (e.g. linear) and in their ability to relate the learned dynamics back to the observed behaviour (e.g. no time lag). To this end, we introduce Targeted Neural Dynamical Modeling (TNDM), a nonlinear state-space model that jointly models the neural activity and external behavioural variables. TNDM decomposes neural dynamics into behaviourally relevant and behaviourally irrelevant dynamics; the relevant dynamics are used to reconstruct the behaviour through a flexible linear decoder and both sets of dynamics are used to reconstruct the neural activity through a linear decoder with no time lag. We implement TNDM as a sequential variational autoencoder and validate it on recordings taken from the premotor and motor cortex of a monkey performing a center-out reaching task. We show that TNDM is able to learn low-dimensional latent dynamics that are highly predictive of behaviour without sacrificing its fit to the neural data.

# Installing the package

In a virtual environment, install all the dependencies and the package using the following commands:
```
pip install -e .
```

# Getting started

```
python tndm -r <your-settings>.yaml
```
