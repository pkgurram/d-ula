# Decentralized Langevin Dynamics for Bayesian Learning

This repository is the official implementation of [Decentralized Langevin Dynamics for Bayesian Learning](https://arxiv.org/abs/2007.06799).

## Requirements

python 3.x, numpy, matplotlib

pytorch (>=1.0), torchvision

scikit-learn

## Scripts for experiments

### Toy Data - GMM

Centralized ULA - toy_data_sgld.py

Decentralized ULA - toy_data_dist_sgld.py

### Logistic Regression

Centralized ULA - a9a_sgld.py

Decentralized ULA - a9a_dist_sgld.py

### Image Classification

SGD - mnist_sgd_svhn_pred_scores.py

Centralized ULA - mnist_sgld_svhn_pred_scores.py

Decentralized ULA - mnist_dist_sgld_svhn_pred_scores.py
