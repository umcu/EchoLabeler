# EchoLabeler

Set of classes to perform label extraction training/inference based on echocardiogram reports.

**Span labelling**
* Approximate list lookup using regular expressions
* MedCAT + biLSTM
* Spacy SpanCategorizer

The identification can be followed up by an aggregation over the whole document
to perform document classification.

**Document labelling**
* BOW, using TF-IDF, latent Dirichlet allocation, and topic modelling enrichment, followed by a standard gradient-boosted classifier.
* SetFit using RobBERT
* MedRoBERTa.nl
* biGRU
* CNN

# fixes:
the current toml will not directly work.
* in gensim/matutils.py -> change ```from scipy.linalg import triu``` to ```from numpy import triu```
* 