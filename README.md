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
* SetFit using RobBERTv2
* MedRoBERTa.nl
* biGRU
* CNN

# fixes:
the current toml will not directly work.
* in gensim/matutils.py -> change ```from scipy.linalg import triu``` to ```from numpy import triu```


# Reference

Arxiv:
```
@misc{arends2024diagnosisextractionunstructureddutch,
      title={Diagnosis extraction from unstructured Dutch echocardiogram reports using span- and document-level characteristic classification}, 
      author={Bauke Arends and Melle Vessies and Dirk van Osch and Arco Teske and Pim van der Harst and René van Es and Bram van Es},
      year={2024},
      eprint={2408.06930},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.06930}, 
}
```

[The models are on Huggingface](https://huggingface.co/collections/UMCUtrecht/dutch-echocardiogram-information-extraction-66bb501be5904d821f6ea66a)
