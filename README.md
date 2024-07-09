# EchoLabeler

Set of classes to perform label extraction training/inference based on echocardiograms.

**Span labeling**
* Direct concept lookup
  * Embeddings in lookup table
  * NN-search
* MedCAT + biLSTM
* MedCAT + Transformer
* Spacy SpanCat

The identification can be followed up by an aggregation over the whole document
to perform document classification.

**Document labeling**
* TFIDF - CBOW followed by a simple classifier such as Logistic Regression or SVM.
* SetFit
* MedRoBERTa.nl


