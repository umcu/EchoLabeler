import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

import os
import sys
from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MultiLabelBinarizer
import benedict

from keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from keras.models import Sequential
from keras import layers, models, utils as keras_utils
from keras import optimizers
from tensorflow.keras.models import Model
from keras.utils import plot_model
import tensorflow as tf
import spacy
import re 

from sklearn.preprocessing import LabelEncoder
import gc

from typing import Callable, Tuple, Dict, List, Literal, Union

def CurrentModel(modelselection: Literal['bigru', 'bilstm', 'cnn', 'hstacked_dcnn', 'textcnn'], 
                 embeddingdim: int=128,
                 pre_embeddingdim: int=None,
                 maxlen: int=256, 
                 vocabsize: int=50_000,
                 numclasses: int=5,
                 learningrate: float=0.001,
                 num_layers: int=128,
                 pre_trained_vectors: bool=False,
                 multilabel: bool=False,
                 dilation: int=4, 
                 dropout: float=0.,
                 dilations: List[int]=[1,2,4,8],
                 filters: List[int]=[1,2,3,5],
                 attention: bool=False
                 ):
    # TODO: test Attention
    # TODO: we can added concatenations of various kernel sizesm see e.g. https://github.com/ShawnyXiao/TextClassification-Keras/blob/master/model/
    

    if pre_embeddingdim is None:
        pre_embeddingdim = embeddingdim
        
    accuracy = 'binary_accuracy'
    if multilabel:
        finalact = 'sigmoid'
        lossfunction = 'binary_crossentropy'
    else:
        finalact = 'softmax'
        lossfunction = 'categorical_crossentropy'
        
    if pre_trained_vectors:
        num_dense = 20
    else:
        num_dense = 10
            
    ##################################################################################################
    if modelselection =='rcnn':
        inputs = layers.Input(shape=(maxlen,), name='InputData', dtype='int32')
        if pre_trained_vectors:
            pre_trained_inputs = layers.Input(shape=(maxlen, pre_embeddingdim), name='PreTrainedInputData')
            emb_non_trainable = pre_trained_inputs
            emb_trainable = layers.Embedding(input_dim=vocabsize, output_dim=embeddingdim,
                                             input_length=maxlen, name='EmbeddingLayerDynamic', trainable=True)(inputs)
            embs = layers.concatenate([emb_non_trainable, emb_trainable], axis=-1)
        else:
            embs = layers.Embedding(vocabsize, embeddingdim, input_length=maxlen, name='EmbeddingLayer')(inputs)
            
        bigru_layer = layers.Bidirectional(layers.GRU(num_layers,  name='biGRU', return_sequences=True))(embs)
        
        conv_layer = layers.Conv1D(num_layers, 5, activation = 'relu', dilation_rate = dilation, name='Conv')(bigru_layer)
        # convs = []
        # for k in [1,2,3,4,5]:
        #    convs.append(layers.Conv1D(num_layers, k, activation = 'relu', dilation_rate = dilation, name='Conv'))(bigru_layer)
        # pools = [layers.GlobalAveragePooling1D(x) for x in convs] + [layers.GlobalMaxPooling1D(x) for x in convs]
        # concat = layers.concatenate(pools)
        
        pool_layer = layers.GlobalMaxPooling1D(name='GlobalMaxPooling')(conv_layer)

        if attention:
            pool_layer = layers.Attention(use_scale=True, name='attention')(pool_layer)
            
        dropout_layer = layers.Dropout(dropout,  name='dropout')(pool_layer)
        dense1_layer = layers.Dense(num_dense, activation = 'relu', name='Dense1')(dropout_layer)
        output_layer = layers.Dense(numclasses, activation = finalact, name='Dense2')(dense1_layer)
        if pre_trained_vectors:
            model = Model(inputs=[inputs, pre_trained_inputs], outputs=output_layer)
        else:
            model = Model(inputs=inputs, outputs=output_layer)
        model.compile(optimizer = optimizers.Adam(learning_rate=learningrate),
                      loss = lossfunction,
                      metrics = [accuracy])
        return model
    elif modelselection =='rcnn_vstacked':
        inputs = layers.Input(shape=(maxlen,), name='InputData', dtype='int32')
        if pre_trained_vectors:
            pre_trained_inputs = layers.Input(shape=(maxlen, pre_embeddingdim), name='PreTrainedInputData')
            emb_non_trainable = pre_trained_inputs
            emb_trainable = layers.Embedding(input_dim=vocabsize, output_dim=embeddingdim,
                                             input_length=maxlen, name='EmbeddingLayerDynamic', trainable=True)(inputs)
            embs = layers.concatenate([emb_non_trainable, emb_trainable], axis=-1)
        else:
            embs = layers.Embedding(vocabsize, embeddingdim, input_length=maxlen, name='EmbeddingLayer')(inputs)
        
        bigru_layer = layers.Bidirectional(layers.GRU(num_layers,  name='biGRU', return_sequences=pre_trained_vectors))(embs)
        conv_layer = layers.Conv1D(num_layers, 5, activation = 'relu', dilation_rate = dilation, name='Conv')(embs)
        pool_layer_bigru = layers.GlobalMaxPooling1D(name='GlobalMaxPoolingBiGRU')(bigru_layer)
        pool_layer_conv= layers.GlobalMaxPooling1D(name='GlobalMaxPoolingConv')(conv_layer)

        concat_layer = layers.concatenate([pool_layer_bigru, pool_layer_conv])

        if attention:
            concat_layer = layers.Attention(use_scale=True, name='attention')(concat_layer)
            
        dropout_layer = layers.Dropout(dropout,  name='dropout')(concat_layer)
        dense1_layer = layers.Dense(num_dense, activation = 'relu', name='Dense1')(dropout_layer)
        output_layer = layers.Dense(numclasses, activation = finalact, name='Dense2')(dense1_layer)
        if pre_trained_vectors:
            model = Model(inputs=[inputs, pre_trained_inputs], outputs=output_layer)
        else:
            model = Model(inputs=inputs, outputs=output_layer)
        model.compile(optimizer = optimizers.Adam(learning_rate=learningrate),
                      loss = lossfunction,
                      metrics = [accuracy])
        return model
    ##################################################################################################    
    elif modelselection in ['bilstm', 'bigru']:
        # PREP-LAYERS
        inputs = layers.Input(shape=(maxlen,), name='InputData', dtype='int32')
        if pre_trained_vectors:
            pre_trained_inputs = layers.Input(shape=(maxlen, pre_embeddingdim), name='PreTrainedInputData')
            emb_non_trainable = pre_trained_inputs
            emb_trainable = layers.Embedding(input_dim=vocabsize, output_dim=embeddingdim,
                                             input_length=maxlen, name='EmbeddingLayerDynamic', trainable=True)(inputs)
            embs = layers.concatenate([emb_non_trainable, emb_trainable], axis=-1)
        else:
            embs = layers.Embedding(input_dim=vocabsize, output_dim=embeddingdim, 
                                    input_length=maxlen, name='EmbeddingLayer', trainable=True)(inputs)
        # FEATURE EXTRACTION LAYERS
        if modelselection=='bigru':
            #bilstm_layer = layers.Bidirectional(layers.CuDNNGRU(num_layers, return_sequences=True))(embs)
            bilstm_layer = layers.Bidirectional(layers.GRU(num_layers, return_sequences=True))(embs)
        elif modelselection=='bilstm':
            bilstm_layer = layers.Bidirectional(layers.LSTM(num_layers, return_sequences=True))(embs)

        # COLLECTION LAYERS
        pool_layer = layers.GlobalMaxPooling1D(name='GlobalMaxPooling')(bilstm_layer)

        if attention:
            pool_layer = layers.Attention(use_scale=True, name='attention')(pool_layer)
            
        dropout_layer = layers.Dropout(dropout,  name='dropout')(pool_layer)
        dense1_layer = layers.Dense(num_dense, activation = 'relu', name='Dense1')(dropout_layer)
        output_layer = layers.Dense(numclasses, activation = finalact, name='Dense2')(dense1_layer)
        if pre_trained_vectors:
            model = Model(inputs=[inputs, pre_trained_inputs], outputs=output_layer)
        else:
            model = Model(inputs=inputs, outputs=output_layer)
        model.compile(optimizer = optimizers.Adam(learning_rate=learningrate),
                      loss = lossfunction,
                      metrics = [accuracy])
        return model
    ##################################################################################################        
    elif modelselection =='cnn':
        # basic cnn with one dilation
        # PREP-LAYERS
        inputs = layers.Input(shape=(maxlen,), name='InputData', dtype='int32')
        if pre_trained_vectors:
            pre_trained_inputs = layers.Input(shape=(maxlen, pre_embeddingdim), name='PreTrainedInputData')
            emb_non_trainable = pre_trained_inputs
            emb_trainable = layers.Embedding(input_dim=vocabsize, output_dim=embeddingdim,
                                             input_length=maxlen, name='EmbeddingLayerDynamic', trainable=True)(inputs)
            embs = layers.concatenate([emb_non_trainable, emb_trainable], axis=-1)
        else:
            embs = layers.Embedding(vocabsize, embeddingdim, input_length = maxlen, name='EmbeddingLayer', trainable=True)(inputs)

        # FEATURE EXTRACTION LAYERS
        conv_layer = layers.Conv1D(num_layers, 5, activation='relu', dilation_rate=dilation, name='conv')(embs)

        # COLLECTION LAYERS
        pool_layer = layers.GlobalMaxPooling1D(name='GlobalMaxPooling')(conv_layer)

        if attention:
            pool_layer = layers.Attention(use_scale=True, name='attention')(pool_layer)
            
        dropout_layer = layers.Dropout(dropout,  name='dropout')(pool_layer)
        dense1_layer = layers.Dense(num_dense, activation = 'relu', name='Dense1')(dropout_layer)
        output_layer = layers.Dense(numclasses, activation = finalact, name='Dense2')(dense1_layer)
        if pre_trained_vectors:
            model = Model(inputs=[inputs, pre_trained_inputs], outputs=output_layer)
        else:
            model = Model(inputs=inputs, outputs=output_layer)
        model.compile(optimizer = optimizers.Adam(learning_rate=learningrate),
                      loss = lossfunction,
                      metrics = [accuracy])
        return model
    ##################################################################################################    
    ##################################################################################################    
    elif modelselection == 'vstacked_dcnn':
        # CNN with multiple dilations; 1,2,4,8 stacked, before going into the final layer
        # basic cnn with one dilation
        # TODO: variant with different kernel sizes.
        inputs = layers.Input(shape=(maxlen,), name='InputData', dtype='int32')
        if pre_trained_vectors:
            pre_trained_inputs = layers.Input(shape=(maxlen, pre_embeddingdim),
                                              name='PreTrainedInputData')
            emb_non_trainable = pre_trained_inputs
            emb_trainable = layers.Embedding(input_dim=vocabsize, output_dim=embeddingdim,
                                             input_length=maxlen, name='EmbeddingLayerDynamic', 
                                             trainable=True)(inputs)
            embs = layers.concatenate([emb_non_trainable, emb_trainable], axis=-1)    
        else:
            embs = layers.Embedding(vocabsize, embeddingdim, input_length = maxlen, name='EmbeddingLayer', trainable=True)(inputs)
        
        
        Flows = []
        for k in [1,2,4,8,16]:
            DilLayer = layers.Conv1D(num_layers, 5, activation = 'relu', dilation_rate = k, name=f'ConvDilation{k}')(embs)
            DilLayer = layers.GlobalMaxPooling1D(name=f'GlobalMaxPooling{k}')(DilLayer)
            Flows.append(DilLayer)
        concat = layers.concatenate(Flows, axis=-1)
        
        if attention:
            concat = layers.Attention(use_scale=True, name='attention')(concat)
            
        dropout_layer = layers.Dropout(dropout,  name='dropout')(concat)
        dense1_layer = layers.Dense(num_dense, activation = 'relu', name='Dense1')(dropout_layer)
        output_layer = layers.Dense(numclasses, activation = finalact, name='Dense2')(dense1_layer)
        model = Model(inputs=inputs, outputs=output_layer)
        model.compile(optimizer = optimizers.Adam(learning_rate=learningrate),
                      loss = lossfunction,
                      metrics = [accuracy])
        return model        
             
    ##################################################################################################    
    elif modelselection == 'textcnn':
        # also see: https://github.com/ShaneTian/TextCNN/blob/master/text_cnn.py and https://arxiv.org/abs/1408.5882
        inputs = layers.Input(shape = (maxlen,),  name='InputData',  dtype='int32')
        if pre_trained_vectors:
            pre_trained_inputs = layers.Input(shape=(maxlen, pre_embeddingdim),
                                              name='PreTrainedInputData')
            emb_non_trainable = pre_trained_inputs
            emb_trainable = layers.Embedding(input_dim=vocabsize, output_dim=embeddingdim,
                                             input_length=maxlen, name='EmbeddingLayerDynamic', 
                                             trainable=True)(inputs)
            xlayer = layers.concatenate([emb_non_trainable, emb_trainable], axis=-1)    
        else:
            xlayer = layers.Embedding(vocabsize, embeddingdim, input_length = maxlen, name='EmbeddingLayer', trainable=True)(inputs)
        
        xlayer = layers.Reshape((maxlen, embeddingdim, 1))(xlayer)
        
        pool_outputs = []
        for filter_size in filters:
            filter_shape = (filter_size, embeddingdim)
            conv  = layers.Conv2D(num_layers, 
                                 kernel_initializer='he_normal',
                                 kernel_size=filter_shape, 
                                 activation = 'relu',
                                 name=f'conv_{filter_size}')(xlayer)
            pool  = layers.MaxPool2D(pool_size=(maxlen-filter_size+1, 1),
                                     strides=(1, 1), padding='valid',
                                     data_format='channels_last',
                                     name='max_pooling_{:d}'.format(filter_size))(conv)
            pool_outputs.append(pool)
            
        x = layers.concatenate(pool_outputs, axis=-1, name='concatenate')
        x = layers.Flatten(data_format='channels_last', name='flatten')(x)
        
        if attention:
            x = layers.Attention(use_scale=True, name='attention')(x)
            
        x = layers.Dropout(dropout,  name='dropout')(x)
        x = layers.Dense(num_dense, activation = 'relu', name='Dense1')(x)
        outputs = layers.Dense(numclasses, activation = finalact, name='Dense2')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer = optimizers.Adam(learning_rate=learningrate),
                      loss = lossfunction,
                      metrics = [accuracy])
        return model
      
    return False
