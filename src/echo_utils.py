# add autoreload
import numpy as np
from typing import Callable, List, Union, Literal
import spacy
from gensim.models import Word2Vec as W2V, keyedvectors as KV, Doc2Vec, FastText
import dotenv
import os
import re

ENV = dotenv.load_env('../.env')
EMB_PATH  = os.environ['word_embeddings']

def report_filter(texts: List[str],
                  Tokenizer: Callable = None,
                  min_tokens: int = 4,
                  min_chars: int = 15,
                  flag_terms: List[str] = None,
                  save_terms: List[str] = None,
                  suspect_terms: List[str] = None):
    '''
    :param texts: texts to filter, List of strings
    :param Tokenizer: Tokenizer to be used for filtering, use standard tokenizer if None
           The tokenizer is expected to ingest a string, and output a list of strings
    :param min_tokens: texts with < min_tokens are filtered out
    :param min_chars: texts with < min_char are filtered out
    :param flag_terms: texts with any of these spans are filtered out
    :param save_terms: texts with these terms should remain
    :param suspect_terms: texts with any of these spans are flagged as suspect for removal
    :return:
    '''
    if Tokenizer is None:
        Tokenizer = lambda x: x.split()

    filtered_texts = []
    suspect_texts = []
    for text in texts:
        Tokens = Tokenizer(text)

        numtokens = len(Tokens)
        numchars = len(text)

        if (numtokens <= min_tokens) or (numchars <= min_chars):
            if any([st in text.lower() for st in save_terms]):
                filtered_texts.append(text)
            else:
                filtered_texts.append(None)
            continue

        if any([ft in text.lower() for ft in flag_terms]):
            filtered_texts.append(None)
            continue

        if (isinstance(suspect_terms, list)) and (any([st in text for st in suspect_terms])):
            filtered_texts.append(text)
            suspect_texts.append(text)
            continue

        filtered_texts.append(text)
    return filtered_texts, suspect_texts

def text_to_vectors(texts: List[str]=None,
                    maxlen: int=200, 
                    source: Literal['spacy', 'cardio']='spacy',
                    how: Literal['fasttext', 'sbert']='fasttext',
                    padmethod: Literal['zero', 'random', 'mean']='mean', 
                    window: int=3):
    '''
    # TODO: add option for other embeddings
    # TODO: for padding don't use zero vector but random or mean vector
    :param texts: texts to embed, list of strings
    :param maxlen: max length, in tokens, of the texts
    :return: array of arrays
    '''
    if source == 'spacy':
        nlp = spacy.load("nl_core_news_lg", disable=['parser', 'ner'])
        _docs = nlp.pipe(texts)
        # We truncate or pad the document vector to a fixed size
        array_list = []
        for doc in _docs:
            vectors = [token.vector for token in doc]
            mwv = np.mean(vectors, axis=0)
            
            if len(vectors) > maxlen:
                vectors = vectors[:maxlen]
            elif len(vectors) < max_len:
                if padmethod == 'zero':
                    vectors += [np.zeros((nlp.vocab.vectors_length,))] * (maxlen - len(vectors))
                elif padmethod == 'random':
                    vectors += [np.random.rand(nlp.vocab.vectors_length)] * (maxlen - len(vectors))
                elif padmethod == 'mean':
                    vectors += [mwv] * (maxlen - len(vectors))
                
            array_list.append(np.array(vectors))
    elif source == 'cardio':
        if how == 'fasttext':
            wvs = FastText.load(os.path.join(EMB_PATH, 'cardio_sg.model'))
            vocab_size = len(wvs.wv.vectors_vocab.shape[0])
        # go through documents
        array_list = []
        splitter = re.compile(r'[^\w]')
        for txt in texts:
            words = splitter.split(txt)    
            vectors = []
            for k in range(maxlen)
                vectors.append(wvs.wv[words[k]])
                
            mwv = np.mean(vectors, axis=0)            
            
            if len(vectors) < max_len:
                if padmethod == 'zero':
                    vectors += [np.zeros((vocab_size,))] * (maxlen - len(vectors))
                elif padmethod == 'random':
                    vectors += [np.random.rand(vocab_size)] * (maxlen - len(vectors))
                elif padmethod == 'mean':
                    vectors += [mwv] * (maxlen - len(vectors)) 
            
            array_list.append(np.array(vectors))    
            # if word not in vocab, take closest one according to Levenhstein or mean of surrounding tokens

    return np.array(array_list)